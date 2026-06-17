"""Qwen2.5-VL single-agent reasoning (baseline + LoRA path)."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from src import metrics
from src.config import setup_env

# Keyed by (model_id, lora_path) so different LoRA adapters are cached separately.
_VL_MODEL_CACHE: dict[tuple[str, str | None], tuple[Any, Any]] = {}


REASONING_SCHEMA = """
Return ONLY valid JSON with keys:
{
  "variant": {"gene": "...", "protein_change": "...", "hgvs_p": "..."},
  "structure": {... numeric features verbatim ...},
  "evidence": [...],
  "classification": "known_driver|likely_driver|vus|unknown",
  "evidence_tier": "strong|weak|none",
  "target_reasoning": {
    "mechanism": "...",
    "mechanism_hypothesis": "...",
    "therapy": {
      "sensitivity": [...],
      "resistance": [...],
      "context": "...",
      "recommendation_status": "confident|insufficient_evidence"
    },
    "next_best_action": "standard_of_care|tumor_board|clinical_trial|structural_rescue",
    "confidence": "0.0-1.0"
  }
}
Use ONLY the provided numeric structure features; do not invent pLDDT values.
If evidence_tier is weak or none, set recommendation_status to insufficient_evidence.
"""


def parse_reasoning_json(text: str) -> dict[str, Any]:
    """Parse model output; strip ```json fences and tolerate truncated JSON."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```\s*$", "", raw)
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start < 0 or end <= start:
        return {"raw": text}
    chunk = raw[start:end]
    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        return {"raw": text, "parse_error": True}


def build_prompt(target: dict, structure: dict, evidence: list[dict]) -> str:
    return (
        f"Variant: {target['gene']} {target['mutation']} ({target.get('class')})\n"
        f"Disease context: {target.get('disease_context', '')}\n"
        f"Verified structural features (from AlphaFold {target['uniprot']}):\n"
        f"{json.dumps({k: structure[k] for k in structure if not k.endswith('_html') and k != 'pdb_path'}, indent=2)}\n\n"
        f"Evidence items:\n{json.dumps(evidence, indent=2)}\n\n"
        f"{REASONING_SCHEMA}"
    )


def _load_model(model_id: str, lora_path: str | None = None):
    cache_key = (model_id, lora_path)
    if cache_key in _VL_MODEL_CACHE:
        return _VL_MODEL_CACHE[cache_key]

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(model_id)
    kwargs = {"dtype": torch.bfloat16, "device_map": "cuda"}
    if lora_path and Path(lora_path).exists():
        from peft import PeftModel
        from src import progress

        progress.log("single", f"loading Qwen2.5-VL + LoRA ({lora_path})")
        base = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        model = PeftModel.from_pretrained(base, lora_path)
    else:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)

    _VL_MODEL_CACHE[cache_key] = (proc, model)
    return proc, model


def vl_generate(
    prompt: str,
    *,
    image_path: str | None = None,
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    lora_path: str | None = None,
    max_new_tokens: int = 1024,
    agent_role: str = "SingleAgent",
    architecture: str = "single",
    label: str = "qwen_vl_generate",
    query_id: str = "",
    gene: str = "",
    mutation: str = "",
) -> dict[str, Any]:
    """Shared VL generate path for single agent and Structure expert."""
    from src import progress

    mm = bool(image_path and Path(image_path).exists())
    if mm:
        progress.log(architecture, "multimodal_image=true | structure PNG attached", gene=gene, mutation=mutation)

    with metrics.track(label, agent_role=agent_role, model=model_id, round_idx=1) as m:
        import torch
        from qwen_vl_utils import process_vision_info

        proc, model = _load_model(model_id, lora_path)
        content = [{"type": "text", "text": prompt}]
        if mm:
            content.insert(0, {"type": "image", "image": f"file://{Path(image_path).resolve()}"})
        messages = [{"role": "user", "content": content}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = proc(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to("cuda")
        in_len = inputs["input_ids"].shape[1]
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        latency = time.perf_counter() - t0
        gen = proc.decode(out[0][in_len:], skip_special_tokens=True)
        out_tok = int(out[0].shape[0] - in_len)
        m.set_tokens(ingress=in_len, egress=out_tok)
        metrics.log_llm_call(
            agent_role,
            model_id,
            1,
            in_len,
            gen,
            out_tok,
            latency,
            query_id=query_id,
            architecture=architecture,
            label=label,
            gene=gene,
            mutation=mutation,
            multimodal_image=mm,
            prompt_text=prompt,
        )
        from src.llm_client import split_completion

        reasoning, output = split_completion(gen)
        return {
            "content": gen,
            "prompt": prompt,
            "system_prompt": "",
            "reasoning": reasoning,
            "output": output or gen,
            "metadata": {
                "model": model_id,
                "prompt_tokens": in_len,
                "completion_tokens": out_tok,
                "total_tokens": in_len + out_tok,
                "multimodal_image": mm,
            },
        }


def reason_single(
    target: dict,
    structure: dict,
    evidence: list[dict],
    *,
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    lora_path: str | None = None,
    image_path: str | None = None,
    max_new_tokens: int = 1024,
) -> dict[str, Any]:
    setup_env()
    prompt = build_prompt(target, structure, evidence)
    qid = f"{target['gene']}_{target['mutation']}"

    with metrics.phase("reason_single", model=model_id):
        resp = vl_generate(
            prompt,
            image_path=image_path,
            model_id=model_id,
            lora_path=lora_path,
            max_new_tokens=max_new_tokens,
            query_id=qid,
            gene=target["gene"],
            mutation=target["mutation"],
        )
        parsed = parse_reasoning_json(resp["content"])
        return {
            **parsed,
            "llm_io": {
                "prompt": resp.get("prompt", prompt),
                "reasoning": resp.get("reasoning", ""),
                "output": resp.get("output", resp["content"]),
                "content": resp["content"],
            },
        }
