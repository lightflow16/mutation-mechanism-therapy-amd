"""Qwen2.5-VL single-agent reasoning (baseline + LoRA path)."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from src import metrics
from src.config import setup_env


REASONING_SCHEMA = """
Return ONLY valid JSON with keys:
{
  "variant": {"gene": "...", "protein_change": "...", "hgvs_p": "..."},
  "structure": {... numeric features verbatim ...},
  "evidence": [...],
  "target_reasoning": {
    "mechanism": "...",
    "therapy": {"sensitivity": [...], "resistance": [...], "context": "..."},
    "confidence": "0.0-1.0"
  }
}
Use ONLY the provided numeric structure features; do not invent pLDDT values.
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
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(model_id)
    kwargs = {"dtype": torch.bfloat16, "device_map": "cuda"}
    if lora_path and Path(lora_path).exists():
        from peft import PeftModel
        base = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        model = PeftModel.from_pretrained(base, lora_path)
    else:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    return proc, model


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

    with metrics.phase("reason_single", model=model_id):
        with metrics.track("qwen_vl_generate", agent_role="SingleAgent", model=model_id, round_idx=1) as m:
            import torch
            from qwen_vl_utils import process_vision_info

            proc, model = _load_model(model_id, lora_path)
            content = [{"type": "text", "text": prompt}]
            if image_path and Path(image_path).exists():
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
            out = model.generate(**inputs, max_new_tokens=max_new_tokens)
            latency = time.perf_counter() - t0
            gen = proc.decode(out[0][in_len:], skip_special_tokens=True)
            out_tok = int(out[0].shape[0] - in_len)
            m.set_tokens(ingress=in_len, egress=out_tok)
            metrics.log_llm_call(
                "SingleAgent",
                model_id,
                1,
                in_len,
                gen,
                out_tok,
                latency,
                query_id=f"{target['gene']}_{target['mutation']}",
                architecture="single",
                label="qwen_vl_generate",
                gene=target["gene"],
                mutation=target["mutation"],
            )
            parsed = parse_reasoning_json(gen)
            return parsed
