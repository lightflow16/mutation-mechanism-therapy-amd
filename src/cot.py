"""Chain-of-Thought single-model reasoning baseline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src import metrics, progress
from src.config import load_config
from src.llm_client import call_llm
from src.reason import build_prompt, parse_reasoning_json, vl_generate


def run_cot(
    target: dict,
    structure: dict,
    evidence: list[dict],
    *,
    image_path: str | None = None,
) -> dict[str, Any]:
    cfg = load_config()
    ep = cfg.get("serving", {}).get("endpoints", {}).get("reasoner", {})
    base_url = ep.get("base_url", "http://localhost:8000/v1")
    model = ep.get("model", "qwen2.5-vl-7b")
    img = image_path or structure.get("structure_image_path")
    qid = f"{target['gene']}_{target['mutation']}"

    prompt = build_prompt(target, structure, evidence)
    cot_prompt = (
        f"{prompt}\n\nThink step by step in <think>...</think>, "
        "then output the final JSON."
    )

    progress.banner(f"CoT | {target['gene']} {target['mutation']} | starting chain-of-thought")

    with metrics.phase(f"cot_{target['gene']}_{target['mutation']}", model=model):
        if img and Path(img).exists():
            resp = vl_generate(
                cot_prompt,
                image_path=img,
                agent_role="CoT",
                architecture="cot",
                label="cot_reason",
                query_id=qid,
                gene=target["gene"],
                mutation=target["mutation"],
            )
            text = resp["content"]
            total_tokens = resp["metadata"]["total_tokens"]
            multimodal = True
        else:
            resp = call_llm(
                cot_prompt,
                base_url=base_url,
                model=model,
                system_prompt="You are a precision oncology reasoning assistant.",
                agent_role="CoT",
                round_idx=1,
                label="cot_reason",
                query_id=qid,
                architecture="cot",
                gene=target["gene"],
                mutation=target["mutation"],
            )
            text = resp["content"]
            total_tokens = resp["metadata"]["total_tokens"]
            multimodal = False

        parsed = parse_reasoning_json(text)
        if "target_reasoning" in parsed and isinstance(parsed["target_reasoning"], dict):
            inner = parsed["target_reasoning"]
            if "target_reasoning" in inner:
                parsed = {**parsed, "target_reasoning": inner["target_reasoning"]}
            elif "therapy" in inner or "mechanism" in inner:
                parsed = inner

    return {
        "architecture": "cot",
        "target_reasoning": parsed,
        "total_tokens": total_tokens,
        "multimodal_image": multimodal,
        "llm_io": {
            "prompt": resp.get("prompt", cot_prompt),
            "system_prompt": resp.get("system_prompt", ""),
            "reasoning": resp.get("reasoning", ""),
            "output": resp.get("output", text),
            "content": text,
        },
    }
