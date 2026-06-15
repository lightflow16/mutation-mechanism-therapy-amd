"""Chain-of-Thought single-model reasoning baseline."""
from __future__ import annotations

import json
import time
from typing import Any

from src import metrics
from src.config import load_config
from src.llm_client import call_llm
from src.reason import build_prompt


def run_cot(
    target: dict,
    structure: dict,
    evidence: list[dict],
) -> dict[str, Any]:
    cfg = load_config()
    ep = cfg.get("serving", {}).get("endpoints", {}).get("reasoner", {})
    base_url = ep.get("base_url", "http://localhost:8000/v1")
    model = ep.get("model", "qwen2.5-vl-7b")

    prompt = build_prompt(target, structure, evidence)
    cot_prompt = (
        f"{prompt}\n\nThink step by step in <think>...</think>, "
        "then output the final JSON."
    )

    with metrics.phase(f"cot_{target['gene']}_{target['mutation']}", model=model):
        t0 = time.perf_counter()
        resp = call_llm(
            cot_prompt,
            base_url=base_url,
            model=model,
            system_prompt="You are a precision oncology reasoning assistant.",
            agent_role="CoT",
            round_idx=1,
            label="cot_reason",
        )
        metrics.log_llm_call(
            "CoT", model, 1,
            resp["metadata"]["prompt_tokens"],
            resp["content"],
            resp["metadata"]["completion_tokens"],
            time.perf_counter() - t0,
            query_id=f"{target['gene']}_{target['mutation']}",
        )
        text = resp["content"]
        try:
            start, end = text.find("{"), text.rfind("}") + 1
            parsed = json.loads(text[start:end]) if start >= 0 else {"raw": text}
        except json.JSONDecodeError:
            parsed = {"raw": text}

    return {"architecture": "cot", "target_reasoning": parsed, "total_tokens": resp["metadata"]["total_tokens"]}
