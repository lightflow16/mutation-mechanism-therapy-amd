"""Adversarial debate architecture: sensitivity vs resistance advocates + Judge."""
from __future__ import annotations

import json
from typing import Any

from src import metrics, progress
from src.config import load_config
from src.llm_client import call_llm, trace_step_from_response
from src.reason import parse_reasoning_json


def run_debate(
    target: dict,
    structure: dict,
    evidence: list[dict],
) -> dict[str, Any]:
    cfg = load_config()
    ep = cfg.get("serving", {}).get("endpoints", {}).get("reasoner", {})
    base_url = ep.get("base_url", "http://localhost:8000/v1")
    model = ep.get("model", "qwen2.5-vl-7b")
    problem = (
        f"Analyze {target['gene']} {target['mutation']} ({target.get('class')}) "
        f"for mechanism and therapy in {target.get('disease_context', 'cancer')}.\n"
        f"Structure: {json.dumps({k: v for k, v in structure.items() if k not in ('render_html', 'pdb_path', 'structure_image_path')})}\n"
        f"Evidence: {json.dumps(evidence)}"
    )
    qid = f"{target['gene']}_{target['mutation']}"
    llm_ctx = dict(
        query_id=qid,
        architecture="debate",
        gene=target["gene"],
        mutation=target["mutation"],
    )
    trace: list[dict] = []
    total_tokens = 0

    progress.banner(f"Debate | {target['gene']} {target['mutation']}")

    with metrics.phase(f"debate_{target['gene']}_{target['mutation']}", model=model):
        pro = call_llm(
            f"{problem}\n\nArgue FOR sensitivity therapies with evidence.",
            base_url=base_url, model=model,
            system_prompt="You are DebatePro — advocate sensitivity/response therapies.",
            agent_role="DebatePro", round_idx=1, label="debate_pro",
            **llm_ctx,
        )
        trace.append(trace_step_from_response("DebatePro", "argument", pro))
        total_tokens += pro["metadata"]["total_tokens"]
        progress.log("debate", "DebatePro stance", preview=pro["content"][:200])

        con = call_llm(
            f"{problem}\n\nArgue FOR resistance mechanisms and context-specific limitations.",
            base_url=base_url, model=model,
            system_prompt="You are DebateCon — advocate resistance/context caveats.",
            agent_role="DebateCon", round_idx=1, label="debate_con",
            **llm_ctx,
        )
        trace.append(trace_step_from_response("DebateCon", "argument", con))
        total_tokens += con["metadata"]["total_tokens"]
        progress.log("debate", "DebateCon stance", preview=con["content"][:200])

        judge = call_llm(
            f"Merge these debate positions into final JSON (mechanism + therapy + confidence).\n"
            f"PRO: {pro['content'][:1200]}\nCON: {con['content'][:1200]}",
            base_url=base_url, model=model,
            system_prompt="You are DebateJudge. Return valid JSON only.",
            agent_role="DebateJudge", round_idx=2, label="debate_judge",
            **llm_ctx,
        )
        trace.append(trace_step_from_response("DebateJudge", "decision", judge))
        total_tokens += judge["metadata"]["total_tokens"]

    parsed = parse_reasoning_json(judge["content"])
    return {
        "architecture": "debate",
        "target_reasoning": parsed,
        "debate_trace": trace,
        "total_tokens": total_tokens,
    }
