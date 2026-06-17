"""Adversarial debate architecture: sensitivity vs resistance advocates + Judge."""
from __future__ import annotations

import json
from typing import Any

from src import metrics, progress
from src.config import load_config
from src.llm_client import call_llm, trace_step_from_response
from src.reason import parse_reasoning_json, vl_generate


def _debate_call(
    prompt: str,
    *,
    lora_path: str | None,
    base_url: str,
    model: str,
    system_prompt: str | None = None,
    agent_role: str,
    round_idx: int,
    label: str,
    query_id: str,
    gene: str,
    mutation: str,
) -> dict[str, Any]:
    """Route a debate-agent LLM call through the LoRA-aware VL model when a
    fine-tuned adapter is available, otherwise fall back to call_llm."""
    if lora_path:
        return vl_generate(
            prompt,
            lora_path=lora_path,
            system_prompt=system_prompt,
            agent_role=agent_role,
            architecture="debate",
            label=label,
            query_id=query_id,
            gene=gene,
            mutation=mutation,
        )
    return call_llm(
        prompt,
        base_url=base_url,
        model=model,
        system_prompt=system_prompt,
        agent_role=agent_role,
        round_idx=round_idx,
        label=label,
        query_id=query_id,
        architecture="debate",
        gene=gene,
        mutation=mutation,
    )


def run_debate(
    target: dict,
    structure: dict,
    evidence: list[dict],
    *,
    lora_path: str | None = None,
) -> dict[str, Any]:
    """Adversarial debate (DebatePro / DebateCon / DebateJudge).

    When lora_path is provided all three agent calls go through the LoRA-aware
    VL backbone so that debates are grounded in fine-tuned oncology knowledge.
    """
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
    _call_kwargs = dict(
        lora_path=lora_path,
        base_url=base_url,
        model=model,
        query_id=qid,
        gene=target["gene"],
        mutation=target["mutation"],
    )
    trace: list[dict] = []
    total_tokens = 0

    progress.banner(f"Debate | {target['gene']} {target['mutation']}")

    with metrics.phase(f"debate_{target['gene']}_{target['mutation']}", model=model):
        pro = _debate_call(
            f"{problem}\n\nArgue FOR sensitivity therapies with evidence.",
            system_prompt="You are DebatePro — advocate sensitivity/response therapies.",
            agent_role="DebatePro", round_idx=1, label="debate_pro",
            **_call_kwargs,
        )
        trace.append(trace_step_from_response("DebatePro", "argument", pro))
        total_tokens += pro["metadata"]["total_tokens"]
        progress.log("debate", "DebatePro stance", preview=pro["content"][:200])

        con = _debate_call(
            f"{problem}\n\nArgue FOR resistance mechanisms and context-specific limitations.",
            system_prompt="You are DebateCon — advocate resistance/context caveats.",
            agent_role="DebateCon", round_idx=1, label="debate_con",
            **_call_kwargs,
        )
        trace.append(trace_step_from_response("DebateCon", "argument", con))
        total_tokens += con["metadata"]["total_tokens"]
        progress.log("debate", "DebateCon stance", preview=con["content"][:200])

        judge = _debate_call(
            f"Merge these debate positions into final JSON (mechanism + therapy + confidence).\n"
            f"PRO: {pro['content'][:1200]}\nCON: {con['content'][:1200]}",
            system_prompt="You are DebateJudge. Return valid JSON only.",
            agent_role="DebateJudge", round_idx=2, label="debate_judge",
            **_call_kwargs,
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
