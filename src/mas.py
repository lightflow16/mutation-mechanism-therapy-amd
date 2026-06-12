"""Oncology blackboard MAS: Planner -> Experts -> Critic -> ConflictResolver -> Decider."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from src import metrics
from src.config import load_config
from src.llm_client import call_vllm

BMAS_ROOT = Path(__file__).resolve().parents[1] / "external" / "sde_project_bMAS"
if str(BMAS_ROOT) not in sys.path:
    sys.path.insert(0, str(BMAS_ROOT))


EXPERTS = [
    ("Structure", "Interpret ONLY the provided numeric AlphaFold features; cite pLDDT and region verbatim."),
    ("Mechanism", "Propose a biological mechanism linking the variant to pathway effects using structure + evidence."),
    ("Evidence", "Summarize ClinVar/CIViC evidence; list therapies with sensitivity/resistance direction."),
    ("Therapy", "Recommend FDA-approved or investigational therapies with citations; respect disease context."),
]


def _endpoint(cfg: dict, role: str) -> tuple[str, str]:
    ep = cfg.get("serving", {}).get("endpoints", {})
    key = {
        "Planner": "planner",
        "Therapy": "reasoner",
        "Decider": "reasoner",
    }.get(role, "mechanism")
    e = ep.get(key, ep.get("reasoner", {}))
    return e.get("base_url", "http://localhost:8000/v1"), e.get("model", "qwen2.5-vl-7b")


def _bb_text(public: list[dict]) -> str:
    return "\n".join(f"[{m['agent']}|{m['type']}] {m['content'][:800]}" for m in public[-12:])


def run_blackboard(
    target: dict,
    structure: dict,
    evidence: list[dict],
    *,
    max_rounds: int = 2,
) -> dict[str, Any]:
    cfg = load_config()
    problem = (
        f"Analyze {target['gene']} {target['mutation']} ({target['class']}) "
        f"for mechanism and therapy in {target.get('disease_context', 'cancer')}.\n"
        f"Structure features: {json.dumps({k: v for k, v in structure.items() if k not in ('render_html', 'pdb_path')})}\n"
        f"Evidence: {json.dumps(evidence)}\n"
        f"Route: {target.get('pathway')}."
    )
    public: list[dict] = []
    total_tokens = 0
    dr = {"content": "{}"}

    with metrics.phase(f"blackboard_{target['gene']}_{target['mutation']}", model="bMAS"):
        base_url, model = _endpoint(cfg, "Planner")
        pr = call_vllm(
            f"Plan the analysis steps for:\n{problem}",
            base_url=base_url, model=model,
            system_prompt="You are the Planner. Output a short numbered plan.",
            agent_role="Planner", round_idx=0, label="planner",
        )
        public.append({"agent": "Planner", "type": "plan", "content": pr["content"]})
        total_tokens += pr["metadata"]["total_tokens"]

        for rnd in range(1, max_rounds + 1):
            for role, desc in EXPERTS:
                bu, mo = _endpoint(cfg, role)
                prompt = (
                    f"Role: {role}. {desc}\nProblem:\n{problem}\n\nBlackboard:\n{_bb_text(public)}"
                )
                er = call_vllm(
                    prompt, base_url=bu, model=mo,
                    system_prompt=f"You are the {role} expert.",
                    agent_role=role, round_idx=rnd, label=f"expert_{role.lower()}",
                )
                public.append({"agent": role, "type": "expert", "content": er["content"]})
                total_tokens += er["metadata"]["total_tokens"]

            bu, mo = _endpoint(cfg, "Critic")
            cr = call_vllm(
                f"Check claims vs evidence. Flag hallucinated pLDDT or therapy claims.\n{_bb_text(public)}",
                base_url=bu, model=mo, system_prompt="You are the Critic.",
                agent_role="Critic", round_idx=rnd, label="critic",
            )
            public.append({"agent": "Critic", "type": "critique", "content": cr["content"]})
            total_tokens += cr["metadata"]["total_tokens"]

            bu, mo = _endpoint(cfg, "ConflictResolver")
            xr = call_vllm(
                f"Resolve sensitivity vs resistance conflicts by disease context.\n{_bb_text(public)}",
                base_url=bu, model=mo, system_prompt="You are the ConflictResolver.",
                agent_role="ConflictResolver", round_idx=rnd, label="conflict_resolver",
            )
            public.append({"agent": "ConflictResolver", "type": "resolution", "content": xr["content"]})
            total_tokens += xr["metadata"]["total_tokens"]

        bu, mo = _endpoint(cfg, "Decider")
        dr = call_vllm(
            f"Produce final JSON reasoning (mechanism + therapy + confidence).\n{_bb_text(public)}",
            base_url=bu, model=mo,
            system_prompt="You are the Decider. Return valid JSON only.",
            agent_role="Decider", round_idx=max_rounds + 1, label="decider",
        )
        public.append({"agent": "Decider", "type": "decision", "content": dr["content"]})
        total_tokens += dr["metadata"]["total_tokens"]

    try:
        text = dr["content"]
        start, end = text.find("{"), text.rfind("}") + 1
        parsed = json.loads(text[start:end]) if start >= 0 else {"raw": text}
    except Exception:
        parsed = {"raw": dr["content"]}

    return {
        "architecture": "blackboard",
        "target_reasoning": parsed,
        "blackboard_trace": public,
        "total_tokens": total_tokens,
    }
