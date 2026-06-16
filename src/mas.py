"""Oncology blackboard MAS: Planner -> Experts -> Critic -> ConflictResolver -> Decider."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from src import metrics, progress
from src.config import load_config
from src.llm_client import call_llm, trace_step_from_response
from src.reason import parse_reasoning_json, vl_generate

BMAS_ROOT = Path(__file__).resolve().parents[1] / "external" / "sde_project_bMAS"
if str(BMAS_ROOT) not in sys.path:
    sys.path.insert(0, str(BMAS_ROOT))


EXPERTS = [
    ("Structure", "Interpret ONLY the provided numeric AlphaFold features; cite pLDDT and region verbatim."),
    ("Mechanism", "Propose a biological mechanism linking the variant to pathway effects using structure + evidence."),
    ("Evidence", "Summarize ClinVar/CIViC evidence; list therapies with sensitivity/resistance direction."),
    ("Therapy", "Recommend FDA-approved or investigational therapies with citations; respect disease context."),
]

_VUS_SUFFIX = (
    "\nIMPORTANT: evidence is weak or absent. "
    "Output therapy.recommendation_status=insufficient_evidence and "
    "next_best_action=tumor_board. Do not invent drug names."
)


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


def _round1_consensus(critic_text: str, conflict_text: str) -> bool:
    combined = f"{critic_text}\n{conflict_text}".lower()
    flags = ("hallucin", "contradict", "conflict", "disagree", "incorrect", "unsupported", "error")
    return not any(f in combined for f in flags)


def _parse_rubric_score(text: str) -> tuple[int, str]:
    """Parse mechanism rubric 0-2 from Critic response."""
    m = re.search(r"(?:score|rubric)[:\s]*([0-2])", text, re.I)
    score = int(m.group(1)) if m else 1
    gaps = ""
    gm = re.search(r"gaps?[:\s]*(.+)", text, re.I)
    if gm:
        gaps = gm.group(1).strip()[:300]
    return max(0, min(2, score)), gaps


def _invoke_critic_rubric(
    cfg: dict,
    public: list[dict],
    *,
    llm_ctx: dict,
    rnd: int,
) -> tuple[int, str, int]:
    bu, mo = _endpoint(cfg, "Critic")
    prompt = (
        "Score the Mechanism expert claim 0-2 (0=unsupported, 1=partial, 2=well-grounded). "
        "Reply with: score: N\\ngaps: ...\\n\n"
        f"{_bb_text(public)}"
    )
    cr = call_llm(
        prompt,
        base_url=bu,
        model=mo,
        system_prompt="You are the Mechanism rubric Critic.",
        agent_role="CriticRubric",
        round_idx=rnd,
        label="mechanism_rubric",
        **llm_ctx,
    )
    score, gaps = _parse_rubric_score(cr["content"])
    return score, gaps, cr["metadata"]["total_tokens"]


def run_blackboard(
    target: dict,
    structure: dict,
    evidence: list[dict],
    *,
    max_rounds: int = 2,
    image_path: str | None = None,
) -> dict[str, Any]:
    cfg = load_config()
    img = image_path or structure.get("structure_image_path")
    allow_therapy = target.get("allow_confident_therapy", True)
    vus_note = "" if allow_therapy else _VUS_SUFFIX

    problem = (
        f"Analyze {target['gene']} {target['mutation']} ({target['class']}) "
        f"for mechanism and therapy in {target.get('disease_context', 'cancer')}.\n"
        f"Classification: {target.get('classification', 'unknown')} | "
        f"evidence_tier: {target.get('evidence_tier', 'unknown')}\n"
        f"Structure features: {json.dumps({k: v for k, v in structure.items() if k not in ('render_html', 'pdb_path', 'structure_image_path')})}\n"
        f"Evidence: {json.dumps(evidence)}\n"
        f"Route: {target.get('pathway')}."
    )
    public: list[dict] = []
    total_tokens = 0
    dr = {"content": "{}"}
    qid = f"{target['gene']}_{target['mutation']}"
    llm_ctx = dict(
        query_id=qid,
        architecture="blackboard",
        gene=target["gene"],
        mutation=target["mutation"],
    )
    early_exit = False
    rubric_before = rubric_after = None

    progress.banner(f"Blackboard | {target['gene']} {target['mutation']}")

    with metrics.phase(f"blackboard_{target['gene']}_{target['mutation']}", model="bMAS"):
        base_url, model = _endpoint(cfg, "Planner")
        pr = call_llm(
            f"Plan the analysis steps for:\n{problem}",
            base_url=base_url, model=model,
            system_prompt="You are the Planner. Output a short numbered plan.",
            agent_role="Planner", round_idx=0, label="planner",
            **llm_ctx,
        )
        public.append(trace_step_from_response("Planner", "plan", pr))
        total_tokens += pr["metadata"]["total_tokens"]
        progress.echo_blackboard_step(
            target["gene"], target["mutation"], 0, max_rounds, "Planner", pr["content"]
        )

        rounds_done = 0
        for rnd in range(1, max_rounds + 1):
            if early_exit:
                break
            rounds_done = rnd
            for role, desc in EXPERTS:
                prompt = (
                    f"Role: {role}. {desc}\nProblem:\n{problem}\n\nBlackboard:\n{_bb_text(public)}"
                    + (vus_note if role in ("Therapy",) else "")
                )
                if role == "Structure" and img and Path(img).exists():
                    llm_resp = vl_generate(
                        prompt,
                        image_path=img,
                        agent_role=role,
                        architecture="blackboard",
                        label=f"expert_{role.lower()}",
                        query_id=qid,
                        gene=target["gene"],
                        mutation=target["mutation"],
                    )
                else:
                    bu, mo = _endpoint(cfg, role)
                    llm_resp = call_llm(
                        prompt, base_url=bu, model=mo,
                        system_prompt=f"You are the {role} expert.",
                        agent_role=role, round_idx=rnd, label=f"expert_{role.lower()}",
                        **llm_ctx,
                    )
                content = llm_resp["content"]
                total_tokens += llm_resp["metadata"]["total_tokens"]
                public.append(trace_step_from_response(role, "expert", llm_resp))
                progress.echo_blackboard_step(
                    target["gene"], target["mutation"], rnd, max_rounds, role, content
                )

                if role == "Mechanism" and rnd == 1:
                    rubric_before, gaps, tok = _invoke_critic_rubric(
                        cfg, public, llm_ctx=llm_ctx, rnd=rnd
                    )
                    total_tokens += tok
                    if rubric_before < 2:
                        bu, mo = _endpoint(cfg, "Mechanism")
                        reflex = call_llm(
                            f"Revise mechanism using Critic feedback (gaps: {gaps}).\n"
                            f"Problem:\n{problem}\n\nBlackboard:\n{_bb_text(public)}",
                            base_url=bu, model=mo,
                            system_prompt="You are the Mechanism expert (reflexion pass).",
                            agent_role="Mechanism", round_idx=rnd, label="mechanism_reflexion",
                            **llm_ctx,
                        )
                        public.append(trace_step_from_response("Mechanism", "reflexion", reflex))
                        total_tokens += reflex["metadata"]["total_tokens"]
                        rubric_after, _, tok2 = _invoke_critic_rubric(
                            cfg, public, llm_ctx=llm_ctx, rnd=rnd
                        )
                        total_tokens += tok2
                        metrics.log_self_correction(
                            gene=target["gene"],
                            mutation=target["mutation"],
                            rubric_before=rubric_before,
                            rubric_after=rubric_after,
                        )
                    else:
                        rubric_after = rubric_before

            bu, mo = _endpoint(cfg, "Critic")
            cr = call_llm(
                f"Check claims vs evidence. Flag hallucinated pLDDT or therapy claims.\n{_bb_text(public)}",
                base_url=bu, model=mo, system_prompt="You are the Critic.",
                agent_role="Critic", round_idx=rnd, label="critic",
                **llm_ctx,
            )
            public.append(trace_step_from_response("Critic", "critique", cr))
            total_tokens += cr["metadata"]["total_tokens"]
            progress.echo_blackboard_step(
                target["gene"], target["mutation"], rnd, max_rounds, "Critic", cr["content"]
            )

            bu, mo = _endpoint(cfg, "ConflictResolver")
            xr = call_llm(
                f"Resolve sensitivity vs resistance conflicts by disease context.\n{_bb_text(public)}",
                base_url=bu, model=mo, system_prompt="You are the ConflictResolver.",
                agent_role="ConflictResolver", round_idx=rnd, label="conflict_resolver",
                **llm_ctx,
            )
            public.append(trace_step_from_response("ConflictResolver", "resolution", xr))
            total_tokens += xr["metadata"]["total_tokens"]
            progress.echo_blackboard_step(
                target["gene"], target["mutation"], rnd, max_rounds, "ConflictResolver", xr["content"]
            )

            if rnd == 1 and _round1_consensus(cr["content"], xr["content"]):
                early_exit = True
                progress.echo_blackboard_step(
                    target["gene"], target["mutation"], rnd, max_rounds, "system",
                    "consensus reached", early_exit=True,
                )

        bu, mo = _endpoint(cfg, "Decider")
        decider_prompt = (
            f"Produce final JSON reasoning (mechanism + therapy + confidence).\n{_bb_text(public)}"
            + vus_note
        )
        dr = call_llm(
            decider_prompt,
            base_url=bu, model=mo,
            system_prompt="You are the Decider. Return valid JSON only.",
            agent_role="Decider", round_idx=rounds_done + 1, label="decider",
            **llm_ctx,
        )
        public.append(trace_step_from_response("Decider", "decision", dr))
        total_tokens += dr["metadata"]["total_tokens"]
        progress.echo_blackboard_step(
            target["gene"], target["mutation"], rounds_done, max_rounds, "Decider", dr["content"]
        )

    try:
        parsed = parse_reasoning_json(dr["content"])
    except Exception:
        parsed = {"raw": dr["content"]}

    return {
        "architecture": "blackboard",
        "target_reasoning": parsed,
        "blackboard_trace": public,
        "total_tokens": total_tokens,
        "early_exit": early_exit,
        "mechanism_rubric_before": rubric_before,
        "mechanism_rubric_after": rubric_after,
        "multimodal_image": bool(img and Path(img).exists()),
    }
