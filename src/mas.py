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
    (
        "Structure",
        "Interpret ONLY numeric AlphaFold features (pLDDT, region, residue). "
        "Do NOT list therapies or CIViC evidence.",
    ),
    (
        "Mechanism",
        "Propose ONE pathway mechanism (GOF/LOF, binding, signaling). "
        "Do NOT repeat the Structure expert's pLDDT recap verbatim.",
    ),
    (
        "Evidence",
        "List ONLY curated evidence items (source, therapy, direction, level). "
        "Do NOT repeat mechanism prose or structural feature dumps.",
    ),
    (
        "Therapy",
        "Recommend therapies with sensitivity/resistance and disease context. "
        "Do NOT paste the Evidence expert's bullet list verbatim.",
    ),
]

_JSON_ONLY = (
    "Respond with ONLY a JSON object (no markdown fences):\n"
    '{"stance_summary": "<=2 sentences, YOUR unique role view>", '
    '"key_claim": "<=1 sentence, role-specific biochemical claim>", '
    '"detail": "<=120 words, optional bullets>"}\n'
    "Do NOT copy prior agents. Do NOT use headers like '### Summary of Evidence'."
)

_VUS_SUFFIX = (
    "\nIMPORTANT: evidence is weak or absent. "
    'Set key_claim to abstention; in detail include recommendation_status=insufficient_evidence '
    "and next_best_action=tumor_board. Do not invent drug names."
)

_ECHO_HEADER = re.compile(r"^#+\s*summary of evidence", re.I)


def _endpoint(cfg: dict, role: str) -> tuple[str, str]:
    ep = cfg.get("serving", {}).get("endpoints", {})
    key = {
        "Planner": "planner",
        "Therapy": "reasoner",
        "Decider": "reasoner",
    }.get(role, "mechanism")
    e = ep.get(key, ep.get("reasoner", {}))
    return e.get("base_url", "http://localhost:8000/v1"), e.get("model", "qwen2.5-vl-7b")


def _first_sentence(text: str, limit: int = 200) -> str:
    t = (text or "").strip().replace("\n", " ")
    for sep in (". ", "; ", " — "):
        if sep in t:
            t = t.split(sep, 1)[0] + sep.strip()
            break
    return t[:limit] if len(t) > limit else t


def _parse_role_json(text: str) -> dict[str, str]:
    """Extract stance_summary / key_claim from structured agent output."""
    out: dict[str, str] = {}
    if not text:
        return out
    try:
        obj = parse_reasoning_json(text)
        if isinstance(obj, dict):
            for k in ("stance_summary", "key_claim", "detail"):
                if obj.get(k):
                    out[k] = str(obj[k]).strip()
            if out:
                return out
    except Exception:
        pass
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict):
            for k in ("stance_summary", "key_claim", "detail"):
                if obj.get(k):
                    out[k] = str(obj[k]).strip()
            if out:
                return out
    except json.JSONDecodeError:
        pass
    clean = text.strip()
    if _ECHO_HEADER.match(clean):
        clean = re.sub(r"^#+\s*.+\n+", "", clean, count=1, flags=re.I).strip()
    out["stance_summary"] = _first_sentence(clean, 220)
    out["key_claim"] = _first_sentence(clean, 120)
    out["detail"] = clean[:1200]
    return out


def _bb_compact(public: list[dict], *, max_entries: int = 8) -> str:
    """Compact blackboard: one line per prior agent (structured fields preferred)."""
    lines: list[str] = []
    for m in public[-max_entries:]:
        agent = m.get("agent", "?")
        stance = m.get("stance_summary") or _first_sentence(m.get("content", ""), 160)
        claim = m.get("key_claim", "")
        line = f"[{agent}] {stance}"
        if claim and claim != stance:
            line += f" | claim: {claim[:100]}"
        lines.append(line)
    return "\n".join(lines)


def _append_step(
    agent: str,
    step_type: str,
    llm_resp: dict[str, Any],
) -> dict[str, Any]:
    parsed = _parse_role_json(llm_resp.get("content", ""))
    step = trace_step_from_response(agent, step_type, llm_resp)
    if parsed.get("stance_summary"):
        step["stance_summary"] = parsed["stance_summary"]
    if parsed.get("key_claim"):
        step["key_claim"] = parsed["key_claim"]
    if parsed.get("detail"):
        step["content"] = parsed["detail"]
    elif parsed.get("stance_summary"):
        step["content"] = parsed["stance_summary"]
    return step


def _round1_consensus(critic_text: str, conflict_text: str) -> bool:
    combined = f"{critic_text}\n{conflict_text}".lower()
    flags = ("hallucin", "contradict", "conflict", "disagree", "incorrect", "unsupported", "error")
    return not any(f in combined for f in flags)


def _parse_rubric_score(text: str) -> tuple[int, str]:
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
        f"{_bb_compact(public)}"
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


def _expert_system(role: str) -> str:
    return (
        f"You are the {role} expert on a tumor board blackboard. "
        f"{_JSON_ONLY} "
        f"Never echo prior agents' headers or bullet lists."
    )


def _critic_system() -> str:
    return (
        "You are the Critic. " + _JSON_ONLY +
        " stance_summary = issues found; key_claim = pass/fail on evidence grounding."
    )


def _resolver_system() -> str:
    return (
        "You are the ConflictResolver. " + _JSON_ONLY +
        " stance_summary = conflicts identified; key_claim = resolution rule applied."
    )


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

    structure_json = {
        k: v
        for k, v in structure.items()
        if k not in ("render_html", "pdb_path", "structure_image_path")
    }
    problem = (
        f"Analyze {target['gene']} {target['mutation']} ({target['class']}) "
        f"for mechanism and therapy in {target.get('disease_context', 'cancer')}.\n"
        f"Classification: {target.get('classification', 'unknown')} | "
        f"evidence_tier: {target.get('evidence_tier', 'unknown')}\n"
        f"Structure features: {json.dumps(structure_json)}\n"
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
            f"Plan the analysis steps for:\n{problem}\n\n"
            "Output a short numbered plan (plain text, max 10 steps).",
            base_url=base_url,
            model=model,
            system_prompt="You are the Planner. Output a short numbered plan only.",
            agent_role="Planner",
            round_idx=0,
            label="planner",
            **llm_ctx,
        )
        step = trace_step_from_response("Planner", "plan", pr)
        step["stance_summary"] = _first_sentence(pr["content"], 200)
        step["key_claim"] = "Analysis plan staged for specialist experts."
        public.append(step)
        total_tokens += pr["metadata"]["total_tokens"]
        progress.echo_blackboard_step(
            target["gene"], target["mutation"], 0, max_rounds, "Planner", step["content"]
        )

        rounds_done = 0
        for rnd in range(1, max_rounds + 1):
            if early_exit:
                break
            rounds_done = rnd
            for role, desc in EXPERTS:
                prior = _bb_compact(public)
                prompt = (
                    f"Role: {role}. Task: {desc}\n\n"
                    f"Case data:\n{problem}\n\n"
                    f"Prior blackboard (one-line summaries only — do not copy):\n{prior}\n\n"
                    f"{_JSON_ONLY}"
                    + (vus_note if role == "Therapy" else "")
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
                        prompt,
                        base_url=bu,
                        model=mo,
                        system_prompt=_expert_system(role),
                        agent_role=role,
                        round_idx=rnd,
                        label=f"expert_{role.lower()}",
                        **llm_ctx,
                    )
                step = _append_step(role, "expert", llm_resp)
                public.append(step)
                total_tokens += llm_resp["metadata"]["total_tokens"]
                progress.echo_blackboard_step(
                    target["gene"],
                    target["mutation"],
                    rnd,
                    max_rounds,
                    role,
                    step.get("stance_summary") or step["content"],
                )

                if role == "Mechanism" and rnd == 1:
                    rubric_before, gaps, tok = _invoke_critic_rubric(
                        cfg, public, llm_ctx=llm_ctx, rnd=rnd
                    )
                    total_tokens += tok
                    if rubric_before < 2:
                        bu, mo = _endpoint(cfg, "Mechanism")
                        reflex = call_llm(
                            f"Revise mechanism JSON using Critic feedback (gaps: {gaps}).\n"
                            f"Case:\n{problem}\n\nPrior:\n{_bb_compact(public)}\n\n{_JSON_ONLY}",
                            base_url=bu,
                            model=mo,
                            system_prompt=_expert_system("Mechanism") + " (reflexion pass)",
                            agent_role="Mechanism",
                            round_idx=rnd,
                            label="mechanism_reflexion",
                            **llm_ctx,
                        )
                        public.append(_append_step("Mechanism", "reflexion", reflex))
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
                f"Check claims vs evidence. Flag hallucinated pLDDT or unsupported therapies.\n"
                f"Blackboard:\n{_bb_compact(public)}\n\n{_JSON_ONLY}",
                base_url=bu,
                model=mo,
                system_prompt=_critic_system(),
                agent_role="Critic",
                round_idx=rnd,
                label="critic",
                **llm_ctx,
            )
            step = _append_step("Critic", "critique", cr)
            public.append(step)
            total_tokens += cr["metadata"]["total_tokens"]
            progress.echo_blackboard_step(
                target["gene"],
                target["mutation"],
                rnd,
                max_rounds,
                "Critic",
                step.get("stance_summary") or step["content"],
            )

            bu, mo = _endpoint(cfg, "ConflictResolver")
            xr = call_llm(
                f"Resolve sensitivity vs resistance conflicts by disease context.\n"
                f"Blackboard:\n{_bb_compact(public)}\n\n{_JSON_ONLY}",
                base_url=bu,
                model=mo,
                system_prompt=_resolver_system(),
                agent_role="ConflictResolver",
                round_idx=rnd,
                label="conflict_resolver",
                **llm_ctx,
            )
            step = _append_step("ConflictResolver", "resolution", xr)
            public.append(step)
            total_tokens += xr["metadata"]["total_tokens"]
            progress.echo_blackboard_step(
                target["gene"],
                target["mutation"],
                rnd,
                max_rounds,
                "ConflictResolver",
                step.get("stance_summary") or step["content"],
            )

            if rnd == 1 and _round1_consensus(
                step.get("stance_summary") or cr["content"],
                public[-1].get("stance_summary") or xr["content"],
            ):
                early_exit = True
                progress.echo_blackboard_step(
                    target["gene"],
                    target["mutation"],
                    rnd,
                    max_rounds,
                    "system",
                    "consensus reached",
                    early_exit=True,
                )

        bu, mo = _endpoint(cfg, "Decider")
        decider_prompt = (
            f"Produce the final decision as JSON with EXACTLY this structure:\n"
            f'{{"mechanism": {{"description": "...", "pathway": "..."}},\n'
            f' "therapy": {{"sensitivity": ["drug1", ...], "resistance": ["drug1", ...], "context": "..."}},\n'
            f' "confidence": "0.0-1.0",\n'
            f' "next_best_action": "standard_of_care|tumor_board|clinical_trial|structural_rescue"}}\n\n'
            f"Blackboard summaries:\n{_bb_compact(public)}"
            + vus_note
        )
        dr = call_llm(
            decider_prompt,
            base_url=bu,
            model=mo,
            system_prompt="You are the Decider. Return valid JSON only. Do NOT wrap the JSON in a 'reasoning' key.",
            agent_role="Decider",
            round_idx=rounds_done + 1,
            label="decider",
            **llm_ctx,
        )
        step = trace_step_from_response("Decider", "decision", dr)
        step["stance_summary"] = "Final therapy decision synthesized."
        step["key_claim"] = _first_sentence(dr.get("content", ""), 120)
        public.append(step)
        total_tokens += dr["metadata"]["total_tokens"]
        progress.echo_blackboard_step(
            target["gene"],
            target["mutation"],
            rounds_done,
            max_rounds,
            "Decider",
            step.get("key_claim") or step["content"],
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
