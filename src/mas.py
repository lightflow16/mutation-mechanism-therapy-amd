"""Oncology blackboard MAS: Planner -> Experts -> Critic -> ConflictResolver -> Decider."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from src import metrics, progress
from src.config import load_config
from src.llm_client import call_llm, call_transformers_batch, trace_step_from_response, use_vllm
from src.reason import parse_reasoning_json, vl_generate

BMAS_ROOT = Path(__file__).resolve().parents[1] / "external" / "sde_project_bMAS"
if str(BMAS_ROOT) not in sys.path:
    sys.path.insert(0, str(BMAS_ROOT))


def _bb_call(
    prompt: str,
    *,
    lora_path: str | None,
    base_url: str,
    model: str,
    system_prompt: str | None = None,
    agent_role: str = "",
    round_idx: int | str = 1,
    label: str = "llm_call",
    query_id: str = "",
    gene: str = "",
    mutation: str = "",
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> dict[str, Any]:
    """Route a blackboard agent call through the LoRA-aware VL model when a
    fine-tuned adapter is available, otherwise fall back to call_llm.

    This lets every agent in the blackboard (Planner, Structure, Mechanism,
    Evidence, Therapy, Critic, ConflictResolver, Decider) benefit from the
    domain-specific LoRA weights trained on oncology variant data, enabling
    a proper base-model vs fine-tuned comparison across all architectures.
    """
    if lora_path:
        return vl_generate(
            prompt,
            lora_path=lora_path,
            system_prompt=system_prompt,
            agent_role=agent_role,
            architecture="blackboard",
            label=label,
            query_id=query_id,
            gene=gene,
            mutation=mutation,
            max_new_tokens=max_tokens,
        )
    return call_llm(
        prompt,
        base_url=base_url,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        agent_role=agent_role,
        round_idx=round_idx,
        label=label,
        query_id=query_id,
        architecture="blackboard",
        gene=gene,
        mutation=mutation,
    )


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


def _extract_drugs_from_trace(public: list[dict]) -> tuple[list[str], list[str]]:
    """Last-resort drug extraction from Therapy / Evidence / Decider trace text."""
    sensitivity: list[str] = []
    resistance: list[str] = []
    for step in reversed(public):
        if step.get("agent") not in ("Therapy", "Evidence", "Decider"):
            continue
        content = " ".join(filter(None, [
            step.get("detail"), step.get("content"), step.get("stance_summary"),
        ]))
        sm = re.search(r"sensitivity[:\s]+([A-Za-z][A-Za-z0-9\s,;/\-]+?)(?:\n|resistance|context|$)", content, re.I)
        if sm:
            sensitivity.extend(
                d.strip() for d in re.split(r"[,;]", sm.group(1))
                if d.strip() and 2 < len(d.strip()) < 50
            )
        rm = re.search(r"resistance[:\s]+([A-Za-z][A-Za-z0-9\s,;/\-]+?)(?:\n|sensitivity|context|$)", content, re.I)
        if rm:
            resistance.extend(
                d.strip() for d in re.split(r"[,;]", rm.group(1))
                if d.strip() and 2 < len(d.strip()) < 50
            )
        if sensitivity or resistance:
            break
    return list(dict.fromkeys(sensitivity)), list(dict.fromkeys(resistance))


def _normalize_decider_output(parsed: dict, public: list[dict]) -> dict:
    """Normalise Decider JSON to {mechanism, therapy:{sensitivity,resistance}, confidence, next_best_action}.

    Handles:
    - {"reasoning": {...}} wrapper the model sometimes adds despite instructions
    - {"structure": {"therapy": "<prose>", ...}} nesting (PIK3CA pattern)
    - therapy as a plain string instead of a dict
    - mechanism as a dict instead of a string
    Falls back to trace extraction when therapy lists remain empty after normalisation.
    """
    if not isinstance(parsed, dict) or "raw" in parsed:
        return parsed

    # Unwrap spurious {"reasoning": {...}} wrapper
    if "reasoning" in parsed and isinstance(parsed["reasoning"], dict) and "therapy" not in parsed:
        parsed = parsed["reasoning"]

    # Unwrap {"structure": {...}} nesting
    inner = parsed.get("structure")
    if isinstance(inner, dict) and "therapy" not in parsed and (
        "mechanism" in inner or "therapy" in inner or "confidence" in inner
    ):
        parsed = {**parsed, **inner}
        parsed.pop("structure", None)

    # Normalise mechanism dict → string
    mech = parsed.get("mechanism")
    if isinstance(mech, dict):
        parsed["mechanism"] = (
            mech.get("description") or mech.get("summary") or mech.get("pathway") or str(mech)
        )

    # Normalise therapy string → dict with empty lists (prose stored as context)
    therapy = parsed.get("therapy")
    if isinstance(therapy, str):
        parsed["therapy"] = {"sensitivity": [], "resistance": [], "context": therapy}
    elif not isinstance(therapy, dict):
        parsed["therapy"] = {"sensitivity": [], "resistance": [], "context": ""}
    else:
        parsed["therapy"].setdefault("sensitivity", [])
        parsed["therapy"].setdefault("resistance", [])

    # If lists are still empty, try the blackboard trace as a fallback
    if not parsed["therapy"]["sensitivity"] and not parsed["therapy"]["resistance"]:
        sens, res = _extract_drugs_from_trace(public)
        if sens or res:
            parsed["therapy"]["sensitivity"] = sens
            parsed["therapy"]["resistance"] = res

    return parsed


def _decider_context(public: list[dict]) -> str:
    """Return a tightly scoped context for the Decider (≤3 key claims).

    Instead of re-feeding the full compacted blackboard (~8 agent summaries,
    ~10× ingress amplification), emit only the three claims the Decider
    actually needs: mechanism, therapy evidence, and the conflict resolution rule.
    This reduces Decider ingress tokens by ~70-80%.
    """
    picks: dict[str, str] = {}
    for step in reversed(public):
        agent = step.get("agent", "")
        claim = step.get("key_claim") or step.get("stance_summary") or ""
        if not claim:
            continue
        if agent == "Mechanism" and "mechanism" not in picks:
            picks["mechanism"] = claim[:200]
        elif agent in ("Evidence", "Therapy") and "therapy" not in picks:
            picks["therapy"] = claim[:200]
        elif agent == "ConflictResolver" and "conflict" not in picks:
            picks["conflict"] = claim[:200]
        if len(picks) == 3:
            break
    lines = []
    if picks.get("mechanism"):
        lines.append(f"[Mechanism] {picks['mechanism']}")
    if picks.get("therapy"):
        lines.append(f"[Therapy/Evidence] {picks['therapy']}")
    if picks.get("conflict"):
        lines.append(f"[ConflictResolver] {picks['conflict']}")
    return "\n".join(lines) or _bb_compact(public, max_entries=3)


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
    lora_path: str | None = None,
) -> tuple[int, str, int]:
    bu, mo = _endpoint(cfg, "Critic")
    prompt = (
        "Score the Mechanism expert claim 0-2 (0=unsupported, 1=partial, 2=well-grounded). "
        "Reply with: score: N\\ngaps: ...\\n\n"
        f"{_bb_compact(public)}"
    )
    cr = _bb_call(
        prompt,
        lora_path=lora_path,
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


def _build_expert_request(
    role: str,
    desc: str,
    problem: str,
    public: list[dict],
    vus_note: str,
    cfg: dict,
    llm_ctx: dict,
    rnd: int,
) -> dict:
    """Build a call_transformers_batch request dict for a text expert."""
    bu, mo = _endpoint(cfg, role)
    prior = _bb_compact(public)
    prompt = (
        f"Role: {role}. Task: {desc}\n\n"
        f"Case data:\n{problem}\n\n"
        f"Prior blackboard (one-line summaries only — do not copy):\n{prior}\n\n"
        f"{_JSON_ONLY}"
        + (vus_note if role == "Therapy" else "")
    )
    return {
        "prompt": prompt,
        "system_prompt": _expert_system(role),
        "temperature": 0.2,
        "max_tokens": 512,
        "agent_role": role,
        "label": f"expert_{role.lower()}",
        "_role": role,
        "_bu": bu,
        "_mo": mo,
        **{k: v for k, v in llm_ctx.items()},
    }


def run_blackboard(
    target: dict,
    structure: dict,
    evidence: list[dict],
    *,
    max_rounds: int = 2,
    image_path: str | None = None,
    lora_path: str | None = None,
) -> dict[str, Any]:
    """Blackboard MAS with Planner → Experts → Critic → ConflictResolver → Decider.

    When lora_path is provided every agent call is routed through the LoRA-aware
    VL backbone (via _bb_call), so the full multi-agent pipeline benefits from
    the fine-tuned oncology weights.  Batching is automatically disabled when
    the LoRA path is active because vl_generate is a sequential single-model
    call that cannot be batched in the same way as call_transformers_batch.
    """
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
    parsed: dict[str, Any] = {}

    progress.banner(f"Blackboard | {target['gene']} {target['mutation']}")

    with metrics.phase(f"blackboard_{target['gene']}_{target['mutation']}", model="bMAS"):
        base_url, model = _endpoint(cfg, "Planner")
        pr = _bb_call(
            f"Plan the analysis steps for:\n{problem}\n\n"
            "Output a short numbered plan (plain text, max 10 steps).",
            lora_path=lora_path,
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

        pcfg = cfg.get("pipeline", {})
        # Disable batching when LoRA is active: vl_generate is sequential-only.
        do_batch = bool(pcfg.get("batch_expert_calls", True)) and not use_vllm() and not lora_path

        rounds_done = 0
        for rnd in range(1, max_rounds + 1):
            if early_exit:
                break
            rounds_done = rnd

            # ── Step A: Structure expert (VL model / image — always sequential) ──
            struct_role, struct_desc = EXPERTS[0]
            struct_prompt = (
                f"Role: {struct_role}. Task: {struct_desc}\n\n"
                f"Case data:\n{problem}\n\n"
                f"Prior blackboard (one-line summaries only — do not copy):\n{_bb_compact(public)}\n\n"
                f"{_JSON_ONLY}"
            )
            if img and Path(img).exists():
                struct_resp = vl_generate(
                    struct_prompt,
                    image_path=img,
                    lora_path=lora_path,
                    agent_role=struct_role,
                    architecture="blackboard",
                    label="expert_structure",
                    query_id=qid,
                    gene=target["gene"],
                    mutation=target["mutation"],
                )
            else:
                bu, mo = _endpoint(cfg, struct_role)
                struct_resp = _bb_call(
                    struct_prompt,
                    lora_path=lora_path,
                    base_url=bu, model=mo,
                    system_prompt=_expert_system(struct_role),
                    agent_role=struct_role, round_idx=rnd,
                    label="expert_structure", **llm_ctx,
                )
            step = _append_step(struct_role, "expert", struct_resp)
            public.append(step)
            total_tokens += struct_resp["metadata"]["total_tokens"]
            progress.echo_blackboard_step(
                target["gene"], target["mutation"], rnd, max_rounds,
                struct_role, step.get("stance_summary") or step["content"],
            )

            # ── Step B: Mechanism (sequential — Critic rubric fires immediately after) ──
            mech_role, mech_desc = EXPERTS[1]
            mech_prompt = (
                f"Role: {mech_role}. Task: {mech_desc}\n\n"
                f"Case data:\n{problem}\n\n"
                f"Prior blackboard (one-line summaries only — do not copy):\n{_bb_compact(public)}\n\n"
                f"{_JSON_ONLY}"
            )
            bu, mo = _endpoint(cfg, mech_role)
            mech_resp = _bb_call(
                mech_prompt,
                lora_path=lora_path,
                base_url=bu, model=mo,
                system_prompt=_expert_system(mech_role),
                agent_role=mech_role, round_idx=rnd,
                label="expert_mechanism", **llm_ctx,
            )
            step = _append_step(mech_role, "expert", mech_resp)
            public.append(step)
            total_tokens += mech_resp["metadata"]["total_tokens"]
            progress.echo_blackboard_step(
                target["gene"], target["mutation"], rnd, max_rounds,
                mech_role, step.get("stance_summary") or step["content"],
            )

            if rnd == 1:
                rubric_before, gaps, rtok = _invoke_critic_rubric(
                    cfg, public, llm_ctx=llm_ctx, rnd=rnd, lora_path=lora_path
                )
                total_tokens += rtok
                if rubric_before < 2:
                    bu2, mo2 = _endpoint(cfg, "Mechanism")
                    reflex = _bb_call(
                        f"Revise mechanism JSON using Critic feedback (gaps: {gaps}).\n"
                        f"Case:\n{problem}\n\nPrior:\n{_bb_compact(public)}\n\n{_JSON_ONLY}",
                        lora_path=lora_path,
                        base_url=bu2, model=mo2,
                        system_prompt=_expert_system("Mechanism") + " (reflexion pass)",
                        agent_role="Mechanism", round_idx=rnd,
                        label="mechanism_reflexion", **llm_ctx,
                    )
                    public.append(_append_step("Mechanism", "reflexion", reflex))
                    total_tokens += reflex["metadata"]["total_tokens"]
                    rubric_after, _, rtok2 = _invoke_critic_rubric(
                        cfg, public, llm_ctx=llm_ctx, rnd=rnd, lora_path=lora_path
                    )
                    total_tokens += rtok2
                    metrics.log_self_correction(
                        gene=target["gene"], mutation=target["mutation"],
                        rubric_before=rubric_before, rubric_after=rubric_after,
                    )
                else:
                    rubric_after = rubric_before

            # ── Step C: Evidence + Therapy — BATCHED (2 prompts → 1 generate call) ──
            # Both see the same blackboard snapshot (including Structure + Mechanism).
            # Evidence does not need Therapy's output and vice versa, so batching
            # is semantically safe and raises effective GPU utilisation.
            ev_role, ev_desc = EXPERTS[2]
            th_role, th_desc = EXPERTS[3]
            if do_batch:
                bb_snap = _bb_compact(public)
                batch_reqs = [
                    _build_expert_request(ev_role, ev_desc, problem, public, vus_note, cfg, llm_ctx, rnd),
                    _build_expert_request(th_role, th_desc, problem, public, vus_note, cfg, llm_ctx, rnd),
                ]
                _, mo_batch = _endpoint(cfg, ev_role)
                batch_resps = call_transformers_batch(batch_reqs, model_id=mo_batch)
                for (role, desc), resp in zip(EXPERTS[2:], batch_resps):
                    resp["metadata"].setdefault("round_idx", rnd)
                    bstep = _append_step(role, "expert", resp)
                    public.append(bstep)
                    total_tokens += resp["metadata"]["total_tokens"]
                    progress.echo_blackboard_step(
                        target["gene"], target["mutation"], rnd, max_rounds,
                        role, bstep.get("stance_summary") or bstep["content"],
                    )
            else:
                for role, desc in EXPERTS[2:]:
                    prior = _bb_compact(public)
                    prompt = (
                        f"Role: {role}. Task: {desc}\n\n"
                        f"Case data:\n{problem}\n\n"
                        f"Prior blackboard (one-line summaries only — do not copy):\n{prior}\n\n"
                        f"{_JSON_ONLY}"
                        + (vus_note if role == "Therapy" else "")
                    )
                    bu, mo = _endpoint(cfg, role)
                    llm_resp = _bb_call(
                        prompt, lora_path=lora_path, base_url=bu, model=mo,
                        system_prompt=_expert_system(role),
                        agent_role=role, round_idx=rnd,
                        label=f"expert_{role.lower()}", **llm_ctx,
                    )
                    step = _append_step(role, "expert", llm_resp)
                    public.append(step)
                    total_tokens += llm_resp["metadata"]["total_tokens"]
                    progress.echo_blackboard_step(
                        target["gene"], target["mutation"], rnd, max_rounds,
                        role, step.get("stance_summary") or step["content"],
                    )

            # ── Step D: Critic + ConflictResolver — BATCHED ──
            bb_now = _bb_compact(public)
            critic_prompt = (
                f"Check claims vs evidence. Flag hallucinated pLDDT or unsupported therapies.\n"
                f"Blackboard:\n{bb_now}\n\n{_JSON_ONLY}"
            )
            resolver_prompt = (
                f"Resolve sensitivity vs resistance conflicts by disease context.\n"
                f"Blackboard:\n{bb_now}\n\n{_JSON_ONLY}"
            )
            bu_c, mo_c = _endpoint(cfg, "Critic")
            bu_r, mo_r = _endpoint(cfg, "ConflictResolver")

            if do_batch and mo_c == mo_r:
                crit_res_resps = call_transformers_batch(
                    [
                        {"prompt": critic_prompt, "system_prompt": _critic_system(),
                         "agent_role": "Critic", "label": "critic", "max_tokens": 256},
                        {"prompt": resolver_prompt, "system_prompt": _resolver_system(),
                         "agent_role": "ConflictResolver", "label": "conflict_resolver", "max_tokens": 256},
                    ],
                    model_id=mo_c,
                )
                cr_resp, xr_resp = crit_res_resps
            else:
                cr_resp = _bb_call(
                    critic_prompt, lora_path=lora_path, base_url=bu_c, model=mo_c,
                    system_prompt=_critic_system(),
                    agent_role="Critic", round_idx=rnd, label="critic", **llm_ctx,
                )
                xr_resp = _bb_call(
                    resolver_prompt, lora_path=lora_path, base_url=bu_r, model=mo_r,
                    system_prompt=_resolver_system(),
                    agent_role="ConflictResolver", round_idx=rnd,
                    label="conflict_resolver", **llm_ctx,
                )

            cr_step = _append_step("Critic", "critique", cr_resp)
            public.append(cr_step)
            total_tokens += cr_resp["metadata"]["total_tokens"]
            progress.echo_blackboard_step(
                target["gene"], target["mutation"], rnd, max_rounds,
                "Critic", cr_step.get("stance_summary") or cr_step["content"],
            )

            xr_step = _append_step("ConflictResolver", "resolution", xr_resp)
            public.append(xr_step)
            total_tokens += xr_resp["metadata"]["total_tokens"]
            progress.echo_blackboard_step(
                target["gene"], target["mutation"], rnd, max_rounds,
                "ConflictResolver", xr_step.get("stance_summary") or xr_step["content"],
            )

            if rnd == 1 and _round1_consensus(
                xr_step.get("stance_summary") or cr_resp.get("content", ""),
                public[-1].get("stance_summary") or xr_resp.get("content", ""),
            ):
                early_exit = True
                progress.echo_blackboard_step(
                    target["gene"], target["mutation"], rnd, max_rounds,
                    "system", "consensus reached", early_exit=True,
                )

        bu, mo = _endpoint(cfg, "Decider")
        # Concrete one-shot example keeps the model on the right schema.
        decider_prompt = (
            f"Produce the final decision as JSON with EXACTLY this structure "
            f"(example: {{\"mechanism\": \"EGFR L858R activates kinase domain\", "
            f"\"therapy\": {{\"sensitivity\": [\"Osimertinib\"], \"resistance\": [], \"context\": \"NSCLC\"}}, "
            f"\"confidence\": \"0.9\", \"next_best_action\": \"standard_of_care\"}}):\n"
            f'{{"mechanism": "...",\n'
            f' "therapy": {{"sensitivity": ["drug1", ...], "resistance": ["drug1", ...], "context": "..."}},\n'
            f' "confidence": "0.0-1.0",\n'
            f' "next_best_action": "standard_of_care|tumor_board|clinical_trial|structural_rescue"}}\n\n'
            f"Key claims (mechanism · therapy · conflict resolution):\n{_decider_context(public)}"
            + vus_note
        )
        dr = _bb_call(
            decider_prompt,
            lora_path=lora_path,
            base_url=bu,
            model=mo,
            system_prompt=(
                "You are the Decider. Return valid JSON only. "
                "Do NOT wrap the JSON in a 'reasoning' key. "
                "Do NOT use markdown fences. Output one JSON object."
            ),
            agent_role="Decider",
            round_idx=rounds_done + 1,
            label="decider",
            temperature=0,
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

        # Normalise nesting / string-therapy / mechanism-dict before extraction.
        parsed = _normalize_decider_output(parsed, public)

        # Repair pass: if therapy lists are still empty, re-prompt with the
        # Therapy expert's full detail text and a minimal schema.
        if not parsed.get("therapy", {}).get("sensitivity") and not parsed.get("therapy", {}).get("resistance"):
            therapy_detail = next(
                (s.get("detail") or s.get("content") or "" for s in reversed(public) if s.get("agent") == "Therapy"),
                "",
            )
            if therapy_detail:
                rp = _bb_call(
                    f"Based only on this therapy analysis:\n{therapy_detail[:800]}\n\n"
                    f'Return ONLY this JSON with no wrapper and no markdown:\n'
                    f'{{"sensitivity": ["drug_name"], "resistance": [], "context": "..."}}',
                    lora_path=lora_path,
                    base_url=bu,
                    model=mo,
                    system_prompt="You are the Decider. Return JSON only. No markdown. No 'reasoning' key.",
                    agent_role="Decider",
                    round_idx=rounds_done + 2,
                    label="decider_repair",
                    temperature=0,
                    **llm_ctx,
                )
                rp_parsed = parse_reasoning_json(rp["content"])
                if isinstance(rp_parsed, dict) and (
                    rp_parsed.get("sensitivity") or rp_parsed.get("resistance")
                ):
                    parsed.setdefault("therapy", {})["sensitivity"] = list(rp_parsed.get("sensitivity") or [])
                    parsed["therapy"]["resistance"] = list(rp_parsed.get("resistance") or [])
                    parsed["therapy"]["context"] = rp_parsed.get("context", "")
                total_tokens += rp["metadata"]["total_tokens"]

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
