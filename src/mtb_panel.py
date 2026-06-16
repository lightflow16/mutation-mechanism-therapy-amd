"""Molecular Tumor Board panel formatter from blackboard traces."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _stance_summary(content: str, limit: int = 120) -> str:
    text = (content or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _biochemical_claim(agent: str, content: str) -> str:
    low = content.lower()
    if agent == "Structure":
        for key in ("plddt", "region", "loop", "domain", "residue"):
            if key in low:
                return _stance_summary(content, 80)
    if agent == "Mechanism":
        for key in ("activat", "signal", "pathway", "kinase", "binding", "loss"):
            if key in low:
                return _stance_summary(content, 80)
    if agent == "Evidence":
        for key in ("civic", "clinvar", "sensitivity", "resistance", "fda"):
            if key in low:
                return _stance_summary(content, 80)
    if agent == "Therapy":
        for key in ("osimertinib", "alpelisib", "gefitinib", "erlotinib", "inhibitor"):
            if key in low:
                return _stance_summary(content, 80)
    if agent in ("Critic", "ConflictResolver", "Decider"):
        return _stance_summary(content, 80)
    return _stance_summary(content, 60)


def format_vus_summary(tr: dict, routing: dict) -> str:
    """One-screen abstention card for VUS demo."""
    therapy = tr.get("therapy") or {}
    if isinstance(tr.get("target_reasoning"), dict):
        therapy = tr["target_reasoning"].get("therapy") or therapy
    lines = [
        "=== VUS Summary ===",
        f"classification: {routing.get('classification', tr.get('classification', 'unknown'))}",
        f"evidence_tier: {routing.get('evidence_tier', tr.get('evidence_tier', 'unknown'))}",
        f"allow_confident_therapy: {routing.get('allow_confident_therapy', False)}",
        f"mechanism_hypothesis: {(tr.get('mechanism_hypothesis') or tr.get('mechanism') or '')[:200]}",
        f"therapy.sensitivity: {therapy.get('sensitivity', [])}",
        f"recommendation_status: {therapy.get('recommendation_status', 'unknown')}",
        f"next_best_action: {tr.get('next_best_action', 'tumor_board')}",
    ]
    return "\n".join(lines)


def format_mtb_panel(trace: dict | Path | str) -> str:
    """Render Agent | Stance | Key Biochemical Claim table from a blackboard trace."""
    if isinstance(trace, (str, Path)):
        trace = json.loads(Path(trace).read_text())

    reasoning = trace.get("reasoning", {})
    bb = reasoning.get("blackboard_trace") or []
    gene = trace.get("target", {}).get("gene") or trace.get("gene", "")
    mutation = trace.get("target", {}).get("mutation") or trace.get("mutation", "")
    header = f"MTB Panel — {gene} {mutation} (blackboard)"
    lines = [
        header,
        "-" * len(header),
        f"{'Agent':<18} | {'Stance Summary':<42} | Key Biochemical Claim",
        "-" * 90,
    ]
    seen: set[str] = set()
    for entry in bb:
        agent = entry.get("agent", "?")
        typ = entry.get("type", "")
        key = f"{agent}:{typ}"
        if key in seen and typ == "expert":
            continue
        seen.add(key)
        content = entry.get("content", "")
        role = f"{agent} ({typ})" if typ else agent
        lines.append(
            f"{role:<18} | {_stance_summary(content, 42):<42} | {_biochemical_claim(agent, content)}"
        )
    return "\n".join(lines)


def mtb_panel_dict(trace: dict | Path | str) -> dict[str, Any]:
    """Structured MTB panel for embedding in comparison JSON."""
    if isinstance(trace, (str, Path)):
        trace = json.loads(Path(trace).read_text())
    rows = []
    for entry in trace.get("reasoning", {}).get("blackboard_trace") or []:
        agent = entry.get("agent", "")
        content = entry.get("content", "")
        rows.append(
            {
                "agent": agent,
                "type": entry.get("type", ""),
                "stance_summary": _stance_summary(content, 120),
                "biochemical_claim": _biochemical_claim(agent, content),
            }
        )
    target = trace.get("target", {})
    return {
        "gene": target.get("gene", ""),
        "mutation": target.get("mutation", ""),
        "rows": rows,
        "formatted": format_mtb_panel(trace),
    }
