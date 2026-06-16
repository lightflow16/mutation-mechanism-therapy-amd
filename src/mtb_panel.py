"""Molecular Tumor Board panel formatter from blackboard traces."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_ECHO_HEADER = re.compile(r"^#+\s*summary of evidence", re.I)


def _stance_summary(content: str, limit: int = 120) -> str:
    text = (content or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _normalize_echo_key(text: str, n: int = 100) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip().lower())
    t = re.sub(r"^#+\s*", "", t)
    return t[:n]


def _is_echo(content: str, seen_keys: set[str]) -> bool:
    key = _normalize_echo_key(content)
    if not key or len(key) < 40:
        return False
    if key in seen_keys:
        return True
    if _ECHO_HEADER.match(content.strip()):
        return key in seen_keys
    return False


def _extract_role_snippet(agent: str, content: str) -> str:
    """Best-effort unique snippet from legacy (pre-structured) traces."""
    if not content:
        return ""
    text = content.strip()
    if _ECHO_HEADER.match(text):
        text = re.sub(r"^#+\s*Summary of Evidence[^\n]*\n+", "", text, flags=re.I)
        text = re.sub(r"^#+\s*Mutation Details[^\n]*\n+", "", text, flags=re.I)

    low = text.lower()
    if agent == "Critic":
        for pat in (
            r"(no (signs of )?hallucin[^.]{0,80}\.)",
            r"(hallucin[^.]{0,80}\.)",
            r"(unsupported[^.]{0,80}\.)",
            r"(pass[^.]{0,60}\.)",
        ):
            m = re.search(pat, low, re.I)
            if m:
                return _stance_summary(text[m.start() : m.end()], 100)
    if agent == "ConflictResolver":
        for pat in (
            r"(resolv[^.]{0,80}\.)",
            r"(sensitiv[^.]{0,80}resist[^.]{0,40}\.)",
            r"(context-dependent[^.]{0,80}\.)",
        ):
            m = re.search(pat, low, re.I)
            if m:
                return _stance_summary(text[m.start() : m.end()], 100)
    if agent == "Therapy":
        for drug in ("alpelisib", "osimertinib", "gefitinib", "erlotinib", "afatinib"):
            if drug in low:
                idx = low.find(drug)
                return _stance_summary(text[max(0, idx - 20) : idx + 80], 100)
    if agent == "Mechanism":
        for kw in ("activation loop", "e545k", "constitutive", "pathway", "p85"):
            if kw in low:
                idx = low.find(kw)
                return _stance_summary(text[max(0, idx - 10) : idx + 90], 100)
    if agent == "Evidence":
        for kw in ("civic", "clinvar", "level a", "sensitivity", "resistance"):
            if kw in low:
                idx = low.find(kw)
                return _stance_summary(text[max(0, idx - 5) : idx + 85], 100)
    if agent == "Structure":
        for kw in ("plddt", "94.1", "activation loop", "residue 545"):
            if kw in low:
                idx = low.find(kw)
                return _stance_summary(text[max(0, idx - 5) : idx + 85], 100)
    return _stance_summary(text, 80)


def _biochemical_claim(agent: str, content: str, entry: dict) -> str:
    if entry.get("key_claim"):
        return _stance_summary(str(entry["key_claim"]), 80)
    return _extract_role_snippet(agent, content) or _stance_summary(content, 80)


def _display_stance(entry: dict) -> str:
    if entry.get("stance_summary"):
        return _stance_summary(str(entry["stance_summary"]), 42)
    return _extract_role_snippet(entry.get("agent", ""), entry.get("content", ""))[:42]


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
    seen_echo: set[str] = set()
    seen_panel: set[tuple[str, str]] = set()

    for entry in bb:
        agent = entry.get("agent", "?")
        typ = entry.get("type", "")
        content = entry.get("content", "")

        panel_key = (agent, typ)
        if typ in ("expert", "critique", "resolution", "plan"):
            if panel_key in seen_panel:
                continue
            seen_panel.add(panel_key)
        elif typ == "decision" and panel_key in seen_panel:
            continue
        else:
            seen_panel.add(panel_key)

        echo_key = _normalize_echo_key(entry.get("stance_summary") or content)
        if _is_echo(content, seen_echo) and not entry.get("stance_summary"):
            stance = f"[echo trimmed] {_extract_role_snippet(agent, content)[:36]}"
            claim = _biochemical_claim(agent, content, entry)
        else:
            stance = _display_stance(entry)
            claim = _biochemical_claim(agent, content, entry)
        if echo_key:
            seen_echo.add(echo_key)

        role = f"{agent} ({typ})" if typ else agent
        lines.append(f"{role:<18} | {stance:<42} | {claim}")
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
                "stance_summary": entry.get("stance_summary") or _display_stance(entry),
                "biochemical_claim": entry.get("key_claim") or _biochemical_claim(agent, content, entry),
            }
        )
    target = trace.get("target", {})
    return {
        "gene": target.get("gene", ""),
        "mutation": target.get("mutation", ""),
        "rows": rows,
        "formatted": format_mtb_panel(trace),
    }
