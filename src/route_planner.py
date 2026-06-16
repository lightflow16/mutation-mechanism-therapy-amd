"""LLM route planner — optional override for rule-based route_target()."""
from __future__ import annotations

import json
from typing import Any

from src import metrics
from src.llm_client import call_llm


def route_with_planner(
    target: dict,
    structure: dict,
    evidence: list[dict],
    *,
    rule_route: str,
) -> dict[str, Any]:
    """Call Planner LLM to choose inhibitor_rag vs structural_rescue."""
    summary = {
        "gene": target.get("gene"),
        "mutation": target.get("mutation"),
        "class": target.get("class"),
        "pathway_yaml": target.get("pathway"),
        "pLDDT_at_residue": structure.get("pLDDT_at_residue"),
        "region": structure.get("region"),
        "evidence_tier": target.get("evidence_tier"),
        "n_evidence": len(evidence),
        "rule_route": rule_route,
    }
    prompt = (
        "Choose route: inhibitor_rag OR structural_rescue.\n"
        f"Context: {json.dumps(summary)}\n"
        'Reply JSON: {"route": "...", "rationale": "..."}'
    )
    resp = call_llm(
        prompt,
        system_prompt="You are RoutePlanner for precision oncology pipelines.",
        agent_role="RoutePlanner",
        architecture="router",
        label="route_planner",
        query_id=f"{target.get('gene')}_{target.get('mutation')}",
        gene=target.get("gene", ""),
        mutation=target.get("mutation", ""),
        max_tokens=256,
    )
    route = rule_route
    rationale = resp["content"]
    try:
        start, end = resp["content"].find("{"), resp["content"].rfind("}") + 1
        if start >= 0:
            obj = json.loads(resp["content"][start:end])
            route = obj.get("route") or rule_route
            rationale = obj.get("rationale") or rationale
    except json.JSONDecodeError:
        pass
    if route not in ("inhibitor_rag", "structural_rescue"):
        route = rule_route
    return {
        "route": route,
        "rule_route": rule_route,
        "route_agreement": route == rule_route,
        "rationale": rationale[:500],
    }
