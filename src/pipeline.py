"""End-to-end pipeline orchestrator with router + cached traces."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from src import metrics
from src.config import get_target, load_config, metrics_dir, setup_env, shared_dir
from src.cot import run_cot
from src.evidence import gather_evidence
from src.mas import run_blackboard
from src.reason import reason_single
from src.rescue import run_rescue
from src.structure import analyze_target

Architecture = Literal["single", "cot", "blackboard"]


def route_target(target: dict, structure: dict, evidence: list[dict]) -> str:
    """Planner-style route: inhibitor_rag vs structural_rescue."""
    if target.get("pathway") == "structural_rescue":
        return "structural_rescue"
    if target.get("class") == "TUMOR_SUPPRESSOR_LOF":
        return "structural_rescue"
    therapies = [e.get("therapies", "") for e in evidence if e.get("therapies")]
    if not any(t.strip() for t in therapies):
        return "structural_rescue"
    return "inhibitor_rag"


def load_cached_trace(gene: str, mutation: str, architecture: str) -> dict | None:
    p = Path(__file__).resolve().parents[1] / "data" / "traces" / f"{gene}_{mutation}_{architecture}.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def run_case(
    gene: str,
    mutation: str,
    *,
    architecture: Architecture = "single",
    live_evidence: bool = False,
    run_rescue_branch: bool | None = None,
    use_cached_trace: bool = True,
    lora_path: str | None = None,
) -> dict[str, Any]:
    setup_env()
    metrics.set_metrics_dir(str(metrics_dir()))
    cfg = load_config()
    target = get_target(cfg, gene, mutation)
    cache = shared_dir(cfg) / "structures"

    cached = load_cached_trace(gene, mutation, architecture) if use_cached_trace else None
    if cached and not live_evidence:
        return cached

    with metrics.SysSampler(f"pipeline_{gene}_{mutation}_{architecture}"):
        with metrics.phase(f"pipeline_{gene}_{mutation}_{architecture}"):
            structure = analyze_target(target, cache)
            evidence = gather_evidence(target, live=live_evidence)
            route = route_target(target, structure, evidence)

            result: dict[str, Any] = {
                "target": target,
                "structure": {k: v for k, v in structure.items() if k != "render_html"},
                "evidence": evidence,
                "route": route,
            }

            if architecture == "blackboard":
                result["reasoning"] = run_blackboard(target, structure, evidence)
            elif architecture == "cot":
                result["reasoning"] = run_cot(target, structure, evidence)
            else:
                result["reasoning"] = reason_single(
                    target, structure, evidence, lora_path=lora_path
                )

            do_rescue = run_rescue_branch if run_rescue_branch is not None else route == "structural_rescue"
            if do_rescue:
                pdb = Path(structure["pdb_path"])
                result["rescue"] = run_rescue(target, pdb)

            trace_path = metrics_dir() / f"trace_{gene}_{mutation}_{architecture}.json"
            trace_path.write_text(json.dumps(result, indent=2, default=str))
            metrics.aggregate_ablation()
            return result
