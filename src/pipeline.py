"""End-to-end pipeline orchestrator with router + cached traces."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from src import metrics
from src.config import get_target, load_config, metrics_dir, setup_env, shared_dir
from src.cot import run_cot
from src.debate import run_debate
from src.evidence import gather_evidence
from src.mas import run_blackboard
from src.plm_prior import compute_plm_llr
from src.reason import reason_single
from src.rescue import interpret_rescue, run_rescue
from src.route_planner import route_with_planner
from src.structure import analyze_target
from src.variant_router import route_variant

try:
    from src import progress
except ImportError:
    progress = None  # type: ignore

Architecture = Literal["single", "cot", "blackboard", "debate"]


def _flush_gpu_cache() -> None:
    """Release fragmented VRAM between heavy pipeline phases."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


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


def load_architecture_trace(gene: str, mutation: str, architecture: str) -> dict | None:
    """Load a saved trace from repo cache or a prior live run under metrics/."""
    cached = load_cached_trace(gene, mutation, architecture)
    if cached:
        return cached
    live_trace = metrics_dir() / f"trace_{gene}_{mutation}_{architecture}.json"
    if live_trace.exists():
        return json.loads(live_trace.read_text())
    return None


def _run_reasoning(
    architecture: Architecture,
    target: dict,
    structure: dict,
    evidence: list[dict],
    *,
    lora_path: str | None = None,
    image_path: str | None = None,
) -> dict[str, Any]:
    img = image_path or structure.get("structure_image_path")
    if architecture == "debate":
        return run_debate(target, structure, evidence)
    if architecture == "blackboard":
        return run_blackboard(target, structure, evidence, image_path=img)
    if architecture == "cot":
        return run_cot(target, structure, evidence, image_path=img)
    return reason_single(target, structure, evidence, lora_path=lora_path, image_path=img)


def _unwrap_target_reasoning(response_dict: dict) -> dict:
    """Defensively unwrap nested target_reasoning structures caused by CoT parser schemas.

    Handles three common Decider/CoT output shapes:
      1. {"target_reasoning": {"therapy": ...}}          — standard CoT schema
      2. {"target_reasoning": {"target_reasoning": ...}} — double-nested CoT
      3. {"reasoning": {"therapy": ...}}                 — Decider 7B schema
    """
    if not response_dict:
        return {}
    if "target_reasoning" in response_dict:
        inner = response_dict["target_reasoning"]
        if isinstance(inner, dict) and "target_reasoning" in inner:
            nested = inner["target_reasoning"]
            return nested if isinstance(nested, dict) else inner
        return inner if isinstance(inner, dict) else {}
    # Decider (Qwen-7B) uses {"reasoning": {"mechanism": ..., "therapy": ...}}
    if "reasoning" in response_dict and isinstance(response_dict["reasoning"], dict):
        return response_dict["reasoning"]
    return response_dict


def extract_target_reasoning(result: dict) -> dict:
    if not result:
        return {}
    r = result.get("reasoning", {})
    if not isinstance(r, dict):
        return {}
    if "target_reasoning" in r:
        return _unwrap_target_reasoning(r["target_reasoning"])
    return _unwrap_target_reasoning(r)


def extract_therapies_from_reasoning(reasoning: dict) -> tuple[list[str], list[str]]:
    tr = reasoning.get("therapy") or reasoning
    if isinstance(tr, dict):
        return list(tr.get("sensitivity") or []), list(tr.get("resistance") or [])
    return [], []


def summarize_rescue(rescue: dict | None) -> dict | None:
    if not rescue:
        return None
    return {
        "mutant_ddg_kcal_mol": rescue.get("mutant_ddg_kcal_mol"),
        "destabilizing": rescue.get("destabilizing"),
        "fold_method": rescue.get("fold_method"),
        "boltz_pdb": rescue.get("boltz_pdb"),
        "esmfold_pdb": rescue.get("esmfold_pdb"),
    }


def compare_architecture_results(
    gene: str,
    mutation: str,
    by_architecture: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Side-by-side comparison of single / cot / blackboard on the same mutation."""
    rows: list[dict[str, Any]] = []
    routes: dict[str, str | None] = {}
    sens_by: dict[str, list[str]] = {}
    res_by: dict[str, list[str]] = {}
    mech_by: dict[str, str] = {}
    conf_by: dict[str, Any] = {}
    tokens_by: dict[str, Any] = {}

    for arch, result in by_architecture.items():
        tr = extract_target_reasoning(result)
        sens, res = extract_therapies_from_reasoning(tr)
        routes[arch] = result.get("route")
        sens_by[arch] = sens
        res_by[arch] = res
        mech_by[arch] = (tr.get("mechanism") or "")[:500]
        conf_by[arch] = tr.get("confidence")
        reasoning = result.get("reasoning", {})
        tokens_by[arch] = reasoning.get("total_tokens", 0)
        rows.append(
            {
                "architecture": arch,
                "route": routes[arch],
                "therapy_sensitivity": sens,
                "therapy_resistance": res,
                "mechanism_preview": mech_by[arch][:200],
                "confidence": conf_by[arch],
                "total_tokens": tokens_by[arch],
            }
        )

    sens_sets = [set(v) for v in sens_by.values()]
    res_sets = [set(v) for v in res_by.values()]
    sens_overlap = set.intersection(*sens_sets) if sens_sets else set()
    sens_union = set.union(*sens_sets) if sens_sets else set()
    res_overlap = set.intersection(*res_sets) if res_sets else set()
    res_union = set.union(*res_sets) if res_sets else set()

    rescue = None
    for result in by_architecture.values():
        if result.get("rescue"):
            rescue = summarize_rescue(result["rescue"])
            break

    return {
        "gene": gene,
        "mutation": mutation,
        "route_by_architecture": routes,
        "route_agreement": len(set(routes.values())) <= 1,
        "therapy_sensitivity_by_architecture": sens_by,
        "therapy_resistance_by_architecture": res_by,
        "therapy_sensitivity_overlap": sorted(sens_overlap),
        "therapy_sensitivity_union": sorted(sens_union),
        "therapy_sensitivity_agreement": len(sens_union) == 0 or sens_overlap == sens_union,
        "therapy_resistance_overlap": sorted(res_overlap),
        "therapy_resistance_union": sorted(res_union),
        "mechanism_by_architecture": mech_by,
        "confidence_by_architecture": conf_by,
        "total_tokens_by_architecture": tokens_by,
        "rescue": rescue,
        "rows": rows,
    }


def format_comparison_report(comparison: dict[str, Any]) -> str:
    lines = [
        f"=== {comparison['gene']} {comparison['mutation']} — architecture comparison ===",
        f"Route agreement: {comparison['route_agreement']} | routes: {comparison['route_by_architecture']}",
        f"Therapy sensitivity overlap: {comparison['therapy_sensitivity_overlap']}",
        f"Therapy sensitivity union: {comparison['therapy_sensitivity_union']}",
    ]
    for row in comparison["rows"]:
        lines.append(
            f"  [{row['architecture']}] route={row['route']} "
            f"sens={row['therapy_sensitivity']} res={row['therapy_resistance']} "
            f"conf={row['confidence']} tokens={row['total_tokens']}"
        )
    if comparison.get("rescue"):
        lines.append(f"  rescue: {comparison['rescue']}")
    return "\n".join(lines)


def _embed_mtb_panels(comparison: dict[str, Any], by_architecture: dict[str, dict]) -> None:
    try:
        from src.mtb_panel import mtb_panel_dict

        bb = by_architecture.get("blackboard")
        if bb:
            comparison["mtb_panel"] = mtb_panel_dict(bb)
    except Exception:
        pass


def _write_mutation_comparison(
    gene: str,
    mutation: str,
    comparison: dict[str, Any],
    by_architecture: dict[str, dict[str, Any]],
) -> None:
    out_dir = metrics_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"comparison_{gene}_{mutation}.json").write_text(
        json.dumps(comparison, indent=2, default=str)
    )
    (out_dir / f"comparison_{gene}_{mutation}_full.json").write_text(
        json.dumps(by_architecture, indent=2, default=str)
    )


def run_mutation_comparison(
    gene: str,
    mutation: str,
    *,
    architectures: list[Architecture] | None = None,
    lora_path: str | None = None,
    live_evidence: bool | None = None,
    use_cached_trace: bool = False,
    run_rescue_branch: bool | None = None,
) -> dict[str, Any]:
    """Run single, cot, and blackboard on one mutation (shared structure/evidence) and compare."""
    setup_env()
    metrics.set_metrics_dir(str(metrics_dir()))
    cfg = load_config()
    pcfg = cfg.get("pipeline", {})
    if architectures is None:
        architectures = pcfg.get("architectures", ["single", "cot", "blackboard"])
    if live_evidence is None:
        live_evidence = bool(pcfg.get("live_evidence", True))

    target = get_target(cfg, gene, mutation)
    cache = shared_dir(cfg) / "structures"
    by_arch: dict[str, dict[str, Any]] = {}
    missing: list[str] = []

    if use_cached_trace and not live_evidence:
        for arch in architectures:
            cached = load_architecture_trace(gene, mutation, arch)
            if cached:
                by_arch[arch] = cached
            else:
                missing.append(arch)
        if by_arch:
            comparison = compare_architecture_results(gene, mutation, by_arch)
            _embed_mtb_panels(comparison, by_arch)
            if missing:
                comparison["missing_architectures"] = missing
                comparison["note"] = (
                    "Missing traces for "
                    + ", ".join(missing)
                    + ". Run run_all_modes() live first; eval does not load LLMs."
                )
            _write_mutation_comparison(gene, mutation, comparison, by_arch)
            return {"architectures": by_arch, "comparison": comparison, "missing_architectures": missing}
        raise FileNotFoundError(
            f"No cached traces for {gene} {mutation}. "
            f"Run run_all_modes() live, or add data/traces/{gene}_{mutation}_{{single,cot,blackboard}}.json"
        )

    structure: dict[str, Any] | None = None
    evidence: list[dict] | None = None
    route: str | None = None

    with metrics.SysSampler(f"comparison_{gene}_{mutation}"):
        case_idx = 0
        for arch in architectures:
            with metrics.phase(f"comparison_{gene}_{mutation}_{arch}"):
                if structure is None:
                    structure = analyze_target(target, cache)
                    evidence = gather_evidence(target, live=live_evidence)
                    routing = route_variant(target, structure, evidence)
                    target = {**target, **routing}
                    plm = compute_plm_llr(
                        target["uniprot"], target["residue"], target["wt_aa"], target["mut_aa"]
                    )
                    if plm.get("plm_llr") is not None:
                        structure["plm_llr"] = plm["plm_llr"]
                        structure["plm_perplexity_band"] = plm.get("plm_perplexity_band")
                    rule_route = route_target(target, structure, evidence)
                    route = rule_route
                    pcfg = load_config().get("pipeline", {})
                    if pcfg.get("use_llm_router", False):
                        planned = route_with_planner(target, structure, evidence, rule_route=rule_route)
                        route = planned["route"]
                        target["route_planner"] = planned
                    else:
                        route = routing.get("pathway") or rule_route
                if progress:
                    progress.banner(f"CASE — {gene} {mutation} | route={route}")
                if progress:
                    progress.log("pipeline", f"architecture: {arch}", gene=gene, mutation=mutation)

                _flush_gpu_cache()
                result: dict[str, Any] = {
                    "target": target,
                    "structure": {
                        k: v for k, v in structure.items()
                        if k not in ("render_html",)
                    },
                    "evidence": evidence,
                    "route": route,
                    "architecture": arch,
                    "variant_routing": {
                        k: target.get(k)
                        for k in (
                            "evidence_tier", "classification", "vus_branch",
                            "allow_confident_therapy",
                        )
                    },
                }
                result["reasoning"] = _run_reasoning(
                    arch,
                    target,
                    structure,
                    evidence,
                    lora_path=lora_path if arch == "single" else None,
                    image_path=structure.get("structure_image_path"),
                )
                by_arch[arch] = result
                trace_path = metrics_dir() / f"trace_{gene}_{mutation}_{arch}.json"
                trace_path.write_text(json.dumps(result, indent=2, default=str))

        assert structure is not None and route is not None
        do_rescue = run_rescue_branch if run_rescue_branch is not None else route == "structural_rescue"
        if do_rescue:
            _flush_gpu_cache()
            rescue = run_rescue(target, Path(structure["pdb_path"]))
            rescue["interpreter"] = interpret_rescue(target, rescue)
            for arch in by_arch:
                by_arch[arch]["rescue"] = rescue
                trace_path = metrics_dir() / f"trace_{gene}_{mutation}_{arch}.json"
                trace_path.write_text(json.dumps(by_arch[arch], indent=2, default=str))

    comparison = compare_architecture_results(gene, mutation, by_arch)
    _embed_mtb_panels(comparison, by_arch)
    _write_mutation_comparison(gene, mutation, comparison, by_arch)
    metrics.aggregate_ablation()
    return {"architectures": by_arch, "comparison": comparison}


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
            routing = route_variant(target, structure, evidence)
            target = {**target, **routing}
            route = routing.get("pathway") or route_target(target, structure, evidence)

            result: dict[str, Any] = {
                "target": target,
                "structure": {k: v for k, v in structure.items() if k != "render_html"},
                "evidence": evidence,
                "route": route,
            }

            _flush_gpu_cache()
            img = structure.get("structure_image_path")
            if architecture == "debate":
                result["reasoning"] = run_debate(target, structure, evidence)
            elif architecture == "blackboard":
                result["reasoning"] = run_blackboard(target, structure, evidence, image_path=img)
            elif architecture == "cot":
                result["reasoning"] = run_cot(target, structure, evidence, image_path=img)
            else:
                result["reasoning"] = reason_single(
                    target, structure, evidence, lora_path=lora_path, image_path=img
                )

            do_rescue = run_rescue_branch if run_rescue_branch is not None else route == "structural_rescue"
            if do_rescue:
                _flush_gpu_cache()
                pdb = Path(structure["pdb_path"])
                result["rescue"] = run_rescue(target, pdb)

            trace_path = metrics_dir() / f"trace_{gene}_{mutation}_{architecture}.json"
            trace_path.write_text(json.dumps(result, indent=2, default=str))
            metrics.aggregate_ablation()
            return result


def run_all_modes(
    *,
    lora_path: str | None = None,
    live_evidence: bool | None = None,
    use_cached_baseline: bool | None = None,
) -> dict[str, Any]:
    """Run every architecture on every demo case and compare results per mutation."""
    cfg = load_config()
    pcfg = cfg.get("pipeline", {})
    cases = [tuple(x) for x in pcfg.get("demo_cases", [])]
    architectures: list[Architecture] = list(pcfg.get("architectures", ["single", "cot", "blackboard"]))
    debate_cases = {tuple(x) for x in pcfg.get("debate_cases", [])}
    if live_evidence is None:
        live_evidence = bool(pcfg.get("live_evidence", True))
    if use_cached_baseline is None:
        use_cached_baseline = bool(pcfg.get("use_cached_baseline", True))

    results: dict[str, Any] = {"cached": {}, "live": {}, "comparisons": {}}
    if use_cached_baseline:
        for gene, mut in cases:
            key = f"{gene}_{mut}"
            cached_run = run_mutation_comparison(
                gene,
                mut,
                architectures=architectures,
                use_cached_trace=True,
                live_evidence=False,
            )
            results["cached"][key] = cached_run
            results["comparisons"][f"cached_{key}"] = cached_run["comparison"]

    live_comparisons: dict[str, Any] = {}
    for gene, mut in cases:
        key = f"{gene}_{mut}"
        kwargs: dict[str, Any] = dict(
            architectures=architectures,
            use_cached_trace=False,
            live_evidence=live_evidence,
        )
        if lora_path:
            kwargs["lora_path"] = lora_path
        if gene.upper() == "TP53" and mut.upper() == "R175H":
            kwargs["run_rescue_branch"] = True
        live_run = run_mutation_comparison(gene, mut, **kwargs)
        if (gene, mut) in debate_cases:
            try:
                debate_run = run_mutation_comparison(
                    gene, mut,
                    architectures=["debate"],
                    use_cached_trace=use_cached_baseline and not live_evidence,
                    live_evidence=live_evidence,
                )
                if debate_run["architectures"].get("debate"):
                    live_run["architectures"]["debate"] = debate_run["architectures"]["debate"]
                    # Re-compute and re-write the merged comparison so the debate
                    # sub-run does not overwrite the single/cot/blackboard results.
                    merged_comparison = compare_architecture_results(gene, mut, live_run["architectures"])
                    _embed_mtb_panels(merged_comparison, live_run["architectures"])
                    _write_mutation_comparison(gene, mut, merged_comparison, live_run["architectures"])
                    live_run["comparison"] = merged_comparison
            except FileNotFoundError:
                pass
        results["live"][key] = live_run
        live_comparisons[key] = live_run["comparison"]
        results["comparisons"][key] = live_run["comparison"]
        for arch, payload in live_run["architectures"].items():
            results["live"][f"{gene}_{mut}_{arch}"] = payload

    if live_comparisons:
        out_path = metrics_dir() / "architecture_comparison.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(live_comparisons, indent=2, default=str))

    try:
        from src.metrics_bundle import write_platform_summary

        write_platform_summary()
    except Exception:
        pass

    return results
