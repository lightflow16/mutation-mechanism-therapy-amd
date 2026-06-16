"""Ablation: compare single / CoT / blackboard per mutation; Therapy F1 + direction acc."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import csv
import json

from src import metrics
from src.config import load_config, metrics_dir
from src.pipeline import (
    compare_architecture_results,
    extract_target_reasoning,
    extract_therapies_from_reasoning,
    run_mutation_comparison,
)

ROOT = Path(__file__).resolve().parents[1]


def therapy_f1(pred: list[str], gold: list[str]) -> float:
    def norm(xs):
        aliases = {
            "gefitinib": "gefitinib",
            "erlotinib": "erlotinib",
            "afatinib": "afatinib",
            "osimertinib": "osimertinib",
            "alpelisib": "alpelisib",
        }
        out = set()
        for x in xs:
            k = x.lower().strip().split()[0]
            out.add(aliases.get(k, k))
        return out

    p, g = norm(pred), norm(gold)
    if not g:
        return 1.0 if not p else 0.0
    prec = len(p & g) / len(p) if p else 0.0
    rec = len(p & g) / len(g)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def direction_accuracy(pred: dict, gold: dict) -> float:
    ps = set((pred.get("therapy") or {}).get("sensitivity") or [])
    pr = set((pred.get("therapy") or {}).get("resistance") or [])
    gs = set((gold.get("therapy") or {}).get("sensitivity") or [])
    gr = set((gold.get("therapy") or {}).get("resistance") or [])
    ok = (bool(ps & gs) or not gs) and (bool(pr & gr) or not gr)
    return 1.0 if ok else 0.0


def gold_case(gene: str, mutation: str) -> dict:
    p = ROOT / "data" / "cases" / f"{gene}_{mutation}.json"
    return json.loads(p.read_text())


def main():
    cfg = load_config()
    metrics.set_metrics_dir(str(metrics_dir()))
    pcfg = cfg.get("pipeline", {})
    cases = [tuple(x) for x in pcfg.get("demo_cases", [])]
    architectures = tuple(pcfg.get("architectures", ["single", "cot", "blackboard"]))

    rows = []
    comparisons = {}
    for gene, mut in cases:
        gold = gold_case(gene, mut)
        gold_reasoning = gold["target_reasoning"]
        gold_sens, gold_res = extract_therapies_from_reasoning(gold_reasoning)

        run: dict = {"architectures": {}}
        with metrics.phase(f"eval_{gene}_{mut}_compare"):
            try:
                run = run_mutation_comparison(
                    gene,
                    mut,
                    architectures=list(architectures),
                    live_evidence=False,
                    use_cached_trace=True,
                )
                comparison = run["comparison"]
            except Exception as exc:
                comparison = compare_architecture_results(gene, mut, {})
                comparison["error"] = str(exc)

        comparisons[f"{gene}_{mut}"] = comparison

        for arch in architectures:
            result = run.get("architectures", {}).get(arch)
            if not result:
                rows.append(
                    {
                        "gene": gene,
                        "mutation": mut,
                        "architecture": arch,
                        "route": None,
                        "therapy_f1": None,
                        "direction_acc": None,
                        "route_agreement": comparison.get("route_agreement"),
                        "therapy_sensitivity_agreement": comparison.get("therapy_sensitivity_agreement"),
                        "therapy_sensitivity_overlap": ",".join(
                            comparison.get("therapy_sensitivity_overlap") or []
                        ),
                        "status": "missing_trace",
                    }
                )
                continue
            reasoning = extract_target_reasoning(result)
            pred_sens, pred_res = extract_therapies_from_reasoning(reasoning)
            f1 = therapy_f1(pred_sens + pred_res, gold_sens + gold_res)
            dacc = direction_accuracy(reasoning, gold_reasoning)
            rows.append(
                {
                    "gene": gene,
                    "mutation": mut,
                    "architecture": arch,
                    "route": result.get("route"),
                    "therapy_f1": round(f1, 3),
                    "direction_acc": round(dacc, 3),
                    "route_agreement": comparison.get("route_agreement"),
                    "therapy_sensitivity_agreement": comparison.get("therapy_sensitivity_agreement"),
                    "therapy_sensitivity_overlap": ",".join(
                        comparison.get("therapy_sensitivity_overlap") or []
                    ),
                    "status": "ok",
                }
            )

    out_json = metrics_dir() / "ablation_results.json"
    out_json.write_text(json.dumps(rows, indent=2))
    csv_path = metrics_dir() / "ablation_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)

    cmp_path = metrics_dir() / "architecture_comparison_eval.json"
    cmp_path.write_text(json.dumps(comparisons, indent=2, default=str))

    debate_rows = []
    for gene, mut in [tuple(x) for x in pcfg.get("debate_cases", [])]:
        gold = gold_case(gene, mut)
        gold_reasoning = gold["target_reasoning"]
        gold_sens, gold_res = extract_therapies_from_reasoning(gold_reasoning)
        try:
            debate_run = run_mutation_comparison(
                gene, mut, architectures=["debate"], live_evidence=False, use_cached_trace=True
            )
            result = debate_run["architectures"].get("debate")
        except Exception as exc:
            debate_rows.append({
                "gene": gene, "mutation": mut, "architecture": "debate",
                "therapy_f1": None, "direction_acc": None, "status": f"error:{exc}",
            })
            continue
        if not result:
            debate_rows.append({
                "gene": gene, "mutation": mut, "architecture": "debate",
                "therapy_f1": None, "direction_acc": None, "status": "missing_trace",
            })
            continue
        reasoning = extract_target_reasoning(result)
        pred_sens, pred_res = extract_therapies_from_reasoning(reasoning)
        debate_rows.append({
            "gene": gene, "mutation": mut, "architecture": "debate",
            "therapy_f1": round(therapy_f1(pred_sens + pred_res, gold_sens + gold_res), 3),
            "direction_acc": round(direction_accuracy(reasoning, gold_reasoning), 3),
            "debate_steps": len((result.get("reasoning") or {}).get("debate_trace") or []),
            "status": "ok",
        })

    vus_rows = []
    for gene, mut in [tuple(x) for x in pcfg.get("vus_demo_cases", [])]:
        arch = pcfg.get("vus_demo_architecture", "single")
        try:
            vus_run = run_mutation_comparison(
                gene, mut, architectures=[arch], live_evidence=False, use_cached_trace=True
            )
            result = vus_run["architectures"].get(arch)
        except Exception as exc:
            vus_rows.append({"gene": gene, "mutation": mut, "status": f"error:{exc}"})
            continue
        if not result:
            vus_rows.append({"gene": gene, "mutation": mut, "status": "missing_trace"})
            continue
        reasoning = extract_target_reasoning(result)
        sens, res = extract_therapies_from_reasoning(reasoning)
        therapy = reasoning.get("therapy") or {}
        abstained = not sens and not res
        status_ok = abstained or therapy.get("recommendation_status") == "insufficient_evidence"
        vus_rows.append({
            "gene": gene, "mutation": mut, "architecture": arch,
            "abstained": abstained,
            "recommendation_status": therapy.get("recommendation_status"),
            "evidence_tier": (result.get("variant_routing") or {}).get("evidence_tier"),
            "status": "ok" if status_ok else "therapy_leak",
        })

    if debate_rows:
        (metrics_dir() / "debate_eval.json").write_text(json.dumps(debate_rows, indent=2))
    if vus_rows:
        (metrics_dir() / "vus_eval.json").write_text(json.dumps(vus_rows, indent=2))

    metrics.aggregate_ablation()
    try:
        from src.metrics_bundle import write_platform_summary

        write_platform_summary()
    except Exception:
        pass
    print(f"Wrote {out_json}, {csv_path}, and {cmp_path}")


if __name__ == "__main__":
    main()
