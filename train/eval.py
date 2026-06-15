"""Ablation: base vs LoRA x single/CoT/blackboard; Therapy F1 + direction acc + metrics."""
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
from src.pipeline import run_case

ROOT = Path(__file__).resolve().parents[1]
CASES = [("EGFR", "L858R"), ("PIK3CA", "E545K"), ("TP53", "R175H")]
ARCHITECTURES = ("single", "cot", "blackboard")


def therapy_f1(pred: list[str], gold: list[str]) -> float:
    def norm(xs):
        aliases = {"gefitinib": "gefitinib", "erlotinib": "erlotinib", "afatinib": "afatinib",
                   "osimertinib": "osimertinib", "alpelisib": "alpelisib"}
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


def extract_reasoning(result: dict) -> dict:
    r = result.get("reasoning", {})
    if "target_reasoning" in r:
        return r["target_reasoning"]
    return r


def extract_therapies(reasoning: dict) -> list[str]:
    tr = reasoning.get("therapy") or reasoning
    if isinstance(tr, dict):
        return list(tr.get("sensitivity") or []) + list(tr.get("resistance") or [])
    return []


def gold_case(gene: str, mutation: str) -> dict:
    p = ROOT / "data" / "cases" / f"{gene}_{mutation}.json"
    return json.loads(p.read_text())


def main():
    cfg = load_config()
    metrics.set_metrics_dir(str(metrics_dir()))
    rows = []
    for gene, mut in CASES:
        gold = gold_case(gene, mut)
        for arch in ARCHITECTURES:
            if gene == "TP53" and mut == "R175H" and arch != "blackboard":
                continue  # rescue-focused; still run blackboard for demo trace
            with metrics.phase(f"eval_{gene}_{mut}_{arch}"):
                try:
                    result = run_case(
                        gene, mut, architecture=arch,
                        live_evidence=False, use_cached_trace=True,
                    )
                    reasoning = extract_reasoning(result)
                    pred = extract_therapies(reasoning)
                    f1 = therapy_f1(pred, extract_therapies(gold["target_reasoning"]))
                    dacc = direction_accuracy(reasoning, gold["target_reasoning"])
                    summ = metrics.summary()
                except Exception as e:
                    f1 = dacc = 0.0
                    summ = {"error": str(e)}
                rows.append({
                    "gene": gene, "mutation": mut, "architecture": arch,
                    "therapy_f1": round(f1, 3), "direction_acc": round(dacc, 3),
                    "total_tokens": summ.get("total", 0),
                })
    out_json = metrics_dir() / "ablation_results.json"
    out_json.write_text(json.dumps(rows, indent=2))
    csv_path = metrics_dir() / "ablation_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)
    metrics.aggregate_ablation()
    print(f"Wrote {out_json} and {csv_path}")


if __name__ == "__main__":
    main()
