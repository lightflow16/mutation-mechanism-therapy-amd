#!/usr/bin/env python3
"""Generate fold confidence benchmark figures F1–F4 from benchmark_confidence.csv."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, metrics_dir, setup_env


def _f(val, default=0.0):
    try:
        if val in ("", "NA", None):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def main() -> None:
    setup_env()
    md = metrics_dir()
    csv_path = md / "benchmark_confidence.csv"
    if not csv_path.exists():
        print("Run fold_confidence_eval first.")
        return
    rows = list(csv.DictReader(csv_path.open()))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed")
        return

    cb = load_config().get("confidence_benchmark", {})
    op_thresh = cb.get("operational_accept_threshold", 0.78)

    llm = [_f(r["llm_confidence_norm"]) for r in rows if r.get("llm_confidence_norm")]
    labels = [int(_f(r["good_structure_label"])) for r in rows if r.get("llm_confidence_norm")]
    plddt = [_f(r["target_residue_plddt"]) / 100.0 for r in rows if r.get("target_residue_plddt")]
    plddt = plddt[: len(llm)]

    # F1 — correlation scatter
    fig, ax = plt.subplots(figsize=(5, 4))
    y = plddt if len(plddt) == len(llm) else [_f(r.get("mean_plddt")) / 100.0 for r in rows[: len(llm)]]
    ax.scatter(llm, y, alpha=0.75, c=labels if len(labels) == len(llm) else "C0")
    ax.set_xlabel("LLM confidence (norm)")
    ax.set_ylabel("pLDDT@site / 100")
    ax.set_title("F1: LLM vs structure quality proxy")
    fig.tight_layout()
    fig.savefig(md / "confidence_scatter_F1.svg")
    fig.savefig(md / "confidence_scatter_F1.png", dpi=120)
    plt.close(fig)

    # F2 — reliability diagram (deciles)
    if llm and labels and len(llm) == len(labels):
        fig, ax = plt.subplots(figsize=(5, 4))
        bins = np.linspace(0, 1, 11)
        centers, accs = [], []
        for i in range(10):
            lo, hi = bins[i], bins[i + 1]
            bucket = [labels[j] for j, p in enumerate(llm) if lo <= p < hi or (i == 9 and p == 1.0)]
            if bucket:
                centers.append((lo + hi) / 2)
                accs.append(sum(bucket) / len(bucket))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
        ax.plot(centers, accs, "o-", label="empirical")
        ax.set_xlabel("Mean predicted confidence")
        ax.set_ylabel("Fraction good_structure")
        ax.set_title("F2: Reliability diagram")
        ax.legend()
        fig.tight_layout()
        fig.savefig(md / "confidence_reliability_F2.svg")
        plt.close(fig)

    # F3 — precision-recall sweep
    if llm and labels and len(llm) == len(labels):
        fig, ax = plt.subplots(figsize=(5, 4))
        thresholds = np.linspace(0, 1, 21)
        prec, rec = [], []
        pos = sum(labels)
        for t in thresholds:
            pred = [1 if p >= t else 0 for p in llm]
            tp = sum(1 for p, y in zip(pred, labels) if p and y)
            fp = sum(1 for p, y in zip(pred, labels) if p and not y)
            fn = sum(1 for p, y in zip(pred, labels) if not p and y)
            prec.append(tp / (tp + fp) if (tp + fp) else 1.0)
            rec.append(tp / pos if pos else 0.0)
        ax.plot(rec, prec, "-o", markersize=3)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("F3: PR curve (LLM confidence)")
        fig.tight_layout()
        fig.savefig(md / "confidence_pr_F3.svg")
        plt.close(fig)

    # F4 — threshold tradeoff
    if llm and labels and len(llm) == len(labels):
        fig, ax = plt.subplots(figsize=(5, 4))
        thresholds = np.linspace(0, 1, 21)
        f1s = []
        for t in thresholds:
            pred = [1 if p >= t else 0 for p in llm]
            tp = sum(1 for p, y in zip(pred, labels) if p and y)
            fp = sum(1 for p, y in zip(pred, labels) if p and not y)
            fn = sum(1 for p, y in zip(pred, labels) if not p and y)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        ax.plot(thresholds, f1s, label="F1")
        ax.axvline(op_thresh, color="r", linestyle="--", label=f"op={op_thresh}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("F1")
        ax.set_title("F4: Threshold tradeoff")
        ax.legend()
        fig.tight_layout()
        fig.savefig(md / "confidence_threshold_F4.svg")
        plt.close(fig)

    # F8 — mutant/rescue paired (TP53 rescue rows vs confidence_only)
    rescue_rows = [r for r in rows if r.get("task_type") in ("rescue_design", "mutant_fold")]
    conf_rows = [r for r in rows if r.get("task_type") == "confidence_only"]
    if rescue_rows or conf_rows:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        for ax, key, title in zip(
            axes,
            ("llm_confidence_norm", "mean_plddt", "ddg_pred"),
            ("LLM conf", "mean pLDDT", "ddG pred"),
        ):
            conf_vals = [_f(r.get(key)) for r in conf_rows if r.get(key) not in ("", None)]
            rescue_vals = [_f(r.get(key)) for r in rescue_rows if r.get(key) not in ("", None)]
            if conf_vals:
                ax.boxplot([conf_vals], positions=[1], widths=0.5, labels=["confidence_only"])
            if rescue_vals:
                pos = 2 if conf_vals else 1
                ax.boxplot([rescue_vals], positions=[pos], widths=0.5, labels=["rescue/mutant"])
            ax.set_title(title)
        fig.suptitle("F8: confidence_only vs rescue/mutant panels")
        fig.tight_layout()
        fig.savefig(md / "confidence_rescue_F8.svg")
        plt.close(fig)

    html = f"""<!DOCTYPE html><html><body>
<h1>Fold confidence figures</h1>
<p>Generated from {csv_path.name} ({len(rows)} rows)</p>
<img src="confidence_scatter_F1.svg" width="600"/>
<img src="confidence_reliability_F2.svg" width="600"/>
<img src="confidence_pr_F3.svg" width="600"/>
<img src="confidence_threshold_F4.svg" width="600"/>
<img src="confidence_rescue_F8.svg" width="600"/>
</body></html>"""
    (md / "confidence_benchmark_figures.html").write_text(html)
    print(f"Wrote F1–F4 figures to {md}")


if __name__ == "__main__":
    main()
