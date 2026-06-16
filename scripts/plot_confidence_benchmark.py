#!/usr/bin/env python3
"""Generate fold confidence benchmark figures (SVG) from benchmark_confidence.csv."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import metrics_dir, setup_env


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
    except ImportError:
        print("matplotlib not installed")
        return

    llm = [float(r["llm_confidence"]) for r in rows if r.get("llm_confidence")]
    fold = [float(r["fold_confidence"]) for r in rows if r.get("fold_confidence")]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(llm, fold, alpha=0.7)
    ax.set_xlabel("LLM confidence")
    ax.set_ylabel("Fold confidence (pTM / pLDDT)")
    ax.set_title("F1: LLM vs fold confidence")
    out = md / "confidence_scatter_F1.svg"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
