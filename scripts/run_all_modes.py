#!/usr/bin/env python3
"""Run all pipeline modes per mutation and compare single / cot / blackboard."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import setup_env
from src.metrics_bundle import export_metrics_bundle
from src.pipeline import format_comparison_report, run_all_modes


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run full pipeline matrix and compare architectures per mutation."
    )
    ap.add_argument("--lora-path", default=None, help="LoRA adapter for single-agent runs")
    ap.add_argument("--no-cached-baseline", action="store_true")
    ap.add_argument("--export-bundle", action="store_true", help="Export metrics .tgz after run")
    ap.add_argument("--out", default=None, help="Write JSON summary to this path")
    args = ap.parse_args()

    setup_env()
    results = run_all_modes(
        lora_path=args.lora_path,
        use_cached_baseline=not args.no_cached_baseline,
    )

    for key, comparison in results.get("comparisons", {}).items():
        if str(key).startswith("cached_"):
            continue
        print(format_comparison_report(comparison))
        print()

    summary = {
        "comparisons": {
            k: v
            for k, v in results.get("comparisons", {}).items()
            if not str(k).startswith("cached_")
        }
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    if args.export_bundle:
        bundle = export_metrics_bundle()
        print(f"Metrics bundle: {bundle}")


if __name__ == "__main__":
    main()
