#!/usr/bin/env python3
"""Compare metrics bundles from Colab (CUDA) vs AMD (ROCm)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import setup_env
from src.metrics_bundle import compare_platform_bundles, export_metrics_bundle


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two metrics bundles (NVIDIA vs AMD).")
    ap.add_argument("bundle_a", help="First bundle .tgz or directory (e.g. Colab export)")
    ap.add_argument("bundle_b", help="Second bundle .tgz or directory (e.g. AMD export)")
    ap.add_argument("--label-a", default="colab_cuda")
    ap.add_argument("--label-b", default="amd_rocm")
    ap.add_argument("--out", default=None, help="Write platform_comparison.json here")
    args = ap.parse_args()

    setup_env()
    out = compare_platform_bundles(
        Path(args.bundle_a),
        Path(args.bundle_b),
        label_a=args.label_a,
        label_b=args.label_b,
        out_path=Path(args.out) if args.out else None,
    )
    print(json.dumps(out, indent=2))
    bundle = export_metrics_bundle(label="platform_comparison_merged")
    print(f"Merged comparison bundle: {bundle}")


if __name__ == "__main__":
    main()
