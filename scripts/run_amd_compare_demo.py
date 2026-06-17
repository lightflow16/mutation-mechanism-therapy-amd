#!/usr/bin/env python3
"""CPU demo: compare Colab reference bundle vs local metrics (amd-compare todo)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import metrics_dir, setup_env
from src.metrics_bundle import compare_platform_bundles


def _default_colab_bundle() -> Path | None:
    candidates = [
        # Colab bundle committed to repo under uploads/ (git pull to get it)
        ROOT / "uploads" / "metrics_bundle_colab_cuda_20260617_121238",
        # Legacy backups paths
        ROOT.parent / "backups" / "colab_artifacts_20260616_045708" / "metrics_bundle_colab_cuda_20260615_231801",
        # Any bundle placed manually under ../backups/colab_artifacts_*/
        *(sorted((ROOT.parent / "backups").glob("colab_artifacts_*/metrics_bundle_*"))
          if (ROOT.parent / "backups").is_dir() else []),
        ROOT / "metrics" / "local",
    ]
    for p in candidates:
        if p.is_dir() and (p / "ablation_results.csv").is_file():
            return p
    return None


def main() -> None:
    setup_env()
    colab = _default_colab_bundle()
    local = metrics_dir()
    if not colab:
        print("No Colab reference bundle found under backups/")
        sys.exit(1)
    if not (local / "ablation_results.csv").is_file():
        print(f"Local metrics missing at {local} — run eval or integration test first.")
        sys.exit(1)
    out = compare_platform_bundles(
        colab,
        local,
        label_a="colab_cuda_reference",
        label_b="local_cpu_or_amd",
        out_path=local / "platform_comparison.json",
    )
    print(json.dumps(out, indent=2))
    print(f"Wrote {local / 'platform_comparison.json'}")


if __name__ == "__main__":
    main()
