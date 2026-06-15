#!/usr/bin/env python3
"""Generate productive throughput report + HTML trace dashboard from metrics dir."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import setup_env
from src.metrics_bundle import export_metrics_bundle, write_platform_summary
from src.productive_metrics import write_productive_metrics_report
from src.ror_analysis import write_ror_benchmark
from src.trace_viz import generate_trace_html


def main() -> None:
    setup_env()
    write_platform_summary()
    json_path = write_productive_metrics_report()
    ror_path = write_ror_benchmark()
    html_path = generate_trace_html()
    bundle = export_metrics_bundle()
    print(f"productive_metrics: {json_path}")
    print(f"ror_benchmark: {ror_path}")
    print(f"dashboard: {html_path}")
    print(f"bundle: {bundle}")


if __name__ == "__main__":
    main()
