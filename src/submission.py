"""Run every configured flow on every run: live matrix + eval + metrics export."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _run_py(script: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    subprocess.check_call([sys.executable, str(ROOT / script)], env=env)


def _lora_path(cfg: dict, shared_dir_fn) -> str | None:
    paths = cfg.get("paths", {})
    for key in ("lora_adapter_final", "lora_ckpts"):
        p = paths.get(key)
        if p and Path(p).is_dir() and any(Path(p).iterdir()):
            return str(p)
    local = shared_dir_fn(cfg) / "lora_adapter_final"
    return str(local) if local.is_dir() and any(local.iterdir()) else None


def run_full_submission(
    *,
    lora_path: str | None = None,
    run_lora_train: bool | None = None,
    skip_live: bool = False,
    reload_modules: bool = True,
) -> dict[str, Any]:
    """Always run all architectures × all demo cases (live), eval, and export metrics."""
    if reload_modules:
        from src.config import reload_src_modules

        reload_src_modules()

    from src import metrics
    from src.config import load_config, metrics_dir, setup_env, shared_dir
    from src.metrics_bundle import export_metrics_bundle, write_platform_summary
    from src.pipeline import format_comparison_report, run_all_modes
    from src.platform import detect_platform, write_run_manifest

    setup_env()
    cfg = load_config()
    pcfg = cfg.get("pipeline", {})
    metrics.set_metrics_dir(str(metrics_dir()))

    platform = detect_platform()
    manifest_path = write_run_manifest({"mode": "full_submission"})
    print(f"Platform: {platform['platform_id']} | GPU: {platform['gpu_name']}")
    print(f"Manifest: {manifest_path}")

    if lora_path is None:
        lora_path = _lora_path(cfg, shared_dir)
    if run_lora_train is None:
        run_lora_train = bool(pcfg.get("run_lora_train", False))

    if run_lora_train:
        print("Running LoRA SFT...")
        _run_py("train/lora_sft.py")
        lora_path = lora_path or _lora_path(cfg, shared_dir)

    report: dict[str, Any] = {
        "platform": platform,
        "manifest": str(manifest_path),
        "steps": [],
    }

    if not skip_live:
        print("Running live matrix: all architectures × all demo cases...")
        with metrics.SysSampler("full_submission_live"):
            live = run_all_modes(
                lora_path=lora_path,
                live_evidence=bool(pcfg.get("live_evidence", True)),
                use_cached_baseline=bool(pcfg.get("use_cached_baseline", True)),
            )
        report["live"] = live
        report["steps"].append("live_matrix")
        for key, comp in live.get("comparisons", {}).items():
            if str(key).startswith("cached_"):
                continue
            print(format_comparison_report(comp))
            print()

    if bool(pcfg.get("run_eval_after_live", True)):
        print("Running eval (scores saved traces)...")
        _run_py("train/eval.py")
        report["steps"].append("eval")

    metrics.aggregate_ablation()
    write_platform_summary()
    try:
        from src.productive_metrics import write_productive_metrics_report
        from src.ror_analysis import write_ror_benchmark
        from src.trace_viz import generate_trace_html

        write_productive_metrics_report()
        write_ror_benchmark()
        generate_trace_html()
    except Exception:
        pass
    report["steps"].append("productive_dashboard")

    if bool(pcfg.get("export_metrics_bundle", True)):
        bundle = export_metrics_bundle()
        report["metrics_bundle"] = str(bundle)
        report["steps"].append("export_bundle")
        print(f"Metrics bundle: {bundle}")

    summary_path = metrics_dir() / "full_submission_report.json"
    summary_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Full submission report: {summary_path}")
    return report
