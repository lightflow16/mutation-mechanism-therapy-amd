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


def _lora_loaded(path: str | None) -> bool:
    if not path:
        return False
    p = Path(path)
    if not p.is_dir():
        return False
    weights = list(p.glob("adapter_model.*")) + list(p.glob("*.safetensors"))
    return bool(weights)


def run_full_submission(
    *,
    lora_path: str | None = None,
    run_lora_train: bool | None = None,
    skip_live: bool = False,
    reload_modules: bool = True,
    run_lora_comparison: bool | None = None,
) -> dict[str, Any]:
    """Always run all architectures × all demo cases (live), eval, and export metrics.

    When a LoRA adapter is loaded and run_lora_comparison is True (or the
    pipeline.run_lora_comparison config key is set), also runs the full 2×4
    evaluation matrix (base model vs fine-tuned × single/cot/blackboard/debate)
    and writes lora_comparison.csv + lora_comparison_summary.txt alongside the
    other metrics outputs.
    """
    if reload_modules:
        from src.config import reload_src_modules

        reload_src_modules()

    from src import metrics
    from src.config import load_config, metrics_dir, setup_env, shared_dir
    from src.metrics_bundle import export_metrics_bundle, write_platform_summary
    from src.pipeline import format_comparison_report, run_all_modes
    from src.platform import detect_platform, write_run_manifest

    try:
        from src import progress
    except ImportError:
        progress = None

    setup_env()
    cfg = load_config()
    pcfg = cfg.get("pipeline", {})
    metrics.set_metrics_dir(str(metrics_dir()))

    platform = detect_platform()

    if lora_path is None:
        lora_path = _lora_path(cfg, shared_dir)
    lora_ok = _lora_loaded(lora_path)
    if progress:
        progress.log("submission", f"LoRA path={lora_path} loaded={lora_ok}")
    print(f"LoRA: path={lora_path} | loaded={lora_ok}")

    manifest_path = write_run_manifest({
        "mode": "full_submission",
        "lora_path": lora_path,
        "lora_loaded": lora_ok,
    })
    print(f"Platform: {platform['platform_id']} | GPU: {platform['gpu_name']}")
    print(f"Manifest: {manifest_path}")

    if run_lora_train is None:
        run_lora_train = bool(pcfg.get("run_lora_train", False))

    distill_info: dict[str, Any] | None = None
    if bool(pcfg.get("lora_distill_from_traces", False)):
        print("Building LoRA dataset from teacher (blackboard) traces...")
        from train.build_dataset import build_dataset_from_traces

        path, n_rows = build_dataset_from_traces()
        distill_info = {"path": str(path), "n_rows": n_rows}
        print(f"[distill] {n_rows} teacher rows -> {path}")

    if run_lora_train:
        print("Running LoRA SFT...")
        _run_py("train/lora_sft.py")
        lora_path = lora_path or _lora_path(cfg, shared_dir)

    report: dict[str, Any] = {
        "platform": platform,
        "manifest": str(manifest_path),
        "lora_path": lora_path,
        "lora_loaded": lora_ok,
        "steps": [],
    }
    if distill_info:
        report["lora_distill"] = distill_info

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

    try:
        from src.hallucination_eval import write_report as write_hallucination
        from src.fold_confidence_eval import write_benchmark as write_fold_confidence
        from src.agent_autonomy_eval import run as run_autonomy

        write_hallucination()
        write_fold_confidence()
        live_probes = bool(pcfg.get("live_safety_probes", False))
        run_autonomy(live_probes=live_probes)
        from src.evidence_ablation import write_ablation_report

        write_ablation_report()
        from src.extended_thinking_ablation import write_reports as write_extended_thinking

        write_extended_thinking()
        report["steps"].append("trust_eval")
    except Exception as exc:
        report["trust_eval_error"] = str(exc)

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

    # ── LoRA base-vs-fine-tuned comparison (2×4 matrix) ──────────────────────
    if run_lora_comparison is None:
        run_lora_comparison = bool(pcfg.get("run_lora_comparison", True))
    if lora_ok and run_lora_comparison:
        print("\nRunning base vs LoRA comparison matrix (2 model variants × 4 architectures)...")
        print("  LoRA traces reused from the live matrix above — only base model runs fresh.")
        try:
            # Pass --reuse-lora-traces so the script reads the LoRA traces that were
            # already written by the live matrix above instead of re-running the model.
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT)
            import subprocess as _sp
            _sp.check_call(
                [sys.executable, str(ROOT / "scripts" / "run_lora_comparison.py"),
                 "--reuse-lora-traces"],
                env=env,
            )
            report["steps"].append("lora_comparison")
            cmp_csv = metrics_dir() / "lora_comparison.csv"
            report["lora_comparison_csv"] = str(cmp_csv) if cmp_csv.is_file() else None
        except Exception as exc:
            report["lora_comparison_error"] = str(exc)
            print(f"  [warn] LoRA comparison failed: {exc}")
    elif not lora_ok:
        print("\nSkipping LoRA comparison — no adapter weights found.")

    if bool(pcfg.get("export_metrics_bundle", True)):
        bundle = export_metrics_bundle()
        report["metrics_bundle"] = str(bundle)
        report["steps"].append("export_bundle")
        print(f"Metrics bundle: {bundle}")
        try:
            from src.metrics_bundle import export_artifacts_bundle

            art = export_artifacts_bundle()
            if art:
                report["artifacts_bundle"] = str(art)
                print(f"Artifacts bundle: {art}")
        except Exception:
            pass

    summary_path = metrics_dir() / "full_submission_report.json"
    summary_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Full submission report: {summary_path}")
    return report
