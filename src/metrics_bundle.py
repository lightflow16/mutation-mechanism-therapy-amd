"""Export and compare full metrics bundles (CSV, JSONL, traces, comparisons)."""
from __future__ import annotations

import csv
import json
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any

from src.config import metrics_dir
from src.platform import detect_platform
from src.productive_metrics import write_productive_metrics_report
from src.ror_analysis import write_ror_benchmark
from src.trace_viz import generate_trace_html

METRICS_GLOBS = (
    "calls.csv",
    "phases.csv",
    "llm_calls.jsonl",
    "system_samples.csv",
    "ablation_summary.csv",
    "ablation_results.csv",
    "architecture_comparison.json",
    "architecture_comparison_eval.json",
    "run_manifest.json",
    "platform_summary.json",
    "architecture_metrics.csv",
    "llm_call_summary.csv",
    "productive_throughput.csv",
    "before_after_comparison.csv",
    "productive_metrics.json",
    "workflow_trace_dashboard.html",
    "return_on_reasoning.csv",
    "ror_benchmark.json",
    "hallucination_report.csv",
    "hallucination_summary.json",
    "benchmark_confidence.csv",
    "benchmark_confidence_minimal.csv",
    "fold_confidence_panel.csv",
    "fold_confidence_summary.json",
    "autonomy_report.json",
    "autonomy_traits.csv",
    "biodesignbench_style.csv",
    "task_suite.csv",
    "able_metrics.csv",
    "dbtl_metrics.json",
    "tevv_lite.csv",
    "evidence_ablation.csv",
    "extended_thinking_ablation.csv",
    "extended_thinking_summary.json",
    "multimodal_ablation.csv",
    "platform_comparison.json",
    "full_submission_report.json",
    "platform_comparison.json",
    "blackboard_ingress_by_role.csv",
    "comparison_*.json",
    "trace_*.json",
)


def export_artifacts_bundle(dest: Path | None = None) -> Path | None:
    """Copy PDB paths referenced in traces + rescue outputs into artifacts_bundle.tgz."""
    from src.config import load_config

    md = metrics_dir()
    cfg = load_config()
    if not cfg.get("pipeline", {}).get("export_artifacts_bundle", False):
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    platform = detect_platform()
    bundle_name = f"artifacts_bundle_{platform['platform_id']}_{stamp}"
    staging = md / bundle_name
    staging.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for trace_path in md.glob("trace_*.json"):
        trace = json.loads(trace_path.read_text())
        for block in (trace.get("structure", {}), trace.get("rescue") or {}):
            for key in ("pdb_path", "boltz_pdb", "esmfold_pdb", "folded_candidate_pdb"):
                p = block.get(key)
                if p and Path(p).is_file():
                    dest_file = staging / Path(p).name
                    if not dest_file.exists():
                        shutil.copy2(p, dest_file)
                        copied.append(str(p))
    (staging / "artifacts_manifest.json").write_text(
        json.dumps({"copied_files": copied, "n_files": len(copied)}, indent=2)
    )
    if not copied:
        shutil.rmtree(staging)
        return None
    out = Path(dest) if dest else md.parent / f"{bundle_name}.tgz"
    with tarfile.open(out, "w:gz") as tar:
        tar.add(staging, arcname=bundle_name)
    shutil.rmtree(staging)
    return out


def _collect_metrics_files(src: Path) -> list[Path]:
    found: list[Path] = []
    for pattern in METRICS_GLOBS:
        found.extend(sorted(src.glob(pattern)))
    # de-dupe
    seen: set[Path] = set()
    out: list[Path] = []
    for p in found:
        if p.is_file() and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def rollup_architecture_metrics(phases_csv: Path) -> list[dict[str, Any]]:
    """Aggregate phases.csv rows by architecture label suffix (_single/_cot/_blackboard)."""
    if not phases_csv.exists():
        return []
    buckets: dict[str, dict[str, Any]] = {}
    for row in csv.DictReader(phases_csv.open()):
        label = row.get("label") or ""
        arch = "other"
        for suffix in ("single", "cot", "blackboard"):
            if label.endswith(f"_{suffix}") or f"_{suffix}_" in label:
                arch = suffix
                break
        b = buckets.setdefault(
            arch,
            {
                "architecture": arch,
                "n_phases": 0,
                "cpu_time_s": 0.0,
                "gpu_active_s": 0.0,
                "gpu_attached_s": 0.0,
                "ingress_tokens": 0,
                "egress_tokens": 0,
                "reasoning_tokens": 0,
                "latency_s": 0.0,
            },
        )
        b["n_phases"] += 1
        for key in ("cpu_time_s", "gpu_active_s", "gpu_attached_s", "latency_s"):
            try:
                b[key] += float(row.get(key) or 0)
            except (TypeError, ValueError):
                pass
        for key in ("ingress_tokens", "egress_tokens", "reasoning_tokens"):
            try:
                b[key] += int(float(row.get(key) or 0))
            except (TypeError, ValueError):
                pass
    rows = []
    for b in buckets.values():
        rows.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()})
    return sorted(rows, key=lambda r: r["architecture"])


def rollup_llm_calls(llm_log: Path) -> list[dict[str, Any]]:
    """Summarize llm_calls.jsonl by query_id + architecture."""
    if not llm_log.exists():
        return []
    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    for line in llm_log.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row.get("query_id") or "unknown", row.get("architecture") or "unknown")
        b = buckets.setdefault(
            key,
            {
                "query_id": key[0],
                "architecture": key[1],
                "n_calls": 0,
                "ingress_tokens": 0,
                "egress_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": 0,
                "latency_s": 0.0,
                "agents": set(),
            },
        )
        b["n_calls"] += 1
        for fld in ("ingress_tokens", "egress_tokens", "reasoning_tokens", "total_tokens"):
            b[fld] += int(row.get(fld) or 0)
        try:
            b["latency_s"] += float(row.get("latency_s") or 0)
        except (TypeError, ValueError):
            pass
        if row.get("agent_role"):
            b["agents"].add(row["agent_role"])
    out = []
    for b in buckets.values():
        agents = b.pop("agents")
        b["agents"] = sorted(agents)
        b["latency_s"] = round(b["latency_s"], 2)
        b["agents"] = ",".join(b["agents"])
        out.append(b)
    return sorted(out, key=lambda r: (r["query_id"], r["architecture"]))


def write_platform_summary(out_dir: Path | None = None) -> Path:
    """Write platform_summary.json + architecture_metrics.csv into metrics dir."""
    md = out_dir or metrics_dir()
    md.mkdir(parents=True, exist_ok=True)
    platform = detect_platform()
    arch_rows = rollup_architecture_metrics(md / "phases.csv")
    llm_rows = rollup_llm_calls(md / "llm_calls.jsonl")
    arch_csv = md / "architecture_metrics.csv"
    if arch_rows:
        with arch_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(arch_rows[0].keys()))
            w.writeheader()
            w.writerows(arch_rows)
    llm_csv = md / "llm_call_summary.csv"
    if llm_rows:
        with llm_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(llm_rows[0].keys()))
            w.writeheader()
            w.writerows(llm_rows)
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **platform,
        "architecture_metrics": arch_rows,
        "llm_call_summary": llm_rows,
        "files_present": [p.name for p in _collect_metrics_files(md)],
    }
    path = md / "platform_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    write_productive_metrics_report(md)
    write_ror_benchmark(md)
    generate_trace_html(md)
    return path


def export_metrics_bundle(
    dest: Path | None = None,
    *,
    label: str | None = None,
) -> Path:
    """Copy all metrics artifacts into a tarball for download / cross-platform compare."""
    md = metrics_dir()
    write_platform_summary(md)
    platform = detect_platform()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    bundle_name = label or f"metrics_bundle_{platform['platform_id']}_{stamp}"
    if dest is None:
        dest = md.parent / f"{bundle_name}.tgz"
    dest = Path(dest)
    staging = md / bundle_name
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for src in _collect_metrics_files(md):
        shutil.copy2(src, staging / src.name)
    manifest = staging / "bundle_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "bundle_name": bundle_name,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                **platform,
                "files": sorted(p.name for p in staging.iterdir() if p.is_file()),
            },
            indent=2,
        )
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w:gz") as tar:
        tar.add(staging, arcname=bundle_name)
    shutil.rmtree(staging)
    return dest


def load_bundle_dir(path: Path) -> Path:
    """Return directory containing metrics files (extract .tgz if needed)."""
    path = Path(path)
    if path.is_dir():
        return path
    if path.suffixes[-2:] == [".tar", ".gz"] or path.suffix == ".tgz":
        extract_to = path.parent / path.name.replace(".tgz", "").replace(".tar.gz", "_extracted")
        extract_to.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(extract_to)
        subs = [p for p in extract_to.iterdir() if p.is_dir()]
        return subs[0] if len(subs) == 1 else extract_to
    raise FileNotFoundError(path)


def compare_platform_bundles(
    bundle_a: Path,
    bundle_b: Path,
    *,
    label_a: str | None = None,
    label_b: str | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Compare two exported metrics bundles (e.g. colab_cuda vs amd_rocm)."""
    dir_a = load_bundle_dir(bundle_a)
    dir_b = load_bundle_dir(bundle_b)

    def _read_json(p: Path) -> dict:
        return json.loads(p.read_text()) if p.exists() else {}

    def _phase_map(d: Path) -> dict[str, dict]:
        csv_path = d / "phases.csv"
        if not csv_path.exists():
            return {}
        out: dict[str, dict] = {}
        for row in csv.DictReader(csv_path.open()):
            out[row.get("label", "")] = row
        return out

    manifest_a = _read_json(dir_a / "run_manifest.json") or _read_json(dir_a / "bundle_manifest.json")
    manifest_b = _read_json(dir_b / "run_manifest.json") or _read_json(dir_b / "bundle_manifest.json")
    phases_a = _phase_map(dir_a)
    phases_b = _phase_map(dir_b)
    common_labels = sorted(set(phases_a) & set(phases_b))

    phase_diffs = []
    for label in common_labels:
        ra, rb = phases_a[label], phases_b[label]
        diff: dict[str, Any] = {"label": label}
        for key in ("cpu_time_s", "gpu_active_s", "gpu_attached_s", "latency_s", "total_tokens"):
            try:
                va = float(ra.get(key) or 0)
                vb = float(rb.get(key) or 0)
                diff[f"{key}_a"] = va
                diff[f"{key}_b"] = vb
                diff[f"{key}_delta"] = round(vb - va, 4)
                if va > 0:
                    diff[f"{key}_ratio_b_over_a"] = round(vb / va, 3)
            except (TypeError, ValueError):
                pass
        phase_diffs.append(diff)

    arch_a = _read_json(dir_a / "platform_summary.json").get("architecture_metrics", [])
    arch_b = _read_json(dir_b / "platform_summary.json").get("architecture_metrics", [])
    arch_compare = []
    for arch in ("single", "cot", "blackboard", "other"):
        ma = next((x for x in arch_a if x.get("architecture") == arch), {})
        mb = next((x for x in arch_b if x.get("architecture") == arch), {})
        if not ma and not mb:
            continue
        arch_compare.append(
            {
                "architecture": arch,
                "latency_s_a": ma.get("latency_s"),
                "latency_s_b": mb.get("latency_s"),
                "gpu_active_s_a": ma.get("gpu_active_s"),
                "gpu_active_s_b": mb.get("gpu_active_s"),
                "egress_tokens_a": ma.get("egress_tokens"),
                "egress_tokens_b": mb.get("egress_tokens"),
            }
        )

    reasoning_a = _read_json(dir_a / "architecture_comparison.json")
    reasoning_b = _read_json(dir_b / "architecture_comparison.json")
    therapy_compare = []
    for case_key in sorted(set(reasoning_a) | set(reasoning_b)):
        ca = reasoning_a.get(case_key, {})
        cb = reasoning_b.get(case_key, {})
        therapy_compare.append(
            {
                "case": case_key,
                "route_agreement_a": ca.get("route_agreement"),
                "route_agreement_b": cb.get("route_agreement"),
                "therapy_overlap_a": ca.get("therapy_sensitivity_overlap"),
                "therapy_overlap_b": cb.get("therapy_sensitivity_overlap"),
            }
        )

    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform_a": label_a or manifest_a.get("platform_id", dir_a.name),
        "platform_b": label_b or manifest_b.get("platform_id", dir_b.name),
        "manifest_a": manifest_a,
        "manifest_b": manifest_b,
        "phase_diffs": phase_diffs,
        "architecture_compare": arch_compare,
        "reasoning_compare": therapy_compare,
    }
    out_path = out_path or metrics_dir() / "platform_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    csv_path = out_path.with_suffix(".csv")
    if phase_diffs:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(phase_diffs[0].keys()))
            w.writeheader()
            w.writerows(phase_diffs)
    return result
