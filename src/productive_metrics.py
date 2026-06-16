"""Productive throughput + workflow density metrics (not raw GPU %).

Built from native hooks in src/metrics.py (calls/phases/llm_calls/system_samples).
Use for demo narrative: token-per-GPU-second, latency-to-decision, workflow density.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from src.config import metrics_dir


def _read_llm_calls(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _read_phases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(csv.DictReader(path.open()))


def _read_system(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(csv.DictReader(path.open()))


def _f(val: Any, default: float = 0.0) -> float:
    try:
        if val in ("", "NA", None):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def _infer_architecture(row: dict) -> str:
    arch = row.get("architecture") or ""
    if arch:
        return arch
    label = row.get("label") or ""
    for suffix in ("single", "cot", "blackboard"):
        if suffix in label:
            return suffix
    return "unknown"


def compute_productive_rows(
    llm_rows: list[dict[str, Any]],
    phase_rows: list[dict[str, Any]],
    sys_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Per query_id × architecture productive throughput table."""
    by_key: dict[tuple[str, str], dict[str, Any]] = {}

    for row in llm_rows:
        qid = row.get("query_id") or "unknown"
        arch = _infer_architecture(row)
        key = (qid, arch)
        b = by_key.setdefault(
            key,
            {
                "query_id": qid,
                "gene": row.get("gene", ""),
                "mutation": row.get("mutation", ""),
                "architecture": arch,
                "platform_id": row.get("platform_id", ""),
                "n_agent_steps": 0,
                "n_models_used": set(),
                "ingress_tokens": 0,
                "egress_tokens": 0,
                "reasoning_tokens": 0,
                "latency_to_decision_s": 0.0,
                "round_2_latency_s": 0.0,
                "weight_cache_hits": 0,
                "planner_calls": 0,
                "expert_calls": 0,
            },
        )
        b["n_agent_steps"] += 1
        b["ingress_tokens"] += int(row.get("ingress_tokens") or 0)
        b["egress_tokens"] += int(row.get("egress_tokens") or 0)
        b["reasoning_tokens"] += int(row.get("reasoning_tokens") or 0)
        b["latency_to_decision_s"] += _f(row.get("latency_s"))
        if int(row.get("round") or 0) >= 2:
            b["round_2_latency_s"] += _f(row.get("latency_s"))
        if row.get("weight_cache_hit"):
            b["weight_cache_hits"] += 1
        model = row.get("model") or ""
        if model:
            b["n_models_used"].add(model)
        role = row.get("agent_role") or ""
        if role == "Planner":
            b["planner_calls"] += 1
        elif role in ("Structure", "Mechanism", "Evidence", "Therapy"):
            b["expert_calls"] += 1

    gpu_by_arch: dict[str, dict[str, float]] = {}
    for prow in phase_rows:
        label = prow.get("label") or ""
        arch = "other"
        for suffix in ("single", "cot", "blackboard"):
            if f"_{suffix}" in label or label.endswith(suffix):
                arch = suffix
                break
        g = gpu_by_arch.setdefault(arch, {"gpu_active_s": 0.0, "gpu_attached_s": 0.0})
        g["gpu_active_s"] += _f(prow.get("gpu_active_s"))
        g["gpu_attached_s"] += _f(prow.get("gpu_attached_s"))

    gfx_vals = [_f(r.get("gfx_util")) for r in sys_rows if _f(r.get("gfx_util")) > 0]
    vram_vals = [_f(r.get("torch_peak_gib")) for r in sys_rows if _f(r.get("torch_peak_gib")) > 0]
    mean_gfx = round(statistics.mean(gfx_vals), 1) if gfx_vals else "NA"
    peak_vram = round(max(vram_vals), 3) if vram_vals else "NA"

    out: list[dict[str, Any]] = []
    for (_, arch), b in sorted(by_key.items(), key=lambda x: (x[0][0], x[0][1])):
        gpu = gpu_by_arch.get(arch, {})
        gpu_active = gpu.get("gpu_active_s", 0.0)
        gpu_attached = gpu.get("gpu_attached_s", 0.0)
        lat = b["latency_to_decision_s"]
        egress = b["egress_tokens"]
        ingress = b["ingress_tokens"]
        steps = b["n_agent_steps"]
        models = b.pop("n_models_used")
        productive_tps = round(egress / gpu_active, 2) if gpu_active > 0 else "NA"
        gpu_productivity = round(gpu_active / gpu_attached, 3) if gpu_attached > 0 else "NA"
        workflow_density = round(steps / lat, 3) if lat > 0 else "NA"
        ingress_amp = round(ingress / egress, 2) if egress > 0 else "NA"
        cache_rate = round(b["weight_cache_hits"] / steps, 3) if steps else 0
        early_exit_savings_pct = round(100 * b["round_2_latency_s"] / lat, 1) if lat > 0 else 0
        row = {
            **b,
            "n_models_used": len(models),
            "heterogeneous_models": ",".join(sorted(models)),
            "gpu_active_s": round(gpu_active, 3),
            "gpu_attached_s": round(gpu_attached, 3),
            "gpu_productivity_ratio": gpu_productivity,
            "productive_egress_tokens_per_gpu_s": productive_tps,
            "workflow_density_steps_per_s": workflow_density,
            "ingress_amplification": ingress_amp,
            "weight_cache_hit_rate": cache_rate,
            "early_exit_savings_pct_if_round1_consensus": early_exit_savings_pct,
            "mean_gfx_util_sampled": mean_gfx,
            "peak_vram_gib_sampled": peak_vram,
        }
        out.append(row)
    return out


def compute_before_after(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Baseline (single) vs depth (blackboard) side-by-side for demo slides."""
    by_qid: dict[str, dict[str, dict]] = {}
    for r in rows:
        qid = r["query_id"]
        by_qid.setdefault(qid, {})[r["architecture"]] = r

    table = []
    for qid, archs in sorted(by_qid.items()):
        base = archs.get("single", {})
        depth = archs.get("blackboard", {})
        if not base and not depth:
            continue
        table.append(
            {
                "query_id": qid,
                "metric": "latency_to_decision_s",
                "baseline_single": base.get("latency_to_decision_s"),
                "blackboard_mas": depth.get("latency_to_decision_s"),
            }
        )
        table.append(
            {
                "query_id": qid,
                "metric": "productive_egress_tokens_per_gpu_s",
                "baseline_single": base.get("productive_egress_tokens_per_gpu_s"),
                "blackboard_mas": depth.get("productive_egress_tokens_per_gpu_s"),
            }
        )
        table.append(
            {
                "query_id": qid,
                "metric": "gpu_productivity_ratio",
                "baseline_single": base.get("gpu_productivity_ratio"),
                "blackboard_mas": depth.get("gpu_productivity_ratio"),
            }
        )
        table.append(
            {
                "query_id": qid,
                "metric": "workflow_density_steps_per_s",
                "baseline_single": base.get("workflow_density_steps_per_s"),
                "blackboard_mas": depth.get("workflow_density_steps_per_s"),
            }
        )
        table.append(
            {
                "query_id": qid,
                "metric": "n_agent_steps",
                "baseline_single": base.get("n_agent_steps"),
                "blackboard_mas": depth.get("n_agent_steps"),
            }
        )
    return table


def build_agent_trace(llm_rows: list[dict[str, Any]], query_id: str, architecture: str = "blackboard") -> dict:
    """Node/edge trace for visualization."""
    filtered = [
        r
        for r in llm_rows
        if (r.get("query_id") or "unknown") == query_id
        and _infer_architecture(r) == architecture
    ]
    nodes = []
    edges = []
    prev_id = None
    t0 = None
    for i, r in enumerate(filtered):
        ts = r.get("timestamp", "")
        lat = _f(r.get("latency_s"))
        node_id = f"n{i}"
        if t0 is None:
            t0 = ts
        nodes.append(
            {
                "id": node_id,
                "agent": r.get("agent_role"),
                "model": r.get("model"),
                "round": r.get("round"),
                "latency_s": lat,
                "egress_tokens": r.get("egress_tokens"),
                "timestamp": ts,
            }
        )
        if prev_id:
            edges.append({"from": prev_id, "to": node_id, "type": "sequential"})
        prev_id = node_id
    total_lat = sum(_f(r.get("latency_s")) for r in filtered)
    return {
        "query_id": query_id,
        "architecture": architecture,
        "n_nodes": len(nodes),
        "total_latency_s": round(total_lat, 2),
        "nodes": nodes,
        "edges": edges,
    }


def write_productive_metrics_report(out_dir: Path | None = None) -> Path:
    md = out_dir or metrics_dir()
    md.mkdir(parents=True, exist_ok=True)
    llm = _read_llm_calls(md / "llm_calls.jsonl")
    phases = _read_phases(md / "phases.csv")
    sys_rows = _read_system(md / "system_samples.csv")
    rows = compute_productive_rows(llm, phases, sys_rows)

    # Enrich from traces: early_exit actual, rubric, conflict resolution
    trace_meta: dict[tuple[str, str], dict] = {}
    for tp in md.glob("trace_*_*.json"):
        import json as _json

        tr = _json.loads(tp.read_text())
        parts = tp.stem.replace("trace_", "").rsplit("_", 1)
        if len(parts) != 2:
            continue
        qid = parts[0]
        arch = parts[1]
        reasoning = tr.get("reasoning") or {}
        bb = reasoning.get("blackboard_trace") or []
        cr_hits = sum(1 for m in bb if m.get("agent") == "ConflictResolver")
        trace_meta[(qid, arch)] = {
            "early_exit_actual": reasoning.get("early_exit", False),
            "mechanism_rubric_after": reasoning.get("mechanism_rubric_after"),
            "conflict_resolution_rate": 1.0 if cr_hits else 0.0,
            "reasoning_depth_tokens": reasoning.get("total_tokens", 0),
        }
    for r in rows:
        meta = trace_meta.get((r["query_id"], r["architecture"]), {})
        r.update(meta)
        if meta.get("early_exit_actual"):
            r["early_exit_savings_pct_if_round1_consensus"] = r.get(
                "early_exit_savings_pct_if_round1_consensus", 0
            )

    before_after = compute_before_after(rows)

    prod_csv = md / "productive_throughput.csv"
    if rows:
        with prod_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    ba_csv = md / "before_after_comparison.csv"
    if before_after:
        with ba_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(before_after[0].keys()))
            w.writeheader()
            w.writerows(before_after)

    traces = {}
    for r in rows:
        if r["architecture"] == "blackboard" and r["query_id"] != "unknown":
            traces[r["query_id"]] = build_agent_trace(llm, r["query_id"], "blackboard")

    report = {
        "narrative": (
            "Prefer productive_egress_tokens_per_gpu_s and gpu_productivity_ratio over mean gfx %."
        ),
        "productive_throughput": rows,
        "before_after_single_vs_blackboard": before_after,
        "agent_traces": traces,
    }
    path = md / "productive_metrics.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    return path
