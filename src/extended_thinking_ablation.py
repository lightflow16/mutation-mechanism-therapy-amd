"""Headline reasoning ablation: thinking tokens, rubric, conflict resolution, multimodal."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.config import load_config, metrics_dir
from src.pipeline import extract_target_reasoning


def _read_llm(md: Path) -> list[dict]:
    log = md / "llm_calls.jsonl"
    if not log.is_file():
        return []
    return [json.loads(l) for l in log.read_text().splitlines() if l.strip()]


def _trace_rows(md: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tp in sorted(md.glob("trace_*.json")):
        parts = tp.stem.replace("trace_", "").rsplit("_", 1)
        if len(parts) != 2:
            continue
        qid, arch = parts[0], parts[1]
        gene, mutation = qid.rsplit("_", 1) if "_" in qid else (qid, "")
        trace = json.loads(tp.read_text())
        reasoning = trace.get("reasoning") or {}
        if isinstance(reasoning, dict) and reasoning.get("target_reasoning") and "early_exit" not in reasoning:
            for k in ("early_exit", "mechanism_rubric_before", "mechanism_rubric_after", "blackboard_trace"):
                if k in reasoning:
                    continue
        bb = reasoning.get("blackboard_trace") or []
        cr = sum(1 for m in bb if m.get("agent") == "ConflictResolver")
        rows.append({
            "query_id": qid,
            "gene": gene,
            "mutation": mutation,
            "architecture": arch,
            "early_exit": bool(reasoning.get("early_exit")),
            "mechanism_rubric_before": reasoning.get("mechanism_rubric_before"),
            "mechanism_rubric_after": reasoning.get("mechanism_rubric_after"),
            "conflict_resolution_rate": 1.0 if cr else 0.0,
            "multimodal_image": bool(reasoning.get("multimodal_image")),
        })
    return rows


def write_reports(md: Path | None = None) -> Path:
    md = md or metrics_dir()
    md.mkdir(parents=True, exist_ok=True)
    llm = _read_llm(md)
    traces = _trace_rows(md)

    by_arch: dict[str, dict[str, float]] = {}
    for arch in ("single", "cot", "blackboard", "debate"):
        sub = [r for r in llm if r.get("architecture") == arch]
        if not sub:
            continue
        think = sum(int(r.get("reasoning_tokens") or r.get("thinking_tokens") or 0) for r in sub)
        mm = sum(1 for r in sub if r.get("multimodal_image"))
        by_arch[arch] = {
            "n_calls": len(sub),
            "thinking_tokens": think,
            "multimodal_calls": mm,
            "mean_latency_s": round(sum(float(r.get("latency_s") or 0) for r in sub) / len(sub), 2),
        }

    headline: list[dict[str, Any]] = []
    for tr in traces:
        arch = tr["architecture"]
        agg = by_arch.get(arch, {})
        headline.append({
            **tr,
            "thinking_tokens": agg.get("thinking_tokens", 0),
            "multimodal_calls_arch": agg.get("multimodal_calls", 0),
            "headline_ablation": f"{tr['architecture']}_vs_single",
        })

    if not headline and by_arch:
        for arch, agg in by_arch.items():
            headline.append({
                "architecture": arch,
                "thinking_tokens": agg["thinking_tokens"],
                "multimodal_calls_arch": agg["multimodal_calls"],
                "conflict_resolution_rate": "",
                "mechanism_rubric_after": "",
                "early_exit": "",
            })

    out = md / "extended_thinking_ablation.csv"
    if headline:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(headline[0].keys()))
            w.writeheader()
            w.writerows(headline)

    mm_rows = []
    for arch, agg in by_arch.items():
        mm_rows.append({
            "architecture": arch,
            "multimodal_image_calls": agg.get("multimodal_calls", 0),
            "text_only_calls": agg.get("n_calls", 0) - agg.get("multimodal_calls", 0),
            "thinking_tokens": agg.get("thinking_tokens", 0),
        })
    if mm_rows:
        with open(md / "multimodal_ablation.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(mm_rows[0].keys()))
            w.writeheader()
            w.writerows(mm_rows)

    manifest = md / "run_manifest.json"
    lora_info = {}
    if manifest.is_file():
        m = json.loads(manifest.read_text())
        lora_info = {
            "lora_path": m.get("lora_path"),
            "lora_loaded": m.get("lora_loaded"),
        }
    summary = {
        "by_architecture": by_arch,
        "n_trace_rows": len(traces),
        **lora_info,
    }
    (md / "extended_thinking_summary.json").write_text(json.dumps(summary, indent=2))
    return out
