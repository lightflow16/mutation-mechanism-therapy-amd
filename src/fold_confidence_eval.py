"""Fold confidence benchmark: traces + rescue + ECE/Brier vs LLM confidence."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from src.config import load_config, metrics_dir
from src.pipeline import extract_target_reasoning


def _f(val: Any, default: float = 0.0) -> float:
    try:
        if val in ("", "NA", None):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def _parse_boltz_json(rescue: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("boltz_ptm", "boltz_complex_plddt", "boltz_plddt"):
        if rescue.get(key) is not None:
            out[key] = _f(rescue.get(key))
    raw = rescue.get("boltz_scores_json")
    if raw and isinstance(raw, str):
        try:
            obj = json.loads(raw)
            for k in ("ptm", "complex_plddt", "plddt"):
                if k in obj:
                    out[f"boltz_{k}"] = _f(obj[k])
        except json.JSONDecodeError:
            pass
    return out


def _ece_brier(pairs: list[tuple[float, float]]) -> dict[str, float]:
    if not pairs:
        return {"ece": 0.0, "brier": 0.0}
    brier = sum((p - y) ** 2 for p, y in pairs) / len(pairs)
    bins = 10
    ece = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        bucket = [(p, y) for p, y in pairs if lo <= p < hi or (i == bins - 1 and p == 1.0)]
        if not bucket:
            continue
        acc = sum(y for _, y in bucket) / len(bucket)
        conf = sum(p for p, _ in bucket) / len(bucket)
        ece += abs(acc - conf) * len(bucket) / len(pairs)
    return {"ece": round(ece, 4), "brier": round(brier, 4)}


def write_benchmark(md: Path | None = None) -> Path:
    md = md or metrics_dir()
    cfg = load_config().get("rescue", {})
    rows: list[dict[str, Any]] = []
    conf_pairs: list[tuple[float, float]] = []

    for trace_path in sorted(md.glob("trace_*.json")):
        trace = json.loads(trace_path.read_text())
        structure = trace.get("structure") or {}
        rescue = trace.get("rescue") or {}
        boltz = _parse_boltz_json(rescue)
        parts = trace_path.stem.replace("trace_", "").rsplit("_", 1)
        case, arch = (parts[0], parts[1]) if len(parts) == 2 else (trace_path.stem, "")
        gene, mutation = case.rsplit("_", 1) if "_" in case else (case, "")
        tr = extract_target_reasoning(trace)
        llm_conf = _f(tr.get("confidence"))
        fold_conf = boltz.get("boltz_ptm") or _f(structure.get("pLDDT_at_residue")) / 100.0
        bvr = 1.0 if rescue.get("fold_method") else 0.0
        if llm_conf > 0:
            conf_pairs.append((llm_conf, min(fold_conf, 1.0)))

        rows.append({
            "gene": gene,
            "mutation": mutation,
            "architecture": arch,
            "plddt_at_residue": _f(structure.get("pLDDT_at_residue")),
            "mean_plddt": _f(structure.get("mean_pLDDT_protein")),
            "llm_confidence": llm_conf,
            "fold_confidence": round(fold_conf, 3),
            "mutant_ddg_kcal_mol": _f(rescue.get("mutant_ddg_kcal_mol")),
            "destabilizing": bool(rescue.get("destabilizing")),
            "fold_method": rescue.get("fold_method") or "",
            "boltz_ptm": boltz.get("boltz_ptm", _f(rescue.get("boltz_ptm"))),
            "boltz_complex_plddt": boltz.get("boltz_complex_plddt", _f(rescue.get("boltz_complex_plddt"))),
            "esmfold_plddt": _f(rescue.get("esmfold_plddt")),
            "ptm_threshold": _f(cfg.get("fold_confidence_ptm_threshold", 0.15)),
        })

    out = md / "benchmark_confidence.csv"
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    metrics = _ece_brier(conf_pairs)
    summary = {
        "n_rows": len(rows),
        "mean_plddt_at_residue": round(sum(r["plddt_at_residue"] for r in rows) / len(rows), 2) if rows else 0,
        "rescue_runs": sum(1 for r in rows if r["fold_method"]),
        **metrics,
    }
    (md / "fold_confidence_summary.json").write_text(json.dumps(summary, indent=2))
    return out
