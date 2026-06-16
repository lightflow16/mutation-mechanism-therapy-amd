"""Fold confidence benchmark: canonical CSV + calibration summary (§14)."""
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import ROOT, load_config, metrics_dir
from src.fold_scores import pdb_plddt_stats
from src.pipeline import extract_target_reasoning

CANONICAL_COLUMNS = [
    "run_id", "case_id", "protein_id", "variant", "task_type", "model_family", "model_version",
    "llm_model", "prompt_mode", "structure_source", "chain_id", "n_residues", "mean_plddt",
    "min_plddt", "target_residue_plddt", "ptm", "iptm", "pae_mean", "qa_external_score",
    "rmsd_to_exp", "tm_score_to_exp", "lddt_to_exp", "good_structure_label", "ddg_pred",
    "llm_confidence_raw", "llm_confidence_norm", "llm_confidence_scope", "llm_explanation",
    "accepted_by_threshold", "calibration_bin", "split", "notes", "architecture",
    "locally_confident_site", "gene",
]

MINIMAL_COLUMNS = [
    "case_id", "protein_id", "variant", "model_family", "prompt_mode",
    "mean_plddt", "ptm", "qa_external_score", "rmsd_to_exp", "tm_score_to_exp",
    "good_structure_label", "llm_confidence_norm",
]


def _f(val: Any, default: float | None = 0.0) -> float | None:
    try:
        if val in ("", "NA", None):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def _norm_llm_conf(raw: Any) -> float:
    v = _f(raw, 0.0) or 0.0
    return round(v / 100.0 if v > 1.0 else v, 4)


def _calibration_bin(norm: float) -> str:
    if norm <= 0:
        return "NA"
    i = min(9, int(norm * 10))
    lo, hi = i / 10, (i + 1) / 10
    return f"{lo:.1f}-{hi:.1f}"


def _good_structure_label(row: dict[str, Any], cfg: dict) -> int:
    tm = _f(row.get("tm_score_to_exp"), None)
    lddt = _f(row.get("lddt_to_exp"), None)
    if tm is not None and tm >= cfg.get("good_tm_threshold", 0.7):
        return 1
    if lddt is not None and lddt >= cfg.get("good_lddt_threshold", 0.8):
        return 1
    mean_p = _f(row.get("mean_plddt"), None)
    target_p = _f(row.get("target_residue_plddt"), None)
    proxy = cfg.get("proxy_good_mean_plddt", 70)
    if mean_p is not None and target_p is not None:
        if mean_p >= proxy and target_p >= 50:
            return 1
    return 0


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


def _auc_pr(pairs: list[tuple[float, float]]) -> float:
    """Trapezoid AUC-PR from sorted (score, label) pairs."""
    if not pairs or not any(y for _, y in pairs):
        return 0.0
    ranked = sorted(pairs, key=lambda x: -x[0])
    tp = fp = 0
    pos = sum(y for _, y in pairs)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return 0.0
    auc = 0.0
    prev_r = 0.0
    for score, label in ranked:
        if label:
            tp += 1
        else:
            fp += 1
        r = tp / pos
        p = tp / (tp + fp) if (tp + fp) else 0.0
        auc += (r - prev_r) * p
        prev_r = r
    return round(auc, 4)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return round(num / den, 4) if den else None


def _load_llm_model(md: Path, qid: str, arch: str) -> str:
    log = md / "llm_calls.jsonl"
    if not log.is_file():
        return ""
    for line in reversed(log.read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("query_id") == qid and row.get("architecture") == arch:
            return row.get("model") or ""
    return ""


def _run_id(md: Path) -> str:
    manifest = md / "run_manifest.json"
    if manifest.is_file():
        obj = json.loads(manifest.read_text())
        if obj.get("run_id"):
            return str(obj["run_id"])
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_trace_name(path: Path) -> tuple[str, str, str]:
    stem = path.stem
    if stem.startswith("trace_"):
        stem = stem[6:]
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        case, arch = parts
    else:
        case, arch = stem, ""
    gene, mutation = case.rsplit("_", 1) if "_" in case else (case, "")
    return gene, mutation, arch


def _base_row(
    *,
    run_id: str,
    gene: str,
    mutation: str,
    arch: str,
    trace: dict,
    cfg: dict,
    cb_cfg: dict,
    md: Path,
    task_type: str = "confidence_only",
    model_family: str = "AlphaFold",
    structure_source: str = "predicted",
    rescue: dict | None = None,
    design_idx: int | None = None,
) -> dict[str, Any]:
    structure = trace.get("structure") or {}
    target = trace.get("target") or {}
    tr = extract_target_reasoning(trace)
    qid = f"{gene}_{mutation}"
    llm_raw = tr.get("confidence")
    llm_norm = _norm_llm_conf(llm_raw)
    accept_thresh = cb_cfg.get("operational_accept_threshold", 0.78)
    local_thresh = cb_cfg.get("local_plddt_threshold", 70)

    mean_plddt = _f(structure.get("mean_pLDDT_protein"))
    target_plddt = _f(structure.get("pLDDT_at_residue"))
    min_plddt = None
    n_res = None
    ptm = None
    pae_mean = None
    ddg_pred = None
    notes = ""
    variant = mutation
    case_id = f"{gene}_{mutation}_nsclc" if gene != "TP53" else f"{gene}_{mutation}_rescue"

    if rescue:
        ddg_pred = _f(rescue.get("mutant_ddg_kcal_mol"), None)
        ptm = _f(rescue.get("boltz_ptm"), None)
        if ptm is None and rescue.get("esmfold_plddt"):
            ptm = round(_f(rescue.get("esmfold_plddt"), 0) / 100.0, 4)
        pdb = rescue.get("boltz_pdb") or rescue.get("esmfold_pdb")
        if pdb:
            stats = pdb_plddt_stats(pdb)
            if stats:
                mean_plddt = mean_plddt or stats.get("mean_plddt")
                min_plddt = stats.get("min_plddt")
                n_res = int(stats.get("n_residues", 0))
        if design_idx is not None:
            variant = f"rescued_seq_{design_idx:02d}"
            case_id = f"{gene}_{mutation}_rescue{design_idx:02d}"
            task_type = "rescue_design"
            model_family = "ESMFold+Boltz" if rescue.get("boltz_pdb") and rescue.get("esmfold_pdb") else (
                "Boltz" if rescue.get("boltz_pdb") else "ESMFold"
            )
            structure_source = "rescue_candidate"
        else:
            task_type = "mutant_fold"
            notes = "TP53 rescue fold QA"
            model_family = "ESMFold+Boltz" if rescue.get("fold_method") == "boltz+esmfold" else str(rescue.get("fold_method") or "ESMFold")

    if mean_plddt is None and target_plddt is not None:
        mean_plddt = target_plddt

    row: dict[str, Any] = {
        "run_id": run_id,
        "case_id": case_id,
        "protein_id": target.get("uniprot") or gene,
        "variant": variant,
        "task_type": task_type,
        "model_family": model_family,
        "model_version": "AF2_v6" if model_family == "AlphaFold" else ("esmfold_v1" if "ESM" in model_family else "boltz-2.2.1"),
        "llm_model": _load_llm_model(md, qid, arch),
        "prompt_mode": arch or trace.get("architecture") or "single",
        "structure_source": structure_source,
        "chain_id": structure.get("chain") or "A",
        "n_residues": n_res or "",
        "mean_plddt": mean_plddt if mean_plddt is not None else "",
        "min_plddt": min_plddt if min_plddt is not None else "",
        "target_residue_plddt": target_plddt if target_plddt is not None else "",
        "ptm": ptm if ptm is not None else "",
        "iptm": "",
        "pae_mean": pae_mean if pae_mean is not None else "",
        "qa_external_score": "",
        "rmsd_to_exp": "",
        "tm_score_to_exp": "",
        "lddt_to_exp": "",
        "good_structure_label": 0,
        "ddg_pred": ddg_pred if ddg_pred is not None else "",
        "llm_confidence_raw": llm_raw if llm_raw is not None else "",
        "llm_confidence_norm": llm_norm,
        "llm_confidence_scope": "mechanism" if task_type == "confidence_only" else "rescue_success",
        "llm_explanation": (tr.get("mechanism") or "")[:240],
        "accepted_by_threshold": 1 if llm_norm >= accept_thresh else 0,
        "calibration_bin": _calibration_bin(llm_norm),
        "split": "benchmark",
        "notes": notes or ("proxy good_structure (no exp ref)" if task_type == "confidence_only" else ""),
        "architecture": arch,
        "locally_confident_site": 1 if (target_plddt or 0) >= local_thresh else 0,
        "gene": gene,
    }
    row["good_structure_label"] = _good_structure_label(row, cb_cfg)
    return row


def _trace_sources(md: Path) -> list[Path]:
    traces = sorted(md.glob("trace_*.json"))
    if traces:
        return traces
    return sorted((ROOT / "data" / "traces").glob("*.json"))


def write_benchmark(md: Path | None = None) -> Path:
    md = md or metrics_dir()
    md.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    cb_cfg = cfg.get("confidence_benchmark", {})
    run_id = _run_id(md)
    rows: list[dict[str, Any]] = []

    for trace_path in _trace_sources(md):
        gene, mutation, arch = _parse_trace_name(trace_path)
        if not gene:
            continue
        trace = json.loads(trace_path.read_text())
        rows.append(_base_row(
            run_id=run_id, gene=gene, mutation=mutation, arch=arch, trace=trace,
            cfg=cfg, cb_cfg=cb_cfg, md=md,
        ))
        rescue = trace.get("rescue") or {}
        if rescue.get("fold_method"):
            rows.append(_base_row(
                run_id=run_id, gene=gene, mutation=mutation, arch=arch, trace=trace,
                cfg=cfg, cb_cfg=cb_cfg, md=md, rescue=rescue, task_type="mutant_fold",
            ))
            for i, _design in enumerate(rescue.get("designs") or [], start=1):
                rows.append(_base_row(
                    run_id=run_id, gene=gene, mutation=mutation, arch=arch, trace=trace,
                    cfg=cfg, cb_cfg=cb_cfg, md=md, rescue=rescue, design_idx=i,
                ))

    out = md / "benchmark_confidence.csv"
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CANONICAL_COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        minimal = [{k: r.get(k, "") for k in MINIMAL_COLUMNS} for r in rows]
        with open(md / "benchmark_confidence_minimal.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=MINIMAL_COLUMNS)
            w.writeheader()
            w.writerows(minimal)
        (md / "fold_confidence_panel.csv").write_bytes(out.read_bytes())

    llm_pairs = [
        (_f(r["llm_confidence_norm"], 0.0) or 0.0, float(r["good_structure_label"]))
        for r in rows if r.get("llm_confidence_norm") not in ("", None)
    ]
    plddt_pairs = [
        ((_f(r["mean_plddt"], 0.0) or 0.0) / 100.0, float(r["good_structure_label"]))
        for r in rows if r.get("mean_plddt") not in ("", None)
    ]
    llm_cal = _ece_brier(llm_pairs)
    plddt_cal = _ece_brier(plddt_pairs)
    xs = [_f(r["llm_confidence_norm"], None) for r in rows if _f(r["llm_confidence_norm"], None) is not None]
    ys = [_f(r["target_residue_plddt"], None) for r in rows if _f(r["target_residue_plddt"], None) is not None]
    if len(xs) == len(ys) and xs:
        corr = _pearson(xs, ys)
    else:
        corr = _pearson(
            [_f(r["llm_confidence_norm"], 0.0) or 0.0 for r in rows],
            [_f(r["mean_plddt"], 0.0) or 0.0 for r in rows if _f(r["mean_plddt"], None) is not None][: len(rows)],
        )

    summary = {
        "run_id": run_id,
        "n_rows": len(rows),
        "proxy_mode": True,
        "llm_calibration": llm_cal,
        "mean_plddt_calibration": plddt_cal,
        "auc_pr_llm_confidence": _auc_pr(llm_pairs),
        "auc_pr_mean_plddt": _auc_pr(plddt_pairs),
        "r_pearson_llm_vs_target_plddt": corr,
        "rescue_runs": sum(1 for r in rows if r.get("task_type") == "rescue_design"),
    }
    (md / "fold_confidence_summary.json").write_text(json.dumps(summary, indent=2))

    plot_script = ROOT / "scripts" / "plot_confidence_benchmark.py"
    if plot_script.is_file() and rows:
        try:
            subprocess.run(
                [sys.executable, str(plot_script)],
                cwd=str(ROOT),
                env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            pass

    return out
