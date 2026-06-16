"""Rule-based hallucination metrics (6 HR metrics + BVR)."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from src.config import ROOT, load_config, metrics_dir
from src.pipeline import extract_target_reasoning, extract_therapies_from_reasoning


def _gold_case(gene: str, mutation: str) -> dict:
    p = ROOT / "data" / "cases" / f"{gene}_{mutation}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _norm_drugs(xs: list[str]) -> set[str]:
    return {x.lower().strip().split()[0] for x in xs if x}


def _hr_design(trace: dict) -> float:
    tr = extract_target_reasoning(trace)
    mech = (tr.get("mechanism") or "").strip()
    if not mech or mech.lower() in ("unknown", "n/a"):
        return 1.0
    if len(mech) < 20:
        return 1.0
    return 0.0


def _hr_property(trace: dict) -> float:
    tr = extract_target_reasoning(trace)
    mech = (tr.get("mechanism") or "").lower()
    structure = trace.get("structure") or {}
    rescue = trace.get("rescue") or {}
    flags = 0
    cited = re.findall(r"plddt[^0-9]*([0-9]+(?:\.[0-9]+)?)", mech, re.I)
    if cited:
        try:
            if abs(float(cited[0]) - float(structure.get("pLDDT_at_residue", 0))) > 0.5:
                flags += 1
        except (TypeError, ValueError):
            pass
    ddg = rescue.get("mutant_ddg_kcal_mol")
    if ddg is not None:
        destab = bool(rescue.get("destabilizing"))
        if any(w in mech for w in ("stabiliz", "restore", "rescued")) and destab:
            flags += 1
    return float(min(flags, 1))


def _hr_evidence(gene: str, mutation: str, pred_sens: list[str], pred_res: list[str]) -> float:
    gold = _gold_case(gene, mutation).get("target_reasoning", {}).get("therapy", {})
    gs = _norm_drugs(gold.get("sensitivity") or [])
    gr = _norm_drugs(gold.get("resistance") or [])
    ps = _norm_drugs(pred_sens)
    pr = _norm_drugs(pred_res)
    g = gs | gr
    p = ps | pr
    if not g and not p:
        return 0.0
    if not g:
        return 1.0 if p else 0.0
    return round((len(p - g) + len(g - p)) / max(len(g | p), 1), 3)


def _hr_tool(trace: dict) -> float:
    rescue = trace.get("rescue") or {}
    if not rescue:
        return 0.0
    tr = extract_target_reasoning(trace)
    text = (tr.get("mechanism") or "").lower()
    ddg = rescue.get("mutant_ddg_kcal_mol")
    if ddg is not None and "ddg" in text:
        try:
            nums = [float(x) for x in re.findall(r"-?[0-9]+\.[0-9]+", text)]
            if nums and abs(nums[0] - float(ddg)) > 0.5:
                return 1.0
        except (TypeError, ValueError):
            pass
    return 0.0


def _bvr(trace: dict) -> float:
    rescue = trace.get("rescue") or {}
    if not rescue:
        return 1.0
    cfg = load_config().get("rescue", {})
    thresh = float(cfg.get("fold_confidence_ptm_threshold", 0.15))
    ptm = rescue.get("boltz_ptm")
    passed = 0
    total = 0
    if ptm is not None:
        total += 1
        if float(ptm) >= thresh or rescue.get("destabilizing") is False:
            passed += 1
    if rescue.get("fold_method"):
        total += 1
        passed += 1
    return round(passed / total, 3) if total else 0.0


def _hr_policy(trace: dict) -> float:
    tier = (trace.get("variant_routing") or trace.get("target", {})).get("evidence_tier")
    if tier is None:
        tier = trace.get("target", {}).get("evidence_tier")
    tr = extract_target_reasoning(trace)
    sens, res = extract_therapies_from_reasoning(tr)
    if tier in ("none", "weak") and (sens or res):
        therapy = tr.get("therapy") or {}
        status = (therapy.get("recommendation_status") or "").lower()
        if status != "insufficient_evidence":
            return 1.0
    return 0.0


def write_report(md: Path | None = None) -> Path:
    md = md or metrics_dir()
    rows: list[dict[str, Any]] = []
    for trace_path in sorted(md.glob("trace_*.json")):
        parts = trace_path.stem.replace("trace_", "").rsplit("_", 1)
        if len(parts) != 2:
            continue
        case, arch = parts[0], parts[1]
        gene, mutation = case.rsplit("_", 1) if "_" in case else (case, "")
        trace = json.loads(trace_path.read_text())
        tr = extract_target_reasoning(trace)
        sens, res = extract_therapies_from_reasoning(tr)
        rows.append({
            "gene": gene,
            "mutation": mutation,
            "architecture": arch,
            "HR_design": _hr_design(trace),
            "HR_property": _hr_property(trace),
            "HR_evidence": _hr_evidence(gene, mutation, sens, res),
            "HR_tool": _hr_tool(trace),
            "BVR": _bvr(trace),
            "HR_policy": _hr_policy(trace),
            "structural_hallucination_rate": round(1 - _bvr(trace), 3),
        })

    csv_path = md / "hallucination_report.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    keys = ["HR_design", "HR_property", "HR_evidence", "HR_tool", "HR_policy"]
    summary = {"n_traces": len(rows)}
    for k in keys + ["BVR", "structural_hallucination_rate"]:
        summary[f"mean_{k}"] = round(sum(r[k] for r in rows) / len(rows), 3) if rows else 0
    (md / "hallucination_summary.json").write_text(json.dumps(summary, indent=2))
    return csv_path
