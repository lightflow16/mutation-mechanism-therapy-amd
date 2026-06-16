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


def _hr_evidence(gene: str, mutation: str, pred_sens: list[str], pred_res: list[str], trace: dict) -> float:
    """Mismatch rate vs gold case and retrieved evidence directions."""
    gold = _gold_case(gene, mutation).get("target_reasoning", {}).get("therapy", {})
    gs = _norm_drugs(gold.get("sensitivity") or [])
    gr = _norm_drugs(gold.get("resistance") or [])
    ps = _norm_drugs(pred_sens)
    pr = _norm_drugs(pred_res)
    g = gs | gr
    p = ps | pr
    mismatches = len(p - g) + len(g - p)
    total = max(len(g | p), 1)

    ev_items = trace.get("evidence") or []
    ev_refs = 0
    ev_bad = 0
    for item in ev_items:
        therapies = item.get("therapies") or ""
        if not therapies:
            continue
        ev_refs += len([t for t in therapies.split(",") if t.strip()])
        direction = (item.get("direction") or "").lower()
        for tok in therapies.split(","):
            drug = tok.strip().lower().split()[0]
            if not drug:
                continue
            in_sens = drug in ps
            in_res = drug in pr
            if "sensit" in direction and in_res and not in_sens:
                ev_bad += 1
            if "resist" in direction and in_sens and not in_res:
                ev_bad += 1

    if ev_refs:
        return round((mismatches + ev_bad) / (total + ev_refs), 3)
    return round(mismatches / total, 3)


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
    cfg = load_config()
    rescue_cfg = cfg.get("rescue", {})
    cb_cfg = cfg.get("confidence_benchmark", {})
    ptm_thresh = float(rescue_cfg.get("fold_confidence_ptm_threshold", 0.15))
    plddt_thresh = float(cb_cfg.get("proxy_good_mean_plddt", 70))
    passed = 0
    total = 0
    ptm = rescue.get("boltz_ptm")
    if ptm is not None:
        total += 1
        if float(ptm) >= ptm_thresh:
            passed += 1
    esm_plddt = rescue.get("esmfold_plddt")
    if esm_plddt is not None:
        total += 1
        if float(esm_plddt) >= plddt_thresh:
            passed += 1
    if rescue.get("fold_method"):
        total += 1
        passed += 1
    designs = rescue.get("designs") or []
    if designs:
        total += 1
        if not rescue.get("destabilizing") or len(designs) >= 1:
            passed += 1
    return round(passed / total, 3) if total else 1.0


def _hr_rescue(trace: dict) -> float:
    rescue = trace.get("rescue") or {}
    if not rescue:
        return 0.0
    tr = extract_target_reasoning(trace)
    text = (tr.get("mechanism") or "").lower() + " " + str(rescue.get("interpreter") or "").lower()
    destab = bool(rescue.get("destabilizing"))
    if any(w in text for w in ("rescued", "stabiliz", "restored fold", "success")):
        if destab and not rescue.get("fold_method"):
            return 1.0
    if "rescued" in text and destab:
        return 1.0
    return 0.0


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


def _hr_safety(trace: dict) -> float:
    return _hr_policy(trace)


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
            "HR_evidence": _hr_evidence(gene, mutation, sens, res, trace),
            "HR_tool": _hr_tool(trace),
            "HR_rescue": _hr_rescue(trace),
            "HR_safety": _hr_safety(trace),
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

    keys = ["HR_design", "HR_property", "HR_evidence", "HR_tool", "HR_rescue", "HR_safety", "HR_policy"]
    summary = {"n_traces": len(rows)}
    for k in keys + ["BVR", "structural_hallucination_rate"]:
        summary[f"mean_{k}"] = round(sum(r[k] for r in rows) / len(rows), 3) if rows else 0

    by_arch: dict[str, dict[str, float]] = {}
    for arch in sorted({r["architecture"] for r in rows}):
        sub = [r for r in rows if r["architecture"] == arch]
        by_arch[arch] = {
            f"mean_{k}": round(sum(r[k] for r in sub) / len(sub), 3)
            for k in keys + ["BVR"]
        }
    summary["by_architecture"] = by_arch
    single_hr = by_arch.get("single", {}).get("mean_HR_evidence", 0)
    bb_hr = by_arch.get("blackboard", {}).get("mean_HR_evidence", 0)
    summary["blackboard_vs_single_delta_HR_evidence"] = round(bb_hr - single_hr, 3)
    (md / "hallucination_summary.json").write_text(json.dumps(summary, indent=2))
    return csv_path
