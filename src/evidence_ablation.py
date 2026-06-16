"""Evidence context tier ablation: CIViC-only vs +PubMed vs +structure."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.config import ROOT, metrics_dir
from src.evidence import load_case_evidence, search_literature
from src.pipeline import extract_target_reasoning, extract_therapies_from_reasoning


def _gold_case(gene: str, mutation: str) -> dict:
    p = ROOT / "data" / "cases" / f"{gene}_{mutation}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _therapy_f1(pred: list[str], gold: list[str]) -> float:
    p = {x.lower().strip().split()[0] for x in pred if x}
    g = {x.lower().strip().split()[0] for x in gold if x}
    if not g:
        return 1.0 if not p else 0.0
    prec = len(p & g) / len(p) if p else 0.0
    rec = len(p & g) / len(g)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def _tier_evidence(gene: str, mutation: str, tier: str) -> list[dict]:
    base = load_case_evidence(gene, mutation)
    if tier == "T1_civic_only":
        return [e for e in base if (e.get("source") or "").lower() == "civic"]
    if tier == "T2_civic_pubmed":
        ev = list(base)
        ev.extend(search_literature(f"{gene} {mutation} cancer therapy", k=2))
        return ev
    return base


def write_ablation_report(md: Path | None = None) -> Path:
    md = md or metrics_dir()
    rows: list[dict[str, Any]] = []
    for trace_path in sorted(md.glob("trace_*_*.json")):
        trace = json.loads(trace_path.read_text())
        parts = trace_path.stem.replace("trace_", "").rsplit("_", 1)
        if len(parts) != 2:
            continue
        case, arch = parts
        gene, mutation = case.rsplit("_", 1) if "_" in case else (case, "")
        gold = _gold_case(gene, mutation)
        gs, gr = extract_therapies_from_reasoning(gold.get("target_reasoning", {}))
        tr = extract_target_reasoning(trace)
        ps, pr = extract_therapies_from_reasoning(tr)
        f1 = _therapy_f1(ps + pr, gs + gr)
        for tier in ("T1_civic_only", "T2_civic_pubmed", "T3_full"):
            ev = _tier_evidence(gene, mutation, tier)
            rows.append({
                "gene": gene,
                "mutation": mutation,
                "architecture": arch,
                "evidence_tier": tier,
                "n_evidence_items": len(ev),
                "therapy_f1_live_trace": f1,
            })
    out = md / "evidence_ablation.csv"
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return out
