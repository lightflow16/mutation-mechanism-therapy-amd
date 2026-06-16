"""Return on Reasoning (RoR): semantic fidelity vs token/latency cost by architecture."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.config import ROOT, load_config, metrics_dir
from src.pipeline import extract_target_reasoning, extract_therapies_from_reasoning


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
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


def _gold_therapy_f1(gene: str, mutation: str, pred_sens: list[str], pred_res: list[str]) -> float:
    case_path = ROOT / "data" / "cases" / f"{gene}_{mutation}.json"
    if not case_path.exists():
        return 0.0
    gold = json.loads(case_path.read_text())
    tr = gold.get("target_reasoning", {}).get("therapy", {})
    gold_s = {x.lower().split()[0] for x in tr.get("sensitivity", [])}
    gold_r = {x.lower().split()[0] for x in tr.get("resistance", [])}
    pred = {x.lower().split()[0] for x in pred_sens + pred_res}

    def _f1(g: set, p: set) -> float:
        if not g:
            return 1.0 if not p else 0.0
        prec = len(p & g) / len(p) if p else 0.0
        rec = len(p & g) / len(g)
        return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    return (_f1(gold_s, pred) + _f1(gold_r, pred)) / 2


def _therapies_from_trace(trace: dict) -> tuple[list[str], list[str]]:
    return extract_therapies_from_reasoning(extract_target_reasoning(trace))


def _load_eval_scores(md: Path) -> dict[tuple[str, str, str], dict]:
    """(gene, mutation, architecture) -> eval row."""
    out: dict[tuple[str, str, str], dict] = {}
    for row in _read_csv(md / "ablation_results.csv"):
        key = (row.get("gene", ""), row.get("mutation", ""), row.get("architecture", ""))
        out[key] = row
    return out


def _load_llm_cost(md: Path) -> dict[tuple[str, str], dict]:
    """(query_id, architecture) -> cost row from llm_call_summary or llm_calls rollup."""
    summary = _read_csv(md / "llm_call_summary.csv")
    if summary:
        return {(r.get("query_id", ""), r.get("architecture", "")): r for r in summary}

    # fallback: aggregate llm_calls.jsonl
    buckets: dict[tuple[str, str], dict] = {}
    log = md / "llm_calls.jsonl"
    if not log.exists():
        return {}
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row.get("query_id") or "unknown", row.get("architecture") or "unknown")
        b = buckets.setdefault(key, {"total_tokens": 0, "latency_s": 0.0, "ingress_tokens": 0})
        b["total_tokens"] += int(row.get("total_tokens") or 0)
        b["latency_s"] += _f(row.get("latency_s"))
        b["ingress_tokens"] += int(row.get("ingress_tokens") or 0)
    return buckets


def compute_ingress_by_role(md: Path) -> list[dict[str, Any]]:
    """Blackboard ingress accumulation by agent role (compaction ROI baseline)."""
    log = md / "llm_calls.jsonl"
    if not log.exists():
        return []
    by_role: dict[str, dict[str, Any]] = {}
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("architecture") != "blackboard":
            continue
        role = row.get("agent_role") or "unknown"
        b = by_role.setdefault(role, {"agent_role": role, "n_calls": 0, "ingress_tokens": 0, "egress_tokens": 0})
        b["n_calls"] += 1
        b["ingress_tokens"] += int(row.get("ingress_tokens") or 0)
        b["egress_tokens"] += int(row.get("egress_tokens") or 0)
    rows = []
    for b in by_role.values():
        ing = b["ingress_tokens"]
        eg = b["egress_tokens"]
        b["ingress_waste_estimate"] = max(0, ing - eg)
        b["ingress_amplification"] = round(ing / eg, 2) if eg else "NA"
        rows.append(b)
    return sorted(rows, key=lambda r: -int(r["ingress_tokens"]))


def compute_ror_rows(md: Path | None = None) -> list[dict[str, Any]]:
    """Return on Reasoning rows: one per mutation × architecture."""
    md = md or metrics_dir()
    eval_scores = _load_eval_scores(md)
    costs = _load_llm_cost(md)
    comparisons = _read_json(md / "architecture_comparison.json")

    # baseline single token cost per query
    single_tokens: dict[str, float] = {}
    for (qid, arch), cost in costs.items():
        if arch == "single":
            single_tokens[qid] = _f(cost.get("total_tokens"))

    rows: list[dict[str, Any]] = []
    pipe = load_config().get("pipeline", {})
    cases = [tuple(x) for x in pipe.get("demo_cases", [
        ["EGFR", "L858R"],
        ["PIK3CA", "E545K"],
        ["TP53", "R175H"],
    ])]
    debate_cases = [tuple(x) for x in pipe.get("debate_cases", [])]
    architectures = list(pipe.get("architectures", ["single", "cot", "blackboard"]))

    def _append_row(gene: str, mutation: str, arch: str) -> None:
        qid = f"{gene}_{mutation}"
        comp = comparisons.get(qid, comparisons.get(f"{gene}_{mutation}", {}))
        cost = costs.get((qid, arch), {})
        ev = eval_scores.get((gene, mutation, arch), {})
        trace_path = md / f"trace_{gene}_{mutation}_{arch}.json"
        if not trace_path.exists() and arch == "debate":
            cached = ROOT / "data" / "traces" / f"{gene}_{mutation}_debate.json"
            trace = _read_json(cached) if cached.exists() else {}
        else:
            trace = _read_json(trace_path) if trace_path.exists() else {}
        sens, res = _therapies_from_trace(trace)

        therapy_f1 = _f(ev.get("therapy_f1")) if ev.get("therapy_f1") not in (None, "") else None
        direction_acc = _f(ev.get("direction_acc")) if ev.get("direction_acc") not in (None, "") else None
        if therapy_f1 is None and trace:
            therapy_f1 = _gold_therapy_f1(gene, mutation, sens, res)

        semantic = None
        if therapy_f1 is not None and direction_acc is not None:
            semantic = round((therapy_f1 + direction_acc) / 2, 3)
        elif therapy_f1 is not None:
            semantic = round(therapy_f1, 3)

        total_tokens = int(_f(cost.get("total_tokens")))
        latency = round(_f(cost.get("latency_s")), 2)
        base_tok = single_tokens.get(qid) or total_tokens or 1
        cost_multiplier = round(total_tokens / base_tok, 2) if base_tok else "NA"

        ror_tokens = round(semantic / (total_tokens / 1000), 4) if semantic and total_tokens else "NA"
        ror_latency = round(semantic / latency, 4) if semantic and latency > 0 else "NA"
        ror_vs_single = round(semantic / cost_multiplier, 4) if semantic and cost_multiplier not in ("NA", 0) else "NA"

        rows.append(
            {
                "query_id": qid,
                "gene": gene,
                "mutation": mutation,
                "architecture": arch,
                "therapy_f1": therapy_f1,
                "direction_acc": direction_acc,
                "semantic_accuracy_composite": semantic,
                "total_tokens": total_tokens,
                "latency_s": latency,
                "cost_multiplier_vs_single": cost_multiplier,
                "n_agent_steps": cost.get("n_calls") or cost.get("n_agent_steps"),
                "route": trace.get("route"),
                "therapy_sensitivity_overlap": ",".join(comp.get("therapy_sensitivity_overlap") or []),
                "route_agreement_across_arch": comp.get("route_agreement"),
                "return_on_reasoning_per_1k_tokens": ror_tokens,
                "return_on_reasoning_per_second": ror_latency,
                "return_on_reasoning_vs_cost_multiplier": ror_vs_single,
            }
        )

    for gene, mutation in cases:
        for arch in architectures:
            _append_row(gene, mutation, arch)

    for gene, mutation in debate_cases:
        _append_row(gene, mutation, "debate")
    return rows


def compute_efficiency_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Scatter-ready points: latency vs semantic accuracy, sized by tokens."""
    frontier = []
    for r in rows:
        if r.get("semantic_accuracy_composite") is None:
            continue
        frontier.append(
            {
                "query_id": r["query_id"],
                "architecture": r["architecture"],
                "x_latency_s": r["latency_s"],
                "y_semantic_accuracy": r["semantic_accuracy_composite"],
                "size_tokens": r["total_tokens"],
                "ror_per_1k_tokens": r["return_on_reasoning_per_1k_tokens"],
            }
        )
    return frontier


def write_ror_benchmark(md: Path | None = None) -> Path:
    md = md or metrics_dir()
    md.mkdir(parents=True, exist_ok=True)
    rows = compute_ror_rows(md)
    ingress = compute_ingress_by_role(md)
    frontier = compute_efficiency_frontier(rows)

    csv_path = md / "return_on_reasoning.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    ingress_csv = md / "blackboard_ingress_by_role.csv"
    if ingress:
        with ingress_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ingress[0].keys()))
            w.writeheader()
            w.writerows(ingress)

    # narrative summary for blackboard vs single
    by_arch: dict[str, list] = {}
    for r in rows:
        by_arch.setdefault(r["architecture"], []).append(r)
    summary = {}
    for arch, rs in by_arch.items():
        sem = [r["semantic_accuracy_composite"] for r in rs if r["semantic_accuracy_composite"] is not None]
        tok = [r["total_tokens"] for r in rs if r["total_tokens"]]
        summary[arch] = {
            "mean_semantic_accuracy": round(sum(sem) / len(sem), 3) if sem else None,
            "mean_total_tokens": round(sum(tok) / len(tok)) if tok else None,
            "mean_cost_multiplier_vs_single": round(
                sum(_f(r["cost_multiplier_vs_single"]) for r in rs) / len(rs), 2
            )
            if rs
            else None,
        }

    report = {
        "title": "Return on Reasoning (RoR) Benchmark",
        "narrative": (
            "Blackboard costs ~14× tokens vs single but may yield higher semantic accuracy "
            "on complex structural-rescue cases. RoR = semantic composite / cost."
        ),
        "architecture_summary": summary,
        "rows": rows,
        "efficiency_frontier": frontier,
        "blackboard_ingress_by_role": ingress,
    }
    manifest = _read_json(md / "run_manifest.json")
    if manifest.get("lora_loaded"):
        report["lora_ablation"] = {
            "lora_path": manifest.get("lora_path"),
            "lora_loaded": True,
            "note": "Single-agent runs use PeftModel when lora_loaded=true in run_manifest",
        }
    debate_rows = [r for r in rows if r.get("architecture") == "debate"]
    if debate_rows:
        report["debate_vs_blackboard"] = {
            "debate_mean_semantic": round(
                sum(_f(r["semantic_accuracy_composite"]) for r in debate_rows) / len(debate_rows), 3
            ),
            "blackboard_mean_semantic": summary.get("blackboard", {}).get("mean_semantic_accuracy"),
        }
    path = md / "ror_benchmark.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    return path
