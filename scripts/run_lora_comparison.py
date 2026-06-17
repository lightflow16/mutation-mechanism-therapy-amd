#!/usr/bin/env python3
"""Full 2×4 evaluation matrix: base model vs LoRA fine-tuned × 4 architectures.

Runs every combination of:
  model_variant : base | lora
  architecture  : single | cot | blackboard | debate

for each case in demo_cases (+ debate_cases), scores against gold labels, and
produces a side-by-side comparison report.

Outputs (all under metrics/local/ unless --out-dir is given):
  lora_comparison.csv          — per-cell therapy_f1 / direction_accuracy / delta
  lora_comparison_summary.txt  — ASCII table for quick inspection
  lora_comparison.json         — full results for downstream analysis
  trace_{gene}_{mut}_{arch}_base.json   — traces for base-model runs
  trace_{gene}_{mut}_{arch}_lora.json   — traces for LoRA runs

Usage:
  python scripts/run_lora_comparison.py --lora-path shared/lora_adapter_final
  python scripts/run_lora_comparison.py          # auto-discovers adapter
  python scripts/run_lora_comparison.py --skip-base   # only run lora cells
  python scripts/run_lora_comparison.py --skip-lora   # only run base cells
  python scripts/run_lora_comparison.py --dry-run     # print plan, no LLM calls
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, metrics_dir, setup_env, shared_dir
from src.pipeline import (
    extract_target_reasoning,
    extract_therapies_from_reasoning,
    run_mutation_comparison,
)

ARCHITECTURES = ["single", "cot", "blackboard", "debate"]

# ── metric helpers (self-contained so this script needs no train/ import) ──────

_THERAPY_ALIASES: dict[str, str] = {
    "gefitinib": "gefitinib",
    "erlotinib": "erlotinib",
    "afatinib": "afatinib",
    "osimertinib": "osimertinib",
    "alpelisib": "alpelisib",
}


def _norm(names: list[str]) -> set[str]:
    out: set[str] = set()
    for x in names:
        k = x.lower().strip().split()[0]
        out.add(_THERAPY_ALIASES.get(k, k))
    return out


def therapy_f1(pred: list[str], gold: list[str]) -> float:
    p, g = _norm(pred), _norm(gold)
    if not g:
        return 1.0 if not p else 0.0
    prec = len(p & g) / len(p) if p else 0.0
    rec = len(p & g) / len(g)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def direction_accuracy(pred: dict, gold: dict) -> float:
    ps = set((pred.get("therapy") or {}).get("sensitivity") or [])
    pr = set((pred.get("therapy") or {}).get("resistance") or [])
    gs = set((gold.get("therapy") or {}).get("sensitivity") or [])
    gr = set((gold.get("therapy") or {}).get("resistance") or [])
    ok = (bool(ps & gs) or not gs) and (bool(pr & gr) or not gr)
    return 1.0 if ok else 0.0


# ── LoRA adapter auto-discovery ─────────────────────────────────────────────────

def _find_lora_adapter(cfg: dict) -> str | None:
    paths = cfg.get("paths", {})
    for key in ("lora_adapter_final", "lora_ckpts"):
        p = paths.get(key)
        if p and Path(p).is_dir() and any(Path(p).iterdir()):
            return str(p)
    local = shared_dir(cfg) / "lora_adapter_final"
    if local.is_dir():
        weights = list(local.glob("adapter_model.*")) + list(local.glob("*.safetensors"))
        if weights:
            return str(local)
    return None


# ── gold label loader ───────────────────────────────────────────────────────────

def _gold_label(gene: str, mutation: str) -> dict | None:
    p = ROOT / "data" / "cases" / f"{gene}_{mutation}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("target_reasoning", {})


# ── core runner ─────────────────────────────────────────────────────────────────

def _load_from_existing_traces(
    gene: str,
    mutation: str,
    architectures: list[str],
    *,
    out_dir: Path,
    model_tag: str,
) -> dict[str, dict]:
    """Reuse trace files already written by the main submission run.

    The full submission writes trace_{gene}_{mutation}_{arch}.json for every
    architecture it runs.  When --reuse-lora-traces is passed we read those
    files directly instead of re-running the LoRA model, saving GPU time.
    A tagged copy (trace_*_{model_tag}.json) is written for record-keeping.
    """
    by_arch: dict[str, dict] = {}
    print(f"\n  [{model_tag.upper()}] {gene} {mutation} — loading from existing traces")
    for arch in architectures:
        src = out_dir / f"trace_{gene}_{mutation}_{arch}.json"
        if src.is_file():
            result = json.loads(src.read_text())
            by_arch[arch] = result
            tagged = out_dir / f"trace_{gene}_{mutation}_{arch}_{model_tag}.json"
            tagged.write_text(src.read_text())
            print(f"    {arch:<12} loaded from {src.name}")
        else:
            print(f"    {arch:<12} ✗ trace not found at {src} — will be scored as missing")
    return by_arch


def _run_one_variant(
    gene: str,
    mutation: str,
    architectures: list[str],
    *,
    lora_path: str | None,
    out_dir: Path,
    model_tag: str,
    dry_run: bool,
) -> dict[str, dict]:
    """Run all requested architectures for one (gene, mutation, model_variant).

    Returns {arch: result_dict} with reasoning already extracted.
    Saves traces to out_dir/trace_{gene}_{mutation}_{arch}_{model_tag}.json so
    base and lora runs never overwrite each other.
    """
    print(f"\n  [{model_tag.upper()}] {gene} {mutation} — architectures: {architectures}")
    if dry_run:
        print("    (dry-run: skipping LLM calls)")
        return {}

    t0 = time.perf_counter()
    run = run_mutation_comparison(
        gene,
        mutation,
        architectures=architectures,
        lora_path=lora_path,
        live_evidence=False,
        use_cached_trace=False,
    )
    elapsed = round(time.perf_counter() - t0, 1)
    print(f"    done in {elapsed}s")

    by_arch = run.get("architectures", {})

    # Save tagged traces so base and lora files coexist.
    for arch, result in by_arch.items():
        tagged_path = out_dir / f"trace_{gene}_{mutation}_{arch}_{model_tag}.json"
        tagged_path.write_text(json.dumps(result, indent=2, default=str))

    return by_arch


# ── scoring ──────────────────────────────────────────────────────────────────────

def _score_result(
    result: dict | None,
    gold: dict | None,
    *,
    gene: str,
    mutation: str,
    architecture: str,
    model_tag: str,
) -> dict[str, Any]:
    base_row: dict[str, Any] = {
        "gene": gene,
        "mutation": mutation,
        "architecture": architecture,
        "model_variant": model_tag,
        "therapy_f1": None,
        "direction_acc": None,
        "sensitivity_predicted": [],
        "resistance_predicted": [],
        "confidence": None,
        "status": "ok",
    }
    if result is None:
        base_row["status"] = "missing_result"
        return base_row
    if gold is None:
        base_row["status"] = "missing_gold"
        return base_row

    reasoning = extract_target_reasoning(result)
    pred_sens, pred_res = extract_therapies_from_reasoning(reasoning)
    gold_sens, gold_res = extract_therapies_from_reasoning(gold)

    f1 = therapy_f1(pred_sens + pred_res, gold_sens + gold_res)
    dacc = direction_accuracy(reasoning, gold)
    base_row.update(
        {
            "therapy_f1": round(f1, 3),
            "direction_acc": round(dacc, 3),
            "sensitivity_predicted": pred_sens,
            "resistance_predicted": pred_res,
            "confidence": reasoning.get("confidence"),
        }
    )
    return base_row


# ── report formatting ────────────────────────────────────────────────────────────

def _build_summary_table(rows: list[dict]) -> str:
    """Build an ASCII comparison table grouped by (gene, mutation)."""
    # Index: (gene, mut, arch) → {base: row, lora: row}
    idx: dict[tuple, dict[str, dict]] = {}
    for r in rows:
        key = (r["gene"], r["mutation"], r["architecture"])
        idx.setdefault(key, {})[r["model_variant"]] = r

    cases = sorted({(r["gene"], r["mutation"]) for r in rows})
    archs = ARCHITECTURES

    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("  BASE MODEL vs LORA FINE-TUNED — Therapy F1 / Direction Accuracy")
    lines.append("=" * 90)

    for gene, mut in cases:
        gold = _gold_label(gene, mut)
        gold_sens = (gold or {}).get("therapy", {}).get("sensitivity", []) if gold else []
        lines.append(f"\n  {gene} {mut}  (gold sensitivity: {gold_sens})")
        lines.append(f"  {'Architecture':<14} {'Base F1':>8} {'LoRA F1':>8} {'Δ F1':>8}  "
                     f"{'Base DirAcc':>11} {'LoRA DirAcc':>11} {'Δ DirAcc':>9}")
        lines.append("  " + "-" * 82)
        for arch in archs:
            key = (gene, mut, arch)
            base = idx.get(key, {}).get("base", {})
            lora = idx.get(key, {}).get("lora", {})

            bf1   = base.get("therapy_f1")
            lf1   = lora.get("therapy_f1")
            bda   = base.get("direction_acc")
            lda   = lora.get("direction_acc")

            def _fmt(v: float | None) -> str:
                return f"{v:.3f}" if v is not None else "  N/A "

            def _delta(a: float | None, b: float | None) -> str:
                if a is None or b is None:
                    return "   N/A"
                d = b - a
                sign = "+" if d >= 0 else ""
                return f"{sign}{d:.3f}"

            lines.append(
                f"  {arch:<14} {_fmt(bf1):>8} {_fmt(lf1):>8} {_delta(bf1, lf1):>8}  "
                f"{_fmt(bda):>11} {_fmt(lda):>11} {_delta(bda, lda):>9}"
            )

    lines.append("\n" + "=" * 90)

    # Aggregate means
    def _mean(tag: str, metric: str) -> float | None:
        vals = [r[metric] for r in rows if r["model_variant"] == tag and r[metric] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    bf1_mean  = _mean("base", "therapy_f1")
    lf1_mean  = _mean("lora", "therapy_f1")
    bda_mean  = _mean("base", "direction_acc")
    lda_mean  = _mean("lora", "direction_acc")

    def _delta(a: float | None, b: float | None) -> str:
        if a is None or b is None:
            return "N/A"
        d = b - a
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.3f}"

    lines.append(f"  {'MEAN (all cells)':<14} {str(bf1_mean):>8} {str(lf1_mean):>8} "
                 f"{_delta(bf1_mean, lf1_mean):>8}  "
                 f"{str(bda_mean):>11} {str(lda_mean):>11} "
                 f"{_delta(bda_mean, lda_mean):>9}")
    lines.append("=" * 90)

    return "\n".join(lines)


# ── delta rows for CSV ───────────────────────────────────────────────────────────

def _build_delta_rows(rows: list[dict]) -> list[dict]:
    """One row per (gene, mutation, architecture) with base, lora, and delta columns."""
    idx: dict[tuple, dict[str, dict]] = {}
    for r in rows:
        key = (r["gene"], r["mutation"], r["architecture"])
        idx.setdefault(key, {})[r["model_variant"]] = r

    out: list[dict] = []
    for (gene, mut, arch), variants in sorted(idx.items()):
        base = variants.get("base", {})
        lora = variants.get("lora", {})

        def _delta(k: str) -> float | None:
            bv, lv = base.get(k), lora.get(k)
            if bv is None or lv is None:
                return None
            return round(lv - bv, 3)

        out.append(
            {
                "gene": gene,
                "mutation": mut,
                "architecture": arch,
                "base_therapy_f1": base.get("therapy_f1"),
                "lora_therapy_f1": lora.get("therapy_f1"),
                "delta_therapy_f1": _delta("therapy_f1"),
                "base_direction_acc": base.get("direction_acc"),
                "lora_direction_acc": lora.get("direction_acc"),
                "delta_direction_acc": _delta("direction_acc"),
                "base_confidence": base.get("confidence"),
                "lora_confidence": lora.get("confidence"),
                "base_sensitivity": "|".join(base.get("sensitivity_predicted") or []),
                "lora_sensitivity": "|".join(lora.get("sensitivity_predicted") or []),
                "base_resistance": "|".join(base.get("resistance_predicted") or []),
                "lora_resistance": "|".join(lora.get("resistance_predicted") or []),
                "base_status": base.get("status", "missing"),
                "lora_status": lora.get("status", "missing"),
            }
        )
    return out


# ── main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run base vs LoRA fine-tuned comparison across all 4 architectures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--lora-path", default=None,
        help="Path to LoRA adapter directory (auto-discovered if omitted).",
    )
    ap.add_argument(
        "--architectures", nargs="+",
        default=ARCHITECTURES,
        choices=ARCHITECTURES,
        metavar="ARCH",
        help="Subset of architectures to run (default: all 4).",
    )
    ap.add_argument(
        "--cases", nargs="+", default=None, metavar="GENE_MUT",
        help="Subset of cases to run, e.g. EGFR_L858R PIK3CA_E545K (default: all demo_cases).",
    )
    ap.add_argument(
        "--skip-base", action="store_true",
        help="Skip base-model runs (use when base traces already exist).",
    )
    ap.add_argument(
        "--skip-lora", action="store_true",
        help="Skip LoRA runs (score base model only).",
    )
    ap.add_argument(
        "--reuse-lora-traces", action="store_true",
        help=(
            "Load lora results from existing trace_{gene}_{mut}_{arch}.json files "
            "written by the main submission run instead of re-running the LoRA model. "
            "Use this when called from run_full_submission to avoid duplicate GPU work."
        ),
    )
    ap.add_argument(
        "--out-dir", default=None,
        help="Directory to write outputs (default: metrics/local/).",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print plan and skip all LLM calls.",
    )
    args = ap.parse_args()

    setup_env()
    cfg = load_config()
    pcfg = cfg.get("pipeline", {})

    out_dir = Path(args.out_dir) if args.out_dir else metrics_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── resolve cases ──────────────────────────────────────────────────────────
    if args.cases:
        cases: list[tuple[str, str]] = []
        for raw in args.cases:
            parts = raw.split("_", 1)
            if len(parts) != 2:
                print(f"[warn] Cannot parse case '{raw}' — expected GENE_MUTATION format, skipping.")
                continue
            cases.append((parts[0], parts[1]))
    else:
        demo = [tuple(x) for x in pcfg.get("demo_cases", [])]
        debate_extra = {tuple(x) for x in pcfg.get("debate_cases", [])} - set(demo)
        cases = demo + sorted(debate_extra)  # type: ignore[assignment]

    archs: list[str] = args.architectures

    # ── resolve LoRA path ──────────────────────────────────────────────────────
    lora_path: str | None = args.lora_path or _find_lora_adapter(cfg)
    if not args.skip_lora and lora_path is None:
        print(
            "[error] No LoRA adapter found.  Either:\n"
            "  • Run 'python train/lora_sft.py' first to train one, or\n"
            "  • Pass --lora-path <dir> explicitly, or\n"
            "  • Use --skip-lora to run base-only scoring."
        )
        sys.exit(1)

    # ── print plan ─────────────────────────────────────────────────────────────
    n_cells = len(cases) * len(archs) * (0 + (not args.skip_base) + (not args.skip_lora))
    print(f"\n{'='*60}")
    print(f"  LoRA Comparison Matrix")
    print(f"{'='*60}")
    print(f"  Cases       : {[f'{g}_{m}' for g, m in cases]}")
    print(f"  Architectures: {archs}")
    print(f"  Model variants: "
          + (", ".join(filter(None, [
              None if args.skip_base else "base",
              None if args.skip_lora else f"lora ({lora_path})",
          ]))))
    print(f"  Total LLM runs: {n_cells}")
    print(f"  Output dir  : {out_dir}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("  [dry-run] No LLM calls will be made.\n")

    # ── run all cells ──────────────────────────────────────────────────────────
    all_score_rows: list[dict] = []

    variants: list[tuple[str, str | None]] = []
    if not args.skip_base:
        variants.append(("base", None))
    if not args.skip_lora:
        variants.append(("lora", lora_path))

    for model_tag, lp in variants:
        print(f"\n{'─'*60}")
        print(f"  Running model_variant = {model_tag.upper()}")
        print(f"{'─'*60}")
        for gene, mut in cases:
            gold = _gold_label(gene, mut)
            if gold is None:
                print(f"  [warn] No gold label found for {gene} {mut} — will score as N/A")

            # When called from run_full_submission the main live matrix has already
            # run all architectures with LoRA and saved traces.  Reuse those instead
            # of running the LoRA model a second time.
            if model_tag == "lora" and args.reuse_lora_traces:
                by_arch = _load_from_existing_traces(
                    gene, mut, archs,
                    out_dir=out_dir,
                    model_tag=model_tag,
                )
            else:
                by_arch = _run_one_variant(
                    gene, mut, archs,
                    lora_path=lp,
                    out_dir=out_dir,
                    model_tag=model_tag,
                    dry_run=args.dry_run,
                )

            for arch in archs:
                result = by_arch.get(arch)
                row = _score_result(
                    result, gold,
                    gene=gene, mutation=mut,
                    architecture=arch, model_tag=model_tag,
                )
                all_score_rows.append(row)
                if not args.dry_run:
                    status = f"F1={row['therapy_f1']}  DirAcc={row['direction_acc']}"
                    print(f"    {arch:<12} {status}  [{row['status']}]")

    # ── write outputs ──────────────────────────────────────────────────────────
    delta_rows = _build_delta_rows(all_score_rows)
    summary_text = _build_summary_table(all_score_rows)

    # CSV
    csv_path = out_dir / "lora_comparison.csv"
    if delta_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(delta_rows[0].keys()))
            w.writeheader()
            w.writerows(delta_rows)

    # Summary text
    summary_path = out_dir / "lora_comparison_summary.txt"
    summary_path.write_text(summary_text)

    # Full JSON
    json_path = out_dir / "lora_comparison.json"
    json_path.write_text(json.dumps(
        {
            "lora_path": lora_path,
            "architectures": archs,
            "cases": [f"{g}_{m}" for g, m in cases],
            "score_rows": all_score_rows,
            "delta_rows": delta_rows,
        },
        indent=2,
        default=str,
    ))

    print(f"\n{summary_text}")
    print(f"\nOutputs written to {out_dir}:")
    print(f"  {csv_path.name}")
    print(f"  {summary_path.name}")
    print(f"  {json_path.name}")


if __name__ == "__main__":
    main()
