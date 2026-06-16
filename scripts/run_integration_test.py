#!/usr/bin/env python3
"""CPU integration test: Tier-2 headline features without live GPU inference."""
from __future__ import annotations

import json
import os
import shutil
import sys
import csv
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BACKUP_BUNDLE = ROOT.parent / "backups" / "colab_artifacts_20260616_045708" / "metrics_bundle_colab_cuda_20260615_231801"


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _fail(msg: str) -> None:
    print(f"  ✗ {msg}")
    raise SystemExit(1)


def test_llm_text_capture() -> None:
    from src.llm_client import split_completion
    from src import metrics
    import tempfile
    import os

    raw = (
        "<think>Step 1: check evidence.</think>"
        '{"mechanism": "test", "therapy": {"sensitivity": ["DrugA"]}}'
    )
    reasoning, output = split_completion(raw)
    if "Step 1" not in reasoning:
        _fail("reasoning text not extracted")
    if "DrugA" not in output:
        _fail("output text not extracted")

    with tempfile.TemporaryDirectory() as td:
        metrics.set_metrics_dir(td)
        os.environ["METRICS_DIR"] = td
        metrics.log_llm_call(
            "TestAgent", "test-model", 1, 10, raw, 20, 0.5,
            gene="EGFR", mutation="L858R", architecture="test",
            prompt_text="user prompt here", system_prompt="system here",
        )
        row = json.loads(Path(td, "llm_calls.jsonl").read_text().strip().splitlines()[-1])
        for key in ("prompt_text", "system_prompt", "reasoning_text", "output_text", "completion_text"):
            if key not in row:
                _fail(f"llm_calls.jsonl missing {key}")
    _ok("LLM prompt/reasoning/output text capture")


def test_cot_unwrap() -> None:
    from src.pipeline import extract_target_reasoning, extract_therapies_from_reasoning

    nested = {
        "reasoning": {
            "target_reasoning": {
                "target_reasoning": {
                    "mechanism": "test",
                    "therapy": {"sensitivity": ["Osimertinib"], "resistance": []},
                }
            }
        }
    }
    tr = extract_target_reasoning(nested)
    sens, _ = extract_therapies_from_reasoning(tr)
    if "Osimertinib" not in sens:
        _fail("CoT nested unwrap failed")
    _ok("CoT nested target_reasoning unwrap")


def test_distill_from_traces() -> None:
    from train.build_dataset import build_dataset_from_traces

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "lora_train.jsonl"
        path, n = build_dataset_from_traces(out_path=out)
        if n < 1:
            _fail(f"distill produced {n} rows (expected >= 1 blackboard trace)")
        lines = path.read_text().strip().splitlines()
        row = json.loads(lines[0])
        if "messages" not in row and "instruction" not in row:
            _fail("distill row missing training fields")
    _ok(f"build_dataset_from_traces ({n} rows from blackboard traces)")


def test_vus_abstention() -> None:
    from src.pipeline import extract_target_reasoning, extract_therapies_from_reasoning, run_mutation_comparison
    from src.variant_router import route_variant

    trace = json.loads((ROOT / "data/traces/EGFR_G719S_single.json").read_text())
    routing = route_variant(trace["target"], trace["structure"], trace["evidence"])
    if routing.get("allow_confident_therapy"):
        _fail("VUS router should block confident therapy for G719S")
    run = run_mutation_comparison("EGFR", "G719S", architectures=["single"], use_cached_trace=True, live_evidence=False)
    result = run["architectures"].get("single")
    if not result:
        _fail("EGFR G719S cached trace missing")
    tr = extract_target_reasoning(result)
    sens, res = extract_therapies_from_reasoning(tr)
    if sens or res:
        _fail(f"VUS case should abstain from therapy; got sens={sens} res={res}")
    _ok("EGFR G719S VUS abstention (§20)")


def test_debate_cache() -> None:
    from src.pipeline import extract_target_reasoning, extract_therapies_from_reasoning, run_mutation_comparison

    run = run_mutation_comparison(
        "PIK3CA", "E545K", architectures=["debate"], use_cached_trace=True, live_evidence=False
    )
    result = run["architectures"].get("debate")
    if not result:
        _fail("PIK3CA E545K debate trace missing")
    debate_trace = (result.get("reasoning") or {}).get("debate_trace") or []
    if len(debate_trace) < 3:
        _fail(f"debate_trace expected Pro/Con/Judge; got {len(debate_trace)} steps")
    tr = extract_target_reasoning(result)
    sens, res = extract_therapies_from_reasoning(tr)
    if not any("alpelisib" in s.lower() for s in sens + res):
        _fail(f"debate therapy missing Alpelisib: sens={sens} res={res}")
    _ok(f"debate architecture ({len(debate_trace)}-step trace, Alpelisib)")


def test_eval_and_trust(md: Path) -> None:
    from src import metrics
    from src.agent_autonomy_eval import run as run_autonomy
    from src.hallucination_eval import write_report as write_hallucination
    from src.fold_confidence_eval import write_benchmark as write_fold_confidence

    metrics.set_metrics_dir(str(md))
    os.environ["METRICS_DIR"] = str(md)

    import train.eval as eval_mod

    eval_mod.main()

    debate_path = md / "debate_eval.json"
    vus_path = md / "vus_eval.json"
    if not debate_path.exists():
        _fail("debate_eval.json not written")
    debate = json.loads(debate_path.read_text())
    if not any(r.get("status") == "ok" and (r.get("therapy_f1") or 0) > 0 for r in debate):
        _fail(f"debate eval failed: {debate}")
    if not vus_path.exists():
        _fail("vus_eval.json not written")
    vus = json.loads(vus_path.read_text())
    if not all(r.get("status") == "ok" for r in vus):
        _fail(f"VUS eval failed: {vus}")

    write_hallucination()
    write_fold_confidence()
    from src.extended_thinking_ablation import write_reports as write_extended

    write_extended()
    if not (md / "extended_thinking_ablation.csv").exists() and not (md / "extended_thinking_summary.json").exists():
        _fail("extended_thinking reports not written")
    autonomy = run_autonomy(md, live_probes=False)
    for key in ("task_suite", "able_metrics", "dbtl_level3_tp53", "task_pass_rate"):
        if key not in autonomy:
            _fail(f"autonomy_report missing {key}")
    if not (md / "task_suite.csv").exists() or not (md / "able_metrics.csv").exists():
        _fail("task_suite.csv or able_metrics.csv not written")
    with (md / "tevv_lite.csv").open() as f:
        n_tevv = sum(1 for _ in csv.DictReader(f))
    if n_tevv < 5:
        _fail(f"tevv_lite expected >=5 rows, got {n_tevv}")
    _ok(f"eval + trust (debate F1={debate[0].get('therapy_f1')}, VUS ok, tasks={autonomy.get('task_pass_rate')}, tevv={n_tevv})")


def main() -> None:
    print("=== Tier-2 integration test (CPU) ===\n")

    test_llm_text_capture()
    test_cot_unwrap()
    test_distill_from_traces()
    test_vus_abstention()
    test_debate_cache()

    with tempfile.TemporaryDirectory() as td:
        md = Path(td) / "metrics"
        md.mkdir()
        if BACKUP_BUNDLE.is_dir():
            for p in BACKUP_BUNDLE.glob("trace_*.json"):
                shutil.copy2(p, md / p.name)
            _ok(f"seeded metrics from Colab bundle ({len(list(md.glob('trace_*.json')))} traces)")
        else:
            for p in (ROOT / "data" / "traces").glob("*.json"):
                arch = p.stem.rsplit("_", 1)[-1]
                gene_mut = p.stem[: -(len(arch) + 1)]
                shutil.copy2(p, md / f"trace_{gene_mut}_{arch}.json")
            _ok("seeded metrics from data/traces (no Colab bundle found)")

        test_eval_and_trust(md)

    print("\n=== All Tier-2 integration checks passed ===")


if __name__ == "__main__":
    main()
