"""Patch sde_project_bMAS llm_integration to vLLM + metrics; oncology expert pool."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from src import metrics
from src.config import load_config
from src.llm_client import call_llm

BMAS_ROOT = Path(__file__).resolve().parents[1] / "external" / "sde_project_bMAS"
if str(BMAS_ROOT) not in sys.path:
    sys.path.insert(0, str(BMAS_ROOT))


ONCOLOGY_EXPERTS = [
    ("Structure", "structure", "Interpret numeric AlphaFold features only; cite pLDDT/region verbatim."),
    ("Mechanism", "mechanism", "Link variant to pathway effects using structure + literature."),
    ("Evidence", "evidence", "Summarize CIViC/ClinVar items; note sensitivity vs resistance."),
    ("Therapy", "therapy", "Recommend therapies with citations for the disease context."),
]


def _endpoint_for(role: str) -> tuple[str, str]:
    cfg = load_config()
    ep = cfg.get("serving", {}).get("endpoints", {})
    mapping = {
        "planner": "planner",
        "structure": "mechanism",
        "mechanism": "mechanism",
        "evidence": "mechanism",
        "therapy": "reasoner",
        "critic": "mechanism",
        "conflict_resolver": "mechanism",
        "decider": "reasoner",
    }
    key = mapping.get(role, "reasoner")
    e = ep.get(key, ep.get("reasoner", {}))
    return e.get("base_url", "http://localhost:8000/v1"), e.get("model", "qwen2.5-vl-7b")


def patch_bmas_llm() -> None:
    """Monkey-patch bMAS call_llm to route through vLLM OpenAI API."""
    import llm_integration.api as api

    def patched_call_llm(prompt, model_name=None, temperature=0.7, system_prompt=None):
        role = model_name or "agent"
        base_url, model = _endpoint_for(role if role in _endpoint_for.__code__.co_varnames else "reasoner")
        t0 = time.perf_counter()
        resp = call_llm(
            prompt,
            base_url=base_url,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            agent_role=str(model_name or "agent"),
            round_idx=0,
            label=f"bmas_{model_name or 'agent'}",
        )
        metrics.log_llm_call(
            str(model_name or "agent"), model, 0,
            resp["metadata"]["prompt_tokens"],
            resp["content"],
            resp["metadata"]["completion_tokens"],
            time.perf_counter() - t0,
        )
        return resp

    api.call_llm = patched_call_llm


def run_bmas_experiment(problem: str, max_rounds: int = 2) -> dict[str, Any]:
    """Run bMAS experiment_runner with vLLM backend."""
    patch_bmas_llm()
    from bMAS.experiment_runner.run_experiment import run_single_experiment

    with metrics.phase("bmas_experiment", model="bMAS"):
        return run_single_experiment(problem=problem, max_rounds=max_rounds, enable_logging=False)


def build_oncology_problem(target: dict, structure: dict, evidence: list[dict]) -> str:
    return (
        f"Oncology case: {target['gene']} {target['mutation']} class={target['class']} "
        f"pathway={target['pathway']} disease={target.get('disease_context')}\n"
        f"Structure: {json.dumps({k: v for k, v in structure.items() if k not in ('render_html', 'pdb_path')})}\n"
        f"Evidence: {json.dumps(evidence)}\n"
        "Produce mechanism + therapy JSON with citations."
    )
