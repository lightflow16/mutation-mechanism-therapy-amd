"""Gradio demo: select case -> structure + reasoning + evidence + rescue + metrics."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr

from src.config import load_config, metrics_dir, setup_env, shared_dir, get_target
from src import metrics
from src.pipeline import run_case
from src.structure import analyze_target


def list_cases():
    cfg = load_config()
    return [f"{t['gene']} {t['mutation']}" for t in cfg["targets"]]


def analyze(selected: str, architecture: str, live_api: bool, use_cache: bool):
    setup_env()
    metrics.set_metrics_dir(str(metrics_dir()))
    gene, mutation = selected.split(maxsplit=1)
    arch = architecture.lower()
    if arch not in ("single", "cot", "blackboard"):
        arch = "single"
    try:
        result = run_case(
            gene, mutation, architecture=arch,
            live_evidence=live_api, use_cached_trace=use_cache,
        )
    except Exception as e:
        empty = "", "", "", "", "", f"Error: {e}"
        return empty

    struct = result.get("structure", {})
    reasoning = result.get("reasoning", {})
    evidence = result.get("evidence", [])
    rescue = result.get("rescue")

    struct_text = json.dumps(struct, indent=2)
    reason_text = json.dumps(reasoning, indent=2, default=str)
    ev_text = json.dumps(evidence, indent=2)
    rescue_text = json.dumps(rescue, indent=2) if rescue else "N/A (GOF inhibitor path)"

    html = ""
    try:
        target = get_target(load_config(), gene, mutation)
        full = analyze_target(target, shared_dir() / "structures")
        html = full.get("render_html", "")
    except Exception:
        pass

    metrics_text = format_metrics()
    return struct_text, reason_text, ev_text, rescue_text, html, metrics_text


def format_metrics():
    calls = metrics_dir() / "calls.csv"
    phases = metrics_dir() / "phases.csv"
    parts = [f"summary: {metrics.summary()}"]
    if calls.exists():
        parts.append(calls.read_text().splitlines()[-5:])
    if phases.exists():
        parts.append(phases.read_text().splitlines()[-3:])
    return "\n".join(str(p) for p in parts)


def main():
    with gr.Blocks(title="Mutation -> Mechanism -> Therapy") as demo:
        gr.Markdown("# Mutation to Mechanism to Therapy (AMD MI300X / Track 2)")
        with gr.Row():
            case = gr.Dropdown(list_cases(), label="Demo case", value="EGFR L858R")
            arch = gr.Radio(["single", "cot", "blackboard"], value="blackboard", label="Architecture")
            live = gr.Checkbox(label="Live API fallback", value=False)
            cache = gr.Checkbox(label="Use cached trace (instant demo)", value=True)
        btn = gr.Button("Analyze", variant="primary")
        struct_out = gr.Textbox(label="Structure features", lines=8)
        reason_out = gr.Textbox(label="Reasoning output", lines=12)
        ev_out = gr.Textbox(label="Evidence", lines=6)
        rescue_out = gr.Textbox(label="Rescue branch", lines=6)
        viewer = gr.HTML(label="Structure viewer")
        metrics_out = gr.Textbox(label="Metrics (latest)", lines=8)
        btn.click(
            analyze,
            inputs=[case, arch, live, cache],
            outputs=[struct_out, reason_out, ev_out, rescue_out, viewer, metrics_out],
        )
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_PORT", "7860")))


if __name__ == "__main__":
    main()
