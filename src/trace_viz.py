"""Self-contained HTML trace timeline for live demo (no external dashboard deps)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import metrics_dir
from src.productive_metrics import (
    _f,
    _read_llm_calls,
    _read_phases,
    _read_system,
    build_agent_trace,
    compute_before_after,
    compute_productive_rows,
)
from src.ror_analysis import compute_efficiency_frontier, compute_ingress_by_role, compute_ror_rows


def _bar_color(agent: str) -> str:
    return {
        "Planner": "#6366f1",
        "Structure": "#0ea5e9",
        "Mechanism": "#14b8a6",
        "Evidence": "#22c55e",
        "Therapy": "#eab308",
        "Critic": "#f97316",
        "ConflictResolver": "#a855f7",
        "Decider": "#ef4444",
        "CoT": "#64748b",
        "SingleAgent": "#334155",
    }.get(agent or "", "#94a3b8")


def _scatter_svg(frontier: list[dict], width: int = 720, height: int = 360) -> str:
    if not frontier:
        return "<p>No RoR scatter data yet — run full submission + eval first.</p>"
    colors = {"single": "#64748b", "cot": "#38bdf8", "blackboard": "#a855f7"}
    xs = [_f(p["x_latency_s"]) for p in frontier]
    ys = [_f(p["y_semantic_accuracy"]) for p in frontier]
    max_x = max(xs) or 1
    max_y = 1.0
    dots = ""
    for p in frontier:
        x = 60 + (width - 90) * (_f(p["x_latency_s"]) / max_x)
        y = height - 40 - (height - 70) * (_f(p["y_semantic_accuracy"]) / max_y)
        c = colors.get(p.get("architecture"), "#94a3b8")
        dots += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{c}" opacity="0.85">'
        dots += f'<title>{p.get("query_id")} {p.get("architecture")} lat={p.get("x_latency_s")} acc={p.get("y_semantic_accuracy")}</title></circle>'
    return f"""
    <svg width="{width}" height="{height}" style="background:#1e293b;border-radius:8px">
      <text x="12" y="20" fill="#94a3b8" font-size="12">Efficiency frontier: Latency → vs Semantic accuracy ↑</text>
      <line x1="50" y1="{height-40}" x2="{width-20}" y2="{height-40}" stroke="#475569"/>
      <line x1="50" y1="30" x2="50" y2="{height-40}" stroke="#475569"/>
      {dots}
    </svg>"""


def generate_trace_html(out_dir: Path | None = None) -> Path:
    md = out_dir or metrics_dir()
    llm = _read_llm_calls(md / "llm_calls.jsonl")
    phases = _read_phases(md / "phases.csv")
    sys_rows = _read_system(md / "system_samples.csv")
    prod_rows = compute_productive_rows(llm, phases, sys_rows)
    before_after = compute_before_after(prod_rows)
    ror_rows = compute_ror_rows(md)
    frontier = compute_efficiency_frontier(ror_rows)
    ingress = compute_ingress_by_role(md)

    traces = []
    for r in prod_rows:
        if r["architecture"] == "blackboard" and r["query_id"] not in ("", "unknown"):
            traces.append(build_agent_trace(llm, r["query_id"], "blackboard"))

    cards_html = ""
    for r in prod_rows:
        cards_html += f"""
        <div class="card">
          <h3>{r.get('query_id')} · {r.get('architecture')}</h3>
          <ul>
            <li><b>Latency-to-decision:</b> {r.get('latency_to_decision_s')} s</li>
            <li><b>Productive egress / GPU-s:</b> {r.get('productive_egress_tokens_per_gpu_s')}</li>
            <li><b>GPU productivity ratio:</b> {r.get('gpu_productivity_ratio')} (active/attached)</li>
            <li><b>Workflow density:</b> {r.get('workflow_density_steps_per_s')} steps/s</li>
            <li><b>Agent steps:</b> {r.get('n_agent_steps')} · models: {r.get('heterogeneous_models')}</li>
            <li><b>Weight cache hit rate:</b> {r.get('weight_cache_hit_rate')} (sticky weights, no cold reload)</li>
            <li><b>Round-2 savings if early consensus:</b> {r.get('early_exit_savings_pct_if_round1_consensus')}%</li>
          </ul>
        </div>"""

    trace_blocks = ""
    for tr in traces:
        total = tr["total_latency_s"] or 1
        bars = ""
        for n in tr["nodes"]:
            w = max(4, 100 * _f(n.get("latency_s")) / total)
            bars += f"""
            <div class="bar" style="width:{w}%;background:{_bar_color(n.get('agent'))}"
                 title="{n.get('agent')} r{n.get('round')} · {n.get('latency_s')}s · {n.get('egress_tokens')} tok">
              {n.get('agent')}
            </div>"""
        trace_blocks += f"""
        <section>
          <h2>Trace · {tr['query_id']} · blackboard ({tr['n_nodes']} steps, {tr['total_latency_s']}s)</h2>
          <div class="timeline">{bars}</div>
        </section>"""

    ba_rows = ""
    for row in before_after[:20]:
        ba_rows += f"<tr><td>{row['query_id']}</td><td>{row['metric']}</td><td>{row.get('baseline_single')}</td><td>{row.get('blackboard_mas')}</td></tr>"

    ror_rows_html = ""
    for r in ror_rows:
        ror_rows_html += (
            f"<tr><td>{r['query_id']}</td><td>{r['architecture']}</td>"
            f"<td>{r.get('semantic_accuracy_composite')}</td><td>{r.get('total_tokens')}</td>"
            f"<td>{r.get('cost_multiplier_vs_single')}×</td>"
            f"<td>{r.get('return_on_reasoning_per_1k_tokens')}</td>"
            f"<td>{r.get('return_on_reasoning_vs_cost_multiplier')}</td></tr>"
        )

    ingress_rows = ""
    for row in ingress:
        ingress_rows += (
            f"<tr><td>{row['agent_role']}</td><td>{row['n_calls']}</td>"
            f"<td>{row['ingress_tokens']}</td><td>{row['egress_tokens']}</td>"
            f"<td>{row.get('ingress_waste_estimate')}</td></tr>"
        )

    scatter = _scatter_svg(frontier)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>RoR · Workflow Density Dashboard</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#0f172a;color:#e2e8f0}}
.card{{background:#1e293b;border-radius:8px;padding:12px 16px;margin:8px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}}
.timeline{{display:flex;height:48px;border-radius:6px;overflow:hidden;border:1px solid #334155}}
.bar{{display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:600;color:#fff;overflow:hidden;white-space:nowrap}}
table{{border-collapse:collapse;width:100%;margin-top:12px}}
td,th{{border:1px solid #334155;padding:6px 10px;text-align:left}}
th{{background:#1e293b}}
.note{{color:#94a3b8;font-size:14px;margin-bottom:16px}}
.legend span{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px}}
</style></head><body>
<h1>Return on Reasoning · Infrastructure Dashboard</h1>
<p class="note"><b>Hackathon demo + thesis documentation.</b> Native metrics only — no LangGraph/Phoenix.
Blackboard ~14× token cost vs single; RoR quantifies semantic fidelity per 1k tokens.</p>
<h2>Efficiency frontier (Latency vs Semantic accuracy)</h2>
<p class="legend"><span style="background:#64748b"></span>single
<span style="background:#38bdf8"></span>cot
<span style="background:#a855f7"></span>blackboard</p>
{scatter}
<h2>Return on Reasoning (RoR)</h2>
<table><tr><th>Case</th><th>Arch</th><th>Semantic acc</th><th>Tokens</th><th>Cost vs single</th><th>RoR / 1k tok</th><th>RoR / cost×</th></tr>{ror_rows_html}</table>
<h2>Blackboard ingress by role (compaction ROI baseline)</h2>
<table><tr><th>Agent</th><th>Calls</th><th>Ingress</th><th>Egress</th><th>Ingress waste est.</th></tr>{ingress_rows}</table>
<h2>Productive throughput</h2>
<div class="grid">{cards_html}</div>
{trace_blocks}
<h2>Before &amp; After · single vs blackboard</h2>
<table><tr><th>Case</th><th>Metric</th><th>Baseline single</th><th>Blackboard MAS</th></tr>{ba_rows}</table>
</body></html>"""
    out = md / "workflow_trace_dashboard.html"
    out.write_text(html, encoding="utf-8")
    return out
