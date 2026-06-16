# Slide 1 — Problem
Precision oncology requires connecting **mutation → structure → mechanism → therapy**.
Manual KB/literature review is slow; structural context is often ignored.

# Slide 2 — Solution (Multimodal + MAS)
- AlphaFold numeric features + structure render (py3Dmol)
- CIViC/ClinVar/PubMed evidence (cached + live)
- Blackboard MAS: Planner → Experts → Critic → ConflictResolver → Decider
- LoRA-tuned Qwen2.5-VL-7B on MI300X (ROCm)

# Slide 3 — Three-pillar demo
1. **Direct Inhibition** (EGFR L858R, PIK3CA E545K) — small-molecule therapy
2. **Rational Rescue** (TP53 R175H) — ProteinMPNN + ThermoMPNN ddG
3. **Cross-Verification** — GPU DL ddG vs CPU PyRosetta (offline)

# Slide 4 — AMD story + productive throughput (not raw GPU %)
Open vendor-neutral stack on Instinct MI300X: transformers + ESMFold + Boltz + ThermoMPNN.
Native metrics (`src/metrics.py`): latency-to-decision, workflow density, egress tokens/GPU-s, weight-cache hit rate.
Live demo: open `metrics/workflow_trace_dashboard.html` — blackboard agent timeline + before/after vs single baseline.

# Slide 4b — Metrics that matter (judge narrative)
| Avoid | Prefer |
| --- | --- |
| Mean GPU % (hides idle) | **GPU productivity ratio** = gpu_active / gpu_attached |
| Raw tok/s alone | **Productive egress tok / GPU-s** |
| Single latency number | **Workflow density** = agent steps / decision time |
| Cold-start inference | **Weight cache hit rate** (sticky Qwen weights across 14 MAS steps) |

# Slide 5 — Return on Reasoning (RoR)
`return_on_reasoning.csv` + `ror_benchmark.json`: semantic accuracy / token cost / cost multiplier vs single.
Thesis + hackathon: open `workflow_trace_dashboard.html` (efficiency frontier scatter + ingress-by-role compaction baseline).

# Slide 6 — Results & future
Ablation: single / CoT / blackboard / debate × base / LoRA (Therapy F1, direction accuracy).
Future: RFdiffusion/BindCraft de novo binder design (out of live scope).

# Slide 7 — Literature + trust measurement
- **MOAlmanac** — multi-source oncology RAG grounding (CIViC/PubMed/ClinVar fusion)
- **MTBBench** — tumor-board multi-agent conflict resolution (blackboard Critic → Decider)
- **PFUA / tool grounding** — ThermoMPNN + ESMFold/Boltz numeric checks vs agent text (`HR_tool`, `HR_property`)
- **Measured hallucination** — six HR metrics + BVR on every trace (`hallucination_summary.json`); not Critic-only
- **Fold confidence separation** — LLM decision confidence vs pLDDT/pTM (`benchmark_confidence.csv`, ECE/Brier/AUC-PR)
