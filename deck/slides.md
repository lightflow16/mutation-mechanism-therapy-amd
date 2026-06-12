# Slide 1 — Problem
Precision oncology requires connecting **mutation → structure → mechanism → therapy**.
Manual KB/literature review is slow; structural context is often ignored.

# Slide 2 — Solution (Track 2 Multimodal + MAS)
- AlphaFold numeric features + structure render (py3Dmol)
- CIViC/ClinVar/PubMed evidence (cached + live)
- Blackboard MAS: Planner → Experts → Critic → ConflictResolver → Decider
- LoRA-tuned Qwen2.5-VL-7B on MI300X (ROCm)

# Slide 3 — Three-pillar demo
1. **Direct Inhibition** (EGFR L858R, PIK3CA E545K) — small-molecule therapy
2. **Rational Rescue** (TP53 R175H) — ProteinMPNN + ThermoMPNN ddG
3. **Cross-Verification** — GPU DL ddG vs CPU PyRosetta (offline)

# Slide 4 — AMD story
Open vendor-neutral stack on Instinct MI300X: vLLM + ESMFold + Boltz (`--no_kernels`) + ThermoMPNN.
No BioNeMo lock-in. Metrics: CPU/GPU time + ingress/egress/reasoning tokens per agent.

# Slide 5 — Results & future
Ablation: single / CoT / blackboard × base / LoRA (Therapy F1, direction accuracy).
Future: RFdiffusion/BindCraft de novo binder design (out of live scope).
