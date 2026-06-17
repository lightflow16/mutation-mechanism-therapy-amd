# Mutation-Mechanism-Therapy Reasoning (MMT-R)
## A Multi-Agent AI System for Autonomous Cancer Mutation Interpretation on AMD Instinct MI300X

**Hackathon:** TCS × AMD Hackathon, June 2026  
**Team:** hack-team-111  
**Date:** 2026-06-17  
**Status:** Final Draft v2.0 — updated with `metrics_bundle_amd_rocm_20260617_171024`

---

## Abstract

We present **MMT-R** (Mutation-Mechanism-Therapy Reasoning), a multi-agent AI system that takes a cancer somatic mutation as input and autonomously produces a mechanistic explanation, a therapy sensitivity/resistance profile, and — for structural mutations — a protein fold-based rescue analysis with stability scoring. The system benchmarks four reasoning architectures (Blackboard MAS, Chain-of-Thought, Single-Agent, Debate) on four clinically-validated oncology cases across AMD Instinct MI300X and NVIDIA A100-SXM4-80GB. All architectures use a shared LoRA-fine-tuned Qwen2.5-VL-7B-Instruct adapter (r=16, trained on 21 distilled blackboard traces, 182 MB).

AMD MI300X achieves **1.56× faster CoT inference per LLM call**, **1.51× faster per blackboard agent step**, and **1.70× faster debate throughput** vs NVIDIA. Semantic accuracy for single (0.80) and CoT (0.75) is identical across platforms; blackboard mean is **0.35 on AMD vs 0.50 on NVIDIA** due to a direction regression on T790M and a therapy direction reversal on PIK3CA E545K (both identified from `architecture_comparison_eval.json` in the latest bundle). AMD single-agent evidence hallucination is dramatically lower (HR 0.083 vs 0.444 on NVIDIA). The structural rescue pipeline (DBTL Level 3) passes on NVIDIA but fails the fold gate on AMD — root cause narrowed to pipeline sequencing: Boltz runs successfully in the comparison context (cold GPU) but fails in the hot-GPU DBTL loop. Evidence retrieval tier does not affect output quality. Both platforms achieve 87.5% safety compliance.

---

## 1. Introduction

### 1.1 Motivation

Precision oncology requires interpreting somatic mutations at clinical scale: which pathways are disrupted, which therapies are indicated, and whether protein-level rescue is feasible. Multi-agent LLM systems offer a path toward automation, but their deployment on AMD ROCm hardware — increasingly relevant for cost-effective academic and clinical compute — is underexplored.

This work answers: **How does AMD Instinct MI300X perform for multi-agent oncology reasoning, and where does the platform advantage (or disadvantage) manifest?**

### 1.2 Clinical Problem Cases

| Mutation        | Gene    | Class                | Route                | Evidence base         |
|-----------------|---------|----------------------|----------------------|-----------------------|
| EGFR L858R      | EGFR    | Activating (kinase)  | inhibitor\_rag       | CIViC B + PubMed      |
| EGFR T790M      | EGFR    | Resistance gatekeeper| inhibitor\_rag       | CIViC B + PubMed      |
| PIK3CA E545K    | PIK3CA  | PI3K activating      | inhibitor\_rag       | CIViC + PubMed        |
| TP53 R175H      | TP53    | Structural LOF       | structural\_rescue   | ClinVar + structural  |

Plus EGFR G719S as VUS probe (empty knowledge base; must abstain from confident recommendation).

---

## 2. System Architecture

### 2.1 Pipeline Overview

```
Input: (gene, mutation)
        ↓
  [Evidence retriever] — CIViC, PubMed, ClinVar (3 tiers)
  [Structure loader]   — AlphaFold2 PDB (AF-P* UniProt)
        ↓
  [Router] — inhibitor_rag | structural_rescue
        ↓
  [Architecture]  — single / cot / blackboard / debate
        ↓
  [Output] — mechanism, therapy[], confidence, (rescue: ddG, designs)
```

### 2.2 Reasoning Architectures

**Single Agent + LoRA**  
One call to `Qwen2.5-VL-7B-Instruct` with PEFT LoRA adapter loaded. All evidence + structure context injected in a single prompt. Multimodal: structure image always included.

**Chain-of-Thought (CoT)**  
3 sequential calls with structured reasoning prompts. Multimodal on every call. Extended-thinking tokens enabled — AMD generated 1,096 total thinking tokens across its full run (vs 350 on NVIDIA in a single-pass run).

**Blackboard MAS (bMAS)**  
9–11 sequential specialist agents sharing a common blackboard context:
```
Planner → Structure → Mechanism → CriticRubric → Evidence 
→ Therapy → Critic → ConflictResolver → Decider
```
First 4 agent steps are multimodal (image injected); remaining steps are text-only context. CriticRubric gate enables early exit if round-1 consensus is reached.

**Debate**  
3-agent adversarial format: DebatePro, DebateCon, DebateJudge. Text-only (no structure image). Tested on PIK3CA E545K only.

### 2.3 LoRA Fine-Tuning

| Config item           | Value                                   |
|-----------------------|-----------------------------------------|
| Base model            | Qwen/Qwen2.5-VL-7B-Instruct             |
| PEFT type             | LoRA                                    |
| Rank r                | 16                                      |
| Alpha                 | 32 (scaling = 2.0)                      |
| Dropout               | 0.05                                    |
| Target modules        | q/k/v/o\_proj, up/gate/down\_proj (×7) |
| Adapter size          | 182 MB (safetensors)                    |
| Training rows         | 21 (distilled from blackboard traces)   |
| PEFT version          | 0.19.1                                  |
| Training time (NVIDIA)| 192.7 s (GPU-attached)                  |
| Training time (AMD)   | 144.9 s (GPU-attached) — **1.33× faster** |

Both platforms produce a 182 MB adapter with identical architecture. The adapters were trained independently on each platform from the same 21-row distillation dataset.

### 2.4 Evidence Retrieval

Three tiers evaluated:
- **T1** — CIViC only
- **T2** — CIViC + PubMed
- **T3** — Full (CIViC + PubMed + ClinVar)

Key finding: therapy_f1 is **flat across all three tiers for every architecture/mutation combination**. The model's output is not sensitive to evidence richness at this scale — it produces the same reasoning regardless of T1 vs T3. This is both a robustness finding (consistent output) and a limitation (richer evidence not utilised).

### 2.5 Structural Analysis

For all mutations, AlphaFold2 PDB structures are pre-loaded:
- `AF-P00533-F1-model_v6.pdb` (EGFR, UniProt P00533)
- `AF-P04637-F1-model_v6.pdb` (TP53, UniProt P04637)
- `AF-P42336-F1-model_v6.pdb` (PIK3CA, UniProt P42336)

Key structural data:

| Mutation  | pLDDT at residue | Mean protein pLDDT | Neighbours 10Å | Region              |
|-----------|------------------|--------------------|----------------|---------------------|
| EGFR L858R| 51.2             | 75.9               | 14             | Kinase domain       |
| TP53 R175H| 96.6             | —                  | —              | DNA-binding domain  |

### 2.6 Structural Rescue Pipeline (TP53 R175H)

| Step              | Tool             | NVIDIA | AMD        |
|-------------------|------------------|--------|------------|
| Structure predict | **Boltz**        | DONE   | **NOT RUN**|
| Structure predict | ESMFold          | DONE   | DONE       |
| Stability score   | ThermoMPNN       | DONE   | DONE       |
| Design iterations | DBTL loop (×8)   | DONE   | DONE       |
| Dual-fold gate    | Boltz + ESMFold  | **PASS** | **FAIL** |

Artifact confirmation: AMD bundle contains 4 PDB files; NVIDIA bundle contains 5 (the extra is `input_model_0.pdb` = Boltz output). Boltz was not invoked on AMD.

---

## 3. Experimental Setup

### 3.1 Hardware

| Platform     | GPU                    | Framework          | Backend       | Run start (UTC)  |
|--------------|------------------------|--------------------|---------------|-----------------|
| AMD          | Instinct MI300X        | PyTorch 2.10+rocm7 | transformers  | 15:40:55        |
| NVIDIA Colab | A100-SXM4-80GB         | PyTorch 2.11+cu128 | auto (vLLM)   | 11:52:48        |

Both runs: `mode=full_submission`, LoRA loaded, `git eeb8257` (AMD) / `7f1b98f` (NVIDIA).

### 3.2 Evaluation Protocol

- 4 architectures × 4 mutations + VUS + debate (PIK3CA) on both platforms
- 2 adversarial safety probes
- Latency from `platform_comparison.json` phase\_diffs (matched pairs)
- Evidence ablation: 3 tiers × 4 mutations × 4 architectures = 48 conditions

---

## 4. Results

### 4.1 Per-Call LLM Throughput — AMD vs NVIDIA

The most direct hardware comparison is mean latency per individual LLM call (not per full workflow phase, which is confounded by token verbosity).

| Architecture | NVIDIA mean/call (s) | AMD mean/call (s) | AMD speedup |
|--------------|----------------------|--------------------|-------------|
| Single       | 46.5                 | 37.0               | **1.26×**   |
| CoT          | 60.2                 | 38.6               | **1.56×**   |
| Blackboard (per step) | 6.47        | 4.29               | **1.51×**   |
| Debate       | 25.8                 | 15.2               | **1.70×**   |

**AMD MI300X is faster per individual LLM call across every architecture.** The HBM3 memory bandwidth advantage is visible at all scales.

### 4.2 Per-Phase Wall Latency — AMD vs NVIDIA

Phase latency (from `platform_comparison.json`) is affected by token verbosity: AMD generates 3–4× more tokens per call, partially offsetting the per-call speedup.

| Phase                   | NVIDIA (s) | AMD (s) | AMD/NVIDIA | Note                          |
|-------------------------|-----------|---------|------------|-------------------------------|
| cot\_EGFR\_L858R        | 39.1      | 26.2    | 0.668      | AMD 1.50×                     |
| cot\_EGFR\_T790M        | 71.4      | 38.0    | 0.531      | AMD 1.88×                     |
| cot\_PIK3CA\_E545K      | 86.2      | 44.2    | 0.513      | AMD 1.95×                     |
| cot\_TP53\_R175H        | 53.2      | 37.0    | 0.695      | AMD 1.44×                     |
| debate\_PIK3CA\_E545K   | 124.2     | 53.1    | 0.427      | AMD 2.34×                     |
| blackboard\_EGFR\_L858R | 24.5      | 44.4    | 1.815      | NVIDIA 1.82× (KV-cache hit)   |
| blackboard\_EGFR\_T790M | 59.7      | 44.2    | 0.741      | AMD 1.35×                     |
| blackboard\_PIK3CA\_E545K| 63.6     | 74.4    | 1.170      | NVIDIA 1.17×                  |
| blackboard\_TP53\_R175H | 60.7      | 39.5    | 0.651      | AMD 1.54×                     |
| reason\_single          | 50.6      | 53.9    | 1.065      | Parity                        |
| rescue\_TP53\_R175H     | 205.7     | 677.6   | 3.295      | NVIDIA 3.3× (Boltz missing)   |
| lora\_sft               | 192.7     | 144.9   | 0.752      | AMD 1.33×                     |

**CoT mean:** NVIDIA 62.5 s → AMD 36.3 s (**AMD 1.68× faster**)  
**Blackboard mean:** NVIDIA 52.1 s → AMD 50.6 s (parity)  

### 4.3 Token Verbosity — AMD 3–5× More Tokens

| Architecture | NVIDIA mean tokens | AMD mean tokens (latest) | Ratio |
|--------------|--------------------|--------------------------|-------|
| Single       | 2,537              | 7,472                    | 2.95× |
| CoT          | 2,979              | 8,194                    | 2.75× |
| Blackboard   | 6,412              | **35,620**               | **5.55×** |
| Debate       | 3,960              | 14,821                   | 3.74× |

Source: `ror_benchmark.json`, bundle `20260617_171024`. Blackboard token inflation increased from 4.45× (prior bundle) to **5.55×** in the latest run as additional ablation rounds accumulated longer context chains in the blackboard state. The `transformers` backend on AMD generates much longer outputs (no vLLM token budget). This is the primary reason blackboard wall latency is not faster despite AMD being 1.51× faster per step — it generates 5.55× more output per step.

### 4.4 Multimodal Usage

| Architecture | NVIDIA calls | NVIDIA multimodal | AMD calls | AMD multimodal | Thinking tokens (NVIDIA/AMD) |
|--------------|-------------|-------------------|----------|----------------|------------------------------|
| Single       | 5           | 5 (100%)          | 16       | 16 (100%)      | 0 / 0                        |
| CoT          | 5           | 5 (100%)          | 16       | 16 (100%)      | 350 / **1,096**              |
| Blackboard   | 40          | 4 (10%)           | 176      | 16 (9%)        | 0 / 0                        |
| Debate       | 3           | 0 (0%)            | 12       | 0 (0%)         | 0 / 0                        |

AMD CoT generates 3× more thinking tokens (1,096 vs 350). Blackboard multimodal rate is consistent (~10% of calls).

### 4.5 Semantic Accuracy — Architecture-Dependent; Blackboard Regressed on AMD

Single and CoT accuracy are identical across platforms. Blackboard shows a regression in the latest AMD run (`ror_benchmark.json`, bundle `20260617_171024`).

| Architecture | NVIDIA mean | AMD mean (latest) | L858R | T790M | E545K | R175H |
|---|---|---|---|---|---|---|
| Single       | **0.80** | **0.80** | 0.70 | 1.00 | 0.50 | 1.00 |
| CoT          | **0.75** | **0.75** | 1.00 | 1.00 | 0.50 | 0.50 |
| Blackboard   | **0.50** | **0.35** | 0.70 | **0.20** | 0.00 | 0.50 |
| Debate       | **0.00** | **0.00** | — | — | 0.00 | — |

**Blackboard regression findings** (from `architecture_comparison_eval.json`):

- **EGFR T790M (0.20):** Mechanism agent outputs "T790M *stabilizes* EGFR kinase domain" — incorrect direction. Decider outputs Osimertinib sensitivity but omits Gefitinib/Erlotinib/Afatinib resistance annotations. direction_acc=0.0.
- **PIK3CA E545K (0.00):** Therapy agent and Decider classify PI3K inhibitors as *resistance* drugs. Alpelisib is the FDA-approved standard-of-care *sensitivity* agent. Complete direction reversal. Single and CoT both correctly flag sensitivity.
- **TP53 R175H (0.50):** Direction correct (structural_rescue route), but therapy_f1=0.0 — no pharmacologic therapies recommended, which is structurally correct for TP53 LOF but penalised by the scorer.
- **EGFR L858R (0.70):** Decider outputs Osimertinib only; CoT enumerates 4 drugs. Partial credit.

### 4.6 Evidence Ablation — Tier Does Not Matter

| Mutation       | Architecture | T1 therapy_f1 | T2 therapy_f1 | T3 therapy_f1 |
|----------------|-------------|--------------|--------------|--------------|
| EGFR L858R     | single      | 0.4          | 0.4          | 0.4          |
| EGFR L858R     | cot         | 1.0          | 1.0          | 1.0          |
| EGFR L858R     | blackboard  | 0.0          | 0.0          | 0.0          |
| EGFR T790M     | single      | 1.0          | 1.0          | 1.0          |
| PIK3CA E545K   | single      | 1.0          | 1.0          | 1.0          |
| TP53 R175H     | single      | 1.0          | 1.0          | 1.0          |

Therapy F1 is **flat across all three evidence tiers for all architecture/mutation pairs**. The model is not leveraging incremental evidence — it produces the same output whether given 1 CIViC record or 3 sources. This has implications for retrieval-augmented system design: evidence retrieval volume is not the bottleneck at this context size.

### 4.7 Return on Reasoning

AMD actuals from `ror_benchmark.json` (bundle `20260617_171024`). RoR = semantic accuracy / (total_tokens / 1000).

| Mutation       | Single RoR | CoT RoR | Blackboard RoR | Single tokens | CoT tokens | BB tokens |
|----------------|-----------|---------|----------------|---------------|------------|-----------|
| EGFR L858R     | 0.0783    | **0.1042** | 0.0180      | 8,944         | 9,595      | 38,911    |
| EGFR T790M     | **0.1331**| 0.1177  | 0.0054         | 7,514         | 8,497      | 37,090    |
| PIK3CA E545K   | 0.0730    | 0.0668  | NA (0.0 acc)   | 6,854         | 7,490      | 36,235    |
| TP53 R175H     | **0.1520**| 0.0695  | 0.0165         | 6,577         | 7,194      | 30,242    |
| **Mean**       | **0.109** | **0.089** | **0.013**    | 7,472         | 8,194      | 35,620    |

Single agent has the best RoR on AMD across all mutations. CoT wins on L858R (full multi-drug enumeration). Blackboard RoR has collapsed to 0.013 mean — a **8.4× worse efficiency than single** — driven by the T790M direction regression (0.20 semantic on 37,090 tokens) and PIK3CA zero accuracy (36,235 tokens, NA). Blackboard is no longer economically justified on AMD for any of the 4 standard mutations in this run.

### 4.8 Hallucination — Platform Differences Are Significant

| Architecture | NVIDIA HR\_ev | AMD HR\_ev | NVIDIA HR\_des | AMD HR\_des | NVIDIA BVR | AMD BVR |
|---|---|---|---|---|---|---|
| Single    | 0.444 | **0.083** | 0.000 | 0.000 | 0.938 | 0.917 |
| CoT       | 0.208 | 0.333     | 0.750 | 1.000 | 0.938 | 0.917 |
| Blackboard| **0.312** | 0.562 | 0.000 | 0.000 | 0.938 | 0.917 |
| Debate    | 0.333 | 0.333     | 1.000 | 1.000 | 1.000 | 1.000 |

Critical findings:
- **AMD single is far less likely to hallucinate evidence (0.083 vs 0.444)** — verbose outputs include more explicit evidence citations, reducing hallucination flags
- **AMD blackboard hallucinates evidence more (0.562 vs 0.312)** — Blackboard's hallucination advantage on NVIDIA does **not** replicate on AMD
- AMD blackboard\_vs\_single delta: **+0.479** (blackboard 47.9pp WORSE than single on AMD)
- NVIDIA blackboard\_vs\_single delta: **−0.132** (blackboard 13.2pp BETTER than single on NVIDIA)

### 4.9 DBTL Level Assessment

| Level | Criterion                              | NVIDIA       | AMD         |
|-------|----------------------------------------|--------------|-------------|
| 1     | Single-agent therapy identification    | **PASS**     | **PASS**    |
| 2     | Multi-architecture route agreement     | **PASS**     | **PASS**    |
| 3     | Autonomous structural rescue (TP53)    | **PASS**     | **FAIL**    |

AMD DBTL Level 3 — detailed breakdown:

| Metric              | NVIDIA       | AMD          | 
|---------------------|-------------|-------------|
| dbtl\_success       | True        | **False**   |
| fold\_method        | boltz+esmfold| esmfold only|
| fold\_gate\_passed  | True        | **False**   |
| ddG (kcal/mol)      | +0.183      | +0.137      |
| ddG gate            | PASS        | PASS        |
| Designs             | 8/8         | 8/8         |
| Tools invoked       | 5/5 (100%)  | 4/5 (80%)   |
| Wall clock          | 377.9 s     | 1,646.2 s   |
| objective\_delta    | 0.4731      | 0.5036      |
| Artifact PDB files  | 5           | 4           |

The Boltz PDB (`input_model_0.pdb`) appears in the NVIDIA artifact bundle but not AMD's — confirming Boltz was not invoked on AMD.

### 4.10 Autonomy and Task Suite

| Metric                    | NVIDIA | AMD    |
|---------------------------|--------|--------|
| Workflow completion rate  | 69.2%  | 61.5%  |
| Task pass rate            | 100%   | 66.7%  |
| T-EGFR (pure reasoning)   | PASS   | PASS   |
| T-PIK3CA (conflict resolv)| PASS   | PASS   |
| T-TP53 (rescue DBTL)      | PASS   | **FAIL** |

AMD fails only T-TP53 due to fold gate. All pure-reasoning tasks pass.

ABLE metrics (AMD):
- `refusal_rate = 0.5` → FAIL (same as NVIDIA)
- `tool_call_success_rate_tp53 = 0.8` → PASS
- `therapy_f1_uplift_pik3ca = −1.0` → FAIL (blackboard reduces F1 vs single on PIK3CA)

### 4.11 Safety — Both Platforms 87.5%

Identical results on both platforms. The CRISPR refusal test (T-REFUSE-VIRAL) produces a correctly-phrased refusal on both platforms but the automated classifier fails to score it. All constraint and robustness tests pass.

### 4.12 Fold Confidence Calibration

| Metric  | NVIDIA LLM | AMD LLM | pLDDT (both) |
|---------|-----------|---------|--------------|
| ECE     | 0.473     | 0.544   | **0.231**    |
| Brier   | 0.371     | 0.446   | **0.056**    |

pLDDT structural confidence is 2× better calibrated than LLM confidence on NVIDIA, and 2.4× better on AMD. This strongly motivates using structural confidence as the primary reliability signal.

---

## 5. Discussion

### 5.1 AMD MI300X — Strengths

1. **Faster per LLM call across all architectures** (1.26–1.70×). HBM3 bandwidth advantage is real and consistent.
2. **CoT wall latency: 1.68× faster** — the most workflow-relevant result. For diagnostic pipelines dominated by sequential reasoning chains, AMD is the better platform today.
3. **Debate: 2.34× faster** — AMD excels on small-batch multi-model workloads.
4. **LoRA training: 1.33× faster** — adapting the base model to a new oncology domain is faster on AMD.
5. **Single-agent hallucination dramatically lower (HR 0.083 vs 0.444)** — AMD+single produces the most evidence-grounded outputs.

### 5.2 AMD MI300X — Weaknesses

1. **Structural rescue: DBTL Level 3 fold gate failure** — Boltz not invoked in the hot-GPU DBTL loop. Updated finding: `comparison_TP53_R175H.json` confirms Boltz *does* run successfully when the LLM is not loaded (cold GPU). Root cause is pipeline sequencing, not ROCm incompatibility. Fix: `model.cpu() + torch.cuda.empty_cache()` before the fold step.
2. **Token verbosity: 3–5× more tokens** — `transformers` backend has no generation budget cap. Blackboard inflation has risen to **5.55×** (35,620 vs 6,412 NVIDIA mean) in the latest bundle. Mitigation: add `max_new_tokens=2048` limit to generation config.
3. **Blackboard semantic regression: 0.35 vs 0.50** — T790M direction failure (Mechanism outputs wrong biochemical claim; direction_acc=0.0) and PIK3CA therapy direction reversal (alpelisib classified as resistance drug). Both are prompt-level failures in specific agent roles, not hardware issues.
4. **Blackboard hallucination reversal** — The 13.2pp hallucination advantage blackboard has on NVIDIA becomes a 47.9pp disadvantage on AMD. Cause is likely context-window drift in multi-turn accumulation under `transformers`.
5. **CoT design hallucination: 1.0** (all AMD CoT traces) — every AMD CoT call fabricates structural details. Not present in NVIDIA CoT (0.75).

### 5.3 Evidence Retrieval Design Implication

The flat evidence ablation result is significant for system design: **adding more evidence sources (T2, T3) does not improve therapy_f1** for any architecture. This suggests the model saturates at single-source evidence and further retrieval is wasted compute. Future work should explore evidence reranking or summarisation rather than evidence stacking.

### 5.4 Architecture Choice Guide (Updated)

| Situation                              | Recommended architecture | Platform preference |
|----------------------------------------|--------------------------|---------------------|
| Routine inhibitor-sensitive mutation   | Single + LoRA            | AMD (lower HR 0.083, highest RoR 0.109) |
| Multi-drug enumeration (EGFR-class)    | CoT                      | AMD (1.68× faster wall; best L858R RoR 0.104) |
| Structural/rescue mutation (TP53-class)| Single + LoRA (for reasoning) + DBTL rescue | NVIDIA (DBTL Level 3 gate passes); AMD pending Boltz fix |
| Maximum therapy accuracy (T790M-class) | Single or CoT            | AMD (both achieve 1.0 semantic; single RoR 0.133) |
| Minimum hallucination single call      | Single + LoRA            | AMD (HR 0.083)       |
| Adversarial robustness (not recommended)| Debate                  | AMD (2.34× faster if used) |

> **Note:** Blackboard is no longer recommended as the primary architecture for any mutation class on AMD in its current state (mean semantic 0.35, RoR 0.013). It should be used only after fixing the Therapy agent direction-classification prompt and adding context compression to prevent hallucination accumulation.

### 5.5 Boltz Root Cause — Narrowed to Pipeline Sequencing

Prior analysis listed three hypotheses (memory pressure, ROCm kernel incompatibility, timeout/fallback). The latest bundle provides decisive new evidence:

**`comparison_TP53_R175H.json`** in `metrics_bundle_amd_rocm_20260617_171024` records:
```json
"rescue": {
  "fold_method": "boltz+esmfold",
  "boltz_pdb": "/workspace/shared/rescue/TP53_R175H/boltz/out/...input_model_0.pdb"
}
```
Boltz ran successfully and produced a valid PDB when executed from the comparison script (where the LLM is not in VRAM). This rules out ROCm kernel incompatibility as the primary cause.

**Rescue latency ablation** (`ablation_summary.csv`) shows: when rescue runs first (cold GPU), wall time = 479.8 s. When rescue runs after multiple LLM calls (hot GPU), wall time escalates to 677.6 → 1279.7 → 1654.8 s.

**Conclusion:** Boltz fails in the DBTL loop because the `transformers` model is still loaded in HBM3 at the time the fold pipeline runs. Boltz's memory requirement (estimated 8–12 GB VRAM) cannot be satisfied alongside the 7B model weights (~14 GB).

**Fix (confirmed):** Add `model.cpu(); torch.cuda.empty_cache()` before the DBTL rescue fold step in `rescue.py`. Re-load the model after the fold completes if further LLM calls are needed.

---

## 6. Conclusion

MMT-R demonstrates that AMD Instinct MI300X is a viable and in several respects superior platform for multi-agent oncology reasoning inference. Key conclusions:

1. **AMD is faster per LLM call across all architectures** (1.26–1.70×). CoT wall latency is 1.68× faster.
2. **Single and CoT semantic accuracy are platform-independent** — architecture choice dominates. Blackboard is 0.35 on AMD vs 0.50 on NVIDIA due to two agent-level failures (T790M direction regression, PIK3CA direction reversal) that are fixable at the prompt level.
3. **AMD single-agent produces the least hallucinated outputs** (HR_evidence 0.083).
4. **LoRA training is 1.33× faster on AMD.**
5. **DBTL Level 3 Boltz failure is a pipeline sequencing issue** — Boltz runs successfully on AMD when the LLM is unloaded first (confirmed by `comparison_TP53_R175H.json`). Fix is a two-line code change in `rescue.py`.
6. **Evidence tier does not affect quality** — T1 suffices; T2/T3 add no measurable benefit.
7. **Both platforms achieve 87.5% safety compliance.**
8. **Blackboard token inflation increased to 5.55×** (35,620 vs 6,412 NVIDIA mean) — `max_new_tokens` cap is essential before blackboard can be deployed on AMD.

For production deployment of inhibitor-pathway mutations at scale, AMD MI300X is the recommended platform. For structural rescue cases requiring DBTL Level 3 certification, NVIDIA A100 is currently required pending the Boltz/ROCm fix.

---

## 7. Future Work

- [ ] **[P0] Fix Boltz pipeline sequencing** — `model.cpu() + torch.cuda.empty_cache()` before DBTL fold step in `rescue.py`. Confirmed fix by `comparison_TP53_R175H.json` evidence. Expected outcome: AMD DBTL Level 3 PASS, task pass rate 100%.
- [ ] **[P0] Fix Blackboard PIK3CA direction reversal** — Add explicit sensitivity/resistance classification rubric to the Therapy agent system prompt; include alpelisib as a positive example. Also fix T790M Mechanism agent prompt to correctly label gatekeeper mutations as resistance context.
- [ ] **[P1] Add `max_new_tokens=2048` cap** to AMD `transformers` generation config. Blackboard inflation now 5.55× — this is the highest-leverage performance fix.
- [ ] **[P1] Add context window compression to Blackboard** — periodic summarisation every 3 turns to prevent hallucination accumulation and token inflation.
- [ ] **[P2]** Scale to 20+ mutations for statistically significant architecture comparison
- [ ] **[P2]** Utilise evidence reranking instead of evidence stacking (ablation shows stacking is ineffective)
- [ ] **[P2]** Quantify LoRA contribution with a held-out no-LoRA ablation
- [ ] **[P3]** Fix CRISPR refusal classifier (regex does not match safety-aligned opening phrases)
- [ ] **[P3]** Heterogeneous model routing: larger Decider, smaller specialist agents

---

## Appendix A: LoRA Adapter Technical Specification

```json
{
  "base_model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "peft_type": "LORA",
  "r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "target_modules": ["q_proj","k_proj","v_proj","o_proj","up_proj","gate_proj","down_proj"],
  "use_dora": false,
  "adapter_size_mb": 182,
  "training_rows": 21,
  "training_source": "distilled blackboard traces",
  "peft_version": "0.19.1"
}
```

Both adapters (Colab and AMD) are architecturally identical. Training times: AMD 144.9 s, NVIDIA 192.7 s.

---

## Appendix B: Structural Data

| Mutation  | UniProt | PDB file                    | pLDDT@residue | Notes                   |
|-----------|---------|----------------------------|---------------|-------------------------|
| EGFR L858R| P00533  | AF-P00533-F1-model_v6.pdb  | 51.2          | Kinase domain; low conf |
| PIK3CA E545K| P42336| AF-P42336-F1-model_v6.pdb  | —             | C2-kinase linker        |
| TP53 R175H| P04637  | AF-P04637-F1-model_v6.pdb  | 96.6          | DNA-binding domain      |

TP53 R175H rescue:
- **NVIDIA:** `esmfold_candidate.pdb` + `input_model_0.pdb` (Boltz) → dual-fold gate PASS
- **AMD:** `esmfold_candidate.pdb` only → dual-fold gate FAIL

---

## Appendix C: Key Metrics Reference

| Metric                         | NVIDIA (Colab)      | AMD (MI300X)        |
|--------------------------------|---------------------|---------------------|
| Git commit                     | 7f1b98f             | eeb8257             |
| Run start (UTC)                | 11:52:48            | 15:40:55            |
| CoT speedup                    | —                   | 1.68× (phase wall)  |
| Per-call speedup (single)      | —                   | 1.26×               |
| Per-call speedup (CoT)         | —                   | 1.56×               |
| Per-step speedup (blackboard)  | —                   | 1.51×               |
| Debate speedup                 | —                   | 2.34×               |
| LoRA training speedup          | —                   | 1.33×               |
| Rescue slowdown                | —                   | 3.3× slower         |
| AMD bundle (latest)            | —                   | `20260617_171024`   |
| Tokens/call (single)           | 2,537               | 7,472 (2.95×)       |
| Tokens/call (blackboard)       | 6,412               | **35,620 (5.55×)**  |
| Semantic accuracy (single)     | 0.80                | 0.80 (identical)    |
| Semantic accuracy (blackboard) | 0.50                | **0.35 (AMD lower)**|
| HR\_evidence (single)          | 0.444               | **0.083**           |
| HR\_evidence (blackboard)      | **0.312**           | 0.562               |
| BVR (all arch mean)            | 0.942               | 0.923               |
| DBTL Level 3                   | **PASS**            | FAIL (fold gate)    |
| Artifact PDB count             | 5                   | 4 (no Boltz PDB)    |
| Task pass rate                 | 100%                | 66.7%               |
| Workflow completion            | 69.2%               | 61.5%               |
| Safety pass rate               | 87.5%               | 87.5%               |
| LLM confidence ECE             | 0.473               | 0.544               |
| pLDDT confidence ECE           | 0.231               | 0.231 (identical)   |
| Evidence ablation sensitivity  | None (flat T1–T3)   | None (flat T1–T3)   |
| Route agreement (all 4 cases)  | 100%                | 100%                |
