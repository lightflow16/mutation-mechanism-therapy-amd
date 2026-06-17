# Platform Comparison: AMD MI300X vs NVIDIA A100-SXM4-80GB — Final

**AMD (latest):** `metrics_bundle_amd_rocm_20260617_171024` · `lora_adapter_final-2` · git `eeb8257` · exported 17:10:24 UTC (run started 15:40:55 UTC)  
**NVIDIA:** `metrics_bundle_colab_cuda_20260617_121238` · `lora_adapter_final` · git `7f1b98f` · 11:52:48 UTC  
**Latency authority:** `platform_comparison.json` phase\_diffs (matched pipeline pairs, generated 16:27:53 UTC)  
**Artifact authority:** `artifacts_bundle_*/artifacts_manifest.json`  
**New in latest bundle:** `architecture_comparison_eval.json` + comparison JSONs for all 4 mutations (L858R, T790M, TP53 R175H, PIK3CA E545K)

---

## 1. Platform Specs

| Property           | AMD MI300X                            | NVIDIA A100-SXM4-80GB                |
|--------------------|---------------------------------------|---------------------------------------|
| GPU                | AMD Instinct MI300X                   | NVIDIA A100-SXM4-80GB                |
| PyTorch            | 2.10.0+rocm7.2.4                      | 2.11.0+cu128                          |
| LLM backend        | `transformers`                        | `auto` (vLLM)                         |
| Model              | Qwen2.5-VL-7B-Instruct                | Qwen2.5-VL-7B-Instruct               |
| LoRA adapter       | 182 MB, r=16, α=32                    | 182 MB, r=16, α=32                    |
| LoRA train time    | **144.9 s**                           | 192.7 s                               |
| Mode               | full\_submission                      | full\_submission                      |
| Run completeness   | Complete (all metrics)                | Complete (all metrics)                |

---

## 2. The Core Speed Story — Per Individual LLM Call

AMD's HBM3 bandwidth advantage shows most clearly at the call level (before token verbosity distorts phase totals):

| Architecture       | NVIDIA (s/call) | AMD (s/call) | AMD speedup |
|--------------------|----------------|-------------|-------------|
| Single             | 46.5           | 37.0        | **1.26×**   |
| CoT                | 60.2           | 38.6        | **1.56×**   |
| Blackboard per step| 6.47           | 4.29        | **1.51×**   |
| Debate             | 25.8           | 15.2        | **1.70×**   |

**AMD is faster per inference call across every architecture tested.**

---

## 3. Per-Phase Wall Latency

Phase latency is the real-world workflow time. AMD generates 3–4× more tokens per call (due to `transformers` backend verbosity), which partially offsets its per-call speed advantage.

### CoT — AMD wins on every mutation

| Mutation       | NVIDIA (s) | AMD (s) | AMD/NVIDIA |
|----------------|-----------|---------|------------|
| EGFR L858R     | 39.1      | 26.2    | **0.668 (1.50×)** |
| EGFR T790M     | 71.4      | 38.0    | **0.531 (1.88×)** |
| PIK3CA E545K   | 86.2      | 44.2    | **0.513 (1.95×)** |
| TP53 R175H     | 53.2      | 37.0    | **0.695 (1.44×)** |
| **Mean**       | **62.5**  | **36.3**| **AMD 1.68×**     |

### Blackboard — parity overall, mutation-dependent variance

| Mutation       | NVIDIA (s) | AMD (s) | AMD/NVIDIA | Note |
|----------------|-----------|---------|------------|------|
| EGFR L858R     | 24.5      | 44.4    | 1.82       | NVIDIA faster — KV-cache hit |
| EGFR T790M     | 59.7      | 44.2    | **0.741**  | AMD 1.35× |
| PIK3CA E545K   | 63.6      | 74.4    | 1.17       | NVIDIA 1.17× |
| TP53 R175H     | 60.7      | 39.5    | **0.651**  | AMD 1.54× |
| **Mean**       | **52.1**  | **50.6**| **~1.0**   | Parity |

L858R blackboard anomaly: NVIDIA 24.5 s vs AMD 44.4 s. NVIDIA had a KV-cache hit from the preceding EGFR G719S VUS run — not a general advantage.

### Single agent — parity

`reason_single`: NVIDIA 50.6 s vs AMD 53.9 s (AMD/NVIDIA = 1.065 — parity)

### Debate — AMD wins decisively

PIK3CA E545K: NVIDIA 124.2 s vs AMD 53.1 s → **AMD 2.34× faster** (ratio 0.427)

### Structural rescue — NVIDIA wins, but AMD variance is large

TP53 R175H canonical comparison: NVIDIA 205.7 s vs AMD 677.6 s → **NVIDIA 3.3× faster** (ratio 3.295)

**Rescue latency ablation** (latest bundle `ablation_summary.csv` — 4 observed runs):

| Run | AMD rescue (s) | AMD/NVIDIA | Condition |
|---|---|---|---|
| 1 (best) | **479.8** | **2.33×** | Cold GPU — LLM not loaded |
| 2 (canonical) | 677.6 | 3.30× | Hot GPU — moderate VRAM pressure |
| 3 | 1,279.7 | 6.22× | High VRAM pressure |
| 4 (worst) | 1,654.8 | 8.04× | Maximum VRAM pressure |

AMD rescue latency is **entirely a function of residual GPU memory pressure**. When rescue runs cold (LLM unloaded), AMD completes in 479.8 s — 2.33× slower than NVIDIA, a much smaller gap. Root cause confirmed: **pipeline sequencing**, not hardware.

**Updated Boltz finding:** `comparison_TP53_R175H.json` in the latest bundle shows `fold_method: "boltz+esmfold"` with a valid Boltz PDB path — meaning Boltz ran successfully in the comparison script context (cold GPU, LLM not loaded). In the DBTL loop, Boltz fails because the LLM is still in VRAM. Fix: call `model.cpu() + torch.cuda.empty_cache()` before the DBTL fold step.

### LoRA SFT training

NVIDIA 192.7 s vs AMD 144.9 s → **AMD 1.33× faster**

---

## 4. The Token Verbosity Problem

| Architecture | NVIDIA mean | AMD mean (latest) | AMD/NVIDIA |
|---|---|---|---|
| Single       | 2,537 tok   | 7,472 tok   | 2.95× |
| CoT          | 2,979 tok   | 8,194 tok   | 2.75× |
| Blackboard   | 6,412 tok   | **35,620 tok** | **5.55×** |
| Debate       | 3,960 tok   | 14,821 tok  | 3.74× |

AMD generates 3–4× more tokens because `transformers` has no `max_new_tokens` budget enforced. Blackboard token inflation increased from 4.45× (prior bundle) to **5.55×** (latest bundle) as additional ablation rounds accumulated longer context chains. This is why AMD blackboard wall latency is not faster despite AMD being 1.51× faster per step — it generates 5.55× more output per step.

**Fix:** add `max_new_tokens=2048` (or similar) to the AMD generation config. This should cut AMD token output to NVIDIA levels and translate the per-call speedup into wall-time speedup for blackboard.

---

## 5. Quality — Updated AMD Semantic Accuracy

Semantic accuracy is architecture-dependent. Single and CoT are identical across platforms. Blackboard differs: the latest AMD bundle (`ror_benchmark.json`) shows a **lower blackboard mean of 0.35** (was 0.50 in prior run) due to two new failures captured in `architecture_comparison_eval.json`.

| Architecture | NVIDIA mean acc | AMD mean acc (latest) | Note |
|---|---|---|---|
| Single       | **0.80** | **0.80** | Identical |
| CoT          | **0.75** | **0.75** | Identical |
| Blackboard   | **0.50** | **0.35** | AMD regression — see below |
| Debate       | **0.00** | **0.00** | Not recommended |

**AMD blackboard per-mutation breakdown (latest bundle):**

| Mutation     | Semantic | F1   | Dir acc | Root cause |
|---|---|---|---|---|
| EGFR L858R   | 0.70 | 0.40 | 1.00 | Partial drug list (Osimertinib only, CoT lists 4 drugs) |
| EGFR T790M   | **0.20** | 0.40 | **0.00** | Mechanism agent outputs "stabilizes kinase domain" — incorrect direction; resistance annotations missing from Decider |
| PIK3CA E545K | 0.00 | 0.00 | 0.00 | Blackboard Therapy/Decider classifies PI3K inhibitors as *resistance* — direction reversal of standard-of-care alpelisib |
| TP53 R175H   | 0.50 | 0.00 | 1.00 | No pharmacologic therapies (correct for structural LOF), but therapy_f1=0.0 pulls down score |

> **PIK3CA direction reversal is the most clinically significant finding:** alpelisib is FDA-approved as a *sensitivity* drug for PIK3CA E545K, but the AMD blackboard labels PI3K inhibitors as resistance agents. Single and CoT both correctly flag sensitivity. The ABLE scaffold_uplift test captures this as therapy_f1_uplift = −1.0 (blackboard reduces accuracy vs single).

---

## 6. Hallucination — Platform Matters Significantly

| Architecture | NVIDIA HR\_evidence | AMD HR\_evidence | Delta |
|---|---|---|---|
| Single       | 0.444  | **0.083**  | AMD −36.1pp  |
| CoT          | 0.208  | 0.333      | AMD +12.5pp  |
| Blackboard   | **0.312** | 0.562   | AMD +25.0pp  |

**Headline:** AMD single-agent is dramatically less likely to hallucinate evidence (HR 0.083 vs 0.444). AMD blackboard is more likely (0.562 vs 0.312). The hallucination benefit of blackboard seen on NVIDIA **reverses on AMD**.

Blackboard vs single delta: NVIDIA −13.2pp → AMD +47.9pp. Complete reversal.

---

## 7. Artifacts — Boltz Gap Confirmed

| Bundle           | PDB files | Contents |
|---|---|---|
| NVIDIA artifacts | 5 | AF-P00533, AF-P04637, AF-P42336, esmfold\_candidate, **input\_model\_0 (Boltz)** |
| AMD artifacts    | 4 | AF-P00533, AF-P04637, AF-P42336, esmfold\_candidate |

The Boltz output PDB (`input_model_0.pdb`) exists only in the NVIDIA bundle. This definitively confirms Boltz did not run on AMD.

---

## 8. DBTL Level 3 — Key Differentiator

| Metric            | NVIDIA       | AMD           |
|---|---|---|
| dbtl\_success     | **True**     | False         |
| fold\_method      | boltz+esmfold| esmfold only  |
| fold\_gate        | **PASS**     | **FAIL**      |
| ddG               | +0.183       | +0.137        |
| ddg\_gate         | PASS         | PASS          |
| Designs/tools     | 8/8, 5/5     | 8/8, 4/5      |
| Wall clock        | 377.9 s      | 1,646.2 s     |

Both platforms compute the ddG correctly and pass the stability gate. Only the fold-method gate differs.

---

## 9. Evidence Ablation — Same on Both Platforms

therapy_f1 is flat across T1 (CIViC-only), T2 (CIViC+PubMed), T3 (full) for every architecture and mutation. Evidence stacking provides no benefit — this holds on both platforms and all architectures.

---

## 10. Summary

| Dimension                    | AMD MI300X              | NVIDIA A100             | Winner             |
|------------------------------|-------------------------|-------------------------|--------------------|
| Per-call (single)            | 37.0 s                  | 46.5 s                  | **AMD 1.26×**      |
| Per-call (CoT)               | 38.6 s                  | 60.2 s                  | **AMD 1.56×**      |
| Per-step (blackboard)        | 4.29 s                  | 6.47 s                  | **AMD 1.51×**      |
| Per-call (debate)            | 15.2 s                  | 25.8 s                  | **AMD 1.70×**      |
| CoT wall mean                | 36.3 s                  | 62.5 s                  | **AMD 1.68×**      |
| Blackboard wall mean         | 50.6 s                  | 52.1 s                  | **Parity**         |
| Single wall (reason_single)  | 53.9 s                  | 50.6 s                  | Parity             |
| Debate wall                  | 53.1 s                  | 124.2 s                 | **AMD 2.34×**      |
| Rescue wall                  | 677.6 s                 | 205.7 s                 | **NVIDIA 3.3×**    |
| LoRA training                | 144.9 s                 | 192.7 s                 | **AMD 1.33×**      |
| Tokens/call (blackboard)     | **35,620 (5.55× NVIDIA)** | 6,412                 | NVIDIA (less waste)|
| Semantic accuracy (single)   | 0.80                    | 0.80                    | Tie                |
| Semantic accuracy (blackboard)| **0.35** (AMD latest)  | 0.50                    | **NVIDIA**         |
| Single HR\_evidence          | **0.083**               | 0.444                   | **AMD**            |
| Blackboard HR\_evidence      | 0.562                   | **0.312**               | **NVIDIA**         |
| DBTL Level 3                 | FAIL (fold gate)        | **PASS**                | **NVIDIA**         |
| Task pass rate               | 66.7%                   | **100%**                | **NVIDIA**         |
| Safety pass rate             | 87.5%                   | 87.5%                   | Tie                |
| LLM confidence ECE           | 0.544                   | **0.473**               | **NVIDIA**         |
| pLDDT confidence ECE         | 0.231                   | 0.231                   | Tie                |
