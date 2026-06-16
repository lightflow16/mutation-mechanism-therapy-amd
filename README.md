# Mutation → Mechanism → Therapy

Structure-aware, multimodal, multi-agent reasoning copilot for precision oncology on **AMD Instinct MI300X / ROCm**, with an optional **Google Colab A100** comparison arm.

Demo cases: **EGFR L858R**, **PIK3CA E545K** (inhibitor path) · **TP53 R175H** (structural rescue path).

## Clone

```bash
git clone --depth 1 https://github.com/lightflow16/mutation-mechanism-therapy-amd.git
cd mutation-mechanism-therapy-amd
```

Open **`01_run_pipeline.ipynb`** and run top-to-bottom.

| Platform | Start directory | Persistent storage |
|----------|-----------------|-------------------|
| **AMD AI Notebook** | `/workspace` | `/workspace/shared/` |
| **Google Colab** | `/content` | `./shared/` + `./metrics/colab/` (session-local) |

## One-time setup

```bash
pip install -r requirements.txt openai   # NEVER pip install torch on AMD
bash scripts/setup_external.sh
python scripts/verify_gpu.py
```

Paths are auto-configured via `src.config.configure_paths()` in the notebook.

## Instant demo (no GPU, no vLLM)

Cached blackboard traces replay in ~1s:

```python
from src.pipeline import run_case
run_case("EGFR", "L858R", architecture="blackboard", use_cached_trace=True)
```

## Full run (all flows — Colab + AMD)

`run_full_submission()` always runs on **every session**:

| Step | What runs |
|------|-----------|
| Live matrix | single + cot + blackboard × EGFR / PIK3CA / TP53 |
| Cached baseline | same 9 combos from `data/traces/` (latency compare) |
| Per-mutation compare | route + therapy overlap across architectures |
| TP53 rescue | ThermoMPNN → MPNN → ESMFold + Boltz (once per mutation) |
| Eval | therapy F1 / direction acc from saved traces |
| Metrics export | `.tgz` with all CSVs, JSONL, traces, comparisons |

```bash
bash scripts/setup_external.sh
python train/lora_sft.py   # optional; or --train-lora
PYTHONPATH=. python scripts/run_full_submission.py --lora-path /workspace/shared/lora_adapter_final
```

**Metrics bundle includes:** `calls.csv`, `phases.csv`, `llm_calls.jsonl`, `system_samples.csv`, `ablation_summary.csv`, `ablation_results.csv`, `architecture_comparison.json`, `architecture_metrics.csv`, `platform_summary.json`, `trace_*.json`, `comparison_*.json`, `run_manifest.json`.

**Colab:** section 4 downloads `metrics_bundle_colab_cuda_*.tgz`.  
**AMD:** copy `/workspace/shared/metrics_bundle_amd_rocm_*.tgz`.

### NVIDIA (Colab) vs AMD comparison

After running on both platforms:

```bash
PYTHONPATH=. python scripts/compare_platforms.py colab_bundle.tgz amd_bundle.tgz
```

Writes `platform_comparison.json` + `.csv` with per-phase CPU/GPU/token deltas and architecture rollups.

**AMD:** transformers only. **Colab:** same stack.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| **`01_run_pipeline.ipynb`** | Main driver: setup → cached demo → GPU steps |
| **`00_env_check.ipynb`** | Per-session GPU/platform sanity check |

## Repo layout

| Path | Role |
|------|------|
| `src/pipeline.py` | End-to-end orchestrator |
| `src/llm_client.py` | vLLM or transformers routing (`call_llm`) |
| `src/mas.py` | Blackboard multi-agent |
| `src/reason.py` | Qwen2.5-VL single-agent (transformers) |
| `src/rescue.py` | ProteinMPNN + ThermoMPNN + ESMFold/Boltz |
| `src/metrics.py` | CPU/GPU/token metrics → CSV |
| `src/metrics_bundle.py` | Export `.tgz` + cross-platform compare |
| `src/submission.py` | Full submission orchestrator |
| `scripts/run_full_submission.py` | CLI: all flows + metrics export |
| `scripts/compare_platforms.py` | Colab vs AMD metrics diff |
| `scripts/verify_gpu.py` | GPU sanity check |
| `scripts/install_vllm_colab.sh` | Colab-only cu128 vLLM wheel |
| `scripts/start_vllm.sh` | CUDA-only vLLM servers (refuses ROCm) |
| `data/traces/` | Pre-cached blackboard traces |

## GPU structural stack (MI300X probe-validated)

- **ESMFold** — required folder (main env)
- **Boltz 2.2.1** — required folder; `--no_kernels`, isolated numpy&lt;2 venv via `scripts/setup_boltz_venv.sh`
- **ThermoMPNN** — ddG gate + scorer
- **ProteinMPNN** — fixed-backbone redesign

## Data licensing

- **Train:** CIViC + ClinVar only
- **Benchmark:** OncoKB validation only (not for training)

## GPU hygiene (AMD)

Attach GPU only for train / live inference / rescue. **Never** `pip install vllm` on AMD. Detach when idle (~4h wall-clock quota).

## Troubleshooting

| Error | Fix |
|-------|-----|
| `libcuda.so.1` on AMD | Restart session; `pip uninstall vllm`; only `requirements.txt` |
| `libcudart.so.13` on Colab | Use `scripts/install_vllm_colab.sh` or skip vLLM |
| `'Event' object is not callable` | `git pull` (metrics.py Thread._stop fix) |
| vLLM endpoints all `False` | Expected on AMD; use transformers or cached traces |
| `No module named 'pytorch_lightning'` | `pip install pytorch-lightning torchmetrics omegaconf wandb` or re-run `setup_external.sh` |
| `KeyError: 'pytorch-lightning_version'` | `git pull` — uses `scripts/thermompnn_ssm.py` to load `.pt` weights correctly |
| `torchao Failed to load ...cutlass/mxfp8` on Colab | Benign — bf16 LoRA does not use those kernels; training still works |
| `FileNotFoundError: 'boltz'` | Run `bash scripts/setup_external.sh` (installs Boltz venv) |
| `ensurepip` failed creating boltz_venv on Colab | `rm -rf external/boltz_venv && bash scripts/setup_boltz_venv.sh` (uses get-pip bootstrap) |
| `eval.py` downloads 15 GB model | Run `run_all_modes()` live first; `eval.py` scores saved traces only (no LLM load) |

## Monitoring (built from scratch — no LangGraph/Phoenix)

Native hooks in `src/metrics.py` write continuously to `METRICS_DIR`:

| File | Use in demo |
|------|-------------|
| `llm_calls.jsonl` | Per-agent live trace (role, round, latency, tokens) |
| `phases.csv` | GPU active vs attached time per pipeline phase |
| `system_samples.csv` | CPU/RAM/gfx/VRAM samples during run |
| `productive_throughput.csv` | **Latency-to-decision**, workflow density, egress/GPU-s |
| `before_after_comparison.csv` | single (baseline) vs blackboard (MAS depth) |
| `workflow_trace_dashboard.html` | Open in browser for live trace timeline |

Regenerate after a run:

```bash
PYTHONPATH=. python scripts/generate_workflow_report.py
```

**Narrative:** avoid mean GPU %; show **GPU productivity ratio**, **workflow density**, and **Return on Reasoning** (therapy F1 + direction acc per 1k tokens).

### Return on Reasoning (RoR)

After full submission + eval:

| File | Purpose |
|------|---------|
| `return_on_reasoning.csv` | Semantic accuracy vs token cost × architecture |
| `ror_benchmark.json` | Efficiency frontier data + architecture summary |
| `blackboard_ingress_by_role.csv` | Compaction ROI baseline (ingress waste by agent) |
| `workflow_trace_dashboard.html` | **Formal demo dashboard** (judges) + thesis figures |

## Advanced LLM capabilities (v7 upgrade)

| Capability | Module | Metrics artifact |
|---|---|---|
| Multimodal VL (structure PNG) | `src/structure.py`, `src/reason.py` | `multimodal_image` in `llm_calls.jsonl` |
| VUS routing + abstention | `src/variant_router.py` | `variant_routing` in trace JSON |
| Mechanism rubric + reflexion | `src/mas.py` | `mechanism_rubric_before/after`, `self_correction` events |
| Blackboard early-exit | `src/mas.py` | `early_exit` in trace + `productive_throughput.csv` |
| Debate architecture (PIK3CA) | `src/debate.py` | `architecture=debate` in llm_calls |
| Teacher–student LoRA | `train/build_dataset.py --from-traces` | LoRA ablation in `ablation_results.csv` |
| Hallucination metrics (6 HR) | `src/hallucination_eval.py` | `hallucination_report.csv`, `hallucination_summary.json` |
| Fold confidence benchmark | `src/fold_confidence_eval.py` | `benchmark_confidence.csv` (35-col), `benchmark_confidence_minimal.csv`, `fold_confidence_summary.json`, F1–F4 figures |
| Agent autonomy / DBTL L3 | `src/agent_autonomy_eval.py` | `autonomy_report.json`, `task_suite.csv`, `able_metrics.csv`, `dbtl_metrics.json`, `tevv_lite.csv` |
| MTB panel | `src/mtb_panel.py` | embedded in `comparison_*.json` |
| Live console echo | `src/progress.py` | mirrors `llm_calls.jsonl` during run |

**Literature positioning:** MOAlmanac-style RAG + MTBBench-style multi-agent conflict + PFUA-style tool grounding + AMix-style rescue verification — on open AMD ROCm. See `deck/slides.md` Slide 7 for judge citations.

### Hallucination metrics (§12 formulas)

Rule-based rates in `src/hallucination_eval.py` (one row per gene × mutation × architecture):

| Metric | Formula |
|---|---|
| `HR_design` | (# outputs with unsupported mechanism/therapy claims) / (# outputs) |
| `HR_property` | (# outputs with claimed pLDDT or stabilizing language misaligned vs structure/rescue) / (# outputs) |
| `HR_evidence` | (# therapy/evidence direction mismatches vs gold + CIViC block) / (# items referenced) |
| `HR_tool` | (# tool numeric summaries misaligned vs rescue block) / (# tool-backed outputs) |
| `BVR` | (# rescue designs passing fold + ddG gates) / (# evaluated); structural rate = `1 − BVR` |
| `HR_policy` | (# VUS/policy violations: confident therapy on `evidence_tier=none`) / (# checked) |

Dashboard card: `metrics/workflow_trace_dashboard.html` → **Trust layer**.

### Fold confidence (§14)

`benchmark_confidence.csv` uses the canonical 35-column schema; proxy `good_structure_label` when no experimental PDB ref (`mean_plddt ≥ 70` and `target_residue_plddt ≥ 50`). Thresholds in `targets.yaml` → `confidence_benchmark:`.

**Cached traces:** 12+ traces under `data/traces/` (4 demo cases × 3 architectures + debate + VUS G719S).

**Platform compare:** `PYTHONPATH=. python scripts/compare_platforms.py colab_bundle.tgz amd_bundle.tgz`

Same metrics bundle works for **Colab vs AMD** (`compare_platforms.py`) and **M.Tech documentation** (CSV/JSON export).
