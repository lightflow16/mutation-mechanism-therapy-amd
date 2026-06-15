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

## Live GPU inference

### AMD MI300X (primary) — transformers, NOT vLLM

`pip install vllm` **breaks ROCm torch**. The repo defaults to `LLM_BACKEND=transformers` on ROCm.

```python
from src.pipeline import run_case
run_case("EGFR", "L858R", architecture="single", use_cached_trace=False)   # Qwen2.5-VL
run_case("EGFR", "L858R", architecture="blackboard", use_cached_trace=False)  # live MAS via transformers
python train/lora_sft.py
```

Blackboard demo for judges: use **cached traces** (`use_cached_trace=True`) — identical outcomes, instant playback.

### Google Colab A100 — transformers (default) or optional vLLM

**Recommended:** same transformers path as AMD (no vLLM install).

**Optional vLLM** (CUDA 12.8 wheel — plain `pip install vllm` fails with `libcudart.so.13`):

```bash
bash scripts/install_vllm_colab.sh
bash scripts/start_vllm.sh
export LLM_BACKEND=vllm
```

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
| `scripts/verify_gpu.py` | GPU sanity check |
| `scripts/install_vllm_colab.sh` | Colab-only cu128 vLLM wheel |
| `scripts/start_vllm.sh` | CUDA-only vLLM servers (refuses ROCm) |
| `data/traces/` | Pre-cached blackboard traces |

## GPU structural stack (MI300X probe-validated)

- **ESMFold** — default folder, main env
- **Boltz 2.2.1** — `--no_kernels`, isolated numpy&lt;2 venv
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
| `FileNotFoundError: 'boltz'` | Expected on Colab; rescue uses ESMFold only. `git pull` skips Boltz when not installed |
