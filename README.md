# Mutation → Mechanism → Therapy

Structure-aware, multimodal, multi-agent reasoning copilot for precision oncology on **AMD Instinct MI300X / ROCm**.

Demo cases: **EGFR L858R**, **PIK3CA E545K** (inhibitor path) · **TP53 R175H** (structural rescue path).

## Clone on AMD AI Notebook

```bash
cd /workspace
git clone --depth 1 https://github.com/lightflow16/mutation-mechanism-therapy-amd.git
cd mutation-mechanism-therapy-amd
```

Open **`01_run_pipeline.ipynb`** and run cells top-to-bottom.

Persistent outputs (metrics, LoRA checkpoints, HF cache) go to `/workspace/shared/` — survives session restarts.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| **`01_run_pipeline.ipynb`** | Main driver: setup → cached demo → GPU steps → bundle download |
| **`00_env_check.ipynb`** | Optional per-session ROCm/torch sanity check |

## One-time setup (inside notebook or terminal)

```bash
pip install -r requirements.txt openai
bash scripts/setup_external.sh   # clones bMAS, ThermoMPNN, ProteinMPNN, mini_protein_pipeline
export HF_HOME=/workspace/shared/hf_cache
export METRICS_DIR=/workspace/shared/metrics
mkdir -p /workspace/shared/{hf_cache,metrics,lora_ckpts}
```

## Instant demo (no GPU, no vLLM)

Cached blackboard traces replay in ~1s:

```python
from src.pipeline import run_case
run_case("EGFR", "L858R", architecture="blackboard", use_cached_trace=True)
```

## GPU steps (attach MI300X first)

```bash
bash scripts/start_vllm.sh          # 3 vLLM servers on ports 8000/8001/8002
python train/build_dataset.py       # rebuild LoRA JSONL if needed
python train/lora_sft.py            # ~10-30 min
python train/eval.py                # ablation table
python app.py                       # Gradio demo
```

## Repo layout

| Path | Role |
|------|------|
| `src/pipeline.py` | End-to-end orchestrator |
| `src/structure.py` | AlphaFold fetch, HGVS, py3Dmol |
| `src/evidence.py` | CIViC/ClinVar/PubMed + cache |
| `src/mas.py` | Blackboard multi-agent (vLLM) |
| `src/rescue.py` | ProteinMPNN + ThermoMPNN + ESMFold/Boltz |
| `src/metrics.py` | CPU/GPU/token metrics → CSV |
| `data/cases/` | Curated case JSONs |
| `data/traces/` | Pre-cached blackboard traces for instant demo |
| `train/` | LoRA dataset, SFT, eval |
| `targets.yaml` | Demo targets + vLLM endpoint map |
| `deck/` | Slide outline + demo script |

External repos are **not** bundled — `scripts/setup_external.sh` clones them into `external/` on the notebook.

## GPU structural stack (MI300X probe-validated)

- **ESMFold** — default folder, main env
- **Boltz 2.2.1** — `--no_kernels`, isolated numpy&lt;2 venv (`scripts/setup_boltz_venv.sh`)
- **ThermoMPNN** — ddG gate + scorer
- **ProteinMPNN** — fixed-backbone redesign

## Data licensing

- **Train:** CIViC + ClinVar only
- **Benchmark:** OncoKB validation only (not for training)

## GPU hygiene

Attach GPU only for fold / train / serve / ablation / rescue. Detach for pip installs, git clones, deck work, and idle time. GPU quota (~4h) is wall-clock while attached — check AMD dashboard.
