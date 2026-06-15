#!/usr/bin/env bash
# Install vLLM with a CUDA 12.8 wheel matching Colab (NOT plain pip install vllm).
set -euo pipefail

echo "Uninstalling any broken vLLM build..."
pip uninstall -y vllm 2>/dev/null || true

WHEEL="https://github.com/vllm-project/vllm/releases/download/v0.10.2/vllm-0.10.2+cu128-cp38-abi3-manylinux1_x86_64.whl"
echo "Installing ${WHEEL}"
pip install "$WHEEL" --extra-index-url https://download.pytorch.org/whl/cu128

python3 - <<'PY'
import vllm._C
print("vLLM OK:", vllm.__version__ if hasattr(vllm, "__version__") else "imported")
PY

echo "Done. Run: bash scripts/start_vllm.sh"
