#!/usr/bin/env bash
# Launch three vLLM servers for heterogeneous models on one CUDA GPU.
# NOT supported on AMD ROCm — use LLM_BACKEND=transformers (default on ROCm).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${VLLM_LOG_DIR:-$ROOT/logs/vllm}"
export HF_HOME="${HF_HOME:-$ROOT/shared/hf_cache}"
UTIL="${GPU_MEMORY_UTILIZATION:-0.25}"

if python3 -c "import torch; print(torch.__version__)" 2>/dev/null | grep -qi rocm; then
  echo "ERROR: vLLM is not supported on AMD ROCm."
  echo "  Use transformers instead (default): export LLM_BACKEND=transformers"
  echo "  Live single-agent: run_case(..., architecture='single', use_cached_trace=False)"
  exit 1
fi

if ! python3 -c "import vllm._C" 2>/dev/null; then
  echo "ERROR: vLLM import failed (libcudart mismatch?)."
  echo "  Colab: bash scripts/install_vllm_colab.sh"
  echo "  Or skip vLLM: export LLM_BACKEND=transformers"
  exit 1
fi

mkdir -p "$LOG_DIR" "$HF_HOME"

start_server() {
  local model="$1" port="$2" log="$3"
  if curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
    echo "port ${port} already up (${model})"
    return 0
  fi
  echo "starting ${model} on port ${port} -> ${log}"
  nohup vllm serve "$model" \
    --port "$port" \
    --gpu-memory-utilization "$UTIL" \
    >>"$log" 2>&1 &
  echo $! >"${log}.pid"
}

wait_port() {
  local port="$1" label="$2" max="${3:-600}"
  local i=0
  while [ "$i" -lt "$max" ]; do
    if curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
      echo "${label} ready on port ${port} (${i}s)"
      return 0
    fi
    sleep 5
    i=$((i + 5))
    if [ $((i % 30)) -eq 0 ]; then
      echo "  waiting for ${label} on ${port}... ${i}s (tail ${LOG_DIR}/*.log)"
    fi
  done
  echo "ERROR: ${label} not ready on port ${port} after ${max}s"
  tail -20 "${LOG_DIR}"/*.log 2>/dev/null || true
  return 1
}

start_server "Qwen/Qwen2.5-VL-7B-Instruct" 8000 "$LOG_DIR/vl7b.log"
wait_port 8000 "reasoner (VL-7B)" 900

start_server "Qwen/Qwen2.5-7B-Instruct" 8001 "$LOG_DIR/qwen7b.log"
wait_port 8001 "mechanism (7B)" 600

start_server "Qwen/Qwen2.5-3B-Instruct" 8002 "$LOG_DIR/qwen3b.log"
wait_port 8002 "planner (3B)" 600

echo "All vLLM servers up on 8000/8001/8002. Logs: ${LOG_DIR}"
