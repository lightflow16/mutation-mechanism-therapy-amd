#!/usr/bin/env bash
# Launch three vLLM servers for heterogeneous models on one MI300X.
# Run AFTER LoRA adapter is ready; set HF_HOME first.
set -euo pipefail
export HF_HOME="${HF_HOME:-/workspace/shared/hf_cache}"
UTIL="${GPU_MEMORY_UTILIZATION:-0.25}"

vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 --gpu-memory-utilization "$UTIL" &
sleep 30
curl -sf http://localhost:8000/v1/models || echo "VL server not ready yet"

vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001 --gpu-memory-utilization "$UTIL" &
sleep 20
curl -sf http://localhost:8001/v1/models || echo "7B server not ready yet"

vllm serve Qwen/Qwen2.5-3B-Instruct --port 8002 --gpu-memory-utilization "$UTIL" &
sleep 20
curl -sf http://localhost:8002/v1/models || echo "3B server not ready yet"

echo "vLLM servers on 8000/8001/8002"
