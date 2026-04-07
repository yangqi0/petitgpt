#!/usr/bin/env bash
set -euo pipefail

# Start an OpenAI-compatible vLLM server for the official Qwen GPTQ int4 model.
#
# Why this model?
# - The base repository Qwen/Qwen3.5-35B-A3B is not the int4 checkpoint.
# - The official int4 repository is Qwen/Qwen3.5-35B-A3B-GPTQ-Int4.
#
# This start command is intentionally conservative for a single 24 GB GPU:
# - --language-model-only skips the vision encoder and frees memory
# - --max-model-len defaults to 8192 instead of the full 262144
# - --served-model-name teacher makes your local OpenAI API calls simpler
#
# If this still OOMs on your pod, reduce:
#   MAX_MODEL_LEN=4096
#   GPU_MEMORY_UTIL=0.88
#
# If it runs comfortably, you can try:
#   MAX_MODEL_LEN=16384

PROJECT_DIR="${PROJECT_DIR:-/workspace/petitgpt}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv-qwen35}"

# MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B-GPTQ-Int4}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-teacher}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-EMPTY}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.85}"

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"

export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "[*] starting vLLM"
echo "    model=$MODEL"
echo "    served_model_name=$SERVED_MODEL_NAME"
echo "    max_model_len=$MAX_MODEL_LEN"
echo "    gpu_memory_utilization=$GPU_MEMORY_UTIL"

# vllm serve "$MODEL" \
#   --served-model-name "$SERVED_MODEL_NAME" \
#   --host "$HOST" \
#   --port "$PORT" \
#   --api-key "$API_KEY" \
#   --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
#   --max-model-len "$MAX_MODEL_LEN" \
#   --reasoning-parser qwen3 \
#   --quantization moe_wna16 \
#   --language-model-only

vllm serve "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --api-key "$API_KEY" \
  --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
  --max-model-len "$MAX_MODEL_LEN"
