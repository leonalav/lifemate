#!/usr/bin/env bash
# Local vLLM server for development.
#
# The competition requires the final system to run fully offline, so this is a
# development convenience only: the Docker image bakes the model in and runs with
# --network none.
#
# Usage: scripts/serve.sh [MODEL_PATH] [PORT]
set -euo pipefail

MODEL_PATH="${1:-Qwen/Qwen3.5-4B}"
PORT="${2:-8000}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --port "$PORT" \
  --max-model-len "${MAX_MODEL_LEN:-16384}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --seed "${SEED:-20261028}"
