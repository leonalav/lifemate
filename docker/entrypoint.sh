#!/usr/bin/env bash
# The container entrypoint.
#
# Discovers the split under /data (the private test arrives as a new folder — the
# split name is never hardcoded) and writes result/submission.jsonl.
#
# The container runs with --network none, so nothing is downloaded here: the LLM and
# the ASR bundle are baked into the image.
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
OUTPUT_FILE="${OUTPUT_FILE:-/app/result/submission.jsonl}"
MODEL_PATH="${MODEL_PATH:-/models/llm}"
ASR_MODEL_DIR="${ASR_MODEL_DIR:-/models/gipformer}"
CHECKPOINT="${CHECKPOINT:-/models/llm}"
MAX_STEPS="${MAX_STEPS:-8}"
TIME_BUDGET_S="${TIME_BUDGET_S:-25}"
TRANSCRIPT_CACHE="${TRANSCRIPT_CACHE:-/app/result/transcripts.jsonl}"

export PYTHONPATH="/app/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$(dirname "$OUTPUT_FILE")"

python3 -m lifemate infer \
  --data-dir "$DATA_DIR" \
  --checkpoint "$CHECKPOINT" \
  --model-path "$MODEL_PATH" \
  --asr-model-dir "$ASR_MODEL_DIR" \
  --output-file "$OUTPUT_FILE" \
  --max-steps "$MAX_STEPS" \
  --time-budget-s "$TIME_BUDGET_S" \
  --transcript-cache "$TRANSCRIPT_CACHE"

echo "submission written to $OUTPUT_FILE"
