#!/usr/bin/env bash
# Public-test inference driver: runs the agent over a split and writes a submission.
#
# Usage:
#   scripts/predict_public.sh DATA_DIR OUTPUT_FILE [split] [checkpoint]
set -euo pipefail

DATA_DIR="${1:?usage: predict_public.sh DATA_DIR OUTPUT_FILE [split] [checkpoint]}"
OUTPUT_FILE="${2:?usage: predict_public.sh DATA_DIR OUTPUT_FILE [split] [checkpoint]}"
SPLIT="${3:-public_test}"
CHECKPOINT="${4:-}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

ARGS=(
  --data-dir "$DATA_DIR"
  --split "$SPLIT"
  --output-file "$OUTPUT_FILE"
  --max-steps "${MAX_STEPS:-8}"
  --time-budget-s "${TIME_BUDGET_S:-25}"
)

if [[ -n "$CHECKPOINT" ]]; then
  ARGS+=(--checkpoint "$CHECKPOINT")
fi
if [[ -n "${TRANSCRIPT_CACHE:-}" ]]; then
  ARGS+=(--transcript-cache "$TRANSCRIPT_CACHE")
fi
if [[ -n "${CASE_MODE:-}" ]]; then
  # "none" (the default) keeps the transcript lowercase: the policy model casts the
  # arguments itself. "rules" and "model" are the recasing baselines.
  ARGS+=(--case-mode "$CASE_MODE")
fi
if [[ "${CASE_HINT:-0}" == "1" ]]; then
  ARGS+=(--case-hint)
fi

python -m lifemate infer "${ARGS[@]}"
