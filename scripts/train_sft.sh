#!/usr/bin/env bash
# Phase A/B training driver.
#
# Runs on a Colab GPU. The dataset lives on local disk (never in Drive, so the unzip
# is fast); checkpoints and logs go to Drive so a disconnect does not lose progress.
#
# Usage:
#   scripts/train_sft.sh DATA_DIR OUTPUT_DIR [extra args passed to `python -m lifemate train`]
set -euo pipefail

DATA_DIR="${1:?usage: train_sft.sh DATA_DIR OUTPUT_DIR [args...]}"
OUTPUT_DIR="${2:?usage: train_sft.sh DATA_DIR OUTPUT_DIR [args...]}"
shift 2

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$REPO_DIR/configs/sft_qlora.yaml}"

echo "repo:   $REPO_DIR"
echo "data:   $DATA_DIR"
echo "output: $OUTPUT_DIR"
echo "config: $CONFIG"

export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

# The raw ASR transcript cache: the model is trained on the text the recogniser
# produces, which is the text it meets at inference time.
if [[ -n "${TRANSCRIPTS:-}" ]]; then
  TRANSCRIPT_ARGS=(--transcripts "$TRANSCRIPTS")
else
  TRANSCRIPT_ARGS=()
fi

python -m lifemate train \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --config "$CONFIG" \
  --resume \
  "${TRANSCRIPT_ARGS[@]}" \
  "$@"
