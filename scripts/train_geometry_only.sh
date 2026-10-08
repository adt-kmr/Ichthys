#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 TRAIN_DIR CHECKPOINT_DIR [VALIDATION_DIR]" >&2
  exit 2
fi

TRAIN_DIR=$1
CHECKPOINT_DIR=$2
PYTHON=${PYTHON:-python3}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATASET_ARGS=("$TRAIN_DIR")
if [[ $# -eq 3 ]]; then
  DATASET_ARGS+=("$3")
fi

"$PYTHON" "$ROOT_DIR/train.py" \
  "${DATASET_ARGS[@]}" \
  --no_dino \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --epochs 400 \
  --warmup 100 \
  --selection_metric train
