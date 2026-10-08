#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 || ( $# -eq 4 && $4 != --check-only ) ]]; then
  echo "Usage: $0 CHECKPOINT SYNFISH_TEST_DIR OUTPUT_DIR [--check-only]" >&2
  exit 2
fi

CHECKPOINT=$1
DATASET_DIR=$2
OUTPUT_DIR=$3
PYTHON=${PYTHON:-python3}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

SCENES=(test1 test2 test3 test4 test5 test6)

# Check every scene before starting an expensive inference run.
for scene in "${SCENES[@]}"; do
  for file in "$DATASET_DIR/$scene.json" "$DATASET_DIR/gt-traj/$scene-gt.csv"; do
    [[ -f "$file" ]] || { echo "Missing file: $file" >&2; exit 1; }
  done
  for cam in 0 1 2; do
    [[ -d "$DATASET_DIR/$scene/cam$cam" ]] || { echo "Missing camera: $DATASET_DIR/$scene/cam$cam" >&2; exit 1; }
  done
  echo "Ready: $scene"
done
if [[ ${4:-} == --check-only ]]; then
  exit 0
fi
[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR/result"

for scene in "${SCENES[@]}"; do
  prediction="$OUTPUT_DIR/$scene.txt"
  "$PYTHON" "$ROOT_DIR/infer.py" \
    "$CHECKPOINT" \
    "$DATASET_DIR/$scene.json" \
    "$DATASET_DIR/$scene" \
    --no_dino \
    --output "$prediction"
  "$PYTHON" "$ROOT_DIR/evaluate.py" \
    --pred "$prediction" \
    --gt "$DATASET_DIR/gt-traj/$scene-gt.csv" \
    --match_pred_range \
    > "$OUTPUT_DIR/result/${scene}_eval.txt"
done
