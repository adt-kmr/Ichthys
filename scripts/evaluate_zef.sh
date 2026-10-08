#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 CHECKPOINT ZEF_ROOT OUTPUT_DIR" >&2
  exit 2
fi

CHECKPOINT=$1
DATASET_DIR=$2
OUTPUT_DIR=$3
PYTHON=${PYTHON:-python3}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

mkdir -p "$OUTPUT_DIR/result"

SCENES=(
  "ZebraFish-02-new:zebra02-new.csv:new_zebra02_merged"
  "ZebraFish-04:zebra04.csv:new_zebra04_merged"
)

for entry in "${SCENES[@]}"; do
  IFS=: read -r scene gt_file output_stem <<<"$entry"
  prediction="$OUTPUT_DIR/$output_stem.txt"
  "$PYTHON" "$ROOT_DIR/infer.py" \
    "$CHECKPOINT" \
    "$DATASET_DIR/$scene.json" \
    "$DATASET_DIR/$scene" \
    --no_dino \
    --output "$prediction"
  "$PYTHON" "$ROOT_DIR/evaluate.py" \
    --pred "$prediction" \
    --gt "$DATASET_DIR/$scene/gt/$gt_file" \
    --match_pred_range \
    > "$OUTPUT_DIR/result/${output_stem}_eval.txt"
done
