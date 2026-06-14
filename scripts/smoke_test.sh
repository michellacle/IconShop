#!/bin/bash
# Smoke test: generate icons from latest checkpoint
# Usage: bash scripts/smoke_test.sh [project_name] [--full]
#   Default project: FIGR_SVG
#   --full  uses full model size (16 layers, 1024 hidden)
#   without --full uses small model (2 layers, 256 hidden)

set -e
cd "$(dirname "$0")/.."

PROJECT="${1:-FIGR_SVG}"
CKPT_DIR="proj_log/$PROJECT"

# Find latest checkpoint
LATEST=$(ls -d "$CKPT_DIR"/epoch_* 2>/dev/null | sort | tail -1)
if [ -z "$LATEST" ]; then
    echo "ERROR: No checkpoint found in $CKPT_DIR"
    exit 1
fi

if [ "$2" = "--full" ]; then
    LAYERS=16; HIDDEN=1024; HEADS=8; PIX_LEN=512; NUM=4
else
    LAYERS=2; HIDDEN=256; HEADS=4; PIX_LEN=128; NUM=2
fi

OUTPUT="output/smoke_test_${PROJECT}_$(basename "$LATEST")"

echo "Checkpoint: $LATEST"
echo "Model: ${LAYERS} layers, ${HIDDEN} hidden, ${HEADS} heads"
echo "Output: $OUTPUT"

.venv/bin/python sample.py \
    --sketch_weight "$LATEST" \
    --output "$OUTPUT" \
    --hidden_dim "$HIDDEN" \
    --num_layers "$LAYERS" \
    --num_heads "$HEADS" \
    --embed_dim 512 \
    --pix_len "$PIX_LEN" \
    --num_sample "$NUM" \
    --prompts "star" "heart" "bug" "car" "trash"

echo ""
echo "Results:"
find "$OUTPUT" -name '*.svg' -exec sh -c 'echo "  {} ($(wc -c < "{}") bytes)"' \;
