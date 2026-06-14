#!/bin/bash
# Train 1 epoch at a time with resume support
# Usage: bash scripts/train_epoch.sh [project_name]
#   First run:  bash scripts/train_epoch.sh FIGR_SVG
#   Resume:     bash scripts/train_epoch.sh FIGR_SVG

set -e
cd "$(dirname "$0")/.."

PROJECT="${1:-FIGR_SVG}"
export NCCL_TIMEOUT=600

# Find latest checkpoint if resuming
CKPT_DIR="proj_log/$PROJECT"
RESUME=""
if [ -d "$CKPT_DIR" ]; then
    LATEST=$(ls -d "$CKPT_DIR"/epoch_* 2>/dev/null | sort | tail -1)
    if [ -n "$LATEST" ]; then
        RESUME="--resume_from_checkpoint $LATEST"
        echo "Resuming from: $LATEST"
    else
        echo "Starting fresh: $CKPT_DIR"
    fi
fi

.venv/bin/accelerate launch --config_file config/train.yaml train.py \
--train_meta_file dataset/FIGR-SVG-train.csv \
--val_meta_file dataset/FIGR-SVG-valid.csv \
--svg_folder dataset/FIGR-SVG-svgo \
--output_dir proj_log/ \
--project_name "$PROJECT" \
--maxlen 512 \
--batchsize 4 \
--num_workers 2 \
--epochs 1 \
--warmup_steps 200 \
--gradient_accumulation 2 \
--log_every 50 \
--save_every 1 \
--val_every 1 \
--hidden_dim 1024 \
--num_layers 16 \
--num_heads 8 \
--embed_dim 512 \
--dropout 0.1 \
$RESUME

echo ""
echo "Epoch complete. Checkpoint saved in: $CKPT_DIR/"
ls -d "$CKPT_DIR"/epoch_* 2>/dev/null | sort
