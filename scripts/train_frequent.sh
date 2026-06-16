#!/bin/bash
# Train with frequent step-based checkpoints
# Saves every 1000 steps (~6 min) so progress is never lost
# Resume: just run this script again

set -e
cd "$(dirname "$0")/.."

export NCCL_TIMEOUT=600

PROJECT="FIGR_SVG"
CKPT_DIR="proj_log/$PROJECT"

# Find latest epoch checkpoint if resuming (step checkpoints not supported for resume)
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
--epochs 10 \
--warmup_steps 200 \
--gradient_accumulation 2 \
--log_every 50 \
--save_every 1 \
--save_every_steps 1000 \
--val_every 1 \
--hidden_dim 1024 \
--num_layers 16 \
--num_heads 8 \
--embed_dim 512 \
--dropout 0.1 \
$RESUME

echo ""
echo "Training complete. Checkpoints in: $CKPT_DIR/"
ls -d "$CKPT_DIR"/step_* "$CKPT_DIR"/epoch_* 2>/dev/null | sort
