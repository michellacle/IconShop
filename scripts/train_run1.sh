#!/bin/bash
# First real training run - conservative settings to avoid lock-up
# 2 epochs, 2 workers/GPU, batch 4/GPU

set -e
cd "$(dirname "$0")/.."

.venv/bin/accelerate launch --config_file config/train.yaml train.py \
--train_meta_file dataset/FIGR-SVG-train.csv \
--val_meta_file dataset/FIGR-SVG-valid.csv \
--svg_folder dataset/FIGR-SVG-svgo \
--output_dir proj_log/ \
--project_name FIGR_SVG_run1 \
--maxlen 512 \
--batchsize 4 \
--num_workers 2 \
--epochs 2 \
--warmup_steps 200 \
--gradient_accumulation 2 \
--log_every 50 \
--save_every 1 \
--val_every 1 \
--hidden_dim 1024 \
--num_layers 16 \
--num_heads 8 \
--embed_dim 512 \
--dropout 0.1
