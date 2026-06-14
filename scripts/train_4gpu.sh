#!/bin/bash
# Full training script for 4x RTX 3090
# Adjusted from original 8-GPU config

.venv/bin/accelerate launch --config_file config/train.yaml train.py \
--train_meta_file dataset/FIGR-SVG-train.csv \
--val_meta_file dataset/FIGR-SVG-valid.csv \
--svg_folder dataset/FIGR-SVG-svgo \
--output_dir proj_log/ \
--project_name FIGR_SVG_4GPU \
--maxlen 512 \
--batchsize 10 \
--epochs 100 \
--warmup_steps 16000 \
--gradient_accumulation 2 \
--log_every 25 \
--save_every 5 \
--val_every 5 \
--hidden_dim 1024 \
--num_layers 16 \
--num_heads 8 \
--embed_dim 512 \
--dropout 0.1
