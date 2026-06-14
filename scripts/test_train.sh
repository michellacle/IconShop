#!/bin/bash
# Minimal test training script
cd /home/michel/code/IconShop
source .venv/bin/activate

accelerate launch --config_file config/train.yaml train.py \
--train_meta_file dataset/FIGR-SVG-train.csv \
--val_meta_file dataset/FIGR-SVG-valid.csv \
--svg_folder dataset/FIGR-SVG-svgo \
--output_dir proj_log/ \
--project_name test_run \
--maxlen 512 \
--batchsize 2 \
--hidden_dim 256 \
--num_layers 2 \
--num_heads 4 \
--epochs 2 \
--warmup_steps 2 \
--gradient_accumulation 1 \
--log_every 5 \
--save_every 999 \
--val_every 1
