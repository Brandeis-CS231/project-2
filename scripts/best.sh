#!/bin/bash

#SBATCH --job-name=grid
#SBATCH --ntasks=12
#SBATCH --mem-per-cpu=4096
#SBATCH --nodelist=student-gpu-003
#SBATCH --gres=gpu:1
#SBATCH --partition gpu48g

UV_CACHE_DIR=/data/$(whoami)/.cache/uv

uv run project-2 train --epochs 4 --learning_rate 0.0001 --batch_size 32 \
  --num_enc_layers 4 --num_dec_layers 4 --d_model 256 --num_heads 2 \
  --d_ff 512 --dropout 0.1 --max_src_len 50 --max_tgt_len 50 \
  --strategy greedy --save ./data/train.json ./data/dev.json ./data/test.json
