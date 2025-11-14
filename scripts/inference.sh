#!/bin/bash

#SBATCH --job-name=grid
#SBATCH --ntasks=12
#SBATCH --mem-per-cpu=4096
#SBATCH --nodelist=student-gpu-003
#SBATCH --gres=gpu:1
#SBATCH --partition gpu48g

UV_CACHE_DIR=/data/$(whoami)/.cache/uv

uv run project-2 inference --checkpoint_path $1 --strategy greedy

uv run project-2 inference --checkpoint_path $1 --strategy beam_search --beam_width 3
uv run project-2 inference --checkpoint_path $1 --strategy beam_search --beam_width 5
uv run project-2 inference --checkpoint_path $1 --strategy beam_search --beam_width 10
