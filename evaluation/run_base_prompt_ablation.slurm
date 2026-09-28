#!/bin/bash
#SBATCH --job-name=base-prompt
#SBATCH --partition=earlytalent
#SBATCH --qos=gpulimit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=evaluation/results/base_prompt_ablation_%j.log

set -euo pipefail

PROJECT_DIR="$HOME/llm-systems-poc-clean"
CONTAINER="llm-training"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

enroot start \
    --mount "$PROJECT_DIR:/workspace" \
    --rw \
    "$CONTAINER" \
    bash -c '
        cd /workspace
        python evaluation/base_prompt_ablation.py
    '