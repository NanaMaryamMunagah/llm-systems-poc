#!/bin/bash
#SBATCH --job-name=gemma-lora
#SBATCH --partition=earlytalent
#SBATCH --qos=gpulimit
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --time=00:20:00
#SBATCH --output=training/results/train_%j.log

set -euo pipefail

PROJECT_DIR="$HOME/llm-systems-poc-clean"
CONTAINER="llm-training"

# Use the first allocated node as the coordinator
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Use the Slurm job ID to reduce the chances of port collisions
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

export MASTER_ADDR
export MASTER_PORT

echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "GPUs per node: 2"

srun \
    --nodes=2 \
    --ntasks=2 \
    --ntasks-per-node=1 \
    enroot start \
    --mount "$PROJECT_DIR:/workspace" \
    --rw "$CONTAINER" \
    bash -c '
        cd /workspace
        HOSTNAME=$(hostname)

if [ "$HOSTNAME" = "worker-0" ]; then
    NODE_RANK=0
elif [ "$HOSTNAME" = "worker-1" ]; then
    NODE_RANK=1
else
    echo "Unexpected hostname: $HOSTNAME"
    exit 1
fi

echo "Node: $HOSTNAME"
echo "Node rank: $NODE_RANK"



torchrun \
    --nnodes=2 \
    --nproc_per_node=2 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    training/train_lora.py \
    --epochs 3 \
    --output-dir training/results/gemma3-4b-mmlu-math-lora
    '