#!/bin/bash
#SBATCH --job-name=cluster_validator
#SBATCH --nodes=2
#SBATCH --partition=earlytalent
#SBATCH --qos=gpulimit
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=validator_%j.out

set -euo pipefail

PROJECT_DIR="$HOME/llm-systems-poc/cluster-validator"
CONTAINER="pytorch-runtime"

RAW_DIR="$PROJECT_DIR/results/raw_$SLURM_JOB_ID"
mkdir -p "$RAW_DIR"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

export MASTER_ADDR MASTER_PORT


echo "Cluster Validator"
echo "Job:   $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Start: $(date)"
echo

# 1. Checking the details of the GPUs
echo " 1. Details of the GPUs "

srun enroot start \
    --mount "$PROJECT_DIR:/workspace/cluster-validator" \
    "$CONTAINER" \
    python /workspace/cluster-validator/validator/gpu_check.py \
    2>&1 | tee "$RAW_DIR/gpu.log"

# 2. Checking the compute performance of the GPUs
echo
echo " 2. Compute Performance of GPUs "

srun bash -c '
enroot start \
    --mount "'"$PROJECT_DIR"':/workspace/cluster-validator" \
    "'"$CONTAINER"'" \
    torchrun \
        --nnodes=2 \
        --nproc-per-node=2 \
        --node-rank=$SLURM_NODEID \
        --master-addr='"$MASTER_ADDR"' \
        --master-port='"$MASTER_PORT"' \
        /workspace/cluster-validator/validator/bf16_benchmark.py
' 2>&1 | tee "$RAW_DIR/bf16.log"

# 3. Checking the memory performance of the GPU
echo
echo " 3. Memory Performance of GPUs "

srun bash -c '
enroot start \
    --mount "'"$PROJECT_DIR"':/workspace/cluster-validator" \
    "'"$CONTAINER"'" \
    torchrun \
        --nnodes=2 \
        --nproc-per-node=2 \
        --node-rank=$SLURM_NODEID \
        --master-addr='"$MASTER_ADDR"' \
        --master-port='"$MASTER_PORT"' \
        /workspace/cluster-validator/validator/hbm_benchmark.py
' 2>&1 | tee "$RAW_DIR/hbm.log"

# 4. Checking that the GPUs can communicate with each other 
echo
echo "4. Communication Check "

srun bash -c '
enroot start \
    --mount "'"$PROJECT_DIR"':/workspace/cluster-validator" \
    "'"$CONTAINER"'" \
    torchrun \
        --nnodes=2 \
        --nproc-per-node=2 \
        --node-rank=$SLURM_NODEID \
        --master-addr='"$MASTER_ADDR"' \
        --master-port='"$MASTER_PORT"' \
        /workspace/cluster-validator/validator/nccl_check.py
' 2>&1 | tee "$RAW_DIR/nccl_check.log"

# 5. Checking the communication bandwidth and transport
echo
echo " 5. Communication Bandwidth and Transport "

srun bash -c '
enroot start \
    --env NCCL_DEBUG=INFO \
    --env NCCL_DEBUG_SUBSYS=NET \
    --mount "'"$PROJECT_DIR"':/workspace/cluster-validator" \
    "'"$CONTAINER"'" \
    torchrun \
        --nnodes=2 \
        --nproc-per-node=2 \
        --node-rank=$SLURM_NODEID \
        --master-addr='"$MASTER_ADDR"' \
        --master-port='"$MASTER_PORT"' \
        /workspace/cluster-validator/validator/nccl_benchmark.py
' 2>&1 | tee "$RAW_DIR/nccl_benchmark.log"

# 6. Checking shared storage performance
echo
echo "6. Shared Storage Performance "

TESTFILE="$HOME/validator_storage_$SLURM_JOB_ID.bin"

WRITE_OUTPUT=$(
    dd if=/dev/zero \
       of="$TESTFILE" \
       bs=1M count=1024 \
       conv=fdatasync 2>&1
)

READ_OUTPUT=$(
    dd if="$TESTFILE" \
       of=/dev/null \
       bs=1M \
       iflag=direct 2>&1
)

rm -f "$TESTFILE"

echo "$WRITE_OUTPUT"
echo "$READ_OUTPUT"

# Get the storage speeds and convert them to MB/s.
python - "$WRITE_OUTPUT" "$READ_OUTPUT" "$RAW_DIR/storage.json" <<'PY'
import json
import re
import sys

write_text = sys.argv[1]
read_text = sys.argv[2]
output = sys.argv[3]

def parse_rate(text):
    match = re.search(r'([\d.]+)\s+([GM]B/s)', text)

    if not match:
        raise RuntimeError(f"Could not parse dd throughput: {text}")

    value = float(match.group(1))
    unit = match.group(2)

    if unit == "GB/s":
        value *= 1000

    return value

result = {
    "write_mbs": parse_rate(write_text),
    "read_mbs": parse_rate(read_text),
}

with open(output, "w") as f:
    json.dump(result, f, indent=2)

print("Storage result:", result)
PY


# 7. Collecting measurements and generating report
echo
echo " 7. Collecting Measurements and Generating Report "

cd "$PROJECT_DIR"

python validator/collect_results.py "$RAW_DIR"
python validator/report.py results/measurements.json

echo
echo "Validation complete."
echo "Report: $PROJECT_DIR/results/validation_report.md"
echo "JSON:   $PROJECT_DIR/results/validation_report.json"
echo "Raw:    $RAW_DIR"
echo "End:    $(date)"

