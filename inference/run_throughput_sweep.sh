#!/bin/bash
set -euo pipefail

REQUESTS=128
MAX_TOKENS=64

for CONCURRENCY in 1 4 8 16 32 64 128
do
    echo "Running concurrency=${CONCURRENCY}"

    python3 /home/nmm/llm-systems-poc-clean/inference/benchmark_throughput.py \
        --concurrency "${CONCURRENCY}" \
        --requests "${REQUESTS}" \
        --max-tokens "${MAX_TOKENS}" \
        --output "/home/nmm/llm-systems-poc-clean/inference/results/concurrency_${CONCURRENCY}.json"
done

echo "Benchmark sweep complete."