# LLM Systems PoC

An end-to-end proof of concept for validating GPU infrastructure, fine-tuning an open-source LLM across multiple GPUs, evaluating model quality, and optimizing inference performance.

The project is organized around three stages of an LLM deployment workflow:

1. **Cluster Validation** — verify that the GPU infrastructure is functioning correctly and delivering expected compute, memory, networking, and storage performance.
2. **Distributed Fine-Tuning** — fine-tune an open-source LLM on an MMLU category using a multi-node, multi-GPU Slurm environment.
3. **Evaluation and Inference Optimization** — measure model quality against the original model and optimize inference performance through controlled experiments.

## Repository Structure

```text
llm-systems-poc/
├── README.md
├── cluster-validator/
│   ├── Dockerfile
│   ├── README.md
│   ├── scripts/
│   ├── validator/
│   └── results/
├── training/
├── evaluation/
└── inference/
```

The training, evaluation, and inference directories will contain the subsequent stages of the PoC.

## Phase 0 — Cluster Validation

The first stage validates the infrastructure before consuming GPU time on the target LLM workload.

The validator checks:

- GPU visibility and memory
- BF16 compute performance
- HBM bandwidth
- multi-node NCCL correctness
- NCCL transport selection
- GPU Direct RDMA
- inter-node AllReduce performance
- shared storage performance

The final validation run across four NVIDIA H200 GPUs on two worker nodes passed all configured checks.

Detailed implementation, methodology, acceptance criteria, and results are documented in [`cluster-validator/CLUSTER_VALIDATION.md`](cluster-validator/CLUSTER_VALIDATION.md).

## Phase 1 — Distributed LLM Fine-Tuning

Phase 1 will establish a baseline on a selected MMLU category and fine-tune an open-source LLM using four GPUs distributed across two Slurm nodes.

This phase will document:

- model and dataset selection
- prompt design
- fine-tuning strategy
- hyperparameters
- distributed training configuration
- experiment monitoring
- reproducibility

## Phase 2 — Evaluation and Inference

Phase 2 will compare the fine-tuned model with the original model on held-out evaluation data and measure the resulting accuracy change.

The inference portion will then optimize a selected performance objective, such as throughput or latency, using controlled experiments and documented measurements.

## Environment

The PoC environment provides:

- 2 Slurm worker nodes
- NVIDIA H200 GPUs
- 4 GPUs allocated for experiments
- PyTorch
- NCCL
- Enroot
- InfiniBand with GPU Direct RDMA

Each phase is designed to be reproducible through version-controlled scripts, configuration, results, and documentation.