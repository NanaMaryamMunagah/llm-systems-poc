# LLM Systems PoC

An end-to-end proof of concept for validating GPU infrastructure, distributed fine-tuning of an open-source LLM, evaluating model quality, and optimizing inference throughput.

The PoC models a workflow for a startup preparing to train an LLM and host its own inference service on GPU cloud infrastructure. Before scaling to a larger deployment, the goal is to validate the infrastructure, demonstrate a working distributed training pipeline, verify that fine-tuning improves the target task, and characterize inference performance.

## Overview

The project is organized into three stages:

1. **Cluster Validation** — validate GPU compute, memory, networking, communication, and shared storage before running the ML workload.
2. **Distributed Fine-Tuning and Evaluation** — establish a baseline, fine-tune Gemma 3 4B on MMLU math data across multiple GPUs and nodes, and evaluate the resulting model on a held-out test set.
3. **Inference Optimization** — serve the fine-tuned model with vLLM and measure how request concurrency affects aggregate generation throughput and latency.

## Key Results

| Stage | Result |
|---|---|
| Cluster validation | All configured checks passed across 4 NVIDIA H200 GPUs on 2 nodes |
| Distributed training | LoRA fine-tuning completed across 2 nodes × 2 GPUs |
| Trainable parameters | 11.9M / 4.31B parameters (0.276%) |
| Baseline accuracy | 35.52% |
| Fine-tuned accuracy | 41.80% |
| Accuracy improvement | +6.28 percentage points |
| Baseline inference throughput | 196.52 output tokens/s at concurrency 1 |
| Highest observed throughput | 4,332.11 output tokens/s at concurrency 128 |
| Throughput improvement | ~22x |

The model-quality comparison uses the same frozen 366-example custom held-out MMLU-math test split and the same evaluation procedure before and after fine-tuning.

The inference result represents the highest throughput observed among the concurrency settings tested on one H200 GPU. It is not intended as a direct estimate of performance on a larger multi-GPU deployment.

## Repository Structure

```text
llm-systems-poc/
├── README.md
│
├── cluster-validator/
│   ├── Dockerfile
│   ├── CLUSTER_VALIDATION.md
│   ├── scripts/
│   ├── validator/
│   └── results/
│
├── training/
│   ├── data/
│   ├── scripts/
│   └── train_lora.py
│
├── evaluation/
│   ├── evaluate_mmlu.py
│   ├── evaluate_finetuned_mmlu.py
│   └── results/
│
└── inference/
    ├── benchmark_throughput.py
    ├── run_throughput_sweep.sh
    ├── inference_performance.md
    └── results/
```

Large model weights, container images, and generated training checkpoints are intentionally excluded from Git.

---

## Phase 0 — Cluster Validation

Before running the LLM workload, I validated the underlying GPU infrastructure.

The validator checks:

- GPU visibility and memory
- BF16 matrix multiplication performance
- HBM bandwidth
- multi-node NCCL correctness
- NCCL transport selection
- GPU Direct RDMA
- inter-node AllReduce performance
- shared storage performance

The final validation run used four NVIDIA H200 GPUs distributed across two worker nodes.

### Final Validation Results

| Check | Result |
|---|---:|
| GPU memory | ~139.8 GiB per H200 |
| BF16 compute | ~768–806 TFLOPS per GPU |
| HBM bandwidth | ~4.28 TB/s effective read + write |
| NCCL correctness | PASS |
| NCCL transport | InfiniBand |
| GPU Direct RDMA | Enabled |
| Inter-node AllReduce algorithm bandwidth | 63.17 GB/s |
| Shared storage read | ~920 MB/s |

A lightweight PyTorch runtime image was used for the final validator. RDMA userspace libraries were added to enable NCCL to use InfiniBand and GPU Direct RDMA rather than socket networking.

The container recipe is captured in:

`cluster-validator/Dockerfile`

Detailed methodology, acceptance criteria, implementation decisions, and limitations are documented in:

[`cluster-validator/CLUSTER_VALIDATION.md`](cluster-validator/CLUSTER_VALIDATION.md)

---

## Phase 1 — Distributed Fine-Tuning

### Model

The PoC uses:

`google/gemma-3-4b-pt`

A 4B parameter pretrained model was selected to keep iteration and debugging practical while still exercising the distributed training and inference workflow.

The purpose of this PoC is to validate the end-to-end system and methodology before scaling to a larger model.

### Dataset

Fine-tuning uses five mathematics-related subjects from `cais/mmlu`:

- elementary mathematics
- high school mathematics
- high school statistics
- college mathematics
- abstract algebra

The examples were pooled, exact duplicates were removed using the question and answer choices, and the remaining examples were split deterministically using seed `1234`.

| Split | Examples |
|---|---:|
| Train | 719 |
| Validation | 118 |
| Test | 366 |

The test set was frozen before fine-tuning and was not used for hyperparameter selection.

Because the original MMLU splits were pooled before creating the custom split, these results should be interpreted as accuracy on the **custom held-out MMLU-math split**, not as an official MMLU benchmark score.

### Prompt Format

Training and evaluation use the same multiple-choice structure:

```text
Question: <question>

A. <choice>
B. <choice>
C. <choice>
D. <choice>

Answer:
```

During supervised fine-tuning, the target completion is the correct answer letter.

### Fine-Tuning Strategy

I used parameter-efficient supervised fine-tuning with LoRA.

LoRA was applied to:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`

Configuration:

| Parameter | Value |
|---|---:|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Trainable parameters | 11,898,880 |
| Total parameters | 4,311,978,352 |
| Trainable percentage | 0.276% |
| Precision | BF16 |
| Epochs | 3 |
| Learning rate | 2e-4 |
| Weight decay | 0.01 |
| Per-GPU batch size | 4 |
| Gradient accumulation | 4 |
| Effective global batch size | 64 |

Only the answer token contributes to the supervised loss. Prompt tokens and padding tokens are masked from the loss.

### Distributed Training

The final training run used:

- 2 Slurm worker nodes
- 2 H200 GPUs per node
- 4 GPUs total
- PyTorch DistributedDataParallel through `torchrun`
- NCCL for distributed GPU communication
- InfiniBand / GPU Direct RDMA available between nodes

The final run completed 3 epochs and 36 optimizer steps.

Validation loss:

| Epoch | Validation loss |
|---:|---:|
| 1 | 1.529 |
| 2 | 1.273 |
| 3 | 1.297 |

Final training runtime was approximately 27.4 seconds for the fine-tuning workload.

---

## Phase 2A — Model Evaluation

The base and fine-tuned models were evaluated using exactly the same:

- 366-example held-out test set
- prompt format
- answer choices
- next-token scoring procedure

For each question, the model's logits for the answer tokens `A`, `B`, `C`, and `D` were compared and the highest-scoring option was selected.

### Accuracy Results

| Subject | Baseline | Fine-Tuned |
|---|---:|---:|
| Abstract Algebra | 22.22% | 25.00% |
| College Mathematics | 36.11% | 33.33% |
| Elementary Mathematics | 42.97% | 44.53% |
| High School Mathematics | 27.17% | 39.13% |
| High School Statistics | 39.19% | 52.70% |
| **Overall** | **35.52%** | **41.80%** |

Overall accuracy increased by **6.28 percentage points**, from 130/366 correct answers to 153/366.

Four of the five evaluated subject areas improved. College mathematics decreased slightly, which is retained in the results rather than excluded.

Evaluation scripts and raw predictions are stored under:

`evaluation/`

---

## Phase 2B — Inference Throughput Optimization

The fine-tuned LoRA adapter was served using vLLM on one NVIDIA H200 GPU through its OpenAI-compatible completion API.

For this experiment, I optimized **aggregate output throughput**.

The following variables were held constant:

- model and LoRA adapter
- GPU
- prompt
- 128 total requests
- 64 generated tokens per request
- temperature = 0
- fixed output length

Request concurrency was varied across:

`1, 4, 8, 16, 32, 64, 128`

### Throughput Results

| Concurrency | Output tokens/s | Requests/s | Average latency |
|---:|---:|---:|---:|
| 1 | 196.52 | 3.07 | 0.326 s |
| 4 | 774.96 | 12.11 | 0.330 s |
| 8 | 1,456.15 | 22.75 | 0.351 s |
| 16 | 2,499.79 | 39.06 | 0.406 s |
| 32 | 3,497.44 | 54.65 | 0.549 s |
| 64 | 3,825.90 | 59.78 | 0.876 s |
| 128 | **4,332.11** | **67.69** | **1.031 s** |

Increasing concurrency from 1 to 128 increased aggregate generation throughput by approximately **22x**.

The experiment also demonstrates the throughput/latency tradeoff. Higher concurrency gives vLLM more opportunities to batch work and keep the GPU busy, but individual request latency increases as load grows.

Concurrency 128 produced the highest throughput among the configurations tested. It should not be interpreted as a universal optimum because the appropriate production configuration depends on workload characteristics and latency requirements.

Detailed methodology and reproduction instructions are documented in:

[`inference/inference_performance.md`](inference/inference_performance.md)

---

## Reproducibility

The repository includes the scripts and measurements used for each stage of the PoC.

### Cluster Validation

```bash
sbatch cluster-validator/scripts/run_validator.slurm
```

The validator writes machine-readable measurements and a human-readable validation report under:

`cluster-validator/results/`

### Training

The training workflow uses Slurm and `torchrun` to launch four distributed processes across two nodes.

Training code and launch scripts are stored under:

`training/`

### Evaluation

Baseline and fine-tuned evaluation scripts are stored under:

`evaluation/`

Both evaluations use the same frozen test split and scoring procedure.

### Inference

The inference benchmark is implemented in:

`inference/benchmark_throughput.py`

The controlled concurrency sweep is implemented in:

`inference/run_throughput_sweep.sh`

Individual benchmark measurements are stored as JSON files under:

`inference/results/`

See [`inference/inference_performance.md`](inference/inference_performance.md) for the complete serving and benchmark procedure.

---

## Environment

The PoC was developed and tested in a Slurm-managed GPU environment with:

- 2 worker nodes
- NVIDIA H200 GPUs
- 4 GPUs allocated for distributed training experiments
- PyTorch
- NCCL
- Enroot
- InfiniBand
- GPU Direct RDMA
- vLLM

The cluster-validation runtime and ML workload environments were kept separate so that changes required for model training and inference did not modify the validated cluster-checking environment.

---

## Scope and Production Considerations

This repository demonstrates a functional PoC rather than a full production deployment.

The distributed training experiment validates the training path across two nodes and four GPUs. The inference experiment intentionally uses a single H200 GPU so the effect of request concurrency can be measured independently.

The results should not be extrapolated linearly to a 512-GPU H100 deployment.

Before production deployment, I would extend the same methodology to include:

- larger-model distributed training
- multi-GPU and multi-node inference
- tensor parallelism
- inference replicas and load balancing
- realistic request-length distributions
- p50 and p95 latency measurements
- longer-duration load and reliability testing
- failure recovery testing
- monitoring and regression thresholds

The goal of this PoC is to establish a reproducible process for validating infrastructure, training behavior, model quality, and serving performance before scaling the workload.