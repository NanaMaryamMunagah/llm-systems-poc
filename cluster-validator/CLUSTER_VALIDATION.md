# GPU Cluster Validator

A lightweight, containerized validation suite for checking whether a multi-node GPU cluster is ready for distributed LLM training and inference workloads.

This validator was developed for a proof-of-concept environment consisting of two Slurm worker nodes with four NVIDIA H200 GPUs allocated across the nodes. The goal is to detect infrastructure problems before running expensive training or inference workloads.

## What It Validates

The suite checks:

- GPU visibility, model, and available memory
- BF16 matrix multiplication performance
- HBM memory bandwidth
- Multi-node NCCL collective correctness
- Inter-node NCCL AllReduce bandwidth
- NCCL transport selection
- GPU Direct RDMA usage
- Shared storage read performance

The validator produces both human-readable and machine-readable reports with PASS, WARN, or FAIL status.

## Why These Checks Matter

A cluster can appear functional while still having significant performance problems.

For example, a distributed NCCL operation may complete correctly while silently falling back from InfiniBand/RDMA to TCP sockets. For this reason, the validator checks both collective correctness and the actual NCCL transport path.

The acceptance thresholds in this project are conservative proof-of-concept health checks intended to identify major regressions or misconfiguration. They are not vendor certification thresholds or guaranteed hardware specifications.

## Container

The validator uses the following runtime-focused PyTorch image:

```text
pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime
```

The container also installs the RDMA userspace packages:

```text
libibverbs1
ibverbs-providers
```

These packages allow NCCL to use the available InfiniBand devices from inside the container.

The runtime image was selected instead of the larger NVIDIA PyTorch development image to reduce container size while retaining the CUDA, PyTorch, NCCL, and RDMA functionality required by the validator.

The compressed base image was approximately 5.3 GB compared with approximately 23 GB for the development image used during initial testing.

### Build

On a system with Docker available:

```bash
docker build -t cluster-validator:latest .
```

The test cluster did not expose a Docker daemon on the compute environment, so Enroot was used to execute the container under Slurm.

## Running the Validator

Submit the validation suite through Slurm:

```bash
sbatch scripts/run_validator.slurm
```

The validation job requests:

- 2 worker nodes
- 2 GPUs per node
- 4 GPUs total

The launcher runs the individual tests, collects their measurements, and automatically generates the final validation report.

## Validation Results

The final end-to-end validation was performed across four NVIDIA H200 GPUs on two worker nodes.

| Check | Result |
|---|---:|
| GPU memory | 139.80 GiB per GPU |
| BF16 compute | 769.24–806.20 TFLOPS |
| HBM bandwidth | 4275.74–4278.74 GB/s |
| Multi-node NCCL correctness | PASS |
| NCCL transport | InfiniBand |
| GPU Direct RDMA | Enabled |
| 256 MiB 4-GPU AllReduce algorithm bandwidth | 63.23 GB/s |
| Shared storage read | 880 MB/s |
| Overall status | **PASS** |

The complete generated reports are available at:

```text
results/validation_report.md
results/validation_report.json
```

The normalized measurements used to generate the reports are stored in:

```text
results/measurements.json
```

Raw logs from the final validation run are retained under:

```text
results/raw_473/
```

## Container Networking Validation

During development, the smaller PyTorch runtime container initially passed the multi-node NCCL correctness test, but NCCL diagnostics showed that inter-node communication was using:

```text
NET/Socket
```

The container could see the host's `mlx5` devices, but inspection showed that the required RDMA userspace libraries were missing.

Adding:

```text
libibverbs1
ibverbs-providers
```

restored the intended communication path. NCCL diagnostics subsequently reported channels using:

```text
NET/IB/.../GDRDMA
```

This confirmed that inter-node NCCL traffic was using InfiniBand with GPU Direct RDMA rather than silently falling back to socket transport.

After the change, the lightweight runtime container achieved approximately 63.3 GB/s algorithm bandwidth for the 256 MiB four-GPU AllReduce benchmark. The larger NVIDIA PyTorch development image measured approximately 63.4 GB/s in the same lightweight benchmark, so no meaningful performance difference was observed in this test.

This result demonstrates why the validator checks transport selection and performance in addition to distributed correctness.

## Repository Structure

```text
cluster-validator/
├── Dockerfile
├── README.md
├── scripts/
│   └── run_validator.slurm
├── validator/
│   ├── gpu_check.py
│   ├── bf16_benchmark.py
│   ├── hbm_benchmark.py
│   ├── nccl_check.py
│   ├── nccl_benchmark.py
│   ├── collect_results.py
│   └── report.py
└── results/
    ├── measurements.json
    ├── validation_report.json
    ├── validation_report.md
    └── raw_473/
```

## Validator Components

### GPU Inventory

`gpu_check.py` verifies that the allocated GPUs are visible to PyTorch and records the GPU model and available memory.

### BF16 Compute

`bf16_benchmark.py` performs BF16 matrix multiplication on each allocated GPU and reports application-level throughput in TFLOPS.

This is intended as a lightweight compute health check rather than an official hardware certification benchmark.

### HBM Bandwidth

`hbm_benchmark.py` performs large GPU memory copies and reports effective memory bandwidth.

The reported value counts both bytes read and bytes written.

### NCCL Correctness

`nccl_check.py` initializes a multi-node NCCL process group and performs an AllReduce operation. Each rank verifies that the resulting value matches the expected sum.

This confirms that distributed GPU communication is functionally correct.

### NCCL Bandwidth and Transport

`nccl_benchmark.py` performs repeated 256 MiB AllReduce operations across four GPUs and reports algorithm and bus bandwidth.

NCCL diagnostics are also collected so that the validator can determine whether communication is using InfiniBand and GPU Direct RDMA.

### Shared Storage

The Slurm launcher performs a simple 1 GiB shared-filesystem write/read test to detect major storage performance problems before starting the target workload.

## Reporting

The individual tests emit structured JSON measurements.

`collect_results.py` parses the raw output and combines the measurements into:

```text
results/measurements.json
```

`report.py` evaluates those measurements against the configured acceptance thresholds and generates:

```text
results/validation_report.md
results/validation_report.json
```

Each check is classified as PASS, WARN, or FAIL.

A FAIL represents a condition that should be investigated before running the target ML workload. A WARN represents degraded or unexpected behavior that may warrant investigation but does not necessarily prevent execution.

## Acceptance Criteria

The thresholds used by this validator are intentionally conservative proof-of-concept health checks. Their purpose is to catch large regressions, missing functionality, or obvious infrastructure misconfiguration before consuming GPU time on training or inference.

Examples include:

- H200 GPU memory must be at least 135 GiB
- NCCL AllReduce must produce the expected result on every rank
- Multi-node communication is expected to use InfiniBand
- GPU Direct RDMA is expected for the high-performance multi-node path
- Inter-node AllReduce algorithm bandwidth warns below 20 GB/s and fails below 8 GB/s
- Shared storage read performance warns below 500 MB/s and fails below 100 MB/s

These values should not be interpreted as NVIDIA or Nebius hardware guarantees.

## Scope and Limitations

The validator is intentionally lightweight so that it can run before an expensive training or inference workload without consuming significant cluster time.

The BF16 and HBM tests are application-level PyTorch checks rather than official hardware certification benchmarks.

The NCCL bandwidth test is a lightweight PyTorch distributed benchmark rather than the full `nccl-tests` suite.

The storage check is a basic shared-filesystem smoke and performance test rather than a comprehensive storage benchmark.

For a production acceptance process, the validator could be extended with:

- additional NCCL collective types and message sizes
- small-message communication latency
- GPU peer-to-peer bandwidth
- host-to-device transfer bandwidth
- sustained-duration compute tests
- dedicated storage benchmarks
- topology-aware performance validation

## Final Result

The final validation run passed all configured checks across the four allocated H200 GPUs.

The cluster demonstrated functional multi-node NCCL communication, InfiniBand transport with GPU Direct RDMA, consistent BF16 compute and HBM performance across GPUs, and shared storage performance above the configured proof-of-concept thresholds.
