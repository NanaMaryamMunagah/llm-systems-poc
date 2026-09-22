# Cluster Validation Report

**Overall status: PASS**

| Check | Measurement | Status | Acceptance criteria |
|---|---:|---|---|
| worker-0 GPU 0 memory | 139.80 GiB | PASS | Minimum 135 GiB |
| worker-0 GPU 1 memory | 139.80 GiB | PASS | Minimum 135 GiB |
| worker-1 GPU 0 memory | 139.80 GiB | PASS | Minimum 135 GiB |
| worker-1 GPU 1 memory | 139.80 GiB | PASS | Minimum 135 GiB |
| worker-1 GPU 0 BF16 | 805.86 TFLOPS | PASS | Warn < 550, fail < 400 |
| worker-1 GPU 1 BF16 | 768.05 TFLOPS | PASS | Warn < 550, fail < 400 |
| worker-0 GPU 0 BF16 | 797.94 TFLOPS | PASS | Warn < 550, fail < 400 |
| worker-0 GPU 1 BF16 | 783.16 TFLOPS | PASS | Warn < 550, fail < 400 |
| worker-0 GPU 1 HBM | 4277.22 GB/s | PASS | Warn < 3400, fail < 2500 |
| worker-0 GPU 0 HBM | 4276.03 GB/s | PASS | Warn < 3400, fail < 2500 |
| worker-1 GPU 0 HBM | 4275.76 GB/s | PASS | Warn < 3400, fail < 2500 |
| worker-1 GPU 1 HBM | 4275.06 GB/s | PASS | Warn < 3400, fail < 2500 |
| Multi-node NCCL correctness | True | PASS | All ranks must receive expected AllReduce result |
| NCCL transport | IB | PASS | Expected InfiniBand transport |
| GPU Direct RDMA | True | PASS | Expected for high-performance multi-node communication |
| Inter-node AllReduce algorithm bandwidth | 63.17 GB/s | PASS | Warn < 20, fail < 8 |
| Shared storage read | 920.00 MB/s | PASS | Warn < 500, fail < 100 |
