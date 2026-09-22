import json
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python report.py <measurements.json>")
    sys.exit(2)

input_path = Path(sys.argv[1])

with open(input_path) as f:
    data = json.load(f)

# Conservative PoC thresholds.
thresholds = {
    "gpu_memory_gib_min": 135,
    "bf16_tflops_warn": 550,
    "bf16_tflops_fail": 400,
    "hbm_gbs_warn": 3400,
    "hbm_gbs_fail": 2500,
    "inter_node_alg_bw_gbs_warn": 20,
    "inter_node_alg_bw_gbs_fail": 8,
    "storage_read_mbs_warn": 500,
    "storage_read_mbs_fail": 100,
}

checks = []


def add_check(name, value, status, detail):
    checks.append({
        "name": name,
        "value": value,
        "status": status,
        "detail": detail,
    })


# GPU memory
for gpu in data["gpus"]:
    memory = gpu["memory_gib"]

    status = (
        "PASS"
        if memory >= thresholds["gpu_memory_gib_min"]
        else "FAIL"
    )

    add_check(
        f'{gpu["host"]} GPU {gpu["gpu"]} memory',
        f"{memory:.2f} GiB",
        status,
        f'Minimum {thresholds["gpu_memory_gib_min"]} GiB',
    )


# BF16 compute
for gpu in data["bf16"]:
    value = gpu["tflops"]

    if value < thresholds["bf16_tflops_fail"]:
        status = "FAIL"
    elif value < thresholds["bf16_tflops_warn"]:
        status = "WARN"
    else:
        status = "PASS"

    add_check(
        f'{gpu["host"]} GPU {gpu["gpu"]} BF16',
        f"{value:.2f} TFLOPS",
        status,
        f'Warn < {thresholds["bf16_tflops_warn"]}, '
        f'fail < {thresholds["bf16_tflops_fail"]}',
    )


# HBM bandwidth
for gpu in data["hbm"]:
    value = gpu["bandwidth_gbs"]

    if value < thresholds["hbm_gbs_fail"]:
        status = "FAIL"
    elif value < thresholds["hbm_gbs_warn"]:
        status = "WARN"
    else:
        status = "PASS"

    add_check(
        f'{gpu["host"]} GPU {gpu["gpu"]} HBM',
        f"{value:.2f} GB/s",
        status,
        f'Warn < {thresholds["hbm_gbs_warn"]}, '
        f'fail < {thresholds["hbm_gbs_fail"]}',
    )


# NCCL correctness
add_check(
    "Multi-node NCCL correctness",
    data["nccl"]["correctness"],
    "PASS" if data["nccl"]["correctness"] else "FAIL",
    "All ranks must receive expected AllReduce result",
)


# NCCL transport
transport = data["nccl"]["transport"]

add_check(
    "NCCL transport",
    transport,
    "PASS" if transport == "IB" else "FAIL",
    "Expected InfiniBand transport",
)


# GPU Direct RDMA
gdr = data["nccl"]["gpu_direct_rdma"]

add_check(
    "GPU Direct RDMA",
    gdr,
    "PASS" if gdr else "WARN",
    "Expected for high-performance multi-node communication",
)


# Inter-node NCCL bandwidth
value = data["nccl"]["inter_node_alg_bw_gbs"]

if value < thresholds["inter_node_alg_bw_gbs_fail"]:
    status = "FAIL"
elif value < thresholds["inter_node_alg_bw_gbs_warn"]:
    status = "WARN"
else:
    status = "PASS"

add_check(
    "Inter-node AllReduce algorithm bandwidth",
    f"{value:.2f} GB/s",
    status,
    f'Warn < {thresholds["inter_node_alg_bw_gbs_warn"]}, '
    f'fail < {thresholds["inter_node_alg_bw_gbs_fail"]}',
)


# Storage read
value = data["storage"]["read_mbs"]

if value < thresholds["storage_read_mbs_fail"]:
    status = "FAIL"
elif value < thresholds["storage_read_mbs_warn"]:
    status = "WARN"
else:
    status = "PASS"

add_check(
    "Shared storage read",
    f"{value:.2f} MB/s",
    status,
    f'Warn < {thresholds["storage_read_mbs_warn"]}, '
    f'fail < {thresholds["storage_read_mbs_fail"]}',
)


statuses = [c["status"] for c in checks]

if "FAIL" in statuses:
    overall = "FAIL"
elif "WARN" in statuses:
    overall = "WARN"
else:
    overall = "PASS"


report = {
    "overall_status": overall,
    "thresholds": thresholds,
    "checks": checks,
    "measurements": data,
}


results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

json_path = results_dir / "validation_report.json"

with open(json_path, "w") as f:
    json.dump(report, f, indent=2)


md_path = results_dir / "validation_report.md"

with open(md_path, "w") as f:
    f.write("# Cluster Validation Report\n\n")
    f.write(f"**Overall status: {overall}**\n\n")

    f.write("| Check | Measurement | Status | Acceptance criteria |\n")
    f.write("|---|---:|---|---|\n")

    for check in checks:
        f.write(
            f'| {check["name"]} | '
            f'{check["value"]} | '
            f'{check["status"]} | '
            f'{check["detail"]} |\n'
        )

print(f"Overall status: {overall}")
print(f"Wrote {json_path}")
print(f"Wrote {md_path}")

if overall == "FAIL":
    sys.exit(1)
