import json
import re
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python collect_results.py <raw_results_dir>")
    sys.exit(2)

raw_dir = Path(sys.argv[1])


def load_results(filename):
    path = raw_dir / filename
    text = path.read_text()

    results = []

    # Find and parse each result separately in case outputs from different processes run together.
    decoder = json.JSONDecoder()

    for match in re.finditer(r"RESULT_JSON=", text):
        start = match.end()

        try:
            result, _ = decoder.raw_decode(text[start:].lstrip())
            results.append(result)
        except json.JSONDecodeError as exc:
            print(
                f"WARNING: Could not parse result in {filename}: {exc}",
                file=sys.stderr,
            )

    return results


gpu_results = load_results("gpu.log")
bf16_results = load_results("bf16.log")
hbm_results = load_results("hbm.log")
nccl_check_results = load_results("nccl_check.log")
nccl_bw_results = load_results("nccl_benchmark.log")


gpus = []
for result in gpu_results:
    gpus.extend(result["gpus"])


nccl_correctness = (
    len(nccl_check_results) > 0
    and all(result["passed"] for result in nccl_check_results)
)


if nccl_bw_results:
    slowest = min(
        nccl_bw_results,
        key=lambda x: x["alg_bw_gbs"],
    )
    inter_node_alg_bw = slowest["alg_bw_gbs"]
else:
    inter_node_alg_bw = 0.0


nccl_log = (raw_dir / "nccl_benchmark.log").read_text()

uses_ib = (
    "NET/IB" in nccl_log
    or "IBext" in nccl_log
)

uses_gdr = (
    "GPU Direct RDMA Enabled" in nccl_log
    or "GDRDMA" in nccl_log
)


storage_path = raw_dir / "storage.json"

if storage_path.exists():
    with open(storage_path) as f:
        storage = json.load(f)
else:
    storage = {
        "read_mbs": 0.0,
        "write_mbs": 0.0,
    }


measurements = {
    "gpus": gpus,
    "bf16": bf16_results,
    "hbm": hbm_results,
    "nccl": {
        "correctness": nccl_correctness,
        "transport": "IB" if uses_ib else "UNKNOWN",
        "gpu_direct_rdma": uses_gdr,
        "inter_node_alg_bw_gbs": inter_node_alg_bw,
    },
    "storage": storage,
}


output = Path("results/measurements.json")
output.parent.mkdir(exist_ok=True)

with open(output, "w") as f:
    json.dump(measurements, f, indent=2)

print(f"Wrote {output}")
