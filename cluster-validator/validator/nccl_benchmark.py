import json
import os
import time
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

dist.init_process_group(
    backend="nccl",
    init_method="env://",
)

# 256 MiB float32 tensor per GPU.
num_elements = 256 * 1024 * 1024 // 4
tensor = torch.ones(
    num_elements,
    dtype=torch.float32,
    device=device,
)

tensor_bytes = tensor.numel() * tensor.element_size()

# Warm-up.
for _ in range(5):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

torch.cuda.synchronize()
dist.barrier()

iterations = 20

start = time.perf_counter()

for _ in range(iterations):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

torch.cuda.synchronize()

elapsed = time.perf_counter() - start
avg_time = elapsed / iterations

# Algorithm bandwidth.
alg_bw = tensor_bytes / avg_time / 1e9

# Conventional AllReduce bus-bandwidth normalization.
bus_bw = alg_bw * (2 * (world_size - 1) / world_size)

result = {
    "rank": rank,
    "world_size": world_size,
    "tensor_mib": 256,
    "avg_time_ms": round(avg_time * 1000, 3),
    "alg_bw_gbs": round(alg_bw, 2),
    "bus_bw_gbs": round(bus_bw, 2),
}

print("RESULT_JSON=" + json.dumps(result), flush=True)

dist.destroy_process_group()
