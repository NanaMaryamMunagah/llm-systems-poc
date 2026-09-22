import json
import os
import socket
import sys
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

tensor = torch.tensor([float(rank + 1)], device=device)

dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

expected = world_size * (world_size + 1) / 2
actual = tensor.item()
passed = actual == expected

result = {
    "rank": rank,
    "host": socket.gethostname(),
    "gpu": local_rank,
    "world_size": world_size,
    "expected": expected,
    "actual": actual,
    "passed": passed,
}

print("RESULT_JSON=" + json.dumps(result), flush=True)

dist.barrier()
dist.destroy_process_group()

if not passed:
    sys.exit(1)
