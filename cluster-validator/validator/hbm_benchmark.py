import json
import os
import socket
import time
import torch

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])

torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

# 2 GiB BF16 tensor.
num_elements = 2 * 1024**3 // 2

src = torch.empty(num_elements, dtype=torch.bfloat16, device=device)
dst = torch.empty_like(src)

# Warm-up.
for _ in range(5):
    dst.copy_(src)

torch.cuda.synchronize()

iterations = 20
start = time.perf_counter()

for _ in range(iterations):
    dst.copy_(src)

torch.cuda.synchronize()

elapsed = time.perf_counter() - start
avg_time = elapsed / iterations

bytes_per_tensor = src.numel() * src.element_size()

# Each copy reads src and writes dst.
bytes_moved = 2 * bytes_per_tensor
bandwidth_gbs = bytes_moved / avg_time / 1e9

result = {
    "rank": rank,
    "host": socket.gethostname(),
    "gpu": local_rank,
    "gpu_name": torch.cuda.get_device_name(device),
    "avg_time_ms": round(avg_time * 1000, 3),
    "bandwidth_gbs": round(bandwidth_gbs, 2),
}

print("RESULT_JSON=" + json.dumps(result), flush=True)
