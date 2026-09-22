import json
import os
import socket
import time
import torch

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])

torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

# Test GPU compute performance using a large BF16 matrix multiplication..
N = 8192

a = torch.randn((N, N), dtype=torch.bfloat16, device=device)
b = torch.randn((N, N), dtype=torch.bfloat16, device=device)

# Run a few times before measuring so the initial GPU setup doesn't affect the results.
for _ in range(5):
    torch.matmul(a, b)

torch.cuda.synchronize()

iterations = 10
start = time.perf_counter()

for _ in range(iterations):
    torch.matmul(a, b)

torch.cuda.synchronize()

elapsed = time.perf_counter() - start
avg_time = elapsed / iterations

# Calculate the number of operations used to measure GPU compute performance.
flops = 2 * (N ** 3)
tflops = flops / avg_time / 1e12

result = {
    "rank": rank,
    "host": socket.gethostname(),
    "gpu": local_rank,
    "gpu_name": torch.cuda.get_device_name(device),
    "matrix_size": N,
    "avg_time_ms": round(avg_time * 1000, 3),
    "tflops": round(tflops, 2),
}

print("RESULT_JSON=" + json.dumps(result), flush=True)
