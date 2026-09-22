import json
import socket
import torch

host = socket.gethostname()

result = {
    "host": host,
    "pytorch_version": torch.__version__,
    "pytorch_cuda_version": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
    "gpus": [],
}

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)

        result["gpus"].append({
            "host": host,
            "gpu": i,
            "name": props.name,
            "memory_gib": round(props.total_memory / (1024 ** 3), 2),
        })
print("RESULT_JSON=" + json.dumps(result))
