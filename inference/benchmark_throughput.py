import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


PROMPT = """Explain why distributed training is useful for training large machine learning models. Give a clear technical explanation."""


def send_request(url, model, max_tokens):
    payload = {
        "model": model,
        "prompt": PROMPT,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()

    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))

    latency = time.perf_counter() - start

    return {
        "latency": latency,
        "output_tokens": result["usage"]["completion_tokens"],
    }


def run_benchmark(url, model, concurrency, requests, max_tokens):
    start = time.perf_counter()

    results = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_request, url, model, max_tokens)
            for _ in range(requests)
        ]

        for future in as_completed(futures):
            results.append(future.result())

    total_time = time.perf_counter() - start

    total_tokens = sum(result["output_tokens"] for result in results)
    average_latency = sum(result["latency"] for result in results) / len(results)

    throughput = total_tokens / total_time
    request_throughput = requests / total_time

    return {
        "concurrency": concurrency,
        "requests": requests,
        "output_tokens_per_request": max_tokens,
        "total_output_tokens": total_tokens,
        "total_time_seconds": total_time,
        "output_tokens_per_second": throughput,
        "requests_per_second": request_throughput,
        "average_latency_seconds": average_latency,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--model", default="mmlu-math")
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", type=str)

    args = parser.parse_args()

    result = run_benchmark(
        args.url,
        args.model,
        args.concurrency,
        args.requests,
        args.max_tokens,
    )

    print(json.dumps(result, indent=2))

    if args.output:
        with open(args.output, "w") as file:
            json.dump(result, file, indent=2)


if __name__ == "__main__":
    main()