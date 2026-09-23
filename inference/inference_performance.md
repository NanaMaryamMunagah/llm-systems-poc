# Inference Performance Evaluation

## Objective

The customer plans to host its own inference service after fine-tuning the model. For the inference portion of the PoC, I optimized for **aggregate output throughput**, measured in output tokens per second.

Throughput was selected because it measures how much generation work the server can complete as request load increases. Request latency was also recorded to show the tradeoff between maximizing throughput and individual request response time.

## Serving Setup

The fine-tuned Gemma 3 4B model was served using vLLM on one NVIDIA H200 GPU.

The base model was loaded from:

`/workspace/models/gemma-3-4b-pt`

The LoRA adapter produced during fine-tuning was loaded as:

`mmlu-math`

The server used:

- BF16 precision
- maximum model length of 2048 tokens
- vLLM generation configuration
- OpenAI-compatible completion API
- one NVIDIA H200 GPU

The LoRA adapter was served directly through vLLM rather than merging it into the base model.

## Benchmark Design

The benchmark sends the same prompt repeatedly to the inference server while varying request concurrency.

The following variables were held constant:

- model and LoRA adapter
- GPU
- prompt
- 128 total requests per experiment
- 64 generated tokens per request
- temperature = 0
- EOS termination disabled so each request generates the same number of output tokens

The only experimental variable was request concurrency:

`1, 4, 8, 16, 32, 64, 128`

For each run, I measured:

- aggregate output tokens per second
- requests per second
- average request latency

Using a fixed output length makes the throughput comparison less sensitive to differences in when individual generations would normally terminate.

## Results

| Concurrency | Output tokens/s | Requests/s | Average latency (s) |
|------------:|----------------:|-----------:|--------------------:|
| 1 | 196.52 | 3.07 | 0.326 |
| 4 | 774.96 | 12.11 | 0.330 |
| 8 | 1,456.15 | 22.75 | 0.351 |
| 16 | 2,499.79 | 39.06 | 0.406 |
| 32 | 3,497.44 | 54.65 | 0.549 |
| 64 | 3,825.90 | 59.78 | 0.876 |
| 128 | 4,332.11 | 67.69 | 1.031 |

The highest observed throughput was **4,332.11 output tokens/s at concurrency 128**.

Compared with concurrency 1, this represents approximately a **22x increase in aggregate output throughput**.

The improvement comes with a latency tradeoff. Average request latency increased from approximately 0.33 seconds at concurrency 1 to 1.03 seconds at concurrency 128.

The results show why concurrency is an important serving parameter. At low concurrency, the GPU has less work available for batching. Increasing concurrent requests gives vLLM more opportunities to batch generation work and improves GPU utilization and aggregate throughput.

The throughput improvement begins to flatten at higher concurrency levels. For example, the increase from concurrency 32 to 64 is smaller than the increases observed at lower concurrency levels, while latency increases more noticeably.

Concurrency 128 produced the highest throughput among the configurations tested, but it should not be interpreted as a universal optimum. The appropriate production setting depends on workload characteristics and the customer's latency service-level objectives.

## Reproducing the Benchmark

Start the vLLM server with the fine-tuned adapter:

```bash
srun \
  --partition=earlytalent \
  --qos=gpulimit \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:1 \
  --time=00:30:00 \
  enroot start \
  --mount "$HOME/llm-systems-poc-clean:/workspace" \
  llm-inference \
  /workspace/models/gemma-3-4b-pt \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --generation-config vllm \
  --enable-lora \
  --lora-modules mmlu-math=/workspace/training/results/gemma3-4b-mmlu-math-lora/final_adapter
```

Because the vLLM server binds to `127.0.0.1`, the benchmark client must run on the same compute node as the server. After starting the server through Slurm, use its job ID to launch the benchmark as an overlapping step in the same allocation:

```bash
srun \
  --jobid=<VLLM_JOB_ID> \
  --overlap \
  --ntasks=1 \
  /home/nmm/llm-systems-poc-clean/inference/run_throughput_sweep.sh
```

Individual benchmark results are written to:

`inference/results/`

## Scope and Limitations

This experiment evaluates the serving path on a single H200 GPU. It demonstrates the effect of request concurrency and vLLM batching on inference throughput, but the results should not be extrapolated linearly to the customer's proposed 512-GPU H100 cluster.

A production evaluation would additionally test multi-GPU serving, tensor parallelism, multiple replicas, realistic prompt and output length distributions, longer-duration load tests, and latency percentiles such as p50 and p95.

The single-GPU experiment is intended as a controlled PoC that validates the inference software path and provides a reproducible method for performance tuning before scaling to a larger deployment.