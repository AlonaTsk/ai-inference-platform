# Benchmark results (CPU only)

**Measured:** 2026-10-28
**Hardware:** MacBook Pro, Intel Core i9-9880H (8 cores / 16 threads, 2.3 GHz), macOS, Docker Desktop
**Model:** TF-IDF (1–2-grams) + logistic regression, 4 classes, scikit-learn 1.5.2

> These results are for the scikit-learn service. It runs on CPU only; a linear model on sparse TF-IDF features would not benefit from a GPU. GPU work is under [GPU](#gpu).

## 1. Model only, in-process (`model_bench.py`)

Python 3.10.11, single process, 20 warmup iterations and then 200 timed iterations per batch size. Texts are freshly generated (seed 999), not training samples.

| batch | p50 ms | p95 ms | throughput (texts/s) |
|------:|-------:|-------:|---------------------:|
| 1 | 0.557 | 0.660 | 1,796 |
| 8 | 0.772 | 1.035 | 10,362 |
| 32 | 1.443 | 1.786 | 22,183 |
| 64 | 2.222 | 2.661 | 28,805 |
| 256 | 6.278 | 7.836 | 40,780 |

Batching amortises per-call overhead: throughput rises about 23× from batch 1 to batch 256, while latency rises only about 11×.

## 2. End to end over HTTP (`http_bench.py`)

Path: client → Go API (Docker Compose) → Python inference (1 uvicorn worker). 100 warmup requests, then 2000 timed requests, one text per request. Client and server ran on the same laptop.

| concurrency | requests | errors | p50 ms | p95 ms | p99 ms | throughput (req/s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2000 | 0 | 3.7 | 5.2 | 6.3 | 251 |
| 8 | 2000 | 0 | 20.3 | 30.2 | 36.0 | 351 |
| 32 | 2000 | 0 | 73.6 | 97.2 | 108.7 | 417 |

## Interpretation

- The model takes about 0.6 ms of the 3.7 ms single-request latency. Most of the time goes to HTTP, JSON and pydantic serialisation, and the Docker network hop.
- Throughput levels off at about 400 req/s while latency grows with concurrency. Requests are **queueing** in front of a single Python worker, which is the classic latency-versus-throughput trade-off. The next steps would be more uvicorn workers or replicas, or batching several API requests into one model call.

## Limitations

- The client, Docker VM and services share one laptop's CPUs, so the client competes with the server.
- The Python HTTP client (thread pool + urllib) has its own overhead. A dedicated load tool (k6, hey) would push harder.
- The runs are short and single, with no repeated trials or confidence intervals.
- The results are from Docker Compose, not kind.

## GPU

The PyTorch model, the CUDA-aware benchmark and the Colab notebook are in [`gpu/`](../gpu). Method, CPU results and the pending GPU run are described in [docs/gpu.md](../docs/gpu.md). Note that this file benchmarks the scikit-learn model, while `gpu/` benchmarks a different (PyTorch) model, so the numbers are not directly comparable.
