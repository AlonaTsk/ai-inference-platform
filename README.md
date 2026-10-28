# AI Inference Platform

[![CI](https://github.com/AlonaTsk/ai-inference-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/AlonaTsk/ai-inference-platform/actions/workflows/ci.yml)

This project classifies customer-support tickets into **account**, **billing**, **shipping** or **technical**. It has a **Go REST API** in front of a **Python inference service**. Both run in Docker and deploy to a local **Kubernetes (kind)** cluster with health probes, resource limits and replicas.

It is a learning and portfolio project, built to practise the workflow end to end: train, evaluate, serve, containerise, deploy, test and benchmark.

```
client ──► Go API (validation, timeouts, error mapping) ──► Python inference (TF-IDF + logistic regression)
           Deployment ×2 + Service                          Deployment ×2 + Service
```

More detail: [architecture](docs/architecture.md) · [GPU](docs/gpu.md) · [design decisions](docs/decisions.md) · [troubleshooting log](docs/troubleshooting.md) · [benchmarks](benchmarks/RESULTS.md)

## Results at a glance

| | |
|---|---|
| Held-out test set | 800 tickets (20%, stratified, split before fitting) |
| Accuracy / macro F1 | **0.954 / 0.954** ([full metrics + confusion matrix](docs/metrics.json)) |
| Model latency (CPU, batch 1) | 0.56 ms p50 |
| End-to-end latency via API (concurrency 1) | 3.7 ms p50, 251 req/s |
| GPU: PyTorch model on Tesla T4 (Colab) | batch 4096: 3.3 ms FP32 / 1.1 ms FP16, about 14× / 43× faster than a 16-thread laptop CPU; batch 1: the CPU is about 2× faster ([details](docs/gpu.md#gpu-results-tesla-t4-google-colab)) |
| Tests | 13 Go test cases + 17 pytest cases (one CUDA-only, skipped without a GPU) + an end-to-end script run on kind in CI |

> **The data is synthetic.** The tickets are generated from templates, with cross-class distractor phrases and 5% deliberately flipped labels (see [decisions](docs/decisions.md#3-synthetic-data-explicitly-labelled-as-such)). About 95% is roughly the best any model can score on this data, so these numbers show the pipeline works. They do not predict accuracy on real tickets.

## Repository layout

```
api/                  Go service: POST /v1/predict, /healthz, /readyz
inference/            Python: data generator, training, FastAPI inference service, tests
deployments/k8s/      Namespace, ConfigMap, Deployments, Services (Kustomize) + kind config
                      gpu/: GPU benchmark Job; gpu-mock/: kind overlay for the mocked-GPU demo
benchmarks/           CPU benchmark scripts and recorded results
gpu/                  PyTorch model, CUDA-aware CPU/GPU benchmark, Colab notebook
tests/e2e.sh          End-to-end smoke test against a running stack
.github/workflows/    CI: Go tests, Python tests + training, kind deploy + e2e
docs/                 Architecture, decisions, troubleshooting, metrics
```

## Prerequisites

Go 1.23+, Python 3.10+, Docker, plus [kind](https://kind.sigs.k8s.io/docs/user/quick-start/) and kubectl for the Kubernetes part.

## Run it

### 1. Unit tests and training

```bash
make venv          # create .venv and install Python deps
make test          # go vet + go test -race, pytest
make train         # train, print the classification report, write inference/model/
```

### 2. Docker Compose

```bash
make up            # builds both images; the inference image trains the model during the build
make e2e           # end-to-end smoke test
make down
```

Is port 8080 busy? Use `make up API_PORT=18081` and `make e2e API_PORT=18081`.

### 3. Kubernetes (kind)

```bash
make kind-up       # create the cluster
make kind-deploy   # build, load images into kind, apply manifests, wait for rollout
make kind-e2e      # port-forward svc/api and run the e2e test
make kind-down     # delete the cluster
```

Inspect the cluster:

```bash
kubectl -n ai-inference get pods,svc
kubectl -n ai-inference logs deploy/api
kubectl -n ai-inference describe pod -l app=inference
kubectl -n ai-inference port-forward svc/api 8080:80
```

## API

### `POST /v1/predict`

```bash
curl -s -X POST localhost:8080/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "I was charged twice and the app keeps crashing"}'
```

```json
{
  "label": "billing",
  "confidence": 0.6373,
  "scores": {"account": 0.0791, "billing": 0.6373, "shipping": 0.069, "technical": 0.2146},
  "model_version": "2026-10-28T15:06:29+00:00"
}
```

The ticket above mixes two topics, so the confidence is lower and "technical" scores second.

| Status | When |
|---|---|
| 200 | Prediction returned |
| 400 | Body is not `{"text": "..."}`, the text is empty, or it is longer than 2000 characters |
| 502 | Inference service unreachable, returned an error, or returned an invalid response |
| 504 | Inference service did not answer within `INFERENCE_TIMEOUT` (default 3s) |

### Health

| Endpoint | Used by | Checks |
|---|---|---|
| `GET /healthz` | liveness probe | The process is up |
| `GET /readyz` | readiness probe | The inference service is ready |

### Configuration (API)

| Variable | Default | |
|---|---|---|
| `INFERENCE_URL` | `http://localhost:8000` | Base URL of the inference service |
| `INFERENCE_TIMEOUT` | `3s` | Per-request upstream timeout |
| `PORT` | `8080` | Listen port |

The inference service also exposes `GET /metadata`, which returns the training metrics of the loaded model.

## Benchmarks

`make bench` runs both CPU benchmarks. The recorded results, hardware and limitations are in [benchmarks/RESULTS.md](benchmarks/RESULTS.md). In short, the model takes about 0.6 ms per ticket. Throughput levels off at about 400 req/s because of the single Python HTTP worker, not the model.

### GPU

[`gpu/`](gpu) contains a PyTorch version of the classifier and a CUDA-aware benchmark. It covers warmup, `torch.cuda.synchronize()`, compute versus end-to-end timing including host↔device copies, and FP16. It also has a Kubernetes GPU Job (`nvidia.com/gpu` limits, tolerations) and a scheduling demo on kind with a **mocked** GPU. It was benchmarked on a real **Tesla T4** in Google Colab with the [notebook](gpu/colab_benchmark.ipynb). The GPU wins only with batching: at batch 1 a multi-core CPU has lower latency. Details: [docs/gpu.md](docs/gpu.md).

## Limitations and next steps

- The data is synthetic; the next step is real labelled tickets and a confidence-calibration check.
- The model is trained at image build time; in production it would be a versioned artifact.
- There is no Ingress, TLS, autoscaling or metrics endpoint yet.
- The GPU was benchmarked on hosted Colab only. The GPU Kubernetes Job has not run on a real GPU node (scheduling was demonstrated with a mocked GPU), and the HTTP service still serves the CPU model.

See [docs/decisions.md](docs/decisions.md#what-i-would-change-before-production) for the full list.
