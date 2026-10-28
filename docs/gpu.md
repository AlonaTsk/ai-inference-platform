# GPU: PyTorch model, Colab benchmark, Kubernetes GPU scheduling

There are three separate pieces here, which are easy to confuse:

| Piece | Where it runs | What it proves | Status |
|---|---|---|---|
| PyTorch classifier + CUDA-aware benchmark (`gpu/`) | Any machine | The code places the model and data on a device correctly and times it correctly | Done; tested on CPU |
| Real GPU numbers | Google Colab (Tesla T4) | Measured CPU-versus-GPU latency and throughput | Done 2026-10-28 ([results](#gpu-results-tesla-t4-google-colab)) |
| Kubernetes GPU Job (`deployments/k8s/gpu/`) | A cluster with NVIDIA nodes | How a GPU workload is requested and scheduled | Manifest written; scheduling demonstrated on kind with a **mocked** GPU |

## Why there is no CUDA locally

The development machine is an Intel MacBook Pro (i9-9880H, Intel/AMD graphics). CUDA needs an NVIDIA GPU, and Docker on macOS runs inside a Linux VM that cannot get an NVIDIA device anyway. A GPU container needs a Linux host with:
- the NVIDIA driver,
- the NVIDIA Container Toolkit,
- in Kubernetes, the NVIDIA device plugin (or GPU Operator), which both **advertises** `nvidia.com/gpu` on the node and **injects** the device into pods that request it.

## The PyTorch model (`gpu/textclf.py`)

- Same synthetic tickets as the scikit-learn service. The shuffled 80/20 split happens before the vocabulary is built.
- `EmbeddingBag` (mean of unigram and bigram embeddings, dim 256) → MLP 1024 → 1024 → 4. About 1.5M parameters plus the embedding table.
- Held-out results on CPU: **accuracy 0.928, macro F1 0.927** (8 epochs). This is slightly *below* the TF-IDF + logistic regression baseline (0.954). The production service keeps the linear model: it is more accurate here, smaller and faster. The PyTorch model exists to exercise the GPU path, not to replace it.

## Benchmark method (`gpu/benchmark.py`)

- Warmup iterations before timing. The first CUDA calls pay for context creation and kernel selection.
- `torch.cuda.synchronize()` before every timer read. CUDA kernel launches return immediately, so without this the timer measures the launch, not the work.
- Two numbers per batch size:
  - **compute**: inputs already on the device,
  - **end-to-end**: includes the host→device copy of inputs (from pinned memory) and the device→host copy of predictions.
- Optional FP16 run on CUDA (`--fp16`).
- `--require-cuda` exits with an error rather than producing CPU numbers labelled as GPU. The JSON output records the GPU name, CUDA runtime, cuDNN and torch versions.

### CPU results (Intel i9-9880H, torch 2.2.2, 16 threads)

| device | batch | compute p50 ms | compute texts/s | end-to-end p50 ms | end-to-end texts/s |
|---|---:|---:|---:|---:|---:|
| cpu | 1 | 0.148 | 6,755 | 0.170 | 5,879 |
| cpu | 32 | 0.384 | 83,408 | 0.521 | 61,356 |
| cpu | 256 | 2.222 | 115,207 | 1.942 | 131,799 |
| cpu | 1024 | 8.139 | 125,821 | 7.969 | 128,493 |
| cpu | 4096 | 43.312 | 94,570 | 45.940 | 89,161 |

Raw data: [gpu/results/cpu-i9-9880H.json](../gpu/results/cpu-i9-9880H.json). This was a single run of 50 iterations per batch on a laptop, so treat small differences as noise. For example, end-to-end beating compute at batch 256 is noise, not a real effect.

### GPU results: Tesla T4, Google Colab

**Measured:** 2026-10-28, Tesla T4 (15.6 GB), CUDA runtime 13.0, cuDNN 9.27, torch 2.11.0+cu130, Python 3.13. Settings: 20 warmup + 100 timed iterations per batch size. The model was trained on the GPU for 8 epochs in 2.9 s, reaching held-out **accuracy 0.931, macro F1 0.931**. In the same session the unit tests gave `6 passed, 1 skipped`. `test_cuda_matches_cpu_predictions` passed, so GPU outputs match CPU outputs within 1e-4 for the same weights. The skipped test checks the *no-CUDA* error path and only applies without a GPU. Raw data: [gpu/results/colab-t4.json](../gpu/results/colab-t4.json).

End-to-end p50 latency and throughput (including host↔device copies), all measured in the same Colab session:

| batch | Colab CPU (1 thread) | T4 FP32 | T4 FP16 | FP32 speedup | FP16 speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.478 ms · 2,090/s | 0.344 ms · 2,903/s | 0.301 ms · 3,320/s | 1.4× | 1.6× |
| 32 | 1.878 ms · 17,038/s | 0.324 ms · 98,654/s | 0.338 ms · 94,634/s | 5.8× | 5.6× |
| 256 | 8.007 ms · 31,971/s | 0.556 ms · 459,982/s | 0.321 ms · 797,267/s | 14.4× | 24.9× |
| 1024 | 31.709 ms · 32,294/s | 1.063 ms · 963,280/s | 0.344 ms · 2,980,539/s | 29.8× | 92.3× |
| 4096 | 129.774 ms · 31,562/s | 3.312 ms · 1,236,802/s | 1.072 ms · 3,820,245/s | 39.2× | 121.0× |

**How to read this:**

- **The Colab CPU baseline is weak.** The VM has 2 vCPUs and PyTorch used **1 thread**, so the 39–121× speedups overstate what a GPU buys over a real server CPU. Compared with the 16-thread laptop CPU above (a different machine and torch version, so only a rough comparison):
  - At **batch 1**, the laptop CPU is **about 2× faster** than the T4 (0.170 ms against 0.344 ms). Copying the data and launching the GPU work cost more than the tiny model's compute.
  - At **batch 4096**, the T4 is about **14× faster** in FP32 (45.9 ms against 3.3 ms) and **about 43× faster** in FP16.
- **GPU latency barely moves as batches grow**, up to batch 32 in FP32 and up to batch 1024 in FP16. Until then the GPU is mostly idle and dominated by fixed overhead; it is not yet limited by compute. So throughput grows almost linearly with batch size.
- **FP16** helps from batch 256 upward, where the T4's tensor cores take over. Below that it makes no difference, because overhead dominates.
- **Host↔device copies** add about 0.08–0.16 ms. That matters at small batches (batch 1: 0.18 → 0.34 ms in FP32) but is noise at large ones.

**What this means for a real service:** sending one request per GPU call wastes the GPU. A GPU inference server should use **dynamic batching**, collecting requests for a few milliseconds and running them as one batch. That trades a little latency for one to two orders of magnitude more throughput. For a small model at low traffic, a multi-core CPU is simpler and has lower latency.

**Limitations:**
- Each configuration was a single run in a shared cloud VM, with no repeated trials or confidence intervals.
- The CPU and GPU comparisons on different machines also mix torch versions (2.2.2 against 2.11).
- FP16 predictions were timed but not checked for accuracy against FP32.
- Only the model call is timed. The HTTP serving path was not run on the GPU.

**To reproduce:** open [gpu/colab_benchmark.ipynb](../gpu/colab_benchmark.ipynb) in Google Colab (*File → Upload notebook*), choose *Runtime → Change runtime type → T4 GPU*, then *Run all*. If the repository is not cloneable, upload `dist/gpu-bench.zip` (`make gpu-zip`) when the notebook asks.

## Kubernetes GPU Job (`deployments/k8s/gpu/benchmark-job.yaml`)

- `resources.limits: nvidia.com/gpu: 1`. GPUs are an *extended resource*: whole units only, and requests must equal limits.
- A toleration for the common `nvidia.com/gpu:NoSchedule` taint on GPU nodes.
- The image is built from the CUDA wheel index (`--build-arg TORCH_INDEX=.../cu124`). The wheel bundles the CUDA runtime, so the node only needs the driver.
- `--require-cuda`, so a misconfigured node fails the Job instead of producing misleading numbers.

### Mocked-GPU scheduling demo on kind (real output)

`deployments/k8s/gpu/mock-gpu-node.sh` patches the node status to *advertise* `nvidia.com/gpu: 1`, using the Kubernetes [extended-resource](https://kubernetes.io/docs/tasks/administer-cluster/extended-resource-node/) mechanism. It adds **no device**. The `gpu-mock` overlay runs the same Job with the CPU image (`make gpu-mock-demo`).

**1. No GPU advertised: the scheduler holds the pod.**

```text
$ kubectl apply -k deployments/k8s/gpu-mock
$ kubectl -n ai-inference get pods -l app=gpu-benchmark
NAME                  READY   STATUS    RESTARTS   AGE
gpu-benchmark-5j5ng   0/1     Pending   0          8s
$ kubectl -n ai-inference describe pod -l app=gpu-benchmark
  Warning  FailedScheduling  default-scheduler  0/1 nodes are available: 1 Insufficient nvidia.com/gpu.
```

**2. A fake GPU is advertised: the pod is scheduled and started.**

```text
$ deployments/k8s/gpu/mock-gpu-node.sh add
node/ai-inference-control-plane patched
  Normal   Scheduled  default-scheduler  Successfully assigned ai-inference/gpu-benchmark-5j5ng to ai-inference-control-plane
  Normal   Started    kubelet            Container started
```

**3. The container has no CUDA, so the benchmark refuses to run.**

```text
$ kubectl -n ai-inference logs job/gpu-benchmark
CUDA is not available (torch.cuda.is_available() is False); refusing to run with --require-cuda
```

**Takeaway:** the scheduler only does accounting. It trusts whatever capacity the node advertises. Making the GPU actually usable inside the container is the device plugin's job. That is why the workload checks for CUDA itself instead of trusting that it was scheduled onto a "GPU node".
