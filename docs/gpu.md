# GPU: PyTorch model, Colab benchmark, Kubernetes GPU scheduling

There are three separate pieces here, which are easy to confuse:

| Piece | Where it runs | What it proves | Status |
|---|---|---|---|
| PyTorch classifier + CUDA-aware benchmark (`gpu/`) | Any machine | The code places the model and data on a device correctly and times it correctly | Done; tested on CPU |
| Real GPU numbers | Google Colab (T4) | Measured CPU-versus-GPU latency and throughput | **Pending: run [colab_benchmark.ipynb](../gpu/colab_benchmark.ipynb)** |
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

**Expected pattern** (to be confirmed by the Colab run): at batch 1 the GPU is likely *slower* end to end, because the transfer and launch overhead outweigh a tiny amount of compute. The GPU should win at large batches. For a real service, that means GPUs pay off only with request batching.

### CPU results (Intel i9-9880H, torch 2.2.2, 16 threads)

| device | batch | compute p50 ms | compute texts/s | end-to-end p50 ms | end-to-end texts/s |
|---|---:|---:|---:|---:|---:|
| cpu | 1 | 0.148 | 6,755 | 0.170 | 5,879 |
| cpu | 32 | 0.384 | 83,408 | 0.521 | 61,356 |
| cpu | 256 | 2.222 | 115,207 | 1.942 | 131,799 |
| cpu | 1024 | 8.139 | 125,821 | 7.969 | 128,493 |
| cpu | 4096 | 43.312 | 94,570 | 45.940 | 89,161 |

Raw data: [gpu/results/cpu-i9-9880H.json](../gpu/results/cpu-i9-9880H.json). This was a single run of 50 iterations per batch on a laptop, so treat small differences as noise. For example, end-to-end beating compute at batch 256 is noise, not a real effect.

### GPU results

**Not measured yet.** To produce them:
1. Open [gpu/colab_benchmark.ipynb](../gpu/colab_benchmark.ipynb) in Google Colab (*File → Upload notebook*).
2. *Runtime → Change runtime type → T4 GPU*, then *Run all*.
3. If the repository is not public, the notebook asks for `dist/gpu-bench.zip` (build it with `make gpu-zip`).
4. Commit the downloaded `colab-gpu.json` to `gpu/results/`.

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
