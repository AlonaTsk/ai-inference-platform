"""CPU vs NVIDIA GPU inference benchmark for the PyTorch ticket classifier.

Correctness rules this script follows:
  * warmup iterations before timing (CUDA context, cuDNN/cuBLAS kernel selection),
  * torch.cuda.synchronize() before every timer read, because CUDA calls are async,
  * two numbers per batch: "compute" (inputs already on the device) and
    "end-to-end" (includes host->device copy of inputs and device->host copy of results),
  * the device that actually ran is recorded; no GPU row is printed without CUDA.

Usage:
  python gpu/benchmark.py                      # CPU, plus CUDA if available
  python gpu/benchmark.py --require-cuda       # fail instead of skipping the GPU
  python gpu/benchmark.py --json results.json  # also write machine-readable results
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import torch

from textclf import TicketClassifier, Vocab, generate, pick_device, train


@dataclass(frozen=True)
class RunConfig:
    warmup: int
    iters: int
    half: bool = False  # FP16 weights (CUDA only)


def p95(samples: list[float]) -> float:
    return statistics.quantiles(samples, n=20)[18]


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def time_batches(model: TicketClassifier, vocab: Vocab, texts: list[str], device: torch.device, cfg: RunConfig) -> dict:
    batch = len(texts)
    model = model.to(device).eval()
    if cfg.half:
        model = model.half()
    ids_cpu, offsets_cpu = vocab.encode_batch(texts)
    if device.type == "cuda":
        # Pinned host memory makes the H2D copy async-capable and faster.
        ids_cpu, offsets_cpu = ids_cpu.pin_memory(), offsets_cpu.pin_memory()
    ids_dev, offsets_dev = ids_cpu.to(device), offsets_cpu.to(device)

    def compute():
        return model(ids_dev, offsets_dev)

    def end_to_end():
        out = model(ids_cpu.to(device, non_blocking=True), offsets_cpu.to(device, non_blocking=True))
        return out.argmax(1).cpu()

    results = {}
    for name, fn in (("compute", compute), ("end_to_end", end_to_end)):
        for _ in range(cfg.warmup):
            fn()
        sync(device)
        times = []
        for _ in range(cfg.iters):
            sync(device)
            t0 = time.perf_counter()
            fn()
            sync(device)
            times.append((time.perf_counter() - t0) * 1000)
        p50 = statistics.median(times)
        results[name] = {
            "p50_ms": round(p50, 4),
            "p95_ms": round(p95(times), 4),
            "texts_per_s": round(batch / (p50 / 1000)),
        }
    return results


def environment(devices: list[torch.device]) -> dict:
    env = {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": platform.processor() or platform.machine(),
        "cpu_threads_used": torch.get_num_threads(),
        "logical_cpus": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
    }
    if any(d.type == "cuda" for d in devices):
        props = torch.cuda.get_device_properties(0)
        env.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "gpu_memory_gb": round(props.total_memory / 1e9, 1),
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
            }
        )
    return env


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batches", default="1,32,256,1024,4096")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--fp16", action="store_true", help="also time half precision on CUDA")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--json", help="write results to this file")
    args = p.parse_args()

    if args.require_cuda and not torch.cuda.is_available():
        sys.exit("CUDA is not available (torch.cuda.is_available() is False); refusing to run with --require-cuda")

    train_device = pick_device("auto")
    t0 = time.perf_counter()
    model, vocab, metrics = train(train_device, epochs=args.epochs)
    train_s = time.perf_counter() - t0
    print(
        f"trained on {train_device} in {train_s:.1f}s; held-out accuracy {metrics['accuracy']}, "
        f"macro F1 {metrics['macro_f1']} (n={metrics['n_test']})"
    )

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    else:
        print("CUDA not available: GPU rows skipped (this is a CPU-only run)")

    batches = [int(b) for b in args.batches.split(",")]
    texts, _ = generate(n=max(batches), seed=999)
    env = environment(devices)
    print(json.dumps(env, indent=2))

    rows = []
    configs = [(d, False) for d in devices]
    if args.fp16 and torch.cuda.is_available():
        configs.append((torch.device("cuda"), True))
    for device, half in configs:
        cfg = RunConfig(warmup=args.warmup, iters=args.iters, half=half)
        for bs in batches:
            r = time_batches(model, vocab, texts[:bs], device, cfg)
            rows.append({"device": device.type + ("-fp16" if half else ""), "batch": bs, **r})

    print("\n| device | batch | compute p50 ms | compute texts/s | end-to-end p50 ms | end-to-end texts/s |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        c, e = r["compute"], r["end_to_end"]
        print(
            f"| {r['device']} | {r['batch']} | {c['p50_ms']:.3f} | {c['texts_per_s']:,} | "
            f"{e['p50_ms']:.3f} | {e['texts_per_s']:,} |"
        )

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(
                {
                    "environment": env,
                    "train": {"device": str(train_device), "seconds": round(train_s, 1), **metrics},
                    "settings": vars(args),
                    "results": rows,
                },
                f,
                indent=2,
            )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
