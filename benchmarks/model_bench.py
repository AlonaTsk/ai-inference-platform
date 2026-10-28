"""In-process CPU benchmark of the trained classifier (no HTTP).

Measures latency and throughput of model.predict_proba for several batch
sizes, after warmup, and prints a Markdown table plus the hardware used.

Usage (from repo root, with inference/ on the import path):
  PYTHONPATH=inference python benchmarks/model_bench.py --model inference/model/model.joblib
"""

import argparse
import os
import platform
import statistics
import time

import joblib

from app.data import generate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="inference/model/model.joblib")
    p.add_argument("--batches", default="1,8,32,64,256")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=200)
    args = p.parse_args()

    model = joblib.load(args.model)
    # Different seed from training so the texts are fresh samples.
    texts, _ = generate(n=2048, seed=999)

    cpu = platform.processor() or platform.platform()
    print(f"python {platform.python_version()} | {platform.machine()} | {cpu} | {os.cpu_count()} logical CPUs\n")
    print("| batch | p50 ms | p95 ms | throughput (texts/s) |")
    print("|------:|-------:|-------:|---------------------:|")
    for bs in (int(b) for b in args.batches.split(",")):
        batch = texts[:bs]
        for _ in range(args.warmup):
            model.predict_proba(batch)
        times = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            model.predict_proba(batch)
            times.append((time.perf_counter() - t0) * 1000)
        p50 = statistics.median(times)
        p95 = statistics.quantiles(times, n=20)[18]
        print(f"| {bs} | {p50:.3f} | {p95:.3f} | {bs / (p50 / 1000):,.0f} |")


if __name__ == "__main__":
    main()
