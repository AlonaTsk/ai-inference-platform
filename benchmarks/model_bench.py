"""In-process CPU benchmark of the trained classifier (no HTTP).

Measures latency and throughput of model.predict_proba for several batch
sizes, after warmup, and prints a Markdown table plus the hardware used.

Usage (from repo root):
  python benchmarks/model_bench.py --model inference/model/model.joblib
"""

import argparse
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
from app.data import generate  # noqa: E402


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

    print(f"python {platform.python_version()} | {platform.machine()} | {platform.processor() or platform.platform()} | "
          f"{os.cpu_count()} logical CPUs | single process\n")
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
        times.sort()
        p50 = statistics.median(times)
        p95 = times[int(0.95 * len(times)) - 1]
        print(f"| {bs} | {p50:.3f} | {p95:.3f} | {bs / (p50 / 1000):,.0f} |")


if __name__ == "__main__":
    main()
