"""End-to-end HTTP benchmark through the Go API (stdlib only).

Sends N requests with C concurrent workers to POST /v1/predict after a
warmup, then reports latency percentiles, throughput and error count.

Usage: python benchmarks/http_bench.py --url http://localhost:8080 -n 2000 -c 8
"""

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

TEXTS = [
    "I was charged twice this month",
    "my package has not arrived yet",
    "the password reset email never arrives",
    "the app crashes when I open settings",
]


def call(url: str, i: int) -> tuple[float, bool]:
    body = json.dumps({"text": TEXTS[i % len(TEXTS)]}).encode()
    req = urllib.request.Request(url + "/v1/predict", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
            ok = resp.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        ok = False
    return (time.perf_counter() - t0) * 1000, ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8080")
    p.add_argument("-n", type=int, default=2000)
    p.add_argument("-c", type=int, default=8)
    p.add_argument("--warmup", type=int, default=100)
    args = p.parse_args()

    for i in range(args.warmup):
        call(args.url, i)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.c) as pool:
        results = list(pool.map(lambda i: call(args.url, i), range(args.n)))
    wall = time.perf_counter() - t0

    lat = [r[0] for r in results]
    errors = sum(1 for r in results if not r[1])
    pct = statistics.quantiles(lat, n=100)  # pct[k - 1] is the k-th percentile
    print("| concurrency | requests | errors | p50 ms | p95 ms | p99 ms | throughput (req/s) |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    print(
        f"| {args.c} | {args.n} | {errors} | {statistics.median(lat):.1f} | "
        f"{pct[94]:.1f} | {pct[98]:.1f} | {args.n / wall:,.0f} |"
    )


if __name__ == "__main__":
    main()
