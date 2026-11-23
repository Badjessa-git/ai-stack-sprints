"""Static vs continuous batching across load levels (Part A).

    python3 benchmarks/sim_report.py
    python3 benchmarks/sim_report.py --max-batch 16 --rates 2 5 10 20 40

Simulated clock, so it runs in seconds. Rate = requests per second
arriving. Watch three things as load rises: throughput (tokens/sec),
p50 latency, and p99 time to first token.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sim import ContinuousBatcher, StaticBatcher, simulate, summarize, workload  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--rates", type=float, nargs="+", default=[0.5, 1, 2, 4])
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()
    print(f"max_batch {args.max_batch}, {args.n} requests, prompts 20-200, outputs 10-300 tokens\n")
    print(f"{'req/s':>5}  {'policy':<13} {'tokens/sec':>10} {'latency p50':>12} {'TTFT p99':>9}")
    for rate in args.rates:
        for name, sched in [("one-at-a-time", ContinuousBatcher(1)),
                            ("static", StaticBatcher(args.max_batch)),
                            ("continuous", ContinuousBatcher(args.max_batch))]:
            s = summarize(simulate(workload(args.n, rate, seed=0), sched))
            print(f"{rate:>5g}  {name:<13} {s['throughput']:>10.0f} {s['latency_p50']:>11.1f}s {s['ttft_p99']:>8.1f}s")
        print()
