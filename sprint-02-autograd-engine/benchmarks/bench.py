"""Benchmark harness — run AFTER tests pass: python3 benchmarks/bench.py

Two measurements, both about scaling — both get quoted in your post:
1. Graph-size sweep: build a chain of N multiply-adds, time forward (graph
   construction) and backward() separately. Expect roughly linear scaling.
2. MLP micro-benchmark: forward+backward of MLP(2,[16,16,1]) on one input,
   timed per pass.

Save this output. The post quotes the v1 numbers; Sunday's v2 compares.

Note: the sweep builds graphs deeper than Python's recursion limit. If this
crashes with RecursionError, your topological sort recurses — rewrite it
iteratively (see README discussion). That discovery IS a benchmark result;
write it up in the post.
"""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine import Value
from src.nn import MLP


def bench_chain(n):
    x = Value(2.0)
    w = Value(1.001)
    b = Value(0.001)
    t0 = time.perf_counter()
    y = x
    for _ in range(n):
        y = y * w + b
    t_forward = time.perf_counter() - t0
    t0 = time.perf_counter()
    y.backward()
    t_backward = time.perf_counter() - t0
    return t_forward, t_backward, x.grad  # dy/dx; y.grad is 1.0 by construction


def bench_mlp(iters=30):
    random.seed(0)
    mlp = MLP(2, [16, 16, 1])
    x = [0.5, -0.3]
    # warmup (also shakes out first-call overhead)
    out = mlp(x)
    mlp.zero_grad()
    out.backward()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = mlp(x)
        mlp.zero_grad()
        out.backward()
    return (time.perf_counter() - t0) / iters


print("== graph-size sweep: y = x; repeat N: y = y * w + b ==")
print(f"{'N':>8} {'forward (s)':>12} {'backward (s)':>13} {'dy/dx':>12}")
for n in [100, 1000, 5000, 20000]:
    tf, tb, g = bench_chain(n)
    print(f"{n:>8} {tf:>12.3f} {tb:>13.3f} {g:>12.3g}")

print()
print("== MLP(2,[16,16,1]) micro-benchmark: one forward+backward pass ==")
per_pass = bench_mlp()
print(f"{per_pass * 1000:.2f} ms per pass")
