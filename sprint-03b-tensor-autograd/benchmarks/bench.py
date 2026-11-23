"""Forward + backward time for one attention head and one MLP layer, at
growing sizes. Run after the tests pass:  make benchmark

Compare the numbers with Sprint 3's src/numpy_v2 engine (the scaffolded
one) on the same shapes: same math, and now both engines are yours.
"""

import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tensor import Tensor, cross_entropy  # noqa: E402


def attention_step(B, T, d):
    rng = np.random.default_rng(0)
    x = Tensor(rng.standard_normal((B, T, d)))
    wq, wk, wv = (Tensor(rng.standard_normal((d, d)) * d**-0.5) for _ in range(3))
    mask = np.triu(np.full((T, T), -1e9), k=1)
    q, k, v = x @ wq, x @ wk, x @ wv
    att = ((q @ k.swap_last()) * (1 / np.sqrt(d)) + mask).softmax(axis=-1)
    loss = ((att @ v) ** 2).mean()
    loss.backward()


def mlp_step(N, d, C):
    rng = np.random.default_rng(0)
    x = Tensor(rng.standard_normal((N, d)))
    w1 = Tensor(rng.standard_normal((d, 4 * d)) * d**-0.5)
    w2 = Tensor(rng.standard_normal((4 * d, C)) * (4 * d) ** -0.5)
    loss = cross_entropy((x @ w1).relu() @ w2, rng.integers(0, C, N))
    loss.backward()


def timeit(fn, *args, reps=5):
    fn(*args)
    t = time.perf_counter()
    for _ in range(reps):
        fn(*args)
    return (time.perf_counter() - t) / reps * 1000


if __name__ == "__main__":
    print(f"{'op':<12} {'shape':<24} {'fwd+bwd ms':>11}")
    for B, T, d in [(16, 64, 64), (16, 64, 256), (16, 128, 512)]:
        print(f"{'attention':<12} {f'B{B} T{T} d{d}':<24} {timeit(attention_step, B, T, d):>11.2f}")
    for N, d, C in [(1024, 64, 512), (1024, 256, 512), (1024, 512, 4096)]:
        print(f"{'mlp':<12} {f'N{N} d{d} C{C}':<24} {timeit(mlp_step, N, d, C):>11.2f}")
