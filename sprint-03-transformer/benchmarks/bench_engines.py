"""Compare the scalar and NumPy engines on the same model and data.

Usage (from sprint-03-transformer/):
    python3 benchmarks/bench_engines.py

Runs the same config on both engines — d_model=16, 1 layer, block 8,
vocab 58 (data/corpus.txt), 10 training steps from the same seed — and
reports time per step, final loss, and the speedup. The two engines
should reach (nearly) the same loss: they are the same math, just a
different representation. If the losses differ, one engine has a bug.

This is the comparison point for the Sprint 3 post (the ~4000x number
came from a bigger config; this small one is the reproducible check).
"""

import os
import random
import sys
import time

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.numpy.transformer import Transformer as NumpyTransformer  # noqa: E402
from src.scalar.transformer import Transformer as ScalarTransformer  # noqa: E402

CORPUS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "corpus.txt",
)
STEPS = 10
LR = 0.05


def run(engine_name, Transformer, data, vocab_size):
    random.seed(1)
    model = Transformer(
        vocab_size, d_model=16, n_head=2, n_layer=1, block_size=8
    )
    t0 = time.time()
    losses = []
    for step in range(STEPS):
        start = random.randint(0, len(data) - 8 - 1)
        idx = data[start : start + 8]
        loss = model.loss(idx)
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= LR * p.grad
        losses.append(float(loss.data))
    dt = time.time() - t0
    return dt / STEPS, losses[-1], losses[0]


def main():
    text = open(CORPUS).read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = [stoi[c] for c in text]

    print(f"config: d16, 1 layer, block 8, vocab {len(chars)}, {STEPS} steps, lr {LR}")
    print(f"{'engine':<8} {'s/step':>8} {'loss 0':>8} {'loss N':>8} {'tokens/s':>9}")
    results = {}
    for name, Transformer in (
        ("scalar", ScalarTransformer),
        ("numpy", NumpyTransformer),
    ):
        sps, last, first = run(name, Transformer, data, len(chars))
        results[name] = sps
        print(f"{name:<8} {sps:8.3f} {first:8.4f} {last:8.4f} {8 / sps:9.1f}")
    print(f"\nspeedup (numpy vs scalar): {results['scalar'] / results['numpy']:.0f}x")


if __name__ == "__main__":
    main()
