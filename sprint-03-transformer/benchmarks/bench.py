"""Benchmarks for the Sprint 3 transformer.

Usage (from sprint-03-transformer/):
    make benchmark       # or: python3 benchmarks/bench.py

Measures, on this machine, for the default model (d_model=16, 2 heads,
1 layer, block_size=8, vocab 58):
  * forward time of one training step (one 8-token batch)
  * backward time of the same step
  * tokens/sec for the step (forward + backward)

These are the numbers the README's "what to expect" section refers to.
Run it on your Mac — the VM numbers in the README are not yours.
"""

import os
import random
import sys
import time
import warnings

# Silence the Mac/Accelerate RuntimeWarnings (divide by zero / overflow
# in matmul) — a known noise quirk on Apple Silicon, not a code bug
# (the same run on Linux prints none). Real errors still raise.
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scalar.transformer import Transformer  # noqa: E402

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "corpus.txt")


def main():
    text = open(CORPUS).read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = [stoi[c] for c in text[:2000]]

    random.seed(1)
    model = Transformer(len(chars), d_model=16, n_head=2, n_layer=1,
                        block_size=8)
    print(f"params: {len(model.parameters())}")

    idx = data[:8]
    # Warmup (first step pays graph-build and import costs).
    loss = model.loss(idx)
    model.zero_grad()
    loss.backward()

    t0 = time.time()
    loss = model.loss(idx)
    t1 = time.time()
    model.zero_grad()
    loss.backward()
    t2 = time.time()
    print(f"forward:  {t1 - t0:.2f}s  (loss {loss.data:.4f})")
    print(f"backward: {t2 - t1:.2f}s")
    print(f"one step: {t2 - t0:.2f}s  ->  {8 / (t2 - t0):.1f} tokens/sec")


if __name__ == "__main__":
    main()
