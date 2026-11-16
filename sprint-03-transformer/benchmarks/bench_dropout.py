"""Compare training with and without dropout, on train and held-out loss.

Usage (from sprint-03-transformer/):
    python3 benchmarks/bench_dropout.py [--steps N] [--dropout P]

Trains the same NumPy model twice from the same seed — once with
dropout=0 (the default) and once with dropout=P (default 0.1) — on the
first 90% of data/corpus.txt, and reports the average training loss
and the average loss on the last 10% (held out, dropout off) for each.
If dropout helps generalization, the dropout run should have a lower
validation loss even if its training loss is higher (dropout makes
training harder — some neurons are off — so train loss is not the
measure; validation loss is).

If dropout() is not implemented yet (it is a stub in
src/numpy/engine.py), the dropout run raises NotImplementedError and
this script prints that instead of a comparison — implement dropout
first, then re-run.
"""

import argparse
import os
import random
import sys
import time

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src.numpy.transformer import Transformer  # noqa: E402

CORPUS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "corpus.txt",
)


def val_loss(model, data, block_size, batches=20, seed=99):
    """Average loss on held-out data, dropout off (training=False)."""
    rng = random.Random(seed)
    losses = []
    for _ in range(batches):
        start = rng.randint(0, len(data) - block_size - 1)
        idx = data[start : start + block_size]
        logits = model(idx, training=False)
        from src.numpy.transformer import softmax

        probs = softmax(logits[:-1], axis=1)
        rows = np.arange(len(idx) - 1)
        targets = np.array(idx[1:], dtype=np.int64)
        target_probs = probs[(rows, targets)]
        losses.append(float(-target_probs.log().mean().data))
    return sum(losses) / len(losses)


def run(dropout, train_data, val_data, vocab_size, steps, lr, seed):
    random.seed(seed)
    model = Transformer(
        vocab_size, d_model=16, n_head=2, n_layer=1, block_size=8,
        dropout=dropout,
    )
    train_losses = []
    t0 = time.time()
    for step in range(steps):
        start = random.randint(0, len(train_data) - 8 - 1)
        idx = train_data[start : start + 8]
        loss = model.loss(idx)
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= lr * p.grad
        train_losses.append(float(loss.data))
    dt = time.time() - t0
    return (
        sum(train_losses) / len(train_losses),
        val_loss(model, val_data, 8),
        dt / steps,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    text = open(CORPUS).read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = [stoi[c] for c in text]
    split = int(len(data) * 0.9)
    train_data, val_data = data[:split], data[split:]

    print(
        f"config: d16, 1 layer, block 8, vocab {len(chars)}, "
        f"{args.steps} steps, lr {args.lr}, train {len(train_data)} chars, "
        f"val {len(val_data)} chars"
    )
    print(f"{'dropout':>8} {'train loss':>11} {'val loss':>9} {'s/step':>7}")
    for p in (0.0, args.dropout):
        try:
            train_l, val_l, sps = run(
                p, train_data, val_data, len(chars), args.steps, args.lr, seed=1
            )
        except NotImplementedError as e:
            print(f"{p:>8.2f}  not implemented yet: {e}")
            continue
        print(f"{p:>8.2f} {train_l:11.4f} {val_l:9.4f} {sps:7.3f}")
    print("\nRead it as: dropout should lower val loss (generalization) even")
    print("if train loss is higher. If val loss is higher too, dropout is")
    print("not helping at this rate / scale.")


if __name__ == "__main__":
    main()
