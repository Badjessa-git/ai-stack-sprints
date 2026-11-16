"""Train the tiny character transformer on Tiny Shakespeare.

Usage (from sprint-03-transformer/):
    make train                 # 100 steps, d_model=16, prints a sample
    make train ARGS="--steps 300 --lr 0.05"
    python3 benchmarks/train.py --steps 20 --d-model 32 --block-size 16

The loop is the standard one, written out so the shape is visible:
    predict -> loss -> zero_grad -> backward -> SGD update -> repeat

Defaults are sized for this machine (see benchmarks/bench.py): one step
at d_model=16 takes ~2s on the VM and ~1s on a Mac. d_model=32 is the
scale-up stretch — about 5x slower per step. If you raise the steps,
expect the loss to be noisy: each step trains on a different random
batch, so watch the trend over tens of steps, not step to step.
"""

import argparse
import json
import os
import random
import sys
import time

# Keep the BLAS math library to one thread. The matrices here are tiny,
# so threads buy nothing — but on Apple Silicon, NumPy's Accelerate
# backend can segfault when it spreads small matrix ops across threads
# (seen Nov 16, 2026: the d128 run died with a segfault on an M4 Pro).
# These must be set before NumPy is imported, so they live here at the
# top. setdefault respects a value you already set in the shell.
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CORPUS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "corpus.txt"
)


def _save_weights(model, path):
    """Save parameters as JSON. A NumPy array becomes a nested list; a
    plain number stays a number. generate.py loads it back with the
    matching --engine."""
    data = [
        p.data.tolist() if hasattr(p.data, "tolist") else p.data
        for p in model.parameters()
    ]
    with open(path, "w") as f:
        json.dump(data, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "adam"],
        default="sgd",
        help="sgd: plain gradient descent (what we've used so far). "
        "adam: momentum + per-weight adaptive step size — converges "
        "faster and lower for transformers.",
    )
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-layer", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--sample",
        type=int,
        default=40,
        help="chars to generate at the end (each one costs "
        "a full forward pass — no KV cache yet, "
        "that is Sprint 6)",
    )
    parser.add_argument(
        "--engine",
        choices=["scalar", "numpy"],
        default="scalar",
        help="autograd engine: scalar (one Value per number) or numpy "
        "(arrays — same model, same math, much faster)",
    )
    parser.add_argument(
        "--save", type=str, default=None, help="path to save the trained weights (JSON)"
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="path to weights JSON to start from (resume training). "
        "Architecture flags must match the run that saved them.",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=CORPUS,
        help="training text file (default: data/corpus.txt)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="save weights every N steps (0 = only at the end, "
        "via --save)",
    )

    args = parser.parse_args()

    if args.engine == "numpy":
        from src.numpy.transformer import Transformer
    else:
        from src.scalar.transformer import Transformer

    text = open(args.corpus).read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = [stoi[c] for c in text]

    random.seed(args.seed)
    model = Transformer(
        vocab_size,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        block_size=args.block_size,
    )
    if args.load:
        import numpy as np

        weights = json.load(open(args.load))
        assert len(weights) == len(model.parameters()), (
            f"weights file has {len(weights)} tensors, model has "
            f"{len(model.parameters())} — architecture flags must match "
            "the run that saved them"
        )
        for p, w in zip(model.parameters(), weights):
            arr = np.array(w) if args.engine == "numpy" else w
            assert np.shape(arr) == np.shape(p.data), (
                f"weight shape {np.shape(arr)} != model shape "
                f"{np.shape(p.data)} — architecture flags must match "
                "the run that saved them"
            )
            p.data = arr
        print(f"resumed from {args.load} ({len(weights)} weights)")
    # Adam keeps two running averages per parameter: m (momentum — the
    # smoothed gradient) and v (squared gradients — how big they've
    # been). Same shape as the parameter, all zeros to start.
    adam_state = None
    if args.optimizer == "adam":
        adam_state = []
        for p in model.parameters():
            if hasattr(p.data, "tolist"):  # numpy array
                import numpy as np

                adam_state.append(
                    [np.zeros_like(p.data), np.zeros_like(p.data)]
                )
            else:  # scalar float
                adam_state.append([0.0, 0.0])
    print(
        f"engine {args.engine}, vocab {vocab_size}, params {len(model.parameters())}, "
        f"d_model {args.d_model}, block {args.block_size}, "
        f"steps {args.steps}, lr {args.lr}"
    )

    t0 = time.time()
    for step in range(args.steps):
        start = random.randint(0, len(data) - args.block_size - 1)
        idx = data[start : start + args.block_size]
        loss = model.loss(idx)
        model.zero_grad()
        loss.backward()
        if args.optimizer == "adam":
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            t = step + 1
            for state, p in zip(adam_state, model.parameters()):
                m, v = state
                g = p.grad
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * (g ** 2)
                state[0], state[1] = m, v
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p.data -= args.lr * m_hat / (v_hat ** 0.5 + eps)
        else:
            for p in model.parameters():
                p.data -= args.lr * p.grad
        if (
            args.save
            and args.checkpoint_every
            and (step + 1) % args.checkpoint_every == 0
        ):
            _save_weights(model, args.save)
            print(f"checkpoint step {step + 1} -> {args.save}")
        if step % 10 == 0 or step == args.steps - 1:
            print(
                f"step {step:4d}  loss {loss.data:.4f}  " f"({time.time() - t0:.0f}s)"
            )

    if args.save:
        _save_weights(model, args.save)
        print(f"saved {len(model.parameters())} weights to {args.save}")

    print(
        f"\n{time.time() - t0:.0f}s total, "
        f"{(time.time() - t0) / args.steps:.2f}s/step"
    )
    print(f"\n--- sample (greedy, {args.sample} chars) ---")
    start = random.randint(0, len(data) - args.block_size - 1)
    gen = model.generate(
        data[start : start + args.block_size], args.sample, greedy=True
    )
    print("".join(itos[i] for i in gen))


if __name__ == "__main__":
    main()
