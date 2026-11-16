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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformer import Transformer  # noqa: E402

CORPUS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "corpus.txt"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.05)
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
        "--save", type=str, default=None, help="path to save the trained weights (JSON)"
    )

    args = parser.parse_args()

    text = open(CORPUS).read()
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
    print(
        f"vocab {vocab_size}, params {len(model.parameters())}, "
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
        for p in model.parameters():
            p.data -= args.lr * p.grad
        if step % 10 == 0 or step == args.steps - 1:
            print(
                f"step {step:4d}  loss {loss.data:.4f}  " f"({time.time() - t0:.0f}s)"
            )

    if args.save:
        with open(args.save, "w") as f:
            json.dump([p.data for p in model.parameters()], f)
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
