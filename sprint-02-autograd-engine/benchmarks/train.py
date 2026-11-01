"""Training harness — run AFTER tests pass: python3 benchmarks/train.py

Trains MLP(2, [16, 16, 1]) on data/moons.csv (200 points, labels -1/+1) with
MSE loss and plain SGD. Prints loss every 10 steps, final train accuracy,
and wall-clock time.

These are your v1 numbers for the post. Sunday's work compares against them:
did the loss actually fall, how long did 50 steps take, and was the
bottleneck the forward pass or backward()?
"""

import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine import Value
from src.nn import MLP


def load_data():
    path = Path(__file__).resolve().parent.parent / "data" / "moons.csv"
    with open(path) as f:
        return [([float(x1), float(x2)], float(y)) for x1, x2, y in csv.reader(f)]


random.seed(0)
data = load_data()
mlp = MLP(2, [16, 16, 1])
lr = 0.1
steps = 50


def loss_and_acc():
    total = Value(0.0)
    correct = 0
    for x, y in data:
        out = mlp(x)
        total = total + (out - y) ** 2
        if (out.data > 0) == (y > 0):
            correct += 1
    return total * (1.0 / len(data)), correct / len(data)


t0 = time.perf_counter()
for step in range(steps):
    loss, acc = loss_and_acc()
    mlp.zero_grad()
    loss.backward()
    for p in mlp.parameters():
        p.data -= lr * p.grad
    if step % 10 == 0 or step == steps - 1:
        print(f"step {step:3d}  loss {loss.data:.4f}  acc {acc:.3f}", flush=True)
print(f"done in {time.perf_counter() - t0:.1f}s")
