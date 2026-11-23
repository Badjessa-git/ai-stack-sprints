"""NumPy nn: the same Module / linear-layer shapes as src/scalar/nn.py, but the
numbers live in arrays (Tensor) instead of one Value per scalar.

The weight init draws random.uniform(-1, 1) * 1/sqrt(n_in) in exactly the
same order as the scalar Layer — for each output neuron, the bias first, then its
weights — so a model built with the same seed has identical weights.
That is what makes the two engines directly comparable: same seed, same
weights, same outputs, same gradients.
"""

import random

import numpy as np

from src.numpy_v2.engine import Tensor


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)


class Linear(Module):
    """One linear layer: y = x @ W.T + b.

    W is (n_out, n_in), b is (n_out,). This replaces Layer(..., nonlin=
    False) from src/scalar/nn.py: the scalar Layer is a list of Neurons, each
    computing one output as a dot product; this is the same computation
    as one matrix multiply over all outputs at once.
    """

    def __init__(self, n_in, n_out):
        # U(-1, 1) scaled by 1/sqrt(n_in) (PyTorch's nn.Linear default).
        # Unscaled U(-1, 1) gives outputs with std sqrt(n_in / 3) — 18 at
        # d1024 — which saturates softmax/attention at init and made wide
        # models start at loss ~59 instead of ln(vocab). Same draw order as
        # the scalar Neuron, so the engines still produce identical weights.
        scale = 1.0 / np.sqrt(n_in)
        W = np.empty((n_out, n_in))
        b = np.empty(n_out)
        for j in range(n_out):
            b[j] = random.uniform(-1, 1) * scale
            for i in range(n_in):
                W[j, i] = random.uniform(-1, 1) * scale
        self.weight = Tensor(W)
        self.bias = Tensor(b)

    def __call__(self, x):
        """x: Tensor of shape (..., n_in). Returns Tensor (..., n_out)."""
        return x @ self.weight.transpose() + self.bias

    def parameters(self):
        return [self.weight, self.bias]
