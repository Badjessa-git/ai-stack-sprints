"""NumPy nn: the same Module / linear-layer shapes as src/scalar/nn.py, but the
numbers live in arrays (Tensor) instead of one Value per scalar.

The weight init draws random.uniform(-1, 1) in exactly the same order as
the scalar Layer — for each output neuron, the bias first, then its
weights — so a model built with the same seed has identical weights.
That is what makes the two engines directly comparable: same seed, same
weights, same outputs, same gradients.
"""

import random

import numpy as np

from src.numpy.engine import Tensor


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
        W = np.empty((n_out, n_in))
        b = np.empty(n_out)
        for j in range(n_out):
            b[j] = random.uniform(-1, 1)
            for i in range(n_in):
                W[j, i] = random.uniform(-1, 1)
        self.weight = Tensor(W)
        self.bias = Tensor(b)

    def __call__(self, x):
        """x: Tensor of shape (..., n_in). Returns Tensor (..., n_out)."""
        return x @ self.weight.transpose() + self.bias

    def parameters(self):
        return [self.weight, self.bias]
