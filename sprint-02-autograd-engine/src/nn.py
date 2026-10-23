"""Tiny neural-net library on top of the scalar engine — SCAFFOLD.

A Neuron is n_in weights + 1 bias feeding tanh. A Layer is a row of neurons.
An MLP is a list of layers. All parameters are engine.Value objects, so a
forward pass builds a graph and loss.backward() differentiates it.

Spec (tests rely on this):
- Weights and bias are drawn from random.uniform(-1, 1) at construction.
- Neuron(n_in, nonlin=True): nonlin=True -> tanh activation, False -> linear.
- Layer(n_in, n_out, **kwargs) forwards kwargs to each Neuron.
- __call__ returns a single Value when the output size is 1, else a list.
- Module.parameters() returns every parameter Value in the module.
- Module.zero_grad() sets every parameter's grad to 0.0.
"""

import random
from src.engine import Value


class Module:
    def parameters(self):
        raise NotImplementedError("your turn: collect every parameter Value")

    def zero_grad(self):
        raise NotImplementedError("your turn: grad = 0.0 on every parameter")


class Neuron(Module):
    def __init__(self, n_in, nonlin=True):
        raise NotImplementedError("your turn: weights + bias ~ uniform(-1, 1)")

    def __call__(self, x):
        """x: list of numbers or Values, length n_in. Returns a Value."""
        raise NotImplementedError("your turn: w·x + b, then tanh if nonlin")

    def parameters(self):
        raise NotImplementedError("your turn")


class Layer(Module):
    def __init__(self, n_in, n_out, **kwargs):
        raise NotImplementedError("your turn: n_out neurons")

    def __call__(self, x):
        """Returns a Value when n_out == 1, else a list of Values."""
        raise NotImplementedError("your turn")

    def parameters(self):
        raise NotImplementedError("your turn")


class MLP(Module):
    def __init__(self, n_in, n_outs):
        """n_outs: e.g. [16, 16, 1] -> Layer(n_in,16), Layer(16,16), Layer(16,1)."""
        raise NotImplementedError("your turn")

    def __call__(self, x):
        raise NotImplementedError("your turn: thread x through the layers")

    def parameters(self):
        raise NotImplementedError("your turn")
