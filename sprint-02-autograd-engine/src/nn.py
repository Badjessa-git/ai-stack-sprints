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
        return []

    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0


class Neuron(Module):
    def __init__(self, n_in, nonlin=True):
        self.flag = nonlin
        self.bias = Value(random.uniform(-1, 1))
        self.weights = [Value(random.uniform(-1, 1))] * n_in
        self.n_in = n_in

    def __call__(self, x):
        """x: list of numbers or Values, length n_in. Returns a Value."""
        value = 0
        for i in range(self.n_in):
            value += x[i] * self.weights[i]

        value += self.bias

        if self.flag:
            value.tanh()

        return value

    def parameters(self):
        return self.weights + [self.bias]


class Layer(Module):
    def __init__(self, n_in, n_out, **kwargs):
        self.neurons = [Neuron(n_in, **kwargs)] * n_out
        self.n_out = n_out

    def __call__(self, x):
        """Returns a Value when n_out == 1, else a list of Values."""
        if self.n_out == 1:
            return self.neurons[0](x)
        else:
            out = []
            for neuron in self.neurons:
                out.append(neuron(x))
            return out

    def parameters(self):
        out = []
        for neuron in self.neurons:
            out.extend(neuron.parameters())
        return out


class MLP(Module):
    def __init__(self, n_in, n_outs):
        """n_outs: e.g. [16, 16, 1] -> Layer(n_in,16), Layer(16,16), Layer(16,1)."""
        self.layers = [Layer(n_in, n_outs[0])]
        i = 0
        while i < len(n_outs) - 1:
            self.layers.append(Layer(n_outs[i], n_outs[i + 1]))
            i += 1

    def __call__(self, x):
        output = []
        for layer in self.layers:
            output = layer(x)
            x = output if isinstance(output, list) else [output]
        return output

    def parameters(self):
        out = []
        for x in self.layers:
            out.extend(x.parameters())
        return out
