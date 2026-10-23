"""NN-layer tests. Run after test_engine.py is green.

The last test is the payoff: a tiny MLP that actually learns. It needs the
whole engine plus all of nn.py, so save it for last on Saturday.
"""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine import Value
from src.nn import MLP, Layer, Module, Neuron


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


class TestNN(unittest.TestCase):
    def test_neuron_param_count(self):
        # n_in weights + 1 bias
        self.assertEqual(len(Neuron(3).parameters()), 4)

    def test_layer_param_count(self):
        self.assertEqual(len(Layer(2, 3).parameters()), 9)

    def test_mlp_param_count(self):
        # (2*4+4) + (4*4+4) + (4*1+1) = 37
        self.assertEqual(len(MLP(2, [4, 4, 1]).parameters()), 37)

    def test_mlp_forward_shapes(self):
        random.seed(0)
        x = [2.0, 3.0]
        out1 = MLP(2, [4, 1])(x)
        self.assertIsInstance(out1, Value)
        out2 = MLP(2, [4, 2])(x)
        self.assertIsInstance(out2, list)
        self.assertEqual(len(out2), 2)
        self.assertTrue(all(isinstance(v, Value) for v in out2))

    def test_zero_grad(self):
        random.seed(1)
        mlp = MLP(2, [4, 1])
        loss = mlp([2.0, -1.0]) ** 2
        loss.backward()
        self.assertTrue(any(p.grad != 0.0 for p in mlp.parameters()))
        mlp.zero_grad()
        self.assertTrue(all(p.grad == 0.0 for p in mlp.parameters()))

    def test_training_loss_decreases(self):
        # Six points, two obvious clusters. Plain SGD, 100 steps.
        # If the engine is right, the loss MUST go down.
        random.seed(0)
        mlp = MLP(2, [4, 1])
        data = [
            ([2.0, 3.0], 1.0), ([3.0, 1.0], 1.0), ([1.0, 2.0], 1.0),
            ([-2.0, -3.0], -1.0), ([-3.0, -1.0], -1.0), ([-1.0, -2.0], -1.0),
        ]

        def loss_fn():
            errs = [ (mlp(x) - y) ** 2 for x, y in data ]
            return sum(errs, Value(0.0)) * (1.0 / len(data))

        initial = loss_fn().data
        lr = 0.1
        for _ in range(100):
            loss = loss_fn()
            mlp.zero_grad()
            loss.backward()
            for p in mlp.parameters():
                p.data -= lr * p.grad
        final = loss_fn().data
        self.assertLess(final, initial,
                        f"loss did not decrease: {initial:.4f} -> {final:.4f}")


if __name__ == "__main__":
    unittest.main()
