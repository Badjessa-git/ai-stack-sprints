"""Tests for dropout (src/numpy/engine.py dropout()).

Dropout is YOUR implementation — these tests define the contract and
fail until you write it. The Transformer only calls dropout when the
rate is > 0, so a model with dropout=0 never reaches the function and
those tests pass even before you implement it.
"""

import os
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.numpy.engine import Tensor, dropout  # noqa: E402
from src.numpy.transformer import Transformer  # noqa: E402


class TestDropoutContract(unittest.TestCase):
    def test_eval_mode_returns_input_unchanged(self):
        """training=False: dropout is the identity, no mask, no scale."""
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        out = dropout(x, p=0.5, training=False)
        np.testing.assert_allclose(out.data, x.data)

    def test_p_zero_returns_input_unchanged(self):
        """p=0: nothing is dropped, no scale."""
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        out = dropout(x, p=0.0, training=True)
        np.testing.assert_allclose(out.data, x.data)

    def test_training_zeros_some_and_scales_kept(self):
        """Training, p=0.5, on a big array: about half the entries are 0,
        the kept entries are exactly x / (1 - p) = 2x, and no other
        values appear."""
        np.random.seed(0)
        x = Tensor(np.ones((200, 200)))
        out = dropout(x, p=0.5, training=True)
        vals, counts = np.unique(out.data, return_counts=True)
        self.assertEqual(set(vals.tolist()), {0.0, 2.0})
        frac_zero = counts[list(vals).index(0.0)] / out.data.size
        self.assertAlmostEqual(frac_zero, 0.5, delta=0.05)

    def test_backward_flows_through_mask(self):
        """Gradient into a dropped entry is 0; into a kept entry it is
        the output grad scaled by 1 / (1 - p)."""
        np.random.seed(1)
        x = Tensor(np.ones((4, 4)))
        out = dropout(x, p=0.5, training=True)
        out.grad = np.ones_like(out.data)
        # call the recorded backward directly: it pushes out.grad to x
        out._backward()
        # entries where out is 0 had grad 0; where out is 2, grad is 2
        expected = np.where(out.data == 0.0, 0.0, 2.0)
        np.testing.assert_allclose(x.grad, expected)

    def test_dropout_is_differentiable_end_to_end(self):
        """A loss built on a dropped tensor backpropagates without error
        and the grad has the mask shape (zeros where dropped)."""
        np.random.seed(2)
        x = Tensor(np.ones((3, 3)))
        out = dropout(x, p=0.5, training=True)
        loss = (out * out).sum()
        loss.backward()
        # grad is 2 * out (d/dx of out^2), masked the same way
        expected = 2 * out.data
        np.testing.assert_allclose(x.grad, expected)


class TestDropoutWiring(unittest.TestCase):
    def test_transformer_dropout_zero_never_calls_dropout(self):
        """dropout=0 (the default): the model trains fine even though
        dropout() is a stub — the Block never calls it."""
        random.seed(1)
        model = Transformer(58, d_model=16, n_head=2, n_layer=1, block_size=8,
                            dropout=0.0)
        loss = model.loss([15, 40, 49, 50, 1, 2, 3, 4])
        model.zero_grad()
        loss.backward()
        self.assertTrue(np.isfinite(float(loss.data)))

    def test_transformer_generate_never_drops(self):
        """generate() uses training=False, so no dropout even if the
        rate is > 0 — generation is deterministic given the weights."""
        random.seed(1)
        model = Transformer(58, d_model=16, n_head=2, n_layer=1, block_size=8,
                            dropout=0.5)
        out = model.generate([15, 40, 49, 50], 5, greedy=True)
        self.assertEqual(len(out), 9)


if __name__ == "__main__":
    unittest.main()
