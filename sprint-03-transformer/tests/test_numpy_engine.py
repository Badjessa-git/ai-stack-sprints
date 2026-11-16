"""Tests for the NumPy engine (src/numpy_engine.py).

Every op gets a gradcheck: the analytic gradient from backward() against
central finite differences. If an op's forward is right but its backward
is wrong, the gradcheck catches it — the same discipline as the scalar
engine's tests.

Run from sprint-03-transformer/:
    python3 -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.numpy_engine import Tensor, concat  # noqa: E402


def gradcheck_tensor(fn, xs, eps=1e-6, tol=1e-4):
    """Central finite differences vs .grad after backward().

    fn() must build a FRESH graph on every call (same parameters, new
    Tensors) and return a scalar Tensor. xs are the parameter Tensors
    whose gradients are checked, element by element.
    """
    loss = fn()
    loss.backward()
    analytic = [x.grad.copy() for x in xs]
    numeric = []
    for x in xs:
        num = np.zeros_like(x.data)
        it = np.nditer(x.data, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = x.data[idx]
            x.data[idx] = orig + eps
            lp = float(fn().data)
            x.data[idx] = orig - eps
            lm = float(fn().data)
            x.data[idx] = orig
            num[idx] = (lp - lm) / (2 * eps)
            it.iternext()
        numeric.append(num)
    for a, n in zip(analytic, numeric):
        assert np.allclose(a, n, atol=tol), (
            f"gradcheck failed:\nanalytic {a}\nnumeric {n}"
        )


class TestExpLog(unittest.TestCase):
    def test_exp_forward(self):
        self.assertAlmostEqual(float(Tensor(2.0).exp().data), 7.389056, places=5)

    def test_log_forward(self):
        self.assertAlmostEqual(float(Tensor(2.0).log().data), 0.693147, places=5)

    def test_exp_grad(self):
        x = Tensor(1.0)
        x.exp().backward()
        self.assertAlmostEqual(float(x.grad), 2.718282, places=4)

    def test_log_grad(self):
        x = Tensor(2.0)
        x.log().backward()
        self.assertAlmostEqual(float(x.grad), 0.5, places=4)

    def test_exp_log_gradcheck(self):
        x = Tensor(np.array([1.0, 2.0]))
        gradcheck_tensor(lambda: x.exp().sum(), [x])
        gradcheck_tensor(lambda: x.log().sum(), [x])


class TestBinaryOps(unittest.TestCase):
    def test_add_broadcast(self):
        # (2, 3) + (3,) broadcasts the row; the bias grad must sum rows.
        a = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        b = Tensor(np.array([10.0, 20.0, 30.0]))
        gradcheck_tensor(lambda: (a + b).sum(), [a, b])
        gradcheck_tensor(lambda: ((a + b) ** 2).sum(), [a, b])

    def test_mul_broadcast(self):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = Tensor(np.array([2.0, 3.0]))
        gradcheck_tensor(lambda: (a * b).sum(), [a, b])

    def test_sub_div(self):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = Tensor(np.array([[2.0, 1.0], [1.0, 2.0]]))
        gradcheck_tensor(lambda: (a - b).sum(), [a, b])
        gradcheck_tensor(lambda: (a / b).sum(), [a, b])
        gradcheck_tensor(lambda: (a / 2.0).sum(), [a])
        gradcheck_tensor(lambda: (2.0 / a).sum(), [a])

    def test_scalar_ops(self):
        a = Tensor(np.array([1.0, 2.0, 3.0]))
        gradcheck_tensor(lambda: (a * 3.0 + 1.0).sum(), [a])
        gradcheck_tensor(lambda: (1.0 + a).sum(), [a])
        gradcheck_tensor(lambda: (-a).sum(), [a])


class TestMatmul(unittest.TestCase):
    def test_forward(self):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]))
        out = a @ b
        np.testing.assert_allclose(
            out.data, np.array([[19.0, 22.0], [43.0, 50.0]]), atol=1e-12
        )

    def test_gradcheck(self):
        a = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        b = Tensor(np.array([[0.5, -1.0], [2.0, 0.5], [-1.0, 1.5]]))
        gradcheck_tensor(lambda: (a @ b).sum(), [a, b])
        gradcheck_tensor(lambda: ((a @ b) ** 2).sum(), [a, b])


class TestUnaryOps(unittest.TestCase):
    def test_tanh_gradcheck(self):
        x = Tensor(np.array([[0.5, -1.0], [0.0, 2.0]]))
        gradcheck_tensor(lambda: x.tanh().sum(), [x])

    def test_pow_gradcheck(self):
        x = Tensor(np.array([0.5, 1.5, 2.0]))
        gradcheck_tensor(lambda: (x**2).sum(), [x])
        gradcheck_tensor(lambda: (x**3).sum(), [x])
        gradcheck_tensor(lambda: (x**0.5).sum(), [x])


class TestReductions(unittest.TestCase):
    def test_sum_axes(self):
        x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        gradcheck_tensor(lambda: x.sum(), [x])
        gradcheck_tensor(lambda: x.sum(axis=0).sum(), [x])
        gradcheck_tensor(lambda: x.sum(axis=1).sum(), [x])
        gradcheck_tensor(lambda: x.sum(axis=1, keepdims=True).sum(), [x])

    def test_mean_axes(self):
        x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        gradcheck_tensor(lambda: x.mean(), [x])
        gradcheck_tensor(lambda: x.mean(axis=1).sum(), [x])
        gradcheck_tensor(lambda: x.mean(axis=0, keepdims=True).sum(), [x])

    def test_transpose(self):
        x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        gradcheck_tensor(lambda: x.transpose().sum(), [x])
        gradcheck_tensor(lambda: (x.transpose() @ x).sum(), [x])


class TestGetitem(unittest.TestCase):
    def test_row_lookup(self):
        w = Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        gradcheck_tensor(lambda: w[[0, 2]].sum(), [w])
        gradcheck_tensor(lambda: w[[1, 1]].sum(), [w])  # repeated index

    def test_slice(self):
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        gradcheck_tensor(lambda: x[:-1].sum(), [x])
        gradcheck_tensor(lambda: x[1].sum(), [x])

    def test_fancy_index(self):
        x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        rows = np.array([0, 1, 0])
        cols = np.array([2, 0, 1])
        gradcheck_tensor(lambda: x[(rows, cols)].sum(), [x])


class TestConcat(unittest.TestCase):
    def test_forward_and_grad(self):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = Tensor(np.array([[5.0, 6.0]]))
        out = concat([a, b], axis=0)
        np.testing.assert_allclose(
            out.data, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), atol=1e-12
        )
        gradcheck_tensor(lambda: concat([a, b], axis=0).sum(), [a, b])
        c = Tensor(np.array([[7.0], [8.0]]))
        gradcheck_tensor(lambda: concat([a, c], axis=1).sum(), [a, c])


if __name__ == "__main__":
    unittest.main()
