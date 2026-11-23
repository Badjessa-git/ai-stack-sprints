"""The spec for src/tensor.py. From the sprint folder: make test

Almost every test is a gradient check: compute the gradient with your
backward pass, compute it again by nudging each input number by ±1e-6 and
watching the output move, and require the two to agree. If they agree,
the backward pass is right, whatever it looks like inside.
"""

import os
import sys
import unittest
import warnings

import numpy as np

# NumPy 2.x on macOS (Accelerate) prints spurious 'overflow in matmul'
# warnings on ordinary inputs. Sprint 3 silences them the same way.
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tensor import Tensor, cross_entropy, unbroadcast  # noqa: E402

RNG = np.random.default_rng(0)


def numerical_grad(f, arrays, eps=1e-6):
    """Central differences: d f / d a[i] ~ (f(a+eps) - f(a-eps)) / 2eps."""
    grads = []
    for a in arrays:
        g = np.zeros_like(a)
        it = np.nditer(a, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = a[i]
            a[i] = old + eps
            hi = f(*arrays)
            a[i] = old - eps
            lo = f(*arrays)
            a[i] = old
            g[i] = (hi - lo) / (2 * eps)
        grads.append(g)
    return grads


def gradcheck(test, build, *shapes, positive=False, tol=1e-5):
    """build(*tensors) -> scalar Tensor. Checks every input's grad."""
    arrays = [RNG.standard_normal(s) for s in shapes]
    if positive:
        arrays = [np.abs(a) + 0.5 for a in arrays]
    tensors = [Tensor(a.copy()) for a in arrays]
    out = build(*tensors)
    test.assertEqual(out.shape, (), "build must return a scalar")
    out.backward()
    expected = numerical_grad(lambda *xs: build(*[Tensor(x) for x in xs]).data.item(), arrays)
    for i, (t, e) in enumerate(zip(tensors, expected)):
        test.assertEqual(t.grad.shape, e.shape, f"input {i}: grad shape")
        np.testing.assert_allclose(t.grad, e, rtol=tol, atol=tol, err_msg=f"input {i}")


class TestUnbroadcast(unittest.TestCase):
    def test_leading_axis(self):
        np.testing.assert_array_equal(unbroadcast(np.ones((2, 3)), (3,)), [2, 2, 2])

    def test_size_one_axis(self):
        np.testing.assert_array_equal(unbroadcast(np.ones((2, 3)), (2, 1)), [[3], [3]])
        np.testing.assert_array_equal(unbroadcast(np.ones((2, 3)), (1, 3)), [[2, 2, 2]])

    def test_both(self):
        g = unbroadcast(np.ones((4, 2, 3)), (2, 1))
        self.assertEqual(g.shape, (2, 1))
        np.testing.assert_array_equal(g, [[12], [12]])

    def test_to_scalar(self):
        self.assertEqual(float(unbroadcast(np.ones((2, 3)), ())), 6.0)

    def test_no_broadcast_is_identity(self):
        g = RNG.standard_normal((2, 3))
        np.testing.assert_array_equal(unbroadcast(g, (2, 3)), g)


class TestElementwise(unittest.TestCase):
    def test_add_broadcast(self):  # already passes once unbroadcast works
        gradcheck(self, lambda a, b: ((a + b) ** 2).sum(), (2, 3), (3,))

    def test_mul(self):
        gradcheck(self, lambda a, b: (a * b).sum(), (2, 3), (2, 3))

    def test_mul_broadcast(self):
        gradcheck(self, lambda a, b: (a * b).sum(), (2, 3), (2, 1))
        gradcheck(self, lambda a, b: (a * b * a).sum(), (4, 2, 3), (3,))

    def test_pow_neg_sub_div(self):
        gradcheck(self, lambda a: (a**3).sum(), (3, 2))
        gradcheck(self, lambda a, b: (a - b).sum() + (-a).sum(), (3,), (3,))
        gradcheck(self, lambda a, b: (a / b).sum(), (2, 2), (2, 2), positive=True)

    def test_plain_numbers(self):
        gradcheck(self, lambda a: (2 * a + 1 - a / 4).sum(), (3,))
        gradcheck(self, lambda a: (1 / a).sum(), (3,), positive=True)

    def test_exp_log(self):
        gradcheck(self, lambda a: a.exp().sum(), (2, 3))
        gradcheck(self, lambda a: a.log().sum(), (2, 3), positive=True)

    def test_tanh_relu(self):
        gradcheck(self, lambda a: a.tanh().sum(), (2, 3))
        # relu at exactly 0 has no derivative; random normals never hit it
        gradcheck(self, lambda a: (a.relu() * a).sum(), (2, 3))


class TestReductions(unittest.TestCase):
    def test_sum_all(self):
        gradcheck(self, lambda a: (a.sum() ** 2), (2, 3))

    def test_sum_axis(self):
        w = Tensor(RNG.standard_normal(3))
        gradcheck(self, lambda a: (a.sum(axis=0) * w).sum(), (2, 3))
        gradcheck(self, lambda a: (a.sum(axis=1) ** 2).sum(), (2, 3))
        gradcheck(self, lambda a: (a.sum(axis=-1) ** 2).sum(), (4, 2, 3))

    def test_sum_keepdims(self):
        gradcheck(self, lambda a: (a * a.sum(axis=1, keepdims=True)).sum(), (2, 3))

    def test_sum_tuple_axis(self):
        gradcheck(self, lambda a: (a.sum(axis=(0, 2)) ** 2).sum(), (4, 2, 3))

    def test_mean(self):
        t = Tensor(np.arange(6.0).reshape(2, 3))
        self.assertAlmostEqual(t.mean().data.item(), 2.5)
        gradcheck(self, lambda a: (a.mean(axis=0) ** 2).sum(), (4, 3))


class TestShape(unittest.TestCase):
    def test_reshape(self):
        gradcheck(self, lambda a: (a.reshape(3, 2) * Tensor(np.arange(6.0).reshape(3, 2))).sum(), (2, 3))

    def test_transpose(self):
        w = Tensor(RNG.standard_normal((3, 4, 2)))
        gradcheck(self, lambda a: (a.transpose(1, 2, 0) * w).sum(), (2, 3, 4))
        w2 = Tensor(RNG.standard_normal((3, 2)))
        gradcheck(self, lambda a: (a.transpose() * w2).sum(), (2, 3))


class TestMatmul(unittest.TestCase):
    def test_2d(self):
        a, b = RNG.standard_normal((2, 3)), RNG.standard_normal((3, 4))
        np.testing.assert_allclose((Tensor(a) @ Tensor(b)).data, a @ b)
        gradcheck(self, lambda a, b: ((a @ b) ** 2).sum(), (2, 3), (3, 4))

    def test_batched(self):
        gradcheck(self, lambda a, b: ((a @ b) ** 2).sum(), (4, 2, 3), (4, 3, 5))

    def test_batch_broadcast(self):
        # one weight matrix shared by every item in the batch
        gradcheck(self, lambda x, w: ((x @ w) ** 2).sum(), (4, 2, 3), (3, 5))
        gradcheck(self, lambda a, b: ((a @ b) ** 2).sum(), (2, 1, 2, 3), (5, 3, 2))


class TestSoftmaxCrossEntropy(unittest.TestCase):
    def test_softmax_rows_sum_to_one(self):
        s = Tensor(RNG.standard_normal((3, 5))).softmax(axis=-1).data
        np.testing.assert_allclose(s.sum(axis=-1), 1.0)
        self.assertTrue((s > 0).all())

    def test_softmax_is_stable(self):
        s = Tensor(np.array([[1000.0, 1000.0]])).softmax().data
        np.testing.assert_allclose(s, [[0.5, 0.5]])

    def test_softmax_grad(self):
        w = Tensor(RNG.standard_normal((3, 5)))
        gradcheck(self, lambda a: (a.softmax(axis=-1) * w).sum(), (3, 5))
        w0 = Tensor(RNG.standard_normal((3, 5)))
        gradcheck(self, lambda a: (a.softmax(axis=0) * w0).sum(), (3, 5))

    def test_cross_entropy_value(self):
        logits = RNG.standard_normal((4, 6))
        targets = np.array([0, 5, 2, 2])
        p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        expected = -np.log(p[np.arange(4), targets]).mean()
        self.assertAlmostEqual(cross_entropy(Tensor(logits), targets).data.item(), expected)

    def test_cross_entropy_uniform(self):
        # equal logits: loss is ln(C) whatever the targets
        out = cross_entropy(Tensor(np.zeros((3, 10))), [1, 2, 3])
        self.assertAlmostEqual(out.data.item(), np.log(10))

    def test_cross_entropy_grad(self):
        targets = np.array([1, 0, 3])
        gradcheck(self, lambda a: cross_entropy(a, targets) * 3.0, (3, 4))


class TestBackward(unittest.TestCase):
    def test_shared_node(self):
        # y = x*x + x: x feeds three edges. dy/dx = 2x + 1.
        x = Tensor(np.array([1.0, -2.0, 3.0]))
        (x * x + x).sum().backward()
        np.testing.assert_allclose(x.grad, [3.0, -3.0, 7.0])

    def test_diamond(self):
        # a -> b, a -> c, (b, c) -> d: a's grad must arrive via both paths,
        # and d's backward must run before b's and c's.
        a = Tensor(np.array([2.0]))
        b, c = a * 3.0, a * a
        (b * c).sum().backward()  # d = 3a^3, dd/da = 9a^2 = 36
        np.testing.assert_allclose(a.grad, [36.0])

    def test_deep_chain_runs(self):
        # 2,000 ops in a row. Must not hit Python's recursion limit if your
        # topological sort is recursive... raise the limit, or go iterative.
        x = Tensor(np.array([0.5]))
        y = x
        for _ in range(2000):
            y = y * 1.0001
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))
        y.sum().backward()
        self.assertAlmostEqual(x.grad[0], 1.0001**2000, places=6)


class TestAcceptance(unittest.TestCase):
    def test_causal_attention_head(self):
        """One attention head, built only from your ops: the same math as
        Sprint 3's CausalSelfAttention, gradient-checked end to end."""
        T, d = 4, 3
        mask = np.triu(np.full((T, T), -1e9), k=1)

        def head(x, wq, wk, wv):
            q, k, v = x @ wq, x @ wk, x @ wv
            att = (q @ k.swap_last()) * (1 / np.sqrt(d)) + mask
            return ((att.softmax(axis=-1) @ v) ** 2).sum()

        gradcheck(self, head, (2, T, d), (d, d), (d, d), (d, d), tol=1e-4)

    def test_mlp_learns(self):
        """2-layer MLP on a tiny 3-class problem: loss must fall below 0.1."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((60, 2))
        y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int)
        w1 = Tensor(rng.standard_normal((2, 32)) * 0.5)
        b1 = Tensor(np.zeros(32))
        w2 = Tensor(rng.standard_normal((32, 3)) * 0.5)
        b2 = Tensor(np.zeros(3))
        params = [w1, b1, w2, b2]
        for _ in range(500):
            for p in params:
                p.grad = np.zeros_like(p.data)
            loss = cross_entropy((Tensor(X) @ w1 + b1).tanh() @ w2 + b2, y)
            loss.backward()
            for p in params:
                p.data -= 0.5 * p.grad
        self.assertLess(loss.data.item(), 0.1)


if __name__ == "__main__":
    unittest.main()
