"""Tests are the spec. Run: python3 -m unittest discover -s tests -v

Attack them in file order: forward ops first (each needs only __init__ plus
one op), then backward on tiny graphs, then the numerical gradient check —
that one is the real correctness gate. Every test currently fails with
NotImplementedError. That is the design.
"""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine import Value


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


class TestForward(unittest.TestCase):
    def test_add_forward(self):
        self.assertEqual((Value(2.0) + Value(3.0)).data, 5.0)

    def test_add_scalar_coercion(self):
        self.assertEqual((Value(2.0) + 3).data, 5.0)
        self.assertEqual((3 + Value(2.0)).data, 5.0)

    def test_mul_forward(self):
        self.assertEqual((Value(2.0) * Value(3.0)).data, 6.0)

    def test_pow_forward(self):
        self.assertEqual((Value(3.0) ** 2).data, 9.0)

    def test_div_forward(self):
        self.assertEqual((Value(6.0) / Value(2.0)).data, 3.0)

    def test_neg_sub_forward(self):
        self.assertEqual((-Value(2.0)).data, -2.0)
        self.assertEqual((Value(5.0) - Value(3.0)).data, 2.0)

    def test_relu_forward(self):
        self.assertEqual(Value(-2.0).relu().data, 0.0)
        self.assertEqual(Value(3.0).relu().data, 3.0)

    def test_tanh_forward(self):
        self.assertTrue(close(Value(0.0).tanh().data, 0.0))
        self.assertTrue(close(Value(1.0).tanh().data, math.tanh(1.0)))


class TestBackward(unittest.TestCase):
    def test_backward_simple(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        d = c + a
        d.backward()
        self.assertTrue(close(d.grad, 1.0))
        self.assertTrue(close(c.grad, 1.0))
        self.assertTrue(close(a.grad, 4.0))
        self.assertTrue(close(b.grad, 2.0))

    def test_gradient_accumulation(self):
        # x feeds TWO ops: its grad must be the SUM of both paths, not one.
        # (+=, never = — this is the bug the test is hunting.)
        x = Value(3.0)
        y = x + x
        y.backward()
        self.assertTrue(close(y.data, 6.0))
        self.assertTrue(close(x.grad, 2.0))

    def test_diamond(self):
        a = Value(2.0)
        b = a + a
        c = b * b
        c.backward()
        self.assertTrue(close(c.data, 16.0))
        self.assertTrue(close(a.grad, 16.0))

    def test_chain_rule_multi_level(self):
        a = Value(2.0)
        b = Value(-3.0)
        c = Value(10.0)
        e = a * b
        d = e + c
        f = Value(-2.0)
        L = d * f
        L.backward()
        # e=-6, d=4, L=-8; hand-derived below, check yours against it
        expect = [
            (L, 1.0),
            (d, -2.0),
            (f, 4.0),
            (e, -2.0),
            (c, -2.0),
            (a, 6.0),
            (b, -4.0),
        ]
        for node, g in expect:
            self.assertTrue(close(node.grad, g), f"expected grad {g}, got {node.grad}")

    def test_relu_backward(self):
        x = Value(-2.0)
        y = x.relu()
        y.backward()
        self.assertTrue(close(x.grad, 0.0))
        x2 = Value(3.0)
        y2 = x2.relu()
        y2.backward()
        self.assertTrue(close(x2.grad, 1.0))

    def test_pow_backward(self):
        x = Value(3.0)
        y = x**2
        y.backward()
        self.assertTrue(close(x.grad, 6.0))

    def test_div_backward(self):
        a = Value(6.0)
        b = Value(2.0)
        c = a / b
        c.backward()
        self.assertTrue(close(a.grad, 0.5))
        self.assertTrue(close(b.grad, -1.5))

    def test_deep_chain(self):
        # 100-deep chain with the same node feeding 100 ops: stresses both
        # topological order and accumulation. y = 101*x, so dy/dx = 101.
        x = Value(2.0)
        y = x
        for _ in range(100):
            y = y + x
        self.assertTrue(close(y.data, 202.0))
        y.backward()
        self.assertTrue(close(x.grad, 101.0))

    def test_backward_twice_resets_grads(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        c.backward()
        first = b.grad
        c.backward()
        self.assertTrue(
            close(b.grad, first),
            "second backward() must recompute, not pile onto old grads",
        )

    def test_numerical_gradient_check(self):
        # THE correctness gate: analytic grads vs central finite differences.
        # If this passes, your chain rule is right; if not, nothing else matters.
        def build(av, bv):
            a = Value(av)
            b = Value(bv)
            x = a * b + b**2
            return a, b, x.tanh()

        a, b, y = build(2.0, -3.0)
        y.backward()
        analytic = [a.grad, b.grad]

        h = 1e-5
        base = [2.0, -3.0]
        for i in range(2):
            up = base.copy()
            dn = base.copy()
            up[i] += h
            dn[i] -= h
            _, _, yu = build(*up)
            _, _, yd = build(*dn)
            numeric = (yu.data - yd.data) / (2 * h)
            self.assertTrue(
                close(analytic[i], numeric, tol=1e-4),
                f"leaf {i}: analytic {analytic[i]} vs numeric {numeric}",
            )


if __name__ == "__main__":
    unittest.main()
