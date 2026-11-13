"""Tests for the Sprint 3 transformer — every test is a contract.

Run from sprint-03-transformer/:
    make test          # or: python3 -m unittest discover -s tests -v

How to read a failure:
  * A test that raises NotImplementedError means that piece is not built yet.
  * A test that returns wrong numbers means the math is off — check the
    forward formula first (the docstrings in src/transformer.py state each
    one in plain words), then the reverse share.
  * TestTransformerTraining is the acceptance test for the whole sprint.
    It only passes when the transformer is right AND one training step
    finishes in reasonable time. If it hangs, the problem is in
    Value.backward() — see the Step 0 note in README.md before debugging
    the transformer itself.

All expected numbers were computed from a reference implementation of this
same API. Tolerances are loose (1e-4/1e-5) on purpose: a different but
correct order of additions changes the last digits, not the answer.
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine import Value  # noqa: E402
from src.transformer import (  # noqa: E402
    Block,
    CausalSelfAttention,
    Embedding,
    FeedForward,
    LayerNorm,
    MultiHeadAttention,
    Transformer,
    gelu,
    softmax,
)

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "corpus.txt")


def gradcheck(fn, xs, eps=1e-6, tol=1e-4):
    """Central finite differences vs .grad after backward().

    fn() must build a FRESH forward graph on every call (same parameters,
    new Values), and return a scalar Value. xs are the parameter Values
    whose gradients are checked.
    """
    loss = fn()
    loss.backward()
    analytic = [x.grad for x in xs]
    numeric = []
    for x in xs:
        orig = x.data
        x.data = orig + eps
        lp = fn().data
        x.data = orig - eps
        lm = fn().data
        x.data = orig
        numeric.append((lp - lm) / (2 * eps))
    for a, n in zip(analytic, numeric):
        assert abs(a - n) < tol, f"gradcheck failed: analytic {a}, numeric {n}"


def set_layer(layer, per_neuron):
    """Overwrite a Layer's weights with fixed constants (biases to 0)."""
    for neuron, w in zip(layer.neurons, per_neuron):
        for weight in neuron.weights:
            weight.data = w
        neuron.bias.data = 0.0


def approx_list(got, want, tol=1e-5):
    assert len(got) == len(want), f"length {len(got)} != {len(want)}"
    for g, w in zip(got, want):
        assert abs(g - w) < tol, f"got {g}, want {w}"


class TestExpLog(unittest.TestCase):
    def test_exp_forward(self):
        self.assertAlmostEqual(Value(2.0).exp().data, 7.389056, places=5)

    def test_log_forward(self):
        self.assertAlmostEqual(Value(2.0).log().data, 0.693147, places=5)

    def test_exp_grad(self):
        # d/dx exp(x) at x=1 is e.
        x = Value(1.0)
        out = x.exp()
        out.backward()
        self.assertAlmostEqual(x.grad, 2.718282, places=4)

    def test_log_grad(self):
        # d/dx log(x) at x=2 is 1/2.
        x = Value(2.0)
        out = x.log()
        out.backward()
        self.assertAlmostEqual(x.grad, 0.5, places=4)

    def test_exp_log_gradcheck(self):
        x = Value(1.0)
        gradcheck(lambda: x.exp(), [x])
        y = Value(2.0)
        gradcheck(lambda: y.log(), [y])


class TestSoftmax(unittest.TestCase):
    def test_known_values(self):
        out = softmax([Value(1.0), Value(2.0), Value(3.0)])
        approx_list([v.data for v in out], [0.090031, 0.244729, 0.665241], tol=1e-4)

    def test_sums_to_one(self):
        out = softmax([Value(0.3), Value(-1.2), Value(2.5), Value(0.0)])
        self.assertAlmostEqual(sum(v.data for v in out), 1.0, places=9)

    def test_no_overflow_on_large_inputs(self):
        # Without max-subtraction this is exp(1000) = inf.
        out = softmax([Value(1000.0), Value(1001.0)])
        approx_list([v.data for v in out], [0.268941, 0.731059], tol=1e-5)


class TestGelu(unittest.TestCase):
    def test_known_values(self):
        self.assertAlmostEqual(gelu(Value(0.0)).data, 0.0, places=6)
        self.assertAlmostEqual(gelu(Value(1.0)).data, 0.841192, places=5)
        self.assertAlmostEqual(gelu(Value(-1.0)).data, -0.158808, places=5)
        self.assertAlmostEqual(gelu(Value(0.5)).data, 0.345714, places=5)

    def test_gradcheck(self):
        x = Value(0.5)
        gradcheck(lambda: gelu(x), [x])


class TestAttention(unittest.TestCase):
    def _fixed_attn(self):
        attn = CausalSelfAttention(4, 2)
        set_layer(attn.Wq, [0.5, -0.5])
        set_layer(attn.Wk, [0.25, 0.75])
        set_layer(attn.Wv, [0.1, 0.3])
        return attn

    def _fixed_x(self):
        return [[Value(v) for v in row]
                for row in [[0.5, -0.5, 1.0, 0.0], [1.0, 0.0, -1.0, 0.5]]]

    def test_fixed_weights_expected_output(self):
        out = self._fixed_attn()(self._fixed_x())
        approx_list([v.data for v in out[0]], [0.1, 0.3])
        approx_list([v.data for v in out[1]], [0.074448, 0.223343])

    def test_causal_mask(self):
        """Changing the LAST token must not change earlier outputs."""
        random.seed(0)
        attn = CausalSelfAttention(4, 2)
        x = self._fixed_x()
        first = attn(x)
        x2 = [[Value(v) for v in row]
              for row in [[0.5, -0.5, 1.0, 0.0], [9.0, 9.0, 9.0, 9.0]]]
        second = attn(x2)
        approx_list([v.data for v in first[0]], [v.data for v in second[0]],
                    tol=1e-9)

    def test_gradcheck(self):
        attn = self._fixed_attn()
        w = attn.Wq.neurons[0].weights[0]

        def total():
            out = attn(self._fixed_x())
            s = Value(0.0)
            for vec in out:
                for v in vec:
                    s = s + v
            return s

        gradcheck(total, [w])


class TestMultiHead(unittest.TestCase):
    def test_equals_concat_of_heads_with_identity_wout(self):
        mha = MultiHeadAttention(4, 2)
        # Wout = identity: output is just the two heads side by side.
        for i, neuron in enumerate(mha.Wout.neurons):
            for j, weight in enumerate(neuron.weights):
                weight.data = 1.0 if i == j else 0.0
            neuron.bias.data = 0.0
        x = [[Value(v) for v in row]
             for row in [[0.5, -0.5, 1.0, 0.0], [1.0, 0.0, -1.0, 0.5]]]
        out = mha(x)
        for t in range(2):
            manual = mha.heads[0](x)[t] + mha.heads[1](x)[t]
            approx_list([v.data for v in out[t]], [v.data for v in manual],
                        tol=1e-9)

    def test_shape(self):
        mha = MultiHeadAttention(8, 2)
        x = [[Value(0.1 * i) for i in range(8)] for _ in range(3)]
        out = mha(x)
        self.assertEqual(len(out), 3)
        self.assertEqual(len(out[0]), 8)


class TestLayerNorm(unittest.TestCase):
    def test_known_values(self):
        ln = LayerNorm(4)
        out = ln([Value(v) for v in [1.0, 2.0, 3.0, 4.0]])
        approx_list([v.data for v in out],
                    [-1.341641, -0.447214, 0.447214, 1.341641], tol=1e-4)

    def test_output_mean_zero_var_one(self):
        ln = LayerNorm(4)
        out = [v.data for v in ln([Value(v) for v in [1.0, 2.0, 3.0, 4.0]])]
        mean = sum(out) / 4
        var = sum((o - mean) ** 2 for o in out) / 4
        self.assertAlmostEqual(mean, 0.0, places=6)
        self.assertAlmostEqual(var, 1.0, places=4)

    def test_gradcheck(self):
        ln = LayerNorm(4)
        xs = [Value(v) for v in [1.0, 2.0, 3.0, 4.0]]
        gradcheck(lambda: sum(ln(xs), Value(0.0)), xs)


class TestFeedForward(unittest.TestCase):
    def test_fixed_weights(self):
        ff = FeedForward(4, hidden=8)
        set_layer(ff.fc1, [0.1] * 8)
        set_layer(ff.fc2, [0.2] * 4)
        out = ff([Value(v) for v in [0.5, -0.5, 1.0, 0.0]])
        approx_list([v.data for v in out], [0.086372] * 4, tol=1e-5)

    def test_shape(self):
        ff = FeedForward(4)
        out = ff([Value(0.5) for _ in range(4)])
        self.assertEqual(len(out), 4)


class TestBlock(unittest.TestCase):
    def test_zero_weights_is_identity(self):
        """All weights zero => attention and MLP both output 0, and the two
        residual adds return the input unchanged. The sharpest test in the
        suite: it fails if a residual is missing or backwards."""
        random.seed(0)
        block = Block(4, 2)
        for p in block.parameters():
            p.data = 0.0
        x = [[Value(v) for v in row]
             for row in [[0.5, -0.5, 1.0, 0.0], [1.0, 0.0, -1.0, 0.5]]]
        out = block(x)
        for t in range(2):
            approx_list([v.data for v in out[t]], [v.data for v in x[t]],
                        tol=1e-9)

    def test_shape(self):
        block = Block(8, 2)
        x = [[Value(0.1) for _ in range(8)] for _ in range(3)]
        out = block(x)
        self.assertEqual(len(out), 3)
        self.assertEqual(len(out[0]), 8)


class TestEmbedding(unittest.TestCase):
    def test_returns_the_actual_row(self):
        emb = Embedding(5, 3)
        row = emb(2)
        for a, b in zip(row, emb.weight[2]):
            self.assertIs(a, b)


class TestTransformer(unittest.TestCase):
    def _corpus_ids(self):
        with open(CORPUS) as f:
            text = f.read()
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        return [stoi[c] for c in text[:2000]], chars

    def test_forward_shape(self):
        random.seed(7)
        model = Transformer(58, d_model=8, n_head=2, n_layer=1, block_size=4)
        logits = model([15, 40, 49, 50])
        self.assertEqual(len(logits), 4)
        self.assertEqual(len(logits[0]), 58)

    def test_loss_value(self):
        random.seed(7)
        data, _ = self._corpus_ids()
        model = Transformer(58, d_model=8, n_head=2, n_layer=1, block_size=4)
        loss = model.loss(data[:4])
        self.assertAlmostEqual(loss.data, 4.190602, places=4)

    def test_training_loss_decreases(self):
        """The acceptance test: 5 SGD steps on one fixed 4-token batch must
        lower the loss. If this hangs, read the Step 0 note in README.md —
        the fault is in Value.backward(), not in the transformer."""
        random.seed(7)
        data, _ = self._corpus_ids()
        model = Transformer(58, d_model=8, n_head=2, n_layer=1, block_size=4)
        idx = data[:4]
        before = model.loss(idx).data
        for _ in range(5):
            loss = model.loss(idx)
            model.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.data -= 0.05 * p.grad
        after = model.loss(idx).data
        self.assertLess(after, before,
                        f"loss did not decrease: {before} -> {after}")

    def test_generate_greedy(self):
        random.seed(7)
        data, chars = self._corpus_ids()
        model = Transformer(58, d_model=8, n_head=2, n_layer=1, block_size=4)
        out = model.generate(data[:4], 8, greedy=True)
        self.assertEqual(len(out), 12)
        self.assertEqual(out[:4], data[:4])
        self.assertTrue(all(0 <= i < 58 for i in out))


if __name__ == "__main__":
    unittest.main()
