"""Tests for the NumPy transformer (src/numpy/transformer.py).

These mirror tests/test_transformer.py: same expected numbers, same
shapes, same training acceptance test — because the NumPy engine is the
same math on arrays instead of one Value per number. Where the scalar
test uses a list of Values, this one uses one Tensor of shape (T, d).

The last class, TestCrossEngine, is the proof the two engines agree:
same seed, same loss, same logits, same gradients, on the real model.

Run from sprint-03-transformer/:
    python3 -m unittest discover -s tests -v
"""

import os
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.numpy.engine import Tensor  # noqa: E402
from src.numpy.transformer import (  # noqa: E402
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
from src.scalar.transformer import Transformer as ScalarTransformer  # noqa: E402
from test_numpy_engine import gradcheck_tensor  # noqa: E402

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "corpus.txt")


def set_linear(linear, per_output):
    """Overwrite a Linear's weights: every weight in output row j is
    per_output[j], biases to 0. Mirrors set_layer in the scalar tests."""
    W = np.zeros_like(linear.weight.data)
    for j, w in enumerate(per_output):
        W[j, :] = w
    linear.weight.data = W
    linear.bias.data = np.zeros_like(linear.bias.data)


def approx_array(got, want, tol=1e-5):
    got = np.asarray(got)
    want = np.asarray(want)
    assert got.shape == want.shape, f"shape {got.shape} != {want.shape}"
    assert np.allclose(got, want, atol=tol), f"got {got}, want {want}"


class TestSoftmax(unittest.TestCase):
    def test_known_values(self):
        out = softmax(Tensor(np.array([1.0, 2.0, 3.0])))
        approx_array(out.data, [0.090031, 0.244729, 0.665241], tol=1e-4)

    def test_sums_to_one(self):
        out = softmax(Tensor(np.array([0.3, -1.2, 2.5, 0.0])))
        self.assertAlmostEqual(float(out.data.sum()), 1.0, places=9)

    def test_no_overflow_on_large_inputs(self):
        out = softmax(Tensor(np.array([1000.0, 1001.0])))
        approx_array(out.data, [0.268941, 0.731059], tol=1e-5)


class TestGelu(unittest.TestCase):
    def test_known_values(self):
        self.assertAlmostEqual(float(gelu(Tensor(0.0)).data), 0.0, places=6)
        self.assertAlmostEqual(float(gelu(Tensor(1.0)).data), 0.841192, places=5)
        self.assertAlmostEqual(float(gelu(Tensor(-1.0)).data), -0.158808, places=5)
        self.assertAlmostEqual(float(gelu(Tensor(0.5)).data), 0.345714, places=5)

    def test_gradcheck(self):
        x = Tensor(np.array([0.5]))
        gradcheck_tensor(lambda: gelu(x).sum(), [x])


class TestAttention(unittest.TestCase):
    def _fixed_attn(self):
        attn = CausalSelfAttention(4, 2)
        set_linear(attn.Wq, [0.5, -0.5])
        set_linear(attn.Wk, [0.25, 0.75])
        set_linear(attn.Wv, [0.1, 0.3])
        return attn

    def _fixed_x(self):
        return Tensor(
            np.array([[0.5, -0.5, 1.0, 0.0], [1.0, 0.0, -1.0, 0.5]])
        )

    def test_fixed_weights_expected_output(self):
        out = self._fixed_attn()(self._fixed_x())
        approx_array(out.data[0], [0.1, 0.3])
        approx_array(out.data[1], [0.074448, 0.223343])

    def test_causal_mask(self):
        """Changing the LAST token must not change earlier outputs."""
        random.seed(0)
        attn = CausalSelfAttention(4, 2)
        x = self._fixed_x()
        first = attn(x)
        x2 = Tensor(np.array([[0.5, -0.5, 1.0, 0.0], [9.0, 9.0, 9.0, 9.0]]))
        second = attn(x2)
        approx_array(first.data[0], second.data[0], tol=1e-9)

    def test_gradcheck(self):
        attn = self._fixed_attn()
        w = attn.Wq.weight

        def total():
            return attn(self._fixed_x()).sum()

        gradcheck_tensor(total, [w])


class TestMultiHead(unittest.TestCase):
    def test_equals_concat_of_heads_with_identity_wout(self):
        mha = MultiHeadAttention(4, 2)
        # Wout = identity: output is just the two heads side by side.
        mha.Wout.weight.data = np.eye(4)
        mha.Wout.bias.data = np.zeros(4)
        x = Tensor(np.array([[0.5, -0.5, 1.0, 0.0], [1.0, 0.0, -1.0, 0.5]]))
        out = mha(x)
        for t in range(2):
            manual = np.concatenate(
                [mha.heads[0](x).data[t], mha.heads[1](x).data[t]]
            )
            approx_array(out.data[t], manual, tol=1e-9)

    def test_shape(self):
        mha = MultiHeadAttention(8, 2)
        x = Tensor(np.array([[0.1 * i for i in range(8)] for _ in range(3)]))
        out = mha(x)
        self.assertEqual(out.data.shape, (3, 8))


class TestLayerNorm(unittest.TestCase):
    def test_known_values(self):
        ln = LayerNorm(4)
        out = ln(Tensor(np.array([[1.0, 2.0, 3.0, 4.0]])))
        approx_array(
            out.data[0], [-1.341641, -0.447214, 0.447214, 1.341641], tol=1e-4
        )

    def test_output_mean_zero_var_one(self):
        ln = LayerNorm(4)
        out = ln(Tensor(np.array([[1.0, 2.0, 3.0, 4.0]]))).data[0]
        mean = out.mean()
        var = ((out - mean) ** 2).mean()
        self.assertAlmostEqual(float(mean), 0.0, places=6)
        self.assertAlmostEqual(float(var), 1.0, places=4)

    def test_gradcheck(self):
        ln = LayerNorm(4)
        xs = Tensor(np.array([[1.0, 2.0, 3.0, 4.0]]))
        gradcheck_tensor(lambda: ln(xs).sum(), [xs])


class TestFeedForward(unittest.TestCase):
    def test_fixed_weights(self):
        ff = FeedForward(4, hidden=8)
        set_linear(ff.fc1, [0.1] * 8)
        set_linear(ff.fc2, [0.2] * 4)
        out = ff(Tensor(np.array([[0.5, -0.5, 1.0, 0.0]])))
        approx_array(out.data[0], [0.086372] * 4, tol=1e-5)

    def test_shape(self):
        ff = FeedForward(4)
        out = ff(Tensor(np.array([[0.5] * 4])))
        self.assertEqual(out.data.shape, (1, 4))


class TestBlock(unittest.TestCase):
    def test_zero_weights_is_identity(self):
        """All weights zero => attention and MLP both output 0, and the
        two residual adds return the input unchanged."""
        random.seed(0)
        block = Block(4, 2)
        for p in block.parameters():
            p.data = np.zeros_like(p.data)
        x = Tensor(np.array([[0.5, -0.5, 1.0, 0.0], [1.0, 0.0, -1.0, 0.5]]))
        out = block(x)
        approx_array(out.data, x.data, tol=1e-9)

    def test_shape(self):
        block = Block(8, 2)
        x = Tensor(np.array([[0.1] * 8 for _ in range(3)]))
        out = block(x)
        self.assertEqual(out.data.shape, (3, 8))


class TestEmbedding(unittest.TestCase):
    def test_returns_the_row(self):
        emb = Embedding(5, 3)
        row = emb([2])
        approx_array(row.data[0], emb.weight.data[2], tol=1e-12)

    def test_grad_only_hits_used_rows(self):
        emb = Embedding(5, 3)
        emb([1, 3]).sum().backward()
        self.assertTrue(np.all(emb.weight.grad[0] == 0))
        self.assertTrue(np.all(emb.weight.grad[2] == 0))
        self.assertTrue(np.all(emb.weight.grad[4] == 0))
        self.assertFalse(np.all(emb.weight.grad[1] == 0))
        self.assertFalse(np.all(emb.weight.grad[3] == 0))


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
        self.assertEqual(logits.data.shape, (4, 58))

    def test_loss_value(self):
        random.seed(7)
        data, _ = self._corpus_ids()
        model = Transformer(58, d_model=8, n_head=2, n_layer=1, block_size=4)
        loss = model.loss(data[:4])
        self.assertAlmostEqual(float(loss.data), 4.190602, places=4)

    def test_training_loss_decreases(self):
        """Same acceptance test as the scalar suite: 5 SGD steps on one
        fixed 4-token batch must lower the loss."""
        random.seed(7)
        data, _ = self._corpus_ids()
        model = Transformer(58, d_model=8, n_head=2, n_layer=1, block_size=4)
        idx = data[:4]
        before = float(model.loss(idx).data)
        for _ in range(5):
            loss = model.loss(idx)
            model.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.data -= 0.05 * p.grad
        after = float(model.loss(idx).data)
        self.assertLess(after, before, f"loss did not decrease: {before} -> {after}")

    def test_generate_greedy(self):
        random.seed(7)
        data, chars = self._corpus_ids()
        model = Transformer(58, d_model=8, n_head=2, n_layer=1, block_size=4)
        out = model.generate(data[:4], 8, greedy=True)
        self.assertEqual(len(out), 12)
        self.assertEqual(out[:4], data[:4])
        self.assertTrue(all(0 <= i < 58 for i in out))


class TestCrossEngine(unittest.TestCase):
    """Same seed, same model shape: the two engines must agree on the
    loss, the logits, and the gradients. If the NumPy engine's math or
    its weight init drifted from the scalar engine, this catches it."""

    def _models(self):
        random.seed(7)
        scalar = ScalarTransformer(
            58, d_model=8, n_head=2, n_layer=1, block_size=4
        )
        random.seed(7)
        numpy = Transformer(58, d_model=8, n_head=2, n_layer=1, block_size=4)
        return scalar, numpy

    def test_loss_matches(self):
        scalar, numpy = self._models()
        with open(CORPUS) as f:
            text = f.read()
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        idx = [stoi[c] for c in text[:2000]][:4]
        s_loss = scalar.loss(idx).data
        n_loss = float(numpy.loss(idx).data)
        self.assertAlmostEqual(s_loss, n_loss, places=9)

    def test_logits_match(self):
        scalar, numpy = self._models()
        s_logits = np.array([[v.data for v in row] for row in scalar([15, 40, 49, 50])])
        n_logits = numpy([15, 40, 49, 50]).data
        np.testing.assert_allclose(s_logits, n_logits, atol=1e-9)

    def test_gradients_match(self):
        scalar, numpy = self._models()
        with open(CORPUS) as f:
            text = f.read()
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        idx = [stoi[c] for c in text[:2000]][:4]
        scalar.loss(idx).backward()
        numpy.loss(idx).backward()
        # spot-check gradients in several places, scalar vs NumPy
        pairs = [
            (scalar.tok_emb.weight[15][0].grad, numpy.tok_emb.weight.grad[15, 0]),
            (scalar.lm_head.neurons[0].weights[0].grad, numpy.lm_head.weight.grad[0, 0]),
            (scalar.lm_head.neurons[0].bias.grad, numpy.lm_head.bias.grad[0]),
            (
                scalar.blocks[0].attn.heads[0].Wq.neurons[0].weights[0].grad,
                numpy.blocks[0].attn.heads[0].Wq.weight.grad[0, 0],
            ),
            (scalar.blocks[0].ln1.gamma[0].grad, numpy.blocks[0].ln1.gamma.grad[0]),
            (
                scalar.blocks[0].mlp.fc1.neurons[0].weights[0].grad,
                numpy.blocks[0].mlp.fc1.weight.grad[0, 0],
            ),
        ]
        for s_grad, n_grad in pairs:
            self.assertAlmostEqual(s_grad, float(n_grad), places=7)


if __name__ == "__main__":
    unittest.main()
