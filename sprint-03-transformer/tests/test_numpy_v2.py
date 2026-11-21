"""Regression tests for the training fixes: init scale, batching, float32,
stable loss, and the sampler."""

import math
import os
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.numpy_v2 import engine  # noqa: E402
from src.numpy_v2.transformer import Transformer  # noqa: E402


class TestInitScale(unittest.TestCase):
    def test_initial_loss_near_uniform_at_every_width(self):
        """A fresh model knows nothing, so its loss should be ~ln(vocab)
        regardless of d_model. Unscaled U(-1,1) init gave 26.7 / 41.4 /
        59.0 at d256 / d512 / d1024 with vocab 512."""
        V = 512
        rng = np.random.default_rng(0)
        idx = rng.integers(0, V, size=32).tolist()
        for d in (64, 256, 512):
            random.seed(1)
            model = Transformer(V, d_model=d, n_head=2, n_layer=2, block_size=32)
            loss = float(model.loss(idx).data)
            self.assertLess(abs(loss - math.log(V)), 1.0, f"d{d}: loss {loss}")

    def test_attention_not_saturated_at_init(self):
        random.seed(1)
        model = Transformer(64, d_model=256, n_head=2, n_layer=1, block_size=16)
        head = model.blocks[0].attn.heads[0]
        x = model.blocks[0].ln1(model.tok_emb(list(range(16))) + model.pos_emb(list(range(16))))
        q, k = head.Wq(x).data, head.Wk(x).data
        scores = q @ k.T / math.sqrt(head.head_dim)
        self.assertLess(scores.std(), 3.0)


class TestBatching(unittest.TestCase):
    def _model(self):
        random.seed(3)
        return Transformer(40, d_model=16, n_head=2, n_layer=2, block_size=8)

    def test_batched_loss_equals_mean_of_rows(self):
        model = self._model()
        batch = [[1, 5, 9, 2, 7, 3, 0, 4], [8, 8, 1, 2, 3, 9, 6, 5], [0, 1, 2, 3, 4, 5, 6, 7]]
        batched = float(model.loss(batch).data)
        rows = [float(model.loss(r).data) for r in batch]
        self.assertAlmostEqual(batched, sum(rows) / len(rows), places=10)

    def test_batched_grads_equal_mean_of_row_grads(self):
        model = self._model()
        batch = [[1, 5, 9, 2, 7, 3, 0, 4], [8, 8, 1, 2, 3, 9, 6, 5]]
        model.zero_grad()
        model.loss(batch).backward()
        batched = [p.grad.copy() for p in model.parameters()]
        acc = [np.zeros_like(p.data) for p in model.parameters()]
        for r in batch:
            model.zero_grad()
            model.loss(r).backward()
            for a, p in zip(acc, model.parameters()):
                a += p.grad / len(batch)
        for b, a in zip(batched, acc):
            np.testing.assert_allclose(b, a, atol=1e-10)

    def test_batched_logits_match_unbatched(self):
        model = self._model()
        batch = [[1, 5, 9, 2], [8, 8, 1, 2]]
        out = model(batch, training=False).data
        for i, r in enumerate(batch):
            np.testing.assert_allclose(out[i], model(r, training=False).data, atol=1e-12)


class TestFloat32(unittest.TestCase):
    def tearDown(self):
        engine.set_dtype(np.float64)

    def test_float32_model_stays_float32(self):
        engine.set_dtype(np.float32)
        random.seed(3)
        model = Transformer(40, d_model=16, n_head=2, n_layer=1, block_size=8)
        loss = model.loss([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
        loss.backward()
        self.assertEqual(loss.data.dtype, np.float32)
        for p in model.parameters():
            self.assertEqual(p.data.dtype, np.float32)
            self.assertEqual(p.grad.dtype, np.float32)

    def test_loss_finite_for_confident_model(self):
        """log(softmax) underflows to -inf in float32 when the target is
        very unlikely; log-softmax must stay finite."""
        engine.set_dtype(np.float32)
        random.seed(3)
        model = Transformer(40, d_model=16, n_head=2, n_layer=1, block_size=8)
        model.lm_head.weight.data *= 200.0
        loss = float(model.loss([1, 2, 3, 4, 5, 6]).data)
        self.assertTrue(math.isfinite(loss), loss)


class TestSampler(unittest.TestCase):
    def test_sampled_ids_in_range_and_deterministic(self):
        random.seed(3)
        model = Transformer(40, d_model=16, n_head=2, n_layer=1, block_size=8)
        random.seed(11)
        a = model.generate([1, 2, 3], 30, temperature=1.0)
        random.seed(11)
        b = model.generate([1, 2, 3], 30, temperature=1.0)
        self.assertEqual(a, b)
        self.assertTrue(all(0 <= i < 40 for i in a))


if __name__ == "__main__":
    unittest.main()
