"""The cache must change nothing but speed: every test compares cached
logits against your Sprint 4 GPT2.forward on the same tokens."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deps import sprint  # noqa: E402
from src.kv_cache import KVCache, attention_cached, forward_cached  # noqa: E402

g4 = sprint("sprint-04-gpt2").gpt2
CFG = g4.GPT2Config(vocab_size=40, n_ctx=24, n_embd=12, n_head=3, n_layer=2)
RNG = np.random.default_rng(0)


def tiny_model(seed=0):
    return g4.GPT2(CFG, g4.random_params(CFG, seed=seed, scale=0.3))


class TestKVCache(unittest.TestCase):
    def test_update_returns_everything_so_far(self):
        c = KVCache(n_layer=2, n_head=3, head_dim=4, max_len=10)
        k1, v1 = RNG.standard_normal((3, 2, 4)), RNG.standard_normal((3, 2, 4))
        ka, va = c.update(0, k1, v1)
        np.testing.assert_allclose(ka, k1, atol=1e-6)
        self.assertEqual(c.length, 0, "update() must not advance")
        c.advance(2)
        k2, v2 = RNG.standard_normal((3, 1, 4)), RNG.standard_normal((3, 1, 4))
        ka, va = c.update(0, k2, v2)
        self.assertEqual(ka.shape, (3, 3, 4))
        np.testing.assert_allclose(ka[:, :2], k1, atol=1e-6)
        np.testing.assert_allclose(va[:, 2:], v2, atol=1e-6)

    def test_layers_are_separate(self):
        c = KVCache(2, 1, 2, 5)
        c.update(0, np.ones((1, 1, 2)), np.ones((1, 1, 2)))
        k, _ = c.update(1, np.zeros((1, 1, 2)), np.zeros((1, 1, 2)))
        np.testing.assert_array_equal(k, np.zeros((1, 1, 2)))

    def test_overflow_raises(self):
        c = KVCache(1, 1, 2, 3)
        c.advance(2)
        with self.assertRaises(ValueError):
            c.update(0, np.zeros((1, 2, 2)), np.zeros((1, 2, 2)))


class TestAttentionCached(unittest.TestCase):
    def setUp(self):
        C = 12
        self.w = (RNG.standard_normal((C, 3 * C)) * 0.3, RNG.standard_normal(3 * C) * 0.1,
                  RNG.standard_normal((C, C)) * 0.3, RNG.standard_normal(C) * 0.1)
        self.x = RNG.standard_normal((6, C))

    def test_prefill_matches_sprint4(self):
        c = KVCache(1, 3, 4, 10)
        np.testing.assert_allclose(
            attention_cached(self.x, *self.w, 3, c, 0), g4.attention(self.x, *self.w, 3), atol=1e-5
        )

    def test_one_token_at_a_time_matches(self):
        full = g4.attention(self.x, *self.w, 3)
        c = KVCache(1, 3, 4, 10)
        for t in range(6):
            row = attention_cached(self.x[t:t + 1], *self.w, 3, c, 0)
            c.advance(1)
            np.testing.assert_allclose(row[0], full[t], atol=1e-5, err_msg=f"position {t}")

    def test_chunk_after_prefix_uses_offset_mask(self):
        # positions 0-2 cached, then 3-5 in one call: row 3 must NOT see 4, 5
        full = g4.attention(self.x, *self.w, 3)
        c = KVCache(1, 3, 4, 10)
        attention_cached(self.x[:3], *self.w, 3, c, 0)
        c.advance(3)
        np.testing.assert_allclose(attention_cached(self.x[3:], *self.w, 3, c, 0), full[3:], atol=1e-5)


class TestForwardCached(unittest.TestCase):
    def setUp(self):
        self.m = tiny_model()
        self.ids = [3, 1, 4, 1, 5, 9, 2, 6]
        self.full = self.m.forward(self.ids)

    def test_prefill(self):
        c = KVCache.for_model(self.m)
        np.testing.assert_allclose(forward_cached(self.m, self.ids, c), self.full, atol=1e-4)
        self.assertEqual(c.length, 8)

    def test_token_by_token(self):
        c = KVCache.for_model(self.m)
        for t, i in enumerate(self.ids):
            np.testing.assert_allclose(forward_cached(self.m, [i], c)[0], self.full[t], atol=1e-4,
                                       err_msg=f"position {t}")

    def test_mixed_chunks(self):
        c = KVCache.for_model(self.m)
        a = forward_cached(self.m, self.ids[:5], c)
        b = forward_cached(self.m, self.ids[5:7], c)
        d = forward_cached(self.m, self.ids[7:], c)
        np.testing.assert_allclose(np.concatenate([a, b, d]), self.full, atol=1e-4)

    def test_stays_float32(self):
        c = KVCache.for_model(self.m)
        self.assertEqual(forward_cached(self.m, self.ids, c).dtype, np.float32)
        self.assertEqual(forward_cached(self.m, [1], c).dtype, np.float32)

    def test_reset_starts_over(self):
        c = KVCache.for_model(self.m)
        forward_cached(self.m, [7, 7, 7], c)
        c.reset()
        np.testing.assert_allclose(forward_cached(self.m, self.ids, c), self.full, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
