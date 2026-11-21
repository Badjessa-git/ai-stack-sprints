import os
import sys
import tempfile
import unittest
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deps import sprint  # noqa: E402
from src.llama import (  # noqa: E402
    Llama, LlamaConfig, apply_rope, attention, expected_shapes, map_hf_names,
    random_params, repeat_kv, rms_norm, rope_tables, silu, swiglu,
)

TINY = LlamaConfig(vocab_size=30, dim=16, n_layer=2, n_head=4, n_kv_head=2, hidden=24, max_pos=32)
RNG = np.random.default_rng(0)


class TestOps(unittest.TestCase):
    def test_rms_norm(self):
        x = np.array([[3.0, 4.0]])  # mean square 12.5
        np.testing.assert_allclose(rms_norm(x, np.ones(2), eps=0), x / np.sqrt(12.5))
        y = rms_norm(RNG.standard_normal((5, 16)) * 7, np.ones(16))
        np.testing.assert_allclose((y ** 2).mean(axis=-1), 1.0, atol=1e-4)

    def test_rms_norm_does_not_center(self):
        y = rms_norm(np.array([[1.0, 1.0, 1.0]]), np.ones(3), eps=0)
        np.testing.assert_allclose(y, [[1.0, 1.0, 1.0]])  # LayerNorm would give 0s

    def test_silu(self):
        np.testing.assert_allclose(silu(np.array([0.0, 1.0, -1.0])), [0.0, 0.7310586, -0.2689414], atol=1e-6)

    def test_swiglu_layout(self):
        x = RNG.standard_normal((3, 4))
        g, u, d = RNG.standard_normal((6, 4)), RNG.standard_normal((6, 4)), RNG.standard_normal((4, 6))
        expected = (silu(x @ g.T) * (x @ u.T)) @ d.T
        np.testing.assert_allclose(swiglu(x, g, u, d), expected)


class TestRope(unittest.TestCase):
    def setUp(self):
        self.cos, self.sin = rope_tables(8, 32)

    def test_tables(self):
        self.assertEqual(self.cos.shape, (32, 8))
        self.assertEqual(self.cos.dtype, np.float32)
        np.testing.assert_allclose(self.cos[0], 1.0)
        np.testing.assert_allclose(self.sin[0], 0.0)
        # inv_freq[0] = 1: dimension 0 turns 1 radian per position
        np.testing.assert_allclose(self.cos[3, 0], np.cos(3.0), atol=1e-6)
        # "rotate half" layout: dimension i and i + D/2 share an angle
        np.testing.assert_allclose(self.cos[:, :4], self.cos[:, 4:])

    def test_pairs_i_with_i_plus_half(self):
        x = np.zeros((1, 1, 8), np.float32)
        x[0, 0, 0] = 1.0
        out = apply_rope(x, self.cos, self.sin, [2])[0, 0]
        expected = np.zeros(8)
        expected[0], expected[4] = np.cos(2.0), np.sin(2.0)
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_position_zero_is_identity(self):
        x = RNG.standard_normal((2, 1, 8)).astype(np.float32)
        np.testing.assert_allclose(apply_rope(x, self.cos, self.sin, [0]), x, atol=1e-6)

    def test_rotation_keeps_length(self):
        x = RNG.standard_normal((2, 5, 8)).astype(np.float32)
        y = apply_rope(x, self.cos, self.sin, np.arange(5) + 7)
        np.testing.assert_allclose(np.linalg.norm(y, axis=-1), np.linalg.norm(x, axis=-1), atol=1e-5)

    def test_only_relative_position_matters(self):
        q = RNG.standard_normal((1, 1, 8)).astype(np.float32)
        k = RNG.standard_normal((1, 1, 8)).astype(np.float32)
        dot = lambda m, n: float((apply_rope(q, self.cos, self.sin, [m]) * apply_rope(k, self.cos, self.sin, [n])).sum())
        self.assertAlmostEqual(dot(5, 2), dot(15, 12), places=4)
        self.assertNotAlmostEqual(dot(5, 2), dot(5, 4), places=2)


class TestAttention(unittest.TestCase):
    def setUp(self):
        self.p = random_params(TINY, seed=1, scale=0.3)
        self.cos, self.sin = rope_tables(TINY.head_dim, TINY.max_pos)
        self.x = RNG.standard_normal((6, 16)).astype(np.float32)

    def test_repeat_kv(self):
        x = np.arange(2 * 1 * 3).reshape(2, 1, 3)
        r = repeat_kv(x, 3)
        self.assertEqual(r.shape, (6, 1, 3))
        np.testing.assert_array_equal(r[:3], np.repeat(x[:1], 3, axis=0))  # heads 0-2 share kv head 0
        self.assertIs(repeat_kv(x, 1), x)

    def test_gqa_equals_mha_with_shared_weights(self):
        # 4 query heads sharing 2 kv heads == 4 kv heads where each pair
        # is a copy: grouping is just weight sharing.
        mha_cfg = LlamaConfig(**{**TINY.__dict__, "n_kv_head": 4})
        mha = dict(self.p)
        D = TINY.head_dim
        for w in ("wk", "wv"):
            heads = self.p[f"layers.0.{w}"].reshape(2, D, 16)
            mha[f"layers.0.{w}"] = np.repeat(heads, 2, axis=0).reshape(4 * D, 16)
        a = attention(self.x, self.p, 0, TINY, self.cos, self.sin)
        b = attention(self.x, mha, 0, mha_cfg, self.cos, self.sin)
        np.testing.assert_allclose(a, b, atol=1e-5)

    def test_causal(self):
        a = attention(self.x, self.p, 0, TINY, self.cos, self.sin)
        x2 = self.x.copy()
        x2[4:] += 1.0
        b = attention(x2, self.p, 0, TINY, self.cos, self.sin)
        np.testing.assert_allclose(a[:4], b[:4], atol=1e-5)

    def test_cache_stores_kv_heads_only(self):
        cache = Llama(TINY, self.p).new_cache()
        self.assertEqual(cache.k.shape[1], TINY.n_kv_head)
        full = attention(self.x, self.p, 0, TINY, self.cos, self.sin)
        part = attention(self.x[:4], self.p, 0, TINY, self.cos, self.sin, cache)
        cache.advance(4)
        rest = attention(self.x[4:], self.p, 0, TINY, self.cos, self.sin, cache)
        np.testing.assert_allclose(np.concatenate([part, rest]), full, atol=1e-5)


class TestLoading(unittest.TestCase):
    def hf(self):
        inv = {v: k for k, v in __import__("src.llama", fromlist=["HF_NAMES"]).HF_NAMES.items()}
        raw = {}
        for k, v in random_params(TINY).items():
            if k.startswith("layers."):
                _, i, ours = k.split(".", 2)
                raw[f"model.layers.{i}.{inv[ours]}"] = v
            else:
                raw["model." + inv[k]] = v
        raw["model.layers.0.self_attn.rotary_emb.inv_freq"] = np.ones(2, np.float32)
        raw["lm_head.weight"] = raw["model.embed_tokens.weight"]
        return raw

    def test_maps(self):
        p = map_hf_names(self.hf(), TINY)
        self.assertEqual(set(p), set(expected_shapes(TINY)))

    def test_untied_keeps_lm_head(self):
        cfg = LlamaConfig(**{**TINY.__dict__, "tied": False})
        self.assertIn("lm_head", map_hf_names(self.hf(), cfg))

    def test_strict(self):
        raw = self.hf()
        del raw["model.layers.1.mlp.up_proj.weight"]
        with self.assertRaises(KeyError):
            map_hf_names(raw, TINY)
        raw = self.hf()
        raw["model.layers.0.self_attn.k_proj.weight"] = np.zeros((16, 16), np.float32)
        with self.assertRaises(ValueError):
            map_hf_names(raw, TINY)  # GQA: k_proj is (n_kv_head * D, dim) = (8, 16)
        raw = self.hf()
        raw["model.layers.0.self_attn.q_norm.weight"] = np.zeros(4, np.float32)
        with self.assertRaises(ValueError):
            map_hf_names(raw, TINY)

    def test_through_a_file(self):
        save = sprint("sprint-04-gpt2").safetensors.save
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.safetensors")
            save(path, self.hf(), bf16={"model.norm.weight"})
            p = map_hf_names(sprint("sprint-04-gpt2").safetensors.load(path), TINY)
        np.testing.assert_allclose(p["norm"], 1.0)


class TestModel(unittest.TestCase):
    def setUp(self):
        self.m = Llama(TINY, random_params(TINY, seed=2, scale=0.3))
        self.ids = [3, 1, 4, 1, 5, 9, 2, 6]

    def test_shape_and_dtype(self):
        out = self.m.forward(self.ids)
        self.assertEqual(out.shape, (8, 30))
        self.assertEqual(out.dtype, np.float32)

    def test_causal(self):
        a, b = self.m.forward([1, 2, 3, 4]), self.m.forward([1, 2, 3, 7])
        np.testing.assert_allclose(a[:3], b[:3], atol=1e-5)

    def test_rope_has_no_absolute_position(self):
        # The same token three times gives the same output three times:
        # every position attends over identical values, and RoPE only
        # encodes *relative* offsets. (GPT-2's wpe table would make these
        # rows differ.) Order still matters once the tokens differ:
        out = self.m.forward([5, 5, 5])
        np.testing.assert_allclose(out[0], out[2], atol=1e-5)
        self.assertFalse(np.allclose(self.m.forward([5, 6])[-1], self.m.forward([6, 5])[-1]))

    def test_cached_equals_full(self):
        full = self.m.forward(self.ids)
        c = self.m.new_cache()
        a = self.m.forward(self.ids[:5], c)
        b = [self.m.forward([t], c)[0] for t in self.ids[5:]]
        np.testing.assert_allclose(np.concatenate([a, np.stack(b)]), full, atol=1e-4)
        self.assertEqual(c.length, 8)

    def test_generate_runs(self):
        self.assertEqual(len(self.m.generate([1, 2], 5)), 5)


if __name__ == "__main__":
    unittest.main()
