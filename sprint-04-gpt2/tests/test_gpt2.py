"""The model, on a tiny random GPT-2 (no download needed)."""

import os
import sys
import tempfile
import unittest
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)  # NumPy 2 + Accelerate noise
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gpt2 import (  # noqa: E402
    GPT2,
    GPT2Config,
    attention,
    block,
    expected_shapes,
    gelu,
    layer_norm,
    map_hf_names,
    mlp,
    random_params,
    softmax,
)
from src.safetensors import load, save  # noqa: E402

TINY = GPT2Config(vocab_size=50, n_ctx=16, n_embd=12, n_head=3, n_layer=2)
RNG = np.random.default_rng(0)


class TestOps(unittest.TestCase):
    def test_layer_norm_normalizes(self):
        x = RNG.standard_normal((4, 12)) * 5 + 3
        y = layer_norm(x, np.ones(12), np.zeros(12))
        np.testing.assert_allclose(y.mean(axis=-1), 0, atol=1e-6)
        np.testing.assert_allclose(y.var(axis=-1), 1, atol=1e-4)

    def test_layer_norm_scale_shift(self):
        x = np.array([[1.0, 3.0]])  # mean 2, var 1
        y = layer_norm(x, np.array([2.0, 2.0]), np.array([10.0, 0.0]), eps=0)
        np.testing.assert_allclose(y, [[8.0, 2.0]])

    def test_gelu_values(self):
        np.testing.assert_allclose(
            gelu(np.array([-3.0, -1.0, 0.0, 1.0, 3.0])),
            [-0.0036373, -0.1588080, 0.0, 0.8411920, 2.9963627],
            atol=1e-6,
        )

    def test_softmax(self):
        s = softmax(np.array([[1000.0, 1000.0, -np.inf]]))
        np.testing.assert_allclose(s, [[0.5, 0.5, 0.0]])


def attn_args(C=12, seed=1):
    r = np.random.default_rng(seed)
    return (r.standard_normal((C, 3 * C)) * 0.3, r.standard_normal(3 * C) * 0.1,
            r.standard_normal((C, C)) * 0.3, r.standard_normal(C) * 0.1)


class TestAttention(unittest.TestCase):
    def test_shape(self):
        x = RNG.standard_normal((5, 12))
        self.assertEqual(attention(x, *attn_args(), n_head=3).shape, (5, 12))

    def test_one_head_by_hand(self):
        C, T = 4, 3
        x = RNG.standard_normal((T, C))
        wqkv, bqkv, wo, bo = attn_args(C)
        q, k, v = (x @ wqkv + bqkv)[:, :C], (x @ wqkv + bqkv)[:, C:2 * C], (x @ wqkv + bqkv)[:, 2 * C:]
        out = np.zeros((T, C))
        for t in range(T):
            s = np.array([q[t] @ k[u] / np.sqrt(C) for u in range(t + 1)])
            w = np.exp(s - s.max()) / np.exp(s - s.max()).sum()
            out[t] = sum(w[u] * v[u] for u in range(t + 1))
        np.testing.assert_allclose(attention(x, wqkv, bqkv, wo, bo, 1), out @ wo + bo, atol=1e-10)

    def test_heads_are_independent_slices(self):
        # With 3 heads, head h only uses columns h*D:(h+1)*D of q, k and v.
        C, H, T = 12, 3, 4
        D = C // H
        x = RNG.standard_normal((T, C))
        wqkv, bqkv, _, _ = attn_args(C)
        eye, zero = np.eye(C), np.zeros(C)
        full = attention(x, wqkv, bqkv, eye, zero, H)
        for h in range(H):
            cols = np.r_[h * D:(h + 1) * D]
            sub_w = np.concatenate([wqkv[:, cols], wqkv[:, C + cols], wqkv[:, 2 * C + cols]], axis=1)
            sub_b = np.concatenate([bqkv[cols], bqkv[C + cols], bqkv[2 * C + cols]])
            one = attention(x, sub_w, sub_b, np.eye(D), np.zeros(D), 1)
            # attention's 1/sqrt(D) scale uses the head size, so the same
            # numbers come out of the single-head call
            np.testing.assert_allclose(full[:, cols], one, atol=1e-10)

    def test_causal(self):
        x = RNG.standard_normal((6, 12))
        a = attention(x, *attn_args(), n_head=3)
        x2 = x.copy()
        x2[4:] = RNG.standard_normal((2, 12))
        b = attention(x2, *attn_args(), n_head=3)
        np.testing.assert_allclose(a[:4], b[:4], atol=1e-10)
        self.assertFalse(np.allclose(a[4:], b[4:]))

    def test_first_position_sees_only_itself(self):
        C = 12
        x = RNG.standard_normal((3, C))
        wqkv, bqkv, wo, bo = attn_args(C)
        v0 = (x @ wqkv + bqkv)[0, 2 * C:]
        np.testing.assert_allclose(attention(x, wqkv, bqkv, wo, bo, 3)[0], v0 @ wo + bo, atol=1e-10)


class TestBlock(unittest.TestCase):
    def test_mlp_shape_and_value(self):
        x = RNG.standard_normal((3, 4))
        w1, b1 = RNG.standard_normal((4, 16)), RNG.standard_normal(16)
        w2, b2 = RNG.standard_normal((16, 4)), RNG.standard_normal(4)
        np.testing.assert_allclose(mlp(x, w1, b1, w2, b2), gelu(x @ w1 + b1) @ w2 + b2)

    def test_zero_weights_is_identity(self):
        # All linear weights and biases zero: both branches output 0, so
        # the residuals pass x through untouched. A missing or misplaced
        # residual fails here.
        p = {k: np.zeros(s, np.float32) for k, s in expected_shapes(TINY).items()}
        for k in p:
            if k.endswith(("ln_1.weight", "ln_2.weight")):
                p[k] = np.ones_like(p[k])
        x = RNG.standard_normal((5, 12)).astype(np.float32)
        np.testing.assert_allclose(block(x, p, 0, 3), x, atol=1e-6)

    def test_block_uses_its_own_layer(self):
        p = random_params(TINY, seed=3)
        x = RNG.standard_normal((4, 12)).astype(np.float32)
        self.assertFalse(np.allclose(block(x, p, 0, 3), block(x, p, 1, 3)))


class TestLoading(unittest.TestCase):
    def hf_style(self, prefix="", extras=True):
        raw = {prefix + k: v for k, v in random_params(TINY).items()}
        if extras:
            for i in range(TINY.n_layer):
                raw[f"{prefix}h.{i}.attn.bias"] = np.tril(np.ones((1, 1, 16, 16), np.float32))
                raw[f"{prefix}h.{i}.attn.masked_bias"] = np.array(-1e4, np.float32)
            raw["lm_head.weight"] = raw[prefix + "wte.weight"]
        return raw

    def test_maps_plain_names(self):
        p = map_hf_names(self.hf_style(), TINY)
        self.assertEqual(set(p), set(expected_shapes(TINY)))
        self.assertTrue(all(v.dtype == np.float32 for v in p.values()))

    def test_strips_prefix(self):
        p = map_hf_names(self.hf_style("transformer."), TINY)
        self.assertIn("h.1.mlp.c_proj.bias", p)

    def test_missing_raises(self):
        raw = self.hf_style()
        del raw["h.1.ln_2.bias"]
        with self.assertRaises(KeyError) as e:
            map_hf_names(raw, TINY)
        self.assertIn("h.1.ln_2.bias", str(e.exception))

    def test_wrong_shape_raises(self):
        raw = self.hf_style()
        raw["h.0.attn.c_attn.weight"] = raw["h.0.attn.c_attn.weight"].T  # (3C, C)
        with self.assertRaises(ValueError):
            map_hf_names(raw, TINY)

    def test_unknown_name_raises(self):
        raw = self.hf_style()
        raw["h.0.attn.rotary"] = np.zeros(3, np.float32)
        with self.assertRaises(ValueError):
            map_hf_names(raw, TINY)

    def test_through_a_real_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.safetensors")
            save(path, self.hf_style())
            p = map_hf_names(load(path), TINY)
        np.testing.assert_array_equal(p["wpe.weight"], random_params(TINY)["wpe.weight"])


class TestModel(unittest.TestCase):
    def setUp(self):
        self.m = GPT2(TINY, random_params(TINY, seed=0, scale=0.3))

    def test_shape(self):
        self.assertEqual(self.m.forward([1, 2, 3]).shape, (3, 50))

    def test_causal(self):
        a = self.m.forward([5, 6, 7, 8])
        b = self.m.forward([5, 6, 7, 9])
        np.testing.assert_allclose(a[:3], b[:3], atol=1e-5)

    def test_position_matters(self):
        a = self.m.forward([5, 5])
        self.assertFalse(np.allclose(a[0], a[1]), "wpe not added?")

    def test_tied_head(self):
        # With every block zeroed out, logits = ln_f(wte[id] + wpe[0]) @ wteᵀ
        p = {k: np.zeros(s, np.float32) for k, s in expected_shapes(TINY).items()}
        base = random_params(TINY, seed=4, scale=0.5)
        for k in ("wte.weight", "wpe.weight"):
            p[k] = base[k]
        for k in p:
            if k.endswith(("ln_1.weight", "ln_2.weight", "ln_f.weight")):
                p[k] = np.ones_like(p[k])
        logits = GPT2(TINY, p).forward([7])
        x = layer_norm(p["wte.weight"][7] + p["wpe.weight"][0], 1.0, 0.0)
        np.testing.assert_allclose(logits[0], x @ p["wte.weight"].T, atol=1e-4)

    def test_stays_float32(self):
        # float32 weights in, float32 logits out. A float64 here means some
        # NumPy scalar (np.sqrt, np.float64(...)) promoted the activations:
        # same numbers, many times slower on the real model.
        self.assertEqual(self.m.forward([1, 2, 3]).dtype, np.float32)

    def test_too_long_raises(self):
        with self.assertRaises(AssertionError):
            self.m.forward(list(range(17)))

    def test_generate_greedy_runs(self):
        out = self.m.generate_greedy([1, 2], 5)
        self.assertEqual(len(out), 7)
        self.assertEqual(out[:2], [1, 2])


if __name__ == "__main__":
    unittest.main()
