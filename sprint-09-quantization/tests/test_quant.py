import os
import sys
import unittest
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deps import sprint  # noqa: E402
from src.quant import (  # noqa: E402
    QuantizedParams, dequantize_int4, dequantize_int8, pack_int4,
    quantize_int4, quantize_int8, unpack_int4,
)

RNG = np.random.default_rng(0)


class TestInt8(unittest.TestCase):
    def test_shapes_and_types(self):
        w = RNG.standard_normal((8, 5)).astype(np.float32)
        q, s = quantize_int8(w)
        self.assertEqual((q.dtype, q.shape), (np.int8, (8, 5)))
        self.assertEqual((s.dtype, s.shape), (np.float32, (1, 5)))

    def test_error_at_most_half_a_step(self):
        w = RNG.standard_normal((64, 16)).astype(np.float32)
        q, s = quantize_int8(w)
        self.assertTrue((np.abs(dequantize_int8(q, s) - w) <= s / 2 + 1e-7).all())

    def test_max_maps_to_127(self):
        w = np.array([[1.0, -4.0], [-0.5, 2.0]], np.float32)
        q, _ = quantize_int8(w)
        np.testing.assert_array_equal(q, [[127, -127], [-64, 64]])

    def test_zero_column(self):
        w = np.zeros((4, 2), np.float32)
        w[:, 1] = 1.0
        q, s = quantize_int8(w)
        self.assertTrue(np.isfinite(s).all())
        np.testing.assert_array_equal(dequantize_int8(q, s)[:, 0], 0.0)

    def test_per_channel_beats_per_tensor_with_an_outlier(self):
        w = RNG.standard_normal((64, 8)).astype(np.float32) * 0.02
        w[:, 3] *= 100  # one loud column
        q, s = quantize_int8(w, axis=0)
        err_channel = np.abs(dequantize_int8(q, s) - w)[:, [0, 1, 2]].mean()
        qt, st = quantize_int8(w, axis=None)  # one scale for everything
        err_tensor = np.abs(dequantize_int8(qt, st) - w)[:, [0, 1, 2]].mean()
        self.assertLess(err_channel * 10, err_tensor)


class TestPack(unittest.TestCase):
    def test_roundtrip_every_value(self):
        q = np.arange(-8, 8, dtype=np.int8).reshape(2, 8)
        p = pack_int4(q)
        self.assertEqual((p.dtype, p.shape), (np.uint8, (2, 4)))
        np.testing.assert_array_equal(unpack_int4(p), q)

    def test_layout(self):
        # -8 -> 0 (low nibble), 7 -> 15 (high nibble): 0xF0
        self.assertEqual(int(pack_int4(np.array([-8, 7], np.int8))[0]), 0xF0)
        self.assertEqual(int(pack_int4(np.array([1, 0], np.int8))[0]), 0x89)

    def test_odd_length_raises(self):
        with self.assertRaises(ValueError):
            pack_int4(np.zeros(3, np.int8))


class TestInt4(unittest.TestCase):
    def test_shapes(self):
        w = RNG.standard_normal((64, 6)).astype(np.float32)
        p, s = quantize_int4(w, group_size=32)
        self.assertEqual((p.dtype, p.shape), (np.uint8, (6, 32)))
        self.assertEqual(s.shape, (2, 1, 6))
        self.assertEqual(dequantize_int4(p, s, 32).shape, (64, 6))

    def test_error_at_most_half_a_step(self):
        w = RNG.standard_normal((64, 6)).astype(np.float32)
        p, s = quantize_int4(w, group_size=16)
        err = np.abs(dequantize_int4(p, s, 16) - w).reshape(4, 16, 6)
        self.assertTrue((err <= s / 2 + 1e-6).all())

    def test_smaller_groups_are_more_accurate(self):
        w = RNG.standard_normal((256, 8)).astype(np.float32)
        err = lambda g: np.abs(dequantize_int4(*quantize_int4(w, g), g) - w).mean()
        self.assertLess(err(16), err(128))

    def test_bad_group_raises(self):
        with self.assertRaises(ValueError):
            quantize_int4(np.zeros((40, 2), np.float32), group_size=32)


g4 = sprint("sprint-04-gpt2").gpt2
CFG = g4.GPT2Config(vocab_size=50, n_ctx=16, n_embd=64, n_head=4, n_layer=2)


class TestQuantizedParams(unittest.TestCase):
    def setUp(self):
        self.p = g4.random_params(CFG, seed=0, scale=0.1)

    def test_only_block_matrices_are_quantized(self):
        qp = QuantizedParams.from_params(self.p, bits=8)
        self.assertEqual(len(qp.packed), 4 * CFG.n_layer)
        self.assertIn("wte.weight", dict(qp))
        self.assertIn("h.0.mlp.c_fc.weight", qp)
        self.assertEqual(qp["h.0.mlp.c_fc.weight"].dtype, np.float32)

    def test_forward_close_to_float(self):
        # Error relative to the size of the logits: int8 is nearly exact,
        # int4 visibly worse but still the same function.
        ids = [1, 2, 3, 4, 5]
        ref = g4.GPT2(CFG, self.p).forward(ids)
        rel = {}
        for bits in (8, 4):
            out = g4.GPT2(CFG, QuantizedParams.from_params(self.p, bits=bits, group_size=32)).forward(ids)
            rel[bits] = np.abs(out - ref).max() / np.abs(ref).max()
        self.assertLess(rel[8], 0.02)
        self.assertLess(rel[4], 0.5)  # random weights are a worst case; real ones do better
        self.assertLess(rel[8], rel[4])

    def test_sizes(self):
        # int8: 1 byte per weight + one float32 scale per column.
        # int4: half a byte per weight + one float32 scale per 32 weights.
        quantized = {k: v for k, v in self.p.items() if k.endswith(QuantizedParams.QUANTIZE)}
        rest = sum(v.nbytes for k, v in self.p.items() if k not in quantized)
        n = sum(v.size for v in quantized.values())
        cols = sum(v.shape[1] for v in quantized.values())
        self.assertEqual(QuantizedParams.from_params(self.p, bits=8).nbytes(), rest + n + 4 * cols)
        self.assertEqual(QuantizedParams.from_params(self.p, bits=4).nbytes(), rest + n // 2 + 4 * (n // 32))


if __name__ == "__main__":
    unittest.main()
