import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deps import sprint  # noqa: E402
from src.paged_kv import BlockAllocator, OutOfBlocks, PagedKVCache, SequenceView  # noqa: E402

RNG = np.random.default_rng(0)


class TestAllocator(unittest.TestCase):
    def test_alloc_free(self):
        a = BlockAllocator(3)
        got = {a.alloc(), a.alloc(), a.alloc()}
        self.assertEqual(got, {0, 1, 2})
        self.assertEqual(a.num_free, 0)
        with self.assertRaises(OutOfBlocks):
            a.alloc()
        a.free(1)
        self.assertEqual((a.num_free, a.alloc()), (1, 1))

    def test_double_free_raises(self):
        a = BlockAllocator(2)
        b = a.alloc()
        a.free(b)
        with self.assertRaises(ValueError):
            a.free(b)
        with self.assertRaises(ValueError):
            a.free(1)  # never allocated


def kv(T, H=2, D=3):
    return RNG.standard_normal((H, T, D)).astype(np.float32), RNG.standard_normal((H, T, D)).astype(np.float32)


class TestPagedCache(unittest.TestCase):
    def setUp(self):
        self.c = PagedKVCache(n_layer=2, n_head=2, head_dim=3, num_blocks=6, block_size=4)
        self.c.add_sequence("a")

    def write(self, sid, T, layer=0):
        k, v = kv(T)
        self.c.reserve(sid, T)
        out = self.c.update(sid, layer, k, v)
        return k, v, out

    def test_reserve_counts_blocks(self):
        self.c.reserve("a", 1)
        self.assertEqual(len(self.c.tables["a"]), 1)
        self.c.reserve("a", 4)
        self.assertEqual(len(self.c.tables["a"]), 1, "4 positions fit in one block")
        self.c.reserve("a", 9)
        self.assertEqual(len(self.c.tables["a"]), 3)

    def test_update_returns_contiguous_across_blocks(self):
        k1, v1, _ = self.write("a", 6)  # spans 2 blocks
        self.c.advance("a", 6)
        k2, v2, (ka, va) = self.write("a", 3)  # 6..8, crosses into block 3
        self.assertEqual(ka.shape, (2, 9, 3))
        np.testing.assert_array_equal(ka, np.concatenate([k1, k2], axis=1))
        np.testing.assert_array_equal(va, np.concatenate([v1, v2], axis=1))

    def test_sequences_dont_clobber_each_other(self):
        self.c.add_sequence("b")
        ka1, _, _ = self.write("a", 5)
        self.c.advance("a", 5)
        kb1, _, _ = self.write("b", 5)
        self.c.advance("b", 5)
        ka2, _, (ka, _) = self.write("a", 1)
        np.testing.assert_array_equal(ka, np.concatenate([ka1, ka2], axis=1))
        self.assertTrue(set(self.c.tables["a"]).isdisjoint(self.c.tables["b"]))

    def test_layers_are_separate(self):
        k0, _, _ = self.write("a", 2, layer=0)
        k1, _ = self.c.update("a", 1, *kv(2))
        self.assertFalse(np.array_equal(k0, k1))

    def test_reserve_is_all_or_nothing(self):
        self.c.reserve("a", 16)  # 4 of 6 blocks
        self.c.add_sequence("b")
        with self.assertRaises(OutOfBlocks):
            self.c.reserve("b", 12)  # needs 3, only 2 free
        self.assertEqual(self.c.tables["b"], [])
        self.assertEqual(self.c.allocator.num_free, 2)

    def test_free_sequence(self):
        self.write("a", 9)
        self.c.free_sequence("a")
        self.assertEqual(self.c.allocator.num_free, 6)
        self.assertNotIn("a", self.c.tables)


g4 = sprint("sprint-04-gpt2").gpt2
s5 = sprint("sprint-05-kv-cache")
CFG = g4.GPT2Config(vocab_size=40, n_ctx=32, n_embd=12, n_head=3, n_layer=2)


class TestSequenceView(unittest.TestCase):
    def setUp(self):
        self.m = g4.GPT2(CFG, g4.random_params(CFG, seed=0, scale=0.3))
        self.cache = PagedKVCache.for_model(self.m, num_blocks=10, block_size=4)
        self.ids = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]

    def test_sprint5_forward_cached_runs_on_pages(self):
        full = self.m.forward(self.ids)
        view = SequenceView(self.cache, 0, CFG.n_ctx)
        a = s5.kv_cache.forward_cached(self.m, self.ids[:6], view)
        b = [s5.kv_cache.forward_cached(self.m, [t], view)[0] for t in self.ids[6:]]
        np.testing.assert_allclose(np.concatenate([a, np.stack(b)]), full, atol=1e-4)
        self.assertEqual(view.length, 10)
        self.assertEqual(len(self.cache.tables[0]), 3)  # ceil(10 / 4)

    def test_max_len(self):
        view = SequenceView(self.cache, 1, max_len=4)
        s5.kv_cache.forward_cached(self.m, [1, 2, 3, 4], view)
        with self.assertRaises(ValueError):
            s5.kv_cache.forward_cached(self.m, [5], view)


if __name__ == "__main__":
    unittest.main()
