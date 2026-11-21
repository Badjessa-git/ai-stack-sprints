import os
import sys
import threading
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.batch_engine import BatchingEngine, batched_decode  # noqa: E402
from src.deps import sprint  # noqa: E402
from src.paged_kv import PagedKVCache, SequenceView  # noqa: E402

g4 = sprint("sprint-04-gpt2").gpt2
s5 = sprint("sprint-05-kv-cache")
s6 = sprint("sprint-06-inference-server")
CFG = g4.GPT2Config(vocab_size=40, n_ctx=32, n_embd=12, n_head=3, n_layer=2)
PROMPTS = [[1, 2, 3], [7, 7, 7, 7, 7, 1], [9, 4], [5, 6, 7, 8]]


class FakeTok:
    eot_id = 39

    def encode(self, t):
        return [ord(c) % 39 for c in t]

    def token_bytes(self, i):
        return bytes([97 + i % 26])


def model():
    return g4.GPT2(CFG, g4.random_params(CFG, seed=0, scale=0.3))


def req(**kw):
    kw.setdefault("temperature", 0.0)
    kw.setdefault("max_tokens", 12)
    return s6.protocol.CompletionRequest(prompt="", **kw)


def sequential(m, ids, max_tokens, eot=39):
    return list(s5.sampling.generate(m, ids, min(max_tokens, CFG.n_ctx - len(ids)),
                                     temperature=0, stop_ids={eot}))


class TestBatchedDecode(unittest.TestCase):
    def test_matches_one_at_a_time(self):
        m = model()
        cache = PagedKVCache.for_model(m, num_blocks=20, block_size=4)
        views, last = [], []
        for i, p in enumerate(PROMPTS):
            v = SequenceView(cache, i, CFG.n_ctx)
            s5.kv_cache.forward_cached(m, p[:-1], v)
            views.append(v)
            last.append(p[-1])
        logits = batched_decode(m, last, views)
        self.assertEqual(logits.shape, (4, 40))
        self.assertEqual(logits.dtype, np.float32)
        for i, p in enumerate(PROMPTS):
            np.testing.assert_allclose(logits[i], m.forward(p)[-1], atol=1e-4, err_msg=f"sequence {i}")
            self.assertEqual(views[i].length, len(p))


class TestEngine(unittest.TestCase):
    def run_all(self, eng, reqs):
        out = [None] * len(reqs)

        def one(i):
            out[i] = list(eng.generate(*reqs[i]))

        ts = [threading.Thread(target=one, args=(i,)) for i in range(len(reqs))]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=60)
        return out

    def test_concurrent_greedy_equals_sequential(self):
        m = model()
        eng = BatchingEngine(m, FakeTok(), max_batch=4, block_size=4)
        got = self.run_all(eng, [(p, req()) for p in PROMPTS])
        eng.close()
        for p, g in zip(PROMPTS, got):
            self.assertEqual(g, sequential(m, p, 12))

    def test_requests_were_actually_batched(self):
        m = model()
        eng = BatchingEngine(m, FakeTok(), max_batch=4, block_size=4, start=False)
        gens = [eng.generate(p, req(max_tokens=8)) for p in PROMPTS]
        firsts = []
        # submit all four before the engine runs, by pulling one token each
        # on separate threads, then drive step() by hand
        ts = [threading.Thread(target=lambda g=g: firsts.append(next(g))) for g in gens]
        for t in ts:
            t.start()
        while len(eng.waiting) < 4:
            pass
        eng.step()
        for t in ts:
            t.join(timeout=10)
        self.assertEqual(eng.stats["max_running"], 4)

    def test_max_batch_respected(self):
        m = model()
        eng = BatchingEngine(m, FakeTok(), max_batch=2, block_size=4)
        self.run_all(eng, [(p, req()) for p in PROMPTS])
        eng.close()
        self.assertLessEqual(eng.stats["max_running"], 2)

    def test_max_tokens_and_eot(self):
        m = model()
        eng = BatchingEngine(m, FakeTok(), max_batch=4, block_size=4)
        short = self.run_all(eng, [(PROMPTS[0], req(max_tokens=3))])[0]
        self.assertEqual(len(short), 3)
        eng.close()

    def test_memory_is_returned(self):
        m = model()
        eng = BatchingEngine(m, FakeTok(), max_batch=4, block_size=4)
        self.run_all(eng, [(p, req()) for p in PROMPTS])
        eng.close()
        self.assertEqual(eng.cache.allocator.num_free, eng.cache.allocator.num_blocks)

    def test_sampled_with_seed_is_reproducible(self):
        m = model()
        eng = BatchingEngine(m, FakeTok(), max_batch=4, block_size=4)
        a = self.run_all(eng, [(p, req(temperature=1.0, seed=5)) for p in PROMPTS])
        b = self.run_all(eng, [(p, req(temperature=1.0, seed=5)) for p in PROMPTS])
        eng.close()
        self.assertEqual(a, b)


class TestPreemption(unittest.TestCase):
    def test_runs_out_of_blocks_and_recovers(self):
        # All 4 queued before the first step, so all 4 are admitted (5
        # blocks of prompt). Grown to prompt + 12 tokens they need 17 blocks
        # of 4; the engine has 10, so some must be preempted and recomputed.
        m = model()
        eng = BatchingEngine(m, FakeTok(), max_batch=4, num_blocks=10, block_size=4, start=False)
        got = [None] * 4

        def one(i):
            got[i] = list(eng.generate(PROMPTS[i], req()))

        ts = [threading.Thread(target=one, args=(i,)) for i in range(4)]
        for t in ts:
            t.start()
        while len(eng.waiting) < 4:
            pass
        while eng.waiting or eng.running:
            eng.step()
        for t in ts:
            t.join(timeout=10)
        self.assertGreater(eng.stats["preemptions"], 0)
        for p, g in zip(PROMPTS, got):
            self.assertEqual(g, sequential(m, p, 12), "preemption must not change the output")
        self.assertEqual(eng.cache.allocator.num_free, 10)


if __name__ == "__main__":
    unittest.main()
