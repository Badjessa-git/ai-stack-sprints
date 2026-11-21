import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deps import sprint  # noqa: E402
from src.sampling import (  # noqa: E402
    apply_temperature,
    generate,
    sample,
    softmax,
    top_k_filter,
    top_p_filter,
)

g4 = sprint("sprint-04-gpt2").gpt2
CFG = g4.GPT2Config(vocab_size=40, n_ctx=24, n_embd=12, n_head=3, n_layer=2)
LOGITS = np.log(np.array([0.5, 0.3, 0.15, 0.05]))


class TestFilters(unittest.TestCase):
    def test_temperature(self):
        np.testing.assert_allclose(apply_temperature([2.0, 4.0], 2.0), [1.0, 2.0])
        sharp = softmax(apply_temperature(LOGITS, 0.5))
        self.assertGreater(sharp[0], 0.5)
        flat = softmax(apply_temperature(LOGITS, 5.0))
        self.assertLess(flat[0], 0.5)

    def test_top_k(self):
        out = top_k_filter(LOGITS, 2)
        self.assertTrue(np.isfinite(out[:2]).all())
        self.assertTrue(np.isneginf(out[2:]).all())
        np.testing.assert_array_equal(top_k_filter(LOGITS, 10), LOGITS)

    def test_top_k_unsorted(self):
        out = top_k_filter(np.array([0.0, 5.0, 1.0, 4.0]), 2)
        np.testing.assert_array_equal(np.isfinite(out), [False, True, False, True])

    def test_top_p(self):
        keep = lambda p: int(np.isfinite(top_p_filter(LOGITS, p)).sum())
        # probabilities 0.5, 0.3, 0.15, 0.05. Values sit off the exact
        # boundaries: in floating point 0.5 + 0.3 is 0.7999999..., not 0.8.
        self.assertEqual(keep(0.4), 1)
        self.assertEqual(keep(0.6), 2)   # 0.5 + 0.3 = 0.8 covers it
        self.assertEqual(keep(0.79), 2)
        self.assertEqual(keep(0.81), 3)
        self.assertEqual(keep(0.99), 4)
        self.assertEqual(keep(1.0), 4)
        self.assertEqual(keep(0.0), 1)   # always at least one

    def test_top_p_unsorted(self):
        x = np.log(np.array([0.05, 0.5, 0.15, 0.3]))
        np.testing.assert_array_equal(np.isfinite(top_p_filter(x, 0.79)), [False, True, False, True])

    def test_filters_dont_mutate(self):
        x = LOGITS.copy()
        top_k_filter(x, 1)
        top_p_filter(x, 0.1)
        np.testing.assert_array_equal(x, LOGITS)


class TestSample(unittest.TestCase):
    def test_zero_temperature_is_argmax(self):
        self.assertEqual(sample(np.array([1.0, 9.0, 3.0]), temperature=0), 1)

    def test_seeded(self):
        a = [sample(LOGITS, rng=np.random.default_rng(7)) for _ in range(5)]
        b = [sample(LOGITS, rng=np.random.default_rng(7)) for _ in range(5)]
        self.assertEqual(a, b)

    def test_matches_the_distribution(self):
        rng = np.random.default_rng(0)
        n = 20000
        counts = np.bincount([sample(LOGITS, rng=rng) for _ in range(n)], minlength=4)
        np.testing.assert_allclose(counts / n, [0.5, 0.3, 0.15, 0.05], atol=0.015)

    def test_top_k_one_is_greedy(self):
        rng = np.random.default_rng(0)
        self.assertTrue(all(sample(LOGITS, top_k=1, rng=rng) == 0 for _ in range(50)))

    def test_never_samples_filtered(self):
        rng = np.random.default_rng(0)
        picks = {sample(LOGITS, top_p=0.79, rng=rng) for _ in range(500)}
        self.assertEqual(picks, {0, 1})


class TestGenerate(unittest.TestCase):
    def setUp(self):
        self.m = g4.GPT2(CFG, g4.random_params(CFG, seed=0, scale=0.3))

    def test_is_a_generator(self):
        it = generate(self.m, [1, 2], 3, temperature=0)
        self.assertTrue(hasattr(it, "__next__"))
        self.assertIsInstance(next(it), int)

    def test_greedy_matches_sprint4(self):
        prompt = [1, 2, 3]
        out = list(generate(self.m, prompt, 10, temperature=0))
        self.assertEqual(prompt + out, self.m.generate_greedy(prompt, 10))

    def test_max_new(self):
        self.assertEqual(len(list(generate(self.m, [1], 7, seed=0))), 7)

    def test_same_seed_same_text(self):
        a = list(generate(self.m, [1, 2], 12, temperature=1.0, seed=3))
        b = list(generate(self.m, [1, 2], 12, temperature=1.0, seed=3))
        c = list(generate(self.m, [1, 2], 12, temperature=1.0, seed=4))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_stop_ids(self):
        greedy = list(generate(self.m, [1, 2, 3], 10, temperature=0))
        stop = greedy[4]
        out = list(generate(self.m, [1, 2, 3], 10, temperature=0, stop_ids={stop}))
        self.assertEqual(out[-1], stop)
        self.assertEqual(out, greedy[: greedy.index(stop) + 1])

    def test_stops_when_context_is_full(self):
        out = list(generate(self.m, list(range(20)), 50, temperature=0))
        self.assertLessEqual(20 + len(out), 24 + 1)


if __name__ == "__main__":
    unittest.main()
