"""Cached generation on real GPT-2 (uses Sprint 4's data/gpt2/)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deps import gpt2_dir, sprint  # noqa: E402

HAVE = os.path.exists(os.path.join(gpt2_dir(), "model.safetensors"))


@unittest.skipUnless(HAVE, "run make download in sprint-04-gpt2 first")
class TestRealCached(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s4 = sprint("sprint-04-gpt2")
        d = gpt2_dir()
        cls.tok = s4.gpt2_tokenizer.GPT2Tokenizer.from_files(
            os.path.join(d, "vocab.json"), os.path.join(d, "merges.txt"))
        cls.model = s4.gpt2.GPT2.from_dir(d)

    def test_cached_greedy_is_sprint4_greedy(self):
        from src.sampling import generate

        ids = self.tok.encode("Albert Einstein was a")
        cached = list(generate(self.model, ids, 12, temperature=0))
        self.assertEqual(ids + cached, self.model.generate_greedy(ids, 12))

    def test_stops_at_endoftext(self):
        from src.sampling import generate

        out = list(generate(self.model, self.tok.encode("Hi"), 60, temperature=1.0,
                            top_p=0.9, seed=0, stop_ids={self.tok.eot_id}))
        self.assertTrue(len(out) == 60 or out[-1] == self.tok.eot_id)


if __name__ == "__main__":
    unittest.main()
