"""Acceptance: SmolLM-135M on your code. Skipped until: make download"""

import os
import sys
import unittest
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deps import smollm_dir  # noqa: E402

HAVE = os.path.exists(os.path.join(smollm_dir(), "model.safetensors"))
PARAGRAPH = (
    "The city is located on the banks of the river, about 40 kilometres from "
    "the coast. It was founded in the 12th century and became an important "
    "centre of trade. Today it has a population of around 200,000 people, and "
    "its old town is a popular destination for tourists."
)


@unittest.skipUnless(HAVE, "run make download first")
class TestRealSmolLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.hf_tokenizer import HFTokenizer
        from src.llama import Llama

        cls.tok = HFTokenizer.from_tokenizer_json(os.path.join(smollm_dir(), "tokenizer.json"))
        cls.model = Llama.from_dir(smollm_dir())

    def test_tokenizer(self):
        self.assertEqual(self.tok.encode("Hello world"), [19556, 905])
        self.assertEqual(self.tok.encode("in 1994."), [254, 216, 33, 41, 41, 36, 30])
        self.assertEqual(self.tok.eot_id, 0)
        self.assertEqual(self.tok.decode(self.tok.encode(PARAGRAPH)), PARAGRAPH)

    def test_parameter_count(self):
        self.assertEqual(sum(v.size for v in self.model.p.values()), 134_515_008)

    def test_perplexity(self):
        # SmolLM-135M: ~1.5 nats/token here (GPT-2 small: ~2.9).
        ids = self.tok.encode(PARAGRAPH)
        lg = self.model.forward(ids).astype(np.float64)
        lg -= lg.max(axis=1, keepdims=True)
        lp = lg - np.log(np.exp(lg).sum(axis=1, keepdims=True))
        self.assertLess(-lp[np.arange(len(ids) - 1), ids[1:]].mean(), 2.0)

    def test_knows_paris(self):
        out = self.model.generate(self.tok.encode("The capital of France is"), 2)
        self.assertEqual(self.tok.decode(out), " Paris.")


if __name__ == "__main__":
    unittest.main()
