"""Quantized real GPT-2 (uses Sprint 4's data/gpt2/)."""

import os
import sys
import unittest
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deps import gpt2_dir, sprint  # noqa: E402

HAVE = os.path.exists(os.path.join(gpt2_dir(), "model.safetensors"))
PARAGRAPH = (
    "The city is located on the banks of the river, about 40 kilometres from "
    "the coast. It was founded in the 12th century and became an important "
    "centre of trade. Today it has a population of around 200,000 people, and "
    "its old town is a popular destination for tourists."
)


@unittest.skipUnless(HAVE, "run make download in sprint-04-gpt2 first")
class TestRealQuantized(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s4 = sprint("sprint-04-gpt2")
        d = gpt2_dir()
        cls.s4 = s4
        cls.tok = s4.gpt2_tokenizer.GPT2Tokenizer.from_files(os.path.join(d, "vocab.json"), os.path.join(d, "merges.txt"))
        cls.base = s4.gpt2.GPT2.from_dir(d)
        cls.ids = cls.tok.encode(PARAGRAPH)

    def nll(self, params):
        lg = self.s4.gpt2.GPT2(self.base.cfg, params).forward(self.ids).astype(np.float64)
        lg -= lg.max(axis=1, keepdims=True)
        lp = lg - np.log(np.exp(lg).sum(axis=1, keepdims=True))
        return -lp[np.arange(len(self.ids) - 1), self.ids[1:]].mean()

    def test_int8_is_almost_free(self):
        from src.quant import QuantizedParams

        self.assertLess(abs(self.nll(QuantizedParams.from_params(self.base.p, 8)) - self.nll(self.base.p)), 0.02)

    def test_int4_costs_a_little(self):
        from src.quant import QuantizedParams

        diff = self.nll(QuantizedParams.from_params(self.base.p, 4, 32)) - self.nll(self.base.p)
        self.assertLess(diff, 0.3)

    def test_int8_halves_the_file(self):
        from src.quant import QuantizedParams

        float_mb = sum(v.nbytes for v in self.base.p.values()) / 1e6
        self.assertLess(QuantizedParams.from_params(self.base.p, 8).nbytes() / 1e6, float_mb * 0.5)


if __name__ == "__main__":
    unittest.main()
