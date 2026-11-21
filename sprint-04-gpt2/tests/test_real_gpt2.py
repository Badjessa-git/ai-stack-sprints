"""End to end against OpenAI's real GPT-2 (124M). Skipped until you run:
    python3 data/download_gpt2.py          (~550 MB into data/gpt2/)

These are the acceptance tests: if they pass, your code is GPT-2.
"""

import os
import sys
import unittest
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
GPT2_DIR = os.path.join(HERE, "data", "gpt2")
HAVE = os.path.exists(os.path.join(GPT2_DIR, "model.safetensors"))

PARAGRAPH = (
    "The city is located on the banks of the river, about 40 kilometres from "
    "the coast. It was founded in the 12th century and became an important "
    "centre of trade. Today it has a population of around 200,000 people, and "
    "its old town is a popular destination for tourists."
)


@unittest.skipUnless(HAVE, "run python3 data/download_gpt2.py first")
class TestRealTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.gpt2_tokenizer import GPT2Tokenizer

        cls.tok = GPT2Tokenizer.from_files(
            os.path.join(GPT2_DIR, "vocab.json"), os.path.join(GPT2_DIR, "merges.txt")
        )

    def test_known_ids(self):
        # The ids Hugging Face's own docs show for these strings.
        self.assertEqual(self.tok.encode("Hello world"), [15496, 995])
        self.assertEqual(self.tok.encode("Hello, my dog is cute"), [15496, 11, 616, 3290, 318, 13779])
        self.assertEqual(self.tok.encode(" the"), [262])

    def test_eot(self):
        self.assertEqual(self.tok.eot_id, 50256)
        self.assertEqual(len(self.tok.vocab), 50257)

    def test_roundtrip(self):
        for s in [PARAGRAPH, "café naïve 😀\n\n  x", "it's 3.14 @-@ =="]:
            self.assertEqual(self.tok.decode(self.tok.encode(s)), s)


@unittest.skipUnless(HAVE, "run python3 data/download_gpt2.py first")
class TestRealModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.gpt2 import GPT2
        from src.gpt2_tokenizer import GPT2Tokenizer

        cls.tok = GPT2Tokenizer.from_files(
            os.path.join(GPT2_DIR, "vocab.json"), os.path.join(GPT2_DIR, "merges.txt")
        )
        cls.model = GPT2.from_dir(GPT2_DIR)

    def test_loads_124m_parameters(self):
        n = sum(v.size for v in self.model.p.values())
        self.assertEqual(n, 124_439_808)

    def test_perplexity(self):
        # GPT-2 small: ~2.9 nats/token on this paragraph. A model with a
        # transposed matrix or a wrong layer norm scores 7+ (near random).
        ids = self.tok.encode(PARAGRAPH)
        logits = self.model.forward(ids).astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        nll = -logp[np.arange(len(ids) - 1), ids[1:]].mean()
        self.assertLess(nll, 3.5)

    def test_greedy_text(self):
        ids = self.tok.encode("Albert Einstein was a")
        out = self.tok.decode(self.model.generate_greedy(ids, 4))
        self.assertEqual(out, "Albert Einstein was a great scientist, but")


if __name__ == "__main__":
    unittest.main()
