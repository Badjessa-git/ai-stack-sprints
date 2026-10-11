"""Part 2 equivalence tests.

The optimized tokenizer in ../src/tokenizer.py must behave EXACTLY like
the frozen v1 in ./tokenizer_v1.py: identical merges (pairs, ids, order),
identical encodings, identical decodings. Only speed may differ.

Run from the sprint folder:
    python3 -m unittest discover -s part2 -p "test_*.py"
"""

import importlib.util
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPRINT = HERE.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v1 = _load("tokenizer_v1", HERE / "tokenizer_v1.py")
v2 = _load("tokenizer_current", SPRINT / "src" / "tokenizer.py")

CORPUS = (SPRINT / "data" / "corpus.txt").read_text(encoding="utf-8")

TEXTS = [
    CORPUS,
    "the quick brown fox jumps over the lazy dog",
    "strawberry strawberry strawberry",
    "héllo wörld — ünïcode, émojis 🚀🔥, and açcénts",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "ababababababababababab",
    "The the the the, and and and, of of of",
    "a",
    "",
    " ",
    "\n\n\n",
    "1234567890 !@#$%^&*() 1234567890",
]


def trained(module, vocab_size: int):
    tok = module.BPETokenizer()
    tok.train(CORPUS, vocab_size)
    return tok


class TestMergesIdentical(unittest.TestCase):
    def test_merges_match_at_several_vocab_sizes(self):
        for vocab_size in (280, 320, 384, 512):
            with self.subTest(vocab_size=vocab_size):
                t1, t2 = trained(v1, vocab_size), trained(v2, vocab_size)
                # items() comparison checks pairs, ids AND order (rank).
                self.assertEqual(list(t1.merges.items()), list(t2.merges.items()))

    def test_train_edge_cases_match(self):
        for text in ("", "a", "ab"):
            with self.subTest(text=text):
                t1, t2 = v1.BPETokenizer(), v2.BPETokenizer()
                t1.train(text, 512)
                t2.train(text, 512)
                self.assertEqual(t1.merges, t2.merges)

    def test_train_rejects_small_vocab_in_both(self):
        for module in (v1, v2):
            with self.assertRaises(ValueError):
                module.BPETokenizer().train("hello", 255)


class TestEncodeDecodeIdentical(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t1 = trained(v1, 512)
        cls.t2 = trained(v2, 512)

    def test_encode_identical(self):
        for text in TEXTS:
            with self.subTest(text=text[:30]):
                self.assertEqual(self.t1.encode(text), self.t2.encode(text))

    def test_decode_identical_and_roundtrips(self):
        for text in TEXTS:
            with self.subTest(text=text[:30]):
                ids = self.t1.encode(text)
                self.assertEqual(self.t1.decode(ids), text)
                self.assertEqual(self.t2.decode(ids), text)

    def test_decode_of_long_mixed_sequence_identical(self):
        # Concatenate the encodings of every sample text: a long, valid
        # sequence exercising every merge the tokenizer learned.
        ids = []
        for text in TEXTS:
            ids.extend(self.t1.encode(text))
        self.assertEqual(self.t1.decode(ids), self.t2.decode(ids))
        self.assertEqual(self.t2.decode(ids), "".join(TEXTS))


class TestHelpersIdentical(unittest.TestCase):
    def test_get_stats_identical(self):
        for ids in ([1, 2, 1, 2, 1], [], [7], [5, 5, 5, 5], list(range(50))):
            with self.subTest(ids=str(ids)[:20]):
                self.assertEqual(v1.get_stats(ids), v2.get_stats(ids))

    def test_merge_pair_identical(self):
        cases = [
            ([1, 2, 1, 2, 1], (1, 2), 99),
            ([1, 1, 1], (1, 1), 7),
            ([], (1, 2), 3),
            ([9], (9, 9), 9),
        ]
        for ids, pair, new_id in cases:
            with self.subTest(ids=ids, pair=pair):
                self.assertEqual(
                    v1.merge_pair(list(ids), pair, new_id),
                    v2.merge_pair(list(ids), pair, new_id),
                )


if __name__ == "__main__":
    unittest.main()
