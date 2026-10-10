"""Tests are the spec. Run: python3 -m unittest discover -s tests -v

They are grouped: helpers first, then training, then roundtrip.
Get them passing in that order.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer import BPETokenizer, get_stats, merge_pair


class TestHelpers(unittest.TestCase):
    def test_get_stats_basic(self):
        self.assertEqual(get_stats([1, 2, 1, 2, 1]), {(1, 2): 2, (2, 1): 2})

    def test_get_stats_empty_and_single(self):
        self.assertEqual(get_stats([]), {})
        self.assertEqual(get_stats([7]), {})

    def test_merge_pair_basic(self):
        self.assertEqual(merge_pair([1, 2, 1, 2, 1], (1, 2), 99), [99, 99, 1])

    def test_merge_pair_no_match_returns_equal_list(self):
        self.assertEqual(merge_pair([1, 2, 3], (3, 1), 99), [1, 2, 3])

    def test_merge_pair_does_not_mutate_input(self):
        original = [1, 2, 1, 2]
        merge_pair(original, (1, 2), 99)
        self.assertEqual(original, [1, 2, 1, 2])


class TestTraining(unittest.TestCase):
    def test_rejects_vocab_below_256(self):
        tok = BPETokenizer()
        with self.assertRaises(ValueError):
            tok.train("hello hello hello", vocab_size=255)

    def test_training_reaches_requested_vocab(self):
        tok = BPETokenizer()
        tok.train("the quick brown fox jumps over the lazy dog. " * 10, vocab_size=266)
        self.assertEqual(len(tok.merges), 10)

    def test_most_frequent_pair_is_merged_first(self):
        # "ab" occurs 3 times, every other pair at most twice.
        tok = BPETokenizer()
        tok.train("ab ab ab", vocab_size=257)
        first_pair = next(iter(tok.merges))
        self.assertEqual(first_pair, (ord("a"), ord("b")))

    def test_training_is_deterministic(self):
        text = "the quick brown fox jumps over the lazy dog. " * 20
        a, b = BPETokenizer(), BPETokenizer()
        a.train(text, vocab_size=300)
        b.train(text, vocab_size=300)
        self.assertEqual(a.merges, b.merges)


class TestRoundtrip(unittest.TestCase):
    def setUp(self):
        self.tok = BPETokenizer()
        self.tok.train(
            "the quick brown fox jumps over the lazy dog. " * 30,
            vocab_size=320,
        )

    def test_roundtrip_ascii(self):
        for s in ["hello world", "the quick brown fox", "a", "zzzzzz"]:
            self.assertEqual(self.tok.decode(self.tok.encode(s)), s)

    def test_roundtrip_unicode_is_lossless(self):
        for s in ["café", "naïve résumé", "emoji: 🚀🔥", "日本語テスト"]:
            self.assertEqual(self.tok.decode(self.tok.encode(s)), s)

    def test_empty_string(self):
        self.assertEqual(self.tok.encode(""), [])
        self.assertEqual(self.tok.decode([]), "")

    def test_untrained_tokenizer_is_byte_identity(self):
        tok = BPETokenizer()
        s = "anything at all"
        self.assertEqual(tok.encode(s), list(s.encode("utf-8")))
        self.assertEqual(tok.decode(tok.encode(s)), s)

    def test_repetitive_text_compresses(self):
        s = "the quick brown fox jumps over the lazy dog. " * 10
        self.assertLess(len(self.tok.encode(s)), len(s.encode("utf-8")))

    def test_merge_rank_matters(self):
        # Training on "aaaa..." learns (a,a) -> new token first, then
        # (new,new) -> next token. Encoding must apply merges in rank
        # order, so "aaaa" collapses all the way to that second token:
        # exactly one token, and specifically the second merge learned.
        tok = BPETokenizer()
        tok.train("aaaaaaaa" * 20, vocab_size=258)
        encoded = tok.encode("aaaa")
        self.assertEqual(encoded, [257])


if __name__ == "__main__":
    unittest.main()
