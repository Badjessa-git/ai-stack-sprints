"""Part 3 tests are the spec. From sprint-01-bpe-tokenizer/:

    python3 -m unittest part3/test_regex_tokenizer.py -v

Grouped in the order to make them pass: pre-tokenization, counting,
training, encoding, then whole-tokenizer behaviour.
"""

import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from regex_tokenizer import (  # noqa: E402
    RegexBPETokenizer,
    get_stats_weighted,
    pre_tokenize,
    word_counts,
)

CORPUS = (HERE.parent / "data" / "corpus.txt").read_text()

TRICKY = [
    "",
    " ",
    "Hello, world!",
    "  leading and trailing  ",
    "tabs\tand\nnewlines\n\n\nhere",
    "café naïve résumé",
    "emoji 😀🎉 mixed",
    "日本語のテキスト",
    "numbers 1234567 and 3.14",
    "WikiText style @-@ 1 @,@ 000 = = Title = =",
    "it's we'll they're I'd you've",
    "under_score snake_case",
    "punctuation!!! ??? ...",
]


def slow_encode(text, merges):
    """Oracle: apply every merge, in learned order, inside each chunk."""
    out = []
    for chunk in pre_tokenize(text):
        ids = list(chunk.encode("utf-8"))
        for (l, r), nid in sorted(merges.items(), key=lambda kv: kv[1]):
            merged, i = [], 0
            while i < len(ids):
                if i + 1 < len(ids) and ids[i] == l and ids[i + 1] == r:
                    merged.append(nid)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged
        out.extend(ids)
    return out


class Test1PreTokenize(unittest.TestCase):
    def test_words_keep_their_leading_space(self):
        self.assertEqual(pre_tokenize("Hello world"), ["Hello", " world"])

    def test_contractions_split_off(self):
        self.assertEqual(pre_tokenize("don't stop"), ["don", "'t", " stop"])

    def test_numbers_are_their_own_chunks(self):
        self.assertEqual(pre_tokenize("in 1994 ."), ["in", " 1994", " ."])

    def test_punctuation_splits_from_words(self):
        self.assertEqual(pre_tokenize("end."), ["end", "."])

    def test_extra_spaces_stay_behind_the_next_word(self):
        # "x   y": two spaces on their own, the third goes with "y"
        self.assertEqual(pre_tokenize("x   y"), ["x", "  ", " y"])
        self.assertEqual(pre_tokenize("  hi"), [" ", " hi"])

    def test_newlines(self):
        self.assertEqual(pre_tokenize("a\n\nb"), ["a", "\n", "\n", "b"])

    def test_wikitext_markup(self):
        self.assertEqual(pre_tokenize(" @-@ "), [" @-@", " "])

    def test_unicode_letters_are_letters(self):
        self.assertEqual(pre_tokenize("café 😀!"), ["café", " 😀!"])

    def test_lossless(self):
        for s in TRICKY + [CORPUS]:
            self.assertEqual("".join(pre_tokenize(s)), s, repr(s[:40]))


class Test2WordCounts(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(word_counts(["a", " b", "a"]), {"a": 2, " b": 1})

    def test_empty(self):
        self.assertEqual(word_counts([]), {})


class Test2bGetStatsWeighted(unittest.TestCase):
    def test_weights_by_count(self):
        chunks = [([97, 98], 3), ([97, 98, 99], 2)]
        self.assertEqual(get_stats_weighted(chunks), {(97, 98): 5, (98, 99): 2})

    def test_never_pairs_across_chunks(self):
        # (98, 99) would only exist if "ab" and "cd" were glued together.
        self.assertEqual(
            get_stats_weighted([([97, 98], 1), ([99, 100], 1)]),
            {(97, 98): 1, (99, 100): 1},
        )

    def test_overlapping_pairs_count(self):
        self.assertEqual(get_stats_weighted([([1, 1, 1], 4)]), {(1, 1): 8})

    def test_short_and_empty(self):
        self.assertEqual(get_stats_weighted([]), {})
        self.assertEqual(get_stats_weighted([([7], 10), ([], 3)]), {})

    def test_does_not_mutate_input(self):
        chunks = [([97, 98], 3)]
        get_stats_weighted(chunks)
        self.assertEqual(chunks, [([97, 98], 3)])


class Test3Train(unittest.TestCase):
    def test_vocab_too_small_raises(self):
        with self.assertRaises(ValueError):
            RegexBPETokenizer().train_from_counts({"ab": 5}, 255)

    def test_vocab_256_learns_nothing(self):
        tok = RegexBPETokenizer()
        tok.train_from_counts({"ab": 5}, 256)
        self.assertEqual(tok.merges, {})

    def test_counts_weight_the_pairs(self):
        # (a,b) occurs 3 + 2 = 5 times, (b,c) 2 times.
        tok = RegexBPETokenizer()
        tok.train_from_counts({"ab": 3, "abc": 2}, 258)
        self.assertEqual(tok.merges, {(97, 98): 256, (256, 99): 257})

    def test_ties_pick_the_smallest_pair(self):
        tok = RegexBPETokenizer()
        tok.train_from_counts({"cd": 1, "ab": 1}, 257, min_frequency=1)
        self.assertEqual(tok.merges, {(97, 98): 256})

    def test_min_frequency_stops_early(self):
        tok = RegexBPETokenizer()
        tok.train_from_counts({"ab": 1}, 300)
        self.assertEqual(tok.merges, {})

    def test_never_merges_across_chunks(self):
        # Part 1 would learn (b, space): "ab ab ab ab" has it 3 times.
        tok = RegexBPETokenizer()
        tok.train("ab ab ab ab", 258)
        self.assertNotIn((98, 32), tok.merges)
        self.assertEqual(tok.merges, {(97, 98): 256, (32, 256): 257})

    def test_retraining_resets(self):
        tok = RegexBPETokenizer()
        tok.train("ab ab ab ab", 258)
        tok.train("cd cd cd cd", 257)
        self.assertEqual(tok.merges, {(99, 100): 256})

    def test_spaces_only_lead_word_tokens(self):
        # The property the whole Part exists for: a token with a letter in
        # it never has a space anywhere but its first byte.
        tok = RegexBPETokenizer()
        tok.train(CORPUS, 400)
        self.assertGreater(len(tok.merges), 50)
        for b in tok.token_bytes:
            if any(chr(c).isalpha() for c in b if c < 128):
                self.assertNotIn(b" ", b[1:], b)


class Test4EncodeChunk(unittest.TestCase):
    def test_merges_within_each_chunk(self):
        tok = RegexBPETokenizer()
        tok.set_merges({(97, 98): 256, (256, 99): 257})
        self.assertEqual(tok._encode_chunk(b"abc"), [257])
        self.assertEqual(tok._encode_chunk(b" ab"), [32, 256])
        self.assertEqual(tok.encode("abc ab"), [257, 32, 256])

    def test_lowest_rank_wins_not_leftmost(self):
        # (b,c) was learned first, so "abc" -> a + bc, not ab + c.
        tok = RegexBPETokenizer()
        tok.set_merges({(98, 99): 256, (97, 98): 257})
        self.assertEqual(tok._encode_chunk(b"abc"), [97, 256])

    def test_repeated_pair_merges_left_to_right(self):
        tok = RegexBPETokenizer()
        tok.set_merges({(97, 97): 256})
        self.assertEqual(tok._encode_chunk(b"aaa"), [256, 97])

    def test_no_merges_means_bytes(self):
        tok = RegexBPETokenizer()
        self.assertEqual(tok._encode_chunk("é".encode()), [195, 169])

    def test_matches_applying_merges_in_order(self):
        tok = RegexBPETokenizer()
        tok.train(CORPUS, 400)
        for s in TRICKY + [CORPUS]:
            self.assertEqual(tok.encode(s), slow_encode(s, tok.merges), repr(s[:40]))


class Test5Tokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = RegexBPETokenizer()
        cls.tok.train(CORPUS, 400)

    def test_empty(self):
        self.assertEqual(self.tok.encode(""), [])
        self.assertEqual(self.tok.decode([]), "")

    def test_roundtrip(self):
        for s in TRICKY + [CORPUS]:
            self.assertEqual(self.tok.decode(self.tok.encode(s)), s, repr(s[:40]))

    def test_compresses(self):
        self.assertLess(len(self.tok.encode(CORPUS)), len(CORPUS.encode()) * 0.7)

    def test_trailing_space_does_not_change_the_word_tokens(self):
        # Part 1's tokenizer re-tokenised "a" vs "a " differently at the
        # boundary. Here the prefix tokens are identical; the space is
        # just one more token at the end.
        a = self.tok.encode("Albert Einstein was a")
        b = self.tok.encode("Albert Einstein was a ")
        self.assertEqual(b[: len(a)], a)

    def test_save_load(self):
        path = HERE / "_tmp_merges.json"
        try:
            self.tok.save(str(path))
            other = RegexBPETokenizer.load(str(path))
            self.assertEqual(other.merges, self.tok.merges)
            self.assertEqual(other.encode(CORPUS), self.tok.encode(CORPUS))
        finally:
            if path.exists():
                os.remove(path)


if __name__ == "__main__":
    unittest.main()
