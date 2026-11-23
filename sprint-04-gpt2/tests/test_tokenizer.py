import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gpt2_tokenizer import GPT2Tokenizer, bytes_to_unicode  # noqa: E402


class TestByteTable(unittest.TestCase):
    def setUp(self):
        self.t = bytes_to_unicode()

    def test_covers_every_byte_once(self):
        self.assertEqual(sorted(self.t), list(range(256)))
        self.assertEqual(len(set(self.t.values())), 256)

    def test_printable_bytes_are_themselves(self):
        for c in "AZaz09!~¡¬®ÿ":
            self.assertEqual(self.t[ord(c)], c)

    def test_the_famous_ones(self):
        self.assertEqual(self.t[0], chr(256))
        self.assertEqual(self.t[ord(" ")], "Ġ")  # U+0120
        self.assertEqual(self.t[ord("\n")], "Ċ")  # U+010A
        self.assertEqual(self.t[127], chr(256 + 33))  # after 0..32
        self.assertEqual(self.t[173], chr(256 + 67))  # soft hyphen, the last one

    def test_every_char_is_printable(self):
        self.assertTrue(all(c.isprintable() and not c.isspace() for c in self.t.values()))


def synthetic():
    """A tiny GPT-2-style vocab where rank order and id order DISAGREE.
    Ids are given in reverse of the order the merges were learned."""
    t = bytes_to_unicode()
    vocab = {t[b]: b for b in range(256)}
    merges = [("a", "b"), ("Ġ", "ab"), ("b", "c"), ("Ġab", "c")]
    # merged tokens get ids 300, 299, 298, 297: id order is reversed rank order
    for n, (l, r) in enumerate(merges):
        vocab[l + r] = 300 - n
    vocab["<|endoftext|>"] = 301
    return GPT2Tokenizer(vocab, merges), vocab


class TestEncodeSynthetic(unittest.TestCase):
    def setUp(self):
        self.tok, self.vocab = synthetic()

    def test_ranks(self):
        v = self.vocab
        self.assertEqual(self.tok.ranks[(v["a"], v["b"])], (0, 300))
        self.assertEqual(self.tok.ranks[(v["Ġab"], v["c"])], (3, 297))

    def test_rank_not_id(self):
        # "abc": (a,b) is rank 0, (b,c) rank 2 -> ab + c. Comparing ids
        # instead (lower id = (b,c) = 298) would give a + bc.
        v = self.vocab
        self.assertEqual(self.tok.encode("abc"), [v["ab"], v["c"]])

    def test_merges_build_on_merges(self):
        v = self.vocab
        self.assertEqual(self.tok.encode(" abc"), [v["Ġabc"]])
        self.assertEqual(self.tok.encode("x abc"), [v["x"], v["Ġabc"]])

    def test_no_merges_means_byte_tokens(self):
        self.assertEqual(self.tok.encode("é"), [0xC3, 0xA9])

    def test_eot_id(self):
        self.assertEqual(self.tok.eot_id, 301)

    def test_roundtrip(self):
        for s in ["", "abc abc", "x  y\n\nz", "café 😀", "it's 1994."]:
            self.assertEqual(self.tok.decode(self.tok.encode(s)), s, repr(s))

    def test_decode_partial_utf8(self):
        # one byte of a two-byte character: replaced, not an exception
        self.assertEqual(self.tok.decode([0xC3]), "�")


if __name__ == "__main__":
    unittest.main()
