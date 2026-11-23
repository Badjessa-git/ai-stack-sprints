import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deps import sprint  # noqa: E402
from src.hf_tokenizer import HFTokenizer  # noqa: E402

btu = sprint("sprint-04-gpt2").gpt2_tokenizer.bytes_to_unicode


def write_json(merges_as_lists):
    t = btu()
    vocab = {t[b]: b + 1 for b in range(256)}  # ids 1..256; 0 is the special token
    merges = [("a", "b"), ("Ġ", "ab")]
    for n, (l, r) in enumerate(merges):
        vocab[l + r] = 257 + n
    j = {
        "added_tokens": [{"id": 0, "content": "<|endoftext|>", "special": True}],
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Digits", "individual_digits": True}, {"type": "ByteLevel"}]},
        "model": {"type": "BPE", "vocab": vocab,
                  "merges": [list(m) for m in merges] if merges_as_lists else [f"{l} {r}" for l, r in merges]},
    }
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(j, f)
    f.close()
    return f.name, vocab


class TestHFTokenizer(unittest.TestCase):
    def test_both_merge_formats(self):
        for as_lists in (False, True):
            path, vocab = write_json(as_lists)
            tok = HFTokenizer.from_tokenizer_json(path)
            os.remove(path)
            self.assertEqual(tok.encode(" ab"), [vocab["Ġab"]], f"lists={as_lists}")

    def test_added_tokens(self):
        path, _ = write_json(False)
        tok = HFTokenizer.from_tokenizer_json(path)
        os.remove(path)
        self.assertEqual(tok.eot_id, 0)
        self.assertEqual(tok.vocab["<|endoftext|>"], 0)

    def test_digits_split_one_by_one(self):
        path, _ = write_json(False)
        tok = HFTokenizer.from_tokenizer_json(path)
        os.remove(path)
        self.assertEqual(tok.pre_tokenize("in 1994."), ["in", " ", "1", "9", "9", "4", "."])
        self.assertEqual(tok.pre_tokenize("x2y"), ["x", "2", "y"])
        self.assertEqual(tok.pre_tokenize("no digits here"), ["no", " digits", " here"])

    def test_lossless(self):
        path, _ = write_json(False)
        tok = HFTokenizer.from_tokenizer_json(path)
        os.remove(path)
        for s in ["", "a1b22c333", " 12 34 ", "café 😀 3.14\n\n"]:
            self.assertEqual("".join(tok.pre_tokenize(s)), s)
            self.assertEqual(tok.decode(tok.encode(s)), s)


if __name__ == "__main__":
    unittest.main()
