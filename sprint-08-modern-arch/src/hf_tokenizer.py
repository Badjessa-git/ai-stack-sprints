"""Hugging Face's tokenizer.json, on top of your Sprint 4 GPT-2 tokenizer.

Most modern models ship one file, tokenizer.json, instead of GPT-2's
vocab.json + merges.txt. For byte-level BPE models like SmolLM it holds
the same two things:

    model.vocab   {token_string: id}       (49,152 entries for SmolLM)
    model.merges  ["Ġ t", "Ġ a", ...]      in rank order; newer files
                  store pairs as lists, ["Ġ", "t"]: accept both
    added_tokens  [{"id": 0, "content": "<|endoftext|>", "special": true}, ...]

plus a description of the pre-tokenizer. SmolLM's is:

    Sequence[ Digits(individual_digits=True), ByteLevel(use_regex=True) ]

meaning: first cut every digit out on its own, then run GPT-2's regex
on each piece. So "in 1994." becomes "in", " ", "1", "9", "9", "4", ".",
and the model sees numbers one digit at a time. (GPT-2 would keep " 1994"
whole. Splitting digits makes arithmetic much easier to learn: an
example of a tokenizer decision changing what a model can do.)

The BPE itself (ranks, chunk encoding, byte table, decoding) is your
Sprint 4 GPT2Tokenizer, unchanged. This file only loads the JSON and
changes the split.
"""

import json
import re

from .deps import sprint

_t4 = sprint("sprint-04-gpt2").gpt2_tokenizer
DIGIT = re.compile(r"\d")


class HFTokenizer(_t4.GPT2Tokenizer):
    @classmethod
    def from_tokenizer_json(cls, path):
        """Read model.vocab, model.merges (strings "a b" OR lists ["a", "b"])
        and added_tokens (merge them into the vocab: id -> content).
        Return cls(vocab, merges) with self.eot_id set to the id of
        "<|endoftext|>" (0 for SmolLM)."""
        raise NotImplementedError("from_tokenizer_json (TestHFTokenizer)")

    def pre_tokenize(self, text):
        """Digits first, then the GPT-2 split: return the list of chunks.

        1. split text into pieces where every single digit is its own
           piece and the runs between digits are pieces too (keep
           everything: "".join(pieces) == text)
        2. self._pattern.findall on each non-digit piece; digits stay as
           they are."""
        raise NotImplementedError("pre_tokenize (TestHFTokenizer)")

    def encode(self, text):
        out = []
        for chunk in self.pre_tokenize(text):
            ids = self._cache.get(chunk)
            if ids is None:
                ids = self._cache[chunk] = self._encode_chunk(chunk)
            out.extend(ids)
        return out
