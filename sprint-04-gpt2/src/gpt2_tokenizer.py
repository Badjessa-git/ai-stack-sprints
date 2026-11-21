"""GPT-2's tokenizer, on top of your Sprint 1 Part 3 tokenizer.

GPT-2's tokenizer IS the algorithm you built in Part 3: regex pre-split,
then byte-level BPE inside each chunk, lowest-ranked merge first. Three
things differ, and they are this file:

1. **The byte table (bytes_to_unicode).** GPT-2's files store tokens as
   *strings*, not bytes, so every one of the 256 bytes needs a printable
   character to stand for it. Printable bytes stand for themselves ('A' is
   'A'). The other 68 (control bytes, space, and a few more) are moved
   to unused characters from U+0100 up, in order. That's why a space shows
   up as 'Ġ' (U+0120) in GPT-2's vocab and a newline as 'Ċ' (U+010A).

2. **Rank is not id.** In Part 3, merge number r created token 256 + r, so
   "lowest rank" and "lowest id" were the same thing. GPT-2's ids come from
   vocab.json, and its ranks are the line order of merges.txt. They differ,
   so encoding must look up ranks, not compare ids.

3. **The files.** vocab.json maps token string -> id (50,257 entries,
   including <|endoftext|> = 50256). merges.txt has a "#version" line, then
   one merge per line: "Ġ t" means: the token "Ġ" followed by the token "t"
   merges into "Ġt". Line order is rank (0 = learned first).

The split pattern is your Part 3 SPLIT_PATTERN. GPT-2's original uses
\\p{L} and \\p{N} from the third-party `regex` module. Yours matches it on
everything except a few rare characters (superscripts like ², fractions
like ½), which yours groups with letters and GPT-2 groups with numbers.
"""

import json
import os
import re
import sys

_PART3 = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "sprint-01-bpe-tokenizer",
    "part3",
)
sys.path.insert(0, _PART3)
from regex_tokenizer import SPLIT_PATTERN  # noqa: E402  (your Part 3 pattern)

EOT = "<|endoftext|>"


def bytes_to_unicode():
    """Return {byte: char} for all 256 bytes.

    Kept as themselves (char = chr(byte)), in this order:
        ord("!") .. ord("~")     (33..126)
        ord("¡") .. ord("¬")     (161..172)
        ord("®") .. ord("ÿ")     (174..255)
    Every other byte b, in increasing order, gets chr(256 + n), where n
    counts 0, 1, 2, ... over those leftover bytes. So byte 0 -> chr(256),
    byte 32 (space) -> chr(288) = 'Ġ'."""
    raise NotImplementedError("bytes_to_unicode (TestByteTable)")


class GPT2Tokenizer:
    def __init__(self, vocab, merges):
        """vocab: {token_string: id}. merges: [(left_string, right_string)]
        in rank order. Use from_files() to load the real ones."""
        self.vocab = vocab
        self.inverse = {i: s for s, i in vocab.items()}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {c: b for b, c in self.byte_encoder.items()}
        self.ranks = self.build_ranks(vocab, merges)
        self.eot_id = vocab.get(EOT)
        self._pattern = re.compile(SPLIT_PATTERN)
        self._cache = {}

    @classmethod
    def from_files(cls, vocab_path, merges_path):
        vocab = json.load(open(vocab_path, encoding="utf-8"))
        merges = []
        for line in open(merges_path, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line or line.startswith("#version"):
                continue
            left, right = line.split(" ")
            merges.append((left, right))
        return cls(vocab, merges)

    @staticmethod
    def build_ranks(vocab, merges):
        """Return {(left_id, right_id): (rank, merged_id)}.

        For merge number r, (left, right): look up the ids of left, right and
        left + right in vocab. Rank r is its line position. This table is
        what lets _encode_chunk ask "which adjacent pair was learned first?"
        without ever comparing ids."""
        raise NotImplementedError("build_ranks (TestEncodeSynthetic)")

    def _encode_chunk(self, chunk):
        """One pre-split chunk (a str) -> list of token ids.

        Start: chunk.encode("utf-8"), each byte -> its table char -> its id
        in vocab. Then, like Part 3's _encode_chunk: repeatedly find the
        adjacent pair with the LOWEST RANK in self.ranks, merge every
        occurrence of it left to right into its merged id, and stop when no
        adjacent pair is in self.ranks."""
        raise NotImplementedError("_encode_chunk (TestEncodeSynthetic)")

    def encode(self, text):
        """Split with the Part 3 pattern, encode each chunk (cached)."""
        out = []
        for chunk in self._pattern.findall(text):
            ids = self._cache.get(chunk)
            if ids is None:
                ids = self._cache[chunk] = self._encode_chunk(chunk)
            out.extend(ids)
        return out

    def decode(self, ids, errors="replace"):
        """ids -> token strings -> table chars -> bytes -> UTF-8 text.
        A single id can be half of a multi-byte character, so decoding a
        token at a time (when streaming) needs errors="replace"."""
        raise NotImplementedError("decode (TestEncodeSynthetic)")

    def token_bytes(self, i):
        """The raw bytes of one token (scaffold helper for streaming)."""
        return bytes(self.byte_decoder[c] for c in self.inverse[i])
