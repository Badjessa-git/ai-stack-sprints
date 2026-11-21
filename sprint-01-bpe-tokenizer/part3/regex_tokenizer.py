"""Regex-split byte-level BPE — SPRINT 1, PART 3.

Part 1's tokenizer runs BPE over the raw byte stream, so merges cross word
boundaries ("e " + "t" ...). On WikiText-103 at vocab 512, 126 of its 256
merges END in a space and only 3 START with one: tokens glue a word's tail
to the next space. That's why a trailing space in the prompt changed the
transformer's output, and why the v2 models invent words like
"somovolobobid" once they're unsure.

Part 3 does what GPT-2 does: split the text into word-like chunks with a
regex FIRST, then run BPE inside each chunk only. A merge can never cross a
chunk boundary, so every token is a piece of one word (with its leading
space), a number, a punctuation run, or whitespace.

WHO WRITES WHAT
    Yours (stubs below raise NotImplementedError), in this order:
        1. SPLIT_PATTERN + pre_tokenize
        2. word_counts
        3. RegexBPETokenizer.train_from_counts
        4. RegexBPETokenizer._encode_chunk
    Scaffold (already written): train(), encode() with a per-chunk cache,
    encode_to_array(), decode(), token_bytes, save()/load().

Spec: part3/README.md. Tests: part3/test_regex_tokenizer.py
    python3 -m unittest part3/test_regex_tokenizer.py -v     (from sprint-01-bpe-tokenizer/)
Standard library only (`re` is fine; the `regex` package isn't installed).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from array import array

# Part 1's helpers, shared through src/bpe_common.py — use them in your stubs:
#   get_stats([1, 2, 1, 2, 1])            == {(1, 2): 2, (2, 1): 2}
#   merge_pair([1, 2, 1, 2, 1], (1, 2), 99) == [99, 99, 1]
# (Imported as `bpe_common`, not `src.bpe_common`: under Sprint 3's
# train_v3.py the name `src` is Sprint 3's own package.)
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)
from bpe_common import get_stats, merge_pair  # noqa: E402,F401

MIN_VOCAB_SIZE = 256

# ---------------------------------------------------------------------------
# 1. Pre-tokenization (yours)
# ---------------------------------------------------------------------------

# The split pattern, as a raw string. GPT-2's is in README.md — but it uses
# \p{L} / \p{N}, which stdlib `re` doesn't support. Work out the stdlib
# equivalent (hint: "a letter" = a word character that is not a digit and
# not an underscore). The tests pin the behaviour, not the exact string.
SPLIT_PATTERN: str = (
    r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"""
)
_PATTERN = re.compile(SPLIT_PATTERN)


def pre_tokenize(text: str) -> list[str]:
    """Split `text` into chunks with SPLIT_PATTERN, left to right.

    Examples (GPT-2 behaviour, see the tests for more):
        pre_tokenize("Hello world") == ["Hello", " world"]
        pre_tokenize("don't stop")  == ["don", "'t", " stop"]

    Must be lossless: "".join(pre_tokenize(s)) == s for every string s.
    Compile the pattern once, not on every call.
    """
    return _PATTERN.findall(text)


def word_counts(chunks: list[str]) -> dict[str, int]:
    """How many times each chunk occurs.

    Example: word_counts(["a", " b", "a"]) == {"a": 2, " b": 1}

    This is the whole trick that makes training fast: WikiText-103's 19M
    chunks collapse to ~242k distinct ones, and BPE only ever needs to
    look at each distinct chunk once, weighted by its count.
    """
    word_counts = {}
    for chunk in chunks:
        word_counts[chunk] = word_counts.get(chunk, 0) + 1
    return word_counts


def get_stats_weighted(
    chunks: list[tuple[list[int], int]],
) -> dict[tuple[int, int], int]:
    """Count adjacent pairs across all chunks, each weighted by its chunk's count.

    `chunks` is a list of (token_ids, count): one entry per distinct chunk,
    with how many times that chunk appears in the text.

    Example:
        get_stats_weighted([([97, 98], 3), ([97, 98, 99], 2)])
            == {(97, 98): 5, (98, 99): 2}
        "ab" x3 gives (a,b) 3 times; "abc" x2 gives (a,b) 2 more and (b,c) 2.

    Rules:
      - Pairs are counted INSIDE each chunk only. The last id of one chunk
        and the first id of the next are never a pair.
      - A pair seen n times in a chunk adds n * count.
      - Overlapping pairs both count, like get_stats: [1, 1, 1] has (1, 1) twice.
      - Chunks with fewer than 2 ids add nothing.

    Hint: one dict for the whole call, one loop over chunks, one loop over
    each chunk's pairs. Same shape as get_stats, plus the count.
    """
    stats = {}
    for chunk in chunks:
        i = 0
        length = len(chunk[0])
        if length < 2:
            continue

        ids = chunk[0]
        while i < length - 1:
            pair = (ids[i], ids[i + 1])
            stats[pair] = stats.get(pair, 0) + chunk[1]
            i += 1

    return stats


# ---------------------------------------------------------------------------
# The tokenizer
# ---------------------------------------------------------------------------


class RegexBPETokenizer:
    """Byte-level BPE inside regex chunks.

    Same id scheme as Part 1: ids 0-255 are bytes, learned merges get
    256, 257, ... in the order they're learned. `self.merges` maps
    (left_id, right_id) -> new_id; a lower new_id = learned earlier =
    higher priority when encoding.
    """

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self._token_bytes: list[bytes] | None = None
        self._cache: dict[str, list[int]] = {}
        self.cache_limit = 1_000_000  # distinct chunks remembered by encode()
        # Progress output during training: a line every `log_every` merges,
        # or every `log_seconds` seconds, whichever comes first. 0 = off.
        self.log_every = 256
        self.log_seconds = 15.0

    # -- training (yours) -------------------------------------------------

    def train_from_counts(
        self, counts: dict[str, int], vocab_size: int, min_frequency: int = 2
    ) -> None:
        """Learn merges from chunk counts until the vocab reaches vocab_size.

        Steps:
          1. vocab_size < 256 -> ValueError. Reset any previous merges.
          2. Turn each chunk into its list of UTF-8 byte ids, remembering
             its count.
          3. Repeat until the vocab is full:
               - count every adjacent pair INSIDE each chunk, weighted by
                 that chunk's count ("ab" seen 3 times -> (a, b) += 3).
                 Pairs never span two chunks.
               - pick the most frequent pair. Ties -> the smallest pair as
                 a tuple, e.g. (97, 98) beats (99, 100). (Deterministic,
                 and independent of dict order.)
               - stop if no pairs are left or the best count < min_frequency
               - record the merge, then merge that pair in every chunk.
          4. Store the result in self.merges, in learned order.

        Recounting all pairs every round is fine here: ~0.5 s a round on
        WikiText-103, ~30 min for a 4k vocab, done once and cached.
        Making it incremental (your Part 2 trick) is a stretch goal.
        """

        if vocab_size < MIN_VOCAB_SIZE:
            raise ValueError("vocab size cannot be less than 256")

        self.merges = {}
        chunks = [(list(word.encode("utf-8")), count) for word, count in counts.items()]

        # Progress (see log_every / log_seconds in __init__)
        target = vocab_size - MIN_VOCAB_SIZE
        token_text = [bytes([b]) for b in range(MIN_VOCAB_SIZE)]  # id -> bytes, for display
        t_start = t_last = time.time()
        if self.log_every and target >= self.log_every:
            self._log(
                f"train: {len(chunks):,} distinct chunks, "
                f"{sum(len(ids) for ids, _ in chunks):,} bytes in them, "
                f"learning up to {target:,} merges"
            )

        i = 0
        new_id = 256
        while i < vocab_size - MIN_VOCAB_SIZE:
            stats = get_stats_weighted(chunks)
            if not stats:
                break

            top_pair_min = min(stats, key=lambda pair: (-stats[pair], pair))
            frequency = stats[top_pair_min]
            if frequency < min_frequency:
                break

            self.merges[top_pair_min] = new_id
            chunks = [
                (merge_pair(ids, top_pair_min, new_id), count) for ids, count in chunks
            ]

            token_text.append(token_text[top_pair_min[0]] + token_text[top_pair_min[1]])
            now = time.time()
            if self.log_every and (
                (i + 1) % self.log_every == 0
                or (self.log_seconds and now - t_last >= self.log_seconds)
            ):
                done = i + 1
                per = (now - t_start) / done
                self._log(
                    f"  merge {done:>5,}/{target:,}  "
                    f"{token_text[new_id].decode('utf-8', 'replace')!r:<16} "
                    f"count {frequency:>10,}  "
                    f"{now - t_start:>6.0f}s elapsed  "
                    f"~{per * (target - done):.0f}s left at this rate"
                )
                t_last = now

            i += 1
            new_id += 1
            del stats  # stats gets computed on every run

        if self.log_every and target >= self.log_every:
            self._log(
                f"train: done, {len(self.merges):,} merges in {time.time() - t_start:.0f}s"
            )

    # -- encoding (yours) -------------------------------------------------

    def _encode_chunk(self, chunk: bytes) -> list[int]:
        """Encode ONE chunk's bytes to token ids.

        Start from the bytes. Repeatedly: among the adjacent pairs present,
        find the one with the LOWEST merge id (learned earliest); if none
        of them has a merge, stop; otherwise merge every occurrence of that
        pair, left to right, non-overlapping. Return the ids.

        This is the "proper" encode Part 1's docstring mentioned: the result
        must equal applying every merge in learned order, but it only looks
        at pairs that are actually in the chunk, so it's fast for short
        chunks. (The tests check it against the slow way.)
        """
        token_ids = list(chunk)
        while True:
            best_id = None
            best_pair = None
            pairs = list(zip(token_ids, token_ids[1:]))
            for pair in pairs:
                if pair in self.merges:
                    merge_id = self.merges[pair]
                    if best_id is None or merge_id < best_id:
                        best_pair = pair
                        best_id = merge_id

            if best_id is None:
                return token_ids

            token_ids = merge_pair(token_ids=token_ids, pair=best_pair, new_id=best_id)

    # -- everything below is scaffold ------------------------------------

    def train(self, text: str, vocab_size: int, min_frequency: int = 2) -> None:
        """Pre-tokenize, count, then learn merges from the counts."""
        loud = self.log_every and vocab_size - MIN_VOCAB_SIZE >= self.log_every
        t = time.time()
        chunks = pre_tokenize(text)
        if loud:
            self._log(
                f"train: split {len(text):,} chars into {len(chunks):,} chunks "
                f"in {time.time() - t:.1f}s"
            )
        t = time.time()
        counts = word_counts(chunks)
        del chunks
        if loud:
            self._log(
                f"train: counted {len(counts):,} distinct chunks in {time.time() - t:.1f}s"
            )
        self.train_from_counts(counts, vocab_size, min_frequency)
        self._token_bytes = None
        self._cache = {}

    def _log(self, msg: str) -> None:
        print(msg, flush=True)

    @property
    def vocab_size(self) -> int:
        return MIN_VOCAB_SIZE + len(self.merges)

    def set_merges(self, merges: list[list[int]] | dict[tuple[int, int], int]) -> None:
        """Install merges (from save() / a cache) without training."""
        if isinstance(merges, dict):
            merges = [[l, r, nid] for (l, r), nid in merges.items()]
        self.merges = {(l, r): nid for l, r, nid in sorted(merges, key=lambda m: m[2])}
        self._token_bytes = None
        self._cache = {}

    @property
    def token_bytes(self) -> list[bytes]:
        """token_bytes[i] = the bytes token i decodes to."""
        if self._token_bytes is None:
            table = [bytes([i]) for i in range(MIN_VOCAB_SIZE)]
            for (l, r), nid in sorted(self.merges.items(), key=lambda kv: kv[1]):
                assert nid == len(table), "merge ids must be 256, 257, ... with no gaps"
                table.append(table[l] + table[r])
            self._token_bytes = table
        return self._token_bytes

    def encode(self, text: str) -> list[int]:
        """Pre-tokenize, then encode each chunk (cached by chunk text)."""
        return list(self.encode_to_array(text))

    def encode_to_array(self, text: str) -> array:
        """encode() into a compact int32 array — for whole corpora.

        WikiText-103 has 19M chunks but only ~242k distinct ones, so the
        cache means _encode_chunk runs ~242k times, not 19M.
        """
        out = array("i")
        cache = self._cache
        for chunk in pre_tokenize(text):
            ids = cache.get(chunk)
            if ids is None:
                ids = self._encode_chunk(chunk.encode("utf-8"))
                if len(cache) < self.cache_limit:
                    cache[chunk] = ids
            out.extend(ids)
        return out

    def decode(self, token_ids, errors: str = "strict") -> str:
        """Token ids -> text. errors="replace" for model output, which can
        stop mid-way through a multi-byte character."""
        table = self.token_bytes
        return b"".join(table[i] for i in token_ids).decode("utf-8", errors=errors)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(
                {
                    "kind": "bpe-regex",
                    "pattern": SPLIT_PATTERN,
                    "vocab_size": self.vocab_size,
                    "merges": [[l, r, nid] for (l, r), nid in self.merges.items()],
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "RegexBPETokenizer":
        saved = json.load(open(path))
        if saved.get("pattern") not in (None, SPLIT_PATTERN):
            raise ValueError(
                f"{path} was trained with a different SPLIT_PATTERN; "
                "its merges won't encode the same way under the current one"
            )
        tok = cls()
        tok.set_merges(saved["merges"])
        return tok


__all__ = [
    "SPLIT_PATTERN",
    "pre_tokenize",
    "word_counts",
    "get_stats_weighted",
    "RegexBPETokenizer",
]
