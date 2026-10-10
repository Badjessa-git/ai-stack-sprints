"""Byte-level BPE tokenizer — SPRINT 1 CORE FILE.

This file is yours. Every function below is a stub that raises
NotImplementedError. Implement them by hand, in this order:

    1. get_stats        — count adjacent pairs
    2. merge_pair       — replace every occurrence of a pair with a new id
    3. BPETokenizer.train
    4. BPETokenizer.encode
    5. BPETokenizer.decode

Spec: see README.md. Tests: see tests/. Do not change the public signatures.
Standard library only.
"""

from __future__ import annotations


def get_stats(token_ids: list[int]) -> dict[tuple[int, int], int]:
    """Count how often each adjacent pair occurs.

    Example: get_stats([1, 2, 1, 2, 1]) == {(1, 2): 2, (2, 1): 2}

    Hint: one pass, one dict. Overlapping pairs both count.
    """
    raise NotImplementedError("Your turn: get_stats")


def merge_pair(
    token_ids: list[int], pair: tuple[int, int], new_id: int
) -> list[int]:
    """Return a NEW list where every non-overlapping occurrence of `pair`
    is replaced by `new_id`, scanning left to right.

    Example: merge_pair([1, 2, 1, 2, 1], (1, 2), 99) == [99, 99, 1]

    Careful: after merging, the scan continues AFTER the new token —
    the new token does not start a new pair within this pass.
    """
    raise NotImplementedError("Your turn: merge_pair")


class BPETokenizer:
    """Byte-level BPE tokenizer (simplified GPT-2 style).

    Base vocab: the 256 byte values, token id == byte value.
    Learned merges create ids 256, 257, ... in the order they are learned.
    `self.merges` maps (left_id, right_id) -> new_id; its insertion order
    IS the merge rank, and encode() must respect it.
    """

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}

    # -- training ---------------------------------------------------------

    def train(self, text: str, vocab_size: int, min_frequency: int = 2) -> None:
        """Learn merges from `text` until the vocab reaches `vocab_size`.

        Steps (yours to write):
          1. Validate vocab_size (>= 256, else ValueError).
          2. token_ids = list(text.encode("utf-8"))
          3. Repeat (vocab_size - 256) times:
               - count pairs with get_stats
               - pick the most frequent pair (ties: your choice, but be
                 deterministic — the tests rely on it)
               - stop early if its count < min_frequency
               - record the merge, then apply merge_pair to token_ids

        Think before coding: why do we re-count from scratch each round
        instead of updating counts incrementally? What does that cost?
        (That trade-off belongs in your post.)
        """
        raise NotImplementedError("Your turn: train")

    # -- inference --------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode `text` to token ids using the learned merges.

        Start from UTF-8 bytes. Then, repeatedly: find the pair present in
        the sequence whose merge rank is LOWEST (learned earliest), merge
        all its occurrences, repeat until no pair in the sequence has a
        learned merge.

        The naive way — applying merges one full pass at a time in dict
        order — also works and is easier to reason about first. Get it
        correct, benchmark, then optimise if time remains.
        """
        raise NotImplementedError("Your turn: encode")

    def decode(self, token_ids: list[int]) -> str:
        """Decode token ids back to text. Must be the exact inverse of
        encode(): decode(encode(s)) == s for any string s.

        You will need a way to turn a token id back into its byte sequence:
        ids < 256 are their own byte; a merged id is the concatenation of
        its two children's byte sequences. Build that mapping recursively
        (or iteratively) from self.merges — and think about when to build
        it: every decode() call, or once? Decode the bytes with UTF-8,
        errors="replace" is NOT acceptable here — roundtrip must be exact
        for valid token sequences produced by encode().
        """
        raise NotImplementedError("Your turn: decode")
