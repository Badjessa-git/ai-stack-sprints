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

MIN_VOCAB_SIZE = 256 # max size for base vocabulary

def get_stats(token_ids: list[int]) -> dict[tuple[int, int], int]:
    """Count how often each adjacent pair occurs.

    Example: get_stats([1, 2, 1, 2, 1]) == {(1, 2): 2, (2, 1): 2}

    Hint: one pass, one dict. Overlapping pairs both count.
    """

    # Initial approach, two pointers at i and i+1
    # if pair (i, i+1) exist in the dictionay, increase the count
    # if first time then set it to 1 in the hashmap

    result = {}
    for i in range(len(token_ids) -1):
        pair = (token_ids[i], token_ids[i+1])
        if pair in result:
            result[pair] = result[pair] + 1
        else:
            result[pair] = 1

    return result


def merge_pair(
    token_ids: list[int], pair: tuple[int, int], new_id: int
) -> list[int]:
    """Return a NEW list where every non-overlapping occurrence of `pair`
    is replaced by `new_id`, scanning left to right.

    Example: merge_pair([1, 2, 1, 2, 1], (1, 2), 99) == [99, 99, 1]

    Careful: after merging, the scan continues AFTER the new token —
    the new token does not start a new pair within this pass.
    """
    result = []
    size = len(token_ids)
    i = 0
    while i <= (size - 1):
        if i == size - 1:
            result.append(token_ids[i])
            return result

        current_pair = (token_ids[i], token_ids[i+1])
        if current_pair == pair:
            result.append(new_id)
            i += 2 # non-overlapping pair so we want to skip 
        else:
            result.append(token_ids[i])
            i += 1
    return result

class BPETokenizer:
    """Byte-level BPE tokenizer (simplified GPT-2 style).

    Base vocab: the 256 byte values, token id == byte value.
    Learned merges create ids 256, 257, ... in the order they are learned.
    `self.merges` maps (left_id, right_id) -> new_id; its insertion order
    IS the merge rank, and encode() must respect it.
    """

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.__byte_lookup_table: dict[int, tuple[int, int]] = {}

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

        if vocab_size < MIN_VOCAB_SIZE:
            raise ValueError("Vocab size bigger than alloted value")

        self.merges = {}
        self.__byte_lookup_table = {}
        
        i = 0
        id = 256
        token_ids = list(text.encode("utf-8"))

        # Optimization instead of sorting the algorithm, we walk through it once and keep track of the 
        while i < vocab_size - MIN_VOCAB_SIZE:
            stats = get_stats(token_ids=token_ids)
            if not stats:
                break;
            
            (left, right), value = self.__find_max_frequency_pair(stats)
            if value < min_frequency:
                break

            self.merges[(left, right)] = id
            token_ids = merge_pair(token_ids=token_ids, pair=(left, right), new_id=id)
            self.__byte_lookup_table[id] = (left, right)
            i += 1 # increase pointer
            id += 1 # new ids

    def __find_max_frequency_pair(self, stats: dict[tuple[int, int], int]) -> tuple[tuple[int, int], int]:
        """
        Return the max frequency pair found in the token_ids list, we've used sorted before for this with 
        has Big O(nlogn), but we can easily find this by doing it one pass improving the performance to 
        Big 0(n)
        """

        if len(stats) == 0:
            raise ValueError("Empty Dictionary provided as input")

        # The pair will never be [(0, 0), 0]
        max_pair = ((0, 0), 0)
        for pair in stats:
            value = stats.get(pair)
            current_max_value = max_pair[1]
            if value > current_max_value:
                max_pair = (pair, value)

        if max_pair == ((0, 0), 0):
            raise ValueError("Invalid dictionary provided as input, cannot contains zero")

        return max_pair


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
        token_ids = list(text.encode("utf-8"))
        size = len(self.merges)
        iterator = iter(self.merges.items())
        for i in range(size):
            (left, right), new_id = next(iterator)
            token_ids = merge_pair(token_ids=token_ids, pair=(left, right), new_id=new_id)
        return token_ids
            
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

        # so we need to replace each encoding with its pair in the text, until we get the original 
        # token list and then call decode with UTF 8 to get the text value
        # [2, 3, 2, 3, 1] => [2, 3] = 2, 1 = 1 => [256, 256, 1]

        decoded_list = []
        for i in range(len(token_ids)):
            decoded_list.extend(self.__decode_single(token_id=token_ids[i]))
        
        byte_array = bytes(decoded_list)
        return byte_array.decode("utf-8")

    def __decode_single(self, token_id: int) -> list[int]:
        """
        Recursively decode the token ids and return a list to the top level function
        which builds the string to be returned
        """
        if (token_id < MIN_VOCAB_SIZE):
            return [token_id]
        
        if token_id in self.__byte_lookup_table:
            (left, right) = self.__byte_lookup_table[token_id]
            left_decode = []
            right_decode = []
            if left >= MIN_VOCAB_SIZE:
                left_decode.extend(self.__decode_single(left))
            else:
                left_decode = [left]
            
            if right >= MIN_VOCAB_SIZE:
                right_decode.extend(self.__decode_single(right))
            else:
                right_decode = [right]

            result = left_decode + right_decode
            return result
        else:
            raise RuntimeError("ID above 255 not available in lookup table, there was an error during training")
            



