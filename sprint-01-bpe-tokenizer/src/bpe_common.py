"""BPE helpers shared by Part 1 (src/tokenizer.py) and Part 3
(part3/regex_tokenizer.py). Moved here unchanged from src/tokenizer.py, which
re-imports them, so `from src.tokenizer import get_stats, merge_pair` still
works everywhere.
"""

from __future__ import annotations


def get_stats(token_ids: list[int]) -> dict[tuple[int, int], int]:
    """Count how often each adjacent pair occurs.

    Example: get_stats([1, 2, 1, 2, 1]) == {(1, 2): 2, (2, 1): 2}

    Hint: one pass, one dict. Overlapping pairs both count.
    """

    # Initial approach, two pointers at i and i+1
    # if pair (i, i+1) exist in the dictionay, increase the count
    # if first time then set it to 1 in the hashmap

    result = {}
    for i in range(len(token_ids) - 1):
        pair = (token_ids[i], token_ids[i + 1])
        if pair in result:
            result[pair] = result[pair] + 1
        else:
            result[pair] = 1

    return result


def merge_pair(token_ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
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

        current_pair = (token_ids[i], token_ids[i + 1])
        if current_pair == pair:
            result.append(new_id)
            i += 2  # non-overlapping pair so we want to skip
        else:
            result.append(token_ids[i])
            i += 1
    return result
