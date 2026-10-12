"""Target 3 checker — scaffolding only, not part of the tokenizer.

WHAT THIS DOES
    It checks ONE question, in plain terms:

        After one merge, does your updated score sheet (stats)
        match a fresh count of the new list?

    It does not train anything. It does not fix your counts.
    The one function you fill in is marked TODO below.

HOW TO USE IT
    1. Fill in `update_stats_after_merge` below. That is the only
       place your idea goes. Do not change `merge_pair` — this
       checker uses the plain `merge_pair` from src/tokenizer.py,
       the same way the "do it inside train()" idea works:
       merge first, then fix the score sheet using the old list
       and the new list side by side.
    2. Run:  python3 part2/stats_checker.py
    3. Read the first FAIL. It says, in plain words, which pair
       is wrong and by how much. Fix that one case. Run again.

THE SHAPE OF THE ONE MERGE
    old list:  [97, 98, 97, 98]          (97 = a, 98 = b)
    pair:      (97, 98)  ->  new id 256
    new list:  [256, 256]

    Pairs that were there and are now gone, pairs that are new —
    only the spots touching a 256 in the new list can change.
    Everywhere else, the old list and the new list are the same.

Standard library only.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer import get_stats, merge_pair


# ---------------------------------------------------------------------
# TODO — THIS IS YOUR PART. Everything else in this file is harness.
# ---------------------------------------------------------------------

def update_stats_after_merge(
    old_ids: list[int],
    new_ids: list[int],
    stats: dict[tuple[int, int], int],
    pair: tuple[int, int],
    new_id: int,
) -> dict[tuple[int, int], int]:
    """Return the score sheet for `new_ids`, starting from `stats`.

    `stats` is the correct score sheet for `old_ids` (a fresh
    get_stats(old_ids)). Work out the new sheet by looking only
    at the spots where a merge happened — every `new_id` in
    `new_ids` marks one of those spots.

    Rules a fresh count follows, so your sheet must too:
      - a pair with count 0 is not in the sheet at all
      - counts are never negative

    Watch the shared-boundary case: old [97, 98, 97, 98] merging
    (97, 98) gives new [256, 256]. The middle pair of the old list,
    (98, 97), is gone once — not twice — and the new pair
    (256, 256) appears once.

    Raise / return: return a NEW dict, or update and return `stats`.
    """
    raise NotImplementedError("your update logic goes here")


# ---------------------------------------------------------------------
# Harness — reporting only. No counting logic of yours lives here.
# ---------------------------------------------------------------------

def plain_diff(
    mine: dict[tuple[int, int], int], fresh: dict[tuple[int, int], int]
) -> list[str]:
    """Say, pair by pair, how two score sheets differ."""
    lines = []
    for pair in sorted(set(mine) | set(fresh)):
        got, want = mine.get(pair), fresh.get(pair)
        if got == want:
            continue
        if want is None:
            lines.append(f"  pair {pair}: you have {got}, fresh count does not list it at all")
        elif got is None:
            lines.append(f"  pair {pair}: fresh count has {want}, yours does not list it")
        else:
            lines.append(f"  pair {pair}: you have {got}, fresh count has {want}")
    return lines


def check_one(old_ids: list[int], pair: tuple[int, int], new_id: int, label: str) -> bool:
    new_ids = merge_pair(old_ids, pair, new_id)
    fresh = get_stats(new_ids)
    try:
        mine = update_stats_after_merge(old_ids, new_ids, get_stats(old_ids), pair, new_id)
    except NotImplementedError:
        print(f"SKIP  {label}: update_stats_after_merge is not filled in yet.")
        return False
    if mine == fresh:
        print(f"PASS  {label}: {old_ids} -> {new_ids}")
        return True
    print(f"FAIL  {label}")
    print(f"  old list: {old_ids}")
    print(f"  pair merged: {pair} -> {new_id}")
    print(f"  new list: {new_ids}")
    for line in plain_diff(mine, fresh):
        print(line)
    return False


def main() -> int:
    a, b, c, n = 97, 98, 99, 256
    fixed_cases = [
        ([a, b, a, b], (a, b), n, "two merges side by side (the hard one)"),
        ([a, b, a, b, a, b], (a, b), n, "three merges in a row"),
        ([a, b, c], (a, b), n, "one merge at the start"),
        ([c, a, b], (a, b), n, "one merge at the end"),
        ([a, b, c, a, b], (a, b), n, "two merges with a gap"),
        ([c, c, c], (a, b), n, "pair does not appear — sheet should not change"),
    ]

    results = [check_one(ids, pair, nid, label) for ids, pair, nid, label in fixed_cases]
    if not all(results):
        print("\nStop here. Make the fixed cases pass before the random ones.")
        return 1

    rng = random.Random(20261011)
    for trial in range(500):
        ids = [rng.choice([a, b, c]) for _ in range(rng.randint(0, 12))]
        stats = get_stats(ids)
        if not stats:
            continue
        pair = rng.choice(sorted(stats))
        if not check_one(ids, pair, n, f"random trial {trial}"):
            print("\nFirst random failure is above. Shrink it by hand until it is tiny, then trace it.")
            return 1

    print("\nAll fixed cases and 500 random trials match a fresh count.")
    print("Problem 1 (the counts) is done. Ties (Problem 2) are a separate question — this checker does not test them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
