# Target 3, in plain words

One page. Two separate problems. Do not mix them.

## The setup

Training keeps a **score sheet**: how many times each pair of tokens
sits side by side. Right now, `train()` rebuilds that sheet from
scratch every round. That is slow, but simple and correct.

Target 3 asks: keep the sheet and **update only what changed**
after each merge, instead of recounting everything.

## Problem 1 — the counts go wrong

When you merge a pair, only the pairs **touching that spot** can
change. Pairs far away are untouched.

The trap: looking for neighbours in the **old list**. Pieces of the
old list disappear as merges happen, so you can end up fixing scores
for pairs that no longer exist, and missing pairs that do.

Small example — list `a b a b`, merging `(a, b)` into `N`:

- Old pairs: `(a, b)` twice, `(b, a)` once.
- New list: `[N, N]`.
- New pairs: `(N, N)` once. That is the whole sheet.

Notice: the pair in the middle, `(b, a)`, went away **once**, even
though it touches both merges. And `(N, N)` is a pair your old-list
view can never see, because `N` is not in the old list.

**You do not have to change `merge_pair` to fix this.** Merge first
with the plain `merge_pair`. Then walk the old list and the new list
side by side. Every `N` in the new list marks a spot where a merge
happened. Fix the scores for just those spots, inside `train()`.

Test for Problem 1, and only Problem 1: `part2/stats_checker.py`.
One merge, then compare your sheet against a fresh count of the
new list. They must match exactly — including this rule: a pair
whose count reaches zero leaves the sheet.

## Problem 2 — ties can pick a different winner

Sometimes two pairs tie for the top score. The code picks the one
that comes **first in the dictionary**.

- Counting from scratch: "first in the dictionary" lines up with
  "first in the text."
- Keeping a running sheet: pairs stay in the order they were first
  written down, maybe many rounds ago. So a tie can go the other way.

Different winner in one round → different merges from then on.
That is not a counting bug. It is a **choice**:

1. Also track where each pair first appears. Fiddly.
2. Keep the running counts, but do a quick scan each round just to
   fix the order. Gives back some of the saved time.
3. Accept the new tie rule and write it up honestly. Real fast
   tokenizers do this — but it breaks this project's
   "same output as v1" rule, so it means calling the stretch goal
   done-with-a-note.

## Order to work in

1. Problem 1 only, with the checker, until it passes every time.
2. Full equivalence tests. If something still differs, expect it
   to be a tie — that is Problem 2 showing up, not Problem 1 back
   from the dead.
3. Pick a tie choice with a clear head, and write down which one
   you picked and why. That write-up is part of the post either way.

Reminder on scope: targets 1 and 2 are already in — decode is about
20× faster. Target 3 is the stretch. A clear "here is why this last
piece is hard" is a finished Part 2, with or without option 1 above.
