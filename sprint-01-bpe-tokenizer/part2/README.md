# Sprint 1, Part 2 — Optimization pass

Part 1 was about getting BPE **correct**. Part 2 is about making it **fast
without changing what it does**.

## The one rule

> **Outputs must stay byte-for-byte identical to v1.**
> Same merges (same pairs, same ids, same order), same encodings, same
> decodings — for every input. Only the speed may change.

That is what makes this an optimization task instead of a rewrite. The
equivalence tests in this folder compare your current `src/tokenizer.py`
against a frozen snapshot of your v1 (`tokenizer_v1.py`, commit `638aeaa`)
and they must stay green the whole time — along with the original 15 tests
(`make test` from the sprint folder).

Unlike Part 1, these tests pass *before* you start. Your job is to keep
them passing while the benchmark numbers move.

## The targets (in order)

### 1. `train`: stop sorting the whole stats dict every round

Your v1 sorts every pair by count just to read off the single largest one.
Replace the full sort with a direct maximum lookup.

Watch the tie-breaking: v1's stable sort keeps the *first-seen* pair in
front on ties, and a first-wins maximum scan over the dict (which is in
first-seen order) does the same. The equivalence tests will catch it if
your version breaks ties differently.

### 2. `decode`: build the token → bytes table once

Your v1 decode re-derives everything on every call: it rebuilds
`set(self.merges.values())` inside the recursion, linearly scans
`self.merges` to find which pair produced a given id, and re-expands shared
subtrees again and again.

Build a table `id -> bytes` **once** and reuse it:

- ids `< 256` map to their own single byte.
- a merged id's bytes are the concatenation of its two children's bytes —
  and because a merge's children always have *smaller* ids than the merge
  itself, you can build the table in id order and every child is already
  there when you need it.
- cache the table on the instance (build it lazily on first decode, or at
  the end of `train` — think about what happens if someone trains again
  after decoding).

`decode` then becomes a lookup per token plus one join.

### 3. Stretch: `train` with incremental pair counts

v1 recounts every pair from scratch after each merge: M merges × a full
O(N) recount, plus a full `merge_pair` scan. But a merge only changes the
pairs **touching the merge sites** — pairs elsewhere in the sequence keep
their counts. Maintain the counts as you go: when you merge a pair at some
position, decrement the pairs that disappeared around it and increment the
new pairs that formed. This is the difference between "recompute the
world" and "update what changed."

Only attempt this after targets 1–2 are green and benchmarked.

### Optional: `encode`

v1 makes one full pass over the sequence per merge rule. An alternative
shape: repeatedly find the *lowest-rank* pair present in the sequence and
merge all of its occurrences, until no known pair remains. Same output,
fewer wasted passes. Optional — only if time remains.

## Workflow

From the sprint folder (`sprint-01-bpe-tokenizer/`):

```bash
# correctness — original suite + equivalence suite
make test
python3 -m unittest discover -s part2 -p "test_*.py"

# speed — frozen v1 vs your current implementation, larger corpus
python3 part2/compare.py
```

`compare.py` first sanity-checks that both implementations agree, then
times train / encode / decode on the sprint corpus repeated 30×
(~140 KB — the 4.6 KB corpus from Part 1 is too small to measure anything
but noise). Record the before/after numbers in `part2/RESULTS.md`.

## Rules (same as Part 1)

- You write the implementation. Only `src/tokenizer.py` changes.
- Do not edit `tokenizer_v1.py`, the tests, or the benchmark harness.
- Public API and signatures unchanged. Standard library only.
- If an optimization changes any output, it's not an optimization — it's
  a bug. The equivalence tests arbitrate.

## Done when

- [ ] All 15 original tests pass
- [ ] All equivalence tests pass
- [ ] `compare.py` shows the improvement; numbers recorded in `RESULTS.md`
- [ ] Part 2 post drafted from `part2/POST.md`
