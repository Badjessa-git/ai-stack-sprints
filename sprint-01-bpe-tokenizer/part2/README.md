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

## Why this part exists

Part 1 proved you can build the thing. But "correct" is only half of
engineering — the other half is "fast enough, and provably still correct."
That is this part.

It also mirrors what the industry actually did with this exact algorithm.
BPE tokenization runs on *every* LLM request, so production tokenizers
are not naive Python: OpenAI's tiktoken is a Python library with a Rust
core, and Hugging Face's "fast" tokenizers are bindings around a Rust
crate. Your v1 → v2 pass replays that story at small scale: same
algorithm, same outputs, engineered until it's fast.

And it sets up the rest of the series. The recurring trick in the AI
stack is **never recompute what didn't change**. Incremental pair counts
here are the same idea as the KV cache in Sprint 6 (don't recompute
attention for tokens already processed) and the caching you'll meet in
Sprint 2's autograd engine. Part 2 is where you learn that pattern with
training wheels on.

## What you're actually learning

- **Measure before you optimize.** Your v1 baselines (~140 KB corpus):
  train 5.6s, encode 3.9s, decode 0.37s per 30k tokens. Decode *feels*
  like the dumb code, but it's a single-digit share of the total —
  optimizing it first would barely move anything. That's Amdahl's law
  in practice: a speedup only matters in proportion to the share of
  time it removes.
- **Optimization under an identical-output rule.** Production performance
  work happens against golden outputs and regression tests — exactly
  like the equivalence suite here. The skill is changing the *cost*
  without changing the *semantics*, including sneaky semantics like
  tie-breaking order.
- **Trading memory for time.** The decode table spends a little memory
  (one bytes object per token id) to delete repeated work. You'll meet
  this trade everywhere: embeddings, caches, materialized views.
- **Algorithmic vs. constant-factor wins.** Max-instead-of-sort keeps the
  same scan but does less work per round; incremental pair counts change
  the complexity of training itself. Knowing which kind of win you're
  getting tells you where the ceiling is.

## Discuss with Cleo during review (before writing the post)

Standing rule since Part 1: the post only asks you to explain things
we've actually covered together. Our review-session checklist:

- [ ] Why production tokenizers are Rust (tiktoken, HF tokenizers), and
      what that says about where tokenization sits in inference latency
- [ ] Pretokenization: GPT-2/GPT-4 split text with a regex *before* BPE
      merges run — a step your v1 skips entirely; what it changes
- [ ] Where the remaining time goes after targets 1–2 (profile the code;
      don't guess)
- [ ] The vocabulary-size trade-off at production scale — Part 1's
      discussion, one level deeper (embedding tables, rare tokens,
      multilingual effects)

## Resources

- Sennrich, Haddow & Birch, *Neural Machine Translation of Rare Words
  with Subword Units* (2016) — the paper that brought BPE to NLP:
  https://arxiv.org/abs/1508.07909
- Andrej Karpathy, *Let's build the GPT Tokenizer* (video) — builds a
  GPT tokenizer from scratch and tours how tiktoken differs:
  https://www.youtube.com/watch?v=zduSFxRajkE — companion code:
  https://github.com/karpathy/minbpe
- Hugging Face, *Summary of the tokenizers* — BPE, WordPiece, and
  Unigram compared, including byte-level BPE:
  https://huggingface.co/docs/transformers/v4.47.1/tokenizer_summary

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
