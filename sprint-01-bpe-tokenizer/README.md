# Sprint 01 — Build a BPE Tokenizer From Scratch

Every other weekend · 48-hour manual coding sprint · Sprint 1 of "Rebuild the Modern AI Stack"

## The mission

Implement a **byte-level Byte Pair Encoding (BPE) tokenizer** — the same family of
algorithm that turns text into tokens for GPT-style models — in pure Python,
standard library only. No `tiktoken`, no `tokenizers`, no regex doing the work
for you. You write the algorithm; the scaffold (this repo) only gives you the
spec, the tests, and the benchmark.

By Sunday night you should be able to explain, from having built it:
- why tokenization is a compression algorithm in disguise
- why LLMs miscount letters in words (the strawberry problem)
- how vocab size trades off against sequence length

## The rules (agreed)

1. **You write the core implementation** — everything in `src/tokenizer.py`.
2. AI may scaffold, explain concepts, review your code, and ask debugging
   questions. It does not paste the core solution.
3. Turn off / don't use autocomplete AI for `src/tokenizer.py`. Struggling with
   the pair-counting loop *is* the workout.
4. Tests are the spec. If a test and this README disagree, the test wins —
   flag it rather than silently changing the test.

## The spec

Byte-level BPE, simplified GPT-2 style:

- The base vocabulary is the **256 possible byte values** (token ids `0–255`).
  Text is first encoded to UTF-8 bytes; each byte starts as its own token.
- **Training** repeatedly finds the most frequent adjacent pair of tokens in
  the corpus and merges it into a new token (id `256`, `257`, …) until the
  requested vocab size is reached or no pair occurs at least `min_frequency`
  times.
- **Encoding** starts from bytes and applies the learned merges **in the order
  they were learned** (merge rank matters — applying them out of order gives
  a different, wrong tokenization).
- **Decoding** maps tokens back to bytes and decodes UTF-8. Encode → decode
  must be **lossless for any string**, including emoji and accented characters.

### Public API (in `src/tokenizer.py`)

```python
class BPETokenizer:
    merges: dict[tuple[int, int], int]   # (left_id, right_id) -> new_id, in learned order

    def train(self, text: str, vocab_size: int, min_frequency: int = 2) -> None
    def encode(self, text: str) -> list[int]
    def decode(self, token_ids: list[int]) -> str
```

- `train` with `vocab_size < 256` must raise `ValueError`.
- `encode("")` returns `[]`; `decode([])` returns `""`.
- Encoding text after training on a repetitive corpus must produce **fewer
  tokens than the text has bytes** (that's the compression check).

### Out of scope (don't build these)

- Special tokens (`<|endoftext|>` etc.), pre-tokenization regex splitting,
  saving/loading vocab files, speed in C. Stretch goals only if the core is
  done early — see the bottom of this README.
  (Out of scope for Part 1 only. Speed became Part 2 and regex splitting
  became Part 3. See "Follow-up parts" below.)

## Weekend shape

**Saturday — naive v1**
- [ ] `get_stats` / pair counting working
- [ ] One merge step working by hand on `"aaabaaab"` — do this on paper first
- [ ] `train` loop complete; `test_training.py` passing
- [ ] `encode` / `decode` roundtrip passing

**Sunday — make it correct, then make it fast, then write**
- [ ] All tests passing (`make test`)
- [ ] Run `make benchmark` — save the output, you'll quote it in the post
- [ ] Edge cases: empty string, single character, text with bytes/pairs never
      seen in training
- [ ] Fill in `POST.md` — your words first, editor second

## Run it

```bash
make test        # python3 -m unittest discover -s tests -v
make benchmark   # trains on data/corpus.txt and reports compression + timing
```

Python 3.9+, standard library only. Nothing to `pip install`.

## Benchmark targets (on `data/corpus.txt`, vocab_size=512)

- Compression: tokens < 75% of the corpus byte count
- Roundtrip: `decode(encode(corpus)) == corpus`, exactly
- Training time: whatever you get — report it honestly in the post

## Stretch (only if core is done by Sunday noon)

1. Vocab-size sweep: train at 256 / 384 / 512 / 1024, chart tokens-vs-vocab
2. Add one special token that is never split
3. Compare your token count on a paragraph against your intuition for what
   `tiktoken` would produce — and write about the gap

## The post

Template is in `POST.md`. Working title: **"Why LLMs can't count the r's in
strawberry — a tokenizer, built by hand"**. Target 900–1,500 words, publish
wherever you like; the draft lives in this repo either way.

## Follow-up parts

The core above is Part 1. Two follow-ups went deeper, each with its own
README, tests and post notes:

| Part | Folder | What it does | Result |
|------|--------|--------------|--------|
| 2: speed | [`part2/`](part2/) | Makes `src/tokenizer.py` fast without changing a single output. Equivalence tests compare it against the frozen v1 (`part2/tokenizer_v1.py`) | See `part2/RESULTS.md` |
| 3: regex split | [`part3/`](part3/) | GPT-2-style pre-split: text is cut into word-like chunks first, and merges never cross a chunk boundary. Trains from chunk counts, so it learns from the full 100M-character WikiText-103 file | Same 2.05 bytes/token as Part 1 at vocab 512, but 0 merges end in a space (Part 1: 126). 3.41 bytes/token at 4096 |

Code shared by Part 1 and Part 3 (`get_stats`, `merge_pair`) lives in
`src/bpe_common.py`.

```bash
python3 -m unittest part2/test_equivalence.py -v
python3 -m unittest part3/test_regex_tokenizer.py -v   # 34 tests
python3 part3/compare_tokenizers.py                    # Part 1 vs Part 3, side by side
```

Sprint 03 trains on both tokenizers: Part 1 in v1/v2, Part 3 in the v3 grid.

## What AI wrote

- **Scaffold:** this README, the tests, `benchmarks/`, the `POST.MD`
  outline, the `part2/` and `part3/` READMEs and test files,
  `part3/compare_tokenizers.py`. <!-- TODO(you): confirm which part2/ tools
  (compare.py, profiler.py, stats_checker.py) were scaffolded vs yours -->

- **Part 3:** the split pattern `SPLIT_PATTERN` (written by Claude on
  request, including the `_` fix), plus the plumbing in
  `regex_tokenizer.py`: `train()`, `encode()` with its chunk cache,
  `decode()`, `save`/`load`, and progress logging.
- **Refactor:** moving `get_stats`/`merge_pair` into `src/bpe_common.py`,
  with no change in behaviour.
- **Hand-written:** `src/tokenizer.py` (Part 1 and the Part 2 speed-ups),
  and in Part 3 `word_counts`, `get_stats_weighted`, `train_from_counts`
  and `_encode_chunk`.
