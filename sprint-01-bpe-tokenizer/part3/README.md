# Sprint 1, Part 3 — Regex pre-split BPE

Part 1 made BPE **correct**. Part 2 made it **fast**. Part 3 makes it
**tokenize like a real LLM tokenizer**: split text into word-like chunks with a
regex first, then run BPE inside each chunk.

## Why this part exists

Sprint 3's transformers showed what Part 1's tokenizer gets wrong. It runs
BPE over the raw byte stream, so the most frequent pairs often straddle a
word boundary. With vocab 512 on WikiText-103:

- 126 of the 256 merges **end** in a space ("e ", "s ", "the "), and only
  3 **start** with one.
- A prompt with a trailing space ("Albert Einstein was a ") got tokenized
  differently from one without, and the model's output changed.
- The v2 models write real phrases for ~20 tokens and then invent words
  ("somovolobobid"), built from 2-byte pieces that don't respect words.

GPT-2 fixed this with one idea: **merges may not cross a chunk boundary**.
Every token then becomes a piece of one word (carrying the word's leading
space), a number, a punctuation run, or whitespace.

## The split pattern

GPT-2's pattern (it needs the third-party `regex` package for `\p{L}`):

```
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

Read it left to right. The first alternative that matches wins:

| Piece | Matches | Example |
|---|---|---|
| `'s\|'t\|…` | English contractions | `don't` → `don` `'t` |
| ` ?\p{L}+` | a word, with at most one leading space | ` world` |
| ` ?\p{N}+` | a number, with at most one leading space | ` 1994` |
| ` ?[^\s\p{L}\p{N}]+` | a punctuation run | ` @-@`, `!!!` |
| `\s+(?!\S)` | whitespace, but not the last space before a word | `x···y` → `··` then ` y` |
| `\s+` | any leftover whitespace | |

Your job is the **stdlib `re` version**. `re` has no `\p{L}`, so you have to
build "a letter" and "not a letter, digit or space" out of `\w`, `\W`, `\d`
and `\s`. The tests pin the behaviour, and one of them is designed to catch
the most common mistake. Every character must land in some chunk:
`"".join(pre_tokenize(s)) == s`.

## Who writes what

| Part | Who | Where |
|---|---|---|
| `SPLIT_PATTERN`, `pre_tokenize` | **you** | `regex_tokenizer.py` |
| `word_counts` | **you** | `regex_tokenizer.py` |
| `train_from_counts` | **you** | `regex_tokenizer.py` |
| `_encode_chunk` (lowest-rank-first) | **you** | `regex_tokenizer.py` |
| encode cache, decode, save/load | scaffold | `regex_tokenizer.py` |
| corpus caching, `--tokenizer bpe-regex` in training | scaffold | `sprint-03-transformer/benchmarks/train_v3.py` |
| tokenizer comparison | scaffold | `part3/compare_tokenizers.py` |

Same rules as Part 1: you write the four stubs. AI can explain and review,
but it doesn't paste the solution.

## Order of work

From `sprint-01-bpe-tokenizer/`:

```bash
python3 -m unittest part3/test_regex_tokenizer.py -v
```

The test classes are numbered in the order to make them pass:

1. **Test1PreTokenize.** The pattern. Run it on a few strings in a REPL first.
2. **Test2WordCounts.** One line with `collections.Counter`, or a dict loop.
3. **Test3Train.** BPE again, but over `{chunk: count}` instead of one long
   byte list. Weight each pair by its chunk's count. Ties go to the smallest
   pair as a tuple.
4. **Test4EncodeChunk.** Not "apply every merge in order" (that's the test's
   slow oracle). Instead: look at the pairs in the chunk, pick the one with
   the lowest merge id, merge it, repeat.
5. **Test5Tokenizer.** Should pass on its own once 1–4 do.

## Then measure it

```bash
python3 part3/compare_tokenizers.py            # Part 1 @512 vs Part 3 @512 and @4096
```

This prints training time, bytes per token, and how each tokenizer splits
"Albert Einstein was a " (with and without the trailing space). The numbers
go into the post.

## Then train with it (Sprint 3)

From `sprint-03-transformer/`:

```bash
python3 benchmarks/train_v3.py --run-name d512-n4-nodrop-rx4k-wiki103 \
  --tokenizer bpe-regex --bpe-vocab 4096 --corpus data/wiki103-100m.txt \
  --optimizer adam --epochs 1 --batch-size 16 --d-model 512 --n-layer 4 \
  --block-size 64 --dropout 0 --eval-every 2000 --checkpoint-every 2000 --sample 0
```

The first run learns the merges from the whole training split (~30 min with
the straightforward implementation) and caches the merges and the encoded
corpus next to it. Later runs load the cache in seconds. Compare runs on
**val bpb**, not val loss: a 4k token covers more text than a 512 token, so
loss per token isn't comparable.

## Questions for the post

- How many of your 4k merges end in a space, and how many start with one?
  Compare with Part 1's 126 vs 3.
- Bytes per token at 512 and 4096. How much shorter are the sequences?
  What does that mean for a model with a 64-token context?
- Recounting every pair each round costs ~0.5 s here. Part 1's recount on
  100M bytes would cost far more. Why does `word_counts` make that cheap?
- The embedding and output layers grow 8× (512 → 4096 rows). At d512, how
  many parameters is that? Is it worth it?
- Stretch: make `train_from_counts` incremental (Part 2's trick). What's
  the speedup?
