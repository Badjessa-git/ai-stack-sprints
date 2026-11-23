# Sprint 3 post notes: what four versions taught me (V1 → V4)

Raw material for the post, pulled from the logs and reports in this
folder. Every number here has a source file next to it. `POST.md` is the
outline. This file is the evidence.

## The four versions

The post's names don't match the folder names, so here is the map:

| Post name | What it is | Code | Logs |
|---|---|---|---|
| **V1 scalar** | Sprint 2 `Value` engine, one number at a time | `src/scalar/`, `train.py --engine scalar` | README "What to expect" |
| **V2 tensor** | The NumPy engine, first scale-up to WikiText-103 | `src/numpy/`, `train.py` | `runs/legacy/` |
| **V3 bug fixes** | The Nov 19 review fixes | `src/numpy_v2/`, `train_v2.py` | `runs/v2/` |
| **V4 regex** | V3 plus the Sprint 1 Part 3 tokenizer | `train_v3.py` | `runs/v3/` |

## The headline table

Held-out bits per byte (lower is better). Same 100M characters of
WikiText-103, one epoch.

| | V2 tensor | V3 bug fixes | V4 regex |
|---|---|---|---|
| Best model | d256-n2, 2.193 | d512-n4, 1.621 | d512-n4 4k vocab, **1.439** |
| d1024-n2 (26M params) | 2.756, **79.7 hours** | 1.742, **10.2 hours** | — |
| Bigger is better? | No: d1024 is worst | Mostly, but depth beats width | Yes, and vocab matters more |

Sources: `reports/2026-11-20-v2-grid.md`, `reports/2026-11-22-v3-grid.md`,
and the `Ns total` lines of `runs/legacy/train-d1024-n2-wiki103-drop01.log`
and `runs/v2/d1024-n2-drop01-wiki103/train.log`.

**One sentence for the post:** the same 26M-parameter model went from
2.756 to 1.742 bits per byte and from 80 hours to 10. The architecture
didn't change. Only the bugs around it did.

---

## V1 scalar: correct but impossibly slow

- 5,354 parameters, d_model 16, 1 layer, vocab 58 (characters).
- **2.1 s per training step, 3.8 tokens/sec** (README, "What to
  expect"). Forward ~2.0 s, backward ~0.08 s.
- Every operation is a Python object holding one float. Attention at T=8
  is fine. At T=64 it's hopeless.
- Moving to the NumPy engine was about **4,000× faster** on a bigger
  config (`bench_engines.py` docstring). The small config in
  `bench_engines.py` is the version you can reproduce.
- `bench_engines.py` checks that both engines reach the same loss on the
  same seed. Same math, different representation. If the losses
  disagree, one of the engines has a bug.

**What it taught me:** autograd is the same idea at any scale. Speed
comes from representation: one array op instead of a million Python
objects.

TODO(you): a quote of a V1 sample, from your terminal or Sprint 2 notes.
No scalar-engine log is saved in `runs/`.

## V2 tensor: fast enough to find out everything else was wrong

**1. The model started out confidently wrong.** A model that knows
nothing should start at loss ln(vocab) = ln(512) = 6.24. The real
step-0 losses were:

| Model | step-0 loss | Should be |
|---|---|---|
| d256-n2 | 27.3 | 6.24 |
| d512-n4 | 38.0 | 6.24 |
| d1024-n2 | 59.0 | 6.24 |

(The first `step 0` line of each `runs/legacy/train-*-wiki103-*.log`.)

The wider the model, the worse the start. That's why **d1024 came out
worst of all** in V2 (2.756, against 2.193 for d256). The initialization
was too big, so outputs grew with width. Bigger models had more damage
to undo before they could learn anything.

**2. One window per step.** d1024 took 758,091 steps and 286,853 s
(79.7 hours) for a single epoch. There was no real batching.

**3. Memory leaked through closures.** Each Tensor's `_backward` closure
referenced the Tensor, which made a reference cycle. Dead graphs were
only freed when Python's cycle collector ran, and it rarely did. d256
peaked at 8.4 GB and d1024 sat at ~11 GB. After the fix: 0.6 GB and
2.4 GB (`train_v2.py` docstring, item 8).

**4. The Mac had opinions.**
- Accelerate printed `divide by zero / overflow encountered in matmul`
  warnings on correct math (every legacy log starts with them).
- Accelerate also segfaulted the d128 run on Nov 16 when it spread tiny
  matmuls across threads. The fix pins BLAS to one thread (`train.py`
  header comment).

**5. No held-out set, so it memorized.** A greedy sample from the
3-epoch Shakespeare d256-n4 run:

> perked up in a glist'ring grief,
> And wear a golden sorrow.

That is line 72,908–72,909 of `data/corpus-combined.txt`, word for word
(Henry VIII). Without held-out text, the training loss called this
success. (`runs/legacy/train-d256-n4.log`, last lines.)

**6. The loss column lied across tokenizers.** The Nov 19 report ranks
models by loss: char models 2.40, BPE models 3.17–4.62
(`reports/2026-11-19-model-test-*.md`). Those numbers aren't comparable.
Loss is per token, and a BPE token covers about 2 characters. That's
why V3 switched to **bits per byte**.

**7. The trailing-space discovery (Nov 19).** The same model, prompted
with and without a space after "Albert Einstein was a":

| Prompt | Output starts |
|---|---|
| `"...was a"` | "was **aily** and sury of a TVicith", "was **ause** of both", "was **avy** and creased" |
| `"...was a "` | "was a cluding to a", "was a collection of internal of the battle" |

Without the space, the model reads "a" as the start of a word and
finishes it. The prompt has to look like the training text at the
token level, not just to a human reader.

**8. The BPE merges came from the first 200k characters only**
(`text[:200_000]`), so the vocabulary reflected whatever article came
first. V3 samples 200k characters spread across the corpus.

## V3 bug fixes: every V3 model beats every V2 model

The fixes from the Nov 19 review are each a flag, so each can be
ablated:
- init scaled by 1/√n_in, giving step-0 loss 6.43 for every size
  (`runs/v2/*/train.log`)
- batch 16
- cosine LR with warmup, plus gradient clipping
- held-out eval in bits per byte
- float32
- log-softmax loss
- gc after every step

**The controlled A/B** (d256-n2, char Shakespeare, the same 6,400
windows):

| | held-out bits/char | time |
|---|---|---|
| `train.py` (V2) | 4.29 | 127 s |
| `train_v2.py` (V3) | **3.08** | **30 s** |

**Same model, same data, before vs after the fixes:**

| Model | V2 bpb, hours | V3 bpb, hours |
|---|---|---|
| d256-n2 nodrop | 2.193, 7.8 h | 1.778, 1.5 h |
| d512-n4 drop | 2.748, 41.1 h | 1.621, 6.2 h |
| d1024-n2 drop | 2.756, 79.7 h | 1.742, 10.2 h |

Caveat: V2 trained on 100% of the tokens, so it saw part of the held-out
text. Its numbers are slightly *flattering* (`score_legacy_v2.sh`). The
real gap is bigger.

**What the V3 grid showed** (`reports/2026-11-20-v2-grid.md`):
- **Depth beats width.** d256-n4 (3.4M) scored 1.711 and d1024-n2
  (26.3M) scored 1.742: 8× fewer parameters, a better score.
- **Dropout hurts at one epoch:** d512-n2 1.677 without dropout, 1.705
  with. Dropout fights overfitting, and one pass over 100M characters
  doesn't overfit.
- **LR barely matters between 5e-4 and 3e-3:** 1.716 / 1.705 / 1.732.
- **Context 128 vs 64 is a tie** at half the steps (1.676 vs 1.677).
- **The samples got real words in their first half:** "was a close
  friend of Baby 's academic decision to get the family . He found
  himself to be placed with a private skeleton ." (d512-n2-blk128). The
  second half still goes to "Orachistaregengre".

## V4 regex: the tokenizer was the next bottleneck

The only change from V3 is `--tokenizer bpe-regex` (Sprint 1 Part 3: a
GPT-2-style regex pre-split before BPE).

| Run | Params | val bpb |
|---|---|---|
| d512-n4 rx4k | 16.8M | **1.439** |
| d256-n2 rx4k | 3.7M | 1.571 |
| d512-n4 rx512 | 13.2M | 1.592 |
| d512-n4 bpe512 (Part 1) | 13.2M | 1.593 |
| d512-n4 bpe512, seed 2 | 13.2M | 1.598 |
| d256-n2 rx512 | 1.9M | 1.744 |

**1. Vocabulary size was the biggest win of the whole project.** At
d256, going from 512 to 4096 tokens gained 0.17 bpb. That's more than
4× the parameters bought at a fixed vocab (d256 rx512 1.744 → d512 rx512
1.592 = 0.15). A 4k token covers 3.54 bytes, against ~2.0 at 512
(`train_v3.py` docstring). The model sees more text per step, and its
64-token context covers ~227 characters instead of ~128.

**2. The regex split alone did nothing at 512.** rx512 scored 1.592 and
Part 1 BPE 1.593. The split matters for *what* the tokens are (no merges
across word boundaries), but at 512 tokens there's no room for it to
show.

**3. Measure your noise before you believe a difference.** Two seeds of
the same config: 1.593 and 1.598. Anything under ~0.005 in these tables
is noise. That covers rx512 vs bpe512 above.

**4. The best model has the most obvious failure.** d512 rx4k writes the
most coherent opening of any run:

> Albert Einstein was a delegate to the United States Senate , and was
> elected a member of the Kentucky Assembly .

Then it loops:

> …was aided by the governorigerkerkerkerkerkerkerkerkerkerkerkerko

d256 rx4k loops the same way ("AEboyleeleeleeleelee"). Three things
add up:
- With 4k tokens, each token is seen less often in training.
- The 4k runs trained for 27,399 steps against 46,849 for the 512 runs.
  It's the same text, but fewer tokens means fewer steps.
- The curve was still falling when the epoch ended (`runs/v3/d512-n4-nodrop-rx4k-wiki103/train.log`):

  | step | val bpb |
  |---|---|
  | 8,000 | 1.658 |
  | 16,000 | 1.514 |
  | 24,000 | 1.449 |
  | 27,399 | 1.439 |

  The model is undertrained, not finished.

At temperature 0.5, once the model enters a rare pattern, repeating the
last token is its most likely move. Best bpb and worst failure in the
same run.

**5. The default that cost a day.** The first V4 launch didn't pass
`--lr`, so every run used the old default of 0.05 instead of 0.001
(`runs/v3-bad-lr005/`):

| Run | lr 0.05 | lr 0.001 |
|---|---|---|
| d512-n4 rx4k | 2.714 | 1.439 |
| d512-n4 rx512 | 3.633 | 1.592 |

At lr 0.05, V4 scored worse than V2. One missing flag undid every fix.
The grid script now passes `--lr` explicitly (`run_grid_v3.sh`).

**6. Where the space goes depends on the tokenizer.** This is the Nov 19
lesson again, with a twist:
- Part 1 BPE learns tokens that end in a space ("the "), so a prompt
  should end with a space.
- The regex tokenizer attaches the space to the *start* of the next word
  (" delegate"). For it, a trailing space becomes a lone " " token the
  model rarely saw, so `test_models_v3.py` strips it.

When d256 rx512 writes "was **acquired**", that's the model's own choice
of continuing the word "a", not a prompt artifact. The right prompt
depends on the tokenizer. (I told you this was a prompt artifact at
first. The script was already handling it correctly.)

**7. "100 tokens" isn't the same length for every tokenizer.** At 3.54
bytes per token, the 4k samples are about twice as long as the 512
samples. That's more room to drift and loop. Bits per byte already
corrects for this, but sample length doesn't.

---

## Lessons that hold across all four versions

1. **A model that starts above ln(vocab) is broken before it learns
   anything.** Check step-0 loss first. It's the cheapest test there is.
2. **Pick a metric that survives your own changes.** Per-token loss
   broke the moment the tokenizer changed. Bits per byte didn't.
3. **Hold out data, or you're measuring memory.** "And wear a golden
   sorrow."
4. **Keep the old code runnable.** V2's logs still load and score
   because `train.py` and `src/numpy/` were never edited. Every
   before/after table in this file depends on that.
5. **Every speedup came from removing waste, not from new ideas:**
   Python objects → arrays, one window → batches of 16, a GC cycle →
   freed memory, float64 → float32. The only modeling change that
   mattered was the tokenizer.
6. **Defaults are code.** The lr default did more damage than any bug
   in the model.
7. **Measure noise once** (two seeds) and you'll stop arguing about
   0.001 differences.

## One prompt across the versions (a possible lead artifact)

"Albert Einstein was a", temp 0.5, seed 1, each version's best model:

| Version | bpb | Output (first ~15 words) |
|---|---|---|
| V2 tensor, d256-n2 | 2.193 | was a collection of internal of the battle of Protests and position of Althern Briter Assing |
| V3 fixes, d512-n4 | 1.621 | was a college member of The London Post @-@ Behind , who was invited to the Poetry of College |
| V4 regex, d512-n4 4k | 1.439 | was a delegate to the United States Senate , and was elected a member of the Kentucky Assembly . |

V1 can't appear in this table, since it never trained on WikiText. At
3.8 tokens/sec, one epoch of 100M characters would take most of a year.

## Still to check before publishing

- The V1 → V2 speedup: re-run `python3 benchmarks/bench_engines.py` and
  quote the number you measure. The docstring cites ~4,000× without
  naming the config.
- A V1 scalar sample to quote.
- d512-n4 nodrop has no V3 run (the V3 d512-n4 used dropout 0.1).
  Comparing V3's 1.621 with V4's no-dropout runs mixes two changes, and
  dropout cost ~0.03 at d512-n2. Say so, or note it as a caveat.
- Whether the 4k model's loops shrink with more training: a 2-epoch rx4k
  run is the experiment.
