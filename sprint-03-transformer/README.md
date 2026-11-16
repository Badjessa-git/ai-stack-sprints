# Sprint 3 — A Transformer Block and a Tiny Character Language Model

## Why this sprint

Sprints 1 and 2 built the two pieces every language model stands on: a way
to turn text into numbers (the BPE tokenizer) and a way to learn from
mistakes (the scalar autograd engine). This sprint builds the thing that
actually *reads*: a transformer.

A transformer is a machine that looks at the tokens so far and guesses the
next one. It does that with one idea, repeated: **attention** — every
position looks back at the positions before it, decides which ones matter
for what comes next, and mixes their information together. Stack that with
a feed-forward network and a normalizer, repeat it a few times, and you
have the architecture behind GPT, Llama, Claude, and everything else in
the series. The model here is tiny (5,354 parameters at the default size)
and trains on 20,000 characters of Shakespeare, but it is the same shape
as the real thing — same operations, same training loop, same loss.

Learning goals:

1. Say, in plain words, what attention computes and why the 1/√d scale
   and the causal mask are there.
2. Implement softmax, GELU, LayerNorm, attention, and a transformer block
   on top of your own `Value` engine — and check each one with a gradient
   check, the way Sprint 2 checked the MLP.
3. Train the model end to end: watch the loss fall, sample text from it,
   and say what the sample shows about what it has and hasn't learned.
4. Know exactly where this toy stops and production starts (checklist at
   the end — read it *before* writing the post).

Prep: Raschka, *Build a Large Language Model (From Scratch)*, chapters 3
(attention) and 4 (GPT architecture); chapter 5 (pretraining) for the
training loop. Karpathy's "Let's build GPT" lecture covers the same
ground in video form. 3Blue1Brown's attention video is the best 20-minute
version of the attention walk in "The operations, defined" below.

## The plan

Work in order. Each step has tests; a step is done when its tests pass.

**Step 0 — the engine, first.** Two things, both in `src/engine.py`
(the frozen copy of your Sprint 2 engine):

- (a) Add `Value.exp()` and `Value.log()`, shaped exactly like `tanh()`.
  Cross-entropy needs them. Tests: `TestExpLog`.
- (b) Make `Value.backward()` finish on this graph. **Measured fact, on
  this VM: one forward pass on the default model takes ~0.8 s; the
  backward pass on the same graph did not finish in 4 minutes.** The
  engine worked fine on Sprint 2's MLP, so this only shows up at
  transformer scale. Do not start debugging the transformer until the
  training test passes — the fault is in `backward()`, not in attention.
  How to measure it yourself: build the small test model (the one in
  `test_training_loss_decreases`), time `model.loss(idx)` and then time
  `loss.backward()` separately. That split — fast forward, never-finishing
  backward — is the whole symptom, and it is the first thing to explain
  when the step is done. Note the family resemblance: Sprint 2's moons
  plateau was also a shared-node problem in the same function, fixed by
  a `visited` set. This one is its cousin, one level deeper.

**Step 1 — softmax and GELU** (`softmax`, `gelu`). Pure functions on
lists of Values / single Values. Tests: `TestSoftmax`, `TestGelu`.

**Step 2 — attention** (`CausalSelfAttention`). The heart of the sprint.
Tests: `TestAttention`. Read the attention walk in "The operations,
defined" before writing a line.

**Step 3 — multi-head, LayerNorm, feed-forward** (`MultiHeadAttention`,
`LayerNorm`, `FeedForward`). Tests: `TestMultiHead`, `TestLayerNorm`,
`TestFeedForward`.

**Step 4 — the block** (`Block`). Two residual adds around attention and
the MLP. Tests: `TestBlock` — including the zero-weights identity test,
the sharpest test in the suite: if a residual is missing or backwards,
it fails.

**Step 5 — the transformer** (`Embedding`, `Transformer`, `loss`,
`generate`). Tests: `TestTransformer`, including the acceptance test
(`test_training_loss_decreases`).

**Step 6 — train it.** `make train` (100 steps, ~4 minutes on the VM,
~2 on a Mac). Read the sample it prints and decide what it proves.

Stretches (only after Step 6): `n_layer=2`; `d_model=32` (about 5x
slower per step — measure it with `make benchmark` first); ReLU instead
of GELU (simpler, slightly worse — a one-line swap, good for the post);
Adam instead of plain SGD (needs a per-parameter state dict — a real
design decision, not a tweak).

## The operations, defined

Every operation below: what it means in plain words, the forward math,
and what it passes backwards. If a term appears here that you haven't
met, that is a signal to stop and ask before coding it.

**exp(x)** — e raised to x. Forward: `math.exp(x)`. Reverse: the gradient
coming in is multiplied by exp(x) itself — exp is its own derivative.

**log(x)** — the natural log. Forward: `math.log(x)`. Reverse: the
incoming gradient is divided by x.

**softmax** — turns a list of scores into weights that sum to 1.
Forward: subtract the biggest score first (so `exp` never overflows),
exponentiate each, divide each by the total. Reverse: each output's
gradient is shared back to every input — raising one score lowers the
others' weights, because the total is shared.

**GELU** — the activation in the feed-forward network. Plain words: it
keeps positive inputs almost as they are and fades negative inputs toward
zero, but smoothly, not with a hard cutoff like ReLU. Forward (tanh
approximation):
`0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
Reverse: flows through the tanh and the multiplications exactly like any
other chain — no special case, which is why the formula is written with
ops your engine already has.

**Attention, walked end to end.** Take two tokens and a head of size 2.
The weights are fixed so you can check every number by hand — this is
the same case as `test_fixed_weights_expected_output`.

Inputs: token 0 is `[0.5, -0.5, 1.0, 0.0]`, token 1 is
`[1.0, 0.0, -1.0, 0.5]`. The head's three projections use constant
weights: Wq uses 0.5 for its first neuron and -0.5 for its second, Wk
uses 0.25 and 0.75, Wv uses 0.1 and 0.3. All biases are 0.

First, each token asks three questions of itself, all by the same
mechanism — a weighted sum of its four numbers:

- **Query (what am I looking for?)**: token 0's query is
  `[0.5, -0.5]`. (First number: 0.5×0.5 + (-0.5)×0.5 + 1.0×0.5 = 0.5.
  Second: the same sums with -0.5 as the weight = -0.5.)
- **Key (what do I offer?)**: token 0's key is `[0.25, 0.75]`.
- **Value (what do I actually say?)**: token 0's value is `[0.1, 0.3]`.

Token 0 can only look at itself (the causal mask — see below), so its
score is its query dotted with its own key, divided by √2 (the scale
keeps the dot product from growing with head size, which would flatten
the softmax): `(0.5×0.25 + (-0.5)×0.75) / 1.414 = -0.177`. Softmax over
a single score is 1.0, so token 0's output is just its own value:
`[0.1, 0.3]`.

Token 1 looks at both tokens. Its query is `[0.25, -0.25]`, its key is
`[0.125, 0.375]`, its value is `[0.05, 0.15]`. Scores: against token 0,
`-0.088`; against itself, `-0.044`. Softmax turns those into
`[0.489, 0.511]` — nearly even, because the scores are close. The output
is the weighted mix: `0.489 × [0.1, 0.3] + 0.511 × [0.05, 0.15] =
[0.0744, 0.2233]`.

Two things to notice. First, the **causal mask**: token 0's output never
touches token 1 — the mask sets future scores to negative infinity before
the softmax, so their weight becomes exactly 0. That is what makes the
model a predictor instead of a peeker: at generation time there *is* no
future token. Second, the **scale**: without the 1/√d, the dot products
grow with head size, the softmax saturates to a one-hot choice, and the
gradients through it die.

Reverse share for attention: the output gradient splits to the values
through the weights, and to the queries and keys through the scores —
which means every weight in Wq, Wk, Wv gets gradient from *every*
position that used it. That sharing is exactly what Step 0(b) is about.

**Multi-head attention** — run several small heads in parallel (each
looking for a different kind of relationship) and glue their outputs
together, then mix them with one more linear layer (Wout). Forward:
concat the head outputs, apply Wout. Reverse: Wout's gradient splits
back to the heads.

**LayerNorm** — rescales one token's vector so it has mean 0 and
variance 1, then lets the model re-scale and re-shift it with learned
`gamma` (starts at 1) and `beta` (starts at 0). Forward: mean, then
population variance (divide by n, not n-1), then `(x - mean) /
sqrt(var + 1e-5)`, then `gamma * x + beta`. Plain words: it stops one
position's numbers from drifting huge or tiny as they pass through the
layers. Reverse: the gradient is shared back through the mean and the
variance to *every* element of the vector — normalizing couples the
components, so no component's gradient is independent.

**Feed-forward network** — the same token, processed alone: expand it to
4x width, GELU, squeeze it back. Forward: Linear → GELU → Linear.
Reverse: ordinary chain rule through the two layers.

**Residual add** — `x + sublayer(x)`. Plain words: the sublayer only has
to learn the *change*, not rebuild the token from scratch. Reverse: the
gradient flows straight through the `+` to x untouched, and also into
the sublayer. That unobstructed path is why deep stacks train at all.

**Embedding** — a lookup table: token id → the row of learned numbers
that *is* that token. Forward: return the row itself (the test checks
identity with `is`). Reverse: a row's gradient is the sum of the
gradients from every position that used that token.

**Cross-entropy loss** — how wrong the prediction was. Forward: take the
logits at position t, softmax them, take the log of the probability of
the token that *actually* came next, negate it, and average over
positions. A loss of ln(58) ≈ 4.06 means "guessing at random among 58
characters". Reverse: the gradient of the loss with respect to a logit
is `softmax(logit) - 1` for the true token and `softmax(logit)` for the
others — the model is pushed up on the right answer and down on all the
others, in proportion to how confident it was.

## The training loop, in full

This is the section Sprint 2's scaffold never taught, so here it is,
complete, before you run anything.

A training step is five lines, always in this order:

1. **Predict.** Pick a batch — a slice of the corpus 8 tokens long
   (`block_size=8`) — and run `model.loss(idx)`. That builds the whole
   graph: embeddings, one block, logits, cross-entropy.
2. **Zero the gradients.** `model.zero_grad()`. Gradients *accumulate*
   (they are `+=`'d), so skipping this step trains on the sum of every
   gradient since the program started. That was a real Sprint 2 bug.
3. **Backward.** `loss.backward()`. Every parameter's `.grad` now holds
   "if I nudge this parameter, how much does the loss move?"
4. **Update.** `p.data -= lr * p.grad` for every parameter. `lr` is the
   learning rate: 0.05 here. Too big and the loss explodes; too small and
   nothing moves in 100 steps.
5. **Repeat**, on a *different* random batch each step.

What to watch while it runs:

- The loss starts near ln(58) ≈ 4.06 for a random model — actually a
  little above it (about 6.5 here, because the random weights are not
  calibrated). Over 100 steps at lr 0.05 it should fall toward ~4.
  That is real learning, but modest: this model is tiny and the data is
  small.
- The loss is **noisy step to step**, because every step trains on a
  different random batch. Judge the trend over tens of steps, never two
  adjacent ones. If you want a clean number, evaluate the loss on one
  fixed batch before and after training — that is exactly what
  `test_training_loss_decreases` does (4.19 → 1.11 in 5 steps on its
  fixed batch).
- The generated sample at the end is the honest report. After 100 steps
  it will look like Shakespeare that has been through a blender — right
  letters, wrong words. That is what 5,354 parameters buys you. The
  point of the sprint is the machinery, not the prose.

Coercion note, carried over from Sprint 2: the engine coerces plain
numbers (`Value(2.0) + 1` works, and `1 + Value(2.0)` works via
`__radd__`). In this sprint the places that matters are the constants:
the `1/sqrt(head_dim)` scale, the `0.5` in GELU, the `1e-5` in
LayerNorm, and the `-1` in the loss. If a test fails with a `TypeError`
about an unsupported operand, a plain number met a `Value` on the wrong
side — check which side the constant is on.

## What to expect (measured on this VM)

- One step at the default size (d_model=16, 2 heads, 1 layer,
  block_size=8, vocab 58): forward ~2.0 s, backward ~0.08 s once Step
  0(b) is done — 2.1 s per step, about 3.8 tokens/sec. On your Mac,
  expect roughly 2x faster (Sprint 2's MLP ran 2.4x faster there).
- At d_model=32, block_size=16: forward ~10.4 s per step on the VM.
  That is the scale-up stretch for a reason.
- `make test` against a correct implementation: 27 tests, ~2.3 s.
- `make train` (100 steps): ~4 minutes on the VM, ~2 on a Mac, ending
  with a greedy sample that reads like broken Shakespeare.
- Sampling is slow on purpose: generating one character costs a full
  forward pass, because every step re-computes attention over the whole
  context. Fixing that is Sprint 6 (KV cache), not this one.

## Before writing the post

Talk through these before POST.md section 5 (production differences).
The post may only claim what was discussed here.

- **Batched matmul vs scalar Values.** Production models do attention
  as matrix multiplies over whole batches on a GPU. This sprint does one
  scalar at a time in Python — the same math, roughly a million times
  slower. Say the number: 3.8 tokens/sec here vs millions in production.
- **FlashAttention.** Production attention never materializes the full
  T×T score matrix; it tiles the computation. Our scores list is the
  naive version — fine at T=8, impossible at T=8,192.
- **KV cache (Sprint 6).** Generation here re-runs the whole model per
  token. Production caches the keys and values.
- **Subword tokenizers vs characters.** Sprint 1's BPE turns text into
  ~500-token vocabularies; here each character is a token (vocab 58).
  Characters make the sprint small enough to train by hand, and make the
  sample look worse than a BPE model would.
- **Scale.** 5,354 parameters, 1 layer, 20,000 characters. GPT-2 small
  is 124M parameters and 40GB of text. The architecture is the same;
  the behavior at this scale is not evidence about the behavior at
  that scale.

## Files

- `src/engine.py`, `src/nn.py` — frozen copies of your merged Sprint 2
  files (commit e883d6d). Step 0 edits the engine copy only.
- `src/transformer.py` — the stub: API, docstrings, and the `parameters()`
  wiring are done; the bodies are yours.
- `tests/test_transformer.py` — 27 tests. Stubs fail 26 of them.
- `benchmarks/bench.py` — one-step timing. `benchmarks/train.py` — the
  training loop (`--steps`, `--lr`, `--d-model`, `--block-size`, `--sample`).
- `data/corpus.txt` — first 20,000 characters of Tiny Shakespeare,
  vocab 58 characters.
- `POST.md` — the post outline, in the series' plain-first register.

## Resources

- Raschka, *Build a Large Language Model (From Scratch)*, ch. 3
  (attention), ch. 4 (GPT architecture), ch. 5 (pretraining).
- Karpathy, "Let's build GPT: from scratch, in code, spelled out"
  (YouTube) — the same model, built live.
- 3Blue1Brown, "Attention in transformers, visually explained" — the
  20-minute version of the attention section above.
