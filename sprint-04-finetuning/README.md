# Sprint 4 — Fine-Tuning: Teach the Transformer a Specific Job

## Why this sprint

Sprint 3 built a transformer that predicts the next token. It is not
good at that job — it writes word-shaped chunks, not sentences. You
already decided that is a data and size ceiling, not a bug: a tiny
model on 8M characters will never be fluent, and making it bigger
would take days for the same ceiling.

This sprint asks a different question: **what is the pretrained model
good for, if not generating text?** The answer is the idea behind all
of fine-tuning. Pre-training taught the model to predict, and in
learning to predict it built up useful internal features — it knows
Shakespearean vocabulary, it can tell a philosophical passage from a
violent one, it recognizes the rhythm of verse. Those features live in
the weights, even though the generation head cannot turn them into
fluent sentences. Fine-tuning reuses those features for a narrower
job, one the model can actually do well.

The job here is **classification**: read a quote and say which of four
plays it comes from. That is a much easier decision than generating
text. Generation asks for one token out of 512, over and over, where
every choice constrains the next. Classification asks for one label
out of four, once. A model that cannot write can still be good at
recognizing. It is the same reason a person who cannot write a novel
can still tell you which author wrote a paragraph.

There are two tracks, and the comparison between them is the point of
the sprint.

**Track 1 — your model.** Fine-tune the d256/n2 BPE transformer from
Sprint 3 (your best combined-corpus model) for play classification.
You add the classification head, the loss, and the accuracy yourself,
on your own NumPy engine. This is the hand-built track.

**Track 2 — GPT-2.** Fine-tune OpenAI's GPT-2 (124M parameters) for
the same task, with PyTorch + Hugging Face. GPT-2 was pretrained on
40GB of text — it already knows English. Same task, same dataset,
different starting point. The comparison shows how much that
pretraining is worth. Track 2 uses the library because GPT-2's weights
only exist in that format; your NumPy engine cannot load them. Using
the library here does not diminish the hand-built model — it is the
only way to get a pretrained model, and the comparison is the lesson.

Learning goals:

1. Say, in plain words, what fine-tuning is and why it works — what
   the pretrained model already knows, what the new head has to learn,
   and why classification is easier than generation.
2. Implement a classification head, a cross-entropy loss over labels,
   and an accuracy metric on top of your own transformer — and check
   each one with the tests.
3. Fine-tune end to end: watch the validation accuracy rise, and say
   what the final number means against the 0.25 chance baseline.
4. Run the GPT-2 track and compare: how much better is a model
   pretrained on 40GB than one pretrained on 8M characters, on the
   same task? Know exactly what the gap is and what causes it.

Prep: Raschka, *Build a Large Language Model (From Scratch)*, chapter
6 (classification fine-tuning). That chapter is the template for Track
1 — it fine-tunes GPT-2 for spam detection, which is the same shape as
this sprint's play classification (a short text in, one label out).
Chapter 6 is also the honest reference for Track 2: Raschka loads
pretrained GPT-2 weights, he does not train them from scratch.

## The plan

Work in order. Each step has tests; a step is done when its tests pass.

**Step 1 — the classification head** (`Classifier.__init__`,
`Classifier.logits`). A Linear layer, d_model -> n_plays, applied to
the final position's hidden state. Tests: `TestHeadShape`.

**Step 2 — the loss** (`Classifier.loss`). Cross-entropy over the play
labels. Tests: `TestLoss`, including the loss-decreases check.

**Step 3 — predict and accuracy** (`Classifier.predict`,
`Classifier.accuracy`). Argmax and the fraction correct. Tests:
`TestPredict`, `TestAccuracy` — including the separable toy set, which
must reach 1.0.

**Step 4 — the training loop** (`train_step` in
`benchmarks/finetune.py`). One step: loss, zero grads, backward, Adam
update. Then run it: `make finetune`. Watch the validation accuracy.

**Step 5 — the GPT-2 track** (`benchmarks/finetune_gpt2.py`). Install
torch, transformers, datasets; fill in the four steps; run it; record
the test accuracy. Compare with Track 1.

**Step 6 — the post.** Write it from POST.md. The comparison is the
story.

## What is scaffolded vs what you implement

Scaffolded (given, do not rewrite):
- The dataset (`data/play_quotes_*.jsonl`, `data/labels.json`) and its
  builder (`data/build_dataset.py`). The quotes are already extracted
  and split.
- The pretrained transformer (reused by import from Sprint 3 — see
  below). The BPE tokenizer loading and the weight loading in
  `benchmarks/finetune.py`.
- The evaluation and saving logic in `benchmarks/finetune.py`
  (validation accuracy reporting, checkpoint saving).
- The GPT-2 script skeleton (`benchmarks/finetune_gpt2.py`) — the
  structure is given, the four steps are marked TODO.
- The tests (`tests/test_classifier.py`) — they define the contract.

You implement (the exercises):
1. `Classifier.__init__` — create the head.
2. `Classifier.logits` — the forward pass (transformer, last
   position, head).
3. `Classifier.loss` — cross-entropy over the labels.
4. `Classifier.predict` — argmax.
5. `Classifier.accuracy` — fraction correct.
6. `Classifier.parameters` — transformer params + head params.
7. `train_step` in `benchmarks/finetune.py` — one training step.
8. The four TODO steps in `benchmarks/finetune_gpt2.py` (Track 2).

## The dataset

Four plays, chosen because they have distinct styles and vocabularies
— the classes are separable but not trivially so. Hamlet is
philosophical and introspective. Macbeth is dark, violent, and full of
prophecy. Romeo and Juliet is romantic and lyrical. A Midsummer
Night's Dream is whimsical, with fairies and mechanicals. A classifier
that can tell them apart has learned something real about style, not
just memorized a few keywords.

Each quote is 1-3 consecutive lines spoken by one character, from the
Complete Works (`sprint-03-transformer/data/shakespeare-full.txt`).
Stage directions, speaker labels, and act/scene headers are stripped.
The builder is `data/build_dataset.py` — read it for the exact rules
(it gates on the Dramatis Personae so the character list does not leak
in as dialogue).

| Split | Rows | Hamlet | Macbeth | Midsummer | Romeo |
|-------|------|--------|---------|-----------|-------|
| train | 3747 | 1200   | 786     | 706       | 1055  |
| val   | 467  | 150    | 98      | 88        | 131   |
| test  | 471  | 150    | 99      | 89        | 133   |

The split is stratified (80/10/10 within each play), so every split
has the same proportions. The classes are mildly imbalanced (Hamlet
has the most dialogue, Midsummer the least) — chance is still 0.25,
and the imbalance is small enough not to distort accuracy. Labels:
hamlet 0, macbeth 1, midsummer 2, romeo 3 (see `data/labels.json`).

## The operations, defined

**Classification head.** A Linear layer that maps one hidden vector
to one number per class. Here it is Linear(d_model, n_plays): it takes
the final position's hidden state (a vector of d_model numbers) and
produces n_plays numbers, called logits — one per play. A high logit
for a play means "this quote looks like that play." The head starts
random; training is what teaches it. It is the only part of the model
that has never seen data.

**Logits.** The raw, unnormalized scores the head produces. They are
not probabilities — they can be any number, positive or negative.
Softmax turns them into probabilities (see below). The predicted play
is the one with the highest logit (argmax).

**Softmax.** Takes a list of numbers and turns them into probabilities
that sum to 1. The biggest number gets the biggest probability. Same
function as Sprint 3 (it is imported from the transformer module), but
here it is applied over 4 play logits instead of a 512-token
vocabulary. The formula is identical; only the classes change.

**Cross-entropy loss (classification).** Softmax the logits, then take
the negative log of the probability assigned to the true play. If the
model gave the right play 90%, the loss is small (-log 0.9 = 0.11). If
it gave it 1%, the loss is large (-log 0.01 = 4.6). This is the same
math as the next-token loss in Sprint 3 — the only difference is what
the classes are (plays, not tokens). A random model scores about
-log(0.25) = 1.39 (chance). A perfect model scores 0.

**Accuracy.** The fraction of examples where the predicted play
(argmax of the logits) equals the true play. It is a plain number
between 0 and 1. Chance is 0.25 (4 plays). This is the number that
matters for the comparison, not the loss — the loss is for training,
accuracy is for judging.

**Fine-tuning (vs pre-training).** Pre-training (Sprint 3) taught the
model to predict the next token, on a large corpus, from random
weights. Fine-tuning starts from those pretrained weights and trains
on a small, task-specific dataset (here, ~3,700 labeled quotes) with a
small learning rate. The pretrained weights already know the language;
fine-tuning teaches the model the task. The learning rate is small
(0.001 Adam) because a large one would destroy what pre-training
built.

**Frozen vs trained.** You can freeze the transformer (train only the
head) or train the whole model (head + transformer). Freezing is for
big models: it keeps the pretrained features intact and avoids
catastrophic forgetting (the model forgetting what it learned in
pre-training). Here the model is small and the dataset is small, so
the default is to train the whole model — there is little to forget,
and letting the transformer adapt helps. If the validation accuracy
goes down while the training loss goes down, that is overfitting, and
freezing is the first thing to try.

**Catastrophic forgetting.** When fine-tuning with too high a learning
rate (or for too long), the model overwrites its pretrained features
with the narrow task data and gets worse at everything else. It is why
the fine-tuning learning rate is much smaller than the pre-training
one. You will not see it much here (the model is small), but it is the
reason the README mentions freezing.

## How the classifier is wired

The transformer is reused by import from Sprint 3 — it is NOT copied
into this folder. `src/classifier.py` adds Sprint 3's root to the path
and imports `Transformer`, `Linear`, and `softmax` from
`src.numpy`. The tests and `benchmarks/finetune.py` load
`src/classifier.py` by file path (importlib) rather than as a package
import, because this folder also has a `src/` directory and the two
would collide. See the test file's module docstring for the full
explanation.

The forward pass (what `logits` does):
1. Embed the quote's tokens (token + positional embeddings).
2. Run each transformer block.
3. Apply the final LayerNorm.
4. Take the LAST position's hidden state. Why the last: attention is
   causal, so the last position is the only one that has seen the
   whole quote. Earlier positions have only seen a prefix.
5. Feed it through the head. The result is (n_plays,) logits.

Shape note: the NumPy engine's matmul is 2-D only. Keep the last
position as a (1, d_model) row (`x[-1:]`) and index `[0]` after the
head, rather than squeezing to a (d_model,) vector first — a 1-D
input breaks the matmul backward. This is the one sharp edge in the
file, and the tests catch it (the loss-decreases test fails with a
matmul shape error if you use `x[-1]`).

## The comparison (what to record)

Run both tracks on the same test split and record:

| Track | Model | Pretraining | Test accuracy |
|-------|-------|-------------|---------------|
| 1 | d256/n2 BPE (hand-built) | 8M chars, from scratch | ? |
| 2 | GPT-2 124M (Hugging Face) | 40GB, OpenAI | ? |

The gap between them is the value of pretraining on 40GB vs 8M
characters. Also record how much data each one needed: try Track 1
with 500, 1000, and all 3,747 training quotes, and see where the
accuracy saturates. A pretrained model usually needs far less
task-specific data — that is part of the comparison too.

## File layout

```
sprint-04-finetuning/
  README.md                  this file
  POST.md                    post outline (hybrid register)
  Makefile                   test / finetune / finetune-gpt2
  data/
    build_dataset.py         builds the quotes from shakespeare-full.txt
    labels.json              label index -> play name
    play_quotes_train.jsonl  3747 labeled quotes
    play_quotes_val.jsonl    467 labeled quotes
    play_quotes_test.jsonl   471 labeled quotes
  src/
    classifier.py            the Classifier (your exercises)
  benchmarks/
    finetune.py              Track 1 training loop (train_step is yours)
    finetune_gpt2.py         Track 2 skeleton (four TODO steps)
  tests/
    test_classifier.py       15 tests (11 classifier + 4 GPT-2 structure)
```

## Production differences (read before writing the post)

- Real fine-tuning uses much larger models and datasets, and the
  classification head is usually trained with the transformer frozen
  for a few steps first (a "warmup"), then unfrozen. Here we train
  everything from step one because the model is small.
- Real systems use a validation set to pick the best checkpoint and
  early stopping (stop when val accuracy stops improving). Here the
  loop just runs a fixed number of steps; adding early stopping is a
  natural extension.
- GPT-2 in production is fine-tuned with LoRA (low-rank adapters) or
  full fine-tuning on GPUs. This sprint does full fine-tuning on CPU,
  which is slow but works at this scale.
- The play-classification task is a toy. Real classification tasks
  (spam, sentiment, intent) use the same shape — a short text in, one
  label out — with far more data and more classes.
