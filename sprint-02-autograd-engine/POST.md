# POST.md — Sprint 2 post outline (~900–1,500 words)

Your draft first, as agreed — Cleo edits for structure/clarity and asks
questions, never ghostwrites. Rule carried over from Sprint 1: every prompt
below covers ground from the README discussion or something you built this
weekend. The production-systems section (§6) is marked HOLD — we discuss it
in review before you draft it, same as the Part 2 pretokenization section.

## 1. The problem in one paragraph

What is backpropagation, and why does training a neural network need it?
(One paragraph. If you can't say it in one paragraph on Sunday, the rest of
the post will tell you which concept is still fuzzy.)

## 2. My mental model on Friday

Your words, not the README's. Start from your capture-sheet line (prompt 3,
Oct 18): the one plain-words phrase for gradient flow. Then: what did you
expect the hard part to be before writing any code?

## 3. The naive build

The Value representation: a scalar, its children, the op, the backward
closure. Walk through one op (your choice — `__mul__` is the most
instructive) and say what each of the four pieces does. Then backward():
topo sort, zero, seed 1.0, reverse walk. Quote your v1 benchmark numbers
here (bench.py sweep table + ms per MLP pass).

## 4. Where it broke (with numbers)

The bug you actually hit. Usual suspects, for orientation only — write up
yours: `=` instead of `+=` on accumulation (which test caught it?),
topological order wrong (what symptom?), grads not zeroed between training
steps (what did the loss curve look like?), RecursionError on the deep
sweep. For each: the symptom, the one-line cause, the fix, and the test or
number that proved it.

## 5. The optimized design / v2, and why

Only if you took a stretch goal: what you changed, why that change, and the
before/after numbers. If you measured a stretch and chose NOT to do it
(e.g. topo caching bought nothing), that is also a result — write it up with
the numbers that killed it. No stretch, no problem: §4 + §7 carry the post.

## 6. Trade-offs / what production systems do differently — HOLD FOR REVIEW

Do not draft this section until the review discussion. Pointers for that
conversation, not for writing yet: PyTorch builds the same graph but over
tensors, in C++, with the same reverse-mode idea; frameworks choose
operator-overloading (like yours) vs. source transformation; your scalar
engine's per-op Python overhead is exactly what production vectorization
erases. We will talk through which of these you can fairly claim from what
you built.

## 7. What I misunderstood on Friday vs Sunday

The close. Two or three sentences: what did Friday-you get wrong about
backprop that Sunday-you can now correct from the code? (In Sprint 1 this
was the "what I misunderstood" section — same function here.)

## Benchmark table (fill from your runs)

| measurement | v1 (Sat) | v2 (Sun) |
|---|---|---|
| bench.py: forward, N=20000 (s) | | |
| bench.py: backward, N=20000 (s) | | |
| ms per MLP(2,[16,16,1]) forward+backward | | |
| train.py: loss step 0 → step 49 | | |
| train.py: accuracy step 49 | | |
| train.py: wall-clock (s) | | |
