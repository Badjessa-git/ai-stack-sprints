---
title: "Rebuilding the AI Stack by Hand, Part 2: Optimizing the Tokenizer"
published: false
description: ""
tags: ai, python, machinelearning, programming
series: "Rebuilding the AI Stack by Hand"
---

# Part 2 outline — Optimizing the tokenizer

Draft in your own words first, same as Part 1. Target ~900–1,500 words.
Link back to Part 1 in the opening lines.

## 1. Where v1 spends its time

Recap v1's costs from the Part 1 post, now with the compare.py framing:
recount-from-scratch training, sort-to-find-one-maximum, per-token reverse
lookups in decode. Which of these did you *expect* to dominate before
measuring?

## 2. The constraint

Optimization under an identical-output rule: same merges, same encodings,
same decodings, byte for byte. Why that constraint is what makes the
exercise honest (and why the equivalence tests exist).

## 3. Optimization 1 — train: max instead of sort

What changed, why ties still break the same way, and what it did to the
train time.

## 4. Optimization 2 — decode: the precomputed token → bytes table

The v1 decode story (reverse scan + repeated set building + re-expanded
subtrees) vs. build-once-and-look-up. Why children always having smaller
ids makes the table buildable in one ordered pass. The numbers.

## 5. Optimization 3 (only if attempted) — incremental pair counts

Updating counts around merge sites instead of recounting the world. The
bookkeeping this introduces, and whether the complexity was worth it.

## 6. The numbers

The RESULTS.md table. What improved most, what barely moved, and what
surprised you. If something you optimized turned out not to matter, say
so — that's the most useful part.

## 7. What production tokenizers do differently

<!-- Discuss this with Cleo during the code review BEFORE writing it —
same rule as Part 1: no explaining trade-offs we haven't covered. -->

## 8. What I misunderstood (the deferred Part 1 reflection)

Close the v1 → v2 arc: what you believed at the start of Sprint 1
(delimiter-splitting mental model, "decode is just reversing encode,"
etc.) vs. what you understand now — about tokenization, and about the
difference between making something correct and making it fast.

## 9. Next

Tease Sprint 2: a scalar autograd engine — teaching the computer to take
derivatives by hand.
