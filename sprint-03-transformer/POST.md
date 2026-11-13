# Sprint 3 post outline — transformer block + tiny character LM

Write it the way the series settled: plain first, term second, proof
third. Plain-words explanation, then the technical term, then a real
artifact (a short code snippet with a "look at this" cue, or a table).
No hook formulas. Lead with the artifact and one true, specific claim.

## 1. The setup — what I built and why

- The series so far in one paragraph: tokenizer, autograd engine, now
  the transformer that turns those pieces into a model that writes.
- The why, in your own words: you use these systems every day and
  wanted to actually understand how they work.
- The artifact: a tiny character-level GPT that trains on 20KB of
  Shakespeare and writes Shakespeare-ish text. No frameworks, no
  PyTorch, just your Sprint 2 engine underneath.

## 2. Attention, in plain words (then the term, then the proof)

- The plain version first: every position gets to look back at the
  positions before it and decide which ones matter, then take a weighted
  average of what they say. That is the whole idea.
- Then the term: attention. Then the proof: one worked example with
  real numbers (the README's worked example), and the code snippet for
  the score calculation with a "look at this" cue on the division by
  sqrt(head_dim) and the causal mask.
- The one-sentence production link: this is the same operation inside
  every LLM; production just runs it on matrices on a GPU instead of
  one number at a time.

## 3. The block — the part that repeats

- Pre-norm, residual, feedforward: what each does in plain words, and
  the one-line reason each exists (residuals let gradients flow,
  LayerNorm keeps training stable, the feedforward does the per-position
  thinking).
- Proof: the zero-weights test — a block with all weights at zero
  returns its input unchanged. That is the residual doing its job.

## 4. Where it broke (with numbers)

- The training run: loss trajectory, step time, what the model actually
  generated at the end (quote it, warts and all).
- The bug you hit, if any — the Sprint 2 post worked because the bug
  story was specific (gradient 7, math said 5). Same standard here:
  real numbers, the actual confusion, no vague "debugging was hard."
- If training was slow, say how slow, with the step time from
  `make benchmark`. That number is the honest bridge to the production
  section: this is why real models run on GPUs and why the engine is
  one number at a time.

## 5. What production does differently

- Only after the discussion in the README's review checklist — the
  learning-before-writing rule. Points the scaffold already covers:
  batched matrix math instead of scalar Values, fused attention kernels
  (FlashAttention), KV caching at inference (Sprint 6), tokenizers
  instead of characters, and scale (d_model in the thousands, dozens
  of layers, billions of tokens).
- Keep it to the differences you can now explain from having built
  the small version, not a list copied from a blog post.

## 6. What I misunderstood on Friday vs Sunday

- The reflection close, same as Parts 1-2. One concrete
  misunderstanding from the start of the weekend, corrected by Sunday,
  in your words.

## 7. Resources

- Raschka, Build a Large Language Model (From Scratch): chapters 3
  (attention), 4 (GPT architecture), 5 (pretraining).
- Karpathy, "Let's build GPT: from scratch, in code, spelled out"
  (YouTube) — the lecture this sprint parallels.
- 3Blue1Brown, attention video — the visual version of section 2.
- The repo: link the sprint folder and the scaffold branch.
