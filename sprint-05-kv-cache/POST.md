# Sprint 05 post outline: KV cache and sampling

Working title: **"The 22× speedup hiding in attention"**

## 1. The artifact

- The speed table, cache vs no cache, at 16 and 400 tokens.
- One sentence: same output, token for token. The tests prove it.

## 2. What gets recomputed for nothing

- Plain version: an old token's key and value never change, so keep them.
- Why queries aren't cached: only the newest token asks a question.
- The offset mask: the one-line bug you'll write (or did) and its symptom.

## 3. Prefill and decode

- Two phases with different shapes, and why decode is memory-bound.
- The bandwidth ÷ model size estimate against your measured tokens/sec.

## 4. Picking the next word

- Greedy's loop, quoted from your own output.
- Temperature, top-k, top-p with one distribution as the running example
  (0.5 / 0.3 / 0.15 / 0.05).
- The floating-point trap: 0.5 + 0.3 isn't 0.8.

## 5. What it costs

- 72 KB of cache per token for GPT-2 small, scaled up to a 70B model.
- The bridge to Sprint 7 (paging).
