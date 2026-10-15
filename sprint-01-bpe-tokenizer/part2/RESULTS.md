# Part 2 — Results

Machine: Mac mini (fill in chip/RAM)
Date:
Command: `python3 part2/compare.py` from `sprint-01-bpe-tokenizer/`
Timing corpus: sprint corpus ×30 (~140 KB), best of repeated runs

| Operation | v1 (frozen) | v2 (optimized) | Speedup |
|---|---|---|---|
| Train (vocab 384, ~140 KB) |2.495s| 2.478s | 1.01× |
| Encode (vocab 512, ~140 KB) | 2.130s	| 2.162s	| 0.99×|
| Decode (30,000 tokens) |0.126s	 |0.010s	 |12.6×
 |

## Notes

- What I changed, in order:
  1.
  2.
  3. (stretch, if attempted)
- Which change actually mattered most — and was it the one I expected?
- Anything that got *slower* or surprised me:
