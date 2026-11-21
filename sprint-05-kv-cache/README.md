# Sprint 05: KV Cache and Sampling

## Why this sprint

Sprint 4's GPT-2 makes every new token by rerunning the model over the
whole text so far. Token 400 redoes the work for tokens 1–399, all over
again. That's why it slows down as it writes.

But most of that work is repeated for nothing. Inside attention, each
token's *key* and *value* depend only on that token and the ones before
it, so they never change once computed. A **KV cache** keeps them. Each
new token then computes just its own query, key and value, appends the
key and value to the cache, and attends over everything stored. That
turns quadratic work into linear work, and it's the single biggest speed
trick in LLM serving.

The second half is **sampling**. Greedy decoding (always take the top
token) gets stuck in loops: "I'm a writer, and I'm a writer". Real
systems pick at random, weighted by the model's probabilities, and use
three knobs to control how adventurous it is: temperature, top-k and
top-p. These are the `temperature` and `top_p` parameters in every LLM
API.

Learning goals:

1. Explain why keys and values can be cached but queries can't, and why
   the causal mask needs an offset once part of the sequence is cached.
2. Measure the speedup and explain its shape: small at short contexts,
   huge at long ones.
3. Implement temperature, top-k and top-p, and explain what each one
   does to the distribution.
4. Build `generate()` as a Python generator, the streaming interface
   Sprint 6's server will use.

## The plan

**Sprint 4 must pass first.** `src/deps.py` imports your
`sprint-04-gpt2/src` directly, so this sprint builds on your code, not a
copy of it.

| Step | File | What | Tests |
|------|------|------|-------|
| 1 | `src/kv_cache.py` | `KVCache.update` | `TestKVCache` |
| 2 | | `attention_cached`: Sprint 4's attention over cached keys, with an offset mask | `TestAttentionCached` |
| 3 | | `forward_cached`: one call for prefill and for decode | `TestForwardCached` |
| 4 | `src/sampling.py` | `apply_temperature`, `top_k_filter`, `top_p_filter` | `TestFilters` |
| 5 | | `sample` | `TestSample` |
| 6 | | `generate`: prefill, then decode, yielding tokens | `TestGenerate` |
| 7 | real GPT-2 | cached greedy must equal Sprint 4's greedy exactly | `tests/test_real_gpt2.py` |

The cache must change *nothing but speed*. Almost every test compares
cached logits with your Sprint 4 `forward` on the same tokens.

**Prefill vs decode.** These are the two phases every inference server
reports separately:
- **Prefill** runs the whole prompt in one call. It's matrix-shaped work
  and fast per token.
- **Decode** runs one token per call. It's vector-shaped work, and its
  cost is mostly reading all 124M weights from memory once per token.

## Run it

```bash
make test       # 30 tests (2 use the real GPT-2 from Sprint 4's make download)
make speed      # cache vs no cache, at short and long contexts
make chat TEXT="Once upon a time" TEMP=0.8 TOP_P=0.9
```

## Results

Reference solution, dev Mac, 2 threads, 48 new tokens:

| Context | No cache | KV cache | Speedup |
|---------|----------|----------|---------|
| 16-token prompt | 12.3 tokens/sec | 50.2 tokens/sec | 4.1× |
| 400-token prompt | 1.3 tokens/sec | 28.7 tokens/sec | 21.9× |

## Before writing the post

- **Memory is the price.** For GPT-2 small, the cache holds 2 (K and V)
  × 12 layers × 768 numbers × 4 bytes = 72 KB per token, or 72 MB for a
  full 1,024-token context. For a 70B model with an 8k context it's
  gigabytes per *request*. That's why Sprint 7 pages the cache.
- **Why decode is memory-bound:** one token = one pass over all the
  weights, so tokens/sec is roughly memory bandwidth ÷ model size.
  Batching (Sprint 7) shares that pass across requests.
- **What production adds:** a fused attention kernel that reads the cache
  in place (FlashAttention / PagedAttention), speculative decoding, and
  the cache stored in lower precision.
- **Sampling in APIs:** `temperature`, `top_p` and `top_k` are exactly
  these three functions. `seed` is the rng.

## What AI wrote

- **Scaffold:** this README, all tests, `src/deps.py` (cross-sprint
  imports), the `KVCache` allocation and `advance`/`reset`/`for_model`,
  `softmax` in sampling, `benchmarks/`, the Makefile, `POST.md`.
- **Hand-written:** `update`, `attention_cached`, `forward_cached`, all
  three filters, `sample`, `generate`.

## Post status

Not started. Outline in `POST.md`.
