# Sprint 08: A Modern Architecture (SmolLM-135M)

## Why this sprint

GPT-2 is from 2019. Almost every open model since (Llama, Mistral, Qwen,
Gemma, SmolLM) is the same transformer with four parts swapped out. This
sprint makes the four swaps and runs **SmolLM-135M** (Hugging Face, 2024,
Apache 2.0) on your code. It's about GPT-2's size, but built and trained
the modern way.

| GPT-2 (Sprint 4) | Llama family (this sprint) | Why |
|---|---|---|
| LayerNorm | **RMSNorm**: no mean, no bias | cheaper, works as well |
| learned position table | **RoPE**: rotate q and k by an angle per position | attention sees *relative* position |
| GELU MLP | **SwiGLU**: a gate decides what passes | better quality per parameter |
| every head has its own k, v | **GQA**: 9 query heads share 3 k/v heads | a 3× smaller KV cache per layer |

Plus two loading changes:
- The weights use the PyTorch layout, `x @ W.T`, where GPT-2 used `x @ W`.
- The tokenizer ships as one `tokenizer.json`. Its BPE is your Sprint 4
  tokenizer, with one twist: every digit is split off on its own.

By the end, "The capital of France is" gets " Paris." from your code,
and your Sprint 6 server serves it.

Learning goals:

1. Explain each of the four swaps in plain words, and what it buys.
2. Implement RoPE and show, with a test, that attention then depends
   only on relative position.
3. Explain GQA's trade: fewer k/v heads, so a smaller cache and less
   memory traffic, at almost no quality cost.
4. Compare a 2019 and a 2024 model of the same size on the same text, and
   say why the answer depends on the text.

## The plan

Needs Sprints 4 and 5 (safetensors, `softmax`, the BPE, the KVCache).

| Step | File | What | Tests |
|------|------|------|-------|
| 1 | `src/llama.py` | `rms_norm`, `silu`, `swiglu` | `TestOps` |
| 2 | | `rope_tables`, `apply_rope` | `TestRope` |
| 3 | | `repeat_kv`, `attention` (GQA, RoPE, Sprint 5 cache) | `TestAttention` |
| 4 | | `map_hf_names` | `TestLoading` |
| 5 | | `Llama.forward` (with and without a cache) | `TestModel` |
| 6 | `src/hf_tokenizer.py` | `from_tokenizer_json`, `pre_tokenize` (digits) | `TestHFTokenizer` |
| 7 | real SmolLM | `make download && make test-real` | `tests/test_real_smollm.py` |

**The RoPE trap.** The original RoPE paper rotates pairs of neighbouring
dimensions (0 with 1, 2 with 3, ...). Hugging Face's Llama pairs
dimension i with i + D/2 ("rotate half"), and its checkpoints were
permuted to match. The model runs with either one, but writes garbage
with the paper's. `test_pairs_i_with_i_plus_half` pins it down.

**A test worth reading:** `test_rope_has_no_absolute_position`. Feed the
same token three times and you get the same output three times. Every
position attends over identical values, and RoPE only encodes offsets.
GPT-2's position table would make the three rows differ.

## Run it

```bash
make test           # 26 tests on a tiny random Llama (no download)
make download       # ~540 MB into data/smollm-135m/ (gitignored)
make test-real      # 4 acceptance tests on the real model
make compare        # GPT-2 vs SmolLM (needs Sprint 4's download too)
make serve          # SmolLM behind your Sprint 6 server, then curl it
```

## Results

Reference solution, dev Mac, 2 threads (`make compare`). Bits per byte,
lower is better:

| Model | Params | WikiText-103 | This repo's READMEs | Decode | KV cache / token |
|---|---|---|---|---|---|
| GPT-2 small (2019) | 124M | **1.138** | 1.905 | 50 tokens/sec | 72 KB |
| SmolLM-135M (2024) | 135M | 1.172 | **1.547** | 39 tokens/sec | 45 KB |

- **Prose:** on technical prose neither model has seen (these READMEs),
  SmolLM is far better.
- **WikiText:** the two are about even. That holds after cleaning up
  WikiText's ` @-@ ` markup too, so it isn't the formatting. GPT-2's
  training data is closer to Wikipedia's style than SmolLM's
  (educational web text, synthetic textbooks, code).
- **Speed:** SmolLM is slower per token despite similar size, because it
  has 30 thinner layers against GPT-2's 12 wide ones. That's more
  sequential steps, each too small to keep the CPU busy.
- **Memory:** the KV cache is 37% smaller per token despite 2.5× the
  layers. That's GQA.

## Before writing the post

- **Why RoPE won:** relative positions, no learned table capping the
  context, and tricks (NTK / YaRN scaling) that stretch a trained model
  to longer contexts.
- **Why GQA matters more on GPUs:** decode is memory-bound, and the KV
  cache is read every step. Fewer k/v heads means fewer bytes per token.
  Llama 2 70B uses 8 k/v heads for 64 query heads.
- **The digit split:** one tokenizer choice changes what a model finds
  easy. Individual digits make arithmetic learnable. GPT-2's " 1994" as
  one token hides the digits.
- **What's still missing for a real chat model:** an instruction-tuned
  checkpoint (SmolLM-135M-Instruct) and its chat template, which becomes
  `/v1/chat/completions`.

## What AI wrote

- **Scaffold:** this README, all tests, `LlamaConfig`, `expected_shapes`,
  `HF_NAMES`, `random_params`, `Llama.from_dir`/`new_cache`/`generate`,
  the tokenizer's `encode` loop, `src/engine.py`, `serve.py`,
  `data/download_smollm.py`, `benchmarks/compare.py`, the Makefile,
  `POST.md`.
- **Hand-written:** RMSNorm, SiLU, SwiGLU, the RoPE tables and rotation,
  `repeat_kv`, GQA attention, weight mapping, `forward`, and the
  `tokenizer.json` loader with the digit split.

## Post status

Not started. Outline in `POST.md`.
