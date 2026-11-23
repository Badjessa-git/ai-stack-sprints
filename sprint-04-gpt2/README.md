# Sprint 04: GPT-2 in Your Own Code

## Why this sprint

Sprint 3's models write word-shaped text. That's a compute ceiling, not a
bug: OpenAI trained GPT-2 on about 40 GB of text with a cluster of GPUs.
So this sprint skips the training. It takes OpenAI's trained weights and
runs them on **code you wrote**: your transformer, your tokenizer, your
file reader. No PyTorch, no `transformers`, no `tiktoken`.

Nothing here is new architecture. GPT-2 is the model you built in
Sprint 3, and its tokenizer is the algorithm you built in Sprint 1 Part 3.
The work is making your code agree with OpenAI's to the last detail:
the byte layout of the weight file, the name of every tensor, which way
each matrix is stored, which GELU it uses. Get one detail wrong and the
model still runs, but writes garbage. That's the lesson: loading a model
correctly is mostly checking.

By the end, your code completes "Albert Einstein was a" with
"great scientist, but he was also a great philosopher", and you can put
GPT-2 in the same bits-per-byte table as your Sprint 3 runs.

Learning goals:

1. Read a weight file format from its spec: header, offsets, dtypes,
   including bfloat16, which NumPy doesn't have.
2. Explain GPT-2's byte-to-character table, and why GPT-2's merge rank
   isn't its token id (your Part 3 assumed it was).
3. Write a forward-only transformer, and explain why serving needs no
   autograd.
4. Map a real checkpoint onto your code, strictly: a missing name, a
   wrong shape, or an unknown tensor is an error, never a silent guess.

Prep: Jay Mody's "GPT in 60 Lines of NumPy" covers the same forward pass.
Read it *after* writing yours, then compare. The safetensors format spec
is on Hugging Face's docs (`safetensors` → "Metadata parsing").

## The plan

| Step | File | What | Tests |
|------|------|------|-------|
| 1 | `src/safetensors.py` | `read_header`, `bf16_to_f32`, `load` | `tests/test_safetensors.py` |
| 2 | `src/gpt2_tokenizer.py` | `bytes_to_unicode`, `build_ranks`, `_encode_chunk`, `decode` | `tests/test_tokenizer.py` |
| 3 | `src/gpt2.py` | `layer_norm`, `gelu`, `softmax` | `TestOps` |
| 4 | | `attention`: all heads with one reshape | `TestAttention` |
| 5 | | `mlp`, `block` | `TestBlock` |
| 6 | | `map_hf_names` | `TestLoading` |
| 7 | | `GPT2.forward` | `TestModel` |
| 8 | download, then the real thing | `make download && make test-real` | `tests/test_real_gpt2.py` |

Step 8 is the acceptance test. It loads the real 124M checkpoint and
checks three things:
- the tokenizer produces the ids Hugging Face documents
- the model's loss on a paragraph is GPT-2's, not a broken model's
- greedy decoding gives exactly "Albert Einstein was a great scientist, but"

Before Step 6, print the checkpoint's names. They tell you most of what
`map_hf_names` has to do:

```python
from src.safetensors import read_header
h, _ = read_header("data/gpt2/model.safetensors"); print(sorted(h)[:20])
```

**Three details that break it silently** (the tests catch each one):
- **Conv1D layout:** GPT-2 stores `c_attn.weight` as (768, 2304), so it's
  `x @ W`, not `x @ W.T`. Sprint 3's Linear multiplied the other way.
- **Tied head:** there's no output matrix. The logits are `x @ wte.T`, the
  token embedding used backwards.
- **GELU:** GPT-2 uses the tanh approximation. The exact erf version is
  close enough to run, but not close enough to match.

**One detail that breaks it *slowly*:** `np.sqrt(D)` and `np.pi` are
NumPy float64 scalars. In NumPy 2 they promote float32 arrays to float64,
so every matmul after that converts the weights on every call. The output
is identical and it runs about 5× slower. The reference solution had this
bug in both `attention` and `gelu`: 2.4 tokens/sec, 12.3 after the fix.
Use `math.sqrt` and `math.pi`. `test_stays_float32` checks for it.

## Run it

```bash
make test                                   # 45 tests, no download needed
make download                               # ~550 MB into data/gpt2/ (gitignored)
make test-real                              # 6 acceptance tests on real GPT-2
make generate TEXT="The meaning of life is"
make bpb                                    # GPT-2 on Sprint 3's held-out text
```

NumPy only. The Sprint 1 Part 3 tokenizer is imported from
`../sprint-01-bpe-tokenizer/part3/` for its split pattern.

## Results

Measured with the reference solution on the dev Mac (2 threads):

| | |
|---|---|
| Load 124M parameters | 0.1 s (the file is memory-mapped) |
| Greedy, no cache | ~12 tokens/sec at 16 tokens of context, ~1.3 at 400 |
| GPT-2 small, held-out WikiText-103 | **~1.12 bits/byte** |
| Your best Sprint 3 model (d256-n2, regex 4k) | 1.571 bits/byte |

## Before writing the post

- **Why safetensors exists.** PyTorch's older `.bin` format is a Python
  pickle, and unpickling can run arbitrary code. safetensors is a JSON
  header plus raw bytes: nothing to execute, and it can be memory-mapped.
- **The gap between 1.12 and 1.57.** GPT-2 has about 33× the parameters
  of your d256 model and saw roughly 400× the text. Same architecture.
  The gap is scale, not code.
- **Is the comparison fair?** GPT-2 never trained on Wikipedia (OpenAI
  removed it from WebText), and your models did. That makes GPT-2's
  number, if anything, harder-won.
- **What's slow and why:** every token reruns the full sequence. That's
  Sprint 5.

## What's next

- **Sprint 05:** a KV cache and sampling.
- **Sprint 04-finetuning** ([`../sprint-04-finetuning/`](../sprint-04-finetuning/)):
  its Track 2 assumed GPT-2 needed
  PyTorch. After this sprint it doesn't. Fine-tuning GPT-2 needs a
  backward pass, so it pairs with Sprint 3.5's tensor autograd.

## What AI wrote

- **Scaffold:** this README, all tests (including the real-weights
  acceptance tests), `data/download_gpt2.py`, `benchmarks/`, the Makefile,
  `POST.md`. In `src/`: the safetensors writer (`save`, used by the tests),
  `GPT2Config`, `expected_shapes`, `random_params`, the tokenizer's
  file loading and `encode` loop, `GPT2.from_dir`, `generate_greedy`.
- **Hand-written:** everything marked `NotImplementedError`: the reader,
  the byte table, rank lookup and chunk encoding, decoding, every model
  op, weight mapping, and `forward`.

## Post status

Not started. Outline in `POST.md`.
