# Rebuild the Modern AI Stack: build-and-write sprints

One repo, one sprint at a time. Each sprint builds one layer of the stack
from scratch in Python and NumPy, measures it, and ends with a technical
post. AI writes the scaffold (spec, tests, benchmarks, post outline). The
core of each sprint is written by hand, and each sprint README says exactly
which parts AI wrote.

**End goal:** an LLM inference server written from scratch. It loads real
pretrained weights (GPT-2, then a small modern model) into your own
transformer, streams tokens over HTTP, and batches many requests at once.

## Roadmap

Every sprint below is scaffolded: README, tests, stubs, benchmarks and a
post outline. Each test suite was checked against a full reference
solution, which is not in the repo. The stubs fail only with
`NotImplementedError` until you write them.

| # | Sprint | You build by hand | Milestone | Status |
|---|--------|-------------------|-----------|--------|
| 01 | [BPE tokenizer](sprint-01-bpe-tokenizer/) | Byte-level BPE; Part 2 speed pass; Part 3 regex pre-split | Lossless, 2× compression | ✅ |
| 02 | [Autograd engine](sprint-02-autograd-engine/) | Scalar `Value` and backprop | Trains an MLP on moons | ✅ |
| 03 | [Transformer + tiny LM](sprint-03-transformer/) | Attention, block, training loop; v2 fixes; v3 tokenizer grid | 1.57 bits/byte on WikiText-103 | 🟡 wrapping up |
| 3.5 | [Tensor autograd](sprint-03b-tensor-autograd/) *(optional)* | NumPy `Tensor` backward: broadcasting, batched matmul, fused softmax | The fast engine is yours too | ⬜ |
| 04 | [GPT-2 in your code](sprint-04-gpt2/) | safetensors reader, GPT-2 byte table on the Part 3 tokenizer, forward-only GPT-2 | **"Albert Einstein was a great scientist"**, 1.12 bits/byte | ⬜ |
| 05 | [KV cache + sampling](sprint-05-kv-cache/) | KV cache with an offset mask; temperature, top-k, top-p; streaming `generate` | 22× faster at 400 tokens of context | ⬜ |
| 06 | [Inference server](sprint-06-inference-server/) | OpenAI-style `/v1/completions`, SSE streaming, UTF-8 and stop strings across tokens | **`curl` your own server running GPT-2** | ⬜ |
| 07 | [Continuous batching + paged KV](sprint-07-batching/) | Scheduler simulator, block allocator, batched decode, preemption | 4 clients: 55 → 101 tokens/sec | ⬜ |
| 08 | [Modern architecture](sprint-08-modern-arch/) | RMSNorm, RoPE, SwiGLU, GQA; `tokenizer.json` | **SmolLM-135M served from your code** ("...is Paris.") | ⬜ |
| 09 | [Quantization + speed](sprint-09-quantization/) | int8 per-channel, int4 per-group, nibble packing, a decode profiler | Half the size, same text | ⬜ |

Why this order: GPT-2 is the architecture Sprint 03 already built, and its
tokenizer is the algorithm Sprint 01 Part 3 built. That makes 04 the
cheapest way to prove your code is a real LLM. After 06 you serve an LLM
from your own code. 08 and 09 bring it from GPT-2 (2019) up to how current
models are built.

**Data-systems track** (later, or in parallel, off the serving path):

| # | Sprint | You build by hand | Post (working title) | Status |
|---|--------|-------------------|----------------------|--------|
| 10 | [Vector search](sprint-10-vector-search/) | Brute force, k-means/IVF, HNSW with the neighbour heuristic | Recall vs latency: the only chart that matters | ⬜ |
| 11 | [Hybrid retrieval + RAG](sprint-11-hybrid-retrieval/) | Inverted index, BM25, SIF embeddings, RRF; RAG against your server | RAG is a search problem wearing a chatbot costume | ⬜ |
| 12 | [LSM-tree KV store](sprint-12-lsm-tree/) | WAL, SSTables, Bloom filters, compaction, crash recovery | Write fast, read… complicated | ⬜ |

**Side quest: fine-tuning.** A Sprint 4 fine-tuning scaffold (play
classification) lives on the branch `scaffold/sprint-04-finetuning`. Its
Track 2 assumed GPT-2 needed PyTorch. After Sprint 04 it doesn't, and
with Sprint 3.5's autograd you can fine-tune GPT-2 on your own engine.
It fits after Sprint 04 or 05.

## How the sprints depend on each other

Later sprints import your earlier code directly (`src/deps.py`), so each
builds on what you wrote, not on a copy:

```
01 tokenizer ──► 03 transformer ──► 04 GPT-2 ──► 05 KV cache ──► 06 server ──► 07 batching
                                       │              │              ▲
                                       │              └──► 08 SmolLM ─┘ (served by 06)
                                       ├──► 09 quantization (runs 04's model unchanged)
                                       └──► 10 vector search (GPT-2's embeddings) ──► 11 retrieval + RAG (asks 08's server)
02 autograd ──► 3.5 tensor autograd                                       12 LSM tree (standalone)
```

Models are downloaded once: GPT-2 by `make download` in Sprint 04 (~550
MB), SmolLM by `make download` in Sprint 08 (~540 MB). Both are
gitignored. Tests that need them skip until they're there.

## Setup

- **Python 3.9 or newer.** Developed on macOS with the system Python 3.9.6.
- **Sprints 01–02:** standard library only.
- **Sprint 03 and later:** `pip install numpy` (developed on 2.0.2).
  Training scripts set `VECLIB_MAXIMUM_THREADS=1` so parallel runs don't
  fight over cores.
- **WikiText-103 (Sprint 03):** the 100M-character training file
  `sprint-03-transformer/data/wiki103-100m.txt` isn't in git. To build it,
  follow the steps in `sprint-03-transformer/download_wiki103.py`: two
  Parquet files from Hugging Face, plus `pip install pyarrow`.
- **Large files stay local.** Token caches (`data/*.tokens.npy`,
  `*.tokens.json`) are rebuilt by the scripts. Model weights
  (`runs/**/weights.npz`, `runs/legacy/weights-*.json`) are gitignored:
  the runs are a learning record, so git keeps their logs, configs, merges
  and reports, not the models. `benchmarks/convert_legacy_weights.py`
  shrinks the old JSON weights to `.npz` (about 5× smaller) on disk.

## Run every test

```bash
cd sprint-01-bpe-tokenizer && make test                         # Part 1: 15 tests
python3 -m unittest part2/test_equivalence.py -v                # Part 2: identical to v1
python3 -m unittest part3/test_regex_tokenizer.py -v            # Part 3: 34 tests
cd ../sprint-02-autograd-engine && make test
cd ../sprint-03-transformer && make test
for s in ../sprint-03b-* ../sprint-0[4-9]-* ../sprint-1[0-2]-*; do (cd $s && make test); done
```

## Repo map

```
README.md                  this file: roadmap, setup, rules
posts/                     finished posts, indexed
sprint-NN-<name>/
  README.md                the sprint spec (template below)
  POST.md                  the post draft
  src/                     the hand-written core
  tests/                   the spec, as tests
  benchmarks/              timing and training scripts
  partN/                   follow-up parts, each with its own README and tests
  runs/<version>/<run>/    one folder per training run: config.json,
                           train.log, weights.npz, merges.json (Sprint 03)
  reports/                 dated result tables quoted in posts
```

## Rules

1. **The core is hand-written.** No AI-generated implementation and no
   autocomplete in each sprint's core files. AI may scaffold, explain,
   review, and ask debugging questions.
2. **Say what AI wrote.** Every sprint README ends with a "What AI wrote"
   section that lists the parts that aren't hand-written: scaffold, fixes
   to shared code, any piece you asked for outright. The posts link to it.
3. **Tests are the spec.** Don't edit a test to make it pass; flag
   disagreements.
4. **Don't rewrite history.** To change code that produced published
   results, make a new version (`train_v2.py`, `src/numpy_v2/`, `partN/`)
   and leave the old one as it was. Old runs stay reproducible and their
   logs stay comparable.
5. **A sprint is done when** tests pass, the benchmark has been run, and
   the post draft exists. Stretch goals never block "done".

## Cadence

Each sprint has a **48-hour core** (a weekend) that ends in a post draft.
After that, optional **Part N follow-ups** go deeper on something the
core uncovered: Sprint 01 Part 2 (speed) and Part 3 (regex split), and
Sprint 03 v2 (bug fixes) and v3 (tokenizer). They take as long as they
take, and each one can be its own post.

## Sprint README template

```
# Sprint NN: <name>
## Why this sprint        what it is, in plain words, and where it fits
## Learning goals         3–4 things you can explain afterwards
## The plan               numbered steps, each with the tests that prove it
## Run it                 exact commands
## Results                measured numbers, linked to reports/
## Before writing the post  where the toy stops and production starts
## What AI wrote          everything not hand-written
## Post status            draft / published + link
```
