# Rebuild the Modern AI Stack — biweekly build-and-write sprints

One repo, one sprint every other weekend, 48 hours each. AI scaffolds
(spec, tests, benchmarks, post outline); the core implementation is written
by hand. Every sprint ends with a technical post — drafts live in each
sprint folder as `POST.md`, finished posts are indexed under `posts/`.

## Why one repo

The sprints are layers of one stack, and later ones reuse earlier code:

- Sprint 05 (hybrid retrieval) imports the Sprint 01 tokenizer and the
  Sprint 04 vector search
- Sprint 08 (batching scheduler) builds on Sprint 03 + Sprint 06
- The optional capstone assembles 01 + 03 + 06 + 08 into one toy
  inference server

Separate repos would make that reuse copy-paste. Here it's an import.

## Roadmap

| # | Sprint | Foundations | Post (working title) |
|---|--------|-------------|----------------------|
| 01 | BPE tokenizer | hashmaps, greedy merging | Why LLMs can't count the r's in strawberry |
| 02 | Autograd engine | DAGs, topological sort | Backprop is just the chain rule with bookkeeping |
| 03 | Transformer + tiny LM | attention, LayerNorm | Attention, implemented until it stopped being magic |
| 04 | Vector search engine | heaps, graphs (HNSW/IVF) | Recall vs latency: the only chart that matters |
| 05 | Hybrid retrieval for RAG | inverted index, BM25 | RAG is a search problem wearing a chatbot costume |
| 06 | KV cache + paged inference | memory allocation, block tables | Inference is memory-bound — here's the benchmark |
| 07 | LSM-tree KV store | sorted runs, Bloom filters | Write fast, read… complicated |
| 08 | Continuous batching scheduler | queues, scheduling | Why GPU schedulers don't work like web servers |

## Rules

1. Core files are hand-written — no AI-generated implementation, no
   autocomplete for the core file of each sprint.
2. Tests are the spec. Don't edit a test to make it pass; flag disagreements.
3. A sprint is done when: tests pass, the benchmark has been run, and the
   post draft exists. Stretch goals never block "done".
