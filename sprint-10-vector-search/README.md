# Sprint 10: Vector Search (data-systems track)

## Why this sprint

RAG, semantic search, recommendations and de-duplication all ask the same
question. Items are turned into vectors, and the question is "which
stored vectors are closest to this one?" Answering it exactly means
comparing against every stored vector. That's fine at 10,000 and
hopeless at a billion. Approximate indexes give up a little **recall**
(the share of the true nearest neighbours they find) for a lot less
work.

You build three:

- **Brute force:** exact, and the baseline every recall number is
  measured against.
- **IVF (inverted file):** k-means the vectors into buckets, then search
  only the few buckets whose centres are nearest the query.
- **HNSW:** a graph where each vector links to near neighbours, with
  sparser "express" layers on top. Search walks greedily downhill from
  the top layer. It's the default in most vector databases.

The real vectors come from your own work: GPT-2's token embedding table
from Sprint 4. The nearest tokens to " king" are " King", " Queen",
" leader", " president" and " god".

Learning goals:

1. Explain recall@k, and why every approximate index is a recall vs
   latency trade.
2. Implement k-means and IVF, and explain what nprobe trades.
3. Implement HNSW, including why its neighbour-selection heuristic
   matters (the test data shows it), and what ef trades.
4. Read a benchmark honestly: count the work, not only the milliseconds.

## The plan

Everything is in `src/vector_search.py`. Pure NumPy; Sprint 4's download
is only needed for the real-vector benchmark.

| Step | What | Tests |
|------|------|-------|
| 1 | `normalize`, `top_k`, `BruteForce.search`, `recall_at_k` | `TestBruteForce` |
| 2 | `kmeans` (spherical: cosine) | `TestKMeans` |
| 3 | `IVFIndex.build`, `search` | `TestIVF` |
| 4 | HNSW `_search_layer` (two heaps) | `TestHNSW` |
| 5 | HNSW `_select` (the heuristic), `_connect` | `TestHNSW` |
| 6 | HNSW `build`, `search` | `TestHNSW` |

**The heuristic, and why it's a step of its own.** The obvious way to
pick a node's links is "its M most similar nodes". On clustered data
(`clustered()` in the tests), those all sit in the node's own cluster,
so no link ever crosses to another cluster. The graph splits into islands
and a search that starts on the wrong one can't leave. With that rule,
the reference solution scored **0.58 recall**. The paper's heuristic
keeps a candidate only if it's closer to the new node than to any link
already kept, which leaves room for links *between* clusters. Recall: 1.0.
Write the simple version first, watch `test_high_recall` fail, then fix it.

## Run it

```bash
make test              # 16 tests, ~5 s
make bench             # GPT-2 token embeddings (Sprint 4's download), ~2 min
make bench-synthetic   # clustered random vectors, no download
```

## Results

Reference solution, dev Mac. First 10,000 GPT-2 token embeddings (768-d),
200 queries, recall@10 (`make bench`):

| Index | Build | Recall | ms / query | Vectors compared |
|---|---|---|---|---|
| brute force | — | 1.000 | 0.91 | 10,000 |
| IVF, nprobe 4 | 0.4 s | 0.707 | 0.53 | 711 |
| IVF, nprobe 16 | 0.4 s | 0.903 | 2.00 | 2,625 |
| HNSW, ef 64 | 86 s | 0.919 | 0.92 | 824 |
| HNSW, ef 128 | 86 s | 0.965 | 1.68 | 1,346 |

**Read the last two columns separately.**
- *Vectors compared:* HNSW reaches 92% recall looking at 8% of the data.
  That's the number that scales: at 100 million vectors, brute force
  compares 100 million and HNSW still compares about a thousand.
- *Milliseconds:* at 10,000 vectors, NumPy's brute force is one optimized
  matrix multiply, while HNSW is a Python loop doing a thousand tiny ones.
  So brute force ties it on milliseconds. In C (FAISS, hnswlib) the
  milliseconds follow the comparisons.

Token embeddings are a hard case for IVF: they don't form tidy clusters.
On the clustered synthetic data, IVF with nprobe 4 reaches recall 1.0.

## Before writing the post

- **What vector databases add:** filtering (metadata plus vectors),
  updates and deletes (HNSW deletes are notoriously awkward),
  persistence, sharding, and quantized vectors (product quantization) so
  a billion of them fit in memory.
- **IVF vs HNSW in practice:** IVF builds fast and compresses well.
  HNSW has better recall per comparison but uses more memory and builds
  slowly (86 s here, in Python).
- **Where the vectors come from:** an embedding model. Sprint 11 makes
  simple ones itself and combines them with keyword search.

## What AI wrote

- **Scaffold:** this README, all tests, `clustered()`, the HNSW class
  skeleton (`__init__`, `_sim`, `_random_level`, `_level_of_entry`), the
  benchmark, the Makefile, `POST.md`.
- **Hand-written:** `normalize`, `top_k`, brute force, recall, k-means,
  IVF, and HNSW's layer search, neighbour heuristic, linking, build and
  search.

## Post status

Not started. Outline in `POST.md`.
