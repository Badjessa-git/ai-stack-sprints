# Sprint 11: Hybrid Retrieval and RAG (data-systems track)

## Why this sprint

Retrieval-augmented generation (RAG) means: find the passages that
answer a question, paste them into the prompt, and let the model answer
from them. The model can only be as right as what retrieval hands it.
So RAG is mostly a search problem, and this sprint builds the search.

There are two families, with opposite strengths:

- **Lexical (BM25):** score documents by the query's words. Rare words
  count more, and long documents don't win just by being long. Exact
  names and numbers are its home ground. It misses synonyms.
- **Dense:** turn texts into vectors and take the nearest (your Sprint 10
  search). It catches meaning, not spelling. Here the vectors are
  **SIF** embeddings, a weighted average of GPT-2's own token vectors with
  the shared "this is English" direction removed. That's the simplest
  real sentence embedding there is, with no training.

**Hybrid** runs both and fuses the rankings with Reciprocal Rank Fusion,
which needs only ranks, not comparable scores. Then `rag.py` closes the
loop: your index retrieves, and your Sprint 8 server writes the answer.

Learning goals:

1. Explain BM25 term by term, and why an inverted index never scans
   every document.
2. Build a sentence embedding from token vectors, and explain each of
   SIF's three steps.
3. Fuse rankings with RRF, and measure honestly when fusion helps and
   when it doesn't.
4. Run RAG end to end and diagnose its failures: retrieval vs generation.

## The plan

Everything is in `src/retrieval.py`. The evaluation needs Sprint 3's
WikiText file and Sprint 4's GPT-2 download, and it uses your Sprint 10
`BruteForce`.

| Step | What | Tests |
|------|------|-------|
| 1 | `tokenize`, `InvertedIndex.add`, `idf`, `bm25_scores`, `search` | `TestBM25` |
| 2 | `SIFEmbedder.fit`, `_finish` | `TestSIF` |
| 3 | `rrf` (with weights), `hybrid_search` | `TestFusion` |
| 4 | `recall_at_k`, `mrr` | `TestMetrics` |
| 5 | `make eval`, `make eval-hard`, `make rag` | |

**The benchmark** (`src/data.py`, scaffold) is cut from WikiText-103:
- Documents are paragraphs: 10,976 of them from 500 articles.
- Each query is one section, written as "<article> <section heading>",
  for example "Valkyria Chronicles III Gameplay". There are 2,008.
- A query's relevant documents are that section's paragraphs.

It favours BM25, because the article name is usually in the paragraphs.
`--headings-only` drops the article name, which is much harder.

## Run it

```bash
make test        # 18 tests, instant
make eval        # ~10 s after loading
make eval-hard   # headings-only queries
make rag Q="What kind of game is Valkyria Chronicles III?"   # with a server running
```

## Results

Reference solution, dev Mac (`make eval`):

| System | Recall@10 | MRR | Headings-only recall@10 |
|---|---|---|---|
| BM25 | **0.458** | **0.371** | 0.106 |
| dense (SIF) | 0.300 | 0.269 | 0.065 |
| hybrid, RRF 1:1 | 0.399 | 0.321 | 0.099 |
| hybrid, RRF 4:1 (BM25 weighted) | 0.440 | 0.350 | **0.109** |

**Hybrid didn't win here.** Equal-weight fusion made BM25 *worse*, and
weighting BM25 4:1 only gets back to roughly where BM25 was. Fusion is
only as good as its weaker half, and SIF over GPT-2's token vectors is
weak on short queries. Hybrid is the industry default because trained
embedding models are much stronger than SIF. Training one with your
Sprint 3.5 autograd (contrastive loss on paragraph pairs) is the
stretch goal that would test that claim on your own numbers.

**RAG, end to end** (`make rag`, SmolLM-135M from Sprint 8):

> Sources: [9] Valkyria Chronicles III: Concept work ...; [13] ... was not released in the west ...
> Answer: Valkyria Chronicles III is a 2012 American action role-playing game (RPG) created by Nintendo.

Both classic failures in one answer:
- **Retrieval missed:** the intro paragraph that says "tactical
  role-playing video game developed by Sega" wasn't retrieved.
- **Generation wasn't grounded:** a 135M base model wrote a fluent,
  wrong answer anyway.

Debug RAG in that order: retrieval first, then generation.

## Before writing the post

- **What production RAG adds:** chunking (paragraphs are a choice), a
  reranker (a second model rescoring the top 50), trained embedders,
  query rewriting, citations, and evaluating answers as well as
  retrieval.
- **Why BM25 is still everywhere:** it's fast, needs no model, handles
  rare names and codes perfectly, and is hard to beat on keyword-heavy
  queries, as this table shows.
- **Why RRF:** BM25 scores and cosine similarities aren't on the same
  scale. RRF only needs ranks.

## What AI wrote

- **Scaffold:** this README, all tests, `src/data.py` (the WikiText
  benchmark), `SIFEmbedder.__init__`/`_raw`/`embed`,
  `benchmarks/evaluate.py`, `rag.py`, the Makefile, `POST.md`.
- **Hand-written:** tokenizer, inverted index, IDF, BM25 scoring and
  search, SIF fitting and the principal-component removal, RRF, hybrid
  search, recall and MRR.

## Post status

Not started. Outline in `POST.md`.
