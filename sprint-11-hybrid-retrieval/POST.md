# Sprint 11 post outline: hybrid retrieval and RAG

Working title: **"RAG is a search problem wearing a chatbot costume"**

## 1. The artifact

- The RAG run: the question, the two retrieved paragraphs, and the
  model's confident wrong answer ("created by Nintendo"). Then explain
  both failures.

## 2. Keyword search that works: BM25

- The formula in plain words: rare words count more, long documents
  don't win by being long. One worked score.
- Why an inverted index never scans every document.

## 3. Meaning without training: SIF

- Averaging GPT-2's token vectors, down-weighting common ones, removing
  the shared direction. The car/auto test.

## 4. Fusion, and the result that didn't go my way

- The evaluation table: hybrid did *not* beat BM25. Why: fusion is only
  as good as its weaker half, and SIF is weak on short queries.
- What would change it: a trained embedding model (the stretch goal).

## 5. What production RAG adds

- Chunking, rerankers, trained embedders, citations, and evaluating the
  answer as well as the retrieval.
