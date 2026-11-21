# Sprint 10 post outline: vector search

Working title: **"Recall vs latency: the only chart that matters"**

## 1. The artifact

- Nearest tokens to " king" in GPT-2's embedding table, from your index.
- The benchmark table.

## 2. Exact is easy, and too slow

- Brute force in three lines. Why O(N) per query dies at scale.

## 3. Buckets: IVF

- k-means in plain words. nprobe as the dial.

## 4. A graph you can walk: HNSW

- Layers as express lanes. The two-heap search.
- The island bug: 0.58 recall with the obvious rule, 1.0 with the heuristic.
  The best story in the sprint. Tell it with the numbers.

## 5. Reading the benchmark honestly

- Comparisons vs milliseconds. Why NumPy brute force ties Python HNSW
  at 10k, and why that flips at scale or in C.
