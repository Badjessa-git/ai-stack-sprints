"""BM25 vs dense (SIF over GPT-2's wte) vs hybrid, on WikiText sections.

    python3 benchmarks/evaluate.py                     # 500 articles
    python3 benchmarks/evaluate.py --headings-only     # harder for BM25

Needs Sprint 3's data/wiki103-100m.txt and Sprint 4's GPT-2 download
(the tokenizer and wte table for the dense side). Dense search uses your
Sprint 10 BruteForce.
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import data  # noqa: E402
from src.deps import REPO, gpt2_dir, sprint  # noqa: E402
from src.retrieval import InvertedIndex, SIFEmbedder, hybrid_search, mrr, recall_at_k  # noqa: E402

CORPUS = os.path.join(REPO, "sprint-03-transformer", "data", "wiki103-100m.txt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", type=int, default=500)
    ap.add_argument("--headings-only", action="store_true")
    args = ap.parse_args()
    docs, queries = data.load(CORPUS, args.articles, headings_only=args.headings_only)
    print(f"{len(docs):,} paragraphs, {len(queries):,} section queries"
          f"{' (headings only)' if args.headings_only else ''}\n", flush=True)

    t = time.time()
    index = InvertedIndex()
    for d, _, text in docs:
        index.add(d, text)
    print(f"BM25 index built in {time.time() - t:.1f}s, {len(index.postings):,} terms", flush=True)

    s4 = sprint("sprint-04-gpt2")
    tok = s4.gpt2_tokenizer.GPT2Tokenizer.from_files(os.path.join(gpt2_dir(), "vocab.json"),
                                                     os.path.join(gpt2_dir(), "merges.txt"))
    wte = s4.safetensors.load(os.path.join(gpt2_dir(), "model.safetensors"), names={"wte.weight"})["wte.weight"]
    t = time.time()
    sif = SIFEmbedder(wte, tok.encode)
    V = sif.fit([text for _, _, text in docs])
    bf = sprint("sprint-10-vector-search").vector_search.BruteForce(V)
    print(f"SIF embeddings built in {time.time() - t:.1f}s\n", flush=True)

    def dense(q, k):
        return [int(i) for i in bf.search(sif.embed(q), k)[0]]

    systems = {
        "BM25": lambda q: index.search(q, 10),
        "dense (SIF)": lambda q: dense(q, 10),
        "hybrid, RRF 1:1": lambda q: hybrid_search(index, dense, q, 10),
        "hybrid, RRF 4:1": lambda q: hybrid_search(index, dense, q, 10, bm25_weight=4.0),
    }
    print(f"{'system':<16} {'recall@10':>10} {'MRR':>7} {'ms/query':>9}")
    for name, fn in systems.items():
        t = time.time()
        res = [(fn(q), rel) for q, rel in queries]
        ms = (time.time() - t) / len(queries) * 1000
        r = np.mean([recall_at_k(f, rel, 10) for f, rel in res])
        m = np.mean([mrr(f, rel) for f, rel in res])
        print(f"{name:<16} {r:>10.3f} {m:>7.3f} {ms:>9.2f}")
