"""Recall vs latency: the only chart that matters for vector search.

    python3 benchmarks/recall_latency.py                # GPT-2 token embeddings
    python3 benchmarks/recall_latency.py --synthetic    # no download needed

Real vectors: GPT-2's token embedding table (wte, from Sprint 4's
download): 768-d vectors, one per token. Queries are other tokens'
vectors, so "nearest neighbours of ' king'" is a real question here.
Every index answers 200 queries; recall@10 is against brute force.
"compared" is how many stored vectors each query was scored against.

Expect a surprise at this size: NumPy's brute force is one BLAS matrix
multiply, and the indexes are Python loops, so brute force can win on
milliseconds while losing badly on vectors compared. The comparison count
is what scales; the milliseconds are about Python.
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.deps import gpt2_dir, sprint  # noqa: E402
from src.vector_search import HNSW, BruteForce, IVFIndex, clustered, normalize, recall_at_k, top_k  # noqa: E402


def timed(fn, queries):
    t = time.perf_counter()
    out = [fn(q) for q in queries]
    return out, (time.perf_counter() - t) / len(queries) * 1000


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    if args.synthetic:
        X = clustered(args.n, 64, clusters=50, seed=0)
        label = f"{args.n} synthetic 64-d vectors"
        tok = None
    else:
        s4 = sprint("sprint-04-gpt2")
        wte = s4.safetensors.load(os.path.join(gpt2_dir(), "model.safetensors"), names={"wte.weight"})["wte.weight"]
        X = wte[: args.n]
        label = f"first {args.n} GPT-2 token embeddings (768-d)"
        tok = s4.gpt2_tokenizer.GPT2Tokenizer.from_files(os.path.join(gpt2_dir(), "vocab.json"),
                                                         os.path.join(gpt2_dir(), "merges.txt"))
    rng = np.random.default_rng(1)
    qi = rng.choice(len(X), 200, replace=False)
    Q = X[qi] + 0.01 * rng.standard_normal(X[qi].shape).astype(np.float32)

    bf = BruteForce(X)
    truth, bf_ms = timed(lambda q: bf.search(q, 10)[0], Q)
    print(f"{label}, 200 queries, recall@10\n")
    print(f"{'index':<26} {'build':>7} {'recall':>7} {'ms/query':>9} {'compared':>9}")
    print(f"{'brute force':<26} {'-':>7} {1.0:>7.3f} {bf_ms:>9.2f} {len(X):>9,}")

    t = time.time()
    ivf = IVFIndex(n_lists=int(np.sqrt(len(X))), seed=0).build(X)
    ivf_build = time.time() - t
    for nprobe in (1, 4, 16):
        found, ms = timed(lambda q: ivf.search(q, 10, nprobe)[0], Q)
        r = np.mean([recall_at_k(f, tr) for f, tr in zip(found, truth)])
        cmp = np.mean([sum(len(ivf.lists[j]) for j in top_k(ivf.centroids @ normalize(q), nprobe)) for q in Q])
        print(f"{f'IVF nprobe={nprobe}':<26} {ivf_build:>6.1f}s {r:>7.3f} {ms:>9.2f} {cmp:>9,.0f}")

    t = time.time()
    h = HNSW(M=16, ef_construction=100, seed=0).build(X)
    h_build = time.time() - t
    for ef in (16, 64, 128):
        h.n_dist = 0
        found, ms = timed(lambda q: h.search(q, 10, ef)[0], Q)
        r = np.mean([recall_at_k(f, tr) for f, tr in zip(found, truth)])
        print(f"{f'HNSW ef={ef}':<26} {h_build:>6.1f}s {r:>7.3f} {ms:>9.2f} {h.n_dist / len(Q):>9,.0f}")

    if tok:
        word = " king"
        i = tok.encode(word)[0]
        ids, _ = bf.search(X[i], 8)
        print(f"\nnearest tokens to {word!r}: " + ", ".join(repr(tok.decode([int(j)])) for j in ids[1:]))
