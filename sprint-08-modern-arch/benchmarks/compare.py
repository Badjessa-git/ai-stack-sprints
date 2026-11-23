"""GPT-2 (2019) vs SmolLM-135M (2024): same size, five years apart.

    python3 benchmarks/compare.py           # needs both downloads; ~2-3 min

Two texts, scored in bits per byte (fair across their different
tokenizers): the held-out end of Sprint 3's WikiText-103 file (as in
Sprint 4's bits_per_byte.py), and this repo's own sprint READMEs
(technical prose neither model trained on). Plus decode speed and
KV-cache memory per token. Which model "wins" depends on the text.
"""

import glob
import os
import re
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from src.deps import REPO, gpt2_dir, smollm_dir, sprint  # noqa: E402
from src.hf_tokenizer import HFTokenizer  # noqa: E402
from src.llama import Llama  # noqa: E402

CORPUS = os.path.join(REPO, "sprint-03-transformer", "data", "wiki103-100m.txt")
N_BYTES = 30_000


def bpb(forward, ids, decode, window=1024):
    nll, nbytes = 0.0, 0
    for s in range(0, len(ids) - 1, window):
        w = ids[s:s + window]
        if len(w) < 2:
            break
        lg = forward(w).astype(np.float64)
        lg -= lg.max(axis=1, keepdims=True)
        lp = lg - np.log(np.exp(lg).sum(axis=1, keepdims=True))
        nll += -lp[np.arange(len(w) - 1), w[1:]].sum()
        nbytes += len(decode(w[1:]).encode())
    return nll / np.log(2) / nbytes


def decode_speed(step, n=32):
    step()
    t = time.time()
    for _ in range(n):
        step()
    return n / (time.time() - t)


if __name__ == "__main__":
    text = open(CORPUS).read()
    sample = text[int(len(text) * 0.99):][:N_BYTES]
    s4, s5 = sprint("sprint-04-gpt2"), sprint("sprint-05-kv-cache")

    g = s4.gpt2.GPT2.from_dir(gpt2_dir())
    gt = s4.gpt2_tokenizer.GPT2Tokenizer.from_files(os.path.join(gpt2_dir(), "vocab.json"),
                                                    os.path.join(gpt2_dir(), "merges.txt"))
    m = Llama.from_dir(smollm_dir())
    mt = HFTokenizer.from_tokenizer_json(os.path.join(smollm_dir(), "tokenizer.json"))

    gc = s5.kv_cache.KVCache.for_model(g)
    s5.kv_cache.forward_cached(g, [1] * 64, gc)
    mc = m.new_cache()
    m.forward([1] * 64, mc)

    readmes = " ".join(open(p).read() for p in sorted(glob.glob(os.path.join(REPO, "sprint-0*", "README.md"))))
    prose = re.sub(r"[|`#*]", "", readmes)[:N_BYTES]
    texts = [("WikiText", sample), ("READMEs", prose)]

    rows = []
    for name, fwd, tok, speed, nparams, kv in [
        ("GPT-2 small (2019)", g.forward, gt,
         lambda: decode_speed(lambda: s5.kv_cache.forward_cached(g, [1], gc)),
         sum(v.size for v in g.p.values()), 2 * g.cfg.n_layer * g.cfg.n_embd * 4),
        ("SmolLM-135M (2024)", m.forward, mt,
         lambda: decode_speed(lambda: m.forward([1], mc)),
         sum(v.size for v in m.p.values()), 2 * m.cfg.n_layer * m.cfg.n_kv_head * m.cfg.head_dim * 4),
    ]:
        print(f"scoring {name} ...", flush=True)
        scores = [bpb(fwd, tok.encode(t), tok.decode) for _, t in texts]
        rows.append((name, nparams, scores, speed(), kv))
    head = "".join(f"{'bpb ' + n:>14}" for n, _ in texts)
    print(f"\n{'model':<20} {'params':>8}{head} {'tokens/sec':>11} {'KV bytes/token':>15}")
    for name, n, scores, sp, kv in rows:
        cols = "".join(f"{b:>14.3f}" for b in scores)
        print(f"{name:<20} {n / 1e6:>7.0f}M{cols} {sp:>11.1f} {kv:>15,}")
