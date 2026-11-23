"""Float32 vs int8 vs int4 GPT-2: size, quality, speed.

    python3 benchmarks/quality.py             # ~3-5 min

Quality is bits per byte on the held-out WikiText-103 text (Sprint 4's
metric; lower is better) plus the greedy completion of one prompt. Speed
is cached decode tokens/sec with Sprint 5's forward_cached, which reads
the weights through QuantizedParams like any dict.
"""

import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.deps import REPO, gpt2_dir, sprint  # noqa: E402
from src.quant import QuantizedParams  # noqa: E402

CORPUS = os.path.join(REPO, "sprint-03-transformer", "data", "wiki103-100m.txt")


def bpb(model, ids, tok):
    nll, nbytes = 0.0, 0
    for s in range(0, len(ids) - 1, 1024):
        w = ids[s:s + 1024]
        lg = model.forward(w).astype(np.float64)
        lg -= lg.max(axis=1, keepdims=True)
        lp = lg - np.log(np.exp(lg).sum(axis=1, keepdims=True))
        nll += -lp[np.arange(len(w) - 1), w[1:]].sum()
        nbytes += len(tok.decode(w[1:]).encode())
    return nll / np.log(2) / nbytes


if __name__ == "__main__":
    s4, s5 = sprint("sprint-04-gpt2"), sprint("sprint-05-kv-cache")
    d = gpt2_dir()
    tok = s4.gpt2_tokenizer.GPT2Tokenizer.from_files(os.path.join(d, "vocab.json"), os.path.join(d, "merges.txt"))
    base = s4.gpt2.GPT2.from_dir(d)
    text = open(CORPUS).read()
    ids = tok.encode(text[int(len(text) * 0.99):][:24000])[:4096]
    prompt = tok.encode("Albert Einstein was a")

    variants = [("float32", base.p), ("int8 per-channel", QuantizedParams.from_params(base.p, 8))]
    for g in (128, 32):
        variants.append((f"int4 group {g}", QuantizedParams.from_params(base.p, 4, g)))

    print(f"{'weights':<18} {'MB':>6} {'bits/byte':>10} {'tokens/sec':>11}  greedy")
    for name, params in variants:
        m = s4.gpt2.GPT2(base.cfg, params)
        mb = (params.nbytes() if isinstance(params, QuantizedParams) else sum(v.nbytes for v in params.values())) / 1e6
        b = bpb(m, ids, tok)
        cache = s5.kv_cache.KVCache.for_model(m)
        s5.kv_cache.forward_cached(m, prompt, cache)
        t = time.time()
        for _ in range(16):
            s5.kv_cache.forward_cached(m, [1], cache)
        speed = 16 / (time.time() - t)
        text_out = tok.decode(list(s5.sampling.generate(m, prompt, 8, temperature=0)))
        print(f"{name:<18} {mb:>6.0f} {b:>10.3f} {speed:>11.1f}  {text_out!r}", flush=True)
