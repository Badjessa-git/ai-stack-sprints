"""Where does a decode step's time go? (The "speed" half of the sprint.)

    python3 benchmarks/profile_decode.py                 # float32 weights
    python3 benchmarks/profile_decode.py --bits 8        # through QuantizedParams

Times one cached decode step on real GPT-2 by part: the 12 blocks' four
matmuls, attention over the cache, layer norms + GELU, and the output head
(x @ wte.T: 50,257 x 768, 31% of all weights). Use it to decide what's
worth optimizing, then measure again.
"""

import argparse
import math
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.deps import gpt2_dir, sprint  # noqa: E402
from src.quant import QuantizedParams  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, choices=[8, 4], default=None)
    ap.add_argument("--context", type=int, default=256)
    args = ap.parse_args()
    s4 = sprint("sprint-04-gpt2")
    g = s4.gpt2
    base = g.GPT2.from_dir(gpt2_dir())
    p = QuantizedParams.from_params(base.p, args.bits) if args.bits else base.p
    C, H = base.cfg.n_embd, base.cfg.n_head
    D = C // H
    x = np.random.default_rng(0).standard_normal((1, C)).astype(np.float32)
    k_cache = np.random.default_rng(1).standard_normal((H, args.context, D)).astype(np.float32)
    t = {"matmuls (4 per block)": 0.0, "attention over cache": 0.0, "layer norm + gelu": 0.0, "output head": 0.0}
    reps = 5
    for _ in range(reps):
        for i in range(base.cfg.n_layer):
            h = f"h.{i}."
            t0 = time.perf_counter()
            qkv = x @ p[h + "attn.c_attn.weight"]
            a = x @ p[h + "attn.c_proj.weight"]
            f = x @ p[h + "mlp.c_fc.weight"]
            _ = f @ p[h + "mlp.c_proj.weight"]
            t1 = time.perf_counter()
            q = qkv[:, :C].reshape(H, 1, D)
            s = g.softmax(q @ k_cache.transpose(0, 2, 1) / math.sqrt(D))
            _ = s @ k_cache
            t2 = time.perf_counter()
            _ = g.layer_norm(x, p[h + "ln_1.weight"], p[h + "ln_1.bias"])
            _ = g.gelu(f)
            t3 = time.perf_counter()
            t["matmuls (4 per block)"] += t1 - t0
            t["attention over cache"] += t2 - t1
            t["layer norm + gelu"] += t3 - t2
        t0 = time.perf_counter()
        _ = x @ p["wte.weight"].T
        t["output head"] += time.perf_counter() - t0
    total = sum(t.values())
    print(f"one decode step, context {args.context}, weights {'int' + str(args.bits) if args.bits else 'float32'}\n")
    for k, v in t.items():
        print(f"  {k:<24} {v / reps * 1000:7.2f} ms  {v / total:5.0%}")
    print(f"  {'total':<24} {total / reps * 1000:7.2f} ms")
