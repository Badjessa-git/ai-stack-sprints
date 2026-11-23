"""GPT-2 vs your Sprint 3 models, on the same metric.

    python3 benchmarks/bits_per_byte.py            # ~32k tokens, ~1-2 min

Scores GPT-2 on the held-out end of Sprint 3's WikiText-103 file (the last
1% of the text, the region Sprint 3's val_bpb is measured on) in bits per
byte, so the number sits in the same table as your runs. Windows of 1,024
tokens; every token after the first in a window is scored.
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from src.gpt2 import GPT2  # noqa: E402
from src.gpt2_tokenizer import GPT2Tokenizer  # noqa: E402

D = os.path.join(HERE, "data", "gpt2")
CORPUS = os.path.join(os.path.dirname(HERE), "sprint-03-transformer", "data", "wiki103-100m.txt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--tokens", type=int, default=32768)
    args = ap.parse_args()
    tok = GPT2Tokenizer.from_files(os.path.join(D, "vocab.json"), os.path.join(D, "merges.txt"))
    model = GPT2.from_dir(D)
    text = open(args.corpus).read()
    held_out = text[int(len(text) * 0.99):]
    ids = tok.encode(held_out[: args.tokens * 6])[: args.tokens]
    total_nll, total_bytes, t = 0.0, 0, time.time()
    for start in range(0, len(ids) - 1, 1024):
        w = ids[start:start + 1024]
        if len(w) < 2:
            break
        lg = model.forward(w).astype(np.float64)
        lg -= lg.max(axis=1, keepdims=True)
        lp = lg - np.log(np.exp(lg).sum(axis=1, keepdims=True))
        total_nll += -lp[np.arange(len(w) - 1), w[1:]].sum()
        total_bytes += len(tok.decode(w[1:]).encode())
        print(f"\r{start + len(w):,}/{len(ids):,} tokens  "
              f"{total_nll / np.log(2) / total_bytes:.3f} bpb  ({time.time() - t:.0f}s)", end="", flush=True)
    print(f"\n\nGPT-2 small (124M): {total_nll / np.log(2) / total_bytes:.3f} bits/byte")
    print("Your best Sprint 3 runs: see sprint-03-transformer/README.md 'Scaling up'")
