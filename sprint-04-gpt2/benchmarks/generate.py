"""Prompt GPT-2, running on your code.

    python3 benchmarks/generate.py "Albert Einstein was a" --tokens 40

Greedy decoding, no KV cache: every new token is a full forward pass over
the whole sequence, so it gets slower as the text grows. Note the
tokens/sec. Sprint 5 adds the cache and sampling.
"""

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from src.gpt2 import GPT2  # noqa: E402
from src.gpt2_tokenizer import GPT2Tokenizer  # noqa: E402

D = os.path.join(HERE, "data", "gpt2")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--tokens", type=int, default=30)
    args = ap.parse_args()
    tok = GPT2Tokenizer.from_files(os.path.join(D, "vocab.json"), os.path.join(D, "merges.txt"))
    model = GPT2.from_dir(D)
    ids = tok.encode(args.prompt)
    t = time.time()
    out = model.generate_greedy(ids, args.tokens)
    dt = time.time() - t
    print(tok.decode(out))
    print(f"\n{args.tokens} tokens in {dt:.1f}s = {args.tokens / dt:.1f} tokens/sec (no cache)")
