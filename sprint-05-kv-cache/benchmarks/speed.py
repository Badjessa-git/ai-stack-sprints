"""Tokens/sec with and without the KV cache, on real GPT-2.

    python3 benchmarks/speed.py                  # 64 new tokens
    python3 benchmarks/speed.py --new 200 --prompt-len 300

Without the cache, each new token reruns the whole sequence, so it slows
down as the text grows. With it, each token is one token's worth of work
plus attention over the cache.
"""

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.deps import gpt2_dir, sprint  # noqa: E402
from src.sampling import generate  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--prompt-len", type=int, default=16)
    args = ap.parse_args()
    model = sprint("sprint-04-gpt2").gpt2.GPT2.from_dir(gpt2_dir())
    prompt = list(range(1000, 1000 + args.prompt_len))

    t = time.time()
    model.generate_greedy(prompt, args.new)
    slow = time.time() - t
    t = time.time()
    list(generate(model, prompt, args.new, temperature=0))
    fast = time.time() - t
    print(f"prompt {args.prompt_len} tokens, {args.new} new tokens")
    print(f"  no cache: {slow:6.1f}s  {args.new / slow:6.1f} tokens/sec")
    print(f"  KV cache: {fast:6.1f}s  {args.new / fast:6.1f} tokens/sec  ({slow / fast:.1f}x)")
