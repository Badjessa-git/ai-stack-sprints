"""One batched decode step on real GPT-2, at growing batch sizes.

    python3 benchmarks/batch_speed.py

If decode is memory-bound, step time barely grows with B, so throughput
(B / step time) grows almost linearly. The numbers here are what
src/sim.py's CostModel defaults came from: base = the B=1 step, per_decode
= the slope.
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.batch_engine import batched_decode  # noqa: E402
from src.deps import gpt2_dir, sprint  # noqa: E402
from src.paged_kv import PagedKVCache, SequenceView  # noqa: E402

if __name__ == "__main__":
    s4, s5 = sprint("sprint-04-gpt2"), sprint("sprint-05-kv-cache")
    m = s4.gpt2.GPT2.from_dir(gpt2_dir())
    print(f"{'batch':>5} {'step ms':>8} {'tokens/sec':>11}")
    for B in [1, 2, 4, 8, 16]:
        cache = PagedKVCache.for_model(m, num_blocks=B * 16, block_size=16)
        views = []
        for b in range(B):
            v = SequenceView(cache, b, 1024)
            s5.kv_cache.forward_cached(m, list(range(100 + b, 164 + b)), v)  # 64-token prompts
            views.append(v)
        batched_decode(m, [1] * B, views)  # warm up
        t, n = time.time(), 10
        for _ in range(n):
            batched_decode(m, [1] * B, views)
        ms = (time.time() - t) / n * 1000
        print(f"{B:>5} {ms:>8.1f} {B / ms * 1000:>11.1f}")
