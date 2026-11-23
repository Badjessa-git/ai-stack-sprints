#!/usr/bin/env python3
"""Sprint 6's server, with the BatchingEngine behind it.

    python3 serve_batched.py --max-batch 8
    python3 ../sprint-06-inference-server/benchmarks/load_test.py --clients 1 4 8

The only change to the server: its lock (one request at a time) is
swapped for a no-op, because the engine now does the scheduling.
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.batch_engine import BatchingEngine  # noqa: E402
from src.deps import gpt2_dir, sprint  # noqa: E402


class NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def make_batched_server(engine, host="127.0.0.1", port=8000, verbose=False):
    server = sprint("sprint-06-inference-server").server.make_server(engine, host, port, verbose)
    server.lock = NoLock()
    return server


def load_engine(max_batch):
    s4 = sprint("sprint-04-gpt2")
    d = gpt2_dir()
    tok = s4.gpt2_tokenizer.GPT2Tokenizer.from_files(os.path.join(d, "vocab.json"), os.path.join(d, "merges.txt"))
    return BatchingEngine(s4.gpt2.GPT2.from_dir(d), tok, max_batch=max_batch)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    server = make_batched_server(load_engine(args.max_batch), args.host, args.port, args.verbose)
    print(f"serving gpt2 (continuous batching, max {args.max_batch}) on "
          f"http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
