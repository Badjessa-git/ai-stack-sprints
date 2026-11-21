#!/usr/bin/env python3
"""Run the server.

    python3 serve.py                 # GPT-2 (Sprint 4's data/gpt2/) on :8000
    python3 serve.py --fake          # the fake engine, no model needed
    python3 serve.py --port 8080 -v  # log every request

Then, from another terminal:

    curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \\
      -d '{"prompt": "Albert Einstein was a", "max_tokens": 30, "temperature": 0.8}'

    curl -N localhost:8000/v1/completions -H 'Content-Type: application/json' \\
      -d '{"prompt": "Once upon a time", "max_tokens": 60, "stream": true}'
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.engine import FakeEngine, GPT2Engine  # noqa: E402
from src.server import make_server  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    engine = FakeEngine(delay=0.02) if args.fake else GPT2Engine()
    server = make_server(engine, args.host, args.port, args.verbose)
    print(f"serving {engine.name} on http://{args.host}:{args.port}  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
