#!/usr/bin/env python3
"""SmolLM-135M behind your Sprint 6 server.

    python3 serve.py --port 8000
    curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \\
      -d '{"prompt": "The capital of France is", "max_tokens": 20, "temperature": 0}'
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.deps import sprint  # noqa: E402
from src.engine import SmolLMEngine  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    engine = SmolLMEngine()
    server = sprint("sprint-06-inference-server").server.make_server(engine, args.host, args.port, args.verbose)
    print(f"serving {engine.name} on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
