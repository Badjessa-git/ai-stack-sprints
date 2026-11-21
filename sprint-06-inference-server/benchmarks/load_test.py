"""Fire N concurrent requests at a running server and report latency.

    python3 serve.py &                                  # in one terminal
    python3 benchmarks/load_test.py --clients 1 4 8     # in another

Reports, per concurrency level: time to first token (TTFT), total latency
(p50 / max), and throughput in tokens/sec across all clients. With one
request at a time under the lock, throughput stays flat as clients grow
and latency grows with the queue: the number Sprint 7 attacks.
"""

import argparse
import json
import statistics
import threading
import time
import urllib.request


def one(base, max_tokens, out):
    body = json.dumps({"prompt": "The history of the city", "max_tokens": max_tokens,
                       "temperature": 0.8, "stream": True, "seed": 1}).encode()
    req = urllib.request.Request(base + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0, first, n = time.time(), None, 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if line.startswith(b"data: {"):
                n += 1
                if first is None:
                    first = time.time() - t0
    out.append((first, time.time() - t0, n))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--clients", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()
    print(f"{'clients':>7} {'TTFT p50':>9} {'TTFT max':>9} {'latency p50':>12} {'latency max':>12} {'tokens/sec':>11}")
    for c in args.clients:
        out = []
        threads = [threading.Thread(target=one, args=(args.base, args.max_tokens, out)) for _ in range(c)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.time() - t0
        ttft = [f for f, _, _ in out]
        lat = [l for _, l, _ in out]
        toks = sum(n for _, _, n in out)
        print(f"{c:>7} {statistics.median(ttft):>8.2f}s {max(ttft):>8.2f}s "
              f"{statistics.median(lat):>11.2f}s {max(lat):>11.2f}s {toks / wall:>11.1f}")
