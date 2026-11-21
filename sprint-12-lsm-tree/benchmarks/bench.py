"""Write fast, read... complicated. The LSM trade-offs, measured.

    python3 benchmarks/bench.py              # 50k writes, ~10-30 s
    python3 benchmarks/bench.py --n 200000

1. writes/sec, with and without fsync on every write
2. read latency for keys that exist and keys that don't, with Bloom
   filters on and off (off = pretend every filter says "maybe")
3. write amplification: bytes written to SSTables / bytes the user put
"""

import argparse
import os
import random
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.lsm import BloomFilter, LSMStore  # noqa: E402


def key(i):
    return f"user:{i:08d}".encode()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--value-bytes", type=int, default=100)
    args = ap.parse_args()
    rng = random.Random(0)
    val = os.urandom(args.value_bytes)
    order = list(range(args.n))
    rng.shuffle(order)
    d = tempfile.mkdtemp()
    try:
        s = LSMStore(os.path.join(d, "fast"), memtable_bytes=256 * 1024, compact_at=8)
        t = time.time()
        for i in order:
            s.put(key(i), val)
        w = time.time() - t
        put_bytes = args.n * (len(key(0)) + args.value_bytes)
        print(f"{args.n:,} random writes, no fsync:   {args.n / w:>9,.0f} writes/sec  "
              f"({s.stats['flushes']} flushes, {s.stats['compactions']} compactions, "
              f"{len(s.tables)} tables now)")
        s2 = LSMStore(os.path.join(d, "sync"), sync=True)
        t = time.time()
        m = min(2000, args.n)
        for i in order[:m]:
            s2.put(key(i), val)
        print(f"{m:,} random writes, fsync each: {m / (time.time() - t):>9,.0f} writes/sec")
        s2.close()
        print(f"write amplification: {s.stats['bytes_written'] / put_bytes:.2f}x "
              f"(SSTable bytes written / bytes put)\n")

        hits = [key(rng.randrange(args.n)) for _ in range(5000)]
        misses = [key(args.n + rng.randrange(args.n)) for _ in range(5000)]
        real = BloomFilter.might_contain
        for label, bloom in (("Bloom filters on", True), ("Bloom filters off", False)):
            if not bloom:
                BloomFilter.might_contain = lambda self, k: True
            for kind, keys in (("existing", hits), ("missing", misses)):
                s.stats.update(sstable_reads=0)
                t = time.time()
                for k in keys:
                    s.get(k)
                us = (time.time() - t) / len(keys) * 1e6
                print(f"{label:<18} {kind:<9} keys: {us:7.1f} µs/read, "
                      f"{s.stats['sstable_reads'] / len(keys):.2f} tables searched/read")
            BloomFilter.might_contain = real
        s.close()
    finally:
        shutil.rmtree(d)
