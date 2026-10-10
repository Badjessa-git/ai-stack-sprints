"""Benchmark harness — scaffolded for you. Run after tests pass:

    python3 benchmarks/benchmark.py

Reports training time, compression on data/corpus.txt, and encode/decode
throughput. Save this output; your post quotes it.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer import BPETokenizer

CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus.txt"


def main() -> None:
    text = CORPUS.read_text(encoding="utf-8")
    corpus_bytes = len(text.encode("utf-8"))
    print(f"corpus: {len(text):,} chars / {corpus_bytes:,} bytes")

    for vocab_size in (256, 384, 512):
        tok = BPETokenizer()
        t0 = time.perf_counter()
        tok.train(text, vocab_size=vocab_size)
        train_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        encoded = tok.encode(text)
        encode_s = time.perf_counter() - t0

        assert tok.decode(encoded) == text, "ROUNDTRIP FAILED"

        ratio = len(encoded) / corpus_bytes
        print(
            f"vocab={vocab_size:>4}  merges={len(tok.merges):>3}  "
            f"tokens={len(encoded):>6,}  ratio={ratio:.2%}  "
            f"bytes/token={1 / ratio:.2f}  "
            f"train={train_s:.2f}s  encode={encode_s:.3f}s"
        )
    print("roundtrip: OK")


if __name__ == "__main__":
    main()
