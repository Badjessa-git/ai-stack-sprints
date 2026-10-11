"""Part 2 before/after benchmark: frozen v1 vs current implementation.

Times train / encode / decode for both implementations on the sprint
corpus repeated 30x (~140 KB). The Part 1 corpus (4.6 KB) is too small:
at that size the timings are mostly noise.

A sanity check runs first: if the two implementations disagree on merges
or encodings, the script refuses to time anything — speed is meaningless
if the outputs moved.

Run from the sprint folder:
    python3 part2/compare.py
"""

import importlib.util
import statistics
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPRINT = HERE.parent

BIG_REPEAT = 30
TRAIN_VOCAB = 384
ENCODE_VOCAB = 512
DECODE_TOKENS = 30_000  # decode a fixed-size slice so runs stay comparable


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v1 = _load("tokenizer_v1", HERE / "tokenizer_v1.py")
v2 = _load("tokenizer_current", SPRINT / "src" / "tokenizer.py")

CORPUS = (SPRINT / "data" / "corpus.txt").read_text(encoding="utf-8")
BIG_TEXT = CORPUS * BIG_REPEAT


def time_best(fn, runs: int) -> float:
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def sanity_check() -> None:
    t1, t2 = v1.BPETokenizer(), v2.BPETokenizer()
    t1.train(CORPUS, TRAIN_VOCAB)
    t2.train(CORPUS, TRAIN_VOCAB)
    assert list(t1.merges.items()) == list(t2.merges.items()), (
        "MERGES DIFFER — outputs changed, refusing to benchmark."
    )
    assert t1.encode(CORPUS) == t2.encode(CORPUS), (
        "ENCODINGS DIFFER — outputs changed, refusing to benchmark."
    )
    print("sanity check: merges and encodings identical ✓\n")


def bench(module, label: str) -> dict:
    def do_train():
        tok = module.BPETokenizer()
        tok.train(BIG_TEXT, TRAIN_VOCAB)
        return tok

    train_s = time_best(do_train, runs=2)

    tok = module.BPETokenizer()
    tok.train(CORPUS, ENCODE_VOCAB)
    encode_s = time_best(lambda: tok.encode(BIG_TEXT), runs=3)
    ids = tok.encode(BIG_TEXT)[:DECODE_TOKENS]
    decode_s = time_best(lambda: tok.decode(ids), runs=3)

    print(f"{label}: train {train_s:.3f}s | encode {encode_s:.3f}s | "
          f"decode[{DECODE_TOKENS} tokens] {decode_s:.3f}s")
    return {"train": train_s, "encode": encode_s, "decode": decode_s}


def main() -> None:
    print(f"corpus: {len(BIG_TEXT.encode('utf-8')):,} bytes "
          f"(sprint corpus x{BIG_REPEAT})")
    sanity_check()
    r1 = bench(v1, "v1 (frozen) ")
    r2 = bench(v2, "v2 (current)")
    print("\nspeedup (v1 / v2): "
          f"train {r1['train'] / r2['train']:.2f}x | "
          f"encode {r1['encode'] / r2['encode']:.2f}x | "
          f"decode {r1['decode'] / r2['decode']:.2f}x")
    print("\nPaste these numbers into part2/RESULTS.md.")


if __name__ == "__main__":
    main()
