"""Part 2 profiler: where does the time actually go?

Runs cProfile over train / encode / decode for the frozen v1 and for the
current implementation in ../src/tokenizer.py, on the same ~140 KB
corpus compare.py uses. Totals from compare.py tell you *how much*
faster v2 is; this tells you *which function ate the time* — read
'tottime' (time in the function itself) and 'cumtime' (including
everything it called).

Run from the sprint folder:
    python3 part2/profiler.py
"""

import cProfile
import importlib.util
import io
import pstats
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPRINT = HERE.parent

BIG_REPEAT = 30
TRAIN_VOCAB = 384
ENCODE_VOCAB = 512
DECODE_TOKENS = 30_000
TOP_N = 10


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v1 = _load("tokenizer_v1", HERE / "tokenizer_v1.py")
v2 = _load("tokenizer_current", SPRINT / "src" / "tokenizer.py")

CORPUS = (SPRINT / "data" / "corpus.txt").read_text(encoding="utf-8")
BIG_TEXT = CORPUS * BIG_REPEAT


def profile(label: str, fn) -> None:
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    stream = io.StringIO()
    pstats.Stats(pr, stream=stream).sort_stats("tottime").print_stats(TOP_N)
    print(f"\n===== {label} =====")
    print(stream.getvalue())


def main() -> None:
    def train(module):
        tok = module.BPETokenizer()
        tok.train(BIG_TEXT, TRAIN_VOCAB)
        return tok

    profile("TRAIN — v1 (frozen)", lambda: train(v1))
    profile("TRAIN — current", lambda: train(v2))

    tok = v2.BPETokenizer()
    tok.train(CORPUS, ENCODE_VOCAB)
    profile("ENCODE — current", lambda: tok.encode(BIG_TEXT))
    ids = tok.encode(BIG_TEXT)[:DECODE_TOKENS]
    profile(f"DECODE — current [{DECODE_TOKENS} tokens]", lambda: tok.decode(ids))


if __name__ == "__main__":
    main()
