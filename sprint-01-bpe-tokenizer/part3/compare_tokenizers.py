"""Part 1 vs Part 3 tokenizers, side by side (scaffold — run it, don't edit it).

For each tokenizer it prints: training time, bytes per token on held-out
text (higher = shorter sequences), how many merges start / end with a
space, and how it splits the Sprint 3 prompt with and without a trailing
space.

From sprint-01-bpe-tokenizer/:
    python3 part3/compare_tokenizers.py
    python3 part3/compare_tokenizers.py --vocabs 512,4096 --train-chars 20000000

Part 1 is trained the way Sprint 3 trained it (first 200k chars) and its
encode is slow (one pass per merge), so it's measured on --eval-chars only.
Part 3 learns from chunk counts over --train-chars of the corpus.
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SPRINT1 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(SPRINT1, "src"))

from regex_tokenizer import RegexBPETokenizer  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402

DEFAULT_CORPUS = os.path.join(
    os.path.dirname(SPRINT1), "sprint-03-transformer", "data", "wiki103-100m.txt"
)
PROMPT = "Albert Einstein was a"


def spaced(merges_bytes):
    starts = sum(1 for b in merges_bytes if b.startswith(b" "))
    ends = sum(1 for b in merges_bytes if b.endswith(b" "))
    return starts, ends


def show(pieces):
    return " | ".join(repr(p)[1:-1] for p in pieces)


def part1_bytes(tok, tid):
    return bytes(tok._BPETokenizer__decode_single(tid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--vocabs", default="512,4096", help="Part 3 vocab sizes")
    ap.add_argument("--train-chars", type=int, default=10_000_000)
    ap.add_argument("--eval-chars", type=int, default=500_000)
    args = ap.parse_args()

    def log(msg):
        print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)

    T0 = time.time()
    log(f"reading {args.corpus}")
    text = open(args.corpus).read()
    eval_text = text[-args.eval_chars :]  # the end, like Sprint 3's held-out split
    eval_bytes = len(eval_text.encode())
    rows = []

    # Part 1, as Sprint 3 used it.
    log("Part 1 @512: training on the first 200k chars (~12s)")
    t = time.time()
    p1 = BPETokenizer()
    p1.train(text[:200_000], vocab_size=512)
    train_s = time.time() - t
    log(f"Part 1 @512: trained in {train_s:.0f}s; encoding {len(eval_text):,} eval chars (slow: one pass per merge, ~20s)")
    ids = p1.encode(eval_text)
    log("Part 1 @512: done")
    mb = [part1_bytes(p1, nid) for nid in p1.merges.values()]
    rows.append(
        dict(
            name="Part 1 @512 (200k chars)",
            train_s=train_s,
            bpt=eval_bytes / len(ids),
            spaced=spaced(mb),
            a=[part1_bytes(p1, i).decode("utf-8", "replace") for i in p1.encode(PROMPT)],
            b=[part1_bytes(p1, i).decode("utf-8", "replace") for i in p1.encode(PROMPT + " ")],
        )
    )

    # Part 3 at each vocab size.
    train_text = text[: min(args.train_chars, len(text) - args.eval_chars)]
    for v in [int(x) for x in args.vocabs.split(",")]:
        log(f"Part 3 @{v}: training on {len(train_text):,} chars")
        tok = RegexBPETokenizer()
        t = time.time()
        tok.train(train_text, vocab_size=v)
        train_s = time.time() - t
        log(f"Part 3 @{v}: trained in {train_s:.0f}s; encoding eval text")
        ids = tok.encode(eval_text)
        log(f"Part 3 @{v}: done")
        table = tok.token_bytes
        rows.append(
            dict(
                name=f"Part 3 @{v} ({len(train_text) / 1e6:.0f}M chars)",
                train_s=train_s,
                bpt=eval_bytes / len(ids),
                spaced=spaced(table[256:]),
                a=[table[i].decode("utf-8", "replace") for i in tok.encode(PROMPT)],
                b=[table[i].decode("utf-8", "replace") for i in tok.encode(PROMPT + " ")],
            )
        )

    print(f"\ncorpus {args.corpus}  ({len(text) / 1e6:.0f}M chars)\n")
    print(f"{'Tokenizer':<30} {'train':>8} {'bytes/token':>12} {'merges start/end with space':>30}")
    for r in rows:
        s, e = r["spaced"]
        print(f"{r['name']:<30} {r['train_s']:>7.0f}s {r['bpt']:>12.2f} {f'{s} / {e}':>30}")
    print(f'\nHow each splits "{PROMPT}" and "{PROMPT} " (| between tokens):')
    for r in rows:
        print(f"\n  {r['name']}")
        print(f"    no space:   {show(r['a'])}")
        print(f"    with space: {show(r['b'])}")
    print()


if __name__ == "__main__":
    main()
