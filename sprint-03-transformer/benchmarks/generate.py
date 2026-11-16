import argparse
import json
import os
import random
import sys
import warnings

# Silence the Mac/Accelerate RuntimeWarnings (divide by zero / overflow
# in matmul) — a known noise quirk on Apple Silicon, not a code bug
# (the same run on Linux prints none). Real errors still raise.
warnings.filterwarnings("ignore", category=RuntimeWarning)

# See benchmarks/train.py: keep BLAS to one thread. On Apple Silicon,
# NumPy's Accelerate backend can segfault on small threaded matrix ops.
# Must be set before NumPy is imported.
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CORPUS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "corpus.txt"
)


def _decode_lenient(tok, token_ids):
    """Decode token ids to text, replacing invalid UTF-8 bytes.

    The tokenizer's decode() is strict (exact roundtrip for encode()
    output), but model-generated ids can produce byte sequences that
    are not valid UTF-8. For display, replace those bytes.
    """
    try:
        return tok.decode(token_ids)
    except UnicodeDecodeError:
        byte_list = []
        for tid in token_ids:
            byte_list.extend(tok._BPETokenizer__decode_single(tid))
        return bytes(byte_list).decode("utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", required=True)
    parser.add_argument("--prompt", default="To be")
    parser.add_argument("--chars", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="always pick the most likely next char (same as train.py's "
        "--sample). At this loss, sampling wanders; greedy reads coherent.",
    )
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-layer", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--engine",
        choices=["scalar", "numpy"],
        default="scalar",
        help="must match the engine the weights were trained with",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=CORPUS,
        help="training text file (must match the training run's corpus)",
    )
    parser.add_argument(
        "--tokenizer",
        choices=["char", "bpe"],
        default="char",
        help="must match the training run: char = one token per "
        "character; bpe = Sprint 1 BPE tokens (needs --merges).",
    )
    parser.add_argument(
        "--merges",
        type=str,
        default="bpe-merges.json",
        help="BPE merges file saved by train.py (only with "
        "--tokenizer bpe).",
    )
    args = parser.parse_args()

    if args.engine == "numpy":
        from src.numpy.transformer import Transformer
    else:
        from src.scalar.transformer import Transformer

    text = open(args.corpus).read()
    tok = None
    if args.tokenizer == "bpe":
        # Rebuild the Sprint 1 tokenizer from the saved merges. The
        # merges dict is (left, right) -> new_id; the private lookup
        # table is its inverse, rebuilt here so decode() works.
        bpe_src = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "sprint-01-bpe-tokenizer",
            "src",
        )
        sys.path.insert(0, bpe_src)
        from tokenizer import BPETokenizer

        saved = json.load(open(args.merges))
        tok = BPETokenizer()
        tok.merges = {(l, r): nid for l, r, nid in saved["merges"]}
        tok._BPETokenizer__byte_lookup_table = {
            nid: (l, r) for l, r, nid in saved["merges"]
        }
        vocab_size = saved["vocab_size"]
    else:
        chars = sorted(set(text))
        vocab_size = len(chars)
        stoi = {c: i for i, c in enumerate(chars)}
        itos = {i: c for c, i in stoi.items()}

    random.seed(args.seed)
    model = Transformer(
        vocab_size,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        block_size=args.block_size,
    )
    weights = json.load(open(args.load))
    assert len(weights) == len(model.parameters())
    for p, w in zip(model.parameters(), weights):
        # NumPy weights were saved as nested lists; rebuild the array.
        arr = np.array(w) if args.engine == "numpy" else w
        assert np.shape(arr) == np.shape(p.data), (
            f"weight shape {np.shape(arr)} != model shape "
            f"{np.shape(p.data)} — architecture flags must match "
            "the run that saved them"
        )
        p.data = arr

    idx = tok.encode(args.prompt) if args.tokenizer == "bpe" else [stoi[c] for c in args.prompt]
    gen = model.generate(
        idx, args.chars, temperature=args.temperature, greedy=args.greedy
    )
    if args.tokenizer == "bpe":
        print(_decode_lenient(tok, gen))
    else:
        print("".join(itos[i] for i in gen))


if __name__ == "__main__":
    main()
