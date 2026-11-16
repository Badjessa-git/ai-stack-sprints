import argparse
import json
import os
import random
import sys

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", required=True)
    parser.add_argument("--prompt", default="To be")
    parser.add_argument("--chars", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
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
    args = parser.parse_args()

    if args.engine == "numpy":
        from src.numpy.transformer import Transformer
    else:
        from src.scalar.transformer import Transformer

    text = open(args.corpus).read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    random.seed(args.seed)
    model = Transformer(
        len(chars),
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        block_size=args.block_size,
    )
    weights = json.load(open(args.load))
    assert len(weights) == len(model.parameters())
    for p, w in zip(model.parameters(), weights):
        # NumPy weights were saved as nested lists; rebuild the array.
        p.data = np.array(w) if args.engine == "numpy" else w

    idx = [stoi[c] for c in args.prompt]
    gen = model.generate(idx, args.chars, temperature=args.temperature)
    print("".join(itos[i] for i in gen))


if __name__ == "__main__":
    main()
