"""Attention heatmaps for a trained transformer.

Runs a prompt through the model and draws one heatmap per (layer,
head): rows are the positions doing the attending (the query), columns
are the positions being attended to (the key). A bright cell means
that token pays a lot of attention to that token. The upper triangle
is always dark — causal masking means a position can never look ahead.

Usage:
    python3 benchmarks/attn_viz.py --load weights-numpy-d256-n2-bpe.json \
        --d-model 256 --n-layer 2 --block-size 64 \
        --corpus data/corpus-combined.txt --prompt "To be, or not to be"

With BPE (--tokenizer bpe --merges bpe-merges.json), the axis labels
are subwords; with the default char tokenizer, they are characters.
Only the NumPy engine is supported (the scalar engine has no
last_weights recording).
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.numpy.transformer import Transformer


def load_tokenizer(args):
    """Return (encode, decode_labels) for the chosen tokenizer.

    encode: text -> list of token ids. decode_labels: token ids ->
    a label per token for the heatmap axes.
    """
    if args.tokenizer == "bpe":
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

        def labels(ids):
            return [tok.decode([i]) for i in ids]

        return tok.encode, labels

    text = open(args.corpus).read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}

    def labels(ids):
        itos = {i: c for c, i in stoi.items()}
        return [itos[i] for i in ids]

    return (lambda s: [stoi[c] for c in s]), labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", required=True)
    parser.add_argument("--prompt", default="To be, or not to be")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument(
        "--tokenizer", choices=["char", "bpe"], default="char",
    )
    parser.add_argument("--merges", type=str, default="bpe-merges.json")
    parser.add_argument(
        "--corpus",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "corpus.txt",
        ),
    )
    args = parser.parse_args()

    encode, labels = load_tokenizer(args)
    idx = encode(args.prompt)
    if len(idx) > args.block_size:
        idx = idx[-args.block_size:]
    toks = labels(idx)

    # Vocab size: BPE uses the saved vocab; char uses the corpus chars.
    if args.tokenizer == "bpe":
        vocab_size = json.load(open(args.merges))["vocab_size"]
    else:
        vocab_size = len(set(open(args.corpus).read()))

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
        arr = np.array(w)
        assert np.shape(arr) == np.shape(p.data), (
            f"weight shape {np.shape(arr)} != model shape {np.shape(p.data)}"
        )
        p.data = arr

    # One forward pass populates each head's last_weights.
    model(idx, training=False)

    n_layer = len(model.blocks)
    n_head = model.blocks[0].attn.n_head
    fig, axes = plt.subplots(
        n_layer, n_head, figsize=(4 * n_head + 2, 4 * n_layer + 1),
        squeeze=False,
    )
    fig.suptitle(f"attention: {args.prompt!r}", fontsize=12)
    for li, block in enumerate(model.blocks):
        for hi, head in enumerate(block.attn.heads):
            ax = axes[li][hi]
            w = head.last_weights
            im = ax.imshow(w, cmap="Blues", vmin=0, vmax=1)
            ax.set_title(f"layer {li} head {hi}")
            ax.set_xticks(range(len(toks)))
            ax.set_xticklabels(toks, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(toks)))
            ax.set_yticklabels(toks, fontsize=8)
            ax.set_xlabel("attended to (key)")
            ax.set_ylabel("attending (query)")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="attention weight")
    plt.show()


if __name__ == "__main__":
    main()
