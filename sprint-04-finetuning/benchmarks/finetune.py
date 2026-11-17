"""Fine-tune the Sprint 3 transformer for play classification (Track 1).

Usage (from sprint-04-finetuning/):
    make finetune
    python3 benchmarks/finetune.py --steps 500 --lr 0.001
    python3 benchmarks/finetune.py --steps 2000 --lr 0.001 --save classifier-d256.json

What this does:
- Loads the pretrained d256/n2 BPE transformer (Sprint 3's best
  combined-corpus model) and wraps it in the Classifier from
  src/classifier.py (the classification head you build in the
  exercises).
- Trains it on (quote, label) pairs from data/play_quotes_train.jsonl:
  each step samples one random quote, computes the classification
  loss, and updates the WHOLE model (head + transformer) with Adam.
- Reports training loss and validation accuracy as it goes. Accuracy
  is the number that matters here, not loss — it is the fraction of
  held-out quotes the model attributes to the right play. Chance is
  0.25 (4 plays).

The training loop is the standard one from Sprint 3, with one change:
the loss is over play labels, not next tokens. The per-step logic
lives in train_step() below — that is one of your exercises. The rest
(loading, evaluation, saving) is scaffolded.

The pretrained weights live in sprint-03-transformer/ and are loaded
with --load. The default is the d256/n2 BPE model (Sprint 3's best).
The quotes are encoded with the same BPE tokenizer (from the merges
file), so the token ids match what the model was trained on.
"""

import argparse
import importlib.util
import json
import os
import random
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SPRINT3 = os.path.join(
    os.path.dirname(ROOT), "sprint-03-transformer"
)
sys.path.insert(0, SPRINT3)

from src.numpy.transformer import Transformer  # noqa: E402


def _load_classifier_module():
    """Load src/classifier.py by file path (see tests/test_classifier.py
    for why: this folder's src/ and Sprint 3's src/ would collide as
    packages, so the classifier is loaded by path, not by import)."""
    path = os.path.join(ROOT, "src", "classifier.py")
    spec = importlib.util.spec_from_file_location("classifier", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_bpe_tokenizer(merges_path):
    """Rebuild the Sprint 1 BPE tokenizer from a saved merges file."""
    bpe_src = os.path.join(
        os.path.dirname(SPRINT3), "sprint-01-bpe-tokenizer", "src"
    )
    sys.path.insert(0, bpe_src)
    from tokenizer import BPETokenizer

    saved = json.load(open(merges_path))
    tok = BPETokenizer()
    tok.merges = {(l, r): nid for l, r, nid in saved["merges"]}
    tok._BPETokenizer__byte_lookup_table = {
        nid: (l, r) for l, r, nid in saved["merges"]
    }
    return tok, saved["vocab_size"]


def _load_quotes(split):
    """Load one split of the play-quotes dataset."""
    path = os.path.join(ROOT, "data", f"play_quotes_{split}.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f]


def _save_weights(classifier, path):
    """Save the classifier's parameters as JSON (transformer + head).

    The format matches train.py's _save_weights: a flat list of
    tensors, in parameters() order. To resume or to evaluate later,
    load these back onto a Classifier built the same way.
    """
    data = [
        p.data.tolist() if hasattr(p.data, "tolist") else p.data
        for p in classifier.parameters()
    ]
    with open(path, "w") as f:
        json.dump(data, f)


def train_step(clf, idx, label, adam_state, lr, t):
    """One fine-tuning step: loss -> zero grads -> backward -> Adam update.

    clf: the Classifier. idx: token ids of one quote. label: the true
    play index. adam_state: list of [m, v] pairs, one per parameter
    (zeros to start, mutated in place). lr: learning rate. t: 1-based
    step count (for Adam's bias correction).

    Returns the loss value (a plain float) for logging.

    This is one of your exercises. The pattern is the same as
    benchmarks/train.py in Sprint 3, with two differences:
    - The loss comes from clf.loss(idx, label) (classification loss
      over plays), not model.loss(idx) (next-token loss).
    - The parameters come from clf.parameters() (transformer + head),
      not model.parameters(). If you use the model's parameters, the
      head never learns.
    """
    # TODO (exercise): implement the step.
    #   1. loss = clf.loss(idx, label)
    #   2. Zero the gradients (clf.zero_grad(), inherited from Module).
    #   3. loss.backward()
    #   4. Adam update over clf.parameters() with adam_state, lr, t
    #      (beta1=0.9, beta2=0.999, eps=1e-8 — same as train.py).
    #   5. Return float(loss.data).
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--load",
        default=os.path.join(SPRINT3, "weights-numpy-d256-n2-bpe.json"),
        help="pretrained transformer weights (Sprint 3). The "
        "architecture flags must match the run that saved them.",
    )
    parser.add_argument(
        "--merges",
        default=os.path.join(SPRINT3, "bpe-merges.json"),
        help="BPE merges file (Sprint 3). Must match the tokenizer "
        "the pretrained model was trained with.",
    )
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=100,
        help="evaluate validation accuracy every N steps",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="path to save the fine-tuned weights (JSON)",
    )
    args = parser.parse_args()

    classifier_mod = _load_classifier_module()
    Classifier = classifier_mod.Classifier

    # Load the quotes and encode them with the BPE tokenizer. Quotes
    # are truncated to block_size tokens (the model never sees more
    # than that at once). Encoding is done once, up front — the quotes
    # are short, so this is fast.
    tok, vocab_size = _load_bpe_tokenizer(args.merges)
    labels = json.load(open(os.path.join(ROOT, "data", "labels.json")))
    n_plays = len(labels)

    def encode_rows(rows):
        encoded = []
        for row in rows:
            idx = tok.encode(row["text"])[: args.block_size]
            encoded.append((idx, row["label"]))
        return encoded

    train_rows = encode_rows(_load_quotes("train"))
    val_rows = encode_rows(_load_quotes("val"))
    print(
        f"train {len(train_rows)} quotes, val {len(val_rows)} quotes, "
        f"{n_plays} plays, vocab {vocab_size}"
    )

    # Build the pretrained transformer and load its Sprint 3 weights,
    # then wrap it in the Classifier. The classifier's head starts
    # random; the transformer starts from the pretrained weights.
    random.seed(args.seed)
    model = Transformer(
        vocab_size,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        block_size=args.block_size,
    )
    weights = json.load(open(args.load))
    assert len(weights) == len(model.parameters()), (
        f"weights file has {len(weights)} tensors, model has "
        f"{len(model.parameters())} — architecture flags must match "
        "the run that saved them"
    )
    for p, w in zip(model.parameters(), weights):
        arr = np.array(w)
        assert np.shape(arr) == np.shape(p.data), (
            f"weight shape {np.shape(arr)} != model shape "
            f"{np.shape(p.data)}"
        )
        p.data = arr
    print(f"loaded pretrained weights from {args.load}")

    clf = Classifier(model, n_plays)
    params = clf.parameters()
    adam_state = [
        [np.zeros_like(p.data), np.zeros_like(p.data)] for p in params
    ]
    print(f"classifier parameters: {len(params)} tensors")

    rng = random.Random(args.seed)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx, label = rng.choice(train_rows)
        loss_val = train_step(clf, idx, label, adam_state, args.lr, step)
        if step % 10 == 0 or step == args.steps:
            print(
                f"step {step:4d}  loss {loss_val:.4f}  "
                f"({time.time() - t0:.0f}s)"
            )
        if step % args.eval_every == 0 or step == args.steps:
            acc = clf.accuracy(val_rows)
            print(f"  val accuracy {acc:.3f} (chance 0.25)")

    if args.save:
        _save_weights(clf, args.save)
        print(f"saved {len(params)} weights to {args.save}")

    print(f"\n{time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
