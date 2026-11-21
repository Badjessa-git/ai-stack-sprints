"""train.py v2 — the same training loop with the fixes from the Nov 19
code review. train.py and src/numpy/ are left untouched so old runs stay
reproducible and their logs comparable. This uses src/numpy_v2/.

What changed vs train.py (each is a flag, so you can ablate it):
  1. Init: Linear weights/biases are U(-1,1) * 1/sqrt(n_in) (src/numpy_v2/nn.py).
     Old init gave step-0 loss 26.7 / 41.4 / 59.0 at d256 / d512 / d1024
     instead of ln(vocab) = 6.24. Not a flag — it is the v2 model.
  2. --batch-size B (default 16): B windows per step as one (B, T) batch.
     --batch-size 1 reproduces the old one-window step.
  3. Held-out eval: the last --val-frac of the tokens is never trained on;
     every --eval-every steps prints
        eval step N  val_loss X  val_bpb Y
     val_bpb (bits per byte) is comparable across char / BPE / vocab sizes.
  4. --lr-schedule cosine with --warmup, and --grad-clip (global norm).
     --lr-schedule constant --warmup 0 --grad-clip 0 = old behaviour.
  5. --dtype float32 (default) — about 2x faster BLAS. float64 = old.
  6. --bpe-sample spread (default): BPE merges learned from 200k chars
     spread across the corpus, not text[:200_000]. Token cache gets a
     "-spread" suffix so the old cache is never overwritten. head = old.
  7. Loss via log-softmax (no log(0) in float32); sampler can no longer
     fall off the end of the CDF.
  8. Memory: gc.collect() after every step and eval batch. Each Tensor's
     _backward closure references the Tensor (a cycle), so dead graphs
     were only freed by the cyclic GC, which rarely ran: 8.4 GB peak at
     d256 (the legacy d1024 run sits at ~11 GB). Now 0.6 GB / 2.4 GB at
     d256 / d1024. BPE ids are also memory-mapped from a shared int32
     .npy (written once next to the legacy JSON cache) instead of each
     process parsing the 224 MB JSON.

Measured A/B (d256/n2, char, Shakespeare, same 6,400 windows, Adam 1e-3):
  train.py     held-out 4.29 bits/char, 127s
  train_v2.py  held-out 3.08 bits/char,  30s

Usage (from sprint-03-transformer/):
  python3 benchmarks/train_v2.py --run-name d256-n2-bpe-wiki103 \
      --tokenizer bpe --corpus data/wiki103-100m.txt --optimizer adam \
      --lr 0.001 --d-model 256 --n-layer 2 --block-size 64 --epochs 1

  -> runs/v2/d256-n2-bpe-wiki103/{config.json, train.log, weights.npz, merges.json}
  Then: python3 benchmarks/test_models_v2.py   (prompts every v2 run)
  Resume: same --run-name plus --load runs/v2/<name>/weights.npz
"""

import argparse
import gc
import json
import os
import random
import sys
import time
import warnings

# Silence the Mac/Accelerate RuntimeWarnings (divide by zero / overflow
# in matmul) — a known noise quirk on Apple Silicon, not a code bug
# (the same run on Linux prints none). Real errors still raise.
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Keep the BLAS math library to one thread. The matrices here are tiny,
# so threads buy nothing — but on Apple Silicon, NumPy's Accelerate
# backend can segfault when it spreads small matrix ops across threads
# (seen Nov 16, 2026: the d128 run died with a segfault on an M4 Pro).
# These must be set before NumPy is imported, so they live here at the
# top. setdefault respects a value you already set in the shell.
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

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


def _bpe_sample(text, n_chars=200_000, n_chunks=100, seed=0):
    """Pick the text the BPE merges are learned from: n_chunks evenly
    spaced slices across the whole corpus (n_chars total), instead of
    text[:n_chars]. On WikiText-103 the first 200k chars are only a few
    articles, so the merges fit those topics, not the corpus."""
    if len(text) <= n_chars:
        return text
    chunk = n_chars // n_chunks
    stride = len(text) // n_chunks
    rng = random.Random(seed)
    parts = []
    for i in range(n_chunks):
        lo = i * stride
        start = lo + rng.randint(0, max(0, stride - chunk))
        parts.append(text[start : start + chunk])
    return "".join(parts)


def _lr_at(step, args):
    """Linear warmup for args.warmup steps, then cosine decay to
    args.lr * args.min_lr_frac at the last step ("constant" = flat)."""
    import math

    if args.lr_schedule == "constant":
        return args.lr
    if args.warmup and step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    span = max(1, args.steps - args.warmup)
    progress = min(1.0, (step - args.warmup) / span)
    floor = args.lr * args.min_lr_frac
    return floor + 0.5 * (args.lr - floor) * (1 + math.cos(math.pi * progress))


def _evaluate(model, val_windows, byte_len, batch_size, engine):
    """Held-out loss with dropout off. Returns (nats per token, bits per
    byte). bits/byte = sum(NLL) / ln 2 / sum(bytes of the predicted
    tokens) — comparable across char, BPE-512, BPE-4k, ... because it
    normalises by text, not by token count."""
    import math

    import numpy as np

    total_nll, total_tokens, total_bytes = 0.0, 0, 0
    for i in range(0, len(val_windows), batch_size):
        chunk = val_windows[i : i + batch_size]
        if engine == "numpy":
            nll = model.token_nll(chunk, training=False).data
            total_nll += float(nll.sum())
            gc.collect()  # eval builds graphs too; free them (see train loop)
        else:
            for w in chunk:
                total_nll += float(model.loss(w).data) * (len(w) - 1)
        for w in chunk:
            total_tokens += len(w) - 1
            total_bytes += int(byte_len[np.asarray(w[1:])].sum())
    return total_nll / total_tokens, total_nll / math.log(2) / total_bytes


RUNS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "v2"
)


class _Tee:
    """Write stdout to the terminal and the run's train.log."""

    def __init__(self, path):
        self.file = open(path, "a", buffering=1)
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def _git_commit():
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def load_weights_into(model, path):
    """Load .npz (v2) or .json (legacy nested lists) weights into model,
    casting to the model's dtype. Shapes must match."""
    import numpy as np

    if path.endswith(".npz"):
        with np.load(path) as z:
            weights = [z[f"p{i}"] for i in range(len(z.files))]
    else:
        weights = json.load(open(path))
    params = model.parameters()
    assert len(weights) == len(params), (
        f"weights file has {len(weights)} tensors, model has {len(params)} — "
        "architecture flags must match the run that saved them"
    )
    for p, w in zip(params, weights):
        arr = np.asarray(w, dtype=p.data.dtype)
        assert arr.shape == p.data.shape, (
            f"weight shape {arr.shape} != model shape {p.data.shape}"
        )
        p.data = arr


def _save_weights(model, path):
    """.npz (one array per parameter, in parameters() order, as p0, p1,
    ...) — ~5x smaller than JSON in float32 and much faster to load.
    Any other extension falls back to the legacy JSON format."""
    if path.endswith(".npz"):
        import numpy as np

        tmp = path + ".tmp.npz"
        np.savez(tmp, **{f"p{i}": p.data for i, p in enumerate(model.parameters())})
        os.replace(tmp, path)  # atomic: a crash mid-save never corrupts it
        return
    _save_weights_json(model, path)


def _save_weights_json(model, path):
    """Save parameters as JSON. A NumPy array becomes a nested list; a
    plain number stays a number. generate.py loads it back with the
    matching --engine."""
    data = [
        p.data.tolist() if hasattr(p.data, "tolist") else p.data
        for p in model.parameters()
    ]
    with open(path, "w") as f:
        json.dump(data, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--epochs",
        type=float,
        default=None,
        help="passes over the corpus (overrides --steps). One epoch is "
        "len(corpus) // block_size steps.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="put everything for this run in <runs-root>/<run-name>/: "
        "config.json, train.log, weights.npz, merges.json. Overrides "
        "--save and --merges-out.",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default=RUNS_ROOT,
        help="parent folder for --run-name (default: runs/v2/)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow --run-name to reuse a folder that already has a run",
    )
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="windows per step (NumPy engine runs them as one (B, T) "
        "batch; scalar engine accumulates). 1 = the old behaviour.",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=["cosine", "constant"],
        default="cosine",
        help="cosine: linear warmup then cosine decay; constant: flat lr",
    )
    parser.add_argument("--warmup", type=int, default=500, help="warmup steps")
    parser.add_argument(
        "--min-lr-frac",
        type=float,
        default=0.1,
        help="cosine decays to lr * this at the last step",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="clip the global gradient norm to this (0 = off)",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.01,
        help="fraction of the token stream (the end) held out for eval",
    )
    parser.add_argument(
        "--eval-every", type=int, default=2000, help="eval every N steps (0 = end only)"
    )
    parser.add_argument(
        "--eval-tokens",
        type=int,
        default=32768,
        help="held-out tokens per eval (fixed set, same every time)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="NumPy engine array dtype (float32 ~2x faster)",
    )
    parser.add_argument(
        "--bpe-sample",
        choices=["spread", "head"],
        default="spread",
        help="text the BPE merges learn from: spread = 200k chars sampled "
        "across the corpus; head = the first 200k chars (old behaviour, "
        "reuses the old token cache)",
    )
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "adam"],
        default="sgd",
        help="sgd: plain gradient descent (what we've used so far). "
        "adam: momentum + per-weight adaptive step size — converges "
        "faster and lower for transformers.",
    )
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-layer", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--sample",
        type=int,
        default=40,
        help="chars to generate at the end (each one costs "
        "a full forward pass — no KV cache yet, "
        "that is Sprint 6)",
    )
    parser.add_argument(
        "--engine",
        choices=["numpy"],
        default="numpy",
        help="autograd engine: scalar (one Value per number) or numpy "
        "(arrays — same model, same math, much faster)",
    )
    parser.add_argument(
        "--save", type=str, default=None, help="path to save the trained weights (JSON)"
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="path to weights JSON to start from (resume training). "
        "Architecture flags must match the run that saved them.",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=CORPUS,
        help="training text file (default: data/corpus.txt)",
    )
    parser.add_argument(
        "--tokenizer",
        choices=["char", "bpe"],
        default="char",
        help="char: one token per character (vocab = distinct chars, "
        "the Sprint 3 default). bpe: use the Sprint 1 BPE tokenizer "
        "(vocab 512) — each token is a subword, so block_size tokens "
        "cover ~5x more text and the model learns words, not spelling.",
    )
    parser.add_argument(
        "--bpe-vocab",
        type=int,
        default=512,
        help="BPE vocab size (only with --tokenizer bpe).",
    )
    parser.add_argument(
        "--merges-out",
        type=str,
        default="bpe-merges.json",
        help="where to save the learned BPE merges (only with "
        "--tokenizer bpe); generate.py loads them with --merges.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="save weights every N steps (0 = only at the end, "
        "via --save)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="fraction of activations randomly zeroed during training "
        "(NumPy engine only; 0.0 = no dropout, the default). The "
        "dropout function is a stub until you implement it — with "
        "0.0 it is never called, so training is unaffected.",
    )

    args = parser.parse_args()

    run_dir = None
    if args.run_name:
        import datetime

        run_dir = os.path.join(args.runs_root, args.run_name)
        if os.path.exists(os.path.join(run_dir, "config.json")) and not (
            args.overwrite or args.load
        ):
            sys.exit(
                f"{run_dir} already has a run. Pick another --run-name, or "
                "pass --overwrite (or --load to resume)."
            )
        os.makedirs(run_dir, exist_ok=True)
        args.save = os.path.join(run_dir, "weights.npz")
        args.merges_out = os.path.join(run_dir, "merges.json")
        sys.stdout = _Tee(os.path.join(run_dir, "train.log"))
        started = datetime.datetime.now().isoformat(timespec="seconds")

    if args.engine == "numpy":
        import numpy as np

        from src.numpy_v2 import engine as np_engine

        np_engine.set_dtype(getattr(np, args.dtype))
        from src.numpy_v2.transformer import Transformer
    else:
        from src.scalar.transformer import Transformer

    import numpy as np

    if args.tokenizer == "bpe":
        # Sprint 1 BPE: train merges on a 200k-char sample (enough to
        # learn the common subwords; ~12s), then encode the corpus.
        # The encoded ids are cached next to the corpus file, so repeat
        # runs skip the encode (~45s per 1M chars).
        bpe_src = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "sprint-01-bpe-tokenizer",
            "src",
        )
        sys.path.insert(0, bpe_src)
        from tokenizer import BPETokenizer

        suffix = "" if args.bpe_sample == "head" else "-spread"
        cache_path = (
            os.path.splitext(args.corpus)[0]
            + f"-bpe{args.bpe_vocab}{suffix}.tokens.json"
        )
        # v2 keeps the ids as a compact int32 .npy next to the legacy JSON
        # cache (+ a small .meta.json with the merges) and memory-maps it:
        # every parallel run shares one ~190 MB copy instead of each
        # parsing the 224 MB JSON into ~1.7 GB of Python ints (9 GB peak).
        # The legacy JSON cache is only read, never modified.
        stem = cache_path[: -len(".json")]
        npy_path, meta_path = stem + ".npy", stem + ".meta.json"
        if os.path.exists(npy_path) and os.path.exists(meta_path):
            meta = json.load(open(meta_path))
            merges_list, vocab_size = meta["merges"], meta["vocab_size"]
            data = np.load(npy_path, mmap_mode="r")
            print(f"tokenizer bpe: mapped {len(data)} cached tokens from {npy_path}")
        else:
            if os.path.exists(cache_path):
                cached = json.load(open(cache_path))
                merges_list, vocab_size = cached["merges"], cached["vocab_size"]
                data = np.asarray(cached["ids"], dtype=np.int32)
                del cached
                print(f"tokenizer bpe: converted {cache_path} -> {npy_path}")
            else:
                text = open(args.corpus).read()
                tok = BPETokenizer()
                sample = (
                    text[:200_000] if args.bpe_sample == "head" else _bpe_sample(text)
                )
                tok.train(sample, vocab_size=args.bpe_vocab)
                data = np.asarray(tok.encode(text), dtype=np.int32)
                vocab_size = args.bpe_vocab
                merges_list = [[l, r, nid] for (l, r), nid in tok.merges.items()]
                print(
                    f"tokenizer bpe, vocab {vocab_size}, {len(text)} chars -> "
                    f"{len(data)} tokens ({len(data) / len(text):.2%} token-to-byte), "
                    f"cached to {npy_path}"
                )
                del text
            tmp = npy_path + ".tmp.npy"
            np.save(tmp, data)
            os.replace(tmp, npy_path)
            with open(meta_path + ".tmp", "w") as f:
                json.dump({"vocab_size": vocab_size, "merges": merges_list}, f)
            os.replace(meta_path + ".tmp", meta_path)
            data = np.load(npy_path, mmap_mode="r")
        tok = BPETokenizer()
        tok.merges = {(l, r): nid for l, r, nid in merges_list}
        tok._BPETokenizer__byte_lookup_table = {
            nid: (l, r) for l, r, nid in merges_list
        }
        itos = None  # decode via the tokenizer, not a table
        stoi = None
        merges = [[l, r, nid] for (l, r), nid in tok.merges.items()]
        with open(args.merges_out, "w") as f:
            json.dump({"vocab_size": vocab_size, "merges": merges}, f)
        byte_len = [
            len(tok._BPETokenizer__decode_single(i)) for i in range(vocab_size)
        ]
    else:
        text = open(args.corpus).read()
        chars = sorted(set(text))
        vocab_size = len(chars)
        stoi = {c: i for i, c in enumerate(chars)}
        itos = {i: c for c, i in stoi.items()}
        data = np.asarray([stoi[c] for c in text], dtype=np.int32)
        tok = None
        byte_len = [len(itos[i].encode("utf-8")) for i in range(vocab_size)]

    byte_len = np.asarray(byte_len, dtype=np.int64)

    # Hold out the end of the token stream. Contiguous, so no training
    # window overlaps it. The eval set is a fixed list of windows from it.
    n_val = int(len(data) * args.val_frac)
    if n_val < 2 * (args.block_size + 1):
        n_val = 0
    train_data = data[: len(data) - n_val] if n_val else data
    val_data = data[len(data) - n_val :] if n_val else data[:0]
    W = args.block_size
    val_windows = [
        val_data[i : i + W] for i in range(0, len(val_data) - W + 1, W)
    ][: max(1, args.eval_tokens // W)] if n_val else []
    print(
        f"split: {len(train_data)} train tokens, {len(val_data)} held out, "
        f"{len(val_windows)} eval windows"
    )
    if run_dir:
        config = dict(vars(args))
        config.update(
            {
                "vocab_size": vocab_size,
                "chars": None if itos is None else [itos[i] for i in range(vocab_size)],
                "train_tokens": len(train_data),
                "val_tokens": len(val_data),
                "started": started,
                "git_commit": _git_commit(),
                "script": "benchmarks/train_v2.py",
            }
        )
        # A resume keeps the original config.json and records its own
        # settings next to it (the log is appended, step count restarts).
        cfg_path = os.path.join(run_dir, "config.json")
        if args.load and os.path.exists(cfg_path) and not args.overwrite:
            cfg_path = os.path.join(
                run_dir, f"config-resume-{started.replace(':', '')}.json"
            )
        with open(cfg_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    if args.epochs is not None:
        args.steps = int(
            args.epochs * len(train_data) / (args.block_size * args.batch_size)
        )
        print(f"epochs {args.epochs} -> {args.steps} steps")

    random.seed(args.seed)
    if args.engine == "numpy":
        np.random.seed(args.seed)  # dropout masks use np.random
        model = Transformer(
            vocab_size,
            d_model=args.d_model,
            n_head=args.n_head,
            n_layer=args.n_layer,
            block_size=args.block_size,
            dropout=args.dropout,
        )
    else:
        model = Transformer(
            vocab_size,
            d_model=args.d_model,
            n_head=args.n_head,
            n_layer=args.n_layer,
            block_size=args.block_size,
        )
    if args.load:
        load_weights_into(model, args.load)
        print(f"resumed from {args.load} ({len(model.parameters())} weights)")
    # Adam keeps two running averages per parameter: m (momentum — the
    # smoothed gradient) and v (squared gradients — how big they've
    # been). Same shape as the parameter, all zeros to start.
    adam_state = None
    if args.optimizer == "adam":
        adam_state = []
        for p in model.parameters():
            if hasattr(p.data, "tolist"):  # numpy array
                import numpy as np

                adam_state.append(
                    [np.zeros_like(p.data), np.zeros_like(p.data)]
                )
            else:  # scalar float
                adam_state.append([0.0, 0.0])
    print(
        f"engine {args.engine}, vocab {vocab_size}, params {len(model.parameters())}, "
        f"d_model {args.d_model}, block {args.block_size}, "
        f"steps {args.steps}, lr {args.lr}, batch {args.batch_size}, "
        f"schedule {args.lr_schedule}, warmup {args.warmup}, "
        f"clip {args.grad_clip}, dtype {args.dtype if args.engine == 'numpy' else 'python'}"
    )

    t0 = time.time()
    def run_eval(step):
        if not val_windows:
            return
        vl, bpb = _evaluate(
            model, val_windows, byte_len, max(args.batch_size, 16), args.engine
        )
        print(f"eval step {step}  val_loss {vl:.4f}  val_bpb {bpb:.4f}")

    run_eval(0)
    for step in range(args.steps):
        lr = _lr_at(step, args)
        starts = [
            random.randint(0, len(train_data) - args.block_size - 1)
            for _ in range(args.batch_size)
        ]
        batch = [train_data[s : s + args.block_size] for s in starts]
        if args.engine == "numpy":
            loss = model.loss(batch)
            model.zero_grad()
            loss.backward()
            loss_val = float(loss.data)
        else:
            # Scalar engine has no batch axis: accumulate grads per window.
            acc = [0.0] * len(model.parameters())
            loss_val = 0.0
            for w in batch:
                l = model.loss(w)
                model.zero_grad()
                l.backward()
                loss_val += l.data / len(batch)
                for i, p in enumerate(model.parameters()):
                    acc[i] += p.grad / len(batch)
            for p, g in zip(model.parameters(), acc):
                p.grad = g
            loss = None
        if args.grad_clip > 0:
            sq = 0.0
            for p in model.parameters():
                g = p.grad
                sq += float((g * g).sum()) if hasattr(g, "sum") else g * g
            norm = sq**0.5
            if norm > args.grad_clip:
                c = args.grad_clip / (norm + 1e-6)
                for p in model.parameters():
                    p.grad = p.grad * c
        # Free this step's graph now. Each Tensor's _backward closure
        # references the Tensor itself (a reference cycle), so refcounting
        # never frees the graph; only the cyclic GC can, and it triggers on
        # object *count* — a NumPy graph is few objects but large arrays,
        # so without this, dead graphs pile up to many GB (8+ GB at d256).
        loss = None
        gc.collect()
        if args.optimizer == "adam":
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            t = step + 1
            for state, p in zip(adam_state, model.parameters()):
                m, v = state
                g = p.grad
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * (g ** 2)
                state[0], state[1] = m, v
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p.data -= lr * m_hat / (v_hat ** 0.5 + eps)
        else:
            for p in model.parameters():
                p.data -= lr * p.grad
        if (
            args.save
            and args.checkpoint_every
            and (step + 1) % args.checkpoint_every == 0
        ):
            _save_weights(model, args.save)
            print(f"checkpoint step {step + 1} -> {args.save}")
        if step % 10 == 0 or step == args.steps - 1:
            print(f"step {step:4d}  loss {loss_val:.4f}  ({time.time() - t0:.0f}s)")
        if args.eval_every and (step + 1) % args.eval_every == 0:
            run_eval(step + 1)
    if not (args.eval_every and args.steps % args.eval_every == 0):
        run_eval(args.steps)

    if args.save:
        _save_weights(model, args.save)
        print(f"saved {len(model.parameters())} weights to {args.save}")

    print(
        f"\n{time.time() - t0:.0f}s total, "
        f"{(time.time() - t0) / max(1, args.steps):.2f}s/step"
    )
    print(
        f"\n--- sample (greedy, {args.sample} "
        f"{'tokens' if args.tokenizer == 'bpe' else 'chars'}) ---"
    )
    if args.sample <= 0:
        return
    start = random.randint(0, len(train_data) - args.block_size - 1)
    gen = model.generate(
        train_data[start : start + args.block_size].tolist(), args.sample, greedy=True
    )
    if args.tokenizer == "bpe":
        print(_decode_lenient(tok, gen))
    else:
        print("".join(itos[i] for i in gen))


if __name__ == "__main__":
    main()
