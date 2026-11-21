"""Prompt every v2 and v3 run and show held-out bits/byte next to its output.

Same as test_models_v2.py, plus v3 runs (--tokenizer bpe-regex). By default
it scans runs/v2/ and runs/v3/ together, so tokenizers compare in one
table on val bpb.

The v2 counterpart of test_models.py. Instead of guessing a model's
config from its filename, it reads runs/v2/<run>/config.json (written
by train_v2.py --run-name), so every flag is known exactly. The metric
is the last held-out val_bpb from the run's train.log, which is
comparable across char / BPE / vocab sizes — unlike train loss.

Usage (from sprint-03-transformer/):
    python3 benchmarks/test_models_v3.py                      # every run
    python3 benchmarks/test_models_v3.py --runs d256,d512     # name filter
    python3 benchmarks/test_models_v3.py --prompt "Once upon a time "
    python3 benchmarks/test_models_v3.py --save reports/2026-11-22-v3.md
"""

import argparse
import glob
import json
import os
import random
import re
import sys
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "sprint-01-bpe-tokenizer", "src"))

import numpy as np  # noqa: E402

from src.numpy_v2 import engine  # noqa: E402

RUNS_ROOTS = [os.path.join(ROOT, "runs", "v2"), os.path.join(ROOT, "runs", "v3")]
VAL_RE = re.compile(r"eval step\s+(\d+)\s+val_loss\s+([0-9.]+)\s+val_bpb\s+([0-9.]+)")
STEP_RE = re.compile(r"^step\s+(\d+)\s+loss", re.M)

PROMPT_WIKI = "Albert Einstein was a "
PROMPT_OTHER = "To be, or not to be "


def read_run(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    log_path = os.path.join(run_dir, "train.log")
    log = open(log_path, errors="replace").read() if os.path.exists(log_path) else ""
    evals = VAL_RE.findall(log)
    steps = STEP_RE.findall(log)
    return {
        "name": os.path.basename(run_dir),
        "dir": run_dir,
        "cfg": cfg,
        "val_bpb": float(evals[-1][2]) if evals else None,
        "val_loss": float(evals[-1][1]) if evals else None,
        "step": int(steps[-1]) + 1 if steps else 0,
        # config.json is written before --epochs is turned into a step count,
        # so recompute it the same way train_v2 does.
        "steps": (
            int(cfg["epochs"] * cfg["train_tokens"] / (cfg["block_size"] * cfg["batch_size"]))
            if cfg.get("epochs") and cfg.get("train_tokens")
            else cfg.get("steps")
        ),
    }


def load(run):
    cfg = run["cfg"]
    engine.set_dtype(getattr(np, cfg.get("dtype", "float32")))
    from src.numpy_v2.transformer import Transformer

    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    from train_v3 import load_weights_into

    model = Transformer(
        cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        block_size=cfg["block_size"],
        dropout=0.0,
    )
    load_weights_into(model, os.path.join(run["dir"], "weights.npz"))

    if cfg["tokenizer"] == "bpe-regex":
        sys.path.insert(
            0, os.path.join(os.path.dirname(ROOT), "sprint-01-bpe-tokenizer", "part3")
        )
        from regex_tokenizer import RegexBPETokenizer

        rtok = RegexBPETokenizer.load(os.path.join(run["dir"], "merges.json"))

        def encode(text):
            return rtok.encode(text)

        def decode(ids):
            return rtok.decode(ids, errors="replace")
    elif cfg["tokenizer"] == "bpe":
        from tokenizer import BPETokenizer

        saved = json.load(open(os.path.join(run["dir"], "merges.json")))
        tok = BPETokenizer()
        tok.merges = {(l, r): nid for l, r, nid in saved["merges"]}
        tok._BPETokenizer__byte_lookup_table = {
            nid: (l, r) for l, r, nid in saved["merges"]
        }

        def encode(text):
            return tok.encode(text)

        def decode(ids):
            out = []
            for t in ids:
                out.extend(tok._BPETokenizer__decode_single(t))
            return bytes(out).decode("utf-8", errors="replace")
    else:
        chars = cfg["chars"]
        stoi = {c: i for i, c in enumerate(chars)}

        def encode(text):
            return [stoi[c] for c in text if c in stoi]

        def decode(ids):
            return "".join(chars[i] for i in ids)

    return model, encode, decode


def n_params(cfg):
    d, n, V, T = cfg["d_model"], cfg["n_layer"], cfg["vocab_size"], cfg["block_size"]
    return n * (12 * d * d + 13 * d) + 2 * V * d + V + T * d + 2 * d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", action="append", default=None, help="folder of runs (repeatable; default: runs/v2 and runs/v3)")
    ap.add_argument("--runs", default=None, help="comma-separated substrings of run names")
    ap.add_argument("--prompt", default=None, help="one prompt for every run")
    ap.add_argument(
        "--keep-trailing-space",
        action="store_true",
        help="don't strip the prompt's trailing space for bpe-regex runs",
    )
    ap.add_argument("--tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save", default=None, help="also write a Markdown report")
    args = ap.parse_args()

    roots = args.runs_root or RUNS_ROOTS
    dirs = sorted(
        os.path.dirname(p)
        for root in roots
        for p in glob.glob(os.path.join(root, "*", "config.json"))
    )
    if args.runs:
        keys = [k.strip() for k in args.runs.split(",") if k.strip()]
        dirs = [d for d in dirs if any(k in os.path.basename(d) for k in keys)]
    runs = [read_run(d) for d in dirs if os.path.exists(os.path.join(d, "weights.npz"))]
    if not runs:
        print(f"no runs with weights.npz under {roots}")
        return

    results = []
    for run in runs:
        model, encode, decode = load(run)
        corpus = run["cfg"].get("corpus", "")
        prompt = args.prompt or (PROMPT_WIKI if "wiki" in corpus else PROMPT_OTHER)
        # Regex-split tokenizers attach a space to the START of the next word
        # (" a", " was"), so a trailing space becomes a lone " " token, which
        # in training only appears before another space or a newline. Strip
        # it so the model sees a prompt shaped like its training text. Part 1
        # BPE runs keep the trailing space, as in the v2 reports.
        if run["cfg"].get("tokenizer") == "bpe-regex" and not args.keep_trailing_space:
            prompt = prompt.rstrip(" ")
        ids = encode(prompt)
        random.seed(args.seed)
        out_ids = model.generate(ids, args.tokens, temperature=args.temperature)
        results.append((run, prompt, decode(out_ids)))

    def bpb(r):
        return f"{r['val_bpb']:.3f}" if r["val_bpb"] is not None else "-"

    def progress(r):
        return f"{r['step']:,}/{r['steps']:,}" if r["steps"] else f"{r['step']:,}"

    print(f"\n{'Run':<34} {'Params':>8} {'Steps':>17} {'val bpb':>8}  Output")
    print("-" * 34 + " " + "-" * 8 + " " + "-" * 17 + " " + "-" * 8 + "  " + "-" * 50)
    for run, prompt, out in results:
        ex = " ".join(out.split())
        ex = ex[:47] + "..." if len(ex) > 50 else ex
        print(f"{run['name']:<34} {n_params(run['cfg']) / 1e6:>7.1f}M {progress(run):>17} {bpb(run):>8}  {ex}")
    print()

    if args.save:
        lines = [
            f"# model test — {len(results)} runs, {args.tokens} tokens, "
            f"temp {args.temperature}, seed {args.seed}",
            "",
            "| Run | Params | Steps | val bpb | Prompt | Output |",
            "|---|---|---|---|---|---|",
        ]
        for run, prompt, out in results:
            cell = " ".join(out.split()).replace("|", "\\|")
            lines.append(
                f"| {run['name']} | {n_params(run['cfg']) / 1e6:.1f}M | {progress(run)} | "
                f"{bpb(run)} | {prompt} | {cell} |"
            )
        lines.append("")
        for run, prompt, out in results:
            lines += [f"## {run['name']}", f'Prompt: "{prompt}"', "", out, "", "---", ""]
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        with open(args.save, "w") as f:
            f.write("\n".join(lines))
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
