"""Compare every trained model side by side.

Scans for weights files (weights-*.json in the current directory, or
explicit --weights paths), infers each model's config from its
filename, generates text from each one, and prints the results in a
grouped comparison so you can see which model writes best.

The prompt depends on the corpus the model trained on:
  - combined corpus (Shakespeare + TinyStories) -> "To be, or not to be"
  - WikiText -> "Albert Einstein was a"
Pass --prompt to override both with one prompt, or --prompt-combined /
--prompt-wikitext to set them separately.

Config inference (from the filename):
  d256 -> --d-model 256, -n2 -> --n-layer 2, "bpe" -> BPE tokenizer,
  "wikitext" -> WikiText corpus. Block size defaults to 64, heads to 2.
  Files whose inferred architecture does not match the saved weights
  (wrong layer count, etc.) are skipped with a warning instead of
  crashing the whole run.

Usage:
    python3 benchmarks/compare_models.py
    python3 benchmarks/compare_models.py --weights weights-numpy-d256-n2-bpe.json weights-numpy-d512-n2-bpe.json
    python3 benchmarks/compare_models.py --chars 100 --temperature 0.5 --save comparison.md
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
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.numpy.transformer import Transformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMBINED = os.path.join(ROOT, "data", "corpus-combined.txt")
WIKITEXT = os.path.join(ROOT, "data", "wiki.train.tokens")
WIKI103 = os.path.join(ROOT, "data", "wiki103-100m.txt")

PROMPT_COMBINED = "To be, or not to be"
PROMPT_WIKITEXT = "Albert Einstein was a"


def _decode_lenient(tok, token_ids):
    """Decode token ids to text, replacing invalid UTF-8 bytes."""
    try:
        return tok.decode(token_ids)
    except UnicodeDecodeError:
        byte_list = []
        for tid in token_ids:
            byte_list.extend(tok._BPETokenizer__decode_single(tid))
        return bytes(byte_list).decode("utf-8", errors="replace")


def infer_config(path):
    """Infer a model's config from its filename.

    Returns a dict with d_model, n_layer, n_head, block_size,
    tokenizer, corpus, and merges. Returns None if the filename
    carries no usable config.
    """
    name = os.path.basename(path)
    m = re.search(r"d(\d+)", name)
    if not m:
        return None
    d_model = int(m.group(1))
    n = re.search(r"-n(\d+)", name)
    n_layer = int(n.group(1)) if n else 2
    b = re.search(r"-b(\d+)", name)
    block_size = int(b.group(1)) if b else 64
    tokenizer = "bpe" if "bpe" in name else "char"
    is_wiki103 = "wiki103" in name
    is_wiki = is_wiki103 or "wikitext" in name
    if is_wiki103:
        corpus = WIKI103
        merges = "bpe-merges-wiki103.json"
    elif is_wiki:
        corpus = WIKITEXT
        merges = "bpe-merges-wikitext.json"
    else:
        corpus = COMBINED
        merges = "bpe-merges.json"
    return {
        "d_model": d_model,
        "n_layer": n_layer,
        "n_head": 2,
        "block_size": block_size,
        "tokenizer": tokenizer,
        "corpus": corpus,
        "merges": merges,
        "is_wiki": is_wiki,
    }


def load_model(cfg, weights_path):
    """Build the Transformer for cfg and load weights. Returns
    (model, encode_fn, decode_fn) or raises on a shape mismatch."""
    text = open(cfg["corpus"]).read()
    tok = None
    if cfg["tokenizer"] == "bpe":
        bpe_src = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "sprint-01-bpe-tokenizer",
            "src",
        )
        sys.path.insert(0, bpe_src)
        from tokenizer import BPETokenizer

        saved = json.load(open(cfg["merges"]))
        tok = BPETokenizer()
        tok.merges = {(l, r): nid for l, r, nid in saved["merges"]}
        tok._BPETokenizer__byte_lookup_table = {
            nid: (l, r) for l, r, nid in saved["merges"]
        }
        vocab_size = saved["vocab_size"]
        encode = tok.encode
        decode = lambda ids: _decode_lenient(tok, ids)
    else:
        chars = sorted(set(text))
        vocab_size = len(chars)
        stoi = {c: i for i, c in enumerate(chars)}
        itos = {i: c for c, i in stoi.items()}
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda ids: "".join(itos[i] for i in ids)

    model = Transformer(
        vocab_size,
        d_model=cfg["d_model"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        block_size=cfg["block_size"],
    )
    weights = json.load(open(weights_path))
    if len(weights) != len(model.parameters()):
        raise ValueError(
            f"weight count {len(weights)} != model params "
            f"{len(model.parameters())} — inferred architecture "
            f"(d{cfg['d_model']}, n{cfg['n_layer']}) does not match "
            "the file"
        )
    for p, w in zip(model.parameters(), weights):
        arr = np.array(w)
        if np.shape(arr) != np.shape(p.data):
            raise ValueError(
                f"weight shape {np.shape(arr)} != model shape "
                f"{np.shape(p.data)}"
            )
        p.data = arr
    return model, encode, decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        nargs="+",
        default=None,
        help="weights files to compare (default: all weights-*.json "
        "in the current directory)",
    )
    parser.add_argument("--chars", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--prompt",
        default=None,
        help="one prompt for every model (overrides the per-corpus prompts)",
    )
    parser.add_argument("--prompt-combined", default=PROMPT_COMBINED)
    parser.add_argument("--prompt-wikitext", default=PROMPT_WIKITEXT)
    parser.add_argument(
        "--save",
        default=None,
        help="also write the comparison to a Markdown file",
    )
    args = parser.parse_args()

    paths = args.weights or sorted(glob.glob("weights-*.json"))
    if not paths:
        print("no weights files found")
        return

    results = []
    skipped = []
    for path in paths:
        cfg = infer_config(path)
        if cfg is None:
            skipped.append((path, "no d-model in filename"))
            continue
        prompt = args.prompt or (
            args.prompt_wikitext if cfg["is_wiki"] else args.prompt_combined
        )
        try:
            model, encode, decode = load_model(cfg, path)
        except (ValueError, FileNotFoundError, KeyError) as e:
            skipped.append((path, str(e)))
            continue
        random.seed(args.seed)
        idx = encode(prompt)
        gen = model.generate(
            idx, args.chars, temperature=args.temperature
        )
        out = decode(gen)
        results.append((path, cfg, prompt, out))

    lines = []
    lines.append(
        f"# Model comparison — {len(results)} models, "
        f"{args.chars} tokens, temp {args.temperature}, seed {args.seed}"
    )
    lines.append("")
    for path, cfg, prompt, out in results:
        tag = (
            f"d{cfg['d_model']} / n{cfg['n_layer']} / "
            f"{cfg['tokenizer']} / "
            f"{'WikiText' if cfg['is_wiki'] else 'combined'}"
        )
        lines.append(f"## {os.path.basename(path)}")
        lines.append(f"*{tag}*")
        lines.append(f'Prompt: "{prompt}"')
        lines.append("")
        lines.append(out)
        lines.append("")
        lines.append("---")
        lines.append("")

    report = "\n".join(lines)
    print(report)
    if args.save:
        with open(args.save, "w") as f:
            f.write(report)
        print(f"saved to {args.save}")

    if skipped:
        print("\nskipped:")
        for path, reason in skipped:
            print(f"  {os.path.basename(path)}: {reason}")


if __name__ == "__main__":
    main()
