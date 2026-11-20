"""Interactive model tester — test all trained models at once from the
terminal and see the results as a table.

Scans for weights files (weights-*.json in the current directory),
shows them as a numbered list with each model's final loss, lets you
pick which ones to test, then generates text from each and prints a
table (model, final loss, output excerpt) followed by the full outputs.
Optionally saves the table as a Markdown file.

The prompt defaults to the per-corpus prompt (combined -> "To be, or
not to be", WikiText / WikiText-103 -> "Albert Einstein was a"), the
same as compare_models.py, so outputs stay comparable across runs.

Usage:
    python3 benchmarks/test_models.py
    # Non-interactive (same as compare_models.py):
    python3 benchmarks/test_models.py --all --temperature 0.5 --chars 100
    python3 benchmarks/test_models.py --models 1,3,5 --prompt "The city"
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

LOSS_RE = re.compile(r"step\s+(\d+)\s+loss\s+([0-9.]+)")


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
    tokenizer, corpus, merges, and is_wiki. Returns None if the
    filename carries no usable config.
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
        corpus_name = "WikiText-103"
    elif is_wiki:
        corpus = WIKITEXT
        merges = "bpe-merges-wikitext.json"
        corpus_name = "WikiText-2"
    else:
        corpus = COMBINED
        merges = "bpe-merges.json"
        corpus_name = "combined"
    return {
        "d_model": d_model,
        "n_layer": n_layer,
        "n_head": 2,
        "block_size": block_size,
        "tokenizer": tokenizer,
        "corpus": corpus,
        "merges": merges,
        "is_wiki": is_wiki,
        "corpus_name": corpus_name,
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
            f"{len(model.parameters())}"
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


def final_loss(weights_path):
    """Last-20-step average loss for a weights file, read from the
    train log that saved it. Returns None if no matching log found."""
    name = os.path.basename(weights_path)
    for log in glob.glob("train-*.log"):
        try:
            with open(log, errors="replace") as f:
                content = f.read()
        except OSError:
            continue
        if name not in content:
            continue
        losses = [float(m.group(2)) for m in LOSS_RE.finditer(content)]
        if losses:
            last20 = losses[-20:]
            return sum(last20) / len(last20)
    return None


def config_label(cfg):
    return (
        f"d{cfg['d_model']}/n{cfg['n_layer']} "
        f"{cfg['tokenizer']} {cfg['corpus_name']}"
    )


def parse_selection(raw, count):
    """Parse 'all', '', '1,3,5', or '1-4' into a list of 0-based indices."""
    raw = raw.strip().lower()
    if raw in ("", "all"):
        return list(range(count))
    picked = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                lo, hi = int(a), int(b)
            except ValueError:
                continue
            for i in range(lo, hi + 1):
                if 1 <= i <= count:
                    picked.add(i - 1)
        else:
            try:
                i = int(part)
            except ValueError:
                continue
            if 1 <= i <= count:
                picked.add(i - 1)
    return sorted(picked)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive model tester."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="non-interactive: test every model found",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="non-interactive selection: '1,3,5' or '1-4' (1-based)",
    )
    parser.add_argument("--prompt", default=None,
                        help="one prompt for every model")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--chars", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save", default=None,
                        help="also write the table to a Markdown file")
    args = parser.parse_args()

    paths = sorted(glob.glob("weights-*.json"))
    if not paths:
        print("no weights files found in this directory")
        return

    # Build the model list with inferred configs and final losses.
    entries = []
    for path in paths:
        cfg = infer_config(path)
        if cfg is None:
            continue
        entries.append((path, cfg, final_loss(path)))
    if not entries:
        print("no weights files with a usable config in the filename")
        return

    print(f"\nFound {len(entries)} models:\n")
    for i, (path, cfg, loss) in enumerate(entries, 1):
        loss_s = f"{loss:.4f}" if loss is not None else "-"
        print(f"  [{i:>2}] {config_label(cfg):<28} loss {loss_s}")
    print("")

    interactive = not (args.all or args.models)
    if interactive:
        raw = input("Test which models? [all] ")
        picked = parse_selection(raw, len(entries))
        if not picked:
            print("nothing selected, exiting")
            return
        prompt = input(
            "Prompt? [auto: 'To be, or not to be' for combined, "
            "'Albert Einstein was a' for WikiText] "
        ).strip()
        prompt = prompt or None
        temp_raw = input("Temperature? [0.5] ").strip()
        try:
            temperature = float(temp_raw) if temp_raw else 0.5
        except ValueError:
            temperature = 0.5
        chars_raw = input("Tokens to generate? [100] ").strip()
        try:
            chars = int(chars_raw) if chars_raw else 100
        except ValueError:
            chars = 100
        save = input("Save table to a Markdown file? [leave blank for no] ").strip()
        save = save or None
    else:
        picked = (
            parse_selection(args.models, len(entries))
            if args.models
            else list(range(len(entries)))
        )
        prompt = args.prompt
        temperature = args.temperature
        chars = args.chars
        save = args.save

    if not picked:
        print("nothing selected, exiting")
        return

    results = []
    for i in picked:
        path, cfg, loss = entries[i]
        use_prompt = prompt or (
            PROMPT_WIKITEXT if cfg["is_wiki"] else PROMPT_COMBINED
        )
        print(f"\nGenerating: {config_label(cfg)} ...", flush=True)
        try:
            model, encode, decode = load_model(cfg, path)
        except (ValueError, FileNotFoundError, KeyError) as e:
            print(f"  skipped {os.path.basename(path)}: {e}")
            continue
        random.seed(args.seed)
        idx = encode(use_prompt)
        gen = model.generate(idx, chars, temperature=temperature)
        out = decode(gen)
        results.append((path, cfg, loss, use_prompt, out))

    if not results:
        print("no models produced output")
        return

    # Summary table: model, final loss, output excerpt.
    print("\n")
    print(f"{'Model':<28} {'Loss':<8} Output (excerpt)")
    print("-" * 28 + " " + "-" * 8 + " " + "-" * 60)
    for path, cfg, loss, use_prompt, out in results:
        loss_s = f"{loss:.4f}" if loss is not None else "-"
        excerpt = " ".join(out.split())
        if len(excerpt) > 60:
            excerpt = excerpt[:57] + "..."
        print(f"{config_label(cfg):<28} {loss_s:<8} {excerpt}")
    print("")

    # Full outputs.
    for path, cfg, loss, use_prompt, out in results:
        print(f"## {config_label(cfg)}")
        print(f'Prompt: "{use_prompt}"')
        print("")
        print(out)
        print("")
        print("---")
        print("")

    if save:
        lines = []
        lines.append(
            f"# Model test — {len(results)} models, "
            f"{chars} tokens, temp {temperature}, seed {args.seed}"
        )
        lines.append("")
        lines.append("| Model | Loss | Prompt | Output |")
        lines.append("|---|---|---|---|")
        for path, cfg, loss, use_prompt, out in results:
            loss_s = f"{loss:.4f}" if loss is not None else "-"
            excerpt = " ".join(out.split())
            lines.append(
                f"| {config_label(cfg)} | {loss_s} | {use_prompt} | "
                f"{excerpt} |"
            )
        lines.append("")
        for path, cfg, loss, use_prompt, out in results:
            lines.append(f"## {config_label(cfg)}")
            lines.append(f'Prompt: "{use_prompt}"')
            lines.append("")
            lines.append(out)
            lines.append("")
            lines.append("---")
            lines.append("")
        with open(save, "w") as f:
            f.write("\n".join(lines))
        print(f"saved to {save}")


if __name__ == "__main__":
    main()
