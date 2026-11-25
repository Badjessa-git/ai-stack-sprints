"""Prompt one trained run and print the model's full output, nothing else.

test_models_v3.py compares many runs in a table and truncates each output
to fit the terminal. This loads one run (v2 or v3) and prints the whole
generated text. With no --prompt it keeps asking for prompts until you
press Enter on an empty line or Ctrl-D.

Usage (from sprint-03-transformer/):
    python3 benchmarks/prompt.py rx4k                     # pick a run by name, then type prompts
    python3 benchmarks/prompt.py rx4k --prompt "The history of the Roman Empire"
    python3 benchmarks/prompt.py d512-n4-nodrop-rx4k-wiki103 --tokens 200 --temperature 0.8

The run argument is a substring of the run folder name. If it matches more
than one run, the matches are listed and nothing is loaded.
"""

import argparse
import glob
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_models_v3 import RUNS_ROOTS, load, read_run  # noqa: E402


def find_run(key):
    dirs = sorted(
        os.path.dirname(p)
        for root in RUNS_ROOTS
        for p in glob.glob(os.path.join(root, "*", "weights.npz"))
    )
    exact = [d for d in dirs if os.path.basename(d) == key]
    matches = exact or [d for d in dirs if key in os.path.basename(d)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        sys.exit(f"no run with weights.npz matches {key!r}. Runs:\n  " + "\n  ".join(os.path.basename(d) for d in dirs))
    sys.exit(f"{key!r} matches {len(matches)} runs, be more specific:\n  " + "\n  ".join(os.path.basename(d) for d in matches))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run name, or a unique part of it (e.g. rx4k-2ep)")
    ap.add_argument("--prompt", default=None, help="one prompt; omit to type prompts interactively")
    ap.add_argument("--tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=None, help="fix for repeatable output (default: random)")
    args = ap.parse_args()

    run = read_run(find_run(args.run))
    model, encode, decode = load(run)
    regex = run["cfg"].get("tokenizer") == "bpe-regex"
    bpb = f"{run['val_bpb']:.3f}" if run["val_bpb"] is not None else "-"
    print(f"{run['name']}  (val bpb {bpb}, context {run['cfg']['block_size']} tokens)", file=sys.stderr)

    def generate(prompt):
        # Same prompt handling as test_models_v3.py: the regex tokenizer puts
        # the space at the start of the next word, so a trailing space would
        # be a token the model rarely saw.
        if regex:
            prompt = prompt.rstrip(" ")
        if args.seed is not None:
            random.seed(args.seed)
        out = model.generate(encode(prompt), args.tokens, temperature=args.temperature)
        print(decode(out))

    if args.prompt is not None:
        generate(args.prompt)
        return
    while True:
        try:
            prompt = input("\nprompt> ")
        except EOFError:
            print()
            break
        if not prompt.strip():
            break
        generate(prompt)


if __name__ == "__main__":
    main()
