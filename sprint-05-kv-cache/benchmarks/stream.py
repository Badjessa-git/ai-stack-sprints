"""Stream sampled text from GPT-2 to the terminal, token by token.

    python3 benchmarks/stream.py "Once upon a time" --temperature 0.8 --top-p 0.9
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.deps import gpt2_dir, sprint  # noqa: E402
from src.sampling import generate  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    s4, d = sprint("sprint-04-gpt2"), gpt2_dir()
    tok = s4.gpt2_tokenizer.GPT2Tokenizer.from_files(os.path.join(d, "vocab.json"), os.path.join(d, "merges.txt"))
    model = s4.gpt2.GPT2.from_dir(d)
    print(args.prompt, end="", flush=True)
    pending = b""  # a token can be half of a UTF-8 character: hold bytes until they decode
    for t in generate(model, tok.encode(args.prompt), args.tokens, args.temperature,
                      args.top_k, args.top_p, args.seed, stop_ids={tok.eot_id}):
        pending += tok.token_bytes(t)
        try:
            print(pending.decode("utf-8"), end="", flush=True)
            pending = b""
        except UnicodeDecodeError:
            pass
    print()
