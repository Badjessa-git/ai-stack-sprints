#!/usr/bin/env python3
"""Download OpenAI's GPT-2 small (124M) from Hugging Face into data/gpt2/.

    python3 data/download_gpt2.py            (from the sprint folder)

Four files, ~550 MB, standard library only. They're gitignored.
"""

import os
import sys
import urllib.request

BASE = "https://huggingface.co/openai-community/gpt2/resolve/main/"
FILES = ["config.json", "vocab.json", "merges.txt", "model.safetensors"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpt2")


def main():
    os.makedirs(OUT, exist_ok=True)
    for name in FILES:
        path = os.path.join(OUT, name)
        if os.path.exists(path):
            print(f"have {name}")
            continue

        def progress(blocks, size, total):
            if total > 10_000_000:
                done = min(blocks * size, total)
                sys.stdout.write(f"\r  {name}: {done / 1e6:.0f}/{total / 1e6:.0f} MB")
                sys.stdout.flush()

        print(f"get  {name}")
        urllib.request.urlretrieve(BASE + name, path + ".part", progress)
        os.replace(path + ".part", path)
        print()
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
