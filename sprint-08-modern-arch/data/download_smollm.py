#!/usr/bin/env python3
"""Download SmolLM-135M (Hugging Face, Apache 2.0) into data/smollm-135m/.

    python3 data/download_smollm.py          (from the sprint folder)

Three files, ~540 MB, standard library only. They are gitignored.
"""

import os
import sys
import urllib.request

BASE = "https://huggingface.co/HuggingFaceTB/SmolLM-135M/resolve/main/"
FILES = ["config.json", "tokenizer.json", "model.safetensors"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smollm-135m")


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
