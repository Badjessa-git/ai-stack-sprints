#!/usr/bin/env python3
"""Convert WikiText-103 Parquet files (downloaded from the browser)
into data/wiki103-100m.txt (first ~100M chars of the train split).

1. Download these two files into sprint-03-transformer/data/:
   https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/train-00000-of-00002.parquet
   https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/train-00001-of-00002.parquet
2. pip install pyarrow
3. Run from sprint-03-transformer/:
   python3 convert_wiki103_parquet.py
"""

import pyarrow.parquet as pq

TARGET_CHARS = 100_000_000
OUT_PATH = "data/wiki103-100m.txt"
FILES = [
    "data/train-00000-of-00002.parquet",
    "data/train-00001-of-00002.parquet",
]


def main():
    total = 0
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for path in FILES:
            print(f"Reading {path} ...", flush=True)
            table = pq.read_table(path, columns=["text"])
            for text in table.column("text").to_pylist():
                if not text:
                    continue
                out.write(text + "\n")
                total += len(text) + 1
                if total >= TARGET_CHARS:
                    break
            if total >= TARGET_CHARS:
                break
    print(f"Saved {total:,} chars to {OUT_PATH}")
    with open(OUT_PATH, "r", encoding="utf-8") as f:
        print("First 300 chars:")
        print(f.read(300))


if __name__ == "__main__":
    main()
