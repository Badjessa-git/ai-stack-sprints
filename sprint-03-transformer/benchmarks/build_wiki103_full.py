"""Build the full WikiText-103 training corpus for v3 bpe-regex runs, so a run
on it is directly comparable with the runs on data/wiki103-100m.txt.

A plain `train_v3.py --corpus data/wiki103-full.txt` would not be comparable:
  - it would learn a new set of merges (a different tokenizer), and
  - its held-out set would be the last 1% of the full corpus: different text,
    and the 100M runs' held-out text would land in the training data.

So this writes the token cache train_v3.py looks for, built from the SAME
merges as the 100M cache, in this order:

    [100M subset, training part] [rest of WikiText-103 train] [100M subset, held-out part]

and prints the --val-frac that makes train_v3.py hold out exactly that last
part. The eval windows are then token-for-token the ones every earlier rx run
was scored on, so val_bpb compares directly. train_v3.py is not modified: it
finds the cache and memory-maps it instead of tokenizing.

It also writes data/wiki103-full.txt (the same text, in the same order) so
the corpus file exists for anything that reads it.

Usage (from sprint-03-transformer/, needs the two parquet files and pyarrow):
    python3 benchmarks/build_wiki103_full.py            # 4096 vocab (default)
    python3 benchmarks/build_wiki103_full.py --vocab 512
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
PARQUETS = [os.path.join(DATA, f"train-0000{i}-of-00002.parquet") for i in range(2)]
SUBSET = os.path.join(DATA, "wiki103-100m.txt")
OUT_TXT = os.path.join(DATA, "wiki103-full.txt")
TARGET_CHARS = 100_000_000  # what download_wiki103.py cut the subset at
VAL_FRAC = 0.01  # what the 100M runs used

sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "sprint-01-bpe-tokenizer", "part3"))
import regex_tokenizer as rt  # noqa: E402


def read_lines():
    """Every non-empty line of the train split, in order, exactly as
    download_wiki103.py writes them (text + newline)."""
    for path in PARQUETS:
        for text in pq.read_table(path, columns=["text"]).column("text").to_pylist():
            if text:
                yield text + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=4096)
    args = ap.parse_args()

    # The 100M cache this run must match.
    pattern = os.path.join(DATA, f"wiki103-100m-bpe{args.vocab}-regex-*.tokens.npy")
    found = glob.glob(pattern)
    if len(found) != 1:
        sys.exit(f"expected exactly one cache matching {pattern}, found {found}")
    old_npy = found[0]
    tag = old_npy.split("-regex-")[1].split(".")[0]
    old_meta = old_npy.replace(".tokens.npy", ".meta.json")
    meta = json.load(open(old_meta))
    import hashlib

    if hashlib.sha1(rt.SPLIT_PATTERN.encode()).hexdigest()[:8] != tag:
        sys.exit("SPLIT_PATTERN changed since the 100M cache was built; the tokens would not match")
    new_stem = os.path.join(DATA, f"wiki103-full-bpe{args.vocab}-regex-{tag}")
    new_npy, new_meta = new_stem + ".tokens.npy", new_stem + ".meta.json"
    if os.path.exists(new_npy):
        sys.exit(f"{new_npy} already exists; delete it to rebuild")

    old = np.load(old_npy, mmap_mode="r")
    n_val = int(len(old) * VAL_FRAC)
    print(f"100M cache: {len(old):,} tokens, held-out tail {n_val:,} tokens")

    # Split the train split into the 100M subset and the rest, with the same
    # stopping rule as download_wiki103.py, and check the subset is identical.
    t = time.time()
    subset, rest, total = [], [], 0
    for line in read_lines():
        if total < TARGET_CHARS:
            subset.append(line)
            total += len(line)
        else:
            rest.append(line)
    subset_text, rest_text = "".join(subset), "".join(rest)
    del subset, rest
    with open(SUBSET, encoding="utf-8") as f:
        if f.read() != subset_text:
            sys.exit(f"{SUBSET} doesn't match the first {TARGET_CHARS:,} chars of the parquet files")
    print(f"read parquet: subset {len(subset_text):,} chars (matches {os.path.basename(SUBSET)}), "
          f"rest {len(rest_text):,} chars, {time.time() - t:.0f}s")

    tok = rt.RegexBPETokenizer()
    tok.set_merges(meta["merges"])

    # Sanity check: the tokenizer reproduces the 100M cache.
    head = np.frombuffer(tok.encode_to_array(subset_text[:1_000_000]), dtype=np.int32)
    if not np.array_equal(head[:-50], old[: len(head) - 50]):
        sys.exit("re-encoding the subset doesn't reproduce the cached tokens; stopping")

    t = time.time()
    rest_ids = np.frombuffer(tok.encode_to_array(rest_text), dtype=np.int32)
    print(f"encoded rest: {len(rest_ids):,} tokens "
          f"({len(rest_text.encode()) / len(rest_ids):.2f} bytes/token), {time.time() - t:.0f}s")

    data = np.concatenate([old[: len(old) - n_val], rest_ids, old[len(old) - n_val :]]).astype(np.int32)
    # train_v3.py holds out int(len * val_frac) tokens; pick val_frac so that is
    # exactly n_val (the +0.5 keeps float rounding from landing one short).
    val_frac = (n_val + 0.5) / len(data)
    assert int(len(data) * val_frac) == n_val
    assert np.array_equal(data[len(data) - n_val :], old[len(old) - n_val :])

    # The text file, in the same order: decode the two 100M parts so the cut
    # falls exactly where the tokens split.
    token_bytes = tok.token_bytes
    def decode(ids):
        return b"".join(token_bytes[i] for i in ids)
    t = time.time()
    with open(OUT_TXT + ".tmp", "wb") as f:
        f.write(decode(old[: len(old) - n_val]))
        f.write(rest_text.encode())
        f.write(decode(old[len(old) - n_val :]))
    os.replace(OUT_TXT + ".tmp", OUT_TXT)
    print(f"wrote {OUT_TXT} ({os.path.getsize(OUT_TXT):,} bytes), {time.time() - t:.0f}s")

    np.save(new_npy + ".tmp.npy", data)
    os.replace(new_npy + ".tmp.npy", new_npy)
    json.dump({**meta, "built_by": "benchmarks/build_wiki103_full.py",
               "order": "100M train part, rest of train split, 100M held-out part",
               "val_tokens": n_val, "val_frac": val_frac}, open(new_meta, "w"))
    print(f"wrote {new_npy}: {len(data):,} tokens")
    print(f"\nTrain with:  --corpus data/wiki103-full.txt --val-frac {val_frac!r}")


if __name__ == "__main__":
    main()
