#!/bin/bash
# Score the finished legacy WikiText-103 runs on the SAME held-out split
# train_v2.py uses (last 1% of tokens, 32,768 eval tokens, val_bpb), so
# old and new runs share one metric. No training (--steps 0): it loads the
# legacy JSON weights, evals once, and saves a converted weights.npz into
# runs/v2/legacy-<name>/ so test_models_v2.py includes them.
# Caveat: legacy runs trained on 100% of the tokens, so they saw part of
# this held-out text — their numbers are, if anything, slightly flattering.
#   bash score_legacy_v2.sh        (sequential, ~1 min each)
set -e
cd "$(dirname "$0")"
L=runs/legacy
score() {
  local name="$1" weights="$2"; shift 2
  python3 benchmarks/train_v2.py --run-name "legacy-$name" --load "$L/$weights" \
    --tokenizer bpe --bpe-sample head --corpus data/wiki103-100m.txt \
    --steps 0 --sample 0 --overwrite "$@" | grep "eval step"
}
score d256-n2-nodrop-wiki103 weights-numpy-d256-n2-bpe-wiki103-nodrop.json --d-model 256 --n-layer 2 --block-size 64
score d256-n2-drop01-wiki103 weights-numpy-d256-n2-bpe-wiki103.json        --d-model 256 --n-layer 2 --block-size 64
score d256-n4-drop01-wiki103 weights-numpy-d256-n4-bpe-wiki103.json        --d-model 256 --n-layer 4 --block-size 64
score d512-n2-nodrop-wiki103 weights-numpy-d512-n2-bpe-wiki103-nodrop.json --d-model 512 --n-layer 2 --block-size 64
score d512-n2-drop01-wiki103 weights-numpy-d512-n2-bpe-wiki103.json        --d-model 512 --n-layer 2 --block-size 64
score d512-n4-drop01-wiki103 weights-numpy-d512-n4-bpe-wiki103.json        --d-model 512 --n-layer 4 --block-size 64
score d512-n2-lr0005-wiki103 weights-numpy-d512-n2-bpe-wiki103-lr0005.json --d-model 512 --n-layer 2 --block-size 64
score d512-n2-blk128-wiki103 weights-numpy-d512-n2-bpe-wiki103-b128.json   --d-model 512 --n-layer 2 --block-size 128
score d1024-n2-drop01-wiki103 weights-numpy-d1024-n2-bpe-wiki103.json      --d-model 1024 --n-layer 2 --block-size 64
