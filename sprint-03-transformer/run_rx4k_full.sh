#!/bin/bash
# The best v3 model (d512-n4 rx4k) for 1 epoch over ALL of WikiText-103
# (153M tokens, 5.4x the 100M subset) instead of the first 100M chars.
# Every other flag matches run_grid_v3.sh.
#
# Question it answers: at about the same compute, is more new text worth more
# than more passes? Compare with d512-n4-nodrop-rx4k-2ep-wiki103 (2 passes over
# 28M tokens, 1.372 bpb) and d512-n4-nodrop-rx4k-wiki103 (1 pass, 1.439).
#
# Comparable by construction: benchmarks/build_wiki103_full.py builds the token
# cache with the same 4k merges and puts the 100M runs' held-out tokens at the
# end; --val-frac below holds out exactly those 283,402 tokens. Step-0 val_bpb
# is 3.4472, the same as every earlier rx4k run.
#
#   python3 benchmarks/build_wiki103_full.py   # once (~30 s, ~15 GB RAM peak)
#   bash run_rx4k_full.sh                      # ~149k steps, ~15 h alone
#   bash watch_v3.sh
set -e
cd "$(dirname "$0")"
export VECLIB_MAXIMUM_THREADS=1
[ -f data/wiki103-full-bpe4096-regex-c0cde5b4.tokens.npy ] || {
  echo "run python3 benchmarks/build_wiki103_full.py first"; exit 1; }
name=d512-n4-nodrop-rx4k-full-wiki103
mkdir -p "runs/v3/$name"
nohup python3 -u benchmarks/train_v3.py --run-name "$name" \
  --corpus data/wiki103-full.txt --val-frac 0.0018514287040269533 \
  --optimizer adam --lr 0.001 --epochs 1 --batch-size 16 \
  --block-size 64 --dropout 0 --checkpoint-every 2000 --eval-every 2000 \
  --sample 0 --d-model 512 --n-layer 4 \
  --tokenizer bpe-regex --bpe-vocab 4096 > /dev/null 2> "runs/v3/$name/stderr.log" &
echo "Launched $name (pid $!). Watch with: bash watch_v3.sh"
