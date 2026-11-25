#!/bin/bash
# The best v3 model (d512-n4 rx4k, 1.439 bpb after 1 epoch) trained for 2
# epochs instead of 1. Every other flag matches run_grid_v3.sh exactly, so the
# only difference is the number of passes over the data (and the cosine
# schedule stretched over twice the steps).
#
# Questions it answers (see POST-notes-v1-v4.md, "Still to check"):
#   - was the 1-epoch model undertrained? (its val bpb was still falling)
#   - do the "kerkerkerker" repetition loops shrink with more training?
#
#   bash run_rx4k_2ep.sh
#   bash watch_v3.sh                 # it shows up next to the other v3 runs
#   python3 benchmarks/test_models_v3.py --runs rx4k
set -e
cd "$(dirname "$0")"
export VECLIB_MAXIMUM_THREADS=1
name=d512-n4-nodrop-rx4k-2ep-wiki103
mkdir -p "runs/v3/$name"
nohup python3 -u benchmarks/train_v3.py --run-name "$name" \
  --corpus data/wiki103-100m.txt --optimizer adam --lr 0.001 --epochs 2 --batch-size 16 \
  --block-size 64 --dropout 0 --checkpoint-every 2000 --eval-every 2000 \
  --sample 0 --d-model 512 --n-layer 4 \
  --tokenizer bpe-regex --bpe-vocab 4096 > /dev/null 2> "runs/v3/$name/stderr.log" &
echo "Launched $name (pid $!). Watch with: bash watch_v3.sh"
