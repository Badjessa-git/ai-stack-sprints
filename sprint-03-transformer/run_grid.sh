#!/bin/bash
# Batch grid: config permutations on WikiText-103 50M (BPE, Adam, 1 epoch).
# Run from sprint-03-transformer/ once data/wiki103-50m.txt is on the branch:
#   bash run_grid.sh
# Each run launches in the background with nohup. Check with:
#   ps aux | grep benchmarks/train.py
# Longest run is the d1024 (~39 hrs); everything else finishes sooner.
set -e
cd "$(dirname "$0")"

CORPUS=data/wiki103-50m.txt
MERGES=bpe-merges-wiki103.json

if [ ! -f "$CORPUS" ]; then
  echo "ERROR: $CORPUS not found. Pull the branch first (git pull) so the WikiText-103 data is present."
  exit 1
fi

run() {
  local name="$1"; shift
  echo "Launching $name ..."
  nohup python3 -u benchmarks/train.py --engine numpy --tokenizer bpe --epochs 1 --optimizer adam --corpus "$CORPUS" --merges-out "$MERGES" --checkpoint-every 5000 --sample 0 "$@" > "train-$name.log" 2>&1 &
}

# d256 (each ~2 hrs for 1 epoch)
run d256-n2-wiki103-nodrop --lr 0.001 --d-model 256 --n-layer 2 --block-size 64 --dropout 0.0 --save weights-numpy-d256-n2-bpe-wiki103-nodrop.json
run d256-n2-wiki103-drop01 --lr 0.001 --d-model 256 --n-layer 2 --block-size 64 --dropout 0.1 --save weights-numpy-d256-n2-bpe-wiki103.json
run d256-n4-wiki103-drop01 --lr 0.001 --d-model 256 --n-layer 4 --block-size 64 --dropout 0.1 --save weights-numpy-d256-n4-bpe-wiki103.json

# d512 (each ~9.5 hrs for 1 epoch; block-128 run ~8 hrs)
run d512-n2-wiki103-nodrop --lr 0.001 --d-model 512 --n-layer 2 --block-size 64 --dropout 0.0 --save weights-numpy-d512-n2-bpe-wiki103-nodrop.json
run d512-n2-wiki103-drop01 --lr 0.001 --d-model 512 --n-layer 2 --block-size 64 --dropout 0.1 --save weights-numpy-d512-n2-bpe-wiki103.json
run d512-n4-wiki103-drop01 --lr 0.001 --d-model 512 --n-layer 4 --block-size 64 --dropout 0.1 --save weights-numpy-d512-n4-bpe-wiki103.json
run d512-n2-wiki103-lr0005 --lr 0.0005 --d-model 512 --n-layer 2 --block-size 64 --dropout 0.1 --save weights-numpy-d512-n2-bpe-wiki103-lr0005.json
run d512-n2-wiki103-b128 --lr 0.001 --d-model 512 --n-layer 2 --block-size 128 --dropout 0.1 --save weights-numpy-d512-n2-bpe-wiki103-b128.json

# d1024 (~39 hrs for 1 epoch)
run d1024-n2-wiki103-drop01 --lr 0.001 --d-model 1024 --n-layer 2 --block-size 64 --dropout 0.1 --save weights-numpy-d1024-n2-bpe-wiki103.json

echo "All 9 runs launched. Check with: ps aux | grep benchmarks/train.py"
echo "When they finish, compare with: python3 benchmarks/compare_models.py --save comparison-wiki103.md"
