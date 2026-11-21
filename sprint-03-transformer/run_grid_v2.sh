#!/bin/bash
# The run_grid.sh grid again, on train_v2.py (init fix, batch 16, float32,
# warmup + cosine, grad clip, held-out val_bpb). Same corpus, SAME tokenizer
# (--bpe-sample head reuses the legacy BPE-512 token cache), same lr 1e-3,
# same sizes/dropout — so the difference vs the legacy grid is the training
# fixes alone. Plus one extra: d512-n2 at lr 3e-3 (batching makes it safe).
#
# 1 epoch = 47,363 steps of 16 x 64 tokens (same tokens as the old 758k).
# Rough single-process time: d256 ~1.3h, d256-n4 ~2.3h, d512 ~2.7h,
# d512-n4 ~5.5h, d1024 ~11h (slower when all run at once).
# Memory: ~0.6 GB (d256) to ~2.4 GB (d1024) per run, ~12 GB for all 10.
#   bash run_grid_v2.sh
# Each run writes runs/v2/<name>/{config.json,train.log,weights.npz};
# crashes go to runs/v2/<name>/stderr.log. Watch progress with:
#   bash watch_v2.sh
set -e
cd "$(dirname "$0")"
CORPUS=data/wiki103-100m.txt
if [ ! -f "${CORPUS%.txt}-bpe512.tokens.json" ]; then
  echo "ERROR: token cache missing; run the legacy grid's tokenizer step first"; exit 1
fi
run() {
  local name="$1"; shift
  mkdir -p "runs/v2/$name"
  echo "Launching $name ..."
  nohup python3 -u benchmarks/train_v2.py --run-name "$name" \
    --tokenizer bpe --bpe-sample head --corpus "$CORPUS" --optimizer adam \
    --epochs 1 --batch-size 16 --checkpoint-every 2000 --eval-every 2000 \
    --sample 0 "$@" > /dev/null 2> "runs/v2/$name/stderr.log" &
}
run d256-n2-nodrop-wiki103 --lr 0.001  --d-model 256  --n-layer 2 --block-size 64  --dropout 0.0
run d256-n2-drop01-wiki103 --lr 0.001  --d-model 256  --n-layer 2 --block-size 64  --dropout 0.1
run d256-n4-drop01-wiki103 --lr 0.001  --d-model 256  --n-layer 4 --block-size 64  --dropout 0.1
run d512-n2-nodrop-wiki103 --lr 0.001  --d-model 512  --n-layer 2 --block-size 64  --dropout 0.0
run d512-n2-drop01-wiki103 --lr 0.001  --d-model 512  --n-layer 2 --block-size 64  --dropout 0.1
run d512-n4-drop01-wiki103 --lr 0.001  --d-model 512  --n-layer 4 --block-size 64  --dropout 0.1
run d512-n2-lr0005-wiki103 --lr 0.0005 --d-model 512  --n-layer 2 --block-size 64  --dropout 0.1
run d512-n2-blk128-wiki103 --lr 0.001  --d-model 512  --n-layer 2 --block-size 128 --dropout 0.1
run d512-n2-lr003-wiki103  --lr 0.003  --d-model 512  --n-layer 2 --block-size 64  --dropout 0.1
run d1024-n2-drop01-wiki103 --lr 0.001 --d-model 1024 --n-layer 2 --block-size 64  --dropout 0.1
echo "10 runs launched. Watch: bash watch_v2.sh"
