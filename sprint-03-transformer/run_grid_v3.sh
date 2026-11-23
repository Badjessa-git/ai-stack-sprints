#!/bin/bash
# v3 grid: does the Sprint 1 Part 3 tokenizer (regex pre-split BPE) help?
#
# Every run uses identical training settings to the v2 grid (train_v3.py =
# train_v2.py + --tokenizer bpe-regex): lr 1e-3, batch 16, block 64, cosine,
# warmup 500, clip 1.0, float32, 1 epoch, dropout 0. Only the tokenizer and
# model size change. Compare on val_bpb (bits per byte), never val_loss.
#
# --lr 0.001 is passed explicitly: train_v3's default is 0.05 (the scalar
# SGD default). The first launch on 2026-11-21 forgot it, and every run
# stalled near 5.2 nats/token; those logs are kept in runs/v3-bad-lr005/.
#
#   tokenizer \ model        d256-n2                    d512-n4
#   Part 1 BPE @512          v2 run (1.778, reused)     bpe512  + bpe512-s2 (seed 2)
#   Part 3 regex @512        rx512                      rx512
#   Part 3 regex @4096       rx4k                       rx4k
#
#   bpe512 vs rx512     -> effect of the regex split alone (same vocab)
#   rx512  vs rx4k      -> effect of the bigger vocab
#   bpe512 vs bpe512-s2 -> run-to-run noise: differences smaller than this
#                          gap aren't real
#
# Step 1 (foreground, visible): build the two regex token caches once, so the
# parallel runs don't each learn the merges. ~3 min at 512, ~21 min at 4096.
# Step 2 (background): launch the 6 runs. Single-process estimates: d256
# ~50 min, d512-n4 ~3-4 h; with 6 at once expect roughly 1.5-2x that.
#
#   bash run_grid_v3.sh
#   bash watch_v3.sh                      # progress + latest val_bpb
#   python3 benchmarks/test_models_v3.py  # v2 + v3 runs in one table
set -e
cd "$(dirname "$0")"
export VECLIB_MAXIMUM_THREADS=1
CORPUS=data/wiki103-100m.txt
PREP=$(mktemp -d)

echo "== step 1: token caches =="
for V in 512 4096; do
  echo "-- bpe-regex @$V"
  python3 -u benchmarks/train_v3.py --tokenizer bpe-regex --bpe-vocab "$V" \
    --corpus "$CORPUS" --steps 0 --d-model 64 --n-layer 1 --block-size 64 \
    --sample 0 --merges-out "$PREP/merges-$V.json" | grep -v "^step"
done
rm -rf "$PREP"

echo
echo "== step 2: launching runs =="
run() {
  local name="$1"; shift
  mkdir -p "runs/v3/$name"
  echo "Launching $name ..."
  nohup python3 -u benchmarks/train_v3.py --run-name "$name" \
    --corpus "$CORPUS" --optimizer adam --lr 0.001 --epochs 1 --batch-size 16 \
    --block-size 64 --dropout 0 --checkpoint-every 2000 --eval-every 2000 \
    --sample 0 "$@" > /dev/null 2> "runs/v3/$name/stderr.log" &
}
BPE512="--tokenizer bpe --bpe-vocab 512 --bpe-sample head"   # same cache as v2
RX512="--tokenizer bpe-regex --bpe-vocab 512"
RX4K="--tokenizer bpe-regex --bpe-vocab 4096"

run d512-n4-nodrop-bpe512-wiki103    --d-model 512 --n-layer 4 $BPE512
run d512-n4-nodrop-bpe512-s2-wiki103 --d-model 512 --n-layer 4 $BPE512 --seed 2
run d512-n4-nodrop-rx512-wiki103     --d-model 512 --n-layer 4 $RX512
run d512-n4-nodrop-rx4k-wiki103      --d-model 512 --n-layer 4 $RX4K
run d256-n2-nodrop-rx512-wiki103     --d-model 256 --n-layer 2 $RX512
run d256-n2-nodrop-rx4k-wiki103      --d-model 256 --n-layer 2 $RX4K

echo
echo "All 6 launched. Watch with: bash watch_v3.sh"
