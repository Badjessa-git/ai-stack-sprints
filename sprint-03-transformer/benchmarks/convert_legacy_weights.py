"""Convert legacy JSON weights (runs/legacy/weights-numpy-*.json) to .npz.

The JSON files are text: ~20 bytes per weight. The .npz (float32, the
format train_v2/v3 save) is 4 bytes per weight, ~5x smaller. Same layout
as train_v2._save_weights: one array per parameter, in parameters()
order, named p0, p1, ... so load_weights_into() reads either file.

The JSON originals are kept on disk (gitignored); only the .npz is
committed. Models already converted by score_legacy_v2.sh (the
WikiText-103 runs, in runs/v2/legacy-*/weights.npz) are skipped.

    python3 benchmarks/convert_legacy_weights.py           # convert + verify
"""

import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY = os.path.join(ROOT, "runs", "legacy")

# Converted already, by score_legacy_v2.sh -> runs/v2/legacy-<name>/weights.npz
ALREADY = {
    "weights-numpy-d256-n2-bpe-wiki103-nodrop.json": "legacy-d256-n2-nodrop-wiki103",
    "weights-numpy-d256-n2-bpe-wiki103.json": "legacy-d256-n2-drop01-wiki103",
    "weights-numpy-d256-n4-bpe-wiki103.json": "legacy-d256-n4-drop01-wiki103",
    "weights-numpy-d512-n2-bpe-wiki103-nodrop.json": "legacy-d512-n2-nodrop-wiki103",
    "weights-numpy-d512-n2-bpe-wiki103.json": "legacy-d512-n2-drop01-wiki103",
    "weights-numpy-d512-n4-bpe-wiki103.json": "legacy-d512-n4-drop01-wiki103",
    "weights-numpy-d512-n2-bpe-wiki103-lr0005.json": "legacy-d512-n2-lr0005-wiki103",
    "weights-numpy-d512-n2-bpe-wiki103-b128.json": "legacy-d512-n2-blk128-wiki103",
    "weights-numpy-d1024-n2-bpe-wiki103.json": "legacy-d1024-n2-drop01-wiki103",
}


def main():
    paths = sorted(glob.glob(os.path.join(LEGACY, "weights-numpy-*.json")))
    worst = 0.0
    for path in paths:
        name = os.path.basename(path)
        if name in ALREADY:
            print(f"skip    {name}  (-> runs/v2/{ALREADY[name]}/weights.npz)")
            continue
        out = path[: -len(".json")] + ".npz"
        t = time.time()
        weights = json.load(open(path))
        arrays = [np.asarray(w, dtype=np.float32) for w in weights]
        tmp = out + ".tmp.npz"
        np.savez(tmp, **{f"p{i}": a for i, a in enumerate(arrays)})
        os.replace(tmp, out)
        # Verify: same tensors, same shapes, float32 rounding only.
        with np.load(out) as z:
            assert len(z.files) == len(weights)
            err = max(
                float(np.max(np.abs(z[f"p{i}"] - np.asarray(w, dtype=np.float64))))
                if np.size(z[f"p{i}"]) else 0.0
                for i, w in enumerate(weights)
            )
        worst = max(worst, err)
        print(
            f"convert {name}  {os.path.getsize(path) / 1e6:6.0f} MB -> "
            f"{os.path.getsize(out) / 1e6:5.0f} MB  max |diff| {err:.1e}  "
            f"({time.time() - t:.0f}s)",
            flush=True,
        )
        del weights, arrays
    print(f"done. worst float32 rounding error: {worst:.1e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
