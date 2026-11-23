# Sprint 09: Quantization and Speed

## Why this sprint

Every model you've run stores each weight as a 32-bit float: 4 bytes,
about 500 MB for GPT-2 small. Running models on laptops and phones
(llama.cpp, Ollama, MLX) depends on storing them in 8 or 4 bits instead.
Quantization keeps a small integer per weight and one float **scale**
per group of weights:

    w  ≈  q × scale       q in [-127, 127] (int8) or [-7, 7] (int4)

The worst error is half a step, and a step is set by the group's largest
weight. So quantization is mostly about choosing groups: one outlier in a
group makes every step in it coarse. You'll see that directly in a test.

The second half is measurement. Quantization on NumPy makes the model
*smaller* and *slower*. That's not a bug in your code. NumPy can't
multiply int8 matrices, so every read dequantizes a whole matrix back to
float32. The speedups people quote come from kernels that compute on the
integers directly. Seeing both sides with numbers is the point.

Learning goals:

1. Explain symmetric quantization: scale, rounding, the half-step bound,
   and why per-channel and per-group beat per-tensor.
2. Pack two 4-bit values into a byte and back, exactly.
3. Run your Sprint 4 GPT-2 on quantized weights *without changing it*,
   and measure size, quality (bits per byte) and speed.
4. Profile a decode step and say where the time goes, and what would
   actually make it faster.

## The plan

Needs Sprint 4 (and Sprint 5 for the benchmarks).

| Step | What | Tests |
|------|------|-------|
| 1 | `quantize_int8`, `dequantize_int8` (one scale per output column) | `TestInt8` |
| 2 | `pack_int4`, `unpack_int4` | `TestPack` |
| 3 | `quantize_int4`, `dequantize_int4` (one scale per 32 weights) | `TestInt4` |
| 4 | `QuantizedParams`: a dict that dequantizes on read | `TestQuantizedParams` |
| 5 | real GPT-2: quality and size | `tests/test_real_gpt2.py` |
| 6 | `make quality`, `make profile`, `make profile-int8`; then explain the numbers | |

Step 4 is the same trick as Sprint 7's `SequenceView`. Your Sprint 4
model only ever does `p[name]`, so a dict subclass whose `__getitem__`
dequantizes lets the unchanged model run on quantized weights.

## Run it

```bash
make test           # 18 tests (3 need Sprint 4's GPT-2 download)
make quality        # float32 vs int8 vs int4 on real GPT-2, ~3-5 min
make profile        # where one decode step's time goes
make profile-int8   # the same, through QuantizedParams
```

## Results

Reference solution, dev Mac, 2 threads (`make quality`):

| Weights | Size | Bits/byte | Decode | Greedy "Albert Einstein was a" |
|---|---|---|---|---|
| float32 | 498 MB | 1.192 | 49.5 tokens/sec | great scientist, but he was also a |
| int8 per-channel | 243 MB | 1.192 | 9.3 tokens/sec | great scientist, but he was also a |
| int4, group 128 | 203 MB | 1.265 | 2.6 tokens/sec | great physicist, and he was a great |
| int4, group 32 | 211 MB | 1.230 | 3.7 tokens/sec | great physicist, but he was a great |

- **int8 is free in quality:** the same bits per byte to three decimals,
  and the same text.
- **int4 costs a little,** and smaller groups cost less, for a few more
  bytes of scales.
- **The size floor is the embedding table.** `wte` is 154 MB of float32
  (50,257 × 768). It stays float because it's also the output head.
  Quantizing it per row is the obvious next step.
- **Slower, as predicted.** `make profile-int8` shows the four block
  matmuls going from 60% to 91% of the step: that's dequantization.

Where a float32 decode step goes (`make profile`, 256 tokens of context):

| Part | Time | Share |
|---|---|---|
| block matmuls (4 per block × 12) | 7.6 ms | 60% |
| output head (x @ wteᵀ) | 3.5 ms | 28% |
| attention over the cache | 0.8 ms | 6% |
| layer norm + GELU | 0.8 ms | 6% |

(Measured while other jobs shared the CPU, so absolute times move.
The shares are what to compare.)

## Before writing the post

- **Why int8 is nearly lossless:** trained weights are roughly bell
  curves. 255 levels per column is plenty, and per-channel scales keep
  loud columns from swamping quiet ones.
- **Why real kernels are faster:** decode is memory-bound. Reading 1
  byte per weight instead of 4 means 4× less memory traffic, *if* the
  kernel computes on the bytes without expanding them.
- **What production does:** GPTQ and AWQ choose rounding with
  calibration data instead of rounding each weight to nearest, and
  llama.cpp's K-quants use two-level scales. Activations and the KV cache
  get quantized too (FP8 on new GPUs).
- **Stretch ideas:** quantize `wte` per row; compute `(x @ q) * scale`
  so the scale multiply happens on the output instead of the matrix;
  measure the effect of thread count (`VECLIB_MAXIMUM_THREADS`).

## What AI wrote

- **Scaffold:** this README, all tests, `QuantizedParams.__init__`,
  `__contains__` and its choice of which weights to quantize, the
  benchmarks (`quality.py`, `profile_decode.py`), the Makefile, `POST.md`.
- **Hand-written:** int8 and int4 quantize/dequantize, nibble packing,
  `from_params`, `__getitem__`, `nbytes`.

## Post status

Not started. Outline in `POST.md`.
