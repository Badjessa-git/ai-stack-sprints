"""Quantization: store weights in 8 or 4 bits instead of 32.

A float32 weight takes 4 bytes. GPT-2 small is 124M of them, about
500 MB. Quantizing stores each weight as a small integer plus, for each
group of weights, one float scale:

    w  ~  q * scale,     q an integer in [-127, 127] (int8) or [-7, 7] (int4)

Symmetric quantization: scale = max(|w| in the group) / qmax, then
q = round(w / scale). The worst error is half a step, scale / 2. So the
whole game is choosing groups whose max is small: one outlier makes the
whole group's steps coarse.

- **per-tensor**: one scale for the whole matrix. One outlier ruins it.
- **per-channel**: one scale per output column. That's the int8 standard.
- **per-group**: one scale per 32 (or 64, 128) consecutive weights. With
  4 bits you need groups this small to keep quality. That's how llama.cpp's
  Q4 formats and GPTQ/AWQ checkpoints work.

Two int4 values fit in one byte, so int4 storage packs them.

Order of work (tests/test_quant.py):
    1. quantize_int8 / dequantize_int8      TestInt8
    2. pack_int4 / unpack_int4              TestPack
    3. quantize_int4 / dequantize_int4      TestInt4
    4. QuantizedParams                      TestQuantizedParams
"""

import numpy as np


def quantize_int8(w, axis=0):
    """Symmetric int8, one scale per slice along `axis`. Return (q, scale).

    GPT-2's weights are (in, out) (Sprint 4's Conv1D layout), so axis=0
    reduces over the inputs and gives one scale per output column. scale
    has w's shape with that axis kept as size 1 (keepdims), so q * scale
    broadcasts back.

    - scale = max(|w|) over `axis` / 127, as float32
    - a slice that is all zeros gets scale 1.0 (avoid dividing by 0)
    - q = round(w / scale), clipped to [-127, 127], as int8"""
    raise NotImplementedError("quantize_int8 (TestInt8)")


def dequantize_int8(q, scale):
    raise NotImplementedError("dequantize_int8 (TestInt8)")


def pack_int4(q):
    """Pack int values in [-8, 7] two per byte along the last axis (which
    must have even length, else ValueError). Shift each value by +8 to get
    0..15. The EVEN index goes in the low 4 bits, the odd one in the high 4.
    Returns uint8 with the last axis halved."""
    raise NotImplementedError("pack_int4 (TestPack)")


def unpack_int4(packed):
    """The inverse of pack_int4: uint8 (..., n/2) -> int8 (..., n)."""
    raise NotImplementedError("unpack_int4 (TestPack)")


def quantize_int4(w, group_size=32):
    """Symmetric int4 in groups along axis 0 (the input axis of a GPT-2
    (in, out) weight): every `group_size` consecutive rows of each column
    share a scale. Return (packed, scale).

    - w.shape[0] must be divisible by group_size (else ValueError)
    - view w as (n_groups, group_size, out); scale = max(|w|) per group
      / 7, shape (n_groups, 1, out), float32; all-zero groups get 1.0
    - q = round(w / scale) clipped to [-7, 7], back to (in, out)
    - pack along axis 0: pack_int4 packs the LAST axis, so pack q.T and
      store that: packed is (out, in / 2) uint8"""
    raise NotImplementedError("quantize_int4 (TestInt4)")


def dequantize_int4(packed, scale, group_size=32):
    """The inverse: (out, in/2) uint8 + (n_groups, 1, out) scales ->
    (in, out) float32."""
    raise NotImplementedError("dequantize_int4 (TestInt4)")


class QuantizedParams(dict):
    """A params dict that stores some weights quantized and dequantizes
    them when they're read, so Sprint 4's GPT2 (which only ever does
    p[name]) runs on it unchanged:

        qp = QuantizedParams.from_params(model.p, bits=8)
        GPT2(cfg, qp).forward(ids)

    At rest, the quantized weights take 1/4 (int8) or about 1/8 (int4) of
    the memory. The price: every read dequantizes a whole matrix, so
    NumPy runs SLOWER, not faster. Real speedups need kernels that compute
    on the integers directly (llama.cpp, bitsandbytes). The benchmark
    measures both sides honestly."""

    QUANTIZE = (".attn.c_attn.weight", ".attn.c_proj.weight", ".mlp.c_fc.weight", ".mlp.c_proj.weight")

    def __init__(self, bits=8, group_size=32):
        super().__init__()
        self.bits, self.group_size = bits, group_size
        self.packed = {}  # name -> (q, scale)

    @classmethod
    def from_params(cls, params, bits=8, group_size=32):
        """Quantize every block matrix (names ending in QUANTIZE); keep
        embeddings, layer norms and biases in float32 (they're small, or
        sensitive: wte is also the output head). Store the quantized ones
        in self.packed and the rest as normal dict entries."""
        raise NotImplementedError("from_params (TestQuantizedParams)")

    def __getitem__(self, name):
        """Dequantize on read if the name is in self.packed; else the
        normal dict lookup."""
        raise NotImplementedError("__getitem__ (TestQuantizedParams)")

    def __contains__(self, name):
        return name in self.packed or dict.__contains__(self, name)

    def nbytes(self):
        """Bytes at rest: the float entries plus every q and scale."""
        raise NotImplementedError("nbytes (TestQuantizedParams)")
