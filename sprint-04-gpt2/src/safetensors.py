"""Read .safetensors files with nothing but NumPy.

The whole format, which is why it's the first thing you build:

    [8 bytes]  N, the header length, as a little-endian unsigned 64-bit int
    [N bytes]  a JSON object (UTF-8):
                 {"wte.weight": {"dtype": "F32", "shape": [50257, 768],
                                 "data_offsets": [0, 154389504]},
                  ...,
                  "__metadata__": {...}}          <- optional, skip it
    [rest]     raw tensor bytes, little-endian, row-major (C order)

data_offsets are [begin, end) byte positions counted from the START OF THE
DATA SECTION (that is, from byte 8 + N), not from the start of the file.

There is no compression, no pickle and no code: a file is a header plus a
big array of bytes. That's why Hugging Face moved to it from PyTorch's
pickle-based .bin files, which can run code when loaded.
"""

import json
import struct

import numpy as np

# safetensors dtype string -> NumPy dtype. BF16 has no NumPy dtype: it's
# handled separately in load().
DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


def read_header(path):
    """Return (header, data_start).

    header: the parsed JSON dict WITHOUT the "__metadata__" entry.
    data_start: the file offset where the data section begins (8 + N).

    struct.unpack("<Q", eight_bytes)[0] reads a little-endian uint64."""
    raise NotImplementedError("read_header (TestReadHeader)")


def bf16_to_f32(raw_u16):
    """bfloat16 is the top 16 bits of a float32. To widen it: take the
    uint16 values, widen them to uint32, shift left by 16, and reinterpret
    those bits as float32 (.view(np.float32), not .astype)."""
    raise NotImplementedError("bf16_to_f32 (TestLoad.test_bf16)")


def load(path, names=None):
    """Return {name: np.ndarray} for every tensor (or only those in `names`).

    For each entry: slice bytes [data_start + begin, data_start + end),
    interpret them with the entry's dtype, reshape to its shape. A shape of
    [] is a scalar (reshape(()) works).

    Use np.memmap(path, dtype=np.uint8, mode="r") for the file, so a 500 MB
    model isn't read twice, and return arrays that don't depend on the
    memmap staying open (np.array(..., copy=True) or .astype makes a copy).

    BF16 entries come back as float32 via bf16_to_f32. Anything else not
    in DTYPES raises ValueError naming the dtype."""
    raise NotImplementedError("load (TestLoad)")


# ---- scaffold: the writer (used by the tests to build small files) ------

_NP_TO_ST = {np.dtype(v): k for k, v in DTYPES.items()}


def save(path, tensors, metadata=None, bf16=()):
    """Write {name: array} as .safetensors. Names in `bf16` are stored as
    bfloat16 (truncated, like real BF16 checkpoints)."""
    header, chunks, offset = {}, [], 0
    for name in sorted(tensors):
        arr = np.array(tensors[name], order="C")  # (ascontiguousarray turns 0-d into 1-d)
        if name in bf16:
            raw = (arr.astype(np.float32).view(np.uint32) >> 16).astype("<u2").tobytes()
            dtype = "BF16"
        else:
            raw = arr.astype(arr.dtype.newbyteorder("<")).tobytes()
            dtype = _NP_TO_ST[arr.dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        chunks.append(raw)
        offset += len(raw)
    if metadata:
        header["__metadata__"] = metadata
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)  # pad to 8 bytes, like the reference writer
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for c in chunks:
            f.write(c)
