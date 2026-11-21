import json
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.safetensors import bf16_to_f32, load, read_header, save  # noqa: E402


def handmade_file(path):
    """A file built byte by byte, so your reader isn't only tested against
    the scaffold's writer: two tensors, data_offsets from the data start."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="<f4")  # 16 bytes
    b = np.array([7, -8, 9], dtype="<i8")  # 24 bytes
    header = {
        "__metadata__": {"format": "pt"},
        "a": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
        "b": {"dtype": "I64", "shape": [3], "data_offsets": [16, 40]},
    }
    blob = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(a.tobytes() + b.tobytes())
    return 8 + len(blob)


class Tmp(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "t.safetensors")

    def tearDown(self):
        self.dir.cleanup()


class TestReadHeader(Tmp):
    def test_handmade(self):
        start = handmade_file(self.path)
        header, data_start = read_header(self.path)
        self.assertEqual(data_start, start)
        self.assertEqual(set(header), {"a", "b"}, "drop __metadata__")
        self.assertEqual(header["b"]["data_offsets"], [16, 40])

    def test_empty_file_of_tensors(self):
        save(self.path, {})
        header, start = read_header(self.path)
        self.assertEqual(header, {})
        self.assertEqual(start, os.path.getsize(self.path))


class TestLoad(Tmp):
    def test_handmade(self):
        handmade_file(self.path)
        t = load(self.path)
        np.testing.assert_array_equal(t["a"], [[1, 2], [3, 4]])
        self.assertEqual(t["a"].dtype, np.float32)
        np.testing.assert_array_equal(t["b"], [7, -8, 9])
        self.assertEqual(t["b"].dtype, np.int64)

    def test_roundtrip_dtypes_and_shapes(self):
        rng = np.random.default_rng(0)
        tensors = {
            "f32": rng.standard_normal((3, 4, 5)).astype(np.float32),
            "f16": rng.standard_normal((7,)).astype(np.float16),
            "f64": rng.standard_normal((2, 2)),
            "i32": np.arange(10, dtype=np.int32).reshape(2, 5),
            "u8": np.arange(256, dtype=np.uint8),
            "scalar": np.array(3.5, dtype=np.float32),
        }
        save(self.path, tensors, metadata={"note": "hi"})
        out = load(self.path)
        self.assertEqual(set(out), set(tensors))
        for k, v in tensors.items():
            self.assertEqual(out[k].shape, v.shape, k)
            self.assertEqual(out[k].dtype, v.dtype, k)
            np.testing.assert_array_equal(out[k], v, k)

    def test_subset(self):
        save(self.path, {"x": np.ones(3, np.float32), "y": np.zeros(2, np.float32)})
        self.assertEqual(set(load(self.path, names={"y"})), {"y"})

    def test_bf16(self):
        # bf16 keeps float32's exponent and the top 7 mantissa bits, so
        # numbers with short mantissas survive exactly.
        x = np.array([1.0, -2.5, 0.15625, 3e38, 0.0], dtype=np.float32)
        save(self.path, {"w": x}, bf16={"w"})
        out = load(self.path)["w"]
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out, x, rtol=1e-2)
        np.testing.assert_array_equal(out[[0, 1, 2, 4]], x[[0, 1, 2, 4]])

    def test_bf16_bits(self):
        # 0x3F80 is the top half of 1.0f (0x3F800000)
        np.testing.assert_array_equal(bf16_to_f32(np.array([0x3F80, 0xC000], np.uint16)), [1.0, -2.0])

    def test_unknown_dtype_raises(self):
        header = {"z": {"dtype": "F8_E4M3", "shape": [1], "data_offsets": [0, 1]}}
        blob = json.dumps(header).encode()
        with open(self.path, "wb") as f:
            f.write(struct.pack("<Q", len(blob)) + blob + b"\x00")
        with self.assertRaises(ValueError):
            load(self.path)

    def test_arrays_outlive_the_file(self):
        save(self.path, {"x": np.arange(4, dtype=np.float32)})
        x = load(self.path)["x"]
        os.remove(self.path)
        self.assertEqual(float(x.sum()), 6.0)


if __name__ == "__main__":
    unittest.main()
