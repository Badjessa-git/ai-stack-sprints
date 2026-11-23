import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lsm import WAL, BloomFilter, LSMStore, SSTable, write_sstable  # noqa: E402


def k(i):
    return f"key{i:05d}".encode()


def v(i, gen=0):
    return f"value-{i}-{gen}".encode()


class Tmp(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.dir = self._d.name

    def tearDown(self):
        self._d.cleanup()


class TestBloom(unittest.TestCase):
    def test_sizing(self):
        b = BloomFilter(1000, 0.01)
        self.assertEqual(b.m, 9592)  # ceil(-1000 ln 0.01 / ln2^2) = 9586, rounded up to 8s
        self.assertEqual(b.k, 7)
        self.assertEqual(len(b.bits), 9592 // 8)

    def test_no_false_negatives(self):
        b = BloomFilter(2000, 0.01)
        for i in range(2000):
            b.add(k(i))
        self.assertTrue(all(b.might_contain(k(i)) for i in range(2000)))

    def test_false_positive_rate(self):
        b = BloomFilter(2000, 0.01)
        for i in range(2000):
            b.add(k(i))
        fp = sum(b.might_contain(k(i)) for i in range(10000, 30000)) / 20000
        self.assertLess(fp, 0.02)

    def test_empty_says_no(self):
        self.assertFalse(BloomFilter(10).might_contain(b"x"))


class TestWAL(Tmp):
    def test_replay(self):
        p = os.path.join(self.dir, "w.log")
        w = WAL(p)
        w.append(b"a", b"1")
        w.append(b"b", None)
        w.append(b"a", b"2")
        w.close()
        self.assertEqual(list(WAL.replay(p)), [(b"a", b"1"), (b"b", None), (b"a", b"2")])

    def test_torn_tail_is_dropped(self):
        p = os.path.join(self.dir, "w.log")
        w = WAL(p)
        w.append(b"a", b"1")
        w.append(b"b", b"2")
        w.close()
        data = open(p, "rb").read()
        open(p, "wb").write(data[:-3])  # crash mid-write
        self.assertEqual(list(WAL.replay(p)), [(b"a", b"1")])

    def test_corruption_stops_replay(self):
        p = os.path.join(self.dir, "w.log")
        w = WAL(p)
        for i in range(3):
            w.append(k(i), v(i))
        w.close()
        data = bytearray(open(p, "rb").read())
        first = 8 + struct.unpack_from("<I", data, 4)[0]
        data[first + 10] ^= 0xFF  # flip a byte inside entry 2
        open(p, "wb").write(bytes(data))
        self.assertEqual(list(WAL.replay(p)), [(k(0), v(0))])

    def test_missing_file(self):
        self.assertEqual(list(WAL.replay(os.path.join(self.dir, "nope"))), [])


class TestSSTable(Tmp):
    def make(self, n=100, **kw):
        items = [(k(i), v(i)) for i in range(0, 2 * n, 2)]
        p = write_sstable(os.path.join(self.dir, "t.sst"), items, **kw)
        return SSTable(p), items

    def test_get_every_key(self):
        t, items = self.make()
        for key, val in items:
            self.assertEqual(t.get(key), (True, val))

    def test_missing_keys(self):
        t, _ = self.make()
        for key in (k(1), k(51), b"a", b"zzz", k(99999)):
            self.assertEqual(t.get(key), (False, None))

    def test_tombstone(self):
        items = [(k(i), None if i == 3 else v(i)) for i in range(5)]
        t = SSTable(write_sstable(os.path.join(self.dir, "u.sst"), items))
        self.assertEqual(t.get(k(3)), (True, None), "found, and it's a deletion")
        self.assertEqual(t.get(k(4)), (True, v(4)))

    def test_survives_reopen(self):
        _, items = self.make()
        t = SSTable(os.path.join(self.dir, "t.sst"))
        self.assertTrue(all(t.get(key) == (True, val) for key, val in items))

    def test_sparse_index(self):
        t, _ = self.make(100, index_every=16)
        self.assertEqual(len(t.index_keys), 7)  # ceil(100 / 16)
        self.assertEqual(t.index_offs[0], 0)

    def test_items_in_order(self):
        t, items = self.make()
        self.assertEqual(list(t.items()), items)

    def test_unsorted_raises(self):
        with self.assertRaises(ValueError):
            write_sstable(os.path.join(self.dir, "x.sst"), [(b"b", b"1"), (b"a", b"2")])

    def test_no_tmp_file_left(self):
        self.make()
        self.assertEqual(sorted(os.listdir(self.dir)), ["t.sst"])

    def test_bad_magic(self):
        p = os.path.join(self.dir, "bad.sst")
        open(p, "wb").write(b"\x00" * 64)
        with self.assertRaises(ValueError):
            SSTable(p)


class TestStore(Tmp):
    def test_put_get_overwrite_delete(self):
        s = LSMStore(self.dir)
        s.put(b"a", b"1")
        s.put(b"a", b"2")
        s.put(b"b", b"3")
        s.delete(b"b")
        self.assertEqual((s.get(b"a"), s.get(b"b"), s.get(b"c")), (b"2", None, None))
        s.close()

    def test_across_flushes(self):
        s = LSMStore(self.dir, memtable_bytes=500, compact_at=100)
        for i in range(300):
            s.put(k(i), v(i))
        for i in range(0, 300, 3):
            s.put(k(i), v(i, gen=1))
        for i in range(0, 300, 7):
            s.delete(k(i))
        self.assertGreater(s.stats["flushes"], 5)
        for i in range(300):
            want = None if i % 7 == 0 else v(i, 1) if i % 3 == 0 else v(i)
            self.assertEqual(s.get(k(i)), want, i)
        s.close()

    def test_bloom_filters_skip_tables(self):
        s = LSMStore(self.dir, memtable_bytes=500, compact_at=100)
        for i in range(300):
            s.put(k(i), v(i))
        s.flush()
        s.stats.update(sstable_reads=0, bloom_skips=0)
        for i in range(1000, 1100):
            s.get(k(i))
        self.assertGreater(s.stats["bloom_skips"], 10 * s.stats["sstable_reads"])
        s.close()


class TestRecovery(Tmp):
    def test_unflushed_writes_survive_a_crash(self):
        s = LSMStore(self.dir, memtable_bytes=10**9)
        for i in range(50):
            s.put(k(i), v(i))
        s.delete(k(7))
        # no close(): the process "dies" here
        s2 = LSMStore(self.dir)
        self.assertEqual(s2.get(k(49)), v(49))
        self.assertIsNone(s2.get(k(7)))
        s2.close()

    def test_flushed_and_unflushed(self):
        s = LSMStore(self.dir, memtable_bytes=400, compact_at=100)
        for i in range(200):
            s.put(k(i), v(i))
        s.close()
        s2 = LSMStore(self.dir)
        self.assertTrue(all(s2.get(k(i)) == v(i) for i in range(200)))
        self.assertEqual(len(s2.tables), s.stats["flushes"])
        s2.close()

    def test_wal_is_reset_after_flush(self):
        s = LSMStore(self.dir, memtable_bytes=10**9)
        for i in range(20):
            s.put(k(i), v(i))
        s.flush()
        self.assertEqual(os.path.getsize(os.path.join(self.dir, "wal.log")), 0)
        s.close()


class TestCompaction(Tmp):
    def test_compaction_keeps_latest_and_drops_tombstones(self):
        s = LSMStore(self.dir, memtable_bytes=300, compact_at=4)
        for gen in range(3):
            for i in range(100):
                s.put(k(i), v(i, gen))
        for i in range(0, 100, 2):
            s.delete(k(i))
        s.flush()
        s.compact()
        self.assertEqual(len(s.tables), 1)
        self.assertEqual(s.tables[0].n, 50)  # tombstones and old values gone
        self.assertTrue(all(s.get(k(i)) == (None if i % 2 == 0 else v(i, 2)) for i in range(100)))
        self.assertEqual(len([f for f in os.listdir(self.dir) if f.endswith(".sst")]), 1)
        s.close()

    def test_automatic(self):
        s = LSMStore(self.dir, memtable_bytes=200, compact_at=3)
        for i in range(300):
            s.put(k(i % 50), v(i))
        self.assertGreater(s.stats["compactions"], 0)
        self.assertLess(len(s.tables), 3)
        s.close()


class TestScan(Tmp):
    def test_range_newest_wins_no_tombstones(self):
        s = LSMStore(self.dir, memtable_bytes=300, compact_at=100)
        for i in range(60):
            s.put(k(i), v(i))
        s.put(k(10), b"new")
        s.delete(k(11))
        got = list(s.scan(k(8), k(14)))
        self.assertEqual([key for key, _ in got], [k(8), k(9), k(10), k(12), k(13)])
        self.assertEqual(dict(got)[k(10)], b"new")
        self.assertEqual(len(list(s.scan())), 59)
        s.close()


if __name__ == "__main__":
    unittest.main()
