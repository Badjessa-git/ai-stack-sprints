"""An LSM-tree key-value store: the design under LevelDB, RocksDB,
Cassandra and most write-heavy databases.

The idea: never update a file in place. Writes go to two places:

1. a **write-ahead log** (WAL): appended to disk first, so a crash loses
   nothing
2. a **memtable**: an in-memory dict, so reads of fresh data are instant

When the memtable gets big, it's written out once, sorted, as an
immutable **SSTable** (sorted string table) file, and the WAL is reset.
Writes are therefore always sequential appends, which is why LSM trees
write fast.

Reads pay for it. A key may live in the memtable or in any SSTable, so
a read checks newest to oldest. Two things keep that cheap:

- a **Bloom filter** per SSTable answers "definitely not here" without
  touching the file
- **compaction** merges SSTables into fewer, dropping overwritten values
  and deletions

Deletes are writes too: a **tombstone** (value None) that hides older
values until compaction removes both.

Keys and values are bytes. Order of work (tests/test_lsm.py):
    1. BloomFilter            TestBloom
    2. WAL.append / replay    TestWAL
    3. write_sstable, SSTable TestSSTable
    4. LSMStore               TestStore, TestRecovery, TestCompaction
"""

import hashlib
import heapq
import json
import math
import os
import struct
import zlib

TOMBSTONE_LEN = 0xFFFFFFFF  # val_len that marks a deletion
MAGIC = 0x4C534D31  # "LSM1"


# ---- scaffold: record encoding --------------------------------------------------


def encode_record(key, value):
    """[u32 key_len][u32 val_len, or TOMBSTONE_LEN if value is None][key][value]"""
    if value is None:
        return struct.pack("<II", len(key), TOMBSTONE_LEN) + key
    return struct.pack("<II", len(key), len(value)) + key + value


def decode_record(buf, off):
    """-> (key, value or None, next offset)"""
    klen, vlen = struct.unpack_from("<II", buf, off)
    off += 8
    key = bytes(buf[off:off + klen])
    off += klen
    if vlen == TOMBSTONE_LEN:
        return key, None, off
    return key, bytes(buf[off:off + vlen]), off + vlen


# ---- Bloom filter ----------------------------------------------------------------


class BloomFilter:
    """A bit array and k hash functions. add(key) sets k bits;
    might_contain(key) is False only if one of its k bits is unset. So it
    never says "no" to a key that was added (no false negatives), and says
    "maybe" to a key that wasn't with probability ~fp_rate."""

    def __init__(self, n, fp_rate=0.01, m=None, k=None):
        """Size it for n keys at fp_rate:
            m = ceil(-n * ln(fp_rate) / ln(2)**2) bits, rounded UP to a
                multiple of 8 (at least 8)
            k = max(1, round(m / n * ln 2)) hashes
        (m and k may be given directly, when loading from a file.)
        Store the bits in a bytearray of m / 8 bytes.

        Why the rounding: the SSTable stores the bits as whole bytes and
        rebuilds m as len(bytes) * 8. If m weren't a multiple of 8, the
        reloaded filter would compute every position modulo a different m,
        and every lookup would miss."""
        raise NotImplementedError("BloomFilter.__init__ (TestBloom)")

    def _positions(self, key):
        """k bit positions by double hashing: h1, h2 = the two little-endian
        u64 halves of hashlib.blake2b(key, digest_size=16).digest();
        position i = (h1 + i * h2) % m."""
        raise NotImplementedError("_positions (TestBloom)")

    def add(self, key):
        raise NotImplementedError("add (TestBloom)")

    def might_contain(self, key):
        raise NotImplementedError("might_contain (TestBloom)")


# ---- write-ahead log ------------------------------------------------------------------


class WAL:
    """Each entry on disk: [u32 crc32 of payload][u32 payload length][payload],
    payload = encode_record(key, value)."""

    def __init__(self, path, sync=False):
        self.path, self.sync = path, sync
        self.f = open(path, "ab")

    def append(self, key, value):
        """Write one entry and flush it (os.fsync too if self.sync: that's
        what makes it survive a power cut, and what makes it slow)."""
        raise NotImplementedError("WAL.append (TestWAL)")

    @staticmethod
    def replay(path):
        """Yield (key, value) for every complete, intact entry, in order.

        A crash can leave a half-written last entry: if the header or the
        payload is cut short, or the crc doesn't match, stop there (don't
        raise). Everything before it was written whole and is kept."""
        raise NotImplementedError("WAL.replay (TestWAL)")

    def reset(self):
        self.f.close()
        self.f = open(self.path, "wb")

    def close(self):
        self.f.close()


# ---- SSTables ---------------------------------------------------------------------------

FOOTER = struct.Struct("<QQQQIII")  # index_off, index_len, bloom_off, bloom_len, n, bloom_k, magic


def write_sstable(path, items, index_every=16, fp_rate=0.01):
    """Write sorted (key, value-or-None) pairs as one SSTable file:

        [data]    every record, encode_record, in key order
        [index]   every index_every-th record's key and its data offset:
                  encode_record(key, struct.pack("<Q", offset))
        [bloom]   the Bloom filter's bits (over every key, tombstones too)
        [footer]  FOOTER.pack(index_off, index_len, bloom_off, bloom_len,
                              n_records, bloom.k, MAGIC)

    Keys must be strictly increasing (raise ValueError if not). Write to
    path + ".tmp", fsync, then os.replace: a crash mid-write never leaves
    a half file under the real name."""
    raise NotImplementedError("write_sstable (TestSSTable)")


class SSTable:
    def __init__(self, path):
        """Read the footer (the last FOOTER.size bytes; check MAGIC, else
        ValueError), then load the index into two lists, self.index_keys
        and self.index_offs, and rebuild the Bloom filter
        (BloomFilter(n, m=bloom_len * 8, k=bloom_k) with its bits set to the
        stored bytes). Keep the data section in memory as self.data
        (bytes): fine at this size; real systems read blocks on demand."""
        raise NotImplementedError("SSTable.__init__ (TestSSTable)")

    def get(self, key):
        """Return (found, value). value None with found True = tombstone.

        1. bloom says no -> (False, None) without touching the data
        2. find the last index key <= key (bisect.bisect_right - 1); none
           -> not found
        3. scan records from that offset until a key > key or the end of
           that index block"""
        raise NotImplementedError("SSTable.get (TestSSTable)")

    def items(self):
        """Every (key, value-or-None) in order."""
        off = 0
        while off < len(self.data):
            k, v, off = decode_record(self.data, off)
            yield k, v


# ---- the store ----------------------------------------------------------------------------------


class LSMStore:
    """
    dir/
      MANIFEST          {"tables": ["000001.sst", ...] oldest first, "next": 2}
      wal.log           writes since the last flush
      000001.sst ...
    """

    def __init__(self, path, memtable_bytes=64 * 1024, compact_at=4, sync=False):
        """Open (or create) the store: read MANIFEST if it exists, open its
        SSTables (oldest first in self.tables), replay wal.log into the
        memtable (self.mem, a dict), then open the WAL for appending.
        Track the memtable's size as the sum of len(key) + len(value or b"")."""
        raise NotImplementedError("LSMStore.__init__ (TestStore, TestRecovery)")

    def _mem_put(self, key, value):
        if key in self.mem:
            self.mem_size -= len(key) + len(self.mem[key] or b"")
        self.mem[key] = value
        self.mem_size += len(key) + len(value or b"")

    def put(self, key, value):
        """WAL first, then memtable; flush when mem_size > memtable_bytes."""
        raise NotImplementedError("put (TestStore)")

    def delete(self, key):
        self.put(key, None)

    def get(self, key):
        """Memtable, then SSTables newest to oldest. The first place the key
        is found wins; a tombstone means None. Count stats: "bloom_skips"
        when a table's bloom filter rules the key out, "sstable_reads" when
        a table's data is actually searched."""
        raise NotImplementedError("get (TestStore)")

    def _write_manifest(self):
        """Atomically: write MANIFEST.tmp, fsync, os.replace."""
        tmp = os.path.join(self.path, "MANIFEST.tmp")
        with open(tmp, "w") as f:
            json.dump({"tables": [os.path.basename(t.path) for t in self.tables], "next": self.next_id}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, os.path.join(self.path, "MANIFEST"))

    def _new_table(self, items):
        name = os.path.join(self.path, f"{self.next_id:06d}.sst")
        self.next_id += 1
        write_sstable(name, items)
        self.stats["bytes_written"] += os.path.getsize(name)
        return SSTable(name)

    def flush(self):
        """Memtable -> a new SSTable (sorted, tombstones included), add it
        to self.tables, write the MANIFEST, THEN reset the WAL and the
        memtable (in that order: if we crash before the manifest is
        written, the WAL still has everything). Then compact if
        len(self.tables) >= compact_at."""
        raise NotImplementedError("flush (TestStore, TestRecovery)")

    def _merged(self, sources):
        """k-way merge of sorted (key, value) iterators given NEWEST FIRST:
        yield each key once, with the newest value (tombstones included).
        heapq.merge over (key, source_index, value) tuples does it: equal
        keys come out newest first; skip the rest."""
        raise NotImplementedError("_merged (TestCompaction, TestScan)")

    def compact(self):
        """Full compaction: merge every SSTable into ONE (newest value wins),
        dropping tombstones (nothing older is left for them to hide).
        Write the new table and the MANIFEST, then delete the old files."""
        raise NotImplementedError("compact (TestCompaction)")

    def scan(self, lo=None, hi=None):
        """Yield (key, value) with lo <= key < hi (None = unbounded), in key
        order, newest value per key, tombstones hidden. Merge the memtable
        (sorted) and every SSTable, newest first."""
        raise NotImplementedError("scan (TestScan)")

    def close(self):
        self.wal.close()
