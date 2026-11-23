# Sprint 12: An LSM-Tree Key-Value Store (data-systems track)

## Why this sprint

Most write-heavy databases (LevelDB, RocksDB, Cassandra, ScyllaDB, the
storage layers under many vector and time-series databases) are built
on one idea: **never update a file in place.**

- A write goes to a **write-ahead log** (an append-only file, so a crash
  loses nothing) and to a **memtable** (a dict in memory).
- When the memtable is full, it's written out once, sorted, as an
  immutable **SSTable** file, and the log starts over.

Every disk write is therefore a sequential append, which is why LSM
trees write fast.

Reads pay for it. A key might be in the memtable or in any of the
SSTables, so a read checks newest to oldest. Two tricks keep that cheap:

- a **Bloom filter** per table answers "definitely not here" from a few
  bits in memory
- **compaction** merges tables, throwing away overwritten values and
  deleted keys

Deletes are writes too: a **tombstone** that hides older values until
compaction removes both.

Learning goals:

1. Explain the write path and the read path, and why one is fast and the
   other "complicated".
2. Make a log crash-safe: checksums, torn writes, and the order in which
   files must be written so that a crash at any moment loses nothing.
3. Build an SSTable (sorted data, sparse index, Bloom filter, footer) and
   a Bloom filter, and measure what the filter saves.
4. Implement compaction with a k-way merge, and explain write
   amplification.

## The plan

Everything is in `src/lsm.py`. Standard library only.

| Step | What | Tests |
|------|------|-------|
| 1 | `BloomFilter`: sizing, double hashing, add, might_contain | `TestBloom` |
| 2 | `WAL.append`, `WAL.replay` (crc32, stop at a torn tail) | `TestWAL` |
| 3 | `write_sstable` (atomic), `SSTable` (footer, index, bloom, get) | `TestSSTable` |
| 4 | `LSMStore.__init__` (manifest + WAL replay), `put`, `get` | `TestStore` |
| 5 | `flush` (and its write order), crash recovery | `TestRecovery` |
| 6 | `_merged`, `compact` | `TestCompaction` |
| 7 | `scan` | `TestScan` |

**The write order in `flush` is the whole crash-safety argument.** The
SSTable is written and synced, then the MANIFEST is atomically replaced,
and only then is the WAL emptied. Crash before the manifest and the WAL
still holds everything. Crash after it and the SSTable does. The tests
simulate the crash by reopening a store that was never closed.

**A bug the reference solution had, and the test that catches it:** the
Bloom filter's size has to survive a round-trip through whole bytes. A
filter sized at 9,586 bits is stored as 1,199 bytes and reloaded as
9,592 bits. Every hash position then lands in a different place, and
every lookup misses. The fix is rounding the size up to a multiple of 8.
`test_get_every_key` fails loudly until you do.

## Run it

```bash
make test     # 26 tests, under a second
make bench    # 50k writes: throughput, read paths, write amplification
```

## Results

Reference solution, dev Mac (`make bench`), 50,000 random writes of
100-byte values:

| | |
|---|---|
| writes/sec, no fsync | 111,000 (21 flushes, 2 compactions) |
| writes/sec, fsync every write | 32,500 (see the note below) |
| write amplification | 2.2× (SSTable bytes written ÷ bytes put) |

| Reads | Bloom on | Bloom off |
|---|---|---|
| existing keys | 17 µs, 1.03 tables searched | 26 µs, 5.9 tables |
| missing keys | 12 µs, **0.08 tables searched** | 45 µs, 7.0 tables |

The Bloom filter's job is the last row. A missing key without it means
searching every table; with it, almost none.

**The fsync number is a lie the OS tells.** On macOS, `os.fsync` hands
data to the drive but doesn't force the drive to write it to storage.
That takes `fcntl.fcntl(fd, fcntl.F_FULLFSYNC)`, which is far slower.
Databases on macOS (SQLite included) have options for exactly this.
Linux's `fsync` does flush the drive. Durability is a contract with the
whole stack, not one function call.

## Before writing the post

- **Leveled vs tiered compaction:** this store merges everything into
  one table (a "full" compaction). RocksDB keeps levels of growing size
  and merges a little at a time. That gives lower read amplification for
  more write amplification, and the trade is tunable.
- **The three amplifications:** write (bytes written ÷ bytes put), read
  (tables touched per get), space (stale bytes on disk). Every LSM design
  picks a point among them.
- **What's missing:** block-based reads (this loads each table into
  memory), a block cache, compression, concurrent readers and writers,
  range tombstones, and snapshots.
- **Connection to the series:** this storage engine design sits under
  many vector databases (Sprint 10) and search engines (Sprint 11's
  inverted index is often stored in one).

## What AI wrote

- **Scaffold:** this README, all tests, record encoding
  (`encode_record`/`decode_record`), `WAL.reset`/`close`, `SSTable.items`,
  `LSMStore._mem_put`, `_write_manifest`, `_new_table`, `delete`, `close`,
  the benchmark, the Makefile, `POST.md`.
- **Hand-written:** the Bloom filter, WAL append and replay, the SSTable
  writer and reader, the store's open/put/get/flush, the k-way merge,
  compaction, and scan.

## Post status

Not started. Outline in `POST.md`.
