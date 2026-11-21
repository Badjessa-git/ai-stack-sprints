# Sprint 12 post outline: an LSM-tree store

Working title: **"Write fast, read… complicated"**

## 1. The artifact

- The benchmark: 111k writes/sec, and the Bloom-filter read table.

## 2. Never update in place

- The write path in plain words: log, memtable, flush.
- Why sequential appends are fast on any disk.

## 3. Surviving a crash

- The WAL with checksums: the torn-tail test.
- The flush write order, and why it's the whole argument.
- The fsync surprise on macOS.

## 4. Reads: newest to oldest

- The Bloom filter in one picture. 7 tables searched vs 0.08.
- The byte-rounding bug: how a filter can be right and still miss everything.

## 5. Compaction and its bill

- k-way merge, tombstones disappearing, write amplification 2.2×.
- Leveled vs tiered in one paragraph.
