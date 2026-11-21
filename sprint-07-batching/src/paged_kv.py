"""Part B: a paged KV cache (the idea behind vLLM's PagedAttention).

Sprint 5's cache preallocates max_len positions for every request. With
many requests that's ruinous: a request that stops after 20 tokens still
holds 1,024 slots, and you can't admit a new request until memory for
its *worst case* is free. Most of the memory is reserved and empty.

Paging fixes it the way an operating system does. The cache is one big
pool of fixed-size **blocks** (here 16 positions each). A sequence owns a
**block table**, the list of block ids holding its positions in order:

    position p lives in block  table[p // block_size],  slot  p % block_size

A sequence takes a new block only when it fills its last one, and gives
every block back the moment it finishes. Waste is at most one partly
filled block per sequence.

Order of work (tests/test_paged_kv.py):
    1. BlockAllocator                          TestAllocator
    2. PagedKVCache.reserve / update / free    TestPagedCache
    3. SequenceView                            TestSequenceView
       (the adapter that lets Sprint 5's forward_cached run on a paged
       sequence, unchanged)
"""

import numpy as np


class OutOfBlocks(RuntimeError):
    """No free block: the scheduler must wait (or preempt someone)."""


class BlockAllocator:
    def __init__(self, num_blocks):
        """Track which of num_blocks block ids are free. A list used as a
        stack is enough: alloc pops, free pushes."""
        raise NotImplementedError("BlockAllocator.__init__ (TestAllocator)")

    def alloc(self):
        """Return a free block id. Raise OutOfBlocks if there is none."""
        raise NotImplementedError("alloc (TestAllocator)")

    def free(self, block):
        """Return a block to the pool. Freeing a block that isn't
        allocated is a bug in the caller: raise ValueError."""
        raise NotImplementedError("free (TestAllocator)")

    @property
    def num_free(self):
        raise NotImplementedError("num_free (TestAllocator)")


class PagedKVCache:
    def __init__(self, n_layer, n_head, head_dim, num_blocks, block_size=16, dtype=np.float32):
        # the pool: one array per K and V, every layer, every block
        self.k = np.zeros((n_layer, num_blocks, n_head, block_size, head_dim), dtype)
        self.v = np.zeros_like(self.k)
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks)
        self.tables = {}  # seq_id -> [block ids]
        self.lengths = {}  # seq_id -> positions filled

    @classmethod
    def for_model(cls, model, num_blocks, block_size=16):
        c = model.cfg
        return cls(c.n_layer, c.n_head, c.n_embd // c.n_head, num_blocks, block_size)

    def add_sequence(self, seq_id):
        self.tables[seq_id] = []
        self.lengths[seq_id] = 0

    def reserve(self, seq_id, n_new):
        """Make sure the sequence has blocks for positions
        [length, length + n_new): allocate blocks until
        len(table) * block_size >= length + n_new.

        All or nothing: if the allocator runs out partway, give back the
        blocks this call took, then re-raise OutOfBlocks."""
        raise NotImplementedError("reserve (TestPagedCache)")

    def update(self, seq_id, layer, k_new, v_new):
        """Sprint 5's KVCache.update, paged. k_new, v_new: (H, T_new, D) for
        positions [length, length + T_new) (blocks already reserved).
        Write each position p into block table[p // bs], slot p % bs, then
        return (k_all, v_all) as CONTIGUOUS (H, length + T_new, D) arrays,
        gathered from the blocks in table order.

        Like Sprint 5: don't change the length here (advance does).
        Gathering: self.k[layer, table] is (n_blocks, H, bs, D); move H to
        the front and merge the block and slot axes, then cut to length."""
        raise NotImplementedError("update (TestPagedCache)")

    def advance(self, seq_id, n):
        self.lengths[seq_id] += n

    def free_sequence(self, seq_id):
        """Give back every block of the sequence and forget it."""
        raise NotImplementedError("free_sequence (TestPagedCache)")


class SequenceView:
    """One sequence of a PagedKVCache, dressed up as Sprint 5's KVCache:
    .length, .max_len, .update(layer, k, v), .advance(n). So Sprint 5's
    forward_cached(model, ids, view) runs on paged memory unchanged.

    forward_cached calls update() for layer 0, 1, 2, ... and then advance()
    once, so update() must reserve blocks when layer == 0 (that's the
    first moment it knows how many new positions are coming)."""

    def __init__(self, cache, seq_id, max_len):
        self.cache, self.seq_id, self.max_len = cache, seq_id, max_len
        if seq_id not in cache.tables:
            cache.add_sequence(seq_id)

    @property
    def length(self):
        return self.cache.lengths[self.seq_id]

    def update(self, layer, k_new, v_new):
        """Reserve (on layer 0), then delegate to the cache. Raise
        ValueError, like Sprint 5, if length + T_new > max_len."""
        raise NotImplementedError("SequenceView.update (TestSequenceView)")

    def advance(self, n):
        self.cache.advance(self.seq_id, n)
