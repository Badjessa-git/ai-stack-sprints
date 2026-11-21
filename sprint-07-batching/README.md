# Sprint 07: Continuous Batching and Paged Memory

## Why this sprint

Sprint 6's load test showed the problem. Four clients got the same
55 tokens/sec as one client, split four ways, and the last in line waited
1.4 s for its first word. Every request read all 124M weights once per
token, on its own.

**Batching** fixes it. A decode step for 16 sequences multiplies a
(16, 768) matrix by each weight instead of a (1, 768) vector. The weights
are read from memory once and used 16 times. On the dev Mac that step
costs about 4× a single-sequence step, not 16×.

Doing it well is a scheduling problem and a memory problem. This sprint
handles them in that order:

- **Part A, a simulator.** You compare policies on a simulated clock
  before touching a model. *Static* batching runs a batch until its
  slowest member finishes. *Continuous* batching (Orca, 2022) re-decides
  every step: finished requests leave at once and new ones join at once.
- **Part B, paged KV memory.** Sprint 5's cache reserves a full context
  per request. Paging (vLLM's PagedAttention idea, 2023) hands out
  16-position blocks as a sequence grows and takes them back when it
  ends. It's the operating system's virtual-memory trick, applied to
  attention.
- **Part C, the real thing.** You build a batched decode step for GPT-2
  and an engine thread that runs Part A's policy on Part B's memory. It
  goes behind your Sprint 6 server, unchanged, and when memory runs out
  it **preempts** a request and recomputes it later without the client
  noticing.

Learning goals:

1. Explain why batching decode is almost free (memory-bound work), and
   predict tokens/sec from a step-time table.
2. Explain why continuous batching beats static batching on latency and
   throughput, using numbers from your own simulator.
3. Implement block allocation and block tables, and explain the waste
   they remove.
4. Build an engine where concurrent requests give *exactly* the output
   they'd give alone, even when they're preempted and recomputed.

## The plan

Part A needs nothing. Parts B and C import your Sprints 4, 5 and 6.

| Step | File | What | Tests |
|------|------|------|-------|
| A1 | `src/sim.py` | `StaticBatcher`, `_advance` | `TestStatic` |
| A2 | | `ContinuousBatcher` | `TestContinuous` |
| A3 | | `summarize` (throughput, p50/p99) | `TestSummary`, `TestPolicies` |
| | | `make sim`: the policy table | |
| B1 | `src/paged_kv.py` | `BlockAllocator` | `TestAllocator` |
| B2 | | `PagedKVCache.reserve`, `update`, `free_sequence` | `TestPagedCache` |
| B3 | | `SequenceView`: your Sprint 5 `forward_cached` on pages, unchanged | `TestSequenceView` |
| C1 | `src/batch_engine.py` | `batched_decode` | `TestBatchedDecode` |
| C2 | | `BatchingEngine.step`: admit, make room, decode | `TestEngine`, `TestPreemption` |
| C3 | real GPT-2 | `make serve` + `make load` | `tests/test_real_gpt2.py` |

The rule for Part C is the same as Sprint 5's: **batching changes nothing
but speed.** The tests run requests concurrently, including with
preemption, and require exactly the tokens each would get alone.

## Run it

```bash
make test          # 31 tests (1 needs Sprint 4's GPT-2 download)
make sim           # static vs continuous vs one-at-a-time, simulated
make batch-speed   # real GPT-2 decode step time vs batch size
make serve         # Sprint 6's server + the batching engine on :8000
make load          # (another terminal) 1, 4, 8 concurrent clients
```

## Results

Reference solution, dev Mac, 2 threads.

**One batched decode step, GPT-2 small** (`make batch-speed`):

| Batch | Step | Tokens/sec |
|-------|------|------------|
| 1 | 16 ms | 63 |
| 4 | 37 ms | 109 |
| 8 | 47 ms | 171 |
| 16 | 62 ms | 258 |

(Batch 1 is extra fast because NumPy's math library uses a
matrix-*vector* routine for one row. From 2 rows up it's matrix-matrix.)

**Simulator, max batch 16, 1 request/sec** (`make sim`):

| Policy | Tokens/sec | Latency p50 | TTFT p99 |
|--------|-----------|-------------|----------|
| One at a time | 35 | 617 s | 1,312 s |
| Static | 145 | 20.5 s | 35.3 s |
| Continuous | 154 | 8.1 s | 2.7 s |

**Real server, 24 tokens per request** (`make load`), against Sprint 6:

| Clients | Sprint 6 tokens/sec | Batched tokens/sec | Sprint 6 TTFT max | Batched TTFT max |
|---------|---------------------|--------------------|-------------------|------------------|
| 1 | 55 | 53 | 0.02 s | 0.03 s |
| 4 | 55 | 101 | 1.44 s | 0.08 s |
| 8 | — | 141 | — | 0.27 s |

## Before writing the post

- **Why not just threads?** Two threads each running a matrix-vector
  multiply both read all the weights. Batching turns them into one
  matrix-matrix multiply that reads the weights once.
- **Paging's cost:** gathering blocks into a contiguous array every step
  is a copy. vLLM's PagedAttention kernel reads the blocks in place
  instead. That's the stretch goal here, and the real kernel's reason to
  exist.
- **Preemption: recompute vs swap.** vLLM can also copy a preempted
  sequence's blocks to CPU memory and back. On a CPU-only server there's
  nowhere to swap to, so recomputing is the only option here.
- **What production adds:** chunked prefill (a long prompt shouldn't
  stall everyone's decode), prefix sharing (one block, many sequences,
  reference counts), priority and fairness, and speculative decoding.

## What AI wrote

- **Scaffold:** this README, all tests, the simulator's event loop
  (`simulate`), `workload`, `CostModel`, the `Batch`/`Request`/`Seq`
  records, the `PagedKVCache` pool allocation, `SequenceView`'s plumbing,
  the engine's thread, queue and `generate` interface plus the
  `_sample`/`_emit`/`_finish`/`_preempt` helpers, `serve_batched.py`,
  `benchmarks/`, the Makefile, `POST.md`.
- **Hand-written:** both batching policies, `summarize`, the allocator,
  `reserve`/`update`/`free_sequence`, `SequenceView.update`,
  `batched_decode`, and `BatchingEngine.step`.

## Post status

Not started. Outline in `POST.md`.
