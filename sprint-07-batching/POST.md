# Sprint 07 post outline: continuous batching and paged memory

Working title: **"Why LLM servers don't work like web servers"**

## 1. The artifact

- The load-test table, before (Sprint 6) and after: 4 clients,
  55 → 101 tokens/sec, worst first token 1.44 s → 0.08 s.

## 2. Why batching is nearly free

- Decode reads every weight once per step. Show the batch_speed table:
  16 sequences cost about 4× one sequence's step, not 16×.

## 3. Static vs continuous

- The simulator's table. The plain version: a static batch is held
  hostage by its longest request.
- Why you design this in a simulator first.

## 4. Memory, paged

- Sprint 5's cache reserves the worst case. Paging allocates as you go.
- The block-table picture, and the one-line address formula.

## 5. Preemption

- What happens when memory runs out mid-reply, and why recomputing is
  simpler than swapping. The test that proves the client never notices.

## 6. What vLLM adds

- Only what the README's checklist covers.
