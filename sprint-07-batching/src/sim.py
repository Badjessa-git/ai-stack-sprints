"""Part A: a scheduler simulator. Design the policy before touching a model.

Sprint 6 showed the problem: one request at a time, each re-reading all
124M weights per token, so four clients share 55 tokens/sec. The fix is
batching: run one decode step for many requests at once, so the weights
are read once and used B times. A step for B sequences costs a bit more
than a step for one, not B times more.

How you form batches decides everything, and it's cheaper to find out
in a simulator than on a real model. Time here is simulated: a step's
cost comes from a cost model, so a day of traffic runs in a second.

Two policies:

- **Static batching**: wait until there are requests, take up to
  max_batch of them, and run that batch until EVERY one of them
  finishes. Requests that arrive meanwhile wait, and short requests sit
  idle in the batch until the longest one is done.
- **Continuous batching** (Orca, 2022; vLLM): decide every step. Finished
  requests leave immediately, and waiting ones join immediately (their
  prompt is prefilled in that step). The batch is never held hostage by
  its slowest member.

A request's life: arrives -> waits -> prefill (makes its first token) ->
decode steps (one token each) -> done after output_len tokens.

Order of work (tests/test_sim.py):
    1. StaticBatcher.schedule / complete       TestStatic
    2. ContinuousBatcher.schedule / complete   TestContinuous
    3. summarize                               TestSummary
Then make sim: both policies across load levels.
"""

import random
from dataclasses import dataclass, field


@dataclass
class Request:
    id: int
    arrival: float  # seconds
    prompt_len: int  # tokens
    output_len: int  # tokens to generate (>= 1)
    # filled in as it runs:
    generated: int = 0
    first_token: float = None
    finished: float = None


@dataclass
class Batch:
    prefill: list = field(default_factory=list)  # requests whose prompt runs this step
    decode: list = field(default_factory=list)  # requests generating one token this step

    def __len__(self):
        return len(self.prefill) + len(self.decode)


@dataclass
class CostModel:
    """Seconds per step. The defaults are fitted to benchmarks/batch_speed.py
    on the reference solution (GPT-2 small, dev Mac CPU): a decode step's
    cost is mostly reading the weights once (base), plus a little per
    sequence (B=4: ~37 ms, B=16: ~62 ms). Re-fit them to your own numbers."""

    base: float = 0.025  # read every weight once
    per_decode: float = 0.0023  # each sequence in the decode batch
    per_prefill_token: float = 0.0015  # each prompt token being prefilled (~1.1-1.8 ms measured)

    def __call__(self, batch):
        return (self.base + self.per_decode * len(batch.decode)
                + self.per_prefill_token * sum(r.prompt_len for r in batch.prefill))


class StaticBatcher:
    def __init__(self, max_batch):
        self.max_batch = max_batch
        self.waiting = []  # arrived, not started (in arrival order)
        self.running = []  # the current batch, including members already done

    def add(self, req):
        self.waiting.append(req)

    def has_work(self):
        return bool(self.waiting or self.running)

    def schedule(self, now):
        """Return the Batch to run next.

        - If no batch is running: start one with up to max_batch requests
          from the front of self.waiting (move them to self.running). The
          step is their prefill: Batch(prefill=those).
        - Else: Batch(decode=[running requests not yet finished]).
        (simulate() only calls this when has_work() is true.)"""
        raise NotImplementedError("StaticBatcher.schedule (TestStatic)")

    def complete(self, batch, now):
        """The step finished at time `now`. Every request in the batch made
        one token: set first_token (if it's the first), add 1 to generated,
        and set finished = now when generated reaches output_len.
        Then, if every request in self.running is finished, the batch is
        over: empty self.running."""
        raise NotImplementedError("StaticBatcher.complete (TestStatic)")


class ContinuousBatcher:
    def __init__(self, max_batch):
        self.max_batch = max_batch
        self.waiting = []
        self.running = []  # only unfinished requests

    def add(self, req):
        self.waiting.append(req)

    def has_work(self):
        return bool(self.waiting or self.running)

    def schedule(self, now):
        """Every step: everything in self.running decodes, and as many
        waiting requests as fit (len(running) + new <= max_batch) are
        admitted, in arrival order, and prefilled in this same step.
        Admitted requests move to self.running."""
        raise NotImplementedError("ContinuousBatcher.schedule (TestContinuous)")

    def complete(self, batch, now):
        """Same token bookkeeping as StaticBatcher; then drop finished
        requests from self.running right away."""
        raise NotImplementedError("ContinuousBatcher.complete (TestContinuous)")


def _advance(batch, now):
    """Shared token bookkeeping for both policies: every request in the
    batch made one token at time `now` (set first_token if unset, add 1 to
    generated, set finished when generated reaches output_len)."""
    raise NotImplementedError("_advance (TestStatic)")


def summarize(requests):
    """Return a dict for finished requests:
        throughput      total generated tokens / (last finish - first arrival)
        ttft_p50, ttft_p99          time to first token (first_token - arrival)
        latency_p50, latency_p99    finished - arrival
    Percentile: sort the values, take index min(n - 1, int(q * n)) (the
    'nearest rank' rule: no interpolation)."""
    raise NotImplementedError("summarize (TestSummary)")


# ---- scaffold: the event loop and a workload ---------------------------------


def simulate(requests, scheduler, cost=None):
    """Run the requests through the scheduler on a simulated clock. Returns
    the same request objects, filled in. Idle time skips straight to the
    next arrival."""
    cost = cost or CostModel()
    pending = sorted(requests, key=lambda r: r.arrival)
    now, i = 0.0, 0
    while i < len(pending) or scheduler.has_work():
        while i < len(pending) and pending[i].arrival <= now:
            scheduler.add(pending[i])
            i += 1
        if not scheduler.has_work():
            now = pending[i].arrival
            continue
        batch = scheduler.schedule(now)
        if not len(batch):
            raise RuntimeError("scheduler returned an empty batch while it has work")
        now += cost(batch)
        scheduler.complete(batch, now)
    return requests


def workload(n=200, rate=8.0, seed=0, prompt=(20, 200), output=(10, 300)):
    """n requests, Poisson arrivals at `rate` per second, prompt and output
    lengths uniform in the given ranges. Real traffic is skewed toward short
    outputs with a long tail; the uniform range keeps it readable."""
    rng = random.Random(seed)
    t, reqs = 0.0, []
    for i in range(n):
        t += rng.expovariate(rate)
        reqs.append(Request(i, t, rng.randint(*prompt), rng.randint(*output)))
    return reqs
