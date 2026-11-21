"""Part C: continuous batching for real, behind Sprint 6's server.

Two pieces:

1. batched_decode: ONE decode step for B sequences at once. Every
   Linear layer (qkv, attention output, both MLP matrices, the head) runs
   as a single (B, C) @ W matmul, so the weights are read once for all B
   sequences. That's the entire win. Attention can't be batched the same
   way, because each sequence has its own length and its own cache, so it
   loops over sequences. It's small next to the matmuls.

2. BatchingEngine: Part A's ContinuousBatcher, with Part B's paged
   memory, running in a background thread. Server threads call
   engine.generate(ids, req) (the Sprint 6 engine interface) and get
   tokens from a queue. The engine thread loops on step(): admit,
   prefill, one batched decode, hand out tokens, retire the finished.

When memory runs out mid-decode, it **preempts**: it frees the newest
sequence's blocks and puts it back at the front of the queue with
everything it has generated so far. When it's re-admitted, its prefill
recomputes those tokens and it carries on. The client never notices
(vLLM's "recompute" preemption).

Order of work (tests/test_batch_engine.py):
    1. batched_decode            TestBatchedDecode
    2. BatchingEngine.step       TestEngine, TestPreemption
"""

import collections
import math
import queue
import threading
from dataclasses import dataclass

import numpy as np

from .deps import sprint
from .paged_kv import OutOfBlocks, PagedKVCache, SequenceView

g4 = sprint("sprint-04-gpt2").gpt2
s5 = sprint("sprint-05-kv-cache")


def batched_decode(model, tokens, views):
    """tokens: B ints (each sequence's newest token). views: B
    SequenceViews (same order). Returns logits (B, V) and advances every
    view by 1.

    Like Sprint 5's forward_cached with T_new = 1, but with B rows:
    - x = wte[tokens] + wpe[each view's length]      (B, C)
    - per block: layer_norm, qkv = x @ W + b for ALL rows at once (B, 3C),
      split into q, k, v of shape (B, H, D)
    - attention, per sequence b: views[b].update(layer, k[b][:, None],
      v[b][:, None]) gives (H, L_b, D); scores = q[b] (H, 1, D) @ kᵀ; no
      mask needed (the newest token may see everything); softmax; @ v
    - stack the B results to (B, C), then c_proj, residual, then the MLP
      on all rows at once
    - after the last block, advance every view by 1; ln_f; tied head.
    Keep float32 (math.sqrt: see Sprint 4)."""
    raise NotImplementedError("batched_decode (TestBatchedDecode)")


@dataclass
class Seq:
    id: int
    ids: list  # prompt + every token generated so far
    prompt_len: int
    max_new: int  # tokens to generate in total
    req: object  # Sprint 6's CompletionRequest (temperature, top_p, ...)
    out: queue.Queue  # tokens for the client; None = finished
    rng: np.random.Generator
    view: SequenceView = None

    @property
    def generated(self):
        return len(self.ids) - self.prompt_len


class BatchingEngine:
    def __init__(self, model, tokenizer, max_batch=8, num_blocks=None, block_size=16,
                 name="gpt2", start=True):
        self.model, self.tok, self.name = model, tokenizer, name
        self.eot_id = tokenizer.eot_id
        self.max_batch = max_batch
        n_ctx = model.cfg.n_ctx
        # default: enough blocks for max_batch full contexts
        num_blocks = num_blocks or max_batch * -(-n_ctx // block_size)
        self.cache = PagedKVCache.for_model(model, num_blocks, block_size)
        self.waiting = collections.deque()
        self.running = []
        self.stats = {"steps": 0, "max_running": 0, "preemptions": 0}
        self._next_id = 0
        self._cv = threading.Condition()
        self._stop = False
        if start:
            threading.Thread(target=self._loop, daemon=True).start()

    # ---- the Sprint 6 engine interface ----------------------------------

    def encode(self, text):
        return self.tok.encode(text)

    def token_bytes(self, i):
        return self.tok.token_bytes(i)

    def generate(self, ids, req):
        """Queue the request; yield its tokens as the engine thread makes
        them. Safe to call from many threads at once."""
        room = self.model.cfg.n_ctx - len(ids)
        if room <= 0 or not ids:
            return
        out = queue.Queue()
        with self._cv:
            seq = Seq(self._next_id, list(ids), len(ids), min(req.max_tokens, room), req, out,
                      np.random.default_rng(req.seed))
            self._next_id += 1
            self.waiting.append(seq)
            self._cv.notify()
        while True:
            tok = out.get()
            if tok is None:
                return
            yield tok

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify()

    def _loop(self):
        while True:
            with self._cv:
                while not (self.waiting or self.running or self._stop):
                    self._cv.wait()
                if self._stop:
                    return
            self.step()

    # ---- helpers for step() -----------------------------------------------

    def _sample(self, seq, logits):
        r = seq.req
        return s5.sampling.sample(logits, r.temperature, r.top_k,
                                  r.top_p if r.top_p < 1.0 else None, seq.rng)

    def _emit(self, seq, tok):
        """Record and send one token. Returns True if the sequence is done."""
        seq.ids.append(tok)
        seq.out.put(tok)
        return tok == self.eot_id or seq.generated >= seq.max_new

    def _finish(self, seq):
        self.cache.free_sequence(seq.id)
        seq.view = None
        seq.out.put(None)

    def _preempt(self, seq):
        """Free its memory; back to the FRONT of the queue, keeping its
        tokens so far. Re-admission prefills seq.ids (prompt + generated)."""
        self.cache.free_sequence(seq.id)
        seq.view = None
        self.running.remove(seq)
        with self._cv:
            self.waiting.appendleft(seq)
        self.stats["preemptions"] += 1

    # ---- the scheduler ----------------------------------------------------

    def step(self):
        """One scheduling round.

        1. Admit: while len(self.running) < max_batch and something is
           waiting (popleft under self._cv): give it a SequenceView(
           self.cache, seq.id, n_ctx) and prefill it with Sprint 5's
           s5.kv_cache.forward_cached(model, seq.ids, view). If that raises
           OutOfBlocks: free_sequence(seq.id), put the seq back at the
           front of self.waiting, and stop admitting. Otherwise sample a
           token from the last row, _emit it, and either _finish it (done)
           or add it to self.running.

           A preempted sequence comes back with generated > 0: its prefill
           covers prompt + generated, and the token sampled from it is
           simply the next one. Nothing gets re-sent.

        2. Make room: every running sequence needs 1 more position. If
           self.cache.allocator.num_free < the number of sequences whose
           next position starts a new block (length % block_size == 0),
           _preempt the most recently admitted sequence (self.running[-1])
           and check again.

        3. Decode: if anything is running, batched_decode all of them
           (newest token = seq.ids[-1]), sample one token each, _emit, and
           _finish + remove the done ones.

        4. stats: "steps" += 1; "max_running" = max(itself, len(running))
           as measured during the decode."""
        raise NotImplementedError("step (TestEngine, TestPreemption)")
