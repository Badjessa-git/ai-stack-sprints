"""A KV cache for your Sprint 4 GPT-2.

Without a cache, generating token number t reruns the model over all t
tokens so far, so a 500-token reply costs about 500²/2 token-passes. But
a token's key and value never change once computed: position 3's key is the
same whether the sequence is 4 tokens long or 400. So keep them. Each new
token then costs one token-pass: compute its own q, k, v, append k and v
to the cache, and attend over everything cached.

The cache is two arrays per layer, preallocated to the longest sequence
you'll allow:

    k[layer] : (H, max_len, D)       v[layer] : (H, max_len, D)
    length   : how many positions are filled (the same for every layer)

Order of work (tests/test_kv_cache.py):
    1. KVCache.update            TestKVCache
    2. attention_cached          TestAttentionCached
    3. forward_cached            TestForwardCached
"""

import math

import numpy as np

from .deps import sprint

g4 = sprint("sprint-04-gpt2").gpt2  # your Sprint 4 model: layer_norm, mlp, softmax, ...


class KVCache:
    def __init__(self, n_layer, n_head, head_dim, max_len, dtype=np.float32):
        self.k = np.zeros((n_layer, n_head, max_len, head_dim), dtype)
        self.v = np.zeros((n_layer, n_head, max_len, head_dim), dtype)
        self.max_len = max_len
        self.length = 0

    @classmethod
    def for_model(cls, model, max_len=None):
        c = model.cfg
        return cls(c.n_layer, c.n_head, c.n_embd // c.n_head, max_len or c.n_ctx)

    def update(self, layer, k_new, v_new):
        """Write k_new, v_new (H, T_new, D) into this layer at positions
        [length, length + T_new), and return (k_all, v_all), the filled part
        [0, length + T_new) of this layer.

        Don't change self.length here: every layer writes the same
        positions, so forward_cached calls advance() once, after the last
        layer. Raise ValueError if the new tokens don't fit in max_len."""
        raise NotImplementedError("update (TestKVCache)")

    def advance(self, n):
        self.length += n

    def reset(self):
        self.length = 0


def attention_cached(x, w_qkv, b_qkv, w_out, b_out, n_head, cache, layer):
    """Sprint 4's attention, for x = the NEW tokens only, (T_new, C), which
    sit at absolute positions [start, start + T_new), start = cache.length.

    Same steps as Sprint 4, except:
    - k and v for the new tokens go through cache.update(layer, k, v),
      which hands back ALL keys and values so far, (H, start + T_new, D)
    - scores are (H, T_new, start + T_new): new queries x every key
    - the causal mask is offset: new token i (absolute position start + i)
      may see keys 0 .. start + i. With start = 0 this is Sprint 4's mask.
    Keep it float32 (math.sqrt, not np.sqrt: see Sprint 4's attention).
    """
    raise NotImplementedError("attention_cached (TestAttentionCached)")


def forward_cached(model, ids, cache):
    """Sprint 4's GPT2.forward for the NEW ids only, using and filling the
    cache. Returns logits (T_new, V) for the new positions.

    - positions are [cache.length, cache.length + T_new): use those rows
      of wpe, not 0..T_new
    - each block: Sprint 4's block, but with attention_cached in place of
      attention (use g4.layer_norm and g4.mlp for the rest)
    - after the last block: cache.advance(T_new), then ln_f and the tied head

    Prefill (the whole prompt at once) and decode (one token at a time) are
    the same call with different T_new."""
    raise NotImplementedError("forward_cached (TestForwardCached)")
