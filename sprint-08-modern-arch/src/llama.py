"""A modern small LLM: the Llama architecture, running SmolLM-135M.

GPT-2 is from 2019. Nearly every open model since (Llama, Mistral, Qwen,
SmolLM, Gemma) is the same transformer with four parts swapped out. Each
swap is small, and each has a reason:

| GPT-2 (Sprint 4)            | Llama family (this file)        | Why                                  |
|-----------------------------|---------------------------------|--------------------------------------|
| LayerNorm (mean + variance) | RMSNorm (no mean, no bias)      | cheaper, works as well               |
| learned position table wpe  | RoPE: rotate q and k by angle   | relative positions, longer contexts  |
| GELU MLP, 4x wide           | SwiGLU: silu(gate) * up, ~2.7x  | better quality per parameter         |
| every head has its own k, v | GQA: heads share k, v groups    | 3x smaller KV cache here             |

And one layout change: Hugging Face's Llama stores Linear weights the
PyTorch way, (out_features, in_features), so it's x @ W.T. GPT-2's
Conv1D was (in, out), x @ W. Same math, the other way round.

SmolLM-135M: 30 layers, 576 wide, 9 query heads sharing 3 key/value
heads (head size 64), MLP width 1536, vocab 49,152, tied embeddings.

Order of work (tests/test_llama.py):
    1. rms_norm, silu, swiglu                     TestOps
    2. rope_tables, apply_rope                    TestRope
    3. repeat_kv, attention (GQA, with cache)     TestAttention
    4. map_hf_names                               TestLoading
    5. Llama.forward / forward_cached             TestModel
Then tests/test_real_smollm.py.
"""

import json
import math
import os
from dataclasses import dataclass

import numpy as np

from .deps import sprint

_s4 = sprint("sprint-04-gpt2")
load_safetensors = _s4.safetensors.load
softmax = _s4.gpt2.softmax  # yours, from Sprint 4


@dataclass
class LlamaConfig:
    vocab_size: int = 49152
    dim: int = 576
    n_layer: int = 30
    n_head: int = 9
    n_kv_head: int = 3
    hidden: int = 1536
    max_pos: int = 2048
    eps: float = 1e-5
    rope_theta: float = 10000.0
    tied: bool = True

    @property
    def head_dim(self):
        return self.dim // self.n_head

    @classmethod
    def from_json(cls, path):
        c = json.load(open(path))
        return cls(
            vocab_size=c["vocab_size"], dim=c["hidden_size"], n_layer=c["num_hidden_layers"],
            n_head=c["num_attention_heads"],
            n_kv_head=c.get("num_key_value_heads", c["num_attention_heads"]),
            hidden=c["intermediate_size"], max_pos=c.get("max_position_embeddings", 2048),
            eps=c.get("rms_norm_eps", 1e-5), rope_theta=c.get("rope_theta", 10000.0),
            tied=c.get("tie_word_embeddings", False),
        )


# ---- ops --------------------------------------------------------------------


def rms_norm(x, w, eps=1e-5):
    """x / sqrt(mean(x**2 over the last axis) + eps) * w. No mean is
    subtracted and there is no bias: that's the whole difference from
    LayerNorm."""
    raise NotImplementedError("rms_norm (TestOps)")


def silu(x):
    """x * sigmoid(x), also called swish."""
    raise NotImplementedError("silu (TestOps)")


def swiglu(x, w_gate, w_up, w_down):
    """The Llama MLP. Weights are PyTorch layout (out, in):
        (silu(x @ w_gate.T) * (x @ w_up.T)) @ w_down.T
    The gate decides, per hidden unit, how much of `up` gets through."""
    raise NotImplementedError("swiglu (TestOps)")


def rope_tables(head_dim, max_pos, theta=10000.0):
    """Return (cos, sin), each (max_pos, head_dim), float32.

    Frequencies: inv_freq[i] = 1 / theta ** (2i / head_dim) for i in
    0 .. head_dim/2 - 1. Angle for position p: p * inv_freq, shape
    (max_pos, head_dim/2). Then concatenate the angles WITH THEMSELVES
    along the last axis to get (max_pos, head_dim): this is Hugging Face's
    "rotate half" layout, where dimension i pairs with i + head_dim/2
    (not with i + 1 as in the original RoPE paper). The checkpoint's q/k
    weights were permuted to match it, so using the paper's layout breaks
    the model."""
    raise NotImplementedError("rope_tables (TestRope)")


def apply_rope(x, cos, sin, positions):
    """Rotate x (H, T, D) by the angles of `positions` (T ints).

        rotate_half(x) = concat(-x[..., D/2:], x[..., :D/2])
        out = x * cos[positions] + rotate_half(x) * sin[positions]

    Each pair (x_i, x_{i+D/2}) is a 2-D point turned by its angle. The dot
    product of a rotated q at position m with a rotated k at position n
    then depends only on m - n: attention sees relative position for free."""
    raise NotImplementedError("apply_rope (TestRope)")


def repeat_kv(x, n_rep):
    """(KVH, T, D) -> (KVH * n_rep, T, D): each k/v head serves n_rep
    query heads in a row. Query head h uses kv head h // n_rep, so repeat
    each head n_rep times in place (np.repeat on axis 0, not np.tile)."""
    raise NotImplementedError("repeat_kv (TestAttention)")


def attention(x, p, i, cfg, cos, sin, cache=None):
    """Grouped-query causal attention for block i. x: (T, dim) -> (T, dim).

    1. q = x @ wq.T -> (T, H, D) -> (H, T, D); k, v the same with KVH heads
    2. positions = start .. start + T - 1 (start = cache.length, or 0)
    3. RoPE on q and k (not v) at those positions
    4. with a cache (Sprint 5's KVCache, built with n_head = n_kv_head):
       k, v = cache.update(i, k, v). The cache stores the KVH heads only,
       that's where GQA saves memory.
    5. repeat_kv k and v up to H heads; scores / sqrt(D); causal mask with
       Sprint 5's offset; softmax; @ v; merge heads; @ wo.T

    p[f"layers.{i}.wq"] etc. (see expected_shapes). Keep float32."""
    raise NotImplementedError("attention (TestAttention)")


# ---- weights ------------------------------------------------------------------


def expected_shapes(cfg):
    s = {"embed": (cfg.vocab_size, cfg.dim), "norm": (cfg.dim,)}
    kv = cfg.n_kv_head * cfg.head_dim
    for i in range(cfg.n_layer):
        L = f"layers.{i}."
        s.update({
            L + "attn_norm": (cfg.dim,), L + "mlp_norm": (cfg.dim,),
            L + "wq": (cfg.dim, cfg.dim), L + "wk": (kv, cfg.dim), L + "wv": (kv, cfg.dim),
            L + "wo": (cfg.dim, cfg.dim),
            L + "w_gate": (cfg.hidden, cfg.dim), L + "w_up": (cfg.hidden, cfg.dim),
            L + "w_down": (cfg.dim, cfg.hidden),
        })
    if not cfg.tied:
        s["lm_head"] = (cfg.vocab_size, cfg.dim)
    return s


# Hugging Face name (after "model.") -> this file's name
HF_NAMES = {
    "embed_tokens.weight": "embed",
    "norm.weight": "norm",
    "input_layernorm.weight": "attn_norm",
    "post_attention_layernorm.weight": "mlp_norm",
    "self_attn.q_proj.weight": "wq",
    "self_attn.k_proj.weight": "wk",
    "self_attn.v_proj.weight": "wv",
    "self_attn.o_proj.weight": "wo",
    "mlp.gate_proj.weight": "w_gate",
    "mlp.up_proj.weight": "w_up",
    "mlp.down_proj.weight": "w_down",
}


def map_hf_names(tensors, cfg):
    """Hugging Face Llama names -> this file's names, as strict as Sprint 4:

        model.embed_tokens.weight                 -> embed
        model.norm.weight                         -> norm
        model.layers.{i}.self_attn.q_proj.weight  -> layers.{i}.wq   (HF_NAMES)
        lm_head.weight                            -> lm_head, or dropped if cfg.tied
        anything ending in rotary_emb.inv_freq    -> dropped (a cached table)

    Missing -> KeyError, wrong shape or unknown name -> ValueError.
    Return float32 arrays."""
    raise NotImplementedError("map_hf_names (TestLoading)")


def random_params(cfg, seed=0, scale=0.1):
    rng = np.random.default_rng(seed)
    return {k: (np.ones(s, np.float32) if k.endswith("norm") else
                (rng.standard_normal(s) * scale).astype(np.float32))
            for k, s in expected_shapes(cfg).items()}


# ---- the model ----------------------------------------------------------------


class Llama:
    def __init__(self, cfg, params):
        self.cfg, self.p = cfg, params
        self.cos, self.sin = rope_tables(cfg.head_dim, cfg.max_pos, cfg.rope_theta)

    @classmethod
    def from_dir(cls, path):
        cfg = LlamaConfig.from_json(os.path.join(path, "config.json"))
        return cls(cfg, map_hf_names(load_safetensors(os.path.join(path, "model.safetensors")), cfg))

    def new_cache(self, max_len=None):
        """Sprint 5's KVCache, sized for the KV heads only."""
        c = self.cfg
        return sprint("sprint-05-kv-cache").kv_cache.KVCache(
            c.n_layer, c.n_kv_head, c.head_dim, max_len or c.max_pos)

    def forward(self, ids, cache=None):
        """ids -> logits (T, vocab). With a cache: only the new ids, like
        Sprint 5's forward_cached (advance the cache after the last layer).

        x = embed[ids]  (no position table: RoPE does positions)
        each block:  x = x + attention(rms_norm(x, attn_norm))
                     x = x + swiglu(rms_norm(x, mlp_norm))
        then rms_norm(x, norm), and logits = x @ embed.T (tied) or
        x @ lm_head.T."""
        raise NotImplementedError("forward (TestModel)")

    def generate(self, ids, n_new, temperature=0.0, top_p=None, seed=None, stop_ids=()):
        """Scaffold: prefill + cached decode with Sprint 5's sample()."""
        sample = sprint("sprint-05-kv-cache").sampling.sample
        rng = np.random.default_rng(seed)
        cache = self.new_cache(len(ids) + n_new)
        logits = self.forward(ids, cache)[-1]
        out = []
        for _ in range(n_new):
            t = sample(logits, temperature, None, top_p, rng)
            out.append(t)
            if t in stop_ids:
                break
            logits = self.forward([t], cache)[-1]
        return out
