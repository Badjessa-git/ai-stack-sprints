"""GPT-2, forward pass only, in plain NumPy.

This is your Sprint 3 transformer, with three changes:

- **No autograd.** Serving a model never calls backward(), so every
  function here takes arrays and returns arrays. No Tensor, no graph.
  It's simpler and much faster than the training engine.
- **GPT-2's exact shapes and names.** Weights come from OpenAI's trained
  checkpoint, so the code has to match how they were stored: one fused
  (C, 3C) matrix for q, k and v; Conv1D layout (multiply as x @ W, no
  transpose); and the output head reuses the token embedding (tied).
- **Real scale.** 124M parameters, 12 layers, 12 heads, 768 wide, a
  1,024-token context. Same block, repeated 12 times.

Shapes below: T = sequence length, C = n_embd, H = n_head, D = C // H.
Everything works on ONE sequence, x of shape (T, C). Batching comes in
Sprint 7.

Order of work (tests/test_gpt2.py):
    1. layer_norm, gelu, softmax          TestOps
    2. attention                          TestAttention
    3. mlp, block                         TestBlock
    4. map_hf_names                       TestLoading
    5. GPT2.forward                       TestModel
Then tests/test_real_gpt2.py, once data/gpt2/ is downloaded.
"""

import json
import math
import os
from dataclasses import dataclass

import numpy as np

from .safetensors import load as load_safetensors


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_ctx: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    eps: float = 1e-5

    @classmethod
    def from_json(cls, path):
        c = json.load(open(path))
        return cls(
            vocab_size=c["vocab_size"],
            n_ctx=c.get("n_positions", c.get("n_ctx", 1024)),
            n_embd=c["n_embd"],
            n_head=c["n_head"],
            n_layer=c["n_layer"],
            eps=c.get("layer_norm_epsilon", 1e-5),
        )


# ---- the ops --------------------------------------------------------------


def layer_norm(x, w, b, eps=1e-5):
    """Normalize each row of x to mean 0, variance 1 over the last axis, then
    scale by w and shift by b. Variance is the plain mean of squared
    deviations (np.var's default), and eps goes inside the square root."""
    raise NotImplementedError("layer_norm (TestOps)")


def gelu(x):
    """GPT-2's GELU is the tanh approximation:
        0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))
    Compute the constant with math, not np (see attention's note on
    float64 scalars): float32 in, float32 out."""
    raise NotImplementedError("gelu (TestOps)")


def softmax(x, axis=-1):
    """Stable softmax: subtract the max along `axis` before exp."""
    raise NotImplementedError("softmax (TestOps)")


def attention(x, w_qkv, b_qkv, w_out, b_out, n_head):
    """Causal multi-head self-attention. x: (T, C) -> (T, C).

    1. qkv = x @ w_qkv + b_qkv                 (T, 3C)
    2. q, k, v = split qkv into three (T, C)   (q first, then k, then v;
       take C from qkv's width // 3, not from x, so a test can pass a
       narrower w_qkv to check one head on its own)
    3. split heads: (T, C) -> (T, H, D) -> (H, T, D)
    4. scores = q @ kᵀ / sqrt(D)               (H, T, T)
       Use math.sqrt(D), not np.sqrt(D). np.sqrt returns a NumPy float64
       scalar, and NumPy 2 lets that promote float32 arrays to float64.
       Everything downstream then runs in float64 and every weight
       matrix gets converted on every call: about 10x slower, with
       identical output. (TestModel.test_stays_float32 checks.)
    5. causal mask: position t may not see t' > t (set those to -inf,
       or a big negative number, before softmax)
    6. out = softmax(scores) @ v               (H, T, D)
    7. merge heads: (H, T, D) -> (T, H, D) -> (T, C)
    8. return out @ w_out + b_out

    Sprint 3 did this with one module per head. Here all heads are one
    reshape: same math, one matrix multiply instead of H."""
    raise NotImplementedError("attention (TestAttention)")


def mlp(x, w_fc, b_fc, w_proj, b_proj):
    """Feed-forward: (T, C) -> (T, 4C) -> gelu -> (T, C)."""
    raise NotImplementedError("mlp (TestBlock)")


def block(x, p, i, n_head, eps=1e-5):
    """Transformer block i, pre-norm, two residual adds (Sprint 3 Step 4):
        x = x + attention(layer_norm(x, ln_1))
        x = x + mlp(layer_norm(x, ln_2))
    p is the params dict: block i's weights are p[f"h.{i}.ln_1.weight"],
    p[f"h.{i}.attn.c_attn.weight"], ... (see expected_shapes)."""
    raise NotImplementedError("block (TestBlock)")


# ---- weights --------------------------------------------------------------


def expected_shapes(cfg):
    """Every parameter GPT-2 needs, by name, with its shape."""
    C, V = cfg.n_embd, cfg.vocab_size
    shapes = {"wte.weight": (V, C), "wpe.weight": (cfg.n_ctx, C),
              "ln_f.weight": (C,), "ln_f.bias": (C,)}
    for i in range(cfg.n_layer):
        h = f"h.{i}."
        shapes.update({
            h + "ln_1.weight": (C,), h + "ln_1.bias": (C,),
            h + "attn.c_attn.weight": (C, 3 * C), h + "attn.c_attn.bias": (3 * C,),
            h + "attn.c_proj.weight": (C, C), h + "attn.c_proj.bias": (C,),
            h + "ln_2.weight": (C,), h + "ln_2.bias": (C,),
            h + "mlp.c_fc.weight": (C, 4 * C), h + "mlp.c_fc.bias": (4 * C,),
            h + "mlp.c_proj.weight": (4 * C, C), h + "mlp.c_proj.bias": (C,),
        })
    return shapes


def map_hf_names(tensors, cfg):
    """Turn the raw tensors from Hugging Face's model.safetensors into the
    params dict this file uses. Return {name: float32 array}.

    What the checkpoint actually contains (print the names and look):
    - names may carry a "transformer." prefix (some exports) or not: strip it
    - per-layer "h.{i}.attn.bias" and "h.{i}.attn.masked_bias" are NOT
      weights: they're the stored causal mask. Drop them.
    - "lm_head.weight", if present, is the same matrix as wte.weight (tied
      embeddings): drop it too; forward() reuses wte.

    Then check against expected_shapes(cfg): a missing name raises KeyError
    naming it; a wrong shape raises ValueError naming the tensor and both
    shapes; an extra, unknown name raises ValueError too. Loading a
    checkpoint silently wrong is the worst possible failure, so be strict."""
    raise NotImplementedError("map_hf_names (TestLoading)")


def random_params(cfg, seed=0, scale=0.02):
    """Small random weights with the right names and shapes (for tests)."""
    rng = np.random.default_rng(seed)
    p = {}
    for name, shape in expected_shapes(cfg).items():
        if name.endswith(("ln_1.weight", "ln_2.weight", "ln_f.weight")):
            p[name] = np.ones(shape, np.float32)
        elif name.endswith(".bias") and "ln" in name:
            p[name] = np.zeros(shape, np.float32)
        else:
            p[name] = (rng.standard_normal(shape) * scale).astype(np.float32)
    return p


# ---- the model ------------------------------------------------------------


class GPT2:
    def __init__(self, cfg, params):
        self.cfg = cfg
        self.p = params

    @classmethod
    def from_dir(cls, path):
        """Load data/gpt2/ (config.json + model.safetensors)."""
        cfg = GPT2Config.from_json(os.path.join(path, "config.json"))
        tensors = load_safetensors(os.path.join(path, "model.safetensors"))
        return cls(cfg, map_hf_names(tensors, cfg))

    def forward(self, ids):
        """ids: list/array of T token ids (T <= n_ctx) -> logits (T, V).

        x = wte[ids] + wpe[0..T-1]; every block in order; final layer_norm
        (ln_f); logits = x @ wteᵀ (the tied head: the same matrix that turned
        ids into vectors turns vectors back into a score per token)."""
        raise NotImplementedError("forward (TestModel)")

    def generate_greedy(self, ids, n_new):
        """Scaffold: the slow way, a full forward pass per new token.
        Sprint 5 replaces this with a KV cache."""
        ids = list(ids)
        for _ in range(n_new):
            window = ids[-self.cfg.n_ctx:]
            ids.append(int(np.argmax(self.forward(window)[-1])))
        return ids
