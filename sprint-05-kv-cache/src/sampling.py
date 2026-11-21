"""Turning logits into a choice, and the generation loop.

Greedy decoding (always take the top token) is deterministic and dull: it
loops ("I'm a writer, and I'm a writer"). Sampling picks at random in
proportion to the model's probabilities, with three knobs:

- **temperature**: divide the logits by T before softmax. T < 1 sharpens
  (closer to greedy), T > 1 flattens (more surprising). T = 0 means greedy.
- **top-k**: only the k most likely tokens can be picked.
- **top-p** ("nucleus"): only the smallest set of top tokens whose
  probabilities add up to at least p. It adapts: when the model is sure,
  that set is 1-2 tokens; when it's unsure, it's hundreds.

Order of work (tests/test_sampling.py):
    1. apply_temperature, top_k_filter, top_p_filter     TestFilters
    2. sample                                            TestSample
    3. generate                                          TestGenerate
"""

import numpy as np

from .kv_cache import KVCache, forward_cached


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def apply_temperature(logits, temperature):
    """logits / temperature, as float64. (temperature > 0; 0 is handled in
    sample() as greedy.)"""
    raise NotImplementedError("apply_temperature (TestFilters)")


def top_k_filter(logits, k):
    """Return a copy with everything except the k largest logits set to
    -inf. k >= len(logits) changes nothing. (np.argpartition finds the top k
    without a full sort.)"""
    raise NotImplementedError("top_k_filter (TestFilters)")


def top_p_filter(logits, p):
    """Return a copy keeping the smallest set of most-likely tokens whose
    probabilities sum to >= p; the rest become -inf. Always keep at least
    the top token.

    Sort probabilities high to low, take the running total (np.cumsum), and
    keep every token up to and including the first one where the total
    reaches p."""
    raise NotImplementedError("top_p_filter (TestFilters)")


def sample(logits, temperature=1.0, top_k=None, top_p=None, rng=None):
    """Pick one token id from a (V,) logits vector.

    temperature == 0: return the argmax (greedy), ignoring k and p.
    Otherwise: temperature, then top-k (if given), then top-p (if given),
    then softmax, then rng.choice over the ids with those probabilities.
    rng is a np.random.Generator; default np.random.default_rng()."""
    raise NotImplementedError("sample (TestSample)")


def generate(model, ids, max_new, temperature=1.0, top_k=None, top_p=None,
             seed=None, stop_ids=(), cache=None):
    """Yield new token ids one at a time (a generator, so a server can
    stream them as they're made).

    1. prefill: forward_cached on the whole prompt; sample from the LAST row
    2. decode: forward_cached on just the token you picked; sample again
    3. stop after max_new tokens, after yielding a token in stop_ids
       (e.g. GPT-2's <|endoftext|>), or when the cache is full.

    Use one rng = np.random.default_rng(seed) for the whole call, so the
    same seed gives the same text. Make a cache with KVCache.for_model if
    none is passed."""
    raise NotImplementedError("generate (TestGenerate)")
