"""SmolLM behind your Sprint 6 server (scaffold: nothing to write here).

Same five-part engine interface as Sprint 6's GPT2Engine. Generation is
prefill + cached decode with your Llama.forward and Sprint 5's sample().
"""

import os

import numpy as np

from .deps import smollm_dir, sprint
from .hf_tokenizer import HFTokenizer
from .llama import Llama


class SmolLMEngine:
    name = "smollm-135m"

    def __init__(self, path=None):
        d = path or smollm_dir()
        self.tok = HFTokenizer.from_tokenizer_json(os.path.join(d, "tokenizer.json"))
        self.model = Llama.from_dir(d)
        self.eot_id = self.tok.eot_id
        self._sample = sprint("sprint-05-kv-cache").sampling.sample

    def encode(self, text):
        return self.tok.encode(text)

    def token_bytes(self, i):
        return self.tok.token_bytes(i)

    def generate(self, ids, req):
        room = self.model.cfg.max_pos - len(ids)
        n = min(req.max_tokens, room)
        if n <= 0 or not ids:
            return
        rng = np.random.default_rng(req.seed)
        cache = self.model.new_cache(len(ids) + n)
        logits = self.model.forward(ids, cache)[-1]
        top_p = req.top_p if req.top_p < 1.0 else None
        for k in range(n):
            t = self._sample(logits, req.temperature, req.top_k, top_p, rng)
            yield t
            if t == self.eot_id or k == n - 1:
                return
            logits = self.model.forward([t], cache)[-1]
