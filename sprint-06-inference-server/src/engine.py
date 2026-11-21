"""What the server talks to. (Scaffold: nothing to write here.)

An engine has five things:
    name                         model name, echoed in responses
    eot_id                       end-of-text token id
    encode(text) -> [ids]
    token_bytes(id) -> bytes     one token's raw bytes
    generate(ids, req)           yields new token ids, one at a time

FakeEngine replies with a fixed text, one byte per token, so the tests
exercise UTF-8 splitting and stop strings without loading a model.
GPT2Engine wires in your Sprint 4 tokenizer and Sprint 5 generate().
"""

import os
import time

from .deps import gpt2_dir, sprint


class FakeEngine:
    name = "fake"
    eot_id = 256

    def __init__(self, reply="Hello! Café 😀. The end.", delay=0.0):
        self.reply = reply.encode("utf-8")
        self.delay = delay  # seconds per token, to simulate a slow model

    def encode(self, text):
        return list(text.encode("utf-8"))

    def token_bytes(self, i):
        return bytes([i])

    def generate(self, ids, req):
        for i, b in enumerate(list(self.reply) + [self.eot_id]):
            if i >= req.max_tokens:
                return
            if self.delay:
                time.sleep(self.delay)
            yield b


class GPT2Engine:
    name = "gpt2"

    def __init__(self, path=None):
        s4, s5 = sprint("sprint-04-gpt2"), sprint("sprint-05-kv-cache")
        d = path or gpt2_dir()
        self.tok = s4.gpt2_tokenizer.GPT2Tokenizer.from_files(
            os.path.join(d, "vocab.json"), os.path.join(d, "merges.txt")
        )
        self.model = s4.gpt2.GPT2.from_dir(d)
        self.eot_id = self.tok.eot_id
        self._generate = s5.sampling.generate

    def encode(self, text):
        return self.tok.encode(text)

    def token_bytes(self, i):
        return self.tok.token_bytes(i)

    def generate(self, ids, req):
        room = self.model.cfg.n_ctx - len(ids)
        if room <= 0:
            return iter(())
        return self._generate(
            self.model, ids, min(req.max_tokens, room), req.temperature,
            req.top_k, req.top_p if req.top_p < 1.0 else None, req.seed,
            stop_ids={self.eot_id},
        )
