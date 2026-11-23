"""The batched server with real GPT-2 (uses Sprint 4's data/gpt2/)."""

import json
import os
import sys
import threading
import unittest
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from src.deps import gpt2_dir  # noqa: E402

HAVE = os.path.exists(os.path.join(gpt2_dir(), "model.safetensors"))
PROMPTS = ["Albert Einstein was a", "The capital of France is", "Once upon a time", "Hello, my name is"]


@unittest.skipUnless(HAVE, "run make download in sprint-04-gpt2 first")
class TestRealBatchedServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from serve_batched import load_engine, make_batched_server

        cls.engine = load_engine(max_batch=4)
        cls.server = make_batched_server(cls.engine, port=0)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.engine.close()

    def complete(self, prompt):
        body = json.dumps({"prompt": prompt, "max_tokens": 8, "temperature": 0}).encode()
        r = urllib.request.urlopen(urllib.request.Request(self.base + "/v1/completions", data=body), timeout=120)
        return json.loads(r.read())["choices"][0]["text"]

    def test_concurrent_equals_alone(self):
        alone = [self.complete(p) for p in PROMPTS]
        together = [None] * len(PROMPTS)

        def one(i):
            together[i] = self.complete(PROMPTS[i])

        ts = [threading.Thread(target=one, args=(i,)) for i in range(len(PROMPTS))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertEqual(together, alone)
        self.assertTrue(alone[0].startswith(" great scientist"))
        self.assertGreater(self.engine.stats["max_running"], 1, "requests weren't batched")


if __name__ == "__main__":
    unittest.main()
