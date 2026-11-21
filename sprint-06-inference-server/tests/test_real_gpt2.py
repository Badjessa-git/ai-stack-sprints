"""The server with real GPT-2 behind it (uses Sprint 4's data/gpt2/)."""

import json
import os
import sys
import threading
import unittest
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deps import gpt2_dir  # noqa: E402

HAVE = os.path.exists(os.path.join(gpt2_dir(), "model.safetensors"))


@unittest.skipUnless(HAVE, "run make download in sprint-04-gpt2 first")
class TestRealServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.engine import GPT2Engine
        from src.server import make_server

        cls.server = make_server(GPT2Engine(), port=0)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_greedy_completion(self):
        body = json.dumps({"prompt": "Albert Einstein was a", "max_tokens": 4, "temperature": 0}).encode()
        req = urllib.request.Request(self.base + "/v1/completions", data=body)
        b = json.loads(urllib.request.urlopen(req, timeout=60).read())
        self.assertEqual(b["choices"][0]["text"], " great scientist, but")
        self.assertEqual(b["model"], "gpt2")
        self.assertEqual(b["usage"], {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8})


if __name__ == "__main__":
    unittest.main()
