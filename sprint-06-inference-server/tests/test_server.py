"""The real HTTP server on a free port, with FakeEngine, driven by urllib."""

import json
import os
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine import FakeEngine  # noqa: E402
from src.server import make_server  # noqa: E402

REPLY = "Hello! Café 😀. The end."


class ServerTest(unittest.TestCase):
    engine_args = {}

    @classmethod
    def setUpClass(cls):
        cls.server = make_server(FakeEngine(REPLY, **cls.engine_args), port=0)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def get(self, path):
        try:
            with urllib.request.urlopen(self.base + path, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def post(self, obj, raw=None):
        data = raw if raw is not None else json.dumps(obj).encode()
        req = urllib.request.Request(self.base + "/v1/completions", data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            r = urllib.request.urlopen(req, timeout=10)
            return r.status, r
        except urllib.error.HTTPError as e:
            return e.code, e

    def complete(self, **kw):
        status, r = self.post({"prompt": "Hi", **kw})
        return status, json.loads(r.read())

    def stream(self, **kw):
        status, r = self.post({"prompt": "Hi", "stream": True, **kw})
        self.assertEqual(status, 200)
        self.assertEqual(r.headers["Content-Type"], "text/event-stream")
        events = []
        for line in r:
            line = line.decode().rstrip("\n")
            if line.startswith("data: "):
                p = line[6:]
                events.append(p if p == "[DONE]" else json.loads(p))
        return events


class TestRoutes(ServerTest):
    def test_health(self):
        self.assertEqual(self.get("/health"), (200, {"status": "ok"}))

    def test_models(self):
        status, b = self.get("/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(b["data"][0]["id"], "fake")

    def test_404(self):
        status, b = self.get("/nope")
        self.assertEqual(status, 404)
        self.assertIn("error", b)


class TestCompletions(ServerTest):
    def test_basic(self):
        status, b = self.complete(max_tokens=100)
        self.assertEqual(status, 200)
        self.assertEqual(b["choices"][0]["text"], REPLY)
        self.assertEqual(b["choices"][0]["finish_reason"], "stop")
        self.assertTrue(b["id"].startswith("cmpl-"))
        self.assertEqual(b["usage"]["prompt_tokens"], 2)
        self.assertEqual(b["usage"]["completion_tokens"], len(REPLY.encode()) + 1)

    def test_max_tokens(self):
        _, b = self.complete(max_tokens=5)
        self.assertEqual(b["choices"][0]["text"], "Hello")
        self.assertEqual(b["choices"][0]["finish_reason"], "length")
        self.assertEqual(b["usage"]["completion_tokens"], 5)

    def test_stop(self):
        _, b = self.complete(max_tokens=100, stop=["😀"])
        self.assertEqual(b["choices"][0]["text"], "Hello! Café ")
        self.assertEqual(b["choices"][0]["finish_reason"], "stop")

    def test_bad_json_is_400(self):
        status, r = self.post(None, raw=b"{oops")
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(r.read())["error"]["type"], "invalid_request_error")

    def test_bad_field_is_400(self):
        status, b = self.complete(temperature=9)
        self.assertEqual(status, 400)
        self.assertIn("temperature", b["error"]["message"])

    def test_wrong_path_is_404(self):
        req = urllib.request.Request(self.base + "/v1/chat", data=b"{}")
        with self.assertRaises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(e.exception.code, 404)


class TestStreaming(ServerTest):
    def test_events_then_done(self):
        events = self.stream(max_tokens=100)
        self.assertEqual(events[-1], "[DONE]")
        chunks = events[:-1]
        self.assertGreater(len(chunks), 3, "stream token by token, not all at once")
        self.assertEqual("".join(c["choices"][0]["text"] for c in chunks), REPLY)
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        self.assertTrue(all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1]))
        self.assertEqual(len({c["id"] for c in chunks}), 1)

    def test_stream_equals_non_stream(self):
        events = self.stream(max_tokens=9, stop=["!"])
        _, b = self.complete(max_tokens=9, stop=["!"])
        self.assertEqual("".join(c["choices"][0]["text"] for c in events[:-1]), b["choices"][0]["text"])

    def test_no_replacement_chars(self):
        events = self.stream(max_tokens=100)
        self.assertTrue(all("�" not in c["choices"][0]["text"] for c in events[:-1]))


class TestSlowEngine(ServerTest):
    engine_args = {"delay": 0.01}

    def test_first_token_arrives_before_the_last(self):
        # With 10 ms per token, the first event must arrive well before
        # the whole ~0.3 s reply is done: that's what streaming is for.
        status, r = self.post({"prompt": "Hi", "stream": True, "max_tokens": 100})
        t0 = time.time()
        first = None
        for line in r:
            if line.startswith(b"data: ") and first is None:
                first = time.time() - t0
        total = time.time() - t0
        self.assertLess(first, total / 2)

    def test_concurrent_requests_all_succeed(self):
        results = []

        def one():
            results.append(self.complete(max_tokens=10))

        threads = [threading.Thread(target=one) for _ in range(4)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual([s for s, _ in results], [200] * 4)
        # one at a time under the lock: 4 x 10 tokens x 10 ms >= 0.4 s
        self.assertGreaterEqual(time.time() - t0, 0.35)


if __name__ == "__main__":
    unittest.main()
