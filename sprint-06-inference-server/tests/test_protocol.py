import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.protocol import BadRequest, chunk_body, completion_body, parse_request, sse  # noqa: E402


def body(**kw):
    return json.dumps(kw).encode()


class TestParseRequest(unittest.TestCase):
    def test_defaults(self):
        r = parse_request(body(prompt="hi"))
        self.assertEqual((r.prompt, r.max_tokens, r.temperature, r.top_p, r.top_k, r.stream, r.stop, r.seed),
                         ("hi", 16, 1.0, 1.0, None, False, [], None))

    def test_all_fields(self):
        r = parse_request(body(prompt="x", max_tokens=5, temperature=0, top_p=0.5, top_k=3,
                               stream=True, stop=["\n", "END"], seed=7, user="ignored", n=1))
        self.assertEqual((r.max_tokens, r.temperature, r.top_p, r.top_k, r.stream, r.stop, r.seed),
                         (5, 0.0, 0.5, 3, True, ["\n", "END"], 7))

    def test_stop_string_becomes_list(self):
        self.assertEqual(parse_request(body(prompt="x", stop="\n\n")).stop, ["\n\n"])
        self.assertEqual(parse_request(body(prompt="x", stop=None)).stop, [])

    def test_bad_requests(self):
        bad = [
            b"not json", b"[1, 2]", body(), body(prompt=5),
            body(prompt="x", max_tokens=0), body(prompt="x", max_tokens=2.5),
            body(prompt="x", max_tokens=True),
            body(prompt="x", temperature=-0.1), body(prompt="x", temperature=2.5),
            body(prompt="x", temperature="hot"),
            body(prompt="x", top_p=0), body(prompt="x", top_p=1.5),
            body(prompt="x", top_k=0), body(prompt="x", stream="yes"),
            body(prompt="x", stop=[""]), body(prompt="x", stop=["a", "b", "c", "d", "e"]),
            body(prompt="x", stop=5), body(prompt="x", seed="one"),
        ]
        for b in bad:
            with self.assertRaises(BadRequest, msg=b):
                parse_request(b)

    def test_message_names_the_field(self):
        with self.assertRaises(BadRequest) as e:
            parse_request(body(prompt="x", top_p=3))
        self.assertIn("top_p", str(e.exception))


class TestBodies(unittest.TestCase):
    def test_completion_body(self):
        b = completion_body("cmpl-1", 100, "gpt2", " hi", "length", 4, 2)
        self.assertEqual(b["object"], "text_completion")
        self.assertEqual(b["choices"], [{"text": " hi", "index": 0, "logprobs": None, "finish_reason": "length"}])
        self.assertEqual(b["usage"], {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6})
        self.assertEqual((b["id"], b["created"], b["model"]), ("cmpl-1", 100, "gpt2"))

    def test_chunk_body(self):
        b = chunk_body("cmpl-1", 100, "gpt2", "x")
        self.assertNotIn("usage", b)
        self.assertIsNone(b["choices"][0]["finish_reason"])

    def test_sse(self):
        self.assertEqual(sse("[DONE]"), b"data: [DONE]\n\n")
        e = sse({"a": "é"})
        self.assertTrue(e.startswith(b"data: ") and e.endswith(b"\n\n"))
        self.assertEqual(json.loads(e[6:-2]), {"a": "é"})


if __name__ == "__main__":
    unittest.main()
