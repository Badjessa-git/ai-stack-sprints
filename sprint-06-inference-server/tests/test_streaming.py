import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine import FakeEngine  # noqa: E402
from src.protocol import CompletionRequest  # noqa: E402
from src.streaming import StopBuffer, TextStream, run_completion  # noqa: E402


class TestTextStream(unittest.TestCase):
    def test_ascii_passes_straight_through(self):
        ts = TextStream()
        self.assertEqual(ts.push(b"ab"), "ab")

    def test_holds_half_a_character(self):
        ts = TextStream()
        emoji = "😀".encode()  # 4 bytes
        self.assertEqual(ts.push(emoji[:1]), "")
        self.assertEqual(ts.push(emoji[1:3]), "")
        self.assertEqual(ts.push(emoji[3:] + b"!"), "😀!")

    def test_flush_incomplete(self):
        ts = TextStream()
        ts.push("é".encode()[:1])
        self.assertEqual(ts.flush(), "�")


class TestStopBuffer(unittest.TestCase):
    def test_docstring_example(self):
        sb = StopBuffer(["END"])
        self.assertEqual(sb.push("abcE"), ("abc", False))
        self.assertEqual(sb.push("N"), ("", False))
        self.assertEqual(sb.push("D!"), ("", True))

    def test_false_alarm_is_released(self):
        sb = StopBuffer(["END"])
        self.assertEqual(sb.push("xEN"), ("x", False))
        self.assertEqual(sb.push("x"), ("ENx", False))

    def test_stop_inside_one_push(self):
        self.assertEqual(StopBuffer(["\n\n"]).push("one\n\ntwo"), ("one", True))

    def test_earliest_stop_wins(self):
        self.assertEqual(StopBuffer(["b", "a"]).push("xxabc"), ("xx", True))

    def test_no_stops(self):
        sb = StopBuffer([])
        self.assertEqual(sb.push("anything"), ("anything", False))
        self.assertEqual(sb.flush(), "")

    def test_flush_releases_held(self):
        sb = StopBuffer(["END"])
        sb.push("xE")
        self.assertEqual(sb.flush(), "E")


def run(engine, **kw):
    req = CompletionRequest(prompt="p", **kw)
    return list(run_completion(engine, req, engine.encode("p")))


class TestRunCompletion(unittest.TestCase):
    def test_whole_reply_then_stop(self):
        pieces = run(FakeEngine("Hi é"), max_tokens=100)
        self.assertEqual("".join(p for p, _, _ in pieces), "Hi é")
        self.assertEqual(pieces[-1][1], "stop")  # the eot token
        self.assertTrue(all(f is None for _, f, _ in pieces[:-1]))

    def test_no_half_characters_ever(self):
        pieces = run(FakeEngine("😀é😀"), max_tokens=100)
        self.assertTrue(all("�" not in p for p, _, _ in pieces))
        self.assertEqual("".join(p for p, _, _ in pieces), "😀é😀")

    def test_length(self):
        pieces = run(FakeEngine("Hello"), max_tokens=3)
        self.assertEqual("".join(p for p, _, _ in pieces), "Hel")
        self.assertEqual(pieces[-1][1:], ("length", 3))

    def test_length_mid_character_flushes(self):
        pieces = run(FakeEngine("aé"), max_tokens=2)  # 'a' + first byte of é
        self.assertEqual("".join(p for p, _, _ in pieces), "a�")

    def test_stop_string_across_tokens(self):
        pieces = run(FakeEngine("one\n\ntwo"), max_tokens=100, stop=["\n\n"])
        self.assertEqual("".join(p for p, _, _ in pieces), "one")
        self.assertEqual(pieces[-1][1], "stop")

    def test_token_count(self):
        pieces = run(FakeEngine("abc"), max_tokens=100)
        self.assertEqual(pieces[-1][2], 4)  # 3 bytes + the eot token


if __name__ == "__main__":
    unittest.main()
