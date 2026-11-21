"""Turning a stream of tokens into a stream of text. Two problems the
non-streaming version never had:

1. **Half characters.** A token is bytes, and one UTF-8 character can be
   split across two tokens ('é' is 2 bytes, '😀' is 4). Decoding each
   token on its own prints '�' garbage. TextStream holds back an
   incomplete character until its last byte arrives.

2. **Stop strings that span tokens.** With stop=["\\n\\n"], the model may
   produce "\\n" in one token and "\\n" in the next. By the time you know
   it's a stop, you've already sent the first "\\n" to the client. So
   StopBuffer holds back any tail of the text that could still turn into
   a stop string, and only releases it once it can't.

Then run_completion ties it together: tokens in, (text, finish_reason)
pieces out. Both the streaming and non-streaming endpoints use it.

Order of work: TextStream, StopBuffer, run_completion (tests/test_streaming.py)
"""

import codecs


class TextStream:
    def __init__(self):
        """Hint: codecs.getincrementaldecoder("utf-8")(errors="replace")
        does exactly the holding back. Read its docs, then decide whether
        to use it or write the byte logic yourself (the leading byte tells
        you how many bytes the character has)."""
        raise NotImplementedError("TextStream.__init__ (TestTextStream)")

    def push(self, data):
        """Add a token's bytes; return the text that is now complete
        (possibly "")."""
        raise NotImplementedError("TextStream.push (TestTextStream)")

    def flush(self):
        """End of stream: return whatever is left (an incomplete character
        becomes '�')."""
        raise NotImplementedError("TextStream.flush (TestTextStream)")


class StopBuffer:
    def __init__(self, stops):
        self.stops = list(stops)
        self.held = ""
        self.stopped = False

    def push(self, text):
        """Add text; return (safe_text, stopped).

        - If held + text contains a stop string: return everything before
          the EARLIEST stop occurrence, and stopped=True. The stop string
          itself is never returned (OpenAI's behaviour).
        - Otherwise: return everything except the longest tail that is a
          prefix of some stop string (keep that tail in self.held), and
          stopped=False.

        With stops=["END"]: push("abcE") -> ("abc", False), holding "E";
        push("N") -> ("", False), holding "EN"; push("D!") -> ("", True)."""
        raise NotImplementedError("StopBuffer.push (TestStopBuffer)")

    def flush(self):
        """End of stream without a stop: release what's held."""
        out, self.held = self.held, ""
        return out


def run_completion(engine, req, prompt_ids):
    """Yield (text, finish_reason, n_tokens) pieces for one request.

    For each token from engine.generate(prompt_ids, req):
      - count it (n_tokens is the running total of tokens generated)
      - if it's engine.eot_id: stop, finish_reason "stop" (don't decode it)
      - otherwise push its bytes (engine.token_bytes) through a TextStream,
        then the text through a StopBuffer(req.stop); yield non-empty text
        with finish_reason None; if the StopBuffer says stopped, yield the
        final piece with finish_reason "stop" and end.
    If the tokens run out (max_tokens reached) without a stop: flush the
    TextStream and then the StopBuffer, and yield the remainder with
    finish_reason "length".

    The LAST thing yielded always has a finish_reason, even if its text is
    "" (the client needs to know why it ended)."""
    raise NotImplementedError("run_completion (TestRunCompletion)")
