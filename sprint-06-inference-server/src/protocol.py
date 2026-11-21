"""The wire format: OpenAI's /v1/completions, the de-facto standard.

Any OpenAI-compatible client (the `openai` Python package, LangChain,
curl scripts) can talk to a server that speaks this. A request:

    POST /v1/completions
    {"prompt": "Once upon a time", "max_tokens": 40, "temperature": 0.8,
     "top_p": 0.9, "stream": false, "stop": ["\\n\\n"], "seed": 1}

A response (stream: false):

    {"id": "cmpl-...", "object": "text_completion", "created": 1732200000,
     "model": "gpt2",
     "choices": [{"text": " there was...", "index": 0,
                  "logprobs": null, "finish_reason": "length"}],
     "usage": {"prompt_tokens": 4, "completion_tokens": 40, "total_tokens": 44}}

finish_reason is "length" (hit max_tokens) or "stop" (a stop string, or
the model's end-of-text token).

Streaming (stream: true) sends Server-Sent Events instead: one
`data: <json>\\n\\n` line per piece of text, the same shape as above but
"object": "text_completion" with only the new text in choices[0].text,
then a final `data: [DONE]\\n\\n`.

Order of work: parse_request, completion_body, sse (tests/test_protocol.py)
"""

import json
from dataclasses import dataclass, field


class BadRequest(ValueError):
    """Anything wrong with the client's request: becomes an HTTP 400."""


@dataclass
class CompletionRequest:
    prompt: str
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = None
    stream: bool = False
    stop: list = field(default_factory=list)
    seed: int = None


def parse_request(body):
    """bytes -> CompletionRequest. Raise BadRequest (with a message a human
    can act on) for:

    - body that isn't valid JSON, or isn't a JSON object
    - missing "prompt", or a prompt that isn't a string
    - max_tokens not an int >= 1          (bool is an int in Python: reject it)
    - temperature not a number in [0, 2]
    - top_p not a number in (0, 1]
    - top_k not null or an int >= 1
    - stream not a bool
    - stop not null, a string, or a list of up to 4 non-empty strings
      (a single string becomes a one-item list; null becomes [])
    - seed not null or an int

    Unknown fields are ignored (clients send extras like "n" or "user")."""
    raise NotImplementedError("parse_request (TestParseRequest)")


def completion_body(cid, created, model, text, finish_reason, prompt_tokens, completion_tokens):
    """The non-streaming response dict (see the module docstring)."""
    raise NotImplementedError("completion_body (TestBodies)")


def chunk_body(cid, created, model, text, finish_reason=None):
    """One streaming event's dict: same shape, no "usage", only new text,
    finish_reason null until the last chunk."""
    raise NotImplementedError("chunk_body (TestBodies)")


def sse(data):
    """One Server-Sent Event as bytes: b"data: " + payload + b"\\n\\n".
    payload is json.dumps(data) for a dict, or the string itself
    (for "[DONE]"). Encode as UTF-8."""
    raise NotImplementedError("sse (TestBodies)")


def error_body(message, kind="invalid_request_error"):
    return {"error": {"message": message, "type": kind}}
