# Sprint 06: An Inference Server

## Why this sprint

After Sprint 5 you have a model that writes at about 50 tokens a second,
but only from a Python prompt. This sprint puts it behind HTTP, speaking
the same API as OpenAI's `/v1/completions`, so anything that can call
OpenAI can call your server: `curl`, the `openai` Python package, or the
RAG sprint later in the series.

```bash
curl -N localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt": "Once upon a time", "max_tokens": 60, "stream": true}'
```

It's built on the standard library's `http.server`, with no framework, so
you see every byte of the protocol. The interesting parts aren't the HTTP
itself. They're the problems streaming creates:

- **Streaming** (Server-Sent Events): text has to reach the client as
  it's generated, so the response has no length up front, and every
  piece must be flushed the moment it exists.
- **Half characters:** a token is bytes, and '😀' is 4 of them. They can
  arrive in separate tokens, and decoding one token at a time prints
  garbage.
- **Stop strings that span tokens:** with `stop: ["\n\n"]` the first
  "\n" might be its own token. If you send it before you know a second
  one is coming, you can't take it back.
- **Validation:** every field a client can send is checked, and errors
  come back as an OpenAI-style 400, never a crashed thread.

The server runs one request at a time, on purpose. The load test shows
what that costs, and that's Sprint 7's problem.

Learning goals:

1. Explain what an HTTP response actually is, and how Server-Sent Events
   stream one without knowing its length.
2. Handle UTF-8 and stop strings correctly across token boundaries.
3. Explain time to first token (TTFT) vs total latency, and measure both
   under load.
4. Know exactly what "OpenAI-compatible" means: which fields, which
   shapes, which errors.

## The plan

**Sprints 4 and 5 must pass first** for the GPT-2 engine. All tests
except `test_real_gpt2.py` use `FakeEngine` (`src/engine.py`), which
replies with a fixed string one byte per token, so they run instantly.

| Step | File | What | Tests |
|------|------|------|-------|
| 1 | `src/protocol.py` | `parse_request`: every field, every error | `TestParseRequest` |
| 2 | | `completion_body`, `chunk_body`, `sse` | `TestBodies` |
| 3 | `src/streaming.py` | `TextStream`: never emit half a character | `TestTextStream` |
| 4 | | `StopBuffer`: hold back what might become a stop string | `TestStopBuffer` |
| 5 | | `run_completion`: tokens in, text pieces + finish reason out | `TestRunCompletion` |
| 6 | `src/server.py` | `send_json`, `do_GET` | `TestRoutes` |
| 7 | | `do_POST`: non-streaming completions and errors | `TestCompletions` |
| 8 | | `_stream`: SSE, flushed per piece | `TestStreaming`, `TestSlowEngine` |
| 9 | real GPT-2 | `make serve`, then `curl` it | `tests/test_real_gpt2.py` |

## Run it

```bash
make test          # 38 tests (1 needs Sprint 4's GPT-2 download)
make serve         # GPT-2 on localhost:8000
make serve-fake    # no model, for poking at the protocol
make load          # with a server running: 1, 2, 4, 8 concurrent clients
```

## Results

Reference solution, dev Mac. Streaming, 24 tokens per request:

| Clients | TTFT p50 | TTFT max | Latency max | Throughput |
|---------|----------|----------|-------------|------------|
| 1 | 0.02 s | 0.02 s | 0.45 s | 55 tokens/sec |
| 4 | 0.71 s | 1.44 s | 1.80 s | 55 tokens/sec |

Throughput doesn't move. Four clients get the same 55 tokens/sec one
client did, split four ways, and the last one in the queue waits 1.4 s
for its first token. Every request reads all 124M weights once per token
on its own. Sprint 7 makes them share that read.

## Before writing the post

- **SSE vs WebSockets:** SSE is plain HTTP, one direction, text only,
  and it's what OpenAI and Anthropic both use for streaming.
- **HTTP/1.0 vs chunked encoding:** this server closes the connection to
  end a stream. HTTP/1.1 servers use `Transfer-Encoding: chunked` instead
  and keep the connection open. That's a stretch goal.
- **What production adds:** an async server (one thread can't block on
  one client), authentication, rate limits, request cancellation that
  actually stops the model, `/v1/chat/completions` with chat templates,
  and metrics.
- **Why one at a time:** two threads generating at once on one CPU just
  share it, and nobody gets faster. The fix isn't more threads, it's
  batching.

## What AI wrote

- **Scaffold:** this README, all tests, `src/engine.py` (FakeEngine and
  the GPT2Engine wiring), `src/deps.py`, `make_server`, `serve.py`,
  `benchmarks/load_test.py`, the Makefile, `POST.md`.
- **Hand-written:** request validation, the response and SSE bodies,
  `TextStream`, `StopBuffer`, `run_completion`, and every HTTP handler.

## Post status

Not started. Outline in `POST.md`.
