"""An OpenAI-compatible HTTP server, standard library only.

    GET  /health            {"status": "ok"}
    GET  /v1/models         {"object": "list", "data": [{"id": <name>, "object": "model"}]}
    POST /v1/completions    see protocol.py

Python's http.server does the HTTP parsing. A ThreadingHTTPServer gives
each connection its own thread, and Handler's do_GET / do_POST are called
with the request already parsed: self.path, self.headers, and self.rfile
(the body, read exactly Content-Length bytes of it) / self.wfile (where
the response goes).

Everything after the headers is plain bytes you write yourself:

    self.send_response(200)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

**One request at a time.** The model runs under self.server.lock, so two
requests never generate at once. The second waits for the first to
finish. That's correct, and it's exactly the bottleneck Sprint 7 removes
with batching. benchmarks/load_test.py measures it.

Order of work (tests/test_server.py):
    1. send_json, do_GET                         TestRoutes
    2. do_POST: parse, errors, non-streaming     TestCompletions
    3. _stream: SSE                              TestStreaming
"""

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .protocol import BadRequest, chunk_body, completion_body, error_body, parse_request, sse
from .streaming import run_completion


class Handler(BaseHTTPRequestHandler):
    # HTTP/1.0: the connection closes after each response, so a streamed
    # body needs no Content-Length or chunked encoding. (Stretch: 1.1.)
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt, *args):  # quiet by default; server.verbose turns it on
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    def send_json(self, status, obj):
        """Write a complete JSON response: status line, Content-Type
        application/json, Content-Length, then the UTF-8 body."""
        raise NotImplementedError("send_json (TestRoutes)")

    def do_GET(self):
        """/health and /v1/models as above (the model id is
        self.server.engine.name); anything else is a 404 with error_body."""
        raise NotImplementedError("do_GET (TestRoutes)")

    def do_POST(self):
        """POST /v1/completions:

        1. anything but that path -> 404
        2. read the body: int(self.headers.get("Content-Length", 0)) bytes
        3. parse_request; BadRequest -> 400 with error_body(str(e))
        4. encode the prompt (engine.encode); id "cmpl-" + uuid4 hex,
           created = int(time.time())
        5. under `with self.server.lock:`
             stream -> self._stream(...)
             else   -> join every piece from run_completion, send 200 with
                       completion_body (usage counts from the prompt ids
                       and the last n_tokens)"""
        raise NotImplementedError("do_POST (TestCompletions)")

    def _stream(self, req, ids, cid, created):
        """Server-Sent Events:

        - send 200 with headers Content-Type: text/event-stream and
          Cache-Control: no-cache (no Content-Length: the length isn't
          known yet), then end_headers()
        - for each (text, finish_reason, n) from run_completion: write
          sse(chunk_body(...)) and self.wfile.flush() so it leaves NOW,
          not when the buffer fills. Skip pieces with empty text unless
          they carry a finish_reason.
        - finally write sse("[DONE]")
        - if the client disconnects mid-stream, writing raises
          BrokenPipeError / ConnectionResetError: stop generating quietly."""
        raise NotImplementedError("_stream (TestStreaming)")


def make_server(engine, host="127.0.0.1", port=8000, verbose=False):
    """Scaffold: a ThreadingHTTPServer carrying the engine and the lock.
    port=0 picks a free port (the tests use it): server.server_address[1]."""
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.engine = engine
    server.lock = threading.Lock()
    server.verbose = verbose
    return server
