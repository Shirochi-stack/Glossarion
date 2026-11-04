"""Fake OpenAI-compatible chat server for the multi-instance key isolation tests.

Records the API key (``Authorization: Bearer``) and model of every request and answers per key:

* ``"ok"`` (default): a normal chat completion (streamed when the request asks for it);
* ``"prohibited"``: an empty answer with ``finish_reason: content_filter`` (the client's safety
  filter path: main-key retry, glossary keys, Fallback pool);
* ``"blocked"``: HTTP 400 "content_policy_violation" (the prohibited-content error path);
* ``"401"`` / ``"429"`` / ``"500"``: that HTTP error (rotation / cooldown / main-key retry).

Used in-process (a thread) by tests/test_instance_key_isolation.py; an instance process reaches it
over HTTP. Python 3.10 compatible, stdlib only.
"""

import http.server
import json
import threading
import time

__all__ = ["FakeKeyServer"]


class _Handler(http.server.BaseHTTPRequestHandler):
    server_version = "FakeKeyServer/1"

    def log_message(self, *args):
        pass

    def _send_json(self, status, payload):
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):  # model listings
        self._send_json(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in
                                                          self.server.owner.models]})

    def do_POST(self):
        owner = self.server.owner
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            body = {}
        auth = self.headers.get("Authorization", "") or self.headers.get("x-api-key", "")
        key = auth.split(" ", 1)[1] if auth.lower().startswith("bearer ") else auth
        behaviour = owner.behaviour(key)
        stream = bool(body.get("stream"))
        owner.record({"key": key, "model": body.get("model"), "path": self.path, "stream": stream,
                      "behaviour": behaviour, "ts": time.time()})
        if owner.delay:
            time.sleep(owner.delay)
        if behaviour in ("401", "429", "500"):
            status = int(behaviour)
            message = {"401": "Incorrect API key provided", "429": "Rate limit reached for requests",
                       "500": "The server had an error"}[behaviour]
            kind = {"401": "invalid_request_error", "429": "rate_limit_error", "500": "server_error"}[behaviour]
            self._send_json(status, {"error": {"message": message, "type": kind, "code": behaviour}})
            return
        if behaviour == "blocked":
            self._send_json(400, {"error": {
                "message": "Your request was rejected as a result of our safety system: content_policy_violation",
                "type": "invalid_request_error", "code": "content_policy_violation"}})
            return
        if behaviour == "prohibited":
            content, finish = "", "content_filter"
        else:
            content, finish = owner.answer(key, body), "stop"
        model = body.get("model")
        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            first = {"id": "x", "object": "chat.completion.chunk", "created": 1, "model": model,
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": content},
                                  "finish_reason": None}]}
            last = {"id": "x", "object": "chat.completion.chunk", "created": 1, "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
            self.wfile.write(f"data: {json.dumps(first)}\n\ndata: {json.dumps(last)}\n\ndata: [DONE]\n\n".encode())
            self.wfile.flush()
            return
        self._send_json(200, {
            "id": "x", "object": "chat.completion", "created": 1, "model": model,
            "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        })


class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class FakeKeyServer:
    """``with FakeKeyServer({"sk-bad": "prohibited"}) as server: ... server.keys()``."""

    def __init__(self, behaviours=None, *, delay=0.0, models=("gpt-4o-mini",)):
        self.behaviours = dict(behaviours or {})
        self.delay = delay
        self.models = list(models)
        self.requests = []
        self._lock = threading.Lock()
        self._server = None
        self._thread = None

    # -- per-key behaviour and answers -------------------------------------------
    def behaviour(self, key):
        return self.behaviours.get(key, "ok")

    def answer(self, key, body):
        """A normal answer naming the key that served it."""
        return f"<p>Translated by the fake server for key {key}: the paragraph text is here.</p>"

    def record(self, entry):
        with self._lock:
            self.requests.append(entry)

    # -- results -----------------------------------------------------------------
    def keys(self):
        with self._lock:
            return [entry["key"] for entry in self.requests]

    def clear(self):
        with self._lock:
            self.requests.clear()

    # -- lifecycle ---------------------------------------------------------------
    @property
    def port(self):
        return self._server.server_address[1]

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}/v1"

    def start(self):
        self._server = _Server(("127.0.0.1", 0), _Handler)
        self._server.owner = self
        self._thread = threading.Thread(target=self._server.serve_forever, name="fake-key-server", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False
