"""NanoGPT chat always streams, so it is not disconnected at 340 s; with the toggle off the Log stays quiet.

The real cutoff was measured on the owner's authnan/ account on 2026-11-02 (httpx: 340 s every time, streaming: past
900 s). These tests run against a local server and check what the app sends and what it does with the reply.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from unified_api_client import UnifiedClient

WORDS = ["Hello", " there", ", world"]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.bodies.append(body)
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for word in WORDS:
                chunk = {"choices": [{"index": 0, "delta": {"content": word}, "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            done = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            self.wfile.write(f"data: {json.dumps(done)}\n\ndata: [DONE]\n\n".encode())
            return
        reply = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(WORDS)},
                              "finish_reason": "stop"}]}
        payload = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.bodies = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _send(monkeypatch, tmp_path, server, provider, streaming):
    monkeypatch.setenv("USE_CUSTOM_OPENAI_ENDPOINT", "0")
    monkeypatch.setenv("ENABLE_STREAMING", "1" if streaming else "0")
    monkeypatch.setenv("LOG_STREAM_CHUNKS", "1")
    monkeypatch.delenv("GRACEFUL_STOP", raising=False)
    client = UnifiedClient("test-key", "authnan/glm-5.3", str(tmp_path))
    monkeypatch.setattr(client, "_get_max_retries", lambda: 1)
    monkeypatch.setattr(client, "_get_send_interval", lambda: 0)
    monkeypatch.setattr(client, "_is_stop_requested", lambda: False)
    monkeypatch.setattr(client, "_save_response", lambda *args, **kwargs: None)
    monkeypatch.setattr(client, "_should_show_api_lifecycle_logs", lambda: False)
    return client._send_openai_compatible(
        [{"role": "user", "content": "Reply."}], 0.5, 1000,
        f"http://127.0.0.1:{server.server_address[1]}/v1", "quiet-stream-test", provider=provider,
        model_override="glm-5.3", api_key_override="test-key", allow_endpoint_override=False,
    )


def test_toggle_off_still_streams_and_collects_the_whole_reply(monkeypatch, tmp_path, server, capsys):
    response = _send(monkeypatch, tmp_path, server, "nanogpt", streaming=False)
    assert server.bodies[-1]["stream"] is True
    assert response.content == "".join(WORDS)
    assert response.finish_reason == "stop"
    # Quiet, as non-streaming is: the words only arrive in the reply, not one by one in the Log.
    assert not any(word.strip() in capsys.readouterr().out for word in WORDS[1:])


def test_toggle_on_streams_too(monkeypatch, tmp_path, server):
    response = _send(monkeypatch, tmp_path, server, "nanogpt", streaming=True)
    assert server.bodies[-1]["stream"] is True and response.content == "".join(WORDS)


def test_other_providers_keep_the_toggle(monkeypatch, tmp_path, server):
    response = _send(monkeypatch, tmp_path, server, "deepseek", streaming=False)
    assert not server.bodies[-1].get("stream") and response.content == "".join(WORDS)
