"""HTTP logging must leave streaming responses available to their consumer."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _streaming_probe(case, log_directory):
    # Run in a subprocess: installing the logger monkey-patches global clients.
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import requests
    from http_logger import enable_detailed_http_logging

    acknowledgment = threading.Event()
    expired = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.do_GET()

        def do_GET(self):
            if self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(b'data: {"arena_progress":"dispatch"}\n\n')
                self.wfile.flush()
                if not acknowledgment.wait(1.5):
                    expired.set()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                self.close_connection = True
                return
            if self.path == "/ack":
                acknowledgment.set()
            body = json.dumps({"ok": True, "path": self.path}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    enable_detailed_http_logging(log_directory)
    response = None
    client = None
    try:
        if case == "nonstream":
            response = requests.post(base_url + "/regular", json={"message": "example"}, timeout=5)
            assert response.json() == {"ok": True, "path": "/regular"}
            records = [json.loads(path.read_text(encoding="utf-8")) for path in Path(log_directory).glob("*.json")]
            assert records, "The non-streaming response was not logged"
            assert any(item["response"]["body"] == {"ok": True, "path": "/regular"} for item in records)
            assert any(item["request"]["body"] == {"message": "example"} for item in records)
            return
        if case == "requests_post":
            response = requests.post(base_url + "/stream", stream=True, timeout=5)
        elif case == "requests_request":
            response = requests.request("GET", base_url + "/stream", stream=True, timeout=5)
        elif case == "requests_session_default":
            client = requests.Session()
            client.stream = True
            response = client.get(base_url + "/stream", timeout=5)
        elif case == "httpx":
            import httpx
            client = httpx.Client(timeout=5)
            response = client.send(client.build_request("GET", base_url + "/stream"), stream=True)
        else:
            raise AssertionError(f"Unknown probe {case}")
        assert not expired.is_set(), (
            "HTTP logging consumed the SSE response before returning it: "
            "the dispatch acknowledgment expired while the caller was blocked"
        )
        lines = response.iter_lines() if case == "httpx" else response.iter_lines(chunk_size=1, decode_unicode=True)
        first = next(lines)
        if isinstance(first, bytes):
            first = first.decode()
        assert first == 'data: {"arena_progress":"dispatch"}'
        requests.post(base_url + "/ack", timeout=5).raise_for_status()
        remaining = list(lines)
        assert "data: [DONE]" in remaining or b"data: [DONE]" in remaining
        assert not expired.is_set(), "The streamed dispatch event was buffered until expiry"
    finally:
        acknowledgment.set()
        if response is not None:
            response.close()
        if client is not None:
            client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class HttpLoggerStreamingTest(unittest.TestCase):
    def _probe(self, case):
        with tempfile.TemporaryDirectory(prefix="http-logger-test-") as directory:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--probe", case, directory],
                capture_output=True, text=True, timeout=15,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_requests_post_preserves_dispatch_handshake(self):
        self._probe("requests_post")

    def test_requests_request_preserves_dispatch_handshake(self):
        self._probe("requests_request")

    def test_requests_session_stream_default_preserves_dispatch_handshake(self):
        self._probe("requests_session_default")

    def test_httpx_preserves_dispatch_handshake(self):
        self._probe("httpx")

    def test_nonstream_response_body_still_logged(self):
        self._probe("nonstream")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--probe":
        _streaming_probe(sys.argv[2], sys.argv[3])
    else:
        unittest.main()
