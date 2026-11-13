"""Owner 2026-11-13 (v9.15.18, glossary extraction in batch mode on ollamapull/, Ollama not installed at the start):
every setup line was logged twice; all 18 requests were sent before the model had loaded; Stop did not end them, so
the next run still printed their "Still waiting" lines and got "503 Loading model" from the llama-server one of them
had left half started.

A scripted server stands in for Ollama (and for the llama-server of Block other scripts): real sockets, nothing
installed."""
from __future__ import annotations

import json
import logging
import select
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import ollamapull  # noqa: E402

MODEL = "tiny:1b"
MESSAGES = [{"role": "user", "content": "x"}]


class FakeOllama:
    """Ollama's API as far as the route uses it. ``/api/generate`` loads the model (``load_seconds``, or until
    ``finish_load`` is set), ``/api/chat`` answers once ``release`` is set, and both notice a closed connection."""

    def __init__(self):
        self.loaded = False
        self.load_seconds = 0.0
        self.finish_load = threading.Event()
        self.finish_load.set()
        self.release = threading.Event()
        self.release.set()
        self.endless_stream = False
        self.loads = 0
        self.chats = 0
        self.chats_before_load = 0
        self.open_chats = 0
        self.aborted = 0
        self.closing = threading.Event()
        self.lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def _json(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _gone(self):
                readable, _, _ = select.select([self.connection], [], [], 0)
                if not readable:
                    return False
                try:
                    return self.connection.recv(1, socket.MSG_PEEK) == b""
                except OSError:
                    return True

            def _hold(self, done):
                """Wait like a busy server; False when the client closed the connection meanwhile."""
                while not done() and not fake.closing.is_set():
                    if self._gone():
                        with fake.lock:
                            fake.aborted += 1
                        return False
                    time.sleep(0.02)
                return not fake.closing.is_set()

            def do_GET(self):
                if self.path == "/api/ps":
                    models = [{"name": MODEL, "model": MODEL, "context_length": 8192}] if fake.loaded else []
                    self._json(200, {"models": models})
                elif self.path == "/api/tags":
                    self._json(200, {"models": [{"name": MODEL, "model": MODEL}]})
                else:
                    self._json(200, {"version": "0.99.0"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if self.path == "/api/generate":
                    if body.get("keep_alive") == 0:
                        fake.loaded = False
                        return self._json(200, {"done": True, "done_reason": "unload"})
                    with fake.lock:
                        fake.loads += 1
                    until = time.monotonic() + fake.load_seconds
                    if not self._hold(lambda: fake.finish_load.is_set() and time.monotonic() >= until):
                        self.close_connection = True
                        return None
                    fake.loaded = True
                    return self._json(200, {"done": True, "done_reason": "load"})
                with fake.lock:
                    fake.chats += 1
                    fake.open_chats += 1
                    if not fake.loaded:
                        fake.chats_before_load += 1
                try:
                    if fake.endless_stream:  # headers and a first line, then nothing more: like a long answer
                        self.send_response(200)
                        self.send_header("Content-Type", "application/x-ndjson")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        self.wfile.write(json.dumps({"message": {"content": "a"}, "done": False}).encode() + b"\n")
                        self.wfile.flush()
                        self._hold(lambda: False)
                        self.close_connection = True
                        return None
                    if not self._hold(fake.release.is_set):
                        self.close_connection = True
                        return None
                    lines = [{"message": {"content": "ok"}, "done": False},
                             {"done": True, "done_reason": "stop", "prompt_eval_count": 1, "eval_count": 1}]
                    data = b"".join(json.dumps(line).encode() + b"\n" for line in lines)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                finally:
                    with fake.lock:
                        fake.open_chats -= 1
                return None

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.closing.set()
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake(monkeypatch):
    server = FakeOllama()
    monkeypatch.setenv("OLLAMAPULL_BASE_URL", server.url)
    monkeypatch.delenv("OLLAMA_SETTINGS_JSON", raising=False)
    for name in ("_ban_server", ):
        monkeypatch.setattr(ollamapull, name, None)
    monkeypatch.setattr(ollamapull, "_ban_contexts", {})
    yield server
    server.close()


def _run(count, target):
    """``count`` threads on ``target(index)``; returns (threads, results, errors)."""
    results, errors = {}, {}

    def work(index):
        try:
            results[index] = target(index)
        except BaseException as exc:  # noqa: BLE001 - the test reads what the request ended with
            errors[index] = exc

    threads = [threading.Thread(target=work, args=(index,), daemon=True) for index in range(count)]
    for thread in threads:
        thread.start()
    return threads, results, errors


def _join(threads, seconds):
    deadline = time.monotonic() + seconds
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    return [thread for thread in threads if thread.is_alive()]


def _until(condition, seconds=5.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


# ---- 1. each line once ---------------------------------------------------------------------------------------

def _client(model=f"ollamapull/{MODEL}"):
    from unified_api_client import UnifiedClient

    client = UnifiedClient.__new__(UnifiedClient)
    client.model = model
    client.client_type = "ollamapull"
    client._get_active_request_model = lambda: model
    client._is_stop_requested = lambda: False
    client._get_thread_local_client = lambda: SimpleNamespace()
    client._streaming_enabled = lambda: True
    client._stream_logging_enabled = lambda _streaming: False
    client._stream_thinking_logging_enabled = lambda: False
    client._script_ban_request = lambda _messages, _context: ((), frozenset())
    client._script_ban_allow_kaomoji = lambda: False
    client._repeat_watch = lambda _messages: None
    return client


def test_setup_lines_reach_the_run_log_once(monkeypatch, capsys, caplog):
    """The run log shows both what is printed and what the client logs: every installer line was in it twice."""
    import requests

    for name in ("TRANSLATION_CANCELLED", "GRACEFUL_STOP", "GRACEFUL_STOP_COMPLETED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(ollamapull, "_ollama_executable", lambda: None)
    monkeypatch.setattr(ollamapull, "_api_get", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError()))

    def installer(progress, _should_stop):
        ollamapull._progress(progress, "Downloading the official Ollama installer...")
        ollamapull._progress(progress, "#                                        3.7%")

    monkeypatch.setattr(ollamapull, "_run_installer", installer)
    monkeypatch.setattr(ollamapull, "_after_update", lambda *a: None)
    monkeypatch.setattr(ollamapull, "_start_server", lambda *a: None)
    monkeypatch.setattr(ollamapull, "_models", lambda: [{"name": MODEL}])
    monkeypatch.setattr(ollamapull, "chat", lambda *a, **k: {
        "content": "ok", "finish_reason": "stop", "usage": {}, "raw_response": {}})
    with caplog.at_level(logging.INFO):
        assert _client()._send_ollamapull(MESSAGES, 0.2, 64, None).content == "ok"
    run_log = capsys.readouterr().out.splitlines() + [record.getMessage() for record in caplog.records]
    for line in ("Downloading the official Ollama installer...", "3.7%"):
        assert sum(line in entry for entry in run_log) == 1, [entry for entry in run_log if line in entry]


# ---- 2. the batch waits for the model ------------------------------------------------------------------------

def test_one_request_loads_the_model_and_the_batch_waits_for_it(fake):
    fake.load_seconds = 0.5
    fake.release.clear()
    status = []
    threads, results, errors = _run(6, lambda _i: ollamapull.chat(f"ollamapull/{MODEL}", MESSAGES,
                                                                progress=status.append))
    assert _until(lambda: fake.chats == 6), (fake.chats, errors)
    fake.release.set()
    assert not _join(threads, 5) and not errors, errors
    assert fake.loads == 1  # one request loaded it
    assert fake.chats_before_load == 0  # nothing reached the model while it was loading
    assert [result["content"] for result in results.values()] == ["ok"] * 6
    assert sum(line.startswith(f"Loading {MODEL}") for line in status) == 1
    assert sum(f"{MODEL} is loaded" in line for line in status) == 1
    assert sum(line.startswith("Sending request for") for line in status) == 1  # said once for the batch
    assert status.index(next(line for line in status if "is loaded" in line)) < status.index(
        next(line for line in status if line.startswith("Sending request for")))


def test_a_loaded_model_is_not_loaded_again(fake):
    fake.loaded = True
    assert ollamapull.chat(f"ollamapull/{MODEL}", MESSAGES)["content"] == "ok"
    assert fake.loads == 0


def test_the_wait_is_reported_once_for_the_whole_batch(fake, monkeypatch):
    monkeypatch.setattr(ollamapull, "_WAIT_REPORT_SECONDS", 0.4, raising=False)
    fake.loaded = True
    fake.release.clear()
    status = []
    threads, _results, errors = _run(5, lambda _i: ollamapull.chat(f"ollamapull/{MODEL}", MESSAGES,
                                                                 progress=status.append))
    assert _until(lambda: fake.chats == 5)
    assert _until(lambda: any(line.startswith("Still waiting") for line in status), 4)
    time.sleep(0.3)  # a line per request would all have arrived by now
    waiting = [line for line in status if line.startswith("Still waiting")]
    fake.release.set()
    assert not _join(threads, 5) and not errors, errors
    assert len(waiting) == 1 and "5 requests" in waiting[0], waiting


# ---- 3. stop -------------------------------------------------------------------------------------------------

def test_stop_ends_a_request_the_server_has_not_answered(fake):
    """Ollama sends nothing, not even headers, until the first token: that wait ignored Stop."""
    fake.loaded = True
    fake.release.clear()
    stop = threading.Event()
    threads, _results, errors = _run(3, lambda _i: ollamapull.chat(f"ollamapull/{MODEL}", MESSAGES,
                                                                 should_stop=stop.is_set))
    assert _until(lambda: fake.open_chats == 3)
    stop.set()
    assert not _join(threads, 3), "the requests outlived Stop"
    assert all(isinstance(error, ollamapull.OllamaPullCancelled) for error in errors.values()) and len(errors) == 3
    assert _until(lambda: fake.aborted == 3 and fake.open_chats == 0), "the server still works on them"


def test_stop_ends_an_answer_that_is_streaming(fake):
    """Also on a "Connection: close" response, where http.client itself keeps no socket to close."""
    fake.loaded = True
    fake.endless_stream = True
    stop = threading.Event()
    status = []
    threads, _results, errors = _run(2, lambda _i: ollamapull.chat(f"ollamapull/{MODEL}", MESSAGES,
                                                                 progress=status.append, should_stop=stop.is_set))
    assert _until(lambda: sum("First text token" in line for line in status) == 2)
    stop.set()
    assert not _join(threads, 3), "the requests outlived Stop"
    assert len(errors) == 2 and all(isinstance(error, ollamapull.OllamaPullCancelled) for error in errors.values())
    assert _until(lambda: fake.aborted == 2 and fake.open_chats == 0), "the server still works on them"


def test_stop_ends_the_requests_that_wait_for_the_model(fake):
    fake.finish_load.clear()  # the load does not finish
    stop = threading.Event()
    threads, _results, errors = _run(4, lambda _i: ollamapull.chat(f"ollamapull/{MODEL}", MESSAGES,
                                                                 should_stop=stop.is_set))
    assert _until(lambda: fake.loads == 1)
    time.sleep(0.3)
    stop.set()
    assert not _join(threads, 3), "the requests outlived Stop"
    assert len(errors) == 4 and all(isinstance(error, ollamapull.OllamaPullCancelled) for error in errors.values())
    assert fake.chats == 0 and fake.loads == 1
    assert _until(lambda: fake.aborted == 1)  # the load request was closed too


def test_stop_ends_a_request_that_waits_for_another_ones_setup(monkeypatch):
    """ensure_ready: a request that waits for the one installing / pulling ends on Stop, not after the pull."""
    holding, release = threading.Event(), threading.Event()
    monkeypatch.setattr(ollamapull, "_ollama_executable", lambda: "ollama")
    monkeypatch.setattr(ollamapull, "_models", lambda: [{"name": MODEL}])

    def slow_start(_progress, _should_stop):
        holding.set()
        release.wait(10)

    monkeypatch.setattr(ollamapull, "_start_server", slow_start)
    first, _r, _e = _run(1, lambda _i: ollamapull.ensure_ready(MODEL, auto_update=False))
    assert holding.wait(3)
    stop = threading.Event()
    second, _results, errors = _run(1, lambda _i: ollamapull.ensure_ready(MODEL, auto_update=False,
                                                                         should_stop=stop.is_set))
    time.sleep(0.3)
    stop.set()
    alive = _join(second, 3)
    release.set()
    _join(first, 3)
    assert not alive, "the waiting request outlived Stop"
    assert isinstance(errors.get(0), ollamapull.OllamaPullCancelled)


@pytest.fixture
def only_this_route_is_cancelled(monkeypatch):
    """hard_cancel_all also cancels every other route and silences the HTTP loggers for the whole process: keep
    that out of the tests that run after this one."""
    import unified_api_client

    for name in dir(unified_api_client):
        if name.endswith("_cancel_stream"):
            monkeypatch.setattr(unified_api_client, name, None)
    monkeypatch.setattr(unified_api_client, "_authnan_auth", None)
    levels = {name: logging.getLogger(name).level for name in ("httpx", "openai", "urllib3")}
    yield
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)
    unified_api_client.UnifiedClient.set_global_cancellation(False)


def test_a_stopped_run_is_over_when_the_next_one_starts(fake, monkeypatch, caplog, only_this_route_is_cancelled):
    """Owner: the second run showed 17 "Still waiting" lines of the first. A hard stop closes the connections
    (hard_cancel_all); the stop flags are cleared right after for the new run, and the old requests must not
    wake up in it."""
    from unified_api_client import UnifiedClient

    for name in ("TRANSLATION_CANCELLED", "GRACEFUL_STOP", "GRACEFUL_STOP_COMPLETED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(ollamapull, "_WAIT_REPORT_SECONDS", 0.3, raising=False)
    monkeypatch.setattr(ollamapull, "ensure_ready", lambda *a, **k: None)
    fake.loaded = True
    fake.release.clear()
    client = _client()
    with caplog.at_level(logging.INFO):
        threads, _results, errors = _run(4, lambda _i: client._send_ollamapull(MESSAGES, 0.2, 64, None))
        assert _until(lambda: fake.open_chats == 4)
        UnifiedClient.hard_cancel_all()  # Stop
        UnifiedClient.set_global_cancellation(False)  # the next run begins
        alive = _join(threads, 3)
        assert not alive, "requests of the stopped run are still running"
        assert _until(lambda: fake.open_chats == 0), "Ollama still works on the stopped run's requests"
        assert len(errors) == 4 and all(getattr(error, "error_type", "") == "cancelled" for error in errors.values())
        caplog.clear()
        time.sleep(0.8)
        assert not [record.getMessage() for record in caplog.records if "Still waiting" in record.getMessage()]


def test_a_waiting_request_ends_only_on_the_configured_timeout(fake, monkeypatch):
    """A request of a batch waits its turn in silence. That wait had a fixed limit of 600 s which no setting
    reached; the limit is the app's own timeout settings now, and there is none while they are off."""
    for name in ("TRANSLATION_CANCELLED", "GRACEFUL_STOP", "GRACEFUL_STOP_COMPLETED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(ollamapull, "ensure_ready", lambda *a, **k: None)
    fake.loaded = True
    fake.release.clear()
    sent = []
    real_post = ollamapull._post
    monkeypatch.setattr(ollamapull, "_post", lambda *a, **k: sent.append(k.get("timeout")) or real_post(*a, **k))

    monkeypatch.setenv("RETRY_TIMEOUT", "0")  # Auto-retry Slow Processing off
    monkeypatch.setenv("ENABLE_HTTP_TUNING", "0")  # HTTP timeout overrides off
    threads, results, _errors = _run(1, lambda _i: _client()._send_ollamapull(MESSAGES, 0.2, 64, None))
    assert _until(lambda: fake.open_chats == 1)
    assert sent[-1][1] is None, "a limit on the wait although both timeout settings are off"
    fake.release.set()
    assert not _join(threads, 5) and results[0].content == "ok"

    fake.release.clear()
    monkeypatch.setenv("RETRY_TIMEOUT", "1")
    monkeypatch.setenv("ENABLE_HTTP_TUNING", "1")
    monkeypatch.setenv("CONNECT_TIMEOUT", "5")
    monkeypatch.setenv("READ_TIMEOUT", "0.5")
    with pytest.raises(Exception, match="timed out"):
        _client()._send_ollamapull(MESSAGES, 0.2, 64, None)
    assert sent[-1] == (5.0, 0.5)


# ---- 4. Block other scripts: our llama-server ----------------------------------------------------------------

class FakeLlamaServer:
    """llama-server: 503 "Loading model" on /health and on chat until the model is in memory."""

    def __init__(self, port=0):
        self.loaded = threading.Event()
        self.chats = 0
        self.refused = 0
        self.reload_on_first_chat = 0.0
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def _json(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _loading(self):
                self._json(503, {"error": {"message": "Loading model", "type": "unavailable_error", "code": 503}})

            def do_GET(self):
                self._json(200, {"status": "ok"}) if fake.loaded.is_set() else self._loading()

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                fake.chats += 1
                if fake.reload_on_first_chat and fake.chats == 1:
                    fake.loaded.clear()
                    threading.Timer(fake.reload_on_first_chat, fake.loaded.set).start()
                if not fake.loaded.is_set():
                    fake.refused += 1
                    return self._loading()
                return self._json(200, {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                                        "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def llama(fake, monkeypatch, tmp_path):
    """``_start_ban_server`` with a llama-server that is a FakeLlamaServer on the port it was given."""
    started = []
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"GGUF")

    def popen(args, **_kwargs):
        server = FakeLlamaServer(int(args[args.index("--port") + 1]))
        started.append(server)
        return SimpleNamespace(poll=lambda: None, returncode=None, terminate=server.close,
                               wait=lambda timeout=None: 0, kill=lambda: None)

    monkeypatch.setattr(ollamapull.mobile_runtime, "subprocesses_available", lambda: True)
    monkeypatch.setattr(ollamapull, "_llama_server_executable", lambda: "llama-server")
    monkeypatch.setattr(ollamapull, "_llama_backend_env", lambda _server: {})
    monkeypatch.setattr(ollamapull, "_model_files", lambda _name: (str(model_file), ""))
    monkeypatch.setattr(ollamapull.subprocess, "Popen", popen)
    yield started
    for server in started:
        try:
            server.close()
        except Exception:
            pass


def test_a_start_that_stop_ended_does_not_hand_out_a_loading_server(fake, llama):
    """The stopped request left the server registered while it was still loading; the next one got its URL at
    once and its chat request was answered 503 Loading model."""
    stop = threading.Event()
    first, _results, errors = _run(1, lambda _i: ollamapull._start_ban_server(MODEL, {}, False, None, stop.is_set))
    assert _until(lambda: len(llama) == 1)
    stop.set()
    assert not _join(first, 3) and isinstance(errors.get(0), ollamapull.OllamaPullCancelled)

    second, results, errors = _run(1, lambda _i: ollamapull._start_ban_server(MODEL, {}, False, None, None))
    assert _join(second, 0.6), "the request got a server whose model is still loading"
    llama[0].loaded.set()
    assert not _join(second, 3) and not errors, errors
    assert results[0][0] == llama[0].url and len(llama) == 1  # the same server, not a second one


def test_a_batch_shares_one_llama_server(fake, llama, monkeypatch):
    """Each request asked Ollama for the context size on its own; the ones that asked after the first had
    unloaded the model read none, took the fallback, and restarted the server under the others' requests
    ("Remote end closed connection without response")."""
    monkeypatch.setattr(ollamapull, "_LLAMA_FALLBACK_CONTEXT", 4096)
    fake.load_seconds = 0.2
    real_get = ollamapull._api_get

    def slow_ps(path, **kwargs):
        if path == "/api/ps" and threading.current_thread().name != "first":
            time.sleep(0.5)  # this request reads the context after the first has unloaded the model
        return real_get(path, **kwargs)

    monkeypatch.setattr(ollamapull, "_api_get", slow_ps)
    results, errors = {}, {}

    def work(index):
        try:
            results[index] = ollamapull._start_ban_server(MODEL, {}, False, None, None)
        except BaseException as exc:  # noqa: BLE001
            errors[index] = exc

    threads = [threading.Thread(target=work, args=(index,), name="first" if index == 0 else f"later{index}",
                                daemon=True) for index in range(4)]
    for thread in threads:
        thread.start()
    assert _until(lambda: len(llama) >= 1)
    llama[0].loaded.set()
    for _ in range(40):  # a restarted server would need its model "loaded" too
        for server in llama:
            server.loaded.set()
        if not _join(threads, 0.1):
            break
    assert not _join(threads, 1) and not errors, errors
    assert len(llama) == 1, f"llama-server was started {len(llama)} times"
    assert {url for url, _file in results.values()} == {llama[0].url}


def test_loading_model_503_waits_and_sends_again(fake, monkeypatch):
    """503 "Loading model" is not a failed request: it waits for /health and goes out again."""
    server = FakeLlamaServer()
    try:
        server.loaded.set()
        server.reload_on_first_chat = 0.5
        monkeypatch.delenv("OLLAMAPULL_BASE_URL", raising=False)
        monkeypatch.setattr(ollamapull.mobile_runtime, "subprocesses_available", lambda: True)
        monkeypatch.setattr(ollamapull, "_ban_server", {
            "process": SimpleNamespace(poll=lambda: None, returncode=None, terminate=lambda: None,
                                       wait=lambda timeout=None: 0, kill=lambda: None),
            "url": server.url, "key": "k", "name": MODEL, "started": time.monotonic(), "ready": True})
        monkeypatch.setattr(ollamapull, "_token_ban_request", lambda *a, **k: (
            f"{server.url}/v1/chat/completions", {"messages": MESSAGES, "stream": False}))
        result = ollamapull.chat(f"ollamapull/{MODEL}", MESSAGES, stream=False, ban_scripts=("han",))
        assert result["content"] == "ok"
        assert server.refused == 1 and server.chats == 2
    finally:
        server.close()
