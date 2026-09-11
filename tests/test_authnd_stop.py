"""Delayed transports must not outlive cancellation or leak into the next run."""
import contextlib
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import authnd_auth as authnd


@pytest.fixture(autouse=True)
def cancellation_state(monkeypatch):
    monkeypatch.delenv("TRANSLATION_CANCELLED", raising=False)
    monkeypatch.setattr(authnd, "_cancel_event", threading.Event())
    monkeypatch.setattr(authnd, "_cancel_generation", threading.Event())
    monkeypatch.setattr(authnd, "_active_response_closers", set())
    monkeypatch.setattr(authnd, "_active_sessions", set())
    monkeypatch.setattr(authnd, "_active_helper_processes", set())


@pytest.mark.parametrize("phase", ["headers", "error_body", "stream"])
def test_stop_closes_client_and_ignores_late_result_after_reset(monkeypatch, phase):
    entered, release, finished = (threading.Event() for _ in range(3))
    clients, logs, errors = [], [], []

    def wait():
        entered.set()
        assert release.wait(3)

    class Response:
        status_code = 200 if phase == "stream" else 500
        headers = {}
        reason_phrase = "Internal Server Error"
        def close(self):
            pass
        def read(self):
            if phase == "error_body":
                wait()
            return b'{"detail":"Retries exhausted: 3/3"}'
        def iter_raw(self):
            wait()
            yield b'data: {"choices":[{"delta":{"content":"late"}}]}\n\n'
            yield b'data: [DONE]\n\n'

    class Client:
        def __init__(self, **kwargs):
            self.closed = False
            clients.append(self)
        def __enter__(self):
            return self
        def close(self):
            self.closed = True
        def __exit__(self, *args):
            self.close()
            finished.set()
        @contextlib.contextmanager
        def stream(self, *args, **kwargs):
            if phase == "headers":
                wait()
            yield Response()

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(
        Client=Client, Timeout=lambda *args, **kwargs: None))

    @authnd._request_scope
    def request(*, log_fn):
        return authnd._send_prediction_payload(
            {}, url="http://unused.invalid", headers={}, timeout=30,
            connect_timeout=10, stream=True, log_fn=log_fn,
            log_stream=True, progress_label=None, max_tokens=None)

    def call():
        try:
            request(log_fn=logs.append)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=call)
    thread.start()
    try:
        assert entered.wait(2)
        authnd.cancel_stream()
        assert clients[0].closed  # Registered even before headers arrive.
        authnd.reset_cancel()
        assert not authnd._is_cancelled()  # A new run may proceed.
        thread.join(1)
        assert not thread.is_alive()
        assert len(errors) == 1 and "cancelled" in str(errors[0])
        before = list(logs)
        release.set()
        assert finished.wait(2)
        assert logs == before
        assert not any("HTTP failure" in line for line in logs)
    finally:
        release.set()
        thread.join(2)


def test_requests_closes_response_arriving_after_cancellation():
    entered, release, closed = (threading.Event() for _ in range(3))
    errors = []
    class Session:
        def post(self):
            entered.set()
            assert release.wait(3)
            return SimpleNamespace(close=closed.set)
        def close(self):
            pass
    @authnd._request_scope
    def call():
        try:
            authnd._post_with_cancel(Session())
        except RuntimeError as exc:
            errors.append(exc)
    thread = threading.Thread(target=call)
    thread.start()
    try:
        assert entered.wait(2)
        authnd.cancel_stream()
        authnd.reset_cancel()
        thread.join(1)
        assert not thread.is_alive()
        assert errors and "cancelled" in str(errors[0])
        release.set()
        assert closed.wait(2)
    finally:
        release.set()
        thread.join(2)


def test_active_http_error_is_not_suppressed():
    logs = []
    authnd._log(logs.append, "⚠️ AuthND HTTP failure: HTTP 500")
    assert len(logs) == 1
