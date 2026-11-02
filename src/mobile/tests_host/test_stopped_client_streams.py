"""The garbage collector finishing a stopped run's client leaves the next run's requests alone.

A force-stopped run's ``UnifiedClient`` stays ``_cancelled`` until a collection frees it, which can be in the middle of
the Resume. Its ``__del__`` closed every stream in the class-wide ``_active_streams``, so a request of the new run lost
its socket on an unrelated thread (``[Errno 9] Bad file descriptor``, WinError 10038 on Windows) and went out again.
"""

from __future__ import annotations

import pytest

import local_llm_fakes as fakes  # noqa: F401  (sys.path for the app and src)


class _Closable:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def api():
    import unified_api_client

    return unified_api_client


@pytest.fixture
def new_run_stream(api):
    client_class = api.UnifiedClient
    stream = _Closable()
    with client_class._active_streams_lock:
        client_class._active_streams.add(stream)
    yield stream
    with client_class._active_streams_lock:
        client_class._active_streams.discard(stream)


def _stopped_client(api):
    client = object.__new__(api.UnifiedClient)
    client._cancelled = True
    client.openai_client = _Closable()
    client.gemini_client = _Closable()
    return client


def test_a_stopped_clients_del_leaves_the_active_streams_open(api, new_run_stream):
    _stopped_client(api).__del__()
    assert not new_run_stream.closed
    assert new_run_stream in api.UnifiedClient._active_streams


def test_a_del_during_a_stop_leaves_them_open_too(api, new_run_stream, monkeypatch):
    """The stop itself closes them (``hard_cancel_all``), on purpose and once; a collection adds nothing to that."""
    monkeypatch.setenv("TRANSLATION_CANCELLED", "1")
    client = object.__new__(api.UnifiedClient)
    client._cancelled = False
    client.__del__()
    assert not new_run_stream.closed


def test_a_stopped_clients_del_still_closes_its_own_clients(api):
    client = _stopped_client(api)
    own = client.openai_client, client.gemini_client
    client.__del__()
    assert all(item.closed for item in own)


def test_a_force_stop_still_closes_the_active_streams(api, new_run_stream, monkeypatch):
    client_class = api.UnifiedClient
    for name in ("_all_sessions", "_all_httpx_clients", "_all_openai_clients"):
        monkeypatch.setattr(client_class, name, set())
    monkeypatch.setattr(api, "_api_watchdog_reset", lambda: None)
    monkeypatch.setattr(client_class, "set_global_cancellation", classmethod(lambda cls, value: None))
    client_class.hard_cancel_all()
    assert new_run_stream.closed
