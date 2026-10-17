"""On-device engine robustness (``services/local_llm`` + the shared ``local/`` route): the endpoint outlives
an Android ``detach`` while a job runs and keeps its address across a restart, the client reads the
current address on every retry, concurrent OpenAI SDK streams queue without connection errors, the
progress lines (request / waiting / loading / loaded) reach the client's log, Stop (graceful and
immediate) reaches queued and running generations, and selecting another model frees the loaded one.
FakeEngine only; no network, no model.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import socket
import threading
import time

import pytest

import local_llm_fakes as fakes  # noqa: F401  (sys.path for the app and src)
from glossarion_mobile.services.local_llm import engine as eng
from glossarion_mobile.services.local_llm import service as svc
from test_local_llm_server import make_service


@pytest.fixture()
def running(tmp_path, monkeypatch):
    monkeypatch.delenv(svc.URL_ENV, raising=False)
    monkeypatch.delenv("GRACEFUL_STOP", raising=False)
    monkeypatch.chdir(tmp_path)  # the shared client writes truncation logs under the cwd
    for name in ("OUTPUT_DIRECTORY", "HOME", "USERPROFILE", "GLOSSARION_LIBRARY_DIR"):
        monkeypatch.setenv(name, str(tmp_path))
    made = []

    def start(**kwargs):
        service, engine = make_service(tmp_path, **kwargs)
        service.start()
        made.append(service)
        return service, engine

    yield start
    for service in made:
        service.stop()
    svc.set_default_service(None)


def _post(service, text, *, stream=False, timeout=30):
    conn = http.client.HTTPConnection("127.0.0.1", service.server.port, timeout=timeout)
    body = {"model": "fake-echo", "stream": stream, "messages": [{"role": "user", "content": text}]}
    conn.request("POST", f"/{service.server.token}/v1/chat/completions", body=json.dumps(body),
                 headers={"Host": f"127.0.0.1:{service.server.port}", "Content-Type": "application/json"})
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response.status, data


def _wait(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _long_reply(count):
    words = " ".join(f"w{i}" for i in range(count))
    return lambda messages: words


# ---- the endpoint stays reachable ------------------------------------------------------------------


def test_detach_keeps_the_endpoint_up_for_a_running_job(running):
    """Root cause of the "Connection error" retries: Android's ``detach`` (the activity destroyed, the
    process and the job's foreground service going on) stopped the endpoint under the running job."""
    job = {"model": "local/fake-echo"}
    service, engine = running(active_job_model=lambda: job["model"])
    url = service.server.base_url
    assert _post(service, "one")[0] == 200
    assert service.on_lifecycle("detach") == "detached"
    assert service.server.running and os.environ[svc.URL_ENV] == url and service.loaded_stem == "fake-echo"
    status, data = _post(service, "two")  # the job's next request
    assert status == 200 and "Echo: two" in data.decode()
    job["model"] = None
    service.on_lifecycle("detach")  # no job: the model is freed, the (idle) endpoint stays
    assert service.loaded_stem is None and service.server.running and svc.default_service() is service


def test_a_restarted_endpoint_keeps_its_address(running):
    service, _engine = running()
    url = service.server.base_url
    service.server.stop()
    service.server.start()
    assert service.server.base_url == url and _post(service, "again")[0] == 200


def test_the_client_reads_the_current_address_on_every_retry(running, monkeypatch, caplog):
    import unified_api_client as uac

    caplog.set_level(logging.INFO, logger="unified_api_client")
    service, _engine = running()
    good = service.server.base_url
    dead = socket.socket()
    dead.bind(("127.0.0.1", 0))
    dead_port = dead.getsockname()[1]
    dead.close()  # nothing listens there
    monkeypatch.setenv(svc.URL_ENV, f"http://127.0.0.1:{dead_port}/{service.server.token}/v1")
    monkeypatch.setenv("ENABLE_STREAMING", "1")
    monkeypatch.setenv("SEND_INTERVAL_SECONDS", "0.1")
    monkeypatch.setenv("MAX_RETRIES", "40")
    result = {}

    def ask():
        try:
            client = uac.UnifiedClient(api_key="", model="local/fake-echo")
            reply = client.send([{"role": "user", "content": "moved"}], temperature=0.3, max_tokens=32)
            result["text"] = reply[0] if isinstance(reply, tuple) else reply.content
        except Exception as exc:  # pragma: no cover - the failure being guarded against
            result["error"] = exc

    thread = threading.Thread(target=ask, daemon=True)
    thread.start()
    assert _wait(lambda: "Connection error" in caplog.text, 60)  # the old address failed
    os.environ[svc.URL_ENV] = good  # the endpoint published again (another app / a restart)
    thread.join(60)
    assert result.get("text", "").strip() == "Echo: moved", result


def test_concurrent_sdk_streams_queue_without_connection_errors(running):
    """Four manga workers at once, the first answer slow and the model loading: one generation at a
    time, every stream completes (the transport the shared client uses: one connection per client)."""
    import httpx
    import openai

    service, engine = running(engine=eng.FakeEngine(delay=0.01, load_delay=0.5, reply=_long_reply(60)))
    results = {}

    def ask(index):
        http_client = httpx.Client(timeout=httpx.Timeout(connect=10, read=None, write=None, pool=10),
                                   limits=httpx.Limits(max_connections=1, max_keepalive_connections=0,
                                                       keepalive_expiry=0))
        client = openai.OpenAI(api_key="dummy", base_url=service.server.base_url, max_retries=0,
                               http_client=http_client)
        try:
            chunks = list(client.chat.completions.create(model="fake-echo", stream=True,
                                                         messages=[{"role": "user", "content": f"q{index}"}]))
            results[index] = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
        except Exception as exc:
            results[index] = exc
        finally:
            http_client.close()

    threads = [threading.Thread(target=ask, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert all(isinstance(results.get(i), str) and results[i].startswith("w0 w1") for i in range(4)), results
    assert len(engine.loads) == 1 and len(engine.calls) == 4 and not service.gate.busy


# ---- progress lines -------------------------------------------------------------------------------------


@pytest.mark.parametrize("stream", [True, False])
def test_progress_lines_reach_the_client_log(running, monkeypatch, caplog, stream):
    import unified_api_client as uac

    caplog.set_level(logging.INFO, logger="unified_api_client")
    service, engine = running(engine=eng.FakeEngine(load_delay=0.3))
    monkeypatch.setenv("ENABLE_STREAMING", "1" if stream else "0")
    client = uac.UnifiedClient(api_key="", model="local/fake-echo")
    client.send([{"role": "user", "content": "first"}], temperature=0.3, max_tokens=32)
    out = caplog.text
    assert "📱 [local] On-device request: fake-echo" in out
    assert "⏳ Loading on-device model fake-echo" in out
    assert "✓ On-device model fake-echo loaded in 0." in out
    assert service.server.token not in out  # the endpoint's path token never reaches a log
    caplog.clear()
    client.send([{"role": "user", "content": "second"}], temperature=0.3, max_tokens=32)
    out = caplog.text
    assert "On-device request: fake-echo" in out and "Loading on-device model" not in out


def test_a_queued_request_logs_that_it_waits(running, monkeypatch, caplog):
    import unified_api_client as uac

    caplog.set_level(logging.INFO, logger="unified_api_client")
    service, engine = running(engine=eng.FakeEngine(delay=0.02, reply=_long_reply(150)))
    monkeypatch.setenv("ENABLE_STREAMING", "0")
    client = uac.UnifiedClient(api_key="", model="local/fake-echo")
    assert _post(service, "warm")[0] == 200
    busy = threading.Thread(target=_post, args=(service, "slow"))
    busy.start()
    assert _wait(lambda: service.gate.busy)
    client.send([{"role": "user", "content": "next"}], temperature=0.3, max_tokens=500)
    busy.join(30)
    assert "⏳ Waiting for the on-device model… (1 request ahead)" in caplog.text
    assert [c["messages"][0]["content"] for c in engine.calls] == ["warm", "slow", "next"]


# ---- Stop -------------------------------------------------------------------------------------------------


def test_graceful_stop_drops_queued_requests_and_finishes_the_running_answer(running, monkeypatch):
    service, engine = running(engine=eng.FakeEngine(delay=0.01, reply=_long_reply(120)))
    results = {}

    def ask(name):
        results[name] = _post(service, name)

    first = threading.Thread(target=ask, args=("first",))
    first.start()
    assert _wait(lambda: service.gate.busy and engine.generated > 0)
    queued = [threading.Thread(target=ask, args=(name,)) for name in ("q1", "q2")]
    for thread in queued:
        thread.start()
    assert _wait(lambda: service.gate.waiting() == 2)
    monkeypatch.setenv("GRACEFUL_STOP", "1")
    for thread in queued:
        thread.join(5)
    assert [results[n][0] for n in ("q1", "q2")] == [499, 499]
    assert json.loads(results["q1"][1])["error"]["code"] == "client_closed"
    first.join(30)
    assert results["first"][0] == 200 and engine.generated == 120  # ran to its end
    assert len(engine.calls) == 1 and not service.gate.busy
    assert _post(service, "later")[0] == 499  # nothing new starts while the graceful stop runs
    monkeypatch.setenv("GRACEFUL_STOP", "0")
    assert _post(service, "after")[0] == 200


def test_immediate_stop_aborts_the_running_answer_and_the_queue(running, monkeypatch):
    import unified_api_client as uac

    service, engine = running(engine=eng.FakeEngine(delay=0.01, reply=_long_reply(2000)))
    results = {}

    def ask(name):
        results[name] = _post(service, name, stream=True)

    threads = [threading.Thread(target=ask, args=(name,)) for name in ("run", "queued")]
    threads[0].start()
    assert _wait(lambda: engine.generated > 10)
    threads[1].start()
    assert _wait(lambda: service.gate.waiting() == 1)
    monkeypatch.setattr(uac.UnifiedClient, "_global_cancelled", True)
    stopped_at = time.monotonic()
    assert _wait(lambda: not service.gate.busy and not service.gate.waiting(), 5)
    assert time.monotonic() - stopped_at < 1.5 and engine.generated < 400
    for thread in threads:
        thread.join(10)
    assert results["queued"][0] == 499 and len(engine.calls) == 1


def test_hard_cancel_all_frees_the_engine_at_once(running, monkeypatch):
    """The app's force stop (``UnifiedClient.hard_cancel_all``) through the real client: the running
    generation and the queued request end within a second and the engine is free for the next run."""
    import unified_api_client as uac

    levels = {name: logging.getLogger(name).level for name in ("httpx", "openai", "urllib3")}
    service, engine = running(engine=eng.FakeEngine(delay=0.01, reply=_long_reply(3000)))
    monkeypatch.setenv("ENABLE_STREAMING", "1")
    errors = []

    def ask(text):
        try:
            uac.UnifiedClient(api_key="", model="local/fake-echo").send(
                [{"role": "user", "content": text}], temperature=0.3, max_tokens=4000)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=ask, args=(t,), daemon=True) for t in ("a", "b")]
    try:
        for thread in threads:
            thread.start()
        assert _wait(lambda: engine.generated > 20 and service.gate.waiting() == 1, 15)
        uac.UnifiedClient.hard_cancel_all()
        stopped_at = time.monotonic()
        assert _wait(lambda: not service.gate.busy and not service.gate.waiting(), 5)
        assert time.monotonic() - stopped_at < 1.5
        for thread in threads:
            thread.join(10)
        assert len(errors) == 2 and len(engine.calls) == 1
    finally:
        uac.UnifiedClient.set_global_cancellation(False)
        uac.set_stop_flag(False)
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)
    engine.reply = None  # a short answer: the engine serves the next run
    assert _post(service, "next")[0] == 200


def test_a_queued_request_cancelled_on_stop_is_not_retried(running, monkeypatch):
    """A 499 from the on-device endpoint ends the request (cancelled); it is never sent again."""
    import unified_api_client as uac

    service, engine = running()
    asked = []
    chat = service.chat

    def counting(*args, **kwargs):
        asked.append(1)
        return chat(*args, **kwargs)

    monkeypatch.setattr(service, "chat", counting)
    monkeypatch.setattr(service, "stop_state", lambda: "graceful")  # the server side of a Stop only
    monkeypatch.setenv("ENABLE_STREAMING", "0")
    monkeypatch.setenv("SEND_INTERVAL_SECONDS", "0.1")
    client = uac.UnifiedClient(api_key="", model="local/fake-echo")
    with pytest.raises(uac.UnifiedClientError) as info:
        client.send([{"role": "user", "content": "x"}], temperature=0.3, max_tokens=32)
    assert "cancel" in str(info.value).lower() and asked == [1] and not engine.calls


# ---- model switch -------------------------------------------------------------------------------------------


def _chat(service, model, text="hi"):
    with service.chat(model, {"messages": [{"role": "user", "content": text}]}, lambda: False) as run:
        return "".join(run.deltas)


def test_selecting_another_model_unloads_the_loaded_one(tmp_path):
    job = {"model": None}
    service, engine = make_service(tmp_path, stems=("a", "b"), active_job_model=lambda: job["model"])
    _chat(service, "local/a")
    assert service.selected_model_changed("local/A") is None and service.loaded_stem == "a"
    assert service.selected_model_changed("local/b") == "unloaded" and service.loaded_stem is None
    _chat(service, "local/a")
    assert service.selected_model_changed("gpt-6") == "unloaded" and engine.unloads == 2
    _chat(service, "local/a")
    job["model"] = "local/a"  # a running job keeps the model it uses
    assert service.selected_model_changed("gpt-6") is None and service.loaded_stem == "a"


def test_a_switch_during_an_answer_unloads_after_it(tmp_path):
    service, engine = make_service(tmp_path, stems=("a",))
    run = service.chat("local/a", {"messages": [{"role": "user", "content": "one two three"}]}, lambda: False)
    assert service.selected_model_changed("gpt-6") == "deferred" and service.loaded_stem == "a"
    assert "".join(run.deltas) == "Echo: one two three"  # the answer is not cut short
    run.close()
    assert service.loaded_stem is None and engine.unloads == 1
    # switching back before the answer ends cancels the pending unload
    run = service.chat("local/a", {"messages": [{"role": "user", "content": "x"}]}, lambda: False)
    service.selected_model_changed("gpt-6")
    service.selected_model_changed("local/a")
    run.close()
    assert service.loaded_stem == "a"
