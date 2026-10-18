"""The on-device endpoint (``services/local_llm/server.py`` + ``service.LocalEngineService``) with
``FakeEngine``: the token and Host guards, ``/v1/models``, plain and SSE answers through the real
``openai`` SDK and through ``UnifiedClient`` with ``local/<stem>`` (the shared route end to end), the
error answers (404 / 400 / 409 / 503), stop on client disconnect, ``<think>`` stripping and
``enable_thinking``. No network, no model.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import threading
import time

import pytest

import local_llm_fakes as fakes
from glossarion_mobile.services.local_llm import downloads as dl
from glossarion_mobile.services.local_llm import engine as eng
from glossarion_mobile.services.local_llm import service as svc
from glossarion_mobile.services.local_llm.server import ThinkStripper


def make_service(tmp_path, stems=("fake-echo",), engine=None, prefs=None, **kwargs):
    registry = dl.Registry(str(tmp_path / "models"))
    os.makedirs(registry.root, exist_ok=True)
    for stem in stems:
        with open(registry.file_path(stem), "wb") as handle:
            handle.write(b"GGUF")
        registry.add(stem, {"repo": f"t/{stem}", "file": f"{stem}.gguf", "revision": "r", "sha256": "0" * 64,
                            "size": 4, "downloaded_at": 1})
    engine = engine or eng.FakeEngine()
    kwargs.setdefault("memory", lambda: (0, None))
    service = svc.LocalEngineService(prefs=prefs or fakes.FakePrefs(), registry=registry,
                                     engine_factory=lambda: engine, watch=False, platform="desktop", **kwargs)
    return service, engine


@pytest.fixture()
def running(tmp_path, monkeypatch):
    monkeypatch.delenv(svc.URL_ENV, raising=False)
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


def _raw(service, method, path, body=None, host=None):
    conn = http.client.HTTPConnection("127.0.0.1", service.server.port, timeout=10)
    headers = {"Host": host or f"127.0.0.1:{service.server.port}", "Content-Type": "application/json"}
    conn.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers)
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response.status, data


def test_token_and_host_guards(running):
    service, _engine = running()
    token = service.server.token
    assert _raw(service, "GET", f"/{token}/v1/models")[0] == 200
    assert _raw(service, "GET", "/v1/models")[0] == 404
    assert _raw(service, "GET", f"/{token[:-1]}x/v1/models")[0] == 404
    assert _raw(service, "GET", f"/{token}/v1/models", host="evil.example:80")[0] == 404
    assert _raw(service, "POST", f"/{token}/v1/embeddings", body={})[0] == 404
    assert _raw(service, "GET", f"/{token}/nothing")[0] == 404
    assert os.environ[svc.URL_ENV] == service.server.base_url and service.server.base_url.endswith(f"/{token}/v1")
    status, data = _raw(service, "GET", f"/{token}/health")
    assert status == 200 and json.loads(data) == {"state": "idle", "model": None, "queue": 0, "busy": False,
                                                  "load_seconds": None}
    status, data = _raw(service, "GET", f"/{token}/health?model=local/FAKE-ECHO")
    assert json.loads(data)["target"] == {"stem": "fake-echo", "size": 4, "loaded": False}


def test_models_lists_downloaded_files(running):
    service, _engine = running(stems=("Qwen3.5-0.8B-Q4_K_M", "fake-echo"))
    status, data = _raw(service, "GET", f"/{service.server.token}/v1/models")
    body = json.loads(data)
    assert status == 200 and body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["fake-echo", "Qwen3.5-0.8B-Q4_K_M"]
    assert body["data"][0]["owned_by"] == "local" and body["data"][0]["meta"]["loaded"] is False


def test_openai_sdk_plain_and_streaming(running):
    import openai

    service, engine = running()
    client = openai.OpenAI(base_url=service.server.base_url, api_key="ignored")
    answer = client.chat.completions.create(model="local/fake-echo", messages=[{"role": "user", "content": "hi there"}],
                                            temperature=0.1, max_tokens=50)
    assert answer.choices[0].message.content == "Echo: hi there" and answer.model == "local/fake-echo"
    assert answer.usage.completion_tokens == 3 and answer.usage.prompt_tokens > 0
    # on-device sampling replaces the request's temperature (owner 2026-10-18: Qwen's 0.7, not the request's 0.1)
    assert engine.calls[-1]["params"]["temperature"] == 0.7 and engine.calls[-1]["params"]["max_tokens"] == 50
    chunks = list(client.chat.completions.create(model="FAKE-ECHO", stream=True,
                                                 messages=[{"role": "user", "content": "a b c"}]))
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == "Echo: a b c" and chunks[-1].choices[0].finish_reason == "stop"
    assert service.snapshot().state == "ready" and service.snapshot().model == "fake-echo"


@pytest.mark.parametrize("stream", [False, True])
def test_unified_client_reaches_the_engine_through_the_local_route(running, monkeypatch, stream):
    import unified_api_client as uac

    service, engine = running()
    monkeypatch.setenv("ENABLE_STREAMING", "1" if stream else "0")
    client = uac.UnifiedClient(api_key="", model="local/fake-echo")
    result = client.send([{"role": "system", "content": "Translate."}, {"role": "user", "content": "안녕 friend"}],
                         temperature=0.3, max_tokens=64)
    content = result[0] if isinstance(result, tuple) else result.content
    assert content.strip() == "Echo: 안녕 friend"
    assert engine.calls[-1]["messages"][0] == {"role": "system", "content": "Translate."}


def test_error_answers(running):
    service, engine = running(prefs=fakes.FakePrefs({svc.SETTINGS_KEY: {"n_ctx": 512}}))
    token = service.server.token
    status, data = _raw(service, "POST", f"/{token}/v1/chat/completions",
                        body={"model": "local/missing", "messages": [{"role": "user", "content": "x"}]})
    error = json.loads(data)["error"]
    assert status == 404 and error["code"] == "model_not_found"
    assert error["message"] == "local/missing is not downloaded. Download it in Model manager › On-device."
    status, data = _raw(service, "POST", f"/{token}/v1/chat/completions", body={
        "model": "fake-echo", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this"}, {"type": "image_url", "image_url": {"url": "data:,"}}]}]})
    assert status == 400 and json.loads(data)["error"]["message"] == "This on-device model has no vision support"
    status, data = _raw(service, "POST", f"/{token}/v1/chat/completions", body={
        "model": "fake-echo", "messages": [{"role": "user", "content": "word " * 600}]})
    error = json.loads(data)["error"]
    assert status == 400 and error["code"] == "context_length_exceeded"
    assert error["message"].startswith("This model's maximum context length is 512 tokens")
    # max_tokens is clamped to what the context leaves
    status, _ = _raw(service, "POST", f"/{token}/v1/chat/completions", body={
        "model": "fake-echo", "max_tokens": 100000, "messages": [{"role": "user", "content": "word " * 100}]})
    assert status == 200 and engine.calls[-1]["params"]["max_tokens"] == 512 - engine.count_prompt(
        [{"role": "user", "content": "word " * 100}])
    status, _ = _raw(service, "POST", f"/{token}/v1/chat/completions", body={"model": "fake-echo"})
    assert status == 400
    conn = http.client.HTTPConnection("127.0.0.1", service.server.port, timeout=10)
    conn.request("POST", f"/{token}/v1/chat/completions", body=b"not json",
                 headers={"Host": f"127.0.0.1:{service.server.port}"})
    assert conn.getresponse().status == 400


def test_a_running_job_pins_its_model(running):
    job = {"model": "local/fake-echo"}
    service, _engine = running(stems=("fake-echo", "other"), active_job_model=lambda: job["model"])
    token = service.server.token
    status, data = _raw(service, "POST", f"/{token}/v1/chat/completions",
                        body={"model": "local/other", "messages": [{"role": "user", "content": "x"}]})
    assert status == 409 and json.loads(data)["error"]["message"] == (
        "The on-device engine is busy with fake-echo for a running job")
    assert _raw(service, "POST", f"/{token}/v1/chat/completions",
                body={"model": "local/fake-echo", "messages": [{"role": "user", "content": "x"}]})[0] == 200
    job["model"] = "gpt-6"  # a cloud job pins nothing
    assert _raw(service, "POST", f"/{token}/v1/chat/completions",
                body={"model": "local/other", "messages": [{"role": "user", "content": "x"}]})[0] == 200


def test_requests_queue_in_order_and_overflow_gets_503(running):
    service, engine = running(engine=eng.FakeEngine(delay=0.05))
    service.gate.max_waiters = 1
    token = service.server.token
    results = {}

    def ask(name, delay):
        time.sleep(delay)
        results[name] = _raw(service, "POST", f"/{token}/v1/chat/completions",
                             body={"model": "fake-echo", "messages": [{"role": "user", "content": f"{name} one two"}]})

    threads = [threading.Thread(target=ask, args=(name, i * 0.05)) for i, name in enumerate(("a", "b", "c"))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(15)
    assert results["a"][0] == 200 and results["b"][0] == 200
    assert results["c"][0] == 503 and json.loads(results["c"][1])["error"]["message"].startswith("On-device model busy")
    assert [c["messages"][0]["content"] for c in engine.calls] == ["a one two", "b one two"]  # FIFO


def test_a_client_disconnect_stops_generation(running):
    reply = " ".join(f"w{i}" for i in range(400))
    service, engine = running(engine=eng.FakeEngine(delay=0.01, reply=lambda messages: reply))
    sock = socket.create_connection(("127.0.0.1", service.server.port), timeout=10)
    body = json.dumps({"model": "fake-echo", "stream": True, "messages": [{"role": "user", "content": "go"}]}).encode()
    sock.sendall(f"POST /{service.server.token}/v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{service.server.port}\r\n"
                 f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)
    received = b""
    while received.count(b"data: ") < 5:
        received += sock.recv(4096)
    sock.close()
    closed_at = time.monotonic()
    while service.gate.busy and time.monotonic() - closed_at < 5:
        time.sleep(0.01)
    assert not service.gate.busy and time.monotonic() - closed_at < 1.0
    assert engine.generated < 150, engine.generated  # stopped within a few tokens, not 400
    assert service.snapshot().state == "ready"


def test_think_blocks_are_stripped_unless_thinking_is_asked_for(running):
    import openai

    service, engine = running(engine=eng.FakeEngine(think=True))
    client = openai.OpenAI(base_url=service.server.base_url, api_key="x")
    plain = client.chat.completions.create(model="fake-echo", messages=[{"role": "user", "content": "hello"}])
    assert plain.choices[0].message.content == "Echo: hello"
    assert engine.calls[-1]["params"]["enable_thinking"] is False
    chunks = client.chat.completions.create(model="fake-echo", stream=True, messages=[{"role": "user", "content": "x y"}])
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == "Echo: x y"
    thinking = client.chat.completions.create(model="fake-echo", messages=[{"role": "user", "content": "hm"}],
                                              extra_body={"chat_template_kwargs": {"enable_thinking": True}})
    assert thinking.choices[0].message.content.startswith("<think>")
    assert engine.calls[-1]["params"]["enable_thinking"] is True


def test_think_stripper_handles_split_tags():
    stripper = ThinkStripper()
    pieces = ["  <thi", "nk>\nlong ", "reasoning</th", "ink>", "\n\n", "Hello", " world"]
    assert "".join(stripper.feed(p) for p in pieces) + stripper.flush() == "Hello world"
    plain = ThinkStripper()
    assert "".join(plain.feed(p) for p in ["<b>", "bold"]) + plain.flush() == "<b>bold"
    early = ThinkStripper()
    assert early.feed("<thi") == "" and early.flush() == "<thi"
