"""The shared keyless ``local/`` route (a llama.cpp-style OpenAI-compatible server: llama-server on the
desktop, Glossarion Mobile's on-device engine): route resolution, the GLOSSARION_LOCAL_LLM_URL override,
the no-key rule, the catalog spec and that ``local/`` is stripped from the request model."""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import model_options  # noqa: E402
import unified_api_client as uac  # noqa: E402

UC = uac.UnifiedClient


def test_route_resolution_and_env_override(monkeypatch):
    monkeypatch.delenv("GLOSSARION_LOCAL_LLM_URL", raising=False)
    assert UC._local_openai_route("local/Qwen3.5-2B-Q4_K_M") == ("local/", "local", "http://127.0.0.1:8080/v1")
    assert UC._local_openai_route("LOCAL/x") == ("local/", "local", "http://127.0.0.1:8080/v1")
    monkeypatch.setenv("GLOSSARION_LOCAL_LLM_URL", "http://127.0.0.1:5555/tok/v1/")
    assert UC._local_openai_route("local/x")[2] == "http://127.0.0.1:5555/tok/v1"
    assert UC._local_openai_route("ollama/x")[2] == "http://localhost:11434/v1"  # others ignore the variable
    assert UC._local_openai_route("localmodel") is None and UC._local_openai_route("gpt-6") is None
    assert UC.MODEL_PROVIDERS["local/"] == "local" and "local" in UC._LOCAL_OPENAI_PROVIDERS


def test_only_the_published_on_device_endpoint_is_treated_as_it(monkeypatch):
    # the long read-timeout floor and /health preflight apply to Glossarion Mobile's endpoint only:
    # a desktop ollama / lmstudio / llama-server keeps the configured CHUNK_TIMEOUT
    monkeypatch.delenv("GLOSSARION_LOCAL_LLM_URL", raising=False)
    assert UC._is_on_device_endpoint("http://127.0.0.1:8080/v1") is False
    monkeypatch.setenv("GLOSSARION_LOCAL_LLM_URL", "http://127.0.0.1:5555/tok/v1/")
    assert UC._is_on_device_endpoint("http://127.0.0.1:5555/tok/v1") is True
    assert UC._is_on_device_endpoint("http://localhost:11434/v1") is False


def test_local_models_need_no_key():
    assert UC._model_needs_api_key("local/Qwen3.5-0.8B-Q4_K_M") is False
    assert "local/" in UC._NO_API_KEY_PREFIXES


def test_catalog_spec_and_prefix_map(monkeypatch):
    spec = next(s for s in model_options.PROVIDER_CATALOG_SPECS if s.name == "local")
    assert spec.prefix == "local/" and spec.public and spec.base_url_env == "GLOSSARION_LOCAL_LLM_URL"
    monkeypatch.delenv("GLOSSARION_LOCAL_LLM_URL", raising=False)
    assert model_options._provider_models_url(spec) == "http://127.0.0.1:8080/v1/models"
    monkeypatch.setenv("GLOSSARION_LOCAL_LLM_URL", "http://127.0.0.1:5555/tok/v1")
    assert model_options._provider_models_url(spec) == "http://127.0.0.1:5555/tok/v1/models"
    assert model_options.catalog_provider_for_model("local/x") == "local"
    # no static local/ suggestions: the desktop model list is unchanged
    assert not [m for m in model_options._get_static_model_options() if str(m).startswith("local/")]


def test_local_catalog_available_probes_the_override_port(monkeypatch):
    import socket

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        monkeypatch.setenv("GLOSSARION_LOCAL_LLM_URL", f"http://127.0.0.1:{port}/tok/v1")
        assert model_options.local_catalog_available("local") is True
    finally:
        listener.close()
    monkeypatch.setenv("GLOSSARION_LOCAL_LLM_URL", f"http://127.0.0.1:{port}/tok/v1")
    assert model_options.local_catalog_available("local") is False


@pytest.fixture()
def fake_llama_server(monkeypatch):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
            out = json.dumps({"id": "x", "object": "chat.completion", "created": 1, "model": body["model"],
                              "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"},
                                           "finish_reason": "stop"}],
                              "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("GLOSSARION_LOCAL_LLM_URL", f"http://127.0.0.1:{server.server_address[1]}/secret/v1")
    monkeypatch.setenv("ENABLE_STREAMING", "0")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-cloud-key-never-sent")
    yield seen
    server.shutdown()
    server.server_close()


def test_requests_strip_the_prefix_and_never_send_a_cloud_key(fake_llama_server):
    client = UC(api_key="sk-cloud-key-never-sent", model="local/Qwen3.5-2B-Q4_K_M")
    result = client.send([{"role": "user", "content": "ping"}], temperature=0.1, max_tokens=8)
    content = result[0] if isinstance(result, tuple) else result.content
    assert content == "pong"
    request = fake_llama_server[-1]
    assert request["path"] == "/secret/v1/chat/completions"
    assert request["body"]["model"] == "Qwen3.5-2B-Q4_K_M"
    assert "sk-cloud-key-never-sent" not in str(request["auth"])

def _send_local(context=None):
    client = UC(api_key="", model="local/Qwen3.5-0.8B-UD-Q4_K_XL")
    result = client.send([{"role": "user", "content": "ping"}], temperature=0.1, max_tokens=8, context=context)
    content = result[0] if isinstance(result, tuple) else result.content
    assert content == "pong"


def test_on_device_requests_say_what_they_are_for(fake_llama_server, monkeypatch):
    """Glossarion Mobile's on-device engine bans CJK tokens in translations (mobile local_llm.script_ban):
    it gets the request kind, the target language and the single-pass auto glossary flag."""
    monkeypatch.setenv("OUTPUT_LANGUAGE", "English")
    monkeypatch.setenv("AUTO_GLOSSARY_MODE", "balanced")
    monkeypatch.delenv("SINGLE_PASS_GLOSSARY_MODE", raising=False)
    _send_local("translation")
    body = fake_llama_server[-1]["body"]
    assert body["glossarion_task"] == "translation" and body["target_language"] == "English"
    assert body["glossarion_single_pass"] is False
    _send_local("auto glossary (3/10)")  # GlossaryManager's progress label
    assert "glossary" in fake_llama_server[-1]["body"]["glossarion_task"]
    _send_local("glossary")
    assert fake_llama_server[-1]["body"]["glossarion_task"] == "glossary"
    monkeypatch.setenv("AUTO_GLOSSARY_MODE", "single_pass")
    _send_local("translation")
    assert fake_llama_server[-1]["body"]["glossarion_single_pass"] is True
    monkeypatch.setenv("AUTO_GLOSSARY_MODE", "off")
    monkeypatch.setenv("SINGLE_PASS_GLOSSARY_MODE", "1")
    _send_local()
    assert fake_llama_server[-1]["body"]["glossarion_single_pass"] is True
    monkeypatch.delenv("OUTPUT_LANGUAGE")
    _send_local("metadata")
    body = fake_llama_server[-1]["body"]
    assert body["glossarion_task"] == "metadata" and "target_language" not in body


def test_other_local_endpoints_get_no_glossarion_fields(fake_llama_server, monkeypatch):
    """A desktop llama-server / ollama / lmstudio endpoint (anything but the published on-device one)
    gets the request exactly as before."""
    monkeypatch.setenv("OUTPUT_LANGUAGE", "English")
    monkeypatch.setattr(UC, "_is_on_device_endpoint", classmethod(lambda cls, base_url: False))
    _send_local("translation")
    body = fake_llama_server[-1]["body"]
    assert not any(key in body for key in ("glossarion_task", "glossarion_single_pass", "target_language"))
