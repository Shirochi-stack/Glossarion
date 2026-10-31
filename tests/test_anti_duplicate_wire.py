"""The anti-duplicate settings as they reach a custom OpenAI-compatible endpoint: a real UnifiedClient
against a local fake server, read from the request body, for translation and glossary."""
import http.server
import json
import os
import socket
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


class _Server(http.server.ThreadingHTTPServer):
    bodies: list


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.server.bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        raw = json.dumps({"id": "x", "object": "chat.completion", "model": "m", "created": 1,
                          "choices": [{"index": 0, "finish_reason": "stop",
                                       "message": {"role": "assistant", "content": "Hello there."}}],
                          "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


_LAST = {}
CLOUD_HOST = "my-router.example.test"
TRANSLATION = {"ENABLE_ANTI_DUPLICATE": "1", "TOP_P": "0.9", "TOP_K": "40", "MIN_P": "0.05",
               "FREQUENCY_PENALTY": "0.2", "PRESENCE_PENALTY": "0.3", "REPETITION_PENALTY": "1.1"}
GLOSSARY = {"GLOSSARY_" + key: value for key, value in {
    "ENABLE_ANTI_DUPLICATE": "1", "TOP_P": "0.8", "TOP_K": "30", "MIN_P": "0.04",
    "FREQUENCY_PENALTY": "0.1", "PRESENCE_PENALTY": "0.15", "REPETITION_PENALTY": "1.2"}.items()}


@pytest.fixture
def endpoint(monkeypatch, tmp_path):
    for name in ("HOME", "USERPROFILE", "OUTPUT_DIRECTORY", "GLOSSARION_LIBRARY_DIR"):
        monkeypatch.setenv(name, str(tmp_path))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("GLOSSARY_") or key in TRANSLATION or key == "BYPASS_MIN_P_ALLOWLIST":
            monkeypatch.delenv(key)
    server = _Server(("127.0.0.1", 0), _Handler)
    server.bodies = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    real_getaddrinfo = socket.getaddrinfo

    def resolve(host, *args, **kwargs):  # a cloud-looking URL that still lands on the fake server
        return real_getaddrinfo("127.0.0.1" if host == CLOUD_HOST else host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setenv("USE_CUSTOM_OPENAI_ENDPOINT", "1")
    yield server
    server.shutdown()


def _send(server, monkeypatch, tmp_path, host, context, env, model="gemma-4-26b-a4b-styletune-v2-qat_q4_k_xl"):
    from unified_api_client import UnifiedClient

    monkeypatch.setenv("OPENAI_CUSTOM_BASE_URL", f"http://{host}:{server.server_port}/v1")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    client = UnifiedClient(model=model, api_key="sk-test", output_dir=str(tmp_path))
    client.context = context
    _LAST["client"] = client
    client.send([{"role": "user", "content": "hi"}], temperature=0.8, max_tokens=100, context=context)
    return server.bodies[-1]


@pytest.mark.parametrize("host", ["127.0.0.1", CLOUD_HOST])
def test_translation_settings_reach_the_endpoint(endpoint, monkeypatch, tmp_path, host):
    body = _send(endpoint, monkeypatch, tmp_path, host, "translation", TRANSLATION)
    assert (body["top_p"], body["top_k"], body["min_p"]) == (0.9, 40, 0.05)
    assert (body["frequency_penalty"], body["presence_penalty"]) == (0.2, 0.3)
    assert body["repetition_penalty"] == 1.1
    assert ("repeat_penalty" in body) == (host != CLOUD_HOST)


@pytest.mark.parametrize("host", ["127.0.0.1", CLOUD_HOST])
def test_glossary_uses_its_own_settings(endpoint, monkeypatch, tmp_path, host):
    body = _send(endpoint, monkeypatch, tmp_path, host, "glossary", {**TRANSLATION, **GLOSSARY})
    assert (body["top_p"], body["top_k"], body["min_p"]) == (0.8, 30, 0.04)
    assert (body["frequency_penalty"], body["presence_penalty"]) == (0.1, 0.15)
    assert body["repetition_penalty"] == 1.2


@pytest.mark.parametrize("host", ["127.0.0.1", CLOUD_HOST])
@pytest.mark.parametrize("context,env,prefix", [("translation", TRANSLATION, ""),
                                                ("glossary", {**TRANSLATION, **GLOSSARY}, "GLOSSARY_")])
def test_saved_payload_lists_what_was_sent(endpoint, monkeypatch, tmp_path, host, context, env, prefix):
    body = _send(endpoint, monkeypatch, tmp_path, host, context, env)
    with open(_LAST["client"]._get_thread_local_client().last_payload_path, encoding="utf-8") as f:
        payload = json.load(f)
    wire = {key: body[key] for key in ("top_p", "top_k", "min_p", "frequency_penalty", "presence_penalty",
                                       "repetition_penalty", "repeat_penalty") if key in body}
    assert wire["top_k"] == int(env[prefix + "TOP_K"])
    assert payload["anti_duplicate"]["sent"] == wire
    assert {key: payload["request_params"][key] for key in wire} == wire


def test_top_p_of_one_is_sent_so_the_server_preset_does_not_apply(endpoint, monkeypatch, tmp_path):
    body = _send(endpoint, monkeypatch, tmp_path, "127.0.0.1", "translation", {**TRANSLATION, "TOP_P": "1.0"})
    assert body["top_p"] == 1.0


def test_sliders_at_off_leave_the_server_settings_alone(endpoint, monkeypatch, tmp_path):
    off = {"ENABLE_ANTI_DUPLICATE": "1", "TOP_P": "0.0", "TOP_K": "0", "MIN_P": "0.0", "FREQUENCY_PENALTY": "0.0",
           "PRESENCE_PENALTY": "0.0", "REPETITION_PENALTY": "1.0"}
    body = _send(endpoint, monkeypatch, tmp_path, "127.0.0.1", "translation", off)
    for key in ("top_p", "top_k", "min_p", "frequency_penalty", "presence_penalty", "repetition_penalty",
                "repeat_penalty"):
        assert key not in body


def test_toggle_off_sends_none_of_them(endpoint, monkeypatch, tmp_path):
    body = _send(endpoint, monkeypatch, tmp_path, "127.0.0.1", "translation",
                 {**TRANSLATION, "ENABLE_ANTI_DUPLICATE": "0"})
    for key in ("top_p", "top_k", "min_p", "frequency_penalty", "presence_penalty", "repetition_penalty"):
        assert key not in body


def test_log_never_claims_what_was_not_sent(endpoint, monkeypatch, tmp_path, caplog):
    import logging

    from unified_api_client import UnifiedClient

    for key, value in TRANSLATION.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("USE_CUSTOM_OPENAI_ENDPOINT", "0")
    client = UnifiedClient(model="gpt-4o", api_key="sk-test", output_dir=str(tmp_path))
    with caplog.at_level(logging.INFO):
        client._log_anti_duplicate_applied({"top_p": 0.9})
    assert "applied: top_p=0.9" in caplog.text
    assert "top_k=40" in caplog.text.split("not sent on this route", 1)[1]
    assert "top_k" not in caplog.text.split("not sent on this route", 1)[0]
    assert client._get_anti_duplicate_endpoint_params("openai") == {}  # official OpenAI rejects top_k
