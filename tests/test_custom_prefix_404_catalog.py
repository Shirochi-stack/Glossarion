"""A 404 on a custom prefix leaves the client's retry loop at once (so custom_endpoint_discovery runs),
and the custom prefix model catalog is polled for every endpoint type with the discovery candidate bases
as fallback. Fake local HTTP only."""
import json
import os
import sys
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import custom_endpoint_discovery as ced  # noqa: E402
import model_options  # noqa: E402
import unified_api_client as uac  # noqa: E402

KEY = "test-key-not-real"
UC = uac.UnifiedClient


# ------------------------------------------------------------------ the request path (real client)

@pytest.fixture()
def server(monkeypatch):
    """/chat/completions 404s; /v1/chat/completions exists (401 without key, 200 with it)."""
    state = {"requests": [], "good": True, "messages": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _reply(self, status, payload):
            out = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            auth = bool(self.headers.get("Authorization") or self.headers.get("x-api-key"))
            state["requests"].append(("POST", self.path, auth))
            if self.path == "/v1/messages" and state["messages"]:
                if not auth:
                    return self._reply(401, {"error": "no key"})
                return self._reply(200, {
                    "id": "m", "type": "message", "role": "assistant", "model": "x",
                    "content": [{"type": "text", "text": "pong"}], "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1}})
            if self.path == "/v1/chat/completions" and state["good"]:
                if not auth:
                    return self._reply(401, {"error": "no key"})
                model = json.loads(body or b"{}").get("model", "")
                return self._reply(200, {
                    "id": "x", "object": "chat.completion", "created": 1, "model": model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
            self._reply(404, {"error": "Not Found"})

        def do_GET(self):
            state["requests"].append(("GET", self.path, bool(self.headers.get("Authorization"))))
            if self.path == "/v1/models" and (state["good"] or state["messages"]):
                return self._reply(200, {"data": [{"id": "m1"}]})
            self._reply(404, {"error": "Not Found"})

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    state["base"] = f"http://127.0.0.1:{httpd.server_address[1]}"
    monkeypatch.setenv("CUSTOM_OPENAI_PREFIX_ROUTES", json.dumps(
        [{"prefix": "graf/", "routing": state["base"], "endpoint_type": "/chat/completions"}]))
    monkeypatch.setenv("ENABLE_STREAMING", "0")
    monkeypatch.setenv("SEND_INTERVAL_SECONDS", "0.01")
    monkeypatch.setattr(ced, "persist_route_to_config_file", lambda *a, **k: False)
    ced.reset_session()
    yield state
    ced.reset_session()
    httpd.shutdown()
    httpd.server_close()


def _chat_posts(state, path):
    return [r for r in state["requests"] if r[0] == "POST" and r[1] == path and r[2]]


def test_404_leaves_the_retry_loop_and_one_retry_runs_on_the_found_base(server):
    client = UC(api_key=KEY, model="graf/some-model")
    result = client.send([{"role": "user", "content": "ping"}], temperature=0.1, max_tokens=8)
    assert (result[0] if isinstance(result, tuple) else result.content) == "pong"
    # exactly one request on the wrong URL (no SDK retries), one on the found base
    assert len(_chat_posts(server, "/chat/completions")) == 1
    assert len(_chat_posts(server, "/v1/chat/completions")) == 1
    # the key reached only the confirmation GET and the real request, never a probe
    assert [r for r in server["requests"] if r[0] == "POST" and r[1] == "/v1/chat/completions" and not r[2]]


def test_a_404_on_chat_completions_switches_the_route_to_v1_messages(server, monkeypatch):
    server["good"] = False
    server["messages"] = True
    saved = []
    ced.register_route_listener(lambda *args, **kw: saved.append((args, kw)) or True)
    try:
        client = UC(api_key=KEY, model="graf/some-model")
        result = client.send([{"role": "user", "content": "ping"}], temperature=0.1, max_tokens=8)
    finally:
        ced._listeners.clear()
    assert (result[0] if isinstance(result, tuple) else result.content) == "pong"
    assert len(_chat_posts(server, "/chat/completions")) == 1  # the failed request is not retried
    assert len(_chat_posts(server, "/v1/messages")) == 1  # one retry, on the new type
    routes = json.loads(os.environ["CUSTOM_OPENAI_PREFIX_ROUTES"])
    assert routes[0]["endpoint_type"] == "/v1/messages" and routes[0]["routing"] == server["base"]
    (args, kw), = saved
    assert args == ("graf/", server["base"], server["base"])
    assert kw == {"endpoint_type": "/v1/messages", "old_endpoint_type": "/chat/completions"}


def test_endpoint_type_found_by_discovery_is_kept_with_the_old_value():
    config = {"custom_prefix_routes": [{"prefix": "graf/", "routing": "https://s.example.net",
                                        "endpoint_type": "/chat/completions"}]}
    assert ced.apply_route_to_config(config, "graf/", "https://s.example.net", "https://s.example.net",
                                     "/v1/messages", "/chat/completions")
    assert config["custom_prefix_routes"][0]["endpoint_type"] == "/v1/messages"
    assert config[ced.PREVIOUS_ENDPOINT_TYPE_KEY] == {"graf/": "/chat/completions"}
    assert ced.PREVIOUS_ROUTING_KEY not in config  # the URL did not change


def test_image_ocr_and_fal_routes_are_never_switched_to_a_text_type():
    probed = []
    result = ced.discover_endpoint("https://s.example.net", "/images/generations", KEY,
                                   probe=lambda url: probed.append(url) or 404)
    assert result.found is None and result.found_endpoint is None
    assert not any(url.endswith(("/v1/messages", "/chat/completions")) for url in probed)


def test_no_endpoint_found_fails_once_with_a_clear_error(server, capsys):
    server["good"] = False
    client = UC(api_key=KEY, model="graf/some-model")
    try:
        result = client.send([{"role": "user", "content": "ping"}], temperature=0.1, max_tokens=8)
        finish = result[1] if isinstance(result, tuple) else getattr(result, "finish_reason", "")
        assert finish == "error"
    except Exception as exc:  # the same failure raised instead of returned
        assert "graf/" in str(exc)
    out = capsys.readouterr().out
    assert "graf/" in out and server["base"] + "/chat/completions" in out and "404" in out
    assert "auto-retrying" not in out  # no global backoff loop on the same 404
    assert len(_chat_posts(server, "/chat/completions")) == 1  # one request, no SDK retries


def test_the_not_found_error_is_only_for_a_404_on_a_custom_prefix(monkeypatch):
    monkeypatch.setenv("CUSTOM_OPENAI_PREFIX_ROUTES", json.dumps(
        [{"prefix": "graf/", "routing": "https://status.example.net", "endpoint_type": "/chat/completions"}]))
    client = object.__new__(UC)
    client.model = "graf/x"
    error = client._custom_prefix_not_found_error("custom_openai", 404)
    assert error.http_status == 404 and "https://status.example.net/chat/completions" in str(error)
    for status in (400, 401, 429, 500, 502, 504):
        assert client._custom_prefix_not_found_error("custom_openai", status) is None
    assert client._custom_prefix_not_found_error("openai", 404) is None
    client.model = "unrelated-model"
    assert client._custom_prefix_not_found_error("custom_openai", 404) is None


# ------------------------------------------------------------------ catalog polling

def _specs(endpoint_type, routing="https://api.example.net/v1", model="graf/x"):
    return model_options._custom_catalog_specs(
        [{"prefix": "graf/", "routing": routing, "endpoint_type": endpoint_type}], model)


@pytest.mark.parametrize("endpoint_type,routing,url,style", [
    ("/chat/completions", "https://h.example.net/v1", "https://h.example.net/v1/models", "bearer"),
    ("openai_chat", "https://h.example.net", "https://h.example.net/models", "bearer"),
    ("/images/generations", "https://h.example.net/v1", "https://h.example.net/v1/models", "bearer"),
    ("/v1/messages", "https://h.example.net", "https://h.example.net/v1/models?limit=1000", "anthropic"),
    ("/v1/messages", "https://h.example.net/v1", "https://h.example.net/v1/models?limit=1000", "anthropic"),
    ("/v1/ocr", "https://h.example.net", "https://h.example.net/v1/models", "bearer"),
    ("/custom/path", "https://h.example.net", "https://h.example.net/models", "bearer"),
    ("/chat/completions", "https://generativelanguage.googleapis.com/v1beta",
     "https://generativelanguage.googleapis.com/v1beta/models", "query"),
])
def test_every_endpoint_type_gets_its_own_catalog_shape(endpoint_type, routing, url, style):
    (spec,) = _specs(endpoint_type, routing)
    assert spec.models_url == url and spec.auth_style == style


def test_fal_style_routes_have_no_catalog_and_other_prefixes_are_not_polled():
    assert _specs("/{model_id}") == []
    assert _specs("/chat/completions", model="other/x") == []


def test_fallback_urls_are_the_discovery_candidates():
    (spec,) = _specs("/chat/completions", "https://status.example.net")
    expected = [f"{c}/models" for c in ced.candidate_bases("https://status.example.net", "/chat/completions")]
    assert list(spec.fallback_urls) == expected and expected


def _fake_get(table, calls):
    def fake(url, headers, timeout):
        calls.append((url, dict(headers)))
        answer = table.get(url)
        if isinstance(answer, BaseException):
            raise answer
        if answer is None:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return answer
    return fake


def test_catalog_falls_back_to_the_first_candidate_that_returns_models(monkeypatch):
    (spec,) = _specs("/chat/completions", "https://status.example.net")
    first, second = spec.fallback_urls[0], spec.fallback_urls[1]
    calls = []
    monkeypatch.setattr(model_options, "_http_get_json", _fake_get(
        {first: ValueError("not json"), second: {"data": [{"id": "a"}, {"id": "b"}]}}, calls))
    models = model_options._fetch_provider_catalog(spec, KEY, 1)
    assert models == ["graf/a", "graf/b"]
    assert [c[0] for c in calls] == [spec.models_url, first, second]  # stops at the first catalog


def test_unreachable_route_falls_back_and_all_failing_raises_the_first_error(monkeypatch):
    (spec,) = _specs("/chat/completions", "https://status.example.net")
    calls = []
    monkeypatch.setattr(model_options, "_http_get_json", _fake_get(
        {spec.models_url: urllib.error.URLError("refused")}, calls))
    with pytest.raises(urllib.error.URLError):
        model_options._fetch_provider_catalog(spec, KEY, 1)
    assert len(calls) == 1 + len(spec.fallback_urls)


def test_a_rejected_key_does_not_search_other_bases(monkeypatch):
    (spec,) = _specs("/chat/completions", "https://status.example.net")
    calls = []
    err = urllib.error.HTTPError(spec.models_url, 401, "Unauthorized", {}, None)
    monkeypatch.setattr(model_options, "_http_get_json", _fake_get({spec.models_url: err}, calls))
    with pytest.raises(urllib.error.HTTPError):
        model_options._fetch_provider_catalog(spec, KEY, 1)
    assert len(calls) == 1  # the key is not sent to any other URL after a rejection


def test_anthropic_style_route_sends_x_api_key_and_gemini_style_names_are_stripped(monkeypatch):
    (spec,) = _specs("/v1/messages", "https://h.example.net")
    calls = []
    monkeypatch.setattr(model_options, "_http_get_json", _fake_get(
        {spec.models_url: {"data": [{"id": "claude-x"}]}}, calls))
    assert model_options._fetch_provider_catalog(spec, KEY, 1) == ["graf/claude-x"]
    assert calls[0][1]["x-api-key"] == KEY and "Authorization" not in calls[0][1]

    (spec,) = _specs("/chat/completions", "https://generativelanguage.googleapis.com/v1beta")
    calls = []
    monkeypatch.setattr(model_options, "_http_get_json", lambda url, headers, timeout: (
        calls.append(url) or {"models": [{"name": "models/gem-1"}]}))
    assert model_options._fetch_provider_catalog(spec, KEY, 1) == ["graf/gem-1"]
    assert calls[0].endswith("/models?key=" + KEY)


def test_refresh_polls_a_non_chat_route_and_reports_its_status(tmp_path, monkeypatch):
    monkeypatch.setenv("GLOSSARION_MODEL_CATALOG_CACHE", str(tmp_path / "c.json"))
    monkeypatch.setattr(model_options, "_model_catalog_cache_path", lambda: str(tmp_path / "c.json"))
    seen = []

    def fake_get(url, headers, timeout):
        seen.append(url)
        if url == "https://h.example.net/v1/models?limit=1000":
            return {"data": [{"id": "claude-x"}]}
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(model_options, "_http_get_json", fake_get)
    result = model_options.refresh_provider_model_catalogs(
        active_model="graf/claude-x", active_api_key=KEY, timeout=0.1, only_provider="custom:graf/",
        custom_routes=[{"prefix": "graf/", "routing": "https://h.example.net", "endpoint_type": "/v1/messages"}])
    assert result.statuses["custom:graf/"] == "online (1 models)"
    assert result.provider_models["custom:graf/"] == ["graf/claude-x"]
    # the route's own key rule is unchanged: no key, no poll
    result = model_options.refresh_provider_model_catalogs(
        active_model="graf/claude-x", active_api_key="", timeout=0.1, only_provider="custom:graf/",
        custom_routes=[{"prefix": "graf/", "routing": "https://h.example.net", "endpoint_type": "/v1/messages"}])
    assert result.statuses["custom:graf/"] == "static fallback (no provider credential)"


# ------------------------------------------------------------------ only custom prefix routes are affected

class _Resp:
    status_code = 404
    text = "Not Found"
    headers = {}
    url = "https://api.openai.com/v1/chat/completions"

    def json(self):
        return {"error": "Not Found"}

    def close(self):
        pass


def test_a_built_in_provider_404_keeps_its_retry_behaviour(monkeypatch):
    monkeypatch.setenv("CUSTOM_OPENAI_PREFIX_ROUTES", json.dumps(
        [{"prefix": "graf/", "routing": "https://api.openai.com/v1", "endpoint_type": "/chat/completions"}]))
    client = UC(api_key=KEY, model="gpt-4o")
    client._sleep_with_cancel = lambda *a, **k: True
    calls = []
    monkeypatch.setattr(uac.requests, "request", lambda *a, **k: calls.append(a) or _Resp())
    with pytest.raises(Exception):
        client._http_request_with_retries("POST", "https://api.openai.com/v1/chat/completions", headers={},
                                          json={}, expected_status=(200,), max_retries=3, provider_name="openai")
    assert len(calls) == 3  # retried as before: no custom prefix handling for a built-in model
    assert client._custom_prefix_not_found_error("openai", 404, "https://api.openai.com/v1/chat/completions") is None
    assert client._custom_prefix_not_found_error("custom_openai", 404) is None  # no route for gpt-4o
    # a custom prefix model's request to some other host is not classified either
    client.model = "graf/x"
    assert client._custom_prefix_not_found_error("HTTP", 404, "https://elsewhere.example.com/x") is None
    assert client._custom_prefix_not_found_error("HTTP", 404, "https://api.openai.com/v1/chat/completions") is not None
