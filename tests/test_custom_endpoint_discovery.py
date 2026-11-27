"""A 404 on a custom prefix route searches the real endpoint once and updates the route
(src/custom_endpoint_discovery.py, wired in UnifiedClient._send_custom_openai_with_endpoint_discovery).

A local fake HTTP server stands in for the provider: 404 on the wrong base, 401 on the right base
without a key, 200 JSON on ``/models`` with the key.
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import custom_endpoint_discovery as ced  # noqa: E402

KEY = "test-key-not-real"


class FakeProvider:
    """``good_prefix``: base path that exists (POST {} -> ``post_status`` without a key)."""

    def __init__(self, good_prefix="/v1", post_status=401, models_ok=True):
        self.good_prefix = good_prefix
        self.post_status = post_status
        self.models_ok = models_ok
        self.requests = []  # (method, path, has_authorization)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, status, body=b""):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                if n:
                    self.rfile.read(n)
                outer.requests.append(("POST", self.path, bool(self.headers.get("Authorization"))))
                if outer.good_prefix is not None and self.path.startswith(outer.good_prefix + "/"):
                    self._reply(outer.post_status, b'{"error":"missing key"}')
                else:
                    self._reply(404, b"not found")

            def do_GET(self):
                auth = self.headers.get("Authorization") or ""
                outer.requests.append(("GET", self.path, bool(auth)))
                if (outer.models_ok and self.path == outer.good_prefix + "/models"
                        and auth == f"Bearer {KEY}"):
                    self._reply(200, b'{"data":[{"id":"m"}]}')
                else:
                    self._reply(404, b"not found")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def posts(self):
        return [r for r in self.requests if r[0] == "POST"]


@pytest.fixture
def provider():
    made = []

    def make(**kw):
        p = FakeProvider(**kw)
        made.append(p)
        return p

    yield make
    for p in made:
        p.close()


@pytest.fixture(autouse=True)
def fresh_session(monkeypatch):
    ced.reset_session()
    monkeypatch.delenv("CUSTOM_OPENAI_PREFIX_ROUTES", raising=False)
    with ced._lock:
        saved = list(ced._listeners)
        ced._listeners.clear()
    yield
    ced.reset_session()
    with ced._lock:
        ced._listeners[:] = saved


# ------------------------------------------------------------------ candidates

def test_candidates_for_a_subdomain_base_in_the_fixed_order():
    assert ced.candidate_bases("https://status.5n7.net", "/chat/completions") == [
        "https://status.5n7.net/v1",
        "https://api.5n7.net/v1",
        "https://api.5n7.net",
        "https://5n7.net/v1",
    ]


def test_candidates_for_a_base_that_already_has_v1():
    assert ced.candidate_bases("https://api.5n7.net/v1/", "/chat/completions") == [
        "https://api.5n7.net",
        "https://5n7.net/v1",
        "https://api.5n7.net/v2",
    ]
    assert ced.candidate_bases("https://x.example.com/v2", "/responses")[-1] == "https://x.example.com/v1"


def test_candidates_never_include_the_current_url_or_duplicates():
    for base in ("https://api.5n7.net", "https://5n7.net/v1", "https://a.b.5n7.net/v1"):
        found = ced.candidate_bases(base, "/chat/completions")
        assert base.rstrip("/").lower() not in [c.lower() for c in found]
        assert len(found) == len(set(found))


def test_candidates_stay_on_the_same_registrable_domain():
    for base in ("https://status.5n7.net", "https://a.b.example.co.uk/x", "https://svc.example.com:8443/api"):
        domain = ced.registrable_domain(base.split("//")[1].split("/")[0].split(":")[0])
        for cand in ced.candidate_bases(base, "/chat/completions"):
            host = cand.split("//")[1].split("/")[0].split(":")[0]
            assert host == domain or host.endswith("." + domain), (base, cand)


def test_registrable_domain_handles_country_second_levels():
    assert ced.registrable_domain("status.5n7.net") == "5n7.net"
    assert ced.registrable_domain("a.b.example.co.uk") == "example.co.uk"
    assert ced.registrable_domain("example.com") == "example.com"


def test_candidates_keep_the_port():
    assert ced.candidate_bases("https://gw.example.com:8443", "/chat/completions")[0] == "https://gw.example.com:8443/v1"
    assert "https://api.example.com:8443/v1" in ced.candidate_bases("https://gw.example.com:8443", "/chat/completions")


def test_local_and_ip_bases_only_try_v1_variants():
    assert ced.candidate_bases("http://localhost:11434", "/chat/completions") == ["http://localhost:11434/v1"]
    assert ced.candidate_bases("http://127.0.0.1:8000/v1", "/chat/completions") == ["http://127.0.0.1:8000/v2"]
    assert ced.candidate_bases("http://192.168.1.5:8080", "/chat/completions") == ["http://192.168.1.5:8080/v1"]


def test_endpoint_types_that_carry_their_own_version_do_not_get_a_second_one():
    found = ced.candidate_bases("https://status.5n7.net", "/v1/messages")
    assert found == ["https://api.5n7.net", "https://5n7.net"]
    assert all("/v1" not in c for c in found)


def test_placeholder_endpoint_types_are_not_searched():
    assert not ced.supports_discovery("/{model_id}")
    assert ced.supports_discovery("/chat/completions") and ced.supports_discovery("/responses")


# ------------------------------------------------------------------ probing

def test_search_finds_the_right_base_and_never_sends_the_key_to_a_probe(provider):
    p = provider()
    result = ced.discover_endpoint(p.base, "/chat/completions", KEY)
    assert result.found == p.base + "/v1"
    assert result.evidence == "models"
    assert result.tried == [(p.base + "/v1", 401)]
    assert p.posts() == [("POST", "/chat/completions", False), ("POST", "/v1/chat/completions", False)]
    assert not any(auth for method, _path, auth in p.requests if method == "POST")
    assert ("GET", "/v1/models", True) in p.requests


def test_without_a_models_answer_a_401_probe_is_the_evidence(provider):
    p = provider(models_ok=False)
    result = ced.discover_endpoint(p.base, "/chat/completions", KEY)
    assert result.found == p.base + "/v1" and result.evidence == "401"


def test_weak_evidence_without_a_models_answer_is_not_accepted(provider):
    p = provider(post_status=405, models_ok=False)
    result = ced.discover_endpoint(p.base, "/chat/completions", KEY)
    assert result.found is None  # 405 alone is no proof; the first hit still ends the search


@pytest.mark.parametrize("status", [400, 401, 422])
def test_probe_statuses_that_prove_the_endpoint(provider, status):
    p = provider(post_status=status, models_ok=False)
    assert ced.discover_endpoint(p.base, "/chat/completions", KEY).found == p.base + "/v1"


def test_no_hit_lists_what_was_tried(provider):
    p = provider(good_prefix=None)
    result = ced.discover_endpoint("http://127.0.0.1:%d/api" % p.port, "/chat/completions", KEY)
    assert result.found is None
    assert [c for c, _s in result.tried] == ["http://127.0.0.1:%d/api/v1" % p.port]


def test_a_current_url_that_answers_is_not_searched(provider):
    p = provider(good_prefix="")  # every POST under the base answers 401
    result = ced.discover_endpoint(p.base, "/chat/completions", KEY)
    assert result.current_ok and result.found is None and result.tried == []
    assert len(p.posts()) == 1


def test_a_connection_error_counts_as_wrong():
    result = ced.discover_endpoint("http://127.0.0.1:9", "/chat/completions", "", probe=lambda url: None)
    assert result.found is None


# ------------------------------------------------------------------ session, env, config

def test_handle_not_found_updates_env_once_per_prefix(provider, monkeypatch):
    p = provider()
    monkeypatch.setenv("CUSTOM_OPENAI_PREFIX_ROUTES", json.dumps(
        [{"prefix": "graf/", "routing": p.base, "endpoint_type": "/chat/completions"}]))
    logs = []
    saved = []
    ced.register_route_listener(lambda prefix, old, new: saved.append((prefix, old, new)) or True)
    new = ced.handle_not_found("graf/", p.base, "/chat/completions", KEY, log=logs.append)
    assert new == p.base + "/v1"
    assert json.loads(os.environ["CUSTOM_OPENAI_PREFIX_ROUTES"])[0]["routing"] == p.base + "/v1"
    assert saved == [("graf/", p.base, p.base + "/v1")]
    assert len(logs) == 1 and "graf/" in logs[0] and "old URL kept" in logs[0] and p.base + "/v1" in logs[0]
    count = len(p.requests)
    # a worker whose request failed on the old URL gets the answer without another search
    assert ced.handle_not_found("graf/", p.base, "/chat/completions", KEY, log=logs.append) == p.base + "/v1"
    # a request that failed on the new URL itself starts nothing
    assert ced.handle_not_found("graf/", p.base + "/v1", "/chat/completions", KEY, log=logs.append) is None
    assert len(p.requests) == count and len(logs) == 1


def test_a_failed_search_is_not_repeated(provider):
    p = provider(good_prefix=None)
    logs = []
    assert ced.handle_not_found("x/", p.base, "/chat/completions", KEY, log=logs.append) is None
    count = len(p.requests)
    assert ced.handle_not_found("x/", p.base, "/chat/completions", KEY, log=logs.append) is None
    assert len(p.requests) == count and len(logs) == 1 and "tried" in logs[0]


def test_apply_route_to_config_keeps_the_old_url():
    config = {"custom_prefix_routes": [
        {"prefix": "graf/", "routing": "https://status.5n7.net", "endpoint_type": "/chat/completions"},
        {"prefix": "other/", "routing": "https://o.example.com", "endpoint_type": "/chat/completions"}]}
    assert ced.apply_route_to_config(config, "graf/", "https://status.5n7.net", "https://api.5n7.net/v1")
    assert config["custom_prefix_routes"][0]["routing"] == "https://api.5n7.net/v1"
    assert config["custom_prefix_routes"][1]["routing"] == "https://o.example.com"
    assert config[ced.PREVIOUS_ROUTING_KEY] == {"graf/": "https://status.5n7.net"}
    assert not ced.apply_route_to_config(config, "missing/", "a", "b")


def test_config_file_round_trip_keeps_other_keys(tmp_path):
    path = str(tmp_path / "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": "gpt", "custom_prefix_routes": [
            {"prefix": "graf/", "routing": "https://status.5n7.net", "endpoint_type": "/chat/completions"}]}, f)
    assert ced.persist_route_to_config_file("graf/", "https://status.5n7.net", "https://api.5n7.net/v1", path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["model"] == "gpt"
    assert data["custom_prefix_routes"][0]["routing"] == "https://api.5n7.net/v1"
    assert data[ced.PREVIOUS_ROUTING_KEY] == {"graf/": "https://status.5n7.net"}


def test_without_a_listener_the_config_file_takes_the_change(provider, monkeypatch, tmp_path):
    import app_paths
    p = provider()
    path = str(tmp_path / "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"custom_prefix_routes": [
            {"prefix": "graf/", "routing": p.base, "endpoint_type": "/chat/completions"}]}, f)
    monkeypatch.setattr(app_paths, "config_file_path", lambda: path)
    monkeypatch.setenv("CUSTOM_OPENAI_PREFIX_ROUTES", json.dumps(
        [{"prefix": "graf/", "routing": p.base, "endpoint_type": "/chat/completions"}]))
    assert ced.handle_not_found("graf/", p.base, "/chat/completions", KEY, log=lambda _m: None)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["custom_prefix_routes"][0]["routing"] == p.base + "/v1"
    assert data[ced.PREVIOUS_ROUTING_KEY] == {"graf/": p.base}


# ------------------------------------------------------------------ the request path

class _NotFound(Exception):
    def __init__(self, status):
        super().__init__(f"Error code: {status}")
        self.status_code = status


def _client(monkeypatch, base, outcomes):
    """A UnifiedClient whose router replays *outcomes* (exceptions raise, anything else returns)."""
    import unified_api_client as uac
    monkeypatch.setenv("CUSTOM_OPENAI_PREFIX_ROUTES", json.dumps(
        [{"prefix": "graf/", "routing": base, "endpoint_type": "/chat/completions"}]))
    client = object.__new__(uac.UnifiedClient)
    client.model = "graf/some-model"
    client.api_key = KEY
    client._is_stop_requested = lambda: False
    calls = []

    def router(*args, **kwargs):
        route = uac.UnifiedClient._get_custom_prefix_route_for_model(client.model)
        calls.append(route["routing"])
        outcome = outcomes[min(len(calls), len(outcomes)) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    client._send_openai_provider_router = router
    return client, calls


def test_a_404_is_retried_once_on_the_found_base(provider, monkeypatch):
    p = provider()
    client, calls = _client(monkeypatch, p.base, [_NotFound(404), "answer"])
    assert client._send_custom_openai_with_endpoint_discovery([], 0.3, 100, "resp") == "answer"
    assert calls == [p.base, p.base + "/v1"]


def test_the_retry_happens_once_even_if_it_404s_again(provider, monkeypatch):
    p = provider()
    client, calls = _client(monkeypatch, p.base, [_NotFound(404)])
    with pytest.raises(_NotFound):
        client._send_custom_openai_with_endpoint_discovery([], 0.3, 100, "resp")
    assert len(calls) == 2
    # and the next 404 of the session does not search again
    count = len(p.requests)
    with pytest.raises(_NotFound):
        client._send_custom_openai_with_endpoint_discovery([], 0.3, 100, "resp")
    assert len(p.requests) == count


@pytest.mark.parametrize("status", [401, 403, 429, 500, 502])
def test_other_statuses_start_no_search(provider, monkeypatch, status):
    p = provider()
    client, calls = _client(monkeypatch, p.base, [_NotFound(status)])
    with pytest.raises(_NotFound):
        client._send_custom_openai_with_endpoint_discovery([], 0.3, 100, "resp")
    assert calls == [p.base] and p.requests == []


def test_no_hit_raises_the_original_error_without_a_retry(provider, monkeypatch):
    p = provider(good_prefix=None)
    client, calls = _client(monkeypatch, p.base, [_NotFound(404)])
    with pytest.raises(_NotFound):
        client._send_custom_openai_with_endpoint_discovery([], 0.3, 100, "resp")
    assert calls == [p.base]


def test_the_custom_openai_handler_is_the_discovery_wrapper():
    import inspect
    import unified_api_client as uac
    source = inspect.getsource(uac.UnifiedClient)
    assert "'custom_openai': self._send_custom_openai_with_endpoint_discovery," in source


# ------------------------------------------------------------------ the desktop owner

def test_desktop_listener_updates_owner_config_env_and_file(monkeypatch):
    translator_gui = pytest.importorskip("translator_gui")
    import config_store
    from run_env import RunEnvMixin

    class Owner(RunEnvMixin):
        pass

    owner = Owner()
    routes = [{"prefix": "graf/", "routing": "https://status.5n7.net", "endpoint_type": "/chat/completions"}]
    owner.config = {"custom_prefix_routes": routes}
    owner.custom_prefix_routes = list(routes)
    written = {}
    monkeypatch.setattr(config_store, "update_config_file", lambda values, path=None: written.update(values))
    monkeypatch.delenv("CUSTOM_OPENAI_PREFIX_ROUTES", raising=False)
    listener = translator_gui.TranslatorGUI._on_custom_prefix_endpoint_found.__get__(owner)
    assert listener("graf/", "https://status.5n7.net", "https://api.5n7.net/v1") is True
    assert owner.custom_prefix_routes[0]["routing"] == "https://api.5n7.net/v1"
    assert owner.config["custom_prefix_routes"] is owner.custom_prefix_routes
    assert owner.config[ced.PREVIOUS_ROUTING_KEY] == {"graf/": "https://status.5n7.net"}
    assert json.loads(os.environ["CUSTOM_OPENAI_PREFIX_ROUTES"])[0]["routing"] == "https://api.5n7.net/v1"
    assert written["custom_prefix_routes"][0]["routing"] == "https://api.5n7.net/v1"
    # the save_config path normalizes self.custom_prefix_routes again: the new URL survives it
    assert owner._normalize_custom_prefix_routes(owner.custom_prefix_routes)[0]["routing"] == "https://api.5n7.net/v1"
    monkeypatch.delenv("CUSTOM_OPENAI_PREFIX_ROUTES", raising=False)


def test_the_new_module_is_in_every_spec_and_both_blocks():
    import glob
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    specs = glob.glob(os.path.join(root, "*.spec"))
    assert len(specs) == 14
    for spec in specs:
        text = open(spec, encoding="utf-8").read()
        assert "('custom_endpoint_discovery.py', '.')," in text, spec
        assert "'custom_endpoint_discovery'," in text, spec
        data = open(spec, "rb").read()
        assert data.count(b"\r\n") in (0, data.count(b"\n")), spec
