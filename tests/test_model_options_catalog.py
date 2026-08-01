import io
import sys
import types
import urllib.error

import model_options


def _isolated_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "model_catalog_cache.json"
    monkeypatch.setenv("GLOSSARION_MODEL_CATALOG_CACHE", str(cache_path))
    monkeypatch.setattr(model_options, "_MODEL_CATALOG_MEMORY_CACHE", None)
    # Never let catalog tests discover or invoke the developer's real OcAgy
    # account. Individual OcAgy tests explicitly opt into a fake account.
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(tmp_path / "empty-opencode-config"))
    for spec in model_options.PROVIDER_CATALOG_SPECS:
        for env_name in spec.api_key_envs:
            monkeypatch.delenv(env_name, raising=False)
        if spec.base_url_env:
            monkeypatch.delenv(spec.base_url_env, raising=False)
    return cache_path


def test_openrouter_online_catalog_replaces_static_provider_section(tmp_path, monkeypatch):
    cache_path = _isolated_cache(tmp_path, monkeypatch)

    def fake_get(url, headers, timeout):
        assert "openrouter.ai/api/v1/models" in url
        assert "Authorization" not in headers
        return {"data": [{"id": "vendor/new-model"}, {"id": "openrouter/free"}]}

    monkeypatch.setattr(model_options, "_http_get_json", fake_get)

    result = model_options.refresh_provider_model_catalogs(timeout=0.1)

    assert result.provider_models["openrouter"] == [
        "or/vendor/new-model",
        "or/openrouter/free",
    ]
    assert "or/vendor/new-model" in result.models
    assert "or/openai/gpt-5" not in result.models
    assert result.statuses["openrouter"] == "online (2 models)"
    assert cache_path.is_file()

    monkeypatch.setattr(model_options, "_MODEL_CATALOG_MEMORY_CACHE", None)
    assert "or/vendor/new-model" in model_options.get_model_options()


def test_failed_online_catalog_uses_corrected_static_openrouter_list(tmp_path, monkeypatch):
    _isolated_cache(tmp_path, monkeypatch)

    def fail_get(_url, _headers, _timeout):
        raise OSError("offline")

    monkeypatch.setattr(model_options, "_http_get_json", fail_get)

    result = model_options.refresh_provider_model_catalogs(timeout=0.1)

    assert result.statuses["openrouter"] == "static fallback (OSError — offline)"
    assert "or/openrouter/free" in result.models
    assert "or/openai/gpt-5.4-nano" in result.models
    assert "or/google/gemini-2.5-pro" in result.models
    assert "or/openai/gpt-5.4-nanoor/google/gemini-2.5-pro" not in result.models


def test_generic_key_is_sent_only_to_the_selected_provider(tmp_path, monkeypatch):
    _isolated_cache(tmp_path, monkeypatch)
    calls = []

    def fake_get(url, headers, timeout):
        calls.append((url, dict(headers)))
        if "groq.com" in url:
            return {"data": [{"id": "llama-new"}]}
        if "openrouter.ai" in url:
            return {"data": [{"id": "router/new"}]}
        raise AssertionError(f"unexpected provider request: {url}")

    monkeypatch.setattr(model_options, "_http_get_json", fake_get)

    result = model_options.refresh_provider_model_catalogs(
        active_model="groq/llama-old",
        active_api_key="groq-secret",
        timeout=0.1,
    )

    groq_call = next(call for call in calls if "groq.com" in call[0])
    router_call = next(call for call in calls if "openrouter.ai" in call[0])
    assert groq_call[1]["Authorization"] == "Bearer groq-secret"
    assert "Authorization" not in router_call[1]
    assert result.provider_models["groq"] == ["groq/llama-new"]
    assert result.provider_models["openrouter"] == ["or/router/new"]


def test_public_openrouter_catalog_does_not_send_api_key(tmp_path, monkeypatch):
    _isolated_cache(tmp_path, monkeypatch)
    calls = []

    def fake_get(url, headers, timeout):
        calls.append((url, dict(headers)))
        if "openrouter.ai" in url:
            return {"data": [{"id": "router/new"}]}
        raise OSError("local proxy is not running")

    monkeypatch.setattr(model_options, "_http_get_json", fake_get)
    model_options.refresh_provider_model_catalogs(
        active_model="or/router/old",
        active_api_key="openrouter-secret",
        timeout=0.1,
    )

    router_call = next(call for call in calls if "openrouter.ai" in call[0])
    assert "Authorization" not in router_call[1]


def test_selected_custom_openai_route_can_poll_its_models(tmp_path, monkeypatch):
    _isolated_cache(tmp_path, monkeypatch)
    calls = []

    def fake_get(url, headers, timeout):
        calls.append((url, dict(headers)))
        if "openrouter.ai" in url:
            return {"data": [{"id": "openrouter/free"}]}
        if "localhost:9999" in url:
            return {"data": [{"id": "local-current"}]}
        raise AssertionError(f"unexpected provider request: {url}")

    monkeypatch.setattr(model_options, "_http_get_json", fake_get)
    result = model_options.refresh_provider_model_catalogs(
        active_model="lab/local-current",
        active_api_key="local-key",
        custom_routes=[{
            "prefix": "lab/",
            "routing": "http://localhost:9999/v1",
            "endpoint_type": "/chat/completions",
        }],
        timeout=0.1,
    )

    assert "lab/local-current" in result.models
    local_call = next(call for call in calls if "localhost:9999" in call[0])
    assert local_call[0] == "http://localhost:9999/v1/models"
    assert local_call[1]["Authorization"] == "Bearer local-key"


def test_running_antigravity_proxy_replaces_its_static_catalog(tmp_path, monkeypatch):
    _isolated_cache(tmp_path, monkeypatch)

    def fake_get(url, headers, timeout):
        if "openrouter.ai" in url:
            return {"data": [{"id": "openrouter/free"}]}
        if "localhost:3000/v1/models" in url:
            return {"data": [{"id": "gemini-live"}, {"id": "claude-live"}]}
        raise AssertionError(f"unexpected provider request: {url}")

    monkeypatch.setattr(model_options, "_http_get_json", fake_get)
    result = model_options.refresh_provider_model_catalogs(timeout=0.1)

    assert result.provider_models["antigravity"] == [
        "antigravity/gemini-live",
        "antigravity/claude-live",
    ]
    assert "antigravity/gemini-3-flash" not in result.models


def test_selected_authgrok_uses_existing_session_without_login(tmp_path, monkeypatch):
    _isolated_cache(tmp_path, monkeypatch)
    token_calls = []

    class FakeStore:
        def get_valid_access_token(self, auto_login=True):
            token_calls.append(auto_login)
            return "oauth-token"

    fake_authgrok = types.SimpleNamespace(
        get_store=lambda account_id: FakeStore(),
        fetch_available_models=lambda token, timeout: ["grok-live", "grok-build"],
    )
    monkeypatch.setitem(sys.modules, "authgrok_auth", fake_authgrok)

    def fake_get(url, headers, timeout):
        if "openrouter.ai" in url:
            return {"data": [{"id": "openrouter/free"}]}
        raise OSError("local proxy is not running")

    monkeypatch.setattr(model_options, "_http_get_json", fake_get)
    result = model_options.refresh_provider_model_catalogs(
        active_model="authgrok2/grok-old",
        timeout=0.1,
    )

    assert token_calls == [False]
    assert result.provider_models["authgrok:2"] == [
        "authgrok2/grok-live",
        "authgrok2/grok-build",
    ]


def test_logged_in_ocagy_catalog_replaces_its_static_section(tmp_path, monkeypatch):
    _isolated_cache(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(model_options, "_ocagy_has_account", lambda: True)
    monkeypatch.setattr(
        model_options,
        "_fetch_ocagy_catalog",
        lambda timeout: calls.append(timeout) or [
            "ocagy/gemini-3.1-pro-high",
            "ocagy/future-live-model",
        ],
    )

    result = model_options.refresh_provider_model_catalogs(
        active_model="ocagy/gemini-3.1-pro-high",
        only_provider="ocagy",
        timeout=0.25,
    )

    assert calls == [0.25]
    assert result.statuses["ocagy"] == "online (2 models)"
    assert result.provider_models["ocagy"] == [
        "ocagy/gemini-3.1-pro-high",
        "ocagy/future-live-model",
    ]
    assert "ocagy/future-live-model" in result.models
    assert "ocagy/gemini-3.1-pro-low" not in result.models


def test_ocagy_poll_requires_a_logged_in_account(tmp_path, monkeypatch):
    _isolated_cache(tmp_path, monkeypatch)
    monkeypatch.setattr(model_options, "_ocagy_has_account", lambda: False)
    monkeypatch.setattr(
        model_options,
        "_fetch_ocagy_catalog",
        lambda _timeout: (_ for _ in ()).throw(AssertionError("OcAgy should not be polled")),
    )

    assert model_options.due_provider_catalog_for_model(
        "ocagy/gemini-3.1-pro-high"
    ) is None
    result = model_options.refresh_provider_model_catalogs(
        active_model="ocagy/gemini-3.1-pro-high",
        only_provider="ocagy",
        timeout=0.1,
    )

    assert result.statuses["ocagy"] == "static fallback (no provider credential)"
    assert result.provider_models == {}
    assert "ocagy/gemini-3.1-pro-high" in result.models


def test_logged_in_ocagy_is_eligible_for_automatic_polling(tmp_path, monkeypatch):
    _isolated_cache(tmp_path, monkeypatch)
    monkeypatch.setattr(model_options, "_ocagy_has_account", lambda: True)

    assert model_options.catalog_provider_for_model(
        "ocagy/gemini-3.1-pro-high"
    ) == "ocagy"
    assert model_options.due_provider_catalog_for_model(
        "ocagy/gemini-3.1-pro-high"
    ) == "ocagy"


def test_authgem_key_catalog_strips_gemini_resource_prefix(tmp_path, monkeypatch):
    _isolated_cache(tmp_path, monkeypatch)

    def fake_get(url, headers, timeout):
        if "openrouter.ai" in url:
            return {"data": [{"id": "openrouter/free"}]}
        if "generativelanguage.googleapis.com" in url:
            assert "key=gemini-secret" in url
            return {
                "models": [{
                    "name": "models/gemini-current",
                    "supportedGenerationMethods": ["generateContent"],
                }]
            }
        raise OSError("local proxy is not running")

    monkeypatch.setattr(model_options, "_http_get_json", fake_get)
    result = model_options.refresh_provider_model_catalogs(
        active_model="authgem-key/gemini-current",
        active_api_key="gemini-secret",
        timeout=0.1,
    )

    assert result.provider_models["authgem_key"] == [
        "authgem-key/gemini-current",
    ]


def test_http_catalog_failure_reports_status_and_safe_response_detail(tmp_path, monkeypatch):
    _isolated_cache(tmp_path, monkeypatch)

    def fake_get(url, headers, timeout):
        if "openrouter.ai" in url:
            return {"data": [{"id": "openrouter/free"}]}
        if "api.x.ai" in url:
            raise urllib.error.HTTPError(
                url,
                403,
                "Forbidden",
                {},
                io.BytesIO(b'{"error":{"message":"Missing ListModels permission"}}'),
            )
        raise OSError("local proxy is not running")

    monkeypatch.setattr(model_options, "_http_get_json", fake_get)
    result = model_options.refresh_provider_model_catalogs(
        active_model="grok-3-mini",
        active_api_key="xai-secret-that-must-not-appear",
        timeout=0.1,
    )

    status = result.statuses["xai"]
    assert status == "static fallback (HTTP 403 — Missing ListModels permission)"
    assert "xai-secret" not in status
    assert model_options.due_provider_catalog_for_model(
        "grok-3-mini", "xai-secret-that-must-not-appear"
    ) is None


def test_scoped_auto_poll_contacts_only_selected_provider_once_per_day(tmp_path, monkeypatch):
    _isolated_cache(tmp_path, monkeypatch)
    now = 1_800_000_000.0
    monkeypatch.setattr(model_options.time, "time", lambda: now)
    calls = []

    def fake_get(url, headers, timeout):
        calls.append(url)
        assert url == "https://api.x.ai/v1/models"
        assert headers["Authorization"] == "Bearer xai-secret"
        return {"data": [{"id": "grok-current"}]}

    monkeypatch.setattr(model_options, "_http_get_json", fake_get)
    assert model_options.due_provider_catalog_for_model(
        "grok-3-mini", "xai-secret"
    ) == "xai"

    result = model_options.refresh_provider_model_catalogs(
        active_model="grok-3-mini",
        active_api_key="xai-secret",
        only_provider="xai",
        timeout=0.1,
    )

    assert calls == ["https://api.x.ai/v1/models"]
    assert result.requested_provider == "xai"
    assert result.provider_models["xai"] == ["grok-current"]
    assert model_options.due_provider_catalog_for_model(
        "grok-3-mini", "xai-secret"
    ) is None

    monkeypatch.setattr(
        model_options.time,
        "time",
        lambda: now + model_options._MODEL_CATALOG_CACHE_TTL_SECONDS + 1,
    )
    assert model_options.due_provider_catalog_for_model(
        "grok-3-mini", "xai-secret"
    ) == "xai"


def test_model_catalog_cache_uses_macos_caches_directory(monkeypatch):
    monkeypatch.delenv("GLOSSARION_MODEL_CATALOG_CACHE", raising=False)
    monkeypatch.setattr(model_options.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(model_options.os.path, "expanduser", lambda _path: "/Users/tester")

    assert model_options._model_catalog_cache_path().replace("\\", "/") == (
        "/Users/tester/Library/Caches/Glossarion/model_catalog_cache.json"
    )


def test_last_successful_catalog_markers_survive_failure_and_reload(tmp_path, monkeypatch):
    _isolated_cache(tmp_path, monkeypatch)

    monkeypatch.setattr(
        model_options,
        "_http_get_json",
        lambda url, headers, timeout: {"data": [{"id": "grok-confirmed"}]},
    )
    model_options.refresh_provider_model_catalogs(
        active_model="grok-3-mini",
        active_api_key="xai-secret",
        only_provider="xai",
        timeout=0.1,
    )
    assert model_options.get_last_successful_provider_models()["xai"] == [
        "grok-confirmed"
    ]

    monkeypatch.setattr(
        model_options,
        "_http_get_json",
        lambda url, headers, timeout: (_ for _ in ()).throw(OSError("offline")),
    )
    failed = model_options.refresh_provider_model_catalogs(
        active_model="grok-3-mini",
        active_api_key="xai-secret",
        only_provider="xai",
        timeout=0.1,
    )
    assert failed.provider_models == {}

    # Force a disk reload to prove the marker state survives app restarts, not
    # merely the current process's in-memory cache.
    monkeypatch.setattr(model_options, "_MODEL_CATALOG_MEMORY_CACHE", None)
    assert model_options.get_last_successful_provider_models()["xai"] == [
        "grok-confirmed"
    ]

    monkeypatch.setattr(
        model_options,
        "_http_get_json",
        lambda url, headers, timeout: {"data": [{"id": "grok-replacement"}]},
    )
    model_options.refresh_provider_model_catalogs(
        active_model="grok-3-mini",
        active_api_key="xai-secret",
        only_provider="xai",
        timeout=0.1,
    )
    assert model_options.get_last_successful_provider_models()["xai"] == [
        "grok-replacement"
    ]
