import base64
import io
import sys
import types
import urllib.error
from types import SimpleNamespace

import model_options
import pytest
from unified_api_client import UnifiedClient, UnifiedClientError


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


@pytest.mark.parametrize(
    ("needs_api_key", "expected_timeout"),
    [
        (False, 60),
        (True, 30),
    ],
)
def test_multi_key_test_timeout_uses_optional_api_key_classification(
    monkeypatch,
    needs_api_key,
    expected_timeout,
):
    import multi_api_key_manager

    monkeypatch.setattr(
        multi_api_key_manager,
        "_model_needs_api_key",
        lambda _model: needs_api_key,
    )

    assert multi_api_key_manager._api_key_test_timeout_seconds("example/model") == expected_timeout


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


def test_selecting_antigravity_does_not_poll_before_proxy_start(tmp_path, monkeypatch):
    _isolated_cache(tmp_path, monkeypatch)

    assert model_options.catalog_provider_for_model(
        "antigravity/gemini-3-flash"
    ) == "antigravity"
    assert model_options.provider_model_catalog_refresh_due("antigravity")
    assert model_options.due_provider_catalog_for_model(
        "antigravity/gemini-3-flash"
    ) is None


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
    assert model_options.catalog_provider_for_model(
        "ocagy2/gemini-3.1-pro-high"
    ) == "ocagy"
    assert model_options.due_provider_catalog_for_model(
        "ocagy2/gemini-3.1-pro-high"
    ) == "ocagy"


def test_signed_in_authgpt_numbered_account_polls_codex_manifest(tmp_path, monkeypatch):
    _isolated_cache(tmp_path, monkeypatch)
    token_calls = []
    fetch_calls = []

    class FakeStore:
        has_tokens = True

        def get_valid_access_token(self, auto_login=True):
            token_calls.append(auto_login)
            return "chatgpt-oauth"

    fake_authgpt = types.SimpleNamespace(
        get_store=lambda account_id: FakeStore(),
        fetch_available_models=lambda token, timeout: (
            fetch_calls.append((token, timeout))
            or ["gpt-live", "gpt-next"]
        ),
    )
    monkeypatch.setitem(sys.modules, "authgpt_auth", fake_authgpt)

    assert model_options.catalog_provider_for_model("authgpt2/gpt-old") == "authgpt:2"
    assert model_options.due_provider_catalog_for_model(
        "authgpt2/gpt-old"
    ) == "authgpt:2"

    result = model_options.refresh_provider_model_catalogs(
        active_model="authgpt2/gpt-old",
        only_provider="authgpt:2",
        timeout=0.25,
    )

    assert token_calls == [False]
    assert fetch_calls == [("chatgpt-oauth", 1)]
    assert result.provider_models["authgpt:2"] == [
        "authgpt2/gpt-live",
        "authgpt2/gpt-next",
    ]


def test_authgem_poll_uses_selected_account_quota_catalog(tmp_path, monkeypatch):
    _isolated_cache(tmp_path, monkeypatch)
    fetch_calls = []

    class FakeStore:
        has_tokens = True

        def get_valid_access_token(self, auto_login=True):
            assert auto_login is False
            return "google-oauth"

    fake_authgem = types.SimpleNamespace(
        get_store=lambda account_id: FakeStore(),
        fetch_available_models=lambda token, timeout, account_id: (
            fetch_calls.append((token, timeout, account_id))
            or ["gemini-account-live"]
        ),
    )
    monkeypatch.setitem(sys.modules, "authgem_auth", fake_authgem)

    result = model_options.refresh_provider_model_catalogs(
        active_model="authgem7/gemini-old",
        only_provider="authgem:7",
        timeout=0.1,
    )

    assert fetch_calls == [("google-oauth", 1, 7)]
    assert result.provider_models["authgem:7"] == [
        "authgem7/gemini-account-live",
    ]


def test_authnd_poll_validates_curated_chat_models_against_public_catalog(
    tmp_path, monkeypatch
):
    _isolated_cache(tmp_path, monkeypatch)
    fake_authnd = types.SimpleNamespace(
        fetch_available_models=lambda timeout: [
            "nvidia/nemotron-3-ultra-550b-a55b",
            "baai/bge-m3",
        ],
    )
    monkeypatch.setitem(sys.modules, "authnd_auth", fake_authnd)

    result = model_options.refresh_provider_model_catalogs(
        active_model="authnd/nvidia/old",
        only_provider="authnd",
        timeout=0.1,
    )

    assert result.provider_models["authnd"] == [
        "authnd/nvidia/nemotron-3-ultra-550b-a55b",
    ]
    assert "authnd/baai/bge-m3" not in result.models


def test_authza_poll_reads_existing_selector_without_login(tmp_path, monkeypatch):
    _isolated_cache(tmp_path, monkeypatch)
    fetch_calls = []

    class FakeStore:
        has_tokens = True

    fake_authza = types.SimpleNamespace(
        get_store=lambda account_id: FakeStore(),
        fetch_available_models=lambda account_id, timeout: (
            fetch_calls.append((account_id, timeout)) or ["GLM-5", "GLM-4.7"]
        ),
    )
    monkeypatch.setitem(sys.modules, "authza_auth", fake_authza)

    result = model_options.refresh_provider_model_catalogs(
        active_model="authza3/GLM-4.7",
        only_provider="authza:3",
        timeout=0.1,
    )

    assert fetch_calls == [(3, 60)]
    assert result.provider_models["authza:3"] == [
        "authza3/GLM-5",
        "authza3/GLM-4.7",
    ]


def test_only_vertex_style_routes_remain_static_by_design():
    static_routes = model_options.STATIC_ONLY_PROVIDER_PREFIXES

    assert "authgpt/" not in static_routes
    assert "authcd/" not in static_routes
    assert "authgem/" not in static_routes
    assert "authnd/" not in static_routes
    assert "authza/" not in static_routes
    assert "authgem-vertex*/" in static_routes


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


def test_gemini_catalog_keeps_native_video_and_lyria_music_models(tmp_path, monkeypatch):
    _isolated_cache(tmp_path, monkeypatch)

    def fake_get(url, headers, timeout):
        assert "generativelanguage.googleapis.com" in url
        assert "key=gemini-secret" in url
        return {
            "models": [
                {
                    "name": "models/gemini-current",
                    "supportedGenerationMethods": ["generateContent"],
                },
                {
                    "name": "models/veo-3.1-generate-preview",
                    "supportedGenerationMethods": ["predictLongRunning"],
                },
                {
                    "name": "models/gemini-omni-flash-preview",
                    "supportedGenerationMethods": ["interactionsOnly"],
                },
                {
                    "name": "models/lyria-3-clip-preview",
                    "supportedGenerationMethods": ["interactionsOnly"],
                },
                {
                    "name": "models/lyria-realtime-exp",
                    "supportedGenerationMethods": ["bidiGenerateContent"],
                },
                {
                    "name": "models/text-embedding-current",
                    "supportedGenerationMethods": ["embedContent"],
                },
            ]
        }

    monkeypatch.setattr(model_options, "_http_get_json", fake_get)
    result = model_options.refresh_provider_model_catalogs(
        active_model="veo-3.1-generate-preview",
        active_api_key="gemini-secret",
        only_provider="gemini",
        timeout=0.1,
    )

    assert model_options.catalog_provider_for_model("veo-3.1-generate-preview") == "gemini"
    assert model_options.catalog_provider_for_model("lyria-3-pro-preview") == "gemini"
    assert result.provider_models["gemini"] == [
        "gemini-current",
        "veo-3.1-generate-preview",
        "gemini-omni-flash-preview",
        "lyria-3-clip-preview",
        "lyria-realtime-exp",
    ]
    assert "text-embedding-current" not in result.models


def _bare_gemini_video_client(tmp_path, monkeypatch):
    client = UnifiedClient.__new__(UnifiedClient)
    client.output_dir = str(tmp_path)
    client.request_timeout = 36000
    client._actual_output_filename = None
    client._get_image_output_aspect_ratio = lambda default="auto": "auto"
    client._extract_generation_prompt_and_image_url = lambda _messages: (
        "A cinematic waterfall",
        None,
    )
    client._get_thread_local_client = lambda: SimpleNamespace()
    client._should_abort_retry = lambda: False
    client._sleep_with_cancel = lambda *_args, **_kwargs: True
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("RETRY_TIMEOUT", "0")
    return client


def test_veo_and_omni_are_detected_as_native_video_models():
    assert UnifiedClient._is_video_gen_model("veo-3.1-generate-preview")
    assert UnifiedClient._is_video_gen_model("gemini-omni-flash-preview")
    assert UnifiedClient._gemini_video_model_kind("veo-3.1-fast-generate-preview") == "veo"
    assert UnifiedClient._gemini_video_model_kind("gemini-omni-flash-preview") == "omni"
    assert UnifiedClient._provider_from_model_name("veo-3.1-generate-preview") == "gemini"


def test_lyria_is_detected_as_native_gemini_music():
    assert UnifiedClient._is_music_gen_model("lyria-3-clip-preview")
    assert UnifiedClient._is_music_gen_model("models/lyria-realtime-exp")
    assert UnifiedClient._gemini_music_model_kind("lyria-3-pro-preview") == "interactions"
    assert UnifiedClient._gemini_music_model_kind("lyria-realtime-exp") == "realtime"
    assert UnifiedClient._provider_from_model_name("lyria-3-clip-preview") == "gemini"


def test_lyria_interactions_saves_mp3_and_returns_audio_marker(tmp_path, monkeypatch):
    client = _bare_gemini_video_client(tmp_path, monkeypatch)
    client._extract_generation_prompt_and_image_url = lambda _messages: (
        "Upbeat orchestral game music",
        None,
    )
    expected_audio = b"fake-lyria-mp3"
    calls = []

    class FakeInteractions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                status="completed",
                output_audio=SimpleNamespace(
                    data=base64.b64encode(expected_audio).decode("ascii"),
                    mime_type="audio/mpeg",
                ),
            )

    client._get_gemini_native_client = lambda _api_key: SimpleNamespace(
        interactions=FakeInteractions()
    )
    response = client._send_gemini_native_music(
        [{"role": "user", "content": "Upbeat orchestral game music"}],
        "response.txt",
        "lyria-3-clip-preview",
        "gemini-secret",
    )

    output_path = tmp_path / "response.mp3"
    assert output_path.read_bytes() == expected_audio
    assert response.content == f"[GENERATED_AUDIO:{output_path}]"
    assert calls[0]["model"] == "lyria-3-clip-preview"
    assert calls[0]["input"] == "Upbeat orchestral game music"
    assert calls[0]["timeout"] is None


def test_gemini_omni_uses_interactions_and_saves_mp4(tmp_path, monkeypatch):
    client = _bare_gemini_video_client(tmp_path, monkeypatch)
    calls = []
    expected_video = b"omni-mp4-bytes"

    class FakeInteractions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                status="completed",
                output_video=SimpleNamespace(
                    data=base64.b64encode(expected_video).decode("ascii"),
                    mime_type="video/mp4",
                ),
            )

    fake_sdk = SimpleNamespace(interactions=FakeInteractions())
    response = client._send_gemini_omni_video(
        fake_sdk,
        [{"role": "user", "content": "A cinematic waterfall"}],
        "response.txt",
        "gemini-omni-flash-preview",
    )

    output_path = tmp_path / "response.mp4"
    assert output_path.read_bytes() == expected_video
    assert response.content == f"[GENERATED_VIDEO:{output_path}]"
    assert calls[0]["model"] == "gemini-omni-flash-preview"
    assert calls[0]["response_format"] == {"type": "video", "delivery": "uri"}
    assert calls[0]["timeout"] is None


def test_veo_uses_long_running_operation_polling_and_saves_mp4(tmp_path, monkeypatch):
    client = _bare_gemini_video_client(tmp_path, monkeypatch)
    expected_video = b"veo-mp4-bytes"
    submitted = []
    pending = SimpleNamespace(done=False)
    finished = SimpleNamespace(
        done=True,
        error=None,
        response=SimpleNamespace(
            generated_videos=[SimpleNamespace(video=SimpleNamespace(uri="video-result"))],
            rai_media_filtered_count=0,
            rai_media_filtered_reasons=[],
        ),
    )

    class FakeModels:
        def generate_videos(self, **kwargs):
            submitted.append(kwargs)
            return pending

    class FakeOperations:
        def get(self, operation):
            assert operation is pending
            return finished

    class FakeFiles:
        def download(self, *, file):
            assert file == "video-result"
            return expected_video

    fake_sdk = SimpleNamespace(
        models=FakeModels(),
        operations=FakeOperations(),
        files=FakeFiles(),
    )
    response = client._send_gemini_veo_video(
        fake_sdk,
        [{"role": "user", "content": "A cinematic waterfall"}],
        "response.txt",
        "veo-3.1-generate-preview",
    )

    output_path = tmp_path / "response.mp4"
    assert output_path.read_bytes() == expected_video
    assert response.content == f"[GENERATED_VIDEO:{output_path}]"
    assert submitted[0]["model"] == "veo-3.1-generate-preview"
    assert submitted[0]["prompt"] == "A cinematic waterfall"


def test_gemini_video_provider_safety_error_is_prohibited_content(tmp_path, monkeypatch):
    client = _bare_gemini_video_client(tmp_path, monkeypatch)
    with pytest.raises(UnifiedClientError) as raised:
        client._raise_gemini_video_error(
            "Veo generation failed",
            UnifiedClientError(
                "Veo operation failed: response blocked by provider safety filter",
                error_type="api_error",
                http_status=400,
            ),
        )
    assert raised.value.error_type == "prohibited_content"
    assert raised.value.http_status == 400


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


def test_proxy_start_poll_ignores_failed_attempt_but_reuses_fresh_success(tmp_path, monkeypatch):
    _isolated_cache(tmp_path, monkeypatch)
    now = 1_800_000_000.0
    monkeypatch.setattr(model_options.time, "time", lambda: now)

    monkeypatch.setattr(
        model_options,
        "_http_get_json",
        lambda _url, _headers, _timeout: (_ for _ in ()).throw(
            OSError("local proxy is not running")
        ),
    )
    model_options.refresh_provider_model_catalogs(
        active_model="antigravity/gemini-3-flash",
        only_provider="antigravity",
        timeout=0.1,
    )

    assert not model_options.provider_model_catalog_refresh_due("antigravity")
    assert model_options.provider_model_catalog_refresh_due(
        "antigravity", successful_only=True
    )

    monkeypatch.setattr(
        model_options,
        "_http_get_json",
        lambda _url, _headers, _timeout: {
            "data": [{"id": "gemini-3-flash"}]
        },
    )
    model_options.refresh_provider_model_catalogs(
        active_model="antigravity/gemini-3-flash",
        only_provider="antigravity",
        timeout=0.1,
    )

    assert not model_options.provider_model_catalog_refresh_due(
        "antigravity", successful_only=True
    )


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
