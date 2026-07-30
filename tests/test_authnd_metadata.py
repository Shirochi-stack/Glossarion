import pytest
import threading
from types import SimpleNamespace

import authnd_auth


class _FakeResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code
        self.headers = {"content-type": "application/json"}
        self.reason = "Internal Server Error" if status_code >= 500 else ""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise authnd_auth.requests.HTTPError(f"HTTP {self.status_code}")


@pytest.fixture(autouse=True)
def _clear_authnd_metadata_state(monkeypatch):
    authnd_auth._metadata_cache.clear()
    for name in (
        "AUTHND_NVCF_FUNCTION_ID",
        "AUTHND_NGC_ORG",
        "AUTHND_PREDICT_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    authnd_auth._metadata_cache.clear()


def _model_html(*, artifact, function_id, model, namespace="qc69jvmznzxy"):
    return (
        rf'<script>payload \"model\":\"{model}\" '
        rf'\"namespace\":\"{namespace}\" '
        rf'\"nvcfFunctionId\":\"{function_id}\" '
        rf'\"artifactName\":\"{artifact}\"</script>'
    )


def test_normalize_model_preserves_current_numeric_dot_slug():
    publisher, model_id, page_url = authnd_auth._normalize_model(
        "authnd/moonshotai/kimi-k2.6"
    )

    assert publisher == "moonshotai"
    assert model_id == "kimi-k2.6"
    assert page_url == "https://build.nvidia.com/moonshotai/kimi-k2.6"


def test_model_page_urls_keep_legacy_numeric_underscore_fallback():
    assert authnd_auth._model_page_urls("meta", "llama-3.1-70b-instruct") == [
        "https://build.nvidia.com/meta/llama-3.1-70b-instruct",
        "https://build.nvidia.com/meta/llama-3_1-70b-instruct",
    ]


def test_metadata_parser_rejects_literal_none_as_a_function_id(monkeypatch):
    html = _model_html(
        artifact="kimi-k2.6",
        function_id="None",
        model="moonshotai/kimi-k2.6",
    )
    monkeypatch.setattr(
        authnd_auth.requests,
        "get",
        lambda *_args, **_kwargs: _FakeResponse(html),
    )

    metadata = authnd_auth._resolve_model_metadata(
        "https://build.nvidia.com/moonshotai/kimi-k2.6"
    )

    assert metadata["artifact_name"] == "kimi-k2.6"
    assert metadata["endpoint_id"] == "kimi-k2.6"
    assert metadata["function_id"] == ""
    assert metadata["function_id_unavailable"] is True
    assert metadata["function_id_source"] == ""
    assert metadata["payload_model"] == "moonshotai/kimi-k2.6"


def test_model_page_resolution_prefers_exact_valid_slug(monkeypatch):
    calls = []
    html = _model_html(
        artifact="kimi-k2.6",
        function_id="None",
        model="moonshotai/kimi-k2.6",
    )

    def fake_get(url, **_kwargs):
        calls.append(url)
        return _FakeResponse(html)

    monkeypatch.setattr(authnd_auth.requests, "get", fake_get)

    page_url, metadata = authnd_auth._resolve_model_page_metadata(
        "moonshotai", "kimi-k2.6"
    )

    assert page_url == "https://build.nvidia.com/moonshotai/kimi-k2.6"
    assert metadata["artifact_name"] == "kimi-k2.6"
    assert calls == [page_url]


def test_configured_function_id_overrides_broken_page_metadata(monkeypatch):
    configured_id = "11111111-2222-4333-8444-555555555555"
    html = _model_html(
        artifact="custom-model",
        function_id="None",
        model="example/custom-model",
    )
    monkeypatch.setenv("AUTHND_NVCF_FUNCTION_ID", configured_id)
    monkeypatch.setattr(
        authnd_auth.requests,
        "get",
        lambda *_args, **_kwargs: _FakeResponse(html),
    )

    metadata = authnd_auth._resolve_model_metadata(
        "https://build.nvidia.com/example/custom-model"
    )

    assert metadata["function_id"] == configured_id
    assert metadata["function_id_source"] == "configured"
    assert metadata["endpoint_id"] == "custom-model"


def test_model_page_resolution_uses_legacy_slug_when_exact_page_has_no_metadata(
    monkeypatch,
):
    calls = []
    valid_html = _model_html(
        artifact="llama-3_1-70b-instruct",
        function_id="8f723982-f99d-4978-a0cb-1334163e0e07",
        model="meta/llama-3.1-70b-instruct",
    )

    def fake_get(url, **_kwargs):
        calls.append(url)
        if "llama-3.1" in url:
            return _FakeResponse("<html>model not found</html>")
        return _FakeResponse(valid_html)

    monkeypatch.setattr(authnd_auth.requests, "get", fake_get)

    page_url, metadata = authnd_auth._resolve_model_page_metadata(
        "meta", "llama-3.1-70b-instruct"
    )

    assert page_url == "https://build.nvidia.com/meta/llama-3_1-70b-instruct"
    assert metadata["function_id"] == "8f723982-f99d-4978-a0cb-1334163e0e07"
    assert metadata["function_id_source"] == "page"
    assert metadata["endpoint_id"] == "llama-3_1-70b-instruct"
    assert len(calls) == 2


def test_null_page_function_id_does_not_block_browser_token_flow(monkeypatch):
    metadata = {
        "artifact_name": "kimi-k2.6",
        "endpoint_id": "kimi-k2.6",
        "function_id": "",
        "function_id_unavailable": True,
        "function_id_source": "",
    }
    monkeypatch.setattr(
        authnd_auth,
        "_resolve_model_page_metadata",
        lambda *_args: (
            "https://build.nvidia.com/moonshotai/kimi-k2.6",
            metadata,
        ),
    )

    calls = []

    def fake_token_flow(*_args, **_kwargs):
        calls.append("token")
        return "token"

    monkeypatch.setattr(
        authnd_auth,
        "_get_captcha_token_for_request",
        fake_token_flow,
    )
    monkeypatch.setattr(
        authnd_auth,
        "_post_prediction",
        lambda **_kwargs: (
            calls.append(_kwargs["page_url"])
            or {
                "content": "ok",
                "finish_reason": "stop",
                "finish_reason_explicit": True,
            }
        ),
    )

    result = authnd_auth.send_chat_completion(
        messages=[{"role": "user", "content": "hello"}],
        model="authnd/moonshotai/kimi-k2.6",
    )

    assert result["content"] == "ok"
    assert calls == [
        "token",
        "https://build.nvidia.com/moonshotai/kimi-k2.6",
    ]


def test_function_id_none_http_failure_is_classified_as_upstream_metadata():
    response = _FakeResponse(
        (
            "Invalid URL: Cannot parse `function_id` with value `None`: "
            "UUID parsing failed: invalid character"
        ),
        status_code=500,
    )

    with pytest.raises(authnd_auth.AuthNDUpstreamMetadataError):
        authnd_auth._raise_for_status(response)


def test_streaming_function_id_none_failure_gets_same_classification():
    class _StreamingResponse:
        status_code = 500
        headers = {"content-type": "application/json"}
        reason_phrase = "Internal Server Error"

        @staticmethod
        def read():
            return (
                b"Invalid URL: Cannot parse `function_id` with value `None`: "
                b"UUID parsing failed"
            )

    error = authnd_auth._httpx_status_error(_StreamingResponse())

    assert isinstance(error, authnd_auth.AuthNDUpstreamMetadataError)


def test_other_http_failure_remains_a_generic_runtime_error():
    response = _FakeResponse("service temporarily unavailable", status_code=503)

    with pytest.raises(RuntimeError) as exc_info:
        authnd_auth._raise_for_status(response)

    assert not isinstance(exc_info.value, authnd_auth.AuthNDUpstreamMetadataError)


def test_unified_client_does_not_retry_upstream_metadata_failure(monkeypatch):
    import unified_api_client

    calls = []

    def fail_once(**_kwargs):
        calls.append(_kwargs["model"])
        raise authnd_auth.AuthNDUpstreamMetadataError(
            "AuthND HTTP 500: Cannot parse `function_id` with value `None`: "
            "UUID parsing failed"
        )

    monkeypatch.setattr(unified_api_client, "_authnd_send", fail_once)
    monkeypatch.setattr(unified_api_client, "_authnd_reset_cancel", None)

    class _Client:
        _model_limits_lock = threading.Lock()
        _model_token_limits = {}
        request_timeout = 30

        def _get_active_request_model(self):
            return "authnd/moonshotai/kimi-k2.6"

        def _get_max_retries(self):
            return 7

        def _get_thinking_status_label(self):
            return ""

        def _should_abort_retry(self):
            return False

        def _is_stop_requested(self):
            return False

        def _get_anti_duplicate_params(self, *_args, **_kwargs):
            return {}

        def _streaming_enabled(self):
            return True

        def _get_thread_local_client(self):
            return SimpleNamespace()

        def _get_send_interval(self):
            pytest.fail("upstream metadata failures must not enter retry backoff")

    with pytest.raises(unified_api_client.UnifiedClientError) as exc_info:
        unified_api_client.UnifiedClient._send_authnd(
            _Client(),
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.2,
            max_tokens=32,
            response_name="test",
        )

    assert exc_info.value.error_type == "upstream_metadata"
    assert "NVIDIA returned invalid function metadata" in str(exc_info.value)
    assert exc_info.value.__suppress_context__ is True
    assert calls == ["moonshotai/kimi-k2.6"]
