import json
import time
import types

import pytest

import antigravity_proxy
import unified_api_client
from html_output_utils import ensure_utf8_html_document
from model_options import get_model_options
from unified_api_client import UnifiedClient, UnifiedClientError


@pytest.fixture(autouse=True)
def _reset_antigravity_cancel_state():
    antigravity_proxy.reset_cancel()
    yield
    antigravity_proxy.reset_cancel()


class FakeStreamResponse:
    status_code = 200

    def __init__(self, lines):
        self._lines = lines
        self.closed = False

    def iter_lines(self, decode_unicode=True, chunk_size=1):
        yield from self._lines

    def close(self):
        self.closed = True


class FakeHttpxStreamResponse:
    status_code = 200

    def __init__(self, lines):
        self._lines = lines
        self.closed = False

    def iter_lines(self):
        yield from self._lines

    def close(self):
        self.closed = True


class FakeHTTPResponse:
    def __init__(self, json_data=None, text="", content=b""):
        self._json_data = json_data
        self.text = text
        self.content = content

    def json(self):
        return self._json_data

    def raise_for_status(self):
        return None


class EncodingAwareStreamResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload
        self.encoding = "iso-8859-1"
        self.closed = False

    def iter_lines(self, decode_unicode=True, chunk_size=1):
        raw = ("data: " + json.dumps(self.payload, ensure_ascii=False)).encode("utf-8")
        yield raw.decode(self.encoding) if decode_unicode else raw
        yield "data: [DONE]"

    def close(self):
        self.closed = True


def _sse_event(payload):
    return "data: " + json.dumps(payload)


def _unified_antigravity_client(model="antigravity/gemini-3.1-pro-low"):
    client = UnifiedClient.__new__(UnifiedClient)
    client.model = model
    client.client_type = "antigravity"
    client.current_key_output_token_limit = None
    client._cancelled = False
    client._ignore_graceful_stop = False
    client._get_thread_local_client = lambda: types.SimpleNamespace(
        output_token_limit=None,
        per_key_max_output_tokens=None,
    )
    client._is_o_series_model = lambda: False
    return client


def test_normalize_model_name_prefixes_gemini_ids_for_upstream_proxy():
    assert (
        antigravity_proxy._normalize_model_name("gemini-2.5-flash")
        == "antigravity-gemini-2.5-flash"
    )
    assert (
        antigravity_proxy._normalize_model_name("antigravity/gemini-2.5-pro")
        == "antigravity-gemini-2.5-pro"
    )
    assert (
        antigravity_proxy._normalize_model_name("antigravity2/gemini-3.5-flash-medium")
        == "antigravity-gemini-3.5-flash-medium"
    )


def test_normalize_model_name_prefixes_sandbox_ids_for_upstream_proxy():
    assert (
        antigravity_proxy._normalize_model_name("claude-sonnet-4-6")
        == "antigravity-claude-sonnet-4-6"
    )
    assert (
        antigravity_proxy._normalize_model_name("antigravity/gemini-3.1-pro-low")
        == "antigravity-gemini-3.1-pro-low"
    )
    assert (
        antigravity_proxy._normalize_model_name("antigravity-claude-opus-4-6-thinking-high")
        == "antigravity-claude-opus-4-6-thinking-high"
    )


def test_parse_openai_chat_response():
    data = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "translated text"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }

    parsed = antigravity_proxy._parse_openai_chat_response(data)

    assert parsed["content"] == "translated text"
    assert parsed["finish_reason"] == "stop"
    assert parsed["usage"]["total_tokens"] == 5
    assert parsed["raw_response"] is data


def test_consume_openai_stream_collects_content_and_usage():
    response = FakeStreamResponse(
        [
            _sse_event({"choices": [{"delta": {"reasoning_content": "think"}, "finish_reason": None}]}),
            _sse_event(
                {
                    "choices": [{"delta": {"content": "Hel"}, "finish_reason": None}],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
                }
            ),
            _sse_event({"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]}),
            "data: [DONE]",
        ]
    )

    result = antigravity_proxy._consume_openai_stream(response, log_fn=lambda _: None, log_stream=False)

    assert result["content"] == "Hello"
    assert result["finish_reason"] == "stop"
    assert result["usage"]["total_tokens"] == 5
    assert response.closed is True


def test_consume_openai_stream_supports_httpx_iter_lines():
    response = FakeHttpxStreamResponse(
        [
            _sse_event({"choices": [{"delta": {"content": "Hel"}, "finish_reason": None}]}),
            _sse_event({"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]}),
            "data: [DONE]",
        ]
    )

    result = antigravity_proxy._consume_openai_stream(response, log_fn=lambda _: None, log_stream=False)

    assert result["content"] == "Hello"
    assert result["finish_reason"] == "stop"
    assert response.closed is True


def test_consume_openai_stream_raises_on_error_event():
    response = FakeStreamResponse(
        [
            _sse_event(
                {
                    "error": {
                        "message": "Quota Exhausted: All accounts failed or are exhausted for this model.",
                        "code": "insufficient_quota",
                    }
                }
            ),
            "data: [DONE]",
        ]
    )

    with pytest.raises(RuntimeError, match="Quota Exhausted"):
        antigravity_proxy._consume_openai_stream(response, log_fn=lambda _: None, log_stream=False)

    assert response.closed is True


def test_consume_openai_stream_forces_utf8_for_unicode_content():
    response = EncodingAwareStreamResponse(
        {"choices": [{"delta": {"content": "I’m telling you."}, "finish_reason": "stop"}]}
    )

    result = antigravity_proxy._consume_openai_stream(response, log_fn=lambda _: None, log_stream=False)

    assert result["content"] == "I’m telling you."
    assert response.encoding == "utf-8"


def test_cancel_stream_closes_registered_active_response():
    response = FakeStreamResponse([])
    antigravity_proxy._register_active_response(response)

    antigravity_proxy.cancel_stream()

    assert response.closed is True
    assert antigravity_proxy.is_cancelled() is True


def test_cancelled_chat_preflight_never_posts(monkeypatch):
    post_calls = []
    monkeypatch.setattr(
        antigravity_proxy.requests,
        "post",
        lambda *_args, **_kwargs: post_calls.append((_args, _kwargs)),
    )
    antigravity_proxy.cancel_stream()

    with pytest.raises(RuntimeError, match="cancelled by user"):
        antigravity_proxy._post_chat(
            {"messages": []},
            timeout=30,
            stream=True,
            headers={"Content-Type": "application/json"},
        )

    assert post_calls == []


def test_graceful_stop_aborts_antigravity_429_retry_before_second_post(monkeypatch):
    client = _unified_antigravity_client()
    client.request_timeout = 300
    client._is_stop_requested = lambda: False
    client._get_max_retries = lambda: 3
    client._get_send_interval = lambda: 0.01
    client._streaming_enabled = lambda: True
    client._is_rate_limit_error = lambda _exc: True

    monkeypatch.setenv("GRACEFUL_STOP", "0")
    monkeypatch.setenv("GRACEFUL_STOP_COMPLETED", "0")
    monkeypatch.delenv("TRANSLATION_CANCELLED", raising=False)
    monkeypatch.setattr(unified_api_client, "ANTIGRAVITY_AVAILABLE", True)
    monkeypatch.setattr(unified_api_client, "_antigravity_send", lambda **_kwargs: None)
    monkeypatch.setattr(
        unified_api_client,
        "_antigravity_ensure_running",
        lambda log_fn=None: {"running": True},
    )
    monkeypatch.setattr(unified_api_client, "_antigravity_cancel_stream", lambda: None)

    post_calls = []

    def rate_limited_send(**_kwargs):
        post_calls.append(True)
        monkeypatch.setenv("GRACEFUL_STOP", "1")
        raise RuntimeError("Antigravity: HTTP 429 - quota exhausted")

    monkeypatch.setattr(unified_api_client, "_antigravity_send_stream", rate_limited_send)

    with pytest.raises(UnifiedClientError) as exc_info:
        client._send_antigravity([], 0.2, 64000, "response.txt")

    assert exc_info.value.error_type == "cancelled"
    assert post_calls == [True]


def test_antigravity_worker_never_resets_shared_cancel_event(monkeypatch):
    client = _unified_antigravity_client("antigravity1/gemini-3.5-flash-low")
    client.request_timeout = 300
    client._is_stop_requested = lambda: False
    client._should_abort_retry = lambda: False
    client._get_max_retries = lambda: 1
    client._streaming_enabled = lambda: True

    reset_calls = []
    monkeypatch.setattr(unified_api_client, "ANTIGRAVITY_AVAILABLE", True)
    monkeypatch.setattr(
        unified_api_client,
        "_antigravity_ensure_running",
        lambda log_fn=None: {"running": True},
    )
    monkeypatch.setattr(unified_api_client, "_antigravity_reset_cancel", lambda: reset_calls.append(True))
    monkeypatch.setattr(unified_api_client, "_antigravity_send", lambda **_kwargs: None)
    monkeypatch.setattr(
        unified_api_client,
        "_antigravity_send_stream",
        lambda **_kwargs: {"content": "ok", "finish_reason": "stop", "usage": None},
    )

    result = client._send_antigravity([], 0.2, 64000, "response.txt")

    assert result.content == "ok"
    assert reset_calls == []


def test_antigravity_fresh_request_clears_orphaned_adapter_cancel(monkeypatch):
    """An adapter-only cancel from an older run must not kill a fresh request."""
    client = _unified_antigravity_client("antigravity/claude-opus-4-6-thinking")
    client.request_timeout = 300
    client._is_stop_requested = lambda: False
    client._should_abort_retry = lambda: False
    client._get_max_retries = lambda: 1
    client._streaming_enabled = lambda: True

    monkeypatch.setattr(unified_api_client, "ANTIGRAVITY_AVAILABLE", True)
    monkeypatch.setattr(
        unified_api_client,
        "_antigravity_ensure_running",
        lambda log_fn=None: {"running": True},
    )
    monkeypatch.setattr(unified_api_client, "_antigravity_send", lambda **_kwargs: None)

    send_cancel_states = []

    def successful_send(**_kwargs):
        send_cancel_states.append(antigravity_proxy.is_cancelled())
        return {"content": "ok", "finish_reason": "stop", "usage": None}

    monkeypatch.setattr(unified_api_client, "_antigravity_send_stream", successful_send)
    antigravity_proxy.cancel_stream()
    assert antigravity_proxy.is_cancelled() is True

    result = client._send_antigravity([], 0.2, 64000, "response.txt")

    assert result.content == "ok"
    assert send_cancel_states == [False]
    assert antigravity_proxy.is_cancelled() is False


def test_should_abort_retry_treats_graceful_stop_as_retry_only_cancel(monkeypatch):
    client = _unified_antigravity_client()
    client._is_stop_requested = lambda: False
    monkeypatch.setenv("GRACEFUL_STOP", "1")
    monkeypatch.delenv("TRANSLATION_CANCELLED", raising=False)

    assert client._should_abort_retry() is True

    client._ignore_graceful_stop = True
    assert client._should_abort_retry() is False


def test_outer_rate_limit_sleep_exits_immediately_on_graceful_stop(monkeypatch):
    client = _unified_antigravity_client()
    client._is_stop_requested = lambda: False
    monkeypatch.setenv("GRACEFUL_STOP", "1")
    monkeypatch.setattr(
        unified_api_client.time,
        "sleep",
        lambda _seconds: pytest.fail("cancelled retry wait must not sleep"),
    )

    assert client._sleep_with_cancel(60, 0.5) is False


def test_antigravity_cancel_resets_only_on_explicit_new_run_reset(monkeypatch):
    calls = []
    monkeypatch.setattr(unified_api_client, "_antigravity_cancel_stream", lambda: calls.append("cancel"))
    monkeypatch.setattr(unified_api_client, "_antigravity_reset_cancel", lambda: calls.append("reset"))

    try:
        unified_api_client.set_stop_flag(True)
        unified_api_client.set_stop_flag(False)
    finally:
        unified_api_client.global_stop_flag = False
        UnifiedClient.set_global_cancellation(False)

    assert calls == ["cancel", "reset"]


def test_antigravity_payload_clamps_model_token_limits():
    claude_payload = antigravity_proxy._payload_for_openai_chat(
        [], "claude-sonnet-4-6", 0.2, 200000, False
    )
    gemini_payload = antigravity_proxy._payload_for_openai_chat(
        [], "gemini-3.5-flash-low", 0.2, 200000, False
    )

    assert claude_payload["max_tokens"] == 64000
    assert gemini_payload["max_tokens"] == 64000


def test_antigravity_token_limit_log_reports_clamp():
    payload = antigravity_proxy._payload_for_openai_chat(
        [], "gemini-2.5-flash", 0.2, 65536, False
    )
    messages = []

    antigravity_proxy._log_payload_token_limit(messages.append, 65536, payload)

    assert payload["max_tokens"] == 64000
    assert messages == [
        "🎚️ Antigravity: max_tokens clamped 65,536 -> 64,000 (model=antigravity-gemini-2.5-flash)"
    ]


def test_min_accounts_for_auth_retry_follows_numbered_prefix_slots():
    assert antigravity_proxy._min_accounts_for_auth_retry("Quota Exhausted: All accounts failed") == 1
    assert antigravity_proxy._min_accounts_for_auth_retry("No accounts configured", account_id=2) == 2


def test_quota_access_denied_does_not_trigger_auth_wait(monkeypatch):
    monkeypatch.setattr(antigravity_proxy, "_proxy_has_accounts", lambda: True)
    body = json.dumps(
        {
            "error": {
                "message": "Access denied: quota_exhausted",
                "type": "access_denied",
                "code": "403",
            }
        }
    )

    assert antigravity_proxy._error_text_suggests_rate_limit(body) is True
    assert antigravity_proxy._should_wait_for_auth_status_error(403, body) is False


def test_proxy_status_message_includes_quota_and_cooldown_context(tmp_path, monkeypatch):
    accounts_file = tmp_path / "antigravity-accounts.json"
    accounts_file.write_text(
        json.dumps(
            {
                "accounts": [
                    {
                        "email": "limited@example.test",
                        "refreshToken": "redacted",
                        "quota": [
                            {
                                "groupName": "Gemini 3 Flash",
                                "quotaLeft": "0%",
                                "resetIn": "12m",
                            }
                        ],
                        "cooldowns": {
                            "sandbox|Gemini 3 Flash": int((time.time() + 90) * 1000),
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTIGRAVITY_ACCOUNTS_FILE", str(accounts_file))

    message = antigravity_proxy._format_proxy_status_message(
        403,
        json.dumps({"error": {"message": "Access denied: quota_exhausted"}}),
        payload={"model": "antigravity-gemini-3-flash"},
        account_id=1,
    )

    assert "quota_exhausted" in message
    assert "Antigravity quota/rate limit detail" in message
    assert "limited@example.test" in message
    assert "cooldown sandbox|Gemini 3 Flash resets in" in message
    assert "Gemini 3 Flash 0% left, reset in 12m" in message


def test_proxy_status_message_prioritizes_upstream_429_over_final_403():
    body = json.dumps(
        {
            "error": {
                "message": "Access denied: unknown_error - API disabled",
                "type": "access_denied",
                "code": "403",
                "attempts": [
                    {
                        "email": "limited@example.test",
                        "status": 429,
                        "reason": "quota_exhausted",
                        "message": "Individual quota reached. Resets in 112h0m45s.",
                    },
                    {
                        "email": "limited@example.test",
                        "status": 403,
                        "reason": "unknown_error",
                        "message": "API disabled",
                    },
                ],
            }
        }
    )

    message = antigravity_proxy._format_proxy_status_message(
        403,
        body,
        payload={"model": "antigravity-gemini-3.5-flash-low"},
        account_id=3,
    )

    assert "Antigravity: HTTP 429" in message
    assert "HTTP 429" in message
    assert "quota_exhausted" in message
    assert "Individual quota reached. Resets in 112h0m45s." in message
    assert "HTTP 403" not in message
    assert "API disabled" not in message
    assert "Antigravity quota/rate limit detail" in message


def test_numbered_antigravity_prefix_forces_account_header(monkeypatch):
    fake_summary = {
        "healthy": True,
        "accounts": [
            {"email": "first@example.test"},
            {"email": "second@example.test"},
        ],
    }
    monkeypatch.setattr(
        antigravity_proxy,
        "get_account_summary",
        lambda: fake_summary,
    )
    monkeypatch.setattr(
        antigravity_proxy,
        "get_stored_account_summary",
        lambda: fake_summary,
    )

    assert antigravity_proxy._extract_antigravity_account_id("antigravity/gemini-2.5-flash") == 1
    assert antigravity_proxy._extract_antigravity_account_id("antigravity1/gemini-2.5-flash") == 2
    assert antigravity_proxy._extract_antigravity_account_id("antigravity12/gemini-2.5-flash") == 13

    headers = antigravity_proxy._build_headers(account_id=2)

    assert headers["X-Antigravity-Account"] == "second@example.test"
    assert headers["X-Client-Id"] == "glossarion-antigravity2"
    assert (
        antigravity_proxy._account_slot_log_message(2, headers)
        == "🧭 Antigravity: using account slot #2 (second@example.test)"
    )


def test_stream_chat_with_httpx_disables_compression(monkeypatch):
    class FakeTimeout:
        def __init__(self, timeout, connect=None):
            self.timeout = timeout
            self.connect = connect

    class FakeHttpx:
        Timeout = FakeTimeout

        def __init__(self):
            self.call = None

        def stream(self, method, url, **kwargs):
            self.call = {"method": method, "url": url, **kwargs}
            return object()

    fake_httpx = FakeHttpx()
    monkeypatch.setattr(antigravity_proxy, "httpx", fake_httpx)

    result = antigravity_proxy._stream_chat_with_httpx(
        "http://localhost:3000/v1/chat/completions",
        {"stream": True},
        {"Content-Type": "application/json"},
        300,
    )

    assert result is not None
    assert fake_httpx.call["method"] == "POST"
    assert fake_httpx.call["headers"]["Accept"] == "text/event-stream"
    assert fake_httpx.call["headers"]["Accept-Encoding"] == "identity"
    assert fake_httpx.call["timeout"].timeout == 300
    assert fake_httpx.call["timeout"].connect == 30.0


def test_wait_for_auth_keeps_httpx_stream_open_until_consumer_closes(monkeypatch):
    response = FakeHttpxStreamResponse(["data: [DONE]"])

    class FakeStreamContext:
        def __init__(self):
            self.exited = False

        def __enter__(self):
            return response

        def __exit__(self, exc_type, exc, tb):
            self.exited = True

    context = FakeStreamContext()

    monkeypatch.setattr(antigravity_proxy, "httpx", object())
    monkeypatch.setattr(antigravity_proxy, "_wait_for_cancel", lambda _seconds: False)
    monkeypatch.setattr(antigravity_proxy, "_open_auth_browser_once", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(antigravity_proxy, "_proxy_has_accounts", lambda: True)
    monkeypatch.setattr(antigravity_proxy, "_proxy_account_count", lambda: 1)
    monkeypatch.setattr(
        antigravity_proxy,
        "_stream_chat_with_httpx",
        lambda *_args, **_kwargs: context,
    )

    retry_resp = antigravity_proxy._wait_for_auth(
        "http://localhost:3000/v1/chat/completions",
        {"stream": True},
        {"Content-Type": "application/json"},
        "http://localhost:3000",
        log_fn=lambda _message: None,
        max_wait=5,
        poll_interval=5,
        stream=True,
        request_timeout=300,
        prefer_httpx_stream=True,
    )

    assert retry_resp is not None
    assert retry_resp.status_code == 200
    assert response.closed is False
    assert context.exited is False

    retry_resp.close()

    assert response.closed is True
    assert context.exited is True


def test_wait_for_auth_cancel_does_not_launch_retry_post(monkeypatch):
    post_calls = []
    monkeypatch.setattr(antigravity_proxy, "_open_auth_browser_once", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(antigravity_proxy, "_proxy_account_count", lambda: 1)
    monkeypatch.setattr(
        antigravity_proxy,
        "_wait_for_cancel",
        lambda _seconds: (antigravity_proxy.cancel_stream() or True),
    )
    monkeypatch.setattr(
        antigravity_proxy.requests,
        "post",
        lambda *_args, **_kwargs: post_calls.append((_args, _kwargs)),
    )

    with pytest.raises(RuntimeError, match="cancelled by user"):
        antigravity_proxy._wait_for_auth(
            "http://localhost:3000/v1/chat/completions",
            {"stream": False},
            {"Content-Type": "application/json"},
            "http://localhost:3000",
            log_fn=lambda _message: None,
            max_wait=5,
            poll_interval=5,
        )

    assert post_calls == []


def test_utf8_html_output_helper_adds_charset_to_fragments_and_documents():
    fragment = ensure_utf8_html_document("<h1>Title</h1><p>I’m here.</p>")
    document = ensure_utf8_html_document("<html><body><p>I’m here.</p></body></html>")

    assert '<meta charset="utf-8">' in fragment
    assert "<body>" in fragment
    assert '<meta charset="utf-8">' in document
    assert document.index("<head>") < document.index("<body>")


def test_model_options_match_current_antigravity_dashboard_catalog():
    antigravity_options = {
        option for option in get_model_options()
        if str(option).startswith("antigravity/")
    }

    expected = {
        "antigravity/gemini-3-flash",
        "antigravity/gemini-3-flash-agent",
        "antigravity/gemini-3.1-flash-image",
        "antigravity/gemini-3.1-flash-lite",
        "antigravity/gemini-3.5-flash-extra-low",
        "antigravity/gemini-3.5-flash-low",
        "antigravity/gemini-3.1-pro-low",
        "antigravity/gemini-pro-agent",
        "antigravity/gemini-2.5-flash",
        "antigravity/gemini-2.5-flash-lite",
        "antigravity/gemini-2.5-flash-thinking",
        "antigravity/gemini-2.5-pro",
        "antigravity/claude-opus-4-6-thinking",
        "antigravity/claude-sonnet-4-6",
        "antigravity/gpt-oss-120b-medium",
    }

    assert antigravity_options == expected


def test_latest_proxy_version_prefers_newer_github_tag_without_git(monkeypatch):
    def fake_get(url, headers=None, timeout=15):
        assert url == antigravity_proxy.PROXY_GITHUB_API_TAGS
        return FakeHTTPResponse(
            json_data=[
                {"name": "v0.7.0", "zipball_url": "https://example.test/v0.7.0.zip"},
                {"name": "v1.7.1", "zipball_url": "https://example.test/v1.7.1.zip"},
            ]
        )

    monkeypatch.delenv("ANTIGRAVITY_PROXY_TAG", raising=False)
    monkeypatch.delenv("ANTIGRAVITY_PROXY_VERSION", raising=False)
    monkeypatch.setattr(antigravity_proxy.requests, "get", fake_get)

    release = antigravity_proxy._latest_proxy_release()

    assert release["version"] == "1.7.1"
    assert release["archive_url"].endswith("/archive/refs/tags/v1.7.1.zip")


def test_latest_antigravity_client_version_uses_google_public_bundle(monkeypatch):
    def fake_get(url, timeout=15):
        if url == antigravity_proxy.ANTIGRAVITY_SITE_URL:
            return FakeHTTPResponse(text='<script src="main.js" type="module"></script>')
        if url == "https://antigravity.google/main.js":
            return FakeHTTPResponse(
                text=(
                    'href:"https://storage.googleapis.com/antigravity-public/'
                    'antigravity-hub/2.2.1-5287492581195776/windows-x64/Antigravity-x64.exe",'
                    'version:"2.1.4<br>June 11, 2026"'
                )
            )
        raise AssertionError(url)

    monkeypatch.delenv("ANTIGRAVITY_CLIENT_VERSION", raising=False)
    monkeypatch.setattr(antigravity_proxy.requests, "get", fake_get)

    assert antigravity_proxy._latest_antigravity_client_version() == "2.2.1"


def test_patch_runtime_antigravity_client_version(tmp_path):
    headers_dir = tmp_path / "src" / "utils"
    headers_dir.mkdir(parents=True)
    headers_file = headers_dir / "headers.ts"
    headers_file.write_text(
        'const ANTIGRAVITY_VERSION = "2.0.1";\n'
        'export const ua = `antigravity/${ANTIGRAVITY_VERSION}`;\n',
        encoding="utf-8",
    )

    assert antigravity_proxy._patch_runtime_antigravity_client_version(str(tmp_path), "2.2.1")
    assert 'const ANTIGRAVITY_VERSION = "2.2.1";' in headers_file.read_text(encoding="utf-8")


def test_patch_runtime_gemini35_flash_support(tmp_path):
    utils_dir = tmp_path / "src" / "utils"
    utils_dir.mkdir(parents=True)
    transform_file = utils_dir / "transform.ts"
    transform_file.write_text(
        '    const nativelySupported = [\n'
        '      "gemini-3.1-pro-preview",\n'
        '      "gemini-3.5-flash-low",\n'
        '      "gemini-3-flash",\n'
        '  ];\n'
        '        if (isNative) {\n'
        '            if (baseModel.includes("gemini-3.5-flash")) {\n'
        '                googleModel = "gemini-3.5-flash-low";\n'
        '            } else if (baseModel.includes("gemini-3.1-pro")) {\n'
        '                googleModel = `gemini-3.1-pro-${extractedTier || "high"}`;\n'
        '            } else if (baseModel.includes("gemini-3-pro")) {\n'
        '                googleModel = `gemini-3-pro-${extractedTier || "high"}`;\n'
        '            }\n'
        '        }\n',
        encoding="utf-8",
    )

    assert antigravity_proxy._patch_runtime_gemini35_flash_support(str(tmp_path))

    content = transform_file.read_text(encoding="utf-8")
    assert '"gemini-3.5-flash-high"' in content
    assert '"gemini-3.5-flash-medium"' in content
    assert '"gemini-3.5-flash-low"' in content
    assert '"gemini-3.5-flash",' in content
    assert 'googleModel = "gemini-3.5-flash-low";' not in content
    assert '`gemini-3.5-flash-${extractedTier || "medium"}`' in content


def test_patch_runtime_account_reset_support_clears_capabilities(tmp_path):
    server_file = tmp_path / "src" / "server.ts"
    manager_file = tmp_path / "src" / "auth" / "manager.ts"
    server_file.parent.mkdir(parents=True)
    manager_file.parent.mkdir(parents=True)
    server_file.write_text(
        "for (const acc of accounts) {\n"
        "            acc.modelScores = {};\n"
        "            acc.history = [];\n"
        "}\n",
        encoding="utf-8",
    )
    manager_file.write_text(
        "resetAccount(account) {\n"
        "        account.modelScores = {};\n"
        "        account.history = [];\n"
        "}\n",
        encoding="utf-8",
    )

    assert antigravity_proxy._patch_runtime_account_reset_support(str(tmp_path))
    assert "acc.capabilities = {};" in server_file.read_text(encoding="utf-8")
    assert "account.capabilities = {};" in manager_file.read_text(encoding="utf-8")


def test_patch_runtime_forced_account_support(tmp_path):
    server_file = tmp_path / "src" / "server.ts"
    manager_file = tmp_path / "src" / "auth" / "manager.ts"
    server_file.parent.mkdir(parents=True)
    manager_file.parent.mkdir(parents=True)
    server_file.write_text(
        'import { initManager, getBestAccount, updateAccountUsage, addAccount, getAccounts, removeAccount } from "./auth/manager";\n'
        '      const clientId = req.headers.get("x-client-id") || url.searchParams.get("client_id") || "unknown";\n'
        '            let account = await getBestAccount(useCliPool ? "cli" : "sandbox", openaiBody.model, clientId, triedEmails, true);\n'
        '            if (!account && !isSandboxOnlyModel && !isCliOnlyModel) {\n'
        '            }\n'
        '            if (!account) {\n'
        '                account = await getBestAccount(useCliPool ? "cli" : "sandbox", openaiBody.model, clientId, triedEmails, false);\n'
        '            }\n'
        '        while (attempts < MAX_ATTEMPTS) {\n',
        encoding="utf-8",
    )
    manager_file.write_text(
        "export function getAccounts() { return accounts; }\n"
        "async function ensureAccountReady(account: AntigravityAccount): Promise<AntigravityAccount | null> { return account; }\n",
        encoding="utf-8",
    )

    assert antigravity_proxy._patch_runtime_forced_account_support(str(tmp_path))

    server = server_file.read_text(encoding="utf-8")
    manager = manager_file.read_text(encoding="utf-8")
    assert "getAccountByEmail" in server
    assert "forcedAccountEmail" in server
    assert "X-Antigravity-Account" in server
    assert "!account && !forcedAccountEmail" in server
    assert "while (attempts < (forcedAccountEmail ? 1 : MAX_ATTEMPTS))" in server
    assert "export async function getAccountByEmail" in manager


def test_antigravity_exhausted_quota_is_not_retried(monkeypatch):
    client = _unified_antigravity_client()
    client.request_timeout = 300
    client._get_max_retries = lambda: 7
    client._is_stop_requested = lambda: False
    monkeypatch.setenv("GRACEFUL_STOP", "0")
    monkeypatch.setenv("GRACEFUL_STOP_COMPLETED", "0")
    monkeypatch.delenv("TRANSLATION_CANCELLED", raising=False)
    monkeypatch.setattr(unified_api_client, "ANTIGRAVITY_AVAILABLE", True)
    monkeypatch.setattr(
        unified_api_client,
        "_antigravity_ensure_running",
        lambda log_fn=None: {"running": True},
    )
    calls = []

    def exhausted_send(**_kwargs):
        calls.append(True)
        raise RuntimeError(
            "Antigravity: HTTP 429 - Quota exhausted: Individual quota reached. "
            "Resets in 61h20m31s."
        )

    monkeypatch.setattr(unified_api_client, "_antigravity_send_stream", exhausted_send)

    with pytest.raises(UnifiedClientError) as exc_info:
        client._send_antigravity([], 0.2, 64000, "response.txt")

    assert exc_info.value.error_type == "rate_limit"
    assert calls == [True]


def test_patch_runtime_verbose_access_denied_preserves_upstream_details(tmp_path):
    server_file = tmp_path / "src" / "server.ts"
    errors_file = tmp_path / "src" / "utils" / "errors.ts"
    server_file.parent.mkdir(parents=True)
    errors_file.parent.mkdir(parents=True)
    server_file.write_text(
        "const attemptLogs: Array<{ email: string, status: number, reason: string }> = [];\n"
        "attemptLogs.push({ email: account.email, status, reason: parsedError.reason });\n"
        '                   await updateAccountUsage(account.email, false, openaiBody.model, useCliPool ? "cli" : "sandbox", clientId, status);\n'
        '                   return new Response(JSON.stringify({ \n'
        '                       error: { message: "Access denied: " + parsedError.reason, type: "access_denied", code: status.toString() } \n'
        '                   }), { \n'
        '                       status, \n'
        '                       headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*", "X-Antigravity-Attempts": attempts.toString() } \n'
        '                   });\n',
        encoding="utf-8",
    )
    errors_file.write_text(
        'export function parseGoogleError(body: string): { message?: string; status: number; } {\n'
        '  let reason = "unknown_error";\n'
        "  let validationUrl: string | undefined;\n"
        "  let isQuotaExhausted = false;\n"
        "  let isChallengeRequired = false;\n"
        "  let isModelUnsupported = false;\n"
        "  let status = 500;\n"
        "  let message: string | undefined;\n"
        "  const json = JSON.parse(body);\n"
        "  const err = json.error;\n"
        "  if (err) {\n"
        "      message = err.message;\n"
        '      if (err.status === "RESOURCE_EXHAUSTED" || err.message?.includes("quota")) {\n'
        "        isQuotaExhausted = true;\n"
        '        reason = "quota_exhausted";\n'
        "        status = 429;\n"
        "      }\n"
        "      if (err.details) {\n"
        "        for (const detail of err.details) {\n"
        '          if (detail.reason === "RATE_LIMIT_EXCEEDED") {\n'
        "            isQuotaExhausted = true;\n"
        '            reason = "quota_exhausted";\n'
        "            status = 429;\n"
        "          }\n"
        "        }\n"
        "      }\n"
        "  }\n"
        "  return { reason, validationUrl, isQuotaExhausted, isChallengeRequired, isModelUnsupported, status };\n"
        "}\n",
        encoding="utf-8",
    )

    assert antigravity_proxy._patch_runtime_verbose_access_denied(str(tmp_path))

    server = server_file.read_text(encoding="utf-8")
    errors = errors_file.read_text(encoding="utf-8")
    assert 'message: "Access denied: " + parsedError.reason' not in server
    assert "accessDeniedMessage" in server
    assert "body: errText.slice(0, 2000)" in server
    assert "hasQuotaAttempt" in server
    assert "Quota exhausted:" in server
    assert "status: accessDeniedStatus" in server
    assert "attempts: responseAttempts" in server
    assert "google_body" in server
    assert "insufficient_quota" in server
    assert "combinedErrorText" in errors
    assert "detailText" in errors
    assert "status, message" in errors


def test_account_summary_strips_tokens_and_reports_unsupported_models():
    summary = antigravity_proxy._safe_account_summary(
        {
            "email": "user@example.test",
            "accessToken": "secret-access-token",
            "refreshToken": "secret-refresh-token",
            "projectId": "project-id",
            "healthScore": 42,
            "quota": [{"groupName": "Gemini", "quotaLeft": "100%", "resetIn": "1h"}],
            "modelScores": {"antigravity-gemini-3.5-flash-medium|sandbox": 90},
            "capabilities": {"antigravity-gemini-3.5-flash-high": False},
        }
    )

    assert "accessToken" not in summary
    assert "refreshToken" not in summary
    assert summary["email"] == "user@example.test"
    assert summary["quota"][0]["name"] == "Gemini"
    assert summary["unsupported_models"] == ["antigravity-gemini-3.5-flash-high"]


def test_stored_account_summary_detects_login_without_proxy_and_strips_tokens(tmp_path, monkeypatch):
    accounts_file = tmp_path / "antigravity-accounts.json"
    accounts_file.write_text(
        json.dumps(
            {
                "accounts": [
                    {
                        "email": "stored@example.test",
                        "accessToken": "secret-access-token",
                        "refreshToken": "secret-refresh-token",
                        "projectId": "project-id",
                        "healthScore": 88,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTIGRAVITY_ACCOUNTS_FILE", str(accounts_file))
    monkeypatch.delenv("ACCOUNTS_FILE", raising=False)

    summary = antigravity_proxy.get_stored_account_summary()

    assert summary["healthy"] is True
    assert summary["stored"] is True
    assert summary["accounts"][0]["email"] == "stored@example.test"
    assert "accessToken" not in summary["accounts"][0]
    assert "refreshToken" not in summary["accounts"][0]
    assert antigravity_proxy._account_email_for_id(1) == "stored@example.test"


def test_find_proxy_launch_command_uses_npx_bun_for_downloaded_runtime(tmp_path, monkeypatch):
    def fake_candidate(name):
        return "npx" if name == "npx" else None

    runtime_dir = str(tmp_path)
    monkeypatch.delenv("ANTIGRAVITY_PROXY_LAUNCH_CMD", raising=False)
    monkeypatch.setattr(antigravity_proxy, "_candidate_executable", fake_candidate)

    cmd = antigravity_proxy._find_proxy_launch_command(runtime_dir)

    assert cmd[:5] == ["npx", "--yes", "--package", antigravity_proxy.BUN_NPM_PACKAGE, "bun"]
    assert cmd[-2] == "run"
    assert cmd[-1].endswith("src\\server.ts") or cmd[-1].endswith("src/server.ts")


def test_write_proxy_runtime_package_json_updates_stale_version(tmp_path):
    package_json = tmp_path / "package.json"
    package_json.write_text(
        json.dumps({"name": "glossarion-antigravity-proxy-data", "version": "0.0.0"}),
        encoding="utf-8",
    )

    antigravity_proxy._write_proxy_runtime_package_json(str(tmp_path), version="1.7.1")

    data = json.loads(package_json.read_text(encoding="utf-8"))
    assert data["name"] == "antigravity-proxy"
    assert data["version"] == "1.7.1"
    assert data["private"] is True
