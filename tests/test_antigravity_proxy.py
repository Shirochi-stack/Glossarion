import json
import sys
import threading
import time
import types

import pytest

import antigravity_proxy
import unified_api_client
from html_output_utils import ensure_utf8_html_document
from installer_utils import run_logged_subprocess
from model_options import get_model_options
from unified_api_client import UnifiedClient, UnifiedClientError


@pytest.fixture(autouse=True)
def _reset_antigravity_cancel_state():
    antigravity_proxy.reset_cancel()
    antigravity_proxy.allow_proxy_update_retry_for_new_run()
    antigravity_proxy.set_proxy_started_callback(None)
    antigravity_proxy._proxy_last_update_check_at = 0.0
    yield
    antigravity_proxy.reset_cancel()
    antigravity_proxy.allow_proxy_update_retry_for_new_run()
    antigravity_proxy.set_proxy_started_callback(None)
    antigravity_proxy._proxy_last_update_check_at = 0.0


def test_proxy_started_callback_is_optional_and_notified_once():
    calls = []
    antigravity_proxy.set_proxy_started_callback(lambda: calls.append("ready"))

    antigravity_proxy._notify_proxy_started()

    assert calls == ["ready"]


def test_manual_proxy_start_can_suppress_automatic_started_callback(tmp_path, monkeypatch):
    calls = []
    health_checks = iter([
        {"healthy": False},
        {"healthy": False},
        {"healthy": True},
    ])

    class FakeProcess:
        pid = 1234

        @staticmethod
        def poll():
            return None

    monkeypatch.setattr(antigravity_proxy, "_proxy_process", None)
    monkeypatch.setattr(antigravity_proxy, "_ensure_proxy_config", lambda: str(tmp_path))
    monkeypatch.setattr(antigravity_proxy, "check_proxy_health", lambda: next(health_checks))
    monkeypatch.setattr(
        antigravity_proxy,
        "_ensure_proxy_runtime",
        lambda *_args, **_kwargs: str(tmp_path),
    )
    monkeypatch.setattr(
        antigravity_proxy,
        "_find_proxy_launch_command",
        lambda _runtime_dir: ["fake-proxy"],
    )
    monkeypatch.setattr(antigravity_proxy.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())
    monkeypatch.setattr(antigravity_proxy.time, "sleep", lambda _seconds: None)
    antigravity_proxy.set_proxy_started_callback(lambda: calls.append("ready"))

    status = antigravity_proxy.ensure_proxy_running(notify_started=False)

    assert status == {"running": True, "auto_launched": True}
    assert calls == []


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


def test_proxy_archive_download_prefers_curl_and_uses_ninety_second_timeout(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return types.SimpleNamespace(
            returncode=0,
            stdout=b"PK\x03\x04archive",
            stderr=b"",
        )

    monkeypatch.setattr(
        antigravity_proxy,
        "_candidate_executable",
        lambda name: "curl.exe" if name == "curl" else None,
    )
    monkeypatch.setattr(antigravity_proxy.subprocess, "run", fake_run)
    monkeypatch.setattr(
        antigravity_proxy.requests,
        "get",
        lambda *_args, **_kwargs: pytest.fail("requests fallback should not run"),
    )

    archive = antigravity_proxy._download_proxy_archive_bytes(
        "https://codeload.github.com/example/repo/zip/revision"
    )

    assert archive == b"PK\x03\x04archive"
    command = captured["command"]
    assert command[command.index("--connect-timeout") + 1] == "30"
    assert command[command.index("--max-time") + 1] == "90"
    assert captured["kwargs"]["timeout"] == 100


def test_proxy_archive_download_falls_back_to_requests(monkeypatch):
    captured = {}

    def fake_get(url, headers=None, timeout=None):
        captured.update(url=url, headers=headers, timeout=timeout)
        return FakeHTTPResponse(content=b"PK\x03\x04archive")

    monkeypatch.setattr(antigravity_proxy, "_candidate_executable", lambda _name: None)
    monkeypatch.setattr(antigravity_proxy.requests, "get", fake_get)

    archive = antigravity_proxy._download_proxy_archive_bytes(
        "https://codeload.github.com/example/repo/zip/revision"
    )

    assert archive == b"PK\x03\x04archive"
    assert captured["timeout"] == pytest.approx(90, abs=0.1)


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
        antigravity_proxy._normalize_model_name("antigravity/gemini-3.1-pro-high")
        == "antigravity-gemini-3.1-pro-high"
    )
    assert (
        antigravity_proxy._normalize_model_name("antigravity/gemini-3.5-flash-medium")
        == "antigravity-gemini-3.5-flash-medium"
    )
    assert (
        antigravity_proxy._normalize_model_name("antigravity/gemini-3.5-flash-high")
        == "antigravity-gemini-3.5-flash-high"
    )
    assert (
        antigravity_proxy._normalize_model_name("antigravity/gemini-3.7-flash-medium")
        == "antigravity-gemini-3.7-flash-medium"
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
    assert parsed["provider_finish_reason"] == "stop"
    assert parsed["finish_reason_observed"] is True
    assert parsed["usage"]["total_tokens"] == 5
    assert parsed["raw_response"] is data


@pytest.mark.parametrize("provider_finish_reason", ["length", "MAX_TOKENS", 2])
def test_parse_openai_chat_response_detects_explicit_length(provider_finish_reason):
    data = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "partial"},
                "finish_reason": provider_finish_reason,
            }
        ]
    }

    parsed = antigravity_proxy._parse_openai_chat_response(data)

    assert parsed["finish_reason"] == "length"
    assert parsed["provider_finish_reason"] == provider_finish_reason


def test_parse_openai_chat_response_rejects_missing_finish_reason():
    data = {
        "choices": [
            {"message": {"role": "assistant", "content": "partial"}}
        ]
    }

    with pytest.raises(RuntimeError, match="without an explicit finish_reason"):
        antigravity_proxy._parse_openai_chat_response(data)


def test_parse_openai_chat_response_preserves_missing_provider_reason():
    data = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "blocked"},
                "finish_reason": "content_filter",
            }
        ],
        "provider_finish_reason": None,
        "provider_block_reason": "GOOGLE_PROHIBITED_USE_POLICY_MESSAGE",
    }

    parsed = antigravity_proxy._parse_openai_chat_response(data)

    assert parsed["finish_reason"] == "content_filter"
    assert parsed["provider_finish_reason"] is None
    assert parsed["provider_block_reason"] == "GOOGLE_PROHIBITED_USE_POLICY_MESSAGE"


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
    assert result["provider_finish_reason"] == "stop"
    assert result["finish_reason_observed"] is True
    assert result["stream_done_observed"] is True
    assert result["usage"]["total_tokens"] == 5
    assert response.closed is True


def test_consume_openai_stream_logs_account_selected_by_auto_rotation(monkeypatch):
    response = FakeStreamResponse(
        [
            _sse_event({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}),
            "data: [DONE]",
        ]
    )
    response.headers = {"X-Antigravity-Account": "auto@example.test"}
    logs = []
    monkeypatch.setattr(
        antigravity_proxy,
        "get_stored_account_summary",
        lambda: {
            "accounts": [
                {"email": "first@example.test"},
                {"email": "auto@example.test"},
            ]
        },
    )

    result = antigravity_proxy._consume_openai_stream(
        response,
        log_fn=logs.append,
        log_stream=False,
        account_id=0,
    )

    assert result["content"] == "ok"
    assert (
        "🧭 Antigravity: automatic rotation selected account slot #2 "
        "(auto@example.test)"
    ) in logs


@pytest.mark.parametrize("provider_finish_reason", ["length", "MAX_TOKENS", 2])
def test_consume_openai_stream_detects_only_explicit_length(provider_finish_reason):
    response = FakeStreamResponse(
        [
            _sse_event({"choices": [{"delta": {"content": "partial"}, "finish_reason": None}]}),
            _sse_event({"choices": [{"delta": {}, "finish_reason": provider_finish_reason}]}),
            "data: [DONE]",
        ]
    )

    result = antigravity_proxy._consume_openai_stream(
        response,
        log_fn=lambda _: None,
        log_stream=False,
    )

    assert result["content"] == "partial"
    assert result["finish_reason"] == "length"
    assert result["provider_finish_reason"] == provider_finish_reason
    assert response.closed is True


def test_consume_openai_stream_does_not_guess_length_from_usage():
    response = FakeStreamResponse(
        [
            _sse_event(
                {
                    "choices": [{"delta": {"content": "partial"}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 1000,
                        "total_tokens": 1100,
                    },
                }
            ),
            "data: [DONE]",
        ]
    )

    result = antigravity_proxy._consume_openai_stream(
        response,
        log_fn=lambda _: None,
        log_stream=False,
    )

    assert result["finish_reason"] == "stop"
    assert result["provider_finish_reason"] == "stop"


def test_consume_openai_stream_rejects_missing_finish_reason():
    response = FakeStreamResponse(
        [
            _sse_event({"choices": [{"delta": {"content": "partial"}, "finish_reason": None}]}),
            "data: [DONE]",
        ]
    )

    with pytest.raises(RuntimeError, match="without an explicit finish_reason"):
        antigravity_proxy._consume_openai_stream(
            response,
            log_fn=lambda _: None,
            log_stream=False,
        )

    assert response.closed is True


def test_consume_openai_stream_preserves_missing_provider_reason():
    logs = []
    response = FakeStreamResponse(
        [
            _sse_event(
                {
                    "choices": [
                        {
                            "delta": {"content": "blocked"},
                            "finish_reason": "content_filter",
                        }
                    ],
                    "provider_finish_reason": None,
                    "provider_block_reason": "GOOGLE_PROHIBITED_USE_POLICY_MESSAGE",
                }
            ),
            "data: [DONE]",
        ]
    )

    result = antigravity_proxy._consume_openai_stream(
        response,
        log_fn=logs.append,
        log_stream=False,
    )

    assert result["finish_reason"] == "content_filter"
    assert result["provider_finish_reason"] is None
    assert result["provider_block_reason"] == "GOOGLE_PROHIBITED_USE_POLICY_MESSAGE"
    assert any(log.startswith("🛡️ Antigravity: terminal metadata ") for log in logs)


def test_forced_antigravity_stream_includes_reasoning_when_thinking_toggle_is_off(monkeypatch):
    response = FakeStreamResponse(
        [
            _sse_event({"choices": [{"delta": {"reasoning_content": "Think live"}}]}),
            _sse_event({"choices": [{"delta": {"content": "Answer"}, "finish_reason": "stop"}]}),
            "data: [DONE]",
        ]
    )
    logs = []
    monkeypatch.setenv("STREAM_THINKING_LOGS", "0")

    result = antigravity_proxy._consume_openai_stream(
        response,
        log_fn=logs.append,
        log_stream=True,
    )

    assert result["content"] == "Answer"
    assert any("Thinking" in line for line in logs)
    assert any("Think live" in line for line in logs)


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


def test_cancelled_stream_cannot_revive_after_new_run_resets_cancel_flag():
    entered_stream = threading.Event()
    release_stream = threading.Event()
    errors = []
    logs = []

    class DelayedOldResponse(FakeStreamResponse):
        def iter_lines(self, decode_unicode=True, chunk_size=1):
            entered_stream.set()
            release_stream.wait(2)
            yield _sse_event({
                "choices": [{"delta": {"content": "STALE OUTPUT"}, "finish_reason": "stop"}]
            })
            yield "data: [DONE]"

    response = DelayedOldResponse([])
    old_generation = antigravity_proxy.capture_cancel_generation()

    def consume_old_stream():
        try:
            antigravity_proxy._consume_openai_stream(
                response,
                log_fn=logs.append,
                log_stream=True,
                cancel_generation=old_generation,
            )
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=consume_old_stream)
    worker.start()
    assert entered_stream.wait(1)

    antigravity_proxy.cancel_stream()
    assert response.closed is True
    antigravity_proxy.reset_cancel()
    assert antigravity_proxy.is_cancelled() is False

    release_stream.set()
    worker.join(2)

    assert worker.is_alive() is False
    assert len(errors) == 1
    assert "cancelled by user" in str(errors[0]).lower()
    assert all("STALE OUTPUT" not in line for line in logs)
    assert antigravity_proxy.is_cancel_generation_cancelled(old_generation) is True


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


def test_antigravity_lifecycle_logs_connect_before_ready_and_progress_after(monkeypatch):
    client = _unified_antigravity_client("antigravity0/gemini-3.6-flash-low")
    client.request_timeout = 300
    client._is_stop_requested = lambda: False
    client._should_abort_retry = lambda: False
    client._get_max_retries = lambda: 1
    client._should_show_api_lifecycle_logs = lambda: True
    client._get_thinking_status_label = lambda: ""
    client._get_thread_local_client = lambda: types.SimpleNamespace(
        current_request_label="Chapter 12",
        current_request_context="translation",
        output_token_limit=None,
        per_key_max_output_tokens=None,
    )

    events = []
    client._debug_log = lambda message: events.append(message)

    def ensure_running(log_fn=None):
        events.append("proxy_ready")
        return {"running": True}

    def successful_send(**_kwargs):
        events.append("send")
        return {"content": "ok", "finish_reason": "stop", "usage": None}

    monkeypatch.setenv("GRACEFUL_STOP", "0")
    monkeypatch.delenv("TRANSLATION_CANCELLED", raising=False)
    monkeypatch.setattr(unified_api_client, "ANTIGRAVITY_AVAILABLE", True)
    monkeypatch.setattr(unified_api_client, "_antigravity_send", lambda **_kwargs: None)
    monkeypatch.setattr(unified_api_client, "_antigravity_ensure_running", ensure_running)
    monkeypatch.setattr(unified_api_client, "_antigravity_send_stream", successful_send)

    result = client._send_antigravity([], 0.2, 64000, "response.txt")

    connecting_index = next(
        index for index, event in enumerate(events)
        if "Connecting to Antigravity proxy" in event
    )
    progress_index = next(
        index for index, event in enumerate(events)
        if "API call in progress" in event
    )
    assert result.content == "ok"
    assert connecting_index < events.index("proxy_ready")
    assert events.index("proxy_ready") < progress_index < events.index("send")


def test_antigravity_forced_stream_logs_ignore_general_streaming_toggle(monkeypatch):
    client = _unified_antigravity_client()
    client.request_timeout = 300
    client._should_abort_retry = lambda: False
    client._get_max_retries = lambda: 1
    client._streaming_enabled = lambda: False

    captured = {}

    def successful_send(**kwargs):
        captured.update(kwargs)
        return {"content": "LIVE", "finish_reason": "stop", "usage": None}

    monkeypatch.setenv("BATCH_TRANSLATION", "0")
    monkeypatch.setenv("ENABLE_STREAMING", "0")
    monkeypatch.setenv("LOG_STREAM_CHUNKS", "1")
    monkeypatch.setattr(unified_api_client, "ANTIGRAVITY_AVAILABLE", True)
    monkeypatch.setattr(unified_api_client, "_antigravity_send", lambda **_kwargs: None)
    monkeypatch.setattr(
        unified_api_client,
        "_antigravity_ensure_running",
        lambda log_fn=None: {"running": True},
    )
    monkeypatch.setattr(unified_api_client, "_antigravity_is_cancelled", lambda: False)
    monkeypatch.setattr(unified_api_client, "_antigravity_send_stream", successful_send)

    result = client._send_antigravity([], 0.2, 64000, "response.txt")

    assert result.content == "LIVE"
    assert captured["log_stream"] is True


def test_antigravity_fresh_request_uses_explicit_lifecycle_cancel_reset(monkeypatch):
    """Only the new-run lifecycle reset may enable a fresh request."""
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
    unified_api_client.set_stop_flag(False)
    assert antigravity_proxy.is_cancelled() is False

    result = client._send_antigravity([], 0.2, 64000, "response.txt")

    assert result.content == "ok"
    assert send_cancel_states == [False]
    assert antigravity_proxy.is_cancelled() is False


def test_antigravity_cancelled_generation_cannot_retry_after_reset(monkeypatch):
    client = _unified_antigravity_client("antigravity/gemini-3.7-flash-high")
    client.request_timeout = 300
    client._is_stop_requested = lambda: False
    client._should_abort_retry = lambda: False
    client._get_max_retries = lambda: 2
    client._get_send_interval = lambda: 0
    client._sleep_with_cancel = lambda *_args, **_kwargs: True

    monkeypatch.setattr(unified_api_client, "ANTIGRAVITY_AVAILABLE", True)
    monkeypatch.setattr(
        unified_api_client,
        "_antigravity_ensure_running",
        lambda log_fn=None: {"running": True},
    )
    monkeypatch.setattr(unified_api_client, "_antigravity_send", lambda **_kwargs: None)

    send_calls = []

    def interrupted_send(**kwargs):
        send_calls.append(kwargs["cancel_generation"])
        antigravity_proxy.cancel_stream()
        antigravity_proxy.reset_cancel()
        raise RuntimeError("socket interrupted")

    monkeypatch.setattr(unified_api_client, "_antigravity_send_stream", interrupted_send)

    with pytest.raises(UnifiedClientError) as exc_info:
        client._send_antigravity([], 0.2, 64000, "response.txt")

    assert exc_info.value.error_type == "cancelled"
    assert len(send_calls) == 1
    assert antigravity_proxy.is_cancel_generation_cancelled(send_calls[0]) is True


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


def test_disable_temperature_omits_parameter_from_request_payloads(monkeypatch):
    monkeypatch.setenv("DISABLE_TEMPERATURE", "1")
    client = UnifiedClient.__new__(UnifiedClient)
    client._get_active_request_model = lambda: "example-model"
    client._is_o_series_model = lambda: False

    effective_temperature = client._effective_temperature(0.3)
    openai_payload = client._build_openai_params(
        [{"role": "user", "content": "hello"}],
        effective_temperature,
        1024,
    )
    anthropic_payload = client._build_anthropic_payload(
        [{"role": "user", "content": "hello"}],
        effective_temperature,
        1024,
        {},
    )
    antigravity_payload = antigravity_proxy._payload_for_openai_chat(
        [], "claude-sonnet-4-6", effective_temperature, 1024, False
    )

    assert effective_temperature is None
    assert "temperature" not in openai_payload
    assert "temperature" not in anthropic_payload
    assert "temperature" not in antigravity_payload


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


def test_numbered_antigravity_prefix_controls_account_routing(monkeypatch):
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
    assert antigravity_proxy._extract_antigravity_account_id("antigravity0/gemini-2.5-flash") == 0
    assert antigravity_proxy._extract_antigravity_account_id("antigravity1/gemini-2.5-flash") == 2
    assert antigravity_proxy._extract_antigravity_account_id("antigravity12/gemini-2.5-flash") == 13

    rotating_headers = antigravity_proxy._build_headers(account_id=0)
    assert "X-Antigravity-Account" not in rotating_headers
    assert "X-Client-Id" not in rotating_headers
    assert rotating_headers["X-Antigravity-Rotation"] == "round-robin"

    headers = antigravity_proxy._build_headers(account_id=2)

    assert headers["X-Antigravity-Account"] == "second@example.test"
    assert headers["X-Client-Id"] == "glossarion-antigravity2"
    assert (
        antigravity_proxy._account_slot_log_message(2, headers)
        == "🧭 Antigravity: using account slot #2 (second@example.test)"
    )


def test_antigravity_zero_prefix_reaches_proxy_without_forced_account(monkeypatch):
    captured = {}

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "choices": [
                    {
                        "message": {"content": "rotated"},
                        "finish_reason": "stop",
                    }
                ]
            }

    monkeypatch.setattr(
        antigravity_proxy,
        "_ensure_proxy_for_request",
        lambda _log: captured.setdefault("update_checked", True),
    )
    monkeypatch.setattr(antigravity_proxy, "_ensure_proxy_log_forwarder", lambda _log: None)
    monkeypatch.setattr(
        antigravity_proxy,
        "_ensure_account_slot_available",
        lambda account_id, *_args: captured.setdefault("account_id", account_id),
    )

    def fake_post(_payload, **kwargs):
        captured["headers"] = kwargs["headers"]
        return Response()

    monkeypatch.setattr(antigravity_proxy, "_post_chat", fake_post)

    result = antigravity_proxy.send_message(
        [{"role": "user", "content": "hello"}],
        model="antigravity0/gemini-2.5-flash",
        log_fn=lambda _message: None,
    )

    assert result["content"] == "rotated"
    assert captured["update_checked"] is True
    assert captured["account_id"] == 0
    assert "X-Antigravity-Account" not in captured["headers"]
    assert "X-Client-Id" not in captured["headers"]
    assert captured["headers"]["X-Antigravity-Rotation"] == "round-robin"
    client = _unified_antigravity_client("antigravity0/gemini-2.5-flash")
    assert client._extract_antigravity_account_id(client.model) == 0


def test_stream_request_checks_proxy_updater_before_opening_http_stream(monkeypatch):
    calls = []

    def stop_after_update_check(_log):
        calls.append("update")
        raise RuntimeError("update-check-sentinel")

    monkeypatch.setattr(
        antigravity_proxy,
        "_ensure_proxy_for_request",
        stop_after_update_check,
    )

    with pytest.raises(RuntimeError, match="update-check-sentinel"):
        antigravity_proxy.send_message_stream(
            [{"role": "user", "content": "hello"}],
            model="antigravity/gemini-3.7-flash-medium",
            log_fn=lambda _message: None,
        )

    assert calls == ["update"]


def test_proxy_request_readiness_propagates_update_failure(monkeypatch):
    monkeypatch.setattr(
        antigravity_proxy,
        "ensure_proxy_running",
        lambda log_fn=None: {"running": False, "error": "download failed"},
    )

    with pytest.raises(RuntimeError, match="download failed"):
        antigravity_proxy._ensure_proxy_for_request()


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
    monkeypatch.setattr(
        antigravity_proxy,
        "_wait_for_cancel",
        lambda _seconds, _generation=None: False,
    )
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
        lambda _seconds, _generation=None: (antigravity_proxy.cancel_stream() or True),
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
        "antigravity/gemini-3.6-flash-low",
        "antigravity/gemini-3.6-flash-medium",
        "antigravity/gemini-3.6-flash-high",
        "antigravity/gemini-3.7-flash-low",
        "antigravity/gemini-3.7-flash-medium",
        "antigravity/gemini-3.7-flash-high",
        "antigravity/gemini-3.5-flash-extra-low",
        "antigravity/gemini-3.5-flash-low",
        "antigravity/gemini-3.5-flash-medium",
        "antigravity/gemini-3.5-flash-high",
        "antigravity/gemini-3.1-pro-high",
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


def test_latest_proxy_release_uses_fork_main_revision_without_git(monkeypatch):
    def fake_get(url, headers=None, timeout=15):
        if url == antigravity_proxy.PROXY_GITHUB_API_MAIN:
            return FakeHTTPResponse(json_data={"sha": "a" * 40})
        assert url == antigravity_proxy.PROXY_GITHUB_RAW_PACKAGE_URL.format(
            revision="a" * 40
        )
        return FakeHTTPResponse(json_data={"version": "0.7.2"})

    monkeypatch.delenv("ANTIGRAVITY_PROXY_TAG", raising=False)
    monkeypatch.delenv("ANTIGRAVITY_PROXY_VERSION", raising=False)
    monkeypatch.setattr(antigravity_proxy.requests, "get", fake_get)

    release = antigravity_proxy._latest_proxy_release()

    assert release["version"] == "0.7.2"
    assert release["tag"] == f"main-{'a' * 12}"
    assert release["resolved"] is True
    assert release["archive_url"] == (
        "https://codeload.github.com/Shirochi-stack/antigravity-proxy/zip/"
        f"{'a' * 40}"
    )


def test_cached_runtime_update_detects_new_fork_revision_once_per_interval(
    tmp_path, monkeypatch
):
    runtime_dir = tmp_path / "runtime" / "main-old"
    runtime_dir.mkdir(parents=True)
    release_calls = []

    monkeypatch.setattr(antigravity_proxy, "_cached_runtime_needs_patch", lambda _data: False)
    monkeypatch.setattr(
        antigravity_proxy,
        "_latest_existing_runtime",
        lambda _root: str(runtime_dir),
    )
    monkeypatch.setattr(
        antigravity_proxy,
        "_read_runtime_metadata",
        lambda _runtime: {"tag": "main-old", "version": "0.7.1"},
    )
    monkeypatch.setattr(
        antigravity_proxy,
        "_latest_proxy_release",
        lambda: release_calls.append(True) or {
            "tag": "main-new",
            "version": "0.7.2",
            "resolved": True,
        },
    )
    monkeypatch.setattr(antigravity_proxy.time, "monotonic", lambda: 1000.0)

    assert antigravity_proxy._cached_runtime_needs_update(str(tmp_path)) is True
    assert antigravity_proxy._cached_runtime_needs_update(str(tmp_path)) is False
    assert len(release_calls) == 1


def test_cached_runtime_update_ignores_unresolved_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(antigravity_proxy, "_cached_runtime_needs_patch", lambda _data: False)
    monkeypatch.setattr(
        antigravity_proxy,
        "_latest_proxy_release",
        lambda: {
            "tag": "main-fallback",
            "version": "0.7.2",
            "resolved": False,
        },
    )
    monkeypatch.setattr(antigravity_proxy.time, "monotonic", lambda: 1000.0)

    assert antigravity_proxy._cached_runtime_needs_update(str(tmp_path)) is False


def test_cached_runtime_update_compares_running_proxy_version(tmp_path, monkeypatch):
    monkeypatch.setattr(antigravity_proxy, "_cached_runtime_needs_patch", lambda _data: False)
    monkeypatch.setattr(
        antigravity_proxy,
        "_latest_proxy_release",
        lambda: {
            "tag": "main-current",
            "version": "0.7.2",
            "resolved": True,
        },
    )
    monkeypatch.setattr(antigravity_proxy.time, "monotonic", lambda: 1000.0)

    assert antigravity_proxy._cached_runtime_needs_update(
        str(tmp_path), running_version="0.7.1"
    ) is True


def test_failed_download_with_patched_cache_retries_only_on_next_gui_run(
    tmp_path, monkeypatch
):
    existing_runtime = tmp_path / "runtime" / "main-old"
    existing_runtime.mkdir(parents=True)
    logs = []
    release = {
        "tag": "main-new",
        "version": "0.7.7",
        "resolved": True,
    }

    monkeypatch.setattr(antigravity_proxy, "_latest_proxy_release", lambda: release)
    monkeypatch.setattr(
        antigravity_proxy,
        "_latest_antigravity_client_version",
        lambda: "2.2.1",
    )
    monkeypatch.setattr(antigravity_proxy, "_write_proxy_runtime_package_json", lambda *_: None)
    monkeypatch.setattr(antigravity_proxy, "_runtime_metadata_matches", lambda *_: False)
    monkeypatch.setattr(
        antigravity_proxy,
        "_download_proxy_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("download timed out")),
    )
    monkeypatch.setattr(
        antigravity_proxy,
        "_latest_existing_runtime",
        lambda _root: str(existing_runtime),
    )
    monkeypatch.setattr(antigravity_proxy, "_patch_cached_runtime", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(antigravity_proxy.time, "monotonic", lambda: 4321.0)

    runtime = antigravity_proxy._ensure_proxy_runtime(
        str(tmp_path),
        log_fn=logs.append,
        force_update=True,
    )

    assert runtime == str(existing_runtime)
    assert antigravity_proxy._proxy_update_retry_blocked is True
    assert any("next Run Translation or Extract Glossary" in message for message in logs)

    monkeypatch.setattr(antigravity_proxy, "_cached_runtime_needs_patch", lambda _data: False)
    release_checks = []
    monkeypatch.setattr(
        antigravity_proxy,
        "_latest_proxy_release",
        lambda: release_checks.append(True) or release,
    )

    assert antigravity_proxy._cached_runtime_needs_update(
        str(tmp_path), running_version="0.7.6"
    ) is False
    assert release_checks == []

    antigravity_proxy.allow_proxy_update_retry_for_new_run()

    assert antigravity_proxy._cached_runtime_needs_update(
        str(tmp_path), running_version="0.7.6"
    ) is True
    assert release_checks == [True]


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


def test_runtime_fork_feature_check_rejects_relabelled_upstream_cache(tmp_path):
    runtime = tmp_path / "runtime"
    transform_file = runtime / "src" / "utils" / "transform.ts"
    transform_file.parent.mkdir(parents=True)
    transform_file.write_text("// upstream transform\n", encoding="utf-8")
    (runtime / "src" / "server.ts").write_text("// server\n", encoding="utf-8")
    (runtime / "package.json").write_text("{}\n", encoding="utf-8")

    assert not antigravity_proxy._runtime_has_fork_features(str(runtime))
    assert not antigravity_proxy._runtime_metadata_matches(
        str(runtime), "main-example", "2.2.1"
    )


def test_runtime_fork_feature_check_accepts_native_model_routing(tmp_path):
    transform_file = tmp_path / "src" / "utils" / "transform.ts"
    transform_file.parent.mkdir(parents=True)
    transform_file.write_text(
        "GEMINI_37_FLASH_ALIASES\n"
        "GEMINI_37_FLASH_WIRE_MODEL\n"
        "gemini-3.7-flash-high\n"
        "GEMINI_31_PRO_HIGH_WIRE_MODEL\n"
        "GEMINI_35_FLASH_ALIASES\n"
        "gemini35FlashWireModel\n",
        encoding="utf-8",
    )

    assert antigravity_proxy._runtime_has_fork_features(str(tmp_path))


def test_finish_reason_patch_accepts_native_prompt_block_wrapper(tmp_path):
    transform_file = tmp_path / "src" / "utils" / "transform.ts"
    transform_file.parent.mkdir(parents=True)
    transform_file.write_text(
        '''  const candidate = data.candidates[0];
  const finishReason = candidate.finishReason;
  let openaiFinishReason: string | null = null;
  if (hasPromptBlock || hasBlockedCandidateSafetyRating) {
    openaiFinishReason = "content_filter";
  } else if (finishReason) {
    if (toolCalls.length > 0 || hasPriorToolCalls) {
      openaiFinishReason = "tool_calls";
    } else if (finishReason === "STOP") {
      openaiFinishReason = "stop";
    } else if (finishReason === "MAX_TOKENS") {
      openaiFinishReason = "length";
    } else if (finishReason === "SAFETY") {
      openaiFinishReason = "content_filter";
    } else if (finishReason === "MALFORMED_FUNCTION_CALL") {
      openaiFinishReason = "tool_calls";
    } else {
      openaiFinishReason = "stop";
    }
  }
''',
        encoding="utf-8",
    )

    assert antigravity_proxy._patch_runtime_finish_reason_mapping(str(tmp_path))
    assert antigravity_proxy._patch_runtime_finish_reason_mapping(str(tmp_path))

    patched = transform_file.read_text(encoding="utf-8")
    assert "if (hasPromptBlock || hasBlockedCandidateSafetyRating)" in patched
    assert "candidate.finishReason ?? candidate.finish_reason" in patched
    assert "const finishReasonCode = Number(finishReason);" in patched
    assert "} else if (finishReason !== undefined && finishReason !== null)" in patched


def test_patch_runtime_prompt_blocks_to_content_filter(tmp_path):
    transform_file = tmp_path / "src" / "utils" / "transform.ts"
    transform_file.parent.mkdir(parents=True)
    transform_file.write_text(
        '''export function transformGoogleEventToOpenAI(googleData: any, model: string, requestId?: string) {
  const data = googleData.response || googleData;
  const requestIdActual = requestId || "chatcmpl-" + Math.random().toString(36).substring(7);
  const usage = data.usageMetadata ? {} : undefined;
  if (!data.candidates || data.candidates.length === 0) {
    if (usage) return { choices: [], usage: usage };
    return null;
  }
  const candidate = data.candidates[0];
  const parts = candidate.content?.parts || [];
  const finishReason = candidate.finishReason;
  if (parts.length === 0 && !finishReason && !usage) return null;
  const toolCalls: any[] = [];
  const hasPriorToolCalls = false;
  let openaiFinishReason: string | null = null;
  if (finishReason) {
    if (toolCalls.length > 0 || hasPriorToolCalls) openaiFinishReason = "tool_calls";
  }
  const extractedSignature = undefined;
  const extractedThought = undefined;
  return {
    choices: [{ finish_reason: openaiFinishReason }],
    usage: usage,
    _signature: extractedSignature,
    _thought: extractedThought
  };
}
''',
        encoding="utf-8",
    )

    assert antigravity_proxy._patch_runtime_prompt_block_finish_reason(str(tmp_path))
    assert antigravity_proxy._patch_runtime_prompt_block_finish_reason(str(tmp_path))

    patched = transform_file.read_text(encoding="utf-8")
    assert "data.promptFeedback || data.prompt_feedback" in patched
    assert 'finish_reason: "content_filter"' in patched
    assert "hasBlockedCandidateSafetyRating" in patched
    assert "provider_block_reason: promptBlockReasonText" in patched
    assert "provider_block_reason: hasPromptBlock" in patched
    assert "} else if (finishReason) {" in patched


def test_prompt_block_patch_accepts_native_canonical_policy_block(tmp_path):
    transform_file = tmp_path / "src" / "utils" / "transform.ts"
    transform_file.parent.mkdir(parents=True)
    transform_file.write_text(
        '''const hasPromptBlock = Boolean(promptBlockReasonText);
provider_block_reason: promptBlockReasonText;
const hasBlockedCandidateSafetyRating = true;
  if (parts.length === 0 && !finishReason && !usage && !hasPromptBlock && !hasBlockedCandidateSafetyRating) return null;
if (hasPromptBlock || hasBlockedCandidateSafetyRating || hasCanonicalGooglePolicyBlock) {
  openaiFinishReason = "content_filter";
}
provider_block_reason: hasPromptBlock;
''',
        encoding="utf-8",
    )

    assert antigravity_proxy._patch_runtime_prompt_block_finish_reason(str(tmp_path))
    assert antigravity_proxy._patch_runtime_prompt_block_finish_reason(str(tmp_path))

    patched = transform_file.read_text(encoding="utf-8")
    assert "hasCanonicalGooglePolicyBlock" in patched


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


def test_patch_runtime_forced_account_support_preserves_round_robin_layout(tmp_path):
    server_file = tmp_path / "src" / "server.ts"
    manager_file = tmp_path / "src" / "auth" / "manager.ts"
    server_file.parent.mkdir(parents=True)
    manager_file.parent.mkdir(parents=True)
    server_file.write_text(
        'import { initManager, getBestAccount, getAccounts, removeAccount } from "./auth/manager";\n'
        '      const requestedRotation = req.headers.get("x-antigravity-rotation")?.trim().toLowerCase();\n'
        '      const forceRoundRobin = requestedRotation === "round-robin";\n'
        '      const clientId = forceRoundRobin\n'
        '        ? undefined\n'
        '        : req.headers.get("x-client-id") || url.searchParams.get("client_id") || "unknown";\n'
        '        while (attempts < MAX_ATTEMPTS) {\n'
        '            let account = await getBestAccount(useCliPool ? "cli" : "sandbox", openaiBody.model, clientId, triedEmails, true, forceRoundRobin);\n'
        '            if (!account && !isSandboxOnlyModel && !isCliOnlyModel) {\n'
        '            }\n'
        '            if (!account) {\n'
        '                account = await getBestAccount(useCliPool ? "cli" : "sandbox", openaiBody.model, clientId, triedEmails, false, forceRoundRobin);\n'
        '            }\n',
        encoding="utf-8",
    )
    manager_file.write_text(
        "export function getAccounts() { return accounts; }\n"
        "async function ensureAccountReady(account: AntigravityAccount): Promise<AntigravityAccount | null> { return account; }\n",
        encoding="utf-8",
    )

    assert antigravity_proxy._patch_runtime_forced_account_support(str(tmp_path))

    server = server_file.read_text(encoding="utf-8")
    assert "forcedAccountEmail" in server
    assert "Forced account" in server
    assert "true, forceRoundRobin" in server
    assert "false, forceRoundRobin" in server
    assert "!account && !forcedAccountEmail" in server
    assert "while (attempts < (forcedAccountEmail ? 1 : MAX_ATTEMPTS))" in server


def test_patch_runtime_selected_account_header(tmp_path):
    server_file = tmp_path / "src" / "server.ts"
    server_file.parent.mkdir(parents=True)
    server_file.write_text(
        '                return new Response(readable, {\n'
        '                  status: 200,\n'
        '                  headers: {\n'
        '                    "Content-Type": "text/event-stream",\n'
        '                    "X-Antigravity-Attempts": attempts.toString()\n'
        '                  }\n'
        '                });\n'
        '                 return new Response(JSON.stringify(responseBody), {\n'
        '                   status: 200,\n'
        '                   headers: {\n'
        '                       "Content-Type": "application/json",\n'
        '                       "X-Antigravity-Attempts": attempts.toString()\n'
        '                   }\n'
        '                 });\n',
        encoding="utf-8",
    )

    assert antigravity_proxy._patch_runtime_selected_account_header(str(tmp_path))

    server = server_file.read_text(encoding="utf-8")
    assert server.count('"X-Antigravity-Account": account.email') == 2


def test_patch_runtime_auto_rotation_support(tmp_path):
    server_file = tmp_path / "src" / "server.ts"
    manager_file = tmp_path / "src" / "auth" / "manager.ts"
    server_file.parent.mkdir(parents=True)
    manager_file.parent.mkdir(parents=True)
    server_file.write_text(
        '      const clientId = req.headers.get("x-client-id") || url.searchParams.get("client_id") || "unknown";\n'
        '      const userIdent = openaiBody.user || clientId;\n'
        'account = await getBestAccount(pool, model, clientId, triedEmails, true);\n'
        'account = await getBestAccount(otherPool, model, clientId, triedEmails, true);\n'
        'account = await getBestAccount(pool, model, clientId, triedEmails, false);\n',
        encoding="utf-8",
    )
    manager_file.write_text(
        'const clientStickyMap = new Map<string, string>();\n'
        "export async function getBestAccount(pool?: 'cli' | 'sandbox', model?: string, clientId?: string, excludeEmails: string[] = [], skipRescue: boolean = false): Promise<AntigravityAccount | null> {\n"
        '    if (candidates.length === 0) return null;\n'
        '    \n'
        '    if (clientId && excludeEmails.length === 0) {\n'
        '    }\n'
        '}\n',
        encoding="utf-8",
    )

    assert antigravity_proxy._patch_runtime_auto_rotation_support(str(tmp_path))

    server = server_file.read_text(encoding="utf-8")
    manager = manager_file.read_text(encoding="utf-8")
    assert 'const forceRoundRobin = requestedRotation === "round-robin";' in server
    assert server.count("forceRoundRobin)") == 3
    assert "export function selectRoundRobinCandidate" in manager
    assert "if (forceRoundRobin)" in manager


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


def test_candidate_executable_finds_user_local_bun_install(tmp_path, monkeypatch):
    bun_root = tmp_path / "custom-bun"
    bun_executable = bun_root / "bin" / "bun"
    bun_executable.parent.mkdir(parents=True)
    bun_executable.write_text("", encoding="utf-8")

    monkeypatch.setattr(antigravity_proxy.sys, "platform", "linux")
    monkeypatch.setattr(antigravity_proxy.shutil, "which", lambda _name: None)
    monkeypatch.setenv("BUN_INSTALL", str(bun_root))

    assert antigravity_proxy._candidate_executable("bun") == str(bun_executable)


def test_automatic_bun_install_command_uses_official_windows_installer(monkeypatch):
    monkeypatch.delenv("ANTIGRAVITY_BUN_INSTALL_CMD", raising=False)
    monkeypatch.setattr(antigravity_proxy.sys, "platform", "win32")
    monkeypatch.setattr(
        antigravity_proxy,
        "_candidate_executable",
        lambda name: "powershell.exe" if name == "powershell" else None,
    )

    command = antigravity_proxy._automatic_bun_install_command()

    assert command[0] == "powershell.exe"
    assert "https://bun.sh/install.ps1" in command[-1]


def test_install_bun_automatically_runs_installer_and_redetects_bun(monkeypatch):
    state = {"installed": False}
    calls = []
    logs = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        kwargs["log_fn"]("Downloading Bun...")
        state["installed"] = True
        return {"returncode": 0, "output": "Bun installed", "timed_out": False}

    monkeypatch.setenv("ANTIGRAVITY_BUN_INSTALL_CMD", "install-bun --quiet")
    monkeypatch.setattr(antigravity_proxy, "run_logged_subprocess", fake_run)
    monkeypatch.setattr(
        antigravity_proxy,
        "_candidate_executable",
        lambda name: "/home/test/.bun/bin/bun" if name == "bun" and state["installed"] else None,
    )

    result = antigravity_proxy._install_bun_automatically(log_fn=logs.append)

    assert result["installed"] is True
    assert result["executable"] == "/home/test/.bun/bin/bun"
    assert calls[0][0] == ["install-bun", "--quiet"]
    assert calls[0][1]["timeout"] == antigravity_proxy.BUN_INSTALL_TIMEOUT_SECONDS
    assert any("installing Bun automatically" in message for message in logs)
    assert "Downloading Bun..." in logs
    assert any("Bun installed successfully" in message for message in logs)


def test_install_bun_automatically_reports_installer_failure(monkeypatch):
    logs = []

    monkeypatch.setenv("ANTIGRAVITY_BUN_INSTALL_CMD", "install-bun")
    monkeypatch.setattr(
        antigravity_proxy,
        "run_logged_subprocess",
        lambda command, **_kwargs: {
            "returncode": 7,
            "output": "download blocked",
            "timed_out": False,
        },
    )

    result = antigravity_proxy._install_bun_automatically(log_fn=logs.append)

    assert result["installed"] is False
    assert "code 7" in result["error"]
    assert "download blocked" in result["error"]
    assert any("download blocked" in message for message in logs)


def test_installer_runner_streams_stdout_and_stderr_progress():
    logs = []
    command = [
        sys.executable,
        "-u",
        "-c",
        (
            "import sys, time; "
            "print('Downloading 25%', flush=True); "
            "time.sleep(0.05); "
            "print('Installing package', file=sys.stderr, flush=True); "
            "time.sleep(0.05); "
            "print('Installing package', flush=True); "
            "print('Complete', flush=True)"
        ),
    ]

    result = run_logged_subprocess(command, log_fn=logs.append, timeout=5)

    assert result["returncode"] == 0
    assert result["timed_out"] is False
    assert any("Downloading 25%" in message for message in logs)
    assert any("Installing package" in message for message in logs)
    assert any("Complete" in message for message in logs)
    assert sum("Installing package" in message for message in logs) == 1
    assert "Downloading 25%" in result["output"]


def test_installer_runner_reports_timeout_without_waiting_for_child():
    result = run_logged_subprocess(
        [sys.executable, "-u", "-c", "import time; time.sleep(30)"],
        log_fn=lambda _message: None,
        timeout=0.2,
    )

    assert result["timed_out"] is True
    assert result["returncode"] != 0


def test_ensure_proxy_running_restarts_healthy_stale_runtime(tmp_path, monkeypatch):
    health_checks = iter([
        {"healthy": True},
        {"healthy": False},
        {"healthy": True},
    ])
    killed_ports = []
    runtime_calls = []

    class FakeProcess:
        pid = 4321

        @staticmethod
        def poll():
            return None

    monkeypatch.setattr(antigravity_proxy, "_proxy_process", None)
    monkeypatch.setattr(antigravity_proxy, "_ensure_proxy_config", lambda: str(tmp_path))
    monkeypatch.setattr(antigravity_proxy, "check_proxy_health", lambda: next(health_checks))
    monkeypatch.setattr(
        antigravity_proxy,
        "_cached_runtime_needs_update",
        lambda _data, _running_version=None: True,
    )
    monkeypatch.setattr(
        antigravity_proxy,
        "_kill_proxy_by_port",
        lambda port: killed_ports.append(port),
    )
    monkeypatch.setattr(
        antigravity_proxy,
        "_ensure_proxy_runtime",
        lambda *_args, **kwargs: runtime_calls.append(kwargs) or str(tmp_path),
    )
    monkeypatch.setattr(
        antigravity_proxy,
        "_find_proxy_launch_command",
        lambda _runtime_dir: ["bun", "run", "server.ts"],
    )
    monkeypatch.setattr(antigravity_proxy.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())
    monkeypatch.setattr(antigravity_proxy.time, "sleep", lambda _seconds: None)

    status = antigravity_proxy.ensure_proxy_running(log_fn=lambda _message: None)

    assert status == {"running": True, "auto_launched": True}
    assert killed_ports == [3000]
    assert runtime_calls[0]["force_update"] is True


def test_parallel_proxy_checks_share_one_update_and_restart(tmp_path, monkeypatch):
    state = {"running": True, "update_needed": True}
    killed_ports = []
    runtime_calls = []
    results = []
    start_barrier = threading.Barrier(3)

    class FakeProcess:
        pid = 4321

        @staticmethod
        def poll():
            return None

    def fake_health():
        return {
            "healthy": state["running"],
            "details": {"version": "0.7.6"},
        }

    def fake_kill(port):
        killed_ports.append(port)
        state["running"] = False

    def fake_runtime(*_args, **_kwargs):
        runtime_calls.append(True)
        state["update_needed"] = False
        return str(tmp_path)

    def fake_popen(*_args, **_kwargs):
        state["running"] = True
        return FakeProcess()

    def worker():
        start_barrier.wait()
        results.append(antigravity_proxy.ensure_proxy_running(log_fn=lambda _message: None))

    monkeypatch.setattr(antigravity_proxy, "_proxy_process", None)
    monkeypatch.setattr(antigravity_proxy, "_ensure_proxy_config", lambda: str(tmp_path))
    monkeypatch.setattr(antigravity_proxy, "check_proxy_health", fake_health)
    monkeypatch.setattr(
        antigravity_proxy,
        "_cached_runtime_needs_update",
        lambda *_args, **_kwargs: state["update_needed"],
    )
    monkeypatch.setattr(antigravity_proxy, "_kill_proxy_by_port", fake_kill)
    monkeypatch.setattr(antigravity_proxy, "_ensure_proxy_runtime", fake_runtime)
    monkeypatch.setattr(
        antigravity_proxy,
        "_find_proxy_launch_command",
        lambda _runtime_dir: ["bun", "run", "server.ts"],
    )
    monkeypatch.setattr(antigravity_proxy.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(antigravity_proxy.time, "sleep", lambda _seconds: None)

    workers = [threading.Thread(target=worker) for _ in range(2)]
    for thread in workers:
        thread.start()
    start_barrier.wait()
    for thread in workers:
        thread.join(2)

    assert len(results) == 2
    assert killed_ports == [3000]
    assert runtime_calls == [True]
    assert sum(result["auto_launched"] for result in results) == 1


def test_ensure_proxy_running_installs_bun_when_no_launcher_exists(tmp_path, monkeypatch):
    health_checks = iter([
        {"healthy": False},
        {"healthy": False},
        {"healthy": True},
    ])
    launch_checks = iter([None, ["bun", "run", "server.ts"]])
    install_calls = []

    class FakeProcess:
        pid = 4321

        @staticmethod
        def poll():
            return None

    monkeypatch.setattr(antigravity_proxy, "_proxy_process", None)
    monkeypatch.setattr(antigravity_proxy, "_ensure_proxy_config", lambda: str(tmp_path))
    monkeypatch.setattr(antigravity_proxy, "check_proxy_health", lambda: next(health_checks))
    monkeypatch.setattr(
        antigravity_proxy,
        "_ensure_proxy_runtime",
        lambda *_args, **_kwargs: str(tmp_path),
    )
    monkeypatch.setattr(
        antigravity_proxy,
        "_find_proxy_launch_command",
        lambda _runtime_dir: next(launch_checks),
    )
    monkeypatch.setattr(
        antigravity_proxy,
        "_install_bun_automatically",
        lambda log_fn=None: install_calls.append(log_fn) or {"installed": True},
    )
    monkeypatch.setattr(antigravity_proxy.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())
    monkeypatch.setattr(antigravity_proxy.time, "sleep", lambda _seconds: None)

    status = antigravity_proxy.ensure_proxy_running(log_fn=lambda _message: None)

    assert status == {"running": True, "auto_launched": True}
    assert len(install_calls) == 1


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


def test_watchdog_request_cleanup_preserves_sibling_with_same_chapter_number():
    unified_api_client._api_watchdog_reset()
    try:
        unified_api_client._api_watchdog_started(
            "translation", model="gemini-3.5-flash", request_id="frontmatter-a",
            chapter=0, label="chapter_notice0001.xhtml", queued=True,
        )
        unified_api_client._api_watchdog_started(
            "translation", model="gemini-3.5-flash", request_id="frontmatter-b",
            chapter=0, label="chapter_notice0002.xhtml", queued=True,
        )
        unified_api_client._api_watchdog_mark_in_flight("frontmatter-a", "gemini-3.5-flash")
        unified_api_client._api_watchdog_mark_in_flight("frontmatter-b", "gemini-3.5-flash")

        unified_api_client._api_watchdog_clear_request("frontmatter-a")

        state = unified_api_client.get_api_watchdog_state()
        assert state["in_flight"] == 1
        assert [entry["request_id"] for entry in state["in_flight_entries"]] == ["frontmatter-b"]
    finally:
        unified_api_client._api_watchdog_reset()


def test_watchdog_tracks_lazy_backlog_without_writing_every_dequeue(monkeypatch):
    writes = []
    monkeypatch.setattr(
        unified_api_client,
        "_api_watchdog_external_write",
        lambda state: writes.append(dict(state)),
    )
    unified_api_client._api_watchdog_reset()
    writes.clear()

    try:
        unified_api_client._api_watchdog_set_backlog(1000, publish=True)
        unified_api_client._api_watchdog_set_backlog(999)
        unified_api_client._api_watchdog_set_backlog(998)

        assert unified_api_client.get_api_watchdog_state()["backlog"] == 998
        assert [state["backlog"] for state in writes] == [1000]

        unified_api_client._api_watchdog_set_backlog(0, publish=True)
        assert [state["backlog"] for state in writes] == [1000, 0]
    finally:
        unified_api_client._api_watchdog_reset()


def test_watchdog_graceful_clear_removes_only_pending_requests(monkeypatch):
    writes = []
    monkeypatch.setattr(
        unified_api_client,
        "_api_watchdog_external_write",
        lambda state: writes.append(dict(state)),
    )
    unified_api_client._api_watchdog_reset()
    writes.clear()

    try:
        unified_api_client._api_watchdog_started(
            "translation", request_id="active", chapter=1, queued=True,
        )
        unified_api_client._api_watchdog_mark_in_flight("active", "model-a")
        unified_api_client._api_watchdog_started(
            "translation", request_id="queued", chapter=2, queued=True,
        )
        unified_api_client._api_watchdog_started(
            "translation", request_id="cooldown", chapter=3, queued=True,
        )
        unified_api_client._api_watchdog_mark_waiting(
            "cooldown", "model-b", "delay",
        )
        unified_api_client._api_watchdog_set_backlog(50)
        unified_api_client._api_watchdog_set_scheduler_queue(4)
        writes.clear()

        removed = unified_api_client._api_watchdog_clear_pending_requests()

        state = unified_api_client.get_api_watchdog_state()
        assert removed == 2
        assert state["in_flight"] == 1
        assert state["backlog"] == 0
        assert state["scheduler_queued"] == 0
        assert [
            entry["request_id"] for entry in state["in_flight_entries"]
        ] == ["active"]
        assert len(writes) == 1
    finally:
        unified_api_client._api_watchdog_reset()


def test_shared_instance_cancel_does_not_cancel_sibling_batch_request(monkeypatch):
    client = object.__new__(UnifiedClient)
    client._cancelled = True
    client._stop_callback = lambda: False
    client.context = "translation"

    monkeypatch.setenv("BATCH_TRANSLATION", "1")
    monkeypatch.delenv("TRANSLATION_CANCELLED", raising=False)
    monkeypatch.setattr(unified_api_client, "global_stop_flag", False)
    UnifiedClient.set_global_cancellation(False)

    assert client._is_instance_cancel_requested() is False
    assert client._is_stop_requested() is False


def test_instance_cancel_still_stops_non_batch_request(monkeypatch):
    client = object.__new__(UnifiedClient)
    client._cancelled = True
    client._stop_callback = lambda: False
    client.context = "translation"

    monkeypatch.setenv("BATCH_TRANSLATION", "0")
    monkeypatch.setattr(unified_api_client, "global_stop_flag", False)
    UnifiedClient.set_global_cancellation(False)

    assert client._is_instance_cancel_requested() is True
    assert client._is_stop_requested() is True


def test_nested_retry_keeps_original_run_owner(monkeypatch):
    client = object.__new__(UnifiedClient)
    tls = client._get_thread_local_client()
    monkeypatch.setattr(tls, "current_request_run_id", "old-run")
    monkeypatch.setenv("GLOSSARION_RUN_ID", "new-run")

    assert client._bind_thread_run_id_for_request() == "old-run"
    assert unified_api_client._RUN_ID_CVAR.get("") == "old-run"
    assert client._is_stale_request_run() is True


def test_local_cancellation_reason_distinguishes_hard_stop_from_timeout(monkeypatch):
    client = object.__new__(UnifiedClient)
    tls = client._get_thread_local_client()
    monkeypatch.setattr(tls, "local_cancel_check", lambda: True)
    monkeypatch.setattr(tls, "local_cancel_reason", lambda: "hard stop")

    assert client._cancellation_reason() == "request-local hard stop"
