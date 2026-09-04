import copy
import json
from types import SimpleNamespace

import pytest

import unified_api_client as api
from unified_api_client import UnifiedClient, UnifiedClientError


ERROR = {
    "error": {
        "message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",
        "type": "invalid_request_error", "param": "max_tokens", "code": "unsupported_parameter",
    }
}


class ParameterError(Exception):
    status_code = 400
    body = ERROR


def bare_client(model="gpt-6-astra"):
    client = UnifiedClient.__new__(UnifiedClient)
    client.model = model
    client.client_type = "openai"
    client._get_active_request_model = lambda: model
    client._is_stop_requested = lambda: False
    client._active_per_key_output_token_limit = lambda: None
    client.get_cached_output_token_limit = lambda model: None
    return client


@pytest.mark.parametrize("model", ["gpt-6-astra", "openai/gpt-6-astra", "gpt-6", "gpt6-astra", "gpt-6-astra-2026-09-05"])
def test_gpt6_uses_completion_token_limit(model):
    client = bare_client(model)
    assert client._is_o_series_model()
    assert client._normalize_token_params(128000, None) == (None, 128000)
    body = client._build_openai_params([], 0.5, 128000)
    assert body["max_completion_tokens"] == 128000
    assert "max_tokens" not in body
    assert "temperature" not in body


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-60-test", "not-gpt-6-astra"])
def test_other_models_keep_existing_parameter_behavior(model):
    client = bare_client(model)
    assert not client._is_o_series_model()
    assert client._build_openai_params([], 0.5, 1234) == {
        "model": model, "messages": [], "temperature": 0.5, "max_tokens": 1234,
    }


def test_responses_api_keeps_max_output_tokens():
    body = {"model": "gpt-6-astra", "max_output_tokens": 128000, "temperature": 1,
            "top_p": 0.9, "logprobs": True, "top_logprobs": 2}
    UnifiedClient._apply_gpt6_openai_constraints(body, use_responses_api=True)
    assert body == {"model": "gpt-6-astra", "max_output_tokens": 128000}


@pytest.mark.parametrize("stream", [False, True])
def test_sdk_repairs_explicit_token_error_once(stream):
    calls = []
    result = object()

    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        if len(calls) == 1:
            raise ParameterError(ERROR["error"]["message"])
        return result

    kwargs = {"model": "custom-alias", "messages": [], "max_tokens": 128000,
              "stream": stream, "reasoning_effort": "max"}
    assert bare_client()._create_chat_completion_with_token_retry(create, kwargs, "openai") is result
    assert len(calls) == 2
    assert calls[1] == {**{k: v for k, v in calls[0].items() if k != "max_tokens"},
                        "max_completion_tokens": 128000}


def test_sdk_corrected_failure_is_not_retried_again():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        raise ParameterError("still rejected")

    with pytest.raises(ParameterError):
        bare_client()._create_chat_completion_with_token_retry(create, {"max_tokens": 128000}, "openai")
    assert len(calls) == 2


@pytest.mark.parametrize("status,error", [
    (429, ERROR), (401, ERROR),
    (400, {"error": {"message": "max_tokens exceeds the context window"}}),
    (400, {"error": {"message": "Unsupported parameter: 'temperature'"}}),
])
def test_unrelated_error_does_not_change_token_limit(status, error):
    body = {"max_tokens": 128000}
    assert not UnifiedClient._repair_chat_completion_token_limit(body, status, error)
    assert body == {"max_tokens": 128000}


def test_sdk_cancellation_prevents_corrected_retry():
    client = bare_client()
    client._is_stop_requested = lambda: True
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        raise ParameterError("rejected")

    with pytest.raises(UnifiedClientError, match="cancelled"):
        client._create_chat_completion_with_token_retry(create, {"max_tokens": 32}, "openai")
    assert len(calls) == 1


def test_direct_http_repairs_token_error_even_with_one_attempt(monkeypatch):
    client = bare_client()
    client._bind_thread_run_id_for_request = lambda: None
    client._get_send_interval = lambda: 0
    client._get_thread_directory = lambda: None
    client._ignore_graceful_stop = False
    client.request_timeout = 30
    monkeypatch.delenv("GRACEFUL_STOP", raising=False)
    monkeypatch.setattr(api, "_save_outgoing_request", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_save_incoming_response", lambda *args, **kwargs: None)
    calls = []
    closed = []
    success = SimpleNamespace(status_code=200, headers={}, json=lambda: {"ok": True})

    def request(*args, **kwargs):
        calls.append(copy.deepcopy(kwargs["json"]))
        if len(calls) == 1:
            return SimpleNamespace(status_code=400, headers={}, text=json.dumps(ERROR),
                                   json=lambda: ERROR, close=lambda: closed.append(True))
        return success

    monkeypatch.setattr(api.requests, "request", request)
    result = client._http_request_with_retries(
        "POST", "https://api.openai.com/v1/chat/completions",
        json={"model": "alias", "max_tokens": 128000}, max_retries=1,
    )
    assert result is success
    assert len(calls) == 2
    assert calls[1] == {"model": "alias", "max_completion_tokens": 128000}
    assert closed == [True]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-custom-reasoning-alias"])
def test_active_sdk_path_preflight_and_corrected_retry(monkeypatch, tmp_path, stream, model):
    calls = []

    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        if "max_tokens" in kwargs:
            raise ParameterError(ERROR["error"]["message"])
        if stream:
            return iter([SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content="OK"), finish_reason="stop",
            )])])
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="OK"), finish_reason="stop",
        )])

    fake_sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=lambda: None)
    monkeypatch.setattr(api.openai, "OpenAI", lambda **kwargs: fake_sdk)
    monkeypatch.setattr(api, "httpx", None)
    monkeypatch.setenv("USE_CUSTOM_OPENAI_ENDPOINT", "0")
    monkeypatch.setenv("ENABLE_GPT_THINKING", "0")
    monkeypatch.setenv("PASS_THINKING_TO_OPENAI_COMPATIBLE", "0")
    monkeypatch.delenv("GRACEFUL_STOP", raising=False)
    client = UnifiedClient("test-key", model, str(tmp_path))
    monkeypatch.setattr(client, "_get_max_retries", lambda: 1)
    monkeypatch.setattr(client, "_get_send_interval", lambda: 0)
    monkeypatch.setattr(client, "_streaming_enabled", lambda: stream)
    monkeypatch.setattr(client, "_stream_logging_enabled", lambda enabled: False)
    monkeypatch.setattr(client, "_is_stop_requested", lambda: False)
    monkeypatch.setattr(client, "_save_response", lambda *args, **kwargs: None)
    monkeypatch.setattr(client, "_should_show_api_lifecycle_logs", lambda: False)
    monkeypatch.setattr(client, "_get_anti_duplicate_params", lambda *args, **kwargs: {"top_p": 0.8})
    result = client._send_openai_compatible(
        [{"role": "user", "content": "Reply OK."}], 0.5, 128000,
        "https://api.openai.com/v1", "token-test", provider="openai",
    )
    assert result.content == "OK"
    assert calls[-1]["max_completion_tokens"] == 128000
    assert "max_tokens" not in calls[-1]
    if model == "gpt-6-astra":
        assert len(calls) == 1
        assert "temperature" not in calls[0]
        assert "top_p" not in calls[0]
    else:
        assert len(calls) == 2
