import json
import sys
import types

import pytest

import authgpt_auth


@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-5.2", "gpt-5.6-sol"])
@pytest.mark.parametrize("transport", ["httpx", "requests"])
@pytest.mark.parametrize("max_tokens", [128000, None])
def test_authgpt_output_limit_and_note(monkeypatch, model, transport, max_tokens):
    captured = []
    logs = []
    result = {"content": "Translated text", "finish_reason": "stop"}

    def capture_request(url, body, *args, **kwargs):
        captured.append((url, body))
        return result

    monkeypatch.setitem(
        sys.modules, "httpx", types.ModuleType("httpx") if transport == "httpx" else None
    )
    monkeypatch.setattr(
        authgpt_auth, "_stream_with_httpx",
        lambda module, *args, **kwargs: capture_request(*args, **kwargs),
    )
    monkeypatch.setattr(authgpt_auth, "_stream_with_requests", capture_request)

    response = authgpt_auth.send_chat_completion(
        access_token="test-token",
        messages=[
            {"role": "system", "content": "Translate the text."},
            {"role": "user", "content": "Hello"},
        ],
        model=model,
        max_tokens=max_tokens,
        reasoning={"effort": "high"},
        base_url="https://example.test/backend-api",
        log_fn=logs.append,
    )

    assert response == result
    assert len(captured) == 1
    url, body = captured[0]
    assert url == "https://example.test/backend-api/codex/responses"
    assert not {"max_tokens", "max_completion_tokens", "temperature"} & body.keys()
    if max_tokens is None or model == "gpt-6-astra":
        assert "max_output_tokens" not in body
    else:
        assert body["max_output_tokens"] == max_tokens
    assert body["model"] == model
    assert body["reasoning"] == {"effort": "high"}
    assert body["instructions"] == "Translate the text."
    assert body["input"][0]["content"] == [{"type": "input_text", "text": "Hello"}]
    assert body["stream"] is True
    assert body["store"] is False
    if max_tokens is not None:
        if model == "gpt-6-astra":
            assert any(line.startswith("📝") and "does not apply" in line for line in logs)
            assert not any(line.startswith("📏") for line in logs)
        else:
            assert f"📏 AuthGPT max_output_tokens={max_tokens}" in logs
            assert not any("backend" in line for line in logs)


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_authgpt_preserves_selected_reasoning_effort(monkeypatch, effort):
    from unified_api_client import UnifiedClient

    monkeypatch.setenv("ENABLE_GPT_THINKING", "1")
    monkeypatch.setenv("GPT_EFFORT", effort)
    reasoning = UnifiedClient._get_authgpt_reasoning_param(None)
    body = authgpt_auth._build_responses_body(
        [{"role": "user", "content": "Hello"}],
        model="gpt-6-astra", reasoning=reasoning,
    )
    assert body["reasoning"]["effort"] == effort


@pytest.fixture(params=["httpx", "requests"])
def fake_transport(request, monkeypatch):
    responses = []
    bodies = []

    class Response:
        def __init__(self, status, payload):
            self.status_code = status
            self.payload = payload
            self.text = json.dumps(payload)
            self.reason = self.reason_phrase = "Bad Request" if status >= 400 else "OK"
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def close(self):
            self.closed = True

        def read(self):
            return self.text.encode()

        def json(self):
            return self.payload

        def iter_lines(self, **kwargs):
            line = "data: " + self.text
            yield line if request.param == "httpx" else line.encode()

    def send(*args, **kwargs):
        bodies.append(dict(kwargs["json"]))
        return responses[len(bodies) - 1]

    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(
        Timeout=lambda *args, **kwargs: None, stream=send,
    ) if request.param == "httpx" else None)
    monkeypatch.setattr(authgpt_auth.requests, "post", send)
    monkeypatch.setattr(authgpt_auth, "is_cancelled", lambda: False)
    monkeypatch.setenv("LOG_STREAM_CHUNKS", "0")
    monkeypatch.setenv("BATCH_TRANSLATION", "0")
    return responses, bodies, Response


@pytest.mark.parametrize("error", [
    {"detail": "Unsupported parameter: max_output_tokens"},
    {"error": {"message": "Unsupported parameter: 'max_output_tokens'."}},
])
def test_rejected_limit_retries_once_and_preserves_request(fake_transport, error):
    responses, bodies, Response = fake_transport
    responses.extend([
        Response(400, error),
        Response(200, {"type": "response.completed", "response": {
            "status": "completed", "id": "test-response", "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "OK"}]}
            ],
        }}),
    ])
    logs = []
    result = authgpt_auth.send_chat_completion(
        "test-token", [{"role": "user", "content": "Reply OK."}],
        model="gpt-5.6-sol", max_tokens=32, reasoning={"effort": "max"},
        log_fn=logs.append,
    )
    assert result["content"] == "OK"
    assert len(bodies) == 2
    assert bodies[0]["max_output_tokens"] == 32
    assert bodies[1] == {k: v for k, v in bodies[0].items() if k != "max_output_tokens"}
    assert responses[0].closed
    assert any("cannot be enforced" in line for line in logs)
    assert not any("❌" in line for line in logs)


@pytest.mark.parametrize("status,message", [
    (400, "Unsupported parameter: temperature"),
    (400, "max_output_tokens exceeds the maximum allowed value"),
    (401, "Unsupported parameter: max_output_tokens"),
    (429, "Unsupported parameter: max_output_tokens"),
])
def test_other_errors_do_not_drop_limit(fake_transport, status, message):
    responses, bodies, Response = fake_transport
    responses.append(Response(status, {"detail": message}))
    with pytest.raises(RuntimeError):
        authgpt_auth.send_chat_completion(
            "test-token", [], model="gpt-5.6-sol", max_tokens=32, log_fn=lambda _: None,
        )
    assert len(bodies) == 1


def test_retry_failure_surfaces_without_retry_loop(fake_transport):
    responses, bodies, Response = fake_transport
    responses.extend([
        Response(400, {"detail": "Unsupported parameter: max_output_tokens"}),
        Response(400, {"detail": "Unsupported parameter: max_output_tokens"}),
    ])
    with pytest.raises(RuntimeError, match="400"):
        authgpt_auth.send_chat_completion(
            "test-token", [], model="gpt-5.6-sol", max_tokens=32, log_fn=lambda _: None,
        )
    assert len(bodies) == 2


def test_cancel_prevents_compatibility_retry(fake_transport, monkeypatch):
    responses, bodies, Response = fake_transport
    responses.append(Response(400, {"detail": "Unsupported parameter: max_output_tokens"}))
    monkeypatch.setattr(authgpt_auth, "is_cancelled", lambda: True)
    with pytest.raises(RuntimeError, match="cancelled"):
        authgpt_auth.send_chat_completion(
            "test-token", [], model="gpt-5.6-sol", max_tokens=32, log_fn=lambda _: None,
        )
    assert len(bodies) == 1


def test_astra_succeeds_on_first_request_with_note(fake_transport):
    responses, bodies, Response = fake_transport
    responses.append(Response(200, {"type": "response.completed", "response": {
        "status": "completed", "id": "test-response", "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "OK"}]}
        ],
    }}))
    logs = []
    result = authgpt_auth.send_chat_completion(
        "test-token", [{"role": "user", "content": "Reply OK."}],
        model="gpt-6-astra", max_tokens=128000, reasoning={"effort": "max"},
        log_fn=logs.append,
    )
    assert result["content"] == "OK"
    assert len(bodies) == 1
    assert "max_output_tokens" not in bodies[0]
    assert bodies[0]["reasoning"] == {"effort": "max"}
    assert any(line.startswith("📝") and "128000" in line for line in logs)
    assert not any("retrying" in line or "❌" in line for line in logs)
