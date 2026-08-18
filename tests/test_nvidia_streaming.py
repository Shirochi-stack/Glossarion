import json
import logging
from types import SimpleNamespace

import unified_api_client as api_module
from unified_api_client import UnifiedClient


def _sse_event(payload):
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


def test_nvidia_stream_uses_raw_httpx_and_emits_reasoning(monkeypatch, tmp_path, caplog):
    captured = {}
    caplog.set_level(logging.INFO, logger=api_module.__name__)

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        def __init__(self):
            self.closed = False

        def iter_raw(self):
            yield _sse_event({
                "choices": [{
                    "delta": {"reasoning_content": "Checking the translation.\n"},
                    "finish_reason": None,
                }]
            })
            yield _sse_event({
                "choices": [{
                    "delta": {"reasoning_content": "Resolving terminology.\n"},
                    "finish_reason": None,
                }]
            })
            yield _sse_event({
                "choices": [{
                    "delta": {"content": "<p>Translated text.</p>"},
                    "finish_reason": None,
                }]
            })
            yield _sse_event({
                "choices": [{"delta": {}, "finish_reason": "stop"}]
            })
            yield b"data: [DONE]\n\n"

        def close(self):
            self.closed = True

    class FakeHttpxClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False

        def build_request(self, method, url, headers=None, json=None):
            captured.update({
                "method": method,
                "url": url,
                "headers": headers,
                "payload": json,
            })
            return SimpleNamespace(method=method, url=url)

        def send(self, request, stream=False):
            captured["stream"] = stream
            return FakeResponse()

        def close(self):
            self.closed = True

    class FakeSdkCompletions:
        def create(self, **kwargs):
            raise AssertionError("NVIDIA streaming must bypass the OpenAI SDK")

    class FakeOpenAIClient:
        def __init__(self, **kwargs):
            self._client = kwargs.get("http_client")
            self.chat = SimpleNamespace(completions=FakeSdkCompletions())
            self.responses = SimpleNamespace(create=FakeSdkCompletions().create)

        def close(self):
            pass

    fake_httpx = SimpleNamespace(
        Client=FakeHttpxClient,
        Timeout=lambda **kwargs: kwargs,
        Limits=lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(api_module, "httpx", fake_httpx)
    monkeypatch.setattr(api_module, "openai", SimpleNamespace(OpenAI=FakeOpenAIClient))
    monkeypatch.setenv("ENABLE_STREAMING", "1")
    monkeypatch.setenv("LOG_STREAM_CHUNKS", "1")
    monkeypatch.setenv("STREAM_THINKING_LOGS", "1")
    monkeypatch.setenv("BATCH_TRANSLATION", "0")
    monkeypatch.setenv("ENABLE_GPT_THINKING", "1")
    monkeypatch.setenv("PASS_THINKING_TO_OPENAI_COMPATIBLE", "0")
    monkeypatch.setenv("GPT_EFFORT", "high")

    client = UnifiedClient(
        "test-key",
        "nd/deepseek-ai/deepseek-v4-flash-0731",
        str(tmp_path),
    )
    monkeypatch.setattr(client, "_save_response", lambda *args, **kwargs: None)
    monkeypatch.setattr(client, "_should_show_api_lifecycle_logs", lambda: False)

    response = client._send_openai_compatible(
        messages=[{"role": "user", "content": "Translate this."}],
        temperature=0.3,
        max_tokens=1024,
        base_url="https://integrate.api.nvidia.com/v1",
        response_name="nvidia-stream-test",
        provider="nvidia",
    )

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert response.content == "<p>Translated text.</p>"
    assert response.finish_reason == "stop"
    assert captured["method"] == "POST"
    assert captured["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert captured["stream"] is True
    assert captured["payload"]["stream"] is True
    assert "reasoning_effort" not in captured["payload"]
    assert captured["payload"]["chat_template_kwargs"] == {
        "thinking": True,
        "reasoning_effort": "high",
    }
    assert captured["headers"]["Accept"] == "text/event-stream"
    assert captured["headers"]["Accept-Encoding"] == "identity"
    assert "🧠 [nvidia] Thinking..." in logs
    assert "Checking the translation." in logs
    assert "Resolving terminology." in logs
    assert "📡 [nvidia] Text streaming..." in logs


def test_nvidia_deepseek_v4_thinking_uses_model_specific_effort_values(monkeypatch):
    monkeypatch.setenv("ENABLE_GPT_THINKING", "1")
    monkeypatch.setenv("GPT_EFFORT", "xhigh")
    payload = {}

    applied = UnifiedClient._apply_nvidia_deepseek_v4_thinking(
        payload,
        "deepseek-ai/deepseek-v4-flash-0731",
    )

    assert applied is True
    assert payload == {
        "chat_template_kwargs": {
            "thinking": True,
            "reasoning_effort": "max",
        }
    }


def test_nvidia_deepseek_v4_thinking_can_be_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_GPT_THINKING", "0")
    monkeypatch.setenv("GPT_EFFORT", "high")
    payload = {}

    applied = UnifiedClient._apply_nvidia_deepseek_v4_thinking(
        payload,
        "deepseek-ai/deepseek-v4-pro",
    )

    assert applied is True
    assert payload == {
        "chat_template_kwargs": {
            "thinking": False,
            "reasoning_effort": "none",
        }
    }


def test_nvidia_deepseek_v4_status_label_matches_payload(monkeypatch):
    monkeypatch.setenv("ENABLE_GPT_THINKING", "1")
    monkeypatch.setenv("GPT_EFFORT", "xhigh")
    client = object.__new__(UnifiedClient)
    client.model = "nd/deepseek-ai/deepseek-v4-flash-0731"

    assert client._get_thinking_status_label() == " (reasoning_effort: max)"
