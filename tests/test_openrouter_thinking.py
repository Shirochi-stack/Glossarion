import os
import sys
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from other_settings import DEEPSEEK_V4_EFFORT_OPTIONS, _apply_gpt_thinking_toggle
import unified_api_client as api_module
from unified_api_client import UnifiedClient


def test_toggle_off_explicitly_disables_openrouter_thinking(monkeypatch):
    """Off must replace any normalized reasoning config with the provider-native toggle."""
    monkeypatch.setenv("ENABLE_GPT_THINKING", "0")
    payload = {"reasoning": {"enabled": True, "effort": "high"}}

    applied = UnifiedClient._apply_openrouter_thinking_disabled(payload)

    assert applied is True
    assert payload == {
        "reasoning": {"effort": "none"},
        "thinking": {"type": "disabled"},
    }


def test_toggle_on_preserves_openrouter_reasoning_payload(monkeypatch):
    """The disabled helper must not alter an enabled OpenRouter request."""
    monkeypatch.setenv("ENABLE_GPT_THINKING", "1")
    payload = {"reasoning": {"enabled": True, "effort": "high"}}

    applied = UnifiedClient._apply_openrouter_thinking_disabled(payload)

    assert applied is False
    assert payload == {"reasoning": {"enabled": True, "effort": "high"}}


def test_other_settings_toggle_off_updates_live_openrouter_state(monkeypatch):
    monkeypatch.setenv("ENABLE_GPT_THINKING", "1")
    monkeypatch.setenv("GPT_REASONING_TOKENS", "4096")
    control_refreshes = []
    gui = SimpleNamespace(
        config={"enable_gpt_thinking": True},
        enable_gpt_thinking_var=True,
        toggle_gpt_reasoning_controls=lambda: control_refreshes.append(True),
    )

    enabled = _apply_gpt_thinking_toggle(gui, False)

    assert enabled is False
    assert gui.enable_gpt_thinking_var is False
    assert gui.config["enable_gpt_thinking"] is False
    assert os.environ["ENABLE_GPT_THINKING"] == "0"
    assert os.environ["GPT_REASONING_TOKENS"] == ""
    assert control_refreshes == [True]


def test_deepseek_v4_effort_options_and_normalization_include_none_and_low():
    assert DEEPSEEK_V4_EFFORT_OPTIONS == ("none", "low", "high", "max")
    assert UnifiedClient._normalize_deepseek_v4_effort("none") == "none"
    assert UnifiedClient._normalize_deepseek_v4_effort("low") == "low"
    assert UnifiedClient._normalize_deepseek_v4_effort("high") == "high"
    assert UnifiedClient._normalize_deepseek_v4_effort("max") == "max"
    assert UnifiedClient._normalize_deepseek_v4_effort("xhigh") == "max"


def test_deepseek_responses_toggle_routes_and_passes_none(monkeypatch, tmp_path):
    captured = {}

    class FakeResponse:
        def model_dump_json(self):
            return '{"output_text":"ok","status":"completed"}'

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResponse()

    class FakeChatCompletions:
        def create(self, **kwargs):
            raise AssertionError("DeepSeek should use the Responses API when enabled")

    class FakeOpenAIClient:
        def __init__(self, **kwargs):
            self.responses = FakeResponses()
            self.chat = SimpleNamespace(completions=FakeChatCompletions())

        def close(self):
            pass

    monkeypatch.setenv("DEEPSEEK_USE_RESPONSES_API", "1")
    monkeypatch.setenv("DEEPSEEK_EFFORT", "none")
    monkeypatch.setenv("ENABLE_STREAMING", "0")
    monkeypatch.setattr(api_module, "openai", SimpleNamespace(OpenAI=FakeOpenAIClient))
    monkeypatch.setattr(api_module, "httpx", None)

    client = UnifiedClient("test-key", "deepseek-v4-flash", str(tmp_path))
    monkeypatch.setattr(client, "_save_response", lambda *args, **kwargs: None)
    monkeypatch.setattr(client, "_should_show_api_lifecycle_logs", lambda: False)

    response = client._send_openai_compatible(
        messages=[
            {"role": "system", "content": "Translate."},
            {"role": "user", "content": "Text"},
        ],
        temperature=0.3,
        max_tokens=500,
        base_url="https://api.deepseek.com/v1",
        response_name="responses-toggle-test",
        provider="deepseek",
    )

    assert response.content == "ok"
    assert captured["model"] == "deepseek-v4-flash"
    assert captured["reasoning"] == {"effort": "none"}
    assert captured["instructions"] == "Translate."
    assert "messages" not in captured
    assert "thinking" not in captured.get("extra_body", {})


def test_deepseek_responses_streaming_collects_semantic_events(monkeypatch, tmp_path, caplog):
    captured = {}

    class FakeStream:
        def __iter__(self):
            return iter([
                SimpleNamespace(type="response.created"),
                SimpleNamespace(type="response.reasoning_text.delta", delta="Checking..."),
                SimpleNamespace(type="response.output_text.delta", delta="Translated "),
                SimpleNamespace(type="response.output_text.delta", delta="text"),
                SimpleNamespace(type="response.completed"),
            ])

        def close(self):
            pass

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeStream()

    class FakeChatCompletions:
        def create(self, **kwargs):
            raise AssertionError("DeepSeek should use Responses streaming")

    class FakeOpenAIClient:
        def __init__(self, **kwargs):
            self.responses = FakeResponses()
            self.chat = SimpleNamespace(completions=FakeChatCompletions())

        def close(self):
            pass

    monkeypatch.setenv("DEEPSEEK_USE_RESPONSES_API", "1")
    monkeypatch.setenv("DEEPSEEK_EFFORT", "low")
    monkeypatch.setenv("ENABLE_STREAMING", "1")
    monkeypatch.setenv("LOG_STREAM_CHUNKS", "1")
    monkeypatch.setenv("STREAM_THINKING_LOGS", "1")
    monkeypatch.setenv("BATCH_TRANSLATION", "0")
    monkeypatch.setattr(api_module, "openai", SimpleNamespace(OpenAI=FakeOpenAIClient))
    monkeypatch.setattr(api_module, "httpx", None)

    client = UnifiedClient("test-key", "deepseek-v4-flash", str(tmp_path))
    monkeypatch.setattr(client, "_save_response", lambda *args, **kwargs: None)
    monkeypatch.setattr(client, "_should_show_api_lifecycle_logs", lambda: False)

    response = client._send_openai_compatible(
        messages=[{"role": "user", "content": "Text"}],
        temperature=0.3,
        max_tokens=500,
        base_url="https://api.deepseek.com/v1",
        response_name="responses-stream-test",
        provider="deepseek",
    )

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert captured["stream"] is True
    assert captured["reasoning"] == {"effort": "low"}
    assert response.content == "Translated text"
    assert response.finish_reason == "stop"
    assert "Thinking..." in logs
    assert "Text streaming..." in logs
    assert "Translated text" in logs
