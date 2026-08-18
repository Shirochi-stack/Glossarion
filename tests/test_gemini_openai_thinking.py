import pytest

from unified_api_client import UnifiedClient


@pytest.mark.parametrize("model", [
    "gemini-3-flash-preview",
    "models/gemini-3.1-pro-preview",
    "gemini-3.5-flash",
])
def test_gemini_openai_uses_thinking_level_for_gemini_3(monkeypatch, model):
    monkeypatch.setenv("ENABLE_GEMINI_THINKING", "1")
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "medium")
    monkeypatch.setenv("THINKING_BUDGET", "8192")

    config = UnifiedClient._build_gemini_openai_thinking_config(model)

    assert config == {
        "thinking_level": "MEDIUM",
        "include_thoughts": True,
    }
    assert "thinking_budget" not in config


@pytest.mark.parametrize("model", [
    "gemini-2.5-pro",
    "models/gemini-2.5-flash",
    "gemini-2.0-flash-thinking-exp",
])
def test_gemini_openai_uses_thinking_budget_below_gemini_3(monkeypatch, model):
    monkeypatch.setenv("ENABLE_GEMINI_THINKING", "1")
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "high")
    monkeypatch.setenv("THINKING_BUDGET", "8192")

    config = UnifiedClient._build_gemini_openai_thinking_config(model)

    assert config == {
        "thinking_budget": 8192,
        "include_thoughts": True,
    }
    assert "thinking_level" not in config


def test_gemini_openai_keeps_minimum_budget_for_gemini_2_5_pro(monkeypatch):
    monkeypatch.setenv("ENABLE_GEMINI_THINKING", "1")
    monkeypatch.setenv("THINKING_BUDGET", "0")

    config = UnifiedClient._build_gemini_openai_thinking_config("gemini-2.5-pro")

    assert config == {
        "thinking_budget": 128,
        "include_thoughts": True,
    }


def test_gemini_openai_omits_thinking_config_when_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_GEMINI_THINKING", "0")

    assert UnifiedClient._build_gemini_openai_thinking_config("gemini-2.5-pro") is None
