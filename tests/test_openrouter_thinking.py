import os
import sys
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from other_settings import DEEPSEEK_V4_EFFORT_OPTIONS, _apply_gpt_thinking_toggle
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
