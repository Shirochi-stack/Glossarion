from unified_api_client import UnifiedClient


def test_toggle_off_explicitly_disables_openrouter_thinking(monkeypatch):
    """Off must replace any normalized reasoning config with the provider-native toggle."""
    monkeypatch.setenv("ENABLE_GPT_THINKING", "0")
    payload = {"reasoning": {"enabled": True, "effort": "high"}}

    applied = UnifiedClient._apply_openrouter_thinking_disabled(payload)

    assert applied is True
    assert payload == {"thinking": {"type": "disabled"}}


def test_toggle_on_preserves_openrouter_reasoning_payload(monkeypatch):
    """The disabled helper must not alter an enabled OpenRouter request."""
    monkeypatch.setenv("ENABLE_GPT_THINKING", "1")
    payload = {"reasoning": {"enabled": True, "effort": "high"}}

    applied = UnifiedClient._apply_openrouter_thinking_disabled(payload)

    assert applied is False
    assert payload == {"reasoning": {"enabled": True, "effort": "high"}}
