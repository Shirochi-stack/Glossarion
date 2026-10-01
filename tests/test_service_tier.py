from types import SimpleNamespace

import pytest

import unified_api_client as api
from unified_api_client import UnifiedClient, UnifiedClientError


@pytest.mark.parametrize('choice,openai,gemini,nanogpt,openrouter', [
    ('off', None, None, None, None),
    ('standard', 'default', 'standard', 'default', 'default'),
    ('flex', 'flex', 'flex', 'flex', 'flex'),
    ('fast', 'priority', 'priority', 'fast', 'priority'),
    ('priority', 'priority', 'priority', 'priority', 'priority'),
])
def test_service_tier_provider_mapping(monkeypatch, choice, openai, gemini, nanogpt, openrouter):
    monkeypatch.setenv('GEMINI_SERVICE_TIER', choice)
    assert UnifiedClient._selected_service_tier('openai') == openai
    assert UnifiedClient._selected_service_tier('gemini-native') == gemini
    assert UnifiedClient._selected_service_tier('nanogpt') == nanogpt
    assert UnifiedClient._selected_service_tier('openrouter') == openrouter


def test_removed_gemini_flex_flag_no_longer_pins_openrouter(monkeypatch):
    monkeypatch.setenv('OPENROUTER_GEMINI_FLEX', '1')
    monkeypatch.setenv('OPENROUTER_PREFERRED_PROVIDER', 'Auto')
    assert UnifiedClient._openrouter_provider_routing(
        UnifiedClient.__new__(UnifiedClient), 'google/gemini-2.5-flash'
    ) is None


def test_or_prefix_uses_openrouter_transport(monkeypatch, tmp_path):
    client = UnifiedClient('test-key', 'or/google/gemini-2.5-flash', str(tmp_path))
    monkeypatch.setattr(client, '_apply_individual_key_endpoint_if_needed', lambda: None)
    monkeypatch.setattr(client, '_send_openai_compatible', lambda **kwargs: kwargs)
    request = client._send_openai_provider_router(
        [{'role': 'user', 'content': 'Hello'}], 0.5, 100, response_name='route-test'
    )
    assert request['provider'] == 'openrouter'
    assert request['base_url'] == 'https://openrouter.ai/api/v1'


def test_nanogpt_tier_preflight_uses_detailed_catalog_and_provider_alias(monkeypatch):
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {'data': [{'id': 'openai/gpt-5.5',
                                    'supported_service_tiers': ['default', 'flex', 'priority']}]},
        )

    monkeypatch.setattr(api.requests, 'get', get)
    tier = UnifiedClient._validate_nanogpt_service_tier(
        'https://nano-gpt.com/api/v1', 'test-key', 'openai/gpt-5.5', 'fast'
    )
    assert tier == 'priority'
    assert calls == [('https://nano-gpt.com/api/v1/models', {
        'params': {'detailed': 'true'},
        'headers': {'Authorization': 'Bearer test-key'},
        'timeout': 10,
    })]


def test_nanogpt_tier_preflight_rejects_unsupported_tier(monkeypatch):
    monkeypatch.setattr(api.requests, 'get', lambda *args, **kwargs: SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {'data': [{'id': 'openai/gpt-5.5', 'supported_service_tiers': []}]},
    ))
    with pytest.raises(UnifiedClientError, match='does not support the flex service tier'):
        UnifiedClient._validate_nanogpt_service_tier(
            'https://nano-gpt.com/api/v1', 'test-key', 'openai/gpt-5.5', 'flex'
        )


def test_nanogpt_default_tier_accepts_default_only_model(monkeypatch):
    monkeypatch.setattr(api.requests, 'get', lambda *args, **kwargs: SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {'data': [{'id': 'openai/gpt-5.5', 'supported_service_tiers': []}]},
    ))
    assert UnifiedClient._validate_nanogpt_service_tier(
        'https://nano-gpt.com/api/v1', 'test-key', 'openai/gpt-5.5', 'default'
    ) == 'default'


@pytest.mark.parametrize('provider,model,base_url,choice,expected', [
    ('openai', 'gpt-4o', 'https://api.openai.com/v1', 'off', None),
    ('openai', 'gpt-4o', 'https://api.openai.com/v1', 'standard', 'default'),
    ('openai', 'gpt-4o', 'https://api.openai.com/v1', 'fast', 'priority'),
    ('nanogpt', 'nan/openai/gpt-5.5', 'https://nano-gpt.com/api/v1', 'off', None),
    ('nanogpt', 'nan/openai/gpt-5.5', 'https://nano-gpt.com/api/v1', 'standard', 'default'),
    ('nanogpt', 'nan/openai/gpt-5.5', 'https://nano-gpt.com/api/v1', 'flex', 'flex'),
    ('nanogpt', 'nan/google/gemini-flash-latest', 'https://nano-gpt.com/api/v1', 'flex', 'flex'),
    ('nanogpt', 'nan/anthropic/claude-opus-latest', 'https://nano-gpt.com/api/v1', 'flex', None),
    ('nanogpt', 'nan/deepseek/deepseek-v4-pro', 'https://nano-gpt.com/api/v1', 'standard', None),
    ('nanogpt', 'nan/qwen-3.6-plus', 'https://nano-gpt.com/api/v1', 'priority', None),
    ('openrouter', 'or/google/gemini-2.5-flash', 'https://openrouter.ai/api/v1', 'off', None),
    ('openrouter', 'or/google/gemini-2.5-flash', 'https://openrouter.ai/api/v1', 'standard', 'default'),
    ('openrouter', 'or/google/gemini-2.5-flash', 'https://openrouter.ai/api/v1', 'flex', 'flex'),
    ('openrouter', 'or/openai/gpt-5.5', 'https://openrouter.ai/api/v1', 'flex', 'flex'),
])
@pytest.mark.parametrize('use_sdk', [True, False])
def test_service_tier_reaches_request_payload(
    monkeypatch, tmp_path, use_sdk, provider, model, base_url, choice, expected
):
    calls = []
    monkeypatch.setenv('GEMINI_SERVICE_TIER', choice)
    monkeypatch.setenv('USE_CUSTOM_OPENAI_ENDPOINT', '0')
    monkeypatch.setenv('ENABLE_STREAMING', '0')
    monkeypatch.setenv('OPENROUTER_PREFERRED_PROVIDER', 'Auto')
    request_model = model.removeprefix('nan/').removeprefix('or/')
    if provider == 'nanogpt':
        def catalog_get(*args, **kwargs):
            if expected is None:
                raise AssertionError('models without a tier must not query the tier catalog')
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {'data': [{'id': request_model, 'supported_service_tiers': ['flex']}]},
            )
        monkeypatch.setattr(api.requests, 'get', catalog_get)
    if use_sdk:
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='OK'), finish_reason='stop')],
                usage=None,
            )
        fake_sdk = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            close=lambda: None,
        )
        monkeypatch.setattr(api.openai, 'OpenAI', lambda **kwargs: fake_sdk)
        monkeypatch.setattr(api, 'httpx', None)
    client = UnifiedClient('test-key', model, str(tmp_path))
    monkeypatch.setattr(client, '_get_max_retries', lambda: 1)
    monkeypatch.setattr(client, '_get_send_interval', lambda: 0)
    monkeypatch.setattr(client, '_streaming_enabled', lambda: False)
    monkeypatch.setattr(client, '_is_stop_requested', lambda: False)
    monkeypatch.setattr(client, '_save_response', lambda *args, **kwargs: None)
    monkeypatch.setattr(client, '_should_show_api_lifecycle_logs', lambda: False)
    if not use_sdk:
        monkeypatch.setattr(api, 'openai', None)
        monkeypatch.setattr(client, '_http_request_with_retries', lambda **kwargs: (
            calls.append(kwargs['json']) or SimpleNamespace(
                headers={'content-type': 'application/json'},
                json=lambda: {'choices': [{'message': {'content': 'OK'}, 'finish_reason': 'stop'}]},
            )
        ))
    result = client._send_openai_compatible(
        [{'role': 'user', 'content': 'Reply OK.'}], 0.5, 100,
        base_url, 'tier-test', provider=provider, model_override=request_model,
    )
    assert result.content == 'OK'
    if expected is None:
        assert 'service_tier' not in calls[0]
    else:
        assert calls[0]['service_tier'] == expected
