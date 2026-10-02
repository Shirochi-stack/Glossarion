import os
from types import SimpleNamespace

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import unified_api_client as api
from multi_api_key_manager import APIKeyEntry
from request_parameters import (
    display_parameter_value, normalize_request_parameters, parse_parameter_value,
)
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
    ('openrouter', 'or/~openai/gpt-latest', 'https://openrouter.ai/api/v1', 'fast', 'priority'),
    ('openrouter', 'or/anthropic/claude-opus-latest', 'https://openrouter.ai/api/v1', 'priority', None),
    ('openrouter', 'or/deepseek/deepseek-v4-pro', 'https://openrouter.ai/api/v1', 'flex', None),
    ('openrouter', 'or/google/gemma-3-27b-it', 'https://openrouter.ai/api/v1', 'standard', None),
    ('openrouter', 'or/x-ai/grok-4', 'https://openrouter.ai/api/v1', 'flex', None),
])
@pytest.mark.parametrize('use_sdk', [True, False])
def test_service_tier_reaches_request_payload(
    monkeypatch, use_sdk, provider, model, base_url, choice, expected
):
    calls = []
    logs = []
    monkeypatch.setenv('GEMINI_SERVICE_TIER', choice)
    monkeypatch.setenv('USE_CUSTOM_OPENAI_ENDPOINT', '0')
    monkeypatch.setenv('ENABLE_STREAMING', '0')
    monkeypatch.setenv('OPENROUTER_PREFERRED_PROVIDER', 'Auto')
    budget_mode = (
        provider == 'openrouter' and model == 'or/google/gemini-2.5-flash' and choice == 'flex'
    )
    if provider == 'openrouter':
        monkeypatch.setenv('ENABLE_GPT_THINKING', '1')
        monkeypatch.setenv('GPT_REASONING_TOKENS', '2000')
        monkeypatch.setenv('GPT_EFFORT', 'high')
        monkeypatch.setenv('OPENROUTER_USE_REASONING_TOKENS', '1' if budget_mode else '0')
        monkeypatch.setattr(api, 'print', lambda *args, **_kwargs: logs.append(' '.join(map(str, args))))
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
    client = UnifiedClient('test-key', model, os.getcwd())
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
    if provider == 'openrouter':
        log = '\n'.join(logs)
        request_logs = [line for line in logs if '🧠 [openrouter] Thinking' in line]
        assert request_logs
        reasoning = calls[0].get('reasoning', calls[0].get('extra_body', {}).get('reasoning'))
        if budget_mode:
            assert reasoning['max_tokens'] == 2000
            assert 'effort' not in reasoning
            assert f'Thinking enabled for {request_model}: max_tokens=2,000' in log
        else:
            assert reasoning['effort'] == 'high'
            assert 'max_tokens' not in reasoning
            assert f'Thinking enabled for {request_model}: effort=high' in log
        if expected is None:
            assert all('service_tier=' not in line for line in request_logs)
        else:
            assert any(f'service_tier={expected}' in line for line in request_logs)


def test_openrouter_sdk_parse_fallback_preserves_tier_and_logs_http_request(monkeypatch):
    calls = []
    logs = []
    model = 'openai/gpt-6-luna'
    monkeypatch.setenv('GEMINI_SERVICE_TIER', 'flex')
    monkeypatch.setenv('ENABLE_GPT_THINKING', '1')
    monkeypatch.setenv('GPT_REASONING_TOKENS', '2000')
    monkeypatch.setenv('GPT_EFFORT', 'high')
    monkeypatch.setenv('OPENROUTER_USE_REASONING_TOKENS', '0')
    monkeypatch.setenv('ENABLE_STREAMING', '0')
    monkeypatch.setattr(api, 'print', lambda *args, **_kwargs: logs.append(' '.join(map(str, args))))

    def sdk_create(**_kwargs):
        raise ValueError('Expecting value')

    fake_sdk = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=sdk_create)),
        close=lambda: None,
    )
    monkeypatch.setattr(api.openai, 'OpenAI', lambda **_kwargs: fake_sdk)
    monkeypatch.setattr(api, 'httpx', None)

    client = UnifiedClient('test-key', f'or/{model}', os.getcwd())
    monkeypatch.setattr(client, '_get_max_retries', lambda: 1)
    monkeypatch.setattr(client, '_get_send_interval', lambda: 0)
    monkeypatch.setattr(client, '_streaming_enabled', lambda: False)
    monkeypatch.setattr(client, '_is_stop_requested', lambda: False)
    monkeypatch.setattr(client, '_save_response', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(client, '_http_request_with_retries', lambda **kwargs: (
        calls.append(kwargs['json']) or SimpleNamespace(
            headers={'content-type': 'application/json'},
            json=lambda: {'choices': [{'message': {'content': 'OK'}, 'finish_reason': 'stop'}]},
        )
    ))

    result = client._send_openai_compatible(
        [{'role': 'user', 'content': 'Reply OK.'}], 0.5, 100,
        'https://openrouter.ai/api/v1', 'fallback-tier-test',
        provider='openrouter', model_override=model,
    )

    assert result.content == 'OK'
    assert calls[0]['service_tier'] == 'flex'
    assert calls[0]['reasoning']['effort'] == 'high'
    assert 'max_tokens' not in calls[0]['reasoning']
    assert sum('Thinking enabled for openai/gpt-6-luna: effort=high, service_tier=flex' in line
               for line in logs) == 2


@pytest.mark.parametrize('use_sdk', [True, False])
@pytest.mark.parametrize('tier', ['off', 'flex'])
def test_nanogpt_logs_sent_effort_and_tier_without_budget_tokens(
    monkeypatch, use_sdk, tier,
):
    calls = []
    logs = []
    model = 'openai/gpt-6-luna'
    monkeypatch.setattr(api, 'print', lambda *args, **_kwargs: logs.append(' '.join(map(str, args))))
    monkeypatch.setenv('ENABLE_GPT_THINKING', '1')
    monkeypatch.setenv('GPT_REASONING_TOKENS', '2000')
    monkeypatch.setenv('GPT_EFFORT', 'medium')
    monkeypatch.setenv('GEMINI_SERVICE_TIER', tier)
    monkeypatch.setenv('ENABLE_STREAMING', '0')
    monkeypatch.setenv('USE_CUSTOM_OPENAI_ENDPOINT', '0')
    if tier == 'flex':
        monkeypatch.setattr(api.requests, 'get', lambda *_args, **_kwargs: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {'data': [{'id': model, 'supported_service_tiers': ['flex']}]},
        ))
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
        monkeypatch.setattr(api.openai, 'OpenAI', lambda **_kwargs: fake_sdk)
        monkeypatch.setattr(api, 'httpx', None)

    client = UnifiedClient('test-key', f'nan/{model}', os.getcwd())
    monkeypatch.setattr(client, '_get_max_retries', lambda: 1)
    monkeypatch.setattr(client, '_get_send_interval', lambda: 0)
    monkeypatch.setattr(client, '_streaming_enabled', lambda: False)
    monkeypatch.setattr(client, '_is_stop_requested', lambda: False)
    monkeypatch.setattr(client, '_save_response', lambda *_args, **_kwargs: None)
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
        'https://nano-gpt.com/api/v1', 'nano-options-test',
        provider='nanogpt', model_override=model,
    )

    assert result.content == 'OK'
    payload = calls[0]
    reasoning = payload.get('extra_body', payload) if use_sdk else payload
    assert reasoning['reasoning_effort'] == 'medium'
    assert 'thinking' not in reasoning
    assert 'budget_tokens' not in str(payload)
    assert payload.get('service_tier') == (None if tier == 'off' else 'flex')
    log = '\n'.join(logs)
    assert f'Thinking enabled for {model}: effort=medium' in log
    assert 'budget_tokens' not in log
    assert ('service_tier=flex' in log) == (tier == 'flex')


def test_custom_request_parameter_values_and_key_serialization():
    parameters = {
        'service_tier': 'flex', 'top_p': 0.7,
        'metadata': {'source': 'test'}, 'model': 'blocked',
    }
    key = APIKeyEntry('key', 'nan/openai/gpt-5.5', request_parameters=parameters)
    assert key.request_parameters == {
        'service_tier': 'flex', 'top_p': 0.7, 'metadata': {'source': 'test'},
    }
    assert APIKeyEntry.from_dict(key.to_dict()).request_parameters == key.request_parameters
    assert parse_parameter_value(display_parameter_value('123')) == '123'
    assert parse_parameter_value(display_parameter_value({'enabled': True})) == {'enabled': True}
    assert normalize_request_parameters({'messages': [], 'x': float('nan')}) == {}


def test_custom_request_parameter_double_click_suggestion():
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QLabel
    from individual_endpoint_dialog import RequestParametersEditor

    app = QApplication.instance() or QApplication([])
    editor = RequestParametersEditor()
    suggestion = next(label for label in editor.findChildren(QLabel)
                      if 'double-click to add' in label.text())
    QTest.mouseDClick(suggestion, Qt.LeftButton)
    editor.add_row('unused', '')
    editor.add_row('', '')
    assert editor.parameters() == {'service_tier': 'flex'}
    editor.set_parameter('explicit_empty', '""')
    assert editor.parameters()['explicit_empty'] == ''
    editor.close()
    assert app is not None


def test_custom_request_parameters_save_without_endpoint():
    from PySide6.QtWidgets import QApplication
    from individual_endpoint_dialog import IndividualEndpointDialog

    app = QApplication.instance() or QApplication([])
    key = APIKeyEntry('key', 'nan/openai/gpt-5.5')
    config = {'multi_api_keys': [key.to_dict()]}
    gui = SimpleNamespace(config=config, save_config=lambda **kwargs: None)
    dialog = IndividualEndpointDialog(None, gui, key, lambda: None, lambda message: None)
    dialog.request_parameters_editor.set_parameter('service_tier', 'flex')
    dialog._on_save()
    assert key.request_parameters == {'service_tier': 'flex'}
    assert config['multi_api_keys'][0]['request_parameters'] == {'service_tier': 'flex'}
    assert not key.use_individual_endpoint
    dialog.close()
    assert app is not None


@pytest.mark.parametrize('use_sdk', [True, False])
def test_custom_request_parameters_reach_selected_nanogpt_key(monkeypatch, use_sdk):
    calls = []
    monkeypatch.setenv('GEMINI_SERVICE_TIER', 'standard')
    monkeypatch.setenv('USE_CUSTOM_OPENAI_ENDPOINT', '0')
    monkeypatch.setenv('ENABLE_STREAMING', '0')
    monkeypatch.setenv('SAVE_PAYLOAD', '0')
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

    client = UnifiedClient('test-key', 'nan/openai/gpt-5.5', '.')
    key = APIKeyEntry('test-key', 'nan/openai/gpt-5.5', request_parameters={
        'service_tier': 'flex', 'top_p': 0.7,
    })
    client._apply_key_runtime_overrides(key_entry=key)
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
        'https://nano-gpt.com/api/v1', 'parameter-test',
        provider='nanogpt', model_override='openai/gpt-5.5',
    )
    assert result.content == 'OK'
    assert calls[0]['service_tier'] == 'flex'
    assert (calls[0]['extra_body']['top_p'] if use_sdk else calls[0]['top_p']) == 0.7

    client._apply_key_runtime_overrides(key_entry=APIKeyEntry('other', 'nan/openai/gpt-5.5'))
    assert client._active_request_parameters() == {}
