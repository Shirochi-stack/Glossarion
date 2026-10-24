"""Refinement key pool: switching it off and on again within one session.

After a request routed through the Refinement pool, the client and the thread kept
the pool key's identity (``GlossaryRefinementKey#1 (...)``). Later requests that went
out with the main key, including refinement requests while the pool was off, were
still identified (and logged) as the Refinement key.
"""
import json

import pytest

from unified_api_client import UnifiedClient

REFINEMENT_KEYS = [{'api_key': 'sk-refine', 'model': 'gpt-4o-mini', 'enabled': True}]
MESSAGES = [{'role': 'user', 'content': 'hello'}]


@pytest.fixture(autouse=True)
def clean_pools(monkeypatch):
    for name in ('USE_MULTI_API_KEYS', 'USE_GLOSSARY_KEYS', 'USE_GLOSSARY_REFINEMENT_KEYS',
                 'USE_VISION_KEYS', 'USE_QA_SCAN_KEYS', 'USE_INPAINTER_KEYS',
                 'USE_ROLLING_SUMMARY_KEYS', 'USE_TRUNCATION_RETRY_KEYS', 'USE_TTS_KEYS',
                 'GRACEFUL_STOP', 'GRACEFUL_STOP_COMPLETED', 'TRANSLATION_CANCELLED',
                 'BATCH_TRANSLATION'):
        monkeypatch.setenv(name, '0')
    monkeypatch.setenv('GLOSSARY_REFINEMENT_API_KEYS', '[]')
    monkeypatch.setattr(UnifiedClient, '_glossary_refinement_key_pool', None)
    monkeypatch.setattr(UnifiedClient, '_in_memory_glossary_refinement_keys', None)
    UnifiedClient._global_cancelled = False


def set_refinement_pool(monkeypatch, enabled):
    """What the Key Pools dialog does when the Refinement switch is toggled."""
    monkeypatch.setenv('USE_GLOSSARY_REFINEMENT_KEYS', '1' if enabled else '0')
    monkeypatch.setenv('GLOSSARY_REFINEMENT_API_KEYS', json.dumps(REFINEMENT_KEYS))
    if enabled:
        UnifiedClient.set_in_memory_glossary_refinement_keys(REFINEMENT_KEYS)
    else:
        UnifiedClient.clear_in_memory_glossary_refinement_keys()


def test_refinement_pool_disable_and_reenable_routes_and_identifies_the_right_key(tmp_path, monkeypatch):
    client = UnifiedClient('sk-main', 'gpt-4o-mini', output_dir=str(tmp_path))
    sent = []

    def record(*args, **kwargs):
        pool = client.__dict__.get('_api_key_pool')
        sent.append((client.api_key, client.key_identifier,
                     pool is not None and pool is UnifiedClient._glossary_refinement_key_pool))
        return 'ok', 'stop'

    monkeypatch.setattr(client, '_send_internal', record)

    def send(context='glossary_refinement'):
        sent.clear()
        assert client._send_core(MESSAGES, context=context) == ('ok', 'stop')
        return sent[0]

    set_refinement_pool(monkeypatch, True)
    assert send() == ('sk-refine', 'GlossaryRefinementKey#1 (gpt-4o-mini)', True)
    assert client.key_identifier == 'Single Key'

    assert send('translation') == ('sk-main', 'Single Key', False)

    set_refinement_pool(monkeypatch, False)
    assert send() == ('sk-main', 'Single Key', False)

    set_refinement_pool(monkeypatch, True)
    assert send() == ('sk-refine', 'GlossaryRefinementKey#1 (gpt-4o-mini)', True)
    assert send('translation') == ('sk-main', 'Single Key', False)
    tls = client._get_thread_local_client()
    assert getattr(tls, 'key_identifier', None) != 'GlossaryRefinementKey#1 (gpt-4o-mini)'


@pytest.mark.parametrize('env_value,config_value,expected', [
    ('0', True, False),   # switched off in the dialog; config.json still says on
    ('1', False, True),
    (None, True, True),   # standalone run without the GUI's env: config decides
    (None, False, False),
])
def test_glossary_client_follows_the_live_refinement_switch_over_stale_config(
        tmp_path, monkeypatch, env_value, config_value, expected):
    import extract_glossary_from_epub as extractor

    if env_value is None:
        monkeypatch.delenv('USE_GLOSSARY_REFINEMENT_KEYS', raising=False)
    else:
        monkeypatch.setenv('USE_GLOSSARY_REFINEMENT_KEYS', env_value)
    monkeypatch.setenv('GLOSSARY_REFINEMENT_API_KEYS', '[]')
    config = {'use_glossary_refinement_keys': config_value, 'glossary_refinement_keys': REFINEMENT_KEYS}

    extractor.create_client_with_multi_key_support(
        'sk-main', 'gpt-4o-mini', str(tmp_path), config, context='glossary_refinement')

    assert (UnifiedClient._glossary_refinement_key_pool is not None) is expected
    assert (extractor.os.environ.get('USE_GLOSSARY_REFINEMENT_KEYS') == '1') is expected


@pytest.mark.parametrize('env_value,config_value,expected', [
    ('0', True, 8000),    # switched off in the dialog; config.json still says on
    ('1', False, 1000),
    (None, True, 1000),   # standalone run without the GUI's env: config decides
    (None, False, 8000),
])
def test_glossary_output_limit_follows_the_live_glossary_switch_over_stale_config(
        monkeypatch, env_value, config_value, expected):
    import extract_glossary_from_epub as extractor

    if env_value is None:
        monkeypatch.delenv('USE_GLOSSARY_KEYS', raising=False)
    else:
        monkeypatch.setenv('USE_GLOSSARY_KEYS', env_value)
    monkeypatch.setenv('GLOSSARY_API_KEYS', '[]')
    monkeypatch.setenv('GLOSSARY_MAX_OUTPUT_TOKENS', '8000')
    monkeypatch.setattr(extractor, '_peek_live_pool_output_limit', lambda: None)
    config = {
        'use_glossary_keys': config_value,
        'glossary_keys': [{'api_key': 'sk-g', 'model': 'gpt-4o-mini', 'individual_output_token_limit': 1000}],
    }

    assert extractor._effective_glossary_output_limit(config, 'test-model-without-cached-limit') == expected


@pytest.mark.parametrize('env_value,config_value,expected', [
    ('0', True, 8000),    # switched off in the dialog; config.json still says on
    ('1', False, 1000),
    (None, True, 1000),   # standalone run without the GUI's env: config decides
    (None, False, 8000),
])
def test_glossary_output_limit_follows_the_live_multi_key_switch_over_stale_config(
        monkeypatch, env_value, config_value, expected):
    import extract_glossary_from_epub as extractor

    monkeypatch.setenv('USE_GLOSSARY_KEYS', '0')
    if env_value is None:
        monkeypatch.delenv('USE_MULTI_API_KEYS', raising=False)
    else:
        monkeypatch.setenv('USE_MULTI_API_KEYS', env_value)
    monkeypatch.setenv('MULTI_API_KEYS', '[]')
    monkeypatch.setenv('GLOSSARY_MAX_OUTPUT_TOKENS', '8000')
    monkeypatch.setattr(extractor, '_peek_live_pool_output_limit', lambda: None)
    config = {
        'use_multi_api_keys': config_value,
        'multi_api_keys': [{'api_key': 'sk-m', 'model': 'gpt-4o-mini', 'individual_output_token_limit': 1000}],
    }

    assert extractor._effective_glossary_output_limit(config, 'test-model-without-cached-limit') == expected
