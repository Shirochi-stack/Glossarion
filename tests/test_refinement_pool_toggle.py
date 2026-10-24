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
