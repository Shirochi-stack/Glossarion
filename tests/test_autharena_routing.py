from types import SimpleNamespace
import json

import pytest

import unified_api_client as api


@pytest.fixture
def arena_client(monkeypatch):
    for name in ('GRACEFUL_STOP', 'GRACEFUL_STOP_COMPLETED', 'TRANSLATION_CANCELLED',
                 'USE_CUSTOM_OPENAI_ENDPOINT', 'CUSTOM_OPENAI_PREFIX_ROUTES'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(api, 'AUTHARENA_AVAILABLE', True)
    monkeypatch.setattr(api, '_autharena_reset_cancel', lambda: None)
    client = api.UnifiedClient.__new__(api.UnifiedClient)
    # The active thread's model must win over the shared client model.
    client.model = 'autharena/gpt-6-astra-medium'
    client.client_type = 'autharena'
    client.request_timeout = 42
    client._ignore_graceful_stop = False
    client._should_abort_retry = lambda: False
    client._is_stop_requested = lambda: client._is_local_cancel_requested()
    client._streaming_enabled = lambda: False
    tls = SimpleNamespace(model='autharena3/other-model', initialized=True)
    client._get_thread_local_client = lambda: tls
    return client, tls


def completed(**overrides):
    return dict(content='Translated text', finish_reason='stop',
                finish_reason_explicit=True, usage={'completion_tokens': 3},
                reasoning_content='Reasoning', **overrides)


@pytest.mark.parametrize('model', ['autharena/', 'autharena/gpt-6-astra-medium',
                                  'AUTHARENA3/model', 'autharena0/model'])
def test_arena_resolves_without_api_key(monkeypatch, model):
    monkeypatch.delenv('CUSTOM_OPENAI_PREFIX_ROUTES', raising=False)
    assert api.UnifiedClient._provider_from_model_name(model) == 'autharena'
    assert not api.UnifiedClient._model_needs_api_key(model)


def test_arena_route_passes_active_account_and_defers_progress(arena_client, monkeypatch):
    client, tls = arena_client
    events = []
    tls.current_request_id = 'arena-request'
    tls.pre_api_call_callback = lambda: events.append('progress')

    def claim(request_id, model, **kwargs):
        assert (request_id, model) == ('arena-request', 'autharena3/other-model')
        events.append('claim')
        return True

    def send(**kwargs):
        assert events == []
        assert kwargs['model'] == 'other-model'
        assert kwargs['account_id'] == 3
        assert kwargs['timeout'] == 42
        assert kwargs['messages'] == [{'role': 'user', 'content': 'Translate'}]
        assert not kwargs['cancel_check']()
        kwargs['before_send_callback']()
        kwargs['before_send_callback']()  # Progress is emitted once.
        events.append('post')
        return completed()

    monkeypatch.setattr(api, '_api_watchdog_mark_in_flight', claim)
    monkeypatch.setattr(api, '_autharena_send', send)
    result = client._send_autharena([{'role': 'user', 'content': 'Translate'}], .2, 99, 'chapter')
    assert events == ['claim', 'progress', 'post']
    assert result.content == 'Translated text'
    assert result.finish_reason == 'stop'
    assert result.usage == {'completion_tokens': 3}
    assert result.raw_response['reasoning_content'] == 'Reasoning'
    assert tls.pre_api_call_callback is None


@pytest.mark.parametrize('batch,shared,forced,provider,expected', [
    ('0', '1', '0', '1', True),
    ('0', '0', '1', '1', False),
    ('0', '1', '1', '0', False),
    ('1', '1', '0', '1', False),
    ('1', '0', '1', '1', True),
])
def test_arena_forced_stream_visibility_is_independent_of_optional_streaming(arena_client, monkeypatch,
                                                                          batch, shared, forced, provider, expected):
    client, _ = arena_client
    client._streaming_enabled = lambda: pytest.fail('optional transport toggle used for forced Arena stream')
    for key, value in {'BATCH_TRANSLATION': batch, 'LOG_STREAM_CHUNKS': shared,
                       'ALLOW_AUTHGPT_BATCH_STREAM_LOGS': forced, 'AUTHARENA_LOG_STREAM_CHUNKS': provider,
                       'ENABLE_STREAMING': '0'}.items():
        monkeypatch.setenv(key, value)

    def send(**kwargs):
        assert kwargs['stream'] is True
        assert kwargs['log_stream'] is expected
        return completed()

    monkeypatch.setattr(api, '_autharena_send', send)
    client._send_autharena([], .2, 99, 'chapter')


def test_arena_fragment_pipe_flushes_each_delta_and_preserves_exact_text(arena_client, monkeypatch):
    from streaming_log import decode_stream_fragment
    client, _ = arena_client
    emitted = []
    # This module routes print through its synchronous GUI logger wrapper.
    monkeypatch.setattr(api, 'print', lambda *args, **kwargs: emitted.append((args, kwargs)))

    def send(**kwargs):
        kwargs['log_fn']('📡 AuthArena: Text streaming...')
        assert emitted[-1][1]['flush'] is True
        for channel, text in [('content', 'a'), ('content', ' \n\t你'), ('reasoning', 'look')]:
            kwargs['log_chunk_fn'](channel, text)
            args, options = emitted[-1]
            assert options['flush'] is True
            assert decode_stream_fragment(args[0]) == {'channel': channel, 'text': text}
        return completed()

    monkeypatch.setattr(api, '_autharena_send', send)
    client._send_autharena([], .2, 99, 'chapter')


def test_graceful_stop_during_preparation_prevents_send(arena_client, monkeypatch):
    client, tls = arena_client
    events = []
    tls.pre_api_call_callback = lambda: events.append('progress')

    def send(**kwargs):
        monkeypatch.setenv('GRACEFUL_STOP', '1')
        assert kwargs['cancel_check']()
        kwargs['before_send_callback']()
        pytest.fail('A queued request must not be sent after Graceful Stop')

    monkeypatch.setattr(api, '_autharena_send', send)
    with pytest.raises(api.UnifiedClientError) as caught:
        client._send_autharena([], .2, 99, 'chapter')
    assert caught.value.error_type == 'cancelled'
    assert events == []


def test_graceful_stop_allows_dispatched_response_to_finish(arena_client, monkeypatch):
    client, _ = arena_client

    def send(**kwargs):
        kwargs['before_send_callback']()
        monkeypatch.setenv('GRACEFUL_STOP', '1')
        assert not kwargs['cancel_check']()
        return completed()

    monkeypatch.setattr(api, '_autharena_send', send)
    assert client._send_autharena([], .2, 99, 'chapter').content == 'Translated text'


def test_cleared_pending_request_cannot_dispatch_without_stop_flags(arena_client, monkeypatch):
    client, tls = arena_client
    tls.current_request_id = 'cleared-pending-row'
    monkeypatch.setattr(api, '_api_watchdog_mark_in_flight', lambda *args, **kwargs: False)

    def send(**kwargs):
        assert not kwargs['cancel_check']()
        kwargs['before_send_callback']()
        pytest.fail('A removed pending row must not be sent')

    monkeypatch.setattr(api, '_autharena_send', send)
    with pytest.raises(api.UnifiedClientError) as caught:
        client._send_autharena([], .2, 99, 'chapter')
    assert caught.value.error_type == 'cancelled'


def test_rejected_request_becomes_cancellable_during_verification(arena_client, monkeypatch):
    client, _ = arena_client

    def send(**kwargs):
        kwargs['before_send_callback']()
        monkeypatch.setenv('GRACEFUL_STOP', '1')
        assert not kwargs['cancel_check']()
        kwargs['after_rejection_callback']()
        assert kwargs['cancel_check']()
        kwargs['before_send_callback']()
        pytest.fail('Rejected request must not be retried after Graceful Stop')

    monkeypatch.setattr(api, '_autharena_send', send)
    with pytest.raises(api.UnifiedClientError) as caught:
        client._send_autharena([], .2, 99, 'chapter')
    assert caught.value.error_type == 'cancelled'


def test_local_cancel_remains_effective_after_dispatch(arena_client, monkeypatch):
    client, tls = arena_client

    def send(**kwargs):
        kwargs['before_send_callback']()
        tls.local_cancel_check = lambda: True
        assert kwargs['cancel_check']()
        raise RuntimeError('stream cancelled')

    monkeypatch.setattr(api, '_autharena_send', send)
    with pytest.raises(api.UnifiedClientError) as caught:
        client._send_autharena([], .2, 99, 'chapter')
    assert caught.value.error_type == 'cancelled'


def test_custom_stop_remains_effective_after_dispatch(arena_client, monkeypatch):
    client, _ = arena_client

    def send(**kwargs):
        kwargs['before_send_callback']()
        client._is_stop_requested = lambda: True
        assert kwargs['cancel_check']()
        raise RuntimeError('stream cancelled')

    monkeypatch.setattr(api, '_autharena_send', send)
    with pytest.raises(api.UnifiedClientError) as caught:
        client._send_autharena([], .2, 99, 'chapter')
    assert caught.value.error_type == 'cancelled'


def test_response_dispatch_keeps_browser_preflight_outside_watchdog(arena_client, monkeypatch):
    client, tls = arena_client
    client._effective_temperature = lambda value: value
    client._bind_thread_run_id_for_request = lambda: None
    client._restore_thread_endpoint_state_if_needed = lambda: None
    client._apply_api_call_stagger = lambda: None
    client._remember_actual_request_model = lambda: None
    client._normalize_token_params = lambda max_tokens, max_completion: (max_tokens, max_completion)
    tls.current_request_id = 'arena-dispatch'
    events = []
    tls.pre_api_call_callback = lambda: events.append('progress')
    monkeypatch.setattr(api, '_api_watchdog_mark_in_flight',
                        lambda *args, **kwargs: events.append('claim') or True)

    def send(**kwargs):
        assert events == []
        events.append('browser ready')
        kwargs['before_send_callback']()
        events.append('post')
        return completed()

    monkeypatch.setattr(api, '_autharena_send', send)
    response = client._get_response([], .2, 99, None, 'chapter', request_id='arena-dispatch')
    assert response.content == 'Translated text'
    assert events == ['browser ready', 'claim', 'progress', 'post']


@pytest.mark.parametrize('status,error_type', [(429, 'rate_limit'), (401, 'auth_error'),
                                             (403, 'auth_error'), (500, 'api_error')])
def test_http_errors_preserve_status_and_do_not_replay(arena_client, monkeypatch, status, error_type):
    client, _ = arena_client
    calls = []

    def send(**kwargs):
        calls.append(True)
        error = RuntimeError('Provider rejected request')
        error.status_code = status
        error.retry_after = 60
        raise error

    monkeypatch.setattr(api, '_autharena_send', send)
    with pytest.raises(api.UnifiedClientError) as caught:
        client._send_autharena([], .2, 99, 'chapter')
    assert caught.value.error_type == error_type
    assert caught.value.http_status == status
    assert caught.value.details['retry_after'] == 60
    assert calls == [True]


def test_missing_terminal_is_not_success_or_prohibited_fallback(arena_client, monkeypatch):
    client, _ = arena_client
    monkeypatch.setattr(api, '_autharena_send', lambda **_: {
        'content': 'Partial', 'finish_reason': 'stop', 'finish_reason_explicit': False,
    })
    with pytest.raises(api.UnifiedClientError, match='completion event') as caught:
        client._send_autharena([], .2, 99, 'chapter')
    assert caught.value.error_type == 'api_error'
    assert caught.value.details['request_dispatched'] is True


def test_arena_does_not_report_unsupported_reasoning_settings(arena_client, monkeypatch):
    client, tls = arena_client
    tls.model = 'autharena3/gpt-6-astra-medium'
    monkeypatch.setenv('ENABLE_GPT_THINKING', '1')
    monkeypatch.setenv('GPT_EFFORT', 'max')
    assert client._get_thinking_status_label() == ''


@pytest.mark.parametrize('route,passed_model,account_id', [
    ('autharena/model', 'model', 0),
    ('AUTHARENA0/model', 'autharena0/model', 0),
    ('autharena4/model', 'model', 4),
])
def test_default_pool_and_numbered_routes_remain_distinct(arena_client, monkeypatch, route, passed_model, account_id):
    client, tls = arena_client
    tls.model = route
    selected_account = 2 if route.lower().startswith('autharena0/') else account_id

    def send(**kwargs):
        assert kwargs['model'] == passed_model
        assert kwargs['account_id'] == account_id
        return completed(account_id=selected_account)

    monkeypatch.setattr(api, '_autharena_send', send)
    response = client._send_autharena([], .2, 99, 'chapter')
    assert response.raw_response['account_id'] == selected_account


@pytest.fixture
def retrying_arena_client(arena_client, monkeypatch, tmp_path):
    """Exercise the real provider router, internal retries, and public send wrapper."""
    for name, value in {
        'USE_MULTI_API_KEYS': '0', 'MAX_RETRIES': '3', 'RETRY_TIMEOUT': '1',
        'INDEFINITE_RATE_LIMIT_RETRY': '0', 'BATCH_TRANSLATION': '0',
        'SEND_INTERVAL_SECONDS': '0', 'THREAD_SUBMISSION_DELAY_SECONDS': '0',
        'USE_FALLBACK_KEYS': '0', 'USE_GLOSSARY_KEYS': '0',
        'DISABLE_REFUSAL_CHECKS': '1', 'SYSTEM_PROMPT_TO_USER': '0',
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(api, '_payloads_dir', lambda: str(tmp_path / 'payloads'))
    monkeypatch.setattr(api.time, 'sleep', lambda *_: None)
    for name in ('_save_payload', '_save_failed_request', '_save_response', '_track_stats',
                 '_apply_api_call_stagger', '_apply_thread_submission_delay',
                 '_refresh_rotation_settings_from_environment'):
        monkeypatch.setattr(api.UnifiedClient, name, lambda *a, **k: None)
    for name in ('_api_watchdog_started', '_api_watchdog_finished', '_api_watchdog_record_retry'):
        monkeypatch.setattr(api, name, lambda *a, **k: None)
    monkeypatch.setattr(api, '_api_watchdog_mark_in_flight', lambda *a, **k: True)
    client = api.UnifiedClient('', 'autharena0/gpt-6-astra-medium',
                               output_dir=str(tmp_path), _skip_cancel_reset=True)
    client._should_abort_retry = lambda: False
    client._is_stop_requested = lambda: False
    client._get_max_retries = lambda: 3
    client._compute_backoff = lambda *a, **k: 0
    client._sleep_with_cancel = lambda *a, **k: True
    client._ensure_thread_client = lambda: None
    return client


@pytest.mark.parametrize('multi_key', [False, True])
@pytest.mark.parametrize('failure', ['timeout', 'eof', 'parser', 'missing_terminal'])
def test_outer_send_does_not_replay_ambiguous_arena_generation(
    retrying_arena_client, monkeypatch, multi_key, failure,
):
    client = retrying_arena_client
    client._multi_key_mode = multi_key
    attempts = []
    rotations = []
    client._handle_rate_limit_for_thread = lambda: rotations.append(True)

    def send(**kwargs):
        attempts.append(kwargs['model'])
        kwargs['before_send_callback']()
        if failure == 'missing_terminal':
            return {'content': 'Partial', 'finish_reason_explicit': False}
        error_class = {'timeout': TimeoutError, 'eof': RuntimeError, 'parser': ValueError}[failure]
        error = error_class(f'Arena response {failure}')
        error.request_dispatched = True
        raise error

    monkeypatch.setattr(api, '_autharena_send', send)
    with pytest.raises(api.UnifiedClientError) as caught:
        client._send_core([{'role': 'user', 'content': 'Translate this sentence.'}],
                          .2, 99, context='translation')
    assert caught.value.details == {'provider': 'autharena', 'request_dispatched': True}
    assert attempts == ['autharena0/gpt-6-astra-medium']
    assert rotations == []


def test_explicit_arena_rejection_remains_retryable(retrying_arena_client, monkeypatch):
    client = retrying_arena_client
    attempts = []

    def send(**kwargs):
        attempts.append(True)
        kwargs['before_send_callback']()
        if len(attempts) == 1:
            error = RuntimeError('Arena explicitly rejected request')
            error.status_code = 500
            # This explicit adapter decision must override the callback's started state.
            error.request_dispatched = False
            raise error
        return completed()

    monkeypatch.setattr(api, '_autharena_send', send)
    result = client._send_core([{'role': 'user', 'content': 'Translate this sentence.'}],
                               .2, 99, context='translation')
    assert result == ('Translated text', 'stop')
    assert len(attempts) == 2


def test_ambiguous_guard_does_not_change_other_provider_retries(retrying_arena_client, monkeypatch):
    client = retrying_arena_client
    attempts = []

    def get_response(*args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise api.UnifiedClientError('Server error', error_type='api_error',
                                        details={'provider': 'openai', 'request_dispatched': True})
        return api.UnifiedResponse(content='Translated text', finish_reason='stop')

    monkeypatch.setattr(client, '_get_response', get_response)
    result = client._send_core([{'role': 'user', 'content': 'Translate this sentence.'}],
                               .2, 99, context='translation')
    assert result == ('Translated text', 'stop')
    assert len(attempts) == 2


def test_outer_multi_key_rotation_preserves_ambiguous_error(retrying_arena_client, monkeypatch):
    client = retrying_arena_client
    client._multi_key_mode = True
    error = api.UnifiedClientError('Rate limit response interrupted', error_type='rate_limit',
                                   http_status=429,
                                   details={'provider': 'autharena', 'request_dispatched': True})
    calls = []
    rotations = []

    def send_internal(*args, **kwargs):
        calls.append(True)
        raise error

    monkeypatch.setattr(client, '_send_internal', send_internal)
    monkeypatch.setattr(client, '_handle_rate_limit_for_thread', lambda: rotations.append(True))
    with pytest.raises(api.UnifiedClientError) as caught:
        client._send_core([{'role': 'user', 'content': 'Translate this sentence.'}],
                          .2, 99, context='translation')
    assert caught.value is error
    assert calls == [True]
    assert rotations == []


@pytest.mark.parametrize('method', ['_try_fallback_keys_direct', '_retry_with_main_key',
                                   '_try_glossary_keys_direct'])
def test_fallback_pools_stop_after_ambiguous_arena_attempt(
    retrying_arena_client, monkeypatch, method,
):
    client = retrying_arena_client
    # These tests exercise the actual fallback loops and their temporary clients.
    client._multi_key_mode = method == '_retry_with_main_key'
    keys = [{'model': f'autharena{i}/gpt-6-astra-medium', 'api_key': 'unused',
             'api_call_delay': 0.01} for i in (1, 2)]
    monkeypatch.setenv('USE_MAIN_KEY_FALLBACK', '0')
    monkeypatch.setenv('USE_FALLBACK_KEYS', '1')
    monkeypatch.setenv('USE_GLOSSARY_KEYS', '1')
    monkeypatch.setenv('FALLBACK_KEYS', json.dumps(keys))
    monkeypatch.setenv('GLOSSARY_API_KEYS', json.dumps(keys))
    monkeypatch.setattr(api, '_fallback_key_last_used', {})
    monkeypatch.setattr(api, '_fallback_key_in_use', set())
    attempts = []

    def send(**kwargs):
        attempts.append(kwargs['account_id'])
        kwargs['before_send_callback']()
        error = RuntimeError('Arena response disconnected')
        error.request_dispatched = True
        raise error

    monkeypatch.setattr(api, '_autharena_send', send)
    with pytest.raises(api.UnifiedClientError) as caught:
        getattr(client, method)([{'role': 'user', 'content': 'Translate this sentence.'}],
                                 .2, 99, context='translation', request_id='arena-pool-test')
    assert caught.value.details['request_dispatched'] is True
    assert attempts == [1]
    assert api._fallback_key_in_use == set()
