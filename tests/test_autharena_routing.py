from types import SimpleNamespace

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


def test_arena_does_not_report_unsupported_reasoning_settings(arena_client, monkeypatch):
    client, tls = arena_client
    tls.model = 'autharena3/gpt-6-astra-medium'
    monkeypatch.setenv('ENABLE_GPT_THINKING', '1')
    monkeypatch.setenv('GPT_EFFORT', 'max')
    assert client._get_thinking_status_label() == ''
