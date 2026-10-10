import ast
import copy
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import authnan_auth as auth
import unified_api_client as api
import model_options
import settings_rules

CHAT = 'https://nano-gpt.com/api/subscription/v1/chat/completions'
MESSAGES = [{'role': 'user', 'content': 'Reply OK.'}]


@pytest.fixture(autouse=True)
def isolation(monkeypatch, tmp_path):
    monkeypatch.setattr(auth, '_DEFAULT_TOKEN_DIR', str(tmp_path))
    monkeypatch.setattr(auth, '_DEFAULT_TOKEN_FILE', str(tmp_path / 'authnan_tokens.json'))
    auth._stores.clear()
    auth._cancel_event.clear()
    auth._pool_rotation_cursor = 0
    for name in ('AUTHNAN_TOKEN_FILE', 'TRANSLATION_CANCELLED', 'GRACEFUL_STOP', 'MULTI_API_KEYS'):
        monkeypatch.delenv(name, raising=False)
    for name in ('ENABLE_IMAGE_OUTPUT_MODE', 'ENABLE_VIDEO_OUTPUT_MODE', 'ENABLE_STREAMING', 'USE_CUSTOM_OPENAI_ENDPOINT'):
        monkeypatch.setenv(name, '0')
    monkeypatch.setenv('GEMINI_SERVICE_TIER', 'off')
    monkeypatch.setattr(api, '_save_outgoing_request', lambda *a, **kw: None)
    monkeypatch.setattr(api, '_save_incoming_response', lambda *a, **kw: None)
    yield
    auth.cancel_stream()
    auth._cancel_event.clear()


def make_client(monkeypatch, tmp_path, model='authnan/openai/gpt-test', retries=2):
    client = api.UnifiedClient('', model, str(tmp_path))
    monkeypatch.setattr(client, '_get_max_retries', lambda: retries)
    monkeypatch.setattr(client, '_get_send_interval', lambda: 0)
    monkeypatch.setattr(client, '_streaming_enabled', lambda: False)
    monkeypatch.setattr(client, '_is_stop_requested', lambda: False)
    monkeypatch.setattr(client, '_should_abort_retry', lambda: False)
    monkeypatch.setattr(client, '_save_response', lambda *a, **kw: None)
    monkeypatch.setattr(client, '_should_show_api_lifecycle_logs', lambda: False)
    monkeypatch.setattr(client, '_sleep_with_cancel', lambda *a, **kw: True)
    return client


class Response:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code = status
        self.headers = {'content-type': 'application/json', **(headers or {})}
        self.payload = payload or {'choices': [{'message': {'content': 'OK'}, 'finish_reason': 'stop'}]}
        self.text = str(self.payload)
        self.closed = False
    def json(self): return self.payload
    def close(self): self.closed = True


def transport(monkeypatch, client, use_sdk, outcomes, calls):
    def send(url, key, payload):
        calls.append((url, key, copy.deepcopy(payload)))
        result = outcomes.pop(0) if outcomes else 200
        if callable(result): return result(url, key, payload)
        if use_sdk and result != 200:
            error = RuntimeError('HTTP rejection')
            error.status_code = result
            error.response = Response(result, headers={'Retry-After': '17'})
            raise error
        if use_sdk:
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='OK'), finish_reason='stop')], usage=None)
        return Response(result, headers={'Retry-After': '17'})
    if use_sdk:
        def sdk(**kwargs):
            return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
                create=lambda **payload: send(kwargs['base_url'].rstrip('/') + '/chat/completions', kwargs['api_key'], payload))), close=lambda: None)
        monkeypatch.setattr(api.openai, 'OpenAI', sdk)
        monkeypatch.setattr(api, 'httpx', None)
    else:
        monkeypatch.setattr(api, 'openai', None)
        monkeypatch.setattr(client, '_get_session', lambda *_: SimpleNamespace(request=lambda method, url, **kw:
            send(url, kw['headers']['Authorization'].removeprefix('Bearer '), kw['json'])))


@pytest.mark.parametrize('route,slot', [('authnan/m', 0), ('AUTHNAN9/m', 9), ('AuthNan9999/m', 9999)])
def test_recognition_and_blank_keys(monkeypatch, tmp_path, route, slot):
    client = make_client(monkeypatch, tmp_path, route)
    assert not client._model_needs_api_key(route)
    assert client._provider_from_model_name(route) == 'authnan'
    assert client.client_type == client._get_actual_provider() == 'authnan'
    auth.get_store(slot).save_tokens({'key': 'saved-key'})
    monkeypatch.setattr(client, '_send_openai_compatible', lambda **kw: kw)
    request = client._send_authnan(MESSAGES, 0.5, 30, 'test')
    assert request['api_key_override'] == 'saved-key'
    assert request['model_override'] == 'm'
    assert request['base_url'] + '/chat/completions' == CHAT


@pytest.mark.parametrize('use_sdk', [True, False])
@pytest.mark.parametrize('first_status', [200, 401, 429])
def test_subscription_transport_retries_and_custom_endpoint_isolation(monkeypatch, tmp_path, use_sdk, first_status):
    auth.get_store(7).save_tokens({'key': 'slot-seven'})
    auth.get_store(8).save_tokens({'key': 'other-account'})
    monkeypatch.setenv('USE_CUSTOM_OPENAI_ENDPOINT', '1')
    monkeypatch.setenv('OPENAI_CUSTOM_BASE_URL', 'https://wrong.invalid/v1')
    monkeypatch.setenv('CUSTOM_IMAGE_EDIT_BASE_URL', 'https://wrong.invalid/images')
    monkeypatch.setenv('NANOGPT_API_URL', 'https://wrong.invalid')
    monkeypatch.setenv('CUSTOM_MODEL_ROUTES', '[{"prefix":"authnan", "routing":"https://wrong.invalid/v1"}]')
    client = make_client(monkeypatch, tmp_path, 'AuthNan7/openai/gpt-test')
    monkeypatch.setenv('USE_CUSTOM_OPENAI_ENDPOINT', '1')
    tls = client._get_thread_local_client()
    tls.initialized, tls.model, tls.api_key = True, 'AuthNan7/openai/gpt-test', 'wrong-key'
    tls.use_individual_endpoint, tls.azure_endpoint = True, 'https://wrong.invalid/v1'
    client.current_key_use_individual_endpoint = True
    client.current_key_endpoint = 'https://wrong.invalid/v1'
    assert client._get_actual_provider() == 'authnan'
    logins = []
    def login(store, **kw):
        logins.append(store._account_id)
        store.save_tokens({'key': 'replacement-seven'})
        return store.load_tokens()
    monkeypatch.setattr(auth, 'run_oauth_flow', login)
    waits = []
    monkeypatch.setattr(client, '_sleep_with_cancel', lambda wait, *_: waits.append(wait) or True)
    calls = []
    transport(monkeypatch, client, use_sdk, [first_status, 200], calls)
    result = client._send_authnan(MESSAGES, 0.5, 30, 'test')
    assert result.content == 'OK'
    assert all(url == CHAT and payload['model'] == 'openai/gpt-test' for url, key, payload in calls)
    assert [key for _, key, _ in calls] == (['slot-seven'] if first_status == 200 else ['slot-seven', 'replacement-seven' if first_status == 401 else 'slot-seven'])
    assert logins == ([7] if first_status == 401 else [])
    assert waits == ([17] if first_status == 429 else [])
    assert auth.get_store(8).get_valid_access_token(False) == 'other-account'


@pytest.mark.parametrize('use_sdk', [True, False])
def test_one_reauthentication_then_invalidate_only_rejected_slot(monkeypatch, tmp_path, use_sdk):
    auth.get_store().save_tokens({'key': 'old'})
    auth.get_store(1).save_tokens({'key': 'keep'})
    logins = []
    def login(store, **_):
        logins.append(store._account_id)
        store.save_tokens({'key': 'new'})
        return store.load_tokens()
    monkeypatch.setattr(auth, 'run_oauth_flow', login)
    client = make_client(monkeypatch, tmp_path)
    calls = []
    transport(monkeypatch, client, use_sdk, [401, 401, 200], calls)
    with pytest.raises(api.UnifiedClientError) as error:
        client._send_authnan(MESSAGES, 0.5, 30, 'test')
    assert error.value.http_status == 401 and logins == [0]
    assert len(calls) == 2 and all(call[0] == CHAT for call in calls)
    assert not auth.get_store().has_tokens and auth.get_store(1).has_tokens


@pytest.mark.parametrize('use_sdk', [True, False])
@pytest.mark.parametrize('status', [401, 429])
def test_pool_advances_without_browser_and_preserves_rate_metadata(monkeypatch, tmp_path, use_sdk, status):
    for slot in (0, 2): auth.get_store(slot).save_tokens({'key': f'key-{slot}', 'user_id': f'user-{slot}'})
    monkeypatch.setattr(auth, 'run_oauth_flow', lambda **_: pytest.fail('pool opened browser'))
    client = make_client(monkeypatch, tmp_path, 'authnan0/openai/gpt-test')
    calls = []
    transport(monkeypatch, client, use_sdk, [status, 200], calls)
    assert client._send_authnan(MESSAGES, 0.5, 30, 'test').content == 'OK'
    assert [key for _, key, _ in calls] == ['key-0', 'key-2']
    assert auth.get_store().has_tokens == (status == 429)
    transport(monkeypatch, client, use_sdk, [429, 429], calls)
    with pytest.raises(api.UnifiedClientError) as error:
        client._send_authnan(MESSAGES, 0.5, 30, 'test')
    assert error.value.http_status == 429
    assert error.value.details['retry_after_seconds'] == 17
    assert all(url == CHAT for url, _, _ in calls)


def test_empty_pool_never_opens_browser(monkeypatch, tmp_path):
    monkeypatch.setattr(auth, 'run_oauth_flow', lambda **_: pytest.fail('empty pool login'))
    client = make_client(monkeypatch, tmp_path, 'authnan0/m')
    with pytest.raises(api.UnifiedClientError, match='pool is empty'):
        client._send_authnan(MESSAGES, 0.5, 30, 'test')


@pytest.mark.parametrize('slot', [0, 3])
def test_missing_default_or_pinned_key_can_login(monkeypatch, tmp_path, slot):
    def login(store, **_):
        assert store._account_id == slot
        store.save_tokens({'key': 'new'})
        return store.load_tokens()
    monkeypatch.setattr(auth, 'run_oauth_flow', login)
    client = make_client(monkeypatch, tmp_path, f'authnan{slot or ""}/m')
    monkeypatch.setattr(client, '_send_nanogpt', lambda *a, **kw: kw['api_key_override'])
    assert client._send_authnan(MESSAGES, 0.5, 30, 'test') == 'new'


@pytest.mark.parametrize('use_sdk', [True, False])
def test_concurrent_requests_keep_their_own_account_and_model(monkeypatch, tmp_path, use_sdk):
    for slot in (1, 2): auth.get_store(slot).save_tokens({'key': f'key-{slot}'})
    client = make_client(monkeypatch, tmp_path)
    barrier = threading.Barrier(2)
    calls = []
    def receive(url, key, payload):
        barrier.wait(timeout=5)
        assert key == 'key-' + payload['model'].split('-')[-1]
        assert url == CHAT
        if use_sdk: return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=key), finish_reason='stop')], usage=None)
        return Response(payload={'choices': [{'message': {'content': key}, 'finish_reason': 'stop'}]})
    transport(monkeypatch, client, use_sdk, [receive, receive], calls)
    def send(slot):
        tls = client._get_thread_local_client()
        tls.initialized, tls.model, tls.api_key = True, f'authnan{slot}/model-{slot}', 'stale'
        return client._send_authnan(MESSAGES, 0.5, 30, 'test').content
    with ThreadPoolExecutor(2) as workers:
        assert set(workers.map(send, (1, 2))) == {'key-1', 'key-2'}


@pytest.mark.parametrize('mode,url', [('image', 'https://nano-gpt.com/api/v1/images/generations'), ('video', 'https://nano-gpt.com/api/generate-video')])
def test_media_uses_separate_routes_and_key(monkeypatch, tmp_path, mode, url):
    auth.get_store().save_tokens({'key': 'media-key'})
    client = make_client(monkeypatch, tmp_path, 'authnan/media-model')
    monkeypatch.setenv('ENABLE_IMAGE_OUTPUT_MODE' if mode == 'image' else 'ENABLE_VIDEO_OUTPUT_MODE', '1')
    calls = []
    monkeypatch.setattr(api.requests, 'post', lambda actual_url, **kw: calls.append((actual_url, kw)) or
        Response(payload={'data': [{'url': 'image-result'}]} if mode == 'image' else {'runId': 'run'}))
    monkeypatch.setattr(api.requests, 'get', lambda actual_url, **kw: calls.append((actual_url, kw)) or
        Response(payload={'status': 'COMPLETED', 'videoUrl': 'video-result'}))
    monkeypatch.setattr(api.time, 'sleep', lambda *_: None)
    monkeypatch.setattr(auth, 'run_oauth_flow', lambda **_: pytest.fail('media polling login'))
    result = client._send_authnan(MESSAGES, 0.5, 30, 'test')
    assert calls[0][0] == url
    header = 'Authorization' if mode == 'image' else 'x-api-key'
    assert calls[0][1]['headers'][header] == ('Bearer media-key' if mode == 'image' else 'media-key')
    assert result.content == mode + '-result'
    if mode == 'video': assert calls[1][0] == 'https://nano-gpt.com/api/generate-video/status'


def test_authenticated_catalog_keeps_pool_prefix_and_never_logs_in(monkeypatch):
    monkeypatch.setattr(auth, 'run_oauth_flow', lambda **_: pytest.fail('catalog login'))
    with pytest.raises(RuntimeError): model_options._fetch_authenticated_catalog('authnan/model', 5)
    auth.get_store(2).save_tokens({'key': 'key-2'})
    monkeypatch.setattr(auth, 'fetch_available_models', lambda token, **kw: ['text-model', 'image-model', 'video-model'])
    name, models = model_options._fetch_authenticated_catalog('AUTHNAN0/model', 5)
    assert name == 'authnan:pool' and all(model.startswith('authnan0/') for model in models)
    name, models = model_options._fetch_authenticated_catalog('AUTHNAN2/model', 5)
    assert name == 'authnan:2' and all(model.startswith('authnan2/') for model in models)


@pytest.mark.parametrize('pool', ['multi_api_keys', 'fallback_keys', 'glossary_keys', 'glossary_refinement_keys', 'metadata_keys', 'qa_scan_keys', 'rolling_summary_keys', 'truncation_retry_keys', 'inpainter_keys', 'tts_keys'])
def test_enabled_pools_reveal_login_controls(pool):
    toggle = settings_rules.ENABLED_KEY_POOL_TOGGLES[pool]
    config = {pool: [{'model': 'AUTHNAN0/model', 'api_key': '', 'enabled': True}], toggle: True}
    controls = settings_rules.route_controls('other/model', config)
    assert controls.needs_login('authnan') and controls.authnan_pool
    config[pool][0]['model'] = 'AUTHNAN7/model'
    assert settings_rules.model_account_ids('other/model', config)['authnan'] == {7}
    config[toggle] = False
    assert not settings_rules.route_controls('other/model', config).needs_login('authnan')


@pytest.fixture(scope='module')
def gui_methods():
    tree = ast.parse((Path(__file__).parents[1] / 'src/translator_gui.py').read_text(encoding='utf-8-sig'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TranslatorGUI')
    names = {'_get_authnan_account_id', '_update_authnan_login_status', '_authnan_login_clicked', '_refresh_auth_account_arrows', '_on_auth_acct_combo_changed'}
    ns = {'threading': threading, 'Qt': SimpleNamespace(QueuedConnection=1), 'QMetaObject': SimpleNamespace(invokeMethod=lambda *a: None)}
    exec(compile(ast.Module(body=[n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]), '<gui>', 'exec'), ns)
    return ns


def test_pinned_control_ignores_other_rotating_pools(gui_methods):
    gui = SimpleNamespace(model_var='authnan7/m', config={'use_multi_api_keys': True, 'multi_api_keys': [{'model': 'authnan0/m'}]},
        _auth_account_ids={'authnan': [0, 2, 7]}, _auth_account_idx={'authnan': 1})
    assert gui_methods['_get_authnan_account_id'](gui) == 7


def test_live_manager_pinned_hint_follows_its_slot(gui_methods):
    gui = SimpleNamespace(model_var='other/model', config={}, _multi_key_manager_authnan_model_hint='AUTHNAN5/model')
    assert gui_methods['_get_authnan_account_id'](gui) == 5
    assert not settings_rules.authnan_login_needed('other/model')
    assert not settings_rules.authnan_pool_route_requested('other/model')


@pytest.mark.parametrize('new_account', [False, True])
def test_login_captures_slot_before_worker_and_model_switch(gui_methods, monkeypatch, new_account):
    gui = SimpleNamespace(model_var='authnan1/m', config={}, _auth_status_cache={('authnan', 1): SimpleNamespace(has_tokens=False)},
        _update_authnan_login_status=lambda: None, _authnan_fresh_account_ids={1} if new_account else set())
    gui._get_authnan_account_id = lambda: gui_methods['_get_authnan_account_id'](gui)
    work = []
    monkeypatch.setitem(gui_methods, 'threading', SimpleNamespace(Thread=lambda target, **_: SimpleNamespace(start=lambda: work.append(target))))
    signed_in = []
    monkeypatch.setattr(auth, 'run_oauth_flow', lambda store, new_account: signed_in.append((store._account_id, new_account)))
    gui_methods['_authnan_login_clicked'](gui)
    gui.model_var = 'authnan2/m'
    work[0]()
    assert signed_in == [(1, new_account)] and gui._authnan_login_account_id == 1
    assert gui._authnan_fresh_account_ids == set()


def test_failed_add_account_keeps_fresh_browser_on_retry(gui_methods, monkeypatch):
    gui = SimpleNamespace(_get_authnan_account_id=lambda: 1, _update_authnan_login_status=lambda: None,
        _auth_status_cache={('authnan', 1): SimpleNamespace(has_tokens=False)}, _authnan_fresh_account_ids={1})
    work = []
    monkeypatch.setitem(gui_methods, 'threading', SimpleNamespace(Thread=lambda target, **_: SimpleNamespace(start=lambda: work.append(target))))
    def login(store, new_account):
        assert store._account_id == 1 and new_account
        raise RuntimeError('approval denied')
    monkeypatch.setattr(auth, 'run_oauth_flow', login)
    gui_methods['_authnan_login_clicked'](gui)
    work[0]()
    assert gui._authnan_fresh_account_ids == {1} and gui._authnan_login_error == 'approval denied'


def test_status_and_billing_tooltip(gui_methods):
    class Button:
        def setEnabled(self, value): self.enabled = value
        def setText(self, value): self.text = value
        def setToolTip(self, value): self.tip = value
        def setStyleSheet(self, value): pass
    button = Button()
    gui = SimpleNamespace(authnan_login_btn=button, _get_authnan_account_id=lambda: 7,
        _auth_status_snapshot=lambda _: SimpleNamespace(has_tokens=True, account_info={'email': '<private>'}))
    gui_methods['_update_authnan_login_status'](gui)
    assert button.enabled and '#7' in button.text
    assert '&lt;private&gt;' in button.tip and 'overage' in button.tip and 'provider selection' in button.tip


def test_all_desktop_packages_and_shutdown_include_authnan():
    src = Path(__file__).parents[1] / 'src'
    variants = [path for path in src.glob('*.spec') if 'authgpt_auth' in path.read_text(encoding='utf-8-sig')]
    assert len(variants) == 14
    for path in variants:
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        literals = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)]
        assert 'authnan_auth' in literals and 'authnan_auth.py' in literals, path.name
    assert 'authnan_auth' in (src / 'shutdown_utils.py').read_text(encoding='utf-8-sig')


@pytest.mark.parametrize('use_sdk', [True, False])
def test_temperature_repair_keeps_subscription_url_and_key(monkeypatch, tmp_path, use_sdk):
    auth.get_store().save_tokens({'key': 'repair-key'})
    client = make_client(monkeypatch, tmp_path, 'authnan/unique-temperature-test')
    calls = []
    def rejected(url, key, payload):
        if use_sdk:
            error = RuntimeError('temperature is not supported')
            error.status_code, error.response = 400, Response(400, {'error': {'message': str(error)}})
            raise error
        return Response(400, {'error': {'message': 'temperature is not supported'}})
    transport(monkeypatch, client, use_sdk, [rejected, 200], calls)
    # Each transport needs to exercise its own compatibility repair cache.
    monkeypatch.setattr(api, 'model_rejects_temperature', lambda *_: False)
    assert client._send_authnan(MESSAGES, 0.5, 30, 'repair').content == 'OK'
    assert len(calls) == 2
    assert all(url == CHAT and key == 'repair-key' for url, key, _ in calls)
    assert 'temperature' in calls[0][2] and 'temperature' not in calls[1][2]


@pytest.mark.parametrize('use_sdk', [True, False])
@pytest.mark.parametrize('model,expected_tier', [('openai/gpt-test', 'flex'), ('google/gemini-test', 'flex'), ('anthropic/claude-test', None)])
def test_thinking_and_service_tier_eligibility(monkeypatch, tmp_path, use_sdk, model, expected_tier):
    auth.get_store().save_tokens({'key': 'tier-key'})
    client = make_client(monkeypatch, tmp_path, 'authnan/' + model)
    monkeypatch.setenv('ENABLE_GPT_THINKING', '1')
    monkeypatch.setenv('GPT_EFFORT', 'high')
    monkeypatch.setenv('GEMINI_SERVICE_TIER', 'flex')
    catalogs = []
    def catalog(url, **kw):
        catalogs.append((url, kw))
        response = Response(payload={'data': [{'id': model, 'supported_service_tiers': ['flex']}]})
        response.raise_for_status = lambda: None
        return response
    monkeypatch.setattr(api.requests, 'get', catalog)
    calls = []
    transport(monkeypatch, client, use_sdk, [200], calls)
    assert client._send_authnan(MESSAGES, 0.5, 30, 'tier').content == 'OK'
    payload = calls[0][2]
    assert payload.get('service_tier') == expected_tier
    assert payload.get('reasoning_effort', payload.get('extra_body', {}).get('reasoning_effort')) == 'high'
    if expected_tier:
        assert catalogs[0][0] == 'https://nano-gpt.com/api/subscription/v1/models'
        assert catalogs[0][1]['params'] == {'detailed': 'true'}
        assert catalogs[0][1]['headers']['Authorization'] == 'Bearer tier-key'
    else: assert not catalogs


def test_sdk_stream_stays_on_subscription_transport(monkeypatch, tmp_path):
    auth.get_store().save_tokens({'key': 'stream-key'})
    client = make_client(monkeypatch, tmp_path)
    monkeypatch.setattr(client, '_streaming_enabled', lambda: True)
    calls = []
    class Stream:
        closed = False
        def __iter__(self):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='OK'), finish_reason='stop')])
        def close(self): self.closed = True
    stream = Stream()
    def create(url, key, payload):
        assert payload['stream'] is True
        return stream
    transport(monkeypatch, client, True, [create], calls)
    assert client._send_authnan(MESSAGES, 0.5, 30, 'stream').content == 'OK'
    assert calls[0][0:2] == (CHAT, 'stream-key') and stream.closed


@pytest.mark.parametrize('status', [401, 429])
def test_video_status_error_never_opens_login_or_submits_again(monkeypatch, tmp_path, status):
    auth.get_store().save_tokens({'key': 'video-key'})
    client = make_client(monkeypatch, tmp_path, 'authnan/video-model')
    monkeypatch.setenv('ENABLE_VIDEO_OUTPUT_MODE', '1')
    submits = []
    monkeypatch.setattr(api.requests, 'post', lambda *a, **kw: submits.append(a[0]) or Response(payload={'runId': 'existing-job'}))
    monkeypatch.setattr(api.requests, 'get', lambda *a, **kw: Response(status, headers={'Retry-After': '21'}))
    monkeypatch.setattr(auth, 'run_oauth_flow', lambda **_: pytest.fail('video poll opened login'))
    with pytest.raises(api.UnifiedClientError) as error:
        client._send_authnan(MESSAGES, 0.5, 30, 'video')
    assert len(submits) == 1 and error.value.details['job_submitted']
    assert error.value.details['run_id'] == 'existing-job'
    assert error.value.details['retry_after_seconds'] == 21
    assert auth.get_store().has_tokens == (status != 401)


@pytest.fixture(scope='module')
def qapp():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize('model,visible,slot', [('authnan/m', False, 0), ('authnan7/m', False, 7), ('authnan0/m', True, 2)])
def test_desktop_account_selector_and_add_account(qapp, gui_methods, monkeypatch, model, visible, slot):
    from PySide6.QtWidgets import QComboBox, QPushButton
    combo, button = QComboBox(), QPushButton()
    gui = SimpleNamespace(model_var=model, config={}, authnan_acct_combo=combo, authnan_login_btn=button,
        _auth_status_cache={}, _auth_account_ids={'authnan': [0, 2, 7]}, _auth_account_idx={'authnan': 1},
        _collect_auth_account_ids_from_pools=lambda: {p: ({0, 2, 7} if p == 'authnan' else set()) for p in settings_rules.AUTH_ACCOUNT_ROUTE_PATTERNS},
        _authgrok_pool_route_requested=lambda _: False, _authgpt_pool_route_requested=lambda _: False,
        _authgem_vertex_control_model=lambda: '')
    gui._refresh_auth_account_arrows = lambda: gui_methods['_refresh_auth_account_arrows'](gui)
    gui._get_authnan_account_id = lambda: gui_methods['_get_authnan_account_id'](gui)
    gui._refresh_auth_account_arrows()
    assert combo.isVisible() == visible and gui._get_authnan_account_id() == slot
    if visible:
        assert combo.itemText(combo.count() - 1) == '+ N'
        logins = []
        gui._authnan_login_clicked = lambda: logins.append(gui._get_authnan_account_id())
        monkeypatch.setitem(gui_methods, 'QTimer', SimpleNamespace(singleShot=lambda _, cb: cb()))
        gui_methods['_on_auth_acct_combo_changed'](gui, 'authnan', combo.count() - 1)
        assert logins == [1] and combo.currentData() == 1
        assert gui._auth_status_cache[('authnan', 1)].has_tokens is False
        assert gui._authnan_fresh_account_ids == {1}


def test_desktop_logout_removes_only_captured_slot(gui_methods, monkeypatch):
    auth.get_store(1).save_tokens({'key': 'remove'})
    auth.get_store(2).save_tokens({'key': 'keep'})
    gui = SimpleNamespace(_get_authnan_account_id=lambda: 1,
        _auth_status_cache={('authnan', 1): SimpleNamespace(has_tokens=True)}, _update_authnan_login_status=lambda: None)
    monkeypatch.setitem(gui_methods, 'QMessageBox', SimpleNamespace(Yes=1, No=2, question=lambda *a: 1))
    work = []
    monkeypatch.setitem(gui_methods, 'threading', SimpleNamespace(Thread=lambda target, **_: SimpleNamespace(start=lambda: work.append(target))))
    gui_methods['_authnan_login_clicked'](gui)
    work[0]()
    assert not auth.get_store(1).has_tokens and auth.get_store(2).has_tokens
