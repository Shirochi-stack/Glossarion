import base64
import hashlib
import time
from urllib.parse import parse_qs, urlparse, urlencode

import pytest
import requests
import authnan_auth as auth


@pytest.fixture(autouse=True)
def isolated_auth(monkeypatch, tmp_path):
    monkeypatch.setattr(auth, '_DEFAULT_TOKEN_DIR', str(tmp_path))
    monkeypatch.setattr(auth, '_DEFAULT_TOKEN_FILE', str(tmp_path / 'authnan_tokens.json'))
    monkeypatch.delenv('AUTHNAN_TOKEN_FILE', raising=False)
    monkeypatch.delenv('TRANSLATION_CANCELLED', raising=False)
    auth._stores.clear()
    auth._cancel_event.clear()
    auth._pool_rotation_cursor = 0
    yield
    auth.cancel_stream()
    auth._cancel_event.clear()


def send_callback(session, **query):
    with requests.Session() as client:
        client.trust_env = False
        return client.get(session.redirect_uri + '?' + urlencode(query), timeout=5)


def test_pkce_and_dynamic_loopback():
    first = auth.begin_oauth(timeout=5)
    second = auth.begin_oauth(timeout=5)
    try:
        query = parse_qs(urlparse(first.auth_url).query)
        digest = base64.urlsafe_b64encode(hashlib.sha256(first.code_verifier.encode()).digest()).rstrip(b'=').decode()
        assert query['code_challenge'] == [digest]
        assert query['code_challenge_method'] == ['S256']
        assert query['scope'] == ['api.use models.read']
        assert query['callback_url'] == [first.redirect_uri]
        assert urlparse(first.redirect_uri).hostname == '127.0.0.1'
        assert first.port != second.port
        assert first.state != second.state
    finally:
        first.close()
        second.close()


def test_full_browser_handoff_and_encrypted_restart(monkeypatch):
    calls = []
    def exchange(url, **kwargs):
        calls.append((url, kwargs))
        return type('Response', (), {'status_code': 200, 'json': lambda self: {
            'key': 'sk-nano-private-test', 'user_id': 'user-2', 'scope': 'api.use models.read'}, 'close': lambda self: None})()
    monkeypatch.setattr(auth.requests, 'post', exchange)
    store = auth.get_store(2)
    captured = []
    def open_browser(url):
        captured.append(url)
        query = parse_qs(urlparse(url).query)
        with requests.Session() as browser:
            browser.trust_env = False
            response = browser.get(query['callback_url'][0], params={'code': 'approved-code', 'state': query['state'][0]}, timeout=5)
            assert response.status_code == 200
    tokens = auth.run_oauth_flow(store=store, open_browser=open_browser, timeout=5)
    assert tokens['access_token'] == 'sk-nano-private-test'
    assert calls[0][0] == 'https://nano-gpt.com/api/v1/auth/keys'
    assert calls[0][1]['json']['grant_type'] == 'authorization_code'
    assert calls[0][1]['json']['code'] == 'approved-code'
    assert calls[0][1]['json']['code_verifier']
    assert not auth._sessions
    assert not store.load_pending_oauth()
    data = open(store._token_file, 'rb').read()
    assert data.startswith(b'GLSE1:') and b'sk-nano-private-test' not in data
    restarted = auth.AuthNanTokenStore(store._token_file, account_id=2)
    assert restarted.get_valid_access_token(auto_login=False) == tokens['access_token']
    assert restarted.account_info['user_id'] == 'user-2'
    restarted.clear_tokens()
    assert not restarted.has_tokens


@pytest.mark.parametrize('query', [{'code': 'code', 'state': 'wrong'}, {'error': 'access_denied'}, {'state': 'correct'}])
def test_callback_rejection_never_exchanges(monkeypatch, query):
    store = auth.get_store()
    session = auth.begin_oauth(store=store, timeout=5)
    query = {**query, 'state': session.state if query.get('state') != 'wrong' else 'wrong'}
    monkeypatch.setattr(auth, 'exchange_code_for_tokens', lambda *_: pytest.fail('rejected callback exchanged'))
    response = send_callback(session, **query)
    assert response.status_code == 400
    with pytest.raises(RuntimeError):
        auth.complete_from_redirect(session)
    assert session.closed and not store.has_tokens and not store.load_pending_oauth()


def test_pasted_redirect_requires_state_and_closes(monkeypatch):
    session = auth.begin_oauth(timeout=5)
    monkeypatch.setattr(auth, 'exchange_code_for_tokens', lambda *_: pytest.fail('missing state exchanged'))
    with pytest.raises(RuntimeError, match='state mismatch'):
        auth.complete_from_redirect(session, 'code=abc')
    assert session.closed


def test_timeout_and_cancel_release_listeners():
    session = auth.begin_oauth(timeout=0.1)
    deadline = time.monotonic() + 3
    while not session.closed and time.monotonic() < deadline:
        time.sleep(0.01)
    session.close()
    assert session.closed and not session.server_running
    other = auth.begin_oauth(timeout=5)
    auth.cancel_stream()
    assert other.closed and not auth._sessions


@pytest.mark.parametrize('cancel', [False, True])
def test_run_timeout_or_cancel_cleans_pending(cancel):
    store = auth.get_store()
    with pytest.raises(RuntimeError, match='cancelled' if cancel else 'timed out'):
        auth.run_oauth_flow(store=store, timeout=0.1,
            open_browser=lambda _: auth.cancel_stream() if cancel else None)
    assert not store.load_pending_oauth() and not auth._sessions


@pytest.mark.parametrize('payload', [None, {}, {'key': ''}, {'access_token': 'bad key'}])
def test_malformed_exchange(payload, monkeypatch):
    response = type('Response', (), {'status_code': 200, 'json': lambda self: payload, 'close': lambda self: None})()
    monkeypatch.setattr(auth.requests, 'post', lambda *a, **kw: response)
    with pytest.raises(RuntimeError):
        auth.exchange_code_for_tokens('code', 'verifier')


def test_rotation_deduplicates_identity_and_skips_expired():
    for slot, user in [(0, 'same'), (1, 'same'), (2, 'different')]:
        auth.get_store(slot).save_tokens({'key': f'key-{slot}', 'user_id': user})
    auth.get_store(3).save_tokens({'key': 'expired', 'expires_at': time.time() - 10})
    assert [slot for slot, _ in auth.get_rotating_account_pool()] == [0, 2]
    assert [slot for slot, _ in auth.get_rotating_account_pool()] == [2, 0]


def test_default_override_and_numbered_isolation(monkeypatch, tmp_path):
    monkeypatch.setenv('AUTHNAN_TOKEN_FILE', str(tmp_path / 'custom.json'))
    assert auth.get_store()._token_file.endswith('custom.json')
    assert auth.get_store(1)._token_file.endswith('authnan_tokens_1.json')
    with pytest.raises(ValueError):
        auth.get_store(10000)


def test_concurrent_rejection_does_not_clear_replacement(monkeypatch):
    store = auth.get_store()
    store.save_tokens({'key': 'replacement'})
    monkeypatch.setattr(auth, 'run_oauth_flow', lambda **_: pytest.fail('redundant login'))
    assert store.recover_from_unauthorized('old-key') == 'replacement'


def test_browser_login_does_not_lock_identity_reads(monkeypatch):
    import threading
    store = auth.get_store(2)
    def login(store):
        read = threading.Event()
        thread = threading.Thread(target=lambda: (store.load_tokens(), read.set()), daemon=True)
        thread.start()
        assert read.wait(1), 'browser login held the credential lock'
        tokens = {'access_token': 'new-key', 'user_id': 'new-user'}
        store.save_tokens(tokens)
        thread.join(1)
        return tokens
    monkeypatch.setattr(auth, 'run_oauth_flow', login)
    assert store.get_valid_access_token() == 'new-key'


def test_catalog_subscription_text_and_separate_media(monkeypatch):
    import model_options
    urls = []
    def get(url, headers, timeout):
        urls.append(url)
        assert headers['Authorization'] == 'Bearer saved-key'
        return {'data': [{'id': 'text' if '/subscription/' in url else 'image' if 'image-models' in url else 'video'}]}
    monkeypatch.setattr(model_options, '_http_get_json', get)
    assert auth.fetch_available_models('saved-key') == ['text', 'image', 'video']
    assert urls == ['https://nano-gpt.com/api/subscription/v1/models?detailed=true',
                    'https://nano-gpt.com/api/v1/image-models', 'https://nano-gpt.com/api/v1/video-models']


def test_expiry_and_identity_metadata_survive_restart():
    store = auth.get_store(4)
    store.save_tokens({'key': 'key-4', 'user_id': 'user-4', 'email': 'user@example.com',
                       'scope': 'api.use models.read', 'expires_in': 120, 'name': 'Example'})
    restarted = auth.AuthNanTokenStore(store._token_file, 4)
    assert restarted.has_tokens
    assert restarted.load_tokens()['expires_at'] > time.time() + 100
    assert restarted.load_tokens()['name'] == 'Example'
    assert restarted.account_info == {'user_id': 'user-4', 'email': 'user@example.com', 'scope': 'api.use models.read'}


def test_encryption_failure_does_not_write_plaintext(monkeypatch):
    import token_encryption
    store = auth.get_store()
    monkeypatch.setattr(token_encryption, 'save_encrypted_tokens', lambda *a: (_ for _ in ()).throw(OSError('cannot encrypt')))
    with pytest.raises(OSError): store.save_tokens({'key': 'private'})
    assert not store.has_tokens and not __import__('os').path.exists(store._token_file)


def test_cancel_during_exchange_does_not_save_key(monkeypatch):
    store = auth.get_store()
    session = auth.begin_oauth(store=store)
    def exchange(*_):
        auth.cancel_stream()
        return {'access_token': 'never-save'}
    monkeypatch.setattr(auth, 'exchange_code_for_tokens', exchange)
    with pytest.raises(RuntimeError, match='cancelled'):
        auth.complete_from_redirect(session, 'code=approved&state=' + session.state)
    assert session.closed and not store.has_tokens and not store.load_pending_oauth()


def test_pending_redirect_can_complete_after_restart(monkeypatch):
    store = auth.get_store()
    session = auth.begin_oauth(store=store)
    redirect = session.redirect_uri + '?' + urlencode({'code': 'approved', 'state': session.state})
    session.close()
    restarted = auth.AuthNanTokenStore(store._token_file)
    monkeypatch.setattr(auth, 'exchange_code_for_tokens', lambda *a: {'access_token': 'restarted-key'})
    assert auth.complete_from_redirect(redirect_url_or_code=redirect, store=restarted)['access_token'] == 'restarted-key'
    assert not restarted.load_pending_oauth()


@pytest.mark.parametrize('existing_slot', [0, 7])
def test_add_account_rejects_duplicate_identity_before_saving(monkeypatch, existing_slot):
    auth.get_store(existing_slot).save_tokens({'key': 'existing-key', 'user_id': 'same-user'})
    store = auth.get_store(2)
    session = auth.begin_oauth(store=store, new_account=True)
    monkeypatch.setattr(auth, 'exchange_code_for_tokens', lambda *a: {'access_token': 'duplicate-key', 'user_id': 'same-user'})
    with pytest.raises(RuntimeError, match=f'already saved in slot #{existing_slot}'):
        auth.complete_from_redirect(session, 'code=approved&state=' + session.state)
    assert not store.has_tokens and not store.load_pending_oauth() and session.closed
    assert auth.get_store(existing_slot).get_valid_access_token(False) == 'existing-key'


def test_add_account_accepts_new_identity(monkeypatch):
    auth.get_store().save_tokens({'key': 'existing-key', 'user_id': 'first-user'})
    store = auth.get_store(2)
    session = auth.begin_oauth(store=store, new_account=True)
    monkeypatch.setattr(auth, 'exchange_code_for_tokens', lambda *a: {'access_token': 'new-key', 'user_id': 'second-user'})
    auth.complete_from_redirect(session, 'code=approved&state=' + session.state)
    assert store.account_info['user_id'] == 'second-user'


@pytest.mark.parametrize('firefox', [False, True])
def test_fresh_browser_uses_separate_empty_profile_and_closes(monkeypatch, tmp_path, firefox):
    binary = str(tmp_path / ('firefox.exe' if firefox else 'chrome.exe'))
    monkeypatch.setattr(auth, '_fresh_browser_binary', lambda: binary)
    processes = []
    class Process:
        pid = 999999
        def __init__(self, args, **kwargs): self.args, self.closed = args, False; processes.append(self)
        def poll(self): return 0 if self.closed else None
        def terminate(self): self.closed = True
        def wait(self, timeout): return 0
    monkeypatch.setattr(auth.subprocess, 'Popen', Process)
    monkeypatch.setattr(auth.subprocess, 'run', lambda args, **kwargs: None)
    with auth._fresh_account_browser('https://nano-gpt.com/auth?state=test') as first:
        arg = first.args[first.args.index('-profile') + 1] if firefox else next(a.split('=', 1)[1] for a in first.args if a.startswith('--user-data-dir='))
        assert __import__('pathlib').Path(arg).is_dir()
        assert list(__import__('pathlib').Path(arg).iterdir()) == []
        assert first.args[-1] == 'https://nano-gpt.com/auth?state=test'
        with auth._fresh_account_browser('https://nano-gpt.com/auth?state=next') as second:
            assert first.args != second.args
    assert all(p.closed for p in processes)
    assert not __import__('pathlib').Path(arg).exists()
    assert not auth._fresh_browsers


def test_cancel_closes_fresh_browser_synchronously(monkeypatch):
    closed = []
    monkeypatch.setattr(auth, '_fresh_browser_binary', lambda: 'chrome.exe')
    class Process:
        pid = 999999
        def poll(self): return 0 if closed else None
        def terminate(self): closed.append(True)
        def wait(self, timeout): return 0
    monkeypatch.setattr(auth.subprocess, 'Popen', lambda *a, **kw: Process())
    monkeypatch.setattr(auth.subprocess, 'run', lambda *a, **kw: None)
    with auth._fresh_account_browser('https://nano-gpt.com/auth'):
        directory, _ = next(iter(auth._fresh_browsers.values()))
        auth.cancel_stream()
        assert closed == [True] and not directory.exists() and not auth._fresh_browsers
    assert closed == [True]


@pytest.mark.parametrize('outcome', ['success', 'timeout', 'cancel', 'closed'])
def test_add_account_fresh_browser_flow_cleanup(monkeypatch, outcome):
    from contextlib import contextmanager
    store = auth.get_store(2)
    monkeypatch.setattr(auth.webbrowser, 'open', lambda *_: pytest.fail('reused normal browser session'))
    monkeypatch.setattr(auth, 'exchange_code_for_tokens', lambda *a: {'access_token': 'fresh-key', 'user_id': 'new-user'})
    closed = []
    @contextmanager
    def fresh(url):
        query = parse_qs(urlparse(url).query)
        if outcome == 'success':
            with requests.Session() as browser:
                browser.trust_env = False
                assert browser.get(query['callback_url'][0], params={'code': 'code', 'state': query['state'][0]}, timeout=5).status_code == 200
        elif outcome == 'cancel':
            auth.cancel_stream()
        try:
            yield type('Process', (), {'poll': lambda self: 0 if outcome == 'closed' else None})()
        finally:
            closed.append(True)
    monkeypatch.setattr(auth, '_fresh_account_browser', fresh)
    if outcome == 'success':
        assert auth.run_oauth_flow(store=store, new_account=True, timeout=0.2)['user_id'] == 'new-user'
    else:
        with pytest.raises(RuntimeError): auth.run_oauth_flow(store=store, new_account=True, timeout=0.1)
    assert closed == [True] and not auth._sessions and not store.load_pending_oauth()


def test_new_account_browser_unavailable_never_reuses_session(monkeypatch):
    store = auth.get_store(2)
    monkeypatch.setattr(auth, '_fresh_browser_binary', lambda: (_ for _ in ()).throw(RuntimeError('No fresh browser')))
    monkeypatch.setattr(auth.webbrowser, 'open', lambda *_: pytest.fail('reused logged-in browser'))
    with pytest.raises(RuntimeError, match='No fresh browser'):
        auth.run_oauth_flow(store=store, new_account=True)
    assert not auth._sessions and not store.has_tokens and not store.load_pending_oauth()
