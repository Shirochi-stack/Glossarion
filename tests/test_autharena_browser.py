import json
from pathlib import Path
import queue
from types import SimpleNamespace

import pytest

import autharena_browser as browser


def test_external_launch_uses_real_browser_and_isolates_profile(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(browser.subprocess, 'Popen', lambda command, **kwargs: calls.append((command, kwargs)) or 'process')
    profile = tmp_path / '4'
    (profile / 'chromium').mkdir(parents=True)
    (profile / 'chromium' / 'DevToolsActivePort').write_text('stale')
    process, port = browser._launch_browser('C:/Browser/chrome.exe', profile, visible=True)
    command, kwargs = calls[0]
    assert process == 'process'
    assert command[0] == 'C:/Browser/chrome.exe'
    assert f'--user-data-dir={profile.resolve() / "chromium"}' in command
    assert '--remote-debugging-address=127.0.0.1' in command
    assert '--remote-debugging-port=0' in command
    assert '--new-window' in command and '--headless=new' not in command
    assert not any('disable-web-security' in part or 'remote-allow-origins' in part for part in command)
    assert command[-1] == 'about:blank'
    assert kwargs['stdin'] is browser.subprocess.DEVNULL
    assert not port.exists()


def test_noninteractive_pool_does_not_open_a_login_window(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(browser.subprocess, 'Popen', lambda command, **_: calls.append(command) or None)
    browser._launch_browser('browser', tmp_path, visible=False)
    assert '--headless=new' in calls[0]
    assert '--new-window' not in calls[0]


def test_explicit_browser_path_is_used_without_shell_arguments(tmp_path, monkeypatch):
    executable = tmp_path / 'My Browser' / 'chrome.exe'
    executable.parent.mkdir()
    executable.touch()
    monkeypatch.setenv('AUTHARENA_BROWSER', f'"{executable}"')
    assert browser._find_browser() == str(executable.resolve())
    monkeypatch.setenv('AUTHARENA_BROWSER', f'{executable} --extra-flag')
    with pytest.raises(browser.BrowserConfigurationError):
        browser._find_browser()


def test_endpoint_comes_only_from_our_new_local_browser(tmp_path):
    port = tmp_path / 'DevToolsActivePort'
    port.write_text('12345\n/devtools/browser/a-b-123\n')
    process = SimpleNamespace(poll=lambda: None)
    assert browser._wait_endpoint(process, port, browser.time.monotonic() + 1, lambda: False) == 'ws://127.0.0.1:12345/devtools/browser/a-b-123'
    process = SimpleNamespace(poll=lambda: 0)
    with pytest.raises(browser.BrowserConfigurationError, match='closed before connecting'):
        browser._wait_endpoint(process, port, browser.time.monotonic() + 1, lambda: False)


class FakePage:
    def __init__(self, states):
        self.states = iter(states)
        self.evaluations = []
        self.calls = []

    def evaluate(self, expression, session):
        assert session == 'owned-session'
        self.evaluations.append(expression)
        if expression == browser._POLL_SCRIPT:
            state = next(self.states)
            if isinstance(state, Exception):
                raise state
            return state

    def call(self, method, params=None, *, session=None):
        self.calls.append((method, session))
        return {}


def state(phase=None, *events, document='1', arena=True):
    return dict(arena=arena, loaded=True, phase=phase, document=document, events=list(events))


def drive(monkeypatch, states, *, login=False, interactive=True, on_emit=None):
    now = [1.0]
    monkeypatch.setattr(browser.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(browser.time, 'sleep', lambda seconds: now.__setitem__(0, now[0] + seconds))
    page = FakePage(states)
    commands = queue.Queue()
    events = []
    preparations = []

    def emit(kind, **details):
        events.append((kind, details))
        if kind == 'ready':
            commands.put('dispatch')
        if on_emit:
            on_emit(kind, commands)

    def prepare(payload, model, timeout, rejections=0, **kwargs):
        preparations.append(kwargs)
        return 'login-probe' if kwargs['login_only'] else 'prepare-request'

    config = dict(login=login, allow_interactive=interactive, model='model', payload={'id': 'fresh'})
    result = browser._control_page(page, 'owned-session', config, prepare, commands, emit, 20)
    return result, page, preparations, events


def test_login_is_verified_automatically_without_continue_button(monkeypatch):
    result, page, preparations, events = drive(monkeypatch, [
        state(),
        state('done', {'event': 'verified', 'logged_in': True, 'tou_accepted': True}, {'event': 'done'}),
    ], login=True)
    assert result == 0
    assert preparations == [{'login_only': True, 'allow_interactive': True}]
    assert [event for event, _ in events] == ['verified', 'done']
    assert browser._DISPATCH_SCRIPT not in page.evaluations


def test_login_polls_after_native_modal_changes_without_navigation(monkeypatch):
    waiting = state('waiting', {'event': 'action', 'status_code': 401, 'message': 'Sign in'})
    result, page, preparations, events = drive(monkeypatch, [
        state(), waiting, *[state('waiting') for _ in range(21)],
        state('done', {'event': 'verified', 'logged_in': True, 'tou_accepted': True}, {'event': 'done'}),
    ], login=True)
    assert result == 0
    assert len(preparations) == 2
    assert all(item['login_only'] for item in preparations)
    assert page.calls == [('Page.bringToFront', 'owned-session')]


def test_normal_request_resumes_after_verified_sign_in_and_dispatches_once(monkeypatch):
    result, page, preparations, events = drive(monkeypatch, [
        state(), state('waiting', {'event': 'action', 'status_code': 401, 'message': 'Sign in'}),
        *[state('waiting') for _ in range(21)],
        state('done', {'event': 'verified', 'logged_in': True, 'tou_accepted': True}, {'event': 'done'}),
        state('ready', {'event': 'ready'}),
        state('done', {'event': 'chunk', 'data': 'answer'}, {'event': 'done'}),
    ])
    assert [item['login_only'] for item in preparations] == [False, True, False]
    assert page.evaluations.count(browser._DISPATCH_SCRIPT) == 1
    assert [kind for kind, _ in events].count('done') == 1
    assert result == 0


def test_live_captcha_is_not_overwritten_by_login_probes(monkeypatch):
    result, page, preparations, events = drive(monkeypatch, [
        state(), state('ready', {'event': 'ready'}),
        state('challenge', {'event': 'rejected'}, {'event': 'action', 'challenge': True, 'message': 'Complete challenge'}),
        *[state('challenge') for _ in range(30)],
        state('ready', {'event': 'ready'}), state('done', {'event': 'done'}),
    ])
    assert result == 0 and len(preparations) == 1
    assert page.evaluations.count(browser._DISPATCH_SCRIPT) == 2
    assert [kind for kind, _ in events] == ['ready', 'rejected', 'status', 'ready', 'done']


@pytest.mark.parametrize('navigation', [state(arena=False), state(document='2'),
                                         browser._ContextChanged('navigating'), state()])
def test_navigation_or_lost_state_after_dispatch_is_never_replayed(monkeypatch, navigation):
    with pytest.raises(RuntimeError, match='not retried'):
        drive(monkeypatch, [state(), state('ready', {'event': 'ready'}), navigation])


def test_navigation_before_dispatch_reprepares_safely(monkeypatch):
    result, page, preparations, events = drive(monkeypatch, [
        state(), state(arena=False), state(document='2'),
        state('ready', {'event': 'ready'}, document='2'), state('done', {'event': 'done'}, document='2'),
    ])
    assert result == 0 and len(preparations) == 2
    assert page.evaluations.count(browser._DISPATCH_SCRIPT) == 1


def test_cancel_during_login_prevents_dispatch(monkeypatch):
    def cancel(kind, commands):
        if kind == 'status':
            commands.put('cancel')

    with pytest.raises(RuntimeError, match='cancelled'):
        drive(monkeypatch, [state(), state('waiting', {'event': 'action', 'message': 'Sign in'})],
              login=True, on_emit=cancel)


def test_queued_cancel_takes_priority_over_dispatch(monkeypatch):
    commands = queue.Queue()
    commands.put('dispatch')
    commands.put('cancel')
    page = FakePage([])
    with pytest.raises(RuntimeError, match='cancelled'):
        browser._control_page(page, 'owned-session', {'model': 'model'}, None, commands, None,
                              browser.time.monotonic() + 10)
    assert page.evaluations == []


def test_parent_disconnect_cancels_before_pending_dispatch(monkeypatch):
    commands = queue.Queue()
    commands.put('dispatch')
    page = FakePage([])
    with pytest.raises(RuntimeError, match='cancelled'):
        browser._control_page(page, 'owned-session', {'model': 'model'}, None, commands, None,
                              browser.time.monotonic() + 10, cancelled=lambda: True)
    assert page.evaluations == []


def test_no_interactive_pool_failure_is_forwarded_without_showing_browser(monkeypatch):
    result, page, preparations, events = drive(monkeypatch, [
        state(), state('error', {'event': 'logged_out'},
                       {'event': 'error', 'status_code': 401, 'safe_to_rotate': True, 'message': 'Sign in'}),
    ], interactive=False)
    assert result == 1 and not page.calls
    assert events[-1][1]['safe_to_rotate'] is True
    assert preparations == [{'login_only': False, 'allow_interactive': False}]


def test_cdp_uses_only_own_session_and_ignores_unrelated_event_payloads(monkeypatch):
    import websocket
    replies = iter([json.dumps({'method': 'Runtime.consoleAPICalled', 'params': {'ignored': True}}),
                    json.dumps({'id': 1, 'result': {'result': {'value': {'phase': 'ready'}}}})])
    sent = []
    sockets = []
    connection = SimpleNamespace(settimeout=lambda _: None, send=lambda value: sent.append(json.loads(value)),
                                 recv=lambda: next(replies), close=lambda: None)
    monkeypatch.setattr(websocket, 'create_connection', lambda url, **kwargs: sockets.append((url, kwargs)) or connection)
    cdp = browser._CDP('ws://127.0.0.1:123/devtools/browser/owned', browser.time.monotonic() + 20)
    assert cdp.evaluate('status-only', 'owned-session') == {'phase': 'ready'}
    assert sent[0]['sessionId'] == 'owned-session'
    assert sockets[0][1]['suppress_origin'] is True
    assert sockets[0][1]['http_no_proxy'] == ['127.0.0.1', 'localhost']


def test_browser_close_flushes_only_the_process_started_here():
    events = []
    cdp = SimpleNamespace(call=lambda *args, **kwargs: events.append(('command', args[0])),
                          close=lambda: events.append(('connection', 'close')))
    process = SimpleNamespace(poll=lambda: None, wait=lambda **_: events.append(('process', 'wait')))
    browser._close_browser(cdp, process)
    assert events == [('command', 'Browser.close'), ('connection', 'close'), ('process', 'wait')]


def test_closed_external_window_is_reported_as_cancelled_not_retryable_transport(monkeypatch, capsys):
    process = SimpleNamespace(poll=lambda: 0)
    monkeypatch.setattr(browser, '_find_browser', lambda: '/browser/chrome')
    monkeypatch.setattr(browser, '_launch_browser', lambda *args, **kwargs: (process, Path('/profile/DevToolsActivePort')))
    monkeypatch.setattr(browser, '_wait_endpoint', lambda *args: 'ws://127.0.0.1:123/devtools/browser/owned')
    monkeypatch.setattr(browser.threading, 'Thread', lambda **kwargs: SimpleNamespace(start=lambda: None))
    responses = iter([{'targetInfos': [{'type': 'page', 'url': 'about:blank', 'targetId': 'owned'}]},
                      {'sessionId': 'owned-session'}, {}, {}])
    cdp = SimpleNamespace(call=lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(browser, '_CDP', lambda *args: cdp)

    def closed(*args):
        raise RuntimeError('Connection closed')

    monkeypatch.setattr(browser, '_control_page', closed)
    cleanup = []
    monkeypatch.setattr(browser, '_close_browser', lambda connection, proc: cleanup.append((connection, proc)))
    assert browser.run_browser_helper({'timeout': 10, 'profile': '/profile', 'model': 'model'}, prepare_script=None) == 1
    event = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert event['event'] == 'error' and event['error_type'] == 'cancelled'
    assert 'cancelled' in event['message']
    assert cleanup == [(cdp, process)]
