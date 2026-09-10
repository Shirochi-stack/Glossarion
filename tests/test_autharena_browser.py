import queue

import pytest

import autharena_browser as browser


def config(**kwargs):
    return dict({'bridge': {'url': 'http://127.0.0.1:18874', 'control_token': 'PRIVATE-CONTROL'},
                 'profile': '/not-used', 'account_id': 0, 'login': False, 'model': 'selected-model',
                 'timeout': 30, 'payload': {'prompt': 'Translate'}}, **kwargs)


def harness(monkeypatch, batches, *, connect_url=None):
    calls, opened = [], []
    events = iter(batches)

    def request(bridge, method, path, body=None, **kwargs):
        calls.append((method, path, body))
        if path == '/control/jobs':
            return {'job_id': 'request-1', 'connect_url': connect_url}
        if path.endswith('/events'):
            value = next(events)
            if isinstance(value, Exception):
                raise value
            return {'events': value, 'terminal': any(e['event'] in ('done', 'error') for e in value)}
        return {'ok': True}

    monkeypatch.setattr(browser, 'request', request)
    monkeypatch.setattr(browser.webbrowser, 'open', lambda url: opened.append(url) or True)
    return calls, opened


def test_login_uses_current_browser_without_model_or_custom_profile(monkeypatch):
    url = 'http://127.0.0.1:18874/connect#one-time-pairing'
    calls, opened = harness(monkeypatch, [[{'event': 'verified', 'logged_in': True, 'tou_accepted': True},
                                        {'event': 'done'}]], connect_url=url)
    output = []
    assert browser._control_request(config(login=True, model='', payload=None), queue.Queue(),
                                    lambda kind, **data: output.append((kind, data))) == 0
    assert opened == [url]
    assert calls[0][2]['model'] == ''
    assert 'profile' not in calls[0][2] and 'bridge' not in calls[0][2]
    assert [kind for kind, _ in output] == ['status', 'verified', 'done']
    assert 'PRIVATE-CONTROL' not in str(output)


def test_signed_in_request_streams_before_done_and_requires_parent_dispatch(monkeypatch):
    calls, opened = harness(monkeypatch, [
        [{'event': 'verified', 'logged_in': True, 'tou_accepted': True}, {'event': 'ready'}],
        [{'event': 'chunk', 'data': 'a0:"Hel"\n'}],
        [{'event': 'chunk', 'data': 'a0:"lo"\n'}, {'event': 'done'}],
    ])
    commands, output = queue.Queue(), []

    def emit(kind, **data):
        output.append((kind, data))
        if kind == 'ready':
            assert not any(body == {'command': 'dispatch'} for _, _, body in calls)
            commands.put('dispatch')
        if kind == 'chunk':
            assert 'done' not in [event for event, _ in output]

    assert browser._control_request(config(), commands, emit) == 0
    assert opened == []
    assert sum(body == {'command': 'dispatch'} for _, _, body in calls) == 1
    assert [data['data'] for kind, data in output if kind == 'chunk'] == ['a0:"Hel"\n', 'a0:"lo"\n']


def test_cancel_beats_queued_dispatch_and_closes_only_the_job(monkeypatch):
    calls, opened = harness(monkeypatch, [[{'event': 'ready'}]])
    commands = queue.Queue()

    def emit(kind, **_):
        if kind == 'ready':
            commands.put('dispatch')
            commands.put('cancel')

    with pytest.raises(RuntimeError, match='cancelled'):
        browser._control_request(config(), commands, emit)
    assert not any(body == {'command': 'dispatch'} for _, _, body in calls)
    assert calls[-1] == ('POST', '/control/jobs/request-1/command', {'command': 'cancel'})
    assert opened == []


def test_cancel_before_login_does_not_open_a_tab(monkeypatch):
    calls, opened = harness(monkeypatch, [], connect_url='http://127.0.0.1:18874/connect#nonce')
    commands = queue.Queue()
    commands.put('cancel')
    with pytest.raises(RuntimeError, match='cancelled'):
        browser._control_request(config(login=True), commands, lambda *args, **kwargs: None)
    assert calls == opened == []


def test_lost_bridge_after_dispatch_is_reported_without_replay(monkeypatch):
    calls, opened = harness(monkeypatch, [[{'event': 'ready'}], RuntimeError('Connection lost')])
    commands = queue.Queue()

    def emit(kind, **_):
        if kind == 'ready':
            commands.put('dispatch')

    with pytest.raises(RuntimeError, match='Connection lost'):
        browser._control_request(config(), commands, emit)
    assert sum(path == '/control/jobs' for _, path, _ in calls) == 1
    assert sum(body == {'command': 'dispatch'} for _, _, body in calls) == 1
    assert calls[-1][2] == {'command': 'cancel'}


def test_rejection_can_request_a_fresh_dispatch_but_not_a_new_job(monkeypatch):
    calls, _ = harness(monkeypatch, [
        [{'event': 'ready'}],
        [{'event': 'rejected'}, {'event': 'action', 'message': 'Complete the challenge'}],
        [{'event': 'ready'}], [{'event': 'done'}],
    ])
    commands, output = queue.Queue(), []

    def emit(kind, **data):
        output.append((kind, data))
        if kind == 'ready':
            commands.put('dispatch')

    assert browser._control_request(config(), commands, emit) == 0
    assert sum(path == '/control/jobs' for _, path, _ in calls) == 1
    assert sum(body == {'command': 'dispatch'} for _, _, body in calls) == 2
    assert ('status', {'message': 'Complete the challenge'}) in output


def test_offline_or_expired_account_error_keeps_rotation_metadata(monkeypatch):
    calls, opened = harness(monkeypatch, [[{'event': 'error', 'status_code': 503, 'safe_to_rotate': True,
                                          'message': 'Browser profile is not connected'}]])
    output = []
    assert browser._control_request(config(allow_interactive=False), queue.Queue(),
                                    lambda kind, **data: output.append((kind, data))) == 1
    assert opened == []
    assert output[-1][1]['safe_to_rotate'] is True
    assert not any(body == {'command': 'cancel'} for _, _, body in calls)
