import copy
import json
from types import SimpleNamespace

import pytest

import unified_api_client as api
from unified_api_client import UnifiedClient, UnifiedClientError
from reasoning_compatibility import (
    ReasoningEffortRejected, normalize_none_effort, supported_reasoning_efforts,
    repair_reasoning_effort, call_with_reasoning_retry,
)


# The managed Ollama route uses native request options instead of OpenAI token
# parameter names. Keep these transport and lifecycle checks alongside the
# other provider parameter compatibility tests.


def test_ollamapull_shutdown_targets_only_verified_local_ollama(monkeypatch):
    import ollamapull
    import psutil

    stopped = []

    class Process:
        pid = 12345

        def name(self):
            return "ollama.exe"

        def parent(self):
            return None

        def terminate(self):
            stopped.append(self.pid)

    connections = [
        SimpleNamespace(laddr=SimpleNamespace(ip="127.0.0.1", port=11434),
                        status=psutil.CONN_LISTEN, pid=12345),
        SimpleNamespace(laddr=SimpleNamespace(ip="127.0.0.1", port=1234),
                        status=psutil.CONN_LISTEN, pid=99999),
    ]
    monkeypatch.setattr(psutil, "net_connections", lambda **_: connections)
    monkeypatch.setattr(psutil, "Process", lambda pid: Process())
    monkeypatch.setattr(psutil, "wait_procs", lambda processes, **_: (processes, []))
    monkeypatch.setattr(ollamapull, "_managed_server_process", None)

    assert ollamapull.shutdown_ollama() is True
    assert stopped == [12345]


def test_ollamapull_shutdown_refuses_unidentified_server(monkeypatch):
    import ollamapull
    import psutil

    monkeypatch.setattr(psutil, "net_connections", lambda **_: [])
    monkeypatch.setattr(ollamapull, "_managed_server_process", None)
    monkeypatch.setattr(ollamapull, "_server_version", lambda: "1.0")
    with pytest.raises(ollamapull.OllamaPullError, match="could not be identified"):
        ollamapull.shutdown_ollama()

def test_ollamapull_route_wins_over_custom_and_individual_endpoints(monkeypatch):
    monkeypatch.setenv('CUSTOM_OPENAI_PREFIX_ROUTES', json.dumps([
        {'prefix': 'ollamapull/', 'routing': 'http://127.0.0.1:9999/v1'}]))
    assert UnifiedClient._provider_from_model_name('ollamapull/gemma3') == 'ollamapull'
    assert not UnifiedClient._model_needs_api_key('ollamapull/gemma3')
    client = bare_client('ollamapull/gemma3')
    client._get_thread_local_client = lambda: SimpleNamespace(
        use_individual_endpoint=True, azure_endpoint='https://other.example/v1',
        client_type='openai')
    assert client._get_actual_provider() == 'ollamapull'
    client._setup_client()
    assert client.client_type == 'ollamapull'
    client._get_custom_prefix_route_for_model = lambda _model: (_ for _ in ()).throw(
        AssertionError('custom endpoint should not be considered'))
    client._apply_custom_endpoint_if_needed()
    client._apply_individual_key_endpoint_if_needed()
    assert client.client_type == 'ollamapull'


@pytest.mark.parametrize('prefix,provider,base_url', [
    ('ollama/', 'ollama', 'http://localhost:11434/v1'),
    ('lmstudio/', 'lmstudio', 'http://localhost:1234/v1'),
])
def test_builtin_local_routes_ignore_custom_and_individual_endpoints(
        monkeypatch, prefix, provider, base_url):
    model = prefix + 'org/model:small'
    monkeypatch.setenv('CUSTOM_OPENAI_PREFIX_ROUTES', json.dumps([
        {'prefix': prefix, 'routing': 'https://other.example/v1'}]))
    monkeypatch.setenv('USE_CUSTOM_OPENAI_ENDPOINT', '1')
    monkeypatch.setenv('OPENAI_CUSTOM_BASE_URL', 'https://global.example/v1')
    assert UnifiedClient._provider_from_model_name(model) == provider
    assert not UnifiedClient._model_needs_api_key(model)

    client = bare_client(model)
    client._get_thread_local_client = lambda: SimpleNamespace(
        use_individual_endpoint=True, azure_endpoint='https://individual.example/v1',
        client_type='openai')
    client.current_key_use_individual_endpoint = True
    client.current_key_azure_endpoint = 'https://individual.example/v1'
    assert client._get_actual_provider() == provider
    client._setup_client()
    assert client.client_type == provider
    client._apply_custom_endpoint_if_needed()
    client._apply_individual_key_endpoint_if_needed()
    assert client.client_type == provider

    captured = {}
    client._send_openai_compatible = lambda **kwargs: captured.update(kwargs)
    client._send_openai_provider_router(
        [{'role': 'user', 'content': 'Hi'}], 0.2, 128, 'test')
    assert captured['base_url'] == base_url
    assert captured['provider'] == provider


@pytest.mark.parametrize('model,provider,url,expected_model', [
    ('ollama/gemma3:4b', 'ollama', 'http://localhost:11434/v1/chat/completions', 'gemma3:4b'),
    ('ollama/qwen2.5-7b-instruct', 'ollama', 'http://localhost:11434/v1/chat/completions', 'qwen2.5-7b-instruct'),
    ('lmstudio/org/model:small', 'lmstudio', 'http://localhost:1234/v1/chat/completions', 'org/model:small'),
])
def test_builtin_local_routes_strip_prefix_for_http_request(
        monkeypatch, tmp_path, model, provider, url, expected_model):
    monkeypatch.setattr(api, 'openai', None)
    monkeypatch.setenv('USE_CUSTOM_OPENAI_ENDPOINT', '1')
    monkeypatch.setenv('OPENAI_CUSTOM_BASE_URL', 'https://global.example/v1')
    monkeypatch.setenv('ENABLE_GPT_THINKING', '0')
    client = UnifiedClient('cloud-secret', model, str(tmp_path))
    monkeypatch.setattr(client, '_get_max_retries', lambda: 1)
    monkeypatch.setattr(client, '_get_send_interval', lambda: 0)
    monkeypatch.setattr(client, '_is_stop_requested', lambda: False)
    monkeypatch.setattr(client, '_save_response', lambda *args, **kwargs: None)
    monkeypatch.setattr(client, '_should_show_api_lifecycle_logs', lambda: False)
    captured = {}

    def request(**kwargs):
        captured.update(kwargs)
        payload = {'choices': [{'message': {'content': 'OK'}, 'finish_reason': 'stop'}]}
        return SimpleNamespace(headers={'content-type': 'application/json'}, json=lambda: payload)

    monkeypatch.setattr(client, '_http_request_with_retries', request)
    response = client._send_openai_provider_router(
        [{'role': 'user', 'content': 'Hi'}], 0.2, 128, 'test')
    assert response.content == 'OK'
    assert captured['url'] == url
    assert captured['json']['model'] == expected_model
    assert captured['provider_name'] == provider
    assert captured['headers']['Authorization'] == 'Bearer dummy-key-for-local-llm'


def test_ollamapull_native_chat_strips_prefix_and_applies_model_settings(monkeypatch):
    import ollamapull

    monkeypatch.setenv('OLLAMA_SETTINGS_JSON', json.dumps({
        'auto_update': False,
        'models': {'qwen3:8b': {
            'options': {'num_ctx': 8192, 'draft_num_predict': 4, 'num_predict': 300},
            'think': 'high', 'keep_alive': '10m', 'format': 'json',
            'request': {'logprobs': True},
        }},
    }))
    captured = {}

    class Response:
        ok = True
        def iter_lines(self):
            yield b'{"message":{"content":"Hello "},"done":false}'
            yield b'{"message":{"content":"world"},"done":false}'
            yield b'{"done":true,"done_reason":"length","prompt_eval_count":5,"eval_count":2}'
        def close(self): pass

    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(ollamapull.requests, 'post', post)
    result = ollamapull.chat('ollamapull/qwen3:8b', [{'role': 'user', 'content': 'Hi'}],
                             temperature=0.3, max_tokens=200, stream=True)
    payload = captured['json']
    assert captured['url'] == 'http://127.0.0.1:11434/api/chat'
    assert payload['model'] == 'qwen3:8b'
    assert payload['options'] == {
        'temperature': 0.3, 'num_predict': 300, 'num_ctx': 8192,
        'draft_num_predict': 4,
    }
    assert (payload['think'], payload['keep_alive'], payload['format'], payload['logprobs']) == (
        'high', '10m', 'json', True)
    assert result['content'] == 'Hello world'
    assert result['finish_reason'] == 'length'
    assert result['usage']['total_tokens'] == 7


def test_ollamapull_native_stream_logs_reasoning_separately(monkeypatch, capsys):
    import ollamapull

    monkeypatch.delenv('OLLAMA_SETTINGS_JSON', raising=False)
    progress = []

    class Response:
        ok = True
        def iter_lines(self):
            yield b'{"message":{"thinking":"I need to reason first.\\n"},"done":false}'
            yield b'{"message":{"thinking":"The answer is ready."},"done":false}'
            yield b'{"message":{"content":"Final answer"},"done":false}'
            yield b'{"done":true,"eval_count":2,"prompt_eval_count":3}'
        def close(self): pass

    monkeypatch.setattr(ollamapull.requests, 'post', lambda *a, **k: Response())
    result = ollamapull.chat('ollamapull/test', [{'role': 'user', 'content': 'Hi'}],
                             log_stream=True, log_thinking=True,
                             progress=progress.append)
    output = capsys.readouterr().out
    assert '🧠 [Ollama] Thinking...' in output
    assert '    I need to reason first.' in output
    assert '    The answer is ready.' in output
    assert '📡 [Ollama] Text streaming...' in output
    assert 'Final answer' in output
    assert result['content'] == 'Final answer'
    assert 'I need to reason' not in result['content']
    assert progress[0].startswith('Sending request for test;')
    assert any('generating reasoning' in message for message in progress)
    assert any('First text token received' in message for message in progress)
    assert progress[-1] == 'Ollama stream complete (2 generated tokens).'


def test_ollamapull_native_stream_hides_reasoning_when_disabled(monkeypatch, capsys):
    import ollamapull

    monkeypatch.delenv('OLLAMA_SETTINGS_JSON', raising=False)

    class Response:
        ok = True
        def iter_lines(self):
            yield b'{"message":{"thinking":"private thought"},"done":false}'
            yield b'{"message":{"content":"public result"},"done":false}'
            yield b'{"done":true}'
        def close(self): pass

    monkeypatch.setattr(ollamapull.requests, 'post', lambda *a, **k: Response())
    result = ollamapull.chat('ollamapull/test', [], log_stream=True,
                             log_thinking=False)
    output = capsys.readouterr().out
    assert 'private thought' not in output
    assert '🧠 [Ollama]' not in output
    assert 'public result' in output
    assert result['content'] == 'public result'


def test_ollamapull_status_distinguishes_installed_from_loaded(monkeypatch):
    import ollamapull

    monkeypatch.setattr(ollamapull, '_ollama_executable', lambda: 'ollama')
    monkeypatch.setattr(ollamapull, '_binary_version', lambda: '0.1.0')
    monkeypatch.setattr(ollamapull, '_server_version', lambda: '0.1.0')
    monkeypatch.setattr(ollamapull, 'latest_version', lambda: '0.1.0')
    monkeypatch.setattr(ollamapull, 'get_model_details', lambda *_: {})
    monkeypatch.setattr(ollamapull, '_models', lambda: [{'name': 'test:latest'}])
    monkeypatch.setattr(ollamapull, '_api_get',
                        lambda path, **_: {'models': []} if path == '/api/ps' else {})
    installed_only = ollamapull.get_status('ollamapull/test')
    assert installed_only['model_installed'] is True
    assert installed_only['model_loaded'] is False

    monkeypatch.setattr(ollamapull, '_api_get',
                        lambda path, **_: {'models': [{'model': 'test:latest'}]})
    loaded = ollamapull.get_status('ollamapull/test')
    assert loaded['model_loaded'] is True


def test_ollamapull_unified_handler_returns_normalized_response(monkeypatch):
    import ollamapull

    called = []
    request_flags = {}
    monkeypatch.setattr(ollamapull, 'ensure_ready', lambda model, **kwargs: called.append(model))
    def fake_chat(model, messages, **kwargs):
        request_flags.update(kwargs)
        return {
            'content': 'translated', 'finish_reason': 'stop',
            'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3},
            'raw_response': {'done': True},
        }
    monkeypatch.setattr(ollamapull, 'chat', fake_chat)
    monkeypatch.setenv('ENABLE_STREAMING', '1')
    monkeypatch.setenv('STREAM_THINKING_LOGS', '1')
    monkeypatch.setenv('LOG_STREAM_CHUNKS', '1')
    monkeypatch.setenv('BATCH_TRANSLATION', '0')
    client = bare_client('ollamapull/gemma3')
    client.client_type = 'ollamapull'
    response = client._send_ollamapull([{'role': 'user', 'content': 'Translate'}], 0.2, 128, None)
    assert called == ['ollamapull/gemma3']
    assert request_flags['stream'] is True
    assert request_flags['log_stream'] is True
    assert request_flags['log_thinking'] is True
    assert callable(request_flags['progress'])
    assert response.content == 'translated'
    assert response.finish_reason == 'stop'
    assert response.usage['total_tokens'] == 3


def test_ollamapull_graceful_stop_preserves_active_chat_but_hard_stop_cancels(monkeypatch):
    import ollamapull

    monkeypatch.setenv('GRACEFUL_STOP', '0')
    monkeypatch.setenv('GRACEFUL_STOP_COMPLETED', '0')
    monkeypatch.delenv('TRANSLATION_CANCELLED', raising=False)
    monkeypatch.setattr(ollamapull, 'ensure_ready', lambda _model, **_kwargs: None)
    client = bare_client('ollamapull/gemma3')
    client.client_type = 'ollamapull'
    client._is_stop_requested = lambda: True  # GUI callback after Stop is clicked

    def fake_chat(_model, _messages, **kwargs):
        monkeypatch.setenv('GRACEFUL_STOP', '1')
        assert kwargs['should_stop']() is False
        return {'content': 'done', 'finish_reason': 'stop', 'usage': {}, 'raw_response': {}}

    # A stop before chat should still prevent starting a new request.
    monkeypatch.setattr(ollamapull, 'chat', fake_chat)
    monkeypatch.setenv('GRACEFUL_STOP', '1')
    with pytest.raises(Exception, match='stopped before chat'):
        client._send_ollamapull([], None, None, None)

    monkeypatch.setenv('GRACEFUL_STOP', '0')
    client._is_stop_requested = lambda: False
    assert client._send_ollamapull([], None, None, None).content == 'done'

    def hard_stopped_chat(_model, _messages, **kwargs):
        monkeypatch.setenv('TRANSLATION_CANCELLED', '1')
        assert kwargs['should_stop']() is True
        raise ollamapull.OllamaPullCancelled('hard stop')

    monkeypatch.setenv('GRACEFUL_STOP', '0')
    monkeypatch.setattr(ollamapull, 'chat', hard_stopped_chat)
    with pytest.raises(Exception, match='hard stop'):
        client._send_ollamapull([], None, None, None)


def test_ollamapull_stagger_log_does_not_claim_chat_started(monkeypatch):
    monkeypatch.setenv('SEND_INTERVAL_SECONDS', '0.01')
    monkeypatch.setattr(UnifiedClient, '_last_api_call_start_by_scope', {}, raising=False)
    client = bare_client('ollamapull/qwen3')
    client._get_thread_local_client = lambda: SimpleNamespace()
    client._get_api_stagger_scope = lambda: 'ollamapull-test'
    client._should_show_api_lifecycle_logs = lambda: True
    messages = []
    client._debug_log = messages.append

    client._apply_api_call_stagger()

    assert not any('API call in progress' in message for message in messages)
    assert not any('Sending API call' in message for message in messages)


def test_ollamapull_advanced_request_cannot_replace_routing(monkeypatch):
    import ollamapull

    monkeypatch.setenv('OLLAMA_SETTINGS_JSON', json.dumps({
        'models': {'test': {'request': {'model': 'other'}}}}))
    with pytest.raises(ollamapull.OllamaPullError, match='cannot override'):
        ollamapull.chat('ollamapull/test', [], stream=False)


def test_ollamapull_missing_install_start_and_pull(monkeypatch):
    import ollamapull
    import requests

    calls = []
    monkeypatch.setattr(ollamapull, '_ollama_executable', lambda: None)
    monkeypatch.setattr(ollamapull, '_api_get', lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError()))
    monkeypatch.setattr(ollamapull, '_run_installer', lambda *a: calls.append('install'))
    monkeypatch.setattr(ollamapull, '_start_server', lambda *a: calls.append('start'))
    monkeypatch.setattr(ollamapull, '_models', lambda: [])
    monkeypatch.setattr(ollamapull, 'pull_model', lambda *a, **k: calls.append('pull'))
    ollamapull.ensure_ready('ollamapull/gemma3')
    assert calls == ['install', 'start', 'pull']


def test_ollamapull_accepts_installer_started_server_after_duplicate_serve_exits(monkeypatch):
    import ollamapull
    import requests

    probes = []
    messages = []
    duplicate = SimpleNamespace(poll=lambda: 1, returncode=1)
    monkeypatch.setattr(ollamapull, '_managed_server_process', None)
    monkeypatch.setattr(ollamapull, '_ollama_executable', lambda: 'ollama')
    monkeypatch.setattr(ollamapull.subprocess, 'Popen', lambda *a, **k: duplicate)
    monkeypatch.setattr(ollamapull.time, 'sleep', lambda _seconds: None)

    def probe(*_args, **_kwargs):
        probes.append(True)
        if len(probes) < 3:
            raise requests.ConnectionError('server still starting')
        return {'version': '0.34.4'}

    monkeypatch.setattr(ollamapull, '_api_get', probe)
    ollamapull._start_server(messages.append, None)

    assert len(probes) == 3
    assert ollamapull._managed_server_process is None
    assert any('checking whether its app server is starting' in message for message in messages)


def test_ollamapull_reports_exited_server_when_no_server_becomes_ready(monkeypatch):
    import ollamapull
    import requests

    now = [0]
    probes = []
    messages = []
    duplicate = SimpleNamespace(poll=lambda: 1, returncode=1)
    monkeypatch.setattr(ollamapull, '_managed_server_process', None)
    monkeypatch.setattr(ollamapull, '_ollama_executable', lambda: 'ollama')
    monkeypatch.setattr(ollamapull.subprocess, 'Popen', lambda *a, **k: duplicate)
    monkeypatch.setattr(ollamapull.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(ollamapull.time, 'sleep', lambda _seconds: now.__setitem__(0, now[0] + 5))

    def unavailable(*_args, **_kwargs):
        probes.append(True)
        raise requests.ConnectionError('server unavailable')

    monkeypatch.setattr(ollamapull, '_api_get', unavailable)
    with pytest.raises(ollamapull.OllamaPullError,
                       match=r'Ollama server exited \(code 1\); no local server became available'):
        ollamapull._start_server(messages.append, None)

    assert len(probes) > 2  # Keep checking for a server after the child exits.
    assert ollamapull._managed_server_process is None
    assert sum('checking whether its app server is starting' in message for message in messages) == 1


def test_ollamapull_server_start_can_be_cancelled_while_waiting_for_app(monkeypatch):
    import ollamapull
    import requests

    stopped = [False]
    duplicate = SimpleNamespace(poll=lambda: 1, returncode=1)
    monkeypatch.setattr(ollamapull, '_managed_server_process', None)
    monkeypatch.setattr(ollamapull, '_ollama_executable', lambda: 'ollama')
    monkeypatch.setattr(ollamapull.subprocess, 'Popen', lambda *a, **k: duplicate)
    monkeypatch.setattr(ollamapull.time, 'sleep', lambda _seconds: None)
    monkeypatch.setattr(ollamapull, '_api_get', lambda *_a, **_k: (_ for _ in ()).throw(
        requests.ConnectionError('server still starting')))

    def progress(message):
        if 'checking whether its app server is starting' in message:
            stopped[0] = True

    with pytest.raises(ollamapull.OllamaPullCancelled):
        ollamapull._start_server(progress, lambda: stopped[0])


def test_ollamapull_auto_update_attempted_once_and_toggle(monkeypatch):
    import ollamapull

    calls = []
    ollamapull._attempted_auto_updates.clear()
    monkeypatch.setattr(ollamapull, '_ollama_executable', lambda: '/usr/bin/ollama')
    monkeypatch.setattr(ollamapull, '_installed_version', lambda: '0.1.0')
    monkeypatch.setattr(ollamapull, 'latest_version', lambda: '0.2.0')
    monkeypatch.setattr(ollamapull, '_run_installer', lambda *a: calls.append('update'))
    monkeypatch.setattr(ollamapull, '_start_server', lambda *a: None)
    ollamapull.ensure_ready()
    ollamapull.ensure_ready()
    assert calls == ['update']
    ollamapull._attempted_auto_updates.clear()
    monkeypatch.setenv('OLLAMA_SETTINGS_JSON', '{"auto_update": false, "models": {}}')
    ollamapull.ensure_ready()
    assert calls == ['update']


def test_ollamapull_pull_progress_and_graceful_cancel(monkeypatch):
    import ollamapull

    events = []
    stopped = [False]

    class Response:
        ok = True
        def __enter__(self): return self
        def __exit__(self, *args): self.close()
        def close(self): events.append('closed')
        def iter_lines(self):
            yield b'{"status":"downloading","total":100,"completed":50}'
            yield b'{"status":"success"}'

    monkeypatch.setattr(ollamapull.requests, 'post', lambda *a, **k: Response())
    def progress(message):
        events.append(message)
        if '50%' in message:
            stopped[0] = True
    with pytest.raises(ollamapull.OllamaPullCancelled):
        ollamapull.pull_model('ollamapull/test', progress=progress,
                              should_stop=lambda: stopped[0])
    assert any('50%' in event for event in events)
    assert 'closed' in events


def test_ollamapull_version_check_hides_windows_console(monkeypatch):
    import ollamapull
    import subprocess

    captured = {}
    monkeypatch.setattr(ollamapull.sys, 'platform', 'win32')
    monkeypatch.setattr(ollamapull, '_ollama_executable', lambda: r'C:\Ollama\ollama.exe')
    def fake_run(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(stdout='ollama version 0.34.4', stderr='')
    monkeypatch.setattr(ollamapull.subprocess, 'run', fake_run)

    assert ollamapull._binary_version() == '0.34.4'
    assert captured['creationflags'] & subprocess.CREATE_NO_WINDOW
    assert captured['startupinfo'].dwFlags & subprocess.STARTF_USESHOWWINDOW
    assert captured['startupinfo'].wShowWindow == subprocess.SW_HIDE


def test_ollamapull_pull_progress_shows_speed_and_eta(monkeypatch):
    import ollamapull
    import model_options

    clock = [0.0]
    monkeypatch.setattr(ollamapull.time, 'monotonic', lambda: clock[0])
    messages = []

    class Response:
        ok = True
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def close(self): pass
        def iter_lines(self):
            yield b'{"status":"pulling abc","digest":"abc","total":100000000,"completed":10000000}'
            clock[0] = 2.0
            yield b'{"status":"pulling abc","digest":"abc","total":100000000,"completed":30000000}'
            yield b'{"status":"success"}'

    monkeypatch.setattr(ollamapull.requests, 'post', lambda *_args, **_kwargs: Response())
    monkeypatch.setattr(model_options, 'refresh_ollamapull_model_catalog', lambda **_kwargs: None)
    ollamapull.pull_model('ollamapull/test', progress=messages.append)

    assert any('30%' in message and '10.0 MB/s' in message and 'ETA 7s' in message
               for message in messages)


def test_ollamapull_installer_platform_commands_and_graphical_elevation(monkeypatch):
    import ollamapull
    import subprocess
    from pathlib import Path

    script = Path('install.sh')
    monkeypatch.setattr(ollamapull.sys, 'platform', 'win32')
    assert ollamapull._installer_command(script)[-2:] == ['-File', str(script)]
    monkeypatch.setattr(ollamapull.sys, 'platform', 'linux')
    monkeypatch.setattr(ollamapull.os, 'geteuid', lambda: 1000, raising=False)
    monkeypatch.setattr(ollamapull.subprocess, 'run', lambda *a, **k: (_ for _ in ()).throw(
        subprocess.CalledProcessError(1, 'sudo')))
    monkeypatch.setattr(ollamapull.shutil, 'which', lambda name: '/usr/bin/pkexec' if name == 'pkexec' else None)
    monkeypatch.setenv('DISPLAY', ':0')
    assert ollamapull._installer_command(script) == ['/usr/bin/pkexec', '/bin/sh', str(script)]
    monkeypatch.delenv('DISPLAY')
    monkeypatch.delenv('WAYLAND_DISPLAY', raising=False)
    with pytest.raises(ollamapull.OllamaPullError, match='no graphical'):
        ollamapull._installer_command(script)


def test_ollamapull_macos_uses_user_app_installer(monkeypatch):
    import ollamapull

    calls = []
    monkeypatch.setattr(ollamapull.sys, 'platform', 'darwin')
    monkeypatch.setattr(ollamapull, '_install_macos_app', lambda *a: calls.append('mac-app'))
    ollamapull._run_installer(None, None)
    assert calls == ['mac-app']


def test_ollamapull_status_reports_old_external_server_after_update(monkeypatch):
    import ollamapull

    monkeypatch.setattr(ollamapull, '_ollama_executable', lambda: '/usr/bin/ollama')
    monkeypatch.setattr(ollamapull, '_binary_version', lambda: '0.2.0')
    monkeypatch.setattr(ollamapull, '_server_version', lambda: '0.1.0')
    monkeypatch.setattr(ollamapull, 'latest_version', lambda: '0.2.0')
    status = ollamapull.get_status()
    assert status['server_running'] is True
    assert status['update_available'] is False
    assert status['restart_required'] is True
    assert status['model_installed'] is None


def test_ollamapull_rejects_bad_official_installer_response(monkeypatch):
    import ollamapull

    monkeypatch.setattr(ollamapull.sys, 'platform', 'win32')
    response = SimpleNamespace(content=b'<html>not an installer</html>', raise_for_status=lambda: None)
    monkeypatch.setattr(ollamapull.requests, 'get', lambda *a, **k: response)
    with pytest.raises(ollamapull.OllamaPullError, match='not a valid install script'):
        ollamapull._run_installer(None, None)


ERROR = {
    "error": {
        "message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",
        "type": "invalid_request_error", "param": "max_tokens", "code": "unsupported_parameter",
    }
}


class ParameterError(Exception):
    status_code = 400
    body = ERROR


def bare_client(model="gpt-6-astra"):
    client = UnifiedClient.__new__(UnifiedClient)
    client.model = model
    client.client_type = "openai"
    client._get_active_request_model = lambda: model
    client._is_stop_requested = lambda: False
    client._active_per_key_output_token_limit = lambda: None
    client.get_cached_output_token_limit = lambda model: None
    return client


@pytest.mark.parametrize('prefix', ['authnd/', 'authnd2/'])
def test_authnd_progress_uses_actual_kimi_effort(monkeypatch, prefix):
    monkeypatch.setenv('ENABLE_GPT_THINKING', '1')
    monkeypatch.setenv('GPT_EFFORT', 'max')
    monkeypatch.delenv('AUTHND_ENABLE_THINKING', raising=False)
    monkeypatch.delenv('AUTHND_REASONING_EFFORT', raising=False)
    client = bare_client(prefix + 'moonshotai/kimi-k3')
    assert client._get_thinking_status_label() == ' (reasoning_effort: max)'


@pytest.mark.parametrize("model", ["gpt-6-astra", "openai/gpt-6-astra", "gpt-6", "gpt6-astra", "gpt-6-astra-2026-09-05"])
def test_gpt6_uses_completion_token_limit(model):
    client = bare_client(model)
    assert client._is_o_series_model()
    assert client._normalize_token_params(128000, None) == (None, 128000)
    body = client._build_openai_params([], 0.5, 128000)
    assert body["max_completion_tokens"] == 128000
    assert "max_tokens" not in body
    assert "temperature" not in body


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-60-test", "not-gpt-6-astra"])
def test_other_models_keep_existing_parameter_behavior(model):
    client = bare_client(model)
    assert not client._is_o_series_model()
    assert client._build_openai_params([], 0.5, 1234) == {
        "model": model, "messages": [], "temperature": 0.5, "max_tokens": 1234,
    }


def test_responses_api_keeps_max_output_tokens():
    body = {"model": "gpt-6-astra", "max_output_tokens": 128000, "temperature": 1,
            "top_p": 0.9, "logprobs": True, "top_logprobs": 2}
    UnifiedClient._apply_gpt6_openai_constraints(body, use_responses_api=True)
    assert body == {"model": "gpt-6-astra", "max_output_tokens": 128000}


@pytest.mark.parametrize("stream", [False, True])
def test_sdk_repairs_explicit_token_error_once(stream):
    calls = []
    result = object()

    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        if len(calls) == 1:
            raise ParameterError(ERROR["error"]["message"])
        return result

    kwargs = {"model": "custom-alias", "messages": [], "max_tokens": 128000,
              "stream": stream, "reasoning_effort": "max"}
    assert bare_client()._create_chat_completion_with_token_retry(create, kwargs, "openai") is result
    assert len(calls) == 2
    assert calls[1] == {**{k: v for k, v in calls[0].items() if k != "max_tokens"},
                        "max_completion_tokens": 128000}


def test_sdk_corrected_failure_is_not_retried_again():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        raise ParameterError("still rejected")

    with pytest.raises(ParameterError):
        bare_client()._create_chat_completion_with_token_retry(create, {"max_tokens": 128000}, "openai")
    assert len(calls) == 2


@pytest.mark.parametrize("status,error", [
    (429, ERROR), (401, ERROR),
    (400, {"error": {"message": "max_tokens exceeds the context window"}}),
    (400, {"error": {"message": "Unsupported parameter: 'temperature'"}}),
])
def test_unrelated_error_does_not_change_token_limit(status, error):
    body = {"max_tokens": 128000}
    assert not UnifiedClient._repair_chat_completion_token_limit(body, status, error)
    assert body == {"max_tokens": 128000}


def test_sdk_cancellation_prevents_corrected_retry():
    client = bare_client()
    client._is_stop_requested = lambda: True
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        raise ParameterError("rejected")

    with pytest.raises(UnifiedClientError, match="cancelled"):
        client._create_chat_completion_with_token_retry(create, {"max_tokens": 32}, "openai")
    assert len(calls) == 1


def test_direct_http_repairs_token_error_even_with_one_attempt(monkeypatch):
    client = bare_client()
    client._bind_thread_run_id_for_request = lambda: None
    client._get_send_interval = lambda: 0
    client._get_thread_directory = lambda: None
    client._ignore_graceful_stop = False
    client.request_timeout = 30
    monkeypatch.delenv("GRACEFUL_STOP", raising=False)
    monkeypatch.setattr(api, "_save_outgoing_request", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_save_incoming_response", lambda *args, **kwargs: None)
    calls = []
    closed = []
    success = SimpleNamespace(status_code=200, headers={}, json=lambda: {"ok": True})

    def request(*args, **kwargs):
        calls.append(copy.deepcopy(kwargs["json"]))
        if len(calls) == 1:
            return SimpleNamespace(status_code=400, headers={}, text=json.dumps(ERROR),
                                   json=lambda: ERROR, close=lambda: closed.append(True))
        return success

    monkeypatch.setattr(api.requests, "request", request)
    result = client._http_request_with_retries(
        "POST", "https://api.openai.com/v1/chat/completions",
        json={"model": "alias", "max_tokens": 128000}, max_retries=1,
    )
    assert result is success
    assert len(calls) == 2
    assert calls[1] == {"model": "alias", "max_completion_tokens": 128000}
    assert closed == [True]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-custom-reasoning-alias"])
def test_active_sdk_path_preflight_and_corrected_retry(monkeypatch, tmp_path, stream, model):
    calls = []
    logs = []
    monkeypatch.setattr(api, "print", lambda *args, **kwargs: logs.append(" ".join(map(str, args))))
    reported_usage = SimpleNamespace(prompt_tokens=10, completion_tokens=1240, total_tokens=1250,
                                     completion_tokens_details=SimpleNamespace(reasoning_tokens=1234))
    usage_dict = {"prompt_tokens": 10, "completion_tokens": 1240, "total_tokens": 1250,
                  "completion_tokens_details": {"reasoning_tokens": 1234}}
    reported_usage.model_dump = lambda: usage_dict

    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        if "max_tokens" in kwargs:
            raise ParameterError(ERROR["error"]["message"])
        if model == "gpt-6-astra":
            # Mirror the public endpoint: max is accepted only by Responses.
            assert "messages" not in kwargs
            body = {"status": "completed", "output": [{"type": "message", "content": [
                {"type": "output_text", "text": "OK"}]}],
                "usage": {"input_tokens": 10, "output_tokens": 1240, "total_tokens": 1250,
                          "output_tokens_details": {"reasoning_tokens": 1234}}}
            if stream:
                return iter([
                    SimpleNamespace(type="response.reasoning_summary_text.delta", delta="Checking the request."),
                    SimpleNamespace(type="response.output_text.delta", delta="OK"),
                    SimpleNamespace(type="response.completed", response=SimpleNamespace(model_dump=lambda: body)),
                ])
            return body
        if stream:
            return iter([SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content="OK"), finish_reason="stop",
            )]), SimpleNamespace(choices=[], usage=reported_usage)])
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="OK"), finish_reason="stop",
        )], usage=reported_usage)

    fake_sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                               responses=SimpleNamespace(create=create), close=lambda: None)
    monkeypatch.setattr(api.openai, "OpenAI", lambda **kwargs: fake_sdk)
    monkeypatch.setattr(api, "httpx", None)
    monkeypatch.setenv("USE_CUSTOM_OPENAI_ENDPOINT", "0")
    monkeypatch.setenv("ENABLE_GPT_THINKING", "1")
    monkeypatch.setenv("GPT_EFFORT", "max")
    monkeypatch.setenv("PASS_THINKING_TO_OPENAI_COMPATIBLE", "0")
    monkeypatch.delenv("GRACEFUL_STOP", raising=False)
    client = UnifiedClient("test-key", model, str(tmp_path))
    monkeypatch.setattr(client, "_get_max_retries", lambda: 1)
    monkeypatch.setattr(client, "_get_send_interval", lambda: 0)
    monkeypatch.setattr(client, "_streaming_enabled", lambda: stream)
    monkeypatch.setattr(client, "_stream_logging_enabled", lambda enabled: False)
    monkeypatch.setattr(client, "_is_stop_requested", lambda: False)
    monkeypatch.setattr(client, "_save_response", lambda *args, **kwargs: None)
    monkeypatch.setattr(client, "_should_show_api_lifecycle_logs", lambda: False)
    monkeypatch.setattr(client, "_get_anti_duplicate_params", lambda *args, **kwargs: {"top_p": 0.8})
    result = client._send_openai_compatible(
        [{"role": "user", "content": "Reply OK."}], 0.5, 128000,
        "https://api.openai.com/v1", "token-test", provider="openai",
    )
    assert result.content == "OK"
    assert "max_tokens" not in calls[-1]
    if model == "gpt-6-astra":
        assert len(calls) == 1
        assert calls[0]["reasoning"] == {"effort": "max", "summary": "auto"}
        assert calls[0]["max_output_tokens"] == 128000
        assert "reasoning_effort" not in calls[0]
        assert "max_completion_tokens" not in calls[0]
        assert "temperature" not in calls[0]
        assert "top_p" not in calls[0]
        assert result.usage == usage_dict
        if stream:
            assert "stream_options" not in calls[0]
            assert result._streaming_thinking_chunks == 1
            assert any("Thinking tokens used: 1,234 (reported by API)" in line for line in logs)
    else:
        assert len(calls) == 2
        assert calls[-1]["max_completion_tokens"] == 128000


@pytest.mark.parametrize("pass_all", ["0", "1"])
@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_native_gpt6_effort_does_not_require_pass_all(monkeypatch, pass_all, effort):
    monkeypatch.setenv("ENABLE_GPT_THINKING", "1")
    monkeypatch.setenv("GPT_EFFORT", effort)
    monkeypatch.setenv("PASS_THINKING_TO_OPENAI_COMPATIBLE", pass_all)
    assert bare_client()._get_openai_compatible_reasoning_effort("openai", "gpt-6-astra") == effort


def test_gpt6_does_not_send_unsupported_thinking_toggle(monkeypatch):
    monkeypatch.setenv("ENABLE_GPT_THINKING", "0")
    monkeypatch.setenv("PASS_THINKING_TO_OPENAI_COMPATIBLE", "1")
    client = bare_client()
    assert client._get_openai_compatible_reasoning_effort("openai", "gpt-6-astra") == "low"
    assert not client._get_openai_compatible_thinking_disabled("openai", "gpt-6-astra")


@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-6-luna", "gpt-6-sol", "gpt-6-pro"])
def test_http_path_sends_native_gpt6_effort(monkeypatch, tmp_path, model):
    monkeypatch.setattr(api.openai, "OpenAI", lambda **kwargs: SimpleNamespace(close=lambda: None))
    monkeypatch.setenv("USE_CUSTOM_OPENAI_ENDPOINT", "0")
    monkeypatch.setenv("ENABLE_GPT_THINKING", "1")
    monkeypatch.setenv("GPT_EFFORT", "max")
    monkeypatch.setenv("PASS_THINKING_TO_OPENAI_COMPATIBLE", "0")
    client = UnifiedClient("test-key", model, str(tmp_path))
    monkeypatch.setattr(api, "openai", None)
    monkeypatch.setattr(client, "_is_stop_requested", lambda: False)
    monkeypatch.setattr(client, "_get_max_retries", lambda: 1)
    monkeypatch.setattr(client, "_get_send_interval", lambda: 0)
    monkeypatch.setattr(client, "_save_response", lambda *args, **kwargs: None)
    monkeypatch.setattr(client, "_should_show_api_lifecycle_logs", lambda: False)
    bodies = []
    urls = []

    def request(**kwargs):
        bodies.append(kwargs["json"])
        urls.append(kwargs["url"])
        if kwargs["url"].endswith("/responses"):
            payload = {"status": "completed", "output": [{"type": "message", "content": [
                {"type": "output_text", "text": "OK"}]}]}
        else:
            payload = {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}
        return SimpleNamespace(headers={"content-type": "application/json"}, json=lambda: payload)

    monkeypatch.setattr(client, "_http_request_with_retries", request)
    response = client._send_openai_compatible(
        [{"role": "user", "content": "Reply OK."}], 0.5, 128000,
        "https://api.openai.com/v1", "effort-test", provider="openai",
    )
    assert response.content == "OK"
    assert urls == ["https://api.openai.com/v1/responses"]
    assert bodies[0]["max_output_tokens"] == 128000
    assert "reasoning_effort" not in bodies[0]
    assert "max_completion_tokens" not in bodies[0]
    if model != "gpt-6-pro":
        assert bodies[0]["reasoning"] == {"effort": "max", "summary": "auto"}
    else:
        assert bodies[0]["reasoning"] == {"effort": "max"}


@pytest.mark.parametrize("provider,model,url,expected", [
    ("openai", "gpt-6-astra", "https://api.openai.com/v1", True),
    ("openai", "gpt-6-astra-2026-09-05", "https://api.openai.com/v1/", True),
    ("openai", "gpt-5.6-sol", "https://api.openai.com/v1", False),
    ("openai", "gpt-6-astra", "https://custom.example/v1", False),
    ("openrouter", "openai/gpt-6-astra", "https://openrouter.ai/api/v1", False),
    ("authgpt", "gpt-6-astra", "https://chatgpt.com/backend-api/codex", False),
])
def test_astra_responses_route_is_limited_to_public_openai(provider, model, url, expected):
    assert UnifiedClient._uses_astra_responses_api(provider, model, url) is expected


@pytest.mark.parametrize('endpoint', ['chat/completions', 'responses'])
@pytest.mark.parametrize('fails_twice', [False, True])
def test_http_reasoning_retry_is_corrected_and_bounded(monkeypatch, endpoint, fails_twice):
    client = bare_client('custom-model')
    client._bind_thread_run_id_for_request = lambda: None
    client._get_send_interval = lambda: 0
    client._get_thread_directory = lambda: None
    client._ignore_graceful_stop = False
    client.request_timeout = 30
    monkeypatch.delenv('GRACEFUL_STOP', raising=False)
    monkeypatch.setattr(api, '_save_outgoing_request', lambda *args, **kwargs: None)
    monkeypatch.setattr(api, '_save_incoming_response', lambda *args, **kwargs: None)
    calls = []
    error = {'error': {'param': 'reasoning_effort', 'message':
                      "Unsupported value: 'max'. Supported values are: 'low', 'medium', 'high', 'xhigh'."}}

    def request(*args, **kwargs):
        calls.append(copy.deepcopy(kwargs['json']))
        status = 400 if len(calls) == 1 or fails_twice else 200
        return SimpleNamespace(status_code=status, headers={}, text=json.dumps(error),
                               json=lambda: error, close=lambda: None)

    monkeypatch.setattr(api.requests, 'request', request)
    payload = {'model': 'custom-model', 'reasoning_effort': 'max'} if endpoint == 'chat/completions' else {
        'model': 'custom-model', 'reasoning': {'effort': 'max'}}
    def send():
        return client._http_request_with_retries('POST', 'https://example.test/v1/' + endpoint,
                                                json=payload, max_retries=7)
    if fails_twice:
        with pytest.raises(UnifiedClientError):
            send()
    else:
        assert send().status_code == 200
    assert len(calls) == 2
    assert (calls[1].get('reasoning_effort') or calls[1]['reasoning']['effort']) == 'xhigh'


@pytest.mark.parametrize('responses', [False, True])
def test_sdk_reasoning_retry_preserves_other_parameters(responses):
    client = bare_client('custom-model')
    error = ParameterError('Unsupported reasoning effort')
    error.body = {'error': {'param': 'reasoning_effort', 'message':
                           "Unsupported value: 'max'. Supported values are: 'low', 'high'."}}
    calls = []
    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        if len(calls) == 1:
            raise error
        return 'OK'
    payload = {'model': 'custom-model', 'max_output_tokens' if responses else 'max_completion_tokens': 128000}
    payload.update({'reasoning': {'effort': 'max'}} if responses else {'reasoning_effort': 'max'})
    send = client._create_with_reasoning_retry if responses else client._create_chat_completion_with_token_retry
    assert send(create, payload, 'openai') == 'OK'
    assert len(calls) == 2
    assert (calls[1].get('reasoning_effort') or calls[1]['reasoning']['effort']) == 'high'
    assert calls[1]['max_output_tokens' if responses else 'max_completion_tokens'] == 128000


def rejection(supported, param='reasoning_effort'):
    return {'error': {'param': param, 'code': 'unsupported_value',
                     'message': f"Unsupported value for {param}. Supported values are: "
                     + ', '.join(repr(value) for value in supported) + '.'}}


class Rejected(Exception):
    status_code = 400

    def __init__(self, supported):
        self.body = rejection(supported)
        super().__init__(str(self.body))


@pytest.mark.parametrize('requested,supported,expected', [
    ('max', ['low', 'medium', 'high', 'xhigh'], 'xhigh'),
    ('max', ['low', 'medium', 'high'], 'high'),
    ('none', ['low', 'medium', 'high'], 'low'),
    ('medium', ['low', 'high'], 'low'),
    ('medium', ['high', 'low'], 'low'),
    ('medium', ['low'], 'low'),
    ('xhigh', ['high', 'max'], 'high'),
    ('low', ['medium', 'high'], 'medium'),
])
@pytest.mark.parametrize('shape', ['chat', 'responses', 'extra_body'])
def test_nearest_supported_effort(requested, supported, expected, shape):
    body = {'reasoning_effort': requested} if shape == 'chat' else {'reasoning': {'effort': requested, 'summary': 'auto'}}
    if shape == 'extra_body':
        body = {'extra_body': body}
    body.update(model='custom-model', max_tokens=128000)
    logs = []
    calls = []

    def create(payload):
        calls.append(copy.deepcopy(payload))
        if len(calls) == 1:
            raise Rejected(supported)
        return 'OK'

    assert call_with_reasoning_retry(create, body, logs.append, lambda: None) == 'OK'
    assert len(calls) == 2
    final = calls[1].get('extra_body', calls[1])
    assert (final['reasoning_effort'] if shape == 'chat' else final['reasoning']['effort']) == expected
    assert calls[1]['max_tokens'] == 128000
    assert logs[0].startswith('📝')


def test_corrective_retry_is_bounded():
    calls = []

    def create(payload):
        calls.append(copy.deepcopy(payload))
        raise Rejected(['low', 'high'])

    with pytest.raises(ReasoningEffortRejected):
        call_with_reasoning_retry(create, {'reasoning_effort': 'max'}, print, lambda: None)
    assert len(calls) == 2


def test_cancellation_prevents_corrective_retry():
    def cancel():
        raise RuntimeError('cancelled')
    body = {'reasoning_effort': 'max'}
    with pytest.raises(RuntimeError, match='cancelled'):
        call_with_reasoning_retry(lambda payload: (_ for _ in ()).throw(Rejected(['high'])), body, print, cancel)
    assert body['reasoning_effort'] == 'max'


@pytest.mark.parametrize('status,error', [
    (429, rejection(['low'])), (401, rejection(['low'])),
    (400, rejection(['high'], 'temperature')),
    (400, {'error': {'param': 'reasoning_effort', 'message': 'Request timed out'}}),
])
def test_unrelated_errors_are_not_repaired(status, error):
    assert supported_reasoning_efforts(status, error) is None


@pytest.mark.parametrize('model', ['gpt-6-astra', 'authgpt/gpt-6-astra', 'or/openai/gpt-6-astra', 'gpt-6-sol'])
def test_none_preflight(model):
    body = {'model': model, 'reasoning': {'effort': 'none'}, 'thinking': {'type': 'disabled'}}
    logs = []
    normalize_none_effort(body, logs.append)
    assert body['reasoning']['effort'] == 'low'
    assert 'thinking' not in body
    assert len(logs) == 1
    if 'astra' in model:
        assert logs == ['📝 Astra does not support none, using low instead']


def test_other_models_none_unchanged():
    body = {'model': 'gpt-5.6-sol', 'reasoning_effort': 'none'}
    normalize_none_effort(body)
    assert body['reasoning_effort'] == 'none'


def test_no_advertised_values_does_not_guess():
    error = {'error': {'param': 'reasoning.effort', 'message': "Unsupported value: 'max'"}}
    assert supported_reasoning_efforts(400, error) == []
    assert not repair_reasoning_effort({'reasoning_effort': 'max'}, [], print)
