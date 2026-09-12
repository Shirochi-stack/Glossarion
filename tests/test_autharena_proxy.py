import ast
import base64
import json
import os
from pathlib import Path
import sys
import time
import threading
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import autharena_proxy as arena


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTHARENA_PROXY_DATA_DIR", str(tmp_path / "Arena runtime with spaces"))
    arena.reset_cancel()
    yield
    arena.reset_cancel()


@pytest.mark.parametrize("route,slot", [("autharena/m", 0), ("autharena0/m", None),
                                         ("autharena1/m", 1), ("autharena2/m", 2),
                                         ("autharena17/m", 17), ("AUTHARENA1/m", 1)])
def test_prefix_numbers_match_authgpt_with_zero_reserved_for_rotation(route, slot):
    assert arena.parse_route(route) == (slot, "m")


def test_selected_slot_zero_never_generates_rotation():
    assert arena.route_for_slot(0, "m") == "autharena/m"
    assert arena.parse_route(arena.route_for_slot(4, "m"))[0] == 4


def event(delta=None, finish=None, usage=None):
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}], "usage": usage})


def test_stream_is_delivered_before_completion_and_preserves_whitespace():
    logged = []
    def lines():
        yield event({"content": " hello "})
        assert logged == ["📡 Arena: Text streaming..."]
        yield event({"content": "world\n"})
        assert logged[-1] == " hello world"
        yield event({"reasoning_content": "thinking\n"})
        assert logged[-1] == "    thinking"
        yield event(finish="length", usage={"total_tokens": 10})
        yield "data: [DONE]"
    result = arena.consume_stream(lines(), logged.append)
    assert result["content"] == " hello world\n"
    assert result["finish_reason"] == "length"
    assert result["usage"] == {"total_tokens": 10}


def test_token_sized_html_chunks_are_grouped_before_completion():
    logged = []
    paragraph = '<p>Every time her throat was stabbed, Sia gagged.</p>'
    def lines():
        for token in ('<p>Every ', 'time ', 'her ', 'throat ', 'was ', 'stabbed, ', 'Sia ', 'gagged.', '</p', '>'):
            yield event({"content": token})
        assert logged == ['📡 Arena: Text streaming...', paragraph]
        yield event(finish='stop')
        yield 'data: [DONE]'
    assert arena.consume_stream(lines(), logged.append)['content'] == paragraph
    assert logged[-1] == '📡 Arena: Stream complete'


def test_long_plain_text_stream_flushes_without_waiting_for_completion():
    logged = []
    def lines():
        for _ in range(31):
            yield event({"content": 'word '})
        assert logged == ['📡 Arena: Text streaming...', 'word ' * 31]
        yield event(finish='stop')
        yield 'data: [DONE]'
    assert arena.consume_stream(lines(), logged.append)['content'] == 'word ' * 31


def test_stream_phase_switches_flush_without_mixing_reasoning_and_content():
    logged = []
    result = arena.consume_stream([
        event({'reasoning_content': 'Think '}), event({'reasoning_content': 'first'}),
        event({'content': 'Hello '}), event({'content': 'world'}),
        event({'reasoning_content': 'Check again'}), event({'content': '!'}),
        event(finish='stop'), 'data: [DONE]',
    ], logged.append)
    assert result['content'] == 'Hello world!'
    assert result['reasoning_content'] == 'Think firstCheck again'
    assert logged == [
        '🧠 [autharena] Thinking...', '    Think first', '─' * 50,
        '📡 Arena: Text streaming...', 'Hello world', '─' * 50,
        '🧠 [autharena] Thinking...', '    Check again', '─' * 50,
        '📡 Arena: Text streaming...', '!', '📡 Arena: Stream complete',
    ]


def test_interrupted_stream_flushes_remainder_without_complete_banner():
    logged = []
    with pytest.raises(arena.ArenaStreamError):
        arena.consume_stream([event({'content': 'unfinished'}), 'data: {"error":"connection lost"}'], logged.append)
    assert logged == ['📡 Arena: Text streaming...', 'unfinished']


def test_stream_display_escaping_does_not_change_returned_text():
    logged = []
    result = arena.consume_stream([event({'content': '\x1ftext'}), event(finish='stop'), 'data: [DONE]'], logged.append)
    assert result['content'] == '\x1ftext'
    assert '\\x1Ftext' in logged


def test_hidden_stream_has_no_content_or_phase_logs():
    logged = []
    result = arena.consume_stream([event({'reasoning_content': 'think', 'content': 'text'}), event(finish='stop'), 'data: [DONE]'], logged.append, log_stream=False)
    assert result['content'] == 'text'
    assert result['reasoning_content'] == 'think'
    assert logged == []


@pytest.mark.parametrize('message', ['', None, '   '])
def test_empty_upstream_error_after_reasoning_explains_interrupted_phase(message):
    lines = [event({'reasoning_content': 'still thinking'}),
             'data: ' + json.dumps({'error': {'message': message, 'type': 'internal_error'}})]
    with pytest.raises(arena.ArenaStreamError, match='after reasoning, before answer text') as error:
        arena.consume_stream(lines, log_stream=False)
    assert error.value.partial_response is True


@pytest.mark.parametrize("ending", [[], ["data: [DONE]"], [event(finish="stop")]])
def test_incomplete_stream_is_not_reported_as_success(ending):
    with pytest.raises(RuntimeError, match="interrupted"):
        arena.consume_stream([event({"content": "partial"}), *ending], log_stream=False)


def test_cancellation_generation_survives_reset():
    generation = arena.capture_cancel_generation()
    arena.cancel_stream()
    arena.reset_cancel()
    with pytest.raises(RuntimeError, match="cancelled"):
        arena.consume_stream([event({"content": "late"})], cancel_generation=generation)


@pytest.mark.parametrize("batch,visible,forced,expected", [(0, 1, 0, True), (0, 0, 1, False), (1, 1, 0, False), (1, 0, 1, True)])
def test_stream_visibility_matches_antigravity(monkeypatch, batch, visible, forced, expected):
    monkeypatch.setenv("BATCH_TRANSLATION", str(batch))
    monkeypatch.setenv("LOG_STREAM_CHUNKS", str(visible))
    monkeypatch.setenv("ALLOW_AUTHGPT_BATCH_STREAM_LOGS", str(forced))
    monkeypatch.setenv("ENABLE_STREAMING", "0")
    monkeypatch.setenv("ALLOW_BATCH_STREAM_LOGS", "0")
    monkeypatch.setenv("STREAM_THINKING_LOGS", "0")
    assert arena.visible_stream() is expected


def test_archive_traversal_rejected(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escape.py", "bad")
    with pytest.raises(RuntimeError, match="Unsafe"):
        arena._extract(archive, tmp_path / "extract")
    assert not (tmp_path / "escape.py").exists()


def test_large_translation_environment_not_inherited(monkeypatch):
    for i in range(4):
        monkeypatch.setenv(f"SYSTEM_PROMPT_{i}", "x" * 20000)
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    env = arena._env()
    assert not any(k.startswith("SYSTEM_PROMPT") for k in env)
    assert "OPENAI_API_KEY" not in env
    assert sum(len(k) + len(v) for k, v in env.items()) < 32767


def test_cached_runtime_does_not_download_or_require_system_python(monkeypatch):
    runtime = arena.data_dir() / ("bridge-" + arena.REVISION)
    python = runtime / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.touch()
    (runtime / "ready").touch()
    browser = runtime / "chromium.exe"
    browser.touch()
    (runtime / "qt-browser-ready").write_text("qt6", encoding="utf-8")
    monkeypatch.setattr(arena, "_download", lambda *args: pytest.fail("Unexpected download"))
    assert arena._ensure_runtime() == (runtime, python)


def test_install_lock_does_not_block_gui_accounts_or_cancellation():
    completed = threading.Event()
    def query():
        assert arena.list_accounts() == []
        arena.cancel_stream()
        completed.set()
    with arena.disk_lock("setup"):
        worker = threading.Thread(target=query)
        worker.start()
        assert completed.wait(2), "Background installation blocked the GUI account reader or Stop"
    worker.join()


def test_split_cookie_capture_excludes_other_sites_and_guest_accounts():
    session = {"user": {"id": "user", "email": "user@example.test", "is_anonymous": False}, "expires_at": time.time() + 1000}
    token = "base64-" + base64.urlsafe_b64encode(json.dumps(session).encode()).decode()
    third = len(token) // 3
    cookies = [{"name": f"arena-auth-prod-v1.{i}", "domain": ".arena.ai", "value": value}
               for i, value in enumerate((token[:third], token[third:third * 2], token[third * 2:]))]
    cookies.append({"name": "private", "domain": "example.com", "value": "secret"})
    result = arena.session_from_cookies(cookies)
    assert result["token"] == token
    assert len(result["cookies"]) == 3
    session["user"]["is_anonymous"] = True
    cookie = {"name": "arena-auth-prod-v1", "domain": "arena.ai", "value": "base64-" + base64.urlsafe_b64encode(json.dumps(session).encode()).decode()}
    assert arena.session_from_cookies([cookie]) is None


def test_numbered_catalog_and_completions():
    import model_options
    for route in ("autharena/", "autharena0/", "autharena1/", "autharena29/"):
        assert model_options.catalog_provider_for_model(route + "model") == "autharena"
    assert model_options.numbered_model_completion_values(["autharena/model"], "autharena2/") == ["autharena2/model"]


@pytest.fixture
def qt(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets")
    app = widgets.QApplication.instance() or widgets.QApplication([])
    yield widgets, app


def test_account_selector_visibility_and_numeric_labels(qt, monkeypatch):
    widgets, app = qt
    monkeypatch.setattr(arena, "list_accounts", lambda: [{"slot": 0}, {"slot": 1}])
    model = ["autharena/model"]
    parent = widgets.QWidget()
    control = arena.create_login_controls(parent, lambda: model[0], lambda value: model.__setitem__(0, value))
    assert control.layout().itemAt(0).widget() is control.login_button
    assert control.layout().itemAt(1).widget() is control.accounts
    from PySide6.QtTest import QTest
    deadline = time.monotonic() + 2
    while not control.accounts_ready and time.monotonic() < deadline:
        QTest.qWait(10)
    assert control.login_button.text() == "✅ Arena"
    assert "#0" in control.login_button.toolTip()
    assert control.accounts.isHidden()
    model[0] = "autharena0/model"
    control.refresh()
    assert not control.accounts.isHidden()
    assert [control.accounts.itemText(i) for i in range(control.accounts.count())] == ["#0", "#1", "+ N"]
    control.select_account(1)
    control.refresh()
    assert control.accounts.currentData() == 1
    assert control.login_button.text() == "✅ Arena #1"
    assert model[0] == "autharena0/model"
    model[0] = "autharena1/model"
    control.refresh()
    assert control.accounts.isHidden()
    assert control.accounts.currentData() == 1
    assert control.login_button.text() == "✅ Arena #1"
    model[0] = "autharena2/model"
    control.refresh()
    assert control.login_button.text() == "Arena #2 Login"
    # Pool login status belongs to the selected slot, not any other saved slot.
    control.saved_accounts = [{"slot": 0}]
    model[0] = "autharena0/model"
    control.refresh()
    assert control.login_button.text() == "Arena #1 Login"
    model[0] = "autharena/model"
    control.refresh()
    assert control.login_button.text() == "✅ Arena"
    parent.close()


@pytest.mark.parametrize("failure", [False, True])
def test_login_animates_immediately_and_resets(qt, monkeypatch, failure):
    from PySide6.QtTest import QTest
    widgets, app = qt
    monkeypatch.setattr(arena, "list_accounts", lambda: [])
    monkeypatch.setattr(widgets.QMessageBox, "warning", lambda *args: None)
    class PendingWorker:
        def __init__(self, **kwargs):
            pass
        def start(self):
            pass
    monkeypatch.setattr(arena.threading, "Thread", PendingWorker)
    parent = widgets.QWidget()
    control = arena.create_login_controls(parent, lambda: "autharena1/m", lambda value: None, log_fn=lambda value: None)
    control.receive_accounts([], None)
    control.start(0, False)
    assert control.busy and control.spinner.isActive()
    assert not control.login_button.isEnabled()
    assert control.login_button.text() == "Signing in…"
    assert not control.login_button.icon().isNull()
    angle = control.spinner_angle
    deadline = time.monotonic() + 1
    while control.spinner_angle == angle and time.monotonic() < deadline:
        QTest.qWait(50)
    assert control.spinner_angle != angle
    control.show_progress("Arena: downloading Chromium")
    assert control.login_button.toolTip() == "Arena: downloading Chromium"
    control.finished(None if failure else ({"slot": 0}, "autharena1/m", False), "Test failure" if failure else None)
    assert not control.spinner.isActive()
    assert control.login_button.isEnabled()
    assert control.login_button.icon().isNull()
    assert control.login_button.text() == "Arena #1 Login"
    parent.close()


def test_model_typing_does_not_block_on_arena_accounts_or_start_proxy(qt, monkeypatch):
    from PySide6.QtCore import QTimer
    from PySide6.QtTest import QTest
    widgets, app = qt
    release = threading.Event()
    entered = threading.Event()
    readers = []
    def slow_accounts():
        readers.append(threading.get_ident())
        entered.set()
        assert release.wait(3)
        return [{"slot": 0, "email": "test@example.com"}]
    def forbidden_proxy(*args, **kwargs):
        pytest.fail("Typing must not start the Arena proxy")
    monkeypatch.setattr(arena, "list_accounts", slow_accounts)
    monkeypatch.setattr(arena, "ensure_proxy_running", forbidden_proxy)
    parent = widgets.QWidget()
    combo = widgets.QComboBox(parent)
    combo.setEditable(True)
    arena.install_combo_login(parent, combo)
    try:
        combo.setCurrentText("autharena/model")
        assert entered.wait(1)
        ticks = []
        QTimer.singleShot(0, lambda: ticks.append(True))
        for index in range(25):
            combo.setCurrentText("autharena/model" + str(index))
        QTest.qWait(20)
        assert ticks and len(readers) == 1
        assert readers[0] != threading.get_ident()
        release.set()
        control = combo._autharena_controls
        deadline = time.monotonic() + 2
        while not control.accounts_ready and time.monotonic() < deadline:
            QTest.qWait(10)
        assert control.login_button.text() == "✅ Arena"
        combo.setCurrentText("autharena/another-model")
        assert len(readers) == 1
    finally:
        release.set()
        parent.close()


def test_login_reuses_started_service(monkeypatch):
    state = {"url": "http://127.0.0.1:12345", "key": "test"}
    calls = []
    monkeypatch.setattr(arena, "ensure_proxy_running", lambda **kwargs: calls.append("start") or state)
    def request(path, payload, **kwargs):
        assert kwargs["status"] is state
        assert path == "/login" and payload == {"slot": 0}
        return {"slot": 0}
    monkeypatch.setattr(arena, "_request", request)
    assert arena.open_login(log_fn=None) == {"slot": 0}
    assert calls == ["start"]


def test_multi_key_login_control_follows_only_its_row(qt, monkeypatch):
    widgets, app = qt
    monkeypatch.setattr(arena, "list_accounts", lambda: [])
    parent = widgets.QWidget()
    combos = [widgets.QComboBox(parent), widgets.QComboBox(parent)]
    for combo in combos:
        combo.setEditable(True)
        arena.install_combo_login(parent, combo)
    combos[0].setCurrentText("autharena1/m")
    combos[1].setCurrentText("autharena2/m")
    control = combos[1]._autharena_controls
    control.finished(({"slot": 3}, "autharena2/m", True), None)
    assert combos[0].currentText() == "autharena1/m"
    assert combos[1].currentText() == "autharena3/m"
    combos[1].setCurrentText("other/m")
    control.finished(({"slot": 5}, "autharena3/m", True), None)
    assert combos[1].currentText() == "other/m"
    parent.close()


def test_frozen_specs_include_managed_worker_source():
    root = Path(__file__).resolve().parents[1] / "src"
    for spec in root.glob("translator*.spec"):
        text = spec.read_text(encoding="utf-8")
        assert "('autharena_proxy.py', '.')" in text, spec.name
        assert "('token_encryption.py', '.')" in text, spec.name
        ast.parse(text)


def test_internal_browser_installs_automatically_and_reuses_cache(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    browser = tmp_path / "managed-browser.exe"
    browser.touch()
    calls = []
    def run(args, log_fn):
        calls.append(args)
        return str(browser) if "-c" in args else ""
    monkeypatch.setattr(arena, "_run", run)
    arena._ensure_browser(runtime, "managed-python", lambda message: None)
    assert calls[0][1:] == ["pip", "install", "--python", "managed-python", "PySide6>=6.8", "playwright>=1.60"]
    assert len(calls) == 2
    arena._ensure_browser(runtime, "managed-python", lambda message: None)
    assert len(calls) == 2
    assert arena._env()["PLAYWRIGHT_BROWSERS_PATH"] == str(arena.data_dir() / "browsers")


def test_failed_browser_install_is_not_cached(tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError("download failed")
    monkeypatch.setattr(arena, "_run", fail)
    with pytest.raises(RuntimeError, match="download failed"):
        arena._ensure_browser(tmp_path, "managed-python", lambda message: None)
    assert not (tmp_path / "qt-browser-ready").exists()


@pytest.mark.parametrize("version", [None, 1])
def test_previous_external_browser_worker_is_detected(monkeypatch, version):
    from types import SimpleNamespace
    monkeypatch.setattr(arena, "_load", lambda name: {"url": "http://127.0.0.1:1234", "key": "test"})
    monkeypatch.setattr(arena.requests, "get", lambda *args, **kwargs: SimpleNamespace(
        ok=True, json=lambda: {"revision": arena.REVISION, "adapter_version": version}))
    status = arena.check_proxy_health()
    assert status["outdated"] is True
    assert status["running"] is False


def test_refreshed_session_is_encrypted_and_survives_process_restart():
    import subprocess
    from token_encryption import is_encrypted
    session = {"user": {"id": "saved-user", "email": "saved@example.test"},
               "expires_at": time.time() + 3600, "refresh_token": "test-refresh-secret"}
    cookie = {"name": "arena-auth-prod-v1", "domain": ".arena.ai", "path": "/", "value":
              "base64-" + base64.urlsafe_b64encode(json.dumps(session).encode()).decode()}
    arena._save("accounts.enc", {"0": {"user_id": "saved-user", "expires_at": 1}})
    arena._persist_session(0, [cookie])
    path = arena.data_dir() / "accounts.enc"
    assert is_encrypted(str(path))
    assert b"saved@example.test" not in path.read_bytes()
    assert cookie["value"].encode() not in path.read_bytes()
    code = "import autharena_proxy as a; s=a._load('accounts.enc')['0']; assert s['user_id']=='saved-user'; assert s['expires_at']>1; assert s['cookies']; print('RESTORED')"
    env = dict(os.environ, PYTHONPATH=str(Path(arena.__file__).parent))
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "RESTORED"
    session["user"]["id"] = "different-user"
    cookie["value"] = "base64-" + base64.urlsafe_b64encode(json.dumps(session).encode()).decode()
    with pytest.raises(RuntimeError, match="different account"):
        arena._persist_session(0, [cookie])
    assert arena._load("accounts.enc")["0"]["user_id"] == "saved-user"


def test_older_parallel_session_cannot_overwrite_refreshed_credentials():
    expiration = int(time.time()) + 3600
    session = {"user": {"id": "saved-user", "email": "saved@example.test"},
               "expires_at": expiration, "refresh_token": "older-refresh"}

    def cookie_for(value):
        return {"name": "arena-auth-prod-v1", "domain": ".arena.ai", "path": "/", "value":
                "base64-" + base64.urlsafe_b64encode(json.dumps(value).encode()).decode()}

    older_cookie = cookie_for(session)
    session.update(expires_at=expiration + 100, refresh_token="newer-refresh")
    newer_cookie = cookie_for(session)
    arena._save("accounts.enc", {"0": {"user_id": "saved-user", "expires_at": 1}})
    arena._persist_session(0, [newer_cookie])
    restored = arena._persist_session(0, [older_cookie])
    # The old context still reports its own session, but disk retains the newer
    # refresh credentials for the next restart or newly created context.
    assert restored["token"] == older_cookie["value"]
    saved = arena._load("accounts.enc")["0"]
    assert saved["expires_at"] == expiration + 100
    assert saved["cookies"] == [newer_cookie]

    # An old context refreshing during a reconnect must not replace the new
    # login, even when its newly minted token has a later expiration.
    session.update(expires_at=expiration + 200, refresh_token="obsolete-login-refresh")
    arena._persist_session(0, [cookie_for(session)], expected_token=older_cookie["value"])
    assert arena._load("accounts.enc")["0"] == saved


def test_interrupted_install_is_not_published(tmp_path, monkeypatch):
    uv = arena.data_dir() / ("uv.exe" if os.name == "nt" else "uv")
    uv.touch()
    def fail(*args):
        raise RuntimeError("Download interrupted")
    monkeypatch.setattr(arena, "_download", fail)
    with pytest.raises(RuntimeError, match="Download interrupted"):
        arena._ensure_runtime(log_fn=lambda value: None)
    assert not list(arena.data_dir().glob("bridge-*/ready"))
    assert not list(arena.data_dir().glob("setup-*"))


def test_catalog_parser_handles_split_nextjs_frames_and_changed_following_field():
    records = [{"id": "model-uuid", "publicName": "model-name", "displayName": "日本語", "capabilities": {"chat": True}}]
    content = '1:{"initialModels":' + json.dumps(records, ensure_ascii=False) + ',"someNewField":true}'
    parts = [content[:26], content[26:]]
    html = ''.join('<script>self.__next_f.push(' + json.dumps([1, part], ensure_ascii=False) + ')</script>' for part in parts)
    assert arena._extract_catalog(html) == records
    assert arena._extract_catalog('<html>Just a moment...</html>') == []


def test_catalog_cache_survives_restart_with_full_model_ids(monkeypatch):
    import asyncio
    import subprocess
    records = [{"id": "model-uuid", "publicName": "test-model", "capabilities": {"chat": True}}]
    arena._save_catalog(records)
    async def unavailable(context):
        raise AssertionError("Fresh cached catalog must not access the browser")
    monkeypatch.setattr(arena, "_discover_catalog", unavailable)
    assert asyncio.run(arena._ensure_catalog(None)) == records
    env = dict(os.environ, PYTHONPATH=str(Path(arena.__file__).parent))
    code = "import asyncio, autharena_proxy as a; m=asyncio.run(a._ensure_catalog(None)); assert m[0]['id']=='model-uuid'; assert m[0]['capabilities']['chat']; print('CACHED')"
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "CACHED"


def test_catalog_refresh_failure_keeps_stale_full_metadata(monkeypatch):
    import asyncio
    records = [{"id": "model-uuid", "publicName": "test-model"}]
    arena._save("models.enc", {"fetched_at": 1, "models": records})
    before = (arena.data_dir() / "models.enc").read_bytes()
    async def unavailable(context):
        raise RuntimeError("Arena catalog page returned HTTP 403.")
    monkeypatch.setattr(arena, "_discover_catalog", unavailable)
    assert asyncio.run(arena._ensure_catalog(None)) == records
    assert (arena.data_dir() / "models.enc").read_bytes() == before
    with pytest.raises(ValueError, match="retained"):
        arena._save_catalog([])
    assert arena._load_catalog()["models"] == records


def test_catalog_without_cache_reports_actual_fetch_failure(monkeypatch):
    import asyncio
    async def unavailable(context):
        raise RuntimeError("Arena catalog page returned HTTP 403.")
    monkeypatch.setattr(arena, "_discover_catalog", unavailable)
    with pytest.raises(RuntimeError, match="no saved model IDs.*HTTP 403"):
        asyncio.run(arena._ensure_catalog(None))


def test_dispatch_ack_follows_callback_and_progress_ignores_stream_visibility(monkeypatch):
    order, logs = [], []
    class Response:
        ok = True
        status_code = 200
        headers = {}
        def iter_lines(self, **kwargs):
            for stage in ("captcha", "token", "dispatch", "headers"):
                yield 'data: ' + json.dumps({"arena_progress": stage})
            yield event({"content": "done"})
            yield event(finish="stop")
            yield "data: [DONE]"
        def raise_for_status(self):
            pass
        def close(self):
            pass
    def post(url, **kwargs):
        if url.endswith('/v1/chat/completions'):
            assert kwargs['json']['stream_timeout'] == 42.5
        if url.endswith('/dispatch'):
            assert order == ['watchdog']
            order.append('dispatch')
        return Response()
    monkeypatch.setattr(arena, "ensure_proxy_running", lambda **kwargs: {"url": "http://localhost:1", "key": "test"})
    monkeypatch.setattr(arena.requests, "post", post)
    result = arena.send_message_stream([], "test", log_fn=logs.append, log_stream=False,
        before_send_callback=lambda: order.append('watchdog'), timeout=42.5)
    assert order == ['watchdog', 'dispatch']
    assert result['content'] == 'done'
    assert any('token received' in line for line in logs)
    assert sum('API call in progress' in line for line in logs) == 1
    assert not any(line == 'done' for line in logs)


def test_expired_dispatch_is_not_reported_as_user_cancellation(monkeypatch):
    class Response:
        ok = True
        status_code = 200
        headers = {}
        def iter_lines(self, **kwargs):
            yield 'data: {"arena_progress":"dispatch"}'
        def close(self):
            pass
    def post(url, **kwargs):
        response = Response()
        if url.endswith('/dispatch'):
            response.status_code = 409
        return response
    monkeypatch.setattr(arena, "ensure_proxy_running", lambda **kwargs: {"url": "http://localhost:1", "key": "test"})
    monkeypatch.setattr(arena.requests, "post", post)
    with pytest.raises(RuntimeError, match="dispatch preparation ended") as error:
        arena.send_message_stream([], "test", log_fn=None)
    assert 'cancel' not in str(error.value).lower()


@pytest.mark.parametrize("partial", [False, True])
def test_upstream_rejection_retains_status_and_retry_after(partial):
    lines = [event({"content": "partial"})] if partial else []
    lines.append('data: ' + json.dumps({"error": {"message": 'Arena HTTP 429: {"error":"prompt failed"}',
                                                "status_code": 429, "retry_after": "60"}}))
    with pytest.raises(arena.ArenaStreamError, match="Arena HTTP 429") as error:
        arena.consume_stream(lines, log_stream=False)
    assert error.value.http_status == 429
    assert error.value.retry_after == '60'
    assert error.value.partial_response == partial
    assert 'cancel' not in str(error.value).lower()


@pytest.mark.parametrize("status", [400, 401, 403, 404, 408, 409, 413, 422, 429, 500, 502, 503, 504])
def test_local_http_errors_retain_status_and_retry_after(monkeypatch, status):
    class Response:
        ok = False
        status_code = status
        headers = {"Retry-After": "120"}
        closed = False

        def json(self):
            return {"detail": "Provider rejected the request"}

        def iter_lines(self, **kwargs):
            pytest.fail("An HTTP rejection must not be consumed as a successful stream")

        def close(self):
            self.closed = True

    response = Response()
    monkeypatch.setattr(arena, "ensure_proxy_running", lambda **kwargs: {"url": "http://localhost:1", "key": "test"})
    monkeypatch.setattr(arena.requests, "post", lambda *args, **kwargs: response)
    with pytest.raises(arena.ArenaStreamError, match=f"Arena HTTP {status}") as error:
        arena.send_message_stream([], "test", log_fn=None)
    assert error.value.http_status == status
    assert error.value.retry_after == "120"
    assert error.value.partial_response is False
    assert "Provider rejected" in str(error.value)
    assert response.closed


@pytest.mark.parametrize("body", [{"error": {"message": "Unavailable"}}, {"detail": [{"msg": "Unavailable"}]}, None])
def test_http_error_bodies_keep_status_even_without_json(body):
    class Response:
        status_code = 503
        headers = {"retry-after": "60"}
        text = "Unavailable"

        def json(self):
            if body is None:
                raise ValueError("Not JSON")
            return body

    error = arena._http_response_error(Response())
    assert error.http_status == 503
    assert error.retry_after == "60"
    assert "Unavailable" in str(error)


def test_sse_status_is_numeric_and_interrupted_reasoning_is_partial():
    lines = ['data: ' + json.dumps({"error": {"message": "Rate limited", "status_code": "429"}})]
    with pytest.raises(arena.ArenaStreamError) as error:
        arena.consume_stream(lines, log_stream=False)
    assert error.value.http_status == 429
    with pytest.raises(arena.ArenaStreamError) as partial:
        arena.consume_stream([event({"reasoning_content": "Thinking"})], log_stream=False)
    assert partial.value.partial_response is True


def test_qt_linux_environment_preserves_display_auth_and_software_rendering(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":1")
    monkeypatch.setenv("XAUTHORITY", "/tmp/test-xauthority")
    monkeypatch.setattr(arena.platform, "system", lambda: "Linux")
    env = arena._qt_browser_env(12345, visible=True, recovery=True)
    assert env["XAUTHORITY"] == "/tmp/test-xauthority"
    assert env["QT_QPA_PLATFORM"] == "xcb"
    assert env["QTWEBENGINE_REMOTE_DEBUGGING"] == "127.0.0.1:12345"
    assert "--disable-dev-shm-usage" in env["QTWEBENGINE_CHROMIUM_FLAGS"]
    assert "--disable-software-rasterizer" not in env["QTWEBENGINE_CHROMIUM_FLAGS"]
    assert arena._qt_browser_env(12345, visible=False)["QT_QPA_PLATFORM"] == "offscreen"
