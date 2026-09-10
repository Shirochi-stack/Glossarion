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
        assert logged == [" hello "]
        yield event({"reasoning_content": "thinking"})
        assert logged[-1] == "thinking"
        yield event({"content": "world\n"})
        yield event(finish="length", usage={"total_tokens": 10})
        yield "data: [DONE]"
    result = arena.consume_stream(lines(), logged.append)
    assert result["content"] == " hello world\n"
    assert result["finish_reason"] == "length"
    assert result["usage"] == {"total_tokens": 10}


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
    (runtime / "browser-ready").write_text(str(browser), encoding="utf-8")
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
    assert control.login_button.text() == "Arena Login"
    assert control.accounts.isHidden()
    model[0] = "autharena0/model"
    control.refresh()
    assert control.accounts.isHidden()
    model[0] = "autharena1/model"
    control.refresh()
    assert not control.accounts.isHidden()
    assert [control.accounts.itemText(i) for i in range(control.accounts.count())] == ["#0", "#1", "+ New"]
    assert control.accounts.currentData() == 1
    control.select_account(0)
    assert model[0] == "autharena/model"
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
    assert control.login_button.text() == "Arena Login"
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
    assert calls[0] == ["managed-python", "-m", "playwright", "install", "chromium"]
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
    assert not (tmp_path / "browser-ready").exists()


@pytest.mark.parametrize("version", [None, 1])
def test_previous_external_browser_worker_is_detected(monkeypatch, version):
    from types import SimpleNamespace
    monkeypatch.setattr(arena, "_load", lambda name: {"url": "http://127.0.0.1:1234", "key": "test"})
    monkeypatch.setattr(arena.requests, "get", lambda *args, **kwargs: SimpleNamespace(
        ok=True, json=lambda: {"revision": arena.REVISION, "adapter_version": version}))
    status = arena.check_proxy_health()
    assert status["outdated"] is True
    assert status["running"] is False


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
