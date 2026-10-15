"""Actual Qt event-loop tests with delayed/offline providers; no real network/install."""
import json
import os
from pathlib import Path
import sys
import threading
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
import requests
from PySide6.QtCore import QThread, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import update_manager
from update_core import UpdateCoreMixin

NAME = 'Glossarion v9.15.0.exe'
ASSET = 'https://github.com/Shirochi-stack/Glossarion/releases/download/v9.15.0/' + NAME
GH = UpdateCoreMixin.GITHUB_LATEST_URL
GL = UpdateCoreMixin.GITLAB_API_URL


class Response:
    headers = {}
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def iter_content(self, size): yield json.dumps(self.payload).encode()
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError('HTTP ' + str(self.status_code), response=self)


def release():
    return {'tag_name': 'v9.15.0', 'body': 'notes', 'assets': [
        {'name': NAME, 'size': 60_000_000, 'digest': 'sha256:' + '0' * 64, 'browser_download_url': ASSET}]}


def pump_until(condition, timeout=3):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        QTest.qWait(10)
    assert condition(), 'Qt check failed to finish within the bounded test interval'


@pytest.fixture
def ui(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    gui = QMainWindow()
    gui.config = {}
    gui.setWindowTitle('Glossarion v9.14.5')
    button = QPushButton('Still interactive', gui)
    gui.setCentralWidget(button)
    events = {'ticks': 0, 'clicks': 0, 'shown': [], 'messages': [], 'saves': []}
    gui.save_config = lambda **kw: events['saves'].append(threading.get_ident())
    button.clicked.connect(lambda: events.__setitem__('clicks', events['clicks'] + 1))
    manager = update_manager.UpdateManager(gui, str(tmp_path))
    monkeypatch.setattr(manager, '_detect_platform', lambda: 'windows')
    monkeypatch.setattr(manager, '_detect_arch', lambda: 'x64')
    monkeypatch.setattr(manager, 'show_update_dialog', lambda: events['shown'].append(threading.get_ident()))
    timer = QTimer()
    timer.timeout.connect(lambda: events.__setitem__('ticks', events['ticks'] + 1))
    timer.start(10)
    original_exec = update_manager.QMessageBox.exec
    def message_exec(box):
        assert QThread.currentThread() == app.thread(), 'Message box created/executed off the GUI thread'
        events['messages'].append((threading.get_ident(), box.text()))
        QTimer.singleShot(80, box.accept)
        return original_exec(box)
    monkeypatch.setattr(update_manager.QMessageBox, 'exec', message_exec)
    gui.show()
    yield app, gui, button, manager, events
    if manager.worker is not None:
        manager.cancel_update_check()
        pump_until(lambda: not manager._check_in_progress)
    timer.stop()
    manager._close_loading_dialog()
    gui.close()
    app.processEvents()


@pytest.mark.parametrize('blocked_stage', ['github', 'gitlab', 'checksum', 'history'])
def test_slow_network_keeps_qt_timers_and_main_window_interactive(ui, monkeypatch, blocked_stage):
    app, gui, button, manager, events = ui
    main = threading.get_ident()
    entered, gate = threading.Event(), threading.Event()
    calls = []
    def get(url, **kwargs):
        assert threading.get_ident() != main, 'HTTP ran on GUI thread'
        calls.append((url, threading.get_ident()))
        assert kwargs['timeout'][0] <= 3 and kwargs['timeout'][1] <= 5
        should_block = ((blocked_stage == 'github' and url == GH)
            or (blocked_stage == 'gitlab' and '/repository/tags?' in url)
            or (blocked_stage == 'checksum' and url.endswith('.sha256.json'))
            or (blocked_stage == 'history' and 'github.com' in url and '?per_page=' in url))
        if should_block:
            entered.set()
            assert gate.wait(3), 'Test failed to release the fake network'
        if blocked_stage in ('github', 'history'):
            return Response(release() if url == GH else [])
        if url == GH: return Response({}, 404)
        if '/repository/tags?' in url:
            return Response([{'name': 'v9.15.0', 'release': {'tag_name': 'v9.15.0'}}])
        if url.endswith('/releases/v9.15.0'):
            return Response({'tag_name': 'v9.15.0', 'description': 'notes', 'assets': {'links': [
                {'name': NAME, 'url': GL + '/packages/generic/glossarion/v9.15.0/' + NAME},
                {'name': NAME + '.sha256.json', 'url': GL + '/metadata.sha256.json'}]}})
        if url.endswith('.sha256.json'):
            return Response({'schema_version': 1, 'tag': 'v9.15.0', 'name': NAME,
                             'size': 60_000_000, 'sha256': '0' * 64})
        raise AssertionError(url)
    monkeypatch.setattr(requests, 'get', get)
    started = time.monotonic()
    manager.check_for_updates_manual()
    assert time.monotonic() - started < .25
    worker = manager.worker
    try:
        pump_until(entered.is_set)
        ticks = events['ticks']
        QTest.qWait(150)
        QTest.mouseClick(button, Qt.LeftButton)
        assert events['ticks'] - ticks >= 5
        assert gui.isEnabled() and events['clicks'] == 1
        assert not manager.loading_dialog.isModal()
        for _ in range(10): manager.check_for_updates_manual()
        assert manager.worker is worker
    finally:
        gate.set()
    pump_until(lambda: not manager._check_in_progress)
    assert events['shown'] == [main]
    assert events['messages'] == []
    assert events['saves'] == [main]
    assert manager.latest_release['tag_name'] == 'v9.15.0'
    assert all(thread != main for _, thread in calls)


def test_screenshot_404_path_reports_error_on_main_thread_without_sync_history_retry(ui, monkeypatch):
    app, gui, button, manager, events = ui
    main, calls = threading.get_ident(), []
    def get(url, **kwargs):
        calls.append((url, threading.get_ident()))
        assert threading.get_ident() != main
        return Response({}, 404)
    monkeypatch.setattr(requests, 'get', get)
    manager.check_for_updates_manual()
    pump_until(lambda: not manager._check_in_progress)
    assert len(events['messages']) == 1
    assert events['messages'][0][0] == main
    assert '404' in events['messages'][0][1]
    assert events['shown'] == []
    assert len(calls) == 2  # GitHub latest, GitLab tags; no GUI-thread history retry
    assert events['ticks'] >= 3  # actual error dialog's nested event loop is responsive
    assert gui.isEnabled()


@pytest.mark.parametrize('failure', [requests.Timeout('slow'), requests.ConnectionError('offline')])
def test_unreachable_providers_do_not_hang_the_gui(ui, monkeypatch, failure):
    app, gui, button, manager, events = ui
    main = threading.get_ident()
    def get(url, **kwargs):
        assert threading.get_ident() != main
        time.sleep(.05)
        raise failure
    monkeypatch.setattr(requests, 'get', get)
    manager.check_for_updates_manual()
    pump_until(lambda: not manager._check_in_progress)
    assert events['ticks'] >= 5 and len(events['messages']) == 1
    assert events['shown'] == [] and gui.isEnabled()


@pytest.mark.parametrize('cancel_action', ['button', 'close'])
def test_cancel_repeated_clicks_and_late_success_then_new_check(ui, monkeypatch, cancel_action):
    app, gui, button, manager, events = ui
    main = threading.get_ident()
    gate, entered = threading.Event(), threading.Event()
    calls = []
    def get(url, **kwargs):
        calls.append((url, threading.get_ident()))
        if url == GH:
            entered.set()
            assert gate.wait(3)
            return Response(release())
        return Response([])
    monkeypatch.setattr(requests, 'get', get)
    manager.check_for_updates_manual()
    pump_until(entered.is_set)
    worker = manager.worker
    cancel = manager.loading_dialog.findChild(QPushButton, 'cancel_update_check')
    if cancel_action == 'button':
        QTest.mouseClick(cancel, Qt.LeftButton)
    else:
        manager.loading_dialog.close()
    assert manager.loading_dialog is None and gui.isEnabled()
    for _ in range(10): manager.check_for_updates_manual()
    assert manager.worker is worker and len(calls) == 1
    gate.set()
    pump_until(lambda: not manager._check_in_progress)
    assert not events['shown'] and not events['messages'] and not events['saves']
    assert 'last_update_check_time' not in gui.config
    manager.check_for_updates_manual()
    pump_until(lambda: not manager._check_in_progress)
    assert events['shown'] == [main] and events['saves'] == [main]


def test_empty_history_entry_point_never_fetches_synchronously(ui, monkeypatch):
    app, gui, button, manager, events = ui
    # Call the real entry point, since the fixture only intercepts successful rendering.
    main = threading.get_ident()
    def get(url, **kwargs):
        assert threading.get_ident() != main
        return Response(release() if url == GH else [])
    monkeypatch.setattr(requests, 'get', get)
    update_manager.UpdateManager.show_update_dialog(manager)
    pump_until(lambda: not manager._check_in_progress)
    assert events['shown'] == [main]


def test_metadata_check_time_budget_does_not_launch_unbounded_fallback(ui, monkeypatch):
    app, gui, button, manager, events = ui
    manager._update_check_timeout = .05
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        time.sleep(.08)
        return Response({}, 404)
    monkeypatch.setattr(requests, 'get', get)
    manager.check_for_updates_manual()
    pump_until(lambda: not manager._check_in_progress)
    assert calls == [GH]
    assert len(events['messages']) == 1 and gui.isEnabled()


def test_cached_background_check_preserves_previous_release_without_writes(ui, monkeypatch):
    app, gui, button, manager, events = ui
    manager.latest_release = release()
    manager.all_releases = [manager.latest_release]
    manager.update_available = True
    manager._last_check_time = time.time()
    gui.config['last_update_check_time'] = manager._last_check_time
    monkeypatch.setattr(requests, 'get', lambda *a, **kw: pytest.fail('cached check performed HTTP'))
    manager.check_for_updates_async(silent=True)
    pump_until(lambda: not manager._check_in_progress)
    assert manager.latest_release['tag_name'] == 'v9.15.0' and manager.update_available
    assert not events['saves'] and not events['shown'] and not events['messages']
