"""Exercise backup restore without touching user config or restarting the runner."""
import importlib.util
import ast
import json
import os
from pathlib import Path
import sys
import types

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication, QDialog, QMainWindow, QMessageBox, QPushButton

import shutdown_utils


@pytest.fixture
def backup_module(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / 'src' / 'config_backup.py'
    spec = importlib.util.spec_from_file_location('_config_backup_under_test', path)
    module = importlib.util.module_from_spec(spec)
    # Omit the unrelated full application startup imported for these constants.
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, 'translator_gui', types.SimpleNamespace(
            CONFIG_FILE=str(tmp_path / 'config.json'), decrypt_config=lambda value: value,
        ))
        spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize('frozen', [False, True])
def test_restore_is_atomic_preserves_bytes_and_builds_restart_command(backup_module, tmp_path, monkeypatch, frozen):
    config = Path(backup_module.CONFIG_FILE)
    config.write_text('{"current": true}', encoding='utf-8')
    backup = tmp_path / 'backup.json.bak'
    restored = b'{"restored": true, "encrypted_key": "unchanged"}'
    backup.write_bytes(restored)
    safety = []

    def create_safety_backup():
        safety.append(config.read_bytes())
        backup.unlink()  # Age-based cleanup can remove the selected backup.

    owner = types.SimpleNamespace(_backup_config_file=create_safety_backup)
    monkeypatch.setattr(sys, 'frozen', frozen, raising=False)
    monkeypatch.setattr(sys, 'argv', ['app.py', '--example'])
    backup_module._restore_config_backup_file(owner, str(backup))
    assert config.read_bytes() == restored
    assert safety == [b'{"current": true}']
    assert owner._config_restore_pending
    assert owner._restart_command == [sys.executable, *(['--example'] if frozen else sys.argv)]
    assert not list(tmp_path.glob('.config_restore_*'))


@pytest.mark.parametrize('failure', ['invalid_json', 'invalid_shape', 'replace_error'])
def test_failed_restore_leaves_current_config_intact(backup_module, tmp_path, monkeypatch, failure):
    config = Path(backup_module.CONFIG_FILE)
    config.write_bytes(b'{"current": true}')
    backup = tmp_path / 'backup.json.bak'
    backup.write_bytes({'invalid_json': b'{', 'invalid_shape': b'[]'}.get(failure, b'{}'))
    owner = types.SimpleNamespace(_backup_config_file=lambda: None)
    if failure == 'replace_error':
        def fail_replace(*args):
            raise OSError('simulated locked config')
        monkeypatch.setattr(backup_module.os, 'replace', fail_replace)
    with pytest.raises((ValueError, OSError)):
        backup_module._restore_config_backup_file(owner, str(backup))
    assert config.read_bytes() == b'{"current": true}'
    assert not getattr(owner, '_config_restore_pending', False)
    assert not list(tmp_path.glob('.config_restore_*'))


def test_restore_dialog_and_callbacks_stay_on_gui_thread(backup_module, tmp_path, monkeypatch, qapp):
    config = Path(backup_module.CONFIG_FILE)
    config.write_text('{}', encoding='utf-8')
    backups = tmp_path / 'config_backups'
    backups.mkdir()
    (backups / 'config_20260905_010000.json.bak').write_text('{"restored": true}', encoding='utf-8')
    window = QMainWindow()
    window._backup_config_file = lambda: None
    closed = []
    window.close = lambda: closed.append(QThread.currentThread())
    messages = []

    def message_exec(message):
        assert QThread.currentThread() == qapp.thread()
        messages.append(message.windowTitle())
        if message.windowTitle() == 'Restore Complete':
            assert window._config_restore_pending
        return QMessageBox.Yes

    monkeypatch.setattr(QMessageBox, 'exec', message_exec)
    def forbidden_exec(*args):
        pytest.fail('Backup manager must not run a nested/background event loop')
    monkeypatch.setattr(QDialog, 'exec', forbidden_exec)
    try:
        backup_module._manual_restore_config(window)
        dialog = window._backup_dialog
        assert dialog.isVisible()
        assert dialog.thread() == qapp.thread()
        backup_module._manual_restore_config(window)
        assert window._backup_dialog is dialog
        button = next(b for b in dialog.findChildren(QPushButton) if 'Restore' in b.text())
        button.click()
        qapp.processEvents()
        assert messages == ['Confirm Restore', 'Restore Complete']
        assert json.loads(config.read_text()) == {'restored': True}
        assert closed == [qapp.thread()]
        assert window._backup_dialog is None
    finally:
        window.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize('frozen', [False, True])
def test_restart_launches_after_cleanup_before_exit(monkeypatch, frozen):
    calls = []
    for name in ('_ensure_safe_tempdir', '_run_cleanup_fns', 'drain_qt_events_for_shutdown',
                 'cleanup_browser_generated_state_for_shutdown', '_cleanup_pyinstaller_temp_dir',
                 '_taskkill_self_tree'):
        monkeypatch.setattr(shutdown_utils, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(shutdown_utils, '_terminate_all_children_for_shutdown', lambda **kwargs: calls.append('cleanup'))
    monkeypatch.setattr(sys, 'frozen', frozen, raising=False)
    monkeypatch.delenv('PYINSTALLER_RESET_ENVIRONMENT', raising=False)
    def launch(command, **kwargs):
        calls.append('launch')
        assert command == ['application', '--example']
        assert kwargs['env'].get('PYINSTALLER_RESET_ENVIRONMENT') == ('1' if frozen else None)
    monkeypatch.setattr(shutdown_utils.subprocess, 'Popen', launch)
    def exit_process(code):
        calls.append('exit')
        raise SystemExit(code)
    monkeypatch.setattr(shutdown_utils.os, '_exit', exit_process)
    with pytest.raises(SystemExit):
        shutdown_utils.force_shutdown(0, cleanup_epub_reader_caches=False,
                                      restart_command=['application', '--example'])
    assert calls == ['cleanup', 'launch', 'exit']


@pytest.mark.parametrize('filename, method_name', [
    ('translator_gui.py', 'save_config'), ('other_settings.py', 'save_profiles'),
])
def test_pending_restore_blocks_config_savers(filename, method_name):
    path = Path(__file__).resolve().parents[1] / 'src' / filename
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    if filename == 'translator_gui.py':
        tree = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'TranslatorGUI')
    method = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    accesses = []
    class PendingRestore:
        _config_restore_pending = True
        def __getattr__(self, name):
            accesses.append(name)
            raise AttributeError(name)
    assert namespace[method_name](PendingRestore()) is False
    assert accesses == []
