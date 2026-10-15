"""Exercise the desktop install gate with process launching mocked out."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PySide6.QtWidgets import QApplication, QMainWindow

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import update_manager
from update_core import UpdateCoreMixin


@pytest.fixture
def manager(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    gui = QMainWindow()
    gui.config = {}
    gui.save_config = lambda **kwargs: None
    obj = update_manager.UpdateManager(gui, str(tmp_path))
    obj._test_gui = gui
    obj._test_app = app
    monkeypatch.setattr(UpdateCoreMixin, 'MIN_UPDATE_SIZE', 8)
    monkeypatch.setattr(update_manager.QMessageBox, 'exec', lambda self: 0)
    return obj


@pytest.mark.parametrize('case', ['missing', 'corrupt', 'wrong_size'])
def test_unverified_file_never_launches_or_exits(manager, tmp_path, monkeypatch, case):
    path = tmp_path / 'Glossarion.exe'
    data = b'installer bytes fixture'
    path.write_bytes(data)
    asset = {'size': len(data), '_sha256': hashlib.sha256(data).hexdigest()}
    manager._verified_download_asset = asset if case != 'missing' else None
    if case == 'corrupt': path.write_bytes(b'X' + data[1:])
    if case == 'wrong_size': path.write_bytes(data[:-1])
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **kw: pytest.fail('unverified installer launched'))
    monkeypatch.setattr(os, '_exit', lambda *a: pytest.fail('app exited for an unverified update'))
    monkeypatch.setattr(update_manager.QDesktopServices, 'openUrl', lambda *a: pytest.fail('unverified package opened'))
    manager.install_update(str(path))


def test_verified_windows_source_install_rehashes_before_launch(manager, tmp_path, monkeypatch):
    path = tmp_path / 'Glossarion.exe'
    data = b'installer bytes fixture'
    path.write_bytes(data)
    manager._verified_download_asset = {'size': len(data), '_sha256': hashlib.sha256(data).hexdigest()}
    calls = []
    monkeypatch.setattr(sys, 'frozen', False, raising=False)
    monkeypatch.setattr(manager, '_detect_platform', lambda: 'windows')
    monkeypatch.setattr(subprocess, 'Popen', lambda args, **kw: calls.append(('launch', args)))
    monkeypatch.setattr(os, '_exit', lambda code: calls.append(('exit', code)))
    manager.install_update(str(path))
    assert calls == [('launch', [str(path)]), ('exit', 0)]


def test_verified_macos_package_opens_without_windows_replacement(manager, tmp_path, monkeypatch):
    path = tmp_path / 'Glossarion_MAC.dmg'
    data = b'dmg bytes fixture'
    path.write_bytes(data)
    manager._verified_download_asset = {'size': len(data), '_sha256': hashlib.sha256(data).hexdigest()}
    opened = []
    monkeypatch.setattr(manager, '_detect_platform', lambda: 'macos')
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **kw: pytest.fail('Windows replacement launched'))
    monkeypatch.setattr(update_manager.QDesktopServices, 'openUrl', lambda url: opened.append(url.toLocalFile()) or True)
    manager.install_update(str(path))
    assert opened == [str(path).replace('\\', '/')]



@pytest.mark.parametrize('same_name', [True, False])
def test_frozen_install_launches_normal_exe_and_preserves_staging_until_shutdown(manager, tmp_path, monkeypatch, same_name):
    from types import SimpleNamespace
    current = tmp_path / 'Glossarion v9.14.6.exe'
    current.write_bytes(b'running old executable')
    update = Path(str(current) + '.new') if same_name else tmp_path / 'Glossarion v9.14.7.exe'
    payload = b'verified executable bytes fixture'
    update.write_bytes(payload)
    manager._verified_download_asset = {'size': len(payload), '_sha256': hashlib.sha256(payload).hexdigest()}
    monkeypatch.setattr(manager, '_detect_platform', lambda: 'windows')
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, 'executable', str(current))
    calls = []
    monkeypatch.setattr(subprocess, 'Popen', lambda args, **kw: calls.append(('launch', args)) or SimpleNamespace(pid=12345))
    monkeypatch.setattr(os, '_exit', lambda code: calls.append(('exit', code)))
    manager.install_update(str(update))
    script_path = tmp_path / 'update_glossarion.bat'
    script = script_path.read_text()
    target = current if same_name else update
    assert f'start "" "{target}"' in script
    assert f'start "" "{update}"' not in script if same_name else True
    assert calls == [('launch', ['cmd.exe', '/c', str(script_path)]), ('exit', 0)]
    # Creating the handoff must never touch the locked current file or staged bytes.
    assert current.read_bytes() == b'running old executable'
    assert update.read_bytes() == payload
    if same_name:
        assert f'move /y "{update}" "{current}"' in script
        assert 'if %update_attempt% GEQ 15 goto update_failed' in script
        assert f'del /f /q "{current}"' not in script
        assert script.index('goto start_new') < script.index('goto update_failed') < script.index(':start_new')


def test_windows_handoff_escapes_percent_paths_and_disables_delayed_expansion():
    current = r'C:\owner%name!\Glossarion.exe'
    staged = current + '.new'
    script, launch = update_manager._windows_update_script(current, staged)
    assert launch == current
    assert 'setlocal DisableDelayedExpansion' in script
    assert 'owner%%name!' in script and 'owner%name!' not in script
