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
