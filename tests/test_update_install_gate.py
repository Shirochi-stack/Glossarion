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



@pytest.mark.parametrize('frozen', [False, True])
def test_restart_environment_is_child_only_and_uses_public_pyinstaller_reset(manager, tmp_path, monkeypatch, frozen):
    from types import SimpleNamespace
    current = tmp_path / 'Glossarion.exe'
    update = Path(str(current) + '.new') if frozen else tmp_path / 'Glossarion next.exe'
    payload = b'verified independent restart fixture'
    update.write_bytes(payload)
    manager._verified_download_asset = {'size': len(payload), '_sha256': hashlib.sha256(payload).hexdigest()}
    monkeypatch.setattr(manager, '_detect_platform', lambda: 'windows')
    monkeypatch.setattr(sys, 'frozen', frozen, raising=False)
    monkeypatch.setattr(sys, 'executable', str(current))
    monkeypatch.setenv('PYINSTALLER_RESET_ENVIRONMENT', '0')
    monkeypatch.setenv('_PYI_ARCHIVE_FILE', str(current))
    monkeypatch.setenv('_PYI_APPLICATION_HOME_DIR', str(tmp_path / 'removed-old-extraction'))
    monkeypatch.setenv('_PYI_PARENT_PROCESS_LEVEL', '1')
    original = os.environ.copy()
    captured = []
    def launch(args, **kwargs):
        captured.append(kwargs.get('env'))
        return SimpleNamespace(pid=12345)
    monkeypatch.setattr(subprocess, 'Popen', launch)
    monkeypatch.setattr(os, '_exit', lambda code: None)
    manager.install_update(str(update))
    assert len(captured) == 1
    environment = captured[0]
    assert environment is not os.environ
    assert environment.get('PYINSTALLER_RESET_ENVIRONMENT') == '1'
    # Let the bootloader reset its private state; never rewrite those internals ourselves.
    assert all(environment.get(key) == value for key, value in original.items()
               if key != 'PYINSTALLER_RESET_ENVIRONMENT')
    assert all(os.environ.get(key) == value for key, value in original.items())
    assert set(os.environ) == set(original)


@pytest.mark.skipif(sys.platform != 'win32', reason='Exercises the real detached Windows command handoff')
def test_real_detached_handoff_survives_parent_and_passes_restart_environment(tmp_path):
    """Real cmd/move/Popen/exit; safe Python probe replaces only the final EXE launch.

    This proves shell/environment/lifetime behavior, not frozen bootloader startup.
    It never runs Glossarion, builds a binary, or touches src/dist.
    """
    import json
    import time
    actual_python = sys.executable
    source_dir = str(Path(update_manager.__file__).parent)
    current = tmp_path / 'Glossarion fixture.exe'
    staged = Path(str(current) + '.new')
    current.write_bytes(b'old fixture only')
    payload = b'verified replacement fixture only'
    staged.write_bytes(payload)
    output = tmp_path / 'child-result.json'
    probe = tmp_path / 'probe.py'
    probe.write_text("""import ctypes,json,os,sys
from pathlib import Path
k=ctypes.windll.kernel32
k.OpenProcess.restype=ctypes.c_void_p
k.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
k.CloseHandle.argtypes=[ctypes.c_void_p]
h=k.OpenProcess(0x00100000,False,int(sys.argv[2]))
alive=bool(h and k.WaitForSingleObject(h,0)==258)
if h:k.CloseHandle(h)
Path(sys.argv[1]).write_text(json.dumps({'parent_still_alive':alive,'reset':os.environ.get('PYINSTALLER_RESET_ENVIRONMENT'),'old_home':os.environ.get('_PYI_APPLICATION_HOME_DIR')}))
""", encoding='utf-8')
    launcher = tmp_path / 'launcher.py'
    launcher.write_text(f"""import hashlib,os,subprocess,sys
sys.path.insert(0,{source_dir!r})
from PySide6.QtWidgets import QApplication,QMainWindow
import update_manager
app=QApplication(['handoff-test','-platform','offscreen'])
gui=QMainWindow();gui.config={{}};gui.save_config=lambda **kw:None
manager=update_manager.UpdateManager(gui,{str(tmp_path)!r})
manager._detect_platform=lambda:'windows'
update_manager.UpdateCoreMixin.MIN_UPDATE_SIZE=8
manager._verified_download_asset={{'size':{len(payload)},'_sha256':{hashlib.sha256(payload).hexdigest()!r}}}
original_script=update_manager._windows_update_script
def safe_script(current,update):
    script,target=original_script(current,update)
    command=subprocess.list2cmdline([{actual_python!r},'-B',{str(probe)!r},{str(output)!r},str(os.getpid())])
    expected='start "" "'+target+'"'
    assert script.count(expected)==1
    return script.replace(expected,'start "" /wait '+command),target
update_manager._windows_update_script=safe_script
sys.frozen=True;sys.executable={str(current)!r}
manager.install_update({str(staged)!r})
raise SystemExit('install_update did not exit')
""", encoding='utf-8')
    environment = os.environ.copy()
    environment.update({'PYINSTALLER_RESET_ENVIRONMENT': '0', '_PYI_ARCHIVE_FILE': str(current),
                        '_PYI_APPLICATION_HOME_DIR': str(tmp_path / 'removed-old-extraction'),
                        '_PYI_PARENT_PROCESS_LEVEL': '1'})
    result = subprocess.run([actual_python, '-B', str(launcher)], env=environment,
                            capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr.decode(errors='replace')[-1000:]
    deadline = time.monotonic() + 15
    while not output.exists() and time.monotonic() < deadline:
        time.sleep(.05)
    assert output.exists(), 'Detached handoff child did not finish'
    evidence = json.loads(output.read_text())
    assert evidence['reset'] == '1'
    assert evidence['parent_still_alive'] is False
    assert evidence['old_home'] == environment['_PYI_APPLICATION_HOME_DIR']
    assert not Path(evidence['old_home']).exists()
    assert current.read_bytes() == payload and not staged.exists()
