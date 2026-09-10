"""Installer contracts without launching or controlling a personal browser."""
import base64
import json
import os
from pathlib import Path
import subprocess
import shutil

import pytest

import autharena_setup as setup


@pytest.fixture
def extension(tmp_path):
    folder = tmp_path / "Arena's helper $ & folder"
    folder.mkdir()
    for name in setup._REQUIRED_FILES:
        (folder / name).write_text('// helper', encoding='utf-8')
    (folder / 'manifest.json').write_text(json.dumps({
        'name': 'Glossarion Arena Browser Companion', 'manifest_version': 3,
    }), encoding='utf-8')
    return folder


def forbid_process(*args, **kwargs):
    pytest.fail('This path must not launch a process')


@pytest.mark.parametrize('platform', ['linux', 'darwin'])
def test_other_platforms_give_manual_instructions(extension, monkeypatch, platform):
    monkeypatch.setattr(setup.sys, 'platform', platform)
    monkeypatch.setattr(setup.subprocess, 'Popen', forbid_process)
    result = setup.install_extension(extension, 'chrome')
    assert result['status'] == 'manual_required'
    assert 'Load unpacked' in result['message']


@pytest.mark.parametrize('hint', ['firefox', '--load-extension=x', 'chrome.exe', 'edge://extensions'])
def test_invalid_browser_hint_cannot_launch_anything(extension, monkeypatch, hint):
    monkeypatch.setattr(setup.subprocess, 'Popen', forbid_process)
    assert setup.install_extension(extension, hint)['status'] == 'error'


@pytest.mark.parametrize('missing', setup._REQUIRED_FILES)
def test_incomplete_extension_does_not_launch_browser(extension, monkeypatch, missing):
    (extension / missing).unlink()
    monkeypatch.setattr(setup.subprocess, 'Popen', forbid_process)
    assert setup.install_extension(extension)['status'] == 'manual_required'


def test_cancelled_install_does_not_open_browser(extension, monkeypatch):
    monkeypatch.setattr(setup.subprocess, 'Popen', forbid_process)
    assert setup.install_extension(extension, cancel_check=lambda: True)['status'] == 'cancelled'


@pytest.mark.parametrize('family,url', [('chrome', 'chrome://extensions/'), ('edge', 'edge://extensions/')])
def test_install_reuses_normal_browser_without_profile_or_security_flags(extension, monkeypatch, family, url):
    monkeypatch.setattr(setup.sys, 'platform', 'win32')
    executable = Path('C:/Apps') / ('chrome.exe' if family == 'chrome' else 'msedge.exe')
    monkeypatch.setattr(setup, '_find_browser', lambda hint: (executable, url))
    launched = []
    monkeypatch.setattr(setup.subprocess, 'Popen', lambda args, **kwargs: launched.append((args, kwargs)))
    automated = []
    def run(folder, browser, *, cancel_check):
        automated.append((folder, browser, cancel_check()))
        return {'status': 'awaiting_connection', 'message': 'Submitted, not yet paired.'}
    monkeypatch.setattr(setup, '_run_windows_installer', run)
    assert setup.install_extension(extension, family)['status'] == 'awaiting_connection'
    assert launched[0][0] == [str(executable), url]
    assert automated == [(extension.resolve(), executable, False)]


def test_unknown_default_browser_does_not_silently_switch_profiles(extension, monkeypatch):
    monkeypatch.setattr(setup.sys, 'platform', 'win32')
    monkeypatch.setattr(setup, '_find_browser', lambda hint: None)
    monkeypatch.setattr(setup.subprocess, 'Popen', forbid_process)
    assert setup.install_extension(extension)['status'] == 'manual_required'


def test_browser_launch_failure_is_manual_fallback(extension, monkeypatch):
    monkeypatch.setattr(setup.sys, 'platform', 'win32')
    monkeypatch.setattr(setup, '_find_browser', lambda hint: (Path('chrome.exe'), 'chrome://extensions/'))
    def fail(*args, **kwargs):
        raise OSError('blocked launch')
    monkeypatch.setattr(setup.subprocess, 'Popen', fail)
    assert setup.install_extension(extension)['status'] == 'manual_required'


class FakeInstaller:
    def __init__(self, output=b'', timeout=False):
        self.output = output
        self.timeout = timeout
        self.killed = False
        self.completed = False

    def communicate(self, timeout=None):
        if self.timeout and timeout is not None:
            raise subprocess.TimeoutExpired('installer', timeout)
        self.completed = True
        return self.output, b''

    def poll(self):
        return 0 if self.killed or self.completed else None

    def kill(self):
        self.killed = True


def test_powershell_uses_static_script_and_paths_as_data(extension, monkeypatch):
    process = FakeInstaller(b'{"status":"awaiting_connection","message":"Submitted"}')
    launches = []
    def popen(args, **kwargs):
        launches.append((args, kwargs))
        return process
    monkeypatch.setattr(setup.subprocess, 'Popen', popen)
    result = setup._run_windows_installer(extension, Path('C:/Apps/chrome.exe'), cancel_check=lambda: False)
    args, options = launches[0]
    assert len(subprocess.list2cmdline(args)) < 8191
    assert 'GLOSSARION_ARENA_SETUP_SCRIPT' in base64.b64decode(args[-1]).decode('utf-16-le')
    assert base64.b64decode(options['env']['GLOSSARION_ARENA_SETUP_SCRIPT']).decode('utf-8') == setup._INSTALL_SCRIPT
    assert str(extension) not in setup._INSTALL_SCRIPT
    assert options['env']['GLOSSARION_ARENA_SETUP_FOLDER'] == str(extension)
    assert options['env']['GLOSSARION_ARENA_SETUP_URL'] == 'chrome://extensions/'
    assert args[args.index('-WindowStyle') + 1] == 'Hidden'
    assert result['status'] == 'awaiting_connection'
    assert not process.killed


def test_cancellation_terminates_only_owned_installer(extension, monkeypatch):
    process = FakeInstaller(timeout=True)
    monkeypatch.setattr(setup.subprocess, 'Popen', lambda *args, **kwargs: process)
    result = setup._run_windows_installer(extension, Path('chrome.exe'), cancel_check=lambda: True)
    assert result['status'] == 'cancelled'
    assert process.killed


def test_timeout_terminates_only_owned_installer(extension, monkeypatch):
    process = FakeInstaller(timeout=True)
    monkeypatch.setattr(setup.subprocess, 'Popen', lambda *args, **kwargs: process)
    result = setup._run_windows_installer(extension, Path('chrome.exe'), cancel_check=lambda: False, timeout=0)
    assert result['status'] == 'manual_required'
    assert process.killed


@pytest.mark.parametrize('output', [b'', b'not JSON', b'{"status":"installed","message":"unverified"}'])
def test_unrecognized_installer_output_never_claims_success(extension, monkeypatch, output):
    process = FakeInstaller(output)
    monkeypatch.setattr(setup.subprocess, 'Popen', lambda *args, **kwargs: process)
    result = setup._run_windows_installer(extension, Path('chrome.exe'), cancel_check=lambda: False)
    assert result['status'] == 'manual_required'


def test_open_folder_uses_exact_prepared_path(extension, monkeypatch):
    opened = []
    monkeypatch.setattr(setup.sys, 'platform', 'win32')
    monkeypatch.setattr(setup.os, 'startfile', opened.append, raising=False)
    assert setup.open_extension_folder(extension)['status'] == 'opened'
    assert opened == [str(extension.resolve())]


def test_windows_installer_script_parses_without_running_ui_automation():
    powershell = shutil.which('powershell.exe')
    if powershell is None:
        pytest.skip('PowerShell parser is only checked when installed')
    # ParseInput builds an AST only. The installer script is never invoked.
    data = base64.b64encode(setup._INSTALL_SCRIPT.encode('utf-8')).decode('ascii')
    command = (
        "$tokens = $null; $errors = $null; "
        "$source = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($env:ARENA_SCRIPT_PARSE_TEST)); "
        "[void][System.Management.Automation.Language.Parser]::ParseInput($source, [ref]$tokens, [ref]$errors); "
        "if ($errors.Count) { $errors | ForEach-Object { $_.Message }; exit 1 }"
    )
    encoded = base64.b64encode(command.encode('utf-16-le')).decode('ascii')
    result = subprocess.run(
        [powershell, '-NoProfile', '-NonInteractive', '-WindowStyle', 'Hidden', '-EncodedCommand', encoded],
        capture_output=True, text=True, timeout=15,
        env=dict(os.environ, ARENA_SCRIPT_PARSE_TEST=data),
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_installer_process_roundtrip_without_browser_automation(extension, monkeypatch):
    if setup.sys.platform != 'win32' or shutil.which('powershell.exe') is None:
        pytest.skip('Windows PowerShell process check')
    # Exercise the real hidden child-process/bootstrap and environment encoding
    # with a harmless script. No UIA code or browser executable is invoked.
    monkeypatch.setattr(setup, '_INSTALL_SCRIPT', """
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
@{status='awaiting_connection'; message=$env:GLOSSARION_ARENA_SETUP_FOLDER} | ConvertTo-Json -Compress
""")
    result = setup._run_windows_installer(extension, Path('chrome.exe'), cancel_check=lambda: False, timeout=10)
    assert result == {'status': 'awaiting_connection', 'message': str(extension)}
