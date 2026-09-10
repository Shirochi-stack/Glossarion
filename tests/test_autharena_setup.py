"""Installer contracts without launching or controlling a personal browser."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import shutil

import pytest

import autharena_setup as setup


CONNECT_URL = 'http://127.0.0.1:43187/connect#test-connection_nonce_1234567890'


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
    assert setup.install_extension(extension)['status'] == 'error'


def test_cancelled_install_does_not_open_browser(extension, monkeypatch):
    monkeypatch.setattr(setup.subprocess, 'Popen', forbid_process)
    assert setup.install_extension(extension, cancel_check=lambda: True)['status'] == 'cancelled'


@pytest.mark.parametrize('family', ['', 'chrome', 'edge'])
def test_install_passes_exact_connection_context_without_launching_browser(extension, monkeypatch, family):
    monkeypatch.setattr(setup.sys, 'platform', 'win32')
    monkeypatch.setattr(setup.subprocess, 'Popen', forbid_process)
    automated = []
    progress = lambda message: None

    def run(folder, browser_hint, *, connect_url, cancel_check, progress=None):
        automated.append((folder, browser_hint, connect_url, cancel_check(), progress))
        return {'status': 'awaiting_connection', 'message': 'Submitted, not yet paired.'}

    monkeypatch.setattr(setup, '_run_windows_installer', run)
    result = setup.install_extension(extension, family, connect_url=CONNECT_URL, progress=progress)
    assert result['status'] == 'awaiting_connection'
    assert automated == [(extension.resolve(), family, CONNECT_URL, False, progress)]


@pytest.mark.parametrize('connect_url', [
    '', 'https://arena.ai/', 'http://example.com/connect#test-connection_nonce_1234567890',
    'http://127.0.0.1:43187/connect', 'http://127.0.0.1:43187/connect#short',
    'http://127.0.0.1:43187/other#test-connection_nonce_1234567890',
    'http://127.0.0.1:43187/connect?redirect=1#test-connection_nonce_1234567890',
    'http://user:password@127.0.0.1:43187/connect#test-connection_nonce_1234567890',
    'http://127.0.0.1:0/connect#test-connection_nonce_1234567890',
    'http://127.0.0.1:65536/connect#test-connection_nonce_1234567890',
    'http://localhost:43187/connect#test-connection_nonce_1234567890',
    'http://127.0.0.1:043187/connect#test-connection_nonce_1234567890',
    ' http://127.0.0.1:43187/connect#test-connection_nonce_1234567890',
    None,
])
def test_windows_setup_requires_exact_loopback_connection_context(extension, monkeypatch, connect_url):
    monkeypatch.setattr(setup.sys, 'platform', 'win32')
    monkeypatch.setattr(setup.subprocess, 'Popen', forbid_process)
    monkeypatch.setattr(setup, '_run_windows_installer', forbid_process)
    assert setup.install_extension(extension, connect_url=connect_url)['status'] in {'error', 'manual_required'}


def test_hidden_installer_launch_failure_is_explicit_error(extension, monkeypatch):
    monkeypatch.setattr(setup.sys, 'platform', 'win32')

    def fail(*args, **kwargs):
        raise OSError('blocked launch')

    monkeypatch.setattr(setup.subprocess, 'Popen', fail)
    assert setup.install_extension(extension, connect_url=CONNECT_URL)['status'] == 'error'


class FakeInstaller:
    def __init__(self, output=b'', timeout=False):
        self.output = output
        self.timeout = timeout
        self.killed = False
        self.completed = not timeout
        self.stdout = io.BytesIO(output)
        self.stderr = io.BytesIO()

    def communicate(self, timeout=None):
        if self.timeout and timeout is not None:
            raise subprocess.TimeoutExpired('installer', timeout)
        self.completed = True
        return self.output, b''

    def poll(self):
        return 0 if self.killed or self.completed else None

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        if self.timeout and not self.killed:
            raise subprocess.TimeoutExpired('installer', timeout)
        self.completed = True
        return 0


def test_powershell_uses_static_script_and_paths_as_data(extension, monkeypatch):
    process = FakeInstaller(b'{"status":"awaiting_connection","message":"Submitted"}')
    launches = []
    def popen(args, **kwargs):
        launches.append((args, kwargs))
        return process
    monkeypatch.setattr(setup.subprocess, 'Popen', popen)
    result = setup._run_windows_installer(extension, 'chrome', connect_url=CONNECT_URL, cancel_check=lambda: False)
    args, options = launches[0]
    assert len(launches) == 1
    assert Path(args[0]).name.lower() == 'powershell.exe'
    assert len(subprocess.list2cmdline(args)) < 8191
    assert 'GLOSSARION_ARENA_SETUP_SCRIPT' in base64.b64decode(args[-1]).decode('utf-16-le')
    assert base64.b64decode(options['env']['GLOSSARION_ARENA_SETUP_SCRIPT']).decode('utf-8') == setup._INSTALL_SCRIPT
    assert str(extension) not in setup._INSTALL_SCRIPT
    assert options['env']['GLOSSARION_ARENA_SETUP_FOLDER'] == str(extension)
    assert options['env']['GLOSSARION_ARENA_SETUP_CONNECT_URL'] == CONNECT_URL
    assert options['env']['GLOSSARION_ARENA_SETUP_BROWSER_HINT'] == 'chrome'
    assert CONNECT_URL not in subprocess.list2cmdline(args)
    assert CONNECT_URL not in setup._INSTALL_SCRIPT
    assert args[args.index('-WindowStyle') + 1] == 'Hidden'
    assert options['creationflags'] == getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    assert result['status'] == 'awaiting_connection'
    assert not process.killed


def test_progress_records_are_delivered_before_terminal_result(extension, monkeypatch):
    updates = []
    process = FakeInstaller(
        b'Ignored library banner\n'
        b'{"status":"running","message":"Confirming current connection tab"}\n'
        b'{"status":"running","message":"Opening Extensions in this window"}\n'
        b'{"status":"awaiting_connection","message":"Waiting for helper"}\n'
    )
    monkeypatch.setattr(setup.subprocess, 'Popen', lambda *args, **kwargs: process)
    result = setup._run_windows_installer(
        extension, '', connect_url=CONNECT_URL, cancel_check=lambda: False,
        progress=updates.append,
    )
    assert updates == [
        {'status': 'running', 'message': 'Confirming current connection tab'},
        {'status': 'running', 'message': 'Opening Extensions in this window'},
    ]
    assert result == {'status': 'awaiting_connection', 'message': 'Waiting for helper'}


def test_progress_callback_exception_cleans_up_owned_helper(extension, monkeypatch):
    process = FakeInstaller(
        b'{"status":"running","message":"Current tab confirmed"}\n'
        b'{"status":"awaiting_connection","message":"Waiting for helper"}\n',
        timeout=True,
    )
    monkeypatch.setattr(setup.subprocess, 'Popen', lambda *args, **kwargs: process)

    def destroyed_progress_receiver(_update):
        raise RuntimeError('UI was closed')

    with pytest.raises(RuntimeError, match='UI was closed'):
        setup._run_windows_installer(
            extension, '', connect_url=CONNECT_URL, cancel_check=lambda: False,
            progress=destroyed_progress_receiver,
        )
    assert process.killed
    assert process.stdout.closed
    assert process.stderr.closed


def test_missing_terminal_record_reports_error_even_after_progress(extension, monkeypatch):
    updates = []
    process = FakeInstaller(b'{"status":"running","message":"Opening Extensions"}\n')
    monkeypatch.setattr(setup.subprocess, 'Popen', lambda *args, **kwargs: process)
    result = setup._run_windows_installer(
        extension, '', connect_url=CONNECT_URL, cancel_check=lambda: False,
        progress=updates.append,
    )
    assert updates == [{'status': 'running', 'message': 'Opening Extensions'}]
    assert result['status'] == 'error'
    assert 'Opening Extensions' in result['message']


def test_progress_and_terminal_messages_redact_connection_url_and_nonce(extension, monkeypatch):
    nonce = CONNECT_URL.split('#', 1)[1]
    records = [
        {'status': 'running', 'message': f'Context {CONNECT_URL} and nonce {nonce}'},
        {'status': 'error', 'message': f'Could not continue {CONNECT_URL}: {nonce}'},
    ]
    process = FakeInstaller(('\n'.join(json.dumps(row) for row in records) + '\n').encode())
    monkeypatch.setattr(setup.subprocess, 'Popen', lambda *args, **kwargs: process)
    updates = []
    result = setup._run_windows_installer(
        extension, '', connect_url=CONNECT_URL, cancel_check=lambda: False,
        progress=updates.append,
    )
    assert updates[0]['status'] == 'running'
    assert result['status'] == 'error'
    visible_messages = json.dumps(updates + [result])
    assert CONNECT_URL not in visible_messages
    assert nonce not in visible_messages


def test_cancellation_terminates_only_owned_installer(extension, monkeypatch):
    process = FakeInstaller(timeout=True)
    monkeypatch.setattr(setup.subprocess, 'Popen', lambda *args, **kwargs: process)
    result = setup._run_windows_installer(extension, 'chrome', connect_url=CONNECT_URL, cancel_check=lambda: True)
    assert result['status'] == 'cancelled'
    assert process.killed


def test_timeout_terminates_only_owned_installer(extension, monkeypatch):
    process = FakeInstaller(timeout=True)
    monkeypatch.setattr(setup.subprocess, 'Popen', lambda *args, **kwargs: process)
    result = setup._run_windows_installer(extension, 'chrome', connect_url=CONNECT_URL, cancel_check=lambda: False, timeout=0)
    assert result['status'] == 'manual_required'
    assert process.killed


@pytest.mark.parametrize('output', [
    b'', b'not JSON', b'{"status":"installed","message":"unverified"}',
    b'{"status":"awaiting_connection","message":null}', b'[]',
])
def test_unrecognized_installer_output_never_claims_success(extension, monkeypatch, output):
    process = FakeInstaller(output)
    monkeypatch.setattr(setup.subprocess, 'Popen', lambda *args, **kwargs: process)
    result = setup._run_windows_installer(extension, 'chrome', connect_url=CONNECT_URL, cancel_check=lambda: False)
    assert result['status'] == 'error'


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


def test_windows_installer_prelude_compiles_without_accessing_browser_state():
    powershell = shutil.which('powershell.exe')
    if powershell is None:
        pytest.skip('PowerShell compiler is only checked when installed')
    # The prelude only loads assemblies and declares the native helper type.
    # No production body, native window query, or UIA tree access is executed.
    data = base64.b64encode(setup._INSTALL_PRELUDE.encode('utf-8')).decode('ascii')
    command = (
        "$source = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($env:ARENA_PRELUDE_TEST)); "
        "& ([ScriptBlock]::Create($source)); if (-not $?) { exit 1 }"
    )
    encoded = base64.b64encode(command.encode('utf-16-le')).decode('ascii')
    result = subprocess.run(
        [powershell, '-NoProfile', '-NonInteractive', '-WindowStyle', 'Hidden', '-EncodedCommand', encoded],
        capture_output=True, text=True, timeout=20,
        env=dict(os.environ, ARENA_PRELUDE_TEST=data),
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('statement, step', [
    ('Add-Type -AssemblyName UIAutomationClient', 'loading UIAutomationClient'),
    ('Add-Type -AssemblyName UIAutomationTypes', 'loading UIAutomationTypes'),
    ('Add-Type -AssemblyName System.Windows.Forms', 'loading Windows Forms'),
    ("Add-Type -TypeDefinition @'", 'compiling the Windows helper'),
])
def test_initialization_failure_is_reported_as_initialization_error(extension, monkeypatch, statement, step):
    if setup.sys.platform != 'win32' or shutil.which('powershell.exe') is None:
        pytest.skip('Windows PowerShell initialization check')
    # Assembly loading is allowed; no native APIs or browser access occur.
    script = setup._INSTALL_PRELUDE.replace(
        statement, "throw 'simulated initialization failure';\n" + statement,
    )
    monkeypatch.setattr(setup, '_INSTALL_SCRIPT', script)
    result = setup._run_windows_installer(
        extension, 'chrome', connect_url=CONNECT_URL, cancel_check=lambda: False, timeout=10,
    )
    assert result['status'] == 'error'
    assert 'initialization failed' in result['message']
    assert step in result['message']
    assert 'simulated initialization failure' in result['message']
    assert 'PowerShell 5.' in result['message']


def test_external_process_environment_excludes_only_bundle_paths(tmp_path, monkeypatch):
    bundle = tmp_path / '_MEI1234'
    sibling = tmp_path / '_MEI12345'
    entries = [str(bundle), str(bundle / 'PySide6'), str(sibling), r'C:\Windows\System32']
    monkeypatch.setattr(setup.sys, '_MEIPASS', str(bundle), raising=False)
    monkeypatch.setenv('PATH', os.pathsep.join(entries))
    env = setup._external_process_env()
    assert env['PATH'].split(os.pathsep) == entries[2:]
    assert os.environ['PATH'] == os.pathsep.join(entries)


@pytest.mark.parametrize('large_environment', [False, True])
def test_full_script_initialization_through_real_runner_without_browser_access(extension, monkeypatch, large_environment):
    if setup.sys.platform != 'win32' or shutil.which('powershell.exe') is None:
        pytest.skip('Windows PowerShell initialization check')
    # Preserve the full script/environment size and parse context, but exit
    # after initialization, before ANY native window or browser access.
    script = setup._INSTALL_SCRIPT.replace(
        '$arenaFolder =',
        "if ($null -ne [Environment]::GetEnvironmentVariable('GLOSSARION_ARENA_SETUP_SCRIPT')) "
        "{ throw 'Script transport must not reach the compiler environment' }\n"
        "@{status='awaiting_connection';message='Initialization complete'} | ConvertTo-Json -Compress\n"
        "exit 0\n$arenaFolder =", 1,
    )
    monkeypatch.setattr(setup, '_INSTALL_SCRIPT', script)
    if large_environment:
        # Reproduce the real 77,837-byte compiler failure with ASCII padding.
        # Windows starts PowerShell successfully, but Framework Add-Type fails
        # unless the large script variable is removed before it starts csc.exe.
        env = setup._external_process_env()
        size = sum(len(key) + len(value) + 2 for key, value in env.items())
        size += len(base64.b64encode(script.encode('utf-8'))) + 500
        remaining = max(0, 78000 - size)
        index = 0
        while remaining:
            amount = min(16000, remaining)
            monkeypatch.setenv(f'ARENA_ENVIRONMENT_REGRESSION_{index}', 'x' * amount)
            remaining -= amount
            index += 1
    result = setup._run_windows_installer(
        extension, 'chrome', connect_url=CONNECT_URL, cancel_check=lambda: False, timeout=20,
    )
    assert result == {'status': 'awaiting_connection', 'message': 'Initialization complete'}


def test_production_window_and_tab_guards_with_mocked_browser_state():
    powershell = shutil.which('powershell.exe')
    if powershell is None:
        pytest.skip('PowerShell guard tests')
    # Extract only the production logical guards, then supply fake window/tab
    # state. No production prelude, native window API, or UIA traversal runs.
    command = r'''
$ErrorActionPreference = 'Stop'
$source = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($env:ARENA_GUARD_TEST))
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseInput($source, [ref]$tokens, [ref]$errors)
$names = @('ConnectAddressMatches','SelectedNativeTab','AssertBoundWindow','AssertConnectionPage','AssertNewTab')
foreach ($name in $names) {
    $definition = $ast.Find({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name}, $true)
    if ($null -eq $definition) { throw "Missing production guard $name" }
    Invoke-Expression $definition.Extent.Text
}
function Manual([string]$message) { throw ('MANUAL: ' + $message) }
function CurrentWindowHandle { return $script:foreground }
function NativeAddress($window) { return 'mock-address' }
function AddressPattern($address) { return [pscustomobject]@{Current=[pscustomobject]@{Value=$script:addressValue}} }
function NativeTabs($window) { return $script:tabs }
function Tab([string]$id, [bool]$selected) { return [pscustomobject]@{Id=$id;Selected=$selected} }
function Reset {
    $script:arenaWindowHandle = 11; $script:foreground = 11
    $script:arenaBrowserPid = 22
    $script:arenaWindow = [pscustomobject]@{Current=[pscustomobject]@{ProcessId=22}}
    $script:arenaConnectUrl = 'http://127.0.0.1:18874/connect#test_nonce_1234567890'
    $script:addressValue = $script:arenaConnectUrl
    $script:arenaConnectionTabId = 'connection'
    $script:arenaBeforeTabIds = @('connection')
    $script:arenaNewTabId = 'new'
    $script:tabs = @(Tab 'connection' $true)
}
$cases = 0
function Allows([scriptblock]$action) { & $action; $script:cases++ }
function Blocks([scriptblock]$action) {
    $blocked = $false
    try { & $action } catch { if (-not $_.Exception.Message.StartsWith('MANUAL:')) { throw }; $blocked = $true }
    if (-not $blocked) { throw 'The unsafe window/tab transition was allowed' }
    $script:cases++
}
Reset
Allows { AssertConnectionPage }
$script:addressValue = $arenaConnectUrl.Substring(7)
Allows { AssertConnectionPage }
$script:addressValue = $arenaConnectUrl.Replace('test_nonce_', 'wrong_nonce_')
Blocks { AssertConnectionPage }
Reset; $script:addressValue = $arenaConnectUrl.Replace('18874', '18875')
Blocks { AssertConnectionPage }
Reset; $script:addressValue = 'https://example.com/connect#test_nonce_1234567890'
Blocks { AssertConnectionPage }
Reset; $script:foreground = 99
Blocks { AssertConnectionPage }
Reset; $script:arenaWindow.Current.ProcessId = 99
Blocks { AssertConnectionPage }
Reset; $script:tabs = @(Tab 'another' $true)
Blocks { AssertConnectionPage }
Reset
Blocks { AssertNewTab } # New Tab failed: original connection tab is untouched.
Reset; $script:tabs = @((Tab 'connection' $false), (Tab 'new' $true))
Allows { AssertNewTab }
$script:tabs = @((Tab 'connection' $true), (Tab 'new' $false))
Blocks { AssertNewTab } # User switched back before navigation.
Reset; $script:tabs = @(Tab 'new' $true)
Blocks { AssertNewTab } # Replaced old tab rather than opening another one.
Reset; $script:tabs = @((Tab 'connection' $false), (Tab 'unexpected' $true))
Blocks { AssertNewTab }
Reset; $script:tabs = @((Tab 'connection' $false), (Tab 'new' $true)); $script:foreground = 99
Blocks { AssertNewTab }
@{cases=$cases;passed=$true} | ConvertTo-Json -Compress
'''
    encoded = base64.b64encode(command.encode('utf-16-le')).decode('ascii')
    result = subprocess.run(
        [powershell, '-NoProfile', '-NonInteractive', '-WindowStyle', 'Hidden', '-EncodedCommand', encoded],
        capture_output=True, text=True, timeout=15,
        env=dict(os.environ, ARENA_GUARD_TEST=base64.b64encode(setup._INSTALL_SCRIPT.encode('utf-8')).decode('ascii')),
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {'cases': 14, 'passed': True}


def test_real_installer_process_roundtrip_without_browser_automation(extension, monkeypatch):
    if setup.sys.platform != 'win32' or shutil.which('powershell.exe') is None:
        pytest.skip('Windows PowerShell process check')
    # Exercise the real hidden child-process/bootstrap and environment encoding
    # with a harmless script. No UIA code or browser executable is invoked.
    gate = extension / 'progress-was-delivered'
    monkeypatch.setattr(setup, '_INSTALL_SCRIPT', r"""
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$folder = $env:GLOSSARION_ARENA_SETUP_FOLDER
@{status='running'; message=$folder} | ConvertTo-Json -Compress
[Console]::Out.Flush()
$gate = Join-Path $folder 'progress-was-delivered'
$deadline = [DateTime]::UtcNow.AddSeconds(5)
while (-not (Test-Path -LiteralPath $gate)) {
    if ([DateTime]::UtcNow -gt $deadline) { throw 'Progress was buffered until terminal output.' }
    Start-Sleep -Milliseconds 30
}
$context = $env:GLOSSARION_ARENA_SETUP_CONNECT_URL + '|' + $env:GLOSSARION_ARENA_SETUP_BROWSER_HINT
$digest = [Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($context))
$hex = [BitConverter]::ToString($digest).Replace('-', '').ToLowerInvariant()
@{status='awaiting_connection'; message=$hex} | ConvertTo-Json -Compress
""")
    updates = []

    def on_progress(update):
        updates.append(update)
        gate.write_text('delivered before terminal', encoding='utf-8')

    result = setup._run_windows_installer(
        extension, 'chrome', connect_url=CONNECT_URL, cancel_check=lambda: False,
        progress=on_progress, timeout=10,
    )
    digest = hashlib.sha256((CONNECT_URL + '|chrome').encode('utf-8')).hexdigest()
    assert result == {'status': 'awaiting_connection', 'message': digest}
    assert updates == [{'status': 'running', 'message': str(extension)}]
    assert gate.read_text(encoding='utf-8') == 'delivered before terminal'
    assert CONNECT_URL not in json.dumps(updates + [result])


def test_real_helper_cancels_after_incremental_progress_without_browser_automation(extension, monkeypatch):
    if setup.sys.platform != 'win32' or shutil.which('powershell.exe') is None:
        pytest.skip('Windows PowerShell process check')
    monkeypatch.setattr(setup, '_INSTALL_SCRIPT', r"""
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
@{status='running'; message='Waiting for the current tab'} | ConvertTo-Json -Compress
[Console]::Out.Flush()
Start-Sleep -Seconds 30
throw 'The cancelled helper should have been terminated.'
""")
    launches = []
    popen = setup.subprocess.Popen

    def record_process(args, **kwargs):
        process = popen(args, **kwargs)
        launches.append((args, process))
        return process

    monkeypatch.setattr(setup.subprocess, 'Popen', record_process)
    updates = []
    result = setup._run_windows_installer(
        extension, '', connect_url=CONNECT_URL, cancel_check=lambda: bool(updates),
        progress=updates.append, timeout=10,
    )
    assert result['status'] == 'cancelled'
    assert updates == [{'status': 'running', 'message': 'Waiting for the current tab'}]
    assert len(launches) == 1
    assert Path(launches[0][0][0]).name.lower() == 'powershell.exe'
    assert launches[0][1].poll() is not None
