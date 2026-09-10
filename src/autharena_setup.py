"""User-initiated Arena companion setup using the browser's normal install UI.

Windows uses built-in UI Automation, without an extra Python dependency. Only
recognized Chrome/Edge controls are invoked. An unfamiliar or blocked browser
UI falls back to instructions; pairing, not this installer, verifies success.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit


_REQUIRED_FILES = ('manifest.json', 'background.js', 'connect.js', 'arena_page.js')
_DLL_LAUNCH_LOCK = threading.Lock()


def _result(status, message):
    return {'status': status, 'message': message}


def _validated_folder(extension_path):
    path = Path(extension_path).resolve(strict=True)
    if not path.is_dir() or any(not (path / name).is_file() for name in _REQUIRED_FILES):
        raise ValueError('The prepared Arena extension is incomplete. Restart Glossarion and try Arena Login again.')
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('name') != 'Glossarion Arena Browser Companion'
            or manifest.get('manifest_version') != 3):
        raise ValueError('The selected folder is not the Glossarion Arena companion.')
    return path


def _validated_connect_url(value):
    """Accept only the broker's exact one-time local connection URL."""
    if not isinstance(value, str) or len(value) > 1024:
        raise ValueError('Invalid Arena setup link.')
    parsed = urlsplit(value)
    if (parsed.scheme != 'http' or parsed.hostname != '127.0.0.1'
            or parsed.username is not None or parsed.password is not None
            or parsed.port is None or not 1 <= parsed.port <= 65535
            or parsed.path != '/connect' or parsed.query
            or not re.fullmatch(r'[A-Za-z0-9_-]{16,256}', parsed.fragment)
            or value != f'http://127.0.0.1:{parsed.port}/connect#{parsed.fragment}'):
        raise ValueError('Invalid Arena setup link. Click Arena Login again.')
    return value


# The prepared folder and exact connection URL are passed as environment data.
# UIA stays in the existing browser window and its owned folder picker. No
# coordinates, credential stores, browser profiles, or policies are accessed.
_INSTALL_PRELUDE = r'''
$ErrorActionPreference = 'Stop'
$arenaInitStep = 'configuring output'
try {
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    $arenaInitStep = 'loading UIAutomationClient'
    Add-Type -AssemblyName UIAutomationClient
    $arenaInitStep = 'loading UIAutomationTypes'
    Add-Type -AssemblyName UIAutomationTypes
    $arenaInitStep = 'loading Windows Forms'
    Add-Type -AssemblyName System.Windows.Forms
    $arenaInitStep = 'compiling the Windows helper'
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class ArenaSetupWindow {
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] public static extern IntPtr GetAncestor(IntPtr window, uint flags);
    public static volatile bool InvokeFailed;
    public static void InvokeAsync(object pattern) {
        InvokeFailed = false;
        var worker = new System.Threading.Thread(() => {
            try { pattern.GetType().GetMethod("Invoke").Invoke(pattern, null); }
            catch { InvokeFailed = true; }
        });
        worker.IsBackground = true;
        worker.SetApartmentState(System.Threading.ApartmentState.MTA);
        worker.Start();
    }
}
'@
} catch {
    $arenaInitError = $_.Exception
    while ($null -ne $arenaInitError.InnerException) { $arenaInitError = $arenaInitError.InnerException }
    $arenaInitReason = ($arenaInitError.Message -replace '[\r\n]+', ' ').Trim()
    $arenaInitCode = '0x' + $arenaInitError.HResult.ToString('X8')
    $arenaInitMode = [string]$ExecutionContext.SessionState.LanguageMode
    @{status='error';message=("Windows setup initialization failed while " + $arenaInitStep + ": " +
        $arenaInitReason + " (" + $_.FullyQualifiedErrorId + ", " + $arenaInitCode +
        "; PowerShell " + $PSVersionTable.PSVersion + ", " + $arenaInitMode + ").")} | ConvertTo-Json -Compress
    exit 1
}
'''

_INSTALL_SCRIPT = _INSTALL_PRELUDE + r'''
$arenaFolder = [Environment]::GetEnvironmentVariable('GLOSSARION_ARENA_SETUP_FOLDER')
$arenaBrowserHint = [string][Environment]::GetEnvironmentVariable('GLOSSARION_ARENA_SETUP_BROWSER_HINT')
$arenaConnectUrl = [string][Environment]::GetEnvironmentVariable('GLOSSARION_ARENA_SETUP_CONNECT_URL')
$arenaDeadline = [DateTime]::UtcNow.AddSeconds(40)
$arenaWindow = $null
$arenaWindowHandle = 0
$arenaBrowserPid = 0
$arenaManagementUrl = ''
$arenaConnectionTabId = ''
$arenaNewTabId = ''
$arenaBeforeTabIds = @()
$script:arenaSetupPhase = 'checking the connection window'

function Finish([string]$status, [string]$message) {
    @{status=$status;message=$message} | ConvertTo-Json -Compress
    exit 0
}
function Manual([string]$message) { Finish 'manual_required' $message }
function Progress([string]$phase, [string]$message) {
    $script:arenaSetupPhase = $phase
    @{status='running';message=$message} | ConvertTo-Json -Compress
    [Console]::Out.Flush()
}
function NameCondition([string]$name) {
    return [System.Windows.Automation.PropertyCondition]::new(
        [System.Windows.Automation.AutomationElement]::NameProperty, $name)
}
function TypeCondition($type) {
    return [System.Windows.Automation.PropertyCondition]::new(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty, $type)
}
function Visible($element) {
    return $null -ne $element -and $element.Current.IsEnabled -and -not $element.Current.IsOffscreen
}
function CurrentWindowHandle {
    return [ArenaSetupWindow]::GetForegroundWindow().ToInt64()
}
function NativeElement($element, $window) {
    $ancestor = [System.Windows.Automation.TreeWalker]::ControlViewWalker.GetParent($element)
    while ($null -ne $ancestor) {
        if ([System.Windows.Automation.Automation]::Compare($ancestor, $window)) { return $true }
        if ($ancestor.Current.ControlType -eq [System.Windows.Automation.ControlType]::Document) { return $false }
        $ancestor = [System.Windows.Automation.TreeWalker]::ControlViewWalker.GetParent($ancestor)
    }
    return $false
}
function NativeAddress($window) {
    $edits = $window.FindAll([System.Windows.Automation.TreeScope]::Descendants,
        (TypeCondition ([System.Windows.Automation.ControlType]::Edit)))
    $address = $null
    foreach ($edit in $edits) {
        if (-not (Visible $edit) -or $edit.Current.Name -ne 'Address and search bar' -or
            -not (NativeElement $edit $window)) { continue }
        if ($null -ne $address) { Manual 'The native browser address bar is ambiguous. Finish setup manually in the connection window.' }
        $address = $edit
    }
    if ($null -eq $address) { Manual 'The native browser address bar is unavailable. Keep the connection window visible and retry, or finish setup manually.' }
    return $address
}
function AddressPattern($address) {
    $value = $null
    if (-not $address.TryGetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern, [ref]$value)) {
        Manual 'The browser address bar does not support this setup step. Finish setup manually in the connection window.'
    }
    return $value
}
function ConnectAddressMatches([string]$value) {
    # Chrome can omit the displayed HTTP scheme. No host, path or nonce changes
    # are accepted, and the expected loopback URL is supplied by the broker.
    return $value -ceq $arenaConnectUrl -or
        ($arenaConnectUrl.StartsWith('http://', [StringComparison]::Ordinal) -and
         $value -ceq $arenaConnectUrl.Substring(7))
}
function NativeTabs($window) {
    $elements = $window.FindAll([System.Windows.Automation.TreeScope]::Descendants,
        (TypeCondition ([System.Windows.Automation.ControlType]::TabItem)))
    foreach ($element in $elements) {
        if (-not (NativeElement $element $window)) { continue }
        $selection = $null
        if (-not $element.TryGetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern, [ref]$selection)) { continue }
        [pscustomobject]@{
            Element=$element
            Id=[string]::Join(',', [int[]]$element.GetRuntimeId())
            Selected=$selection.Current.IsSelected
        }
    }
}
function SelectedNativeTab($tabs) {
    $selected = @($tabs | Where-Object { $_.Selected })
    if ($selected.Count -ne 1) { return $null }
    return $selected[0]
}
function AssertBoundWindow {
    if ((CurrentWindowHandle) -ne $arenaWindowHandle -or
        $arenaWindow.Current.ProcessId -ne $arenaBrowserPid) {
        Manual 'Setup paused because the active browser window changed. Return to the connection window and retry, or finish setup manually.'
    }
}
function AssertConnectionPage {
    AssertBoundWindow
    $address = NativeAddress $arenaWindow
    $value = AddressPattern $address
    $selected = SelectedNativeTab @(NativeTabs $arenaWindow)
    if (-not (ConnectAddressMatches $value.Current.Value) -or $null -eq $selected -or
        $selected.Id -cne $arenaConnectionTabId) {
        Manual 'Setup paused because the Arena connection tab changed. Return to the original connection page and retry.'
    }
}
function AssertNewTab {
    AssertBoundWindow
    $tabs = @(NativeTabs $arenaWindow)
    $selected = SelectedNativeTab $tabs
    if ($null -eq $selected -or $selected.Id -cne $arenaNewTabId -or
        $arenaBeforeTabIds -ccontains $selected.Id -or $tabs.Count -le $arenaBeforeTabIds.Count) {
        Manual 'Setup paused because its newly opened tab changed. Return to the Arena connection page and retry, or finish setup manually.'
    }
}
function AssertManagementPage {
    AssertNewTab
    $address = NativeAddress $arenaWindow
    $value = AddressPattern $address
    if ($value.Current.Value.TrimEnd('/') -cne $arenaManagementUrl.TrimEnd('/')) {
        Manual 'Setup only acts on its browser Extensions tab. Finish setup there, or return to the Arena connection page and retry.'
    }
}
function OwnedPicker($picker) {
    return $null -ne $picker -and $picker.Current.ClassName -eq '#32770' -and
        $picker.Current.ProcessId -eq $arenaBrowserPid -and
        [ArenaSetupWindow]::GetAncestor([IntPtr]$picker.Current.NativeWindowHandle, 3).ToInt64() -eq $arenaWindowHandle
}
function AssertPicker($picker) {
    if ((CurrentWindowHandle) -ne $picker.Current.NativeWindowHandle -or -not (OwnedPicker $picker)) {
        Manual 'Setup paused because the folder picker changed or belongs to another window. Select the Arena folder in the intended browser.'
    }
}
function ExtensionsDocument {
    $condition = [System.Windows.Automation.AndCondition]::new(
        (TypeCondition ([System.Windows.Automation.ControlType]::Document)), (NameCondition 'Extensions'))
    return $arenaWindow.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $condition)
}
function FindButton($root, [string]$name) {
    $condition = [System.Windows.Automation.AndCondition]::new(
        (TypeCondition ([System.Windows.Automation.ControlType]::Button)), (NameCondition $name))
    return $root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $condition)
}
function Invoke($element, $window, [bool]$async = $false) {
    if ($window.Current.ClassName -eq '#32770') { AssertPicker $window }
    else { AssertManagementPage }
    if (-not (Visible $element)) { Manual 'A browser control is unavailable. Complete Developer mode and Load unpacked manually.' }
    $pattern = $null
    if (-not $element.TryGetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern, [ref]$pattern)) {
        Manual 'The browser does not expose this control to automation. Use Load unpacked and select the prepared folder.'
    }
    if ($async) { [ArenaSetupWindow]::InvokeAsync($pattern) }
    else { $pattern.Invoke() }
}

try {
    if ($arenaConnectUrl -cnotmatch '^http://127\.0\.0\.1:[0-9]{1,5}/connect#[a-zA-Z0-9_-]{16,256}$' -or
        @('', 'chrome', 'edge') -cnotcontains $arenaBrowserHint) {
        Finish 'error' 'Windows setup received an invalid connection request. Start Arena Login again.'
    }
    Progress 'checking the connection window' 'Checking the browser window containing this Arena connection page.'
    $handle = CurrentWindowHandle
    if ($handle -eq 0) { Manual 'Keep the Arena connection page in the foreground and retry setup.' }
    $window = [System.Windows.Automation.AutomationElement]::FromHandle([IntPtr]$handle)
    $process = Get-Process -Id $window.Current.ProcessId -ErrorAction SilentlyContinue
    $executable = if ($null -ne $process) { [IO.Path]::GetFileName($process.Path) } else { '' }
    $family = if ($executable -ieq 'chrome.exe') { 'chrome' } elseif ($executable -ieq 'msedge.exe') { 'edge' } else { '' }
    if (-not $family -or ($arenaBrowserHint -and $family -cne $arenaBrowserHint)) {
        Manual 'Keep this Arena connection page open in Chrome or Edge in the foreground, then retry setup.'
    }
    $address = NativeAddress $window
    $value = AddressPattern $address
    if (-not (ConnectAddressMatches $value.Current.Value)) {
        Manual 'The foreground tab is not this Arena connection page. Return to it and retry setup.'
    }
    $arenaWindow = $window
    $arenaWindowHandle = $handle
    $arenaBrowserPid = $window.Current.ProcessId
    $arenaManagementUrl = if ($family -eq 'edge') { 'edge://extensions/' } else { 'chrome://extensions/' }
    $beforeTabs = @(NativeTabs $arenaWindow)
    $connectionTab = SelectedNativeTab $beforeTabs
    if ($beforeTabs.Count -eq 0 -or $null -eq $connectionTab) {
        Manual 'The browser tab strip could not be identified. Finish setup manually without closing the Arena connection page.'
    }
    $arenaBeforeTabIds = @($beforeTabs | ForEach-Object { $_.Id })
    $arenaConnectionTabId = $connectionTab.Id
    AssertConnectionPage

    Progress 'opening a new tab' 'Opening a new tab in the same browser window and profile.'
    $newButtons = @($arenaWindow.FindAll([System.Windows.Automation.TreeScope]::Descendants,
        (TypeCondition ([System.Windows.Automation.ControlType]::Button))) | Where-Object {
            (Visible $_) -and $_.Current.Name -match '^New tab(?: \(Ctrl\+T\))?$' -and (NativeElement $_ $arenaWindow)
        })
    $newPattern = $null
    $invokeNew = $newButtons.Count -eq 1 -and $newButtons[0].TryGetCurrentPattern(
        [System.Windows.Automation.InvokePattern]::Pattern, [ref]$newPattern)
    AssertConnectionPage
    if ($invokeNew) { [ArenaSetupWindow]::InvokeAsync($newPattern) }
    else { [System.Windows.Forms.SendKeys]::SendWait('^t') }
    # Never retry this action: a delayed success could otherwise open extra tabs.
    $tabDeadline = [DateTime]::UtcNow.AddSeconds(8)
    while ([DateTime]::UtcNow -lt $tabDeadline -and [DateTime]::UtcNow -lt $arenaDeadline) {
        AssertBoundWindow
        $tabs = @(NativeTabs $arenaWindow)
        $selected = SelectedNativeTab $tabs
        if ($tabs.Count -gt $arenaBeforeTabIds.Count -and $null -ne $selected -and
            $arenaBeforeTabIds -cnotcontains $selected.Id) {
            $arenaNewTabId = $selected.Id
            break
        }
        if ($invokeNew -and [ArenaSetupWindow]::InvokeFailed) { break }
        Start-Sleep -Milliseconds 100
    }
    if (-not $arenaNewTabId) {
        Manual 'A newly opened tab could not be confirmed. The connection page was kept intact; finish setup manually or retry.'
    }

    Progress 'opening Extensions' 'Opening Extensions in the newly created tab.'
    AssertNewTab
    $address = NativeAddress $arenaWindow
    $value = AddressPattern $address
    if ($value.Current.IsReadOnly) { Manual 'The browser address bar is read-only. Finish setup manually in the new tab.' }
    $address.SetFocus()
    AssertNewTab
    if (-not $address.Current.HasKeyboardFocus) { Manual 'The browser address bar did not receive focus. Finish setup manually in the new tab.' }
    $value.SetValue($arenaManagementUrl)
    AssertNewTab
    $address = NativeAddress $arenaWindow
    $value = AddressPattern $address
    if (-not $address.Current.HasKeyboardFocus -or $value.Current.Value -cne $arenaManagementUrl) {
        Manual 'The browser address bar changed before navigation. Finish setup manually in the new tab.'
    }
    AssertBoundWindow
    [System.Windows.Forms.SendKeys]::SendWait('{ENTER}')

    $document = $null
    $navigationDeadline = [DateTime]::UtcNow.AddSeconds(10)
    while ([DateTime]::UtcNow -lt $navigationDeadline -and [DateTime]::UtcNow -lt $arenaDeadline) {
        AssertNewTab
        $address = NativeAddress $arenaWindow
        $value = AddressPattern $address
        if ($value.Current.Value.TrimEnd('/') -ceq $arenaManagementUrl.TrimEnd('/')) {
            $document = ExtensionsDocument
            if ($null -ne $document) { break }
        }
        Start-Sleep -Milliseconds 150
    }
    if ($null -eq $document) {
        Manual 'The Extensions page did not become available in the new tab. Finish setup manually there, keeping the Arena connection page open.'
    }
    AssertManagementPage
    $existing = $document.FindFirst([System.Windows.Automation.TreeScope]::Descendants,
        (NameCondition 'Glossarion Arena Browser Companion'))
    if ($null -ne $existing) {
        Manual 'The Arena helper is already listed. Enable or reload its card, then return to the Arena connection page.'
    }

    Progress 'enabling Developer mode' 'Checking Developer mode on the Extensions page.'
    $load = FindButton $document 'Load unpacked'
    if (-not (Visible $load)) {
        $controls = $document.FindAll([System.Windows.Automation.TreeScope]::Descendants,
            (NameCondition 'Developer mode'))
        $toggled = $false
        foreach ($control in $controls) {
            $toggle = $null
            if ((Visible $control) -and $control.TryGetCurrentPattern(
                [System.Windows.Automation.TogglePattern]::Pattern, [ref]$toggle)) {
                AssertManagementPage
                if ($toggle.Current.ToggleState -eq [System.Windows.Automation.ToggleState]::Off) { $toggle.Toggle() }
                $toggled = $true
                break
            }
        }
        if (-not $toggled) { Manual 'Enable Developer mode on the Extensions page, then choose Load unpacked and select the prepared Arena folder.' }
        for ($attempt = 0; $attempt -lt 20 -and [DateTime]::UtcNow -lt $arenaDeadline; $attempt++) {
            AssertManagementPage
            $load = FindButton $document 'Load unpacked'
            if (Visible $load) { break }
            Start-Sleep -Milliseconds 150
        }
    }
    Progress 'opening the folder picker' 'Opening Load unpacked in this browser window.'
    Invoke $load $arenaWindow $true
    $picker = $null
    while ([DateTime]::UtcNow -lt $arenaDeadline) {
        $foreground = CurrentWindowHandle
        if ($foreground -ne $arenaWindowHandle) {
            if ($foreground -eq 0) { Manual 'Setup paused because the browser window is no longer active. Finish Load unpacked manually.' }
            $candidate = [System.Windows.Automation.AutomationElement]::FromHandle([IntPtr]$foreground)
            if ((OwnedPicker $candidate) -and
                $candidate.Current.Name -match '(?i)(select.*(extension|folder)|load.*(extension|unpacked))') {
                $picker = $candidate
                break
            }
            Manual 'Setup paused because another window became active. Finish Load unpacked in the intended browser.'
        }
        if ([ArenaSetupWindow]::InvokeFailed) { break }
        Start-Sleep -Milliseconds 150
    }
    if ($null -eq $picker) { Manual 'Select the prepared Arena folder in the browser folder picker, then return to the Arena connection page.' }

    Progress 'selecting the Arena folder' 'Selecting the prepared Arena companion folder.'
    $edits = $picker.FindAll([System.Windows.Automation.TreeScope]::Descendants,
        (TypeCondition ([System.Windows.Automation.ControlType]::Edit)))
    $folderEdit = $null
    foreach ($edit in $edits) {
        if ((Visible $edit) -and $edit.Current.Name -match '^(Folder|File name):?$') {
            if ($null -ne $folderEdit) { Manual 'Select the prepared Arena folder in the open folder picker.' }
            $folderEdit = $edit
        }
    }
    $value = $null
    if ($null -eq $folderEdit -or -not $folderEdit.TryGetCurrentPattern(
        [System.Windows.Automation.ValuePattern]::Pattern, [ref]$value)) {
        Manual 'Paste the prepared Arena folder path into the folder picker and select that folder.'
    }
    AssertPicker $picker
    $value.SetValue($arenaFolder)
    $select = FindButton $picker 'Select Folder'
    if ($null -eq $select) { $select = FindButton $picker 'Select folder' }
    if ($null -eq $select) { $select = FindButton $picker 'Select' }
    Invoke $select $picker
    Finish 'awaiting_connection' 'The Arena folder was submitted to the browser. Return to the Arena connection page; installation is confirmed when the helper connects.'
} catch {
    Finish 'manual_required' ("Windows setup stopped while " + $script:arenaSetupPhase + ". Finish the normal browser steps manually, or return to the Arena connection page and retry.")
}
'''


def _external_process_env():
    """Keep bundled native libraries out of the system PowerShell process."""
    env = os.environ.copy()
    bundle = getattr(sys, '_MEIPASS', None)
    if bundle:
        bundle = os.path.normcase(os.path.abspath(bundle))
        def bundled(entry):
            entry = os.path.normcase(os.path.abspath(entry.strip('"')))
            return entry == bundle or entry.startswith(bundle + os.sep)
        env['PATH'] = os.pathsep.join(entry for entry in env.get('PATH', '').split(os.pathsep)
                                      if entry and not bundled(entry))
    return env


@contextmanager
def _system_dll_search():
    """Restore PyInstaller's DLL directory immediately after spawning the child."""
    if sys.platform != 'win32' or not getattr(sys, '_MEIPASS', None):
        yield
        return
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    get_directory = kernel.GetDllDirectoryW
    get_directory.argtypes = [wintypes.DWORD, wintypes.LPWSTR]
    get_directory.restype = wintypes.DWORD
    set_directory = kernel.SetDllDirectoryW
    set_directory.argtypes = [wintypes.LPCWSTR]
    set_directory.restype = wintypes.BOOL
    with _DLL_LAUNCH_LOCK:
        buffer = ctypes.create_unicode_buffer(32768)
        length = get_directory(len(buffer), buffer)
        if length >= len(buffer):
            raise OSError('The native DLL directory is too long.')
        previous = buffer.value if length else None
        if not set_directory(None):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            yield
        finally:
            if not set_directory(previous):
                raise ctypes.WinError(ctypes.get_last_error())


def _run_windows_installer(folder, browser_hint, *, connect_url, cancel_check, progress=None, timeout=50):
    env = _external_process_env()
    env['GLOSSARION_ARENA_SETUP_FOLDER'] = str(folder)
    env['GLOSSARION_ARENA_SETUP_BROWSER_HINT'] = browser_hint
    env['GLOSSARION_ARENA_SETUP_CONNECT_URL'] = connect_url
    # Keep the Windows command line short; the complete static script is data
    # in this child process's environment, not a command-line interpolation.
    env['GLOSSARION_ARENA_SETUP_SCRIPT'] = base64.b64encode(_INSTALL_SCRIPT.encode('utf-8')).decode('ascii')
    powershell = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
    bootstrap = (
        "$arenaSetupSource = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String("
        "[Environment]::GetEnvironmentVariable('GLOSSARION_ARENA_SETUP_SCRIPT'))); "
        # Add-Type starts the .NET Framework C# compiler, whose inherited
        # environment is limited to 65535 bytes. Keep the decoded script in
        # memory, but remove its large transport variable before compilation.
        "[Environment]::SetEnvironmentVariable('GLOSSARION_ARENA_SETUP_SCRIPT', $null); "
        "& ([ScriptBlock]::Create($arenaSetupSource))"
    )
    encoded = base64.b64encode(bootstrap.encode('utf-16-le')).decode('ascii')
    with _system_dll_search():
        process = subprocess.Popen(
            [str(powershell), '-NoLogo', '-NoProfile', '-NonInteractive', '-WindowStyle', 'Hidden', '-EncodedCommand', encoded],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
    events = queue.Queue()

    def read_output():
        try:
            for line in process.stdout:
                events.put(('line', line))
        finally:
            events.put(('eof', None))

    def drain_errors():
        # Errors can contain expanded script arguments. Drain this pipe to
        # avoid deadlocks; report the structured stage or numeric exit code.
        while process.stderr.read(4096):
            pass

    readers = [threading.Thread(target=read_output, name='arena-setup-output', daemon=True),
               threading.Thread(target=drain_errors, name='arena-setup-errors', daemon=True)]
    for reader in readers:
        reader.start()
    deadline = time.monotonic() + timeout
    last_stage = ''
    nonce = urlsplit(connect_url).fragment

    def safe_message(value):
        value = value.replace(connect_url, '[Arena login link]')
        if nonce:
            value = value.replace(nonce, '[redacted]')
        return value[:2000]

    try:
        while True:
            if cancel_check():
                return _result('cancelled', 'Arena extension setup was cancelled.')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                step = f' Last step: {last_stage}' if last_stage else ''
                return _result('manual_required', 'Automatic setup timed out.' + step +
                               ' Return to Arena Login for the manual installation steps.')
            try:
                kind, line = events.get(timeout=min(.2, remaining))
            except queue.Empty:
                continue
            if kind == 'eof':
                try:
                    code = process.wait(timeout=min(1, max(.01, remaining)))
                except subprocess.TimeoutExpired:
                    code = None
                detail = f' (exit code {code})' if code is not None else ''
                step = f' Last step: {last_stage}' if last_stage else ''
                return _result('error', 'The Windows setup helper ended without a result' + detail + '.' + step +
                               ' Return to Arena Login and try again or use Manual installation.')
            try:
                result = json.loads(line.decode('utf-8-sig', errors='replace'))
            except (ValueError, AttributeError):
                continue
            if not isinstance(result, dict) or not isinstance(result.get('message'), str):
                continue
            message = safe_message(result['message'])
            if result.get('status') == 'running':
                last_stage = message
                if progress is not None:
                    progress(_result('running', message))
            elif result.get('status') in {'awaiting_connection', 'manual_required', 'cancelled', 'error'}:
                return _result(result['status'], message)
    finally:
        if process.poll() is None:
            try:
                process.wait(timeout=.2)
            except subprocess.TimeoutExpired:
                process.kill()  # Only this short-lived installer; never the browser.
        process.wait(timeout=3)
        for reader in readers:
            reader.join(timeout=.5)
        process.stdout.close()
        process.stderr.close()


def install_extension(extension_path, browser_hint='', *, connect_url='', cancel_check=lambda: False, progress=None):
    """Install from the existing window containing this exact setup link."""
    if browser_hint not in ('', 'chrome', 'edge'):
        return _result('error', 'Unsupported browser selection.')
    try:
        folder = _validated_folder(extension_path)
        if cancel_check():
            return _result('cancelled', 'Arena extension setup was cancelled.')
        if sys.platform != 'win32':
            return _result('manual_required', 'Open your browser Extensions page, enable Developer mode, choose Load unpacked, and select the prepared Arena folder.')
        try:
            connect_url = _validated_connect_url(connect_url)
        except ValueError:
            return _result('error', 'Open Arena Login in Glossarion again, then click Install in browser from that page.')
        if cancel_check():
            return _result('cancelled', 'Arena extension setup was cancelled.')
        # Never start a browser executable: Chrome can route that launch to
        # its profile picker instead of the profile containing the setup page.
        return _run_windows_installer(folder, browser_hint, connect_url=connect_url,
                                      cancel_check=cancel_check, progress=progress)
    except (OSError, ValueError, subprocess.SubprocessError):
        return _result('error', 'The Windows setup helper could not start. Return to Arena Login and try again or use Manual installation.')


def open_extension_folder(extension_path):
    """Open only the prepared folder, after an explicit setup-page action."""
    try:
        folder = _validated_folder(extension_path)
        if sys.platform == 'win32':
            os.startfile(str(folder))
        else:
            subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open', str(folder)], close_fds=True)
        return _result('opened', 'The prepared Arena extension folder is open.')
    except (OSError, ValueError, subprocess.SubprocessError):
        return _result('error', 'Could not open the folder. Copy the path shown on this page instead.')
