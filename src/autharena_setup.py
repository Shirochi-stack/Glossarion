"""User-initiated Arena companion setup using the browser's normal install UI.

Windows uses built-in UI Automation, without an extra Python dependency. Only
recognized Chrome/Edge controls are invoked. An unfamiliar or blocked browser
UI falls back to instructions; pairing, not this installer, verifies success.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time


_REQUIRED_FILES = ('manifest.json', 'background.js', 'connect.js', 'arena_page.js')
_BROWSERS = {
    'chrome': ('chrome.exe', 'Google/Chrome/Application/chrome.exe', 'chrome://extensions/'),
    'edge': ('msedge.exe', 'Microsoft/Edge/Application/msedge.exe', 'edge://extensions/'),
}


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


def _registry_browser_path(executable):
    import winreg
    key = rf'Software\Microsoft\Windows\CurrentVersion\App Paths\{executable}'
    for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(root, key) as handle:
                value, _ = winreg.QueryValueEx(handle, None)
            candidate = Path(os.path.expandvars(str(value).strip().strip('"')))
            if candidate.name.lower() == executable and candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _default_browser_family():
    import winreg
    try:
        key = r'Software\Microsoft\Windows\Shell\Associations\UrlAssociations\https\UserChoice'
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key) as handle:
            progid, _ = winreg.QueryValueEx(handle, 'ProgId')
        value = str(progid).lower()
        return 'edge' if value.startswith('msedge') else 'chrome' if value.startswith('chrome') else ''
    except OSError:
        return ''


def _find_browser(browser_hint):
    family = browser_hint or _default_browser_family()
    if family not in _BROWSERS:
        return None
    executable, relative, url = _BROWSERS[family]
    installed = _registry_browser_path(executable)
    if installed is None:
        for name in ('LOCALAPPDATA', 'PROGRAMFILES', 'PROGRAMFILES(X86)'):
            root = os.environ.get(name)
            candidate = Path(root) / relative if root else None
            if candidate is not None and candidate.is_file():
                installed = candidate
                break
    return (installed, url) if installed is not None else None


# Folder and browser paths are passed as environment data, never script text.
# UIA reads only the foreground browser's Extensions document and its own folder
# picker. No coordinates, credentials, browser profiles, or policies are used.
_INSTALL_SCRIPT = r'''
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
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
$arenaFolder = [Environment]::GetEnvironmentVariable('GLOSSARION_ARENA_SETUP_FOLDER')
$arenaBrowser = [Environment]::GetEnvironmentVariable('GLOSSARION_ARENA_SETUP_BROWSER')
$arenaManagementUrl = [Environment]::GetEnvironmentVariable('GLOSSARION_ARENA_SETUP_URL')
$arenaDeadline = [DateTime]::UtcNow.AddSeconds(40)
$arenaWindowHandle = 0
function Finish([string]$status, [string]$message) {
    @{status=$status;message=$message} | ConvertTo-Json -Compress
    exit 0
}
function Manual([string]$message) { Finish 'manual_required' $message }
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
function ForegroundBrowser {
    $handle = [ArenaSetupWindow]::GetForegroundWindow()
    if ($handle -eq [IntPtr]::Zero) { return $null }
    $window = [System.Windows.Automation.AutomationElement]::FromHandle($handle)
    $process = Get-Process -Id $window.Current.ProcessId -ErrorAction SilentlyContinue
    if ($null -eq $process -or $process.Path -ine $arenaBrowser) { return $null }
    return $window
}
function AssertForeground($window) {
    if ([ArenaSetupWindow]::GetForegroundWindow().ToInt64() -ne $window.Current.NativeWindowHandle) {
        Manual 'Setup paused because the active window changed. Return to Extensions and finish Load unpacked, or try Install in browser again.'
    }
}
function AssertManagementPage($window) {
    AssertForeground $window
    $edits = $window.FindAll([System.Windows.Automation.TreeScope]::Descendants,
        (TypeCondition ([System.Windows.Automation.ControlType]::Edit)))
    $address = $null
    foreach ($edit in $edits) {
        if (-not (Visible $edit) -or $edit.Current.Name -ne 'Address and search bar') { continue }
        # A web page can copy this label. Only accept browser chrome outside
        # every Document ancestor, never a page-owned textbox with that name.
        $ancestor = [System.Windows.Automation.TreeWalker]::ControlViewWalker.GetParent($edit)
        $insideDocument = $false
        $reachedBrowserRoot = $false
        while ($null -ne $ancestor) {
            if ([System.Windows.Automation.Automation]::Compare($ancestor, $window)) { $reachedBrowserRoot = $true; break }
            if ($ancestor.Current.ControlType -eq [System.Windows.Automation.ControlType]::Document) { $insideDocument = $true; break }
            $ancestor = [System.Windows.Automation.TreeWalker]::ControlViewWalker.GetParent($ancestor)
        }
        if ($insideDocument -or -not $reachedBrowserRoot) { continue }
        if ($null -ne $address) { Manual 'The browser address bar could not be identified safely. Complete Load unpacked manually.' }
        $address = $edit
    }
    $value = $null
    if ($null -eq $address -or -not $address.TryGetCurrentPattern(
        [System.Windows.Automation.ValuePattern]::Pattern, [ref]$value) -or
        $value.Current.Value.TrimEnd('/') -cne $arenaManagementUrl.TrimEnd('/')) {
        Manual 'Setup only acts on the browser Extensions page. Return there and use Load unpacked, or try Install in browser again.'
    }
}
function OwnedPicker($picker) {
    return $picker.Current.ClassName -eq '#32770' -and
        [ArenaSetupWindow]::GetAncestor([IntPtr]$picker.Current.NativeWindowHandle, 3).ToInt64() -eq $arenaWindowHandle
}
function AssertPicker($picker) {
    AssertForeground $picker
    if (-not (OwnedPicker $picker)) { Manual 'Setup paused because the folder picker belongs to another window. Select the Arena folder in the intended browser.' }
}
function ExtensionsDocument($window) {
    $condition = [System.Windows.Automation.AndCondition]::new(
        (TypeCondition ([System.Windows.Automation.ControlType]::Document)),
        (NameCondition 'Extensions'))
    return $window.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $condition)
}
function FindButton($root, [string]$name) {
    $condition = [System.Windows.Automation.AndCondition]::new(
        (TypeCondition ([System.Windows.Automation.ControlType]::Button)), (NameCondition $name))
    return $root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $condition)
}
function Invoke($element, $window, [bool]$async = $false) {
    if ($window.Current.ClassName -eq '#32770') { AssertPicker $window }
    else { AssertManagementPage $window }
    if (-not (Visible $element)) { Manual 'A browser control is unavailable. Complete Developer mode and Load unpacked manually.' }
    $pattern = $null
    if (-not $element.TryGetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern, [ref]$pattern)) {
        Manual 'The browser does not expose this control to automation. Use Load unpacked and select the prepared folder.'
    }
    if ($async) { [ArenaSetupWindow]::InvokeAsync($pattern) }
    else { $pattern.Invoke() }
}
try {
    $window = $null
    $document = $null
    $navigationDeadline = [DateTime]::UtcNow.AddSeconds(12)
    while ([DateTime]::UtcNow -lt $navigationDeadline) {
        $window = ForegroundBrowser
        if ($null -ne $window) { $document = ExtensionsDocument $window }
        if ($null -ne $document) { break }
        Start-Sleep -Milliseconds 200
    }
    if ($null -eq $document) {
        Manual 'Open your browser Extensions page. Enable Developer mode, choose Load unpacked, and select the prepared Arena folder. Automatic setup currently recognizes the English Chrome and Edge controls.'
    }
    AssertManagementPage $window
    $arenaWindowHandle = $window.Current.NativeWindowHandle
    $existing = $document.FindFirst([System.Windows.Automation.TreeScope]::Descendants,
        (NameCondition 'Glossarion Arena Browser Companion'))
    if ($null -ne $existing) {
        Manual 'The Arena helper is already listed. Enable or reload its card, then return to Arena Login.'
    }
    $load = FindButton $document 'Load unpacked'
    if (-not (Visible $load)) {
        $controls = $document.FindAll([System.Windows.Automation.TreeScope]::Descendants,
            (NameCondition 'Developer mode'))
        $toggled = $false
        foreach ($control in $controls) {
            $toggle = $null
            if ((Visible $control) -and $control.TryGetCurrentPattern(
                [System.Windows.Automation.TogglePattern]::Pattern, [ref]$toggle)) {
                AssertManagementPage $window
                if ($toggle.Current.ToggleState -eq [System.Windows.Automation.ToggleState]::Off) { $toggle.Toggle() }
                $toggled = $true
                break
            }
        }
        if (-not $toggled) { Manual 'Enable Developer mode on the Extensions page, then choose Load unpacked and select the prepared Arena folder.' }
        for ($attempt = 0; $attempt -lt 20; $attempt++) {
            $load = FindButton $document 'Load unpacked'
            if (Visible $load) { break }
            Start-Sleep -Milliseconds 150
        }
    }
    # A provider may keep Invoke blocked until its folder dialog closes. The
    # C# worker permits this thread to observe and complete that owned dialog.
    Invoke $load $window $true
    $picker = $null
    while ([DateTime]::UtcNow -lt $arenaDeadline) {
        $candidate = ForegroundBrowser
        if ($null -ne $candidate -and (OwnedPicker $candidate) -and
            $candidate.Current.Name -match '(?i)(select.*(extension|folder)|load.*(extension|unpacked))') {
            $picker = $candidate
            break
        }
        if ([ArenaSetupWindow]::InvokeFailed) { break }
        Start-Sleep -Milliseconds 150
    }
    if ($null -eq $picker) { Manual 'Select the prepared Arena folder in the browser folder picker, then return to Arena Login.' }
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
    Finish 'awaiting_connection' 'The Arena folder was submitted to the browser. Return to Arena Login; installation is confirmed when the helper connects.'
} catch {
    Manual 'The browser did not allow automatic setup. Use Developer mode and Load unpacked to select the prepared Arena folder, then return to Arena Login.'
}
'''


def _run_windows_installer(folder, executable, *, cancel_check, timeout=50):
    env = os.environ.copy()
    env['GLOSSARION_ARENA_SETUP_FOLDER'] = str(folder)
    env['GLOSSARION_ARENA_SETUP_BROWSER'] = str(executable)
    family = 'edge' if executable.name.lower() == 'msedge.exe' else 'chrome'
    env['GLOSSARION_ARENA_SETUP_URL'] = _BROWSERS[family][2]
    # Keep the Windows command line short; the complete static script is data
    # in this child process's environment, not a command-line interpolation.
    env['GLOSSARION_ARENA_SETUP_SCRIPT'] = base64.b64encode(_INSTALL_SCRIPT.encode('utf-8')).decode('ascii')
    powershell = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
    bootstrap = (
        "$arenaSetupSource = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String("
        "[Environment]::GetEnvironmentVariable('GLOSSARION_ARENA_SETUP_SCRIPT'))); "
        "& ([ScriptBlock]::Create($arenaSetupSource))"
    )
    encoded = base64.b64encode(bootstrap.encode('utf-16-le')).decode('ascii')
    process = subprocess.Popen(
        [str(powershell), '-NoLogo', '-NoProfile', '-NonInteractive', '-WindowStyle', 'Hidden', '-EncodedCommand', encoded],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
    )
    deadline = time.monotonic() + timeout
    try:
        while True:
            if cancel_check():
                return _result('cancelled', 'Arena extension setup was cancelled.')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return _result('manual_required', 'Automatic setup timed out. Finish Load unpacked in the browser, then return to Arena Login.')
            try:
                output, _errors = process.communicate(timeout=min(.25, remaining))
            except subprocess.TimeoutExpired:
                continue
            for line in reversed(output.decode('utf-8-sig', errors='replace').splitlines()):
                try:
                    result = json.loads(line)
                except ValueError:
                    continue
                if (isinstance(result, dict) and result.get('status') in
                        {'awaiting_connection', 'manual_required', 'cancelled', 'error'}
                        and isinstance(result.get('message'), str)):
                    return result
            return _result('manual_required', 'Automatic setup is unavailable. Enable Developer mode, choose Load unpacked, and select the prepared Arena folder.')
    finally:
        if process.poll() is None:
            process.kill()  # Only this short-lived installer; never the browser.
        process.communicate()


def install_extension(extension_path, browser_hint='', *, cancel_check=lambda: False):
    """Attempt normal installation after the user clicks Install in browser."""
    if browser_hint not in ('', 'chrome', 'edge'):
        return _result('error', 'Unsupported browser selection.')
    try:
        folder = _validated_folder(extension_path)
        if cancel_check():
            return _result('cancelled', 'Arena extension setup was cancelled.')
        if sys.platform != 'win32':
            return _result('manual_required', 'Open your browser Extensions page, enable Developer mode, choose Load unpacked, and select the prepared Arena folder.')
        browser = _find_browser(browser_hint)
        if browser is None:
            return _result('manual_required', 'Open Extensions in the browser you use for Arena, enable Developer mode, and load the prepared folder.')
        executable, url = browser
        if cancel_check():
            return _result('cancelled', 'Arena extension setup was cancelled.')
        # Normal browser invocation reuses its usual profile. No temporary
        # profile, load-extension flag, policy edit, or elevated launch is used.
        subprocess.Popen([str(executable), url], close_fds=True, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return _run_windows_installer(folder, executable, cancel_check=cancel_check)
    except (OSError, ValueError, subprocess.SubprocessError):
        return _result('manual_required', 'Automatic setup could not start. Use Developer mode and Load unpacked with the prepared Arena folder.')


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
