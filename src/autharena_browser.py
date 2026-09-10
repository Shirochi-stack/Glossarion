"""External Chromium transport for AuthArena's isolated account profiles.

Only the browser process started here is controlled. Site credentials stay in
its own profile and page; the helper protocol contains status and output only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import quote


class BrowserConfigurationError(RuntimeError):
    pass


class _ContextChanged(RuntimeError):
    pass


def _windows_candidates():
    """Prefer the default HTTPS browser when it supports Chromium's protocol."""
    import winreg

    def read(root, key, name=""):
        try:
            with winreg.OpenKey(root, key) as handle:
                return winreg.QueryValueEx(handle, name)[0]
        except OSError:
            return ""

    progid = read(winreg.HKEY_CURRENT_USER,
                  r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\https\UserChoice", "ProgId")
    command = read(winreg.HKEY_CLASSES_ROOT, str(progid) + r"\shell\open\command") if progid else ""
    match = re.match(r'^\s*(?:"([^"]+\.exe)"|(.+?\.exe))(?:\s|$)', command, re.I)
    if match:
        executable = os.path.expandvars(match.group(1) or match.group(2))
        if Path(executable).name.lower() in {"chrome.exe", "msedge.exe", "brave.exe", "chromium.exe", "vivaldi.exe"}:
            yield executable
    for name in ("chrome.exe", "msedge.exe", "brave.exe", "chromium.exe", "vivaldi.exe"):
        for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            value = read(root, rf"Software\Microsoft\Windows\CurrentVersion\App Paths\{name}")
            if value:
                yield os.path.expandvars(value).strip('"')
    for base in (os.environ.get("LOCALAPPDATA"), os.environ.get("PROGRAMFILES"), os.environ.get("PROGRAMFILES(X86)")):
        if base:
            for suffix in ("Google/Chrome/Application/chrome.exe", "Microsoft/Edge/Application/msedge.exe",
                           "BraveSoftware/Brave-Browser/Application/brave.exe", "Chromium/Application/chrome.exe"):
                yield str(Path(base) / suffix)


def _find_browser():
    override = os.getenv("AUTHARENA_BROWSER", "").strip().strip('"')
    if override:
        executable = shutil.which(override) or os.path.expanduser(os.path.expandvars(override))
        if Path(executable).is_file():
            return str(Path(executable).resolve())
        raise BrowserConfigurationError("AUTHARENA_BROWSER must point to an installed Chrome, Edge, Brave or Chromium executable.")
    candidates = []
    if os.name == "nt":
        candidates.extend(_windows_candidates())
    elif sys.platform == "darwin":
        for base in (Path("/Applications"), Path.home() / "Applications"):
            candidates.extend(str(base / app / "Contents/MacOS" / binary) for app, binary in (
                ("Google Chrome.app", "Google Chrome"), ("Microsoft Edge.app", "Microsoft Edge"),
                ("Brave Browser.app", "Brave Browser"), ("Chromium.app", "Chromium")))
    candidates.extend(shutil.which(name) for name in (
        "google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge", "brave-browser"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise BrowserConfigurationError(
        "Arena needs an installed Chrome, Edge, Brave or Chromium browser. Install one or set AUTHARENA_BROWSER to its executable.")


def _launch_browser(executable, profile, *, visible):
    profile = (Path(profile).resolve() / "chromium")
    profile.mkdir(parents=True, exist_ok=True)
    port_file = profile / "DevToolsActivePort"
    # The caller holds the account's process/thread lock. A previous crashed
    # browser's stale port must never attach us to a different running process.
    port_file.unlink(missing_ok=True)
    command = [executable, f"--user-data-dir={profile}", "--remote-debugging-port=0",
               "--remote-debugging-address=127.0.0.1", "--no-first-run", "--no-default-browser-check"]
    if visible:
        command.append("--new-window")
    else:
        command.append("--headless=new")
    command.append("about:blank")
    kwargs = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, **kwargs)
    return process, port_file


def _wait_endpoint(process, port_file, deadline, cancelled):
    startup_deadline = min(deadline, time.monotonic() + 25)
    while time.monotonic() < startup_deadline:
        if cancelled():
            raise RuntimeError("Arena browser request cancelled")
        if process.poll() is not None:
            raise BrowserConfigurationError(
                "The external Arena browser closed before connecting. Close any browser already using this Arena account profile, then retry.")
        try:
            lines = port_file.read_text(encoding="utf-8").splitlines()
            if len(lines) >= 2 and re.fullmatch(r"\d{1,5}", lines[0]) and 0 < int(lines[0]) < 65536:
                if re.fullmatch(r"/devtools/browser/[a-zA-Z0-9-]+", lines[1]):
                    return f"ws://127.0.0.1:{int(lines[0])}{lines[1]}"
        except OSError:
            pass
        time.sleep(.1)
    raise TimeoutError("Timed out opening the external Arena browser")


class _CDP:
    """Small synchronous client restricted to our newly launched browser."""
    def __init__(self, endpoint, deadline):
        import websocket
        self.deadline = deadline
        self.serial = 0
        self.socket = websocket.create_connection(endpoint, timeout=min(5, max(.05, deadline - time.monotonic())),
                                                  suppress_origin=True, http_no_proxy=["127.0.0.1", "localhost"])

    def call(self, method, params=None, *, session=None, timeout=5):
        self.serial += 1
        packet = {"id": self.serial, "method": method, "params": params or {}}
        if session:
            packet["sessionId"] = session
        end = min(self.deadline, time.monotonic() + timeout)
        self.socket.settimeout(max(.01, end - time.monotonic()))
        self.socket.send(json.dumps(packet, ensure_ascii=False))
        while time.monotonic() < end:
            self.socket.settimeout(max(.01, end - time.monotonic()))
            data = self.socket.recv()
            if not data:
                raise RuntimeError("The external Arena browser was closed")
            message = json.loads(data)
            if message.get("id") != self.serial:
                continue
            if "error" in message:
                error = message["error"]
                if error.get("code") == -32000 and re.search(r"context|navigat", error.get("message", ""), re.I):
                    raise _ContextChanged("Arena page is navigating")
                # Do not include command arguments or browser exception values:
                # neither prompts nor credentials belong in diagnostic output.
                raise RuntimeError(f"Arena browser command {method} failed")
            return message.get("result", {})
        raise TimeoutError("Arena browser connection timed out")

    def evaluate(self, expression, session):
        result = self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True,
                                               "awaitPromise": False}, session=session)
        if result.get("exceptionDetails"):
            raise RuntimeError("Arena browser script failed")
        return result.get("result", {}).get("value")

    def close(self):
        self.socket.close()


_POLL_SCRIPT = r"""(() => {
  if (location.origin !== 'https://arena.ai') return {arena:false};
  const s = window.__glossarionArena;
  return {arena:true, loaded:document.readyState !== 'loading',
    document:String(performance.timeOrigin), phase:s?.phase || null,
    rejections:s?.rejections || 0, events:s?.events?.splice(0,64) || []};
})()"""
_DISPATCH_SCRIPT = "void window.__glossarionArena?.dispatch()"


def _control_page(cdp, session, config, prepare_script, commands, emit, deadline, cancelled=lambda: False):
    """Forward the page protocol without ever replaying an uncertain POST."""
    login_only = bool(config.get("login"))
    interactive = bool(config.get("allow_interactive", True)) or login_only
    checking_login = login_only
    pending_login_probe = False
    dispatched = False
    document = None
    installed = False
    rejections = 0
    next_probe = 0
    last_action = None
    while time.monotonic() < deadline:
        pending = []
        while True:
            try:
                pending.append(commands.get_nowait())
            except queue.Empty:
                break
        if cancelled() or "cancel" in pending:
            raise RuntimeError("Arena browser request cancelled")
        if "dispatch" in pending:
            if login_only or checking_login or dispatched:
                raise RuntimeError("Unexpected Arena dispatch command")
            # Set this before evaluating: a lost connection afterwards is an
            # uncertain submission and must never lead to a second request.
            dispatched = True
            cdp.evaluate(_DISPATCH_SCRIPT, session)
        try:
            state = cdp.evaluate(_POLL_SCRIPT, session)
        except _ContextChanged:
            if dispatched:
                raise RuntimeError("Arena navigated after dispatch; completion is uncertain and the request was not retried")
            time.sleep(.1)
            continue
        if not isinstance(state, dict) or not state.get("arena"):
            if dispatched:
                raise RuntimeError("Arena navigated after dispatch; completion is uncertain and the request was not retried")
            time.sleep(.2)
            continue
        current_document = state.get("document")
        if document != current_document:
            if dispatched:
                raise RuntimeError("Arena reloaded after dispatch; completion is uncertain and the request was not retried")
            document = current_document
            installed = False
            pending_login_probe = False
        for event in state.get("events", []):
            kind = event.get("event")
            if kind == "action":
                message = event.get("message", "Complete sign-in or verification in the external Arena browser.")
                if message != last_action:
                    emit("status", message=message)
                    last_action = message
                    if interactive:
                        cdp.call("Page.bringToFront", session=session)
                # A displayed CAPTCHA owns its page state until the user solves
                # it. Only ordinary waiting states can be replaced by probes.
                if state.get("phase") == "waiting":
                    pending_login_probe = True
                    next_probe = time.monotonic() + 2
                continue
            if kind == "rejected":
                dispatched = False
                rejections = max(rejections, state.get("rejections", 0))
            if kind == "done" and checking_login and not login_only:
                # Authentication was confirmed with a fresh server snapshot;
                # now prepare the original isolated request, exactly once.
                checking_login = False
                installed = False
                pending_login_probe = False
                continue
            if kind in {"verified", "logged_out", "rejected", "ready", "chunk", "error", "done", "status"}:
                emit(kind, **{key: value for key, value in event.items() if key != "event"})
            if kind in {"error", "done"}:
                return 0 if kind == "done" else 1
        if installed and state.get("phase") is None:
            if dispatched:
                raise RuntimeError("Arena request state was lost after dispatch; the request was not retried")
            installed = False
        if pending_login_probe and time.monotonic() >= next_probe and state.get("phase") == "waiting":
            checking_login = True
            installed = False
            pending_login_probe = False
        if not installed and state.get("loaded"):
            script = prepare_script(config.get("payload"), config["model"], max(.01, deadline - time.monotonic()),
                                    rejections, login_only=checking_login, allow_interactive=interactive)
            try:
                cdp.evaluate(script, session)
                installed = True
            except _ContextChanged:
                pass
        time.sleep(.1)
    raise TimeoutError("Arena request timed out while waiting for the external browser")


def _close_browser(cdp, process):
    if cdp:
        try:
            cdp.call("Browser.close", timeout=1)
        except Exception:
            pass
        try:
            cdp.close()
        except Exception:
            pass
    if process and process.poll() is None:
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                try:
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=3)
                except (OSError, subprocess.SubprocessError):
                    process.kill()
            else:
                process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()


def run_browser_helper(config, *, prepare_script):
    """Entry point used by the normal and frozen JSON-line child process."""
    def emit(event, **details):
        print(json.dumps({"autharena": 1, "event": event, **details}, ensure_ascii=False), flush=True)

    commands = queue.Queue()
    stopped = threading.Event()

    def read_commands():
        try:
            for line in sys.stdin:
                try:
                    command = json.loads(line).get("command")
                except (ValueError, AttributeError):
                    continue
                if command in {"dispatch", "cancel"}:
                    commands.put(command)
                if command == "cancel":
                    stopped.set()
        finally:
            stopped.set()
            commands.put("cancel")

    process = cdp = None
    try:
        # Import before launching a window, so missing dependencies fail cleanly.
        try:
            import websocket  # noqa: F401
        except ImportError as exc:
            raise BrowserConfigurationError("Arena external login needs websocket-client. Install the project's requirements.") from exc
        executable = _find_browser()
        deadline = time.monotonic() + float(config["timeout"])
        threading.Thread(target=read_commands, daemon=True).start()
        visible = bool(config.get("login") or config.get("allow_interactive", True))
        emit("status", message=f"Opening external {Path(executable).stem} for Arena account {config.get('account_id', 0)}.")
        process, port_file = _launch_browser(executable, config["profile"], visible=visible)
        endpoint = _wait_endpoint(process, port_file, deadline, stopped.is_set)
        cdp = _CDP(endpoint, deadline)
        targets = cdp.call("Target.getTargets").get("targetInfos", [])
        target = next((item["targetId"] for item in targets if item.get("type") == "page" and item.get("url") == "about:blank"), None)
        if target is None:
            target = cdp.call("Target.createTarget", {"url": "about:blank"})["targetId"]
        session = cdp.call("Target.attachToTarget", {"targetId": target, "flatten": True})["sessionId"]
        cdp.call("Page.enable", session=session)
        url = "https://arena.ai/text/direct?model_a=" + quote(config["model"], safe="")
        cdp.call("Page.navigate", {"url": url}, session=session)
        return _control_page(cdp, session, config, prepare_script, commands, emit, deadline, stopped.is_set)
    except Exception as exc:
        if process is not None and process.poll() is not None and not isinstance(exc, BrowserConfigurationError):
            emit("error", message="Arena request cancelled because the external browser was closed.", error_type="cancelled")
        else:
            emit("error", message=str(exc), error_type=("configuration" if isinstance(exc, BrowserConfigurationError)
                                                       else "timeout" if isinstance(exc, TimeoutError) else "transport"))
        return 1
    finally:
        _close_browser(cdp, process)
