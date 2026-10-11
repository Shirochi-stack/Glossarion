"""Shared Chrome/Edge DevTools plumbing for the browser-backed web-chat routes.

``authds/`` (DeepSeek) and ``authchatgpt/`` (ChatGPT) each drive a dedicated, persistent
Chrome/Edge profile over the DevTools protocol: one browser process per route, one tab per
completion, so batch requests stream in parallel while only profile start-up and control
commands are serialized. Each route keeps its own error class, messages and site logic and
passes them in here; this module never imports a route.

Desktop only: :func:`launch` refuses to start a process where ``mobile_runtime`` says none can run;
routes check it first to give their own message.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import urllib.request
from contextlib import contextmanager

import mobile_runtime


def find_browser_binary(env_var, error):
    """Path of Chrome or Edge; ``env_var`` overrides. ``error(message)`` builds the route's error."""
    configured = os.environ.get(env_var)
    if configured:
        if Path(configured).is_file():
            return configured
        raise error(f"{env_var} does not point to a browser executable.")
    candidates = []
    for root in (os.environ.get("PROGRAMFILES"), os.environ.get("PROGRAMFILES(X86)"),
                 os.environ.get("LOCALAPPDATA")):
        if root:
            candidates += [str(Path(root) / p) for p in
                           ("Google/Chrome/Application/chrome.exe", "Microsoft/Edge/Application/msedge.exe")]
    candidates += ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                   "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    for name in ("google-chrome", "chromium", "chromium-browser", "microsoft-edge"):
        found = shutil.which(name)
        if found:
            return found
    return None


class CdpPage:
    """One DevTools target connection. ``check(cancel_check)`` raises the route's cancel error;
    ``error(message, error_type)`` builds its other errors."""

    command_failed = "browser command failed."
    command_timed_out = "browser command timed out."
    script_failed = "browser script failed."

    def __init__(self, url, cancel_check=None):
        import websocket
        self.ws = websocket.create_connection(url, timeout=1, suppress_origin=True,
                                              http_no_proxy=["127.0.0.1", "localhost"])
        self.sequence = 0
        self.cancel_check = cancel_check
        self.debugger_url = url

    def check(self, cancel_check):
        raise NotImplementedError

    def error(self, message, error_type):
        raise NotImplementedError

    def call(self, method, params=None, timeout=20):
        import websocket
        self.sequence += 1
        request_id = self.sequence
        self.ws.send(json.dumps({"id": request_id, "method": method, "params": params or {}}))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.check(self.cancel_check)
            try:
                message = json.loads(self.ws.recv())
            except websocket.WebSocketTimeoutException:
                continue
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise self.error(self.command_failed, "browser_error")
            return message.get("result", {})
        raise self.error(self.command_timed_out, "timeout")

    def evaluate(self, expression):
        result = self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True})
        if result.get("exceptionDetails"):
            raise self.error(self.script_failed, "browser_error")
        return result.get("result", {}).get("value")


@contextmanager
def serialized(gate, check, cancel_check=None):
    """Hold ``gate`` while staying cancellable; ``check(cancel_check)`` raises on cancel."""
    while not gate.acquire(timeout=.2):
        check(cancel_check)
    try:
        check(cancel_check)
        yield
    finally:
        gate.release()


@contextmanager
def launch(*, binary, profile, start_url, ready, page_factory, check, error, closed_message,
           connect_message, visible=False, extra_args=(), cancel_check=None):
    """Start ``binary`` on ``profile`` and yield its first tab once ``ready(page)`` is true.

    ``start_url`` is the address the first tab opens; the tab whose URL starts with it is used.
    """
    directory = Path(profile)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    port_file = directory / "DevToolsActivePort"
    port_file.unlink(missing_ok=True)
    args = [binary, "--remote-debugging-port=0", "--remote-debugging-address=127.0.0.1",
            f"--user-data-dir={directory}", "--no-first-run", "--no-default-browser-check",
            *extra_args, start_url]
    if not visible:
        args.insert(1, "--headless=new")
    # The mobile backend collector needs the process-spawn gate in this function.
    if not mobile_runtime.subprocesses_available():
        raise error(closed_message, "config_error")
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    page = None
    try:
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            check(cancel_check)
            if proc.poll() is not None:
                raise error(closed_message, "browser_error")
            try:
                port = int(port_file.read_text().splitlines()[0])
                # Bypass global HTTP proxy settings for the local debugging connection.
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(f"http://127.0.0.1:{port}/json/list", timeout=1) as response:
                    tabs = json.load(response)
                target = next(t for t in tabs if t.get("type") == "page" and
                              t.get("url", "").startswith(start_url))
                page = page_factory(target["webSocketDebuggerUrl"], cancel_check)
                ready(page, cancel_check)
                break
            except (OSError, ValueError, StopIteration, KeyError):
                time.sleep(.2)
        if page is None:
            raise error(connect_message, "browser_error")
        yield page
    finally:
        if page:
            try:
                page.ws.send(json.dumps({"id": 999999, "method": "Browser.close"}))
            except Exception:
                pass
            page.ws.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


class BrowserLeases:
    """One shared browser per route; each lease is an independent tab of it.

    ``launch(visible=, cancel_check=)`` is the route's launch context manager and
    ``page_factory(url, cancel_check)`` connects to a tab. ``prepare(page)`` runs on a new
    blank tab before it navigates (cookies, init scripts); without it the tab opens
    ``url`` directly. ``ready(page, cancel_check)`` waits until the tab is usable.
    """

    def __init__(self, *, launch, page_factory, check, url, ready, prepare=None):
        self.launch = launch
        self.page_factory = page_factory
        self.check = check
        self.url = url
        self.ready = ready
        self.prepare = prepare
        self.gate = threading.Lock()
        self.session = None

    def serialized(self, cancel_check=None, gate=None):
        return serialized(self.gate if gate is None else gate, self.check, cancel_check)

    @contextmanager
    def open(self, *, visible=False, cancel_check=None, prepare=None):
        prepare = prepare or self.prepare
        # A visible login needs the profile released by an existing headless browser.
        # Background requests may also share an already visible browser after login.
        while True:
            with self.serialized(cancel_check):
                if self.session is None:
                    manager = self.launch(visible=visible, cancel_check=cancel_check)
                    control = manager.__enter__()
                    control.cancel_check = None  # this connection belongs to all leases
                    self.session = {"manager": manager, "control": control,
                                    "visible": visible, "users": 0}
                session = self.session
                if not visible or session["visible"]:
                    session["users"] += 1
                    break
            self.check(cancel_check)
            time.sleep(.2)

        page = None
        target_id = None
        try:
            with self.serialized(cancel_check):
                control = session["control"]
                target_id = control.call("Target.createTarget",
                                         {"url": "about:blank" if prepare else self.url})["targetId"]
                page = self.page_factory(control.debugger_url.rsplit("/", 1)[0] + "/" + target_id,
                                         cancel_check)
            if prepare:
                prepare(page)
                page.call("Page.navigate", {"url": self.url})
            self.ready(page, cancel_check)
            yield page
        finally:
            # Cleanup must run even after cancellation; closing one tab must never
            # terminate the browser used by another active completion.
            if page:
                try:
                    page.ws.close()
                except Exception:
                    pass
            with self.gate:
                try:
                    if target_id:
                        session["control"].call("Target.closeTarget", {"targetId": target_id})
                except Exception:
                    pass
                finally:
                    session["users"] -= 1
                    if session["users"] == 0:
                        self.session = None
                        session["manager"].__exit__(None, None, None)


def wait_until(page, predicate, *, timeout, check, error, message, cancel_check=None,
               tolerate=("browser_error",), interval=.2):
    """Poll ``predicate`` (a JS expression) until true. Errors of a ``tolerate``d type are
    retried: navigation destroys the old JS context."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        check(cancel_check)
        try:
            if page.evaluate(predicate):
                return
        except Exception as exc:
            if getattr(exc, "error_type", None) not in tolerate:
                raise
        time.sleep(interval)
    raise error(message, "browser_error")
