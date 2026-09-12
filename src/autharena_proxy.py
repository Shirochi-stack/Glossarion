"""Arena sessions and the automatically provisioned LMArenaBridge runtime.

Route labels match AuthGPT: autharena/ is slot 0, autharena1/ is slot 1.
autharena0/ is reserved for the rotating pool.
This file is also the managed Python worker entry point (including frozen builds).
"""
from __future__ import annotations

import atexit
import base64
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import zipfile

import requests

REVISION = "e9655ea6d74cddabdfdd651da285aa4ca60091ad"
ADAPTER_VERSION = 23
CATALOG_TTL_SECONDS = 24 * 60 * 60
ARENA_RECAPTCHA_V3_SITEKEY = "6LeTGMcsAAAAALuIlkVwIxaAuZA8VledA6d3Nnb0"
UV_VERSION = "0.8.22"
ROUTE_RE = re.compile(r"^autharena(\d{0,4})(?:/|$)", re.I)
_qt_host_command = None
_lock = threading.RLock()
_file_locks = {"state": threading.RLock(), "setup": threading.RLock()}
_cancel = threading.Event()
_generation = 0
_responses = set()
_pending_requests = {}
_owned = None
_started_callback = None


class ArenaStreamError(RuntimeError):
    def __init__(self, message, http_status=None, retry_after=None, partial_response=False):
        super().__init__(message)
        try:
            status = int(http_status)
        except (ValueError, TypeError):
            status = 0
        self.http_status = status if 400 <= status <= 599 else None
        self.retry_after = retry_after
        self.partial_response = partial_response


def _http_response_error(response):
    """Preserve HTTP failures returned before the local SSE stream starts."""
    status = response.status_code
    try:
        body = response.json()
        detail = body.get("detail", body.get("error", body)) if isinstance(body, dict) else body
        if isinstance(detail, dict):
            detail = detail.get("message") or detail.get("detail") or detail
        if not isinstance(detail, str):
            detail = json.dumps(detail, ensure_ascii=False)
    except (ValueError, TypeError):
        detail = response.text
    detail = (detail or "").strip()[:8192]
    retry_after = response.headers.get("Retry-After") or response.headers.get("retry-after")
    message = f"Arena HTTP {status}" + (f": {detail}" if detail else "")
    if retry_after:
        message += f" (Retry-After: {retry_after})"
    return ArenaStreamError(message, status, retry_after)


async def _run_stream_with_idle_timeout(awaitable, activity, timeout):
    """Allow a long active response; time out only when upstream stops sending."""
    import asyncio
    task = asyncio.create_task(awaitable)
    try:
        if timeout is None:
            return await task
        while not task.done():
            activity.clear()
            wake = asyncio.create_task(activity.wait())
            try:
                ready, _ = await asyncio.wait((task, wake), timeout=timeout,
                                               return_when=asyncio.FIRST_COMPLETED)
                if not ready:
                    raise RuntimeError(f"Arena stream stalled: no upstream data for {timeout:g} seconds. Partial output was not replayed.")
            finally:
                wake.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await wake
        return await task
    except asyncio.TimeoutError as exc:
        raise RuntimeError("Arena browser stream timed out. Partial output was not replayed.") from exc
    finally:
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


async def _open_arena_login(page, navigation):
    """Reveal the sidebar once if necessary, then activate the visible login control."""
    async def click_visible(controls):
        for index in range(await controls.count()):
            control = controls.nth(index)
            if await control.is_visible():
                try:
                    await control.click(timeout=1500)
                    return True
                except Exception:
                    # Hydration/navigation can replace a control. Retry next poll.
                    return False
        return False

    async def click_login():
        for role in ("button", "link"):
            if await click_visible(page.get_by_role(
                role, name=re.compile(r"^\s*(sign\s*in|log\s*in|login)\s*$", re.I)
            )):
                return True
        return False

    if await click_login():
        return True
    if not navigation.get("sidebar_opened"):
        triggers = page.get_by_role("button", name=re.compile(
            r"(open|expand|toggle).*sidebar|sidebar.*(open|expand|toggle)|^menu$", re.I
        ))
        opened = await click_visible(triggers)
        if not opened:
            opened = await click_visible(page.locator(
                'button[data-sidebar="trigger"], button[data-slot="sidebar-trigger"], '
                'button:has(svg.lucide-panel-left), button:has(svg.lucide-panel-left-open)'
            ))
        if opened:
            navigation["sidebar_opened"] = True
            return await click_login()
    return False


def _qt_helper_command():
    if _qt_host_command is not None:
        return list(_qt_host_command)
    import importlib.util
    try:
        available = all(importlib.util.find_spec(name) is not None for name in (
            "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets"))
    except (ImportError, ValueError):
        available = False
    if not available:
        raise RuntimeError("Arena requires Qt6 WebEngine in the running application. "
                           "This installation/build does not include it; no Qt download was attempted.")
    if getattr(sys, "frozen", False):
        return [sys.executable, "--autharena-qt-browser"]
    return [sys.executable, str(_source_file("autharena_proxy.py")), "--qt-browser"]


def _qt_browser_helper(visible=False):
    """Own Qt pages on the GUI thread; the bridge drives them over loopback CDP."""
    # Windowed PyInstaller builds set Python's stdio objects to None even
    # when the parent supplied pipes. Recover those inherited handles.
    if os.name == "nt" and (sys.stdin is None or sys.stderr is None):
        import ctypes
        import msvcrt
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetStdHandle.argtypes = [ctypes.c_ulong]
        kernel.GetStdHandle.restype = ctypes.c_void_p
        for name, number, mode, flags in (("stdin", -10, "r", os.O_RDONLY),
                                          ("stderr", -12, "w", os.O_WRONLY)):
            if getattr(sys, name) is None:
                handle = kernel.GetStdHandle(number & 0xffffffff)
                fd = msvcrt.open_osfhandle(handle, flags)
                setattr(sys, name, os.fdopen(fd, mode, encoding="utf-8", buffering=1))
    from PySide6.QtCore import QObject, Signal, QTimer, QUrl, qInstallMessageHandler
    if sys.stderr is not None:
        def qt_message(kind, context, message):
            sys.stderr.write(message + "\n")
            sys.stderr.flush()
        qInstallMessageHandler(qt_message)
    from PySide6.QtWidgets import QApplication
    from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
    from PySide6.QtWebEngineWidgets import QWebEngineView

    app = QApplication(["Arena Browser"])
    app.setQuitOnLastWindowClosed(False)
    # Off-the-record profile: Arena cookies are persisted only in our encrypted store.
    profile = QWebEngineProfile(app)
    profile.setHttpCacheType(QWebEngineProfile.MemoryHttpCache)
    profile.setHttpCacheMaximumSize(16 * 1024 * 1024)
    profile.setPersistentCookiesPolicy(QWebEngineProfile.NoPersistentCookies)
    pages = {}

    class Page(QWebEnginePage):
        def createWindow(self, window_type):
            return create_page(True).page()

    def create_page(show=False):
        view = QWebEngineView()
        view.setPage(Page(profile, view))
        view.setWindowTitle("Arena Login / Verification")
        view.resize(1100, 780)
        target = view.page().devToolsId()
        pages[target] = view
        view.setUrl(QUrl("about:blank"))
        if show:
            view.show()
        return view

    class Commands(QObject):
        received = Signal(object)
    commands = Commands()
    def execute(command):
        action = command.get("action")
        if action == "new":
            create_page(True)  # Offscreen helpers still need an active rendered view.
        elif action == "show":
            view = pages.get(command.get("target"))
            if view:
                view.show(); view.raise_(); view.activateWindow()
        elif action == "close":
            view = pages.pop(command.get("target"), None)
            if view:
                view.close(); view.deleteLater()
        elif action == "quit":
            app.quit()
    commands.received.connect(execute)
    def read_commands():
        for line in sys.stdin:
            try:
                commands.received.emit(json.loads(line))
            except (ValueError, RuntimeError):
                break
        commands.received.emit({"action": "quit"})
    threading.Thread(target=read_commands, daemon=True).start()
    create_page(False)  # Keeper page for profile cookie operations.
    result = app.exec()
    for view in pages.values():
        view.close()
        view.deleteLater()
    return result


class _QtArenaPage:
    def __init__(self, owner, page, target):
        self.owner, self.page, self.target = owner, page, target

    def __getattr__(self, name):
        return getattr(self.page, name)

    async def close(self):
        if self in self.owner._pages:
            self.owner._pages.remove(self)
        self.owner.command("close", self.target)

    async def bring_to_front(self):
        self.owner.command("show", self.target)


class _QtArenaContext:
    """Adapt Qt-owned pages to the small BrowserContext surface used by Arena."""
    def __init__(self, process, browser):
        self.process, self.browser = process, browser
        self.context = browser.contexts[0]
        self._pages = []
        self._closed = False

    def is_connected(self):
        return not self._closed and self.process.poll() is None and self.browser.is_connected()

    @property
    def pages(self):
        return [p for p in self._pages if not p.page.is_closed()]

    def command(self, action, target=None):
        if self.process.poll() is not None:
            raise RuntimeError("Arena Qt browser exited.")
        self.process.stdin.write(json.dumps({"action": action, "target": target}) + "\n")
        self.process.stdin.flush()

    async def new_page(self):
        async with self.context.expect_page(timeout=15000) as pending:
            self.command("new")
        page = await pending.value
        session = await self.context.new_cdp_session(page)
        try:
            info = await session.send("Target.getTargetInfo")
        finally:
            await session.detach()
        wrapped = _QtArenaPage(self, page, info["targetInfo"]["targetId"])
        self._pages.append(wrapped)
        return wrapped

    async def _cookie_command(self, name, args=None):
        session = await self.context.new_cdp_session(self.context.pages[0])
        try:
            return await session.send(name, args or {})
        finally:
            await session.detach()

    async def add_cookies(self, cookies):
        await self._cookie_command("Network.setCookies", {"cookies": cookies})

    async def cookies(self, urls=None):
        if isinstance(urls, str):
            urls = [urls]
        method = "Network.getCookies" if urls else "Network.getAllCookies"
        result = await self._cookie_command(method, {"urls": urls} if urls else {})
        return result["cookies"]

    async def close(self):
        import asyncio
        if self._closed:
            return
        self._closed = True
        if self.process.poll() is None:
            with contextlib.suppress(Exception):
                self.command("quit")
            try:
                await asyncio.to_thread(self.process.wait, timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                await asyncio.to_thread(self.process.wait)
        with contextlib.suppress(Exception):
            await self.browser.close()
        self.process.stdin.close()


def _qt_browser_env(port, visible, recovery=False):
    env = _env()
    env["QTWEBENGINE_REMOTE_DEBUGGING"] = f"127.0.0.1:{port}"
    # Keep software rasterization available on Linux machines without a GPU.
    env["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu --disable-dev-shm-usage"
    if not visible:
        env["QT_QPA_PLATFORM"] = "offscreen"
    elif recovery and platform.system() == "Linux" and env.get("DISPLAY"):
        env["QT_QPA_PLATFORM"] = "xcb"
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        env["QTWEBENGINE_DISABLE_SANDBOX"] = "1"
    return env


async def _open_qt_browser(playwright, visible=False):
    """Retry startup only, before any login or translation has been submitted."""
    import asyncio
    import socket
    for attempt in range(2):
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        process = browser = None
        with tempfile.TemporaryFile(mode="w+b") as startup_log:
            try:
                process = subprocess.Popen(
                    _qt_helper_command()
                    + (["--visible"] if visible else []),
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=startup_log,
                    text=True, encoding="utf-8", env=_qt_browser_env(port, visible, attempt > 0),
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f"Qt WebEngine exited during startup (exit {process.returncode}).")
                    startup_log.seek(0)
                    output = startup_log.read().decode("utf-8", errors="replace")
                    endpoint = re.search(
                        rf"DevTools listening on (ws://127\.0\.0\.1:{port}/devtools/browser/[a-fA-F0-9-]+)", output)
                    if endpoint:
                        browser = await playwright.chromium.connect_over_cdp(
                            endpoint.group(1), no_defaults=True, timeout=15000)
                        return _QtArenaContext(process, browser)
                    await asyncio.sleep(.1)
                raise RuntimeError("Qt WebEngine did not become ready.")
            except BaseException as exc:
                if process is not None:
                    if process.poll() is None:
                        process.kill()
                    await asyncio.to_thread(process.wait)
                    process.stdin.close()
                if browser is not None:
                    with contextlib.suppress(Exception):
                        await browser.close()
                if not isinstance(exc, Exception) or attempt == 1:
                    raise
                print("🔄 Arena: browser startup failed; retrying with a fresh Qt WebEngine process…", flush=True)


@contextlib.asynccontextmanager
async def _regular_login_browser(playwright):
    context = await _open_qt_browser(playwright, visible=True)
    try:
        yield context
    finally:
        await context.close()


def set_proxy_started_callback(callback):
    global _started_callback
    _started_callback = callback


def data_dir():
    p = Path(os.environ.get("AUTHARENA_PROXY_DATA_DIR", str(Path.home() / ".glossarion" / "autharena_proxy")))
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_route(model):
    """Return (stored slot or None for rotation, unprefixed model)."""
    match = ROUTE_RE.match(str(model).strip())
    if not match:
        raise ValueError("Expected autharena/, autharena0/, or autharenaN/.")
    number = match.group(1)
    slot = None if number and int(number) == 0 else (int(number) if number else 0)
    return slot, str(model).strip()[match.end():]


def route_for_slot(slot, model):
    return f"autharena{int(slot) if int(slot) else ''}/{model}"


@contextlib.contextmanager
def disk_lock(name="state"):
    """Serialize setup and encrypted state writes across translation processes."""
    with _file_locks[name], open(data_dir() / (name + ".lock"), "a+b") as f:
        if f.tell() == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(.1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _load(name):
    from token_encryption import load_encrypted_tokens
    path = data_dir() / name
    return load_encrypted_tokens(str(path)) if path.exists() else {}


def _save(name, value):
    from token_encryption import save_encrypted_tokens
    path = data_dir() / name
    temporary = path.with_suffix(".tmp")
    save_encrypted_tokens(value, str(temporary))
    temporary.chmod(0o600)
    os.replace(temporary, path)


def list_accounts():
    with disk_lock():
        accounts = _load("accounts.enc")
    return [{"slot": int(k), "email": v.get("email", ""), "user_id": v.get("user_id", "")}
            for k, v in sorted(accounts.items(), key=lambda x: int(x[0]))]


def _catalog_models(value):
    """Validate full upstream records; display names alone cannot route requests."""
    if not isinstance(value, list) or not value or len(value) > 20000:
        return []
    models = []
    for item in value:
        if (not isinstance(item, dict) or not isinstance(item.get("id"), str)
                or not item["id"].strip() or not isinstance(item.get("publicName"), str)
                or not item["publicName"].strip()):
            return []
        models.append(item)
    return models


def _load_catalog():
    with disk_lock():
        saved = _load("models.enc")
    if not isinstance(saved, dict) or not _catalog_models(saved.get("models")):
        return {"models": [], "fetched_at": 0}
    return saved


def _save_catalog(models):
    models = _catalog_models(models)
    if not models:
        raise ValueError("Arena returned no valid model IDs; the saved catalog was retained.")
    with disk_lock():
        _save("models.enc", {"models": models, "fetched_at": time.time()})
    return models


def _extract_catalog(page_html):
    """Decode Next.js stream strings without relying on the following field name."""
    decoder = json.JSONDecoder()
    chunks = []
    for match in re.finditer(r"(?:self\.)?__next_f\.push\(", page_html):
        try:
            frame, _ = decoder.raw_decode(page_html[match.end():].lstrip())
            if isinstance(frame, list) and len(frame) > 1 and frame[0] == 1 and isinstance(frame[1], str):
                chunks.append(frame[1])
        except (ValueError, TypeError):
            continue
    for content in ("".join(chunks), page_html):
        for match in re.finditer(r'"initialModels"\s*:\s*', content):
            try:
                value, _ = decoder.raw_decode(content[match.end():])
                models = _catalog_models(value)
                if models:
                    return models
            except ValueError:
                continue
    return []


async def _discover_catalog(context):
    import asyncio
    page = await context.new_page()
    try:
        response = await page.goto("https://arena.ai/", wait_until="domcontentloaded", timeout=30000)
        if response is not None and response.status >= 400:
            raise RuntimeError(f"Arena catalog page returned HTTP {response.status}.")
        for _ in range(10):
            models = _extract_catalog(await page.content())
            if models:
                return models
            title = (await page.title()).lower()
            if "just a moment" in title or "verify you are human" in title:
                raise RuntimeError("Arena is requesting browser verification before loading its catalog.")
            await asyncio.sleep(.5)
        raise RuntimeError("Arena loaded, but its page did not contain readable model IDs.")
    finally:
        await page.close()


async def _ensure_catalog(context):
    saved = _load_catalog()
    if saved["models"] and time.time() - float(saved.get("fetched_at", 0)) < CATALOG_TTL_SECONDS:
        return saved["models"]
    try:
        return _save_catalog(await _discover_catalog(context))
    except Exception as exc:
        if saved["models"]:
            return saved["models"]
        raise RuntimeError(f"Arena has no saved model IDs and catalog retrieval failed: {exc} Open Arena Login to refresh the catalog from its browser.") from exc


def _env():
    # Do not inherit translation prompts or API keys into compiler/runtime processes.
    keep = {"systemroot", "windir", "comspec", "path", "pathext", "temp", "tmp",
            "home", "userprofile", "localappdata", "appdata", "programfiles", "programfiles(x86)", "lang", "lc_all",
            "display", "xauthority", "wayland_display", "xdg_session_type", "xdg_runtime_dir", "xdg_config_home", "dbus_session_bus_address"}
    env = {k: v for k, v in os.environ.items() if k.lower() in keep}
    env.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", UV_PYTHON_INSTALL_DIR=str(data_dir() / "python"))
    env["AUTHARENA_PROXY_DATA_DIR"] = str(data_dir())
    env["PLAYWRIGHT_BROWSERS_PATH"] = str(data_dir() / "browsers")
    return env


def _run(args, log_fn=print):
    result = subprocess.run([str(a) for a in args], env=_env(), capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=900,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError("Arena runtime setup failed: " + result.stderr[-1800:])
    return result.stdout.strip()


def _ensure_browser(runtime, python, log_fn=print, refresh=False):
    marker = runtime / "qt-bridge-ready"
    if not refresh and marker.exists():
        return
    probe = [python, "-c", "from playwright.async_api import BrowserType; import inspect; assert 'no_defaults' in inspect.signature(BrowserType.connect_over_cdp).parameters"]
    try:
        _run(probe, log_fn)
    except RuntimeError:
        log_fn("🌐 Arena: updating Playwright bridge support (Qt WebEngine is reused from the app)…")
        uv = data_dir() / ("uv.exe" if os.name == "nt" else "uv")
        _run([uv, "pip", "install", "--python", python, "playwright>=1.60"], log_fn)
        _run(probe, log_fn)
    marker.write_text("qt6", encoding="ascii")


def _download(url, target):
    with requests.get(url, stream=True, timeout=(15, 180)) as response:
        response.raise_for_status()
        with open(target, "wb") as out:
            for block in response.iter_content(1024 * 1024):
                out.write(block)


def _extract(archive, target):
    """Reject traversal and links before extracting downloaded archives."""
    target = Path(target).resolve()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                if not (target / item.filename).resolve().is_relative_to(target):
                    raise RuntimeError("Unsafe runtime archive")
                if (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise RuntimeError("Runtime archive contains a symbolic link")
            z.extractall(target)
    else:
        with tarfile.open(archive) as t:
            for item in t.getmembers():
                if not (target / item.name).resolve().is_relative_to(target) or item.issym() or item.islnk() or item.isdev():
                    raise RuntimeError("Unsafe runtime archive")
            t.extractall(target, filter="data")


def _source_file(name):
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    for p in (root / name, root / "src" / name, Path(__file__).with_name(name)):
        if p.is_file():
            return p
    raise RuntimeError(f"Arena packaged runtime source missing: {name}")


def _ensure_runtime(log_fn=print):
    root = data_dir()
    runtime = root / ("bridge-" + REVISION)
    python = runtime / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if (runtime / "ready").exists() and python.exists():
        _ensure_browser(runtime, python, log_fn)
        return runtime, python
    log_fn("Arena: downloading the managed Python runtime and LMArenaBridge…")
    machine = platform.machine().lower()
    arch = "aarch64" if machine in ("arm64", "aarch64") else "x86_64"
    system = platform.system()
    triples = {"Windows": f"{arch}-pc-windows-msvc", "Darwin": f"{arch}-apple-darwin", "Linux": f"{arch}-unknown-linux-gnu"}
    if system not in triples or machine not in ("amd64", "x86_64", "arm64", "aarch64"):
        raise RuntimeError("Arena automatic runtime setup does not support this OS/architecture.")
    uv = root / ("uv.exe" if os.name == "nt" else "uv")
    with tempfile.TemporaryDirectory(prefix="setup-", dir=root) as work:
        work = Path(work)
        if not uv.exists():
            asset = f"uv-{triples[system]}." + ("zip" if os.name == "nt" else "tar.gz")
            archive = work / asset
            _download(f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{asset}", archive)
            _extract(archive, work / "uv")
            candidate = next((work / "uv").rglob(uv.name))
            shutil.copy2(candidate, uv)
            uv.chmod(0o700)
        archive = work / "bridge.zip"
        _download(f"https://github.com/CloudWaddie/LMArenaBridge/archive/{REVISION}.zip", archive)
        _extract(archive, work / "bridge")
        extracted = next((work / "bridge").iterdir())
        code = (extracted / "src/main.py").read_text(encoding="utf-8")
        for symbol in ("async def api_chat_completions", "async def get_initial_data", "STRICT_BROWSER_FETCH_MODELS"):
            if symbol not in code:
                raise RuntimeError("LMArenaBridge adapter compatibility check failed.")
        # venv paths are absolute; build in a version-specific final directory.
        # The ready marker is the atomic publication boundary for other processes.
        runtime.mkdir(exist_ok=True)
        shutil.copytree(extracted, runtime / "bridge", dirs_exist_ok=True)
        _run([uv, "python", "install", "3.12"], log_fn)
        _run([uv, "venv", "--python", "3.12", runtime / "venv"], log_fn)
        log_fn("Arena: installing proxy dependencies…")
        _run([uv, "pip", "install", "--python", python, "-r", runtime / "bridge/requirements.txt", "requests", "cryptography"], log_fn)
        _ensure_browser(runtime, python, log_fn, refresh=True)
        (runtime / "ready").write_text(REVISION, encoding="ascii")
    return runtime, python


def check_proxy_health():
    try:
        state = _load("service.enc")
        response = requests.get(state["url"] + "/health", headers={"Authorization": "Bearer " + state["key"]}, timeout=2)
        compatible = response.ok and response.json().get("revision") == REVISION
        return {"running": compatible and response.json().get("adapter_version") == ADAPTER_VERSION,
                "outdated": compatible and response.json().get("adapter_version") != ADAPTER_VERSION, **state}
    except Exception:
        return {"running": False}


def ensure_proxy_running(log_fn=print, notify_started=True):
    global _owned
    with disk_lock("setup"):
        status = check_proxy_health()
        if status["running"]:
            return status
        if status.get("outdated"):
            requests.post(status["url"] + "/shutdown", headers={"Authorization": "Bearer " + status["key"]}, timeout=5).raise_for_status()
        qt_command = _qt_helper_command()
        runtime, python = _ensure_runtime(log_fn or print)
        for name in ("autharena_proxy.py", "token_encryption.py"):
            shutil.copy2(_source_file(name), runtime / name)
        key = secrets.token_urlsafe(32)
        # The child binds port 0 itself and publishes its authenticated endpoint.
        _owned = subprocess.Popen([str(python), str(runtime / "autharena_proxy.py"), "--worker"],
                                 stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 env=_env(), cwd=runtime, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        _owned.stdin.write(json.dumps({"key": key, "qt_helper_command": qt_command}).encode() + b"\n")
        _owned.stdin.close()
        # Worker publishes state without taking the parent-held setup lock.
        for _ in range(120):
            status = check_proxy_health()
            if status["running"] and status.get("key") == key:
                if notify_started and _started_callback:
                    _started_callback()
                return status
            if _owned.poll() is not None:
                raise RuntimeError("Arena proxy exited during initialization. Check the installed runtime dependencies.")
            time.sleep(.25)
    raise RuntimeError("Arena proxy did not become healthy.")


def _request(path, payload=None, timeout=600, status=None):
    if status is None:
        status = ensure_proxy_running()
    method = requests.post if payload is not None else requests.get
    response = method(status["url"] + path, headers={"Authorization": "Bearer " + status["key"]},
                      **({"json": payload} if payload is not None else {}), timeout=timeout)
    if not response.ok:
        raise RuntimeError(response.json().get("detail", "Arena request failed"))
    return response.json()


def open_login(log_fn=print, account_id=0):
    if log_fn:
        log_fn("Arena Login: preparing the proxy and internal browser…")
    status = ensure_proxy_running(log_fn=log_fn)
    if log_fn:
        log_fn("Arena Login: opening the internal browser. Finish sign-in on the Arena page.")
    return _request("/login", {"slot": account_id}, timeout=660, status=status)


def list_models(timeout=30):
    # A stopped local worker is not a logged-out account. Catalog polling runs
    # off the GUI thread; _request starts the worker and restores saved sessions.
    if not list_accounts():
        return []
    return _request("/v1/models", timeout=timeout).get("data", [])


def capture_cancel_generation():
    return _generation


def is_cancel_generation_cancelled(generation):
    return _cancel.is_set() or (generation is not None and generation != _generation)


def is_cancelled():
    return _cancel.is_set()


def cancel_stream():
    global _generation
    with _lock:
        _generation += 1
        _cancel.set()
        active = list(_responses)
        pending = list(_pending_requests.items())
    def notify_worker():
        for request_id, state in pending:
            with contextlib.suppress(Exception):
                requests.post(state["url"] + "/cancel", json={"id": request_id},
                              headers={"Authorization": "Bearer " + state["key"]}, timeout=2)
    if pending:
        threading.Thread(target=notify_worker, daemon=True, name="Arena cancellation").start()
    for response in active:
        # Interrupt an outstanding read before close() waits for urllib3's lock.
        with contextlib.suppress(Exception):
            import socket
            response.raw._fp.fp.raw._sock.shutdown(socket.SHUT_RDWR)
        with contextlib.suppress(Exception):
            response.close()


def reset_cancel():
    _cancel.clear()


def visible_stream():
    name, default = ("ALLOW_AUTHGPT_BATCH_STREAM_LOGS", "0") if os.getenv("BATCH_TRANSLATION") == "1" else ("LOG_STREAM_CHUNKS", "1")
    return os.getenv(name, default).strip().lower() not in ("", "0", "false", "no", "off")


class _ArenaStreamLog:
    """Group deltas into readable log records, as the AuthGPT adapter does."""
    def __init__(self, log_fn):
        self.log_fn = log_fn
        self.phase = None
        self.buffer = ""

    def _emit(self, text):
        prefix = "    " if self.phase == "thinking" and text else ""
        self.log_fn(prefix + text.replace("\x1f", "\\x1F"))

    def flush(self):
        if self.log_fn and self.buffer.strip(" \t\r\n"):
            self._emit(self.buffer.strip(" \t\r\n"))
        self.buffer = ""

    def append(self, chunk, phase):
        if not self.log_fn:
            return
        if phase != self.phase:
            self.flush()
            if self.phase is not None:
                self.log_fn("─" * 50)
            self.log_fn("📡 Arena: Text streaming..." if phase == "text" else "🧠 [autharena] Thinking...")
            self.phase = phase
        combined = self.buffer + chunk
        for tag in ("</h1>", "</h2>", "</h3>", "</h4>", "</h5>", "</h6>", "</p>"):
            combined = combined.replace(tag, tag + "\n")
        if "\n" in combined:
            parts = combined.split("\n")
            for part in parts[:-1]:
                self._emit(part)
            self.buffer = parts[-1]
        elif len(combined) > 150:
            self._emit(combined)
            self.buffer = ""
        else:
            self.buffer = combined


def consume_stream(lines, log_fn=print, log_stream=True, cancel_generation=None, progress_callback=None):
    text, thinking, usage, finish = [], [], None, None
    done = False
    display = _ArenaStreamLog(log_fn if log_stream else None)
    try:
        for line in lines:
            if is_cancel_generation_cancelled(cancel_generation):
                raise RuntimeError("Arena stream cancelled")
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            if event.get("arena_progress"):
                if progress_callback:
                    progress_callback(event["arena_progress"])
                continue
            if event.get("error"):
                error = event["error"]
                message = str((error.get("message") or "") if isinstance(error, dict) else error).strip()
                if not message:
                    phase = "after reasoning, before answer text" if thinking and not text else "before completion"
                    kind = str(error.get("type") or "unspecified upstream error") if isinstance(error, dict) else "unspecified upstream error"
                    message = f"Arena ended the stream {phase} without an error message ({kind})."
                status = error.get("status_code") if isinstance(error, dict) else None
                retry_after = error.get("retry_after") if isinstance(error, dict) else None
                if retry_after:
                    message += f" (Retry-After: {retry_after})"
                raise ArenaStreamError("Arena stream failed: " + message, status, retry_after, bool(text or thinking))
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                if choice.get("index", 0) != 0:
                    continue
                delta = choice.get("delta", {})
                for field, target in (("content", text), ("reasoning_content", thinking), ("reasoning", thinking)):
                    chunk = delta.get(field)
                    if isinstance(chunk, str) and chunk:
                        target.append(chunk)
                        display.append(chunk, "text" if field == "content" else "thinking")
                finish = choice.get("finish_reason") or finish
    finally:
        display.flush()
    if is_cancel_generation_cancelled(cancel_generation):
        raise RuntimeError("Arena stream cancelled")
    if not done or not finish:
        raise ArenaStreamError("Arena stream interrupted before its completion marker; partial output was not retried.",
                               partial_response=bool(text or thinking))
    if display.log_fn and display.phase is not None:
        display.log_fn("📡 Arena: Stream complete")
    return {"content": "".join(text), "reasoning_content": "".join(thinking), "usage": usage, "finish_reason": finish}


def send_message_stream(messages, model, temperature=0.7, max_tokens=None, timeout=600,
                        log_fn=print, log_stream=None, account_id=0, cancel_generation=None,
                        before_send_callback=None, progress_label=None):
    generation = capture_cancel_generation() if cancel_generation is None else cancel_generation
    if is_cancel_generation_cancelled(generation):
        raise RuntimeError("Arena stream cancelled")
    status = ensure_proxy_running(log_fn=log_fn)
    if is_cancel_generation_cancelled(generation):
        raise RuntimeError("Arena stream cancelled")
    payload = {"model": model, "messages": messages, "stream": True, "temperature": temperature, "account_slot": account_id, "dispatch_ack": True, "stream_timeout": timeout}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    request_id = secrets.token_hex(16)
    payload["request_id"] = request_id
    with _lock:
        _pending_requests[request_id] = status
    response = None
    def progress(stage):
        if stage == "dispatch":
            if is_cancel_generation_cancelled(generation):
                raise RuntimeError("Arena stream cancelled")
            if before_send_callback:
                before_send_callback()
            approved = requests.post(status["url"] + "/dispatch", json={"id": request_id},
                                     headers={"Authorization": "Bearer " + status["key"]}, timeout=10)
            if approved.status_code == 409:
                if is_cancel_generation_cancelled(generation):
                    raise RuntimeError("Arena stream cancelled")
                raise ArenaStreamError("Arena dispatch preparation ended before handoff; no automatic replay was attempted.", 409)
            if approved.status_code >= 400:
                raise _http_response_error(approved)
            approved.raise_for_status()
            if log_fn:
                log_fn("📨 Arena: captcha token acquired; sending Arena request")
                log_fn(progress_label or f"📤 [{threading.current_thread().name}] API call in progress")
        elif log_fn:
            if stage.startswith("headers:"):
                log_fn("📥 Arena: response headers received (HTTP " + stage.split(":", 1)[1] + ")")
                return
            labels = {"captcha": "🔐 Arena: requesting a fresh captcha token…",
                      "token": "✅ Arena: captcha token received",
                      "retry": "🔁 Arena: captcha rejected; retrying with a fresh token",
                      "verification": "🔄 Arena: CAPTCHA rejected; LMArenaBridge is obtaining a fresh token in the Arena browser…",
                      "browser_verification": "🌐 Arena: security verification required. Complete it in the Arena browser window; this request will wait before submission.",
                      "headers": "📥 Arena: response headers received"}
            if stage in labels:
                log_fn(labels[stage])
    try:
        if log_fn:
            log_fn("🌐 Arena: preparing saved session and browser…")
        if is_cancel_generation_cancelled(generation):
            raise RuntimeError("Arena stream cancelled")
        response = requests.post(status["url"] + "/v1/chat/completions", json=payload,
                                 headers={"Authorization": "Bearer " + status["key"]}, stream=True, timeout=(15, timeout))
        with _lock:
            _responses.add(response)
        if is_cancel_generation_cancelled(generation):
            raise RuntimeError("Arena stream cancelled")
        if not response.ok:
            raise _http_response_error(response)
        selected = getattr(response, "headers", {}).get("X-Arena-Account-Slot")
        if log_fn and selected is not None and str(selected).isdigit():
            selected = int(selected)
            saved = next((a for a in list_accounts() if a["slot"] == selected), {})
            identity = saved.get("email") or "saved account"
            log_fn(f"👤 Arena: using account #{selected} ({identity})" + (" — rotation" if account_id is None else ""))
        response.encoding = "utf-8"
        return consume_stream(response.iter_lines(decode_unicode=True, chunk_size=1), log_fn,
                              visible_stream() if log_stream is None else log_stream, generation, progress)
    except Exception:
        with contextlib.suppress(Exception):
            requests.post(status["url"] + "/cancel", json={"id": request_id},
                          headers={"Authorization": "Bearer " + status["key"]}, timeout=2)
        if is_cancel_generation_cancelled(generation):
            raise RuntimeError("Arena stream cancelled") from None
        raise
    finally:
        with _lock:
            _responses.discard(response)
            _pending_requests.pop(request_id, None)
        if response is not None:
            response.close()


send_message = send_message_stream


def shutdown_proxy():
    cancel_stream()
    if _owned is not None and _owned.poll() is None:
        try:
            state = _load("service.enc")
            requests.post(state["url"] + "/shutdown", headers={"Authorization": "Bearer " + state["key"]}, timeout=2)
            _owned.wait(timeout=5)
        except Exception:
            _owned.terminate()


atexit.register(shutdown_proxy)


def session_from_cookies(cookies):
    allowed = [c for c in cookies if c.get("domain", "").lstrip(".") in ("arena.ai", "lmarena.ai")]
    chunks = {c["name"]: c["value"] for c in allowed}
    token = chunks.get("arena-auth-prod-v1", "")
    if not token:
        parts = sorted((int(k.rsplit(".", 1)[1]), v) for k, v in chunks.items()
                       if re.fullmatch(r"arena-auth-prod-v1\.\d+", k))
        token = "".join(v for _, v in parts)
    try:
        raw = token.removeprefix("base64-")
        session = json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))
        user = session.get("user", {})
        if not user.get("id") or user.get("is_anonymous", False) or not user.get("email"):
            return None
        if session.get("expires_at", 0) <= time.time():
            return None
        return {"token": token, "cookies": allowed, "email": user["email"], "user_id": user["id"],
                "expires_at": session["expires_at"]}
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def _persist_session(slot, cookies, expected_token=None):
    """Keep refreshed Arena credentials across worker/app restarts."""
    session = session_from_cookies(cookies)
    if session is None:
        return None
    with disk_lock():
        accounts = _load("accounts.enc")
        previous = accounts.get(str(slot))
        if not previous or previous.get("user_id") != session["user_id"]:
            raise RuntimeError("Arena restored a different account; reconnect the intended account with Arena Login.")
        if expected_token is not None and previous.get("token") != expected_token:
            # Restoration started before another request or Arena Login saved
            # a different session. It must not overwrite those credentials.
            return session
        # Concurrent contexts can finish out of order. Keep newer refreshed
        # credentials even if an older, still-valid session finishes last.
        if session["expires_at"] >= previous.get("expires_at", 0):
            accounts[str(slot)] = session
            _save("accounts.enc", accounts)
    return session


async def _restore_login_session(context, slot):
    """Seed a fresh login profile from the selected account's encrypted cookies."""
    if slot is None:
        return  # + N must open a separate, unsigned-in account.
    with disk_lock():
        account = _load("accounts.enc").get(str(slot))
    if account:
        # Preserve expired access sessions too: their refresh cookie may still
        # allow Arena's session client to renew them after navigation.
        await context.add_cookies(account.get("cookies", []))


async def _serve_worker(key, qt_helper_command=None):
    """Authenticated loopback broker; independent browser/bridge state per request."""
    global _qt_host_command
    _qt_host_command = qt_helper_command
    import asyncio
    import copy
    import importlib
    import socket
    import types
    from fastapi import FastAPI, HTTPException, Request
    globals()["Request"] = Request  # FastAPI resolves postponed endpoint annotations.
    from starlette.responses import JSONResponse, StreamingResponse
    from playwright.async_api import async_playwright
    import uvicorn

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    playwright = await async_playwright().start()
    slots = {}
    slot_pool_lock = asyncio.Lock()
    login_lock = asyncio.Lock()
    cursor = 0
    jobs = {}
    cancelled_jobs = set()
    dispatch_acks = {}

    @app.middleware("http")
    async def authorize(request, call_next):
        if not secrets.compare_digest(request.headers.get("authorization", ""), "Bearer " + key):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        try:
            return await call_next(request)
        except Exception:
            return JSONResponse({"detail": "Arena operation failed. Reconnect with Arena Login."}, status_code=503)

    @app.get("/health")
    async def health():
        return {"revision": REVISION, "adapter_version": ADAPTER_VERSION}

    @app.post("/login")
    async def login(request: Request):
        body = await request.json()
        async with login_lock:
            login_browser = _regular_login_browser(playwright)
            try:
                context = await login_browser.__aenter__()
            except Exception as exc:
                raise HTTPException(503, f"Arena Qt browser startup failed after automatic recovery: {exc}")
            try:
                await _restore_login_session(context, body.get("slot"))
                page = context.pages[0] if context.pages else await context.new_page()
                await page.goto("https://arena.ai/", wait_until="domcontentloaded")
                await page.bring_to_front()
                clicked = False
                navigation = {}
                deadline = time.monotonic() + 300
                while time.monotonic() < deadline:
                    cookies = await context.cookies(["https://arena.ai/", "https://lmarena.ai/"])
                    account = session_from_cookies(cookies)
                    if account and body.get("slot") is not None:
                        expected = next((item for item in list_accounts() if item["slot"] == int(body["slot"])), None)
                        if expected and expected.get("user_id") and expected["user_id"] != account["user_id"]:
                            raise HTTPException(409, f"The internal browser is signed into a different Arena account. Switch accounts on the Arena page, then reconnect account {body['slot']}.")
                    if account and body.get("slot") is None:
                        known = list_accounts()
                        if any(v["user_id"] == account["user_id"] for v in known):
                            await asyncio.sleep(1)
                            continue
                    if account:
                        # Reconnect future requests immediately. Active requests
                        # finish in their own contexts before those are retired.
                        async with slot_pool_lock:
                            with disk_lock():
                                saved = _load("accounts.enc")
                                slot = body.get("slot")
                                if slot is None:
                                    slot = max([int(k) for k in saved] + [-1]) + 1
                                slot = int(slot)
                                if slot < 0 or slot > 9998:
                                    raise HTTPException(400, "Invalid Arena account slot")
                                saved[str(slot)] = account
                                _save("accounts.enc", saved)
                            previous = slots.pop(slot, [])
                            for state in previous:
                                state["retired"] = True
                        for state in previous:
                            if not state["lock"].locked():
                                await close_slot(state)
                        # Capture full routing metadata while the regular login
                        # browser is available; background requests can reuse it.
                        catalog_cached = False
                        try:
                            page_models = _extract_catalog(await page.content())
                            if page_models:
                                _save_catalog(page_models)
                            else:
                                await _ensure_catalog(context)
                            catalog_cached = True
                        except Exception:
                            pass  # Keep the successful encrypted login.
                        return {"slot": slot, "email": account["email"], "catalog_cached": catalog_cached}
                    if not clicked:
                        clicked = await _open_arena_login(page, navigation)
                    await asyncio.sleep(1 if clicked else .2)
                raise HTTPException(408, "Arena Login timed out before a signed-in session was available.")
            finally:
                with contextlib.suppress(Exception):
                    await login_browser.__aexit__(None, None, None)

    async def get_slot(slot):
        """Reserve an idle context, or create another for overlapping requests."""
        while True:
            async with slot_pool_lock:
                pool = slots.setdefault(slot, [])
                for state in list(pool):
                    if not state["context"].is_connected():
                        state["retired"] = True
                        pool.remove(state)
                        if not state["lock"].locked():
                            await close_slot(state)
                    elif not state["lock"].locked():
                        await state["lock"].acquire()
                        return state
            # Navigation and catalog loading must not block other batch calls.
            state = await initialize_slot(slot)
            reserved = False
            try:
                async with slot_pool_lock:
                    if slots.get(slot) is pool:
                        await state["lock"].acquire()
                        pool.append(state)
                        reserved = True
                        return state
            finally:
                if not reserved:
                    # Login changed the account during restoration, or the
                    # caller cancelled while waiting to reserve this context.
                    await close_slot(state)

    async def close_slot(state):
        state["retired"] = True
        with contextlib.suppress(Exception):
            await state["context"].close()
        namespace = state["namespace"]
        for name in list(sys.modules):
            if name == namespace or name.startswith(namespace + "."):
                sys.modules.pop(name, None)

    async def release_slot(state):
        try:
            # Upstream launches browser tasks separately from its stream
            # iterator. Drain them before this context can be reused.
            pending = [task for task in state.get("transport_tasks", ())
                       if task is not asyncio.current_task() and not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            main = state["main"]
            main.chat_sessions.clear()
            main.conversation_tokens.clear()
            main.request_failed_tokens.clear()
            async with slot_pool_lock:
                pool = slots.get(state.get("slot"), [])
                if not state["retired"] and any(other is not state and not other["lock"].locked() for other in pool):
                    state["retired"] = True
                    if state in pool:
                        pool.remove(state)
            if state["retired"]:
                await close_slot(state)
        finally:
            state["lock"].release()

    async def restore_context(slot, refresh=False):
        with disk_lock():
            account = _load("accounts.enc").get(str(slot))
        if not account:
            raise HTTPException(401, f"Arena account {slot} needs Arena Login.")
        context = await _open_qt_browser(playwright)
        try:
            await context.add_cookies(account["cookies"])
            if refresh or account.get("expires_at", 0) <= time.time() + 60:
                page = await context.new_page()
                try:
                    await page.goto("https://arena.ai/", wait_until="domcontentloaded")
                    # Let Arena's own session client refresh its saved session.
                    # Access-token expiry alone must not discard a refresh token.
                    for _ in range(60):
                        restored = _persist_session(slot, await context.cookies(["https://arena.ai/", "https://lmarena.ai/"]),
                                                    expected_token=account.get("token"))
                        if restored:
                            account = restored
                            break
                        await asyncio.sleep(.5)
                    else:
                        raise HTTPException(401, f"Arena account {slot} could not refresh its saved session. Use Arena Login.")
                finally:
                    await page.close()
            return context, account
        except BaseException:
            await context.close()
            raise

    async def initialize_slot(slot):
        context, account = await restore_context(slot)
        namespace = f"arena_slot_{slot}_{secrets.token_hex(4)}"
        try:
            return await configure_slot(slot, context, account, namespace)
        except BaseException:
            await close_slot({"context": context, "namespace": namespace})
            raise

    async def configure_slot(slot, context, account, namespace):
        package = types.ModuleType(namespace)
        package.__path__ = [str(Path(__file__).parent / "bridge")]
        sys.modules[namespace] = package
        main = importlib.import_module(namespace + ".src.main")
        config_module = importlib.import_module(namespace + ".src.config")
        auth = importlib.import_module(namespace + ".src.auth")
        transport = importlib.import_module(namespace + ".src.transport")
        recaptcha = importlib.import_module(namespace + ".src.recaptcha")
        if not all(callable(getattr(recaptcha, name, None)) for name in (
            "_mint_recaptcha_v3_token_in_page", "refresh_recaptcha_token", "get_cached_recaptcha_token"
        )):
            raise RuntimeError("Pinned LMArenaBridge CAPTCHA helpers are incompatible with this adapter.")
        cfg = {"auth_tokens": [account["token"]], "auth_token": account["token"], "api_keys": [{"key": key, "rpm": 100000}],
               "persist_arena_auth_cookie": False, "browser_cookies": {c["name"]: c["value"] for c in account["cookies"]}}
        config_module._apply_config_defaults(cfg)
        # Upstream's default 120s outer timeout includes navigation, interactive
        # verification, token minting and dispatch acknowledgment together.
        cfg["chrome_fetch_outer_timeout_seconds"] = 300
        cfg["camoufox_fetch_outer_timeout_seconds"] = 300
        try:
            models = list(await _ensure_catalog(context))
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise HTTPException(503, str(exc)) from exc
        def save_config(value, **kwargs):
            cfg.update(copy.deepcopy(value))
        def save_models(value):
            validated = _save_catalog(value)
            models[:] = validated
        for module in (main, config_module):
            module.get_config = lambda: copy.deepcopy(cfg)
            module.save_config = save_config
            module.get_models = lambda: list(models)
            module.save_models = save_models
        main.DEBUG = False
        main.debug_print = lambda *args, **kwargs: None
        main.print = lambda *args, **kwargs: None
        # Discovery and requests use isolated contexts in the app-owned browser.
        # No extension, personal browser profile, or CAPTCHA solver is used.
        class ContextLease:
            def __init__(self, **kwargs):
                self.before = set(context.pages)
            async def __aenter__(self):
                return context
            async def __aexit__(self, *args):
                for p in context.pages:
                    if p not in self.before:
                        await p.close()
        async def no_challenge_click(*args, **kwargs):
            return False
        main.AsyncCamoufox = ContextLease
        main.click_turnstile = no_challenge_click
        main._userscript_proxy_is_active = lambda: False
        main.find_chrome_executable = lambda: "managed-chromium"
        async def no_refresh(*args, **kwargs):
            return None
        main.maybe_refresh_expired_auth_tokens = no_refresh
        main.maybe_refresh_expired_auth_tokens_via_lmarena_http = no_refresh
        # Keep upstream cache/refresh; mint only in this account's request page.
        captcha_page = None
        async def mint_for_slot():
            if captcha_page is None:
                return None
            return await recaptcha._mint_recaptcha_v3_token_in_page(
                captcha_page, sitekey=ARENA_RECAPTCHA_V3_SITEKEY, action="chat_submit")
        recaptcha.get_recaptcha_v3_token = mint_for_slot
        main.refresh_recaptcha_token = recaptcha.refresh_recaptcha_token
        main.get_cached_recaptcha_token = recaptcha.get_cached_recaptcha_token
        async def refresh_initial_data():
            models[:] = await _ensure_catalog(context)
        main.get_initial_data = refresh_initial_data
        main.STRICT_BROWSER_FETCH_MODELS = {m["publicName"] for m in models}
        state = {"main": main, "context": context, "lock": asyncio.Lock(), "models": models,
                 "namespace": namespace, "slot": slot, "retired": False, "submitted": False, "usage": None}

        async def fetch(http_method, url, payload, auth_token="", timeout_seconds=120, **kwargs):
            verification = bool(kwargs.pop("_verification", False))
            interactive = bool(kwargs.pop("_interactive", verification))
            if interactive:
                timeout_seconds = max(timeout_seconds, 360)
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if parsed.hostname not in ("arena.ai", "lmarena.ai"):
                raise RuntimeError("Unexpected Arena transport origin")
            if (http_method.upper() == "POST"
                    and parsed.path.rstrip("/") == "/nextjs-api/stream/create-evaluation"
                    and payload.get("mode") == "direct"):
                # Arena's current Direct UI creates direct-battle sessions.
                # Legacy direct sessions can continue but cannot be created.
                # Each Glossarion request is a fresh first turn with model A.
                payload = dict(payload, mode="direct-battle")
            page = await context.new_page()
            queue = asyncio.Queue(maxsize=32)
            headers_ready = asyncio.Event()
            done_event = asyncio.Event()
            upstream_activity = asyncio.Event()
            result = {"status": 502, "headers": {}}
            state["upstream_error"] = None
            request_events = state["events"]
            request_dispatch_ack = state.get("dispatch_ack")
            async def emit(source, item):
                if item.get("activity"):
                    upstream_activity.set()
                elif item.get("dispatching"):
                    if request_dispatch_ack is not None:
                        request_dispatch_ack.clear()
                        await request_events.put({"arena_progress": "dispatch"})
                        await request_dispatch_ack.wait()
                    state["dispatched"] = True
                elif item.get("token_received"):
                    await request_events.put({"arena_progress": "token"})
                elif "status" in item:
                    upstream_activity.set()
                    result.update(item)
                    if item["status"] >= 400:
                        state["upstream_error"] = {
                            "message": f"Arena HTTP {item['status']}: {item.get('error_body', '')}",
                            "status_code": item["status"],
                            "retry_after": (item.get("headers", {}).get("retry-after")
                                            or item.get("headers", {}).get("Retry-After")),
                        }
                    headers_ready.set()
                    await request_events.put({"arena_progress": "headers:" + str(item["status"])})
                elif "line" in item:
                    upstream_activity.set()
                    await queue.put(item["line"])
            await page.expose_binding("arenaEmit", emit)
            navigation = await page.goto("https://arena.ai/", wait_until="domcontentloaded")
            if interactive:
                await page.bring_to_front()
            # A security interstitial has no Arena reCAPTCHA loader. Waiting for
            # grecaptcha there can never work; allow the user to verify first.
            if "just a moment" in (await page.title()).lower():
                if not interactive:
                    await page.close()
                    raise RuntimeError("ARENA_BROWSER_CHALLENGE")
                await state["events"].put({"arena_progress": "browser_verification"})
                try:
                    await page.wait_for_function(
                        "() => !document.title.toLowerCase().includes('just a moment') && !!document.querySelector('script[src*=\"recaptcha/\"]')",
                        timeout=180000)
                except Exception as exc:
                    await page.close()
                    raise RuntimeError("Arena security verification did not complete. If the challenge is unresponsive, check DNS/network access to its challenge domain. No translation was submitted.") from exc
            elif navigation is not None and navigation.status >= 400:
                await page.close()
                raise RuntimeError(f"Arena homepage returned HTTP {navigation.status}; no translation was submitted.")
            sitekey, action = main.get_recaptcha_settings(cfg)
            # Arena's getRecaptchaV3Token uses this distinct v3 key. The loader
            # render parameter may instead be its v2 widget key; they are not
            # interchangeable. The pinned bridge still ships an older v3 key.
            sitekey = ARENA_RECAPTCHA_V3_SITEKEY
            async def pump():
                nonlocal captcha_page
                try:
                    await state["events"].put({"arena_progress": "captcha"})
                    captcha_page = page
                    token = await recaptcha.refresh_recaptcha_token(force_new=True)
                    if not token:
                        raise RuntimeError("Arena: LMArenaBridge could not obtain a CAPTCHA token; no translation was submitted.")
                    request_payload = dict(payload)
                    request_payload.pop("recaptchaV2Token", None)
                    request_payload["recaptchaV3Token"] = token
                    await emit(None, {"token_received": True})
                    await emit(None, {"dispatching": True})
                    await _run_stream_with_idle_timeout(page.evaluate(r"""async ({url, method, payload}) => {
                        const response = await fetch(url, {method, credentials:'include',
                            headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
                        if (!response.ok) {
                            // Buffer errors before publishing headers: the bridge's
                            // synchronous raise_for_status needs the response body.
                            const reader = response.body?.getReader();
                            const decoder = new TextDecoder(); let errorBody = '';
                            if (reader) {
                                try {
                                    while (errorBody.length < 8192) {
                                        const {done,value} = await reader.read();
                                        if (done) break;
                                        errorBody += decoder.decode(value, {stream:true});
                                    }
                                    errorBody += decoder.decode();
                                } finally { await reader.cancel(); }
                            }
                            await arenaEmit({status:response.status,
                                headers:Object.fromEntries(response.headers), error_body:errorBody.slice(0,8192)});
                            return;
                        }
                        await arenaEmit({status:response.status, headers:Object.fromEntries(response.headers)});
                        const reader = response.body.getReader(); const decoder = new TextDecoder(); let pending='';
                        while (true) { const {done,value} = await reader.read();
                            if (value?.length) await arenaEmit({activity:true});
                            pending += decoder.decode(value || new Uint8Array(), {stream:!done});
                            let end; while ((end=pending.indexOf('\n')) >= 0) {
                                await arenaEmit({line:pending.slice(0,end).replace(/\r$/, '')}); pending=pending.slice(end+1);
                            }
                            if (done) break;
                        }
                        if (pending) await arenaEmit({line:pending});
                    }""", {"url": "https://arena.ai" + parsed.path, "method": http_method, "payload": request_payload}),
                        upstream_activity, state["stream_timeout"])
                finally:
                    headers_ready.set()
                    captcha_page = None
                    done_event.set()
            task = asyncio.create_task(pump())
            request_tasks = state["transport_tasks"]
            request_tasks.add(task)
            task.add_done_callback(request_tasks.discard)
            try:
                await asyncio.wait_for(headers_ready.wait(), 300)
                if task.done() and task.exception():
                    raise task.exception()
            except BaseException as exc:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
                await page.close()
                if isinstance(exc, asyncio.TimeoutError):
                    raise RuntimeError("Arena browser setup timed out before response headers arrived.") from exc
                raise
            class Response(transport.BrowserFetchStreamResponse):
                async def aiter_lines(self):
                    async for line in super().aiter_lines():
                        value = line.removeprefix("data:").strip()
                        if value.startswith("ad:"):
                            metadata = json.loads(value[3:])
                            if not metadata.get("finishReason"):
                                raise RuntimeError("Arena omitted the finish reason")
                            state["usage"] = metadata.get("usage") or state["usage"]
                        elif value.startswith("{"):
                            event = json.loads(value)
                            state["usage"] = event.get("usage") or state["usage"]
                            for choice in event.get("choices", []):
                                if choice.get("finish_reason"):
                                    yield line
                                    yield "ad:" + json.dumps({"finishReason": choice["finish_reason"]})
                                    break
                            else:
                                yield line
                            continue
                        yield line
                    await task
                async def aclose(self):
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task
                    with contextlib.suppress(Exception):
                        cookies = await context.cookies(["https://arena.ai/", "https://lmarena.ai/"])
                        if not state["retired"]:
                            _persist_session(slot, cookies)
                    await page.close()
                async def __aexit__(self, *args):
                    await self.aclose()
            return Response(result["status"], result["headers"], text=result.get("error_body", ""),
                            lines_queue=queue if result["status"] < 400 else None,
                            done_event=done_event, method=http_method, url=url)
        async def interactive_fetch(args, kwargs, verification):
            nonlocal context
            await state["events"].put({"arena_progress": "verification" if verification else "browser_verification"})
            async with login_lock:
                original_context = context
                lease = _regular_login_browser(playwright)
                context = await lease.__aenter__()
                try:
                    await context.add_cookies(await original_context.cookies(["https://arena.ai/", "https://lmarena.ai/"]))
                    verified = await fetch(*args, **dict(kwargs, _interactive=True, _verification=verification))
                    if not verification and verified.status_code == 403 and "captcha" in verified.text.lower():
                        await verified.aclose()
                        await state["events"].put({"arena_progress": "verification"})
                        verified = await fetch(*args, **dict(kwargs, _interactive=True, _verification=True))
                except BaseException:
                    await lease.__aexit__(None, None, None)
                    context = original_context
                    raise
                original_close = verified.aclose
                closed = False
                async def close_verified():
                    nonlocal context, closed
                    if closed:
                        return
                    closed = True
                    try:
                        await original_close()
                        await original_context.add_cookies(await context.cookies(["https://arena.ai/", "https://lmarena.ai/"]))
                    finally:
                        await lease.__aexit__(None, None, None)
                        context = original_context
                verified.aclose = close_verified
                if verified.status_code >= 400:
                    detail = f"Arena HTTP {verified.status_code}: {verified.text}"
                    if verified.status_code == 403 and "captcha" in verified.text.lower():
                        detail += " Arena rejected the refreshed CAPTCHA token."
                    await verified.aclose()
                    state["terminal_error"] = transport.BrowserFetchStreamResponse(400, {}, text=detail)
                    return state["terminal_error"]
                return verified

        async def submit_once(*args, **kwargs):
            nonlocal context
            request_tasks = state["transport_tasks"]
            transport_task = asyncio.current_task()
            request_tasks.add(transport_task)
            transport_task.add_done_callback(request_tasks.discard)
            if state["submitted"]:
                return state.get("terminal_error") or transport.BrowserFetchStreamResponse(
                    400, {}, text="Arena response was interrupted after submission; it was not replayed. Retry the request explicitly.")
            state["submitted"] = True
            for attempt in range(2):
                state["dispatched"] = False
                before = set(context.pages)
                try:
                    response = await fetch(*args, **kwargs)
                    if response.status_code not in (401, 403):
                        if response.status_code >= 400:
                            # The pinned bridge retries 429/5xx internally,
                            # delaying the actual rejection behind keepalives.
                            # Stop that loop with its non-retryable status;
                            # upstream_chunks preserves the real status/body
                            # and Retry-After from state["upstream_error"].
                            detail = f"Arena HTTP {response.status_code}: {response.text}"
                            await response.aclose()
                            state["terminal_error"] = transport.BrowserFetchStreamResponse(
                                400, {}, text=detail)
                            return state["terminal_error"]
                        return response
                    detail = f"Arena HTTP {response.status_code}: {response.text}"
                    captcha_rejected = response.status_code == 403 and "captcha" in response.text.lower()
                    await response.aclose()
                    if captcha_rejected:
                        if attempt == 0:
                            return await interactive_fetch(args, kwargs, verification=True)
                        detail += " Arena rejected a fresh CAPTCHA token. Complete any verification offered on Arena's website before retrying."
                        state["terminal_error"] = transport.BrowserFetchStreamResponse(400, {}, text=detail)
                        return state["terminal_error"]
                    safe_to_retry = True  # Explicit rejection, no accepted stream.
                except asyncio.CancelledError:
                    for page in list(context.pages):
                        if page not in before:
                            with contextlib.suppress(Exception):
                                await page.close()
                    raise
                except Exception as exc:
                    if str(exc) == "ARENA_BROWSER_CHALLENGE" and not state["dispatched"]:
                        try:
                            return await interactive_fetch(args, kwargs, verification=False)
                        except Exception as interactive_error:
                            state["terminal_error"] = transport.BrowserFetchStreamResponse(400, {}, text=str(interactive_error))
                            return state["terminal_error"]
                    if "ARENA_CAPTCHA_" in str(exc):
                        marker = re.search(r"ARENA_CAPTCHA_[A-Z_]+", str(exc)).group(0)
                        detail = ("Arena verification was cancelled; no additional translation was submitted."
                                  if "CANCELLED" in str(exc) else
                                  f"Arena CAPTCHA could not complete ({marker}); no additional translation was submitted. Check the Arena Verification window and whether verification scripts are blocked.")
                        state["terminal_error"] = transport.BrowserFetchStreamResponse(400, {}, text=detail)
                        return state["terminal_error"]
                    detail = f"Arena browser connection failed ({type(exc).__name__})."
                    safe_to_retry = not state["dispatched"]
                    for page in list(context.pages):
                        if page not in before:
                            with contextlib.suppress(Exception):
                                await page.close()
                if attempt == 0 and safe_to_retry:
                    with contextlib.suppress(Exception):
                        await context.close()
                    try:
                        context, restored = await restore_context(slot, refresh=True)
                        state["context"] = context
                        cfg.update(auth_token=restored["token"], auth_tokens=[restored["token"]],
                                   browser_cookies={c["name"]: c["value"] for c in restored["cookies"]})
                        continue
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        detail += " Automatic session reconnection failed. Use Arena Login if the session was revoked."
                # Stop upstream retry loops from masking the actual rejection.
                state["terminal_error"] = transport.BrowserFetchStreamResponse(400, {}, text=detail)
                return state["terminal_error"]
        main.fetch_lmarena_stream_via_chrome = submit_once
        main.fetch_lmarena_stream_via_camoufox = submit_once
        return state

    @app.get("/v1/models")
    async def models_route():
        accounts = list_accounts()
        if not accounts:
            return {"data": []}
        state = await get_slot(accounts[0]["slot"])
        try:
            state["models"][:] = await _ensure_catalog(state["context"])
            state["main"].STRICT_BROWSER_FETCH_MODELS = {m["publicName"] for m in state["models"]}
        finally:
            await release_slot(state)
        return {"data": [{"id": m["publicName"]} for m in state["models"]]}

    @app.post("/cancel")
    async def cancel_request(request: Request):
        request_id = str((await request.json()).get("id", ""))
        if len(cancelled_jobs) > 1024:
            cancelled_jobs.clear()
        cancelled_jobs.add(request_id)
        task = jobs.get(request_id)
        if task is not None:
            task.cancel()
        return {"cancelled": True}

    @app.post("/dispatch")
    async def approve_dispatch(request: Request):
        request_id = str((await request.json()).get("id", ""))
        if request_id in cancelled_jobs or request_id not in dispatch_acks:
            raise HTTPException(409, "Arena request is no longer waiting to send")
        dispatch_acks[request_id].set()
        return {"approved": True}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        request_id = str(body.pop("request_id", secrets.token_hex(16)))
        if request_id in cancelled_jobs:
            cancelled_jobs.discard(request_id)
            raise HTTPException(409, "Arena request cancelled")
        jobs[request_id] = asyncio.current_task()
        try:
            return await prepare_chat(request, body, request_id)
        except BaseException:
            jobs.pop(request_id, None)
            raise

    async def prepare_chat(request, body, request_id):
        nonlocal cursor
        stream_timeout = body.pop("stream_timeout", 600)
        if stream_timeout is not None:
            try:
                stream_timeout = float(stream_timeout)
                if not math.isfinite(stream_timeout) or stream_timeout <= 0:
                    raise ValueError()
            except (ValueError, TypeError):
                raise HTTPException(400, "Arena stream_timeout must be positive or null")
        slot = body.pop("account_slot", 0)
        if slot is None:
            accounts = list_accounts()
            if not accounts:
                raise HTTPException(401, "No Arena accounts. Use Arena Login.")
            slot = accounts[cursor % len(accounts)]["slot"]
            cursor += 1
        body["stream"] = True
        body.pop("conversation_id", None)
        dispatch_ack = body.pop("dispatch_ack", False)
        request._body = json.dumps(body).encode()
        request._json = body
        state = await get_slot(int(slot))
        main = state["main"]
        main.chat_sessions.clear()
        state["submitted"] = False
        state["terminal_error"] = None
        state["usage"] = None
        state["stream_timeout"] = stream_timeout
        state["upstream_error"] = None
        state["events"] = asyncio.Queue(maxsize=64)
        state["transport_tasks"] = set()
        state["dispatch_ack"] = asyncio.Event() if dispatch_ack else None
        if state["dispatch_ack"] is not None:
            dispatch_acks[request_id] = state["dispatch_ack"]
        try:
            response = await main.api_chat_completions(request, {"key": key, "rpm": 100000})
        except BaseException:
            dispatch_acks.pop(request_id, None)
            await release_slot(state)
            raise
        if not hasattr(response, "body_iterator"):
            dispatch_acks.pop(request_id, None)
            await release_slot(state)
            jobs.pop(request_id, None)
            return response
        async def upstream_chunks():
            try:
                jobs[request_id] = asyncio.current_task()
                if request_id in cancelled_jobs:
                    raise asyncio.CancelledError()
                async for chunk in response.body_iterator:
                    if isinstance(chunk, str) and chunk.startswith("data: {"):
                        event = json.loads(chunk[6:])
                        if event.get("error") and state["upstream_error"]:
                            event["error"] = state["upstream_error"]
                            chunk = "data: " + json.dumps(event) + "\n\n"
                        elif state["usage"] and any(c.get("finish_reason") for c in event.get("choices", [])):
                            event["usage"] = state["usage"]
                            chunk = "data: " + json.dumps(event) + "\n\n"
                    yield chunk
            finally:
                close_iterator = getattr(response.body_iterator, "aclose", None)
                if close_iterator is not None:
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await close_iterator()
                jobs.pop(request_id, None)
        async def chunks():
            events = state["events"]
            end = object()
            async def produce():
                upstream = upstream_chunks()
                try:
                    async for chunk in upstream:
                        await events.put(chunk)
                except asyncio.CancelledError:
                    with contextlib.suppress(asyncio.QueueFull):
                        events.put_nowait(asyncio.CancelledError())
                    raise
                except BaseException as exc:
                    await events.put(exc)
                finally:
                    await upstream.aclose()
                await events.put(end)
            producer = asyncio.create_task(produce())
            try:
                while True:
                    try:
                        item = await asyncio.wait_for(events.get(), 5)
                    except asyncio.TimeoutError:
                        if producer.done():
                            if producer.cancelled():
                                raise asyncio.CancelledError()
                            failure = producer.exception()
                            if failure:
                                raise failure
                            break
                        yield ": Arena request pending\n\n"
                        continue
                    if item is end:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    yield "data: " + json.dumps(item) + "\n\n" if isinstance(item, dict) else item
            finally:
                producer.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await producer
                dispatch_acks.pop(request_id, None)
                cancelled_jobs.discard(request_id)
                await release_slot(state)
        return StreamingResponse(chunks(), media_type="text/event-stream",
                                 headers={"X-Arena-Account-Slot": str(slot)})

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    _save("service.enc", {"url": f"http://127.0.0.1:{sock.getsockname()[1]}", "key": key})
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
    @app.post("/shutdown")
    async def shutdown():
        for task in list(jobs.values()):
            task.cancel()
        server.should_exit = True
        return {"stopping": True}
    try:
        await server.serve(sockets=[sock])
    finally:
        for pool in slots.values():
            for state in pool:
                await close_slot(state)
        await playwright.stop()


def create_login_controls(parent, get_model, set_model, log_fn=print, on_login=None, selector_enabled=None):
    """Shared Qt controls; capture the target slot before starting background work."""
    from PySide6.QtCore import Signal, Slot, QTimer, Qt
    from PySide6.QtGui import QIcon, QPixmap, QPainter, QPen, QColor
    from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QComboBox, QInputDialog, QMessageBox

    class Controls(QWidget):
        completed = Signal(object, object)
        progress = Signal(str)
        accounts_loaded = Signal(object, object)

        def __init__(self):
            super().__init__(parent)
            row = QHBoxLayout(self)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(3)
            self.accounts = QComboBox(self)
            self.accounts.setFixedWidth(54)
            self.accounts.setStyleSheet(
                "QComboBox { background-color: #2a3a4a; color: #ccc; font-weight: bold; "
                "font-size: 9pt; padding: 1px 2px 1px 4px; border: 1px solid #555; border-radius: 3px; "
                "min-width: 28px; max-width: 46px; } "
                "QComboBox:hover { background-color: #3a4a5a; color: white; border-color: #888; } "
                "QComboBox::drop-down { width: 12px; border: none; } "
                "QComboBox::down-arrow { image: none; border: none; width: 0px; } "
                "QComboBox QAbstractItemView { background-color: #2a3a4a; color: #e0e0e0; "
                "selection-background-color: #4a6a8a; border: 1px solid #555; }"
            )
            self.login_slot = 0
            self.login_button = QPushButton("Arena Login", self)
            self.login_button.setStyleSheet(
                "background-color: #10a37f; color: white; font-weight: bold; "
                "font-size: 10pt; padding: 4px 8px; border-radius: 4px;"
            )
            self.login_button.setToolTip("Log into Arena in the automatically installed internal browser")
            row.addWidget(self.login_button)
            row.addWidget(self.accounts)
            self.login_button.clicked.connect(self.login)
            self.accounts.activated.connect(self.select_account)
            self.completed.connect(self.finished)
            self.progress.connect(log_fn or print)
            self.progress.connect(self.show_progress)
            self.spinner = QTimer(self)
            self.spinner.setInterval(80)
            self.spinner.timeout.connect(self.animate)
            self.spinner_angle = 0
            self.account_snapshot = None
            self.saved_accounts = []
            self.accounts_ready = False
            self.accounts_loading = False
            self.accounts_reload_pending = False
            self.accounts_checked_at = None
            self.accounts_loaded.connect(self.receive_accounts)
            self.busy = False
            self.refresh()

        def load_accounts_async(self):
            if self.accounts_loading:
                return
            self.accounts_loading = True
            def worker():
                try:
                    result, error = list_accounts(), None
                except Exception as exc:
                    result, error = None, str(exc)
                with contextlib.suppress(RuntimeError):
                    self.accounts_loaded.emit(result, error)
            threading.Thread(target=worker, daemon=True, name="Arena account list").start()

        @Slot(object, object)
        def receive_accounts(self, accounts, error):
            self.accounts_loading = False
            self.accounts_checked_at = time.monotonic()
            if error is None:
                self.saved_accounts = accounts
                self.accounts_ready = True
            else:
                self.progress.emit("⚠️ Arena: could not load saved accounts: " + error)
                QTimer.singleShot(5000, self.refresh)
            if self.accounts_reload_pending:
                self.accounts_reload_pending = False
                self.accounts_checked_at = None
            self.refresh()

        def animate(self):
            pixmap = QPixmap(20, 20)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(QColor("white"), 2.5))
            painter.drawArc(3, 3, 14, 14, self.spinner_angle * 16, 270 * 16)
            painter.end()
            icon = QIcon()
            icon.addPixmap(pixmap, QIcon.Mode.Normal)
            icon.addPixmap(pixmap, QIcon.Mode.Disabled)
            self.login_button.setIcon(icon)
            self.spinner_angle = (self.spinner_angle - 30) % 360

        @Slot(str)
        def show_progress(self, message):
            if self.busy:
                self.login_button.setToolTip(message)

        def refresh(self, *args):
            model = str(get_model() or "")
            match = ROUTE_RE.match(model.strip())
            self.setVisible(bool(match))
            if not match:
                return
            slot, _ = parse_route(model)
            if self.accounts_checked_at is None or time.monotonic() - self.accounts_checked_at >= 5:
                self.load_accounts_async()
            saved_accounts = self.saved_accounts
            login_slot = self.login_slot if slot is None else slot
            account_suffix = f" #{login_slot}" if login_slot else ""
            selected_accounts = [a for a in saved_accounts if a["slot"] == login_slot]
            if not self.busy:
                self.login_button.setText(
                    f"✅ Arena{account_suffix}" if selected_accounts
                    else f"Arena{account_suffix} Login"
                )
                if selected_accounts:
                    identities = ", ".join(f"#{a['slot']} ({a.get('email') or 'saved account'})" for a in selected_accounts)
                    self.login_button.setToolTip(
                        ("Selected pool account: " if slot is None else "Saved Arena account: ") + identities
                        + ". Credentials are encrypted and restored automatically. Click to reconnect. "
                        "Saved login does not guarantee CAPTCHA acceptance.")
                    snapshot = (login_slot, identities)
                    if snapshot != self.account_snapshot:
                        first = selected_accounts[0]
                        summary = f"account #{login_slot}"
                        self.progress.emit(f"🔓 Arena: restored {summary} from encrypted storage — first: {first.get('email') or 'email unavailable'}.")
                    self.account_snapshot = snapshot
                else:
                    self.login_button.setToolTip("Log into Arena in the automatically installed internal browser")
                    self.account_snapshot = None
            self.accounts.blockSignals(True)
            self.accounts.clear()
            ids = sorted({0, login_slot, *[a["slot"] for a in saved_accounts]})
            for aid in ids:
                self.accounts.addItem(f"#{aid}", aid)
            self.accounts.addItem("+ N", "new")
            self.accounts.setCurrentIndex(max(0, self.accounts.findData(self.login_slot if slot is None else slot)))
            self.accounts.blockSignals(False)
            self.accounts.setVisible(match.group(1) == "0" and (selector_enabled is None or selector_enabled()))
            self.accounts.setEnabled(not self.busy and self.accounts_ready)
            self.login_button.setEnabled(not self.busy and self.accounts_ready)
            if not self.accounts_ready:
                self.login_button.setToolTip("Loading saved Arena accounts…")

        def select_account(self, index):
            selected = self.accounts.itemData(index)
            if selected == "new":
                self.start(None, False)
            else:
                self.login_slot = selected
                self.refresh()

        def login(self):
            slot, _ = parse_route(get_model())
            if slot is None:
                slot = self.login_slot
            self.start(slot, False)

        def start(self, slot, update_route):
            if self.busy:
                return
            self.busy = True
            self.accounts.setEnabled(False)
            self.login_button.setEnabled(False)
            self.login_button.setText("Signing in…")
            self.login_button.setToolTip("Preparing Arena login. First-time setup can take a few minutes.")
            self.animate()
            self.spinner.start()
            # A row/model can change while login is open. Never retarget the
            # credential write or overwrite a later model edit on completion.
            original = get_model()
            def worker():
                try:
                    if slot is None:
                        self.progress.emit("Arena: sign in with the new account in the internal browser.")
                    result = open_login(log_fn=self.progress.emit, account_id=slot)
                    self.completed.emit((result, original, update_route), None)
                except Exception as exc:
                    with contextlib.suppress(RuntimeError):
                        self.completed.emit(None, str(exc))
            threading.Thread(target=worker, daemon=True, name="Arena Login").start()

        @Slot(object, object)
        def finished(self, result, error):
            self.busy = False
            self.spinner.stop()
            self.login_button.setIcon(QIcon())
            self.login_button.setText("Arena Login")
            self.login_button.setToolTip("Log into Arena in the automatically installed internal browser")
            if error:
                self.progress.emit("Arena Login: " + error)
                self.login_button.setToolTip(error)
            else:
                account, original, update_route = result
                if get_model() == original:
                    self.login_slot = account["slot"]
                if update_route and get_model() == original:
                    _, model = parse_route(original)
                    set_model(route_for_slot(account["slot"], model))
                self.progress.emit(f"✅ Arena account #{account['slot']} ({account.get('email') or 'saved account'}) connected.")
                if account.get("catalog_cached") is False:
                    self.progress.emit("Arena login was saved, but model IDs could not be refreshed. The last successful catalog will be reused if available.")
                if on_login:
                    on_login()
            self.accounts_checked_at = None
            self.accounts_reload_pending = self.accounts_loading
            self.refresh()

    return Controls()


def install_combo_login(parent, combo, log_fn=print, on_login=None):
    """Attach the shared control to dynamically created multi-key model fields."""
    from PySide6.QtCore import QTimer
    if getattr(combo, "_autharena_controls", None) is not None or not combo.isEditable():
        return
    editor = combo.lineEdit()
    controls = create_login_controls(editor, combo.currentText, combo.setCurrentText, log_fn, on_login)
    combo._autharena_controls = controls
    margins = editor.textMargins()
    previous_width = 0
    def position(*args):
        nonlocal previous_width
        from shiboken6 import isValid
        if not isValid(combo) or not isValid(controls):
            return
        controls.refresh()
        width = controls.sizeHint().width() if ROUTE_RE.match(combo.currentText().strip()) else 0
        controls.resize(width, editor.height())
        controls.move(max(0, editor.width() - width), 0)
        controls.raise_()
        current = editor.textMargins()
        if width:
            editor.setTextMargins(current.left(), current.top(), width + 2, current.bottom())
        elif previous_width and current.right() == previous_width + 2:
            editor.setTextMargins(current.left(), current.top(), margins.right(), current.bottom())
        previous_width = width
    original = editor.resizeEvent
    def resized(event):
        original(event)
        position()
    editor.resizeEvent = resized
    combo.currentTextChanged.connect(position)
    QTimer.singleShot(0, position)


if __name__ == "__main__":
    if "--qt-browser" in sys.argv:
        sys.exit(_qt_browser_helper("--visible" in sys.argv))
    elif "--worker" in sys.argv:
        import asyncio
        startup = json.loads(sys.stdin.readline())
        asyncio.run(_serve_worker(startup["key"], startup.get("qt_helper_command")))
    else:
        import argparse
        parser = argparse.ArgumentParser(description="Manage Glossarion's Arena proxy; no arguments starts the service until Ctrl+C.")
        parser.add_argument("--status", action="store_true", help="Check health without starting the service or browser")
        parser.add_argument("--login", metavar="PREFIX", help="Sign in to an account, e.g. autharena/ for account #0, autharena1/ for #1")
        args = parser.parse_args()
        try:
            if args.status:
                print("Arena proxy is " + ("running" if check_proxy_health().get("running") else "stopped"))
            elif args.login:
                slot, _ = parse_route(args.login)
                account = open_login(account_id=slot)
                print(f"Arena account {account['slot']} connected.")
            else:
                ensure_proxy_running()
                print("Arena proxy is running. Press Ctrl+C to stop.")
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            pass
