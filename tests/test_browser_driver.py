"""The browser-driver seam of the browser-backed routes (src/browser_driver.py; U9 WebViewBridge).

* Desktop is unchanged: no driver is registered, so ``authnd_auth.get_captcha_token`` and
  ``gemini_free.send_chat_completion`` still pick their QtWebEngine helper (subprocess by
  default, inline on request) and ``cancel_stream`` only gains a no-op call.
* A registered driver runs the *same* flow: the same page scripts in the same order as the
  QtWebEngine helper (checked against a scripted fake Qt page and a fake driver page that
  model the same website).
* Without helper processes (Glossarion Mobile) a route never reaches ``subprocess.Popen``: it
  uses the driver, or fails with a clear ``BrowserUnavailable`` message.
* ``cancel_stream`` cancels the route's pages (pending calls end with "stream cancelled").
* The NIM / AuthND token helper and Gemini Free browser chunking settings are available on
  mobile, except the helper-subprocess limit.

Python 3.10 compatible; no Qt or Flet needed (PySide6 is faked where the Qt helper runs).
Run:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_browser_driver.py
"""

import ast
import json
import os
import re
import subprocess
import sys
import threading
import time
import types
import uuid
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:  # CI has no tests/conftest.py
    sys.path.insert(0, str(SRC_DIR))

import authnd_auth as authnd  # noqa: E402
import browser_driver  # noqa: E402
import gemini_free  # noqa: E402

MOBILE_ENV = ("GLOSSARION_MOBILE", "GLOSSARION_NO_PROCESSES", "FLET_PLATFORM")
MODE_ENV = ("AUTHND_TOKEN_MODE", "GEMINI_FREE_MODE")
PAGE_URL = "https://build.nvidia.com/z-ai/glm-5.1"


@pytest.fixture
def desktop(monkeypatch, tmp_path):
    """A desktop process with no driver registered and HOME pointing at a scratch dir."""
    for name in MOBILE_ENV + MODE_ENV + ("TRANSLATION_CANCELLED",):
        monkeypatch.delenv(name, raising=False)
    if sys.platform in ("android", "ios", "emscripten", "wasi"):  # pragma: no cover - host tests only
        monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("GEMINI_FREE_WAIT_AFTER_LOAD_MS", "0")
    monkeypatch.setenv("GEMINI_FREE_STABLE_MS", "0")
    monkeypatch.setattr(authnd, "CAPTCHA_DOCUMENT_STABLE_SECONDS", 0.0)
    browser_driver.unregister_driver()
    authnd._cancel_event.clear()
    authnd.reset_cancel()
    gemini_free._cancel_event.clear()
    yield monkeypatch
    browser_driver.unregister_driver()
    authnd.reset_cancel()
    gemini_free.reset_cancel()


@pytest.fixture
def mobile(desktop):
    desktop.setenv("GLOSSARION_MOBILE", "1")
    desktop.setenv("GLOSSARION_NO_PROCESSES", "1")
    return desktop


@pytest.fixture
def no_spawn(monkeypatch):
    """Fail the test on any helper process start."""
    calls = []

    def popen(*args, **kwargs):
        calls.append(args)
        raise AssertionError("a helper subprocess was started")

    monkeypatch.setattr(subprocess, "Popen", popen)
    return calls


# ============================================================================ the websites
class NvidiaBuild:
    """build.nvidia.com as both fakes see it: an untitled document first, then the titled
    one; ``reloads`` same-URL document replacements after the injection."""

    def __init__(self, reloads=0, token="tok-xyz"):
        self.reloads = reloads
        self.token = token
        self.marker = ""

    def answer(self, script, renavigate):
        if "const marker =" in script and "return marker" in script:
            self.marker = json.loads(re.search(r"const marker = (\".*?\");", script).group(1))
            return self.marker
        if "result: window.__authndResult" in script:
            if self.reloads:
                self.reloads -= 1
                self.marker = ""
                renavigate()
                return json.dumps({"marker": "", "result": None, "readyState": "loading"})
            return json.dumps({"marker": self.marker, "result": {
                "marker": self.marker, "pending": False, "step": "complete", "token": self.token,
                "error": None}, "readyState": "complete"})
        return None


class AiMode:
    """Google Search AI Mode: a textarea, a Send button that is ready on the second try and
    an answer on the first snapshot after the click."""

    def __init__(self, answer_text="Bonjour"):
        self.answer_text = answer_text
        self.clicks = 0

    def answer(self, script, renavigate):
        if "AI Mode textarea not found" in script:
            return json.dumps({"ok": True, "textareaCount": 1, "valueLength": 5, "placeholder": "Ask anything"})
        if "AI Mode send button not ready" in script:
            self.clicks += 1
            if self.clicks < 2:
                return json.dumps({"ok": False, "error": "AI Mode send button not ready", "buttonCount": 1,
                                   "labels": ["Send"]})
            return json.dumps({"ok": True, "label": "Send", "buttonCount": 1})
        if "answerMarkdown" in script:
            prompt = json.loads(re.search(r"const promptText = (\".*?\");", script).group(1))
            text = f"You said:\n{prompt}\n{self.answer_text}\nAI can make mistakes"
            return json.dumps({"url": "https://www.google.com/search?udm=50", "title": "AI Mode",
                               "ready": "complete", "text": text, "busy": False,
                               "answerText": self.answer_text,
                               "answerHtml": f"<p>{self.answer_text}</p>", "htmlLength": 999})
        return None


# ============================================================================ fake Qt helper
def install_fake_qt(monkeypatch, site):
    """PySide6 stand-ins that serve ``site`` (runJavaScript answers synchronously)."""
    pages = []

    class QUrl:
        def __init__(self, value=""):
            self.value = str(value)

        def toString(self):
            return self.value

    class QSize:
        def __init__(self, width, height):
            self.size = (width, height)

    class QEventLoop:
        def quit(self):
            pass

        def exec(self):
            for page in pages:
                page.advance()

    class QTimer:
        @staticmethod
        def singleShot(_ms, _callback):
            pass

    class QApplication:
        _instance = None

        def __init__(self, _args):
            type(self)._instance = self

        @classmethod
        def instance(cls):
            return cls._instance

    class QWebEngineProfile:
        def __init__(self, *_args):
            pass

        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    class Signal:
        def __init__(self):
            self.callbacks = []

        def connect(self, callback):
            self.callbacks.append(callback)

        def emit(self, *args):
            for callback in list(self.callbacks):
                callback(*args)

    class QWebEnginePage:
        def __init__(self, *_args):
            self.loadStarted, self.loadFinished, self.loadingChanged = Signal(), Signal(), Signal()
            self._url, self._title, self.pending = QUrl(""), "", False
            self.scripts = []
            pages.append(self)

        def url(self):
            return self._url

        def title(self):
            return self._title

        def setViewportSize(self, _size):
            pass

        def load(self, url):
            self._url, self._title = QUrl(url.toString()), ""
            self.loadStarted.emit()
            self.loadFinished.emit(True)
            self.pending = True

        def advance(self):
            if self.pending:
                self.pending = False
                self.renavigate()

        def renavigate(self):
            self.loadStarted.emit()
            self._title = "Site"
            self.loadFinished.emit(True)

        def runJavaScript(self, script, callback):
            self.scripts.append(script)
            callback(site.answer(script, self.renavigate))

        def deleteLater(self):
            pass

    core = types.ModuleType("PySide6.QtCore")
    core.QEventLoop, core.QTimer, core.QUrl, core.QSize = QEventLoop, QTimer, QUrl, QSize
    web = types.ModuleType("PySide6.QtWebEngineCore")
    web.QWebEnginePage, web.QWebEngineProfile = QWebEnginePage, QWebEngineProfile
    widgets = types.ModuleType("PySide6.QtWidgets")
    widgets.QApplication = QApplication
    for name, module in (("PySide6", types.ModuleType("PySide6")), ("PySide6.QtCore", core),
                         ("PySide6.QtWebEngineCore", web), ("PySide6.QtWidgets", widgets)):
        monkeypatch.setitem(sys.modules, name, module)
    shutdown = types.ModuleType("shutdown_utils")
    shutdown.cleanup_generated_browser_profile_dir = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "shutdown_utils", shutdown)
    # The Qt helpers write these into os.environ; let monkeypatch restore them.
    monkeypatch.setenv("QTWEBENGINE_CHROMIUM_FLAGS", "")
    monkeypatch.setenv("QTWEBENGINE_DISABLE_SANDBOX", "1")
    return pages


# ============================================================================ fake driver
class FakePage:
    """A ``browser_driver.BrowserPage`` serving ``site`` (the website the fake Qt serves)."""

    def __init__(self, driver, site, owner, viewport, cancel_check):
        self.driver, self.site, self.owner = driver, site, owner
        self.viewport, self.cancel_check = viewport, cancel_check
        self.load_state = browser_driver.new_load_state()
        self.scripts, self.loads = [], []
        self._url, self._title, self.pending = "", "", False
        self.closed = False
        self.cancelled = threading.Event()

    def _started(self):
        state = self.load_state
        state.update(generation=state["generation"] + 1, finished_generation=-1, finished_at=0.0, ok=False)

    def _finished(self):
        state = self.load_state
        state.update(finished_generation=state["generation"], finished_at=time.monotonic(), ok=True)

    def renavigate(self):
        self._started()
        self._title = "Site"
        self._finished()

    def load(self, url):
        self.loads.append(url)
        self._url, self._title = url, ""
        self._started()
        self._finished()
        self.pending = True

    def url(self):
        return self._url

    def title(self):
        return self._title

    def _check(self):
        if self.cancelled.is_set() or (self.cancel_check is not None and self.cancel_check()):
            raise browser_driver.BrowserCancelled()

    def run_js(self, script, js_timeout_ms=15000):
        self._check()
        self.scripts.append(script)
        if self.driver.block_js:
            while not self.cancelled.wait(0.02):
                self._check()
            self._check()
        return self.site.answer(script, self.renavigate)

    def wait(self, ms=100):
        self._check()
        if self.pending:
            self.pending = False
            self.renavigate()

    def close(self):
        self.closed = True


class FakeDriver:
    name = "fake"

    def __init__(self, site_factory, block_js=False):
        self.site_factory = site_factory
        self.block_js = block_js
        self.pages = []
        self.cancels = []

    def open_page(self, *, owner, user_agent=None, viewport=None, cancel_check=None):
        page = FakePage(self, self.site_factory(), owner, viewport, cancel_check)
        page.user_agent = user_agent
        self.pages.append(page)
        return page

    def cancel_pages(self, owner=None, reason=browser_driver.CANCELLED):
        self.cancels.append(owner)
        for page in self.pages:
            if owner is None or page.owner == owner:
                page.cancelled.set()


def _normalize_authnd(scripts):
    out = []
    for script in scripts:
        script = re.sub(r"[0-9a-f]{32}", "<id>", script)
        out.append(re.sub(r"const captchaWaitTimeoutMs = \d+;", "const captchaWaitTimeoutMs = <ms>;", script))
    return out


# ============================================================================ the registry
def test_use_driver_policy(desktop):
    assert browser_driver.get_driver() is None
    assert browser_driver.use_driver("AUTHND_TOKEN_MODE") is False          # desktop default
    desktop.setenv("AUTHND_TOKEN_MODE", "inline")
    assert browser_driver.use_driver("AUTHND_TOKEN_MODE") is False
    desktop.setenv("AUTHND_TOKEN_MODE", "driver")
    assert browser_driver.use_driver("AUTHND_TOKEN_MODE") is True           # asked for explicitly
    desktop.delenv("AUTHND_TOKEN_MODE")
    driver = FakeDriver(NvidiaBuild)
    assert browser_driver.register_driver(driver) is None
    assert browser_driver.get_driver() is driver
    assert browser_driver.use_driver("AUTHND_TOKEN_MODE") is True           # registered + unset
    desktop.setenv("AUTHND_TOKEN_MODE", "subprocess")
    assert browser_driver.use_driver("AUTHND_TOKEN_MODE") is False          # explicit helper wins
    assert browser_driver.unregister_driver(FakeDriver(NvidiaBuild)) is False  # not the registered one
    assert browser_driver.unregister_driver(driver) is True
    assert browser_driver.get_driver() is None
    desktop.setenv("GLOSSARION_MOBILE", "1")
    assert browser_driver.use_driver("AUTHND_TOKEN_MODE") is True           # no helper processes
    with pytest.raises(browser_driver.BrowserUnavailable, match="hidden in-app"):
        browser_driver.require_driver("AuthND")


def test_cancel_pages_is_a_safe_no_op_without_a_driver(desktop):
    browser_driver.cancel_pages("authnd")                                    # no driver: nothing
    driver = FakeDriver(NvidiaBuild)
    browser_driver.register_driver(driver)
    browser_driver.cancel_pages("authnd")
    assert driver.cancels == ["authnd"]

    class Broken(FakeDriver):
        def cancel_pages(self, owner=None, reason=""):
            raise RuntimeError("boom")

    browser_driver.register_driver(Broken(NvidiaBuild))
    browser_driver.cancel_pages()                                             # never raises


def test_wait_until_loaded(desktop):
    page = FakeDriver(NvidiaBuild).open_page(owner="x")
    ticks = []
    page.wait = lambda ms=100: ticks.append(ms)
    assert browser_driver.wait_until_loaded(page, 0.05, poll_ms=5) is False   # nothing loaded
    page.load_state.update(generation=2, finished_generation=1, ok=True)      # an older load finished
    assert browser_driver.wait_until_loaded(page, 0.05, poll_ms=5) is False
    page.load_state.update(finished_generation=2)
    assert browser_driver.wait_until_loaded(page, None) is True
    page.load_state.update(ok=False)
    assert browser_driver.wait_until_loaded(page, None) is False
    assert ticks and set(ticks) == {5}


# ============================================================================ AuthND
def test_authnd_desktop_keeps_the_qtwebengine_helper(desktop):
    calls = []
    desktop.setattr(authnd, "_mint_captcha_token_subprocess",
                    lambda url, timeout, cancel_check=None: calls.append(("subprocess", url, timeout)) or "sub")
    desktop.setattr(authnd, "_mint_captcha_token_qt", lambda url, timeout: calls.append(("qt", url, timeout)) or "qt")
    desktop.setattr(authnd, "_mint_captcha_token_driver", lambda *a, **k: pytest.fail("driver used on desktop"))
    assert authnd.get_captcha_token(PAGE_URL, 60) == "sub"
    desktop.setenv("AUTHND_TOKEN_MODE", "inline")
    assert authnd.get_captcha_token(PAGE_URL, 60) == "qt"
    assert calls == [("subprocess", PAGE_URL, 60), ("qt", PAGE_URL, 60)]
    authnd.cancel_stream()                                                    # no driver: unchanged
    assert authnd._cancel_event.is_set()


def test_authnd_driver_runs_the_qt_helper_flow(desktop):
    """Same scripts, same order, same token as the QtWebEngine helper, including the
    re-injection after NVIDIA replaced the document."""
    qt_pages = install_fake_qt(desktop, NvidiaBuild(reloads=1))
    qt_token = authnd._mint_captcha_token_qt(PAGE_URL, 30)
    driver = FakeDriver(lambda: NvidiaBuild(reloads=1))
    browser_driver.register_driver(driver)
    token = authnd.get_captcha_token(PAGE_URL, 30)
    assert token == qt_token == "tok-xyz"
    (page,) = driver.pages
    assert page.owner == authnd.BROWSER_OWNER and page.user_agent == authnd.USER_AGENT
    assert page.loads == [PAGE_URL] and page.closed
    assert _normalize_authnd(page.scripts) == _normalize_authnd(qt_pages[0].scripts)
    markers = [re.search(r"const marker = \"(.*?)\";", s).group(1) for s in page.scripts if "const marker =" in s]
    assert len(markers) == 2 and markers[1].endswith(":2:3")                 # attempt 2 after the reload
    assert page.scripts[1] == authnd.CAPTCHA_STATE_SCRIPT


def test_authnd_on_mobile_never_spawns(mobile, no_spawn):
    with pytest.raises(browser_driver.BrowserUnavailable, match="AuthND needs an embedded browser"):
        authnd.get_captcha_token(PAGE_URL, 30)
    with pytest.raises(browser_driver.BrowserUnavailable):
        authnd._mint_captcha_token_subprocess(PAGE_URL, 30)
    mobile.setenv("AUTHND_TOKEN_MODE", "subprocess")                          # the env cannot force a spawn
    driver = FakeDriver(NvidiaBuild)
    browser_driver.register_driver(driver)
    assert authnd.get_captcha_token(PAGE_URL, 30) == "tok-xyz"
    assert no_spawn == [] and len(driver.pages) == 1


def test_authnd_cancel_stream_cancels_the_pages(mobile):
    driver = FakeDriver(NvidiaBuild, block_js=True)
    browser_driver.register_driver(driver)
    outcome = {}

    def mint():
        try:
            outcome["token"] = authnd.get_captcha_token(PAGE_URL, 120)
        except Exception as exc:  # noqa: BLE001 - recorded for the assertion
            outcome["error"] = exc

    worker = threading.Thread(target=mint, daemon=True)
    worker.start()
    deadline = time.monotonic() + 5
    while not (driver.pages and driver.pages[0].scripts) and time.monotonic() < deadline:
        time.sleep(0.01)
    started = time.monotonic()
    authnd.cancel_stream()
    worker.join(5)
    assert not worker.is_alive() and time.monotonic() - started < 2
    assert "stream cancelled" in str(outcome.get("error"))
    assert driver.cancels == [authnd.BROWSER_OWNER] and driver.pages[0].closed


def test_authnd_request_cancel_check_reaches_the_page(mobile):
    driver = FakeDriver(NvidiaBuild, block_js=True)
    browser_driver.register_driver(driver)
    flag = threading.Event()
    threading.Timer(0.2, flag.set).start()
    with pytest.raises(RuntimeError, match="stream cancelled"):
        authnd.get_captcha_token(PAGE_URL, 120, cancel_check=flag.is_set)


# ============================================================================ Gemini Free
def test_gemini_driver_page_runs_the_qt_helper_flow(desktop):
    prompt = "User:\nTranslate: bonjour"
    qt_pages = install_fake_qt(desktop, AiMode())
    qt_result = gemini_free.load_ai_mode_prompt_text(prompt, timeout=30)
    driver = FakeDriver(AiMode)
    browser_driver.register_driver(driver)
    result = gemini_free.load_ai_mode_prompt_text(prompt, timeout=30)
    assert result == qt_result
    assert result["submit_mode"] == "ui" and result["submit_state"]["click"]["ok"] is True
    (page,) = driver.pages
    assert page.scripts == qt_pages[0].scripts
    assert page.scripts[0] == gemini_free._ai_mode_set_prompt_script(prompt)
    assert page.scripts[1] == page.scripts[2] == gemini_free.AI_MODE_CLICK_SEND_SCRIPT
    assert page.loads == [gemini_free._build_search_base_url()] and page.closed
    assert page.owner == gemini_free.BROWSER_OWNER
    assert page.viewport == gemini_free._default_viewport_size() == (1280, 900)


def test_gemini_on_mobile_uses_driver_pages_and_never_spawns(mobile, no_spawn):
    logs = []
    with pytest.raises(browser_driver.BrowserUnavailable, match="Gemini Free needs an embedded browser"):
        gemini_free.send_chat_completion(messages=[{"role": "user", "content": "hi"}], model="search/gemini",
                                         log_fn=logs.append)
    driver = FakeDriver(lambda: AiMode("Salut"))
    browser_driver.register_driver(driver)
    result = gemini_free.send_chat_completion(messages=[{"role": "user", "content": "hi"}],
                                              model="search/gemini", log_fn=logs.append)
    assert result["content"] == "Salut" and result["finish_reason"] == "stop"
    assert no_spawn == []
    assert [page.owner for page in driver.pages] == [gemini_free.BROWSER_OWNER] and driver.pages[0].closed
    assert any("in-app browser" in line for line in logs)
    assert not any("Qt WebEngine helper subprocess" in line for line in logs)


def test_gemini_subchunks_run_on_driver_pages(mobile, no_spawn):
    """The helper orchestration (adaptive sub-chunks) on mobile: one page per sub-chunk."""
    mobile.setenv("GEMINI_FREE_SUBCHUNK_START_DELAY", "0")
    driver = FakeDriver(lambda: AiMode("part"))
    browser_driver.register_driver(driver)
    chunks = [[{"role": "user", "content": f"chunk {index}"}] for index in range(3)]
    result = gemini_free._run_search_subprocess_sequential(
        source_messages=[{"role": "user", "content": "source"}], chunks=chunks,
        split_metadata={"target_prompt_chars": 1000}, model="search/gemini", timeout=30, max_tokens=None)
    assert result["content"] == "part\npart\npart"
    assert len(driver.pages) == 3 and all(page.closed for page in driver.pages)
    assert no_spawn == []


def test_gemini_cancel_stream_cancels_the_pages(mobile):
    driver = FakeDriver(AiMode, block_js=True)
    browser_driver.register_driver(driver)
    outcome = {}

    def ask():
        try:
            outcome["result"] = gemini_free.send_chat_completion(
                messages=[{"role": "user", "content": "hi"}], model="search/gemini")
        except Exception as exc:  # noqa: BLE001 - recorded for the assertion
            outcome["error"] = exc

    worker = threading.Thread(target=ask, daemon=True)
    worker.start()
    deadline = time.monotonic() + 5
    while not (driver.pages and driver.pages[0].scripts) and time.monotonic() < deadline:
        time.sleep(0.01)
    gemini_free.cancel_stream()
    worker.join(5)
    assert not worker.is_alive()
    assert "stream cancelled" in str(outcome.get("error"))
    assert driver.cancels == [gemini_free.BROWSER_OWNER]


def test_gemini_desktop_default_still_spawns_the_helper(desktop):
    spawned = []

    class FakeProc:
        returncode = 0

        def __init__(self, cmd, **kwargs):
            spawned.append(cmd)

        def communicate(self, timeout=None):
            return json.dumps({"content": "from helper", "finish_reason": "stop"}), ""

        def poll(self):
            return 0

    desktop.setattr(gemini_free.subprocess, "Popen", FakeProc)
    desktop.setattr(gemini_free, "_run_search_driver_once", lambda **k: pytest.fail("driver used on desktop"))
    result = gemini_free.send_chat_completion(messages=[{"role": "user", "content": "hi"}], model="search/gemini")
    assert result["content"] == "from helper"
    assert len(spawned) == 1
    assert "--search-helper" in spawned[0] or "--gemini-free-search" in spawned[0]
    gemini_free.cancel_stream()                                               # no driver: unchanged


# ============================================================================ settings / hygiene
def test_browser_route_settings_are_available_on_mobile():
    import settings_schema as ss

    keys = [key for key in ss.keys() if key.startswith(("authnd_", "gemini_free_"))]
    assert "authnd_token_concurrency" in keys and "gemini_free_adaptive_split" in keys
    for key in keys:
        available, reason = ss.is_available(key, "mobile")
        if key == "authnd_token_subprocess_concurrency":
            assert available is False and "subprocess" in reason
        else:
            assert (available, reason) == (True, ""), key
        assert ss.is_available(key, "desktop") == (True, "")


def test_browser_driver_is_gui_free_and_py310():
    for name in ("browser_driver.py", "authnd_auth.py", "gemini_free.py"):
        source = (SRC_DIR / name).read_text(encoding="utf-8")
        ast.parse(source, feature_version=(3, 10))
    code = (
        "import sys\n"
        "for name in ('PySide6', 'shiboken6', 'tkinter', 'flet', 'translator_gui'):\n"
        "    sys.modules[name] = None\n"
        "import browser_driver, authnd_auth, gemini_free\n"
        "assert browser_driver.get_driver() is None\n"
        "print('ok')\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, (str(SRC_DIR), os.environ.get("PYTHONPATH")))))
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120,
                          cwd=str(SRC_DIR))
    assert proc.returncode == 0 and proc.stdout.strip().endswith("ok"), proc.stderr[-2000:]
    data = (SRC_DIR / "browser_driver.py").read_bytes()
    assert data.count(b"\r\n") in (0, data.count(b"\n"))                      # uniform line endings
