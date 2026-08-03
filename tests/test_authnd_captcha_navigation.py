import json
import re
import sys
import types
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import authnd_auth as authnd


@pytest.mark.parametrize(
    ("selected", "expected"),
    [
        ("none", "none"),
        ("low", "low"),
        ("medium", "high"),
        ("high", "high"),
        ("xhigh", "max"),
    ],
)
def test_deepseek_v4_reasoning_effort_mapping(selected, expected):
    assert authnd._deepseek_v4_reasoning_effort(selected) == expected


def test_deepseek_v4_none_is_sent_as_reasoning_effort(monkeypatch):
    monkeypatch.setenv("ENABLE_GPT_THINKING", "1")
    monkeypatch.setenv("GPT_EFFORT", "none")
    monkeypatch.delenv("AUTHND_ENABLE_THINKING", raising=False)
    monkeypatch.delenv("AUTHND_REASONING_EFFORT", raising=False)
    payload = {}

    authnd._apply_reasoning_payload(payload, "deepseek-ai/deepseek-v4-flash")

    assert payload["reasoning_effort"] == "none"
    assert payload["chat_template_kwargs"]["enable_thinking"] is True


class _Signal:
    def __init__(self):
        self._callbacks = []

    def connect(self, callback):
        self._callbacks.append(callback)

    def emit(self, *args):
        for callback in list(self._callbacks):
            callback(*args)


def _install_fake_qt(monkeypatch, tmp_path, *, reloads_after_injection):
    pages = []

    class FakeQUrl:
        def __init__(self, value=""):
            self._value = str(value)

        def toString(self):
            return self._value

    class FakeEventLoop:
        def quit(self):
            pass

        def exec(self):
            for page in list(pages):
                page.advance_pending_navigation()

    class FakeTimer:
        @staticmethod
        def singleShot(_delay, _callback):
            pass

    class FakeApplication:
        _instance = None

        def __init__(self, _args):
            type(self)._instance = self

        @classmethod
        def instance(cls):
            return cls._instance

    class FakeProfile:
        def __init__(self, _name, _app):
            pass

        def setHttpUserAgent(self, _value):
            pass

        def setPersistentStoragePath(self, _value):
            pass

        def setCachePath(self, _value):
            pass

        def deleteLater(self):
            pass

    class FakePage:
        def __init__(self, _profile, _app):
            self.loadStarted = _Signal()
            self.loadFinished = _Signal()
            self.loadingChanged = _Signal()
            self._url = FakeQUrl("about:blank")
            self._title = ""
            self._pending_titled_reload = False
            self._reloads_remaining = reloads_after_injection
            self._marker = ""
            self.markers = []
            self.injection_scripts = []
            pages.append(self)

        def url(self):
            return self._url

        def title(self):
            return self._title

        def load(self, url):
            # NVIDIA first completes an untitled document at the requested URL,
            # then replaces it with a second, titled document at the same URL.
            self._url = FakeQUrl(url.toString())
            self._title = ""
            self.loadStarted.emit()
            self.loadFinished.emit(True)
            self._pending_titled_reload = True

        def advance_pending_navigation(self):
            if not self._pending_titled_reload:
                return
            self._pending_titled_reload = False
            self.loadStarted.emit()
            self._title = "NVIDIA NIM"
            self.loadFinished.emit(True)

        def _same_url_reload(self):
            self._marker = ""
            self.loadStarted.emit()
            self._title = "NVIDIA NIM"
            self.loadFinished.emit(True)

        def runJavaScript(self, script, callback):
            if "const marker =" in script and "return marker" in script:
                marker_match = re.search(r"const marker = (\".*?\");", script)
                assert marker_match, "injection marker was not embedded in the script"
                self._marker = json.loads(marker_match.group(1))
                self.markers.append(self._marker)
                self.injection_scripts.append(script)
                callback(self._marker)
                return

            if "result: window.__authndResult" in script:
                if self._reloads_remaining:
                    self._reloads_remaining -= 1
                    self._same_url_reload()
                    callback(json.dumps({"marker": "", "result": None, "readyState": "loading"}))
                    return
                callback(
                    json.dumps(
                        {
                            "marker": self._marker,
                            "result": {
                                "marker": self._marker,
                                "pending": False,
                                "step": "complete",
                                "token": "secret-captcha-token",
                                "error": None,
                            },
                            "readyState": "complete",
                        }
                    )
                )
                return

            callback(None)

        def deleteLater(self):
            pass

    pyside_module = types.ModuleType("PySide6")
    core_module = types.ModuleType("PySide6.QtCore")
    core_module.QEventLoop = FakeEventLoop
    core_module.QTimer = FakeTimer
    core_module.QUrl = FakeQUrl
    webengine_module = types.ModuleType("PySide6.QtWebEngineCore")
    webengine_module.QWebEnginePage = FakePage
    webengine_module.QWebEngineProfile = FakeProfile
    widgets_module = types.ModuleType("PySide6.QtWidgets")
    widgets_module.QApplication = FakeApplication
    monkeypatch.setitem(sys.modules, "PySide6", pyside_module)
    monkeypatch.setitem(sys.modules, "PySide6.QtCore", core_module)
    monkeypatch.setitem(sys.modules, "PySide6.QtWebEngineCore", webengine_module)
    monkeypatch.setitem(sys.modules, "PySide6.QtWidgets", widgets_module)

    shutdown_module = types.ModuleType("shutdown_utils")
    shutdown_module.cleanup_generated_browser_profile_dir = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "shutdown_utils", shutdown_module)
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("QTWEBENGINE_CHROMIUM_FLAGS", "")
    monkeypatch.setenv("QTWEBENGINE_DISABLE_SANDBOX", "1")
    monkeypatch.setattr(authnd, "CAPTCHA_DOCUMENT_STABLE_SECONDS", 0.0)
    authnd._cancel_event.clear()
    return pages


def test_qt_captcha_reinjects_after_same_url_document_replacement(
    monkeypatch, tmp_path, capsys
):
    pages = _install_fake_qt(
        monkeypatch,
        tmp_path,
        reloads_after_injection=1,
    )
    monkeypatch.setenv("AUTHND_DEBUG", "1")

    token = authnd._mint_captcha_token_qt(
        "https://build.nvidia.com/deepseek-ai/deepseek-v4-flash?private=do-not-log",
        30,
    )

    assert token == "secret-captcha-token"
    assert len(pages) == 1
    assert len(pages[0].markers) == 2
    assert pages[0].markers[0].endswith(":1:2")
    assert pages[0].markers[1].endswith(":2:3")
    assert all("render=explicit&onload=" in script for script in pages[0].injection_scripts)

    diagnostics = capsys.readouterr().err
    assert "injection invalidated by navigation" in diagnostics
    assert "token_length=20" in diagnostics
    assert "do-not-log" not in diagnostics
    assert "secret-captcha-token" not in diagnostics


def test_qt_captcha_caps_destroyed_document_reinjection(monkeypatch, tmp_path):
    pages = _install_fake_qt(
        monkeypatch,
        tmp_path,
        reloads_after_injection=10,
    )

    with pytest.raises(RuntimeError, match="invalidated after 3 attempts"):
        authnd._mint_captcha_token_qt(
            "https://build.nvidia.com/deepseek-ai/deepseek-v4-flash",
            30,
        )

    assert len(pages) == 1
    assert len(pages[0].markers) == 3
