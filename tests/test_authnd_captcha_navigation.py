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


def _catalog_endpoint(
    name,
    publisher,
    *,
    values=(),
    unresolved_values=(),
    available="true",
):
    return {
        "resourceType": "ENDPOINT",
        "resourceId": f"{authnd.DEFAULT_ORG_ID}/{name}",
        "orgName": authnd.DEFAULT_ORG_ID,
        "name": name.replace(".", "_"),
        "displayName": name,
        "isPublic": True,
        "guestAccess": True,
        "labels": [
            {
                "key": "general",
                "values": list(values),
                "unresolvedValues": list(unresolved_values),
            },
            {"key": "publisher", "values": [publisher]},
        ],
        "attributes": [{"key": "AVAILABLE", "value": available}],
    }


class _CatalogResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_fetch_available_models_uses_paginated_build_free_chat_catalog(monkeypatch):
    monkeypatch.setattr(authnd, "CATALOG_PAGE_SIZE", 2)
    requested_pages = []
    deepseek = _catalog_endpoint(
        "deepseek-v4-flash-0731",
        "deepseek-ai",
        values=("chat", "Free Endpoint"),
        # This flag is inconsistent with real hidden-route usability and must
        # not override the explicit free-chat labels.
        available="false",
    )
    embedding = _catalog_endpoint(
        "bge-m3",
        "baai",
        values=("Embeddings", "Free Endpoint"),
    )
    glm = _catalog_endpoint(
        "glm-5.2",
        "z-ai",
        unresolved_values=("playgroundtype_chat", "nim_type_preview"),
    )
    image_model = _catalog_endpoint(
        "image-only",
        "example",
        values=("Image Generation", "Free Endpoint"),
    )
    pages = {
        0: {
            "resultTotal": 4,
            "results": [
                {"resources": [deepseek, embedding]},
                # NGC can repeat records in multiple result groups.
                {"resources": [deepseek]},
            ],
        },
        1: {
            "resultTotal": 4,
            "results": [{"resources": [glm, image_model]}],
        },
    }

    def fake_get(url, *, params, headers, timeout):
        assert url == authnd.CATALOG_SEARCH_URL
        assert headers["Accept"] == "application/json"
        assert timeout == 7
        query = json.loads(params["q"])
        requested_pages.append(query["page"])
        assert query["pageSize"] == 2
        assert query["filters"] == [
            {"field": "orgName", "value": authnd.DEFAULT_ORG_ID}
        ]
        return _CatalogResponse(pages[query["page"]])

    monkeypatch.setattr(authnd.requests, "get", fake_get)

    assert authnd.fetch_available_models(timeout=7) == [
        "deepseek-ai/deepseek-v4-flash-0731",
        "z-ai/glm-5.2",
    ]
    assert requested_pages == [0, 1]


def test_fetch_available_models_rejects_incomplete_pagination(monkeypatch):
    monkeypatch.setattr(authnd, "CATALOG_PAGE_SIZE", 1)
    first_model = _catalog_endpoint(
        "deepseek-v4-flash-0731",
        "deepseek-ai",
        values=("chat", "Free Endpoint"),
    )

    def fake_get(_url, *, params, **_kwargs):
        page = json.loads(params["q"])["page"]
        if page:
            raise RuntimeError("second catalog page failed")
        return _CatalogResponse({
            "resultTotal": 2,
            "results": [{"resources": [first_model]}],
        })

    monkeypatch.setattr(authnd.requests, "get", fake_get)

    with pytest.raises(RuntimeError, match="second catalog page failed"):
        authnd.fetch_available_models(timeout=3)


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"resultTotal": 1, "results": "not-a-list"}, "malformed endpoint catalog"),
        ({"resultTotal": 0, "results": []}, "no compatible free chat model IDs"),
    ],
)
def test_fetch_available_models_rejects_malformed_or_empty_catalogs(
    monkeypatch, payload, error
):
    monkeypatch.setattr(
        authnd.requests,
        "get",
        lambda *_args, **_kwargs: _CatalogResponse(payload),
    )

    with pytest.raises(ValueError, match=error):
        authnd.fetch_available_models()


def test_fetch_available_models_rejects_chat_endpoint_without_publisher(monkeypatch):
    malformed = _catalog_endpoint(
        "missing-publisher",
        "",
        values=("chat", "Free Endpoint"),
    )
    payload = {"resultTotal": 1, "results": [{"resources": [malformed]}]}
    monkeypatch.setattr(
        authnd.requests,
        "get",
        lambda *_args, **_kwargs: _CatalogResponse(payload),
    )

    with pytest.raises(ValueError, match="without a usable publisher/model ID"):
        authnd.fetch_available_models()


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
    wait_match = re.search(
        r"const captchaWaitTimeoutMs = (\d+);",
        pages[0].injection_scripts[0],
    )
    assert wait_match
    assert 20_000 < int(wait_match.group(1)) <= 30_000
    assert "timeoutMs = captchaWaitTimeoutMs" in pages[0].injection_scripts[0]

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


def test_send_chat_completion_retries_token_helper_failure(monkeypatch):
    authnd._cancel_event.clear()
    token_calls = []
    logs = []

    def fake_get_token(page_url, timeout, log_fn=None):
        token_calls.append((page_url, timeout, log_fn))
        if len(token_calls) == 1:
            raise RuntimeError("AuthND hCaptcha failed: timeout waiting for hcaptcha")
        return "fresh-token"

    def fake_post_prediction(**kwargs):
        assert kwargs["captcha_token"] == "fresh-token"
        return {"content": "ok", "finish_reason": "stop"}

    monkeypatch.setattr(authnd, "_get_captcha_token_for_request", fake_get_token)
    monkeypatch.setattr(authnd, "_post_prediction", fake_post_prediction)
    monkeypatch.setenv("AUTHND_TOKEN_TIMEOUT", "75")

    result = authnd.send_chat_completion(
        messages=[{"role": "user", "content": "test"}],
        model="z-ai/glm-5.1",
        stream=False,
        log_fn=logs.append,
    )

    assert result["content"] == "ok"
    assert len(token_calls) == 2
    assert all(call[1] == 75 for call in token_calls)
    assert any("captcha token flow failed (attempt 1/2)" in message for message in logs)
    assert any("fresh browser helper" in message for message in logs)


def test_send_chat_completion_honors_request_local_queue_cancel(monkeypatch):
    authnd._cancel_event.clear()
    state = {"cancelled": False}
    post_calls = []

    def fake_get_token(page_url, timeout, log_fn=None, cancel_check=None):
        assert callable(cancel_check)
        state["cancelled"] = True
        assert cancel_check() is True
        raise RuntimeError("stream cancelled")

    monkeypatch.setattr(authnd, "_get_captcha_token_for_request", fake_get_token)
    monkeypatch.setattr(
        authnd,
        "_post_prediction",
        lambda **kwargs: post_calls.append(kwargs),
    )

    with pytest.raises(RuntimeError, match="stream cancelled"):
        authnd.send_chat_completion(
            messages=[{"role": "user", "content": "test"}],
            model="z-ai/glm-5.1",
            stream=False,
            cancel_check=lambda: state["cancelled"],
        )

    assert post_calls == []
    assert authnd._cancel_event.is_set() is False


def test_authnd_provider_boundary_callback_runs_immediately_before_post(monkeypatch):
    authnd._cancel_event.clear()
    events = []

    monkeypatch.setattr(
        authnd,
        "_get_captcha_token_for_request",
        lambda *args, **kwargs: "fresh-token",
    )

    def fake_post_prediction(**kwargs):
        events.append("post")
        return {"content": "ok", "finish_reason": "stop"}

    monkeypatch.setattr(authnd, "_post_prediction", fake_post_prediction)

    result = authnd.send_chat_completion(
        messages=[{"role": "user", "content": "test"}],
        model="z-ai/glm-5.1",
        stream=False,
        before_send_callback=lambda: events.append("boundary"),
    )

    assert result["content"] == "ok"
    assert events == ["boundary", "post"]


def test_hcaptcha_timeout_hint_is_actionable(monkeypatch):
    monkeypatch.setenv("AUTHND_TOKEN_CONCURRENCY", "3")

    hint = authnd._captcha_token_failure_hint(
        RuntimeError("AuthND hCaptcha failed: timeout waiting for hcaptcha")
    )

    assert "js.hcaptcha.com" in hint
    assert "AUTHND_TOKEN_TIMEOUT" in hint
    assert "currently 3; try 1" in hint
