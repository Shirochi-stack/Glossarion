import sys
import types

import pytest

from google_free_translate import (
    GOOGLE_TRANSLATE_CO_IN_ENDPOINT,
    GOOGLETRANS_AJAX_ENDPOINT,
    GoogleFreeTranslateNew,
    google_translate_language_code,
)


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


@pytest.mark.parametrize(
    ("display_name", "code"),
    [
        ("English", "en"),
        ("Spanish", "es"),
        ("French", "fr"),
        ("German", "de"),
        ("Italian", "it"),
        ("Portuguese", "pt"),
        ("Russian", "ru"),
        ("Arabic", "ar"),
        ("Hindi", "hi"),
        ("Chinese (Simplified)", "zh-CN"),
        ("Chinese (Traditional)", "zh-TW"),
        ("Japanese", "ja"),
        ("Korean", "ko"),
        ("Turkish", "tr"),
        ("Vietnamese", "vi"),
    ],
)
def test_google_translate_language_code_matches_gui_dropdown(display_name, code):
    assert google_translate_language_code(display_name) == code


def test_google_free_translate_uses_co_in_first(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "unified_api_client",
        types.SimpleNamespace(is_stop_requested=lambda: False),
    )
    monkeypatch.setattr(GoogleFreeTranslateNew, "_use_fallback_only", False, raising=False)

    calls = []

    def fake_post(url, data=None, headers=None, timeout=None):
        calls.append(("post", url, dict(data or {})))
        assert url == GOOGLE_TRANSLATE_CO_IN_ENDPOINT
        assert data["client"] == "gtx"
        return FakeResponse(200, [[["Hola", "Hello", None, None, 10]], None, "en"])

    monkeypatch.setattr("google_free_translate.requests.post", fake_post)

    translator = GoogleFreeTranslateNew(source_language="auto", target_language="Spanish")
    translator.rate_limit = 0

    result = translator.translate("Hello")

    assert result["translatedText"] == "Hola"
    assert result["endpoint"] == GOOGLE_TRANSLATE_CO_IN_ENDPOINT
    assert calls[0] == (
        "post",
        GOOGLE_TRANSLATE_CO_IN_ENDPOINT,
        {"client": "gtx", "sl": "auto", "tl": "es", "dt": "t", "q": "Hello"},
    )


def test_google_free_translate_co_in_uses_dropdown_target(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "unified_api_client",
        types.SimpleNamespace(is_stop_requested=lambda: False),
    )
    monkeypatch.setattr(GoogleFreeTranslateNew, "_use_fallback_only", False, raising=False)

    def fake_post(url, data=None, headers=None, timeout=None):
        assert url == GOOGLE_TRANSLATE_CO_IN_ENDPOINT
        assert data["client"] == "gtx"
        assert data["tl"] == "vi"
        return FakeResponse(200, [[["Xin chao", "Hello", None, None, 10]], None, "en"])

    monkeypatch.setattr("google_free_translate.requests.post", fake_post)

    translator = GoogleFreeTranslateNew(source_language="auto", target_language="Vietnamese")
    translator.rate_limit = 0

    result = translator.translate("Hello")

    assert result["translatedText"] == "Xin chao"
    assert result["endpoint"] == GOOGLE_TRANSLATE_CO_IN_ENDPOINT


def test_google_free_translate_keeps_ajax_endpoint_last(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "unified_api_client",
        types.SimpleNamespace(is_stop_requested=lambda: False),
    )
    monkeypatch.setattr(GoogleFreeTranslateNew, "_use_fallback_only", False, raising=False)

    calls = []

    def fake_post(url, data=None, headers=None, timeout=None):
        calls.append(("post", url, dict(data or {})))
        raise RuntimeError("post endpoint unavailable")

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(("get", url, dict(params or {})))
        assert url == GOOGLETRANS_AJAX_ENDPOINT
        assert params["client"] == "gtx"
        return FakeResponse(200, [[["Hola", "Hello", None, None, 10]], None, "en"])

    monkeypatch.setattr("google_free_translate.requests.post", fake_post)
    monkeypatch.setattr("google_free_translate.requests.get", fake_get)

    translator = GoogleFreeTranslateNew(source_language="auto", target_language="Spanish")
    translator.rate_limit = 0

    result = translator.translate("Hello")

    assert result["translatedText"] == "Hola"
    assert result["endpoint"] == GOOGLETRANS_AJAX_ENDPOINT
    assert calls[0][0] == "post"
    assert calls[0][1] == GOOGLE_TRANSLATE_CO_IN_ENDPOINT
    assert calls[-1] == (
        "get",
        GOOGLETRANS_AJAX_ENDPOINT,
        {"client": "gtx", "sl": "auto", "tl": "es", "dt": "t", "q": "Hello"},
    )
