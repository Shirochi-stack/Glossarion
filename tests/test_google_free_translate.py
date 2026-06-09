import sys
import time
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


class SlowInterruptClient:
    def __init__(self, sleep_seconds=0.6):
        self.sleep_seconds = sleep_seconds
        self.cancel_calls = 0
        self._cancelled = False
        self._in_cleanup = False
        self.tls = types.SimpleNamespace(local_cancel_check=None)

    def _get_thread_local_client(self):
        return self.tls

    def is_globally_cancelled(self):
        return False

    def cancel_current_operation(self):
        self.cancel_calls += 1
        self._cancelled = True

    def send(self, messages, temperature=0.0, max_tokens=None, context=None):
        time.sleep(self.sleep_seconds)
        return "late result"


def _install_fake_argos(monkeypatch, translate_func):
    calls = []

    class FakeTranslation:
        def translate(self, text):
            calls.append(text)
            return translate_func(text)

    translation = FakeTranslation()

    class FakeLanguage:
        def __init__(self, code):
            self.code = code

        def get_translation(self, to_lang):
            if self.code == "ko" and getattr(to_lang, "code", "") == "en":
                return translation
            return None

    package_mod = types.ModuleType("argostranslate.package")
    package_mod.update_package_index = lambda: None
    package_mod.get_available_packages = lambda: []
    package_mod.install_from_path = lambda path: None

    translate_mod = types.ModuleType("argostranslate.translate")
    translate_mod.get_installed_languages = lambda: [FakeLanguage("ko"), FakeLanguage("en")]

    parent_mod = types.ModuleType("argostranslate")
    parent_mod.package = package_mod
    parent_mod.translate = translate_mod

    monkeypatch.setitem(sys.modules, "argostranslate", parent_mod)
    monkeypatch.setitem(sys.modules, "argostranslate.package", package_mod)
    monkeypatch.setitem(sys.modules, "argostranslate.translate", translate_mod)
    return calls


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


def test_google_free_translate_honors_global_stop_by_default(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "unified_api_client",
        types.SimpleNamespace(is_stop_requested=lambda: True),
    )
    monkeypatch.setattr(GoogleFreeTranslateNew, "_use_fallback_only", False, raising=False)

    def fake_post(*_args, **_kwargs):
        raise AssertionError("stopped translations must not hit Google endpoints")

    monkeypatch.setattr("google_free_translate.requests.post", fake_post)

    translator = GoogleFreeTranslateNew(source_language="auto", target_language="Spanish")
    translator.rate_limit = 0

    result = translator.translate("Hello")

    assert result["translatedText"] == ""
    assert result["provider"] == "auto"
    assert "Operation cancelled" in result["error"]


def test_google_free_translate_can_ignore_global_stop_for_viewer(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "unified_api_client",
        types.SimpleNamespace(is_stop_requested=lambda: True),
    )
    monkeypatch.setattr(GoogleFreeTranslateNew, "_use_fallback_only", False, raising=False)

    def fake_post(url, data=None, headers=None, timeout=None):
        assert url == GOOGLE_TRANSLATE_CO_IN_ENDPOINT
        return FakeResponse(200, [[["Hola", "Hello", None, None, 10]], None, "en"])

    monkeypatch.setattr("google_free_translate.requests.post", fake_post)

    translator = GoogleFreeTranslateNew(
        source_language="auto",
        target_language="Spanish",
        honor_global_stop=False,
    )
    translator.rate_limit = 0

    result = translator.translate("Hello")

    assert result["translatedText"] == "Hola"
    assert result["endpoint"] == GOOGLE_TRANSLATE_CO_IN_ENDPOINT


def test_send_with_interrupt_timeout_does_not_mark_user_cancel(monkeypatch):
    from TransateKRtoEN import send_with_interrupt
    from unified_api_client import UnifiedClientError

    monkeypatch.setenv("RETRY_TIMEOUT", "1")
    monkeypatch.setenv("THREAD_SUBMISSION_DELAY_SECONDS", "0")
    monkeypatch.delenv("GRACEFUL_STOP", raising=False)

    client = SlowInterruptClient()

    with pytest.raises(UnifiedClientError, match="timed out"):
        send_with_interrupt(
            [{"role": "user", "content": "hello"}],
            client,
            temperature=0.0,
            max_tokens=8,
            stop_check_fn=lambda: False,
            chunk_timeout=0.01,
            context="translation",
        )

    assert client.cancel_calls == 0
    assert client._cancelled is False


def test_send_with_interrupt_user_stop_marks_client_cancel(monkeypatch):
    from TransateKRtoEN import send_with_interrupt
    from unified_api_client import UnifiedClientError

    monkeypatch.setenv("RETRY_TIMEOUT", "1")
    monkeypatch.setenv("THREAD_SUBMISSION_DELAY_SECONDS", "0")
    monkeypatch.delenv("GRACEFUL_STOP", raising=False)

    client = SlowInterruptClient()

    with pytest.raises(UnifiedClientError, match="Translation stopped by user"):
        send_with_interrupt(
            [{"role": "user", "content": "hello"}],
            client,
            temperature=0.0,
            max_tokens=8,
            stop_check_fn=lambda: True,
            chunk_timeout=10,
            context="translation",
        )

    assert client.cancel_calls == 1
    assert client._cancelled is True


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


def test_auto_argos_fallback_reports_failed_google_endpoints(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "unified_api_client",
        types.SimpleNamespace(is_stop_requested=lambda: False),
    )
    monkeypatch.setattr(GoogleFreeTranslateNew, "_use_fallback_only", False, raising=False)
    _install_fake_argos(monkeypatch, lambda _text: "Hello")

    def fake_post(url, data=None, headers=None, timeout=None):
        raise RuntimeError(f"blocked {url}")

    def fake_get(url, params=None, headers=None, timeout=None):
        raise RuntimeError(f"blocked {url}")

    monkeypatch.setattr("google_free_translate.requests.post", fake_post)
    monkeypatch.setattr("google_free_translate.requests.get", fake_get)

    translator = GoogleFreeTranslateNew(source_language="Korean", target_language="English")
    translator.rate_limit = 0

    result = translator.translate("안녕")

    assert result["provider"] == "argos"
    assert result["translatedText"] == "Hello"
    assert result["fallback_from_provider"] == "google"
    assert result["fallback_failed_endpoints"] == [
        "translate.google.co.in/single",
        "translate.google.com/single",
        "clients5.google.com/t",
        "clients5.google.com/single",
        "clients1.google.com/single",
        "clients3.google.com/t",
        "translate.googleapis.com/single",
    ]
    assert result["fallback_note"] == (
        "Auto fell back to Argos after Google endpoints failed: "
        "translate.google.co.in/single, translate.google.com/single, clients5.google.com/t, "
        "clients5.google.com/single, clients1.google.com/single, clients3.google.com/t, "
        "translate.googleapis.com/single"
    )


def test_argos_fallback_translates_marked_html_segments_without_tag_bleed(monkeypatch):
    monkeypatch.setattr(GoogleFreeTranslateNew, "_use_fallback_only", True, raising=False)
    calls = _install_fake_argos(
        monkeypatch,
        lambda text: {
            "Title raw": "Title translated News /p>",
            "Body raw": "Body translated </p><p>",
        }.get(text, text),
    )
    batch_html = '<h1 data-sdl-tip="0">Title raw</h1>\n<p data-sdl-tip="1">Body raw</p>'

    translator = GoogleFreeTranslateNew(source_language="Korean", target_language="English")
    translator.rate_limit = 0
    result = translator.translate(batch_html)

    assert calls == ["Title raw", "Body raw"]
    assert result["provider"] == "argos"
    assert result["translatedText"] == (
        '<h1 data-sdl-tip="0">Title translated News</h1>\n'
        '<p data-sdl-tip="1">Body translated</p>'
    )
    assert "News /p>" not in result["translatedText"]
    assert "</p><p>" not in result["translatedText"]


def test_argos_fallback_sanitizes_plain_text_tag_fragments(monkeypatch):
    monkeypatch.setattr(GoogleFreeTranslateNew, "_use_fallback_only", True, raising=False)
    _install_fake_argos(
        monkeypatch,
        lambda _text: "First sentence. News /p> Second p> <p>Third</p>",
    )

    translator = GoogleFreeTranslateNew(source_language="Korean", target_language="English")
    translator.rate_limit = 0
    result = translator.translate("Plain source")

    assert result["provider"] == "argos"
    assert result["translatedText"] == "First sentence. News Second Third"


def test_deepl_provider_requires_api_key_without_raw_fallback(monkeypatch):
    monkeypatch.delenv("DEEPL_API_KEY", raising=False)

    translator = GoogleFreeTranslateNew(source_language="auto", target_language="English", provider="deepl")
    result = translator.translate("Hello")

    assert result["provider"] == "deepl"
    assert result["translatedText"] == ""
    assert "API key is not configured" in result["error"]


def test_deepl_provider_uses_auth_key_and_html_tag_handling(monkeypatch):
    calls = []

    def fake_post(url, data=None, headers=None, timeout=None):
        calls.append((url, dict(data or {}), dict(headers or {}), timeout))
        return FakeResponse(
            200,
            {
                "translations": [
                    {
                        "text": '<p data-sdl-tip="0">Hello</p>',
                        "detected_source_language": "KO",
                    }
                ]
            },
        )

    monkeypatch.setattr("google_free_translate.requests.post", fake_post)

    translator = GoogleFreeTranslateNew(
        source_language="Korean",
        target_language="English",
        provider="deepl",
        api_keys={"deepl": {"api_key": "deepl-key:fx"}},
    )
    result = translator.translate('<p data-sdl-tip="0">안녕</p>')

    assert result["provider"] == "deepl"
    assert result["translatedText"] == '<p data-sdl-tip="0">Hello</p>'
    assert calls[0][0] == "https://api-free.deepl.com/v2/translate"
    assert calls[0][1]["tag_handling"] == "html"
    assert calls[0][1]["target_lang"] == "EN-US"
    assert calls[0][2]["Authorization"] == "DeepL-Auth-Key deepl-key:fx"


def test_bing_provider_uses_subscription_key_and_html_mode(monkeypatch):
    calls = []

    def fake_post(url, params=None, headers=None, json=None, timeout=None):
        calls.append((url, dict(params or {}), dict(headers or {}), json, timeout))
        return FakeResponse(
            200,
            [
                {
                    "detectedLanguage": {"language": "ko"},
                    "translations": [{"text": '<p data-sdl-tip="0">Hello</p>', "to": "en"}],
                }
            ],
        )

    monkeypatch.setattr("google_free_translate.requests.post", fake_post)

    translator = GoogleFreeTranslateNew(
        source_language="auto",
        target_language="English",
        provider="bing",
        api_keys={"bing": {"api_key": "bing-key", "region": "eastus"}},
    )
    result = translator.translate('<p data-sdl-tip="0">안녕</p>')

    assert result["provider"] == "bing"
    assert result["translatedText"] == '<p data-sdl-tip="0">Hello</p>'
    assert calls[0][0] == "https://api.cognitive.microsofttranslator.com/translate"
    assert calls[0][1]["textType"] == "html"
    assert calls[0][1]["to"] == "en"
    assert calls[0][2]["Ocp-Apim-Subscription-Key"] == "bing-key"
    assert calls[0][2]["Ocp-Apim-Subscription-Region"] == "eastus"
    assert calls[0][3] == [{"Text": '<p data-sdl-tip="0">안녕</p>'}]


def test_yandex_provider_uses_api_key_and_folder_id(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append((url, dict(headers or {}), dict(json or {}), timeout))
        return FakeResponse(
            200,
            {
                "translations": [
                    {
                        "text": '<p data-sdl-tip="0">Hello</p>',
                        "detectedLanguageCode": "ko",
                    }
                ]
            },
        )

    monkeypatch.setattr("google_free_translate.requests.post", fake_post)

    translator = GoogleFreeTranslateNew(
        source_language="Korean",
        target_language="English",
        provider="yandex",
        api_keys={"yandex": {"api_key": "yandex-key", "folder_id": "folder-123"}},
    )
    result = translator.translate('<p data-sdl-tip="0">안녕</p>')

    assert result["provider"] == "yandex"
    assert result["translatedText"] == '<p data-sdl-tip="0">Hello</p>'
    assert calls[0][0] == "https://translate.api.cloud.yandex.net/translate/v2/translate"
    assert calls[0][1]["Authorization"] == "Api-Key yandex-key"
    assert calls[0][2]["folderId"] == "folder-123"
    assert calls[0][2]["format"] == "HTML"
    assert calls[0][2]["texts"] == ['<p data-sdl-tip="0">안녕</p>']
