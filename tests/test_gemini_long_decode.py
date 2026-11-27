"""Gemini Long Decode setting and the google-genai 3 client rules (AI SDK upgrade).

* gemini_policy.gemini_long_decode: Off (default) / On / Auto (output limit above one 65,536-token
  hop), Gemini 4 only.
* UnifiedClient._apply_gemini_long_decode writes ``automatic_continuation`` explicitly, because
  google-genai 3 continues by default: Off must send False.
* Gemini Developer API clients stay on v1beta (google-genai 3 defaults to v1, which lacks the
  -latest aliases, previews, Live API and caches).
* FinishReason CONTINUATION counts as truncation.
"""

import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gemini_policy as gp


@pytest.mark.parametrize("mode, model, limit, expected", [
    ("off", "gemini-4-argon", 200000, False),
    ("on", "gemini-4-argon", 1000, True),
    ("on", "gemini-3.5-flash", 200000, False),       # not a Gemini 4 model
    ("auto", "gemini-4-hydrogen", 65536, False),     # one hop is enough
    ("auto", "gemini-4-hydrogen", 65537, True),
    ("auto", "vertex/gemini-4-neon", 131072, True),
    ("bogus", "gemini-4-carbon", 200000, False),     # unknown value -> off
])
def test_long_decode_modes(mode, model, limit, expected):
    enabled, reason = gp.gemini_long_decode(model, limit, mode)
    assert enabled is expected
    assert (reason is not None) is expected


def test_long_decode_reads_the_env_and_defaults_to_off(monkeypatch):
    monkeypatch.delenv("GEMINI_LONG_DECODE", raising=False)
    assert gp.gemini_long_decode("gemini-4-argon", 10 ** 6) == (False, None)
    monkeypatch.setenv("GEMINI_LONG_DECODE", "on")
    assert gp.gemini_long_decode("gemini-4-argon", 10)[0] is True


class _FakeTypes:
    class HttpOptions:
        def __init__(self, api_version=None):
            self.api_version = api_version


class _FakeGenai:
    types = _FakeTypes

    class Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs


def test_developer_api_client_is_pinned_to_v1beta():
    client = gp.new_gemini_api_client("key", _FakeGenai)
    assert client.kwargs["api_key"] == "key"
    assert client.kwargs["http_options"].api_version == "v1beta"


def test_real_sdk_client_uses_v1beta():
    genai = pytest.importorskip("google.genai")
    client = gp.new_gemini_api_client("dummy-key", genai)
    assert client._api_client._http_options.api_version == "v1beta"


@pytest.fixture
def uac(monkeypatch):
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    import unified_api_client
    return unified_api_client


def _bare_client(uac):
    client = uac.UnifiedClient.__new__(uac.UnifiedClient)
    client._is_stop_requested = lambda: False
    return client


@pytest.mark.parametrize("mode, limit, expected", [("off", 200000, False), ("on", 1000, True),
                                                   ("auto", 200000, True), ("auto", 65536, False)])
def test_apply_long_decode_sets_automatic_continuation(uac, monkeypatch, mode, limit, expected):
    types = pytest.importorskip("google.genai.types")
    if "automatic_continuation" not in types.GenerateContentConfig.model_fields:
        pytest.skip("google-genai without automatic continuation")
    monkeypatch.setenv("GEMINI_LONG_DECODE", mode)
    params = {"max_output_tokens": limit}
    assert _bare_client(uac)._apply_gemini_long_decode(params, "gemini-4-argon", limit) is expected
    # always explicit: google-genai 3 continues when the field is left out
    assert params["automatic_continuation"] is expected
    assert params["max_output_tokens"] == limit
    types.GenerateContentConfig(**params)  # the SDK accepts the field


def test_continuation_finish_reason_is_truncation():
    source = (SRC / "unified_api_client.py").read_text(encoding="utf-8")
    assert "elif fr_name in ('MAX_TOKENS', 'CONTINUATION'):" in source
    assert "_FR_LENGTH = {'MAX_TOKENS', 'CONTINUATION'}" in source


def test_no_aiplatform_or_vertexai_imports():
    source = (SRC / "unified_api_client.py").read_text(encoding="utf-8")
    assert "from google.cloud import aiplatform" not in source
    assert "import vertexai" not in source
    assert "from vertexai" not in source
