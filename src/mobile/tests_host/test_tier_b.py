"""Tier-B dependencies (U9): the plan's dependency rule applied to every "needs component" feature.

A non-npm feature ships when its packages resolve for Android + iOS (``tools/check_mobile_wheels.py``
over all five targets); otherwise a REST equivalent is used; only native-impossible items stay
disabled with a reason. Outcomes (2026-10-08, check_mobile_wheels with ``--wheelhouse-plan`` for
the security pins; the only errors without it are the documented cryptography / pillow ones):

=====================================  ==========================================================
Feature                                Outcome
=====================================  ==========================================================
Gemini gRPC transport                  pinned: grpcio 1.81.0 + grpcio-status 1.81.0 (grpcio-status
                                       >= 1.84 needs grpcio >= 1.84, which has no mobile wheel) +
                                       google-api-core 2.41.0 + google-ai-generativelanguage 0.12.1
Google Cloud Translate (paid route)    pinned: google-cloud-translate 3.28.0 (translate_v2 = REST)
Google Cloud Text-to-Speech            pinned: google-cloud-texttospeech 2.38.0 (resolves only with
                                       the grpcio-status pin; U7 saw the 1.84 trap)
Google Cloud Vision (manga OCR)        pinned: google-cloud-vision 3.16.0; google_vision_rest (U8)
                                       stays the no-SDK fallback
Azure Document Intelligence            no pin: azure-ai-documentintelligence 1.0.2 resolves but no
                                       code imports it (ocr_manager imports azure.ai.formrecognizer;
                                       DISCREPANCIES "U8 Integrate", OCR providers, desktop bug 3);
                                       azure_document_intelligence_rest (U8) stays the
                                       mobile path
Vertex AI / Model Garden               google-cloud-aiplatform fails: every release needs protobuf<7
                                       and the app pins protobuf 7.35.0 -> REST path: Gemini through
                                       google-genai (vertexai=True), Claude through
                                       anthropic.AnthropicVertex, google-auth service-account tokens;
                                       unified_api_client skips the unused aiplatform / vertexai
                                       imports on mobile only
deep-translator                        already a base pin (1.11.4): QA silent-truncation heuristic
QA silent-truncation embeddings        disabled: sentence-transformers needs torch + hf-xet (no
                                       Android/iOS wheels) -> settings_schema UNAVAILABLE_RULES
Argos offline MT                       disabled: argostranslate needs ctranslate2 + sentencepiece
                                       (+ stanza -> torch) -> settings_schema UNAVAILABLE_VALUE_RULES
=====================================  ==========================================================

The runtime tests drive the real SDKs (protobuf 7.35 runtime) against loopback servers: an
in-process gRPC server for Gemini gRPC / Cloud TTS / Cloud Vision and an HTTP server for the
Translation v2 REST client and the Vertex REST paths (OAuth token endpoint included). They need
the Tier-B packages (``uv sync`` from uv.lock); outside CI they skip when the env predates them.

Real data is never touched: HOME, USERPROFILE, OUTPUT_DIRECTORY, GLOSSARION_LIBRARY_DIR,
GLOSSARION_DATA_DIR and CONFIG_FILE point at the test's tmp dir, HTTP logging is off and only
loopback sockets are used.

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_tier_b.py
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import types
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - the mobile host env is 3.13
    tomllib = None

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
TOOLS_DIR = MOBILE_DIR / "tools"
for _p in (str(APP_DIR), str(TOOLS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

PYPROJECT = MOBILE_DIR / "pyproject.toml"
UV_LOCK = MOBILE_DIR / "uv.lock"
MANIFEST = MOBILE_DIR / "backend_manifest.toml"

#: The Tier-B pins U9 adds to [project].dependencies (distribution -> exact version).
TIER_B_PINS = {
    "grpcio": "1.81.0",
    "grpcio-status": "1.81.0",
    "google-api-core": "2.41.0",
    "google-ai-generativelanguage": "0.12.1",
    "google-cloud-translate": "3.28.0",
    "google-cloud-texttospeech": "2.38.0",
    "google-cloud-vision": "3.16.0",
}
#: Checked and deliberately not shipped (distribution -> word the manifest reason must contain).
NOT_SHIPPED = {
    "google-cloud-aiplatform": "protobuf",
    "vertexai": "google-genai",
    "azure-ai-documentintelligence": "formrecognizer",
    "azure-ai-formrecognizer": "REST",
    "sentence-transformers": "torch",
    "argostranslate": "ctranslate2",
}
#: Import names the backend uses -> the pinned distribution ([thirdparty.map]).
IMPORT_MAP = {
    "grpc": "grpcio",
    "google.api_core": "google-api-core",
    "google.ai.generativelanguage_v1beta": "google-ai-generativelanguage",
    "google.cloud.translate_v2": "google-cloud-translate",
    "google.cloud.texttospeech": "google-cloud-texttospeech",
    "google.cloud.vision": "google-cloud-vision",
}
TIER_B_MODULES = ("grpc", "google.api_core", "google.ai.generativelanguage_v1beta", "google.cloud.translate_v2",
                  "google.cloud.texttospeech", "google.cloud.vision")

_PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def _norm(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(".", "-")


def _pins(requirements) -> dict:
    out = {}
    for req in requirements:
        name, sep, version = str(req).partition("==")
        out[_norm(name.split("[")[0].split(";")[0])] = version.split(";")[0].strip() if sep else None
    return out


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _tier_b_installed() -> bool:
    return all(_has(m) for m in TIER_B_MODULES)


def _require_tier_b() -> None:
    """Skip when the env predates the U9 pins; CI's env comes from uv.lock, so there it must exist."""
    if _tier_b_installed():
        return
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        pytest.fail("Tier-B packages are missing from the CI env built from uv.lock")
    pytest.skip("Tier-B packages not installed in this env (uv sync from src/mobile/uv.lock)")


needs_toml = pytest.mark.skipif(tomllib is None, reason="tomllib needs Python 3.11+")


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Scratch HOME / output / library / data dirs, desktop process (no GLOSSARION_MOBILE), loopback
    sockets without proxies, HTTP logging off."""
    for name, sub in (("HOME", "home"), ("USERPROFILE", "home"), ("OUTPUT_DIRECTORY", "out"),
                      ("GLOSSARION_LIBRARY_DIR", "lib"), ("GLOSSARION_DATA_DIR", "data"),
                      ("GLOSSARION_APP_DIR", "data")):
        path = tmp_path / sub
        path.mkdir(exist_ok=True)
        monkeypatch.setenv(name, str(path))
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "data" / "config.json"))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    for name in _PROXY_VARS + ("GLOSSARION_MOBILE", "GLOSSARION_NO_PROCESSES", "FLET_PLATFORM",
                               "GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_CREDENTIALS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(scope="module")
def uac():
    """unified_api_client, imported with the HTTP logger off (it patches requests otherwise)."""
    if "unified_api_client" not in sys.modules:
        saved = os.environ.get("GLOSSARION_HTTP_LOG")
        os.environ["GLOSSARION_HTTP_LOG"] = "0"
        try:
            importlib.import_module("unified_api_client")
        finally:
            if saved is None:
                os.environ.pop("GLOSSARION_HTTP_LOG", None)
            else:
                os.environ["GLOSSARION_HTTP_LOG"] = saved
    return sys.modules["unified_api_client"]


# ============================================================================ pins, lock, manifest


@needs_toml
def test_pyproject_pins_exactly_the_tier_b_set():
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    deps = _pins(data["project"]["dependencies"])
    for dist, version in TIER_B_PINS.items():
        assert deps.get(dist) == version, f"{dist}=={version} missing from [project].dependencies"
    # protobuf stays at the one Android/iOS build the Google SDKs accept (<8, >=6.33.5).
    assert deps.get("protobuf") == "7.35.0"
    platform_deps = {}
    for os_name in ("android", "ios"):
        platform_deps.update(_pins(data["tool"]["flet"].get(os_name, {}).get("dependencies", [])))
    for dist in NOT_SHIPPED:
        assert dist not in deps and dist not in platform_deps, f"{dist} must not be pinned (see the module docstring)"
    # The security pins are never touched by Tier B.
    assert deps.get("cryptography") == "50.0.2" and deps.get("pillow") == "12.3.0"


@needs_toml
def test_uv_lock_resolves_the_tier_b_pins():
    lock = tomllib.loads(UV_LOCK.read_text(encoding="utf-8"))
    versions = {_norm(p["name"]): p["version"] for p in lock["package"]}
    for dist, version in TIER_B_PINS.items():
        assert versions.get(dist) == version, (dist, versions.get(dist))
    assert versions.get("protobuf") == "7.35.0"
    for dist in NOT_SHIPPED:
        assert dist not in versions, f"{dist} is in uv.lock"
    root = next(p for p in lock["package"] if p["name"] == "glossarion")
    requires = {_norm(r["name"]): r.get("specifier") for r in root["metadata"]["requires-dist"]}
    for dist, version in TIER_B_PINS.items():
        assert requires.get(dist) == f"=={version}", (dist, requires.get(dist))


@needs_toml
def test_manifest_maps_tier_b_imports_and_keeps_the_disabled_ones():
    manifest = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    mapping = manifest["thirdparty"]["map"]
    unavailable = {_norm(k): v for k, v in manifest["thirdparty"]["unavailable"].items()}
    for module, dist in IMPORT_MAP.items():
        assert mapping.get(module) == dist, module
        assert _norm(dist) not in unavailable, f"{dist} is pinned but still listed as unavailable"
    for dist in TIER_B_PINS:
        assert _norm(dist) not in unavailable, dist
    for dist, word in NOT_SHIPPED.items():
        reason = unavailable.get(_norm(dist), "")
        assert word in reason, (dist, reason)


def test_host_smoke_no_longer_blocks_the_tier_b_packages():
    import host_smoke

    for platform in ("android", "ios"):
        blocked = set(host_smoke.blocked_packages(platform))
        for module in ("grpc", "google.ai.generativelanguage", "google.ai.generativelanguage_v1beta",
                       "google.cloud.translate", "google.cloud.texttospeech", "google.cloud.vision",
                       "google.api_core"):
            assert module not in blocked, (platform, module)
        for module in ("google.cloud.aiplatform", "vertexai", "sentence_transformers", "argostranslate"):
            assert module in blocked, (platform, module)


@needs_toml
def test_collector_classifies_tier_b_imports_as_pinned(tmp_path):
    """The real backend closure: every Tier-B import is 'pinned', the disabled ones 'unavailable'."""
    if not (SRC_DIR / "TransateKRtoEN.py").exists():
        pytest.skip("desktop sources not present")
    import collect_backend as cb

    cache = tmp_path / "cache.json"
    if cb.DEFAULT_CACHE.exists():
        shutil.copyfile(cb.DEFAULT_CACHE, cache)
    result = cb.Collector(SRC_DIR, cb.load_manifest(cb.DEFAULT_MANIFEST), cb.load_pins(cb.DEFAULT_PYPROJECT),
                          mobile_dir=MOBILE_DIR, cache_path=cache).run()
    third = result.thirdparty
    for dist in ("grpcio", "google-api-core", "google-ai-generativelanguage", "google-cloud-translate",
                 "google-cloud-texttospeech", "google-cloud-vision", "deep-translator", "google-genai",
                 "google-auth", "anthropic"):
        assert third.get(dist, {}).get("status") == "pinned", (dist, third.get(dist))
    for dist in ("google-cloud-aiplatform", "vertexai", "sentence-transformers", "argostranslate"):
        assert third.get(dist, {}).get("status") == "unavailable", (dist, third.get(dist))
    tier_b_errors = [f for f in result.errors if any(n in f.message for n in TIER_B_PINS)]
    assert not tier_b_errors, [f.message for f in tier_b_errors]
    assert "grpc_gemini_client" in set(result.closure)


# ============================================================================ availability (schema + Endpoints)


def test_schema_overlay_marks_only_the_native_impossible_items():
    import settings_schema as schema

    ok, reason = schema.is_available("qa_scanner_settings.truncation_embed_threshold", "mobile")
    assert not ok and "sentence-transformers" in reason
    assert schema.is_available("qa_scanner_settings.truncation_embed_threshold", "desktop") == (True, "")
    for value in ("argos", "Argos", "argostranslate"):
        ok, reason = schema.is_value_available("sdlxliff_machine_translation_provider", value, "mobile")
        assert not ok and "ctranslate2" in reason, value
    assert schema.is_value_available("sdlxliff_machine_translation_provider", "argos", "desktop") == (True, "")
    for value in ("auto", "google", "deepl", "bing", "yandex"):
        assert schema.is_value_available("sdlxliff_machine_translation_provider", value, "mobile") == (True, "")
    # The Tier-B routes ship: their settings stay enabled on mobile.
    for key in ("gemini_openai_endpoint", "use_gemini_openai_endpoint", "vertex_ai_location", "google_cloud_credentials",
                "tts_voice", "qa_scanner_settings.check_silent_truncation",
                "qa_scanner_settings.truncation_cheap_threshold", "qa_scanner_settings.truncation_borderline_score"):
        assert schema.is_available(key, "mobile") == (True, ""), key


def test_endpoint_dependency_reasons_follow_the_build(monkeypatch):
    from glossarion_mobile.ui.screens import endpoints

    present = set()
    monkeypatch.setattr(endpoints, "_has", lambda module: module in present)
    reasons = endpoints.dependency_reasons()
    assert set(reasons) == {"gemini_openai_endpoint", "vertex_ai_location", "tts_voice"}
    assert reasons["gemini_openai_endpoint"][0] == "Needs grpcio · not in this build"
    assert "google-ai-generativelanguage" in reasons["gemini_openai_endpoint"][1]
    assert "google-cloud-texttospeech" in reasons["tts_voice"][1]

    present.update({"grpc"})  # grpc without the generativelanguage protos is not enough
    assert "gemini_openai_endpoint" in endpoints.dependency_reasons()
    present.update({"google.ai.generativelanguage_v1beta", "google.cloud.texttospeech"})
    assert set(endpoints.dependency_reasons()) == {"vertex_ai_location"}
    # Vertex: the REST path (google-genai + google-auth) is enough, aiplatform is never required.
    present.update({"google.genai", "google.auth"})
    assert endpoints.dependency_reasons() == {}
    present.difference_update({"google.genai", "google.auth"})
    present.add("google.cloud.aiplatform")
    assert endpoints.dependency_reasons() == {}


def test_endpoint_dependency_reasons_are_empty_with_the_tier_b_env():
    _require_tier_b()
    from glossarion_mobile.ui.screens import endpoints

    assert endpoints.dependency_reasons() == {}


# ============================================================================ the SDKs under protobuf 7


def test_installed_tier_b_versions_match_the_pins():
    _require_tier_b()
    for dist, version in TIER_B_PINS.items():
        assert importlib.metadata.version(dist) == version, dist
    assert importlib.metadata.version("protobuf") == "7.35.0"
    assert importlib.metadata.version("deep-translator") == "1.11.4"
    from deep_translator import GoogleTranslator  # QA silent-truncation heuristic (scan_html_folder)

    assert GoogleTranslator(source="auto", target="en") is not None


def test_unified_api_client_sees_the_tier_b_sdks(uac):
    _require_tier_b()
    # _ensure_*: the import-time values on desktop, the first-use import on mobile (deferred, below)
    assert uac._ensure_grpc_gemini() is True and uac.GrpcGeminiClient is not None
    assert uac._ensure_google_translate() is True and uac.google_translate is not None
    # uv.lock has no aiplatform / vertexai (protobuf<7): the REST path below serves Vertex.
    assert uac.VERTEX_AI_AVAILABLE is (_has("vertexai") and _has("google.cloud.aiplatform"))


_DEFERRED_IMPORT_PROBE = r"""
import json, os, sys
sys.path.insert(0, sys.argv[1])
import unified_api_client as uac
loaded = lambda: {"grpc": "grpc" in sys.modules, "translate": "google.cloud.translate_v2" in sys.modules,
                  "generativelanguage": any(m.startswith("google.ai.generativelanguage") for m in sys.modules)}
out = {"import": loaded(), "flags": [uac.GEMINI_GRPC_AVAILABLE, uac.GOOGLE_TRANSLATE_AVAILABLE]}
out["ensure"] = [uac._ensure_grpc_gemini(), uac._ensure_google_translate()]
out["after"] = loaded()
out["client"] = uac.GrpcGeminiClient is not None and uac.GrpcGeminiError is not None and uac.google_translate is not None
print("PROBE " + json.dumps(out))
"""


@pytest.mark.parametrize("mobile", [True, False])
def test_grpc_sdks_are_imported_on_first_use_on_mobile_only(isolated, mobile):
    """The launch warm import (runtime_bootstrap.WARM_IMPORT_MODULES) imports unified_api_client; on
    mobile it must not pull grpcio in through the two opt-in routes (grpc_gemini_client: ~110
    google.ai.generativelanguage modules; google.cloud.translate_v2: the gRPC translate_v3 package).
    The first request that uses one imports it. Desktop keeps the import-time imports."""
    _require_tier_b()
    env = dict(os.environ, GLOSSARION_HTTP_LOG="0", PYTHONIOENCODING="utf-8")
    for name in ("GLOSSARION_MOBILE", "GLOSSARION_NO_PROCESSES"):
        env.pop(name, None)
    if mobile:
        env.update(GLOSSARION_MOBILE="1", GLOSSARION_NO_PROCESSES="1")
    out = subprocess.run([sys.executable, "-c", _DEFERRED_IMPORT_PROBE, str(SRC_DIR)], cwd=str(isolated), env=env,
                         capture_output=True, text=True, encoding="utf-8", timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    probe = json.loads(next(line for line in out.stdout.splitlines() if line.startswith("PROBE "))[6:])
    everything = {"grpc": True, "translate": True, "generativelanguage": True}
    if mobile:
        assert probe["import"] == {"grpc": False, "translate": False, "generativelanguage": False}
        assert probe["flags"] == [False, False]
    else:
        assert probe["import"] == everything and probe["flags"] == [True, True]
    assert probe["ensure"] == [True, True] and probe["after"] == everything and probe["client"] is True


@contextlib.contextmanager
def _grpc_loopback(routes):
    """In-process gRPC server on 127.0.0.1 for proto-plus unary methods.

    ``routes``: {"/pkg.Service/Method": (RequestType, ResponseType, handler(request, context))}.
    Yields a factory of insecure channels to it."""
    from concurrent import futures

    import grpc

    class _Generic(grpc.GenericRpcHandler):
        def service(self, details):
            route = routes.get(details.method)
            if route is None:
                return None
            request_type, response_type, handler = route
            return grpc.unary_unary_rpc_method_handler(handler, request_deserializer=request_type.deserialize,
                                                       response_serializer=response_type.serialize)

    pool = futures.ThreadPoolExecutor(max_workers=2)
    server = grpc.server(pool)
    server.add_generic_rpc_handlers((_Generic(),))
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channels = []

    def channel():
        ch = grpc.insecure_channel(f"127.0.0.1:{port}")
        channels.append(ch)
        return ch

    try:
        yield channel
    finally:
        for ch in channels:
            ch.close()
        server.stop(None).wait(10)
        pool.shutdown(wait=True)


def test_gemini_grpc_transport_round_trip(isolated):
    """grpc_gemini_client (the bare-host Gemini endpoint) over a real gRPC channel."""
    _require_tier_b()
    import grpc_gemini_client as ggc
    from google.ai.generativelanguage_v1beta import (GenerateContentRequest, GenerateContentResponse,
                                                     GenerativeServiceClient)
    from google.ai.generativelanguage_v1beta.services.generative_service.transports import (
        GenerativeServiceGrpcTransport)

    seen = {}

    def generate(request, context):
        seen["model"] = request.model
        seen["user"] = [part.text for content in request.contents for part in content.parts]
        seen["system"] = [part.text for part in request.system_instruction.parts]
        seen["temperature"] = request.generation_config.temperature
        seen["max_output_tokens"] = request.generation_config.max_output_tokens
        seen["metadata"] = dict(context.invocation_metadata())
        return GenerateContentResponse(
            candidates=[{"content": {"role": "model", "parts": [{"text": "Hello there."}]}, "finish_reason": 1}],
            usage_metadata={"prompt_token_count": 7, "candidates_token_count": 3, "total_token_count": 10})

    route = "/google.ai.generativelanguage.v1beta.GenerativeService/GenerateContent"
    with _grpc_loopback({route: (GenerateContentRequest, GenerateContentResponse, generate)}) as channel:
        client = ggc.GrpcGeminiClient("AIza-test-key", endpoint="generativelanguage.googleapis.com")
        assert client.endpoint == "generativelanguage.googleapis.com"
        client._client = GenerativeServiceClient(transport=GenerativeServiceGrpcTransport(channel=channel()))
        response = client.generate_content(
            "gemini-2.5-flash", [{"role": "system", "content": "Translate to English."},
                                 {"role": "user", "content": "안녕하세요."}],
            temperature=0.3, max_output_tokens=128)
    assert response.text == "Hello there."
    assert response.finish_reason == "stop"
    assert seen["model"] == "models/gemini-2.5-flash"
    assert "안녕하세요." in seen["user"]
    assert seen["system"] == ["Translate to English."]
    assert seen["max_output_tokens"] == 128 and abs(seen["temperature"] - 0.3) < 1e-6
    assert seen["metadata"].get("x-goog-api-key") == "AIza-test-key"


def test_google_cloud_tts_route_round_trip(isolated, uac, monkeypatch):
    """unified_api_client._text_to_speech_google_cloud (Audio mode, Google Cloud voices) over gRPC."""
    _require_tier_b()
    from google.cloud import texttospeech
    from google.cloud.texttospeech_v1.services.text_to_speech.transports import TextToSpeechGrpcTransport

    for name in ("TTS_LANGUAGE_CODE", "TTS_VOICE", "TTS_AUDIO_FORMAT", "GOOGLE_TTS_API_KEY", "GOOGLE_API_KEY",
                 "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    seen = {}

    def synthesize(request, context):
        seen["text"] = request.input.text
        seen["voice"] = (request.voice.language_code, request.voice.name)
        seen["encoding"] = request.audio_config.audio_encoding
        return texttospeech.SynthesizeSpeechResponse(audio_content=b"ID3-fake-mp3-bytes")

    route = "/google.cloud.texttospeech.v1.TextToSpeech/SynthesizeSpeech"
    real_client = texttospeech.TextToSpeechClient
    with _grpc_loopback({route: (texttospeech.SynthesizeSpeechRequest, texttospeech.SynthesizeSpeechResponse,
                                 synthesize)}) as channel:
        monkeypatch.setattr(texttospeech, "TextToSpeechClient",
                            lambda: real_client(transport=TextToSpeechGrpcTransport(channel=channel())))
        owner = types.SimpleNamespace(api_key="", _tts_pool_key_active=False, _should_abort_retry=lambda: False,
                                      _is_force_stop_requested=lambda: False)
        out = isolated / "out" / "tts" / "line.mp3"
        result = uac.UnifiedClient._text_to_speech_google_cloud(owner, "Hello from Glossarion.", str(out),
                                                                voice="en-US-Neural2-J", audio_format="mp3")
    assert result == str(out) and out.read_bytes() == b"ID3-fake-mp3-bytes"
    assert seen == {"text": "Hello from Glossarion.", "voice": ("en-US", "en-US-Neural2-J"),
                    "encoding": texttospeech.AudioEncoding.MP3}


def test_google_cloud_vision_sdk_round_trip(isolated):
    """The google-cloud-vision calls manga_translator makes (document_text_detection and the ROI
    batch with Feature(type=...)) over gRPC; the manga service now reports the SDK backend."""
    _require_tier_b()
    from google.cloud import vision
    from google.cloud.vision_v1.services.image_annotator.transports import ImageAnnotatorGrpcTransport

    from glossarion_mobile.services import manga as manga_service

    seen = []

    def batch(request, context):
        seen.append([(r.image.content, [f.type_ for f in r.features], list(r.image_context.language_hints))
                     for r in request.requests])
        return vision.BatchAnnotateImagesResponse(responses=[
            vision.AnnotateImageResponse(full_text_annotation=vision.TextAnnotation(text="こんにちは"),
                                         text_annotations=[vision.EntityAnnotation(description="こんにちは")])
            for _ in request.requests])

    route = "/google.cloud.vision.v1.ImageAnnotator/BatchAnnotateImages"
    with _grpc_loopback({route: (vision.BatchAnnotateImagesRequest, vision.BatchAnnotateImagesResponse,
                                 batch)}) as channel:
        client = vision.ImageAnnotatorClient(transport=ImageAnnotatorGrpcTransport(channel=channel()))
        single = client.document_text_detection(image=vision.Image(content=b"page-bytes"),
                                                image_context=vision.ImageContext(language_hints=["ja"]))
        feature = vision.Feature(type=vision.Feature.Type.TEXT_DETECTION)  # manga_translator's ROI spelling
        requests = [vision.AnnotateImageRequest(image=vision.Image(content=b"roi-%d" % i), features=[feature],
                                                image_context=vision.ImageContext(language_hints=["ja", "ko"]))
                    for i in range(2)]
        batched = client.batch_annotate_images(requests=requests)
    assert single.full_text_annotation.text == "こんにちは"
    assert [r.text_annotations[0].description for r in batched.responses] == ["こんにちは", "こんにちは"]
    assert seen[0] == [(b"page-bytes", [vision.Feature.Type.DOCUMENT_TEXT_DETECTION], ["ja"])]
    assert seen[1] == [(b"roi-0", [vision.Feature.Type.TEXT_DETECTION], ["ja", "ko"]),
                       (b"roi-1", [vision.Feature.Type.TEXT_DETECTION], ["ja", "ko"])]
    assert manga_service.google_backend() == "SDK"
    assert manga_service.azure_docintel_backend() == "REST"  # Azure DI stays on the U8 REST client


class _LoopbackHttp:
    """Threaded HTTP server on 127.0.0.1: an OAuth token endpoint plus canned JSON answers."""

    def __init__(self, answers):
        self.answers = answers  # [(predicate(path), payload or callable(path, body) -> payload)]
        self.calls = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                owner.calls.append((self.path, self.headers.get("Authorization"), body))
                payload = None
                if self.path.startswith("/token"):
                    payload = {"access_token": "loopback-token", "expires_in": 3600, "token_type": "Bearer"}
                else:
                    for predicate, answer in owner.answers:
                        if predicate(self.path):
                            payload = answer(self.path, body) if callable(answer) else answer
                            break
                if payload is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                data = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(10)


def _service_account(tmp_path, token_uri, project_id="demo-project"):
    """A syntactically valid service-account file (fresh RSA key) whose token_uri is the loopback."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption()).decode("ascii")
    path = tmp_path / "service-account.json"
    path.write_text(json.dumps({
        "type": "service_account", "project_id": project_id, "private_key_id": "test-key-1", "private_key": pem,
        "client_email": f"glossarion-test@{project_id}.iam.gserviceaccount.com", "client_id": "1",
        "token_uri": token_uri}), encoding="utf-8")
    return path


def test_google_translate_paid_route_round_trip(isolated, uac, monkeypatch):
    """The paid google-translate route (_send_google_translate) through the translate_v2 REST client."""
    _require_tier_b()
    from google.auth.credentials import AnonymousCredentials

    def translate(path, body):
        request = json.loads(body.decode("utf-8"))
        return {"data": {"translations": [{"translatedText": "<p>Hello, world.</p>",
                                           "detectedSourceLanguage": request.get("source") or "ko"}
                                          for _ in request["q"]]}}

    assert uac._ensure_google_translate()  # deferred when the module was imported on the mobile runtime
    with _LoopbackHttp([(lambda p: p.startswith("/language/translate/v2"), translate)]) as server:
        real_client = uac.google_translate.Client
        monkeypatch.setattr(uac.google_translate, "Client", lambda: real_client(
            credentials=AnonymousCredentials(), client_options={"api_endpoint": server.base}))
        creds = isolated / "gcloud.json"
        creds.write_text("{}", encoding="utf-8")
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(creds))  # restored after the test
        monkeypatch.setenv("OUTPUT_LANGUAGE", "English")
        owner = types.SimpleNamespace(config={"google_cloud_credentials": str(creds)})
        response = uac.UnifiedClient._send_google_translate(
            owner, [{"role": "system", "content": "ignored by Google Translate"},
                    {"role": "user", "content": "<p>안녕하세요, 세계.</p>"}])
    assert "Hello, world." in response.content
    (path, _auth, body), = server.calls
    sent = json.loads(body.decode("utf-8"))
    assert path.startswith("/language/translate/v2")
    assert sent["q"] == ["<p>안녕하세요, 세계.</p>"] and sent["target"] == "en"
    assert sent["source"] == "ko" and sent["format"] == "html"


def _vertex_answers():
    claude = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude",
              "content": [{"type": "text", "text": "Hello from Vertex Claude"}], "stop_reason": "end_turn",
              "stop_sequence": None, "usage": {"input_tokens": 5, "output_tokens": 4}}
    gemini = {"candidates": [{"content": {"role": "model", "parts": [{"text": "Hello from Vertex Gemini"}]},
                              "finishReason": "STOP"}],
              "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 4, "totalTokenCount": 7}}
    return [(lambda p: ":rawPredict" in p, claude), (lambda p: ":generateContent" in p, gemini)]


def _vertex_env(monkeypatch, isolated, server, uac, *, mobile):
    sa = _service_account(isolated, server.base + "/token")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(sa))
    monkeypatch.setenv("VERTEX_AI_LOCATION", "us-east5")
    monkeypatch.setenv("ANTHROPIC_VERTEX_BASE_URL", server.base + "/v1")
    monkeypatch.setenv("GOOGLE_VERTEX_BASE_URL", server.base + "/")
    monkeypatch.setenv("ENABLE_STREAMING", "0")  # one JSON answer (UnifiedClient._streaming_enabled)
    if mobile:
        monkeypatch.setenv("GLOSSARION_MOBILE", "1")
        monkeypatch.setenv("GLOSSARION_NO_PROCESSES", "1")
    # google-cloud-aiplatform / vertexai are absent on mobile (protobuf<7): block them everywhere.
    monkeypatch.setitem(sys.modules, "google.cloud.aiplatform", None)
    monkeypatch.setitem(sys.modules, "vertexai", None)
    # Gemini safety-config payloads go to the test dir, never next to the module.
    payloads = isolated / "data" / "Payloads"
    payloads.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(uac, "_payloads_resolved_dir", str(payloads))
    return sa


@pytest.mark.parametrize("model, expected", [
    ("vertex/claude-sonnet-4@20250514", "Hello from Vertex Claude"),
    ("vertex/gemini-2.5-flash", "Hello from Vertex Gemini"),
])
def test_vertex_runs_over_rest_without_aiplatform_on_mobile(isolated, uac, monkeypatch, model, expected):
    """Vertex Model Garden on mobile: no google-cloud-aiplatform, REST + google-auth service-account
    tokens (Claude through anthropic.AnthropicVertex, Gemini through google-genai vertexai=True)."""
    pytest.importorskip("google.genai")
    pytest.importorskip("anthropic")
    with _LoopbackHttp(_vertex_answers()) as server:
        sa = _vertex_env(monkeypatch, isolated, server, uac, mobile=True)
        client = uac.UnifiedClient(api_key=str(sa), model=model)
        assert client.client_type == "vertex_model_garden"
        response = client._send_vertex_model_garden(
            [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
            temperature=0.2, max_tokens=64)
    assert response.content == expected
    paths = [path for path, _auth, _body in server.calls]
    assert paths[0] == "/token"  # google-auth exchanged the service-account JWT on the loopback
    model_name = model.split("/", 1)[1]
    if "claude" in model:
        assert paths[-1] == (f"/v1/projects/demo-project/locations/us-east5/publishers/anthropic/models/"
                             f"{model_name}:rawPredict")
    else:
        assert paths[-1].endswith(f"/projects/demo-project/locations/us-east5/publishers/google/models/"
                                  f"{model_name}:generateContent")
    assert server.calls[-1][1] == "Bearer loopback-token"


def test_vertex_without_aiplatform_keeps_the_desktop_error(isolated, uac, monkeypatch):
    """Desktop is unchanged: without google-cloud-aiplatform the route still fails before any request."""
    with _LoopbackHttp(_vertex_answers()) as server:
        sa = _vertex_env(monkeypatch, isolated, server, uac, mobile=False)
        client = uac.UnifiedClient(api_key=str(sa), model="vertex/claude-sonnet-4@20250514")
        with pytest.raises(uac.UnifiedClientError) as info:
            client._send_vertex_model_garden([{"role": "user", "content": "Hi"}], temperature=0.2, max_tokens=64)
    message = str(info.value)
    assert "Vertex AI Model Garden error" in message
    # the import error itself ("google.cloud.aiplatform", or "google.cloud" when no google-cloud-* is installed)
    assert "aiplatform" in message or "google.cloud" in message, message
    assert server.calls == []


# ============================================================================ device runtime note


def test_bootstrap_env_contract_selects_the_native_grpc_resolver(monkeypatch, tmp_path):
    """The Android grpcio build compiles c-ares in, which finds no DNS servers on Android 8+ without
    its JNI init, so gRPC (Gemini gRPC transport, Cloud TTS, Cloud Vision SDK) resolves through
    getaddrinfo: GRPC_DNS_RESOLVER=native in the bootstrap env contract (U9 Integrate)."""
    from glossarion_mobile import runtime_bootstrap as rb

    for var, sub in (("FLET_APP_STORAGE_DATA", "data"), ("FLET_APP_STORAGE_CACHE", "cache"),
                     ("FLET_APP_STORAGE_TEMP", "temp")):
        monkeypatch.setenv(var, str(tmp_path / sub))
    for platform in ("android", "ios"):
        env = rb.resolve_paths(APP_DIR, platform=platform).env_contract()
        assert env.get("GRPC_DNS_RESOLVER") == "native", platform


def test_bootstrap_silences_only_the_grpcio_pqc_warning():
    """google.api_core warns (FutureWarning) on every gRPC-based Google import while grpcio < 1.83 (no
    post-quantum TLS); 1.81.0 is the newest mobile build, so bootstrap filters exactly that warning."""
    import inspect

    from glossarion_mobile import runtime_bootstrap as rb

    assert "_quiet_known_warnings()" in inspect.getsource(rb.bootstrap)
    try:
        from google.api_core._python_package_support import PQC_GRPC_WARNING_TEMPLATE as template
    except ImportError:  # env without the Tier-B pins: the template as google-api-core 2.41.0 has it
        template = ("Package {consumer_package} depends on {dependency_package}, currently installed at version "
                    "{version_used_string}. grpcio < 1.83.0 does not support Post-Quantum Cryptography (PQC). "
                    "Support for non-PQC environments is deprecated.")
    message = template.format(consumer_package="google.api_core", dependency_package="grpcio",
                              version_used_string="1.81.0", consumer_import_package="google.api_core",
                              dependency_import_package="grpc", consumer_distribution_package="google-api-core",
                              dependency_distribution_package="grpcio", minimum_fully_supported_version="1.83.0",
                              recommendation="", version_used="1.81.0")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rb._quiet_known_warnings()
        rb._quiet_known_warnings()  # idempotent
        warnings.warn(message, FutureWarning)
        warnings.warn("Package google.api_core depends on protobuf, currently installed at version 5.0.", FutureWarning)
    assert [str(w.message) for w in caught] == [
        "Package google.api_core depends on protobuf, currently installed at version 5.0."]
