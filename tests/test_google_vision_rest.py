"""google_vision_rest: Google Cloud Vision text detection over REST (Glossarion mobile rewrite, U8).

The manga pipeline talks to the ``google.cloud.vision`` SDK; phones may not have it, so
``manga_translator`` falls back to ``google_vision_rest.vision`` ONLY when
``from google.cloud import vision`` fails (module level, ``_ensure_google_client`` and the ROI
batching). Checked here against a local HTTP server standing in for vision.googleapis.com:

* the request JSON the SDK would send (base64 content, feature types, image context);
* API-key auth (``?key=``) and service-account auth (a real JWT-bearer token exchange through
  ``google-auth`` against the local token endpoint, then ``Authorization: Bearer``);
* the proto3 response semantics the manga code relies on (defaults for omitted fields, falsy
  empty messages, unknown fields raise);
* batching, per-image errors, retries on 5xx, HTTP errors, missing credentials;
* manga_translator with the SDK import blocked uses the REST client end to end (ROI batching
  and the client re-init), and keeps the SDK when it is importable.

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_google_vision_rest.py
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import google_vision_rest as gvr  # noqa: E402
from google_vision_rest import vision  # noqa: E402



@pytest.fixture(autouse=True)
def _no_http_logs_in_src():
    """unified_api_client (imported by an earlier test file in the same run) patches requests to log
    every call into src/http_requests: switch that log off for these local-server tests (own
    MonkeyPatch, so a test's ``monkeypatch.undo()`` cannot switch it back on)."""
    patch = pytest.MonkeyPatch()
    try:
        patch.setenv("GLOSSARION_HTTP_LOG", "0")
        http_logger = sys.modules.get("http_logger")
        if http_logger is not None and getattr(http_logger, "_log_folder", None) is not None:
            patch.setattr(http_logger, "_log_folder", None)  # _save_http_log returns early
        yield
    finally:
        patch.undo()

PAGE_RESPONSE = {
    "textAnnotations": [
        {"locale": "ja", "description": "こんにちは\n世界",
         "boundingPoly": {"vertices": [{"x": 5}, {"x": 40, "y": 0}, {"x": 40, "y": 20}, {"y": 20}]}},
        {"description": "こんにちは", "boundingPoly": {"vertices": [{"x": 5, "y": 1}, {"x": 30, "y": 1}]}},
    ],
    "fullTextAnnotation": {
        "text": "こんにちは\n世界\n",
        "pages": [{
            "width": 64, "height": 32,
            "blocks": [{
                "boundingBox": {"vertices": [{"x": 5}, {"x": 40}, {"x": 40, "y": 20}, {"y": 20}]},
                "blockType": "TEXT",
                "paragraphs": [{"words": [
                    {"confidence": 0.98, "symbols": [{"text": "こ"}, {"text": "ん"}]},
                    {"symbols": [{"text": "世", "property": {"detectedBreak": {"type": "LINE_BREAK"}}}]},
                ]}],
            }],
        }],
    },
}


class FakeVisionServer:
    """vision.googleapis.com + an OAuth token endpoint on localhost, with scripted replies."""

    def __init__(self):
        self.requests = []
        self.token_requests = []
        self.script = []          # list of (status, body) for the next annotate calls
        self.default = (200, {"responses": [PAGE_RESPONSE]})
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
                url = urlparse(self.path)
                if url.path == "/token":
                    server.token_requests.append(parse_qs(body.decode("utf-8")))
                    return self._reply(200, {"access_token": "ya29.test-token", "expires_in": 3600,
                                             "token_type": "Bearer"})
                server.requests.append({
                    "path": url.path, "query": parse_qs(url.query),
                    "auth": self.headers.get("Authorization"), "json": json.loads(body.decode("utf-8")),
                })
                status, reply = server.script.pop(0) if server.script else server.default
                if callable(reply):
                    reply = reply(server.requests[-1]["json"])
                return self._reply(status, reply)

            def _reply(self, status, payload):
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    srv = FakeVisionServer()
    yield srv
    srv.close()


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch):
    for key in ("GOOGLE_VISION_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_VISION_REST_ENDPOINT",
                "GOOGLE_VISION_REST_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GOOGLE_VISION_REST_RETRIES", "2")


def _client(server, **kwargs):
    kwargs.setdefault("client_options", {"api_key": "k-123"})
    return vision.ImageAnnotatorClient(endpoint=server.url, **kwargs)


# ---------------------------------------------------------------------------
# requests
# ---------------------------------------------------------------------------


def test_document_text_detection_sends_the_sdk_request_with_an_api_key(server):
    client = _client(server)
    image = vision.Image(content=b"\x89PNG-bytes")
    context = vision.ImageContext(language_hints=["ja", "ko"])
    context.text_detection_params = vision.TextDetectionParams(enable_text_detection_confidence_score=True)
    response = client.document_text_detection(image=image, image_context=context)
    sent = server.requests[-1]
    assert sent["path"] == "/v1/images:annotate"
    assert sent["query"] == {"key": ["k-123"]} and sent["auth"] is None
    assert sent["json"] == {"requests": [{
        "image": {"content": base64.b64encode(b"\x89PNG-bytes").decode("ascii")},
        "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
        "imageContext": {"languageHints": ["ja", "ko"],
                         "textDetectionParams": {"enableTextDetectionConfidenceScore": True}},
    }]}
    assert response.full_text_annotation.text == "こんにちは\n世界\n"


def test_text_detection_and_batches_use_the_requested_features(server):
    client = _client(server)
    client.text_detection(image=vision.Image(content=b"a"), max_results=3)
    assert server.requests[-1]["json"]["requests"][0]["features"] == [{"type": "TEXT_DETECTION", "maxResults": 3}]
    server.default = (200, {"responses": [PAGE_RESPONSE, {"error": {"code": 3, "message": "Bad image data."}}]})
    feature = vision.Feature(type=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
    requests = [vision.AnnotateImageRequest(image=vision.Image(content=data), features=[feature],
                                            image_context=vision.ImageContext(language_hints=["zh"]))
                for data in (b"one", b"two")]
    batch = client.batch_annotate_images(requests=requests)
    sent = server.requests[-1]["json"]["requests"]
    assert [r["features"] for r in sent] == [[{"type": "DOCUMENT_TEXT_DETECTION"}]] * 2
    assert [r["image"]["content"] for r in sent] == [base64.b64encode(b).decode() for b in (b"one", b"two")]
    first, second = batch.responses
    assert first.full_text_annotation.text and not first.error.message
    assert second.error.message == "Bad image data." and second.error.code == 3
    assert not second.full_text_annotation and second.text_annotations == []
    # dict requests and the type_ alias are accepted like the SDK
    client.batch_annotate_images(requests=[{"image": {"content": b"x"}, "features": [{"type_": "TEXT_DETECTION"}]}])
    assert server.requests[-1]["json"]["requests"][0]["features"] == [{"type": "TEXT_DETECTION"}]
    assert vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION).type is vision.Feature.Type.TEXT_DETECTION


# ---------------------------------------------------------------------------
# response semantics
# ---------------------------------------------------------------------------


def test_responses_follow_proto3_defaults(server):
    response = _client(server).document_text_detection(image=vision.Image(content=b"x"))
    assert response.error.message == "" and not response.error
    page = response.full_text_annotation.pages[0]
    block = page.blocks[0]
    assert [(v.x, v.y) for v in block.bounding_box.vertices] == [(5, 0), (40, 0), (40, 20), (0, 20)]
    words = block.paragraphs[0].words
    assert getattr(words[0], "confidence", 0.0) == pytest.approx(0.98)
    assert getattr(words[1], "confidence", 0.0) == 0.0
    assert ["".join(s.text for s in w.symbols) for w in words] == ["こん", "世"]
    assert words[1].symbols[0].property.detected_break.type_ == "LINE_BREAK"
    assert response.text_annotations[0].description.startswith("こんにちは")
    assert [(v.x, v.y) for v in response.text_annotations[1].bounding_poly.vertices] == [(5, 1), (30, 1)]
    assert response.text_annotations[1].bounding_poly.normalized_vertices == []
    with pytest.raises(AttributeError):
        response.not_a_field
    with pytest.raises(AttributeError):
        response.error = None
    assert "fullTextAnnotation" in response.to_dict()


# ---------------------------------------------------------------------------
# errors, retries, auth
# ---------------------------------------------------------------------------


def test_5xx_is_retried_and_4xx_raises(server, monkeypatch):
    monkeypatch.setattr(gvr.time, "sleep", lambda _s: None)
    server.script = [(503, {"error": {"message": "busy"}}), (200, {"responses": [PAGE_RESPONSE]})]
    response = _client(server).document_text_detection(image=vision.Image(content=b"x"))
    assert response.full_text_annotation.text and len(server.requests) == 2
    server.script = [(400, {"error": {"code": 400, "message": "Request must specify image and features."}})]
    with pytest.raises(gvr.VisionAPIError) as info:
        _client(server).text_detection(image=vision.Image(content=b"x"))
    assert info.value.status_code == 400
    assert "Bad Request" in str(info.value) and "must specify image" in str(info.value)
    server.script = [(500, {})] * 3
    with pytest.raises(gvr.VisionAPIError, match="500"):
        _client(server).text_detection(image=vision.Image(content=b"x"))


def test_missing_credentials_fail_like_the_sdk(server, tmp_path):
    with pytest.raises(gvr.VisionAPIError, match="credentials"):
        vision.ImageAnnotatorClient(endpoint=server.url)
    client = vision.ImageAnnotatorClient(endpoint=server.url, credentials_path=str(tmp_path / "missing.json"))
    with pytest.raises(gvr.VisionAPIError, match="not found"):
        client.text_detection(image=vision.Image(content=b"x"))


def test_api_key_from_the_environment(server, monkeypatch):
    monkeypatch.setenv("GOOGLE_VISION_API_KEY", "env-key")
    monkeypatch.setenv("GOOGLE_VISION_REST_ENDPOINT", server.url)
    vision.ImageAnnotatorClient().text_detection(image=vision.Image(content=b"x"))
    assert server.requests[-1]["query"] == {"key": ["env-key"]}


def _service_account_file(tmp_path, token_uri):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption()).decode("ascii")
    info = {
        "type": "service_account", "project_id": "glossarion-test", "private_key_id": "kid-1",
        "private_key": pem, "client_email": "manga@glossarion-test.iam.gserviceaccount.com",
        "client_id": "123", "token_uri": token_uri,
    }
    path = tmp_path / "service-account.json"
    path.write_text(json.dumps(info), encoding="utf-8")
    return path


def test_service_account_signs_a_jwt_and_sends_a_bearer_token(server, tmp_path, monkeypatch):
    pytest.importorskip("google.auth")
    pytest.importorskip("cryptography")
    creds = _service_account_file(tmp_path, server.url + "/token")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(creds))
    client = vision.ImageAnnotatorClient(endpoint=server.url)
    response = client.document_text_detection(image=vision.Image(content=b"x"))
    assert response.full_text_annotation.text
    assert server.requests[-1]["auth"] == "Bearer ya29.test-token" and "key" not in server.requests[-1]["query"]
    grant = server.token_requests[-1]
    assert grant["grant_type"] == ["urn:ietf:params:oauth:grant-type:jwt-bearer"]
    header, claims, _sig = grant["assertion"][0].split(".")
    claims = json.loads(base64.urlsafe_b64decode(claims + "=" * (-len(claims) % 4)))
    assert claims["iss"] == "manga@glossarion-test.iam.gserviceaccount.com"
    assert "cloud-vision" in claims["scope"] and claims["aud"].endswith("/token")  # google-auth's audience
    client.text_detection(image=vision.Image(content=b"y"))
    assert len(server.token_requests) == 1, "the token is cached while valid"


# ---------------------------------------------------------------------------
# manga_translator: REST only when the SDK import fails
# ---------------------------------------------------------------------------

_MANGA_PROBE = textwrap.dedent(r"""
    import json, os, sys, threading, types
    block = sys.argv[1] == 'block'
    if block:
        sys.modules['google.cloud.vision'] = None
    import manga_translator, google_vision_rest
    out = {'rest': manga_translator.vision is google_vision_rest.vision,
           'available': manga_translator.GOOGLE_CLOUD_VISION_AVAILABLE}
    if block:
        mt = object.__new__(manga_translator.MangaTranslator)
        mt.ocr_config = {'google_credentials_path': os.environ['PROBE_CREDS']}
        mt.vision_client = None
        logs = []
        mt._log = lambda message, level='info': logs.append((level, message))
        mt._ensure_google_client()
        out['client'] = type(mt.vision_client).__module__
        out['creds_env'] = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') == os.environ['PROBE_CREDS']
        mt.main_gui = types.SimpleNamespace(config={'manga_settings': {'ocr': {}}})
        mt._cache_lock = threading.Lock()
        mt.ocr_roi_cache = {}
        rois = [{'bbox': (1, 2, 30, 40), 'bytes': b'roi-1', 'type': 'text_bubble'},
                {'bbox': (50, 2, 30, 40), 'bytes': b'roi-2', 'type': 'free_text'}]
        regions = mt._google_ocr_rois_batched(rois, {'ocr_request_delay_ms': 0, 'min_text_length': 1,
                                                     'exclude_english_text': False, 'language_hints': ['ja']},
                                              batch_size=8, max_concurrency=1, page_hash='h')
        out['regions'] = [(r.text, list(r.bounding_box), getattr(r, 'bubble_type', None)) for r in regions]
        out['errors'] = [m for level, m in logs if level == 'error']
    print('PROBE' + json.dumps(out, ensure_ascii=False))
""")


def _run_probe(mode, env_extra):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", **env_extra)
    env["PYTHONPATH"] = os.pathsep.join([str(SRC)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])
    proc = subprocess.run([sys.executable, "-c", _MANGA_PROBE, mode], cwd=str(SRC), env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=600)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("PROBE")), None)
    assert proc.returncode == 0 and line, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads(line[len("PROBE"):])


def test_manga_translator_uses_rest_only_without_the_sdk(server, tmp_path):
    creds = _service_account_file(tmp_path, server.url + "/token") if _has_google_auth() else tmp_path / "c.json"
    if not creds.exists():
        creds.write_text("{}", encoding="utf-8")
    roi_text = {"responses": [
        {"fullTextAnnotation": {"text": "ROI one"}},
        {"textAnnotations": [{"description": "ROI two"}]},
    ]}
    server.default = (200, roi_text)
    env = {"PROBE_CREDS": str(creds), "GOOGLE_VISION_REST_ENDPOINT": server.url}
    if not _has_google_auth():
        env["GOOGLE_VISION_API_KEY"] = "probe-key"
    blocked = _run_probe("block", env)
    assert blocked["rest"] is True and blocked["available"] is True
    assert blocked["client"] == "google_vision_rest" and blocked["creds_env"] and not blocked["errors"]
    assert sorted(blocked["regions"]) == [["ROI one", [1, 2, 30, 40], "text_bubble"],
                                          ["ROI two", [50, 2, 30, 40], "free_text"]]
    sent = server.requests[-1]["json"]["requests"]
    assert [r["features"] for r in sent] == [[{"type": "DOCUMENT_TEXT_DETECTION"}]] * 2
    assert sent[0]["imageContext"] == {"languageHints": ["ja"]}
    try:
        import google.cloud.vision  # noqa: F401
    except Exception:
        return
    with_sdk = _run_probe("sdk", {})
    assert with_sdk["rest"] is False and with_sdk["available"] is True


def _has_google_auth():
    try:
        import google.auth  # noqa: F401
        import cryptography  # noqa: F401
        return True
    except Exception:
        return False


def test_no_qt_or_heavy_imports():
    code = ("import sys\n"
            "for n in ('PySide6', 'requests'):\n    sys.modules.setdefault(n + '_probe', None)\n"
            "import google_vision_rest\n"
            "assert 'google.auth' not in sys.modules and 'PySide6' not in sys.modules\nprint('ok')\n")
    env = dict(os.environ, PYTHONPATH=str(SRC))
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(SRC), env=env, capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode == 0 and proc.stdout.strip() == "ok", proc.stderr
