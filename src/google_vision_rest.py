"""Google Cloud Vision text detection over REST: the fallback for ``google-cloud-vision``.

Shared GUI-free core (Glossarion mobile rewrite, milestone U8). The desktop manga
pipeline talks to the ``google.cloud.vision`` SDK (gRPC). Phones may not have that SDK
(grpcio wheels), so ``manga_translator`` falls back to this module, and ONLY when
``from google.cloud import vision`` fails::

    try:
        from google.cloud import vision
    except ImportError:
        from google_vision_rest import vision

``vision`` here mirrors the slice of the SDK surface the manga code uses:
``ImageAnnotatorClient`` (``document_text_detection``, ``text_detection``,
``annotate_image``, ``batch_annotate_images``), ``Image``, ``ImageSource``,
``ImageContext``, ``TextDetectionParams``, ``Feature`` (+ ``Feature.Type``) and
``AnnotateImageRequest``. Responses are read-only views of the REST JSON with the SDK's
(proto3) attribute semantics, so the SDK-consuming code runs unchanged:

* snake_case attribute names (``full_text_annotation``, ``bounding_poly``) map to the
  REST camelCase keys (``fullTextAnnotation``, ``boundingPoly``);
* a field the server omitted reads as its proto default: ``0`` / ``0.0`` / ``''`` /
  ``False`` for scalars (REST omits zero vertex coordinates), ``[]`` for repeated fields
  and an empty, falsy message for message fields (``response.error.message == ''``);
* ``bool(message)`` is True when any field is set; an unknown field raises
  ``AttributeError`` like proto-plus.

Authentication, in this order:

1. an API key: ``ImageAnnotatorClient(client_options={"api_key": ...})``, the ``api_key``
   argument, or the ``GOOGLE_VISION_API_KEY`` environment variable (sent as ``?key=``);
2. ``credentials=`` (any ``google.auth`` credentials object);
3. the JSON file in ``GOOGLE_APPLICATION_CREDENTIALS`` (what the manga code sets from
   ``google_credentials_path``): a service account is signed into an OAuth token (JWT
   bearer flow) with ``google-auth``, which also reads authorized-user files.

HTTP errors raise :class:`VisionAPIError` (``"<code> <reason>: <message>"``); per-image
errors inside a 200 batch come back in ``response.error`` like the SDK. 5xx / 429 are
retried with backoff (``GOOGLE_VISION_REST_RETRIES``, default 2). The endpoint can be
redirected with ``client_options={"api_endpoint": ...}`` or ``GOOGLE_VISION_REST_ENDPOINT``
(tests point it at a local server).

Rules: Python 3.10 compatible; stdlib + ``requests`` (+ ``google-auth`` lazily for
service accounts); never import PySide6, translator_gui or dpi_setup.
"""

from __future__ import annotations

import base64
import enum
import json
import os
import random
import threading
import time
from typing import Any, Dict, Iterable, List, Optional

__all__ = [
    "AnnotateImageRequest",
    "AnnotateImageResponse",
    "BatchAnnotateImagesResponse",
    "DEFAULT_ENDPOINT",
    "Feature",
    "Image",
    "ImageAnnotatorClient",
    "ImageContext",
    "ImageSource",
    "SCOPES",
    "TextDetectionParams",
    "VisionAPIError",
    "vision",
]

#: REST host (``https://vision.googleapis.com/v1/images:annotate``).
DEFAULT_ENDPOINT = "https://vision.googleapis.com"
#: OAuth scopes for service-account tokens (the SDK's default scopes).
SCOPES = (
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/cloud-vision",
)
_DEFAULT_TIMEOUT = 120.0
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class VisionAPIError(Exception):
    """An HTTP-level failure of the Vision REST API (the SDK raises google.api_core errors)."""

    def __init__(self, message: str, *, status_code: Optional[int] = None, payload: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = status_code
        self.payload = payload


# ---------------------------------------------------------------------------
# Request types (constructed by the manga code exactly like the SDK's)
# ---------------------------------------------------------------------------


def _camel(name: str) -> str:
    """``full_text_annotation`` -> ``fullTextAnnotation`` (``type_`` -> ``type``)."""
    name = name.rstrip("_")
    head, *rest = name.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in rest)


def _to_json(value: Any) -> Any:
    """Request objects / dicts / enums -> REST JSON."""
    if isinstance(value, _RequestMessage):
        return value._json()
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, (bytes, bytearray)):
        return base64.b64encode(bytes(value)).decode("ascii")
    if isinstance(value, dict):
        return {_camel(str(k)): _to_json(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_to_json(v) for v in value]
    return value


class _RequestMessage:
    """Keyword-constructed request message; unset (None) fields are not sent."""

    _FIELDS: tuple = ()
    _ALIASES: Dict[str, str] = {}

    def __init__(self, mapping: Optional[dict] = None, **kwargs):
        values = dict(mapping or {})
        values.update(kwargs)
        object.__setattr__(self, "_values", {})
        for key, value in values.items():
            setattr(self, key, value)

    def __setattr__(self, key, value):
        key = self._ALIASES.get(key, key)
        if key not in self._FIELDS:
            raise AttributeError(f"Unknown field for {type(self).__name__}: {key}")
        self._values[key] = value

    def __getattr__(self, key):
        key = type(self)._ALIASES.get(key, key)
        values = object.__getattribute__(self, "_values")
        if key in values:
            return values[key]
        if key in type(self)._FIELDS:
            return None
        raise AttributeError(f"Unknown field for {type(self).__name__}: {key}")

    def _json(self) -> dict:
        return {_camel(k): _to_json(v) for k, v in self._values.items() if v is not None}

    def __repr__(self):
        return f"{type(self).__name__}({self._values!r})"


class ImageSource(_RequestMessage):
    _FIELDS = ("gcs_image_uri", "image_uri")


class Image(_RequestMessage):
    """``Image(content=<bytes>)`` (base64 on the wire) or ``Image(source=ImageSource(...))``."""

    _FIELDS = ("content", "source")


class TextDetectionParams(_RequestMessage):
    _FIELDS = ("enable_text_detection_confidence_score", "advanced_ocr_options")


class ImageContext(_RequestMessage):
    _FIELDS = ("language_hints", "text_detection_params", "lat_long_rect", "crop_hints_params",
               "web_detection_params", "product_search_params")


class Feature(_RequestMessage):
    """``Feature(type=Feature.Type.DOCUMENT_TEXT_DETECTION)`` (``type_`` accepted too, like the SDK)."""

    class Type(enum.IntEnum):
        TYPE_UNSPECIFIED = 0
        FACE_DETECTION = 1
        LANDMARK_DETECTION = 2
        LOGO_DETECTION = 3
        LABEL_DETECTION = 4
        TEXT_DETECTION = 5
        DOCUMENT_TEXT_DETECTION = 11
        SAFE_SEARCH_DETECTION = 6
        IMAGE_PROPERTIES = 7
        CROP_HINTS = 9
        WEB_DETECTION = 10
        PRODUCT_SEARCH = 12
        OBJECT_LOCALIZATION = 19

    _FIELDS = ("type", "max_results", "model")
    _ALIASES = {"type_": "type"}


class AnnotateImageRequest(_RequestMessage):
    _FIELDS = ("image", "features", "image_context")


# ---------------------------------------------------------------------------
# Response views (proto3 attribute semantics over the REST JSON)
# ---------------------------------------------------------------------------

_STR, _INT, _FLOAT, _BOOL, _ENUM = "str", "int", "float", "bool", "enum"
_SCALAR_DEFAULTS = {_STR: "", _INT: 0, _FLOAT: 0.0, _BOOL: False, _ENUM: 0}


def _msg(type_name):
    return ("msg", type_name)


def _rep(type_name):
    return ("rep", type_name)


#: message type -> {snake_case field: kind}. Only the text-detection slice the manga code reads;
#: fields outside it still read through when the server sent them.
_SCHEMA: Dict[str, Dict[str, Any]] = {
    "BatchAnnotateImagesResponse": {"responses": _rep("AnnotateImageResponse")},
    "AnnotateImageResponse": {
        "text_annotations": _rep("EntityAnnotation"),
        "full_text_annotation": _msg("TextAnnotation"),
        "error": _msg("Status"),
        "context": _msg("ImageAnnotationContext"),
    },
    "ImageAnnotationContext": {"uri": _STR, "page_number": _INT},
    "Status": {"code": _INT, "message": _STR, "details": ("rep_scalar", None)},
    "EntityAnnotation": {
        "mid": _STR, "locale": _STR, "description": _STR, "score": _FLOAT, "confidence": _FLOAT,
        "topicality": _FLOAT, "bounding_poly": _msg("BoundingPoly"), "locations": ("rep_scalar", None),
        "properties": ("rep_scalar", None),
    },
    "BoundingPoly": {"vertices": _rep("Vertex"), "normalized_vertices": _rep("NormalizedVertex")},
    "Vertex": {"x": _INT, "y": _INT},
    "NormalizedVertex": {"x": _FLOAT, "y": _FLOAT},
    "TextAnnotation": {"pages": _rep("Page"), "text": _STR},
    "Page": {"property": _msg("TextProperty"), "width": _INT, "height": _INT,
             "blocks": _rep("Block"), "confidence": _FLOAT},
    "Block": {"property": _msg("TextProperty"), "bounding_box": _msg("BoundingPoly"),
              "paragraphs": _rep("Paragraph"), "block_type": _ENUM, "confidence": _FLOAT},
    "Paragraph": {"property": _msg("TextProperty"), "bounding_box": _msg("BoundingPoly"),
                  "words": _rep("Word"), "confidence": _FLOAT},
    "Word": {"property": _msg("TextProperty"), "bounding_box": _msg("BoundingPoly"),
             "symbols": _rep("Symbol"), "confidence": _FLOAT},
    "Symbol": {"property": _msg("TextProperty"), "bounding_box": _msg("BoundingPoly"),
               "text": _STR, "confidence": _FLOAT},
    "TextProperty": {"detected_languages": _rep("DetectedLanguage"), "detected_break": _msg("DetectedBreak")},
    "DetectedLanguage": {"language_code": _STR, "confidence": _FLOAT},
    "DetectedBreak": {"type_": _ENUM, "is_prefix": _BOOL},
}


def _generic(value):
    if isinstance(value, dict):
        return _Message(value, "")
    if isinstance(value, list):
        return [_generic(v) for v in value]
    return value


class _Message:
    """Read-only proto3-style view of one REST JSON object."""

    __slots__ = ("_data", "_type")

    def __init__(self, data: Optional[dict], type_name: str):
        object.__setattr__(self, "_data", data if isinstance(data, dict) else {})
        object.__setattr__(self, "_type", type_name)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        data = object.__getattribute__(self, "_data")
        type_name = object.__getattribute__(self, "_type")
        fields = _SCHEMA.get(type_name, {})
        key = _camel(name)
        if name in fields:
            kind = fields[name]
            value = data.get(key)
            if isinstance(kind, tuple):
                tag, child = kind
                if tag == "msg":
                    return _Message(value, child)
                if tag == "rep":
                    return [_Message(v, child) for v in (value or [])]
                return [_generic(v) for v in (value or [])]
            if value is None:
                return _SCALAR_DEFAULTS[kind]
            if kind == _INT:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return value
            if kind == _FLOAT:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return value
            return value
        if key in data:
            return _generic(data[key])
        raise AttributeError(f"Unknown field for {type_name or 'Message'}: {name}")

    def __setattr__(self, name, value):
        raise AttributeError(f"{type(self).__name__} is read-only")

    def __bool__(self):
        return bool(object.__getattribute__(self, "_data"))

    def __contains__(self, name):
        return _camel(name) in object.__getattribute__(self, "_data")

    def __eq__(self, other):
        if isinstance(other, _Message):
            return self._data == other._data
        return NotImplemented

    def __repr__(self):
        return f"{self._type or 'Message'}({json.dumps(self._data, ensure_ascii=False)[:200]})"

    def to_dict(self) -> dict:
        """The REST JSON object (camelCase keys, defaults omitted)."""
        return json.loads(json.dumps(self._data))


def AnnotateImageResponse(data: Optional[dict] = None) -> _Message:  # noqa: N802 (SDK name)
    return _Message(data or {}, "AnnotateImageResponse")


def BatchAnnotateImagesResponse(data: Optional[dict] = None) -> _Message:  # noqa: N802 (SDK name)
    return _Message(data or {}, "BatchAnnotateImagesResponse")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def _option(options, name):
    if options is None:
        return None
    if isinstance(options, dict):
        return options.get(name)
    return getattr(options, name, None)


def _reason(status_code: int) -> str:
    try:
        from http import HTTPStatus

        return HTTPStatus(status_code).phrase
    except Exception:
        return "Error"


class ImageAnnotatorClient:
    """REST stand-in for ``google.cloud.vision.ImageAnnotatorClient`` (thread-safe)."""

    def __init__(self, *, credentials=None, client_options=None, api_key: Optional[str] = None,
                 credentials_path: Optional[str] = None, endpoint: Optional[str] = None,
                 timeout: Optional[float] = None, transport=None, session=None, **_ignored):
        self._api_key = (api_key or _option(client_options, "api_key")
                         or os.environ.get("GOOGLE_VISION_API_KEY", "").strip() or None)
        self._credentials = credentials
        self._credentials_path = credentials_path
        if self._api_key is None and self._credentials is None and not self._credentials_path:
            self._credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip() or None
        base = (endpoint or _option(client_options, "api_endpoint")
                or os.environ.get("GOOGLE_VISION_REST_ENDPOINT", "").strip() or DEFAULT_ENDPOINT)
        if "://" not in base:
            base = "https://" + base
        self._url = base.rstrip("/") + "/v1/images:annotate"
        try:
            self._timeout = float(timeout if timeout is not None
                                  else os.environ.get("GOOGLE_VISION_REST_TIMEOUT", _DEFAULT_TIMEOUT))
        except (TypeError, ValueError):
            self._timeout = _DEFAULT_TIMEOUT
        try:
            self._retries = max(0, int(os.environ.get("GOOGLE_VISION_REST_RETRIES", "2")))
        except ValueError:
            self._retries = 2
        self._session = session
        self._local = threading.local()
        self._auth_lock = threading.Lock()
        if self._api_key is None and self._credentials is None and not self._credentials_path:
            raise VisionAPIError(
                "Google Vision REST needs credentials: set GOOGLE_APPLICATION_CREDENTIALS to a "
                "service-account JSON file or provide an API key")

    # ---- auth ------------------------------------------------------------------------------
    def _load_credentials(self):
        if self._credentials is None:
            if not os.path.exists(self._credentials_path):
                raise VisionAPIError(f"Google credentials file not found: {self._credentials_path}")
            import google.auth  # google-auth (pinned on mobile)

            creds, _project = google.auth.load_credentials_from_file(self._credentials_path, scopes=list(SCOPES))
            self._credentials = creds
        return self._credentials

    def _bearer_token(self) -> str:
        with self._auth_lock:
            creds = self._load_credentials()
            if not getattr(creds, "valid", False) or not getattr(creds, "token", None):
                import google.auth.transport.requests as google_requests

                creds.refresh(google_requests.Request())
            return creds.token

    def _http(self):
        if self._session is not None:
            return self._session
        session = getattr(self._local, "session", None)
        if session is None:
            import requests

            session = requests.Session()
            self._local.session = session
        return session

    # ---- transport -------------------------------------------------------------------------
    def _post(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json; charset=utf-8"}
        params = {}
        if self._api_key:
            params["key"] = self._api_key
        attempt = 0
        while True:
            if not self._api_key:
                headers["Authorization"] = f"Bearer {self._bearer_token()}"
            response = self._http().post(self._url, params=params or None, data=json.dumps(payload),
                                         headers=headers, timeout=self._timeout)
            status = int(getattr(response, "status_code", 0) or 0)
            if status == 200:
                try:
                    return response.json()
                except ValueError as exc:
                    raise VisionAPIError(f"Invalid JSON from the Vision API: {exc}", status_code=status)
            if status in _RETRY_STATUSES and attempt < self._retries:
                attempt += 1
                time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))) + random.random() * 0.25)
                continue
            if status == 401 and self._credentials is not None and not self._api_key and attempt == 0:
                attempt += 1   # stale token: refresh once
                with self._auth_lock:
                    try:
                        import google.auth.transport.requests as google_requests

                        self._credentials.refresh(google_requests.Request())
                    except Exception:
                        pass
                continue
            message = ""
            body = None
            try:
                body = response.json()
                message = ((body or {}).get("error") or {}).get("message", "")
            except ValueError:
                message = (getattr(response, "text", "") or "")[:500]
            raise VisionAPIError(f"{status} {_reason(status)}: {message}".rstrip(": "),
                                 status_code=status, payload=body)

    @staticmethod
    def _request_json(request) -> dict:
        if isinstance(request, AnnotateImageRequest):
            return request._json()
        if isinstance(request, dict):
            return _to_json(request)
        raise TypeError(f"Unsupported request type: {type(request).__name__}")

    # ---- SDK methods -----------------------------------------------------------------------
    def batch_annotate_images(self, request=None, *, requests: Optional[Iterable] = None,
                              retry=None, timeout=None, metadata=(), **_ignored):
        if request is not None and requests is None:
            requests = request.get("requests") if isinstance(request, dict) else getattr(request, "requests", None)
        payload = {"requests": [self._request_json(r) for r in (requests or [])]}
        return BatchAnnotateImagesResponse(self._post(payload))

    def annotate_image(self, request, *, retry=None, timeout=None, metadata=(), **_ignored):
        batch = self.batch_annotate_images(requests=[request])
        responses = batch.responses
        return responses[0] if responses else AnnotateImageResponse({})

    def _detect(self, feature_type, image, max_results, kwargs):
        feature = {"type": feature_type.name}
        if max_results is not None:
            feature["maxResults"] = int(max_results)
        request = {"image": _to_json(image), "features": [feature]}
        image_context = kwargs.get("image_context")
        if image_context is not None:
            request["imageContext"] = _to_json(image_context)
        return self.annotate_image(request)

    def document_text_detection(self, image, *, max_results=None, retry=None, timeout=None,
                                metadata=(), **kwargs):
        return self._detect(Feature.Type.DOCUMENT_TEXT_DETECTION, image, max_results, kwargs)

    def text_detection(self, image, *, max_results=None, retry=None, timeout=None, metadata=(), **kwargs):
        return self._detect(Feature.Type.TEXT_DETECTION, image, max_results, kwargs)


class _VisionNamespace:
    """``from google_vision_rest import vision`` -> the SDK-shaped names above."""

    ImageAnnotatorClient = ImageAnnotatorClient
    Image = Image
    ImageSource = ImageSource
    ImageContext = ImageContext
    TextDetectionParams = TextDetectionParams
    Feature = Feature
    AnnotateImageRequest = AnnotateImageRequest
    AnnotateImageResponse = staticmethod(AnnotateImageResponse)
    BatchAnnotateImagesResponse = staticmethod(BatchAnnotateImagesResponse)
    VisionAPIError = VisionAPIError
    REST_FALLBACK = True

    def __repr__(self):
        return "<google_vision_rest.vision (REST fallback for google.cloud.vision)>"


vision = _VisionNamespace()
