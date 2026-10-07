"""Azure AI Document Intelligence ("prebuilt-read") over REST: the mobile fallback for the SDK.

Shared GUI-free core (Glossarion mobile rewrite, milestone U8). The desktop manga OCR
provider ``ocr_manager.AzureDocumentIntelligenceProvider`` talks to the
``azure-ai-formrecognizer`` SDK (``DocumentAnalysisClient``), and manga_translator's
empty-block fallback OCR calls ``provider.client.begin_analyze_document(...)`` directly.
Glossarion Mobile does not ship that SDK, so there (and only there: desktop keeps its
"SDK not installed" path) the provider builds :class:`DocumentAnalysisRestClient`, which
mirrors the slice of the SDK surface the manga code uses::

    client = DocumentAnalysisRestClient(endpoint, key)
    poller = client.begin_analyze_document("prebuilt-read", document=image_bytes, locale="ja")
    result = poller.result()
    for page in result.pages:
        for line in page.lines:
            line.content, [(point.x, point.y) for point in line.polygon]

Like the SDK's ``DocumentLine``, a line has no ``confidence`` attribute (the provider's
``getattr(line, 'confidence', 0.9)`` keeps its 0.9), and ``polygon`` is a list of points
with ``.x`` / ``.y``.

REST protocol (Document Intelligence v4 GA, the API azure-ai-documentintelligence uses):
``POST {endpoint}/documentintelligence/documentModels/{model}:analyze?api-version=2024-11-30
[&locale=..]`` with the ``Ocp-Apim-Subscription-Key`` header and the image bytes as the
body answers ``202`` with an ``Operation-Location`` to poll (``Retry-After`` honoured)
until ``status`` is ``succeeded`` (``analyzeResult``) or ``failed`` (``error``). A
resource that does not know that route (HTTP 404) is retried once on the Form Recognizer
route the formrecognizer SDK uses (``/formrecognizer/...?api-version=2023-07-31``); both
return the same ``analyzeResult`` shape. 429 / 5xx are retried with backoff.

Rules: Python 3.10 compatible; stdlib + ``requests``; never import PySide6, translator_gui
or dpi_setup.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote

__all__ = [
    "API_VERSION",
    "LEGACY_API_VERSION",
    "AnalyzeResult",
    "DocumentAnalysisRestClient",
    "DocumentIntelligenceError",
    "DocumentLine",
    "DocumentPage",
    "DocumentWord",
    "Point",
    "parse_analyze_result",
]

API_VERSION = "2024-11-30"
LEGACY_API_VERSION = "2023-07-31"
_ROUTES = (
    ("documentintelligence", API_VERSION),
    ("formrecognizer", LEGACY_API_VERSION),
)
_KEY_HEADER = "Ocp-Apim-Subscription-Key"


class DocumentIntelligenceError(Exception):
    """An HTTP error or a failed analyze operation (``status_code`` 0 for operation errors)."""

    def __init__(self, message: str, status_code: int = 0, code: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class DocumentWord:
    content: str
    polygon: List[Point] = field(default_factory=list)
    confidence: float = 0.0


@dataclass(frozen=True)
class DocumentLine:
    content: str
    polygon: List[Point] = field(default_factory=list)


@dataclass(frozen=True)
class DocumentPage:
    page_number: int
    width: Optional[float] = None
    height: Optional[float] = None
    unit: Optional[str] = None
    angle: Optional[float] = None
    lines: List[DocumentLine] = field(default_factory=list)
    words: List[DocumentWord] = field(default_factory=list)


@dataclass(frozen=True)
class AnalyzeResult:
    api_version: str
    model_id: str
    content: str = ""
    pages: List[DocumentPage] = field(default_factory=list)


def _points(flat: Any) -> List[Point]:
    """REST ``polygon`` [x1, y1, x2, y2, ...] -> [Point, ...] (the SDK's shape)."""
    values = list(flat or ())
    return [Point(float(values[i]), float(values[i + 1])) for i in range(0, len(values) - 1, 2)]


def parse_analyze_result(data: Dict[str, Any]) -> AnalyzeResult:
    """The ``analyzeResult`` JSON object -> AnalyzeResult."""
    data = data or {}
    pages = []
    for page in data.get("pages") or ():
        pages.append(DocumentPage(
            page_number=int(page.get("pageNumber") or len(pages) + 1),
            width=page.get("width"),
            height=page.get("height"),
            unit=page.get("unit"),
            angle=page.get("angle"),
            lines=[DocumentLine(str(line.get("content") or ""), _points(line.get("polygon")))
                   for line in page.get("lines") or ()],
            words=[DocumentWord(str(word.get("content") or ""), _points(word.get("polygon")),
                                float(word.get("confidence") or 0.0))
                   for word in page.get("words") or ()],
        ))
    return AnalyzeResult(
        api_version=str(data.get("apiVersion") or ""),
        model_id=str(data.get("modelId") or ""),
        content=str(data.get("content") or ""),
        pages=pages,
    )


def _error_message(response) -> tuple:
    try:
        payload = response.json()
    except Exception:
        payload = None
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        inner = error.get("innererror") if isinstance(error.get("innererror"), dict) else {}
        message = error.get("message") or inner.get("message") or ""
        return str(error.get("code") or inner.get("code") or ""), str(message)
    return "", (getattr(response, "text", "") or "")[:300]


def _retry_after(response, default: float) -> float:
    value = (getattr(response, "headers", {}) or {}).get("Retry-After")
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return default


class _AnalyzePoller:
    """The SDK poller's ``result()`` / ``done()`` / ``status()`` for one analyze operation."""

    def __init__(self, client: "DocumentAnalysisRestClient", operation_url: str, model_id: str,
                 first_wait: float):
        self._client = client
        self._url = operation_url
        self._model_id = model_id
        self._wait = first_wait
        self._status = "notStarted"
        self._result: Optional[AnalyzeResult] = None

    def status(self) -> str:
        return self._status

    def done(self) -> bool:
        return self._result is not None or self._status in ("succeeded", "failed", "canceled")

    def result(self, timeout: Optional[float] = None) -> AnalyzeResult:
        if self._result is not None:
            return self._result
        limit = self._client.poll_timeout if timeout is None else float(timeout)
        deadline = time.monotonic() + limit
        while True:
            self._client._sleep(self._wait)
            response = self._client._request("GET", self._url)
            payload = response.json() or {}
            self._status = str(payload.get("status") or "").strip()
            if self._status == "succeeded":
                result = parse_analyze_result(payload.get("analyzeResult") or {})
                if not result.model_id:
                    result = AnalyzeResult(result.api_version, self._model_id, result.content, result.pages)
                self._result = result
                return result
            if self._status in ("failed", "canceled"):
                error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
                raise DocumentIntelligenceError(
                    f"Document Intelligence analyze {self._status}: "
                    f"{error.get('code') or ''} {error.get('message') or ''}".strip(),
                    code=str(error.get("code") or ""),
                )
            if time.monotonic() >= deadline:
                raise DocumentIntelligenceError(f"Document Intelligence analyze timed out after {limit:.0f}s")
            self._wait = _retry_after(response, self._client.poll_interval)


class DocumentAnalysisRestClient:
    """``azure.ai.formrecognizer.DocumentAnalysisClient`` stand-in over REST (key auth)."""

    def __init__(self, endpoint: str, key: str, *, timeout: float = 60.0, poll_interval: float = 1.0,
                 poll_timeout: float = 180.0, retries: int = 2, backoff: float = 1.0,
                 stop_check: Optional[Callable[[], bool]] = None, session: Any = None):
        endpoint = str(endpoint or "").strip()
        if not endpoint or not str(key or "").strip():
            raise ValueError("Azure Document Intelligence needs an endpoint and a key")
        if "://" not in endpoint:
            endpoint = "https://" + endpoint
        self.endpoint = endpoint.rstrip("/")
        self._key = str(key).strip()
        self.timeout = float(timeout)
        self.poll_interval = max(0.0, float(poll_interval))
        self.poll_timeout = float(poll_timeout)
        self.retries = max(0, int(retries))
        self.backoff = max(0.0, float(backoff))
        self._stop_check = stop_check
        self._session = session

    # -- transport --------------------------------------------------------------------------
    def _stopped(self) -> bool:
        try:
            return bool(self._stop_check and self._stop_check())
        except Exception:
            return False

    def _sleep(self, seconds: float) -> None:
        deadline = time.monotonic() + max(0.0, seconds)
        while True:
            if self._stopped():
                raise DocumentIntelligenceError("Document Intelligence request cancelled", code="Cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.1, remaining))

    def _http(self):
        if self._session is not None:
            return self._session
        import requests
        return requests

    def _request(self, method: str, url: str, **kwargs: Any):
        headers = dict(kwargs.pop("headers", None) or {})
        headers[_KEY_HEADER] = self._key
        attempt = 0
        while True:
            if self._stopped():
                raise DocumentIntelligenceError("Document Intelligence request cancelled", code="Cancelled")
            response = self._http().request(method, url, headers=headers, timeout=self.timeout, **kwargs)
            status = int(getattr(response, "status_code", 0) or 0)
            if status < 400:
                return response
            code, message = _error_message(response)
            if (status == 429 or status >= 500) and attempt < self.retries:
                attempt += 1
                self._sleep(_retry_after(response, self.backoff * attempt))
                continue
            raise DocumentIntelligenceError(f"HTTP {status} {code}: {message}".strip(), status_code=status, code=code)

    # -- SDK surface ------------------------------------------------------------------------
    def begin_analyze_document(self, model_id: str, document: Any = None, *, locale: Optional[str] = None,
                               **kwargs: Any) -> _AnalyzePoller:
        """Start an analysis of ``document`` (bytes or a binary file object)."""
        if document is None:
            document = kwargs.pop("body", None)
        if hasattr(document, "read"):
            document = document.read()
        if not isinstance(document, (bytes, bytearray)):
            raise TypeError("document must be bytes or a binary file object")
        last_error: Optional[DocumentIntelligenceError] = None
        for route, version in _ROUTES:
            url = f"{self.endpoint}/{route}/documentModels/{quote(str(model_id), safe='')}:analyze"
            params = {"api-version": version}
            if locale:
                params["locale"] = str(locale)
            try:
                response = self._request("POST", url, params=params, data=bytes(document),
                                         headers={"Content-Type": "application/octet-stream"})
            except DocumentIntelligenceError as exc:
                if exc.status_code == 404:
                    last_error = exc
                    continue
                raise
            operation = (response.headers or {}).get("Operation-Location") or \
                (response.headers or {}).get("operation-location")
            if not operation:
                raise DocumentIntelligenceError("Document Intelligence: no Operation-Location in the response",
                                                status_code=int(getattr(response, "status_code", 0) or 0))
            return _AnalyzePoller(self, operation, str(model_id), _retry_after(response, self.poll_interval))
        raise last_error or DocumentIntelligenceError("Document Intelligence: no analyze route", status_code=404)

    def close(self) -> None:
        session = self._session
        if session is not None and hasattr(session, "close"):
            session.close()

    def __enter__(self) -> "DocumentAnalysisRestClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
