"""Fast, resumable PDF extraction for Glossarion.

The legacy extractor in :mod:`pdf_extractor` is intentionally kept intact.
This module provides the two modern extraction paths:

``fast_semantic``
    Reading-oriented HTML built from sorted text blocks.

``fast_layout``
    MuPDF XHTML text with external, deduplicated image placements.  Image
    bytes are excluded from the TextPage so they are never base64 encoded.

Both paths cache source-page artifacts.  Bookmark sections remain a derived
view of those pages, which means an outline-only PDF update can regroup the
cached pages without rendering them again.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import hashlib
import html
import io
import json
import os
import re
import shutil
import threading
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from statistics import median
from typing import Callable, Dict, List, Optional, Sequence, Tuple


FAST_EXTRACTOR_VERSION = 6
FAST_MODES = {"fast_semantic", "fast_layout"}
PDF_PARAGRAPH_ALIGNMENTS = {"source", "left", "center", "right"}
PDF_HEADER_ALIGNMENTS = {"source", "left", "center", "right"}
PDF_PARAGRAPH_JUSTIFICATIONS = {"source", "justify", "none"}

_PDF_HASH_IMAGE_RE = re.compile(
    r"^pdfimg_[0-9a-f]{64}\.[a-z0-9]{1,8}$", re.IGNORECASE
)
_PDF_LEGACY_IMAGE_RE = re.compile(
    r"^page_(\d+)_img_(\d+)\.[a-z0-9]{1,8}$", re.IGNORECASE
)
_PDF_FRIENDLY_IMAGE_RE = re.compile(
    r"^(?:pdf_section_.+?_img_\d+|\d+_Cover)\.[a-z0-9]{1,8}$",
    re.IGNORECASE,
)


class PDFExtractionCancelled(RuntimeError):
    """Raised when the active PDF extraction receives a stop request."""


_PROCESS_PDF_DOCUMENT = None
_PROCESS_PDF_PATH = ""
_PROCESS_STREAM_CACHE: Dict[int, str] = {}


def _stop_requested(stop_callback: Optional[Callable[[], bool]] = None) -> bool:
    if stop_callback:
        try:
            if stop_callback():
                return True
        except Exception:
            pass
    if os.environ.get("TRANSLATION_CANCELLED") == "1":
        return True
    stop_file = os.environ.get("PDF_EXTRACTION_STOP_FILE", "")
    return bool(stop_file and os.path.exists(stop_file))


def _signal_stop_file() -> None:
    stop_file = os.environ.get("PDF_EXTRACTION_STOP_FILE", "")
    if not stop_file:
        return
    try:
        Path(stop_file).parent.mkdir(parents=True, exist_ok=True)
        Path(stop_file).write_text("stop", encoding="utf-8")
    except OSError:
        pass


class _ExtractionProgressMonitor:
    """Emit time-based heartbeats and bridge GUI cancellation to workers."""

    def __init__(self, stop_callback, stop_file: Path, owns_stop_file: bool):
        self.stop_callback = stop_callback
        self.stop_file = stop_file
        self.owns_stop_file = owns_stop_file
        self.started = time.perf_counter()
        self.total_pages = 0
        self.completed_pages = 0
        self.completed_jobs = 0
        self.total_jobs = 0
        self.phase = "initializing"
        try:
            self.heartbeat_seconds = max(
                0.05,
                float(os.environ.get("PDF_PROGRESS_HEARTBEAT_SECONDS", "3")),
            )
        except (TypeError, ValueError):
            self.heartbeat_seconds = 3.0
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name="pdf-fast-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def configure(self, total_pages: int, total_jobs: int, phase: str = "extracting") -> None:
        with self._lock:
            self.total_pages = max(0, int(total_pages or 0))
            self.total_jobs = max(0, int(total_jobs or 0))
            self.phase = phase

    def update(self, *, pages: int = 0, jobs: int = 0, phase: Optional[str] = None) -> None:
        with self._lock:
            self.completed_pages += max(0, int(pages or 0))
            self.completed_jobs += max(0, int(jobs or 0))
            if phase:
                self.phase = phase

    def _run(self) -> None:
        next_heartbeat = time.perf_counter() + self.heartbeat_seconds
        poll_seconds = min(0.2, self.heartbeat_seconds)
        while not self._closed.wait(poll_seconds):
            if _stop_requested(self.stop_callback):
                _signal_stop_file()
            now = time.perf_counter()
            if now < next_heartbeat:
                continue
            with self._lock:
                total_pages = self.total_pages
                completed_pages = self.completed_pages
                completed_jobs = self.completed_jobs
                total_jobs = self.total_jobs
                phase = self.phase
            elapsed = max(0, int(now - self.started))
            if total_pages:
                percent = min(100, int(completed_pages * 100 / total_pages))
                print(
                    f"⏳ Fast PDF extraction heartbeat: {phase}, "
                    f"{completed_pages}/{total_pages} pages ({percent}%), "
                    f"{completed_jobs}/{total_jobs or '?'} jobs, {elapsed}s elapsed"
                )
            else:
                print(f"⏳ Fast PDF extraction heartbeat: {phase}, {elapsed}s elapsed")
            next_heartbeat = now + self.heartbeat_seconds

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=0.5)
        if self.owns_stop_file:
            try:
                self.stop_file.unlink()
            except OSError:
                pass


def _initialize_pdf_worker(pdf_path: str) -> None:
    """Open one persistent document per worker instead of once per job."""
    global _PROCESS_PDF_DOCUMENT, _PROCESS_PDF_PATH, _PROCESS_STREAM_CACHE
    import fitz

    _PROCESS_PDF_PATH = os.path.abspath(pdf_path)
    _PROCESS_PDF_DOCUMENT = fitz.open(pdf_path)
    _PROCESS_STREAM_CACHE = {}


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            if _stop_requested():
                raise PDFExtractionCancelled("PDF extraction cancelled")
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, default=None):
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except Exception:
        return default


def _atomic_write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
    try:
        os.replace(temporary, path)
    except PermissionError:
        # Two page-range workers can discover the same shared xref at once.
        # On Windows, replacing a destination during the other process's
        # atomic rename can briefly raise ACCESS_DENIED.  A complete winning
        # file is already present, so the losing identical map can be dropped.
        if path.is_file():
            try:
                temporary.unlink()
            except OSError:
                pass
            return
        raise


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            str(path),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        )
    except FileExistsError:
        return
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value)


def _outline_digest(toc_entries: Sequence[Sequence]) -> str:
    normalized = []
    for entry in toc_entries or []:
        if len(entry) < 3:
            continue
        try:
            level = max(1, int(entry[0]))
            page = max(0, int(entry[2]))
        except (TypeError, ValueError):
            continue
        title = " ".join(str(entry[1] or "").split())
        normalized.append([level, title, page])
    encoded = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _rect_list(value) -> List[float]:
    try:
        return [float(value.x0), float(value.y0), float(value.x1), float(value.y1)]
    except AttributeError:
        try:
            return [float(value[index]) for index in range(4)]
        except Exception:
            return [0.0, 0.0, 0.0, 0.0]


def _rect_area(value) -> float:
    rect = _rect_list(value)
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def _rect_intersection_area(left, right) -> float:
    first = _rect_list(left)
    second = _rect_list(right)
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _rect_center_inside(inner, outer) -> bool:
    source = _rect_list(inner)
    target = _rect_list(outer)
    center_x = (source[0] + source[2]) / 2.0
    center_y = (source[1] + source[3]) / 2.0
    return (
        target[0] <= center_x <= target[2]
        and target[1] <= center_y <= target[3]
    )


def _stream_digest(doc, xref: int, cache: Dict[int, str]) -> str:
    if xref in cache:
        return cache[xref]
    digest = hashlib.sha256()
    try:
        digest.update(doc.xref_object(xref, compressed=False).encode("utf-8", "replace"))
    except Exception:
        pass
    try:
        stream = doc.xref_stream_raw(xref)
        if stream:
            digest.update(stream)
    except Exception:
        pass
    value = digest.hexdigest()
    cache[xref] = value
    return value


def _page_source_signature(doc, page, stream_cache: Dict[int, str]) -> str:
    """Hash all source objects that can affect one extracted page.

    This intentionally reads raw streams rather than decoding images.  It is
    used only when the PDF file hash changed and lets outline-only updates
    retain the expensive page extraction cache.
    """

    digest = hashlib.sha256()
    digest.update(f"page:{page.number};rect:{tuple(page.rect)}".encode("ascii", "replace"))
    try:
        digest.update(doc.xref_object(page.xref, compressed=False).encode("utf-8", "replace"))
    except Exception:
        pass

    for xref in page.get_contents() or []:
        if _stop_requested():
            raise PDFExtractionCancelled("PDF extraction cancelled")
        try:
            digest.update(f"content:{int(xref)}:".encode("ascii"))
            digest.update(_stream_digest(doc, int(xref), stream_cache).encode("ascii"))
        except Exception:
            continue

    try:
        image_xrefs = sorted({int(item[0]) for item in page.get_images(full=True) if int(item[0]) > 0})
    except Exception:
        image_xrefs = []
    for xref in image_xrefs:
        if _stop_requested():
            raise PDFExtractionCancelled("PDF extraction cancelled")
        digest.update(f"image:{xref}:".encode("ascii"))
        digest.update(_stream_digest(doc, xref, stream_cache).encode("ascii"))

    try:
        links = []
        for link in page.get_links() or []:
            links.append(
                {
                    "from": _rect_list(link.get("from")),
                    "page": link.get("page"),
                    "uri": link.get("uri"),
                    "kind": link.get("kind"),
                }
            )
        digest.update(json.dumps(links, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    except Exception:
        pass
    return digest.hexdigest()


def _bookmark_jobs(toc_entries: Sequence[Sequence], total_pages: int, chunk_pages: int):
    """Return page-range jobs aligned to outline boundaries where possible."""

    starts: Dict[int, str] = {}
    for entry in toc_entries or []:
        if len(entry) < 3:
            continue
        try:
            page_index = int(entry[2]) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= page_index < total_pages:
            starts.setdefault(page_index, " ".join(str(entry[1] or "").split()))

    boundaries = sorted(set([0, total_pages, *starts.keys()]))
    jobs = []
    for boundary_index in range(len(boundaries) - 1):
        section_start = boundaries[boundary_index]
        section_end = boundaries[boundary_index + 1]
        for start in range(section_start, section_end, chunk_pages):
            jobs.append((start, min(start + chunk_pages, section_end)))
    if not jobs and total_pages:
        jobs = [
            (start, min(start + chunk_pages, total_pages))
            for start in range(0, total_pages, chunk_pages)
        ]
    return jobs, starts


def resolve_pdf_extraction_workers(
    configured=None,
    *,
    cpu_count: Optional[int] = None,
) -> int:
    """Resolve the dedicated PDF worker setting to a safe numeric value.

    ``auto`` means half of the available logical CPUs.  The legacy
    ``EXTRACTION_WORKERS`` value is only a compatibility fallback when the
    dedicated ``PDF_EXTRACTION_WORKERS`` variable has not been initialized.
    """
    try:
        available_cpus = max(
            1,
            int(cpu_count if cpu_count is not None else (os.cpu_count() or 1)),
        )
    except (TypeError, ValueError):
        available_cpus = 1
    automatic = max(1, available_cpus // 2)
    raw_value = configured
    if raw_value is None:
        raw_value = os.environ.get(
            "PDF_EXTRACTION_WORKERS",
            os.environ.get("EXTRACTION_WORKERS", "auto"),
        )
    normalized = str(raw_value or "auto").strip().lower()
    if normalized in {"", "auto", "automatic", "default", "0"}:
        requested = automatic
    else:
        try:
            requested = int(normalized)
        except (TypeError, ValueError):
            requested = automatic
    return max(1, min(requested, available_cpus))


def _fast_pdf_worker_count(total_pages: int, job_count: int) -> int:
    """Return the bounded number of bookmark jobs to run concurrently."""
    if total_pages < 8 or job_count <= 1:
        return 1
    try:
        cpu_count = max(1, int(os.cpu_count() or 1))
    except (TypeError, ValueError):
        cpu_count = 1
    requested = resolve_pdf_extraction_workers(cpu_count=cpu_count)
    try:
        safety_cap = int(str(os.environ.get("PDF_FAST_MAX_WORKERS", cpu_count)).strip())
    except (TypeError, ValueError):
        safety_cap = cpu_count
    return max(
        1,
        min(
            int(job_count),
            max(1, requested),
            cpu_count,
            max(1, safety_cap),
        ),
    )


def _image_occurrences(page) -> List[Dict]:
    """Return displayed image placements without decoding every image hash.

    ``get_image_info(hashes=True)`` decodes and hashes every image before the
    extractor immediately decodes it again for output.  PDFs with thousands of
    illustrations spend most of their time in that duplicated work.  Resource
    xrefs plus ``get_image_rects`` provide the same displayed placements; the
    final extracted bytes remain content-hashed by :func:`_extract_asset`.
    """

    items = []
    try:
        resources = page.get_images(full=True) or []
    except Exception:
        resources = []

    seen_xrefs = set()
    for resource_index, resource in enumerate(resources):
        try:
            xref = int(resource[0] or 0)
        except (TypeError, ValueError, IndexError):
            continue
        if xref <= 0 or xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        try:
            rectangles = page.get_image_rects(xref) or []
        except Exception:
            rectangles = []
        try:
            width = int(resource[2] or 0)
            height = int(resource[3] or 0)
            has_mask = bool(int(resource[1] or 0))
        except (TypeError, ValueError, IndexError):
            width = height = 0
            has_mask = False
        for rectangle in rectangles:
            items.append(
                {
                    "index": len(items),
                    "number": len(items),
                    "resource_index": resource_index,
                    "xref": xref,
                    "source_digest": f"xref:{xref}",
                    "bbox": _rect_list(rectangle),
                    "width": width,
                    "height": height,
                    "has_mask": has_mask,
                }
            )

    if items:
        return items

    # Narrow fallback for inline images, which have no reusable xref.
    try:
        raw_items = page.get_image_info(hashes=False, xrefs=False) or []
    except Exception:
        raw_items = []
    for index, item in enumerate(raw_items):
        items.append({
            "index": index,
            "number": int(item.get("number", index)),
            "resource_index": index,
            "xref": 0,
            "source_digest": f"inline:{page.number}:{index}",
            "bbox": _rect_list(item.get("bbox")),
            "width": int(item.get("width") or 0),
            "height": int(item.get("height") or 0),
            "has_mask": bool(item.get("has-mask", False)),
        })
    return items


_PLAIN_URL_RE = re.compile(
    r"(?i)(?:https?://|ftp://|www\.)[^\s<>\"']+"
)


def _safe_link_href(link: Dict) -> Tuple[str, bool]:
    """Return a safe HTML target and whether it is an external URL."""
    try:
        target_page = int(link.get("page"))
    except (TypeError, ValueError):
        target_page = -1
    if target_page >= 0:
        return f"#page-{target_page + 1}", False

    uri = str(link.get("uri") or "").strip()
    if uri:
        if re.match(r"(?i)^(?:javascript|data|vbscript):", uri):
            return "", False
        external = bool(re.match(r"(?i)^(?:https?|ftp|mailto|tel):", uri))
        return uri, external

    filename = str(link.get("file") or "").strip()
    if filename:
        return filename, False
    named = str(link.get("nameddest") or link.get("name") or "").strip()
    if named:
        return f"#{named.lstrip('#')}", False
    return "", False


def _page_link_records(page) -> List[Dict]:
    """Read PDF annotations and the exact words covered by each link."""
    try:
        source_links = page.get_links() or []
    except Exception:
        source_links = []
    if not source_links:
        return []

    try:
        words = page.get_text("words", sort=True) or []
    except Exception:
        words = []
    records = []
    for source_link in source_links:
        href, external = _safe_link_href(source_link)
        bbox = _rect_list(source_link.get("from"))
        if not href or _rect_area(bbox) <= 0:
            continue

        matched_words = []
        for word in words:
            if len(word) < 5:
                continue
            word_bbox = [float(word[index]) for index in range(4)]
            word_area = _rect_area(word_bbox)
            overlap = _rect_intersection_area(word_bbox, bbox)
            if not (
                _rect_center_inside(word_bbox, bbox)
                or (word_area > 0 and overlap / word_area >= 0.2)
            ):
                continue
            order = tuple(word[index] if len(word) > index else 0 for index in (5, 6, 7))
            matched_words.append((order, float(word[1]), float(word[0]), str(word[4])))
        matched_words.sort(key=lambda item: (item[0], item[1], item[2]))
        phrase = " ".join(item[3] for item in matched_words)
        phrase = " ".join(phrase.split())
        if not phrase:
            try:
                import fitz

                phrase = " ".join(
                    str(page.get_text("text", clip=fitz.Rect(*bbox)) or "").split()
                )
            except Exception:
                phrase = ""
        records.append(
            {
                "bbox": bbox,
                "href": href,
                "external": external,
                "text": phrase,
                "used": False,
            }
        )
    return records


def _link_attributes(href: str, external: bool) -> str:
    attributes = f'href="{html.escape(str(href), quote=True)}"'
    if external:
        attributes += ' target="_blank" rel="noopener noreferrer"'
    return attributes


def _free_text_interval(
    text: str,
    needle: str,
    occupied: Sequence[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    if not needle:
        return None
    folded_text = text.casefold()
    folded_needle = needle.casefold()
    start = 0
    while True:
        index = folded_text.find(folded_needle, start)
        if index < 0:
            return None
        interval = (index, index + len(needle))
        if not any(interval[0] < end and begin < interval[1] for begin, end in occupied):
            return interval
        start = index + 1


def _linked_text_html(
    value: str,
    bbox: Optional[Sequence[float]],
    links: Sequence[Dict],
) -> str:
    """Escape text while retaining annotation links and plain URL strings."""
    text_value = " ".join(str(value or "").split())
    if not text_value:
        return ""

    intervals: List[Tuple[int, int, str, bool, Optional[Dict]]] = []
    occupied: List[Tuple[int, int]] = []
    for link in links or []:
        if link.get("used"):
            continue
        link_bbox = link.get("bbox") or []
        if bbox and _rect_intersection_area(bbox, link_bbox) <= 0:
            continue
        phrase = " ".join(str(link.get("text") or "").split())
        interval = _free_text_interval(text_value, phrase, occupied)
        if interval is None:
            href_text = str(link.get("href") or "")
            interval = _free_text_interval(text_value, href_text, occupied)
        if interval is None:
            continue
        occupied.append(interval)
        intervals.append(
            (
                interval[0],
                interval[1],
                str(link.get("href") or ""),
                bool(link.get("external")),
                link,
            )
        )

    for match in _PLAIN_URL_RE.finditer(text_value):
        start, end = match.span()
        while end > start and text_value[end - 1] in ".,;:!?":
            end -= 1
        if end <= start or any(start < right and left < end for left, right in occupied):
            continue
        label = text_value[start:end]
        href = label if not label.casefold().startswith("www.") else f"https://{label}"
        occupied.append((start, end))
        intervals.append((start, end, href, True, None))

    if not intervals:
        return html.escape(text_value)
    intervals.sort(key=lambda item: (item[0], item[1]))
    rendered = []
    cursor = 0
    for start, end, href, external, source_link in intervals:
        if start < cursor:
            continue
        rendered.append(html.escape(text_value[cursor:start]))
        rendered.append(
            f"<a {_link_attributes(href, external)}>"
            f"{html.escape(text_value[start:end])}</a>"
        )
        if source_link is not None:
            source_link["used"] = True
        cursor = end
    rendered.append(html.escape(text_value[cursor:]))
    return "".join(rendered)


def _extract_page_tables(page, drawings=None) -> List[Dict]:
    """Return usable PyMuPDF tables with row and cell geometry."""
    if not hasattr(page, "find_tables"):
        return []
    try:
        # PyMuPDF emits an optional-layout-package advertisement on every
        # call. Keep extraction logs limited to actionable progress messages.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            finder = page.find_tables(paths=drawings) if drawings is not None else page.find_tables()
    except Exception:
        return []

    extracted_tables = []
    for table_index, table in enumerate(getattr(finder, "tables", []) or []):
        try:
            values = table.extract() or []
            row_objects = list(getattr(table, "rows", []) or [])
            bbox = _rect_list(table.bbox)
        except Exception:
            continue
        rows = []
        nonempty = 0
        for row_index, raw_row in enumerate(values):
            cells = []
            if row_index < len(row_objects):
                cells = [
                    _rect_list(cell) if cell is not None else None
                    for cell in (getattr(row_objects[row_index], "cells", []) or [])
                ]
            normalized = []
            for cell in list(raw_row or []):
                cell_text = " ".join(str(cell or "").split())
                normalized.append(cell_text)
                if cell_text:
                    nonempty += 1
            rows.append({"values": normalized, "cells": cells})
        if not rows or nonempty < 2 or _rect_area(bbox) <= 0:
            continue

        header_values = []
        header_external = False
        try:
            header = table.header
            header_values = [" ".join(str(item or "").split()) for item in (header.names or [])]
            header_external = bool(header.external)
        except Exception:
            pass

        header_row = None
        if header_values and any(header_values):
            if header_external:
                header_row = {"values": header_values, "cells": []}
            elif rows and [item.casefold() for item in rows[0]["values"]] == [
                item.casefold() for item in header_values
            ]:
                header_row = rows.pop(0)
        extracted_tables.append(
            {
                "bbox": bbox,
                "number": table_index,
                "header": header_row,
                "rows": rows,
            }
        )
    return extracted_tables


def _vector_only_svg(page_svg: str, bbox: Sequence[float]) -> str:
    """Clip a page SVG to one drawing cluster and remove text/raster nodes."""
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    try:
        root = ET.fromstring(page_svg)
    except (ET.ParseError, TypeError, ValueError):
        return ""
    for parent in list(root.iter()):
        for child in list(parent):
            local_name = str(child.tag).rsplit("}", 1)[-1].casefold()
            if local_name in {"text", "image"}:
                parent.remove(child)
    x0, y0, x1, y1 = _rect_list(bbox)
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    if width <= 0 or height <= 0:
        return ""
    root.set("width", f"{width:.3f}")
    root.set("height", f"{height:.3f}")
    root.set("viewBox", f"{x0:.3f} {y0:.3f} {width:.3f} {height:.3f}")
    root.set("preserveAspectRatio", "xMinYMin meet")
    return ET.tostring(root, encoding="unicode")


def _vector_graphic_occurrences(
    page,
    drawings,
    *,
    excluded_bboxes: Sequence[Sequence[float]] = (),
    image_bboxes: Sequence[Sequence[float]] = (),
    starting_number: int = 0,
) -> List[Dict]:
    """Turn meaningful vector drawing clusters into sharp external SVGs."""
    if not drawings or not hasattr(page, "cluster_drawings"):
        return []
    page_area = max(1.0, _rect_area(page.rect))
    meaningful_drawings = []
    for drawing in drawings:
        bbox = _rect_list(drawing.get("rect"))
        width = max(0.0, bbox[2] - bbox[0])
        height = max(0.0, bbox[3] - bbox[1])
        fill = drawing.get("fill")
        stroke = drawing.get("color")
        # Many exported web PDFs paint an opaque white rectangle behind every
        # text line and use sub-point filled rectangles for hyperlink
        # underlines. They are styling artifacts, not standalone graphics.
        if (
            stroke is None
            and fill
            and all(float(channel) >= 0.985 for channel in fill[:3])
        ):
            continue
        if stroke is None and fill and min(width, height) <= 1.5:
            continue
        if _rect_area(bbox) >= page_area * 0.8:
            continue
        meaningful_drawings.append(drawing)
    if not meaningful_drawings:
        return []
    try:
        clusters = page.cluster_drawings(
            drawings=meaningful_drawings,
            x_tolerance=4,
            y_tolerance=4,
            final_filter=True,
        ) or []
    except Exception:
        return []

    accepted = []
    for cluster in clusters:
        bbox = _rect_list(cluster)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = _rect_area(bbox)
        if width < 6 or height < 6 or area < 100 or area >= page_area * 0.8:
            continue
        if any(
            _rect_intersection_area(bbox, excluded) / max(area, 1.0) >= 0.65
            for excluded in excluded_bboxes
        ):
            continue
        if any(
            _rect_intersection_area(bbox, image_bbox) / max(area, 1.0) >= 0.65
            for image_bbox in image_bboxes
        ):
            continue
        accepted.append(bbox)
    if not accepted:
        return []

    try:
        page_svg = page.get_svg_image(text_as_path=False)
    except Exception:
        return []
    occurrences = []
    for vector_index, bbox in enumerate(accepted):
        svg = _vector_only_svg(page_svg, bbox)
        if not svg:
            continue
        source_digest = hashlib.sha256(svg.encode("utf-8")).hexdigest()
        occurrences.append(
            {
                "index": starting_number + vector_index,
                "number": starting_number + vector_index,
                "resource_index": None,
                "xref": 0,
                "kind": "vector",
                "source_digest": source_digest,
                "bbox": bbox,
                "width": max(1, int(round(bbox[2] - bbox[0]))),
                "height": max(1, int(round(bbox[3] - bbox[1]))),
                "asset_text": svg,
                "asset_extension": "svg",
            }
        )
    return occurrences


def _normalize_extension(extension: str) -> str:
    extension = str(extension or "png").lower().lstrip(".")
    if extension == "jpeg":
        return "jpg"
    if not re.fullmatch(r"[a-z0-9]{1,8}", extension):
        return "png"
    return extension


def _extract_asset(
    doc,
    page,
    occurrence: Dict,
    images_dir: Path,
    xref_map_dir: Path,
) -> Optional[Dict]:
    xref = int(occurrence.get("xref") or 0)
    source_digest = occurrence.get("source_digest") or ""
    fallback_key = hashlib.sha256(
        str(source_digest or occurrence.get("index") or 0).encode("utf-8")
    ).hexdigest()
    map_key = f"xref_{xref}" if xref else f"inline_{fallback_key}"
    map_path = xref_map_dir / f"{map_key}.json"
    cached = _load_json(map_path, {}) or {}
    cached_filename = cached.get("filename")
    if cached_filename and (images_dir / cached_filename).is_file():
        return cached

    image_bytes = None
    extension = "png"
    width = int(occurrence.get("width") or 0)
    height = int(occurrence.get("height") or 0)
    asset_text = occurrence.get("asset_text")
    if asset_text:
        image_bytes = str(asset_text).encode("utf-8")
        extension = _normalize_extension(occurrence.get("asset_extension") or "svg")
    if xref:
        try:
            extracted = doc.extract_image(xref)
        except Exception:
            extracted = None
        if extracted and extracted.get("image"):
            image_bytes = extracted["image"]
            extension = _normalize_extension(extracted.get("ext"))
            width = int(extracted.get("width") or width)
            height = int(extracted.get("height") or height)

    # Inline images have no reusable PDF xref. Render only their rectangle as
    # the narrow fallback; the normal path never rasterizes the whole page.
    if image_bytes is None:
        try:
            import fitz

            bbox = occurrence.get("bbox") or [0, 0, 0, 0]
            clip = fitz.Rect(*bbox)
            if clip.is_empty or clip.is_infinite:
                return None
            pixmap = page.get_pixmap(clip=clip, alpha=False)
            image_bytes = pixmap.tobytes("png")
            extension = "png"
            width = pixmap.width
            height = pixmap.height
        except Exception:
            return None

    content_digest = hashlib.sha256(image_bytes).hexdigest()
    filename = f"pdfimg_{content_digest}.{extension}"
    output_path = images_dir / filename
    _atomic_write_bytes(output_path, image_bytes)
    result = {
        "filename": filename,
        "path": str(output_path),
        "digest": content_digest,
        "xref": xref,
        "width": width,
        "height": height,
    }
    _atomic_write_json(map_path, result)
    return result


def _materialize_images(doc, page, occurrences, images_dir: Path, xref_map_dir: Path):
    local_assets = {}
    materialized = []
    for occurrence in occurrences:
        if _stop_requested():
            raise PDFExtractionCancelled("PDF extraction cancelled")
        key = (occurrence.get("xref"), occurrence.get("source_digest"))
        asset = local_assets.get(key)
        if asset is None:
            asset = _extract_asset(doc, page, occurrence, images_dir, xref_map_dir)
            if asset:
                local_assets[key] = asset
        if not asset:
            continue
        item = {
            name: value
            for name, value in occurrence.items()
            if name not in {"asset_text", "asset_extension"}
        }
        item.update(asset)
        materialized.append(item)
    return materialized


def _text_alignment(
    bbox: Sequence[float],
    page_width: float,
    *,
    is_heading: bool = False,
) -> str:
    if len(bbox) < 4 or page_width <= 0:
        return "left"
    left = float(bbox[0])
    right = page_width - float(bbox[2])
    # A normal novel paragraph often fills the text column, which naturally
    # leaves similar outer margins even though every line starts at the left
    # edge.  Equal margins alone therefore do not mean centered text.  Require
    # a materially inset block as well; this still recognizes centered titles
    # while keeping full-width body paragraphs left aligned.
    if (
        abs(left - right) <= max(12.0, page_width * 0.04)
        and (is_heading or min(left, right) >= page_width * 0.18)
    ):
        # Equal outer margins are sufficient for a confirmed bookmark title.
        # The 18% inset remains required for generic text so a full-width body
        # paragraph is not mistaken for centered text.
        return "center"
    if right < page_width * 0.08 and left > page_width * 0.2:
        return "right"
    return "left"


def normalize_pdf_paragraph_alignment(value=None) -> str:
    normalized = str(
        value
        if value is not None
        else os.environ.get("PDF_PARAGRAPH_ALIGNMENT", "source")
    ).strip().lower()
    normalized = {
        "": "source",
        "default": "source",
        "source_pdf": "source",
        "centre": "center",
    }.get(normalized, normalized)
    return normalized if normalized in PDF_PARAGRAPH_ALIGNMENTS else "source"


def normalize_pdf_header_alignment(value=None) -> str:
    """Normalize the section-heading alignment override."""
    normalized = str(
        value
        if value is not None
        else os.environ.get("PDF_HEADER_ALIGNMENT", "source")
    ).strip().lower()
    normalized = {
        "": "source",
        "default": "source",
        "source_pdf": "source",
        "centre": "center",
    }.get(normalized, normalized)
    return normalized if normalized in PDF_HEADER_ALIGNMENTS else "source"


def resolve_pdf_header_alignment(
    source_alignment: str,
    *,
    alignment_override=None,
) -> str:
    """Resolve a source heading alignment against the user override."""
    source = str(source_alignment or "left").strip().lower()
    if source not in {"left", "center", "right"}:
        source = "left"
    override = normalize_pdf_header_alignment(alignment_override)
    return source if override == "source" else override


def normalize_pdf_paragraph_justification(value=None) -> str:
    normalized = str(
        value
        if value is not None
        else os.environ.get("PDF_PARAGRAPH_JUSTIFICATION", "source")
    ).strip().lower()
    normalized = {
        "": "source",
        "default": "source",
        "source_pdf": "source",
        "justified": "justify",
        "not_justified": "none",
        "off": "none",
    }.get(normalized, normalized)
    return normalized if normalized in PDF_PARAGRAPH_JUSTIFICATIONS else "source"


def pdf_rtl_paragraph_layout_enabled(value=None) -> bool:
    """Return whether PDF text should use an explicit RTL paragraph layout."""
    raw_value = (
        value
        if value is not None
        else os.environ.get("PDF_RTL_PARAGRAPH_LAYOUT", "0")
    )
    if isinstance(raw_value, bool):
        return raw_value
    return str(raw_value or "").strip().lower() in {
        "1", "true", "yes", "on", "enabled", "rtl"
    }


def _text_direction_alignment(text: str) -> str:
    """Return the natural edge for a non-justified paragraph."""
    for character in str(text or ""):
        direction = unicodedata.bidirectional(character)
        if direction in {"R", "AL"}:
            return "right"
        if direction == "L":
            return "left"
    return "left"


def resolve_pdf_paragraph_alignment(
    source_alignment: str,
    text: str = "",
    *,
    alignment_override=None,
    justification_override=None,
    rtl_layout=None,
) -> str:
    """Resolve source formatting and the two independent user overrides."""
    source = str(source_alignment or "left").strip().lower()
    if source not in {"left", "center", "right", "justify"}:
        source = _text_direction_alignment(text)
    alignment = normalize_pdf_paragraph_alignment(alignment_override)
    justification = normalize_pdf_paragraph_justification(justification_override)

    # An explicit justification choice has precedence over horizontal
    # alignment. Choosing an explicit alignment disables source justification.
    if justification == "justify":
        resolved = "justify"
    elif justification == "none":
        if alignment != "source":
            resolved = alignment
        else:
            resolved = _text_direction_alignment(text) if source == "justify" else source
    elif alignment != "source":
        resolved = alignment
    else:
        resolved = source

    # Direction alone does not move a source-left paragraph to the right.
    # In RTL mode, use the natural right edge unless the user explicitly chose
    # an alignment. Centered and justified source paragraphs remain unchanged.
    if (
        pdf_rtl_paragraph_layout_enabled(rtl_layout)
        and alignment == "source"
        and resolved == "left"
    ):
        return "right"
    return resolved


def _semantic_layout_geometry(page, flags: int):
    """Read line geometry and infer the dominant text-column bounds."""
    try:
        layout = page.get_text("dict", sort=True, flags=flags) or {}
    except TypeError:
        try:
            layout = page.get_text("dict", sort=True) or {}
        except Exception:
            return {}, None
    except Exception:
        return {}, None
    if not isinstance(layout, dict):
        return {}, None

    page_width = float(getattr(getattr(page, "rect", None), "width", 0.0) or 0.0)
    by_number = {}
    column_candidates = []
    for block in layout.get("blocks") or []:
        if not isinstance(block, dict) or int(block.get("type", 0)) != 0:
            continue
        try:
            number = int(block.get("number"))
        except (TypeError, ValueError):
            continue
        by_number[number] = block
        bbox = block.get("bbox") or []
        lines = block.get("lines") or []
        if len(bbox) >= 4 and (
            len(lines) >= 2 or float(bbox[2]) - float(bbox[0]) >= page_width * 0.38
        ):
            column_candidates.append((float(bbox[0]), float(bbox[2])))
    column_bounds = None
    if column_candidates:
        column_bounds = (
            float(median(item[0] for item in column_candidates)),
            float(median(item[1] for item in column_candidates)),
        )
    return by_number, column_bounds


def _layout_paragraph_alignment(
    block: Optional[Dict],
    page_width: float,
    column_bounds,
    text: str,
) -> str:
    """Detect left/center/right/justify from PDF line positions."""
    if not isinstance(block, dict):
        return _text_direction_alignment(text)
    usable_lines = []
    for line in block.get("lines") or []:
        line_text = "".join(
            str(span.get("text") or "")
            for span in (line.get("spans") or [])
            if isinstance(span, dict)
        ).strip()
        bbox = line.get("bbox") or []
        if line_text and len(bbox) >= 4:
            usable_lines.append([float(value) for value in bbox[:4]])
    if not usable_lines:
        return _text_direction_alignment(text)

    lefts = [line[0] for line in usable_lines]
    rights = [line[2] for line in usable_lines]
    widths = [right - left for left, right in zip(lefts, rights)]
    tolerance = max(1.5, page_width * 0.004)

    if len(usable_lines) >= 2:
        widest_right = max(rights)
        common_left = min(lefts)
        full_right_lines = sum(
            abs(right - widest_right) <= tolerance for right in rights
        )
        stable_left_lines = sum(
            abs(left - common_left) <= tolerance for left in lefts
        )
        if (
            full_right_lines >= 2
            and full_right_lines * 2 >= len(usable_lines)
            and stable_left_lines * 4 >= len(usable_lines) * 3
            and max(widths) >= page_width * 0.3
        ):
            return "justify"

        centers = [(left + right) / 2.0 for left, right in zip(lefts, rights)]
        if (
            max(centers) - min(centers) <= max(5.0, page_width * 0.012)
            and max(widths) - min(widths) > tolerance * 2
            # A wrapped, left-aligned paragraph can coincidentally have
            # similar line centers when its last line is only moderately
            # shorter.  True centered text varies at both edges; identical
            # left edges are positive evidence that the paragraph is left
            # aligned (and likewise for identical right edges).
            and max(lefts) - min(lefts) > tolerance * 2
            and max(rights) - min(rights) > tolerance * 2
        ):
            return "center"
        if max(rights) - min(rights) <= tolerance and max(lefts) - min(lefts) > tolerance:
            return "right"

    bbox = block.get("bbox") or []
    if len(bbox) >= 4 and column_bounds:
        block_left = float(bbox[0])
        block_right = float(bbox[2])
        column_left, column_right = column_bounds
        edge_tolerance = max(6.0, page_width * 0.015)
        if abs(block_left - column_left) <= edge_tolerance:
            return "left"
        if abs(block_right - column_right) <= edge_tolerance:
            return "right"
        if abs(
            (block_left + block_right) / 2.0 - (column_left + column_right) / 2.0
        ) <= edge_tolerance:
            return "center"
    return _text_direction_alignment(text)


def _semantic_title_key(value: str) -> str:
    """Normalize PDF line-wrap whitespace for bookmark-title matching."""
    return re.sub(r"\s+", "", str(value or "").casefold())


def _link_for_visual_bbox(bbox: Sequence[float], links: Sequence[Dict]) -> Optional[Dict]:
    area = _rect_area(bbox)
    if area <= 0:
        return None
    best_link = None
    best_score = 0.0
    for link in links or []:
        if link.get("used"):
            continue
        link_bbox = link.get("bbox") or []
        link_area = _rect_area(link_bbox)
        overlap = _rect_intersection_area(bbox, link_bbox)
        score = overlap / max(1.0, min(area, link_area))
        if score > best_score:
            best_score = score
            best_link = link
    return best_link if best_score >= 0.4 else None


def _semantic_table_html(table: Dict, links: Sequence[Dict]) -> str:
    parts = ['<table class="pdf-table">']
    header_row = table.get("header")
    if header_row:
        parts.append("<thead><tr>")
        values = header_row.get("values") or []
        cells = header_row.get("cells") or []
        for index, value in enumerate(values):
            bbox = cells[index] if index < len(cells) else table.get("bbox")
            parts.append(f"<th>{_linked_text_html(value, bbox, links)}</th>")
        parts.append("</tr></thead>")
    parts.append("<tbody>")
    for row in table.get("rows") or []:
        parts.append("<tr>")
        values = row.get("values") or []
        cells = row.get("cells") or []
        for index, value in enumerate(values):
            bbox = cells[index] if index < len(cells) else table.get("bbox")
            parts.append(f"<td>{_linked_text_html(value, bbox, links)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _semantic_page_html(
    page,
    page_number: int,
    images: List[Dict],
    section_title: str,
    *,
    tables: Optional[List[Dict]] = None,
    links: Optional[List[Dict]] = None,
) -> str:
    import fitz

    if tables is None:
        tables = _extract_page_tables(page)
    if links is None:
        links = _page_link_records(page)
    table_bboxes = [table.get("bbox") or [] for table in tables]
    flags = int(getattr(fitz, "TEXTFLAGS_BLOCKS", 195)) & ~int(
        getattr(fitz, "TEXT_PRESERVE_IMAGES", 4)
    )
    try:
        blocks = page.get_text("blocks", sort=True, flags=flags) or []
    except TypeError:
        blocks = page.get_text("blocks", sort=True) or []
    dict_flags = int(getattr(fitz, "TEXTFLAGS_DICT", 199)) & ~int(
        getattr(fitz, "TEXT_PRESERVE_IMAGES", 4)
    )
    layout_by_number, column_bounds = _semantic_layout_geometry(page, dict_flags)

    events = []
    for block in blocks:
        if len(block) < 7 or int(block[6]) != 0:
            continue
        text_value = " ".join(str(block[4] or "").split())
        if not text_value:
            continue
        block_bbox = [float(block[index]) for index in range(4)]
        block_area = max(1.0, _rect_area(block_bbox))
        if any(
            _rect_center_inside(block_bbox, table_bbox)
            or _rect_intersection_area(block_bbox, table_bbox) / block_area >= 0.5
            for table_bbox in table_bboxes
        ):
            continue
        events.append(
            (
                float(block[1]),
                float(block[0]),
                int(block[5]),
                "text",
                {
                    "text": text_value,
                    "bbox": block_bbox,
                    "layout": layout_by_number.get(int(block[5])),
                },
            )
        )
    for table in tables:
        bbox = table.get("bbox") or [0, 0, 0, 0]
        events.append(
            (
                float(bbox[1]),
                float(bbox[0]),
                int(table.get("number", 0)),
                "table",
                table,
            )
        )
    for image_info in images:
        bbox = image_info.get("bbox") or [0, 0, 0, 0]
        events.append(
            (
                float(bbox[1]),
                float(bbox[0]),
                int(image_info.get("number", 0)),
                "image",
                image_info,
            )
        )
    kind_order = {"table": 0, "text": 1, "image": 2}
    events.sort(key=lambda item: (item[0], item[1], kind_order.get(item[3], 9), item[2]))

    title_key = _semantic_title_key(section_title)
    title_written = False
    rtl_layout = pdf_rtl_paragraph_layout_enabled()
    article_classes = "pdf-fast-semantic-page"
    article_attributes = ""
    if rtl_layout:
        article_classes += " pdf-rtl-layout"
        article_attributes = ' dir="rtl" data-pdf-rtl-layout="true"'
    parts = [
        '<!DOCTYPE html><html><head><meta charset="utf-8">',
        '<style>'
        '.pdf-fast-semantic-page img{max-width:100%;height:auto}'
        '.pdf-image,.pdf-vector-graphic{text-align:center;margin:1em 0}'
        '.pdf-table{border-collapse:collapse;width:100%;margin:1em 0}'
        '.pdf-table th,.pdf-table td{border:1px solid #777;padding:.35em .5em;'
        'vertical-align:top;text-align:left}'
        '.pdf-align-left{text-align:left}'
        '.pdf-align-center{text-align:center}'
        '.pdf-align-right{text-align:right}'
        '.pdf-align-justify{text-align:justify;text-justify:auto}'
        '.pdf-rtl-layout{direction:rtl}'
        '.pdf-rtl-layout p,.pdf-rtl-layout li,.pdf-rtl-layout td,'
        '.pdf-rtl-layout th{direction:rtl;unicode-bidi:plaintext}'
        '.pdf-rtl-layout p.pdf-align-justify{text-align-last:right}'
        '.pdf-fast-semantic-page a{text-decoration:underline}'
        '.pdf-links{font-size:.9em}'
        '</style>',
        '</head><body>',
        f'<article class="{article_classes}" data-pdf-page="{page_number}"'
        f'{article_attributes}>',
        f'<a id="page-{page_number}"></a>',
    ]
    for _y, _x, _number, kind, value in events:
        if kind == "table":
            parts.append(_semantic_table_html(value, links))
            continue
        if kind == "image":
            filename = html.escape(str(value.get("filename") or ""), quote=True)
            if filename:
                visual_link = _link_for_visual_bbox(value.get("bbox") or [], links)
                image_tag = (
                    f'<img src="images/{filename}" alt="PDF graphic" loading="lazy">'
                )
                if visual_link:
                    image_tag = (
                        f'<a {_link_attributes(visual_link["href"], bool(visual_link.get("external")))}>'
                        f"{image_tag}</a>"
                    )
                    visual_link["used"] = True
                figure_class = (
                    "pdf-vector-graphic"
                    if value.get("kind") == "vector"
                    else "pdf-image"
                )
                parts.append(
                    f'<figure class="{figure_class}">{image_tag}</figure>'
                )
            continue

        text_value = str(value.get("text") or "")
        escaped = _linked_text_html(text_value, value.get("bbox") or [], links)
        text_key = _semantic_title_key(text_value)
        source_alignment = _text_alignment(
            value.get("bbox") or [],
            float(page.rect.width),
            is_heading=True,
        )
        is_title = bool(
            not title_written
            and title_key
            and (text_key == title_key or title_key in text_key)
        )
        if is_title:
            alignment = resolve_pdf_header_alignment(source_alignment)
            parts.append(
                f'<h1 data-pdf-source-alignment="{source_alignment}" '
                f'style="text-align:{alignment}">{escaped}</h1>'
            )
            title_written = True
        else:
            source_alignment = _layout_paragraph_alignment(
                value.get("layout"),
                float(page.rect.width),
                column_bounds,
                text_value,
            )
            alignment = resolve_pdf_paragraph_alignment(
                source_alignment,
                text_value,
            )
            parts.append(
                f'<p class="pdf-align-{alignment}" '
                f'data-pdf-source-alignment="{source_alignment}" '
                f'style="text-align:{alignment}">{escaped}</p>'
            )
    unresolved = []
    seen_hrefs = set()
    for link in links:
        href = str(link.get("href") or "")
        if link.get("used") or not href or href in seen_hrefs:
            continue
        seen_hrefs.add(href)
        label = str(link.get("text") or "").strip() or href
        unresolved.append(
            f'<li><a {_link_attributes(href, bool(link.get("external")))}>'
            f"{html.escape(label)}</a></li>"
        )
    if unresolved:
        parts.append(
            '<aside class="pdf-links" aria-label="PDF links"><p>Links:</p><ul>'
            + "".join(unresolved)
            + "</ul></aside>"
        )
    parts.extend(["</article>", "</body></html>"])
    return "\n".join(parts)


def _xhtml_parts(xhtml: str) -> Tuple[str, str]:
    head_match = re.search(r"<head[^>]*>(.*?)</head>", xhtml or "", re.I | re.S)
    body_match = re.search(r"<body[^>]*>(.*?)</body>", xhtml or "", re.I | re.S)
    head = head_match.group(1) if head_match else ""
    body = body_match.group(1) if body_match else (xhtml or "")
    body = re.sub(r"<\?xml[^>]*>\s*", "", body, flags=re.I)
    body = re.sub(r"<!DOCTYPE[^>]*>\s*", "", body, flags=re.I)
    return head, body


def _layout_page_html(
    page,
    page_number: int,
    images: List[Dict],
    *,
    links: Optional[List[Dict]] = None,
) -> str:
    import fitz

    flags = int(getattr(fitz, "TEXTFLAGS_HTML", 199)) & ~int(
        getattr(fitz, "TEXT_PRESERVE_IMAGES", 4)
    )
    textpage = page.get_textpage(flags=flags)
    try:
        xhtml = page.get_text("html", textpage=textpage)
    except Exception:
        xhtml = textpage.extractHTML()
    head, body = _xhtml_parts(xhtml)
    body = re.sub(
        r'id=("|\')page\d+\1',
        f'id="mupdf-page-{page_number}"',
        body,
        count=1,
        flags=re.I,
    )

    image_tags = []
    for image_info in images:
        bbox = image_info.get("bbox") or [0, 0, 0, 0]
        x0, y0, x1, y1 = [float(value) for value in bbox]
        width = max(0.0, x1 - x0)
        height = max(0.0, y1 - y0)
        filename = html.escape(str(image_info.get("filename") or ""), quote=True)
        if not filename or width <= 0 or height <= 0:
            continue
        image_tags.append(
            f'<img class="pdf-fast-layout-image" src="images/{filename}" alt="PDF image" '
            f'style="position:absolute;left:{x0:.3f}pt;top:{y0:.3f}pt;'
            f'width:{width:.3f}pt;height:{height:.3f}pt">'
        )

    if links is None:
        links = _page_link_records(page)
    link_tags = []
    for link in links:
        x0, y0, x1, y1 = _rect_list(link.get("bbox") or [])
        width = max(0.0, x1 - x0)
        height = max(0.0, y1 - y0)
        href = str(link.get("href") or "")
        if not href or width <= 0 or height <= 0:
            continue
        link_tags.append(
            f'<a class="pdf-fast-layout-link" '
            f'{_link_attributes(href, bool(link.get("external")))} '
            f'style="position:absolute;left:{x0:.3f}pt;top:{y0:.3f}pt;'
            f'width:{width:.3f}pt;height:{height:.3f}pt" '
            f'aria-label="{html.escape(str(link.get("text") or href), quote=True)}"></a>'
        )

    rtl_layout = pdf_rtl_paragraph_layout_enabled()
    layout_classes = "pdf-fast-layout-page"
    layout_attributes = ""
    if rtl_layout:
        layout_classes += " pdf-rtl-layout"
        layout_attributes = ' dir="rtl" data-pdf-rtl-layout="true"'
    page_width = float(page.rect.width)
    page_height = float(page.rect.height)
    return "\n".join(
        [
            '<!DOCTYPE html><html><head><meta charset="utf-8">',
            head,
            '<style>.pdf-fast-layout-page{position:relative;margin:0 auto;overflow:hidden}'
            '.pdf-fast-layout-page>div{position:absolute;left:0;top:0;z-index:1}'
            '.pdf-fast-layout-page p{position:absolute;white-space:pre;margin:0}'
            '.pdf-rtl-layout{direction:rtl}'
            '.pdf-rtl-layout p{direction:rtl;unicode-bidi:plaintext}'
            '.pdf-fast-layout-image{z-index:0}'
            '.pdf-fast-layout-link{display:block;z-index:5;background:transparent}</style>',
            '</head><body>',
            f'<div class="{layout_classes}" data-pdf-page="{page_number}"'
            f'{layout_attributes} '
            f'style="width:{page_width:.3f}pt;height:{page_height:.3f}pt">',
            f'<a id="page-{page_number}"></a>',
            body,
            *image_tags,
            *link_tags,
            '</div></body></html>',
        ]
    )


def _page_cache_path(cache_root: Path, mode: str, page_number: int) -> Path:
    return cache_root / "pages" / mode / f"page_{page_number:06d}.json"


def _fast_pdf_settings(mode: str, extract_images: bool) -> Dict:
    return {
        "version": FAST_EXTRACTOR_VERSION,
        "mode": mode,
        "extract_images": bool(extract_images),
        "header_alignment": normalize_pdf_header_alignment(),
        "paragraph_alignment": normalize_pdf_paragraph_alignment(),
        "paragraph_justification": normalize_pdf_paragraph_justification(),
        "rtl_paragraph_layout": pdf_rtl_paragraph_layout_enabled(),
    }


def _cache_entry_is_usable(cache_root: Path, entry: Dict) -> bool:
    cache_file = cache_root / str(entry.get("cache_file") or "")
    if not cache_file.is_file():
        return False
    result = _load_json(cache_file, {}) or {}
    if result.get("extractor_version") != FAST_EXTRACTOR_VERSION:
        return False
    for image_info in result.get("images") or []:
        path = image_info.get("path")
        if path and not os.path.isfile(path):
            return False
    return bool(result.get("page_number"))


def _extract_page_range(args):
    (
        pdf_path,
        start_page,
        end_page,
        mode,
        extract_images,
        images_dir_text,
        cache_root_text,
        xref_map_dir_text,
        prior_pages,
        section_titles,
        *optional,
    ) = args
    stop_callback = optional[0] if optional else None

    import fitz

    images_dir = Path(images_dir_text)
    cache_root = Path(cache_root_text)
    xref_map_dir = Path(xref_map_dir_text)
    results = []
    stream_cache: Dict[int, str] = {}
    global _PROCESS_PDF_DOCUMENT, _PROCESS_PDF_PATH
    normalized_pdf_path = os.path.abspath(pdf_path)
    if (
        _PROCESS_PDF_DOCUMENT is not None
        and _PROCESS_PDF_PATH == normalized_pdf_path
        and not getattr(_PROCESS_PDF_DOCUMENT, "is_closed", True)
    ):
        doc = _PROCESS_PDF_DOCUMENT
        owns_document = False
        stream_cache = _PROCESS_STREAM_CACHE
    else:
        doc = fitz.open(pdf_path)
        owns_document = True
    try:
        for page_index in range(start_page, end_page):
            if _stop_requested(stop_callback):
                _signal_stop_file()
                raise PDFExtractionCancelled("PDF extraction cancelled")
            page_number = page_index + 1
            page = doc[page_index]
            signature = _page_source_signature(doc, page, stream_cache)
            prior = (prior_pages or {}).get(str(page_number)) or {}
            if prior.get("signature") == signature and _cache_entry_is_usable(cache_root, prior):
                results.append(
                    {
                        "page_number": page_number,
                        "signature": signature,
                        "cache_file": prior["cache_file"],
                        "reused": True,
                    }
                )
                continue

            links = _page_link_records(page)
            drawings = []
            if extract_images or mode == "fast_semantic":
                try:
                    drawings = page.get_drawings() or []
                except Exception:
                    drawings = []
            tables = (
                _extract_page_tables(page, drawings=drawings)
                if mode == "fast_semantic"
                else []
            )
            occurrences = _image_occurrences(page) if extract_images else []
            if extract_images and drawings:
                occurrences.extend(
                    _vector_graphic_occurrences(
                        page,
                        drawings,
                        excluded_bboxes=[
                            table.get("bbox") or [] for table in tables
                        ],
                        image_bboxes=[
                            occurrence.get("bbox") or [] for occurrence in occurrences
                        ],
                        starting_number=len(occurrences),
                    )
                )
            images = _materialize_images(doc, page, occurrences, images_dir, xref_map_dir)
            if mode == "fast_layout":
                page_html = _layout_page_html(
                    page,
                    page_number,
                    images,
                    links=links,
                )
            else:
                page_html = _semantic_page_html(
                    page,
                    page_number,
                    images,
                    str((section_titles or {}).get(str(page_index), "")),
                    tables=tables,
                    links=links,
                )

            cache_path = _page_cache_path(cache_root, mode, page_number)
            page_result = {
                "page_number": page_number,
                "extractor_version": FAST_EXTRACTOR_VERSION,
                "signature": signature,
                "html": page_html,
                "images": images,
            }
            _atomic_write_json(cache_path, page_result)
            results.append(
                {
                    "page_number": page_number,
                    "signature": signature,
                    "cache_file": str(cache_path.relative_to(cache_root)),
                    "reused": False,
                }
            )
    finally:
        if owns_document:
            doc.close()
    return results


def _load_results(
    cache_root: Path,
    page_entries: Dict[str, Dict],
    total_pages: int,
    stop_callback: Optional[Callable[[], bool]] = None,
):
    pages = []
    images_by_page: Dict[int, List[Dict]] = {}
    for page_number in range(1, total_pages + 1):
        if _stop_requested(stop_callback):
            _signal_stop_file()
            raise PDFExtractionCancelled("PDF extraction cancelled")
        entry = page_entries.get(str(page_number)) or {}
        cache_file = cache_root / str(entry.get("cache_file") or "")
        result = _load_json(cache_file, {}) or {}
        if not result:
            raise RuntimeError(f"Missing cached PDF page {page_number}")
        pages.append((page_number, str(result.get("html") or "")))
        page_images = result.get("images") or []
        if page_images:
            images_by_page[page_number - 1] = page_images
    return pages, images_by_page


def _legacy_image_slots(images: Sequence[Dict]) -> List[Dict]:
    """Return the one-slot-per-PDF-resource order used by legacy filenames."""
    slots: List[Dict] = []
    seen = set()
    for position, image_info in enumerate(images or []):
        resource_index = image_info.get("resource_index")
        xref = int(image_info.get("xref") or 0)
        if resource_index is not None:
            key = ("resource", int(resource_index))
        elif xref:
            key = ("xref", xref)
        else:
            key = ("placement", position)
        if key in seen:
            continue
        seen.add(key)
        slots.append(image_info)
    return slots


def _cached_pdf_page_image_slots(cache_root: Path) -> Dict[int, List[Dict]]:
    """Index page image slots from normal and targeted fast-PDF caches."""
    candidates: List[Path] = []
    pages_root = cache_root / "pages"
    if pages_root.is_dir():
        candidates.extend(pages_root.glob("*/*.json"))
    targeted_root = cache_root / "targeted_images"
    if targeted_root.is_dir():
        candidates.extend(targeted_root.glob("*.json"))

    indexed: Dict[int, List[Dict]] = {}
    for cache_path in sorted(candidates):
        payload = _load_json(cache_path, {}) or {}
        try:
            page_number = int(payload.get("page_number") or 0)
        except (TypeError, ValueError):
            continue
        images = payload.get("images") if isinstance(payload, dict) else None
        if page_number <= 0 or not isinstance(images, list):
            continue
        slots = _legacy_image_slots(
            [item for item in images if isinstance(item, dict)]
        )
        if slots and page_number not in indexed:
            indexed[page_number] = slots
    return indexed


def _rewrite_pdf_image_references(markup: str, rename_map: Dict[str, str]) -> str:
    """Rewrite local image basenames in HTML without changing path prefixes."""
    if not markup or not rename_map:
        return markup
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(markup, "html.parser")
    changed = False
    attributes = (("img", "src"), ("object", "data"), ("video", "poster"))
    for tag_name, attribute in attributes:
        for node in soup.find_all(tag_name):
            original = str(node.get(attribute) or "")
            if not original or original.startswith(("data:", "http://", "https://")):
                continue
            clean = original.split("?", 1)[0].split("#", 1)[0].replace("\\", "/")
            basename = clean.rsplit("/", 1)[-1]
            replacement = rename_map.get(basename)
            if not replacement or replacement == basename:
                continue
            node[attribute] = re.sub(
                re.escape(basename) + r"(?=([?#].*)?$)",
                replacement,
                original,
                count=1,
            )
            changed = True
    return str(soup) if changed else markup


def _rewrite_pdf_image_cache(
    cache_root: Path,
    images_dir: Path,
    rename_map: Dict[str, str],
) -> int:
    """Keep page/xref caches reusable after public image files are renamed."""
    updated = 0
    if not cache_root.is_dir() or not rename_map:
        return updated

    def rewrite(value):
        nonlocal changed
        if isinstance(value, dict):
            for key, item in list(value.items()):
                if key == "filename" and isinstance(item, str):
                    basename = os.path.basename(item)
                    replacement = rename_map.get(basename)
                    if replacement and replacement != basename:
                        value[key] = replacement
                        changed = True
                elif key == "path" and isinstance(item, str):
                    basename = os.path.basename(item)
                    replacement = rename_map.get(basename)
                    if replacement and replacement != basename:
                        value[key] = str(images_dir / replacement)
                        changed = True
                elif key == "html" and isinstance(item, str):
                    rewritten = _rewrite_pdf_image_references(item, rename_map)
                    if rewritten != item:
                        value[key] = rewritten
                        changed = True
                else:
                    rewrite(item)
        elif isinstance(value, list):
            for item in value:
                rewrite(item)

    for cache_path in cache_root.rglob("*.json"):
        payload = _load_json(cache_path, None)
        if payload is None:
            continue
        changed = False
        rewrite(payload)
        if changed:
            _atomic_write_json(cache_path, payload)
            updated += 1
    return updated


def apply_pdf_image_rename_logic(
    chapters: List[Dict],
    output_dir: str,
    *,
    word_count_dir: Optional[str] = None,
) -> List[Dict]:
    """Apply the normal chapter-based image names to fast-PDF resources.

    Fast extraction first uses content hashes so parallel workers can safely
    deduplicate shared PDF resources.  This post-pass assigns the same public
    ``<chapter>_img_N`` names used by EPUB extraction, rewrites every cached
    reference, and only then removes the internal hash filenames.
    """
    output_path = Path(output_dir).resolve()
    images_dir = output_path / "images"
    if not chapters or not images_dir.is_dir():
        return chapters

    existing_files = {
        item.name: item for item in images_dir.iterdir() if item.is_file()
    }

    cache_root = output_path / ".pdf_extraction_cache"
    page_slots = _cached_pdf_page_image_slots(cache_root)
    canonical_cache_names = {
        os.path.basename(str(image_info.get("filename") or ""))
        for slots in page_slots.values()
        for image_info in slots
        if image_info.get("filename")
    }
    if word_count_dir:
        mirrored_images = Path(word_count_dir) / "images"
        for canonical_name in canonical_cache_names:
            if canonical_name in existing_files:
                continue
            mirrored_path = mirrored_images / canonical_name
            target_path = images_dir / canonical_name
            if mirrored_path.is_file():
                shutil.copy2(mirrored_path, target_path)
                existing_files[canonical_name] = target_path
    if not existing_files:
        return chapters

    map_path = output_path / "image_rename_map.json"
    saved_map = _load_json(map_path, {}) or {}
    if not isinstance(saved_map, dict):
        saved_map = {}
    saved_map = {
        os.path.basename(str(old_name)): os.path.basename(str(new_name))
        for old_name, new_name in saved_map.items()
        if old_name and new_name
    }
    assignments: Dict[str, str] = {}
    source_targets: Dict[str, str] = {}
    claimed_sources = set()

    # Older EPUB-only response repair could rename a canonical PDF image a
    # second time (canonical -> stable response ID). Reverse those chains when
    # the page cache identifies the original canonical name.
    for canonical_name, response_name in list(saved_map.items()):
        if (
            canonical_name in canonical_cache_names
            and canonical_name != response_name
            and _PDF_FRIENDLY_IMAGE_RE.fullmatch(canonical_name)
            and _PDF_FRIENDLY_IMAGE_RE.fullmatch(response_name)
            and response_name in existing_files
            and canonical_name in existing_files
        ):
            assignments[response_name] = canonical_name
            source_targets[response_name] = canonical_name
            saved_map.pop(canonical_name, None)

    from bs4 import BeautifulSoup

    for chapter in chapters:
        body = str(chapter.get("body") or "")
        filename = str(chapter.get("filename") or "")
        chapter_stem = Path(filename).stem or f"pdf_section_{chapter.get('num', 0)}"
        if not body:
            continue
        try:
            soup = BeautifulSoup(body, "html.parser")
        except Exception:
            continue
        image_number = 1
        chapter_seen = set()
        for node in soup.find_all("img"):
            src = str(node.get("src") or "")
            basename = (
                src.split("?", 1)[0]
                .split("#", 1)[0]
                .replace("\\", "/")
                .rsplit("/", 1)[-1]
            )
            if not basename or basename in chapter_seen:
                continue
            chapter_seen.add(basename)

            source_name = basename if basename in existing_files else ""
            legacy_match = _PDF_LEGACY_IMAGE_RE.fullmatch(basename)
            if legacy_match:
                page_number = int(legacy_match.group(1))
                slot_number = int(legacy_match.group(2))
                slots = page_slots.get(page_number) or []
                if 1 <= slot_number <= len(slots):
                    source_name = os.path.basename(
                        str(slots[slot_number - 1].get("filename") or "")
                    )
            if not source_name or source_name not in existing_files:
                continue
            if _PDF_FRIENDLY_IMAGE_RE.fullmatch(source_name):
                assignments[basename] = source_name
                claimed_sources.add(source_name)
                continue
            preferred_name = saved_map.get(basename) or saved_map.get(source_name)
            if preferred_name and _PDF_FRIENDLY_IMAGE_RE.fullmatch(preferred_name):
                assignments[basename] = preferred_name
                assignments[source_name] = preferred_name
                if source_name != preferred_name:
                    source_targets[source_name] = preferred_name
                claimed_sources.add(source_name)
                continue

            target_name = source_targets.get(source_name)
            if not target_name:
                extension = Path(source_name).suffix
                target_name = f"{chapter_stem}_img_{image_number}{extension}"
                while target_name in source_targets.values() or (
                    target_name in existing_files and target_name != source_name
                ):
                    image_number += 1
                    target_name = f"{chapter_stem}_img_{image_number}{extension}"
                source_targets[source_name] = target_name
                image_number += 1
            assignments[basename] = target_name
            assignments[source_name] = target_name
            claimed_sources.add(source_name)

    cover_number = 0
    for source_name in sorted(existing_files):
        if source_name in claimed_sources or not _PDF_HASH_IMAGE_RE.fullmatch(source_name):
            continue
        extension = Path(source_name).suffix
        target_name = f"{cover_number}_Cover{extension}"
        while target_name in source_targets.values() or target_name in existing_files:
            cover_number += 1
            target_name = f"{cover_number}_Cover{extension}"
        source_targets[source_name] = target_name
        assignments[source_name] = target_name
        cover_number += 1

    if not source_targets and not any(
        old_name != target_name for old_name, target_name in assignments.items()
    ):
        return chapters

    # Create all friendly aliases first so no cache or HTML reference can be
    # left pointing at a missing file if the operation is interrupted.
    for source_name, target_name in source_targets.items():
        source_path = existing_files.get(source_name)
        target_path = images_dir / target_name
        if source_path and source_path.is_file() and not target_path.is_file():
            shutil.copy2(source_path, target_path)

    cache_updates = _rewrite_pdf_image_cache(cache_root, images_dir, assignments)

    for chapter in chapters:
        body = str(chapter.get("body") or "")
        rewritten_body = _rewrite_pdf_image_references(body, assignments)
        chapter["body"] = rewritten_body
        # Progress hashes must describe the final translation payload.  The
        # old code hashed the internal pdfimg_<hash> references and then
        # renamed those references afterwards, so loading the same split cache
        # produced a different hash and incorrectly invalidated every
        # completed bookmark on the next run.
        chapter["content_hash"] = hashlib.sha256(
            rewritten_body.encode("utf-8")
        ).hexdigest()
        chapter["file_size"] = len(rewritten_body)

    disk_updates = 0
    html_roots = [output_path]
    if word_count_dir:
        html_roots.append(Path(word_count_dir))
    seen_paths = set()
    for html_root in html_roots:
        if not html_root.is_dir():
            continue
        for html_path in html_root.rglob("*.htm*"):
            if cache_root in html_path.parents or html_path in seen_paths:
                continue
            seen_paths.add(html_path)
            try:
                original = html_path.read_text(encoding="utf-8")
                rewritten = _rewrite_pdf_image_references(original, assignments)
                if rewritten != original:
                    html_path.write_text(rewritten, encoding="utf-8")
                    disk_updates += 1
            except (OSError, UnicodeError):
                continue

    if word_count_dir:
        mirrored_images = Path(word_count_dir) / "images"
        mirrored_images.mkdir(parents=True, exist_ok=True)
        for target_name in set(source_targets.values()):
            source_path = images_dir / target_name
            target_path = mirrored_images / target_name
            if source_path.is_file() and not target_path.is_file():
                shutil.copy2(source_path, target_path)

    # Preserve any unrelated/older mapping entries and add both the legacy
    # page aliases and the internal hash aliases used by fast extraction.
    saved_map.update(assignments)
    with map_path.open("w", encoding="utf-8") as stream:
        json.dump(saved_map, stream, ensure_ascii=False, indent=2)

    for old_name, target_name in assignments.items():
        source_path = images_dir / old_name
        if old_name != target_name and (images_dir / target_name).is_file():
            try:
                source_path.unlink()
            except OSError:
                pass
        if word_count_dir:
            mirrored_old = Path(word_count_dir) / "images" / old_name
            if old_name != target_name and mirrored_old.is_file():
                try:
                    mirrored_old.unlink()
                except OSError:
                    pass

    print(
        f"PDF image rename: {len(source_targets)} asset(s) now use chapter names; "
        f"updated {disk_updates} HTML file(s) and {cache_updates} cache file(s)"
    )
    return chapters


def extract_pdf_page_range_for_reader(
    pdf_path: str,
    output_dir: str,
    *,
    start_page: int,
    end_page: int,
    mode: str = "fast_semantic",
    extract_images: bool = True,
    section_title: str = "",
):
    """Extract one inclusive 1-based PDF range for the HTML reader.

    Unlike :func:`extract_pdf_fast`, this never scans unrelated pages and does
    not rewrite the full-document manifest.  It still uses the same page
    signature and image-deduplication caches, so a range extracted during the
    original translation is normally an immediate cache hit.
    """
    if mode not in FAST_MODES:
        raise ValueError(f"Unsupported fast PDF mode: {mode}")
    import fitz

    output_path = Path(output_dir)
    images_dir = output_path / "images"
    cache_root = output_path / ".pdf_extraction_cache"
    images_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
    start = max(1, int(start_page))
    end = min(total_pages, int(end_page))
    if start > end:
        raise ValueError(
            f"PDF page range {start_page}-{end_page} is outside a {total_pages}-page document"
        )

    manifest_path = cache_root / f"manifest_{mode}.json"
    previous = _load_json(manifest_path, {}) or {}
    expected_settings = _fast_pdf_settings(mode, bool(extract_images))
    previous_pages = (
        previous.get("pages")
        if isinstance(previous, dict) and previous.get("settings") == expected_settings
        else {}
    )
    previous_pages = previous_pages if isinstance(previous_pages, dict) else {}
    relevant_prior = {
        str(page_number): previous_pages.get(str(page_number), {})
        for page_number in range(start, end + 1)
    }
    stat = os.stat(pdf_path)
    source_token = hashlib.sha256(
        f"{os.path.abspath(pdf_path)}\0{stat.st_size}\0{stat.st_mtime_ns}".encode(
            "utf-8", "replace"
        )
    ).hexdigest()[:24]
    task = (
        pdf_path,
        start - 1,
        end,
        mode,
        bool(extract_images),
        str(images_dir),
        str(cache_root),
        str(cache_root / "xref_maps" / source_token),
        relevant_prior,
        {str(start - 1): str(section_title or "")},
    )
    extracted = _extract_page_range(task)
    page_items = []
    for item in sorted(extracted, key=lambda value: int(value["page_number"])):
        cache_file = cache_root / str(item.get("cache_file") or "")
        result = _load_json(cache_file, {}) or {}
        if not result.get("page_number"):
            raise RuntimeError(f"Missing cached PDF page {item.get('page_number')}")
        page_items.append((int(result["page_number"]), str(result.get("html") or "")))
    return page_items


def _extract_pdf_fast_impl(
    pdf_path: str,
    output_dir: str,
    *,
    mode: str = "fast_semantic",
    extract_images: bool = True,
    page_by_page: bool = False,
    stop_callback: Optional[Callable[[], bool]] = None,
    progress_monitor: Optional[_ExtractionProgressMonitor] = None,
):
    """Extract a PDF using the modern cached pipeline."""

    if mode not in FAST_MODES:
        raise ValueError(f"Unsupported fast PDF mode: {mode}")

    import fitz

    started = time.perf_counter()
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    cache_root = output_path / ".pdf_extraction_cache"
    images_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    source_hash = _sha256_file(pdf_path)
    source_stat = os.stat(pdf_path)
    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        toc_entries = doc.get_toc(simple=True) or []
    if progress_monitor:
        progress_monitor.configure(total_pages, 0, "checking the page cache")
    if _stop_requested(stop_callback):
        _signal_stop_file()
        raise PDFExtractionCancelled("PDF extraction cancelled")
    outline_hash = _outline_digest(toc_entries)

    settings = _fast_pdf_settings(mode, bool(extract_images))
    manifest_path = cache_root / f"manifest_{mode}.json"
    previous = _load_json(manifest_path, {}) or {}
    previous_pages = previous.get("pages") if previous.get("settings") == settings else {}
    previous_pages = previous_pages if isinstance(previous_pages, dict) else {}

    exact_source = (
        previous.get("settings") == settings
        and previous.get("source", {}).get("sha256") == source_hash
        and int(previous.get("page_count") or -1) == total_pages
    )
    exact_cache_usable = exact_source
    if exact_cache_usable:
        for page_number in range(1, total_pages + 1):
            if _stop_requested(stop_callback):
                _signal_stop_file()
                raise PDFExtractionCancelled("PDF extraction cancelled")
            if not _cache_entry_is_usable(
                cache_root,
                previous_pages.get(str(page_number), {}),
            ):
                exact_cache_usable = False
                break
    if exact_cache_usable:
        if progress_monitor:
            progress_monitor.configure(total_pages, 1, "loading cached pages")
        pages, images_by_page = _load_results(
            cache_root,
            previous_pages,
            total_pages,
            stop_callback,
        )
        previous["outline_digest"] = outline_hash
        previous["stats"] = {
            "reused_pages": total_pages,
            "extracted_pages": 0,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        _atomic_write_json(manifest_path, previous)
        if progress_monitor:
            progress_monitor.update(pages=total_pages, jobs=1, phase="complete")
        print(f"Fast PDF cache hit: reused {total_pages}/{total_pages} pages")
        if page_by_page:
            return pages, images_by_page
        return "\n\n".join(page_html for _, page_html in pages), images_by_page

    try:
        chunk_pages = max(1, min(64, int(os.environ.get("PDF_FAST_CHUNK_PAGES", "16"))))
    except ValueError:
        chunk_pages = 16
    jobs, section_titles = _bookmark_jobs(toc_entries, total_pages, chunk_pages)
    if progress_monitor:
        progress_monitor.configure(total_pages, len(jobs), "extracting pages")
    worker_count = _fast_pdf_worker_count(total_pages, len(jobs))

    xref_map_dir = cache_root / "xref_maps" / source_hash[:24]
    tasks = []
    for start_page, end_page in jobs:
        relevant_prior = {
            str(page_number): previous_pages.get(str(page_number), {})
            for page_number in range(start_page + 1, end_page + 1)
        }
        tasks.append(
            (
                pdf_path,
                start_page,
                end_page,
                mode,
                bool(extract_images),
                str(images_dir),
                str(cache_root),
                str(xref_map_dir),
                relevant_prior,
                {str(key): value for key, value in section_titles.items()},
            )
        )

    print(
        f"Fast PDF extraction: {total_pages} pages, {len(jobs)} bookmark-aware job(s), "
        f"{worker_count} worker(s), mode={mode}"
    )
    if worker_count > 1:
        print(
            f"Fast PDF scheduler: running up to {worker_count} bookmark jobs "
            "concurrently"
        )
    extracted_entries = []

    def _record_job_result(result, start_page, end_page):
        extracted_entries.extend(result)
        completed_pages = len(result)
        if progress_monitor:
            progress_monitor.update(pages=completed_pages, jobs=1)
            done_pages = progress_monitor.completed_pages
            done_jobs = progress_monitor.completed_jobs
        else:
            done_pages = len(extracted_entries)
            done_jobs = 0
        percent = int(done_pages * 100 / total_pages) if total_pages else 100
        reused = sum(1 for entry in result if entry.get("reused"))
        print(
            f"📄 Fast PDF progress: {done_pages}/{total_pages} pages ({percent}%), "
            f"job pages {start_page + 1}-{end_page} complete, "
            f"{reused}/{completed_pages} page(s) reused"
        )

    if worker_count == 1:
        global _PROCESS_PDF_DOCUMENT, _PROCESS_PDF_PATH, _PROCESS_STREAM_CACHE
        _PROCESS_PDF_PATH = os.path.abspath(pdf_path)
        _PROCESS_PDF_DOCUMENT = fitz.open(pdf_path)
        _PROCESS_STREAM_CACHE = {}
        try:
            for task in tasks:
                if _stop_requested(stop_callback):
                    _signal_stop_file()
                    raise PDFExtractionCancelled("PDF extraction cancelled")
                direct_task = (*task, stop_callback)
                result = _extract_page_range(direct_task)
                _record_job_result(result, task[1], task[2])
        finally:
            _PROCESS_PDF_DOCUMENT.close()
            _PROCESS_PDF_DOCUMENT = None
            _PROCESS_PDF_PATH = ""
            _PROCESS_STREAM_CACHE = {}
    else:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_initialize_pdf_worker,
            initargs=(pdf_path,),
        )
        futures = {
            executor.submit(_extract_page_range, task): (task[1], task[2])
            for task in tasks
        }
        pending = set(futures)
        cancelled = False
        try:
            while pending:
                if _stop_requested(stop_callback):
                    cancelled = True
                    _signal_stop_file()
                    for future in pending:
                        future.cancel()
                    raise PDFExtractionCancelled("PDF extraction cancelled")
                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=0.25,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    start_page, end_page = futures[future]
                    result = future.result()
                    _record_job_result(result, start_page, end_page)
        except PDFExtractionCancelled:
            cancelled = True
            _signal_stop_file()
            for future in pending:
                future.cancel()
            raise
        finally:
            # Keep the stop file alive until every running worker reaches its
            # next cooperative cancellation boundary. Otherwise removing it in
            # the outer cleanup can let a late worker continue after Stop.
            executor.shutdown(wait=True, cancel_futures=cancelled)

    page_entries = {
        str(entry["page_number"]): {
            "signature": entry["signature"],
            "cache_file": entry["cache_file"],
        }
        for entry in extracted_entries
    }
    if len(page_entries) != total_pages:
        raise RuntimeError(
            f"Fast PDF extraction completed {len(page_entries)}/{total_pages} pages"
        )

    reused_pages = sum(1 for entry in extracted_entries if entry.get("reused"))
    manifest = {
        "settings": settings,
        "source": {
            "path": os.path.abspath(pdf_path),
            "sha256": source_hash,
            "size": int(source_stat.st_size),
            "mtime_ns": int(source_stat.st_mtime_ns),
        },
        "page_count": total_pages,
        "outline_digest": outline_hash,
        "pages": page_entries,
        "stats": {
            "reused_pages": reused_pages,
            "extracted_pages": total_pages - reused_pages,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
    }
    _atomic_write_json(manifest_path, manifest)
    if progress_monitor:
        progress_monitor.update(phase="assembling bookmark sections")
    pages, images_by_page = _load_results(
        cache_root,
        page_entries,
        total_pages,
        stop_callback,
    )
    print(
        f"Fast PDF extraction complete: {total_pages - reused_pages} extracted, "
        f"{reused_pages} reused"
    )
    if page_by_page:
        return pages, images_by_page
    return "\n\n".join(page_html for _, page_html in pages), images_by_page


def extract_pdf_fast(
    pdf_path: str,
    output_dir: str,
    *,
    mode: str = "fast_semantic",
    extract_images: bool = True,
    page_by_page: bool = False,
    stop_callback: Optional[Callable[[], bool]] = None,
):
    """Extract a PDF with visible progress and cooperative cancellation."""

    existing_stop_file = os.environ.get("PDF_EXTRACTION_STOP_FILE", "").strip()
    owns_stop_file = not existing_stop_file
    stop_file = Path(existing_stop_file) if existing_stop_file else (
        Path(output_dir) / ".pdf_extraction_cache" / "active_extraction.stop"
    )
    if owns_stop_file:
        try:
            stop_file.unlink()
        except OSError:
            pass
        os.environ["PDF_EXTRACTION_STOP_FILE"] = str(stop_file)

    monitor = _ExtractionProgressMonitor(stop_callback, stop_file, owns_stop_file)
    try:
        print("📄 Fast PDF phase: fingerprinting source and reading bookmarks")
        result = _extract_pdf_fast_impl(
            pdf_path,
            output_dir,
            mode=mode,
            extract_images=extract_images,
            page_by_page=page_by_page,
            stop_callback=stop_callback,
            progress_monitor=monitor,
        )
        return result
    except PDFExtractionCancelled:
        print("🛑 Fast PDF extraction stopped by user")
        raise
    finally:
        monitor.close()
        if owns_stop_file:
            os.environ.pop("PDF_EXTRACTION_STOP_FILE", None)


def ensure_pdf_page_images(
    pdf_path: str,
    output_dir: str,
    page_numbers: Sequence[int],
    *,
    stop_callback: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[int, List[Dict]]:
    """Materialize images only for specifically requested 1-based pages.

    The PDF compiler uses this to repair legacy ``page_N_img_M`` references.
    It never walks intervening pages, so a missing image on page 700 does not
    cause pages 1-699 to be scanned.
    """

    def report(message: str) -> None:
        if progress_callback:
            progress_callback(message)
        else:
            print(message)

    import fitz

    output_path = Path(output_dir)
    images_dir = output_path / "images"
    cache_root = output_path / ".pdf_extraction_cache"
    targeted_root = cache_root / "targeted_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    targeted_root.mkdir(parents=True, exist_ok=True)

    source_stat = os.stat(pdf_path)
    source_token = hashlib.sha256(
        f"{os.path.abspath(pdf_path)}\0{source_stat.st_size}\0{source_stat.st_mtime_ns}".encode(
            "utf-8", "replace"
        )
    ).hexdigest()[:24]
    xref_map_dir = cache_root / "xref_maps" / f"targeted_{source_token}"

    requested = sorted({int(page) for page in page_numbers if int(page) > 0})
    results: Dict[int, List[Dict]] = {}
    with fitz.open(pdf_path) as doc:
        requested = [page for page in requested if page <= len(doc)]
        total = len(requested)
        for position, page_number in enumerate(requested, 1):
            if _stop_requested(stop_callback):
                _signal_stop_file()
                raise PDFExtractionCancelled("PDF image recovery cancelled")

            targeted_path = targeted_root / f"{source_token}_page_{page_number:06d}.json"
            cached = _load_json(targeted_path, {}) or {}
            cached_images = cached.get("images") if isinstance(cached, dict) else []
            if cached_images and all(
                os.path.isfile(str(item.get("path") or ""))
                for item in cached_images
            ):
                results[page_number] = cached_images
                reused = True
            else:
                page = doc[page_number - 1]
                occurrences = _image_occurrences(page)
                try:
                    drawings = page.get_drawings() or []
                except Exception:
                    drawings = []
                tables = _extract_page_tables(page, drawings=drawings)
                occurrences.extend(
                    _vector_graphic_occurrences(
                        page,
                        drawings,
                        excluded_bboxes=[
                            table.get("bbox") or [] for table in tables
                        ],
                        image_bboxes=[
                            occurrence.get("bbox") or [] for occurrence in occurrences
                        ],
                        starting_number=len(occurrences),
                    )
                )
                page_images = _materialize_images(
                    doc,
                    page,
                    occurrences,
                    images_dir,
                    xref_map_dir,
                )
                results[page_number] = page_images
                _atomic_write_json(
                    targeted_path,
                    {
                        "source_token": source_token,
                        "page_number": page_number,
                        "images": page_images,
                    },
                )
                reused = False

            percent = int(position * 100 / total) if total else 100
            report(
                f"🖼️ PDF image recovery: {position}/{total} requested page(s) "
                f"({percent}%), page {page_number}, "
                f"{len(results[page_number])} image(s)"
                + (" reused" if reused else " extracted")
            )
    return results


def extract_pdf_images_deduplicated(pdf_path: str, output_dir: str) -> Dict[int, List[Dict]]:
    """Extract every displayed image once and retain all page placements.

    This function is also used by the legacy and Vision OCR paths so those
    callers benefit from xref/content deduplication without changing their
    surrounding behavior.
    """

    import fitz

    output_path = Path(output_dir)
    images_dir = output_path / "images"
    cache_root = output_path / ".pdf_extraction_cache"
    source_hash = _sha256_file(pdf_path)
    xref_map_dir = cache_root / "xref_maps" / source_hash[:24]
    images_dir.mkdir(parents=True, exist_ok=True)
    result: Dict[int, List[Dict]] = {}
    last_decile = -1
    with fitz.open(pdf_path) as doc:
        for page_index in range(len(doc)):
            if _stop_requested():
                break
            page = doc[page_index]
            occurrences = _image_occurrences(page)
            images = _materialize_images(doc, page, occurrences, images_dir, xref_map_dir)
            if images:
                result[page_index] = images
            if len(doc):
                percent = int((page_index + 1) * 100 / len(doc))
                decile = min(10, percent // 10)
                if page_index == 0 or page_index + 1 == len(doc) or decile > last_decile:
                    last_decile = decile
                    print(f"    Scanning images: {percent}% ({page_index + 1}/{len(doc)} pages)")
    placements = sum(len(items) for items in result.values())
    unique_assets = {
        item.get("filename")
        for items in result.values()
        for item in items
        if item.get("filename")
    }
    print(
        f"Extracted {placements} image placement(s) as "
        f"{len(unique_assets)} unique asset(s)"
    )
    return result


def cached_pdf_image_paths(pdf_path: str, output_dir: str) -> List[str]:
    """Return valid fast-cache images for the current source PDF, if present."""

    cache_root = Path(output_dir) / ".pdf_extraction_cache"
    try:
        source_hash = _sha256_file(pdf_path)
    except Exception:
        return []
    paths = set()
    for mode in sorted(FAST_MODES):
        manifest = _load_json(cache_root / f"manifest_{mode}.json", {}) or {}
        if manifest.get("source", {}).get("sha256") != source_hash:
            continue
        for entry in (manifest.get("pages") or {}).values():
            cache_file = cache_root / str(entry.get("cache_file") or "")
            page_result = _load_json(cache_file, {}) or {}
            for image_info in page_result.get("images") or []:
                path = image_info.get("path")
                if path and os.path.isfile(path):
                    paths.add(os.path.abspath(path))
    return sorted(paths)
