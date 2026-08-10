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
import hashlib
import html
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple


FAST_EXTRACTOR_VERSION = 2
FAST_MODES = {"fast_semantic", "fast_layout"}

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
    map_key = f"xref_{xref}" if xref else f"inline_{source_digest or occurrence['index']}"
    map_path = xref_map_dir / f"{map_key}.json"
    cached = _load_json(map_path, {}) or {}
    cached_filename = cached.get("filename")
    if cached_filename and (images_dir / cached_filename).is_file():
        return cached

    image_bytes = None
    extension = "png"
    width = int(occurrence.get("width") or 0)
    height = int(occurrence.get("height") or 0)
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
        item = dict(occurrence)
        item.update(asset)
        materialized.append(item)
    return materialized


def _text_alignment(bbox: Sequence[float], page_width: float) -> str:
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
        min(left, right) >= page_width * 0.18
        and abs(left - right) <= max(12.0, page_width * 0.04)
    ):
        return "center"
    if right < page_width * 0.08 and left > page_width * 0.2:
        return "right"
    return "left"


def _semantic_page_html(page, page_number: int, images: List[Dict], section_title: str) -> str:
    import fitz

    flags = int(getattr(fitz, "TEXTFLAGS_BLOCKS", 195)) & ~int(
        getattr(fitz, "TEXT_PRESERVE_IMAGES", 4)
    )
    try:
        blocks = page.get_text("blocks", sort=True, flags=flags) or []
    except TypeError:
        blocks = page.get_text("blocks", sort=True) or []

    events = []
    for block in blocks:
        if len(block) < 7 or int(block[6]) != 0:
            continue
        text_value = " ".join(str(block[4] or "").split())
        if not text_value:
            continue
        events.append(
            (
                float(block[1]),
                float(block[0]),
                int(block[5]),
                "text",
                {
                    "text": text_value,
                    "bbox": [float(block[index]) for index in range(4)],
                },
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
    events.sort(key=lambda item: (item[0], item[1], item[2], item[3] != "text"))

    title_normalized = " ".join((section_title or "").casefold().split())
    title_written = False
    parts = [
        '<!DOCTYPE html><html><head><meta charset="utf-8">',
        '<style>.pdf-fast-semantic-page img{max-width:100%;height:auto}.pdf-image{text-align:center;margin:1em 0}</style>',
        '</head><body>',
        f'<article class="pdf-fast-semantic-page" data-pdf-page="{page_number}">',
        f'<a id="page-{page_number}"></a>',
    ]
    for _y, _x, _number, kind, value in events:
        if kind == "image":
            filename = html.escape(str(value.get("filename") or ""), quote=True)
            if filename:
                parts.append(
                    f'<figure class="pdf-image"><img src="images/{filename}" '
                    f'alt="PDF image" loading="lazy"></figure>'
                )
            continue

        text_value = str(value.get("text") or "")
        escaped = html.escape(text_value)
        normalized = " ".join(text_value.casefold().split())
        alignment = _text_alignment(value.get("bbox") or [], float(page.rect.width))
        is_title = bool(
            not title_written
            and title_normalized
            and (normalized == title_normalized or title_normalized in normalized)
        )
        if is_title:
            parts.append(f'<h1 style="text-align:{alignment}">{escaped}</h1>')
            title_written = True
        else:
            parts.append(
                f'<p class="pdf-align-{alignment}" '
                f'style="text-align:{alignment}">{escaped}</p>'
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


def _layout_page_html(page, page_number: int, images: List[Dict]) -> str:
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

    page_width = float(page.rect.width)
    page_height = float(page.rect.height)
    return "\n".join(
        [
            '<!DOCTYPE html><html><head><meta charset="utf-8">',
            head,
            '<style>.pdf-fast-layout-page{position:relative;margin:0 auto;overflow:hidden}'
            '.pdf-fast-layout-page>div{position:absolute;left:0;top:0;z-index:1}'
            '.pdf-fast-layout-page p{position:absolute;white-space:pre;margin:0}'
            '.pdf-fast-layout-image{z-index:0}</style>',
            '</head><body>',
            f'<div class="pdf-fast-layout-page" data-pdf-page="{page_number}" '
            f'style="width:{page_width:.3f}pt;height:{page_height:.3f}pt">',
            f'<a id="page-{page_number}"></a>',
            body,
            *image_tags,
            '</div></body></html>',
        ]
    )


def _page_cache_path(cache_root: Path, mode: str, page_number: int) -> Path:
    return cache_root / "pages" / mode / f"page_{page_number:06d}.json"


def _cache_entry_is_usable(cache_root: Path, entry: Dict) -> bool:
    cache_file = cache_root / str(entry.get("cache_file") or "")
    if not cache_file.is_file():
        return False
    result = _load_json(cache_file, {}) or {}
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

            occurrences = _image_occurrences(page) if extract_images else []
            images = _materialize_images(doc, page, occurrences, images_dir, xref_map_dir)
            if mode == "fast_layout":
                page_html = _layout_page_html(page, page_number, images)
            else:
                page_html = _semantic_page_html(
                    page,
                    page_number,
                    images,
                    str((section_titles or {}).get(str(page_index), "")),
                )

            cache_path = _page_cache_path(cache_root, mode, page_number)
            page_result = {
                "page_number": page_number,
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
    previous_pages = previous.get("pages") if isinstance(previous, dict) else {}
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

    settings = {
        "version": FAST_EXTRACTOR_VERSION,
        "mode": mode,
        "extract_images": bool(extract_images),
    }
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
