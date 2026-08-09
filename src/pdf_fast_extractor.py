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
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


FAST_EXTRACTOR_VERSION = 2
FAST_MODES = {"fast_semantic", "fast_layout"}


def _stop_requested() -> bool:
    stop_file = os.environ.get("PDF_EXTRACTION_STOP_FILE", "")
    return bool(stop_file and os.path.exists(stop_file))


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
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


def _image_occurrences(page) -> List[Dict]:
    try:
        raw_items = page.get_image_info(hashes=True, xrefs=True) or []
    except Exception:
        raw_items = []

    items = []
    for index, item in enumerate(raw_items):
        digest = item.get("digest")
        if isinstance(digest, bytes):
            digest = digest.hex()
        items.append(
            {
                "index": index,
                "number": int(item.get("number", index)),
                "xref": int(item.get("xref") or 0),
                "source_digest": str(digest or ""),
                "bbox": _rect_list(item.get("bbox")),
                "width": int(item.get("width") or 0),
                "height": int(item.get("height") or 0),
                "has_mask": bool(item.get("has-mask", False)),
            }
        )
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
    if abs(left - right) <= max(12.0, page_width * 0.04):
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
            parts.append(f'<p style="text-align:{alignment}">{escaped}</p>')
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
    ) = args

    import fitz

    images_dir = Path(images_dir_text)
    cache_root = Path(cache_root_text)
    xref_map_dir = Path(xref_map_dir_text)
    results = []
    stream_cache: Dict[int, str] = {}
    with fitz.open(pdf_path) as doc:
        for page_index in range(start_page, end_page):
            if _stop_requested():
                raise RuntimeError("PDF extraction cancelled")
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
    return results


def _load_results(cache_root: Path, page_entries: Dict[str, Dict], total_pages: int):
    pages = []
    images_by_page: Dict[int, List[Dict]] = {}
    for page_number in range(1, total_pages + 1):
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


def extract_pdf_fast(
    pdf_path: str,
    output_dir: str,
    *,
    mode: str = "fast_semantic",
    extract_images: bool = True,
    page_by_page: bool = False,
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
    if exact_source and all(
        _cache_entry_is_usable(cache_root, previous_pages.get(str(page_number), {}))
        for page_number in range(1, total_pages + 1)
    ):
        pages, images_by_page = _load_results(cache_root, previous_pages, total_pages)
        previous["outline_digest"] = outline_hash
        previous["stats"] = {
            "reused_pages": total_pages,
            "extracted_pages": 0,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        _atomic_write_json(manifest_path, previous)
        print(f"Fast PDF cache hit: reused {total_pages}/{total_pages} pages")
        if page_by_page:
            return pages, images_by_page
        return "\n\n".join(page_html for _, page_html in pages), images_by_page

    try:
        chunk_pages = max(1, min(64, int(os.environ.get("PDF_FAST_CHUNK_PAGES", "16"))))
    except ValueError:
        chunk_pages = 16
    jobs, section_titles = _bookmark_jobs(toc_entries, total_pages, chunk_pages)
    try:
        requested_workers = int(os.environ.get("EXTRACTION_WORKERS", "2"))
    except ValueError:
        requested_workers = 2
    worker_count = max(1, min(4, requested_workers, len(jobs) or 1))
    if total_pages < 8:
        worker_count = 1

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
    extracted_entries = []
    if worker_count == 1:
        for task in tasks:
            extracted_entries.extend(_extract_page_range(task))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_extract_page_range, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                extracted_entries.extend(future.result())

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
    pages, images_by_page = _load_results(cache_root, page_entries, total_pages)
    print(
        f"Fast PDF extraction complete: {total_pages - reused_pages} extracted, "
        f"{reused_pages} reused"
    )
    if page_by_page:
        return pages, images_by_page
    return "\n\n".join(page_html for _, page_html in pages), images_by_page


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
