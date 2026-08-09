"""Compile translated response files in a PDF workspace into one PDF."""

from __future__ import annotations

import html
import json
import os
import re
import threading
import time
import unicodedata
from typing import Callable
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup


LogCallback = Callable[[str], None]


class PDFCompilationCancelled(RuntimeError):
    """Raised when compilation is cancelled between renderer phases."""


def _log(callback: LogCallback | None, message: str) -> None:
    if callback:
        callback(message)
    else:
        try:
            print(message)
        except UnicodeEncodeError:
            print(message.encode("ascii", errors="backslashreplace").decode("ascii"))


def _natural_key(value: str) -> list:
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(value or ""))
    ]


def _numeric_order(value, fallback: int) -> tuple:
    try:
        return (0, float(value), fallback)
    except (TypeError, ValueError):
        return (1, fallback, fallback)


def _workspace_response_entries(folder: str) -> list[tuple[str, str]]:
    """Return existing translated response files as ``(path, title)``."""
    progress_path = os.path.join(folder, "translation_progress.json")
    entries: list[tuple[tuple, str, str]] = []
    seen: set[str] = set()
    try:
        with open(progress_path, "r", encoding="utf-8") as handle:
            progress = json.load(handle)
        chapters = progress.get("chapters", {}) if isinstance(progress, dict) else {}
    except (OSError, ValueError, TypeError):
        chapters = {}

    if isinstance(chapters, dict):
        for index, (key, info) in enumerate(chapters.items()):
            if not isinstance(info, dict):
                continue
            output_name = str(info.get("output_file") or "").strip()
            if not output_name.lower().endswith((".html", ".xhtml", ".htm")):
                continue
            output_path = (
                output_name
                if os.path.isabs(output_name)
                else os.path.join(folder, output_name)
            )
            if not os.path.isfile(output_path):
                continue
            normalized = os.path.normcase(os.path.normpath(output_path))
            if normalized in seen:
                continue
            seen.add(normalized)
            title = str(
                info.get("pdf_section_title")
                or info.get("pdf_toc_title")
                or info.get("title")
                or f"Section {info.get('actual_num') or key}"
            ).strip()
            entries.append((
                _numeric_order(info.get("actual_num", key), index),
                output_path,
                title,
            ))

    if entries:
        entries.sort(key=lambda item: item[0])
        return [(path, title) for _order, path, title in entries]

    fallback = []
    try:
        for entry in os.scandir(folder):
            if not entry.is_file(follow_symlinks=False):
                continue
            name = entry.name.lower()
            if name.startswith("response_") and name.endswith(
                (".html", ".xhtml", ".htm")
            ):
                fallback.append(entry.path)
    except (OSError, PermissionError):
        pass
    fallback.sort(key=lambda path: _natural_key(os.path.basename(path)))
    return [
        (path, f"Section {index}")
        for index, path in enumerate(fallback, 1)
    ]


def _fragment_body(content: str) -> str:
    soup = BeautifulSoup(content or "", "html.parser")
    container = soup.body if soup.body is not None else soup
    return "".join(str(child) for child in container.contents)


def _image_reference_basename(src: str) -> str:
    try:
        path = unquote(urlsplit(str(src or "")).path)
    except Exception:
        path = str(src or "")
    return os.path.basename(path.replace("\\", "/"))


def _legacy_page_image_reference(filename: str):
    match = re.fullmatch(
        r"page_(\d+)_img_(\d+)\.[a-z0-9]+",
        str(filename or ""),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _valid_cached_page_images(folder: str, page_number: int) -> list[dict]:
    cache_root = os.path.join(folder, ".pdf_extraction_cache")
    candidates = [
        os.path.join(cache_root, "pages", mode, f"page_{page_number:06d}.json")
        for mode in ("fast_semantic", "fast_layout")
    ]
    targeted_dir = os.path.join(cache_root, "targeted_images")
    if os.path.isdir(targeted_dir):
        candidates.extend(
            os.path.join(targeted_dir, name)
            for name in os.listdir(targeted_dir)
            if name.endswith(f"_page_{page_number:06d}.json")
        )
    images_dir = os.path.join(folder, "images")
    for cache_path in candidates:
        try:
            with open(cache_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError, TypeError):
            continue
        images = payload.get("images") if isinstance(payload, dict) else None
        if not isinstance(images, list):
            continue
        valid = []
        for item in images:
            if not isinstance(item, dict):
                continue
            filename = os.path.basename(str(item.get("filename") or ""))
            if filename and os.path.isfile(os.path.join(images_dir, filename)):
                normalized = dict(item)
                normalized["filename"] = filename
                valid.append(normalized)
        if valid:
            return valid
    return []


def _legacy_image_slots(images: list[dict]) -> list[dict]:
    """Collapse repeated placements to legacy one-slot-per-image-resource order."""
    slots = []
    seen = set()
    for position, image in enumerate(images or []):
        resource_index = image.get("resource_index")
        xref = int(image.get("xref") or 0)
        key = (
            "resource",
            int(resource_index),
        ) if resource_index is not None else (
            "xref",
            xref,
        ) if xref else (
            "placement",
            position,
        )
        if key in seen:
            continue
        seen.add(key)
        slots.append(image)
    return slots


def _repair_pdf_image_references(
    folder: str,
    contents: list[str],
    *,
    log_callback: LogCallback | None = None,
    stop_callback: Callable[[], bool] | None = None,
) -> tuple[list[str], dict]:
    """Resolve legacy page image names to content-addressed fast assets."""

    soups = [BeautifulSoup(content or "", "html.parser") for content in contents]
    referenced_nodes = []
    for soup in soups:
        for node in soup.find_all("img"):
            src = str(node.get("src") or "").strip()
            if not src or src.startswith(("data:", "http://", "https://")):
                continue
            referenced_nodes.append((node, src, _image_reference_basename(src)))

    unique_names = {name for _node, _src, name in referenced_nodes if name}
    _log(
        log_callback,
        f"🖼️ PDF image preflight: {len(referenced_nodes)} reference(s), "
        f"{len(unique_names)} unique file(s)",
    )
    images_dir = os.path.join(folder, "images")
    os.makedirs(images_dir, exist_ok=True)
    alias_map: dict[str, str] = {}
    required_page_slots = {}
    already_present = set()
    rename_map = {}
    try:
        with open(os.path.join(folder, "image_rename_map.json"), "r", encoding="utf-8") as handle:
            loaded_rename_map = json.load(handle)
        if isinstance(loaded_rename_map, dict):
            rename_map = {
                os.path.basename(str(old_name or "")): os.path.basename(str(new_name or ""))
                for old_name, new_name in loaded_rename_map.items()
                if old_name and new_name
            }
    except (OSError, ValueError, TypeError):
        rename_map = {}
    renamed_from = {new_name: old_name for old_name, new_name in rename_map.items()}
    for name in unique_names:
        if os.path.isfile(os.path.join(images_dir, name)):
            already_present.add(name)
            continue
        page_slot = _legacy_page_image_reference(name)
        if not page_slot and name in renamed_from:
            page_slot = _legacy_page_image_reference(renamed_from[name])
        if page_slot:
            required_page_slots[name] = page_slot

    required_pages = sorted({page for page, _slot in required_page_slots.values()})
    page_images = {
        page: _valid_cached_page_images(folder, page)
        for page in required_pages
    }
    missing_pages = [page for page in required_pages if not page_images.get(page)]
    recovered_pages = {}
    if missing_pages:
        try:
            from output_workspace import read_workspace_source_path
            from pdf_fast_extractor import ensure_pdf_page_images

            source_pdf = read_workspace_source_path(folder)
            if source_pdf.lower().endswith(".pdf") and os.path.isfile(source_pdf):
                _log(
                    log_callback,
                    f"🖼️ Recovering images from {len(missing_pages)} specifically "
                    "referenced PDF page(s); unrelated pages will not be scanned",
                )
                recovered_pages = ensure_pdf_page_images(
                    source_pdf,
                    folder,
                    missing_pages,
                    stop_callback=stop_callback,
                    progress_callback=lambda message: _log(log_callback, message),
                )
        except Exception as exc:
            if exc.__class__.__name__ == "PDFExtractionCancelled":
                raise PDFCompilationCancelled(
                    "PDF compilation stopped by user"
                ) from exc
            _log(log_callback, f"⚠️ Targeted PDF image recovery failed: {exc}")
    page_images.update(recovered_pages)

    for old_name, (page_number, image_number) in required_page_slots.items():
        images = _legacy_image_slots(page_images.get(page_number) or [])
        if 1 <= image_number <= len(images):
            alias_map[old_name] = str(images[image_number - 1].get("filename") or "")

    # Section-level renames were written as old page name -> new section name.
    # Apply that same alias to the content-addressed cache target.
    for old_base, new_base in rename_map.items():
        if old_base in alias_map and new_base:
            alias_map[new_base] = alias_map[old_base]

    repaired = 0
    unresolved = set()
    for node, original_src, basename in referenced_nodes:
        if basename in already_present:
            continue
        target = alias_map.get(basename)
        if target and os.path.isfile(os.path.join(images_dir, target)):
            node["src"] = re.sub(
                re.escape(basename) + r"(?=([?#].*)?$)",
                target,
                original_src,
                count=1,
            )
            repaired += 1
        else:
            unresolved.add(basename)

    _log(
        log_callback,
        f"🖼️ PDF image preflight complete: {len(already_present)} already valid, "
        f"{repaired} reference(s) repaired, {len(unresolved)} unresolved",
    )
    if unresolved:
        preview = ", ".join(sorted(unresolved)[:5])
        suffix = "…" if len(unresolved) > 5 else ""
        _log(log_callback, f"⚠️ Missing PDF images: {preview}{suffix}")
    return [str(soup) for soup in soups], {
        "references": len(referenced_nodes),
        "repaired": repaired,
        "unresolved": len(unresolved),
    }


def _renderer_heartbeat(log_callback, done_event: threading.Event, section_count: int) -> None:
    started = time.perf_counter()
    try:
        heartbeat_seconds = max(
            0.05,
            float(os.environ.get("PDF_COMPILE_HEARTBEAT_SECONDS", "3")),
        )
    except (TypeError, ValueError):
        heartbeat_seconds = 3.0
    while not done_event.wait(heartbeat_seconds):
        elapsed = int(time.perf_counter() - started)
        _log(
            log_callback,
            f"⏳ PDF renderer heartbeat: laying out {section_count} section(s), "
            f"{elapsed}s elapsed",
        )


def _source_pdf_stem(folder: str) -> str:
    try:
        from output_workspace import read_workspace_source_path

        source = read_workspace_source_path(folder)
    except Exception:
        source = ""
    if source.lower().endswith(".pdf"):
        return os.path.splitext(os.path.basename(source))[0]
    leaf = os.path.basename(os.path.normpath(folder)) or "translated"
    return re.sub(r"_PDF(?:_\d+)?$", "", leaf, flags=re.IGNORECASE)


def _outline_title_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(normalized.split()).casefold()


def _keep_only_section_bookmarks(pdf_path: str, titles: list[str]) -> None:
    """Normalize the generated outline to exactly one entry per response."""
    import fitz

    document = fitz.open(pdf_path)
    temp_path = f"{pdf_path}.outline.tmp"
    try:
        generated = document.get_toc(simple=True) or []
        cursor = 0
        cleaned = []
        for title in titles:
            wanted = _outline_title_key(title)
            matched_page = None
            for position in range(cursor, len(generated)):
                row = generated[position]
                if len(row) >= 3 and _outline_title_key(row[1]) == wanted:
                    matched_page = max(1, int(row[2]))
                    cursor = position + 1
                    break
            if matched_page is None:
                # WeasyPrint exposes the explicit bookmark anchors in its
                # outline. This fallback keeps the result valid if another
                # renderer dropped them, without reintroducing sentence-level
                # heading bookmarks.
                matched_page = cleaned[-1][2] if cleaned else 1
            cleaned.append([1, title, matched_page])
        document.set_toc(cleaned)
        document.save(temp_path, garbage=4, deflate=True)
    finally:
        document.close()
    os.replace(temp_path, pdf_path)


def compile_pdf_workspace(
    folder: str,
    log_callback: LogCallback | None = None,
    stop_callback: Callable[[], bool] | None = None,
) -> str:
    """Build a translated PDF from the current response HTML files."""
    if not folder or not os.path.isdir(folder):
        raise ValueError("PDF output workspace does not exist.")
    entries = _workspace_response_entries(folder)
    if not entries:
        raise ValueError("No translated response HTML files were found.")

    _log(log_callback, f"📄 Compiling PDF from {len(entries)} translated section(s)…")
    source_contents = []
    titles = []
    last_decile = -1
    for index, (path, title) in enumerate(entries, 1):
        if stop_callback and stop_callback():
            raise PDFCompilationCancelled("PDF compilation stopped by user")
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            content = handle.read()
        title = title or f"Section {index}"
        titles.append(title)
        source_contents.append(content)
        percent = int(index * 100 / len(entries))
        decile = percent // 10
        if index == 1 or index == len(entries) or decile > last_decile:
            last_decile = decile
            _log(
                log_callback,
                f"📄 PDF preparation: {index}/{len(entries)} sections ({percent}%)",
            )

    source_contents, image_stats = _repair_pdf_image_references(
        folder,
        source_contents,
        log_callback=log_callback,
        stop_callback=stop_callback,
    )
    sections = []
    for index, (content, title) in enumerate(zip(source_contents, titles), 1):
        sections.append(
            f'<section class="compiled-pdf-section" data-section="{index}">'
            f'<div id="pdf-section-{index}" class="pdf-bookmark-anchor">'
            f"{html.escape(title)}</div>"
            f"{_fragment_body(content)}</section>"
        )

    stem = _source_pdf_stem(folder)
    html_path = os.path.join(folder, f"{stem}_translated.html")
    pdf_path = os.path.join(folder, f"{stem}_translated.pdf")
    document_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(stem)} - Translated</title>
  <link rel="stylesheet" href="styles.css">
  <style>
    h1, h2, h3, h4, h5, h6 {{ bookmark-level: none !important; }}
    .pdf-bookmark-anchor {{
      bookmark-level: 1 !important;
      bookmark-label: content(text);
      height: 0; margin: 0; padding: 0; overflow: hidden;
      color: transparent; font-size: 0; line-height: 0;
    }}
    .compiled-pdf-section {{ margin: 0; padding: 0; }}
  </style>
</head>
<body>
{''.join(sections)}
</body>
</html>"""
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(document_html)

    from pdf_extractor import create_pdf_from_html

    css_path = os.path.join(folder, "styles.css")
    images_dir = os.path.join(folder, "images")
    if stop_callback and stop_callback():
        raise PDFCompilationCancelled("PDF compilation stopped by user")
    _log(
        log_callback,
        f"📄 Rendering PDF ({image_stats['references']} image reference(s), "
        f"{image_stats['unresolved']} unresolved)…",
    )
    renderer_done = threading.Event()
    heartbeat = threading.Thread(
        target=_renderer_heartbeat,
        args=(log_callback, renderer_done, len(entries)),
        name="pdf-compiler-heartbeat",
        daemon=True,
    )
    heartbeat.start()
    try:
        success = create_pdf_from_html(
            document_html,
            pdf_path,
            css_path=css_path if os.path.isfile(css_path) else None,
            images_dir=images_dir if os.path.isdir(images_dir) else None,
        )
    finally:
        renderer_done.set()
        heartbeat.join(timeout=0.5)
    if not success or not os.path.isfile(pdf_path):
        raise RuntimeError("The PDF renderer did not create an output file.")
    _log(log_callback, "📑 Normalizing PDF bookmarks (one per translated section)…")
    _keep_only_section_bookmarks(pdf_path, titles)
    _log(log_callback, f"✅ PDF compilation complete: {pdf_path}")
    return pdf_path
