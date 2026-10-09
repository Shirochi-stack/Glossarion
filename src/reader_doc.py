"""reader_doc: the GUI-free EPUB reader document builder (moved verbatim from epub_library).

Shared GUI-free core (Glossarion mobile rewrite, milestone U5). The desktop reader
(``epub_library.EpubReaderDialog``) and Glossarion Mobile's Reader build the same
documents from the same code:

* EPUB loading: ``_EpubLoaderThread.run`` (spine-first chapter resolution, lazy image
  descriptors, the special-file filter, the pickle cache) is ``EpubLoaderMixin.run``;
  ``_EpubCacheLoaderThread`` / ``_OverlayMergeThread`` / ``_ReaderImagePreloadThread`` /
  ``_WorkspaceReaderLoaderThread`` / ``_EpubSearchThread`` likewise inherit their
  ``run`` from the mixins below (the plan's mixin rule: the Qt thread lists the mixin
  first; a mobile job is a plain object with the same attributes and ``emit`` hooks);
* the page builder ``ReaderDocMixin``: ``_process_html`` (image materialisation,
  multi-image paragraph split, full-page image wrapper), ``_get_embedded_css``
  (override CSS, translated workspace CSS, zip CSS, data-URI fonts), ``_wrap_html``
  (paged CSS-column document + JS, scroll document), ``_get_theme`` and the image
  cache helpers, all byte-for-byte;
* module helpers: plain text / excerpts for search, the lazy image resolver and cache,
  the pickle cache, native TOC (``TOC.txt`` / ``toc.ncx``) parsing and mapping, the
  overlay signature, layouts, ``_READER_THEMES`` and the Google Translate codes.

Adapted (documented in tests/parity/DISCREPANCIES.md, U5):

* ``QUrl(src).scheme()`` (pure computation) became :func:`_url_scheme` (QUrl's rule);
  ``QUrl.fromLocalFile(path).toString()`` in ``_process_html`` is the
  ``_reader_file_url`` hook (desktop overrides it with the Qt original, mobile serves
  images from the in-app localhost server);
* ``_reader_image_is_sizeable`` probed dimensions with ``QImageReader``; it now uses
  ``library_covers._probe_image_size`` (header parse, Pillow fallback);
* ``_epub_cache_dir`` honours ``set_epub_cache_dir()`` (mobile: the app cache).

New (mobile-first, desktop may adopt later): ``wrap_reader_html(..., mobile=True)``
(viewport meta, safe-area insets, ``100dvh``, ``-webkit-`` column fallbacks, tap-zone /
swipe paging JS posting ``GLRDR:``-prefixed console events with a ``fetch('/__ev')``
fallback), :func:`build_bilingual_chapter` and :func:`html_to_blocks` (native fallback).

Log records keep the ``epub_library`` logger name. Rules: Python 3.10 compatible; never
import PySide6, translator_gui or dpi_setup.
"""

import hashlib
import json
import logging
import os
import re
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

from chapter_display_numbering import (
    filename_chapter_number,
    nonreset_chapter_display_numbers,
)
from html_tag_entities import unescape_valid_html_tag_entities
from library_core import (
    _epub_cache_key,
    _extract_html_title_fast,
    _is_configured_special_file,
    _is_special_spine_item,
    _read_translated_chapter_title,
    _reader_worker_count,
    _resolve_output_roots,
    _special_file_settings_signature,
)
from library_covers import _probe_image_size

# Same logger as before the move (records keep the "epub_library" name).
logger = logging.getLogger("epub_library")

#: Process-wide EPUB pickle-cache folder (None = ``%TEMP%/Glossarion_EpubCache``).
_EPUB_CACHE_DIR_OVERRIDE = None


def set_epub_cache_dir(path) -> None:
    """Point the reader's EPUB pickle cache at *path* (None restores the default).

    Glossarion Mobile calls this once (``library_core.install_library_env``); desktop
    never does, so the cache stays ``%TEMP%/Glossarion_EpubCache``.
    """
    global _EPUB_CACHE_DIR_OVERRIDE
    _EPUB_CACHE_DIR_OVERRIDE = os.path.abspath(str(path)) if path else None


# ---------------------------------------------------------------------------
# Moved verbatim from epub_library (U5)
# ---------------------------------------------------------------------------


def _epub_plain_chapter_text(html: str) -> str:
    """Return searchable visible text, matching the reader DOM search."""
    html = html or ''
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for node in soup(["script", "style", "noscript", "img"]):
            node.decompose()
        return soup.get_text("", strip=False)
    except Exception:
        import html as html_lib
        cleaned = re.sub(
            r'<(script|style|noscript)\b[^>]*>.*?</\1>',
            ' ', html, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r'<[^>]+>', ' ', cleaned)
        return html_lib.unescape(cleaned)


def _epub_search_excerpt(plain: str, start: int, end: int,
                         radius: int = 60) -> str:
    left = max(0, start - radius)
    right = min(len(plain), end + radius)
    excerpt = plain[left:right]
    excerpt = re.sub(r"\s+", " ", excerpt).strip()
    if left > 0:
        excerpt = "..." + excerpt
    if right < len(plain):
        excerpt += "..."
    return excerpt


def _epub_cache_dir() -> str:
    d = _EPUB_CACHE_DIR_OVERRIDE or os.path.join(
        tempfile.gettempdir(), "Glossarion_EpubCache")
    os.makedirs(d, exist_ok=True)
    return d


_LAZY_EPUB_IMAGE_TAG = "__glossarion_epub_image_member__"


_READER_IMAGE_EXTS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp",
    ".avif", ".jxl",
)


def _discover_epub_image_members(epub_path: str) -> set[str]:
    """Return case-folded ZIP members declared as images in the OPF."""
    import posixpath
    import zipfile
    from urllib.parse import unquote
    from xml.etree import ElementTree as ET

    members: set[str] = set()
    try:
        with zipfile.ZipFile(epub_path, "r") as archive:
            container = ET.fromstring(archive.read("META-INF/container.xml"))
            rootfile = next((
                node.attrib.get("full-path", "")
                for node in container.iter()
                if node.tag.rsplit("}", 1)[-1] == "rootfile"
                and node.attrib.get("full-path")
            ), "")
            if not rootfile:
                return members
            package = ET.fromstring(archive.read(rootfile))
            opf_dir = posixpath.dirname(rootfile)
            for node in package.iter():
                if node.tag.rsplit("}", 1)[-1] != "item":
                    continue
                href = unquote(str(node.attrib.get("href") or ""))
                media_type = str(node.attrib.get("media-type") or "").lower()
                if not href:
                    continue
                if (not media_type.startswith("image/")
                        and not href.lower().endswith(_READER_IMAGE_EXTS)):
                    continue
                member = posixpath.normpath(posixpath.join(opf_dir, href))
                members.add(member.lstrip("/").casefold())
    except Exception:
        logger.debug("Could not pre-index EPUB image members: %s",
                     traceback.format_exc())
    return members


def _lazy_epub_image(member_name: str) -> tuple[str, str]:
    """Return a pickle-safe descriptor for an unextracted EPUB image."""
    return (_LAZY_EPUB_IMAGE_TAG, str(member_name or ""))


def _lazy_epub_image_member(value) -> str:
    """Return the archive member stored in a lazy image descriptor."""
    if (isinstance(value, (tuple, list)) and len(value) == 2
            and value[0] == _LAZY_EPUB_IMAGE_TAG):
        return str(value[1] or "")
    return ""


_URL_SCHEME_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*\Z")


def _url_scheme(src) -> str:
    """Lowercase URL scheme of *src*, parsed like ``QUrl(src).scheme().lower()``.

    Replaces the Qt call (pure computation) with QUrl's own rule: the text before the
    first ``:`` that precedes any ``?`` / ``#`` is the scheme when it is a valid RFC 3986
    scheme (ASCII letter, then letters, digits, ``+``, ``-``, ``.``). Like QUrl nothing
    is trimmed (``" http://x"`` has no scheme) and a malformed authority does not clear
    it (``"http://[::1"`` is still ``http``); ``urllib.parse.urlsplit`` differs on both.
    """
    text = str(src or "")
    for mark in ("#", "?"):
        cut = text.find(mark)
        if cut >= 0:
            text = text[:cut]
    colon = text.find(":")
    if colon <= 0:
        return ""
    scheme = text[:colon]
    if not _URL_SCHEME_RE.match(scheme):
        return ""
    return scheme.lower()


def _reader_image_candidates(src: str) -> list[str]:
    """Return the legacy-compatible lookup variants for an image reference."""
    src = str(src or "")
    candidates = [
        src,
        os.path.basename(src),
        src.lstrip("../"),
        src.lstrip("./"),
    ]
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def _reader_image_resource(src: str, images: dict | None,
                           extra_image_dirs, epub_path: str) -> dict | None:
    """Resolve *src* without loading its bytes.

    The returned descriptor is safe to hand to the background preloader. Its
    ``identity`` changes when a filesystem resource changes and stays stable
    for an EPUB member during the lifetime of an unchanged source archive.
    """
    images = images or {}
    for candidate in _reader_image_candidates(src):
        if candidate not in images:
            continue
        value = images[candidate]
        member = _lazy_epub_image_member(value)
        if member:
            try:
                stamp = os.path.getmtime(epub_path)
            except OSError:
                stamp = 0
            return {
                "kind": "epub",
                "member": member,
                "identity": f"epub:{os.path.abspath(epub_path or '')}:{stamp}:{member}",
            }
        if isinstance(value, (bytes, bytearray, memoryview)):
            return {
                "kind": "bytes",
                "data": bytes(value),
                "identity": f"memory:{candidate}:{id(value)}:{len(value)}",
            }

    img_basename = os.path.basename(str(src or ""))
    for extra_dir in extra_image_dirs or []:
        if not extra_dir or not os.path.isdir(extra_dir):
            continue
        disk_path = os.path.join(extra_dir, img_basename)
        if os.path.isfile(disk_path):
            return _reader_file_image_resource(disk_path)

    if epub_path:
        epub_dir = os.path.dirname(epub_path)
        if epub_dir:
            rel_candidate = os.path.normpath(os.path.join(epub_dir, str(src or "")))
            if os.path.isfile(rel_candidate):
                return _reader_file_image_resource(rel_candidate)
            for sub in ("images", "Images", "translated_images"):
                disk_path = os.path.join(epub_dir, sub, img_basename)
                if os.path.isfile(disk_path):
                    return _reader_file_image_resource(disk_path)
    return None


def _reader_file_image_resource(path: str) -> dict:
    path = os.path.abspath(path)
    try:
        stat = os.stat(path)
        stamp = f"{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        stamp = "0:0"
    return {
        "kind": "file",
        "path": path,
        "identity": f"file:{path}:{stamp}",
    }


def _read_epub_member_from_zip(zf, member: str,
                               name_lookup: dict[str, str] | None = None) -> bytes:
    """Read an EPUB member with a case-insensitive fallback."""
    try:
        return zf.read(member)
    except KeyError:
        lookup = name_lookup
        if lookup is None:
            lookup = {name.casefold(): name for name in zf.namelist()}
        member_folded = str(member or "").lstrip("/").casefold()
        actual = lookup.get(member_folded)
        if not actual:
            suffix = "/" + member_folded
            matches = [
                candidate for folded, candidate in lookup.items()
                if folded.endswith(suffix)
            ]
            if len(matches) == 1:
                actual = matches[0]
        if not actual:
            return b""
        try:
            return zf.read(actual)
        except (KeyError, OSError):
            return b""


def _reader_image_is_sizeable(image_data: bytes) -> bool:
    """Return whether an image should receive full-page reader treatment."""
    sizeable = len(image_data or b"") > 5120
    # The byte-size rule already classifies substantial images. Avoid
    # QImage.fromData() in that common case: it fully decodes multi-megapixel
    # scans just to ask for their dimensions, which can dominate raw-reader
    # loading time and allocate a large temporary bitmap.
    if sizeable:
        return True
    try:
        # Header probe (PNG/GIF/JPEG/BMP/WebP/SVG, Pillow fallback) instead of
        # QImageReader: the same dimensions without Qt.
        dimensions = _probe_image_size(image_data or b"")
        sizeable = bool(
            dimensions
            and dimensions[0] >= 220
            and dimensions[1] >= 220
        )
    except Exception:
        pass
    return sizeable


def _reader_image_cache_path(temp_dir: str, src: str) -> str:
    """Return the existing reader-compatible cached image path."""
    safe_name = os.path.basename(str(src or "")).replace("/", "_").replace("\\", "_")
    if not safe_name:
        safe_name = hashlib.md5(str(src or "").encode()).hexdigest() + ".img"
    return os.path.join(temp_dir, safe_name)


def _write_reader_image_cache(temp_dir: str, src: str, image_data: bytes,
                              source_path: str = "") -> str:
    """Materialize image bytes once and return their local cache path."""
    os.makedirs(temp_dir, exist_ok=True)
    img_path = _reader_image_cache_path(temp_dir, src)
    needs_write = not os.path.isfile(img_path)
    if not needs_write:
        try:
            needs_write = os.path.getsize(img_path) != len(image_data)
            if source_path and not needs_write:
                needs_write = os.path.getmtime(img_path) < os.path.getmtime(source_path)
        except OSError:
            needs_write = True
    if needs_write:
        with open(img_path, "wb") as f:
            f.write(image_data)
    return img_path


def _reader_image_map_signature(images: dict | None) -> tuple:
    """Return a cheap identity signature without comparing image payloads."""
    signature = []
    for key, value in (images or {}).items():
        member = _lazy_epub_image_member(value)
        if member:
            marker = ("epub", member)
        elif isinstance(value, (bytes, bytearray, memoryview)):
            marker = ("bytes", id(value), len(value))
        else:
            marker = (type(value).__name__, id(value))
        signature.append((str(key), marker))
    return tuple(sorted(signature))


def _load_epub_cache(epub_path: str, show_special_files: bool = True,
                     config: dict | None = None):
    """Try to load cached EPUB data.

    Returns ``(chapters, images, filenames)`` where ``filenames`` is a parallel
    list of source item names (one per chapter entry) or ``None`` on failure.
    ``filenames`` is empty when the cache predates that field — callers must
    handle that gracefully.

    *show_special_files* is forwarded to :func:`_epub_cache_key` so the
    on / off variants of the cache don't collide.

    Cache entries with an **empty chapter list** are treated as invalid and
    discarded. They're almost always the fingerprint of a past load failure
    (e.g. an EPUB with non-standard ``media-type="text/html"`` that the old
    strict ITEM_DOCUMENT walker couldn't see). Returning None here forces a
    re-parse with the current, lenient spine-first resolver.
    """
    import pickle
    try:
        key = _epub_cache_key(epub_path, show_special_files, config)
        cache_file = os.path.join(_epub_cache_dir(), f"{key}.pkl")
        if os.path.isfile(cache_file):
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and "chapters" in data and "images" in data:
                chapters = data["chapters"] or []
                if not chapters:
                    # Stale / bad cache — drop it and force a fresh parse.
                    try:
                        os.remove(cache_file)
                    except OSError:
                        pass
                    return None
                return chapters, data["images"], data.get("filenames", [])
    except Exception:
        pass
    return None


def _save_epub_cache(epub_path: str, chapters, images, filenames=None,
                     show_special_files: bool = True,
                     config: dict | None = None):
    """Save parsed EPUB data to disk cache.

    *show_special_files* is forwarded to :func:`_epub_cache_key` so the
    on / off variants of the cache are stored under distinct keys.
    """
    import pickle
    try:
        key = _epub_cache_key(epub_path, show_special_files, config)
        cache_file = os.path.join(_epub_cache_dir(), f"{key}.pkl")
        with open(cache_file, "wb") as f:
            pickle.dump({
                "chapters": chapters,
                "images": images,
                "filenames": list(filenames or []),
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass


def _parse_native_toc_txt(path: str) -> list[dict[str, str]]:
    """Parse Glossarion's ``TOC.txt`` cache into ordered reader entries."""
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as stream:
            text = stream.read()
    except OSError:
        return []

    entries: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if re.match(r"^Chapter\s+\d+\s*:\s*$", line, re.IGNORECASE):
            current = {}
            continue
        match = re.match(
            r"^(Original|Translated|Target\s+URI)\s*:\s*(.*)$",
            line,
            re.IGNORECASE,
        )
        if not match:
            continue
        key = match.group(1).lower().replace(" ", "_")
        current[key] = match.group(2).strip()
        if key != "target_uri":
            continue
        title = current.get("translated") or current.get("original") or ""
        target = current.get("target_uri") or ""
        if title and target:
            entries.append({
                "title": title,
                "target": target,
                "source": os.path.abspath(path),
            })
        current = {}
    return entries


def _parse_native_toc_ncx(data: bytes | str, source: str = "") -> list[dict[str, str]]:
    """Parse NCX navPoints without depending on a particular XML namespace."""
    from xml.etree import ElementTree as ET

    try:
        root = ET.fromstring(data)
    except (ET.ParseError, TypeError, ValueError):
        return []

    def _local_name(tag) -> str:
        return str(tag or "").rsplit("}", 1)[-1].rsplit(":", 1)[-1].lower()

    entries: list[dict[str, str]] = []
    for nav_point in root.iter():
        if _local_name(nav_point.tag) != "navpoint":
            continue
        title = ""
        target = ""
        for child in list(nav_point):
            local = _local_name(child.tag)
            if local == "navlabel" and not title:
                for label_child in child.iter():
                    if _local_name(label_child.tag) == "text":
                        title = " ".join("".join(label_child.itertext()).split())
                        if title:
                            break
            elif local == "content" and not target:
                target = str(child.get("src") or "").strip()
        if title and target:
            entries.append({
                "title": title,
                "target": target,
                "source": source,
            })
    return entries


def _find_reader_sidecar(directory: str, filename: str) -> str:
    """Return a case-insensitive sidecar match from *directory*."""
    if not directory or not os.path.isdir(directory):
        return ""
    direct = os.path.join(directory, filename)
    if os.path.isfile(direct):
        return direct
    wanted = filename.casefold()
    try:
        for entry in os.scandir(directory):
            if (entry.is_file(follow_symlinks=False)
                    and entry.name.casefold() == wanted):
                return entry.path
    except OSError:
        pass
    return ""


def _load_reader_native_toc(output_dir: str, epub_path: str) -> list[dict[str, str]]:
    """Load TOC.txt, a sidecar NCX, or finally the EPUB's embedded NCX."""
    toc_txt = _find_reader_sidecar(output_dir, "TOC.txt")
    if toc_txt:
        entries = _parse_native_toc_txt(toc_txt)
        if entries:
            return entries

    sidecar_ncx = _find_reader_sidecar(output_dir, "toc.ncx")
    if sidecar_ncx:
        try:
            with open(sidecar_ncx, "rb") as stream:
                entries = _parse_native_toc_ncx(
                    stream.read(), source=os.path.abspath(sidecar_ncx))
            if entries:
                return entries
        except OSError:
            pass

    if not (epub_path and os.path.isfile(epub_path)
            and epub_path.lower().endswith(".epub")):
        return []
    try:
        import zipfile
        with zipfile.ZipFile(epub_path, "r") as archive:
            member = next(
                (name for name in archive.namelist()
                 if name.lower().endswith("toc.ncx")),
                "",
            )
            if member:
                return _parse_native_toc_ncx(
                    archive.read(member),
                    source=f"{os.path.abspath(epub_path)}::{member}",
                )
    except (OSError, ValueError, zipfile.BadZipFile):
        pass
    return []


def _native_toc_target_key(target: str) -> tuple[str, str]:
    """Return ``(chapter basename key, fragment)`` for a TOC target URI."""
    from html import unescape
    from urllib.parse import unquote

    value = unquote(unescape(str(target or "").strip())).replace("\\", "/")
    path, separator, fragment = value.partition("#")
    path = path.split("?", 1)[0]
    basename = path.rsplit("/", 1)[-1].casefold()
    if basename.startswith("response_"):
        basename = basename[len("response_"):]
    stem = os.path.splitext(basename)[0]
    return stem, fragment if separator else ""


def _map_native_toc_to_chapters(
    entries: list[dict[str, str]],
    chapter_filenames: list[str],
) -> list[dict]:
    """Attach each native TOC entry to its matching loaded spine chapter."""
    chapter_by_key: dict[str, int] = {}
    for index, filename in enumerate(chapter_filenames or []):
        key, _fragment = _native_toc_target_key(filename)
        if key:
            chapter_by_key.setdefault(key, index)

    mapped: list[dict] = []
    for entry in entries or []:
        key, fragment = _native_toc_target_key(entry.get("target", ""))
        if key not in chapter_by_key:
            continue
        mapped_entry = dict(entry)
        mapped_entry["chapter_index"] = chapter_by_key[key]
        mapped_entry["fragment"] = fragment
        mapped.append(mapped_entry)
    return mapped


def _reader_overlay_signature(overlay: dict) -> tuple:
    """Snapshot the files a reader merge is about to consume."""
    signature = []
    for key in sorted(overlay):
        entry = overlay[key] or {}
        path = entry.get("path") or ""
        try:
            stat = os.stat(path)
            file_signature = (
                stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino,
            )
        except OSError:
            file_signature = None
        signature.append((
            key, path, file_signature, entry.get("title") or "",
            str(entry.get("status") or "").strip().lower(),
        ))
    return tuple(signature)


def _workspace_reader_placeholder(title: str, message: str) -> str:
    """Return a small reader-safe placeholder document."""
    import html as _html

    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>'
        '<div style="max-width:48em;margin:4em auto;text-align:center;opacity:.72">'
        f'<h2>{_html.escape(str(title or "Section"))}</h2>'
        f'<p>{_html.escape(str(message or ""))}</p>'
        '</div></body></html>'
    )


# Layout modes
LAYOUT_SCROLL = "scroll"         # Single chapter, scrollable


LAYOUT_SINGLE = "single_page"   # Single chapter, viewport-paginated (page turns)


LAYOUT_DOUBLE = "double_page"   # Two side-by-side readers, viewport-paginated


LAYOUT_ALL    = "all_scroll"    # All chapters concatenated, scrollable


# Google Translate language codes, keyed by the translator's ``output_language``
# dropdown values. The reader's right-click menu picks the current value from
# ``config['output_language']`` and uses this map to fill ``tl=`` in the
# translate.google.com URL. Source is left as ``sl=auto`` so Google sniffs
# the language of the selected passage itself.
_READER_GT_LANG_CODES: dict[str, str] = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "russian": "ru",
    "arabic": "ar",
    "hindi": "hi",
    "chinese": "zh-CN",
    "chinese (simplified)": "zh-CN",
    "simplified chinese": "zh-CN",
    "chinese (traditional)": "zh-TW",
    "traditional chinese": "zh-TW",
    "japanese": "ja",
    "korean": "ko",
    "turkish": "tr",
    "vietnamese": "vi",
    "bahasa indonesia": "id",
    "indonesian": "id",
    "bengali": "bn",
    "urdu": "ur",
    "marathi": "mr",
    "punjabi": "pa",
    "gujarati": "gu",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "malayalam": "ml",
    "nepali": "ne",
    "sinhala": "si",
    "malay": "ms",
    "filipino": "tl",
    "thai": "th",
    "burmese": "my",
    "khmer": "km",
    "lao": "lo",
    "dutch": "nl",
    "polish": "pl",
    "ukrainian": "uk",
    "persian": "fa",
    "hebrew": "he",
    "greek": "el",
    "romanian": "ro",
    "swedish": "sv",
    "czech": "cs",
    "catalan": "ca",
    "bulgarian": "bg",
    "croatian": "hr",
    "serbian": "sr",
    "slovak": "sk",
    "slovenian": "sl",
    "hungarian": "hu",
    "danish": "da",
    "finnish": "fi",
    "norwegian": "no",
    "swahili": "sw",
    "afrikaans": "af",
    "amharic": "am",
    "hausa": "ha",
    "yoruba": "yo",
    "zulu": "zu",
    "azerbaijani": "az",
    "kazakh": "kk",
    "uzbek": "uz",
}


def _target_lang_to_google_code(name: str) -> str:
    """Map the translator's target-language name to a Google Translate code.

    Falls back to English when the dropdown is empty or carries a custom
    label the map hasn't been taught (users can type any value into the
    editable combo, so a hard error isn't appropriate).
    """
    if not name:
        return "en"
    key = str(name).strip().lower()
    return _READER_GT_LANG_CODES.get(key, "en")


def _google_translate_url(text: str, target_code: str) -> str:
    """translate.google.com URL for *text* into *target_code* ("" when there is no text)."""
    text = (text or "").strip()
    if not text:
        return ""
    # URL-escape but keep spaces as '+' for readability in the address bar.
    from urllib.parse import quote
    encoded = quote(text, safe="")
    return (
        f"https://translate.google.com/?sl=auto&tl={target_code}"
        f"&text={encoded}&op=translate"
    )


def _define_url(text: str) -> str:
    """Google ``define`` search URL for *text* ("" when there is no text)."""
    text = (text or "").strip()
    if not text:
        return ""
    from urllib.parse import quote
    encoded = quote(f"define {text}", safe="")
    return f"https://www.google.com/search?q={encoded}"


def _chapter_display_numbers(filenames, config=None) -> list:
    """Non-resetting chapter display numbers for the reader's lowercase basenames."""
    return nonreset_chapter_display_numbers(
        filename_chapter_number(
            filename,
            is_special=_is_configured_special_file(
                filename, config
            ),
        )
        for filename in filenames
    )


# Reader themes — first one is the default and matches translator_gui.py's dark palette
_READER_THEMES = [
    {"name": "Dark",     "bg": "#1e1e1e", "fg": "#d4d4d4", "heading": "#c8c8f0",
     "link": "#6c9bd2", "code_bg": "#252530", "border": "#333333"},
    {"name": "Light",    "bg": "#faf9f6", "fg": "#2c2c2c", "heading": "#333333",
     "link": "#1a73e8", "code_bg": "#eeeeee", "border": "#dddddd"},
    {"name": "Sepia",    "bg": "#f4ecd8", "fg": "#5b4636", "heading": "#3e2c1c",
     "link": "#8b5e3c", "code_bg": "#ece0c8", "border": "#d4c8a8"},
    {"name": "Midnight", "bg": "#0d1117", "fg": "#c9d1d9", "heading": "#58a6ff",
     "link": "#58a6ff", "code_bg": "#161b22", "border": "#21262d"},
    {"name": "Forest",   "bg": "#1a2e1a", "fg": "#c8d8c8", "heading": "#7ec87e",
     "link": "#5dbd5d", "code_bg": "#1e3a1e", "border": "#2a4a2a"},
    {"name": "Rose",     "bg": "#2e1a2e", "fg": "#e0c8e0", "heading": "#d89ad8",
     "link": "#c074c0", "code_bg": "#3a1e3a", "border": "#4a2a4a"},
]


class EpubCacheLoaderMixin:
    """``_EpubCacheLoaderThread``: read the pickled EPUB cache (``hit`` / ``miss``).
    """

    def run(self):
        if self._should_stop():
            return
        try:
            cached = _load_epub_cache(
                self._epub_path,
                show_special_files=self._show_special_files,
                config=self._config,
            )
        except Exception:
            logger.debug("Cache load failed in worker: %s",
                             traceback.format_exc())
            cached = None
        if self._should_stop():
            return
        if cached:
            chapters, images, filenames = cached
            self.hit.emit(chapters, images, list(filenames or []))
        else:
            self.miss.emit()


class OverlayMergeMixin:
    """``_OverlayMergeThread``: merge translated response HTML over the raw chapters.
    """

    def run(self):
        if self._should_stop():
            return
        # Keep the snapshot from BEFORE reading. Comparing two freshly-statted
        # overlay maps later would miss every rewrite at an unchanged path.
        self._read_signature = _reader_overlay_signature(self._overlay)
        raw = self._raw_chapters
        overlaid = raw
        overlay_applied = False

        # --- Overlay merge: per-chapter translated HTML reads ---
        # Done through a small ThreadPoolExecutor so N sequential disk
        # hits collapse into a couple of parallel batches. Errors per
        # chapter are logged and the raw content is kept so one bad
        # overlay file can't poison the whole merge.
        if self._overlay and raw:
            filenames = self._filenames
            overlay = self._overlay

            def _fetch_overlay(idx_title_content):
                idx, (title, content) = idx_title_content
                if self._should_stop():
                    return (idx, title, content, False, False)
                fname = filenames[idx] if idx < len(filenames) else ""
                key = os.path.basename(fname).lower() if fname else ""
                ov = overlay.get(key) if key else None
                if not ov:
                    return (idx, title, content, False, False)
                path = ov.get("path") or ""
                if not (path and os.path.isfile(path)):
                    return (idx, title, content, False, False)

                def retry_later():
                    # A writer may briefly lock or truncate a response. Keep
                    # the last readable translation while a later tick retries.
                    previous = (
                        self._previous_chapters[idx]
                        if idx < len(self._previous_chapters) else (title, content)
                    )
                    return (idx, previous[0], previous[1],
                            previous != (title, content), True)

                try:
                    with open(path, "rb") as f:
                        data = f.read()
                except OSError:
                    logger.debug("Overlay read failed: %s",
                                 traceback.format_exc())
                    return retry_later()
                if self._should_stop():
                    return (idx, title, content, False, False)
                if not data.strip():
                    return retry_later()
                translated_html = data.decode("utf-8", errors="replace")
                new_title = title
                if ov.get("title"):
                    new_title = str(ov["title"])
                else:
                    # Progress Manager intentionally supplies path-only
                    # overlays so its context-menu action can open the reader
                    # immediately.  We already have the translated bytes in
                    # this worker, so derive the translated TOC title here
                    # without adding any GUI-thread file I/O.
                    extracted_title = _extract_html_title_fast(data)
                    if extracted_title:
                        new_title = extracted_title
                return (idx, new_title, translated_html, True, False)

            try:
                items = list(enumerate(raw))
                workers = _reader_worker_count(len(items), config=self._config)
                if workers <= 1:
                    results = [_fetch_overlay(item) for item in items]
                else:
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        results = list(pool.map(_fetch_overlay, items))
            except Exception:
                logger.debug("Overlay parallel merge failed, "
                             "falling back to sequential: %s",
                             traceback.format_exc())
                results = [_fetch_overlay(item)
                           for item in enumerate(raw)]

            if self._should_stop():
                return
            merged = [None] * len(raw)
            for idx, title, content, applied, retry in results:
                merged[idx] = (title, content)
                if applied:
                    overlay_applied = True
                if retry:
                    self._retry_required = True
            overlaid = merged

        # Extra image directories stay as paths and are resolved lazily by the
        # active/next chapter. The old implementation read every image in
        # ``images/`` and ``translated_images/`` during this merge even when
        # the user never opened a chapter that referenced most of them.
        images = self._images

        if not self._should_stop():
            if _reader_overlay_signature(self._overlay) != self._read_signature:
                self._retry_required = True
            self._result_ready = True
            self.done.emit(overlaid, images, overlay_applied)


class ReaderImagePreloadMixin:
    """``_ReaderImagePreloadThread``: materialise the next chapter's local images.
    """

    def run(self):
        resources: dict[str, dict] = {}
        zf = None
        try:
            from bs4 import BeautifulSoup
            import zipfile

            content = unescape_valid_html_tag_entities(self._html_content)
            soup = BeautifulSoup(content, "html.parser")
            sources: list[str] = []
            for tag in soup.find_all("img"):
                src = str(tag.get("src") or "")
                if src:
                    sources.append(src)
            for tag in soup.find_all("image"):
                src = next((
                    str(tag.get(attr) or "")
                    for attr in (
                        "href", "xlink:href",
                        "{http://www.w3.org/1999/xlink}href",
                    )
                    if tag.get(attr)
                ), "")
                if src:
                    sources.append(src)

            zip_names = None
            for src in dict.fromkeys(sources):
                if self._should_stop():
                    return
                if _url_scheme(src) in ("http", "https"):
                    continue
                resource = _reader_image_resource(
                    src, self._images, self._extra_dirs, self._epub_path)
                if not resource:
                    continue
                identity = resource["identity"]
                kind = resource.get("kind")
                data = b""
                if kind == "bytes":
                    data = resource.get("data") or b""
                elif kind == "file":
                    try:
                        with open(resource.get("path") or "", "rb") as f:
                            data = f.read()
                    except OSError:
                        data = b""
                elif kind == "epub" and self._epub_path:
                    try:
                        if zf is None:
                            zf = zipfile.ZipFile(self._epub_path, "r")
                            zip_names = {
                                name.casefold(): name for name in zf.namelist()
                            }
                        data = _read_epub_member_from_zip(
                            zf, resource.get("member") or "", zip_names)
                    except (OSError, zipfile.BadZipFile):
                        data = b""
                if not data or self._should_stop():
                    continue
                try:
                    cached_path = _write_reader_image_cache(
                        self._temp_dir,
                        src,
                        data,
                        resource.get("path") or "",
                    )
                except OSError:
                    continue
                resources[identity] = {
                    "path": cached_path,
                    "sizeable": _reader_image_is_sizeable(data),
                    "classified": True,
                }
        except Exception:
            logger.debug("Next-chapter image preload failed: %s",
                         traceback.format_exc())
        finally:
            if zf is not None:
                try:
                    zf.close()
                except Exception:
                    pass
        if not self._should_stop():
            self.done.emit(self._preload_key, resources)


class WorkspaceReaderLoaderMixin:
    """``_WorkspaceReaderLoaderThread``: chapters of a PDF / HTML workspace manifest.
    """

    def run(self):
        raw_chapters = []
        translated_chapters = []
        filenames = []
        try:
            for entry in self._manifest.get("entries", []) or []:
                if self._should_stop():
                    return
                title = str(entry.get("title") or entry.get("filename") or "Section")
                filenames.append(str(entry.get("filename") or ""))
                raw_chapters.append((
                    title,
                    _workspace_reader_placeholder(
                        title,
                        "Raw PDF pages are extracted and cached when this section is opened.",
                    ),
                ))
                translated_path = str(entry.get("translated_path") or "")
                translated_html = ""
                translated_title = ""
                if translated_path and os.path.isfile(translated_path):
                    try:
                        with open(translated_path, "rb") as stream:
                            translated_html = stream.read().decode(
                                "utf-8", errors="replace"
                            )
                        if self._manifest.get("source_format") == "pdf":
                            from pdf_workspace_compiler import (
                                normalize_pdf_workspace_translated_html,
                            )

                            translated_html = normalize_pdf_workspace_translated_html(
                                translated_html,
                                str(self._manifest.get("workspace") or ""),
                            )
                        # Use the same translated-heading resolver as Book
                        # Details so the reader sidebar and details list cannot
                        # disagree. Raw mode keeps the source bookmark title.
                        translated_title = _read_translated_chapter_title(
                            translated_path
                        )
                    except OSError:
                        translated_html = ""
                if not translated_html:
                    translated_html = _workspace_reader_placeholder(
                        title,
                        "This section has not been translated yet.",
                    )
                translated_chapters.append((
                    translated_title or title,
                    translated_html,
                ))
        except Exception as exc:
            self.error.emit(str(exc))
            return
        if not self._should_stop():
            self.done.emit(raw_chapters, translated_chapters, filenames)


class EpubSearchMixin:
    """``_EpubSearchThread``: search plain chapter text, 120-row ordered batches.
    """

    def run(self):
        query = self._query.strip()
        if self._should_stop():
            return
        if not query:
            self.results_batch_ready.emit(self._search_id, query, [], True)
            return
        try:
            pattern = re.compile(re.escape(query), re.IGNORECASE)
        except re.error:
            self.results_batch_ready.emit(self._search_id, query, [], True)
            return

        def _scan_chapter(item):
            chapter_idx, chapter = item
            title, html = chapter
            if self._should_stop():
                return []
            plain = _epub_plain_chapter_text(html)
            if self._should_stop():
                return []
            rows = []
            local_occurrence = 0
            display_title = title or f"Chapter {chapter_idx + 1}"
            for match in pattern.finditer(plain):
                if self._should_stop():
                    return []
                rows.append({
                    "chapter_idx": chapter_idx,
                    "local_occurrence": local_occurrence,
                    "match_count": 1,
                    "text": query,
                    "title": display_title,
                    "excerpt": _epub_search_excerpt(
                        plain, match.start(), match.end(), radius=34),
                })
                local_occurrence += 1
            return rows

        batch: list[dict] = []
        total_matches = 0

        def _flush_batch(done: bool = False) -> None:
            nonlocal batch
            if self._should_stop():
                return
            if batch or done:
                self.results_batch_ready.emit(
                    self._search_id, query, list(batch), bool(done))
                batch = []

        def _append_rows(rows) -> None:
            nonlocal total_matches
            for row in rows or []:
                if self._should_stop():
                    return
                row["global_occurrence"] = total_matches
                total_matches += 1
                batch.append(row)
                if len(batch) >= 120:
                    _flush_batch(False)

        items = list(enumerate(self._chapters))
        try:
            workers = _reader_worker_count(len(items), config=self._config)
            if workers <= 1:
                for item in items:
                    if self._should_stop():
                        return
                    _append_rows(_scan_chapter(item))
            else:
                pool = ThreadPoolExecutor(max_workers=workers)
                futures = {
                    pool.submit(_scan_chapter, item): idx
                    for idx, item in enumerate(items)
                }
                pending: dict[int, list] = {}
                next_idx = 0
                try:
                    for future in as_completed(futures):
                        if self._should_stop():
                            break
                        idx = futures[future]
                        try:
                            pending[idx] = future.result()
                        except Exception:
                            logger.debug(
                                "EPUB search chapter failed: %s",
                                traceback.format_exc())
                            pending[idx] = []
                        while next_idx in pending:
                            if self._should_stop():
                                break
                            _append_rows(pending.pop(next_idx))
                            next_idx += 1
                finally:
                    if self._should_stop():
                        for future in futures:
                            future.cancel()
                        try:
                            pool.shutdown(wait=False, cancel_futures=True)
                        except TypeError:
                            pool.shutdown(wait=False)
                    else:
                        pool.shutdown(wait=True)
                if self._should_stop():
                    return
        except Exception:
            logger.debug("Parallel EPUB search failed, falling back: %s",
                         traceback.format_exc())
            for item in items:
                if self._should_stop():
                    return
                _append_rows(_scan_chapter(item))

        if self._should_stop():
            return
        _flush_batch(True)


class EpubLoaderMixin:
    """``_EpubLoaderThread``: parse an EPUB (spine first, lazy images) into the cache.
    """

    def run(self):
        if self._should_stop():
            return
        try:
            import ebooklib
            from ebooklib import epub as epub_mod
            from bs4 import BeautifulSoup

            lazy_image_members = _discover_epub_image_members(self._epub_path)

            class _LazyImageEpubReader(epub_mod.EpubReader):
                """Let ebooklib build image items without inflating payloads."""

                def read_file(reader_self, name):
                    normalized = str(name or "").replace("\\", "/").lstrip("/")
                    if (normalized.casefold() in lazy_image_members
                            or normalized.lower().endswith(_READER_IMAGE_EXTS)):
                        return b""
                    return super().read_file(name)

            epub_reader = _LazyImageEpubReader(
                self._epub_path, options={"ignore_ncx": True})
            book = epub_reader.load()
            epub_reader.process()
            if self._should_stop():
                return

            images: dict[str, object] = {}
            for item in book.get_items():
                if self._should_stop():
                    return
                item_name = item.get_name() or ""
                item_media = ""
                try:
                    item_media = item.get_media_type() or ""
                except Exception:
                    item_media = getattr(item, "media_type", "") or ""
                lower_name = item_name.lower()
                is_image = (
                    item.get_type() == ebooklib.ITEM_IMAGE
                    or str(item_media).lower().startswith("image/")
                    or lower_name.endswith(_READER_IMAGE_EXTS)
                )
                if not is_image or not item_name:
                    continue
                # Keep only the archive-member name here. Reading every image
                # eagerly made image-heavy books pay their full compressed I/O
                # and pickle cost before page one could render. The reader now
                # materializes only references used by the current chapter (or
                # the background-preloaded next chapter).
                descriptor = _lazy_epub_image(item_name)
                images[item_name] = descriptor
                stripped = item_name.lstrip("./")
                if stripped and stripped not in images:
                    images[stripped] = descriptor
                basename = os.path.basename(item_name)
                if basename and basename not in images:
                    images[basename] = descriptor

            # --- Chapter item resolution ------------------------------------
            #
            # Strategy (matches Calibre / KOReader / iBooks leniency):
            #
            #   1. **Spine-first.** Walk ``book.spine`` in reading order and
            #      pick up every item it references. The spine is
            #      authoritative; anything that's in the spine IS a content
            #      document regardless of what media-type the manifest
            #      declares. This fixes EPUBs produced by buggy tools (e.g.
            #      WebToEpub) that mark every chapter as
            #      ``media-type="text/html"`` — those become ITEM_UNKNOWN
            #      inside ebooklib and are invisible to
            #      ``get_items_of_type(ITEM_DOCUMENT)`` even though they're
            #      perfectly readable HTML.
            #
            #   2. **ITEM_DOCUMENT fallback.** If the spine is missing /
            #      empty / unusable, fall back to ebooklib's strict
            #      classification. This preserves the historical behavior
            #      for well-formed EPUBs whose spine pointer is broken but
            #      whose manifest is clean.
            #
            #   3. **Extension-only last resort.** If neither pass turned
            #      up anything beyond (at most) a cover page, sweep the
            #      full manifest and include every item whose filename ends
            #      in .html / .xhtml / .htm. Ordering is then manifest
            #      order — not ideal, but vastly better than an empty
            #      "No readable content" dialog.
            _HTML_EXTS = (".html", ".xhtml", ".htm")

            # ``chapter_items`` entries are (item, authoritative_flag). The
            # flag is True when the item was sourced from the spine or
            # ebooklib's ITEM_DOCUMENT pass — i.e. the author explicitly
            # declared it as reading content. Those items survive even when
            # they're text-light (cover pages, nav pages, TOC stubs, etc.).
            # False entries came from the extension-only last-resort sweep
            # and are still subject to the strict text filter so noisy
            # manifests don't dump random empty fragments into the TOC.
            chapter_items: list[tuple[object, bool]] = []
            seen_names: set[str] = set()

            def _add_item(it, authoritative: bool) -> None:
                if it is None:
                    return
                name = it.get_name() or ""
                if not name or name in seen_names:
                    return
                if not name.lower().endswith(_HTML_EXTS):
                    return
                seen_names.add(name)
                chapter_items.append((it, authoritative))

            # Pass 1: spine order (authoritative).
            try:
                spine = getattr(book, "spine", None) or []
                for entry in spine:
                    if self._should_stop():
                        return
                    # Spine entries are commonly (idref, linear_flag) but
                    # some producers emit a bare idref string. Accept both.
                    if isinstance(entry, (tuple, list)):
                        item_id = entry[0] if entry else None
                    else:
                        item_id = entry
                    if not item_id:
                        continue
                    _add_item(book.get_item_with_id(str(item_id)), True)
            except Exception:
                logger.debug("Spine walk failed: %s", traceback.format_exc())

            # Pass 2: ebooklib's ITEM_DOCUMENT classification (authoritative).
            if not chapter_items:
                for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                    if self._should_stop():
                        return
                    _add_item(item, True)

            # Pass 3: extension-only sweep across the whole manifest
            # (non-authoritative). Only runs when we found nothing (or just
            # a single cover-like item) via the authoritative passes, so
            # well-formed EPUBs don't pay any extra cost here.
            if len(chapter_items) <= 1:
                for item in book.get_items():
                    if self._should_stop():
                        return
                    _add_item(item, False)

            chapter_sources: list[tuple[object, str, bool]] = []
            for item, authoritative in chapter_items:
                if self._should_stop():
                    return
                try:
                    # "Show special files" toggle: when OFF, drop configured
                    # non-chapter pages so the TOC matches what the
                    # translator considers real chapters. This still
                    # respects spine ordering for the chapters that DO
                    # survive — we just prune the specials.
                    item_name = item.get_name() or ""
                    if not self._show_special_files and _is_special_spine_item(
                            item_name, self._config):
                        continue

                    chapter_sources.append((item, item_name, authoritative))
                except Exception:
                    logger.debug("Skipped chapter source: %s",
                                 traceback.format_exc())

            def _collect_chapter_payload(source):
                item, item_name, authoritative = source
                if self._should_stop():
                    return None
                try:
                    raw_content = item.get_content()
                    if isinstance(raw_content, bytes):
                        content = raw_content.decode("utf-8", errors="replace")
                    else:
                        content = str(raw_content or "")
                    if self._should_stop():
                        return None
                    return item_name, authoritative, content
                except Exception:
                    logger.debug("Skipped chapter payload: %s",
                                 traceback.format_exc())
                    return None

            try:
                workers = _reader_worker_count(
                    len(chapter_sources), config=self._config)
                if workers <= 1:
                    collected_payloads = [
                        _collect_chapter_payload(source)
                        for source in chapter_sources
                    ]
                else:
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        collected_payloads = list(pool.map(
                            _collect_chapter_payload, chapter_sources))
            except Exception:
                logger.debug("Parallel EPUB content collection failed, "
                             "falling back: %s", traceback.format_exc())
                collected_payloads = [
                    _collect_chapter_payload(source)
                    for source in chapter_sources
                ]

            if self._should_stop():
                return
            chapter_payloads: list[tuple[str, bool, str]] = [
                payload for payload in collected_payloads if payload
            ]

            def _parse_chapter_payload(payload):
                item_name, authoritative, content = payload
                if self._should_stop():
                    return None
                try:
                    soup = BeautifulSoup(content, "html.parser")
                    text = soup.get_text(strip=True)
                    # Non-authoritative items (pass 3) must clear a minimum
                    # text bar to keep the TOC free of fragmentary noise.
                    # Authoritative items (spine / ITEM_DOCUMENT) are kept
                    # even when text-light because the author put them in
                    # the reading order deliberately — e.g. the cover page
                    # (just an <img>) or a navigation/TOC page whose visible
                    # text is mostly the chapter titles themselves.
                    if not authoritative and (not text or len(text) < 10):
                        return None
                    # Authoritative-but-totally-empty items (no text AND no
                    # images AND no links) are still dropped — they're
                    # almost always accidental spine entries (e.g. a
                    # placeholder that never got populated).
                    if authoritative and not text:
                        has_img = bool(soup.find("img"))
                        has_svg = bool(soup.find("svg"))
                        has_link = bool(soup.find("a"))
                        if not (has_img or has_svg or has_link):
                            return None

                    title = None
                    title_tag = soup.find("title")
                    if title_tag and title_tag.string:
                        title = title_tag.string.strip()
                    if not title:
                        for heading in soup.find_all(["h1", "h2", "h3"]):
                            ht = heading.get_text(strip=True)
                            if ht:
                                title = ht
                                break
                    if not title:
                        title = os.path.splitext(os.path.basename(item_name))[0]
                        title = title.replace("_", " ").replace("-", " ").title()
                    if len(title) > 50:
                        title = title[:47] + "\u2026"
                    if self._should_stop():
                        return None
                    return title, content, item_name
                except Exception:
                    logger.debug("Skipped chapter parse: %s",
                                 traceback.format_exc())
                    return None

            chapters: list[tuple[str, str]] = []
            filenames: list[str] = []
            try:
                workers = _reader_worker_count(
                    len(chapter_payloads), config=self._config)
                if workers <= 1:
                    parsed_chapters = [
                        _parse_chapter_payload(payload)
                        for payload in chapter_payloads
                    ]
                else:
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        parsed_chapters = list(pool.map(
                            _parse_chapter_payload, chapter_payloads))
            except Exception:
                logger.debug("Parallel EPUB chapter parse failed, "
                             "falling back: %s", traceback.format_exc())
                parsed_chapters = [
                    _parse_chapter_payload(payload)
                    for payload in chapter_payloads
                ]

            if self._should_stop():
                return
            for parsed in parsed_chapters:
                if not parsed:
                    continue
                title, content, item_name = parsed
                chapters.append((title, content))
                # Record the source item name (e.g. 'OEBPS/chapter0001.xhtml')
                # in parallel so downstream code can correlate reader
                # chapters with spine filenames.
                filenames.append(item_name)

            # Write to cache (avoids emitting large data through Qt signals).
            # Key-scoped by the Show-special-files state so the two
            # variants don't overwrite each other.
            _save_epub_cache(
                self._epub_path, chapters, images, filenames,
                show_special_files=self._show_special_files,
                config=self._config,
            )
            if not self._should_stop():
                self.done.emit()
        except Exception as exc:
            if not self._should_stop():
                logger.error("EPUB load error: %s\n%s", exc, traceback.format_exc())
                self.error.emit(f"{exc}\n\n{traceback.format_exc()}")


class ReaderDocMixin:
    """``EpubReaderDialog``'s page builder: themes, image materialisation, embedded CSS
    and the paged / scroll document. Hook ``_reader_file_url(path)`` (desktop:
    ``QUrl.fromLocalFile``; mobile: the in-app server URL).
    """

    def _get_theme(self):
        idx = self._theme_index if 0 <= self._theme_index < len(_READER_THEMES) else 0
        return _READER_THEMES[idx]

    def _all_chapters_html(self) -> str:
        """The "Scroll All" body: every chapter under a numbered heading."""
        all_html = ""
        for idx, (title, content) in enumerate(self._chapters):
            processed = self._process_html(content)
            chapter_number = self._reader_chapter_display_number(idx)
            all_html += f"<h2 style='color: {self._get_theme()['heading']}; border-bottom: 1px solid {self._get_theme()['border']}; padding-bottom: 6px; margin-top: 30px;'>Chapter {chapter_number}: {title}</h2>\n{processed}\n<hr style='border: none; border-top: 1px solid {self._get_theme()['border']}; margin: 20px 0;'>"
        return all_html

    def _reader_chapter_display_number(self, row):
        try:
            return self._chapter_display_numbers[int(row)]
        except (AttributeError, IndexError, TypeError, ValueError):
            return int(row) + 1

    def _ensure_reader_image_temp_dir(self) -> str:
        """Return a per-source image cache directory, invalidated by mtime."""
        current = getattr(self, "_img_temp_dir", "")
        if current:
            os.makedirs(current, exist_ok=True)
            return current
        try:
            stamp = os.path.getmtime(self._epub_path)
        except OSError:
            stamp = 0
        source_key = f"{self._epub_path}|{stamp}"
        epub_hash = hashlib.md5(source_key.encode()).hexdigest()[:10]
        current = os.path.join(
            tempfile.gettempdir(), "Glossarion_EpubImages", epub_hash)
        os.makedirs(current, exist_ok=True)
        self._img_temp_dir = current
        return current

    def _close_epub_image_zip(self) -> None:
        zf = getattr(self, "_epub_image_zip", None)
        self._epub_image_zip = None
        self._epub_image_zip_path = ""
        self._epub_image_zip_names = {}
        if zf is not None:
            try:
                zf.close()
            except Exception:
                pass

    def _load_reader_image_resource(self, resource: dict) -> bytes:
        """Load one resolved image resource, reusing the active EPUB archive."""
        kind = resource.get("kind")
        if kind == "bytes":
            return resource.get("data") or b""
        if kind == "file":
            try:
                with open(resource.get("path") or "", "rb") as f:
                    return f.read()
            except OSError:
                return b""
        if kind != "epub" or not self._epub_path:
            return b""
        try:
            import zipfile

            active_path = os.path.abspath(self._epub_path)
            zf = getattr(self, "_epub_image_zip", None)
            if (zf is None
                    or getattr(self, "_epub_image_zip_path", "") != active_path):
                self._close_epub_image_zip()
                zf = zipfile.ZipFile(active_path, "r")
                self._epub_image_zip = zf
                self._epub_image_zip_path = active_path
                self._epub_image_zip_names = {
                    name.casefold(): name for name in zf.namelist()
                }
            return _read_epub_member_from_zip(
                zf,
                resource.get("member") or "",
                self._epub_image_zip_names,
            )
        except Exception:
            logger.debug("Lazy EPUB image extraction failed: %s",
                         traceback.format_exc())
            self._close_epub_image_zip()
            return b""

    def _invalidate_processed_reader_cache(self) -> None:
        """Invalidate chapter/image metadata after the resource set changes."""
        self._image_cache_generation = int(getattr(
            self, "_image_cache_generation", 0) or 0) + 1
        getattr(self, "_processed_html_cache", {}).clear()
        getattr(self, "_image_sizeable_cache", {}).clear()
        getattr(self, "_preloaded_chapter_keys", set()).clear()
        self._pending_image_chapter_activation = None

    def _set_reader_images(self, images: dict | None) -> None:
        """Install an image map and invalidate rendering only when it changed."""
        new_images = images or {}
        signature = _reader_image_map_signature(new_images)
        changed = signature != getattr(self, "_image_resource_signature", ())
        self._images = new_images
        self._image_resource_signature = signature
        if changed:
            self._invalidate_processed_reader_cache()

    def _processed_reader_html_key(self, html_content: str) -> str:
        digest = hashlib.md5(str(html_content or "").encode("utf-8")).hexdigest()
        dirs = "|".join(os.path.abspath(str(path)) for path in
                        (getattr(self, "_extra_image_dirs", []) or []))
        generation = int(getattr(self, "_image_cache_generation", 0) or 0)
        return f"{generation}|{self._epub_path}|{dirs}|{digest}"

    def _chapter_image_preload_key(self, row: int,
                                   html_content: str = "") -> str:
        """Return the generation-scoped preload key for one chapter."""
        if not html_content and 0 <= row < len(self._chapters):
            html_content = str(self._chapters[row][1] or "")
        generation = int(getattr(self, "_image_cache_generation", 0) or 0)
        digest = hashlib.md5(str(html_content).encode("utf-8")).hexdigest()
        return f"{generation}:{row}:{digest}"

    def _process_html(self, html_content: str) -> str:
        """Process chapter HTML: resolve image paths to temp files."""
        cache = getattr(self, "_processed_html_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._processed_html_cache = cache
        cache_key = self._processed_reader_html_key(html_content)
        cached_html = cache.get(cache_key)
        if cached_html is not None:
            return cached_html
        try:
            from bs4 import BeautifulSoup

            temp_dir = self._ensure_reader_image_temp_dir()
            preloaded = getattr(self, "_preloaded_image_resources", None)
            if not isinstance(preloaded, dict):
                preloaded = {}
                self._preloaded_image_resources = preloaded
            sizeable_cache = getattr(self, "_image_sizeable_cache", None)
            if not isinstance(sizeable_cache, dict):
                sizeable_cache = {}
                self._image_sizeable_cache = sizeable_cache

            # Translation output can contain safely escaped markup when it was
            # produced before a tag was added to the shared HTML allowlist.
            # Rehydrate known tags here so existing workspaces also benefit
            # from reader fixes without requiring a fresh translation run.
            html_content = unescape_valid_html_tag_entities(html_content)
            soup = BeautifulSoup(html_content, "html.parser")

            def _materialize_image(src: str, classify_size: bool = True):
                """Return (local URL, sizeable) using preload/cache state."""
                resource = _reader_image_resource(
                    src,
                    getattr(self, "_images", {}) or {},
                    getattr(self, "_extra_image_dirs", []) or [],
                    getattr(self, "_epub_path", "") or "",
                )
                if not resource:
                    return "", False
                identity = str(resource["identity"])
                warmed = preloaded.get(identity) or {}
                warmed_path = str(warmed.get("path") or "")
                classification_ready = (
                    not classify_size
                    or identity in sizeable_cache
                    or bool(warmed.get("classified"))
                )
                if (warmed_path and os.path.isfile(warmed_path)
                        and classification_ready):
                    sizeable = bool(sizeable_cache.get(
                        identity, warmed.get("sizeable", False)))
                    return self._reader_file_url(warmed_path), sizeable

                image_data = self._load_reader_image_resource(resource)
                if not image_data:
                    return "", False
                img_path = _write_reader_image_cache(
                    temp_dir,
                    src,
                    image_data,
                    resource.get("path") or "",
                )
                sizeable = False
                if classify_size:
                    if identity not in sizeable_cache:
                        sizeable_cache[identity] = _reader_image_is_sizeable(
                            image_data)
                    sizeable = bool(sizeable_cache[identity])
                preloaded[identity] = {
                    "path": img_path,
                    "sizeable": sizeable,
                    "classified": bool(classify_size),
                }
                return self._reader_file_url(img_path), sizeable

            # Pre-pass: split <p> tags that contain multiple <img> tags
            # into separate <p> tags, one per image. Without this, the
            # full-page-img wrapper grabs the parent <p> and both images
            # end up in one column, clipping the second image.
            #   Before: <p><img/><br/><img/></p>
            #   After:  <p><img/></p><p><img/></p>
            for p_tag in soup.find_all('p'):
                imgs_in_p = p_tag.find_all('img', recursive=False)
                if len(imgs_in_p) < 2:
                    continue
                # Collect all children, split into groups at each <img>.
                # Each group becomes its own <p>.
                groups = []
                current_group = []
                for child in list(p_tag.children):
                    child.extract()
                    if child.name == 'img':
                        # Start a new group for each image
                        if current_group:
                            groups.append(current_group)
                            current_group = []
                        current_group.append(child)
                    elif child.name == 'br':
                        # Drop <br/> separators between images
                        continue
                    else:
                        current_group.append(child)
                if current_group:
                    groups.append(current_group)
                # Replace original <p> with split groups
                for group in reversed(groups):
                    new_p = soup.new_tag('p')
                    for el in group:
                        new_p.append(el)
                    p_tag.insert_after(new_p)
                p_tag.decompose()

            for img_tag in soup.find_all("img"):
                src = img_tag.get("src", "")
                if not src:
                    continue
                # Let Chromium present text/layout without waiting for a large
                # scan to finish decoding. This preserves the original image
                # bytes and dimensions; it only changes decode scheduling.
                if not img_tag.get("decoding"):
                    img_tag["decoding"] = "async"
                # Remote image URLs are loaded directly by QWebEngine. Do not
                # reinterpret them as relative filesystem paths; the reader
                # view explicitly permits its local file:// page to request
                # HTTP(S) image resources.
                if _url_scheme(src) in ("http", "https"):
                    continue
                image_url, image_is_sizeable = _materialize_image(src)
                if image_url:
                    img_tag["src"] = image_url
                    if image_is_sizeable:
                        wrapper = soup.new_tag("div")
                        wrapper["class"] = ["full-page-img"]
                        # Find the block-level container of this img
                        # (typically <p><img/></p> or <div><img/></div>).
                        # Do not wrap a mixed content parent like:
                        #   <div><img/><img/><h1>...</h1><p>...</p></div>
                        # because the full-page wrapper clips overflow in
                        # paginated modes and would hide the translated text.
                        container = img_tag
                        if img_tag.parent and img_tag.parent.name in ('p', 'div', 'figure'):
                            parent = img_tag.parent
                            parent_imgs = parent.find_all('img')
                            parent_text = parent.get_text(" ", strip=True)
                            if len(parent_imgs) == 1 and len(parent_text) <= 240:
                                container = parent
                        # Collect preceding siblings to pull into the wrapper:
                        #   header + p + img, header + img, p + img, or just img
                        _HEADERS = ('h1', 'h2', 'h3', 'h4', 'h5', 'h6')
                        to_pull = []  # elements to insert before the image
                        prev = container.find_previous_sibling()
                        if prev and prev.name == 'p' and not prev.find('img'):
                            to_pull.append(prev)
                            prev2 = prev.find_previous_sibling()
                            if prev2 and prev2.name in _HEADERS:
                                to_pull.append(prev2)
                        elif prev and prev.name in _HEADERS:
                            to_pull.append(prev)
                        # A full-page image normally starts a fresh column.
                        # When it is the first meaningful item in a chapter,
                        # however, that break creates an entirely blank first
                        # page and strands the image in the next column. Mark
                        # that leading case explicitly so paginated CSS can
                        # suppress only the unnecessary initial break.
                        leading_node = to_pull[-1] if to_pull else container
                        has_content_before = False
                        for sibling in leading_node.previous_siblings:
                            sibling_name = getattr(sibling, "name", None)
                            if sibling_name:
                                if (
                                    sibling_name == "a"
                                    and not sibling.get_text(" ", strip=True)
                                    and not sibling.find("img")
                                ):
                                    continue
                                has_content_before = True
                                break
                            if str(sibling).strip():
                                has_content_before = True
                                break
                        if not has_content_before:
                            wrapper["class"].append("full-page-img-first")
                        # Extract siblings, wrap container, then re-insert in order
                        for el in to_pull:
                            el.extract()
                        container.wrap(wrapper)
                        for el in reversed(to_pull):
                            wrapper.insert(0, el)

            # SVG uses <image href="..."> or the EPUB2-compatible
            # <image xlink:href="..."> instead of HTML's <img src="...">.
            # Resolve all common spellings to the same cached local resources.
            svg_href_attrs = (
                "href",
                "xlink:href",
                "{http://www.w3.org/1999/xlink}href",
            )
            for image_tag in soup.find_all("image"):
                href_attr = next(
                    (attr for attr in svg_href_attrs if image_tag.get(attr)),
                    None,
                )
                if not href_attr:
                    continue
                src = image_tag.get(href_attr, "")
                if not src or _url_scheme(src) in ("http", "https"):
                    continue
                image_url, _sizeable = _materialize_image(
                    src, classify_size=False)
                if image_url:
                    image_tag[href_attr] = image_url

            processed = str(soup)
            cache[cache_key] = processed
            return processed
        except Exception:
            logger.debug("HTML processing failed: %s", traceback.format_exc())
            return html_content

    def _get_embedded_css(self) -> str:
        """Lazily extract and return the EPUB's embedded CSS.

        Sources (highest priority first):
          0. ``EPUB_CSS_OVERRIDE_PATH`` env var — if set, this is the
             **only** CSS used (matches what the compiled EPUB gets).
          1. CSS and font files inside the EPUB zip.
          2. Font files in the extracted folder's ``fonts/`` subdirectory.
          3. CSS files in the extracted folder's ``css/`` subdirectory
             (appended after zip CSS so they can override originals).

        Font ``url(...)`` references are rewritten to inline ``data:``
        URIs so the result is self-contained.
        """
        cache_attr = '_embedded_css_cache'
        attach_css_enabled = self._resolve_attach_css_to_chapters()
        translated_view = bool(
            not getattr(self, "_show_raw", False)
            and (
                self._translated_overlay
                or self._raw_epub_alt_path
                or self._translated_css_dirs
                or self._workspace_mode
            )
        )
        translated_css_mode = bool(
            attach_css_enabled
            and (self._translated_overlay or self._workspace_mode)
            and not getattr(self, "_show_raw", False)
            and self._translated_css_dirs
        )
        suppress_translated_css = translated_view and not attach_css_enabled
        cache_key = (
            "translated" if translated_css_mode else "active_epub",
            os.path.abspath(str(self._epub_path or "")),
            tuple(os.path.abspath(p) for p in self._translated_css_dirs),
            int(bool(attach_css_enabled)),
            int(bool(suppress_translated_css)),
        )
        if hasattr(self, cache_attr):
            cached = getattr(self, cache_attr)
            if isinstance(cached, tuple) and len(cached) == 2:
                old_key, old_css = cached
                if old_key == cache_key:
                    return old_css
            elif isinstance(cached, str) and not translated_css_mode:
                return cached

        if suppress_translated_css:
            setattr(self, cache_attr, (cache_key, ""))
            return ""

        import zipfile, re, base64
        css_text = ''
        font_data: dict[str, bytes] = {}
        _FONT_EXTS = ('.ttf', '.otf', '.woff', '.woff2')

        def _collect_fonts(folder: str) -> None:
            if not folder or not os.path.isdir(folder):
                return
            try:
                for root, _dirs, files in os.walk(folder):
                    for fname in files:
                        ext = os.path.splitext(fname)[1].lower()
                        if ext not in _FONT_EXTS:
                            continue
                        bname = fname.lower()
                        if bname in font_data:
                            continue
                        try:
                            with open(os.path.join(root, fname), 'rb') as ff:
                                font_data[bname] = ff.read()
                        except OSError:
                            pass
            except OSError:
                pass

        def _append_css_dir(css_dir: str) -> None:
            nonlocal css_text
            if not css_dir or not os.path.isdir(css_dir):
                return
            try:
                for fname in sorted(os.listdir(css_dir)):
                    if not fname.lower().endswith('.css'):
                        continue
                    try:
                        with open(os.path.join(css_dir, fname), 'r',
                                  encoding='utf-8', errors='replace') as cf:
                            css_text += cf.read() + '\n'
                    except OSError:
                        pass
            except OSError:
                pass

        # --- Source 0: Explicit CSS override from the GUI -------------------
        override_path = os.environ.get('EPUB_CSS_OVERRIDE_PATH', '').strip()
        has_override = bool(override_path and os.path.isfile(override_path))
        if has_override:
            try:
                with open(override_path, 'r', encoding='utf-8',
                          errors='replace') as f:
                    css_text = f.read() + '\n'
            except OSError:
                has_override = False  # fall through to normal sources

        # In in-progress translated reader mode, ``_epub_path`` still points
        # at the raw EPUB because translated HTML is overlaid in memory. Pull
        # Embedded CSS from the output folder instead, so translated view uses
        # dist/<book>/css/style.css while Raw keeps the source EPUB styling.
        if translated_css_mode:
            for css_dir in self._translated_css_dirs:
                css_abs = os.path.abspath(css_dir)
                output_root = (
                    os.path.dirname(css_abs)
                    if os.path.basename(css_abs).lower() == "css"
                    else css_abs
                )
                _collect_fonts(os.path.join(output_root, 'fonts'))
                _collect_fonts(css_abs)
                if not has_override:
                    _append_css_dir(css_abs)

        # --- Source 1: EPUB zip contents ------------------------------------
        # Always read fonts from the zip; only read CSS if no override.
        try:
            with zipfile.ZipFile(self._epub_path, 'r') as zf:
                for entry in zf.namelist():
                    ext = os.path.splitext(entry)[1].lower()
                    if ext in _FONT_EXTS:
                        bname = os.path.basename(entry).lower()
                        if bname not in font_data:
                            font_data[bname] = zf.read(entry)
                if not has_override and not translated_css_mode:
                    for entry in zf.namelist():
                        if entry.lower().endswith('.css'):
                            try:
                                raw_css = zf.read(entry).decode('utf-8', errors='replace')
                                css_text += raw_css + '\n'
                            except Exception:
                                pass
        except Exception:
            pass

        # --- Source 2 & 3: Extracted folder on disk -------------------------
        epub_dir = os.path.dirname(self._epub_path) if self._epub_path else ''
        if epub_dir:
            # Fonts from <epub_dir>/fonts/ (supplement zip fonts)
            fonts_dir = os.path.join(epub_dir, 'fonts')
            _collect_fonts(fonts_dir)
            # CSS from <epub_dir>/css/ (only if no override)
            if not has_override and not translated_css_mode:
                css_dir = os.path.join(epub_dir, 'css')
                _append_css_dir(css_dir)

        # --- Rewrite font URLs to data URIs ---------------------------------
        def _replace_font_url(m):
            url_val = m.group(1)
            bname = os.path.basename(url_val).lower()
            if bname in font_data:
                ext = os.path.splitext(bname)[1]
                mime_map = {'.ttf': 'font/ttf', '.otf': 'font/otf',
                            '.woff': 'font/woff', '.woff2': 'font/woff2'}
                mime = mime_map.get(ext, 'application/octet-stream')
                b64 = base64.b64encode(font_data[bname]).decode('ascii')
                return f'url(data:{mime};base64,{b64})'
            return m.group(0)

        css_text = re.sub(
            r'url\s*\(\s*["\']?([^"\')\s]+\.(?:ttf|otf|woff2?))["\'\s]*\)',
            _replace_font_url, css_text)

        setattr(self, cache_attr, (cache_key, css_text))
        return css_text

    def _resolve_attach_css_to_chapters(self) -> bool:
        """Return the effective Attach CSS setting for reader CSS handling."""
        env = os.environ.get("ATTACH_CSS_TO_CHAPTERS", "").strip().lower()
        if env in ("1", "true", "yes", "on"):
            return True
        if env in ("0", "false", "no", "off"):
            return False
        return bool(self._config.get("attach_css_to_chapters", False))

    def _wrap_html(self, body_html: str, paginated: bool = False,
                   spread_pages: int = 1) -> str:
        """Wrap processed HTML in a full styled document.

        When *paginated* is True, a proper CSS multi-column layout is used:
          html/body — zero-padded, overflow:hidden (viewport clip)
          #columns  — column layout container (translateX for navigation)
          #content  — inner padding for readability
        """
        t = self._get_theme()
        _use_embedded = (self._font_family or '').strip() == 'Embedded CSS'
        # Build a CSS font stack: user-selected family first, then common
        # fallbacks so missing fonts degrade gracefully. Any embedded single
        # quotes in the family name are stripped to keep the stylesheet valid.
        _fam = (self._font_family or 'Georgia').replace("'", "").strip() or 'Georgia'
        _is_mono = _fam.lower() in {'consolas', 'courier new', 'courier', 'menlo',
                                    'monaco', 'lucida console', 'cascadia mono',
                                    'cascadia code', 'source code pro', 'fira code'}
        _generic = 'monospace' if _is_mono else 'serif'
        _font_stack = f"'{_fam}', 'Georgia', 'Noto Serif', {_generic}"
        # When using embedded CSS, inject the EPUB's own stylesheet and
        # let its @font-face / font-family rules take precedence.
        _embedded_css_block = ''
        if _use_embedded:
            epub_css = self._get_embedded_css()
            if epub_css:
                _embedded_css_block = epub_css
                # Don't override font-family — let the embedded CSS dictate it
                _font_stack = "inherit"
        # Use px units (integer device pixels) for sharper glyph rasterization.
        # 1pt = 1/72 inch, 1px = 1/96 inch → px = pt * 96/72.
        _font_px = int(round(self._font_size * 96 / 72))
        _has_embedded_css = bool(_use_embedded and _embedded_css_block)
        # Older PDF extraction output used <h3> for every body text block.
        # Browser defaults make those paragraphs bold even though the source
        # PDF is regular weight. Scope the compatibility rule to translated
        # PDF workspaces so real EPUB headings and raw PDF markup are intact.
        _pdf_workspace_body_css = (
            "body h3 { font-size: 1em; font-weight: normal !important; margin: 0.6em 0; "
            "padding: 0; }"
            ".pdf-fast-semantic-page p.pdf-align-left { text-align: left !important; }"
            ".pdf-fast-semantic-page p.pdf-align-center { text-align: center !important; }"
            ".pdf-fast-semantic-page p.pdf-align-right { text-align: right !important; }"
            ".pdf-fast-semantic-page p.pdf-align-justify { "
            "text-align: justify !important; text-justify: auto; }"
            if (getattr(self, "_workspace_mode", False)
                and not getattr(self, "_show_raw", False))
            else ""
        )
        _pdf_workspace_rtl_css = (
            "body, .pdf-fast-semantic-page, .pdf-fast-layout-page { direction: rtl; }"
            ".pdf-fast-semantic-page p, .pdf-fast-semantic-page li, "
            ".pdf-fast-semantic-page td, .pdf-fast-semantic-page th { "
            "direction: rtl; unicode-bidi: plaintext; }"
            + (
                ".pdf-fast-semantic-page p.pdf-align-left { "
                "text-align: right !important; }"
                if os.environ.get("PDF_PARAGRAPH_ALIGNMENT", "source") == "source"
                else ""
            )
            + ".pdf-fast-semantic-page p.pdf-align-justify { "
            "text-align-last: right !important; }"
            if (
                getattr(self, "_workspace_mode", False)
                and os.environ.get("PDF_RTL_PARAGRAPH_LAYOUT", "0") == "1"
            )
            else ""
        )
        if paginated:
            _spread_pages = max(1, int(spread_pages or 1))
            return (
                f"<html><head><style>"
                f"{_embedded_css_block}"
                f"* {{ box-sizing: border-box; }}"
                # Grayscale AA + geometricPrecision kill the subpixel LCD
                # fringing ("red shift") that appears on text inside a
                # GPU-composited transformed layer. This trade-off is
                # intentional for paginated modes; see _js_scroll_to.
                f"html, body {{ margin: 0; padding: 10px 0 26px 0; overflow: hidden; "
                f"background: {t['bg']}; color: {t['fg']}; "
                f"-webkit-font-smoothing: antialiased; "
                f"-moz-osx-font-smoothing: grayscale; "
                f"text-rendering: geometricPrecision; "
                f"-webkit-text-size-adjust: 100%; }}"
                f"#columns {{ column-fill: auto; column-gap: 1px; "
                f"transition: none; opacity: 0; "
                f"overflow: hidden; width: 100%; transform: none; "
                f"font-family: {_font_stack}; "
                f"font-size: {_font_px}px; line-height: {self._line_spacing}; }}"
                f"#content {{ padding: 0 40px; overflow-wrap: anywhere; word-break: normal; }}"
                # Calibre explicitly suppresses leading page/column breaks
                # because EPUB CSS often puts break-before on the first
                # block, which Chromium turns into a blank first page.
                f"#content > :first-child, #content > div:first-child > :first-child "
                f"{{ break-before: avoid !important; page-break-before: avoid !important; }}"
                f"h1, h2, h3, h4, h5, h6 {{ color: {t['heading']}; margin: 0; padding: 0; }}"
                f"{_pdf_workspace_body_css}"
                f"{_pdf_workspace_rtl_css}"
                f"img, svg {{ display: block; max-width: 100%; max-height: calc(100vh - 60px); "
                f"height: auto; object-fit: contain; "
                f"border-radius: 4px; margin: 12px auto; break-inside: avoid; }}"
                f".full-page-img {{ break-inside: avoid; break-before: column; "
                f"display: flex; flex-direction: column; align-items: center; justify-content: center; "
                f"min-height: calc(100vh - 40px); overflow: hidden; "
                f"padding: 0; margin: 0; }}"
                f".full-page-img-first {{ break-before: avoid !important; }}"
                f"#content > .full-page-img:first-child {{ break-before: avoid !important; }}"
                f".full-page-img + .full-page-img {{ margin-top: 0; break-before: column; }}"
                f".full-page-img img {{ margin: 0 auto; max-height: calc(100vh - 100px); }}"
                f".full-page-img h1, .full-page-img h2, .full-page-img h3, "
                f".full-page-img h4, .full-page-img h5, .full-page-img h6 "
                f"{{ margin: 4px 0 8px 0; flex-shrink: 0; }}"
                f".full-page-img p {{ margin: 4px 0; flex-shrink: 0; text-align: center; "
                f"font-size: 0.9em; max-width: 80%; }}"
                f"p {{ margin: 0.6em 0; orphans: 2; widows: 2; }}"
                f"a {{ color: {t['link']}; }}"
                f"code {{ background: {t['code_bg']}; padding: 1px 4px; border-radius: 3px; }}"
                f"</style>"
                f"<script>"
                f"var _PAGE_W = 0;"
                f"var _PAGE_GAP = 1;"
                f"var _SPREAD_PAGES = {_spread_pages};"
                f"function _viewerWidthFor(c) {{"
                f"  var r = c ? c.getBoundingClientRect() : null;"
                f"  return Math.max(1, Math.floor((c && c.clientWidth) || "
                f"    (r && r.width) || document.documentElement.clientWidth || "
                f"    window.innerWidth || 1));"
                f"}}"
                f"function _pageWidthFor(c) {{"
                f"  var visible = Math.max(1, _SPREAD_PAGES || 1);"
                f"  var viewportW = _viewerWidthFor(c);"
                f"  _PAGE_W = Math.max(1, Math.floor((viewportW - "
                f"    ((visible - 1) * _PAGE_GAP)) / visible));"
                f"  return _PAGE_W;"
                f"}}"
                f"function _pageCountFor(c) {{"
                f"  if (!c) return 1;"
                f"  var gap = Math.max(0, _PAGE_GAP || 0);"
                f"  var span = Math.max(1, _pageWidthFor(c) + gap);"
                # Chromium exposes scrollWidth as an integer even when page
                # geometry crosses fractional device pixels.  A final column
                # can therefore measure a fraction below the ideal multiple;
                # ceil keeps that real column navigable while floor drops it.
                f"  return Math.max(1, Math.ceil((c.scrollWidth + gap) / span));"
                f"}}"
                # _CURRENT_PAGE is maintained by _js_scroll_to so that
                # _setupColumns() can re-anchor the transform whenever the
                # viewport width changes (window resize, TOC toggle). Without
                # this, changing column widths left the old translateX offset
                # stale and required hiding/revealing content to mask the jump.
                f"var _CURRENT_PAGE = 0;"
                f"function _setupColumns() {{"
                f"  var c = document.getElementById('columns');"
                f"  if (!c) return;"
                # Floor to integer pixels so column boundaries and the
                # translate offset (page * _PAGE_W) always land on whole
                # pixels — prevents subpixel text rendering shifts.
                f"  _PAGE_W = _pageWidthFor(c);"
                f"  c.style.columnWidth = _PAGE_W + 'px';"
                f"  c.style.columnGap = _PAGE_GAP + 'px';"
                f"  c.style.height = (window.innerHeight - 36) + 'px';"
                # Re-apply the current page offset to the new column width in
                # the browser's native inline scroll coordinates. Calibre's
                # paged mode does the same kind of native column scrolling;
                # keeping search and paging in this coordinate system lets
                # Ctrl+F jump to exact matches instead of only the spine item.
                f"  var _t = c.style.transition;"
                f"  c.style.transition = 'none';"
                f"  c.style.transform = 'none';"
                f"  c.scrollLeft = Math.round(_CURRENT_PAGE * (_PAGE_W + _PAGE_GAP));"
                f"  void c.offsetHeight;"
                f"  c.style.transition = _t || 'none';"
                f"  /* Clean up whitespace between consecutive full-page images */"

                f"  var imgs = c.querySelectorAll('.full-page-img');"
                f"  imgs.forEach(function(el) {{"
                f"    var next = el.nextSibling;"
                f"    while (next && next.nodeType === 3 && !next.textContent.trim()) {{"
                f"      var toRemove = next;"
                f"      next = next.nextSibling;"
                f"      toRemove.parentNode.removeChild(toRemove);"
                f"    }}"

                f"  }});"
                f"}}"
                f"document.addEventListener('DOMContentLoaded', _setupColumns);"
                f"window.addEventListener('resize', _setupColumns);"
                f"</script>"
                f"</head><body>"
                f"<div id='columns'><div id='content'>{body_html}</div></div>"
                f"</body></html>"
            )
        else:
            # Non-paginated (scroll / all) modes: no GPU-composited transform,
            # so we let Chromium use native OS rendering (ClearType subpixel
            # AA on Windows) which is noticeably sharper than forced
            # grayscale AA.
            _scroll_typography = (
                ""
                if _has_embedded_css
                else f"font-family: {_font_stack}; "
                     f"font-size: {_font_px}px; line-height: {self._line_spacing}; "
            )
            _scroll_heading_css = (
                "" if _has_embedded_css
                else f"h1, h2, h3 {{ color: {t['heading']}; }}"
            )
            _scroll_paragraph_css = (
                "" if _has_embedded_css else "p { margin: 0.6em 0; }"
            )
            return (
                f"<html><head><style>"
                f"{_embedded_css_block}"
                f"body {{ background: {t['bg']}; color: {t['fg']}; "
                f"{_scroll_typography}"
                f"-webkit-font-smoothing: auto; "
                f"-moz-osx-font-smoothing: auto; "
                f"text-rendering: optimizeLegibility; "
                f"-webkit-text-size-adjust: 100%; "
                f"padding: 10px 20px 28px 20px; margin: 0 auto; }}"
                f"{_scroll_heading_css}"
                f"{_pdf_workspace_body_css}"
                f"{_pdf_workspace_rtl_css}"
                f"img, svg {{ display: block; max-width: 100%; height: auto; "
                f"border-radius: 4px; margin: 12px auto; }}"
                f"{_scroll_paragraph_css}"
                f"a {{ color: {t['link']}; }}"
                f"code {{ background: {t['code_bg']}; padding: 1px 4px; border-radius: 3px; }}"
                f"::-webkit-scrollbar {{ width: 8px; }}"
                f"::-webkit-scrollbar-track {{ background: {t['bg']}; }}"
                f"::-webkit-scrollbar-thumb {{ background: {t['border']}; border-radius: 4px; }}"
                f"</style></head><body>{body_html}</body></html>"
            )

    def _reader_file_url(self, path: str) -> str:
        """URL a materialised image is referenced by (desktop overrides with ``QUrl``)."""
        return Path(path).as_uri()


# ---------------------------------------------------------------------------
# Public API (U5): plain-object jobs over the mixins above, for Glossarion Mobile
# and any caller without Qt. Every function returns what the desktop thread emits.
# ---------------------------------------------------------------------------

READER_THEMES = _READER_THEMES
READER_THEME_NAMES = tuple(theme["name"] for theme in _READER_THEMES)
READER_LAYOUTS = (LAYOUT_SINGLE, LAYOUT_SCROLL, LAYOUT_ALL, LAYOUT_DOUBLE)
#: Prefix of the reader page's console events (flet-webview ``on_console_message``).
MOBILE_EVENT_PREFIX = "GLRDR:"
#: Same-origin fallback endpoint for reader page events (``fetch`` POST, JSON body).
MOBILE_EVENT_PATH = "/__ev"

plain_chapter_text = _epub_plain_chapter_text
search_excerpt = _epub_search_excerpt
load_epub_cache = _load_epub_cache
save_epub_cache = _save_epub_cache
overlay_signature = _reader_overlay_signature
load_native_toc = _load_reader_native_toc
map_native_toc = _map_native_toc_to_chapters
target_lang_to_google_code = _target_lang_to_google_code


class _Emitter:
    """Minimal stand-in for a Qt ``Signal`` on plain jobs: ``emit`` records the call."""

    __slots__ = ("calls", "_callback")

    def __init__(self, callback=None):
        self.calls = []
        self._callback = callback

    def emit(self, *args):
        self.calls.append(args)
        if self._callback is not None:
            self._callback(*args)

    @property
    def last(self):
        return self.calls[-1] if self.calls else None


class _PlainJob:
    """Shared ``_should_stop`` for the plain (non-Qt) jobs."""

    def __init__(self, should_stop=None):
        self._cancelled = False
        self._stop_callback = should_stop if callable(should_stop) else None

    def cancel(self) -> None:
        self._cancelled = True

    def _should_stop(self) -> bool:
        if self._cancelled:
            return True
        if self._stop_callback is not None:
            try:
                return bool(self._stop_callback())
            except Exception:
                return False
        return False


class ReaderLoadError(RuntimeError):
    """Raised by :func:`load_epub_chapters` with the desktop loader's error text."""


class _EpubLoadJob(EpubLoaderMixin, _PlainJob):
    def __init__(self, epub_path, show_special_files=True, config=None, should_stop=None):
        _PlainJob.__init__(self, should_stop)
        self._epub_path = epub_path
        self._show_special_files = bool(show_special_files)
        self._config = config or {}
        self.done = _Emitter()
        self.error = _Emitter()


def load_epub_chapters(epub_path: str, show_special_files: bool = True,
                       config: dict | None = None, should_stop=None, use_cache: bool = True):
    """Return ``(chapters, images, filenames)`` for *epub_path* like the desktop reader.

    The cache is consulted first (``_load_epub_cache``); a miss runs the desktop
    loader (``EpubLoaderMixin.run``), which writes the cache, and the result is read
    back from it exactly as ``EpubReaderDialog`` does after ``done``. ``chapters`` is
    ``[(title, html)]``, ``images`` maps member names to lazy descriptors and
    ``filenames`` is the parallel list of spine member names. Returns None when
    *should_stop* fired; raises :class:`ReaderLoadError` on a load failure.
    """
    config = config or {}
    if use_cache:
        cached = _load_epub_cache(epub_path, show_special_files=show_special_files, config=config)
        if cached:
            chapters, images, filenames = cached
            return chapters, images, list(filenames or [])
    job = _EpubLoadJob(epub_path, show_special_files, config, should_stop)
    job.run()
    if job.error.calls:
        raise ReaderLoadError(str(job.error.last[0]))
    if not job.done.calls:
        return None
    cached = _load_epub_cache(epub_path, show_special_files=show_special_files, config=config)
    if not cached:
        raise ReaderLoadError("The EPUB was parsed but its reader cache could not be written.")
    chapters, images, filenames = cached
    return chapters, images, list(filenames or [])


class OverlayMergeResult:
    """What ``_OverlayMergeThread`` hands its dialog (``done`` payload + flags)."""

    __slots__ = ("chapters", "images", "overlay_applied", "read_signature", "retry_required")

    def __init__(self, chapters, images, overlay_applied, read_signature, retry_required):
        self.chapters = chapters
        self.images = images
        self.overlay_applied = overlay_applied
        self.read_signature = read_signature
        self.retry_required = retry_required


class _OverlayMergeJob(OverlayMergeMixin, _PlainJob):
    def __init__(self, raw_chapters, images, filenames, overlay_map, extra_image_dirs,
                 config=None, previous_chapters=None, should_stop=None):
        _PlainJob.__init__(self, should_stop)
        self._raw_chapters = list(raw_chapters or [])
        self._images = dict(images or {})
        self._filenames = list(filenames or [])
        self._overlay = dict(overlay_map or {})
        self._extra_dirs = list(extra_image_dirs or [])
        self._config = dict(config or {})
        self._previous_chapters = list(previous_chapters or [])
        self._read_signature = None
        self._retry_required = False
        self._result_ready = False
        self.done = _Emitter()


def merge_overlay(raw_chapters, images, filenames, overlay_map, extra_image_dirs=(),
                  config: dict | None = None, previous_chapters=None, should_stop=None):
    """Merge translated response HTML over the raw chapters (``_OverlayMergeThread``).

    *overlay_map* is ``{source basename lower: {path, title, status}}`` (Book Details'
    ``_build_translated_overlay`` or ``reader_overlay.make_epub_overlay_provider``).
    Returns an :class:`OverlayMergeResult`, or None when *should_stop* fired.
    ``retry_required`` is True when a response was locked / empty / rewritten during the
    merge (the desktop reader retries on its next 3 s tick).
    """
    job = _OverlayMergeJob(raw_chapters, images, filenames, overlay_map, extra_image_dirs,
                           config, previous_chapters, should_stop)
    job.run()
    if not job.done.calls:
        return None
    chapters, merged_images, applied = job.done.last
    return OverlayMergeResult(chapters, merged_images, applied, job._read_signature, job._retry_required)


class _WorkspaceLoadJob(WorkspaceReaderLoaderMixin, _PlainJob):
    def __init__(self, manifest, should_stop=None):
        _PlainJob.__init__(self, should_stop)
        self._manifest = dict(manifest or {})
        self.done = _Emitter()
        self.error = _Emitter()


def load_workspace_chapters(manifest: dict, should_stop=None):
    """``(raw_chapters, translated_chapters, filenames)`` for a PDF/HTML workspace manifest.

    *manifest* comes from ``workspace_reader.build_workspace_reader_manifest``; raw
    chapters are placeholders until ``workspace_reader.ensure_pdf_raw_section`` renders
    a section. Raises :class:`ReaderLoadError` on failure, None when stopped.
    """
    job = _WorkspaceLoadJob(manifest, should_stop)
    job.run()
    if job.error.calls:
        raise ReaderLoadError(str(job.error.last[0]))
    if not job.done.calls:
        return None
    raw_chapters, translated_chapters, filenames = job.done.last
    return raw_chapters, translated_chapters, list(filenames)


class _ImagePreloadJob(ReaderImagePreloadMixin, _PlainJob):
    def __init__(self, preload_key, html_content, images, extra_image_dirs, epub_path, temp_dir,
                 should_stop=None):
        _PlainJob.__init__(self, should_stop)
        self._preload_key = str(preload_key or "")
        self._html_content = str(html_content or "")
        self._images = dict(images or {})
        self._extra_dirs = list(extra_image_dirs or [])
        self._epub_path = str(epub_path or "")
        self._temp_dir = str(temp_dir or "")
        self.done = _Emitter()


def preload_chapter_images(html_content: str, images=None, extra_image_dirs=(), epub_path: str = "",
                           temp_dir: str = "", preload_key: str = "", should_stop=None) -> dict:
    """Materialise a chapter's local images into *temp_dir* (``_ReaderImagePreloadThread``).

    Returns ``{identity: {path, sizeable, classified}}`` for ``ReaderDocument`` (pass it
    to :meth:`ReaderDocument.add_preloaded_resources`).
    """
    job = _ImagePreloadJob(preload_key, html_content, images, extra_image_dirs, epub_path,
                           temp_dir, should_stop)
    job.run()
    return dict(job.done.last[1]) if job.done.calls else {}


class _SearchJob(EpubSearchMixin, _PlainJob):
    def __init__(self, query, chapters, config=None, should_stop=None, on_batch=None):
        _PlainJob.__init__(self, should_stop)
        self._search_id = 0
        self._query = query or ""
        self._chapters = list(chapters or [])
        self._config = dict(config or {})
        self.rows = []
        self.finished = False

        def _batch(_search_id, _query, rows, done):
            self.rows.extend(rows)
            if done:
                self.finished = True
            if on_batch is not None:
                on_batch(list(rows), bool(done))

        self.results_batch_ready = _Emitter(_batch)
        self.results_ready = _Emitter()


def search_chapters(chapters, query: str, config: dict | None = None, should_stop=None,
                    on_batch=None) -> list:
    """Search ``[(title, html)]`` chapters like the reader's Search panel.

    Rows: ``{chapter_idx, local_occurrence, global_occurrence, match_count, text, title,
    excerpt}`` in chapter order (excerpt radius 34). *on_batch(rows, done)* receives the
    same 120-row batches the desktop list paints. Returns every row (empty when stopped).
    """
    job = _SearchJob(query, chapters, config, should_stop, on_batch)
    job.run()
    return list(job.rows)


class ReaderDocument(ReaderDocMixin):
    """Plain reader state for ``ReaderDocMixin`` (what ``EpubReaderDialog`` keeps on self).

    Mobile builds one per open book: ``process_html`` materialises images (through
    ``image_url_for``; the mobile Reader registers them with its localhost server) and
    ``wrap`` returns the full page (``mobile=True`` adds the touch shell).
    """

    def __init__(self, epub_path: str = "", *, images=None, extra_image_dirs=(), config=None,
                 theme=0, font_family: str = "Embedded CSS", font_size=14, line_spacing=1.8,
                 show_raw: bool = False, translated_overlay=None, raw_epub_alt_path: str = "",
                 translated_css_dirs=(), workspace_mode: bool = False, image_url_for=None,
                 chapter_filenames=()):
        self._epub_path = str(epub_path or "")
        self._images = dict(images or {})
        self._image_resource_signature = _reader_image_map_signature(self._images)
        self._extra_image_dirs = list(extra_image_dirs or [])
        self._config = config if config is not None else {}
        self._theme_override = None
        self._theme_index = 0
        self.set_theme(theme)
        self._font_family = font_family
        self._font_size = font_size
        self._line_spacing = line_spacing
        self._show_raw = bool(show_raw)
        self._translated_overlay = translated_overlay or None
        self._raw_epub_alt_path = raw_epub_alt_path or ""
        self._translated_css_dirs = list(translated_css_dirs or [])
        self._workspace_mode = bool(workspace_mode)
        self._image_url_for = image_url_for
        self._processed_html_cache = {}
        self._image_sizeable_cache = {}
        self._preloaded_image_resources = {}
        self._preloaded_chapter_keys = set()
        self._image_cache_generation = 0
        self._chapters = []
        self._chapter_display_numbers = []
        if chapter_filenames:
            self.set_chapter_filenames(chapter_filenames)

    # -- hooks -------------------------------------------------------------------------
    def _reader_file_url(self, path: str) -> str:
        if self._image_url_for is not None:
            return self._image_url_for(path)
        return Path(path).as_uri()

    def _get_theme(self):
        if self._theme_override is not None:
            return self._theme_override
        return ReaderDocMixin._get_theme(self)

    # -- state -------------------------------------------------------------------------
    def set_theme(self, theme) -> None:
        """Theme by index, name (``"Sepia"``) or a dict with the ``_READER_THEMES`` keys."""
        self._theme_override = None
        if isinstance(theme, dict):
            base = dict(_READER_THEMES[0])
            base.update(theme)
            self._theme_override = base
            return
        if isinstance(theme, str):
            for index, candidate in enumerate(_READER_THEMES):
                if candidate["name"].casefold() == theme.strip().casefold():
                    self._theme_index = index
                    return
            self._theme_index = 0
            return
        try:
            self._theme_index = int(theme or 0)
        except (TypeError, ValueError):
            self._theme_index = 0

    def set_images(self, images) -> None:
        self._set_reader_images(images)

    def add_preloaded_resources(self, resources: dict) -> None:
        for identity, info in (resources or {}).items():
            self._preloaded_image_resources[str(identity)] = dict(info)
            if isinstance(info, dict) and "sizeable" in info:
                self._image_sizeable_cache[str(identity)] = bool(info["sizeable"])

    def set_chapter_filenames(self, filenames) -> None:
        self._chapter_display_numbers = chapter_display_numbers(filenames, self._config)

    def close(self) -> None:
        self._close_epub_image_zip()

    # -- documents ---------------------------------------------------------------------
    def process_html(self, html_content: str) -> str:
        return self._process_html(html_content)

    def embedded_css(self) -> str:
        return self._get_embedded_css()

    def wrap(self, body_html: str, paginated: bool = False, spread_pages: int = 1, *,
             mobile: bool = False, event_url: str = MOBILE_EVENT_PATH, chapter=None,
             initial_page=None) -> str:
        page = self._wrap_html(body_html, paginated=paginated, spread_pages=spread_pages)
        if mobile:
            page = _mobilize_reader_html(page, paginated=paginated, event_url=event_url,
                                         chapter=chapter, initial_page=initial_page)
        return page

    def chapter_page(self, html_content: str, paginated: bool = False, spread_pages: int = 1,
                     **mobile_kwargs) -> str:
        """``wrap(process_html(html))`` — one chapter as a complete page."""
        return self.wrap(self.process_html(html_content), paginated, spread_pages, **mobile_kwargs)

    def all_chapters_body(self, chapters) -> str:
        """The "Scroll All" body (desktop ``_render_current`` LAYOUT_ALL branch)."""
        self._chapters = list(chapters or [])
        return self._all_chapters_html()


def process_chapter_html(html_content: str, *, images=None, extra_image_dirs=(), epub_path: str = "",
                         image_url_for=None, config=None) -> str:
    """One-shot ``_process_html`` (prefer a long-lived :class:`ReaderDocument`)."""
    doc = ReaderDocument(epub_path, images=images, extra_image_dirs=extra_image_dirs,
                         config=config, image_url_for=image_url_for)
    try:
        return doc.process_html(html_content)
    finally:
        doc.close()


def get_embedded_css(epub_path: str, *, config=None, show_raw: bool = False, translated_overlay=None,
                     raw_epub_alt_path: str = "", translated_css_dirs=(), workspace_mode: bool = False) -> str:
    """The reader's "Embedded CSS" stylesheet for *epub_path* (``_get_embedded_css``)."""
    doc = ReaderDocument(epub_path, config=config, show_raw=show_raw,
                         translated_overlay=translated_overlay, raw_epub_alt_path=raw_epub_alt_path,
                         translated_css_dirs=translated_css_dirs, workspace_mode=workspace_mode)
    return doc.embedded_css()


def resolve_attach_css_to_chapters(config=None) -> bool:
    """Effective "Attach CSS to chapters" setting (env ``ATTACH_CSS_TO_CHAPTERS`` wins)."""
    return ReaderDocument(config=config or {})._resolve_attach_css_to_chapters()


def reader_theme(theme=0) -> dict:
    """The theme dict for an index / name / dict (index out of range -> Dark)."""
    return ReaderDocument(theme=theme)._get_theme()


def wrap_reader_html(body_html: str, theme=0, *, font_family: str = "Embedded CSS", font_size=14,
                     line_spacing=1.8, paginated: bool = False, spread_pages: int = 1,
                     embedded_css=None, epub_path: str = "", config=None, show_raw: bool = False,
                     workspace_mode: bool = False, mobile: bool = False,
                     event_url: str = MOBILE_EVENT_PATH, chapter=None, initial_page=None) -> str:
    """Wrap processed chapter HTML in the reader's full page (``EpubReaderDialog._wrap_html``).

    *embedded_css*: None computes the "Embedded CSS" stylesheet from *epub_path* (only
    when *font_family* is ``"Embedded CSS"``, like desktop); a string is used as-is.
    ``mobile=True`` adds the touch shell (:func:`_mobilize_reader_html`); the desktop
    output is unchanged when it is False.
    """
    doc = ReaderDocument(epub_path, config=config, theme=theme, font_family=font_family,
                         font_size=font_size, line_spacing=line_spacing, show_raw=show_raw,
                         workspace_mode=workspace_mode)
    if embedded_css is not None:
        doc._get_embedded_css = lambda: embedded_css  # noqa: E731 - instance override
    return doc.wrap(body_html, paginated=paginated, spread_pages=spread_pages, mobile=mobile,
                    event_url=event_url, chapter=chapter, initial_page=initial_page)


def chapter_display_numbers(filenames, config=None) -> list:
    """Non-resetting display numbers for the reader's chapter list (``_finalize_post_load``)."""
    return _chapter_display_numbers(
        [os.path.basename(f or "").lower() for f in (filenames or [])], config)


def google_translate_url(text: str, target_language: str = "English") -> str:
    """translate.google.com URL for *text* (reader selection menu, Raw mode); "" when empty."""
    return _google_translate_url(text, _target_lang_to_google_code(target_language))


define_url = _define_url


def reader_image_temp_dir(epub_path: str) -> str:
    """Per-source image cache folder (``_ensure_reader_image_temp_dir``)."""
    return ReaderDocument(epub_path)._ensure_reader_image_temp_dir()


# ---------------------------------------------------------------------------
# Mobile page shell (new in U5; desktop never passes mobile=True)
# ---------------------------------------------------------------------------

_MOBILE_VIEWPORT = (
    '<meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1, '
    'maximum-scale=1, user-scalable=no, viewport-fit=cover">'
)

_MOBILE_COMMON_CSS = (
    "html { -webkit-tap-highlight-color: transparent; -webkit-touch-callout: none; }"
    "img, svg { -webkit-column-break-inside: avoid; }"
)

# Paged layouts: the desktop rule pads html AND body (10px / 26px each), so only <body> carries the
# safe-area insets here (added once, never on both), and #columns gets the height that is really left
# between them: a stylesheet !important wins over the inline ``innerHeight - 36`` of the desktop
# ``_setupColumns``. With env() = 0 this is the desktop geometry (columns from 20px to innerHeight - 16px);
# the image limits keep the desktop offsets from that column height (100vh first, then 100dvh where
# the WebView has dynamic viewport units). A full-page illustration box that also holds a paragraph pulled
# in before the picture (process_html's lead-in: a <p> that is not the box's last child - the picture's
# container always is) is laid out as plain blocks: the desktop box is one monolithic, overflow-hidden flex
# column, and on a phone-width column paragraph + picture are taller than the column, so its end (the
# picture, the paragraph's last lines) was cut off. As blocks the paragraph flows across columns like text
# and the picture (break-inside avoid, at most a column high) follows whole. (:has() cannot nest.)
_MOBILE_PAGED_CSS = (
    "html { height: 100%; height: 100dvh; padding: 0; touch-action: pan-y; }"
    "body { height: 100%; height: 100dvh; "
    "padding-top: calc(20px + env(safe-area-inset-top, 0px)); "
    "padding-bottom: calc(16px + env(safe-area-inset-bottom, 0px)); touch-action: pan-y; }"
    "#columns { "
    "height: calc(100vh - 36px - env(safe-area-inset-top, 0px) - env(safe-area-inset-bottom, 0px)) !important; "
    "height: calc(100dvh - 36px - env(safe-area-inset-top, 0px) - env(safe-area-inset-bottom, 0px)) !important; }"
    "#content { padding: 0 max(18px, env(safe-area-inset-right)) 0 "
    "max(18px, env(safe-area-inset-left)); }"
    "#content > :first-child, #content > div:first-child > :first-child "
    "{ -webkit-column-break-before: avoid !important; }"
    ".full-page-img { -webkit-column-break-before: always; -webkit-column-break-inside: avoid; }"
    ".full-page-img-first, #content > .full-page-img:first-child "
    "{ -webkit-column-break-before: avoid !important; }"
    ".full-page-img + .full-page-img { -webkit-column-break-before: always; }"
    "img, svg { "
    "max-height: calc(100vh - 60px - env(safe-area-inset-top, 0px) - env(safe-area-inset-bottom, 0px)); "
    "max-height: calc(100dvh - 60px - env(safe-area-inset-top, 0px) - env(safe-area-inset-bottom, 0px)); }"
    ".full-page-img { "
    "min-height: calc(100vh - 40px - env(safe-area-inset-top, 0px) - env(safe-area-inset-bottom, 0px)); "
    "min-height: calc(100dvh - 40px - env(safe-area-inset-top, 0px) - env(safe-area-inset-bottom, 0px)); }"
    ".full-page-img img { "
    "max-height: calc(100vh - 100px - env(safe-area-inset-top, 0px) - env(safe-area-inset-bottom, 0px)); "
    "max-height: calc(100dvh - 100px - env(safe-area-inset-top, 0px) - env(safe-area-inset-bottom, 0px)); }"
    ".full-page-img:has(> p ~ *) { display: block; overflow: visible; min-height: 0; "
    "break-inside: auto; -webkit-column-break-inside: auto; }"
)

_MOBILE_SCROLL_CSS = (
    "body { padding: max(10px, env(safe-area-inset-top)) max(16px, env(safe-area-inset-right)) "
    "max(28px, env(safe-area-inset-bottom)) max(16px, env(safe-area-inset-left)) !important; }"
)

# The bridge talks to Python two ways: console.log('GLRDR:' + json) for flet-webview's
# console event, and a same-origin fetch POST (the in-app server's /__ev) as the fallback.
# Every event carries a sequence number so a receiver that gets both can de-duplicate;
# GLRDR.setTransport('console'|'fetch'|'both') narrows it once Python knows what works.
_MOBILE_BRIDGE_JS = r"""
(function () {
  if (window.GLRDR) { return; }
  var PREFIX = %(prefix)s, EVENT_URL = %(event_url)s, CHAPTER = %(chapter)s;
  var INITIAL_PAGE = %(initial_page)s;
  var transport = 'both', seq = 0, page = 0, count = 1, ready = false;
  function cols() { return document.getElementById('columns'); }
  function post(evt) {
    evt = evt || {};
    evt.seq = ++seq;
    if (CHAPTER !== null && evt.chapter === undefined) { evt.chapter = CHAPTER; }
    var text = JSON.stringify(evt);
    if (transport !== 'fetch') { try { console.log(PREFIX + text); } catch (e) {} }
    if (transport !== 'console' && window.fetch) {
      try {
        fetch(EVENT_URL, {method: 'POST', credentials: 'same-origin', keepalive: true,
                          headers: {'Content-Type': 'application/json'}, body: text})
          .catch(function () {});
      } catch (e) {}
    }
  }
  function measure() {
    var c = cols();
    if (!c) { count = 1; return count; }
    if (typeof _setupColumns === 'function') { _setupColumns(); }
    count = (typeof _pageCountFor === 'function') ? _pageCountFor(c) : 1;
    return count;
  }
  function goTo(target, reason) {
    var c = cols();
    if (!c) { return; }
    measure();
    var p = Math.max(0, Math.min(parseInt(target, 10) || 0, count - 1));
    var gap = (typeof _PAGE_GAP !== 'undefined') ? _PAGE_GAP : 0;
    var w = (typeof _pageWidthFor === 'function') ? _pageWidthFor(c)
      : Math.max(1, Math.floor(c.clientWidth || window.innerWidth || 1));
    var span = Math.max(1, w + gap);
    _CURRENT_PAGE = p;
    c.style.transition = 'none';
    c.style.transform = 'none';
    c.scrollLeft = Math.round(p * span);
    page = p;
    post({type: 'page', page: page, count: count, reason: reason || 'goto'});
  }
  function next() {
    if (!cols()) { post({type: 'tap', zone: 'right'}); return; }
    if (page < count - 1) { goTo(page + 1, 'next'); }
    else { post({type: 'edge', edge: 'end', page: page, count: count}); }
  }
  function prev() {
    if (!cols()) { post({type: 'tap', zone: 'left'}); return; }
    if (page > 0) { goTo(page - 1, 'prev'); }
    else { post({type: 'edge', edge: 'start', page: page, count: count}); }
  }
  function goToFraction(fraction) {
    if (cols()) { measure(); goTo(Math.round((count - 1) * (+fraction || 0)), 'fraction'); return; }
    var max = Math.max(0, document.documentElement.scrollHeight - window.innerHeight);
    window.scrollTo(0, Math.round(max * (+fraction || 0)));
  }
  function selectionText() {
    try { return String(window.getSelection ? window.getSelection() : '').trim(); }
    catch (e) { return ''; }
  }
  var touch = null, swiped = false, pinch = null;
  function dist(t) {
    var dx = t[0].clientX - t[1].clientX, dy = t[0].clientY - t[1].clientY;
    return Math.sqrt(dx * dx + dy * dy);
  }
  document.addEventListener('touchstart', function (e) {
    swiped = false;
    if (e.touches.length === 2) { pinch = {start: dist(e.touches), scale: 1}; touch = null; return; }
    if (e.touches.length === 1) {
      touch = {x: e.touches[0].clientX, y: e.touches[0].clientY, t: Date.now()};
    }
  }, {passive: true});
  document.addEventListener('touchmove', function (e) {
    if (pinch && e.touches.length === 2 && pinch.start > 0) {
      pinch.scale = dist(e.touches) / pinch.start;
    }
  }, {passive: true});
  document.addEventListener('touchend', function (e) {
    if (pinch) {
      if (e.touches.length === 0) {
        if (Math.abs(pinch.scale - 1) > 0.08) { post({type: 'scale', scale: pinch.scale}); }
        pinch = null;
      }
      return;
    }
    if (!touch || !e.changedTouches.length) { return; }
    var dx = e.changedTouches[0].clientX - touch.x;
    var dy = e.changedTouches[0].clientY - touch.y;
    touch = null;
    if (Math.abs(dx) > 40 && Math.abs(dx) > Math.abs(dy) * 1.5 && !selectionText()) {
      swiped = true;
      if (dx < 0) { next(); } else { prev(); }
    }
  }, {passive: true});
  document.addEventListener('click', function (e) {
    if (swiped) { swiped = false; return; }
    var node = e.target;
    while (node && node !== document.body) {
      if (node.tagName === 'A' && node.getAttribute('href')) {
        post({type: 'link', href: node.getAttribute('href')});
        e.preventDefault();
        return;
      }
      node = node.parentNode;
    }
    if (selectionText()) { return; }
    var w = window.innerWidth || document.documentElement.clientWidth || 1;
    var x = e.clientX / w;
    if (x < 1 / 3) { prev(); }
    else if (x > 2 / 3) { next(); }
    else { post({type: 'tap', zone: 'center'}); }
  });
  var selTimer = null;
  document.addEventListener('selectionchange', function () {
    if (selTimer) { clearTimeout(selTimer); }
    selTimer = setTimeout(function () {
      var text = selectionText();
      if (text) { post({type: 'selection', text: text.slice(0, 2000)}); }
    }, 350);
  });
  var scrollTimer = null;
  window.addEventListener('scroll', function () {
    if (cols()) { return; }
    if (scrollTimer) { return; }
    scrollTimer = setTimeout(function () {
      scrollTimer = null;
      var max = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
      post({type: 'scroll', fraction: Math.min(1, Math.max(0, window.scrollY / max))});
    }, 250);
  }, {passive: true});
  window.addEventListener('resize', function () {
    if (!ready || !cols()) { return; }
    var old = count, keep = page;
    measure();
    goTo(old > 1 ? Math.round(keep * (count - 1) / Math.max(1, old - 1)) : keep, 'resize');
  });
  function start() {
    if (ready) { return; }
    ready = true;
    var c = cols();
    var hash = String(location.hash || ''), m;
    var first = INITIAL_PAGE;
    if ((m = /[#&]p=(\d+)/.exec(hash))) { first = parseInt(m[1], 10); }
    if (c) {
      measure();
      if ((m = /[#&]f=([0-9.]+)/.exec(hash))) { first = Math.round((count - 1) * parseFloat(m[1])); }
      goTo(first || 0, 'initial');
      c.style.opacity = '1';
    } else if ((m = /[#&]f=([0-9.]+)/.exec(hash))) {
      goToFraction(parseFloat(m[1]));
    }
    post({type: 'ready', page: page, count: count, paginated: !!c});
  }
  window.GLRDR = {
    goTo: function (p) { goTo(p, 'api'); }, next: next, prev: prev,
    goToFraction: goToFraction, measure: measure,
    page: function () { return page; }, count: function () { return count; },
    setTransport: function (mode) { transport = mode || 'both'; }, post: post
  };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { setTimeout(start, 0); });
  } else { setTimeout(start, 0); }
  window.addEventListener('load', function () { if (ready && cols()) { goTo(page, 'load'); } });
})();
"""


def _mobilize_reader_html(page: str, *, paginated: bool, event_url: str = MOBILE_EVENT_PATH,
                          chapter=None, initial_page=None) -> str:
    """Add the touch shell to a ``_wrap_html`` page (viewport, safe areas, paging bridge)."""
    css = _MOBILE_COMMON_CSS + (_MOBILE_PAGED_CSS if paginated else _MOBILE_SCROLL_CSS)
    script = _MOBILE_BRIDGE_JS % {
        "prefix": json.dumps(MOBILE_EVENT_PREFIX),
        "event_url": json.dumps(str(event_url or MOBILE_EVENT_PATH)),
        "chapter": json.dumps(chapter),
        "initial_page": json.dumps(int(initial_page or 0)),
    }
    head_at = page.find("<head>")
    if head_at >= 0:
        page = page[:head_at + 6] + _MOBILE_VIEWPORT + page[head_at + 6:]
    style_end = page.find("</style>")
    if style_end >= 0:
        page = page[:style_end] + css + page[style_end:]
    head_end = page.find("</head>")
    if head_end >= 0:
        page = page[:head_end] + "<script>" + script + "</script>" + page[head_end:]
    return page


# ---------------------------------------------------------------------------
# Bilingual chapters and native blocks (new in U5)
# ---------------------------------------------------------------------------

_BILINGUAL_BLOCK_TAGS = (
    "p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre", "table",
    "figure", "dt", "dd", "figcaption", "caption", "hr",
)

_BILINGUAL_CSS = (
    "<style>"
    ".glr-bi-pair { margin: 0 0 1.1em 0; }"
    ".glr-bi-src { opacity: .72; font-size: .92em; margin: 0 0 .25em 0; }"
    ".glr-bi-src > *, .glr-bi-tr > * { margin-top: 0; margin-bottom: 0; }"
    ".glr-bi-section-label { opacity: .6; font-size: .8em; letter-spacing: .08em; "
    "text-transform: uppercase; margin: 1.2em 0 .6em 0; }"
    "hr.glr-bi-sep { border: none; border-top: 1px solid currentColor; opacity: .25; margin: 2em 0; }"
    "</style>"
)


def _bilingual_soup_body(html_text: str):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(unescape_valid_html_tag_entities(str(html_text or "")), "html.parser")
    return soup.body or soup


def _bilingual_blocks(root) -> list:
    """Leaf block elements of *root* in document order (images outside blocks included)."""
    blocks = []
    for node in root.find_all(True):
        name = (node.name or "").lower()
        if name in _BILINGUAL_BLOCK_TAGS:
            if any((getattr(parent, "name", "") or "").lower() in _BILINGUAL_BLOCK_TAGS
                   for parent in node.parents):
                continue
            if name != "hr" and not node.get_text(strip=True) and not node.find(["img", "svg", "image"]):
                continue
            blocks.append(node)
        elif name in ("img", "svg"):
            if not any((getattr(parent, "name", "") or "").lower() in _BILINGUAL_BLOCK_TAGS
                       for parent in node.parents):
                blocks.append(node)
        elif name == "div" and not node.find(list(_BILINGUAL_BLOCK_TAGS) + ["div", "img", "svg"]):
            if node.get_text(strip=True) and not any(
                    (getattr(parent, "name", "") or "").lower() in _BILINGUAL_BLOCK_TAGS
                    for parent in node.parents):
                blocks.append(node)
    return blocks


def bilingual_alignment(raw_html: str, translated_html: str, threshold: float = 0.15):
    """Return ``("blocks", [(raw_block_html, translated_block_html), ...])`` or ``("sections", [])``.

    Blocks are paired by position when the block counts differ by at most *threshold*
    (relative to the larger count); otherwise the chapter falls back to whole-section
    order (original, then translated).
    """
    raw_blocks = _bilingual_blocks(_bilingual_soup_body(raw_html))
    tr_blocks = _bilingual_blocks(_bilingual_soup_body(translated_html))
    n_raw, n_tr = len(raw_blocks), len(tr_blocks)
    largest = max(n_raw, n_tr)
    if not n_raw or not n_tr or abs(n_raw - n_tr) / float(largest) > float(threshold):
        return "sections", []
    pairs = []
    for index in range(largest):
        raw_part = str(raw_blocks[index]) if index < n_raw else ""
        tr_part = str(tr_blocks[index]) if index < n_tr else ""
        pairs.append((raw_part, tr_part))
    return "blocks", pairs


def build_bilingual_chapter(raw_html: str, translated_html: str, threshold: float = 0.15,
                            original_label: str = "Original", translated_label: str = "Translated") -> str:
    """Interleave a chapter's original and translated text (Reader "Bilingual" mode).

    Block-level alignment (each original block followed by its translation) when the
    two versions have block counts within *threshold*; whole-section order otherwise.
    Returns body HTML for :meth:`ReaderDocument.process_html` / ``wrap``.
    """
    mode, pairs = bilingual_alignment(raw_html, translated_html, threshold)
    if mode == "blocks":
        parts = [_BILINGUAL_CSS]
        for raw_part, tr_part in pairs:
            parts.append('<div class="glr-bi-pair">')
            if raw_part:
                parts.append(f'<div class="glr-bi-src">{raw_part}</div>')
            if tr_part:
                parts.append(f'<div class="glr-bi-tr">{tr_part}</div>')
            parts.append("</div>")
        return "".join(parts)
    import html as _html
    raw_body = "".join(str(child) for child in _bilingual_soup_body(raw_html).contents)
    tr_body = "".join(str(child) for child in _bilingual_soup_body(translated_html).contents)
    return (
        _BILINGUAL_CSS
        + f'<div class="glr-bi-section-label">{_html.escape(original_label)}</div>'
        + f'<section class="glr-bi-original">{raw_body}</section>'
        + '<hr class="glr-bi-sep"/>'
        + f'<div class="glr-bi-section-label">{_html.escape(translated_label)}</div>'
        + f'<section class="glr-bi-translated">{tr_body}</section>'
    )


def html_to_blocks(html_text: str) -> list:
    """Flatten chapter HTML into simple blocks for a native (non-WebView) reader.

    Each block is ``{"kind": "heading"|"paragraph"|"image"|"rule"|"pre"|"quote"|"item",
    "text", "level", "src", "alt"}`` in document order; text is whitespace-collapsed.
    """
    root = _bilingual_soup_body(html_text)
    for junk in root.find_all(["script", "style", "noscript"]):
        junk.decompose()
    out = []

    def _text(node) -> str:
        return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()

    for node in _bilingual_blocks(root):
        name = (node.name or "").lower()
        if name in ("img", "svg"):
            out.append({"kind": "image", "src": str(node.get("src") or ""),
                        "alt": str(node.get("alt") or ""), "text": "", "level": 0})
            continue
        for image in node.find_all("img"):
            out.append({"kind": "image", "src": str(image.get("src") or ""),
                        "alt": str(image.get("alt") or ""), "text": "", "level": 0})
        if name == "hr":
            out.append({"kind": "rule", "text": "", "level": 0, "src": "", "alt": ""})
            continue
        text = node.get_text() if name == "pre" else _text(node)
        if not text:
            continue
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            kind, level = "heading", int(name[1])
        elif name == "pre":
            kind, level = "pre", 0
        elif name == "blockquote":
            kind, level = "quote", 0
        elif name in ("li", "dt", "dd"):
            kind, level = "item", 0
        else:
            kind, level = "paragraph", 0
        out.append({"kind": kind, "text": text, "level": level, "src": "", "alt": ""})
    return out


__all__ = [
    "LAYOUT_ALL",
    "LAYOUT_DOUBLE",
    "LAYOUT_SCROLL",
    "LAYOUT_SINGLE",
    "MOBILE_EVENT_PATH",
    "MOBILE_EVENT_PREFIX",
    "READER_LAYOUTS",
    "READER_THEMES",
    "READER_THEME_NAMES",
    "EpubCacheLoaderMixin",
    "EpubLoaderMixin",
    "EpubSearchMixin",
    "OverlayMergeMixin",
    "OverlayMergeResult",
    "ReaderDocMixin",
    "ReaderDocument",
    "ReaderImagePreloadMixin",
    "ReaderLoadError",
    "WorkspaceReaderLoaderMixin",
    "_LAZY_EPUB_IMAGE_TAG",
    "_READER_GT_LANG_CODES",
    "_READER_IMAGE_EXTS",
    "_READER_THEMES",
    "_discover_epub_image_members",
    "_epub_cache_dir",
    "_epub_plain_chapter_text",
    "_epub_search_excerpt",
    "_find_reader_sidecar",
    "_lazy_epub_image",
    "_lazy_epub_image_member",
    "_load_epub_cache",
    "_load_reader_native_toc",
    "_map_native_toc_to_chapters",
    "_native_toc_target_key",
    "_parse_native_toc_ncx",
    "_parse_native_toc_txt",
    "_read_epub_member_from_zip",
    "_reader_file_image_resource",
    "_reader_image_cache_path",
    "_reader_image_candidates",
    "_reader_image_is_sizeable",
    "_reader_image_map_signature",
    "_reader_image_resource",
    "_reader_overlay_signature",
    "_save_epub_cache",
    "_target_lang_to_google_code",
    "_url_scheme",
    "_workspace_reader_placeholder",
    "_chapter_display_numbers",
    "_define_url",
    "_google_translate_url",
    "_write_reader_image_cache",
    "bilingual_alignment",
    "build_bilingual_chapter",
    "chapter_display_numbers",
    "define_url",
    "get_embedded_css",
    "google_translate_url",
    "html_to_blocks",
    "load_epub_cache",
    "load_epub_chapters",
    "load_native_toc",
    "load_workspace_chapters",
    "map_native_toc",
    "merge_overlay",
    "overlay_signature",
    "plain_chapter_text",
    "preload_chapter_images",
    "process_chapter_html",
    "reader_image_temp_dir",
    "reader_theme",
    "resolve_attach_css_to_chapters",
    "save_epub_cache",
    "search_chapters",
    "search_excerpt",
    "set_epub_cache_dir",
    "target_lang_to_google_code",
    "wrap_reader_html",
]
