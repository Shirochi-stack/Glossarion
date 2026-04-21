# epub_library.py
"""
EPUB Library & Reader for Glossarion Desktop (Windows / macOS).

Provides:
  - scan_for_epubs(): recursively find .epub files across output dirs
  - EpubLibraryDialog: grid-card browser with cover thumbnails
  - EpubReaderDialog: simple in-app EPUB reader with TOC navigation
"""

import os
import re
import sys
import hashlib
import logging
import shutil
import subprocess
import tempfile
import platform
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QScrollArea, QWidget, QLineEdit, QFrame, QSplitter, QTextBrowser,
    QListWidget, QListWidgetItem, QMessageBox, QSizePolicy, QToolButton,
    QApplication, QMenu, QComboBox, QStackedWidget
)
from PySide6.QtCore import Qt, QSize, QRect, Signal, Slot, QThread, QTimer, QSizeF, QUrl
from PySide6.QtGui import QPixmap, QFont, QFontMetrics, QIcon, QImage, QCursor, QShortcut, QKeySequence, QTransform

# Use QWebEngineView for full CSS support (images, block layout, etc.)
try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWebEngineCore import QWebEnginePage
    _HAS_WEBENGINE = True
except ImportError:
    _HAS_WEBENGINE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & Utilities
# ---------------------------------------------------------------------------

def get_library_dir() -> str:
    """Return the dedicated Glossarion Library folder (root).

    The library is organized into two subfolders:
      * ``Raw/``        — curated raw source EPUBs the user has imported.
      * ``Translated/`` — curated compiled EPUBs (finished translations).
    """
    docs = Path.home() / "Documents" / "Glossarion" / "Library"
    try:
        docs.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return str(docs)


def get_library_raw_dir() -> str:
    """Return ``Library/Raw`` — home for imported raw source EPUBs."""
    raw = Path(get_library_dir()) / "Raw"
    try:
        raw.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return str(raw)


def get_library_translated_dir() -> str:
    """Return ``Library/Translated`` — home for curated compiled EPUBs."""
    trans = Path(get_library_dir()) / "Translated"
    try:
        trans.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return str(trans)


def _migrate_legacy_library_layout() -> None:
    """One-time migration: move any EPUBs sitting in ``Library/`` root into
    ``Library/Translated/``.

    Older builds stored organized EPUBs directly under ``Library/``. The
    new layout reserves that folder for the ``Raw/`` and ``Translated/``
    subfolders plus the origins registry. Running this on every library
    scan is idempotent — once there are no root-level EPUBs left, it's a
    no-op. Each moved file is recorded in ``origins['translated']`` so the
    user can reverse it via the Undo Move button.
    """
    lib = get_library_dir()
    trans = get_library_translated_dir()
    try:
        entries = list(os.scandir(lib))
    except (PermissionError, OSError):
        return
    origins = _load_origins()
    trans_origins = dict(origins.get("translated", {}) or {})
    moved_any = False
    for entry in entries:
        if not entry.is_file(follow_symlinks=False):
            continue
        if not entry.name.lower().endswith(".epub"):
            continue  # keep origins file + other stray text files intact
        src = entry.path
        dst = os.path.join(trans, entry.name)
        if os.path.normcase(os.path.normpath(os.path.abspath(src))) == \
                os.path.normcase(os.path.normpath(os.path.abspath(dst))):
            continue
        if os.path.isfile(dst):
            # Don't clobber an existing Translated copy.
            logger.debug("Legacy migration skipped (dst exists): %s", dst)
            continue
        try:
            shutil.move(src, dst)
            trans_origins[entry.name] = os.path.abspath(src)
            moved_any = True
            logger.info("Legacy library migration: %s -> %s", src, dst)
        except OSError as exc:
            logger.debug("Legacy migration move failed for %s: %s", src, exc)
    if moved_any:
        origins["translated"] = trans_origins
        _save_origins(origins)


def get_library_raw_inputs_file() -> str:
    """Return ``Library/Raw/library_raw_inputs.txt`` — one path per line.

    Tracks every raw input file that has been run through the translator.
    The list is append-only; duplicates are collapsed on read.
    """
    return os.path.join(get_library_raw_dir(), "library_raw_inputs.txt")


def load_library_raw_inputs() -> list[str]:
    """Return the deduplicated list of raw input paths recorded so far."""
    path = get_library_raw_inputs_file()
    if not os.path.isfile(path):
        return []
    seen: set[str] = set()
    out: list[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if not p:
                    continue
                key = os.path.normcase(os.path.normpath(os.path.abspath(p)))
                if key in seen:
                    continue
                seen.add(key)
                out.append(p)
    except OSError:
        return []
    return out


def record_library_raw_input(path: str) -> None:
    """Append *path* to the raw-inputs registry (no-op on failure / dup)."""
    if not path:
        return
    try:
        abs_path = os.path.abspath(path)
    except (TypeError, ValueError):
        return
    existing = {
        os.path.normcase(os.path.normpath(os.path.abspath(p)))
        for p in load_library_raw_inputs()
    }
    if os.path.normcase(os.path.normpath(abs_path)) in existing:
        return
    reg = get_library_raw_inputs_file()
    try:
        with open(reg, "a", encoding="utf-8") as f:
            f.write(abs_path + "\n")
    except OSError:
        logger.debug("Could not append to %s", reg)


def _origins_file() -> str:
    """Path to the library origins mapping file."""
    return os.path.join(get_library_dir(), "library_origins.txt")


# Structured origins format (v3):
#   { "version": 3,
#     "raw":        { "<basename in Library/Raw>":        "<absolute original path>" },
#     "translated": { "<basename in Library/Translated>": "<absolute original path>" },
#     "pairs":      { "<basename in Library/Translated>": "<basename in Library/Raw>" } }
#
# The ``pairs`` bucket is a direct translated↔raw link for books whose
# compiled and source EPUBs were organized together. It's consulted by
# :func:`_find_raw_source_for_library_epub` *first* because filename-stem
# matching fails when the raw is named in one language and the compiled
# in another (common for translations) — and the output-folder fallback
# breaks as soon as that folder is deleted or its ``source_epub.txt``
# sidecar drifts out of sync.
#
# Legacy v1 format was a flat ``{basename: original_path}`` dict and is
# promoted into the ``translated`` bucket on read so the user's existing
# undo history is preserved.


def _load_origins() -> dict:
    """Load the structured origins mapping (auto-upgrades the legacy format)."""
    import json
    try:
        with open(_origins_file(), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"version": 3, "raw": {}, "translated": {}, "pairs": {}}
    if not isinstance(data, dict):
        return {"version": 3, "raw": {}, "translated": {}, "pairs": {}}
    # Legacy flat mapping → promote to ``translated``.
    if "version" not in data and "raw" not in data and "translated" not in data:
        return {"version": 3, "raw": {}, "translated": dict(data), "pairs": {}}
    data.setdefault("version", 3)
    data.setdefault("raw", {})
    data.setdefault("translated", {})
    data.setdefault("pairs", {})
    if not isinstance(data["raw"], dict):
        data["raw"] = {}
    if not isinstance(data["translated"], dict):
        data["translated"] = {}
    if not isinstance(data["pairs"], dict):
        data["pairs"] = {}
    return data


def _save_origins(origins: dict):
    """Persist the structured origins mapping."""
    import json
    try:
        with open(_origins_file(), "w", encoding="utf-8") as f:
            json.dump(origins, f, indent=2, ensure_ascii=False)
    except OSError:
        pass


def _cover_cache_dir() -> str:
    d = os.path.join(tempfile.gettempdir(), "Glossarion_CoverCache")
    os.makedirs(d, exist_ok=True)
    return d


def _epub_cache_dir() -> str:
    d = os.path.join(tempfile.gettempdir(), "Glossarion_EpubCache")
    os.makedirs(d, exist_ok=True)
    return d


# Bump this salt whenever the loader's output schema changes so old
# pickled caches miss cleanly instead of being served back forever.
#   v2  — spine-first chapter resolution + text/html fallback.
#   v3  — authoritative items (cover, nav, TOC pages) no longer dropped
#          by the text-length filter.
#   v4  — reader respects the Show-special-files toggle (cache key now
#          embeds its state so on/off entries don't collide).
_EPUB_CACHE_SCHEMA = "v4"


def _epub_cache_key(epub_path: str, show_special_files: bool = True) -> str:
    """Generate a cache key from path + file modification time + schema.

    *show_special_files* is baked into the key so switching the
    Show-special-files toggle forces a fresh parse instead of serving a
    cache produced under the opposite toggle state.
    """
    try:
        mtime = os.path.getmtime(epub_path)
    except OSError:
        mtime = 0
    salt = f"{_EPUB_CACHE_SCHEMA}|special={int(bool(show_special_files))}"
    raw = f"{epub_path}|{mtime}|{salt}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:16]


def _load_epub_cache(epub_path: str, show_special_files: bool = True):
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
        key = _epub_cache_key(epub_path, show_special_files)
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
                     show_special_files: bool = True):
    """Save parsed EPUB data to disk cache.

    *show_special_files* is forwarded to :func:`_epub_cache_key` so the
    on / off variants of the cache are stored under distinct keys.
    """
    import pickle
    try:
        key = _epub_cache_key(epub_path, show_special_files)
        cache_file = os.path.join(_epub_cache_dir(), f"{key}.pkl")
        with open(cache_file, "wb") as f:
            pickle.dump({
                "chapters": chapters,
                "images": images,
                "filenames": list(filenames or []),
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass


def _find_halgakos_icon() -> str | None:
    """Locate the Halgakos.ico fallback icon."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "Halgakos.ico"),
        os.path.join(os.path.dirname(__file__), "Halgakos.png"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "Halgakos.png"),
    ]
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        candidates.insert(0, os.path.join(exe_dir, "Halgakos.ico"))
        candidates.insert(1, os.path.join(exe_dir, "Halgakos.png"))
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _extract_cover(epub_path: str) -> str | None:
    cache_dir = _cover_cache_dir()
    name_hash = hashlib.md5(epub_path.encode("utf-8")).hexdigest()[:12]
    cached = os.path.join(cache_dir, f"{name_hash}.jpg")
    if os.path.isfile(cached):
        # Guard against stale 0-byte caches left behind by crashed writes:
        # returning one of those to QPixmap produces a null pixmap and hides
        # the cover silently. Re-extract when the cache is clearly empty.
        try:
            if os.path.getsize(cached) > 0:
                return cached
            os.remove(cached)
        except OSError:
            pass

    # Fast path: read EPUB as a zip and extract cover via OPF metadata
    # This avoids the heavy ebooklib.read_epub() which fully parses the DOM
    try:
        import zipfile
        import posixpath
        from xml.etree import ElementTree as ET

        with zipfile.ZipFile(epub_path, "r") as zf:
            names = zf.namelist()
            names_set = set(names)
            cover_data = None

            # --- Step 1: Find and parse the OPF file ---
            opf_path = None
            # Check container.xml for the OPF path
            if "META-INF/container.xml" in names_set:
                try:
                    container_xml = zf.read("META-INF/container.xml").decode("utf-8", errors="replace")
                    container_tree = ET.fromstring(container_xml)
                    ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
                    rootfile = container_tree.find(".//c:rootfile", ns)
                    if rootfile is not None:
                        opf_path = rootfile.get("full-path")
                except Exception:
                    pass

            # Fallback: find any .opf file
            if not opf_path:
                for zname in names:
                    if zname.lower().endswith(".opf"):
                        opf_path = zname
                        break

            opf_dir = ""
            manifest_items = {}  # id -> (href, media_type)
            cover_meta_id = None

            if opf_path and opf_path in names_set:
                try:
                    opf_xml = zf.read(opf_path).decode("utf-8", errors="replace")
                    opf_tree = ET.fromstring(opf_xml)
                    opf_dir = posixpath.dirname(opf_path)

                    # Strip namespace for easier matching
                    opf_ns = {"opf": "http://www.idpf.org/2007/opf", "dc": "http://purl.org/dc/elements/1.1/"}

                    # Find cover image ID from <meta name="cover" content="..."/>
                    for meta_el in opf_tree.findall(".//{http://www.idpf.org/2007/opf}meta"):
                        if meta_el.get("name") == "cover":
                            cover_meta_id = meta_el.get("content")
                            break

                    # Build manifest lookup
                    for item_el in opf_tree.findall(".//{http://www.idpf.org/2007/opf}item"):
                        item_id = item_el.get("id", "")
                        item_href = item_el.get("href", "")
                        item_media = item_el.get("media-type", "")
                        item_props = item_el.get("properties", "")
                        full_href = posixpath.normpath(posixpath.join(opf_dir, item_href)) if item_href else ""
                        manifest_items[item_id] = (full_href, item_media, item_props)
                except Exception:
                    pass

            # --- Step 2: Try cover by OPF metadata ID ---
            if cover_meta_id and cover_meta_id in manifest_items:
                href, media_type, _ = manifest_items[cover_meta_id]
                if href in names_set and media_type.startswith("image/"):
                    cover_data = zf.read(href)

            # --- Step 3: Try cover by properties="cover-image" (EPUB3) ---
            if not cover_data:
                for item_id, (href, media_type, props) in manifest_items.items():
                    if "cover-image" in props and href in names_set:
                        cover_data = zf.read(href)
                        break

            # --- Step 4: Try images with "cover" in filename ---
            if not cover_data:
                img_exts = (".jpg", ".jpeg", ".png", ".gif", ".webp")
                for zname in names:
                    lower = zname.lower()
                    if any(lower.endswith(ext) for ext in img_exts) and "cover" in os.path.basename(lower):
                        cover_data = zf.read(zname)
                        break

            # --- Step 5: First <img> in first HTML chapter ---
            if not cover_data:
                try:
                    from bs4 import BeautifulSoup
                    html_exts = (".xhtml", ".html", ".htm")
                    for zname in sorted(names):
                        if any(zname.lower().endswith(ext) for ext in html_exts):
                            html = zf.read(zname).decode("utf-8", errors="replace")
                            soup = BeautifulSoup(html, "html.parser")
                            img_tag = soup.find("img")
                            if img_tag and img_tag.get("src"):
                                html_dir = posixpath.dirname(zname)
                                img_path = posixpath.normpath(posixpath.join(html_dir, img_tag["src"]))
                                if img_path in names_set:
                                    cover_data = zf.read(img_path)
                                    break
                except Exception:
                    pass

            # --- Step 6: First image file in the zip ---
            if not cover_data:
                img_exts = (".jpg", ".jpeg", ".png", ".gif", ".webp")
                for zname in names:
                    if any(zname.lower().endswith(ext) for ext in img_exts):
                        cover_data = zf.read(zname)
                        break

            if cover_data:
                with open(cached, "wb") as f:
                    f.write(cover_data)
                return cached
    except Exception as exc:
        logger.debug("Cover extraction (zipfile) failed for %s: %s\n%s", epub_path, exc, traceback.format_exc())

    # Last resort fallback: ebooklib (heavy, but handles edge cases)
    try:
        import ebooklib
        from ebooklib import epub as epub_mod

        book = epub_mod.read_epub(epub_path, options={"ignore_ncx": True})
        cover_data = None

        for meta in book.get_metadata("OPF", "cover"):
            if meta and meta[1]:
                cover_id = meta[1].get("content")
                if cover_id:
                    for item in book.get_items():
                        if item.get_id() == cover_id:
                            cover_data = item.get_content()
                            break
                break

        if not cover_data:
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_IMAGE and "cover" in item.get_name().lower():
                    cover_data = item.get_content()
                    break

        if not cover_data:
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_IMAGE:
                    cover_data = item.get_content()
                    break

        if cover_data:
            with open(cached, "wb") as f:
                f.write(cover_data)
            return cached
    except Exception as exc:
        logger.debug("Cover extraction (ebooklib) failed for %s: %s\n%s", epub_path, exc, traceback.format_exc())

    return None


def _open_folder_in_explorer(path: str):
    """Reveal *path* in the OS file manager.

    On Windows the subprocess call passes ``CREATE_NO_WINDOW`` so the
    spawned ``explorer.exe`` invocation doesn't flash a console window
    beside the cursor before the Explorer pane appears. The viewer-open
    helpers elsewhere in this module already use the same flag; this
    brings ``_open_folder_in_explorer`` into line so there are no
    unsuppressed Popen paths left.
    """
    # CREATE_NO_WINDOW lives on ``subprocess`` on Windows and is a
    # no-op elsewhere, so we resolve it with ``getattr`` for a
    # cross-platform single-expression form. 0x08000000 is the raw
    # value for older / frozen interpreters that don't expose the
    # attribute directly.
    _no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    try:
        folder = os.path.dirname(path) if os.path.isfile(path) else path
        if platform.system() == "Windows":
            if os.path.isfile(path):
                subprocess.Popen(
                    ["explorer", "/select,", os.path.normpath(path)],
                    creationflags=_no_window,
                )
            else:
                os.startfile(folder)
        elif platform.system() == "Darwin":
            subprocess.Popen(
                ["open", "-R", path] if os.path.isfile(path) else ["open", folder]
            )
        else:
            subprocess.Popen(["xdg-open", folder])
    except Exception as exc:
        logger.warning("Failed to open folder: %s\n%s", exc, traceback.format_exc())


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def _resolve_output_roots(config: dict | None = None) -> list[str]:
    """Return the directory the translator writes output folders into.

    When the OUTPUT_DIRECTORY override is configured (env or
    ``config['output_directory']``) it is used *strictly* — the default dir is
    NOT also scanned. This matches the user expectation that turning on an
    output override in Other Settings points the in-progress reader at that
    path alone. When no override is set, only the default directory (app dir
    on Windows, CWD elsewhere) is used.
    """
    config = config or {}
    override = os.environ.get("OUTPUT_DIRECTORY") or config.get("output_directory")
    if override:
        override_abs = os.path.abspath(override)
        if os.path.isdir(override_abs):
            return [override_abs]
        # Override is set but points at a missing directory — still treat it
        # as strict (return no roots) rather than silently falling back to
        # the default, which would violate user expectations.
        return []
    if platform.system() == "Windows":
        if getattr(sys, "frozen", False):
            default_dir = os.path.dirname(sys.executable)
        else:
            default_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        default_dir = os.getcwd()
    if os.path.isdir(default_dir):
        return [os.path.abspath(default_dir)]
    return []


def _is_gallery_filename(name: str) -> bool:
    """Return True if *name* refers to the auto-generated gallery page.

    Matches ``gallery.xhtml``, ``gallery.html``, ``Gallery.xhtml``,
    ``response_gallery.*`` and similar variants (case-insensitive,
    extension-agnostic). The gallery is injected by the translator's
    compile step, never a real source chapter, so it must never count
    toward the translation progress fraction nor render a status badge.
    """
    if not name:
        return False
    base = os.path.basename(str(name)).lower()
    if base.startswith("response_"):
        base = base[len("response_"):]
    stem = os.path.splitext(base)[0]
    return stem == "gallery"


def _read_progress_summary(progress_file: str, exclude_special: bool = False) -> dict | None:
    """Return a lightweight summary of translation_progress.json or None on failure.

    The summary counts chapter statuses so the library card can render a
    fraction/percentage without paying for the full OPF-aware match.

    When *exclude_special* is True, entries whose ``original_basename``
    (or ``output_file``) has no digit in its stem — ``cover.xhtml``,
    ``nav.xhtml``, … — are dropped from both the total and the status
    tallies. This matches the heuristic used by :func:`_count_epub_spine_items`
    and :func:`_count_translated_response_files` so the toggle actually
    shifts both numerator and denominator together (otherwise the
    filesystem / spine counts shrink but the progress-file count stays
    fixed and the dreaded 98 %↛1 00 % pair never changes).

    **File-existence verification**: entries whose ``status`` is
    ``"completed"`` are only counted as completed when the
    corresponding ``output_file`` still exists on disk (relative to
    the progress file's folder). The translator writes
    ``response_*.html`` files during translation and marks the
    progress entry ``completed`` — but if the user later manually
    deletes one of those files, the JSON still carries the stale
    ``completed`` status. Without this verification the card would
    keep reading e.g. 60/60 even though only 55 of the response
    files remain on disk. Any demoted entries are counted as
    ``in_progress`` instead so the fraction reflects reality.
    """
    import json as _json
    import re as _re
    try:
        with open(progress_file, "r", encoding="utf-8") as f:
            prog = _json.load(f)
    except (OSError, _json.JSONDecodeError):
        return None
    chapters = prog.get("chapters", {}) or {}
    total = 0
    completed = 0
    in_progress = 0
    failed = 0
    output_folder = os.path.dirname(progress_file) if progress_file else ""
    output_folder_ok = bool(output_folder) and os.path.isdir(output_folder)
    for key, ch in chapters.items():
        if not isinstance(ch, dict):
            continue
        name = (ch.get("original_basename")
                or ch.get("output_file")
                or str(key)
                or "")
        # Auto-generated gallery entries never count toward progress,
        # regardless of the translate-special-files toggle.
        if _is_gallery_filename(name):
            continue
        if exclude_special:
            # Prefer original_basename (source file) over output_file
            # (response_*.html) so a run started with translate_special=ON
            # and flipped OFF still filters correctly. Fall back to the
            # progress-file key as a last resort.
            low = name.lower()
            if low.startswith("response_"):
                low = low[len("response_"):]
            stem = os.path.splitext(os.path.basename(low))[0]
            # Only skip when we HAVE a filename to judge by — entries
            # with no recognizable name stay in the count rather than
            # silently disappearing.
            if stem and not _re.search(r"\d", stem):
                continue
        total += 1
        status = ch.get("status", "")
        # Phantom-completion check: the progress JSON may say
        # "completed" but the actual output file could have been
        # deleted by the user. Verify it's still on disk before
        # counting it as done — otherwise the card fraction lies.
        if status == "completed" and output_folder_ok:
            of = ch.get("output_file") or ""
            if of:
                candidate = of if os.path.isabs(of) else os.path.join(
                    output_folder, of)
                if not os.path.isfile(candidate):
                    # Demote locally for counting. The JSON itself
                    # isn't rewritten — the translator's own progress
                    # manager is the source of truth for that.
                    status = "in_progress"
        if status == "completed":
            completed += 1
        elif status in ("in_progress", "pending"):
            in_progress += 1
        elif status in ("failed", "qa_failed"):
            failed += 1
    return {
        "total": total,
        "completed": completed,
        "in_progress": in_progress,
        "failed": failed,
        "prog": prog,
    }


# Process-level cache for EPUB spine-item counts, keyed by ``path|mtime``.
_SPINE_COUNT_CACHE: dict[str, int] = {}


def _count_epub_spine_items(epub_path: str, exclude_special: bool = False) -> int:
    """Return the number of itemrefs in the EPUB's spine (0 on failure).

    Cheap: reads only ``META-INF/container.xml`` + the OPF, never the
    chapter HTML. Results are memoised per ``(path, mtime, exclude_special)``
    so repeated scans don't re-open the zip. This is the authoritative
    chapter count for EPUB workspaces — ``translation_progress.json`` only
    tracks chapters the translator has actually processed, so it
    under-reports the real length for freshly imported / early-progress
    novels.

    When *exclude_special* is True, spine items whose manifest href has no
    digit in its basename (cover, nav, toc, info, message, …) are skipped,
    matching the translator's default behavior of not translating special
    files unless ``translate_special_files`` is explicitly enabled.
    """
    if not epub_path or not os.path.isfile(epub_path):
        return 0
    try:
        base_key = _epub_cache_key(epub_path)
    except Exception:
        base_key = ""
    cache_key = f"{base_key}|excl={int(bool(exclude_special))}" if base_key else ""
    if cache_key and cache_key in _SPINE_COUNT_CACHE:
        return _SPINE_COUNT_CACHE[cache_key]
    count = 0
    try:
        import zipfile
        import re as _re
        from xml.etree import ElementTree as ET
        with zipfile.ZipFile(epub_path, "r") as zf:
            names = zf.namelist()
            names_set = set(names)
            opf_path = None
            if "META-INF/container.xml" in names_set:
                try:
                    container_xml = zf.read("META-INF/container.xml").decode(
                        "utf-8", errors="replace")
                    tree = ET.fromstring(container_xml)
                    ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
                    rootfile = tree.find(".//c:rootfile", ns)
                    if rootfile is not None:
                        opf_path = rootfile.get("full-path")
                except Exception:
                    pass
            if not opf_path:
                for zname in names:
                    if zname.lower().endswith(".opf"):
                        opf_path = zname
                        break
            if opf_path and opf_path in names_set:
                try:
                    opf_xml = zf.read(opf_path).decode("utf-8", errors="replace")
                    tree = ET.fromstring(opf_xml)
                    OPF = "http://www.idpf.org/2007/opf"
                    manifest: dict[str, str] = {}
                    for item in tree.findall(f".//{{{OPF}}}item"):
                        item_id = item.get("id", "")
                        href = item.get("href", "")
                        if item_id and href:
                            manifest[item_id] = href
                    spine = tree.find(f".//{{{OPF}}}spine")
                    if spine is not None:
                        for itemref in spine.findall(f"{{{OPF}}}itemref"):
                            idref = itemref.get("idref") or ""
                            href = manifest.get(idref, "")
                            basename = os.path.basename(href)
                            # Auto-generated gallery page never counts
                            # toward the spine total, no matter the
                            # translate-special-files toggle.
                            if _is_gallery_filename(basename):
                                continue
                            if exclude_special:
                                if not _re.search(r"\d", basename):
                                    continue
                            count += 1
                except Exception:
                    pass
    except Exception:
        logger.debug("Spine count failed for %s: %s",
                     epub_path, traceback.format_exc())
    if cache_key:
        _SPINE_COUNT_CACHE[cache_key] = count
    return count


def _resolve_translate_special_files(config: dict | None) -> bool:
    """Return the effective ``translate_special_files`` setting.

    Environment variable ``TRANSLATE_SPECIAL_FILES`` takes precedence over
    the config dict so runtime overrides used by the translator work for
    the library scanner too.
    """
    env = os.environ.get("TRANSLATE_SPECIAL_FILES", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    return bool((config or {}).get("translate_special_files", False))


def _resolve_show_special_files(config: dict | None) -> bool:
    """Return the effective "show special files" flag for reader / details.

    Mirrors the logic baked into :class:`BookDetailsDialog.__init__` so
    callers that don't go through Book Details (e.g. the library card's
    "Open in Reader" context-menu action) still pick up the same resolved
    state: the explicit ``epub_details_show_special_files`` preference
    when set, else the global ``translate_special_files`` flag. An
    explicit False is only honoured when the global is also False —
    turning ON "translate special" always propagates through.
    """
    cfg = config or {}
    translate_special = _resolve_translate_special_files(cfg)
    stored = cfg.get("epub_details_show_special_files", None)
    if stored is None:
        return translate_special
    return bool(stored) or translate_special


def _is_special_spine_item(name: str) -> bool:
    """Return True if *name* is a "special" (non-numbered) spine item.

    Matches the digit-stem heuristic used by :func:`_count_epub_spine_items`
    and :func:`_count_translated_response_files`: any spine entry whose
    basename (minus a ``response_`` prefix, minus the extension) contains
    no digit is treated as a named special page — cover, nav, toc, info,
    message, afterword, etc. — rather than a numbered chapter. The
    reader hides these when the Show-special-files toggle is off so the
    TOC mirrors what the translator will actually process.
    """
    import re as _re
    if not name:
        return False
    base = os.path.basename(str(name)).lower()
    if base.startswith("response_"):
        base = base[len("response_"):]
    stem = os.path.splitext(base)[0]
    if not stem:
        return False
    return not _re.search(r"\d", stem)


def _count_translated_response_files(folder: str, exclude_special: bool = False) -> int:
    """Count ``response_*.{html,xhtml,htm,txt}`` files inside *folder*.

    Acts as a filesystem-based "done" count fallback for cards whose
    ``translation_progress.json`` is missing / empty (e.g. a crashed run
    that never flushed ``status=completed`` to the sidecar).

    When *exclude_special* is True, response files whose stem has no
    digit — ``response_cover.html``, ``response_nav.xhtml``, … — are
    skipped, matching the same heuristic used by
    :func:`_count_epub_spine_items`. This keeps the denominator and the
    numerator consistent when the "translate special files" toggle is
    off.
    """
    if not folder or not os.path.isdir(folder):
        return 0
    import re as _re
    count = 0
    try:
        for entry in os.scandir(folder):
            if not entry.is_file(follow_symlinks=False):
                continue
            lower = entry.name.lower()
            if not lower.startswith("response_"):
                continue
            if not lower.endswith((".html", ".xhtml", ".htm", ".txt")):
                continue
            # Gallery is auto-generated — never count it toward done.
            if _is_gallery_filename(entry.name):
                continue
            if exclude_special:
                stem = os.path.splitext(lower[len("response_"):])[0]
                if not _re.search(r"\d", stem):
                    continue
            count += 1
    except (PermissionError, OSError):
        return 0
    return count


def _folder_has_output_epub(folder: str) -> str | None:
    """Return the path to the first .epub in *folder* or None."""
    try:
        for entry in os.scandir(folder):
            if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(".epub"):
                return entry.path
    except (PermissionError, OSError):
        pass
    return None


def _folder_has_compiled_output(folder: str) -> tuple[str, str] | None:
    """Return ``(path, kind)`` for any compiled translation output, or None.

    Recognized kinds (in priority order):
      * ``"epub"`` — any ``*.epub`` at the folder root.
      * ``"pdf"``  — a ``*_translated.pdf`` alongside the progress file.
      * ``"txt"``  — a ``*_translated.txt`` alongside the progress file.
      * ``"html"`` — a ``*_translated.html`` alongside the progress file.
    """
    outputs = _list_compiled_outputs(folder)
    return outputs[0] if outputs else None


def _list_compiled_outputs(folder: str) -> list[tuple[str, str]]:
    """Return every compiled output in *folder* as ``[(path, kind), …]``.

    Same detection rules as :func:`_folder_has_compiled_output`, but
    returns the FULL list in priority order instead of just the first.
    Used by :func:`scan_output_folders` to flag folders that contain
    more than one compiled artefact (e.g. two ``.epub`` files from
    successive recompiles, or a ``.epub`` paired with a leftover
    ``*_translated.html``) so the card can render a warning badge and
    the user can investigate / clean up.
    """
    results: list[tuple[str, str]] = []
    try:
        entries = list(os.scandir(folder))
    except (PermissionError, OSError):
        return results
    # EPUBs win when present — that's the typical compiled shelf artifact.
    for entry in entries:
        if not entry.is_file(follow_symlinks=False):
            continue
        if entry.name.lower().endswith(".epub"):
            results.append((entry.path, "epub"))
    # Fall-back compiled outputs for TXT/PDF/HTML translations.
    priority = (("_translated.pdf", "pdf"),
                ("_translated.txt", "txt"),
                ("_translated.html", "html"))
    for suffix, kind in priority:
        for entry in entries:
            if not entry.is_file(follow_symlinks=False):
                continue
            nl = entry.name.lower()
            if nl.endswith(suffix):
                results.append((entry.path, kind))
    return results


def _detect_workspace_kind(folder: str, source_epub_path: str = "") -> str:
    """Best-effort classification of a translation workspace.

    Returns one of ``"epub"``, ``"txt"``, ``"pdf"``, ``"image"`` or ``"other"``.
    The heuristic is:
      1. If *source_epub_path* is provided and has a known extension, use it.
      2. Folder contains ``content.opf`` / ``*.epub`` → EPUB.
      3. Folder contains a compiled ``*_translated.{txt,pdf,html}`` → that kind.
      4. Folder contains ``word_count/`` and no OPF → TXT (the Glossarion
         text translator writes word_count per chunk, EPUBs don't).
      5. Otherwise → ``"other"``.
    """
    if source_epub_path:
        low = source_epub_path.lower()
        for ext, kind in ((".epub", "epub"), (".txt", "txt"), (".pdf", "pdf")):
            if low.endswith(ext):
                return kind
        if low.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")):
            return "image"
    try:
        has_opf = False
        has_epub = False
        has_word_count_dir = False
        translated_ext: str | None = None
        for entry in os.scandir(folder):
            nm = entry.name.lower()
            if entry.is_file(follow_symlinks=False):
                if nm.endswith(".opf"):
                    has_opf = True
                elif nm.endswith(".epub"):
                    has_epub = True
                elif nm.endswith("_translated.txt"):
                    translated_ext = translated_ext or "txt"
                elif nm.endswith("_translated.pdf"):
                    translated_ext = translated_ext or "pdf"
                elif nm.endswith("_translated.html"):
                    translated_ext = translated_ext or "html"
            elif entry.is_dir(follow_symlinks=False):
                if nm == "word_count":
                    has_word_count_dir = True
    except (PermissionError, OSError):
        return "other"
    if has_opf or has_epub:
        return "epub"
    if translated_ext:
        return translated_ext
    if has_word_count_dir:
        return "txt"
    return "other"


def _read_source_epub_pointer(folder: str) -> str | None:
    """Return the source EPUB path recorded in ``source_epub.txt`` if present.

    The translator drops this sidecar file inside the output folder pointing at
    the original input EPUB — a more reliable hint than matching folder names.
    Forward and backward slashes are normalized; relative paths are resolved
    against the output folder itself.
    """
    sidecar = os.path.join(folder, "source_epub.txt")
    if not os.path.isfile(sidecar):
        return None
    try:
        with open(sidecar, "r", encoding="utf-8") as f:
            raw = f.read().strip()
    except OSError:
        return None
    if not raw:
        return None
    candidate = raw.replace("/", os.sep).replace("\\", os.sep)
    if not os.path.isabs(candidate):
        candidate = os.path.join(folder, candidate)
    return os.path.normpath(candidate) if os.path.isfile(candidate) else None


def _find_raw_source_for_folder(folder: str) -> str | None:
    """Try to locate the raw source file that fed this output folder.

    Search order:
      1. ``<folder>/source_epub.txt`` — authoritative pointer.
      2. ``Library/Raw/<folder_basename>.<ext>`` — imported raw source.
      3. ``<folder_basename>.<ext>`` in the allowed output roots.
      4. Any entry in ``library_raw_inputs.txt`` whose stem matches.
    """
    # 1. source_epub.txt pointer
    pointed = _read_source_epub_pointer(folder)
    if pointed and os.path.isfile(pointed):
        return pointed
    folder_name = os.path.basename(os.path.normpath(folder))
    if not folder_name:
        return None
    # 2. Library/Raw
    raw_dir = get_library_raw_dir()
    for ext in (".epub", ".txt", ".pdf", ".html"):
        candidate = os.path.join(raw_dir, folder_name + ext)
        if os.path.isfile(candidate):
            return candidate
    # 3. Raw-inputs registry
    for p in load_library_raw_inputs():
        if not os.path.isfile(p):
            continue
        stem, _ = os.path.splitext(os.path.basename(p))
        if stem == folder_name:
            return p
    return None


def _resolve_book_output_folder(book: dict) -> str:
    """Return the output-folder path that belongs to *book*, or ``""``.

    A "real" output folder is one the translator actually wrote into —
    i.e. the workspace holding ``translation_progress.json`` /
    ``response_*.html`` artefacts. This helper is deliberately strict
    about not dressing up ``Library/Translated`` (where a compiled EPUB
    just happens to live) as an output folder, because that's never
    what the user means when they ask for the "output folder".

    Resolution order:

      1. ``book['output_folder']`` — set by :func:`scan_output_folders`
         for in-progress + promoted-compiled cards.
      2. ``library_origins['translated']`` — for ``Library/Translated``
         entries the stored "pre-organize" path's *parent* was the
         original output folder; usually still on disk because organize
         only moves the compiled EPUB out of it.

    Returns ``""`` when neither resolves to an existing directory so the
    caller can disable the button / take a different fallback instead
    of opening the book's own containing folder (which, for library
    entries, is ``Library/Translated`` — not what the user wants).
    """
    out = (book.get("output_folder") or "") if isinstance(book, dict) else ""
    if out and os.path.isdir(out):
        return out
    if isinstance(book, dict) and book.get("in_library"):
        try:
            origins = _load_origins()
            trans_map = origins.get("translated", {}) or {}
            orig_path = trans_map.get(os.path.basename(book.get("path", "")))
            if orig_path:
                orig_folder = os.path.dirname(str(orig_path))
                if orig_folder and os.path.isdir(orig_folder):
                    return orig_folder
        except Exception:
            logger.debug("Output-folder origins lookup failed: %s",
                         traceback.format_exc())
    return ""


def _find_raw_source_for_library_epub(library_epub_path: str) -> str:
    """Best-effort lookup for the raw source of a Library/Translated EPUB.

    Unlike :func:`_find_raw_source_for_folder` (which starts from an
    output-folder layout complete with ``source_epub.txt``), this helper
    works purely from a compiled EPUB sitting inside ``Library/Translated``
    and has to recover the original source by name matching + the origins
    registry. Search order:

      0. ``library_origins['pairs']`` — an explicit translated↔raw
         mapping written at organize time. This is the ONLY step that
         survives a raw whose filename stem doesn't match the compiled
         EPUB's stem (e.g. a Korean raw paired with an English
         translation), which is the common case for real translations.
      1. ``Library/Raw/<same-stem>.epub`` — direct basename match, the
         common case for files organized into the library together.
      2. ``load_library_raw_inputs()`` — any registered raw input whose
         filename stem matches.
      3. ``library_origins['translated']`` → source output folder →
         ``source_epub.txt`` pointer. Works until the output folder is
         deleted or the sidecar is stale.

    Returns an absolute path string, or ``""`` when nothing matched.
    """
    if not library_epub_path or not os.path.isfile(library_epub_path):
        return ""
    stem = os.path.splitext(os.path.basename(library_epub_path))[0]
    if not stem:
        return ""
    raw_dir = get_library_raw_dir()
    # 0. Explicit translated→raw pairing persisted at organize time.
    try:
        origins = _load_origins()
        pairs = origins.get("pairs", {}) or {}
        pair_raw_basename = pairs.get(os.path.basename(library_epub_path))
        if pair_raw_basename:
            pair_path = os.path.join(raw_dir, pair_raw_basename)
            if os.path.isfile(pair_path):
                return os.path.abspath(pair_path)
    except Exception:
        logger.debug("Library-raw pairs lookup failed: %s",
                     traceback.format_exc())
        origins = {}
    # 1. Library/Raw/<stem>.epub (case-insensitive extension match)
    try:
        with os.scandir(raw_dir) as it:
            for entry in it:
                if not entry.is_file(follow_symlinks=False):
                    continue
                nm = entry.name
                nl = nm.lower()
                if not nl.endswith(".epub"):
                    continue
                if os.path.splitext(nm)[0] == stem:
                    return os.path.abspath(entry.path)
    except (PermissionError, OSError, FileNotFoundError):
        pass
    # 2. Raw-inputs registry
    for p in load_library_raw_inputs():
        if not p or not os.path.isfile(p):
            continue
        if not p.lower().endswith(".epub"):
            continue
        if os.path.splitext(os.path.basename(p))[0] == stem:
            return os.path.abspath(p)
    # 3. origins["translated"] → source output folder → raw resolver
    try:
        trans_map = (origins or _load_origins()).get("translated", {}) or {}
        orig_path = trans_map.get(os.path.basename(library_epub_path))
        if orig_path:
            orig_folder = os.path.dirname(orig_path)
            if orig_folder and os.path.isdir(orig_folder):
                resolved = _find_raw_source_for_folder(orig_folder)
                if resolved and os.path.isfile(resolved):
                    return os.path.abspath(resolved)
    except Exception:
        logger.debug("Library-raw origins lookup failed: %s",
                     traceback.format_exc())
    return ""


# ---------------------------------------------------------------------------
# Cover helpers
# ---------------------------------------------------------------------------

def _find_cover_in_dir(folder: str) -> str | None:
    """Look for a cover image inside *folder* (or its ``images/`` subfolders).

    Preference order:
      1. *cover* images directly in the folder.
      2. Any image in the folder.
      3. *cover* images in ``images/`` or ``translated_images/``.
      4. The smallest-numbered image in ``images/``.
    """
    import re as _re
    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

    def _natural_key(p):
        name = os.path.basename(p).lower()
        nums = _re.findall(r"\d+", name)
        return int(nums[0]) if nums else 0

    def _scan(dir_path: str) -> str | None:
        if not dir_path or not os.path.isdir(dir_path):
            return None
        covers: list[str] = []
        any_imgs: list[str] = []
        try:
            for entry in os.scandir(dir_path):
                if not entry.is_file(follow_symlinks=False):
                    continue
                nl = entry.name.lower()
                ext = os.path.splitext(nl)[1]
                if ext not in _IMG_EXTS:
                    continue
                if "cover" in nl:
                    covers.append(entry.path)
                any_imgs.append(entry.path)
        except (PermissionError, OSError):
            return None
        if covers:
            covers.sort(key=_natural_key)
            return covers[0]
        if any_imgs:
            any_imgs.sort(key=_natural_key)
            return any_imgs[0]
        return None

    direct = _scan(folder)
    if direct:
        return direct
    for sub in ("images", "translated_images"):
        r = _scan(os.path.join(folder, sub))
        if r:
            return r
    return None


# ---------------------------------------------------------------------------
# Tab scanners: Completed (Library) and In Progress (output folders)
# ---------------------------------------------------------------------------

def scan_library_completed(config: dict | None = None) -> list[dict]:
    """Scan ``Library/Translated`` for completed EPUBs.

    The Translated subfolder is a curated shelf: every EPUB inside is treated
    as a finished work with no translation context. Before scanning we run
    a one-time migration that moves any EPUBs still sitting in the legacy
    Library root into ``Translated/`` so pre-refactor layouts aren't lost.
    """
    # Legacy layout support — idempotent, safe to call every scan.
    _migrate_legacy_library_layout()
    library_dir = os.path.normpath(os.path.abspath(get_library_translated_dir()))
    results: list[dict] = []
    seen: set[str] = set()

    def _walk(root: str, max_depth: int = 4, depth: int = 0):
        if depth > max_depth:
            return
        try:
            with os.scandir(root) as it:
                for entry in it:
                    try:
                        if entry.is_file(follow_symlinks=False):
                            lower = entry.name.lower()
                            if not lower.endswith(".epub"):
                                continue
                            norm = os.path.normpath(os.path.abspath(entry.path))
                            if norm in seen:
                                continue
                            seen.add(norm)
                            try:
                                stat = os.stat(entry.path)
                            except OSError:
                                continue
                            # Resolve a possible raw counterpart in
                            # Library/Raw so the reader's "Raw" toggle
                            # can flip between compiled and source
                            # without re-picking the file manually.
                            raw_counterpart = _find_raw_source_for_library_epub(
                                entry.path)
                            results.append({
                                "name": os.path.splitext(entry.name)[0],
                                "path": entry.path,
                                "size": stat.st_size,
                                "mtime": stat.st_mtime,
                                "in_library": True,
                                "type": "epub",
                                "raw_source_path": raw_counterpart,
                            })
                        elif entry.is_dir(follow_symlinks=False) and not entry.name.startswith("."):
                            _walk(entry.path, max_depth, depth + 1)
                    except (PermissionError, OSError):
                        pass
        except (PermissionError, OSError):
            pass

    if os.path.isdir(library_dir):
        _walk(library_dir)
    results.sort(key=lambda r: r["mtime"], reverse=True)
    return results


def scan_output_folders(config: dict | None = None) -> list[dict]:
    """Scan the output root(s) for translation folders.

    Honors the OUTPUT_DIRECTORY override strictly via :func:`_resolve_output_roots`.
    A folder qualifies when it has either a compiled ``.epub`` *or* a
    translation_progress.json with at least one recorded chapter.

    Each result carries ``metadata_json`` (already loaded) so the card/details
    dialog can render without a second filesystem hit.

    Folders with a compiled ``.epub`` are emitted as ``type="epub"`` so they
    render identically to Library entries in the Completed tab (``path`` is
    the .epub itself). Folders without a compiled EPUB are emitted as
    ``type="in_progress"`` for the In Progress tab. Callers use
    :func:`split_output_folders_by_status` to partition the two lists.
    """
    import json as _json
    config = config or {}
    roots = _resolve_output_roots(config)
    if not roots:
        return []
    # Honor the "translate special files" toggle: when OFF (default), special
    # files (cover/nav/toc/info/…) are never translated, so they shouldn't
    # count toward the card's total. Otherwise a fully-translated EPUB sits
    # at 98/100 forever because two special files were skipped by design.
    exclude_special = not _resolve_translate_special_files(config)

    results: list[dict] = []
    seen_folders: set[str] = set()
    for root in roots:
        try:
            it = os.scandir(root)
        except (PermissionError, OSError):
            continue
        with it:
            for entry in it:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                folder = entry.path
                key = os.path.normcase(os.path.normpath(folder))
                if key in seen_folders:
                    continue
                seen_folders.add(key)

                progress_file = os.path.join(folder, "translation_progress.json")
                metadata_file = os.path.join(folder, "metadata.json")
                # Walk EVERY compiled output so we can warn the user
                # when a folder contains more than one (e.g. a stale
                # ``*.epub`` from a previous compile left next to a new
                # one, or a ``*_translated.html`` paired with a
                # compiled ``.epub``). The first entry keeps its usual
                # "primary compiled output" role.
                compiled_outputs = _list_compiled_outputs(folder)
                compiled = compiled_outputs[0] if compiled_outputs else None
                output_epub = compiled[0] if (compiled and compiled[1] == "epub") else None
                compiled_path = compiled[0] if compiled else None
                compiled_kind = compiled[1] if compiled else None
                # Conflicts = every compiled file beyond the primary.
                # We record (basename, kind) so the _BookCard tooltip
                # can name each one without showing full absolute paths.
                compiled_conflicts = [
                    (os.path.basename(p), k)
                    for p, k in compiled_outputs[1:]
                ]
                has_progress = os.path.isfile(progress_file)
                has_metadata = os.path.isfile(metadata_file)

                # Eagerly load the progress summary so we can tell a *seed*
                # progress file (created when the user loaded a glossary but
                # never ran a translation) apart from an active one. Pass
                # ``exclude_special`` so the sidecar's counts line up with
                # the spine / filesystem counts — otherwise the translate-
                # special-files toggle shifts the denominator without
                # budging the numerator and the card reads as 100 % no
                # matter what the user does.
                summary = _read_progress_summary(
                    progress_file, exclude_special=exclude_special
                ) if has_progress else None
                progress_total = summary["total"] if summary else 0
                progress_done = summary["completed"] if summary else 0
                failed = summary["failed"] if summary else 0

                # Raw source must be resolvable for the In Progress tab.
                # Cards whose source EPUB is missing (moved/deleted/offline)
                # are hidden rather than shown as ghost cards.
                raw_source_path = _find_raw_source_for_folder(folder)
                raw_abs = os.path.normcase(os.path.normpath(
                    os.path.abspath(get_library_raw_dir())))
                raw_in_library = bool(raw_source_path) and (
                    os.path.normcase(os.path.normpath(
                        os.path.abspath(os.path.dirname(raw_source_path)))) == raw_abs
                )

                # Enrich the card fraction with richer data sources than
                # translation_progress.json alone:
                #   * total = authoritative spine count from the source EPUB's
                #             content.opf (the progress file only tracks
                #             chapters the translator has processed, so it
                #             under-reports for fresh / early-progress novels).
                #   * done  = progress file's "completed" count when the
                #             sidecar has real entries (authoritative —
                #             status="in_progress" entries don't count as
                #             done, which is what the user expects for
                #             specials still being translated). Filesystem
                #             response_*.html count is only used as a
                #             fallback when the progress file is missing or
                #             empty. The filesystem count also respects
                #             exclude_special so partial stub files for
                #             skipped specials can't inflate the fraction.
                fs_done = _count_translated_response_files(
                    folder, exclude_special=exclude_special)
                spine_total = 0
                if raw_source_path and raw_source_path.lower().endswith(".epub"):
                    spine_total = _count_epub_spine_items(
                        raw_source_path, exclude_special=exclude_special)
                total = max(progress_total, spine_total)
                if progress_total > 0:
                    # Progress file is authoritative — in-progress specials
                    # have ``status="in_progress"`` and don't get counted
                    # here, so a book whose only pending chapters are
                    # specials stays below 100% until they're done (when
                    # ``translate_special_files`` is on).
                    done = progress_done
                else:
                    done = fs_done

                # Workspace must prove it's a real translation candidate.
                # Filter order:
                #   * compiled output present                                  → keep (Completed).
                #   * at least one chapter recorded in progress file          → keep.
                #   * ``Library/Raw`` contains the raw source for this folder → keep
                #     as "Not started" so freshly imported EPUBs surface even
                #     though their progress file is a 68-byte seed.
                #   * otherwise → skip (stray/abandoned folder).
                if not compiled and progress_total <= 0 and not raw_in_library:
                    continue
                # A resolvable raw source is required even for the fresh
                # imports above; without it we can't render any meaningful
                # content for the card.
                if not compiled and not raw_source_path:
                    continue

                # Compute translation_state for the UI. This decides whether
                # the card shows a "Not started" pill or an "In progress"
                # fraction in :class:`_BookCard` AND which tab it lands on.
                #
                # IMPORTANT: the done/total fraction is the authoritative
                # signal, NOT the presence of a compiled ``.epub`` in the
                # folder. A compiled EPUB from a previous partial run can
                # sit next to an in-progress translation whose spine isn't
                # fully covered yet — e.g. 58/60 with specials still
                # pending and ``translate_special_files`` on. In that case
                # we MUST classify as "in_progress" so the toggle actually
                # moves the card between tabs.
                if done >= total and total > 0:
                    translation_state = "completed"
                elif progress_total <= 0 and fs_done == 0:
                    # No meaningful progress data at all — trust the
                    # presence of a compiled EPUB as a "completed" signal
                    # (library import / pre-existing build). Otherwise
                    # this is a freshly scaffolded workspace.
                    translation_state = "completed" if compiled else "not_started"
                else:
                    translation_state = "in_progress"

                # Classify source kind (epub / txt / pdf / image / other) so
                # the card can show a meaningful type badge. We prefer the
                # resolved raw source extension, then fall back to the
                # filesystem heuristic applied to the output folder.
                workspace_kind = _detect_workspace_kind(folder, raw_source_path or "")

                metadata_json: dict = {}
                if has_metadata:
                    try:
                        with open(metadata_file, "r", encoding="utf-8") as f:
                            loaded = _json.load(f)
                        if isinstance(loaded, dict):
                            metadata_json = loaded
                    except (OSError, _json.JSONDecodeError):
                        metadata_json = {}

                # Title precedence: metadata.json title → folder basename.
                raw_title = (metadata_json.get("title")
                             or metadata_json.get("original_title")
                             or entry.name)

                # Card shape follows the real translation state, NOT the
                # mere presence of a compiled ``.epub``. A compiled file
                # next to an in-progress translation (partial run / user
                # recompiled mid-way with specials still pending) keeps
                # the folder-based card so ribbons + progress pills still
                # render. Only a genuinely completed workspace promotes
                # the card to a plain EPUB/PDF/TXT/HTML entry.
                promote_to_compiled = bool(compiled) and translation_state == "completed"
                if promote_to_compiled:
                    card_path = compiled_path
                    card_type = compiled_kind  # "epub" / "pdf" / "txt" / "html"
                    is_in_progress = False
                    try:
                        stat = os.stat(compiled_path)
                    except OSError:
                        continue
                else:
                    card_path = folder
                    card_type = "in_progress"
                    is_in_progress = (translation_state != "completed")
                    try:
                        stat = os.stat(folder)
                    except OSError:
                        continue

                results.append({
                    "name": raw_title,
                    "folder_name": entry.name,
                    "path": card_path,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "in_library": False,
                    "type": card_type,
                    # Source file on disk (resolved via source_epub.txt,
                    # Library/Raw match, or raw-inputs registry). Used by
                    # the "Load for translation" action.
                    "raw_source_path": raw_source_path or "",
                    # The workspace kind is the *source* type (epub/txt/pdf
                    # /image/other). Independent of ``type`` (which may be
                    # the compiled output kind). Used by the card to pick
                    # the right badge for in-progress folders.
                    "workspace_kind": workspace_kind,
                    # Translation lifecycle: not_started / in_progress /
                    # completed. Drives the pill label on the card.
                    "translation_state": translation_state,
                    "is_in_progress": is_in_progress,
                    "output_folder": folder,
                    "progress_file": progress_file if has_progress else "",
                    "metadata_json_path": metadata_file if has_metadata else "",
                    "metadata_json": metadata_json,
                    "total_chapters": total,
                    "completed_chapters": done,
                    "failed_chapters": failed,
                    "pending_chapters": max(0, total - done),
                    "has_output_epub": bool(output_epub),
                    "output_epub_path": output_epub or "",
                    "has_compiled_output": bool(compiled),
                    "compiled_output_path": compiled_path or "",
                    "compiled_output_kind": compiled_kind or "",
                    # List of ``(basename, kind)`` tuples for compiled
                    # artefacts in the folder BEYOND the primary one.
                    # Empty list for normal single-output folders;
                    # populated when the scanner detected stale or
                    # duplicate compiled files so the card can show
                    # a ⚠ warning badge.
                    "compiled_conflicts": compiled_conflicts,
                })
    results.sort(key=lambda r: r["mtime"], reverse=True)
    return results


def split_output_folders_by_status(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split ``scan_output_folders`` results into (completed, in_progress).

    Routing is driven by ``translation_state`` — ``"completed"`` means
    done >= total (the scanner already treats that as authoritative, so
    a compiled ``.epub`` alone doesn't force the card to Completed when
    only 58/60 chapters are actually translated). Missing / empty state
    falls back to ``has_compiled_output`` / ``has_output_epub`` for
    backwards compatibility with pre-v2 rows.
    """
    completed: list[dict] = []
    in_progress: list[dict] = []
    for r in rows:
        state = r.get("translation_state")
        if state == "completed":
            completed.append(r)
        elif state in ("in_progress", "not_started"):
            in_progress.append(r)
        elif r.get("has_compiled_output") or r.get("has_output_epub"):
            completed.append(r)
        else:
            in_progress.append(r)
    return completed, in_progress


def _find_in_progress_novels(config: dict | None = None) -> list[dict]:
    """Locate novels whose translation is in progress (no output EPUB yet).

    Strictly only inspects the output roots returned by :func:`_resolve_output_roots`.
    For each first-level subfolder that contains a `translation_progress.json`
    and has NO output `.epub`, try to find the source EPUB via (in priority order):
      1. ``<folder>/source_epub.txt`` — authoritative pointer written by the
         translator.
      2. A basename match (folder name == EPUB basename) in the allowed roots.

    EPUBs already organized into the Library folder are *never* reported as
    in-progress: the Library is the curated/read-only shelf and the progress
    view is deliberately limited to active translations.

    Returns a list of dicts compatible with the rest of the scanner (with
    extra in-progress metadata).
    """
    roots = _resolve_output_roots(config)
    if not roots:
        return []

    library_dir_norm = os.path.normcase(os.path.normpath(os.path.abspath(get_library_dir())))

    def _is_in_library(path: str) -> bool:
        """True when *path* resolves inside the Glossarion Library folder."""
        try:
            norm = os.path.normcase(os.path.normpath(os.path.abspath(path)))
        except (TypeError, ValueError):
            return False
        return (norm == library_dir_norm
                or norm.startswith(library_dir_norm + os.sep)
                or os.path.normcase(os.path.dirname(norm)) == library_dir_norm)

    # Index every EPUB in the roots by basename (without extension) so we can
    # match folders to source files without a second filesystem walk per folder.
    epub_by_base: dict[str, str] = {}
    for root in roots:
        try:
            for entry in os.scandir(root):
                if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(".epub"):
                    base = os.path.splitext(entry.name)[0]
                    # Prefer first match; tolerate case-insensitive dupes
                    epub_by_base.setdefault(base, entry.path)
                    epub_by_base.setdefault(base.lower(), entry.path)
        except (PermissionError, OSError):
            continue

    results: list[dict] = []
    seen_folders: set[str] = set()
    for root in roots:
        try:
            it = os.scandir(root)
        except (PermissionError, OSError):
            continue
        with it:
            for entry in it:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                folder = entry.path
                folder_key = os.path.normcase(os.path.normpath(folder))
                if folder_key in seen_folders:
                    continue
                seen_folders.add(folder_key)
                progress_file = os.path.join(folder, "translation_progress.json")
                if not os.path.isfile(progress_file):
                    continue
                if _folder_has_output_epub(folder):
                    # Already done — normal scan will pick up the .epub.
                    continue
                summary = _read_progress_summary(progress_file)
                if summary is None:
                    continue
                folder_name = entry.name
                # 1. Authoritative pointer file written by the translator.
                source_path = _read_source_epub_pointer(folder)
                # 2. Fall back to folder-name↔basename match within the allowed roots.
                if not source_path:
                    source_path = epub_by_base.get(folder_name) or epub_by_base.get(folder_name.lower())
                if not source_path or not os.path.isfile(source_path):
                    # No source EPUB visible — skip so we don't surface a ghost entry.
                    continue
                # Never mark Library-organized EPUBs as in-progress: the user
                # wants those to look clean (no progress badge) in the library
                # unless a progress file is explicitly present inside the
                # configured output root — which this path is not.
                if _is_in_library(source_path):
                    continue
                try:
                    stat = os.stat(source_path)
                except OSError:
                    continue
                results.append({
                    "name": folder_name,
                    "path": source_path,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "in_library": False,
                    "type": "epub",
                    "is_in_progress": True,
                    "output_folder": folder,
                    "progress_file": progress_file,
                    "total_chapters": summary["total"],
                    "completed_chapters": summary["completed"],
                    "failed_chapters": summary["failed"],
                    "pending_chapters": max(0, summary["total"] - summary["completed"]),
                })
    return results


def scan_for_epubs(config: dict | None = None) -> list[dict]:
    config = config or {}
    results: list[dict] = []
    seen: set[str] = set()

    library_dir = os.path.normpath(os.path.abspath(get_library_dir()))

    def _add(path: str, file_type: str = "epub"):
        norm = os.path.normpath(os.path.abspath(path))
        if norm in seen:
            return
        seen.add(norm)
        try:
            stat = os.stat(path)
            in_lib = norm.startswith(library_dir + os.sep) or os.path.dirname(norm) == library_dir
            results.append({
                "name": os.path.splitext(os.path.basename(path))[0],
                "path": path,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "in_library": in_lib,
                "type": file_type,
            })
        except OSError:
            pass

    def _walk(root: str, max_depth: int = 3, depth: int = 0):
        if depth > max_depth:
            return
        try:
            with os.scandir(root) as it:
                for entry in it:
                    try:
                        if entry.is_file(follow_symlinks=False):
                            lower = entry.name.lower()
                            if lower.endswith(".epub"):
                                _add(entry.path, "epub")
                            elif lower.endswith(".pdf"):
                                _add(entry.path, "pdf")
                            elif lower.endswith(".txt") and "_translated" in lower:
                                _add(entry.path, "txt")
                        elif entry.is_dir(follow_symlinks=False) and not entry.name.startswith("."):
                            _walk(entry.path, max_depth, depth + 1)
                    except (PermissionError, OSError):
                        pass
        except (PermissionError, OSError):
            pass

    _walk(library_dir, max_depth=4)

    # Strictly honor the OUTPUT_DIRECTORY override: when set, we walk only
    # that directory for translation outputs / in-progress sources. Without
    # an override, we walk the default (app dir / CWD). This mirrors the
    # same rule applied by :func:`_find_in_progress_novels`.
    for root in _resolve_output_roots(config):
        _walk(root)

    # In-progress novels: source EPUBs with a translation_progress.json but no
    # compiled .epub yet. Strictly limited to the default / override output dirs.
    try:
        in_progress = _find_in_progress_novels(config)
    except Exception:
        logger.debug("In-progress scan failed: %s", traceback.format_exc())
        in_progress = []
    for ip in in_progress:
        norm = os.path.normpath(os.path.abspath(ip["path"]))
        if norm in seen:
            # Annotate the existing result with in-progress data — but never
            # for library-organized EPUBs. The Library shelf is supposed to
            # stay status-free unless a matching progress file was found in
            # the configured output root; _find_in_progress_novels already
            # guards against that, but be defensive here as well.
            for r in results:
                if os.path.normpath(os.path.abspath(r["path"])) == norm:
                    if r.get("in_library"):
                        break
                    r.update({
                        "is_in_progress": True,
                        "output_folder": ip["output_folder"],
                        "progress_file": ip["progress_file"],
                        "total_chapters": ip["total_chapters"],
                        "completed_chapters": ip["completed_chapters"],
                        "failed_chapters": ip["failed_chapters"],
                        "pending_chapters": ip["pending_chapters"],
                    })
                    break
            continue
        seen.add(norm)
        results.append(ip)

    results.sort(key=lambda r: r["mtime"], reverse=True)

    # Attach original source paths for files that were moved to Library.
    # v2 origins are split into raw/translated buckets; flatten them for
    # this lookup since we just want "original path for this basename".
    origins = _load_origins()
    flat_origins: dict[str, str] = {}
    for bucket in ("raw", "translated"):
        flat_origins.update(origins.get(bucket, {}) or {})
    for r in results:
        basename = os.path.basename(r["path"])
        if basename in flat_origins:
            r["original_path"] = flat_origins[basename]

    return results


# ---------------------------------------------------------------------------
# Library Dialog — constants & helpers
# ---------------------------------------------------------------------------

SORT_DATE = "date"
SORT_NAME = "name"
SORT_SIZE = "size"

SIZE_COMPACT = "compact"
SIZE_NORMAL = "normal"
SIZE_LARGE = "large"
SIZE_XL = "xl"
SIZE_2XL = "2xl"
SIZE_3XL = "3xl"
SIZE_4XL = "4xl"
SIZE_5XL = "5xl"
SIZE_6XL = "6xl"

_ALL_SIZES = [SIZE_COMPACT, SIZE_NORMAL, SIZE_LARGE, SIZE_XL, SIZE_2XL, SIZE_3XL, SIZE_4XL, SIZE_5XL, SIZE_6XL]

# Title rendering: there's no hard character cap anymore — the title is
# rendered at ``title_size`` first; if it overflows ``title_max_h`` vertically
# we dynamically shrink the font down to ``title_min_size`` and only truncate
# with an ellipsis if even that's not enough. See :func:`_fit_title_text`.
_SIZE_PRESETS = {
    SIZE_COMPACT: {"card_w": 110, "cover_h": 140, "title_size": "8.5pt",  "title_min_size": "6.5pt", "title_max_h": 48, "spacing": 3},
    SIZE_NORMAL:  {"card_w": 140, "cover_h": 175, "title_size": "9pt",    "title_min_size": "7pt",   "title_max_h": 52, "spacing": 4},
    SIZE_LARGE:   {"card_w": 180, "cover_h": 225, "title_size": "9.5pt",  "title_min_size": "7.5pt", "title_max_h": 58, "spacing": 5},
    SIZE_XL:      {"card_w": 230, "cover_h": 290, "title_size": "10pt",   "title_min_size": "8pt",   "title_max_h": 64, "spacing": 6},
    SIZE_2XL:     {"card_w": 290, "cover_h": 365, "title_size": "10.5pt", "title_min_size": "8pt",   "title_max_h": 72, "spacing": 8},
    SIZE_3XL:     {"card_w": 360, "cover_h": 450, "title_size": "11pt",   "title_min_size": "8.5pt", "title_max_h": 80, "spacing": 10},
    SIZE_4XL:     {"card_w": 440, "cover_h": 550, "title_size": "11.5pt", "title_min_size": "9pt",   "title_max_h": 88, "spacing": 12},
    SIZE_5XL:     {"card_w": 530, "cover_h": 660, "title_size": "12pt",   "title_min_size": "9.5pt", "title_max_h": 96, "spacing": 14},
    SIZE_6XL:     {"card_w": 630, "cover_h": 790, "title_size": "12.5pt", "title_min_size": "10pt",  "title_max_h": 104, "spacing": 16},
}


def _parse_pt(pt_str) -> float:
    """Parse a CSS-like point-size string (e.g. ``"8.5pt"``) to a float."""
    try:
        return float(str(pt_str).replace("pt", "").strip())
    except (ValueError, TypeError):
        return 9.0


def _fit_title_text(
    text: str,
    avail_width: int,
    max_height: int,
    base_pt: float,
    base_font: QFont | None = None,
    min_pt: float = 6.5,
    step: float = 0.5,
) -> tuple[str, float]:
    """Return (rendered_text, font_pt) that fits inside the given box.

    Strategy:
      1. Render *text* at *base_pt* with word-wrap inside ``avail_width``.
      2. If the wrapped text overflows ``max_height`` vertically, shrink the
         font size in ``step`` pt increments down to ``min_pt``.
      3. If even at ``min_pt`` the text still overflows, truncate it with
         an ellipsis at the longest prefix that does fit (binary search).

    No hard character cap — short titles always render at full size, and
    long titles gracefully scale / trim as needed.
    """
    text = text or ""
    avail_width = max(1, int(avail_width))
    max_height = max(1, int(max_height))
    base_pt = float(base_pt)
    min_pt = min(float(min_pt), base_pt)

    def _height(s: str, pt_size: float) -> int:
        f = QFont(base_font) if base_font is not None else QFont()
        f.setPointSizeF(pt_size)
        f.setBold(True)
        fm = QFontMetrics(f)
        rect = fm.boundingRect(
            QRect(0, 0, avail_width, 100_000),
            int(Qt.TextWordWrap),
            s,
        )
        return rect.height()

    pt = base_pt
    while pt > min_pt and _height(text, pt) > max_height:
        pt = max(min_pt, pt - step)

    if _height(text, pt) <= max_height:
        return text, pt

    # Overflows even at the minimum size — truncate with an ellipsis.
    ellipsis = "\u2026"
    lo, hi, best = 1, max(1, len(text)), 1
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip() + ellipsis
        if _height(candidate, pt) <= max_height:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return text[:best].rstrip() + ellipsis, pt


def _find_folder_cover(file_path: str, config: dict | None = None, original_path: str | None = None) -> str | None:
    """Find a cover image for a PDF/TXT file.

    Search order:
      1. *cover* images in the file's own directory
      2. Original source path directory (from library_origins.txt)
      3. Output folder by base name — covers files moved to Library
    """
    import re as _re
    folder = os.path.dirname(file_path)

    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

    def _natural_key(p):
        name = os.path.basename(p)
        nums = _re.findall(r'\d+', name)
        return int(nums[0]) if nums else 0

    def _scan_for_cover(search_dir: str) -> str | None:
        """Look for *cover* images in search_dir, then any image in images/ subfolder."""
        if not os.path.isdir(search_dir):
            return None
        candidates = []
        try:
            for entry in os.scandir(search_dir):
                if entry.is_file(follow_symlinks=False):
                    nl = entry.name.lower()
                    ext = os.path.splitext(nl)[1]
                    if ext in _IMG_EXTS and "cover" in nl:
                        candidates.append(entry.path)
        except (PermissionError, OSError):
            pass
        if candidates:
            candidates.sort(key=_natural_key)
            return candidates[0]

        # Check images/ subfolder
        img_dir = os.path.join(search_dir, "images")
        if os.path.isdir(img_dir):
            img_cands = []
            try:
                for entry in os.scandir(img_dir):
                    if entry.is_file(follow_symlinks=False):
                        ext = os.path.splitext(entry.name.lower())[1]
                        if ext in _IMG_EXTS:
                            img_cands.append(entry.path)
            except (PermissionError, OSError):
                pass
            if img_cands:
                img_cands.sort(key=_natural_key)
                return img_cands[0]
        return None

    # 1. Check the file's own directory
    result = _scan_for_cover(folder)
    if result:
        return result

    # 2. Check original source path directory (persisted when moved to Library)
    if original_path:
        orig_dir = os.path.dirname(original_path)
        result = _scan_for_cover(orig_dir)
        if result:
            return result

    # 3. Check the original output folder by base name
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    config = config or {}
    output_dirs_to_check = []

    override = os.environ.get("OUTPUT_DIRECTORY") or config.get("output_directory")
    if override and os.path.isdir(override):
        output_dirs_to_check.append(os.path.join(os.path.abspath(override), base_name))

    # App directory (same logic as scan_for_epubs)
    if platform.system() == "Windows":
        if getattr(sys, "frozen", False):
            app_dir = os.path.dirname(sys.executable)
        else:
            app_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        app_dir = os.getcwd()
    output_dirs_to_check.append(os.path.join(app_dir, base_name))

    for out_dir in output_dirs_to_check:
        result = _scan_for_cover(out_dir)
        if result:
            return result

    return None


class _LibraryScannerThread(QThread):
    """Run scan_for_epubs() off the UI thread so the loading spinner animates.

    Retained for backward compatibility — external callers (e.g. android UI)
    still depend on the merged scan. The tabbed dialog uses
    :class:`_DualScannerThread` instead.
    """
    finished = Signal(list)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self._config = config or {}

    def run(self):
        try:
            results = scan_for_epubs(self._config)
        except Exception:
            logger.debug("Library scan failed: %s", traceback.format_exc())
            results = []
        self.finished.emit(results)


class _DualScannerThread(QThread):
    """Scan both library and output roots, partitioning by completion status.

    Emits ``(in_progress_list, completed_list)`` where:
      * in_progress_list — output folders with a progress file but no compiled EPUB yet.
      * completed_list — Library EPUBs + output folders whose compiled EPUB exists.

    Entries whose compiled EPUB already lives in the Library folder are
    deduped (library entry wins) so the same book can't appear twice.
    """
    finished = Signal(list, list)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self._config = config or {}

    def run(self):
        try:
            output_rows = scan_output_folders(self._config)
        except Exception:
            logger.debug("Output scan failed: %s", traceback.format_exc())
            output_rows = []
        try:
            library_rows = scan_library_completed(self._config)
        except Exception:
            logger.debug("Library scan failed: %s", traceback.format_exc())
            library_rows = []

        completed_from_output, in_progress = split_output_folders_by_status(output_rows)

        # Post-organize dedup: any output folder whose compiled EPUB was
        # moved into Library/Translated shows up in TWO scans:
        #   * ``scan_library_completed`` — the new Library/Translated file.
        #   * ``scan_output_folders``    — the owning folder (no compiled
        #     EPUB anymore but progress=100% still classifies it as
        #     ``completed``).
        # The origins registry tells us which output folders were the
        # source of a library-filed EPUB; we skip those ghost
        # folder-only rows so the user sees a single card per book.
        try:
            origins = _load_origins()
            trans_map = origins.get("translated", {}) or {}
        except Exception:
            trans_map = {}
        organized_folders: set[str] = set()
        for orig_path in trans_map.values():
            if not orig_path:
                continue
            folder = os.path.dirname(str(orig_path))
            if folder:
                organized_folders.add(
                    os.path.normcase(os.path.normpath(
                        os.path.abspath(folder)))
                )

        def _is_organized_ghost(row: dict) -> bool:
            """True when *row* is an output-folder scan result whose
            compiled EPUB has been organized into Library/Translated."""
            folder = row.get("output_folder") or ""
            if not folder:
                return False
            fk = os.path.normcase(os.path.normpath(
                os.path.abspath(folder)))
            return fk in organized_folders

        # Merge: library entries first (they're the curated shelf), then
        # output-folder completions that aren't already represented by path.
        seen_paths: set[str] = set()
        completed: list[dict] = []
        for r in library_rows:
            key = os.path.normcase(os.path.normpath(os.path.abspath(r["path"])))
            if key in seen_paths:
                continue
            seen_paths.add(key)
            completed.append(r)
        for r in completed_from_output:
            key = os.path.normcase(os.path.normpath(os.path.abspath(r["path"])))
            if key in seen_paths:
                continue
            if _is_organized_ghost(r):
                continue
            seen_paths.add(key)
            completed.append(r)
        completed.sort(key=lambda r: r["mtime"], reverse=True)

        # Apply the same ghost-filter to the In Progress tab so a
        # post-organize folder that somehow slips through the "completed"
        # classification (partial undo, mismatched progress file, etc.)
        # doesn't linger as a phantom in-progress card either.
        in_progress = [r for r in in_progress if not _is_organized_ghost(r)]

        self.finished.emit(in_progress, completed)


class _CoverLoader(QThread):
    finished = Signal(str, str)

    def __init__(self, file_path: str, file_type: str = "epub", config: dict | None = None,
                 original_path: str | None = None, raw_source_path: str | None = None,
                 parent=None):
        super().__init__(parent)
        self._file_path = file_path
        self._file_type = file_type
        self._config = config or {}
        self._original_path = original_path
        # For in-progress cards the output folder has no images yet, so the
        # thumbnail must be pulled directly from the resolved raw source EPUB.
        self._raw_source_path = raw_source_path or ""

    def run(self):
        if self._file_type == "epub":
            cover = _extract_cover(self._file_path)
        elif self._file_type == "in_progress":
            # For an in-progress card the "path" is the output folder itself.
            cover = None
            # Primary source: the resolved raw EPUB in Library/Raw (or
            # wherever source_epub.txt points). This is the only way to
            # produce a real thumbnail for Not Started cards, whose output
            # folder is still empty.
            if (self._raw_source_path
                    and self._raw_source_path.lower().endswith(".epub")
                    and os.path.isfile(self._raw_source_path)):
                cover = _extract_cover(self._raw_source_path)
            # Secondary: images the translator has produced in the output
            # folder so far (mid-translation or retranslation runs).
            if not cover:
                cover = _find_cover_in_dir(self._file_path)
            # Tertiary: compiled output EPUB (if any) for finished-but-not-
            # organized novels.
            if not cover:
                try:
                    for entry in os.scandir(self._file_path):
                        if (entry.is_file(follow_symlinks=False)
                                and entry.name.lower().endswith(".epub")):
                            cover = _extract_cover(entry.path)
                            if cover:
                                break
                except (PermissionError, OSError):
                    pass
        else:
            cover = _find_folder_cover(self._file_path, config=self._config,
                                       original_path=self._original_path)
        self.finished.emit(self._file_path, cover or "")


class _SelectableGrid(QWidget):
    """Grid container that supports click-drag rubber-band selection.

    Emits :attr:`rubber_band_selection` once the user finishes a drag.
    Plain clicks on empty space emit :attr:`empty_clicked` so the parent
    can clear the current selection. Clicks that land on a :class:`_BookCard`
    child are handled by the card itself (this widget never receives those).
    """
    # (list_of_books_inside_band, modifiers)
    rubber_band_selection = Signal(list, object)
    # (modifiers) — fired on a plain click with no drag
    empty_clicked = Signal(object)

    # Minimum pointer movement (Manhattan distance in px) before we start
    # showing the rubber band. Prevents a stray twitch on mouse press from
    # accidentally replacing the current multi-selection with nothing.
    _DRAG_THRESHOLD = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rubber_band = None
        self._drag_origin = None
        self._drag_modifiers = None
        self._drag_started = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_origin = event.pos()
            self._drag_modifiers = event.modifiers()
            self._drag_started = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_origin is not None and (event.buttons() & Qt.LeftButton):
            from PySide6.QtCore import QRect, QSize
            from PySide6.QtWidgets import QRubberBand
            delta = event.pos() - self._drag_origin
            distance = abs(delta.x()) + abs(delta.y())
            if not self._drag_started and distance > self._DRAG_THRESHOLD:
                self._drag_started = True
                if self._rubber_band is None:
                    self._rubber_band = QRubberBand(QRubberBand.Rectangle, self)
                self._rubber_band.setGeometry(QRect(self._drag_origin, QSize()))
                self._rubber_band.show()
            if self._drag_started and self._rubber_band is not None:
                rect = QRect(self._drag_origin, event.pos()).normalized()
                self._rubber_band.setGeometry(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        try:
            if self._drag_started and self._rubber_band is not None:
                rect = self._rubber_band.geometry()
                self._rubber_band.hide()
                books: list = []
                for child in self.children():
                    if isinstance(child, _BookCard):
                        if rect.intersects(child.geometry()):
                            books.append(child.book)
                self.rubber_band_selection.emit(
                    books, self._drag_modifiers or Qt.NoModifier)
            elif (event.button() == Qt.LeftButton
                    and self._drag_origin is not None):
                # Plain click on empty space (no drag) — let the parent
                # decide whether to clear the selection.
                self.empty_clicked.emit(self._drag_modifiers or Qt.NoModifier)
        finally:
            self._drag_origin = None
            self._drag_modifiers = None
            self._drag_started = False
        super().mouseReleaseEvent(event)


def _card_raw_title(book: dict) -> str:
    """Best-guess raw / source-language title for a library flash card.

    Resolution order:
      1. ``metadata.json`` explicit ``original_title`` / ``raw_title`` /
         ``source_title`` keys when the translator stored one.
      2. ``folder_name`` — for in-progress workspaces this equals the raw
         EPUB's basename (output folders are scaffolded from the source
         filename).
      3. Stem of ``raw_source_path`` when the scanner resolved one.
      4. Stem of ``original_path`` recorded in the origins registry (for
         Library-organized files that were moved from elsewhere).
      5. Fall back to the card's default ``name``.
    """
    md = book.get("metadata_json") or {}
    for key in ("original_title", "raw_title", "source_title"):
        val = md.get(key)
        if val:
            return str(val)
    fn = book.get("folder_name")
    if fn:
        return str(fn)
    for path_key in ("raw_source_path", "original_path"):
        p = book.get(path_key) or ""
        if p:
            return os.path.splitext(os.path.basename(p))[0]
    return str(book.get("name", ""))


class _BookCard(QFrame):
    clicked = Signal(dict)
    context_menu_requested = Signal(dict, object)
    # Emitted on left-click so the parent dialog can manage multi-selection
    # (Ctrl-click toggles, plain click replaces). Payload: (book, modifiers).
    select_requested = Signal(dict, object)

    # Selectors target the widget by object name (rather than its Python
    # class name via a type selector) because Qt's metaobject className for
    # PySide6-subclassed QFrames isn't always reliable — an ID selector is
    # the one form guaranteed to match exactly this widget. The border is
    # kept at 2 px in both states so the content area doesn't reflow on
    # selection (which previously caused visually stuck hover rendering).
    _BASE_STYLE = (
        "QFrame#bookCard { background: #1e1e2e; border: 2px solid #2a2a3e;"
        " border-radius: 6px; }"
        "QFrame#bookCard:hover { border: 2px solid #6c63ff; background: #252540; }"
    )
    _SELECTED_STYLE = (
        "QFrame#bookCard { background: #2a2d5a; border: 2px solid #a097ff;"
        " border-radius: 6px; }"
        "QFrame#bookCard:hover { border: 2px solid #c0b8ff; background: #343670; }"
    )

    def __init__(self, book: dict, preset: dict | None = None, parent=None,
                 show_raw_title: bool = False):
        super().__init__(parent)
        self.book = book
        p = preset or _SIZE_PRESETS[SIZE_NORMAL]
        self._card_w = p["card_w"]
        self._cover_h = p["cover_h"]
        self._has_cover = False
        self._selected = False
        self._show_raw_title = bool(show_raw_title)

        self.setObjectName("bookCard")
        self.setFixedWidth(self._card_w)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(self._BASE_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 6)
        layout.setSpacing(3)

        self.cover_label = QLabel()
        self.cover_label.setFixedSize(self._card_w - 8, self._cover_h)
        self.cover_label.setAlignment(Qt.AlignCenter)
        self.cover_label.setStyleSheet("background: #2a2a3e; border-radius: 4px; color: #555; font-size: 28pt;")
        icon_path = _find_halgakos_icon()
        if icon_path:
            self._set_fallback_icon(icon_path)
        else:
            self.cover_label.setText("📖")
        layout.addWidget(self.cover_label)

        # Title: try to render the full name at the preset font size; shrink
        # the font (down to ``title_min_size``) if it wraps past the max
        # height, and fall back to an ellipsis only as a last resort. See
        # :func:`_fit_title_text`. When the library's "Show raw titles"
        # toggle is on, :func:`_card_raw_title` picks the source-language
        # title instead of whatever ``book['name']`` resolved to.
        if self._show_raw_title:
            full_title = _card_raw_title(book) or book.get("name", "")
        else:
            full_title = book["name"]
        base_pt = _parse_pt(p.get("title_size", "9pt"))
        min_pt = _parse_pt(p.get("title_min_size", "6.5pt"))
        max_title_h = p.get("title_max_h", 36)
        title_lbl = QLabel()
        title_lbl.setWordWrap(True)
        # Fixed (not max) height so every card’s title area is the same
        # size — short titles leave blank space below rather than making
        # the card itself shorter.
        title_lbl.setFixedHeight(max_title_h)
        title_lbl.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        title_lbl.setAttribute(Qt.WA_TranslucentBackground)
        title_lbl.setToolTip(full_title)
        fitted_text, fitted_pt = _fit_title_text(
            full_title,
            avail_width=self._card_w - 10,
            max_height=max_title_h,
            base_pt=base_pt,
            base_font=title_lbl.font(),
            min_pt=min_pt,
        )
        title_lbl.setText(fitted_text)
        title_lbl.setStyleSheet(
            f"color: #e0e0e0; font-size: {fitted_pt:g}pt; "
            "font-weight: bold; background: transparent;"
        )
        layout.addWidget(title_lbl)

        # Size + file type badge on same row. For in-progress folders the
        # badge reflects the *source* workspace kind (epub/txt/pdf/image)
        # so users can distinguish a TXT translation's progress file from
        # an EPUB's at a glance.
        file_type = book.get("type", "epub")
        type_info = {
            "epub":  ("\U0001f4d5EPUB",  "#6c63ff"),
            "pdf":   ("\U0001f4c4PDF",   "#e74c3c"),
            "txt":   ("\U0001f4d7TXT",   "#2ecc71"),
            "html":  ("\U0001f310HTML",  "#3498db"),
            "image": ("\U0001f5bc\ufe0fIMG", "#f39c12"),
            "in_progress": ("\U0001f4c1FOLDER", "#ffd166"),
        }
        if file_type == "in_progress":
            kind = (book.get("workspace_kind") or "other").lower()
            badge_text, badge_color = type_info.get(kind, type_info["in_progress"])
        else:
            badge_text, badge_color = type_info.get(file_type, type_info["epub"])
        size_mb = book["size"] / (1024 * 1024)
        size_str = f"{size_mb:.1f} MB" if size_mb >= 1 else f"{book['size'] / 1024:.0f} KB"
        info_row = QHBoxLayout()
        info_row.setContentsMargins(0, 0, 0, 0)
        info_row.setSpacing(4)
        size_lbl = QLabel(size_str)
        size_lbl.setAttribute(Qt.WA_TranslucentBackground)
        size_lbl.setStyleSheet("color: #888; font-size: 7.5pt; background: transparent;")
        info_row.addWidget(size_lbl)
        badge_lbl = QLabel(badge_text)
        badge_lbl.setAttribute(Qt.WA_TranslucentBackground)
        badge_lbl.setStyleSheet(f"color: {badge_color}; font-size: 7pt; font-weight: bold; background: transparent;")
        info_row.addWidget(badge_lbl)
        # Multi-output warning: the scanner detected more than one
        # compiled artefact in this book's output folder (e.g. two
        # ``.epub`` files from successive recompiles, or a stale
        # ``*_translated.html`` sitting next to a fresh ``.epub``).
        # Surface it as a small ⚠ badge so the user can clean up —
        # the primary compiled output is still chosen deterministically
        # (EPUB > PDF > TXT > HTML) but this warns them that the
        # deterministic pick may not be what they expected.
        conflicts = list(book.get("compiled_conflicts") or [])
        if conflicts:
            conflict_lbl = QLabel(f"\u26a0 +{len(conflicts)}")
            conflict_lbl.setAttribute(Qt.WA_TranslucentBackground)
            conflict_lbl.setStyleSheet(
                "color: #ffb347; font-size: 7pt; font-weight: bold; "
                "background: rgba(255, 179, 71, 0.15); "
                "border: 1px solid #ffb347; border-radius: 3px; "
                "padding: 0 4px; margin-left: 2px;"
            )
            # Lightweight human-readable summary for the tooltip:
            # "Extra compiled files in this folder:\n  • foo.epub (EPUB)".
            tip_lines = ["Extra compiled files in this folder:"]
            for name, kind in conflicts[:8]:
                tip_lines.append(f"  \u2022 {name} ({kind.upper()})")
            if len(conflicts) > 8:
                tip_lines.append(f"  \u2026 and {len(conflicts) - 8} more.")
            tip_lines.append(
                "\nThe card displays only the primary output (EPUB > PDF > "
                "TXT > HTML); delete the stale artefacts to avoid "
                "confusion on the next scan."
            )
            conflict_lbl.setToolTip("\n".join(tip_lines))
            info_row.addWidget(conflict_lbl)
        info_row.addStretch()
        layout.addLayout(info_row)

        # In-progress indicator: small status pill + overlay ribbon on the cover
        has_progress_row = False
        if book.get("is_in_progress"):
            total = int(book.get("total_chapters", 0) or 0)
            done = int(book.get("completed_chapters", 0) or 0)
            state = book.get("translation_state") or (
                "in_progress" if total else "not_started"
            )
            # 100% translated = no longer in progress. Such cards render on
            # the Completed tab (see :func:`split_output_folders_by_status`)
            # without any ribbon / pill so they look like regular completed
            # entries even though the user hasn't compiled an EPUB yet.
            if state == "completed":
                pass
            else:
                has_progress_row = True
                pct = int(round((done / total) * 100)) if total else 0
                progress_row = QHBoxLayout()
                progress_row.setContentsMargins(0, 0, 0, 0)
                progress_row.setSpacing(4)
                if state == "not_started":
                    pill = QLabel("\U0001f195 Not started")
                    pill.setToolTip("Imported into Library/Raw, translation not started yet.")
                    pill.setStyleSheet(
                        "color: #8ab4d0; background: rgba(138, 180, 208, 0.15); "
                        "border: 1px solid #8ab4d0; border-radius: 3px; "
                        "font-size: 7pt; font-weight: bold; padding: 1px 5px;"
                    )
                    progress_row.addWidget(pill)
                    ribbon_text = "NOT STARTED"
                    ribbon_bg = "rgba(138, 180, 208, 0.92)"
                else:
                    pill = QLabel(f"\u23f3 {done}/{total}" if total else "\u23f3 In progress")
                    pill.setToolTip(
                        f"Translation in progress \u2014 {pct}% ({done}/{total} chapters)"
                    )
                    pill.setStyleSheet(
                        "color: #ffd166; background: rgba(108, 99, 255, 0.18); "
                        "border: 1px solid #6c63ff; border-radius: 3px; "
                        "font-size: 7pt; font-weight: bold; padding: 1px 5px;"
                    )
                    progress_row.addWidget(pill)
                    if total:
                        pct_lbl = QLabel(f"{pct}%")
                        pct_lbl.setAttribute(Qt.WA_TranslucentBackground)
                        pct_lbl.setStyleSheet("color: #8ab4d0; font-size: 7pt; font-weight: bold; background: transparent;")
                        progress_row.addWidget(pct_lbl)
                    ribbon_text = "IN PROGRESS"
                    ribbon_bg = "rgba(108, 99, 255, 0.92)"
                progress_row.addStretch()
                layout.addLayout(progress_row)

                # Corner ribbon on the cover label (absolutely positioned child)
                self._progress_ribbon = QLabel(ribbon_text, self.cover_label)
                self._progress_ribbon.setStyleSheet(
                    f"color: #fff; background: {ribbon_bg}; "
                    "font-size: 6.5pt; font-weight: bold; padding: 1px 5px; "
                    "border-bottom-right-radius: 3px;"
                )
                self._progress_ribbon.move(0, 0)
                self._progress_ribbon.show()

        # Trailing stretch + fixed card height so every card within the
        # same tab occupies a uniform footprint. Completed cards skip the
        # progress-pill reservation so they don't render with a dead band
        # of empty space at the bottom where the pill would have been on
        # the In Progress tab.
        layout.addStretch()
        reserved_h = 32 + p.get("spacing", 4)
        if has_progress_row:
            reserved_h += 22  # pill row height + extra inter-widget spacing
        self.setFixedHeight(self._cover_h + max_title_h + reserved_h)

    def set_selected(self, selected: bool):
        """Toggle the card's "selected" visual state.

        Used by :class:`EpubLibraryDialog` to render multi-selection for
        batch actions like "Load N for translation". Stylesheet-only
        change — no layout recomputation, so this is cheap.
        """
        new_value = bool(selected)
        if new_value == self._selected:
            return
        self._selected = new_value
        self.setStyleSheet(self._SELECTED_STYLE if new_value else self._BASE_STYLE)
        # Force an immediate repaint so the stylesheet swap is painted on
        # the current tick rather than waiting for the next synthetic event.
        self.update()

    @property
    def selected(self) -> bool:
        return self._selected

    def _set_fallback_icon(self, icon_path: str):
        try:
            pm = QPixmap(icon_path)
            if not pm.isNull():
                target_w = int(self._cover_h * 0.5)
                target_h = int(self._cover_h * 0.6)
                scaled = pm.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.cover_label.setPixmap(scaled)
                self.cover_label.setText("")
        except Exception:
            logger.debug("Fallback icon failed: %s", traceback.format_exc())
            self.cover_label.setText("📖")

    def set_cover(self, image_path: str):
        try:
            pm = QPixmap(image_path)
            if not pm.isNull():
                scaled = pm.scaled(self._card_w - 8, self._cover_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.cover_label.setPixmap(scaled)
                self.cover_label.setText("")
                self._has_cover = True
        except Exception:
            logger.debug("Set cover failed: %s", traceback.format_exc())

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.book)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Forward to the parent dialog so it can update multi-selection
            # state. Ctrl/Shift modifiers extend or toggle the selection;
            # plain click replaces it with just this card. We MUST accept
            # the event so it doesn't propagate to the enclosing
            # :class:`_SelectableGrid`, which would otherwise record a
            # drag origin and fire ``empty_clicked`` on the matching
            # release — clearing the selection we just set.
            self.select_requested.emit(self.book, event.modifiers())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        # Consume the release for the same reason as mousePressEvent: if
        # the event reaches :class:`_SelectableGrid` with no drag movement
        # it triggers the "empty space click" path and wipes the selection.
        if event.button() == Qt.LeftButton:
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        self.context_menu_requested.emit(self.book, event.globalPos())
        event.accept()


# ---------------------------------------------------------------------------
# Library Dialog
# ---------------------------------------------------------------------------

class EpubLibraryDialog(QDialog):
    # Emitted when the user imports a new EPUB from the "In Progress" tab.
    # Parents (e.g. TranslatorGUI) can connect to set it as the input file.
    import_epub_requested = Signal(str)
    # Emitted when the user triggers a multi-card "Load N for translation"
    # action. Payload is a list of absolute paths.
    import_epubs_requested = Signal(list)
    # Emitted after an Organize / Undo Move operation relocates files on
    # disk. Payload is a list of ``(old_abs_path, new_abs_path)`` tuples.
    # Parents (e.g. TranslatorGUI) use it to update any stale paths in
    # their own state — selected_files, entry_epub text, etc. — so the
    # user doesn't end up clicking Run on a path that no longer exists
    # after the raw was moved into Library/Raw.
    files_reorganized = Signal(list)

    def __init__(self, config: dict | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("\U0001f4da Glossarion Library")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
        # Ratio-based sizing relative to screen
        screen = self.screen()
        if screen:
            avail = screen.availableGeometry()
            self.resize(int(avail.width() * 0.55), int(avail.height() * 0.65))
            self.setMinimumSize(int(avail.width() * 0.35), int(avail.height() * 0.4))
        else:
            self.resize(900, 650)
            self.setMinimumSize(500, 400)
        self._config = config or {}
        # Per-tab data. "In Progress" comes from the output root (strict
        # OUTPUT_DIRECTORY override); "Completed" comes from the Library folder.
        self._in_progress_books: list[dict] = []
        self._completed_books: list[dict] = []
        self._ip_cards: list[_BookCard] = []
        self._comp_cards: list[_BookCard] = []
        # Multi-selection state, keyed by ``book['path']``. One set per tab
        # so switching tabs doesn't clobber the other tab's selection. The
        # sets are consulted every time ``_populate_grid_common`` rebuilds
        # the cards (which happens on scan / auto-refresh / sort / filter)
        # so selection survives refreshes as long as the book path is
        # still present in the scan result.
        self._selected_paths_ip: set[str] = set()
        self._selected_paths_comp: set[str] = set()
        self._cover_threads: list[_CoverLoader] = []
        # Restore persisted settings
        self._sort_mode = self._config.get('epub_library_sort', SORT_DATE)
        self._card_size = self._config.get('epub_library_card_size', SIZE_COMPACT)
        self._current_tab = self._config.get('epub_library_tab', 0)
        # When True, every flash card renders the raw source-language title
        # instead of the (possibly translated) ``book['name']``. See
        # :func:`_card_raw_title` for the resolution rules.
        self._show_raw_titles = bool(
            self._config.get('epub_library_show_raw_titles', False)
        )
        self._scanner_thread: _DualScannerThread | None = None
        # Enable drag-and-drop of EPUB / TXT / PDF / HTML files onto the
        # dialog. Drops are routed through the same import pipeline as the
        # "Import EPUB" button (see :meth:`_import_paths_into_library`).
        self.setAcceptDrops(True)
        self._setup_ui()
        # Flip the dialog into its loading state BEFORE the caller
        # shows it. Previously ``_show_loading`` was called by the
        # deferred :meth:`_load_books` tick, which meant the dialog's
        # first paint rendered empty tabs (no spinner, no cards) for
        # one event-loop iteration — visible to the user as a blank
        # window for a noticeable fraction of a second. Doing the
        # widget swap here guarantees the very first paint already
        # shows the Halgakos spinner + "Scanning library…" strip.
        self._show_loading()
        # Defer the filesystem scan so the dialog paints the loading
        # state immediately and the scanner thread only kicks off on
        # the next event-loop tick (keeps the show-flow snappy).
        QTimer.singleShot(0, self._load_books)
        # Auto-refresh library every 2 seconds (start after initial load)
        self._auto_refresh_timer = QTimer(self)
        self._auto_refresh_timer.setInterval(2000)
        self._auto_refresh_timer.timeout.connect(self._auto_refresh)
        QTimer.singleShot(2500, self._auto_refresh_timer.start)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        elif event.modifiers() & Qt.ControlModifier:
            sizes = _ALL_SIZES
            idx = sizes.index(self._card_size) if self._card_size in sizes else 0
            if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
                if idx < len(sizes) - 1:
                    self._set_card_size(sizes[idx + 1])
            elif event.key() == Qt.Key_Minus:
                if idx > 0:
                    self._set_card_size(sizes[idx - 1])
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event):
        """Ctrl+Wheel to zoom card size."""
        if event.modifiers() & Qt.ControlModifier:
            sizes = _ALL_SIZES
            idx = sizes.index(self._card_size) if self._card_size in sizes else 0
            if event.angleDelta().y() > 0 and idx < len(sizes) - 1:
                self._set_card_size(sizes[idx + 1])
            elif event.angleDelta().y() < 0 and idx > 0:
                self._set_card_size(sizes[idx - 1])
            event.accept()
        else:
            super().wheelEvent(event)

    def _make_sort_btn(self, text, tooltip, mode):
        btn = QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setFixedHeight(26)
        btn.setMinimumWidth(40)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setCheckable(True)
        btn.setChecked(mode == self._sort_mode)
        btn.setStyleSheet("""
            QPushButton { background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #b0b0c0; font-size: 8.5pt; font-weight: bold; padding: 2px 8px; }
            QPushButton:hover { background: #3a3a5e; color: #e0e0e0; }
            QPushButton:checked { background: #6c63ff; border-color: #7c73ff; color: #fff; }
        """)
        btn.clicked.connect(lambda: self._set_sort(mode))
        return btn

    def _make_size_btn(self, text, tooltip, size_key):
        btn = QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setFixedSize(28, 26)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setCheckable(True)
        btn.setChecked(size_key == self._card_size)
        btn.setStyleSheet("""
            QPushButton { background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #b0b0c0; font-size: 9pt; font-weight: bold; padding: 0; }
            QPushButton:hover { background: #3a3a5e; color: #e0e0e0; }
            QPushButton:checked { background: #17a2b8; border-color: #20b2cc; color: #fff; }
        """)
        btn.clicked.connect(lambda: self._set_card_size(size_key))
        return btn

    def _setup_ui(self):
        from PySide6.QtWidgets import QTabWidget
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(8)
        # Halgakos icon next to title (HiDPI-aware)
        icon_path = _find_halgakos_icon()
        if icon_path:
            icon_lbl = QLabel()
            pm = QPixmap(icon_path)
            if not pm.isNull():
                dpr = self.devicePixelRatio() or 1.0
                logical_size = 28
                raw = int(logical_size * dpr)
                scaled = pm.scaled(raw, raw, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                scaled.setDevicePixelRatio(dpr)
                icon_lbl.setPixmap(scaled)
                icon_lbl.setFixedSize(logical_size + 2, logical_size + 2)
                header.addWidget(icon_lbl)
        title = QLabel("\U0001f4da  Glossarion Library")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #e0e0e0;")
        header.addWidget(title)
        header.addStretch()

        self._search = QLineEdit()
        self._search.setPlaceholderText("🔍  Filter…")
        self._search.setFixedWidth(200)
        self._search.setStyleSheet(
            "background: #1e1e2e; border: 1px solid #3a3a5e; border-radius: 6px; "
            "padding: 4px 10px; color: #e0e0e0; font-size: 9.5pt;"
        )
        self._search.textChanged.connect(self._apply_filter)
        header.addWidget(self._search)

        refresh_btn = QPushButton("\U0001f504  Refresh")
        refresh_btn.setToolTip("Refresh library")
        refresh_btn.setFixedHeight(28)
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; font-size: 9pt; "
            "color: #888; padding: 2px 8px; }"
            "QPushButton:hover { color: #e0e0e0; }")
        refresh_btn.clicked.connect(self._load_books)
        header.addWidget(refresh_btn)

        root.addLayout(header)
        root.addSpacing(6)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(4)
        sort_lbl = QLabel("Sort:")
        sort_lbl.setStyleSheet("color: #888; font-size: 8.5pt;")
        toolbar.addWidget(sort_lbl)
        self._sort_btns = {}
        for text, tip, mode in [
            ("🕐 Date", "Sort by date (newest first)", SORT_DATE),
            ("A-Z", "Sort by name (alphabetical)", SORT_NAME),
            ("📏 Size", "Sort by file size (largest first)", SORT_SIZE),
        ]:
            btn = self._make_sort_btn(text, tip, mode)
            self._sort_btns[mode] = btn
            toolbar.addWidget(btn)
        toolbar.addSpacing(16)
        size_lbl = QLabel("View:")
        size_lbl.setStyleSheet("color: #888; font-size: 8.5pt;")
        toolbar.addWidget(size_lbl)
        self._size_btns = {}
        for text, tip, key in [
            ("S", "Compact thumbnails", SIZE_COMPACT),
            ("M", "Normal thumbnails", SIZE_NORMAL),
            ("L", "Large thumbnails", SIZE_LARGE),
            ("XL", "Extra large thumbnails", SIZE_XL),
            ("2XL", "2XL thumbnails", SIZE_2XL),
            ("3XL", "3XL thumbnails", SIZE_3XL),
            ("4XL", "4XL thumbnails", SIZE_4XL),
            ("5XL", "5XL thumbnails", SIZE_5XL),
            ("6XL", "6XL thumbnails", SIZE_6XL),
        ]:
            btn = self._make_size_btn(text, tip, key)
            self._size_btns[key] = btn
            toolbar.addWidget(btn)
        toolbar.addSpacing(16)
        # "Show raw titles" toggle: when checked, every flash card displays
        # the raw source-language title instead of the translated one. Same
        # visual language as the sort buttons so it reads as a persistent
        # filter rather than a one-shot action.
        self._raw_titles_btn = QPushButton("\U0001f524  Raw titles")
        self._raw_titles_btn.setToolTip(
            "Show the raw source-language title on every card instead of \n"
            "the translated / compiled-EPUB title. Useful for finding a \n"
            "book by its original name."
        )
        self._raw_titles_btn.setFixedHeight(26)
        self._raw_titles_btn.setCursor(Qt.PointingHandCursor)
        self._raw_titles_btn.setCheckable(True)
        self._raw_titles_btn.setChecked(self._show_raw_titles)
        self._raw_titles_btn.setStyleSheet("""
            QPushButton { background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #b0b0c0; font-size: 8.5pt; font-weight: bold; padding: 2px 10px; }
            QPushButton:hover { background: #3a3a5e; color: #e0e0e0; }
            QPushButton:checked { background: #6c63ff; border-color: #7c73ff; color: #fff; }
        """)
        self._raw_titles_btn.toggled.connect(self._on_raw_titles_toggled)
        toolbar.addWidget(self._raw_titles_btn)
        toolbar.addStretch()
        self._count_label = QLabel("")
        self._count_label.setStyleSheet("color: #888; font-size: 8.5pt;")
        toolbar.addWidget(self._count_label)
        root.addLayout(toolbar)

        # Per-tab action bar: each tab gets its own extra button row with
        # context-specific actions. Organize / Undo are duplicated across
        # both tabs so the user doesn't have to hop over to In Progress to
        # organize a Library entry. Every button label carries a live
        # counter that reflects how many files the action would act on
        # (see :meth:`_update_organize_counts`).
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #2a2a3e; border-radius: 6px;
                                background: #12121e; top: -1px; }
            QTabBar::tab { background: #1e1e2e; color: #b0b0c0;
                            border: 1px solid #2a2a3e; border-bottom: none;
                            border-top-left-radius: 6px; border-top-right-radius: 6px;
                            padding: 6px 16px; font-size: 9.5pt; font-weight: bold;
                            margin-right: 2px; min-width: 110px; }
            QTabBar::tab:selected { background: #12121e; color: #e0e0e0;
                                     border-color: #2a2a3e; }
            QTabBar::tab:hover:!selected { background: #252540; color: #e0e0e0; }
        """)

        # --- In Progress tab ---
        self._ip_tab = QWidget()
        self._ip_tab.setStyleSheet("background: #12121e;")
        ip_layout = QVBoxLayout(self._ip_tab)
        ip_layout.setContentsMargins(6, 8, 6, 6)
        ip_layout.setSpacing(6)
        ip_action_row = QHBoxLayout()
        ip_action_row.setSpacing(6)
        self._ip_count_label = QLabel("")
        self._ip_count_label.setStyleSheet("color: #888; font-size: 8.5pt;")
        ip_action_row.addWidget(self._ip_count_label)
        ip_action_row.addStretch()
        self._ip_organize_btn = self._make_organize_button(kind="ip")
        ip_action_row.addWidget(self._ip_organize_btn)
        self._ip_undo_btn = self._make_undo_button(kind="ip")
        ip_action_row.addWidget(self._ip_undo_btn)
        self._import_btn = QPushButton("\u2795  Import EPUB")
        self._import_btn.setCursor(Qt.PointingHandCursor)
        self._import_btn.setToolTip(
            "Copy an EPUB into Library/Raw and create a new output folder "
            "for it so you can translate it later."
        )
        self._import_btn.setStyleSheet(
            "QPushButton { background: #6c63ff; color: white; border-radius: 4px; "
            "padding: 6px 14px; font-size: 9pt; font-weight: bold; border: none; }"
            "QPushButton:hover { background: #8078ff; }")
        self._import_btn.clicked.connect(self._import_epub)
        ip_action_row.addWidget(self._import_btn)
        ip_layout.addLayout(ip_action_row)
        self._ip_scroll = QScrollArea()
        self._ip_scroll.setWidgetResizable(True)
        self._ip_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QScrollBar:vertical { width: 8px; background: #1a1a2e; }"
            "QScrollBar::handle:vertical { background: #3a3a5e; border-radius: 4px; }")
        # Use :class:`_SelectableGrid` so the user can click-drag a
        # rubber-band rectangle across multiple cards to select them.
        self._ip_grid_container = _SelectableGrid()
        self._ip_grid_container.rubber_band_selection.connect(
            self._on_rubber_band_selection)
        self._ip_grid_container.empty_clicked.connect(self._on_empty_area_clicked)
        self._ip_grid_layout = QGridLayout(self._ip_grid_container)
        self._ip_grid_layout.setContentsMargins(1, 1, 1, 1)
        self._ip_scroll.setWidget(self._ip_grid_container)
        ip_layout.addWidget(self._ip_scroll, 1)
        self._ip_empty_label = QLabel(
            "No translations in progress.\nUse \u201cImport EPUB\u201d to start one.")
        self._ip_empty_label.setAlignment(Qt.AlignCenter)
        self._ip_empty_label.setStyleSheet("color: #555; font-size: 12pt; padding: 40px;")
        self._ip_empty_label.hide()
        ip_layout.addWidget(self._ip_empty_label)

        # --- Completed (Library) tab ---
        self._comp_tab = QWidget()
        self._comp_tab.setStyleSheet("background: #12121e;")
        comp_layout = QVBoxLayout(self._comp_tab)
        comp_layout.setContentsMargins(6, 8, 6, 6)
        comp_layout.setSpacing(6)
        comp_action_row = QHBoxLayout()
        comp_action_row.setSpacing(6)
        self._comp_count_label = QLabel("")
        self._comp_count_label.setStyleSheet("color: #888; font-size: 8.5pt;")
        comp_action_row.addWidget(self._comp_count_label)
        comp_action_row.addStretch()
        # Completed tab gets its own copy of the Organize + Undo buttons so
        # the user can move a compiled EPUB into the Library without
        # having to bounce back to In Progress. Wired to the same
        # handlers — the actions operate on both tabs' books regardless
        # of which button was clicked; only the *label counters* are
        # tab-specific.
        self._comp_organize_btn = self._make_organize_button(kind="comp")
        comp_action_row.addWidget(self._comp_organize_btn)
        self._comp_undo_btn = self._make_undo_button(kind="comp")
        comp_action_row.addWidget(self._comp_undo_btn)
        self._open_library_btn = QPushButton("\U0001f4c1  Open Library Folder")
        self._open_library_btn.setCursor(Qt.PointingHandCursor)
        self._open_library_btn.setToolTip(f"Open {get_library_dir()} in the system file explorer.")
        self._open_library_btn.setStyleSheet(
            "QPushButton { background: #3a5a7a; color: white; border-radius: 4px; "
            "padding: 6px 14px; font-size: 9pt; font-weight: bold; border: none; }"
            "QPushButton:hover { background: #4a6a8a; }")
        self._open_library_btn.clicked.connect(self._open_library_folder)
        comp_action_row.addWidget(self._open_library_btn)
        comp_layout.addLayout(comp_action_row)
        self._comp_scroll = QScrollArea()
        self._comp_scroll.setWidgetResizable(True)
        self._comp_scroll.setStyleSheet(self._ip_scroll.styleSheet())
        self._comp_grid_container = _SelectableGrid()
        self._comp_grid_container.rubber_band_selection.connect(
            self._on_rubber_band_selection)
        self._comp_grid_container.empty_clicked.connect(self._on_empty_area_clicked)
        self._comp_grid_layout = QGridLayout(self._comp_grid_container)
        self._comp_grid_layout.setContentsMargins(1, 1, 1, 1)
        self._comp_scroll.setWidget(self._comp_grid_container)
        comp_layout.addWidget(self._comp_scroll, 1)
        self._comp_empty_label = QLabel(
            "Your Library is empty.\n\n"
            "Drop finished .epub files into the Library folder\n"
            "(or use \u201cOpen Library Folder\u201d above) to see them here.")
        self._comp_empty_label.setAlignment(Qt.AlignCenter)
        self._comp_empty_label.setStyleSheet("color: #555; font-size: 12pt; padding: 40px;")
        self._comp_empty_label.hide()
        comp_layout.addWidget(self._comp_empty_label)

        self._tabs.addTab(self._ip_tab, "\u23f3  In Progress")
        self._tabs.addTab(self._comp_tab, "\u2705  Completed")
        try:
            self._tabs.setCurrentIndex(int(self._current_tab) or 0)
        except (TypeError, ValueError):
            self._tabs.setCurrentIndex(0)
        self._tabs.currentChanged.connect(self._on_tab_changed)
        root.addWidget(self._tabs, 1)

        # ── Loading overlay (shared spinner shown over the current tab) ──
        self._loading_widget = QWidget()
        loading_layout = QVBoxLayout(self._loading_widget)
        loading_layout.setAlignment(Qt.AlignCenter)
        loading_layout.setContentsMargins(0, 20, 0, 20)
        loading_layout.setSpacing(8)
        self._spin_label = QLabel()
        icon_path = _find_halgakos_icon()
        self._spin_pixmap = None
        if icon_path:
            pm = QPixmap(icon_path)
            if not pm.isNull():
                self._spin_pixmap = pm.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self._spin_label.setPixmap(self._spin_pixmap)
        if not self._spin_pixmap:
            self._spin_label.setText("\U0001f4da")
            self._spin_label.setStyleSheet("font-size: 32pt; color: #e0e0e0;")
        self._spin_label.setAlignment(Qt.AlignCenter)
        self._spin_label.setFixedSize(72, 72)
        loading_layout.addWidget(self._spin_label, 0, Qt.AlignCenter)
        self._spin_angle = 0
        self._spin_timer = QTimer(self)
        self._spin_timer.setInterval(25)  # ~40 fps
        # Wrap in a lambda so PySide6 routes the call directly instead
        # of going through Qt's meta-object slot-lookup — certain
        # PySide6 builds raise ``AttributeError: Slot 'EpubLibraryDialog::
        # _rotate_spinner()' not found`` on ``QTimer.timeout`` dispatch
        # even when the method exists and is ``@Slot()``-decorated.
        self._spin_timer.timeout.connect(lambda: self._rotate_spinner())
        loading_text = QLabel("Scanning library\u2026")
        loading_text.setAlignment(Qt.AlignCenter)
        loading_text.setStyleSheet("color: #888; font-size: 11pt; padding-top: 4px;")
        loading_layout.addWidget(loading_text, 0, Qt.AlignCenter)
        from PySide6.QtWidgets import QProgressBar
        self._loading_bar = QProgressBar()
        self._loading_bar.setRange(0, 0)  # indeterminate
        self._loading_bar.setFixedWidth(220)
        self._loading_bar.setFixedHeight(6)
        self._loading_bar.setTextVisible(False)
        self._loading_bar.setStyleSheet("""
            QProgressBar { background: #2a2a2a; border: none; border-radius: 3px; }
            QProgressBar::chunk { background: #6c63ff; border-radius: 3px; }
        """)
        loading_layout.addWidget(self._loading_bar, 0, Qt.AlignCenter)
        self._loading_widget.setStyleSheet("background: transparent;")
        self._loading_widget.hide()
        root.addWidget(self._loading_widget, 1)

        self.setStyleSheet("QDialog { background: #12121e; }")

        # Drag-over overlay: semi-transparent purple panel with a centered
        # "Drop to import" message. Shown in :meth:`dragEnterEvent` and
        # hidden in :meth:`dragLeaveEvent` / :meth:`dropEvent`. It's a
        # plain child widget positioned manually so it covers the whole
        # dialog without participating in the root layout.
        self._drop_overlay = QLabel(self)
        self._drop_overlay.setObjectName("drop-overlay")
        self._drop_overlay.setAlignment(Qt.AlignCenter)
        self._drop_overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._drop_overlay.setText(
            "\U0001f4e5\n\nDrop files to import into your Library\n\n"
            "EPUB \u00b7 PDF \u00b7 TXT \u00b7 HTML"
        )
        self._drop_overlay.setStyleSheet(
            "QLabel#drop-overlay {"
            " background: rgba(108, 99, 255, 0.22);"
            " color: #e8e4ff;"
            " border: 3px dashed #8078ff;"
            " border-radius: 12px;"
            " font-size: 14pt;"
            " font-weight: bold;"
            "}"
        )
        self._drop_overlay.hide()

        # Toast widget: animated non-modal status line that appears at the
        # bottom of the dialog. Used instead of QMessageBox for drag-drop
        # imports so the flow stays click-free.
        self._toast = QLabel(self)
        self._toast.setObjectName("toast")
        self._toast.setAlignment(Qt.AlignCenter)
        self._toast.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._toast.setStyleSheet(
            "QLabel#toast {"
            " background: rgba(42, 45, 90, 0.96);"
            " color: #e8e4ff;"
            " border: 1px solid #8078ff;"
            " border-radius: 10px;"
            " padding: 10px 18px;"
            " font-size: 10pt;"
            " font-weight: bold;"
            "}"
        )
        from PySide6.QtWidgets import QGraphicsOpacityEffect
        self._toast_opacity = QGraphicsOpacityEffect(self._toast)
        self._toast_opacity.setOpacity(0.0)
        self._toast.setGraphicsEffect(self._toast_opacity)
        self._toast.hide()
        self._toast_hide_timer = QTimer(self)
        self._toast_hide_timer.setSingleShot(True)
        self._toast_hide_timer.timeout.connect(self._fade_out_toast)
        self._toast_anim = None  # current QPropertyAnimation (if any)

    # -- Tab / scan / render helpers ----------------------------------------

    @Slot()
    def _rotate_spinner(self):
        """Rotate the Halgakos icon by 15° each tick (matches reader).

        Decorated with ``@Slot()`` so PySide6 registers it in the Qt
        meta-object system. Without the decorator, ``QTimer.timeout``
        can fail to resolve the slot by name at invocation time
        (``AttributeError: Slot 'EpubLibraryDialog::_rotate_spinner()'
        not found``).
        """
        if self._spin_pixmap:
            self._spin_angle = (self._spin_angle + 15) % 360
            t = QTransform().rotate(self._spin_angle)
            rotated = self._spin_pixmap.transformed(t, Qt.FastTransformation)
            self._spin_label.setPixmap(rotated)

    def _show_loading(self):
        """Show the spinner overlay and hide both tab grids."""
        self._spin_angle = 0
        self._spin_timer.start()
        self._tabs.hide()
        self._ip_empty_label.hide()
        self._comp_empty_label.hide()
        self._loading_widget.show()

    def _hide_loading(self):
        """Hide the spinner overlay and restore the tab widget."""
        self._spin_timer.stop()
        self._loading_widget.hide()
        self._tabs.show()

    def _set_sort(self, mode):
        self._sort_mode = mode
        for k, btn in self._sort_btns.items():
            btn.setChecked(k == mode)
        self._refresh_view()

    def _set_card_size(self, size_key):
        self._card_size = size_key
        for k, btn in self._size_btns.items():
            btn.setChecked(k == size_key)
        self._refresh_view()

    def _on_raw_titles_toggled(self, checked: bool):
        """Flip every card between raw and translated title rendering."""
        new_value = bool(checked)
        if new_value == self._show_raw_titles:
            return
        self._show_raw_titles = new_value
        try:
            self._config["epub_library_show_raw_titles"] = self._show_raw_titles
        except Exception:
            pass
        # Cheapest reliable path: rebuild the cards so each :class:`_BookCard`
        # picks the title through its constructor. Cards are lightweight and
        # we already do this on sort / size changes.
        self._refresh_view()

    def _sorted_books(self, books):
        if self._sort_mode == SORT_NAME:
            return sorted(books, key=lambda b: b["name"].lower())
        elif self._sort_mode == SORT_SIZE:
            return sorted(books, key=lambda b: b["size"], reverse=True)
        return sorted(books, key=lambda b: b["mtime"], reverse=True)

    def _on_tab_changed(self, index: int):
        self._current_tab = index
        self._config["epub_library_tab"] = index

    # -- Actions ------------------------------------------------------------

    def _open_library_folder(self):
        """Open the Glossarion Library folder in the system file explorer."""
        lib = get_library_dir()
        try:
            os.makedirs(lib, exist_ok=True)
        except OSError:
            pass
        _open_folder_in_explorer(lib)

    # -- Organize / Undo button factory + counters --------------------------

    def _make_organize_button(self, kind: str) -> QPushButton:
        """Create an "Organize N into Library" button for the given tab kind.

        *kind* is either ``"ip"`` (In Progress — counts raw sources that
        can move into ``Library/Raw``) or ``"comp"`` (Completed — counts
        compiled EPUBs that can move into ``Library/Translated``). The
        label is refreshed by :meth:`_update_organize_counts` after every
        scan.
        """
        btn = QPushButton("\U0001f4e5  Organize (0)")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(
            "QPushButton { background: #3a5a7a; color: white; border-radius: 4px; "
            "padding: 6px 14px; font-size: 9pt; font-weight: bold; border: none; }"
            "QPushButton:hover { background: #4a6a8a; }"
            "QPushButton:disabled { background: #2a2a3e; color: #555; }"
        )
        btn.setToolTip(
            "Move every resolvable raw source into Library/Raw and every "
            "compiled EPUB into Library/Translated. Each move is recorded "
            "so it can be reversed by Undo Move."
        )
        btn.clicked.connect(self._organize_into_library)
        btn.setProperty("_organize_kind", kind)
        return btn

    def _make_undo_button(self, kind: str) -> QPushButton:
        """Create an "Undo N" button for the given tab kind.

        IP tab shows the count of raw moves that can be undone; Completed
        tab shows the translated bucket's count. Clicking either opens
        the same 3-way prompt (Raw / Translated / All).
        """
        btn = QPushButton("\u21a9  Undo (0)")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(
            "QPushButton { background: #6f42c1; color: white; border-radius: 4px; "
            "padding: 6px 14px; font-size: 9pt; font-weight: bold; border: none; }"
            "QPushButton:hover { background: #5a32a3; }"
            "QPushButton:disabled { background: #2a2a3e; color: #555; }"
        )
        btn.setToolTip(
            "Restore previously organized files back to their original "
            "locations. You'll be asked whether to undo Raw, Translated, "
            "or All."
        )
        btn.clicked.connect(self._undo_organize_prompt)
        btn.setProperty("_undo_kind", kind)
        return btn

    def _count_raw_movable(self) -> int:
        """Return how many raw sources aren't already in Library/Raw.

        Walks BOTH the In Progress and Completed book lists — a book at
        100 %% progress lives in ``_completed_books`` yet its raw source
        may still be sitting outside Library/Raw (e.g. the original EPUB
        that fed the translation). Library-tagged entries are skipped
        because their raw, if any, is already filed. Raw paths are
        deduplicated so one file counted against two scan rows doesn't
        double the counter.
        """
        raw_abs = os.path.normcase(os.path.normpath(
            os.path.abspath(get_library_raw_dir())))
        count = 0
        seen: set[str] = set()

        def _bump(book: dict) -> int:
            p = book.get("raw_source_path") or ""
            if not p or not os.path.isfile(p):
                return 0
            parent = os.path.normcase(os.path.normpath(
                os.path.abspath(os.path.dirname(p))))
            if parent == raw_abs:
                return 0
            key = os.path.normcase(os.path.normpath(os.path.abspath(p)))
            if key in seen:
                return 0
            seen.add(key)
            return 1

        for book in self._in_progress_books:
            count += _bump(book)
        for book in self._completed_books:
            if book.get("in_library"):
                continue
            count += _bump(book)
        return count

    def _count_trans_movable(self) -> int:
        """Return how many Completed compiled EPUBs aren't already in Library/Translated."""
        trans_abs = os.path.normcase(os.path.normpath(
            os.path.abspath(get_library_translated_dir())))
        count = 0
        for book in self._completed_books:
            if book.get("in_library"):
                continue
            p = book.get("path") or ""
            if not p or not os.path.isfile(p):
                continue
            if not p.lower().endswith(".epub"):
                continue
            parent = os.path.normcase(os.path.normpath(
                os.path.abspath(os.path.dirname(p))))
            if parent != trans_abs:
                count += 1
        return count

    def _update_organize_counts(self):
        """Refresh the Organize + Undo button labels with live counts.

        Called after every scan (initial + auto-refresh). Disables each
        button when its count is zero so the user can see at a glance
        that there's nothing to do.
        """
        raw_count = self._count_raw_movable()
        trans_count = self._count_trans_movable()
        try:
            origins = _load_origins()
            raw_orig = len(origins.get("raw", {}) or {})
            trans_orig = len(origins.get("translated", {}) or {})
        except Exception:
            raw_orig = 0
            trans_orig = 0
        try:
            self._ip_organize_btn.setText(
                f"\U0001f4e5  Organize ({raw_count})")
            self._ip_organize_btn.setEnabled(raw_count > 0)
            self._ip_undo_btn.setText(f"\u21a9  Undo ({raw_orig})")
            self._ip_undo_btn.setEnabled(raw_orig > 0)
            self._comp_organize_btn.setText(
                f"\U0001f4e5  Organize ({trans_count})")
            self._comp_organize_btn.setEnabled(trans_count > 0)
            self._comp_undo_btn.setText(f"\u21a9  Undo ({trans_orig})")
            self._comp_undo_btn.setEnabled(trans_orig > 0)
        except Exception:
            pass

    def _import_epub(self):
        """Pick raw source EPUB(s), copy them into Library/Raw, and scaffold
        output folders in the configured output root with a source_epub.txt
        pointer. This is how the In Progress tab picks each book up later.

        The import does NOT push the files into the translator's input field
        — that's what the context-menu "Load for translation" action is for.
        """
        from PySide6.QtWidgets import QFileDialog
        start_dir = str(Path.home() / "Downloads")
        if not os.path.isdir(start_dir):
            start_dir = str(Path.home())
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import file(s) into Library", start_dir,
            "EPUB files (*.epub);;All supported (*.epub *.txt *.pdf *.html);;All files (*.*)",
        )
        if not paths:
            return
        self._import_paths_into_library(paths, source="picker", target="raw")

    def _import_paths_into_library(self, paths, source: str = "picker",
                                   target: str = "raw"):
        """Core import pipeline used by both the file picker and drag-drop.

        Copies each supported file into the library (counter-suffix on name
        collision), and refreshes the library once all files have been
        processed. Shows a single summary dialog at the end so batch imports
        don't spam modal messages.

        *target* selects the destination:
          * ``"raw"`` (default): copy into ``Library/Raw`` and scaffold an
            output folder with a ``source_epub.txt`` sidecar + empty
            ``translation_progress.json``. Used by the "Import EPUB" button
            and by drag-drop while the **In Progress** tab is active.
          * ``"translated"``: copy EPUBs into ``Library/Translated`` with no
            scaffolding (the files are already finished compiled outputs).
            Only ``.epub`` is accepted; other types are reported as skipped.
            Used by drag-drop while the **Completed** tab is active.
        """
        if not paths:
            return
        if target == "translated":
            supported_exts = (".epub",)
            dest_label = "Library/Translated"
        else:
            supported_exts = (".epub", ".txt", ".pdf", ".html", ".htm")
            dest_label = "Library/Raw"
        imported: list[str] = []
        skipped: list[str] = []
        errors: list[str] = []
        for raw_path in paths:
            try:
                if not raw_path:
                    continue
                path = os.path.abspath(raw_path)
                if not os.path.isfile(path):
                    skipped.append(f"{os.path.basename(raw_path)} (not a file)")
                    continue
                if not path.lower().endswith(supported_exts):
                    # On the Completed tab only .epub makes sense; non-EPUBs
                    # are called out explicitly so the user understands why
                    # a mixed drop didn't land.
                    if target == "translated":
                        skipped.append(
                            f"{os.path.basename(path)} "
                            f"(only EPUBs go to Library/Translated)"
                        )
                    else:
                        skipped.append(
                            f"{os.path.basename(path)} (unsupported type)"
                        )
                    continue
                dest = self._import_single_file(path, target=target)
                if dest:
                    imported.append(dest)
            except Exception as exc:
                logger.error("Import failed for %s: %s\n%s",
                             raw_path, exc, traceback.format_exc())
                errors.append(f"{os.path.basename(raw_path)}: {exc}")
        if imported:
            QTimer.singleShot(0, self._load_books)
        if not (imported or skipped or errors):
            return
        # Drag-drop imports get animated, non-modal status feedback. File-
        # picker imports keep the detailed summary dialog so users who
        # explicitly chose "Import EPUB" still see per-file diagnostics.
        if source == "drop":
            if imported and not errors:
                self._show_toast(
                    f"\u2705  Imported {len(imported)} file"
                    f"{'s' if len(imported) != 1 else ''} into {dest_label}"
                )
            elif imported and errors:
                self._show_toast(
                    f"\u26a0\ufe0f  Imported {len(imported)} into {dest_label}, "
                    f"failed {len(errors)}"
                )
            elif errors:
                self._show_toast(
                    f"\u26a0\ufe0f  Import failed ({len(errors)} error"
                    f"{'s' if len(errors) != 1 else ''})"
                )
            elif skipped:
                # Translated-target skips are already self-describing
                # ("only EPUBs go to Library/Translated"); use a matching
                # short toast so users understand why nothing landed.
                if target == "translated":
                    self._show_toast(
                        f"\u2139\ufe0f  Only EPUBs can be dropped onto the "
                        f"Completed tab ({len(skipped)} skipped)"
                    )
                else:
                    self._show_toast(
                        f"\u2139\ufe0f  Skipped {len(skipped)} unsupported "
                        f"file{'s' if len(skipped) != 1 else ''}"
                    )
            return
        # Summary dialog — one message per batch, not per file.
        title = "Import"
        parts: list[str] = []
        if imported:
            parts.append(
                f"Imported {len(imported)} file{'s' if len(imported) != 1 else ''} "
                f"into {dest_label}."
            )
            if len(imported) <= 10:
                parts.append("\n".join(
                    "  \u2022 " + os.path.basename(p) for p in imported))
            else:
                parts.append("\n".join(
                    "  \u2022 " + os.path.basename(p) for p in imported[:10]))
                parts.append(f"  \u2026 and {len(imported) - 10} more.")
        if skipped:
            parts.append(
                f"\nSkipped {len(skipped)} file{'s' if len(skipped) != 1 else ''}:"
            )
            parts.append("\n".join("  \u2022 " + s for s in skipped[:8]))
        if errors:
            parts.append(
                f"\n{len(errors)} error{'s' if len(errors) != 1 else ''}:"
            )
            parts.append("\n".join("  \u2022 " + e for e in errors[:5]))
        if imported:
            parts.append(
                "\nRight-click any card and choose \u201cLoad for translation\u201d "
                "when you're ready to translate."
            )
        body = "\n".join(parts)
        if imported and not errors:
            QMessageBox.information(self, title, body)
        else:
            QMessageBox.warning(self, title, body)

    def _import_single_file(self, path: str, target: str = "raw") -> str | None:
        """Copy *path* into the appropriate library subfolder.

        *target* is either ``"raw"`` (→ ``Library/Raw`` + output-folder
        scaffold for the translator) or ``"translated"`` (→
        ``Library/Translated``, no scaffold — the file is treated as a
        finished compiled EPUB). Name collisions append a counter suffix
        (``name (2).epub``) so no pre-existing file is clobbered.

        Returns the destination path on success, None on failure. Raises
        no exceptions — failures are logged and surfaced via the caller's
        aggregated error list.
        """
        if target == "translated":
            dest_dir = get_library_translated_dir()
        else:
            dest_dir = get_library_raw_dir()
        try:
            dest = os.path.join(dest_dir, os.path.basename(path))
            # Don't overwrite a different file with the same name — append a
            # counter suffix like "name (2).epub" as File Explorer does.
            if (os.path.abspath(path) != os.path.abspath(dest)
                    and os.path.isfile(dest)):
                stem, ext = os.path.splitext(os.path.basename(path))
                counter = 2
                while True:
                    candidate = os.path.join(dest_dir, f"{stem} ({counter}){ext}")
                    if not os.path.isfile(candidate):
                        dest = candidate
                        break
                    counter += 1
            if os.path.abspath(path) != os.path.abspath(dest):
                shutil.copy2(path, dest)
            # Only raw imports feed the translator pipeline — the raw-inputs
            # registry + per-book output scaffold are meaningless (and
            # actively misleading) for a finished compiled EPUB that's
            # being filed under Library/Translated.
            if target == "raw":
                record_library_raw_input(dest)
        except Exception as exc:
            logger.error("Import copy failed for %s: %s\n%s",
                         path, exc, traceback.format_exc())
            raise
        if target != "raw":
            return dest
        try:
            roots = _resolve_output_roots(self._config)
            if roots:
                output_root = roots[0]
                base = os.path.splitext(os.path.basename(dest))[0]
                output_folder = os.path.join(output_root, base)
                os.makedirs(output_folder, exist_ok=True)
                sidecar = os.path.join(output_folder, "source_epub.txt")
                with open(sidecar, "w", encoding="utf-8") as f:
                    f.write(dest)
                progress_file_path = os.path.join(
                    output_folder, "translation_progress.json")
                if not os.path.isfile(progress_file_path):
                    import json as _json
                    with open(progress_file_path, "w", encoding="utf-8") as pf:
                        _json.dump(
                            {"chapters": {}, "chapter_chunks": {}, "version": "2.1"},
                            pf, ensure_ascii=False, indent=2,
                        )
        except Exception as exc:
            logger.error("Output scaffold failed for %s: %s\n%s",
                         path, exc, traceback.format_exc())
            raise
        return dest

    def _load_for_translation(self, book: dict):
        """Push the card's raw source path into the translator's input field.

        Emits :attr:`import_epub_requested` which :class:`TranslatorGUI`
        handles. Falls back to a warning when no raw path is available.
        """
        raw = book.get("raw_source_path") or book.get("path") or ""
        if not raw or not os.path.isfile(raw):
            QMessageBox.warning(self, "Load for translation",
                                "No raw source file is available for this card.")
            return
        try:
            self.import_epub_requested.emit(raw)
            record_library_raw_input(raw)
        except Exception:
            logger.debug("Failed to emit import_epub_requested: %s",
                         traceback.format_exc())

    def _load_multi_for_translation(self, paths: list):
        """Push one-or-many raw source paths into the translator's input.

        Deduplicates by normalized path, emits ``import_epubs_requested``
        when more than one file is loaded (so the receiver can show a
        "N files selected" summary), or ``import_epub_requested`` for the
        single-file case. Every loaded path is recorded in the raw-inputs
        registry so subsequent library scans can resolve it.
        """
        if not paths:
            return
        seen: set[str] = set()
        uniq: list[str] = []
        for p in paths:
            if not p or not os.path.isfile(p):
                continue
            key = os.path.normcase(os.path.normpath(os.path.abspath(p)))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(p)
        if not uniq:
            QMessageBox.warning(self, "Load for translation",
                                "No resolvable raw source files in the selection.")
            return
        try:
            if len(uniq) == 1:
                self.import_epub_requested.emit(uniq[0])
            else:
                self.import_epubs_requested.emit(uniq)
            for p in uniq:
                record_library_raw_input(p)
        except Exception:
            logger.debug("Failed to emit multi import: %s",
                         traceback.format_exc())

    def _organize_into_library(self):
        """Move raw sources into Library/Raw *and* compiled EPUBs into
        Library/Translated, recording every move in a reversible origins
        file (``Library/library_origins.txt``). Per-file policy:

          * **Raw** sources are MOVED from their current location into
            Library/Raw and the card's ``source_epub.txt`` pointer is
            rewritten so future scans resolve to the library copy.
          * **Translated** compiled EPUBs are MOVED from their output
            folder into Library/Translated (the output folder itself
            stays on disk, minus the EPUB).

        The operation is idempotent: files already inside their
        destination folder are skipped silently.
        """
        raw_dir = get_library_raw_dir()
        trans_dir = get_library_translated_dir()
        raw_abs = os.path.normcase(os.path.normpath(os.path.abspath(raw_dir)))
        trans_abs = os.path.normcase(os.path.normpath(os.path.abspath(trans_dir)))

        # Collect raw sources that aren't already in Library/Raw. Walk
        # BOTH tabs — a book at 100 %% progress lives in
        # ``_completed_books`` but its raw source may still be sitting
        # outside Library/Raw (the common case for a freshly-compiled
        # translation). Previously only ``_in_progress_books`` was
        # inspected, so finishing a translation effectively hid the raw
        # from "Organize" forever. Raw paths are deduped so a book that
        # appears in both scans doesn't get scheduled twice.
        raw_moves: list[tuple[dict, str]] = []
        seen_raw_keys: set[str] = set()

        def _queue_raw(book: dict) -> None:
            p = book.get("raw_source_path") or ""
            if not p or not os.path.isfile(p):
                return
            parent = os.path.normcase(os.path.normpath(
                os.path.abspath(os.path.dirname(p))))
            if parent == raw_abs:
                return
            key = os.path.normcase(os.path.normpath(os.path.abspath(p)))
            if key in seen_raw_keys:
                return
            seen_raw_keys.add(key)
            raw_moves.append((book, p))

        for book in self._in_progress_books:
            _queue_raw(book)
        for book in self._completed_books:
            # Library entries are already filed — nothing to organize.
            if book.get("in_library"):
                continue
            _queue_raw(book)

        # Collect compiled EPUBs that aren't already in Library/Translated.
        translated_moves: list[tuple[dict, str]] = []
        for book in self._completed_books:
            # Skip Library entries (they already live in Library/Translated).
            if book.get("in_library"):
                continue
            p = book.get("path") or ""
            if not p or not os.path.isfile(p):
                continue
            if not p.lower().endswith(".epub"):
                continue  # only compiled .epub files get organized for now
            parent = os.path.normcase(os.path.normpath(os.path.abspath(os.path.dirname(p))))
            if parent == trans_abs:
                continue
            translated_moves.append((book, p))

        if not raw_moves and not translated_moves:
            QMessageBox.information(
                self, "Organize Files into Library",
                "All resolvable files are already in Library/Raw or "
                "Library/Translated. Nothing to move.")
            return

        preview = []
        if raw_moves:
            preview.append(
                f"Raw \u2192 Library/Raw: {len(raw_moves)} file"
                f"{'s' if len(raw_moves) != 1 else ''}")
        if translated_moves:
            preview.append(
                f"Translated \u2192 Library/Translated: {len(translated_moves)} file"
                f"{'s' if len(translated_moves) != 1 else ''}")

        msg = QMessageBox(self)
        msg.setWindowTitle("Organize Files into Library")
        msg.setText(
            "Move the following files into the Library?\n\n"
            + "\n".join("  \u2022 " + line for line in preview)
            + "\n\nThis is reversible via the Undo Move button."
        )
        msg.setIcon(QMessageBox.Question)
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        if msg.exec() != QMessageBox.Yes:
            return

        origins = _load_origins()
        raw_origins = dict(origins.get("raw", {}) or {})
        trans_origins = dict(origins.get("translated", {}) or {})
        pair_map = dict(origins.get("pairs", {}) or {})

        moved_raw = 0
        moved_trans = 0
        errors: list[str] = []
        # Per-book dest-basename trackers (keyed by ``id(book)``) so we can
        # pair translated↔raw after both moves finish. Books that only
        # have one side moved in this run contribute a partial entry and
        # we fall back to ``raw_source_path`` on the book dict below.
        raw_dest_by_book: dict[int, str] = {}
        trans_dest_by_book: dict[int, str] = {}

        def _unique_dest(directory: str, base_name: str) -> str:
            cand = os.path.join(directory, base_name)
            if not os.path.isfile(cand):
                return cand
            stem, ext = os.path.splitext(base_name)
            counter = 2
            while True:
                cand = os.path.join(directory, f"{stem} ({counter}){ext}")
                if not os.path.isfile(cand):
                    return cand
                counter += 1

        # Raw sources: MOVE + update source_epub.txt pointer.
        for book, src in raw_moves:
            try:
                dest = _unique_dest(raw_dir, os.path.basename(src))
                shutil.move(src, dest)
                raw_origins[os.path.basename(dest)] = os.path.abspath(src)
                record_library_raw_input(dest)
                out_folder = book.get("output_folder") or ""
                if out_folder and os.path.isdir(out_folder):
                    try:
                        with open(os.path.join(out_folder, "source_epub.txt"),
                                  "w", encoding="utf-8") as f:
                            f.write(dest)
                    except OSError as pe:
                        logger.debug("Update source_epub.txt failed: %s", pe)
                raw_dest_by_book[id(book)] = os.path.basename(dest)
                moved_raw += 1
            except Exception as exc:
                errors.append(f"raw:{os.path.basename(src)}: {exc}")

        # Translated compiled EPUBs: MOVE into Library/Translated.
        for book, src in translated_moves:
            try:
                dest = _unique_dest(trans_dir, os.path.basename(src))
                shutil.move(src, dest)
                trans_origins[os.path.basename(dest)] = os.path.abspath(src)
                trans_dest_by_book[id(book)] = os.path.basename(dest)
                moved_trans += 1
            except Exception as exc:
                errors.append(f"translated:{os.path.basename(src)}: {exc}")

        # Pair up translated↔raw so later lookups don't have to rely on
        # filename-stem matching (which fails when raw and translated are
        # in different languages) or the output-folder sidecar (which
        # fails if that folder is later deleted). For each book whose
        # translated was moved, its raw is either:
        #   * in ``raw_dest_by_book`` (we just organized it), or
        #   * already filed under ``Library/Raw`` from a previous import
        #     (pick it up via the book's ``raw_source_path``).
        for book_id, trans_basename in trans_dest_by_book.items():
            raw_basename = raw_dest_by_book.get(book_id)
            if not raw_basename:
                # Fall back to the book's pre-existing raw if it already
                # lives in Library/Raw.
                paired_book = None
                for source_list in (self._completed_books,
                                    self._in_progress_books):
                    for b in source_list:
                        if id(b) == book_id:
                            paired_book = b
                            break
                    if paired_book is not None:
                        break
                if paired_book is not None:
                    rp = paired_book.get("raw_source_path") or ""
                    if rp and os.path.isfile(rp):
                        rp_parent = os.path.normcase(os.path.normpath(
                            os.path.abspath(os.path.dirname(rp))))
                        if rp_parent == raw_abs:
                            raw_basename = os.path.basename(rp)
            if raw_basename:
                pair_map[trans_basename] = raw_basename

        origins["raw"] = raw_origins
        origins["translated"] = trans_origins
        origins["pairs"] = pair_map
        _save_origins(origins)

        summary_parts = []
        if moved_raw:
            summary_parts.append(
                f"Moved {moved_raw} raw source"
                f"{'s' if moved_raw != 1 else ''} into Library/Raw.")
        if moved_trans:
            summary_parts.append(
                f"Moved {moved_trans} compiled EPUB"
                f"{'s' if moved_trans != 1 else ''} into Library/Translated.")
        summary = "\n".join(summary_parts) or "Nothing was moved."
        if errors:
            summary += (f"\n\n{len(errors)} error"
                        f"{'s' if len(errors) != 1 else ''}:\n"
                        + "\n".join(errors[:5]))
        QMessageBox.information(self, "Organize Files into Library", summary)
        self._load_books()

    def _undo_organize_prompt(self):
        """Ask the user which category to undo, then reverse those moves.

        Prompts with three buttons: Raw, Translated, All. Each button
        moves the affected files back to their original locations (from
        the origins registry) and rewrites any stale ``source_epub.txt``
        pointers.
        """
        origins = _load_origins()
        raw_map = dict(origins.get("raw", {}) or {})
        trans_map = dict(origins.get("translated", {}) or {})
        pair_map = dict(origins.get("pairs", {}) or {})
        if not raw_map and not trans_map:
            QMessageBox.information(
                self, "Undo Move",
                "No previous organize operation is recorded. Nothing to undo.")
            return

        msg = QMessageBox(self)
        msg.setWindowTitle("Undo Move")
        raw_count = len(raw_map)
        trans_count = len(trans_map)
        msg.setText(
            f"Which category do you want to restore to the original location?\n\n"
            f"  \u2022 Raw sources in Library/Raw: {raw_count}\n"
            f"  \u2022 Translated EPUBs in Library/Translated: {trans_count}"
        )
        msg.setIcon(QMessageBox.Question)
        btn_raw = msg.addButton("Raw", QMessageBox.AcceptRole)
        btn_trans = msg.addButton("Translated", QMessageBox.AcceptRole)
        btn_all = msg.addButton("All", QMessageBox.AcceptRole)
        btn_cancel = msg.addButton(QMessageBox.Cancel)
        msg.setDefaultButton(btn_all if (raw_map and trans_map) else (btn_raw or btn_trans))
        msg.exec()
        chosen = msg.clickedButton()
        if chosen is None or chosen is btn_cancel:
            return
        restore_raw = chosen is btn_raw or chosen is btn_all
        restore_trans = chosen is btn_trans or chosen is btn_all

        restored_raw = 0
        restored_trans = 0
        errors: list[str] = []

        if restore_raw and raw_map:
            raw_dir = get_library_raw_dir()
            remaining = {}
            for lib_name, orig_path in raw_map.items():
                lib_file = os.path.join(raw_dir, lib_name)
                if not os.path.isfile(lib_file):
                    errors.append(f"raw:{lib_name}: not found in Library/Raw")
                    continue
                try:
                    os.makedirs(os.path.dirname(orig_path) or ".", exist_ok=True)
                except OSError:
                    pass
                try:
                    shutil.move(lib_file, orig_path)
                    restored_raw += 1
                    # Any output folders whose source_epub.txt still points
                    # at the library copy get rewritten back to the original.
                    try:
                        for root in _resolve_output_roots(self._config):
                            try:
                                for sub in os.scandir(root):
                                    if not sub.is_dir(follow_symlinks=False):
                                        continue
                                    sidecar = os.path.join(sub.path, "source_epub.txt")
                                    if not os.path.isfile(sidecar):
                                        continue
                                    try:
                                        with open(sidecar, "r", encoding="utf-8") as f:
                                            raw_text = f.read().strip()
                                    except OSError:
                                        continue
                                    if (os.path.normcase(os.path.normpath(raw_text)) ==
                                            os.path.normcase(os.path.normpath(lib_file))):
                                        try:
                                            with open(sidecar, "w", encoding="utf-8") as f:
                                                f.write(orig_path)
                                        except OSError:
                                            pass
                            except (PermissionError, OSError):
                                continue
                    except Exception:
                        pass
                except Exception as exc:
                    remaining[lib_name] = orig_path
                    errors.append(f"raw:{lib_name}: {exc}")
            origins["raw"] = remaining
            # Drop any pair entries that reference a raw basename we
            # just restored — the raw no longer lives in Library/Raw
            # so a future ``_find_raw_source_for_library_epub`` lookup
            # would otherwise return a stale path.
            restored_basenames = set(raw_map.keys()) - set(remaining.keys())
            if restored_basenames and pair_map:
                pair_map = {
                    tb: rb for tb, rb in pair_map.items()
                    if rb not in restored_basenames
                }

        if restore_trans and trans_map:
            trans_dir = get_library_translated_dir()
            remaining = {}
            for lib_name, orig_path in trans_map.items():
                lib_file = os.path.join(trans_dir, lib_name)
                if not os.path.isfile(lib_file):
                    errors.append(f"translated:{lib_name}: not found in Library/Translated")
                    continue
                try:
                    os.makedirs(os.path.dirname(orig_path) or ".", exist_ok=True)
                except OSError:
                    pass
                try:
                    shutil.move(lib_file, orig_path)
                    restored_trans += 1
                except Exception as exc:
                    remaining[lib_name] = orig_path
                    errors.append(f"translated:{lib_name}: {exc}")
            origins["translated"] = remaining
            # A translated file that's been restored out of Library/
            # Translated can no longer be looked up as a pair key.
            restored_trans_basenames = set(trans_map.keys()) - set(remaining.keys())
            if restored_trans_basenames and pair_map:
                pair_map = {
                    tb: rb for tb, rb in pair_map.items()
                    if tb not in restored_trans_basenames
                }

        origins["pairs"] = pair_map
        _save_origins(origins)

        summary_parts = []
        if restore_raw:
            summary_parts.append(
                f"Raw restored: {restored_raw}/{len(raw_map) if raw_map else 0}")
        if restore_trans:
            summary_parts.append(
                f"Translated restored: {restored_trans}/{len(trans_map) if trans_map else 0}")
        summary = "\n".join(summary_parts) or "Nothing was restored."
        if errors:
            summary += (f"\n\n{len(errors)} error"
                        f"{'s' if len(errors) != 1 else ''}:\n"
                        + "\n".join(errors[:5]))
        QMessageBox.information(self, "Undo Move", summary)
        self._load_books()

    def _refresh_view(self):
        """Re-filter + re-render both tab grids from the cached scan data."""
        self._populate_in_progress(self._filtered(self._in_progress_books))
        self._populate_completed(self._filtered(self._completed_books))

    def _filtered(self, books: list[dict]) -> list[dict]:
        query = self._search.text().strip().lower()
        if query:
            books = [b for b in books if query in str(b.get("name", "")).lower()]
        return self._sorted_books(books)

    def _auto_refresh(self):
        """Lightweight auto-refresh: only reload if either tab changed."""
        if self._scanner_thread and self._scanner_thread.isRunning():
            return
        self._scanner_thread = _DualScannerThread(self._config, self)
        self._scanner_thread.finished.connect(self._on_auto_scan_done)
        self._scanner_thread.start()

    def _on_auto_scan_done(self, in_progress: list[dict], completed: list[dict]):
        """Receive auto-refresh scan; rebuild cards if anything visible changed.

        Previously we diffed only the set of ``book['path']`` values,
        which meant an in-progress book whose chapter count advanced
        (e.g. 0 → 3 translated) kept the same card signature and never
        re-rendered — so the user saw "Not started" / "0 KB" frozen on
        disk long after the translator had made real progress. We now
        diff a richer per-book signature covering progress counts,
        translation state, mtime, and resolved raw-source path so any
        visible field drift triggers a refresh.
        """
        def _sig(books):
            return {
                (b.get("path", "") or ""): (
                    int(b.get("completed_chapters", 0) or 0),
                    int(b.get("total_chapters", 0) or 0),
                    int(b.get("failed_chapters", 0) or 0),
                    int(b.get("pending_chapters", 0) or 0),
                    str(b.get("translation_state", "") or ""),
                    bool(b.get("has_compiled_output", False)),
                    str(b.get("raw_source_path", "") or ""),
                    float(b.get("mtime", 0) or 0),
                    int(b.get("size", 0) or 0),
                    len(b.get("compiled_conflicts") or []),
                )
                for b in books
            }
        ip_new_sig = _sig(in_progress)
        ip_old_sig = _sig(self._in_progress_books)
        comp_new_sig = _sig(completed)
        comp_old_sig = _sig(self._completed_books)
        if ip_new_sig != ip_old_sig or comp_new_sig != comp_old_sig:
            self._in_progress_books = in_progress
            self._completed_books = completed
            self._refresh_view()
        # Always refresh the Organize / Undo counters — origins registry may
        # have changed even when the card list didn't (e.g. after undo).
        self._update_organize_counts()

    def _load_books(self):
        """Kick off an async scan of both tabs and show the loading spinner.

        Safe to call whether the loading state is already active (first
        open — :meth:`__init__` primes it before ``show()``) or not
        (later refreshes from the "Refresh" button) — ``_show_loading``
        is idempotent.
        """
        for t in self._cover_threads:
            try:
                t.quit()
                t.wait(100)
            except Exception:
                pass
        self._cover_threads.clear()
        if self._scanner_thread and self._scanner_thread.isRunning():
            return
        self._show_loading()
        self._scanner_thread = _DualScannerThread(self._config, self)
        self._scanner_thread.finished.connect(self._on_initial_scan_done)
        self._scanner_thread.start()

    def _on_initial_scan_done(self, in_progress: list[dict], completed: list[dict]):
        self._in_progress_books = in_progress
        self._completed_books = completed
        self._hide_loading()
        self._refresh_view()
        self._update_organize_counts()

    def _populate_grid_common(
        self,
        books: list[dict],
        grid_layout: QGridLayout,
        card_list: list[_BookCard],
        count_label: QLabel,
        empty_label: QLabel,
        count_word: str,
        selected_paths: set[str] | None = None,
    ):
        """Shared render pipeline used by both tabs."""
        selected_paths = selected_paths if selected_paths is not None else set()
        # Prune the selection set of paths that no longer exist in this
        # scan result — otherwise organize / undo / delete operations can
        # leave stale entries that silently skew "Load N for translation".
        current_paths = {b.get("path", "") for b in books}
        selected_paths &= current_paths
        for card in card_list:
            try:
                card.setParent(None)
                card.deleteLater()
            except Exception:
                pass
        card_list.clear()

        while grid_layout.count():
            item = grid_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

        if not books:
            empty_label.show()
            count_label.setText("")
            return

        empty_label.hide()
        count_label.setText(
            f"{len(books)} {count_word}{'s' if len(books) != 1 else ''}"
        )

        preset = _SIZE_PRESETS[self._card_size]
        card_w = preset["card_w"]
        spacing = preset["spacing"]
        grid_layout.setHorizontalSpacing(spacing)
        grid_layout.setVerticalSpacing(spacing + 2)
        cols = max(1, (self.width() - 40) // (card_w + spacing))

        show_raw_title = bool(getattr(self, "_show_raw_titles", False))
        for idx, book in enumerate(books):
            card = _BookCard(book, preset=preset, show_raw_title=show_raw_title)
            card.clicked.connect(self._on_card_clicked)
            card.context_menu_requested.connect(self._show_context_menu)
            card.select_requested.connect(self._on_card_select_requested)
            # Re-apply previous selection state from the tab's selection set
            # so auto-refresh doesn't wipe the user's multi-card selection.
            if book.get("path", "") in selected_paths:
                card.set_selected(True)
            card_list.append(card)
            row, col = divmod(idx, cols)
            grid_layout.addWidget(card, row, col, Qt.AlignTop | Qt.AlignLeft)

            loader = _CoverLoader(
                book["path"],
                file_type=book.get("type", "epub"),
                config=self._config,
                original_path=book.get("original_path"),
                raw_source_path=book.get("raw_source_path"),
                parent=self,
            )
            loader.finished.connect(self._on_cover_loaded)
            self._cover_threads.append(loader)
            loader.start()

        for c in range(grid_layout.columnCount()):
            grid_layout.setColumnStretch(c, 0)
        grid_layout.setColumnStretch(cols, 1)

        # Pack cards to the top of the viewport: give the row below the
        # last populated row all of the vertical stretch. Without this,
        # QGridLayout distributes the extra vertical space across rows and
        # leaves a giant gap between rows 1 and 2 in the Completed tab when
        # there are fewer rows than viewport height.
        last_row = (len(books) - 1) // cols + 1
        for r in range(last_row):
            grid_layout.setRowStretch(r, 0)
        grid_layout.setRowStretch(last_row, 1)

    def _populate_in_progress(self, books: list[dict]):
        self._populate_grid_common(
            books, self._ip_grid_layout, self._ip_cards,
            self._ip_count_label, self._ip_empty_label, "novel",
            selected_paths=self._selected_paths_ip,
        )

    def _populate_completed(self, books: list[dict]):
        self._populate_grid_common(
            books, self._comp_grid_layout, self._comp_cards,
            self._comp_count_label, self._comp_empty_label, "book",
            selected_paths=self._selected_paths_comp,
        )

    def _active_selection(self) -> tuple[set[str], list[_BookCard]]:
        """Return (selection_set, card_list) for the currently active tab."""
        if self._tabs.currentIndex() == 0:
            return self._selected_paths_ip, self._ip_cards
        return self._selected_paths_comp, self._comp_cards

    def _on_rubber_band_selection(self, books: list, modifiers):
        """Apply rubber-band drag selection to the active tab.

        Ctrl / Shift preserves the existing selection and adds to it;
        a plain drag replaces the selection with whatever landed in the
        rubber-band rectangle.
        """
        selected, cards = self._active_selection()
        paths = {b.get("path", "") for b in books if b.get("path", "")}
        extend = bool(modifiers & (Qt.ControlModifier | Qt.ShiftModifier))
        if not extend:
            selected.clear()
        selected.update(paths)
        for c in cards:
            c.set_selected(c.book.get("path", "") in selected)

    def _on_empty_area_clicked(self, modifiers):
        """Clear the active-tab selection when user clicks empty grid space.

        Skipped when the user is modifying the existing selection with
        Ctrl / Shift — otherwise a fat-fingered click would wipe a
        carefully-built multi-selection.
        """
        if modifiers & (Qt.ControlModifier | Qt.ShiftModifier):
            return
        selected, cards = self._active_selection()
        if not selected:
            return
        selected.clear()
        for c in cards:
            c.set_selected(False)

    def _on_card_select_requested(self, book: dict, modifiers):
        """Handle left-click selection with modifier semantics.

        - Ctrl-click toggles this card in the active-tab selection set.
        - Shift-click adds without clearing.
        - Plain click replaces the selection with just this card.

        Visual updates are applied to every card in the active tab so
        multi-selection reads correctly (highlighted borders).
        """
        selected, cards = self._active_selection()
        path = book.get("path", "") or ""
        if not path:
            return
        if modifiers & Qt.ControlModifier:
            if path in selected:
                selected.discard(path)
            else:
                selected.add(path)
        elif modifiers & Qt.ShiftModifier:
            selected.add(path)
        else:
            selected.clear()
            selected.add(path)
        for c in cards:
            c.set_selected(c.book.get("path", "") in selected)

    def _on_cover_loaded(self, book_path: str, cover_path: str):
        if not cover_path:
            return
        for card in (*self._ip_cards, *self._comp_cards):
            if card.book.get("path") == book_path:
                card.set_cover(cover_path)

    def _apply_filter(self, text):
        self._refresh_view()

    def _on_card_clicked(self, book):
        file_type = book.get("type", "epub")
        if file_type in ("pdf", "txt"):
            # Open with best available editor/viewer (cross-platform)
            try:
                path = book["path"]
                _no_window = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
                if sys.platform == 'win32':
                    if file_type == "txt":
                        # Prefer Notepad++ for text files
                        _npp_paths = [
                            r'C:\Program Files\Notepad++\notepad++.exe',
                            r'C:\Program Files (x86)\Notepad++\notepad++.exe',
                        ]
                        _npp = next((p for p in _npp_paths if os.path.exists(p)), None)
                        if _npp:
                            subprocess.Popen([_npp, path], creationflags=_no_window)
                        else:
                            subprocess.Popen(['notepad.exe', path], creationflags=_no_window)
                    else:
                        os.startfile(path)  # PDF → default viewer
                elif sys.platform == 'darwin':
                    if file_type == "txt" and shutil.which('code'):
                        subprocess.Popen(['code', path])
                    else:
                        subprocess.Popen(['open', path])
                else:
                    if file_type == "txt":
                        _editors = ['gedit', 'kate', 'code', 'mousepad', 'xed', 'pluma']
                        _editor = next((e for e in _editors if shutil.which(e)), 'xdg-open')
                        subprocess.Popen([_editor, path])
                    else:
                        subprocess.Popen(['xdg-open', path])
            except Exception as exc:
                logger.error("Could not open file: %s\n%s", exc, traceback.format_exc())
                QMessageBox.warning(self, "Error", f"Could not open file:\n{exc}")
            return
        # EPUB: open the web-like book details page instead of jumping directly
        # into the reader. Users who want the old "straight to reader" behavior
        # can use the "Open in Reader" context menu action.
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            dialog = BookDetailsDialog(book, config=self._config, parent=self)
            QApplication.restoreOverrideCursor()
            dialog.setModal(False)
            dialog.setAttribute(Qt.WA_DeleteOnClose)
            self._active_details = dialog  # prevent GC
            dialog.show()
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            tb = traceback.format_exc()
            logger.error("Could not open book details: %s\n%s", exc, tb)
            QMessageBox.warning(self, "Error", f"Could not open book details:\n{exc}\n\nDetails:\n{tb}")

    def _open_reader_direct(self, book):
        """Bypass the details page and open the reader for an EPUB card."""
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            # When the card carries a distinct raw source (Completed-tab
            # EPUB that was organized alongside its Library/Raw pair, or
            # an in-progress promotion) pass it as the reader's alt so
            # the Show-raw pill can flip between translated and source.
            book_path = book.get("path", "")
            alt_path = book.get("raw_source_path", "") or ""
            if alt_path:
                try:
                    if os.path.normcase(os.path.abspath(alt_path)) == \
                            os.path.normcase(os.path.abspath(book_path)):
                        alt_path = ""
                except Exception:
                    alt_path = ""
                if alt_path and not os.path.isfile(alt_path):
                    alt_path = ""
            reader = EpubReaderDialog(
                book_path,
                config=self._config,
                parent=self,
                alt_epub_path=alt_path or None,
            )
            QApplication.restoreOverrideCursor()
            reader.setModal(False)
            reader.setAttribute(Qt.WA_DeleteOnClose)
            self._active_reader = reader  # prevent GC
            reader.show()
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            tb = traceback.format_exc()
            logger.error("Could not open EPUB: %s\n%s", exc, tb)
            QMessageBox.warning(self, "Error", f"Could not open EPUB:\n{exc}\n\nDetails:\n{tb}")

    def _show_context_menu(self, book, pos):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background: #1e1e2e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #e0e0e0; font-size: 9pt; padding: 4px; }
            QMenu::item { padding: 6px 20px; border-radius: 3px; }
            QMenu::item:selected { background: #3a3a5e; }
        """)
        # Selection-aware: if the right-clicked card isn't part of the
        # current selection, treat this as a single-card action (replace
        # selection with just this card). Otherwise keep the existing
        # multi-selection intact so "Load N for translation" can act on it.
        selected, cards = self._active_selection()
        path = book.get("path", "") or ""
        if path and path not in selected:
            selected.clear()
            selected.add(path)
            for c in cards:
                c.set_selected(c.book.get("path", "") in selected)
        selected_books = [c.book for c in cards
                          if c.book.get("path", "") in selected]
        if not selected_books:
            selected_books = [book]
        file_type = book.get("type", "epub")
        if file_type == "in_progress":
            details_action = menu.addAction("\U0001f4d1  Open Book Details")
            details_action.triggered.connect(lambda: self._on_card_clicked(book))
            # Always expose "Open in Reader" when an EPUB source is
            # resolvable — either the raw source (Not Started / mid-
            # translation cards) or a compiled translated EPUB.
            # NOTE: ``QAction.triggered`` emits a ``bool checked`` argument
            # which Qt binds to lambda *default* parameters, so we MUST close
            # over the loop-free locals without using ``lambda x=local: …``
            # (otherwise ``x`` gets overwritten with the checked bool and
            # subsequent ``x.get(…)`` calls crash with AttributeError).
            raw_src = book.get("raw_source_path") or ""
            out_epub = book.get("output_epub_path") or ""
            if (raw_src and raw_src.lower().endswith(".epub")
                    and os.path.isfile(raw_src)):
                raw_reader_action = menu.addAction("\U0001f4d6  Open in Reader")
                raw_reader_action.triggered.connect(
                    lambda: self._open_reader_direct({"path": raw_src, "type": "epub"})
                )
            if out_epub and os.path.isfile(out_epub):
                reader_action = menu.addAction("\U0001f4d6  Open Translated EPUB")
                reader_action.triggered.connect(
                    lambda: self._open_reader_direct({"path": out_epub, "type": "epub"})
                )
        elif file_type == "epub":
            details_action = menu.addAction("\U0001f4d1  Open Book Details")
            details_action.triggered.connect(lambda: self._on_card_clicked(book))
            reader_action = menu.addAction("\U0001f4d6  Open in Reader")
            reader_action.triggered.connect(lambda: self._open_reader_direct(book))
        else:
            open_action = menu.addAction("\U0001f4c2  Open File")
            open_action.triggered.connect(lambda: self._on_card_clicked(book))
        # "Load for translation" — pushes the card's raw source into the
        # translator's input field (moved out of the Import EPUB button).
        # When multiple cards are selected, the label becomes
        # "Load N files for translation" and the action emits all resolvable
        # raw paths at once via :attr:`import_epubs_requested`.
        def _resolve_raw(b: dict) -> str:
            r = b.get("raw_source_path") or ""
            if (not r and b.get("type") == "epub"
                    and os.path.isfile(b.get("path", ""))):
                r = b["path"]
            return r if r and os.path.isfile(r) else ""

        raw_candidates = [p for p in (_resolve_raw(b) for b in selected_books) if p]
        if raw_candidates:
            menu.addSeparator()
            if len(raw_candidates) > 1:
                label = f"\U0001f501  Load {len(raw_candidates)} files for translation"
                tooltip = ("Set these files as the translator's current input "
                           "(batch translation).")
            else:
                label = "\U0001f501  Load for translation"
                tooltip = "Set this file as the translator's current input."
            load_action = menu.addAction(label)
            load_action.setToolTip(tooltip)
            # See "NOTE" above — close over ``raw_candidates`` without a
            # default arg so Qt's ``checked`` bool can't overwrite it.
            load_action.triggered.connect(
                lambda: self._load_multi_for_translation(raw_candidates))
        menu.addSeparator()
        folder_action = menu.addAction("\U0001f4c2  Open Output Folder")
        # Resolution rules:
        #   * In-progress cards: the card ``path`` IS the output folder.
        #   * Anything else: use :func:`_resolve_book_output_folder` so
        #     Library/Translated entries fall back to the origins
        #     registry's original output folder (mirrors the Book
        #     Details 📁 button). Previously this branch used
        #     ``os.path.dirname(book['path'])`` as the fallback, which
        #     for library-organized EPUBs pointed at ``Library/
        #     Translated`` instead of the workspace that actually
        #     contains ``translation_progress.json`` / response_*.
        #   * Last-resort fallback: the book's own containing folder,
        #     same as before, so non-library files without any
        #     registered origin still get *some* folder opened.
        if file_type == "in_progress":
            output_folder = book.get("output_folder") or book.get("path", "")
        else:
            output_folder = (_resolve_book_output_folder(book)
                             or os.path.dirname(book.get("path", "")))
        folder_action.triggered.connect(lambda: _open_folder_in_explorer(output_folder))
        lib_action = menu.addAction("\U0001f4c1  Open Library Folder")
        lib_action.triggered.connect(lambda: _open_folder_in_explorer(get_library_dir()))
        menu.addSeparator()
        copy_path_action = menu.addAction("\U0001f4cb  Copy Path")
        copy_path_action.triggered.connect(lambda: QApplication.clipboard().setText(book["path"]))
        # Delete — tab-specific semantics:
        #   * In Progress → delete the output folder (recursive)
        #   * Completed   → delete the .epub / compiled file
        # Works with multi-selection too ("Delete N items"). Always asks
        # for confirmation first.
        menu.addSeparator()
        delete_targets = list(selected_books) if len(selected_books) > 1 else [book]
        n_del = len(delete_targets)
        if n_del > 1:
            delete_label = f"\U0001f5d1\ufe0f  Delete {n_del} items"
        else:
            delete_label = "\U0001f5d1\ufe0f  Delete"
        delete_action = menu.addAction(delete_label)
        delete_action.setToolTip(
            "In Progress \u2192 deletes the output folder.  "
            "Completed \u2192 deletes the EPUB file.  "
            "(Asks for confirmation first.)"
        )
        delete_action.triggered.connect(
            lambda: self._delete_books_prompt(delete_targets)
        )
        menu.exec(pos)

    def _delete_books_prompt(self, books: list):
        """Confirm + delete the backing file / folder for each book.

        Target resolution:

          * **Library entries** (``in_library=True``) — delete
            ``book['path']`` as a single file. These live in
            ``Library/Translated`` and don't own a surrounding workspace.
          * **Output-folder cards** (``type == "in_progress"`` OR any
            completed card with a resolvable ``output_folder``) —
            delete the folder recursively so every sibling artefact
            goes with it. This previously only fired for in-progress
            cards, which meant a compiled PDF sitting next to a
            ``*_translated.html`` would only remove the PDF — the next
            scan would promote the HTML as the new compiled output and
            re-spawn a card over the same folder, forcing the user to
            hit Delete a second time.
          * Everything else — delete ``book['path']`` as a plain file.

        Deletions are irreversible — the user sees a single
        confirmation dialog summarizing everything that will be
        removed.
        """
        if not books:
            return
        # Resolve targets: (label, path, is_folder)
        targets: list[tuple[str, str, bool]] = []
        seen_targets: set[str] = set()

        def _queue(label: str, pth: str, is_folder: bool) -> None:
            key = os.path.normcase(os.path.normpath(os.path.abspath(pth)))
            if key in seen_targets:
                return
            seen_targets.add(key)
            targets.append((label, pth, is_folder))

        for b in books:
            file_type = b.get("type", "epub")
            in_library = bool(b.get("in_library"))
            output_folder = b.get("output_folder") or ""
            # In-progress cards always delete the folder.
            if file_type == "in_progress":
                folder = output_folder or b.get("path", "") or ""
                if folder and os.path.isdir(folder):
                    _queue(b.get("name") or os.path.basename(folder),
                           folder, True)
                continue
            # Completed but NOT library-filed: the card represents the
            # entire output-folder workspace (compiled file plus any
            # ``response_*``, ``_translated.*``, ``images/``, etc.).
            # Delete the folder so a single click actually cleans up
            # every artefact — otherwise the next auto-refresh scan
            # would promote the leftover ``_translated.html`` /
            # ``_translated.txt`` into a new card.
            if (not in_library and output_folder
                    and os.path.isdir(output_folder)):
                _queue(b.get("name") or os.path.basename(output_folder),
                       output_folder, True)
                continue
            # Library entry (or any other loose file-backed card).
            p = b.get("path", "") or ""
            if p and os.path.isfile(p):
                _queue(b.get("name") or os.path.basename(p), p, False)
        if not targets:
            QMessageBox.information(
                self, "Delete",
                "Nothing to delete \u2014 none of the selected cards "
                "point at a file or folder on disk.",
            )
            return

        # Build a confirmation summary.
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Delete")
        lines = []
        for label, pth, is_folder in targets[:10]:
            kind = "folder" if is_folder else "file"
            lines.append(f"  \u2022 {label}  ({kind})")
        if len(targets) > 10:
            lines.append(f"  \u2026 and {len(targets) - 10} more.")
        msg.setText(
            f"Permanently delete {len(targets)} "
            f"item{'s' if len(targets) != 1 else ''}?\n\n"
            + "\n".join(lines)
            + "\n\nThis cannot be undone."
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Cancel)
        if msg.exec() != QMessageBox.Yes:
            return

        deleted = 0
        errors: list[str] = []
        for label, pth, is_folder in targets:
            try:
                if is_folder:
                    shutil.rmtree(pth)
                else:
                    os.remove(pth)
                deleted += 1
                # Drop the deleted path out of every selection set so the
                # next context-menu open doesn't resurrect it.
                try:
                    self._selected_paths_ip.discard(pth)
                    self._selected_paths_comp.discard(pth)
                except Exception:
                    pass
            except Exception as exc:
                logger.error("Delete failed for %s: %s\n%s",
                             pth, exc, traceback.format_exc())
                errors.append(f"{label}: {exc}")

        summary = f"Deleted {deleted} of {len(targets)} item" \
                  f"{'s' if len(targets) != 1 else ''}."
        if errors:
            summary += (
                f"\n\n{len(errors)} error"
                f"{'s' if len(errors) != 1 else ''}:\n"
                + "\n".join("  \u2022 " + e for e in errors[:5])
            )
            QMessageBox.warning(self, "Delete", summary)
        else:
            QMessageBox.information(self, "Delete", summary)
        # Re-scan so the library immediately reflects the deletions.
        QTimer.singleShot(0, self._load_books)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep the drag overlay + toast glued to the current geometry — they
        # don't participate in the root layout since they float above it.
        self._reposition_overlays()
        # Debounce: re-flow cards once the user stops dragging (300ms).
        if not (self._ip_cards or self._comp_cards):
            return
        if not hasattr(self, '_resize_timer'):
            self._resize_timer = QTimer(self)
            self._resize_timer.setSingleShot(True)
            self._resize_timer.timeout.connect(self._refresh_view)
        self._resize_timer.start(300)

    def _reposition_overlays(self):
        """Re-layout the drop overlay + toast after a resize / show event."""
        if getattr(self, "_drop_overlay", None) is not None:
            margin = 16
            self._drop_overlay.setGeometry(
                margin, margin,
                max(0, self.width() - margin * 2),
                max(0, self.height() - margin * 2),
            )
            self._drop_overlay.raise_()
        if getattr(self, "_toast", None) is not None:
            self._toast.adjustSize()
            tw = min(self._toast.width(), max(240, self.width() - 40))
            th = self._toast.height()
            self._toast.setGeometry(
                (self.width() - tw) // 2,
                self.height() - th - 24,
                tw, th,
            )
            self._toast.raise_()

    def showEvent(self, event):
        super().showEvent(event)
        self._reposition_overlays()

    def closeEvent(self, event):
        """Hide the dialog instead of closing \u2014 persist settings.

        Also asks the translator parent to flush the in-memory config
        dict to ``config.json`` via :func:`_persist_config_via_parent`,
        so library settings survive app crashes / force-quits instead
        of only persisting when the main window's own save runs.
        """
        self._config['epub_library_sort'] = self._sort_mode
        self._config['epub_library_card_size'] = self._card_size
        self._config['epub_library_tab'] = self._current_tab
        _persist_config_via_parent(self)
        event.ignore()
        self.hide()

    # -- Drag & drop import -------------------------------------------------

    # Extensions accepted by each tab's drop zone. The In Progress tab
    # lands everything on the raw source pipeline (EPUB / TXT / PDF /
    # HTML) while the Completed tab only makes sense for finished
    # compiled EPUBs, so it filters down to just ``.epub``.
    _DND_RAW_EXTS = (".epub", ".txt", ".pdf", ".html", ".htm")
    _DND_TRANSLATED_EXTS = (".epub",)

    def _drop_target_kind(self) -> str:
        """Return ``"translated"`` when the Completed tab is active, else ``"raw"``."""
        try:
            return "translated" if self._tabs.currentIndex() == 1 else "raw"
        except Exception:
            return "raw"

    def _allowed_drop_exts(self) -> tuple:
        """Return the tuple of file extensions currently accepted by the drop zone."""
        return (self._DND_TRANSLATED_EXTS
                if self._drop_target_kind() == "translated"
                else self._DND_RAW_EXTS)

    def _dnd_accept_event(self, event) -> bool:
        """Return True iff the drag payload contains at least one file accepted
        by the currently active tab (EPUB-only on Completed, EPUB/TXT/PDF/HTML
        on In Progress)."""
        mime = event.mimeData()
        if not mime or not mime.hasUrls():
            return False
        allowed = self._allowed_drop_exts()
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            path = url.toLocalFile()
            if path and path.lower().endswith(allowed):
                return True
        return False

    def dragEnterEvent(self, event):
        if self._dnd_accept_event(event):
            event.setDropAction(Qt.CopyAction)
            event.acceptProposedAction()
            self._show_drop_overlay(True)
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if self._dnd_accept_event(event):
            event.setDropAction(Qt.CopyAction)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._show_drop_overlay(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        self._show_drop_overlay(False)
        mime = event.mimeData()
        if not mime or not mime.hasUrls():
            event.ignore()
            return
        target = self._drop_target_kind()
        allowed = self._allowed_drop_exts()
        paths: list[str] = []
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            p = url.toLocalFile()
            if p and p.lower().endswith(allowed):
                paths.append(p)
        if not paths:
            event.ignore()
            return
        event.acceptProposedAction()
        # Stay on the active tab: drops on Completed mean "file these
        # compiled EPUBs into Library/Translated" and the user should see
        # the Completed list refresh in-place. Drops on In Progress stay
        # on that tab so the new Raw import card appears where expected.
        dest_label = "Library/Translated" if target == "translated" else "Library/Raw"
        # Immediate toast so the user sees something happened even before
        # the import pipeline finishes. The pipeline will replace this with
        # a final "Imported N into <dest>" status in
        # :meth:`_import_paths_into_library`.
        self._show_toast(
            f"\U0001f4e5  Importing {len(paths)} file"
            f"{'s' if len(paths) != 1 else ''} into {dest_label}\u2026",
            auto_hide_ms=0,
        )
        self._import_paths_into_library(paths, source="drop", target=target)

    # -- Overlay / toast helpers --------------------------------------------

    def _show_drop_overlay(self, visible: bool):
        """Show / hide the purple "drop to import" overlay panel.

        The overlay's text is refreshed on every show so the user sees
        which library subfolder the active tab will route drops into
        (Raw for In Progress, Translated for Completed).
        """
        if getattr(self, "_drop_overlay", None) is None:
            return
        if visible:
            if self._drop_target_kind() == "translated":
                self._drop_overlay.setText(
                    "\U0001f4e5\n\nDrop EPUBs to import into Library/Translated\n\n"
                    "EPUB only"
                )
            else:
                self._drop_overlay.setText(
                    "\U0001f4e5\n\nDrop files to import into Library/Raw\n\n"
                    "EPUB \u00b7 PDF \u00b7 TXT \u00b7 HTML"
                )
            self._reposition_overlays()
            self._drop_overlay.show()
            self._drop_overlay.raise_()
        else:
            self._drop_overlay.hide()

    def _show_toast(self, text: str, auto_hide_ms: int = 2600):
        """Fade in a short status line at the bottom of the dialog.

        *auto_hide_ms* == 0 keeps the toast visible until the next
        :meth:`_show_toast` call or an explicit :meth:`_fade_out_toast`.
        Used by the drag-drop pipeline to stream a two-stage "Importing…"
        → "Imported N" status update without any modal interruption.
        """
        if getattr(self, "_toast", None) is None:
            return
        from PySide6.QtCore import QPropertyAnimation, QEasingCurve
        self._toast_hide_timer.stop()
        self._toast.setText(text)
        self._reposition_overlays()
        self._toast.show()
        self._toast.raise_()
        # Cancel any in-flight fade so a rapid second call doesn't race
        # the previous animation to zero opacity.
        if self._toast_anim is not None:
            try:
                self._toast_anim.stop()
            except Exception:
                pass
        anim = QPropertyAnimation(self._toast_opacity, b"opacity", self)
        anim.setDuration(180)
        anim.setStartValue(float(self._toast_opacity.opacity()))
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.start()
        self._toast_anim = anim
        if auto_hide_ms > 0:
            self._toast_hide_timer.start(auto_hide_ms)

    def _fade_out_toast(self):
        """Fade the toast back to 0 opacity and hide it on completion."""
        if getattr(self, "_toast", None) is None or not self._toast.isVisible():
            return
        from PySide6.QtCore import QPropertyAnimation, QEasingCurve
        if self._toast_anim is not None:
            try:
                self._toast_anim.stop()
            except Exception:
                pass
        anim = QPropertyAnimation(self._toast_opacity, b"opacity", self)
        anim.setDuration(260)
        anim.setStartValue(float(self._toast_opacity.opacity()))
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.InCubic)
        anim.finished.connect(self._toast.hide)
        anim.start()
        self._toast_anim = anim


# ---------------------------------------------------------------------------
# Book Details — metadata / TOC parser
# ---------------------------------------------------------------------------

# Compiled once: used by :func:`_extract_html_title_fast` to yank a chapter
# title out of a 32 KB HTML preview without spinning up BeautifulSoup for
# every file. ``re`` is a C extension that releases the GIL on each search,
# so this also lets a ThreadPoolExecutor actually get work done in parallel.
_RE_HTML_TITLE = re.compile(rb"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_RE_HTML_HEADING = re.compile(rb"<(h[1-6])[^>]*>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
_RE_HTML_STRIP_TAGS = re.compile(rb"<[^>]+>")
_RE_HTML_WS = re.compile(rb"\s+")


def _extract_html_title_fast(raw: bytes) -> str:
    """Return the first ``<title>`` (or h1–h6) text from an HTML chunk.

    Drop-in replacement for a BeautifulSoup title extraction when you just
    need the document title. Roughly an order of magnitude faster on the
    chapter-list hot path for 400-entry spines, and because the underlying
    ``re`` calls release the GIL it's safe to call concurrently from a
    :class:`~concurrent.futures.ThreadPoolExecutor`.
    """
    if not raw:
        return ""
    try:
        from html import unescape
        for regex, group in ((_RE_HTML_TITLE, 1), (_RE_HTML_HEADING, 2)):
            m = regex.search(raw)
            if not m:
                continue
            inner = _RE_HTML_STRIP_TAGS.sub(b" ", m.group(group))
            inner = _RE_HTML_WS.sub(b" ", inner).strip()
            if inner:
                return unescape(inner.decode("utf-8", errors="replace")).strip()
    except Exception:
        logger.debug("Fast title parse failed: %s", traceback.format_exc())
    return ""


def _parse_epub_details(epub_path: str, parse_chapter_titles: bool = True) -> dict:
    """Extract OPF metadata, spine order and per-chapter raw titles.

    Returns a dict shaped roughly like the OPF DC schema plus a ``chapters``
    list of ``{'href', 'filename', 'title'}``. All fields are best-effort and
    may be empty strings/lists on failure.

    When *parse_chapter_titles* is False, per-chapter HTML is not opened
    and chapter titles fall back to filename-derived labels. This is the
    fast path :class:`_BookDetailsLoader` uses for its preview pass so
    the cover + metadata render immediately while the real titles are
    parsed in the background.
    """
    import zipfile
    import posixpath
    from xml.etree import ElementTree as ET
    from html import unescape

    details = {
        "title": "",
        "authors": [],
        "publisher": "",
        "language": "",
        "date": "",
        "description": "",
        "subjects": [],
        "identifier": "",
        "chapters": [],
    }

    try:
        with zipfile.ZipFile(epub_path, "r") as zf:
            names = zf.namelist()
            names_set = set(names)

            opf_path = None
            if "META-INF/container.xml" in names_set:
                try:
                    container_xml = zf.read("META-INF/container.xml").decode("utf-8", errors="replace")
                    container_tree = ET.fromstring(container_xml)
                    ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
                    rootfile = container_tree.find(".//c:rootfile", ns)
                    if rootfile is not None:
                        opf_path = rootfile.get("full-path")
                except Exception:
                    pass
            if not opf_path:
                for zname in names:
                    if zname.lower().endswith(".opf"):
                        opf_path = zname
                        break
            if not opf_path or opf_path not in names_set:
                return details

            opf_xml = zf.read(opf_path).decode("utf-8", errors="replace")
            tree = ET.fromstring(opf_xml)
            opf_dir = posixpath.dirname(opf_path)

            DC = "http://purl.org/dc/elements/1.1/"
            OPF = "http://www.idpf.org/2007/opf"

            def _dc(tag: str) -> list[str]:
                return [
                    (el.text or "").strip()
                    for el in tree.findall(f".//{{{DC}}}{tag}")
                    if (el.text or "").strip()
                ]

            titles = _dc("title")
            details["title"] = titles[0] if titles else ""
            details["authors"] = _dc("creator")
            publishers = _dc("publisher")
            details["publisher"] = publishers[0] if publishers else ""
            languages = _dc("language")
            details["language"] = languages[0] if languages else ""
            dates = _dc("date")
            details["date"] = dates[0] if dates else ""
            descriptions = _dc("description")
            details["description"] = unescape(descriptions[0]) if descriptions else ""
            details["subjects"] = _dc("subject")
            identifiers = _dc("identifier")
            details["identifier"] = identifiers[0] if identifiers else ""

            # Build a manifest id -> (href, media_type) lookup so we can
            # resolve spine itemrefs into concrete chapter files.
            manifest: dict[str, tuple[str, str]] = {}
            for item_el in tree.findall(f".//{{{OPF}}}item"):
                item_id = item_el.get("id", "")
                href = item_el.get("href", "")
                media = item_el.get("media-type", "")
                if not item_id or not href:
                    continue
                full_href = posixpath.normpath(posixpath.join(opf_dir, href)) if opf_dir else href
                manifest[item_id] = (full_href, media)

            spine_el = tree.find(f".//{{{OPF}}}spine")
            ordered_hrefs: list[tuple[str, str]] = []  # [(id, href)]
            if spine_el is not None:
                for itemref in spine_el.findall(f"{{{OPF}}}itemref"):
                    idref = itemref.get("idref")
                    if idref and idref in manifest:
                        ordered_hrefs.append((idref, manifest[idref][0]))

            # Parse each chapter file for a title. Keep it cheap: only peek at
            # the first ~32KB which is more than enough for <title>/<h1>. The
            # per-chapter HTML scan is the dominant cost for large spines
            # (~399 chapters), so callers that only need metadata skip it
            # via ``parse_chapter_titles=False`` and use filename fallbacks.
            #
            # When we DO want chapter titles we do two optimizations:
            #   1. Read every chapter's first 32 KB serially from the zip
            #      (``zipfile.ZipFile`` is not thread-safe for concurrent
            #      reads) — this is fast since it's just stream decompression.
            #   2. Dispatch the actual title extraction to a thread pool
            #      using :func:`_extract_html_title_fast`. That function is
            #      built on the C-backed ``re`` module, so the GIL is
            #      released and we actually get parallel speedup on the
            #      CPU-bound parse step.
            chap_raw: dict[str, bytes] = {}
            if parse_chapter_titles:
                for idref, href in ordered_hrefs:
                    media_type = manifest.get(idref, ("", ""))[1]
                    if media_type and ("html" not in media_type.lower()
                                       and "xhtml" not in media_type.lower()):
                        continue
                    if href in names_set and href not in chap_raw:
                        try:
                            chap_raw[href] = zf.read(href)[:32_768]
                        except Exception:
                            chap_raw[href] = b""

            titles_by_href: dict[str, str] = {}
            if chap_raw:
                max_workers = min(16, max(2, (os.cpu_count() or 2) * 2))
                try:
                    with ThreadPoolExecutor(max_workers=max_workers) as pool:
                        hrefs = list(chap_raw.keys())
                        datas = [chap_raw[h] for h in hrefs]
                        for h, t in zip(hrefs, pool.map(_extract_html_title_fast, datas)):
                            if t:
                                titles_by_href[h] = t
                except Exception:
                    logger.debug("Parallel title parse failed, falling back: %s",
                                 traceback.format_exc())
                    for h, data in chap_raw.items():
                        t = _extract_html_title_fast(data)
                        if t:
                            titles_by_href[h] = t

            chapters = []
            for idref, href in ordered_hrefs:
                media_type = manifest.get(idref, ("", ""))[1]
                if media_type and ("html" not in media_type.lower() and "xhtml" not in media_type.lower()):
                    # Non-text spine item — still include so index matches.
                    chapters.append({"href": href, "filename": os.path.basename(href),
                                     "title": os.path.splitext(os.path.basename(href))[0]})
                    continue
                title = titles_by_href.get(href, "")
                if not title:
                    title = os.path.splitext(os.path.basename(href))[0]
                    title = title.replace("_", " ").replace("-", " ").strip() or title
                chapters.append({"href": href, "filename": os.path.basename(href), "title": title})
            details["chapters"] = chapters
    except Exception:
        logger.debug("EPUB details parse failed: %s", traceback.format_exc())

    return details


def _read_translated_chapter_title(path: str) -> str:
    """Extract a translated-chapter title from a response_*.html file.

    Uses the fast regex-based title extractor instead of BeautifulSoup so
    that batched calls from :class:`_BookDetailsLoader` finish in tens of
    milliseconds rather than seconds on a 400-chapter output folder.
    """
    try:
        with open(path, "rb") as f:
            raw = f.read(32_768)
        return _extract_html_title_fast(raw)
    except Exception:
        logger.debug("Translated title parse failed for %s: %s", path, traceback.format_exc())
    return ""


class _BookDetailsLoader(QThread):
    """Parse EPUB metadata + TOC + translation status off the UI thread.

    Emits two signals during the lifetime of a single ``run()`` call:

    * :attr:`preview_ready` — fires as soon as the OPF metadata + cover
      are extracted (fast path, no per-chapter HTML parsing). The details
      dialog renders the hero (cover, title, synopsis, metadata grid)
      immediately from this payload so the user isn't staring at a
      Halgakos placeholder while ~400 chapter HTML blobs are decoded.
    * :attr:`done` — fires once the full chapters_info list (including
      per-chapter raw titles, translated titles, and on-disk status)
      has been built. Drives the chapter list + progress strip.
    """
    preview_ready = Signal(dict)
    done = Signal(dict)
    error = Signal(str)

    def __init__(self, book: dict, parent=None):
        super().__init__(parent)
        self._book = book

    def run(self):
        try:
            book_path = self._book.get("path", "") or ""
            book_type = self._book.get("type", "epub")
            progress_file = self._book.get("progress_file")
            output_folder = self._book.get("output_folder")

            # Tab-driven dispatch:
            #   * in_progress card: ``path`` is an OUTPUT FOLDER. We look for
            #     the source EPUB via ``source_epub.txt`` or any .epub in the
            #     folder; if none is found we still build the details page
            #     from metadata.json + translation_progress.json alone.
            #   * epub card: ``path`` points at a real .epub. We also probe
            #     the sibling translation_progress.json so completed EPUBs
            #     inside an output folder still get per-chapter status.
            source_epub = ""
            if book_type == "in_progress":
                output_folder = output_folder or book_path
                # Prefer the raw_source_path already resolved by the scanner
                # (validated via source_epub.txt, Library/Raw lookup, and the
                # raw-inputs registry). Freshly imported Not Started cards
                # whose output folder only has the sidecar can still surface
                # a cover + full spine this way.
                raw_source = self._book.get("raw_source_path") or ""
                if raw_source and os.path.isfile(raw_source):
                    source_epub = raw_source
                if output_folder and os.path.isdir(output_folder):
                    progress_file = progress_file or os.path.join(
                        output_folder, "translation_progress.json")
                    # Authoritative pointer file second, then any .epub in
                    # the folder (which may be the compiled output).
                    if not source_epub:
                        pointed = _read_source_epub_pointer(output_folder)
                        if pointed:
                            source_epub = pointed
                    if not source_epub:
                        for entry in os.scandir(output_folder):
                            if (entry.is_file(follow_symlinks=False)
                                    and entry.name.lower().endswith(".epub")):
                                source_epub = entry.path
                                break
            else:
                source_epub = book_path if os.path.isfile(book_path) else ""
                if not progress_file or not output_folder:
                    parent_dir = os.path.dirname(book_path)
                    if parent_dir and os.path.isdir(parent_dir):
                        candidate_pf = os.path.join(parent_dir, "translation_progress.json")
                        if os.path.isfile(candidate_pf):
                            progress_file = progress_file or candidate_pf
                            output_folder = output_folder or parent_dir

            # Only treat the resolved source as an EPUB when its extension
            # actually matches — TXT/PDF raw sources should skip the
            # zip-based parsing so they don't produce empty details silently.
            source_is_epub = (bool(source_epub)
                              and source_epub.lower().endswith(".epub")
                              and os.path.isfile(source_epub))

            # ---- Phase 1: Fast metadata + cover (no per-chapter HTML) ----
            # Skips the BeautifulSoup pass over every spine chapter so the
            # details hero paints instantly; the full chapter titles are
            # re-parsed in Phase 2 below and emitted via ``done``.
            details = _parse_epub_details(source_epub, parse_chapter_titles=False) if source_is_epub else {
                "title": "", "authors": [], "publisher": "", "language": "",
                "date": "", "description": "", "subjects": [], "identifier": "",
                "chapters": [],
            }
            cover = _extract_cover(source_epub) if source_is_epub else None
            # Broaden cover search so TXT/PDF workspaces that keep a cover
            # next to their compiled output still get a real thumbnail, and
            # so EPUBs whose embedded cover extraction somehow fails fall
            # back to any image the output folder has on disk.
            if not cover and output_folder and os.path.isdir(output_folder):
                cover = _find_cover_in_dir(output_folder)

            # Load metadata.json eagerly so the preview already has the
            # translator-overriden title / authors / description.
            metadata_json = None
            if output_folder:
                meta_path = os.path.join(output_folder, "metadata.json")
                if os.path.isfile(meta_path):
                    try:
                        import json as _json
                        with open(meta_path, "r", encoding="utf-8") as f:
                            metadata_json = _json.load(f)
                    except Exception:
                        metadata_json = None

            # Emit the preview so the dialog can paint the hero row now.
            self.preview_ready.emit({
                "details": details,
                "cover": cover or "",
                "metadata_json": metadata_json or {},
            })

            # ---- Phase 2: Slow per-chapter title parsing ----
            # Re-parse the spine WITH per-chapter HTML title extraction so
            # the chapter list can show "Prologue", "Chapter 1: ..." etc.
            # rather than just filename stubs.
            if source_is_epub:
                details = _parse_epub_details(source_epub, parse_chapter_titles=True)

            prog = None
            if progress_file and os.path.isfile(progress_file):
                summary = _read_progress_summary(progress_file)
                if summary is not None:
                    prog = summary["prog"]

            # Resolve each spine chapter to a translation status.
            chapters_info = []
            prog_chapters = (prog or {}).get("chapters", {}) or {}
            # Build lookup by normalized basename (no extension, no response_ prefix).
            def _norm(name: str) -> str:
                base = os.path.basename(name or "")
                if base.lower().startswith("response_"):
                    base = base[len("response_"):]
                while True:
                    stem, ext = os.path.splitext(base)
                    if not ext:
                        break
                    base = stem
                return base.lower()

            # --- Paired raw-EPUB title harvest (library entries only) ---
            #
            # For a Completed-tab library entry the EPUB we just parsed IS
            # the compiled translation — every ``ch['title']`` is already
            # translated, so there's no distinction between raw and
            # translated titles. When a paired raw EPUB exists
            # (``raw_source_path`` from the scanner, or resolved via
            # :func:`_find_raw_source_for_library_epub`), parse its spine
            # too and remember a filename-normalized + index-based lookup
            # of source-language titles. :func:`_resolve_chapter` below
            # then swaps the title semantics so the BookDetails
            # "Show raw titles" toggle has something to flip to.
            library_raw_title_by_norm: dict[str, str] = {}
            library_raw_titles_by_index: list[str] = []
            if (self._book.get("in_library")
                    and not self._book.get("is_in_progress")):
                raw_path = self._book.get("raw_source_path", "") or ""
                if not raw_path:
                    try:
                        raw_path = _find_raw_source_for_library_epub(book_path) or ""
                    except Exception:
                        raw_path = ""
                        logger.debug("Library-raw title resolve failed: %s",
                                     traceback.format_exc())
                if (raw_path and os.path.isfile(raw_path)
                        and raw_path.lower().endswith(".epub")):
                    try:
                        raw_details = _parse_epub_details(
                            raw_path, parse_chapter_titles=True)
                    except Exception:
                        raw_details = None
                        logger.debug("Raw EPUB parse failed for %s: %s",
                                     raw_path, traceback.format_exc())
                    for rc in (raw_details or {}).get("chapters", []) or []:
                        rt = (rc.get("title") or "").strip()
                        library_raw_titles_by_index.append(rt)
                        fn = rc.get("filename") or ""
                        if fn and rt:
                            library_raw_title_by_norm.setdefault(_norm(fn), rt)

            prog_by_basename: dict[str, dict] = {}
            prog_by_output: dict[str, dict] = {}
            for key, info in prog_chapters.items():
                if not isinstance(info, dict):
                    continue
                ob = info.get("original_basename") or ""
                of = info.get("output_file") or ""
                if ob:
                    prog_by_basename.setdefault(_norm(ob), info)
                if of:
                    prog_by_output.setdefault(_norm(of), info)

            # If we could neither load a progress file nor see an output
            # folder on disk, there is no translation context for this book
            # at all — we leave ``status`` empty so the UI renders no badge
            # (instead of misleadingly labeling every chapter "Pending").
            has_progress_context = bool(prog is not None
                                         or (output_folder and os.path.isdir(output_folder)))

            # Source EPUB absent: synthesize a spine from the progress file's
            # ``original_basename`` entries so the Chapters list still works.
            if not details.get("chapters") and prog_chapters:
                def _sort_key(item):
                    info = item[1]
                    try:
                        return int(info.get("actual_num") or info.get("chapter_num") or 0)
                    except (TypeError, ValueError):
                        return 0
                synth = []
                for key, info in sorted(prog_chapters.items(), key=_sort_key):
                    if not isinstance(info, dict):
                        continue
                    ob = info.get("original_basename") or info.get("output_file") or key
                    title = ob
                    title = os.path.splitext(os.path.basename(title))[0]
                    title = title.replace("_", " ").replace("-", " ").strip() or title
                    synth.append({
                        "href": ob,
                        "filename": os.path.basename(ob),
                        "title": title,
                    })
                if synth:
                    details["chapters"] = synth

            # Resolve every chapter's on-disk translation state in parallel.
            # Each per-chapter task is file-I/O bound (a few ``os.path.isfile``
            # probes + a 32 KB read of the matching translated HTML), so a
            # small thread pool turns a serial ~N × latency walk over a
            # 400-chapter output folder into something that finishes in the
            # time of a couple of sequential disk hits.
            chapters = details.get("chapters", []) or []
            output_dir_ok = bool(output_folder and os.path.isdir(output_folder))
            has_digit_re = re.compile(r"\d")

            def _resolve_chapter(item):
                idx, ch = item
                filename = ch["filename"]
                raw_title = ch["title"]
                is_special = not bool(has_digit_re.search(filename))
                is_gallery = _is_gallery_filename(filename)
                norm_key = _norm(filename)
                match = prog_by_basename.get(norm_key) or prog_by_output.get(norm_key)
                status = (match or {}).get("status", "")
                output_file = (match or {}).get("output_file", "")
                translated_title = ""
                translated_path = ""
                if output_dir_ok:
                    candidate_names = []
                    if output_file:
                        candidate_names.append(output_file)
                    base = os.path.splitext(filename)[0]
                    candidate_names.append(f"response_{base}.html")
                    candidate_names.append(f"response_{base}.xhtml")
                    candidate_names.append(f"{base}.html")
                    candidate_names.append(f"{base}.xhtml")
                    for candidate in candidate_names:
                        p = os.path.join(output_folder, candidate)
                        if os.path.isfile(p):
                            translated_path = p
                            if not status:
                                status = "completed"
                            break
                    if translated_path:
                        translated_title = _read_translated_chapter_title(translated_path)
                # Library-paired case: the loader parsed the COMPILED
                # EPUB, so ``raw_title`` right now is already translated.
                # Swap it with the paired raw EPUB's title (filename
                # normalized first, then index as a fallback) and promote
                # the compiled title to ``translated_title`` so
                # _ChapterRow + the Show-raw-titles toggle behave the
                # same way they do for in-progress books.
                if library_raw_title_by_norm or library_raw_titles_by_index:
                    paired_raw = library_raw_title_by_norm.get(norm_key, "")
                    if (not paired_raw
                            and idx < len(library_raw_titles_by_index)):
                        paired_raw = library_raw_titles_by_index[idx]
                    if paired_raw and paired_raw != raw_title:
                        translated_title = raw_title  # compiled title
                        raw_title = paired_raw
                        if not status:
                            # The book IS a completed translation — mark
                            # the row as completed so the default title
                            # policy in _ChapterRow picks the translated
                            # version (not the raw).
                            status = "completed"
                if not status:
                    status = "pending" if has_progress_context else ""
                if is_gallery:
                    status = ""
                return {
                    "index": idx,
                    "filename": filename,
                    "raw_title": raw_title,
                    "translated_title": translated_title,
                    "translated_path": translated_path,
                    "status": status,
                    "is_special": is_special,
                    "is_gallery": is_gallery,
                }

            chapters_info = []
            if chapters:
                items = list(enumerate(chapters))
                max_workers = min(32, max(4, (os.cpu_count() or 2) * 4))
                try:
                    with ThreadPoolExecutor(max_workers=max_workers) as pool:
                        chapters_info = list(pool.map(_resolve_chapter, items))
                except Exception:
                    logger.debug("Parallel chapter resolve failed, falling back: %s",
                                 traceback.format_exc())
                    chapters_info = [_resolve_chapter(it) for it in items]

            # metadata_json was loaded in Phase 1 above.

            self.done.emit({
                "details": details,
                "cover": cover or "",
                "chapters_info": chapters_info,
                "metadata_json": metadata_json or {},
                "progress": prog or {},
            })
        except Exception as exc:
            logger.error("Book details load error: %s\n%s", exc, traceback.format_exc())
            self.error.emit(f"{exc}")


# ---------------------------------------------------------------------------
# Book Details Dialog
# ---------------------------------------------------------------------------

class BookDetailsDialog(QDialog):
    """Web-like book page: cover, metadata, synopsis, and collapsible TOC.

    Clicking a chapter launches the EPUB reader positioned at that chapter.
    Opens instead of jumping straight into the reader when a library card is
    activated.
    """

    def __init__(self, book: dict, config: dict | None = None, parent=None):
        super().__init__(parent)
        self._book = book
        self._config = config or {}
        self._details: dict = {}
        self._chapters_info: list[dict] = []
        self._metadata_json: dict = {}
        self._loader: _BookDetailsLoader | None = None
        self._active_reader: QDialog | None = None
        # Gate chapter-row activations until Phase 2 has actually built the
        # list. Prevents stray double-clicks during the "Loading chapters…"
        # phase from being dispatched to a half-populated chapter list.
        self._chapters_loaded = False
        # Single-selected chapter row (by spine index). ``None`` means no
        # row currently has focus.
        self._selected_chapter_idx: int | None = None
        # Whether the "show special files" toggle is on. Its initial state is
        # coupled to the global "Translate Special Files" setting (Other
        # Settings): if the translator is configured to handle cover / nav /
        # toc files, the dialog defaults to showing them too and counting
        # them in the progress percentage. The user can still override the
        # toggle per-dialog — the override is persisted in config.
        _translate_special = _resolve_translate_special_files(self._config)
        _stored_show = self._config.get("epub_details_show_special_files", None)
        if _stored_show is None:
            self._show_special_files = _translate_special
        else:
            # User has an explicit preference — honor it, but also respect
            # the translate-special toggle so turning it ON auto-propagates.
            self._show_special_files = bool(_stored_show) or _translate_special
        # "Show raw titles" overrides the default "translated when completed,
        # raw otherwise" rule used by :class:`_ChapterRow` so every row shows
        # the source-language title regardless of translation status. Only
        # meaningful for in-progress EPUBs (where translated titles exist);
        # the toggle is hidden otherwise.
        self._show_raw_titles = bool(
            self._config.get("epub_details_show_raw_titles", False)
        )

        epub_path = book["path"]
        pretty = book.get("name") or os.path.splitext(os.path.basename(epub_path))[0]
        self.setWindowTitle(pretty)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
        screen = self.screen()
        if screen:
            avail = screen.availableGeometry()
            self.resize(int(avail.width() * 0.62), int(avail.height() * 0.8))
            self.setMinimumSize(int(avail.width() * 0.42), int(avail.height() * 0.5))
        else:
            self.resize(1100, 780)
            self.setMinimumSize(700, 500)

        icon_path = _find_halgakos_icon()
        if icon_path:
            self.setWindowIcon(QIcon(icon_path))

        self._setup_ui()
        self._start_loading()
        # Auto-refresh details every 2s so chapter status + progress
        # strip catch up with the translator as it writes new
        # ``response_*.html`` files. The timer pauses while a loader
        # is already running or the dialog is hidden — see
        # :meth:`_auto_refresh_details`.
        self._auto_refresh_timer = QTimer(self)
        self._auto_refresh_timer.setInterval(2000)
        self._auto_refresh_timer.timeout.connect(self._auto_refresh_details)
        QTimer.singleShot(2500, self._auto_refresh_timer.start)

    # -- UI construction ----------------------------------------------------

    def _setup_ui(self):
        self.setStyleSheet("""
            QDialog { background: #12121e; }
            QLabel#title { color: #e0e0e0; font-size: 22pt; font-weight: bold; }
            QLabel#author { color: #9aa2b8; font-size: 11pt; }
            QLabel#section { color: #b0b0c0; font-size: 10pt; font-weight: bold;
                             letter-spacing: 1px; }
            QLabel.meta-k { color: #7a8599; font-size: 8.5pt; }
            QLabel.meta-v { color: #e0e0e0; font-size: 9.5pt; font-weight: bold; }
            QLabel.tag { color: #c8cbe0; background: #2a2a3e;
                         border: 1px solid #3a3a5e; border-radius: 10px;
                         padding: 3px 9px; font-size: 8.5pt; }
            QLabel.pending { color: #7a8599; font-size: 8pt; font-style: italic; }
            QLabel.filename { color: #8a8fa8; font-size: 8.5pt; font-family: 'Consolas','Menlo',monospace; }
            QLabel.translated { color: #e0e0e0; font-size: 10pt; font-weight: bold; }
            QLabel.raw { color: #c8cbe0; font-size: 10pt; font-weight: bold; }
            QPushButton#start {
                background: #ff8a3d; color: #1e1616; font-weight: bold;
                font-size: 11pt; padding: 8px 18px; border-radius: 6px; border: none;
            }
            QPushButton#start:hover { background: #ffa05c; }
            QPushButton.icon-btn {
                background: #2a2a3e; color: #e0e0e0; border: 1px solid #3a3a5e;
                border-radius: 6px; padding: 6px 10px; font-size: 10pt;
            }
            QPushButton.icon-btn:hover { background: #3a3a5e; }
            QPushButton.icon-btn:disabled {
                background: #16161f; color: #2f2f3a;
                border: 1px dashed #2a2a3e;
            }
            QPushButton#toc-toggle {
                background: transparent; color: #b0b0c0;
                border: 1px solid #3a3a5e; border-radius: 6px;
                padding: 6px 12px; font-size: 10pt; font-weight: bold;
                text-align: left;
            }
            QPushButton#toc-toggle:hover { color: #e0e0e0; border-color: #6c63ff; }
            QLineEdit#toc-search {
                background: #1e1e2e; color: #e0e0e0;
                border: 1px solid #3a3a5e; border-radius: 6px;
                padding: 6px 10px; font-size: 9.5pt;
            }
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical { width: 10px; background: #12121e; }
            QScrollBar::handle:vertical { background: #3a3a5e; border-radius: 5px;
                                          min-height: 24px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Top bar with a back / close button
        topbar = QHBoxLayout()
        topbar.setContentsMargins(14, 10, 14, 6)
        topbar.setSpacing(8)
        back_btn = QPushButton("\u2190  Back to Library")
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #8a8fa8; border: none;"
            " font-size: 9.5pt; padding: 4px 6px; }"
            "QPushButton:hover { color: #e0e0e0; }"
        )
        back_btn.clicked.connect(self.close)
        topbar.addWidget(back_btn)
        topbar.addStretch()
        outer.addLayout(topbar)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        body = QWidget()
        self._scroll.setWidget(body)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(32, 14, 32, 32)
        body_layout.setSpacing(18)

        # ── Hero row: cover + title + actions + metadata ──
        hero = QHBoxLayout()
        hero.setSpacing(28)

        # Cover
        self._cover_lbl = QLabel()
        self._cover_lbl.setFixedSize(240, 340)
        self._cover_lbl.setAlignment(Qt.AlignCenter)
        self._cover_lbl.setStyleSheet(
            "background: #2a2a3e; border-radius: 6px; color: #555; font-size: 32pt;"
        )
        # Initial placeholder: Halgakos brand icon (or emoji as last resort).
        # ``_apply_halgakos_fallback`` is reused by ``_on_details_ready`` so
        # the same defensive rendering path runs when cover extraction fails.
        self._apply_halgakos_fallback()
        hero.addWidget(self._cover_lbl, 0, Qt.AlignTop)

        # Center column: title, author, actions, synopsis
        center = QVBoxLayout()
        center.setSpacing(8)
        self._title_lbl = QLabel(self._book.get("name", ""))
        self._title_lbl.setObjectName("title")
        self._title_lbl.setWordWrap(True)
        center.addWidget(self._title_lbl)
        self._author_lbl = QLabel("")
        self._author_lbl.setObjectName("author")
        self._author_lbl.setWordWrap(True)
        center.addWidget(self._author_lbl)

        # Progress strip (only for in-progress novels)
        self._progress_strip = QLabel("")
        self._progress_strip.setStyleSheet(
            "color: #ffd166; background: rgba(108, 99, 255, 0.14);"
            " border: 1px solid #6c63ff; border-radius: 6px;"
            " padding: 6px 10px; font-size: 9.5pt; font-weight: bold;"
        )
        self._progress_strip.hide()
        center.addWidget(self._progress_strip)

        # Action row
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self._start_btn = QPushButton("\U0001f4d6  Start reading")
        self._start_btn.setObjectName("start")
        self._start_btn.setCursor(Qt.PointingHandCursor)
        self._start_btn.clicked.connect(lambda: self._open_reader())
        actions.addWidget(self._start_btn)

        # Secondary action (only shown for in-progress novels) to open the
        # untouched source EPUB without the translated overlay.
        self._raw_btn = QPushButton("\U0001f4dc  Read raw source")
        self._raw_btn.setProperty("class", "icon-btn")
        self._raw_btn.setCursor(Qt.PointingHandCursor)
        self._raw_btn.setToolTip("Open the source EPUB without the translated overlay")
        self._raw_btn.clicked.connect(lambda: self._open_reader(raw_only=True))
        self._raw_btn.hide()
        actions.addWidget(self._raw_btn)

        # Icon buttons carry a QGraphicsOpacityEffect so the *emoji itself*
        # dims when the button is disabled — the stylesheet `color:` rule has
        # no effect on emoji glyphs, which render from their own color table.
        from PySide6.QtWidgets import QGraphicsOpacityEffect
        self._folder_btn = QPushButton("\U0001f4c2")
        self._folder_btn.setProperty("class", "icon-btn")
        self._folder_btn.setToolTip("Open output folder in file explorer")
        self._folder_btn.setCursor(Qt.PointingHandCursor)
        self._folder_btn.clicked.connect(self._open_output_folder)
        self._folder_btn_opacity = QGraphicsOpacityEffect(self._folder_btn)
        self._folder_btn_opacity.setOpacity(1.0)
        self._folder_btn.setGraphicsEffect(self._folder_btn_opacity)
        actions.addWidget(self._folder_btn)

        self._source_btn = QPushButton("\U0001f517")
        self._source_btn.setProperty("class", "icon-btn")
        self._source_btn.setToolTip("Reveal source file")
        self._source_btn.setCursor(Qt.PointingHandCursor)
        self._source_btn.clicked.connect(self._reveal_source)
        self._source_btn_opacity = QGraphicsOpacityEffect(self._source_btn)
        self._source_btn_opacity.setOpacity(1.0)
        self._source_btn.setGraphicsEffect(self._source_btn_opacity)
        actions.addWidget(self._source_btn)

        actions.addStretch()
        center.addLayout(actions)

        # Synopsis block
        synopsis_heading = QLabel("SYNOPSIS")
        synopsis_heading.setObjectName("section")
        center.addSpacing(6)
        center.addWidget(synopsis_heading)
        self._synopsis_lbl = QLabel("Loading\u2026")
        self._synopsis_lbl.setWordWrap(True)
        self._synopsis_lbl.setStyleSheet("color: #c8cbe0; font-size: 10pt; line-height: 1.55;")
        center.addWidget(self._synopsis_lbl)
        center.addStretch()
        hero.addLayout(center, 1)

        # Metadata column
        meta_col = QVBoxLayout()
        meta_col.setSpacing(14)
        meta_heading = QLabel("METADATA")
        meta_heading.setObjectName("section")
        meta_col.addWidget(meta_heading)
        self._meta_grid = QGridLayout()
        self._meta_grid.setContentsMargins(0, 0, 0, 0)
        self._meta_grid.setHorizontalSpacing(14)
        # Tight uniform vertical spacing — only the Title row gets extra
        # breathing room, and that’s provided by :func:`_fit_title_text`
        # below (dynamically shrinking the Title font instead of padding
        # every row with 36 px of whitespace).
        self._meta_grid.setVerticalSpacing(8)
        # Let the second column stretch so long titles wrap across more
        # horizontal space instead of clipping vertically.
        self._meta_grid.setColumnStretch(0, 0)
        self._meta_grid.setColumnStretch(1, 1)
        meta_col.addLayout(self._meta_grid)

        # Genres / Tags containers (filled in populate step)
        self._genres_heading = QLabel("GENRES")
        self._genres_heading.setObjectName("section")
        meta_col.addWidget(self._genres_heading)
        self._genres_row = QWidget()
        self._genres_layout = QHBoxLayout(self._genres_row)
        self._genres_layout.setContentsMargins(0, 0, 0, 0)
        self._genres_layout.setSpacing(6)
        self._genres_layout.addStretch()
        meta_col.addWidget(self._genres_row)

        self._tags_heading = QLabel("TAGS")
        self._tags_heading.setObjectName("section")
        meta_col.addWidget(self._tags_heading)
        self._tags_row = QWidget()
        self._tags_layout = QHBoxLayout(self._tags_row)
        self._tags_layout.setContentsMargins(0, 0, 0, 0)
        self._tags_layout.setSpacing(6)
        self._tags_layout.addStretch()
        meta_col.addWidget(self._tags_row)
        meta_col.addStretch()

        meta_wrapper = QWidget()
        meta_wrapper.setLayout(meta_col)
        # Wider column so long titles / author names don't wrap aggressively
        # and collide with the next metadata row. Bumped further so the
        # common CJK novel title fits on one line in the Title row.
        meta_wrapper.setMinimumWidth(380)
        meta_wrapper.setMaximumWidth(540)
        self._meta_wrapper = meta_wrapper
        hero.addWidget(meta_wrapper, 0, Qt.AlignTop)
        body_layout.addLayout(hero)

        # ── Chapters section (expanded by default) ──
        chap_header = QHBoxLayout()
        chap_header.setSpacing(10)
        self._toc_toggle = QPushButton("\u25bc  Chapters")
        self._toc_toggle.setObjectName("toc-toggle")
        self._toc_toggle.setCursor(Qt.PointingHandCursor)
        self._toc_toggle.clicked.connect(self._toggle_chapters)
        chap_header.addWidget(self._toc_toggle)
        chap_header.addStretch()
        # "Show special files" toggle mirrors the Progress Manager's behavior.
        # Hidden by default for EPUBs so files like cover/nav/toc/info don't
        # clutter the list; user state is persisted via config.
        from PySide6.QtWidgets import QCheckBox
        self._special_cb = QCheckBox("Show special files (cover, nav, toc)")
        self._special_cb.setToolTip(
            "When enabled, shows special files (files without chapter numbers "
            "like cover, nav, toc, info, message, etc.)"
        )
        self._special_cb.setChecked(self._show_special_files)
        self._special_cb.setStyleSheet("""
            QCheckBox { color: #c8cbe0; font-size: 9pt; spacing: 6px; padding: 4px 6px; }
            QCheckBox::indicator {
                width: 14px; height: 14px;
                border: 1px solid #5a9fd4; border-radius: 2px;
                background-color: #1e1e2e;
            }
            QCheckBox::indicator:checked {
                background-color: #5a9fd4; border-color: #5a9fd4;
                image: none;
            }
            QCheckBox::indicator:hover { border-color: #7bb3e0; }
        """)
        self._special_cb.toggled.connect(self._on_special_files_toggled)
        chap_header.addWidget(self._special_cb)
        # "Show raw titles" toggle: when checked, every chapter row displays
        # the source-language title instead of the translated one (if any).
        # Only useful for in-progress EPUBs where translated titles exist —
        # hidden otherwise via :meth:`_on_details_ready`.
        self._raw_titles_cb = QCheckBox("Show raw titles")
        self._raw_titles_cb.setToolTip(
            "Show the source-language chapter titles instead of the "
            "translated titles, even for chapters that have already been "
            "translated."
        )
        self._raw_titles_cb.setChecked(self._show_raw_titles)
        self._raw_titles_cb.setStyleSheet(self._special_cb.styleSheet())
        self._raw_titles_cb.toggled.connect(self._on_raw_titles_toggled)
        # Hidden by default; the details-ready handler flips it on for
        # in-progress EPUBs that actually have translated chapters to
        # distinguish from.
        self._raw_titles_cb.hide()
        chap_header.addWidget(self._raw_titles_cb)
        self._toc_search = QLineEdit()
        self._toc_search.setObjectName("toc-search")
        self._toc_search.setPlaceholderText("\U0001f50d  Search chapters\u2026")
        self._toc_search.setFixedWidth(260)
        self._toc_search.textChanged.connect(self._apply_chapter_filter)
        chap_header.addWidget(self._toc_search)
        body_layout.addLayout(chap_header)

        self._chap_container = QWidget()
        self._chap_layout = QVBoxLayout(self._chap_container)
        self._chap_layout.setContentsMargins(4, 4, 4, 4)
        self._chap_layout.setSpacing(4)
        # Visible by default — the TOC is the main reading entry point.
        body_layout.addWidget(self._chap_container)
        # Separate "user intent" flag for the chapter section's
        # visibility. The loading placeholder flips ``_chap_container``'s
        # visibility as an implementation detail, so we can't use
        # :meth:`isVisible` as the source of truth — we'd end up
        # collapsing the section every time the placeholder hid the
        # container. :meth:`_toggle_chapters` updates this flag; every
        # other codepath (populate, refresh, search filter) consults it.
        self._chap_section_expanded = True

        # Standalone "⏳  Loading chapters…" placeholder that sits in the
        # same body slot as the chapter container. Kept as a SIBLING
        # widget (not inside ``_chap_layout``) so it stays on screen
        # uninterrupted while :meth:`_populate_chapters` clears the
        # layout + batches in the new rows — the user sees one loading
        # state that ends with all rows appearing at once, instead of
        # watching rows pop in over multiple seconds.
        self._chap_loading_lbl = QLabel("\u23f3  Loading chapters\u2026")
        self._chap_loading_lbl.setAlignment(Qt.AlignCenter)
        self._chap_loading_lbl.setMinimumHeight(60)
        self._chap_loading_lbl.setStyleSheet(
            "color: #7a8599; font-size: 10pt; padding: 28px 22px 24px 22px;"
        )
        self._chap_loading_lbl.hide()
        body_layout.addWidget(self._chap_loading_lbl)

        body_layout.addStretch()
        outer.addWidget(self._scroll, 1)

    # -- Data population ----------------------------------------------------

    def _start_loading(self):
        self._synopsis_lbl.setText("Loading book details\u2026")
        # Install a placeholder in the chapters section so the user sees
        # activity while the (slower) spine parse is running in the
        # background. Replaced in :meth:`_on_details_ready` with real rows.
        self._show_chapter_placeholder()
        self._is_auto_refreshing = False
        self._loader = _BookDetailsLoader(self._book, self)
        # ``preview_ready`` fires first with cover + metadata so the hero
        # paints immediately; ``done`` follows with the full chapter list.
        self._loader.preview_ready.connect(self._on_preview_ready)
        self._loader.done.connect(self._on_details_ready)
        self._loader.error.connect(self._on_details_error)
        self._loader.start()

    def _auto_refresh_details(self):
        """Re-run the details loader so per-chapter status catches up.

        Skipped when the dialog is hidden (no point paying for disk I/O
        when the user can't see it) or a loader is already mid-flight.
        The refresh runs a silent :meth:`_BookDetailsLoader` pass that
        does NOT show the loading placeholder — we reuse the existing
        hero + chapter rows and swap them in place only when a
        per-chapter status signature actually changed.
        """
        try:
            if not self.isVisible():
                return
        except Exception:
            pass
        if self._loader is not None:
            try:
                if self._loader.isRunning():
                    return
            except Exception:
                pass
        self._is_auto_refreshing = True
        self._loader = _BookDetailsLoader(self._book, self)
        # Only subscribe to ``done`` — ``preview_ready`` is only useful
        # for the initial paint and would re-flash the cover + synopsis
        # with every tick otherwise.
        self._loader.done.connect(self._on_details_ready)
        self._loader.error.connect(self._on_details_error)
        self._loader.start()

    def _show_chapter_placeholder(self):
        """Show the "⏳  Loading chapters…" placeholder while the chapter
        list isn't ready to display.

        Toggles the standalone :attr:`_chap_loading_lbl` sibling widget
        instead of stuffing a label inside ``_chap_layout`` — that way
        the clear-and-rebuild inside :meth:`_populate_chapters` can
        happen silently behind the hidden container while the user
        still sees a steady loading message.
        """
        # Clear any stale rows (e.g. leftover from a previous open)
        # without resurrecting a per-layout placeholder label.
        while self._chap_layout.count():
            item = self._chap_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        # Hide the chapter container (which is currently empty) and show
        # the sibling placeholder in its slot.
        if getattr(self, "_chap_container", None) is not None:
            self._chap_container.hide()
        if getattr(self, "_chap_loading_lbl", None) is not None:
            self._chap_loading_lbl.show()

    def _apply_halgakos_fallback(self):
        """Render the Halgakos brand icon into the cover label.

        Scaled to fit the full cover-label footprint so freshly imported
        Not Started cards (which have no extractable EPUB cover on disk
        yet) still render a visible thumbnail instead of an empty dark
        box. Falls back to a book emoji if the icon file fails to load.
        """
        icon_path = _find_halgakos_icon()
        applied = False
        if icon_path:
            try:
                pm = QPixmap(icon_path)
                if not pm.isNull():
                    # Fill the full label footprint (240×340) rather than a
                    # small 160×160 centered icon so the fallback doesn't
                    # read as "no thumbnail" against the dark background.
                    scaled = pm.scaled(
                        self._cover_lbl.width(),
                        self._cover_lbl.height(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                    self._cover_lbl.setPixmap(scaled)
                    self._cover_lbl.setText("")
                    applied = True
            except Exception:
                logger.debug("Halgakos pixmap failed: %s",
                             traceback.format_exc())
        if not applied:
            self._cover_lbl.setPixmap(QPixmap())
            self._cover_lbl.setText("\U0001f4d6")

    def _apply_hero_payload(self, payload: dict):
        """Paint cover + title + author + synopsis + metadata grid.

        Shared between :meth:`_on_preview_ready` (fast pass, no chapter
        data yet) and :meth:`_on_details_ready` (final pass). Safe to
        call repeatedly — the grid is cleared + rebuilt each invocation.
        """
        self._details = payload.get("details", self._details) or self._details
        self._metadata_json = payload.get("metadata_json", self._metadata_json) or self._metadata_json
        cover_path = payload.get("cover", "")
        cover_applied = False
        if cover_path:
            try:
                pm = QPixmap(cover_path)
                if not pm.isNull():
                    scaled = pm.scaled(self._cover_lbl.width(), self._cover_lbl.height(),
                                       Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self._cover_lbl.setPixmap(scaled)
                    self._cover_lbl.setText("")
                    cover_applied = True
            except Exception:
                logger.debug("Cover pixmap load failed for %s: %s",
                             cover_path, traceback.format_exc())
        if not cover_applied:
            # No real cover could be extracted — keep the Halgakos branding
            # in place so the details dialog never shows an empty box.
            self._apply_halgakos_fallback()

        title = (self._metadata_json.get("title") or self._details.get("title")
                 or self._book.get("name", ""))
        self._title_lbl.setText(title or self._book.get("name", ""))
        authors = list(self._metadata_json.get("authors") or self._details.get("authors") or [])
        if isinstance(authors, str):
            authors = [authors]
        self._author_lbl.setText(", ".join(authors) if authors else "")

        synopsis = (self._metadata_json.get("description")
                    or self._details.get("description") or "").strip()
        if synopsis:
            # Collapse any run of blank lines (i.e. a newline followed by
            # one or more whitespace-only lines) down to a single newline
            # so the synopsis renders compactly instead of with huge gaps
            # between sentences that happen to be paragraph-separated in
            # the source metadata.
            import re as _re
            synopsis = _re.sub(r"\n\s*\n+", "\n", synopsis)
        self._synopsis_lbl.setText(synopsis if synopsis else "No synopsis available.")

        # Metadata grid
        while self._meta_grid.count():
            item = self._meta_grid.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        publisher = self._metadata_json.get("publisher") or self._details.get("publisher") or "\u2014"
        language = self._metadata_json.get("language") or self._details.get("language") or "\u2014"
        year = (self._metadata_json.get("date") or self._details.get("date") or "").strip()
        year = year[:4] if year and year[:4].isdigit() else (year or "\u2014")
        rows = [
            ("\U0001f4d8 Title", title or "\u2014"),
            ("\u270d\ufe0f Author", ", ".join(authors) if authors else "\u2014"),
            ("\U0001f3db\ufe0f Publisher", publisher or "\u2014"),
            ("\U0001f310 Language", language or "\u2014"),
            ("\U0001f4c5 Year", year or "\u2014"),
        ]
        # Estimate the horizontal space available to the value column.
        # ``meta_wrapper`` is between 380–540 px wide; the key column is
        # ~100 px ("\U0001f3db\ufe0f Publisher" is the widest) and the grid
        # has 14 px horizontal spacing. If the wrapper has already been laid
        # out we use its real width; otherwise fall back to the conservative
        # minimum so shrink-to-fit kicks in even on first paint.
        wrapper_w = self._meta_wrapper.width() if getattr(self, "_meta_wrapper", None) else 0
        if wrapper_w < 300:
            wrapper_w = 380
        val_avail_w = max(200, wrapper_w - 100 - 14 - 4)
        for i, (key, val) in enumerate(rows):
            k_lbl = QLabel(key)
            k_lbl.setProperty("class", "meta-k")
            k_lbl.setStyleSheet("color: #7a8599; font-size: 8.5pt;")
            v_lbl = QLabel()
            v_lbl.setProperty("class", "meta-v")
            v_lbl.setWordWrap(True)
            if i == 0:
                # Title row — dynamically shrink the font if the title would
                # otherwise wrap past the reserved box, and only fall back
                # to an ellipsis as a last resort. Matches the flash card
                # title behavior.
                title_box_h = 52
                v_lbl.setFixedHeight(title_box_h)
                v_lbl.setAlignment(Qt.AlignLeft | Qt.AlignTop)
                fitted_text, fitted_pt = _fit_title_text(
                    str(val),
                    avail_width=val_avail_w,
                    max_height=title_box_h,
                    base_pt=9.5,
                    base_font=v_lbl.font(),
                    min_pt=7.5,
                )
                v_lbl.setText(fitted_text)
                v_lbl.setToolTip(str(val))
                v_lbl.setStyleSheet(
                    f"color: #e0e0e0; font-size: {fitted_pt:g}pt; font-weight: bold;"
                )
            else:
                v_lbl.setText(str(val))
                v_lbl.setStyleSheet(
                    "color: #e0e0e0; font-size: 9.5pt; font-weight: bold;"
                )
            self._meta_grid.addWidget(k_lbl, i, 0, Qt.AlignTop | Qt.AlignLeft)
            self._meta_grid.addWidget(v_lbl, i, 1, Qt.AlignTop | Qt.AlignLeft)

        # Genres / tags — OPF subjects is a good default source; the translator
        # doesn't always distinguish between the two so we fan them out across
        # both rows when metadata.json has explicit genres/tags fields.
        subjects = list(self._details.get("subjects") or [])
        genres = list(self._metadata_json.get("genres") or []) or subjects[:3]
        tags = list(self._metadata_json.get("tags") or []) or subjects
        self._fill_chip_row(self._genres_layout, genres)
        self._fill_chip_row(self._tags_layout, tags)
        self._genres_heading.setVisible(bool(genres))
        self._genres_row.setVisible(bool(genres))
        self._tags_heading.setVisible(bool(tags))
        self._tags_row.setVisible(bool(tags))

        # Button availability — also dim the emoji + update tooltip so it's
        # obvious why the action isn't usable. (Previously only computed in
        # _on_details_ready; moved to the shared hero pass so the preview
        # already enables / disables the actions correctly.)
        resolved_out = self._resolve_output_folder_target()
        folder_ok = bool(resolved_out) and os.path.isdir(resolved_out)
        self._folder_btn.setEnabled(folder_ok)
        self._folder_btn.setCursor(Qt.PointingHandCursor if folder_ok else Qt.ForbiddenCursor)
        self._folder_btn_opacity.setOpacity(1.0 if folder_ok else 0.35)
        self._folder_btn.setToolTip(
            f"Open output folder:\n{resolved_out}" if folder_ok
            else "Output folder not available for this book"
        )
        resolved_src = self._resolve_source_file_target()
        source_ok = bool(resolved_src) and os.path.isfile(resolved_src)
        self._source_btn.setEnabled(source_ok)
        self._source_btn.setCursor(Qt.PointingHandCursor if source_ok else Qt.ForbiddenCursor)
        self._source_btn_opacity.setOpacity(1.0 if source_ok else 0.35)
        self._source_btn.setToolTip(
            f"Reveal source file:\n{resolved_src}" if source_ok
            else "Source file not found on disk"
        )

    @Slot(dict)
    def _on_preview_ready(self, payload: dict):
        """Render the hero row from Phase 1 (cover + metadata) immediately.

        Called before ``_on_details_ready`` so the user sees a fully
        populated cover + title + synopsis in milliseconds, without
        waiting for the spine's per-chapter HTML parse.
        """
        self._apply_hero_payload(payload)

    @Slot(dict)
    def _on_details_ready(self, payload: dict):
        new_chapters_info = payload.get("chapters_info", []) or []
        auto = bool(getattr(self, "_is_auto_refreshing", False))
        # On an auto-refresh, skip the expensive TOC rebuild when every
        # per-chapter signal is unchanged — otherwise we'd pointlessly
        # tear down + re-add hundreds of rows every 2 seconds.
        old_sig = [
            (c.get("index"), c.get("status"),
             c.get("translated_path") or "",
             c.get("translated_title") or "")
            for c in self._chapters_info
        ]
        new_sig = [
            (c.get("index"), c.get("status"),
             c.get("translated_path") or "",
             c.get("translated_title") or "")
            for c in new_chapters_info
        ]
        chapters_changed = old_sig != new_sig
        self._chapters_info = new_chapters_info
        # Re-apply the hero with the richer details returned by Phase 2
        # (OPF now includes real chapter titles, metadata_json may have
        # been re-loaded). Hero widgets are idempotent so this is safe.
        if not auto:
            self._apply_hero_payload(payload)

        # In-progress strip + reader-entry-point relabeling
        has_translated = any(c.get("translated_path") for c in self._chapters_info)
        if self._book.get("is_in_progress"):
            self._update_progress_strip()
            # When any chapter has been translated, the primary reader shows
            # the translated content (raw fallback for pending chapters). The
            # secondary "Read raw source" button remains for the raw view.
            if has_translated:
                self._start_btn.setText("\U0001f4d6  Read translated")
                self._raw_btn.show()
            else:
                self._start_btn.setText("\U0001f4d6  Read raw source")
                self._raw_btn.hide()
        else:
            self._progress_strip.hide()
            self._start_btn.setText("\U0001f4d6  Start reading")
            self._raw_btn.hide()

        # "Show raw titles" toggle is meaningful whenever at least one
        # chapter has BOTH a raw title and a distinct translated title
        # — covers in-progress books (titles harvested from source spine
        # + response files) AND Completed-tab library entries that have
        # a paired raw EPUB (titles harvested from both EPUBs by the
        # loader's library-raw pass).
        has_distinct_titles = any(
            (c.get("raw_title") or "")
            and (c.get("translated_title") or "")
            and (c.get("raw_title") or "") != (c.get("translated_title") or "")
            for c in self._chapters_info
        )
        raw_titles_applicable = has_distinct_titles
        # Use the intent flag (not the container's live visibility) —
        # :meth:`_populate_chapters` hides the container while the
        # loading placeholder is up, so ``isVisible()`` would spuriously
        # report "collapsed" here and the checkbox would never surface.
        section_expanded = bool(getattr(self, "_chap_section_expanded", True))
        # Skip toggling checkbox visibility during a silent auto-refresh
        # (the user may be interacting with it right now).
        if not auto:
            self._raw_titles_cb.setVisible(
                raw_titles_applicable and section_expanded
            )
        # Keep the flag accurate: if the toggle becomes inapplicable after
        # a refresh we don't want the checkbox state to stay "checked"
        # invisibly and force raw rendering on an unrelated book.
        if not raw_titles_applicable:
            if self._show_raw_titles:
                self._show_raw_titles = False
                self._raw_titles_cb.setChecked(False)

        # Chapter rows: on the initial load always rebuild; on a silent
        # auto-refresh only rebuild if the per-chapter signature drifted.
        if not auto or chapters_changed:
            self._populate_chapters(silent=auto)
        # Button availability is handled inside :meth:`_apply_hero_payload`
        # so both the preview and the final pass enable / disable actions
        # consistently. No extra work needed here.
        # Reset the auto-refresh flag so the next loader call (triggered
        # by anything other than :meth:`_auto_refresh_details`) goes
        # through the full reveal path.
        self._is_auto_refreshing = False

    def _update_progress_strip(self):
        """Refresh the "Translation in progress" strip.

        Uses ``self._show_special_files`` as the counting toggle so the
        user-visible "Show special files" checkbox drives the percentage
        (in addition to the chapter list filter). Rules:

        - Checkbox ON  → count every spine chapter, including cover /
          nav / toc, toward the denominator.
        - Checkbox OFF → exclude special files so a book doesn't sit at
          98/100 forever when the only untranslated entries are files
          the translator skips by design.

        The checkbox itself is initialized from the global
        ``translate_special_files`` setting (see :meth:`__init__`), so
        enabling that toggle in Other Settings cascades into the dialog.
        """
        if not self._book.get("is_in_progress"):
            self._progress_strip.hide()
            return
        # Gallery pages are unconditionally excluded (translator-generated,
        # not real source chapters). ``is_special`` is additionally used
        # to honor the user's "Show special files" checkbox.
        if self._show_special_files:
            progress_items = [c for c in self._chapters_info
                              if not c.get("is_gallery")]
        else:
            progress_items = [c for c in self._chapters_info
                              if not c.get("is_special")
                              and not c.get("is_gallery")]
        done = sum(1 for c in progress_items if c.get("status") == "completed")
        total = len(progress_items) or int(self._book.get("total_chapters", 0) or 0)
        # When the book has reached 100% translation, the card already
        # renders on the Completed tab without an "in progress" ribbon
        # (see :func:`split_output_folders_by_status`). Hide the details
        # strip too so the dialog doesn't contradict the card.
        translation_done = bool(total) and done >= total
        state = self._book.get("translation_state") or ""
        if translation_done or state == "completed":
            self._progress_strip.hide()
            return
        if total:
            pct = int(round((done / total) * 100))
            self._progress_strip.setText(
                f"\u23f3  Translation in progress \u2014 {done}/{total} chapters ({pct}%)"
            )
        else:
            self._progress_strip.setText("\u23f3  Translation in progress")
        self._progress_strip.show()

    def _fill_chip_row(self, layout: QHBoxLayout, values: list[str]):
        # Remove all but the trailing stretch
        while layout.count() > 1:
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        stretch_item = layout.takeAt(0)
        for v in values[:12]:
            if not v:
                continue
            chip = QLabel(v)
            chip.setProperty("class", "tag")
            chip.setStyleSheet(
                "color: #c8cbe0; background: #2a2a3e; "
                "border: 1px solid #3a3a5e; border-radius: 10px; "
                "padding: 3px 9px; font-size: 8.5pt;"
            )
            layout.addWidget(chip)
        if stretch_item is not None:
            layout.addItem(stretch_item)
        else:
            layout.addStretch()

    # Per-tick batch size for :meth:`_populate_chapters` — small enough
    # that one tick fits comfortably inside a Qt event-loop iteration
    # (so mouse / keyboard / scroll events aren't starved) but big
    # enough that a 700-chapter book finishes in a handful of ticks.
    _POPULATE_BATCH_SIZE = 40

    def _populate_chapters(self, silent: bool = False):
        """Rebuild the chapter row list, yielding between batches.

        A compiled EPUB with hundreds of chapters used to freeze the
        UI for multiple seconds while every :class:`_ChapterRow` widget
        was constructed synchronously. We now build the rows in batches
        of :data:`_POPULATE_BATCH_SIZE` per tick via a zero-interval
        ``QTimer``, so the Qt event loop keeps dispatching in between.

        In **initial** (non-silent) mode, the chapter container stays
        hidden behind the "⏳  Loading chapters…" placeholder while the
        batches land, so the user sees one steady loading state that
        flips to the fully built list in a single swap.

        In **silent** mode (auto-refresh), the loading placeholder is
        NOT shown — the existing rows stay visible while the new
        batches render on top. This avoids a visible flash every 2
        seconds when the auto-refresh timer ticks, and the user's
        scroll position / selection stays intact.

        Any pre-existing batch timer is cancelled first so a toggle
        flip (e.g. Show raw titles) doesn't double-render.
        """
        # Tear down before rebuilding and disable activation for the window
        # of time while we're mutating the list — this flag is read from
        # :meth:`_on_chapter_activated` to squash spurious clicks that
        # might land while Phase 2 is still populating rows.
        self._chapters_loaded = False
        # Cancel any in-flight batch timer from a previous call. Users
        # can retrigger this via the Show-raw-titles toggle or a scan
        # refresh while the previous batch is still rendering.
        populate_timer = getattr(self, "_populate_timer", None)
        if populate_timer is not None and populate_timer.isActive():
            populate_timer.stop()
        # Read user intent from the toggle flag rather than the
        # container's live visibility — the placeholder hides the
        # container as an implementation detail, so isVisible() would
        # incorrectly report "collapsed" on every load.
        target_visible = bool(getattr(self, "_chap_section_expanded", True))
        # Hide the container while we clear + rebuild it, and show the
        # sibling loading placeholder in its slot so the user isn't
        # staring at an empty space or flickering rows.
        # In silent mode (auto-refresh), keep the container visible so
        # the user doesn't see the existing rows disappear + the
        # placeholder flash between every 2s tick — they just see the
        # new rows replace the old ones in place.
        if not silent:
            try:
                self._chap_container.hide()
            except Exception:
                pass
            if getattr(self, "_chap_loading_lbl", None) is not None:
                # Only show the placeholder when the user currently expects
                # the chapter section to be visible. For a collapsed TOC we
                # leave it hidden too — otherwise clicking Refresh would
                # surprise the user with an unwanted reveal.
                self._chap_loading_lbl.setVisible(bool(target_visible))
        # Clear previous rows (placeholder now lives outside this layout).
        while self._chap_layout.count():
            item = self._chap_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()

        # Snapshot the state the worker uses so a mid-flight toggle
        # change can't cross-contaminate the rendering flavor.
        self._populate_infos = list(self._chapters_info)
        self._populate_idx = 0
        self._populate_show_raw_title = bool(self._show_raw_titles)
        self._populate_target_visible = bool(target_visible)
        self._populate_silent = bool(silent)

        if not self._populate_infos:
            self._chap_layout.addStretch()
            self._apply_chapter_filter(self._toc_search.text())
            self._update_toc_toggle_label()
            # Nothing to build — flip the loading placeholder back off
            # and reveal the (empty) container if the user expects it.
            if not silent:
                if getattr(self, "_chap_loading_lbl", None) is not None:
                    self._chap_loading_lbl.hide()
                if target_visible:
                    self._chap_container.show()
            self._chapters_loaded = True
            return

        if populate_timer is None:
            populate_timer = QTimer(self)
            populate_timer.setSingleShot(False)
            populate_timer.setInterval(0)  # yield every event-loop tick
            populate_timer.timeout.connect(self._populate_chapters_batch_tick)
            self._populate_timer = populate_timer

        # Kick the first batch off asynchronously so the loading
        # placeholder actually paints before we start building rows
        # behind it. Starting the timer triggers the first tick on the
        # next event-loop iteration, by which point the UI has already
        # had a chance to swap in the "⏳ Loading chapters…" label.
        populate_timer.start()

    def _populate_chapters_batch_tick(self):
        """Render the next :data:`_POPULATE_BATCH_SIZE` chapter rows.

        Stops the driving ``QTimer`` and finalizes the TOC state once
        every row has been appended to the layout.
        """
        infos = getattr(self, "_populate_infos", None) or []
        start = int(getattr(self, "_populate_idx", 0) or 0)
        end = min(start + self._POPULATE_BATCH_SIZE, len(infos))
        show_raw_title = bool(getattr(self, "_populate_show_raw_title", False))
        selected_idx = self._selected_chapter_idx
        for i in range(start, end):
            info = infos[i]
            row = _ChapterRow(
                info,
                parent=self._chap_container,
                show_raw_title=show_raw_title,
            )
            row.activated.connect(self._on_chapter_activated)
            row.clicked.connect(self._on_chapter_clicked)
            if info.get("index") == selected_idx:
                row.set_selected(True)
            self._chap_layout.addWidget(row)
        self._populate_idx = end
        if end >= len(infos):
            # Final tick: cap the layout with the stretch, apply any
            # active filter, refresh the "(done/total)" TOC header, and
            # unlock activation so double-clicks open the reader.
            timer = getattr(self, "_populate_timer", None)
            if timer is not None and timer.isActive():
                timer.stop()
            self._chap_layout.addStretch()
            self._apply_chapter_filter(self._toc_search.text())
            self._update_toc_toggle_label()
            # Swap: hide the loading placeholder and reveal the fully
            # built chapter container in one go, so the user sees the
            # whole list appear at once instead of the per-batch pop-in.
            # For silent auto-refresh runs the container was never hidden
            # and the placeholder was never shown, so the reveal is a no-op.
            if not bool(getattr(self, "_populate_silent", False)):
                if getattr(self, "_chap_loading_lbl", None) is not None:
                    self._chap_loading_lbl.hide()
                if getattr(self, "_populate_target_visible", True):
                    self._chap_container.show()
            self._chapters_loaded = True

    def _on_chapter_clicked(self, idx: int):
        """Update the single-select focus to the clicked chapter row."""
        self._selected_chapter_idx = idx
        for i in range(self._chap_layout.count()):
            w = self._chap_layout.itemAt(i).widget()
            if isinstance(w, _ChapterRow):
                w.set_selected(w.info.get("index") == idx)

    def _on_chapter_activated(self, idx: int):
        """Open the reader at *idx* only if the chapter list is fully loaded.

        Chapter rows emit ``activated`` on double-click (see
        :class:`_ChapterRow`), but if a click manages to arrive while the
        spine is still being populated we silently ignore it rather than
        opening a reader positioned into a half-built list.
        """
        if not self._chapters_loaded:
            return
        self._open_reader(initial_chapter=idx)

    def _visible_counts(self) -> tuple[int, int]:
        """Return (done, total) considering the special-files toggle.

        Gallery pages are unconditionally excluded — they're
        translator-generated artefacts, not real source chapters.
        """
        if self._show_special_files:
            items = [c for c in self._chapters_info
                     if not c.get("is_gallery")]
        else:
            items = [c for c in self._chapters_info
                     if not c.get("is_special") and not c.get("is_gallery")]
        total = len(items)
        done = sum(1 for c in items if c.get("status") == "completed")
        return done, total

    def _has_progress_context(self) -> bool:
        """True when at least one chapter has a non-empty translation status."""
        return any((c.get("status") or "") for c in self._chapters_info)

    def _update_toc_toggle_label(self):
        done, total = self._visible_counts()
        # Use the intent flag (not the container's live visibility) so
        # the arrow glyph doesn't flip to ▶ while the loading
        # placeholder is masking the container during a batched build.
        prefix = "\u25bc  Chapters" if self._chap_section_expanded else "\u25b6  Chapters"
        if not total:
            suffix = "  (\u2014)"
        elif self._has_progress_context():
            suffix = f"  ({done}/{total})"
        else:
            # No progress file anywhere — just show the total count without a
            # misleading completed/total fraction.
            suffix = f"  ({total})"
        self._toc_toggle.setText(prefix + suffix)

    def _apply_chapter_filter(self, text: str):
        needle = (text or "").strip().lower()
        for i in range(self._chap_layout.count()):
            w = self._chap_layout.itemAt(i).widget()
            if not isinstance(w, _ChapterRow):
                continue
            info = w.info
            # Hide special files unless the toggle is on.
            if not self._show_special_files and info.get("is_special"):
                w.setVisible(False)
                continue
            if not needle:
                w.setVisible(True)
                continue
            hay = " ".join(str(x) for x in (info.get("raw_title", ""),
                                            info.get("translated_title", ""),
                                            info.get("filename", ""))).lower()
            w.setVisible(needle in hay)
        self._update_toc_toggle_label()

    def _on_special_files_toggled(self, checked: bool):
        self._show_special_files = bool(checked)
        # Persist for next time this dialog (or another book) opens.
        try:
            self._config["epub_details_show_special_files"] = self._show_special_files
        except Exception:
            pass
        self._apply_chapter_filter(self._toc_search.text())
        # The "Translation in progress — X/Y" strip now reflects this
        # toggle's state too, so refresh its fraction whenever the user
        # flips the checkbox.
        self._update_progress_strip()

    def _on_raw_titles_toggled(self, checked: bool):
        """Swap every chapter row between translated-title and raw-title mode."""
        new_value = bool(checked)
        if new_value == self._show_raw_titles:
            return
        self._show_raw_titles = new_value
        try:
            self._config["epub_details_show_raw_titles"] = self._show_raw_titles
        except Exception:
            pass
        # Re-populate the chapter list so every _ChapterRow is rebuilt with
        # the new primary-title policy. Cheaper than retrofitting the rows
        # in place and matches the pattern used for other toggles.
        if self._chapters_loaded:
            self._populate_chapters()

    def _toggle_chapters(self):
        # Flip user intent, then apply it to the actual widgets.
        show = not bool(getattr(self, "_chap_section_expanded", True))
        self._chap_section_expanded = show
        self._chap_container.setVisible(show)
        # Loading placeholder never survives a manual toggle — if the
        # user collapsed mid-load, it makes no sense to keep showing it.
        if getattr(self, "_chap_loading_lbl", None) is not None:
            self._chap_loading_lbl.hide()
        self._toc_search.setVisible(show)
        self._special_cb.setVisible(show)
        # Mirror the applicability rule used in :meth:`_on_details_ready`
        # so the toggle surfaces for any book — in-progress OR library
        # — whose chapter rows carry distinct raw + translated titles.
        has_distinct_titles = any(
            (c.get("raw_title") or "")
            and (c.get("translated_title") or "")
            and (c.get("raw_title") or "") != (c.get("translated_title") or "")
            for c in self._chapters_info
        )
        self._raw_titles_cb.setVisible(show and has_distinct_titles)
        self._update_toc_toggle_label()

    @Slot(str)
    def _on_details_error(self, message: str):
        self._synopsis_lbl.setText(f"Failed to load book details: {message}")

    # -- Actions ------------------------------------------------------------

    def _build_translated_overlay(self) -> tuple[dict[str, dict], list[str]]:
        """Return (overlay, extra_image_dirs) for a translated reader view.

        The overlay maps the source chapter's filename (lowercased basename)
        → translated HTML path + title so the reader can swap source content
        with translated content in place. Keying by filename (rather than
        index) avoids ordering/skip mismatches between the reader's loader
        (manifest order, filters short chapters) and our own spine-based
        parser. Image directories let the reader resolve assets that only
        exist in the translator's output.
        """
        overlay: dict[str, dict] = {}
        for ci in self._chapters_info:
            path = ci.get("translated_path") or ""
            if not path or not os.path.isfile(path):
                continue
            filename = ci.get("filename") or ""
            if not filename:
                continue
            key = os.path.basename(filename).lower()
            if not key:
                continue
            overlay[key] = {
                "path": path,
                "title": ci.get("translated_title") or "",
            }
        extra_dirs: list[str] = []
        output_folder = self._book.get("output_folder")
        if output_folder and os.path.isdir(output_folder):
            for sub in ("images", "translated_images"):
                candidate = os.path.join(output_folder, sub)
                if os.path.isdir(candidate):
                    extra_dirs.append(candidate)
        return overlay, extra_dirs

    def _open_reader(self, initial_chapter: int | None = None, raw_only: bool = False):
        """Dispatch to the appropriate viewer based on the resolved source type.

        EPUB sources — whether ``book['path']`` is the EPUB itself (completed
        cards) or the output folder that holds one (in-progress cards) — use
        the integrated :class:`EpubReaderDialog` with the optional translated
        overlay. Only TXT / PDF / HTML / image workspaces fall through to the
        OS default viewer.
        """
        try:
            book_path = self._book.get("path", "") or ""
            raw_source = self._book.get("raw_source_path", "") or ""
            compiled = self._book.get("compiled_output_path", "") or ""

            def _is_epub_file(p: str) -> bool:
                return bool(p) and p.lower().endswith(".epub") and os.path.isfile(p)

            # Resolve the EPUB to hand to EpubReaderDialog. For in-progress
            # cards ``book['path']`` is the OUTPUT FOLDER (not a file), so we
            # MUST consult ``raw_source_path`` / ``compiled_output_path`` too.
            # Priority:
            #   * raw_only or is_in_progress → raw_source first (that's the
            #     reader base the translated overlay sits on top of).
            #   * completed / library        → book_path first (compiled or
            #     library EPUB is what the user wants to read).
            if raw_only or self._book.get("is_in_progress"):
                epub_candidates = [raw_source, book_path, compiled]
            else:
                epub_candidates = [book_path, raw_source, compiled]
            epub_for_reader = next(
                (p for p in epub_candidates if _is_epub_file(p)), ""
            )

            if epub_for_reader:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                QApplication.processEvents()
                overlay: dict[str, dict] = {}
                extra_dirs: list[str] = []
                window_title = None
                # Only offer the translated overlay when the book is actually an
                # in-progress novel with at least one translated chapter on disk.
                if not raw_only and self._book.get("is_in_progress"):
                    overlay, extra_dirs = self._build_translated_overlay()
                    if overlay:
                        # Derive the displayed title from the metadata.json /
                        # OPF title, falling back to the book's name.
                        window_title = (self._metadata_json.get("title")
                                        or self._details.get("title")
                                        or self._book.get("name"))
                        if window_title:
                            window_title = f"{window_title} (Translated)"
                # Translate the spine-index initial_chapter into a filename so the
                # reader resolves it against its own (manifest-ordered) chapter
                # list. This also prevents off-by-one jumps when the source EPUB
                # has nav/toc items that are skipped by the reader's loader.
                initial_filename = None
                if isinstance(initial_chapter, int) and 0 <= initial_chapter < len(self._chapters_info):
                    initial_filename = self._chapters_info[initial_chapter].get("filename") or None
                # Completed-tab mode (no overlay, has raw source): let the
                # reader flip between the compiled EPUB (book_path) and
                # the resolved raw source. Skipped when an overlay is
                # active — overlay mode handles the Raw toggle by
                # swapping in-memory chapter lists instead of reloading.
                alt_for_reader = ""
                if (not overlay and not raw_only
                        and not self._book.get("is_in_progress")
                        and raw_source
                        and os.path.isfile(raw_source)
                        and raw_source.lower().endswith(".epub")):
                    try:
                        if os.path.normcase(os.path.abspath(raw_source)) != \
                                os.path.normcase(os.path.abspath(epub_for_reader)):
                            alt_for_reader = raw_source
                    except Exception:
                        alt_for_reader = ""
                reader = EpubReaderDialog(
                    epub_for_reader,
                    config=self._config,
                    parent=self,
                    initial_chapter=initial_chapter,
                    initial_chapter_filename=initial_filename,
                    translated_overlay=overlay or None,
                    extra_image_dirs=extra_dirs or None,
                    window_title=window_title,
                    # Propagate the dialog's current toggle so the reader's
                    # TOC matches what the Book Details chapter list
                    # shows (cover / nav / toc hidden when this is off).
                    show_special_files=self._show_special_files,
                    alt_epub_path=alt_for_reader or None,
                )
                QApplication.restoreOverrideCursor()
                reader.setModal(False)
                reader.setAttribute(Qt.WA_DeleteOnClose)
                self._active_reader = reader
                reader.show()
                return

            # No EPUB base resolvable — the workspace is TXT / PDF / HTML /
            # image. Hand off to the OS default viewer with a concrete file.
            chapter_translated = ""
            if (not raw_only
                    and isinstance(initial_chapter, int)
                    and 0 <= initial_chapter < len(self._chapters_info)):
                tp = self._chapters_info[initial_chapter].get("translated_path", "") or ""
                if tp and os.path.isfile(tp):
                    chapter_translated = tp

            book_path_is_file = bool(book_path) and os.path.isfile(book_path)
            target = ""
            if raw_only:
                if raw_source and os.path.isfile(raw_source):
                    target = raw_source
                elif book_path_is_file:
                    target = book_path
            else:
                if chapter_translated:
                    target = chapter_translated
                elif compiled and os.path.isfile(compiled):
                    target = compiled
                elif book_path_is_file:
                    target = book_path
                elif raw_source and os.path.isfile(raw_source):
                    target = raw_source

            if not target or not os.path.isfile(target):
                QMessageBox.warning(
                    self, "Error",
                    "No readable file is available for this book.",
                )
                return

            self._open_with_system_viewer(target)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            logger.error("Could not open reader from details: %s\n%s", exc, traceback.format_exc())
            QMessageBox.warning(self, "Error", f"Could not open file:\n{exc}")

    def _open_with_system_viewer(self, path: str):
        """Open *path* with the OS default handler.

        Mirrors the PDF/TXT branch inside
        :meth:`EpubLibraryDialog._on_card_clicked` so the two entry points
        stay behaviorally consistent.
        """
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        _no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        try:
            if sys.platform == "win32":
                if ext == "txt":
                    _npp_paths = [
                        r"C:\Program Files\Notepad++\notepad++.exe",
                        r"C:\Program Files (x86)\Notepad++\notepad++.exe",
                    ]
                    _npp = next((p for p in _npp_paths if os.path.exists(p)), None)
                    if _npp:
                        subprocess.Popen([_npp, path], creationflags=_no_window)
                    else:
                        subprocess.Popen(["notepad.exe", path], creationflags=_no_window)
                else:
                    os.startfile(path)
            elif sys.platform == "darwin":
                if ext == "txt" and shutil.which("code"):
                    subprocess.Popen(["code", path])
                else:
                    subprocess.Popen(["open", path])
            else:
                if ext == "txt":
                    _editors = ["gedit", "kate", "code", "mousepad", "xed", "pluma"]
                    _editor = next((e for e in _editors if shutil.which(e)), "xdg-open")
                    subprocess.Popen([_editor, path])
                else:
                    subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            logger.error("Could not open file %s: %s\n%s", path, exc, traceback.format_exc())
            QMessageBox.warning(self, "Error", f"Could not open file:\n{exc}")

    def _resolve_output_folder_target(self) -> str:
        """Return the output-folder path the 📁 button should open, or "".

        Thin wrapper around :func:`_resolve_book_output_folder` so the
        enable / tooltip state and the click handler share one
        resolver with the card context menu (the two previously drifted
        out of sync: Book Details consulted the origins registry but
        the context menu fell back to the book's containing folder,
        which for library-organized entries was ``Library/Translated``
        instead of the original output folder).
        """
        return _resolve_book_output_folder(self._book)

    def _resolve_source_file_target(self) -> str:
        """Return the raw source file path the 🔗 button should reveal.

        Mirrors the enable / click path for the source button: consult
        ``book['raw_source_path']`` first, then re-run
        :func:`_find_raw_source_for_library_epub` for library entries
        whose cached dict was built before ``origins['pairs']`` was
        populated. Caches any fresh resolution back onto ``self._book``
        so the reader's Raw toggle and other actions in the same dialog
        pick it up without re-computing.
        """
        path = self._book.get("raw_source_path", "") or ""
        if path and os.path.isfile(path):
            return path
        if self._book.get("in_library"):
            lib_path = self._book.get("path", "") or ""
            try:
                resolved = _find_raw_source_for_library_epub(lib_path)
            except Exception:
                resolved = ""
                logger.debug("Source-file library lookup failed: %s",
                             traceback.format_exc())
            if resolved and os.path.isfile(resolved):
                self._book["raw_source_path"] = resolved
                return resolved
        fallback = self._book.get("path", "") or ""
        if fallback and os.path.isfile(fallback):
            return fallback
        return ""

    def _open_output_folder(self):
        """Open the book's output folder in the system file explorer."""
        folder = self._resolve_output_folder_target()
        if folder and os.path.isdir(folder):
            _open_folder_in_explorer(folder)

    def _reveal_source(self):
        """Reveal the raw source file in the system file explorer."""
        path = self._resolve_source_file_target()
        if path and os.path.isfile(path):
            _open_folder_in_explorer(path)

    # -- Qt lifecycle --------------------------------------------------------

    def closeEvent(self, event):
        if self._loader is not None:
            try:
                self._loader.quit()
                self._loader.wait(200)
            except Exception:
                pass
        super().closeEvent(event)


class _ChapterRow(QFrame):
    """One row in the Book Details chapter list.

    Displays translated title + filename when the chapter has been translated;
    otherwise shows the raw source title with a muted "pending" label.
    A single click selects the row (visual focus only); a double click
    activates it and opens the reader. This mirrors the flash-card UX in
    :class:`EpubLibraryDialog`.
    """
    activated = Signal(int)
    # Emitted on a single left-click so the parent dialog can update the
    # "currently focused" chapter row. Separate from :attr:`activated` so
    # selecting a row doesn't also open the reader.
    clicked = Signal(int)

    # Object-name selectors for the same reason as :class:`_BookCard`:
    # they match this exact widget reliably across Qt/PySide versions.
    # Borders are 2 px in both states to avoid layout shift on selection.
    _BASE_STYLE = (
        "QFrame#chapterRow { background: #1a1a2a; border: 2px solid #242438; border-radius: 6px; }"
        "QFrame#chapterRow:hover { border: 2px solid #6c63ff; background: #232340; }"
    )
    _SELECTED_STYLE = (
        "QFrame#chapterRow { background: #2a2d5a; border: 2px solid #a097ff; border-radius: 6px; }"
        "QFrame#chapterRow:hover { border: 2px solid #c0b8ff; background: #343670; }"
    )

    def __init__(self, info: dict, parent=None, show_raw_title: bool = False):
        super().__init__(parent)
        self.info = info
        self._selected = False
        self.setObjectName("chapterRow")
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(
            "Click to select — double-click to open this chapter in the reader"
        )
        self.setStyleSheet(self._BASE_STYLE)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        status = info.get("status", "") or ""
        translated = info.get("translated_title") or ""
        raw = info.get("raw_title") or ""
        # When the caller asks for raw titles explicitly, we always show the
        # source-language title as the primary text (falling back to filename
        # if the spine parse couldn't extract one). Otherwise the normal rule
        # applies: translated title when the chapter is complete, raw title
        # while it's still pending.
        if show_raw_title:
            primary = QLabel(raw or info.get("filename", ""))
            primary.setProperty("class", "raw")
            primary.setStyleSheet("color: #c8cbe0; font-size: 10pt; font-weight: bold;")
        elif translated and status == "completed":
            primary = QLabel(translated)
            primary.setProperty("class", "translated")
            primary.setStyleSheet("color: #e0e0e0; font-size: 10pt; font-weight: bold;")
        else:
            primary = QLabel(raw or info.get("filename", ""))
            primary.setProperty("class", "raw")
            primary.setStyleSheet("color: #c8cbe0; font-size: 10pt; font-weight: bold;")
        primary.setWordWrap(True)
        # If both titles exist and we're in raw-titles mode, expose the
        # translated version as a tooltip so the user still has one-click
        # access to it without flipping the checkbox.
        if show_raw_title and translated and translated != raw:
            primary.setToolTip(f"Translated: {translated}")
        elif (not show_raw_title) and translated and status == "completed" and raw and raw != translated:
            primary.setToolTip(f"Raw: {raw}")
        text_col.addWidget(primary)

        sub = QLabel(info.get("filename", ""))
        sub.setProperty("class", "filename")
        sub.setStyleSheet("color: #8a8fa8; font-size: 8.5pt; font-family: 'Consolas','Menlo',monospace;")
        text_col.addWidget(sub)
        layout.addLayout(text_col, 1)

        badge = None
        # Gallery rows render without any badge — they're auto-generated
        # artefacts, not real source chapters.
        is_gallery = bool(info.get("is_gallery"))
        if is_gallery:
            status = ""
        if status == "completed":
            badge = QLabel("\u2714 Translated")
            badge.setStyleSheet(
                "color: #7ec87e; background: rgba(126, 200, 126, 0.12);"
                " border: 1px solid #7ec87e; border-radius: 10px;"
                " padding: 2px 10px; font-size: 8pt; font-weight: bold;"
            )
        elif status in ("failed", "qa_failed"):
            badge = QLabel("\u26a0 " + ("QA failed" if status == "qa_failed" else "Failed"))
            badge.setStyleSheet(
                "color: #ff9e6d; background: rgba(255, 158, 109, 0.12);"
                " border: 1px solid #ff9e6d; border-radius: 10px;"
                " padding: 2px 10px; font-size: 8pt; font-weight: bold;"
            )
        elif status == "in_progress":
            badge = QLabel("\u23f3 Working")
            badge.setStyleSheet(
                "color: #ffd166; background: rgba(255, 209, 102, 0.12);"
                " border: 1px solid #ffd166; border-radius: 10px;"
                " padding: 2px 10px; font-size: 8pt; font-weight: bold;"
            )
        elif status == "pending":
            badge = QLabel("Pending")
            badge.setStyleSheet(
                "color: #7a8599; background: #2a2a3e;"
                " border: 1px solid #3a3a5e; border-radius: 10px;"
                " padding: 2px 10px; font-size: 8pt;"
            )
        # Empty/unknown status → no badge at all (book has no progress context).
        if badge is not None:
            layout.addWidget(badge, 0, Qt.AlignRight)

    def set_selected(self, selected: bool) -> None:
        """Toggle the row's "focused" visual state (purple border/background)."""
        new_value = bool(selected)
        if new_value == self._selected:
            return
        self._selected = new_value
        self.setStyleSheet(self._SELECTED_STYLE if new_value else self._BASE_STYLE)
        self.update()

    @property
    def selected(self) -> bool:
        return self._selected

    def mousePressEvent(self, event):
        # Single left-click = focus/select. Actual activation happens on
        # double-click via :meth:`mouseDoubleClickEvent`.
        if event.button() == Qt.LeftButton:
            self.clicked.emit(int(self.info.get("index", 0)))
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        # Opening the reader requires a full double-click so accidental
        # single clicks (e.g. while scrolling) don't launch anything.
        if event.button() == Qt.LeftButton:
            self.activated.emit(int(self.info.get("index", 0)))
        super().mouseDoubleClickEvent(event)


# ---------------------------------------------------------------------------
# EPUB Reader — background loader thread
# ---------------------------------------------------------------------------

if _HAS_WEBENGINE:
    class _WheelCapturingView(QWebEngineView):
        """QWebEngineView subclass that surfaces wheel events as a signal.

        Out of the box wheel events on a QWebEngineView are consumed
        by the internal Chromium render widget before they ever reach
        Python. The usual Qt workaround is to install an event filter
        on every child widget the view spawns (they host the actual
        render surface). We do that here plus re-run the install for
        any children added after the first page load (Qt creates those
        lazily). Emits ``wheel_scrolled(delta_y, modifiers)`` for the
        reader dialog to translate into page-turn / zoom behavior.
        """
        wheel_scrolled = Signal(int, object)

        def __init__(self, parent=None):
            super().__init__(parent)
            self._install_filter_on_children()

        def childEvent(self, event):
            from PySide6.QtCore import QEvent
            if event.type() == QEvent.ChildAdded:
                child = event.child()
                try:
                    if hasattr(child, "installEventFilter"):
                        child.installEventFilter(self)
                except Exception:
                    pass
            super().childEvent(event)

        def _install_filter_on_children(self):
            from PySide6.QtWidgets import QWidget
            for c in self.findChildren(QWidget):
                try:
                    c.installEventFilter(self)
                except Exception:
                    pass

        def eventFilter(self, obj, event):
            from PySide6.QtCore import QEvent
            if event.type() == QEvent.Wheel:
                delta = event.angleDelta().y()
                if delta != 0:
                    self.wheel_scrolled.emit(delta, event.modifiers())
                    # Consume the event so Chromium doesn't also
                    # scroll / zoom the page in parallel.
                    return True
            return super().eventFilter(obj, event)


class _EpubCacheLoaderThread(QThread):
    """Read the pickled EPUB cache off the UI thread.

    ``pickle.load`` on a large cache (hundreds of chapters + embedded
    image bytes) blocks the Qt event loop for up to a second, visibly
    freezing the Halgakos spinner that's supposed to be animating while
    the user waits. Running the read in a dedicated QThread keeps the
    spinner smooth and lets the main thread dispatch paint events
    throughout.

    Emits ``hit(chapters, images, filenames)`` on a successful cache
    read, or ``miss()`` when the cache is absent / invalid / empty and
    the caller should fall back to the full :class:`_EpubLoaderThread`
    re-parse.
    """
    hit = Signal(object, object, list)
    miss = Signal()

    def __init__(self, epub_path: str, show_special_files: bool = True,
                 parent=None):
        super().__init__(parent)
        self._epub_path = epub_path
        self._show_special_files = bool(show_special_files)

    def run(self):
        try:
            cached = _load_epub_cache(
                self._epub_path,
                show_special_files=self._show_special_files,
            )
        except Exception:
            logger.debug("Cache load failed in worker: %s",
                         traceback.format_exc())
            cached = None
        if cached:
            chapters, images, filenames = cached
            self.hit.emit(chapters, images, list(filenames or []))
        else:
            self.miss.emit()


class _OverlayMergeThread(QThread):
    """Off-UI-thread merge of the reader's loaded chapters against the
    translated-chapter overlay and any extra image directories.

    Both operations are file-I/O bound (reading per-chapter HTML from
    disk, walking image subfolders) and used to run synchronously in
    :meth:`EpubReaderDialog._on_epub_loaded_from_cache`, making an
    in-progress book with hundreds of translated chapters visibly lag
    the Qt event loop on open. Running them in a dedicated QThread —
    plus a ``ThreadPoolExecutor`` inside for concurrent small reads —
    keeps the UI responsive while the worker hits the disk.

    Emits ``done(overlaid_chapters, merged_images, overlay_applied)``
    once all reads finish. The container payloads ride as ``object``
    (plain Python references) rather than ``list`` / ``dict`` so
    PySide6 doesn't try to marshal them into ``QVariantList`` /
    ``QVariantMap`` — that pathway fails slot lookup on the main
    thread with ``AttributeError: Slot 'EpubReaderDialog::
    _on_overlay_merge_done(QVariantList,QVariantMap,bool)' not
    found``, even though the matching Python method exists. Matches
    the pattern used by :class:`_EpubCacheLoaderThread.hit`.
    """
    done = Signal(object, object, bool)

    def __init__(self, raw_chapters, images, filenames, overlay_map,
                 extra_image_dirs, parent=None):
        super().__init__(parent)
        self._raw_chapters = list(raw_chapters or [])
        self._images = dict(images or {})
        self._filenames = list(filenames or [])
        self._overlay = dict(overlay_map or {})
        self._extra_dirs = list(extra_image_dirs or [])

    def run(self):
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
                fname = filenames[idx] if idx < len(filenames) else ""
                key = os.path.basename(fname).lower() if fname else ""
                ov = overlay.get(key) if key else None
                if not ov:
                    return (idx, title, content, False)
                path = ov.get("path") or ""
                if not (path and os.path.isfile(path)):
                    return (idx, title, content, False)
                try:
                    with open(path, "rb") as f:
                        data = f.read()
                except OSError:
                    logger.debug("Overlay read failed: %s",
                                 traceback.format_exc())
                    return (idx, title, content, False)
                translated_html = data.decode("utf-8", errors="replace")
                new_title = title
                if ov.get("title"):
                    new_title = str(ov["title"])
                return (idx, new_title, translated_html, True)

            try:
                max_workers = min(32, max(4, (os.cpu_count() or 2) * 4))
                items = list(enumerate(raw))
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    results = list(pool.map(_fetch_overlay, items))
            except Exception:
                logger.debug("Overlay parallel merge failed, "
                             "falling back to sequential: %s",
                             traceback.format_exc())
                results = [_fetch_overlay(item)
                           for item in enumerate(raw)]

            merged = [None] * len(raw)
            for idx, title, content, applied in results:
                merged[idx] = (title, content)
                if applied:
                    overlay_applied = True
            overlaid = merged

        # --- Extra image-directory scan ---
        # Augments the EPUB's own image table with files discovered in
        # the translator's output folder (e.g. ``images/``,
        # ``translated_images/``) so overlaid chapters can resolve any
        # assets the compiled EPUB doesn't carry.
        images = self._images
        if self._extra_dirs:
            merged_imgs = dict(images)
            _img_exts = (".jpg", ".jpeg", ".png", ".gif",
                         ".webp", ".svg", ".bmp")
            for dir_path in self._extra_dirs:
                if not dir_path or not os.path.isdir(dir_path):
                    continue
                try:
                    for entry in os.scandir(dir_path):
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        if not entry.name.lower().endswith(_img_exts):
                            continue
                        try:
                            with open(entry.path, "rb") as f:
                                data = f.read()
                        except OSError:
                            continue
                        merged_imgs.setdefault(entry.name, data)
                        rel = os.path.basename(dir_path) + "/" + entry.name
                        merged_imgs.setdefault(rel, data)
                        merged_imgs.setdefault(
                            "images/" + entry.name, data)
                except (PermissionError, OSError):
                    continue
            images = merged_imgs

        self.done.emit(overlaid, images, overlay_applied)


class _EpubLoaderThread(QThread):
    """Load the EPUB in a background thread and write result to cache.

    Emitting large binary data (images) through Qt signals across threads
    can crash the GUI.  Instead, write to a cache file and emit a
    lightweight success signal.
    """
    done = Signal()          # success — data available via cache
    error = Signal(str)

    def __init__(self, epub_path: str, parent=None,
                 show_special_files: bool = True):
        super().__init__(parent)
        self._epub_path = epub_path
        # When False, spine items flagged as "special" (no-digit stem —
        # cover / nav / toc / info / message / …) are excluded from the
        # chapter list so the TOC mirrors what the translator would act
        # on with ``translate_special_files`` off. Baked into the cache
        # key so on / off variants don't collide.
        self._show_special_files = bool(show_special_files)

    def run(self):
        try:
            import ebooklib
            from ebooklib import epub as epub_mod
            from bs4 import BeautifulSoup

            book = epub_mod.read_epub(self._epub_path, options={"ignore_ncx": True})

            images: dict[str, bytes] = {}
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_IMAGE:
                    images[item.get_name()] = item.get_content()
                    basename = os.path.basename(item.get_name())
                    if basename not in images:
                        images[basename] = item.get_content()

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
                    _add_item(item, True)

            # Pass 3: extension-only sweep across the whole manifest
            # (non-authoritative). Only runs when we found nothing (or just
            # a single cover-like item) via the authoritative passes, so
            # well-formed EPUBs don't pay any extra cost here.
            if len(chapter_items) <= 1:
                for item in book.get_items():
                    _add_item(item, False)

            chapters: list[tuple[str, str]] = []
            filenames: list[str] = []
            for item, authoritative in chapter_items:
                try:
                    # "Show special files" toggle: when OFF, drop named
                    # non-chapter pages (cover.xhtml, nav.xhtml, toc.xhtml,
                    # info.html, …) so the TOC matches what the
                    # translator considers real chapters. This still
                    # respects spine ordering for the chapters that DO
                    # survive — we just prune the specials.
                    if not self._show_special_files and _is_special_spine_item(
                            item.get_name() or ""):
                        continue

                    content = item.get_content().decode("utf-8", errors="replace")
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
                        continue
                    # Authoritative-but-totally-empty items (no text AND no
                    # images AND no links) are still dropped — they're
                    # almost always accidental spine entries (e.g. a
                    # placeholder that never got populated).
                    if authoritative and not text:
                        has_img = bool(soup.find("img"))
                        has_svg = bool(soup.find("svg"))
                        has_link = bool(soup.find("a"))
                        if not (has_img or has_svg or has_link):
                            continue

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
                        title = os.path.splitext(os.path.basename(item.get_name()))[0]
                        title = title.replace("_", " ").replace("-", " ").title()
                    if len(title) > 50:
                        title = title[:47] + "\u2026"
                    chapters.append((title, content))
                    # Record the source item name (e.g. 'OEBPS/chapter0001.xhtml')
                    # in parallel so downstream code can correlate reader
                    # chapters with spine filenames.
                    filenames.append(item.get_name() or "")
                except Exception:
                    logger.debug("Skipped chapter: %s", traceback.format_exc())

            # Write to cache (avoids emitting large data through Qt signals).
            # Key-scoped by the Show-special-files state so the two
            # variants don't overwrite each other.
            _save_epub_cache(
                self._epub_path, chapters, images, filenames,
                show_special_files=self._show_special_files,
            )
            self.done.emit()
        except Exception as exc:
            logger.error("EPUB load error: %s\n%s", exc, traceback.format_exc())
            self.error.emit(f"{exc}\n\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# EPUB Reader Dialog
# ---------------------------------------------------------------------------

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


def _persist_config_via_parent(widget) -> bool:
    """Walk *widget*'s parent chain looking for a ``save_config`` method and call it.

    Both :class:`EpubLibraryDialog` and :class:`EpubReaderDialog` mutate
    the shared ``config`` dict on close (layout mode, font size, sort
    order, etc.), but that dict only lives in memory until the
    translator's main window persists it to ``config.json``. On a normal
    quit the main window's own save runs first, but users who close the
    app by killing the tab — or who simply want their reader settings
    to outlive a crash — would lose the just-written values. Walking up
    to the :class:`TranslatorGUI` (or whichever parent exposes
    ``save_config``) and triggering a silent save here makes the in-memory
    changes durable as soon as the dialog closes.

    Returns True when a save ran, False when no eligible parent was
    found or the save itself raised. Uses ``show_message=False`` when
    the signature accepts it so the silent-save path doesn't pop an
    unexpected message box.
    """
    try:
        parent = widget.parent() if hasattr(widget, "parent") else None
    except Exception:
        parent = None
    while parent is not None:
        save = getattr(parent, "save_config", None)
        if callable(save):
            try:
                try:
                    save(show_message=False)
                except TypeError:
                    save()
                return True
            except Exception:
                logger.debug("Parent save_config failed: %s",
                             traceback.format_exc())
                return False
        try:
            parent = parent.parent()
        except Exception:
            break
    return False


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


class EpubReaderDialog(QDialog):
    """EPUB reader with chapter navigation, layout modes, and theme support."""

    def __init__(self, epub_path: str, config: dict | None = None, parent=None,
                 initial_chapter: int | None = None,
                 initial_chapter_filename: str | None = None,
                 translated_overlay: dict | None = None,
                 extra_image_dirs: list[str] | None = None,
                 window_title: str | None = None,
                 show_special_files: bool | None = None,
                 alt_epub_path: str | None = None):
        super().__init__(parent)
        self._epub_path = epub_path
        self._config = config or {}
        # Dual-path mode: when the caller passes ``alt_epub_path`` (the raw
        # source EPUB that pairs with the compiled one), the Show-raw pill
        # swaps ``_epub_path`` between the two and reloads. Used by the
        # Completed tab so users can flip between a compiled translation
        # and its original without dropping out of the reader.
        #
        # The primary file (``_translated_epub_path``) is whatever the
        # caller opened; the alt (``_raw_epub_alt_path``) is the raw
        # counterpart. Either field is "" when not available.
        self._translated_epub_path = str(epub_path or "")
        self._raw_epub_alt_path = ""
        if alt_epub_path:
            alt_abs = os.path.abspath(str(alt_epub_path))
            cur_abs = os.path.abspath(str(epub_path or ""))
            if (os.path.isfile(alt_abs)
                    and os.path.normcase(alt_abs) != os.path.normcase(cur_abs)):
                self._raw_epub_alt_path = alt_abs
        self._chapters: list[tuple[str, str]] = []
        # Parallel (untouched) chapter list kept alongside the possibly-
        # overlaid ``_chapters`` so the Show-raw toolbar toggle can flip
        # between them without re-parsing the EPUB. ``_chapters_overlaid``
        # is the result of applying ``_translated_overlay`` to the raw
        # chapters — when there's no overlay the two lists are identical.
        self._chapters_raw: list[tuple[str, str]] = []
        self._chapters_overlaid: list[tuple[str, str]] = []
        self._chapter_filenames: list[str] = []
        self._images: dict[str, bytes] = {}
        # Whether to surface non-chapter spine items (cover / nav / toc /
        # info / …) in the TOC. Caller can force the flag; otherwise we
        # resolve it the same way BookDetailsDialog does, so opening a
        # reader directly from a library card picks up the user's last
        # toggle state instead of silently diverging.
        if show_special_files is None:
            self._show_special_files = _resolve_show_special_files(self._config)
        else:
            self._show_special_files = bool(show_special_files)
        # Show-raw toggle: when True, the TOC + content pane render the
        # raw source chapters instead of the translated overlay. Two
        # modes back this flag:
        #   * Overlay mode: a ``_translated_overlay`` dict was attached
        #     (typical for the In Progress tab) — flipping the toggle
        #     swaps ``_chapters`` between raw and overlaid copies.
        #   * Dual-path mode: a ``alt_epub_path`` was attached (typical
        #     for the Completed tab when the raw source is resolvable)
        #     — flipping the toggle swaps ``_epub_path`` between the
        #     compiled and raw EPUBs and reloads.
        # The pill is hidden when neither applies. Persisted in config.
        self._show_raw = bool(self._config.get('epub_reader_show_raw', False))
        # If we can satisfy the persisted "raw" state via the dual-path
        # alt (no overlay scenario), start the reader on the raw EPUB so
        # the user sees what they last chose.
        if (self._show_raw and self._raw_epub_alt_path
                and not (translated_overlay or {})):
            self._epub_path = self._raw_epub_alt_path
        # Restore persisted reader settings
        self._font_size = self._config.get('epub_reader_font_size', 14)
        self._line_spacing = self._config.get('epub_reader_line_spacing', 1.8)
        self._theme_index = self._config.get('epub_reader_theme', 0)
        self._font_family = self._config.get('epub_reader_font_family', 'Georgia')
        layout_key = self._config.get('epub_reader_layout', LAYOUT_SINGLE)
        self._layout_mode = layout_key if layout_key in (LAYOUT_SCROLL, LAYOUT_SINGLE, LAYOUT_DOUBLE, LAYOUT_ALL) else LAYOUT_SINGLE
        # Optional — caller can request opening at a specific chapter index.
        # The index is clamped to the available chapter range once the EPUB
        # has finished loading (see _on_epub_loaded_from_cache). When the
        # ``initial_chapter_filename`` is provided it takes precedence (index
        # is resolved by matching the filename basename, so the selection is
        # correct regardless of spine/manifest ordering differences).
        self._initial_chapter = initial_chapter if isinstance(initial_chapter, int) and initial_chapter >= 0 else None
        self._initial_chapter_filename = (
            os.path.basename(str(initial_chapter_filename)).lower()
            if initial_chapter_filename else None
        )
        # Optional translated-content overlay for in-progress novels. Keyed by
        # the lowercased basename of the source chapter filename (e.g.
        # ``"chapter0001.xhtml"``) so overlay mapping is robust across the
        # various spine/manifest ordering differences between the reader's
        # loader and the BookDetailsDialog's spine parser. Each value is a
        # dict of the form ``{"path": str, "title": Optional[str]}``.
        self._translated_overlay: dict[str, dict] = {}
        if isinstance(translated_overlay, dict):
            for k, v in translated_overlay.items():
                if not k:
                    continue
                key = os.path.basename(str(k)).lower()
                if not key:
                    continue
                if isinstance(v, str):
                    self._translated_overlay[key] = {"path": v}
                elif isinstance(v, dict) and v.get("path"):
                    self._translated_overlay[key] = dict(v)
        # Extra image search directories (e.g. <output_folder>/images,
        # <output_folder>/translated_images) used to augment the EPUB's own
        # image table so translated chapters can resolve their assets.
        self._extra_image_dirs: list[str] = list(extra_image_dirs or [])
        self._current_row = 0
        self._current_page = 0  # viewport-based page for single/double page modes
        # Page-count cache + currently-rendered-chapter sentinel need to
        # exist BEFORE ``_setup_ui`` runs — the QWebEngineView widgets
        # created there kick off an initial ``setUrl("about:blank")``
        # asynchronously, which can fire ``loadFinished`` and reach
        # :meth:`_finalize_single_page` before
        # :meth:`_finalize_post_load` initializes these attributes.
        # Leaving them out causes the about:blank load to crash with
        # ``AttributeError: 'EpubReaderDialog' object has no attribute
        # '_chapter_page_cache'``.
        self._chapter_page_cache: dict[int, int] = {}
        self._loaded_chapter: int = -1
        self._loader_thread: _EpubLoaderThread | None = None

        title_text = window_title or os.path.splitext(os.path.basename(epub_path))[0]
        self._window_title_text = title_text
        self.setWindowTitle(f"\U0001f4d6 {title_text}")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
        # Ratio-based sizing relative to screen
        screen = self.screen()
        if screen:
            avail = screen.availableGeometry()
            self.resize(int(avail.width() * 0.55), int(avail.height() * 0.7))
            self.setMinimumSize(int(avail.width() * 0.4), int(avail.height() * 0.4))
        else:
            self.resize(950, 700)
            self.setMinimumSize(600, 400)

        self._setup_ui()
        # Restore combo positions after UI is built
        modes = [LAYOUT_SINGLE, LAYOUT_DOUBLE, LAYOUT_SCROLL, LAYOUT_ALL]
        if self._layout_mode in modes:
            self._layout_combo.blockSignals(True)
            self._layout_combo.setCurrentIndex(modes.index(self._layout_mode))
            self._layout_combo.blockSignals(False)
        if 0 <= self._theme_index < len(_READER_THEMES):
            self._theme_combo.blockSignals(True)
            self._theme_combo.setCurrentIndex(self._theme_index)
            self._theme_combo.blockSignals(False)
        # Restore spacing combo
        spacing_str = str(self._line_spacing)
        idx = self._spacing_combo.findText(spacing_str)
        self._spacing_combo.blockSignals(True)
        if idx >= 0:
            self._spacing_combo.setCurrentIndex(idx)
        else:
            self._spacing_combo.setCurrentText(spacing_str)
        self._spacing_combo.blockSignals(False)
        # Restore font family combo
        fam_idx = self._font_combo.findText(self._font_family)
        self._font_combo.blockSignals(True)
        if fam_idx >= 0:
            self._font_combo.setCurrentIndex(fam_idx)
        else:
            self._font_combo.setCurrentText(self._font_family)
        self._font_combo.blockSignals(False)
        self._start_loading()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Toolbar ──
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(10, 6, 10, 6)
        toolbar.setSpacing(6)

        title_lbl = QLabel(f"\U0001f4d6 {getattr(self, '_window_title_text', os.path.splitext(os.path.basename(self._epub_path))[0])}")
        title_lbl.setStyleSheet("font-size: 11pt; font-weight: bold; color: #e0e0e0;")
        title_lbl.setMaximumWidth(400)
        toolbar.addWidget(title_lbl)

        # TOC toggle button
        self._toc_btn = self._make_toolbar_btn("📑", "Toggle table of contents")
        self._toc_btn.clicked.connect(self._toggle_toc)
        toolbar.addWidget(self._toc_btn)

        toolbar.addStretch()

        # Layout mode dropdown
        self._layout_combo = QComboBox()
        self._layout_combo.addItems(["📄 Single Page", "📖 Double Page", "📜 Scroll", "📃 Scroll All"])
        self._layout_combo.setFixedWidth(145)
        self._layout_combo.setCursor(Qt.PointingHandCursor)
        self._layout_combo.setStyleSheet("""
            QComboBox {
                background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #e0e0e0; font-size: 8.5pt; padding: 3px 8px;
            }
            QComboBox:hover { border-color: #6c63ff; }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox::down-arrow { image: url(noimg); width: 10px; height: 10px; }
            QComboBox QAbstractItemView {
                background: #1e1e2e; color: #e0e0e0; selection-background-color: #3a3a5e;
                border: 1px solid #3a3a5e; font-size: 8.5pt;
            }
        """)
        self._layout_combo.currentIndexChanged.connect(self._on_layout_changed)
        self._layout_combo.setFocusPolicy(Qt.StrongFocus)
        self._layout_combo.installEventFilter(self)
        toolbar.addWidget(self._layout_combo)

        toolbar.addSpacing(6)

        # Show-raw toggle: flips the reader between translated overlay
        # and raw source content on the fly. Hidden until the loader
        # confirms a translated overlay exists (there's nothing to
        # toggle against for a plain EPUB open). Visually matches the
        # "Raw titles" pill on the Library toolbar so the two surfaces
        # read as the same control in different locations.
        self._raw_btn = QPushButton("\U0001f524  Raw")
        self._raw_btn.setToolTip(
            "Show the raw source-language content instead of the translated \n"
            "overlay. Useful for comparing a chapter against its original."
        )
        self._raw_btn.setFixedHeight(26)
        self._raw_btn.setCursor(Qt.PointingHandCursor)
        self._raw_btn.setCheckable(True)
        self._raw_btn.setChecked(self._show_raw)
        self._raw_btn.setStyleSheet("""
            QPushButton { background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #b0b0c0; font-size: 8.5pt; font-weight: bold; padding: 2px 10px; }
            QPushButton:hover { background: #3a3a5e; color: #e0e0e0; }
            QPushButton:checked { background: #6c63ff; border-color: #7c73ff; color: #fff; }
        """)
        self._raw_btn.toggled.connect(self._on_show_raw_toggled)
        # Hidden until we know whether there's a translated overlay to flip
        # against — :meth:`_on_epub_loaded_from_cache` reveals it when the
        # overlaid list actually differs from the raw one.
        self._raw_btn.hide()
        toolbar.addWidget(self._raw_btn)

        toolbar.addSpacing(8)

        # Line spacing dropdown
        spacing_lbl = QLabel("↕")
        spacing_lbl.setStyleSheet("color: #888; font-size: 11pt;")
        spacing_lbl.setToolTip("Line spacing")
        toolbar.addWidget(spacing_lbl)

        self._spacing_combo = QComboBox()
        self._spacing_combo.setEditable(True)
        self._spacing_combo.addItems(["1.0", "1.2", "1.4", "1.6", "1.8", "2.0", "2.2", "2.4", "2.6", "2.8", "3.0"])
        self._spacing_combo.setFixedWidth(58)
        self._spacing_combo.setCursor(Qt.PointingHandCursor)
        self._spacing_combo.setStyleSheet("""
            QComboBox {
                background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #e0e0e0; font-size: 8.5pt; padding: 3px 6px;
            }
            QComboBox:hover { border-color: #6c63ff; }
            QComboBox::drop-down { border: none; width: 18px; }
            QComboBox::down-arrow { image: url(noimg); width: 10px; height: 10px; }
            QComboBox QAbstractItemView {
                background: #1e1e2e; color: #e0e0e0; selection-background-color: #3a3a5e;
                border: 1px solid #3a3a5e;
            }
        """)
        self._spacing_combo.activated.connect(lambda idx: self._on_spacing_changed(self._spacing_combo.itemText(idx)))
        self._spacing_combo.lineEdit().editingFinished.connect(lambda: self._on_spacing_changed(self._spacing_combo.currentText()))
        self._spacing_combo.setFocusPolicy(Qt.StrongFocus)
        self._spacing_combo.installEventFilter(self)
        toolbar.addWidget(self._spacing_combo)

        toolbar.addSpacing(8)

        # Font family dropdown (system fonts, editable for custom families)
        font_family_lbl = QLabel("𝑨")
        font_family_lbl.setStyleSheet("color: #888; font-size: 12pt; padding: 0 2px;")
        font_family_lbl.setToolTip("Font family")
        toolbar.addWidget(font_family_lbl)

        self._font_combo = QComboBox()
        self._font_combo.setEditable(True)
        self._font_combo.setInsertPolicy(QComboBox.NoInsert)
        self._font_combo.setFixedWidth(150)
        self._font_combo.setCursor(Qt.PointingHandCursor)
        self._font_combo.setToolTip("Text font family")
        # Populate with smoothly-scalable system font families (same pattern
        # used in review_dialog.py). Pin common reading fonts to the top.
        try:
            from PySide6.QtGui import QFontDatabase
            families = sorted({
                f for f in QFontDatabase.families()
                if QFontDatabase.isSmoothlyScalable(f)
            })
        except Exception:
            families = []
        _pinned = ["Georgia", "Noto Serif", "Segoe UI", "Cambria", "Times New Roman",
                   "Garamond", "Palatino Linotype", "Arial", "Verdana", "Tahoma",
                   "Consolas"]
        _added: set[str] = set()
        for fam in _pinned:
            if not families or fam in families:
                self._font_combo.addItem(fam)
                _added.add(fam)
        if families:
            self._font_combo.insertSeparator(self._font_combo.count())
            for fam in families:
                if fam not in _added:
                    self._font_combo.addItem(fam)
        self._font_combo.setStyleSheet("""
            QComboBox {
                background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #e0e0e0; font-size: 8.5pt; padding: 3px 8px;
            }
            QComboBox:hover { border-color: #6c63ff; }
            QComboBox::drop-down { border: none; width: 18px; }
            QComboBox::down-arrow { image: url(noimg); width: 10px; height: 10px; }
            QComboBox QAbstractItemView {
                background: #1e1e2e; color: #e0e0e0; selection-background-color: #3a3a5e;
                border: 1px solid #3a3a5e;
            }
        """)
        self._font_combo.activated.connect(
            lambda idx: self._on_font_family_changed(self._font_combo.itemText(idx)))
        self._font_combo.lineEdit().editingFinished.connect(
            lambda: self._on_font_family_changed(self._font_combo.currentText()))
        self._font_combo.setFocusPolicy(Qt.StrongFocus)
        self._font_combo.installEventFilter(self)
        toolbar.addWidget(self._font_combo)

        toolbar.addSpacing(8)

        # Font size controls
        font_down = self._make_toolbar_btn("A−", "Decrease font size")
        font_down.clicked.connect(lambda: self._change_font_size(-1))
        toolbar.addWidget(font_down)

        self._font_label = QLabel(f"{self._font_size}pt")
        self._font_label.setStyleSheet("color: #888; font-size: 8.5pt; padding: 0 2px;")
        toolbar.addWidget(self._font_label)

        font_up = self._make_toolbar_btn("A+", "Increase font size")
        font_up.clicked.connect(lambda: self._change_font_size(1))
        toolbar.addWidget(font_up)

        toolbar.addSpacing(4)

        # Theme dropdown
        self._theme_combo = QComboBox()
        self._theme_combo.addItems([t["name"] for t in _READER_THEMES])
        self._theme_combo.setCurrentIndex(0)
        self._theme_combo.setFixedWidth(95)
        self._theme_combo.setCursor(Qt.PointingHandCursor)
        self._theme_combo.setStyleSheet("""
            QComboBox {
                background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #e0e0e0; font-size: 8.5pt; padding: 3px 8px;
            }
            QComboBox:hover { border-color: #6c63ff; }
            QComboBox::drop-down { border: none; width: 18px; }
            QComboBox::down-arrow { image: url(noimg); width: 10px; height: 10px; }
            QComboBox QAbstractItemView {
                background: #1e1e2e; color: #e0e0e0; selection-background-color: #3a3a5e;
                border: 1px solid #3a3a5e;
            }
        """)
        self._theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        self._theme_combo.setFocusPolicy(Qt.StrongFocus)
        self._theme_combo.installEventFilter(self)
        toolbar.addWidget(self._theme_combo)

        self._toolbar_widget = QWidget()
        self._toolbar_widget.setLayout(toolbar)
        self._toolbar_widget.setStyleSheet("background: #1e1e1e; border-bottom: 1px solid #333333;")
        root.addWidget(self._toolbar_widget)

        # ── Loading indicator ──
        self._loading_widget = QWidget()
        loading_layout = QVBoxLayout(self._loading_widget)
        loading_layout.setAlignment(Qt.AlignCenter)
        # Spinning icon via QTimer rotation
        self._spin_label = QLabel()
        icon_path = _find_halgakos_icon()
        if icon_path:
            pm = QPixmap(icon_path)
            if not pm.isNull():
                self._spin_pixmap = pm.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self._spin_label.setPixmap(self._spin_pixmap)
            else:
                self._spin_pixmap = None
        else:
            self._spin_pixmap = None
        if not self._spin_pixmap:
            self._spin_label.setText("📖")
            self._spin_label.setStyleSheet("font-size: 32pt;")
        self._spin_label.setAlignment(Qt.AlignCenter)
        self._spin_label.setFixedSize(72, 72)
        loading_layout.addWidget(self._spin_label, 0, Qt.AlignCenter)
        self._spin_angle = 0
        self._spin_timer = QTimer(self)
        self._spin_timer.setInterval(25)  # ~40 fps
        # Wrap in a lambda so PySide6 routes the call directly instead
        # of going through Qt's meta-object slot-lookup — certain
        # PySide6 builds raise ``AttributeError: Slot 'EpubReaderDialog::
        # _rotate_spinner()' not found`` on ``QTimer.timeout`` dispatch
        # even when the method exists and is ``@Slot()``-decorated.
        self._spin_timer.timeout.connect(lambda: self._rotate_spinner())
        loading_text = QLabel("Loading EPUB\u2026")
        loading_text.setAlignment(Qt.AlignCenter)
        loading_text.setStyleSheet("color: #888; font-size: 11pt; padding-top: 8px;")
        loading_layout.addWidget(loading_text)
        # Indeterminate progress bar
        from PySide6.QtWidgets import QProgressBar
        self._loading_bar = QProgressBar()
        self._loading_bar.setRange(0, 0)  # indeterminate
        self._loading_bar.setFixedWidth(220)
        self._loading_bar.setFixedHeight(6)
        self._loading_bar.setTextVisible(False)
        self._loading_bar.setStyleSheet("""
            QProgressBar { background: #2a2a2a; border: none; border-radius: 3px; }
            QProgressBar::chunk { background: #6c63ff; border-radius: 3px; }
        """)
        loading_layout.addWidget(self._loading_bar, 0, Qt.AlignCenter)
        self._loading_widget.setStyleSheet("background: #1e1e1e;")
        root.addWidget(self._loading_widget)

        # ── Main content (hidden until loaded) ──
        self._content_widget = QWidget()
        content_layout = QVBoxLayout(self._content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)

        # TOC sidebar
        self._toc_list = QListWidget()
        self._toc_list.setMinimumWidth(100)
        self._toc_list.resize(220, self._toc_list.height())
        self._toc_list.currentRowChanged.connect(self._on_chapter_selected)
        splitter.addWidget(self._toc_list)

        # Reader area — single browser for scroll/single, an HBox for double-page
        self._reader_stack = QStackedWidget()

        def _make_reader_widget():
            if _HAS_WEBENGINE:
                # Use the wheel-capturing subclass so mouse wheel events
                # can drive page navigation / font zoom instead of being
                # silently swallowed by the internal Chromium scroller.
                w = _WheelCapturingView()
                w.setUrl(QUrl("about:blank"))
                # Set page background to match theme (prevents white flash)
                from PySide6.QtGui import QColor
                t = self._get_theme()
                w.page().setBackgroundColor(QColor(t['bg']))
                w.wheel_scrolled.connect(self._on_reader_wheel_scrolled)
            else:
                w = QTextBrowser()
                w.setOpenExternalLinks(False)
                w.setOpenLinks(False)
            # Right-click menu: Google Translate + Web Search for the
            # current selection. Applies to every reader pane so the
            # double-page mode works the same way the single pane does.
            w.setContextMenuPolicy(Qt.CustomContextMenu)
            w.customContextMenuRequested.connect(
                lambda pos, browser=w:
                    self._show_reader_context_menu(browser, pos))
            return w

        # Page 0: single reader (for scroll, single page, all-scroll)
        self._reader = _make_reader_widget()
        if _HAS_WEBENGINE:
            self._reader.loadFinished.connect(self._on_reader_load_finished)
        self._reader_stack.addWidget(self._reader)

        # Page 1: double-page layout (two browsers side by side)
        double_widget = QWidget()
        self._double_widget = double_widget  # keep ref for styling
        double_layout = QHBoxLayout(double_widget)
        double_layout.setContentsMargins(0, 0, 0, 0)
        double_layout.setSpacing(2)
        self._reader_left = _make_reader_widget()
        self._reader_right = _make_reader_widget()
        # Hook loadFinished on both panes so _finalize_double_page actually runs.
        # Without this, the panes never get scrolled to their target pages and
        # stay frozen at the start of the chapter.
        if _HAS_WEBENGINE:
            self._reader_left.loadFinished.connect(self._on_reader_load_finished)
            self._reader_right.loadFinished.connect(self._on_reader_load_finished)
        double_layout.addWidget(self._reader_left)
        double_layout.addWidget(self._reader_right)
        self._reader_stack.addWidget(double_widget)

        splitter.addWidget(self._reader_stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([220, 680])
        self._splitter = splitter  # keep ref for styling
        content_layout.addWidget(splitter, 1)

        # Bottom nav bar for single-page mode
        self._nav_bar = QWidget()
        nav_layout = QHBoxLayout(self._nav_bar)
        nav_layout.setContentsMargins(10, 4, 10, 6)
        nav_layout.setSpacing(8)
        self._prev_btn = QPushButton("◀  Previous")
        self._prev_btn.setCursor(Qt.PointingHandCursor)
        self._prev_btn.setStyleSheet("""
            QPushButton { background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #c0c0d0; font-size: 9pt; padding: 6px 16px; }
            QPushButton:hover { background: #3a3a5e; color: #e0e0e0; }
            QPushButton:disabled { color: #555; }
        """)
        self._prev_btn.clicked.connect(self._prev_chapter)
        nav_layout.addWidget(self._prev_btn)
        nav_layout.addStretch()
        self._page_label = QLabel("")
        self._page_label.setStyleSheet("color: #888; font-size: 9pt;")
        nav_layout.addWidget(self._page_label)
        nav_layout.addStretch()
        self._next_btn = QPushButton("Next  ▶")
        self._next_btn.setCursor(Qt.PointingHandCursor)
        self._next_btn.setStyleSheet("""
            QPushButton { background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #c0c0d0; font-size: 9pt; padding: 6px 16px; }
            QPushButton:hover { background: #3a3a5e; color: #e0e0e0; }
            QPushButton:disabled { color: #555; }
        """)
        self._next_btn.clicked.connect(self._next_chapter)
        nav_layout.addWidget(self._next_btn)
        self._nav_bar.hide()
        content_layout.addWidget(self._nav_bar)

        self._content_widget.hide()
        root.addWidget(self._content_widget, 1)

        # Search bar (hidden by default)
        self._search_bar = QLineEdit()
        self._search_bar.setPlaceholderText("Search across EPUB... (Enter = next, Esc = close)")
        self._search_bar.hide()
        self._search_bar.textChanged.connect(self._on_search_text_changed)
        self._search_bar.returnPressed.connect(self._on_search_next)
        self._search_chapter_idx = 0  # track which chapter we last searched
        root.addWidget(self._search_bar)

        self._apply_reader_style()

        # Shortcuts (work regardless of child focus)
        QShortcut(QKeySequence(Qt.Key_Left), self, self._prev_chapter)
        QShortcut(QKeySequence(Qt.Key_Right), self, self._next_chapter)
        QShortcut(QKeySequence(Qt.Key_F11), self, self._toggle_fullscreen)
        QShortcut(QKeySequence("Ctrl+="), self, lambda: self._change_font_size(1))
        QShortcut(QKeySequence("Ctrl++"), self, lambda: self._change_font_size(1))
        QShortcut(QKeySequence("Ctrl+-"), self, lambda: self._change_font_size(-1))
        QShortcut(QKeySequence("Ctrl+F"), self, self._toggle_search)
        QShortcut(QKeySequence(Qt.Key_Escape), self, self._close_search)

    # ── Event filter (block wheel scroll in paginated modes) ──────────────

    def eventFilter(self, obj, event):
        """Block wheel on combos, handle Ctrl+Wheel for font zoom."""
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.Wheel:
            if isinstance(obj, QComboBox):
                return True  # block wheel on all toolbar combos
            # Ctrl+Wheel = font zoom
            if event.modifiers() & Qt.ControlModifier:
                if event.angleDelta().y() > 0:
                    self._change_font_size(1)
                else:
                    self._change_font_size(-1)
                return True
        return super().eventFilter(obj, event)

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _make_toolbar_btn(self, text, tooltip, width=36):
        btn = QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setFixedSize(width, 28)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(
            "QPushButton { background: #2a2a3e; border-radius: 4px; color: #e0e0e0; "
            "font-size: 10pt; font-weight: bold; border: none; }"
            "QPushButton:hover { background: #3a3a5e; }")
        return btn

    @Slot()
    def _rotate_spinner(self):
        """Rotate the Halgakos icon by 15° each tick.

        Decorated with ``@Slot()`` so PySide6 registers it in the Qt
        meta-object system. Without the decorator, ``QTimer.timeout``
        can fail to resolve the slot by name at invocation time
        (``AttributeError: Slot 'EpubReaderDialog::_rotate_spinner()'
        not found``).
        """
        if self._spin_pixmap:
            self._spin_angle = (self._spin_angle + 15) % 360
            t = QTransform().rotate(self._spin_angle)
            rotated = self._spin_pixmap.transformed(t, Qt.FastTransformation)
            self._spin_label.setPixmap(rotated)

    # ── Loading ───────────────────────────────────────────────────

    def _start_loading(self):
        self._toolbar_widget.hide()
        self._loading_widget.show()
        self._content_widget.hide()
        self._spin_angle = 0
        self._spin_timer.start()
        # Probe the cache in a worker thread so the pickle.load doesn't
        # freeze the spinner. The worker emits ``hit`` with the cached
        # data on success, or ``miss`` to fall through to the full
        # :class:`_EpubLoaderThread` re-parse.
        prev_cache = getattr(self, "_cache_loader_thread", None)
        if prev_cache is not None:
            try:
                prev_cache.hit.disconnect()
                prev_cache.miss.disconnect()
            except Exception:
                pass
            try:
                prev_cache.quit()
            except Exception:
                pass
        self._cache_loader_thread = _EpubCacheLoaderThread(
            self._epub_path,
            show_special_files=self._show_special_files,
            parent=self,
        )
        self._cache_loader_thread.hit.connect(self._on_cache_hit)
        self._cache_loader_thread.miss.connect(self._on_cache_miss)
        self._cache_loader_thread.start()

    @Slot(object, object, list)
    def _on_cache_hit(self, chapters, images, filenames):
        """Cache worker delivered hit — finalize or fall back to reparse."""
        filenames = list(filenames or [])
        # Older caches may not carry filenames. If an overlay is requested
        # but filenames are missing we MUST reparse the EPUB so the overlay
        # can map correctly — otherwise we'd silently keep the raw text.
        if self._translated_overlay and not filenames:
            self._on_cache_miss()
            return
        self._on_epub_loaded_from_cache(chapters, images, filenames)

    @Slot()
    def _on_cache_miss(self):
        """Cache worker had nothing usable — kick off the full reparse."""
        self._loader_thread = _EpubLoaderThread(
            self._epub_path, self,
            show_special_files=self._show_special_files,
        )
        self._loader_thread.done.connect(lambda: self._on_loader_done())
        self._loader_thread.error.connect(lambda msg: self._on_epub_error(msg))
        self._loader_thread.start()

    @Slot()
    def _on_loader_done(self):
        """Loader finished — re-read data from cache in a worker so the
        pickle.load doesn't re-freeze the spinner right before the
        reader is about to render.
        """
        prev_cache = getattr(self, "_cache_loader_thread", None)
        if prev_cache is not None:
            try:
                prev_cache.hit.disconnect()
                prev_cache.miss.disconnect()
            except Exception:
                pass
            try:
                prev_cache.quit()
            except Exception:
                pass
        reader_thread = _EpubCacheLoaderThread(
            self._epub_path,
            show_special_files=self._show_special_files,
            parent=self,
        )
        # Post-reparse cache-miss shouldn't re-trigger the parser (that
        # would loop forever) — surface it as an error instead.
        reader_thread.hit.connect(self._on_cache_hit)
        reader_thread.miss.connect(
            lambda: self._on_epub_error(
                "Failed to read EPUB cache after loading."))
        self._cache_loader_thread = reader_thread
        reader_thread.start()

    def _on_epub_loaded_from_cache(self, chapters, images, filenames=None):
        self._spin_timer.stop()
        filenames = list(filenames or [])
        raw_chapters = list(chapters or [])
        # Fast path: nothing to merge (plain EPUB open, no overlay, no
        # extra image directories). The UI can finalize immediately.
        if not self._translated_overlay and not self._extra_image_dirs:
            self._finalize_post_load(raw_chapters, raw_chapters, images or {},
                                     False, filenames)
            return
        # Heavy path: hand the per-chapter file reads + image-dir scan to
        # :class:`_OverlayMergeThread` so the Qt event loop stays free.
        # Stash raw + filenames on self so the done signal's slot can
        # seed the finalizer without having to round-trip them through
        # the worker (images are modified there, chapters are produced).
        self._pending_raw_chapters = raw_chapters
        self._pending_filenames = filenames
        # Cancel any in-flight worker from a previous load (e.g. the
        # user closed + reopened quickly, or the Raw toggle fired a
        # dual-path reload before the previous merge finished).
        prev = getattr(self, "_overlay_thread", None)
        if prev is not None:
            try:
                prev.done.disconnect()
            except Exception:
                pass
            try:
                prev.quit()
            except Exception:
                pass
        self._overlay_thread = _OverlayMergeThread(
            raw_chapters=raw_chapters,
            images=images or {},
            filenames=filenames,
            overlay_map=self._translated_overlay,
            extra_image_dirs=self._extra_image_dirs,
            parent=self,
        )
        # Lambda-wrap so PySide6 routes the delivery directly instead
        # of going through Qt's meta-object slot-lookup — the same
        # build quirk that hits ``_rotate_spinner`` also surfaces here
        # as ``AttributeError: Slot 'EpubReaderDialog::
        # _on_overlay_merge_done(...)' not found``.
        self._overlay_thread.done.connect(
            lambda overlaid, imgs, applied:
                self._on_overlay_merge_done(overlaid, imgs, applied)
        )
        self._overlay_thread.start()

    @Slot(object, object, bool)
    def _on_overlay_merge_done(self, overlaid_chapters, merged_images,
                               overlay_applied: bool):
        """Merge worker finished: hand off to the main-thread finalizer.

        ``@Slot(object, object, bool)`` matches the updated signature
        of :attr:`_OverlayMergeThread.done` — both sides use ``object``
        so the payload is a raw Python reference and PySide6 doesn't
        try (and fail) to look up a ``(QVariantList, QVariantMap, bool)``
        slot on ``EpubReaderDialog``.
        """
        raw_chapters = getattr(self, "_pending_raw_chapters", []) or []
        filenames = getattr(self, "_pending_filenames", []) or []
        self._pending_raw_chapters = []
        self._pending_filenames = []
        self._finalize_post_load(
            raw_chapters, overlaid_chapters,
            merged_images, bool(overlay_applied), filenames)

    def _finalize_post_load(self, raw_chapters, overlaid_chapters, images,
                            overlay_applied: bool, filenames):
        """Install the loaded chapters + images into the reader UI.

        Split out from :meth:`_on_epub_loaded_from_cache` so the
        translator-overlay merge + extra-image-dir scan can run in
        :class:`_OverlayMergeThread` without blocking the event loop.
        For the fast path (plain EPUB, no overlay) the loader calls this
        directly; for the heavy path it's invoked via the worker's
        ``done`` signal.
        """
        # Persist both flavors + pick the one matching the current toggle.
        self._chapters_raw = raw_chapters
        self._chapters_overlaid = overlaid_chapters
        chapters = raw_chapters if (self._show_raw and overlay_applied) else overlaid_chapters
        # Expose the Show-raw pill when EITHER a translated overlay
        # landed on at least one chapter (overlay mode) OR a dual-path
        # alt EPUB was wired up by the caller (Completed tab with a
        # resolved raw source). Otherwise there's nothing to toggle
        # against and the pill stays hidden.
        has_dual_path = bool(getattr(self, "_raw_epub_alt_path", ""))
        if getattr(self, "_raw_btn", None) is not None:
            self._raw_btn.setVisible(bool(overlay_applied) or has_dual_path)
            # Sync the pill's checked state with the actually-rendered
            # flavor — this can diverge from the persisted default when
            # overlay mode couldn't satisfy the "raw" preference (no
            # chapter had an overlay entry) or when a dual-path reload
            # already swapped ``_epub_path`` to the raw file.
            try:
                raw_alt = getattr(self, "_raw_epub_alt_path", "") or ""
                currently_raw = bool(
                    (overlay_applied and self._show_raw)
                    or (has_dual_path and raw_alt
                        and os.path.normcase(os.path.abspath(
                            self._epub_path))
                        == os.path.normcase(os.path.abspath(raw_alt)))
                )
            except Exception:
                currently_raw = bool(
                    overlay_applied and self._show_raw)
            self._raw_btn.blockSignals(True)
            self._raw_btn.setChecked(currently_raw)
            self._raw_btn.blockSignals(False)
        self._chapters = chapters
        self._images = images
        # Keep filenames around so callers can resolve chapter indices by
        # source filename (used by initial_chapter_filename lookup + TOC jumps).
        self._chapter_filenames = [os.path.basename(f or "").lower() for f in filenames]
        self._chapter_page_cache = {}  # {chapter_index: page_count}
        self._loaded_chapter = -1  # track which chapter's HTML is loaded

        self._toc_list.clear()
        for idx, (title, _) in enumerate(self._chapters):
            item = QListWidgetItem(title)
            self._toc_list.addItem(item)

        self._toolbar_widget.show()
        self._loading_widget.hide()
        self._content_widget.show()

        if self._chapters:
            # Honor an explicit initial chapter if one was requested by the
            # caller; otherwise default to the first chapter. The index is
            # clamped to the loaded chapter range so stale requests (e.g. a
            # chapter that was removed between sessions) fall back cleanly.
            # Filename takes priority — BookDetailsDialog passes the source
            # chapter filename so the selection is stable across ordering
            # differences between its spine-based list and the reader's
            # manifest-ordered list.
            initial = 0
            # A reload carrying a position hint (Show-raw toggle in
            # dual-path mode) takes the highest precedence so the user
            # lands back on the same chapter in the new flavor — the
            # finalizer then consumes the page portion of the hint.
            reload_hint = getattr(self, '_reload_position_hint', None)
            if reload_hint:
                initial = max(0, min(int(reload_hint.get("row", 0) or 0),
                                     len(self._chapters) - 1))
                # Promote to a page hint for the finalizer and clear the
                # reload slot so a later organic open doesn't inherit it.
                self._pending_page_hint = {
                    "row": initial,
                    "was_last_page": bool(reload_hint.get("was_last_page")),
                    "proportion": float(reload_hint.get("proportion") or 0.0),
                }
                self._reload_position_hint = None
            elif self._initial_chapter_filename:
                try:
                    initial = self._chapter_filenames.index(self._initial_chapter_filename)
                except ValueError:
                    initial = 0
            elif self._initial_chapter is not None:
                initial = max(0, min(self._initial_chapter, len(self._chapters) - 1))
            # Select initial chapter silently — the priming sequence below
            # drives the initial render so we don't want setCurrentRow to
            # also fire _on_chapter_selected.
            self._toc_list.blockSignals(True)
            self._toc_list.setCurrentRow(initial)
            self._toc_list.blockSignals(False)
            self._current_row = initial
            self._current_page = 0

            # Prime the reader: if we're opening in a paginated mode
            # (single/double), first render in scroll mode and then switch
            # back to the configured mode. This replicates the manual
            # layout-toggle workaround — without it, QtWebEngine caches the
            # GPU-composited #columns layer at the wrong DPR and text
            # stays blurry until the user toggles modes.
            saved_mode = self._layout_mode
            if saved_mode in (LAYOUT_SINGLE, LAYOUT_DOUBLE):
                # Hide the reader stack during the prime swap so the
                # brief scroll-mode render never becomes visible. The
                # dialog background (theme bg) shows through instead.
                self._reader_stack.hide()
                self._priming_initial_render = True
                self._prime_saved_mode = saved_mode
                self._layout_mode = LAYOUT_SCROLL
                # The swap-back is triggered event-driven the moment the
                # scroll-mode load finishes (see _on_reader_load_finished),
                # avoiding an arbitrary fixed delay.
                self._render_current()
            else:
                self._render_current()
        else:
            self._reader_stack.setCurrentIndex(0)
            self._reader.setHtml(
                "<div style='text-align:center; padding: 60px; color: #888;'>"
                "<p style='font-size: 16pt;'>📭</p>"
                "<p>No readable content found in this EPUB.</p></div>")

    def _get_chapter_pages(self, chapter_idx):
        """Get page count for a chapter (cached or live-compute for current)."""
        if chapter_idx in self._chapter_page_cache:
            return self._chapter_page_cache[chapter_idx]
        return 1  # fallback until chapter is actually rendered

    def _get_global_page_info(self):
        """Return (current_global_page_1based, total_pages)."""
        offset = sum(self._get_chapter_pages(i) for i in range(self._current_row))
        total = sum(self._get_chapter_pages(i) for i in range(len(self._chapters)))
        return (offset + self._current_page + 1, total)

    def _on_epub_error(self, error_msg):
        self._spin_timer.stop()
        self._toolbar_widget.show()
        self._loading_widget.hide()
        self._content_widget.show()
        self._reader_stack.setCurrentIndex(0)
        self._reader.setHtml(
            f"<div style='text-align:center; padding: 60px; color: #ff6b6b;'>"
            f"<p style='font-size: 16pt;'>⚠️</p>"
            f"<p>Failed to load EPUB:<br>{error_msg}</p></div>")

    # ── Theme / Font / Spacing ─────────────────────────────────────────────

    def _get_theme(self):
        idx = self._theme_index if 0 <= self._theme_index < len(_READER_THEMES) else 0
        return _READER_THEMES[idx]

    def _apply_reader_style(self):
        t = self._get_theme()
        bg = t['bg']
        fg = t['fg']
        border = t['border']
        if _HAS_WEBENGINE:
            # For QWebEngineView, styling is done via CSS in _wrap_html.
            # We style surrounding containers only.
            self._reader.setStyleSheet(f"background: {bg}; border: none;")
            self._reader_left.setStyleSheet(f"background: {bg}; border: none; border-right: 1px solid {border};")
            self._reader_right.setStyleSheet(f"background: {bg}; border: none;")
        else:
            css = f"""
                QTextBrowser {{
                    background: {bg}; color: {fg}; border: none;
                    padding: 20px 30px; font-size: {self._font_size}pt;
                }}
                QTextBrowser a {{ color: {t['link']}; }}
                QScrollBar:vertical {{ width: 8px; background: {bg}; }}
                QScrollBar::handle:vertical {{ background: {border}; border-radius: 4px; min-height: 20px; }}
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            """
            self._reader.setStyleSheet(css)
            self._reader_left.setStyleSheet(css + f"QTextBrowser {{ border-right: 1px solid {border}; }}")
            self._reader_right.setStyleSheet(css)
        # Theme all surrounding containers
        self.setStyleSheet(f"QDialog {{ background: {bg}; }}")
        self._reader_stack.setStyleSheet(f"QStackedWidget {{ background: {bg}; border: none; }}")
        self._double_widget.setStyleSheet(f"background: {bg};")
        self._toc_list.setStyleSheet(f"""
            QListWidget {{ background: {bg}; border: none;
                color: {fg}; font-size: 9pt; padding: 4px; }}
            QListWidget::item {{ padding: 6px 8px; border-radius: 4px; }}
            QListWidget::item:selected {{ background: {border}; color: {t['heading']}; }}
            QListWidget::item:hover {{ background: {t['code_bg']}; }}
            QScrollBar:vertical {{ width: 8px; background: {bg}; }}
            QScrollBar::handle:vertical {{ background: {border}; border-radius: 4px; min-height: 20px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QScrollBar:horizontal {{ height: 8px; background: {bg}; }}
            QScrollBar::handle:horizontal {{ background: {border}; border-radius: 4px; min-width: 20px; }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
        """)
        self._toc_list.viewport().setStyleSheet(f"background: {bg};")
        self._nav_bar.setStyleSheet(f"background: {bg}; border-top: 1px solid {border};")
        self._loading_widget.setStyleSheet(f"background: {bg};")
        self._splitter.setStyleSheet(f"QSplitter::handle {{ background: transparent; width: 0px; }}")
        self._search_bar.setStyleSheet(f"""
            QLineEdit {{
                background: {t['code_bg']}; color: {fg}; border: 1px solid {border};
                border-radius: 4px; padding: 6px 12px; font-size: 10pt;
                margin: 4px 40px;
            }}
        """)

    def _toggle_search(self):
        if self._search_bar.isVisible():
            self._close_search()
        else:
            self._search_bar.show()
            self._search_bar.setFocus()
            self._search_bar.selectAll()
            self._search_chapter_idx = self._current_row

    def _close_search(self):
        self._search_bar.hide()
        self._search_bar.clear()
        if _HAS_WEBENGINE:
            for w in [self._reader, self._reader_left, self._reader_right]:
                if hasattr(w, 'findText'):
                    w.findText("")

    def _on_search_text_changed(self, text):
        """Live highlight in current chapter as user types."""
        if not _HAS_WEBENGINE:
            return
        self._search_chapter_idx = self._current_row
        if not text:
            for w in [self._reader, self._reader_left, self._reader_right]:
                if hasattr(w, 'findText'):
                    w.findText("")
            return
        self._reader.findText(text)

    def _on_search_next(self):
        """Enter pressed: find next match across all chapters."""
        text = self._search_bar.text().strip()
        if not text or not self._chapters:
            return
        import re
        _TAG_RE = re.compile(r'<[^>]+>')
        n = len(self._chapters)
        # Search from _search_chapter_idx onward (wrap around)
        for offset in range(n):
            idx = (self._search_chapter_idx + offset) % n
            _, html = self._chapters[idx]
            plain = _TAG_RE.sub('', html)
            if text.lower() in plain.lower():
                if idx != self._current_row:
                    # Navigate to that chapter
                    self._toc_list.blockSignals(True)
                    self._toc_list.setCurrentRow(idx)
                    self._toc_list.blockSignals(False)
                    self._on_chapter_selected(idx)
                    # After load finishes, highlight — use a short timer
                    QTimer.singleShot(300, lambda t=text: self._reader.findText(t))
                else:
                    # Same chapter — just do findText (advances to next match)
                    self._reader.findText(text)
                # Advance for next Enter press
                self._search_chapter_idx = (idx + 1) % n
                return
        # No match found anywhere
        self._search_bar.setStyleSheet(
            self._search_bar.styleSheet() + " QLineEdit { border-color: #c04040; }")
        QTimer.singleShot(800, lambda: self._apply_reader_style())

    def _toggle_toc(self):
        """Show or hide the TOC sidebar using splitter sizes."""
        sizes = self._splitter.sizes()
        if sizes[0] > 0:
            # Hide TOC: remember width, set to 0
            self._toc_saved_width = sizes[0]
            self._splitter.setSizes([0, sizes[1] + sizes[0]])
        else:
            # Show TOC: restore saved width
            w = getattr(self, '_toc_saved_width', 220)
            self._splitter.setSizes([w, max(1, sizes[1] - w)])
        # The reader viewport width just changed. The in-page _setupColumns
        # resize handler re-anchors the transform to _CURRENT_PAGE atomically,
        # so no opacity flash is needed here — we only need to refresh the
        # Python-side page-count cache that drives nav-button state.
        self._resync_page_count()

    def _resync_page_count(self):
        """Requery page count after a viewport width change (no flash)."""
        if not hasattr(self, '_chapter_page_cache'):
            return
        if self._layout_mode not in (LAYOUT_SINGLE, LAYOUT_DOUBLE):
            return
        if not self._chapters:
            return
        self._chapter_page_cache.clear()

        def on_count(count):
            count = int(count)
            self._chapter_page_cache[self._current_row] = count
            # Clamp if the current page is now past the end (e.g. column
            # count shrank) and silently jump to the clamped position.
            step = 2 if self._layout_mode == LAYOUT_DOUBLE else 1
            max_page = max(0, count - step)
            if self._current_page > max_page:
                self._current_page = max_page
                if self._layout_mode == LAYOUT_SINGLE:
                    self._js_scroll_to(self._reader, self._current_page, animate=False)
                else:
                    self._js_scroll_to(self._reader_left, self._current_page, animate=False)
                    self._js_scroll_to(self._reader_right, self._current_page + 1, animate=False)
            self._update_nav_buttons()

        # Give the browser a tick to process the splitter-driven resize
        # before we query the new scrollWidth-based page count.
        def _do():
            if self._layout_mode == LAYOUT_SINGLE:
                self._js_page_count(self._reader, on_count)
            else:
                self._js_page_count(self._reader_left, on_count)
        QTimer.singleShot(60, _do)

    def _change_font_size(self, delta):
        self._font_size = max(8, min(32, self._font_size + delta))
        self._font_label.setText(f"{self._font_size}pt")
        self._apply_reader_style()
        self._chapter_page_cache.clear()
        self._loaded_chapter = -1
        # Hide content before re-render to prevent flash
        _hide = "var c = document.getElementById('columns'); if (c) c.style.opacity = '0';"
        if _HAS_WEBENGINE:
            if self._layout_mode == LAYOUT_DOUBLE:
                self._reader_left.page().runJavaScript(_hide)
                self._reader_right.page().runJavaScript(_hide)
            else:
                self._reader.page().runJavaScript(_hide)
        self._render_current()

    def _on_reader_wheel_scrolled(self, delta_y: int, modifiers) -> None:
        """Route a wheel event from a reader pane to the right action.

        :class:`_WheelCapturingView` consumes every wheel event its
        Chromium child emits and routes it here. We dispatch based on
        modifier + current layout mode:

          * **Ctrl + wheel**  → font size zoom (matches Ctrl+= / Ctrl+-
            shortcuts). Works in every layout.
          * **Plain wheel in paginated modes** (single / double page)
            → previous / next page. Scrolling up turns back, scrolling
            down turns forward — matches what Chromium would do to a
            scrollable page.
          * **Plain wheel in scroll modes** (single-chapter scroll,
            all-scroll) → fall back to a manual ``window.scrollBy``
            inside the page so the normal scrolling behaviour is
            preserved (we swallowed the event at the filter level, so
            Chromium won't do it for us).
        """
        if not delta_y:
            return
        # Ctrl + wheel: font zoom regardless of layout.
        try:
            ctrl = bool(modifiers & Qt.ControlModifier)
        except Exception:
            ctrl = False
        if ctrl:
            self._change_font_size(1 if delta_y > 0 else -1)
            return
        # Paginated modes: turn pages.
        if self._layout_mode in (LAYOUT_SINGLE, LAYOUT_DOUBLE):
            if delta_y > 0:
                self._prev_chapter()
            else:
                self._next_chapter()
            return
        # Scroll modes: replicate the browser's own wheel-scroll since
        # the event filter already swallowed the native behaviour. We
        # invert the sign because angleDelta y>0 means "wheel up" and
        # scrollBy expects positive values to scroll the page DOWN.
        if not _HAS_WEBENGINE:
            return
        pixels = int(-delta_y)
        js = f"window.scrollBy({{top: {pixels}, left: 0, behavior: 'auto'}});"
        try:
            if self._layout_mode == LAYOUT_DOUBLE:
                self._reader_left.page().runJavaScript(js)
                self._reader_right.page().runJavaScript(js)
            else:
                self._reader.page().runJavaScript(js)
        except Exception:
            logger.debug("Wheel scroll passthrough failed: %s",
                         traceback.format_exc())

    def _on_font_family_changed(self, family):
        """Update reader font family and re-render."""
        family = (family or "").strip()
        if not family or family == self._font_family:
            return
        self._font_family = family
        self._chapter_page_cache.clear()
        self._loaded_chapter = -1
        # Hide content before re-render to prevent flash
        _hide = "var c = document.getElementById('columns'); if (c) c.style.opacity = '0';"
        if _HAS_WEBENGINE:
            if self._layout_mode == LAYOUT_DOUBLE:
                self._reader_left.page().runJavaScript(_hide)
                self._reader_right.page().runJavaScript(_hide)
            else:
                self._reader.page().runJavaScript(_hide)
        self._render_current()

    def _on_spacing_changed(self, text):
        try:
            val = float(text)
            self._line_spacing = max(1.0, min(3.0, val))
        except ValueError:
            self._line_spacing = 1.8
        self._chapter_page_cache.clear()
        self._loaded_chapter = -1
        # Hide content before re-render to prevent flash
        _hide = "var c = document.getElementById('columns'); if (c) c.style.opacity = '0';"
        if _HAS_WEBENGINE:
            if self._layout_mode == LAYOUT_DOUBLE:
                self._reader_left.page().runJavaScript(_hide)
                self._reader_right.page().runJavaScript(_hide)
            else:
                self._reader.page().runJavaScript(_hide)
        self._render_current()

    def _on_theme_changed(self, index):
        self._theme_index = index
        # Update QWebEngineView page backgrounds to prevent flash
        if _HAS_WEBENGINE:
            from PySide6.QtGui import QColor
            t = self._get_theme()
            bg = QColor(t['bg'])
            for w in [self._reader, self._reader_left, self._reader_right]:
                if hasattr(w, 'page'):
                    w.page().setBackgroundColor(bg)
        self._apply_reader_style()
        self._loaded_chapter = -1  # force re-render for inline styles
        self._render_current()

    # ── Layout mode ────────────────────────────────────────────────────────

    def _on_layout_changed(self, index):
        modes = [LAYOUT_SINGLE, LAYOUT_DOUBLE, LAYOUT_SCROLL, LAYOUT_ALL]
        self._layout_mode = modes[index] if index < len(modes) else LAYOUT_SINGLE
        self._current_page = 0
        self._loaded_chapter = -1  # force re-render on layout change
        self._chapter_page_cache.clear()
        self._render_current()

    def _render_current(self):
        """Re-render the current chapter in the active layout mode."""
        if not self._chapters:
            return
        self._apply_reader_style()  # refresh theme

        row = self._current_row

        def _set_html(browser, html):
            """Set HTML on either QWebEngineView or QTextBrowser.

            QWebEngineView.setHtml() has a ~2MB limit — base64 images
            easily exceed this.  Write to a temp file and load via URL.
            """
            if _HAS_WEBENGINE:
                tmp_dir = _epub_cache_dir()
                tmp_path = os.path.join(tmp_dir, f"_reader_{id(browser)}.html")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(html)
                browser.setUrl(QUrl.fromLocalFile(tmp_path))
            else:
                browser.setHtml(html)

        if self._layout_mode == LAYOUT_ALL:
            self._reader_stack.setCurrentIndex(0)
            self._nav_bar.hide()
            all_html = ""
            for idx, (title, content) in enumerate(self._chapters):
                processed = self._process_html(content)
                all_html += f"<h2 style='color: {self._get_theme()['heading']}; border-bottom: 1px solid {self._get_theme()['border']}; padding-bottom: 6px; margin-top: 30px;'>Chapter {idx + 1}: {title}</h2>\n{processed}\n<hr style='border: none; border-top: 1px solid {self._get_theme()['border']}; margin: 20px 0;'>"
            _set_html(self._reader, self._wrap_html(all_html, paginated=False))
            self._loaded_chapter = -1
            self._toc_list.blockSignals(True)
            self._toc_list.setCurrentRow(-1)
            self._toc_list.blockSignals(False)

        elif self._layout_mode == LAYOUT_DOUBLE:
            self._reader_stack.setCurrentIndex(1)
            self._nav_bar.show()
            self._double_loads_pending = 2  # track both panes
            if row < len(self._chapters):
                html = self._process_html(self._chapters[row][1])
                full = self._wrap_html(html, paginated=True)
                _set_html(self._reader_left, full)
                _set_html(self._reader_right, full)
                self._loaded_chapter = row
                # finalization happens via loadFinished signal
            else:
                _set_html(self._reader_left, self._wrap_html("", paginated=True))
                _set_html(self._reader_right, self._wrap_html("", paginated=True))
            self._update_nav_buttons()

        elif self._layout_mode == LAYOUT_SINGLE:
            self._reader_stack.setCurrentIndex(0)
            self._nav_bar.show()
            if row < len(self._chapters):
                html = self._process_html(self._chapters[row][1])
                _set_html(self._reader, self._wrap_html(html, paginated=True))
                self._loaded_chapter = row
                # finalization happens via loadFinished signal
            self._update_nav_buttons()

        else:  # LAYOUT_SCROLL
            self._reader_stack.setCurrentIndex(0)
            self._nav_bar.hide()
            if row < len(self._chapters):
                html = self._process_html(self._chapters[row][1])
                _set_html(self._reader, self._wrap_html(html, paginated=False))
                self._loaded_chapter = row

    # ── Pagination helpers (CSS column-based) ───────────────────────────────

    # ── Right-click context menu on the reader pane ────────────────────

    def _get_reader_selection(self, browser) -> str:
        """Return the currently-selected text in *browser*, or ``""``.

        ``QWebEngineView.selectedText`` is synchronous (Qt >= 5.7) so it's
        safe to call from the context-menu handler. ``QTextBrowser``
        exposes the selection via its cursor instead. Both paths are
        wrapped in try/except so a Qt oddity can't block the menu.
        """
        try:
            if _HAS_WEBENGINE and hasattr(browser, "selectedText"):
                return browser.selectedText() or ""
            if hasattr(browser, "textCursor"):
                return browser.textCursor().selectedText() or ""
        except Exception:
            logger.debug("Reader selection read failed: %s",
                         traceback.format_exc())
        return ""

    def _show_reader_context_menu(self, browser, pos):
        """Populate + show the right-click menu for *browser* at *pos*.

        Menu contents are scoped to the reader's current flavor so the
        verb matches what the user is looking at:

          * **Raw** mode — the selection is source-language text, so the
            sole action is **Google Translate** (``translate.google.com``
            with ``sl=auto`` + ``tl=<target_language>``). This is the
            manual-MT sanity-check flow the reader was built for.
          * **Translated** mode — the selection is already in the target
            language, so instead we offer **Define on web**, a Google
            search with the ``define`` operator that surfaces the
            dictionary card for the selected word / phrase.

        Both entries are disabled when no text is selected so the menu
        reads clearly instead of silently doing nothing.
        """
        selected = (self._get_reader_selection(browser) or "").strip()
        has_selection = bool(selected)
        target_lang = (self._config.get("output_language") or "English").strip() or "English"
        target_code = _target_lang_to_google_code(target_lang)

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background: #1e1e2e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #e0e0e0; font-size: 9pt; padding: 4px; }
            QMenu::item { padding: 6px 20px; border-radius: 3px; }
            QMenu::item:selected { background: #3a3a5e; }
            QMenu::item:disabled { color: #555; }
        """)

        if getattr(self, "_show_raw", False):
            gt_action = menu.addAction(
                f"\U0001f310  Google Translate \u2192 {target_lang}")
            gt_action.setToolTip(
                f"Open translate.google.com in your default browser with the "
                f"selection machine-translated into {target_lang} "
                f"(source language auto-detected)."
            )
            gt_action.setEnabled(has_selection)
            gt_action.triggered.connect(
                lambda: self._open_google_translate(selected, target_code))
        else:
            def_action = menu.addAction("\U0001f4d6  Define on web")
            def_action.setToolTip(
                "Look up the selected word / phrase on Google using the "
                "'define' operator — surfaces the dictionary card above "
                "the regular search results."
            )
            def_action.setEnabled(has_selection)
            def_action.triggered.connect(
                lambda: self._open_web_define(selected))

        menu.exec(browser.mapToGlobal(pos))

    def _open_google_translate(self, text: str, target_code: str) -> None:
        """Hand off *text* to translate.google.com via the default browser."""
        text = (text or "").strip()
        if not text:
            return
        # URL-escape but keep spaces as '+' for readability in the address bar.
        from urllib.parse import quote
        from PySide6.QtGui import QDesktopServices
        encoded = quote(text, safe="")
        url = (
            f"https://translate.google.com/?sl=auto&tl={target_code}"
            f"&text={encoded}&op=translate"
        )
        try:
            QDesktopServices.openUrl(QUrl(url))
        except Exception:
            logger.debug("Google Translate open failed: %s",
                         traceback.format_exc())

    def _open_web_define(self, text: str) -> None:
        """Open Google's ``define:`` card for *text* in the default browser.

        Uses the ``define`` query prefix rather than a bare search so the
        dictionary entry (with pronunciation, part of speech, and
        definitions) renders at the top of the results page. For
        multi-word selections Google still surfaces the best-matching
        dictionary card; when no dictionary hit exists Google silently
        degrades to normal results.
        """
        text = (text or "").strip()
        if not text:
            return
        from urllib.parse import quote
        from PySide6.QtGui import QDesktopServices
        encoded = quote(f"define {text}", safe="")
        url = f"https://www.google.com/search?q={encoded}"
        try:
            QDesktopServices.openUrl(QUrl(url))
        except Exception:
            logger.debug("Web define open failed: %s",
                         traceback.format_exc())

    def _on_reader_load_finished(self, ok):
        """Called when QWebEngineView finishes loading HTML."""
        if not ok:
            return
        # Ignore stray load events that fire before a real chapter was
        # ever queued up. ``_make_reader_widget`` calls
        # ``setUrl("about:blank")`` to prime the view, which asynchronously
        # triggers ``loadFinished`` after :meth:`_setup_ui` has connected
        # this handler but before any chapter data is present. Without
        # this guard, :meth:`_finalize_single_page` runs against an
        # empty state and crashes the Python callback pipeline.
        if not self._chapters:
            return
        # Prime phase: the scroll-mode render has just completed. Swap
        # back to the configured paginated mode immediately — this is the
        # event-driven replacement for the previous fixed-delay timer.
        if (getattr(self, '_priming_initial_render', False)
                and self._layout_mode == LAYOUT_SCROLL
                and self.sender() is self._reader):
            self._layout_mode = getattr(self, '_prime_saved_mode', LAYOUT_SINGLE)
            self._loaded_chapter = -1
            self._chapter_page_cache.clear()
            self._render_current()
            return
        if self._layout_mode not in (LAYOUT_SINGLE, LAYOUT_DOUBLE):
            return
        # The signal fires from whichever browser finished loading; route
        # based on the sender rather than layout alone so stale loads from
        # the single-reader don't confuse the double-page pending counter.
        sender = self.sender()
        if self._layout_mode == LAYOUT_SINGLE:
            if sender is self._reader:
                self._finalize_single_page()
        elif self._layout_mode == LAYOUT_DOUBLE:
            if sender not in (self._reader_left, self._reader_right):
                return
            # Wait for both panes to finish loading
            pending = getattr(self, '_double_loads_pending', 0)
            if pending > 1:
                self._double_loads_pending = pending - 1
                return
            self._double_loads_pending = 0
            self._finalize_double_page()

    def _js_scroll_to(self, browser, page_num, animate: bool = True):
        """Navigate to a CSS column page by translating the #columns wrapper.

        When *animate* is False, the CSS transition is suppressed for the
        jump (used right after a load so the page doesn't visibly slide in
        from position 0). Translate values are rounded to whole pixels and
        use translate3d to keep text on a stable GPU layer — otherwise
        fractional offsets produce subpixel LCD-antialiasing fringing
        that looks like a red/colored shift on the text.
        """
        if _HAS_WEBENGINE:
            if animate:
                js = (
                    "var c = document.getElementById('columns');"
                    "if (c) {"
                    "  var w = (typeof _PAGE_W!=='undefined'&&_PAGE_W)?_PAGE_W:Math.floor(window.innerWidth);"
                    # Record the target page so _setupColumns can re-anchor
                    # the transform on the next viewport-width change.
                    f"  _CURRENT_PAGE = {page_num};"
                    f"  c.style.transform = 'translate3d(' + Math.round(-{page_num} * w) + 'px, 0, 0)';"
                    "}"
                )
            else:
                js = (
                    "var c = document.getElementById('columns');"
                    "if (c) {"
                    "  var w = (typeof _PAGE_W!=='undefined'&&_PAGE_W)?_PAGE_W:Math.floor(window.innerWidth);"
                    f"  _CURRENT_PAGE = {page_num};"
                    "  var _t = c.style.transition;"
                    "  c.style.transition = 'none';"
                    f"  c.style.transform = 'translate3d(' + Math.round(-{page_num} * w) + 'px, 0, 0)';"
                    "  void c.offsetHeight;"  # force reflow so the jump is committed w/o transition
                    "  c.style.transition = _t || 'transform 0.3s ease';"
                    "}"
                )
            browser.page().runJavaScript(js)
        else:
            vp = browser.viewport()
            h = vp.height()
            if h > 0:
                browser.verticalScrollBar().setValue(page_num * h)

    def _js_page_count(self, browser, callback):
        """Get page count from CSS column layout (#columns scrollWidth / page width)."""
        if _HAS_WEBENGINE:
            js = (
                "var c = document.getElementById('columns');"
                "var w = (typeof _PAGE_W!=='undefined'&&_PAGE_W)?_PAGE_W:window.innerWidth;"
                "c ? Math.max(1, Math.round(c.scrollWidth / w)) : 1;"
            )
            browser.page().runJavaScript(js, callback)
        else:
            doc = browser.document()
            vp = browser.viewport()
            w, h = vp.width(), vp.height()
            if w <= 0 or h <= 0:
                callback(1)
                return
            doc.setPageSize(QSizeF(w, h))
            callback(max(1, doc.pageCount()))

    def _apply_pending_page_hint(self, count: int) -> None:
        """Resolve ``_pending_page_hint`` (if any) against *count* pages.

        Called by the single / double finalizers after the fresh chapter's
        page count is known. The hint expresses "I was on the last page"
        or a proportional position so the Show-raw toggle can leave the
        user on an equivalent page in the new flavor (page counts drift
        between translations).
        """
        hint = getattr(self, '_pending_page_hint', None)
        if not hint:
            return
        # Only apply the hint if it was recorded against the chapter
        # we're now paginating — otherwise a stale hint from a prior
        # chapter could hijack the position.
        if hint.get("row") not in (None, self._current_row):
            self._pending_page_hint = None
            return
        c = max(1, int(count))
        if hint.get("was_last_page"):
            self._current_page = c - 1
        else:
            prop = float(hint.get("proportion") or 0.0)
            target = round(prop * (c - 1))
            self._current_page = max(0, min(int(target), c - 1))
        self._pending_page_hint = None

    def _finalize_single_page(self):
        """After HTML load: get page count and scroll to current page."""
        def on_count(count):
            count = int(count)
            self._chapter_page_cache[self._current_row] = count
            # Restore the pre-swap reading position when a Show-raw
            # toggle (or equivalent reload) staged a hint.
            self._apply_pending_page_hint(count)
            # animate=False: jump instantly so the reader doesn't visibly
            # slide from page 1 to the current page on theme/chapter change.
            self._js_scroll_to(self._reader, self._current_page, animate=False)
            self._js_reveal(self._reader)
            self._update_nav_buttons()
            self._reveal_reader_stack_after_prime()
        self._js_page_count(self._reader, on_count)

    def _finalize_double_page(self):
        """After HTML load: get page count and position both panes."""
        def on_count(count):
            count = int(count)
            self._chapter_page_cache[self._current_row] = count
            self._apply_pending_page_hint(count)
            self._js_scroll_to(self._reader_left, self._current_page, animate=False)
            self._js_scroll_to(self._reader_right, self._current_page + 1, animate=False)
            self._js_reveal(self._reader_left)
            self._js_reveal(self._reader_right)
            self._update_nav_buttons()
            self._reveal_reader_stack_after_prime()
        self._js_page_count(self._reader_left, on_count)

    def _reveal_reader_stack_after_prime(self):
        """Reveal the reader stack after the post-prime paginated render
        completes. No-op if the prime sequence wasn't used.
        """
        if not getattr(self, '_priming_initial_render', False):
            return
        self._priming_initial_render = False
        self._reader_stack.show()

    def _scroll_to_page_single(self):
        """Navigate single-page reader to current page."""
        self._js_scroll_to(self._reader, self._current_page)
        self._update_nav_buttons()

    def _scroll_to_page_double(self):
        """Navigate double-page panes to current page."""
        self._js_scroll_to(self._reader_left, self._current_page)
        self._js_scroll_to(self._reader_right, self._current_page + 1)
        self._update_nav_buttons()

    def _js_reveal(self, browser):
        """Show the #columns element after positioning."""
        if _HAS_WEBENGINE:
            browser.page().runJavaScript(
                "var c = document.getElementById('columns'); if (c) c.style.opacity = '1';"
            )

    def resizeEvent(self, event):
        """Invalidate page cache on resize, preserving reading position."""
        super().resizeEvent(event)
        self._refresh_pagination_viewport(delay=100)

    def _refresh_pagination_viewport(self, delay: int = 120):
        """Recompute paginated layout after viewport width changes.

        Used by resizeEvent and TOC toggle — anything that changes the
        reader widget's width without rebuilding the HTML.
        """
        if not hasattr(self, '_chapter_page_cache'):
            return
        if self._layout_mode not in (LAYOUT_SINGLE, LAYOUT_DOUBLE):
            return
        if not self._chapters:
            return
        # Save scroll proportion before clearing cache
        old_count = self._chapter_page_cache.get(self._current_row, 0)
        old_page = self._current_page
        self._chapter_page_cache.clear()
        # Proportion-based: map old position to new page count
        proportion = old_page / max(1, old_count) if old_count > 0 else 0
        # Hide content immediately to prevent image flash during resize
        _hide_js = "var c = document.getElementById('columns'); if (c) { c.style.transition = 'none'; c.style.opacity = '0'; }"
        _reveal_js = "var c = document.getElementById('columns'); if (c) { c.style.transition = 'transform 0.25s ease'; c.style.opacity = '1'; }"
        if self._layout_mode == LAYOUT_SINGLE:
            self._reader.page().runJavaScript(_hide_js)
        else:
            self._reader_left.page().runJavaScript(_hide_js)
            self._reader_right.page().runJavaScript(_hide_js)

        def _on_resize_recount():
            if self._layout_mode == LAYOUT_SINGLE:
                def on_count(count):
                    count = int(count)
                    self._chapter_page_cache[self._current_row] = count
                    self._current_page = min(max(0, round(proportion * count)), count - 1)
                    self._js_scroll_to(self._reader, self._current_page)
                    # Reveal after scroll
                    QTimer.singleShot(30, lambda: self._reader.page().runJavaScript(_reveal_js))
                    self._update_nav_buttons()
                self._js_page_count(self._reader, on_count)
            else:
                def on_count(count):
                    count = int(count)
                    self._chapter_page_cache[self._current_row] = count
                    self._current_page = min(max(0, round(proportion * count)), count - 1)
                    self._js_scroll_to(self._reader_left, self._current_page)
                    self._js_scroll_to(self._reader_right, self._current_page + 1)
                    QTimer.singleShot(30, lambda: [br.page().runJavaScript(_reveal_js)
                        for br in (self._reader_left, self._reader_right)])
                    self._update_nav_buttons()
                self._js_page_count(self._reader_left, on_count)
        QTimer.singleShot(delay, _on_resize_recount)

    def _update_nav_buttons(self):
        if self._layout_mode in (LAYOUT_SINGLE, LAYOUT_DOUBLE):
            ch_pages = self._get_chapter_pages(self._current_row)
            step = 2 if self._layout_mode == LAYOUT_DOUBLE else 1
            self._prev_btn.setEnabled(self._current_page > 0 or self._current_row > 0)
            self._next_btn.setEnabled(
                self._current_page + step < ch_pages or
                self._current_row < len(self._chapters) - 1
            )
            cur_global, total_global = self._get_global_page_info()
            self._page_label.setText(f"Page {cur_global}/{total_global}")
        else:
            self._prev_btn.setEnabled(self._current_row > 0)
            self._next_btn.setEnabled(self._current_row < len(self._chapters) - 1)
            self._page_label.setText(f"Chapter {self._current_row + 1} of {len(self._chapters)}")

    def _prev_chapter(self):
        if self._layout_mode in (LAYOUT_SINGLE, LAYOUT_DOUBLE):
            step = 2 if self._layout_mode == LAYOUT_DOUBLE else 1
            if self._current_page >= step:
                # Same chapter, just scroll
                self._current_page -= step
                if self._layout_mode == LAYOUT_SINGLE:
                    self._scroll_to_page_single()
                else:
                    self._scroll_to_page_double()
            elif self._current_row > 0:
                # Go to previous chapter's last page
                self._current_row -= 1
                self._toc_list.blockSignals(True)
                self._toc_list.setCurrentRow(self._current_row)
                self._toc_list.blockSignals(False)
                # Set page to last page of previous chapter (will be clamped in finalize)
                self._current_page = max(0, self._get_chapter_pages(self._current_row) - 1)
                self._render_current()
        else:
            new_row = max(0, self._current_row - 1)
            self._current_row = new_row
            self._toc_list.blockSignals(True)
            self._toc_list.setCurrentRow(new_row)
            self._toc_list.blockSignals(False)
            self._render_current()

    def _next_chapter(self):
        if self._layout_mode in (LAYOUT_SINGLE, LAYOUT_DOUBLE):
            step = 2 if self._layout_mode == LAYOUT_DOUBLE else 1
            ch_pages = self._get_chapter_pages(self._current_row)
            if self._current_page + step < ch_pages:
                # Same chapter, just scroll
                self._current_page += step
                if self._layout_mode == LAYOUT_SINGLE:
                    self._scroll_to_page_single()
                else:
                    self._scroll_to_page_double()
            elif self._current_row < len(self._chapters) - 1:
                # Next chapter, first page
                self._current_row += 1
                self._current_page = 0
                self._toc_list.blockSignals(True)
                self._toc_list.setCurrentRow(self._current_row)
                self._toc_list.blockSignals(False)
                self._render_current()
        else:
            new_row = min(len(self._chapters) - 1, self._current_row + 1)
            self._current_row = new_row
            self._toc_list.blockSignals(True)
            self._toc_list.setCurrentRow(new_row)
            self._toc_list.blockSignals(False)
            self._render_current()

    def closeEvent(self, event):
        """Persist reader settings back into config and flush to disk.

        The priming pass in :meth:`_on_epub_loaded_from_cache` temporarily
        flips ``_layout_mode`` to :data:`LAYOUT_SCROLL` while the first
        render happens (swapping back via ``loadFinished``). If the user
        closes the dialog before that swap-back lands, persisting the
        transient ``_layout_mode`` would poison the config with
        ``"scroll"`` — and since Scroll never triggers priming, every
        subsequent open would open in Scroll and stay there. Fall back
        to ``_prime_saved_mode`` whenever the priming flag is still set.

        After writing into the in-memory ``_config`` dict we also ask
        the translator parent to flush to ``config.json`` via
        :func:`_persist_config_via_parent` so the values actually
        survive to the next session — the main window's own save only
        runs on an orderly quit, and users' library / reader tweaks
        shouldn't hinge on that.
        """
        self._config['epub_reader_font_size'] = self._font_size
        self._config['epub_reader_line_spacing'] = self._line_spacing
        self._config['epub_reader_theme'] = self._theme_index
        if getattr(self, '_priming_initial_render', False):
            self._config['epub_reader_layout'] = getattr(
                self, '_prime_saved_mode', self._layout_mode)
        else:
            self._config['epub_reader_layout'] = self._layout_mode
        self._config['epub_reader_font_family'] = self._font_family
        self._config['epub_reader_show_raw'] = self._show_raw
        _persist_config_via_parent(self)
        super().closeEvent(event)

    # ── Chapter rendering ───────────────────────────────────────────────

    def _on_show_raw_toggled(self, checked: bool):
        """Swap between raw and translated content.

        Two backing modes:
          * **Overlay mode** — both raw and overlaid chapter lists were
            precomputed at load time, so this just flips ``_chapters``
            between them, rebuilds the TOC (titles differ between
            flavors), invalidates the paginated-page cache (page counts
            differ between translations) and re-renders. Cheap, no I/O.
          * **Dual-path mode** — the caller attached an alt EPUB (raw
            counterpart of a compiled Completed-tab book). Flipping the
            toggle swaps ``_epub_path`` between the compiled and raw
            files and restarts the loader via
            :meth:`_reload_epub_from_active_path`.

        Reading position survives the swap in both modes via
        :attr:`_pending_page_hint` / :attr:`_reload_position_hint`:
        the finalizer after pagination consumes the hint and positions
        the reader either at the same proportional page or — if the
        user was on the last page before the swap — at the last page
        of the new flavor (page counts differ between translations).
        """
        new_value = bool(checked)
        if new_value == self._show_raw:
            return
        self._show_raw = new_value
        try:
            self._config['epub_reader_show_raw'] = self._show_raw
        except Exception:
            pass
        # Snapshot the reading position so the finalizer can restore it
        # against the (potentially differently-paginated) new content.
        position_hint = self._capture_position_hint()
        # Dual-path mode takes precedence: it's the only meaningful
        # interpretation when no overlay was supplied. We also fall
        # through to it when overlay mode has nothing loaded yet (e.g.
        # toggle clicked mid-load before chapters exist).
        has_overlay = bool(self._chapters_overlaid) and any(
            (self._chapters_overlaid[i] != self._chapters_raw[i])
            for i in range(min(len(self._chapters_overlaid),
                               len(self._chapters_raw)))
        )
        if self._raw_epub_alt_path and not has_overlay:
            target = (self._raw_epub_alt_path if self._show_raw
                      else self._translated_epub_path)
            if not target or not os.path.isfile(target):
                return
            self._epub_path = target
            # Reload pipeline consumes this hint to pick the same row
            # and hand the page portion to the finalizer.
            self._reload_position_hint = position_hint
            self._reload_epub_from_active_path()
            return
        # Overlay mode — in-memory chapter swap.
        if not self._chapters_raw and not self._chapters_overlaid:
            return
        self._chapters = (self._chapters_raw if self._show_raw
                          else self._chapters_overlaid)
        # Rebuild TOC entries so titles reflect the active flavor. Hold
        # the current row so we don't lose the reading position.
        current_row = max(0, min(self._current_row, len(self._chapters) - 1))
        self._toc_list.blockSignals(True)
        self._toc_list.clear()
        for title, _ in self._chapters:
            self._toc_list.addItem(QListWidgetItem(title))
        self._toc_list.setCurrentRow(current_row)
        self._toc_list.blockSignals(False)
        self._current_row = current_row
        # Force a fresh paginated render: page-count caches differ
        # between translations and the currently-loaded chapter needs to
        # be re-set from the new source. The finalizer will consult
        # ``_pending_page_hint`` to pick a sensible page number once
        # the new pagination lands.
        self._chapter_page_cache = {}
        self._loaded_chapter = -1
        self._pending_page_hint = position_hint
        if self._chapters:
            self._render_current()

    def _capture_position_hint(self) -> dict:
        """Snapshot the current reading position for later restoration.

        Returned dict carries:
          * ``row``          — the current chapter index.
          * ``was_last_page`` — True when the user was on the final page
            of the chapter (so the finalizer can jump to the last page
            of the new flavor regardless of page-count drift).
          * ``proportion``   — relative progress through the chapter
            [0, 1], used when ``was_last_page`` is False to pick the
            closest equivalent page in the new pagination.
        """
        row = int(getattr(self, '_current_row', 0) or 0)
        page = int(getattr(self, '_current_page', 0) or 0)
        pages = int(self._get_chapter_pages(row)) if self._chapters else 0
        was_last = pages > 0 and page >= pages - 1
        if pages > 1:
            proportion = max(0.0, min(1.0, page / (pages - 1)))
        else:
            proportion = 0.0
        return {
            "row": row,
            "was_last_page": bool(was_last),
            "proportion": float(proportion),
        }

    def _reload_epub_from_active_path(self):
        """Restart the loader pipeline against the current ``_epub_path``.

        Called by the Show-raw toggle in dual-path mode: after swapping
        ``_epub_path`` to the raw (or back to the compiled) EPUB, we need
        to reset chapter / image state and re-run the spinner + loader
        exactly like the initial open sequence. The temp image directory
        is left behind on purpose — it's keyed by the active epub_path
        and will be re-created on the first ``_process_html`` call.

        If ``_reload_position_hint`` is set (by the Show-raw toggle), it
        is preserved across the reset so :meth:`_on_epub_loaded_from_cache`
        can seed the initial row / page from the user's pre-swap position
        instead of defaulting to the first chapter.
        """
        self._chapters = []
        self._chapters_raw = []
        self._chapters_overlaid = []
        self._chapter_filenames = []
        self._images = {}
        self._chapter_page_cache = {}
        self._loaded_chapter = -1
        self._current_page = 0
        if hasattr(self, "_img_temp_dir"):
            # Clear so _process_html picks a fresh per-EPUB temp dir.
            try:
                del self._img_temp_dir
            except AttributeError:
                pass
        self._toc_list.blockSignals(True)
        self._toc_list.clear()
        self._toc_list.blockSignals(False)
        self._start_loading()

    def _on_chapter_selected(self, row):
        if row < 0 or row >= len(self._chapters):
            return
        self._current_row = row
        self._current_page = 0  # reset pagination when selecting a new chapter
        self._render_current()

    def _process_html(self, html_content: str) -> str:
        """Process chapter HTML: resolve image paths to temp files."""
        try:
            from bs4 import BeautifulSoup

            # Ensure temp image directory exists for this EPUB
            if not hasattr(self, '_img_temp_dir') or not os.path.isdir(self._img_temp_dir):
                epub_hash = hashlib.md5(self._epub_path.encode()).hexdigest()[:10]
                self._img_temp_dir = os.path.join(
                    tempfile.gettempdir(), "Glossarion_EpubImages", epub_hash)
                os.makedirs(self._img_temp_dir, exist_ok=True)

            soup = BeautifulSoup(html_content, "html.parser")

            for img_tag in soup.find_all("img"):
                src = img_tag.get("src", "")
                if not src:
                    continue
                image_data = None
                candidates = [src, os.path.basename(src), src.lstrip("../"), src.lstrip("./")]
                for candidate in candidates:
                    if candidate in self._images:
                        image_data = self._images[candidate]
                        break
                if image_data:
                    # Write to temp file (skip if already cached)
                    safe_name = os.path.basename(src).replace("/", "_").replace("\\", "_")
                    img_path = os.path.join(self._img_temp_dir, safe_name)
                    if not os.path.isfile(img_path):
                        with open(img_path, "wb") as f:
                            f.write(image_data)
                    img_tag["src"] = QUrl.fromLocalFile(img_path).toString()
                    # Wrap sizeable images in full-page containers
                    # (skip tiny icons/bullets — use byte-size as fast proxy)
                    if len(image_data) > 5120:
                        wrapper = soup.new_tag("div")
                        wrapper["class"] = "full-page-img"
                        # Find the block-level container of this img
                        # (typically <p><img/></p> or <div><img/></div>)
                        container = img_tag
                        if img_tag.parent and img_tag.parent.name in ('p', 'div', 'figure'):
                            container = img_tag.parent
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
                        # Extract siblings, wrap container, then re-insert in order
                        for el in to_pull:
                            el.extract()
                        container.wrap(wrapper)
                        for el in reversed(to_pull):
                            wrapper.insert(0, el)

            return str(soup)
        except Exception:
            logger.debug("HTML processing failed: %s", traceback.format_exc())
            return html_content

    def _wrap_html(self, body_html: str, paginated: bool = False) -> str:
        """Wrap processed HTML in a full styled document.

        When *paginated* is True, a proper CSS multi-column layout is used:
          html/body — zero-padded, overflow:hidden (viewport clip)
          #columns  — column layout container (translateX for navigation)
          #content  — inner padding for readability
        """
        t = self._get_theme()
        # Build a CSS font stack: user-selected family first, then common
        # fallbacks so missing fonts degrade gracefully. Any embedded single
        # quotes in the family name are stripped to keep the stylesheet valid.
        _fam = (self._font_family or 'Georgia').replace("'", "").strip() or 'Georgia'
        _is_mono = _fam.lower() in {'consolas', 'courier new', 'courier', 'menlo',
                                    'monaco', 'lucida console', 'cascadia mono',
                                    'cascadia code', 'source code pro', 'fira code'}
        _generic = 'monospace' if _is_mono else 'serif'
        _font_stack = f"'{_fam}', 'Georgia', 'Noto Serif', {_generic}"
        # Use px units (integer device pixels) for sharper glyph rasterization.
        # 1pt = 1/72 inch, 1px = 1/96 inch → px = pt * 96/72.
        _font_px = int(round(self._font_size * 96 / 72))
        if paginated:
            return (
                f"<html><head><style>"
                f"* {{ box-sizing: border-box; }}"
                # Grayscale AA + geometricPrecision kill the subpixel LCD
                # fringing ("red shift") that appears on text inside a
                # GPU-composited transformed layer. This trade-off is
                # intentional for paginated modes; see _js_scroll_to.
                f"html, body {{ margin: 0; padding: 10px 0; overflow: hidden; "
                f"background: {t['bg']}; color: {t['fg']}; "
                f"-webkit-font-smoothing: antialiased; "
                f"-moz-osx-font-smoothing: grayscale; "
                f"text-rendering: geometricPrecision; "
                f"-webkit-text-size-adjust: 100%; }}"
                f"#columns {{ column-fill: auto; column-gap: 0; "
                f"transition: transform 0.3s ease; opacity: 0; "
                # will-change + backface-visibility stabilize the compositor
                # layer so text isn't re-rasterized at fractional offsets.
                f"will-change: transform; backface-visibility: hidden; "
                f"transform: translate3d(0, 0, 0); "
                f"font-family: {_font_stack}; "
                f"font-size: {_font_px}px; line-height: {self._line_spacing}; }}"
                f"#content {{ padding: 0 40px; }}"
                f"h1, h2, h3, h4, h5, h6 {{ color: {t['heading']}; margin: 0; padding: 0; }}"
                f"img {{ display: block; max-width: 100%; max-height: calc(100vh - 60px); "
                f"height: auto; object-fit: contain; "
                f"border-radius: 4px; margin: 12px auto; break-inside: avoid; }}"
                f".full-page-img {{ break-inside: avoid; "
                f"display: flex; flex-direction: column; align-items: center; justify-content: center; "
                f"min-height: calc(100vh - 40px); overflow: hidden; "
                f"padding: 0; margin: 0; }}"
                f".full-page-img + .full-page-img {{ margin-top: 0; }}"
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
                f"  _PAGE_W = Math.floor(window.innerWidth);"
                f"  c.style.columnWidth = _PAGE_W + 'px';"
                f"  c.style.height = (window.innerHeight - 20) + 'px';"
                # Re-apply the current page offset to the new column width
                # atomically (with transition suppressed) so the user never
                # sees a stale/mis-aligned frame.
                f"  var _t = c.style.transition;"
                f"  c.style.transition = 'none';"
                f"  c.style.transform = 'translate3d(' + Math.round(-_CURRENT_PAGE * _PAGE_W) + 'px, 0, 0)';"
                f"  void c.offsetHeight;"
                f"  c.style.transition = _t || 'transform 0.3s ease';"
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
            return (
                f"<html><head><style>"
                f"body {{ background: {t['bg']}; color: {t['fg']}; "
                f"font-family: {_font_stack}; "
                f"font-size: {_font_px}px; line-height: {self._line_spacing}; "
                f"-webkit-font-smoothing: auto; "
                f"-moz-osx-font-smoothing: auto; "
                f"text-rendering: optimizeLegibility; "
                f"-webkit-text-size-adjust: 100%; "
                f"padding: 10px 20px; margin: 0 auto; }}"
                f"h1, h2, h3 {{ color: {t['heading']}; }}"
                f"img {{ display: block; max-width: 100%; height: auto; "
                f"border-radius: 4px; margin: 12px auto; }}"
                f"p {{ margin: 0.6em 0; }}"
                f"a {{ color: {t['link']}; }}"
                f"code {{ background: {t['code_bg']}; padding: 1px 4px; border-radius: 3px; }}"
                f"::-webkit-scrollbar {{ width: 8px; }}"
                f"::-webkit-scrollbar-track {{ background: {t['bg']}; }}"
                f"::-webkit-scrollbar-thumb {{ background: {t['border']}; border-radius: 4px; }}"
                f"</style></head><body>{body_html}</body></html>"
            )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dlg = EpubLibraryDialog()
    dlg.exec()
