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
from PySide6.QtCore import Qt, QSize, QRect, QRectF, Signal, Slot, QThread, QTimer, QSizeF, QPointF, QUrl
from PySide6.QtGui import QPixmap, QFont, QFontMetrics, QIcon, QImage, QCursor, QShortcut, QKeySequence, QTransform, QTextLayout, QTextOption

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


# Names of tracking / registry files that live alongside the library
# shelves and must NEVER be mistaken for content by the scanners,
# the Undo orphan pass, or the count helpers. Keeping these as plain
# filenames (not paths) lets the same blocklist catch both the new
# ``Library/*.txt`` location and any legacy copy still inside
# ``Library/Raw`` from an earlier build.
_LIBRARY_TRACKING_FILENAMES = frozenset({
    "library_raw_inputs.txt",
    "library_translated_inputs.txt",
    "library_origins.txt",
})


def get_library_raw_inputs_file() -> str:
    """Return ``Library/library_raw_inputs.txt`` — one path per line.

    Tracks every raw input file that has been run through the translator.
    The list is append-only; duplicates are collapsed on read.

    Sits at the Library root (next to ``library_translated_inputs.txt``
    and ``library_origins.txt``) rather than inside ``Library/Raw`` so
    the content-scanning code doesn't mistake the registry for a raw
    source file. A legacy build wrote it into ``Library/Raw``; the
    migration below moves any surviving copy up to the root on first
    access so existing users don't lose their registry state.
    """
    new_path = os.path.join(get_library_dir(), "library_raw_inputs.txt")
    legacy_path = os.path.join(
        get_library_raw_dir(), "library_raw_inputs.txt")
    try:
        legacy_exists = os.path.isfile(legacy_path)
        new_exists = os.path.isfile(new_path)
        if legacy_exists and not new_exists:
            # Promote the legacy copy to the new location.
            try:
                shutil.move(legacy_path, new_path)
            except OSError:
                # Fallback: best-effort copy + remove so the legacy
                # path doesn't keep tripping the content scanners.
                try:
                    shutil.copyfile(legacy_path, new_path)
                    os.remove(legacy_path)
                except OSError:
                    logger.debug(
                        "library_raw_inputs migration failed: %s",
                        traceback.format_exc())
        elif legacy_exists and new_exists:
            # Both exist — merge the legacy entries into the new file
            # then drop the legacy copy so future scans only hit one.
            try:
                merged: list[str] = []
                seen: set[str] = set()
                for src in (new_path, legacy_path):
                    try:
                        with open(src, "r", encoding="utf-8") as f:
                            for line in f:
                                p = line.strip()
                                if not p:
                                    continue
                                key = os.path.normcase(os.path.normpath(
                                    os.path.abspath(p)))
                                if key in seen:
                                    continue
                                seen.add(key)
                                merged.append(p)
                    except OSError:
                        continue
                with open(new_path, "w", encoding="utf-8") as f:
                    for p in merged:
                        f.write(p + "\n")
                try:
                    os.remove(legacy_path)
                except OSError:
                    pass
            except Exception:
                logger.debug(
                    "library_raw_inputs merge failed: %s",
                    traceback.format_exc())
    except Exception:
        logger.debug(
            "library_raw_inputs path resolve failed: %s",
            traceback.format_exc())
    return new_path


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


def remove_library_raw_input(path: str) -> None:
    """Drop *path* from the raw-inputs registry.

    Mirrors :func:`remove_library_translated_input` for the raw side
    so the Delete handler can unregister a card whose backing file
    lives outside the Library / output roots: the physical file stays
    where the user put it, but the flash card disappears from the
    scanner output because its registration is gone.
    """
    if not path:
        return
    try:
        key = os.path.normcase(os.path.normpath(os.path.abspath(path)))
    except (TypeError, ValueError):
        return
    remaining = [
        p for p in load_library_raw_inputs()
        if os.path.normcase(os.path.normpath(os.path.abspath(p))) != key
    ]
    reg = get_library_raw_inputs_file()
    try:
        with open(reg, "w", encoding="utf-8") as f:
            for p in remaining:
                f.write(p + "\n")
    except OSError:
        logger.debug("Could not rewrite %s", reg)


def get_library_translated_inputs_file() -> str:
    """Return ``Library/library_translated_inputs.txt`` — one path per line.

    Tracks every *compiled* EPUB that has been registered with the
    Library via drag-drop / Import but **not yet physically moved**
    into ``Library/Translated``. Mirrors :func:`get_library_raw_inputs_file`
    for the raw side. The scanner reads this file to surface those
    files on the Completed tab even though they live outside the
    library folder; Organize later moves them and prunes their entry.
    """
    return os.path.join(get_library_dir(), "library_translated_inputs.txt")


def load_library_translated_inputs() -> list[str]:
    """Return the deduplicated list of registered translated paths."""
    path = get_library_translated_inputs_file()
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


def record_library_translated_input(path: str) -> None:
    """Append *path* to the translated-inputs registry.

    No-op on failure / duplicate. Mirrors
    :func:`record_library_raw_input` but targets the translated-side
    registry.
    """
    if not path:
        return
    try:
        abs_path = os.path.abspath(path)
    except (TypeError, ValueError):
        return
    existing = {
        os.path.normcase(os.path.normpath(os.path.abspath(p)))
        for p in load_library_translated_inputs()
    }
    if os.path.normcase(os.path.normpath(abs_path)) in existing:
        return
    reg = get_library_translated_inputs_file()
    try:
        with open(reg, "a", encoding="utf-8") as f:
            f.write(abs_path + "\n")
    except OSError:
        logger.debug("Could not append to %s", reg)


def remove_library_translated_input(path: str) -> None:
    """Drop *path* from the translated-inputs registry.

    Called when Organize moves the file into ``Library/Translated`` —
    the in-place registration is no longer meaningful once the
    compiled EPUB has been promoted to the curated shelf.
    """
    if not path:
        return
    try:
        key = os.path.normcase(os.path.normpath(os.path.abspath(path)))
    except (TypeError, ValueError):
        return
    remaining = [
        p for p in load_library_translated_inputs()
        if os.path.normcase(os.path.normpath(os.path.abspath(p))) != key
    ]
    reg = get_library_translated_inputs_file()
    try:
        with open(reg, "w", encoding="utf-8") as f:
            for p in remaining:
                f.write(p + "\n")
    except OSError:
        logger.debug("Could not rewrite %s", reg)


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
    """Load the structured origins mapping (auto-upgrades the legacy format).

    Also sanitizes the loaded mapping by dropping any entry whose key
    is a library registry filename (``library_raw_inputs.txt``,
    ``library_translated_inputs.txt``, ``library_origins.txt``).
    Legacy builds placed ``library_raw_inputs.txt`` inside ``Library/Raw``
    where the organize scan happily treated it as a raw source and wrote
    a ghost entry into ``origins['raw']``. Undo then reported
    ``raw:library_raw_inputs.txt: not found in Library/Raw`` on every
    restore because the registry file had since been promoted to the
    library root. Filtering here means any pre-existing garbage entry
    is forgotten on first read, and the sanitized dict is flushed back
    to disk so the next load stays clean.
    """
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
        data = {"version": 3, "raw": {}, "translated": dict(data), "pairs": {}}
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

    # Drop tracking-file ghosts from every bucket. ``pairs`` holds raw
    # basenames as values too, so a legacy entry pointing at the
    # registry file must also be purged there.
    dirty = False
    for bucket in ("raw", "translated"):
        sanitized = {
            k: v for k, v in data[bucket].items()
            if k and k.lower() not in _LIBRARY_TRACKING_FILENAMES
        }
        if len(sanitized) != len(data[bucket]):
            data[bucket] = sanitized
            dirty = True
    sanitized_pairs = {
        k: v for k, v in data["pairs"].items()
        if k and k.lower() not in _LIBRARY_TRACKING_FILENAMES
        and (not isinstance(v, str)
             or v.lower() not in _LIBRARY_TRACKING_FILENAMES)
    }
    if len(sanitized_pairs) != len(data["pairs"]):
        data["pairs"] = sanitized_pairs
        dirty = True
    if dirty:
        try:
            _save_origins(data)
        except Exception:
            logger.debug("origins sanitize-flush failed: %s",
                         traceback.format_exc())
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
    """Reveal *path* in the OS file manager (non-blocking).

    The actual spawn (``subprocess.Popen`` / ``os.startfile``) runs on
    a short-lived daemon thread because ``CreateProcess`` /
    ``ShellExecuteEx`` on Windows can stall the caller for 0.5–1 s
    while explorer.exe initialises COM and resolves the target. On
    the Qt main thread that shows up as a visible freeze between
    clicking a "Reveal source file" / "Open Output Folder" action and
    the Explorer window actually appearing. Off-loading the spawn
    means the click handler returns immediately and Qt can paint the
    next frame while Windows is still bringing Explorer up.

    On Windows the subprocess call also passes ``CREATE_NO_WINDOW``
    so the spawned helper doesn't flash a console window beside the
    cursor before Explorer paints its pane.
    """
    if not path:
        return
    # Snapshot the values the worker needs so it doesn't poke at
    # shared state from a background thread.
    _no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    try:
        is_file = os.path.isfile(path)
    except OSError:
        is_file = False
    folder = os.path.dirname(path) if is_file else path
    normalized = os.path.normpath(path) if is_file else path
    system_name = platform.system()

    def _worker():
        try:
            if system_name == "Windows":
                if is_file:
                    subprocess.Popen(
                        ["explorer", "/select,", normalized],
                        creationflags=_no_window,
                    )
                else:
                    os.startfile(folder)
            elif system_name == "Darwin":
                subprocess.Popen(
                    ["open", "-R", path] if is_file else ["open", folder]
                )
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as exc:
            logger.warning("Failed to open folder: %s\n%s",
                           exc, traceback.format_exc())

    import threading as _threading
    _threading.Thread(
        target=_worker,
        name="OpenFolderInExplorer",
        daemon=True,
    ).start()


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def _resolve_output_roots(config: dict | None = None) -> list[str]:
    """Return every directory the translator may have written output folders into.

    Both the configured override (``OUTPUT_DIRECTORY`` env var or
    ``config['output_directory']``) AND the default fallback location
    (app dir on Windows, CWD elsewhere) are returned when they exist
    on disk, so the In Progress + Completed tabs surface flash cards
    from either path in the same scan. Results are de-duplicated by
    normalized absolute path so an override pointing at the same dir
    as the fallback doesn't produce two scan passes.

    Previously this helper treated the override as *strict* — when
    set, the fallback was excluded. Users reported losing access to
    older workspaces still sitting in the fallback location after
    setting an override, so the helper now unions the two.
    """
    config = config or {}
    roots: list[str] = []
    seen: set[str] = set()

    def _add(candidate: str) -> None:
        if not candidate:
            return
        try:
            abs_p = os.path.abspath(candidate)
        except (TypeError, ValueError):
            return
        if not os.path.isdir(abs_p):
            return
        key = os.path.normcase(os.path.normpath(abs_p))
        if key in seen:
            return
        seen.add(key)
        roots.append(abs_p)

    # Override first so it wins priority when both locations hold the
    # same workspace basename (the scanner dedups workspaces by
    # folder-level key, first-seen wins in :func:`scan_output_folders`).
    override = os.environ.get("OUTPUT_DIRECTORY") or config.get("output_directory")
    _add(override)

    default_dir = _default_output_root()
    _add(default_dir)

    return roots


def _default_output_root() -> str:
    """Return the implicit default output root used when no override is set.

    Mirrors the fallback rule inside :func:`_resolve_output_roots`: on
    Windows, the frozen app's own directory (or the source file's dir
    in dev runs); on other platforms, the current working directory.
    Surfacing it as a standalone helper lets the "Load for translation"
    flow compare a flash card's backing output folder against the
    active override even when the override is empty (i.e. "use the
    default") — the two cases are indistinguishable without this value.
    """
    if platform.system() == "Windows":
        if getattr(sys, "frozen", False):
            return os.path.dirname(sys.executable)
        return os.path.dirname(os.path.abspath(__file__))
    return os.getcwd()


def _expected_output_root_for_book(book: dict) -> str:
    """Return the output root directory a flash card's translation lives
    under, or ``""`` when the book doesn't carry a resolvable
    ``output_folder``.

    A "output root" here is the PARENT of the workspace folder —
    i.e. whichever of ``OUTPUT_DIRECTORY`` / the default fallback root
    produced this card during :func:`scan_output_folders`. Library-
    organized cards (``in_library=True``) typically have no active
    ``output_folder`` so this helper returns ``""`` for them — the
    caller should treat that as "no mismatch to check" rather than as
    a real root.
    """
    if not isinstance(book, dict):
        return ""
    out = book.get("output_folder") or ""
    if not out:
        return ""
    try:
        out_abs = os.path.abspath(out)
    except Exception:
        return ""
    if not os.path.isdir(out_abs):
        return ""
    parent = os.path.dirname(out_abs)
    return parent if parent and os.path.isdir(parent) else ""


def _output_paths_equal(a: str, b: str) -> bool:
    """Case-insensitive, normalized equality for two filesystem paths.

    Treats two empty strings as equal so the "no override configured"
    state compares cleanly against itself.
    """
    if not a and not b:
        return True
    if not a or not b:
        return False
    try:
        return (os.path.normcase(os.path.normpath(os.path.abspath(a)))
                == os.path.normcase(os.path.normpath(os.path.abspath(b))))
    except Exception:
        return False


# Characters Windows strips / mangles in filenames that the translator
# happily leaves in a book's ``metadata.json`` title. We remove them
# from both sides before comparing so a title like ``"Foo: Bar."`` in
# metadata still matches the on-disk filename ``"Foo Bar.epub"``.
_FILENAME_STRIP_CHARS = '<>:"/\\|?*'


# Per-process cache of EPUB title strings keyed by ``path|mtime`` so a
# repeated scan (auto-refresh, undo, organize, …) doesn't re-open the
# zip for every library EPUB every time.
_EPUB_TITLES_CACHE: dict[str, tuple[str, ...]] = {}


def _extract_epub_titles(epub_path: str) -> tuple[str, ...]:
    """Return title-like strings embedded in *epub_path*'s OPF metadata.

    Reads only the OPF (no content decode), pulling ``dc:title``,
    ``dc:alternative``, and the ``calibre:original_title`` meta element
    the compiler writes at build time (see ``_create_book`` in
    ``epub_converter.py`` where the translator stores the raw source
    title under that field). The returned tuple deliberately keeps
    entries raw — callers pass each through :func:`_norm_book_key`
    before using them as match keys.

    Cached per ``(path, mtime)`` so repeat scans are cheap. Returns an
    empty tuple on any failure — a missing title is never fatal for
    the scanner.
    """
    if not epub_path or not os.path.isfile(epub_path):
        return ()
    try:
        mtime = os.path.getmtime(epub_path)
    except OSError:
        mtime = 0
    cache_key = f"{epub_path}|{mtime}"
    cached = _EPUB_TITLES_CACHE.get(cache_key)
    if cached is not None:
        return cached
    titles: list[str] = []
    try:
        import zipfile
        from xml.etree import ElementTree as ET
        with zipfile.ZipFile(epub_path, "r") as zf:
            names = zf.namelist()
            names_set = set(names)
            opf_path = None
            if "META-INF/container.xml" in names_set:
                try:
                    container_xml = zf.read(
                        "META-INF/container.xml").decode(
                            "utf-8", errors="replace")
                    ctree = ET.fromstring(container_xml)
                    ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
                    rootfile = ctree.find(".//c:rootfile", ns)
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
                opf_xml = zf.read(opf_path).decode(
                    "utf-8", errors="replace")
                otree = ET.fromstring(opf_xml)
                DC = "http://purl.org/dc/elements/1.1/"
                OPF = "http://www.idpf.org/2007/opf"
                for tag in ("title", "alternative"):
                    for el in otree.findall(f".//{{{DC}}}{tag}"):
                        txt = (el.text or "").strip()
                        if txt:
                            titles.append(txt)
                # Calibre-style ``<meta name="calibre:original_title"
                # content="…"/>`` — this is how the translator stores
                # the raw source title when compiling, so it's the
                # single most useful signal for pairing a
                # translated-name library EPUB back to its
                # raw-name workspace.
                for meta_el in otree.findall(f".//{{{OPF}}}meta"):
                    name = meta_el.get("name") or ""
                    if name.lower() in ("calibre:original_title",
                                        "original_title"):
                        content = meta_el.get("content") or ""
                        content = content.strip()
                        if content:
                            titles.append(content)
    except Exception:
        logger.debug("OPF title extraction failed for %s: %s",
                     epub_path, traceback.format_exc())
    result = tuple(dict.fromkeys(titles))  # preserve order, drop dupes
    _EPUB_TITLES_CACHE[cache_key] = result
    return result


def _norm_book_key(value: str) -> str:
    """Normalize a book title/filename for cross-source comparison.

    The scanner links a Library/Translated EPUB to its originating
    workspace through several potentially-misaligned sources:

      * the workspace folder name (source-language raw stem)
      * the resolved raw source stem (also source-language)
      * the workspace's ``metadata.json`` title fields
        (``title`` is usually the *translated* title after compile)
      * the library EPUB filename stem (translated title, post
        Windows filename sanitization)

    Different call-sites produce slightly different strings for the
    same book — e.g. metadata might hold
    ``"… Romance Fantasy."`` (trailing period) while the on-disk
    EPUB is ``"… Romance Fantasy.epub"`` (period stripped because
    NTFS forbids trailing dots). Normalizing through this helper
    makes those two strings equal so the pairing succeeds.

    Rules:
      * strip characters disallowed in Windows filenames
      * collapse whitespace runs into single spaces
      * strip leading/trailing whitespace AND trailing periods
        (NTFS silently drops trailing ``.``)
      * casefold for Unicode-aware lowercase comparison

    Returns ``""`` when the input doesn't contain anything meaningful
    so callers can skip the key instead of collapsing onto the empty
    string (which would spuriously match unrelated unnamed entries).
    """
    if not value:
        return ""
    import re as _re
    s = str(value)
    # Drop Windows-illegal chars so on-disk filenames and in-memory
    # titles collapse to the same key.
    s = s.translate({ord(c): " " for c in _FILENAME_STRIP_CHARS})
    # Collapse whitespace runs.
    s = _re.sub(r"\s+", " ", s)
    # Drop trailing periods + whitespace (NTFS / FAT strip trailing
    # dots from filenames, so a metadata title ending in ``.`` won't
    # match the on-disk basename unless we normalize it away).
    s = s.strip().rstrip(". \t")
    return s.casefold()


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


def _resolve_book_source_file(book: dict) -> str:
    """Return the raw source file path the 🔗 button should reveal, or ``""``.

    Mirrors :class:`BookDetailsDialog._resolve_source_file_target`
    (and is used by it) so the Book Details dialog and the card
    context menu both resolve sources through one code path:

      1. ``book['raw_source_path']`` — populated by the scanner for
         every resolvable card.
      2. For ``Library/Translated`` entries, re-run
         :func:`_find_raw_source_for_library_epub` so cards whose
         scan result predates the origins ``pairs`` entry can still
         find the matching raw.

    Returns ``""`` when no RAW source can be resolved — the caller
    (context menu / 🔗 button) omits / disables the action in that
    case instead of revealing a misleading target. Previously this
    fell back to ``book['path']``, which for ``Library/Translated``
    entries is the *compiled* EPUB sitting inside the translated
    shelf — so the action appeared to "work" but just opened the
    translated subfolder (identical to the "Reveal Translated File"
    action, which is the opposite of what the user asked for).

    Caches any fresh resolution back onto the book dict so repeat
    calls (and the reader's Raw toggle) skip the lookup.
    """
    if not isinstance(book, dict):
        return ""
    path = book.get("raw_source_path", "") or ""
    if path and os.path.isfile(path):
        return path
    if book.get("in_library"):
        lib_path = book.get("path", "") or ""
        try:
            resolved = _find_raw_source_for_library_epub(lib_path)
        except Exception:
            resolved = ""
            logger.debug("Source-file library lookup failed: %s",
                         traceback.format_exc())
        if resolved and os.path.isfile(resolved):
            book["raw_source_path"] = resolved
            return resolved
    return ""


def _resolve_book_translated_file(book: dict) -> str:
    """Return the compiled / translated EPUB path to reveal, or ``""``.

    Shared by :class:`EpubLibraryDialog._show_context_menu` (the
    "Reveal Translated File" action) and :class:`BookDetailsDialog`
    (the 📕 icon button) so the two entry points stay in lockstep.

    Resolution order:

      1. Library entries (``in_library=True``) whose ``path`` ends in
         ``.epub`` — the EPUB sitting inside ``Library/Translated``
         IS the translated artefact.
      2. ``compiled_output_path`` set by the scanner when a workspace
         row was either promoted-to-compiled or state-upgraded via the
         origins/title match pass (``_DualScannerThread``).
      3. ``output_epub_path`` legacy field — same semantics.
      4. ``book['path']`` when it itself is an ``.epub`` on disk
         (covers completed workspace rows whose ``path`` already
         points at the compiled artefact).

    Returns ``""`` when no resolvable translated file exists on disk
    so the caller can omit / disable the action rather than pointing
    at a missing target.
    """
    if not isinstance(book, dict):
        return ""
    lib_path = book.get("path", "") or ""
    if (book.get("in_library")
            and isinstance(lib_path, str)
            and lib_path.lower().endswith(".epub")
            and os.path.isfile(lib_path)):
        return lib_path
    for key in ("compiled_output_path", "output_epub_path"):
        cand = book.get(key) or ""
        if cand and os.path.isfile(cand):
            return cand
    if (isinstance(lib_path, str)
            and lib_path.lower().endswith(".epub")
            and os.path.isfile(lib_path)):
        return lib_path
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
    """Scan ``Library/Translated`` (and registered-in-place EPUBs) for completed books.

    Two sources are merged:

      1. **Physically in ``Library/Translated``** — the curated shelf.
         Every ``.epub`` found via a recursive walk is surfaced with
         ``in_library=True``.
      2. **Registered in place** — paths written to
         ``Library/library_translated_inputs.txt`` by the
         drag-drop / Import pipeline. The file lives wherever the user
         dropped it (Downloads, a cloud-synced folder, etc.); the
         scanner surfaces it with ``in_library=False`` so Organize
         can pick it up later, while the Completed tab still shows
         the card immediately after registration. Missing entries are
         silently dropped.

    Before scanning we run the legacy migration that moves any EPUBs
    still sitting in the legacy Library root into ``Translated/``.
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
                                "raw_source_path": raw_counterpart or "",
                                # Library-filed cards need the same
                                # ``missing_raw_file`` flag as
                                # workspace cards so the \u26a0
                                # \"missing raw\" badge renders when
                                # the compiled EPUB's original raw
                                # can't be resolved via origins /
                                # Library/Raw / the raw-inputs
                                # registry. Without this, a library
                                # card that inherited \"in_progress\"
                                # state via :class:`_DualScannerThread`
                                # silently dropped the badge even
                                # though the raw was genuinely gone
                                # \u2014 and Reveal source file hid
                                # itself (since the raw file was
                                # unresolvable) without a
                                # corresponding warning.
                                "missing_raw_file": not bool(
                                    raw_counterpart),
                            })
                        elif entry.is_dir(follow_symlinks=False) and not entry.name.startswith("."):
                            _walk(entry.path, max_depth, depth + 1)
                    except (PermissionError, OSError):
                        pass
        except (PermissionError, OSError):
            pass

    if os.path.isdir(library_dir):
        _walk(library_dir)

    # Source 2: registered-in-place translated EPUBs. These were
    # dropped / imported onto the Completed tab but deliberately left
    # where they are on disk so the user can reverse the import
    # without fishing the file back out of Library/Translated. They
    # carry ``in_library=False`` + ``registered_translated=True`` so
    # :meth:`_organize_into_library` picks them up as candidates to
    # move into the curated shelf.
    for p in load_library_translated_inputs():
        if not p or not os.path.isfile(p):
            continue
        if not p.lower().endswith(".epub"):
            continue
        norm = os.path.normpath(os.path.abspath(p))
        if norm in seen:
            continue
        seen.add(norm)
        try:
            stat = os.stat(p)
        except OSError:
            continue
        results.append({
            "name": os.path.splitext(os.path.basename(p))[0],
            "path": os.path.abspath(p),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "in_library": False,
            "registered_translated": True,
            "type": "epub",
            "raw_source_path": "",
            # Registered-in-place translated imports have no raw
            # link by construction \u2014 they're drag-dropped
            # compiled EPUBs with no associated workspace. The
            # badge would be misleading here because the
            # concept of a \"raw source\" doesn't apply to these
            # cards at all; default the flag to False so the
            # missing-raw warning stays scoped to workspace /
            # organized library entries.
            "missing_raw_file": False,
        })

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
                # Track the distinction between "parse worked, book has
                # zero chapters" and "parse failed entirely" — the latter
                # indicates an outdated / incompatible progress file
                # written by an older build of the program, and the UI
                # needs to surface that with a dedicated warning pill.
                progress_unparseable = has_progress and summary is None
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
                # Keep any folder that has at least one actionable
                # signal: a compiled output, a progress file (even if
                # unparseable), a resolvable raw source, or translated
                # response files on disk. The old strict gate hid
                # legacy workspaces whose progress file predates the
                # current schema — those now surface with an
                # "outdated_progress" state + warning badge so the user
                # can see and deal with them.
                if (not compiled and not has_progress
                        and not raw_source_path
                        and fs_done == 0 and not raw_in_library):
                    continue

                # Missing raw file flag — set when the folder has
                # meaningful content (compiled / progress / responses)
                # but the original raw EPUB can't be resolved. The UI
                # surfaces this as a ``⚠ missing raw file`` badge; the
                # card stays on whatever tab its other signals land it
                # on (Completed when a compiled artefact exists, In
                # Progress otherwise) instead of being hidden.
                missing_raw_file = bool(not raw_source_path and (
                    compiled or has_progress or fs_done > 0))

                # Compute translation_state for the UI. Drives both the
                # pill label on :class:`_BookCard` AND which tab the
                # card lands on.
                #
                # PROGRESS is the priority signal — the compile gate
                # only fires AFTER progress says the workspace is at
                # 100 %%. Partial or zero-progress workspaces are never
                # promoted by the presence of a stray compiled file:
                #
                #   * unparseable progress      → "outdated_progress"
                #     (pinned to In Progress tab regardless of compile)
                #   * partial progress          → "in_progress"
                #   * no progress + no response → "not_started"
                #   * 100 %% + compiled artefact → "completed"
                #   * 100 %% + no compiled       → "ready_to_compile"
                #
                # The "ready_to_compile" state lives on the In Progress
                # tab with a distinct pill so users know the next step
                # is Build, not Translate. "outdated_progress" also
                # lives on the In Progress tab — the user explicitly
                # asked that we NEVER promote a card whose percentage
                # can't be parsed to the Completed tab, even if a
                # stale compiled EPUB happens to sit next to the
                # broken progress file.
                fully_translated = done >= total and total > 0
                if progress_unparseable:
                    translation_state = "outdated_progress"
                elif fully_translated:
                    if compiled:
                        translation_state = "completed"
                    else:
                        translation_state = "ready_to_compile"
                elif progress_total <= 0 and fs_done == 0:
                    # Partial / zero progress: compile state is
                    # deliberately IGNORED here. A stray compiled
                    # EPUB in an otherwise-empty workspace no longer
                    # silently promotes the card to Completed.
                    translation_state = "not_started"
                else:
                    translation_state = "in_progress"

                # Not-started + missing raw = nothing to render and
                # nothing the user can do with the card (can't open
                # the raw for reading, can't load it for translation,
                # no translated output to show). Hide it entirely
                # rather than surfacing a dead entry on the In
                # Progress tab.
                if (translation_state == "not_started"
                        and missing_raw_file):
                    continue

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
                    # Outdated-progress cards are pinned to the In
                    # Progress tab regardless of compile state — the
                    # user asked that we never promote a card whose
                    # percentage we can't parse to the Completed tab.
                    is_in_progress = (
                        translation_state != "completed"
                        or translation_state == "outdated_progress"
                    )
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
                    # Warning badges the card surfaces next to the
                    # progress pill. ``missing_raw_file`` flags a
                    # workspace whose original source EPUB is no
                    # longer resolvable on disk (the user moved /
                    # deleted / renamed it). ``translation_state ==
                    # "outdated_progress"`` implicitly surfaces its
                    # own badge via the pill.
                    "missing_raw_file": missing_raw_file,
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
        if state == "outdated_progress":
            # Pinned to In Progress regardless of compile state so
            # the "Outdated Progress file" warning is never hidden
            # under a Completed-tab row.
            in_progress.append(r)
        elif state == "completed":
            completed.append(r)
        elif state in ("in_progress", "not_started", "ready_to_compile"):
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

# File-format filter chips on the shared toolbar. ``FORMAT_ALL`` is the
# default (no filter); every other value maps to either the scanned
# row's ``type`` field (for library / compiled cards) or its
# ``workspace_kind`` field (for in-progress folder cards). See
# :meth:`EpubLibraryDialog._format_of_book` for the per-row mapping.
FORMAT_ALL = "all"
FORMAT_EPUB = "epub"
FORMAT_TXT = "txt"
FORMAT_PDF = "pdf"
FORMAT_HTML = "html"
FORMAT_IMAGE = "image"

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
# ``cover_h`` is ~15.5% taller than the raw aspect-match height of
# ``card_w`` (a 10% bump followed by another 5%) so the thumbnail
# area claims more of the card's footprint without the width
# changing — the image renders with more vertical room to fill
# (still via ``KeepAspectRatio``, so wider covers letterbox a
# little less) while leaving the text rows below it untouched.
_SIZE_PRESETS = {
    SIZE_COMPACT: {"card_w": 110, "cover_h": 162, "title_size": "8.5pt",  "title_min_size": "6.5pt", "title_max_h": 50,  "spacing": 3},
    SIZE_NORMAL:  {"card_w": 140, "cover_h": 203, "title_size": "9pt",    "title_min_size": "7pt",   "title_max_h": 55,  "spacing": 4},
    SIZE_LARGE:   {"card_w": 180, "cover_h": 260, "title_size": "9.5pt",  "title_min_size": "7.5pt", "title_max_h": 61,  "spacing": 5},
    SIZE_XL:      {"card_w": 230, "cover_h": 335, "title_size": "10pt",   "title_min_size": "8pt",   "title_max_h": 67,  "spacing": 6},
    SIZE_2XL:     {"card_w": 290, "cover_h": 422, "title_size": "10.5pt", "title_min_size": "8pt",   "title_max_h": 76,  "spacing": 8},
    SIZE_3XL:     {"card_w": 360, "cover_h": 520, "title_size": "11pt",   "title_min_size": "8.5pt", "title_max_h": 84,  "spacing": 10},
    SIZE_4XL:     {"card_w": 440, "cover_h": 635, "title_size": "11.5pt", "title_min_size": "9pt",   "title_max_h": 92,  "spacing": 12},
    SIZE_5XL:     {"card_w": 530, "cover_h": 762, "title_size": "12pt",   "title_min_size": "9.5pt", "title_max_h": 101, "spacing": 14},
    SIZE_6XL:     {"card_w": 630, "cover_h": 912, "title_size": "12.5pt", "title_min_size": "10pt",  "title_max_h": 109, "spacing": 16},
}


def _parse_pt(pt_str) -> float:
    """Parse a CSS-like point-size string (e.g. ``"8.5pt"``) to a float."""
    try:
        return float(str(pt_str).replace("pt", "").strip())
    except (ValueError, TypeError):
        return 9.0


def _fit_pill_font_pt(text: str, max_width: int,
                      base_pt: float = 7.0, min_pt: float = 5.0,
                      step: float = 0.5,
                      horiz_padding: int = 14,
                      bold: bool = True,
                      base_font: QFont | None = None) -> float:
    """Pick a font point size that renders *text* inside *max_width* pixels.

    Pills on the flash cards (“Ready to compile…”, “⚠ missing raw”,
    “⏳ N/M”, …) live inside a fixed-width card and can easily overflow
    horizontally at the preset’s base font size once the card shrinks
    to Compact / Normal or the text happens to be unusually long
    (“Ready to compile (1589/1589)” is ~140 px wide at 7 pt bold).
    Shrinking the font in 0.5 pt increments until it fits mirrors
    :func:`_fit_title_text`’s shrink loop and keeps the short-text
    case rendered at full size.

    *horiz_padding* covers the pill’s CSS padding (5 px left + 5 px
    right), border (1 px × 2), and a small safety buffer so a 1-2 px
    measurement error from ``QFontMetrics.horizontalAdvance`` doesn’t
    let a marginal case clip on render. Falls back to *min_pt* when
    even that doesn’t fit (callers can still truncate if needed).
    """
    if not text:
        return float(base_pt)
    max_width = max(1, int(max_width))
    pt = float(base_pt)
    min_pt = float(min_pt)
    step = float(step)
    while pt > min_pt:
        f = QFont(base_font) if base_font is not None else QFont()
        f.setPointSizeF(pt)
        f.setBold(bold)
        fm = QFontMetrics(f)
        if fm.horizontalAdvance(text) + horiz_padding <= max_width:
            return pt
        pt = max(min_pt, pt - step)
    return min_pt


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

    # Measure via ``QTextLayout`` with ``WrapAtWordBoundaryOrAnywhere``.
    # Plain ``QFontMetrics.boundingRect(TextWordWrap)`` under-measures
    # long Hangul / filename-style runs (it doesn't break between CJK
    # ideographs the way QLabel's renderer actually does), so a 5-line
    # title was being reported as "fits in 3 lines at base_pt" and the
    # shrink loop exited immediately. ``QTextLayout`` is the same
    # engine ``QLabel`` uses to paint word-wrapped text — when we also
    # switch the flash-card and Book-Details renderers to
    # ``_FittedTitleLabel`` (which paints with identical layout
    # options), measurement and render are byte-identical.
    _text_option = QTextOption()
    _text_option.setWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
    _text_option.setAlignment(Qt.AlignLeft | Qt.AlignTop)

    def _height(s: str, pt_size: float) -> int:
        if not s:
            return 0
        f = QFont(base_font) if base_font is not None else QFont()
        f.setPointSizeF(pt_size)
        f.setBold(True)
        layout = QTextLayout(s, f)
        layout.setTextOption(_text_option)
        layout.beginLayout()
        y = 0.0
        while True:
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(float(avail_width))
            line.setPosition(QPointF(0.0, y))
            y += line.height()
        layout.endLayout()
        # Round up so a fractional overflow of a pixel still trips the
        # shrink loop — better to shrink an extra half-step than to let
        # the descender of the last line get clipped.
        return int(y + 0.999)

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


class _FittedTitleLabel(QWidget):
    """Self-painting label that wraps text at *any* point if needed.

    ``QLabel.setWordWrap(True)`` only breaks at Unicode word boundaries,
    which isn't enough for long Hangul / CJK filename-style titles —
    the renderer keeps a single long ideograph run on one line and
    overflows horizontally. This widget paints the text using the same
    ``QTextLayout`` + ``QTextOption.WrapAtWordBoundaryOrAnywhere`` combo
    that :func:`_fit_title_text` uses for measurement, so the measured
    height and the rendered height are byte-identical — the shrink
    loop's chosen font size always lands cleanly inside the title box.

    Public API mirrors the subset of QLabel that the card code needs:
    :meth:`setText`, :meth:`setFont`, :meth:`setToolTip`,
    :meth:`setFixedHeight`, :meth:`setAlignment`.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._text = ""
        self._color = "#e0e0e0"
        self._option = QTextOption()
        self._option.setWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self._option.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setAttribute(Qt.WA_TranslucentBackground)

    # -- public, QLabel-compatible API -------------------------------------
    def setText(self, text: str) -> None:
        self._text = text or ""
        self.update()

    def text(self) -> str:
        return self._text

    def setAlignment(self, alignment) -> None:
        self._option.setAlignment(alignment)
        self.update()

    def setTextColor(self, color: str) -> None:
        """Set the foreground color as a CSS-style string."""
        self._color = color or "#e0e0e0"
        self.update()

    # -- QWidget overrides -------------------------------------------------
    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QColor
        if not self._text:
            return
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.TextAntialiasing, True)
            painter.setFont(self.font())
            painter.setPen(QColor(self._color))
            layout = QTextLayout(self._text, self.font())
            layout.setTextOption(self._option)
            width = float(max(1, self.width()))
            layout.beginLayout()
            y = 0.0
            max_h = float(self.height())
            while True:
                line = layout.createLine()
                if not line.isValid():
                    break
                line.setLineWidth(width)
                line.setPosition(QPointF(0.0, y))
                if y + line.height() > max_h:
                    # No room for another line — bail out so we don't
                    # paint a half-clipped descender.
                    break
                y += line.height()
            layout.endLayout()
            layout.draw(painter, QPointF(0.0, 0.0))
        finally:
            painter.end()

    def sizeHint(self) -> QSize:
        # Enough rows for a 2-line tooltip at the current font; the card
        # layout owns the real height via setFixedHeight anyway.
        fm = QFontMetrics(self.font())
        return QSize(100, fm.height() * 2)


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


def _attach_cross_location_duplicates(completed: list[dict],
                                      output_rows: list[dict]) -> None:
    """Flag completed cards whose EPUB basename exists in two places.

    The ⚠ ``compiled_conflicts`` badge is normally populated by
    :func:`scan_output_folders` for *intra-folder* duplicates (two
    compiled artefacts in the same output workspace). It never fired
    when the duplication was *cross-location* — e.g. one EPUB in
    ``Library/Translated/Foo.epub`` and a second in
    ``<output_root>/Foo/Foo.epub`` — because those two paths come from
    different scans. This helper walks the merged completed list and
    the raw ``output_rows`` (pre-ghost-filter) and appends a conflict
    entry on every card that shares a basename with a compiled output
    at a different path.

    Modifies *completed* in place; does not return anything.
    """
    if not completed:
        return

    # Map lowercased basename → list of compiled abs paths that exist
    # on disk in any of the scanned output folders. ``output_rows``
    # carries ``compiled_output_path`` for every folder the output
    # scanner saw, including the ones that got ghost-filtered from
    # the merged completed list.
    by_basename: dict[str, list[str]] = {}
    for r in output_rows or []:
        cpath = r.get("compiled_output_path") or ""
        if not cpath or not os.path.isfile(cpath):
            continue
        key = os.path.basename(cpath).lower()
        by_basename.setdefault(key, []).append(cpath)
    # Also index the compiled basenames that actually survived into
    # the merged completed list so two non-ghost output-folder cards
    # with the same filename still flag each other.
    for r in completed:
        p = r.get("path", "") or ""
        if not p or not p.lower().endswith(".epub"):
            continue
        key = os.path.basename(p).lower()
        by_basename.setdefault(key, [])
        if p not in by_basename[key]:
            by_basename[key].append(p)
    if not by_basename:
        return

    for r in completed:
        p = r.get("path", "") or ""
        if not p or not p.lower().endswith(".epub"):
            continue
        key = os.path.basename(p).lower()
        siblings = by_basename.get(key, [])
        if len(siblings) < 2:
            continue
        self_abs = os.path.normcase(os.path.normpath(
            os.path.abspath(p)))
        existing = list(r.get("compiled_conflicts") or [])
        # Seed with the basenames already recorded so we don't double
        # up when intra-folder conflicts ALSO exist on the same card.
        seen_labels = {lbl for lbl, _kind in existing}
        for other in siblings:
            other_abs = os.path.normcase(os.path.normpath(
                os.path.abspath(other)))
            if other_abs == self_abs:
                continue
            # Tag the extra copy with its *parent directory* name so
            # the tooltip makes it obvious where the duplicate lives
            # (``Foo.epub (Library/Translated)`` vs. ``Foo.epub (Foo)``).
            parent = os.path.basename(os.path.dirname(other)) or "…"
            label = f"{os.path.basename(other)} ({parent})"
            if label in seen_labels:
                continue
            seen_labels.add(label)
            existing.append((label, "epub"))
        if existing:
            r["compiled_conflicts"] = existing


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

        # Title-based workspace index as a fallback when the origins
        # registry doesn't link a Library/Translated EPUB to its
        # originating workspace. Pulls candidate keys from the
        # workspace folder name, the raw source stem, AND the
        # workspace's ``metadata.json`` title fields — so a workspace
        # whose folder / raw source is named in the *source* language
        # (e.g. "… RoFan Who Is …") still matches its compiled EPUB
        # named in the *translated* language (e.g. "… Romance Fantasy")
        # through the ``metadata_json['title']`` the translator wrote
        # at compile time. Without this fallback the duplicate card
        # appears on BOTH the In Progress tab (workspace row) and
        # the Completed tab (library row) because neither the ghost
        # filter nor the state-inheritance block below can link them.
        #
        # All keys pass through :func:`_norm_book_key` so Windows
        # filename mangling (stripped trailing ``.``, case drift,
        # whitespace collapse) can't desync the two sides of the
        # comparison.
        workspace_by_key: dict[str, dict] = {}
        for ws in output_rows:
            ws_folder = ws.get("output_folder") or ""
            if not ws_folder:
                continue
            candidates: set[str] = set()
            fn = ws.get("folder_name") or os.path.basename(ws_folder)
            if fn:
                candidates.add(_norm_book_key(
                    os.path.splitext(fn)[0]))
            raw = ws.get("raw_source_path") or ""
            if raw:
                candidates.add(_norm_book_key(
                    os.path.splitext(os.path.basename(raw))[0]))
            md = ws.get("metadata_json") or {}
            if isinstance(md, dict):
                for md_key in ("title", "original_title",
                               "translated_title", "raw_title",
                               "source_title", "english_title"):
                    val = md.get(md_key)
                    if isinstance(val, str) and val.strip():
                        candidates.add(_norm_book_key(val))
            for key in candidates:
                if key:
                    workspace_by_key.setdefault(key, ws)

        # Library-side index keyed by the same normalization so the
        # ghost filter + inheritance loop can consult it without
        # recomputing per-row. Each library EPUB contributes MULTIPLE
        # normalized keys so the pairing can land when ANY of them
        # intersects a workspace key:
        #   • filename stem (translated title, post NTFS sanitize)
        #   • ``dc:title`` / ``dc:alternative`` from the EPUB OPF
        #   • ``calibre:original_title`` meta element — the single
        #     most useful signal for raws whose folder is named in
        #     the source language, because the translator writes
        #     the raw title here at compile time.
        library_by_key: dict[str, dict] = {}
        for lr in library_rows:
            if not lr.get("in_library"):
                continue
            lib_path = lr.get("path", "") or ""
            if not lib_path:
                continue
            keys: set[str] = set()
            stem = os.path.splitext(os.path.basename(lib_path))[0]
            k = _norm_book_key(stem)
            if k:
                keys.add(k)
            for opf_title in _extract_epub_titles(lib_path):
                k = _norm_book_key(opf_title)
                if k:
                    keys.add(k)
            for k in keys:
                library_by_key.setdefault(k, lr)

        # Extend the ghost set with workspaces whose title-key
        # matches a Library/Translated entry. The library row will
        # represent the book (with inherited state via the block
        # below) so the workspace-side row must drop off the In
        # Progress tab to avoid the duplicate card.
        #
        # Also persist an authoritative link into ``library_origins.txt``
        # for every pair we discover: ``translated[lib_basename] =
        # <workspace>/<lib_basename>`` (restore target for Undo Move)
        # and ``pairs[lib_basename] = <raw_basename>`` when the raw
        # sits inside ``Library/Raw`` (so
        # :func:`_find_raw_source_for_library_epub` takes the fast
        # origins path on the next call). The user explicitly asked
        # for origins to be updated as we discover pairings — without
        # this, ``_undo_organize_prompt`` has nothing to undo for
        # cards that landed in Library/Translated via any route
        # other than the Organize button.
        origins_dirty = False
        if workspace_by_key and library_by_key:
            trans_map_mut = dict(trans_map) if isinstance(trans_map, dict) else {}
            pair_map_mut = dict(origins.get("pairs", {}) or {}) if isinstance(origins, dict) else {}
            raw_dir_abs = os.path.normcase(os.path.normpath(
                os.path.abspath(get_library_raw_dir())))
            for key, ws in workspace_by_key.items():
                lr = library_by_key.get(key)
                if not lr:
                    continue
                ws_folder = ws.get("output_folder") or ""
                if ws_folder:
                    organized_folders.add(
                        os.path.normcase(os.path.normpath(
                            os.path.abspath(ws_folder)))
                    )
                lib_basename = os.path.basename(lr.get("path", "") or "")
                if not lib_basename or not ws_folder:
                    continue
                existing_entry = trans_map_mut.get(lib_basename) or ""
                desired_entry = os.path.join(ws_folder, lib_basename)
                try:
                    same = bool(existing_entry) and (
                        os.path.normcase(os.path.normpath(
                            os.path.abspath(existing_entry)))
                        == os.path.normcase(os.path.normpath(
                            os.path.abspath(desired_entry)))
                    )
                except Exception:
                    same = False
                if not same:
                    trans_map_mut[lib_basename] = desired_entry
                    origins_dirty = True
                # A ``ready_to_compile`` workspace whose compiled EPUB
                # now lives in ``Library/Translated`` is genuinely
                # completed — the compile step already ran, the
                # artefact just got organized out. Upgrade the
                # in-memory state so:
                #   * the library card stays on the Completed tab
                #     (inheritance block below only rewrites the
                #     library state when the workspace is
                #     non-completed, so a "completed" ws state
                #     keeps the library row in place)
                #   * the workspace row doesn't show a stale
                #     "Ready to compile" pill on the In Progress tab
                #     (it'll be ghost-filtered out anyway, but the
                #     state has to be right for the brief window
                #     where both tabs still contain it).
                #   * ``has_compiled_output`` / ``compiled_output_path``
                #     now point at the library-filed EPUB so callers
                #     that consult those fields resolve to the real
                #     compiled artefact.
                if ws.get("translation_state") == "ready_to_compile":
                    lib_path = lr.get("path", "") or ""
                    ws["translation_state"] = "completed"
                    ws["is_in_progress"] = False
                    if lib_path:
                        ws["has_compiled_output"] = True
                        ws["compiled_output_path"] = lib_path
                        ws["compiled_output_kind"] = "epub"
                        ws["has_output_epub"] = True
                        ws["output_epub_path"] = lib_path
                raw_src = ws.get("raw_source_path") or ""
                if raw_src and os.path.isfile(raw_src):
                    try:
                        raw_parent = os.path.normcase(os.path.normpath(
                            os.path.abspath(os.path.dirname(raw_src))))
                    except Exception:
                        raw_parent = ""
                    if raw_parent == raw_dir_abs:
                        raw_basename = os.path.basename(raw_src)
                        if (raw_basename
                                and pair_map_mut.get(lib_basename) != raw_basename):
                            pair_map_mut[lib_basename] = raw_basename
                            origins_dirty = True
            if origins_dirty:
                try:
                    origins["translated"] = trans_map_mut
                    origins["pairs"] = pair_map_mut
                    _save_origins(origins)
                    trans_map = trans_map_mut
                except Exception:
                    logger.debug(
                        "origins auto-update failed: %s",
                        traceback.format_exc())

        # Unified state upgrade: every ``ready_to_compile`` workspace
        # whose folder is in ``organized_folders`` is actually
        # COMPLETED — its compiled EPUB just lives in
        # ``Library/Translated`` instead of alongside the progress
        # file. Runs regardless of whether the link came from
        # origins.txt, the title-key fallback, or a mix, so the
        # "ready to compile" check now consults the library shelf
        # uniformly via the combined ghost set.
        #
        # Without this upgrade, the inheritance block below would
        # overwrite the library card's state with ``ready_to_compile``
        # and yank it onto the In Progress tab — exactly the
        # duplicate-card / wrong-state behaviour the user hit when
        # organizing an EPUB in then out of Library/Translated.
        if organized_folders:
            trans_dir_abs = get_library_translated_dir()
            # Build a reverse lookup from workspace folder → library
            # path so we can populate ``compiled_output_path`` on the
            # upgraded workspace without another disk scan.
            folder_to_lib_path: dict[str, str] = {}
            for lib_basename, orig_path in (trans_map or {}).items():
                if not orig_path:
                    continue
                try:
                    parent = os.path.normcase(os.path.normpath(
                        os.path.abspath(os.path.dirname(str(orig_path)))))
                except Exception:
                    continue
                candidate = os.path.join(trans_dir_abs, lib_basename)
                if os.path.isfile(candidate):
                    folder_to_lib_path.setdefault(parent, candidate)
            for ws in output_rows:
                if ws.get("translation_state") != "ready_to_compile":
                    continue
                ws_folder = ws.get("output_folder") or ""
                if not ws_folder:
                    continue
                fk = os.path.normcase(os.path.normpath(
                    os.path.abspath(ws_folder)))
                if fk not in organized_folders:
                    continue
                ws["translation_state"] = "completed"
                ws["is_in_progress"] = False
                lib_path = folder_to_lib_path.get(fk, "")
                if lib_path:
                    ws["has_compiled_output"] = True
                    ws["compiled_output_path"] = lib_path
                    ws["compiled_output_kind"] = "epub"
                    ws["has_output_epub"] = True
                    ws["output_epub_path"] = lib_path

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

        # Library-vs-workspace state inheritance via ``origins.txt``
        # (primary) + filename fallback (secondary). NO dedupe —
        # every card stays visible exactly once. Each
        # ``Library/Translated`` entry inherits the translation_state
        # + progress numbers of its owning workspace when they can
        # be linked, regardless of whether that workspace was
        # ghost-filtered above. That fixes:
        #   * the "99%% but shown as Completed" regression after
        #     organize: the ghost filter takes the workspace *row*
        #     off the In Progress list, and this block re-routes the
        #     library row (which replaces it) onto the In Progress
        #     tab while the underlying translation is still at 99 %%.
        #   * the "ready_to_compile shows up on BOTH tabs" bug: when
        #     origins doesn't link the pair, the filename fallback
        #     below still matches them so the library row inherits
        #     ``ready_to_compile`` and moves to In Progress (with
        #     the workspace row ghost-filtered above).
        #
        # ``trans_map`` maps ``library_basename → original_workspace_path``;
        # the parent dir of that path is the workspace folder. Rows
        # without a ``trans_map`` record fall through to the
        # title-based ``workspace_by_key`` lookup built above.
        workspace_by_folder: dict[str, dict] = {}
        for ws in output_rows:
            ws_folder = ws.get("output_folder") or ""
            if not ws_folder:
                continue
            workspace_by_folder[
                os.path.normcase(os.path.normpath(
                    os.path.abspath(ws_folder)))
            ] = ws

        if workspace_by_folder and (trans_map or workspace_by_key):
            for r in completed:
                if not r.get("in_library"):
                    continue
                lib_basename = os.path.basename(r.get("path", "") or "")
                ws_row = None
                # 1. Origins-based link (authoritative).
                if trans_map:
                    orig_path = trans_map.get(lib_basename)
                    if orig_path:
                        orig_folder = os.path.dirname(str(orig_path))
                        if orig_folder:
                            origin_key = os.path.normcase(os.path.normpath(
                                os.path.abspath(orig_folder)))
                            ws_row = workspace_by_folder.get(origin_key)
                # 2. Title-key fallback — mirrors the ghost-set
                #    extension above so both sides of the dedup
                #    (workspace row off In Progress, library row
                #    moved from Completed to In Progress) kick in
                #    for the same pair of cards. The key covers
                #    folder name, raw source stem, and the
                #    metadata.json title fields so raws named in the
                #    source language still match their English-
                #    titled compiled EPUBs.
                if not ws_row:
                    lib_key = _norm_book_key(
                        os.path.splitext(lib_basename)[0])
                    if lib_key:
                        ws_row = workspace_by_key.get(lib_key)
                if not ws_row:
                    continue
                ws_state = ws_row.get("translation_state") or ""
                # Only inherit when the workspace is actually NOT
                # completed — otherwise a finished book would get
                # yanked onto the In Progress tab with stale
                # progress numbers. A completed workspace + library
                # file is the normal post-organize state; leave the
                # library card alone there.
                if ws_state == "completed" or not ws_state:
                    continue
                r["translation_state"] = ws_state
                r["is_in_progress"] = True
                r["total_chapters"] = ws_row.get("total_chapters", 0)
                r["completed_chapters"] = ws_row.get(
                    "completed_chapters", 0)
                r["failed_chapters"] = ws_row.get("failed_chapters", 0)
                r["pending_chapters"] = ws_row.get(
                    "pending_chapters", 0)
                r["output_folder"] = ws_row.get("output_folder", "")
                r["progress_file"] = ws_row.get("progress_file", "")
                # Track the raw source so the In Progress card can
                # render a cover / resolve the raw-open actions
                # even though the row itself lives in Library/Translated.
                raw_src_from_ws = ws_row.get("raw_source_path") or ""
                if raw_src_from_ws:
                    r["raw_source_path"] = raw_src_from_ws
                # Keep ``missing_raw_file`` honest for inherited
                # library rows. If after inheritance the card
                # still has no resolvable raw source (neither its
                # own library-side lookup nor the workspace
                # produced one), flip the flag on so the
                # \u26a0 \"missing raw\" badge surfaces. Otherwise
                # (raw path present on either side) keep the flag
                # off so the badge doesn't render spuriously.
                r["missing_raw_file"] = not bool(
                    r.get("raw_source_path") or "")

            # Move any library rows that inherited a non-completed
            # state over to the In Progress tab, and make sure we
            # don't end up with a duplicate workspace row for the
            # SAME output folder in the in_progress bucket (can
            # happen when the ghost filter didn't kick in).
            moved_in_progress_folders: set[str] = set()
            still_completed: list[dict] = []
            for r in completed:
                state = r.get("translation_state")
                if state and state != "completed" and r.get("in_library"):
                    in_progress.append(r)
                    of = r.get("output_folder") or ""
                    if of:
                        moved_in_progress_folders.add(
                            os.path.normcase(os.path.normpath(
                                os.path.abspath(of)))
                        )
                else:
                    still_completed.append(r)
            completed = still_completed

            if moved_in_progress_folders:
                in_progress = [
                    r for r in in_progress
                    if not r.get("in_library")
                    and os.path.normcase(os.path.normpath(
                        os.path.abspath(r.get("output_folder", ""))))
                    not in moved_in_progress_folders
                    or r.get("in_library")
                ]

        # Cross-location duplicate detection: if a ``Library/Translated``
        # entry has the same basename as a compiled EPUB still sitting in
        # an output folder, surface it on the library card's ⚠ badge.
        # This catches the case where the user organized the EPUB into
        # the library but the original (or a re-compiled copy) still
        # lives in the output folder — without this check, the ghost
        # filter above silently dropped the output row and the user
        # never saw any indication that two physical copies exist.
        _attach_cross_location_duplicates(completed, output_rows)

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
    can clear the current selection.

    Dragging starts from **anywhere** in the grid — including on top of
    a :class:`_BookCard` — because the slivers of empty space between
    cards are too narrow to reliably grab. A mouse press on a card is
    initially handled as a normal card click (the card still emits
    :attr:`_BookCard.select_requested` so single-click selection keeps
    working); as soon as the pointer moves past :data:`_DRAG_THRESHOLD`,
    the grid promotes the gesture to a rubber-band drag starting from
    the original press location. The card's just-fired selection is
    overwritten by the rubber-band result on mouse release (a plain
    drag clears + replaces; Ctrl / Shift extends), so the user never
    sees a stuck "single card selected" state after a drag.
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
        # Filter-based press tracking for drags that start on a card.
        # The card handles the initial press itself (so click-select
        # still works); we watch for subsequent MouseMove on the card
        # — via an event filter installed on every descendant — and
        # promote the gesture to a rubber-band once it passes the
        # drag threshold. Coords are always translated into grid-
        # local space via :func:`QWidget.mapTo` so the band geometry
        # matches what a press on empty space would have produced.
        self._card_drag_origin = None
        self._card_drag_modifiers = None
        # Install the filter on any children that happen to exist at
        # construction time (usually none — the grid is populated
        # later). :meth:`childEvent` handles everything added after.
        self._install_filter_on_descendants()

    # -- event-filter plumbing ---------------------------------------------
    def childEvent(self, event):
        """Install the rubber-band filter on newly added descendants.

        ``ChildAdded`` fires once per direct child, so we handle the
        direct child here and recurse via :meth:`_install_filter_on_descendants`
        to catch the card's own sub-widgets (labels, layout spacers) —
        otherwise a press on a card's title label would slip past the
        filter entirely and the drag-through-card gesture would feel
        unreliable depending on exactly where the user clicked.
        """
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.ChildAdded:
            child = event.child()
            try:
                if isinstance(child, QWidget):
                    child.installEventFilter(self)
                    # Recurse so labels / spacers inside cards are
                    # covered too — a press on ``title_lbl`` otherwise
                    # never reaches the filter because QLabel consumes
                    # mouse events by default.
                    for sub in child.findChildren(QWidget):
                        try:
                            sub.installEventFilter(self)
                        except Exception:
                            pass
            except Exception:
                pass
        super().childEvent(event)

    def _install_filter_on_descendants(self):
        for c in self.findChildren(QWidget):
            try:
                c.installEventFilter(self)
            except Exception:
                pass

    def _card_ancestor(self, obj):
        """Return the :class:`_BookCard` ancestor of *obj*, or None."""
        if not isinstance(obj, QWidget):
            return None
        node = obj
        while node is not None and node is not self:
            if isinstance(node, _BookCard):
                return node
            try:
                node = node.parent()
            except Exception:
                return None
        return None

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent, QRect
        card = self._card_ancestor(obj)
        if card is None:
            return super().eventFilter(obj, event)
        etype = event.type()
        if etype == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            # Record the press origin in grid-local coords but let
            # the card keep the event so ``select_requested`` still
            # fires for a plain click. ``mapTo`` handles the nested
            # widget case (press on a label inside the card).
            try:
                self._card_drag_origin = obj.mapTo(self, event.pos())
            except Exception:
                self._card_drag_origin = None
            self._card_drag_modifiers = event.modifiers()
            self._drag_started = False
            return False
        if etype == QEvent.MouseMove and (event.buttons() & Qt.LeftButton):
            if (self._card_drag_origin is not None
                    and not self._drag_started):
                try:
                    current = obj.mapTo(self, event.pos())
                except Exception:
                    return False
                delta = current - self._card_drag_origin
                if abs(delta.x()) + abs(delta.y()) > self._DRAG_THRESHOLD:
                    self._start_rubber_band(
                        self._card_drag_origin,
                        self._card_drag_modifiers or Qt.NoModifier)
                    self._rubber_band.setGeometry(
                        QRect(self._drag_origin, current).normalized())
                    return True
            if self._drag_started and self._rubber_band is not None:
                try:
                    current = obj.mapTo(self, event.pos())
                except Exception:
                    return False
                self._rubber_band.setGeometry(
                    QRect(self._drag_origin, current).normalized())
                return True
        if etype == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            if self._drag_started:
                # Finalize: emit selection + reset state. Consume the
                # release so the card's own handler doesn't also fire
                # and re-emit ``select_requested`` for a now-stale
                # single-card selection.
                self._finish_rubber_band()
                self._card_drag_origin = None
                self._card_drag_modifiers = None
                return True
            self._card_drag_origin = None
            self._card_drag_modifiers = None
            return False
        return super().eventFilter(obj, event)

    # -- shared start / finish helpers -------------------------------------
    def _start_rubber_band(self, origin, modifiers):
        from PySide6.QtCore import QRect, QSize
        from PySide6.QtWidgets import QRubberBand
        self._drag_origin = origin
        self._drag_modifiers = modifiers
        self._drag_started = True
        if self._rubber_band is None:
            self._rubber_band = QRubberBand(QRubberBand.Rectangle, self)
        self._rubber_band.setGeometry(QRect(origin, QSize()))
        self._rubber_band.show()

    def _finish_rubber_band(self):
        if self._rubber_band is None or not self._drag_started:
            return
        rect = self._rubber_band.geometry()
        self._rubber_band.hide()
        books: list = []
        for child in self.children():
            if isinstance(child, _BookCard):
                if rect.intersects(child.geometry()):
                    books.append(child.book)
        self.rubber_band_selection.emit(
            books, self._drag_modifiers or Qt.NoModifier)
        self._drag_origin = None
        self._drag_modifiers = None
        self._drag_started = False

    # -- direct mouse handling (presses landing on empty grid space) ------
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_origin = event.pos()
            self._drag_modifiers = event.modifiers()
            self._drag_started = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_origin is not None and (event.buttons() & Qt.LeftButton):
            from PySide6.QtCore import QRect
            delta = event.pos() - self._drag_origin
            distance = abs(delta.x()) + abs(delta.y())
            if not self._drag_started and distance > self._DRAG_THRESHOLD:
                self._start_rubber_band(
                    self._drag_origin,
                    self._drag_modifiers or Qt.NoModifier)
            if self._drag_started and self._rubber_band is not None:
                self._rubber_band.setGeometry(
                    QRect(self._drag_origin, event.pos()).normalized())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        try:
            if self._drag_started and self._rubber_band is not None:
                self._finish_rubber_band()
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
    """Best-guess raw / source-language label for a library flash card.

    The "Raw titles" toolbar toggle maps to *this* function, and users
    expect it to reveal the original source *filename* on disk — the
    one they'd use to hunt the file down in Explorer — not a
    translated metadata title that happens to sit in ``metadata.json``.
    Resolution order (filename-first):

      1. Stem of ``raw_source_path`` — the resolved raw EPUB / PDF /
         TXT the scanner matched to this card. This is the authoritative
         source filename whenever it's available.
      2. Stem of ``original_path`` recorded in the origins registry
         (Library-organized files that were moved from elsewhere still
         know where they came from).
      3. ``folder_name`` — for in-progress workspaces this equals the
         raw EPUB's basename because output folders are scaffolded from
         the source filename.
      4. ``metadata.json`` ``original_title`` / ``raw_title`` /
         ``source_title`` when the translator stored one explicitly.
         (Kept as a fallback so books without a resolvable raw source
         still surface *something* source-language-y rather than
         reverting to the translated name.)
      5. Fall back to the card's default ``name``.
    """
    for path_key in ("raw_source_path", "original_path"):
        p = book.get(path_key) or ""
        if p:
            return os.path.splitext(os.path.basename(p))[0]
    fn = book.get("folder_name")
    if fn:
        return str(fn)
    md = book.get("metadata_json") or {}
    for key in ("original_title", "raw_title", "source_title"):
        val = md.get(key)
        if val:
            return str(val)
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
        layout.setContentsMargins(4, 4, 4, 4)
        # Tight inter-row spacing so the "size/badge" row and the
        # progress pill sit closer together. QVBoxLayout applies this
        # between every pair of widgets, including cover ↔ title, but
        # 1 px stays visually clean and mirrors the card's balance.
        layout.setSpacing(1)

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
        # Cards WITHOUT a missing-raw warning inherit the 16 px slot
        # we reserve at the bottom (see ``reserved_h += 16`` below)
        # as extra title room — otherwise that space just sits
        # empty, the card looks bottom-heavy, and longer titles
        # ellipsize unnecessarily. Cards WITH the warning keep the
        # slot for the badge and the title stays at its preset
        # height. Either way the total card height is identical so
        # the grid rows line up.
        _has_warning_preview = bool(book.get("missing_raw_file"))
        effective_max_title_h = max_title_h + (0 if _has_warning_preview else 16)
        # Custom-paint widget whose wrap mode matches :func:`_fit_title_text`'s
        # measurement, so the shrink loop's chosen font size always fits
        # the box even for long Hangul / CJK filename-style raw titles.
        title_lbl = _FittedTitleLabel()
        title_lbl.setFixedWidth(self._card_w - 10)
        title_lbl.setFixedHeight(effective_max_title_h)
        title_lbl.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        title_lbl.setToolTip(full_title)
        fitted_text, fitted_pt = _fit_title_text(
            full_title,
            avail_width=self._card_w - 10,
            max_height=effective_max_title_h,
            base_pt=base_pt,
            base_font=title_lbl.font(),
            min_pt=min_pt,
        )
        # Apply the fitted point size via ``setFont`` directly on the
        # custom widget. The widget's paintEvent uses the same QTextLayout
        # + WrapAtWordBoundaryOrAnywhere pipeline :func:`_fit_title_text`
        # measures with, so measured height == rendered height.
        title_font = QFont(title_lbl.font())
        title_font.setPointSizeF(fitted_pt)
        title_font.setBold(True)
        title_lbl.setFont(title_font)
        title_lbl.setText(fitted_text)
        title_lbl.setTextColor("#e0e0e0")
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
            # Dynamic font size: the conflict badge usually reads
            # “⚠ +2” / “⚠ +12” (short), but Compact / Normal cards still
            # put it next to the size + EPUB badges, so shrink before
            # it pushes its row past the card edge.
            _conflict_budget = max(32, int(self._card_w / 3))
            _conflict_pt = _fit_pill_font_pt(
                conflict_lbl.text(), _conflict_budget,
                base_pt=7.0, min_pt=5.5, horiz_padding=12,
            )
            conflict_lbl.setStyleSheet(
                f"color: #ffb347; font-size: {_conflict_pt}pt; font-weight: bold; "
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

        # Missing-raw-file warning: the workspace still has a
        # compiled / progress / response trail but the ORIGINAL raw
        # source EPUB can't be resolved on disk anymore. Surface it
        # as a ⚠ badge on its OWN row — stacking it beside the
        # size / badge on ``info_row`` used to clip “⚠ missing raw”
        # down to “⚠ …” once the card was Compact / Normal, because
        # the size label + EPUB badge already claimed most of the
        # row’s width.
        has_warnings_row = bool(book.get("missing_raw_file"))
        if has_warnings_row:
            warnings_row = QHBoxLayout()
            warnings_row.setContentsMargins(0, 0, 0, 0)
            warnings_row.setSpacing(4)
            missing_lbl = QLabel("\u26a0 missing raw")
            missing_lbl.setAttribute(Qt.WA_TranslucentBackground)
            _missing_budget = max(40, int(self._card_w - 12))
            _missing_pt = _fit_pill_font_pt(
                missing_lbl.text(), _missing_budget,
                base_pt=7.0, min_pt=5.5, horiz_padding=14,
            )
            missing_lbl.setStyleSheet(
                f"color: #ff9e6d; font-size: {_missing_pt}pt; font-weight: bold; "
                "background: rgba(255, 158, 109, 0.15); "
                "border: 1px solid #ff9e6d; border-radius: 3px; "
                "padding: 0 4px;"
            )
            missing_lbl.setToolTip(
                "The raw source file for this book can't be found on "
                "disk \u2014 Library/Raw, source_epub.txt, and the "
                "raw-inputs registry all came up empty. The compiled "
                "output is still readable, but actions that need the "
                "raw source (Reveal source, Read raw, Load for "
                "translation) will be disabled."
            )
            warnings_row.addWidget(missing_lbl)
            warnings_row.addStretch()
            layout.addLayout(warnings_row)

        # In-progress indicator: small status pill + overlay ribbon on the cover
        has_progress_row = False
        if book.get("is_in_progress"):
            total = int(book.get("total_chapters", 0) or 0)
            done = int(book.get("completed_chapters", 0) or 0)
            state = book.get("translation_state") or (
                "in_progress" if total else "not_started"
            )
            # 100% translated + compiled EPUB = completed — no pill;
            # those cards render plain on the Completed tab.
            if state == "completed":
                pass
            else:
                has_progress_row = True
                pct = int(round((done / total) * 100)) if total else 0
                progress_row = QHBoxLayout()
                progress_row.setContentsMargins(0, 0, 0, 0)
                progress_row.setSpacing(4)
                # Budget for the pill so :func:`_fit_pill_font_pt` can
                # shrink a too-long label (e.g. “Ready to compile
                # (1589/1589)”) down to a size that still fits inside
                # the card’s fixed width. We subtract the card’s own
                # left/right padding (8 px), the ~30 px the “NN%”
                # label to the right takes when the “in_progress”
                # branch shows it, and a generous buffer for the
                # row’s 4 px spacing, border, and Qt’s emoji-width
                # under-measurement (✨, ⏳, ⚠, 🆕 all render a
                # few px wider than ``QFontMetrics.horizontalAdvance``
                # predicts on Windows, which is why “Ready to compile
                # (15/15)” clipped at 7 pt even though the measurement
                # said it fit).
                _needs_pct_lbl = bool(total) and state not in (
                    "outdated_progress", "not_started", "ready_to_compile",
                )
                _pct_reservation = 30 if _needs_pct_lbl else 0
                _pill_budget = max(40, int(self._card_w - 16 - _pct_reservation))
                # Horizontal padding used by every call below covers
                # 10 px CSS padding (5 + 5), 2 px border, and 10 px of
                # safety buffer for emoji-width drift / antialiasing
                # so the shrink loop’s “it fits” result actually fits
                # on every system font. Pair with ``min_pt=4.5`` so
                # the worst-case label (“Ready to compile (1589/1589)”
                # on a Compact card) still lands without truncation.
                _pill_horiz_padding = 22
                _pill_base_font = QFont(self.font())
                _pill_base_font.setBold(True)
                if state == "outdated_progress":
                    pill = QLabel("\u26a0 Outdated Progress file")
                    pill.setToolTip(
                        "The ``translation_progress.json`` in this "
                        "workspace was written by an older version "
                        "of Glossarion and can't be parsed \u2014 the "
                        "card is pinned here so you can re-run the "
                        "translation or remove the folder."
                    )
                    _pill_pt = _fit_pill_font_pt(
                        pill.text(), _pill_budget,
                        base_pt=7.0, min_pt=4.5,
                        horiz_padding=_pill_horiz_padding,
                        base_font=_pill_base_font,
                    )
                    pill.setStyleSheet(
                        "color: #ffb347; "
                        "background: rgba(255, 179, 71, 0.18); "
                        "border: 1px solid #ffb347; border-radius: 3px; "
                        f"font-size: {_pill_pt}pt; font-weight: bold; "
                        "padding: 0 5px 2px 5px;"
                    )
                    progress_row.addWidget(pill)
                    ribbon_text = "OUTDATED PROGRESS"
                    ribbon_bg = "rgba(255, 179, 71, 0.92)"
                elif state == "not_started":
                    pill = QLabel("\U0001f195 Not started")
                    pill.setToolTip("Imported into Library/Raw, translation not started yet.")
                    _pill_pt = _fit_pill_font_pt(
                        pill.text(), _pill_budget,
                        base_pt=7.0, min_pt=4.5,
                        horiz_padding=_pill_horiz_padding,
                        base_font=_pill_base_font,
                    )
                    pill.setStyleSheet(
                        "color: #8ab4d0; background: rgba(138, 180, 208, 0.15); "
                        "border: 1px solid #8ab4d0; border-radius: 3px; "
                        f"font-size: {_pill_pt}pt; font-weight: bold; "
                        "padding: 0 5px 2px 5px;"
                    )
                    progress_row.addWidget(pill)
                    ribbon_text = "NOT STARTED"
                    ribbon_bg = "rgba(138, 180, 208, 0.92)"
                elif state == "ready_to_compile":
                    pill = QLabel(
                        f"\u2728 Ready to compile "
                        f"({done}/{total})" if total
                        else "\u2728 Ready to compile"
                    )
                    pill.setToolTip(
                        "All chapters translated \u2014 compile the "
                        "final EPUB to graduate this card to the "
                        "Completed tab."
                    )
                    _pill_pt = _fit_pill_font_pt(
                        pill.text(), _pill_budget,
                        base_pt=7.0, min_pt=4.5,
                        horiz_padding=_pill_horiz_padding,
                        base_font=_pill_base_font,
                    )
                    pill.setStyleSheet(
                        "color: #6ee8a0; "
                        "background: rgba(110, 232, 160, 0.16); "
                        "border: 1px solid #6ee8a0; border-radius: 3px; "
                        f"font-size: {_pill_pt}pt; font-weight: bold; "
                        "padding: 0 5px 2px 5px;"
                    )
                    progress_row.addWidget(pill)
                    ribbon_text = "READY TO COMPILE"
                    ribbon_bg = "rgba(110, 232, 160, 0.92)"
                else:
                    pill = QLabel(f"\u23f3 {done}/{total}" if total else "\u23f3 In progress")
                    pill.setToolTip(
                        f"Translation in progress \u2014 {pct}% ({done}/{total} chapters)"
                    )
                    _pill_pt = _fit_pill_font_pt(
                        pill.text(), _pill_budget,
                        base_pt=7.0, min_pt=4.5,
                        horiz_padding=_pill_horiz_padding,
                        base_font=_pill_base_font,
                    )
                    pill.setStyleSheet(
                        "color: #ffd166; background: rgba(108, 99, 255, 0.18); "
                        "border: 1px solid #6c63ff; border-radius: 3px; "
                        f"font-size: {_pill_pt}pt; font-weight: bold; "
                        "padding: 0 5px 2px 5px;"
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
                # Pin the progress row to the BOTTOM of the card by
                # inserting a vertical stretch above it. Without this
                # stretch the pill floats mid-card (between info_row /
                # warnings_row and the ``layout.addStretch()`` below),
                # so a card without a warning row sits with a dead
                # band below the pill while another card WITH a
                # warning has the pill tucked in the middle. Pinning
                # the pill to the bottom lines every card’s pill up
                # along the same baseline across the grid.
                layout.addStretch()
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
        # Shrunk alongside ``layout.setSpacing(1)`` / bottom-margin 4
        # so the tighter inter-row gaps don't just migrate into a
        # bigger empty band below the progress pill.
        reserved_h = 24 + p.get("spacing", 4)
        if has_progress_row:
            reserved_h += 20  # pill row height + extra inter-widget spacing
        # Always reserve the warnings-row slot so every card lands at
        # the same fixed height, whether or not it currently carries
        # a “⚠ missing raw” badge. Without this constant, cards
        # with and without the badge drift by the badge’s height and
        # the grid looks ragged row-to-row. 16 px covers the badge
        # (~12 px) + ``layout.setSpacing(1)`` + a bit of breathing
        # room below the progress pill so the card doesn’t feel
        # bottom-cramped.
        reserved_h += 16
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
# Scan-for-Raw dialog
# ---------------------------------------------------------------------------

class _RawScanWorker(QThread):
    """Walk + match raw source files off the UI thread.

    Both the ``os.walk`` and the per-workspace matching run here so
    the only main-thread work after a scan is building the tree
    rows. Previously the matching pass (especially Fuzzy mode's
    ``difflib.SequenceMatcher.ratio()`` across every candidate)
    ran back on the UI thread and stalled the dialog for seconds
    on big folders.

    The worker accepts an optional ``prewalked`` candidate list so
    mode / threshold changes don't have to re-walk the folder —
    the dialog caches the last walk result and hands it back for
    fast re-matching.

    A :class:`concurrent.futures.ThreadPoolExecutor` inside the
    worker fans per-directory classification out across 4 threads
    so the normalization pass doesn't bottleneck on the walking
    thread. Cancellable via :meth:`cancel` so a rapid UI toggle
    can stop a stale scan at the next directory / workspace
    boundary.
    """

    # (scan_folder, candidates, matches_dict)
    results = Signal(str, list, dict)

    def __init__(self, scan_folder: str, ext_suffixes: tuple[str, ...],
                 tracking_names: frozenset,
                 books: list[dict], mode: str, threshold: int,
                 prewalked: list | None = None, parent=None):
        super().__init__(parent)
        self._folder = scan_folder
        self._suffixes = ext_suffixes
        self._tracking = tracking_names
        self._books = list(books or [])
        self._mode = mode
        self._threshold = int(threshold)
        self._prewalked = prewalked
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def _classify(self, root_dir: str,
                  files: list[str]) -> list[tuple[str, str]]:
        """Filter + normalize a single directory's files (executor task).

        Pure function on the inputs plus ``self._suffixes`` /
        ``self._tracking`` — no shared mutable state, so it's safe
        to run across multiple pool workers concurrently.
        """
        out: list[tuple[str, str]] = []
        suffixes = self._suffixes
        tracking = self._tracking
        for name in files:
            if self._cancelled:
                break
            lower = name.lower()
            if not lower.endswith(suffixes):
                continue
            if lower in tracking:
                continue
            if name.startswith("."):
                continue
            stem = os.path.splitext(name)[0]
            key = _norm_book_key(stem)
            if not key:
                continue
            out.append((key, os.path.join(root_dir, name)))
        return out

    def _walk(self) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        folder = self._folder
        if not folder or not os.path.isdir(folder):
            return candidates
        try:
            futures: list = []
            with ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix="ScanForRaw",
            ) as executor:
                for root_dir, _dirs, files in os.walk(folder):
                    if self._cancelled:
                        break
                    if not files:
                        continue
                    futures.append(executor.submit(
                        self._classify, root_dir, list(files)))
                for fut in futures:
                    if self._cancelled:
                        break
                    try:
                        candidates.extend(fut.result())
                    except Exception:
                        logger.debug(
                            "ScanForRaw classify task failed: %s",
                            traceback.format_exc())
        except (PermissionError, OSError) as exc:
            logger.debug(
                "ScanForRaw folder walk failed for %s: %s", folder, exc)
        except Exception:
            logger.debug(
                "ScanForRaw worker crashed: %s", traceback.format_exc())
        return candidates

    @staticmethod
    def _book_keys(book: dict) -> list[str]:
        keys: set[str] = set()
        fn = book.get("folder_name") or os.path.basename(
            book.get("output_folder") or book.get("path") or "")
        if fn:
            keys.add(_norm_book_key(os.path.splitext(fn)[0]))
        md = book.get("metadata_json") or {}
        if isinstance(md, dict):
            for md_key in ("title", "original_title",
                           "translated_title", "raw_title",
                           "source_title", "english_title"):
                val = md.get(md_key)
                if isinstance(val, str) and val.strip():
                    keys.add(_norm_book_key(val))
        return [k for k in keys if k]

    # Maps a workspace's ``workspace_kind`` to the raw file
    # extensions that are legitimately pairable with it. EPUB
    # workspaces only ever want ``.epub`` sources; a ``.pdf``
    # candidate with the same filename stem must NOT win just
    # because the normalized title collides. Kinds that aren't
    # in the map (``""``, ``"other"``, ``"in_progress"``,
    # ``"image"``) fall back to "accept any" because we can't
    # predict the right extension without more signal.
    _KIND_ALLOWED_EXTS = {
        "epub": (".epub",),
        "txt":  (".txt",),
        "pdf":  (".pdf",),
        "html": (".html", ".htm"),
    }

    def _compute_matches(self,
                         candidates: list[tuple[str, str]]) -> dict:
        """Return ``{output_folder: {book, path, ratio, accepted}}``.

        All heavy lifting (including Fuzzy's ``SequenceMatcher``
        calls) runs here on the worker thread. Main thread only
        receives the final dict and paints the tree.
        """
        import difflib
        matches: dict[str, dict] = {}
        mode = self._mode
        threshold = self._threshold / 100.0
        # Pool the SequenceMatcher calls across workers too \u2014 for a
        # big candidate set Fuzzy matching is the real hotspot.
        kind_allowed = self._KIND_ALLOWED_EXTS

        def _best_for(book: dict) -> tuple[str, float]:
            book_keys = self._book_keys(book)
            if not book_keys or not candidates:
                return "", 0.0
            # Per-book extension gate: a workspace that advertises
            # its own kind (EPUB / TXT / PDF / HTML) must only be
            # paired with candidates whose extension matches. This
            # closes the hole where a ``.pdf`` raw whose filename
            # stem collides with an EPUB workspace's title would
            # be auto-accepted at ratio 1.0 just because the
            # normalized-key matched.
            book_kind = (book.get("workspace_kind") or "").lower()
            allowed_exts = kind_allowed.get(book_kind)
            if allowed_exts:
                usable = [
                    (ck, cp) for ck, cp in candidates
                    if cp.lower().endswith(allowed_exts)
                ]
            else:
                usable = candidates
            if not usable:
                return "", 0.0
            if mode == _ScanForRawDialog.MATCH_EXACT:
                book_key_set = set(book_keys)
                for cand_key, cand_path in usable:
                    if self._cancelled:
                        break
                    if cand_key in book_key_set:
                        return cand_path, 1.0
                return "", 0.0
            best_path = ""
            best_ratio = 0.0
            sm = difflib.SequenceMatcher()
            for cand_key, cand_path in usable:
                if self._cancelled:
                    break
                sm.set_seq2(cand_key)
                for wk in book_keys:
                    sm.set_seq1(wk)
                    # Cheap length-ratio prefilter \u2014 skip candidates
                    # that can't possibly reach the threshold so we
                    # don't pay for a full ratio() call on obvious
                    # non-matches. real_quick_ratio is O(1).
                    if sm.real_quick_ratio() < threshold:
                        continue
                    ratio = sm.ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_path = cand_path
                        if best_ratio >= 0.999:
                            return best_path, best_ratio
            if best_ratio >= threshold:
                return best_path, best_ratio
            return "", best_ratio

        try:
            with ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix="ScanForRawMatch",
            ) as executor:
                book_futures = []
                for book in self._books:
                    if self._cancelled:
                        break
                    ws_folder = (book.get("output_folder")
                                 or book.get("path") or "")
                    if not ws_folder:
                        continue
                    book_futures.append(
                        (ws_folder, book,
                         executor.submit(_best_for, book)))
                for ws_folder, book, fut in book_futures:
                    if self._cancelled:
                        break
                    try:
                        matched_path, ratio = fut.result()
                    except Exception:
                        matched_path, ratio = "", 0.0
                        logger.debug(
                            "ScanForRaw match task failed: %s",
                            traceback.format_exc())
                    matches[ws_folder] = {
                        "book": book,
                        "path": matched_path,
                        "ratio": ratio,
                        "accepted": bool(matched_path),
                    }
        except Exception:
            logger.debug(
                "ScanForRaw match pass crashed: %s",
                traceback.format_exc())
        return matches

    def run(self) -> None:
        candidates = (list(self._prewalked)
                      if self._prewalked is not None
                      else self._walk())
        if self._cancelled:
            return
        matches = self._compute_matches(candidates)
        if self._cancelled:
            return
        self.results.emit(self._folder, candidates, matches)


class _ScanForRawDialog(QDialog):
    """Pair every In Progress workspace to a raw source file on disk.

    The user points this dialog at a directory, picks Exact or Fuzzy
    matching (with a slider for the similarity threshold), previews
    the candidate pairings in a table, tweaks the per-row selection
    if they want, and clicks Apply. Each accepted pairing writes:

      * ``<workspace>/source_epub.txt`` — the authoritative pointer
        the scanner consults first, so subsequent scans resolve the
        raw directly without going back through title matching.
      * An entry in ``library_raw_inputs.txt`` for the resolved raw
        so :func:`_find_raw_source_for_folder`'s registry lookup
        picks it up even if the sidecar is lost later.

    Matching heuristic:

      * Exact: the workspace's normalized-key set intersects the
        candidate raw's normalized filename stem.
      * Fuzzy: the highest ``difflib.SequenceMatcher.ratio()`` among
        the workspace's keys vs. the candidate stem, above the
        user-chosen threshold. Ratios are computed against the
        normalized key so the comparison survives Windows filename
        sanitization quirks.
    """

    applied = Signal(int)  # number of pairings written

    MATCH_EXACT = "exact"
    MATCH_FUZZY = "fuzzy"

    _SUPPORTED_EXTS = (".epub", ".txt", ".pdf", ".html", ".htm")

    def __init__(self,
                 in_progress_books: list[dict],
                 config: dict | None = None,
                 parent=None):
        super().__init__(parent)
        self._config = config or {}
        # Copy so we don't hold live pointers into the parent dialog's
        # state (the scanner thread is free to replace the list).
        # We only care about workspace-backed cards — the pairing
        # writes ``source_epub.txt`` into the output folder, so
        # library-filed cards (which don't own a workspace) can't be
        # paired this way and are skipped.
        self._books: list[dict] = [
            dict(b) for b in (in_progress_books or [])
            if bool(b.get("output_folder"))
            and (b.get("missing_raw_file")
                 or not b.get("raw_source_path"))
        ]
        self._mode = self._config.get(
            "epub_library_scan_raw_mode", self.MATCH_EXACT)
        if self._mode not in (self.MATCH_EXACT, self.MATCH_FUZZY):
            self._mode = self.MATCH_EXACT
        try:
            self._threshold = int(
                self._config.get("epub_library_scan_raw_threshold", 70))
        except (TypeError, ValueError):
            self._threshold = 70
        self._threshold = max(40, min(95, self._threshold))
        self._scan_folder = self._config.get(
            "epub_library_scan_raw_folder", "") or ""
        # Extensions the user wants to scan for. Default is Auto:
        # derive the set from each in-progress workspace's known
        # ``workspace_kind`` (the scanner already classifies each
        # folder as epub / txt / pdf / image / other via
        # :func:`_detect_workspace_kind`, which is what drives the
        # per-card extension badge). Auto mode means the scan
        # defaults to exactly the extensions the visible missing-raw
        # cards need — so an EPUB-only library doesn't bother
        # hashing every TXT / PDF file in the chosen folder.
        #
        # The user can flip to manual mode by unchecking the Auto
        # toggle and then toggling individual extension checkboxes.
        # Persisted state:
        #   * ``epub_library_scan_raw_auto``   (bool, default True)
        #   * ``epub_library_scan_raw_exts``   (manual-mode selection)
        valid_exts = {"epub", "txt", "pdf", "html"}
        self._valid_exts = valid_exts
        auto_default = True
        try:
            self._auto_mode = bool(self._config.get(
                "epub_library_scan_raw_auto", auto_default))
        except Exception:
            self._auto_mode = auto_default
        stored_exts = self._config.get(
            "epub_library_scan_raw_exts", None)
        if isinstance(stored_exts, (list, tuple, set)) and stored_exts:
            self._manual_exts: set[str] = {
                str(e).strip().lower().lstrip(".")
                for e in stored_exts
                if str(e).strip().lower().lstrip(".") in valid_exts
            }
        else:
            self._manual_exts = set(valid_exts)
        if not self._manual_exts:
            self._manual_exts = set(valid_exts)
        # Live selection used by :meth:`_rescan_folder`. Recomputed
        # from ``_books`` when Auto is on; copied from
        # ``_manual_exts`` otherwise. Seeded now so the first scan
        # (triggered from ``__init__``) has a value to read.
        self._selected_exts: set[str] = (
            self._derive_auto_exts() if self._auto_mode
            else set(self._manual_exts))
        if not self._selected_exts:
            # Defensive: never leave the set empty — an empty set
            # would walk the folder but match zero files, reading
            # to the user as a silent "no matches" even though the
            # folder is full of candidates.
            self._selected_exts = set(valid_exts)
        # Candidate file index — populated by :meth:`_rescan_folder`,
        # a list of ``(normalized_stem, absolute_path)`` tuples so the
        # fuzzy matcher can iterate without re-scanning the folder on
        # every slider tick.
        self._candidates: list[tuple[str, str]] = []
        # Background worker thread that walks the scan folder.
        # Replaced / cancelled whenever the user changes folder or
        # extension selection so we never chew CPU on a stale scan.
        self._scan_worker: _RawScanWorker | None = None
        # Current matches keyed by workspace output_folder. Each value
        # is ``(matched_path, ratio, accepted)``. ``accepted=False``
        # means the user un-ticked the checkbox in the preview.
        self._matches: dict[str, dict] = {}
        self.setWindowTitle("\U0001f50d  Scan for Raw Sources")
        self.setMinimumSize(760, 480)
        # Dialog background matches the rest of the Glossarion shell so
        # child widgets (radios / checkboxes / slider / labels) don't
        # paint on top of a default-grey / black fill. Each of those
        # widgets also sets ``background: transparent`` in its own
        # stylesheet below so the dialog colour shows through cleanly.
        self.setStyleSheet("QDialog { background: #12121e; }")
        self._setup_ui()
        # Paint a first preview immediately if the persisted folder
        # still exists on disk — otherwise the user lands on an empty
        # dialog that reads as "nothing found" even though they haven't
        # picked a folder yet.
        if self._scan_folder and os.path.isdir(self._scan_folder):
            QTimer.singleShot(0, self._rescan_folder)

    def _derive_auto_exts(self) -> set[str]:
        """Infer the extension set from the missing-raw workspaces.

        Each in-progress workspace already carries a
        ``workspace_kind`` (``epub`` / ``txt`` / ``pdf`` / ``image``
        / ``other``) that drives the per-card badge. "Auto" mode
        reuses that classification so the scan only walks the
        extensions the visible cards actually need. An ``image``
        kind is tolerated but doesn't map to a searchable text
        extension, so it's ignored; ``other`` (unknown) expands to
        every extension since we can't predict what the user will
        bring. An empty selection falls back to the full set so the
        scan doesn't no-op.
        """
        valid = self._valid_exts
        out: set[str] = set()
        for b in self._books:
            kind = (b.get("workspace_kind")
                    or b.get("type") or "").lower()
            if kind in valid:
                out.add(kind)
            elif kind in ("", "other", "in_progress"):
                # Unknown / folder-only — fall back to the full set
                # so the scan still has a chance to pair the card.
                out.update(valid)
        if not out:
            out = set(valid)
        return out

    # -- UI -----------------------------------------------------------------
    def _setup_ui(self):
        from PySide6.QtWidgets import (
            QSlider, QRadioButton, QButtonGroup, QTreeWidget,
            QTreeWidgetItem, QHeaderView, QDialogButtonBox,
            QFileDialog, QCheckBox, QProgressBar,
        )
        self._QTreeWidgetItem = QTreeWidgetItem
        self._QFileDialog = QFileDialog
        # Tracks whether a background walk / match is currently in
        # flight so UI affordances (progress bar, placeholder row in
        # the tree, Browse tooltip) can stay in sync without every
        # caller having to flip them individually.
        self._scanning = False
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        intro = QLabel(
            "Pick a folder that contains your raw source files. "
            "Glossarion will try to match each in-progress workspace "
            "to one of those files. Matched pairings get written to "
            "each workspace's <code>source_epub.txt</code> so future "
            "scans resolve the raw automatically."
        )
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.RichText)
        intro.setAttribute(Qt.WA_TranslucentBackground)
        intro.setStyleSheet(
            "color: #c8cbe0; font-size: 9.5pt; background: transparent;")
        root.addWidget(intro)

        # Folder picker row
        folder_row = QHBoxLayout()
        folder_row.setSpacing(6)
        self._folder_edit = QLineEdit(self._scan_folder)
        self._folder_edit.setPlaceholderText(
            "Folder containing raw EPUB / TXT / PDF / HTML files \u2026")
        self._folder_edit.setStyleSheet(
            "background: #1e1e2e; border: 1px solid #3a3a5e; "
            "border-radius: 4px; padding: 4px 8px; color: #e0e0e0; "
            "font-size: 9.5pt;")
        self._folder_edit.editingFinished.connect(self._on_folder_edit_done)
        folder_row.addWidget(self._folder_edit, 1)
        self._browse_btn = QPushButton("\U0001f4c2  Browse\u2026")
        self._browse_btn.setCursor(Qt.PointingHandCursor)
        self._browse_btn.setStyleSheet(
            "QPushButton { background: #3a5a7a; color: white; "
            "border-radius: 4px; padding: 6px 14px; font-size: 9pt; "
            "font-weight: bold; border: none; }"
            "QPushButton:hover { background: #4a6a8a; }"
            "QPushButton:disabled { background: #2a2a3e; color: #6a6d80; }")
        self._browse_btn.clicked.connect(self._browse_folder)
        folder_row.addWidget(self._browse_btn)
        root.addLayout(folder_row)

        # Mode + threshold row
        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        mode_lbl = QLabel("Match:")
        mode_lbl.setAttribute(Qt.WA_TranslucentBackground)
        mode_lbl.setStyleSheet(
            "color: #888; font-size: 8.5pt; background: transparent;")
        mode_row.addWidget(mode_lbl)
        self._mode_group = QButtonGroup(self)
        self._rb_exact = QRadioButton("Exact")
        self._rb_fuzzy = QRadioButton("Fuzzy")
        self._rb_exact.setAttribute(Qt.WA_TranslucentBackground)
        self._rb_fuzzy.setAttribute(Qt.WA_TranslucentBackground)
        radio_css = (
            "QRadioButton { color: #c8cbe0; font-size: 9pt; spacing: 6px; "
            "background: transparent; padding: 2px 4px; }"
            "QRadioButton::indicator { width: 14px; height: 14px; "
            "border: 1px solid #5a9fd4; border-radius: 8px; "
            "background-color: #1e1e2e; }"
            "QRadioButton::indicator:checked { background-color: "
            "qradialgradient(cx:0.5, cy:0.5, radius:0.5, "
            "fx:0.5, fy:0.5, stop:0 #20b2cc, stop:0.55 #17a2b8, "
            "stop:0.6 #1e1e2e, stop:1 #1e1e2e); "
            "border-color: #20b2cc; }"
            "QRadioButton::indicator:hover { border-color: #7bb3e0; }"
        )
        self._rb_exact.setStyleSheet(radio_css)
        self._rb_fuzzy.setStyleSheet(radio_css)
        self._rb_exact.setChecked(self._mode == self.MATCH_EXACT)
        self._rb_fuzzy.setChecked(self._mode == self.MATCH_FUZZY)
        self._mode_group.addButton(self._rb_exact)
        self._mode_group.addButton(self._rb_fuzzy)
        self._rb_exact.toggled.connect(self._on_mode_changed)
        self._rb_fuzzy.toggled.connect(self._on_mode_changed)
        mode_row.addWidget(self._rb_exact)
        mode_row.addWidget(self._rb_fuzzy)

        mode_row.addSpacing(16)
        thr_lbl = QLabel("Similarity:")
        thr_lbl.setAttribute(Qt.WA_TranslucentBackground)
        thr_lbl.setStyleSheet(
            "color: #888; font-size: 8.5pt; background: transparent;")
        mode_row.addWidget(thr_lbl)
        self._threshold_slider = QSlider(Qt.Horizontal)
        self._threshold_slider.setMinimum(40)
        self._threshold_slider.setMaximum(95)
        self._threshold_slider.setValue(self._threshold)
        self._threshold_slider.setTickPosition(QSlider.TicksBelow)
        self._threshold_slider.setTickInterval(5)
        self._threshold_slider.setFixedWidth(200)
        self._threshold_slider.setAttribute(Qt.WA_TranslucentBackground)
        self._threshold_slider.setStyleSheet(
            "QSlider { background: transparent; }"
            "QSlider::groove:horizontal { background: #2a2a3e; "
            "height: 4px; border-radius: 2px; }"
            "QSlider::sub-page:horizontal { background: #17a2b8; "
            "height: 4px; border-radius: 2px; }"
            "QSlider::add-page:horizontal { background: #2a2a3e; "
            "height: 4px; border-radius: 2px; }"
            "QSlider::handle:horizontal { background: #17a2b8; "
            "width: 14px; margin: -6px 0; border-radius: 7px; "
            "border: 1px solid #20b2cc; }")
        self._threshold_slider.valueChanged.connect(
            self._on_threshold_changed)
        mode_row.addWidget(self._threshold_slider)
        self._threshold_value_lbl = QLabel(f"{self._threshold}%")
        self._threshold_value_lbl.setAttribute(Qt.WA_TranslucentBackground)
        self._threshold_value_lbl.setStyleSheet(
            "color: #c8cbe0; font-size: 9pt; font-weight: bold; "
            "background: transparent;")
        self._threshold_value_lbl.setFixedWidth(40)
        mode_row.addWidget(self._threshold_value_lbl)

        mode_row.addStretch()
        root.addLayout(mode_row)

        # Extension checkboxes — default to Auto (derived from each
        # missing-raw workspace's known kind). The user can opt out
        # of Auto to pick extensions manually via the individual
        # checkboxes.
        ext_row = QHBoxLayout()
        ext_row.setSpacing(6)
        ext_lbl = QLabel("Extensions:")
        ext_lbl.setAttribute(Qt.WA_TranslucentBackground)
        ext_lbl.setStyleSheet(
            "color: #888; font-size: 8.5pt; background: transparent;")
        ext_row.addWidget(ext_lbl)
        ext_css = (
            "QCheckBox { color: #c8cbe0; font-size: 9pt; spacing: 6px;"
            " padding: 2px 6px; background: transparent; }"
            "QCheckBox:disabled { color: #6a6d80; }"
            "QCheckBox::indicator { width: 14px; height: 14px; "
            "border: 1px solid #5a9fd4; border-radius: 2px; "
            "background-color: #1e1e2e; }"
            "QCheckBox::indicator:checked { background-color: #17a2b8; "
            "border-color: #20b2cc; }"
            "QCheckBox::indicator:disabled { border-color: #3a3a5e; "
            "background-color: #1a1a2a; }"
            "QCheckBox::indicator:hover { border-color: #7bb3e0; }"
        )
        self._auto_cb = QCheckBox("Auto")
        self._auto_cb.setToolTip(
            "Automatically pick the extensions to scan for based on "
            "each missing-raw card's workspace kind. Turn this off "
            "to override with the checkboxes to the right."
        )
        self._auto_cb.setChecked(self._auto_mode)
        self._auto_cb.setAttribute(Qt.WA_TranslucentBackground)
        self._auto_cb.setStyleSheet(ext_css)
        self._auto_cb.toggled.connect(self._on_auto_toggled)
        ext_row.addWidget(self._auto_cb)
        self._ext_cbs: dict[str, QCheckBox] = {}
        for ext, label in (
            ("epub", ".epub"),
            ("txt",  ".txt"),
            ("pdf",  ".pdf"),
            ("html", ".html"),
        ):
            cb = QCheckBox(label)
            cb.setChecked(ext in self._selected_exts)
            cb.setAttribute(Qt.WA_TranslucentBackground)
            cb.setStyleSheet(ext_css)
            cb.toggled.connect(
                lambda checked, e=ext: self._on_ext_toggled(e, checked))
            self._ext_cbs[ext] = cb
            ext_row.addWidget(cb)
        # Auto mode disables the individual toggles since their
        # values are derived from the books. The checkboxes still
        # *display* the auto-derived state so the user sees which
        # extensions will be scanned.
        self._apply_auto_mode_ui()
        ext_row.addStretch()
        root.addLayout(ext_row)

        # Preview table
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels([
            "", "Workspace", "Matched file", "Similarity"
        ])
        self._tree.setRootIsDecorated(False)
        self._tree.setSelectionMode(QTreeWidget.SingleSelection)
        self._tree.setStyleSheet(
            "QTreeWidget { background: #1a1a2a; color: #e0e0e0; "
            "border: 1px solid #2a2a3e; border-radius: 6px; "
            "font-size: 9pt; }"
            "QTreeWidget::item { padding: 4px; }"
            "QTreeWidget::item:selected { background: #3a3a5e; }"
            "QHeaderView::section { background: #1e1e2e; color: #b0b0c0; "
            "padding: 4px 8px; border: none; border-bottom: "
            "1px solid #2a2a3e; font-weight: bold; font-size: 8.5pt; }")
        header = self._tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._tree.itemChanged.connect(self._on_tree_item_changed)
        root.addWidget(self._tree, 1)

        # Status line + indeterminate progress bar. The bar is only
        # shown while a background scan is running so the user has a
        # clear visual signal that work is in flight \u2014 the rest
        # of the dialog stays interactive, but widgets that would
        # implicitly restart the scan (Browse button, folder edit)
        # are disabled so a rapid double-click doesn't kick off a
        # second scan before the first finishes.
        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(8)
        self._status_lbl = QLabel("")
        self._status_lbl.setAttribute(Qt.WA_TranslucentBackground)
        self._status_lbl.setStyleSheet(
            "color: #8ab4d0; font-size: 9pt; background: transparent;")
        status_row.addWidget(self._status_lbl, 1)
        self._scan_progress = QProgressBar()
        self._scan_progress.setRange(0, 0)  # indeterminate
        self._scan_progress.setTextVisible(False)
        self._scan_progress.setFixedHeight(6)
        self._scan_progress.setFixedWidth(160)
        self._scan_progress.setStyleSheet(
            "QProgressBar { background: #1a1a2a; border: 1px solid "
            "#2a2a3e; border-radius: 3px; }"
            "QProgressBar::chunk { background: #17a2b8; "
            "border-radius: 3px; }")
        self._scan_progress.hide()
        status_row.addWidget(self._scan_progress, 0)
        root.addLayout(status_row)

        # Dialog buttons
        bbox = QDialogButtonBox(
            QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        self._apply_btn = bbox.button(QDialogButtonBox.Apply)
        self._apply_btn.setText("Apply Pairings")
        self._apply_btn.setCursor(Qt.PointingHandCursor)
        self._apply_btn.clicked.connect(self._apply_matches)
        bbox.rejected.connect(self.reject)
        bbox.setStyleSheet(
            "QDialogButtonBox { background: transparent; }"
            "QPushButton { background: #3a5a7a; color: white; "
            "border-radius: 4px; padding: 6px 14px; font-size: 9pt; "
            "font-weight: bold; border: none; min-width: 120px; }"
            "QPushButton:hover { background: #4a6a8a; }"
            "QPushButton:disabled { background: #2a2a3e; color: #555; }")
        root.addWidget(bbox)

        # Fuzzy slider only meaningful in fuzzy mode — grey it out in
        # exact mode so the UI reads as "similarity doesn't apply".
        self._update_threshold_enabled()

    # -- Scanning-state helper ---------------------------------------------
    def _set_scanning_state(self, active: bool, message: str = ""):
        """Flip every \"scan in progress\" UI affordance at once.

        *active* True shows the progress bar, styles the status
        label as a teal \"scanning\" pill, inserts a placeholder
        row in the tree, and locks the Browse button + folder edit
        so the user can't accidentally restart the scan while the
        previous one is still walking the disk. *active* False
        reverts everything and leaves the tree ready to be
        repainted with real results.
        """
        active = bool(active)
        self._scanning = active
        if active:
            self._scan_progress.show()
            self._status_lbl.setText(
                message or "\u23f3 Scanning\u2026")
            self._status_lbl.setStyleSheet(
                "color: #20b2cc; font-size: 9pt; font-weight: bold; "
                "background: rgba(32, 178, 204, 0.12); "
                "border: 1px solid #17a2b8; border-radius: 4px; "
                "padding: 2px 8px;")
            # Lock the inputs that would implicitly restart the
            # scan. Mode / slider / extension toggles stay live so
            # the user can still tweak them \u2014 those call
            # :meth:`_rematch_only` / :meth:`_rescan_folder` which
            # cancel the in-flight worker first.
            if hasattr(self, "_browse_btn"):
                self._browse_btn.setEnabled(False)
                self._browse_btn.setToolTip(
                    "Scan in progress \u2014 please wait\u2026")
            self._folder_edit.setEnabled(False)
            self._apply_btn.setEnabled(False)
            # Show a clear placeholder row in the tree so stale
            # results from a previous scan don't linger on screen.
            self._tree.blockSignals(True)
            self._tree.clear()
            placeholder = self._QTreeWidgetItem([
                "",
                "\u23f3  Scanning folder\u2026",
                "Matching workspaces against raw sources\u2026",
                "",
            ])
            placeholder.setFlags(Qt.ItemIsEnabled)
            from PySide6.QtGui import QColor
            placeholder.setForeground(1, QColor("#20b2cc"))
            placeholder.setForeground(2, QColor("#8ab4d0"))
            self._tree.addTopLevelItem(placeholder)
            self._tree.blockSignals(False)
        else:
            self._scan_progress.hide()
            self._status_lbl.setStyleSheet(
                "color: #8ab4d0; font-size: 9pt; background: transparent;")
            if hasattr(self, "_browse_btn"):
                self._browse_btn.setEnabled(True)
                self._browse_btn.setToolTip("")
            self._folder_edit.setEnabled(True)

    # -- Folder + mode handlers --------------------------------------------
    def _browse_folder(self):
        # Defence-in-depth: the button is disabled while a scan is
        # running, but guard the handler as well in case the click
        # arrives between state transitions.
        if getattr(self, "_scanning", False):
            return
        start = self._folder_edit.text().strip() or str(Path.home())
        if not os.path.isdir(start):
            start = str(Path.home())
        folder = self._QFileDialog.getExistingDirectory(
            self, "Pick a folder containing raw source files", start)
        if folder:
            self._folder_edit.setText(folder)
            self._on_folder_edit_done()

    def _on_folder_edit_done(self):
        new_folder = self._folder_edit.text().strip()
        if new_folder == self._scan_folder:
            return
        self._scan_folder = new_folder
        try:
            self._config["epub_library_scan_raw_folder"] = new_folder
        except Exception:
            pass
        self._rescan_folder()

    def _on_mode_changed(self, _checked: bool):
        mode = (self.MATCH_EXACT if self._rb_exact.isChecked()
                else self.MATCH_FUZZY)
        if mode == self._mode:
            return
        self._mode = mode
        try:
            self._config["epub_library_scan_raw_mode"] = mode
        except Exception:
            pass
        self._update_threshold_enabled()
        # Mode change is a re-match only, so reuse the cached
        # candidates from the last walk (no folder rescan).
        self._rematch_only()

    def _on_threshold_changed(self, value: int):
        self._threshold = int(value)
        self._threshold_value_lbl.setText(f"{self._threshold}%")
        try:
            self._config["epub_library_scan_raw_threshold"] = self._threshold
        except Exception:
            pass
        if self._mode == self.MATCH_FUZZY:
            self._rematch_only()

    def _update_threshold_enabled(self):
        enabled = self._mode == self.MATCH_FUZZY
        self._threshold_slider.setEnabled(enabled)
        self._threshold_value_lbl.setEnabled(enabled)

    def _on_ext_toggled(self, ext: str, checked: bool):
        """Toggle one of the extension filters and re-scan.

        No-op when Auto is on: the checkbox states are derived from
        the missing-raw workspaces and can't be changed manually
        without first unticking Auto. Refuses to leave the selection
        empty in manual mode — unticking the last checkbox re-ticks
        itself so the user can't put the dialog into a "nothing will
        ever match" state.
        """
        if self._auto_mode:
            return
        ext = ext.lower().lstrip(".")
        if checked:
            self._manual_exts.add(ext)
        else:
            self._manual_exts.discard(ext)
            if not self._manual_exts:
                self._manual_exts.add(ext)
                cb = self._ext_cbs.get(ext)
                if cb is not None:
                    cb.blockSignals(True)
                    cb.setChecked(True)
                    cb.blockSignals(False)
                return
        self._selected_exts = set(self._manual_exts)
        try:
            self._config["epub_library_scan_raw_exts"] = sorted(
                self._manual_exts)
        except Exception:
            pass
        self._rescan_folder()

    def _on_auto_toggled(self, checked: bool):
        """Flip the Auto extension mode on / off."""
        new_value = bool(checked)
        if new_value == self._auto_mode:
            return
        self._auto_mode = new_value
        try:
            self._config["epub_library_scan_raw_auto"] = new_value
        except Exception:
            pass
        if self._auto_mode:
            self._selected_exts = self._derive_auto_exts()
        else:
            self._selected_exts = set(self._manual_exts)
        self._apply_auto_mode_ui()
        self._rescan_folder()

    def _apply_auto_mode_ui(self):
        """Sync the extension checkboxes to the current mode.

        In Auto mode the checkboxes reflect the derived set but are
        disabled so the user can see what WILL be scanned without
        accidentally tweaking it. In manual mode the checkboxes are
        editable and reflect ``_manual_exts``.
        """
        enabled = not self._auto_mode
        display = (self._derive_auto_exts() if self._auto_mode
                   else self._manual_exts)
        for ext, cb in self._ext_cbs.items():
            cb.blockSignals(True)
            cb.setChecked(ext in display)
            cb.setEnabled(enabled)
            cb.blockSignals(False)

    # -- Scanning + matching -----------------------------------------------
    def _cancel_worker(self):
        if self._scan_worker is not None:
            try:
                self._scan_worker.cancel()
                self._scan_worker.results.disconnect()
            except Exception:
                pass
            self._scan_worker = None

    def _ext_suffixes(self) -> tuple[str, ...]:
        """Build the suffix tuple from the current extension selection.

        The HTML checkbox covers both ``.html`` and ``.htm`` because
        they're interchangeable on disk.
        """
        ext_suffixes: tuple[str, ...] = tuple(
            f".{e}" if e != "html" else ".html"
            for e in self._selected_exts
        )
        if "html" in self._selected_exts:
            ext_suffixes = ext_suffixes + (".htm",)
        return ext_suffixes

    def _rescan_folder(self):
        """Kick off a background walk + match of the current folder.

        Walk AND match both run on :class:`_RawScanWorker` so the
        dialog stays responsive even for huge folders with heavy
        Fuzzy matching. Any previously running worker is cancelled
        first so rapid folder / extension toggles don't leave
        multiple scans racing.
        """
        self._candidates.clear()
        self._cancel_worker()
        if not self._scan_folder or not os.path.isdir(self._scan_folder):
            self._set_scanning_state(False)
            self._status_lbl.setText(
                "\u26a0 Pick a folder that exists on disk.")
            self._matches.clear()
            self._tree.clear()
            self._apply_btn.setEnabled(False)
            return
        # Paint a "scanning\u2026" placeholder so the UI reflects the
        # in-flight state rather than silently showing stale rows.
        self._set_scanning_state(
            True,
            f"\u23f3 Scanning {self._scan_folder}\u2026")
        self._scan_worker = _RawScanWorker(
            self._scan_folder, self._ext_suffixes(),
            _LIBRARY_TRACKING_FILENAMES,
            books=self._books, mode=self._mode,
            threshold=self._threshold,
            parent=self,
        )
        self._scan_worker.results.connect(self._on_scan_results)
        self._scan_worker.start()

    def _rematch_only(self):
        """Re-run matching against the cached candidate list.

        Used by mode / slider changes so we don't re-walk the
        folder every time the user drags the Similarity slider.
        Falls back to a full rescan when no cached candidates exist
        yet (e.g. the user hasn't picked a folder).
        """
        if not self._candidates:
            self._rescan_folder()
            return
        self._cancel_worker()
        self._set_scanning_state(
            True, "\u23f3 Re-matching\u2026")
        self._scan_worker = _RawScanWorker(
            self._scan_folder, self._ext_suffixes(),
            _LIBRARY_TRACKING_FILENAMES,
            books=self._books, mode=self._mode,
            threshold=self._threshold,
            prewalked=list(self._candidates),
            parent=self,
        )
        self._scan_worker.results.connect(self._on_scan_results)
        self._scan_worker.start()

    @Slot(str, list, dict)
    def _on_scan_results(self, scan_folder: str,
                         candidates: list,
                         matches: dict) -> None:
        """Worker callback \u2014 runs on the main thread via Qt signal.

        The worker did both the walk and the matching, so all we
        need to do is cache candidates and paint the tree.
        """
        if scan_folder != self._scan_folder:
            return
        self._candidates = list(candidates)
        self._matches = dict(matches)
        # Clear the scanning-state UI BEFORE painting results so the
        # progress bar / placeholder row aren't briefly visible
        # alongside the real rows.
        self._set_scanning_state(False)
        self._populate_tree()

    def _populate_tree(self):
        self._tree.blockSignals(True)
        self._tree.clear()
        hits = 0
        for ws_folder, info in self._matches.items():
            book = info["book"]
            # Column 1 shows the workspace's ON-DISK folder basename
            # so the user can compare it directly to the matched raw
            # filename in column 2. The metadata-driven ``book['name']``
            # is the translated title (e.g. "The Slaves I Kicked Out")
            # which is useless for filename matching — the folder name
            # (e.g. "[393761] ㄝㅇㅏㄴㄴㄴ...") is what the
            # scanner actually pairs against.
            workspace_label = (book.get("folder_name")
                               or os.path.basename(ws_folder)
                               or book.get("name")
                               or "")
            item = self._QTreeWidgetItem([
                "",
                str(workspace_label),
                (os.path.basename(info["path"])
                 if info["path"] else "\u2014 no match"),
                (f"{info['ratio'] * 100:.0f}%"
                 if info["ratio"] > 0 else "\u2014"),
            ])
            # Tooltip on the workspace column reveals the full folder
            # path for users who want to verify which workspace on
            # disk the row points at.
            if ws_folder:
                item.setToolTip(1, ws_folder)
            item.setData(0, Qt.UserRole, ws_folder)
            if info["path"]:
                item.setToolTip(2, info["path"])
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(
                    0,
                    Qt.Checked if info["accepted"] else Qt.Unchecked)
                hits += 1
            else:
                # No candidate — disable the checkbox row and paint
                # the workspace name in a dimmed color so it reads
                # as unreachable. ``QPalette.Disabled`` /
                # ``QPalette.Text`` are *class-level* enums on
                # ``QPalette`` itself; accessing them through a
                # palette *instance* (as in
                # ``QApplication.palette().Disabled``) raises
                # ``AttributeError`` on PySide6 because the instance
                # doesn't re-export the enum values.
                from PySide6.QtGui import QPalette, QColor
                item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable
                              & ~Qt.ItemIsSelectable)
                try:
                    item.setForeground(
                        1, QApplication.palette().brush(
                            QPalette.Disabled, QPalette.Text))
                except Exception:
                    # Palette brush lookup failed on some Qt builds
                    # (themed stylesheets sometimes strip the
                    # Disabled group). Fall back to a hardcoded
                    # dimmed grey so the row still reads as
                    # unreachable.
                    item.setForeground(1, QColor("#6a6d80"))
            self._tree.addTopLevelItem(item)
        self._tree.blockSignals(False)
        mode_label = (
            "exact" if self._mode == self.MATCH_EXACT
            else f"fuzzy \u2265 {self._threshold}%")
        workspace_count = len(self._matches)
        candidate_count = len(self._candidates)
        if workspace_count == 0:
            # No missing-raw cards fed into the dialog in the first
            # place — usually means the button was opened before
            # the scanner populated any workspaces.
            self._status_lbl.setText(
                "\u24d8 No missing-raw workspaces to pair.")
        elif candidate_count == 0:
            self._status_lbl.setText(
                "\u26a0 No candidate files found in this folder "
                f"({mode_label}). Pick a different folder or "
                "enable more extensions."
            )
        elif hits == 0:
            hint = (
                "try lowering the Similarity slider"
                if self._mode == self.MATCH_FUZZY
                else "switch to Fuzzy match or rename the raw "
                     "files to match the workspace folder names")
            self._status_lbl.setText(
                f"\u26a0 0 of {workspace_count} workspace"
                f"{'s' if workspace_count != 1 else ''} matched "
                f"({candidate_count} candidate file"
                f"{'s' if candidate_count != 1 else ''} scanned, "
                f"{mode_label}). Try {hint}."
            )
        else:
            self._status_lbl.setText(
                f"\u2714 {hits} of {workspace_count} workspace"
                f"{'s' if workspace_count != 1 else ''} matched "
                f"({candidate_count} candidate file"
                f"{'s' if candidate_count != 1 else ''} scanned, "
                f"{mode_label})."
            )
        self._apply_btn.setEnabled(hits > 0)

    def _on_tree_item_changed(self, item, column: int):
        if column != 0:
            return
        ws_folder = item.data(0, Qt.UserRole)
        if not ws_folder or ws_folder not in self._matches:
            return
        self._matches[ws_folder]["accepted"] = (
            item.checkState(0) == Qt.Checked)

    # -- Qt lifecycle ------------------------------------------------------
    def closeEvent(self, event):
        """Cancel the background walk before the dialog is torn down.

        Without this the ``QThread`` can outlive the Python object;
        when it finally emits ``results`` it'd try to poke at a
        deleted C++ widget and crash the interpreter.
        """
        if self._scan_worker is not None:
            try:
                self._scan_worker.cancel()
                self._scan_worker.results.disconnect()
                self._scan_worker.quit()
                self._scan_worker.wait(250)
            except Exception:
                pass
            self._scan_worker = None
        super().closeEvent(event)

    # -- Apply -------------------------------------------------------------
    def _apply_matches(self):
        written = 0
        for ws_folder, info in self._matches.items():
            if not info.get("accepted"):
                continue
            raw_path = info.get("path") or ""
            if not raw_path or not os.path.isfile(raw_path):
                continue
            if not ws_folder or not os.path.isdir(ws_folder):
                continue
            try:
                sidecar = os.path.join(ws_folder, "source_epub.txt")
                with open(sidecar, "w", encoding="utf-8") as f:
                    f.write(os.path.abspath(raw_path))
                try:
                    record_library_raw_input(raw_path)
                except Exception:
                    logger.debug(
                        "record_library_raw_input failed for %s: %s",
                        raw_path, traceback.format_exc())
                written += 1
            except OSError as exc:
                logger.debug(
                    "Scan-for-raw sidecar write failed for %s: %s",
                    ws_folder, exc)
        self.applied.emit(written)
        QMessageBox.information(
            self, "Scan for Raw",
            f"Linked {written} workspace"
            f"{'s' if written != 1 else ''} to a raw source file. "
            "The library will refresh in a moment."
            if written else
            "No pairings were applied."
        )
        if written:
            self.accept()


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
        # File-format filter (EPUB / TXT / PDF / HTML / Image / All).
        # Persisted per-user so the chosen chip survives across
        # library-dialog opens.
        self._format_filter = self._config.get(
            'epub_library_format_filter', FORMAT_ALL)
        if self._format_filter not in {
            FORMAT_ALL, FORMAT_EPUB, FORMAT_TXT,
            FORMAT_PDF, FORMAT_HTML, FORMAT_IMAGE,
        }:
            self._format_filter = FORMAT_ALL
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

    def _make_format_btn(self, text, tooltip, fmt_key):
        """Button factory for the file-format filter chips.

        Styled like the sort chips so the grouped toolbar reads as a
        single filter strip. Uses a teal "checked" colour to stay
        visually distinct from the purple sort chips.
        """
        btn = QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setFixedHeight(26)
        btn.setMinimumWidth(40)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setCheckable(True)
        btn.setChecked(fmt_key == self._format_filter)
        btn.setStyleSheet("""
            QPushButton { background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #b0b0c0; font-size: 8.5pt; font-weight: bold; padding: 2px 8px; }
            QPushButton:hover { background: #3a3a5e; color: #e0e0e0; }
            QPushButton:checked { background: #17a2b8; border-color: #20b2cc; color: #fff; }
        """)
        btn.clicked.connect(lambda: self._set_format_filter(fmt_key))
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
        format_lbl = QLabel("Format:")
        format_lbl.setStyleSheet("color: #888; font-size: 8.5pt;")
        toolbar.addWidget(format_lbl)
        self._format_btns = {}
        for text, tip, key in [
            ("All",  "Show every file type", FORMAT_ALL),
            ("EPUB", "Show only EPUB books / workspaces", FORMAT_EPUB),
            ("TXT",  "Show only TXT translations", FORMAT_TXT),
            ("PDF",  "Show only PDF translations", FORMAT_PDF),
            ("HTML", "Show only HTML translations", FORMAT_HTML),
            ("IMG",  "Show only image-based (manga / comic) workspaces", FORMAT_IMAGE),
        ]:
            btn = self._make_format_btn(text, tip, key)
            self._format_btns[key] = btn
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
            "Show the raw source filename on every card instead of the \n"
            "translated / compiled-EPUB title. Falls back to the source-\n"
            "language title from metadata when no source filename is \n"
            "available. Useful for finding a book by the name it has \n"
            "on disk."
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
        # "Open Library Folder" lives on the shared toolbar above the
        # tabs so it's reachable from either tab without having to
        # swap over to Completed first.
        self._open_library_btn = QPushButton(
            "\U0001f4c1  Open Library Folder")
        self._open_library_btn.setCursor(Qt.PointingHandCursor)
        self._open_library_btn.setToolTip(
            f"Open {get_library_dir()} in the system file explorer.")
        self._open_library_btn.setStyleSheet(
            "QPushButton { background: #3a5a7a; color: white; "
            "border-radius: 4px; padding: 6px 14px; font-size: 9pt; "
            "font-weight: bold; border: none; }"
            "QPushButton:hover { background: #4a6a8a; }"
        )
        self._open_library_btn.clicked.connect(self._open_library_folder)
        toolbar.addWidget(self._open_library_btn)
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
        # Base stylesheet: only the In Progress / Completed *content*
        # tabs. The Scan-for-Raw action tab gets its teal-button
        # styling grafted on by :meth:`_apply_tab_stylesheet` only
        # when it's actually visible \u2014 otherwise ``QTabBar::tab:last``
        # would match whichever tab is currently the last visible one
        # (i.e. Completed when Scan-for-Raw is hidden) and the wrong
        # tab would render as a button.
        self._tabs_base_qss = """
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
        """
        # Appended only while the scan tab is visible. When hidden,
        # ``:last`` would bleed onto Completed \u2014 so the rule must
        # be removed, not just ignored.
        self._tabs_scan_button_qss = """
            QTabBar::tab:last { background: #17a2b8; color: white;
                                 border: none; border-radius: 4px;
                                 padding: 6px 14px; font-size: 9pt;
                                 font-weight: bold; margin-left: 8px;
                                 margin-right: 2px; margin-top: 2px;
                                 margin-bottom: 2px; min-width: 0; }
            QTabBar::tab:last:selected { background: #17a2b8; color: white;
                                          border: none; }
            QTabBar::tab:last:hover { background: #20b2cc; color: white;
                                       border: none; }
        """
        self._tabs.setStyleSheet(self._tabs_base_qss)


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
            "Register an EPUB with the Library and scaffold a new output "
            "folder so you can translate it later. The source file stays "
            "exactly where it is on disk \u2014 click Organize when you're "
            "ready to move it into Library/Raw."
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
        # "Add Translation" mirrors the In Progress tab's Import EPUB
        # slot — it sits next to Organize / Undo so registering a
        # compiled EPUB is reachable directly from the Completed tab.
        # Uses the same import pipeline as the Completed tab's
        # drag-drop (``target="translated"``): files are registered in
        # place via ``library_translated_inputs.txt`` and surface as
        # ``registered_translated=True`` cards. Nothing is moved until
        # Organize fires.
        self._add_translation_btn = QPushButton(
            "\U0001f4d5  Add Translation")
        self._add_translation_btn.setCursor(Qt.PointingHandCursor)
        self._add_translation_btn.setToolTip(
            "Pick one or more compiled .epub files to register with "
            "the Library's Completed tab. Files stay where they are "
            "on disk \u2014 same behaviour as dropping them onto "
            "this tab. Click Organize later to move them into "
            "Library/Translated."
        )
        self._add_translation_btn.setStyleSheet(
            "QPushButton { background: #6c63ff; color: white; "
            "border-radius: 4px; padding: 6px 14px; font-size: 9pt; "
            "font-weight: bold; border: none; }"
            "QPushButton:hover { background: #8078ff; }")
        self._add_translation_btn.clicked.connect(self._add_translation)
        comp_action_row.addWidget(self._add_translation_btn)
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
            "Drop finished .epub files here \u2014 or click\n"
            "\u201cAdd Translation\u201d above \u2014 to see them here.")
        self._comp_empty_label.setAlignment(Qt.AlignCenter)
        self._comp_empty_label.setStyleSheet("color: #555; font-size: 12pt; padding: 40px;")
        self._comp_empty_label.hide()
        comp_layout.addWidget(self._comp_empty_label)

        self._tabs.addTab(self._ip_tab, "\u23f3  In Progress")
        self._tabs.addTab(self._comp_tab, "\u2705  Completed")
        # "Scan for Raw" needs to read as a real third tab after
        # Completed, not a corner widget. This placeholder page is
        # never meant to stay selected: :meth:`_on_tab_changed`
        # immediately snaps back to the previous content tab and
        # opens the dialog instead.
        self._scan_tab = QWidget()
        self._scan_tab.setStyleSheet("background: #12121e;")
        self._scan_tab_index = self._tabs.addTab(
            self._scan_tab, "\U0001f50d  Scan for Raw")
        try:
            self._tabs.setTabVisible(self._scan_tab_index, False)
        except AttributeError:
            self._tabs.setTabEnabled(self._scan_tab_index, False)
        try:
            initial_idx = int(self._current_tab) or 0
        except (TypeError, ValueError):
            initial_idx = 0
        if initial_idx == self._scan_tab_index or initial_idx < 0:
            initial_idx = 0
        self._tabs.setCurrentIndex(initial_idx)
        self._prev_content_tab = initial_idx
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
            "\U0001f4e5\n\nDrop files to register them with the Library\n"
            "(files stay where they are \u2014 click Organize to move)\n\n"
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

    def _set_format_filter(self, fmt_key: str):
        """Change the active file-format filter and refresh both tabs."""
        if fmt_key == self._format_filter:
            # Clicking the already-active chip is a no-op — keep the
            # chip checked rather than letting Qt toggle it off and
            # leaving the user in a transient "no filter selected"
            # state that would still render as FORMAT_ALL.
            btn = self._format_btns.get(fmt_key)
            if btn is not None:
                btn.setChecked(True)
            return
        self._format_filter = fmt_key
        try:
            self._config["epub_library_format_filter"] = fmt_key
        except Exception:
            pass
        for k, btn in self._format_btns.items():
            btn.setChecked(k == fmt_key)
        self._refresh_view()

    @staticmethod
    def _format_of_book(book: dict) -> str:
        """Return the FORMAT_* key that describes *book* for filtering.

        Rules:
          * In-progress folder cards are classified by their
            ``workspace_kind`` (the scanner already resolves this from
            the raw source extension or the folder's on-disk contents).
          * Other cards (library entries, promoted compiled workspaces,
            registered-in-place translated imports) are classified by
            their ``type`` field.
          * Unknown / other values collapse to ``FORMAT_ALL`` so they
            never accidentally match a specific chip — the All chip
            always shows them.
        """
        file_type = (book.get("type") or "").lower()
        kind: str
        if file_type == "in_progress":
            kind = (book.get("workspace_kind") or "").lower()
        else:
            kind = file_type
        mapping = {
            "epub":  FORMAT_EPUB,
            "txt":   FORMAT_TXT,
            "pdf":   FORMAT_PDF,
            "html":  FORMAT_HTML,
            "image": FORMAT_IMAGE,
        }
        return mapping.get(kind, FORMAT_ALL)

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
        scan_tab_index = getattr(self, "_scan_tab_index", -1)
        if index == scan_tab_index:
            fallback = getattr(self, "_prev_content_tab", 0)
            if fallback == scan_tab_index or fallback < 0:
                fallback = 0
            # Defer both actions so the clicked tab visibly behaves as
            # a tab item, but the dialog still opens without leaving
            # the QTabWidget parked on an empty placeholder page.
            QTimer.singleShot(
                0, lambda fb=fallback: self._tabs.setCurrentIndex(fb))
            QTimer.singleShot(0, self._open_scan_for_raw)
            return
        self._prev_content_tab = index
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

    def _open_scan_for_raw(self):
        """Open the Scan-for-Raw dialog.

        Feeds the dialog every workspace-backed card with the
        ``missing_raw_file`` warning (In Progress AND Completed tab)
        since a completed workspace whose compiled output still
        points at a lost raw benefits from the pairing too. When
        the user applies pairings the dialog emits ``applied`` with
        the count; we trigger a library reload so the scanner picks
        up the freshly written ``source_epub.txt`` files.
        """
        candidates = [
            b for b in (
                list(self._in_progress_books)
                + list(self._completed_books))
            if b.get("output_folder")
        ]
        dlg = _ScanForRawDialog(
            candidates,
            config=self._config,
            parent=self,
        )
        dlg.applied.connect(lambda _n: QTimer.singleShot(
            0, self._load_books))
        dlg.exec()

    def _add_translation(self):
        """Pick compiled EPUB(s) and register them with the Completed tab.

        Mirrors the Completed tab's drag-drop flow — files stay on disk
        where they are; the Library merely tracks them via
        ``library_translated_inputs.txt`` and surfaces a
        ``registered_translated=True`` card on the Completed tab. The
        user can later promote the file(s) into ``Library/Translated``
        via the Organize button, which is also what the drag-drop path
        expects.

        Only ``.epub`` is accepted here (same contract as the drag-drop
        target) since non-EPUB compiled outputs aren't shelf artefacts.
        """
        from PySide6.QtWidgets import QFileDialog
        start_dir = str(Path.home() / "Downloads")
        if not os.path.isdir(start_dir):
            start_dir = str(Path.home())
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add translated EPUB(s) to Library",
            start_dir,
            "EPUB files (*.epub);;All files (*.*)",
        )
        if not paths:
            return
        self._import_paths_into_library(
            paths, source="picker", target="translated")

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

    def _build_workspace_title_index(self) -> dict:
        """Return a ``{normalized_title_key: workspace_book_dict}`` map.

        Walks both the current in-progress and completed book lists
        (as seen by the dialog) and indexes each workspace-backed
        entry by every candidate title we can derive — folder name,
        raw source stem, and every ``metadata.json`` title field.

        Used by :meth:`_undo_organize_prompt` to pick a restore target
        for Library/Translated files that have no ``origins['translated']``
        entry. Keys run through :func:`_norm_book_key` so comparisons
        survive the NTFS filename mangling that desynchronizes the
        on-disk stem from the metadata title (``"… Fantasy."`` vs
        ``"… Fantasy"``).
        """
        index: dict[str, dict] = {}

        def _ingest(book: dict) -> None:
            ws_folder = book.get("output_folder") or ""
            if not ws_folder:
                return
            keys: set[str] = set()
            fn = book.get("folder_name") or os.path.basename(ws_folder)
            if fn:
                keys.add(_norm_book_key(os.path.splitext(fn)[0]))
            raw = book.get("raw_source_path") or ""
            if raw:
                keys.add(_norm_book_key(
                    os.path.splitext(os.path.basename(raw))[0]))
            md = book.get("metadata_json") or {}
            if isinstance(md, dict):
                for md_key in ("title", "original_title",
                               "translated_title", "raw_title",
                               "source_title", "english_title"):
                    val = md.get(md_key)
                    if isinstance(val, str) and val.strip():
                        keys.add(_norm_book_key(val))
            for key in keys:
                if key:
                    index.setdefault(key, book)

        for book in self._in_progress_books:
            _ingest(book)
        for book in self._completed_books:
            # Library entries don't own a workspace — skip so a library
            # row matching itself doesn't produce a nonsense restore
            # target pointing back at Library/Translated.
            if book.get("in_library"):
                continue
            _ingest(book)
        return index

    def _apply_tab_stylesheet(self, scan_visible: bool) -> None:
        """Attach / detach the teal-button last-tab rule on the tab bar.

        ``QTabBar::tab:last`` matches whichever tab is currently the
        last visible one, so the styling only makes sense while the
        Scan-for-Raw tab is actually present \u2014 otherwise it would
        bleed onto the Completed tab.
        """
        qss = self._tabs_base_qss
        if scan_visible:
            qss = qss + self._tabs_scan_button_qss
        try:
            if self._tabs.styleSheet() != qss:
                self._tabs.setStyleSheet(qss)
        except Exception:
            pass

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
        # Also count files physically sitting in the library shelves.
        # Undo now covers orphan files (no origins entry) too, so the
        # button must stay enabled / labelled as long as SOMETHING is
        # restorable — not only when the origins registry has rows.
        raw_disk = self._count_library_files(get_library_raw_dir())
        trans_disk = self._count_library_files(
            get_library_translated_dir(), epub_only=True)
        raw_undo = max(raw_orig, raw_disk)
        trans_undo = max(trans_orig, trans_disk)
        try:
            self._ip_organize_btn.setText(
                f"\U0001f4e5  Organize ({raw_count})")
            self._ip_organize_btn.setEnabled(raw_count > 0)
            self._ip_undo_btn.setText(f"\u21a9  Undo ({raw_undo})")
            self._ip_undo_btn.setEnabled(raw_undo > 0)
            self._comp_organize_btn.setText(
                f"\U0001f4e5  Organize ({trans_count})")
            self._comp_organize_btn.setEnabled(trans_count > 0)
            self._comp_undo_btn.setText(f"\u21a9  Undo ({trans_undo})")
            self._comp_undo_btn.setEnabled(trans_undo > 0)
        except Exception:
            pass
        # Toggle the Scan-for-Raw TAB based on whether any visible
        # card has the ``missing_raw_file`` warning. Hidden entirely
        # otherwise \u2014 there's nothing actionable for the dialog to
        # do when every workspace's raw is already resolved. The
        # teal-button tab styling is also attached / detached here
        # so the Completed tab never inherits it when the scan tab is
        # hidden (``QTabBar::tab:last`` targets whichever tab is
        # currently the last visible one).
        try:
            missing_raw_count = sum(
                1 for b in (
                    list(self._in_progress_books)
                    + list(self._completed_books))
                if b.get("missing_raw_file"))
            self._apply_tab_stylesheet(missing_raw_count > 0)
            scan_tab_index = getattr(self, "_scan_tab_index", -1)
            if scan_tab_index >= 0:
                self._tabs.setTabText(
                    scan_tab_index,
                    (f"\U0001f50d  Scan for Raw ({missing_raw_count})"
                     if missing_raw_count > 0
                     else "\U0001f50d  Scan for Raw"))
                if missing_raw_count <= 0:
                    fallback = getattr(self, "_prev_content_tab", 0)
                    if (self._tabs.currentIndex() == scan_tab_index
                            and fallback != scan_tab_index):
                        self._tabs.setCurrentIndex(max(0, fallback))
                try:
                    self._tabs.setTabVisible(
                        scan_tab_index, missing_raw_count > 0)
                except AttributeError:
                    self._tabs.setTabEnabled(
                        scan_tab_index, missing_raw_count > 0)
        except Exception:
            pass

    @staticmethod
    def _count_library_files(folder: str,
                             epub_only: bool = False) -> int:
        """Count candidate restorable files sitting directly in *folder*.

        ``epub_only`` restricts the count to ``.epub`` files (used for
        Library/Translated where only compiled EPUBs are tracked);
        other shelves include ``.txt``, ``.pdf``, and ``.html`` too.
        Registry / tracking files (``library_*_inputs.txt``,
        ``library_origins.txt``) are always excluded so a legacy
        copy sitting inside ``Library/Raw`` doesn't inflate the
        Undo counter / enable state.
        """
        if not folder or not os.path.isdir(folder):
            return 0
        exts = (".epub",) if epub_only else (
            ".epub", ".txt", ".pdf", ".html")
        count = 0
        try:
            for entry in os.scandir(folder):
                if not entry.is_file(follow_symlinks=False):
                    continue
                if entry.name.lower() in _LIBRARY_TRACKING_FILENAMES:
                    continue
                if entry.name.lower().endswith(exts):
                    count += 1
        except (PermissionError, OSError):
            return 0
        return count

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

    def _prompt_duplicate_policy(self, collisions: list[tuple[str, str]],
                                 dest_label: str) -> str | None:
        """Ask the user how to handle one or more duplicate filenames.

        *collisions* is a list of ``(source_path, existing_dest_path)``
        tuples — one entry per file whose basename already exists in the
        destination folder. Previously the import code silently resolved
        collisions by appending a counter suffix (``name (2).epub``),
        which meant a raw EPUB dropped twice produced two separate
        copies and, on the translator side, two unrelated output
        folders. The user never got a chance to say "this is the same
        book, just replace it".

        Returns one of: ``"replace"``, ``"keep_both"``, ``"skip"``,
        or ``None`` when the user cancels (the whole import is then
        aborted). The returned policy is applied to *every* colliding
        file in the batch — per-file prompting would be much noisier
        for multi-drop imports and matches what Explorer / Finder do.
        """
        if not collisions:
            return "keep_both"
        n = len(collisions)
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Duplicate files")
        preview_lines = []
        for src, existing in collisions[:6]:
            preview_lines.append(f"  \u2022 {os.path.basename(existing)}")
        if n > 6:
            preview_lines.append(f"  \u2026 and {n - 6} more.")
        if n == 1:
            msg.setText(
                f"A file named \u201c{os.path.basename(collisions[0][1])}\u201d "
                f"already exists in {dest_label}.\n\n"
                f"What would you like to do?"
            )
        else:
            msg.setText(
                f"{n} files being imported already exist in {dest_label}:\n\n"
                + "\n".join(preview_lines)
                + "\n\nWhat would you like to do with the duplicates?"
            )
        btn_replace = msg.addButton(
            "Replace" + (" All" if n > 1 else ""), QMessageBox.AcceptRole)
        btn_keep = msg.addButton(
            "Keep Both" + (" All" if n > 1 else ""), QMessageBox.AcceptRole)
        btn_skip = msg.addButton(
            "Skip" + (" All" if n > 1 else ""), QMessageBox.AcceptRole)
        btn_cancel = msg.addButton(QMessageBox.Cancel)
        btn_replace.setToolTip(
            "Overwrite the existing file(s) with the new one(s). "
            "Cannot be undone."
        )
        btn_keep.setToolTip(
            "Keep both copies. The new file(s) get a counter suffix "
            "like \u201cname (2).epub\u201d."
        )
        btn_skip.setToolTip("Don't import the duplicate(s).")
        msg.setDefaultButton(btn_keep)
        msg.exec()
        chosen = msg.clickedButton()
        if chosen is btn_replace:
            return "replace"
        if chosen is btn_keep:
            return "keep_both"
        if chosen is btn_skip:
            return "skip"
        return None  # cancel

    def _import_paths_into_library(self, paths, source: str = "picker",
                                   target: str = "raw"):
        """Core import pipeline used by both the file picker and drag-drop.

        Both ``target`` modes now register the source file **in place**
        on disk. Nothing is copied or moved by Import / drag-drop:

          * ``"raw"`` (default): appends the absolute path to
            ``Library/Raw/library_raw_inputs.txt`` and scaffolds an
            output folder under the configured output root with a
            ``source_epub.txt`` sidecar pointing at the original
            location. Surfaces as a Not Started card on the In
            Progress tab. Used by the "Import EPUB" button and by
            drag-drop while the **In Progress** tab is active.
          * ``"translated"``: appends the absolute path to
            ``Library/library_translated_inputs.txt``. Surfaces as a
            card on the Completed tab. Only ``.epub`` is accepted;
            other types are reported as skipped. Used by drag-drop
            while the **Completed** tab is active.

        The Organize button is the deliberate move step for both
        modes: it moves the file into ``Library/Raw`` or
        ``Library/Translated`` and records an entry in
        ``library_origins.txt`` so Undo Move can reverse it. Because
        nothing is relocated during Import itself, no collision dialog
        is shown for either target — duplicate registrations simply
        dedupe by normalized path in the registry files.

        The library is refreshed and a summary dialog is shown after
        processing so batch imports don't spam modal messages.
        """
        if not paths:
            return
        if target == "translated":
            supported_exts = (".epub",)
            dest_label = "Library (registered in place)"
            dest_dir = get_library_translated_dir()
        else:
            supported_exts = (".epub", ".txt", ".pdf", ".html", ".htm")
            dest_label = "Library (registered in place)"
            dest_dir = get_library_raw_dir()

        # First pass: validate each path. Neither target relocates
        # files anymore, so there's no collision bucket — everything
        # accepted lands in ``fresh`` and runs through
        # :meth:`_import_single_file`, which just appends to the
        # appropriate input registry (+ scaffolds an output folder
        # for the raw side).
        imported: list[str] = []
        skipped: list[str] = []
        errors: list[str] = []
        fresh: list[str] = []
        for raw_path in paths:
            if not raw_path:
                continue
            try:
                path = os.path.abspath(raw_path)
            except (TypeError, ValueError):
                continue
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
            fresh.append(path)

        collisions: list[tuple[str, str]] = []  # never populated now
        collision_policy = "keep_both"

        def _process(path: str, policy: str) -> None:
            try:
                dest = self._import_single_file(
                    path, target=target, collision_policy=policy)
                if dest:
                    imported.append(dest)
                elif policy == "skip":
                    skipped.append(
                        f"{os.path.basename(path)} (duplicate \u2014 skipped)")
            except Exception as exc:
                logger.error("Import failed for %s: %s\n%s",
                             path, exc, traceback.format_exc())
                errors.append(f"{os.path.basename(path)}: {exc}")

        for p in fresh:
            _process(p, "keep_both")  # policy doesn't matter, no collision
        for src, _existing in collisions:
            _process(src, collision_policy)
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
                    f"\u2705  Registered {len(imported)} file"
                    f"{'s' if len(imported) != 1 else ''} "
                    f"with the Library"
                )
            elif imported and errors:
                self._show_toast(
                    f"\u26a0\ufe0f  Registered {len(imported)} with the "
                    f"Library, failed {len(errors)}"
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
            target_tab = (
                "Library/Translated" if target == "translated"
                else "Library/Raw"
            )
            parts.append(
                f"Registered {len(imported)} file"
                f"{'s' if len(imported) != 1 else ''} with the "
                f"Library \u2014 no files were moved. Click "
                f"\u201cOrganize\u201d when you're ready to move "
                f"them into {target_tab}."
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
        if imported and target != "translated":
            parts.append(
                "\nRight-click any card and choose \u201cLoad for translation\u201d "
                "when you're ready to translate."
            )
        body = "\n".join(parts)
        if imported and not errors:
            QMessageBox.information(self, title, body)
        else:
            QMessageBox.warning(self, title, body)

    def _import_single_file(self, path: str, target: str = "raw",
                             collision_policy: str = "keep_both"
                             ) -> str | None:
        """Register *path* with the library (raw or translated side).

        Both branches now leave the source file **exactly where it is**
        on disk. Nothing is copied or moved — the import just wires
        the path into the library's tracking files so a card can
        surface on the appropriate tab. Moving into ``Library/Raw`` /
        ``Library/Translated`` is a separate, deliberate step
        triggered by the Organize button. Both the raw and translated
        registrations are fully reversible: Organize writes an entry
        into ``library_origins.txt`` before relocating, and Undo Move
        restores the file and re-adds it to the appropriate input
        registry so the card reappears on its tab.

        Per-target wiring:

          * ``"raw"`` — appends the absolute path to
            ``Library/Raw/library_raw_inputs.txt`` and scaffolds an
            output folder under the configured output root with a
            ``source_epub.txt`` sidecar pointing at the *original*
            location. The In Progress tab's scanner uses these to
            surface a Not Started card for the file.
          * ``"translated"`` — appends the absolute path to
            ``Library/library_translated_inputs.txt``. The Completed
            tab's scanner includes these as ``in_library=False`` +
            ``registered_translated=True`` cards so the user can
            read them and the Organize button can promote them into
            ``Library/Translated``.

        *collision_policy* is accepted for backwards compatibility
        but no longer matters — nothing is being relocated here, so
        there's nothing to collide with. It remains in the signature
        so future callers can still pass it without a TypeError.

        Returns the *original* absolute path on success, or ``None``
        on failure. Raises no exceptions — failures are logged and
        surfaced via the caller's aggregated error list.
        """
        path_abs = os.path.abspath(path)
        if target == "translated":
            # ---- Translated branch: register in place.
            try:
                record_library_translated_input(path_abs)
            except Exception as exc:
                logger.error(
                    "Translated registration failed for %s: %s\n%s",
                    path_abs, exc, traceback.format_exc())
                raise
            return path_abs

        # ---- Raw branch: register in place + scaffold output folder.
        try:
            record_library_raw_input(path_abs)
            roots = _resolve_output_roots(self._config)
            if roots:
                output_root = roots[0]
                base = os.path.splitext(os.path.basename(path_abs))[0]
                output_folder = os.path.join(output_root, base)
                os.makedirs(output_folder, exist_ok=True)
                # ``source_epub.txt`` points at the *real* location of
                # the raw source so the translator (and the In Progress
                # scanner) can find it without going through
                # Library/Raw. When the user later runs Organize, it
                # moves the file and rewrites this sidecar to the new
                # ``Library/Raw\…`` path.
                sidecar = os.path.join(output_folder, "source_epub.txt")
                with open(sidecar, "w", encoding="utf-8") as f:
                    f.write(path_abs)
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
            logger.error("Raw registration failed for %s: %s\n%s",
                         path_abs, exc, traceback.format_exc())
            raise
        return path_abs

    def _ensure_output_override_matches(self, books: list) -> bool:
        """Warn the user + switch the output-directory override when it
        doesn't match the folder a flash card's translation lives under.

        Flash cards get scanned from BOTH the configured
        ``OUTPUT_DIRECTORY`` override and the implicit default fallback
        (see :func:`_resolve_output_roots`), so the Library surfaces
        books whose workspaces live in either location. Loading a card
        "for translation" without aligning the override means a
        subsequent Run would write the new output under the current
        override root — splitting the progress across two different
        folders instead of resuming inside the card's existing
        workspace. This helper prompts the user whenever that drift is
        detected, and (on confirmation) rewrites the override in
        ``config['output_directory']``, ``os.environ``, the live
        ``other_settings`` UI entry, and the persisted ``config.json``
        — matching what :func:`other_settings._on_output_dir_changed`
        does when the user edits the field by hand.

        Returns True when loading can proceed (no mismatch, or the user
        accepted the switch); False when the user cancels so the caller
        can abort the emit cleanly.
        """
        # Collect the distinct expected roots across every book. Each
        # root corresponds to a different card-producing output
        # location — normally one, but mixed selections can span more.
        expected_roots: list[str] = []
        seen_keys: set[str] = set()
        for b in books or []:
            root = _expected_output_root_for_book(b)
            if not root:
                continue
            key = os.path.normcase(os.path.normpath(os.path.abspath(root)))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            expected_roots.append(root)
        if not expected_roots:
            return True

        current_override = (
            os.environ.get("OUTPUT_DIRECTORY")
            or (self._config.get("output_directory") if self._config else "")
            or ""
        ).strip()
        default_root = _default_output_root()
        current_effective = current_override or default_root

        mismatched: list[tuple[str, str]] = []
        for b in books or []:
            root = _expected_output_root_for_book(b)
            if not root:
                continue
            if not _output_paths_equal(root, current_effective):
                mismatched.append((b.get("name") or "", root))
        if not mismatched:
            return True

        # Pick the target root. If the first mismatched root IS the
        # implicit default, clear the override (empty string) so the
        # translator falls back to the default root the same way an
        # unset field would. Otherwise set the override to that root.
        target_root = mismatched[0][1]
        new_override = ("" if _output_paths_equal(target_root, default_root)
                        else target_root)

        current_label = current_override or f"{default_root}  (default)"
        new_label = new_override or f"{default_root}  (default)"

        preview: list[str] = []
        for name, root in mismatched[:5]:
            label = name or os.path.basename(os.path.normpath(root))
            preview.append(f"  \u2022 {label}\n      {root}")
        if len(mismatched) > 5:
            preview.append(f"  \u2026 and {len(mismatched) - 5} more")

        multi_root_warning = ""
        if len(expected_roots) > 1:
            multi_root_warning = (
                "\n\n\u26a0 The selection spans multiple output folders; "
                "only the first mismatched root will be applied. Load "
                "cards from one folder at a time to avoid this."
            )

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Output Folder Mismatch")
        msg.setText(
            "The selected translation is saved under a different folder "
            "than the current output-folder override.\n\n"
            f"Current override:\n  {current_label}\n\n"
            f"Will switch to:\n  {new_label}\n\n"
            "Mismatched books:\n"
            + "\n".join(preview)
            + multi_root_warning
            + "\n\nUpdate the override so a new translation run writes "
            "into the same folder as the existing progress?"
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Yes)
        if msg.exec() != QMessageBox.Yes:
            return False

        # Apply the new override to the in-memory config + the active
        # process env var so the next run picks it up immediately.
        try:
            if self._config is not None:
                self._config["output_directory"] = new_override
            if new_override:
                os.environ["OUTPUT_DIRECTORY"] = new_override
            else:
                os.environ.pop("OUTPUT_DIRECTORY", None)
        except Exception:
            logger.debug(
                "Failed to apply output_directory override: %s",
                traceback.format_exc(),
            )

        # Sync the Other Settings dialog's live UI entry if present —
        # the field is bound to the same config key via
        # ``_on_output_dir_changed``, so leaving it stale would let the
        # user see the "old" path in the settings dialog until they
        # next edited it. Walking the parent chain mirrors
        # :func:`_persist_config_via_parent`'s traversal so we find
        # whichever ancestor hosts the translator GUI's attributes.
        try:
            parent = self.parent() if hasattr(self, "parent") else None
        except Exception:
            parent = None
        while parent is not None:
            entry = getattr(parent, "output_dir_entry", None)
            if entry is not None and hasattr(entry, "setText"):
                try:
                    entry.blockSignals(True)
                    entry.setText(new_override)
                    entry.blockSignals(False)
                except Exception:
                    logger.debug(
                        "Failed to sync output_dir_entry: %s",
                        traceback.format_exc(),
                    )
                break
            try:
                parent = parent.parent()
            except Exception:
                break

        # Persist to ``config.json`` so the change survives a restart.
        try:
            _persist_config_via_parent(self)
        except Exception:
            logger.debug(
                "Failed to persist config after override switch: %s",
                traceback.format_exc(),
            )

        return True

    def _load_for_translation(self, book: dict):
        """Push the card's raw source path into the translator's input field.

        Emits :attr:`import_epub_requested` which :class:`TranslatorGUI`
        handles. Falls back to a warning when no raw path is available.
        Consults :meth:`_ensure_output_override_matches` first so a
        card whose workspace lives under a different output root than
        the current override triggers a warning + automatic switch.
        """
        raw = book.get("raw_source_path") or book.get("path") or ""
        if not raw or not os.path.isfile(raw):
            QMessageBox.warning(self, "Load for translation",
                                "No raw source file is available for this card.")
            return
        if not self._ensure_output_override_matches([book]):
            return
        try:
            self.import_epub_requested.emit(raw)
            record_library_raw_input(raw)
        except Exception:
            logger.debug("Failed to emit import_epub_requested: %s",
                         traceback.format_exc())

    def _load_multi_for_translation(self, paths: list,
                                    books: list | None = None):
        """Push one-or-many raw source paths into the translator's input.

        Deduplicates by normalized path, emits ``import_epubs_requested``
        when more than one file is loaded (so the receiver can show a
        "N files selected" summary), or ``import_epub_requested`` for the
        single-file case. Every loaded path is recorded in the raw-inputs
        registry so subsequent library scans can resolve it.

        When *books* is provided (the context-menu caller does so) we
        first consult :meth:`_ensure_output_override_matches` so any
        card whose workspace lives under a different output root than
        the current override prompts the user to switch the override.
        Omitted ``books`` keeps the legacy "just emit the paths"
        behavior for any future caller that doesn't have the source
        dicts on hand.
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
        if books and not self._ensure_output_override_matches(books):
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
        # Two sources are pulled together:
        #   * Completed-tab cards whose ``path`` points at a compiled
        #     EPUB sitting somewhere other than ``Library/Translated``
        #     — includes both promoted-output-folder cards and
        #     registered-in-place translated imports.
        #   * ``library_translated_inputs.txt`` directly, as a safety
        #     net in case a registered entry hasn't made it into the
        #     scan result yet (first auto-refresh still pending, etc.).
        translated_moves: list[tuple[dict, str]] = []
        seen_trans_keys: set[str] = set()

        def _queue_trans(book: dict, p: str) -> None:
            if not p or not os.path.isfile(p):
                return
            if not p.lower().endswith(".epub"):
                return  # only compiled .epub files get organized for now
            parent = os.path.normcase(os.path.normpath(
                os.path.abspath(os.path.dirname(p))))
            if parent == trans_abs:
                return
            key = os.path.normcase(os.path.normpath(os.path.abspath(p)))
            if key in seen_trans_keys:
                return
            seen_trans_keys.add(key)
            translated_moves.append((book, p))

        for book in self._completed_books:
            # Skip Library entries (they already live in Library/Translated).
            if book.get("in_library"):
                continue
            _queue_trans(book, book.get("path") or "")
        # Belt-and-suspenders: also walk the registry directly so a
        # just-dropped file still gets organized even if the card
        # list hasn't refreshed yet.
        for p in load_library_translated_inputs():
            _queue_trans({"registered_translated": True}, p)

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

        # Pre-scan for name collisions in Library/Raw and
        # Library/Translated. If any exist, ask the user once for a
        # single policy that applies to every duplicate in this run —
        # mirrors the drag-drop import prompt so a name collision can
        # no longer silently auto-rename with ``(2)`` / ``(3)``.
        raw_collisions: list[tuple[str, str]] = []
        for _b, src in raw_moves:
            candidate = os.path.join(raw_dir, os.path.basename(src))
            if (os.path.isfile(candidate)
                    and os.path.abspath(src) != os.path.abspath(candidate)):
                raw_collisions.append((src, candidate))
        trans_collisions: list[tuple[str, str]] = []
        for _b, src in translated_moves:
            candidate = os.path.join(trans_dir, os.path.basename(src))
            if (os.path.isfile(candidate)
                    and os.path.abspath(src) != os.path.abspath(candidate)):
                trans_collisions.append((src, candidate))
        all_collisions = raw_collisions + trans_collisions
        collision_policy = "keep_both"
        if all_collisions:
            # Combined dest label for the prompt so a mixed
            # raw+translated batch reads naturally.
            if raw_collisions and trans_collisions:
                dest_label = "Library/Raw and Library/Translated"
            elif raw_collisions:
                dest_label = "Library/Raw"
            else:
                dest_label = "Library/Translated"
            collision_policy = self._prompt_duplicate_policy(
                all_collisions, dest_label)
            if collision_policy is None:
                # User cancelled the organize entirely.
                return
        # Fast-lookup set of source paths whose dest already exists,
        # so each move loop can apply *collision_policy* while still
        # letting non-colliding files fall through unchanged.
        colliding_srcs = {
            os.path.normcase(os.path.normpath(os.path.abspath(s)))
            for s, _d in all_collisions
        }

        def _src_has_collision(src: str) -> bool:
            try:
                key = os.path.normcase(os.path.normpath(
                    os.path.abspath(src)))
            except Exception:
                return False
            return key in colliding_srcs

        origins = _load_origins()
        raw_origins = dict(origins.get("raw", {}) or {})
        trans_origins = dict(origins.get("translated", {}) or {})
        pair_map = dict(origins.get("pairs", {}) or {})

        moved_raw = 0
        moved_trans = 0
        skipped_raw = 0
        skipped_trans = 0
        errors: list[str] = []
        # Per-book dest-basename trackers (keyed by ``id(book)``) so we can
        # pair translated↔raw after both moves finish. Books that only
        # have one side moved in this run contribute a partial entry and
        # we fall back to ``raw_source_path`` on the book dict below.
        raw_dest_by_book: dict[int, str] = {}
        trans_dest_by_book: dict[int, str] = {}
        # Collected (old_abs_path, new_abs_path) pairs for every move in
        # this run — emitted via :attr:`files_reorganized` at the end so
        # the translator GUI can update any stale paths it's holding
        # onto (e.g. the "Input file" line edit still pointing at
        # ``Downloads/novel.epub`` after the raw moved into
        # ``Library/Raw/novel.epub``).
        path_moves: list[tuple[str, str]] = []

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

        def _resolve_dest(directory: str, base_name: str, src: str
                          ) -> str | None:
            """Pick the destination path for *src* under *collision_policy*.

            Returns ``None`` when the file should be skipped (policy =
            "skip" on a collision). For ``"replace"`` the pre-existing
            file is removed so :func:`shutil.move` lands atomically; for
            ``"keep_both"`` we fall through to the counter-suffix path.
            """
            dest = os.path.join(directory, base_name)
            if not _src_has_collision(src):
                return dest
            if collision_policy == "skip":
                return None
            if collision_policy == "replace":
                try:
                    if os.path.isfile(dest):
                        os.remove(dest)
                except OSError as rm_exc:
                    # Log and fall back to keep_both so the move doesn't
                    # hard-fail just because the replace couldn't happen.
                    logger.debug("Replace-on-organize remove failed: %s",
                                 rm_exc)
                    return _unique_dest(directory, base_name)
                return dest
            return _unique_dest(directory, base_name)

        # Raw sources: MOVE + update source_epub.txt pointer.
        for book, src in raw_moves:
            try:
                dest = _resolve_dest(raw_dir, os.path.basename(src), src)
                if dest is None:  # user chose Skip All
                    skipped_raw += 1
                    continue
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
                path_moves.append(
                    (os.path.abspath(src), os.path.abspath(dest)))
                moved_raw += 1
            except Exception as exc:
                errors.append(f"raw:{os.path.basename(src)}: {exc}")

        # Translated compiled EPUBs: MOVE into Library/Translated.
        for book, src in translated_moves:
            try:
                dest = _resolve_dest(trans_dir, os.path.basename(src), src)
                if dest is None:
                    skipped_trans += 1
                    continue
                shutil.move(src, dest)
                trans_origins[os.path.basename(dest)] = os.path.abspath(src)
                trans_dest_by_book[id(book)] = os.path.basename(dest)
                path_moves.append(
                    (os.path.abspath(src), os.path.abspath(dest)))
                # If this file was in the registered-in-place
                # translated registry, drop it now — the in-place
                # registration is superseded by the origins entry
                # above, which is what Undo keys on.
                try:
                    remove_library_translated_input(src)
                except Exception:
                    logger.debug(
                        "Failed to prune translated-inputs entry for %s",
                        src,
                    )
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
        if skipped_raw or skipped_trans:
            total_skipped = skipped_raw + skipped_trans
            summary_parts.append(
                f"Skipped {total_skipped} duplicate"
                f"{'s' if total_skipped != 1 else ''}."
            )
        summary = "\n".join(summary_parts) or "Nothing was moved."
        if errors:
            summary += (f"\n\n{len(errors)} error"
                        f"{'s' if len(errors) != 1 else ''}:\n"
                        + "\n".join(errors[:5]))
        # Notify listeners (the translator GUI) BEFORE the summary
        # modal so when the user dismisses the dialog their input
        # field already reflects the new library location — clicking
        # Run immediately after Organize no longer points at a stale
        # Downloads path.
        if path_moves:
            try:
                self.files_reorganized.emit(list(path_moves))
            except Exception:
                logger.debug("files_reorganized emit failed: %s",
                             traceback.format_exc())
        QMessageBox.information(self, "Organize Files into Library", summary)
        self._load_books()

    def _undo_organize_prompt(self):
        """Ask the user which category to undo, then reverse those moves.

        Prompts with three buttons: Raw, Translated, All. Each button
        moves the affected files back to their original locations — by
        origins registry lookup when possible, otherwise falling back
        to a best-guess workspace target derived from title matching
        (same rules the scanner uses to dedup cross-tab duplicates).
        Also rewrites any stale ``source_epub.txt`` pointers.

        Files actually sitting in ``Library/Raw`` / ``Library/Translated``
        are enumerated *on disk* as well as from the origins registry,
        so orphan files dropped in through any non-organize route
        (manual copy, legacy install with a missing origins.txt, etc.)
        still get processed instead of being silently ignored.
        """
        origins = _load_origins()
        raw_map = dict(origins.get("raw", {}) or {})
        trans_map = dict(origins.get("translated", {}) or {})
        pair_map = dict(origins.get("pairs", {}) or {})

        raw_dir = get_library_raw_dir()
        trans_dir = get_library_translated_dir()

        # Extend raw_map / trans_map with every EPUB actually on disk
        # in the library shelves so "Undo Move" covers orphan files
        # too. For orphans we compute a best-guess restore target via
        # title matching against the scanned in-progress / completed
        # workspaces; if no match lands we fall back to the library
        # parent dir so the file lands somewhere reachable rather
        # than just vanishing. User feedback drove this: clicking
        # Undo on a file with no origins record used to silently do
        # nothing, leaving the file sitting in Library/Translated
        # forever.
        workspace_by_key = self._build_workspace_title_index()

        def _best_guess_restore(lib_path: str,
                                default_parent: str) -> str:
            """Pick a restore destination for *lib_path* when no origin
            entry exists. Tries title-matched workspace first, falls
            back to *default_parent* (the parent of the library dir,
            i.e. where the user is likely to find the file).
            """
            stem = os.path.splitext(os.path.basename(lib_path))[0]
            candidate_keys: set[str] = set()
            k = _norm_book_key(stem)
            if k:
                candidate_keys.add(k)
            try:
                for t in _extract_epub_titles(lib_path):
                    k = _norm_book_key(t)
                    if k:
                        candidate_keys.add(k)
            except Exception:
                logger.debug("Undo title extraction failed: %s",
                             traceback.format_exc())
            for k in candidate_keys:
                ws = workspace_by_key.get(k)
                if not ws:
                    continue
                ws_folder = ws.get("output_folder") or ""
                if ws_folder and os.path.isdir(ws_folder):
                    return os.path.join(
                        ws_folder, os.path.basename(lib_path))
            return os.path.join(
                default_parent, os.path.basename(lib_path))

        raw_orphans: dict[str, str] = {}
        trans_orphans: dict[str, str] = {}
        if os.path.isdir(raw_dir):
            raw_default_parent = os.path.dirname(
                os.path.normpath(raw_dir)) or os.path.expanduser("~")
            try:
                for entry in os.scandir(raw_dir):
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    # Never treat library registry files as content —
                    # a legacy copy of ``library_raw_inputs.txt`` used to
                    # live in ``Library/Raw`` and would trip the
                    # collision prompt if surfaced as an orphan.
                    if entry.name.lower() in _LIBRARY_TRACKING_FILENAMES:
                        continue
                    nl = entry.name.lower()
                    if not (nl.endswith(".epub") or nl.endswith(".txt")
                            or nl.endswith(".pdf") or nl.endswith(".html")):
                        continue
                    if entry.name in raw_map:
                        continue
                    raw_orphans[entry.name] = _best_guess_restore(
                        entry.path, raw_default_parent)
            except (PermissionError, OSError):
                pass
        if os.path.isdir(trans_dir):
            trans_default_parent = os.path.dirname(
                os.path.normpath(trans_dir)) or os.path.expanduser("~")
            try:
                for entry in os.scandir(trans_dir):
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    if entry.name.lower() in _LIBRARY_TRACKING_FILENAMES:
                        continue
                    if not entry.name.lower().endswith(".epub"):
                        continue
                    if entry.name in trans_map:
                        continue
                    trans_orphans[entry.name] = _best_guess_restore(
                        entry.path, trans_default_parent)
            except (PermissionError, OSError):
                pass

        # Merge orphans into the restore maps so the existing loops
        # below process them alongside registry-backed entries.
        raw_map.update(raw_orphans)
        trans_map.update(trans_orphans)

        if not raw_map and not trans_map:
            QMessageBox.information(
                self, "Undo Move",
                "No files to undo — Library/Raw and Library/Translated "
                "are both empty and the origins registry is clean.")
            return

        msg = QMessageBox(self)
        msg.setWindowTitle("Undo Move")
        raw_count = len(raw_map)
        trans_count = len(trans_map)
        orphan_note = ""
        if raw_orphans or trans_orphans:
            pieces = []
            if trans_orphans:
                pieces.append(f"{len(trans_orphans)} translated")
            if raw_orphans:
                pieces.append(f"{len(raw_orphans)} raw")
            orphan_note = (
                f"\n\n({' + '.join(pieces)} orphan file"
                f"{'s' if (len(raw_orphans) + len(trans_orphans)) != 1 else ''} "
                "had no origins record — these will be moved to the "
                "best-guess matching workspace or to the Library's "
                "parent folder.)"
            )
        msg.setText(
            f"Which category do you want to restore to the original location?\n\n"
            f"  \u2022 Raw sources in Library/Raw: {raw_count}\n"
            f"  \u2022 Translated EPUBs in Library/Translated: {trans_count}"
            f"{orphan_note}"
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
        skipped_undo = 0
        errors: list[str] = []
        # Undo also relocates files — track ``(old_lib_path, orig_path)``
        # pairs so we can emit :attr:`files_reorganized` at the end,
        # mirroring the Organize path. The translator GUI uses this to
        # rewrite any stale ``Library/Raw\x.epub`` path it's still
        # holding back to the restored original location.
        path_moves: list[tuple[str, str]] = []

        # Pre-scan for restore collisions: any library file whose
        # original location already has a file (different contents or
        # a replacement). Prompt once for a policy applied across the
        # whole Undo batch so Windows' ``shutil.move`` doesn't hard-fail
        # silently on name conflicts.
        undo_collisions: list[tuple[str, str]] = []
        if restore_raw:
            raw_dir_pre = get_library_raw_dir()
            for lib_name, orig_path in raw_map.items():
                lib_file = os.path.join(raw_dir_pre, lib_name)
                if (os.path.isfile(lib_file) and os.path.isfile(orig_path)
                        and os.path.abspath(lib_file) != os.path.abspath(orig_path)):
                    undo_collisions.append((lib_file, orig_path))
        if restore_trans:
            trans_dir_pre = get_library_translated_dir()
            for lib_name, orig_path in trans_map.items():
                lib_file = os.path.join(trans_dir_pre, lib_name)
                if (os.path.isfile(lib_file) and os.path.isfile(orig_path)
                        and os.path.abspath(lib_file) != os.path.abspath(orig_path)):
                    undo_collisions.append((lib_file, orig_path))
        undo_policy = "keep_both"
        if undo_collisions:
            undo_policy = self._prompt_duplicate_policy(
                undo_collisions, "their original location")
            if undo_policy is None:
                # User cancelled the undo entirely.
                return
        colliding_orig = {
            os.path.normcase(os.path.normpath(os.path.abspath(o)))
            for _l, o in undo_collisions
        }

        def _resolve_undo_dest(orig_path: str) -> str | None:
            """Pick a destination under *undo_policy* for a restored file.

            Returns ``None`` when the restore should be skipped entirely.
            """
            key = os.path.normcase(os.path.normpath(
                os.path.abspath(orig_path)))
            if key not in colliding_orig:
                return orig_path
            if undo_policy == "skip":
                return None
            if undo_policy == "replace":
                try:
                    if os.path.isfile(orig_path):
                        os.remove(orig_path)
                except OSError as rm_exc:
                    logger.debug("Replace-on-undo remove failed: %s", rm_exc)
                    # Fall through to keep_both.
                else:
                    return orig_path
            # keep_both (or replace fallback): counter-suffix in the
            # original's parent directory like Explorer does.
            parent = os.path.dirname(orig_path) or "."
            base = os.path.basename(orig_path)
            stem, ext = os.path.splitext(base)
            counter = 2
            while True:
                cand = os.path.join(parent, f"{stem} ({counter}){ext}")
                if not os.path.isfile(cand):
                    return cand
                counter += 1

        if restore_raw and raw_map:
            raw_dir = get_library_raw_dir()
            remaining = {}
            for lib_name, orig_path in raw_map.items():
                lib_file = os.path.join(raw_dir, lib_name)
                if not os.path.isfile(lib_file):
                    errors.append(f"raw:{lib_name}: not found in Library/Raw")
                    continue
                dest_path = _resolve_undo_dest(orig_path)
                if dest_path is None:  # policy = skip
                    remaining[lib_name] = orig_path
                    skipped_undo += 1
                    continue
                try:
                    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
                except OSError:
                    pass
                try:
                    shutil.move(lib_file, dest_path)
                    restored_raw += 1
                    path_moves.append(
                        (os.path.abspath(lib_file),
                         os.path.abspath(dest_path)))
                    # Any output folders whose source_epub.txt still points
                    # at the library copy get rewritten to where the raw
                    # actually landed (``dest_path`` — may be the original
                    # location or a ``(2)``-suffixed sibling when Keep Both
                    # was chosen on a collision).
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
                                                f.write(dest_path)
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
                dest_path = _resolve_undo_dest(orig_path)
                if dest_path is None:  # policy = skip
                    remaining[lib_name] = orig_path
                    skipped_undo += 1
                    continue
                try:
                    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
                except OSError:
                    pass
                try:
                    shutil.move(lib_file, dest_path)
                    restored_trans += 1
                    path_moves.append(
                        (os.path.abspath(lib_file),
                         os.path.abspath(dest_path)))
                    # Re-add the restored file to the translated-
                    # inputs registry ONLY when the restore target
                    # is NOT inside an active output folder.
                    #
                    # Rationale: the translated-inputs registry is
                    # for orphan compiled EPUBs the user dropped
                    # onto the Completed tab from outside any
                    # workspace. When Undo restores a file BACK
                    # INTO an output folder (the typical
                    # post-Organize case), the workspace's own
                    # scan already surfaces the book on the
                    # appropriate tab via the output-folder row —
                    # re-registering would add a SECOND card on
                    # the Completed tab alongside the workspace's
                    # In Progress card, producing the duplicate
                    # the user just hit.
                    dest_parent = os.path.dirname(dest_path)
                    is_workspace_restore = bool(
                        dest_parent
                        and os.path.isfile(os.path.join(
                            dest_parent, "translation_progress.json"))
                    )
                    if not is_workspace_restore:
                        try:
                            record_library_translated_input(dest_path)
                        except Exception:
                            logger.debug(
                                "Failed to re-register restored translated path %s",
                                dest_path,
                            )
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
        if skipped_undo:
            summary_parts.append(
                f"Skipped {skipped_undo} duplicate"
                f"{'s' if skipped_undo != 1 else ''} at the original location."
            )
        summary = "\n".join(summary_parts) or "Nothing was restored."
        if errors:
            summary += (f"\n\n{len(errors)} error"
                        f"{'s' if len(errors) != 1 else ''}:\n"
                        + "\n".join(errors[:5]))
        # Notify listeners (the translator GUI) before the summary
        # modal so stale input paths get rewritten to the restored
        # originals BEFORE the user can click Run again.
        if path_moves:
            try:
                self.files_reorganized.emit(list(path_moves))
            except Exception:
                logger.debug("files_reorganized emit failed: %s",
                             traceback.format_exc())
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
        if self._format_filter != FORMAT_ALL:
            books = [
                b for b in books
                if self._format_of_book(b) == self._format_filter
            ]
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
                    bool(b.get("missing_raw_file", False)),
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

    @staticmethod
    def _card_signature(book: dict) -> tuple:
        """Return a hashable signature of the card-rendering inputs.

        Used by :meth:`_populate_grid_common` to decide whether a
        cached :class:`_BookCard` for a given path is still valid,
        or needs to be rebuilt because the underlying book changed
        (progress advanced, missing-raw badge appeared, etc.).

        Only fields that :class:`_BookCard.__init__` actually reads
        are included — a stable signature means filter toggles can
        reuse the existing widget without a :func:`_fit_title_text`
        shrink loop or a fresh :class:`_CoverLoader` thread per card.
        """
        return (
            book.get("path", "") or "",
            book.get("name", "") or "",
            int(book.get("completed_chapters", 0) or 0),
            int(book.get("total_chapters", 0) or 0),
            int(book.get("failed_chapters", 0) or 0),
            int(book.get("pending_chapters", 0) or 0),
            str(book.get("translation_state", "") or ""),
            bool(book.get("is_in_progress", False)),
            bool(book.get("missing_raw_file", False)),
            bool(book.get("has_compiled_output", False)),
            str(book.get("workspace_kind", "") or ""),
            str(book.get("type", "") or ""),
            float(book.get("mtime", 0) or 0),
            int(book.get("size", 0) or 0),
            len(book.get("compiled_conflicts") or []),
        )

    def _populate_grid_common(
        self,
        books: list[dict],
        grid_layout: QGridLayout,
        card_list: list[_BookCard],
        count_label: QLabel,
        empty_label: QLabel,
        count_word: str,
        selected_paths: set[str] | None = None,
        card_cache: dict | None = None,
        full_books: list[dict] | None = None,
    ):
        """Shared render pipeline used by both tabs.

        Cards are cached per path in *card_cache* (one dict per tab,
        owned by the dialog). Each entry maps ``path → (card,
        signature, preset_key, show_raw_title)`` so the next call can
        tell at a glance whether the cached widget is still valid.
        Valid cards are detached from the grid (not deleted), re-laid
        out in the new order, and re-shown. Only new / stale entries
        pay the full ``_BookCard`` construction cost (title-fit loop +
        cover-loader thread spawn).

        *books* is the list that actually renders (already sorted /
        search-filtered / format-filtered). *full_books* is the raw
        scan result for this tab — used to distinguish "filtered
        out" from "genuinely gone" so a Format chip toggle never
        deletes cards it's about to need again on the next click.
        Callers that don't pass *full_books* fall back to the
        previous behaviour (any path not in *books* is treated as
        gone).
        """
        selected_paths = selected_paths if selected_paths is not None else set()
        card_cache = card_cache if card_cache is not None else {}
        visible_paths = {b.get("path", "") for b in books}
        # "Known" paths for the underlying data. When the caller
        # hands us the full unfiltered list we use it; otherwise we
        # assume ``books`` IS the full set (back-compat behaviour).
        full_paths = (
            {b.get("path", "") for b in full_books}
            if full_books is not None else visible_paths)
        # Prune the selection set of paths that no longer exist in
        # the scan result (NOT the filtered view) so switching
        # Format chips doesn't silently wipe a multi-selection that
        # only some of whose cards match the current filter.
        selected_paths &= full_paths

        # Detach every cached card from the grid before we rebuild it.
        # Detaching (``setParent(None)``) is cheap and preserves the
        # widget for reuse; deleting is what makes filter toggles
        # slow, so we only delete cards whose path is gone from the
        # underlying scan result — filter misses stay in the cache
        # so the next toggle can re-use them without another
        # ``_BookCard`` constructor pass.
        for path, cached in list(card_cache.items()):
            cached_card = cached[0]
            try:
                grid_layout.removeWidget(cached_card)
                cached_card.setParent(None)
                cached_card.hide()
            except Exception:
                pass

        for stale_path in [p for p in card_cache if p not in full_paths]:
            try:
                cached_card = card_cache[stale_path][0]
                cached_card.setParent(None)
                cached_card.deleteLater()
            except Exception:
                pass
            del card_cache[stale_path]

        # Also clear stragglers the grid might still hold (e.g.
        # widgets we never tracked, or layout items left over from
        # a size-preset change).
        while grid_layout.count():
            item = grid_layout.takeAt(0)
            w = item.widget()
            if w and w not in (c[0] for c in card_cache.values()):
                w.setParent(None)

        card_list.clear()

        if not books:
            empty_label.show()
            count_label.setText("")
            return

        empty_label.hide()
        count_label.setText(
            f"{len(books)} {count_word}{'s' if len(books) != 1 else ''}"
        )

        preset = _SIZE_PRESETS[self._card_size]
        preset_card_w = preset["card_w"]
        spacing = preset["spacing"]
        grid_layout.setHorizontalSpacing(spacing)
        grid_layout.setVerticalSpacing(spacing + 2)
        # Prefer the grid container's own width — that's exactly the
        # surface cards lay out on. Falling back to the dialog width
        # minus a fudge factor covers the first paint when the grid
        # hasn't been sized yet.
        grid_widget = grid_layout.parentWidget()
        try:
            viewport_w = int(grid_widget.width()) if grid_widget else 0
        except Exception:
            viewport_w = 0
        if viewport_w <= 0:
            viewport_w = max(0, self.width() - 40)
        # Reserve room for the vertical scrollbar. The scroll area's
        # stylesheet fixes it to an 8 px track and we need to subtract
        # it ahead of time because the scrollbar only pops in once the
        # grid's total height overflows the viewport — without the
        # reserve the rightmost column gets clipped the instant the
        # scrollbar appears.
        _SCROLLBAR_RESERVE = 8
        viewport_w = max(0, viewport_w - _SCROLLBAR_RESERVE)
        cols = max(1, (viewport_w + spacing) // (preset_card_w + spacing))
        # Redistribute the leftover horizontal space across the cards
        # themselves so the grid fills the viewport instead of leaving
        # a dead column to the right. Each card grows by up to
        # ``(viewport_w - cols*preset_card_w - (cols-1)*spacing) //
        # cols`` pixels, never shrinks below the preset minimum, and
        # caps a few pixels tight of the raw math to stay inside the
        # viewport even when Qt's QGridLayout rounds fractional widths
        # up. The resulting ``card_w`` is propagated through a shallow
        # preset copy so :class:`_BookCard` renders at the new width
        # (cover + title labels + fitted title font all derive from
        # ``preset['card_w']``).
        inner_w = max(preset_card_w * cols,
                      viewport_w - (cols - 1) * spacing)
        card_w = max(preset_card_w, inner_w // cols)
        if card_w != preset_card_w:
            preset = dict(preset)
            preset["card_w"] = card_w
        # Include the effective card width in the cache key so a
        # window resize that changes the expansion factor forces
        # dependent cards to rebuild with the new width. Otherwise
        # the cache would keep serving cards sized for the previous
        # viewport and the new space would re-appear as a dead column.
        preset_key = (self._card_size, card_w)

        show_raw_title = bool(getattr(self, "_show_raw_titles", False))
        for idx, book in enumerate(books):
            path = book.get("path", "") or ""
            sig = self._card_signature(book)
            cached = card_cache.get(path)
            reuse = bool(
                cached
                and cached[1] == sig
                and cached[2] == preset_key
                and cached[3] == show_raw_title
            )
            if reuse:
                card = cached[0]
                # Always refresh the card's backing book dict, even
                # when the signature matched and we're reusing the
                # widget. :meth:`_card_signature` deliberately covers
                # only the fields that affect rendering \u2014 so a
                # late-resolved ``raw_source_path`` (e.g. the next
                # scan finally matched the workspace to a raw in
                # ``Library/Raw`` / the registry) wouldn't shift the
                # signature and the cached widget would keep its
                # OLD book dict forever. That bit the context menu:
                # Load for translation / Reveal source file / Clear
                # saved raw link all read ``card.book.get
                # (\"raw_source_path\")`` directly, so they missed
                # while the warning badge (driven by
                # ``missing_raw_file``) already reflected the fresh
                # resolution. Re-pointing ``card.book`` at the new
                # dict here keeps every downstream consumer in
                # sync without paying the widget-rebuild cost.
                try:
                    card.book = book
                except Exception:
                    pass
            else:
                # Either no cached widget, a book field drifted, or
                # the card-size / raw-titles toggle changed —
                # rebuild and spawn a fresh cover loader.
                if cached:
                    try:
                        cached[0].setParent(None)
                        cached[0].deleteLater()
                    except Exception:
                        pass
                card = _BookCard(
                    book, preset=preset, show_raw_title=show_raw_title)
                card.clicked.connect(self._on_card_clicked)
                card.context_menu_requested.connect(self._show_context_menu)
                card.select_requested.connect(self._on_card_select_requested)
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
            card_cache[path] = (card, sig, preset_key, show_raw_title)
            card.set_selected(path in selected_paths)
            card_list.append(card)
            row, col = divmod(idx, cols)
            grid_layout.addWidget(card, row, col, Qt.AlignTop | Qt.AlignLeft)
            card.show()

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
        if not hasattr(self, "_ip_card_cache"):
            self._ip_card_cache: dict = {}
        self._populate_grid_common(
            books, self._ip_grid_layout, self._ip_cards,
            self._ip_count_label, self._ip_empty_label, "novel",
            selected_paths=self._selected_paths_ip,
            card_cache=self._ip_card_cache,
            # Pass the full unfiltered scan result so the cache's
            # stale-entry sweep only removes cards whose underlying
            # book is genuinely gone. Filter misses stay in the
            # cache, so toggling Format / search is an O(visible)
            # detach + re-add pass instead of a full rebuild.
            full_books=self._in_progress_books,
        )

    def _populate_completed(self, books: list[dict]):
        if not hasattr(self, "_comp_card_cache"):
            self._comp_card_cache: dict = {}
        self._populate_grid_common(
            books, self._comp_grid_layout, self._comp_cards,
            self._comp_count_label, self._comp_empty_label, "book",
            selected_paths=self._selected_paths_comp,
            card_cache=self._comp_card_cache,
            full_books=self._completed_books,
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
        # Pre-resolve ``raw_source_path`` on every selected book
        # BEFORE any gate runs. :func:`_resolve_book_source_file`
        # caches the result on the book dict via the same
        # ``_find_raw_source_for_library_epub`` fallback that Book
        # Details uses, so Load / Reveal / Clear all see the same
        # resolved path. Without this, a library-filed card whose
        # scan left ``raw_source_path`` empty would show Reveal
        # (the resolver's first call) while Load and Clear silently
        # missed \u2014 exactly the \"Book Details finds it fine,
        # context menu can't\" inconsistency the user hit.
        for _b in selected_books:
            try:
                _resolve_book_source_file(_b)
            except Exception:
                logger.debug(
                    "Context-menu raw-source prefetch failed: %s",
                    traceback.format_exc())
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
            # Snapshot ``selected_books`` so
            # :meth:`_ensure_output_override_matches` can consult each
            # book's ``output_folder`` before the emit \u2014 without
            # the dicts we'd only have paths and no way to check
            # whether any card's workspace lives under a mismatched
            # output root.
            load_books_snapshot = list(selected_books)
            load_action.triggered.connect(
                lambda: self._load_multi_for_translation(
                    raw_candidates, books=load_books_snapshot))
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
        # Reveal the raw source file \u2014 identical to the Book Details
        # \U0001f517 button. Resolves the same way (raw_source_path, then
        # origins lookup for Library/Translated entries, then book path).
        # The action is only added when a real source file exists on
        # disk; when it can't be resolved (e.g. compiled EPUB with no
        # matching raw) we omit it rather than surface a dead entry.
        source_target = _resolve_book_source_file(book)
        if source_target and os.path.isfile(source_target):
            source_action = menu.addAction("\U0001f517  Reveal source file")
            source_action.setToolTip(
                f"Reveal source file:\n{source_target}"
            )
            source_action.triggered.connect(
                lambda: _open_folder_in_explorer(source_target))
        # Reveal the compiled / translated EPUB itself. Resolution
        # lives in :func:`_resolve_book_translated_file` so this menu
        # action and the Book Details 📕 button share one code path.
        # Omitted entirely when no translated EPUB exists on disk.
        translation_target = _resolve_book_translated_file(book)
        if translation_target and os.path.isfile(translation_target):
            translation_action = menu.addAction(
                "\U0001f4d5  Reveal Translated File")
            translation_action.setToolTip(
                f"Reveal translated file:\n{translation_target}"
            )
            translation_action.triggered.connect(
                lambda: _open_folder_in_explorer(translation_target))
        menu.addSeparator()
        copy_path_action = menu.addAction("\U0001f4cb  Copy Path")
        copy_path_action.triggered.connect(lambda: QApplication.clipboard().setText(book["path"]))
        # "Clear saved raw link" \u2014 lets the user reset a bad
        # Scan-for-Raw pairing (e.g. an EPUB workspace that got
        # auto-paired to a ``.pdf`` via an earlier broken match).
        # Deletes the ``source_epub.txt`` sidecar and unregisters
        # the stale raw from the raw-inputs registry so the next
        # library scan re-derives ``workspace_kind`` from the real
        # folder contents. Only offered for workspace-backed cards
        # whose sidecar actually exists on disk \u2014 a card without
        # a saved link has nothing to clear. Sits right above Delete
        # because it's a "corrective" action in the same destructive-
        # but-recoverable family.
        clear_targets = [
            b for b in selected_books
            if self._card_has_saved_raw_link(b)
        ]
        if clear_targets:
            if len(clear_targets) > 1:
                clear_label = (
                    f"\u2702\ufe0f  Clear saved raw link for "
                    f"{len(clear_targets)} items")
                clear_tip = (
                    "Remove the saved raw-source pointer "
                    "(source_epub.txt) from each selected "
                    "workspace. Useful when a previous auto-scan "
                    "matched the wrong file \u2014 the next library "
                    "scan will re-detect the workspace kind from "
                    "the folder contents.")
            else:
                clear_label = "\u2702\ufe0f  Clear saved raw link"
                clear_tip = (
                    "Remove this workspace's saved raw-source "
                    "pointer (source_epub.txt). Use this if the "
                    "auto-scan matched a .pdf to an EPUB (or any "
                    "other wrong-kind pairing) \u2014 the library "
                    "will re-scan with the real folder kind.")
            clear_action = menu.addAction(clear_label)
            clear_action.setToolTip(clear_tip)
            clear_action.triggered.connect(
                lambda: self._clear_saved_raw_link(clear_targets))
        # Delete \u2014 tab-specific semantics:
        #   * In Progress \u2192 delete the output folder (recursive)
        #   * Completed   \u2192 delete the .epub / compiled file
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

    @staticmethod
    def _raw_is_in_library_raw(raw_src: str) -> bool:
        """True when *raw_src* lives directly inside ``Library/Raw``.

        A raw sitting in ``Library/Raw`` is resolved by the scanner
        via the implicit ``Library/Raw/<folder_name>.<ext>`` filename
        pattern (route 2 of :func:`_find_raw_source_for_folder`),
        which is NOT a \"saved link\" the user can clear from the
        UI \u2014 the file itself is the match. Detecting that case
        here lets the Clear action hide itself and refuse to run
        for those cards, so the user isn't shown a no-op.
        """
        if not raw_src:
            return False
        try:
            raw_parent = os.path.normcase(os.path.normpath(
                os.path.abspath(os.path.dirname(raw_src))))
            lib_raw = os.path.normcase(os.path.normpath(
                os.path.abspath(get_library_raw_dir())))
        except (TypeError, ValueError, OSError):
            return False
        return bool(lib_raw) and raw_parent == lib_raw

    @staticmethod
    def _library_raw_match_for_book(book: dict) -> str:
        """Return a ``Library/Raw/<folder_name>.<ext>`` file path if
        one exists on disk for this workspace, else ``""``.

        Mirrors route 2 of :func:`_find_raw_source_for_folder` so
        the Clear gate can detect when the card is effectively
        Library/Raw-backed EVEN IF its cached ``raw_source_path``
        points elsewhere (e.g. the sidecar was written before the
        raw was moved into Library/Raw). When this returns a hit,
        the scanner's next pass WILL resolve the raw via route 2
        regardless of sidecar / registry state, so clearing would
        be a no-op.
        """
        if not isinstance(book, dict):
            return ""
        folder_name = book.get("folder_name") or ""
        if not folder_name:
            ws_folder = _resolve_book_output_folder(book)
            if ws_folder:
                folder_name = os.path.basename(
                    os.path.normpath(ws_folder))
        if not folder_name:
            return ""
        raw_dir = get_library_raw_dir()
        for ext in (".epub", ".txt", ".pdf", ".html"):
            candidate = os.path.join(raw_dir, folder_name + ext)
            if os.path.isfile(candidate):
                return candidate
        return ""

    @staticmethod
    def _card_has_saved_raw_link(book: dict) -> bool:
        """Return True when the card has something to clear.

        A card has a \"saved raw link\" when ANY of the following
        surface the raw source for it:

          1. ``<workspace>/source_epub.txt`` exists on disk
             (written by the translator / Scan-for-Raw).
          2. ``book['raw_source_path']`` is listed in
             ``library_raw_inputs.txt`` (registered through the
             translator run, Import, or Scan-for-Raw's Apply pass).

        Route 2 of :func:`_find_raw_source_for_folder` \u2014 the
        implicit ``Library/Raw/<folder_name>.ext`` filename-pattern
        match \u2014 is NOT considered here because there's
        nothing to \"clear\" (the file itself is the match
        source; removing it would require moving / renaming it).
        Cards whose raw is resolvable via that pattern are
        skipped entirely \u2014 including cards whose cached
        ``raw_source_path`` still points at the pre-move location
        but whose workspace folder name would NOW match a file in
        ``Library/Raw``. The scanner will re-resolve via route 2
        on the next pass regardless of what a sidecar / registry
        entry says, so clearing would read as broken.

        Workspace resolution goes through
        :func:`_resolve_book_output_folder` so library-filed cards
        whose compiled EPUB was organized into ``Library/Translated``
        still resolve to their originating workspace via the
        origins registry.
        """
        if not isinstance(book, dict):
            return False
        raw_src = book.get("raw_source_path") or ""
        # Library/Raw-backed raws don't expose a clearable link
        # \u2014 whether the card's cached ``raw_source_path``
        # points there directly, or a matching file sitting in
        # ``Library/Raw`` is waiting to be picked up by route 2
        # on the next scan.
        if EpubLibraryDialog._raw_is_in_library_raw(raw_src):
            return False
        if EpubLibraryDialog._library_raw_match_for_book(book):
            return False
        # 1. Sidecar on disk.
        ws_folder = _resolve_book_output_folder(book)
        if ws_folder and os.path.isdir(ws_folder):
            sidecar = os.path.join(ws_folder, "source_epub.txt")
            if os.path.isfile(sidecar):
                return True
        # 2. Registry-backed link.
        if not raw_src:
            return False
        try:
            raw_key = os.path.normcase(os.path.normpath(
                os.path.abspath(raw_src)))
        except (TypeError, ValueError):
            return False
        try:
            for p in load_library_raw_inputs():
                if not p:
                    continue
                try:
                    reg_key = os.path.normcase(os.path.normpath(
                        os.path.abspath(p)))
                except Exception:
                    continue
                if reg_key == raw_key:
                    return True
        except Exception:
            pass
        return False

    def _clear_saved_raw_link(self, books: list):
        """Delete the ``source_epub.txt`` sidecar(s) after confirmation.

        Also unregisters the previously-pointed-at raw path from the
        raw-inputs registry when it's no longer referenced by any
        other workspace, so the Scan-for-Raw dialog won't try to
        re-apply the same stale pairing on its next run. On success
        the library is reloaded so the scanner re-derives
        ``workspace_kind`` from the real folder contents (which
        fixes the \"EPUB workspace wearing a PDF badge\" symptom).
        """
        if not books:
            return
        # De-dup by workspace folder and snapshot the current raw
        # pointer so we can optionally unregister it afterwards.
        # ``_resolve_book_output_folder`` is used here (not a raw
        # ``book['output_folder']`` lookup) so library-filed cards
        # whose compiled EPUB was organized into ``Library/Translated``
        # still resolve to their originating workspace via the
        # origins registry \u2014 otherwise the Clear action silently
        # skipped them, which the user observed as an inconsistency.
        #
        # Each target carries the workspace folder (if any), the raw
        # pointer recorded on disk (sidecar content OR
        # ``raw_source_path`` as a fallback for registry-only
        # entries), and a flag telling the deletion pass whether a
        # sidecar actually existed on disk. Cards whose raw was
        # resolved only via the registry have no sidecar to delete
        # but still benefit from unregistering the raw so the next
        # scan can re-derive the match cleanly.
        targets: list[tuple[str, str, dict, bool]] = []
        seen_ws: set[str] = set()
        seen_registry: set[str] = set()
        for b in books:
            # Skip cards whose raw lives in ``Library/Raw`` \u2014 the
            # filename-pattern route resolves those regardless of
            # any sidecar / registry state, so \"clearing\" would be
            # a no-op (the scanner would just re-resolve the raw
            # via route 2 on the next scan). The second check
            # covers cards whose cached ``raw_source_path`` is
            # stale but whose workspace folder name now matches a
            # file in ``Library/Raw`` (post-Organize or manual
            # copy). Without it a leftover sidecar would keep the
            # action visible even though clearing it would have
            # no user-visible effect.
            raw_src_full = b.get("raw_source_path") or ""
            if self._raw_is_in_library_raw(raw_src_full):
                continue
            if self._library_raw_match_for_book(b):
                continue
            ws_folder = _resolve_book_output_folder(b)
            sidecar = ""
            sidecar_raw = ""
            if ws_folder and os.path.isdir(ws_folder):
                cand = os.path.join(ws_folder, "source_epub.txt")
                if os.path.isfile(cand):
                    sidecar = cand
                    try:
                        with open(sidecar, "r", encoding="utf-8") as fh:
                            sidecar_raw = fh.read().strip()
                    except OSError:
                        sidecar_raw = ""
            if sidecar:
                ws_key = os.path.normcase(os.path.normpath(
                    os.path.abspath(ws_folder)))
                if ws_key in seen_ws:
                    continue
                seen_ws.add(ws_key)
                targets.append((ws_folder, sidecar_raw, b, True))
                continue
            # No sidecar \u2014 registry-only link?
            raw_src = raw_src_full
            if not raw_src:
                continue
            try:
                raw_key = os.path.normcase(os.path.normpath(
                    os.path.abspath(raw_src)))
            except (TypeError, ValueError):
                continue
            in_registry = False
            try:
                for p in load_library_raw_inputs():
                    if not p:
                        continue
                    try:
                        reg_key = os.path.normcase(os.path.normpath(
                            os.path.abspath(p)))
                    except Exception:
                        continue
                    if reg_key == raw_key:
                        in_registry = True
                        break
            except Exception:
                in_registry = False
            if not in_registry:
                continue
            if raw_key in seen_registry:
                continue
            seen_registry.add(raw_key)
            # ``ws_folder`` may be empty here \u2014 that's fine, we
            # just skip sidecar deletion and only unregister.
            targets.append((ws_folder or "", raw_src, b, False))
        if not targets:
            return
        # Confirmation prompt: list the affected workspace(s) and the
        # raw path each is currently pointing at. The explanatory
        # footer varies based on whether we're deleting sidecars,
        # unregistering raws, or both \u2014 the user shouldn't see
        # \"only source_epub.txt is deleted\" when a registry-only
        # entry is being cleared.
        preview_lines = []
        for ws_folder, old_raw, b, _had_sidecar in targets[:6]:
            label = (b.get("folder_name")
                     or os.path.basename(ws_folder)
                     or b.get("name") or "")
            if old_raw:
                preview_lines.append(
                    f"  \u2022 {label}\n      \u2192 {old_raw}")
            else:
                preview_lines.append(f"  \u2022 {label}")
        if len(targets) > 6:
            preview_lines.append(f"  \u2026 and {len(targets) - 6} more.")
        has_sidecars = any(t[3] for t in targets)
        has_registry_only = any(not t[3] for t in targets)
        if has_sidecars and has_registry_only:
            footer = (
                "The workspace folders are left untouched \u2014 "
                "``source_epub.txt`` is deleted where present, and "
                "the raw path is unregistered from "
                "``library_raw_inputs.txt``. The next library scan "
                "will re-detect the workspace kind from the folder "
                "contents.")
        elif has_sidecars:
            footer = (
                "The workspace folders are left untouched \u2014 "
                "only ``source_epub.txt`` is deleted. The next "
                "library scan will re-detect the workspace kind "
                "from the folder contents.")
        else:
            footer = (
                "No ``source_epub.txt`` sidecar exists for these "
                "cards \u2014 the raw path will be unregistered from "
                "``library_raw_inputs.txt`` instead. The next "
                "library scan will re-detect the workspace kind "
                "from the folder contents.")
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Clear saved raw link")
        if len(targets) == 1:
            msg.setText(
                "Remove the saved raw-source pointer for this "
                "workspace?\n\n"
                + "\n".join(preview_lines)
                + "\n\n" + footer
            )
        else:
            msg.setText(
                f"Remove the saved raw-source pointer for "
                f"{len(targets)} workspace"
                f"{'s' if len(targets) != 1 else ''}?\n\n"
                + "\n".join(preview_lines)
                + "\n\n" + footer
            )
        msg.setStandardButtons(
            QMessageBox.Yes | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Yes)
        if msg.exec() != QMessageBox.Yes:
            return
        cleared = 0
        for ws_folder, old_raw, _b, had_sidecar in targets:
            if had_sidecar and ws_folder:
                sidecar = os.path.join(ws_folder, "source_epub.txt")
                try:
                    os.remove(sidecar)
                    cleared += 1
                except OSError as exc:
                    logger.debug(
                        "Clear saved raw link failed for %s: %s",
                        sidecar, exc)
                    continue
            else:
                # Registry-only link \u2014 no sidecar to delete, but
                # unregistering the raw still counts as \"cleared\".
                cleared += 1
            if not old_raw:
                continue
            if not had_sidecar:
                # Registry-only clear: the registry entry IS the
                # link for this card. Remove it unconditionally,
                # even if other workspaces reference the same raw
                # via their own ``source_epub.txt`` sidecars \u2014
                # those keep resolving through route 1 and don't
                # depend on the registry. The previous code ran
                # the same \"still referenced\" sweep used for
                # sidecar-based clears, which would see those
                # sidecars and refuse to unregister, leaving this
                # card's link stubbornly in place.
                try:
                    remove_library_raw_input(old_raw)
                except Exception:
                    logger.debug(
                        "Registry-only unregister failed: %s",
                        traceback.format_exc())
                continue
            # Sidecar-based clear: the sweep protects against
            # orphaning a raw that another workspace still needs
            # as a registry fallback. Only drop the registry
            # entry when no OTHER workspace points at it.
            try:
                still_referenced = False
                roots_checked: set[str] = set()
                for root in _resolve_output_roots(self._config):
                    root_key = os.path.normcase(os.path.normpath(
                        os.path.abspath(root)))
                    if root_key in roots_checked:
                        continue
                    roots_checked.add(root_key)
                    try:
                        for entry in os.scandir(root):
                            if not entry.is_dir(follow_symlinks=False):
                                continue
                            other_sidecar = os.path.join(
                                entry.path, "source_epub.txt")
                            if not os.path.isfile(other_sidecar):
                                continue
                            try:
                                with open(other_sidecar, "r",
                                          encoding="utf-8") as fh:
                                    val = fh.read().strip()
                            except OSError:
                                continue
                            if not val:
                                continue
                            try:
                                if (os.path.normcase(
                                        os.path.normpath(
                                            os.path.abspath(val)))
                                        == os.path.normcase(
                                            os.path.normpath(
                                                os.path.abspath(old_raw)))):
                                    still_referenced = True
                                    break
                            except Exception:
                                continue
                    except (PermissionError, OSError):
                        continue
                    if still_referenced:
                        break
                if not still_referenced:
                    remove_library_raw_input(old_raw)
            except Exception:
                logger.debug(
                    "Raw-input unregister sweep failed: %s",
                    traceback.format_exc())
        if cleared:
            # Reload so the scanner re-classifies workspace_kind from
            # the folder contents \u2014 that's what flips the card
            # badge back from PDF to EPUB for mis-paired entries.
            self._load_books()
        else:
            QMessageBox.warning(
                self, "Clear saved raw link",
                "None of the selected workspaces could be updated. "
                "Check that the folders still exist on disk.")

    def _delete_books_prompt(self, books: list):
        """Confirm + delete the backing file / folder for each book.

        Target resolution:

          * **Library entries** (``in_library=True``) — delete
            ``book['path']`` as a single file. These live in
            ``Library/Translated`` and don't own a surrounding workspace.
          * **Output-folder cards** (``type == "in_progress"`` OR any
            completed card with a resolvable ``output_folder``) —
            delete the folder recursively so every sibling artefact
            goes with it.
          * Everything else — delete ``book['path']`` as a plain file.

        Confirmation policy:

          * When every selected card is "Not Started" (no translated
            chapters on disk yet), a plain Yes / Cancel dialog is
            enough — there's no meaningful work to lose.
          * Otherwise the user has to type the word "Halgakos"
            (case-insensitive) into a confirmation field before the
            Delete button enables. This is an intentional speed bump
            because the delete is recursive and unrecoverable: a
            completed translation's ``response_*.html`` files,
            ``translation_progress.json``, ``images/``, and the
            compiled EPUB all go together when the card gets removed.
        """
        if not books:
            return
        # Resolve targets: (label, path, is_folder, book_dict). We keep
        # the book dict so the confirmation dialog can classify each
        # target (Not Started vs. In Progress vs. Completed) and
        # enumerate what's inside a folder target.
        targets: list[tuple[str, str, bool, dict]] = []
        seen_targets: set[str] = set()
        # (book_dict, path) pairs for cards whose backing file lives
        # outside the Library + output-root safe zones. For those we
        # only unregister the tracking-file entry so the flash card
        # disappears — the physical file stays exactly where the
        # user put it. Handled silently: no message box, no summary
        # entry. This is the "Add Translation from Downloads" flow:
        # the Library pointed at the file, the user clicks Delete
        # on the card, and they just want the card gone, not the
        # source EPUB wiped off their drive.
        unregister_cards: list[tuple[dict, str]] = []

        # Safe roots for delete: anything under the Library folder
        # (covers Raw / Translated / registry files) or any configured
        # output root. A path outside ALL of these is considered
        # off-limits — a user-owned file from Downloads or wherever,
        # which ``Delete`` must never touch because Glossarion didn't
        # put it there. Computed once per prompt to keep the queue
        # loop cheap.
        safe_roots: list[str] = []
        try:
            lib_abs = os.path.normcase(os.path.normpath(
                os.path.abspath(get_library_dir())))
            if lib_abs:
                safe_roots.append(lib_abs)
        except Exception:
            logger.debug("Library dir resolve failed: %s",
                         traceback.format_exc())
        try:
            for root in _resolve_output_roots(self._config):
                r = os.path.normcase(os.path.normpath(
                    os.path.abspath(root)))
                if r and r not in safe_roots:
                    safe_roots.append(r)
        except Exception:
            logger.debug("Output roots resolve failed: %s",
                         traceback.format_exc())

        def _is_inside_safe_root(pth: str) -> bool:
            """True when *pth* is inside Library/ or an output root."""
            if not pth or not safe_roots:
                return False
            try:
                key = os.path.normcase(os.path.normpath(
                    os.path.abspath(pth)))
            except Exception:
                return False
            for root in safe_roots:
                if key == root or key.startswith(root + os.sep):
                    return True
            return False

        def _queue(label: str, pth: str, is_folder: bool, book: dict) -> None:
            # Hard safety gate: Delete must never reach outside the
            # Library folder or the configured output roots. A raw
            # EPUB registered in place from Downloads, a stray compiled
            # EPUB dropped onto the Completed tab without Organize,
            # etc. all land here — we route them to the
            # unregister-only path so the flash card disappears but
            # the on-disk file stays intact.
            if not _is_inside_safe_root(pth):
                unregister_cards.append((book, pth))
                return
            key = os.path.normcase(os.path.normpath(os.path.abspath(pth)))
            if key in seen_targets:
                return
            seen_targets.add(key)
            targets.append((label, pth, is_folder, book))

        # Library/Raw absolute path prefix — used to decide whether a
        # card's raw source qualifies for auto-cleanup alongside the
        # workspace. Only raws that actually live inside ``Library/Raw``
        # are deletable; raws anywhere else (Downloads, a user's own
        # folder) are explicitly left untouched so deleting a card
        # never removes files the library didn't put there itself.
        raw_dir_abs = os.path.normcase(os.path.normpath(
            os.path.abspath(get_library_raw_dir())))

        def _queue_library_raw_copy(b: dict) -> None:
            rp = b.get("raw_source_path") or ""
            if not rp or not os.path.isfile(rp):
                return
            rp_parent = os.path.normcase(os.path.normpath(
                os.path.abspath(os.path.dirname(rp))))
            if rp_parent != raw_dir_abs:
                # Raw lives outside Library/Raw — never touch it.
                return
            _queue(os.path.basename(rp), rp, False, b)

        for b in books:
            file_type = b.get("type", "epub")
            in_library = bool(b.get("in_library"))
            output_folder = b.get("output_folder") or ""
            # In-progress cards always delete the folder + (when present)
            # the matching raw in Library/Raw. The raw is only queued
            # when it actually sits inside ``Library/Raw`` so we never
            # wipe a source file that originated from Downloads or any
            # other user directory.
            if file_type == "in_progress":
                folder = output_folder or b.get("path", "") or ""
                if folder and os.path.isdir(folder):
                    _queue(b.get("name") or os.path.basename(folder),
                           folder, True, b)
                _queue_library_raw_copy(b)
                continue
            # Completed but NOT library-filed: the card represents the
            # entire output-folder workspace (compiled file plus any
            # ``response_*``, ``_translated.*``, ``images/``, etc.).
            if (not in_library and output_folder
                    and os.path.isdir(output_folder)):
                _queue(b.get("name") or os.path.basename(output_folder),
                       output_folder, True, b)
                _queue_library_raw_copy(b)
                continue
            # Library entry (or any other loose file-backed card).
            # ``Library/Translated`` cards only own the compiled .epub
            # themselves, but if a paired raw still sits in
            # ``Library/Raw`` we queue it too so one Delete click
            # removes both halves of the library pair. Raws living
            # outside ``Library/Raw`` (e.g. the user's Downloads
            # folder) are left alone by :func:`_queue_library_raw_copy`.
            p = b.get("path", "") or ""
            if p and os.path.isfile(p):
                _queue(b.get("name") or os.path.basename(p), p, False, b)
            _queue_library_raw_copy(b)
        # Perform silent unregistrations for unsafe cards (the "Add
        # Translation from Downloads" flow). These operate directly on
        # the tracking files — no confirmation, no message box, no
        # summary line — since the physical file is left untouched.
        def _unregister_unsafe_cards() -> int:
            removed = 0
            for bk, pth in unregister_cards:
                try:
                    if bk.get("in_library"):
                        # A library-filed EPUB inside Library/Translated
                        # can never reach this branch (it's inside the
                        # safe root), so any ``in_library=True`` card
                        # here is a raw-inputs entry — prune it.
                        remove_library_raw_input(pth)
                    elif bk.get("registered_translated"):
                        remove_library_translated_input(pth)
                    elif bk.get("type") == "in_progress":
                        # Not-started / in-progress card whose raw
                        # source sits outside Library/Raw: drop it
                        # from the raw-inputs registry so the card
                        # disappears on the next scan.
                        remove_library_raw_input(pth)
                    else:
                        # Fallback: try both registries. No-op when
                        # the path isn't in either.
                        remove_library_raw_input(pth)
                        remove_library_translated_input(pth)
                    removed += 1
                    try:
                        self._selected_paths_ip.discard(pth)
                        self._selected_paths_comp.discard(pth)
                    except Exception:
                        pass
                except Exception:
                    logger.debug(
                        "Silent unregister failed for %s: %s",
                        pth, traceback.format_exc())
            return removed

        if not targets:
            # No disk deletes to perform — still honor the user's
            # intent by unregistering any unsafe card(s) so the flash
            # card disappears. Silent: no dialog, just refresh.
            if unregister_cards:
                _unregister_unsafe_cards()
                QTimer.singleShot(0, self._load_books)
                return
            QMessageBox.information(
                self, "Delete",
                "Nothing to delete \u2014 none of the selected cards "
                "point at a file or folder on disk.",
            )
            return

        # Classify the batch. Simple prompt is only allowed when EVERY
        # target came from a "Not Started" card — any in-progress or
        # completed workspace (including a Library/Translated compiled
        # EPUB) in the selection bumps the whole batch into the
        # typed-keyword prompt.
        def _is_not_started(b: dict) -> bool:
            state = (b.get("translation_state") or "").lower()
            if state:
                return state == "not_started"
            # Fallback for rows without an explicit ``translation_state``
            # field: only *in-progress workspace* cards can be
            # "not_started". Library/Translated entries and compiled
            # output cards have ``type`` set to the file kind
            # (``"epub"`` / ``"pdf"`` / …) and represent finished work —
            # they must ALWAYS take the typed-keyword prompt path,
            # otherwise the simple Yes/Cancel dialog would let a
            # finished compiled EPUB (or worse, a shelf-filed one) be
            # deleted with a single click. Library entries don't
            # populate ``completed_chapters`` / ``has_compiled_output``
            # so the old progress-based heuristic fell through to
            # ``True`` for them — hence the explicit type gate here.
            if b.get("type") != "in_progress":
                return False
            if b.get("in_library"):
                return False
            return (int(b.get("completed_chapters", 0) or 0) == 0
                    and not b.get("has_compiled_output", False))

        all_not_started = all(_is_not_started(b) for _l, _p, _f, b in targets)

        if all_not_started:
            confirmed_targets = self._confirm_delete_simple(targets)
        else:
            confirmed_targets = self._confirm_delete_with_keyword(targets)
        if not confirmed_targets:
            return
        # The typed-keyword dialog returns the *filtered* subset the
        # user chose to keep enabled in the checkbox list. The simple
        # dialog always returns the full batch (Not Started cards don't
        # offer per-target opt-out since there's nothing to lose).
        targets = list(confirmed_targets)

        deleted = 0
        errors: list[str] = []
        for label, pth, is_folder, _b in targets:
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

        # Silently unregister any unsafe cards that rode along with
        # this batch. The flash cards disappear on the next scan but
        # the summary doesn't mention them — they're an implementation
        # detail, not a user-facing failure.
        _unregister_unsafe_cards()

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

    # -- Delete-confirmation helpers ----------------------------------------
    # Either keyword unlocks the Delete button in the typed-confirmation
    # dialog. ``"halgakos"`` is the thematic brand-name safeguard;
    # ``"delete"`` is the mundane escape hatch for users who don't want
    # to hunt down the Glossarion mascot. Both are matched after
    # ``.strip().lower()`` so case / surrounding whitespace is forgiven.
    _DELETE_KEYWORDS = ("halgakos", "delete")

    @staticmethod
    def _summarize_folder_contents(folder: str) -> list[str]:
        """Return a bulleted breakdown of artefacts inside *folder*.

        Used to build a detailed "exactly what gets deleted" warning
        for output-folder workspaces. Unknown / unclassified files
        fall into "other files" so the counts always add up.
        """
        counts = {
            "translated chapter HTML files": 0,
            "compiled EPUB files": 0,
            "compiled PDF files": 0,
            "translated text files": 0,
            "translated HTML pages": 0,
            "images": 0,
            "glossary files": 0,
            "progress / history files": 0,
            "other files": 0,
        }
        total_bytes = 0
        try:
            for root, _dirs, files in os.walk(folder):
                for name in files:
                    ln = name.lower()
                    fpath = os.path.join(root, name)
                    try:
                        total_bytes += os.path.getsize(fpath)
                    except OSError:
                        pass
                    if ln.startswith("response_") and ln.endswith(
                            (".html", ".htm", ".xhtml")):
                        counts["translated chapter HTML files"] += 1
                    elif ln.endswith(".epub"):
                        counts["compiled EPUB files"] += 1
                    elif "_translated" in ln and ln.endswith(".pdf"):
                        counts["compiled PDF files"] += 1
                    elif "_translated" in ln and ln.endswith(".txt"):
                        counts["translated text files"] += 1
                    elif "_translated" in ln and ln.endswith(
                            (".html", ".htm", ".xhtml")):
                        counts["translated HTML pages"] += 1
                    elif ln.endswith(
                            (".jpg", ".jpeg", ".png", ".webp",
                             ".gif", ".bmp")):
                        counts["images"] += 1
                    elif "glossary" in ln:
                        counts["glossary files"] += 1
                    elif ln in (
                            "translation_progress.json",
                            "translation_history.json",
                            "metadata.json",
                            "source_epub.txt",
                    ):
                        counts["progress / history files"] += 1
                    else:
                        counts["other files"] += 1
        except OSError:
            return []

        lines = [
            f"    \u00b7 {v} {k}"
            for k, v in counts.items() if v > 0
        ]
        if total_bytes:
            if total_bytes >= 1024 * 1024:
                size_str = f"{total_bytes / (1024 * 1024):.1f} MB"
            else:
                size_str = f"{total_bytes / 1024:.0f} KB"
            lines.append(f"    \u00b7 total on disk: {size_str}")
        return lines

    def _format_delete_detail(
        self, targets: list[tuple[str, str, bool, dict]]
    ) -> str:
        """Build a rich "what will be deleted" block for the dialog."""
        lines: list[str] = []
        for label, pth, is_folder, _book in targets[:10]:
            if is_folder:
                lines.append(f"\u25be  {label}  —  output folder")
                lines.append(f"       {pth}")
                contents = self._summarize_folder_contents(pth)
                if contents:
                    lines.extend(contents)
                else:
                    lines.append("    \u00b7 (folder is empty)")
            else:
                try:
                    size = os.path.getsize(pth)
                    if size >= 1024 * 1024:
                        size_str = f"{size / (1024 * 1024):.1f} MB"
                    else:
                        size_str = f"{size / 1024:.0f} KB"
                except OSError:
                    size_str = "?"
                lines.append(f"\u25be  {label}  —  file ({size_str})")
                lines.append(f"       {pth}")
            lines.append("")
        if len(targets) > 10:
            lines.append(f"\u2026 and {len(targets) - 10} more item(s).")
        return "\n".join(lines).rstrip()

    def _build_target_row_widget(
        self, target: tuple[str, str, bool, dict], checkbox
    ) -> QWidget:
        """Render a single delete target as a checkbox + detail block.

        The checkbox is created by the caller so it can track the
        per-target include/exclude state; this method just owns the
        layout and the label that sits next to it (path, size, and for
        folders the artefact breakdown).
        """
        label, pth, is_folder, _book = target
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 4, 0, 4)
        row_layout.setSpacing(8)

        checkbox.setChecked(True)
        checkbox.setStyleSheet(
            "QCheckBox { color: #e0e0e0; spacing: 8px; }"
            "QCheckBox::indicator { width: 16px; height: 16px; "
            "border: 1px solid #5a9fd4; border-radius: 3px; "
            "background: #1a1a2a; }"
            "QCheckBox::indicator:checked { background: #5a9fd4; "
            "border-color: #5a9fd4; }"
            "QCheckBox::indicator:hover { border-color: #7bb3e0; }"
        )
        checkbox.setToolTip(
            "Uncheck to exclude this item from the delete batch."
        )
        row_layout.addWidget(checkbox, 0, Qt.AlignTop)

        # Text block describing what this target is and what's inside.
        lines: list[str] = []
        if is_folder:
            lines.append(f"{label}  \u2014  output folder")
            lines.append(pth)
            contents = self._summarize_folder_contents(pth)
            if contents:
                lines.extend(contents)
            else:
                lines.append("    \u00b7 (folder is empty)")
        else:
            try:
                size = os.path.getsize(pth)
                if size >= 1024 * 1024:
                    size_str = f"{size / (1024 * 1024):.1f} MB"
                else:
                    size_str = f"{size / 1024:.0f} KB"
            except OSError:
                size_str = "?"
            lines.append(f"{label}  \u2014  file ({size_str})")
            lines.append(pth)

        detail = QLabel("\n".join(lines))
        detail.setTextFormat(Qt.PlainText)
        detail.setWordWrap(True)
        detail.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        detail.setStyleSheet(
            "QLabel { color: #e0e0e0; "
            "font-family: Consolas, Menlo, monospace; font-size: 9pt; }"
        )
        row_layout.addWidget(detail, 1)
        return row

    def _confirm_delete_simple(
        self, targets: list[tuple[str, str, bool, dict]]
    ) -> list[tuple[str, str, bool, dict]] | None:
        """Yes / Cancel confirmation for Not-Started-only batches.

        Not-Started cards have no translated work on disk so the
        simple prompt doesn't bother with per-target checkboxes; the
        whole batch goes or nothing does. Returns the full targets
        list on accept, ``None`` on cancel — matching the shape of
        :meth:`_confirm_delete_with_keyword`.
        """
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Delete")
        preview = self._format_delete_detail(targets)
        msg.setText(
            f"Permanently delete {len(targets)} "
            f"Not Started item{'s' if len(targets) != 1 else ''}?\n\n"
            f"{preview}\n\nThis cannot be undone."
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Cancel)
        if msg.exec() != QMessageBox.Yes:
            return None
        return list(targets)

    def _confirm_delete_with_keyword(
        self, targets: list[tuple[str, str, bool, dict]]
    ) -> list[tuple[str, str, bool, dict]] | None:
        """Typed-keyword confirmation for in-progress / completed cards.

        Shows a modal dialog with a scrollable, per-target checkbox
        list so the user can untick anything they don't actually want
        removed (common case: keep the Library/Raw copy when
        deleting a compiled Library/Translated entry). The Delete
        button is disabled until:

          * at least one checkbox is still ticked, AND
          * the user types :attr:`_DELETE_KEYWORDS` (case-insensitive,
            leading/trailing whitespace forgiven) into the entry field.

        The speed-bump is deliberately harder to bypass than a Yes/No
        because each target wipes real translation work (response
        HTMLs, progress JSON, images, compiled EPUBs).

        Returns the filtered list of targets still checked when the
        user clicked Delete, or ``None`` if the dialog was cancelled.
        """
        from PySide6.QtWidgets import (
            QDialog, QDialogButtonBox, QCheckBox, QFrame,
        )

        dlg = QDialog(self)
        dlg.setWindowTitle("Delete \u2014 confirmation required")
        dlg.setModal(True)
        dlg.setMinimumWidth(620)

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(10)

        headline = QLabel(
            f"\u26a0  Permanent delete \u2014 {len(targets)} "
            f"item{'s' if len(targets) != 1 else ''}"
        )
        headline.setStyleSheet(
            "color: #ff8080; font-size: 13pt; font-weight: bold;"
        )
        layout.addWidget(headline)

        # Count folders vs files in the batch for the subtitle.
        folder_count = sum(1 for _l, _p, f, _b in targets if f)
        file_count = len(targets) - folder_count
        subtitle_bits = []
        if folder_count:
            subtitle_bits.append(
                f"{folder_count} output folder"
                f"{'s' if folder_count != 1 else ''}"
            )
        if file_count:
            subtitle_bits.append(
                f"{file_count} file{'s' if file_count != 1 else ''}"
            )
        subtitle = QLabel(
            "Uncheck anything you want to keep. Everything still "
            "checked below will be removed from disk ("
            + " + ".join(subtitle_bits) + "):"
        )
        subtitle.setStyleSheet("color: #c8cbe0; font-size: 10pt;")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # Scrollable list of per-target rows so arbitrarily large batches
        # (multi-select + auto-queued Library/Raw copies) don't blow the
        # dialog off the bottom of the screen.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { background: #1a1a2a; border: 1px solid #3a3a5e; "
            "border-radius: 4px; }"
            "QScrollBar:vertical { width: 10px; background: #12121e; }"
            "QScrollBar::handle:vertical { background: #3a3a5e; "
            "border-radius: 5px; min-height: 24px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical "
            "{ height: 0; }"
        )
        scroll.setMinimumHeight(220)
        scroll.setMaximumHeight(360)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(10, 6, 10, 6)
        inner_layout.setSpacing(2)

        checkboxes: list[QCheckBox] = []
        for idx, target in enumerate(targets):
            if idx > 0:
                sep = QFrame()
                sep.setFrameShape(QFrame.HLine)
                sep.setStyleSheet("color: #2a2a3e; background: #2a2a3e;")
                sep.setFixedHeight(1)
                inner_layout.addWidget(sep)
            cb = QCheckBox()
            checkboxes.append(cb)
            row_widget = self._build_target_row_widget(target, cb)
            inner_layout.addWidget(row_widget)
        inner_layout.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll)

        warning = QLabel(
            "This removes the checked item(s) permanently \u2014 for "
            "output folders that's recursive: every translated chapter, "
            "progress tracker, image, and compiled EPUB inside goes "
            "with it. This cannot be undone."
        )
        warning.setStyleSheet(
            "color: #ffb347; font-size: 9.5pt; "
            "background: rgba(255, 179, 71, 0.12); "
            "border: 1px solid #ffb347; border-radius: 4px; "
            "padding: 6px 10px;"
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)

        # Typed-keyword prompt. Showing both expected words in the
        # instruction + one of them in the placeholder makes it obvious
        # the user needs to type EXACTLY that — no discovery required.
        pretty = " or ".join(
            f"<b>{kw.capitalize()}</b>" for kw in self._DELETE_KEYWORDS
        )
        instr = QLabel(
            f"Type {pretty} below to unlock the Delete button "
            f"(case-insensitive):"
        )
        instr.setTextFormat(Qt.RichText)
        instr.setStyleSheet("color: #c8cbe0; font-size: 10pt;")
        layout.addWidget(instr)

        entry = QLineEdit()
        entry.setPlaceholderText(self._DELETE_KEYWORDS[0].capitalize())
        entry.setStyleSheet(
            "QLineEdit { background: #1e1e2e; color: #e0e0e0; "
            "border: 1px solid #3a3a5e; border-radius: 4px; "
            "padding: 6px 10px; font-size: 10pt; }"
            "QLineEdit:focus { border-color: #6c63ff; }"
        )
        layout.addWidget(entry)

        btns = QDialogButtonBox(QDialogButtonBox.Cancel)
        delete_btn = QPushButton("\U0001f5d1\ufe0f  Delete")
        delete_btn.setEnabled(False)
        delete_btn.setStyleSheet(
            "QPushButton { background: #c0392b; color: white; border: none; "
            "border-radius: 4px; padding: 6px 18px; font-weight: bold; }"
            "QPushButton:hover:enabled { background: #e74c3c; }"
            "QPushButton:disabled { background: #3a3a3a; color: #888; }"
        )
        btns.addButton(delete_btn, QDialogButtonBox.AcceptRole)
        btns.rejected.connect(dlg.reject)
        delete_btn.clicked.connect(dlg.accept)
        layout.addWidget(btns)

        def _recompute_enabled(*_args) -> None:
            keyword_ok = (
                entry.text().strip().lower() in self._DELETE_KEYWORDS
            )
            any_checked = any(cb.isChecked() for cb in checkboxes)
            delete_btn.setEnabled(keyword_ok and any_checked)

        entry.textChanged.connect(_recompute_enabled)
        for cb in checkboxes:
            cb.toggled.connect(_recompute_enabled)
        entry.setFocus()
        if dlg.exec() != QDialog.Accepted:
            return None
        return [
            targets[i] for i, cb in enumerate(checkboxes) if cb.isChecked()
        ]

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

        # "Reveal Translated File" — matches the card context menu's
        # equivalent action. Uses :func:`_resolve_book_translated_file`
        # so this button and the menu item resolve the same target.
        self._translated_btn = QPushButton("\U0001f4d5")
        self._translated_btn.setProperty("class", "icon-btn")
        self._translated_btn.setToolTip("Reveal translated file")
        self._translated_btn.setCursor(Qt.PointingHandCursor)
        self._translated_btn.clicked.connect(self._reveal_translated)
        self._translated_btn_opacity = QGraphicsOpacityEffect(
            self._translated_btn)
        self._translated_btn_opacity.setOpacity(1.0)
        self._translated_btn.setGraphicsEffect(
            self._translated_btn_opacity)
        actions.addWidget(self._translated_btn)

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
                # Same rationale as the flash-card title: apply the fitted
                # size via ``setFont`` so the rendered font exactly matches
                # what :func:`_fit_title_text` measured. Raw CJK / filename
                # titles otherwise slip past the stylesheet font-size
                # resolver and overflow horizontally without triggering
                # the shrink loop.
                title_font = QFont(v_lbl.font())
                title_font.setPointSizeF(fitted_pt)
                title_font.setBold(True)
                v_lbl.setFont(title_font)
                v_lbl.setText(fitted_text)
                v_lbl.setToolTip(str(val))
                v_lbl.setStyleSheet("color: #e0e0e0; font-weight: bold;")
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
        resolved_trans = self._resolve_translated_file_target()
        trans_ok = bool(resolved_trans) and os.path.isfile(resolved_trans)
        self._translated_btn.setEnabled(trans_ok)
        self._translated_btn.setCursor(
            Qt.PointingHandCursor if trans_ok else Qt.ForbiddenCursor)
        self._translated_btn_opacity.setOpacity(
            1.0 if trans_ok else 0.35)
        self._translated_btn.setToolTip(
            f"Reveal translated file:\n{resolved_trans}" if trans_ok
            else "Translated file not found on disk"
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

        Thin wrapper around :func:`_resolve_book_source_file` so the
        Book Details source button and the card context menu's "Reveal
        source file" action share one resolution path.
        """
        return _resolve_book_source_file(self._book)

    def _resolve_translated_file_target(self) -> str:
        """Return the compiled translated EPUB path the 📕 button opens.

        Thin wrapper around :func:`_resolve_book_translated_file` so
        the Book Details translated button and the card context menu's
        "Reveal Translated File" action share one resolution path.
        """
        return _resolve_book_translated_file(self._book)

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

    def _reveal_translated(self):
        """Reveal the compiled / translated EPUB in the file explorer."""
        path = self._resolve_translated_file_target()
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
