# epub_library.py
"""
EPUB Library & Reader for Glossarion Desktop (Windows / macOS).

Provides:
  - scan_for_epubs(): recursively find .epub files across output dirs
  - EpubLibraryDialog: grid-card browser with cover thumbnails
  - EpubReaderDialog: simple in-app EPUB reader with TOC navigation
"""

import os
import sys
import hashlib
import logging
import shutil
import subprocess
import tempfile
import platform
import traceback
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QScrollArea, QWidget, QLineEdit, QFrame, QSplitter, QTextBrowser,
    QListWidget, QListWidgetItem, QMessageBox, QSizePolicy, QToolButton,
    QApplication, QMenu, QComboBox, QStackedWidget
)
from PySide6.QtCore import Qt, QSize, Signal, Slot, QThread, QTimer, QSizeF, QUrl
from PySide6.QtGui import QPixmap, QFont, QIcon, QImage, QCursor, QShortcut, QKeySequence, QTransform

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
    """Return the dedicated Glossarion Library folder."""
    docs = Path.home() / "Documents" / "Glossarion" / "Library"
    try:
        docs.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return str(docs)


def _origins_file() -> str:
    """Path to the library origins mapping file."""
    return os.path.join(get_library_dir(), "library_origins.txt")


def _load_origins() -> dict[str, str]:
    """Load {library_basename: original_source_path} mapping."""
    import json
    try:
        with open(_origins_file(), "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_origins(origins: dict[str, str]):
    """Save the origins mapping."""
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


def _epub_cache_key(epub_path: str) -> str:
    """Generate a cache key from path + file modification time."""
    try:
        mtime = os.path.getmtime(epub_path)
    except OSError:
        mtime = 0
    raw = f"{epub_path}|{mtime}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:16]


def _load_epub_cache(epub_path: str):
    """Try to load cached EPUB data.

    Returns ``(chapters, images, filenames)`` where ``filenames`` is a parallel
    list of source item names (one per chapter entry) or ``None`` on failure.
    ``filenames`` is empty when the cache predates that field — callers must
    handle that gracefully.
    """
    import pickle
    try:
        key = _epub_cache_key(epub_path)
        cache_file = os.path.join(_epub_cache_dir(), f"{key}.pkl")
        if os.path.isfile(cache_file):
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and "chapters" in data and "images" in data:
                return data["chapters"], data["images"], data.get("filenames", [])
    except Exception:
        pass
    return None


def _save_epub_cache(epub_path: str, chapters, images, filenames=None):
    """Save parsed EPUB data to disk cache."""
    import pickle
    try:
        key = _epub_cache_key(epub_path)
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
        return cached

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
    try:
        folder = os.path.dirname(path) if os.path.isfile(path) else path
        if platform.system() == "Windows":
            if os.path.isfile(path):
                subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
            else:
                os.startfile(folder)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", "-R", path] if os.path.isfile(path) else ["open", folder])
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


def _read_progress_summary(progress_file: str) -> dict | None:
    """Return a lightweight summary of translation_progress.json or None on failure.

    The summary counts chapter statuses so the library card can render a
    fraction/percentage without paying for the full OPF-aware match.
    """
    import json as _json
    try:
        with open(progress_file, "r", encoding="utf-8") as f:
            prog = _json.load(f)
    except (OSError, _json.JSONDecodeError):
        return None
    chapters = prog.get("chapters", {}) or {}
    total = len(chapters)
    completed = 0
    in_progress = 0
    failed = 0
    for ch in chapters.values():
        status = (ch or {}).get("status", "")
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


def _folder_has_output_epub(folder: str) -> str | None:
    """Return the path to the first .epub in *folder* or None."""
    try:
        for entry in os.scandir(folder):
            if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(".epub"):
                return entry.path
    except (PermissionError, OSError):
        pass
    return None


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
    """Scan the Library folder for completed EPUBs.

    The Library folder (``Documents/Glossarion/Library``) is a curated shelf:
    all EPUBs inside are treated as finished works with no translation context.
    """
    library_dir = os.path.normpath(os.path.abspath(get_library_dir()))
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
                            results.append({
                                "name": os.path.splitext(entry.name)[0],
                                "path": entry.path,
                                "size": stat.st_size,
                                "mtime": stat.st_mtime,
                                "in_library": True,
                                "type": "epub",
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
    """Scan the output root(s) for translation folders (one card per folder).

    Honors the OUTPUT_DIRECTORY override strictly via :func:`_resolve_output_roots`.
    A folder qualifies when it has at least one of:
      * ``translation_progress.json``
      * ``metadata.json``
      * a compiled ``.epub``

    Each result carries ``metadata_json`` (already loaded) so the card/details
    dialog can render without a second filesystem hit.
    """
    import json as _json
    roots = _resolve_output_roots(config)
    if not roots:
        return []

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
                output_epub = _folder_has_output_epub(folder)
                has_progress = os.path.isfile(progress_file)
                has_metadata = os.path.isfile(metadata_file)
                # Must look like a translation workspace — otherwise random
                # sibling directories of the output root would flood the tab.
                if not has_progress and not has_metadata and not output_epub:
                    continue

                metadata_json: dict = {}
                if has_metadata:
                    try:
                        with open(metadata_file, "r", encoding="utf-8") as f:
                            loaded = _json.load(f)
                        if isinstance(loaded, dict):
                            metadata_json = loaded
                    except (OSError, _json.JSONDecodeError):
                        metadata_json = {}

                summary = _read_progress_summary(progress_file) if has_progress else None
                total = summary["total"] if summary else 0
                done = summary["completed"] if summary else 0
                failed = summary["failed"] if summary else 0

                # Title precedence: metadata.json title → folder basename.
                raw_title = (metadata_json.get("title")
                             or metadata_json.get("original_title")
                             or entry.name)

                try:
                    stat = os.stat(folder)
                except OSError:
                    continue

                results.append({
                    "name": raw_title,
                    "folder_name": entry.name,
                    # Use the folder itself as the card's identifying path so
                    # downstream code can locate progress artifacts trivially.
                    "path": folder,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "in_library": False,
                    "type": "in_progress",
                    "is_in_progress": True,
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
                })
    results.sort(key=lambda r: r["mtime"], reverse=True)
    return results


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

    # Attach original source paths for files that were moved to Library
    origins = _load_origins()
    for r in results:
        basename = os.path.basename(r["path"])
        if basename in origins:
            r["original_path"] = origins[basename]

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

_SIZE_PRESETS = {
    SIZE_COMPACT: {"card_w": 110, "cover_h": 140, "title_size": "8pt", "title_max_len": 18, "spacing": 3},
    SIZE_NORMAL:  {"card_w": 140, "cover_h": 175, "title_size": "8.5pt", "title_max_len": 24, "spacing": 4},
    SIZE_LARGE:   {"card_w": 180, "cover_h": 225, "title_size": "9pt", "title_max_len": 32, "spacing": 5},
    SIZE_XL:      {"card_w": 230, "cover_h": 290, "title_size": "9.5pt", "title_max_len": 40, "spacing": 6},
    SIZE_2XL:     {"card_w": 290, "cover_h": 365, "title_size": "10pt", "title_max_len": 50, "spacing": 8},
    SIZE_3XL:     {"card_w": 360, "cover_h": 450, "title_size": "10.5pt", "title_max_len": 60, "spacing": 10},
    SIZE_4XL:     {"card_w": 440, "cover_h": 550, "title_size": "11pt", "title_max_len": 70, "spacing": 12},
    SIZE_5XL:     {"card_w": 530, "cover_h": 660, "title_size": "11.5pt", "title_max_len": 80, "spacing": 14},
    SIZE_6XL:     {"card_w": 630, "cover_h": 790, "title_size": "12pt", "title_max_len": 90, "spacing": 16},
}


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
    """Scan Library (Completed) and output roots (In Progress) in one shot.

    Emits ``(in_progress_list, completed_list)`` so the tabbed dialog can
    refresh both panes from a single worker run.
    """
    finished = Signal(list, list)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self._config = config or {}

    def run(self):
        try:
            in_progress = scan_output_folders(self._config)
        except Exception:
            logger.debug("In-progress scan failed: %s", traceback.format_exc())
            in_progress = []
        try:
            completed = scan_library_completed(self._config)
        except Exception:
            logger.debug("Completed scan failed: %s", traceback.format_exc())
            completed = []
        self.finished.emit(in_progress, completed)


class _CoverLoader(QThread):
    finished = Signal(str, str)

    def __init__(self, file_path: str, file_type: str = "epub", config: dict | None = None,
                 original_path: str | None = None, parent=None):
        super().__init__(parent)
        self._file_path = file_path
        self._file_type = file_type
        self._config = config or {}
        self._original_path = original_path

    def run(self):
        if self._file_type == "epub":
            cover = _extract_cover(self._file_path)
        elif self._file_type == "in_progress":
            # For an in-progress card the "path" is the output folder itself.
            cover = _find_cover_in_dir(self._file_path)
            # As a secondary pass, try the compiled output EPUB (if any) so
            # finished-but-not-organized novels still show a real cover.
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


class _BookCard(QFrame):
    clicked = Signal(dict)
    context_menu_requested = Signal(dict, object)

    def __init__(self, book: dict, preset: dict | None = None, parent=None):
        super().__init__(parent)
        self.book = book
        p = preset or _SIZE_PRESETS[SIZE_NORMAL]
        self._card_w = p["card_w"]
        self._cover_h = p["cover_h"]
        self._has_cover = False

        self.setFixedWidth(self._card_w)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            _BookCard { background: #1e1e2e; border: 1px solid #2a2a3e; border-radius: 6px; }
            _BookCard:hover { border: 1px solid #6c63ff; background: #252540; }
        """)

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

        title = book["name"]
        max_len = p.get("title_max_len", 24)
        if len(title) > max_len:
            title = title[:max_len - 2] + "…"
        title_lbl = QLabel(title)
        title_lbl.setWordWrap(True)
        title_lbl.setMaximumHeight(36)
        title_lbl.setStyleSheet(f"color: #e0e0e0; font-size: {p.get('title_size', '9pt')}; font-weight: bold;")
        title_lbl.setToolTip(book["name"])
        layout.addWidget(title_lbl)

        # Size + file type badge on same row
        file_type = book.get("type", "epub")
        type_info = {
            "epub": ("\U0001f4d5EPUB", "#6c63ff"),
            "pdf": ("\U0001f4c4PDF", "#e74c3c"),
            "txt": ("\U0001f4d7TXT", "#2ecc71"),
            "in_progress": ("\U0001f4c1FOLDER", "#ffd166"),
        }
        badge_text, badge_color = type_info.get(file_type, ("\U0001f4d5EPUB", "#6c63ff"))
        size_mb = book["size"] / (1024 * 1024)
        size_str = f"{size_mb:.1f} MB" if size_mb >= 1 else f"{book['size'] / 1024:.0f} KB"
        info_row = QHBoxLayout()
        info_row.setContentsMargins(0, 0, 0, 0)
        info_row.setSpacing(4)
        size_lbl = QLabel(size_str)
        size_lbl.setStyleSheet("color: #888; font-size: 7.5pt;")
        info_row.addWidget(size_lbl)
        badge_lbl = QLabel(badge_text)
        badge_lbl.setStyleSheet(f"color: {badge_color}; font-size: 7pt; font-weight: bold;")
        info_row.addWidget(badge_lbl)
        info_row.addStretch()
        layout.addLayout(info_row)

        # In-progress indicator: small percent pill + overlay ribbon on the cover
        if book.get("is_in_progress"):
            total = int(book.get("total_chapters", 0) or 0)
            done = int(book.get("completed_chapters", 0) or 0)
            pct = int(round((done / total) * 100)) if total else 0
            progress_row = QHBoxLayout()
            progress_row.setContentsMargins(0, 0, 0, 0)
            progress_row.setSpacing(4)
            pill = QLabel(f"\u23f3 {done}/{total}" if total else "\u23f3 In progress")
            pill.setToolTip(f"Translation in progress \u2014 {pct}% ({done}/{total} chapters)")
            pill.setStyleSheet(
                "color: #ffd166; background: rgba(108, 99, 255, 0.18); "
                "border: 1px solid #6c63ff; border-radius: 3px; "
                "font-size: 7pt; font-weight: bold; padding: 1px 5px;"
            )
            progress_row.addWidget(pill)
            if total:
                pct_lbl = QLabel(f"{pct}%")
                pct_lbl.setStyleSheet("color: #8ab4d0; font-size: 7pt; font-weight: bold;")
                progress_row.addWidget(pct_lbl)
            progress_row.addStretch()
            layout.addLayout(progress_row)

            # Corner ribbon on the cover label (absolutely positioned child)
            self._progress_ribbon = QLabel("IN PROGRESS", self.cover_label)
            self._progress_ribbon.setStyleSheet(
                "color: #fff; background: rgba(108, 99, 255, 0.92); "
                "font-size: 6.5pt; font-weight: bold; padding: 1px 5px; "
                "border-bottom-right-radius: 3px;"
            )
            self._progress_ribbon.move(0, 0)
            self._progress_ribbon.show()

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
        self.clicked.emit(self.book)
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        self.context_menu_requested.emit(self.book, event.globalPos())


# ---------------------------------------------------------------------------
# Library Dialog
# ---------------------------------------------------------------------------

class EpubLibraryDialog(QDialog):
    # Emitted when the user imports a new EPUB from the "In Progress" tab.
    # Parents (e.g. TranslatorGUI) can connect to set it as the input file.
    import_epub_requested = Signal(str)

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
        self._cover_threads: list[_CoverLoader] = []
        # Restore persisted settings
        self._sort_mode = self._config.get('epub_library_sort', SORT_DATE)
        self._card_size = self._config.get('epub_library_card_size', SIZE_COMPACT)
        self._current_tab = self._config.get('epub_library_tab', 0)
        self._scanner_thread: _DualScannerThread | None = None
        self._setup_ui()
        # Defer the filesystem scan so the dialog paints immediately;
        # on large libraries scan_for_epubs() + cover-thread spawn can otherwise
        # block the UI for a noticeable fraction of a second on first open.
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
        toolbar.addStretch()
        self._count_label = QLabel("")
        self._count_label.setStyleSheet("color: #888; font-size: 8.5pt;")
        toolbar.addWidget(self._count_label)
        root.addLayout(toolbar)

        # Per-tab action bar: each tab gets its own extra button row with
        # context-specific actions (Import EPUB for In Progress, Open Library
        # Folder for Completed). The shared sort/size toolbar above applies
        # to whichever tab is currently active.
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
        self._import_btn = QPushButton("\u2795  Import EPUB")
        self._import_btn.setCursor(Qt.PointingHandCursor)
        self._import_btn.setToolTip(
            "Pick a source EPUB to load into Glossarion for translation."
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
        self._ip_grid_container = QWidget()
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
        self._comp_grid_container = QWidget()
        self._comp_grid_layout = QGridLayout(self._comp_grid_container)
        self._comp_grid_layout.setContentsMargins(1, 1, 1, 1)
        self._comp_scroll.setWidget(self._comp_grid_container)
        comp_layout.addWidget(self._comp_scroll, 1)
        self._comp_empty_label = QLabel(
            "Your Library is empty.\nOrganize finished EPUBs into this folder to see them here.")
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
        self._spin_timer.timeout.connect(self._rotate_spinner)
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

    # -- Tab / scan / render helpers ----------------------------------------

    def _rotate_spinner(self):
        """Rotate the Halgakos icon by 15° each tick (matches reader)."""
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

    def _import_epub(self):
        """Let the user pick a source EPUB; forward it to the parent GUI."""
        from PySide6.QtWidgets import QFileDialog
        start_dir = str(Path.home() / "Downloads")
        if not os.path.isdir(start_dir):
            start_dir = str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Import EPUB to translate", start_dir,
            "EPUB files (*.epub);;All files (*.*)",
        )
        if not path:
            return
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Import EPUB", "Selected path is not a file.")
            return
        # Let the parent GUI (TranslatorGUI) pick it up; fall back to just
        # opening the file's folder if no consumer is connected.
        try:
            self.import_epub_requested.emit(path)
        except Exception:
            logger.debug("Failed to emit import_epub_requested: %s", traceback.format_exc())
        # Give the user a confirmation that the import was accepted.
        QMessageBox.information(
            self, "Import EPUB",
            f"Loaded for translation:\n{os.path.basename(path)}\n\n"
            f"Start the translation from the main window to see it here.",
        )

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
        ip_new = {b["path"] for b in in_progress}
        ip_old = {b["path"] for b in self._in_progress_books}
        comp_new = {b["path"] for b in completed}
        comp_old = {b["path"] for b in self._completed_books}
        if ip_new != ip_old or comp_new != comp_old:
            self._in_progress_books = in_progress
            self._completed_books = completed
            self._refresh_view()

    def _load_books(self):
        """Kick off an async scan of both tabs and show the loading spinner."""
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

    def _populate_grid_common(
        self,
        books: list[dict],
        grid_layout: QGridLayout,
        card_list: list[_BookCard],
        count_label: QLabel,
        empty_label: QLabel,
        count_word: str,
    ):
        """Shared render pipeline used by both tabs."""
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

        for idx, book in enumerate(books):
            card = _BookCard(book, preset=preset)
            card.clicked.connect(self._on_card_clicked)
            card.context_menu_requested.connect(self._show_context_menu)
            card_list.append(card)
            row, col = divmod(idx, cols)
            grid_layout.addWidget(card, row, col, Qt.AlignTop | Qt.AlignLeft)

            loader = _CoverLoader(
                book["path"],
                file_type=book.get("type", "epub"),
                config=self._config,
                original_path=book.get("original_path"),
                parent=self,
            )
            loader.finished.connect(self._on_cover_loaded)
            self._cover_threads.append(loader)
            loader.start()

        for c in range(grid_layout.columnCount()):
            grid_layout.setColumnStretch(c, 0)
        grid_layout.setColumnStretch(cols, 1)

    def _populate_in_progress(self, books: list[dict]):
        self._populate_grid_common(
            books, self._ip_grid_layout, self._ip_cards,
            self._ip_count_label, self._ip_empty_label, "novel",
        )

    def _populate_completed(self, books: list[dict]):
        self._populate_grid_common(
            books, self._comp_grid_layout, self._comp_cards,
            self._comp_count_label, self._comp_empty_label, "book",
        )

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
            reader = EpubReaderDialog(book["path"], config=self._config, parent=self)
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
        file_type = book.get("type", "epub")
        if file_type == "in_progress":
            details_action = menu.addAction("\U0001f4d1  Open Book Details")
            details_action.triggered.connect(lambda: self._on_card_clicked(book))
            out_epub = book.get("output_epub_path") or ""
            if out_epub and os.path.isfile(out_epub):
                reader_action = menu.addAction("\U0001f4d6  Open Translated EPUB")
                reader_action.triggered.connect(
                    lambda p=out_epub: self._open_reader_direct({"path": p, "type": "epub"})
                )
        elif file_type == "epub":
            details_action = menu.addAction("\U0001f4d1  Open Book Details")
            details_action.triggered.connect(lambda: self._on_card_clicked(book))
            reader_action = menu.addAction("\U0001f4d6  Open in Reader")
            reader_action.triggered.connect(lambda: self._open_reader_direct(book))
        else:
            open_action = menu.addAction("\U0001f4c2  Open File")
            open_action.triggered.connect(lambda: self._on_card_clicked(book))
        menu.addSeparator()
        folder_action = menu.addAction("\U0001f4c2  Open Output Folder")
        # For in-progress cards the card "path" IS the output folder; for
        # completed library EPUBs use the file's parent directory.
        if file_type == "in_progress":
            output_folder = book.get("output_folder") or book.get("path", "")
        else:
            output_folder = (book.get("output_folder")
                             or os.path.dirname(book.get("path", "")))
        folder_action.triggered.connect(lambda: _open_folder_in_explorer(output_folder))
        lib_action = menu.addAction("\U0001f4c1  Open Library Folder")
        lib_action.triggered.connect(lambda: _open_folder_in_explorer(get_library_dir()))
        menu.addSeparator()
        copy_path_action = menu.addAction("\U0001f4cb  Copy Path")
        copy_path_action.triggered.connect(lambda: QApplication.clipboard().setText(book["path"]))
        menu.exec(pos)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Debounce: re-flow cards once the user stops dragging (300ms).
        if not (self._ip_cards or self._comp_cards):
            return
        if not hasattr(self, '_resize_timer'):
            self._resize_timer = QTimer(self)
            self._resize_timer.setSingleShot(True)
            self._resize_timer.timeout.connect(self._refresh_view)
        self._resize_timer.start(300)

    def closeEvent(self, event):
        """Hide the dialog instead of closing \u2014 persist settings."""
        self._config['epub_library_sort'] = self._sort_mode
        self._config['epub_library_card_size'] = self._card_size
        self._config['epub_library_tab'] = self._current_tab
        event.ignore()
        self.hide()


# ---------------------------------------------------------------------------
# Book Details — metadata / TOC parser
# ---------------------------------------------------------------------------

def _parse_epub_details(epub_path: str) -> dict:
    """Extract OPF metadata, spine order and per-chapter raw titles.

    Returns a dict shaped roughly like the OPF DC schema plus a ``chapters``
    list of ``{'href', 'filename', 'title'}``. All fields are best-effort and
    may be empty strings/lists on failure.
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
            # the first ~32KB which is more than enough for <title>/<h1>.
            from bs4 import BeautifulSoup
            chapters = []
            for idref, href in ordered_hrefs:
                media_type = manifest.get(idref, ("", ""))[1]
                if media_type and ("html" not in media_type.lower() and "xhtml" not in media_type.lower()):
                    # Non-text spine item — still include so index matches.
                    chapters.append({"href": href, "filename": os.path.basename(href),
                                     "title": os.path.splitext(os.path.basename(href))[0]})
                    continue
                title = ""
                if href in names_set:
                    try:
                        raw = zf.read(href)[:32_768]
                        soup = BeautifulSoup(raw, "html.parser")
                        t = soup.find("title")
                        if t and t.get_text(strip=True):
                            title = t.get_text(strip=True)
                        if not title:
                            for h in soup.find_all(["h1", "h2", "h3"]):
                                ht = h.get_text(strip=True)
                                if ht:
                                    title = ht
                                    break
                    except Exception:
                        logger.debug("Chapter title parse failed: %s", traceback.format_exc())
                if not title:
                    title = os.path.splitext(os.path.basename(href))[0]
                    title = title.replace("_", " ").replace("-", " ").strip() or title
                chapters.append({"href": href, "filename": os.path.basename(href), "title": title})
            details["chapters"] = chapters
    except Exception:
        logger.debug("EPUB details parse failed: %s", traceback.format_exc())

    return details


def _read_translated_chapter_title(path: str) -> str:
    """Extract a translated-chapter title from a response_*.html file."""
    try:
        from bs4 import BeautifulSoup
        with open(path, "rb") as f:
            raw = f.read(32_768)
        soup = BeautifulSoup(raw, "html.parser")
        t = soup.find("title")
        if t and t.get_text(strip=True):
            return t.get_text(strip=True)
        for h in soup.find_all(["h1", "h2", "h3"]):
            ht = h.get_text(strip=True)
            if ht:
                return ht
    except Exception:
        logger.debug("Translated title parse failed for %s: %s", path, traceback.format_exc())
    return ""


class _BookDetailsLoader(QThread):
    """Parse EPUB metadata + TOC + translation status off the UI thread."""
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
                if output_folder and os.path.isdir(output_folder):
                    progress_file = progress_file or os.path.join(
                        output_folder, "translation_progress.json")
                    # Authoritative pointer file first, then any .epub in
                    # the folder (which may be the compiled output).
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

            details = _parse_epub_details(source_epub) if source_epub else {
                "title": "", "authors": [], "publisher": "", "language": "",
                "date": "", "description": "", "subjects": [], "identifier": "",
                "chapters": [],
            }
            cover = _extract_cover(source_epub) if source_epub else None
            if not cover and output_folder and os.path.isdir(output_folder):
                cover = _find_cover_in_dir(output_folder)
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

            import re as _re
            for idx, ch in enumerate(details.get("chapters", [])):
                filename = ch["filename"]
                raw_title = ch["title"]
                # Special files mirror Retranslation_GUI's classifier: no digit
                # anywhere in the filename → special (cover, nav, toc, info,
                # message, etc.). The BookDetailsDialog filters these out by
                # default, matching the Progress Manager's behavior.
                is_special = not bool(_re.search(r"\d", filename))
                norm_key = _norm(filename)
                match = prog_by_basename.get(norm_key) or prog_by_output.get(norm_key)
                status = (match or {}).get("status", "")
                output_file = (match or {}).get("output_file", "")
                translated_title = ""
                translated_path = ""
                # Check disk — the progress file can lag behind real files.
                if output_folder and os.path.isdir(output_folder):
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
                if not status:
                    # Only mark as pending when we actually have a progress
                    # context against which "pending" is meaningful. Without
                    # one, leave status empty so the row shows no badge.
                    status = "pending" if has_progress_context else ""
                chapters_info.append({
                    "index": idx,
                    "filename": filename,
                    "raw_title": raw_title,
                    "translated_title": translated_title,
                    "translated_path": translated_path,
                    "status": status,
                    "is_special": is_special,
                })

            # Optional metadata.json overrides (translator-produced).
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
        # Whether the "show special files" toggle is on (hidden by default to
        # match Retranslation_GUI's Progress Manager behavior for EPUBs).
        self._show_special_files = bool(self._config.get("epub_details_show_special_files", False))

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
        icon_path = _find_halgakos_icon()
        if icon_path:
            pm = QPixmap(icon_path)
            if not pm.isNull():
                scaled = pm.scaled(160, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self._cover_lbl.setPixmap(scaled)
        else:
            self._cover_lbl.setText("\U0001f4d6")
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
        self._source_btn.setToolTip("Reveal source EPUB")
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
        # Bumped vertical spacing keeps wrapped value labels from visually
        # colliding with the next row's key label (e.g. multi-line "Title"
        # running into "Author" on Qt's conservative sizeHint path).
        self._meta_grid.setVerticalSpacing(14)
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
        # and collide with the next metadata row.
        meta_wrapper.setMinimumWidth(320)
        meta_wrapper.setMaximumWidth(380)
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

        body_layout.addStretch()
        outer.addWidget(self._scroll, 1)

    # -- Data population ----------------------------------------------------

    def _start_loading(self):
        self._synopsis_lbl.setText("Loading book details\u2026")
        self._loader = _BookDetailsLoader(self._book, self)
        self._loader.done.connect(self._on_details_ready)
        self._loader.error.connect(self._on_details_error)
        self._loader.start()

    @Slot(dict)
    def _on_details_ready(self, payload: dict):
        self._details = payload.get("details", {}) or {}
        self._chapters_info = payload.get("chapters_info", []) or []
        self._metadata_json = payload.get("metadata_json", {}) or {}
        cover_path = payload.get("cover", "")
        if cover_path:
            try:
                pm = QPixmap(cover_path)
                if not pm.isNull():
                    scaled = pm.scaled(self._cover_lbl.width(), self._cover_lbl.height(),
                                       Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self._cover_lbl.setPixmap(scaled)
                    self._cover_lbl.setText("")
            except Exception:
                pass

        title = (self._metadata_json.get("title") or self._details.get("title")
                 or self._book.get("name", ""))
        self._title_lbl.setText(title or self._book.get("name", ""))
        authors = list(self._metadata_json.get("authors") or self._details.get("authors") or [])
        if isinstance(authors, str):
            authors = [authors]
        self._author_lbl.setText(", ".join(authors) if authors else "")

        synopsis = (self._metadata_json.get("description")
                    or self._details.get("description") or "").strip()
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
        for i, (key, val) in enumerate(rows):
            k_lbl = QLabel(key)
            k_lbl.setProperty("class", "meta-k")
            k_lbl.setStyleSheet("color: #7a8599; font-size: 8.5pt;")
            v_lbl = QLabel(str(val))
            v_lbl.setProperty("class", "meta-v")
            v_lbl.setStyleSheet("color: #e0e0e0; font-size: 9.5pt; font-weight: bold;")
            v_lbl.setWordWrap(True)
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

        # In-progress strip + reader-entry-point relabeling
        has_translated = any(c.get("translated_path") for c in self._chapters_info)
        if self._book.get("is_in_progress"):
            done = sum(1 for c in self._chapters_info if c["status"] == "completed")
            total = len(self._chapters_info) or int(self._book.get("total_chapters", 0) or 0)
            if total:
                pct = int(round((done / total) * 100))
                self._progress_strip.setText(
                    f"\u23f3  Translation in progress \u2014 {done}/{total} chapters ({pct}%)"
                )
            else:
                self._progress_strip.setText("\u23f3  Translation in progress")
            self._progress_strip.show()
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

        # Chapter rows
        self._populate_chapters()

        # Button availability — also dim the emoji + update tooltip so it's
        # obvious why the action isn't usable. Qt's default disabled state
        # only grays the button chrome; emoji glyphs are unaffected by the
        # stylesheet `color:` rule, so we lower the widget's opacity directly.
        out_folder = self._book.get("output_folder") or ""
        folder_ok = bool(out_folder) and os.path.isdir(out_folder)
        self._folder_btn.setEnabled(folder_ok)
        self._folder_btn.setCursor(Qt.PointingHandCursor if folder_ok else Qt.ForbiddenCursor)
        self._folder_btn_opacity.setOpacity(1.0 if folder_ok else 0.35)
        self._folder_btn.setToolTip(
            "Open output folder in file explorer" if folder_ok
            else "Output folder not available for this book"
        )

        src_path = self._book.get("path", "") or ""
        source_ok = bool(src_path) and os.path.isfile(src_path)
        self._source_btn.setEnabled(source_ok)
        self._source_btn.setCursor(Qt.PointingHandCursor if source_ok else Qt.ForbiddenCursor)
        self._source_btn_opacity.setOpacity(1.0 if source_ok else 0.35)
        self._source_btn.setToolTip(
            "Reveal source EPUB" if source_ok
            else "Source EPUB file not found on disk"
        )

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

    def _populate_chapters(self):
        # Clear previous rows
        while self._chap_layout.count():
            item = self._chap_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()

        for info in self._chapters_info:
            row = _ChapterRow(info, parent=self._chap_container)
            row.activated.connect(self._open_reader)
            self._chap_layout.addWidget(row)
        self._chap_layout.addStretch()
        self._apply_chapter_filter(self._toc_search.text())
        self._update_toc_toggle_label()

    def _visible_counts(self) -> tuple[int, int]:
        """Return (done, total) considering the special-files toggle."""
        if self._show_special_files:
            items = self._chapters_info
        else:
            items = [c for c in self._chapters_info if not c.get("is_special")]
        total = len(items)
        done = sum(1 for c in items if c.get("status") == "completed")
        return done, total

    def _has_progress_context(self) -> bool:
        """True when at least one chapter has a non-empty translation status."""
        return any((c.get("status") or "") for c in self._chapters_info)

    def _update_toc_toggle_label(self):
        done, total = self._visible_counts()
        prefix = "\u25bc  Chapters" if self._chap_container.isVisible() else "\u25b6  Chapters"
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

    def _toggle_chapters(self):
        show = not self._chap_container.isVisible()
        self._chap_container.setVisible(show)
        self._toc_search.setVisible(show)
        self._special_cb.setVisible(show)
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
        try:
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
            reader = EpubReaderDialog(
                self._book["path"],
                config=self._config,
                parent=self,
                initial_chapter=initial_chapter,
                initial_chapter_filename=initial_filename,
                translated_overlay=overlay or None,
                extra_image_dirs=extra_dirs or None,
                window_title=window_title,
            )
            QApplication.restoreOverrideCursor()
            reader.setModal(False)
            reader.setAttribute(Qt.WA_DeleteOnClose)
            self._active_reader = reader
            reader.show()
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            logger.error("Could not open reader from details: %s\n%s", exc, traceback.format_exc())
            QMessageBox.warning(self, "Error", f"Could not open EPUB reader:\n{exc}")

    def _open_output_folder(self):
        folder = self._book.get("output_folder") or os.path.dirname(self._book.get("path", ""))
        if folder:
            _open_folder_in_explorer(folder)

    def _reveal_source(self):
        path = self._book.get("path")
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
    """
    activated = Signal(int)

    def __init__(self, info: dict, parent=None):
        super().__init__(parent)
        self.info = info
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            _ChapterRow { background: #1a1a2a; border: 1px solid #242438; border-radius: 6px; }
            _ChapterRow:hover { border-color: #6c63ff; background: #232340; }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        status = info.get("status", "") or ""
        translated = info.get("translated_title") or ""
        raw = info.get("raw_title") or ""
        if translated and status == "completed":
            primary = QLabel(translated)
            primary.setProperty("class", "translated")
            primary.setStyleSheet("color: #e0e0e0; font-size: 10pt; font-weight: bold;")
        else:
            primary = QLabel(raw or info.get("filename", ""))
            primary.setProperty("class", "raw")
            primary.setStyleSheet("color: #c8cbe0; font-size: 10pt; font-weight: bold;")
        primary.setWordWrap(True)
        text_col.addWidget(primary)

        sub = QLabel(info.get("filename", ""))
        sub.setProperty("class", "filename")
        sub.setStyleSheet("color: #8a8fa8; font-size: 8.5pt; font-family: 'Consolas','Menlo',monospace;")
        text_col.addWidget(sub)
        layout.addLayout(text_col, 1)

        badge = None
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

    def mousePressEvent(self, event):
        self.activated.emit(int(self.info.get("index", 0)))
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# EPUB Reader — background loader thread
# ---------------------------------------------------------------------------

class _EpubLoaderThread(QThread):
    """Load the EPUB in a background thread and write result to cache.

    Emitting large binary data (images) through Qt signals across threads
    can crash the GUI.  Instead, write to a cache file and emit a
    lightweight success signal.
    """
    done = Signal()          # success — data available via cache
    error = Signal(str)

    def __init__(self, epub_path: str, parent=None):
        super().__init__(parent)
        self._epub_path = epub_path

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

            chapters: list[tuple[str, str]] = []
            filenames: list[str] = []
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                try:
                    content = item.get_content().decode("utf-8", errors="replace")
                    soup = BeautifulSoup(content, "html.parser")
                    text = soup.get_text(strip=True)
                    if not text or len(text) < 10:
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

            # Write to cache (avoids emitting large data through Qt signals)
            _save_epub_cache(self._epub_path, chapters, images, filenames)
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
                 window_title: str | None = None):
        super().__init__(parent)
        self._epub_path = epub_path
        self._config = config or {}
        self._chapters: list[tuple[str, str]] = []
        self._chapter_filenames: list[str] = []
        self._images: dict[str, bytes] = {}
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
        self._spin_timer.timeout.connect(self._rotate_spinner)
        loading_text = QLabel("Loading EPUB…")
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
                w = QWebEngineView()
                w.setContextMenuPolicy(Qt.NoContextMenu)
                w.setUrl(QUrl("about:blank"))
                # Set page background to match theme (prevents white flash)
                from PySide6.QtGui import QColor
                t = self._get_theme()
                w.page().setBackgroundColor(QColor(t['bg']))
            else:
                w = QTextBrowser()
                w.setOpenExternalLinks(False)
                w.setOpenLinks(False)
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

    def _rotate_spinner(self):
        """Rotate the Halgakos icon by 15° each tick."""
        if self._spin_pixmap:
            self._spin_angle = (self._spin_angle + 15) % 360
            t = QTransform().rotate(self._spin_angle)
            rotated = self._spin_pixmap.transformed(t, Qt.FastTransformation)
            self._spin_label.setPixmap(rotated)

    # ── Loading ────────────────────────────────────────────────────────────

    def _start_loading(self):
        self._toolbar_widget.hide()
        self._loading_widget.show()
        self._content_widget.hide()
        self._spin_angle = 0
        self._spin_timer.start()
        # Try cache first
        cached = _load_epub_cache(self._epub_path)
        if cached:
            chapters, images, filenames = cached
            # Older caches may not carry filenames. If an overlay is requested
            # but filenames are missing we MUST reparse the EPUB so the overlay
            # can map correctly — otherwise we'd silently keep the raw text.
            if self._translated_overlay and not filenames:
                pass  # fall through to loader below
            else:
                QTimer.singleShot(0, lambda: self._on_epub_loaded_from_cache(
                    chapters, images, filenames))
                return
        self._loader_thread = _EpubLoaderThread(self._epub_path, self)
        self._loader_thread.done.connect(lambda: self._on_loader_done())
        self._loader_thread.error.connect(lambda msg: self._on_epub_error(msg))
        self._loader_thread.start()

    @Slot()
    def _on_loader_done(self):
        """Loader finished — read data from cache file (avoids large signal data)."""
        cached = _load_epub_cache(self._epub_path)
        if cached:
            self._on_epub_loaded_from_cache(*cached)
        else:
            self._on_epub_error("Failed to read EPUB cache after loading.")

    def _on_epub_loaded_from_cache(self, chapters, images, filenames=None):
        self._spin_timer.stop()
        filenames = list(filenames or [])
        # Apply translated-content overlay: swap source HTML (and optionally
        # the chapter title) with translated versions for chapters that have
        # a response_*.html on disk. Source chapters without an overlay entry
        # are left untouched so the user can still read pending chapters raw.
        # Overlay is keyed by the source filename basename so chapter-order
        # differences between the reader's loader and the BookDetailsDialog's
        # spine parser don't cause off-by-one mismatches (bug: raw+translated
        # rows appearing as duplicates in the TOC).
        if self._translated_overlay and chapters:
            merged: list[tuple[str, str]] = []
            for idx, (title, content) in enumerate(chapters):
                fname = filenames[idx] if idx < len(filenames) else ""
                key = os.path.basename(fname).lower() if fname else ""
                ov = self._translated_overlay.get(key) if key else None
                if ov:
                    path = ov.get("path") or ""
                    if path and os.path.isfile(path):
                        try:
                            with open(path, "rb") as f:
                                raw = f.read()
                            translated_html = raw.decode("utf-8", errors="replace")
                            content = translated_html
                            if ov.get("title"):
                                title = str(ov["title"])
                        except OSError:
                            logger.debug("Could not read translated overlay %s: %s",
                                         path, traceback.format_exc())
                merged.append((title, content))
            chapters = merged
        # Augment the EPUB's own image table with anything discovered in the
        # translator's output folder(s). Translated chapters often reference
        # images that were renamed/moved during translation.
        if self._extra_image_dirs:
            merged_images = dict(images or {})
            _img_exts = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp")
            for dir_path in self._extra_image_dirs:
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
                        # Store under several aliases so the chapter HTML can
                        # resolve them regardless of how src="…" is written.
                        merged_images.setdefault(entry.name, data)
                        rel = os.path.basename(dir_path) + "/" + entry.name
                        merged_images.setdefault(rel, data)
                        merged_images.setdefault("images/" + entry.name, data)
                except (PermissionError, OSError):
                    continue
            images = merged_images
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
            if self._initial_chapter_filename:
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

    def _on_reader_load_finished(self, ok):
        """Called when QWebEngineView finishes loading HTML."""
        if not ok:
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

    def _finalize_single_page(self):
        """After HTML load: get page count and scroll to current page."""
        def on_count(count):
            self._chapter_page_cache[self._current_row] = int(count)
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
            self._chapter_page_cache[self._current_row] = int(count)
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
        """Persist reader settings back into config."""
        self._config['epub_reader_font_size'] = self._font_size
        self._config['epub_reader_line_spacing'] = self._line_spacing
        self._config['epub_reader_theme'] = self._theme_index
        self._config['epub_reader_layout'] = self._layout_mode
        self._config['epub_reader_font_family'] = self._font_family
        super().closeEvent(event)

    # ── Chapter rendering ──────────────────────────────────────────────────

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
