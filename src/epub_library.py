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
    """Try to load cached EPUB data. Returns (chapters, images) or None."""
    import pickle
    try:
        key = _epub_cache_key(epub_path)
        cache_file = os.path.join(_epub_cache_dir(), f"{key}.pkl")
        if os.path.isfile(cache_file):
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and "chapters" in data and "images" in data:
                return data["chapters"], data["images"]
    except Exception:
        pass
    return None


def _save_epub_cache(epub_path: str, chapters, images):
    """Save parsed EPUB data to disk cache."""
    import pickle
    try:
        key = _epub_cache_key(epub_path)
        cache_file = os.path.join(_epub_cache_dir(), f"{key}.pkl")
        with open(cache_file, "wb") as f:
            pickle.dump({"chapters": chapters, "images": images}, f, protocol=pickle.HIGHEST_PROTOCOL)
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
        logger.debug("Cover extraction failed for %s: %s\n%s", epub_path, exc, traceback.format_exc())
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

    override = os.environ.get("OUTPUT_DIRECTORY") or config.get("output_directory")
    if override and os.path.isdir(override):
        _walk(os.path.abspath(override))

    # 3. App directory — on Windows, CWD can be Downloads/Desktop when
    #    launching an .exe, which is far too broad.  Use the exe/script dir instead.
    if platform.system() == "Windows":
        if getattr(sys, "frozen", False):
            app_dir = os.path.dirname(sys.executable)
        else:
            app_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        # macOS / Linux: CWD is typically the project folder
        app_dir = os.getcwd()
    _walk(app_dir)

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
        type_info = {"epub": ("📕EPUB", "#6c63ff"), "pdf": ("📄PDF", "#e74c3c"), "txt": ("📝TXT", "#2ecc71")}
        badge_text, badge_color = type_info.get(file_type, ("📕EPUB", "#6c63ff"))
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
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.book)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        self.context_menu_requested.emit(self.book, event.globalPos())


# ---------------------------------------------------------------------------
# Library Dialog
# ---------------------------------------------------------------------------

class EpubLibraryDialog(QDialog):

    def __init__(self, config: dict | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📚 Glossarion Library")
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
        self._books: list[dict] = []
        self._cards: list[_BookCard] = []
        self._cover_threads: list[_CoverLoader] = []
        # Restore persisted settings
        self._sort_mode = self._config.get('epub_library_sort', SORT_DATE)
        self._card_size = self._config.get('epub_library_card_size', SIZE_COMPACT)
        self._last_move_log: list[tuple[str, str]] = []  # [(src, dst), ...] for undo
        self._setup_ui()
        self._load_books()
        # Auto-refresh library every 2 seconds
        self._auto_refresh_timer = QTimer(self)
        self._auto_refresh_timer.setInterval(2000)
        self._auto_refresh_timer.timeout.connect(self._auto_refresh)
        self._auto_refresh_timer.start()

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
        title = QLabel("📚  Glossarion Library")
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

        refresh_btn = QPushButton("🔄  Refresh")
        refresh_btn.setToolTip("Refresh library")
        refresh_btn.setFixedHeight(28)
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; font-size: 9pt; "
            "color: #888; padding: 2px 8px; }"
            "QPushButton:hover { color: #e0e0e0; }")
        refresh_btn.clicked.connect(self._load_books)
        header.addWidget(refresh_btn)

        # Toggle to show/hide the organize banner
        self._banner_toggle = QPushButton("↕")
        self._banner_toggle.setToolTip("Show/hide the organize & undo banner")
        self._banner_toggle.setFixedSize(28, 28)
        self._banner_toggle.setCursor(Qt.PointingHandCursor)
        self._banner_toggle.setStyleSheet(
            "QPushButton { background: transparent; border: none; font-size: 12pt; "
            "color: #aaa; padding: 0; }"
            "QPushButton:hover { color: #e0e0e0; }")
        self._banner_toggle.clicked.connect(self._toggle_banner)
        header.addWidget(self._banner_toggle)

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

        # Suggestion banner (gentle tone)
        self._relocate_banner = QFrame()
        self._relocate_banner.setStyleSheet(
            "QFrame { background: #1a2a3a; border: 1px solid #2a4a6a; border-radius: 6px; }")
        banner_layout = QHBoxLayout(self._relocate_banner)
        banner_layout.setContentsMargins(10, 6, 10, 6)
        self._banner_lbl = QLabel("")
        self._banner_lbl.setStyleSheet("color: #8ab4d0; font-size: 8.5pt;")
        banner_layout.addWidget(self._banner_lbl)
        banner_layout.addStretch()
        self._organize_btn = QPushButton("Organize into Library")
        self._organize_btn.setCursor(Qt.PointingHandCursor)
        self._organize_btn.setToolTip("Copy these files into your Library folder for easy access")
        self._organize_btn.setStyleSheet(
            "QPushButton { background: #3a5a7a; color: white; border-radius: 4px; "
            "padding: 3px 10px; font-size: 8.5pt; font-weight: bold; border: none; }"
            "QPushButton:hover { background: #4a6a8a; }")
        self._organize_btn.clicked.connect(self._relocate_to_library)
        self._organize_btn.hide()
        banner_layout.addWidget(self._organize_btn)
        # Undo button (hidden by default, shown after a move)
        self._undo_btn = QPushButton("↩ Undo Move")
        self._undo_btn.setCursor(Qt.PointingHandCursor)
        self._undo_btn.setToolTip("Move the files back to their original locations")
        self._undo_btn.setStyleSheet(
            "QPushButton { background: #6f42c1; color: white; border-radius: 4px; "
            "padding: 3px 10px; font-size: 8.5pt; font-weight: bold; border: none; }"
            "QPushButton:hover { background: #5a32a3; }")
        self._undo_btn.clicked.connect(self._undo_relocate)
        self._undo_btn.hide()
        banner_layout.addWidget(self._undo_btn)
        self._relocate_banner.hide()
        root.addWidget(self._relocate_banner)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QScrollBar:vertical { width: 8px; background: #1a1a2e; }"
            "QScrollBar::handle:vertical { background: #3a3a5e; border-radius: 4px; }")
        self._grid_container = QWidget()
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setContentsMargins(1, 1, 1, 1)
        self._grid_spacer = QWidget()
        self._grid_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll.setWidget(self._grid_container)
        root.addWidget(scroll, 1)

        self._empty_label = QLabel("No files found.\nTranslate a novel to see it here!")
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet("color: #555; font-size: 12pt; padding: 40px;")
        self._empty_label.hide()
        root.addWidget(self._empty_label)

        self.setStyleSheet("QDialog { background: #12121e; }")

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

    def _toggle_banner(self):
        """Force-toggle the organize banner visibility."""
        if self._relocate_banner.isVisible():
            self._relocate_banner.hide()
        else:
            outside = [b for b in self._books if not b.get("in_library")]
            has_origins = bool(self._last_move_log) or bool(_load_origins())
            if outside:
                self._banner_lbl.setText(
                    f"💡  {len(outside)} file{'s' if len(outside) != 1 else ''} could be organized into your Library folder.")
            elif has_origins:
                self._banner_lbl.setText(f"✅  Files moved to Library.")
            else:
                self._banner_lbl.setText("📁  All files are already in your Library folder.")
            self._organize_btn.setVisible(bool(outside))
            self._undo_btn.setVisible(has_origins)
            self._relocate_banner.show()

    def _refresh_view(self):
        query = self._search.text().strip().lower()
        books = self._books
        if query:
            books = [b for b in books if query in b["name"].lower()]
        self._populate_grid(self._sorted_books(books))

    def _auto_refresh(self):
        """Lightweight auto-refresh: only reload if file list changed."""
        try:
            new_books = scan_for_epubs(self._config)
            new_paths = {b["path"] for b in new_books}
            old_paths = {b["path"] for b in self._books}
            if new_paths != old_paths:
                self._books = new_books
                self._refresh_view()
        except Exception:
            pass

    def _load_books(self):
        for t in self._cover_threads:
            try:
                t.quit()
                t.wait(100)
            except Exception:
                pass
        self._cover_threads.clear()
        self._books = scan_for_epubs(self._config)
        self._refresh_view()

    def _populate_grid(self, books):
        for card in self._cards:
            try:
                card.setParent(None)
                card.deleteLater()
            except Exception:
                pass
        self._cards.clear()

        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

        if not books:
            self._empty_label.show()
            self._count_label.setText("")
            self._relocate_banner.hide()
            return

        self._empty_label.hide()
        self._count_label.setText(f"{len(books)} file{'s' if len(books) != 1 else ''}")

        # Suggestion banner — show count of outside files or recent move
        outside = [b for b in self._books if not b.get("in_library")]
        has_origins = bool(self._last_move_log) or bool(_load_origins())
        show_banner = bool(outside) or bool(self._last_move_log)
        if show_banner:
            if outside:
                self._banner_lbl.setText(
                    f"💡  {len(outside)} file{'s' if len(outside) != 1 else ''} could be organized into your Library folder.")
            elif self._last_move_log:
                self._banner_lbl.setText(f"✅  Files moved to Library.")
            self._organize_btn.setVisible(bool(outside))
            self._undo_btn.setVisible(has_origins)
            self._relocate_banner.show()
        else:
            self._relocate_banner.hide()

        preset = _SIZE_PRESETS[self._card_size]
        card_w = preset["card_w"]
        spacing = preset["spacing"]
        self._grid_layout.setHorizontalSpacing(spacing)
        self._grid_layout.setVerticalSpacing(spacing + 2)
        cols = max(1, (self.width() - 20) // (card_w + spacing))

        for idx, book in enumerate(books):
            card = _BookCard(book, preset=preset)
            card.clicked.connect(self._on_card_clicked)
            card.context_menu_requested.connect(self._show_context_menu)
            self._cards.append(card)
            row, col = divmod(idx, cols)
            self._grid_layout.addWidget(card, row, col, Qt.AlignTop | Qt.AlignLeft)

            # Load covers for all file types
            loader = _CoverLoader(book["path"], file_type=book.get("type", "epub"),
                                   config=self._config, original_path=book.get("original_path"),
                                   parent=self)
            loader.finished.connect(self._on_cover_loaded)
            self._cover_threads.append(loader)
            loader.start()

        # Force card columns to their natural width; extra space goes to a trailing stretch column
        for c in range(self._grid_layout.columnCount()):
            self._grid_layout.setColumnStretch(c, 0)
        self._grid_layout.setColumnStretch(cols, 1)

        # Reposition the persistent spacer
        self._grid_layout.addWidget(self._grid_spacer, (len(books) - 1) // cols + 1, 0)

    def _on_cover_loaded(self, epub_path, cover_path):
        if not cover_path:
            return
        for card in self._cards:
            if card.book["path"] == epub_path:
                card.set_cover(cover_path)
                break

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
        try:
            # Set loading cursor
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
        open_label = "📖  Open in Reader" if book.get("type", "epub") == "epub" else "📂  Open File"
        open_action = menu.addAction(open_label)
        open_action.triggered.connect(lambda: self._on_card_clicked(book))
        menu.addSeparator()
        folder_action = menu.addAction("📂  Open Output Folder")
        # Use original_path's directory if available (for files moved to Library)
        original = book.get("original_path", book["path"])
        output_folder = os.path.dirname(original)
        folder_action.triggered.connect(lambda: _open_folder_in_explorer(output_folder))
        lib_action = menu.addAction("📁  Open Library Folder")
        lib_action.triggered.connect(lambda: _open_folder_in_explorer(get_library_dir()))
        menu.addSeparator()
        copy_path_action = menu.addAction("📋  Copy Path")
        copy_path_action.triggered.connect(lambda: QApplication.clipboard().setText(book["path"]))
        menu.exec(pos)

    def _relocate_to_library(self):
        outside = [b for b in self._books if not b.get("in_library")]
        if not outside:
            return

        names = "\n".join(f"  \u2022 {b['name']}{os.path.splitext(b['path'])[1]}" for b in outside[:20])
        if len(outside) > 20:
            names += f"\n  \u2026 and {len(outside) - 20} more"

        msg = QMessageBox(self)
        msg.setWindowTitle("Organize into Library")
        msg.setText(
            f"Move {len(outside)} file{'s' if len(outside) != 1 else ''} into the Library folder?\n\n"
            f"{names}\n\n"
            f"Destination:\n  {get_library_dir()}\n\n"
            f"You can undo this afterwards if needed."
        )
        msg.setIcon(QMessageBox.Question)
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        msg.setStyleSheet("""
            QMessageBox QPushButton { min-width: 80px; min-height: 30px; font-size: 10pt; padding: 4px 16px; }
            QMessageBox QDialogButtonBox { qproperty-centerButtons: true; }
        """)
        if msg.exec() != QMessageBox.Yes:
            return

        lib_dir = get_library_dir()
        moved = 0
        errors = []
        move_log: list[tuple[str, str]] = []
        for book in outside:
            src = book["path"]
            dst = os.path.join(lib_dir, os.path.basename(src))
            if os.path.exists(dst):
                base, ext = os.path.splitext(os.path.basename(src))
                counter = 1
                while os.path.exists(dst):
                    dst = os.path.join(lib_dir, f"{base} ({counter}){ext}")
                    counter += 1
            try:
                shutil.move(src, dst)
                move_log.append((src, dst))
                logger.info("Library move: %s -> %s", src, dst)
                moved += 1
            except Exception as exc:
                errors.append(f"{os.path.basename(src)}: {exc}")

        self._last_move_log = move_log

        # Persist original source paths for cover lookup and Open Output Folder
        if move_log:
            origins = _load_origins()
            for src, dst in move_log:
                origins[os.path.basename(dst)] = src
            _save_origins(origins)

        summary = f"Moved {moved} file{'s' if moved != 1 else ''} to Library."
        if move_log:
            summary += "\n\nOriginal locations:"
            for src, dst in move_log[:10]:
                summary += f"\n  {os.path.basename(dst)}  \u2190  {os.path.dirname(src)}"
            if len(move_log) > 10:
                summary += f"\n  \u2026 and {len(move_log) - 10} more"
        if errors:
            summary += f"\n\n{len(errors)} error{'s' if len(errors) != 1 else ''}:\n" + "\n".join(errors[:5])
        QMessageBox.information(self, "Done", summary)
        self._load_books()

    def _undo_relocate(self):
        """Move files back to their original locations (session or persisted)."""
        # Use in-memory log if available, otherwise rebuild from library_origins.txt
        move_pairs = list(self._last_move_log)  # [(original_src, library_dst), ...]
        if not move_pairs:
            origins = _load_origins()
            lib_dir = get_library_dir()
            for lib_name, orig_src in origins.items():
                lib_dst = os.path.join(lib_dir, lib_name)
                if os.path.isfile(lib_dst):
                    move_pairs.append((orig_src, lib_dst))
        if not move_pairs:
            return

        names = "\n".join(
            f"  \u2022 {os.path.basename(dst)}  \u2192  {os.path.dirname(src)}"
            for src, dst in move_pairs[:15]
        )
        if len(move_pairs) > 15:
            names += f"\n  \u2026 and {len(move_pairs) - 15} more"

        msg = QMessageBox(self)
        msg.setWindowTitle("Undo Move")
        msg.setText(
            f"Move {len(move_pairs)} file{'s' if len(move_pairs) != 1 else ''} back to "
            f"their original locations?\n\n{names}"
        )
        msg.setIcon(QMessageBox.Question)
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        msg.setStyleSheet("""
            QMessageBox QPushButton { min-width: 80px; min-height: 30px; font-size: 10pt; padding: 4px 16px; }
            QMessageBox QDialogButtonBox { qproperty-centerButtons: true; }
        """)
        if msg.exec() != QMessageBox.Yes:
            return

        restored = 0
        errors = []
        for src, dst in move_pairs:
            if not os.path.isfile(dst):
                errors.append(f"{os.path.basename(dst)}: file not found in Library")
                continue
            # Ensure original directory still exists
            orig_dir = os.path.dirname(src)
            try:
                os.makedirs(orig_dir, exist_ok=True)
            except OSError:
                pass
            try:
                shutil.move(dst, src)
                logger.info("Library undo: %s -> %s", dst, src)
                restored += 1
            except Exception as exc:
                errors.append(f"{os.path.basename(dst)}: {exc}")

        # Remove origins entries for undone files
        if restored:
            origins = _load_origins()
            for src, dst in self._last_move_log:
                origins.pop(os.path.basename(dst), None)
            _save_origins(origins)

        self._last_move_log.clear()

        summary = f"Restored {restored} file{'s' if restored != 1 else ''} to original location{'s' if restored != 1 else ''}." 
        if errors:
            summary += f"\n\n{len(errors)} error{'s' if len(errors) != 1 else ''}:\n" + "\n".join(errors[:5])
        QMessageBox.information(self, "Undo Complete", summary)
        self._load_books()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._cards:
            return
        # Debounce: reload after user stops resizing (300ms)
        if not hasattr(self, '_resize_timer'):
            self._resize_timer = QTimer(self)
            self._resize_timer.setSingleShot(True)
            self._resize_timer.timeout.connect(self._load_books)
        self._resize_timer.start(300)

    def closeEvent(self, event):
        """Hide the dialog instead of closing — persist settings."""
        self._config['epub_library_sort'] = self._sort_mode
        self._config['epub_library_card_size'] = self._card_size
        event.ignore()
        self.hide()


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
                        title = title[:47] + "…"
                    chapters.append((title, content))
                except Exception:
                    logger.debug("Skipped chapter: %s", traceback.format_exc())

            # Write to cache (avoids emitting large data through Qt signals)
            _save_epub_cache(self._epub_path, chapters, images)
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

    def __init__(self, epub_path: str, config: dict | None = None, parent=None):
        super().__init__(parent)
        self._epub_path = epub_path
        self._config = config or {}
        self._chapters: list[tuple[str, str]] = []
        self._images: dict[str, bytes] = {}
        # Restore persisted reader settings
        self._font_size = self._config.get('epub_reader_font_size', 14)
        self._line_spacing = self._config.get('epub_reader_line_spacing', 1.8)
        self._theme_index = self._config.get('epub_reader_theme', 0)
        layout_key = self._config.get('epub_reader_layout', LAYOUT_SINGLE)
        self._layout_mode = layout_key if layout_key in (LAYOUT_SCROLL, LAYOUT_SINGLE, LAYOUT_DOUBLE, LAYOUT_ALL) else LAYOUT_SINGLE
        self._current_row = 0
        self._current_page = 0  # viewport-based page for single/double page modes
        self._loader_thread: _EpubLoaderThread | None = None

        self.setWindowTitle(f"📖 {os.path.splitext(os.path.basename(epub_path))[0]}")
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
        self._start_loading()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Toolbar ──
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(10, 6, 10, 6)
        toolbar.setSpacing(6)

        title_lbl = QLabel(f"📖 {os.path.splitext(os.path.basename(self._epub_path))[0]}")
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
        self._layout_combo.addItems(["📄 Single Page", "📖 Double Page (broken)", "📜 Scroll", "📃 Scroll All"])
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

        # TOC sidebar
        self._toc_list = QListWidget()
        self._toc_list.setFixedWidth(220)
        self._toc_list.currentRowChanged.connect(self._on_chapter_selected)
        splitter.addWidget(self._toc_list)

        # Reader area — single browser for scroll/single, an HBox for double-page
        self._reader_stack = QStackedWidget()

        def _make_reader_widget():
            if _HAS_WEBENGINE:
                w = QWebEngineView()
                w.setContextMenuPolicy(Qt.NoContextMenu)
                # Block navigation
                w.setUrl(QUrl("about:blank"))
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
        double_layout.addWidget(self._reader_left)
        double_layout.addWidget(self._reader_right)
        self._reader_stack.addWidget(double_widget)

        splitter.addWidget(self._reader_stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
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

        self._apply_reader_style()

        # Shortcuts (work regardless of child focus)
        QShortcut(QKeySequence(Qt.Key_Left), self, self._prev_chapter)
        QShortcut(QKeySequence(Qt.Key_Right), self, self._next_chapter)
        QShortcut(QKeySequence(Qt.Key_F11), self, self._toggle_fullscreen)
        QShortcut(QKeySequence("Ctrl+="), self, lambda: self._change_font_size(1))
        QShortcut(QKeySequence("Ctrl++"), self, lambda: self._change_font_size(1))
        QShortcut(QKeySequence("Ctrl+-"), self, lambda: self._change_font_size(-1))

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
            QTimer.singleShot(0, lambda: self._on_epub_loaded_from_cache(cached[0], cached[1]))
            return
        self._loader_thread = _EpubLoaderThread(self._epub_path, self)
        self._loader_thread.done.connect(self._on_loader_done)
        self._loader_thread.error.connect(self._on_epub_error)
        self._loader_thread.start()

    @Slot()
    def _on_loader_done(self):
        """Loader finished — read data from cache file (avoids large signal data)."""
        cached = _load_epub_cache(self._epub_path)
        if cached:
            self._on_epub_loaded_from_cache(cached[0], cached[1])
        else:
            self._on_epub_error("Failed to read EPUB cache after loading.")

    def _on_epub_loaded_from_cache(self, chapters, images):
        self._spin_timer.stop()
        self._chapters = chapters
        self._images = images
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
            self._toc_list.setCurrentRow(0)
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
            QListWidget {{ background: {bg}; border: none; border-right: 1px solid {border};
                color: {fg}; font-size: 9pt; padding: 4px; }}
            QListWidget::item {{ padding: 6px 8px; border-radius: 4px; }}
            QListWidget::item:selected {{ background: {border}; color: {t['heading']}; }}
            QListWidget::item:hover {{ background: {t['code_bg']}; }}
        """)
        self._nav_bar.setStyleSheet(f"background: {bg}; border-top: 1px solid {border};")
        self._loading_widget.setStyleSheet(f"background: {bg};")
        self._splitter.setStyleSheet(f"QSplitter::handle {{ background: {border}; width: 1px; }}")

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
        if not ok or self._layout_mode not in (LAYOUT_SINGLE, LAYOUT_DOUBLE):
            return
        if self._layout_mode == LAYOUT_SINGLE:
            self._finalize_single_page()
        elif self._layout_mode == LAYOUT_DOUBLE:
            # Wait for both panes to finish loading
            pending = getattr(self, '_double_loads_pending', 0)
            if pending > 1:
                self._double_loads_pending = pending - 1
                return
            self._double_loads_pending = 0
            self._finalize_double_page()

    def _js_scroll_to(self, browser, page_num):
        """Navigate to a CSS column page by translating the #columns wrapper."""
        if _HAS_WEBENGINE:
            js = (
                f"var c = document.getElementById('columns');"
                f"var w = (typeof _PAGE_W!=='undefined'&&_PAGE_W)?_PAGE_W:window.innerWidth;"
                f"if (c) c.style.transform = 'translateX(' + (-{page_num} * w) + 'px)';"
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
            self._js_scroll_to(self._reader, self._current_page)
            self._js_reveal(self._reader)
            self._update_nav_buttons()
        self._js_page_count(self._reader, on_count)

    def _finalize_double_page(self):
        """After HTML load: get page count and position both panes."""
        def on_count(count):
            self._chapter_page_cache[self._current_row] = int(count)
            self._js_scroll_to(self._reader_left, self._current_page)
            self._js_scroll_to(self._reader_right, self._current_page + 1)
            self._js_reveal(self._reader_left)
            self._js_reveal(self._reader_right)
            self._update_nav_buttons()
        self._js_page_count(self._reader_left, on_count)

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
        if not hasattr(self, '_chapter_page_cache'):
            return
        # Save scroll proportion before clearing cache
        old_count = self._chapter_page_cache.get(self._current_row, 0)
        old_page = self._current_page
        self._chapter_page_cache.clear()
        if self._layout_mode in (LAYOUT_SINGLE, LAYOUT_DOUBLE) and self._chapters:
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
            QTimer.singleShot(100, _on_resize_recount)

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
        super().closeEvent(event)

    # ── Chapter rendering ──────────────────────────────────────────────────

    def _on_chapter_selected(self, row):
        if row < 0 or row >= len(self._chapters):
            return
        self._current_row = row
        self._current_page = 0  # reset pagination when selecting a new chapter
        self._render_current()

    def _process_html(self, html_content: str) -> str:
        """Process chapter HTML: inject inline images as base64 data URIs."""
        try:
            from bs4 import BeautifulSoup
            import base64

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
                    ext = os.path.splitext(src)[1].lower()
                    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                                ".png": "image/png", ".gif": "image/gif",
                                ".svg": "image/svg+xml", ".webp": "image/webp"}
                    mime = mime_map.get(ext, "image/jpeg")
                    b64 = base64.b64encode(image_data).decode("ascii")
                    img_tag["src"] = f"data:{mime};base64,{b64}"
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
                        # Check container's previous sibling for a header
                        prev = container.find_previous_sibling()
                        if prev and prev.name in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
                            prev.extract()
                            container.wrap(wrapper)
                            wrapper.insert(0, prev)
                        else:
                            container.wrap(wrapper)

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
        if paginated:
            return (
                f"<html><head><style>"
                f"* {{ box-sizing: border-box; }}"
                f"html, body {{ margin: 0; padding: 10px 0; overflow: hidden; "
                f"background: {t['bg']}; color: {t['fg']}; }}"
                f"#columns {{ column-fill: auto; column-gap: 0; "
                f"transition: transform 0.3s ease; opacity: 0; "
                f"font-family: 'Georgia', 'Noto Serif', serif; "
                f"font-size: {self._font_size}pt; line-height: {self._line_spacing}; }}"
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
                f"p {{ margin: 0.6em 0; orphans: 2; widows: 2; }}"
                f"a {{ color: {t['link']}; }}"
                f"code {{ background: {t['code_bg']}; padding: 1px 4px; border-radius: 3px; }}"
                f"</style>"
                f"<script>"
                f"var _PAGE_W = 0;"
                f"function _setupColumns() {{"
                f"  var c = document.getElementById('columns');"
                f"  if (!c) return;"
                f"  _PAGE_W = window.innerWidth;"
                f"  c.style.columnWidth = _PAGE_W + 'px';"
                f"  c.style.height = (window.innerHeight - 20) + 'px';"
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
            return (
                f"<html><head><style>"
                f"body {{ background: {t['bg']}; color: {t['fg']}; "
                f"font-family: 'Georgia', 'Noto Serif', serif; "
                f"font-size: {self._font_size}pt; line-height: {self._line_spacing}; "
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
