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
from PySide6.QtCore import Qt, QSize, Signal, QThread, QTimer, QSizeF
from PySide6.QtGui import QPixmap, QFont, QIcon, QImage, QCursor, QShortcut, QKeySequence

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


def _cover_cache_dir() -> str:
    d = os.path.join(tempfile.gettempdir(), "Glossarion_CoverCache")
    os.makedirs(d, exist_ok=True)
    return d


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

    def _add(path: str):
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
                        if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(".epub"):
                            _add(entry.path)
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

_SIZE_PRESETS = {
    SIZE_COMPACT: {"card_w": 110, "cover_h": 140, "title_size": "8pt", "title_max_len": 18, "spacing": 3},
    SIZE_NORMAL:  {"card_w": 140, "cover_h": 175, "title_size": "8.5pt", "title_max_len": 24, "spacing": 4},
    SIZE_LARGE:   {"card_w": 180, "cover_h": 225, "title_size": "9pt", "title_max_len": 32, "spacing": 5},
}


class _CoverLoader(QThread):
    finished = Signal(str, str)

    def __init__(self, epub_path: str, parent=None):
        super().__init__(parent)
        self._epub_path = epub_path

    def run(self):
        cover = _extract_cover(self._epub_path)
        self.finished.emit(self._epub_path, cover or "")


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

        size_mb = book["size"] / (1024 * 1024)
        size_str = f"{size_mb:.1f} MB" if size_mb >= 1 else f"{book['size'] / 1024:.0f} KB"
        size_lbl = QLabel(size_str)
        size_lbl.setStyleSheet("color: #888; font-size: 7.5pt;")
        layout.addWidget(size_lbl)

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

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        else:
            super().keyPressEvent(event)

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
        title = QLabel("📚  EPUB Library")
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

        refresh_btn = QPushButton("🔄")
        refresh_btn.setToolTip("Refresh library")
        refresh_btn.setFixedSize(28, 28)
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.setStyleSheet(
            "QPushButton { background: #2a2a3e; border-radius: 6px; font-size: 12pt; border: none; }"
            "QPushButton:hover { background: #3a3a5e; }")
        refresh_btn.clicked.connect(self._load_books)
        header.addWidget(refresh_btn)
        root.addLayout(header)

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
        relocate_btn = QPushButton("Organize into Library")
        relocate_btn.setCursor(Qt.PointingHandCursor)
        relocate_btn.setToolTip("Copy these EPUBs into your Library folder for easy access")
        relocate_btn.setStyleSheet(
            "QPushButton { background: #3a5a7a; color: white; border-radius: 4px; "
            "padding: 3px 10px; font-size: 8.5pt; font-weight: bold; border: none; }"
            "QPushButton:hover { background: #4a6a8a; }")
        relocate_btn.clicked.connect(self._relocate_to_library)
        banner_layout.addWidget(relocate_btn)
        # Undo button (hidden by default, shown after a move)
        self._undo_btn = QPushButton("↩ Undo Move")
        self._undo_btn.setCursor(Qt.PointingHandCursor)
        self._undo_btn.setToolTip("Move the EPUBs back to their original locations")
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
        scroll.setWidget(self._grid_container)
        root.addWidget(scroll, 1)

        self._empty_label = QLabel("No EPUB files found.\nTranslate a novel to see it here!")
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

    def _refresh_view(self):
        query = self._search.text().strip().lower()
        books = self._books
        if query:
            books = [b for b in books if query in b["name"].lower()]
        self._populate_grid(self._sorted_books(books))

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
        self._count_label.setText(f"{len(books)} EPUB{'s' if len(books) != 1 else ''}")

        # Suggestion banner — show count of outside EPUBs
        outside = [b for b in self._books if not b.get("in_library")]
        show_banner = bool(outside) or bool(self._last_move_log)
        if show_banner:
            if outside:
                self._banner_lbl.setText(
                    f"💡  {len(outside)} EPUB{'s' if len(outside) != 1 else ''} could be organized into your Library folder.")
            elif self._last_move_log:
                self._banner_lbl.setText(f"✅  Files moved to Library.")
            self._relocate_banner.show()
            self._undo_btn.setVisible(bool(self._last_move_log))
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

            loader = _CoverLoader(book["path"], self)
            loader.finished.connect(self._on_cover_loaded)
            self._cover_threads.append(loader)
            loader.start()

        # Force card columns to their natural width; extra space goes to a trailing stretch column
        for c in range(cols):
            self._grid_layout.setColumnStretch(c, 0)
        self._grid_layout.setColumnStretch(cols, 1)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._grid_layout.addWidget(spacer, (len(books) - 1) // cols + 1, 0)

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
        try:
            # Set loading cursor
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            reader = EpubReaderDialog(book["path"], config=self._config, parent=self)
            QApplication.restoreOverrideCursor()
            reader.exec()
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
        open_action = menu.addAction("📖  Open in Reader")
        open_action.triggered.connect(lambda: self._on_card_clicked(book))
        menu.addSeparator()
        folder_action = menu.addAction("📂  Open Output Folder")
        folder_action.triggered.connect(lambda: _open_folder_in_explorer(book["path"]))
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

        names = "\n".join(f"  \u2022 {b['name']}.epub" for b in outside[:20])
        if len(outside) > 20:
            names += f"\n  \u2026 and {len(outside) - 20} more"

        msg = QMessageBox(self)
        msg.setWindowTitle("Organize into Library")
        msg.setText(
            f"Move {len(outside)} EPUB{'s' if len(outside) != 1 else ''} into the Library folder?\n\n"
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
        """Move EPUBs back to their original locations."""
        if not self._last_move_log:
            return

        names = "\n".join(
            f"  \u2022 {os.path.basename(dst)}  \u2192  {os.path.dirname(src)}"
            for src, dst in self._last_move_log[:15]
        )
        if len(self._last_move_log) > 15:
            names += f"\n  \u2026 and {len(self._last_move_log) - 15} more"

        msg = QMessageBox(self)
        msg.setWindowTitle("Undo Move")
        msg.setText(
            f"Move {len(self._last_move_log)} EPUB{'s' if len(self._last_move_log) != 1 else ''} back to "
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
        for src, dst in self._last_move_log:
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

        self._last_move_log.clear()

        summary = f"Restored {restored} file{'s' if restored != 1 else ''} to original location{'s' if restored != 1 else ''}." 
        if errors:
            summary += f"\n\n{len(errors)} error{'s' if len(errors) != 1 else ''}:\n" + "\n".join(errors[:5])
        QMessageBox.information(self, "Undo Complete", summary)
        self._load_books()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._cards:
            preset = _SIZE_PRESETS[self._card_size]
            new_cols = max(1, (self.width() - 30) // (preset["card_w"] + preset["spacing"]))
            current_cols = max(1, self._grid_layout.columnCount())
            if new_cols != current_cols:
                self._refresh_view()

    def closeEvent(self, event):
        """Persist library settings back into config."""
        self._config['epub_library_sort'] = self._sort_mode
        self._config['epub_library_card_size'] = self._card_size
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# EPUB Reader — background loader thread
# ---------------------------------------------------------------------------

class _EpubLoaderThread(QThread):
    """Load the EPUB in a background thread so the UI stays responsive."""
    finished = Signal(list, dict)       # (chapters, images)
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

            self.finished.emit(chapters, images)
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
    {"name": "Dark",     "bg": "#1e1e2e", "fg": "#d4d4d4", "heading": "#c8c8f0",
     "link": "#6c9bd2", "code_bg": "#252540", "border": "#2a2a3e"},
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
        if idx >= 0:
            self._spacing_combo.blockSignals(True)
            self._spacing_combo.setCurrentIndex(idx)
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
            QComboBox QAbstractItemView {
                background: #1e1e2e; color: #e0e0e0; selection-background-color: #3a3a5e;
                border: 1px solid #3a3a5e; font-size: 8.5pt;
            }
        """)
        self._layout_combo.currentIndexChanged.connect(self._on_layout_changed)
        toolbar.addWidget(self._layout_combo)

        toolbar.addSpacing(8)

        # Line spacing dropdown
        spacing_lbl = QLabel("↕")
        spacing_lbl.setStyleSheet("color: #888; font-size: 11pt;")
        spacing_lbl.setToolTip("Line spacing")
        toolbar.addWidget(spacing_lbl)

        self._spacing_combo = QComboBox()
        self._spacing_combo.addItems(["1.0", "1.2", "1.4", "1.6", "1.8", "2.0", "2.4"])
        self._spacing_combo.setCurrentIndex(4)  # 1.8 default
        self._spacing_combo.setFixedWidth(58)
        self._spacing_combo.setCursor(Qt.PointingHandCursor)
        self._spacing_combo.setStyleSheet("""
            QComboBox {
                background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #e0e0e0; font-size: 8.5pt; padding: 3px 6px;
            }
            QComboBox:hover { border-color: #6c63ff; }
            QComboBox::drop-down { border: none; width: 18px; }
            QComboBox QAbstractItemView {
                background: #1e1e2e; color: #e0e0e0; selection-background-color: #3a3a5e;
                border: 1px solid #3a3a5e;
            }
        """)
        self._spacing_combo.currentTextChanged.connect(self._on_spacing_changed)
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
            QComboBox QAbstractItemView {
                background: #1e1e2e; color: #e0e0e0; selection-background-color: #3a3a5e;
                border: 1px solid #3a3a5e;
            }
        """)
        self._theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        toolbar.addWidget(self._theme_combo)

        toolbar_widget = QWidget()
        toolbar_widget.setLayout(toolbar)
        toolbar_widget.setStyleSheet("background: #12121e; border-bottom: 1px solid #2a2a3e;")
        root.addWidget(toolbar_widget)

        # ── Loading indicator ──
        self._loading_widget = QWidget()
        loading_layout = QVBoxLayout(self._loading_widget)
        loading_layout.setAlignment(Qt.AlignCenter)
        # Spinning icon via animated label
        loading_icon = QLabel()
        icon_path = _find_halgakos_icon()
        if icon_path:
            pm = QPixmap(icon_path)
            if not pm.isNull():
                loading_icon.setPixmap(pm.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            loading_icon.setText("📖")
            loading_icon.setStyleSheet("font-size: 32pt;")
        loading_icon.setAlignment(Qt.AlignCenter)
        loading_layout.addWidget(loading_icon)
        loading_text = QLabel("Loading EPUB…")
        loading_text.setAlignment(Qt.AlignCenter)
        loading_text.setStyleSheet("color: #888; font-size: 11pt; padding-top: 8px;")
        loading_layout.addWidget(loading_text)
        self._loading_widget.setStyleSheet("background: #12121e;")
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
        self._toc_list.setStyleSheet("""
            QListWidget { background: #16162a; border: none; border-right: 1px solid #2a2a3e;
                color: #c0c0d0; font-size: 9pt; padding: 4px; }
            QListWidget::item { padding: 6px 8px; border-radius: 4px; }
            QListWidget::item:selected { background: #2a2a4e; color: #e0e0ff; }
            QListWidget::item:hover { background: #222240; }
        """)
        self._toc_list.currentRowChanged.connect(self._on_chapter_selected)
        splitter.addWidget(self._toc_list)

        # Reader area — single browser for scroll/single, an HBox for double-page
        self._reader_stack = QStackedWidget()

        # Page 0: single reader (for scroll, single page, all-scroll)
        self._reader = QTextBrowser()
        self._reader.setOpenExternalLinks(False)
        self._reader.setOpenLinks(False)
        self._reader.viewport().installEventFilter(self)
        self._reader_stack.addWidget(self._reader)

        # Page 1: double-page layout (two QTextBrowsers side by side)
        double_widget = QWidget()
        self._double_widget = double_widget  # keep ref for styling
        double_layout = QHBoxLayout(double_widget)
        double_layout.setContentsMargins(0, 0, 0, 0)
        double_layout.setSpacing(2)
        self._reader_left = QTextBrowser()
        self._reader_left.setOpenExternalLinks(False)
        self._reader_left.setOpenLinks(False)
        self._reader_left.viewport().installEventFilter(self)
        self._reader_right = QTextBrowser()
        self._reader_right.setOpenExternalLinks(False)
        self._reader_right.setOpenLinks(False)
        self._reader_right.viewport().installEventFilter(self)
        double_layout.addWidget(self._reader_left)
        double_layout.addWidget(self._reader_right)
        self._reader_stack.addWidget(double_widget)

        splitter.addWidget(self._reader_stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
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
        self._nav_bar.setStyleSheet("background: #12121e; border-top: 1px solid #2a2a3e;")
        self._nav_bar.hide()
        content_layout.addWidget(self._nav_bar)

        self._content_widget.hide()
        root.addWidget(self._content_widget, 1)

        self._apply_reader_style()
        self.setStyleSheet("QDialog { background: #12121e; }")

        # Shortcuts (work regardless of child focus)
        QShortcut(QKeySequence(Qt.Key_Left), self, self._prev_chapter)
        QShortcut(QKeySequence(Qt.Key_Right), self, self._next_chapter)
        QShortcut(QKeySequence(Qt.Key_F11), self, self._toggle_fullscreen)

    # ── Event filter (block wheel scroll in paginated modes) ──────────────

    def eventFilter(self, obj, event):
        """Block wheel events on reader viewports in paginated modes."""
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.Wheel and self._layout_mode in (LAYOUT_SINGLE, LAYOUT_DOUBLE):
            return True  # swallow the event
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

    # ── Loading ────────────────────────────────────────────────────────────

    def _start_loading(self):
        self._loading_widget.show()
        self._content_widget.hide()
        self._loader_thread = _EpubLoaderThread(self._epub_path, self)
        self._loader_thread.finished.connect(self._on_epub_loaded)
        self._loader_thread.error.connect(self._on_epub_error)
        self._loader_thread.start()

    def _on_epub_loaded(self, chapters, images):
        self._chapters = chapters
        self._images = images
        self._chapter_page_cache = {}  # {chapter_index: page_count}
        self._loaded_chapter = -1  # track which chapter's HTML is loaded

        self._toc_list.clear()
        for idx, (title, _) in enumerate(self._chapters):
            item = QListWidgetItem(f"{idx + 1}. {title}")
            self._toc_list.addItem(item)

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
        css = f"""
            QTextBrowser {{
                background: {bg}; color: {t['fg']}; border: none;
                padding: 20px 30px; font-size: {self._font_size}pt;
            }}
            QTextBrowser a {{ color: {t['link']}; }}
            QScrollBar:vertical {{ width: 8px; background: {bg}; }}
            QScrollBar::handle:vertical {{ background: {t['border']}; border-radius: 4px; min-height: 20px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """
        self._reader.setStyleSheet(css)
        self._reader_left.setStyleSheet(css + f"QTextBrowser {{ border-right: 1px solid {t['border']}; }}")
        self._reader_right.setStyleSheet(css)
        # Match surrounding containers to theme background so there's no "smudge"
        self._reader_stack.setStyleSheet(f"QStackedWidget {{ background: {bg}; border: none; }}")
        self._double_widget.setStyleSheet(f"background: {bg};")

    def _change_font_size(self, delta):
        self._font_size = max(8, min(32, self._font_size + delta))
        self._font_label.setText(f"{self._font_size}pt")
        self._apply_reader_style()
        self._chapter_page_cache.clear()  # pages change with font size
        self._loaded_chapter = -1  # force re-render
        self._render_current()

    def _on_spacing_changed(self, text):
        try:
            self._line_spacing = float(text)
        except ValueError:
            self._line_spacing = 1.8
        self._chapter_page_cache.clear()
        self._loaded_chapter = -1
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
        """Re-render based on current layout mode and selected chapter."""
        if not self._chapters:
            return

        row = self._current_row

        if self._layout_mode == LAYOUT_ALL:
            self._reader_stack.setCurrentIndex(0)
            self._nav_bar.hide()
            self._reader.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self._reader.setUpdatesEnabled(False)
            all_html = ""
            for idx, (title, content) in enumerate(self._chapters):
                processed = self._process_html(content)
                all_html += f"<h2 style='color: {self._get_theme()['heading']}; border-bottom: 1px solid {self._get_theme()['border']}; padding-bottom: 6px; margin-top: 30px;'>Chapter {idx + 1}: {title}</h2>\n{processed}\n<hr style='border: none; border-top: 1px solid {self._get_theme()['border']}; margin: 20px 0;'>"
            self._reader.setHtml(self._wrap_html(all_html))
            self._reader.verticalScrollBar().setValue(0)
            self._reader.setUpdatesEnabled(True)
            self._loaded_chapter = -1
            self._toc_list.blockSignals(True)
            self._toc_list.setCurrentRow(-1)
            self._toc_list.blockSignals(False)

        elif self._layout_mode == LAYOUT_DOUBLE:
            self._reader_stack.setCurrentIndex(1)
            self._nav_bar.show()
            self._reader_left.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self._reader_right.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            if row < len(self._chapters):
                self._reader_left.setUpdatesEnabled(False)
                self._reader_right.setUpdatesEnabled(False)
                html = self._process_html(self._chapters[row][1])
                full = self._wrap_html(html)
                self._reader_left.setHtml(full)
                self._reader_right.setHtml(full)
                self._loaded_chapter = row
                QTimer.singleShot(50, self._finalize_double_page)
            else:
                self._reader_left.setHtml(self._wrap_html(""))
                self._reader_right.setHtml(self._wrap_html(""))
            self._update_nav_buttons()

        elif self._layout_mode == LAYOUT_SINGLE:
            self._reader_stack.setCurrentIndex(0)
            self._nav_bar.show()
            self._reader.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            if row < len(self._chapters):
                self._reader.setUpdatesEnabled(False)
                html = self._process_html(self._chapters[row][1])
                self._reader.setHtml(self._wrap_html(html))
                self._loaded_chapter = row
                QTimer.singleShot(50, self._finalize_single_page)
            self._update_nav_buttons()

        else:  # LAYOUT_SCROLL
            self._reader_stack.setCurrentIndex(0)
            self._nav_bar.hide()
            self._reader.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            if row < len(self._chapters):
                html = self._process_html(self._chapters[row][1])
                self._reader.setHtml(self._wrap_html(html))
                self._reader.verticalScrollBar().setValue(0)
                self._loaded_chapter = row

    # ── Pagination helpers (no-flicker) ────────────────────────────────────

    def _page_count(self, browser):
        """Get page count using QTextDocument layout."""
        doc = browser.document()
        vp = browser.viewport()
        w, h = vp.width(), vp.height()
        if w <= 0 or h <= 0:
            return 1
        doc.setPageSize(QSizeF(w, h))
        return max(1, doc.pageCount())

    def _finalize_single_page(self):
        """After HTML load: position page, cache count, re-enable updates."""
        try:
            vp = self._reader.viewport()
            h = vp.height()
            if h > 0:
                doc = self._reader.document()
                doc.setPageSize(QSizeF(vp.width(), h))
                self._chapter_page_cache[self._current_row] = max(1, doc.pageCount())
                self._reader.verticalScrollBar().setValue(self._current_page * h)
        finally:
            self._reader.setUpdatesEnabled(True)
        self._update_nav_buttons()

    def _finalize_double_page(self):
        """After HTML load: position both panes, cache count, re-enable."""
        try:
            vp = self._reader_left.viewport()
            h = vp.height()
            if h > 0:
                for browser in (self._reader_left, self._reader_right):
                    browser.document().setPageSize(QSizeF(vp.width(), h))
                self._chapter_page_cache[self._current_row] = max(1, self._reader_left.document().pageCount())
                self._reader_left.verticalScrollBar().setValue(self._current_page * h)
                self._reader_right.verticalScrollBar().setValue((self._current_page + 1) * h)
        finally:
            self._reader_left.setUpdatesEnabled(True)
            self._reader_right.setUpdatesEnabled(True)
        self._update_nav_buttons()

    def _scroll_to_page_single(self):
        """Scroll single-page reader to current page (no HTML reload)."""
        vp = self._reader.viewport()
        h = vp.height()
        if h > 0:
            self._reader.document().setPageSize(QSizeF(vp.width(), h))
            self._reader.verticalScrollBar().setValue(self._current_page * h)
        self._update_nav_buttons()

    def _scroll_to_page_double(self):
        """Scroll double-page panes to current page (no HTML reload)."""
        vp = self._reader_left.viewport()
        h = vp.height()
        if h > 0:
            for b in (self._reader_left, self._reader_right):
                b.document().setPageSize(QSizeF(vp.width(), h))
            self._reader_left.verticalScrollBar().setValue(self._current_page * h)
            self._reader_right.verticalScrollBar().setValue((self._current_page + 1) * h)
        self._update_nav_buttons()

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
                    style = img_tag.get("style", "")
                    if "max-width" not in style:
                        img_tag["style"] = f"{style}; max-width: 100%; height: auto;"

            return str(soup)
        except Exception:
            logger.debug("HTML processing failed: %s", traceback.format_exc())
            return html_content

    def _wrap_html(self, body_html: str) -> str:
        """Wrap processed HTML in a full styled document."""
        t = self._get_theme()
        return (
            f"<html><head><style>"
            f"body {{ background: {t['bg']}; color: {t['fg']}; "
            f"font-family: 'Georgia', 'Noto Serif', serif; "
            f"font-size: {self._font_size}pt; line-height: {self._line_spacing}; "
            f"padding: 10px 20px; max-width: 800px; margin: auto; }}"
            f"h1, h2, h3 {{ color: {t['heading']}; }}"
            f"img {{ max-width: 100%; height: auto; border-radius: 4px; margin: 8px 0; }}"
            f"p {{ margin: 0.6em 0; }}"
            f"code {{ background: {t['code_bg']}; padding: 1px 4px; border-radius: 3px; }}"
            f"</style></head><body>{body_html}</body></html>"
        )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dlg = EpubLibraryDialog()
    dlg.exec()
