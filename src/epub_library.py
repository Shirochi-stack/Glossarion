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
import tempfile
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QScrollArea, QWidget, QLineEdit, QFrame, QSplitter, QTextBrowser,
    QListWidget, QListWidgetItem, QMessageBox, QSizePolicy, QToolButton,
    QProgressBar, QApplication
)
from PySide6.QtCore import Qt, QSize, Signal, QThread, QTimer
from PySide6.QtGui import QPixmap, QFont, QIcon, QImage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & Utilities
# ---------------------------------------------------------------------------

def get_library_dir() -> str:
    """Return the dedicated Glossarion Library folder.

    On Windows: ~/Documents/Glossarion/Library/
    On macOS  : ~/Documents/Glossarion/Library/
    """
    docs = Path.home() / "Documents" / "Glossarion" / "Library"
    try:
        docs.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return str(docs)


def _cover_cache_dir() -> str:
    """Temp directory for cached cover thumbnails."""
    d = os.path.join(tempfile.gettempdir(), "Glossarion_CoverCache")
    os.makedirs(d, exist_ok=True)
    return d


def _extract_cover(epub_path: str) -> str | None:
    """Extract cover image from an EPUB, cache it, and return the path."""
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

        # 1. OPF cover metadata
        for meta in book.get_metadata("OPF", "cover"):
            if meta and meta[1]:
                cover_id = meta[1].get("content")
                if cover_id:
                    for item in book.get_items():
                        if item.get_id() == cover_id:
                            cover_data = item.get_content()
                            break
                break

        # 2. First image with "cover" in the name
        if not cover_data:
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_IMAGE and "cover" in item.get_name().lower():
                    cover_data = item.get_content()
                    break

        # 3. Fallback: first image at all
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
        logger.debug("Cover extraction failed for %s: %s", epub_path, exc)
    return None


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def scan_for_epubs(config: dict | None = None) -> list[dict]:
    """Scan output dirs, CWD subfolders, and Library folder for .epub files.

    Returns a deduplicated list of dicts:
        {"name": str, "path": str, "size": int, "mtime": float, "in_library": bool}
    """
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
            in_lib = norm.startswith(library_dir + os.sep) or norm == library_dir
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

    # 1. Library folder (always)
    _walk(library_dir, max_depth=4)

    # 2. Output override directory
    override = os.environ.get("OUTPUT_DIRECTORY") or config.get("output_directory")
    if override and os.path.isdir(override):
        _walk(os.path.abspath(override))

    # 3. CWD subfolders (siblings of translator_gui.py in dev mode)
    cwd = os.getcwd()
    _walk(cwd)

    # Sort by modification time (newest first)
    results.sort(key=lambda r: r["mtime"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Library Dialog
# ---------------------------------------------------------------------------

class _CoverLoader(QThread):
    """Background thread to load a single cover thumbnail."""
    finished = Signal(str, str)  # (epub_path, cover_path_or_empty)

    def __init__(self, epub_path: str, parent=None):
        super().__init__(parent)
        self._epub_path = epub_path

    def run(self):
        cover = _extract_cover(self._epub_path)
        self.finished.emit(self._epub_path, cover or "")


class _BookCard(QFrame):
    """A single clickable card representing an EPUB."""
    clicked = Signal(dict)

    CARD_WIDTH = 160
    COVER_HEIGHT = 200

    def __init__(self, book: dict, parent=None):
        super().__init__(parent)
        self.book = book
        self.setFixedWidth(self.CARD_WIDTH)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            _BookCard {
                background: #1e1e2e;
                border: 1px solid #2a2a3e;
                border-radius: 8px;
            }
            _BookCard:hover {
                border: 1px solid #6c63ff;
                background: #252540;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 8)
        layout.setSpacing(4)

        # Cover placeholder
        self.cover_label = QLabel()
        self.cover_label.setFixedSize(self.CARD_WIDTH - 12, self.COVER_HEIGHT)
        self.cover_label.setAlignment(Qt.AlignCenter)
        self.cover_label.setStyleSheet(
            "background: #2a2a3e; border-radius: 6px; color: #555; font-size: 28pt;"
        )
        self.cover_label.setText("📖")
        layout.addWidget(self.cover_label)

        # Title
        title = book["name"]
        if len(title) > 28:
            title = title[:25] + "…"
        title_lbl = QLabel(title)
        title_lbl.setWordWrap(True)
        title_lbl.setMaximumHeight(40)
        title_lbl.setStyleSheet("color: #e0e0e0; font-size: 9pt; font-weight: bold;")
        layout.addWidget(title_lbl)

        # Size
        size_mb = book["size"] / (1024 * 1024)
        size_str = f"{size_mb:.1f} MB" if size_mb >= 1 else f"{book['size'] / 1024:.0f} KB"
        size_lbl = QLabel(size_str)
        size_lbl.setStyleSheet("color: #888; font-size: 8pt;")
        layout.addWidget(size_lbl)

    def set_cover(self, image_path: str):
        """Set the cover thumbnail from a file path."""
        try:
            pm = QPixmap(image_path)
            if not pm.isNull():
                scaled = pm.scaled(
                    self.CARD_WIDTH - 12, self.COVER_HEIGHT,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.cover_label.setPixmap(scaled)
                self.cover_label.setText("")
        except Exception:
            pass

    def mouseDoubleClickEvent(self, event):
        self.clicked.emit(self.book)
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.book)
        super().mousePressEvent(event)


class EpubLibraryDialog(QDialog):
    """Grid-based EPUB library browser."""

    def __init__(self, config: dict | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📚 Glossarion Library")
        self.resize(900, 650)
        self.setMinimumSize(500, 400)
        self._config = config or {}
        self._books: list[dict] = []
        self._cards: list[_BookCard] = []
        self._cover_threads: list[_CoverLoader] = []
        self._setup_ui()
        self._load_books()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # Header
        header = QHBoxLayout()
        title = QLabel("📚  EPUB Library")
        title.setStyleSheet("font-size: 16pt; font-weight: bold; color: #e0e0e0;")
        header.addWidget(title)
        header.addStretch()

        self._search = QLineEdit()
        self._search.setPlaceholderText("🔍  Filter…")
        self._search.setFixedWidth(220)
        self._search.setStyleSheet(
            "background: #1e1e2e; border: 1px solid #3a3a5e; border-radius: 6px; "
            "padding: 5px 10px; color: #e0e0e0; font-size: 10pt;"
        )
        self._search.textChanged.connect(self._apply_filter)
        header.addWidget(self._search)

        refresh_btn = QPushButton("🔄")
        refresh_btn.setToolTip("Refresh library")
        refresh_btn.setFixedSize(32, 32)
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.setStyleSheet(
            "QPushButton { background: #2a2a3e; border-radius: 6px; font-size: 14pt; border: none; }"
            "QPushButton:hover { background: #3a3a5e; }"
        )
        refresh_btn.clicked.connect(self._load_books)
        header.addWidget(refresh_btn)

        root.addLayout(header)

        # Relocate banner (hidden by default)
        self._relocate_banner = QFrame()
        self._relocate_banner.setStyleSheet(
            "QFrame { background: #1a3a4a; border: 1px solid #2a6a8a; border-radius: 6px; }"
        )
        banner_layout = QHBoxLayout(self._relocate_banner)
        banner_layout.setContentsMargins(12, 8, 12, 8)
        banner_lbl = QLabel("📁  Some EPUBs are outside the Library folder.")
        banner_lbl.setStyleSheet("color: #8ad; font-size: 9pt;")
        banner_layout.addWidget(banner_lbl)
        banner_layout.addStretch()
        relocate_btn = QPushButton("Move to Library")
        relocate_btn.setCursor(Qt.PointingHandCursor)
        relocate_btn.setStyleSheet(
            "QPushButton { background: #17a2b8; color: white; border-radius: 4px; "
            "padding: 4px 12px; font-size: 9pt; font-weight: bold; border: none; }"
            "QPushButton:hover { background: #138496; }"
        )
        relocate_btn.clicked.connect(self._relocate_to_library)
        banner_layout.addWidget(relocate_btn)
        self._relocate_banner.hide()
        root.addWidget(self._relocate_banner)

        # Count label
        self._count_label = QLabel("")
        self._count_label.setStyleSheet("color: #888; font-size: 9pt;")
        root.addWidget(self._count_label)

        # Scrollable grid area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QScrollBar:vertical { width: 8px; background: #1a1a2e; }"
            "QScrollBar::handle:vertical { background: #3a3a5e; border-radius: 4px; }"
        )
        self._grid_container = QWidget()
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setSpacing(12)
        self._grid_layout.setContentsMargins(4, 4, 4, 4)
        scroll.setWidget(self._grid_container)
        root.addWidget(scroll, 1)

        # Empty state
        self._empty_label = QLabel("No EPUB files found.\nTranslate a novel to see it here!")
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet("color: #555; font-size: 12pt; padding: 40px;")
        self._empty_label.hide()
        root.addWidget(self._empty_label)

        # Dialog styling
        self.setStyleSheet("""
            QDialog {
                background: #12121e;
            }
        """)

    # ── Data ───────────────────────────────────────────────────────────────

    def _load_books(self):
        """Scan for EPUBs and populate the grid."""
        # Cleanup old cover threads
        for t in self._cover_threads:
            try:
                t.quit()
                t.wait(100)
            except Exception:
                pass
        self._cover_threads.clear()

        self._books = scan_for_epubs(self._config)
        self._populate_grid(self._books)

    def _populate_grid(self, books: list[dict]):
        """Clear and rebuild the card grid."""
        # Clear existing cards
        for card in self._cards:
            try:
                card.setParent(None)
                card.deleteLater()
            except Exception:
                pass
        self._cards.clear()

        # Remove all items from layout
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)

        if not books:
            self._empty_label.show()
            self._count_label.setText("")
            self._relocate_banner.hide()
            return

        self._empty_label.hide()
        self._count_label.setText(f"{len(books)} EPUB{'s' if len(books) != 1 else ''} found")

        # Check for non-library EPUBs
        outside = [b for b in books if not b.get("in_library")]
        if outside:
            self._relocate_banner.show()
        else:
            self._relocate_banner.hide()

        # Calculate columns based on dialog width
        cols = max(1, (self.width() - 40) // (_BookCard.CARD_WIDTH + 16))

        for idx, book in enumerate(books):
            card = _BookCard(book)
            card.clicked.connect(self._on_card_clicked)
            self._cards.append(card)
            row, col = divmod(idx, cols)
            self._grid_layout.addWidget(card, row, col, Qt.AlignTop)

            # Load cover in background
            loader = _CoverLoader(book["path"], self)
            loader.finished.connect(self._on_cover_loaded)
            self._cover_threads.append(loader)
            loader.start()

        # Add stretch to push cards to the top
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self._grid_layout.addWidget(spacer, len(books) // cols + 1, 0)

    def _on_cover_loaded(self, epub_path: str, cover_path: str):
        """Apply the loaded cover to the matching card."""
        if not cover_path:
            return
        for card in self._cards:
            if card.book["path"] == epub_path:
                card.set_cover(cover_path)
                break

    def _apply_filter(self, text: str):
        """Filter visible cards by search text."""
        query = text.strip().lower()
        if not query:
            self._populate_grid(self._books)
            return
        filtered = [b for b in self._books if query in b["name"].lower()]
        self._populate_grid(filtered)

    def _on_card_clicked(self, book: dict):
        """Open the reader for the selected book."""
        try:
            reader = EpubReaderDialog(book["path"], parent=self)
            reader.exec()
        except Exception as exc:
            QMessageBox.warning(self, "Error", f"Could not open EPUB:\n{exc}")

    # ── Relocate ───────────────────────────────────────────────────────────

    def _relocate_to_library(self):
        """Move EPUBs that are outside the Library folder into it."""
        outside = [b for b in self._books if not b.get("in_library")]
        if not outside:
            return

        names = "\n".join(f"  • {b['name']}.epub" for b in outside[:15])
        if len(outside) > 15:
            names += f"\n  … and {len(outside) - 15} more"

        msg = QMessageBox(self)
        msg.setWindowTitle("Move to Library")
        msg.setText(
            f"Move {len(outside)} EPUB{'s' if len(outside) != 1 else ''} to the Library folder?\n\n"
            f"{names}\n\n"
            f"Destination:\n  {get_library_dir()}"
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
        for book in outside:
            src = book["path"]
            dst = os.path.join(lib_dir, os.path.basename(src))

            # Avoid overwrite
            if os.path.exists(dst):
                base, ext = os.path.splitext(os.path.basename(src))
                counter = 1
                while os.path.exists(dst):
                    dst = os.path.join(lib_dir, f"{base} ({counter}){ext}")
                    counter += 1

            try:
                shutil.move(src, dst)
                moved += 1
            except Exception as exc:
                errors.append(f"{os.path.basename(src)}: {exc}")

        summary = f"Moved {moved} file{'s' if moved != 1 else ''} to Library."
        if errors:
            summary += f"\n\n{len(errors)} error{'s' if len(errors) != 1 else ''}:\n" + "\n".join(errors[:5])
        QMessageBox.information(self, "Done", summary)
        self._load_books()

    def resizeEvent(self, event):
        """Re-layout the grid when the dialog is resized."""
        super().resizeEvent(event)
        if self._books:
            visible = [c.book for c in self._cards if not c.isHidden()] or self._books
            # Only re-layout if column count would change
            new_cols = max(1, (self.width() - 40) // (_BookCard.CARD_WIDTH + 16))
            current_cols = max(1, self._grid_layout.columnCount())
            if new_cols != current_cols and self._cards:
                self._populate_grid(visible)


# ---------------------------------------------------------------------------
# EPUB Reader Dialog
# ---------------------------------------------------------------------------

class EpubReaderDialog(QDialog):
    """Simple EPUB reader with chapter navigation."""

    def __init__(self, epub_path: str, parent=None):
        super().__init__(parent)
        self._epub_path = epub_path
        self._chapters: list[tuple[str, str]] = []  # (title, html_content)
        self._images: dict[str, bytes] = {}  # href -> bytes
        self._font_size = 14
        self._dark_mode = True

        self.setWindowTitle(f"📖 {os.path.splitext(os.path.basename(epub_path))[0]}")
        self.resize(950, 700)
        self.setMinimumSize(600, 400)

        self._setup_ui()
        self._load_epub()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(10, 6, 10, 6)

        title_lbl = QLabel(f"📖 {os.path.splitext(os.path.basename(self._epub_path))[0]}")
        title_lbl.setStyleSheet("font-size: 11pt; font-weight: bold; color: #e0e0e0;")
        title_lbl.setMaximumWidth(500)
        toolbar.addWidget(title_lbl)
        toolbar.addStretch()

        # Font size controls
        font_down = QPushButton("A−")
        font_down.setToolTip("Decrease font size")
        font_down.setFixedSize(36, 28)
        font_down.setCursor(Qt.PointingHandCursor)
        font_down.setStyleSheet(
            "QPushButton { background: #2a2a3e; border-radius: 4px; color: #e0e0e0; font-size: 10pt; font-weight: bold; border: none; }"
            "QPushButton:hover { background: #3a3a5e; }"
        )
        font_down.clicked.connect(lambda: self._change_font_size(-1))
        toolbar.addWidget(font_down)

        self._font_label = QLabel(f"{self._font_size}pt")
        self._font_label.setStyleSheet("color: #888; font-size: 9pt; padding: 0 4px;")
        toolbar.addWidget(self._font_label)

        font_up = QPushButton("A+")
        font_up.setToolTip("Increase font size")
        font_up.setFixedSize(36, 28)
        font_up.setCursor(Qt.PointingHandCursor)
        font_up.setStyleSheet(
            "QPushButton { background: #2a2a3e; border-radius: 4px; color: #e0e0e0; font-size: 10pt; font-weight: bold; border: none; }"
            "QPushButton:hover { background: #3a3a5e; }"
        )
        font_up.clicked.connect(lambda: self._change_font_size(1))
        toolbar.addWidget(font_up)

        # Dark/light toggle
        self._theme_btn = QPushButton("☀️")
        self._theme_btn.setToolTip("Toggle dark/light mode")
        self._theme_btn.setFixedSize(36, 28)
        self._theme_btn.setCursor(Qt.PointingHandCursor)
        self._theme_btn.setStyleSheet(
            "QPushButton { background: #2a2a3e; border-radius: 4px; font-size: 14pt; border: none; }"
            "QPushButton:hover { background: #3a3a5e; }"
        )
        self._theme_btn.clicked.connect(self._toggle_theme)
        toolbar.addWidget(self._theme_btn)

        toolbar_widget = QWidget()
        toolbar_widget.setLayout(toolbar)
        toolbar_widget.setStyleSheet("background: #12121e; border-bottom: 1px solid #2a2a3e;")
        root.addWidget(toolbar_widget)

        # Main content: TOC sidebar + reader
        splitter = QSplitter(Qt.Horizontal)

        # TOC sidebar
        self._toc_list = QListWidget()
        self._toc_list.setFixedWidth(220)
        self._toc_list.setStyleSheet("""
            QListWidget {
                background: #16162a;
                border: none;
                border-right: 1px solid #2a2a3e;
                color: #c0c0d0;
                font-size: 9pt;
                padding: 4px;
            }
            QListWidget::item {
                padding: 6px 8px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background: #2a2a4e;
                color: #e0e0ff;
            }
            QListWidget::item:hover {
                background: #222240;
            }
        """)
        self._toc_list.currentRowChanged.connect(self._on_chapter_selected)
        splitter.addWidget(self._toc_list)

        # Reader
        self._reader = QTextBrowser()
        self._reader.setOpenExternalLinks(False)
        self._reader.setOpenLinks(False)
        self._apply_reader_style()
        splitter.addWidget(self._reader)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

        # Dialog styling
        self.setStyleSheet("QDialog { background: #12121e; }")

    def _apply_reader_style(self):
        """Apply the current theme + font size to the reader."""
        if self._dark_mode:
            bg = "#1a1a2e"
            fg = "#d0d0e0"
            link = "#6c9bd2"
        else:
            bg = "#fafafa"
            fg = "#222222"
            link = "#1a73e8"

        self._reader.setStyleSheet(f"""
            QTextBrowser {{
                background: {bg};
                color: {fg};
                border: none;
                padding: 20px 30px;
                font-size: {self._font_size}pt;
                line-height: 1.7;
            }}
            QTextBrowser a {{ color: {link}; }}
        """)

    def _change_font_size(self, delta: int):
        self._font_size = max(8, min(32, self._font_size + delta))
        self._font_label.setText(f"{self._font_size}pt")
        self._apply_reader_style()

    def _toggle_theme(self):
        self._dark_mode = not self._dark_mode
        self._theme_btn.setText("🌙" if not self._dark_mode else "☀️")
        self._apply_reader_style()

    # ── EPUB Loading ───────────────────────────────────────────────────────

    def _load_epub(self):
        """Load the EPUB and populate the TOC + first chapter."""
        try:
            import ebooklib
            from ebooklib import epub as epub_mod
            from bs4 import BeautifulSoup

            book = epub_mod.read_epub(self._epub_path, options={"ignore_ncx": True})

            # Collect all images for inline rendering
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_IMAGE:
                    self._images[item.get_name()] = item.get_content()
                    # Also store by just the filename for simpler matching
                    basename = os.path.basename(item.get_name())
                    if basename not in self._images:
                        self._images[basename] = item.get_content()

            # Collect document items (chapters)
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                try:
                    content = item.get_content().decode("utf-8", errors="replace")
                    soup = BeautifulSoup(content, "html.parser")
                    text = soup.get_text(strip=True)
                    if not text or len(text) < 10:
                        continue  # Skip empty/near-empty documents

                    # Extract title from <title> or first heading
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
                        # Use item file name
                        title = os.path.splitext(os.path.basename(item.get_name()))[0]
                        title = title.replace("_", " ").replace("-", " ").title()

                    # Truncate long titles
                    if len(title) > 50:
                        title = title[:47] + "…"

                    self._chapters.append((title, content))
                except Exception:
                    pass

            # Populate TOC
            for idx, (title, _) in enumerate(self._chapters):
                item = QListWidgetItem(f"{idx + 1}. {title}")
                self._toc_list.addItem(item)

            # Show first chapter
            if self._chapters:
                self._toc_list.setCurrentRow(0)
            else:
                self._reader.setHtml(
                    "<div style='text-align:center; padding: 60px; color: #888;'>"
                    "<p style='font-size: 16pt;'>📭</p>"
                    "<p>No readable content found in this EPUB.</p></div>"
                )
        except Exception as exc:
            self._reader.setHtml(
                f"<div style='text-align:center; padding: 60px; color: #ff6b6b;'>"
                f"<p style='font-size: 16pt;'>⚠️</p>"
                f"<p>Failed to load EPUB:<br>{exc}</p></div>"
            )

    def _on_chapter_selected(self, row: int):
        """Display the selected chapter in the reader."""
        if row < 0 or row >= len(self._chapters):
            return
        title, html_content = self._chapters[row]

        # Process HTML: inject inline images as base64 data URIs
        try:
            from bs4 import BeautifulSoup
            import base64

            soup = BeautifulSoup(html_content, "html.parser")

            # Replace image src with base64 data URIs
            for img_tag in soup.find_all("img"):
                src = img_tag.get("src", "")
                if not src:
                    continue

                # Try various path resolutions
                image_data = None
                candidates = [
                    src,
                    os.path.basename(src),
                    src.lstrip("../"),
                    src.lstrip("./"),
                ]
                for candidate in candidates:
                    if candidate in self._images:
                        image_data = self._images[candidate]
                        break

                if image_data:
                    # Determine MIME type
                    ext = os.path.splitext(src)[1].lower()
                    mime_map = {
                        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                        ".png": "image/png", ".gif": "image/gif",
                        ".svg": "image/svg+xml", ".webp": "image/webp",
                    }
                    mime = mime_map.get(ext, "image/jpeg")
                    b64 = base64.b64encode(image_data).decode("ascii")
                    img_tag["src"] = f"data:{mime};base64,{b64}"
                    # Set max width for readability
                    style = img_tag.get("style", "")
                    if "max-width" not in style:
                        img_tag["style"] = f"{style}; max-width: 100%; height: auto;"

            processed_html = str(soup)
        except Exception:
            processed_html = html_content

        # Wrap in a styled container
        if self._dark_mode:
            bg, fg = "#1a1a2e", "#d0d0e0"
        else:
            bg, fg = "#fafafa", "#222222"

        styled = (
            f"<html><head><style>"
            f"body {{ background: {bg}; color: {fg}; font-family: 'Georgia', 'Noto Serif', serif; "
            f"font-size: {self._font_size}pt; line-height: 1.8; padding: 10px 20px; max-width: 800px; margin: auto; }}"
            f"h1, h2, h3 {{ color: {'#c0c0ff' if self._dark_mode else '#333'}; }}"
            f"img {{ max-width: 100%; height: auto; border-radius: 4px; margin: 8px 0; }}"
            f"p {{ margin: 0.6em 0; }}"
            f"</style></head><body>{processed_html}</body></html>"
        )

        self._reader.setHtml(styled)
        # Scroll to top
        self._reader.verticalScrollBar().setValue(0)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dlg = EpubLibraryDialog()
    dlg.exec()
