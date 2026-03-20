# library_screen.py
"""
LibraryGUI — EPUB/TXT file browser with cover thumbnails.
KivyMD 1.2.0 compatible.
"""

import os
import logging

from kivy.properties import ObjectProperty, ListProperty
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.lang import Builder

from kivymd.uix.screen import MDScreen
from kivymd.uix.list import TwoLineAvatarListItem, ImageLeftWidget
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.uix.button import MDFloatingActionButton
from kivymd.uix.label import MDLabel

from android_file_utils import scan_for_books, get_library_dir

logger = logging.getLogger(__name__)

KV = '''
<LibraryScreen>:
    BoxLayout:
        orientation: 'vertical'

        MDTopAppBar:
            title: "Glossarion Library"
            right_action_items: [["folder-search-outline", lambda x: root.browse_folder()], ["refresh", lambda x: root.load_books()]]
            elevation: 2
            md_bg_color: app.theme_cls.primary_color

        ScrollView:
            id: scroll_view

            MDList:
                id: book_list
                spacing: dp(2)

        MDLabel:
            id: empty_label
            text: "No books found.\\nTap + to add EPUB or TXT files."
            halign: "center"
            theme_text_color: "Hint"
            font_style: "Body1"
            size_hint_y: None
            height: 0
            opacity: 0

    MDFloatingActionButton:
        icon: "plus"
        pos_hint: {"right": 0.95, "y": 0.02}
        elevation: 4
        on_release: root.pick_file()
'''


class LibraryScreen(MDScreen):
    """Library screen showing all EPUB and TXT files."""

    app = ObjectProperty(None, allownone=True)
    books = ListProperty([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Builder.load_string(KV)
        self._cover_cache = {}

    def load_books(self):
        """Scan the dedicated Library folder for translated EPUB/TXT files."""
        if not self.app:
            Clock.schedule_once(lambda dt: self.load_books(), 0.5)
            return

        self.books.clear()

        # Only scan the dedicated Library folder (translated EPUBs)
        lib_dir = get_library_dir()

        if os.path.isdir(lib_dir):
            found = scan_for_books(lib_dir)
            self.books.extend(found)
            # Also scan subdirectories (for _output folders)
            try:
                for entry in os.scandir(lib_dir):
                    if entry.is_dir():
                        sub_found = scan_for_books(entry.path)
                        self.books.extend(sub_found)
            except (PermissionError, OSError):
                pass

        # Also scan any user-configured extra dirs
        scan_dirs = self.app.config_data.get('library_scan_dirs', [])
        for d in scan_dirs:
            if os.path.isdir(d) and d != lib_dir:
                found = scan_for_books(d)
                self.books.extend(found)

        # Deduplicate
        seen = set()
        unique = []
        for b in self.books:
            if b['path'] not in seen:
                seen.add(b['path'])
                unique.append(b)
        self.books = unique
        self._populate_list()

    def _populate_list(self):
        """Populate the MDList."""
        book_list = self.ids.book_list
        book_list.clear_widgets()

        empty_label = self.ids.empty_label
        if not self.books:
            empty_label.opacity = 1
            empty_label.height = dp(200)
            return
        empty_label.opacity = 0
        empty_label.height = 0

        for book in self.books:
            item = self._create_book_item(book)
            book_list.add_widget(item)

    def _create_book_item(self, book):
        """Create a TwoLineAvatarListItem for a book."""
        size_mb = book['size'] / (1024 * 1024)
        size_str = f"{size_mb:.1f} MB" if size_mb >= 1 else f"{book['size'] / 1024:.0f} KB"
        ext_label = book['ext'].upper().replace('.', '')
        subtitle = f"{ext_label} · {size_str}"

        # Shorten long names with ellipsis
        display_name = book['name']
        if len(display_name) > 45:
            display_name = display_name[:42] + '...'

        item = TwoLineAvatarListItem(
            text=display_name,
            secondary_text=subtitle,
            on_release=lambda x, b=book: self._on_book_tap(b),
        )

        # Cover image or placeholder
        cover_path = self._get_cover_image(book)
        if not cover_path:
            cover_path = self._get_placeholder_icon()

        if cover_path:
            avatar = ImageLeftWidget(source=cover_path)
            item.add_widget(avatar)

        return item

    def _get_cover_image(self, book):
        """Extract cover from EPUB or return None."""
        if book['ext'].lower() != '.epub':
            return None
        if not os.path.isfile(book['path']):
            return None
        if book['path'] in self._cover_cache:
            return self._cover_cache[book['path']]

        try:
            import ebooklib
            from ebooklib import epub

            epub_book = epub.read_epub(book['path'], options={'ignore_ncx': True})
            cover_data = None

            # Try metadata cover
            for meta in epub_book.get_metadata('OPF', 'cover'):
                if meta and meta[1]:
                    cover_id = meta[1].get('content')
                    if cover_id:
                        for item in epub_book.get_items():
                            if item.get_id() == cover_id:
                                cover_data = item.get_content()
                                break
                    break

            # Fallback: first image with "cover" in name
            if not cover_data:
                for item in epub_book.get_items():
                    if item.get_type() == ebooklib.ITEM_IMAGE:
                        if 'cover' in item.get_name().lower():
                            cover_data = item.get_content()
                            break

            # Fallback: first image
            if not cover_data:
                for item in epub_book.get_items():
                    if item.get_type() == ebooklib.ITEM_IMAGE:
                        cover_data = item.get_content()
                        break

            if cover_data:
                from android_file_utils import get_cache_dir
                import hashlib
                cache_dir = os.path.join(get_cache_dir(), 'covers')
                os.makedirs(cache_dir, exist_ok=True)
                name_hash = hashlib.md5(book['path'].encode()).hexdigest()[:12]
                cover_path = os.path.join(cache_dir, f"{name_hash}.jpg")
                with open(cover_path, 'wb') as f:
                    f.write(cover_data)
                self._cover_cache[book['path']] = cover_path
                return cover_path
        except Exception as e:
            logger.debug(f"Cover extraction failed for {book['name']}: {e}")
        return None

    def _get_placeholder_icon(self):
        """Get Halgakos placeholder icon."""
        for p in [
            os.path.join(os.path.dirname(__file__), 'assets', 'Halgakos.png'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Halgakos.png'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Halgakos.ico'),
        ]:
            if os.path.isfile(p):
                return p
        return None

    def _on_book_tap(self, book):
        if self.app:
            self.app.open_reader(book['path'])

    def pick_file(self):
        """Open file chooser."""
        try:
            from plyer import filechooser
            filechooser.open_file(
                title="Select EPUB or TXT file",
                filters=[("Books", "*.epub", "*.txt")],
                on_selection=self._on_file_selected,
                multiple=False,
            )
        except Exception as e:
            logger.error(f"File picker error: {e}")

    def _on_file_selected(self, selection):
        if selection:
            path = selection[0] if isinstance(selection, list) else selection
            if os.path.isfile(path):
                from android_file_utils import copy_file_to_documents
                copied = copy_file_to_documents(path)
                self.load_books()
                if self.app:
                    self.app.open_reader(copied)

    def browse_folder(self):
        try:
            from plyer import filechooser
            filechooser.choose_dir(
                title="Select folder to scan",
                on_selection=self._on_folder_selected,
            )
        except Exception as e:
            logger.error(f"Folder picker error: {e}")

    def _on_folder_selected(self, selection):
        if selection:
            folder = selection[0] if isinstance(selection, list) else selection
            if os.path.isdir(folder) and self.app:
                scan_dirs = self.app.config_data.get('library_scan_dirs', [])
                if folder not in scan_dirs:
                    scan_dirs.append(folder)
                    self.app.config_data['library_scan_dirs'] = scan_dirs
                    from android_config import save_config
                    save_config(self.app.config_data)
                self.load_books()
