# library_screen.py
"""
LibraryGUI — EPUB/TXT file browser with cover thumbnails.
Scans configured directories and displays books as a Material Design card list.
"""

import os
import io
import logging

from kivy.properties import ObjectProperty, ListProperty
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.uix.image import Image

from kivymd.uix.screen import MDScreen
from kivymd.uix.list import (
    MDList, MDListItem, MDListItemLeadingAvatar,
    MDListItemHeadlineText, MDListItemSupportingText,
)
from kivymd.uix.button import MDFabButton
from kivymd.uix.dialog import MDDialog, MDDialogHeadlineText, MDDialogContentContainer, MDDialogButtonContainer
from kivymd.uix.textfield import MDTextField
from kivymd.uix.menu import MDDropdownMenu
from kivy.uix.scrollview import ScrollView
from kivy.lang import Builder

from android_file_utils import scan_for_books, get_documents_dir, get_downloads_dir

logger = logging.getLogger(__name__)

KV = '''
<LibraryScreen>:
    MDBoxLayout:
        orientation: 'vertical'

        # Top bar
        MDTopAppBar:
            MDTopAppBarTitle:
                text: "Glossarion Library"

            MDTopAppBarTrailingButtonContainer:
                MDActionButton:
                    icon: "folder-search-outline"
                    on_release: root.browse_folder()

                MDActionButton:
                    icon: "refresh"
                    on_release: root.load_books()

        # Book list
        ScrollView:
            id: scroll_view

            MDList:
                id: book_list
                spacing: "4dp"
                padding: "8dp"

        # Empty state label
        MDLabel:
            id: empty_label
            text: "No books found.\\nTap + to add EPUB or TXT files."
            halign: "center"
            theme_text_color: "Hint"
            font_style: "Body"
            role: "large"
            opacity: 0

    # Floating action button
    MDFabButton:
        icon: "plus"
        pos_hint: {"right": 0.95, "y": 0.02}
        style: "standard"
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
        """Scan directories for EPUB/TXT files and populate the list."""
        if not self.app:
            Clock.schedule_once(lambda dt: self.load_books(), 0.5)
            return

        self.books.clear()

        # Scan configured directories
        scan_dirs = self.app.config_data.get('library_scan_dirs', [])
        default_dirs = [get_documents_dir()]

        # Also check Downloads
        downloads = get_downloads_dir()
        if os.path.isdir(downloads):
            default_dirs.append(downloads)

        all_dirs = list(set(scan_dirs + default_dirs))

        for d in all_dirs:
            if os.path.isdir(d):
                found = scan_for_books(d)
                self.books.extend(found)

        # Deduplicate by path
        seen = set()
        unique = []
        for b in self.books:
            if b['path'] not in seen:
                seen.add(b['path'])
                unique.append(b)
        self.books = unique

        # Update UI
        self._populate_list()

    def _populate_list(self):
        """Populate the MDList with book items."""
        book_list = self.ids.book_list
        book_list.clear_widgets()

        empty_label = self.ids.empty_label
        if not self.books:
            empty_label.opacity = 1
            return
        empty_label.opacity = 0

        for book in self.books:
            item = self._create_book_item(book)
            book_list.add_widget(item)

    def _create_book_item(self, book):
        """Create a list item widget for a book."""
        # Format file size
        size_mb = book['size'] / (1024 * 1024)
        if size_mb >= 1:
            size_str = f"{size_mb:.1f} MB"
        else:
            size_str = f"{book['size'] / 1024:.0f} KB"

        ext_label = book['ext'].upper().replace('.', '')
        subtitle = f"{ext_label} · {size_str} · {os.path.dirname(book['path'])}"

        item = MDListItem(
            on_release=lambda x, b=book: self._on_book_tap(b),
            # on_long_press not available in MDListItem — use touch events
        )

        # Leading avatar (cover or icon)
        avatar = MDListItemLeadingAvatar()
        cover_path = self._get_cover_image(book)
        if cover_path:
            avatar.source = cover_path
        else:
            # Use Halgakos icon as placeholder
            icon_path = self._get_placeholder_icon()
            if icon_path:
                avatar.source = icon_path
        item.add_widget(avatar)

        # Text content
        item.add_widget(MDListItemHeadlineText(text=book['name']))
        item.add_widget(MDListItemSupportingText(text=subtitle))

        return item

    def _get_cover_image(self, book):
        """Extract cover image from EPUB, or return placeholder for TXT.
        
        Returns path to cover image file, or None.
        """
        if book['ext'] != '.epub':
            return None

        # Check cache
        if book['path'] in self._cover_cache:
            return self._cover_cache[book['path']]

        try:
            import ebooklib
            from ebooklib import epub

            epub_book = epub.read_epub(book['path'], options={'ignore_ncx': True})

            # Try to get cover image
            cover_data = None

            # Method 1: Check metadata for cover image ID
            cover_id = None
            for meta in epub_book.get_metadata('OPF', 'cover'):
                if meta and meta[1]:
                    cover_id = meta[1].get('content')
                    break

            if cover_id:
                for item in epub_book.get_items():
                    if item.get_id() == cover_id:
                        cover_data = item.get_content()
                        break

            # Method 2: Find first image item
            if not cover_data:
                for item in epub_book.get_items():
                    if item.get_type() == ebooklib.ITEM_IMAGE:
                        name = item.get_name().lower()
                        if 'cover' in name:
                            cover_data = item.get_content()
                            break

            if not cover_data:
                for item in epub_book.get_items():
                    if item.get_type() == ebooklib.ITEM_IMAGE:
                        cover_data = item.get_content()
                        break

            if cover_data:
                # Save cover to cache directory
                from android_file_utils import get_cache_dir
                cache_dir = os.path.join(get_cache_dir(), 'covers')
                os.makedirs(cache_dir, exist_ok=True)

                import hashlib
                name_hash = hashlib.md5(book['path'].encode()).hexdigest()[:12]
                cover_path = os.path.join(cache_dir, f"{name_hash}.jpg")

                with open(cover_path, 'wb') as f:
                    f.write(cover_data)

                self._cover_cache[book['path']] = cover_path
                return cover_path

        except Exception as e:
            logger.debug(f"Could not extract cover from {book['name']}: {e}")

        return None

    def _get_placeholder_icon(self):
        """Get the Halgakos.png placeholder icon path."""
        candidates = [
            os.path.join(os.path.dirname(__file__), 'assets', 'Halgakos.png'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Halgakos.png'),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
        return None

    def _on_book_tap(self, book):
        """Handle tapping on a book — open in reader."""
        if self.app:
            self.app.open_reader(book['path'])

    def pick_file(self):
        """Open file chooser to pick EPUB/TXT files."""
        try:
            from plyer import filechooser
            filechooser.open_file(
                title="Select EPUB or TXT file",
                filters=[
                    ("Books", "*.epub", "*.txt"),
                ],
                on_selection=self._on_file_selected,
                multiple=False,
            )
        except ImportError:
            logger.warning("plyer not available for file chooser")
        except Exception as e:
            logger.error(f"File picker error: {e}")

    def _on_file_selected(self, selection):
        """Handle file selection from file chooser."""
        if selection:
            path = selection[0] if isinstance(selection, list) else selection
            if os.path.isfile(path):
                # Optionally copy to documents dir
                from android_file_utils import copy_file_to_documents
                copied = copy_file_to_documents(path)
                self.load_books()

                # Open in reader
                if self.app:
                    self.app.open_reader(copied)

    def browse_folder(self):
        """Let user select a folder to scan for books."""
        try:
            from plyer import filechooser
            filechooser.choose_dir(
                title="Select folder to scan",
                on_selection=self._on_folder_selected,
            )
        except ImportError:
            logger.warning("plyer not available for folder chooser")
        except Exception as e:
            logger.error(f"Folder picker error: {e}")

    def _on_folder_selected(self, selection):
        """Handle folder selection."""
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
