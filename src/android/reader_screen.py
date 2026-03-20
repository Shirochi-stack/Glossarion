# reader_screen.py
"""
EpubReaderGUI — Modern EPUB/TXT reader with full reading controls.
Features: Font adjustments, themes, bookmarks, TOC, search, progress tracking.
"""

import os
import logging
import time

from kivy.properties import ObjectProperty, StringProperty, NumericProperty, ListProperty, BooleanProperty
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.scrollview import ScrollView
from kivy.metrics import dp, sp
from kivy.animation import Animation

from kivymd.uix.screen import MDScreen
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDIconButton
from kivymd.uix.slider import MDSlider
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.navigationdrawer import MDNavigationDrawer, MDNavigationDrawerMenu, MDNavigationDrawerItem
from kivymd.uix.dialog import MDDialog
from kivymd.uix.list import MDList, MDListItem, MDListItemHeadlineText
from kivymd.uix.textfield import MDTextField

logger = logging.getLogger(__name__)

# Theme color presets
READER_THEMES = {
    'dark': {
        'bg': (0.07, 0.07, 0.09, 1),      # #121217
        'text': (0.92, 0.92, 0.92, 1),      # #EBEBEB
        'title_text': (0.7, 0.8, 1.0, 1),   # Blue-tinted
    },
    'light': {
        'bg': (1, 1, 1, 1),                 # #FFFFFF
        'text': (0.12, 0.12, 0.12, 1),       # #1F1F1F
        'title_text': (0.15, 0.3, 0.6, 1),
    },
    'sepia': {
        'bg': (0.97, 0.94, 0.87, 1),        # #F8F1DF
        'text': (0.35, 0.25, 0.15, 1),       # #5A4026
        'title_text': (0.45, 0.3, 0.15, 1),
    },
    'amoled': {
        'bg': (0, 0, 0, 1),                 # Pure black
        'text': (0.85, 0.85, 0.85, 1),       # #D9D9D9
        'title_text': (0.5, 0.7, 1.0, 1),
    },
}

FONT_FAMILIES = {
    'sans-serif': 'Roboto',
    'serif': 'RobotoSlab',
    'monospace': 'RobotoMono',
    'dyslexic': 'OpenDyslexic',
}

KV = '''
<ReaderScreen>:
    MDBoxLayout:
        orientation: 'vertical'
        md_bg_color: root.bg_color

        # Top toolbar
        MDTopAppBar:
            id: top_bar
            md_bg_color: root.bg_color

            MDTopAppBarLeadingButtonContainer:
                MDActionButton:
                    icon: "arrow-left"
                    on_release: root.go_back()

            MDTopAppBarTitle:
                id: title_label
                text: root.book_title
                font_style: "Title"
                role: "small"

            MDTopAppBarTrailingButtonContainer:
                MDActionButton:
                    icon: "bookmark-outline"
                    on_release: root.toggle_bookmark()

                MDActionButton:
                    icon: "magnify"
                    on_release: root.show_search()

                MDActionButton:
                    icon: "table-of-contents"
                    on_release: root.show_toc()

                MDActionButton:
                    icon: "format-font"
                    on_release: root.toggle_settings_panel()

                MDActionButton:
                    icon: "translate"
                    on_release: root.translate_current()

        # Content area
        ScrollView:
            id: content_scroll
            do_scroll_x: False
            bar_width: dp(4)
            effect_cls: 'ScrollEffect'
            on_scroll_y: root._on_scroll(self)

            MDLabel:
                id: content_label
                text: root.current_text
                markup: True
                size_hint_y: None
                height: self.texture_size[1] + dp(40)
                padding: [root.margin_px, dp(16)]
                font_size: root.font_size_px
                line_height: root.line_spacing
                color: root.text_color
                halign: root.text_align

        # Bottom bar: chapter nav + progress
        MDBoxLayout:
            size_hint_y: None
            height: dp(48)
            padding: [dp(8), 0]
            md_bg_color: root.bg_color

            MDIconButton:
                icon: "chevron-left"
                on_release: root.prev_chapter()

            MDLabel:
                id: chapter_label
                text: root.chapter_info
                halign: "center"
                font_style: "Label"
                role: "medium"
                color: root.text_color

            MDIconButton:
                icon: "chevron-right"
                on_release: root.next_chapter()

        # Reading settings panel (slides up from bottom)
        MDBoxLayout:
            id: settings_panel
            orientation: 'vertical'
            size_hint_y: None
            height: 0
            opacity: 0
            padding: [dp(16), dp(8)]
            spacing: dp(8)
            md_bg_color: app.theme_cls.surfaceContainerColor

            # Font size
            MDBoxLayout:
                size_hint_y: None
                height: dp(48)
                spacing: dp(8)

                MDLabel:
                    text: "A"
                    font_size: sp(14)
                    halign: "center"
                    size_hint_x: 0.1
                    color: root.text_color

                MDSlider:
                    id: font_slider
                    min: 12
                    max: 36
                    step: 1
                    value: root.font_size_pt
                    on_value: root.set_font_size(self.value)
                    size_hint_x: 0.7

                MDLabel:
                    text: "A"
                    font_size: sp(28)
                    halign: "center"
                    size_hint_x: 0.1
                    color: root.text_color

            # Font family row
            MDBoxLayout:
                size_hint_y: None
                height: dp(40)
                spacing: dp(4)

                MDIconButton:
                    icon: "format-font"
                    style: "outlined" if root.font_family == "sans-serif" else "standard"
                    on_release: root.set_font_family("sans-serif")

                MDIconButton:
                    icon: "format-text-variant"
                    style: "outlined" if root.font_family == "serif" else "standard"
                    on_release: root.set_font_family("serif")

                MDIconButton:
                    icon: "code-tags"
                    style: "outlined" if root.font_family == "monospace" else "standard"
                    on_release: root.set_font_family("monospace")

            # Line spacing
            MDBoxLayout:
                size_hint_y: None
                height: dp(48)
                spacing: dp(8)

                MDLabel:
                    text: "Line Spacing"
                    size_hint_x: 0.4
                    color: root.text_color

                MDSlider:
                    id: spacing_slider
                    min: 1.0
                    max: 2.5
                    step: 0.1
                    value: root.line_spacing
                    on_value: root.set_line_spacing(self.value)
                    size_hint_x: 0.6

            # Theme row
            MDBoxLayout:
                size_hint_y: None
                height: dp(48)
                spacing: dp(8)

                MDLabel:
                    text: "Theme"
                    size_hint_x: 0.3
                    color: root.text_color

                MDIconButton:
                    icon: "white-balance-sunny"
                    on_release: root.set_theme("light")
                    md_bg_color: (1, 1, 1, 1) if root.reader_theme == "light" else (0.9, 0.9, 0.9, 0.3)

                MDIconButton:
                    icon: "weather-night"
                    on_release: root.set_theme("dark")
                    md_bg_color: (0.2, 0.2, 0.3, 1) if root.reader_theme == "dark" else (0.3, 0.3, 0.3, 0.3)

                MDIconButton:
                    icon: "coffee"
                    on_release: root.set_theme("sepia")
                    md_bg_color: (0.97, 0.94, 0.87, 1) if root.reader_theme == "sepia" else (0.8, 0.7, 0.5, 0.3)

                MDIconButton:
                    icon: "circle"
                    on_release: root.set_theme("amoled")
                    md_bg_color: (0, 0, 0, 1) if root.reader_theme == "amoled" else (0.1, 0.1, 0.1, 0.3)

            # Margins
            MDBoxLayout:
                size_hint_y: None
                height: dp(48)
                spacing: dp(8)

                MDLabel:
                    text: "Margins"
                    size_hint_x: 0.3
                    color: root.text_color

                MDIconButton:
                    icon: "arrow-collapse-horizontal"
                    style: "outlined" if root.margin_preset == "narrow" else "standard"
                    on_release: root.set_margin("narrow")

                MDIconButton:
                    icon: "arrow-left-right"
                    style: "outlined" if root.margin_preset == "medium" else "standard"
                    on_release: root.set_margin("medium")

                MDIconButton:
                    icon: "arrow-expand-horizontal"
                    style: "outlined" if root.margin_preset == "wide" else "standard"
                    on_release: root.set_margin("wide")

            # Alignment
            MDBoxLayout:
                size_hint_y: None
                height: dp(48)
                spacing: dp(8)

                MDLabel:
                    text: "Align"
                    size_hint_x: 0.3
                    color: root.text_color

                MDIconButton:
                    icon: "format-align-left"
                    style: "outlined" if root.text_align == "left" else "standard"
                    on_release: root.set_alignment("left")

                MDIconButton:
                    icon: "format-align-justify"
                    style: "outlined" if root.text_align == "justify" else "standard"
                    on_release: root.set_alignment("justify")

                MDIconButton:
                    icon: "format-align-center"
                    style: "outlined" if root.text_align == "center" else "standard"
                    on_release: root.set_alignment("center")

            # Brightness
            MDBoxLayout:
                size_hint_y: None
                height: dp(48)
                spacing: dp(8)

                MDIconButton:
                    icon: "brightness-4"

                MDSlider:
                    id: brightness_slider
                    min: 0.1
                    max: 1.0
                    step: 0.05
                    value: root.brightness
                    on_value: root.set_brightness(self.value)
'''


class ReaderScreen(MDScreen):
    """EPUB/TXT reader with modern reading controls."""

    app = ObjectProperty(None, allownone=True)
    file_path = StringProperty('')
    book_title = StringProperty('Reader')
    current_text = StringProperty('')
    chapter_info = StringProperty('Chapter 0/0')

    # Reader settings
    font_size_pt = NumericProperty(18)
    font_size_px = NumericProperty(sp(18))
    font_family = StringProperty('sans-serif')
    line_spacing = NumericProperty(1.5)
    text_align = StringProperty('left')
    margin_preset = StringProperty('medium')
    margin_px = NumericProperty(dp(24))
    reader_theme = StringProperty('dark')
    brightness = NumericProperty(1.0)

    # Theme colors (bound to properties for reactive updates)
    bg_color = ListProperty(READER_THEMES['dark']['bg'])
    text_color = ListProperty(READER_THEMES['dark']['text'])

    # Internal state
    chapters = ListProperty([])
    current_chapter_index = NumericProperty(0)

    _settings_visible = BooleanProperty(False)
    _auto_scroll_speed = NumericProperty(0)
    _auto_scroll_event = ObjectProperty(None, allownone=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Builder.load_string(KV)
        self._chapter_titles = []
        self._search_results = []

    def on_enter_data(self, file_path=None, **kwargs):
        """Called when navigating to this screen with data."""
        if file_path and file_path != self.file_path:
            self.file_path = file_path
            self._load_file()

    def _load_file(self):
        """Load an EPUB or TXT file."""
        if not self.file_path or not os.path.isfile(self.file_path):
            self.current_text = "File not found."
            return

        self.book_title = os.path.splitext(os.path.basename(self.file_path))[0]
        ext = os.path.splitext(self.file_path)[1].lower()

        if ext == '.epub':
            self._load_epub()
        elif ext == '.txt':
            self._load_txt()
        else:
            self.current_text = f"Unsupported format: {ext}"

        # Restore reading progress
        self._restore_progress()
        self._apply_reader_settings()

    def _load_epub(self):
        """Load chapters from an EPUB file."""
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup

            book = epub.read_epub(self.file_path, options={'ignore_ncx': True})
            self.chapters = []
            self._chapter_titles = []

            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    content = item.get_content().decode('utf-8', errors='replace')
                    soup = BeautifulSoup(content, 'html.parser')

                    # Extract plain text
                    text = soup.get_text(separator='\n\n')
                    text = text.strip()

                    if not text or len(text) < 10:
                        continue

                    # Try to find a chapter title
                    title = None
                    for tag in ['h1', 'h2', 'h3', 'title']:
                        found = soup.find(tag)
                        if found and found.get_text().strip():
                            title = found.get_text().strip()[:80]
                            break
                    if not title:
                        title = f"Chapter {len(self.chapters) + 1}"

                    self.chapters.append(text)
                    self._chapter_titles.append(title)

            if self.chapters:
                self.current_chapter_index = 0
                self._show_chapter(0)
            else:
                self.current_text = "No readable content found in EPUB."

        except Exception as e:
            logger.error(f"Failed to load EPUB: {e}")
            self.current_text = f"Error loading EPUB:\n{e}"

    def _load_txt(self):
        """Load a TXT file as a single chapter."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            self.chapters = [text]
            self._chapter_titles = [self.book_title]
            self.current_chapter_index = 0
            self._show_chapter(0)
        except Exception as e:
            logger.error(f"Failed to load TXT: {e}")
            self.current_text = f"Error loading file:\n{e}"

    def _show_chapter(self, index):
        """Display a specific chapter."""
        if 0 <= index < len(self.chapters):
            self.current_chapter_index = index
            self.current_text = self.chapters[index]
            self.chapter_info = f"{self._chapter_titles[index]}  ({index + 1}/{len(self.chapters)})"

            # Scroll to top
            try:
                self.ids.content_scroll.scroll_y = 1.0
            except Exception:
                pass

            # Save progress
            self._save_progress()

    def next_chapter(self):
        """Go to the next chapter."""
        if self.current_chapter_index < len(self.chapters) - 1:
            self._show_chapter(self.current_chapter_index + 1)

    def prev_chapter(self):
        """Go to the previous chapter."""
        if self.current_chapter_index > 0:
            self._show_chapter(self.current_chapter_index - 1)

    # ── Reading Settings ──

    def _apply_reader_settings(self):
        """Apply saved reader settings from config."""
        if not self.app:
            return
        cfg = self.app.config_data
        self.font_size_pt = cfg.get('reader_font_size', 18)
        self.font_size_px = sp(self.font_size_pt)
        self.font_family = cfg.get('reader_font_family', 'sans-serif')
        self.line_spacing = cfg.get('reader_line_spacing', 1.5)
        self.text_align = cfg.get('reader_text_align', 'left')
        self.margin_preset = cfg.get('reader_margin', 'medium')
        self.margin_px = self._margin_to_px(self.margin_preset)
        self.reader_theme = cfg.get('reader_theme', 'dark')
        self.brightness = cfg.get('reader_brightness', 1.0)
        self._apply_theme()

    def toggle_settings_panel(self):
        """Show/hide the reading settings panel."""
        panel = self.ids.settings_panel
        self._settings_visible = not self._settings_visible

        if self._settings_visible:
            anim = Animation(height=dp(320), opacity=1, duration=0.25)
        else:
            anim = Animation(height=0, opacity=0, duration=0.2)
        anim.start(panel)

    def set_font_size(self, size):
        """Set font size in points."""
        self.font_size_pt = int(size)
        self.font_size_px = sp(self.font_size_pt)
        self._save_setting('reader_font_size', self.font_size_pt)

    def set_font_family(self, family):
        """Set font family."""
        self.font_family = family
        self._save_setting('reader_font_family', family)

    def set_line_spacing(self, spacing):
        """Set line spacing multiplier."""
        self.line_spacing = round(spacing, 1)
        self._save_setting('reader_line_spacing', self.line_spacing)

    def set_theme(self, theme_name):
        """Set reading theme (dark, light, sepia, amoled)."""
        self.reader_theme = theme_name
        self._apply_theme()
        self._save_setting('reader_theme', theme_name)

    def _apply_theme(self):
        """Apply the current theme colors."""
        theme = READER_THEMES.get(self.reader_theme, READER_THEMES['dark'])
        self.bg_color = list(theme['bg'])
        self.text_color = list(theme['text'])

    def set_margin(self, preset):
        """Set margin preset (narrow, medium, wide)."""
        self.margin_preset = preset
        self.margin_px = self._margin_to_px(preset)
        self._save_setting('reader_margin', preset)

    def _margin_to_px(self, preset):
        """Convert margin preset to pixel value."""
        return {'narrow': dp(8), 'medium': dp(24), 'wide': dp(48)}.get(preset, dp(24))

    def set_alignment(self, align):
        """Set text alignment."""
        self.text_align = align
        self._save_setting('reader_text_align', align)

    def set_brightness(self, value):
        """Set screen brightness (0.1–1.0)."""
        self.brightness = value
        self._save_setting('reader_brightness', value)

        # Apply brightness via window opacity on Android
        try:
            from kivy.utils import platform
            if platform == 'android':
                from jnius import autoclass
                PythonActivity = autoclass('org.kivy.android.PythonActivity')
                activity = PythonActivity.mActivity
                window = activity.getWindow()
                lp = window.getAttributes()
                lp.screenBrightness = float(value)
                activity.runOnUiThread(lambda: window.setAttributes(lp))
        except Exception:
            pass

    def _save_setting(self, key, value):
        """Save a single reader setting to config."""
        if self.app:
            self.app.config_data[key] = value
            from android_config import save_config
            save_config(self.app.config_data)

    # ── Bookmarks ──

    def toggle_bookmark(self):
        """Add or remove a bookmark at the current position."""
        if not self.app:
            return
        from android_config import add_bookmark, get_bookmarks
        bookmarks = get_bookmarks(self.app.config_data, self.file_path)

        # Check if there's already a bookmark for this chapter
        for i, bm in enumerate(bookmarks):
            if bm['chapter'] == self.current_chapter_index:
                from android_config import remove_bookmark
                remove_bookmark(self.app.config_data, self.file_path, i)
                return

        scroll_pos = self.ids.content_scroll.scroll_y
        title = self._chapter_titles[self.current_chapter_index] if self.current_chapter_index < len(self._chapter_titles) else ''
        add_bookmark(self.app.config_data, self.file_path, self.current_chapter_index, scroll_pos, title)

    # ── Table of Contents ──

    def show_toc(self):
        """Show table of contents dialog."""
        if not self._chapter_titles:
            return

        from kivymd.uix.dialog import MDDialog, MDDialogHeadlineText, MDDialogContentContainer

        content = MDList()
        for i, title in enumerate(self._chapter_titles):
            item = MDListItem(
                on_release=lambda x, idx=i: (self._show_chapter(idx), self._toc_dialog.dismiss())
            )
            item.add_widget(MDListItemHeadlineText(
                text=f"{i + 1}. {title}"
            ))
            content.add_widget(item)

        scroll = ScrollView(size_hint_y=None, height=dp(400))
        scroll.add_widget(content)

        self._toc_dialog = MDDialog(
            MDDialogHeadlineText(text="Table of Contents"),
            MDDialogContentContainer(scroll),
        )
        self._toc_dialog.open()

    # ── Search ──

    def show_search(self):
        """Show search dialog."""
        from kivymd.uix.dialog import MDDialog, MDDialogHeadlineText, MDDialogContentContainer, MDDialogButtonContainer
        from kivymd.uix.button import MDButton, MDButtonText

        search_field = MDTextField()
        search_field.hint_text = "Search text..."

        def do_search(*args):
            query = search_field.text.strip()
            if not query:
                return
            self._search_text(query)
            self._search_dialog.dismiss()

        self._search_dialog = MDDialog(
            MDDialogHeadlineText(text="Search"),
            MDDialogContentContainer(search_field),
            MDDialogButtonContainer(
                MDButton(MDButtonText(text="Search"), on_release=do_search),
            ),
        )
        self._search_dialog.open()

    def _search_text(self, query):
        """Search all chapters for a text query."""
        query_lower = query.lower()
        self._search_results = []

        for i, chapter_text in enumerate(self.chapters):
            if query_lower in chapter_text.lower():
                title = self._chapter_titles[i] if i < len(self._chapter_titles) else f"Chapter {i + 1}"
                # Count occurrences
                count = chapter_text.lower().count(query_lower)
                self._search_results.append({
                    'chapter': i,
                    'title': title,
                    'count': count,
                })

        if self._search_results:
            # Jump to first result
            first = self._search_results[0]
            self._show_chapter(first['chapter'])

            # Highlight in text (simple markup)
            highlighted = self.current_text.replace(
                query, f"[color=ffff00]{query}[/color]"
            )
            self.current_text = highlighted

    # ── Progress Tracking ──

    def _on_scroll(self, scroll_view):
        """Handle scroll events for progress tracking."""
        # Debounce save
        if hasattr(self, '_scroll_save_event') and self._scroll_save_event:
            self._scroll_save_event.cancel()
        self._scroll_save_event = Clock.schedule_once(lambda dt: self._save_progress(), 1.0)

    def _save_progress(self):
        """Save reading progress."""
        if not self.app or not self.file_path:
            return
        try:
            scroll_pos = self.ids.content_scroll.scroll_y
        except Exception:
            scroll_pos = 1.0

        # Calculate overall progress
        if len(self.chapters) > 0:
            chapter_progress = (self.current_chapter_index + (1 - scroll_pos)) / len(self.chapters)
            percent = min(100, max(0, chapter_progress * 100))
        else:
            percent = 0

        from android_config import save_reading_progress
        save_reading_progress(
            self.app.config_data, self.file_path,
            self.current_chapter_index, scroll_pos, percent
        )

    def _restore_progress(self):
        """Restore reading progress from config."""
        if not self.app or not self.file_path:
            return
        from android_config import get_reading_progress
        progress = get_reading_progress(self.app.config_data, self.file_path)
        if progress:
            chapter = progress.get('chapter', 0)
            scroll_pos = progress.get('scroll_pos', 1.0)
            if 0 <= chapter < len(self.chapters):
                self._show_chapter(chapter)
                Clock.schedule_once(
                    lambda dt: setattr(self.ids.content_scroll, 'scroll_y', scroll_pos), 0.3
                )

    # ── Navigation ──

    def go_back(self):
        """Navigate back to the library."""
        self._save_progress()
        if self.app:
            self.app.switch_screen('library')

    def translate_current(self):
        """Open translation screen with this file."""
        if self.app and self.file_path:
            self.app.open_translation(self.file_path)

    # ── Auto-scroll ──

    def toggle_auto_scroll(self, speed=0):
        """Toggle auto-scroll mode."""
        if self._auto_scroll_event:
            self._auto_scroll_event.cancel()
            self._auto_scroll_event = None
            self._auto_scroll_speed = 0
        elif speed > 0:
            self._auto_scroll_speed = speed
            self._auto_scroll_event = Clock.schedule_interval(self._do_auto_scroll, 1 / 30)

    def _do_auto_scroll(self, dt):
        """Perform one frame of auto-scrolling."""
        try:
            scroll = self.ids.content_scroll
            new_y = scroll.scroll_y - (self._auto_scroll_speed * 0.001 * dt)
            if new_y <= 0:
                # End of chapter — go to next
                if self.current_chapter_index < len(self.chapters) - 1:
                    self.next_chapter()
                else:
                    self.toggle_auto_scroll()
                    return
            scroll.scroll_y = max(0, new_y)
        except Exception:
            self.toggle_auto_scroll()
