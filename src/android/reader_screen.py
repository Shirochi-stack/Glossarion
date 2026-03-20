# reader_screen.py
"""
EpubReaderGUI — Modern EPUB/TXT reader with full reading controls.
KivyMD 1.2.0 compatible.
"""

import os
import logging

from kivy.properties import (ObjectProperty, StringProperty, NumericProperty,
                              ListProperty, BooleanProperty)
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.metrics import dp, sp
from kivy.animation import Animation
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView

from kivymd.uix.screen import MDScreen
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDIconButton, MDFlatButton, MDRaisedButton
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.uix.slider import MDSlider
from kivymd.uix.dialog import MDDialog
from kivymd.uix.list import OneLineListItem, MDList
from kivymd.uix.textfield import MDTextField

logger = logging.getLogger(__name__)

READER_THEMES = {
    'dark': {'bg': [0.07, 0.07, 0.09, 1], 'text': [0.92, 0.92, 0.92, 1]},
    'light': {'bg': [1, 1, 1, 1], 'text': [0.12, 0.12, 0.12, 1]},
    'sepia': {'bg': [0.97, 0.94, 0.87, 1], 'text': [0.35, 0.25, 0.15, 1]},
    'amoled': {'bg': [0, 0, 0, 1], 'text': [0.85, 0.85, 0.85, 1]},
}

KV = '''
<ReaderScreen>:
    BoxLayout:
        orientation: 'vertical'
        canvas.before:
            Color:
                rgba: root.bg_color
            Rectangle:
                pos: self.pos
                size: self.size

        MDTopAppBar:
            id: top_bar
            title: root.book_title
            left_action_items: [["arrow-left", lambda x: root.go_back()]]
            right_action_items: [["bookmark-outline", lambda x: root.toggle_bookmark()], ["magnify", lambda x: root.show_search()], ["table-of-contents", lambda x: root.show_toc()], ["format-font", lambda x: root.toggle_settings_panel()], ["translate", lambda x: root.translate_current()]]
            md_bg_color: root.bg_color
            specific_text_color: root.text_color
            elevation: 0

        ScrollView:
            id: content_scroll
            do_scroll_x: False
            bar_width: dp(4)
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

        # Chapter nav bar
        BoxLayout:
            size_hint_y: None
            height: dp(48)
            padding: [dp(8), 0]
            canvas.before:
                Color:
                    rgba: root.bg_color
                Rectangle:
                    pos: self.pos
                    size: self.size

            MDIconButton:
                icon: "chevron-left"
                theme_text_color: "Custom"
                text_color: root.text_color
                on_release: root.prev_chapter()

            MDLabel:
                id: chapter_label
                text: root.chapter_info
                halign: "center"
                font_style: "Caption"
                color: root.text_color

            MDIconButton:
                icon: "chevron-right"
                theme_text_color: "Custom"
                text_color: root.text_color
                on_release: root.next_chapter()

        # Settings panel (slides up)
        BoxLayout:
            id: settings_panel
            orientation: 'vertical'
            size_hint_y: None
            height: 0
            opacity: 0
            padding: [dp(16), dp(8)]
            spacing: dp(8)
            canvas.before:
                Color:
                    rgba: app.theme_cls.bg_dark
                Rectangle:
                    pos: self.pos
                    size: self.size

            # Font size
            BoxLayout:
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

            # Line spacing
            BoxLayout:
                size_hint_y: None
                height: dp(48)
                spacing: dp(8)

                MDLabel:
                    text: "Line Spacing"
                    size_hint_x: 0.4
                    color: root.text_color

                MDSlider:
                    min: 1.0
                    max: 2.5
                    step: 0.1
                    value: root.line_spacing
                    on_value: root.set_line_spacing(self.value)
                    size_hint_x: 0.6

            # Themes
            BoxLayout:
                size_hint_y: None
                height: dp(48)
                spacing: dp(8)

                MDLabel:
                    text: "Theme"
                    size_hint_x: 0.25
                    color: root.text_color

                MDIconButton:
                    icon: "white-balance-sunny"
                    on_release: root.set_theme("light")

                MDIconButton:
                    icon: "weather-night"
                    on_release: root.set_theme("dark")

                MDIconButton:
                    icon: "coffee"
                    on_release: root.set_theme("sepia")

                MDIconButton:
                    icon: "circle"
                    on_release: root.set_theme("amoled")

            # Margins
            BoxLayout:
                size_hint_y: None
                height: dp(48)
                spacing: dp(8)

                MDLabel:
                    text: "Margins"
                    size_hint_x: 0.3
                    color: root.text_color

                MDFlatButton:
                    text: "Narrow"
                    on_release: root.set_margin("narrow")

                MDFlatButton:
                    text: "Medium"
                    on_release: root.set_margin("medium")

                MDFlatButton:
                    text: "Wide"
                    on_release: root.set_margin("wide")

            # Alignment
            BoxLayout:
                size_hint_y: None
                height: dp(48)
                spacing: dp(8)

                MDLabel:
                    text: "Align"
                    size_hint_x: 0.3
                    color: root.text_color

                MDIconButton:
                    icon: "format-align-left"
                    on_release: root.set_alignment("left")

                MDIconButton:
                    icon: "format-align-justify"
                    on_release: root.set_alignment("justify")

                MDIconButton:
                    icon: "format-align-center"
                    on_release: root.set_alignment("center")

            # Brightness
            BoxLayout:
                size_hint_y: None
                height: dp(48)
                spacing: dp(8)

                MDIconButton:
                    icon: "brightness-4"

                MDSlider:
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

    font_size_pt = NumericProperty(14)
    font_size_px = NumericProperty(sp(14))
    font_family = StringProperty('sans-serif')
    line_spacing = NumericProperty(1.5)
    text_align = StringProperty('left')
    margin_preset = StringProperty('medium')
    margin_px = NumericProperty(dp(24))
    reader_theme = StringProperty('dark')
    brightness = NumericProperty(1.0)

    bg_color = ListProperty(READER_THEMES['dark']['bg'])
    text_color = ListProperty(READER_THEMES['dark']['text'])

    chapters = ListProperty([])
    current_chapter_index = NumericProperty(0)
    _settings_visible = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Builder.load_string(KV)
        self._chapter_titles = []
        self._toc_dialog = None
        self._search_dialog = None

    def on_enter_data(self, file_path=None, **kwargs):
        if file_path and file_path != self.file_path:
            self.file_path = file_path
            self._load_file()

    def _load_file(self):
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
            self.current_text = f"Unsupported: {ext}"
        self._restore_progress()
        self._apply_reader_settings()

    def _load_epub(self):
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup

            book = epub.read_epub(self.file_path, options={'ignore_ncx': False})
            self.chapters = []
            self._chapter_titles = []

            # Use spine for correct reading order
            spine_items = []
            items_by_id = {item.get_id(): item for item in book.get_items()}
            for item_id, _linear in book.spine:
                if item_id in items_by_id:
                    item = items_by_id[item_id]
                    if item.get_type() == ebooklib.ITEM_DOCUMENT:
                        spine_items.append(item)

            # Fallback: if spine is empty, use all document items
            if not spine_items:
                spine_items = [
                    item for item in book.get_items()
                    if item.get_type() == ebooklib.ITEM_DOCUMENT
                ]

            for item in spine_items:
                raw = item.get_content()

                # Try multiple encodings
                content = None
                for enc in ('utf-8', 'utf-8-sig', 'euc-kr', 'cp949', 'shift_jis',
                            'gb18030', 'big5', 'latin-1'):
                    try:
                        content = raw.decode(enc)
                        break
                    except (UnicodeDecodeError, LookupError):
                        continue
                if content is None:
                    content = raw.decode('utf-8', errors='replace')

                soup = BeautifulSoup(content, 'html.parser')

                # Extract visible text (skip scripts/styles)
                for tag in soup(['script', 'style', 'meta', 'link']):
                    tag.decompose()

                text = soup.get_text(separator='\n\n').strip()
                # Clean up excessive whitespace
                import re
                text = re.sub(r'\n{3,}', '\n\n', text)
                text = re.sub(r'[ \t]+', ' ', text)

                if not text or len(text.strip()) < 10:
                    continue

                # Extract chapter title from headings
                title = None
                for tag_name in ['h1', 'h2', 'h3', 'title']:
                    found = soup.find(tag_name)
                    if found:
                        t = found.get_text().strip()
                        if t and len(t) > 1:
                            title = t[:80]
                            break
                if not title:
                    # Use filename as fallback
                    fname = item.get_name()
                    title = os.path.splitext(os.path.basename(fname))[0] if fname else f"Chapter {len(self.chapters) + 1}"

                self.chapters.append(text)
                self._chapter_titles.append(title)

            if self.chapters:
                self._show_chapter(0)
            else:
                self.current_text = "No readable content found in this EPUB."
        except ImportError:
            self.current_text = "Missing library: pip install ebooklib beautifulsoup4"
        except Exception as e:
            import traceback
            self.current_text = f"Error loading EPUB:\n{e}\n\n{traceback.format_exc()}"

    def _load_txt(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            self.chapters = [text]
            self._chapter_titles = [self.book_title]
            self._show_chapter(0)
        except Exception as e:
            self.current_text = f"Error: {e}"

    def _show_chapter(self, index):
        if 0 <= index < len(self.chapters):
            self.current_chapter_index = index
            self.current_text = self.chapters[index]
            title = self._chapter_titles[index] if index < len(self._chapter_titles) else ""
            self.chapter_info = f"{title}  ({index + 1}/{len(self.chapters)})"
            try:
                self.ids.content_scroll.scroll_y = 1.0
            except Exception:
                pass
            self._save_progress()

    def next_chapter(self):
        if self.current_chapter_index < len(self.chapters) - 1:
            self._show_chapter(self.current_chapter_index + 1)

    def prev_chapter(self):
        if self.current_chapter_index > 0:
            self._show_chapter(self.current_chapter_index - 1)

    # ── Settings ──

    def _apply_reader_settings(self):
        if not self.app:
            return
        cfg = self.app.config_data
        self.font_size_pt = cfg.get('reader_font_size', 14)
        self.font_size_px = sp(self.font_size_pt)
        self.line_spacing = cfg.get('reader_line_spacing', 1.5)
        self.text_align = cfg.get('reader_text_align', 'left')
        self.margin_preset = cfg.get('reader_margin', 'medium')
        self.margin_px = self._margin_to_px(self.margin_preset)
        self.reader_theme = cfg.get('reader_theme', 'dark')
        self.brightness = cfg.get('reader_brightness', 1.0)
        self._apply_theme()

    def toggle_settings_panel(self):
        panel = self.ids.settings_panel
        self._settings_visible = not self._settings_visible
        if self._settings_visible:
            Animation(height=dp(320), opacity=1, d=0.25).start(panel)
        else:
            Animation(height=0, opacity=0, d=0.2).start(panel)

    def set_font_size(self, size):
        self.font_size_pt = int(size)
        self.font_size_px = sp(self.font_size_pt)
        self._save_setting('reader_font_size', self.font_size_pt)

    def set_line_spacing(self, spacing):
        self.line_spacing = round(spacing, 1)
        self._save_setting('reader_line_spacing', self.line_spacing)

    def set_theme(self, name):
        self.reader_theme = name
        self._apply_theme()
        self._save_setting('reader_theme', name)

    def _apply_theme(self):
        theme = READER_THEMES.get(self.reader_theme, READER_THEMES['dark'])
        self.bg_color = list(theme['bg'])
        self.text_color = list(theme['text'])

    def set_margin(self, preset):
        self.margin_preset = preset
        self.margin_px = self._margin_to_px(preset)
        self._save_setting('reader_margin', preset)

    def _margin_to_px(self, preset):
        return {'narrow': dp(8), 'medium': dp(24), 'wide': dp(48)}.get(preset, dp(24))

    def set_alignment(self, align):
        self.text_align = align
        self._save_setting('reader_text_align', align)

    def set_brightness(self, value):
        self.brightness = value
        self._save_setting('reader_brightness', value)
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
        if self.app:
            self.app.config_data[key] = value
            from android_config import save_config
            save_config(self.app.config_data)

    # ── Bookmarks ──

    def toggle_bookmark(self):
        if not self.app:
            return
        from android_config import add_bookmark, get_bookmarks, remove_bookmark
        bookmarks = get_bookmarks(self.app.config_data, self.file_path)
        for i, bm in enumerate(bookmarks):
            if bm['chapter'] == self.current_chapter_index:
                remove_bookmark(self.app.config_data, self.file_path, i)
                return
        scroll_pos = self.ids.content_scroll.scroll_y
        title = self._chapter_titles[self.current_chapter_index] if self.current_chapter_index < len(self._chapter_titles) else ''
        add_bookmark(self.app.config_data, self.file_path, self.current_chapter_index, scroll_pos, title)

    # ── TOC ──

    def show_toc(self):
        if not self._chapter_titles:
            return
        from kivy.uix.scrollview import ScrollView
        from kivy.uix.boxlayout import BoxLayout

        scroll = ScrollView(size_hint_y=None, height=dp(400))
        toc_list = MDList()
        for i, title in enumerate(self._chapter_titles):
            toc_list.add_widget(
                OneLineListItem(
                    text=f"{i + 1}. {title}",
                    on_release=lambda x, idx=i: self._toc_jump(idx),
                )
            )
        scroll.add_widget(toc_list)

        container = BoxLayout(orientation='vertical', size_hint_y=None)
        container.height = dp(400)
        container.add_widget(scroll)

        self._toc_dialog = MDDialog(
            title="Table of Contents",
            type="custom",
            content_cls=container,
        )
        self._toc_dialog.open()

    def _toc_jump(self, idx):
        self._show_chapter(idx)
        if self._toc_dialog:
            self._toc_dialog.dismiss()

    # ── Search ──

    def show_search(self):
        self._search_field = MDTextField(hint_text="Search text...")
        self._search_dialog = MDDialog(
            title="Search",
            type="custom",
            content_cls=self._search_field,
            buttons=[
                MDFlatButton(text="Cancel", on_release=lambda x: self._search_dialog.dismiss()),
                MDRaisedButton(text="Search", on_release=lambda x: self._do_search()),
            ],
        )
        self._search_dialog.open()

    def _do_search(self):
        query = self._search_field.text.strip()
        if not query:
            return
        self._search_dialog.dismiss()
        query_lower = query.lower()
        for i, chapter_text in enumerate(self.chapters):
            if query_lower in chapter_text.lower():
                self._show_chapter(i)
                self.current_text = self.current_text.replace(
                    query, f"[color=ffff00]{query}[/color]"
                )
                break

    # ── Progress ──

    def _on_scroll(self, scroll_view):
        if hasattr(self, '_scroll_save_event') and self._scroll_save_event:
            self._scroll_save_event.cancel()
        self._scroll_save_event = Clock.schedule_once(lambda dt: self._save_progress(), 1.0)

    def _save_progress(self):
        if not self.app or not self.file_path:
            return
        try:
            scroll_pos = self.ids.content_scroll.scroll_y
        except Exception:
            scroll_pos = 1.0
        percent = 0
        if len(self.chapters) > 0:
            percent = min(100, max(0, ((self.current_chapter_index + (1 - scroll_pos)) / len(self.chapters)) * 100))
        from android_config import save_reading_progress
        save_reading_progress(self.app.config_data, self.file_path, self.current_chapter_index, scroll_pos, percent)

    def _restore_progress(self):
        if not self.app or not self.file_path:
            return
        from android_config import get_reading_progress
        progress = get_reading_progress(self.app.config_data, self.file_path)
        if progress:
            ch = progress.get('chapter', 0)
            sp_val = progress.get('scroll_pos', 1.0)
            if 0 <= ch < len(self.chapters):
                self._show_chapter(ch)
                Clock.schedule_once(lambda dt: setattr(self.ids.content_scroll, 'scroll_y', sp_val), 0.3)

    def go_back(self):
        self._save_progress()
        if self.app:
            self.app.switch_screen('library')

    def translate_current(self):
        if self.app and self.file_path:
            self.app.open_translation(self.file_path)
