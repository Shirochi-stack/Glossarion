# reader_screen.py
"""
EpubReaderGUI — Modern EPUB/TXT reader with full reading controls.
KivyMD 1.2.0 compatible.  Includes in-reader single-chapter translation
with KO/EN toggle, streaming output overlay, and collapsible thinking panel.
"""

import os
import json
import hashlib
import logging
import threading
import re

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

        # ── Top bar with KO/EN toggle ──
        BoxLayout:
            size_hint_y: None
            height: dp(56)
            padding: [dp(4), dp(4)]
            spacing: dp(4)
            canvas.before:
                Color:
                    rgba: root.bg_color
                Rectangle:
                    pos: self.pos
                    size: self.size

            MDIconButton:
                icon: "arrow-left"
                theme_text_color: "Custom"
                text_color: root.text_color
                on_release: root.go_back()

            # Book title — takes remaining space
            MDLabel:
                text: root.book_title
                shorten: True
                shorten_from: "right"
                font_style: "Subtitle2"
                color: root.text_color
                valign: "center"

            # KO / EN toggle button
            MDFlatButton:
                id: btn_lang
                text: root.viewing_language.upper()
                size_hint: None, None
                size: dp(36), dp(28)
                min_width: 0
                padding: 0
                pos_hint: {"center_y": 0.5}
                font_size: sp(10)
                theme_text_color: "Custom"
                text_color: [0.4, 0.8, 1, 1] if root.viewing_language == 'en' else [1,1,1,1]
                md_bg_color: [0.2, 0.2, 0.24, 1]
                on_release: root.toggle_language()

            # Toolbar icons
            MDIconButton:
                icon: "translate"
                theme_text_color: "Custom"
                text_color: [0.4, 0.8, 1, 1] if root.is_translating else root.text_color
                on_release: root.translate_current_chapter()

            MDIconButton:
                icon: "bookmark-outline"
                theme_text_color: "Custom"
                text_color: root.text_color
                on_release: root.toggle_bookmark()

            MDIconButton:
                icon: "table-of-contents"
                theme_text_color: "Custom"
                text_color: root.text_color
                on_release: root.show_toc()

            MDIconButton:
                icon: "format-font"
                theme_text_color: "Custom"
                text_color: root.text_color
                on_release: root.toggle_settings_panel()

        # ── Main content ──
        ScrollView:
            id: content_scroll
            do_scroll_x: False
            bar_width: dp(4)
            on_scroll_y: root._on_scroll(self)

            BoxLayout:
                id: content_box
                orientation: 'vertical'
                size_hint_y: None
                height: self.minimum_height
                padding: [root.margin_px, dp(16)]

        # ── Thin translate status bar (visible only during translation) ──
        BoxLayout:
            id: translate_status_bar
            size_hint_y: None
            height: 0
            opacity: 0
            padding: [dp(12), dp(2)]
            spacing: dp(4)
            canvas.before:
                Color:
                    rgba: [0.08, 0.08, 0.12, 0.95]
                Rectangle:
                    pos: self.pos
                    size: self.size

            MDLabel:
                text: "Translating..."
                font_style: "Caption"
                theme_text_color: "Custom"
                text_color: [0.4, 0.8, 1, 1]
                size_hint_x: 0.8

            MDIconButton:
                icon: "stop-circle-outline"
                theme_text_color: "Custom"
                text_color: [1, 0.4, 0.4, 1]
                size_hint_x: None
                width: dp(36)
                on_release: root._stop_translation()

        # ── Chapter nav bar ──
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

        # ── Settings panel (slides up) ──
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
    """EPUB/TXT reader with modern reading controls and in-reader translation."""

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

    # ── Translation properties ──
    viewing_language = StringProperty('ko')
    is_translating = BooleanProperty(False)
    streaming_text = StringProperty('')
    thinking_text = StringProperty('')
    show_thinking = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Builder.load_string(KV)
        self._chapter_titles = []
        self._toc_dialog = None
        self._search_dialog = None
        # Translation state
        self._raw_chapters = []          # raw HTML per chapter (for API)
        self._translated_chapters = {}   # {chapter_index: translated text}
        self._stop_event = threading.Event()
        self._translation_thread = None
        self._translations_dir = None    # set when EPUB loaded
        self._stream_label = None        # streaming display label

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
            import re
            import tempfile
            import shutil

            book = epub.read_epub(self.file_path, options={'ignore_ncx': False})
            self.chapters = []
            self._chapter_titles = []
            self._raw_chapters = []

            # Set up translations cache directory
            epub_dir = os.path.dirname(self.file_path)
            epub_name = os.path.splitext(os.path.basename(self.file_path))[0]
            self._translations_dir = os.path.join(epub_dir, '.translations', epub_name)
            os.makedirs(self._translations_dir, exist_ok=True)

            # Extract all images to a temp directory
            self._epub_image_dir = tempfile.mkdtemp(prefix='glossarion_epub_')
            image_map = {}  # maps EPUB href -> local file path
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_IMAGE:
                    img_name = os.path.basename(item.get_name())
                    # Also store by full relative path for matching
                    img_path = os.path.join(self._epub_image_dir, img_name)
                    try:
                        with open(img_path, 'wb') as f:
                            f.write(item.get_content())
                        image_map[item.get_name()] = img_path
                        image_map[img_name] = img_path
                        # Also map without directory prefix
                        parts = item.get_name().replace('\\', '/').split('/')
                        for i in range(len(parts)):
                            key = '/'.join(parts[i:])
                            image_map[key] = img_path
                    except Exception:
                        pass

            # Use spine for correct reading order
            spine_items = []
            items_by_id = {item.get_id(): item for item in book.get_items()}
            for item_id, _linear in book.spine:
                if item_id in items_by_id:
                    item = items_by_id[item_id]
                    if item.get_type() == ebooklib.ITEM_DOCUMENT:
                        spine_items.append(item)

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

                # Remove scripts/styles
                for tag in soup(['script', 'style', 'meta', 'link']):
                    tag.decompose()

                # Build content blocks: list of ('text', str) or ('image', filepath)
                blocks = []
                # Get the item's directory for resolving relative image paths
                item_dir = os.path.dirname(item.get_name()).replace('\\', '/')

                body = soup.find('body') or soup
                current_text_parts = []

                def _flush_text():
                    t = '\n\n'.join(current_text_parts).strip()
                    t = re.sub(r'\n{3,}', '\n\n', t)
                    t = re.sub(r'[ \t]+', ' ', t)
                    if t:
                        blocks.append(('text', t))
                    current_text_parts.clear()

                for element in body.children:
                    if hasattr(element, 'name') and element.name:
                        # Check if this element IS an image or CONTAINS images
                        img_tags = []
                        if element.name == 'img':
                            img_tags = [element]
                        elif element.name == 'svg':
                            # SVG with embedded image
                            img_tags = element.find_all('image')
                        else:
                            img_tags = element.find_all('img')

                        if img_tags:
                            # Flush accumulated text before the image
                            text_before = element.get_text(separator='\n').strip()
                            # Remove text that's part of image alt
                            if text_before:
                                current_text_parts.append(text_before)
                            _flush_text()

                            for img_tag in img_tags:
                                src = img_tag.get('src') or img_tag.get('xlink:href', '')
                                if src:
                                    # Try to resolve image path
                                    src_clean = src.replace('\\', '/')
                                    # Remove leading ../
                                    while src_clean.startswith('../'):
                                        src_clean = src_clean[3:]

                                    img_file = None
                                    # Try exact match, basename match, relative to item
                                    for candidate in [
                                        src_clean,
                                        os.path.basename(src_clean),
                                        f"{item_dir}/{src_clean}" if item_dir else src_clean,
                                    ]:
                                        if candidate in image_map:
                                            img_file = image_map[candidate]
                                            break

                                    if img_file and os.path.isfile(img_file):
                                        blocks.append(('image', img_file))
                        else:
                            # Pure text element
                            t = element.get_text(separator='\n').strip()
                            if t:
                                current_text_parts.append(t)
                    elif hasattr(element, 'string') and element.string:
                        t = element.string.strip()
                        if t:
                            current_text_parts.append(t)

                _flush_text()

                # Skip empty chapters (no text and no images)
                if not blocks or (len(blocks) == 1 and blocks[0][0] == 'text' and len(blocks[0][1].strip()) < 10):
                    # But allow image-only chapters
                    has_images = any(b[0] == 'image' for b in blocks)
                    if not has_images:
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
                    fname = item.get_name()
                    title = os.path.splitext(os.path.basename(fname))[0] if fname else f"Chapter {len(self.chapters) + 1}"

                self.chapters.append(blocks)
                self._chapter_titles.append(title)
                self._raw_chapters.append(content)

            if self.chapters:
                self._load_cached_translations()
                self._show_chapter(0)
            else:
                self.current_text = "No readable content found in this EPUB."
                self._show_text_only(self.current_text)
        except ImportError:
            self.current_text = "Missing library: pip install ebooklib beautifulsoup4"
            self._show_text_only(self.current_text)
        except Exception as e:
            import traceback
            self.current_text = f"Error loading EPUB:\n{e}\n\n{traceback.format_exc()}"
            self._show_text_only(self.current_text)

    def _load_txt(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            self.chapters = [[('text', text)]]
            self._chapter_titles = [self.book_title]
            self._show_chapter(0)
        except Exception as e:
            self.current_text = f"Error: {e}"
            self._show_text_only(self.current_text)

    def _show_text_only(self, text):
        """Simple text-only display for errors or plain text."""
        try:
            box = self.ids.content_box
            box.clear_widgets()
            lbl = self._make_text_label(text)
            box.add_widget(lbl)
        except Exception:
            pass

    def _show_chapter(self, index):
        if not (0 <= index < len(self.chapters)):
            return
        self.current_chapter_index = index
        title = self._chapter_titles[index] if index < len(self._chapter_titles) else ""

        # Chapter info bar
        en_avail = " ✓EN" if index in self._translated_chapters else ""
        self.chapter_info = f"{title}  ({index + 1}/{len(self.chapters)}){en_avail}"

        # Auto-switch language based on translation availability
        if index in self._translated_chapters:
            self.viewing_language = 'en'
        else:
            self.viewing_language = 'ko'
        use_translated = (self.viewing_language == 'en')

        box = self.ids.content_box
        box.clear_widgets()

        try:
            if use_translated and index in self._translated_chapters:
                self._render_translated(box, index)
            else:
                self._render_original(box, index)
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            logger.error(f"_show_chapter crash ch {index}: {err}")
            try:
                self._render_raw_fallback(box, index)
            except Exception:
                # Absolute last resort
                box.clear_widgets()
                lbl = self._make_text_label(f"ERROR rendering ch {index}:\n{err}")
                box.add_widget(lbl)

        # Always scroll to top and save
        try:
            self.ids.content_scroll.scroll_y = 1.0
        except Exception:
            pass
        self._save_progress()

    @staticmethod
    def _sanitize_text(text):
        """Strip Unicode characters that crash Kivy's SDL2 text renderer.

        Korean web novels often contain zero-width spaces, joiners, BOM marks,
        and other invisible Unicode control characters that cause SDL2's text
        provider to produce a solid black texture instead of rendered text.
        """
        if not text:
            return text
        # Remove zero-width and invisible Unicode characters
        remove_chars = (
            '\u200b'   # zero-width space (VERY common in Korean web novels)
            '\u200c'   # zero-width non-joiner
            '\u200d'   # zero-width joiner
            '\u200e'   # left-to-right mark
            '\u200f'   # right-to-left mark
            '\u202a'   # left-to-right embedding
            '\u202b'   # right-to-left embedding
            '\u202c'   # pop directional formatting
            '\u202d'   # left-to-right override
            '\u202e'   # right-to-left override
            '\u2060'   # word joiner
            '\u2061'   # function application
            '\u2062'   # invisible times
            '\u2063'   # invisible separator
            '\u2064'   # invisible plus
            '\ufeff'   # BOM / zero-width no-break space
            '\ufffe'   # invalid
            '\x00'     # NUL
            '\x0b'     # vertical tab
            '\x0c'     # form feed
            '\xad'     # soft hyphen
        )
        cleaned = text
        for ch in remove_chars:
            cleaned = cleaned.replace(ch, '')
        return cleaned

    def _make_text_label(self, text, markup=False):
        """Create a plain Kivy Label for chapter text."""
        from kivy.uix.label import Label

        # Sanitize text to remove chars that break SDL2 rendering
        clean_text = self._sanitize_text(str(text))

        # ALWAYS use markup mode — Kivy's MarkupLabel handles CJK font
        # fallback correctly, while the basic Label path can produce a solid
        # black texture for Korean/Japanese/Chinese text.
        if not markup:
            # Escape [ and ] so they aren't interpreted as markup tags
            clean_text = clean_text.replace('&', '&amp;')
            clean_text = clean_text.replace('[', '&bl;')
            clean_text = clean_text.replace(']', '&br;')

        # Get actual container width
        container_width = 400
        try:
            cw = self.ids.content_box.width
            pad = self.margin_px * 2 if hasattr(self, 'margin_px') else 48
            if cw and cw > 50:
                container_width = cw - pad
        except Exception:
            pass

        lbl = Label(
            text=clean_text,
            markup=True,  # ALWAYS markup — fixes CJK rendering
            size_hint_y=None,
            size_hint_x=1,
            font_size=self.font_size_px,
            line_height=self.line_spacing,
            color=self.text_color,
            halign=self.text_align,
            valign='top',
            text_size=(container_width, None),
        )
        lbl.bind(width=lambda inst, w: setattr(inst, 'text_size', (w, None)) if w > 50 else None)
        lbl.bind(texture_size=lambda inst, val: setattr(inst, 'height', val[1] + dp(20)))
        return lbl

    def _render_translated(self, box, index):
        """Render the translated version of a chapter."""
        translated_text = self._translated_chapters.get(index, '')
        if not translated_text:
            self._render_original(box, index)
            return
        self.current_text = translated_text
        lbl = self._make_text_label(translated_text, markup=True)
        box.add_widget(lbl)

    def _render_original(self, box, index):
        """Render the original KO chapter — simplified, no-nonsense version."""
        # Step 1: Get raw HTML for this chapter and strip tags to get plain text
        chapter_text = ''

        # Primary: extract from raw HTML using simple regex (no bs4 needed)
        if index < len(self._raw_chapters):
            raw_html = self._raw_chapters[index]
            if raw_html:
                # Strip HTML tags with regex
                import re as _re
                text = _re.sub(r'<script[^>]*>.*?</script>', '', raw_html, flags=_re.DOTALL | _re.IGNORECASE)
                text = _re.sub(r'<style[^>]*>.*?</style>', '', text, flags=_re.DOTALL | _re.IGNORECASE)
                text = _re.sub(r'<[^>]+>', '\n', text)
                text = _re.sub(r'&nbsp;', ' ', text)
                text = _re.sub(r'&lt;', '<', text)
                text = _re.sub(r'&gt;', '>', text)
                text = _re.sub(r'&amp;', '&', text)
                text = _re.sub(r'&#\d+;', '', text)
                text = _re.sub(r'\n{3,}', '\n\n', text)
                chapter_text = text.strip()

        # Secondary fallback: extract from blocks
        if not chapter_text:
            blocks = self.chapters[index]
            parts = []
            for b in blocks:
                if isinstance(b, (list, tuple)) and len(b) >= 2:
                    if b[0] == 'text':
                        parts.append(str(b[1]))
                elif isinstance(b, str):
                    parts.append(b)
            chapter_text = '\n\n'.join(parts)

        if not chapter_text.strip():
            chapter_text = '(No text content in this chapter)'

        self.current_text = chapter_text
        logger.info(f"_render_original ch {index}: text_len={len(chapter_text)}")

        # Step 2: Render images from blocks
        blocks = self.chapters[index]
        try:
            from kivy.uix.image import AsyncImage
            for block in blocks:
                if isinstance(block, (list, tuple)) and len(block) >= 2 and block[0] == 'image':
                    img = AsyncImage(
                        source=str(block[1]),
                        size_hint_y=None,
                        height=dp(300),
                        fit_mode='contain',
                        allow_stretch=True,
                    )
                    box.add_widget(img)
        except Exception:
            pass

        # Step 3: ALWAYS add the text
        lbl = self._make_text_label(chapter_text)
        box.add_widget(lbl)

    def _render_raw_fallback(self, box, index):
        """Last-resort fallback."""
        import re as _re
        fallback = '(Could not render chapter)'
        if index < len(self._raw_chapters):
            raw = self._raw_chapters[index]
            if raw:
                text = _re.sub(r'<[^>]+>', ' ', raw)
                text = _re.sub(r'\s+', ' ', text).strip()
                fallback = text[:5000] or '(empty chapter)'
        self.current_text = fallback
        lbl = self._make_text_label(fallback)
        box.add_widget(lbl)

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
        try:
            from android_config import add_bookmark, get_bookmarks, remove_bookmark
            bookmarks = get_bookmarks(self.app.config_data, self.file_path)
            for i, bm in enumerate(bookmarks):
                if bm['chapter'] == self.current_chapter_index:
                    remove_bookmark(self.app.config_data, self.file_path, i)
                    try:
                        from kivymd.toast import toast
                        toast("Bookmark removed")
                    except Exception:
                        pass
                    return
            scroll_pos = self.ids.content_scroll.scroll_y
            title = self._chapter_titles[self.current_chapter_index] if self.current_chapter_index < len(self._chapter_titles) else ''
            add_bookmark(self.app.config_data, self.file_path, self.current_chapter_index, scroll_pos, title)
            try:
                from kivymd.toast import toast
                toast("Bookmark added")
            except Exception:
                pass
        except Exception as e:
            logger.error(f"toggle_bookmark error: {e}")
            try:
                from kivymd.toast import toast
                toast(f"Bookmark error: {e}")
            except Exception:
                pass

    # ── TOC ──

    def show_toc(self):
        if not self._chapter_titles:
            try:
                from kivymd.toast import toast
                toast("No table of contents available")
            except Exception:
                pass
            return
        try:
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
        except Exception as e:
            logger.error(f"show_toc error: {e}")
            try:
                from kivymd.toast import toast
                toast(f"TOC error: {e}")
            except Exception:
                pass

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

    # ══════════════════════════════════════════════════
    #  IN-READER TRANSLATION
    # ══════════════════════════════════════════════════

    def toggle_language(self, lang=None):
        """Cycle between KO and EN views.

        Only switches the displayed content — never triggers translation.
        The translate button (文A icon) handles that.
        """
        idx = self.current_chapter_index

        if self.viewing_language == 'en':
            # Switch back to original KO view
            self.viewing_language = 'ko'
            self._show_chapter(idx)
        else:
            # Switch to EN view
            self.viewing_language = 'en'
            if idx in self._translated_chapters:
                # Completed translation — show it
                self._show_chapter(idx)
            elif self.is_translating and self._stream_label and self.streaming_text:
                # Translation in progress — show streaming output
                self._prepare_streaming_view()
                self._stream_label.text = self._md_to_kivy_markup(self.streaming_text)
            else:
                # No translation yet — just show EN state on the button
                # User can hit the translate button to start one
                pass

    def translate_current_chapter(self):
        """Translate the current chapter using the configured LLM."""
        if self.is_translating:
            return
        if not self.app:
            return

        idx = self.current_chapter_index
        if idx in self._translated_chapters:
            # Already translated — just switch view
            self.viewing_language = 'en'
            self._show_chapter(idx)
            return

        if idx >= len(self._raw_chapters):
            return

        # Get raw HTML for this single chapter
        raw_html = self._raw_chapters[idx]
        if not raw_html or not raw_html.strip():
            # Empty page (cover, images-only, etc.) — nothing to translate
            try:
                from kivymd.toast import toast
                toast("No text to translate on this page")
            except Exception:
                pass
            return

        self.is_translating = True
        self.streaming_text = ''
        self.thinking_text = ''
        self._stop_event.clear()

        # Clear the content area and prepare for streaming
        self._prepare_streaming_view()

        # Show thin status bar only if thinking is enabled
        thinking_on = self.app.config_data.get('reader_enable_thinking', False) if self.app else False
        if thinking_on:
            self._show_translate_status(True)

        # Scroll to top
        try:
            self.ids.content_scroll.scroll_y = 1.0
        except Exception:
            pass

        # Run in background thread
        self._translation_thread = threading.Thread(
            target=self._translation_worker_inline,
            args=(idx, raw_html),
            daemon=True,
        )
        self._translation_thread.start()

    @staticmethod
    def _md_to_kivy_markup(text):
        """Convert markdown formatting to Kivy markup for real-time display.

        Supports: bold, italic, headers (h1-h3), line breaks.
        Kivy uses [b], [i], [size=X], [color=X] tags.
        """
        import re as _re
        # Bold: **text** or __text__
        text = _re.sub(r'\*\*(.+?)\*\*', r'[b]\1[/b]', text)
        text = _re.sub(r'__(.+?)__', r'[b]\1[/b]', text)
        # Italic: *text* or _text_  (but not inside bold markers)
        text = _re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'[i]\1[/i]', text)
        text = _re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'[i]\1[/i]', text)
        # Headers: ### -> size 18, ## -> size 22, # -> size 26
        text = _re.sub(r'^### +(.+)$', r'[size=18][b]\1[/b][/size]', text, flags=_re.MULTILINE)
        text = _re.sub(r'^## +(.+)$', r'[size=22][b]\1[/b][/size]', text, flags=_re.MULTILINE)
        text = _re.sub(r'^# +(.+)$', r'[size=26][b]\1[/b][/size]', text, flags=_re.MULTILINE)
        # Horizontal rules
        text = _re.sub(r'^---+$', r'━━━━━━━━━━━━━━━━━━━━', text, flags=_re.MULTILINE)
        # Inline code: `code`
        text = _re.sub(r'`([^`]+)`', r'[color=80cbc4]\1[/color]', text)
        return text

    def _prepare_streaming_view(self):
        """Clear content area and insert a streaming label for real-time output."""
        try:
            box = self.ids.content_box
            box.clear_widgets()
            # Create a label for streaming output
            self._stream_label = self._make_text_label('', markup=True)
            box.add_widget(self._stream_label)
        except Exception:
            self._stream_label = None

    def _translation_worker_inline(self, chapter_idx, raw_html):
        """Background thread: call the translation API with raw HTML."""
        try:
            from reader_translator import translate_chapter_streaming

            config_data = self.app.config_data.copy() if self.app else {}

            def _on_chunk(text):
                """Real-time streaming — append and render with markdown→Kivy markup."""
                def _update(dt, t=text):
                    self.streaming_text += t
                    if self._stream_label:
                        # Convert accumulated text through markdown→Kivy markup
                        self._stream_label.text = self._md_to_kivy_markup(self.streaming_text)
                    # Auto-scroll to bottom
                    try:
                        self.ids.content_scroll.scroll_y = 0
                    except Exception:
                        pass
                Clock.schedule_once(_update)

            def _on_thinking(text):
                def _update(dt, t=text):
                    self.thinking_text += t
                Clock.schedule_once(_update)

            def _on_complete(translated):
                def _finish(dt, t=translated, ci=chapter_idx):
                    self._translated_chapters[ci] = t
                    self._save_translation_cache(ci, t)
                    self.is_translating = False
                    self.viewing_language = 'en'
                    self._show_chapter(ci)
                    # Auto-hide status bar
                    Clock.schedule_once(lambda dt: self._show_translate_status(False), 1.0)
                Clock.schedule_once(_finish)

            def _on_error(err):
                def _show_err(dt, e=err):
                    if self._stream_label:
                        self._stream_label.text += f'\n[color=ff4444]Error: {e}[/color]'
                    self.is_translating = False
                    self._show_translate_status(False)
                Clock.schedule_once(_show_err)

            translate_chapter_streaming(
                raw_html=raw_html,
                config_data=config_data,
                on_chunk=_on_chunk,
                on_thinking=_on_thinking,
                on_complete=_on_complete,
                on_error=_on_error,
                stop_event=self._stop_event,
            )
        except Exception as e:
            import traceback
            err_msg = str(e)
            def _show_err(dt, m=err_msg):
                if self._stream_label:
                    self._stream_label.text += f'\n[color=ff4444]{m}[/color]'
                self.is_translating = False
                self._show_translate_status(False)
            Clock.schedule_once(_show_err)

    def _stop_translation(self):
        """Cancel an in-progress translation."""
        self._stop_event.set()
        try:
            import unified_api_client
            if hasattr(unified_api_client, 'set_stop_flag'):
                unified_api_client.set_stop_flag(True)
            if hasattr(unified_api_client, 'UnifiedClient'):
                unified_api_client.UnifiedClient._global_cancelled = True
            if hasattr(unified_api_client, 'hard_cancel_all'):
                unified_api_client.hard_cancel_all()
        except Exception:
            pass
        self.is_translating = False
        if self._stream_label:
            self._stream_label.text += '\n[color=ffaa00]Translation cancelled.[/color]'
        self._show_translate_status(False)

    def _show_translate_status(self, show):
        """Animate the thin translate status bar."""
        bar = self.ids.translate_status_bar
        if show:
            Animation(height=dp(36), opacity=1, d=0.2).start(bar)
        else:
            Animation(height=0, opacity=0, d=0.15).start(bar)

    # ── Translation cache ──

    def _get_cache_path(self, chapter_idx):
        """Path to the cached translation file for a chapter."""
        if not self._translations_dir:
            return None
        return os.path.join(self._translations_dir, f'chapter_{chapter_idx}.txt')

    def _save_translation_cache(self, chapter_idx, translated_text):
        """Save translated text to disk."""
        path = self._get_cache_path(chapter_idx)
        if path:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(translated_text)
            except Exception as e:
                logger.error(f"Failed to cache translation: {e}")

    def _load_cached_translations(self):
        """Load any previously cached translations from disk."""
        if not self._translations_dir or not os.path.isdir(self._translations_dir):
            return
        self._translated_chapters = {}
        for fname in os.listdir(self._translations_dir):
            m = re.match(r'chapter_(\d+)\.txt$', fname)
            if m:
                idx = int(m.group(1))
                path = os.path.join(self._translations_dir, fname)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        self._translated_chapters[idx] = f.read()
                except Exception:
                    pass
        if self._translated_chapters:
            logger.info(f"Loaded {len(self._translated_chapters)} cached translations")
