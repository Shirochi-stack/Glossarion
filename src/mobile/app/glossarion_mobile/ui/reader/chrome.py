"""ReaderChrome + SelectionChipRow (UI_SPEC §3.11 "Chrome" / "Selection", §5.8).

Top bar (translucent ``surface`` at 92%): back · chapter title over book title ·
``SegmentedButton`` Original · Translated · Bilingual ("Orig · Trans · Both"
under 400 dp; shown only when the book has a second flavour, like the desktop
Raw pill) · search · ⋯. Bottom bar: the chapter ``Slider`` with
"Ch 12/48 · 43%" and "Page 3/9" (paged layouts), then ◀ previous chapter ·
☰ Chapters · Aa · 🌐 Translate (or "🛰️ Live view" while a live run is open) ·
▶ next chapter. A centre tap toggles both bars with a 150 ms fade.

``SelectionChipRow`` floats above (or below) the page selection: Copy ·
Google Translate → {language} (Original mode) or Define on web · Add to
glossary · Ask in chat.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Mapping, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components._handlers import call_handler
from glossarion_mobile.ui.reader import model as rm
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["ReaderChrome", "SelectionChipRow"]

FADE_MS = tokens.MOTION["state_ms"]  # 150 ms


def _icon_button(icon: Any, tooltip: str, handler: Callable[..., Any], key: str) -> ft.IconButton:
    return ft.IconButton(icon=icon, tooltip=tooltip, on_click=handler, size_constraints=HIT_TARGET, key=key)


class ReaderChrome:
    def __init__(
        self,
        *,
        on_back: Callable[..., Any],
        on_mode: Callable[[str], Any],
        on_search: Callable[..., Any],
        on_more: Callable[..., Any],
        on_prev: Callable[..., Any],
        on_next: Callable[..., Any],
        on_chapters: Callable[..., Any],
        on_aa: Callable[..., Any],
        on_translate: Callable[..., Any],
        on_slider: Callable[[int], Any],
        narrow: bool = False,
    ) -> None:
        self.on_mode = on_mode
        self.on_slider = on_slider
        self.visible = True
        self._fade_task: Optional[asyncio.Task] = None
        self.narrow = narrow
        self.chapter_title = ft.Text("", theme_style=ft.TextThemeStyle.TITLE_SMALL, max_lines=1,
                                     overflow=ft.TextOverflow.ELLIPSIS, key="reader-chapter-title")
        self.book_title = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_SMALL, max_lines=1,
                                  overflow=ft.TextOverflow.ELLIPSIS, color=ft.Colors.ON_SURFACE_VARIANT,
                                  key="reader-book-title")
        labels = rm.MODE_SHORT_LABELS if narrow else rm.MODE_LABELS
        self.segments = {mode: ft.Segment(value=mode, label=ft.Text(labels[mode])) for mode in rm.READER_MODES}
        self.mode_buttons = ft.SegmentedButton(
            segments=[self.segments[m] for m in (rm.ORIGINAL, rm.TRANSLATED, rm.BILINGUAL)],
            selected=[rm.TRANSLATED],
            show_selected_icon=False,
            on_change=self._on_mode,
            visible=False,
            key="reader-mode",
        )
        self.search_button = _icon_button(ft.Icons.SEARCH, "Search", on_search, "reader-search")
        self.more_button = _icon_button(ft.Icons.MORE_VERT, "More", on_more, "reader-more")
        self.top = ft.Container(
            bgcolor=ft.Colors.with_opacity(0.92, ft.Colors.SURFACE),
            padding=ft.Padding.only(left=4, right=4, top=4, bottom=4),
            animate_opacity=FADE_MS,
            opacity=1.0,
            content=ft.SafeArea(
                content=ft.Column(
                    [
                        ft.Row(
                            [
                                _icon_button(ft.Icons.ARROW_BACK, "Back", on_back, "reader-back"),
                                ft.Column([self.chapter_title, self.book_title], spacing=0, expand=True, tight=True),
                                self.search_button,
                                self.more_button,
                            ],
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            spacing=2,
                        ),
                        ft.Row([self.mode_buttons], alignment=ft.MainAxisAlignment.CENTER),
                    ],
                    spacing=2,
                    tight=True,
                ),
                avoid_intrusions_bottom=False,
            ),
            key="reader-top",
        )
        self.slider = ft.Slider(min=0, max=1, value=0, divisions=1, expand=True, label="{value}",
                                on_change_end=self._on_slider, key="reader-slider")
        self.progress_text = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_MEDIUM, key="reader-progress")
        self.page_text = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_MEDIUM, color=ft.Colors.ON_SURFACE_VARIANT,
                                 key="reader-page")
        self.prev_button = _icon_button(ft.Icons.CHEVRON_LEFT, "Previous chapter", on_prev, "reader-prev")
        self.next_button = _icon_button(ft.Icons.CHEVRON_RIGHT, "Next chapter", on_next, "reader-next")
        self.chapters_button = _icon_button(ft.Icons.TOC, "Chapters", on_chapters, "reader-chapters")
        self.aa_button = _icon_button(ft.Icons.TEXT_FIELDS, "Aa", on_aa, "reader-aa")
        self.translate_button = ft.TextButton(content="🌐 Translate", on_click=on_translate, key="reader-translate",
                                              tooltip="Translate this chapter")
        self.bottom = ft.Container(
            bgcolor=ft.Colors.with_opacity(0.92, ft.Colors.SURFACE),
            padding=ft.Padding.only(left=8, right=8, top=2, bottom=2),
            animate_opacity=FADE_MS,
            opacity=1.0,
            content=ft.SafeArea(
                content=ft.Column(
                    [
                        ft.Row([self.slider]),
                        ft.Row([self.progress_text, self.page_text], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                        ft.Row([self.prev_button, self.chapters_button, self.aa_button, self.translate_button,
                                self.next_button], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    ],
                    spacing=0,
                    tight=True,
                ),
                avoid_intrusions_top=False,
            ),
            key="reader-bottom",
        )

    # ---- state -----------------------------------------------------------------------------

    def set_titles(self, chapter: str, book: str) -> None:
        self.chapter_title.value = chapter
        self.book_title.value = book
        self._push(self.chapter_title, self.book_title)

    def set_modes(self, available: Mapping[str, bool], selected: str, *, show: bool, bilingual_reason: str = "") -> None:
        self.mode_buttons.visible = bool(show)
        for mode, segment in self.segments.items():
            segment.disabled = not available.get(mode, False)
        self.segments[rm.BILINGUAL].tooltip = bilingual_reason or None
        self.mode_buttons.selected = [selected if available.get(selected) else rm.TRANSLATED]
        self._push(self.mode_buttons)

    def set_progress(self, *, index: int, total: int, display_number: Any, percent: int, page: Optional[int],
                     count: Optional[int], paged: bool, show_percent: bool = True) -> None:
        total = max(1, int(total))
        self.slider.max = max(1, total - 1)
        self.slider.divisions = max(1, total - 1)
        self.slider.value = max(0, min(int(index), total - 1))
        self.slider.disabled = total <= 1
        self.progress_text.value = rm.progress_label(display_number, total, percent) if show_percent \
            else f"Ch {display_number}/{total}"
        self.page_text.value = rm.page_label(page or 0, count or 1) if paged and count else ""
        self.prev_button.disabled = index <= 0 and not (paged and (page or 0) > 0)
        self.next_button.disabled = index >= total - 1
        self._push(self.slider, self.progress_text, self.page_text, self.prev_button, self.next_button)

    def set_translate(self, *, visible: bool, live: bool, reason: str = "") -> None:
        self.translate_button.visible = bool(visible or live)
        self.translate_button.content = "🛰️ Live view" if live else "🌐 Translate"
        self.translate_button.tooltip = reason or ("Show the live translation" if live else "Translate this chapter")
        self._push(self.translate_button)

    # ---- visibility --------------------------------------------------------------------------

    def toggle(self) -> bool:
        self.set_visible(not self.visible)
        return self.visible

    def set_visible(self, visible: bool) -> None:
        self.visible = bool(visible)
        for bar in (self.top, self.bottom):
            if self.visible:
                bar.visible = True
                bar.opacity = 1.0
                bar.ignore_interactions = False
            else:
                bar.opacity = 0.0
                bar.ignore_interactions = True
        self._push(self.top, self.bottom)
        if not self.visible:
            self._schedule_hide()

    def _schedule_hide(self) -> None:
        async def hide_later() -> None:
            await asyncio.sleep(FADE_MS / 1000.0 + 0.02)
            if not self.visible:
                self.top.visible = False
                self.bottom.visible = False
                self._push(self.top, self.bottom)

        try:
            self._fade_task = asyncio.ensure_future(hide_later())
        except RuntimeError:
            self.top.visible = False
            self.bottom.visible = False

    # ---- events -------------------------------------------------------------------------------

    def _on_mode(self, e: Any) -> None:
        selected = next(iter(getattr(e.control, "selected", None) or [rm.TRANSLATED]))
        call_handler(self.on_mode, selected)

    def _on_slider(self, e: Any) -> None:
        try:
            index = int(round(float(e.control.value)))
        except (TypeError, ValueError):
            return
        call_handler(self.on_slider, index)

    @staticmethod
    def _push(*controls: Any) -> None:
        for control in controls:
            try:
                control.update()
            except Exception:
                pass


class SelectionChipRow:
    """The floating chip row for a page selection (shown while a selection exists)."""

    def __init__(
        self,
        *,
        on_copy: Callable[[str], Any],
        on_translate: Callable[[str], Any],
        on_define: Callable[[str], Any],
        on_glossary: Callable[[str], Any],
        on_chat: Callable[[str], Any],
    ) -> None:
        self.text = ""
        self.original_mode = False
        self.handlers = {"copy": on_copy, "translate": on_translate, "define": on_define, "glossary": on_glossary,
                         "chat": on_chat}
        self.copy_chip = ft.Chip(label=ft.Text("Copy"), leading=ft.Icon(ft.Icons.CONTENT_COPY, size=16),
                                 on_click=lambda e: self._fire("copy"), key="sel-copy")
        self.web_chip = ft.Chip(label=ft.Text("Define on web"), leading=ft.Icon(ft.Icons.MENU_BOOK, size=16),
                                on_click=lambda e: self._fire("translate" if self.original_mode else "define"),
                                key="sel-web")
        self.glossary_chip = ft.Chip(label=ft.Text("Add to glossary"),
                                     leading=ft.Icon(ft.Icons.PLAYLIST_ADD, size=16),
                                     on_click=lambda e: self._fire("glossary"), key="sel-glossary")
        self.chat_chip = ft.Chip(label=ft.Text("Ask in chat"), leading=ft.Icon(ft.Icons.CHAT_BUBBLE_OUTLINE, size=16),
                                 on_click=lambda e: self._fire("chat"), key="sel-chat")
        self.row = ft.Row([self.copy_chip, self.web_chip, self.glossary_chip, self.chat_chip], spacing=6,
                          scroll=ft.ScrollMode.AUTO)
        self.container = ft.Container(
            content=self.row,
            left=8,
            right=8,
            top=80,
            visible=False,
            padding=ft.Padding.symmetric(horizontal=8, vertical=6),
            bgcolor=ft.Colors.with_opacity(0.96, ft.Colors.SURFACE_CONTAINER_HIGHEST),
            border_radius=tokens.RADII["card"],
            key="sel-row",
        )

    def show(self, text: str, *, original_mode: bool, target_language: str, rect: Optional[tuple],
             height: float) -> None:
        self.text = text
        self.original_mode = bool(original_mode)
        label = self.web_chip.label
        if isinstance(label, ft.Text):
            label.value = f"Google Translate → {target_language}" if original_mode else "Define on web"
        self.web_chip.leading = ft.Icon(ft.Icons.TRANSLATE if original_mode else ft.Icons.MENU_BOOK, size=16)
        self.container.top = self._top_for(rect, height)
        self.container.visible = True
        self._push()

    @staticmethod
    def _top_for(rect: Optional[tuple], height: float) -> float:
        height = max(200.0, float(height or 800))
        top = 80.0
        if rect is not None:
            y, h = float(rect[1]) * height, float(rect[3]) * height
            top = y - 60 if y > 140 else y + h + 12
        return max(56.0, min(top, height - 140))

    def place(self, rect: Optional[tuple], height: float) -> None:
        """Move the row next to the selection (the page reports the rect separately)."""
        self.container.top = self._top_for(rect, height)
        self._push()

    def hide(self) -> None:
        if not self.container.visible:
            return
        self.text = ""
        self.container.visible = False
        self._push()

    @property
    def visible(self) -> bool:
        return bool(self.container.visible)

    def _fire(self, name: str) -> None:
        text = self.text
        if not text:
            return
        call_handler(self.handlers.get(name), text)

    def _push(self) -> None:
        try:
            self.container.update()
        except Exception:
            pass
