"""PlusSheet (UI_SPEC §2.5): what the composer's ＋ opens.

1. Attach tiles (72 dp, radius 14, tonal): Files (long-press: Pick folder…),
   Photos, Camera, From Library, Clipboard.
2. Output mode: the same OutputModeRow as the composer (shared Signal) and the
   active mode's options hint.
3. Tools (also slash commands): Extract glossary · QA scan · Compile EPUB / PDF ·
   Translate headers / metadata · Manga translator · Generate review · Async
   batch (50% off) · Progress manager · Glossary progress · Retranslate chapters.
4. This chat: Glossary policy… · Chat settings….

The callbacks decide what each row does (``ChatView``): Files / Photos pick
through FileBridge and attach (one file per turn), Clipboard pastes, the tools
open their surfaces (or say which milestone ships them). Camera stays visible
but disabled with a ReasonChip until flet-camera is verified for both
platforms (Appendix C item 2). Moved here from ``ui/chat/plus_sheet.py`` (U3),
which re-exports it.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.state.store import Signal
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.chat.mode_options_sheet import MODE_HINTS
from glossarion_mobile.ui.chat.output_mode_row import OutputModeRow
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.theme import icon_data

__all__ = ["ATTACH_TILES", "PlusSheet", "THIS_CHAT", "TOOLS"]

# (id, label, icon, disabled reason)
ATTACH_TILES = (
    ("files", "Files", "ATTACH_FILE", None),
    ("photos", "Photos", "PHOTO_LIBRARY", None),
    ("camera", "Camera", "PHOTO_CAMERA", "Needs flet-camera"),
    ("library", "From Library", "LOCAL_LIBRARY", None),
    ("clipboard", "Clipboard", "CONTENT_PASTE", None),
)

# (id, label, icon)
TOOLS = (
    ("extract_glossary", "Extract glossary", "SPELLCHECK"),
    ("qa", "QA scan", "FACT_CHECK"),
    ("compile", "Compile EPUB / PDF", "MENU_BOOK"),
    ("headers", "Translate headers / metadata", "TITLE"),
    ("manga", "Manga translator", "AUTO_STORIES"),
    ("review", "Generate review", "RATE_REVIEW"),
    ("async", "Async batch (50% off)", "SCHEDULE"),
    ("progress", "Progress manager", "CHECKLIST"),
    ("glossary_progress", "Glossary progress", "PLAYLIST_ADD_CHECK"),
    ("retranslate", "Retranslate chapters", "REPLAY"),
)

THIS_CHAT = (
    ("glossary_policy", "Glossary policy…", "RULE"),
    ("chat_settings", "Chat settings…", "TUNE"),
)


class PlusSheet:
    def __init__(
        self,
        *,
        mode_signal: Optional[Signal] = None,
        row_style: str = "label",
        on_attach: Optional[Callable[[str], Any]] = None,
        on_attach_long_press: Optional[Callable[[str], Any]] = None,
        on_tool: Optional[Callable[[str], Any]] = None,
        on_this_chat: Optional[Callable[[str], Any]] = None,
        on_open_mode_options: Optional[Callable[[str], Any]] = None,
        on_dismiss: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.on_attach = on_attach
        self.on_attach_long_press = on_attach_long_press
        self.on_tool = on_tool
        self.on_this_chat = on_this_chat
        self._page: Any = None
        self.tiles = {tile_id: self._tile(tile_id, label, icon, reason) for tile_id, label, icon, reason in ATTACH_TILES}
        self.output_row = OutputModeRow(
            mode_signal=mode_signal,
            style_name=row_style,
            on_open_options=on_open_mode_options,
            on_mode_changed=self._mode_changed,
        )
        self.mode_hint = ft.Text(
            MODE_HINTS[self.output_row.mode], theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT
        )
        self.tool_tiles = {
            tool_id: ft.ListTile(
                leading=ft.Icon(icon_data(icon)),
                title=ft.Text(label),
                on_click=lambda e, t=tool_id: self._select(self.on_tool, t),
                min_height=tokens.SIZES["hit_target"],
                key=f"tool-{tool_id}",
            )
            for tool_id, label, icon in TOOLS
        }
        self.chat_tiles = {
            item_id: ft.ListTile(
                leading=ft.Icon(icon_data(icon)),
                title=ft.Text(label),
                on_click=lambda e, i=item_id: self._select(self.on_this_chat, i),
                min_height=tokens.SIZES["hit_target"],
                key=f"this-chat-{item_id}",
            )
            for item_id, label, icon in THIS_CHAT
        }
        self.dialog = ft.BottomSheet(
            content=ft.Container(
                padding=ft.Padding.only(bottom=16),
                content=ft.Column(
                    [
                        ft.Container(
                            padding=ft.Padding.symmetric(horizontal=16),
                            content=ft.Row(list(self.tiles.values()), scroll=ft.ScrollMode.AUTO, spacing=8),
                        ),
                        self._header("Output mode"),
                        ft.Container(
                            padding=ft.Padding.symmetric(horizontal=12),
                            content=ft.Column([self.output_row, self.mode_hint], spacing=4, tight=True),
                        ),
                        self._header("Tools"),
                        *self.tool_tiles.values(),
                        self._header("This chat"),
                        *self.chat_tiles.values(),
                    ],
                    spacing=4,
                    tight=True,
                ),
            ),
            show_drag_handle=True,
            draggable=True,
            scrollable=True,
            on_dismiss=on_dismiss,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
        )

    @staticmethod
    def _header(text: str) -> ft.Control:
        return ft.Container(
            padding=ft.Padding.only(left=16, top=12, bottom=4),
            content=ft.Text(text, theme_style=ft.TextThemeStyle.TITLE_SMALL, color=ft.Colors.PRIMARY),
        )

    def _tile(self, tile_id: str, label: str, icon: str, reason: Optional[str]) -> ft.Control:
        disabled = reason is not None
        column: list[ft.Control] = [
            ft.Icon(icon_data(icon), color=ft.Colors.ON_SURFACE_VARIANT if disabled else None),
            ft.Text(label, theme_style=ft.TextThemeStyle.LABEL_SMALL, text_align=ft.TextAlign.CENTER, max_lines=2),
        ]
        if disabled:
            column.append(ReasonChip(reason=reason))
        return ft.Container(
            width=72 if not disabled else 108,
            height=72 if not disabled else 104,
            border_radius=14,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            opacity=0.6 if disabled else 1.0,
            alignment=ft.Alignment.CENTER,
            padding=ft.Padding.all(4),
            on_click=None if disabled else (lambda e, t=tile_id: self._select(self.on_attach, t)),
            on_long_press=None if disabled else (lambda e, t=tile_id: self._select(self.on_attach_long_press, t)),
            content=ft.Column(column, spacing=2, tight=True, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            key=f"attach-{tile_id}",
            tooltip=reason or label,
        )

    def _mode_changed(self, mode_id: str) -> None:
        self.mode_hint.value = MODE_HINTS.get(mode_id, "")
        try:
            self.mode_hint.update()
        except Exception:
            pass

    def _select(self, handler: Optional[Callable[[str], Any]], item_id: str) -> None:
        self.close()
        if handler is not None:
            handler(item_id)

    def show(self, page: Any) -> None:
        self._page = page
        page.show_dialog(self.dialog)

    def close(self) -> None:
        if self._page is not None and getattr(self.dialog, "open", False):
            self._page.pop_dialog()
