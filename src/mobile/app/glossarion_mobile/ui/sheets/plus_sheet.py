"""PlusSheet (UI_SPEC §2.5): what the composer's ＋ opens.

1. Attach tiles (one row of equal outlined tiles): Files (long-press: Pick folder…), Library, Photos,
   Camera. Hotfix 5 (owner: "I only need it show the top part and not take up the entire screen ... So
   it's just attachment, Library, Photos, Camera"): no Clipboard tile (the keyboard pastes), and on the
   phone app the sheet opens on this row alone, like Claude's "Add context" sheet.
2. Output mode: the same OutputModeRow as the composer (shared Signal) and,
   inline below it, the active mode's options (``ModeOptionsContent``, the body of
   the ModeOptionsSheet, U7; the hint line when no factory is given).
3. Tools (also slash commands): Extract glossary · QA scan · Compile EPUB / PDF ·
   Translate headers / metadata · Manga translator · Generate review · Async
   batch (50% off) · Progress manager · Glossary progress · Retranslate chapters.
4. This chat: Glossary policy… · Chat settings….

On the phone app (the extension's ``DraggableSheet``, ``components.drag_sheet``) items 2-4 wait under the
tiles: a swipe up grows the sheet to show them, a swipe down closes it. The composer keeps the output mode
(its chip), the drawer keeps Tools and ⚙ opens Chat settings. Without the extension (desktop ``flet run``)
the sheet scrolls through all four as before.

The callbacks decide what each row does (``ChatView``): Files / Photos pick
through FileBridge and attach (one file per turn), Camera takes a photo with the
phone's camera app (the extension's ``take_photo``) and attaches it, the tools
open their surfaces. Camera is disabled with a ReasonChip where there is no
camera to open (``camera_reason``). Moved here from ``ui/chat/plus_sheet.py`` (U3),
which re-exports it.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.state.store import Signal
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.chat.mode_options_sheet import MODE_HINTS
from glossarion_mobile.ui.chat.output_mode_row import OutputModeRow
from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.components.sheet import sheet_frame
from glossarion_mobile.ui.theme import icon_data

__all__ = ["ATTACH_TILES", "CAMERA_OFF_REASON", "PlusSheet", "THIS_CHAT", "TOOLS", "camera_error_text"]

# (id, label, icon, disabled reason); Camera's reason comes from the caller (``camera_reason``)
ATTACH_TILES = (
    ("files", "Files", "ATTACH_FILE", None),
    # second: one tap away (owner 2026-10-08, device fix #8)
    ("library", "Library", "LOCAL_LIBRARY", None),
    ("photos", "Photos", "PHOTO_LIBRARY", None),
    ("camera", "Camera", "PHOTO_CAMERA", None),
)
#: Camera where there is no camera to open (the desktop client of ``flet run``)
CAMERA_OFF_REASON = "The camera is only available in the phone app"
#: one row of tiles: height, and the row's padding above / below it (dp)
TILE_HEIGHT = 84
TILES_PADDING = (8, 12)

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

_CAMERA_ERRORS = {
    "camera_access_denied": "Camera access is off for Glossarion: turn it on in the phone's Settings › Apps.",
    "no_available_camera": "This device has no camera to take a photo with.",
    "unavailable": CAMERA_OFF_REASON,
}


def camera_error_text(error: Any, message: Any = None) -> str:
    """What the chat says when ＋ › Camera could not take a photo (image_picker's error codes)."""
    known = _CAMERA_ERRORS.get(str(error or ""))
    if known:
        return known
    detail = str(message or error or "").strip()
    return f"The camera could not open: {detail}" if detail else "The camera could not open."


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
        mode_content: Optional[Callable[[str], Any]] = None,  # mode -> ft.Control (the mode's options)
        camera_reason: Optional[str] = CAMERA_OFF_REASON,  # None: the phone app can take a photo
    ) -> None:
        self.on_attach = on_attach
        self.on_attach_long_press = on_attach_long_press
        self.on_tool = on_tool
        self.on_this_chat = on_this_chat
        self._page: Any = None
        self.native = False
        self.tiles = {
            tile_id: self._tile(tile_id, label, icon, camera_reason if tile_id == "camera" else reason)
            for tile_id, label, icon, reason in ATTACH_TILES
        }
        self.tiles_row = ft.Container(
            padding=ft.Padding.only(left=16, right=16, top=TILES_PADDING[0], bottom=TILES_PADDING[1]),
            content=ft.Row(list(self.tiles.values()), spacing=8, vertical_alignment=ft.CrossAxisAlignment.START),
            key="plus-tiles",
        )
        self.output_row = OutputModeRow(
            mode_signal=mode_signal,
            style_name=row_style,
            on_open_options=on_open_mode_options,
            on_mode_changed=self._mode_changed,
        )
        self.mode_hint = ft.Text(
            MODE_HINTS[self.output_row.mode], theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT
        )
        self.mode_content = mode_content
        self.mode_options = ft.AnimatedSwitcher(content=self._options_for(self.output_row.mode), duration=200,
                                                reverse_duration=200, key="plus-mode-options")
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
        #: what waits under the tiles (a swipe up on the phone app)
        self.sections: list = [
            self._header("Output mode"),
            ft.Container(
                padding=ft.Padding.symmetric(horizontal=12),
                content=ft.Column([self.output_row, self.mode_options], spacing=4, tight=True),
            ),
            self._header("Tools"),
            *self.tool_tiles.values(),
            self._header("This chat"),
            *self.chat_tiles.values(),
        ]
        # The body scrolls: with the mode options inline it is taller than a phone, and "This chat ›
        # Chat settings…" is the last row (components.sheet). ``show`` swaps in the draggable sheet on the
        # phone app.
        self.dialog = ft.BottomSheet(
            content=sheet_frame(
                padding=ft.Padding.only(bottom=16),
                content=ft.Column(
                    [self.tiles_row, *self.sections],
                    spacing=4,
                    tight=True,
                    scroll=ft.ScrollMode.AUTO,
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
        """An outlined tile sharing the row equally (Claude's "Add context" tiles): icon over label."""
        disabled = reason is not None
        column: list[ft.Control] = [
            ft.Icon(icon_data(icon), size=26, color=ft.Colors.ON_SURFACE_VARIANT if disabled else None),
            ft.Text(label, theme_style=ft.TextThemeStyle.LABEL_LARGE, text_align=ft.TextAlign.CENTER, max_lines=1,
                    overflow=ft.TextOverflow.ELLIPSIS),
        ]
        if disabled:
            column.append(ReasonChip(reason=reason))
        return ft.Container(
            expand=True,
            height=None if disabled else TILE_HEIGHT,
            border_radius=16,
            border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
            bgcolor=ft.Colors.SURFACE_CONTAINER,
            opacity=0.6 if disabled else 1.0,
            alignment=ft.Alignment.CENTER,
            padding=ft.Padding.symmetric(horizontal=4, vertical=10),
            ink=not disabled,
            on_click=None if disabled else (lambda e, t=tile_id: self._select(self.on_attach, t)),
            on_long_press=None if disabled else (lambda e, t=tile_id: self._select(self.on_attach_long_press, t)),
            content=ft.Column(column, spacing=6, tight=True, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            key=f"attach-{tile_id}",
            tooltip=reason or label,
        )

    def _options_for(self, mode_id: str) -> ft.Control:
        """The active mode's options inline (UI_SPEC §2.5 item 2); the hint line as a fallback."""
        if self.mode_content is not None:
            try:
                control = self.mode_content(mode_id)
            except Exception:
                control = None
            if control is not None:
                return control
        return self.mode_hint

    def _mode_changed(self, mode_id: str) -> None:
        self.mode_hint.value = MODE_HINTS.get(mode_id, "")
        self.mode_options.content = self._options_for(mode_id)
        try:
            self.mode_options.update()
        except Exception:
            pass

    def _select(self, handler: Optional[Callable[[str], Any]], item_id: str) -> None:
        self.close()
        if handler is not None:
            handler(item_id)

    def tiles_height(self) -> float:
        """The tile row's height (dp): what the phone sheet opens on, plus a hint of the row under it."""
        return TILE_HEIGHT + TILES_PADDING[0] + TILES_PADDING[1] + 20

    def _use_draggable(self, page: Any) -> None:
        """On the phone app: open on the tiles alone, the rest under a swipe up (hotfix 5)."""
        from glossarion_mobile.ui.components.drag_sheet import draggable_sheet_class, height_fraction

        sheet_cls = draggable_sheet_class(page)
        self.native = sheet_cls is not None
        if sheet_cls is None:
            return
        initial = height_fraction(page, self.tiles_height(), low=0.12, high=0.5)
        bottom = ft.SafeArea(content=ft.Container(height=16), avoid_intrusions_top=False, key="plus-bottom")
        self.dialog.content = sheet_cls(controls=[*self.sections, bottom], header=self.tiles_row,
                                        initial_size=initial, min_size=round(initial / 2, 3), max_size=1.0,
                                        key="plus-sheet")

    def show(self, page: Any) -> None:
        self._page = page
        self._use_draggable(page)
        page.show_dialog(self.dialog)

    def close(self) -> None:
        close_dialog(self._page, self.dialog)
