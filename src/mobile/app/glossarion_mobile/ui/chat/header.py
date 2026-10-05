"""ChatHeader (UI_SPEC §2.1, §5.4).

☰ · chat title over a tappable "Model · Profile · → Target ▾" subtitle ·
scratch toggle (empty chats only) · New chat · ⋯. Each subtitle span opens the
ModelSheet on its tab (Model / Profile / Language); with >= 160% text scale
only the model span and "▾" remain. A "custom" chip shows while per-chat
overrides are active.

Flet 1.0.3 ``AppBar`` has no ``surface_tint``/``scrolled_under_elevation``, so
the scroll tint is manual: ``bgcolor=surface`` at rest, ``surfaceContainer``
once the transcript is scrolled past 4 px (``set_scrolled``).

Phones put the header in ``View.appbar``; tablets render the same controls as
a 56 dp bar at the top of the main area (``build(tablet=True)``), so the
persistent sidebar is not covered by an app bar.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.state.app_state import ChatContext
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["ChatHeader", "MENU_ITEMS", "SCROLL_TINT_PX"]

SCROLL_TINT_PX = 4

# (action id, label) for the ⋯ menu (Move to Series… arrives with Series, U9).
MENU_ITEMS = (
    ("chat_settings", "Chat settings"),
    ("attachments", "Attachments (0)"),
    ("jump_to", "Jump to…"),
    ("search", "Search in chat"),
    ("text_size", "Text size"),
    ("export", "Export chat"),
    ("delete", "Delete chat"),
)


class ChatHeader:
    def __init__(
        self,
        *,
        title: str = "New chat",
        context: Optional[ChatContext] = None,
        on_menu: Optional[Callable[..., Any]] = None,
        on_open_model_sheet: Optional[Callable[[str], Any]] = None,
        on_new_chat: Optional[Callable[..., Any]] = None,
        on_new_scratch: Optional[Callable[..., Any]] = None,
        on_menu_action: Optional[Callable[[str], Any]] = None,
        on_rename: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.context = context or ChatContext()
        self.on_open_model_sheet = on_open_model_sheet
        self.on_menu_action = on_menu_action
        self.scrolled = False
        self.compact_text = False
        self.tablet = False
        self.title_text = ft.Text(
            title,
            theme_style=ft.TextThemeStyle.TITLE_MEDIUM,
            weight=ft.FontWeight.W_600,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        self.title_gesture = ft.GestureDetector(content=self.title_text, on_long_press_start=on_rename)
        muted = ft.Colors.with_opacity(0.6, ft.Colors.ON_SURFACE)
        self.model_span = ft.TextSpan("", on_click=lambda e: self._open("model"))
        self.profile_sep = ft.TextSpan(" · ")
        self.profile_span = ft.TextSpan("", on_click=lambda e: self._open("profile"))
        self.target_sep = ft.TextSpan(" · ")
        self.target_span = ft.TextSpan("", on_click=lambda e: self._open("language"))
        self.caret_span = ft.TextSpan(" ▾", on_click=lambda e: self._open("model"))
        self.subtitle = ft.Text(
            spans=[self.model_span, self.profile_sep, self.profile_span, self.target_sep, self.target_span, self.caret_span],
            theme_style=ft.TextThemeStyle.LABEL_SMALL,
            color=muted,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        self.custom_badge = ft.Container(
            content=ft.Text("custom", theme_style=ft.TextThemeStyle.LABEL_SMALL),
            bgcolor=ft.Colors.SECONDARY_CONTAINER,
            border_radius=tokens.RADII["badge"],
            padding=ft.Padding.symmetric(horizontal=6),
            height=16,
            visible=False,
            on_click=lambda e: self._menu("chat_settings"),
        )
        self.menu_button = ft.IconButton(
            icon=ft.Icons.MENU,
            tooltip="Open navigation",
            on_click=on_menu,
            size_constraints=HIT_TARGET,
        )
        self.scratch_button = ft.IconButton(
            icon=ft.Icons.CHAT_BUBBLE_OUTLINE,
            tooltip="New scratch chat",
            on_click=on_new_scratch,
            size_constraints=HIT_TARGET,
        )
        self.new_chat_button = ft.IconButton(
            icon=ft.Icons.EDIT_SQUARE,
            tooltip="New chat",
            on_click=on_new_chat,
            size_constraints=HIT_TARGET,
        )
        self.overflow = ft.PopupMenuButton(
            icon=ft.Icons.MORE_VERT,
            tooltip="More",
            items=[
                ft.PopupMenuItem(content=label, on_click=lambda e, a=action: self._menu(a), key=f"chat-menu-{action}")
                for action, label in MENU_ITEMS
            ],
            size_constraints=HIT_TARGET,
        )
        self.title_column = ft.Column(
            [self.title_gesture, ft.Row([self.subtitle, self.custom_badge], spacing=6, tight=True)],
            spacing=0,
            tight=True,
        )
        self.wrapper: Any = None
        self.set_context(self.context)

    # ---- building -----------------------------------------------------------------

    @property
    def actions(self) -> list[ft.Control]:
        return [self.scratch_button, self.new_chat_button, self.overflow]

    def build(self, tablet: bool = False) -> Any:
        """``ft.AppBar`` for phones, a 56 dp ``Container`` bar for tablets."""
        self.tablet = tablet
        bgcolor = ft.Colors.SURFACE_CONTAINER if self.scrolled else ft.Colors.SURFACE
        if tablet:
            self.wrapper = ft.Container(
                height=tokens.SIZES["app_bar"],
                bgcolor=bgcolor,
                padding=ft.Padding.only(left=16, right=4),
                content=ft.Row(
                    [ft.Container(content=self.title_column, expand=True), *self.actions],
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=0,
                ),
            )
        else:
            self.wrapper = ft.AppBar(
                leading=self.menu_button,
                title=self.title_column,
                actions=self.actions,
                bgcolor=bgcolor,
                elevation_on_scroll=0,
                toolbar_height=tokens.SIZES["app_bar"],
                center_title=False,
                automatically_imply_leading=False,
            )
        return self.wrapper

    # ---- state ---------------------------------------------------------------------

    def set_context(self, context: ChatContext) -> None:
        self.context = context
        self.model_span.text = context.model
        self.profile_span.text = context.profile
        self.target_span.text = f"→ {context.target_language}"
        compact = self.compact_text
        if compact:
            self.subtitle.spans = [self.model_span, self.caret_span]
        else:
            self.subtitle.spans = [
                self.model_span,
                self.profile_sep,
                self.profile_span,
                self.target_sep,
                self.target_span,
                self.caret_span,
            ]
        self.custom_badge.visible = context.custom and not compact

    @property
    def subtitle_text(self) -> str:
        return "".join(span.text or "" for span in self.subtitle.spans)

    def set_compact_text(self, compact: bool) -> None:
        self.compact_text = compact
        self.set_context(self.context)

    def set_title(self, title: str) -> None:
        self.title_text.value = title

    def set_empty(self, empty: bool) -> None:
        """The scratch toggle is shown only while the chat is empty (§2.1)."""
        self.scratch_button.visible = empty

    def set_scrolled(self, scrolled: bool) -> bool:
        if scrolled == self.scrolled:
            return False
        self.scrolled = scrolled
        if self.wrapper is not None:
            self.wrapper.bgcolor = ft.Colors.SURFACE_CONTAINER if scrolled else ft.Colors.SURFACE
            try:
                self.wrapper.update()
            except Exception:
                pass
        return True

    def on_transcript_scroll(self, e: Any) -> None:
        """``Transcript.on_scroll`` handler: manual AppBar tint."""
        pixels = getattr(e, "pixels", 0) or 0
        self.set_scrolled(pixels > SCROLL_TINT_PX)

    # ---- events --------------------------------------------------------------------

    def _open(self, tab: str) -> None:
        if self.on_open_model_sheet is not None:
            self.on_open_model_sheet(tab)

    def _menu(self, action: str) -> None:
        if self.on_menu_action is not None:
            self.on_menu_action(action)
