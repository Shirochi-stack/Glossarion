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

U7: the ⋯ menu shows "Attachments (N)"; a scratch chat shows a "Scratch" chip and a Save
button instead of the scratch toggle (UI_SPEC §2.1, §2.16); "Search in chat" turns the title
into the ``ChatSearchBar`` (field · "3/17" · ▲ ▼ · ✕, §2.18).

U9: ⋯ › "Move to Series…" (the optional Series, §2.15) once the SeriesFeature sets its handler
(``set_series_handler``); the "custom" chip also covers series defaults (the chat context reads
the layered overrides).

Narrow phones (``set_width`` < ``NARROW_HEADER_DP``): a scratch chat's actions ("Scratch" chip,
Save, New chat, ⋯; ~217 dp) would leave a title slot narrower than the "custom" chip (~15 dp at
320 dp, a RenderFlex overflow in the Android UI tests), so there Save is an icon button and "New
chat" moves into ⋯; every action stays one or two taps away.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.state.app_state import ChatContext
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["ChatHeader", "ChatSearchBar", "MENU_ITEMS", "NARROW_HEADER_DP", "SCROLL_TINT_PX", "narrow_header"]

SCROLL_TINT_PX = 4

#: Window widths (dp) below which a scratch chat's header actions are compact (see the module docstring).
NARROW_HEADER_DP = 400


def narrow_header(width: Any) -> bool:
    """A window ``width`` (dp) below ``NARROW_HEADER_DP``: a scratch chat's header actions are compact."""
    try:
        return 0 < float(width or 0) < NARROW_HEADER_DP
    except (TypeError, ValueError):
        return False

# (action id, label) for the ⋯ menu. "Move to Series…" (U9, optional Series) shows only once the
# SeriesFeature sets ``on_move_series`` and never in a scratch chat (scratch chats are not saved).
MENU_ITEMS = (
    ("new_chat", "New chat"),  # only while the New chat button is folded in (a scratch chat on a narrow phone)
    ("chat_settings", "Chat settings"),
    ("attachments", "Attachments (0)"),
    ("jump_to", "Jump to…"),
    ("search", "Search in chat"),
    ("text_size", "Text size"),
    ("move_series", "Move to Series…"),
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
        on_save_scratch: Optional[Callable[..., Any]] = None,
        on_search: Optional[Callable[[str], Any]] = None,
        on_search_step: Optional[Callable[[int], Any]] = None,
        on_search_close: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.context = context or ChatContext()
        self.is_scratch = False
        self.empty = True
        self.narrow = False  # a phone narrower than NARROW_HEADER_DP (set_width)
        self.on_new_chat = on_new_chat
        self.on_open_model_sheet = on_open_model_sheet
        self.on_menu_action = on_menu_action
        self.scrolled = False
        self.compact_text = False
        self.text_scale = 1.0  # effective text scale: the bar grows past 56 dp for its two lines
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
        # A loose Flexible in the subtitle row (expand + expand_loose): the row gives a non-flex
        # child unbounded width, so without it the ellipsis never applied and the row overflowed
        # the app bar's title slot on phones narrower than ~470 dp (88 dp wide at 320 dp), hiding
        # the "custom" chip. Loose, so a short subtitle keeps the chip right after it.
        self.subtitle = ft.Text(
            spans=[self.model_span, self.profile_sep, self.profile_span, self.target_sep, self.target_span, self.caret_span],
            theme_style=ft.TextThemeStyle.LABEL_SMALL,
            color=muted,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
            expand=True,
            expand_loose=True,
        )
        self.custom_badge = ft.Container(
            content=ft.Text("custom", theme_style=ft.TextThemeStyle.LABEL_SMALL),
            bgcolor=ft.Colors.SECONDARY_CONTAINER,
            border_radius=tokens.RADII["badge"],
            padding=ft.Padding.symmetric(horizontal=6, vertical=1),  # grows with the text (no fixed height)
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
        self.menu_items = {
            action: ft.PopupMenuItem(content=label, on_click=lambda e, a=action: self._menu(a), key=f"chat-menu-{action}")
            for action, label in MENU_ITEMS
        }
        #: U9 Series: "Move to Series…" handler (SeriesFeature); the item is hidden without one.
        self.on_move_series: Optional[Callable[[], Any]] = None
        self.menu_items["move_series"].visible = False
        self.menu_items["new_chat"].visible = False
        self.overflow = ft.PopupMenuButton(
            icon=ft.Icons.MORE_VERT,
            tooltip="More",
            items=list(self.menu_items.values()),
            size_constraints=HIT_TARGET,
        )
        self.scratch_chip = ft.Container(
            content=ft.Text("Scratch", theme_style=ft.TextThemeStyle.LABEL_SMALL),
            bgcolor=ft.Colors.TERTIARY_CONTAINER,
            border_radius=tokens.RADII["badge"],
            padding=ft.Padding.symmetric(horizontal=8, vertical=2),
            visible=False,
            tooltip="Scratch chat — not saved",
            key="header-scratch-chip",
        )
        self.save_text_button = ft.TextButton(content="Save", visible=False, on_click=on_save_scratch,
                                              key="header-scratch-save")
        self.save_icon_button = ft.IconButton(icon=ft.Icons.SAVE_OUTLINED, tooltip="Save scratch chat", visible=False,
                                              on_click=on_save_scratch, size_constraints=HIT_TARGET,
                                              key="header-scratch-save-icon")
        self.title_column = ft.Column(
            [self.title_gesture, ft.Row([self.subtitle, self.custom_badge], spacing=6, tight=True)],
            spacing=0,
            tight=True,
        )
        self.search_bar = ChatSearchBar(on_query=on_search, on_step=on_search_step, on_close=on_search_close)
        self.title_slot = ft.Container(content=self.title_column, expand=True)
        self.searching = False
        self.wrapper: Any = None
        self.set_context(self.context)

    # ---- building -----------------------------------------------------------------

    @property
    def actions(self) -> list[ft.Control]:
        return [self.scratch_chip, self.save_text_button, self.save_icon_button, self.scratch_button,
                self.new_chat_button, self.overflow]

    @property
    def compact_actions(self) -> bool:
        """A scratch chat on a narrow phone: Save as an icon, New chat in ⋯ (module docstring)."""
        return self.is_scratch and self.narrow and not self.tablet

    @property
    def save_scratch_button(self) -> ft.Control:
        """The scratch chat's Save control in use (the "Save" text button, or its icon on narrow phones)."""
        return self.save_icon_button if self.compact_actions else self.save_text_button

    def build(self, tablet: bool = False) -> Any:
        """``ft.AppBar`` for phones, a 56 dp ``Container`` bar for tablets."""
        self.tablet = tablet
        self._sync_actions()
        bgcolor = ft.Colors.SURFACE_CONTAINER if self.scrolled else ft.Colors.SURFACE
        if tablet:
            self.wrapper = ft.Container(
                height=self.bar_height(),
                bgcolor=bgcolor,
                padding=ft.Padding.only(left=16, right=4),
                content=ft.Row(
                    [self.title_slot, *self.actions],
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=0,
                ),
            )
        else:
            self.wrapper = ft.AppBar(
                leading=self.menu_button,
                title=self.title_slot,
                actions=self.actions,
                bgcolor=bgcolor,
                elevation_on_scroll=0,
                toolbar_height=self.bar_height(),
                center_title=False,
                automatically_imply_leading=False,
            )
        return self.wrapper

    def bar_height(self) -> int:
        """56 dp, or what the title + subtitle lines need at large text (UI_SPEC §7.5: no clipped
        text). Title 22 sp lines + subtitle 14 sp lines, plus 8 dp of breathing room."""
        lines = tokens.TYPE_SCALE["title_medium"].line + tokens.TYPE_SCALE["label_small"].line
        return max(tokens.SIZES["app_bar"], int(round(lines * max(1.0, self.text_scale) + 8)))

    def set_text_scale(self, scale: float) -> None:
        """The effective text scale changed: resize a built bar in place."""
        try:
            self.text_scale = max(0.5, float(scale or 1.0))
        except (TypeError, ValueError):
            self.text_scale = 1.0
        wrapper = self.wrapper
        if wrapper is None:
            return
        height = self.bar_height()
        if isinstance(wrapper, ft.AppBar):
            wrapper.toolbar_height = height
        else:
            wrapper.height = height

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
        """The scratch toggle is shown only while the chat is empty (§2.1), never in a scratch chat."""
        self.empty = empty
        self.scratch_button.visible = empty and not self.is_scratch

    def set_width(self, width: float) -> None:
        """The window width (dp): below ``NARROW_HEADER_DP`` a scratch chat's actions are compact."""
        self.narrow = narrow_header(width)
        self._sync_actions()

    def _sync_actions(self) -> None:
        compact = self.compact_actions
        self.save_text_button.visible = self.is_scratch and not compact
        self.save_icon_button.visible = self.is_scratch and compact
        self.new_chat_button.visible = not compact
        item = self.menu_items.get("new_chat")
        if item is not None:
            item.visible = compact

    def set_scratch(self, scratch: bool) -> None:
        """A scratch chat shows a "Scratch" chip and Save instead of the toggle (§2.1)."""
        self.is_scratch = bool(scratch)
        self.scratch_chip.visible = self.is_scratch
        self._sync_actions()
        self.set_empty(self.empty)
        delete = self.menu_items.get("delete")
        if delete is not None:
            delete.content = "Discard scratch chat" if self.is_scratch else "Delete chat"
        self._sync_series_item()

    def set_series_handler(self, handler: Optional[Callable[[], Any]]) -> None:
        """U9 Series: show "Move to Series…" (``handler`` opens the picker) or hide it (None)."""
        self.on_move_series = handler
        self._sync_series_item()

    def _sync_series_item(self) -> None:
        item = self.menu_items.get("move_series")
        if item is not None:
            item.visible = self.on_move_series is not None and not self.is_scratch

    def set_attachments(self, count: int) -> None:
        item = self.menu_items.get("attachments")
        if item is not None:
            item.content = f"Attachments ({max(0, int(count or 0))})"

    def open_search(self) -> "ChatSearchBar":
        self.searching = True
        self.title_slot.content = self.search_bar
        self.search_bar.reset()
        return self.search_bar

    def close_search(self) -> None:
        self.searching = False
        self.title_slot.content = self.title_column

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
        if action == "move_series" and self.on_move_series is not None:
            self.on_move_series()
            return
        if action == "new_chat" and self.on_new_chat is not None:
            self.on_new_chat(None)  # the New chat button, folded into ⋯
            return
        if self.on_menu_action is not None:
            self.on_menu_action(action)


class ChatSearchBar(ft.Row):
    """Search in chat (UI_SPEC §2.18): field · "3/17" · ▲ ▼ · ✕ (Esc closes)."""

    def __init__(self, *, on_query: Optional[Callable[[str], Any]] = None, on_step: Optional[Callable[[int], Any]] = None,
                 on_close: Optional[Callable[[], Any]] = None) -> None:
        super().__init__(spacing=0, vertical_alignment=ft.CrossAxisAlignment.CENTER, key="chat-search")
        self.on_query = on_query
        self.on_step = on_step
        self.on_close = on_close
        self.field = ft.TextField(hint_text="Search in chat", dense=True, border=ft.NoInputBorder(), expand=True,
                                  autofocus=True, on_submit=lambda e: self._step(1), on_change=self._changed,
                                  key="chat-search-field")
        self.count = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_MEDIUM)
        self.up = ft.IconButton(icon=ft.Icons.KEYBOARD_ARROW_UP, tooltip="Previous match", size_constraints=HIT_TARGET,
                                on_click=lambda e: self._step(-1), key="chat-search-up")
        self.down = ft.IconButton(icon=ft.Icons.KEYBOARD_ARROW_DOWN, tooltip="Next match", size_constraints=HIT_TARGET,
                                  on_click=lambda e: self._step(1), key="chat-search-down")
        self.close_button = ft.IconButton(icon=ft.Icons.CLOSE, tooltip="Close search", size_constraints=HIT_TARGET,
                                          on_click=lambda e: self._close(), key="chat-search-close")
        self.controls = [self.field, self.count, self.up, self.down, self.close_button]

    def reset(self) -> None:
        self.field.value = ""
        self.count.value = ""

    def set_count(self, position: int, total: int) -> None:
        self.count.value = f"{position}/{total}" if total else ("0/0" if (self.field.value or "").strip() else "")
        try:
            self.count.update()
        except Exception:
            pass

    def _changed(self, e: Any = None) -> None:
        if self.on_query is not None:
            self.on_query(self.field.value or "")

    def _step(self, delta: int) -> None:
        if self.on_step is not None:
            self.on_step(delta)

    def _close(self) -> None:
        if self.on_close is not None:
            self.on_close()
