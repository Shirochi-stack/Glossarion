"""ChatDrawer (UI_SPEC §1.3, §5.1): the modal drawer on phones, the sidebar on tablets.

Top to bottom: header row (Halgakos avatar, "Glossarion", New chat, New scratch
chat) · unified search field (results replace the body, grouped Chats · Books ·
Glossaries · Files) · destination chips Library · Jobs (badge) · Glossaries ·
Tools · Pinned (hidden when empty) · Recents (Today / Yesterday / Previous 7
days / <Month YYYY>) · footer: status chip, Settings, Help. Settings lives in
the footer, not among the destination chips (§1.3 item 3).

The same content object is wrapped in ``NavigationDrawer(controls=…)`` on
phones and in a persistent ``Container`` on tablets (AppShell decides).
Chat row long-press opens an ActionSheet (``Chip`` has no long-press in Flet
1.0.3; ``ListTile.on_long_press`` is used). Rows come from the placeholder
``InMemoryChatIndex`` until ``direct_text_store`` backs it (U3).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.state.app_state import AppState
from glossarion_mobile.state.chat_index import ChatSummary
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.empty_state import HALGAKOS_ASSET
from glossarion_mobile.ui.theme import HIT_TARGET, icon_data, semantic

__all__ = ["DESTINATIONS", "ChatDrawer", "SEARCH_GROUPS", "drawer_status"]

# (id, label, icon, route name)
DESTINATIONS = (
    ("library", "Library", "LOCAL_LIBRARY", "library"),
    ("jobs", "Jobs", "WORK_HISTORY", "jobs"),
    ("glossary", "Glossaries", "SPELLCHECK", "glossary"),
    ("tools", "Tools", "HANDYMAN", "tools"),
)
# (id, label, milestone when results appear; None = searchable now)
SEARCH_GROUPS = (
    ("chats", "Chats", None),
    ("books", "Books", "U5"),
    ("glossaries", "Glossaries", "U6"),
    ("files", "Files", "U3"),
)


def drawer_status(state: AppState) -> tuple[str, bool]:
    """(footer chip text, warning?) e.g. "authgpt/gpt-6-luna · Sign in with ChatGPT"."""
    model = state.chat_context.value.model
    if state.backend.value is None:
        return f"{model} · Preparing engine…", False
    if not state.engine_ready:
        return f"{model} · Engine failed to load", True
    if state.needs_chatgpt_sign_in(model):
        return f"{model} · Sign in with ChatGPT", True
    return f"{model} · Ready", False


class ChatDrawer:
    def __init__(
        self,
        *,
        state: AppState,
        on_navigate: Optional[Callable[[str], Any]] = None,  # route name
        on_open_chat: Optional[Callable[[str], Any]] = None,
        on_chat_long_press: Optional[Callable[[ChatSummary], Any]] = None,
        on_new_chat: Optional[Callable[..., Any]] = None,
        on_new_scratch: Optional[Callable[..., Any]] = None,
        on_status: Optional[Callable[..., Any]] = None,
        on_settings: Optional[Callable[..., Any]] = None,
        on_help: Optional[Callable[..., Any]] = None,
        clock: Callable[[], float] = time.time,
        dark: bool = False,
    ) -> None:
        self.state = state
        self.on_navigate = on_navigate
        self.on_open_chat = on_open_chat
        self.on_chat_long_press = on_chat_long_press
        self.clock = clock
        self.dark = dark
        self.query = ""
        self.search_group = "chats"
        self._unsubs: list[Callable[[], None]] = []

        self.header = ft.Container(
            height=tokens.SIZES["drawer_header"],
            padding=ft.Padding.only(left=16, right=4),
            content=ft.Row(
                [
                    ft.CircleAvatar(foreground_image_src=HALGAKOS_ASSET, radius=tokens.SIZES["avatar_small"] / 2),
                    ft.Text("Glossarion", theme_style=ft.TextThemeStyle.TITLE_MEDIUM, weight=ft.FontWeight.W_600, expand=True),
                    ft.IconButton(icon=ft.Icons.EDIT_SQUARE, tooltip="New chat", on_click=on_new_chat, size_constraints=HIT_TARGET),
                    ft.IconButton(
                        icon=ft.Icons.CHAT_BUBBLE_OUTLINE,
                        tooltip="New scratch chat",
                        on_click=on_new_scratch,
                        size_constraints=HIT_TARGET,
                    ),
                ],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )
        self.search_field = ft.TextField(
            hint_text="Search chats, books, glossaries",
            prefix_icon=ft.Icons.SEARCH,
            dense=True,
            filled=True,
            border=ft.NoInputBorder(),
            border_radius=28,
            content_padding=ft.Padding.symmetric(horizontal=12, vertical=10),
            on_change=lambda e: self.set_query(e.control.value or ""),
        )
        self.destination_chips: dict[str, ft.Chip] = {
            dest_id: ft.Chip(
                label=ft.Text(label),
                leading=ft.Icon(icon_data(icon), size=18),
                on_click=lambda e, r=route: self._navigate(r),
                key=f"dest-{dest_id}",
            )
            for dest_id, label, icon, route in DESTINATIONS
        }
        self.body = ft.ListView(expand=True, spacing=0, padding=ft.Padding.symmetric(vertical=4))
        self.status_icon = ft.Icon(ft.Icons.SMART_TOY_OUTLINED, size=16)
        self.status_text = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_MEDIUM, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS)
        self.status_chip = ft.Chip(label=self.status_text, leading=self.status_icon, on_click=on_status, key="drawer-status")
        self.settings_button = ft.IconButton(
            icon=ft.Icons.SETTINGS, tooltip="Settings", on_click=on_settings, size_constraints=HIT_TARGET
        )
        self.help_button = ft.IconButton(icon=ft.Icons.HELP_OUTLINE, tooltip="Help", on_click=on_help, size_constraints=HIT_TARGET)
        self.footer = ft.Container(
            height=tokens.SIZES["drawer_footer"],
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            padding=ft.Padding.only(left=8, right=4),
            content=ft.Row(
                [ft.Container(content=self.status_chip, expand=True), self.settings_button, self.help_button],
                spacing=0,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )
        self.content = ft.Column(
            [
                self.header,
                ft.Container(padding=ft.Padding.symmetric(horizontal=12), content=self.search_field),
                ft.Container(
                    padding=ft.Padding.symmetric(horizontal=12),
                    content=ft.Row(list(self.destination_chips.values()), scroll=ft.ScrollMode.AUTO, spacing=6),
                ),
                self.body,
                self.footer,
            ],
            spacing=tokens.SPACING["sm"],
            expand=True,
        )
        self.section_titles: list[str] = []
        self.chat_rows: dict[str, ft.ListTile] = {}
        self.refresh()

    # ---- subscriptions -----------------------------------------------------------

    def attach(self) -> None:
        """Follow the app state (call once the drawer is part of the page)."""
        if self._unsubs:
            return
        state = self.state
        self._unsubs = [
            state.chats.subscribe(self._changed),
            state.current_chat.subscribe(lambda _v: self._changed()),
            state.chat_context.subscribe(lambda _v: self._changed()),
            state.signed_in.subscribe(lambda _v: self._changed()),
            state.backend.subscribe(lambda _v: self._changed()),
            state.jobs_badge.subscribe(lambda _v: self._changed()),
        ]

    def detach(self) -> None:
        for unsub in self._unsubs:
            unsub()
        self._unsubs = []

    def _changed(self) -> None:
        self.refresh()
        try:
            self.content.update()
        except Exception:
            pass

    # ---- rendering --------------------------------------------------------------------

    def _section_header(self, text: str) -> ft.Control:
        self.section_titles.append(text)
        return ft.Container(
            padding=ft.Padding.only(left=16, top=12, bottom=4),
            content=ft.Text(text, theme_style=ft.TextThemeStyle.LABEL_SMALL, color=ft.Colors.PRIMARY),
            key=f"section-{text}",
        )

    def _chat_row(self, chat: ChatSummary) -> ft.ListTile:
        trailing: list[ft.Control] = []
        if chat.running:
            trailing.append(ft.ProgressRing(width=14, height=14, stroke_width=2))
        if chat.attachments:
            trailing.append(ft.Text(f"📎 {chat.attachments}", theme_style=ft.TextThemeStyle.LABEL_SMALL))
        if chat.pinned:
            trailing.append(ft.Icon(ft.Icons.PUSH_PIN, size=16, color=ft.Colors.ON_SURFACE_VARIANT))
        row = ft.ListTile(
            title=ft.Text(chat.title, theme_style=ft.TextThemeStyle.BODY_MEDIUM, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
            trailing=ft.Row(trailing, spacing=6, tight=True) if trailing else None,
            selected=chat.cid == self.state.current_chat.value,
            selected_tile_color=ft.Colors.SECONDARY_CONTAINER,
            dense=True,
            min_height=tokens.SIZES["hit_target"],
            shape=ft.RoundedRectangleBorder(radius=tokens.RADII["card"]),
            on_click=lambda e, c=chat.cid: self._open_chat(c),
            on_long_press=lambda e, c=chat: self._long_press(c),
            key=f"chat-{chat.cid}",
        )
        self.chat_rows[chat.cid] = row
        return row

    def refresh(self) -> None:
        self.section_titles = []
        self.chat_rows = {}
        controls: list[ft.Control] = []
        if self.query:
            controls.extend(self._search_controls())
        else:
            pinned = self.state.chats.pinned()
            if pinned:
                controls.append(self._section_header("Pinned"))
                controls.extend(self._chat_row(c) for c in pinned)
            for label, chats in self.state.chats.recents(self.clock()):
                controls.append(self._section_header(label))
                controls.extend(self._chat_row(c) for c in chats)
        self.body.controls = controls
        self._refresh_badge()
        self._refresh_status()

    def _search_controls(self) -> list[ft.Control]:
        chips = ft.Row(
            [
                ft.Chip(
                    label=ft.Text(label),
                    selected=group_id == self.search_group,
                    show_checkmark=False,
                    on_select=lambda e, g=group_id: self.set_search_group(g),
                    key=f"search-group-{group_id}",
                )
                for group_id, label, _milestone in SEARCH_GROUPS
            ],
            scroll=ft.ScrollMode.AUTO,
            spacing=6,
        )
        out: list[ft.Control] = [ft.Container(padding=ft.Padding.symmetric(horizontal=12), content=chips)]
        milestone = dict((g, m) for g, _l, m in SEARCH_GROUPS).get(self.search_group)
        if milestone is not None:
            out.append(self._note(f"Searching {self.search_group} arrives in {milestone}."))
            return out
        matches = self.state.chats.search(self.query)
        if not matches:
            out.append(self._note("No matches"))
            return out
        out.append(self._section_header("Chats"))
        out.extend(self._chat_row(c) for c in matches)
        return out

    @staticmethod
    def _note(text: str) -> ft.Control:
        return ft.Container(
            padding=ft.Padding.symmetric(horizontal=16, vertical=12),
            content=ft.Text(text, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
        )

    def _refresh_badge(self) -> None:
        badge = self.state.jobs_badge.value
        chip = self.destination_chips["jobs"]
        chip.badge = ft.Badge(label=str(badge.count)) if badge.count else None

    def _refresh_status(self) -> None:
        text, warning = drawer_status(self.state)
        self.status_text.value = text
        color = semantic("warning", self.dark) if warning else None
        self.status_text.color = color
        self.status_icon.color = color
        self.status_icon.icon = ft.Icons.LOGIN if warning else ft.Icons.SMART_TOY_OUTLINED

    # ---- interaction ---------------------------------------------------------------

    def set_query(self, query: str) -> None:
        self.query = query.strip()
        self._changed()

    def set_search_group(self, group_id: str) -> None:
        self.search_group = group_id
        self._changed()

    def _navigate(self, route_name: str) -> None:
        if self.on_navigate is not None:
            self.on_navigate(route_name)

    def _open_chat(self, cid: str) -> None:
        if self.on_open_chat is not None:
            self.on_open_chat(cid)

    def _long_press(self, chat: ChatSummary) -> None:
        if self.on_chat_long_press is not None:
            self.on_chat_long_press(chat)
