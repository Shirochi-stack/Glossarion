"""ChatDrawer (UI_SPEC §1.3, §5.1): the modal drawer on phones, the sidebar on tablets.

Top to bottom: header row (Halgakos avatar, "Glossarion", New chat, New scratch
chat) · unified search field (results replace the body, grouped Chats · Books ·
Glossaries · Files) · destination chips Library · Jobs (badge) · Glossaries ·
Tools · Pinned (hidden when empty) · Recents (Today / Yesterday / Previous 7
days / <Month YYYY>) · footer: status chip, Settings, API keys, Help. Settings
lives in the footer, not among the destination chips (§1.3 item 3); so does
the Keys button (owner request 17, device report 2026-10-08): it opens the
Multi-Key Manager (``settings.keys``, ``ui/screens/keys.py``) as a shortcut (the
app's ``on_keys``: the page alone on the stack, one Back returns to the chat),
and key settings are not part of Chat settings.

The footer is pinned: ``content`` is a non-scrolling ``Column`` whose only
scrolling child is ``body`` (the chat list / search results, ``expand``), with
the header, search field and destination chips above it and the footer below
it. Only the chat list scrolls; the footer never moves with it.

The same content object is wrapped in ``NavigationDrawer(controls=…)`` on
phones and in a persistent ``Container`` on tablets (AppShell decides; on
phones it sizes the box to the drawer's own list viewport, so that list never
scrolls the whole content, footer included, see ``AppShell._drawer_height``).
Chat row long-press opens an ActionSheet (``Chip`` has no long-press in Flet
1.0.3; ``ListTile.on_long_press`` is used). Rows come from the placeholder
``InMemoryChatIndex`` until ``direct_text_store`` backs it (U3).

U9 Series (optional, §1.3 item 5, §2.15): once the SeriesFeature sets ``series`` (a provider with
``rows()``, ``search(query)``, ``open_series(sid)`` and ``new_chat_in_series(sid)``) and at least
one series exists, a "Series" section sits between Pinned and Recents: one ``ExpansionTile`` per
series (colour dot, name, chat count; children: its chats, "＋ New chat in series", "Series page ›").
A series' chats leave Recents (they are listed under their series; pinned ones stay in Pinned).
The search gains a "Series" group (series-scoped search: a matching series with all its chats, or
the chats of a series that match).

Unified search (UI_SPEC §1.3 item 2, §7.4): Chats are searched in place; Books, Glossaries
and Files come from providers the features register (``register_search``: the Library,
the Glossary Manager, the file tools). A provider may be async (it runs after a 250 ms
debounce, with a thin progress bar under the field); its ``SearchHit`` rows open their
surface and the navigation closes the drawer.

Header and footer rows are 56 dp minimums, not fixed heights: at 200 % text they grow
instead of clipping (UI_SPEC §7.5).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.state.app_state import AppState
from glossarion_mobile.state.chat_index import ChatSummary
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.empty_state import HALGAKOS_AVATAR
from glossarion_mobile.ui.theme import HIT_TARGET, icon_data, semantic

__all__ = ["DESTINATIONS", "ChatDrawer", "KEYS_ROUTE", "SEARCH_DEBOUNCE", "SEARCH_GROUPS", "SEARCH_LIMIT",
           "SearchHit", "drawer_status"]

log = logging.getLogger("glossarion.drawer")

# (id, label, icon, route name)
DESTINATIONS = (
    ("library", "Library", "LOCAL_LIBRARY", "library"),
    ("jobs", "Jobs", "WORK_HISTORY", "jobs"),
    ("glossary", "Glossaries", "SPELLCHECK", "glossary"),
    ("tools", "Tools", "HANDYMAN", "tools"),
)
#: The footer's Keys button: the Multi-Key Manager (API keys of every pool).
KEYS_ROUTE = "settings.keys"
# (id, label, milestone when results appear; None = searchable now). Books, Glossaries and Files
# search through the providers their features register (``ChatDrawer.register_search``).
SEARCH_GROUPS = (
    ("chats", "Chats", None),
    ("books", "Books", None),
    ("glossaries", "Glossaries", None),
    ("files", "Files", None),
)
SEARCH_DEBOUNCE = 0.25
SEARCH_LIMIT = 30


@dataclass(frozen=True)
class SearchHit:
    """One drawer search result from a provider: tapping it calls ``open`` (which navigates)."""

    title: str
    subtitle: str = ""
    icon: str = "DESCRIPTION"
    open: Optional[Callable[[], Any]] = None
    key: str = ""


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
        on_keys: Optional[Callable[..., Any]] = None,  # default: on_navigate(KEYS_ROUTE), like a destination
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
        #: U9 Series provider (SeriesFeature); None = no Series section / search group.
        self.series: Any = None
        self.series_expanded: dict[str, bool] = {}
        self._series_build = 0  # per-build keys: Flet keeps a re-rendered same-key subtree as it was
        # group -> provider(query) -> [SearchHit] (or an awaitable of it); results cached per (group, query)
        self.search_providers: dict[str, Callable[[str], Any]] = {}
        self.search_hits: dict[tuple, list] = {}
        self.search_errors: dict[tuple, str] = {}
        self._search_task: Any = None
        self.search_progress = ft.ProgressBar(visible=False, bar_height=2, key="drawer-search-progress")

        self.header = ft.Container(
            padding=ft.Padding.only(left=16, right=4),
            content=ft.Row(
                [
                    ft.CircleAvatar(foreground_image_src=HALGAKOS_AVATAR, radius=tokens.SIZES["avatar_small"] / 2),
                    ft.Container(  # the title, with the row's 56 dp minimum height (it grows at 200 % text)
                        content=ft.Row([ft.Container(width=0, height=tokens.SIZES["drawer_header"]),
                                        ft.Text("Glossarion", theme_style=ft.TextThemeStyle.TITLE_MEDIUM,
                                                weight=ft.FontWeight.W_600, expand=True)], spacing=0),
                        expand=True,
                    ),
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
            icon=ft.Icons.SETTINGS, tooltip="Settings", on_click=on_settings, size_constraints=HIT_TARGET,
            key="drawer-settings",
        )
        self.keys_button = ft.IconButton(
            icon=ft.Icons.KEY, tooltip="API keys", on_click=on_keys or (lambda e: self._navigate(KEYS_ROUTE)),
            size_constraints=HIT_TARGET, key="drawer-keys",
        )
        self.help_button = ft.IconButton(icon=ft.Icons.HELP_OUTLINE, tooltip="Help", on_click=on_help,
                                         size_constraints=HIT_TARGET, key="drawer-help")
        self.footer = ft.Container(
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            padding=ft.Padding.only(left=8, right=4),
            content=ft.Row(
                [ft.Container(content=self.status_chip, expand=True), self.settings_button, self.keys_button,
                 self.help_button, ft.Container(width=0, height=tokens.SIZES["drawer_footer"])],  # 56 dp minimum
                spacing=0,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            key="drawer-footer",
        )
        # Not scrollable itself: ``body`` (expand) is the one scrolling part, so the footer stays pinned
        # at the bottom and the header, search and chips at the top (owner request 17).
        self.content = ft.Column(
            [
                self.header,
                ft.Container(padding=ft.Padding.symmetric(horizontal=12), content=ft.Column(
                    [self.search_field, self.search_progress], spacing=2, tight=True)),
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
            series_rows = self._series_rows()
            in_series = {c.cid for row in series_rows for c in row.chats}
            if series_rows:
                controls.append(self._section_header("Series"))
                controls.extend(self._series_tile(row) for row in series_rows)
            for label, chats in self.state.chats.recents(self.clock()):
                chats = [c for c in chats if c.cid not in in_series]
                if not chats:
                    continue
                controls.append(self._section_header(label))
                controls.extend(self._chat_row(c) for c in chats)
        self.body.controls = controls
        self._refresh_badge()
        self._refresh_status()

    def search_groups(self) -> tuple:
        """The search filter groups; "Series" joins them once the Series feature is installed (U9)."""
        if self.series is None:
            return SEARCH_GROUPS
        return SEARCH_GROUPS + (("series", "Series", None),)

    def _search_controls(self) -> list[ft.Control]:
        groups = self.search_groups()
        if self.search_group not in {g for g, _l, _m in groups}:
            self.search_group = "chats"
        chips = ft.Row(
            [
                ft.Chip(
                    label=ft.Text(label),
                    selected=group_id == self.search_group,
                    show_checkmark=False,
                    on_select=lambda e, g=group_id: self.set_search_group(g),
                    key=f"search-group-{group_id}",
                )
                for group_id, label, _milestone in groups
            ],
            scroll=ft.ScrollMode.AUTO,
            spacing=6,
        )
        out: list[ft.Control] = [ft.Container(padding=ft.Padding.symmetric(horizontal=12), content=chips)]
        if self.search_group == "series":
            hits = self._series_search(self.query)
            if not hits:
                out.append(self._note("No matches"))
                return out
            out.append(self._section_header("Series"))
            for row in hits:
                out.append(self._series_tile(row, expanded=True))
            return out
        if self.search_group != "chats":
            out.extend(self._provider_controls())
            return out
        matches = self.state.chats.search(self.query)
        if not matches:
            out.append(self._note("No matches"))
            return out
        out.append(self._section_header("Chats"))
        out.extend(self._chat_row(c) for c in matches)
        return out

    def _provider_controls(self) -> list[ft.Control]:
        group = self.search_group
        label = dict((g, text) for g, text, _m in SEARCH_GROUPS).get(group, group.title())
        if group not in self.search_providers:
            return [self._note(f"Searching {label.lower()} is not available in this session.")]
        key = (group, self.query)
        if key in self.search_errors:
            return [self._note(f"Search failed: {self.search_errors[key]}")]
        hits = self.search_hits.get(key)
        if hits is None:
            return [self._note("Searching…")]
        if not hits:
            return [self._note("No matches")]
        rows: list[ft.Control] = [self._section_header(label)]
        for index, hit in enumerate(hits):
            rows.append(ft.ListTile(
                leading=ft.Icon(icon_data(hit.icon), size=20),
                title=ft.Text(hit.title, theme_style=ft.TextThemeStyle.BODY_MEDIUM, max_lines=2,
                              overflow=ft.TextOverflow.ELLIPSIS),
                subtitle=ft.Text(hit.subtitle, theme_style=ft.TextThemeStyle.BODY_SMALL, max_lines=1,
                                 overflow=ft.TextOverflow.ELLIPSIS) if hit.subtitle else None,
                dense=True,
                min_height=tokens.SIZES["hit_target"],
                shape=ft.RoundedRectangleBorder(radius=tokens.RADII["card"]),
                on_click=lambda e, h=hit: self._open_hit(h),
                key=f"search-{group}-{hit.key or index}",
            ))
        return rows

    # ---- search providers ------------------------------------------------------------

    def register_search(self, group: str, provider: Callable[[str], Any]) -> None:
        """``provider(query)`` -> list of ``SearchHit`` (or an awaitable of it) for a search group."""
        self.search_providers[group] = provider
        self.search_hits = {k: v for k, v in self.search_hits.items() if k[0] != group}
        if self.query and self.search_group == group:
            self._schedule_search()

    def _schedule_search(self) -> Any:
        group, query = self.search_group, self.query
        provider = self.search_providers.get(group)
        task = self._search_task
        if task is not None and not task.done():
            task.cancel()
        self._search_task = None
        if group == "chats" or provider is None or not query or (group, query) in self.search_hits:
            self._set_progress(False)
            return None
        try:
            self._search_task = asyncio.ensure_future(self._run_search(group, query, provider))
        except RuntimeError:  # no running loop (host tests): run a synchronous provider at once
            result = provider(query)
            if not asyncio.iscoroutine(result):
                self._store_hits(group, query, result)
            else:
                result.close()
            return None
        self._set_progress(True)
        return self._search_task

    async def _run_search(self, group: str, query: str, provider: Callable[[str], Any]) -> None:
        try:
            await asyncio.sleep(SEARCH_DEBOUNCE)
            result = provider(query)
            if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
                result = await result
            self._store_hits(group, query, result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("drawer search (%s) failed", group)
            self.search_errors[(group, query)] = str(exc) or type(exc).__name__
        finally:
            if (group, query) == (self.search_group, self.query):
                self._set_progress(False)
                self._changed()

    def _store_hits(self, group: str, query: str, result: Any) -> None:
        hits = [hit for hit in list(result or [])[:SEARCH_LIMIT] if isinstance(hit, SearchHit)]
        self.search_hits[(group, query)] = hits
        while len(self.search_hits) > 64:  # a small cache of recent queries
            self.search_hits.pop(next(iter(self.search_hits)))

    def _set_progress(self, visible: bool) -> None:
        if self.search_progress.visible != visible:
            self.search_progress.visible = visible
            try:
                self.search_progress.update()
            except Exception:
                pass

    def _open_hit(self, hit: SearchHit) -> None:
        if hit.open is not None:
            try:
                hit.open()
            except Exception:
                log.exception("opening search result %s failed", hit.title)

    @staticmethod
    def _note(text: str) -> ft.Control:
        return ft.Container(
            padding=ft.Padding.symmetric(horizontal=16, vertical=12),
            content=ft.Text(text, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
        )

    # ---- U9 Series ------------------------------------------------------------------

    def _series_rows(self) -> list:
        provider = self.series
        if provider is None:
            return []
        try:
            return list(provider.rows() or ())
        except Exception:
            return []

    def _series_search(self, query: str) -> list:
        provider = self.series
        if provider is None:
            return []
        try:
            return list(provider.search(query) or ())
        except Exception:
            return []

    def _series_tile(self, row: Any, expanded: Optional[bool] = None) -> ft.ExpansionTile:
        """One series: colour dot, name, chat count; its chats, New chat in series, Series page."""
        self._series_build += 1
        sid = row.sid
        children: list[ft.Control] = [self._chat_row(c) for c in row.chats]
        children.append(ft.ListTile(
            leading=ft.Icon(ft.Icons.ADD, size=18),
            title=ft.Text("New chat in series", theme_style=ft.TextThemeStyle.BODY_MEDIUM),
            dense=True,
            min_height=tokens.SIZES["hit_target"],
            on_click=lambda e, s=sid: self._series_call("new_chat_in_series", s),
            key=f"series-new-{sid}-{self._series_build}",
        ))
        children.append(ft.ListTile(
            leading=ft.Icon(ft.Icons.COLLECTIONS_BOOKMARK_OUTLINED, size=18),
            title=ft.Text("Series page ›", theme_style=ft.TextThemeStyle.BODY_MEDIUM),
            dense=True,
            min_height=tokens.SIZES["hit_target"],
            on_click=lambda e, s=sid: self._series_call("open_series", s),
            key=f"series-page-{sid}-{self._series_build}",
        ))
        is_open = self.series_expanded.get(sid, False) if expanded is None else expanded
        return ft.ExpansionTile(
            title=ft.Text(row.name, theme_style=ft.TextThemeStyle.BODY_MEDIUM, max_lines=1,
                          overflow=ft.TextOverflow.ELLIPSIS),
            leading=ft.Container(width=12, height=12, border_radius=6, bgcolor=row.color),
            trailing=ft.Text(str(row.count), theme_style=ft.TextThemeStyle.LABEL_SMALL,
                             color=ft.Colors.ON_SURFACE_VARIANT),
            controls=children,
            expanded=is_open,
            dense=True,
            min_tile_height=tokens.SIZES["hit_target"],
            controls_padding=ft.Padding.only(left=12),
            shape=ft.RoundedRectangleBorder(radius=tokens.RADII["card"]),
            on_change=lambda e, s=sid: self._series_toggled(s, e),
            key=f"series-{sid}-{self._series_build}",
        )

    def _series_toggled(self, sid: str, e: Any = None) -> None:
        data = getattr(e, "data", None)
        if isinstance(data, bool):
            opened = data
        elif data is not None:
            opened = str(data).lower() == "true"
        else:
            opened = not self.series_expanded.get(sid, False)
        self.series_expanded[sid] = opened

    def _series_call(self, name: str, sid: str) -> None:
        handler = getattr(self.series, name, None) if self.series is not None else None
        if callable(handler):
            handler(sid)

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
        self._schedule_search()
        self._changed()

    def set_search_group(self, group_id: str) -> None:
        self.search_group = group_id
        self._schedule_search()
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
