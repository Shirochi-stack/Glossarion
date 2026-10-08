"""Series page (``/series/<sid>``, UI_SPEC §2.15; Appendix A ``chat/series_page.py``).

Header with the colour, the cover (a linked book's) and the name, and an Edit button (name, colour,
cover; Delete series). Cards:

* **Defaults:** the series' model / prompt profile / target language / output mode / glossary
  policy and Direct Text overrides (unset = All chats); "Edit defaults" opens the Chat settings
  sheet in series scope. Chats of the series inherit them (Global -> Series -> chat).
* **Glossary:** the series' manual glossary (the defaults' ``manual_glossary_path``) with its term
  count: Open in the Glossary editor · Use book glossary (a linked book's) · Clear.
* **Books:** the linked Library books as rows with their progress (tap: Book page; ⋯: Remove
  from series).
* **Chats:** the series' chats (tap: open) and "＋ New chat in series".

An unknown or deleted series shows "This series is no longer available". The page follows the
series store and the chat list while it is shown.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import flet as ft

from glossarion_mobile.state.series import member_chats
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.chat.direct_text_rules import GLOSSARY_OVERRIDE_LABELS
from glossarion_mobile.ui.chat.series_sheets import color_dot
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.empty_state import EmptyState
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.screens.page_base import section
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["DEFAULT_LABELS", "SeriesScreen", "defaults_summary"]

log = logging.getLogger("glossarion.series")

#: Series default -> label on the Defaults card (unset rows read "All chats").
DEFAULT_LABELS = (
    ("model", "Model"),
    ("profile", "Prompt profile"),
    ("target_language", "Target language"),
    ("output_mode", "Default output mode"),
    ("glossary_override_mode", "Glossary"),
    ("attachment_prompt_role", "Attached-text prompt role"),
    ("force_multipass_off", "Force Multipass off"),
    ("disable_thinking", "Disable all thinking"),
    ("skip_prompt_profile", "Skip prompt profile"),
    ("disable_auto_scroll", "Disable conversation auto-scroll"),
    ("rendered_card_limit", "Rendered conversation cards"),
)


def _value_text(key: str, value: Any) -> str:
    if key == "glossary_override_mode":
        return GLOSSARY_OVERRIDE_LABELS.get(str(value), str(value))
    if key == "output_mode":
        try:
            from glossarion_mobile.ui.chat.output_modes import OUTPUT_MODES

            mode = next((m for m in OUTPUT_MODES if m.id == value), None)
            if mode is not None:
                return f"{mode.emoji} {mode.label}"
        except Exception:
            pass
    if isinstance(value, bool):
        return "On" if value else "Off"
    if key == "attachment_prompt_role":
        return str(value).title()
    return str(value)


def defaults_summary(defaults: dict) -> list:
    """(label, text) rows of the set defaults, in Chat settings order."""
    return [(label, _value_text(key, defaults[key])) for key, label in DEFAULT_LABELS
            if defaults.get(key) is not None]


class SeriesScreen(Screen):
    """``feature`` is the SeriesFeature (store, chats, navigation, Library and glossary lookups)."""

    def __init__(self, match: Any, feature: Any) -> None:
        super().__init__(match)
        self.feature = feature
        self.sid = str(match.params.get("sid") or "") if match is not None else ""
        item = feature.store.get(self.sid)
        self.title = item.name if item is not None else "Series"
        self.holder = ft.Column([], spacing=tokens.SPACING["md"], tight=True)
        self.builds = 0
        self.glossary_count: Optional[int] = None
        self._count_path = ""
        self._unsubs: list = []

    # ---- layout --------------------------------------------------------------------------------

    @property
    def series(self) -> Any:
        return self.feature.store.get(self.sid)

    def actions(self) -> list:
        if self.series is None:
            return []
        return [ft.IconButton(icon=ft.Icons.EDIT_OUTLINED, tooltip="Edit series", on_click=lambda e: self.edit(),
                              size_constraints=HIT_TARGET, key="series-edit")]

    def build_body(self) -> ft.Control:
        self.render()
        return ft.ListView([self.holder], expand=True, padding=12, key="series-page")

    def render(self) -> None:
        self.builds += 1
        item = self.series
        if item is None:
            self.holder.controls = [EmptyState(icon="COLLECTIONS_BOOKMARK", title="This series is no longer available",
                                               body="It was deleted. Its chats are still in the drawer.",
                                               key=f"series-missing-{self.builds}")]
            return
        self.holder.controls = [self._header(item), self._defaults_card(item), self._glossary_card(item),
                                self._books_card(item), self._chats_card(item)]
        path = str(item.defaults.get("manual_glossary_path") or "")
        if path != self._count_path:
            self._count_path = path
            self.glossary_count = None
            if path:
                self.feature.spawn(self._count_terms(path))

    def refresh(self) -> None:
        item = self.series
        if item is not None and item.name != self.title:
            self.title = item.name
            if self.app_bar_title is not None:
                self.app_bar_title.value = item.name
                self.feature.push(self.app_bar_title)
        self.render()
        self.feature.push(self.holder)

    def did_show(self) -> None:
        if not self._unsubs:
            self._unsubs.append(self.feature.store.subscribe(self.refresh))
            chats = self.feature.chats
            if chats is not None and hasattr(chats, "subscribe"):
                self._unsubs.append(chats.subscribe(self.refresh))

    def dispose(self) -> None:
        for unsub in self._unsubs:
            try:
                unsub()
            except Exception:
                pass
        self._unsubs = []

    # ---- cards ---------------------------------------------------------------------------------

    def _header(self, item: Any) -> ft.Control:
        cover = self.feature.cover_src(item.cover_bid) if item.cover_bid else None
        art: ft.Control = (ft.Image(src=cover, width=56, height=80, fit=ft.BoxFit.COVER,
                                    border_radius=tokens.RADII["cover"], key="series-cover")
                           if cover else color_dot(item.color_hex, 40, key="series-dot"))
        count = len(member_chats(self.feature.chats, item.id, self.feature.store))
        books = len(item.book_ids)
        line = f"{count} chat{'s' if count != 1 else ''} · {books} book{'s' if books != 1 else ''}"
        return ft.Container(
            content=ft.Row([
                art,
                ft.Column([ft.Text(item.name, theme_style=ft.TextThemeStyle.TITLE_LARGE, max_lines=2,
                                   overflow=ft.TextOverflow.ELLIPSIS),
                           ft.Text(line, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)],
                          spacing=2, tight=True, expand=True),
                ft.TextButton(content="Edit", icon=ft.Icons.EDIT_OUTLINED, on_click=lambda e: self.edit()),
            ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            padding=tokens.SPACING["card_padding"],
            border_radius=tokens.RADII["card"],
            border=ft.Border.only(left=ft.BorderSide(4, item.color_hex)),
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            key=f"series-header-{self.builds}",
        )

    def _defaults_card(self, item: Any) -> ft.Control:
        rows = defaults_summary(item.defaults)
        lines: list[ft.Control] = [
            ft.Row([ft.Text(label, theme_style=ft.TextThemeStyle.LABEL_MEDIUM, expand=True),
                    ft.Text(text, theme_style=ft.TextThemeStyle.BODY_SMALL, max_lines=2,
                            overflow=ft.TextOverflow.ELLIPSIS, text_align=ft.TextAlign.END, expand=True)],
                   spacing=8)
            for label, text in rows
        ]
        if not lines:
            lines.append(ft.Text("No defaults: this series' chats use All chats.", theme_style=ft.TextThemeStyle.BODY_SMALL,
                                 color=ft.Colors.ON_SURFACE_VARIANT))
        lines.append(ft.Row([ft.FilledTonalButton(content="Edit defaults", icon=ft.Icons.TUNE,
                                                  on_click=lambda e: self.feature.open_defaults_sheet(self.sid),
                                                  key="series-edit-defaults")]))
        return section("Defaults", lines, subtitle="Inherited by every chat in the series (a chat's own setting wins).",
                       key=f"series-defaults-{self.builds}")

    def _glossary_card(self, item: Any) -> ft.Control:
        path = str(item.defaults.get("manual_glossary_path") or "")
        controls: list[ft.Control] = []
        if path:
            count = self.glossary_count
            detail = "counting terms…" if count is None and os.path.isfile(path) else (
                f"{count} term{'s' if count != 1 else ''}" if count is not None else "file not found")
            controls.append(ft.Text(os.path.basename(path), theme_style=ft.TextThemeStyle.BODY_MEDIUM, max_lines=2,
                                    overflow=ft.TextOverflow.ELLIPSIS, key="series-glossary-name"))
            controls.append(ft.Text(detail, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                                    key="series-glossary-count"))
        else:
            controls.append(ft.Text("No series glossary. Chats use their own glossary settings.",
                                    theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT))
        buttons: list[ft.Control] = []
        if path:
            buttons.append(ft.TextButton(content="Open in editor", icon=ft.Icons.EDIT_NOTE,
                                         on_click=lambda e: self.feature.open_glossary(path),
                                         disabled=not self.feature.can_open_glossary(), key="series-glossary-open"))
        buttons.append(ft.TextButton(content="Use book glossary", icon=ft.Icons.MENU_BOOK,
                                     on_click=lambda e: self.feature.spawn(self.feature.use_book_glossary(self.sid)),
                                     disabled=not item.book_ids, key="series-glossary-book",
                                     tooltip=None if item.book_ids else "Link a Library book first"))
        if path:
            buttons.append(ft.TextButton(content="Clear", icon=ft.Icons.CLEAR,
                                         on_click=lambda e: self.feature.clear_glossary(self.sid),
                                         key="series-glossary-clear"))
        controls.append(ft.Row(buttons, wrap=True, spacing=4))
        return section("Glossary", controls, key=f"series-glossary-{self.builds}",
                       subtitle="Force Manual Glossary with this file for the series' chats.")

    def _books_card(self, item: Any) -> ft.Control:
        rows: list[ft.Control] = []
        for bid in item.book_ids:
            row = self.feature.book_row(bid, on_more=lambda b=bid: self._book_actions(b))
            if row is not None:
                rows.append(row)
        if not rows:
            rows.append(ft.Text("No linked books. In the Library, use a book's ⋯ › Add to Series (or select "
                                "several books › More › Add to Series).", theme_style=ft.TextThemeStyle.BODY_SMALL,
                                color=ft.Colors.ON_SURFACE_VARIANT))
        rows.append(ft.Row([ft.TextButton(content="Open Library", icon=ft.Icons.LOCAL_LIBRARY,
                                          on_click=lambda e: self.feature.go("library"), key="series-library")]))
        return section("Books", rows, key=f"series-books-{self.builds}")

    def _book_actions(self, bid: str) -> ActionSheet:
        sheet = ActionSheet(
            [ActionItem("Open book", lambda: self.feature.go("library.book", {"bid": bid}), icon="MENU_BOOK"),
             ActionItem("Remove from series", lambda: self.feature.store.unlink_book(self.sid, bid),
                        icon="REMOVE_CIRCLE_OUTLINE")],
            title=self.feature.book_title(bid),
            tablet=self.feature.tablet,
        )
        if self.feature.page is not None:
            sheet.show(self.feature.page)
        return sheet

    def _chats_card(self, item: Any) -> ft.Control:
        rows: list[ft.Control] = []
        for chat in member_chats(self.feature.chats, item.id, self.feature.store):
            rows.append(ft.ListTile(
                leading=ft.Icon(ft.Icons.CHAT_BUBBLE_OUTLINE, size=20),
                title=ft.Text(chat.title, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
                trailing=ft.ProgressRing(width=14, height=14, stroke_width=2) if chat.running else None,
                dense=True,
                min_height=tokens.SIZES["hit_target"],
                on_click=lambda e, c=chat.cid: self.feature.open_chat(c),
                key=f"series-chat-{chat.cid}-{self.builds}",
            ))
        if not rows:
            rows.append(ft.Text("No chats yet.", theme_style=ft.TextThemeStyle.BODY_SMALL,
                                color=ft.Colors.ON_SURFACE_VARIANT))
        rows.append(ft.Row([ft.FilledTonalButton(content="New chat in series", icon=ft.Icons.ADD,
                                                 on_click=lambda e: self.feature.new_chat_in_series(self.sid),
                                                 key="series-new-chat")]))
        return section("Chats", rows, key=f"series-chats-{self.builds}")

    # ---- actions -------------------------------------------------------------------------------

    def edit(self) -> Any:
        return self.feature.edit_series(self.sid)

    async def _count_terms(self, path: str) -> None:
        count = await self.feature.count_terms(path)
        if path != self._count_path:
            return
        self.glossary_count = count
        self.render()
        self.feature.push(self.holder)
