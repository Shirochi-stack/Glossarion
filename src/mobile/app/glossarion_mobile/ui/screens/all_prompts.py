""""All prompts" index (UI_SPEC §4.14): every prompt setting in the schema, searchable.

The list is built from the settings schema, not by hand: every spec the settings
tiles render as a ``PromptTile`` (``ui.settings.model.tile_kind`` = "prompt") - refine,
Full + raw, translation / title / metadata / chunk / image chunk / GTool / Vision OCR /
memory / glossary / QA AI-Hunter / manga / review / image-edit / parallel-pair wrapper
prompts - grouped by their settings section. Tapping one opens its section at that
tile (``/settings/s/<section>#<key>``), where the U2 PromptTile edits it. Prompts that
are unavailable on mobile stay listed with their reason.

``AllPromptsView`` is a control (the Profiles screen's "All prompts" segment);
``AllPromptsScreen`` wraps it as a screen; ``prompt_index`` is the pure listing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.settings.model import help_line, label_for, sample_value, spec_attr, tile_kind

__all__ = ["AllPromptsScreen", "AllPromptsView", "PromptEntry", "filter_entries", "prompt_index"]


@dataclass(frozen=True)
class PromptEntry:
    key: str
    label: str
    section_id: str
    section_title: str
    group: str
    help: str = ""
    unavailable: Optional[str] = None

    @property
    def breadcrumb(self) -> str:
        return f"{self.group} › {self.section_title}"

    def matches(self, query: str) -> bool:
        if not query:
            return True
        haystack = " ".join((self.key, self.label, self.section_title, self.group, self.help)).casefold()
        return all(part in haystack for part in query.casefold().split())


def prompt_index(schema: Any) -> list:
    """``PromptEntry`` for every prompt spec, in settings-home order (group, section, key order)."""
    out: list = []
    seen: set = set()
    if schema is None or not getattr(schema, "available", False):
        return out
    for group, sections in schema.groups():
        for section in sections:
            for spec in schema.specs_for(section):
                key = str(spec_attr(spec, "key", "") or "")
                if not key or key in seen or tile_kind(spec, sample_value(spec)) != "prompt":
                    continue
                seen.add(key)
                try:
                    available, reason = schema.availability(key)
                except Exception:
                    available, reason = True, None
                out.append(PromptEntry(
                    key=key,
                    label=label_for(spec),
                    section_id=section.id,
                    section_title=section.title,
                    group=group,
                    help=help_line(spec),
                    unavailable=None if available else (reason or "Not available on mobile"),
                ))
    return out


def filter_entries(entries: list, query: str) -> list:
    return [entry for entry in entries if entry.matches(query.strip())]


class AllPromptsView(ft.Column):
    def __init__(self, ctx: Any, *, entries: Optional[list] = None) -> None:
        super().__init__(spacing=tokens.SPACING["sm"], expand=True)
        self.ctx = ctx
        self.entries = list(entries) if entries is not None else prompt_index(getattr(ctx, "schema", None))
        self.query = ""
        self.search = ft.TextField(hint_text="Search prompts", prefix_icon=ft.Icons.SEARCH, dense=True,
                                   on_change=lambda e: self.set_query(e.control.value or ""), key="all-prompts-search")
        self.count_text = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)
        self.list_view = ft.ListView(expand=True, spacing=2, padding=ft.Padding.only(bottom=24))
        self.tiles: dict[str, ft.ListTile] = {}
        self.render(push=False)
        self.controls = [ft.Container(padding=ft.Padding.only(left=12, right=12), content=self.search),
                         ft.Container(padding=ft.Padding.only(left=16), content=self.count_text),
                         self.list_view]

    def set_query(self, query: str) -> None:
        self.query = query
        self.render()

    def visible_entries(self) -> list:
        return filter_entries(self.entries, self.query)

    def render(self, push: bool = True) -> None:
        entries = self.visible_entries()
        self.tiles = {}
        controls: list[ft.Control] = []
        last_group = None
        for entry in entries:
            if entry.group != last_group:
                last_group = entry.group
                controls.append(ft.Container(
                    padding=ft.Padding.only(left=16, top=10, bottom=2),
                    content=ft.Text(entry.group, theme_style=ft.TextThemeStyle.TITLE_SMALL, color=ft.Colors.PRIMARY),
                ))
            tile = ft.ListTile(
                title=ft.Text(entry.label, theme_style=ft.TextThemeStyle.BODY_MEDIUM),
                subtitle=ft.Text(entry.section_title + (f" · {entry.help}" if entry.help else ""), max_lines=2,
                                 overflow=ft.TextOverflow.ELLIPSIS, theme_style=ft.TextThemeStyle.BODY_SMALL,
                                 color=ft.Colors.ON_SURFACE_VARIANT),
                trailing=ReasonChip(reason=entry.unavailable) if entry.unavailable else ft.Icon(ft.Icons.CHEVRON_RIGHT),
                on_click=lambda e, en=entry: self.open(en),
                min_height=tokens.SIZES["hit_target"],
                dense=True,
                key=f"prompt-{entry.key}",
            )
            self.tiles[entry.key] = tile
            controls.append(tile)
        if not entries:
            controls.append(ft.Container(padding=16, content=ft.Text(
                "No prompt settings match." if self.entries else "The settings schema is not available in this build.",
                theme_style=ft.TextThemeStyle.BODY_SMALL)))
        self.list_view.controls = controls
        self.count_text.value = f"{len(entries)} prompt{'s' if len(entries) != 1 else ''}"
        if push:
            for control in (self.list_view, self.count_text):
                try:
                    control.update()
                except Exception:
                    pass

    def open(self, entry: PromptEntry) -> Optional[str]:
        opener = getattr(self.ctx, "open_setting", None)
        if callable(opener):
            return opener(entry.section_id, entry.key)
        return None


class AllPromptsScreen(Screen):
    title = "All prompts"

    def __init__(self, match: Any, ctx: Any) -> None:
        super().__init__(match)
        self.ctx = ctx
        self.view: Optional[AllPromptsView] = None

    def build_body(self) -> ft.Control:
        self.view = AllPromptsView(self.ctx)
        return ft.Container(padding=ft.Padding.only(top=8), content=self.view, expand=True)

