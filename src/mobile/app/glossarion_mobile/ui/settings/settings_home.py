"""Settings home (``/settings``, UI_SPEC §4.15).

Top to bottom: search field ("Search settings") with the filter chips
Modified · Locked · Unavailable on mobile; quick-action chips Save now ·
Backup · Import / Export profiles; notices (job running → "Changes apply to
the next run", API keys that could not be decrypted, config.json problems,
missing schema); then the grouped section list. Groups follow the spec order
(General, Translation, Models & keys, Glossary, QA, Manga, Reader & Library,
Data, About): schema sections (``settings_schema.sections()``, with the
number of settings and how many differ from their defaults) plus the static
Settings pages from the route table (Logs & diagnostics, Env preview, ...;
pages of later milestones carry a ReasonChip). While a search or filter is
active the results replace the list.

It extends ``HubScreen`` (the U1 Settings hub), reusing its child-route list.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.info_sheet import InfoSheet
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.components.section_card import SectionCard
from glossarion_mobile.ui.router import ROUTES_BY_NAME, RouteError, RouteMatch, RouteSpec, build_route
from glossarion_mobile.ui.screens.base import HubScreen
from glossarion_mobile.ui.settings.banners import JobBanner, notice
from glossarion_mobile.ui.settings.model import ordered_groups
from glossarion_mobile.ui.settings.schema_access import SearchHit, SectionInfo
from glossarion_mobile.ui.settings.search import SettingsSearch
from glossarion_mobile.ui.theme import icon_data

__all__ = ["PROFILES_REASON", "ROUTE_GROUPS", "SettingsHome"]

# Static Settings pages (route table) and the home group each belongs to.
ROUTE_GROUPS = {
    "settings.appearance": "General",
    "settings.notifications": "General",
    "settings.profiles": "Translation",
    "settings.prefill": "Translation",
    "settings.models": "Models & keys",
    "settings.keys": "Models & keys",
    "settings.accounts": "Models & keys",
    "settings.endpoints": "Models & keys",
    "settings.storage": "Data",
    "settings.backup": "Data",
    "settings.import": "Data",
    "settings.logs": "Data",
    "settings.env_preview": "Data",
    "settings.updates": "About",
    "settings.about": "About",
    "settings.danger": "About",
}
_ROUTE_ICONS = {
    "settings.appearance": "PALETTE_OUTLINED",
    "settings.notifications": "NOTIFICATIONS_OUTLINED",
    "settings.profiles": "DESCRIPTION_OUTLINED",
    "settings.prefill": "SHORT_TEXT",
    "settings.models": "SMART_TOY_OUTLINED",
    "settings.keys": "KEY",
    "settings.accounts": "ACCOUNT_CIRCLE_OUTLINED",
    "settings.endpoints": "LAN_OUTLINED",
    "settings.storage": "STORAGE",
    "settings.backup": "SETTINGS_BACKUP_RESTORE",
    "settings.import": "DOWNLOAD",
    "settings.logs": "TERMINAL",
    "settings.env_preview": "DATA_OBJECT",
    "settings.updates": "SYSTEM_UPDATE",
    "settings.about": "INFO_OUTLINE",
    "settings.danger": "WARNING_AMBER",
}
PROFILES_REASON = "Unavailable in this session"
# Shown only when Profiles & prompts (AccountsProfilesFeature, U4) did not install; once it does,
# the chip opens Settings › Profiles & prompts instead.
_PROFILES_DETAIL = ("Importing and exporting prompt profiles happens in Settings › Profiles & prompts, which "
                    "could not start in this session. Your profiles in config.json are kept untouched.")


class SettingsHome(HubScreen):
    title = "Settings"

    def __init__(self, match: RouteMatch, ctx: Any, *, implemented_routes: Sequence[str] = ()) -> None:
        super().__init__(match, navigate=lambda name: ctx.go(name))
        self.ctx = ctx
        self.implemented = frozenset(implemented_routes)
        self.section_tiles: dict[str, ft.ListTile] = {}
        self.route_tiles: dict[str, ft.ListTile] = {}
        self.group_titles: list[str] = []
        self.banner = JobBanner(ctx)
        self._unsubs: list[Callable[[], None]] = []
        self.search = SettingsSearch(ctx, on_open=self.open_hit, on_change=self._search_changed)

    # ---- routes --------------------------------------------------------------------------

    def children(self) -> list[RouteSpec]:
        out = list(super().children())
        extra = ROUTES_BY_NAME.get("settings.env_preview")
        if extra is not None and extra not in out:
            out.append(extra)
        return out

    def route_group(self, spec: RouteSpec) -> str:
        return ROUTE_GROUPS.get(spec.name, "Data")

    # ---- body ------------------------------------------------------------------------------

    def build_body(self) -> ft.Control:
        chip = lambda label, icon, handler, key: ft.Chip(  # noqa: E731
            label=ft.Text(label), leading=ft.Icon(icon_data(icon), size=18), on_click=handler, key=key
        )
        self.save_chip = chip("Save now", "SAVE_OUTLINED", self._on_save_now, "settings-quick-save")
        self.backup_chip = chip("Backup", "SETTINGS_BACKUP_RESTORE", self._on_backup, "settings-quick-backup")
        self.profiles_chip = chip("Import / Export profiles", "SWAP_VERT", self._on_profiles, "settings-quick-profiles")
        self.quick_row = ft.Row([self.save_chip, self.backup_chip, self.profiles_chip], spacing=6, scroll=ft.ScrollMode.AUTO)
        self.notices = ft.Column(self._notice_controls(), spacing=6, tight=True)
        self.content = ft.ListView(expand=True, spacing=tokens.SPACING["md"],
                                   padding=ft.Padding.only(left=12, right=12, bottom=16, top=4))
        self.refresh(push=False)
        gutter = ft.Padding.symmetric(horizontal=12)
        return ft.Column(
            [
                ft.Container(padding=ft.Padding.only(left=12, right=12, top=8), content=self.search.field),
                ft.Container(padding=gutter, content=self.search.filter_row),
                ft.Container(padding=gutter, content=self.quick_row),
                self.banner.control,
                self.notices,
                self.content,
            ],
            spacing=tokens.SPACING["sm"],
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,  # full-width search field
        )

    def _notice_controls(self) -> list[ft.Control]:
        store = self.ctx.store
        out: list[ft.Control] = []
        if not self.ctx.schema.available:
            out.append(notice(
                f"The settings schema is not available in this build ({self.ctx.schema.error or 'not importable'}). "
                "config.json is kept untouched.", role="warning", icon="WARNING_AMBER", key="settings-notice-schema"))
        undecryptable = store.undecryptable_keys() if store.loaded else []
        if undecryptable:
            out.append(notice(
                "Some API keys could not be decrypted (they were saved with another device's key): "
                + ", ".join(undecryptable) + ". Re-enter them; the encrypted values are kept until you do.",
                role="warning", icon="KEY_OFF", key="settings-notice-keys"))
        if store.load_error:
            text = store.load_error
            if store.corrupt_backup:
                text += f" (copy saved as {os.path.basename(store.corrupt_backup)})"
            out.append(notice(text, role="error", icon="ERROR_OUTLINE", key="settings-notice-load"))
        if store.save_error:
            out.append(notice(store.save_error, role="error", icon="ERROR_OUTLINE", key="settings-notice-save"))
        return out

    def refresh(self, push: bool = True) -> None:
        if self.search.active:
            self.content.controls = self.search.result_controls()
        else:
            self.content.controls = self.group_cards()
        self.notices.controls = self._notice_controls()
        if push:
            self.ctx.push(self.content, self.notices)

    def _search_changed(self) -> None:
        # SettingsSearch refreshed its own results list; mirror them into the home list.
        if hasattr(self, "content"):
            self.content.controls = self.search.results.controls if self.search.active else self.group_cards()
            self.ctx.push(self.content)

    def _section_tile(self, section: SectionInfo) -> ft.ListTile:
        store = self.ctx.store
        modified = sum(1 for key in section.keys if store.is_modified(self.ctx.schema.path_of(key)))
        count = len(section.keys)
        subtitle = f"{count} setting{'s' if count != 1 else ''}"
        if modified:
            subtitle += f" · {modified} changed"
        try:
            build_route("settings.section", {"section": section.id})
            routable = True
        except RouteError:
            routable = False
        tile = ft.ListTile(
            title=ft.Text(section.title, theme_style=ft.TextThemeStyle.BODY_MEDIUM, color=ft.Colors.ON_SURFACE),
            subtitle=ft.Text(subtitle, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
            trailing=ft.Icon(ft.Icons.CHEVRON_RIGHT) if routable else ReasonChip(reason="Unsupported section id"),
            disabled=not routable,
            min_height=tokens.SIZES["hit_target"],
            dense=True,
            on_click=lambda e, sid=section.id: self.open_section(sid),
            key=f"settings-section-{section.id}",
        )
        self.section_tiles[section.id] = tile
        return tile

    def _route_tile(self, spec: RouteSpec) -> ft.ListTile:
        shipped = spec.name in self.implemented
        tile = ft.ListTile(
            title=ft.Text(spec.title, theme_style=ft.TextThemeStyle.BODY_MEDIUM, color=ft.Colors.ON_SURFACE),
            leading=ft.Icon(icon_data(_ROUTE_ICONS.get(spec.name, "CHEVRON_RIGHT")), size=20),
            trailing=ft.Icon(ft.Icons.CHEVRON_RIGHT) if shipped else ReasonChip(reason=f"Arrives in {spec.milestone}"),
            min_height=tokens.SIZES["hit_target"],
            dense=True,
            on_click=lambda e, name=spec.name: self.navigate(name),
            key=f"hub-{spec.name}",
        )
        self.tiles[spec.name] = tile  # HubScreen API
        self.route_tiles[spec.name] = tile
        return tile

    def group_cards(self) -> list[ft.Control]:
        self.section_tiles = {}
        self.route_tiles = {}
        groups: dict[str, list[ft.Control]] = {}
        names: list[str] = []
        section_ids = set()
        for group, sections in self.ctx.schema.groups():
            for section in sections:
                section_ids.add(section.id)
                groups.setdefault(group, []).append(self._section_tile(section))
                names.append(group)
        for spec in self.children():
            slug = spec.pattern.rsplit("/", 1)[-1]
            if slug in section_ids:  # a schema section already covers this page
                continue
            group = self.route_group(spec)
            groups.setdefault(group, []).append(self._route_tile(spec))
            names.append(group)
        self.group_titles = ordered_groups(names)
        return [
            SectionCard(title=group, children=groups[group], key=f"settings-group-{group}")
            for group in self.group_titles
        ]

    # ---- lifecycle ---------------------------------------------------------------------------

    def did_show(self) -> None:
        if self._unsubs:
            return
        self.banner.attach()
        self._unsubs = [
            self.banner.detach,
            self.ctx.store.observe_all(lambda key, value: self.ctx.on_ui(self._on_config_change)),
            self.ctx.store.observe_saves(lambda ok, error: self.ctx.on_ui(self._on_saved)),
        ]

    def dispose(self) -> None:
        for unsub in self._unsubs:
            unsub()
        self._unsubs = []

    def _on_config_change(self) -> None:
        if not self.search.active:
            self.refresh()

    def refresh_all(self) -> None:
        if hasattr(self, "content"):
            if self.search.active:
                self.search.refresh()
            else:
                self.refresh()

    def _on_saved(self) -> None:
        self.notices.controls = self._notice_controls()
        self.ctx.push(self.notices)

    # ---- navigation / actions ---------------------------------------------------------------

    def open_section(self, section_id: str) -> Optional[str]:
        return self.ctx.go("settings.section", {"section": section_id})

    def open_hit(self, hit: SearchHit) -> Optional[str]:
        return self.ctx.open_setting(hit.section_id, hit.key)

    async def _on_save_now(self, e: Any = None) -> bool:
        wrote = await self.ctx.run_io(self.ctx.store.flush)
        error = self.ctx.store.save_error
        self.ctx.say(error if error else ("Settings saved" if wrote else "No unsaved changes"))
        self._on_saved()
        return bool(wrote)

    async def _on_backup(self, e: Any = None) -> Optional[str]:
        try:
            path = await self.ctx.run_io(self.ctx.store.backup_now)
        except Exception as exc:
            self.ctx.say(f"Backup failed: {exc}")
            return None
        if path:
            self.ctx.say(f"Backup created: {os.path.basename(path)}")
        else:
            self.ctx.say("Nothing to back up yet: config.json has not been written")
        return path

    def _on_profiles(self, e: Any = None) -> Any:
        if "settings.profiles" in self.implemented:
            return self.ctx.go("settings.profiles")
        sheet = InfoSheet(title="Import / Export profiles", body=_PROFILES_DETAIL)
        if self.ctx.page is not None:
            sheet.show(self.ctx.page)
        return sheet
