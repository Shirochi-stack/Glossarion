"""ChatSettingsSheet (UI_SPEC §2.14): desktop Direct Text Settings + per-chat overrides.

Scope ``SegmentedButton``: **This chat** (sidecar overrides; rows show "Inherited
from: All chats" or a "custom" badge with ↺ reset) · **All chats** (the global
``direct_text_*`` config keys the desktop shares; the glossary policy also writes
the two legacy booleans exactly like ``_on_glossary_override_toggled``).

Sections (desktop labels and descriptions): Model & prompt (model / profile / target
language / default output mode overrides) · Glossary (No Override · No Override
(Attachments Only) · Force No Glossary · Force Manual Glossary) · Run behaviour
(Force Multipass off · Disable all thinking · Skip prompt profile · Attached-text
prompt role · Skip plan for attachments) · Conversation (Disable conversation
auto-scroll · Rendered conversation cards 4-200 · Text size). Footer: "Reset chat
overrides". Effective values reach the run as JobSpec overrides; nothing here writes
config.json in This-chat scope.

The sheet body scrolls (``components.sheet``): on a phone the sections are taller than the
screen (owner report: "chat settings doesn't scroll down"). A change rebuilds the rows; each
section keeps the expanded / collapsed state the user left it in.

On tablets the same body opens in the shell's SidePanel next to the chat (UI_SPEC §1.1, §2.14;
``components.surface.present_sheet``); ``close`` closes whichever it is.

U9 Series (§2.14 item 5, §2.15): the chat's series defaults sit between All chats and the chat
(``ChatStoreAdapter.overrides`` layers them; "custom" and ↺ use the chat's ``own_overrides``), so a
row the series sets reads "Inherited from: Series <name>"; the SeriesFeature fills ``SERIES_HOOKS``
(that label and the "Series" section: current series · Move to Series…). ``subject="series"`` is
the same sheet over one series' defaults (``state.series.SeriesDefaultsChats``): "This series" ·
All chats, without the chat-only rows (skip plan, text size).
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui.chat.direct_text_rules import (
    ATTACHMENT_PROMPT_ROLES,
    GLOSSARY_OVERRIDE_LABELS,
    GLOSSARY_OVERRIDE_MODES,
    MAX_RENDERED_CARD_LIMIT,
    MIN_RENDERED_CARD_LIMIT,
    DirectTextSettings,
    glossary_override_updates,
    normalize_rendered_card_limit,
)
from glossarion_mobile.ui.chat.output_modes import OUTPUT_MODES
from glossarion_mobile.ui.components import surface
from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.components.sheet import bottom_sheet, scroll_column, sheet_frame
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = [
    "CHAT_SETTING_KEYS",
    "ChatSettingsSheet",
    "DESCRIPTIONS",
    "INTRO",
    "SERIES_HOOKS",
    "global_updates_for",
]

#: U9 Series, installed by ``ui.chat.series_feature.SeriesFeature`` (unset = no Series UI):
#: ``inherited_label(chats, cid, field)`` -> "Series <name>" when the chat's series sets ``field``;
#: ``section(sheet)`` -> the "Series" ExpansionTile (current series · Move to Series…) or None.
SERIES_HOOKS: dict = {"inherited_label": None, "section": None}

INTRO = "These overrides apply only to Direct Text runs and are saved between sessions."

#: Desktop setting-card descriptions (Direct Text Settings tab).
DESCRIPTIONS = {
    "attachment_prompt_role": (
        "When a file is attached, composer text is added to every related API request using this role. "
        "The attached file remains the translation source."
    ),
    "rendered_card_limit": (
        "Limit the saved input/output cards mounted in the transcript. The rendered window follows your "
        "scroll position automatically; currently streaming request cards remain visible until they finish."
    ),
    "force_multipass_off": "Run a single translation pass even when Multipass is enabled in the main window.",
    "none": "Use the main translator's current glossary mode and selected glossary without a Direct Text override.",
    "attachments_only": (
        "Use the main translator's glossary settings for attachments; use No Glossary for Direct Text "
        "without an attachment."
    ),
    "no_glossary": "Ignore automatic and manually loaded glossaries for these chat translations.",
    "manual": (
        "Force Manual Glossary Only and request a glossary file or pasted glossary before every Direct Text "
        "translation."
    ),
    "disable_thinking": "Remove provider thinking/reasoning parameters for Direct Text requests.",
    "skip_prompt_profile": "Ignore the selected main-window prompt profile in its currently configured role.",
    "disable_auto_scroll": (
        "Keep the conversation viewport fixed while responses stream. Uncheck to follow the newest generated "
        "text automatically."
    ),
    "skip_plan": "Start attachment runs immediately instead of showing a Plan card first (mobile).",
}

#: DirectTextSettings field -> its global config key (glossary policy writes three keys).
CHAT_SETTING_KEYS = {
    "attachment_prompt_role": "direct_text_attachment_prompt_role",
    "glossary_override_mode": "direct_text_glossary_override_mode",
    "force_multipass_off": "direct_text_force_multipass_off",
    "disable_thinking": "direct_text_disable_thinking",
    "skip_prompt_profile": "direct_text_skip_prompt_profile",
    "disable_auto_scroll": "direct_text_disable_auto_scroll",
    "rendered_card_limit": "direct_text_rendered_card_limit",
    "output_mode": "direct_text_output_mode",
}


def global_updates_for(field_name: str, value: Any) -> dict:
    """Sparse config writes for one changed setting (All chats scope)."""
    if field_name == "glossary_override_mode":
        return glossary_override_updates(value)
    if field_name == "rendered_card_limit":
        return {CHAT_SETTING_KEYS[field_name]: normalize_rendered_card_limit(value)}
    if field_name == "skip_plan":
        return {}
    return {CHAT_SETTING_KEYS[field_name]: value}


class ChatSettingsSheet:
    def __init__(
        self,
        *,
        cid: str,
        config: Any,  # MobileConfigStore-like: get(key, default), set_many(dict)
        chats: Any,  # ChatStoreAdapter
        profiles: Sequence[str] = (),
        languages: Sequence[str] = (),
        on_changed: Optional[Callable[[], Any]] = None,
        on_choose_model: Optional[Callable[[str], Any]] = None,  # scope -> opens the model sheet
        scope: str = "chat",
        subject: str = "chat",  # "series": the sheet edits one series' defaults (U9)
        title: Optional[str] = None,
    ) -> None:
        self.cid = str(cid)
        self.config = config
        self.chats = chats
        self.profiles = list(profiles)
        self.languages = list(languages)
        self.on_changed = on_changed
        self.on_choose_model = on_choose_model
        self.scope = scope if scope in ("chat", "global") else "chat"
        self.subject = "series" if subject == "series" else "chat"
        self.title = title or ("Series defaults" if self.subject == "series" else "Direct Text settings")
        self._page: Any = None
        self.body = ft.Column([], tight=True, spacing=4)
        #: section id -> ExpansionTile of the last rebuild; its ``expanded`` follows the user's taps.
        self.sections: dict = {}
        self.expanded = {"model": True, "glossary": True, "run": False, "conversation": False, "series": True}
        self.scope_button = ft.SegmentedButton(
            segments=[ft.Segment(value="chat", label="This series" if self.subject == "series" else "This chat"),
                      ft.Segment(value="global", label="All chats")],
            selected=[self.scope],
            on_change=self._on_scope,
        )
        self.column = scroll_column(
            [
                ft.Text(self.title, theme_style=ft.TextThemeStyle.TITLE_LARGE),
                ft.Text(INTRO, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
                self.scope_button,
                self.body,
            ],
            spacing=10,
        )
        self.dialog = bottom_sheet(sheet_frame(self.column, padding=ft.Padding.only(left=16, right=16, bottom=24)),
                                   draggable=True)
        self.rebuild()

    # ---- values ----------------------------------------------------------------------------

    def global_settings(self) -> DirectTextSettings:
        return DirectTextSettings.from_config(self.config.get)

    def overrides(self) -> dict:
        values = dict(self.chats.overrides(self.cid))
        meta = self.chats.meta(self.cid)
        if meta.get("skip_plan") is not None:
            values["skip_plan"] = bool(meta.get("skip_plan"))
        return values

    def own_overrides(self) -> dict:
        """The chat's own values (no Series layer): what "custom" and ↺ act on."""
        own = getattr(self.chats, "own_overrides", None)
        values = dict(own(self.cid) if callable(own) else self.chats.overrides(self.cid))
        meta = self.chats.meta(self.cid)
        if meta.get("skip_plan") is not None:
            values["skip_plan"] = bool(meta.get("skip_plan"))
        return values

    def effective(self) -> DirectTextSettings:
        return self.global_settings().with_overrides(self.overrides())

    def value_of(self, field_name: str) -> Any:
        if field_name in ("model", "profile", "target_language"):
            override = self.overrides().get(field_name)
            if self.scope == "chat" and override:
                return override
            value = self.config.get({"model": "model", "profile": "active_profile",
                                     "target_language": "output_language"}[field_name], "") or ""
            if not value and field_name == "target_language":
                value = self.config.get("glossary_target_language", "") or ""
            # what the owner would use on a fresh config (desktop fallbacks), shown, never written
            return value or {"model": "authgpt/gpt-6-luna", "profile": (self.profiles or ["Universal"])[0],
                             "target_language": "English"}[field_name]
        settings = self.effective() if self.scope == "chat" else self.global_settings()
        return getattr(settings, field_name)

    def is_overridden(self, field_name: str) -> bool:
        return self.scope == "chat" and self.own_overrides().get(field_name) is not None

    def inherited_from(self, field_name: str) -> str:
        """Where a This-chat row's value comes from: "All chats" or "Series <name>" (U9)."""
        label = SERIES_HOOKS.get("inherited_label")
        if self.subject == "chat" and callable(label):
            try:
                text = label(self.chats, self.cid, field_name)
            except Exception:
                text = None
            if text:
                return str(text)
        return "All chats"

    def set_value(self, field_name: str, value: Any) -> None:
        if self.scope == "chat":
            if field_name == "skip_plan":
                self.chats.set_meta(self.cid, "skip_plan", bool(value))
            else:
                self.chats.set_override(self.cid, field_name, value)
        else:
            if field_name in ("model", "profile", "target_language"):
                key = {"model": "model", "profile": "active_profile", "target_language": "output_language"}[field_name]
                from glossarion_mobile.state.setting_writes import write_setting

                write_setting(self.config, key, value)  # profile extraction method, language fan-out
            else:
                updates = global_updates_for(field_name, value)
                if updates:
                    self.config.set_many(updates)
                elif field_name == "skip_plan":
                    self.chats.set_meta(self.cid, "skip_plan", bool(value))
        self.rebuild()
        self._changed()

    def reset(self, field_name: Optional[str] = None) -> None:
        if field_name is None:
            self.chats.reset_overrides(self.cid)
            self.chats.set_meta(self.cid, "skip_plan", None)
        elif field_name == "skip_plan":
            self.chats.set_meta(self.cid, "skip_plan", None)
        else:
            self.chats.set_override(self.cid, field_name, None)
        self.rebuild()
        self._changed()

    def _changed(self) -> None:
        if self.on_changed is not None:
            self.on_changed()
        # the column is mounted in the sheet or in the tablet SidePanel
        for control in (self.column, self.dialog):
            try:
                control.update()
                return
            except Exception:
                continue

    def _on_scope(self, e: Any = None) -> None:
        selected = list(getattr(e.control, "selected", []) or []) if e is not None else []
        self.scope = selected[0] if selected else "chat"
        self.rebuild()
        self._changed()

    # ---- rows ------------------------------------------------------------------------------

    def _badge(self, field_name: str) -> list:
        if self.scope != "chat":
            return []
        if self.is_overridden(field_name):
            return [
                ft.Container(content=ft.Text("custom", theme_style=ft.TextThemeStyle.LABEL_SMALL),
                             bgcolor=ft.Colors.SECONDARY_CONTAINER, border_radius=6, padding=ft.Padding.symmetric(horizontal=6)),
                ft.IconButton(icon=ft.Icons.RESTART_ALT, icon_size=18,
                              tooltip=f"Reset to {self.inherited_from(field_name)}",
                              on_click=lambda e, f=field_name: self.reset(f), size_constraints=HIT_TARGET),
            ]
        return [ft.Text(f"Inherited from: {self.inherited_from(field_name)}", theme_style=ft.TextThemeStyle.LABEL_SMALL,
                        color=ft.Colors.ON_SURFACE_VARIANT)]

    def _switch(self, field_name: str, label: str) -> ft.Control:
        switch = ft.Switch(label=label, value=bool(self.value_of(field_name)),
                           on_change=lambda e, f=field_name: self.set_value(f, bool(e.control.value)))
        return ft.Column(
            [
                ft.Row([ft.Container(content=switch, expand=True), *self._badge(field_name)], spacing=4),
                ft.Text(DESCRIPTIONS.get(field_name, ""), theme_style=ft.TextThemeStyle.BODY_SMALL,
                        color=ft.Colors.ON_SURFACE_VARIANT),
            ],
            spacing=2,
            tight=True,
            key=f"setting-{field_name}",
        )

    def _dropdown(self, field_name: str, label: str, options: Sequence[tuple], allow_inherit: bool = False) -> ft.Control:
        value = str(self.value_of(field_name) or "")
        items = [ft.DropdownOption(key=key, text=text) for key, text in options]
        if value and value not in {key for key, _text in options}:
            items.insert(0, ft.DropdownOption(key=value, text=value))
        dropdown = ft.Dropdown(
            label=label,
            value=value or None,
            options=items,
            on_select=lambda e, f=field_name: self.set_value(f, e.control.value),
            expand=True,
            dense=True,
        )
        return ft.Row([dropdown, *self._badge(field_name)], spacing=4, key=f"setting-{field_name}")

    def rebuild(self) -> None:
        for section_id, tile in self.sections.items():  # the user's expand / collapse taps
            self.expanded[section_id] = bool(getattr(tile, "expanded", False))
        glossary_mode = str(self.value_of("glossary_override_mode"))
        glossary_group = ft.RadioGroup(
            value=glossary_mode,
            on_change=lambda e: self.set_value("glossary_override_mode", e.control.value),
            content=ft.Column(
                [
                    ft.Column(
                        [
                            ft.Radio(value=mode, label=GLOSSARY_OVERRIDE_LABELS[mode]),
                            ft.Text(DESCRIPTIONS[mode], theme_style=ft.TextThemeStyle.BODY_SMALL,
                                    color=ft.Colors.ON_SURFACE_VARIANT),
                        ],
                        spacing=0,
                        tight=True,
                    )
                    for mode in GLOSSARY_OVERRIDE_MODES
                ],
                spacing=6,
                tight=True,
            ),
        )
        model_value = str(self.value_of("model") or "")
        model_row = ft.Row(
            [
                ft.Column([ft.Text("Model", theme_style=ft.TextThemeStyle.LABEL_MEDIUM),
                           ft.Text(model_value or "—", theme_style=ft.TextThemeStyle.BODY_MEDIUM)],
                          spacing=0, tight=True, expand=True),
                ft.TextButton(content="Choose…",
                              on_click=lambda e: self.on_choose_model(self.scope) if self.on_choose_model else None),
                *self._badge("model"),
            ],
            spacing=4,
            key="setting-model",
        )
        limit = int(self.value_of("rendered_card_limit"))
        limit_label = ft.Text(f"{limit} cards", theme_style=ft.TextThemeStyle.LABEL_MEDIUM)
        limit_slider = ft.Slider(
            min=MIN_RENDERED_CARD_LIMIT,
            max=MAX_RENDERED_CARD_LIMIT,
            divisions=(MAX_RENDERED_CARD_LIMIT - MIN_RENDERED_CARD_LIMIT) // 2,
            value=limit,
            on_change_end=lambda e: self.set_value("rendered_card_limit", int(round(e.control.value or limit))),
        )
        role_button = ft.SegmentedButton(
            segments=[ft.Segment(value=value, label=label) for label, value in ATTACHMENT_PROMPT_ROLES],
            selected=[str(self.value_of("attachment_prompt_role"))],
            on_change=lambda e: self.set_value("attachment_prompt_role", (list(e.control.selected) or ["user"])[0]),
        )
        text_scale = float(self.chats.meta(self.cid).get("text_scale") or 1.0)
        self.sections = {
            "model": ft.ExpansionTile(
                title="Model & prompt",
                expanded=self.expanded["model"],
                controls=[
                    model_row,
                    self._dropdown("profile", "Prompt profile", [(p, p) for p in self.profiles]),
                    self._dropdown("target_language", "Target language", [(lang, lang) for lang in self.languages]),
                    self._dropdown("output_mode", "Default output mode", [(m.id, f"{m.emoji} {m.label}") for m in OUTPUT_MODES]),
                ],
            ),
            "glossary": ft.ExpansionTile(
                title="Glossary",
                expanded=self.expanded["glossary"],
                controls=[ft.Row([ft.Container(expand=True), *self._badge("glossary_override_mode")]), glossary_group],
            ),
            "run": ft.ExpansionTile(
                title="Run behaviour",
                expanded=self.expanded["run"],
                controls=[
                    self._switch("force_multipass_off", "Force Multipass off"),
                    self._switch("disable_thinking", "Disable all thinking"),
                    self._switch("skip_prompt_profile", "Skip prompt profile"),
                    ft.Column(
                        [
                            ft.Row([ft.Text("Attached-text prompt role", theme_style=ft.TextThemeStyle.LABEL_MEDIUM,
                                            expand=True), *self._badge("attachment_prompt_role")]),
                            role_button,
                            ft.Text(DESCRIPTIONS["attachment_prompt_role"], theme_style=ft.TextThemeStyle.BODY_SMALL,
                                    color=ft.Colors.ON_SURFACE_VARIANT),
                        ],
                        spacing=4,
                        tight=True,
                    ),
                    self._switch("skip_plan", "Skip plan for attachments"),
                ],
            ),
            "conversation": ft.ExpansionTile(
                title="Conversation",
                expanded=self.expanded["conversation"],
                controls=[
                    self._switch("disable_auto_scroll", "Disable conversation auto-scroll"),
                    ft.Column(
                        [
                            ft.Row([ft.Text("Rendered conversation cards", theme_style=ft.TextThemeStyle.LABEL_MEDIUM,
                                            expand=True), limit_label, *self._badge("rendered_card_limit")]),
                            limit_slider,
                            ft.Text(DESCRIPTIONS["rendered_card_limit"], theme_style=ft.TextThemeStyle.BODY_SMALL,
                                    color=ft.Colors.ON_SURFACE_VARIANT),
                        ],
                        spacing=2,
                        tight=True,
                    ),
                    ft.Column(
                        [
                            ft.Text(f"Text size {int(round(text_scale * 100))}%", theme_style=ft.TextThemeStyle.LABEL_MEDIUM),
                            ft.Slider(min=0.85, max=1.5, divisions=13, value=text_scale,
                                      on_change_end=lambda e: self._set_text_scale(e.control.value)),
                        ],
                        spacing=2,
                        tight=True,
                        key="setting-text_scale",
                    ),
                ],
            ),
        }
        if self.subject == "series":  # chat-only rows (sidecar meta) are not series defaults
            for section_id in ("run", "conversation"):
                tile = self.sections[section_id]
                tile.controls = [c for c in tile.controls
                                 if getattr(c, "key", None) not in ("setting-skip_plan", "setting-text_scale")]
        else:
            self._add_series_section()
        reset_label = "Reset series defaults" if self.subject == "series" else "Reset chat overrides"
        footer = [ft.TextButton(content=reset_label, on_click=lambda e: self.reset())] if self.scope == "chat" else []
        self.body.controls = [*self.sections.values(), *footer]

    def _add_series_section(self) -> None:
        """Section 5 "Series" (U9): current series and Move to Series… (SeriesFeature hook)."""
        hook = SERIES_HOOKS.get("section")
        if not callable(hook):
            return
        try:
            tile = hook(self)
        except Exception:
            tile = None
        if tile is not None:
            tile.expanded = self.expanded.get("series", True)
            self.sections["series"] = tile

    def _set_text_scale(self, value: Any) -> None:
        try:
            scale = max(0.85, min(1.5, float(value)))
        except (TypeError, ValueError):
            return
        self.chats.set_meta(self.cid, "text_scale", round(scale, 2))
        self.rebuild()
        self._changed()

    # ---- presentation -----------------------------------------------------------------------

    @property
    def in_panel(self) -> bool:
        """True while the tablet SidePanel shows this sheet."""
        return surface.hosts(self._page, self.dialog)

    def show(self, page: Any) -> None:
        """A bottom sheet on phones; the SidePanel on tablets."""
        self._page = page
        panel_title = "Series defaults" if self.subject == "series" else "Chat settings"
        if surface.present_sheet(page, self.dialog, title=panel_title, owner=self.dialog):
            return
        page.show_dialog(self.dialog)

    def close(self) -> None:
        close_dialog(self._page, self.dialog)
