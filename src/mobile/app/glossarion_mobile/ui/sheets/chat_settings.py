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

__all__ = [
    "CHAT_SETTING_KEYS",
    "ChatSettingsSheet",
    "DESCRIPTIONS",
    "INTRO",
    "global_updates_for",
]

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
    ) -> None:
        self.cid = str(cid)
        self.config = config
        self.chats = chats
        self.profiles = list(profiles)
        self.languages = list(languages)
        self.on_changed = on_changed
        self.on_choose_model = on_choose_model
        self.scope = scope if scope in ("chat", "global") else "chat"
        self._page: Any = None
        self.body = ft.Column([], tight=True, spacing=4)
        self.scope_button = ft.SegmentedButton(
            segments=[ft.Segment(value="chat", label="This chat"), ft.Segment(value="global", label="All chats")],
            selected=[self.scope],
            on_change=self._on_scope,
        )
        self.dialog = ft.BottomSheet(
            content=ft.Container(
                padding=ft.Padding.only(left=16, right=16, bottom=24),
                content=ft.Column(
                    [
                        ft.Text("Direct Text settings", theme_style=ft.TextThemeStyle.TITLE_LARGE),
                        ft.Text(INTRO, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
                        self.scope_button,
                        self.body,
                    ],
                    tight=True,
                    spacing=10,
                ),
            ),
            show_drag_handle=True,
            scrollable=True,
            draggable=True,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
        )
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
        return self.scope == "chat" and self.overrides().get(field_name) is not None

    def set_value(self, field_name: str, value: Any) -> None:
        if self.scope == "chat":
            if field_name == "skip_plan":
                self.chats.set_meta(self.cid, "skip_plan", bool(value))
            else:
                self.chats.set_override(self.cid, field_name, value)
        else:
            if field_name in ("model", "profile", "target_language"):
                key = {"model": "model", "profile": "active_profile", "target_language": "output_language"}[field_name]
                self.config.set_many({key: value})
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
        try:
            self.dialog.update()
        except Exception:
            pass

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
                ft.IconButton(icon=ft.Icons.RESTART_ALT, icon_size=18, tooltip="Reset to All chats",
                              on_click=lambda e, f=field_name: self.reset(f)),
            ]
        return [ft.Text("Inherited from: All chats", theme_style=ft.TextThemeStyle.LABEL_SMALL,
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
            expand=True,
        )
        role_button = ft.SegmentedButton(
            segments=[ft.Segment(value=value, label=label) for label, value in ATTACHMENT_PROMPT_ROLES],
            selected=[str(self.value_of("attachment_prompt_role"))],
            on_change=lambda e: self.set_value("attachment_prompt_role", (list(e.control.selected) or ["user"])[0]),
        )
        text_scale = float(self.chats.meta(self.cid).get("text_scale") or 1.0)
        sections = [
            ft.ExpansionTile(
                title="Model & prompt",
                expanded=True,
                controls=[
                    model_row,
                    self._dropdown("profile", "Prompt profile", [(p, p) for p in self.profiles]),
                    self._dropdown("target_language", "Target language", [(lang, lang) for lang in self.languages]),
                    self._dropdown("output_mode", "Default output mode", [(m.id, f"{m.emoji} {m.label}") for m in OUTPUT_MODES]),
                ],
            ),
            ft.ExpansionTile(
                title="Glossary",
                expanded=True,
                controls=[ft.Row([ft.Container(expand=True), *self._badge("glossary_override_mode")]), glossary_group],
            ),
            ft.ExpansionTile(
                title="Run behaviour",
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
            ft.ExpansionTile(
                title="Conversation",
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
                    ),
                ],
            ),
        ]
        footer = [ft.TextButton(content="Reset chat overrides", on_click=lambda e: self.reset())] if self.scope == "chat" else []
        self.body.controls = [*sections, *footer]

    def _set_text_scale(self, value: Any) -> None:
        try:
            scale = max(0.85, min(1.5, float(value)))
        except (TypeError, ValueError):
            return
        self.chats.set_meta(self.cid, "text_scale", round(scale, 2))
        self.rebuild()
        self._changed()

    # ---- presentation -----------------------------------------------------------------------

    def show(self, page: Any) -> None:
        self._page = page
        page.show_dialog(self.dialog)

    def close(self) -> None:
        if self._page is not None and getattr(self.dialog, "open", False):
            self._page.pop_dialog()
