"""ChatSettingsSheet (UI_SPEC §2.14): desktop Direct Text Settings + per-chat overrides.

Scope ``SegmentedButton``: **This chat** (sidecar overrides; rows show "Inherited
from: All chats" or a "custom" badge with ↺ reset) · **All chats** (the global
``direct_text_*`` config keys the desktop shares; the glossary policy also writes
the two legacy booleans exactly like ``_on_glossary_override_toggled``).

Sections (desktop labels and descriptions): Model & prompt (model / profile / target
language / default output mode overrides) · Glossary (No Override · No Override
(Attachments Only) · Force No Glossary · Force Manual Glossary; mobile: "Always accept
generated glossaries", All chats in Prefs ``chat_auto_accept_glossary``, This chat in the
chat's sidecar meta, never config.json) · Run behaviour
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

Prompts (device fixes, owner #15/#16): the "Prompt profile" picker lists every profile of the Settings ›
Profiles & prompts listing (the shared ``ProfileService``; all desktop built-ins on a fresh config),
translation profiles first and the task-specific built-ins under "Specialised". Under it the profile's
prompt card (``profiles.profile_prompt_card``): **Edit prompt** (``PromptEditorSheet``; it edits the
shared profile, the desktop model), **New profile…** (a copy of the current profile; This chat / This
series get it as their override, All chats makes it the active profile) and **Manage…** (Settings ›
Profiles & prompts: rename, delete / reset, import / export). Edits from This chat never switch the
global active profile (``keep_active``). A chat whose profile no longer exists shows "<name> (missing)".
No API key settings live here (owner #17): keys have their own Keys button next to Settings.

U9 Series (§2.14 item 5, §2.15): the chat's series defaults sit between All chats and the chat
(``ChatStoreAdapter.overrides`` layers them; "custom" and ↺ use the chat's ``own_overrides``), so a
row the series sets reads "Inherited from: Series <name>"; the SeriesFeature fills ``SERIES_HOOKS``
(that label and the "Series" section: current series · Move to Series…). ``subject="series"`` is
the same sheet over one series' defaults (``state.series.SeriesDefaultsChats``): "This series" ·
All chats, without the chat-only rows (skip plan, text size).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui.chat.direct_text_rules import (
    ATTACHMENT_PROMPT_ROLES,
    AUTO_ACCEPT_GLOSSARY_PREF,
    GLOSSARY_OVERRIDE_LABELS,
    GLOSSARY_OVERRIDE_MODES,
    MAX_RENDERED_CARD_LIMIT,
    MIN_RENDERED_CARD_LIMIT,
    SIDECAR_META_FIELDS,
    DirectTextSettings,
    glossary_override_updates,
    merge_meta_overrides,
    normalize_rendered_card_limit,
)
from glossarion_mobile.state.languages import remember_language, target_languages
from glossarion_mobile.ui.chat.output_modes import OUTPUT_MODES
from glossarion_mobile.ui.components import surface
from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.components.sheet import bottom_sheet, scroll_column, sheet_frame
from glossarion_mobile.ui.screens.profiles import (
    SPECIALISED_GROUP,
    ProfileService,
    ask_profile_name,
    chat_profile_order,
    grouped_profile_names,
    profile_prompt_card,
)
from glossarion_mobile.ui.theme import HIT_TARGET, mono_family

__all__ = [
    "CHAT_SETTING_KEYS",
    "ChatSettingsSheet",
    "DESCRIPTIONS",
    "GROUP_OPTION_PREFIX",
    "INTRO",
    "SERIES_HOOKS",
    "global_updates_for",
    "inherited_value",
]

log = logging.getLogger("glossarion.chat")

#: Key prefix of the profile picker's group heading ("Specialised"): a disabled option, never a value.
GROUP_OPTION_PREFIX = "__group__:"

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
    "auto_accept_glossary": (
        "Skip the Edit / Yes / No card; translation starts with the generated glossary. Desktop always asks."
    ),
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
    if field_name in SIDECAR_META_FIELDS:  # mobile-only: no config.json key (UI_SPEC Appendix B)
        return {}
    return {CHAT_SETTING_KEYS[field_name]: value}


def inherited_value(get: Callable[[str, Any], Any], field_name: str, listing: Any = None,
                    profiles: Sequence[str] = ()) -> str:
    """What a chat without its own ``model`` / ``profile`` / ``target_language`` runs: the All-chats value
    (``get`` is the config store's ``get(key, default)``), else what the owner would use on a fresh config
    (the desktop fallbacks; shown, never written). The profile is the shared listing's active one (the
    desktop start-up rule) when the stored name is missing or unset."""
    value = get({"model": "model", "profile": "active_profile", "target_language": "output_language"}[field_name],
                "") or ""
    if not value and field_name == "target_language":
        value = get("glossary_target_language", "") or ""
    if field_name == "profile" and listing is not None and listing.active and value not in listing.texts:
        value = listing.active  # what the desktop start-up runs for a missing / unset one
    return str(value or {"model": "authgpt/gpt-6-luna", "profile": (list(profiles) or ["Universal"])[0],
                         "target_language": "English"}[field_name])


#: vertical gap between the rows of a chat settings section (U11 item 5)
ROW_GAP = 12
ROW_GAP_DATA = "chat-settings-row-gap"
#: Target language option that opens the "add a language" field (U11 item 9)
ADD_LANGUAGE_OPTION = "__add_language__"


def _spaced(controls: list) -> list:
    """``controls`` with a ``ROW_GAP`` spacer between neighbours (ExpansionTile has no ``spacing``); the
    first and last controls stay first and last."""
    out: list = []
    for index, control in enumerate(controls):
        if index:
            out.append(ft.Container(height=ROW_GAP, data=ROW_GAP_DATA))
        out.append(control)
    return out


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
        prefs: Any = None,  # Prefs-like (get / set): the mobile-only All-chats values (AUTO_ACCEPT_GLOSSARY_PREF)
        # Prompts (owner #15): the shared profile operations (default: a ProfileService over ``config`` when it
        # is the settings store), the SettingsContext the prompt editor opens with (default: the app's
        # settings context) and Manage… (default: close + Settings › Profiles & prompts)
        profile_service: Any = None,
        ctx: Any = None,
        on_manage_profiles: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.cid = str(cid)
        self.config = config
        self.chats = chats
        self.prefs = prefs
        self.ctx = ctx
        self.on_manage_profiles = on_manage_profiles
        if profile_service is None and callable(getattr(config, "snapshot", None)):
            profile_service = ProfileService(config)
        self.profile_service = profile_service
        self.listing: Any = None  # profiles.ProfileList of the shared listing (None without a service)
        self._given_profiles = list(profiles)
        self.profiles = list(profiles)
        self.prompt_editor: Any = None
        self.new_profile_dialog: Any = None
        self.refresh_profiles()
        self.languages = list(languages)
        self.on_changed = on_changed
        self.on_choose_model = on_choose_model
        self.scope = scope if scope in ("chat", "global") else "chat"
        self.subject = "series" if subject == "series" else "chat"
        self.title = title or ("Series defaults" if self.subject == "series" else "Direct Text settings")
        self._page: Any = None
        self.body = ft.Column([], tight=True, spacing=8)
        #: section id -> ExpansionTile of the last rebuild; its ``expanded`` follows the user's taps.
        self.sections: dict = {}
        self.adding_language = False  # the "Add a language" field is open (U11 item 9)
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
        prefs_get = getattr(self.prefs, "get", None) if self.prefs is not None else None
        return DirectTextSettings.from_config(self.config.get, prefs_get=prefs_get)

    @property
    def shows_auto_accept(self) -> bool:
        """The "Always accept generated glossaries" row: chats only, and only with the Prefs store its
        All-chats value lives in (the Series defaults sheet is built without it)."""
        return self.subject == "chat" and self.prefs is not None

    def overrides(self) -> dict:
        return merge_meta_overrides(self.chats.overrides(self.cid), self.chats.meta(self.cid))

    def own_overrides(self) -> dict:
        """The chat's own values (no Series layer): what "custom" and ↺ act on."""
        own = getattr(self.chats, "own_overrides", None)
        return merge_meta_overrides(own(self.cid) if callable(own) else self.chats.overrides(self.cid),
                                    self.chats.meta(self.cid))

    def effective(self) -> DirectTextSettings:
        return self.global_settings().with_overrides(self.overrides())

    def value_of(self, field_name: str) -> Any:
        if field_name in ("model", "profile", "target_language"):
            override = self.overrides().get(field_name)
            if self.scope == "chat" and override:
                return override
            return inherited_value(self.config.get, field_name, self.listing, self.profiles)
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
            if field_name in SIDECAR_META_FIELDS:  # mobile-only chat settings live in the sidecar meta
                self.chats.set_meta(self.cid, field_name, bool(value))
            else:
                self.chats.set_override(self.cid, field_name, value)
        else:
            if field_name in ("model", "profile", "target_language"):
                key = {"model": "model", "profile": "active_profile", "target_language": "output_language"}[field_name]
                from glossarion_mobile.state.setting_writes import write_setting

                write_setting(self.config, key, value)  # profile extraction method, language fan-out
            elif field_name == "auto_accept_glossary":
                if self.prefs is not None:  # Prefs (mobile_state.json), never config.json
                    self.prefs.set(AUTO_ACCEPT_GLOSSARY_PREF, bool(value))
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
            for name in SIDECAR_META_FIELDS:
                self.chats.set_meta(self.cid, name, None)
        elif field_name in SIDECAR_META_FIELDS:
            self.chats.set_meta(self.cid, field_name, None)
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

    def _dropdown(self, field_name: str, label: str, options: Sequence[tuple], allow_inherit: bool = False,
                  missing: str = "{value}") -> ft.Control:
        """``options``: ``(key, text)``; a key starting with ``GROUP_OPTION_PREFIX`` is a disabled heading.
        A current value that is not an option is listed first (``missing`` formats its text)."""
        value = str(self.value_of(field_name) or "")
        items = []
        for key, text in options:
            if str(key).startswith(GROUP_OPTION_PREFIX):
                items.append(ft.DropdownOption(
                    key=key, text=text, disabled=True,
                    content=ft.Text(text, theme_style=ft.TextThemeStyle.LABEL_MEDIUM, color=ft.Colors.PRIMARY)))
            else:
                items.append(ft.DropdownOption(key=key, text=text))
        if value and value not in {key for key, _text in options}:
            items.insert(0, ft.DropdownOption(key=value, text=missing.format(value=value)))
        dropdown = ft.Dropdown(
            label=label,
            value=value or None,
            options=items,
            on_select=lambda e, f=field_name: self._on_select(f, e.control.value),
            expand=True,
            dense=True,
        )
        return ft.Row([dropdown, *self._badge(field_name)], spacing=4, key=f"setting-{field_name}")

    def _on_select(self, field_name: str, value: Any) -> None:
        if field_name == "target_language" and value == ADD_LANGUAGE_OPTION:
            self.adding_language = True
            self.rebuild()
            self._changed()
            return
        if str(value or "").startswith(GROUP_OPTION_PREFIX):  # a group heading: put the shown value back
            self.rebuild()
            self._changed()
            return
        self.set_value(field_name, value)

    # ---- target language (U11 item 9: any language a model can write, not only the built-in list) ----

    def _language_row(self) -> ft.Control:
        options = [(lang, lang) for lang in target_languages(self.prefs, self.languages)]
        options.append((ADD_LANGUAGE_OPTION, "＋ Add a language…"))
        row = self._dropdown("target_language", "Target language", options)
        if not self.adding_language:
            return row
        field = ft.TextField(label="Language name", hint_text="e.g. Tagalog, Esperanto, Old Norse", dense=True,
                             autofocus=True, expand=True, key="setting-add-language-field",
                             on_submit=lambda e: self._add_language(e.control.value))
        return ft.Column([row, ft.Row([field, ft.TextButton(content="Add", key="setting-add-language",
                                                           on_click=lambda e: self._add_language(field.value)),
                                       ft.TextButton(content="Cancel", on_click=lambda e: self._add_language(None))],
                                      spacing=4)], spacing=6, tight=True, key="setting-target_language-box")

    def _add_language(self, name: Any) -> None:
        """Keep the typed name in the user's language list (Prefs) and make it the target language."""
        self.adding_language = False
        name = " ".join(str(name or "").split())
        if name:
            remember_language(self.prefs, name, self.languages)
            self.set_value("target_language", name)
        self.rebuild()
        self._changed()

    # ---- prompt profiles (owner #15/#16) -----------------------------------------------------------

    def refresh_profiles(self) -> None:
        """Re-read the shared profile listing (at open and after a profile operation, not on every row
        change): the picker's names in the chat order, the texts the prompt card shows."""
        listing = None
        if self.profile_service is not None:
            try:
                listing = self.profile_service.listing()
            except Exception:
                log.warning("prompt profile listing failed", exc_info=True)
        self.listing = listing
        names = list(listing.names) if listing is not None and listing.names else list(self._given_profiles)
        self.profiles = chat_profile_order(names)

    def profile_options(self) -> list:
        """The picker's ``(key, text)`` options: translation profiles, then the "Specialised" heading and
        the task-specific built-ins."""
        options: list = []
        for title, names in grouped_profile_names(self.profiles):
            if title == SPECIALISED_GROUP:
                options.append((GROUP_OPTION_PREFIX + title, title))
            options.extend((name, name) for name in names)
        return options

    @property
    def profile_actions_reason(self) -> Optional[str]:
        """Why Edit prompt / New profile… are off (None: available)."""
        service = self.profile_service
        if service is None:
            return "Needs the settings store"
        if not getattr(service, "available", False):
            return "Needs the prompt profiles core"
        return None

    def _prompt_card(self) -> ft.Control:
        from glossarion_mobile.ui.screens.profiles import ProfileList

        name = str(self.value_of("profile") or "")
        listing = self.listing
        if listing is None:  # no settings store: what config.json holds, read-only
            stored = self.config.get("prompt_profiles", None)
            texts = dict(stored) if isinstance(stored, dict) else {}
            listing = ProfileList(names=list(self.profiles), texts=texts)
        return profile_prompt_card(
            listing, name, missing=self.listing is not None and name not in self.listing.texts,
            role_user=bool(self.config.get("system_prompt_to_user", False)),
            skip=bool(self.value_of("skip_prompt_profile")),
            on_edit=self.edit_prompt, on_new=self.new_profile, on_manage=self.manage_profiles,
            disabled_reason=self.profile_actions_reason,
            mono=mono_family(self._page), key="setting-profile-prompt",
        )

    def _settings_ctx(self) -> Any:
        """The SettingsContext the prompt editor opens with: the one given, else the app's settings
        context, else one around this sheet's page and store."""
        if self.ctx is not None:
            return self.ctx
        try:
            from glossarion_mobile.ui.sheets.model_sheet import sheet_env

            ctx = sheet_env().ctx
        except Exception:
            ctx = None
        if ctx is not None:
            return ctx
        from glossarion_mobile.ui.settings.context import SettingsContext

        return SettingsContext(page=self._page, store=self.config, schema=None)

    def _say(self, message: str) -> None:
        say = getattr(self._settings_ctx(), "say", None)
        if callable(say):
            say(message)

    def edit_prompt(self, name: Optional[str] = None) -> Any:
        """Edit prompt: the shared profile's text in ``PromptEditorSheet`` (placeholder chips, token count;
        Reset to default for a built-in). Saving changes the profile itself."""
        from glossarion_mobile.ui.screens.prompt_editor import PromptEditorSheet

        name = str(name or self.value_of("profile") or "")
        reason = self.profile_actions_reason
        listing = self.listing
        if reason or listing is None or name not in listing.texts:
            self._say(reason or f"The prompt profile '{name}' no longer exists")
            return None
        editor = PromptEditorSheet(
            self._settings_ctx(), title=name, subtitle="Prompt profile · shared with every chat and the desktop",
            value=str(listing.texts.get(name, "") or ""), default=listing.defaults.get(name),
            model=lambda: str(self.value_of("model") or ""),
            on_save=lambda text, n=name: self.save_prompt(n, text),
        )
        editor.on_saved = lambda _value: self._profiles_changed()  # once the editor has closed
        self.prompt_editor = editor
        editor.show()
        return editor

    def save_prompt(self, name: str, text: Any) -> Optional[str]:
        """Store an edited prompt (None, or the error the editor shows). Unchanged text writes nothing; a
        built-in put back to its default is the shared reset (exactly the default text); anything else is
        Save Profile under the same name. The global active profile stays as it is."""
        listing = self.listing
        service = self.profile_service
        if listing is None or service is None:
            return self.profile_actions_reason or "Prompt profiles are not available"
        text = str(text or "")
        if text.strip() == str(listing.texts.get(name, "") or "").strip():
            return None
        default = listing.defaults.get(name)
        try:
            if listing.is_builtin(name) and default is not None and text.strip() == str(default).strip():
                service.delete_or_reset(name, keep_active=True)
            else:
                service.save(name, name, text, keep_active=True)
        except Exception as exc:
            return str(exc) or type(exc).__name__
        return None

    def new_profile(self) -> Any:
        """New profile…: a name dialog; the profile starts as a copy of the current one and becomes this
        chat's (This chat), this series' (This series) or the active (All chats) profile, then opens in the
        editor."""
        reason = self.profile_actions_reason
        if reason:
            self._say(reason)
            return None
        source = str(self.value_of("profile") or "")
        texts = self.listing.texts if self.listing is not None else {}
        text = str(texts.get(source, "") or "")
        try:
            initial = self.profile_service.copy_name(source, self.listing) if source in texts else ""
        except Exception:
            initial = ""
        dialog, field_ = ask_profile_name(
            self._page if self._page is not None else getattr(self._settings_ctx(), "page", None),
            title="New profile",
            note=f"Starts as a copy of '{source}'." if source in texts else "Starts empty.",
            initial=initial,
            confirm_label="Create",
            on_submit=lambda name: self.create_profile(name, text),
            on_done=lambda name: self.edit_prompt(name),
        )
        self.new_profile_dialog = (dialog, field_)
        return dialog

    def create_profile(self, name: str, text: str) -> Optional[str]:
        """Create ``name`` with ``text`` without switching the global profile, then make it this scope's
        profile (None, or the error the name dialog shows)."""
        if self.profile_service is None:
            return self.profile_actions_reason
        try:
            created = self.profile_service.save_as(name, text, keep_active=True)
        except Exception as exc:
            return str(exc) or type(exc).__name__
        self.refresh_profiles()
        # This chat: the sidecar override; This series: the series default; All chats: write_setting ->
        # ProfileService.select (active profile + the desktop extraction-method switch)
        self.set_value("profile", created)
        return None

    def manage_profiles(self) -> None:
        """Manage…: Settings › Profiles & prompts (rename, delete / reset, import / export)."""
        if self.on_manage_profiles is not None:
            self.on_manage_profiles()
            return
        self.close()
        go = getattr(self._settings_ctx(), "go", None)
        if callable(go):
            go("settings.profiles")

    def _profiles_changed(self) -> None:
        self.refresh_profiles()
        self.rebuild()
        self._changed()

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
                    self._dropdown("profile", "Prompt profile", self.profile_options(),
                                   missing="{value} (missing)" if self.listing is not None else "{value}"),
                    self._prompt_card(),
                    self._language_row(),
                    self._dropdown("output_mode", "Default output mode", [(m.id, f"{m.emoji} {m.label}") for m in OUTPUT_MODES]),
                ],
            ),
            "glossary": ft.ExpansionTile(
                title="Glossary",
                expanded=self.expanded["glossary"],
                controls=[
                    ft.Row([ft.Container(expand=True), *self._badge("glossary_override_mode")]),
                    glossary_group,
                    *([self._switch("auto_accept_glossary", "Always accept generated glossaries")]
                      if self.shows_auto_accept else []),
                ],
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
        for tile in self.sections.values():  # U11 item 5: rows of a section were packed edge to edge
            tile.controls_padding = ft.Padding.only(left=16, right=16, bottom=12)
            tile.controls = _spaced([c for c in tile.controls if getattr(c, "data", None) != ROW_GAP_DATA])
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
