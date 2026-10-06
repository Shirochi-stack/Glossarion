"""Glossary settings tabs: General · Balanced/Full · Minimal · Refinement (UI_SPEC §4.1 "Settings tabs").

Each tab is the shared schema-driven ``SectionPage`` (settings tiles bound to
``MobileConfigStore``; 🔒 purple "Locked by mode: …" badges from ``settings_rules`` through
``SchemaAccess.lock_reason``; ReasonChips for unavailable settings; live refresh when any
key changes, so the mode locks follow the mode at once) over the Glossary Manager's
schema sections, with a few additions where the desktop tab has them:

* General: the mode row of the desktop General tab (``auto_glossary_mode`` and the
  mode-locked Append Glossary / Auto-Mapping / Fuzzy toggles) above ``glossary.general``,
  plus links to the Unified glossary and the Glossary editor preferences;
* Balanced/Full: the Balanced/Full prompt profile bar (``balanced_full`` bucket) above
  ``glossary.balanced_full``, plus the Anti-Duplicate Parameters sub-page;
* Minimal: the Minimal prompt profile bar above ``glossary.minimal``;
* Refinement: the Refinement profile bar (system + user prompt pairs) above
  ``glossary.refinement``.

The profile bars are the shared ``glossary_document.GlossaryPromptProfiles`` /
``RefinementPromptProfiles`` rows (``GlossaryService.prompt_profiles``: New / Save / Delete
with the desktop rules and texts); editing the prompt tile stages the text into the active
profile, like the desktop ``_auto_save_glossary_prompt_profile``. Changes auto-save; ⋯ Discard
changes since opening comes from SectionPage.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import flet as ft

from glossarion_mobile.services.glossary import PROFILE_BUCKETS, CoreMissing
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.empty_state import EmptyState
from glossarion_mobile.ui.glossary.common import ask, prompt_text
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["GlossarySettingsTab", "ProfileBar", "TAB_SECTIONS", "MODE_KEYS"]

log = logging.getLogger("glossarion.glossary.ui")

#: The desktop General tab's mode row (translator_gui shortcut combo + the mode-locked toggles).
MODE_KEYS = ("auto_glossary_mode", "append_glossary", "append_glossary_auto_load", "fuzzy_auto_mapping",
             "fuzzy_auto_mapping_threshold")

#: tab -> (title, schema section ids, profile bucket, prompt keys of the bucket, extra leading keys)
TAB_SECTIONS = {
    "general": ("General", ("glossary.general",), None, (), MODE_KEYS),
    "balanced": ("Balanced / Full Generation", ("glossary.balanced_full",), "balanced_full",
                 ("manual_glossary_prompt3",), ()),
    "minimal": ("Minimal Generation", ("glossary.minimal",), "minimal", ("unified_auto_glosary_prompt3",), ()),
    "refinement": ("Refinement", ("glossary.refinement",), "refinement",
                   ("glossary_refinement_system_prompt", "glossary_refinement_user_prompt"), ()),
}


class ProfileBar:
    """Profile dropdown + ＋ New · 💾 Save · 🗑 Delete for one Glossary Manager prompt row.

    The row is the shared ``glossary_document.GlossaryPromptProfiles`` /
    ``RefinementPromptProfiles`` (``GlossaryService.prompt_profiles``): every rule, box text and
    log line is the desktop row's; this bar renders its names, maps the buttons to its methods
    and writes what changed to config.json after each action (``persist_prompt_profiles``).
    The prompt tiles below edit the runtime prompt keys; such an edit is staged into the
    selected profile (``_auto_save_glossary_prompt_profile``).
    """

    def __init__(self, ctx: Any, bucket_id: str, *, prompt_keys: Sequence[str]) -> None:
        self.ctx = ctx
        self.service = ctx.service
        self.settings = ctx.settings
        self.bucket_id = bucket_id
        self.prompt_keys = tuple(prompt_keys)
        self.label = PROFILE_BUCKETS[bucket_id]
        self.pair = bucket_id == "refinement"
        self.profiles: Any = None
        self.error: Optional[str] = None
        self._applying = False
        self._unsub: Any = None
        self._stale = False
        self.dropdown = ft.Dropdown(label=self.label, options=[], expand=True, dense=True,
                                    on_select=lambda e: self.select(self.dropdown.value), key=f"pb-{bucket_id}")
        self.note = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                            visible=False, key=f"pb-note-{bucket_id}")
        self.control = ft.Container(content=ft.Column([
            ft.Row([
                self.dropdown,
                ft.IconButton(icon=ft.Icons.ADD, tooltip="+ New Profile", size_constraints=HIT_TARGET,
                              on_click=lambda e: self.new(), key=f"pb-new-{bucket_id}"),
                ft.IconButton(icon=ft.Icons.SAVE_OUTLINED, tooltip="💾 Save Profile", size_constraints=HIT_TARGET,
                              on_click=lambda e: self.ctx.spawn(self.save()), key=f"pb-save-{bucket_id}"),
                ft.IconButton(icon=ft.Icons.DELETE_OUTLINE, tooltip="🗑 Delete Profile", size_constraints=HIT_TARGET,
                              on_click=lambda e: self.ctx.spawn(self.delete()), key=f"pb-delete-{bucket_id}"),
            ], spacing=0, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            self.note,
        ], spacing=2, tight=True), bgcolor=ft.Colors.SURFACE_CONTAINER_LOW, border_radius=tokens.RADII["card"],
            padding=ft.Padding.symmetric(horizontal=8, vertical=4), key=f"pb-card-{bucket_id}")
        self.load()

    def _default_text(self) -> str:
        schema = getattr(self.settings, "schema", None)
        try:
            return str(schema.effective_default(self.prompt_keys[0]) or "") if schema is not None else ""
        except Exception:
            return ""

    def load(self) -> None:
        """Open the row on the current config (the desktop tab opening)."""
        try:
            self.profiles = self.service.prompt_profiles(self.bucket_id, default_text=self._default_text())
            self.error = None
        except CoreMissing as exc:
            self.profiles = None
            self.error = f"Prompt profiles need {exc.name} (not in this build)"
        self._stale = False
        self.render()

    @property
    def selected(self) -> str:
        return self.profiles.selected_name() if self.profiles is not None else ""

    @property
    def names(self) -> list:
        return list(self.profiles.names) if self.profiles is not None else []

    def render(self) -> None:
        if self.profiles is None:
            self.dropdown.options = []
            self.dropdown.disabled = True
            self.note.value = self.error or ""
            self.note.visible = bool(self.error)
        else:
            self.dropdown.disabled = False
            self.dropdown.options = [ft.DropdownOption(key=name, text=name) for name in self.names]
            self.dropdown.value = self.selected
            self.note.visible = False
        self.ctx.push(self.control)

    def _persist(self) -> bool:
        self._applying = True
        try:
            return self.service.persist_prompt_profiles(self.profiles)
        finally:
            self._applying = False

    def _box(self, box: Any) -> Any:
        """A desktop warning box of the row ("Default Profile", "Save Failed", ...) as a snackbar."""
        if box:
            title, text = box
            self.ctx.say(str(text or title))
        return box

    def select(self, name: Optional[str]) -> Any:
        if self.profiles is None or not name:
            return None
        result = self.profiles.select(name)
        if result is None:
            return None
        self._persist()
        self.render()
        return result

    def new(self) -> Optional[str]:
        if self.profiles is None:
            return None
        result = self.profiles.new()
        if self.pair and result:  # RefinementPromptProfiles.new returns the "Save Failed" box
            self._box(result)
            self.render()
            return None
        name = self.selected
        self._persist()
        self.render()
        self.ctx.say(f"Created profile “{name}” — edit the prompt below")
        return name

    def current_content(self) -> Any:
        store = getattr(self.settings, "store", None)
        get = store.get if store is not None else self.service.cfg
        if self.pair:
            return {"system": str(get(self.prompt_keys[0], "") or ""), "user": str(get(self.prompt_keys[1], "") or "")}
        return str(get(self.prompt_keys[0], "") or "")

    async def save(self, name: Optional[str] = None) -> Any:
        """Save Profile under ``name`` (asked, the selected name pre-filled: the desktop's edited combo text)."""
        if self.profiles is None:
            return None
        if name is None:
            name = await prompt_text(self.ctx, title="Save Profile", label="Profile name", value=self.selected)
            if name is None:
                return None
        box = self.profiles.save(name, self.current_content())
        if box:
            self.render()
            return self._box(box)
        self._persist()
        self.render()
        self.ctx.say(f"Saved profile “{self.selected}”")
        return None

    async def delete(self, *, confirm: bool = True) -> Any:
        if self.profiles is None:
            return None
        name = self.selected
        asked: list = []

        def answered(_name: str) -> bool:
            return bool(asked and asked[0])

        if confirm and name in self.names[1:]:  # the desktop asks only for an existing named profile
            asked.append(await ask(self.ctx, title="Delete Profile",
                                   body=f"Delete {'refinement' if self.pair else 'glossary'} prompt profile '{name}'?",
                                   confirm="Yes", cancel="No", destructive=True))
        box = self.profiles.delete(name, confirm=answered if confirm else None)
        if box:
            return self._box(box)
        if confirm and not answered(name):
            return None
        self._persist()
        self.render()
        return None

    def on_prompt_changed(self) -> None:
        """A prompt tile edit stages into the selected profile (``_auto_save_glossary_prompt_profile``)."""
        if self._applying or self.profiles is None:
            return
        content = self.current_content()
        if self.pair:
            if {k: v.strip() for k, v in content.items()} == {k: v.strip() for k, v in self.profiles.pair.items()}:
                return
        elif content.strip() == str(self.profiles.text or "").strip():
            return
        self.profiles.stage(self.selected, content)
        self._persist()

    def attach(self) -> None:
        if self._stale:
            self.load()  # config.json may have changed while the tab was hidden: open the row again
        store = getattr(self.settings, "store", None)
        if store is None or self._unsub is not None:
            return
        try:
            self._unsub = store.observe_keys(self.prompt_keys, lambda key, value: self.ctx.post_ui(
                self.on_prompt_changed))
        except Exception:
            self._unsub = None

    def detach(self) -> None:
        if self._unsub is not None:
            try:
                self._unsub()
            except Exception:
                pass
            self._unsub = None
        self._stale = True


class GlossarySettingsTab:
    """One settings tab: a SectionPage over a synthetic section (extra keys + the schema sections)."""

    def __init__(self, ctx: Any, tab: str) -> None:
        self.ctx = ctx
        self.tab = tab
        self.title, self.section_ids, self.bucket_id, self.prompt_keys, self.leading_keys = TAB_SECTIONS[tab]
        self.page: Any = None
        self.profile_bar: Optional[ProfileBar] = None
        self.root: Optional[ft.Control] = None
        self.shown = False

    def keys(self) -> list:
        schema = self.ctx.settings.schema
        keys: list = []
        for key in self.leading_keys:
            if schema.spec(key) is not None and key not in keys:
                keys.append(key)
        for section_id in self.section_ids:
            section = schema.section(section_id)
            for key in (section.keys if section is not None else ()):
                if key not in keys:
                    keys.append(key)
        return keys

    def build(self) -> ft.Control:
        settings = self.ctx.settings
        if settings is None or not getattr(settings.schema, "available", False):
            self.root = EmptyState(icon="SETTINGS", title="Settings are not available",
                                   body="This build has no settings schema; config.json is kept untouched.",
                                   key=f"gs-{self.tab}-none")
            return self.root
        from glossarion_mobile.ui.settings.schema_access import SectionInfo
        from glossarion_mobile.ui.settings.section_page import SectionPage

        page = SectionPage(None, settings, section_id=self.section_ids[0])
        page.section = SectionInfo(id=f"glossary.tab.{self.tab}", title=self.title, keys=tuple(self.keys()),
                                   group="Glossary")
        page.title = self.title
        self.page = page
        body = page.get_body()
        extras: list = []
        if self.bucket_id is not None:
            self.profile_bar = ProfileBar(self.ctx, self.bucket_id, prompt_keys=self.prompt_keys)
            extras.append(self.profile_bar.control)
        links = self._links()
        if links:
            extras.append(ft.Row(links, wrap=True, spacing=6, run_spacing=4))
        controls = ([ft.Container(content=ft.Column(extras, spacing=6, tight=True),
                                  padding=ft.Padding.only(left=12, right=12, top=6))] if extras else []) + [body]
        self.root = ft.Column(controls, spacing=4, expand=True, key=f"gs-{self.tab}")
        return self.root

    def _links(self) -> list:
        feature = self.ctx.feature
        links: list = []
        if self.tab == "general":
            links.append(ft.TextButton(content="Unified glossary…", icon=ft.Icons.MERGE_TYPE,
                                       on_click=lambda e: self.ctx.go("glossary.unified"), key="gs-unified"))
            links.append(ft.TextButton(content="Editor preferences…", icon=ft.Icons.EDIT_NOTE,
                                       on_click=lambda e: self.ctx.settings.open_setting("glossary.editor"),
                                       key="gs-editor-prefs"))
        elif self.tab == "balanced":
            links.append(ft.TextButton(content="Anti-Duplicate Parameters…", icon=ft.Icons.TUNE,
                                       on_click=lambda e: self.ctx.settings.open_setting("glossary.anti_duplicate"),
                                       key="gs-antidup"))
        elif self.tab == "refinement" and feature is not None:
            links.append(ft.TextButton(content="✨ Refine a glossary now…", on_click=lambda e: feature.refine_from_settings(),
                                       key="gs-refine-now"))
        return links

    def did_show(self) -> None:
        if self.page is not None and not self.shown:
            self.shown = True
            self.page.did_show()
        if self.profile_bar is not None:
            self.profile_bar.attach()

    def dispose(self) -> None:
        if self.page is not None:
            self.page.dispose()
        if self.profile_bar is not None:
            self.profile_bar.detach()
        self.shown = False
