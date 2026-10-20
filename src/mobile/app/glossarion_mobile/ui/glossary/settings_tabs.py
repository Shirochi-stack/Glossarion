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
from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.services.glossary import PROFILE_BUCKETS, PROMPT_PLACEHOLDERS, CoreMissing
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.empty_state import EmptyState
from glossarion_mobile.ui.glossary.common import ask, attach_mode_locks, prompt_text
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["GlossaryPromptCard", "GlossarySettingsScreen", "GlossarySettingsTab", "ProfileBar", "SECTION_TABS", "TAB_SECTIONS", "MODE_KEYS"]

log = logging.getLogger("glossarion.glossary.ui")

#: The desktop General tab's mode row (translator_gui shortcut combo + the mode-locked toggles).
MODE_KEYS = ("auto_glossary_mode", "append_glossary", "append_glossary_auto_load", "fuzzy_auto_mapping",
             "fuzzy_auto_mapping_threshold")

#: tab -> (title, schema section ids, profile bucket, prompt keys of the bucket, extra leading keys)
TAB_SECTIONS = {
    "general": ("General", ("glossary.general",), None, (), MODE_KEYS),
    # glossary_target_language leads both generation tabs (desktop Glossary Manager Balanced/Full and
    # Minimal tabs: "Target language" combo above the prompt; schema section main.prompt).
    "balanced": ("Balanced / Full Generation", ("glossary.balanced_full",), "balanced_full",
                 ("manual_glossary_prompt3",), ("glossary_target_language",)),
    "minimal": ("Minimal Generation", ("glossary.minimal",), "minimal", ("unified_auto_glosary_prompt3",),
                ("glossary_target_language",)),
    "refinement": ("Refinement", ("glossary.refinement",), "refinement",
                   ("glossary_refinement_system_prompt", "glossary_refinement_user_prompt"), ()),
}


#: Settings › Glossary section ids -> the tab that hosts them (UI_SPEC §4.1 / §4.15: "also at
#: Settings › Glossary, the same pages as the Glossaries tabs").
SECTION_TABS = {
    "glossary.general": "general",
    "glossary.balanced_full": "balanced",
    "glossary.minimal": "minimal",
    "glossary.refinement": "refinement",
}


class ProfileBar:
    """Profile dropdown + ＋ New · 💾 Save · 🗑 Delete for one Glossary Manager prompt row.

    The row is the shared ``glossary_document.GlossaryPromptProfiles`` /
    ``RefinementPromptProfiles`` (``GlossaryService.prompt_profiles``): every rule, box text and
    log line is the desktop row's; this bar renders its names, maps the buttons to its methods
    and writes what changed to config.json after each action (``persist_prompt_profiles``).
    The prompt tiles below edit the runtime prompt keys; such an edit is staged into the
    selected profile (``_auto_save_glossary_prompt_profile``).

    Other prompt profile rows reuse the bar with a ``source`` (``open()`` -> a shared profiles object,
    ``persist(profiles)`` -> bool), their ``label``, the profile's prompt ``fields`` (one per prompt key)
    and the ``noun`` of its box texts: the review generator's ``review_generator.ReviewPromptProfiles``
    (tools/review.py). A profiles object with built-ins (``is_builtin`` / ``reset``:
    ``prompt_profiles.PromptSetProfiles``) turns 🗑 Delete into ↺ Reset for them. ``on_change`` runs after
    every render.
    """

    def __init__(self, ctx: Any, bucket_id: str, *, prompt_keys: Sequence[str], source: Any = None,
                 label: Optional[str] = None, fields: Optional[Sequence[str]] = None, noun: Optional[str] = None,
                 on_change: Optional[Callable[[], Any]] = None) -> None:
        self.ctx = ctx
        self.service = getattr(ctx, "service", None)
        self.settings = getattr(ctx, "settings", None)
        self.bucket_id = bucket_id
        self.prompt_keys = tuple(prompt_keys)
        self.source = source
        self.label = label or PROFILE_BUCKETS[bucket_id]
        self.fields = tuple(fields) if fields else (("system", "user") if bucket_id == "refinement" else ())
        self.pair = bool(self.fields)
        self.noun = noun or ("refinement" if bucket_id == "refinement" else "glossary")
        self.on_change = on_change
        self.profiles: Any = None
        self.error: Optional[str] = None
        self._applying = False
        self._unsub: Any = None
        self._stale = False
        self.dropdown = ft.Dropdown(label=self.label, options=[], expand=True, dense=True,
                                    on_select=lambda e: self.select(self.dropdown.value), key=f"pb-{bucket_id}")
        self.note = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                            visible=False, key=f"pb-note-{bucket_id}")
        self.delete_button = ft.IconButton(icon=ft.Icons.DELETE_OUTLINE, tooltip="🗑 Delete Profile",
                                           size_constraints=HIT_TARGET, key=f"pb-delete-{bucket_id}",
                                           on_click=lambda e: self.ctx.spawn(self.delete()))
        self.control = ft.Container(content=ft.Column([
            ft.Row([
                self.dropdown,
                ft.IconButton(icon=ft.Icons.ADD, tooltip="+ New Profile", size_constraints=HIT_TARGET,
                              on_click=lambda e: self.new(), key=f"pb-new-{bucket_id}"),
                ft.IconButton(icon=ft.Icons.SAVE_OUTLINED, tooltip="💾 Save Profile", size_constraints=HIT_TARGET,
                              on_click=lambda e: self.ctx.spawn(self.save()), key=f"pb-save-{bucket_id}"),
                self.delete_button,
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
            if self.source is not None:
                self.profiles = self.source.open()
            else:
                self.profiles = self.service.prompt_profiles(self.bucket_id, default_text=self._default_text())
            self.error = None
        except CoreMissing as exc:
            self.profiles = None
            self.error = f"Prompt profiles need {exc.name} (not in this build)"
        except Exception as exc:
            if self.source is None:
                raise
            log.warning("%s prompt profiles unavailable", self.noun, exc_info=True)
            self.profiles = None
            self.error = f"Prompt profiles unavailable ({exc})"
        self._stale = False
        self.render()

    @property
    def selected(self) -> str:
        return self.profiles.selected_name() if self.profiles is not None else ""

    @property
    def names(self) -> list:
        return list(self.profiles.names) if self.profiles is not None else []

    def is_builtin(self, name: Optional[str] = None) -> bool:
        """A built-in profile of a row that has Reset (``PromptSetProfiles``: Default and the shipped ones)."""
        check = getattr(self.profiles, "is_builtin", None)
        return bool(callable(check) and callable(getattr(self.profiles, "reset", None))
                    and check(self.selected if name is None else name))

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
        builtin = self.profiles is not None and self.is_builtin()
        self.delete_button.icon = ft.Icons.RESTORE if builtin else ft.Icons.DELETE_OUTLINE
        self.delete_button.tooltip = "↺ Reset Profile" if builtin else "🗑 Delete Profile"
        self.ctx.push(self.control)
        if self.on_change is not None:
            self.on_change()

    def _persist(self) -> bool:
        self._applying = True
        try:
            if self.source is not None:
                return self.source.persist(self.profiles)
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
        if self.pair and result:  # PromptSetProfiles.new returns the "Save Failed" box
            self._box(result)
            self.render()
            return None
        name = self.selected
        self._persist()
        self.render()
        return name

    def current_content(self) -> Any:
        if self.source is not None and self.profiles is not None and self.pair:
            return dict(self.profiles.texts)  # its prompt cards stage every edit into the profile itself
        store = getattr(self.settings, "store", None)
        get = store.get if store is not None else self.service.cfg
        if self.pair:
            return {name: str(get(key, "") or "") for name, key in zip(self.fields, self.prompt_keys)}
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
        return None

    async def delete(self, *, confirm: bool = True) -> Any:
        if self.profiles is None:
            return None
        name = self.selected
        if self.is_builtin(name):  # ↺ Reset Profile: the shipped prompts back
            if confirm and not await ask(self.ctx, title="Reset Profile",
                                         body=f"Reset the built-in '{name}' {self.noun} prompt profile to its "
                                              "shipped prompts?", confirm="Yes", cancel="No"):
                return None
            box = self.profiles.reset(name)
            if box:
                return self._box(box)
            self._persist()
            self.render()
            return None
        asked: list = []

        def answered(_name: str) -> bool:
            return bool(asked and asked[0])

        if confirm and name in self.names[1:]:  # the desktop asks only for an existing named profile
            asked.append(await ask(self.ctx, title="Delete Profile",
                                   body=f"Delete {self.noun} prompt profile '{name}'?",
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
            if {k: v.strip() for k, v in content.items()} == {k: v.strip() for k, v in self.profiles.texts.items()}:
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


class GlossaryPromptCard:
    """One Balanced/Full or Minimal glossary prompt profile as the shared prompt card of Chat settings ›
    Model & prompt (``profiles.profile_prompt_card``): its text, size, the "Profiles are shared" note and
    Edit prompt · New profile… · Manage…, for Chat settings › Glossary and the Plan glossary sheet.

    Edit prompt edits the profile itself (``GlossaryService.save_prompt_profile_text``: Save Profile of the
    bucket's row; Default is the bucket's Default prompt) without changing which profile is selected. New
    profile… copies it into a new profile of the bucket and hands the name to ``on_created`` (the caller
    selects it: this chat's override or the active profile). Manage… opens the bucket's Settings › Glossary
    tab. ``on_changed`` runs after a save (the caller re-reads its row)."""

    def __init__(self, service: Any, bucket_id: str, name: str, *, ctx: Any, page: Any = None,
                 default_text: str = "", on_changed: Optional[Callable[[], Any]] = None,
                 on_created: Optional[Callable[[str], Any]] = None, on_manage: Optional[Callable[[], Any]] = None,
                 model: Callable[[], str] = lambda: "", key: str = "glossary-prompt", preview_lines: int = 3) -> None:
        self.service = service
        self.bucket_id = bucket_id
        self.name = str(name or "")
        self.ctx = ctx  # the SettingsContext the prompt editor opens with
        self.page = page if page is not None else getattr(ctx, "page", None)
        self.default_text = default_text
        self.on_changed = on_changed
        self.on_created = on_created
        self.on_manage = on_manage
        self.model = model
        self.key = key
        self.preview_lines = preview_lines
        self.editor: Any = None
        self.name_dialog: Any = None
        self.texts: dict = {}
        self.builtins: dict = {}
        self.error: Optional[str] = None
        self.load()

    def load(self) -> None:
        """Read the bucket's profile texts (``{name: prompt}``) and its built-in texts."""
        try:
            self.texts = self.service.prompt_profile_texts(self.bucket_id, default_text=self.default_text)
            self.builtins = self.service.builtin_prompt_texts(self.bucket_id, default_text=self.default_text)
            self.error = None
        except Exception as exc:
            log.warning("glossary prompt profile texts unavailable", exc_info=True)
            self.texts, self.builtins = {}, {}
            self.error = f"Glossary prompt profiles need {getattr(exc, 'name', None) or 'the glossary core'}"

    def listing(self) -> Any:
        from glossarion_mobile.ui.screens.profiles import ProfileList

        names = list(self.texts)
        return ProfileList(names=names, texts=dict(self.texts), defaults=dict(self.builtins),
                           protected=frozenset(self.builtins) | frozenset(names[:1]))  # names[0]: Default

    def build(self) -> ft.Control:
        from glossarion_mobile.ui.screens.profiles import profile_prompt_card
        from glossarion_mobile.ui.theme import mono_family

        return profile_prompt_card(
            self.listing(), self.name, heading="Extraction prompt",
            missing=self.error is None and self.name not in self.texts,
            on_edit=self.edit_prompt, on_new=self.new_profile, on_manage=self.manage,
            disabled_reason=self.error, mono=mono_family(self.page), key=self.key,
            preview_lines=self.preview_lines,
        )

    def _say(self, message: str) -> None:
        say = getattr(self.ctx, "say", None)
        if callable(say):
            say(message)

    def edit_prompt(self, name: Optional[str] = None) -> Any:
        """Edit prompt: the profile's text in ``PromptEditorSheet`` (the bucket's placeholder chips, token
        count; Reset to default for Default and On-device). Saving changes the profile itself."""
        from glossarion_mobile.ui.screens.prompt_editor import PromptEditorSheet

        name = str(name or self.name)
        if self.error or name not in self.texts:
            self._say(self.error or f"The glossary prompt profile '{name}' no longer exists")
            return None
        editor = PromptEditorSheet(
            self.ctx, title=name,
            subtitle=f"{PROFILE_BUCKETS[self.bucket_id]} · glossary prompt · shared with every chat and the desktop",
            value=str(self.texts.get(name, "") or ""), default=self.builtins.get(name),
            placeholders=PROMPT_PLACEHOLDERS.get(self.bucket_id, ()), model=self.model,
            on_save=lambda text, n=name: self.save_prompt(n, text),
        )
        editor.on_saved = lambda _value: self._saved()  # once the editor has closed
        self.editor = editor
        editor.show()
        return editor

    def save_prompt(self, name: str, text: Any) -> Optional[str]:
        """Store an edited prompt (None, or the error the editor shows); unchanged text writes nothing."""
        if str(text or "").strip() == str(self.texts.get(name, "") or "").strip():
            return None
        try:
            error = self.service.save_prompt_profile_text(self.bucket_id, name, text, default_text=self.default_text)
        except Exception as exc:
            return str(exc) or type(exc).__name__
        if error is None:
            self.load()
        return error

    def _saved(self) -> None:
        if self.on_changed is not None:
            self.on_changed()

    def new_profile(self) -> Any:
        """New profile…: a name dialog; the profile starts as a copy of this one, is selected by
        ``on_created`` and opens in the editor."""
        from glossarion_mobile.ui.screens.profiles import ask_profile_name, copy_profile_name

        if self.error:
            self._say(self.error)
            return None
        source = self.name
        known = source in self.texts
        text = str(self.texts.get(source, "") or "")
        dialog, field_ = ask_profile_name(
            self.page, title="New glossary profile",
            note=f"Starts as a copy of '{source}'." if known else "Starts empty.",
            initial=copy_profile_name(source, self.texts) if known else "",
            confirm_label="Create",
            on_submit=lambda name: self.create_profile(name, text),
            on_done=lambda name: self.edit_prompt(name),
        )
        self.name_dialog = (dialog, field_)
        return dialog

    def create_profile(self, name: str, text: str) -> Optional[str]:
        """Create ``name`` holding ``text`` in the bucket, then let the caller select it (None, or the
        error the name dialog shows)."""
        try:
            error = self.service.create_prompt_profile(self.bucket_id, name, text, default_text=self.default_text)
        except Exception as exc:
            return str(exc) or type(exc).__name__
        if error:
            return error
        self.load()
        if self.on_created is not None:
            self.on_created(str(name).strip())
        return None

    def manage(self) -> None:
        """Manage…: the bucket's Settings › Glossary tab (its profile row, the prompt and its options)."""
        if self.on_manage is not None:
            self.on_manage()
            return
        open_setting = getattr(self.ctx, "open_setting", None)
        if callable(open_setting):
            open_setting(f"glossary.{self.bucket_id}")


class GlossarySettingsTab:
    """One settings tab: a SectionPage over a synthetic section (extra keys + the schema sections)."""

    def __init__(self, ctx: Any, tab: str, match: Any = None) -> None:
        self.ctx = ctx
        self.tab = tab
        self.match = match  # a Settings search hit / All prompts tap: #<key> is focused (U9)
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

        page = SectionPage(self.match, settings, section_id=self.section_ids[0])
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
            # include_book_title_glossary / auto_inject_book_title live in Metadata, TOC & headers
            links.append(ft.TextButton(content="Book title in glossary…", icon=ft.Icons.TITLE,
                                       on_click=lambda e: self.ctx.settings.open_setting(
                                           "other.meta_data", "include_book_title_glossary"),
                                       key="gs-book-title"))
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


class GlossarySettingsScreen(Screen):
    """``/settings/s/glossary.<tab>``: one Glossary Manager settings tab as a pushed screen (the mode
    row, the prompt profile bar and the links above the schema tiles), so prompt profiles and the
    glossary mode can be managed without opening a glossary file (GlossaryFeature routes the
    Settings › Glossary sections here)."""

    def __init__(self, match: Any, ctx: Any, tab: str) -> None:
        super().__init__(match)
        self.ctx = ctx
        self.tab = GlossarySettingsTab(ctx, tab, match=match)
        self.title = f"Glossary · {self.tab.title}"
        self._lock_unsubs: Optional[list] = None

    def build_body(self) -> ft.Control:
        return self.tab.build()

    def actions(self) -> list:
        """The SectionPage's Search and ⋯ Save now · Discard changes since opening · Reset this section."""
        self.get_body()  # the shell asks for the actions before the body
        page = self.tab.page
        return page.actions() if page is not None else []

    def did_show(self) -> None:
        if self._lock_unsubs is None:
            # U9: the Glossary Manager's mode locks here too (the Glossary view's lock pass, shared): a
            # mode switched on this page writes the locked toggles' forced values, as desktop does
            self._lock_unsubs = attach_mode_locks(self.ctx)
        self.tab.did_show()

    def dispose(self) -> None:
        for unsub in self._lock_unsubs or ():
            try:
                unsub()
            except Exception:
                pass
        self._lock_unsubs = None
        self.tab.dispose()
