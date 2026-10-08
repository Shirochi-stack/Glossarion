"""Profiles & prompts (``/settings/profiles``, ``/settings/profiles/<pid>``; UI_SPEC §4.14).

* **Profiles list:** every prompt profile in desktop order (built-in badge, a dot when a
  built-in differs from its default, the active one marked), FAB "New profile", app-bar
  ⋯ Import / Export (JSON, the desktop "Import Profiles" / "Export Profiles" format), a
  segment switch to the **All prompts** index (``all_prompts.AllPromptsView``) and a link
  to Assistant prefill (``/settings/prefill``).
* **Profile page:** name, the ``PromptEditorPane`` (mono, ``{target_lang}`` /
  ``{split_marker_instruction}`` chips, token count), the System ⇄ User role switch
  (config ``system_prompt_to_user``, desktop ↕️/🔀), the extraction note for
  ``*_BeautifulSoup`` / ``*_html2text`` profiles, and Save · Save as · Use this profile ·
  Reset to default (built-ins) / Delete (custom).

All profile semantics come from the shared ``prompt_profiles`` core (moved verbatim from
other_settings ``on_profile_select`` / ``save_profile`` / ``delete_profile`` /
``save_profiles`` / ``import_profiles`` / ``export_profiles`` and the main window's
``_quick_new_profile``): built-ins are protected (Delete becomes "Reset to default"),
saving a custom profile under a new name renames it in place, saving a built-in under a
new name makes a copy, selecting a ``*_BeautifulSoup`` / ``*_html2text`` profile switches
``text_extraction_method``. ``ProfilesCore`` is the only place that talks to that module;
``ProfileService`` turns each operation into a sparse ``MobileConfigStore.set_many`` of
exactly the keys the desktop writes (``prompt_profiles``, ``active_profile``,
``profile_name_autofill``, ``profile_mousewheel_locked``, ``text_extraction_method``), so
a profile edited on the phone round-trips into a desktop config unchanged.

Routes carry an opaque profile id (``profile_id(name)``: 12 hex of SHA-1), never the name.

Shared with the chat (device fixes, owner #15/#16): ``ProfileService.listing`` is the one profile
list (the chat pickers order it with ``chat_profile_order``: translation profiles first, the
task-specific built-ins under "Specialised"); ``keep_active=True`` edits a profile without
switching the global/desktop active profile (Chat settings › This chat); ``ask_profile_name``,
``profile_badges``, ``prompt_preview`` and ``profile_prompt_card`` are the pieces Chat settings
reuses; ``follow_profile`` tells where a chat's or series' stored profile went after a rename
(noted by ``ProfileService.save``) or a delete (None: inherit).
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib
import inspect
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.dialogs import ConfirmDialog, close_dialog
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.screens.prompt_editor import PLACEHOLDERS, PromptEditorPane
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = [
    "PROFILE_CONFIG_KEYS",
    "SPECIALISED_GROUP",
    "SPECIALISED_PROFILES",
    "TRANSLATION_GROUP",
    "ProfileDetailScreen",
    "ProfileList",
    "ProfileService",
    "ProfilesCore",
    "ProfilesScreen",
    "ProfilesUnavailable",
    "ask_profile_name",
    "chat_profile_order",
    "extraction_note",
    "follow_profile",
    "grouped_profile_names",
    "note_renamed",
    "profile_badges",
    "profile_group",
    "profile_id",
    "profile_prompt_card",
    "prompt_preview",
]

log = logging.getLogger("glossarion.profiles")

#: config.json keys the desktop profile actions write (other_settings ``save_profiles`` and
#: ``on_profile_select``); only these are copied back after a core operation.
PROFILE_CONFIG_KEYS = (
    "prompt_profiles",
    "active_profile",
    "profile_name_autofill",
    "profile_mousewheel_locked",
    "text_extraction_method",
)
ROLE_KEY = "system_prompt_to_user"

#: Built-in profiles made for one task (refinement, manga, glossary cleanup, game tools, image
#: generation, SDLXLIFF, subtitles, pass-through). The chat's profile pickers list them after the
#: translation profiles under "Specialised" (owner decision 2026-10-08); Settings › Profiles &
#: prompts keeps the desktop order of everything, like the desktop Profile combo.
SPECIALISED_PROFILES = frozenset((
    "Refinement",
    "Manga_JP",
    "Manga_KR",
    "Manga_CN",
    "Glossary_Editor",
    "RPGMaker_GTool",
    "RPGMaker_GTool_Image",
    "NanoBanana_Image",
    "Original",
    "SDLXLIFF Editing v2",
    "Subtitle Translation",
))
TRANSLATION_GROUP = "Translation"
SPECIALISED_GROUP = "Specialised"


def profile_group(name: str) -> str:
    """``SPECIALISED_GROUP`` for a task-specific built-in, else ``TRANSLATION_GROUP`` (custom profiles too)."""
    return SPECIALISED_GROUP if str(name) in SPECIALISED_PROFILES else TRANSLATION_GROUP


def grouped_profile_names(names: Any) -> list:
    """``[(group title, [names])]``: translation profiles first, then "Specialised"; each group keeps the
    listing's (desktop) order and an empty group is left out."""
    groups: dict = {TRANSLATION_GROUP: [], SPECIALISED_GROUP: []}
    for name in names or ():
        groups[profile_group(name)].append(name)
    return [(title, items) for title, items in groups.items() if items]


def chat_profile_order(names: Any) -> list:
    """The names in the chat pickers' order (``grouped_profile_names`` flattened)."""
    return [name for _title, items in grouped_profile_names(names) for name in items]


#: Profiles renamed on this device (old name -> new name). ``ProfileService.save`` notes a rename before
#: the write reaches config.json, so a ``prompt_profiles`` observer (the chat feature) can move the chats
#: and series that use the old name to the new one (``follow_profile``).
_RENAMED: dict = {}


def note_renamed(old: Any, new: Any) -> None:
    if old and new and str(old) != str(new):
        _RENAMED[str(old)] = str(new)


def follow_profile(name: Any, names: Any) -> Optional[str]:
    """What a chat's or series' stored profile choice means now: ``name`` while it exists, the profile it
    was renamed to on this device (following renames of renames), else None: the profile is gone and the
    chat inherits the series / All chats profile (never a silent first profile)."""
    known = set(names or ())
    current = str(name or "")
    seen: set = set()
    while current and current not in known:
        if current in seen:
            return None
        seen.add(current)
        current = _RENAMED.get(current, "")
    return current or None


def prompt_preview(text: Any, lines: int = 3) -> str:
    """The first ``lines`` non-empty lines of a prompt (stripped), or "(empty)"."""
    picked = [line.strip() for line in str(text or "").splitlines() if line.strip()][:max(1, int(lines))]
    return "\n".join(picked) if picked else "(empty)"


class ProfilesUnavailable(RuntimeError):
    """The shared ``prompt_profiles`` core is missing in this build (or lacks an operation)."""


def profile_id(name: str) -> str:
    """Opaque route id of a profile name (route pattern ``slug``)."""
    return hashlib.sha1(str(name).encode("utf-8")).hexdigest()[:12]


def extraction_note(name: str) -> Optional[str]:
    """The desktop auto-switch of ``on_profile_select`` described for the profile page."""
    lowered = str(name or "").lower()
    if "beautifulsoup" in lowered:
        return "Using this profile switches text extraction to BeautifulSoup (standard)."
    if "html2text" in lowered:
        return "Using this profile switches text extraction to html2text (enhanced)."
    return None


@dataclass
class ProfileList:
    """What the screens show: profiles in desktop order + their flags."""

    names: list = field(default_factory=list)
    texts: dict = field(default_factory=dict)
    active: str = ""
    protected: frozenset = frozenset()
    defaults: dict = field(default_factory=dict)

    def is_builtin(self, name: str) -> bool:
        return name in self.protected

    def is_modified(self, name: str) -> bool:
        # outer whitespace does not count: the shared save_profile strips what it saves (desktop too), so
        # a built-in saved back unchanged (or "Reset to default" + Save in an editor) is not "modified"
        return name in self.defaults and \
            str(self.texts.get(name, "") or "").strip() != str(self.defaults.get(name, "") or "").strip()

    def name_for(self, pid: str) -> Optional[str]:
        for name in self.names:
            if profile_id(name) == pid:
                return name
        return None


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if not isinstance(obj, Mapping) and hasattr(obj, name):
            return getattr(obj, name)
    return default


#: Non-error string results of the core operations.
_OK_STRINGS = ("reset", "deleted", "saved", "ok")


def _error_of(result: Any, *, string_is_error: bool = False) -> Optional[str]:
    """Normalise a core result: ``False`` / ``result.error`` -> message, else None.

    ``string_is_error``: the operation returns ``None`` or an error message (save, delete);
    other operations return a value (``select`` the prompt text, ``new`` the new name).
    """
    if result is False:
        return "The profile could not be saved."
    if isinstance(result, str):
        return result if (string_is_error and result and result not in _OK_STRINGS) else None
    error = _get(result, "error", default=None) if result is not None and not isinstance(result, bool) else None
    return str(error) if error else None


#: Parameter names the core functions may use for each value the adapter supplies.
_ALIASES = {
    "state": ("state", "profile_state", "profiles_state", "owner", "prefill_state"),
    "profiles": ("prompt_profiles", "profiles"),
    "config": ("config", "cfg"),
    "name": ("name", "profile_name", "new_name", "target_name"),
    "content": ("content", "text", "prompt", "prompt_text"),
    "source": ("source_name", "source", "old_name", "current_name"),
    "data": ("data", "imported", "incoming", "imported_profiles", "new_profiles"),
    "active": ("active_profile", "active", "profile_var", "active_name"),
}


def bind_call(fn: Callable[..., Any], values: Mapping[str, Any], aliases: Optional[Mapping[str, tuple]] = None) -> Any:
    """Call a core function, binding each of its parameters by name (``_ALIASES``).

    The adapter supplies ``state`` / ``profiles`` / ``config`` / ``name`` / ``content`` /
    ``source`` / ``data``; a required parameter it cannot fill raises ``ProfilesUnavailable``
    (the core's API differs from the contract), optional ones keep their defaults.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        raise ProfilesUnavailable(f"prompt_profiles.{getattr(fn, '__name__', fn)}() has no signature") from None
    table = aliases if aliases is not None else _ALIASES
    positional: list = []
    keywords: dict = {}
    for param in params.values():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        key = next((k for k, names in table.items() if param.name in names and k in values), None)
        if key is None:
            if param.default is inspect.Parameter.empty:
                raise ProfilesUnavailable(
                    f"prompt_profiles.{getattr(fn, '__name__', fn)}() needs {param.name!r}, which this app does not supply")
            continue
        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional.append(values[key])
        else:
            keywords[param.name] = values[key]
    return fn(*positional, **keywords)


class ProfilesCore:
    """Adapter over the shared ``prompt_profiles`` module (the only place this screen calls it).

    Expected core API (U4 shared contract): ``ProfileState`` built from a config dict
    (``profile_state_from_config(config)`` or ``ProfileState.from_config(config)``) with
    ``prompt_profiles`` / ``profile_var`` (active) / ``default_prompts`` / ``protected``;
    ``select_profile(state, name, config)``, ``save_profile(state, name, content, config)``,
    ``new_profile(state, config)``, ``delete_or_reset_profile(state, name, config)``,
    ``merge_imported_profiles(prompt_profiles, data)``, ``export_profiles_json(prompt_profiles)``
    and ``apply_profiles_to_config(state, config)`` (the ``save_profiles`` key set) - or, without
    it, ``write_profiles_to_config_file`` run against a scratch file. The operations mutate
    ``state`` / ``config`` the way the desktop mutates ``self`` / ``self.config``;
    ``ProfileService`` then writes the changed keys sparsely. Arguments are bound by parameter
    name (``bind_call``), so the adapter follows the core's own signatures.
    """

    MODULE = "prompt_profiles"

    def __init__(self, module: Any = None) -> None:
        self._module = module
        self._tried = module is not None
        self.error: Optional[str] = None

    @property
    def module(self) -> Any:
        if not self._tried:
            self._tried = True
            try:
                self._module = importlib.import_module(self.MODULE)
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                self._module = None
        return self._module

    @property
    def available(self) -> bool:
        return self.module is not None

    def fn(self, *names: str) -> Callable[..., Any]:
        module = self.module
        if module is None:
            raise ProfilesUnavailable(f"The prompt profiles core is not available in this build ({self.error}).")
        for name in names:
            candidate = getattr(module, name, None)
            if callable(candidate):
                return candidate
        raise ProfilesUnavailable(f"prompt_profiles has no {names[0]}()")

    @staticmethod
    def _profiles_of(state: Any) -> dict:
        return dict(_get(state, "prompt_profiles", "profiles", default={}) or {})

    @staticmethod
    def _active_of(state: Any) -> str:
        return str(_get(state, "profile_var", "active_profile", "active", default="") or "")

    @staticmethod
    def _set_profiles(state: Any, profiles: Mapping) -> None:
        for attr in ("prompt_profiles", "profiles"):
            if isinstance(state, dict) and attr in state:
                state[attr] = dict(profiles)
                return
            if not isinstance(state, dict) and hasattr(state, attr):
                setattr(state, attr, dict(profiles))
                return

    def _call(self, names: tuple, state: Any, **values: Any) -> Any:
        fn = self.fn(*names)
        result = bind_call(fn, {"state": state, "profiles": self._profiles_of(state), "active": self._active_of(state),
                                **values})
        bound = set(inspect.signature(fn).parameters)
        if isinstance(result, Mapping) and not (bound & set(_ALIASES["state"])) and bound & set(_ALIASES["profiles"]):
            self._set_profiles(state, result)  # a dict-in / dict-out core function (e.g. merge)
        return result

    # ---- state ----------------------------------------------------------------------------

    def state(self, config: dict) -> Any:
        module = self.module
        if module is None:
            raise ProfilesUnavailable(f"The prompt profiles core is not available in this build ({self.error}).")
        factory = getattr(module, "profile_state_from_config", None)
        if not callable(factory):
            cls = getattr(module, "ProfileState", None)
            factory = getattr(cls, "from_config", None) if cls is not None else None
        if not callable(factory):
            raise ProfilesUnavailable("prompt_profiles has no profile_state_from_config()")
        try:
            return factory(config)
        except ImportError as exc:  # the desktop start-up it replays needs the backend packages (owner_state)
            self.error = f"{type(exc).__name__}: {exc}"
            raise ProfilesUnavailable(f"The prompt profiles core cannot start in this build ({self.error}).") from exc

    def listing(self, state: Any) -> ProfileList:
        profiles = dict(_get(state, "prompt_profiles", "profiles", default={}) or {})
        protected = _get(state, "protected", "protected_profiles", default=None)
        if callable(protected):
            protected = protected()
        return ProfileList(
            names=list(profiles),
            texts=profiles,
            active=str(_get(state, "profile_var", "active_profile", "active", default="") or ""),
            protected=frozenset(protected or ()),
            defaults=dict(_get(state, "default_prompts", "defaults", default={}) or {}),
        )

    # ---- operations (mutate state + config like the desktop handlers) -------------------------

    def select(self, state: Any, name: str, config: dict) -> Any:
        return self._call(("select_profile",), state, name=name, config=config)

    def save(self, state: Any, source: str, name: str, content: str, config: dict) -> Any:
        return self._call(("save_profile",), state, name=name, content=content, source=source, config=config)

    def new(self, state: Any, config: dict) -> Any:
        return self._call(("new_profile", "quick_new_profile"), state, config=config)

    def delete_or_reset(self, state: Any, name: str, config: dict) -> Any:
        return self._call(("delete_or_reset_profile",), state, name=name, config=config)

    def merge_imported(self, state: Any, data: Any, config: dict) -> Any:
        return self._call(("merge_imported_profiles",), state, data=data, config=config)

    def export_json(self, state: Any) -> str:
        return str(self._call(("export_profiles_json",), state))

    def persist(self, state: Any, config: dict) -> None:
        """Put the ``save_profiles`` key set into ``config`` (the desktop file write, minus the file)."""
        module = self.module
        if module is not None and not callable(getattr(module, "apply_profiles_to_config", None)) and \
                not callable(getattr(module, "profiles_config_updates", None)) and \
                callable(getattr(module, "write_profiles_to_config_file", None)):
            self._persist_via_file_writer(state, config)
            return
        result = self._call(("apply_profiles_to_config", "profiles_config_updates"), state, config=config)
        if isinstance(result, Mapping):
            config.update(result)

    def _persist_via_file_writer(self, state: Any, config: dict) -> None:
        """Run the core's desktop config writer against a scratch copy and take its profile keys."""
        handle, path = tempfile.mkstemp(prefix="glossarion_profiles_", suffix=".json")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as scratch:
                json.dump({}, scratch)
            writer = self.fn("write_profiles_to_config_file")
            values = {"state": state, "profiles": self._profiles_of(state), "active": self._active_of(state),
                      "config": config}
            try:
                params = inspect.signature(writer).parameters
            except (TypeError, ValueError):
                params = {}
            path_name = next((n for n in params if n in ("config_file", "path", "config_path", "file_path",
                                                          "filename")), None)
            if path_name is None:
                raise ProfilesUnavailable("prompt_profiles.write_profiles_to_config_file() takes no config path")
            values["path"] = path
            result = bind_call(writer, values, {**_ALIASES, "path": (path_name,)})
            if result is False:
                raise ValueError("Failed to save profiles")
            with open(path, "r", encoding="utf-8") as written:
                data = json.load(written)
            for key in PROFILE_CONFIG_KEYS:
                if key in data:
                    config[key] = data[key]
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


class ProfileService:
    """Profile operations against ``MobileConfigStore`` (sparse writes of the desktop key set)."""

    def __init__(self, store: Any, *, core: Optional[ProfilesCore] = None) -> None:
        self.store = store
        self.core = core or ProfilesCore()
        self.message: str = ""

    @property
    def available(self) -> bool:
        return self.core.available

    @property
    def error(self) -> Optional[str]:
        return self.core.error

    def _config(self) -> dict:
        return self.store.snapshot() if self.store is not None else {}

    def core_listing(self) -> Optional[ProfileList]:
        """The listing desktop start-up builds (the shared core), or None when the core cannot run here."""
        try:
            return self.core.listing(self.core.state(self._config()))
        except ProfilesUnavailable:
            return None

    def listing(self) -> ProfileList:
        """Blocking-light (pure dict work): the profiles as the desktop shows them."""
        listing = self.core_listing()
        if listing is not None:
            return listing
        config = self._config()
        profiles = config.get("prompt_profiles") if isinstance(config.get("prompt_profiles"), dict) else {}
        return ProfileList(names=list(profiles), texts=dict(profiles), active=str(config.get("active_profile") or ""))

    def _run(self, op: Callable[[Any, dict], Any], *, string_is_error: bool = False, keep_active: bool = False,
             before_write: Optional[Callable[[Any], Any]] = None) -> tuple[Any, list]:
        """Run one core operation on a config copy and write the changed desktop keys.

        ``keep_active``: the operation edits a profile without choosing it (Chat settings › This chat):
        ``active_profile`` and ``text_extraction_method`` keep their stored values (an absent key stays
        absent) - unless the profile in use (the desktop start-up rule's choice) was renamed or deleted,
        where the core's choice stands like on the desktop. ``before_write(result)`` runs once the
        operation succeeded, before config.json observers hear of it."""
        config = self._config()
        before = copy.deepcopy(config)
        state = self.core.state(config)
        in_use = self.core._active_of(state)
        result = op(state, config)
        error = _error_of(result, string_is_error=string_is_error)
        if error:
            raise ValueError(error)
        self.core.persist(state, config)
        if keep_active:
            profiles = config.get("prompt_profiles")
            if isinstance(profiles, Mapping) and in_use in profiles:
                for key in ("active_profile", "text_extraction_method"):
                    if key in before:
                        config[key] = before[key]
                    else:
                        config.pop(key, None)
        if before_write is not None:
            before_write(result)
        updates = {key: config[key] for key in PROFILE_CONFIG_KEYS
                   if key in config and (key not in before or before[key] != config[key])}
        changed = self.store.set_many(updates) if updates else []
        return result, changed

    def select(self, name: str) -> list:
        _result, changed = self._run(lambda state, config: self.core.select(state, name, config))
        return changed

    def save(self, source: str, name: str, content: str, *, keep_active: bool = False) -> str:
        """Desktop Save Profile: rename a custom profile, copy a built-in under a new name. A rename is
        noted (``note_renamed``) so chats and series using the old name follow it."""
        name = str(name or "").strip()
        if not name:
            raise ValueError("Profile cannot be empty.")
        renamed: list = []

        def op(state: Any, config: dict) -> Any:
            had_source = source in self.core._profiles_of(state)
            result = self.core.save(state, source, name, content, config)
            after = self.core._profiles_of(state)
            if had_source and source != name and source not in after and name in after:
                renamed.append(source)
            return result

        self._run(op, string_is_error=True, keep_active=keep_active,
                  before_write=lambda _result: [note_renamed(old, name) for old in renamed])
        return name

    def save_as(self, name: str, content: str, *, keep_active: bool = False) -> str:
        """A new profile with ``content`` (``save_profile`` with the new name as its own source)."""
        name = str(name or "").strip()
        if not name:
            raise ValueError("Profile cannot be empty.")
        if name in self.listing().texts:
            raise ValueError("A profile with this name already exists. Choose another name.")
        self._run(lambda state, config: self.core.save(state, name, name, content, config), string_is_error=True,
                  keep_active=keep_active)
        return name

    def copy_name(self, name: str, listing: Optional[ProfileList] = None) -> str:
        """The first free "<name> (copy)" / "<name> (copy N)" (Duplicate's naming)."""
        texts = (listing or self.listing()).texts
        candidate, n = f"{name} (copy)", 2
        while candidate in texts:
            candidate = f"{name} (copy {n})"
            n += 1
        return candidate

    def duplicate(self, name: str, *, keep_active: bool = False) -> str:
        listing = self.listing()
        return self.save_as(self.copy_name(name, listing), listing.texts.get(name, ""), keep_active=keep_active)

    def new(self, *, keep_active: bool = False) -> str:
        result, _changed = self._run(lambda state, config: self.core.new(state, config), keep_active=keep_active)
        if isinstance(result, str) and result:
            return result
        return self.listing().active

    def delete_or_reset(self, name: str, *, keep_active: bool = False) -> str:
        """``"reset"`` for a built-in (latest default prompt), ``"deleted"`` for a custom profile."""
        builtin = name in self.listing().protected
        result, _changed = self._run(lambda state, config: self.core.delete_or_reset(state, name, config),
                                     string_is_error=True, keep_active=keep_active)
        if isinstance(result, str) and result in ("reset", "deleted"):
            return result
        return "reset" if builtin else "deleted"

    def import_json(self, text: str) -> int:
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise ValueError(f"Failed to import profiles: {exc}") from None
        if not isinstance(data, dict):
            raise ValueError("Failed to import profiles: the file does not contain a profiles object.")
        self._run(lambda state, config: self.core.merge_imported(state, data, config), string_is_error=True)
        return len(data)

    def export_json(self) -> str:
        return self.core.export_json(self.core.state(self._config()))

    def role_is_user(self) -> bool:
        return bool(self.store.get(ROLE_KEY, False)) if self.store is not None else False

    def set_role_user(self, value: bool) -> None:
        if self.store is not None:
            self.store.set(ROLE_KEY, bool(value))


# ==========================================================================================
# Shared pieces (the Profiles screens and Chat settings)
# ==========================================================================================


def profile_badges(listing: ProfileList, name: str, *, show_active: bool = True) -> list:
    """The "Active" pill and the dot of a built-in that differs from its default."""
    badges: list = []
    if show_active and name == listing.active:
        badges.append(ft.Container(content=ft.Text("Active", theme_style=ft.TextThemeStyle.LABEL_SMALL,
                                                   color=ft.Colors.ON_PRIMARY),
                                   bgcolor=ft.Colors.PRIMARY, border_radius=8,
                                   padding=ft.Padding.symmetric(horizontal=6, vertical=2)))
    if listing.is_modified(name):
        badges.append(ft.Container(width=8, height=8, border_radius=4, bgcolor=ft.Colors.PRIMARY,
                                   tooltip="Differs from the built-in default"))
    return badges


def ask_profile_name(
    page: Any,
    *,
    title: str,
    on_submit: Callable[[str], Optional[str]],
    on_done: Optional[Callable[[str], Any]] = None,
    label: str = "New profile name",
    initial: str = "",
    note: Optional[str] = None,
    confirm_label: str = "Save",
    push: Optional[Callable[..., Any]] = None,
) -> tuple:
    """The profile-name dialog (Profile page "Save as", Chat settings "New profile…"). ``on_submit(name)``
    stores it and returns an error message (shown on the field, the dialog stays) or None (the dialog
    closes, then ``on_done(name)`` runs). Returns ``(dialog, field)``."""
    field_ = ft.TextField(label=label, value=initial or "", autofocus=True, dense=True)
    done_names: list = []

    def done(e: Any = None) -> None:
        if done_names:  # a second tap while the dialog closes
            return
        name = str(field_.value or "")
        error = on_submit(name)
        if error:  # the error shows where the user is looking (under the dialog nothing is visible)
            field_.error = str(error)
            if push is not None:
                push(field_)
            else:
                try:
                    field_.update()
                except Exception:
                    pass
            return
        done_names.append(name.strip())
        # By identity: on_submit may have shown a snackbar, which pop_dialog() would close instead.
        close_dialog(page, dialog)
        if on_done is not None:
            on_done(name.strip())

    content: Any = field_
    if note:
        content = ft.Column([ft.Text(note, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
                             field_], tight=True, spacing=8)
    dialog = ft.AlertDialog(
        title=ft.Text(title),
        content=content,
        actions=[ft.TextButton(content="Cancel", on_click=lambda e: close_dialog(page, dialog)),
                 ft.FilledButton(content=confirm_label, on_click=done)],
    )
    if page is not None:
        page.show_dialog(dialog)
    return dialog, field_


def profile_prompt_card(
    listing: ProfileList,
    name: str,
    *,
    role_user: bool = False,
    skip: bool = False,
    on_edit: Optional[Callable[[], Any]] = None,
    on_new: Optional[Callable[[], Any]] = None,
    on_manage: Optional[Callable[[], Any]] = None,
    disabled_reason: Optional[str] = None,
    missing: Optional[bool] = None,
    mono: str = "monospace",
    key: str = "profile-prompt-card",
) -> ft.Control:
    """One profile's prompt as a card (Chat settings › Model & prompt): the desktop role label
    ("System prompt" / "User prompt", ``system_prompt_to_user``), Built-in / modified marks, the first lines,
    the size, the extraction and skip notes, and Edit prompt · New profile… · Manage…. A profile that no
    longer exists (``missing``; default: not in the listing) can only be replaced (Edit is off)."""
    from glossarion_mobile.ui.settings.editors import count_label

    missing = (name not in listing.texts) if missing is None else bool(missing)
    text = str(listing.texts.get(name, "") or "")
    builtin = listing.is_builtin(name) or name in listing.defaults
    heading = ft.Row(
        [ft.Text("User prompt" if role_user else "System prompt", theme_style=ft.TextThemeStyle.LABEL_MEDIUM,
                 expand=True),
         *([ft.Text("Built-in", theme_style=ft.TextThemeStyle.LABEL_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)]
           if builtin and not missing else []),
         *([] if missing else profile_badges(listing, name, show_active=False))],
        spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )
    rows: list = [heading]
    if missing:
        rows.append(ft.Text(f"The prompt profile '{name}' no longer exists. Choose another one, or reset this "
                            "row to inherit.", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ERROR,
                            key=f"{key}-missing"))
    else:
        rows.append(ft.Text(prompt_preview(text), font_family=mono, size=12, max_lines=3,
                            overflow=ft.TextOverflow.ELLIPSIS, color=ft.Colors.ON_SURFACE_VARIANT,
                            key=f"{key}-preview"))
        rows.append(ft.Text(count_label(text), theme_style=ft.TextThemeStyle.LABEL_SMALL,
                            color=ft.Colors.ON_SURFACE_VARIANT))
        note = extraction_note(name)
        if note:
            rows.append(ft.Text(note, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT))
    if skip:
        rows.append(ft.Text("Skip prompt profile is on: runs ignore this prompt.",
                            theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                            key=f"{key}-skip"))
    rows.append(ft.Text("Profiles are shared: editing changes this profile everywhere it is used, including "
                        "the desktop.", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT))
    off = bool(disabled_reason)
    rows.append(ft.Row(
        [ft.TextButton(content="Edit prompt", icon=ft.Icons.EDIT_NOTE, on_click=lambda e: on_edit() if on_edit else None,
                       disabled=off or missing or on_edit is None, key=f"{key}-edit"),
         ft.TextButton(content="New profile…", icon=ft.Icons.ADD, on_click=lambda e: on_new() if on_new else None,
                       disabled=off or on_new is None, key=f"{key}-new"),
         ft.TextButton(content="Manage…", icon=ft.Icons.TUNE, on_click=lambda e: on_manage() if on_manage else None,
                       disabled=on_manage is None, key=f"{key}-manage")],
        wrap=True, spacing=0,
    ))
    if off:
        rows.append(ReasonChip(reason=str(disabled_reason)))
    return ft.Container(
        content=ft.Column(rows, spacing=4, tight=True),
        padding=ft.Padding.only(left=12, right=8, top=8, bottom=4),
        border_radius=tokens.RADII["card"],
        bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
        key=key,
    )


# ==========================================================================================
# Screens
# ==========================================================================================


class _ProfilesBase(Screen):
    def __init__(self, match: Any, ctx: Any, *, service: Optional[ProfileService] = None,
                 model: Callable[[], str] = lambda: "") -> None:
        super().__init__(match)
        self.ctx = ctx
        self.service = service or ProfileService(getattr(ctx, "store", None))
        self.model = model

    def say(self, message: str) -> None:
        say = getattr(self.ctx, "say", None)
        if callable(say):
            say(message)
        else:
            log.info("profiles: %s", message)

    def push(self, *controls: Any) -> None:
        push = getattr(self.ctx, "push", None)
        if callable(push):
            push(*controls)

    def unavailable_notice(self) -> Optional[ft.Control]:
        if self.service.available:
            return None
        return ft.Container(
            padding=12,
            border_radius=tokens.RADII["card"],
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            content=ft.Row([
                ft.Icon(ft.Icons.INFO_OUTLINE),
                ft.Text("Editing profiles needs the shared prompt_profiles core, which is missing in this build. "
                        "Your profiles in config.json are shown read-only and kept untouched.",
                        theme_style=ft.TextThemeStyle.BODY_SMALL, expand=True),
            ], spacing=8),
            key="profiles-unavailable",
        )


class ProfilesScreen(_ProfilesBase):
    title = "Profiles & prompts"

    def __init__(self, match: Any, ctx: Any, *, service: Optional[ProfileService] = None,
                 model: Callable[[], str] = lambda: "", share_file: Optional[Callable[[str], Any]] = None,
                 pick_files: Optional[Callable[..., Any]] = None, temp_dir: Optional[str] = None) -> None:
        super().__init__(match, ctx, service=service, model=model)
        self.share_file = share_file
        self.pick_files = pick_files
        self.temp_dir = temp_dir
        self.listing = ProfileList()
        self.rows: dict[str, ft.ListTile] = {}
        self.segment = "profiles"
        self.list_view = ft.ListView(expand=True, spacing=4, padding=ft.Padding.only(left=8, right=8, bottom=88))
        self.all_prompts: Any = None
        self.sheet: Any = None
        self._unsub: Optional[Callable[[], None]] = None

    # ---- body ------------------------------------------------------------------------------

    def actions(self) -> list[ft.Control]:
        return [ft.IconButton(icon=ft.Icons.MORE_VERT, tooltip="Import / Export", size_constraints=HIT_TARGET,
                              on_click=lambda e: self.overflow())]

    def build_body(self) -> ft.Control:
        self.switch = ft.SegmentedButton(
            segments=[ft.Segment(value="profiles", label=ft.Text("Profiles")),
                      ft.Segment(value="all", label=ft.Text("All prompts"))],
            selected=["profiles"],
            on_change=lambda e: self.show_segment(next(iter(e.control.selected or ["profiles"]))),
        )
        self.prefill_link = ft.ListTile(
            leading=ft.Icon(ft.Icons.SHORT_TEXT),
            title=ft.Text("Assistant prefill"),
            subtitle=ft.Text("Optional assistant message sent before your content",
                             theme_style=ft.TextThemeStyle.BODY_SMALL),
            trailing=ft.Icon(ft.Icons.CHEVRON_RIGHT),
            on_click=lambda e: self.ctx.go("settings.prefill"),
            min_height=tokens.SIZES["hit_target"],
            key="profiles-prefill-link",
        )
        self.content = ft.Container(content=self.list_view, expand=True)
        self.fab = ft.FloatingActionButton(icon=ft.Icons.ADD, content="New profile", on_click=lambda e: self.new_profile(),
                                           disabled=not self.service.available, key="profiles-new")
        self.refresh(push=False)
        controls: list[ft.Control] = [ft.Container(padding=ft.Padding.only(left=12, right=12, top=8), content=self.switch)]
        notice = self.unavailable_notice()
        if notice is not None:
            controls.append(ft.Container(padding=ft.Padding.symmetric(horizontal=12), content=notice))
        controls.append(self.content)
        return ft.Stack([
            ft.Column(controls, spacing=6, expand=True),
            ft.Container(content=self.fab, right=16, bottom=16),
        ], expand=True)

    def _row(self, name: str) -> ft.ListTile:
        listing = self.listing
        builtin = listing.is_builtin(name)
        first = prompt_preview(listing.texts.get(name, ""), 1)
        badges = profile_badges(listing, name)
        row = ft.ListTile(
            leading=ft.Icon(ft.Icons.VERIFIED_OUTLINED if builtin else ft.Icons.DESCRIPTION_OUTLINED,
                            tooltip="Built-in profile" if builtin else "Custom profile"),
            title=ft.Row([ft.Text(name, theme_style=ft.TextThemeStyle.BODY_MEDIUM, expand=True,
                                  max_lines=1, overflow=ft.TextOverflow.ELLIPSIS), *badges],
                         spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            subtitle=ft.Text(("Built-in · " if builtin else "") + first[:120], max_lines=1,
                             overflow=ft.TextOverflow.ELLIPSIS, theme_style=ft.TextThemeStyle.BODY_SMALL,
                             color=ft.Colors.ON_SURFACE_VARIANT),
            on_click=lambda e, n=name: self.open_profile(n),
            on_long_press=lambda e, n=name: self.row_actions(n),
            min_height=tokens.SIZES["hit_target"],
            dense=True,
            key=f"profile-{profile_id(name)}",
        )
        self.rows[name] = row
        return row

    def refresh(self, push: bool = True) -> ProfileList:
        self.listing = self.service.listing()
        self.rows = {}
        self.list_view.controls = [self.prefill_link] + [self._row(name) for name in self.listing.names]
        if push:
            self.push(self.list_view)
        return self.listing

    def show_segment(self, segment: str) -> None:
        self.segment = segment
        if segment == "all":
            if self.all_prompts is None:
                from glossarion_mobile.ui.screens.all_prompts import AllPromptsView

                self.all_prompts = AllPromptsView(self.ctx)
            self.content.content = self.all_prompts
            self.fab.visible = False
        else:
            self.content.content = self.list_view
            self.fab.visible = True
            self.refresh(push=False)
        self.push(self.content, self.fab)

    # ---- lifecycle -------------------------------------------------------------------------

    def did_show(self) -> None:
        store = getattr(self.ctx, "store", None)
        if self._unsub is None and store is not None:
            on_ui = getattr(self.ctx, "on_ui", lambda fn, *a: fn(*a))
            self._unsub = store.observe_keys(("prompt_profiles", "active_profile"),
                                             lambda key, value: on_ui(self._config_changed))

    def dispose(self) -> None:
        if self._unsub is not None:
            self._unsub()
            self._unsub = None

    def _config_changed(self) -> None:
        if self.segment == "profiles":
            self.refresh()

    # ---- actions ----------------------------------------------------------------------------

    def open_profile(self, name: str) -> Optional[str]:
        return self.ctx.go("settings.profiles.detail", {"pid": profile_id(name)})

    def new_profile(self) -> Optional[str]:
        try:
            name = self.service.new()
        except Exception as exc:
            self.say(str(exc))
            return None
        self.say(f"✅ Created new profile: '{name}'")
        self.refresh()
        self.open_profile(name)
        return name

    def row_actions(self, name: str) -> ActionSheet:
        builtin = self.listing.is_builtin(name)
        items = [
            ActionItem("Use this profile", lambda: self.use(name), icon="CHECK_CIRCLE_OUTLINE",
                       disabled_reason="Already active" if name == self.listing.active else None),
            ActionItem("Duplicate", lambda: self.duplicate(name), icon="CONTENT_COPY",
                       disabled_reason=None if self.service.available else "Needs the profiles core"),
            ActionItem("Reset to default" if builtin else "Delete", lambda: self.confirm_delete(name),
                       icon="RESTART_ALT" if builtin else "DELETE_OUTLINE", destructive=not builtin,
                       disabled_reason=None if self.service.available else "Needs the profiles core"),
        ]
        sheet = ActionSheet(items, title=name, tablet=bool(getattr(self.ctx, "tablet", False)))
        self.sheet = sheet
        page = getattr(self.ctx, "page", None)
        if page is not None:
            sheet.show(page)
        return sheet

    def use(self, name: str) -> bool:
        try:
            self.service.select(name)
        except Exception as exc:
            self.say(str(exc))
            return False
        self.refresh()
        return True

    def duplicate(self, name: str) -> Optional[str]:
        try:
            new = self.service.duplicate(name)
        except Exception as exc:
            self.say(str(exc))
            return None
        self.refresh()
        self.say(f"✅ Profile '{new}' saved")
        return new

    def confirm_delete(self, name: str) -> ConfirmDialog:
        builtin = self.listing.is_builtin(name)
        dialog = ConfirmDialog(
            title="Reset Profile" if builtin else "Delete",
            body=(f"Reset built-in profile '{name}' to the latest default prompt?" if builtin
                  else f"Are you sure you want to delete profile '{name}'?"),  # desktop delete_profile text
            confirm_label="Reset" if builtin else "Delete",
            cancel_label="No",
            destructive=not builtin,
            on_confirm=lambda: self.delete_or_reset(name),
        )
        page = getattr(self.ctx, "page", None)
        if page is not None:
            dialog.show(page)
        return dialog

    def delete_or_reset(self, name: str) -> Optional[str]:
        try:
            kind = self.service.delete_or_reset(name)
        except Exception as exc:
            self.say(str(exc))
            return None
        self.refresh()
        self.say(f"Profile '{name}' reset to its default" if kind == "reset" else f"Profile '{name}' deleted")
        return kind

    def overflow(self) -> ActionSheet:
        needs = None if self.service.available else "Needs the profiles core"
        sheet = ActionSheet([
            ActionItem("Import profiles…", lambda: self._spawn(self.import_profiles()), icon="FILE_OPEN_OUTLINED",
                       disabled_reason=needs or (None if self.pick_files else "File picker unavailable")),
            ActionItem("Export profiles…", lambda: self._spawn(self.export_profiles()), icon="IOS_SHARE",
                       disabled_reason=needs or (None if self.share_file else "Sharing unavailable")),
        ], title="Prompt profiles", tablet=bool(getattr(self.ctx, "tablet", False)))
        self.sheet = sheet
        page = getattr(self.ctx, "page", None)
        if page is not None:
            sheet.show(page)
        return sheet

    async def import_profiles(self, path: Optional[str] = None) -> int:
        if path is None:
            if self.pick_files is None:
                return 0
            picked = self.pick_files(["json"], False)
            picked = (await picked) if asyncio.iscoroutine(picked) else picked
            if not picked:
                return 0
            path = str(picked[0])

        def read() -> str:
            with open(path, "r", encoding="utf-8") as handle:
                return handle.read()

        try:
            text = await self.ctx.run_io(read)
            count = self.service.import_json(text)
        except Exception as exc:
            self.say(str(exc) if str(exc).startswith("Failed") else f"Failed to import profiles: {exc}")
            return 0
        self.refresh()
        self.say(f"Imported {count} profiles.")
        return count

    def export_path(self) -> str:
        directory = self.temp_dir or tempfile.gettempdir()
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, "glossarion_profiles.json")

    async def export_profiles(self) -> Optional[str]:
        try:
            text = self.service.export_json()
        except Exception as exc:
            self.say(f"Failed to export profiles: {exc}")
            return None
        path = self.export_path()

        def write() -> None:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)

        await self.ctx.run_io(write)
        if self.share_file is not None:
            result = self.share_file(path)
            if asyncio.iscoroutine(result):
                await result
        return path

    @staticmethod
    def _spawn(coro: Any) -> Any:
        try:
            return asyncio.ensure_future(coro)
        except RuntimeError:
            coro.close()
            return None


class ProfileDetailScreen(_ProfilesBase):
    title = "Profile"

    def __init__(self, match: Any, ctx: Any, *, service: Optional[ProfileService] = None,
                 model: Callable[[], str] = lambda: "", mono: str = "monospace",
                 on_close: Optional[Callable[[], Any]] = None) -> None:
        super().__init__(match, ctx, service=service, model=model)
        self.mono = mono
        self.on_close = on_close
        self.pid = str(getattr(match, "params", {}).get("pid", "")) if match is not None else ""
        self.listing = self.service.listing()
        self.name = self.listing.name_for(self.pid) or ""
        self.title = self.name or "Profile"
        self.error_text = ft.Text("", color=ft.Colors.ERROR, theme_style=ft.TextThemeStyle.BODY_SMALL, visible=False)

    def build_body(self) -> ft.Control:
        if not self.name:
            return ft.Container(padding=16, content=ft.Text("This profile no longer exists.", key="profile-missing"))
        listing = self.listing
        builtin = listing.is_builtin(self.name)
        editable = self.service.available
        self.name_field = ft.TextField(label="Profile name", value=self.name, dense=True, read_only=not editable)
        self.editor = PromptEditorPane(value=listing.texts.get(self.name, ""), placeholders=PLACEHOLDERS, mono=self.mono,
                                       run_io=getattr(self.ctx, "run_io", None), model=self.model)
        self.editor.field.read_only = not editable
        self.role_switch = ft.Switch(
            label="Send as user message (🔀)", value=self.service.role_is_user(),
            on_change=lambda e: self.service.set_role_user(bool(e.control.value)),
        )
        rows: list[ft.Control] = []
        notice = self.unavailable_notice()
        if notice is not None:
            rows.append(notice)
        rows.append(ft.Row([self.name_field] + ([ReasonChip(reason="Built-in", detail="Built-in profiles cannot be "
                                                            "deleted; Reset restores the latest default prompt.")]
                                                if builtin else []),
                           vertical_alignment=ft.CrossAxisAlignment.CENTER))
        note = extraction_note(self.name)
        if note:
            rows.append(ft.Text(note, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                                key="profile-extraction-note"))
        rows.append(self.role_switch)
        rows.append(self.editor)
        rows.append(self.error_text)
        self.save_button = ft.FilledButton(content="Save", on_click=lambda e: self.save(), disabled=not editable)
        self.save_as_button = ft.TextButton(content="Save as", on_click=lambda e: self.ask_save_as(), disabled=not editable)
        self.use_button = ft.TextButton(content="Use this profile", on_click=lambda e: self.use(),
                                        disabled=not editable or self.name == listing.active)
        self.delete_button = ft.TextButton(
            content="Reset to default" if builtin else "Delete",
            style=None if builtin else ft.ButtonStyle(color=ft.Colors.ERROR),
            on_click=lambda e: self.confirm_delete(), disabled=not editable,
        )
        rows.append(ft.Row([self.save_button, self.save_as_button, self.use_button, self.delete_button],
                           wrap=True, spacing=4))
        return ft.Container(padding=12, expand=True, content=ft.Column(rows, spacing=8, expand=True))

    def _error(self, message: Optional[str]) -> None:
        self.error_text.value = message or ""
        self.error_text.visible = bool(message)
        self.push(self.error_text)

    def save(self) -> Optional[str]:
        new_name = (self.name_field.value or "").strip()
        try:
            saved = self.service.save(self.name, new_name, self.editor.value)
        except Exception as exc:
            self._error(str(exc))
            return None
        self._error(None)
        self.editor.mark_saved()
        renamed = saved != self.name
        self.name = saved
        self.title = saved
        self.pid = profile_id(saved)
        self.listing = self.service.listing()
        self.say(f"✅ Profile '{saved}' saved")
        if renamed:
            self.ctx.go("settings.profiles.detail", {"pid": self.pid})
        return saved

    def ask_save_as(self) -> ft.AlertDialog:
        def submit(name: str) -> Optional[str]:
            if self.save_as(name):
                return None
            # the page's error line is under the dialog: the field shows it
            return self.error_text.value or "The profile could not be saved."

        dialog, field_ = ask_profile_name(getattr(self.ctx, "page", None), title="Save as", on_submit=submit,
                                          push=self.push)
        self.save_as_dialog = (dialog, field_)
        return dialog

    def save_as(self, name: str) -> Optional[str]:
        try:
            saved = self.service.save_as(name, self.editor.value)
        except Exception as exc:
            self._error(str(exc))
            return None
        self._error(None)
        self.say(f"✅ Profile '{saved}' saved")
        self.ctx.go("settings.profiles.detail", {"pid": profile_id(saved)})
        return saved

    def use(self) -> bool:
        try:
            self.service.select(self.name)
        except Exception as exc:
            self._error(str(exc))
            return False
        self.listing = self.service.listing()
        self.use_button.disabled = True
        self.push(self.use_button)
        self.say(f"Using profile '{self.name}'")
        return True

    def confirm_delete(self) -> ConfirmDialog:
        builtin = self.listing.is_builtin(self.name)
        dialog = ConfirmDialog(
            title="Reset Profile" if builtin else "Delete",
            body=(f"Reset built-in profile '{self.name}' to the latest default prompt?" if builtin
                  else f"Are you sure you want to delete profile '{self.name}'?"),
            confirm_label="Reset" if builtin else "Delete",
            cancel_label="No",
            destructive=not builtin,
            on_confirm=self.delete_or_reset,
        )
        page = getattr(self.ctx, "page", None)
        if page is not None:
            dialog.show(page)
        return dialog

    def delete_or_reset(self) -> Optional[str]:
        try:
            kind = self.service.delete_or_reset(self.name)
        except Exception as exc:
            self._error(str(exc))
            return None
        if kind == "reset":
            self.listing = self.service.listing()
            self.editor.set_value(self.listing.texts.get(self.name, ""))
            self.say(f"Profile '{self.name}' reset to its default")
        else:
            self.say(f"Profile '{self.name}' deleted")
            if self.on_close is not None:
                self.on_close()
        return kind
