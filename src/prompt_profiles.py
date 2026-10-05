"""Prompt profiles without their widgets: the shared core of the desktop profile actions.

Glossarion mobile rewrite, milestone U4 (shared-core design section 3.6, P5). Desktop and the
mobile app run the same functions:

* **Main prompt profiles** (main window "Profile" row, Other Settings): moved verbatim from
  other_settings ``on_profile_select`` / ``save_profile`` / ``delete_profile`` /
  ``save_profiles`` / ``import_profiles`` / ``export_profiles``, and TranslatorGUI
  ``_quick_new_profile`` as ``new_profile``. Each function takes a *state*
  object carrying the attributes those handlers read and write on ``TranslatorGUI``:
  ``prompt_profiles``, ``profile_var``, ``config``, ``_original_profile_content``,
  ``_active_profile_for_autosave``, ``text_extraction_method_var``, ``default_prompts`` and
  the ConfigStateMixin helpers ``_get_protected_prompt_profiles`` /
  ``_reset_prompt_profile_to_default``.

  - The desktop passes ``self``. The other_settings functions are thin wrappers: they keep
    the message boxes, the profile combo box, the prompt editor and the extraction radios,
    and hand those steps in as keyword hooks that run at the exact point of the original
    sequence (a combo box rebuild or an editor update fires Qt signals whose handlers read
    the state, so the order is part of the behaviour).
  - The mobile app passes a ``ProfileState`` from ``profile_state_from_config(config)``: the
    desktop start-up profile reconciliation (ConfigStateMixin._init_variables) run on a
    scratch owner, plus the same attributes. Without hooks the functions do what the
    desktop does without widgets (for example selecting the next profile after a delete).

* **Assistant prefill profiles** (TranslatorGUI.show_assistant_prompt_dialog, "Asst. Prompt"):
  ``PrefillState`` and the ``prefill_*`` functions are the data steps of the dialog's closures
  (``stage_prompt``, ``select_profile``, ``persist_profiles``, ``new_profile``,
  ``save_profile``, ``delete_profile``). The dialog owns a ``PrefillState`` (``prefill_load``
  from the three config values it reads; mobile: ``prefill_state_from_config``) and its closures
  call these functions; the combo box, editor, token count, message boxes, the Yes/No box
  (``prefill_delete(..., confirm=...)``), the save / rollback and the log lines stay in
  translator_gui. ``_quick_new_profile`` likewise calls ``new_profile`` with its combo box /
  editor updates as the ``update_widgets`` hook (tests/parity/test_u4_dialog_parity.py:
  legacy dialog vs working tree).

Config keys written: ``prompt_profiles``, ``active_profile``, ``text_extraction_method``,
``profile_name_autofill``, ``profile_mousewheel_locked`` (main profiles, through
``profiles_config_updates`` / ``write_profiles_to_config_file``) and
``assistant_prompt_profiles``, ``assistant_prompt_profile_default``,
``active_assistant_prompt_profile``, ``assistant_prompt`` (prefill).

Rules: GUI-free, Python 3.10 compatible, never imports PySide6, translator_gui or dpi_setup;
owner_state / settings_rules are imported lazily.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

__all__ = [
    # main prompt profiles
    "ProfileState", "SelectOutcome", "SaveOutcome", "DeleteOutcome",
    "FALLBACK_PROTECTED_PROFILES", "NEW_PROFILE_PATTERN",
    "profile_state_from_config", "protected_profiles", "select_profile", "save_profile",
    "delete_or_reset_profile", "new_profile", "profiles_config_updates", "apply_profiles_to_config",
    "write_profiles_to_config_file", "read_profiles_file", "merge_imported_profiles",
    "export_profiles_json", "json_export_path", "write_profiles_json", "extraction_method_for_profile",
    # assistant prefill profiles
    "PrefillState", "PrefillOutcome", "PREFILL_DEFAULT_NAME", "PREFILL_CONFIG_KEYS",
    "prefill_state_from_config", "prefill_load", "prefill_stage", "prefill_select", "prefill_new", "prefill_save",
    "prefill_delete", "prefill_config_updates", "prefill_apply_to_config",
]

#: other_settings.delete_profile's protected set when ``_get_protected_prompt_profiles`` fails.
FALLBACK_PROTECTED_PROFILES = frozenset({
    "Universal",
    "Korean_BeautifulSoup",
    "Japanese_BeautifulSoup",
    "Chinese_BeautifulSoup",
    "Korean_html2text",
    "Japanese_html2text",
    "Chinese_html2text",
    "Subtitle Translation",
})

#: Name pattern of new empty profiles (TranslatorGUI._quick_new_profile, the prefill dialog).
NEW_PROFILE_PATTERN = "New Profile #{n}"


# =============================================================================================
# 1. Main prompt profiles: state
# =============================================================================================

class ProfileState:
    """GUI-free stand-in for the TranslatorGUI attributes the profile actions use.

    ``config`` is the caller's dict (mutated in place, like ``self.config``). The two
    ConfigStateMixin helpers are the desktop methods themselves (looked up lazily).
    """

    def __init__(self, config, prompt_profiles, profile_var, default_prompts=None,
                 text_extraction_method_var='standard'):
        self.config = config
        self.prompt_profiles = prompt_profiles
        self.profile_var = profile_var
        self.default_prompts = dict(default_prompts or {})
        self.text_extraction_method_var = text_extraction_method_var
        self._original_profile_content = {}
        self._active_profile_for_autosave = profile_var

    @classmethod
    def from_config(cls, config):
        return profile_state_from_config(config)

    # ConfigStateMixin methods, run on this state (no widgets: no editor refresh)
    def _get_protected_prompt_profiles(self):
        from owner_state import ConfigStateMixin
        return ConfigStateMixin._get_protected_prompt_profiles(self)

    def _reset_prompt_profile_to_default(self, name):
        from owner_state import ConfigStateMixin
        return ConfigStateMixin._reset_prompt_profile_to_default(self, name)

    @property
    def protected(self):
        return frozenset(self._get_protected_prompt_profiles())

    @property
    def active_profile(self):
        return self.profile_var

    def __repr__(self):
        return (f"ProfileState(active={self.profile_var!r}, profiles={list(self.prompt_profiles)!r}, "
                f"text_extraction_method={self.text_extraction_method_var!r})")


_HOLDER_CLASS = None


def _holder_class():
    """A scratch ConfigStateMixin owner: config + the built-in prompt defaults, nothing else.

    ``default_*`` prompt attributes the desktop loads from the glossary modules (heavy
    imports) are not needed for the profile state and read as ''.
    """
    global _HOLDER_CLASS
    if _HOLDER_CLASS is None:
        from owner_state import ConfigStateMixin

        class _ProfileHolder(ConfigStateMixin):
            def __init__(self, config):
                self.config = config

            def __getattr__(self, name):
                if name.startswith('default_'):
                    return ''
                raise AttributeError(name)

        _HOLDER_CLASS = _ProfileHolder
    return _HOLDER_CLASS


def _built_in_defaults(holder):
    from refinement_prompts import DEFAULT_REFINEMENT_SYSTEM_PROMPT

    # ConfigStateMixin._init_default_prompts sets this before the profile dict is built
    holder.default_refinement_system_prompt = DEFAULT_REFINEMENT_SYSTEM_PROMPT
    holder._init_default_prompt_profiles()
    return holder.default_prompts


def profile_state_from_config(config):
    """The profile state desktop start-up builds from *config* (never mutates *config*).

    Runs the desktop code on a scratch owner: ConfigStateMixin._init_default_prompt_profiles
    (built-in profiles), _init_variables (profile reconciliation: missing built-ins added in
    their priority order, active profile validated) with a private environment, and
    initialize_extraction_variables (text extraction method).
    """
    from owner_state import initialize_extraction_variables
    from settings_rules import _run_isolated

    holder = _holder_class()(copy.deepcopy(dict(config)))
    defaults = _built_in_defaults(holder)
    _run_isolated(type(holder)._init_variables, holder)
    initialize_extraction_variables(holder)
    state = ProfileState(
        config,
        holder.prompt_profiles,
        holder.profile_var,
        default_prompts=defaults,
        text_extraction_method_var=holder.text_extraction_method_var,
    )
    return state


def protected_profiles(state):
    """Built-in profile names: reset instead of delete (other_settings.delete_profile)."""
    try:
        return set(state._get_protected_prompt_profiles())
    except Exception:
        return set(FALLBACK_PROTECTED_PROFILES)


def extraction_method_for_profile(name):
    """'standard' for *_BeautifulSoup profiles, 'enhanced' for *_html2text profiles, else None."""
    profile_lower = name.lower()
    if 'beautifulsoup' in profile_lower:
        return 'standard'
    if 'html2text' in profile_lower:
        return 'enhanced'
    return None


# =============================================================================================
# 2. Main prompt profiles: actions
# =============================================================================================

@dataclass(frozen=True)
class SelectOutcome:
    name: str
    prompt: str                         # the profile's last saved text (loaded into the editor)
    replace_text: Optional[bool]        # the editor differed and was replaced (None: no editor)
    extraction_method: Optional[str]    # 'standard' / 'enhanced' when the profile switched it


@dataclass(frozen=True)
class SaveOutcome:
    ok: bool
    error: str = ''                     # message shown to the user ('' when none)
    title: str = ''                     # desktop message-box title
    level: str = ''                     # 'critical' / 'warning'
    name: str = ''
    renamed_from: Optional[str] = None
    persisted: Optional[bool] = None    # persist() result (None: no persist hook)


@dataclass(frozen=True)
class DeleteOutcome:
    kind: str                           # missing / cancelled / reset_failed / reset / deleted
    error: str = ''
    title: str = ''
    level: str = ''
    name: str = ''
    new_active: Optional[str] = None


def select_profile(state, name, config=None, *, editor_text=None, replace_editor_text=None,
                   extraction_method_changed=None):
    """Load a profile's last saved prompt (other_settings.on_profile_select).

    Returns ``SelectOutcome`` or None when the name is blank or not a profile (nothing
    changes). Hooks (desktop): ``editor_text()`` reads the editor, ``replace_editor_text(prompt)``
    replaces its text when it differs, ``extraction_method_changed(method)`` syncs the
    Other Settings radios after the var and config were set. *config* is accepted for
    adapters; the state's own config is the one written.
    """
    # Skip if the name is empty or whitespace only
    if not name or not name.strip():
        return None

    # Only update if the profile actually exists in prompt_profiles
    # This prevents switching to non-existent profiles while typing
    if name in state.prompt_profiles:
        # When switching profiles, revert any unsaved changes by loading from original content
        if not hasattr(state, '_original_profile_content'):
            state._original_profile_content = {}

        # If this profile hasn't been saved yet, store its original content
        if name not in state._original_profile_content:
            state._original_profile_content[name] = state.prompt_profiles.get(name, "")

        # Load the original (last saved) content, not the in-memory staged edits
        prompt = state._original_profile_content.get(name, "")
        replace_text = None
        if editor_text is not None:
            current_text = editor_text().strip()
            replace_text = current_text != prompt.strip()
            if replace_text and replace_editor_text is not None:
                replace_editor_text(prompt)

        # Also revert the in-memory profile to original content
        state.prompt_profiles[name] = prompt

        # Update profile_var to match only when profile exists
        state.profile_var = name
        state.config['active_profile'] = name

        # Set this as the active profile for autosave
        state._active_profile_for_autosave = name

        # AUTO-SWITCH EXTRACTION METHOD BASED ON PROFILE NAME
        # Update both the variable AND the config so it persists
        method = extraction_method_for_profile(name)
        if method is not None:
            state.text_extraction_method_var = method
            state.config['text_extraction_method'] = method
            if extraction_method_changed is not None:
                extraction_method_changed(method)
        return SelectOutcome(name, prompt, replace_text, method)
    return None


def _text(value):
    return value() if callable(value) else value


def save_profile(state, name, content, source_name=None, config=None, *, persist=None,
                 rebuild_menu=None, restore_menu=None):
    """Save the editor text, renaming the selected custom profile if needed (other_settings.save_profile).

    *name* is the (stripped) combo box text; *content* the editor text, or a zero-argument
    callable read at the original point (after the name checks). *source_name* is the
    profile being edited (desktop: the autosave profile, else profile_var). Built-in
    profiles keep their names: saving one under a new name creates a custom copy.

    Hooks (desktop): ``rebuild_menu(names, current)`` after the state changed,
    ``persist()`` (other_settings.save_profiles; False rolls everything back) and
    ``restore_menu(names, source_name, edit_text)`` after a rollback.
    """
    name = name.strip()
    if not name:
        return SaveOutcome(False, "Profile cannot be empty.", "Error", "critical")

    if source_name is None:
        source_name = getattr(state, '_active_profile_for_autosave', None) or state.profile_var
    if name in state.prompt_profiles and name != source_name:
        return SaveOutcome(False, "A profile with this name already exists. Choose another name.",
                           "Profile Name Exists", "warning", name=name)
    get_protected = getattr(state, '_get_protected_prompt_profiles', None)
    protected = set(get_protected() if callable(get_protected) else getattr(state, 'default_prompts', {}))
    renaming = source_name in state.prompt_profiles and name != source_name and source_name not in protected
    previous_profiles = dict(state.prompt_profiles)
    previous_originals = dict(getattr(state, '_original_profile_content', {}))
    previous_profile_var = state.profile_var
    previous_autosave = getattr(state, '_active_profile_for_autosave', None)
    previous_active = state.config.get('active_profile', previous_profile_var)

    content = _text(content).strip()

    if renaming:
        state.prompt_profiles = {
            (name if key == source_name else key): (content if key == source_name else value)
            for key, value in state.prompt_profiles.items()
        }
    else:
        # Built-in profiles retain their required names; saving one under a new
        # name creates a custom copy, as it does for the glossary Default.
        if name != source_name and source_name in previous_originals:
            state.prompt_profiles[source_name] = previous_originals[source_name]
        state.prompt_profiles[name] = content
    state.config['prompt_profiles'] = state.prompt_profiles
    state.config['active_profile'] = name
    state.profile_var = name
    state._active_profile_for_autosave = name

    # Update the original content to match the saved content
    if not hasattr(state, '_original_profile_content'):
        state._original_profile_content = {}
    if renaming:
        state._original_profile_content.pop(source_name, None)
    state._original_profile_content[name] = content

    # Rebuild without selection callbacks loading old saved content mid-rename.
    if rebuild_menu is not None:
        rebuild_menu(list(state.prompt_profiles), name)

    persisted = None
    if persist is not None:
        persisted = persist()
        if persisted is False:
            state.prompt_profiles = previous_profiles
            state.config['prompt_profiles'] = previous_profiles
            state.config['active_profile'] = previous_active
            state._original_profile_content = previous_originals
            state.profile_var = previous_profile_var
            state._active_profile_for_autosave = previous_autosave
            if restore_menu is not None:
                restore_menu(list(previous_profiles), source_name, name)
            return SaveOutcome(False, name=name, renamed_from=source_name if renaming else None,
                               persisted=False)
    return SaveOutcome(True, name=name, renamed_from=source_name if renaming else None, persisted=persisted)


def delete_or_reset_profile(state, name, config=None, *, confirm=None, after_reset=None,
                            after_delete=None, persist=None):
    """Delete a custom profile, or reset a built-in one to its default (other_settings.delete_profile).

    Hooks (desktop): ``confirm(kind, name)`` ('reset' / 'delete') shows the Yes/No box;
    ``after_reset(name)`` refreshes the combo + editor (default: ``select_profile``);
    ``after_delete(new_name or None)`` rebuilds the combo and selects the first profile
    (default: ``select_profile``); ``persist()`` is other_settings.save_profiles.
    """
    if name not in state.prompt_profiles:
        return DeleteOutcome('missing', f"Profile '{name}' not found.", "Error", "critical", name=name)

    # Built-in/required profiles: treat delete as reset-to-default.
    protected = protected_profiles(state)

    if name in protected:
        if confirm is not None and not confirm('reset', name):
            return DeleteOutcome('cancelled', name=name)
        ok = False
        try:
            ok = bool(state._reset_prompt_profile_to_default(name))
        except Exception:
            ok = False

        if not ok:
            return DeleteOutcome('reset_failed', f"Could not reset '{name}' (default prompt not found).",
                                 "Reset Failed", "warning", name=name)

        # Refresh editor selection
        if after_reset is not None:
            after_reset(name)
        else:
            # Without widgets the reset profile is the one being edited (the desktop deletes
            # the combo's current profile, so _reset_prompt_profile_to_default also refreshed
            # the editor's last-saved copy before on_profile_select reloads it).
            if not hasattr(state, '_original_profile_content'):
                state._original_profile_content = {}
            state._original_profile_content[name] = state.prompt_profiles[name]
            select_profile(state, name)

        if persist is not None:
            persist()
        return DeleteOutcome('reset', name=name, new_active=state.profile_var)

    # Custom profiles: delete as before
    if confirm is not None and not confirm('delete', name):
        return DeleteOutcome('cancelled', name=name)

    del state.prompt_profiles[name]
    state.config['prompt_profiles'] = state.prompt_profiles

    if state.prompt_profiles:
        new = next(iter(state.prompt_profiles))
        state.profile_var = new

        if after_delete is not None:
            after_delete(new)
        else:
            select_profile(state, new)
    else:
        state.profile_var = ""
        if after_delete is not None:
            after_delete(None)

    if persist is not None:
        persist()
    return DeleteOutcome('deleted', name=name, new_active=state.profile_var)


def new_profile(state, config=None, *, existing_names=(), update_widgets=None, persist=None):
    """Create an empty "New Profile #n" and make it active (TranslatorGUI._quick_new_profile).

    *existing_names* are extra names to avoid (desktop: the profile combo box items).
    *update_widgets(name)* runs after the config writes and before the active profile is
    switched (desktop: add the combo item, select it, clear the editor; their Qt signals
    read the state at that point). *persist()* runs last (desktop: ``save_profiles``).
    Returns the new name.
    """
    existing = set(state.prompt_profiles.keys())
    for item in existing_names:
        existing.add(item)

    n = 1
    while NEW_PROFILE_PATTERN.format(n=n) in existing:
        n += 1
    name = NEW_PROFILE_PATTERN.format(n=n)

    state.prompt_profiles[name] = ""
    state.config['prompt_profiles'] = state.prompt_profiles
    state.config['active_profile'] = name

    if update_widgets is not None:
        update_widgets(name)

    state.profile_var = name
    if not hasattr(state, '_original_profile_content'):
        state._original_profile_content = {}
    state._original_profile_content[name] = ""
    state._active_profile_for_autosave = name

    if persist is not None:
        persist()
    return name


def _active_profile_name(state):
    return state.profile_menu.currentText() if hasattr(state, 'profile_menu') else state.profile_var


def profiles_config_updates(state, config=None, active_profile=None):
    """The keys other_settings.save_profiles writes, in its order.

    *config* defaults to the state's config (autofill / mouse-wheel preferences);
    *active_profile* (a name or a zero-argument callable) defaults to the profile combo
    text, else profile_var.
    """
    config = state.config if config is None else config
    return _profile_file_updates(state.prompt_profiles, config or {},
                                 active_profile if active_profile is not None else (lambda: _active_profile_name(state)))


def _profile_file_updates(prompt_profiles, config, active_profile):
    # Each argument may be a zero-argument callable: the desktop wrapper reads its
    # attributes / widgets at the original points (after the config file was read).
    data = {}
    data['prompt_profiles'] = _text(prompt_profiles)
    data['profile_name_autofill'] = bool(_text(config).get('profile_name_autofill', True))
    data['profile_mousewheel_locked'] = bool(_text(config).get('profile_mousewheel_locked', True))

    # Get current profile from combobox or profile_var
    data['active_profile'] = _text(active_profile)
    return data


def apply_profiles_to_config(state, config=None):
    """Put the save_profiles key set into *config* (default: the state's config); returns it."""
    target = state.config if config is None else config
    updates = profiles_config_updates(state, target)
    target.update(updates)
    return updates


def write_profiles_to_config_file(config_file, prompt_profiles, active_profile, config=None, *, opener=open):
    """other_settings.save_profiles' file write: read-modify-write of the raw config.json.

    Only the four profile keys change; every other key (including ``ENC:`` secrets) is
    written back as read. *prompt_profiles*, *active_profile* and *config* may be
    zero-argument callables, evaluated after the file was read (desktop: its attributes and
    the profile combo text). *opener* opens the file (desktop: other_settings' ``open``).
    Raises on errors; returns True.
    """
    if config is None:
        config = {}
    data = {}
    if os.path.exists(config_file):
        with opener(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    data.update(_profile_file_updates(prompt_profiles, config, active_profile))

    with opener(config_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return True


def read_profiles_file(path, *, opener=open):
    """Load an exported profiles file (other_settings.import_profiles)."""
    with opener(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def merge_imported_profiles(state, data, config=None):
    """Merge imported profiles into the state (same names are replaced); returns len(data)."""
    state.prompt_profiles.update(data)
    state.config['prompt_profiles'] = state.prompt_profiles
    return len(data)


def export_profiles_json(prompt_profiles):
    """The text other_settings.export_profiles writes (JSON, UTF-8, indent 2)."""
    return json.dumps(prompt_profiles, ensure_ascii=False, indent=2)


def json_export_path(path):
    """Add .json extension if not present (other_settings.export_profiles)."""
    if not path.endswith('.json'):
        path += '.json'
    return path


def write_profiles_json(path, prompt_profiles, *, opener=open):
    """other_settings.export_profiles' file write (*prompt_profiles* may be a callable, read
    once the file is open)."""
    with opener(path, 'w', encoding='utf-8') as f:
        json.dump(_text(prompt_profiles), f, ensure_ascii=False, indent=2)
    return path


# =============================================================================================
# 3. Assistant prefill profiles (TranslatorGUI.show_assistant_prompt_dialog closures)
# =============================================================================================

PREFILL_DEFAULT_NAME = "Default"
#: Keys the dialog's persist_profiles writes.
PREFILL_CONFIG_KEYS = (
    'assistant_prompt_profiles',
    'assistant_prompt_profile_default',
    'active_assistant_prompt_profile',
    'assistant_prompt',
)


@dataclass
class PrefillState:
    """The dialog's drafts: named profiles, the Default text, the active name ('' = Default)
    and the editor text."""

    profiles: Dict[str, str] = field(default_factory=dict)
    default_prompt: str = ''
    active_name: str = ''
    text: str = ''

    @property
    def options(self):
        return [PREFILL_DEFAULT_NAME] + list(self.profiles)


@dataclass(frozen=True)
class PrefillOutcome:
    ok: bool
    error: str = ''
    title: str = ''
    name: str = ''


def prefill_state_from_config(config, current_prompt=None):
    """The dialog's drafts when it opens (the active profile, or Default, selected).

    *current_prompt* is the live ``assistant_prompt`` (default: the config value, the
    _init_variables start-up value). Default is seeded from it when upgrading an older config.
    """
    if current_prompt is None:
        from prompt_defaults import DEFAULT_ASSISTANT_PROMPT
        current_prompt = config.get('assistant_prompt', DEFAULT_ASSISTANT_PROMPT)
    current_prompt = str(current_prompt or '')
    state = PrefillState()
    prefill_load(state, config.get('assistant_prompt_profiles', {}),
                 config.get('assistant_prompt_profile_default', current_prompt),
                 config.get('active_assistant_prompt_profile', ''))
    return state


def prefill_load(state, stored_profiles, default_prompt, active_name):
    """Fill *state* with the drafts of the three stored values (the desktop dialog reads them
    from ``self.config``; Default is seeded from the live prefill by the caller's default):
    unusable profile names / texts dropped, an unknown active name means Default, and the
    active profile (or Default) selected. Returns *state*."""
    # Keep drafts local so Cancel discards edits since the last explicit save.
    # Seed Default from the existing prefill when upgrading an older config.
    profiles = {
        name: text for name, text in stored_profiles.items()
        if isinstance(name, str) and name.strip()
        and name.strip().casefold() != 'default' and isinstance(text, str)
    } if isinstance(stored_profiles, dict) else {}
    default_prompt = str(default_prompt or '')
    if not isinstance(active_name, str) or active_name not in profiles:
        active_name = ''
    state.profiles = profiles
    state.default_prompt = default_prompt
    state.active_name = active_name
    # the dialog opens on the active profile (or Default) and shows its text
    prefill_select(state, _selected_name(state))
    return state


def prefill_stage(state, name, text):
    """An editor edit (stage_prompt): the draft text goes to the active profile, or to Default."""
    name = name.strip()
    text = text.strip()
    state.text = text
    if state.active_name in state.profiles:
        # Editing the name does not change which profile owns this draft.
        state.profiles[state.active_name] = text
    elif not name or name.casefold() == 'default':
        state.default_prompt = text


def prefill_select(state, name):
    """Switch the dialog to *name* ("Default" or a profile); returns the loaded text, or None."""
    name = name.strip()
    if name.casefold() == 'default':
        state.active_name = ''
        text = state.default_prompt
    elif name in state.profiles:
        state.active_name = name
        text = state.profiles[name]
    else:
        return None
    state.text = text
    return text


def _selected_name(state):
    return state.active_name or PREFILL_DEFAULT_NAME


def prefill_new(state):
    """+ New Profile: an empty "New Profile #n", selected. Returns its name."""
    n = 1
    while NEW_PROFILE_PATTERN.format(n=n) in state.profiles:
        n += 1
    state.active_name = NEW_PROFILE_PATTERN.format(n=n)
    state.profiles[state.active_name] = ''
    prefill_select(state, _selected_name(state))
    return state.active_name


def prefill_save(state, name, content):
    """Save Profile: *name* (the edited combo text) gets *content*; renaming keeps the order."""
    name = name.strip()
    if not name:
        return PrefillOutcome(False, "Enter a profile name before saving.", "Profile Name Required")
    if name.casefold() == 'default' and state.active_name:
        return PrefillOutcome(False, "Default is reserved. Choose another profile name.", "Default Profile")
    if name in state.profiles and name != state.active_name:
        return PrefillOutcome(False, "A profile with this name already exists. Choose another name.",
                              "Profile Name Exists")
    text = content.strip()
    state.text = text
    if name.casefold() == 'default':
        state.default_prompt = text
        state.active_name = ''
    else:
        if state.active_name in state.profiles and name != state.active_name:
            # Replace the key in place, preserving the dropdown order.
            state.profiles = {
                (name if key == state.active_name else key): (text if key == state.active_name else value)
                for key, value in state.profiles.items()
            }
        else:
            state.profiles[name] = text
        state.active_name = name
    return PrefillOutcome(True, name=state.active_name or PREFILL_DEFAULT_NAME)


def prefill_delete(state, name, *, confirm=None):
    """Delete Profile: the first remaining profile, else Default, is selected.

    *confirm(name)* is the dialog's Yes/No box, asked after the checks; a False answer
    returns ``PrefillOutcome(False, name=name)`` (no error text) and changes nothing.
    """
    name = name.strip()
    if name.casefold() == 'default':
        return PrefillOutcome(False, "The Default assistant prompt profile cannot be deleted.", "Default Profile")
    if not name or name not in state.profiles:
        return PrefillOutcome(False, "Select an existing profile to delete.", "Profile Not Found")
    if confirm is not None and not confirm(name):
        return PrefillOutcome(False, name=name)
    del state.profiles[name]
    state.active_name = next(iter(state.profiles), '')
    prefill_select(state, _selected_name(state))
    return PrefillOutcome(True, name=name)


def prefill_config_updates(state, content=None):
    """persist_profiles' config updates; *content* is the editor text (default: the state's)."""
    text = state.text if content is None else content
    return {
        'assistant_prompt_profiles': dict(state.profiles),
        'assistant_prompt_profile_default': state.default_prompt,
        'active_assistant_prompt_profile': state.active_name,
        'assistant_prompt': str(text or '').strip(),
    }


def prefill_apply_to_config(state, config, content=None):
    """Write persist_profiles' updates into *config*; returns them."""
    updates = prefill_config_updates(state, content)
    config.update(updates)
    return updates
