"""Global setting writes with the desktop control's side effects (U9).

The desktop selectors do more than store their own key:

* Output mode (``other_settings._set_output_mode``) also writes the legacy flags
  ``enable_image_translation`` / ``enable_image_output_mode`` / ``enable_video_output_mode`` /
  ``enable_audio_output_mode`` / ``enable_refinement_output_mode`` that book jobs read;
* the glossary-mode shortcut writes ``enable_auto_glossary`` / ``append_glossary`` /
  ``append_glossary_auto_load`` / ``fuzzy_auto_mapping``;
* Stream thinking logs locks Enable thoughts;
* Target Language (``ConfigStateMixin.update_target_language``) fans out to
  ``glossary_target_language``, the manga manual-edit language and the AI Hunter target;
* the main-window Profile combo (``other_settings.on_profile_select`` ->
  ``prompt_profiles.select_profile``) also switches ``text_extraction_method`` for the
  ``*_BeautifulSoup`` / ``*_html2text`` profiles;
* Output Token Limit / Auto Compression Factor (main window, Other Settings) recompute
  ``compression_factor`` (or hold the manual chunk size), and the Glossary Manager's Auto box /
  glossary output limit recompute ``glossary_compression_factor`` (U9).

Every global writer on mobile (Settings tiles, ModelSheet / Plan chips, Chat settings › All chats,
the Welcome flow) goes through ``write_setting`` / ``write_settings``, which run the shared rules
(``settings_rules.apply_change``, the Profiles service over ``prompt_profiles``) on a copy of the
config and store every key that changed (sparse ``set_many``). Nothing here is a copy of desktop
logic; without the backend (a broken build) the plain key is written.

``store`` is anything with ``snapshot()`` and ``set_many()`` (``MobileConfigStore``, the Plan
card's ``RunOverlayStore``); a store without ``snapshot`` gets plain writes.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Mapping, Optional

__all__ = [
    "CONTEXT_MODE_CHOICES",
    "CONTEXT_MODE_KEY",
    "CONTEXT_MODE_WRITES",
    "PROFILE_KEY",
    "RULE_KEYS",
    "context_mode_of",
    "extraction_method_for_profile",
    "implied_changes",
    "output_mode_values",
    "write_setting",
    "write_settings",
]

log = logging.getLogger("glossarion.settings")

#: Keys whose desktop control writes more than the key itself (``settings_rules._CHANGE_RULES``).
RULE_KEYS = frozenset({"output_mode", "auto_glossary_mode", "stream_thinking_logs", "output_language",
                       # U9: the output token limits and Auto boxes recompute the compression factors
                       "max_output_tokens", "auto_compression_factor", "glossary_auto_compression",
                       "glossary_max_output_tokens"})
#: The main prompt profile (``prompt_profiles.select_profile``).
PROFILE_KEY = "active_profile"
#: U9: the desktop main-window Context Mode combo - not a config key: it writes ``contextual`` /
#: ``use_rolling_summary`` / ``rolling_summary_mode`` as one choice and the batching constraint
#: (``settings_rules.apply_context_mode``; owner_state._on_context_mode_changed).
CONTEXT_MODE_KEY = "context_mode"
#: The combo's items (translator_gui ``context_mode_combo.addItem``): (value, label).
CONTEXT_MODE_CHOICES = (("off", "Off"), ("contextual_history", "Contextual History"),
                        ("rolling_summary_replace", "Rolling Summary (Replace)"),
                        ("rolling_summary_append", "Rolling Summary (Append)"))
#: The keys the Context Mode combo writes (``apply_context_mode`` also enforces the batching mode).
CONTEXT_MODE_WRITES = ("contextual", "use_rolling_summary", "rolling_summary_mode", "batching_mode")


def _rules() -> Any:
    try:
        return importlib.import_module("settings_rules")
    except Exception:
        log.debug("settings_rules unavailable", exc_info=True)
        return None


def implied_changes(config: dict, key: str, value: Any) -> dict:
    """``settings_rules.apply_change(config, key, value)`` (``config`` is changed in place):
    ``{key: value}`` for every key that changed; ``{key: value}`` alone without the rules."""
    rules = _rules()
    if rules is None or not hasattr(rules, "apply_change"):
        config[key] = value
        return {key: value}
    try:
        changed, _env = rules.apply_change(config, key, value)
    except Exception:
        log.exception("settings rule for %s failed; writing the key alone", key)
        config[key] = value
        return {key: value}
    return dict(changed)


def output_mode_values(mode: Any) -> dict:
    """The config values the desktop Output Mode selector writes for ``mode``
    (``settings_rules.output_mode_flags(mode).config_values``); ``{}`` without the rules."""
    rules = _rules()
    if rules is None or not mode:
        return {}
    try:
        return dict(rules.output_mode_flags(str(mode)).config_values)
    except Exception:
        log.debug("output_mode_flags failed", exc_info=True)
        return {}


def context_mode_of(config: Any) -> str:
    """The Context Mode the stored flags represent (``settings_rules.context_mode``); 'off' without the rules."""
    rules = _rules()
    if rules is None or not hasattr(rules, "context_mode"):
        return "off"
    try:
        return str(rules.context_mode(dict(config or {})) or "off")
    except Exception:
        log.debug("context_mode failed", exc_info=True)
        return "off"


def _write_context_mode(store: Any, mode: Any) -> list:
    """The Context Mode combo: ``apply_context_mode`` on a copy of the config, then every key it changed."""
    snapshot = getattr(store, "snapshot", None)
    rules = _rules()
    if rules is None or not hasattr(rules, "apply_context_mode") or not callable(snapshot):
        return []
    try:
        config = dict(snapshot() or {})
    except Exception:
        return []
    before = {key: config.get(key) for key in CONTEXT_MODE_WRITES}
    try:
        rules.apply_context_mode(config, str(mode or "off"))
    except Exception:
        log.exception("apply_context_mode failed")
        return []
    changes = {key: config[key] for key in CONTEXT_MODE_WRITES if key in config and config.get(key) != before[key]}
    for key in ("contextual", "use_rolling_summary", "rolling_summary_mode"):  # the combo always writes these
        if key in config and not (store.has(key) if hasattr(store, "has") else False):
            changes.setdefault(key, config[key])
    return _set_many(store, changes) if changes else []


def extraction_method_for_profile(name: Any) -> Optional[str]:
    """``prompt_profiles.extraction_method_for_profile``: 'standard' / 'enhanced' / None."""
    if not isinstance(name, str) or not name.strip():
        return None
    try:
        module = importlib.import_module("prompt_profiles")
        return module.extraction_method_for_profile(name)
    except Exception:
        log.debug("prompt_profiles unavailable", exc_info=True)
        return None


def _select_profile(store: Any, name: Any) -> list:
    """Settings › Profiles' "Use this profile" (``ProfileService.select`` -> ``select_profile``)."""
    try:
        from glossarion_mobile.ui.screens.profiles import ProfileService

        changed = list(ProfileService(store).select(str(name)))
    except Exception:
        log.debug("profile select through prompt_profiles failed", exc_info=True)
        changed = []
    try:
        current = store.get(PROFILE_KEY)
    except Exception:
        current = None
    if current != name:  # not a known profile (or no backend): the plain key, as before
        changed += _set_many(store, {PROFILE_KEY: name})
    return changed


def _set_many(store: Any, values: Mapping[Any, Any]) -> list:
    set_many = getattr(store, "set_many", None)
    if callable(set_many):
        return list(set_many(dict(values)) or [])
    return [k for k, v in values.items() if store.set(k, v)]


def write_setting(store: Any, key: Any, value: Any) -> list:
    """Store one global setting the way its desktop control does; returns the changed keys."""
    if store is None:
        return []
    if isinstance(key, tuple):
        if len(key) != 1:
            return _set_many(store, {key: value})
        key = key[0]
    if key == PROFILE_KEY and value:
        return _select_profile(store, value)
    if key == CONTEXT_MODE_KEY:
        return _write_context_mode(store, value)
    snapshot = getattr(store, "snapshot", None)
    if key not in RULE_KEYS or not callable(snapshot):
        return _set_many(store, {key: value})
    try:
        config = snapshot()
    except Exception:
        config = None
    if not isinstance(config, dict):
        return _set_many(store, {key: value})
    changes = implied_changes(config, key, value)
    if key not in changes:
        try:
            unchanged = store.has(key) and store.get(key) == value
        except Exception:
            unchanged = False
        if not unchanged:
            changes[key] = value
    return _set_many(store, changes) if changes else []


def write_settings(store: Any, values: Mapping[Any, Any]) -> list:
    """``write_setting`` for several keys, in order (the rules of each key apply)."""
    changed: list = []
    for key, value in dict(values or {}).items():
        changed.extend(write_setting(store, key, value))
    return changed
