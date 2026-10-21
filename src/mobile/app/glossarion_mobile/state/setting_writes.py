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

Glossarion Mobile's one Streaming switch (``STREAMING_KEY``, owner 2026-10-08) stands for the desktop
"Real-time Translation (Streaming)" group: ``settings_rules.apply_streaming`` sets the four toggles the way
their checkboxes do (stream thinking keeps the desktop Enable thoughts lock) and only the keys whose
value changes are stored. An absent toggle counts as ON on mobile (``MOBILE_STREAMING_DEFAULT``; the
desktop default stays off): every job's config snapshot gets it (``with_mobile_streaming_defaults``), and
chat / Reader runs force streaming only while it is on (``streaming_enabled``).

Choosing an on-device ``local/`` model (mobile only) also selects the built-in On-device translation
profile of the text extraction method (``ON_DEVICE_TRANSLATION_PROFILES``: "On-device_BeautifulSoup" for
'standard', "On-device_html2text" for 'enhanced'; switching the method while one is active switches to the
other) and the "On-device" glossary and review generator prompt profiles (the review profile holds very short review and final
prompts: ``review_generator.ON_DEVICE_REVIEW_PROMPT`` / ``ON_DEVICE_FINAL_REVIEW_PROMPT``), drops the
glossary's extra columns (the "description" field; owner 2026-10-19: a small model only writes name,
translated name and gender) and sets its Request size (``ON_DEVICE_SETTINGS``: temperature 0.4, batch
size 1, output limit half the context, input token limit the whole context); choosing a model
that is not ``local/`` again restores everything that was there before, where it still holds the
On-device value (a value or profile the user changed meanwhile stays) and the defaults where nothing was
saved (``ON_DEVICE_PROFILE``, ``ON_DEVICE_BEFORE_KEY``).

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
    "MOBILE_MAX_OUTPUT_TOKENS",
    "MOBILE_DEFAULT_MODEL",
    "default_batch_size",
    "MOBILE_STREAMING_DEFAULT",
    "EXTRACTION_KEY",
    "ON_DEVICE_BEFORE_KEY",
    "ON_DEVICE_PROFILE",
    "ON_DEVICE_TRANSLATION_PROFILES",
    "on_device_translation_profile",
    "PROFILE_KEY",
    "RULE_KEYS",
    "STREAMING_KEY",
    "STREAMING_KEYS",
    "STREAMING_WRITES",
    "THOUGHTS_DEFAULT",
    "context_mode_of",
    "effective_streaming_values",
    "extraction_method_for_profile",
    "implied_changes",
    "output_mode_values",
    "reset_streaming",
    "streaming_enabled",
    "streaming_mode",
    "streaming_states",
    "streaming_summary",
    "local_output_limit",
    "local_context_env",
    "with_mobile_streaming_defaults",
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
#: Glossarion Mobile's one Streaming switch (owner 2026-10-08) - not a config key: the desktop
#: "Real-time Translation (Streaming)" group (Enable streaming responses, Stream thinking/reasoning logs,
#: Allow streaming logs during batch mode, Allow forced-stream batch log) plus the Enable thoughts toggle
#: the stream-thinking lock drives (``settings_rules.apply_streaming``).
STREAMING_KEY = "streaming"
#: ``settings_rules.STREAMING_KEYS`` (a literal: this module loads before the backend is on sys.path).
STREAMING_KEYS = ("enable_streaming", "stream_thinking_logs", "allow_batch_stream_logs",
                  "allow_authgpt_batch_stream_logs")
#: Everything the switch writes: the four toggles and the thoughts lock (other_settings._sync_thoughts_lock_state).
STREAMING_WRITES = STREAMING_KEYS + ("enable_thoughts",)
#: What an absent streaming toggle means on Glossarion Mobile (owner 2026-10-08: ON; the desktop default
#: stays off). A saved or imported value always wins, and the default itself is never written.
MOBILE_STREAMING_DEFAULT = True
#: the mobile output-token budget when none is saved (desktop default 128000; owner, U13)
MOBILE_MAX_OUTPUT_TOKENS = 16384
#: The model when none is saved on Glossarion Mobile (owner 2026-10-12): ChatGPT's web chat, which needs
#: no API key and no sign-in. The desktop default stays authgpt/gpt-6-luna; the default is never written.
MOBILE_DEFAULT_MODEL = "authchatgpt/auto"
#: The built-in prompt profile for small on-device models (the Balanced/Full and Minimal glossary profiles and
#: the review profile share the name; the translation profiles are ``ON_DEVICE_TRANSLATION_PROFILES``):
#: choosing a ``local/`` model selects them, and choosing a model that is not ``local/`` again brings back what
#: was active before (``ON_DEVICE_BEFORE_KEY``) wherever the On-device profile is still the active one.
ON_DEVICE_PROFILE = "On-device"
#: The built-in translation profiles for on-device models by ``text_extraction_method`` (owner 2026-10-19;
#: ``prompt_profiles.ON_DEVICE_TRANSLATION_PROFILES`` - a literal: this module loads before the backend).
ON_DEVICE_TRANSLATION_PROFILES = {"standard": "On-device_BeautifulSoup", "enhanced": "On-device_html2text"}
#: The text extraction method (Welcome, Settings › Text Extraction Method; desktop default 'standard').
EXTRACTION_KEY = "text_extraction_method"
#: the glossary's extra columns (desktop Glossary Manager › Custom Fields) and its "description was
#: removed on purpose" flag (owner_state adds "description" back to an empty list without it)
CUSTOM_FIELDS_KEY = "custom_glossary_fields"
#: the temperature an on-device model sets (owner 2026-10-21: 0.4; was Qwen's 0.7 until 2026-10-20). Only
#: on-device models get it; cloud models keep the saved / desktop default, and leaving restores it.
ON_DEVICE_TEMPERATURE = 0.4
#: Request size settings an on-device model sets and a cloud model gets back (owner 2026-10-19: "choosing
#: a non on device model should restore all the original defaults and vice versa"); the output limit is
#: half the model's context (``local_output_limit``). The review generator's short prompts are the built-in
#: "On-device" review prompt profile (``review_generator``), selected like the other On-device profiles.
#: The Manga translator reads with Hayai OCR on the phone (owner 2026-10-21; "Your Own key" OCR sends the
#: pages to the translation model, which a small on-device model reads poorly).
ON_DEVICE_OCR_PROVIDER = "hayai-ocr"
ON_DEVICE_SETTINGS = {"translation_temperature": ON_DEVICE_TEMPERATURE, "batch_size": 1,
                      "manga_ocr_provider": ON_DEVICE_OCR_PROVIDER}
OUTPUT_LIMIT_KEY = "max_output_tokens"
#: Auto glossary on an on-device model (owner 2026-10-19): NO GLOSSARY (the Welcome card's writes,
#: ``welcome_flow.welcome_glossary_updates("no_glossary")``), except the 9B model, which can extract one
ON_DEVICE_GLOSSARY_SETTINGS = {"auto_glossary_mode": "no_glossary", "enable_auto_glossary": False}
#: the input token limit (off unless enabled, ``token_limit_disabled``): the model's whole context
INPUT_LIMIT_KEY = "token_limit"
DESCRIPTION_REMOVED_KEY = "custom_field_description_removed"
ON_DEVICE_BEFORE_KEY = "profiles_before_on_device"
_GLOSSARY_PROFILE_BUCKETS = ("balanced_full", "minimal")
_GLOSSARY_PROFILE_KEYS = ("glossary_prompt_profiles", "active_glossary_prompt_profiles",
                          "glossary_prompt_profile_defaults")
#: what selecting a review prompt profile writes (``review_generator.set_active_review_prompt_profile``:
#: the profiles a migration creates, the active name and the two prompt mirrors)
_REVIEW_PROFILE_KEYS = ("review_prompt_profiles", "active_review_prompt_profile", "review_system_prompt",
                        "review_final_prompt")
#: ``settings_rules.DEFAULT_BATCH_SIZE`` / ``SERIAL_BATCH_ROUTES`` (literals: loads before the backend).
_BATCH_SIZE = 5
_SERIAL_BATCH_ROUTES = ("authds/", "authchatgpt/")
#: Enable thoughts when absent (owner_state / run_env: ``config.get('enable_thoughts', True)``).
THOUGHTS_DEFAULT = True
_STREAMING_DEFAULTS = {**{key: MOBILE_STREAMING_DEFAULT for key in STREAMING_KEYS}, "enable_thoughts": THOUGHTS_DEFAULT}


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


def _stored_value(source: Any, key: str) -> tuple:
    """``(present, value)`` of ``key`` in a config store (``has`` / ``get``) or a plain mapping."""
    if source is None:
        return False, None
    if isinstance(source, Mapping):
        return (True, source[key]) if key in source else (False, None)
    try:
        return (True, source.get(key)) if source.has(key) else (False, None)
    except Exception:
        return False, None


def effective_streaming_values(source: Any) -> dict:
    """``STREAMING_WRITES`` as a mobile run reads them: the stored value, else the mobile default (the
    toggles ON, Enable thoughts on). ``source`` is a config store or a config mapping."""
    values = {}
    for key in STREAMING_WRITES:
        present, value = _stored_value(source, key)
        values[key] = value if present else _STREAMING_DEFAULTS[key]
    return values


def _stored_toggles(source: Any) -> dict:
    """The four streaming toggles that are stored in ``source`` (absent ones left out)."""
    stored = {}
    for key in STREAMING_KEYS:
        present, value = _stored_value(source, key)
        if present:
            stored[key] = value
    return stored


def streaming_states(source: Any) -> dict:
    """``{toggle: bool}`` the way a mobile run reads the four toggles (``settings_rules.streaming_states``
    over the stored values, an absent toggle ON)."""
    stored = _stored_toggles(source)
    rules = _rules()
    if rules is not None and hasattr(rules, "streaming_states"):
        try:
            return dict(rules.streaming_states(stored, unset=MOBILE_STREAMING_DEFAULT))
        except Exception:
            log.debug("streaming_states failed", exc_info=True)
    return {key: bool(stored.get(key, MOBILE_STREAMING_DEFAULT)) for key in STREAMING_KEYS}


def streaming_mode(source: Any) -> str:
    """'on' / 'off' / 'custom' (the toggles differ: an imported desktop config, or the U8/U9 builds'
    four switches) - what the Streaming switch shows (``settings_rules.streaming_mode`` over the stored
    values, an absent toggle ON)."""
    rules = _rules()
    if rules is not None and hasattr(rules, "streaming_mode"):
        try:
            return str(rules.streaming_mode(_stored_toggles(source), unset=MOBILE_STREAMING_DEFAULT))
        except Exception:
            log.debug("streaming_mode failed", exc_info=True)
    states = streaming_states(source).values()  # no backend (a broken build)
    return "on" if all(states) else "off" if not any(states) else "custom"


def streaming_summary(source: Any) -> str:
    """The Streaming switch's value line: "On", "Off" or "Custom · N of 4 on"."""
    states = list(streaming_states(source).values())
    count = sum(1 for on in states if on)
    if count in (0, len(states)):
        return "On" if count else "Off"
    return f"Custom · {count} of {len(states)} on"


def streaming_enabled(source: Any) -> bool:
    """Requests stream on mobile: Enable streaming responses (stored, else ON). The chat and the Reader's
    live translation force every streaming switch on (desktop: always) only while this is on."""
    return bool(streaming_states(source)["enable_streaming"])


def default_batch_size(model: Any) -> int:
    """``settings_rules.default_batch_size``: 1 for the web-chat routes (authds/, authchatgpt/), else 5;
    1 for an on-device ``local/`` model too (mobile only: its engine answers one request at a time)."""
    if str(model or "").strip().lower().startswith("local/"):
        return 1
    rules = _rules()
    if rules is not None and hasattr(rules, "default_batch_size"):
        try:
            return int(rules.default_batch_size(model))
        except Exception:
            log.debug("default_batch_size failed", exc_info=True)
    return 1 if str(model or "").strip().lower().startswith(_SERIAL_BATCH_ROUTES) else _BATCH_SIZE


#: Job config of a ``local/`` (on-device) model, whatever the saved settings say.
LOCAL_ENGINE_JOB_CONFIG = {"batch_translation": False, "enable_parallel_extraction": False, "extraction_workers": 1}


def local_output_limit(model: Any) -> Optional[int]:
    """The output-token limit of a job on an on-device ``local/<stem>`` model: half its configured context
    (``LocalEngineService.output_limit_for``), or None without the app's engine service."""
    value = str(model or "").strip()
    if not value.lower().startswith("local/"):
        return None
    try:
        from glossarion_mobile.services.local_llm.service import default_service

        service = default_service()
        return int(service.output_limit_for(value[len("local/"):])) if service is not None else None
    except Exception:
        log.debug("local_output_limit failed", exc_info=True)
        return None


def is_large_on_device_model(model: Any) -> bool:
    """An on-device ``local/`` model big enough to extract its own glossary: the 9B one (the Heavy preset,
    Qwen3.5-9B); smaller on-device models default to NO GLOSSARY."""
    value = str(model or "").strip().lower()
    return value.startswith("local/") and "9b" in value[len("local/"):].replace("-", "").replace("_", "")


def local_context_limit(model: Any) -> Optional[int]:
    """The configured context of an on-device ``local/<stem>`` model (``LocalEngineService.context_for``),
    or None without the app's engine service."""
    value = str(model or "").strip()
    if not value.lower().startswith("local/"):
        return None
    try:
        from glossarion_mobile.services.local_llm.service import default_service

        service = default_service()
        return int(service.context_for(value[len("local/"):])) if service is not None else None
    except Exception:
        log.debug("local_context_limit failed", exc_info=True)
        return None


def local_context_env() -> Optional[str]:
    """The ``CONTEXT_ENV`` value a job runs with: ``local/<stem>=<tokens>`` for every on-device model
    (``LocalEngineService.context_for``, the window its requests must fit), joined by ``;``; None without
    the app's engine service or a downloaded model."""
    try:
        from glossarion_mobile.services.local_llm.service import default_service

        service = default_service()
        if service is None:
            return None
        pairs = [f"local/{stem}={int(service.context_for(stem))}" for stem in sorted(service.registry.load())]
        return ";".join(pairs) or None
    except Exception:
        log.debug("local_context_env failed", exc_info=True)
        return None


def with_mobile_streaming_defaults(config: Any) -> Any:
    """A job's config snapshot (changed in place and returned) with the mobile defaults for absent keys:
    Streaming on, the 16,384-token budget, the model (``MOBILE_DEFAULT_MODEL``) and the model's batch
    size (:func:`default_batch_size`), plus the desktop thoughts lock (stream thinking on keeps Enable
    thoughts on, ``settings_rules.apply_thoughts_lock``). Saved values win; config.json is never written."""
    if not isinstance(config, dict):
        return config
    for key in STREAMING_KEYS:
        config.setdefault(key, MOBILE_STREAMING_DEFAULT)
    saved_output = config.get("max_output_tokens")
    # owner (U13): a phone run asks for at most 16,384 output tokens unless the user saved another budget
    config.setdefault("max_output_tokens", MOBILE_MAX_OUTPUT_TOKENS)
    if not config.get("model"):
        config["model"] = MOBILE_DEFAULT_MODEL
    if config.get("batch_size") in (None, ""):
        config["batch_size"] = default_batch_size(config.get("model"))
    if str(config.get("model") or "").strip().lower().startswith("local/"):
        # the on-device engine (services.local_llm) answers one request at a time: parallel requests would
        # only queue at its endpoint and spend the client's timeout waiting
        config.update(LOCAL_ENGINE_JOB_CONFIG)
        # the saved output budget (owner 2026-10-20: editable on the Welcome), never more than the model's
        # context; without one, half the context (owner 2026-10-18), the prompt keeping the rest
        context = local_context_limit(config.get("model"))
        try:
            saved_output = int(saved_output) if saved_output not in (None, "") else 0
        except (TypeError, ValueError):
            saved_output = 0
        limit = saved_output or local_output_limit(config.get("model"))
        if limit and context:
            limit = min(int(limit), int(context))
        if limit:
            config["max_output_tokens"] = int(limit)
    if bool(config.get("stream_thinking_logs")) and not bool(config.get("enable_thoughts", THOUGHTS_DEFAULT)):
        rules = _rules()
        try:
            rules.apply_thoughts_lock(config, True)
        except Exception:
            config["enable_thoughts"] = True
    return config


def _write_streaming(store: Any, on: Any) -> list:
    """The Streaming switch: ``settings_rules.apply_streaming`` on a copy of the config (absent toggles
    count as ON), then one write of every key whose value changed - never a no-op write (stream thinking's
    rule would turn Enable thoughts off on one)."""
    snapshot = getattr(store, "snapshot", None)
    if not callable(snapshot):
        return []
    try:
        config = dict(snapshot() or {})
    except Exception:
        return []
    before = effective_streaming_values(config)
    rules = _rules()
    if rules is not None and hasattr(rules, "apply_streaming"):
        try:
            rules.apply_streaming(config, bool(on), unset=MOBILE_STREAMING_DEFAULT)
        except Exception:
            log.exception("apply_streaming failed")
            return []
    else:  # no backend (a broken build): the four toggles alone
        config.update({key: bool(on) for key in STREAMING_KEYS})
    after = effective_streaming_values(config)
    changes = {key: after[key] for key in STREAMING_WRITES if bool(after[key]) != bool(before[key])}
    return _set_many(store, changes) if changes else []


def reset_streaming(store: Any) -> dict:
    """Reset of the Streaming switch: the toggles' keys removed (absent = the mobile default, ON), and a
    stored Enable thoughts OFF with them (the lock keeps thoughts on while streaming is on). Returns
    ``{key: old value}`` of what was removed (Undo: ``store.set_many(old)``)."""
    old = {}
    for key in STREAMING_WRITES:
        present, value = _stored_value(store, key)
        if not present or (key == "enable_thoughts" and (bool(value) or not MOBILE_STREAMING_DEFAULT)):
            continue
        old[key] = value
    for key in old:
        store.unset(key)
    return old


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


def on_device_translation_profile(method: Any = None) -> str:
    """The On-device translation profile for a text extraction method ('standard' when unknown)."""
    return ON_DEVICE_TRANSLATION_PROFILES.get(str(method or ""), ON_DEVICE_TRANSLATION_PROFILES["standard"])


def _is_on_device_translation(name: Any) -> bool:
    """One of the On-device translation profiles."""
    return name in ON_DEVICE_TRANSLATION_PROFILES.values()


def _write_extraction_method(store: Any, method: Any) -> list:
    """The text extraction method (any model, any profile: a profile without a _BeautifulSoup / _html2text
    suffix such as Universal never changes it, so this is what decides); with an On-device translation
    profile active, the On-device profile of that method.

    A stale legacy ``extraction_mode`` 'enhanced' would win over 'standard' at run time
    (``owner_state.initialize_extraction_variables``): it becomes the filtering level, as the desktop
    save writes it (``settings_persistence``: 'enhanced' or the file filtering level)."""
    values = {EXTRACTION_KEY: method}
    _present, legacy_mode = _stored_value(store, "extraction_mode")
    if method != "enhanced" and legacy_mode == "enhanced":
        _present, level = _stored_value(store, "file_filtering_level")
        values["extraction_mode"] = level or "smart"
    changed = _set_many(store, values)
    _present, profile = _stored_value(store, PROFILE_KEY)
    if _is_on_device_translation(profile) and method in ON_DEVICE_TRANSLATION_PROFILES:
        wanted = on_device_translation_profile(method)
        if profile != wanted:
            changed += _select_profile(store, wanted)
    return changed


def _select_profile(store: Any, name: Any) -> list:
    """Settings › Profiles' "Use this profile" (``ProfileService.select`` -> ``select_profile``).

    Like the desktop combo it only ever switches to an existing profile: with the profiles core loaded a
    name it does not know is refused (nothing written; a stored unknown name would make every run silently
    use the first profile). Without the core (names unknown) the plain key is written, as before."""
    try:
        from glossarion_mobile.ui.screens.profiles import ProfileService

        service = ProfileService(store)
        listing = service.core_listing()
        if listing is not None and listing.names and str(name) not in listing.names:
            log.warning("not a prompt profile, not selected: %r", name)
            return []
        changed = list(service.select(str(name)))
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


def _unset(store: Any, key: str) -> list:
    present, _value = _stored_value(store, key)
    if not present:
        return []
    unset = getattr(store, "unset", None)
    if callable(unset):
        unset(key)
        return [key]
    return _set_many(store, {key: None})


def _is_local_model(model: Any) -> bool:
    return str(model or "").strip().lower().startswith("local/")


def _select_glossary_profiles(store: Any, names: Mapping[str, str]) -> list:
    """The Glossary Manager's Balanced/Full and Minimal profile rows picking ``names[bucket]``
    (``glossary_document.GlossaryPromptProfiles.select``: the active profile plus the current prompt keys
    runs read); a name the bucket does not have picks Default. Only the keys that changed are written."""
    snapshot = getattr(store, "snapshot", None)
    if not callable(snapshot) or not names:
        return []
    try:
        import copy
        import types

        gd = importlib.import_module("glossary_document")
        config = dict(snapshot() or {})
        owner = types.SimpleNamespace(config=copy.deepcopy(config))
        keys = list(_GLOSSARY_PROFILE_KEYS)
        for bucket, name in names.items():
            meta = gd.glossary_prompt_profile_meta(bucket)
            keys += [k for k in (meta["config_key"], meta.get("legacy_config_key")) if k]
            current = owner.config.get(meta["config_key"])
            if not current:
                # no current prompt stored: open the row on the active profile's own text, so opening it
                # stages nothing new into that profile
                profiles = gd.glossary_prompt_profiles_for(owner, bucket)
                active = gd.active_glossary_prompt_profile_for(owner, bucket)
                current = profiles[active] if active in profiles else gd.default_glossary_prompt_profile_text(
                    owner, bucket)
            setattr(owner, meta["attr"], str(current or ""))
            row = gd.GlossaryPromptProfiles(owner, bucket, str(current or ""))
            if row.select(name) is None:
                row.select(gd.GLOSSARY_PROMPT_DEFAULT_PROFILE)
    except Exception:
        log.debug("glossary prompt profile select failed", exc_info=True)
        return []
    updates = {k: owner.config[k] for k in dict.fromkeys(keys) if k in owner.config and owner.config[k] != config.get(k)}
    return _set_many(store, updates) if updates else []


def _active_glossary_profiles(store: Any) -> dict:
    _present, active = _stored_value(store, "active_glossary_prompt_profiles")
    active = active if isinstance(active, Mapping) else {}
    return {bucket: str(active.get(bucket) or "") for bucket in _GLOSSARY_PROFILE_BUCKETS}


def _review_core() -> Any:
    try:
        rg = importlib.import_module("review_generator")
    except Exception:
        log.debug("review_generator unavailable", exc_info=True)
        return None
    return rg if callable(getattr(rg, "set_active_review_prompt_profile", None)) else None


def _change_review_profiles(store: Any, change: Any) -> list:
    """Run ``change(review_generator, config)`` on a copy of the config and write the review profile keys it
    changed (a migration's new profile included)."""
    snapshot = getattr(store, "snapshot", None)
    rg = _review_core()
    if not callable(snapshot) or rg is None:
        return []
    try:
        import copy

        config = dict(snapshot() or {})
        mine = copy.deepcopy(config)
        change(rg, mine)
    except Exception:
        log.debug("review prompt profile change failed", exc_info=True)
        return []
    updates = {k: mine[k] for k in _REVIEW_PROFILE_KEYS if k in mine and mine[k] != config.get(k)}
    return _set_many(store, updates) if updates else []


def _select_review_profile(store: Any, name: str) -> list:
    """The review generator's profile picker choosing ``name`` (an unknown name picks Default)."""
    def change(rg: Any, config: dict) -> None:
        if not rg.set_active_review_prompt_profile(config, name):
            rg.set_active_review_prompt_profile(config, rg.REVIEW_PROMPT_DEFAULT_PROFILE)

    return _change_review_profiles(store, change)


def _active_review_profile(store: Any) -> Optional[str]:
    """The selected review prompt profile (after the migration of older prompts); None without the core."""
    snapshot = getattr(store, "snapshot", None)
    rg = _review_core()
    if not callable(snapshot) or rg is None:
        return None
    try:
        return str(rg.active_review_prompt_profile(dict(snapshot() or {})))
    except Exception:
        log.debug("active review prompt profile unavailable", exc_info=True)
        return None


def _use_on_device_profiles(store: Any, model: Any = None) -> list:
    """A ``local/`` model was chosen: select the On-device translation profile of the text extraction method
    and the On-device glossary profiles and remember
    what was active (``ON_DEVICE_BEFORE_KEY``). A part already on On-device keeps the remembered value."""
    _present, record = _stored_value(store, ON_DEVICE_BEFORE_KEY)
    record = record if isinstance(record, Mapping) else {}
    remembered = record.get("glossary") if isinstance(record.get("glossary"), Mapping) else {}
    _present, profile = _stored_value(store, PROFILE_KEY)
    glossary = _active_glossary_profiles(store)
    before = {
        PROFILE_KEY: (str(record.get(PROFILE_KEY) or "") if _is_on_device_translation(profile)
                      else str(profile or "")),
        "glossary": {bucket: str(remembered.get(bucket) or "") if name == ON_DEVICE_PROFILE else name
                     for bucket, name in glossary.items()},
    }
    review = _active_review_profile(store)
    if review is not None:
        before["review"] = str(record.get("review") or "") if review == ON_DEVICE_PROFILE else review
    fields_present, fields = _stored_value(store, CUSTOM_FIELDS_KEY)
    removed_present, removed = _stored_value(store, DESCRIPTION_REMOVED_KEY)
    if "fields" in record:  # already On-device: keep what was remembered then
        before["fields"] = record.get("fields")
    elif fields_present or removed_present:
        before["fields"] = {CUSTOM_FIELDS_KEY: list(fields) if isinstance(fields, (list, tuple)) else fields,
                            DESCRIPTION_REMOVED_KEY: removed}
    else:
        before["fields"] = None  # nothing stored: the desktop default ("description") again
    applied = dict(ON_DEVICE_SETTINGS)
    limit = local_output_limit(model)
    if limit:
        applied[OUTPUT_LIMIT_KEY] = int(limit)
    context = local_context_limit(model)
    if context:
        applied[INPUT_LIMIT_KEY] = int(context)
    if not is_large_on_device_model(model):
        applied.update(ON_DEVICE_GLOSSARY_SETTINGS)
    if isinstance(record.get("settings"), Mapping):  # already On-device: keep what was remembered then
        before["settings"] = dict(record["settings"])
    else:
        before["settings"] = {}
        for key in applied:
            present, value = _stored_value(store, key)
            before["settings"][key] = [bool(present), value]
    before["applied"] = applied
    _present, method = _stored_value(store, EXTRACTION_KEY)
    changed = _select_profile(store, on_device_translation_profile(method))
    changed += _select_glossary_profiles(store, {bucket: ON_DEVICE_PROFILE for bucket in _GLOSSARY_PROFILE_BUCKETS})
    if review is not None:
        changed += _select_review_profile(store, ON_DEVICE_PROFILE)
    changed += _set_many(store, applied)
    # no extra glossary columns: the "removed" flag keeps the desktop from adding "description" back
    changed += _set_many(store, {CUSTOM_FIELDS_KEY: [], DESCRIPTION_REMOVED_KEY: True, ON_DEVICE_BEFORE_KEY: before})
    return changed


def _restore_profiles_before_on_device(store: Any) -> list:
    """A model that is not ``local/`` replaced a ``local/`` one: bring back the remembered profiles where
    On-device is still the active one (a profile the user picked meanwhile stays)."""
    present, record = _stored_value(store, ON_DEVICE_BEFORE_KEY)
    if not present:
        return []
    record = record if isinstance(record, Mapping) else {}
    changed: list = []
    _present, profile = _stored_value(store, PROFILE_KEY)
    if _is_on_device_translation(profile):
        previous = str(record.get(PROFILE_KEY) or "")
        # nothing was stored before: the first profile again, as before
        changed += _select_profile(store, previous) if previous else _unset(store, PROFILE_KEY)
    remembered = record.get("glossary") if isinstance(record.get("glossary"), Mapping) else {}
    names = {bucket: str(remembered.get(bucket) or "") or "Default"
             for bucket, name in _active_glossary_profiles(store).items() if name == ON_DEVICE_PROFILE}
    changed += _select_glossary_profiles(store, names)
    settings = record.get("settings") if isinstance(record.get("settings"), Mapping) else {}
    applied = record.get("applied") if isinstance(record.get("applied"), Mapping) else {}
    if _active_review_profile(store) == ON_DEVICE_PROFILE and "review" in record:
        changed += _select_review_profile(store, str(record.get("review") or "") or "Default")
    _present, fields = _stored_value(store, CUSTOM_FIELDS_KEY)
    if "fields" in record and not fields:  # columns the user added meanwhile stay
        previous = record.get("fields")
        if isinstance(previous, Mapping):
            changed += _set_many(store, {CUSTOM_FIELDS_KEY: previous.get(CUSTOM_FIELDS_KEY) or [],
                                         DESCRIPTION_REMOVED_KEY: bool(previous.get(DESCRIPTION_REMOVED_KEY))})
        else:
            changed += _unset(store, CUSTOM_FIELDS_KEY) + _unset(store, DESCRIPTION_REMOVED_KEY)
    for key, on_device_value in applied.items():
        _present, current = _stored_value(store, key)
        if current != on_device_value:  # changed meanwhile: the user's value stays
            continue
        previous = settings.get(key)
        if isinstance(previous, (list, tuple)) and len(previous) == 2 and previous[0]:
            changed += _set_many(store, {key: previous[1]})
        else:  # nothing was saved before: the default again
            changed += _unset(store, key)
    changed += _unset(store, ON_DEVICE_BEFORE_KEY)
    return changed


def _write_model(store: Any, value: Any) -> list:
    """The global model, plus the On-device prompt profiles when it moves to or from a ``local/`` model."""
    _present, old = _stored_value(store, "model")
    changed = _set_many(store, {"model": value})
    if _is_local_model(value) and not _is_local_model(old):
        changed += _use_on_device_profiles(store, value)
    elif value and _is_local_model(old) and not _is_local_model(value):
        changed += _restore_profiles_before_on_device(store)
    return changed


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
    if key == STREAMING_KEY:
        return _write_streaming(store, value)
    if key == "model":
        return _write_model(store, value)
    if key == EXTRACTION_KEY:
        return _write_extraction_method(store, value)
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
