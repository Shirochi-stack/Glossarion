"""Shared GUI-free settings rules: what a setting change implies and which controls a model needs.

Glossarion mobile rewrite, milestone U4 (shared-core design section 2 "settings_rules"). The
desktop handlers and the mobile app read the same rules:

* **Model-route controls** (``route_controls``; ``TranslatorGUI.on_model_change`` and its
  helpers). The TranslatorGUI helpers ``_iter_enabled_key_pool_models``,
  ``_model_needs_google_creds``, ``_has_google_creds_model_in_key_pools``,
  ``_has_vertex_model_in_key_pools``, ``_has_auth*_in_key_pools``,
  ``_authgpt_pool_route_requested`` / ``_authgrok_pool_route_requested``,
  ``_authgem_vertex_control_model`` and ``_collect_auth_account_ids_from_pools`` were moved
  here verbatim and are thin wrappers now; ``on_model_change`` takes its decisions from
  ``google_credentials_route`` / ``authgpt_login_needed`` / ``authgrok_login_needed`` /
  ``authcd_login_needed`` / ``authgem_login_needed`` / ``google_creds_ready_text`` and then
  updates its widgets. The excluded desktop-only routes (antigravity, ocagy, Z.AI, Arena)
  keep their desktop blocks; ``excluded_route_reason`` gives mobile their reason chip.
* **Main-window Chunk Size** (``TranslatorGUI._on_chunk_size_edited`` / ``_apply_chunk_size`` /
  ``_remember_manual_chunk_size`` / ``_hold_manual_chunk_size``): ``parse_chunk_size_text``,
  ``factor_for_chunk_size``, ``record_chunk_size``, ``remember_manual_chunk_size``,
  ``held_manual_chunk_size`` (moved; the desktop methods call them).
* **Rules whose desktop code already is a shared mixin method** (moved verbatim in U2 and
  pinned byte-for-byte by tests/test_headless_owner.py): the auto-glossary shortcut handler,
  the context-mode / batching handlers, the auto compression factor, the chunk budget and the
  target-language fan-out. They are *run*, never copied: ``_RuleOwner`` is a GUI-free owner
  (``ConfigStateMixin`` + ``RunEnvMixin``, no widgets, ``save_config`` a no-op) whose ``*_var``
  attributes are seeded from the config through the ``settings_schema`` var names and
  ``_init_variables`` defaults, and the handlers run with a private ``os.environ``
  (``_run_isolated``) so a UI thread never writes the process environment (a running job
  owns it; mobile settings apply to the next run).
* ``evaluate_locks(config)``: ``{key: LockInfo}`` for settings UIs, aggregating every
  registered lock rule: context batching, disable temperature, the stream-thinking lock of
  "Enable thoughts" and the Glossary Manager mode locks (🔒 Append Glossary / Auto-Mapping /
  Fuzzy Auto-Mapping and its similarity slider).
* **Other Settings / Glossary Manager rules** (sections 9-11): ``thoughts_lock_state``
  (other_settings._sync_thoughts_lock_state), ``output_mode_flags`` / ``output_mode_sub_settings``
  (other_settings._set_output_mode and the Image Translation sub-settings),
  ``glossary_mode_from_display`` / ``glossary_mode_toggle_steps``
  (GlossaryManager_GUI update_auto_glossary_state). The desktop handlers
  call these and keep their widget work. Section 9b holds the streaming group's two dialog
  notes (``STREAMING_TRUNCATION_WARNING`` / ``FORCED_STREAM_NOTE``, other_settings shows them) and
  ``apply_streaming`` / ``streaming_mode``, the group as one switch (Glossarion Mobile).
* ``evaluate(rule_id, config)``: the evaluator of ``settings_schema`` ``visible_if`` /
  ``locked_if`` rule ids (``lock:<key>`` -> lock reason or ''; visibility ids -> bool), and
  ``apply_change(config, key, value)`` for the controls whose change has side effects.

Rules: Python 3.10 compatible; never import PySide6, translator_gui or dpi_setup; backend and
mixin modules are imported lazily (importing this module stays cheap for a UI thread).
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import threading
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

__all__ = [
    # model-route controls
    "ENABLED_KEY_POOL_TOGGLES", "LOGIN_KEY_POOL_TOGGLES", "AUTH_ACCOUNT_ROUTE_PATTERNS",
    "EXCLUDED_ROUTE_PREFIXES", "GOOGLE_CREDS_READ_ERROR", "GOOGLE_CREDS_FILE_MISSING",
    "RouteControls",
    "iter_enabled_key_pool_models", "is_vertex_route", "model_needs_google_creds",
    "has_google_creds_model_in_key_pools", "has_vertex_model_in_key_pools",
    "has_authgpt_in_key_pools", "has_authgrok_in_key_pools", "has_authgem_in_key_pools",
    "has_authgem_vertex_in_key_pools", "has_authcd_in_key_pools",
    "authgpt_pool_route_requested", "authgrok_pool_route_requested", "authgem_vertex_control_model",
    "collect_auth_account_ids_from_pools", "model_account_ids",
    "google_credentials_route", "google_creds_ready_text", "google_creds_missing_text", "google_creds_status",
    "INVALID_GOOGLE_CREDENTIALS", "google_credentials_error", "google_credentials_load_error",
    "is_google_service_account",
    "authgpt_login_needed", "authgrok_login_needed", "authcd_login_needed", "authgem_login_needed",
    "excluded_route_reason", "route_controls",
    # chunk size / compression
    "parse_chunk_size_text", "factor_for_chunk_size", "record_chunk_size", "remember_manual_chunk_size",
    "held_manual_chunk_size", "chunk_budget_margin", "auto_compression_factor", "apply_auto_compression_factor",
    "chunk_budget", "chunk_budget_for_config", "chunk_size_display", "apply_chunk_size", "set_chunk_size_text",
    "hold_manual_chunk_size", "COMPRESSION_LOCK_REASON", "compression_lock",
    # Glossary Manager auto compression factor
    "GLOSSARY_COMPRESSION_LOCK_REASON", "glossary_output_limit", "glossary_auto_compression_factor",
    "apply_glossary_auto_compression", "glossary_compression_lock",
    # context mode / batching
    "BATCHING_RADIOS", "ChoiceState", "ContextBatching", "context_mode", "apply_context_mode",
    "enforce_context_batching", "context_batching_controls",
    # auto glossary mode
    "AUTO_GLOSSARY_MODE_KEYS", "auto_glossary_modes", "auto_glossary_mode_index", "auto_glossary_mode_flags",
    "apply_auto_glossary_mode",
    # target language
    "fan_out_target_language",
    # locks
    "LockInfo", "register_lock_rule", "lock_rules", "evaluate_locks",
    # thinking (other_settings)
    "THOUGHTS_LOCK_KEYS", "THOUGHTS_LOCK_REASON", "thoughts_lock_state", "apply_thoughts_lock", "thoughts_lock",
    # streaming (other_settings texts; the Glossarion Mobile Streaming switch)
    "STREAMING_TRUNCATION_WARNING", "FORCED_STREAM_NOTE", "STREAMING_ENV", "STREAMING_KEYS", "streaming_states",
    "streaming_mode", "apply_streaming",
    # output mode (other_settings)
    "OUTPUT_MODES", "OUTPUT_MODE_INDEX", "OutputModeFlags", "normalize_output_mode", "output_mode_flags",
    "apply_output_mode", "current_output_mode", "output_mode_sub_settings",
    # Glossary Manager mode locks
    "GLOSSARY_MODE_DISPLAY", "GLOSSARY_MODES_WITHOUT_EXTRACTION",
    "GLOSSARY_APPEND_FREE_MODES", "GLOSSARY_AUTOMAP_ON_MODES", "GLOSSARY_AUTOMAP_OFF_MODES",
    "GLOSSARY_FUZZY_MODES", "GLOSSARY_MODE_TOGGLES", "ToggleStep", "glossary_mode_from_display",
    "glossary_mode_label", "glossary_mode_extracts", "glossary_mode_targeted_extraction",
    "glossary_manager_mode", "glossary_mode_toggle_steps",
    "glossary_mode_toggle_states", "glossary_mode_locks", "apply_glossary_mode_locks",
    # rule ids (settings_schema visible_if / locked_if) and setting changes
    "LOCK_RULE_PREFIX", "lock_rule_id", "lock_keys", "register_visibility_rule", "visibility_rules",
    "is_visible", "evaluate", "evaluate_rule", "apply_change",
    # U9: the Text Extraction Method the desktop shows (extraction:standard / extraction:enhanced rules)
    "text_extraction_method",
]


# =============================================================================================
# 1. Model-route controls (TranslatorGUI.on_model_change and its key-pool helpers)
# =============================================================================================

#: TranslatorGUI._iter_enabled_key_pool_models: every key pool the route controls scan
#: (pool key -> its enable toggle), in the desktop order.
ENABLED_KEY_POOL_TOGGLES = {
    'multi_api_keys': 'use_multi_api_keys',
    'fallback_keys': 'use_fallback_keys',
    'glossary_keys': 'use_glossary_keys',
    'glossary_refinement_keys': 'use_glossary_refinement_keys',
    'metadata_keys': 'use_metadata_keys',
    'qa_scan_keys': 'use_qa_scan_keys',
    'rolling_summary_keys': 'use_rolling_summary_keys',
    'truncation_retry_keys': 'use_truncation_retry_keys',
    'inpainter_keys': 'use_inpainter_keys',
    'tts_keys': 'use_tts_keys',
}

#: The six pools the login-route scans read (TranslatorGUI._has_authgpt_in_key_pools,
#: _has_authgem_in_key_pools, _has_authgem_vertex_in_key_pools, _has_authcd_in_key_pools,
#: _collect_auth_account_ids_from_pools): chat pools only, not metadata/QA/inpainter/TTS.
LOGIN_KEY_POOL_TOGGLES = {
    'multi_api_keys': 'use_multi_api_keys',
    'fallback_keys': 'use_fallback_keys',
    'glossary_keys': 'use_glossary_keys',
    'glossary_refinement_keys': 'use_glossary_refinement_keys',
    'rolling_summary_keys': 'use_rolling_summary_keys',
    'truncation_retry_keys': 'use_truncation_retry_keys',
}

#: Numbered account routes (TranslatorGUI._collect_auth_account_ids_from_pools): the digits
#: after the provider name are the token-store slot (``authgpt/`` = slot 0).
AUTH_ACCOUNT_ROUTE_PATTERNS = {
    'authgpt': re.compile(r'^authgpt(\d{0,4})/'),
    'authgrok': re.compile(r'^authgrok(\d{0,4})/'),
    'authcd': re.compile(r'^authcd(\d{0,4})/'),
    'authgem': re.compile(r'^authgem(?:-vertex)?(\d{0,4})/'),
}

GOOGLE_CREDS_READ_ERROR = "⚠ Error reading credentials"
GOOGLE_CREDS_FILE_MISSING = "⚠ Credentials file not found"


def _pool_source(config, pool_models):
    """Zero-argument callable yielding ``(pool_key, toggle_key, model)`` triples.

    The desktop wrappers pass ``lambda: self._iter_enabled_key_pool_models()`` (evaluated
    lazily, inside the same try blocks as before); config callers get the shared scan.
    """
    if pool_models is not None:
        return pool_models

    def source():
        return iter_enabled_key_pool_models(config)

    return source


def iter_enabled_key_pool_models(config):
    """Yield ``(pool_key, toggle_key, model)`` for each enabled entry of each enabled key pool.

    Moved from TranslatorGUI._iter_enabled_key_pool_models (the desktop generator delegates).
    """
    pool_map = ENABLED_KEY_POOL_TOGGLES
    for pool_key, toggle_key in pool_map.items():
        if not config.get(toggle_key, False):
            continue
        for key_data in config.get(pool_key, []):
            if isinstance(key_data, dict):
                if not key_data.get('enabled', True):
                    continue
                model = key_data.get('model', '')
            else:
                if not getattr(key_data, 'enabled', True):
                    continue
                model = getattr(key_data, 'model', '')
            yield pool_key, toggle_key, model


def is_vertex_route(model):
    """Vertex AI route test of on_model_change (raw model text): ``model@version``, vertex/, vertex_ai/."""
    return '@' in model or model.startswith('vertex/') or model.startswith('vertex_ai/')


def model_needs_google_creds(model):
    """Return True when a model route requires Google Cloud service-account creds.

    Moved from TranslatorGUI._model_needs_google_creds.
    """
    model = (model or '').strip().lower()
    return (
        is_vertex_route(model)
        or model == 'google-translate'
    )


def has_google_creds_model_in_key_pools(config=None, *, pool_models=None):
    """Check enabled key pools for models that require Google Cloud credentials.

    Moved from TranslatorGUI._has_google_creds_model_in_key_pools.
    """
    source = _pool_source(config, pool_models)
    try:
        for _pool_key, _toggle_key, model in source():
            if model_needs_google_creds(model):
                return True
    except Exception:
        pass
    return False


def has_vertex_model_in_key_pools(config=None, *, pool_models=None):
    """Check enabled key pools for Vertex-style models that need the location field.

    Moved from TranslatorGUI._has_vertex_model_in_key_pools.
    """
    source = _pool_source(config, pool_models)
    try:
        for _pool_key, _toggle_key, model in source():
            model = (model or '').strip().lower()
            if is_vertex_route(model):
                return True
    except Exception:
        pass
    return False


def has_authgpt_in_key_pools(config):
    """Check if any enabled key pool contains an enabled authgpt model.

    Moved from TranslatorGUI._has_authgpt_in_key_pools.
    """
    try:
        pool_map = LOGIN_KEY_POOL_TOGGLES
        for pool_key, toggle_key in pool_map.items():
            if config.get(toggle_key, False):
                for key_data in config.get(pool_key, []):
                    if isinstance(key_data, dict):
                        if not key_data.get('enabled', True):
                            continue
                        m = key_data.get('model', '')
                    else:
                        m = getattr(key_data, 'model', '')
                    if re.match(r'^authgpt\d{0,4}/', m):
                        return True
    except Exception:
        pass
    return False


def has_authgrok_in_key_pools(config=None, *, pool_models=None):
    """Check if any enabled key pool contains an AuthGrok model.

    Moved from TranslatorGUI._has_authgrok_in_key_pools.
    """
    source = _pool_source(config, pool_models)
    try:
        for _pool_key, _toggle_key, model in source():
            if re.match(r'^authgrok\d{0,4}/', str(model or '').strip().lower()):
                return True
    except Exception:
        pass
    return False


def has_authgem_in_key_pools(config):
    """Check if any enabled key pool contains an enabled authgem model.

    Moved from TranslatorGUI._has_authgem_in_key_pools.
    """
    try:
        pool_map = LOGIN_KEY_POOL_TOGGLES
        for pool_key, toggle_key in pool_map.items():
            if config.get(toggle_key, False):
                for key_data in config.get(pool_key, []):
                    if isinstance(key_data, dict):
                        if not key_data.get('enabled', True):
                            continue
                        m = key_data.get('model', '')
                    else:
                        m = getattr(key_data, 'model', '')
                    if m.startswith('authgem') and ('/' in m):
                        return True
    except Exception:
        pass
    return False


def has_authgem_vertex_in_key_pools(config):
    """Check if any enabled key pool contains an enabled authgem-vertex model.

    Moved from TranslatorGUI._has_authgem_vertex_in_key_pools.
    """
    try:
        pool_map = LOGIN_KEY_POOL_TOGGLES
        for pool_key, toggle_key in pool_map.items():
            if config.get(toggle_key, False):
                for key_data in config.get(pool_key, []):
                    if isinstance(key_data, dict):
                        if not key_data.get('enabled', True):
                            continue
                        m = key_data.get('model', '')
                    else:
                        m = getattr(key_data, 'model', '')
                    if re.match(r'^authgem-vertex\d{0,4}/', m):
                        return True
    except Exception:
        pass
    return False


def has_authcd_in_key_pools(config):
    """Check if any enabled key pool contains an enabled authcd model.

    Moved from TranslatorGUI._has_authcd_in_key_pools.
    """
    try:
        pool_map = LOGIN_KEY_POOL_TOGGLES
        for pool_key, toggle_key in pool_map.items():
            if config.get(toggle_key, False):
                for key_data in config.get(pool_key, []):
                    if isinstance(key_data, dict):
                        if not key_data.get('enabled', True):
                            continue
                        m = key_data.get('model', '')
                    else:
                        m = getattr(key_data, 'model', '')
                    if re.match(r'^authcd\d{0,4}/', m):
                        return True
    except Exception:
        pass
    return False


def authgpt_pool_route_requested(active_model, config=None, *, hint=False, pool_models=None):
    """Return whether authgpt0/ (rotation over every signed-in ChatGPT slot) is requested.

    *active_model* is the model under test (the desktop passes the model argument, else
    model_var / config['model']); *hint* is the Multi-Key Manager's live editor hint. Moved
    from TranslatorGUI._authgpt_pool_route_requested.
    """
    source = _pool_source(config, pool_models)
    active_model = str(active_model).strip().lower()
    if re.match(r'^authgpt0(?:/|$)', active_model):
        return True
    if bool(hint):
        return True
    try:
        for _pool_key, _toggle_key, pool_model in source():
            if re.match(
                r'^authgpt0(?:/|$)',
                str(pool_model or '').strip().lower(),
            ):
                return True
    except Exception:
        pass
    return False


def authgrok_pool_route_requested(active_model, config=None, *, hint=False, pool_models=None):
    """Return whether authgrok0/ is requested by the model, the live hint or an enabled pool.

    Moved from TranslatorGUI._authgrok_pool_route_requested.
    """
    source = _pool_source(config, pool_models)
    active_model = str(active_model).strip().lower()
    if re.match(r'^authgrok0(?:/|$)', active_model):
        return True
    if bool(hint):
        return True
    try:
        for _pool_key, _toggle_key, pool_model in source():
            if re.match(
                r'^authgrok0(?:/|$)',
                str(pool_model or '').strip().lower(),
            ):
                return True
    except Exception:
        pass
    return False


def authgem_vertex_control_model(primary, hint='', config=None, *, pool_models=None):
    """Resolve the Vertex route from the main field, live editor hint, or enabled pools ('' if none).

    Moved from TranslatorGUI._authgem_vertex_control_model (the pool scan is not guarded,
    as on desktop).
    """
    source = _pool_source(config, pool_models)
    primary = str(primary or '').strip().lower()
    pattern = r'^authgem-vertex\d{0,4}(?:/|$)'
    if re.match(pattern, primary):
        return primary
    hint = str(hint or '').strip().lower()
    if re.match(pattern, hint):
        return hint
    for _, _, route in source():
        route = str(route or '').strip().lower()
        if re.match(pattern, route):
            return route
    return ''


def collect_auth_account_ids_from_pools(config):
    """Scan enabled key pools and return sets of account IDs per provider.

    Returns dict: {'authgpt': {0, 2}, 'authgrok': {0}, 'authcd': {1}, 'authgem': {0, 3}}.
    Moved from TranslatorGUI._collect_auth_account_ids_from_pools.
    """
    result = {'authgpt': set(), 'authgrok': set(), 'authcd': set(), 'authgem': set()}
    patterns = AUTH_ACCOUNT_ROUTE_PATTERNS
    try:
        pool_map = LOGIN_KEY_POOL_TOGGLES
        for pool_key, toggle_key in pool_map.items():
            if config.get(toggle_key, False):
                for key_data in config.get(pool_key, []):
                    if isinstance(key_data, dict):
                        if not key_data.get('enabled', True):
                            continue
                        m = key_data.get('model', '')
                    else:
                        m = getattr(key_data, 'model', '')
                    m = str(m or '').strip().lower()
                    for provider, pat in patterns.items():
                        match = pat.match(m)
                        if match:
                            acct_id = int(match.group(1)) if match.group(1) else 0
                            result[provider].add(acct_id)
    except Exception:
        pass
    return result


def model_account_ids(model, config=None):
    """Account slots per provider referenced by *model* plus the enabled key pools.

    The pool part is ``collect_auth_account_ids_from_pools``; the model's own slot uses the
    same patterns (TranslatorGUI._refresh_auth_account_arrows adds it the same way).
    """
    ids = collect_auth_account_ids_from_pools(config if config is not None else {})
    text = str(model or '').strip().lower()
    for provider, pat in AUTH_ACCOUNT_ROUTE_PATTERNS.items():
        match = pat.match(text)
        if match:
            ids[provider].add(int(match.group(1)) if match.group(1) else 0)
    return ids


def google_credentials_route(model, config=None, *, pools_need_creds=None, pools_have_vertex=None, hint=False):
    """``(needs_google_creds, vertex_location)`` for *model* (TranslatorGUI.on_model_change).

    *vertex_location*: True = show the Vertex AI location field (Vertex routes, Vertex pool
    entries, the Multi-Key Manager's creds hint), False = hide it (paid ``google-translate``),
    None = untouched (the desktop then hides it with the credential row when no credentials
    are needed). *pools_need_creds* / *pools_have_vertex* are zero-argument callables (the
    desktop passes its key-pool helpers); config callers get the shared scans.
    """
    if pools_need_creds is None:
        def pools_need_creds():
            return has_google_creds_model_in_key_pools(config)
    if pools_have_vertex is None:
        def pools_have_vertex():
            return has_vertex_model_in_key_pools(config)
    # Show Google Cloud Credentials button for Vertex AI models AND Google Translate (paid)
    needs_google_creds = (
        model_needs_google_creds(model)
        or pools_need_creds()
        or bool(hint)
    )
    vertex_location = None
    if (
        is_vertex_route(model)
        or pools_have_vertex()
        or bool(hint)
    ):
        needs_google_creds = True
        vertex_location = True  # Show location selector for Vertex
    elif model.lower() == 'google-translate':  # Exact match for paid Google Translate (not google-translate-free)
        needs_google_creds = True
        vertex_location = False  # Hide location selector for Google Translate
    return needs_google_creds, vertex_location


#: The desktop Google credential pickers' refusal (translator_gui.select_google_credentials,
#: multi_api_key_manager._browse_google_credentials and its fallback / glossary variants).
INVALID_GOOGLE_CREDENTIALS = "Invalid Google Cloud credentials file. Please select a valid service account JSON file."


def is_google_service_account(creds_data):
    """The desktop pickers' acceptance test of the picked JSON: ``'type'`` and ``'project_id'`` present (a
    scalar raises TypeError, which the pickers report like a read failure)."""
    return 'type' in creds_data and 'project_id' in creds_data


def google_credentials_load_error(exc):
    """The pickers' message when reading the file (or using it) raised ``exc``."""
    return f"Failed to load credentials: {str(exc)}"


def google_credentials_error(creds_path):
    """The desktop pickers' check of a picked credentials file: None for a service-account JSON
    (``is_google_service_account``), else the message they show (``INVALID_GOOGLE_CREDENTIALS``, or
    ``google_credentials_load_error`` when the file cannot be read as JSON). The desktop pickers use
    the same rule and messages; U9: Glossarion Mobile's KeyEditor and Settings › Google Cloud
    credentials refuse the same files."""
    try:
        with open(creds_path, 'r') as f:
            creds_data = json.load(f)
            if is_google_service_account(creds_data):
                return None
            return INVALID_GOOGLE_CREDENTIALS
    except Exception as e:
        return google_credentials_load_error(e)


def google_creds_ready_text(model, creds_path):
    """Status line for loaded Google Cloud credentials (reads the JSON; raises on read errors).

    on_model_change shows ``GOOGLE_CREDS_READ_ERROR`` when this raises.
    """
    with open(creds_path, 'r') as f:
        creds_data = json.load(f)
        project_id = creds_data.get('project_id', 'Unknown')

        # Different status messages for different services
        if model == 'google-translate':
            status_text = f"✓ Google Translate ready\n(Project: {project_id})"
        else:
            status_text = f"✓ Credentials: {os.path.basename(creds_path)} (Project: {project_id})"
    return status_text


def google_creds_missing_text(model):
    """Prompt shown when a route needs Google Cloud credentials and none are selected."""
    # Different prompts for different services
    if model == 'google-translate':
        warning_text = "⚠ Google Cloud credentials needed for Translate API"
    else:
        warning_text = "⚠ No Google Cloud credentials selected"
    return warning_text


def google_creds_status(model, config):
    """``(text, level)`` of the credential row for *model*: level is ready / error / warning.

    The desktop branch order of on_model_change (credentials set -> file exists -> readable).
    """
    if config.get('google_cloud_credentials'):
        creds_path = config['google_cloud_credentials']
        if os.path.exists(creds_path):
            try:
                return google_creds_ready_text(model, creds_path), 'ready'
            except Exception:
                return GOOGLE_CREDS_READ_ERROR, 'error'
        return GOOGLE_CREDS_FILE_MISSING, 'error'
    return google_creds_missing_text(model), 'warning'


def authgpt_login_needed(model, config=None, *, pool_route_requested=None, in_key_pools=None):
    """Whether the ChatGPT (AuthGPT) login control applies to *model* (on_model_change).

    True for ``authgpt/`` / ``authgptN/`` routes, an authgpt0/ pool request, or an enabled
    pool entry with an authgpt model.
    """
    if pool_route_requested is None:
        def pool_route_requested(m):
            return authgpt_pool_route_requested(m, config)
    if in_key_pools is None:
        def in_key_pools():
            return has_authgpt_in_key_pools(config)
    _gpt_match = re.match(r'^authgpt\d{0,4}/', model)
    needs_authgpt = _gpt_match is not None or pool_route_requested(model)

    # Also check enabled key pools for authgpt models
    if not needs_authgpt:
        needs_authgpt = in_key_pools()
    return needs_authgpt


def authgrok_login_needed(model, config=None, *, in_key_pools=None, hint=False):
    """Whether the Grok (AuthGrok) login control applies to *model* (on_model_change)."""
    if in_key_pools is None:
        def in_key_pools():
            return has_authgrok_in_key_pools(config)
    needs_authgrok = re.match(r'^authgrok\d{0,4}/', model) is not None
    if not needs_authgrok:
        needs_authgrok = in_key_pools()
    if not needs_authgrok:
        needs_authgrok = bool(hint)
    return needs_authgrok


def authcd_login_needed(model, config=None, *, in_key_pools=None):
    """Whether the Claude (AuthCD) login control applies to *model* (on_model_change)."""
    if in_key_pools is None:
        def in_key_pools():
            return has_authcd_in_key_pools(config)
    _cd_match = re.match(r'^authcd\d{0,4}/', model)
    needs_authcd = _cd_match is not None

    if not needs_authcd:
        needs_authcd = in_key_pools()
    return needs_authcd


def authgem_login_needed(model, config=None, *, vertex_model=None, in_key_pools=None):
    """``(needs_authgem, needs_vertex)``: Gemini (AuthGem) login, and its GCP project picker.

    The project picker is ONLY needed for authgem-vertex/ (Vertex AI); authgem/ uses AI
    Studio, which needs no GCP project. *vertex_model* is a zero-argument callable (desktop:
    ``_authgem_vertex_control_model``) evaluated after the route test, as on desktop.
    """
    if vertex_model is None:
        def vertex_model():
            return authgem_vertex_control_model(model, '', config)
    if in_key_pools is None:
        def in_key_pools():
            return has_authgem_in_key_pools(config)
    _ag_match = re.match(r'^authgem(?:-vertex)?\d{0,4}/', model)
    vertex = vertex_model()
    needs_authgem = _ag_match is not None or bool(vertex)

    # Also check enabled key pools for authgem models
    if not needs_authgem:
        needs_authgem = in_key_pools()

    # GCP project dropdown is ONLY needed for authgem-vertex/ (Vertex AI)
    needs_vertex = bool(vertex)
    return needs_authgem, needs_vertex


#: Model routes that cannot run on mobile (plan "Excluded on mobile": npm/bun/desktop-binary
#: routes). Their rows stay visible, disabled with this reason; a value chosen on desktop is
#: kept in config.json untouched.
EXCLUDED_ROUTE_PREFIXES = {
    'ocagy': "OCAGY runs through a desktop npm/bun CLI, which a phone cannot start.",
    'ocz/': "OpenCode Zen free models need the desktop npm/bun CLI.",
    'autharena': "Arena needs a desktop browser session proxy (QtWebEngine).",
    'search/opera': "Opera Aria mints its token by driving a desktop Opera browser.",
}


def excluded_route_reason(model, platform='mobile'):
    """Reason *model* cannot run on *platform* (mobile excludes the npm/bun/desktop-binary
    routes), or None. Desktop runs every route."""
    if platform != 'mobile':
        return None
    value = str(model or '').strip().lower()
    for prefix, reason in EXCLUDED_ROUTE_PREFIXES.items():
        if value.startswith(prefix):
            return reason
    return None


@dataclass(frozen=True)
class RouteControls:
    """The route-driven controls of one model (main-window row under the model field)."""

    model: str
    provider: Optional[str]                  # model_options.catalog_provider_for_model
    excluded: Optional[str]                  # platform reason (mobile), else None
    needs_google_creds: bool                 # Google Cloud service-account JSON row
    vertex_location: bool                    # Vertex AI location field
    google_creds_text: str                   # the credential row's status line ('' when not needed)
    google_creds_level: str                  # ready / error / warning / ''
    logins: Tuple[str, ...]                  # login controls, desktop order: authgpt, authgrok, authcd, authgem
    authgem_vertex: bool                     # GCP project picker (authgem-vertex/)
    authgpt_pool: bool                       # authgpt0/ rotation over every signed-in slot
    authgrok_pool: bool                      # authgrok0/
    account_ids: Mapping[str, Tuple[int, ...]] = field(default_factory=dict)
    needs_api_key: Optional[bool] = None     # UnifiedClient._model_needs_api_key (None: unknown)

    def needs_login(self, provider):
        return provider in self.logins


def _needs_api_key(model):
    try:
        from unified_api_client import UnifiedClient
        return bool(UnifiedClient._model_needs_api_key(model))
    except Exception:
        return None


def _catalog_provider(model, config):
    try:
        import model_options
        return model_options.catalog_provider_for_model(model, config.get('custom_prefix_routes', []))
    except Exception:
        return None


def route_controls(model, config=None, *, platform='mobile', hints=None, api_key_check=True):
    """Which login / key / Google Cloud controls *model* needs, from *config* alone.

    The decisions are the ones TranslatorGUI.on_model_change takes (same functions), with the
    key-pool scans run on *config*. *hints* may carry the Multi-Key Manager's live editor
    hints (``needs_google_creds``, ``authgpt_pool``, ``authgrok_pool``, ``authgem_vertex_model``)
    while a key editor is open. ``needs_api_key`` comes from UnifiedClient (lazy import; None
    when the backend is not importable or *api_key_check* is False).
    """
    config = config if config is not None else {}
    hints = dict(hints or {})
    model = str(model or '')
    excluded = excluded_route_reason(model, platform)
    needs_google_creds, vertex_location = google_credentials_route(
        model, config, hint=hints.get('needs_google_creds', False))
    creds_text, creds_level = ('', '')
    if needs_google_creds:
        creds_text, creds_level = google_creds_status(model, config)

    def pool_route(m):
        return authgpt_pool_route_requested(m, config, hint=hints.get('authgpt_pool', False))

    logins = []
    if authgpt_login_needed(model, config, pool_route_requested=pool_route):
        logins.append('authgpt')
    if authgrok_login_needed(model, config, hint=hints.get('authgrok_pool', False)):
        logins.append('authgrok')
    if authcd_login_needed(model, config):
        logins.append('authcd')

    def vertex_model():
        return authgem_vertex_control_model(model, hints.get('authgem_vertex_model', ''), config)

    needs_authgem, needs_vertex = authgem_login_needed(model, config, vertex_model=vertex_model)
    if needs_authgem:
        logins.append('authgem')
    ids = model_account_ids(model, config)
    return RouteControls(
        model=model,
        provider=_catalog_provider(model, config),
        excluded=excluded,
        needs_google_creds=bool(needs_google_creds),
        vertex_location=vertex_location is True,
        google_creds_text=creds_text,
        google_creds_level=creds_level,
        logins=tuple(logins),
        authgem_vertex=bool(needs_authgem and needs_vertex),
        authgpt_pool=pool_route(model.strip().lower()),
        authgrok_pool=authgrok_pool_route_requested(model, config, hint=hints.get('authgrok_pool', False)),
        account_ids={provider: tuple(sorted(slots)) for provider, slots in ids.items()},
        needs_api_key=_needs_api_key(model) if api_key_check else None,
    )


# =============================================================================================
# 2. GUI-free rule owner: the shared mixin handlers run on config-seeded state
# =============================================================================================

class _Recorder:
    """Stand-in widget for the handlers' ``hasattr``-guarded widget syncs (records state)."""

    def __init__(self, text='', checked=False, data=None):
        self._text = text
        self._checked = checked
        self._data = data
        self.enabled = True
        self.visible = True
        self.style = ''

    def text(self):
        return self._text

    def setText(self, text):
        self._text = text

    def isChecked(self):
        return self._checked

    def setChecked(self, value):
        self._checked = bool(value)

    def currentData(self):
        return self._data

    def setEnabled(self, value):
        self.enabled = bool(value)

    def isEnabled(self):
        return self.enabled

    def setVisible(self, value):
        self.visible = bool(value)

    def setStyleSheet(self, style):
        self.style = style

    def blockSignals(self, value):
        return False


_OWNER_CLASS = None
_OWNER_LOCK = threading.Lock()
_VAR_INDEX = None


def _rule_owner_class():
    """``_RuleOwner(ConfigStateMixin, RunEnvMixin)``: the U2 mixins with GUI-free hooks."""
    global _OWNER_CLASS
    with _OWNER_LOCK:
        if _OWNER_CLASS is None:
            from owner_state import ConfigStateMixin
            from run_env import RunEnvMixin

            class _RuleOwner(ConfigStateMixin, RunEnvMixin):
                """GUI-free owner the shared handlers run on: no widgets unless a rule adds
                recorders, ``save_config`` is a no-op (callers persist), logs are dropped."""

                def __init__(self, config, **attrs):
                    self.config = config
                    for name, value in attrs.items():
                        setattr(self, name, value)

                def save_config(self, show_message=False):
                    return True

                def append_log(self, message):
                    return None

            _OWNER_CLASS = _RuleOwner
        return _OWNER_CLASS


def _var_index():
    """``{var_name: (config key, _init_variables default)}`` from the settings schema."""
    global _VAR_INDEX
    if _VAR_INDEX is None:
        import settings_schema

        index = {}
        for spec in settings_schema.all_specs():
            default = spec.init_default
            if default is settings_schema.MISSING:
                default = spec.default
            for var in spec.var_names:
                index.setdefault(var, (spec.key, default))
        _VAR_INDEX = index
    return _VAR_INDEX


def _config_var(config, var_name):
    """The value desktop/HeadlessOwner start-up gives ``self.<var_name>`` for *config*."""
    key, default = _var_index()[var_name]
    return config.get(key, default)


def _owner(config, var_names=(), **attrs):
    seeded = {name: _config_var(config, name) for name in var_names}
    seeded.update(attrs)
    return _rule_owner_class()(config, **seeded)


class _PrivateEnvOs:
    """``os`` for an isolated handler run: every attribute is the real one except ``environ``."""

    def __init__(self, environ):
        self.environ = environ

    def __getattr__(self, name):
        return getattr(os, name)


def _run_isolated(func, owner, *args, **kwargs):
    """Run the shared handler *func* on *owner* with a private ``os.environ`` and no stdout.

    The handler's code object runs with a copy of its module globals in which ``os.environ``
    is a dict, so its environment exports never reach the process (a mobile UI thread must
    not write the env a running job reads), and ``print`` is silent (a running job may be
    capturing stdout into its log). Returns ``(result, {exported env})``.
    """
    environ: Dict[str, str] = {}
    func = getattr(func, '__func__', func)
    globals_ = dict(func.__globals__)
    globals_['os'] = _PrivateEnvOs(environ)
    globals_['print'] = _silent_print
    isolated = types.FunctionType(func.__code__, globals_, func.__name__, func.__defaults__, func.__closure__)
    isolated.__kwdefaults__ = func.__kwdefaults__
    result = isolated(owner, *args, **kwargs)
    return result, environ


def _silent_print(*_args, **_kwargs):
    return None


def _mixin_function(name):
    from owner_state import ConfigStateMixin
    from run_env import RunEnvMixin

    for cls in (ConfigStateMixin, RunEnvMixin):
        if name in vars(cls):
            return vars(cls)[name]
    raise AttributeError(name)


# =============================================================================================
# 3. Main-window Chunk Size and the compression factor
# =============================================================================================

def chunk_budget_margin():
    """RunEnvMixin._CHUNK_BUDGET_SAFETY_MARGIN (tokens kept free of the output limit)."""
    from run_env import RunEnvMixin

    return RunEnvMixin._CHUNK_BUDGET_SAFETY_MARGIN


def parse_chunk_size_text(text):
    """Chunk Size field text -> None for "Auto"/blank (auto compression), else an int (0 = invalid).

    From TranslatorGUI._on_chunk_size_edited.
    """
    text = text.strip().replace(',', '')
    if not text or text.lower() == 'auto':
        return None
    try:
        chunk_size = int(float(text))
    except (TypeError, ValueError):
        chunk_size = 0
    return chunk_size


def factor_for_chunk_size(output_tokens, chunk_size, margin=None):
    """Compression factor whose max input chunk budget equals *chunk_size* (None if impossible).

    From TranslatorGUI._apply_chunk_size: ``available = output_tokens - margin``; nothing fits
    for a non-positive chunk size or no room.
    """
    if margin is None:
        margin = chunk_budget_margin()
    available = output_tokens - margin
    if chunk_size <= 0 or available <= 0:
        return None

    # Round the factor down so int(available / factor) lands on the chunk size.
    factor = max(0.000001, math.floor(available / chunk_size * 1_000_000) / 1_000_000)
    return factor


def record_chunk_size(config, chunk_size, factor):
    """Persist a manual chunk size and the factor that produces it (TranslatorGUI._apply_chunk_size)."""
    config['compression_factor'] = factor
    config['manual_chunk_size'] = chunk_size


def remember_manual_chunk_size(config, chunk_budget):
    """Record the current budget as the chunk size to hold when the output limit changes.

    *chunk_budget* is a zero-argument callable (desktop: ``self._compression_chunk_budget``).
    From TranslatorGUI._remember_manual_chunk_size.
    """
    if config.get('auto_compression_factor', True):
        return
    budget = chunk_budget()
    if budget is not None:
        config['manual_chunk_size'] = budget


def held_manual_chunk_size(config):
    """The manual chunk size to re-apply after an output-limit change, or None (auto / unset).

    From TranslatorGUI._hold_manual_chunk_size.
    """
    if config.get('auto_compression_factor', True):
        return None
    return config.get('manual_chunk_size')


def auto_compression_factor(max_output_tokens):
    """The auto compression factor for an output token limit (None when the limit is invalid).

    Runs ConfigStateMixin._update_auto_compression_factor (the desktop's threshold table).
    """
    owner = _owner({'auto_compression_factor': True}, max_output_tokens=max_output_tokens)
    _run_isolated(_mixin_function('_update_auto_compression_factor'), owner)
    return owner.config.get('compression_factor')


def apply_auto_compression_factor(config, max_output_tokens=None):
    """When auto compression is on, set ``config['compression_factor']`` for the output limit.

    ConfigStateMixin._update_auto_compression_factor on *config* (in place). Returns the factor
    in effect afterwards.
    """
    attrs = {}
    if max_output_tokens is not None:
        attrs['max_output_tokens'] = max_output_tokens
    owner = _owner(config, ('max_output_tokens', 'compression_factor_var'), **attrs)
    _run_isolated(_mixin_function('_update_auto_compression_factor'), owner)
    return owner.config.get('compression_factor', owner.compression_factor_var)


def chunk_budget(max_output_tokens, compression_factor):
    """Max input chunk budget for an output limit and factor (RunEnvMixin._compression_chunk_budget)."""
    owner = _owner({}, max_output_tokens=max_output_tokens, compression_factor_var=compression_factor)
    return owner._compression_chunk_budget()


def chunk_budget_for_config(config):
    """RunEnvMixin._compression_chunk_budget with the owner state *config* starts with."""
    return _owner(config, ('max_output_tokens', 'compression_factor_var'))._compression_chunk_budget()


def chunk_size_display(config):
    """What the main-window Chunk Size field shows: "Auto" or the budget ("12,345")."""
    entry = _Recorder(text=None)
    owner = _owner(config, ('max_output_tokens', 'compression_factor_var'), chunk_size_entry=entry)
    owner._sync_chunk_size_entry()
    return entry.text()


def apply_chunk_size(config, chunk_size, max_output_tokens=None):
    """Set the factor so the max input chunk budget equals *chunk_size* (TranslatorGUI._apply_chunk_size).

    Returns the factor, or None when the size does not fit (config untouched).
    """
    try:
        output_tokens = int(max_output_tokens if max_output_tokens is not None
                            else _config_var(config, 'max_output_tokens'))
        chunk_size = int(chunk_size)
    except (TypeError, ValueError):
        return None
    factor = factor_for_chunk_size(output_tokens, chunk_size)
    if factor is None:
        return None
    record_chunk_size(config, chunk_size, factor)
    return factor


def set_chunk_size_text(config, text, max_output_tokens=None):
    """The Chunk Size field edit (TranslatorGUI._on_chunk_size_edited) on *config*.

    "Auto"/blank turns auto compression on (and recomputes the factor); a number fixes the
    factor to that budget. Returns the text the field shows afterwards, or None when the
    number was rejected (config untouched).
    """
    chunk_size = parse_chunk_size_text(text)
    if chunk_size is None:
        config['auto_compression_factor'] = True
        apply_auto_compression_factor(config, max_output_tokens)
        return chunk_size_display(config)
    if apply_chunk_size(config, chunk_size, max_output_tokens) is None:
        return None
    config['auto_compression_factor'] = False
    config['manual_chunk_size'] = chunk_size
    return chunk_size_display(config)


def hold_manual_chunk_size(config, max_output_tokens=None):
    """After an output-limit change keep the manual chunk size (TranslatorGUI._hold_manual_chunk_size)."""
    chunk_size = held_manual_chunk_size(config)
    if chunk_size is not None:
        return apply_chunk_size(config, chunk_size, max_output_tokens)
    return None


#: Why the manual compression factor is locked while Auto Compression Factor is on (the Other
#: Settings ``_on_auto_compression_toggle`` disables the field; the main window recomputes it from
#: the output token limit, ConfigStateMixin._update_auto_compression_factor).
COMPRESSION_LOCK_REASON = ("Auto compression factor is on: the factor follows the output token limit "
                           "(<16379: 1.5 | <32769: 2.0 | <65536: 2.5 | ≥65536: 3.0).")


def compression_lock(config):
    """``{'compression_factor': LockInfo}``: locked while ``auto_compression_factor`` is on."""
    if _flag(config, 'auto_compression_factor'):
        return {'compression_factor': LockInfo(locked=True, reason=COMPRESSION_LOCK_REASON)}
    return {'compression_factor': LockInfo(locked=False)}


# Glossary Manager › Balanced/Full Extraction Settings: "Compression Factor" + "Auto" (the
# ``_update_glossary_compression`` closure of GlossaryManager_GUI._setup_manual_glossary_tab,
# split in two so the closure keeps its widget work in the same order).

#: Why the glossary compression factor is locked while its Auto box is ticked.
GLOSSARY_COMPRESSION_LOCK_REASON = ("Auto is on: the glossary compression factor follows the glossary output "
                                    "token limit (-1 = the main output limit): <16379: 1.0 | <32769: 1.2 | "
                                    "<65536: 1.4 | ≥65536: 1.5.")


def glossary_output_limit(limit_text, max_output_tokens=65536):
    """``(actual_limit, helper_text)`` for the Glossary output token limit field's text.

    Text that is not an integer counts as 65536; -1 resolves to the main output token limit
    (*max_output_tokens*) and the helper label reads "(Auto: N)", otherwise it is empty.
    """
    # Update helper label for token limit
    try:
        limit_val = int(limit_text)
    except ValueError:
        limit_val = 65536

    # Resolve -1 to actual max_output_tokens
    if limit_val == -1:
        actual_limit = max_output_tokens
        return actual_limit, f"(Auto: {actual_limit})"
    actual_limit = limit_val
    return actual_limit, ""


def glossary_auto_compression_factor(actual_limit):
    """The Auto glossary compression factor for a resolved glossary output limit."""
    # Logic: 1.0 | 1.2 | 1.4 | 1.5
    if actual_limit < 16379:
        factor = 1.0
    elif actual_limit < 32769:
        factor = 1.2
    elif actual_limit < 65536:
        factor = 1.4
    else:
        factor = 1.5
    return factor


def apply_glossary_auto_compression(config, max_output_tokens=None):
    """With ``glossary_auto_compression`` on (default True), set ``config['glossary_compression_factor']``
    from ``glossary_max_output_tokens`` (-1: the main output limit) like the Glossary Manager's Auto
    box (its save stores ``float(entry.text())``). Returns the factor, or None when Auto is off
    (*config* untouched)."""
    if not _flag(config, 'glossary_auto_compression'):  # default on (the Auto box starts ticked)
        return None
    if max_output_tokens is None:
        max_output_tokens = _config_var(config, 'max_output_tokens')
    try:
        max_output_tokens = int(max_output_tokens)
    except (TypeError, ValueError):
        max_output_tokens = 65536
    actual_limit, _helper = glossary_output_limit(str(config.get('glossary_max_output_tokens', -1)),
                                                  max_output_tokens)
    factor = float(str(glossary_auto_compression_factor(actual_limit)))
    config['glossary_compression_factor'] = factor
    return factor


def glossary_compression_lock(config):
    """``{'glossary_compression_factor': LockInfo}``: locked while the glossary Auto box is on."""
    if _flag(config, 'glossary_auto_compression'):
        return {'glossary_compression_factor': LockInfo(locked=True, reason=GLOSSARY_COMPRESSION_LOCK_REASON)}
    return {'glossary_compression_factor': LockInfo(locked=False)}


# =============================================================================================
# 4. Context mode and batching (ConfigStateMixin handlers)
# =============================================================================================

#: The Other Settings batching radios -> batching mode (ConfigStateMixin._set_batching_mode).
BATCHING_RADIOS = {
    'batch_conservative_radio': 'conservative',
    'batch_direct_radio': 'direct',
    'batch_no_batching_radio': 'aggressive',
}

_CONTEXT_VARS = ('rolling_summary_var', 'rolling_summary_mode_var', 'contextual_var', 'batch_mode_var')


@dataclass(frozen=True)
class ChoiceState:
    enabled: bool
    locked: bool          # shown with 🔒: required by the context mode


@dataclass(frozen=True)
class ContextBatching:
    context_mode: str
    batching_mode: str
    last_batched_mode: Optional[str]
    choices: Mapping[str, ChoiceState]
    status: str

    @property
    def allowed(self) -> Tuple[str, ...]:
        return tuple(mode for mode, state in self.choices.items() if state.enabled)


def context_mode(config):
    """The Context Mode the flags in *config* represent (RunEnvMixin._context_mode_from_flags)."""
    return _owner(config, _CONTEXT_VARS)._context_mode_from_flags()


def _batching_owner(config, mode, last_batched_mode):
    radios = {attr: _Recorder() for attr in BATCHING_RADIOS}
    attrs = dict(radios)
    attrs['batch_context_lock_label'] = _Recorder()
    if mode is not None:
        attrs['context_mode_var'] = mode
    if last_batched_mode is not None:
        attrs['_context_last_batched_mode'] = last_batched_mode
    return _owner(config, _CONTEXT_VARS, **attrs)


def _batching_result(owner):
    choices = {}
    for attr, mode in BATCHING_RADIOS.items():
        radio = getattr(owner, attr)
        choices[mode] = ChoiceState(enabled=radio.enabled, locked=str(radio.text() or '').startswith('🔒'))
    label = getattr(owner, 'batch_context_lock_label')
    return ContextBatching(
        context_mode=getattr(owner, 'context_mode_var', None) or owner._context_mode_from_flags(),
        batching_mode=owner.batch_mode_var,
        last_batched_mode=getattr(owner, '_context_last_batched_mode', None),
        choices=choices,
        status=str(label.text() or ''),
    )


def enforce_context_batching(config, *, context_mode=None, last_batched_mode=None):
    """Apply the context mode's batching constraint to *config* (in place).

    ConfigStateMixin._enforce_context_batching_mode: Context Mode Off forces No batching
    (remembering a direct/conservative choice); a context mode restores the remembered
    batched mode when No batching was set. *context_mode* overrides the mode the config flags
    represent; *last_batched_mode* is the remembered choice (default: direct).
    """
    owner = _batching_owner(config, context_mode, last_batched_mode)
    owner._enforce_context_batching_mode()
    return _batching_result(owner)


def context_batching_controls(config, *, context_mode=None):
    """The batching choices (enabled / 🔒 locked) and status line for a context mode (read-only)."""
    owner = _batching_owner(copy.deepcopy(dict(config)), context_mode, None)
    owner._refresh_context_batching_controls()
    return _batching_result(owner)


def apply_context_mode(config, mode, *, last_batched_mode=None):
    """Select a Context Mode (ConfigStateMixin._on_context_mode_changed) on *config* (in place).

    Writes contextual / use_rolling_summary / rolling_summary_mode and enforces batching.
    """
    owner = _batching_owner(config, None, last_batched_mode)
    owner.context_mode_combo = _Recorder(data=mode)
    owner._on_context_mode_changed()
    return _batching_result(owner)


# =============================================================================================
# 5. Auto-glossary mode (ConfigStateMixin._on_auto_glossary_shortcut_changed)
# =============================================================================================

#: Config keys the glossary-mode shortcut handler writes.
AUTO_GLOSSARY_MODE_KEYS = ('auto_glossary_mode', 'enable_auto_glossary', 'append_glossary',
                           'append_glossary_auto_load', 'fuzzy_auto_mapping')

_MODES = None


def _shortcut_handler(owner, index):
    handler = _mixin_function('_on_auto_glossary_shortcut_changed')
    return _run_isolated(handler, owner, index)


def auto_glossary_modes():
    """The glossary modes in shortcut-combo order (read from the desktop handler's own map)."""
    global _MODES
    if _MODES is None:
        modes = []
        for index in range(64):
            owner = _owner({})
            _shortcut_handler(owner, index)
            mode = owner.config.get('auto_glossary_mode')
            if index and mode == 'off':  # the handler's fallback: past the last mode
                break
            modes.append(mode)
        _MODES = tuple(modes)
    return _MODES


def auto_glossary_mode_index(mode):
    """Shortcut-combo index of *mode* (ConfigStateMixin._saved_auto_glossary_shortcut_index)."""
    return _owner({'auto_glossary_mode': mode})._saved_auto_glossary_shortcut_index()


def auto_glossary_mode_flags(mode):
    """What selecting *mode* writes: ``{key: value}`` for AUTO_GLOSSARY_MODE_KEYS, None = untouched.

    Off / Off (Fuzzy Mapping) / Balanced / Full / Single Pass turn Auto-Mapping on, Manual
    Glossary Only / Minimal turn it off; only Off (Fuzzy Mapping) enables fuzzy mapping; every
    mode but No Glossary turns Append Glossary on. The desktop start-up force-sync (the
    ``__init__`` config block) applies the same Auto-Mapping / fuzzy values.
    """
    owner = _owner({})
    _shortcut_handler(owner, auto_glossary_mode_index(mode))
    return {key: owner.config.get(key) for key in AUTO_GLOSSARY_MODE_KEYS}


def apply_auto_glossary_mode(config, mode):
    """Select a glossary mode on *config* (in place), exactly as the desktop shortcut combo does.

    Returns the keys whose value changed (``{key: new value}``). Unknown modes select Off,
    as on desktop.
    """
    before = {key: config.get(key, _ABSENT) for key in AUTO_GLOSSARY_MODE_KEYS}
    owner = _owner(config)
    _shortcut_handler(owner, auto_glossary_mode_index(mode))
    return {key: config[key] for key in AUTO_GLOSSARY_MODE_KEYS
            if key in config and (before[key] is _ABSENT or before[key] != config[key])}


class _Absent:
    def __repr__(self):
        return '<absent>'


_ABSENT = _Absent()


# =============================================================================================
# 6. Target language (ConfigStateMixin.update_target_language)
# =============================================================================================

def fan_out_target_language(config, language):
    """Set the target language everywhere the desktop combo does, on *config* (in place).

    ConfigStateMixin.update_target_language: output_language, glossary_target_language, the
    manga manual-edit language and the AI Hunter language-detection target. The handler's
    environment exports are returned instead of being written (``{'OUTPUT_LANGUAGE': ...,
    'GLOSSARY_TARGET_LANGUAGE': ...}``): a job exports them from its own config.
    """
    owner = _owner(config)
    _result, env = _run_isolated(_mixin_function('update_target_language'), owner, language)
    return env


# =============================================================================================
# 7. Locks: {key: LockInfo} for settings UIs
# =============================================================================================

@dataclass(frozen=True)
class LockInfo:
    """A setting the current configuration constrains.

    ``locked``: the value is dictated by another setting (desktop shows 🔒 / disables it);
    ``allowed``: the choices still selectable (None = any); ``value``: the forced value when
    there is exactly one; ``reason``: the desktop's explanation.
    """

    locked: bool
    reason: str = ''
    allowed: Optional[Tuple[Any, ...]] = None
    value: Any = None


_LOCK_RULES: Dict[str, Tuple[Tuple[str, ...], Callable[[Mapping[str, Any]], Mapping[str, LockInfo]]]] = {}


def register_lock_rule(name, keys, rule):
    """Register ``rule(config) -> {key: LockInfo}`` for *keys* under *name* (re-registering replaces)."""
    _LOCK_RULES[name] = (tuple(keys), rule)


def lock_rules():
    """``{name: keys}`` of the registered lock rules."""
    return {name: keys for name, (keys, _rule) in _LOCK_RULES.items()}


def evaluate_locks(config, keys=None):
    """``{key: LockInfo}`` for every registered rule (only *keys* when given). Never mutates *config*."""
    wanted = set(keys) if keys is not None else None
    out: Dict[str, LockInfo] = {}
    for _name, (rule_keys, rule) in _LOCK_RULES.items():
        if wanted is not None and not wanted.intersection(rule_keys):
            continue
        for key, info in (rule(config) or {}).items():
            if wanted is None or key in wanted:
                out[key] = info
    return out


def _batching_lock(config):
    state = context_batching_controls(config)
    allowed = state.allowed
    return {'batching_mode': LockInfo(
        locked=any(choice.locked for choice in state.choices.values()),
        reason=state.status.replace('ℹ️ ', '').replace('\n   ', ' '),
        allowed=allowed,
        value=allowed[0] if len(allowed) == 1 else None,
    )}


def _temperature_lock(config):
    entry = _Recorder()
    owner = _owner(copy.deepcopy(dict(config)), ('disable_temperature_var',), trans_temp=entry)
    _run_isolated(_mixin_function('_on_disable_temperature_toggle'), owner)
    if entry.enabled:
        return {'translation_temperature': LockInfo(locked=False)}
    return {'translation_temperature': LockInfo(
        locked=True,
        reason="Disable temperature is on: requests omit the temperature parameter and use the "
               "provider or model default.",
    )}


register_lock_rule('context_batching', ('batching_mode',), _batching_lock)
register_lock_rule('disable_temperature', ('translation_temperature',), _temperature_lock)


# =============================================================================================
# 8. Shared lookups for the dialog rules below
# =============================================================================================

def _setting(config, key):
    """The value desktop start-up gives the setting *key* for *config* (init default when unset)."""
    if key in config:
        return config[key]
    import settings_schema

    spec = settings_schema.spec(key)
    if spec.init_default is not settings_schema.MISSING and not settings_schema._is_marker(spec.init_default):
        return spec.init_default
    return settings_schema.effective_default(key)


def _flag(config, key):
    value = _setting(config, key)
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


# =============================================================================================
# 9. Thinking: stream thinking logs lock "Enable thoughts" (other_settings)
# =============================================================================================

#: Config keys other_settings._sync_thoughts_lock_state writes.
THOUGHTS_LOCK_KEYS = ('enable_thoughts',)
THOUGHTS_LOCK_REASON = ("Stream thinking/reasoning logs is on, so thoughts stay enabled "
                        "(turn streaming of thinking logs off to change this).")


def thoughts_lock_state(stream_thinking_on):
    """``(enable_thoughts, ENABLE_THOUGHTS, locked)`` for a stream-thinking-logs state.

    other_settings._sync_thoughts_lock_state: stream thinking ON force-checks "Enable
    thoughts" and locks it (purple, 🔒, non-interactive); turning it OFF unlocks the toggle
    and also unchecks thoughts.
    """
    if stream_thinking_on:
        # Force-check and lock
        return True, '1', True
    # Unlock - also uncheck thoughts when stream thinking is turned off
    return False, '0', False


def apply_thoughts_lock(config, stream_thinking_on):
    """The stream-thinking toggle's effect on *config* (in place); returns its env export."""
    enabled, env_value, _locked = thoughts_lock_state(stream_thinking_on)
    config['enable_thoughts'] = enabled
    return {'ENABLE_THOUGHTS': env_value}


def thoughts_lock(config):
    """``{'enable_thoughts': LockInfo}``: locked on while stream thinking logs are on.

    The Other Settings dialog applies the lock when it opens with stream thinking on.
    """
    if _flag(config, 'stream_thinking_logs'):
        enabled, _env, _locked = thoughts_lock_state(True)
        return {'enable_thoughts': LockInfo(locked=True, reason=THOUGHTS_LOCK_REASON, allowed=(enabled,),
                                            value=enabled)}
    return {'enable_thoughts': LockInfo(locked=False)}


# =============================================================================================
# 9b. Streaming (other_settings "Real-time Translation (Streaming)"; the mobile Streaming switch)
# =============================================================================================

#: The amber note under "Enable streaming responses" (other_settings._create_response_handling_section).
STREAMING_TRUNCATION_WARNING = "⚠️ Enabling this may result in silent truncation"
#: The note under the forced-stream batch log toggle (same dialog).
FORCED_STREAM_NOTE = ("🔐 AuthGPT, AuthGrok, AuthGem, AuthCD, Arena, Antigravity, and OcAgy always stream "
                      "— this controls batch log visibility")
#: The group's four toggles: config key -> the env name a run exports (run_env, translation_pipeline).
STREAMING_ENV = {
    'enable_streaming': 'ENABLE_STREAMING',
    'stream_thinking_logs': 'STREAM_THINKING_LOGS',
    'allow_batch_stream_logs': 'ALLOW_BATCH_STREAM_LOGS',
    'allow_authgpt_batch_stream_logs': 'ALLOW_AUTHGPT_BATCH_STREAM_LOGS',
}
STREAMING_KEYS = tuple(STREAMING_ENV)


def streaming_states(config, unset=False):
    """``{key: bool}`` of the four streaming toggles as a run reads them (``bool(config.get(key,
    unset))``, run_env / translation_pipeline); *unset* is what an absent key means (desktop: False)."""
    return {key: bool(config.get(key, unset)) for key in STREAMING_KEYS}


def streaming_mode(config, unset=False):
    """'on' (all four toggles on), 'off' (all off) or 'custom' (they differ)."""
    states = streaming_states(config, unset)
    if all(states.values()):
        return 'on'
    if not any(states.values()):
        return 'off'
    return 'custom'


def apply_streaming(config, on, unset=False):
    """Every streaming toggle to *on* (in place), each the way its desktop checkbox writes it; returns
    the env exports.

    Only the toggles whose value changes are applied: the checkboxes fire on a real change only, and
    stream thinking logs runs the thoughts lock (``_change_stream_thinking``: OFF also unchecks Enable
    thoughts) even when its own value does not change. Turning on always applies stream thinking, so
    the lock is re-applied the way the dialog does when it opens with stream thinking on (that never
    unchecks thoughts). *unset*: what an absent key means (desktop False).
    """
    on = bool(on)
    states = streaming_states(config, unset)
    env = {}
    for key in STREAMING_KEYS:
        if key == 'stream_thinking_logs':
            if on or states[key]:
                env.update(_change_stream_thinking(config, on))
        elif states[key] != on:
            config[key] = on
            env[STREAMING_ENV[key]] = '1' if on else '0'
    return env


# =============================================================================================
# 10. Output mode (other_settings._set_output_mode and the Image Translation sub-settings)
# =============================================================================================

#: Output modes in radio / main dropdown order (index = main-window dropdown index).
OUTPUT_MODES = ('text', 'vision', 'image', 'video', 'audio', 'refinement')
OUTPUT_MODE_INDEX = {'text': 0, 'vision': 1, 'image': 2, 'video': 3, 'audio': 4, 'refinement': 5}


@dataclass(frozen=True)
class OutputModeFlags:
    """What selecting an output mode writes, in the desktop order.

    ``var_values``: TranslatorGUI attributes; ``config_values``: config keys; ``env``: the
    variables the setter exports immediately so they never stay stale.
    """

    mode: str
    var_values: Mapping[str, Any]
    config_values: Mapping[str, Any]
    env: Mapping[str, str]


def normalize_output_mode(mode):
    """other_settings._set_output_mode's normalisation ('refine' -> 'refinement', unknown -> 'text').

    Raises like the desktop setter for a non-string mode (its caller catches that).
    """
    mode = mode.lower().strip()
    if mode == 'refine':
        mode = 'refinement'
    if mode not in ('text', 'vision', 'image', 'video', 'audio', 'refinement'):
        mode = 'text'
    return mode


def output_mode_flags(mode, config=None):
    """Text = image translation OFF; Vision / Image / Video = image translation ON (Image and
    Video also generate output); Audio = TTS from existing output; Refinement = improve
    existing output. *config* is accepted for symmetry (the flags depend on the mode only).
    """
    mode = normalize_output_mode(mode)

    # --- set legacy booleans ---
    image = (mode == 'image')
    video = (mode == 'video')
    audio = (mode == 'audio')
    refinement = (mode == 'refinement')
    image_translation = mode in ('vision', 'image', 'video')
    var_values = {
        'enable_image_output_mode_var': image,
        'enable_video_output_mode_var': video,
        'enable_audio_output_mode_var': audio,
        'enable_refinement_output_mode_var': refinement,
        'enable_image_translation_var': image_translation,
        'output_mode_var': mode,
    }
    config_values = {
        'enable_image_output_mode': image,
        'enable_video_output_mode': video,
        'enable_audio_output_mode': audio,
        'enable_refinement_output_mode': refinement,
        'enable_image_translation': image_translation,
        'output_mode': mode,
    }
    # --- immediately sync env vars so they never stay stale ---
    env = {
        'ENABLE_IMAGE_OUTPUT_MODE': '1' if image else '0',
        'ENABLE_VIDEO_OUTPUT_MODE': '1' if video else '0',
        'ENABLE_AUDIO_OUTPUT_MODE': '1' if audio else '0',
        'ENABLE_REFINEMENT_OUTPUT_MODE': '1' if refinement else '0',
        'ENABLE_IMAGE_TRANSLATION': '1' if image_translation else '0',
        'OUTPUT_MODE': mode,
    }
    return OutputModeFlags(mode, var_values, config_values, env)


def apply_output_mode(config, mode):
    """Select an output mode on *config* (in place); returns the env exports (not written)."""
    flags = output_mode_flags(mode)
    config.update(flags.config_values)
    return dict(flags.env)


def current_output_mode(config):
    """The output mode *config* selects (RunEnvMixin._get_output_mode, legacy flags included)."""
    return _owner(config)._get_output_mode()


def output_mode_sub_settings(mode):
    """Which Image Translation sub-settings the mode shows (other_settings
    _update_output_mode_sub_settings): ``image`` (output resolution), ``video`` (duration,
    resolution), ``vision_request`` (Vision OCR prompt row and batch slots: Vision and Image),
    ``vision_only`` (skip translation, keep OCR image, the OCR prompt notes)."""
    return {
        'image': mode == 'image',
        'video': mode == 'video',
        'vision_request': mode in ('vision', 'image'),
        'vision_only': mode == 'vision',
    }


# =============================================================================================
# 11. Glossary Manager mode locks (GlossaryManager_GUI update_auto_glossary_state)
# =============================================================================================

#: The Glossary Manager mode combo's display text -> mode key (the old label "Off (No
#: Auto-Mapping)" is still accepted for saved configs / older builds).
GLOSSARY_MODE_DISPLAY = {
    'Off': 'off', 'Off (Fuzzy Mapping)': 'off_fuzzy_automap',
    'Manual Glossary Only': 'off_no_automap',
    'Off (No Auto-Mapping)': 'off_no_automap',
    'No Glossary': 'no_glossary', 'Minimal': 'minimal',
    'Balanced': 'balanced', 'Full': 'full',
    'Single Pass': 'single_pass',
}
#: Modes without automatic extraction (the glossary prompt editor is disabled).
GLOSSARY_MODES_WITHOUT_EXTRACTION = ('off', 'off_no_automap', 'no_glossary')
#: Modes that leave "Append Glossary to System Prompt" free (all others force it on).
GLOSSARY_APPEND_FREE_MODES = ('no_glossary',)
#: Modes that force Auto-Mapping (Auto-Fill) on / off.
GLOSSARY_AUTOMAP_ON_MODES = ('off', 'off_fuzzy_automap', 'balanced', 'full', 'single_pass')
GLOSSARY_AUTOMAP_OFF_MODES = ('off_no_automap', 'minimal', 'no_glossary')
#: The only mode with Fuzzy Auto-Mapping on (every other mode locks it off).
GLOSSARY_FUZZY_MODES = ('off_fuzzy_automap',)
#: Toggles the mode locks (config key -> GlossaryManager checkbox / var attributes).
GLOSSARY_MODE_TOGGLES = {
    'append_glossary': ('append_glossary_checkbox', 'append_glossary_var'),
    'append_glossary_auto_load': ('append_glossary_auto_load_checkbox', 'append_glossary_auto_load_var'),
    'fuzzy_auto_mapping': ('fuzzy_auto_mapping_checkbox', 'fuzzy_auto_mapping_var'),
}


def glossary_mode_from_display(mode_raw):
    """Map display text to internal mode key (unknown text: lower-cased, spaces -> '_')."""
    return GLOSSARY_MODE_DISPLAY.get(mode_raw, mode_raw.lower().replace(' ', '_'))


def glossary_mode_label(mode):
    """Display label of a mode key (the current combo text), else the key itself."""
    for label, key in GLOSSARY_MODE_DISPLAY.items():
        if key == mode:
            return label
    return str(mode)


def glossary_mode_extracts(mode):
    """Whether the mode runs automatic extraction (the glossary prompt editor is enabled)."""
    return mode not in GLOSSARY_MODES_WITHOUT_EXTRACTION


def glossary_mode_targeted_extraction(mode):
    """Targeted Extraction Settings only apply to Minimal mode."""
    return mode == 'minimal'


def glossary_manager_mode(config):
    """The glossary mode the Glossary Manager shows (and locks its toggles for) for *config*.

    Desktop start-up runs the main-window glossary-mode shortcut handler on the saved mode
    before the Glossary Manager can open (ConfigStateMixin._saved_auto_glossary_shortcut_index:
    a missing mode migrates from ``enable_auto_glossary``, an unknown one selects Off), so the
    tab's combo shows that mode.
    """
    modes = auto_glossary_modes()
    index = _owner(config)._saved_auto_glossary_shortcut_index()
    return modes[index] if 0 <= index < len(modes) else 'off'


@dataclass(frozen=True)
class ToggleStep:
    """One step of the mode lock pass, in the desktop order.

    ``locked`` True: force ``value`` and lock the toggle (🔒, purple); False: unlock it.
    ``write_config``: the pass also writes ``config[key]`` and the ``*_var`` attribute
    (otherwise the checkbox's own handler does when its state changes).
    """

    key: str
    locked: bool
    value: Optional[bool] = None
    write_config: bool = False


def glossary_mode_toggle_steps(mode):
    """The lock / unlock steps update_auto_glossary_state runs for *mode*, in order.

    Append Glossary is forced on (and locked) in every mode but No Glossary; Auto-Mapping is
    forced on in Off / Off (Fuzzy Mapping) / Balanced / Full / Single Pass, forced off in
    Manual Glossary Only / Minimal / No Glossary; Fuzzy Auto-Mapping is forced on only in
    Off (Fuzzy Mapping) and locked off everywhere else; No Glossary locks Append Glossary and
    Auto-Mapping off. A later step for the same toggle wins.
    """
    steps = []
    # Auto-enable & lock append glossary when off/minimal/balanced/full is selected
    if mode not in GLOSSARY_APPEND_FREE_MODES:
        steps.append(ToggleStep('append_glossary', True, True))
    else:
        steps.append(ToggleStep('append_glossary', False))

    # Auto-enable & lock auto-map when off/off_fuzzy_automap/balanced/full is selected
    if mode in GLOSSARY_AUTOMAP_ON_MODES:
        steps.append(ToggleStep('append_glossary_auto_load', True, True))
    elif mode not in GLOSSARY_AUTOMAP_OFF_MODES:
        steps.append(ToggleStep('append_glossary_auto_load', False))

    # Fuzzy auto-mapping lock logic
    if mode in GLOSSARY_FUZZY_MODES:
        steps.append(ToggleStep('fuzzy_auto_mapping', True, True, write_config=True))
    else:
        # All other modes: lock fuzzy OFF
        steps.append(ToggleStep('fuzzy_auto_mapping', True, False, write_config=True))

    # "No Glossary" - lock all three toggles OFF
    if mode == 'no_glossary':
        steps.append(ToggleStep('append_glossary', True, False))
        steps.append(ToggleStep('append_glossary_auto_load', True, False, write_config=True))

    # "Manual Glossary Only" / "Minimal" - lock auto-mapping OFF
    if mode in ('off_no_automap', 'minimal'):
        steps.append(ToggleStep('append_glossary_auto_load', True, False, write_config=True))
    return tuple(steps)


def glossary_mode_toggle_states(mode):
    """``{key: (locked, value)}`` after the lock pass (value None: unlocked, unchanged)."""
    states = {}
    for step in glossary_mode_toggle_steps(mode):
        states[step.key] = (step.locked, step.value if step.locked else None)
    return states


def _glossary_lock_reason(mode, key, value):
    label = glossary_mode_label(mode)
    if key == 'append_glossary':
        if value:
            return f"Glossary mode '{label}' always appends the glossary to the system prompt."
        return f"Glossary mode '{label}' runs without a glossary, so nothing is appended."
    if key == 'append_glossary_auto_load':
        if value:
            return f"Glossary mode '{label}' turns Auto-Mapping (Auto-Fill) on."
        return f"Glossary mode '{label}' turns Auto-Mapping (Auto-Fill) off."
    if value:
        return f"Glossary mode '{label}' turns Fuzzy Auto-Mapping on."
    return "Fuzzy Auto-Mapping is only available in the 'Off (Fuzzy Mapping)' glossary mode."


def glossary_mode_locks(mode):
    """``{key: LockInfo}`` for the toggles the Glossary Manager mode locks (🔒).

    Includes ``fuzzy_auto_mapping_threshold``: the similarity slider is locked whenever Fuzzy
    Auto-Mapping is locked off.
    """
    out = {}
    for key, (locked, value) in glossary_mode_toggle_states(mode).items():
        if not locked:
            out[key] = LockInfo(locked=False)
            continue
        out[key] = LockInfo(locked=True, reason=_glossary_lock_reason(mode, key, value), allowed=(value,),
                            value=value)
    fuzzy = out.get('fuzzy_auto_mapping')
    if fuzzy is not None and fuzzy.locked and fuzzy.value is False:
        out['fuzzy_auto_mapping_threshold'] = LockInfo(locked=True, reason=fuzzy.reason)
    else:
        out['fuzzy_auto_mapping_threshold'] = LockInfo(locked=False)
    return out


def apply_glossary_mode_locks(config, mode=None):
    """Apply the mode lock pass to *config* (in place): every locked toggle takes its forced
    value (desktop: the checkbox change writes the setting). Returns ``{key: value}`` changed."""
    if mode is None:
        mode = glossary_manager_mode(config)
    changed = {}
    for key, (locked, value) in glossary_mode_toggle_states(mode).items():
        if locked and config.get(key, _ABSENT) != value:
            config[key] = value
            changed[key] = value
    return changed


# =============================================================================================
# 12. Visibility rules and the rule-id evaluator (settings_schema visible_if / locked_if)
# =============================================================================================

_VISIBILITY_RULES: Dict[str, Callable[[Mapping[str, Any]], bool]] = {}


def register_visibility_rule(rule_id, predicate):
    """Register ``predicate(config) -> bool`` (True: the setting is used) under *rule_id*."""
    _VISIBILITY_RULES[rule_id] = predicate


def visibility_rules():
    """The registered visibility rule ids."""
    return tuple(_VISIBILITY_RULES)


def is_visible(rule_id, config):
    """Whether the settings a visibility rule guards are used with *config*."""
    return bool(_VISIBILITY_RULES[rule_id](config))


LOCK_RULE_PREFIX = 'lock:'


def lock_rule_id(key):
    """The ``locked_if`` rule id of a setting the registered lock rules cover."""
    return LOCK_RULE_PREFIX + key


def lock_keys():
    """Every setting key a registered lock rule may lock."""
    keys = []
    for rule_keys in lock_rules().values():
        for key in rule_keys:
            if key not in keys:
                keys.append(key)
    return tuple(keys)


def evaluate(rule_id, config):
    """Evaluate a settings_schema rule id against *config* (never mutates it).

    ``lock:<key>``: the lock reason (a non-empty string) when *key* is locked, else ''.
    Visibility rule ids: True when the guarded settings are used, False when the desktop
    disables / hides them for this configuration. Unknown ids raise KeyError.
    """
    rule_id = str(rule_id)
    if rule_id.startswith(LOCK_RULE_PREFIX):
        key = rule_id[len(LOCK_RULE_PREFIX):]
        if key not in lock_keys():
            raise KeyError(f"no lock rule covers {key!r}")
        info = evaluate_locks(config, keys=[key]).get(key)
        if info is None or not info.locked:
            return ''
        return info.reason or 'Locked by another setting.'
    if rule_id not in _VISIBILITY_RULES:
        raise KeyError(f"unknown settings rule {rule_id!r}")
    return is_visible(rule_id, config)


evaluate_rule = evaluate


#: Settings a setting change implies (``apply_change``): key -> handler(config, value) -> env.
_CHANGE_RULES: Dict[str, Callable[[Dict[str, Any], Any], Mapping[str, str]]] = {}


def _change_stream_thinking(config, value):
    config['stream_thinking_logs'] = bool(value)
    env = {'STREAM_THINKING_LOGS': '1' if value else '0'}
    env.update(apply_thoughts_lock(config, value))
    return env


def _change_output_mode(config, value):
    return apply_output_mode(config, value)


def _change_glossary_mode(config, value):
    # the main-window glossary-mode shortcut (the Glossary Manager pages apply their own
    # lock pass with apply_glossary_mode_locks when they open)
    apply_auto_glossary_mode(config, value)
    return {}


def _change_target_language(config, value):
    # the main-window Target Language combo (ConfigStateMixin.update_target_language)
    return fan_out_target_language(config, value)


def _change_max_output_tokens(config, value):
    # the main-window Output Token Limit dialog: a manual chunk size stays fixed across limit
    # changes (captured first for settings saved before Chunk Size existed), then the auto
    # compression factor follows the new limit; the Glossary Manager's Auto glossary factor
    # resolves its -1 limit to the same value (mobile has no open dialog to refresh it later)
    if 'manual_chunk_size' not in config:
        remember_manual_chunk_size(config, lambda: chunk_budget_for_config(config))
    config['max_output_tokens'] = value
    hold_manual_chunk_size(config, value)
    apply_auto_compression_factor(config, value)
    apply_glossary_auto_compression(config, value)
    return {}


def _change_auto_compression(config, value):
    # other_settings _on_auto_compression_toggle: on recomputes the factor, off records the
    # current budget as the manual chunk size to hold
    config['auto_compression_factor'] = bool(value)
    if value:
        apply_auto_compression_factor(config)
    else:
        remember_manual_chunk_size(config, lambda: chunk_budget_for_config(config))
    return {}


def _change_glossary_compression_input(key):
    # the Glossary Manager's Auto box and output token limit field run _update_glossary_compression
    def change(config, value):
        config[key] = value
        apply_glossary_auto_compression(config)
        return {}
    return change


_CHANGE_RULES.update({
    'stream_thinking_logs': _change_stream_thinking,
    'output_mode': _change_output_mode,
    'auto_glossary_mode': _change_glossary_mode,
    'output_language': _change_target_language,
    'max_output_tokens': _change_max_output_tokens,
    'auto_compression_factor': _change_auto_compression,
    'glossary_auto_compression': _change_glossary_compression_input('glossary_auto_compression'),
    'glossary_max_output_tokens': _change_glossary_compression_input('glossary_max_output_tokens'),
})


def apply_change(config, key, value):
    """Set *key* to *value* on *config* with the desktop side effects of that control.

    stream_thinking_logs locks / releases Enable thoughts; output_mode writes the legacy mode
    flags; auto_glossary_mode runs the main-window shortcut handler; output_language fans the
    target language out like the main-window combo (glossary, manga manual edit, AI Hunter);
    max_output_tokens holds a manual chunk size and recomputes the auto compression factors;
    auto_compression_factor recomputes (on) or records the manual chunk size (off);
    glossary_auto_compression / glossary_max_output_tokens recompute the Auto glossary factor.
    Other keys are plain writes. Returns ``(changed {key: value}, env exports)``; the env is
    never written (a job exports its own from its config snapshot).
    """
    before = copy.deepcopy(dict(config))
    rule = _CHANGE_RULES.get(key)
    if rule is None:
        config[key] = value
        env = {}
    else:
        env = dict(rule(config, value) or {})
    changed = {k: config[k] for k in config if k not in before or before[k] != config[k]}
    return changed, env


def _gemini_thinking(config):
    return _flag(config, 'enable_gemini_thinking')


def _gpt_thinking(config):
    return _flag(config, 'enable_gpt_thinking')


def _gpt_budget(config):
    # budget_enabled = enabled and openrouter_use_reasoning_tokens (toggle_gpt_reasoning_controls)
    return _flag(config, 'enable_gpt_thinking') and _flag(config, 'openrouter_use_reasoning_tokens')


def _anthropic_thinking(config):
    return _flag(config, 'enable_anthropic_thinking')


def _anthropic_budget(config):
    # the budget entry: enabled and not Force Adaptive (toggle_anthropic_thinking_controls)
    return _flag(config, 'enable_anthropic_thinking') and not _flag(config, 'anthropic_force_adaptive')


def _output_mode_rule(part):
    def predicate(config):
        return output_mode_sub_settings(current_output_mode(config))[part]
    return predicate


def _glossary_extracts(config):
    return glossary_mode_extracts(glossary_manager_mode(config))


def _glossary_targeted(config):
    return glossary_mode_targeted_extraction(glossary_manager_mode(config))


def text_extraction_method(config):
    """The Text Extraction Method radio (Standard / Enhanced) the desktop shows for *config*
    (owner_state.initialize_extraction_variables: ``extraction_mode == 'enhanced'`` wins, else
    ``text_extraction_method``, default 'standard')."""
    from owner_state import initialize_extraction_variables

    owner = types.SimpleNamespace(config=config)
    initialize_extraction_variables(owner)
    return owner.text_extraction_method_var


def _extraction_method_rule(method):
    # other_settings.on_extraction_method_change: the html2text (Enhanced) options frame shows only for
    # Enhanced, the BeautifulSoup options (Fix Stray p&gt;) only for Standard
    def predicate(config):
        return text_extraction_method(config) == method
    return predicate


def _glossary_append_prompt(config):
    # update_append_prompt_state: the append format prompt follows the Append Glossary toggle
    return _flag(config, 'append_glossary')


for _rule_id, _predicate in (
        ('thinking:gemini', _gemini_thinking),
        ('thinking:gpt', _gpt_thinking),
        ('thinking:gpt_budget', _gpt_budget),
        ('thinking:anthropic', _anthropic_thinking),
        ('thinking:anthropic_budget', _anthropic_budget),
        ('output_mode:image', _output_mode_rule('image')),
        ('output_mode:video', _output_mode_rule('video')),
        ('output_mode:vision_request', _output_mode_rule('vision_request')),
        ('output_mode:vision', _output_mode_rule('vision_only')),
        ('glossary:extraction_prompt', _glossary_extracts),
        ('glossary:targeted_extraction', _glossary_targeted),
        ('glossary:append_prompt', _glossary_append_prompt),
        ('extraction:standard', _extraction_method_rule('standard')),
        ('extraction:enhanced', _extraction_method_rule('enhanced'))):
    register_visibility_rule(_rule_id, _predicate)
del _rule_id, _predicate


register_lock_rule('thoughts', THOUGHTS_LOCK_KEYS, thoughts_lock)
register_lock_rule('auto_compression', ('compression_factor',), compression_lock)
register_lock_rule('glossary_auto_compression', ('glossary_compression_factor',), glossary_compression_lock)
register_lock_rule('glossary_mode', tuple(GLOSSARY_MODE_TOGGLES) + ('fuzzy_auto_mapping_threshold',),
                   lambda config: glossary_mode_locks(glossary_manager_mode(config)))
