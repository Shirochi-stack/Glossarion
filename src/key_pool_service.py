"""Key pools without their widgets: the shared core of the desktop Multi API Key Manager.

Glossarion mobile rewrite, milestone U4 (shared-core design section 3.6, P5/P7). The desktop
``multi_api_key_manager.MultiAPIKeyDialog`` / ``RefusalPatternsDialog`` and the mobile app
(Multi-Key Manager, KeyEditor, Refusal patterns) run the same functions. Moved verbatim from
multi_api_key_manager.py; the dialogs keep their widgets, message boxes, threads and Qt
signals and call these:

* **Pool specs.** ``dedicated_pool_specs()`` / ``dedicated_pool_spec()`` (``_dedicated_pool_specs``
  / ``_dedicated_pool_spec``), ``export_pool_order()`` / ``pool_title()`` / ``pool_toggle_key()``
  (``_export_pool_order`` / ``_pool_title`` / ``_pool_toggle_key``) and ``POOL_SPECS``: all eleven
  pools (Translation = ``main``, Fallback, Glossary and the eight dedicated pools) with title,
  label, config key, toggle key and description. The section descriptions of the Translation,
  Fallback and Glossary groups are read from here by the desktop dialog too.
* **Entries.** ``new_key_entry()`` (the dict the Fallback / Glossary / dedicated "Add key"
  buttons append), ``new_main_key_entry()`` (the ``APIKeyEntry`` of the Translation pool's
  "Add Key"), ``missing_model_error()`` and ``added_key_extra_info()``; ``validate_entry()`` for
  editors that build a whole entry (mobile KeyEditor); ``find_duplicate_key()`` finds an entry a
  pool already holds (mobile "Add key" and legacy-list imports refuse exact duplicates).
* **Import / export** (format ``glossarion-key-pools`` version 1): ``sanitize_imported_keys``,
  ``classify_key_import``, ``plan_pool_aware_import``, ``legacy_key_entries``,
  ``pool_import_summary`` and the result messages (``_import_keys`` / ``_import_pool_aware`` /
  ``_import_legacy_list``), ``collect_pools_for_export`` / ``build_export_payload``
  (``_collect_pools_for_export`` / ``_export_keys``). ``export_pools`` / ``import_pools`` /
  ``apply_import_plan`` work on a plain config for the mobile app; unlike the desktop export
  they include the Glossary pool (tests/parity/DISCREPANCIES.md, U4 chain step 3).
* **Key tests.** ``build_test_request(entry, pool)`` returns the request the desktop test
  buttons send (client arguments, per-key client attributes in the original order, messages,
  send arguments, timeout); ``send_test_request`` / ``configure_test_client`` /
  ``test_response_passed`` / ``cancel_test_client`` / ``reset_api_watchdog`` are the bodies of
  the dialog's ``run_api_test`` / timeout closures. ``run_key_test(entry, pool, timeout=...)``
  runs one test with ``UnifiedClient`` for the mobile app; it reports the Audio / TTS and
  Image gen / edit pools as not testable (their keys need a speech or image request; the
  desktop still sends its chat probe).
* **Refusal patterns.** ``DEFAULT_REFUSAL_PATTERNS`` is the single default list used by
  ``RefusalPatternsDialog`` and ``unified_api_client.UnifiedClient._get_refusal_patterns``;
  ``load_refusal_patterns`` / ``load_disable_refusal_checks`` / ``load_refusal_length_limit`` /
  ``parse_refusal_length_limit`` / ``merge_refusal_pattern_lines`` are the dialog's config
  reads, its save normalisation and its "Load Patterns" merge.

``APIKeyEntry`` / ``APIKeyPool`` stay in multi_api_key_manager (importable without Qt with
``GLOSSARION_HEADLESS_KEY_MANAGER=1``); they are imported lazily, as are unified_api_client and
key_contexts / request_parameters.

Rules: GUI-free, Python 3.10 compatible, never imports PySide6, translator_gui or dpi_setup.
"""
from __future__ import annotations

import copy
import json
import os
import threading
from datetime import datetime

__all__ = [
    # refusal patterns
    "DEFAULT_REFUSAL_PATTERNS", "DEFAULT_REFUSAL_PATTERN_LENGTH_LIMIT", "default_refusal_patterns",
    "load_refusal_patterns", "load_disable_refusal_checks", "load_refusal_length_limit",
    "parse_refusal_length_limit", "merge_refusal_pattern_lines",
    # pool specs
    "POOL_SPECS", "POOL_IDS", "CORE_POOL_IDS", "dedicated_pool_specs", "dedicated_pool_spec",
    "pool_spec", "pool_config_key", "export_pool_order", "pool_title", "pool_toggle_key",
    "read_pool_keys",
    # entries
    "NEW_KEY_ENTRY_FIELDS", "DEFAULT_GOOGLE_REGION", "DEFAULT_AZURE_API_VERSION",
    "missing_model_error", "new_key_entry", "new_main_key_entry", "added_key_extra_info",
    "validate_entry", "find_duplicate_key",
    # individual endpoint (U9: IndividualEndpointDialog._validate / its shortcut buttons)
    "INDIVIDUAL_ENDPOINT_SHORTCUTS", "is_azure_endpoint", "individual_endpoint_error",
    # import / export
    "KEY_POOLS_FORMAT", "KEY_POOLS_VERSION", "INVALID_IMPORT_MESSAGE", "NO_VALID_KEYS_MESSAGE",
    "NO_POOLS_MESSAGE", "sanitize_imported_keys", "classify_key_import", "plan_pool_aware_import",
    "legacy_key_entries", "pool_import_summary", "pool_import_result_message",
    "legacy_import_result_message", "collect_pools_for_export", "build_export_payload",
    "count_pool_keys", "count_nonempty_pools", "export_pools", "import_pools", "apply_import_plan",
    # key tests
    "DEFAULT_KEY_TEST_TIMEOUT_SECONDS", "OPTIONAL_API_KEY_TEST_TIMEOUT_SECONDS", "UNTESTABLE_POOLS",
    "model_needs_api_key", "api_key_test_timeout_seconds", "test_messages", "build_test_request",
    "configure_test_client", "send_test_request", "test_response_passed", "cancel_test_client",
    "reset_api_watchdog", "is_rate_limit_error", "run_key_test",
    # key list status / live stats (U9)
    "key_tree_status", "RUNTIME_POOL_ATTRS", "LIVE_STAT_FIELDS", "runtime_key_pool", "live_key_stats",
    "merge_live_stats",
]


# =============================================================================================
# Refusal patterns (RefusalPatternsDialog; unified_api_client._get_refusal_patterns)
# =============================================================================================

#: The default AI refusal patterns (lower-case substrings), in the desktop order.
DEFAULT_REFUSAL_PATTERNS = (
    "i cannot assist", "i can't assist", "i'm not able to assist",
    "i cannot help", "i can't help", "i'm unable to help",
    "i'm afraid i cannot help with that", "designed to ensure appropriate use",
    "as an ai", "as a language model", "as an ai language model",
    "i don't feel comfortable", "i apologize, but i cannot",
    "i'm sorry, but i can't assist", "i'm sorry, but i cannot assist",
    "against my programming", "against my guidelines",
    "violates content policy", "i'm not programmed to",
    "cannot provide that kind", "unable to provide that",
    "i cannot assist with this request",
    "that's not within my capabilities to appropriately assist with",
    "is there something different i can help you with",
    "careful ethical considerations",
    "i could help you with a different question or task",
    "what other topics or questions can i help you explore",
    "i cannot and will not translate",
    "i cannot translate this content",
    "i can't translate this content",
)

DEFAULT_REFUSAL_PATTERN_LENGTH_LIMIT = 1000


def default_refusal_patterns():
    """Get default refusal patterns (a new list on every call; callers edit it in place)."""
    return list(DEFAULT_REFUSAL_PATTERNS)


def load_refusal_patterns(config):
    """Load refusal patterns from config (RefusalPatternsDialog._load_patterns)."""
    return config.get('refusal_patterns', default_refusal_patterns())


def load_disable_refusal_checks(config):
    """Load refusal check disable toggle from config"""
    try:
        return bool(config.get('disable_refusal_checks', True))
    except Exception:
        pass
    return True


def load_refusal_length_limit(config):
    """Load refusal length limit from config"""
    try:
        return int(config.get('refusal_pattern_length_limit', 1000))
    except Exception:
        pass
    return 1000


def parse_refusal_length_limit(raw_text):
    """The length limit the dialog saves for the text of its "Length limit" field.

    Non-digit text and values <= 0 save as 1000. Like the dialog, ``int()`` may still raise for
    digit characters it cannot parse (the dialog then saves 1000 in its ``except``).
    """
    raw_limit = raw_text.strip()
    limit_val = int(raw_limit) if raw_limit.isdigit() else 1000
    if limit_val <= 0:
        limit_val = 1000
    return limit_val


def merge_refusal_pattern_lines(patterns, lines):
    """Merge pattern lines (one per line, ``#`` comments skipped) into *patterns* in place.

    The "Load Patterns" merge of RefusalPatternsDialog: patterns are stripped and lower-cased,
    new ones are appended, ones already present are counted as skipped. Returns
    ``(added, skipped)``.
    """
    added = 0
    skipped = 0
    for line in lines:
        pattern = line.strip().lower()
        if not pattern or pattern.startswith('#'):
            continue
        if pattern in patterns:
            skipped += 1
        else:
            patterns.append(pattern)
            added += 1
    return added, skipped


# =============================================================================================
# Pool specs (MultiAPIKeyDialog._dedicated_pool_specs and the Translation/Fallback/Glossary groups)
# =============================================================================================

def dedicated_pool_specs():
    """Specs for full-parity dedicated key-pool UI sections.

    Adding another section should be a spec-only change when it uses the
    standard key shape: api key, model, per-key limits, temp, delay, Google
    creds, individual endpoint, test/reorder/enable/disable.
    """
    return {
        'glossary_refinement': {
            'title': 'Refinement Keys',
            'label': 'Refinement',
            'config_key': 'glossary_refinement_keys',
            'toggle_key': 'use_glossary_refinement_keys',
            'set_method': 'set_in_memory_glossary_refinement_keys',
            'clear_method': 'clear_in_memory_glossary_refinement_keys',
            'use_envs': ['USE_GLOSSARY_REFINEMENT_KEYS'],
            'keys_envs': ['GLOSSARY_REFINEMENT_API_KEYS'],
            'description': (
                "Configure dedicated keys for translation and glossary refinement API calls.\n"
                "This pool is preferred when the request context is 'refinement' or 'glossary_refinement'; eligible glossary/fallback paths may still apply for some errors."
            ),
        },
        'qa_scan': {
            'title': 'Vision Keys',
            'label': 'Vision',
            'config_key': 'qa_scan_keys',
            'toggle_key': 'use_qa_scan_keys',
            'set_method': 'set_in_memory_vision_keys',
            'clear_method': 'clear_in_memory_vision_keys',
            'use_envs': ['USE_VISION_KEYS', 'USE_QA_SCAN_KEYS'],
            'keys_envs': ['VISION_API_KEYS', 'QA_SCAN_API_KEYS'],
            'description': (
                "Configure dedicated keys for vision OCR and image scan calls.\n"
                "Normal rate-limit retries rotate within this pool; prohibited-content handling may still use configured fallback paths."
            ),
        },
        'metadata': {
            'title': 'Metadata Keys',
            'label': 'metadata',
            'config_key': 'metadata_keys',
            'toggle_key': 'use_metadata_keys',
            'set_method': '_unused_metadata_pool_set',
            'clear_method': '_unused_metadata_pool_clear',
            'use_envs': ['USE_METADATA_KEYS'],
            'keys_envs': ['METADATA_API_KEYS'],
            'description': (
                "Configure dedicated keys for book title, metadata, TOC, and header translation.\n"
                "Normal rate-limit retries rotate within this pool; prohibited-content handling may still use configured fallback paths."
            ),
        },
        'ai_truncation_detection': {
            'title': 'QA Scan Keys (for AI Truncation detection)',
            'label': 'QA scan',
            'config_key': 'ai_truncation_detection_keys',
            'toggle_key': 'use_ai_truncation_detection_keys',
            'set_method': '_unused_ai_truncation_detection_pool_set',
            'clear_method': '_unused_ai_truncation_detection_pool_clear',
            'use_envs': ['USE_AI_TRUNCATION_DETECTION_KEYS'],
            'keys_envs': ['AI_TRUNCATION_DETECTION_API_KEYS'],
            'description': (
                "Configure dedicated QA scan keys for qa_truncation AI checks.\n"
                "Normal rate-limit retries rotate within this pool; prohibited-content handling may still use configured fallback paths."
            ),
        },
        'rolling_summary': {
            'title': 'Rolling Summary Keys',
            'label': 'Rolling summary',
            'config_key': 'rolling_summary_keys',
            'toggle_key': 'use_rolling_summary_keys',
            'set_method': 'set_in_memory_rolling_summary_keys',
            'clear_method': 'clear_in_memory_rolling_summary_keys',
            'use_envs': ['USE_ROLLING_SUMMARY_KEYS'],
            'keys_envs': ['ROLLING_SUMMARY_API_KEYS'],
            'description': (
                "Configure dedicated keys for rolling-summary memory generation calls.\n"
                "Normal rate-limit retries rotate within this pool; prohibited-content handling may still use configured fallback paths."
            ),
        },
        'truncation_retry': {
            'title': 'Truncation Retry Keys',
            'label': 'Truncation retry',
            'config_key': 'truncation_retry_keys',
            'toggle_key': 'use_truncation_retry_keys',
            'set_method': 'set_in_memory_truncation_retry_keys',
            'clear_method': 'clear_in_memory_truncation_retry_keys',
            'use_envs': ['USE_TRUNCATION_RETRY_KEYS'],
            'keys_envs': ['TRUNCATION_RETRY_API_KEYS'],
            'description': (
                "Configure dedicated keys used only when RETRY_TRUNCATED schedules a retry.\n"
                "These keys are separate from Vision/QA truncation scan keys.\n"
                "If available, this pool is preferred for truncation retries; otherwise the retry may use the current request pool."
            ),
        },
        'inpainter': {
            'title': 'Image Gen / Edit Keys',
            'label': 'Image gen/edit',
            'config_key': 'inpainter_keys',
            'toggle_key': 'use_inpainter_keys',
            'set_method': 'set_in_memory_inpainter_keys',
            'clear_method': 'clear_in_memory_inpainter_keys',
            'use_envs': ['USE_INPAINTER_KEYS'],
            'keys_envs': ['INPAINTER_API_KEYS'],
            'description': (
                "Configure dedicated keys for image output and custom image-edit calls.\n"
                "Normal rate-limit retries rotate within this pool; prohibited-content handling may still use configured fallback paths."
            ),
        },
        'tts': {
            'title': 'Audio / TTS Keys',
            'label': 'Audio/TTS',
            'config_key': 'tts_keys',
            'toggle_key': 'use_tts_keys',
            'set_method': 'set_in_memory_tts_keys',
            'clear_method': 'clear_in_memory_tts_keys',
            'use_envs': ['USE_TTS_KEYS'],
            'keys_envs': ['TTS_API_KEYS'],
            'description': (
                "Configure dedicated keys for Audio output mode text-to-speech calls (request context 'tts').\n"
                "The key's model is the TTS model; an AIza key routes to Gemini TTS, others to the OpenAI-compatible speech endpoint or the key's individual endpoint."
            ),
        },
    }


def dedicated_pool_spec(pool_name, specs=None):
    """One dedicated pool's spec; ValueError for any other name.

    *specs* is the spec getter (default ``dedicated_pool_specs``).
    """
    try:
        return (specs or dedicated_pool_specs)()[pool_name]
    except KeyError:
        raise ValueError(f"Unknown dedicated key pool: {pool_name}")


def _core_pool_specs():
    """Translation (main), Fallback and Glossary: the three pools with their own dialog groups.

    ``title`` is the export title, ``section_title`` the dialog group title and ``description``
    the group description the desktop dialog shows (read from here).
    """
    return {
        'main': {
            'title': 'Translation Keys',
            'label': 'Translation',
            'section_title': 'Translation Keys (Main Pool)',
            'config_key': 'multi_api_keys',
            'toggle_key': 'use_multi_api_keys',
            'set_method': 'set_in_memory_multi_keys',
            'clear_method': 'clear_in_memory_multi_keys',
            'use_envs': ['USE_MULTI_KEYS'],
            'keys_envs': [],
            'description': (
                "Configure the main translation key rotation pool used by normal translation requests.\n"
                "Dedicated pools below, such as fallback, glossary, metadata, vision, and image keys, are not restricted by this toggle."
            ),
        },
        'fallback': {
            'title': 'Fallback Keys',
            'label': 'Fallback',
            'section_title': 'Fallback Keys (For Prohibited Content)',
            'config_key': 'fallback_keys',
            'toggle_key': 'use_fallback_keys',
            'set_method': None,
            'clear_method': None,
            'use_envs': ['USE_FALLBACK_KEYS'],
            'keys_envs': [],
            'description': (
                "Configure fallback keys that will be used when content is blocked.\n"
                "These should use different API keys or models that are less restrictive.\n"
                "With Translation Keys enabled: tried when the main rotation encounters prohibited content.\n"
                "In Single-Key Mode: tried directly when main key fails, bypassing main key retry."
            ),
        },
        'glossary': {
            'title': 'Glossary Keys',
            'label': 'Glossary',
            'section_title': 'Glossary Keys (For Glossary API Calls)',
            'config_key': 'glossary_keys',
            'toggle_key': 'use_glossary_keys',
            'set_method': 'set_in_memory_glossary_keys',
            'clear_method': 'clear_in_memory_glossary_keys',
            'use_envs': ['USE_GLOSSARY_KEYS'],
            'keys_envs': [],
            'description': (
                "Configure dedicated keys for glossary-context API calls.\n"
                "These keys will be used exclusively when the translation context is 'Glossary'.\n"
                "Normal rate-limit retries rotate within this pool; prohibited-content handling may still use configured fallback paths."
            ),
        },
    }


#: All eleven pools by id (``main``, ``fallback``, ``glossary``, then the dedicated pools in the
#: desktop section order). Read-only: ``dedicated_pool_specs()`` returns fresh copies.
POOL_SPECS = {**_core_pool_specs(), **dedicated_pool_specs()}
POOL_IDS = tuple(POOL_SPECS)
CORE_POOL_IDS = ('main', 'fallback', 'glossary')


def pool_spec(pool_name):
    """A copy of one pool's spec (all eleven pools); ValueError for an unknown pool."""
    try:
        return copy.deepcopy(POOL_SPECS[pool_name])
    except KeyError:
        raise ValueError(f"Unknown key pool: {pool_name}")


def pool_config_key(pool_name):
    """The config.json list key of a pool (``multi_api_keys`` for ``main``)."""
    return pool_spec(pool_name)['config_key']


def export_pool_order(dedicated_specs=None):
    """Logical pool order: main, fallback, then the dedicated pools.

    The desktop export / import order (``MultiAPIKeyDialog._export_pool_order``); it does not
    list the Glossary pool. *dedicated_specs* is the spec getter (default
    ``dedicated_pool_specs``).
    """
    order = ['main', 'fallback']
    try:
        order.extend(sorted((dedicated_specs or dedicated_pool_specs)().keys()))
    except Exception:
        pass
    return order


def pool_title(pool_name, dedicated_spec=None):
    """Export title of a pool (``MultiAPIKeyDialog._pool_title``; unknown names echo back)."""
    if pool_name == 'main':
        return 'Translation Keys'
    if pool_name == 'fallback':
        return 'Fallback Keys'
    try:
        return (dedicated_spec or dedicated_pool_spec)(pool_name).get('title', pool_name)
    except Exception:
        return pool_name


def pool_toggle_key(pool_name, dedicated_spec=None):
    """Enable-toggle config key of a pool (``MultiAPIKeyDialog._pool_toggle_key``; None if unknown)."""
    if pool_name == 'main':
        return 'use_multi_api_keys'
    if pool_name == 'fallback':
        return 'use_fallback_keys'
    try:
        return (dedicated_spec or dedicated_pool_spec)(pool_name).get('toggle_key')
    except Exception:
        return None


def read_pool_keys(config, pool_name):
    """The list of key dicts a pool holds in *config* ([] for an unknown pool)."""
    if pool_name == 'main':
        return list(config.get('multi_api_keys', []) or [])
    if pool_name == 'fallback':
        return list(config.get('fallback_keys', []) or [])
    try:
        return list(config.get(POOL_SPECS[pool_name]['config_key'], []) or [])
    except KeyError:
        return []


# =============================================================================================
# Entries (the "Add key" buttons)
# =============================================================================================

DEFAULT_GOOGLE_REGION = 'us-east5'
DEFAULT_AZURE_API_VERSION = '2025-01-01-preview'

#: Fields of the dict the Fallback / Glossary / dedicated "Add key" buttons append.
NEW_KEY_ENTRY_FIELDS = (
    'api_key', 'model', 'google_credentials', 'azure_endpoint', 'google_region',
    'azure_api_version', 'use_individual_endpoint', 'individual_output_token_limit',
    'individual_key_temperature', 'api_call_delay', 'enabled', 'times_used',
)


def missing_model_error(model):
    """The add-key validation: the error the dialog shows, or None."""
    if not model:
        return "Please enter a model name"
    return None


def new_key_entry(api_key, model, *, google_credentials=None, azure_endpoint=None,
                  google_region=None, azure_api_version=None, use_individual_endpoint=False):
    """New Fallback / Glossary / dedicated pool key (the dict their "Add key" buttons append)."""
    # Per-key output token limit and temperature default to None (global)
    # Users can set these later via the right-click context menu
    individual_output_token_limit = None
    individual_key_temperature = None

    return {
        'api_key': api_key,
        'model': model,
        'google_credentials': google_credentials,
        'azure_endpoint': azure_endpoint,
        'google_region': google_region,
        'azure_api_version': azure_api_version,
        'use_individual_endpoint': use_individual_endpoint,
        'individual_output_token_limit': individual_output_token_limit,
        'individual_key_temperature': individual_key_temperature,
        'api_call_delay': 0.0,
        'enabled': True,
        'times_used': 0
    }


def _api_key_entry_class():
    from multi_api_key_manager import APIKeyEntry  # GLOSSARION_HEADLESS_KEY_MANAGER=1 on mobile
    return APIKeyEntry


def new_main_key_entry(api_key, model, cooldown=60, *, google_credentials=None, azure_endpoint=None,
                       google_region=None, azure_api_version=None, use_individual_endpoint=False,
                       entry_cls=None):
    """New Translation (main) pool key: the ``APIKeyEntry`` of the dialog's "Add Key".

    ``.to_dict()`` is the config.json shape of the pool (``multi_api_keys``).
    """
    if entry_cls is None:
        entry_cls = _api_key_entry_class()

    # Per-key output token limit and temperature default to None (global)
    # Users can set these later via the right-click context menu
    individual_output_token_limit = None
    individual_key_temperature = None

    # Add to pool with new fields
    return entry_cls(
        api_key,
        model,
        cooldown,
        enabled=True,
        google_credentials=google_credentials,
        azure_endpoint=azure_endpoint,
        google_region=google_region,
        azure_api_version=azure_api_version,
        use_individual_endpoint=use_individual_endpoint,
        individual_output_token_limit=individual_output_token_limit,
        individual_key_temperature=individual_key_temperature,
    )


def added_key_extra_info(google_credentials, azure_endpoint):
    """The ``" (Google: ..., Azure: ...)"`` suffix of the "Added key for model" status."""
    extras = []
    if google_credentials:
        extras.append(f"Google: {os.path.basename(google_credentials)}")
    if azure_endpoint:
        extras.append(f"Azure: {azure_endpoint[:30]}...")

    return f" ({', '.join(extras)})" if extras else ""


#: Fields the runtime pools normalise (``APIKeyEntry``); ``validate_entry`` applies the same rules.
_NORMALIZED_FIELDS = ('individual_output_token_limit', 'individual_key_temperature', 'api_call_delay',
                      'request_parameters', 'disabled_contexts')


#: The Individual Endpoint dialog's shortcut buttons (``IndividualEndpointDialog._make_endpoint_shortcut``).
INDIVIDUAL_ENDPOINT_SHORTCUTS = (
    ("Ollama", "http://localhost:11434/v1"),
    ("LM Studio", "http://localhost:1234/v1"),
    ("TTS", "http://localhost:8000/audio/speech"),
    ("TTS v1", "http://localhost:8000/v1/audio/speech"),
)


def is_azure_endpoint(url):
    """Whether *url* is an Azure OpenAI endpoint (``IndividualEndpointDialog._is_azure_endpoint``)."""
    if not url:
        return False
    url_l = url.lower()
    return (".openai.azure.com" in url_l) or ("azure.com/openai" in url_l) or ("/openai/deployments/" in url_l)


def individual_endpoint_error(enabled, url, api_version):
    """``IndividualEndpointDialog._validate``'s rules: the dialog's "Validation Error" message, or
    None when the per-key endpoint may be saved (off, or an http(s) URL with an API version for Azure)."""
    if not enabled:
        return None
    url = str(url or "").strip()
    if not url:
        return "Endpoint Base URL is required when Enable is ON."
    if not (url.startswith("http://") or url.startswith("https://")):
        return "Endpoint URL must start with http:// or https://"
    if is_azure_endpoint(url):
        ver = str(api_version or "").strip()
        if not ver:
            return "Azure API Version is required for Azure endpoints."
    return None


def validate_entry(entry, pool='main'):
    """``(normalized entry, None)`` or ``(None, error message)`` for a whole key entry.

    For editors that build an entry at once (mobile KeyEditor): API key and model are stripped
    as the add-key buttons strip them and an empty model gives the add-key error. The
    Translation pool stores ``APIKeyEntry.to_dict()`` (its config shape); the other pools keep
    their dict shape, with the fields the runtime pools normalise (per-key output limit,
    temperature, delay, request parameters, disabled contexts) normalised by ``APIKeyEntry``.
    """
    data = dict(entry or {})
    data['api_key'] = str(data.get('api_key') or '').strip()
    data['model'] = str(data.get('model') or '').strip()
    error = missing_model_error(data['model'])
    if error:
        return None, error
    error = individual_endpoint_error(data.get('use_individual_endpoint'), data.get('azure_endpoint'),
                                      data.get('azure_api_version'))
    if error:
        return None, error
    try:
        normalized = _api_key_entry_class().from_dict(data).to_dict()
    except Exception as exc:
        return None, str(exc)
    if pool == 'main':
        data.update(normalized)
        return data, None
    for name in _NORMALIZED_FIELDS:
        if name in data:
            data[name] = normalized[name]
    return data, None


def _duplicate_identity(entry):
    """What makes two entries the same key for a pool: the API key, the model and where / how the
    request is sent (individual endpoint + API version, Google credentials). ``None`` for an
    encrypted (``ENC:``) key, which cannot be compared."""
    get = entry.get if isinstance(entry, dict) else (lambda name, default=None: getattr(entry, name, default))
    api_key = str(get('api_key') or '').strip()
    if api_key.startswith('ENC:'):
        return None
    use_endpoint = bool(get('use_individual_endpoint', False))
    endpoint = str(get('azure_endpoint') or '').strip() if use_endpoint else ''
    version = str(get('azure_api_version') or '').strip() if use_endpoint else ''
    return (api_key, str(get('model') or '').strip(), use_endpoint, endpoint, version,
            str(get('google_credentials') or '').strip())


def find_duplicate_key(keys, entry):
    """Index of the first key in *keys* that duplicates *entry* (same API key, model, individual
    endpoint and Google credentials), or ``None``. Encrypted keys never match."""
    wanted = _duplicate_identity(entry)
    if wanted is None:
        return None
    for index, existing in enumerate(keys or []):
        if _duplicate_identity(existing) == wanted:
            return index
    return None


# =============================================================================================
# Import / export (format glossarion-key-pools, version 1)
# =============================================================================================

KEY_POOLS_FORMAT = 'glossarion-key-pools'
KEY_POOLS_VERSION = 1
INVALID_IMPORT_MESSAGE = "Invalid or unrecognized key file format"
NO_VALID_KEYS_MESSAGE = "No valid keys found in file"
NO_POOLS_MESSAGE = "No recognizable pools found in file"


def sanitize_imported_keys(raw_keys):
    """Filter a raw list down to valid key dicts (lossless, native shape).

    Returns (clean_keys, skipped_count).
    """
    clean, skipped = [], 0
    if not isinstance(raw_keys, list):
        return clean, skipped
    for kd in raw_keys:
        if isinstance(kd, dict) and 'api_key' in kd and 'model' in kd:
            clean.append(kd)
        else:
            skipped += 1
    return clean, skipped


def classify_key_import(data):
    """What a loaded key file is: ``('pools', pools)``, ``('legacy', key_list)`` or ``(None, None)``.

    Supported formats:
      * Pool-aware export: {"format": "glossarion-key-pools", "pools": {...}}
        -> each recognized pool in the file is restored to its matching pool.
      * Legacy flat list:  [ {api_key, model, ...}, ... ]
        -> imported into the Translation (main) pool for backwards compat.
      * A single bare key object (imported like a one-key legacy list).
    """
    if isinstance(data, dict) and isinstance(data.get('pools'), dict):
        return 'pools', data['pools']
    elif isinstance(data, list):
        return 'legacy', data
    elif isinstance(data, dict) and 'api_key' in data and 'model' in data:
        # A single bare key object
        return 'legacy', [data]
    return None, None


def plan_pool_aware_import(pools_obj, known):
    """Plan the restore of every recognized pool found in a structured export file.

    Returns ``(plan, total_skipped, unknown_pools)`` with ``plan`` a list of
    ``(pool_name, clean_keys, enabled)``; *known* lists the pools this side restores.
    """
    known = set(known)
    plan = []            # list of (pool_name, clean_keys, enabled)
    total_skipped = 0
    unknown_pools = []

    for pool_name, value in pools_obj.items():
        if pool_name not in known:
            unknown_pools.append(str(pool_name))
            continue
        if isinstance(value, dict):
            raw_keys = value.get('keys', [])
            enabled = value.get('enabled', None)
        elif isinstance(value, list):
            raw_keys = value
            enabled = None
        else:
            continue
        clean, skipped = sanitize_imported_keys(raw_keys)
        total_skipped += skipped
        plan.append((pool_name, clean, enabled))

    return plan, total_skipped, unknown_pools


def legacy_key_entries(valid, entry_cls=None):
    """``APIKeyEntry`` objects for the valid keys of a legacy flat list: ``(entries, failed)``."""
    if entry_cls is None:
        entry_cls = _api_key_entry_class()
    entries = []
    failed = 0
    for kd in valid:
        try:
            entries.append(entry_cls.from_dict(kd))
        except Exception:
            failed += 1
    return entries, failed


def pool_import_summary(plan, title=None):
    """The bullet list of the "Import key pools" confirmation (one line per planned pool)."""
    title = title or pool_title
    return "\n".join(
        f"  • {title(pn)}: {len(keys)} key(s)" for pn, keys, _en in plan
    )


def pool_import_result_message(applied_keys, pool_count, total_skipped, unknown_pools):
    """The message after a pool-aware import."""
    msg = f"Imported {applied_keys} key(s) across {pool_count} pool(s)"
    extras = []
    if total_skipped:
        extras.append(f"{total_skipped} invalid key(s) skipped")
    if unknown_pools:
        extras.append("ignored unknown pool(s): " + ", ".join(unknown_pools))
    if extras:
        msg += "\n" + "; ".join(extras)
    return msg


def legacy_import_result_message(imported, skipped):
    """The message after a legacy (flat list) import."""
    msg = f"Imported {imported} API key(s) into the Translation pool"
    if skipped:
        msg += f" ({skipped} skipped)"
    return msg


def collect_pools_for_export(config, read_pool_keys, order=None, *, title=None, toggle_key=None):
    """``{pool: {'title', 'enabled', 'keys'}}`` for *order* (default ``export_pool_order()``).

    *read_pool_keys(pool)* returns a pool's current key dicts; *title* / *toggle_key* default
    to ``pool_title`` / ``pool_toggle_key``.
    """
    cfg = config
    title = title or pool_title
    toggle_key = toggle_key or pool_toggle_key
    pools = {}
    for pool_name in (export_pool_order() if order is None else order):
        toggle_key_name = toggle_key(pool_name)
        pools[pool_name] = {
            'title': title(pool_name),
            'enabled': bool(cfg.get(toggle_key_name, False)) if toggle_key_name else False,
            'keys': read_pool_keys(pool_name),
        }
    return pools


def build_export_payload(pools):
    """The ``glossarion-key-pools`` version 1 document for *pools*."""
    return {
        'format': KEY_POOLS_FORMAT,
        'version': KEY_POOLS_VERSION,
        'exported_at': datetime.now().isoformat(timespec='seconds'),
        'pools': pools,
    }


def count_pool_keys(pools):
    """Number of keys across the pools of an export."""
    return sum(len(p['keys']) for p in pools.values())


def count_nonempty_pools(pools):
    """Number of pools of an export that hold at least one key."""
    return sum(1 for p in pools.values() if p['keys'])


def _mobile_pool_order(pools=None):
    order = ['main', 'fallback', 'glossary'] + sorted(dedicated_pool_specs().keys())
    if pools:
        wanted = set(pools)
        order = [p for p in order if p in wanted]
    return order


def export_pools(config, pools=None):
    """Export a plain config's pools (all eleven by default, Glossary included) as a document.

    Deep-copies the keys so the payload never aliases the config.
    """
    order = _mobile_pool_order(pools)
    collected = collect_pools_for_export(
        config,
        lambda pool_name: copy.deepcopy(read_pool_keys(config, pool_name)),
        order,
        title=lambda pool_name: POOL_SPECS[pool_name]['title'],
        toggle_key=lambda pool_name: POOL_SPECS[pool_name]['toggle_key'],
    )
    return build_export_payload(collected)


def import_pools(payload, config=None, *, apply=False, dry_run=False, known=None):
    """Plan (and with ``apply=True`` apply to *config*) the import of a key file.

    *payload* is the loaded JSON (or its text). Returns a dict: ``kind`` ('pools' / 'legacy' /
    None), ``legacy``, ``items`` (``[(pool, keys, enabled)]``, enabled None = unchanged),
    ``skipped``, ``unknown`` (pool names this side does not restore) and ``error`` (None or the
    dialog's message); ``applied`` (number of keys) when applied. A pool-aware file REPLACES
    the listed pools; a legacy list is appended to the Translation pool, each key normalised
    through ``APIKeyEntry`` as the desktop pool does. *known* defaults to all eleven pools
    (the desktop import knows ``export_pool_order()``, without Glossary).
    """
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode('utf-8-sig')
    if isinstance(payload, str):
        payload = json.loads(payload)
    kind, data = classify_key_import(payload)
    result = {'kind': kind, 'legacy': kind == 'legacy', 'items': [], 'skipped': 0, 'unknown': [],
              'error': None}
    if kind is None:
        result['error'] = INVALID_IMPORT_MESSAGE
        return result
    if kind == 'legacy':
        valid, skipped = sanitize_imported_keys(data)
        if not valid:
            result['skipped'] = skipped
            result['error'] = NO_VALID_KEYS_MESSAGE
            return result
        entries, failed = legacy_key_entries(valid)
        result['items'] = [('main', [e.to_dict() for e in entries], None)]
        result['skipped'] = skipped + failed
    else:
        plan, total_skipped, unknown_pools = plan_pool_aware_import(
            data, POOL_IDS if known is None else known)
        result['items'] = [(pool_name, copy.deepcopy(keys), enabled) for pool_name, keys, enabled in plan]
        result['skipped'] = total_skipped
        result['unknown'] = unknown_pools
        if not plan:
            result['error'] = NO_POOLS_MESSAGE
            return result
    if apply and not dry_run and config is not None:
        result['applied'] = apply_import_plan(config, result)
    return result


def apply_import_plan(config, plan):
    """Apply an ``import_pools`` result to *config*; returns the number of keys imported."""
    applied = 0
    legacy = bool(plan.get('legacy'))
    for pool_name, keys, enabled in plan.get('items') or []:
        spec = POOL_SPECS[pool_name]
        keys = [dict(k) for k in keys]
        if legacy:
            config[spec['config_key']] = list(config.get(spec['config_key'], []) or []) + keys
        else:
            config[spec['config_key']] = keys
        if enabled is not None and spec['toggle_key']:
            config[spec['toggle_key']] = bool(enabled)
        applied += len(keys)
    return applied


# =============================================================================================
# Key tests (the dialog's run_api_test / run_with_timeout closures)
# =============================================================================================

DEFAULT_KEY_TEST_TIMEOUT_SECONDS = 30
OPTIONAL_API_KEY_TEST_TIMEOUT_SECONDS = 60

#: Pools whose keys the chat probe cannot test, with the reason (mobile shows "Not testable").
UNTESTABLE_POOLS = {
    'tts': ("Audio / TTS keys need a speech request; the key test sends a chat request, "
            "which speech models reject."),
    'inpainter': ("Image gen / edit keys need an image request; the key test sends a chat request, "
                  "which image models reject."),
}

#: Debug-log wording of the pools whose desktop test logs its client setup.
_TEST_DEBUG_TARGETS = {
    'main': ('key test', 'test'),
    'fallback': ('fallback key test', 'fallback test'),
}


# Models/prefixes that don't require an API key - delegates to UnifiedClient's authoritative list
def model_needs_api_key(model):
    """Return False for models that authenticate without an API key."""
    try:
        from unified_api_client import UnifiedClient
        return UnifiedClient._model_needs_api_key(model)
    except Exception:
        return True


def api_key_test_timeout_seconds(model, needs_api_key=None):
    """Give models with optional API keys more time to finish their test request."""
    if not (needs_api_key or model_needs_api_key)(model):
        return OPTIONAL_API_KEY_TEST_TIMEOUT_SECONDS
    return DEFAULT_KEY_TEST_TIMEOUT_SECONDS


def test_messages():
    """The key-test conversation."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'API test successful' and nothing else."}
    ]


def _entry_value(entry, name, default=None):
    if isinstance(entry, dict):
        return entry.get(name, default)
    return getattr(entry, name, default)


def build_test_request(entry, pool='main', *, timeout=None, api_key=None, model=None):
    """The request a key test sends for *entry* (a key dict or an ``APIKeyEntry``) of *pool*.

    A dict: ``pool``, ``variant`` ('standard' for Translation / Fallback / Glossary,
    'dedicated' for the dedicated pools: their tests set the per-key client attributes
    differently), ``client_kwargs`` (``UnifiedClient(**client_kwargs)``),
    ``max_retries_override``, ``steps`` (per-key client attributes in the original order, each
    ``{'attrs': [(name, value)], 'log': kind or None, 'value': ...}``), ``debug`` (the desktop
    debug-log wording or None), ``messages``, ``send_kwargs``, ``timeout`` (seconds) and
    ``testable`` / ``reason`` (False for the Audio / TTS and Image gen / edit pools; the desktop
    still sends the chat probe for them). *api_key* / *model* override the entry's (the dialog
    passes the values it read when the test was queued).
    """
    if api_key is None:
        api_key = _entry_value(entry, 'api_key', '')
    if model is None:
        model = _entry_value(entry, 'model', '')
    steps = []
    if pool in ('main', 'fallback', 'glossary'):
        variant = 'standard'
        # Set Google credentials and other key-specific settings
        google_credentials = _entry_value(entry, 'google_credentials')
        if google_credentials:
            steps.append({'attrs': [('current_key_google_creds', google_credentials),
                                    ('google_creds_path', google_credentials)],
                          'log': 'google_credentials', 'value': google_credentials})
        google_region = _entry_value(entry, 'google_region')
        if google_region:
            steps.append({'attrs': [('current_key_google_region', google_region)],
                          'log': 'google_region', 'value': google_region})
        # Set Azure endpoint settings if configured
        if _entry_value(entry, 'use_individual_endpoint', False):
            azure_endpoint = _entry_value(entry, 'azure_endpoint')
            if azure_endpoint:
                steps.append({'attrs': [('current_key_azure_endpoint', azure_endpoint),
                                        ('current_key_use_individual_endpoint', True)],
                              'log': 'azure_endpoint', 'value': azure_endpoint})
            azure_api_version = _entry_value(entry, 'azure_api_version')
            if azure_api_version:
                steps.append({'attrs': [('current_key_azure_api_version', azure_api_version)],
                              'log': None, 'value': azure_api_version})
    else:
        variant = 'dedicated'
        for source, target in (
            ('google_credentials', 'current_key_google_creds'),
            ('google_region', 'current_key_google_region'),
            ('azure_endpoint', 'current_key_azure_endpoint'),
            ('azure_api_version', 'current_key_azure_api_version'),
        ):
            value = _entry_value(entry, source)
            if value:
                steps.append({'attrs': [(target, value)], 'log': None, 'value': value})
        if _entry_value(entry, 'use_individual_endpoint'):
            steps.append({'attrs': [('current_key_use_individual_endpoint', True)], 'log': None,
                          'value': True})
    reason = UNTESTABLE_POOLS.get(pool)
    return {
        'pool': pool,
        'variant': variant,
        'client_kwargs': {'api_key': api_key, 'model': model, 'output_dir': None},
        'max_retries_override': 1,
        'steps': steps,
        'debug': _TEST_DEBUG_TARGETS.get(pool),
        'messages': test_messages(),
        'send_kwargs': {'temperature': 0.7, 'max_tokens': 1000},
        'timeout': api_key_test_timeout_seconds(model) if timeout is None else timeout,
        'testable': reason is None,
        'reason': reason,
    }


def configure_test_client(client, request, *, log=None):
    """Apply a test request's retry override and per-key attributes to a ``UnifiedClient``.

    *log* receives the desktop debug lines (the dialog passes ``print``).
    """
    debug = request.get('debug') if log is not None else None

    # Force 1 retries for testing to speed up failure detection
    try:
        tls = client._get_thread_local_client()
        tls.max_retries_override = request['max_retries_override']
        if debug:
            log(f"[DEBUG] Set max_retries_override=1 for {debug[0]}")
    except Exception:
        pass

    for step in request['steps']:
        for name, value in step['attrs']:
            setattr(client, name, value)
        kind = step['log']
        if not debug or not kind:
            continue
        value = step['value']
        if kind == 'google_credentials':
            log(f"[DEBUG] Set Google credentials for {debug[1]}: {os.path.basename(value)}")
        elif kind == 'google_region':
            log(f"[DEBUG] Set Google region for {debug[1]}: {value}")
        elif kind == 'azure_endpoint':
            log(f"[DEBUG] Set Azure endpoint for {debug[1]}: {value[:50]}...")
    return client


def send_test_request(request, *, client_cls=None, on_client=None, log=None):
    """Create the test client, configure it and send the probe: ``(client, response)``.

    *on_client(client)* runs right after the client is created (the dialog keeps it for its
    timeout cancel); exceptions propagate to the caller.
    """
    if client_cls is None:
        from unified_api_client import UnifiedClient as client_cls
    # Real key probes must not reload or rotate the shared translation pool.
    client_factory = getattr(client_cls, 'for_key_test', client_cls)
    client = client_factory(**request['client_kwargs'])
    if on_client is not None:
        on_client(client)
    configure_test_client(client, request, log=log)
    response = client.send(
        list(request['messages']),
        **request['send_kwargs']
    )
    return client, response


def test_response_passed(response, request):
    """Did the probe answer "API test successful"? (May raise on a malformed response, as the
    dialog's check does; the dialog reports that as an error.)"""
    if request.get('variant') == 'dedicated':
        content = response[0] if isinstance(response, tuple) else response
        return bool(content and "test successful" in str(content).lower())
    if response and isinstance(response, tuple):
        content, _ = response
        if content and "test successful" in content.lower():
            return True
    return False


def cancel_test_client(client):
    """Hard-cancel a timed-out test client (stop its retry loops, close its HTTP session)."""
    if client:
        try:
            client._cancelled = True  # signal internal retry loops to stop
            # Close OpenAI SDK client (kills underlying httpx transport)
            oc = getattr(client, 'openai_client', None)
            if oc and hasattr(oc, 'close'):
                oc.close()
            elif oc and hasattr(oc, '_client') and hasattr(oc._client, 'close'):
                oc._client.close()
        except Exception:
            pass


def reset_api_watchdog():
    """Clear the API watchdog so the progress display stops showing an in-flight call."""
    try:
        from unified_api_client import _api_watchdog_reset
        _api_watchdog_reset()
    except Exception:
        pass


def is_rate_limit_error(error_msg):
    """Is a key-test error a rate limit (HTTP 429)?"""
    return "429" in error_msg or "rate limit" in error_msg.lower()


def run_key_test(entry, pool='main', *, timeout=None, client_cls=None, log=None):
    """Test one key with ``UnifiedClient`` (blocking; the mobile app runs it off the UI loop).

    Returns ``{'ok', 'status', 'message', 'last_test_result'}``: status 'passed', 'failed'
    (unexpected answer), 'error', 'rate_limited', 'timeout' or 'untestable' (``ok`` None; the
    Audio / TTS and Image gen / edit pools). A timed-out probe is cancelled like the dialog
    cancels it and left to finish on its daemon thread.
    """
    request = build_test_request(entry, pool, timeout=timeout)
    if not request['testable']:
        return {'ok': None, 'status': 'untestable', 'message': request['reason'],
                'last_test_result': None}
    timeout_seconds = request['timeout']
    client_ref = [None]
    outcome = {}

    def run_api_test():
        try:
            _client, response = send_test_request(
                request, client_cls=client_cls,
                on_client=lambda client: client_ref.__setitem__(0, client), log=log)
            if test_response_passed(response, request):
                outcome['result'] = ('passed', True, "Test passed")
            else:
                outcome['result'] = ('failed', False, "Unexpected response")
        except Exception as exc:
            error_msg = str(exc)
            status = 'rate_limited' if is_rate_limit_error(error_msg) else 'error'
            outcome['result'] = (status, False, f"Error: {error_msg[:200]}")

    worker = threading.Thread(target=run_api_test, name="gl-key-test", daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive() or 'result' not in outcome:
        cancel_test_client(client_ref[0])
        reset_api_watchdog()
        return {'ok': False, 'status': 'timeout', 'message': f"Timed out ({timeout_seconds}s)",
                'last_test_result': 'timeout'}
    status, ok, message = outcome['result']
    return {'ok': ok, 'status': status, 'message': message, 'last_test_result': status}


# ---------------------------------------------------------------------------
# Test API connections (other_settings.test_api_connections, moved in U9 so Glossarion Mobile's
# Settings › Endpoints probes every configured endpoint the same way; the desktop dialog keeps its
# progress / result boxes and calls these)
# ---------------------------------------------------------------------------


def collect_test_endpoints(owner):
    """The endpoints Other Settings › Test Connections probes (moved verbatim from
    ``other_settings.test_api_connections``; ``owner`` carries the desktop vars: use_custom_openai_endpoint_var,
    openai_base_url_var, azure_api_version_var, model_var, groq_base_url_var, fireworks_base_url_var,
    use_gemini_openai_endpoint_var, gemini_openai_endpoint_var). Returns the ``endpoints_to_test`` list of
    ``(name, url, model[, kind])`` tuples (kind ``azure`` / ``grpc_gemini``)."""
    # Collect all configured endpoints
    endpoints_to_test = []

    # OpenAI endpoint - only test if checkbox is enabled
    if owner.use_custom_openai_endpoint_var:
        openai_url = owner.openai_base_url_var
        if openai_url:
            # Check if it's Azure
            if '.azure.com' in openai_url or '.cognitiveservices' in openai_url:
                # Azure endpoint
                deployment = owner.model_var if hasattr(owner, 'model_var') else "gpt-35-turbo"
                api_version = owner.azure_api_version_var if hasattr(owner, 'azure_api_version_var') else "2024-08-01-preview"

                # Format Azure URL
                if '/openai/deployments/' not in openai_url:
                    azure_url = f"{openai_url.rstrip('/')}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
                else:
                    azure_url = openai_url

                endpoints_to_test.append(("Azure OpenAI", azure_url, deployment, "azure"))
            else:
                # Regular custom endpoint
                endpoints_to_test.append(("OpenAI (Custom)", openai_url, owner.model_var if hasattr(owner, 'model_var') else "gpt-3.5-turbo"))
        else:
            # Use default OpenAI endpoint if checkbox is on but no custom URL provided
            endpoints_to_test.append(("OpenAI (Default)", "https://api.openai.com/v1", owner.model_var if hasattr(owner, 'model_var') else "gpt-3.5-turbo"))

    # Groq endpoint
    if hasattr(owner, 'groq_base_url_var'):
        groq_url = owner.groq_base_url_var
        if groq_url:
            # For Groq, we need a groq-prefixed model
            current_model = owner.model_var if hasattr(owner, 'model_var') else "llama-3-70b"
            groq_model = current_model if current_model.startswith('groq/') else current_model.replace('groq/', '')
            endpoints_to_test.append(("Groq/Local", groq_url, groq_model))

    # Fireworks endpoint
    if hasattr(owner, 'fireworks_base_url_var'):
        fireworks_url = owner.fireworks_base_url_var
        if fireworks_url:
            # For Fireworks, we need the accounts/ prefix
            current_model = owner.model_var if hasattr(owner, 'model_var') else "llama-v3-70b-instruct"
            fw_model = current_model if current_model.startswith('accounts/') else f"accounts/fireworks/models/{current_model.replace('fireworks/', '')}"
            endpoints_to_test.append(("Fireworks", fireworks_url, fw_model))

    # Gemini Custom Endpoint — detect gRPC vs OpenAI-compatible REST
    if hasattr(owner, 'use_gemini_openai_endpoint_var') and owner.use_gemini_openai_endpoint_var:
        gemini_url = owner.gemini_openai_endpoint_var
        if gemini_url:
            _ep = gemini_url.strip().lower()
            _is_grpc = not ('/openai' in _ep or _ep.startswith('http://') or _ep.startswith('https://'))

            current_model = owner.model_var if hasattr(owner, 'model_var') else "gemini-2.0-flash-exp"
            gemini_model = current_model.replace('gemini/', '') if current_model.startswith('gemini/') else current_model

            if _is_grpc:
                # Bare hostname → gRPC (eRPC) endpoint
                endpoints_to_test.append(("Gemini (gRPC)", gemini_url.strip(), gemini_model, "grpc_gemini"))
            else:
                # URL with /openai or http(s):// → OpenAI-compatible REST
                if not gemini_url.endswith('/openai/'):
                    if gemini_url.endswith('/'):
                        gemini_url = gemini_url + 'openai/'
                    else:
                        gemini_url = gemini_url + '/openai/'
                endpoints_to_test.append(("Gemini (OpenAI-Compatible)", gemini_url, gemini_model))
    return endpoints_to_test


def run_endpoint_tests(endpoints_to_test, api_key, openai, cancel_event):
    """Probe each endpoint (moved verbatim from ``other_settings.test_api_connections``'s worker):
    gRPC Gemini through ``grpc_gemini_client``, Azure with ``api-key`` headers, any other endpoint with a
    3 s reachability GET, then a 5-token chat completion through ``openai.OpenAI``; the common errors
    are simplified. ``cancel_event`` (threading.Event) stops between endpoints. Returns the
    ``✅ name: ...`` / ``❌ name: ...`` result lines."""
    results = []
    for endpoint_info in endpoints_to_test:
        if cancel_event.is_set():
            break
        if len(endpoint_info) == 4 and endpoint_info[3] == "grpc_gemini":
            # gRPC (eRPC) Gemini endpoint
            name, grpc_host, model, _ = endpoint_info
            try:
                from grpc_gemini_client import GrpcGeminiClient, GRPC_AVAILABLE, GrpcGeminiError
                if not GRPC_AVAILABLE:
                    results.append(f"❌ {name}: gRPC dependencies not installed (pip install grpcio google-ai-generativelanguage)")
                    continue
                client = GrpcGeminiClient(api_key=api_key, endpoint=grpc_host)
                try:
                    resp = client.generate_content(
                        model=model,
                        messages=[{"role": "user", "content": "Hi"}],
                        max_output_tokens=5
                    )
                    results.append(f"✅ {name}: Connected successfully! (Model: {model}, Endpoint: {grpc_host})")
                finally:
                    client.close()
            except Exception as e:
                error_msg = str(e)[:150]
                if "UNAUTHENTICATED" in error_msg or "401" in error_msg or "403" in error_msg:
                    error_msg = "Authentication failed. Check API key."
                elif "UNAVAILABLE" in error_msg:
                    error_msg = f"gRPC endpoint unreachable: {grpc_host}"
                results.append(f"❌ {name}: {error_msg}")
        elif len(endpoint_info) == 4 and endpoint_info[3] == "azure":
            # Azure endpoint
            name, base_url, model, endpoint_type = endpoint_info
            try:
                # Azure uses different headers
                import requests
                headers = {
                    "api-key": api_key,
                    "Content-Type": "application/json"
                }

                response = requests.post(
                    base_url,
                    headers=headers,
                    json={
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 5
                    },
                    timeout=5.0
                )

                if response.status_code == 200:
                    results.append(f"✅ {name}: Connected successfully! (Deployment: {model})")
                else:
                    results.append(f"❌ {name}: {response.status_code} - {response.text[:100]}")

            except Exception as e:
                error_msg = str(e)[:100]
                results.append(f"❌ {name}: {error_msg}")
        else:
            # Regular OpenAI-compatible endpoint
            name, base_url, model = endpoint_info[:3]
            try:
                # Quick endpoint reachability probe (low timeout)
                try:
                    import httpx
                    probe_timeout = 3.0
                    probe_url = base_url.rstrip("/")  # tolerate missing path
                    httpx.get(probe_url, timeout=probe_timeout)
                except Exception as probe_err:
                    results.append(f"❌ {name}: Endpoint unreachable ({probe_err})")
                    continue
                if cancel_event.is_set():
                    break
                # Create client for this endpoint
                test_client = openai.OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=5.0,  # Keep model test short to avoid UI freeze
                    max_retries=0  # Fail fast on 404/connection errors
                )

                # Try a minimal completion
                response = test_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5
                )

                results.append(f"✅ {name}: Connected successfully! (Model: {model})")
            except Exception as e:
                error_msg = str(e)
                # Simplify common error messages
                if "timed out" in error_msg.lower():
                    error_msg = f"Connection timed out. The endpoint is running but the model '{model}' may be too slow to respond."
                elif "404" in error_msg:
                    error_msg = "404 - Endpoint not found. Check URL and model name."
                elif "401" in error_msg or "403" in error_msg:
                    error_msg = "Authentication failed. Check API key."
                elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                    error_msg = f"Model '{model}' not found at this endpoint."

                results.append(f"❌ {name}: {error_msg}")
    return results


# =============================================================================================
# Key list status and live per-key stats (U9)
# =============================================================================================

def key_tree_status(key):
    """The Multi-Key Manager key tree's Status column: ``(status, tags)``.

    Moved verbatim from ``MultiAPIKeyDialog._refresh_key_list`` (the desktop tree calls it; the
    mobile key cards read it). *key* is an ``APIKeyEntry`` or any object with its attributes
    (``last_test_result``, ``last_test_message``, ``enabled``, ``is_cooling_down``,
    ``last_error_time``, ``cooldown``); an expired cooldown clears ``is_cooling_down`` as on desktop.
    """
    import time

    if key.last_test_result is None and hasattr(key, '_testing'):
        status = "⏳ Testing..."
        tags = ('testing',)
    elif not key.enabled:
        status = "Disabled"
        tags = ('disabled',)
    elif key.last_test_result == 'passed':
        status = "✅ Passed"
        tags = ('passed',)
    elif key.last_test_result == 'failed':
        status = "❌ Failed"
        tags = ('failed',)
    elif key.last_test_result == 'timeout':
        status = "⏱️ Timed Out"
        tags = ('timeout',)
    elif key.last_test_result == 'rate_limited':
        status = "⚠️ Rate Limited"
        tags = ('ratelimited',)
    elif key.last_test_result == 'error':
        status = "❌ Error"
        if key.last_test_message:
            status += f": {key.last_test_message[:20]}..."
        tags = ('error',)
    elif key.is_cooling_down and key.last_error_time:
        remaining = int(key.cooldown - (time.time() - key.last_error_time))
        if remaining > 0:
            status = f"Cooling ({remaining}s)"
            tags = ('cooling',)
        else:
            key.is_cooling_down = False
            status = "Active"
            tags = ('active',)
    else:
        status = "Active"
        tags = ('active',)
    return status, tags


#: ``UnifiedClient`` class attribute holding each pool's runtime ``APIKeyPool`` (the fallback
#: pool has none: its keys are tried one by one per request).
RUNTIME_POOL_ATTRS = {
    "main": "_api_key_pool",
    "glossary": "_glossary_key_pool",
    "glossary_refinement": "_glossary_refinement_key_pool",
    "qa_scan": "_qa_scan_key_pool",
    "metadata": "_metadata_key_pool",
    "ai_truncation_detection": "_ai_truncation_detection_key_pool",
    "rolling_summary": "_rolling_summary_key_pool",
    "truncation_retry": "_truncation_retry_key_pool",
    "inpainter": "_inpainter_key_pool",
    "tts": "_tts_key_pool",
}

#: The per-key runtime fields ``live_key_stats`` reports (``APIKeyEntry`` attributes).
LIVE_STAT_FIELDS = ("success_count", "error_count", "times_used", "is_cooling_down", "last_error_time",
                    "cooldown")


def runtime_key_pool(pool_id):
    """The running client's ``APIKeyPool`` of *pool_id*, or None. Never imports unified_api_client:
    only a process that already runs (or ran) a job has the pools."""
    import sys

    module = sys.modules.get("unified_api_client")
    client = getattr(module, "UnifiedClient", None) if module is not None else None
    attr = RUNTIME_POOL_ATTRS.get(pool_id)
    if client is None or not attr:
        return None
    return getattr(client, attr, None)


def live_key_stats(pool_id):
    """``[{api_key, model, success_count, error_count, times_used, is_cooling_down, last_error_time,
    cooldown}, ...]`` of the running client's pool *pool_id* in pool order; [] when it has none."""
    pool = runtime_key_pool(pool_id)
    keys = list(getattr(pool, "keys", None) or ()) if pool is not None else []
    out = []
    for key in keys:
        row = {"api_key": getattr(key, "api_key", None), "model": getattr(key, "model", None)}
        for name in LIVE_STAT_FIELDS:
            row[name] = getattr(key, name, None)
        out.append(row)
    return out


def merge_live_stats(entries, live):
    """Pair config *entries* with *live* rows (same api_key and model, first unused match): a list
    of the live row (or None) per entry."""
    used = set()
    out = []
    for entry in entries:
        match = None
        for index, row in enumerate(live or ()):
            if index in used:
                continue
            if row.get("api_key") == entry.get("api_key") and row.get("model") == entry.get("model"):
                match = index
                break
        if match is None:
            out.append(None)
        else:
            used.add(match)
            out.append(live[match])
    return out
