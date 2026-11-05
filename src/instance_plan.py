"""instance_plan: which input files run in which Glossarion instance (multi-instance translation).

Shared GUI-free core used by the desktop (TranslatorGUI, instance_worker) and Glossarion Mobile.
No Qt, no subprocess: it only reads a config dict and the environment.

Every key row of every key pool (Translation ``config["multi_api_keys"]`` / ``APIKeyEntry``,
Fallback, Glossary and each dedicated pool, :data:`POOLS`) carries an ``instance`` number
(default 1). A run is split into instances only when multi-keys are on, the enabled Translation
keys carry at least two distinct instance numbers and at least two input files are selected;
otherwise everything runs as one instance, exactly as before (instance numbers are ignored).

An instance uses only the keys of its own number, in every pool (``config_for_instance``); a pool
without an enabled key of that instance is off for it, so its tasks use the instance's own
Translation keys. The main GUI key (``api_key``, "Use Main GUI Key as Fallback #1") is instance 1's:
an instance >= 2 never holds it. While a process (or a mobile instance runtime) runs one instance
of a multi-instance run, :func:`key_scope` names that instance and the API client narrows every key
list it is given or reads (:func:`scope_keys`, :func:`scoped_env`), whatever wrote it.

The user's last file -> instance choice is saved as ``config["instance_file_map"]``
({absolute input path: instance}, like ``manual_glossary_map``). Map lookups compare
``os.path.normcase(os.path.abspath(path))``; the functions return the caller's own path strings.

``GLOSSARION_INSTANCE`` is set inside an instance's environment (desktop child process / mobile
instance runtime); ``current_instance()`` reads it (1 when unset). ``GLOSSARION_KEY_INSTANCE``
narrows the API keys of a process to one instance (:func:`key_scope`).

The request-time backstop: under a key scope the API client refuses to send a key that belongs to
another instance only (:func:`foreign_key_owners`). Which instance owns which key is known by key
fingerprints (:func:`key_owners` of the unscoped config, handed to :func:`set_key_owners`, plus
every list :func:`scope_keys` narrows), so an instance process learns which keys are another
instance's without ever receiving them.

Rules: Python 3.10 compatible; stdlib only; never import PySide6, translator_gui or subprocess.
"""

import copy
import functools
import hashlib
import json
import math
import os
import threading

__all__ = [
    "CONFIG_FILE_MAP",
    "CONFIG_KEYS",
    "CONFIG_MAIN_KEY",
    "CONFIG_MAIN_KEY_FALLBACK",
    "CONFIG_USE_KEYS",
    "ENV_INSTANCE",
    "ENV_KEY_INSTANCE",
    "ENV_MAIN_KEY_FALLBACK",
    "KEY_ENV",
    "MAIN_KEY_ENV",
    "POOLS",
    "available_instances",
    "coerce_instance",
    "config_for_instance",
    "current_instance",
    "default_assignment",
    "foreign_key_owners",
    "has_enabled_key",
    "in_instance_process_tree",
    "instance_payloads_dir",
    "is_multi_instance",
    "key_fingerprint",
    "key_instance",
    "key_owners",
    "key_scope",
    "plan",
    "pool_switch",
    "register_instance_process",
    "scope_keys",
    "scoped_env",
    "set_key_owners",
    "unregister_instance_process",
]

#: Set to "1", "2", ... inside an instance's environment.
ENV_INSTANCE = "GLOSSARION_INSTANCE"
#: Set to the instance whose API keys a process may use while it runs one instance of a
#: multi-instance run: "N" in instance N's process (and its helper processes), "1" in the main
#: window while it translates instance 1's files. Unset: every key (single-instance runs).
ENV_KEY_INSTANCE = "GLOSSARION_KEY_INSTANCE"
#: config.json: {absolute input path: instance} of the last confirmed assignment.
CONFIG_FILE_MAP = "instance_file_map"
#: config.json: the Translation key pool and its toggle (key_pool_service.POOL_SPECS['main']).
CONFIG_KEYS = "multi_api_keys"
CONFIG_USE_KEYS = "use_multi_api_keys"

#: Every key pool (key_pool_service.POOL_SPECS): its config.json key list and on/off toggle, and the
#: environment variables the runtime exports it through (run_env / key_pools; unified_api_client
#: reads them): ``(config_key, toggle_key, use_envs, keys_envs)``.
POOLS = {
    "main": (CONFIG_KEYS, CONFIG_USE_KEYS, ("USE_MULTI_API_KEYS", "USE_MULTI_KEYS"), ("MULTI_API_KEYS",)),
    "fallback": ("fallback_keys", "use_fallback_keys", ("USE_FALLBACK_KEYS",), ("FALLBACK_KEYS",)),
    "glossary": ("glossary_keys", "use_glossary_keys", ("USE_GLOSSARY_KEYS",), ("GLOSSARY_API_KEYS",)),
    "glossary_refinement": ("glossary_refinement_keys", "use_glossary_refinement_keys",
                            ("USE_GLOSSARY_REFINEMENT_KEYS",), ("GLOSSARY_REFINEMENT_API_KEYS",)),
    "qa_scan": ("qa_scan_keys", "use_qa_scan_keys", ("USE_VISION_KEYS", "USE_QA_SCAN_KEYS"),
                ("VISION_API_KEYS", "QA_SCAN_API_KEYS")),
    "metadata": ("metadata_keys", "use_metadata_keys", ("USE_METADATA_KEYS",), ("METADATA_API_KEYS",)),
    "ai_truncation_detection": ("ai_truncation_detection_keys", "use_ai_truncation_detection_keys",
                                ("USE_AI_TRUNCATION_DETECTION_KEYS",), ("AI_TRUNCATION_DETECTION_API_KEYS",)),
    "rolling_summary": ("rolling_summary_keys", "use_rolling_summary_keys",
                        ("USE_ROLLING_SUMMARY_KEYS",), ("ROLLING_SUMMARY_API_KEYS",)),
    "truncation_retry": ("truncation_retry_keys", "use_truncation_retry_keys",
                         ("USE_TRUNCATION_RETRY_KEYS",), ("TRUNCATION_RETRY_API_KEYS",)),
    "inpainter": ("inpainter_keys", "use_inpainter_keys", ("USE_INPAINTER_KEYS",), ("INPAINTER_API_KEYS",)),
    "tts": ("tts_keys", "use_tts_keys", ("USE_TTS_KEYS",), ("TTS_API_KEYS",)),
}

#: The main GUI key (instance 1's): config.json's key and "Use Main GUI Key as Fallback #1".
CONFIG_MAIN_KEY = "api_key"
CONFIG_MAIN_KEY_FALLBACK = "use_main_key_fallback"
ENV_MAIN_KEY_FALLBACK = "USE_MAIN_KEY_FALLBACK"
#: The variables run_env exports the main GUI key through.
MAIN_KEY_ENV = ("API_KEY", "OPENAI_API_KEY", "OPENAI_OR_Gemini_API_KEY", "GEMINI_API_KEY")
#: Every variable that holds API keys or switches a key pool: an instance process takes them from
#: its own config snapshot, never from the main window's environment.
KEY_ENV = tuple(name for spec in POOLS.values() for name in spec[2] + spec[3]) + MAIN_KEY_ENV + (
    ENV_MAIN_KEY_FALLBACK,)

_LIST_ENV_POOL = {name: pool for pool, spec in POOLS.items() for name in spec[3]}
_TOGGLE_ENV_POOL = {name: pool for pool, spec in POOLS.items() for name in spec[2]}


def _valid_instance(value):
    """``value`` as an instance number (an integer >= 1), or None when it is not one."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value.strip()) if isinstance(value, str) else float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or not number.is_integer() or number < 1:
        return None
    return int(number)


def coerce_instance(value):
    """An instance number from a stored value: integers >= 1 (``2``, ``"2"``, ``2.0``) are kept,
    anything else (missing, ``0``, ``-1``, ``"abc"``, ``2.5``) reads as 1."""
    return _valid_instance(value) or 1


def key_instance(entry):
    """The instance a Translation key row belongs to (``entry["instance"]``, default 1)."""
    if isinstance(entry, dict):
        return coerce_instance(entry.get("instance"))
    return coerce_instance(getattr(entry, "instance", None))


def _key_enabled(entry):
    # The runtime pool's own test (APIKeyEntry.enabled / key_contexts.key_enabled_for_context)
    if isinstance(entry, dict):
        return bool(entry.get("enabled", True))
    return bool(getattr(entry, "enabled", True))


def has_enabled_key(rows):
    """True when the key-pool list ``rows`` (config dicts or ``APIKeyEntry`` objects) holds an
    enabled key. A pool without one is an empty pool: the runtime treats it as switched off."""
    if not isinstance(rows, (list, tuple)):
        return False
    return any(_key_enabled(entry) for entry in rows if isinstance(entry, dict) or hasattr(entry, "api_key"))


def _translation_keys(config):
    keys = config.get(CONFIG_KEYS) if isinstance(config, dict) else None
    return [entry for entry in (keys or []) if isinstance(entry, dict)]


def available_instances(config):
    """Sorted distinct instance numbers of the enabled Translation keys, or ``[1]`` when the
    Translation pool is off (``use_multi_api_keys``) or has no enabled key."""
    if not isinstance(config, dict) or not config.get(CONFIG_USE_KEYS, False):
        return [1]
    found = {key_instance(entry) for entry in _translation_keys(config) if _key_enabled(entry)}
    return sorted(found) or [1]


def _norm(path):
    return os.path.normcase(os.path.abspath(str(path)))


def _distinct_files(files):
    """``files`` without empty entries and without repeats of the same normalised path."""
    seen = set()
    kept = []
    for path in files or []:
        if not path:
            continue
        key = _norm(path)
        if key in seen:
            continue
        seen.add(key)
        kept.append(path)
    return kept


def is_multi_instance(config, files):
    """True when this run is split into instances: multi-keys on, the enabled Translation keys
    carry at least two instance numbers and at least two distinct input files are selected."""
    return len(available_instances(config)) >= 2 and len(_distinct_files(files)) >= 2


def default_assignment(files, instances, saved_map=None):
    """{file: instance} for the assignment dialog, in file order.

    A file keeps its saved instance (``saved_map``, i.e. ``config["instance_file_map"]``) while
    that instance is still in ``instances``; the other files are dealt round-robin over
    ``instances`` in file order. The same file listed twice gets the same instance.
    """
    pool = []
    for number in instances or []:
        number = _valid_instance(number)
        if number is not None and number not in pool:
            pool.append(number)
    if not pool:
        pool = [1]
    saved = {}
    if isinstance(saved_map, dict):
        for path, number in saved_map.items():
            number = _valid_instance(number)
            if path and number is not None:
                saved[_norm(path)] = number

    assignment = {}
    by_key = {}
    next_slot = 0
    for path in files or []:
        if not path or path in assignment:
            continue
        key = _norm(path)
        if key not in by_key:
            number = saved.get(key)
            if number not in pool:
                number = pool[next_slot % len(pool)]
                next_slot += 1
            by_key[key] = number
        assignment[path] = by_key[key]
    return assignment


def plan(files, assignment):
    """{instance: [files]} in instance order; each file once, in its original order.

    ``assignment`` is looked up by the caller's string first, then by normalised path; a file it
    does not mention runs in instance 1. Instances without files are left out.
    """
    assignment = assignment if isinstance(assignment, dict) else {}
    by_key = {}
    for path, number in assignment.items():
        if path:
            by_key.setdefault(_norm(path), coerce_instance(number))
    grouped = {}
    for path in _distinct_files(files):
        if path in assignment:
            number = coerce_instance(assignment[path])
        else:
            number = by_key.get(_norm(path), 1)
        grouped.setdefault(number, []).append(path)
    return {number: grouped[number] for number in sorted(grouped)}


def _instance_main_key(keys, model):
    """The ``(api_key, model)`` an instance >= 2 runs under in place of the main GUI key and model:
    its first enabled Translation key for the main ``model``, else its first enabled Translation
    key with that key's own model, so every request that falls back to "the main key" (all pool
    keys cooling down, single-key clients) sends a key with a model it serves."""
    enabled = [entry for entry in keys if _key_enabled(entry)]
    for entry in enabled:
        if str(entry.get("model") or "").strip() == str(model or "").strip():
            return str(entry.get("api_key") or ""), model
    if not enabled:
        return "", model
    own_model = str(enabled[0].get("model") or "").strip()
    return str(enabled[0].get("api_key") or ""), own_model or model


def config_for_instance(config, instance):
    """A deep copy of ``config`` as instance ``instance`` runs it: every key pool (:data:`POOLS`)
    holds only that instance's rows, and a pool other than Translation left without an enabled row
    is switched off (its tasks use the instance's own Translation keys).

    For an instance >= 2 the main GUI key, which is instance 1's, is replaced by the instance's own
    Translation key (:func:`_instance_main_key`) and "Use Main GUI Key as Fallback #1" is off, so
    no request of the instance can reach instance 1's key by any "main key" path. ``model`` stays
    unless none of the instance's keys serves it (then it is that key's model, so the stand-in key
    and the model always match); every other setting is copied unchanged. Raises ValueError when
    the instance has no enabled Translation key (it could not send a single request).
    """
    number = coerce_instance(instance)
    snapshot = copy.deepcopy(config) if isinstance(config, dict) else {}
    keys = [entry for entry in _translation_keys(snapshot) if key_instance(entry) == number]
    if not any(_key_enabled(entry) for entry in keys):
        raise ValueError(f"Instance {number} has no enabled Translation API key")
    for pool, (config_key, toggle_key, _use_envs, _keys_envs) in POOLS.items():
        if config_key not in snapshot:
            continue
        rows = snapshot.get(config_key)
        rows = [entry for entry in (rows if isinstance(rows, list) else [])
                if isinstance(entry, dict) and key_instance(entry) == number]
        snapshot[config_key] = rows
        if pool != "main" and snapshot.get(toggle_key) and not any(_key_enabled(entry) for entry in rows):
            snapshot[toggle_key] = False
    if number >= 2:
        main_key, main_model = _instance_main_key(keys, snapshot.get("model"))
        snapshot[CONFIG_MAIN_KEY] = main_key
        if main_model != snapshot.get("model"):
            snapshot["model"] = main_model
        snapshot[CONFIG_MAIN_KEY_FALLBACK] = False
    return snapshot


def current_instance():
    """This instance's number (``GLOSSARION_INSTANCE``); 1 outside a multi-instance child."""
    return coerce_instance(os.environ.get(ENV_INSTANCE))


def key_scope():
    """The instance whose API keys this process may use, or None for every key (single-instance).

    ``GLOSSARION_KEY_INSTANCE`` when set; else this instance's number when it is 2 or more (an
    instance >= 2 only exists inside a multi-instance run, so it never needs the variable)."""
    number = _valid_instance(os.environ.get(ENV_KEY_INSTANCE))
    if number is not None:
        return number
    number = current_instance()
    return number if number >= 2 else None


def scope_keys(keys, instance=None):
    """``keys`` (a key-pool list) as the key scope may use it: only the rows of ``instance``
    (default :func:`key_scope`); ``keys`` itself when there is no scope or it is not a list."""
    number = key_scope() if instance is None else coerce_instance(instance)
    if number is None or not isinstance(keys, (list, tuple)):
        return keys
    _note_key_owners(keys)  # the request guard learns whose keys the dropped rows are
    return [entry for entry in keys if key_instance(entry) == number]


def _env_rows(value):
    """The key rows of a key-list variable's value; None when it is not a JSON list."""
    try:
        rows = json.loads(value) if str(value).strip() not in ("", "null", "None") else []
    except (TypeError, ValueError):
        return None
    return rows if isinstance(rows, list) else None


def scoped_env(name, value, memory_rows=None, environ=None):
    """``value`` (the variable ``name`` as read from the environment) as the key scope may see it.

    Outside a key scope, and for variables that are not key-pool variables, ``value`` itself. Under a
    scope: a key list (``FALLBACK_KEYS``, ``GLOSSARY_API_KEYS`` ...) keeps only the scope's rows
    (an unreadable one reads ``"[]"``); a dedicated pool's switch ("1") reads "0" when neither the
    pool's key lists in ``environ`` (default ``os.environ``) nor ``memory_rows(pool)`` (the client's
    in-memory list) hold an enabled row of the scope, so its requests use the scope's own
    Translation keys instead of failing; ``USE_MAIN_KEY_FALLBACK`` reads "0" for an instance >= 2
    (the main GUI key is instance 1's). The Translation pool's switches pass through unchanged."""
    number = key_scope()
    if number is None or value is None:
        return value
    if name == ENV_MAIN_KEY_FALLBACK:
        return "0" if number >= 2 else value
    if name in _LIST_ENV_POOL:
        rows = _env_rows(value)
        if rows is None:
            return "[]"
        kept = scope_keys(rows, number)
        return value if len(kept) == len(rows) else json.dumps(kept)
    return pool_switch(name, value, memory_rows, environ, instance=number)


def pool_switch(name, value, memory_rows=None, environ=None, instance=None):
    """``value`` (a dedicated pool's switch ``name``, e.g. ``USE_INPAINTER_KEYS``, as read from the
    environment) as the runtime may use it: "1" reads "0" when neither the pool's key lists in
    ``environ`` (default ``os.environ``) nor ``memory_rows(pool)`` (the client's in-memory list)
    hold an enabled row (of ``instance`` when given). A pool whose keys are all disabled, or that
    has none, is an empty pool and counts as switched off: its requests take the normal route
    instead of failing, and the user's saved toggle is left as it is. Other variables, other
    values and the Translation pool's switches pass through unchanged."""
    pool = _TOGGLE_ENV_POOL.get(name)
    if pool is None or pool == "main" or value is None or str(value) != "1":
        return value
    environ = os.environ if environ is None else environ
    rows = []
    for list_name in POOLS[pool][3]:
        rows.extend(_env_rows(environ.get(list_name)) or [])
    if callable(memory_rows):
        try:
            rows.extend(memory_rows(pool) or [])
        except Exception:
            pass
    if instance is not None:
        rows = scope_keys([entry for entry in rows if isinstance(entry, dict)], instance)
    return value if has_enabled_key(rows) else "0"


# ---------------------------------------------------------------------------
# Request-time key guard: whose key is this?
# ---------------------------------------------------------------------------

#: key fingerprint -> the instances whose key rows (or main GUI key: instance 1) hold that key, as
#: this process knows them: :func:`set_key_owners` (the unscoped config, or for an instance process
#: the fingerprints the main window handed it) plus every key list :func:`scope_keys` narrowed.
_KEY_OWNERS = {}
_KEY_OWNERS_LOCK = threading.Lock()


@functools.lru_cache(maxsize=1024)
def key_fingerprint(key):
    """A stable fingerprint of an API key (never the key itself); "" for an empty key."""
    text = str(key or "").strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32] if text else ""


def _row_key(entry):
    value = entry.get("api_key") if isinstance(entry, dict) else getattr(entry, "api_key", None)
    return value if isinstance(value, str) else ""


def key_owners(config):
    """{fingerprint: sorted instances} of every key in every key pool of ``config`` (an unscoped
    config: the main window's), and of the main GUI key, which is instance 1's."""
    owners = {}
    if not isinstance(config, dict):
        return owners

    def add(key, number):
        fingerprint = key_fingerprint(key)
        if fingerprint:
            owners.setdefault(fingerprint, set()).add(number)

    for config_key, _toggle_key, _use_envs, _keys_envs in POOLS.values():
        rows = config.get(config_key)
        for entry in (rows if isinstance(rows, list) else []):
            if isinstance(entry, dict):
                add(_row_key(entry), key_instance(entry))
    add(config.get(CONFIG_MAIN_KEY), 1)
    return {fingerprint: sorted(numbers) for fingerprint, numbers in owners.items()}


def set_key_owners(owners):
    """Replace what this process knows about whose keys are whose (``{fingerprint: [instances]}``,
    :func:`key_owners`); returns the previous knowledge, for the undo."""
    fresh = {}
    for fingerprint, numbers in (owners or {}).items():
        valid = {coerce_instance(number) for number in (numbers or ())}
        if fingerprint and valid:
            fresh[str(fingerprint)] = valid
    global _KEY_OWNERS
    with _KEY_OWNERS_LOCK:
        previous, _KEY_OWNERS = _KEY_OWNERS, fresh
    return {fingerprint: sorted(numbers) for fingerprint, numbers in previous.items()}


def _note_key_owners(rows):
    for entry in rows:
        fingerprint = key_fingerprint(_row_key(entry))
        if fingerprint:
            number = key_instance(entry)
            owned = _KEY_OWNERS.get(fingerprint)
            if owned is None or number not in owned:
                with _KEY_OWNERS_LOCK:
                    _KEY_OWNERS.setdefault(fingerprint, set()).add(number)


def foreign_key_owners(key):
    """Under a key scope: the instances that own ``key`` when the scope's own instance is not one
    of them (the key is another instance's, or the main GUI key inside an instance >= 2); else
    ``()``. A key no pool of any instance holds (a dedicated QA key, an OCR / DeepL credential,
    an endpoint's dummy key) is nobody's and passes. No scope: ``()`` without any lookup."""
    number = key_scope()
    if number is None or not _KEY_OWNERS:
        return ()
    owners = _KEY_OWNERS.get(key_fingerprint(key))
    if not owners or number in owners:
        return ()
    return tuple(sorted(owners))


# ---------------------------------------------------------------------------
# Instance processes of the main window (desktop)
# ---------------------------------------------------------------------------

#: pids of this process's running instance processes (instance_runs registers them).
_INSTANCE_PIDS = set()


def register_instance_process(pid):
    """An instance process of this window's multi-instance run started."""
    if pid:
        _INSTANCE_PIDS.add(int(pid))


def unregister_instance_process(pid):
    """That instance process has ended."""
    _INSTANCE_PIDS.discard(int(pid or 0))


def in_instance_process_tree(proc):
    """True for a process (a ``psutil.Process``) that is one of this window's instance processes
    or was started by one: a "kill every child of this process" sweep (an instance 1 Stop, the
    Manga Stop) leaves them to their own instance's Stop."""
    if not _INSTANCE_PIDS:
        return False
    try:
        if proc.pid in _INSTANCE_PIDS:
            return True
        return any(parent.pid in _INSTANCE_PIDS for parent in proc.parents())
    except Exception:
        return False


def instance_payloads_dir(root):
    """``root`` itself for instance 1; ``root/Instance N`` inside instance N of a multi-instance
    run, so its payload folders never mix with another instance's."""
    number = current_instance()
    return root if number <= 1 else os.path.join(root, f"Instance {number}")
