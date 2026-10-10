"""Multi-Key Manager (``/settings/keys``, ``/settings/keys/<pool>``; UI_SPEC §4.12).

All eleven pools of the desktop ``MultiAPIKeyDialog`` (``key_pool_service.POOL_SPECS``: main,
fallback, glossary and the eight dedicated pools): rotation card (force rotation, every N
requests), pool chips with the pool switch / description / count, key cards in rotation order
(``ReorderableListView`` + drag handles), tap → ``KeyEditor``, long-press → selection with bulk
enable / disable / test / remove / set contexts / move / copy to pool, pool ⋯ (Clear all keys
with confirm, import into / export this pool), app bar ⋯ (import / export all pools in the
``glossarion-key-pools`` v1 format through FileBridge, with a plain-text warning; Refusal
patterns) and the two desktop-only rows (Lock mouse wheel, Key list zoom) disabled with a
ReasonChip. Every write goes through ``MobileConfigStore`` (sparse: only the touched pool list or
toggle changes); the runtime pools are applied by ``key_pools.apply_key_pools_to_runtime`` when a
job starts.

The pure part (``KeyBackend``, ``KeysController``, request-parameter helpers) imports without
Flet; ``KeysScreen`` below needs it. Key tests run off the UI loop through
``key_pool_service.run_key_test`` inside the environment a run would get
(``endpoints.run_in_run_env``) with every key pool switched off, so the probe sends the key under
test; each key gets its own run environment (the engine lock is held for one probe at a time) and
its result is stored on that key even if the list changed meanwhile.
"""

from __future__ import annotations

import asyncio
import copy
import importlib
import inspect
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

__all__ = [
    "IMPORT_EXTENSIONS",
    "ImportPlan",
    "KeyBackend",
    "KeysController",
    "KeysScreen",
    "POOL_TO_SLUG",
    "PoolSpec",
    "SLUG_TO_POOL",
    "parse_request_params",
    "request_param_display",
]

log = logging.getLogger("glossarion.keys")

#: Route slug (router ``KEY_POOLS``) -> key_pool_service pool id.
SLUG_TO_POOL = {
    "translation": "main",
    "fallback": "fallback",
    "glossary": "glossary",
    "glossary_refinement": "glossary_refinement",
    "qa_vision": "qa_scan",
    "metadata": "metadata",
    "ai_truncation": "ai_truncation_detection",
    "rolling_summary": "rolling_summary",
    "truncation_retry": "truncation_retry",
    "inpainter": "inpainter",
    "tts": "tts",
}
POOL_TO_SLUG = {pool: slug for slug, pool in SLUG_TO_POOL.items()}
IMPORT_EXTENSIONS = ("json",)
EXPORT_FORMAT = "glossarion-key-pools"
PLAINTEXT_WARNING = ("The exported file contains your API keys in plain text. Anyone who gets the file can use "
                     "them. Keep it private and delete it after importing it elsewhere.")
SERVICE_MISSING = "Needs the shared key pool service (key_pool_service), which is not in this build"
#: A key test that would run in a running job's environment with its key pools on.
KEY_TEST_BUSY = "A running job is using the key pools. Test this key when it finishes."
#: Run-environment switches that make a new UnifiedClient send pool keys (Translation pool
#: rotation, Fallback retries) instead of the key under test.
_POOL_RUNTIME_ENV = ("USE_MULTI_API_KEYS", "USE_FALLBACK_KEYS")

# The config keys of each pool (the config.json data contract shared with key_pools.py and the
# settings schema), used only while key_pool_service is not importable.
_CONFIG_KEYS = (
    ("main", "Translation Keys", "Translation", "multi_api_keys", "use_multi_api_keys"),
    ("fallback", "Fallback Keys", "Fallback", "fallback_keys", "use_fallback_keys"),
    ("glossary", "Glossary Keys", "Glossary", "glossary_keys", "use_glossary_keys"),
    ("glossary_refinement", "Refinement Keys", "Refinement", "glossary_refinement_keys", "use_glossary_refinement_keys"),
    ("qa_scan", "Vision Keys", "Vision", "qa_scan_keys", "use_qa_scan_keys"),
    ("metadata", "Metadata Keys", "Metadata", "metadata_keys", "use_metadata_keys"),
    ("ai_truncation_detection", "QA Scan Keys (for AI Truncation detection)", "QA scan",
     "ai_truncation_detection_keys", "use_ai_truncation_detection_keys"),
    ("rolling_summary", "Rolling Summary Keys", "Rolling summary", "rolling_summary_keys", "use_rolling_summary_keys"),
    ("truncation_retry", "Truncation Retry Keys", "Truncation retry", "truncation_retry_keys", "use_truncation_retry_keys"),
    ("inpainter", "Image Gen / Edit Keys", "Image gen/edit", "inpainter_keys", "use_inpainter_keys"),
    ("tts", "Audio / TTS Keys", "Audio/TTS", "tts_keys", "use_tts_keys"),
)


# ---- request parameters (request_parameters, shared) ------------------------------------------------


def request_param_display(value: Any) -> str:
    try:
        from request_parameters import display_parameter_value

        return display_parameter_value(value)
    except Exception:
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def parse_request_params(pairs: Iterable[tuple]) -> dict:
    """Editor rows -> request parameters (``request_parameters`` rules); ValueError on a reserved
    or duplicate name, so nothing is silently dropped."""
    from request_parameters import RESERVED_REQUEST_PARAMETERS, normalize_request_parameters, parse_parameter_value

    raw: dict = {}
    for name, text in pairs:
        name = str(name or "").strip()
        if not name:
            if str(text or "").strip():
                raise ValueError("Every request parameter needs a name")
            continue
        if name.lower() in RESERVED_REQUEST_PARAMETERS:
            raise ValueError(f"'{name}' is controlled by Glossarion and cannot be overridden")
        if name in raw:
            raise ValueError(f"'{name}' is listed twice")
        raw[name] = parse_parameter_value(text)
    clean = normalize_request_parameters(raw)
    dropped = [n for n in raw if n not in clean]
    if dropped:
        raise ValueError(f"Not a JSON value: {', '.join(dropped)}")
    return clean


# ---- key_pool_service binding ------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolSpec:
    id: str
    slug: str
    title: str
    label: str
    config_key: str
    toggle_key: str
    description: str = ""
    contexts: tuple = ()
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ImportPlan:
    items: list = field(default_factory=list)  # [(pool_id, keys, enabled or None)]
    skipped: int = 0
    unknown: tuple = ()
    legacy: bool = False  # a flat key list (appended to the Translation pool on desktop)
    error: Optional[str] = None
    duplicates: int = 0  # legacy keys the pool already held (skipped by ``apply_import``)

    @property
    def total(self) -> int:
        return sum(len(keys) for _p, keys, _e in self.items)

    def summary_lines(self, titles: Mapping[str, str]) -> list:
        return [f"{titles.get(pool, pool)}: {len(keys)} key(s)" for pool, keys, _enabled in self.items]


def _accepts(fn: Callable[..., Any], name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` with only the keyword arguments its signature accepts."""
    return fn(*args, **{k: v for k, v in kwargs.items() if _accepts(fn, k)})


def _contexts_for(pool_id: str) -> tuple:
    try:
        from key_contexts import POOL_CONTEXTS

        return tuple(POOL_CONTEXTS.get(pool_id, ()))
    except Exception:
        return ()


def context_labels() -> dict:
    try:
        from key_contexts import CONTEXT_LABELS

        return dict(CONTEXT_LABELS)
    except Exception:
        return {}


class KeyBackend:
    """``key_pool_service`` (U4, extracted from ``multi_api_key_manager``); desktop classes as the
    fallback for entry construction (``APIKeyEntry``) and refusal defaults."""

    def __init__(self, module: Any = None, *, module_name: str = "key_pool_service") -> None:
        self.error: Optional[str] = None
        if module is None:
            try:
                if module_name == "key_pool_service":
                    import key_pool_service as module  # literal: tools/collect_backend.py bundles it
                else:
                    module = importlib.import_module(module_name)
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                module = None
        self.module = module

    @property
    def available(self) -> bool:
        return self.module is not None

    def fn(self, *names: str) -> Optional[Callable[..., Any]]:
        if self.module is None:
            return None
        for name in names:
            candidate = getattr(self.module, name, None)
            if callable(candidate):
                return candidate
        return None

    # ---- pools -----------------------------------------------------------------------------------

    def pool_specs(self) -> list:
        raw = getattr(self.module, "POOL_SPECS", None) if self.module is not None else None
        fallback = {pid: (title, label, cfg, toggle) for pid, title, label, cfg, toggle in _CONFIG_KEYS}
        specs: list = []
        items: list = []
        if isinstance(raw, Mapping):
            items = list(raw.items())
        elif isinstance(raw, (list, tuple)):
            for spec in raw:
                pid = spec.get("id") or spec.get("name") if isinstance(spec, Mapping) else getattr(spec, "id", None)
                items.append((pid, spec))
        seen: set = set()
        for pid, spec in items:
            if pid not in SLUG_TO_POOL.values():
                continue
            get = (spec.get if isinstance(spec, Mapping) else (lambda k, d=None, s=spec: getattr(s, k, d)))
            title, label, cfg, toggle = fallback[pid]
            specs.append(PoolSpec(
                id=pid, slug=POOL_TO_SLUG[pid], title=str(get("title", title) or title),
                label=str(get("label", label) or label), config_key=str(get("config_key", cfg) or cfg),
                toggle_key=str(get("toggle_key", toggle) or toggle), description=str(get("description", "") or ""),
                contexts=_contexts_for(pid),
                extra={k: v for k, v in (spec.items() if isinstance(spec, Mapping) else [])
                       if k not in ("title", "label", "config_key", "toggle_key", "description")},
            ))
            seen.add(pid)
        for pid, title, label, cfg, toggle in _CONFIG_KEYS:
            if pid not in seen:
                specs.append(PoolSpec(pid, POOL_TO_SLUG[pid], title, label, cfg, toggle, "", _contexts_for(pid)))
        order = [pid for pid, *_rest in _CONFIG_KEYS]
        specs.sort(key=lambda s: order.index(s.id))
        return specs

    # ---- entries ---------------------------------------------------------------------------------

    def new_entry(self, api_key: str = "", model: str = "", **fields: Any) -> dict:
        fn = self.fn("new_key_entry")
        if fn is not None:
            entry = _call(fn, api_key, model, **fields)
            return dict(entry if isinstance(entry, Mapping) else getattr(entry, "to_dict")())
        from multi_api_key_manager import APIKeyEntry  # GLOSSARION_HEADLESS_KEY_MANAGER import

        allowed = set(inspect.signature(APIKeyEntry.__init__).parameters) - {"self", "api_key", "model"}
        return APIKeyEntry(api_key, model, **{k: v for k, v in fields.items() if k in allowed}).to_dict()

    def validate(self, entry: Mapping[str, Any], pool_id: str = "main") -> tuple:
        """(normalized entry, None) or (None, message)."""
        fn = self.fn("validate_entry", "validate_key_entry")
        if fn is not None:
            result = _call(fn, dict(entry), pool=pool_id, pool_id=pool_id)
            if isinstance(result, tuple) and len(result) == 2:
                first, second = result
                if isinstance(first, Mapping) or first is None:
                    return (dict(first) if first is not None else None), second
                if isinstance(first, bool):  # (ok, error-or-entry)
                    return (dict(entry) if first else None), (None if first else str(second))
            if isinstance(result, str):
                return None, result
            if isinstance(result, Mapping):
                return dict(result), None
            if result is None or result is True:
                return dict(entry), None
            return None, str(result)
        if not str(entry.get("model") or "").strip():
            return None, "Please enter a model name"
        try:
            from multi_api_key_manager import APIKeyEntry

            normalized = APIKeyEntry.from_dict(dict(entry)).to_dict()
        except Exception as exc:
            return None, str(exc)
        merged = dict(entry)
        merged.update(normalized)
        return merged, None

    def find_duplicate(self, keys: Sequence[Mapping[str, Any]], entry: Mapping[str, Any]) -> Optional[int]:
        """``key_pool_service.find_duplicate_key``: index of an identical key in ``keys`` or None."""
        fn = self.fn("find_duplicate_key")
        if fn is None:
            return None
        try:
            index = fn([dict(k) for k in keys], dict(entry))
        except Exception:
            log.debug("duplicate check failed", exc_info=True)
            return None
        return index if isinstance(index, int) else None

    # ---- import / export --------------------------------------------------------------------------

    def export_payload(self, config: Mapping[str, Any], pools: Optional[Sequence[str]] = None) -> Optional[dict]:
        fn = self.fn("export_pools", "export_key_pools")
        if fn is None:
            return None
        payload = _call(fn, dict(config), pools=list(pools) if pools else None)
        if isinstance(payload, str):
            payload = json.loads(payload)
        payload = dict(payload or {})
        if pools and isinstance(payload.get("pools"), Mapping):
            payload["pools"] = {k: v for k, v in payload["pools"].items() if k in set(pools)}
        return payload

    def import_plan(self, payload: Any, config: Mapping[str, Any]) -> ImportPlan:
        fn = self.fn("import_pools", "plan_import", "import_key_pools")
        if fn is None:
            return ImportPlan(error=SERVICE_MISSING)
        cfg = copy.deepcopy(dict(config))
        try:
            if _accepts(fn, "config"):
                result = _call(fn, payload, config=cfg, apply=False, dry_run=True)
            else:
                result = _call(fn, payload, cfg, apply=False, dry_run=True)
        except Exception as exc:
            return ImportPlan(error=str(exc))
        return _normalize_plan(result, payload)

    # ---- tests --------------------------------------------------------------------------------------

    def run_test(self, entry: Mapping[str, Any], pool_id: str, *, timeout: Optional[float] = None) -> dict:
        from glossarion_mobile.ui.screens.key_editor import normalize_test_result

        fn = self.fn("run_key_test", "test_key")
        if fn is None:
            return {"ok": None, "status": "untestable", "message": SERVICE_MISSING}
        try:
            return normalize_test_result(_call(fn, dict(entry), pool_id, timeout=timeout))
        except Exception as exc:
            return {"ok": False, "status": "error", "message": str(exc)}

    def refusal_defaults(self) -> list:
        """``key_pool_service.default_refusal_patterns()``: the list unified_api_client uses too."""
        fn = self.fn("default_refusal_patterns")
        if fn is not None:
            return list(fn())
        value = getattr(self.module, "DEFAULT_REFUSAL_PATTERNS", None) if self.module is not None else None
        if isinstance(value, (list, tuple)) and value:
            return [str(v) for v in value]
        raise RuntimeError(SERVICE_MISSING)


def _normalize_plan(result: Any, payload: Any) -> ImportPlan:
    if isinstance(result, ImportPlan):
        return result
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], str) and not result[0]:
        return ImportPlan(error=result[1])
    data: Any = result
    if not isinstance(data, Mapping) and data is not None and not isinstance(data, (list, tuple)):
        data = {k: getattr(data, k) for k in ("items", "plan", "pools", "skipped", "unknown", "unknown_pools",
                                              "legacy", "error") if hasattr(data, k)}
    plan = ImportPlan()
    if isinstance(data, (list, tuple)):
        data = {"items": data}
    if not isinstance(data, Mapping):
        return ImportPlan(error="Invalid or unrecognized key file format")
    plan.error = data.get("error") or None
    plan.skipped = int(data.get("skipped") or data.get("total_skipped") or 0)
    plan.unknown = tuple(data.get("unknown") or data.get("unknown_pools") or ())
    plan.legacy = bool(data.get("legacy", isinstance(payload, list)))
    raw_items = data.get("items", data.get("plan", data.get("pools")))
    items: list = []
    if isinstance(raw_items, Mapping):
        for pool, value in raw_items.items():
            if isinstance(value, Mapping):
                items.append((pool, list(value.get("keys") or []), value.get("enabled")))
            else:
                items.append((pool, list(value or []), None))
    elif isinstance(raw_items, (list, tuple)):
        for item in raw_items:
            if isinstance(item, Mapping):
                items.append((item.get("pool") or item.get("pool_name"), list(item.get("keys") or []), item.get("enabled")))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                items.append((item[0], list(item[1] or []), item[2] if len(item) > 2 else None))
    plan.items = [(p, [dict(k) for k in keys if isinstance(k, Mapping)], e) for p, keys, e in items if p]
    if not plan.items and plan.error is None:
        plan.error = "No recognizable pools found in file"
    return plan


# ---- controller --------------------------------------------------------------------------------------


#: ``key_pool_service.key_tree_status`` tags -> the card's status chip (colour) id.
_TAG_STATUS = {"testing": "testing", "disabled": "disabled", "passed": "passed", "failed": "failed",
               "timeout": "failed", "error": "failed", "ratelimited": "cooling", "cooling": "cooling",
               "active": "active"}
#: The desktop status texts shown as such on the card (emoji / "Error: …" prefix dropped by the chip).
_TAG_TEXT = {"timeout": "Timed out", "ratelimited": "Rate limited", "error": "Error"}


def _status_key(entry: Mapping[str, Any], live: Optional[Mapping[str, Any]] = None) -> Any:
    """The attributes ``key_tree_status`` reads: the config entry, with the running client's
    cooldown state (``live_key_stats``) when a job has used the pool."""
    import types

    live = live or {}
    cooldown = live.get("cooldown") if live.get("cooldown") is not None else entry.get("cooldown", 60)
    try:
        cooldown = int(cooldown)
    except (TypeError, ValueError):
        cooldown = 60
    return types.SimpleNamespace(
        last_test_result=entry.get("last_test_result"), last_test_message=entry.get("last_test_message"),
        enabled=bool(entry.get("enabled", True)), is_cooling_down=bool(live.get("is_cooling_down")),
        last_error_time=live.get("last_error_time"), cooldown=cooldown)


def key_status_info(entry: Mapping[str, Any], live: Optional[Mapping[str, Any]] = None) -> tuple:
    """``(status id, text)`` of a key card: the desktop key tree's rule
    (``key_pool_service.key_tree_status``: Disabled / Passed / Failed / Timed out / Rate limited /
    Error / Cooling (Ns) / Active) plus "encrypted" for an undecryptable key."""
    if str(entry.get("api_key") or "").startswith("ENC:"):
        return "encrypted", _STATUS_TEXT["encrypted"]
    try:
        from key_pool_service import key_tree_status

        text, tags = key_tree_status(_status_key(entry, live))
        tag = tags[0] if tags else "active"
    except Exception:  # backend missing: the stored test result alone
        result = entry.get("last_test_result")
        tag = ("disabled" if not entry.get("enabled", True) else
               {"passed": "passed", "failed": "failed", "error": "error", "timeout": "timeout",
                "rate_limited": "ratelimited"}.get(str(result), "active"))
        text = ""
    status = _TAG_STATUS.get(tag, "active")
    if tag == "cooling" and text:
        return status, text  # "Cooling (42s)"
    return status, _TAG_TEXT.get(tag) or _STATUS_TEXT.get(status, status)


def key_status(entry: Mapping[str, Any], live: Optional[Mapping[str, Any]] = None) -> str:
    """Card status: active · disabled · passed · failed · cooling · encrypted."""
    return key_status_info(entry, live)[0]


def key_overrides_text(entry: Mapping[str, Any]) -> str:
    """The per-key overrides the desktop key tree has as columns, when not the default:
    "⌛ 90s · limit 4096 · T 0.3 · delay 2s"."""
    parts = []
    try:
        cooldown = int(entry.get("cooldown")) if entry.get("cooldown") is not None else 60
    except (TypeError, ValueError):
        cooldown = 60
    if cooldown != 60:
        parts.append(f"⌛ {cooldown}s")
    try:
        limit = int(entry.get("individual_output_token_limit") or 0)
    except (TypeError, ValueError):
        limit = 0
    if limit > 0:
        parts.append(f"limit {limit}")
    temperature = entry.get("individual_key_temperature")
    if temperature not in (None, ""):
        parts.append(f"T {temperature}")
    try:
        delay = float(entry.get("api_call_delay") or 0)
    except (TypeError, ValueError):
        delay = 0.0
    if delay > 0:
        parts.append(f"delay {delay:g}s")
    return " · ".join(parts)


def excluded_model_reason(model: Any) -> Optional[tuple]:
    """``(reason, detail)`` for a key whose model is a route excluded on mobile (``model_catalog.model_block``;
    such keys arrive with a desktop config import and fail at run time), else None."""
    if not model:
        return None
    try:
        from glossarion_mobile.services.model_catalog import model_block

        return model_block(str(model))
    except Exception:
        return None


def key_counts_text(entry: Mapping[str, Any], live: Optional[Mapping[str, Any]] = None) -> str:
    """"✅ 12 · ❌ 1": the success / error counts (the running client's, else the stored ones)."""
    source = live if live else entry
    try:
        success = int(source.get("success_count") or 0)
        errors = int(source.get("error_count") or 0)
    except (TypeError, ValueError):
        return ""
    if not success and not errors:
        return ""
    return f"✅ {success} · ❌ {errors}"


class KeysController:
    """Pool reads and sparse writes over ``MobileConfigStore`` (Flet-free; tests use it directly)."""

    def __init__(self, store: Any, backend: Optional[KeyBackend] = None, *,
                 test_runner: Optional[Callable[..., Any]] = None, clock: Callable[[], float] = time.time) -> None:
        self.store = store
        self.backend = backend or KeyBackend()
        self.clock = clock
        self._test_runner = test_runner  # (config, fn) -> fn() result inside the run env
        self._specs: Optional[list] = None
        self._lock = threading.RLock()

    # ---- specs ----------------------------------------------------------------------------------------

    @property
    def specs(self) -> list:
        if self._specs is None:
            self._specs = self.backend.pool_specs()
        return self._specs

    def spec(self, name: str) -> PoolSpec:
        pid = SLUG_TO_POOL.get(name, name)
        for spec in self.specs:
            if spec.id == pid:
                return spec
        raise KeyError(name)

    def titles(self) -> dict:
        return {s.id: s.title for s in self.specs}

    # ---- reads -------------------------------------------------------------------------------------------

    def keys(self, pool: str) -> list:
        value = self.store.get(self.spec(pool).config_key, [])
        return [dict(k) for k in value if isinstance(k, Mapping)] if isinstance(value, list) else []

    def count(self, pool: str) -> int:
        return len(self.keys(pool))

    def enabled(self, pool: str) -> bool:
        return bool(self.store.effective(self.spec(pool).toggle_key) if hasattr(self.store, "effective")
                    else self.store.get(self.spec(pool).toggle_key, False))

    def setting(self, key: str, default: Any = None) -> Any:
        value = self.store.effective(key) if hasattr(self.store, "effective") else None
        return default if value is None else value

    # ---- writes ---------------------------------------------------------------------------------------

    def set_keys(self, pool: str, keys: Sequence[Mapping[str, Any]]) -> None:
        self.store.set(self.spec(pool).config_key, [dict(k) for k in keys])

    def set_enabled(self, pool: str, value: bool) -> None:
        self.store.set(self.spec(pool).toggle_key, bool(value))

    def set_setting(self, key: str, value: Any) -> None:
        self.store.set(key, value)

    def new_entry(self, pool: str = "main", api_key: str = "", model: str = "") -> dict:
        return self.backend.new_entry(api_key or "", model or "")

    def current_key_entry(self, pool: str = "main") -> dict:
        """A new entry with the main API key and model (desktop "Copy Current Key":
        ``MultiAPIKeyDialog._copy_current_settings`` fills the Add Key form from the main window)."""
        api_key = str(self.store.get("api_key", "") or "") if self.store is not None else ""
        model = ""
        if self.store is not None:
            effective = getattr(self.store, "effective", None)
            model = str((effective("model") if callable(effective) else self.store.get("model", "")) or "")
        return self.new_entry(pool, api_key, model)

    def live_stats(self, pool: str) -> list:
        """The running client's per-key stats paired with this pool's keys (None: no runtime row)."""
        keys = self.keys(pool)
        try:
            import key_pool_service as kps

            live = kps.live_key_stats(self.spec(pool).id)
            return kps.merge_live_stats(keys, live)
        except Exception:
            return [None] * len(keys)

    def set_key_fields(self, pool: str, indices: Iterable[int], values: Mapping[str, Any]) -> tuple:
        """Bulk per-key edit (the desktop shared key menu: Change Model, Set Cooldown, Set / Clear
        Output Token Limit, Temperature, API Call Delay, Individual Endpoint, Request Parameters):
        ``values`` replace those fields on every selected key, each key is validated like the
        KeyEditor's save (``validate_entry``). Returns ``(changed, [errors])``."""
        errors: list = []
        with self._lock:
            keys = self.keys(pool)
            changed = 0
            for i in sorted({int(i) for i in indices}):
                if not 0 <= i < len(keys):
                    continue
                entry = dict(keys[i])
                entry.update(copy.deepcopy(dict(values)))
                clean, error = self.backend.validate(entry, self.spec(pool).id)
                if clean is None:
                    errors.append(f"#{i + 1}: {error or 'invalid'}")
                    continue
                if clean != keys[i]:
                    keys[i] = clean
                    changed += 1
            if changed:
                self.set_keys(pool, keys)
        return changed, errors

    def add_key(self, pool: str, entry: Mapping[str, Any]) -> tuple:
        """``(index, None)`` or ``(None, error)``; an exact duplicate of a key in the pool is refused."""
        clean, error = self.backend.validate(entry, self.spec(pool).id)
        if clean is None:
            return None, error
        with self._lock:
            keys = self.keys(pool)
            duplicate = self.backend.find_duplicate(keys, clean)
            if duplicate is not None:
                return None, (f"This key is already in {self.spec(pool).title} "
                              f"(#{duplicate + 1}: {self.key_label(keys[duplicate])}).")
            keys.append(clean)
            self.set_keys(pool, keys)
            return len(keys) - 1, None

    @staticmethod
    def key_label(entry: Mapping[str, Any]) -> str:
        """``sk-…a1B2 · gpt-6``: a key as feedback names it (never the whole secret)."""
        from glossarion_mobile.ui.settings.model import mask_secret

        api_key = str(entry.get("api_key") or "").strip()
        model = str(entry.get("model") or "").strip()
        return f"{mask_secret(api_key) if api_key else 'No key'} · {model}"

    def saved_message(self, pool: str, entry: Mapping[str, Any], *, added: bool) -> str:
        """The confirmation after the key editor closed (desktop: "Added key for model: …" plus the
        ``added_key_extra_info`` suffix); it names the key and the pool."""
        if not added:
            return f"Saved key {self.key_label(entry)}"
        extra = ""
        fn = self.backend.fn("added_key_extra_info")
        if fn is not None:
            endpoint = entry.get("azure_endpoint") if entry.get("use_individual_endpoint") else None
            try:
                extra = str(fn(entry.get("google_credentials"), endpoint) or "")
            except Exception:
                extra = ""
        return f"Added key {self.key_label(entry)} to {self.spec(pool).title}{extra}"

    def update_key(self, pool: str, index: int, entry: Mapping[str, Any]) -> Optional[str]:
        clean, error = self.backend.validate(entry, self.spec(pool).id)
        if clean is None:
            return error or "Invalid key"
        with self._lock:
            keys = self.keys(pool)
            if not 0 <= index < len(keys):
                return "This key no longer exists"
            keys[index] = clean
            self.set_keys(pool, keys)
        return None

    def remove(self, pool: str, indices: Iterable[int]) -> list:
        with self._lock:
            keys = self.keys(pool)
            drop = sorted({i for i in indices if 0 <= i < len(keys)})
            removed = [(i, keys[i]) for i in drop]
            self.set_keys(pool, [k for i, k in enumerate(keys) if i not in set(drop)])
            return removed

    def restore(self, pool: str, removed: Sequence[tuple]) -> None:
        """Undo of ``remove`` / ``clear``: the keys go back to their old positions."""
        with self._lock:
            keys = self.keys(pool)
            for index, entry in sorted(removed, key=lambda item: item[0]):
                keys.insert(min(index, len(keys)), dict(entry))
            self.set_keys(pool, keys)

    def clear(self, pool: str) -> list:
        return self.remove(pool, range(self.count(pool)))

    def move(self, pool: str, old_index: int, new_index: int) -> bool:
        with self._lock:
            keys = self.keys(pool)
            if not 0 <= old_index < len(keys) or old_index == new_index:
                return False
            item = keys.pop(old_index)
            keys.insert(max(0, min(new_index, len(keys))), item)
            self.set_keys(pool, keys)
            return True

    def set_keys_enabled(self, pool: str, indices: Iterable[int], value: bool) -> int:
        with self._lock:
            keys = self.keys(pool)
            changed = 0
            for i in indices:
                if 0 <= i < len(keys) and bool(keys[i].get("enabled", True)) != bool(value):
                    keys[i]["enabled"] = bool(value)
                    changed += 1
            if changed:
                self.set_keys(pool, keys)
            return changed

    def context_states(self, pool: str, indices: Sequence[int]) -> dict:
        """context -> True (all serve it) · False (none) · None (mixed) across ``indices``."""
        keys = self.keys(pool)
        selected = [keys[i] for i in indices if 0 <= i < len(keys)]
        out: dict = {}
        for context in self.spec(pool).contexts:
            values = {context not in set(k.get("disabled_contexts") or ()) for k in selected}
            out[context] = values.pop() if len(values) == 1 else None
        return out

    def set_disabled_contexts(self, pool: str, indices: Iterable[int], changes: Mapping[str, bool]) -> int:
        """``changes``: context -> enabled; contexts not listed stay as they are (tri-state)."""
        try:
            from key_contexts import normalize_disabled_contexts
        except Exception:  # pragma: no cover - backend missing
            normalize_disabled_contexts = lambda v: sorted(set(v or ()))  # noqa: E731
        with self._lock:
            keys = self.keys(pool)
            changed = 0
            for i in indices:
                if not 0 <= i < len(keys):
                    continue
                disabled = set(keys[i].get("disabled_contexts") or ())
                for context, enabled in changes.items():
                    if enabled:
                        disabled.discard(context)
                    else:
                        disabled.add(context)
                new = normalize_disabled_contexts(sorted(disabled))
                if new != list(keys[i].get("disabled_contexts") or []):
                    keys[i]["disabled_contexts"] = new
                    changed += 1
            if changed:
                self.set_keys(pool, keys)
            return changed

    def copy_to(self, pool: str, indices: Iterable[int], target: str, *, move: bool = False) -> int:
        if self.spec(pool).id == self.spec(target).id:
            return 0
        with self._lock:
            keys = self.keys(pool)
            picked = sorted({i for i in indices if 0 <= i < len(keys)})
            if not picked:
                return 0
            dest = self.keys(target)
            dest.extend(copy.deepcopy(keys[i]) for i in picked)
            values = {self.spec(target).config_key: dest}
            if move:
                values[self.spec(pool).config_key] = [k for i, k in enumerate(keys) if i not in set(picked)]
            self.store.set_many(values)
            return len(picked)

    # ---- tests -----------------------------------------------------------------------------------------

    @staticmethod
    def _find_tested_key(keys: Sequence[Mapping[str, Any]], index: int, model: Optional[str],
                         api_key: Optional[str]) -> Optional[int]:
        """Where a test result belongs: the key tested (same api_key and model), at ``index`` when it
        is still there, else wherever it moved while the batch ran (the desktop updates the key
        object); None when it was removed."""
        def same(entry: Mapping[str, Any]) -> bool:
            return ((model is None or entry.get("model") == model)
                    and (api_key is None or entry.get("api_key") == api_key))

        if 0 <= index < len(keys) and same(keys[index]):
            return index
        if api_key is None:
            return None
        return next((i for i, entry in enumerate(keys) if same(entry)), None)

    def record_test(self, pool: str, index: int, result: Mapping[str, Any], *, model: Optional[str] = None,
                    api_key: Optional[str] = None) -> bool:
        """Persist the desktop test fields (``last_test_result/time/message``) on the key tested."""
        status = result.get("status")
        if status in ("untestable", "busy", None):
            return False
        # key_pool_service.run_key_test reports the desktop value (passed / failed / error /
        # rate_limited / timeout); other runners only give a status
        stored = result.get("last_test_result") or (
            "passed" if status == "passed" else status if status in ("rate_limited", "error", "timeout") else "failed")
        with self._lock:
            keys = self.keys(pool)
            target = self._find_tested_key(keys, index, model, api_key)
            if target is None:
                return False
            keys[target]["last_test_result"] = stored
            keys[target]["last_test_time"] = self.clock()
            keys[target]["last_test_message"] = str(result.get("message") or ("Test successful" if stored == "passed" else ""))[:200]
            self.set_keys(pool, keys)
        return True

    def single_key_config(self, config: Mapping[str, Any]) -> dict:
        """``config`` with every key pool switched off, for a key test's run environment.

        With the Translation pool on, the run environment exports ``USE_MULTI_API_KEYS=1`` and
        loads the pool into ``UnifiedClient``, and a new client then sends the pool's keys in
        rotation instead of the key under test (Fallback likewise retries with fallback keys).
        The endpoint settings, proxies and timeouts stay as a run would use them.
        """
        out = dict(config or {})
        for spec in self.specs:
            if spec.toggle_key:
                out[spec.toggle_key] = False
        return out

    def _probe(self, entry: Mapping[str, Any], pool_id: str, timeout: Optional[float]) -> dict:
        """One key test in the current environment. Refused (``busy``) when that environment has
        key pools on: the runner then ran it inside a running job's environment as is."""
        if any(os.environ.get(name, "0") == "1" for name in _POOL_RUNTIME_ENV):
            return {"ok": None, "status": "busy", "message": KEY_TEST_BUSY}
        return self.backend.run_test(entry, pool_id, timeout=timeout)

    def _run_one(self, config: Mapping[str, Any], entry: Mapping[str, Any], pool_id: str,
                 timeout: Optional[float]) -> dict:
        """One key test inside its own run environment (``test_runner`` holds the engine lock only
        for this probe, so a job started meanwhile waits at most one test)."""
        runner = self._test_runner

        def work() -> dict:
            return self._probe(entry, pool_id, timeout)

        if runner is None:
            return work()
        return runner(config, work)

    def _test_config(self) -> dict:
        return self.single_key_config(self.store.snapshot() if hasattr(self.store, "snapshot") else {})

    def test_entry(self, entry: Mapping[str, Any], pool: str, *, timeout: Optional[float] = None) -> dict:
        """Blocking: one key test in the run environment with the key pools off (``test_runner``)."""
        return self._run_one(self._test_config(), entry, self.spec(pool).id, timeout)

    def test_keys(self, pool: str, indices: Sequence[int], *, timeout: Optional[float] = None,
                  on_result: Optional[Callable[[int, dict], Any]] = None,
                  should_stop: Optional[Callable[[], bool]] = None) -> list:
        """Blocking: test ``indices`` one by one, each in its own run environment (results persisted
        on the key tested as they arrive)."""
        keys = self.keys(pool)
        results: list = []
        config = self._test_config()
        pool_id = self.spec(pool).id
        for index in indices:
            if should_stop is not None and should_stop():
                break
            if not 0 <= index < len(keys):
                continue
            entry = keys[index]
            if str(entry.get("api_key") or "").startswith("ENC:"):
                result = {"ok": False, "status": "error", "message": "Encrypted with a key this device does not have"}
            else:
                result = self._run_one(config, entry, pool_id, timeout)
            self.record_test(pool, index, result, model=entry.get("model"), api_key=entry.get("api_key"))
            results.append((index, result))
            if on_result is not None:
                on_result(index, result)
        return results

    # ---- import / export --------------------------------------------------------------------------------

    def export_payload(self, pools: Optional[Sequence[str]] = None) -> Optional[dict]:
        ids = [self.spec(p).id for p in pools] if pools else None
        return self.backend.export_payload(self.store.snapshot(), ids)

    @staticmethod
    def payload_key_count(payload: Mapping[str, Any]) -> int:
        pools = payload.get("pools") if isinstance(payload, Mapping) else None
        if not isinstance(pools, Mapping):
            return 0
        return sum(len(p.get("keys") or []) if isinstance(p, Mapping) else len(p or []) for p in pools.values())

    def import_plan(self, payload: Any, *, target: Optional[str] = None) -> ImportPlan:
        plan = self.backend.import_plan(payload, self.store.snapshot())
        if plan.error or target is None:
            return plan
        target_id = self.spec(target).id
        if plan.legacy:
            plan.items = [(target_id, keys, None) for _pool, keys, _e in plan.items]
        else:
            plan.items = [item for item in plan.items if item[0] == target_id]
            if not plan.items:
                plan.error = f"The file has no {self.spec(target).title}"
        return plan

    def apply_import(self, plan: ImportPlan) -> int:
        """``key_pool_service.apply_import_plan`` on a copy of the pools it touches, written back
        sparsely: a pool-aware file REPLACES the listed pools, a legacy flat list is appended."""
        fn = self.backend.fn("apply_import_plan")
        if fn is None:
            raise RuntimeError(SERVICE_MISSING)
        view: dict = {}
        for pool_id, _keys, _enabled in plan.items:
            spec = self.spec(pool_id)
            for key in (spec.config_key, spec.toggle_key):
                if key and self.store.has(key):
                    view[key] = self.store.get(key)
        before = copy.deepcopy(view)
        items = [(p, list(k), e) for p, k, e in plan.items]
        if plan.legacy:  # appended: never add a key the pool (or the file) already holds
            plan.duplicates = 0
            deduped = []
            for pool_id, keys, enabled in items:
                held = self.keys(pool_id)
                kept: list = []
                for key in keys:
                    if self.backend.find_duplicate(held + kept, key) is None:
                        kept.append(key)
                plan.duplicates += len(keys) - len(kept)
                deduped.append((pool_id, kept, enabled))
            items = deduped
        count = fn(view, {"legacy": plan.legacy, "items": items})
        changed = {k: v for k, v in view.items() if k not in before or before[k] != v}
        if changed:
            self.store.set_many(changed)
        return int(count or 0)

    def import_result_message(self, plan: ImportPlan, applied: int) -> str:
        """The desktop result line (``pool_import_result_message`` / ``legacy_import_result_message``)."""
        if plan.legacy:
            fn = self.backend.fn("legacy_import_result_message")
            message = fn(applied, plan.skipped) if fn is not None else f"Imported {applied} API key(s)"
            if plan.duplicates:
                message += f"\n{plan.duplicates} duplicate key(s) skipped (already in the pool)"
            return message
        fn = self.backend.fn("pool_import_result_message")
        if fn is not None:
            return fn(applied, len(plan.items), plan.skipped, list(plan.unknown))
        return f"Imported {applied} key(s) across {len(plan.items)} pool(s)"


# ---- screen ------------------------------------------------------------------------------------------

try:  # the pure part above must stay importable without Flet (host tests, services)
    import flet as ft

    from glossarion_mobile.ui import tokens
    from glossarion_mobile.ui.components._handlers import call_handler
    from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
    from glossarion_mobile.ui.components.dialogs import ConfirmDialog, close_dialog
    from glossarion_mobile.ui.components.reason_chip import NOT_ON_MOBILE, ReasonChip, unavailable_tile
    from glossarion_mobile.ui.components.section_card import SectionCard
    from glossarion_mobile.ui.components.sheet import bottom_sheet, scroll_column, sheet_frame
    from glossarion_mobile.ui.foreground import poll_sleep
    from glossarion_mobile.ui.screens.base import Screen
    from glossarion_mobile.ui.settings.model import mask_secret
    from glossarion_mobile.ui.theme import HIT_TARGET, status_color
except ImportError:  # pragma: no cover - Flet missing
    ft = None  # type: ignore[assignment]
    Screen = object  # type: ignore[assignment,misc]


_STATUS_TEXT = {"active": "Active", "disabled": "Disabled", "passed": "Passed", "failed": "Failed",
                "cooling": "Cooling", "encrypted": "Re-enter key", "testing": "Testing…"}
_STATUS_PALETTE = {"active": "info", "disabled": "disabled", "passed": "completed", "failed": "failed",
                   "cooling": "cooling", "encrypted": "failed", "testing": "running"}
LIVE_REFRESH_SECONDS = 1.0
DESKTOP_ONLY_ROWS = (
    ("Lock mouse wheel", "Desktop mouse-wheel guard; touch screens have no wheel. The setting is kept."),
    ("Key list zoom", "Mobile follows Settings › Appearance › Text scale. The setting is kept."),
)


def _push(*controls: Any) -> None:
    for control in controls:
        if control is None:
            continue
        try:
            control.update()
        except Exception:
            pass


class KeysScreen(Screen):  # type: ignore[misc,valid-type]
    title = "API keys"

    def __init__(
        self,
        match: Any,
        *,
        controller: KeysController,
        ctx: Any = None,
        sheet_env: Any = None,
        files: Any = None,
        page: Any = None,
        notify: Optional[Callable[..., Any]] = None,
        copy_text: Optional[Callable[[str], Any]] = None,
        read_clipboard: Optional[Callable[[], Any]] = None,
        run_io: Optional[Callable[..., Any]] = None,
        spawn: Optional[Callable[[Any], Any]] = None,
        open_refusal_patterns: Optional[Callable[[], Any]] = None,
        export_dir: Optional[str] = None,
        tablet: bool = False,
        dark: bool = False,
        is_top: Optional[Callable[[Any], bool]] = None,
    ) -> None:
        super().__init__(match)
        self.controller = controller
        self.is_top = is_top  # shell: is this screen shown (top of the stack)? Gates the live-stats ticker
        self.ctx = ctx
        self.sheet_env = sheet_env
        self.files = files
        self.page = page if page is not None else getattr(ctx, "page", None)
        self.notify = notify
        self.copy_text = copy_text
        self.read_clipboard = read_clipboard
        self.run_io = run_io
        self.spawn_fn = spawn
        self.open_refusal_patterns = open_refusal_patterns
        self.export_dir = export_dir
        self.tablet = tablet
        self.dark = dark
        slug = (match.params.get("pool") if match is not None else None) or "translation"
        self.pool = SLUG_TO_POOL.get(slug, "main")
        self.selected: set = set()
        self.testing: set = set()
        self.cards: list = []
        self.pool_chips: dict = {}
        self.editor: Any = None
        self.last_dialog: Any = None
        self.last_sheet: Any = None
        self._unsub: Optional[Callable[[], None]] = None
        self._stop_tests = threading.Event()
        self.cooling = False
        self._ticker: Any = None

    # ---- helpers -------------------------------------------------------------------------------------

    @property
    def spec(self) -> PoolSpec:
        return self.controller.spec(self.pool)

    def say(self, message: str, action: Optional[str] = None, on_action: Any = None) -> None:
        if self.notify is None:
            log.info("keys: %s", message)
            return
        try:
            self.notify(message, action, on_action)
        except TypeError:
            self.notify(message)

    def spawn(self, coro: Any) -> Any:
        if self.spawn_fn is not None:
            return self.spawn_fn(coro)
        return asyncio.ensure_future(coro)

    async def io(self, fn: Callable[..., Any], *args: Any) -> Any:
        if self.run_io is not None:
            return await self.run_io(fn, *args)
        return await asyncio.to_thread(fn, *args)

    def _show(self, dialog: Any) -> Any:
        if self.page is not None:
            dialog.show(self.page)
        self.last_dialog = dialog
        return dialog

    # ---- body ----------------------------------------------------------------------------------------

    def build_body(self) -> "ft.Control":
        rotation_switch = ft.Switch(label="Force key rotation", value=bool(self.controller.setting("force_key_rotation", True)),
                                    on_change=lambda e: self.controller.set_setting("force_key_rotation", bool(e.control.value)))
        self.rotation_switch = rotation_switch
        self.frequency_field = ft.TextField(value=str(self.controller.setting("rotation_frequency", 1)), label="Every",
                                            suffix=ft.Text("requests"), width=150, dense=True,
                                            keyboard_type=ft.KeyboardType.NUMBER, on_blur=self._on_frequency,
                                            on_submit=self._on_frequency, border_radius=tokens.RADII["field"])
        rotation = SectionCard(title="Rotation", icon="AUTORENEW", children=[
            ft.Row([rotation_switch, self.frequency_field], wrap=True, spacing=12,
                   vertical_alignment=ft.CrossAxisAlignment.CENTER)])
        for spec in self.controller.specs:
            self.pool_chips[spec.id] = ft.Chip(label=ft.Text(self._chip_label(spec)), selected=spec.id == self.pool,
                                               show_checkmark=False, on_click=lambda e, p=spec.id: self.select_pool(p),
                                               key=f"pool-{spec.slug}")
        self.chip_row = ft.Row(list(self.pool_chips.values()), scroll=ft.ScrollMode.AUTO, spacing=6)
        self.pool_header = ft.Container()
        self.selection_bar = ft.Container(visible=False)
        self.list_view = ft.ReorderableListView(
            controls=[], expand=True, on_reorder=self._on_reorder, show_default_drag_handles=False,
            padding=ft.Padding.symmetric(horizontal=tokens.SPACING["md"], vertical=4),
            header=ft.Column([rotation, self.chip_row, self.pool_header, self.selection_bar], spacing=8, tight=True),
            footer=self._footer(),
        )
        self.render()
        return self.list_view

    def actions(self) -> list:
        self.menu = ft.PopupMenuButton(
            icon=ft.Icons.MORE_VERT, tooltip="More",
            items=[
                ft.PopupMenuItem(content="Import key pools…", icon=ft.Icons.FILE_OPEN_OUTLINED,
                                 on_click=lambda e: self.spawn(self.import_keys())),
                ft.PopupMenuItem(content="Export all pools…", icon=ft.Icons.IOS_SHARE,
                                 on_click=lambda e: self.confirm_export()),
                ft.PopupMenuItem(content="Refusal patterns", icon=ft.Icons.BLOCK,
                                 on_click=lambda e: self._open_refusal()),
            ],
        )
        return [self.menu]

    def did_show(self) -> None:
        store = self.controller.store
        observe = getattr(store, "observe_keys", None)
        if observe is not None and self._unsub is None:
            keys = [s.config_key for s in self.controller.specs] + [s.toggle_key for s in self.controller.specs] + [
                "force_key_rotation", "rotation_frequency", "use_main_key_fallback", "fallback_key_shuffle"]
            self._unsub = observe(keys, lambda key, value: self._on_store_change(key))
        if self._ticker is None:
            try:
                self._ticker = self.spawn(self._tick_live_stats())
            except Exception:
                self._ticker = None

    def dispose(self) -> None:
        self._stop_tests.set()
        if self._unsub is not None:
            self._unsub()
            self._unsub = None
        if self._ticker is not None and hasattr(self._ticker, "cancel"):
            self._ticker.cancel()
        self._ticker = None

    def _shown(self) -> bool:
        """On top of the stack (not covered by another screen); True without a shell hook."""
        if self.is_top is None:
            return True
        try:
            return bool(self.is_top(self))
        except Exception:
            return True

    async def _tick_live_stats(self, interval: Optional[float] = None) -> None:
        """While the screen is shown: re-render when the running client's pool changed (success /
        error counts) or a key is cooling down (the countdown), like the desktop tree's refresh.
        No ticks while the app is in the background (``poll_sleep`` parks, UI_SPEC §7.3) or another
        screen covers this one; one refresh when it is shown again."""
        interval = LIVE_REFRESH_SECONDS if interval is None else interval
        last = None
        missed = False
        while not self._stop_tests.is_set():
            if await poll_sleep(self.page, interval):
                missed = True
            if self._stop_tests.is_set():
                return
            if not self._shown():
                missed = True
                continue
            try:
                live = self.controller.live_stats(self.pool)
            except Exception:
                continue
            signature = repr([(row or {}).get(k) for row in live for k in ("success_count", "error_count",
                                                                            "is_cooling_down", "times_used")])
            if signature != last or self.cooling or missed:
                last = signature
                if any(live) or self.cooling or missed:
                    missed = False
                    self._external_refresh()

    def _on_store_change(self, key: str) -> None:
        on_ui = getattr(self.ctx, "on_ui", None)
        if on_ui is not None:
            on_ui(self._external_refresh)
        else:
            self._external_refresh()

    def _external_refresh(self) -> None:
        self.render()
        _push(self.list_view)

    def _chip_label(self, spec: PoolSpec) -> str:
        count = self.controller.count(spec.id)
        return f"{spec.label} · {count}" if count else spec.label

    def _on_frequency(self, e: Any = None) -> None:
        text = (self.frequency_field.value or "").strip()
        try:
            value = int(text)
            if value < 1:
                raise ValueError
        except ValueError:
            self.frequency_field.error = "Enter a whole number ≥ 1"
            _push(self.frequency_field)
            return
        self.frequency_field.error = None
        self.controller.set_setting("rotation_frequency", value)
        _push(self.frequency_field)

    def select_pool(self, pool_id: str) -> None:
        self.pool = pool_id
        self.selected.clear()
        self.render()
        _push(self.list_view)

    def render(self) -> None:
        spec = self.spec
        for pid, chip in self.pool_chips.items():
            chip.selected = pid == self.pool
            chip.label = ft.Text(self._chip_label(self.controller.spec(pid)))
        self.pool_header.content = self._pool_header(spec)
        self._render_selection_bar()
        keys = self.controller.keys(self.pool)
        self.selected = {i for i in self.selected if i < len(keys)}
        live = self.controller.live_stats(self.pool)
        live = list(live) + [None] * (len(keys) - len(live))
        self.cooling = any(row and row.get("is_cooling_down") for row in live)
        self.cards = [self._key_card(i, entry, live[i]) for i, entry in enumerate(keys)]
        if not keys:
            self.list_view.controls = [ft.Container(
                key="keys-empty", padding=ft.Padding.all(16),
                content=ft.Text(f"No keys in {spec.title}. Add one below.", color=ft.Colors.ON_SURFACE_VARIANT))]
        else:
            self.list_view.controls = list(self.cards)

    def _pool_header(self, spec: PoolSpec) -> "ft.Control":
        count = self.controller.count(spec.id)
        self.pool_switch = ft.Switch(value=self.controller.enabled(spec.id), tooltip=f"Use {spec.title}",
                                     on_change=lambda e: self.controller.set_enabled(self.pool, bool(e.control.value)))
        pool_menu = ft.PopupMenuButton(
            icon=ft.Icons.MORE_HORIZ, tooltip="Pool actions",
            items=[
                ft.PopupMenuItem(content="Clear all keys", icon=ft.Icons.DELETE_SWEEP_OUTLINED,
                                 on_click=lambda e: self.confirm_clear()),
                ft.PopupMenuItem(content="Import into this pool…", icon=ft.Icons.FILE_OPEN_OUTLINED,
                                 on_click=lambda e: self.spawn(self.import_keys(target=self.pool))),
                ft.PopupMenuItem(content="Export this pool…", icon=ft.Icons.IOS_SHARE,
                                 on_click=lambda e: self.confirm_export([self.pool])),
            ],
        )
        children: list = []
        if spec.description:
            children.append(ft.Text(spec.description, theme_style=ft.TextThemeStyle.BODY_SMALL,
                                    color=ft.Colors.ON_SURFACE_VARIANT))
        if spec.id == "fallback":
            children.append(ft.Switch(label="Use Main GUI Key as Fallback #1",
                                      value=bool(self.controller.setting("use_main_key_fallback", True)),
                                      on_change=lambda e: self.controller.set_setting("use_main_key_fallback",
                                                                                      bool(e.control.value))))
            children.append(ft.Switch(label="Fallback Key Shuffle (skip keys on API delay cooldown)",
                                      value=bool(self.controller.setting("fallback_key_shuffle", False)),
                                      on_change=lambda e: self.controller.set_setting("fallback_key_shuffle",
                                                                                      bool(e.control.value))))
        if not self.controller.backend.available:
            children.append(ReasonChip(reason="Key service missing", detail=SERVICE_MISSING))
        heading = str(spec.extra.get("section_title") or spec.title)  # desktop group title
        return SectionCard(title=f"{heading} · {count}", icon="KEY", subtitle=None,
                           trailing=ft.Row([self.pool_switch, pool_menu], spacing=0, tight=True), children=children)

    def _footer(self) -> "ft.Control":
        self.add_button = ft.FilledButton(content="Add key", icon=ft.Icons.ADD, on_click=lambda e: self.open_editor(None))
        self.test_selected_button = ft.TextButton(content="Test selected", icon=ft.Icons.NETWORK_CHECK,
                                                  on_click=lambda e: self.spawn(self.test_selected()))
        self.test_all_button = ft.TextButton(content="Test all", icon=ft.Icons.FACT_CHECK_OUTLINED,
                                             on_click=lambda e: self.spawn(self.test_all()))
        rows: list = [
            ft.Row([
                self.add_button,
                ft.TextButton(content="Copy current key", icon=ft.Icons.CONTENT_COPY,
                              tooltip="Add the main API key and model to this pool",
                              on_click=lambda e: self.copy_current_key()),
                self.test_selected_button, self.test_all_button,
                ft.TextButton(content="Import", icon=ft.Icons.FILE_OPEN_OUTLINED,
                              on_click=lambda e: self.spawn(self.import_keys())),
                ft.TextButton(content="Export", icon=ft.Icons.IOS_SHARE, on_click=lambda e: self.confirm_export()),
                ft.TextButton(content="Refusal patterns", icon=ft.Icons.BLOCK, on_click=lambda e: self._open_refusal()),
            ], wrap=True, spacing=6, run_spacing=6),
        ]
        # U12 item 1: desktop-only rows are not listed on mobile
        return ft.Container(content=ft.Column(rows, spacing=4, tight=True),
                            padding=ft.Padding.only(top=8, bottom=24))

    def _key_card(self, index: int, entry: Mapping[str, Any], live: Optional[Mapping[str, Any]] = None) -> "ft.Control":
        if index in self.testing:
            status, text = "testing", _STATUS_TEXT["testing"]
        else:
            status, text = key_status_info(entry, live)
        color = status_color(_STATUS_PALETTE.get(status, "info"), self.dark)
        disabled_ctx = list(entry.get("disabled_contexts") or [])
        details = [str(entry.get("model") or "(no model)")]
        overrides = key_overrides_text(entry)
        if overrides:
            details.append(overrides)
        counts = key_counts_text(entry, live)
        if counts:
            details.append(counts)
        if entry.get("use_individual_endpoint") and entry.get("azure_endpoint"):
            details.append("individual endpoint")
        if entry.get("request_parameters"):
            details.append(f"🧩 {len(entry['request_parameters'])}")
        if disabled_ctx:
            details.append(f"{len(disabled_ctx)} context{'s' if len(disabled_ctx) != 1 else ''} off")
        used = (live or {}).get("times_used") or entry.get("times_used") or 0
        if used:
            details.append(f"used {used}×")
        message = str(entry.get("last_test_message") or "") if status in ("failed", "cooling") else ""
        selected = index in self.selected
        chip = ft.Container(
            content=ft.Row([ft.Icon(ft.Icons.CIRCLE, size=10, color=color),
                            ft.Text(text, theme_style=ft.TextThemeStyle.LABEL_SMALL, color=color)], spacing=4, tight=True),
            padding=ft.Padding.symmetric(horizontal=6, vertical=2), border=ft.Border.all(1, color),
            border_radius=tokens.RADII["chip"],
        )
        api_key = str(entry.get("api_key") or "")
        lines: list = [
            ft.Row([ft.Text(mask_secret(api_key) if api_key else "No key (keyless route)",
                            weight=ft.FontWeight.W_600, expand=True, max_lines=1), chip],
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Text(" · ".join(details), theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                    max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
        ]
        if message:
            lines.append(ft.Text(message, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ERROR, max_lines=2,
                                 overflow=ft.TextOverflow.ELLIPSIS))
        excluded = excluded_model_reason(entry.get("model"))
        if excluded is not None:  # U9: a key imported from a desktop config whose route cannot run here
            lines.append(ft.Row([ReasonChip(reason=NOT_ON_MOBILE, detail=excluded[1])], wrap=True,
                                key=f"key-excluded-{index}"))
        leading = ft.Checkbox(value=selected, on_change=lambda e, i=index: self.toggle_select(i)) if self.selected else \
            ft.Icon(ft.Icons.KEY, color=color)
        card = ft.Container(  # no explicit key: rebuilt cards must not be reconciled (frozen) by index
            content=ft.Row([
                leading,
                ft.Container(content=ft.Column(lines, spacing=2, tight=True), expand=True),
                ft.ReorderableDragHandle(content=ft.Icon(ft.Icons.DRAG_HANDLE), mouse_cursor=ft.MouseCursor.GRAB),
            ], spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            padding=ft.Padding.only(left=12, right=4, top=8, bottom=8),
            margin=ft.Margin.only(bottom=6),
            border_radius=tokens.RADII["card"],
            bgcolor=ft.Colors.SECONDARY_CONTAINER if selected else ft.Colors.SURFACE_CONTAINER_LOW,
            on_click=lambda e, i=index: self._on_card_tap(i),
            on_long_press=lambda e, i=index: self.toggle_select(i),
        )
        return card

    # ---- selection ---------------------------------------------------------------------------------------

    def _on_card_tap(self, index: int) -> None:
        if self.selected:
            self.toggle_select(index)
        else:
            self.open_editor(index)

    def toggle_select(self, index: int) -> None:
        if index in self.selected:
            self.selected.discard(index)
        else:
            self.selected.add(index)
        self.render()
        _push(self.list_view)

    def clear_selection(self) -> None:
        self.selected.clear()
        self.render()
        _push(self.list_view)

    def handle_back(self) -> bool:
        """Android back leaves selection mode before it leaves the screen (UI_SPEC §1.6 rule 2)."""
        if self.selected:
            self.clear_selection()
            return True
        return False

    def _render_selection_bar(self) -> None:
        if not self.selected:
            self.selection_bar.visible = False
            self.selection_bar.content = None
            return
        n = len(self.selected)
        self.selection_bar.visible = True
        self.selection_bar.content = ft.Container(
            content=ft.Row([
                ft.IconButton(icon=ft.Icons.CLOSE, tooltip="Done", on_click=lambda e: self.clear_selection(),
                              size_constraints=HIT_TARGET),
                ft.Text(f"{n} selected", expand=True, weight=ft.FontWeight.W_600),
                ft.IconButton(icon=ft.Icons.TOGGLE_ON_OUTLINED, tooltip="Enable", size_constraints=HIT_TARGET,
                              on_click=lambda e: self.bulk_enable(True)),
                ft.IconButton(icon=ft.Icons.TOGGLE_OFF_OUTLINED, tooltip="Disable", size_constraints=HIT_TARGET,
                              on_click=lambda e: self.bulk_enable(False)),
                ft.IconButton(icon=ft.Icons.NETWORK_CHECK, tooltip="Test", size_constraints=HIT_TARGET,
                              on_click=lambda e: self.spawn(self.test_selected())),
                ft.IconButton(icon=ft.Icons.DELETE_OUTLINE, tooltip="Remove", size_constraints=HIT_TARGET,
                              on_click=lambda e: self.remove_selected()),
                ft.IconButton(icon=ft.Icons.MORE_VERT, tooltip="More", size_constraints=HIT_TARGET,
                              on_click=lambda e: self.open_bulk_more()),
            ], spacing=0, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            bgcolor=ft.Colors.SECONDARY_CONTAINER, border_radius=tokens.RADII["card"],
            padding=ft.Padding.symmetric(horizontal=4), key="keys-selection-bar",
        )

    def bulk_enable(self, value: bool) -> int:
        changed = self.controller.set_keys_enabled(self.pool, sorted(self.selected), value)
        self.say(f"{'Enabled' if value else 'Disabled'} {changed} key(s)")
        self.render()
        _push(self.list_view)
        return changed

    def remove_selected(self) -> list:
        removed = self.controller.remove(self.pool, sorted(self.selected))
        self.selected.clear()
        pool = self.pool
        self.render()
        _push(self.list_view)
        self.say(f"Removed {len(removed)} key(s)", "Undo", lambda: self.undo_remove(pool, removed))
        return removed

    def undo_remove(self, pool: str, removed: list) -> None:
        self.controller.restore(pool, removed)
        self.render()
        _push(self.list_view)

    def open_bulk_more(self) -> "ActionSheet":
        others = [s for s in self.controller.specs if s.id != self.pool]
        items = [
            ActionItem("Change model…", lambda: self.open_bulk_field("model"), icon="SMART_TOY_OUTLINED"),
            ActionItem("Set cooldown…", lambda: self.open_bulk_field("cooldown"), icon="TIMER_OUTLINED"),
            ActionItem("Set / clear output token limit…", lambda: self.open_bulk_field("individual_output_token_limit"),
                       icon="DATA_USAGE"),
            ActionItem("Set / clear key temperature…", lambda: self.open_bulk_field("individual_key_temperature"),
                       icon="THERMOSTAT"),
            ActionItem("Set / clear API call delay…", lambda: self.open_bulk_field("api_call_delay"),
                       icon="HOURGLASS_EMPTY"),
            ActionItem("Individual endpoint…", lambda: self.open_bulk_field("endpoint"), icon="LAN_OUTLINED"),
            ActionItem("🧩 Request parameters…", lambda: self.open_bulk_field("request_parameters"), icon="TUNE"),
            ActionItem("Set request contexts…", lambda: self.open_context_sheet(), icon="TUNE"),
            ActionItem("Copy key", lambda: self.spawn(self.copy_key_to_clipboard()), icon="CONTENT_COPY"),
        ]
        items += [ActionItem(f"Move to {s.title}", lambda s=s: self.copy_selected(s.id, move=True), icon="DRIVE_FILE_MOVE_OUTLINE")
                  for s in others]
        items += [ActionItem(f"Copy to {s.title}", lambda s=s: self.copy_selected(s.id, move=False), icon="CONTENT_COPY")
                  for s in others]
        sheet = ActionSheet(items, title=f"{len(self.selected)} selected", tablet=self.tablet)
        self.last_sheet = sheet
        if self.page is not None:
            sheet.show(self.page)
        return sheet

    def open_bulk_field(self, field: str) -> "BulkFieldSheet":
        """One field across the selected keys (desktop shared key menu): a sheet with the value,
        Apply and, where the desktop has it, Clear."""
        indices = sorted(self.selected)
        keys = self.controller.keys(self.pool)
        first = keys[indices[0]] if indices and indices[0] < len(keys) else {}
        sheet = BulkFieldSheet(field, first, count=len(indices), sheet_env=self.sheet_env, page=self.page,
                               on_apply=lambda values: self.apply_bulk_fields(indices, values))
        self.last_sheet = sheet
        if self.page is not None:
            sheet.show(self.page)
        return sheet

    def apply_bulk_fields(self, indices: list, values: Mapping[str, Any]) -> Optional[str]:
        changed, errors = self.controller.set_key_fields(self.pool, indices, values)
        if errors and not changed:
            return "; ".join(errors[:3])
        note = f" ({len(errors)} skipped: {errors[0]})" if errors else ""
        self.say(f"Updated {changed} key(s){note}")
        self.render()
        _push(self.list_view)
        return None

    def copy_selected(self, target: str, *, move: bool) -> int:
        count = self.controller.copy_to(self.pool, sorted(self.selected), target, move=move)
        verb = "Moved" if move else "Copied"
        self.say(f"{verb} {count} key(s) to {self.controller.spec(target).title}")
        if move:
            self.selected.clear()
        self.render()
        _push(self.list_view)
        return count

    def open_context_sheet(self) -> "ContextSheet":
        indices = sorted(self.selected)
        sheet = ContextSheet(
            states=self.controller.context_states(self.pool, indices), labels=context_labels(),
            on_apply=lambda changes: self._apply_contexts(indices, changes),
        )
        self.last_sheet = sheet
        if self.page is not None:
            sheet.show(self.page)
        return sheet

    def _apply_contexts(self, indices: list, changes: Mapping[str, bool]) -> int:
        changed = self.controller.set_disabled_contexts(self.pool, indices, changes)
        self.say(f"Updated request contexts on {changed} key(s)")
        self.render()
        _push(self.list_view)
        return changed

    def _on_reorder(self, e: Any) -> None:
        old, new = int(getattr(e, "old_index", -1)), int(getattr(e, "new_index", -1))
        if self.controller.move(self.pool, old, new):
            self.selected.clear()
        self.render()
        _push(self.list_view)

    # ---- editor --------------------------------------------------------------------------------------------

    def open_editor(self, index: Optional[int], *, prefill: Optional[Mapping[str, Any]] = None) -> Any:
        from glossarion_mobile.ui.screens.key_editor import KeyEditor

        spec = self.spec
        keys = self.controller.keys(self.pool)
        new = index is None or not 0 <= index < len(keys)
        entry = (dict(prefill) if prefill is not None else self.controller.new_entry(spec.id)) if new else keys[index]
        azure_versions: list = []
        schema = getattr(self.ctx, "schema", None)
        try:
            spec_obj = schema.spec("azure_api_version") if schema is not None else None
            choices = getattr(spec_obj, "choices", None) or ()
            azure_versions = [str(c[0] if isinstance(c, (list, tuple)) else c) for c in choices]
        except Exception:
            azure_versions = []

        pool = self.pool

        def save(value: dict) -> Optional[str]:
            if new:
                _index, error = self.controller.add_key(pool, value)
            else:
                error = self.controller.update_key(pool, index, value)
            if error is None:
                self.render()
                _push(self.list_view)
            return error

        def saved(value: dict) -> None:
            # After the editor has closed: a snackbar shown inside save() would be the dialog the
            # close pops, leaving the sheet open with no feedback (owner report).
            self.say(self.controller.saved_message(pool, value, added=new))

        async def test(value: dict) -> dict:
            return await self.io(lambda: self.controller.test_entry(value, self.pool))

        self.editor = KeyEditor(
            self.ctx, entry=entry, pool_id=spec.id, pool_title=spec.title, contexts=spec.contexts,
            context_labels=context_labels(), on_save=save, on_saved=saved, on_test=test, sheet_env=self.sheet_env,
            azure_versions=azure_versions, read_clipboard=self.read_clipboard, copy_text=self.copy_text, new=new,
            test_reason=None if self.controller.backend.available else SERVICE_MISSING,
        ).show()
        return self.editor

    # ---- tests ----------------------------------------------------------------------------------------------

    async def test_selected(self) -> list:
        indices = sorted(self.selected)
        if not indices:
            self.say("Select keys to test (long-press a key)")
            return []
        return await self.run_tests(indices)

    async def test_all(self) -> list:
        """Test all. The Translation pool tests only its enabled keys (desktop ``_test_all``: "Only test
        enabled keys", "No enabled keys to test"); the other pools test every key, as on desktop."""
        keys = self.controller.keys(self.pool)
        if self.pool == "main":
            if not keys:
                self.say("No keys to test")
                return []
            indices = [i for i, entry in enumerate(keys) if entry.get("enabled", True)]
            if not indices:
                self.say("No enabled keys to test")
                return []
            return await self.run_tests(indices)
        return await self.run_tests(list(range(len(keys))))

    async def run_tests(self, indices: list) -> list:
        if not indices:
            self.say("No keys to test")
            return []
        if not self.controller.backend.available:
            self.say(SERVICE_MISSING)
            return []
        pool = self.pool
        self.testing = set(indices)
        self._stop_tests.clear()
        self.render()
        _push(self.list_view)
        loop = asyncio.get_running_loop()

        def on_result(index: int, _result: dict) -> None:
            def apply() -> None:
                self.testing.discard(index)
                if self.pool == pool:
                    self.render()
                    _push(self.list_view)

            loop.call_soon_threadsafe(apply)

        try:
            results = await self.io(lambda: self.controller.test_keys(pool, indices, on_result=on_result,
                                                                      should_stop=self._stop_tests.is_set))
        finally:
            self.testing.clear()
        passed = sum(1 for _i, r in results if r.get("status") == "passed")
        untestable = sum(1 for _i, r in results if r.get("status") == "untestable")
        busy = sum(1 for _i, r in results if r.get("status") == "busy")
        note = f" · {untestable} not testable" if untestable else ""
        if busy:
            note += f" · {busy} skipped (a running job is using the key pools)"
        self.say(f"Test complete: {passed}/{len(results)} passed{note}")
        self.render()
        _push(self.list_view)
        return results

    def copy_current_key(self) -> Any:
        """Desktop "Copy Current Key" (``_copy_current_settings``): the Add key editor pre-filled with
        the main API key and model."""
        entry = self.controller.current_key_entry(self.spec.id)
        if not entry.get("api_key") and not entry.get("model"):
            self.say("No main API key or model to copy")
        return self.open_editor(None, prefill=entry)

    async def copy_key_to_clipboard(self) -> Optional[str]:
        """The selected (else first enabled) key's text to the clipboard."""
        keys = self.controller.keys(self.pool)
        index = min(self.selected) if self.selected else next(
            (i for i, k in enumerate(keys) if k.get("enabled", True)), None)
        if index is None or index >= len(keys):
            self.say("No key to copy")
            return None
        value = str(keys[index].get("api_key") or "")
        if not value or value.startswith("ENC:"):
            self.say("This key cannot be copied")
            return None
        if self.copy_text is not None:
            result = self.copy_text(value)
            if asyncio.iscoroutine(result):
                await result
        return value

    # ---- clear / import / export -------------------------------------------------------------------------

    def confirm_clear(self) -> Any:
        spec = self.spec
        count = self.controller.count(spec.id)
        if not count:
            self.say(f"{spec.title} is already empty")
            return None

        def clear() -> None:
            pool = self.pool
            removed = self.controller.clear(pool)
            if not removed:
                self.say(f"{spec.title} is already empty")
                return
            self.selected.clear()
            self.render()
            _push(self.list_view)
            self.say(f"Removed {len(removed)} key(s)", "Undo", lambda: self.undo_remove(pool, removed))

        return self._show(ConfirmDialog(title="Clear all keys", body=f"Remove all {count} keys from {spec.title}?",
                                        confirm_label="Remove all", destructive=True, on_confirm=clear))

    def confirm_export(self, pools: Optional[Sequence[str]] = None) -> Any:
        if not self.controller.backend.available:
            self.say(SERVICE_MISSING)
            return None
        return self._show(ConfirmDialog(title="Export API keys", body=PLAINTEXT_WARNING, confirm_label="Export",
                                        on_confirm=lambda: self.spawn(self.export_keys(pools))))

    def _export_path(self) -> str:
        base = self.export_dir or os.path.join(os.environ.get("GLOSSARION_DATA_DIR") or os.getcwd(), "Exports")
        os.makedirs(base, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return os.path.join(base, f"glossarion-key-pools-{stamp}.json")

    async def export_keys(self, pools: Optional[Sequence[str]] = None) -> Optional[str]:
        payload = await self.io(lambda: self.controller.export_payload(pools))
        if payload is None:
            self.say(SERVICE_MISSING)
            return None
        total = KeysController.payload_key_count(payload)
        if total == 0:
            self.say("No keys to export in any pool" if not pools else "No keys to export in this pool")
            return None
        path = self._export_path()

        def write() -> None:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)

        if self.files is None:
            await self.io(write)
            self.say(f"Exported {total} API key(s) to {os.path.basename(path)}")
            return path
        options = self.files.export_options(path)

        async def run(option_id: str) -> None:
            # The plain-text file exists only while the chosen export runs: dismissing the
            # sheet leaves nothing behind.
            try:
                await self.io(write)
                result = await self.files.export(option_id, path)
                ok = bool(getattr(result, "ok", result))
                if ok:
                    self.say(f"Exported {total} API key(s)")
            finally:
                await self.io(lambda: os.path.exists(path) and os.remove(path))

        items = [ActionItem(o.label, (lambda o=o: self.spawn(run(o.id))), icon=o.icon, disabled_reason=o.disabled_reason)
                 for o in options]
        sheet = ActionSheet(items, title="Export key pools", subtitle=f"{total} key(s) · plain text", tablet=self.tablet)
        self.last_sheet = sheet
        if self.page is not None:
            sheet.show(self.page)
        return path

    async def import_keys(self, target: Optional[str] = None) -> Optional[ImportPlan]:
        if not self.controller.backend.available:
            self.say(SERVICE_MISSING)
            return None
        if self.files is None:
            self.say("Importing files is not available in this session")
            return None
        picked = await self.files.pick_files(target="inbox", allowed_extensions=list(IMPORT_EXTENSIONS),
                                             allow_multiple=False, dialog_title="Import API Keys")
        if not picked:
            return None
        path = picked[0].path if hasattr(picked[0], "path") else str(picked[0])
        return await self.import_from_path(path, target=target, remove_after=True)

    async def import_from_path(self, path: str, *, target: Optional[str] = None,
                               remove_after: bool = False) -> Optional[ImportPlan]:
        def read() -> Any:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    return json.load(handle)
            finally:
                if remove_after:
                    try:
                        os.remove(path)  # the picked copy holds plain-text keys
                    except OSError:
                        pass

        try:
            payload = await self.io(read)
        except Exception as exc:
            self.say(f"Failed to import: {exc}")
            return None
        plan = await self.io(lambda: self.controller.import_plan(payload, target=target))
        if plan.error:
            self.say(plan.error)
            return plan
        titles = self.controller.titles()
        if plan.legacy:
            body = f"Add {plan.total} key(s) to {titles.get(plan.items[0][0], '')}?"
        else:
            body = "This will REPLACE the contents of these pools:"

        def apply() -> None:
            count = self.controller.apply_import(plan)
            self.say(self.controller.import_result_message(plan, count).replace("\n", " · "))
            self.render()
            _push(self.list_view)

        self._show(ConfirmDialog(title="Import key pools", body=body, items=plan.summary_lines(titles),
                                 confirm_label="Import", destructive=not plan.legacy, on_confirm=apply))
        return plan

    def _open_refusal(self) -> None:
        if self.open_refusal_patterns is not None:
            call_handler(self.open_refusal_patterns)


#: Bulk numeric fields: (label, kind, low, high, clear value or None when the desktop has no Clear).
BULK_NUMBER_FIELDS = {
    "cooldown": ("Cooldown (seconds)", "int", 10, 3600, None),
    "individual_output_token_limit": ("Output token limit", "int", 0, 2_000_000, "clear"),
    "individual_key_temperature": ("Temperature", "float", 0.0, 2.0, "clear"),
    "api_call_delay": ("API call delay (seconds)", "float", 0.0, 3600.0, "clear"),
}
#: What "Clear" writes (the KeyEditor's empty values: the global limit / temperature / delay).
BULK_CLEAR_VALUES = {"individual_output_token_limit": None, "individual_key_temperature": None,
                     "api_call_delay": 0.0, "request_parameters": {}}


def parse_bulk_value(field: str, text: str) -> tuple:
    """``(values, error)`` for one bulk numeric field typed as text."""
    label, kind, low, high, _clear = BULK_NUMBER_FIELDS[field]
    raw = str(text or "").strip()
    try:
        value: Any = int(raw) if kind == "int" else float(raw)
    except ValueError:
        return None, "Enter a whole number" if kind == "int" else "Enter a number"
    if not low <= value <= high:
        return None, f"{label}: {low}–{high}"
    return {field: value}, None


class BulkFieldSheet:
    """Apply one key field to N selected keys (Change model / cooldown / limits / temperature / delay /
    individual endpoint / request parameters)."""

    TITLES = {"model": "Change model", "endpoint": "Individual endpoint", "request_parameters": "🧩 Request parameters"}

    def __init__(self, field: str, sample: Mapping[str, Any], *, count: int,
                 on_apply: Callable[[Mapping[str, Any]], Optional[str]], sheet_env: Any = None,
                 page: Any = None) -> None:
        self.field = field
        self.count = count
        self.on_apply = on_apply
        self.page = page
        self.result: Optional[dict] = None
        self.error = ft.Text("", color=ft.Colors.ERROR, visible=False, theme_style=ft.TextThemeStyle.BODY_SMALL)
        controls: list = []
        self.can_clear = field in BULK_CLEAR_VALUES
        if field == "model":
            from glossarion_mobile.ui.sheets.model_sheet import ModelPicker

            self.input: Any = ModelPicker(value=str(sample.get("model") or ""), label="Model", env=sheet_env,
                                          page=page, key="bulk-model")
            controls.append(self.input)
        elif field == "endpoint":
            self.endpoint_switch = ft.Switch(label="Use individual endpoint",
                                             value=bool(sample.get("use_individual_endpoint", True)))
            self.endpoint_field = ft.TextField(value=str(sample.get("azure_endpoint") or ""), label="Endpoint URL",
                                               hint_text="https://…", dense=True, keyboard_type=ft.KeyboardType.URL)
            self.version_field = ft.TextField(value=str(sample.get("azure_api_version") or "2025-01-01-preview"),
                                              label="Azure API version", dense=True)
            controls += [self.endpoint_switch, self.endpoint_field, self.version_field]
            self.input = self.endpoint_field
        elif field == "request_parameters":
            params = sample.get("request_parameters") or {}
            self.input = ft.TextField(value=json.dumps(params, ensure_ascii=False, indent=2) if params else "",
                                      label="Parameters (JSON object)", multiline=True, min_lines=3, max_lines=10,
                                      hint_text='{"top_p": 0.9}', key="bulk-params")
            controls.append(self.input)
        else:
            label, kind, low, high, _clear = BULK_NUMBER_FIELDS[field]
            value = sample.get(field)
            self.input = ft.TextField(value="" if value in (None, "") else str(value), label=label, dense=True,
                                      keyboard_type=ft.KeyboardType.NUMBER,
                                      helper=f"{low}–{high:,}" if kind == "int" else f"{low}–{high}",
                                      key="bulk-number")
            controls.append(self.input)
            self.can_clear = BULK_NUMBER_FIELDS[field][4] is not None
        title = self.TITLES.get(field) or BULK_NUMBER_FIELDS.get(field, (field,))[0]
        buttons = [ft.TextButton(content="Cancel", on_click=lambda e: self.close())]
        if self.can_clear:
            buttons.append(ft.TextButton(content="Clear", on_click=lambda e: self.apply(clear=True), key="bulk-clear"))
        buttons.append(ft.FilledButton(content="Apply", on_click=lambda e: self.apply(), key="bulk-apply"))
        self.sheet = bottom_sheet(sheet_frame(scroll_column([
            ft.Text(f"{title} · {count} key{'s' if count != 1 else ''}", theme_style=ft.TextThemeStyle.TITLE_MEDIUM,
                    weight=ft.FontWeight.W_600),
            *controls, self.error,
        ], footer=[ft.Row(buttons, alignment=ft.MainAxisAlignment.END)])))

    def values(self, *, clear: bool = False) -> tuple:
        field = self.field
        if clear:
            if field == "endpoint":
                return {"use_individual_endpoint": False}, None
            return {field: copy.deepcopy(BULK_CLEAR_VALUES[field])}, None
        if field == "model":
            model = str(getattr(self.input, "value", "") or "").strip()
            return ({"model": model}, None) if model else (None, "Choose a model")
        if field == "endpoint":
            url = str(self.endpoint_field.value or "").strip()
            on = bool(self.endpoint_switch.value)
            if on and not url:
                return None, "Enter the endpoint URL"
            return {"use_individual_endpoint": on, "azure_endpoint": url or None,
                    "azure_api_version": str(self.version_field.value or "").strip() or None}, None
        if field == "request_parameters":
            text = str(self.input.value or "").strip()
            if not text:
                return {"request_parameters": {}}, None
            try:
                value = json.loads(text)
            except ValueError as exc:
                return None, f"Not valid JSON: {exc}"
            if not isinstance(value, dict):
                return None, "Enter a JSON object"
            return {"request_parameters": value}, None
        return parse_bulk_value(field, str(self.input.value or ""))

    def apply(self, *, clear: bool = False) -> Optional[str]:
        values, error = self.values(clear=clear)
        if error is None and values is not None:
            error = self.on_apply(values)
        if error:
            self.error.value = error
            self.error.visible = True
            _push(self.error)
            return error
        self.result = dict(values or {})
        self.close()
        return None

    def show(self, page: Any) -> None:
        self.page = page
        page.show_dialog(self.sheet)

    def close(self) -> None:
        close_dialog(self.page, self.sheet)


def context_presets(contexts: Sequence[str]) -> list:
    """``[(label, allowed contexts), ...]`` of the desktop context dialog (``key_contexts.context_presets``)."""
    try:
        from key_contexts import context_presets as shared
    except Exception:
        return []
    try:
        return [(str(label), set(allowed)) for label, allowed in shared(tuple(contexts))]
    except Exception:
        return []


def context_preset_row(contexts: Sequence[str], on_preset: Callable[[Any], Any], *, key: str) -> Any:
    """The "Enable all · Disable all · 🖼️ Images only" chip row above a context chip list."""
    chips = [ft.Chip(label=ft.Text(label), on_click=lambda e, a=allowed: on_preset(a),
                     leading=ft.Icon(ft.Icons.DONE_ALL if label == "Enable all" else ft.Icons.REMOVE_DONE
                                     if label == "Disable all" else ft.Icons.IMAGE_OUTLINED, size=16),
                     key=f"{key}-{index}")
             for index, (label, allowed) in enumerate(context_presets(contexts))]
    return ft.Row(chips, wrap=True, spacing=6, run_spacing=6, visible=bool(chips), key=key)


class ContextSheet:
    """Bulk "Set request contexts": tri-state chips (on · off · mixed); untouched ones stay as they are."""

    def __init__(self, *, states: Mapping[str, Optional[bool]], labels: Mapping[str, str],
                 on_apply: Callable[[dict], Any]) -> None:
        self.states = dict(states)
        self.initial = dict(states)
        self.labels = dict(labels)
        self.on_apply = on_apply
        self.changes: dict = {}
        self.applied = False
        self._page: Any = None
        self.chips: dict = {}
        for context in self.states:
            self.chips[context] = ft.Chip(label=ft.Text(self._label(context)), selected=bool(self.states[context]),
                                          on_click=lambda e, c=context: self.cycle(c), key=f"bulk-ctx-{context}")
        self.apply_button = ft.FilledButton(content="Apply", on_click=lambda e: self.apply())
        # the desktop dialog's shortcut buttons (key_contexts.context_presets)
        self.preset_row = context_preset_row(list(self.states), self.apply_preset, key="bulk-ctx-presets")
        # 25 chips are taller than a phone: they scroll, Cancel / Apply stay pinned below them.
        self.dialog = bottom_sheet(sheet_frame(scroll_column([
            ft.Text("Request contexts", theme_style=ft.TextThemeStyle.TITLE_LARGE, weight=ft.FontWeight.W_600),
            ft.Text("Tap to switch a context on or off for every selected key. “–” means the keys differ; "
                    "contexts you do not touch stay as they are.", theme_style=ft.TextThemeStyle.BODY_SMALL),
            self.preset_row,
            ft.Row(list(self.chips.values()), wrap=True, spacing=6, run_spacing=6),
        ], footer=[ft.Row([ft.TextButton(content="Cancel", on_click=lambda e: self.close()), self.apply_button],
                          alignment=ft.MainAxisAlignment.END)])))

    def _label(self, context: str) -> str:
        state = self.states.get(context)
        mark = "–" if state is None else ("✓" if state else "✕")
        return f"{mark} {self.labels.get(context, context)}"

    def cycle(self, context: str) -> Optional[bool]:
        state = self.states.get(context)
        new = True if state is None else (not state)
        if self.initial.get(context) is None and state is False:
            new = None  # back to "leave as is" for a mixed context
        self.states[context] = new
        if new is None or new == self.initial.get(context):
            self.changes.pop(context, None)
        else:
            self.changes[context] = new
        chip = self.chips[context]
        chip.selected = bool(new)
        chip.label = ft.Text(self._label(context))
        _push(chip)
        return new

    def apply_preset(self, allowed: Any) -> None:
        """Enable all · Disable all · Images only: every context on exactly when it is in ``allowed``
        (desktop ``set_routes``: the mixed state is cleared)."""
        allowed = set(allowed or ())
        for context in self.states:
            new = context in allowed
            self.states[context] = new
            if new == self.initial.get(context):
                self.changes.pop(context, None)
            else:
                self.changes[context] = new
            chip = self.chips[context]
            chip.selected = new
            chip.label = ft.Text(self._label(context))
        _push(*self.chips.values())

    def apply(self) -> Any:
        if self.applied:  # a second tap while the sheet closes
            return None
        self.applied = True
        self.close()
        return self.on_apply(dict(self.changes))

    def show(self, page: Any) -> None:
        self._page = page
        page.show_dialog(self.dialog)

    def close(self) -> None:
        close_dialog(self._page, self.dialog)
