"""ModelCatalogService: the model list, provider polls and Model Manager edits (plan §4, design §5.6).

Everything here binds to the shared GUI-free modules the desktop uses; nothing re-implements
catalog logic:

* ``model_options`` (already shared): the built-in + cached catalog (``get_model_options``),
  tombstone-aware merge (``merge_saved_model_options``), provider resolution
  (``catalog_provider_for_model``), the 24 h auto-poll decision
  (``due_provider_catalog_for_model`` / ``provider_model_catalog_supports_anonymous_poll``),
  provider polls (``refresh_provider_model_catalogs``), the seven-day poll markers
  (``get_current_polled_provider_models`` / ``PolledModelKeys`` / ``model_has_polled_marker``)
  and the numbered account aliases (``numbered_model_completion_values``). The catalog cache
  path comes from ``GLOSSARION_MODEL_CATALOG_CACHE`` (set by the bootstrap env contract).
* ``model_catalog_core`` (U4, extracted from ``TranslatorGUI``): explicit re-add of removed
  models (tombstones), applying a provider refresh, the Model Manager's save order and its
  poll merge. Bound through ``CatalogCore``; while the module is not in the build those
  edits report ``core_missing`` and the UI shows them disabled with a reason.
* ``run_env.RunEnvMixin._normalize_custom_prefix_routes`` (shared U2): custom prefix routes.

Credentials are passed explicitly, never through ``os.environ``: the configured ``api_key``
goes only to the selected model's provider (the desktop rule), and ``provider_keys`` holds the
first enabled key-pool key whose model belongs to a provider (mobile has no OPENAI_API_KEY /
GEMINI_API_KEY environment variables, which is where the desktop finds per-provider keys).

All blocking work (catalog file reads, HTTP polls) runs through ``run_io``; listeners get a
``CatalogSnapshot`` (posted through ``post`` when given). Pure Python (3.10), no Flet import.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import importlib
import importlib.util
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

__all__ = [
    "AUTO_POLL_TTL_SECONDS",
    "CatalogCore",
    "CatalogSnapshot",
    "EXCLUDED_PROVIDERS",
    "EXCLUDED_ROUTE_PREFIXES",
    "EXCLUDED_STATUS",
    "FAMILY_TITLES",
    "ModelCatalogService",
    "ModelGroup",
    "PROVIDER_CHIPS",
    "RefreshOutcome",
    "RouteInfo",
    "THINKING_FIELDS",
    "chip_of",
    "default_service",
    "excluded_detail",
    "excluded_route",
    "family_of",
    "group_models",
    "job_model",
    "job_model_block",
    "model_block",
    "mobile_statuses",
    "model_needs_api_key",
    "normalize_custom_routes",
    "provider_excluded",
    "provider_excluded_detail",
    "provider_label",
    "provider_of",
    "rank_models",
    "route_info",
    "set_default_service",
    "validate_prefix_rows",
]

log = logging.getLogger("glossarion.models")

AUTO_POLL_TTL_SECONDS = 24 * 60 * 60  # model_options._MODEL_CATALOG_CACHE_TTL_SECONDS
MAX_RESULTS = 300

def _rules() -> Any:
    try:
        import settings_rules  # shared (U4); cheap to import. A literal import: the collector bundles it

        return settings_rules
    except Exception:
        return None


#: Plan "Excluded on mobile" model routes; the reasons live in ``settings_rules.EXCLUDED_ROUTE_PREFIXES``
#: (shared). This tuple is the U3 alias and the fallback when the backend is not importable.
EXCLUDED_ROUTE_PREFIXES = ("antigravity/", "ocagy", "ocz/", "authza", "autharena", "search/opera", "ollamapull/")
KEPT_NOTE = "A model chosen on the desktop stays selected in config.json, but it cannot run here."


def _excluded_prefixes() -> tuple:
    rules = _rules()
    prefixes = getattr(rules, "EXCLUDED_ROUTE_PREFIXES", None) if rules is not None else None
    return tuple(prefixes) if prefixes else EXCLUDED_ROUTE_PREFIXES


def excluded_route(model: Optional[str]) -> Optional[str]:
    """Short reason (``"ollamapull/ isn't available on mobile"``) for an excluded route, else None."""
    value = str(model or "").strip().lower()
    for prefix in _excluded_prefixes():
        if value.startswith(prefix):
            route = value.split("/", 1)[0] + "/" if "/" in value else value
            return f"{route} isn't available on mobile"
    return None


#: Catalog providers (``model_options`` names) of the excluded routes.
EXCLUDED_PROVIDERS = frozenset({"antigravity", "ocagy", "opencode-zen", "authza", "autharena", "ollamapull"})


def provider_excluded(provider: Optional[str]) -> bool:
    name = str(provider or "").split(":", 1)[0].strip().lower()
    return name in EXCLUDED_PROVIDERS or bool(excluded_route(name + "/"))


def excluded_detail(model: Optional[str]) -> Optional[str]:
    """Long reason for the InfoSheet behind the ReasonChip (``settings_rules.excluded_route_reason``)."""
    if not excluded_route(model):
        return None
    rules = _rules()
    reason = None
    if rules is not None and hasattr(rules, "excluded_route_reason"):
        try:
            reason = rules.excluded_route_reason(model, "mobile")
        except Exception:
            reason = None
    return f"{reason or excluded_route(model)}. {KEPT_NOTE}".replace(".. ", ". ")


#: Catalog status of an excluded provider (the poll's own status - "static fallback (ModuleNotFoundError …)",
#: "connection refused" from a desktop localhost proxy - would read as a failure to fix).
EXCLUDED_STATUS = "Not available on mobile"

#: Job kinds that never need the configured model (compile / file tools, the unified-glossary merge, the QA
#: scan, whose optional AI checks report their own failures).
MODEL_FREE_JOB_KINDS = frozenset({"compile_epub", "compile_pdf", "validate_epub", "rename_outputs",
                                  "md_txt_sidecars", "br_to_paragraphs", "retranslate", "qa_scan",
                                  "unified_glossary"})


def model_block(model: Optional[str]) -> Optional[tuple]:
    """``(reason, detail)`` when ``model`` is a route excluded on mobile (the chat Send rule,
    ``send_state.excluded_route_reason``: "ocz/ isn't available on mobile"), else None."""
    reason = excluded_route(model)
    return (reason, excluded_detail(model) or reason) if reason else None


def job_model(params: Optional[Mapping[str, Any]] = None, config_get: Optional[Callable[..., Any]] = None) -> str:
    """The model a job runs with: its ``config_overrides`` / ``model`` param, else the config's ``model``."""
    params = params if isinstance(params, Mapping) else {}
    overrides = params.get("config_overrides") if isinstance(params.get("config_overrides"), Mapping) else {}
    model = overrides.get("model") or params.get("model")
    if not model and config_get is not None:
        try:
            model = config_get("model", "")
        except Exception:
            model = ""
    return str(model or "").strip()


def job_model_block(kind: Any, params: Optional[Mapping[str, Any]] = None,
                    config_get: Optional[Callable[..., Any]] = None) -> Optional[tuple]:
    """The preflight of every job start (Library › Translate, Tools, glossary extraction, ...): ``(reason,
    detail)`` when the job would run an excluded route (it would fail inside the client with desktop-only
    text such as "OpenCode Zen adapter not found"), else None."""
    name = str(getattr(kind, "value", kind) or "")
    if name in MODEL_FREE_JOB_KINDS:
        return None
    return model_block(job_model(params, config_get))


#: The route prefix of each excluded catalog provider (for its long reason).
_EXCLUDED_PROVIDER_ROUTES = {"antigravity": "antigravity/", "ocagy": "ocagy/", "opencode-zen": "ocz/",
                             "authza": "authza/", "autharena": "autharena/", "ollamapull": "ollamapull/"}


def provider_excluded_detail(provider: Optional[str]) -> str:
    """The InfoSheet text behind an excluded provider's "Not available on mobile" chip."""
    name = str(provider or "").split(":", 1)[0].strip().lower()
    return excluded_detail(_EXCLUDED_PROVIDER_ROUTES.get(name, name + "/")) or EXCLUDED_STATUS


def mobile_statuses(statuses: Mapping[str, Any]) -> dict:
    """Catalog poll statuses with the excluded providers' replaced by ``EXCLUDED_STATUS``."""
    return {str(p): (EXCLUDED_STATUS if provider_excluded(p) else s) for p, s in dict(statuses or {}).items()}


def login_route(model: Optional[str]) -> tuple:
    """``(route, slot)`` of a sign-in model (``settings_rules.AUTH_ACCOUNT_ROUTE_PATTERNS``), else (None, 0)."""
    text = str(model or "").strip().lower()
    rules = _rules()
    patterns = getattr(rules, "AUTH_ACCOUNT_ROUTE_PATTERNS", None) if rules is not None else None
    if patterns:
        for route, pattern in patterns.items():
            match = pattern.match(text)
            if match:
                return route, int(match.group(1) or 0)
        return None, 0
    match = _LOGIN_ROUTE.match(text)
    if match and match.group(1) != "authgem-key":
        route = "authgem" if match.group(1) == "authgem-vertex" else match.group(1)
        return route, int(match.group(2) or 0)
    return None, 0


# ---- provider grouping (presentation) -------------------------------------------------------------------

#: ModelSheet provider chips (UI_SPEC §2.2): id, label.
PROVIDER_CHIPS = (
    ("all", "All"), ("openai", "OpenAI"), ("google", "Google"), ("anthropic", "Anthropic"),
    ("deepseek", "DeepSeek"), ("xai", "xAI"), ("mistral", "Mistral"), ("openrouter", "OpenRouter"),
    ("nvidia", "NVIDIA"), ("local", "Local"), ("custom", "Custom"), ("other", "Other"),
)
_CHIP_BY_PROVIDER = {
    "openai": "openai", "authgpt": "openai",
    "gemini": "google", "authgem": "google", "authgem-vertex": "google", "authgem-key": "google",
    "vertex": "google", "google-translate": "google", "google-translate-free": "google", "search": "google",
    "anthropic": "anthropic", "authcd": "anthropic",
    "deepseek": "deepseek", "chutes": "deepseek",
    "xai": "xai", "authgrok": "xai",
    "mistral": "mistral",
    "openrouter": "openrouter",
    "nvidia": "nvidia", "authnd": "nvidia",
    "ollama": "local", "lmstudio": "local", "ollamapull": "local",
}
_PROVIDER_LABELS = {
    "openai": "OpenAI", "authgpt": "ChatGPT (sign-in)", "gemini": "Google Gemini", "authgem": "Gemini (sign-in)",
    "authgem-vertex": "Gemini Vertex (sign-in)", "authgem-key": "Gemini key route", "vertex": "Vertex AI",
    "anthropic": "Anthropic", "authcd": "Claude (sign-in)", "deepseek": "DeepSeek", "chutes": "Chutes",
    "xai": "xAI", "authgrok": "Grok (sign-in)", "mistral": "Mistral", "openrouter": "OpenRouter",
    "nvidia": "NVIDIA NIM", "authnd": "NVIDIA Build (browser)", "ollama": "Ollama", "lmstudio": "LM Studio",
    "ollamapull": "Managed Ollama", "groq": "Groq", "literouter": "LiteRouter", "opencode": "OpenCode Go",
    "opencode-zen": "OpenCode Zen", "electronhub": "ElectronHub", "nanogpt": "NanoGPT", "sambanova": "SambaNova",
    "together": "Together AI", "zai": "Z.AI", "zhipu": "Zhipu", "fireworks": "Fireworks", "cohere": "Cohere",
    "moonshot": "Moonshot", "autharena": "Arena", "antigravity": "Antigravity", "ocagy": "OCAGY",
    "search": "Search routes", "poe": "Poe", "deepl": "DeepL", "google-translate": "Google Translate",
    "google-translate-free": "Google Translate (free)", "other": "Other",
}

#: Thinking & effort schema keys per model family (Settings › Thinking & reasoning, UI_SPEC §2.2).
THINKING_FIELDS = {
    "gpt": ("enable_gpt_thinking", "gpt_effort", "openrouter_use_reasoning_tokens", "gpt_reasoning_tokens"),
    "gemini": ("enable_gemini_thinking", "thinking_level", "thinking_budget"),
    "anthropic": ("enable_anthropic_thinking", "anthropic_thinking_budget", "anthropic_force_adaptive",
                  "anthropic_effort"),
    "deepseek": ("enable_deepseek_thinking", "deepseek_effort", "deepseek_use_responses_api"),
}
FAMILY_TITLES = {
    "gpt": "GPT / OpenRouter / NIM / OpenCode",
    "gemini": "Gemini",
    "anthropic": "Anthropic",
    "deepseek": "DeepSeek / Chutes",
}
_FAMILY_BY_PROVIDER = {
    "openai": "gpt", "authgpt": "gpt", "openrouter": "gpt", "nvidia": "gpt", "authnd": "gpt", "opencode": "gpt",
    "gemini": "gemini", "authgem": "gemini", "authgem-vertex": "gemini", "authgem-key": "gemini",
    "anthropic": "anthropic", "authcd": "anthropic",
    "deepseek": "deepseek", "chutes": "deepseek",
}

_NUMBERED_PREFIX = re.compile(r"^([a-z][a-z-]*?)(\d{1,4})(?=/)")


def _options() -> Any:
    import model_options  # shared backend (on sys.path after bootstrap)

    return model_options


def normalize_custom_routes(routes: Any) -> list:
    """Validated custom prefix routes through the shared ``RunEnvMixin`` normaliser (run_env, U2)."""
    try:
        from run_env import RunEnvMixin
    except Exception as exc:  # pragma: no cover - backend missing
        log.debug("run_env unavailable: %s", exc)
        return [r for r in routes if isinstance(r, dict)] if isinstance(routes, list) else []
    return RunEnvMixin()._normalize_custom_prefix_routes(routes)


def provider_of(model: Optional[str], custom_routes: Any = None, *, options: Any = None) -> str:
    """Catalog owner of a model (``"openai"``, ``"authgpt"``, ``"custom:lan/"``), else its route prefix."""
    value = str(model or "").strip()
    lowered = value.lower()
    provider: Optional[str] = None
    try:
        provider = (options or _options()).catalog_provider_for_model(value, custom_routes)
    except Exception:
        provider = None
    if provider:
        return provider.split(":", 1)[0] if not provider.startswith("custom:") else provider
    if "/" in lowered:
        head = lowered.split("/", 1)[0]
        numbered = _NUMBERED_PREFIX.match(lowered)
        if numbered:
            head = numbered.group(1)
        return head or "other"
    if lowered in ("deepl", "google-translate", "google-translate-free"):
        return lowered
    return "other"


def provider_label(provider: str) -> str:
    if provider.startswith("custom:"):
        return f"Custom · {provider[7:]}"
    return _PROVIDER_LABELS.get(provider, provider.replace("-", " ").title() if provider else "Other")


def chip_of(provider: str) -> str:
    if provider.startswith("custom:"):
        return "custom"
    return _CHIP_BY_PROVIDER.get(provider, "other")


def family_of(model: Optional[str], provider: Optional[str] = None) -> Optional[str]:
    """Thinking family whose schema fields apply to ``model`` (presentation; see THINKING_FIELDS)."""
    provider = provider or provider_of(model)
    family = _FAMILY_BY_PROVIDER.get(provider)
    if family is None:
        lowered = str(model or "").lower().split("/")[-1]
        if lowered.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
            family = "gpt"
        elif lowered.startswith(("gemini", "gemma")):
            family = "gemini"
        elif lowered.startswith("claude"):
            family = "anthropic"
        elif lowered.startswith("deepseek"):
            family = "deepseek"
    return family


@dataclass(frozen=True)
class ModelGroup:
    provider: str
    label: str
    chip: str
    models: tuple


def group_models(models: Iterable[str], custom_routes: Any = None, *, polled: Any = None,
                 options: Any = None) -> list:
    """Provider groups in first-seen order; polled models first inside each group (UI_SPEC §2.2)."""
    groups: dict = {}
    for model in models:
        provider = provider_of(model, custom_routes, options=options)
        groups.setdefault(provider, []).append(model)
    out = []
    for provider, items in groups.items():
        if polled is not None:
            items = sorted(items, key=lambda m: 0 if polled(m) else 1)  # stable: catalog order kept
        out.append(ModelGroup(provider, provider_label(provider), chip_of(provider), tuple(items)))
    return out


def rank_models(models: Sequence[str], query: str, *, current: str = "", limit: int = MAX_RESULTS,
                options: Any = None) -> list:
    """Search ranking: exact > prefix > path-segment > contains; catalog order inside a rank.

    A numbered account prefix being typed (``authgpt2/``) renders the canonical entries as
    that alias (``model_options.numbered_model_completion_values``, the desktop completer rule).
    """
    needle = str(query or "").strip()
    values = list(models)
    if needle:
        try:
            values = list((options or _options()).numbered_model_completion_values(values, needle))
        except Exception:
            pass
    key = needle.casefold()
    if not key:
        ordered = ([current] if current else []) + [m for m in values if m.casefold() != current.casefold()]
        return ordered[:limit]
    ranked: list = []
    seen: set = set()
    for model in values:
        text = str(model)
        folded = text.casefold()
        if folded in seen:
            continue
        if folded == key:
            rank = 0
        elif folded.startswith(key):
            rank = 1
        elif any(part.startswith(key) for part in re.split(r"[/:@]", folded) if part):
            rank = 2
        elif key in folded:
            rank = 3
        else:
            continue
        seen.add(folded)
        ranked.append((rank, len(ranked), text))
    ranked.sort()
    return [text for _rank, _order, text in ranked[:limit]]


# ---- route controls (ModelSheet route row) -------------------------------------------------------------

_LOGIN_ROUTE = re.compile(r"^(authgpt|authgem-vertex|authgem-key|authgem|authcd|authgrok)(\d{0,4})/", re.IGNORECASE)
LOGIN_TITLES = {"authgpt": "ChatGPT", "authgem": "Gemini", "authcd": "Claude", "authgrok": "Grok"}


@dataclass(frozen=True)
class RouteInfo:
    """What the route row shows for a model (``settings_rules.route_controls`` + Poe/family)."""

    model: str
    provider: str = "other"
    excluded: Optional[str] = None
    excluded_detail: Optional[str] = None
    logins: tuple = ()  # login controls, desktop order: authgpt, authgrok, authcd, authgem
    account: int = 0  # the model's own slot (authgpt2/ -> 2)
    account_ids: Mapping[str, tuple] = field(default_factory=dict)
    needs_key: bool = True
    google_creds: bool = False
    google_creds_text: str = ""
    google_creds_level: str = ""
    vertex_location: bool = False
    gcp_project: bool = False
    poe: bool = False
    family: Optional[str] = None
    source: str = "mobile"  # "settings_rules" when the shared route_controls answered

    @property
    def login(self) -> Optional[str]:
        return self.logins[0] if self.logins else None

    @property
    def login_title(self) -> str:
        return LOGIN_TITLES.get(self.login or "", self.login or "")

    @property
    def account_key(self) -> str:
        """``signed_in`` key: ``authgpt`` for slot 0, ``authgpt2`` for slot 2."""
        return f"{self.login}{self.account}" if self.account else str(self.login or "")


def model_needs_api_key(model: str, custom_routes: Any = None) -> bool:
    """``UnifiedClient._model_needs_api_key`` (the authoritative list); True when unavailable.

    The client reads custom prefix routes from the run environment; outside a run the configured
    ``custom_routes`` are checked with the client's own local-URL rule (a LAN Ollama / LM Studio
    route needs no key)."""
    try:
        from unified_api_client import UnifiedClient
    except Exception:
        return True
    try:
        lowered = str(model or "").strip().lower()
        for route in normalize_custom_routes(custom_routes) if custom_routes else ():
            prefix = str(route.get("prefix") or "").lower()
            if prefix and lowered.startswith(prefix):
                if UnifiedClient._is_local_openai_base_url(route.get("routing", "")):
                    return False
                break
        return bool(UnifiedClient._model_needs_api_key(model))
    except Exception:
        return True


LOCAL_DUMMY_KEY = "dummy-key-for-local-llm"  # what UnifiedClient sends to keyless local endpoints


def is_local_url(url: str) -> bool:
    try:
        from unified_api_client import UnifiedClient

        return bool(UnifiedClient._is_local_openai_base_url(url))
    except Exception:
        lowered = str(url or "").lower()
        return any(part in lowered for part in ("localhost", "127.0.0.1", "192.168.", "10.", ":11434", ":1234"))


def route_info(model: Optional[str], config: Optional[Mapping[str, Any]] = None, *,
               needs_key: Optional[bool] = None, rules: Any = None) -> RouteInfo:
    """Blocking (may import the backend): the route row for ``model``.

    ``settings_rules.route_controls(model, cfg)`` (U4: the desktop ``on_model_change`` decisions,
    key-pool scans included) supplies logins, Google credentials, Vertex location, the GCP project
    picker, the exclusion and ``needs_api_key``; Poe (desktop ``_check_poe_model``) and the thinking
    family are added here."""
    value = str(model or "").strip()
    config = dict(config or {})
    routes = config.get("custom_prefix_routes", [])
    provider = provider_of(value, routes)
    lowered = value.lower()
    route, account = login_route(value)
    info: dict = dict(model=value, provider=provider, excluded=excluded_route(value),
                      excluded_detail=excluded_detail(value), logins=(route,) if route else (), account=account,
                      poe=lowered.startswith("poe/"), family=family_of(value, provider))
    rules = rules if rules is not None else _rules()
    route_controls = getattr(rules, "route_controls", None) if rules is not None else None
    rc = None
    if callable(route_controls):
        try:
            rc = route_controls(value, config, platform="mobile")
        except Exception as exc:
            log.debug("route_controls failed for %s: %s", value, exc)
    if rc is not None:
        info.update(
            source="settings_rules",
            logins=tuple(getattr(rc, "logins", ()) or ()),
            account_ids=dict(getattr(rc, "account_ids", {}) or {}),
            google_creds=bool(getattr(rc, "needs_google_creds", False)),
            google_creds_text=str(getattr(rc, "google_creds_text", "") or ""),
            google_creds_level=str(getattr(rc, "google_creds_level", "") or ""),
            vertex_location=bool(getattr(rc, "vertex_location", False)),
            gcp_project=bool(getattr(rc, "authgem_vertex", False)),
        )
        rc_key = getattr(rc, "needs_api_key", None)
        if needs_key is None and rc_key is not None:
            needs_key = bool(rc_key)
    if needs_key is None:
        needs_key = model_needs_api_key(value, routes)
    elif needs_key and routes:
        needs_key = model_needs_api_key(value, routes)
    info["needs_key"] = bool(needs_key)
    return RouteInfo(**info)


# ---- custom prefix rows (Model Manager › Custom prefixes) -------------------------------------------------

def validate_prefix_rows(rows: Sequence[Mapping[str, Any]]) -> tuple:
    """``(routes, None)`` or ``(None, (title, message))``: ``model_catalog_core.validate_custom_prefix_routes``
    (the desktop Model Manager table checks and messages)."""
    fn = CatalogCore.load().fn("validate_custom_prefix_routes")
    if fn is None:
        return None, ("Unavailable", CORE_MISSING)
    result = fn([dict(r) for r in rows])
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, None


# ---- shared core binding --------------------------------------------------------------------------------


class CatalogCore:
    """``model_catalog_core`` (U4) when it is in the build; ``fn(name)`` is None otherwise."""

    _cached: Optional["CatalogCore"] = None
    _lock = threading.Lock()

    def __init__(self, module: Any = None, error: Optional[str] = None) -> None:
        self.module = module
        self.error = error

    @classmethod
    def load(cls, module_name: str = "model_catalog_core") -> "CatalogCore":
        with cls._lock:
            if cls._cached is not None and cls._cached.module is not None:
                return cls._cached
            try:
                if module_name == "model_catalog_core":
                    import model_catalog_core as module  # literal: tools/collect_backend.py bundles it
                else:
                    module = importlib.import_module(module_name)
                core = cls(module)
            except Exception as exc:
                core = cls(None, f"{type(exc).__name__}: {exc}")
            cls._cached = core
            return core

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._cached = None

    @property
    def available(self) -> bool:
        return self.module is not None

    def fn(self, *names: str) -> Optional[Callable[..., Any]]:
        module = self.module
        if module is None:
            return None
        for name in names:
            candidate = getattr(module, name, None)
            if callable(candidate):
                return candidate
        return None


CORE_MISSING = ("This edit needs the shared model catalog core (model_catalog_core), which is not in this "
                "build. The list stays readable; config.json is untouched.")


# ---- snapshots --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogSnapshot:
    models: tuple = ()  # the picker list: custom_model_list merged with the catalog, minus tombstones
    removed: tuple = ()  # model_manager_removed_models (tombstones)
    polled: frozenset = frozenset()  # casefolded ids confirmed by a poll within 7 days
    polled_by_provider: Mapping[str, frozenset] = field(default_factory=dict)
    statuses: Mapping[str, str] = field(default_factory=dict)
    polling: frozenset = frozenset()  # providers being polled now ("*" = full refresh)
    hide_unpolled: bool = False
    custom_routes: tuple = ()
    loaded: bool = False
    error: Optional[str] = None
    updated_at: float = 0.0

    def is_polled(self, model: str) -> bool:
        try:
            return bool(_options().model_has_polled_marker(model, self.polled))
        except Exception:
            return str(model or "").strip().casefold() in self.polled

    def visible_models(self) -> list:
        if not self.hide_unpolled:
            return list(self.models)
        return [m for m in self.models if self.is_polled(m)]

    def is_polling(self, provider: Optional[str] = None) -> bool:
        if provider is None:
            return bool(self.polling)
        return provider in self.polling or "*" in self.polling


@dataclass(frozen=True)
class RefreshOutcome:
    provider: Optional[str]
    ok: bool
    online: tuple = ()
    statuses: Mapping[str, str] = field(default_factory=dict)
    new_models: int = 0
    total: int = 0
    message: str = ""
    skipped: bool = False


def _casefold_set(values: Iterable[Any]) -> set:
    return {str(v).strip().casefold() for v in values or () if str(v).strip()}


# ---- service ----------------------------------------------------------------------------------------------


class ModelCatalogService:
    """Owns the catalog snapshot shown by the ModelSheet, ModelPicker and Model Manager."""

    def __init__(
        self,
        config: Any,
        *,
        options: Any = None,
        core: Optional[CatalogCore] = None,
        run_io: Optional[Callable[..., Any]] = None,
        post: Optional[Callable[..., Any]] = None,
        clock: Callable[[], float] = time.time,
        auto_poll_ttl: int = AUTO_POLL_TTL_SECONDS,
        is_mobile: Optional[bool] = None,
    ) -> None:
        self.config = config
        self._options = options
        self._core = core
        self.run_io = run_io
        self.post = post
        self.clock = clock
        self.auto_poll_ttl = int(auto_poll_ttl)
        self._is_mobile = is_mobile
        self._lock = threading.RLock()
        self._snapshot = CatalogSnapshot()
        self._listeners: list = []
        self._polling: set = set()
        self._refresh_lock = threading.Lock()  # one catalog poll at a time (shared cache file)
        self._auto_checked: dict = {}  # provider -> clock() of the last auto-poll decision
        self.last_outcome: Optional[RefreshOutcome] = None
        self.last_result: Any = None  # the last model_options.ModelCatalogRefreshResult

    # ---- plumbing ----------------------------------------------------------------------------------

    @property
    def options(self) -> Any:
        if self._options is None:
            self._options = _options()
        return self._options

    @property
    def core(self) -> CatalogCore:
        if self._core is None:
            self._core = CatalogCore.load()
        return self._core

    @property
    def snapshot(self) -> CatalogSnapshot:
        with self._lock:
            return self._snapshot

    def subscribe(self, callback: Callable[[CatalogSnapshot], Any]) -> Callable[[], None]:
        self._listeners.append(callback)

        def unsubscribe() -> None:
            try:
                self._listeners.remove(callback)
            except ValueError:
                pass

        return unsubscribe

    def _publish(self, **changes: Any) -> CatalogSnapshot:
        with self._lock:
            changes.setdefault("polling", frozenset(self._polling))
            changes.setdefault("updated_at", self.clock())
            self._snapshot = dataclasses.replace(self._snapshot, **changes)
            snap = self._snapshot
        for callback in list(self._listeners):
            try:
                if self.post is not None:
                    self.post(callback, snap)
                else:
                    callback(snap)
            except Exception:
                log.exception("catalog listener failed")
        return snap

    async def _io(self, fn: Callable[..., Any], *args: Any) -> Any:
        if self.run_io is not None:
            return await self.run_io(fn, *args)
        return await asyncio.to_thread(fn, *args)

    def _get(self, key: str, default: Any = None) -> Any:
        try:
            return self.config.get(key, default)
        except Exception:
            return default

    def cache_path(self) -> str:
        """The catalog cache file (``GLOSSARION_MODEL_CATALOG_CACHE`` from the bootstrap env contract)."""
        try:
            return str(self.options._model_catalog_cache_path())
        except Exception:
            return str(os.environ.get("GLOSSARION_MODEL_CATALOG_CACHE", ""))

    def mobile(self) -> bool:
        if self._is_mobile is not None:
            return self._is_mobile
        try:
            import mobile_runtime

            return bool(mobile_runtime.is_mobile())
        except Exception:
            return False

    # ---- config views ------------------------------------------------------------------------------

    def custom_routes(self) -> list:
        return normalize_custom_routes(self._get("custom_prefix_routes", []))

    def removed_models(self) -> list:
        removed = self._get("model_manager_removed_models", [])
        return [str(m).strip() for m in removed if str(m).strip()] if isinstance(removed, list) else []

    def saved_models(self) -> Optional[list]:
        saved = self._get("custom_model_list", None)
        return list(saved) if isinstance(saved, list) else None

    def hide_unpolled(self) -> bool:
        return bool(self._get("model_manager_hide_unpolled_models", False))

    def set_hide_unpolled(self, value: bool) -> None:
        """The desktop toggle writes the key at once (it filters every picker, not the saved list)."""
        self.config.set("model_manager_hide_unpolled_models", bool(value))
        self._publish(hide_unpolled=bool(value))

    # ---- loading ---------------------------------------------------------------------------------------

    def polled_state(self) -> dict:
        """Provider -> casefolded model ids confirmed within seven days (``polled_models_by_provider``)."""
        try:
            catalogs = self.options.get_current_polled_provider_models()
        except Exception:
            catalogs = {}
        fn = self.core.fn("polled_models_by_provider")
        by_provider = fn(catalogs or {}) if fn is not None else {
            str(p): _casefold_set(models) for p, models in (catalogs or {}).items()}
        return {str(p): frozenset(ids) for p, ids in by_provider.items()}

    def _polled_keys(self, by_provider: Mapping[str, Iterable[str]]) -> frozenset:
        """Every confirmed id as ``model_options.PolledModelKeys`` (normalised once per snapshot)."""
        fn = self.core.fn("polled_model_keys")
        try:
            if fn is not None:
                return fn(by_provider)
            return self.options.PolledModelKeys(m for ids in by_provider.values() for m in ids)
        except Exception:
            return frozenset(m for ids in by_provider.values() for m in ids)

    def build_models(self, online_models: Optional[Sequence[str]] = None) -> list:
        """Blocking: the picker list (desktop ``merge_saved_model_options(custom, catalog, removed)``)."""
        discovered = list(online_models) if online_models is not None else list(self.options.get_model_options())
        return list(self.options.merge_saved_model_options(self.saved_models(), discovered, self.removed_models()))

    def load_blocking(self) -> CatalogSnapshot:
        try:
            models = self.build_models()
            by_provider = self.polled_state()
            error = None
        except Exception as exc:
            log.warning("model catalog unavailable: %s", exc)
            models, by_provider, error = list(self.snapshot.models), dict(self.snapshot.polled_by_provider), str(exc)
        polled = self._polled_keys(by_provider)
        return self._publish(models=tuple(models), removed=tuple(self.removed_models()), polled=polled,
                             polled_by_provider=by_provider, hide_unpolled=self.hide_unpolled(),
                             custom_routes=tuple(self.custom_routes()), loaded=True, error=error)

    async def load(self) -> CatalogSnapshot:
        return await self._io(self.load_blocking)

    def reload_from_config(self) -> CatalogSnapshot:
        """Cheap re-merge after a config edit (no catalog file read when already loaded)."""
        return self.load_blocking()

    # ---- credentials ----------------------------------------------------------------------------------

    def provider_keys(self, custom_routes: Any = None) -> dict:
        """First enabled key-pool key per catalog provider (explicit credentials for polls)."""
        keys: dict = {}
        for pool_key in ("multi_api_keys", "fallback_keys", "glossary_keys", "glossary_refinement_keys",
                         "qa_scan_keys", "metadata_keys"):
            entries = self._get(pool_key, [])
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict) or not entry.get("enabled", True):
                    continue
                api_key = str(entry.get("api_key") or "").strip()
                if not api_key or api_key.startswith("ENC:"):
                    continue
                try:
                    provider = self.options.catalog_provider_for_model(str(entry.get("model") or ""), custom_routes)
                except Exception:
                    provider = None
                if provider and not provider.startswith("custom:"):
                    keys.setdefault(provider, api_key)
        return keys

    def credentials(self, provider: Optional[str] = None, *, representative: Optional[str] = None) -> dict:
        """Explicit poll credentials. The configured ``api_key`` is only sent to the provider of the
        configured model; a group refresh for another provider uses that provider's pool key."""
        routes = self.custom_routes()
        model = str(self._get("model", "") or "").strip()
        api_key = str(self._get("api_key", "") or "").strip()
        if api_key.startswith("ENC:"):
            api_key = ""
        try:
            model_provider = self.options.catalog_provider_for_model(model, routes)
        except Exception:
            model_provider = None
        active_model, active_key = model, api_key
        if provider is not None and model_provider != provider:
            active_model = representative or ""
            active_key = ""
        if provider is not None and provider.startswith("custom:") and not active_key:
            # A keyless LAN server (Ollama / LM Studio) behind a custom prefix: the client sends
            # its local dummy key; the catalog poll needs one to try the route at all.
            route = next((r for r in routes if str(r.get("prefix", "")).lower() == provider[7:]), None)
            if route is not None and is_local_url(route.get("routing", "")):
                active_key = LOCAL_DUMMY_KEY
        return {"active_model": active_model, "active_api_key": active_key,
                "provider_keys": self.provider_keys(routes), "custom_routes": routes}

    def representative_model(self, provider: str) -> str:
        """A model of ``provider`` (authenticated/custom catalogs resolve their target from it)."""
        routes = list(self.snapshot.custom_routes)
        for model in self.snapshot.models:
            try:
                owner = self.options.catalog_provider_for_model(model, routes)
            except Exception:
                owner = None
            if owner == provider:
                return model
        if provider.startswith("custom:"):
            return provider[7:]
        return ""

    # ---- refresh ----------------------------------------------------------------------------------------

    def refresh_blocking(self, provider: Optional[str] = None, *, explicit: bool = True,
                         timeout: float = 8.0) -> RefreshOutcome:
        """One provider (``only_provider``) or every eligible provider; applies the result.

        Polls are serialized: ``model_options`` reads, merges and rewrites one cache file per poll,
        so a full refresh requested during a provider poll runs right after it (desktop: "a full
        refresh is queued next")."""
        with self._refresh_lock:
            return self._refresh_locked(provider, explicit=explicit, timeout=timeout)

    def _refresh_locked(self, provider: Optional[str], *, explicit: bool, timeout: float) -> RefreshOutcome:
        options = self.options
        representative = self.representative_model(provider) if provider else None
        creds = self.credentials(provider, representative=representative)
        # model_options guards its autharena_proxy / ocagy_cli imports (not bundled on mobile): a full
        # refresh reports Arena / OcAgy / OpenCode Zen as "unavailable in this build" instead of failing;
        # apply_refresh shows every excluded provider as EXCLUDED_STATUS.
        previous_models = list(self.snapshot.models)
        try:
            result = options.refresh_provider_model_catalogs(
                active_model=creds["active_model"], active_api_key=creds["active_api_key"],
                provider_keys=creds["provider_keys"], custom_routes=creds["custom_routes"],
                timeout=timeout, only_provider=provider,
            )
        except Exception as exc:  # ImportError included: a module this build does not bundle
            log.warning("catalog refresh failed: %s", exc)
            return RefreshOutcome(provider, False, message=f"Catalog refresh failed: {exc}")
        if explicit:
            try:
                result = dataclasses.replace(result, restore_removed_models=True)
            except Exception:
                pass
        self.last_result = result
        return self.apply_refresh(result, previous_models=previous_models)

    def apply_refresh(self, result: Any, *, previous_models: Optional[Sequence[str]] = None) -> RefreshOutcome:
        """``model_catalog_core.apply_provider_refresh`` on the config keys it touches (written back
        sparsely): an explicit poll re-adds the confirmed models (tombstones cleared), the picker
        list is re-merged and successful providers replace their seven-day markers."""
        previous = list(previous_models) if previous_models is not None else list(self.snapshot.models)
        # an excluded provider's poll status (a desktop helper module, a localhost proxy) is not a failure
        statuses = mobile_statuses(getattr(result, "statuses", {}) or {})
        requested = getattr(result, "requested_provider", None)
        fn = self.core.fn("apply_provider_refresh")
        if fn is None:
            return RefreshOutcome(requested, False, statuses=statuses, message=CORE_MISSING)
        holder: dict = {}

        def call(apply: Callable[..., Any], view: dict) -> bool:
            holder["refresh"] = apply(view, result, previous_models=previous,
                                      polled_by_provider={p: set(ids) for p, ids in self.polled_state().items()})
            return holder["refresh"].restored is not False

        ok, error = self._run_core_edit(("apply_provider_refresh",), call, reload=False)
        refresh = holder.get("refresh")
        if refresh is None or not getattr(refresh, "applied", False):
            message = error or str(statuses.get(requested, "")) or "No online catalog responded; using static fallbacks."
            self._publish(statuses=statuses)
            outcome = RefreshOutcome(requested, False, statuses=statuses, message=message)
            self.last_outcome = outcome
            return outcome
        by_provider = {str(p): frozenset(ids) for p, ids in refresh.polled_by_provider.items()}
        polled = self._polled_keys(by_provider)
        models = list(refresh.display_models)
        self._publish(models=tuple(models), removed=tuple(self.removed_models()), polled=polled,
                      polled_by_provider=by_provider, statuses=statuses, hide_unpolled=self.hide_unpolled(),
                      custom_routes=tuple(self.custom_routes()), loaded=True, error=None)
        if requested:
            message = refresh.auto_poll_message or ""
            ok = str(statuses.get(requested, "")).startswith("online")
            new_count = len(_casefold_set(dict(getattr(result, "provider_models", {}) or {}).get(requested, []) or [])
                            - _casefold_set(previous))
        else:
            message = refresh.status_text
            ok = bool(refresh.online)
            new_count = 0
        if error and not message:
            message = error
        outcome = RefreshOutcome(requested, ok, tuple(sorted(refresh.online)), statuses, new_count, len(models), message)
        self.last_outcome = outcome
        return outcome

    def manager_poll_models(self, previous_models: Sequence[str], known_keys: Iterable[str],
                            outcome_result: Any = None) -> Optional[list]:
        """The Model Manager list after "Poll Providers" (``model_catalog_core.manager_poll_models``)."""
        result = outcome_result if outcome_result is not None else self.last_result
        fn = self.core.fn("manager_poll_models")
        if fn is None or result is None:
            return None
        online_models = list(getattr(result, "models", []) or [])
        if not online_models:
            return None
        confirmed = self.core.fn("confirmed_catalog_models")
        keys = confirmed(dict(getattr(result, "provider_models", {}) or {}))[1] if confirmed is not None else set()
        return list(fn(list(previous_models), set(known_keys), self.removed_models(), online_models,
                       explicit_poll=True, confirmed_model_keys=keys))

    async def refresh(self, provider: Optional[str] = None, *, explicit: bool = True) -> RefreshOutcome:
        """Refresh online models (one provider, ignoring the 24 h TTL, or all) off the UI loop."""
        tag = provider or "*"
        with self._lock:
            if tag in self._polling or (tag != "*" and "*" in self._polling):
                return RefreshOutcome(provider, False, message="A catalog poll is already running", skipped=True)
            self._polling.add(tag)
        self._publish()
        try:
            return await self._io(lambda: self.refresh_blocking(provider, explicit=explicit))
        finally:
            with self._lock:
                self._polling.discard(tag)
            self._publish()

    def due_provider(self, model: Optional[str] = None) -> Optional[str]:
        """Blocking: the provider to auto-poll for ``model`` (24 h TTL), desktop
        ``_auto_poll_current_provider_catalog`` gating; never for excluded routes."""
        model = str(model if model is not None else self._get("model", "") or "").strip()
        if not model or excluded_route(model):
            return None
        options = self.options
        routes = self.custom_routes()
        configured = str(self._get("model", "") or "").strip()
        api_key = str(self._get("api_key", "") or "").strip() if model == configured else ""
        if api_key.startswith("ENC:"):
            api_key = ""
        try:
            anonymous = options.provider_model_catalog_supports_anonymous_poll(model, routes)
        except Exception:
            anonymous = False
        if not api_key and not anonymous:
            try:
                from unified_api_client import UnifiedClient

                if UnifiedClient._model_needs_api_key(model):
                    provider = options.catalog_provider_for_model(model, routes)
                    if not provider or provider not in self.provider_keys(routes):
                        return None
            except Exception:
                return None
        try:
            provider = options.due_provider_catalog_for_model(model, api_key, routes, max_age=self.auto_poll_ttl)
        except Exception as exc:
            log.debug("auto-poll check failed for %s: %s", model, exc)
            return None
        return provider or None

    async def maybe_auto_poll(self, model: Optional[str] = None, *, min_interval: float = 60.0) -> Optional[RefreshOutcome]:
        """24 h auto-poll of the selected model's provider (automatic: tombstones stay)."""
        provider = await self._io(self.due_provider, model)
        if not provider:
            return None
        now = self.clock()
        with self._lock:
            last = self._auto_checked.get(provider)
            if last is not None and now - last < min_interval:
                return None
            self._auto_checked[provider] = now
        return await self.refresh(provider, explicit=False)

    # ---- Model Manager edits (model_catalog_core) ---------------------------------------------------

    def _apply_changes(self, changes: Mapping[str, Any]) -> None:
        values = {k: v for k, v in changes.items() if v is not _DROP}
        drops = [k for k, v in changes.items() if v is _DROP]
        if values:
            self.config.set_many(values)
        for key in drops:
            unset = getattr(self.config, "unset", None)
            if unset is not None:
                unset(key)

    def _config_view(self) -> dict:
        view: dict = {}
        for key in ("custom_model_list", "model_manager_removed_models", "model_manager_hide_unpolled_models",
                    "custom_prefix_routes", "model"):
            value = self._get(key, _DROP)
            if value is not _DROP:
                view[key] = copy.deepcopy(value)
        return view

    def _run_core_edit(self, names: Sequence[str], call: Callable[[Callable[..., Any], dict], Any], *,
                       reload: bool = True) -> tuple:
        """Run a core edit on a copy of the config keys it touches; write back the keys it changed
        (sparse). Returns (ok, error_message)."""
        fn = self.core.fn(*names)
        if fn is None:
            return False, CORE_MISSING
        view = self._config_view()
        before = copy.deepcopy(view)
        try:
            result = call(fn, view)
        except Exception as exc:
            log.exception("model catalog edit failed")
            return False, str(exc)
        if result is False:
            return False, "config.json could not be written"
        changes: dict = {}
        for key in set(before) | set(view):
            if key not in view:
                changes[key] = _DROP
            elif key not in before or before[key] != view[key]:
                changes[key] = view[key]
        if isinstance(result, Mapping):
            changes.update(result)
        self._apply_changes(changes)
        if reload:
            self.load_blocking()
        return True, None

    def save_order(self, new_order: Sequence[str], previous_models: Optional[Sequence[str]] = None) -> tuple:
        """Model Manager save: ``custom_model_list`` + tombstones for models taken out (core)."""
        order = [str(m).strip() for m in new_order if str(m).strip()]
        if not order:
            return False, "The model list cannot be empty. Add at least one model."
        previous = list(previous_models) if previous_models is not None else list(self.snapshot.models)
        return self._run_core_edit(("save_model_order",), lambda fn, view: fn(view, order, previous))

    def restore_removed(self, models: Sequence[str], *, add_to_saved: bool = False) -> tuple:
        """Explicit re-add (manual entry or a user-triggered poll) clears exact tombstones (core)."""
        values = [str(m).strip() for m in models if str(m).strip()]
        if not values:
            return False, None
        removed = _casefold_set(self.removed_models())
        if not removed & _casefold_set(values):
            return False, None
        return self._run_core_edit(("restore_removed_models",),
                                   lambda fn, view: fn(view, values, add_to_saved=add_to_saved) is not False)

    def add_model(self, model: str) -> tuple:
        """Model Manager "Add": inserted at the top (desktop ``_add_model``); re-adding a removed id
        clears its tombstone."""
        text = str(model or "").strip()
        if not text:
            return False, None
        current = list(self.snapshot.models)
        if any(m == text for m in current):
            return False, f"'{text}' is already in the list."
        return self.save_order([text] + current, current)  # also clears a tombstone of ``text``

    def remove_models(self, models: Sequence[str]) -> tuple:
        drop = {str(m) for m in models}
        current = list(self.snapshot.models)
        return self.save_order([m for m in current if m not in drop], current)

    def move_model(self, old_index: int, new_index: int) -> tuple:
        current = list(self.snapshot.models)
        if not (0 <= old_index < len(current)):
            return False, None
        item = current.pop(old_index)
        new_index = max(0, min(int(new_index), len(current)))
        current.insert(new_index, item)
        return self.save_order(current, list(self.snapshot.models))

    def reset_to_defaults(self) -> tuple:
        """Desktop "🔄 Reset": the built-in + cached catalog replaces the list."""
        current = list(self.snapshot.models)
        return self.save_order(list(self.options.get_model_options()), current)

    def save_prefix_routes(self, rows: Sequence[Mapping[str, Any]]) -> tuple:
        """Validate and store ``custom_prefix_routes`` (the run env exports them at job start)."""
        routes, error = validate_prefix_rows(rows)
        if routes is None:
            return False, error
        self.config.set("custom_prefix_routes", routes)
        self.load_blocking()
        return True, None


class _Drop:
    def __repr__(self) -> str:
        return "DROP"


_DROP: Any = _Drop()


# ---- app-wide default ---------------------------------------------------------------------------------

_DEFAULT: Optional[ModelCatalogService] = None


def set_default_service(service: Optional[ModelCatalogService]) -> None:
    """Installed by ModelsKeysFeature so pickers built elsewhere (chat header, U3 alias) share it."""
    global _DEFAULT
    _DEFAULT = service


def default_service() -> Optional[ModelCatalogService]:
    return _DEFAULT
