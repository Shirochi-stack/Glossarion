"""Host tests for U4 models & keys: ModelCatalogService, ModelSheet, Model Manager, Multi-Key
Manager, KeyEditor, Refusal patterns, Endpoints and Local AI.

Run from src/mobile (Flet venv or the 3.13 review venv):
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_models_keys.py

* The service and controllers are tested with fakes for ``model_options`` polls and the
  ``key_pool_service`` module (so they run before / without the real service and never touch the
  network); the shared ``model_catalog_core`` / ``settings_rules`` modules are the real ones.
* Tests that need backend packages the Flet venv lacks (``run_env`` needs bs4,
  ``unified_api_client`` needs requests) skip there and run in the 3.13 venv.
* Screens are built in the in-memory fake Flet session of ``test_bootstrap``; every config.json
  lives in a temp dir and is written through ``MobileConfigStore`` (plain JSON reader/writer).
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import os
import sys
import threading
import types
from pathlib import Path
from typing import Optional

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")
os.environ.setdefault("GLOSSARION_HEADLESS_KEY_MANAGER", "1")

from glossarion_mobile.services import model_catalog as mc  # noqa: E402
from glossarion_mobile.state.config_store import MobileConfigStore  # noqa: E402
from glossarion_mobile.ui.router import KEY_POOLS, parse_route  # noqa: E402


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _importable(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed")
needs_run_env = pytest.mark.skipif(not _importable("run_env"), reason="run_env needs backend packages (bs4)")
needs_client = pytest.mark.skipif(not _importable("unified_api_client"), reason="unified_api_client not importable")

_TB_SPEC = importlib.util.spec_from_file_location("_glossarion_tb_helpers_models", Path(__file__).with_name("test_bootstrap.py"))
_TB = importlib.util.module_from_spec(_TB_SPEC)
_TB_SPEC.loader.exec_module(_TB)
_fake_session = _TB._fake_session


# ==========================================================================
# helpers
# ==========================================================================


def _store(tmp_path, data: Optional[dict] = None, defaults=None) -> MobileConfigStore:
    path = tmp_path / "config.json"
    if data is not None:
        path.write_text(json.dumps(data), encoding="utf-8")
    store = MobileConfigStore(path, debounce=30, defaults=defaults,
                              reader=lambda p, decrypt=True: json.loads(Path(p).read_text(encoding="utf-8")),
                              writer=lambda disk, p, backup=False: Path(p).write_text(json.dumps(disk), encoding="utf-8"))
    store.load()
    return store


def _on_disk(store: MobileConfigStore) -> dict:
    store.flush()
    return json.loads(Path(store.path).read_text(encoding="utf-8"))


class FakeOptions:
    """model_options with the real pure helpers and fake catalog / polls (no network, no cache file)."""

    def __init__(self, catalog=None, polled=None, results=None):
        import model_options as real

        self.real = real
        self.catalog = list(catalog or ["gpt-6", "gpt-6-mini", "claude-opus-5-5", "gemini-3.5-flash",
                                        "authgpt/gpt-6-luna", "ocz/qwen3-8b"])
        self.polled = dict(polled or {})
        self.results = list(results or [])
        self.refresh_calls: list = []
        self.due = None
        self.due_calls: list = []
        self.merge_saved_model_options = real.merge_saved_model_options
        self.catalog_provider_for_model = real.catalog_provider_for_model
        self.numbered_model_completion_values = real.numbered_model_completion_values
        self.model_has_polled_marker = real.model_has_polled_marker
        self.PolledModelKeys = real.PolledModelKeys
        self.ModelCatalogRefreshResult = real.ModelCatalogRefreshResult

    def get_model_options(self):
        return list(self.catalog)

    def get_current_polled_provider_models(self):
        return {k: list(v) for k, v in self.polled.items()}

    def refresh_provider_model_catalogs(self, **kwargs):
        self.refresh_calls.append(kwargs)
        if self.results:
            result = self.results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        return self.ModelCatalogRefreshResult(list(self.catalog), {}, {"openai": "static fallback (no provider credential)"},
                                              kwargs.get("only_provider"))

    def provider_model_catalog_supports_anonymous_poll(self, model, routes=None):
        return False

    def due_provider_catalog_for_model(self, model, api_key="", routes=None, *, max_age=0):
        self.due_calls.append((model, api_key))
        return self.due


def _service(store, options=None, **kw) -> mc.ModelCatalogService:
    return mc.ModelCatalogService(store, options=options or FakeOptions(), is_mobile=False, **kw)


# ==========================================================================
# ModelCatalogService (pure)
# ==========================================================================


def test_excluded_routes_reasons_come_from_settings_rules():
    import settings_rules

    for prefix in settings_rules.EXCLUDED_ROUTE_PREFIXES:
        model = prefix + ("x" if prefix.endswith("/") else "/x")
        assert mc.excluded_route(model), model
        detail = mc.excluded_detail(model)
        assert settings_rules.excluded_route_reason(model) in detail and mc.KEPT_NOTE in detail
    assert mc.excluded_route("ocz/qwen3") == "ocz/ isn't available on mobile"  # U13: ollamapull/ runs via a PC
    assert mc.excluded_route("ollama/qwen3") is None and mc.excluded_route("authgpt/gpt-6-luna") is None
    assert mc.provider_excluded("opencode-zen") and mc.provider_excluded("autharena") and not mc.provider_excluded("openai")
    # the U3 module is an alias: same names, same objects
    from glossarion_mobile.ui.sheets import model_sheet_min

    assert model_sheet_min.excluded_route is mc.excluded_route
    assert model_sheet_min.EXCLUDED_ROUTE_PREFIXES == mc.EXCLUDED_ROUTE_PREFIXES


def test_rank_models_exact_prefix_segment_contains_and_account_aliases():
    models = ["xgpt-6", "gpt-6-mini", "or/openai/gpt-6", "gpt-6", "claude-opus-5-5", "authgpt/gpt-6-luna"]
    ranked = mc.rank_models(models, "gpt-6")
    assert ranked[0] == "gpt-6" and ranked[1] == "gpt-6-mini"
    assert ranked.index("or/openai/gpt-6") < ranked.index("xgpt-6")  # path segment beats contains
    assert "claude-opus-5-5" not in ranked
    assert mc.rank_models(models, "")[:1] == ["xgpt-6"]
    assert mc.rank_models(models, "", current="gpt-6")[0] == "gpt-6"
    # typing a numbered account prefix renders the canonical route as that alias (desktop completer)
    assert mc.rank_models(models, "authgpt2/") == ["authgpt2/gpt-6-luna"]
    assert mc.login_route("authgpt2/gpt-6-luna") == ("authgpt", 2)
    assert mc.login_route("authgem-vertex3/gemini-3") == ("authgem", 3)
    assert mc.login_route("gpt-6") == (None, 0)


def test_group_models_puts_polled_first_and_labels_custom_routes():
    routes = [{"prefix": "lan/", "routing": "http://192.168.1.2:11434/v1", "endpoint_type": "/chat/completions"}]
    polled = {"gpt-6-mini"}
    groups = mc.group_models(["gpt-6", "gpt-6-mini", "claude-opus-5-5", "lan/qwen3"], routes,
                             polled=lambda m: m in polled)
    by = {g.provider: g for g in groups}
    assert by["openai"].models == ("gpt-6-mini", "gpt-6") and by["openai"].chip == "openai"
    assert by["anthropic"].label == "Anthropic"
    assert by["custom:lan/"].chip == "custom" and by["custom:lan/"].label == "Custom · lan/"
    assert mc.family_of("authgpt/gpt-6-luna") == "gpt" and mc.family_of("claude-opus-5-5") == "anthropic"
    assert mc.family_of("deepseek-chat") == "deepseek" and mc.family_of("ollama/qwen3") is None


def test_cache_path_follows_the_env_contract(tmp_path, monkeypatch):
    target = tmp_path / "cache" / "model_catalog_cache.json"
    monkeypatch.setenv("GLOSSARION_MODEL_CATALOG_CACHE", str(target))
    store = _store(tmp_path, {})
    assert os.path.normcase(mc.ModelCatalogService(store).cache_path()) == os.path.normcase(str(target))
    store._saver.close()


def test_load_merges_saved_order_catalog_and_tombstones(tmp_path):
    store = _store(tmp_path, {"custom_model_list": ["my-model", "gpt-6"],
                              "model_manager_removed_models": ["claude-opus-5-5"]})
    options = FakeOptions(polled={"openai": ["GPT-6"]})
    service = _service(store, options)
    seen = []
    service.subscribe(seen.append)
    snap = service.load_blocking()
    assert snap.models[:2] == ("my-model", "gpt-6")
    assert "claude-opus-5-5" not in snap.models and "gemini-3.5-flash" in snap.models
    assert snap.is_polled("gpt-6") and not snap.is_polled("gpt-6-mini") and seen and seen[-1] is snap
    service.set_hide_unpolled(True)
    assert service.snapshot.visible_models() == ["gpt-6"]
    assert _on_disk(store)["model_manager_hide_unpolled_models"] is True
    store._saver.close()


def test_credentials_never_send_the_main_key_to_another_provider(tmp_path):
    store = _store(tmp_path, {
        "model": "gpt-6", "api_key": "sk-main-key-123",
        "multi_api_keys": [{"api_key": "sk-ant-pool", "model": "claude-opus-5-5", "enabled": True},
                           {"api_key": "sk-off", "model": "deepseek-chat", "enabled": False}],
        "custom_prefix_routes": [{"prefix": "lan/", "routing": "http://192.168.1.20:11434/v1",
                                  "endpoint_type": "/chat/completions"}],
    })
    service = _service(store)
    own = service.credentials("openai")
    assert own["active_model"] == "gpt-6" and own["active_api_key"] == "sk-main-key-123"
    other = service.credentials("anthropic", representative="claude-opus-5-5")
    assert other["active_api_key"] == "" and other["active_model"] == "claude-opus-5-5"
    assert other["provider_keys"] == {"anthropic": "sk-ant-pool"}  # disabled pool keys are not used
    lan = service.credentials("custom:lan/", representative="lan/")
    assert lan["active_api_key"] == mc.LOCAL_DUMMY_KEY and lan["custom_routes"][0]["prefix"] == "lan/"
    store._saver.close()


def _result(options, models, provider_models, statuses, requested=None):
    return options.ModelCatalogRefreshResult(models, provider_models, statuses, requested)


def test_explicit_refresh_clears_tombstones_automatic_keeps_them(tmp_path):
    store = _store(tmp_path, {"model": "gpt-6", "api_key": "sk-x", "model_manager_removed_models": ["gpt-7"]})
    options = FakeOptions(catalog=["gpt-6"])
    service = _service(store, options)
    service.load_blocking()
    auto = _result(options, ["gpt-6", "gpt-7"], {"openai": ["gpt-6", "gpt-7"]}, {"openai": "online (2 models)"}, "openai")
    options.results = [auto]
    outcome = service.refresh_blocking("openai", explicit=False)
    assert outcome.ok and outcome.provider == "openai" and "Auto-poll complete" in outcome.message
    assert store.get("model_manager_removed_models") == ["gpt-7"]  # passive merges never resurrect
    assert "gpt-7" not in service.snapshot.models and service.snapshot.is_polled("gpt-7")
    options.results = [_result(options, ["gpt-6", "gpt-7"], {"openai": ["gpt-6", "gpt-7"]},
                               {"openai": "online (2 models)"}, "openai")]
    outcome = service.refresh_blocking("openai", explicit=True)
    assert store.get("model_manager_removed_models") == [] and "gpt-7" in service.snapshot.models
    call = options.refresh_calls[-1]
    assert call["only_provider"] == "openai" and call["active_api_key"] == "sk-x"
    disk = _on_disk(store)
    assert set(disk) == {"model", "api_key", "model_manager_removed_models"}  # sparse
    # a full poll reports the desktop status line
    options.results = [_result(options, ["gpt-6", "gpt-7", "claude-x"], {"openai": ["gpt-6"], "anthropic": ["claude-x"]},
                               {"openai": "online (1 models)", "anthropic": "online (1 models)",
                                "mistral": "static fallback (no provider credential)"})]
    outcome = service.refresh_blocking(None)
    assert outcome.ok and outcome.message.startswith("✓ 2 online · anthropic, openai")
    assert "1 need credentials" in outcome.message
    store._saver.close()


def test_refresh_failures_are_reported_not_raised(tmp_path):
    store = _store(tmp_path, {})
    options = FakeOptions()
    options.results = [RuntimeError("boom")]
    service = _service(store, options)
    outcome = service.refresh_blocking("openai")
    assert not outcome.ok and "boom" in outcome.message
    store._saver.close()


def test_model_options_skips_the_excluded_arena_catalog_when_it_is_not_bundled(tmp_path, monkeypatch):
    """The mobile bundle has no autharena_proxy: the real model_options reports Arena instead of raising."""
    import model_options

    monkeypatch.setenv("GLOSSARION_MODEL_CATALOG_CACHE", str(tmp_path / "model_catalog_cache.json"))
    monkeypatch.setattr(model_options, "_MODEL_CATALOG_MEMORY_CACHE", None)
    monkeypatch.setitem(sys.modules, "autharena_proxy", None)  # any import of it raises ImportError
    with pytest.raises(ImportError):
        import autharena_proxy  # noqa: F401
    result = model_options.refresh_provider_model_catalogs(
        active_model="autharena/some-model", only_provider="autharena", timeout=0.1)
    assert result.statuses.get("autharena") == "unavailable in this build"
    assert not result.provider_models.get("autharena")
    assert model_options.due_provider_catalog_for_model("autharena/some-model") is None


def test_model_manager_edits_go_through_the_core_with_tombstones(tmp_path):
    store = _store(tmp_path, {"model": "gpt-6"})
    options = FakeOptions(catalog=["gpt-6", "gpt-6-mini", "claude-opus-5-5"])
    service = _service(store, options)
    service.load_blocking()
    ok, error = service.remove_models(["gpt-6-mini"])
    assert ok and error is None
    assert store.get("custom_model_list") == ["gpt-6", "claude-opus-5-5"]
    assert store.get("model_manager_removed_models") == ["gpt-6-mini"]
    assert "gpt-6-mini" not in service.snapshot.models and service.snapshot.removed == ("gpt-6-mini",)
    # Undo = save the previous order: the model comes back and its tombstone goes
    ok, _ = service.save_order(["gpt-6", "gpt-6-mini", "claude-opus-5-5"], list(service.snapshot.models))
    assert ok and store.get("model_manager_removed_models") == []
    # moving keeps every model; an explicit add re-adds a removed id at the top
    assert service.move_model(2, 0)[0] and service.snapshot.models[0] == "claude-opus-5-5"
    service.remove_models(["gpt-6"])
    assert service.add_model("gpt-6")[0] and service.snapshot.models[0] == "gpt-6"
    assert "gpt-6" not in store.get("model_manager_removed_models")
    assert service.add_model("gpt-6") == (False, "'gpt-6' is already in the list.")
    assert service.save_order([], []) == (False, "The model list cannot be empty. Add at least one model.")
    # restore from the Removed view
    service.remove_models(["claude-opus-5-5"])
    assert service.restore_removed(["claude-opus-5-5"], add_to_saved=True)[0]
    assert "claude-opus-5-5" in store.get("custom_model_list") and store.get("model_manager_removed_models") == []
    assert service.restore_removed(["never-removed"]) == (False, None)
    # Reset: the built-in catalog replaces the list; custom entries become tombstones (desktop)
    service.add_model("my-custom")
    assert service.reset_to_defaults()[0]
    assert store.get("custom_model_list") == ["gpt-6", "gpt-6-mini", "claude-opus-5-5"]
    assert "my-custom" in store.get("model_manager_removed_models")
    store._saver.close()


def test_edits_report_a_missing_core(tmp_path):
    store = _store(tmp_path, {})
    service = mc.ModelCatalogService(store, options=FakeOptions(), core=mc.CatalogCore(None, "missing"))
    service.load_blocking()
    assert service.save_order(["gpt-6"], ["gpt-6", "x"]) == (False, mc.CORE_MISSING)
    assert service.remove_models(["gpt-6"]) == (False, mc.CORE_MISSING)
    assert _on_disk(store) == {} if Path(store.path).exists() else True
    store._saver.close()


def test_manager_poll_models_keeps_custom_entries(tmp_path):
    store = _store(tmp_path, {"model_manager_removed_models": ["old-1"]})
    options = FakeOptions(catalog=["a", "b"])
    service = _service(store, options)
    service.load_blocking()
    service.last_result = _result(options, ["a", "b", "c", "old-1"], {"openai": ["c"]}, {"openai": "online (1 models)"})
    refreshed = service.manager_poll_models(["mine", "a", "b"], {"a", "b"})
    assert refreshed[:3] == ["a", "b", "c"] and refreshed[-1] == "mine" and "old-1" not in refreshed
    store._saver.close()


@needs_run_env
def test_prefix_routes_are_validated_by_the_core_with_desktop_messages(tmp_path):
    store = _store(tmp_path, {"custom_prefix_routes": [{"prefix": "keep/", "routing": "https://x/v1",
                                                         "endpoint_type": "/chat/completions"}]})
    service = _service(store)
    assert mc.validate_prefix_rows([{"prefix": "a b", "routing": "http://x"}]) == (
        None, ("Invalid Prefix", "Prefix on row 1 cannot contain spaces."))
    assert mc.validate_prefix_rows([{"prefix": "a", "routing": ""}])[1][0] == "Incomplete Prefix Route"
    assert mc.validate_prefix_rows([{"prefix": "a", "routing": "ftp://x"}])[1][0] == "Invalid Base URL"
    assert mc.validate_prefix_rows([{"prefix": "a", "routing": "http://x", "endpoint_type": "bad type"}])[1][0] == \
        "Invalid Endpoint Type"
    dup = [{"prefix": "a", "routing": "http://x"}, {"prefix": "A/", "routing": "http://y"}]
    assert mc.validate_prefix_rows(dup) == (None, ("Duplicate Prefix", "'A/' is already listed."))
    ok, error = service.save_prefix_routes([{"prefix": "lan", "routing": "http://192.168.1.2:1234/v1/"},
                                            {"prefix": "", "routing": ""}])
    assert ok and store.get("custom_prefix_routes") == [
        {"prefix": "lan/", "routing": "http://192.168.1.2:1234/v1", "endpoint_type": "/chat/completions"}]
    assert service.save_prefix_routes([{"prefix": "x y", "routing": "http://h"}])[0] is False
    store._saver.close()


def test_auto_poll_is_scoped_ttl_gated_and_skips_excluded_routes(tmp_path):
    store = _store(tmp_path, {"model": "or/openai/gpt-6", "api_key": "sk-or"})
    options = FakeOptions()
    options.provider_model_catalog_supports_anonymous_poll = lambda m, r=None: True
    service = _service(store, options)
    service.load_blocking()
    assert service.due_provider("ocz/qwen") is None and not options.due_calls
    options.due = None
    assert asyncio.run(service.maybe_auto_poll()) is None  # not due (24 h TTL)
    options.due = "openrouter"
    options.results = [_result(options, ["or/x"], {"openrouter": ["or/x"]}, {"openrouter": "online (1 models)"},
                               "openrouter")]
    outcome = asyncio.run(service.maybe_auto_poll())
    assert outcome.provider == "openrouter" and options.refresh_calls[-1]["only_provider"] == "openrouter"
    assert asyncio.run(service.maybe_auto_poll()) is None  # debounced per provider
    assert options.due_calls[-1] == ("or/openai/gpt-6", "sk-or")
    store._saver.close()


def test_route_info_binds_the_shared_route_controls():
    info = mc.route_info("authgpt2/gpt-6-luna", {})
    assert info.source == "settings_rules" and info.logins == ("authgpt",) and info.account == 2
    assert info.account_key == "authgpt2" and info.login_title == "ChatGPT"
    vertex = mc.route_info("vertex/gemini-3-pro", {})
    assert vertex.google_creds and vertex.vertex_location and vertex.google_creds_level == "warning"
    gem = mc.route_info("authgem-vertex/gemini-3-pro", {})
    assert gem.gcp_project and gem.logins == ("authgem",)
    pooled = mc.route_info("gpt-6", {"use_multi_api_keys": True,
                                     "multi_api_keys": [{"api_key": "", "model": "authcd/claude-opus-5-5"}]})
    assert "authcd" in pooled.logins  # an enabled pool route needs the Claude login (desktop on_model_change)
    excluded = mc.route_info("ocz/qwen3", {})
    assert excluded.excluded and "npm/bun" in excluded.excluded_detail
    assert mc.route_info("poe/claude", {}).poe


@needs_client
def test_local_custom_routes_need_no_key():
    routes = [{"prefix": "lan/", "routing": "http://192.168.1.2:11434/v1", "endpoint_type": "/chat/completions"}]
    assert mc.model_needs_api_key("lan/qwen3", routes) is False
    assert mc.model_needs_api_key("gpt-6", routes) is True
    assert mc.model_needs_api_key("ollama/qwen3") is False


# ==========================================================================
# Multi-Key Manager (pure)
# ==========================================================================


def _fake_key_service(**overrides):
    module = types.ModuleType("fake_key_pool_service")
    module.POOL_SPECS = {
        "main": {"title": "Translation Keys", "label": "Translation", "config_key": "multi_api_keys",
                 "toggle_key": "use_multi_api_keys", "description": "Main rotation"},
        "glossary": {"title": "Glossary Keys", "label": "Glossary", "config_key": "glossary_keys",
                     "toggle_key": "use_glossary_keys", "description": "Glossary calls"},
        "tts": {"title": "Audio / TTS Keys", "label": "Audio/TTS", "config_key": "tts_keys", "toggle_key": "use_tts_keys",
                "description": "Speech"},
    }
    module.tested = []

    def new_key_entry(api_key="", model="", **fields):
        entry = {"api_key": api_key, "model": model, "cooldown": 60, "enabled": True, "disabled_contexts": [],
                 "request_parameters": {}}
        entry.update(fields)
        return entry

    def validate_entry(entry, pool=None):
        if not str(entry.get("model") or "").strip():
            return None, "Please enter a model name"
        return dict(entry), None

    def export_pools(config, pools=None):
        out = {}
        for pid, spec in module.POOL_SPECS.items():
            if pools and pid not in pools:
                continue
            out[pid] = {"title": spec["title"], "enabled": bool(config.get(spec["toggle_key"], False)),
                        "keys": list(config.get(spec["config_key"], []) or [])}
        return {"format": "glossarion-key-pools", "version": 1, "pools": out}

    def import_pools(payload, config=None):
        if isinstance(payload, list):
            return {"items": [("main", [k for k in payload if "model" in k], None)], "legacy": True,
                    "skipped": sum(1 for k in payload if "model" not in k)}
        pools = payload.get("pools", {})
        items = [(p, v.get("keys", []), v.get("enabled")) for p, v in pools.items() if p in module.POOL_SPECS]
        return {"items": items, "unknown": [p for p in pools if p not in module.POOL_SPECS]}

    def run_key_test(entry, pool, *, timeout=30):
        module.tested.append((entry.get("model"), pool, os.environ.get("GLOSSARION_TEST_RUN_ENV")))
        if pool == "tts":
            return {"testable": False, "message": "Audio keys are tested with a speech request"}
        if "bad" in str(entry.get("api_key")):
            return {"ok": False, "status": "failed", "message": "401 invalid key"}
        return {"ok": True, "status": "passed", "message": "Test successful"}

    module.new_key_entry = new_key_entry
    module.validate_entry = validate_entry
    module.export_pools = export_pools
    module.import_pools = import_pools
    module.run_key_test = run_key_test
    module.DEFAULT_REFUSAL_PATTERNS = ["i cannot assist", "as an ai"]
    import key_pool_service as real

    module.apply_import_plan = real.apply_import_plan
    module.pool_import_result_message = real.pool_import_result_message
    module.legacy_import_result_message = real.legacy_import_result_message
    for name, value in overrides.items():
        setattr(module, name, value)
    return module


def _keys(tmp_path, data=None, module=None, runner=None):
    from glossarion_mobile.ui.screens.keys import KeyBackend, KeysController

    store = _store(tmp_path, data if data is not None else {})
    backend = KeyBackend(module or _fake_key_service())
    return store, KeysController(store, backend, test_runner=runner, clock=lambda: 1234.5)


SLUG_TO_POOL_IDS = ("main", "fallback", "glossary", "glossary_refinement", "qa_scan", "metadata",
                    "ai_truncation_detection", "rolling_summary", "truncation_retry", "inpainter", "tts")


def test_pool_specs_cover_all_eleven_router_pools():
    from glossarion_mobile.ui.screens.keys import POOL_TO_SLUG, SLUG_TO_POOL, KeyBackend

    assert set(SLUG_TO_POOL) == set(KEY_POOLS) and len(SLUG_TO_POOL) == 11
    specs = KeyBackend(_fake_key_service()).pool_specs()
    assert [s.slug for s in specs] == list(KEY_POOLS)
    main = specs[0]
    assert main.id == "main" and main.description == "Main rotation" and main.config_key == "multi_api_keys"
    assert "translation" in main.contexts and specs[-1].contexts == ("tts",)
    missing = KeyBackend(module=None, module_name="key_pool_service_not_here")
    assert not missing.available and len(missing.pool_specs()) == 11 and missing.error
    for slug in KEY_POOLS:
        match = parse_route(f"/settings/keys/{slug}")
        assert match is not None and SLUG_TO_POOL[match.params["pool"]] in POOL_TO_SLUG


def test_controller_edits_are_sparse_and_undoable(tmp_path):
    store, keys = _keys(tmp_path, {"model": "gpt-6", "glossary_keys": [{"api_key": "g1", "model": "gemini-3"}]})
    index, error = keys.add_key("translation", {"api_key": "sk-1", "model": "gpt-6"})
    assert index == 0 and error is None
    assert keys.add_key("main", {"api_key": "sk-2", "model": ""}) == (None, "Please enter a model name")
    keys.add_key("main", {"api_key": "sk-2", "model": "gpt-6-mini"})
    keys.add_key("main", {"api_key": "sk-3", "model": "claude-opus-5-5"})
    assert [k["api_key"] for k in keys.keys("main")] == ["sk-1", "sk-2", "sk-3"]
    assert keys.update_key("main", 1, {"api_key": "sk-2b", "model": "gpt-6-mini", "cooldown": 90}) is None
    assert keys.update_key("main", 9, {"api_key": "x", "model": "y"}) == "This key no longer exists"
    assert keys.move("main", 2, 0) and [k["api_key"] for k in keys.keys("main")] == ["sk-3", "sk-1", "sk-2b"]
    removed = keys.remove("main", [0, 2])
    assert [k["api_key"] for k in keys.keys("main")] == ["sk-1"]
    keys.restore("main", removed)
    assert [k["api_key"] for k in keys.keys("main")] == ["sk-3", "sk-1", "sk-2b"]
    assert keys.set_keys_enabled("main", [1, 2], False) == 2 and not keys.keys("main")[1]["enabled"]
    keys.set_enabled("main", True)
    cleared = keys.clear("main")
    assert keys.count("main") == 0 and len(cleared) == 3
    keys.restore("main", cleared)
    disk = _on_disk(store)
    assert set(disk) == {"model", "glossary_keys", "multi_api_keys", "use_multi_api_keys"}
    assert disk["glossary_keys"] == [{"api_key": "g1", "model": "gemini-3"}]  # untouched pool kept byte-for-byte
    assert keys.setting("force_key_rotation", True) is True
    keys.set_setting("rotation_frequency", 3)
    assert _on_disk(store)["rotation_frequency"] == 3
    store._saver.close()


def test_contexts_are_tri_state_and_keys_copy_or_move_between_pools(tmp_path):
    store, keys = _keys(tmp_path, {"multi_api_keys": [
        {"api_key": "a", "model": "m", "disabled_contexts": ["glossary"]},
        {"api_key": "b", "model": "m", "disabled_contexts": []},
    ]})
    states = keys.context_states("main", [0, 1])
    assert states["glossary"] is None and states["translation"] is True
    assert keys.set_disabled_contexts("main", [0, 1], {"translation": False}) == 2
    after = keys.keys("main")
    assert after[0]["disabled_contexts"] == ["glossary", "translation"] and after[1]["disabled_contexts"] == ["translation"]
    keys.set_disabled_contexts("main", [0], {"glossary": True})
    assert keys.keys("main")[0]["disabled_contexts"] == ["translation"]
    assert keys.copy_to("main", [0], "glossary") == 1 and keys.count("main") == 2 and keys.count("glossary") == 1
    assert keys.copy_to("main", [1], "tts", move=True) == 1 and keys.count("main") == 1 and keys.count("tts") == 1
    assert keys.copy_to("main", [0], "translation") == 0
    store._saver.close()


def test_key_tests_run_in_the_run_env_and_persist_results(tmp_path):
    entered = []

    def runner(config, fn):
        entered.append((config.get("model"), config.get("use_multi_api_keys"), config.get("use_tts_keys")))
        os.environ["GLOSSARION_TEST_RUN_ENV"] = "1"
        try:
            return fn()
        finally:
            os.environ.pop("GLOSSARION_TEST_RUN_ENV", None)

    module = _fake_key_service()
    store, keys = _keys(tmp_path, {"model": "gpt-6", "use_multi_api_keys": True, "use_tts_keys": True, "multi_api_keys": [
        {"api_key": "good", "model": "gpt-6"}, {"api_key": "bad", "model": "gpt-6"},
        {"api_key": "ENC:xyz", "model": "gpt-6"}], "tts_keys": [{"api_key": "t", "model": "tts-1"}]},
        module=module, runner=runner)
    progress = []
    results = keys.test_keys("main", [0, 1, 2], on_result=lambda i, r: progress.append((i, r["status"])))
    assert progress == [(0, "passed"), (1, "failed"), (2, "error")]
    # one run environment (one engine-lock hold) per tested key, built with every key pool off
    assert entered == [("gpt-6", False, False)] * 2 and store.get("use_multi_api_keys") is True
    assert all(env == "1" for _m, _p, env in module.tested)  # inside the run env
    stored = keys.keys("main")
    assert stored[0]["last_test_result"] == "passed" and stored[0]["last_test_time"] == 1234.5
    assert stored[1]["last_test_result"] == "failed" and stored[1]["last_test_message"] == "401 invalid key"
    assert stored[2]["last_test_result"] == "error"
    from glossarion_mobile.ui.screens.keys import key_status

    assert [key_status(k) for k in stored] == ["passed", "failed", "encrypted"]
    tts = keys.test_keys("tts", [0])
    assert tts[0][1]["status"] == "untestable" and "last_test_result" not in keys.keys("tts")[0]
    assert keys.test_entry({"api_key": "good", "model": "gpt-6"}, "main")["status"] == "passed"
    assert len(results) == 3
    store._saver.close()


def test_key_test_results_follow_the_key_tested(tmp_path):
    """A result is stored on the key that was tested (api_key + model), even when the list was
    reordered or shortened while "Test all" ran (the desktop updates the key object)."""
    store, keys = _keys(tmp_path, {"multi_api_keys": [
        {"api_key": "a", "model": "m"}, {"api_key": "b", "model": "m"}, {"api_key": "c", "model": "m"}]})
    passed = {"ok": True, "status": "passed", "message": "ok"}
    keys.set_keys("main", [{"api_key": "b", "model": "m"}, {"api_key": "a", "model": "m"}])  # moved meanwhile
    assert keys.record_test("main", 1, passed, model="m", api_key="b")
    after = keys.keys("main")
    assert after[0]["last_test_result"] == "passed" and "last_test_result" not in after[1]
    assert not keys.record_test("main", 2, passed, model="m", api_key="c")  # removed meanwhile: dropped
    assert keys.record_test("main", 1, {"ok": False, "status": "failed", "message": "401"}, model="m", api_key="a")
    assert keys.keys("main")[1]["last_test_result"] == "failed" and keys.keys("main")[0]["last_test_result"] == "passed"
    assert not keys.record_test("main", 0, {"status": "busy", "message": "job"}, model="m", api_key="b")
    store._saver.close()


def test_key_tests_send_the_tested_key_with_the_translation_pool_on(tmp_path, monkeypatch):
    """With the Translation (and Fallback) pool on, the run environment loads the pool into
    UnifiedClient and exports USE_MULTI_API_KEYS=1: a probe client built there would send the
    pool's keys. Key tests build their run environment with every pool off."""
    pytest.importorskip("job_runner")
    pytest.importorskip("headless_owner")
    kps = pytest.importorskip("key_pool_service")
    unified = pytest.importorskip("unified_api_client")
    from glossarion_mobile.ui.screens import env_preview as ep
    from glossarion_mobile.ui.screens.endpoints import run_in_run_env
    from glossarion_mobile.ui.screens.keys import KEY_TEST_BUSY, KeyBackend, KeysController

    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "absent-config.json"))  # no config.json pool fallback
    monkeypatch.delenv("USE_MULTI_API_KEYS", raising=False)
    monkeypatch.delenv("USE_FALLBACK_KEYS", raising=False)
    seen = []

    def probe(entry, pool, timeout=None):
        request = kps.build_test_request(entry, pool)
        client = unified.UnifiedClient(**request["client_kwargs"])
        seen.append((client._multi_key_mode, client.translator_config.get("use_fallback_keys"), client.api_key))
        return {"ok": True, "status": "passed", "message": "ok", "last_test_result": "passed"}

    module = types.SimpleNamespace(run_key_test=probe, POOL_SPECS=kps.POOL_SPECS)
    pool = [{"api_key": "sk-POOL-A", "model": "gpt-4o-mini", "enabled": True},
            {"api_key": "sk-POOL-B", "model": "gpt-4o-mini", "enabled": True}]
    store = _store(tmp_path, {"model": "gpt-4o-mini", "api_key": "sk-main", "use_multi_api_keys": True,
                              "multi_api_keys": pool, "use_fallback_keys": True,
                              "fallback_keys": [{"api_key": "sk-FALLBACK", "model": "gpt-4o-mini"}]})
    keys = KeysController(store, KeyBackend(module), test_runner=run_in_run_env, clock=lambda: 1.0)
    entry = {"api_key": "sk-TESTED-KEY", "model": "gpt-4o-mini"}
    # a run's environment (pools on): the client would rotate through the pool - what key tests avoid
    run_in_run_env(store.snapshot(), lambda: probe(entry, "fallback"))
    assert seen[-1][:2] == (True, True), seen[-1]
    # key tests: the tested key alone, from every pool's editor / the ModelSheet KeyField
    for pool_id in ("main", "fallback", "glossary", "metadata"):
        assert keys.test_entry(entry, pool_id)["status"] == "passed"
        assert seen[-1] == (False, False, "sk-TESTED-KEY"), (pool_id, seen[-1])
    results = keys.test_keys("main", [0, 1])
    assert [r["status"] for _i, r in results] == ["passed", "passed"]
    assert seen[-2:] == [(False, False, "sk-POOL-A"), (False, False, "sk-POOL-B")]
    assert [k["last_test_result"] for k in keys.keys("main")] == ["passed", "passed"]
    assert os.environ.get("USE_MULTI_API_KEYS") is None  # the run-env scope was restored
    # a running job holds the engine lock with its pools on: the test is refused, not mis-sent
    assert ep.ENV_PREVIEW_LOCK.acquire(timeout=1)
    try:
        monkeypatch.setenv("USE_MULTI_API_KEYS", "1")
        busy_keys = KeysController(store, KeyBackend(module),
                                   test_runner=lambda c, f: run_in_run_env(c, f, lock_timeout=0.05))
        holder: dict = {}
        thread = threading.Thread(target=lambda: holder.update(r=busy_keys.test_entry(entry, "main")))
        thread.start()
        thread.join(10)
        assert holder["r"] == {"ok": None, "status": "busy", "message": KEY_TEST_BUSY}
    finally:
        ep.ENV_PREVIEW_LOCK.release()
    store._saver.close()


def test_import_export_glossarion_key_pools_v1(tmp_path):
    store, keys = _keys(tmp_path, {"use_multi_api_keys": True,
                                   "multi_api_keys": [{"api_key": "a", "model": "m"}],
                                   "glossary_keys": [{"api_key": "g", "model": "m2"}]})
    payload = keys.export_payload()
    assert payload["format"] == "glossarion-key-pools" and payload["version"] == 1
    assert payload["pools"]["glossary"]["keys"] == [{"api_key": "g", "model": "m2"}]  # glossary pool included
    assert keys.payload_key_count(payload) == 2
    only = keys.export_payload(["glossary"])
    assert list(only["pools"]) == ["glossary"]
    replaced = {"format": "glossarion-key-pools", "version": 1, "pools": {
        "main": {"enabled": False, "keys": [{"api_key": "n1", "model": "x"}, {"api_key": "n2", "model": "y"}]},
        "nonsense": {"keys": []}}}
    plan = keys.import_plan(replaced)
    assert plan.error is None and plan.total == 2 and plan.unknown == ("nonsense",)
    assert plan.summary_lines(keys.titles()) == ["Translation Keys: 2 key(s)"]
    assert keys.apply_import(plan) == 2
    assert [k["api_key"] for k in keys.keys("main")] == ["n1", "n2"] and store.get("use_multi_api_keys") is False
    legacy = keys.import_plan([{"api_key": "l1", "model": "z"}, {"api_key": "no-model"}], target="glossary")
    assert legacy.legacy and legacy.skipped == 1 and legacy.items[0][0] == "glossary"
    keys.apply_import(legacy)
    assert [k["api_key"] for k in keys.keys("glossary")] == ["g", "l1"]  # legacy lists append
    assert keys.import_plan(replaced, target="tts").error == "The file has no Audio / TTS Keys"
    store._saver.close()


def test_request_parameters_follow_the_shared_rules():
    from glossarion_mobile.ui.screens.keys import parse_request_params, request_param_display

    assert parse_request_params([("top_p", "0.9"), ("reasoning", '{"effort": "low"}'), ("tag", "hello"), ("", "")]) == {
        "top_p": 0.9, "reasoning": {"effort": "low"}, "tag": "hello"}
    with pytest.raises(ValueError, match="controlled by Glossarion"):
        parse_request_params([("model", "x")])
    with pytest.raises(ValueError, match="listed twice"):
        parse_request_params([("a", "1"), ("a", "2")])
    with pytest.raises(ValueError, match="needs a name"):
        parse_request_params([("", "value")])
    assert request_param_display("true") == '"true"' and request_param_display(0.5) == "0.5"


def test_normalize_test_result_shapes():
    from glossarion_mobile.ui.screens.key_editor import normalize_test_result

    assert normalize_test_result({"ok": True})["status"] == "passed"
    assert normalize_test_result((False, "nope")) == {"ok": False, "status": "failed", "message": "nope"}
    assert normalize_test_result({"status": "rate_limited", "message": "429"})["ok"] is False
    assert normalize_test_result({"testable": False})["status"] == "untestable"
    assert normalize_test_result(None)["status"] == "untestable"
    assert normalize_test_result(types.SimpleNamespace(success=True, message="hi"))["status"] == "passed"


# ==========================================================================
# Refusal patterns / Local AI / Endpoints (pure)
# ==========================================================================


def test_refusal_model_follows_the_desktop_dialog(tmp_path):
    import key_pool_service

    from glossarion_mobile.ui.screens.refusal_patterns import RefusalModel, merge_pattern_lines

    store = _store(tmp_path, {})
    model = RefusalModel(store)
    defaults = list(key_pool_service.DEFAULT_REFUSAL_PATTERNS)
    assert model.patterns() == defaults and not store.has("refusal_patterns")
    assert model.disabled() is True and model.length_limit() == 1000
    assert model.add("  I Will Not Translate  ") == "i will not translate"
    assert model.patterns()[0] == "i will not translate"
    assert model.add("AS AN AI") is None and model.add("   ") is None
    assert model.edit("as an ai", "As A Robot") and "as a robot" in model.patterns()
    assert not model.edit("i cannot assist", "i will not translate")
    before = model.delete(["i cannot assist"])
    assert "i cannot assist" not in model.patterns() and "i cannot assist" in before
    assert model.set_length_limit("abc") == 1000 and model.set_length_limit("0") == 1000
    assert model.set_length_limit("250") == 250 and model.length_limit() == 250
    model.reset()
    assert model.patterns() == defaults
    merged, added, skipped = merge_pattern_lines(["a"], ["A\n", "# comment\n", "b\n", "\n", "b\n"])
    assert merged == ["a", "b"] and (added, skipped) == (1, 2)
    assert model.merge_lines(["New One", "as an ai"]) == (1, 1) and model.patterns()[-1] == "new one"
    disk = _on_disk(store)
    assert set(disk) == {"refusal_patterns", "refusal_pattern_length_limit"}
    store._saver.close()


def test_refusal_defaults_single_source():
    import key_pool_service

    from glossarion_mobile.ui.screens.keys import KeyBackend

    assert KeyBackend(key_pool_service).refusal_defaults() == list(key_pool_service.DEFAULT_REFUSAL_PATTERNS)
    assert KeyBackend(_fake_key_service()).refusal_defaults() == ["i cannot assist", "as an ai"]
    with pytest.raises(RuntimeError):
        KeyBackend(module=None, module_name="key_pool_service_not_here").refusal_defaults()


def test_real_key_pool_service_binding(tmp_path):
    """The adapter on the real shared service: eleven specs, v1 export (Glossary included),
    pool-aware import replace + apply, untestable pools, validation messages."""
    import key_pool_service

    from glossarion_mobile.ui.screens.keys import KeyBackend

    backend = KeyBackend()
    assert backend.module is key_pool_service
    specs = backend.pool_specs()
    assert [s.id for s in specs] == list(SLUG_TO_POOL_IDS) and specs[0].title == "Translation Keys"
    assert all(s.description for s in specs)
    store, keys = _keys(tmp_path, {"use_glossary_keys": True, "glossary_keys": [{"api_key": "g", "model": "m"}],
                                   "multi_api_keys": [{"api_key": "a", "model": "m"}]}, module=key_pool_service)
    payload = keys.export_payload()
    assert payload["format"] == "glossarion-key-pools" and payload["version"] == 1
    assert len(payload["pools"]) == 11 and payload["pools"]["glossary"]["enabled"] is True
    plan = keys.import_plan({"format": "glossarion-key-pools", "version": 1,
                             "pools": {"glossary": {"enabled": False, "keys": [{"api_key": "n", "model": "x"}, {"bad": 1}]},
                                       "mystery": []}})
    assert plan.error is None and plan.skipped == 1 and plan.unknown == ("mystery",) and plan.total == 1
    assert keys.apply_import(plan) == 1
    assert keys.keys("glossary") == [{"api_key": "n", "model": "x"}] and store.get("use_glossary_keys") is False
    assert keys.import_result_message(plan, 1).startswith("Imported 1 key(s) across 1 pool(s)")
    assert keys.import_plan({"nonsense": True}).error == key_pool_service.INVALID_IMPORT_MESSAGE
    for pool in ("tts", "inpainter"):
        result = backend.run_test({"api_key": "k", "model": "m"}, pool)
        assert result["status"] == "untestable" and result["message"] == key_pool_service.UNTESTABLE_POOLS[pool]
    assert backend.validate({"api_key": " k ", "model": " "}, "glossary") == (None, "Please enter a model name")
    entry = backend.new_entry("k", "m")
    assert entry["api_key"] == "k" and entry["enabled"] is True and entry["api_call_delay"] == 0.0
    disk = _on_disk(store)
    assert set(disk) == {"use_glossary_keys", "glossary_keys", "multi_api_keys"}  # sparse
    store._saver.close()


@pytest.mark.skipif(not _importable("multi_api_key_manager"), reason="multi_api_key_manager needs backend packages")
def test_real_service_validates_and_imports_legacy_lists(tmp_path):
    import key_pool_service

    store, keys = _keys(tmp_path, {"multi_api_keys": [{"api_key": "a", "model": "m"}]}, module=key_pool_service)
    clean, error = keys.backend.validate({"api_key": " k ", "model": "gpt-6", "individual_key_temperature": "-1",
                                          "request_parameters": {"model": "x", "top_p": 0.5}}, "main")
    assert error is None and clean["api_key"] == "k" and clean["individual_key_temperature"] is None
    assert clean["request_parameters"] == {"top_p": 0.5} and clean["cooldown"] == 60
    plan = keys.import_plan([{"api_key": "l", "model": "z"}, {"api_key": "no-model"}])
    assert plan.legacy and plan.skipped == 1
    keys.apply_import(plan)
    assert [k["api_key"] for k in keys.keys("main")] == ["a", "l"]
    assert keys.import_result_message(plan, 1) == "Imported 1 API key(s) into the Translation pool (1 skipped)"
    store._saver.close()


def test_lan_routes_validate_addresses_and_keep_other_routes():
    from glossarion_mobile.ui.screens.local_ai import (base_url_of, remove_lan_route, routing_for, set_lan_route,
                                                       validate_base_url)

    assert validate_base_url("192.168.1.10", 11434) == ("http://192.168.1.10:11434", None)
    assert validate_base_url("http://pc.local:8080/v1", 1234) == ("http://pc.local:8080", None)
    assert validate_base_url("https://box", 1234) == ("https://box:1234", None)
    assert validate_base_url("", 1)[1] and validate_base_url("ftp://x", 1)[1] and validate_base_url("bad host!", 1)[1]
    assert routing_for("http://h:1") == "http://h:1/v1" and base_url_of("http://h:1/v1/") == "http://h:1"
    routes = [{"prefix": "keep/", "routing": "https://x/v1", "endpoint_type": "/v1/messages", "note": 1}]
    added = set_lan_route(routes, "ollama-lan/", "http://h:11434")
    assert added[0] == routes[0] and added[1] == {"prefix": "ollama-lan/", "routing": "http://h:11434/v1",
                                                  "endpoint_type": "/chat/completions"}
    changed = set_lan_route(added, "OLLAMA-LAN/", "http://h2:11434")
    assert len(changed) == 2 and changed[1]["routing"] == "http://h2:11434/v1"
    assert remove_lan_route(changed, "ollama-lan/") == routes


def test_ollama_options_use_the_desktop_checks():
    pytest.importorskip("ollama_settings")
    from glossarion_mobile.ui.screens.local_ai import collect_model_options

    result = collect_model_options({"num_ctx": "8192", "temperature": "0.7", "stop": '["</s>"]'},
                                   extra_options='{"custom_opt": 1}', extra_request='{"priority": 2}',
                                   think="false", keep_alive="5m", response_format="json", base={"keep": True})
    assert result == {"keep": True, "options": {"custom_opt": 1, "num_ctx": 8192, "temperature": 0.7, "stop": ["</s>"]},
                      "request": {"priority": 2}, "think": False, "keep_alive": "5m", "format": "json"}
    with pytest.raises(ValueError, match="duplicate a named field"):
        collect_model_options({}, extra_options='{"num_ctx": 1}')
    with pytest.raises(ValueError, match="cannot override"):
        collect_model_options({}, extra_request='{"model": "x"}')
    with pytest.raises(ValueError, match="num_ctx must be greater than zero"):
        collect_model_options({"num_ctx": "0"})
    with pytest.raises(ValueError, match="Response format"):
        collect_model_options({}, response_format="yaml")
    assert "think" not in collect_model_options({}, think="model default", base={"think": True})


def test_run_in_run_env_scopes_the_environment(monkeypatch):
    pytest.importorskip("job_runner")
    from glossarion_mobile.ui.screens import env_preview as ep
    from glossarion_mobile.ui.screens.endpoints import run_in_run_env

    monkeypatch.delenv("OPENAI_CUSTOM_BASE_URL", raising=False)
    seen = {}

    def owner_factory(config, *, host, api_key):
        seen["key"] = api_key
        return types.SimpleNamespace(config=config)

    def env_builder(owner, input_path, api_key):
        return {"OPENAI_CUSTOM_BASE_URL": owner.config["openai_base_url"], "MODEL": owner.config["model"]}

    value = run_in_run_env({"model": "gpt-6", "api_key": "sk", "openai_base_url": "http://lan/v1"},
                           lambda: (os.environ.get("OPENAI_CUSTOM_BASE_URL"), os.environ.get("MODEL")),
                           owner_factory=owner_factory, env_builder=env_builder)
    assert value == ("http://lan/v1", "gpt-6") and seen["key"] == "sk"
    assert "OPENAI_CUSTOM_BASE_URL" not in os.environ  # restored
    # a running job holds the engine lock: the function runs in the job's environment as is
    assert ep.ENV_PREVIEW_LOCK.acquire(timeout=1)
    try:
        holder: dict = {}
        thread = threading.Thread(target=lambda: holder.update(v=run_in_run_env({}, lambda: "direct", lock_timeout=0.05)))
        thread.start()
        thread.join(5)
        assert holder["v"] == "direct"
    finally:
        ep.ENV_PREVIEW_LOCK.release()


def test_endpoint_summary_and_test_connection():
    from glossarion_mobile.ui.screens.endpoints import endpoint_summary, test_connection
    from glossarion_mobile.ui.screens.keys import KeyBackend

    config = {"use_custom_openai_endpoint": True, "openai_base_url": "https://r.openai.azure.com",
              "groq_base_url": "http://192.168.1.2:8080/v1", "use_gemini_openai_endpoint": True,
              "gemini_openai_endpoint": "generativelanguage.googleapis.com", "model": "gpt-6", "api_key": "good"}
    names = [name for name, _url in endpoint_summary(config)]
    assert names == ["Azure OpenAI", "Groq/Local", "Gemini (gRPC)"]
    module = _fake_key_service()
    used = []
    result = test_connection(config, backend=KeyBackend(module), runner=lambda cfg, fn: (used.append(cfg["model"]), fn())[1])
    assert result["status"] == "passed" and used == ["gpt-6"] and module.tested[-1][:2] == ("gpt-6", "main")
    assert test_connection({}, backend=KeyBackend(module), runner=lambda c, f: f())["status"] == "untestable"


# ==========================================================================
# Screens (fake Flet session)
# ==========================================================================


class FakeCatalog:
    """The ModelCatalogService surface the sheet uses, with recorded refreshes."""

    def __init__(self, service):
        self.service = service
        self.refreshed: list = []

    def __getattr__(self, name):
        return getattr(self.service, name)

    async def refresh(self, provider=None, *, explicit=True):
        self.refreshed.append(provider)
        return mc.RefreshOutcome(provider, True, message=f"{provider} — 3 models · 1 new model found")


def _settings_ctx(store, page, notes=None):
    from glossarion_mobile.ui.settings.context import SettingsContext
    from glossarion_mobile.ui.settings.schema_access import SchemaAccess

    return SettingsContext(page=page, store=store, schema=SchemaAccess(),
                           notify=(lambda msg, *a: notes.append(msg)) if notes is not None else None)


@needs_flet
def test_model_sheet_rows_sections_selection_and_route_row(tmp_path):
    import flet as ft

    from glossarion_mobile.ui.components.reason_chip import ReasonChip
    from glossarion_mobile.ui.sheets.model_sheet import ModelSheet, SheetEnv
    from glossarion_mobile.ui.sheets.model_sheet_min import ModelSheetMin
    from glossarion_mobile.state.prefs import Prefs

    async def scenario():
        conn, session = _fake_session("android")
        page = session.page
        store = _store(tmp_path, {"model": "authgpt/gpt-6-luna"})
        service = _service(store, FakeOptions(polled={"openai": ["gpt-6-mini"]}))
        service.load_blocking()
        catalog = FakeCatalog(service)
        prefs = Prefs(tmp_path / "mobile_state.json")
        prefs.load()
        prefs.set("model_favorites", ["claude-opus-5-5"])
        navigated, signed, selected, notes = [], set(), [], []
        env = SheetEnv(catalog=catalog, store=store, prefs=prefs, ctx=_settings_ctx(store, page),
                       navigate=lambda name, params=None: navigated.append(name), notify=notes.append,
                       signed_in_keys=lambda: frozenset(signed), sign_in=lambda r, a: navigated.append(f"sign:{r}{a}"),
                       spawn=lambda coro: asyncio.ensure_future(coro))
        assert ModelSheetMin is ModelSheet
        sheet = ModelSheetMin(current_model="authgpt/gpt-6-luna", current_profile="Universal",
                              current_language="English", profiles=["Universal", "Korean"], languages=["English", "Korean"],
                              signed_in=lambda m: False, tab="model", on_select=lambda f, v, chat: selected.append((f, v, chat)),
                              env=env)
        sheet.show(page)
        page.update()
        rows = sheet.rows
        # excluded route: grey, ReasonChip, not selectable
        assert isinstance(rows["ocz/qwen3-8b"].trailing, ReasonChip) and rows["ocz/qwen3-8b"].on_click is None
        # ChatGPT route not signed in: amber + "Sign in with ChatGPT"
        assert rows["authgpt/gpt-6-luna"].trailing.content == "Sign in with ChatGPT"
        signed.add("authgpt")
        sheet.refresh()
        assert sheet.rows["authgpt/gpt-6-luna"].trailing is None
        # polled marker and favourites section
        assert "✓ polled" in sheet.rows["gpt-6-mini"].subtitle.value
        keys = [getattr(c, "key", None) for c in sheet.list_view.controls]
        assert "group-★ Favorites" in keys and "group-openai" in keys
        # provider chip filter and search (flat ranked list)
        sheet.set_chip("anthropic")
        assert list(sheet.rows) == ["claude-opus-5-5"]
        sheet.set_chip("all")
        sheet.set_query("gpt-6")
        assert list(sheet.rows)[:2] == ["gpt-6", "gpt-6-mini"]
        assert all(isinstance(c, ft.ListTile) for c in sheet.list_view.controls)
        # per-group refresh and the ⋯ refresh of the selected provider
        await sheet.refresh_provider("openai")
        await sheet.refresh_selected_provider()
        assert catalog.refreshed == ["openai", "authgpt"] and notes[-1].startswith("authgpt")
        # route row: sign-in chip for the default model; thinking tiles of the GPT family
        sheet.set_query("")
        assert any(getattr(c, "key", "") == "route-login-authgpt" for row in sheet.route_row.controls
                   for c in getattr(row, "controls", []))
        assert sheet.thinking.visible and set(sheet.thinking_tiles) == set(mc.THINKING_FIELDS["gpt"])
        sheet.thinking_tiles["gpt_effort"].apply("high")
        assert store.get("gpt_effort") == "high"
        # hide unpolled writes the shared key and filters the list
        sheet._toggle_hide_unpolled()
        assert store.get("model_manager_hide_unpolled_models") is True
        assert set(sheet.visible_models()) == {"gpt-6-mini"}
        sheet._toggle_hide_unpolled()
        # long-press actions: favourite toggle and copy
        action_sheet = sheet.open_row_actions("gpt-6")
        assert [i.label for i in action_sheet.items][:2] == ["★ Favorite", "Copy id"]
        assert sheet.toggle_favorite("gpt-6") and "gpt-6" in prefs.get("model_favorites")
        # select: U3 contract (field, value, this chat only) + recents
        sheet.chat_switch.value = True
        sheet.select("model", "gpt-6")
        assert selected == [("model", "gpt-6", True)] and prefs.get("model_recents")[0] == "gpt-6"
        assert sheet.closed
        # selecting an excluded route is refused
        other = ModelSheet(current_model="gpt-6", env=env, on_select=lambda *a: selected.append(a),
                           profiles=["Universal", "Korean"], languages=["English", "Korean"])
        other.select("model", "ocz/qwen3-8b")
        assert len(selected) == 1
        # profile / language tabs
        other.tab = "profile"
        other.refresh()
        assert "Korean" in other.rows and getattr(other.list_view.controls[-1], "key", "") == "profile-extras"
        other.role_toggle.selected = ["user"]
        other._on_role(types.SimpleNamespace(control=other.role_toggle))
        assert store.get("system_prompt_to_user") is True
        other.tab = "language"
        other.query = "Thai"
        other.refresh()
        assert getattr(other.list_view.controls[0], "key", "") == "language-free-text"
        page.update()
        assert conn.bytes_sent > 0
        prefs.close()
        store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_model_sheet_without_services_keeps_the_u3_behaviour(tmp_path, monkeypatch):
    from glossarion_mobile.ui.sheets import model_sheet as ms

    async def scenario():
        conn, session = _fake_session("android")
        monkeypatch.setattr(ms, "_SHEET_ENV", None)
        monkeypatch.setattr(mc, "_DEFAULT", None)
        sheet = ms.ModelSheetMin(current_model="gpt-6", current_profile="Universal", current_language="English",
                                 signed_in=lambda m: False, on_select=lambda *a: None)
        sheet.show(session.page)
        sheet.set_models(["gpt-6", "ollamapull/x", "authgpt/gpt-6-luna"])
        assert set(sheet.rows) == {"gpt-6", "ollamapull/x", "authgpt/gpt-6-luna"}
        assert sheet.rows["authgpt/gpt-6-luna"].trailing.content == "Sign in with ChatGPT"
        assert sheet._needs_key("ollama/qwen") is False and sheet.row_state("gpt-6") == ("ready", None)
        poe = sheet.open_poe_setup()
        assert poe.api_key_for("abc%3D%3D") == "p-b:abc%3D%3D" and poe.api_key_for("p-b:x") == "p-b:x"
        session.page.update()

    asyncio.run(scenario())


@needs_flet
def test_model_sheet_pool_routes_use_any_slot_like_the_send_gate(tmp_path):
    """authgem-vertex0/ (like authgpt0/ and authgrok0/) rotates through every saved slot: any signed-in
    Gemini slot makes it ready. The sheet and the Send gate answer from the same rule."""
    from glossarion_mobile.services.oauth import sign_in_satisfied
    from glossarion_mobile.ui.sheets.model_sheet import ModelSheet, SheetEnv

    store = _store(tmp_path, {"model": "authgpt/gpt-6-luna"})
    signed = {"authgem2"}
    sheet = ModelSheet(current_model="authgpt/gpt-6-luna", env=SheetEnv(store=store, signed_in_keys=lambda: frozenset(signed)))
    cases = {"authgem-vertex0/gemini-3.5-pro": True, "authgem-vertex/gemini-3.5-pro": False,
             "authgem2/gemini-3.5-pro": True, "authgem-vertex2/gemini-3.5-pro": True,
             "authgem/gemini-3.5-flash": False, "authgpt0/gpt-6-luna": False, "authgrok0/grok-4": False}
    for model, expected in cases.items():
        assert sheet.signed_in(model) is expected, model
        assert sign_in_satisfied(model, signed) is expected, model
    assert sheet.row_state("authgem-vertex0/gemini-3.5-pro") == ("ready", None)
    assert sheet.row_state("authgem-vertex/gemini-3.5-pro") == ("sign_in", "Sign in")
    signed.add("authgpt3")
    assert sheet.signed_in("authgpt0/gpt-6-luna") and not sheet.signed_in("authgpt/gpt-6-luna")
    # a route-row chip for another slot asks for exactly that slot
    assert sheet.signed_in("authgem-vertex0/gemini-3.5-pro", "authgem", 3) is False
    assert sheet.signed_in("authgem-vertex0/gemini-3.5-pro", "authgem", 2) is True
    store._saver.close()


@needs_flet
def test_poe_setup_stores_the_cookie_as_the_api_key(tmp_path):
    from glossarion_mobile.ui.sheets.model_sheet import PoeSetupSheet, SheetEnv

    async def scenario():
        conn, session = _fake_session("android")
        store = _store(tmp_path, {"api_key": "p-b:old%3D%3D"})
        tests = []

        async def test_key(value, model):
            tests.append((value, model))
            return {"ok": True}

        sheet = PoeSetupSheet(env=SheetEnv(store=store, test_key=test_key), model="poe/claude")
        sheet.show(session.page)
        assert sheet.field.value == "old%3D%3D"
        sheet.field.set_value("new%3D%3D")
        assert store.get("api_key") == "p-b:new%3D%3D"
        result = await sheet.field.test()
        assert result["status"] == "passed" and tests == [("p-b:new%3D%3D", "poe/claude")]
        session.page.update()
        store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_model_manager_screen_remove_undo_restore_add_and_prefixes(tmp_path):
    from glossarion_mobile.ui.screens.model_manager import ModelManagerScreen, PrefixEditor

    async def scenario():
        conn, session = _fake_session("android")
        page = session.page
        store = _store(tmp_path, {"model": "gpt-6"})
        options = FakeOptions(catalog=["gpt-6", "gpt-6-mini", "claude-opus-5-5"], polled={"openai": ["gpt-6"]})
        service = _service(store, options)
        service.load_blocking()
        notes = []
        screen = ModelManagerScreen(parse_route("/settings/models"), catalog=service, page=page,
                                    notify=lambda m, a=None, cb=None: notes.append((m, a, cb)),
                                    spawn=lambda c: asyncio.ensure_future(c))
        page.views[0].controls.append(screen.get_body())
        screen.actions()
        screen.did_show()
        page.update()
        await asyncio.sleep(0)
        assert screen.reorderable and len(screen.list_view.controls) == 3
        # swipe / delete removes with Undo; the tombstone is written
        assert screen.remove("gpt-6-mini")
        assert store.get("model_manager_removed_models") == ["gpt-6-mini"]
        message, action, undo = notes[-1]
        assert action == "Undo"
        undo()
        assert "gpt-6-mini" in service.snapshot.models and store.get("model_manager_removed_models") == []
        # Removed filter: swipe restores
        screen.remove("claude-opus-5-5")
        screen.set_filter("removed")
        assert screen.visible_models() == ["claude-opus-5-5"]
        assert screen.restore("claude-opus-5-5") and "claude-opus-5-5" in service.snapshot.models
        screen.set_filter("removed")
        # Polled only: the shared hide flag; reordering is off while filtered
        screen.toggle_polled_only()
        assert store.get("model_manager_hide_unpolled_models") is True and not screen.reorderable
        assert screen.visible_models() == ["gpt-6"]
        screen.toggle_polled_only()
        # reorder event (Flet already adjusted new_index)
        screen._on_reorder(types.SimpleNamespace(old_index=2, new_index=0))
        assert service.snapshot.models[0] == "claude-opus-5-5"
        # Add model dialog
        screen.open_add()
        screen.add_field.value = "my-model"
        screen.add_submit()
        assert service.snapshot.models[0] == "my-model"
        # Poll providers: the desktop manager merge (custom entries kept) is saved
        options.results = [options.ModelCatalogRefreshResult(["gpt-6", "gpt-6-mini", "claude-opus-5-5", "gpt-7"],
                                                             {"openai": ["gpt-6", "gpt-7"]},
                                                             {"openai": "online (2 models)"}, None)]
        outcome = await screen.poll()
        assert outcome.ok and "gpt-7" in service.snapshot.models and "my-model" in service.snapshot.models
        assert screen.poll_text.value.startswith("✓ 2 online · openai")
        # Custom prefixes tab
        screen.set_tab("prefixes")
        editor = screen.open_prefix_editor(None)
        assert isinstance(editor, PrefixEditor)
        editor.prefix.value, editor.routing.value = "", ""
        assert editor.save() == "Row 1 needs both a prefix and Base URL."
        page.update()
        store._saver.close()

    asyncio.run(scenario())


@needs_flet
@needs_run_env
def test_prefix_editor_saves_valid_routes(tmp_path):
    from glossarion_mobile.ui.screens.model_manager import ModelManagerScreen

    async def scenario():
        conn, session = _fake_session("android")
        store = _store(tmp_path, {})
        service = _service(store)
        service.load_blocking()
        screen = ModelManagerScreen(parse_route("/settings/models"), catalog=service, page=session.page)
        screen.get_body()
        screen.set_tab("prefixes")
        editor = screen.open_prefix_editor(None)
        editor.prefix.value, editor.routing.value = "my prefix", "http://x"
        assert editor.save() == "Prefix on row 1 cannot contain spaces."
        editor.prefix.value = "mine"
        assert editor.save() is None
        assert store.get("custom_prefix_routes") == [{"prefix": "mine/", "routing": "http://x",
                                                       "endpoint_type": "/chat/completions"}]
        dialog = screen.delete_prefix(0)
        await dialog._on_confirm()
        assert store.get("custom_prefix_routes") == []
        store._saver.close()

    asyncio.run(scenario())


class FakeFiles:
    def __init__(self, picked=None):
        self.picked = picked
        self.exported = []

    def export_options(self, path):
        return [types.SimpleNamespace(id="share", label="Share…", icon="IOS_SHARE", disabled_reason=None)]

    async def export(self, option_id, path):
        self.exported.append((option_id, json.loads(Path(path).read_text(encoding="utf-8"))))
        return True

    async def pick_files(self, **kwargs):
        return [types.SimpleNamespace(path=self.picked)] if self.picked else []


@needs_flet
def test_keys_screen_pools_cards_bulk_actions_and_import_export(tmp_path):
    from glossarion_mobile.ui.screens.keys import ContextSheet, KeysScreen

    async def scenario():
        conn, session = _fake_session("android")
        page = session.page
        store, keys = _keys(tmp_path, {"use_multi_api_keys": True, "multi_api_keys": [
            {"api_key": "sk-aaaaaaaaaaaa1111", "model": "gpt-6", "last_test_result": "passed"},
            {"api_key": "bad-key-0000000000", "model": "gpt-6-mini", "enabled": False}]},
            runner=lambda cfg, fn: fn())
        notes = []
        files = FakeFiles()
        screen = KeysScreen(parse_route("/settings/keys/glossary"), controller=keys, ctx=_settings_ctx(store, page),
                            files=files, page=page, notify=lambda m, a=None, cb=None: notes.append((m, a, cb)),
                            run_io=lambda fn, *a: asyncio.to_thread(fn, *a), spawn=lambda c: asyncio.ensure_future(c),
                            export_dir=str(tmp_path / "exports"))
        assert screen.pool == "glossary"
        page.views[0].controls.append(screen.get_body())
        screen.actions()
        screen.did_show()
        page.update()
        assert len(screen.pool_chips) == 11
        screen.select_pool("main")
        assert len(screen.cards) == 2
        # pool switch / rotation settings
        screen.pool_switch.value = False
        screen.pool_switch.on_change(types.SimpleNamespace(control=screen.pool_switch))
        assert store.get("use_multi_api_keys") is False
        screen.frequency_field.value = "0"
        screen._on_frequency()
        assert screen.frequency_field.error and not store.has("rotation_frequency")
        screen.frequency_field.value = "4"
        screen._on_frequency()
        assert store.get("rotation_frequency") == 4
        # add a key through the editor
        editor = screen.open_editor(None)
        editor.key_field.set_value("sk-new-key-99999999", notify=False)
        editor.model_picker.set_value("claude-opus-5-5")
        assert editor.save() and keys.count("main") == 3
        # selection: bulk disable, remove + undo, contexts, move
        screen.toggle_select(0)
        screen.toggle_select(2)
        assert screen.selection_bar.visible
        assert screen.bulk_enable(False) == 2
        removed = screen.remove_selected()
        assert len(removed) == 2 and keys.count("main") == 1
        notes[-1][2]()  # Undo
        assert keys.count("main") == 3
        screen.toggle_select(1)
        sheet = screen.open_context_sheet()
        assert isinstance(sheet, ContextSheet)
        sheet.cycle("translation")
        sheet.apply()
        assert "translation" in keys.keys("main")[1]["disabled_contexts"]
        screen.toggle_select(1)
        screen.toggle_select(0)
        assert screen.copy_selected("glossary", move=True) == 1 and keys.count("glossary") == 1
        # tests: live per-key results, summary. U9: the Translation pool tests only its enabled keys
        # (desktop _test_all "Only test enabled keys" / "No enabled keys to test")
        main_keys = keys.keys("main")
        keys.set_keys_enabled("main", range(len(main_keys)), False)
        assert await screen.test_all() == [] and notes[-1][0] == "No enabled keys to test"
        keys.set_keys_enabled("main", [0], True)
        results = await screen.test_all()
        assert notes[-1][0].startswith("Test complete:") and [i for i, _r in results] == [0]
        keys.set_keys_enabled("main", range(len(main_keys)), True)
        results = await screen.test_all()
        assert notes[-1][0].startswith("Test complete:") and len(results) == 2
        # export: plain-text warning, then the FileBridge option; the temp file is removed
        confirm = screen.confirm_export()
        assert "plain text" in confirm.dialog.content.content.controls[0].value
        path = await screen.export_keys()
        assert path and screen.last_sheet is not None
        assert not os.path.exists(path)  # nothing in plain text on disk until an export option is chosen
        screen.last_sheet._on_select(None, screen.last_sheet.items[0])
        for _ in range(20):
            await asyncio.sleep(0.02)
            if files.exported and not os.path.exists(path):
                break
        assert files.exported[0][1]["format"] == "glossarion-key-pools" and not os.path.exists(path)
        # import: replaces the listed pools after confirmation; the picked copy is deleted
        picked = tmp_path / "picked.json"
        picked.write_text(json.dumps({"format": "glossarion-key-pools", "version": 1,
                                      "pools": {"tts": {"enabled": True, "keys": [{"api_key": "t", "model": "tts-1"}]}}}),
                          encoding="utf-8")
        files.picked = str(picked)
        plan = await screen.import_keys()
        assert plan.items[0][0] == "tts" and not picked.exists()
        await screen.last_dialog._on_confirm()
        assert keys.count("tts") == 1 and store.get("use_tts_keys") is True
        # clear all with confirm + undo
        screen.select_pool("tts")
        dialog = screen.confirm_clear()
        await dialog._on_confirm()
        assert keys.count("tts") == 0
        notes[-1][2]()
        assert keys.count("tts") == 1
        page.update()
        screen.dispose()
        store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_key_editor_validates_and_round_trips_unknown_fields(tmp_path):
    from glossarion_mobile.ui.screens.key_editor import KeyEditor

    async def scenario():
        conn, session = _fake_session("android")
        store = _store(tmp_path, {})
        ctx = _settings_ctx(store, session.page)
        saved = []
        entry = {"api_key": "ENC:abc", "model": "gpt-6", "times_used": 7, "future_field": "x",
                 "disabled_contexts": ["glossary", "legacy_ctx"], "request_parameters": {"top_p": 0.9}}
        editor = KeyEditor(ctx, entry=entry, pool_id="main", contexts=("translation", "glossary"),
                           on_save=lambda e: saved.append(e) or None).show()
        assert editor.key_field.encrypted and editor.key_field.field.value == ""
        assert editor.context_enabled == {"translation": True, "glossary": False}
        editor.number_fields["cooldown"][0].value = "5"
        assert not editor.save() and "between 10 and 3600" in editor.error_text.value
        editor.number_fields["cooldown"][0].value = "120"
        editor.number_fields["individual_key_temperature"][0].value = "0.4"
        editor.toggle_context("glossary")
        editor._add_param_row("model", "x", push=False)
        assert not editor.save() and "controlled by Glossarion" in editor.error_text.value
        editor._remove_param_row(editor.param_rows[-1][0])
        editor.endpoint_switch.value = True
        editor.endpoint_field.value = "https://res.openai.azure.com"
        creds = editor.open_credentials()
        # U9: the desktop pickers' check (settings_rules.google_credentials_error): a service-account JSON
        bad = tmp_path / "not_sa.json"
        bad.write_text('{"hello": 1}', encoding="utf-8")
        creds.field.value = str(bad)
        assert not creds.save() and creds.error_text.value.startswith("Invalid Google Cloud credentials file")
        good = tmp_path / "sa.json"
        good.write_text('{"type": "service_account", "project_id": "p"}', encoding="utf-8")
        creds.field.value = str(good)
        assert creds.save() and editor.creds_field.value == str(good)
        assert editor.save()
        out = saved[-1]
        assert out["api_key"] == "ENC:abc" and out["times_used"] == 7 and out["future_field"] == "x"
        assert out["cooldown"] == 120 and out["individual_key_temperature"] == 0.4
        assert out["individual_output_token_limit"] is None and out["api_call_delay"] == 0.0
        assert out["disabled_contexts"] == ["legacy_ctx"] and out["request_parameters"] == {"top_p": 0.9}
        assert out["use_individual_endpoint"] and out["azure_endpoint"] == "https://res.openai.azure.com"
        assert out["google_credentials"] == str(tmp_path / "sa.json")
        blank = KeyEditor(ctx, entry={"api_key": "", "model": ""}, new=True, on_save=lambda e: None).show()
        assert not blank.save() and blank.error_text.value == "Please enter a model name"
        # dedicated pools keep the dict shape of their desktop "Add key" button
        import key_pool_service

        plain = KeyEditor(ctx, entry=key_pool_service.new_key_entry("k", "m"), pool_id="glossary",
                          contexts=("glossary",), on_save=lambda e: saved.append(e) or None, new=True).show()
        assert plain.save()
        assert not {"cooldown", "request_parameters", "disabled_contexts"} & set(saved[-1])
        assert set(key_pool_service.NEW_KEY_ENTRY_FIELDS) <= set(saved[-1])
        session.page.update()
        store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_endpoints_screen_tiles_quick_paste_dependency_and_test(tmp_path):
    from glossarion_mobile.ui.components.reason_chip import ReasonChip
    from glossarion_mobile.ui.screens.endpoints import ENDPOINT_SECTIONS, EndpointsScreen

    async def scenario():
        conn, session = _fake_session("android")
        page = session.page
        store = _store(tmp_path, {"model": "gpt-6", "api_key": "sk"})
        ctx = _settings_ctx(store, page)
        tested = []

        def tester(config):
            tested.append(config.get("openai_base_url"))
            return {"ok": True, "status": "passed", "model": config.get("model")}

        opened = []
        screen = EndpointsScreen(parse_route("/settings/endpoints"), ctx, tester=tester,
                                 run_io=lambda fn, *a: asyncio.to_thread(fn, *a), open_local_ai=lambda: opened.append(1),
                                 dependency={"gemini_openai_endpoint": ("Needs grpcio · not in this build", "detail")})
        page.views[0].controls.append(screen.get_body())
        screen.did_show()
        page.update()
        wanted = {k for _t, keys in ENDPOINT_SECTIONS for k in keys}
        assert wanted <= set(screen.tiles)
        assert not screen.tiles["authza_use_general_api"].editable  # unavailable on mobile, shown disabled
        tile = screen.tiles["openai_base_url"]
        assert screen.quick_paste(tile, "http://192.168.1.10:11434/v1")
        assert store.get("openai_base_url") == "http://192.168.1.10:11434/v1"
        found = [c for c in screen.list_view.controls]
        assert any(isinstance(getattr(c, "content", None), ReasonChip) for card in found
                   for c in getattr(getattr(card, "content", None), "controls", []) or [])
        secret = screen.secret_fields["replicate_api_key"]
        assert secret.field.password and secret.test_button.disabled
        secret.set_value("r8_secretvalue123456")
        assert store.get("replicate_api_key") == "r8_secretvalue123456"
        secret.set_value("")
        assert not store.has("replicate_api_key")
        store.set("use_custom_openai_endpoint", True)
        result = await screen.run_test()
        assert result["status"] == "passed" and tested == ["http://192.168.1.10:11434/v1"]
        assert screen.result_text.value.startswith("✅ Connected successfully!")
        assert "OpenAI (Custom): http://192.168.1.10:11434/v1" in screen.summary_text.value
        screen._open_local_ai()
        assert opened == [1]
        screen.dispose()
        page.update()
        store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_local_ai_and_refusal_screens(tmp_path):
    from glossarion_mobile.ui.screens.local_ai import LocalAiScreen
    from glossarion_mobile.ui.screens.refusal_patterns import RefusalModel, RefusalPatternsScreen

    async def scenario():
        conn, session = _fake_session("android")
        page = session.page
        store = _store(tmp_path, {"custom_prefix_routes": [{"prefix": "keep/", "routing": "https://x/v1",
                                                             "endpoint_type": "/chat/completions"}]})

        class Catalog:
            refreshed = []

            def save_prefix_routes(self, rows):
                store.set("custom_prefix_routes", [dict(r) for r in rows])
                return True, None

            async def refresh(self, provider, explicit=True):
                self.refreshed.append(provider)
                return mc.RefreshOutcome(provider, True, statuses={provider: "online (2 models)"})

        catalog = Catalog()
        screen = LocalAiScreen(None, store=store, catalog=catalog, spawn=lambda c: asyncio.ensure_future(c))
        page.views[0].controls.append(screen.get_body())
        page.update()
        screen.url_fields["ollama"].value = "bad host!"
        assert screen.save_server("ollama")
        screen.url_fields["ollama"].value = "192.168.1.10"
        assert screen.save_server("ollama") is None
        routes = store.get("custom_prefix_routes")
        assert routes[0]["prefix"] == "keep/" and routes[1]["routing"] == "http://192.168.1.10:11434/v1"
        outcome = await screen.load_models("ollama")
        assert catalog.refreshed == ["custom:ollama-lan/"] and outcome.ok
        assert "online (2 models)" in screen.status["ollama"].value
        assert screen.remove_server("ollama") and len(store.get("custom_prefix_routes")) == 1
        if screen.options_form is not None:
            form = screen.options_form
            form.model_field.value = "qwen3:8b"
            form.fields["num_ctx"].value = "4096"
            assert form.save() is None
            assert store.get("ollama_settings")["models"]["qwen3:8b"]["options"] == {"num_ctx": 4096}
            form.fields["num_ctx"].value = "-1"
            assert form.save() == "num_ctx must be greater than zero"

        notes = []
        model = RefusalModel(store)
        refusal = RefusalPatternsScreen(None, model=model, page=page, notify=lambda m, a=None, cb=None: notes.append((m, cb)),
                                        run_io=lambda fn, *a: asyncio.to_thread(fn, *a))
        page.views[0].controls.append(refusal.get_body())
        page.update()
        refusal.new_field.value = "I Refuse"
        assert refusal.add_pattern() == "i refuse" and model.patterns()[0] == "i refuse"
        refusal.toggle("as an ai")
        dialog = refusal.confirm_delete()
        await dialog._on_confirm()
        assert "as an ai" not in model.patterns()
        notes[-1][1]()  # Undo
        assert "as an ai" in model.patterns()
        txt = tmp_path / "patterns.txt"
        txt.write_text("New One\n# skip\nas an ai\n", encoding="utf-8")
        assert await refusal.load_from_file(str(txt)) == (1, 1)
        assert notes[-1][0] == "✅ Loaded 1 new, 1 skipped"
        refusal.limit_field.value = "-5"
        assert refusal._on_limit() == 1000
        refusal.disable_switch.on_change(types.SimpleNamespace(control=types.SimpleNamespace(value=False)))
        assert store.get("disable_refusal_checks") is False
        page.update()
        store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_models_keys_feature_installs_screens_and_sheet_env(tmp_path, monkeypatch):
    from glossarion_mobile.ui.screens import model_manager as mm
    from glossarion_mobile.ui.sheets import model_sheet as ms

    async def scenario():
        conn, session = _fake_session("android")
        store = _store(tmp_path, {"model": "gpt-6"})
        monkeypatch.setattr(mm, "ModelCatalogService",
                            lambda s, **kw: mc.ModelCatalogService(s, options=FakeOptions(), is_mobile=False, **kw))
        from glossarion_mobile.ui.screens import keys as keys_module

        real_backend = keys_module.KeyBackend
        monkeypatch.setattr(keys_module, "KeyBackend", lambda *a, **k: real_backend(_fake_key_service()))

        class Shell:
            tablet = False
            current_route = "/settings"

            def __init__(self):
                self.screen_factory = lambda match: ("fallback", match.name)
                self.overlays = []

            def push_overlay(self, view):
                self.overlays.append(view)

        settings = types.SimpleNamespace(ctx=_settings_ctx(store, session.page))
        app = types.SimpleNamespace(page=session.page, dispatcher=None, settings=settings, config_store=store, prefs=None,
                                    shell=Shell(), navigate_to=lambda *a: None, notify=lambda *a: None,
                                    state=types.SimpleNamespace(signed_in=types.SimpleNamespace(value=frozenset({"authgpt"}))),
                                    paths=None, files=None)
        try:
            feature = await mm.ModelsKeysFeature.install(app)
            assert app.models_keys is feature and mc.default_service() is feature.catalog
            assert ms.sheet_env().catalog is feature.catalog and ms.sheet_env().signed_in_keys() == {"authgpt"}
            for route, cls in (("/settings/models", "ModelManagerScreen"), ("/settings/keys", "KeysScreen"),
                               ("/settings/keys/tts", "KeysScreen"), ("/settings/endpoints", "EndpointsScreen")):
                screen = app.shell.screen_factory(parse_route(route))
                assert type(screen).__name__ == cls, route
            assert app.shell.screen_factory(parse_route("/settings/about")) == ("fallback", "settings.about")
            refusal = feature.open_refusal_patterns()
            local = feature.open_local_ai()
            assert len(app.shell.overlays) == 2 and type(local).__name__ == "LocalAiScreen"
            assert refusal.model.patterns()  # the default list (single source)
            await asyncio.sleep(0.05)
            feature.detach()
        finally:
            ms.install_sheet_env(None)
            mc.set_default_service(None)
            store._saver.close()

    asyncio.run(scenario())


# ==========================================================================
# On the real app shell (main.py on the fake session)
# ==========================================================================

app_env = _TB.app_env
storage = _TB.storage


@needs_flet
def test_feature_on_the_real_app_shell(app_env, monkeypatch):
    from glossarion_mobile.ui.screens import keys as keys_module
    from glossarion_mobile.ui.screens import model_manager as mm
    from glossarion_mobile.ui.sheets import model_sheet as ms

    monkeypatch.setattr(mm, "ModelCatalogService",
                        lambda s, **kw: mc.ModelCatalogService(s, options=FakeOptions(), is_mobile=False, **kw))
    real_backend = keys_module.KeyBackend
    monkeypatch.setattr(keys_module, "KeyBackend", lambda *a, **k: real_backend(_fake_key_service()))

    async def scenario():
        main_module = _TB._load_main_module()
        conn, session = _fake_session("android")
        page = session.page
        await main_module.main(page)
        await session.after_event(page)
        app = page.data
        try:
            # GlossarionApp.start installs it once app.py calls ModelsKeysFeature.install (Integrate)
            feature = getattr(app, "models_keys", None) or await mm.ModelsKeysFeature.install(app)
            assert feature.ctx is app.settings.ctx and feature.store is app.config_store
            await _TB._route(session, "/settings/models")
            assert type(app.shell.top_screen).__name__ == "ModelManagerScreen"
            await _TB._route(session, "/settings/keys/fallback")
            screen = app.shell.top_screen
            assert type(screen).__name__ == "KeysScreen" and screen.pool == "fallback"
            await _TB._route(session, "/settings/endpoints")
            assert type(app.shell.top_screen).__name__ == "EndpointsScreen"
            # the chat header's U3 call now opens the full sheet with the app services
            sheet = app.chat_view.open_model_sheet("model")
            assert isinstance(sheet, ms.ModelSheet) and sheet.env.catalog is feature.catalog
            await asyncio.sleep(0.2)
            assert "authgpt/gpt-6-luna" in sheet.rows
            sheet.close()
            page.update()
        finally:
            ms.install_sheet_env(None)
            mc.set_default_service(None)
            await app.dispatcher.stop()
            await asyncio.sleep(0)

    asyncio.run(scenario())


@needs_flet
def test_endpoints_focus_scrolls_to_a_tile_of_a_lower_section(tmp_path):
    """A search hit / ``open_setting`` for a key far down the page (Text-to-speech): every card is built
    (``build_controls_on_demand=False``, like SectionPage), so ``scroll_to(scroll_key=…)`` reaches it."""
    from flet.messaging.protocol import MessageAction

    from glossarion_mobile.ui.screens.endpoints import EndpointsScreen

    async def scenario():
        conn, session = _fake_session("android")
        page = session.page
        store = _store(tmp_path, {"model": "gpt-6"})
        screen = EndpointsScreen(parse_route("/settings/s/other.endpoints#openai_tts_endpoint"), _settings_ctx(store, page),
                                 run_io=lambda fn, *a: asyncio.to_thread(fn, *a))
        page.views[0].controls.append(screen.get_body())
        page.update()
        list_view = screen.list_view
        assert list_view.build_controls_on_demand is False and list_view.auto_scroll is False
        keys = [getattr(c, "key", None) for c in list_view.controls]
        assert keys.index("endpoints-Text-to-speech") >= len(keys) // 2  # one of the lower cards
        assert screen.focus_target == "openai_tts_endpoint"
        conn.messages.clear()
        assert await screen.focus_key("openai_tts_endpoint", highlight_seconds=0)
        scrolls = [(m.body.args or {}).get("scroll_key") for m in conn.messages
                   if m.action == MessageAction.INVOKE_METHOD and m.body.name == "scroll_to"]
        assert [getattr(k, "value", None) for k in scrolls] == ["openai_tts_endpoint"]
        screen.dispose()
        store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_keys_screen_ticker_runs_only_while_shown_and_back_leaves_selection(app_env, monkeypatch):
    """Settings › API keys › <pool> on the real app shell: the live-stats ticker (a cooling key repaints
    every tick) parks while the app is in the background (UI_SPEC §7.3) and skips while another screen
    or an overlay covers it, with one refresh when shown again; Android back leaves selection mode
    before it leaves the screen (UI_SPEC §1.6 rule 2)."""
    from flet.messaging.protocol import MessageAction

    from glossarion_mobile.ui.screens import keys as keys_module
    from glossarion_mobile.ui.screens import model_manager as mm
    from glossarion_mobile.ui.sheets import model_sheet as ms

    monkeypatch.setattr(mm, "ModelCatalogService",
                        lambda s, **kw: mc.ModelCatalogService(s, options=FakeOptions(), is_mobile=False, **kw))
    real_backend = keys_module.KeyBackend
    monkeypatch.setattr(keys_module, "KeyBackend", lambda *a, **k: real_backend(_fake_key_service()))
    monkeypatch.setattr(keys_module, "LIVE_REFRESH_SECONDS", 0.05)

    async def settle(predicate, timeout=3.0):
        for _ in range(int(timeout / 0.05)):
            if predicate():
                return True
            await asyncio.sleep(0.05)
        return predicate()

    def confirm_pops(conn):
        return [(m.body.args or {}).get("should_pop") for m in conn.messages
                if m.action == MessageAction.INVOKE_METHOD and m.body.name == "confirm_pop"]

    async def scenario():
        main_module = _TB._load_main_module()
        conn, session = _fake_session("android")
        page = session.page
        await main_module.main(page)
        await session.after_event(page)
        app = page.data
        try:
            feature = getattr(app, "models_keys", None) or await mm.ModelsKeysFeature.install(app)
            app.config_store.set_many({"use_multi_api_keys": True, "multi_api_keys": [
                {"api_key": "sk-aaaaaaaaaaaa1111", "model": "gpt-6"}, {"api_key": "sk-bbbbbbbbbbbb2222", "model": "gpt-6"}]})
            await _TB._route(session, "/settings/keys/translation")
            screen = app.shell.top_screen
            assert isinstance(screen, keys_module.KeysScreen) and screen.pool == "main"
            # a running job's pool: a key cooling down (key_pool_service.live_key_stats rows)
            screen.controller.live_stats = lambda pool: [{"is_cooling_down": True, "success_count": 1,
                                                          "error_count": 0, "times_used": 1}, None]
            renders: list = []
            original = screen._external_refresh
            screen._external_refresh = lambda: (renders.append(1), original())
            # shown: every tick (LIVE_REFRESH_SECONDS, 0.05 s here) repaints the countdown
            assert await settle(lambda: len(renders) >= 3, timeout=1.5)
            await session.dispatch_event(page._i, "app_lifecycle_state_change", {"state": "hide"})
            assert page.app_visible is False
            await asyncio.sleep(0.15)
            renders.clear()
            await asyncio.sleep(0.4)
            assert renders == []  # backgrounded: parked
            await session.dispatch_event(page._i, "app_lifecycle_state_change", {"state": "resume"})
            assert await settle(lambda: len(renders) >= 1)  # resumed: refreshed again
            await _TB._route(session, "/settings/accounts")
            assert app.shell.top_screen is not screen and any(e.screen is screen for e in app.shell.stack)
            await asyncio.sleep(0.15)
            renders.clear()
            await asyncio.sleep(0.4)
            assert renders == []  # covered by another screen: no repaints
            app.shell.pop()
            assert app.shell.top_screen is screen and await settle(lambda: len(renders) >= 1)
            feature.open_refusal_patterns()  # an overlay View above the keys screen
            assert app.shell.overlays
            await asyncio.sleep(0.15)
            renders.clear()
            await asyncio.sleep(0.4)
            assert renders == []
            app.shell.pop()
            assert not app.shell.overlays and await settle(lambda: len(renders) >= 1)
            # Android back in selection mode: the selection goes, the screen stays
            screen.toggle_select(0)
            entry = next(e for e in app.shell.stack if e.screen is screen)
            assert entry.view.can_pop is False and callable(entry.view.on_confirm_pop)
            conn.messages.clear()
            await entry.view.on_confirm_pop(None)
            assert not screen.selected and app.shell.top_screen is screen and confirm_pops(conn) == [False]
            conn.messages.clear()
            await entry.view.on_confirm_pop(None)  # nothing selected: the View may pop
            assert confirm_pops(conn) == [True]
        finally:
            ms.install_sheet_env(None)
            mc.set_default_service(None)
            await app.dispatcher.stop()
            await asyncio.sleep(0)

    asyncio.run(scenario())
