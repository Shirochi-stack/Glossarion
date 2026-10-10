"""settings_rules: GUI-free settings rules shared by the desktop handlers and the mobile app (U4).

What is checked:

* GUI-free, cheap import (PySide6 blocked; the mixin / backend modules load lazily) and
  Python 3.10 syntax;
* model-route controls: ``route_controls`` (the mobile aggregate) shows exactly the login
  buttons, Google Cloud credential row, Vertex location field and GCP project picker the
  frozen desktop ``on_model_change`` shows for the same model, key pools and live hints
  (tier D in tests/parity covers the rewired desktop methods themselves);
* Chunk Size / compression rules (desktop thresholds, the round-down factor, the manual
  chunk-size hold);
* the adapters over the U2 mixin handlers (context mode / batching, glossary-mode shortcut,
  target-language fan-out) give the same config a HeadlessOwner (desktop start-up replay)
  ends with, and never write the process environment;
* ``evaluate_locks`` and ``register_lock_rule``.

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \\
        tests/test_settings_rules.py
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TESTS = REPO_ROOT / "tests"
for _p in (str(TESTS), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import settings_rules as sr  # noqa: E402
from _headless_env import headless_owner  # noqa: E402

HAS_QT = importlib.util.find_spec("PySide6") is not None


@pytest.fixture(autouse=True)
def _environment_is_never_written():
    before = dict(os.environ)
    yield
    changed = {k for k in set(before) | set(os.environ) if before.get(k) != os.environ.get(k)}
    # importing the backend (unified_api_client) may adjust PATH / PySide6 options once
    changed -= {"PATH", "PYSIDE6_OPTION_PYTHON_ENUM", "PYTEST_CURRENT_TEST"}
    assert not changed, f"settings_rules wrote the process environment: {sorted(changed)}"


# ---------------------------------------------------------------------------
# hygiene
# ---------------------------------------------------------------------------

def test_module_is_gui_free_cheap_and_python_310():
    source = (SRC / "settings_rules.py").read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 10))
    code = (
        "import sys; sys.modules['PySide6'] = None; sys.path.insert(0, %r); "
        "import settings_rules; "
        "heavy = [m for m in ('owner_state', 'run_env', 'unified_api_client', 'translator_gui', "
        "'dpi_setup', 'settings_schema') if m in sys.modules]; print(heavy)" % str(SRC)
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


def test_adapters_run_with_pyside6_blocked():
    code = (
        "import sys; sys.modules['PySide6'] = None; sys.path.insert(0, %r); "
        "import settings_rules as sr; "
        "print(sr.auto_glossary_modes(), sr.auto_compression_factor(65536), "
        "sr.context_mode({'contextual': True}), "
        "sorted(sr.evaluate_locks({})), "
        "sr.route_controls('authgem-vertex0/x', {}, api_key_check=False).logins, "
        "'translator_gui' in sys.modules)" % str(SRC)
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    line = out.stdout.strip().splitlines()[-1]
    # U9 (gap audit round 3): the compression factors lock while their Auto boxes are on
    assert line.endswith("3.0 contextual_history ['append_glossary', 'append_glossary_auto_load', 'batching_mode', "
                         "'compression_factor', 'enable_thoughts', 'fuzzy_auto_mapping', "
                         "'fuzzy_auto_mapping_threshold', 'glossary_compression_factor', "
                         "'translation_temperature'] ('authgem',) False")


# ---------------------------------------------------------------------------
# model-route controls
# ---------------------------------------------------------------------------

POOLS = {
    "none": {},
    "authgpt0_vertex": {"use_multi_api_keys": True, "multi_api_keys": [
        {"model": "authgpt0/gpt-5", "api_key": "k"}, {"model": "vertex/gemini-2.5-pro", "enabled": False},
        {"model": "authgem-vertex2/gemini"}]},
    "translate_cd_grok": {"use_glossary_keys": True, "glossary_keys": [
        {"model": "google-translate"}, {"model": "authcd1/claude"}, {"model": "authgrok0/grok-4"}],
        "use_tts_keys": True, "tts_keys": [{"model": "gemini-tts@001"}]},
    "disabled_toggle": {"use_multi_api_keys": False, "multi_api_keys": [{"model": "authgpt/x"}]},
    "junk": {"use_fallback_keys": True, "fallback_keys": [{"model": "authgem/gemini-3"}, {"model": 5}, "junk"],
             "use_metadata_keys": True, "metadata_keys": [{"model": "authgpt0/x"}]},
}
MODELS = ("authgpt/gpt-6-luna", "authgpt0/gpt-5", "authgpt3/x", "authgrok/grok-4", "authgrok0/x", "authcd/claude",
          "authcd2/x", "authgem/gemini-3", "authgem-vertex/x", "authgem-vertex0/x", "authgem-vertex4/x",
          "vertex/gemini-2.5-pro", "vertex_ai/x", "gemini-2.5-pro@001", "google-translate", "Google-Translate",
          "google-translate-free", "gpt-4o", "", "ocagy/x", "antigravity/claude")
HINTS = ({}, {"needs_google_creds": True}, {"authgpt_pool": True}, {"authgrok_pool": True},
         {"authgem_vertex_model": "authgem-vertex1/x"})
HINT_ATTRS = {"needs_google_creds": "_multi_key_manager_needs_google_creds_hint",
              "authgpt_pool": "_multi_key_manager_authgpt_pool_hint",
              "authgrok_pool": "_multi_key_manager_authgrok_pool_hint",
              "authgem_vertex_model": "_multi_key_manager_authgem_vertex_model_hint"}


@pytest.fixture(scope="module")
def legacy_owner_class():
    """The frozen desktop owner (tests/parity oracle at the U4 parent commit)."""
    if not HAS_QT:
        pytest.skip("the frozen oracle needs PySide6")
    from parity import fuzz_moved as fm

    sess = fm.session()
    try:
        cls = sess.legacy_class()
    except fm.Unavailable as exc:
        pytest.skip(str(exc))
    if fm.resolve_python_mro(cls, "on_model_change") is fm.MISSING:
        pytest.skip("oracle lacks on_model_change (re-run tests/parity/freeze_legacy.py)")
    return cls


def _legacy_on_model_change(cls, model, config, hints):
    from parity import fakes

    owner = cls(fakes.CallRecorder())
    owner.config = copy.deepcopy(config)
    owner.model_var = model
    names = ("vertex_location_entry", "gcloud_button", "gcloud_status_label", "authgpt_login_btn",
             "authgrok_login_btn", "authcd_login_btn", "authgem_login_btn", "authgem_status_btn",
             "authgem_project_combo")
    for name in names:
        widget = fakes.FakeStatefulWidget(name=name)
        widget.hide()
        setattr(owner, name, widget)
    owner.gcloud_button_enabled_style = "on"
    owner.gcloud_button_disabled_style = "off"
    for key, value in hints.items():
        setattr(owner, HINT_ATTRS[key], value)
    owner.on_model_change()
    return owner


@pytest.mark.parametrize("pool", sorted(POOLS))
@pytest.mark.parametrize("hint", range(len(HINTS)))
def test_route_controls_match_the_frozen_desktop_handler(legacy_owner_class, tmp_path, pool, hint):
    creds = tmp_path / "sa.json"
    creds.write_text(json.dumps({"project_id": "proj-x"}), encoding="utf-8")
    hints = HINTS[hint]
    for model, creds_path in itertools.product(MODELS, (None, str(creds), str(tmp_path / "missing.json"))):
        config = copy.deepcopy(POOLS[pool])
        if creds_path:
            config["google_cloud_credentials"] = creds_path
        owner = _legacy_on_model_change(legacy_owner_class, model, config, hints)
        rc = sr.route_controls(model, config, hints=hints, api_key_check=False)
        where = (model, pool, hints, creds_path)
        for provider in ("authgpt", "authgrok", "authcd", "authgem"):
            assert getattr(owner, f"{provider}_login_btn").isVisible() == (provider in rc.logins), where
        assert owner.gcloud_button.isEnabled() == rc.needs_google_creds, where
        assert owner.vertex_location_entry.isVisible() == rc.vertex_location, where
        assert owner.gcloud_status_label.text() == rc.google_creds_text, where
        gem_calls = [c for c in owner._parity_recorder.calls if c[0] == "_update_authgem_login_status"]
        if gem_calls:
            assert gem_calls[-1][2] == {"needs_vertex": rc.authgem_vertex}, where
        assert owner._authgpt_pool_route_requested(model) == rc.authgpt_pool, where


def test_route_controls_for_the_default_model():
    rc = sr.route_controls("authgpt/gpt-6-luna", {})
    assert rc.logins == ("authgpt",) and rc.needs_login("authgpt")
    assert rc.excluded is None and not rc.needs_google_creds and not rc.vertex_location
    assert rc.account_ids["authgpt"] == (0,)
    assert rc.needs_api_key in (False, None)


def test_route_controls_pools_and_slots():
    config = POOLS["authgpt0_vertex"]
    rc = sr.route_controls("gpt-4o", config, api_key_check=False)
    assert rc.logins == ("authgpt", "authgem")
    assert rc.authgpt_pool and rc.authgem_vertex and rc.vertex_location is False
    assert rc.account_ids["authgem"] == (2,) and rc.account_ids["authgpt"] == (0,)
    rc = sr.route_controls("vertex/gemini-2.5-pro", {}, api_key_check=False)
    assert rc.needs_google_creds and rc.vertex_location and rc.google_creds_level == "warning"


@pytest.mark.parametrize("model", ["ocagy/x", "ocz/free", "autharena/x", "search/opera"])
def test_excluded_routes_carry_a_reason_on_mobile_only(model):
    assert sr.excluded_route_reason(model)
    assert sr.excluded_route_reason(model.upper())
    assert sr.excluded_route_reason(model, platform="desktop") is None
    assert sr.route_controls(model, {}, api_key_check=False).excluded


@pytest.mark.parametrize("model", ["antigravity/claude", "authza/glm", "authza2/glm", "ollamapull/llama3",
                                   "authnan/z-ai/glm-5.3"])
def test_remote_and_signin_routes_run_on_mobile(model):
    # U13/U14: antigravity/, authza/ and ollamapull/ through a server on the user's PC, authnan/ via sign-in
    assert sr.excluded_route_reason(model) is None


@pytest.mark.parametrize("model", ["ollama/llama3", "lmstudio/x", "search/gemini", "authgpt/x", "gpt-4o"])
def test_supported_routes_have_no_exclusion(model):
    assert sr.excluded_route_reason(model) is None


def test_google_creds_status_texts(tmp_path):
    good = tmp_path / "sa.json"
    good.write_text('{"project_id": "p1"}', encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert sr.google_creds_status("vertex/x", {"google_cloud_credentials": str(good)}) == (
        "✓ Credentials: sa.json (Project: p1)", "ready")
    assert sr.google_creds_status("google-translate", {"google_cloud_credentials": str(good)}) == (
        "✓ Google Translate ready\n(Project: p1)", "ready")
    assert sr.google_creds_status("vertex/x", {"google_cloud_credentials": str(bad)}) == (
        sr.GOOGLE_CREDS_READ_ERROR, "error")
    assert sr.google_creds_status("vertex/x", {"google_cloud_credentials": str(tmp_path / "x")}) == (
        sr.GOOGLE_CREDS_FILE_MISSING, "error")
    assert sr.google_creds_status("google-translate", {}) == (
        "⚠ Google Cloud credentials needed for Translate API", "warning")


def test_pool_helpers_accept_a_custom_pool_source():
    source = lambda: iter([("glossary_keys", "use_glossary_keys", "authgpt0/model")])  # noqa: E731
    assert sr.authgpt_pool_route_requested("other/model", pool_models=source)
    assert not sr.authgpt_pool_route_requested("other/model", {})
    assert sr.authgpt_pool_route_requested("other/model", {}, hint=True)
    assert sr.authgem_vertex_control_model("x", "AuthGem-Vertex3/y", {}) == "authgem-vertex3/y"
    assert list(sr.iter_enabled_key_pool_models(POOLS["junk"])) == [
        ("fallback_keys", "use_fallback_keys", "authgem/gemini-3"),
        ("fallback_keys", "use_fallback_keys", 5),
        ("fallback_keys", "use_fallback_keys", ""),
        ("metadata_keys", "use_metadata_keys", "authgpt0/x"),
    ]


# ---------------------------------------------------------------------------
# chunk size / compression
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("limit, factor", [(1, 1.5), (16378, 1.5), (16379, 2.0), (32768, 2.0), (32769, 2.5),
                                           (65535, 2.5), (65536, 3.0), (400000, 3.0), ("8000", 1.5),
                                           ("abc", None)])
def test_auto_compression_factor_table(limit, factor, capsys):
    assert sr.auto_compression_factor(limit) == factor


def test_chunk_budget_and_round_trip():
    assert sr.chunk_budget(65536, 3.0) == 21678
    assert sr.chunk_budget(65536, 0) is None and sr.chunk_budget("abc", 3.0) is None
    assert sr.chunk_budget(600, 3.0) == 1000  # clamp
    assert sr.chunk_budget_for_config({}) == sr.chunk_budget(128000, "3.0")
    for output in (16000, 65536, 128000):
        for chunk in range(1000, output - 500, 997):
            factor = sr.factor_for_chunk_size(output, chunk)
            assert sr.chunk_budget(output, factor) == chunk, (output, chunk, factor)
    assert sr.factor_for_chunk_size(65536, 0) is None and sr.factor_for_chunk_size(400, 10) is None


@pytest.mark.parametrize("text, expected", [("Auto", None), (" auto ", None), ("", None), ("12,000", 12000),
                                            ("2.7", 2), ("abc", 0), ("-5", -5), ("1e3", 1000)])
def test_parse_chunk_size_text(text, expected):
    assert sr.parse_chunk_size_text(text) == expected


def test_chunk_size_field_flow():
    config = {"max_output_tokens": 65536}
    assert sr.set_chunk_size_text(config, "20,000") == "20,000"
    assert config["auto_compression_factor"] is False and config["manual_chunk_size"] == 20000
    assert sr.chunk_budget_for_config(config) == 20000
    # the output limit changes: the manual chunk size is held by recomputing the factor
    config["max_output_tokens"] = 128000
    assert sr.hold_manual_chunk_size(config) is not None
    assert sr.chunk_budget_for_config(config) == 20000
    assert sr.set_chunk_size_text(config, "0") is None and config["manual_chunk_size"] == 20000
    assert sr.set_chunk_size_text(config, "auto") == "Auto"
    assert config["auto_compression_factor"] is True and config["compression_factor"] == 3.0
    assert sr.held_manual_chunk_size(config) is None
    remembered = {"auto_compression_factor": False}
    sr.remember_manual_chunk_size(remembered, lambda: 4321)
    assert remembered["manual_chunk_size"] == 4321


# ---------------------------------------------------------------------------
# adapters over the U2 mixin handlers == a HeadlessOwner's start-up replay
# ---------------------------------------------------------------------------

CONTEXT_CONFIGS = [
    {}, {"contextual": True}, {"use_rolling_summary": True}, {"use_rolling_summary": True, "rolling_summary_mode": "append"},
    {"contextual": True, "use_rolling_summary": True}, {"batching_mode": "direct"},
    {"contextual": True, "batching_mode": "aggressive"}, {"contextual": True, "batching_mode": "conservative"},
    {"use_rolling_summary": True, "batching_mode": "bogus"}, {"batching_mode": "conservative"},
]


@pytest.mark.parametrize("index", range(len(CONTEXT_CONFIGS)))
def test_context_batching_matches_the_desktop_start_up(tmp_path, monkeypatch, index):
    config = CONTEXT_CONFIGS[index]
    with headless_owner(tmp_path, monkeypatch, copy.deepcopy(config)) as owner:
        desktop_mode = owner.context_mode_var
        desktop_batching = owner.config["batching_mode"]
        desktop_controls = owner._translation_batching_mode_for_env()
    probe = copy.deepcopy(config)
    probe.setdefault("batching_mode", "aggressive")
    assert sr.context_mode(probe) == desktop_mode
    state = sr.enforce_context_batching(probe)
    assert state.context_mode == desktop_mode
    assert probe["batching_mode"] == state.batching_mode == desktop_batching
    if desktop_mode == "off":
        assert state.allowed == ("aggressive",) and desktop_controls == "aggressive"
    else:
        assert state.allowed == ("conservative", "direct")


@pytest.mark.parametrize("mode", ["off", "contextual_history", "rolling_summary_replace", "rolling_summary_append"])
def test_apply_context_mode_matches_the_desktop_combo(tmp_path, monkeypatch, mode):
    with headless_owner(tmp_path, monkeypatch, {"batching_mode": "aggressive"}) as owner:
        owner.context_mode_combo.setCurrentIndex(owner.context_mode_combo.findData(mode))
        owner._on_context_mode_changed()
        expected = {k: owner.config[k] for k in ("contextual", "use_rolling_summary", "rolling_summary_mode",
                                                   "batching_mode")}
    config = {"batching_mode": "aggressive"}
    state = sr.apply_context_mode(config, mode)
    assert {k: config[k] for k in expected} == expected
    assert state.context_mode == mode


def test_auto_glossary_modes_and_flags():
    assert sr.auto_glossary_modes() == ("off", "off_fuzzy_automap", "off_no_automap", "no_glossary", "minimal",
                                        "balanced", "full", "single_pass")
    flags = {mode: sr.auto_glossary_mode_flags(mode) for mode in sr.auto_glossary_modes()}
    assert [m for m, f in flags.items() if f["append_glossary_auto_load"] is True] == [
        "off", "off_fuzzy_automap", "balanced", "full", "single_pass"]
    assert [m for m, f in flags.items() if f["append_glossary_auto_load"] is False] == ["off_no_automap", "minimal"]
    assert [m for m, f in flags.items() if f["fuzzy_auto_mapping"]] == ["off_fuzzy_automap"]
    assert [m for m, f in flags.items() if f["enable_auto_glossary"]] == ["minimal", "balanced", "full",
                                                                          "single_pass"]
    assert flags["no_glossary"]["append_glossary"] is None
    assert sr.auto_glossary_mode_index("BALANCED") == 5 and sr.auto_glossary_mode_index("bogus") == 0


@pytest.mark.parametrize("mode", ["off", "off_fuzzy_automap", "off_no_automap", "no_glossary", "minimal",
                                  "balanced", "full", "single_pass", "bogus"])
def test_apply_auto_glossary_mode_matches_the_desktop_shortcut(tmp_path, monkeypatch, mode):
    start = {"auto_glossary_mode": "balanced", "append_glossary": False, "fuzzy_auto_mapping": True}
    with headless_owner(tmp_path, monkeypatch, copy.deepcopy(start)) as owner:
        monkeypatch.setattr(owner, "save_config", lambda show_message=False: True, raising=False)
        started = copy.deepcopy(owner.config)  # the config after the desktop start-up replay
        owner._on_auto_glossary_shortcut_changed(sr.auto_glossary_mode_index(mode))
        expected = {k: owner.config.get(k) for k in sr.AUTO_GLOSSARY_MODE_KEYS}
    config = started
    changed = sr.apply_auto_glossary_mode(config, mode)
    assert {k: config.get(k) for k in sr.AUTO_GLOSSARY_MODE_KEYS} == expected
    assert set(changed) <= set(sr.AUTO_GLOSSARY_MODE_KEYS)


def test_fan_out_target_language_matches_the_desktop_combo(tmp_path, monkeypatch):
    start = {"output_language": "English", "ai_hunter_config": {"language_detection": {"target_language": "x"}}}
    with headless_owner(tmp_path, monkeypatch, copy.deepcopy(start)) as owner:
        started = copy.deepcopy(owner.config)  # the config after the desktop start-up replay
        owner.update_target_language("Korean")
        expected_env = {k: os.environ.get(k) for k in ("OUTPUT_LANGUAGE", "GLOSSARY_TARGET_LANGUAGE")}
        keys = ("output_language", "glossary_target_language", "manga_settings", "ai_hunter_config")
        expected = {k: copy.deepcopy(owner.config.get(k)) for k in keys}
    config = started
    env = sr.fan_out_target_language(config, "Korean")
    assert env == expected_env
    assert {k: config.get(k) for k in keys} == expected
    plain = copy.deepcopy(start)
    sr.fan_out_target_language(plain, "Korean")
    assert plain["output_language"] == plain["glossary_target_language"] == "Korean"
    assert plain["manga_settings"]["manual_edit"]["translate_target_language"] == "Korean"
    assert plain["ai_hunter_config"]["language_detection"]["target_language"] == "Korean"


# ---------------------------------------------------------------------------
# locks
# ---------------------------------------------------------------------------

def test_evaluate_locks():
    locks = sr.evaluate_locks({})
    assert locks["batching_mode"].locked and locks["batching_mode"].value == "aggressive"
    assert not locks["translation_temperature"].locked
    locks = sr.evaluate_locks({"contextual": True, "disable_temperature": True}, keys=["translation_temperature"])
    assert set(locks) == {"translation_temperature"} and locks["translation_temperature"].locked
    config = {"contextual": True}
    snapshot = copy.deepcopy(config)
    assert sr.evaluate_locks(config)["batching_mode"].allowed == ("conservative", "direct")
    assert config == snapshot


def test_register_lock_rule_extends_and_replaces():
    sr.register_lock_rule("_test_rule", ("x",), lambda cfg: {"x": sr.LockInfo(True, "because")})
    try:
        assert sr.evaluate_locks({}, keys=["x"]) == {"x": sr.LockInfo(True, "because")}
        assert sr.lock_rules()["_test_rule"] == ("x",)
    finally:
        sr._LOCK_RULES.pop("_test_rule", None)


# ---------------------------------------------------------------------------
# U4 chain 2: Other Settings / Glossary Manager rules (thinking lock, output mode, glossary
# mode locks), evaluate_locks aggregation and the settings_schema rule ids
# ---------------------------------------------------------------------------

class _restore_env:
    """The desktop handlers export env vars; put the process env back for the autouse check."""

    def __enter__(self):
        self.saved = dict(os.environ)
        return self

    def __exit__(self, *exc):
        os.environ.clear()
        os.environ.update(self.saved)
        return False


def test_thoughts_lock_rule():
    assert sr.thoughts_lock_state(True) == (True, "1", True)
    assert sr.thoughts_lock_state(0) == (False, "0", False)
    config = {"enable_thoughts": False}
    assert sr.apply_thoughts_lock(config, True) == {"ENABLE_THOUGHTS": "1"} and config["enable_thoughts"] is True
    assert sr.thoughts_lock({}) == {"enable_thoughts": sr.LockInfo(locked=False)}   # fresh install: off
    info = sr.evaluate_locks({"stream_thinking_logs": True})["enable_thoughts"]
    assert info.locked and info.value is True and info.allowed == (True,) and "Stream thinking" in info.reason
    assert not sr.evaluate_locks({"stream_thinking_logs": "0"})["enable_thoughts"].locked


@pytest.mark.skipif(not HAS_QT, reason="needs PySide6 (other_settings)")
@pytest.mark.parametrize("mode", ["text", "vision", "image", "video", "audio", "refinement", "refine",
                                  " VIDEO ", "bogus", ""])
def test_output_mode_flags_match_the_desktop_setter(mode):
    import types

    owner = types.SimpleNamespace(config={})
    with _restore_env():           # importing the desktop dialogs sets Qt / log env defaults
        import other_settings

        other_settings._set_output_mode(owner, mode)
        env = {k: os.environ[k] for k in sr.output_mode_flags(mode).env}
    flags = sr.output_mode_flags(mode)
    assert {k: getattr(owner, k) for k in flags.var_values} == dict(flags.var_values)
    assert owner.config == dict(flags.config_values) and env == dict(flags.env)
    config = {"kept": 1}
    assert sr.apply_output_mode(config, mode) == dict(flags.env)
    assert config == {"kept": 1, **flags.config_values}
    assert sr.current_output_mode(config) == flags.mode


def test_output_mode_sub_settings_and_current_mode():
    shown = {mode: sr.output_mode_sub_settings(mode) for mode in sr.OUTPUT_MODES}
    assert [m for m, v in shown.items() if v["image"]] == ["image"]
    assert [m for m, v in shown.items() if v["video"]] == ["video"]
    assert [m for m, v in shown.items() if v["vision_request"]] == ["vision", "image"]
    assert [m for m, v in shown.items() if v["vision_only"]] == ["vision"]
    # legacy flags win over a stale 'vision' (RunEnvMixin._get_output_mode)
    assert sr.current_output_mode({"output_mode": "vision", "enable_video_output_mode": True}) == "video"
    assert sr.current_output_mode({}) == "text"
    assert sr.evaluate("output_mode:vision_request", {"output_mode": "image"}) is True
    assert sr.evaluate("output_mode:vision", {"output_mode": "image"}) is False


GLOSSARY_MODES = ("off", "off_fuzzy_automap", "off_no_automap", "no_glossary", "minimal", "balanced", "full",
                  "single_pass", "weird_mode")


def test_glossary_mode_display_mapping_and_flags():
    assert sr.glossary_mode_from_display("Manual Glossary Only") == "off_no_automap"
    assert sr.glossary_mode_from_display("Off (No Auto-Mapping)") == "off_no_automap"
    assert sr.glossary_mode_from_display("Weird Mode") == "weird_mode"
    assert [m for m in GLOSSARY_MODES if sr.glossary_mode_extracts(m)] == [
        "off_fuzzy_automap", "minimal", "balanced", "full", "single_pass", "weird_mode"]
    assert [m for m in GLOSSARY_MODES if sr.glossary_mode_targeted_extraction(m)] == ["minimal"]
    assert sr.glossary_mode_label("off_no_automap") == "Manual Glossary Only"
    states = {m: sr.glossary_mode_toggle_states(m) for m in GLOSSARY_MODES}
    assert states["no_glossary"] == {"append_glossary": (True, False), "append_glossary_auto_load": (True, False),
                                     "fuzzy_auto_mapping": (True, False)}
    assert states["off_fuzzy_automap"]["fuzzy_auto_mapping"] == (True, True)
    assert states["weird_mode"]["append_glossary_auto_load"] == (False, None)   # unknown modes unlock auto-map
    assert states["minimal"]["append_glossary_auto_load"] == (True, False)
    assert states["balanced"] == {"append_glossary": (True, True), "append_glossary_auto_load": (True, True),
                                  "fuzzy_auto_mapping": (True, False)}


@pytest.mark.parametrize("config, mode", [({}, "off"), ({"enable_auto_glossary": True}, "minimal"),
                                           ({"auto_glossary_mode": "FULL"}, "full"),
                                           ({"auto_glossary_mode": "bogus"}, "off"),
                                           ({"auto_glossary_mode": "single_pass"}, "single_pass")])
def test_glossary_manager_mode_follows_the_start_up_shortcut(config, mode):
    assert sr.glossary_manager_mode(config) == mode


@pytest.mark.skipif(not HAS_QT, reason="needs PySide6 (GlossaryManager_GUI)")
@pytest.mark.parametrize("label", ["Off", "Off (Fuzzy Mapping)", "Manual Glossary Only", "No Glossary", "Minimal",
                                   "Balanced", "Full", "Single Pass", "Weird Mode"])
def test_glossary_mode_locks_match_the_glossary_manager_pass(label):
    """glossary_mode_locks(mode) == the toggles the desktop lock pass leaves (checked + locked)."""
    with _restore_env():           # importing the desktop dialogs sets Qt / log env defaults
        _check_glossary_mode_locks(label)


def _check_glossary_mode_locks(label):
    import textwrap
    import types

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QLabel, QSlider, QTextEdit, QWidget

    import GlossaryManager_GUI as gm

    app = QApplication.instance() or QApplication([])
    text = (SRC / "GlossaryManager_GUI.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    tree = ast.parse(text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GlossaryManagerMixin")
    outer = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_setup_auto_glossary_tab")
    node = next(n for n in ast.walk(outer) if isinstance(n, ast.FunctionDef) and n.name == "update_auto_glossary_state")
    source = textwrap.dedent("\n".join(text.split("\n")[node.lineno - 1:node.end_lineno]))
    ns = dict(vars(gm))
    exec("def _factory(self, settings_label_frame, extraction_grid):\n" + textwrap.indent(source, "    ")
         + "\n    return update_auto_glossary_state\n", ns)
    toggles = (("append_glossary_checkbox", "append_glossary"),
               ("append_glossary_auto_load_checkbox", "append_glossary_auto_load"),
               ("fuzzy_auto_mapping_checkbox", "fuzzy_auto_mapping"))
    for start in (False, True):
        owner = types.SimpleNamespace(config={}, append_log=lambda m: None,
                                      _glossary_editor_input_sources=lambda: [])
        holder = QWidget()
        owner.auto_glossary_mode_combo = QComboBox(holder)
        owner.auto_glossary_mode_combo.addItem(label)
        owner.auto_prompt_text = QTextEdit(holder)
        for attr, key in toggles:
            cb = QCheckBox(key, QWidget(holder))
            cb.setChecked(start)
            setattr(owner, attr, cb)
        slider_parent = QWidget(holder)
        owner.fuzzy_mapping_slider = QSlider(slider_parent)
        QLabel("Similarity", slider_parent)
        frame = QWidget(holder)
        grid = types.SimpleNamespace(count=lambda: 0)
        ns["_factory"](owner, frame, grid)()
        mode = sr.glossary_mode_from_display(label)
        locks = sr.glossary_mode_locks(mode)
        for attr, key in toggles:
            cb = getattr(owner, attr)
            info = locks[key]
            assert bool(getattr(cb, "_mode_locked", False)) == info.locked, (label, key)
            assert cb.isChecked() == (info.value if info.locked else start), (label, key)
        assert (not owner.fuzzy_mapping_slider.isEnabled()) == locks["fuzzy_auto_mapping_threshold"].locked
        assert owner.auto_prompt_text.isEnabled() == sr.glossary_mode_extracts(mode)
        assert frame.isEnabled() == sr.glossary_mode_targeted_extraction(mode)
        holder.deleteLater()
    app.processEvents()


def test_apply_glossary_mode_locks_and_apply_change():
    config = {"auto_glossary_mode": "no_glossary", "append_glossary": True, "fuzzy_auto_mapping": True}
    assert sr.apply_glossary_mode_locks(config) == {"append_glossary": False, "append_glossary_auto_load": False,
                                                    "fuzzy_auto_mapping": False}
    changed, env = sr.apply_change({}, "stream_thinking_logs", True)
    assert changed == {"stream_thinking_logs": True, "enable_thoughts": True}
    assert env == {"STREAM_THINKING_LOGS": "1", "ENABLE_THOUGHTS": "1"}
    changed, env = sr.apply_change({"output_mode": "text"}, "output_mode", "refine")
    assert changed["output_mode"] == "refinement" and changed["enable_refinement_output_mode"] is True
    assert env["OUTPUT_MODE"] == "refinement"
    changed, env = sr.apply_change({}, "auto_glossary_mode", "minimal")
    assert changed["auto_glossary_mode"] == "minimal" and changed["enable_auto_glossary"] is True and env == {}
    assert sr.apply_change({"x": 1}, "x", 2) == ({"x": 2}, {})


def test_evaluate_rule_ids_and_the_schema_overlay():
    import settings_schema as ss

    locks = set(sr.lock_keys())
    assert {"enable_thoughts", "append_glossary", "append_glossary_auto_load", "fuzzy_auto_mapping",
            "fuzzy_auto_mapping_threshold", "batching_mode", "translation_temperature"} <= locks
    for key, rule in ss.LOCKED_IF.items():
        assert ss.spec(key).locked_if == rule == sr.lock_rule_id(key) and key in locks
    for key, rule in ss.VISIBLE_IF.items():
        assert ss.spec(key).visible_if == rule and rule in sr.visibility_rules(), key
    assert not set(ss.LOCKED_IF) - locks and set(sr.visibility_rules()) == set(ss.VISIBLE_IF.values())
    config = {"stream_thinking_logs": True, "auto_glossary_mode": "minimal", "enable_gpt_thinking": False}
    snapshot = copy.deepcopy(config)
    assert ss.evaluate_rule("lock:enable_thoughts", config).startswith("Stream thinking")
    assert sr.evaluate("lock:translation_temperature", config) == ""
    assert sr.evaluate("glossary:targeted_extraction", config) is True
    assert sr.evaluate("thinking:gpt", config) is False and sr.evaluate("thinking:gemini", {}) is True
    assert sr.evaluate("thinking:anthropic_budget", {"enable_anthropic_thinking": True}) is True
    assert sr.evaluate("thinking:anthropic_budget", {"enable_anthropic_thinking": True,
                                                     "anthropic_force_adaptive": True}) is False
    assert sr.evaluate("glossary:append_prompt", {"append_glossary": False}) is False
    assert config == snapshot
    for bad in ("lock:model", "nope"):
        with pytest.raises(KeyError):
            sr.evaluate(bad, {})
    for rule in set(ss.VISIBLE_IF.values()) | set(ss.LOCKED_IF.values()):
        for cfg in ({}, {"output_mode": "image", "auto_glossary_mode": "off_fuzzy_automap"}):
            assert isinstance(ss.evaluate_rule(rule, cfg), (bool, str))
