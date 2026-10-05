"""model_catalog_core: the Model Manager / catalog data steps shared by desktop and mobile (U4).

TranslatorGUI's tombstone, poll-marker, provider-refresh, save-order and custom-prefix code
moved into ``src/model_catalog_core.py``; the desktop methods keep their widget work and
call it. Tier D (tests/parity/test_parity_tiers.py, ``moved_functions.REWIRED``) fuzzes the
desktop methods against the frozen oracle; this file checks the module's own contract:

* GUI-free import (PySide6 blocked, no translator_gui / dpi_setup), Python 3.10 syntax;
* tombstones: explicit re-add, add-to-saved, rollback when the save fails, save order;
* poll markers and the provider-refresh helpers;
* ``apply_provider_refresh`` (the mobile composition) takes the same steps as the desktop
  ``_apply_provider_model_catalog_refresh`` on the same config and result;
* ``validate_custom_prefix_routes`` equals the frozen desktop table validator on random rows.

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \\
        tests/test_model_catalog_core.py
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import random
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TESTS = REPO_ROOT / "tests"
for _p in (str(TESTS), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import model_catalog_core as mcc  # noqa: E402
import model_options  # noqa: E402

HAS_QT = importlib.util.find_spec("PySide6") is not None


# ---------------------------------------------------------------------------
# hygiene
# ---------------------------------------------------------------------------

def test_module_is_gui_free_and_python_310():
    source = (SRC / "model_catalog_core.py").read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 10))
    code = (
        "import sys; sys.modules['PySide6'] = None; sys.path.insert(0, %r); "
        "import model_catalog_core; "
        "bad = [m for m in ('translator_gui', 'dpi_setup', 'PySide6.QtWidgets') if m in sys.modules]; "
        "print(bad)" % str(SRC)
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


def test_line_endings_follow_the_checkout():
    data = (SRC / "model_catalog_core.py").read_bytes()
    crlf = data.count(b"\r\n")
    assert crlf in (0, data.count(b"\n")), "mixed line endings"


# ---------------------------------------------------------------------------
# tombstones
# ---------------------------------------------------------------------------

def test_restore_clears_exact_tombstones_and_saves():
    config = {"custom_model_list": ["provider/kept"],
              "model_manager_removed_models": ["provider/deleted", "Gemini-3.7-Flash", "other"]}
    saved = []
    remaining = mcc.restore_removed_models(config, ["gemini-3.7-flash", "gemini-3.7-flash", " "],
                                           add_to_saved=True, save=lambda: saved.append(1) or True)
    assert remaining == ["provider/deleted", "other"]
    assert config["model_manager_removed_models"] is remaining
    assert config["custom_model_list"] == ["provider/kept", "gemini-3.7-flash"]
    assert saved == [1]


@pytest.mark.parametrize("values, config", [
    ([], {"model_manager_removed_models": ["a"]}),
    (["", None], {"model_manager_removed_models": ["a"]}),
    (["a"], None),
    (["a"], ["not", "a", "dict"]),
    (["b"], {"model_manager_removed_models": ["a"]}),
    (["a"], {"model_manager_removed_models": "a"}),
])
def test_restore_without_a_matching_tombstone_changes_nothing(values, config):
    before = copy.deepcopy(config)
    assert mcc.restore_removed_models(config, values, save=lambda: pytest.fail("saved")) is None
    assert config == before


@pytest.mark.parametrize("failure", [lambda: False, lambda: 1 / 0])
@pytest.mark.parametrize("had_custom", [True, False])
def test_restore_rolls_back_when_the_save_fails(failure, had_custom):
    config = {"model_manager_removed_models": ["Gone"]}
    if had_custom:
        config["custom_model_list"] = ["x"]
    before = copy.deepcopy(config)
    logs = []
    assert mcc.restore_removed_models(config, ["gone"], add_to_saved=True, save=failure, log=logs.append) is False
    assert config == before
    assert logs == [mcc.EXPLICIT_READD_NOT_SAVED]


def test_restore_rollback_pops_a_missing_tombstone_key_like_desktop():
    # old value None -> the key is removed on rollback (desktop's ``old_removed is None`` branch)
    config = {"model_manager_removed_models": None}
    assert mcc.restore_removed_models(config, ["x"], save=lambda: False) is None  # nothing to restore
    config = {"model_manager_removed_models": ["x"]}
    assert mcc.restore_removed_models(config, ["x"], save=lambda: False) is False
    assert config == {"model_manager_removed_models": ["x"]}


def test_save_order_tombstones_removed_models_and_rolls_back():
    config = {"model_manager_removed_models": ["older/deleted"], "model_mousewheel_locked": True}
    assert mcc.save_model_order(config, ["a", "b"], ["a", "b", "C"], wheel_locked=False)
    assert config["custom_model_list"] == ["a", "b"]
    assert config["model_manager_removed_models"] == ["c", "older/deleted"]
    assert config["model_mousewheel_locked"] is False

    config = {"custom_model_list": ["a"]}
    before = copy.deepcopy(config)
    assert mcc.save_model_order(config, ["b"], ["a"], wheel_locked=True, save=lambda: False) is False
    assert config == before
    assert mcc.save_model_order(config, [], ["a"]) is False
    assert config == before


def test_model_order_removed_keys_never_tombstones_a_kept_model():
    keys = mcc.model_order_removed_keys([" Kept "], ["kept", "gone", ""], ["KEPT", "old", "  "])
    assert keys == {"gone", "old"}


def test_config_snapshot_restores_absent_and_present_keys():
    config = {"a": 1}
    snap = mcc.ConfigSnapshot(config, ("a", "b"))
    config.update(a=2, b=3)
    snap.restore(config)
    assert config == {"a": 1}


# ---------------------------------------------------------------------------
# poll markers and refresh helpers
# ---------------------------------------------------------------------------

def test_poll_markers():
    by_provider = mcc.polled_models_by_provider({"xai": ["Grok-4", "grok-4"], 5: None})
    assert by_provider == {"xai": {"grok-4"}, "5": set()}
    keys = mcc.polled_model_keys(by_provider)
    assert isinstance(keys, model_options.PolledModelKeys) and set(keys) == {"grok-4"}
    merged = mcc.merge_polled_provider_models({"xai": {"old"}, "gemini": {"g"}},
                                              {"xai": ["New"], "gemini": ["x"]}, ["xai"])
    assert merged == {"xai": {"new"}, "gemini": {"g"}}
    assert mcc.polled_key_set(merged) == {"new", "g"}
    assert mcc.model_poll_marker("NEW", {"new"}, True) == (True, False, mcc.POLLED_MODEL_TOOLTIP)
    assert mcc.model_poll_marker("other", {"new"}, True) == (False, True, "")
    assert mcc.model_poll_marker("other", {"new"}, False) == (False, False, "")


def test_refresh_helpers():
    models, keys = mcc.confirmed_catalog_models({"a": ["X", " "], "b": None})
    assert models == ["X", " "] and keys == {"x"}
    statuses = {"a": "online (2)", "b": "static fallback (no provider credential)", "c": "static fallback (x)"}
    assert mcc.online_catalog_summary(statuses, {"a": ["1", "2"], "c": ["3"]}) == (["a"], 2)
    assert mcc.catalog_skip_counts(statuses) == (1, 1)
    assert mcc.poll_status_text([], 0, 5, 1, 1) == mcc.NO_ONLINE_CATALOG
    assert mcc.poll_status_text(["b", "a"], 3, 9, 1, 2) == (
        "✓ 3 online · a, b\n9 total incl. fallbacks\n1 need credentials · 2 unavailable")
    assert mcc.auto_poll_message("a", statuses, {"a": ["New", "old"]}, {"old"}) == (
        "✅ Auto-poll complete: a — 2 models · 1 new model found")
    assert mcc.auto_poll_message("zz", statuses, {}, set()) == (
        "⚠️ Auto-poll failed: zz — static fallback (no result)")
    assert mcc.catalog_display_models(
        {"custom_model_list": ["kept"], "model_manager_removed_models": ["Gone"]}, ["gone", "new", "kept"]
    ) == ["kept", "new"]
    assert mcc.manager_poll_models(["kept", "draft-deleted-gone", "custom/mine"], {"kept", "dropped"},
                                   ["tomb"], ["kept", "tomb", "dropped", "fresh"],
                                   explicit_poll=True, confirmed_model_keys={"tomb"}) == [
        "kept", "tomb", "fresh", "draft-deleted-gone", "custom/mine"]


def _desktop_refresh(config, result, combo_models, monkeypatch):
    """Run TranslatorGUI._apply_provider_model_catalog_refresh on a plain gui (no manager)."""
    import translator_gui

    monkeypatch.setattr(translator_gui, "get_current_polled_provider_models", lambda: {"xai": ["grok-old"]})
    shown, logs, saved = [], [], []
    gui = types.SimpleNamespace(
        config=config,
        model_combo=types.SimpleNamespace(count=lambda: len(combo_models),
                                          itemText=lambda i: combo_models[i]),
        _refresh_model_combo_catalog=lambda models: shown.append(list(models)),
        save_config=lambda show_message=False: saved.append(show_message) or True,
        append_log=logs.append,
    )
    gui._ensure_polled_model_marker_state = types.MethodType(
        translator_gui.TranslatorGUI._ensure_polled_model_marker_state, gui)
    gui._restore_removed_model_choices = types.MethodType(
        translator_gui.TranslatorGUI._restore_removed_model_choices, gui)
    translator_gui.TranslatorGUI._apply_provider_model_catalog_refresh(gui, result)
    return gui, shown, logs, saved


@pytest.mark.skipif(not HAS_QT, reason="the desktop side needs PySide6 (translator_gui)")
@pytest.mark.parametrize("seed", range(40))
def test_apply_provider_refresh_takes_the_desktop_steps(seed, monkeypatch):
    rng = random.Random(seed)
    pool = ["gpt-4o", "Gemini-3.7-Flash", "provider/kept", "provider/deleted", "grok-4", "new/model"]
    config = {"custom_model_list": rng.sample(pool, rng.randint(0, 3)),
              "model_manager_removed_models": rng.sample(pool, rng.randint(0, 3))}
    provider_models = {p: rng.sample(pool, rng.randint(0, 3)) for p in rng.sample(["xai", "gemini"], 2)}
    statuses = {p: rng.choice(["online (1)", "static fallback (no provider credential)", "error"])
                for p in provider_models}
    result = types.SimpleNamespace(models=rng.sample(pool, rng.randint(1, len(pool))), statuses=statuses,
                                   provider_models=provider_models,
                                   requested_provider=rng.choice([None, "xai", "gemini"]),
                                   restore_removed_models=rng.random() < 0.5)
    combo = rng.sample(pool, 3)

    desktop_config = copy.deepcopy(config)
    gui, shown, logs, saved = _desktop_refresh(desktop_config, result, combo, monkeypatch)

    mobile_config = copy.deepcopy(config)
    mobile_saved, mobile_logs = [], []
    refresh = mcc.apply_provider_refresh(
        mobile_config, result, polled_by_provider={"xai": {"grok-old"}}, previous_models=combo,
        save=lambda: mobile_saved.append(False) or True, log=mobile_logs.append)
    assert refresh.applied
    assert mobile_config == desktop_config
    assert [refresh.display_models] == shown
    assert refresh.polled_by_provider == gui._polled_online_models_by_provider
    assert set(refresh.polled_model_keys) == set(gui._polled_online_model_ids)
    assert len(mobile_saved) == len(saved)
    assert mobile_logs == []
    auto_poll_lines = [line for line in logs if "Auto-poll" in line]
    assert auto_poll_lines == ([refresh.auto_poll_message] if refresh.auto_poll_message else [])


def test_apply_provider_refresh_without_models_changes_nothing():
    config = {"model_manager_removed_models": ["x"]}
    refresh = mcc.apply_provider_refresh(config, types.SimpleNamespace(models=[], restore_removed_models=True))
    assert not refresh.applied and config == {"model_manager_removed_models": ["x"]}


# ---------------------------------------------------------------------------
# custom prefixes
# ---------------------------------------------------------------------------

def test_validate_custom_prefix_routes_messages():
    ok, err = mcc.validate_custom_prefix_routes([
        {"prefix": "/lan", "routing": "http://192.168.1.2:11434/v1/", "endpoint_type": "{base_url}/chat/completions"},
        {"prefix": "", "routing": ""},
        {"prefix": "img", "base_url": "https://x.test", "endpoint_type": "/images/generations"},
        {"prefix": "ocr\\", "routing": "https://ocr.test//"},
    ])
    assert err is None
    assert ok == [
        {"prefix": "lan/", "routing": "http://192.168.1.2:11434/v1", "endpoint_type": "/chat/completions"},
        {"prefix": "img/", "routing": "https://x.test", "endpoint_type": "/images/generations"},
        {"prefix": "ocr/", "routing": "https://ocr.test", "endpoint_type": "/chat/completions"},
    ]
    # the table offers endpoint paths; legacy preset names are not accepted there (desktop rule)
    assert mcc.validate_custom_prefix_routes(
        [{"prefix": "a", "routing": "http://x", "endpoint_type": "openai_chat"}])[1][0] == "Invalid Endpoint Type"
    assert mcc.validate_custom_prefix_routes([{"prefix": "a", "routing": "ftp://x"}]) == (
        None, ("Invalid Base URL", "Base URL on row 1 must start with http:// or https://."))
    assert mcc.validate_custom_prefix_routes([{"prefix": "a b", "routing": "http://x"}])[1][0] == "Invalid Prefix"
    assert mcc.validate_custom_prefix_routes([{"prefix": "a", "routing": ""}])[1][0] == "Incomplete Prefix Route"
    assert mcc.validate_custom_prefix_routes(
        [{"prefix": "a", "routing": "http://x"}, {"prefix": "A/", "routing": "http://y"}]
    ) == (None, ("Duplicate Prefix", "'A/' is already listed."))
    assert mcc.validate_custom_prefix_routes([{"prefix": "a", "routing": "http://x", "endpoint_type": "x y"}])[1] == (
        "Invalid Endpoint Type",
        "Endpoint Type on row 1 must be an absolute path like /chat/completions, /v1/ocr, or /v1/custom.")


@pytest.fixture(scope="module")
def legacy_collect():
    """The frozen desktop table validator (tests/parity oracle at the U4 parent commit)."""
    if not HAS_QT:
        pytest.skip("the frozen oracle needs PySide6")
    from parity import fuzz_moved as fm

    sess = fm.session()
    try:
        cls = sess.legacy_class()
    except fm.Unavailable as exc:
        pytest.skip(str(exc))
    fn = fm.resolve_python_mro(cls, "_collect_custom_prefix_routes_from_table")
    if fn is fm.MISSING:
        pytest.skip("oracle lacks _collect_custom_prefix_routes_from_table")
    return cls, fn


def test_validate_custom_prefix_routes_matches_the_frozen_desktop_validator(legacy_collect, monkeypatch):
    from parity import fakes, fuzz_moved as fm
    from PySide6 import QtWidgets

    cls, fn = legacy_collect
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        lambda _parent, title, message: warnings.append((title, message)))
    owner = cls(fakes.CallRecorder())
    rng = random.Random(7)
    for _ in range(400):
        rows = []
        for _row in range(rng.randint(0, 4)):
            rows.append((rng.choice(fm._U4_PREFIXES), rng.choice(fm._U4_BASE_URLS),
                         rng.choice([e for e in fm._U4_ENDPOINTS if e is not None])))
        table = fakes.FakeTable([(fakes.FakeLineEdit(p), fakes.FakeLineEdit(u), fakes.FakeLineEdit(e))
                                 for p, u, e in rows])
        warnings.clear()
        legacy = fn(owner, table, None)
        routes, problem = mcc.validate_custom_prefix_routes(
            [{"prefix": p.strip(), "routing": u.strip(), "endpoint_type": e.strip()} for p, u, e in rows])
        if legacy is None:
            assert routes is None and warnings == [problem], rows
        else:
            assert problem is None and routes == legacy, rows
