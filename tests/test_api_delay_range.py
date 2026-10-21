"""API call delay ranges (2026-10-21): one field takes a delay ("5") or a range ("5-7").

* parsing / formatting (any dash, a dangling dash, a max below the min, decimals, spaces);
* storage: the min stays the numeric setting (``delay``, the QA dialog's
  ``ai_truncation_api_call_delay``, a key's ``api_call_delay``), the max goes to ``<key>_max``
  ('' = fixed); a config without ``delay`` (a new user) gets 3-7, a saved delay without a max is fixed;
* the env export (SEND_INTERVAL_SECONDS / SEND_INTERVAL_MAX_SECONDS) and the per-call random draw;
* the QA dialog's -1 = the global delay and its range; per-key ranges.
"""
import os
import random
import sys
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import settings_rules as sr  # noqa: E402


# --------------------------------------------------------------------------- parsing / formatting
@pytest.mark.parametrize("text,expected", [
    ("5", (5.0, 5.0)),
    ("5-7", (5.0, 7.0)),
    ("5 - 7", (5.0, 7.0)),
    ("5–7", (5.0, 7.0)),          # en dash
    ("5 — 7", (5.0, 7.0)),        # em dash
    ("5-", (5.0, 5.0)),                # a dangling dash is ignored
    ("5–", (5.0, 5.0)),
    ("5 —", (5.0, 5.0)),
    ("7-5", (7.0, 7.0)),               # the max never below the min
    ("5-5", (5.0, 5.0)),
    ("0.5-1.5", (0.5, 1.5)),
    (" 3-7 ", (3.0, 7.0)),
    (".5-2", (0.5, 2.0)),
    (4, (4.0, 4.0)),
    (2.5, (2.5, 2.5)),
])
def test_parse_delay_range(text, expected):
    assert sr.parse_delay_range(text) == expected


@pytest.mark.parametrize("text", ["", " ", "-1", "abc", "5-x", "-5", "5--7", "1-2-3", None, True, float("nan")])
def test_parse_delay_range_rejects(text):
    assert sr.parse_delay_range(text) is None


@pytest.mark.parametrize("text,shown", [("5", "5"), ("5-7", "5-7"), ("7-5", "7"), ("5-5", "5"), ("5-", "5"),
                                        ("0.5-1.5", "0.5-1.5"), ("5 – 7.5", "5-7.5")])
def test_the_field_shows_the_normalised_range(text, shown):
    assert sr.format_delay_range(*sr.parse_delay_range(text)) == shown


# --------------------------------------------------------------------------- storage and defaults
def test_a_new_user_gets_3_to_7():
    assert sr.delay_range_for({}) == (3.0, 7.0)
    assert sr.delay_range_text({}) == "3-7"
    assert sr.DEFAULT_DELAY == 3.0 and sr.DEFAULT_DELAY_MAX == 7.0


@pytest.mark.parametrize("config,expected", [
    ({"delay": 5.0}, "5"),                       # the owner's config: a saved delay without a max stays fixed
    ({"delay": 10}, "10"),
    ({"delay": 5.0, "delay_max": ""}, "5"),
    ({"delay": 5.0, "delay_max": 7.0}, "5-7"),
    ({"delay": 5.0, "delay_max": 4.0}, "5"),     # a max below the min reads as the min
    ({"delay": 5.0, "delay_max": "x"}, "5"),
    ({"delay_max": ""}, "3"),                    # no delay, the max cleared: fixed at the default min
])
def test_saved_delays(config, expected):
    assert sr.delay_range_text(config) == expected


def test_settings_map_converters_round_trip():
    for text, low, high in (("3-7", 3.0, 7.0), ("5", 5.0, ""), ("7-5", 7.0, ""), ("0.5-1.5", 0.5, 1.5)):
        assert sr.delay_range_min(text) == low
        assert sr.delay_range_max(text) == high
        config = {"delay": low, "delay_max": high}
        assert sr.parse_delay_range(sr.delay_range_text(config)) == (low, high or low)
    # text that is no delay keeps the old safe_float answers (the save check rejects it first)
    assert sr.delay_range_min("abc") == sr.DEFAULT_DELAY and sr.delay_range_max("abc") == ""
    assert sr.delay_range_min(" 7 ") == 7.0


def test_the_change_rules_keep_the_max_at_or_above_the_min():
    config = {"delay": 5.0}
    sr.apply_change(config, "delay", "6-8")
    assert config == {"delay": 6.0, "delay_max": 8.0}
    sr.apply_change(config, "delay", 4)            # a single number clears the max
    assert config == {"delay": 4.0, "delay_max": ""}
    sr.apply_change(config, "delay", "9–7")
    assert config == {"delay": 9.0, "delay_max": ""}
    sr.apply_change(config, "delay_max", 12)
    assert config == {"delay": 9.0, "delay_max": 12.0}
    sr.apply_change(config, "delay_max", 2)        # below the min: no max
    assert config == {"delay": 9.0, "delay_max": ""}


# --------------------------------------------------------------------------- env export
def test_delay_env():
    assert sr.delay_env("3-7") == ("3.0", "7.0")
    assert sr.delay_env("5") == ("5", "")                 # a plain number exports as typed
    assert sr.delay_env("5.0") == ("5.0", "")
    assert sr.delay_env("7-5") == ("7.0", "")
    assert sr.delay_env("abc") == ("abc", "")       # passes through as before
    owner = types.SimpleNamespace(delay_entry=types.SimpleNamespace(text=lambda: "2-4"))
    assert sr.owner_delay_env(owner) == ("2.0", "4.0")
    assert sr.owner_delay_env(types.SimpleNamespace(), "2.0") == ("2.0", "")


def test_headless_owner_exports_the_range(tmp_path, monkeypatch):
    import app_paths
    from headless_owner import HeadlessOwner

    monkeypatch.setattr(app_paths, "CONFIG_FILE", str(tmp_path / "config.json"))
    host = types.SimpleNamespace(log=lambda _m: None)
    for config, low, high, shown in (({"delay": 5.0}, "5", "", "5"),
                                     ({"delay": 2.0, "delay_max": 6.0}, "2.0", "6.0", "2-6")):
        owner = HeadlessOwner(dict(config), host=host)
        assert owner.delay_entry.text() == shown
        env = owner._get_environment_variables(str(tmp_path / "book.epub"), owner.api_key_entry.text())
        assert (env["SEND_INTERVAL_SECONDS"], env["SEND_INTERVAL_MAX_SECONDS"]) == (low, high)


# --------------------------------------------------------------------------- the per-call draw
def test_draw_api_delay_is_uniform_in_the_range():
    import unified_api_client as api

    rng = random.Random(7)
    draws = [api.draw_api_delay(5, 7, rng) for _ in range(500)]
    assert all(5 <= d <= 7 for d in draws)
    assert min(draws) < 5.2 and max(draws) > 6.8
    assert api.draw_api_delay(5, None) == 5.0
    assert api.draw_api_delay(5, 5) == 5.0
    assert api.draw_api_delay(5, 3) == 5.0


@pytest.mark.parametrize("raw,expected", [(None, None), ("", None), (" ", None), ("x", None), ("-1", None),
                                          ("7", 7.0), ("7.5", 7.5)])
def test_send_interval_max_from_env(monkeypatch, raw, expected):
    import unified_api_client as api

    if raw is None:
        monkeypatch.delenv("SEND_INTERVAL_MAX_SECONDS", raising=False)
    else:
        monkeypatch.setenv("SEND_INTERVAL_MAX_SECONDS", raw)
    assert api._send_interval_max_from_env() == expected


def _stagger(monkeypatch, *, low, high, since_finished=1.0, tls=None, qa=None):
    """One _apply_api_call_stagger call on a fake clock; returns the logged wait."""
    import unified_api_client as api
    from unified_api_client import UnifiedClient

    clock = {"now": 1000.0}
    monkeypatch.setattr(api.time, "time", lambda: clock["now"])
    monkeypatch.setattr(api.time, "sleep", lambda s: clock.__setitem__("now", clock["now"] + s))
    monkeypatch.setenv("SEND_INTERVAL_SECONDS", str(low))
    if high is None:
        monkeypatch.delenv("SEND_INTERVAL_MAX_SECONDS", raising=False)
    else:
        monkeypatch.setenv("SEND_INTERVAL_MAX_SECONDS", str(high))
    monkeypatch.delenv("GRACEFUL_STOP", raising=False)
    monkeypatch.setattr(UnifiedClient, "_last_api_call_start_by_scope",
                        {"test-scope": clock["now"] - since_finished}, raising=False)
    client = UnifiedClient.__new__(UnifiedClient)
    client.model = "ollamapull/test"
    client._get_thread_local_client = lambda: tls or types.SimpleNamespace()
    client._get_api_stagger_scope = lambda: "context:qa_truncation" if qa is not None else "test-scope"
    if qa is not None:
        UnifiedClient._last_api_call_start_by_scope["context:qa_truncation"] = clock["now"] - since_finished
        client._qa_truncation_delay_override, client._qa_truncation_delay_max_override = qa
    client._should_show_api_lifecycle_logs = lambda: True
    client._should_abort_retry = lambda: False
    client._is_instance_cancel_requested = lambda: False
    client._get_actual_provider = lambda: "ollamapull"
    client._get_thinking_status_label = lambda: ""
    logs = []
    client._debug_log = logs.append
    start = clock["now"]
    client._apply_api_call_stagger()
    return clock["now"] - start, logs


def test_each_call_waits_a_random_time_in_the_range_from_when_the_last_call_finished(monkeypatch):
    random.seed(3)
    waits = []
    for _ in range(40):
        waited, logs = _stagger(monkeypatch, low=5, high=7, since_finished=1.0)
        assert 4.0 <= waited <= 6.0 + 1e-9                         # drawn in [5, 7], 1 s already passed
        assert any(f"Sending API call in {waited:.1f}s" in line for line in logs), logs
        waits.append(waited)
    assert len({round(w, 3) for w in waits}) > 10                   # drawn per call, not once


def test_a_fixed_delay_is_unchanged(monkeypatch):
    waited, _logs = _stagger(monkeypatch, low=5, high=None, since_finished=1.0)
    assert waited == pytest.approx(5.0)        # 4 s left, snapped to the 5 s grid as before
    waited, _logs = _stagger(monkeypatch, low=5, high="", since_finished=1.0)
    assert waited == pytest.approx(5.0)


def test_a_stale_gap_still_fires_at_once(monkeypatch):
    waited, _logs = _stagger(monkeypatch, low=5, high=7, since_finished=60.0)
    assert waited == 0


def test_a_per_key_range_and_a_fixed_per_key_delay(monkeypatch):
    random.seed(5)
    tls = types.SimpleNamespace(active_api_delay_override=2.0, active_api_delay_max=4.0)
    for _ in range(20):
        waited, _logs = _stagger(monkeypatch, low=50, high=70, since_finished=0.5, tls=tls)
        assert 1.5 <= waited <= 3.5 + 1e-9      # the key's 2-4, not the global 50-70
    tls = types.SimpleNamespace(active_api_delay_override=2.0, active_api_delay_max=None)
    waited, _logs = _stagger(monkeypatch, low=50, high=70, since_finished=0.5, tls=tls)
    assert waited == pytest.approx(2.0)        # 1.5 s left, snapped to the 2 s grid


def test_the_qa_delay_range(monkeypatch):
    random.seed(9)
    for _ in range(20):
        waited, _logs = _stagger(monkeypatch, low=50, high=70, since_finished=0.5, qa=(1.0, 3.0))
        assert 0.5 <= waited <= 2.5 + 1e-9
    waited, _logs = _stagger(monkeypatch, low=50, high=70, since_finished=0.5, qa=(1.0, None))
    assert waited == pytest.approx(0.5)


# --------------------------------------------------------------------------- QA dialog field (-1 = global)
@pytest.mark.parametrize("text,values", [
    ("-1", {"ai_truncation_api_call_delay": -1.0, "ai_truncation_api_call_delay_max": ""}),
    ("", {"ai_truncation_api_call_delay": -1.0, "ai_truncation_api_call_delay_max": ""}),
    ("-0.5", {"ai_truncation_api_call_delay": -1.0, "ai_truncation_api_call_delay_max": ""}),
    ("2", {"ai_truncation_api_call_delay": 2.0, "ai_truncation_api_call_delay_max": ""}),
    ("2-4", {"ai_truncation_api_call_delay": 2.0, "ai_truncation_api_call_delay_max": 4.0}),
    ("4-2", {"ai_truncation_api_call_delay": 4.0, "ai_truncation_api_call_delay_max": ""}),
])
def test_qa_field_values(text, values):
    key = "ai_truncation_api_call_delay"
    assert sr.delay_field_values(text, key, inherit=-1.0) == values
    shown = sr.delay_field_text(values, key, inherit=-1.0)
    assert sr.delay_field_values(shown, key, inherit=-1.0) == values


def test_qa_field_text():
    key = "ai_truncation_api_call_delay"
    assert sr.delay_field_text({}, key, inherit=-1.0) == "-1"
    assert sr.delay_field_text({key: -1.0, key + "_max": 9}, key, inherit=-1.0) == "-1"
    assert sr.delay_field_text({key: 0.5}, key, inherit=-1.0) == "0.5"
    assert sr.delay_field_text({key: 1, key + "_max": 3}, key, inherit=-1.0) == "1-3"
    assert sr.delay_field_values("abc", key, inherit=-1.0) is None


def test_the_qa_check_inherits_the_global_range():
    """-1 = the global delay AND its range (a new user's 3-7); a dedicated range is its own
    (the resolution scan_html_folder's AI truncation check runs before run_ai_truncation_check)."""
    import inspect

    import scan_html_folder as shf

    source = inspect.getsource(shf)
    assert "_ai_delay, _ai_delay_max = delay_range_for(_ai_config)" in source
    for qa, config, expected in (({"ai_truncation_api_call_delay": -1.0}, {}, (3.0, 7.0)),
                                 ({"ai_truncation_api_call_delay": -1.0}, {"delay": 5.0}, (5.0, 5.0)),
                                 ({"ai_truncation_api_call_delay": 1.0, "ai_truncation_api_call_delay_max": 2.0},
                                  {"delay": 5.0, "delay_max": 9.0}, (1.0, 2.0)),
                                 ({"ai_truncation_api_call_delay": 1.0}, {"delay": 5.0, "delay_max": 9.0},
                                  (1.0, 1.0))):
        low = float(qa.get("ai_truncation_api_call_delay", -1.0))
        got = sr.delay_range_for(config) if low < 0 else sr.delay_range_for(
            qa, "ai_truncation_api_call_delay", default_max=None)
        assert got == expected
    assert "api_call_delay_max" in shf.run_ai_truncation_check.__code__.co_varnames


def test_the_qa_env_override_restores_the_global_range(monkeypatch):
    import scan_html_folder as shf

    monkeypatch.setenv("SEND_INTERVAL_SECONDS", "5")
    monkeypatch.setenv("SEND_INTERVAL_MAX_SECONDS", "7")
    with shf._qa_api_overrides_env(delay_override=1.0, delay_max_override=2.0):
        assert (os.environ["SEND_INTERVAL_SECONDS"], os.environ["SEND_INTERVAL_MAX_SECONDS"]) == ("1.0", "2.0")
    with shf._qa_api_overrides_env(delay_override=1.0):
        assert os.environ["SEND_INTERVAL_MAX_SECONDS"] == ""
    assert (os.environ["SEND_INTERVAL_SECONDS"], os.environ["SEND_INTERVAL_MAX_SECONDS"]) == ("5", "7")


# --------------------------------------------------------------------------- per-key ranges
def test_per_key_ranges_round_trip():
    from multi_api_key_manager import APIKeyEntry

    entry = APIKeyEntry("sk-test", "gpt-6", api_call_delay=2.0, api_call_delay_max=5.0)
    data = entry.to_dict()
    assert data["api_call_delay"] == 2.0 and data["api_call_delay_max"] == 5.0
    assert str(sr.key_delay(APIKeyEntry.from_dict(data))) == "2-5"
    fixed = APIKeyEntry("sk-test", "gpt-6", api_call_delay=2.0).to_dict()
    assert "api_call_delay_max" not in fixed               # configs without ranges stay unchanged
    assert APIKeyEntry("sk", "m", api_call_delay=3.0, api_call_delay_max=1.0).api_call_delay_max == ""
    assert APIKeyEntry("sk", "m", api_call_delay=0.0, api_call_delay_max=9.0).api_call_delay_max == ""


def test_store_key_delay_and_the_dialog_value():
    entry = {"api_call_delay": 0.0}
    sr.store_key_delay(entry, sr.DelayRange(*sr.parse_delay_range("2-6")))
    assert entry == {"api_call_delay": 2.0, "api_call_delay_max": 6.0}
    assert str(sr.key_delay(entry)) == "2-6" and f"{sr.key_delay(entry)}" == "2-6"
    sr.store_key_delay(entry, 3.0)                         # a plain number: fixed
    assert entry == {"api_call_delay": 3.0, "api_call_delay_max": ""}
    sr.store_key_delay(entry, 0.0)                         # 0 = the global delay
    assert entry == {"api_call_delay": 0.0, "api_call_delay_max": ""}
    plain = {"api_call_delay": 0.0}
    sr.store_key_delay(plain, 4.0)
    assert plain == {"api_call_delay": 4.0}                # no max key added for a fixed delay
    value = sr.DelayRange(2.0, 6.0)
    assert value > 0 and float(value) == 2.0 and value.maximum == 6.0


def test_the_multi_key_delay_dialog_takes_a_range(monkeypatch):
    import multi_api_key_manager as mkm

    if mkm.QInputDialog is object:  # pragma: no cover - a build without PySide6
        pytest.skip("PySide6 not installed")
    shown = {}

    def get_text(_parent, _title, label, text=""):
        shown.update(label=label, text=text)
        return shown["answer"], True

    monkeypatch.setattr(mkm.QInputDialog, "getText", staticmethod(get_text))
    monkeypatch.setattr(mkm.QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    dialog = mkm.MultiAPIKeyDialog._show_per_key_setting_dialog
    shown["answer"] = "4 – 6"
    value, ok = dialog(None, "api_call_delay", sr.DelayRange(2.0, 5.0))
    assert ok and shown["text"] == "2-5" and "range" in shown["label"]
    assert (float(value), value.maximum) == (4.0, 6.0)
    entry = {"api_call_delay": 2.0, "api_call_delay_max": 5.0}
    sr.store_key_delay(entry, value)
    assert entry == {"api_call_delay": 4.0, "api_call_delay_max": 6.0}
    shown["answer"] = ""                                   # blank = 0 = the global delay
    value, ok = dialog(None, "api_call_delay", 3.0)
    assert ok and float(value) == 0.0 and shown["text"] == "3"
    shown["answer"] = "soon"
    _value, ok = dialog(None, "api_call_delay", 3.0)
    assert not ok


def test_validate_entry_keeps_the_per_key_range():
    import key_pool_service

    data, error = key_pool_service.validate_entry(
        {"api_key": "k", "model": "gpt-6", "api_call_delay": 2.0, "api_call_delay_max": 5.0}, pool="fallback")
    assert error is None and (data["api_call_delay"], data["api_call_delay_max"]) == (2.0, 5.0)
    data, error = key_pool_service.validate_entry(
        {"api_key": "k", "model": "gpt-6", "api_call_delay": 0.0, "api_call_delay_max": 5.0}, pool="main")
    assert error is None and "api_call_delay_max" not in data


def test_runtime_overrides_carry_the_per_key_max():
    from unified_api_client import UnifiedClient

    client = UnifiedClient.__new__(UnifiedClient)
    tls = types.SimpleNamespace()
    client._apply_key_runtime_overrides(key_data={"api_call_delay": 2.0, "api_call_delay_max": 5.0}, tls=tls)
    assert (tls.active_api_delay_override, tls.active_api_delay_max) == (2.0, 5.0)
    client._apply_key_runtime_overrides(key_data={"api_call_delay": 2.0}, tls=tls)
    assert (tls.active_api_delay_override, tls.active_api_delay_max) == (2.0, None)
