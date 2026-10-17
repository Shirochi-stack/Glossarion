"""'Respect provider wait time (Retry-After)' (Other Settings > API Request Retries, default on).

On a 429 every provider path first asks the server's Retry-After (seconds or HTTP-date),
retry-after-ms, a ``retry_after_seconds`` a route already extracted, Gemini's RetryInfo
``retryDelay`` and the "Please try again in ..." hint, and only falls back to its configured /
built-in backoff when there is none. The multi-key rotation rests THAT key for the server's
wait. 'Ignore provider wait time (Retry-After)' (HTTP overrides) and the new toggle are exclusive.

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_respect_retry_after.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import unified_api_client as api
from unified_api_client import UnifiedClient, UnifiedClientError

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture(autouse=True)
def _policy_env(monkeypatch):
    for name in ("RESPECT_RETRY_AFTER", "IGNORE_RETRY_AFTER", "ENABLE_HTTP_TUNING", "INDEFINITE_RATE_LIMIT_RETRY",
                 "GRACEFUL_STOP"):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- header parsing
NOW = datetime(2026, 10, 17, 12, 0, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("value,expected", [
    ("30", 30.0), (" 7 ", 7.0), ("1.5", 1.5), (12, 12.0),
    (format_datetime(NOW + timedelta(seconds=90), usegmt=True), 90.0),
])
def test_retry_after_seconds_and_http_dates(value, expected):
    assert api.parse_retry_after_value(value, now=NOW) == pytest.approx(expected)


@pytest.mark.parametrize("value", [None, "", "   ", "soon", "0", "-5", "nan", "inf",
                                   format_datetime(NOW - timedelta(seconds=5), usegmt=True)])
def test_missing_garbage_or_past_values_give_no_wait(value):
    assert api.parse_retry_after_value(value, now=NOW) is None


def test_retry_after_ms_wins_and_headers_are_case_insensitive():
    assert api.retry_after_from_headers({"Retry-After-Ms": "1500", "Retry-After": "9"}) == 1.5
    assert api.retry_after_from_headers({"RETRY-AFTER": "9"}) == 9.0
    assert api.retry_after_from_headers({}) is None
    assert api.retry_after_from_headers(None) is None
    assert api.retry_after_from_headers({"retry-after-ms": "garbage", "retry-after": "4"}) == 4.0


def test_legacy_parser_keeps_its_int_contract():
    client = UnifiedClient.__new__(UnifiedClient)
    assert client._parse_retry_after("12") == 12
    assert client._parse_retry_after("") == 0
    assert client._parse_retry_after("nonsense") == 0


# --------------------------------------------------------------------------- hints on errors
class FakeSDKRateLimit(Exception):
    """An OpenAI / Anthropic SDK APIStatusError: the httpx response rides on the exception."""

    def __init__(self, headers):
        super().__init__("Error code: 429 - rate limited")
        self.response = SimpleNamespace(headers=headers, status_code=429)


def _wrapped(exc):
    try:
        try:
            raise exc
        except Exception as inner:
            raise UnifiedClientError(f"OpenAI rate limit: {inner}", error_type="rate_limit")
    except UnifiedClientError as outer:
        return outer


def test_sdk_response_headers_are_found_through_the_wrapping_error():
    assert api.server_retry_after_seconds(_wrapped(FakeSDKRateLimit({"retry-after": "21"}))) == 21.0
    assert api.server_retry_after_seconds(_wrapped(FakeSDKRateLimit({"retry-after-ms": "250"}))) == 0.25


def test_gemini_retry_info_and_try_again_hints():
    gemini = Exception("429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'details': [{'@type': "
                       "'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '33s'}]}}")
    assert api.server_retry_after_seconds(_wrapped(gemini)) == 33.0
    structured = UnifiedClientError("quota", error_type="rate_limit")
    structured.details = {"error": {"details": [{"retryDelay": "4.5s"}]}}
    assert api.server_retry_after_seconds(structured) == 4.5
    groq = UnifiedClientError("Rate limit reached. Please try again in 1m30.5s.", error_type="rate_limit")
    assert api.server_retry_after_seconds(groq) == 90.5


def test_route_details_count_and_no_hint_is_none():
    assert api.server_retry_after_seconds(
        UnifiedClientError("x", error_type="rate_limit", details={"retry_after_seconds": 12})) == 12.0
    # 0 / inf (a session block) are not a server wait
    assert api.server_retry_after_seconds(
        UnifiedClientError("x", error_type="rate_limit", details={"retry_after_seconds": 0})) is None
    assert api.server_retry_after_seconds(
        UnifiedClientError("x", error_type="rate_limit", details={"retry_after_seconds": float("inf")})) is None
    assert api.server_retry_after_seconds(UnifiedClientError("429 Too Many Requests")) is None
    assert api.server_retry_after_seconds(None) is None


# --------------------------------------------------------------------------- the toggle
def test_policy_env_respect_is_on_by_default_and_ignore_wins_only_with_http_overrides(monkeypatch):
    assert api.respect_retry_after_enabled() is True
    monkeypatch.setenv("RESPECT_RETRY_AFTER", "0")
    assert api.respect_retry_after_enabled() is False
    monkeypatch.setenv("RESPECT_RETRY_AFTER", "1")
    monkeypatch.setenv("IGNORE_RETRY_AFTER", "1")
    assert api.respect_retry_after_enabled() is True   # ignore needs the HTTP overrides on
    monkeypatch.setenv("ENABLE_HTTP_TUNING", "1")
    assert api.respect_retry_after_enabled() is False


def bare_client(client_type="openai"):
    client = UnifiedClient.__new__(UnifiedClient)
    client.client_type = client_type
    client.model = "gpt-test"
    client._multi_key_mode = False
    client._is_stop_requested = lambda: False
    client._is_instance_cancel_requested = lambda: False
    client._ignore_graceful_stop = False
    return client


def test_wait_is_the_server_value_capped_or_the_fallback(monkeypatch):
    logs = []
    monkeypatch.setattr(api, "print", lambda *args, **kwargs: logs.append(" ".join(map(str, args))), raising=False)
    client = bare_client()
    hinted = UnifiedClientError("x", error_type="rate_limit", details={"retry_after_seconds": 42})
    assert client._respect_retry_after_wait(60, error=hinted) == 42
    assert logs == ["⏳ openai: Server asked to retry after 42 s"]
    huge = UnifiedClientError("x", error_type="rate_limit", details={"retry_after_seconds": 10 ** 9})
    assert client._respect_retry_after_wait(60, error=huge) == api.RETRY_AFTER_MAX_SECONDS
    assert client._respect_retry_after_wait(60, error=UnifiedClientError("429")) == 60
    monkeypatch.setenv("RESPECT_RETRY_AFTER", "0")
    assert client._respect_retry_after_wait(60, error=hinted) == 60


# --------------------------------------------------------------------------- raw HTTP path
class FakeResponse:
    def __init__(self, status, headers=None, text="{}"):
        self.status_code = status
        self.headers = headers or {}
        self.text = text
        self.url = "https://api.example/v1/chat/completions"
        self.closed = False

    def json(self):
        return json.loads(self.text)

    def close(self):
        self.closed = True


def http_client(monkeypatch, responses, multi_key=False):
    client = bare_client()
    client._multi_key_mode = multi_key
    client._bind_thread_run_id_for_request = lambda: None
    client._get_send_interval = lambda: 2.0
    client._get_thread_directory = lambda: None
    client._get_active_request_model = lambda: "gpt-test"
    client.request_timeout = 30
    monkeypatch.setattr(api, "_save_outgoing_request", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_save_incoming_response", lambda *args, **kwargs: None)
    queue = list(responses)
    monkeypatch.setattr(api.requests, "request", lambda *args, **kwargs: queue.pop(0))
    slept = []
    monkeypatch.setattr(api.time, "sleep", lambda seconds: slept.append(seconds))
    return client, slept


def test_a_429_body_hint_counts_when_there_is_no_header():
    body = json.dumps({"error": {"code": 429, "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                                                           "retryDelay": "12s"}]}})
    assert api.server_retry_after_seconds(response=FakeResponse(429, text=body)) == 12.0
    assert api.server_retry_after_seconds(response=FakeResponse(429, {"Retry-After": "3"}, text=body)) == 3.0
    assert api.server_retry_after_seconds(response=FakeResponse(429)) is None


def test_http_429_waits_exactly_the_servers_retry_after(monkeypatch):
    client, slept = http_client(monkeypatch, [FakeResponse(429, {"Retry-After": "7"}), FakeResponse(200)])
    resp = client._http_request_with_retries("POST", "https://api.example/v1/models", json={}, max_retries=3)
    assert resp.status_code == 200
    assert sum(slept) == pytest.approx(7.0)


def test_http_429_without_a_header_keeps_the_configured_backoff(monkeypatch):
    client, slept = http_client(monkeypatch, [FakeResponse(429), FakeResponse(200)])
    client._http_request_with_retries("POST", "https://api.example/v1/models", json={}, max_retries=3)
    assert 2.0 * 10 + 1 <= sum(slept) <= 2.0 * 10 + 5   # api_delay * 10 + jitter, as before


def test_http_429_with_the_toggle_off_behaves_as_before(monkeypatch):
    monkeypatch.setenv("RESPECT_RETRY_AFTER", "0")
    client, slept = http_client(monkeypatch, [FakeResponse(429, {"Retry-After": "7"}), FakeResponse(200)])
    client._http_request_with_retries("POST", "https://api.example/v1/models", json={}, max_retries=3)
    assert 8.0 <= sum(slept) <= 12.0   # Retry-After + jitter (the old path)


def test_http_429_in_multi_key_mode_hands_the_wait_to_the_rotation(monkeypatch):
    first = FakeResponse(429, {"retry-after-ms": "45000"})
    client, slept = http_client(monkeypatch, [first], multi_key=True)
    with pytest.raises(UnifiedClientError) as caught:
        client._http_request_with_retries("POST", "https://api.example/v1/models", json={}, max_retries=3)
    assert caught.value.error_type == "rate_limit"
    assert caught.value.details == {"retry_after_seconds": 45.0}
    assert slept == [] and first.closed


def test_http_wait_honours_stop_while_waiting(monkeypatch):
    client, _slept = http_client(monkeypatch, [FakeResponse(429, {"Retry-After": "600"})])
    calls = {"n": 0}

    def stop_after_a_while():
        calls["n"] += 1
        return calls["n"] > 3

    client._is_stop_requested = stop_after_a_while
    with pytest.raises(UnifiedClientError) as caught:
        client._http_request_with_retries("POST", "https://api.example/v1/models", json={}, max_retries=3)
    assert caught.value.error_type == "cancelled"


# --------------------------------------------------------------------------- key pools
def test_rotation_rests_that_key_for_the_servers_wait(monkeypatch):
    import multi_api_key_manager as mk

    pool = mk.APIKeyPool("test")
    pool.add_key(mk.APIKeyEntry("k1", "gpt-test", cooldown=60))
    pool.add_key(mk.APIKeyEntry("k2", "gpt-test", cooldown=60))
    client = bare_client()
    tls = SimpleNamespace(rate_limit_retry_after=None)
    client._get_thread_local_client = lambda: tls
    error = _wrapped(FakeSDKRateLimit({"retry-after": "5"}))
    assert client._rest_key_for_server_wait(error) == 5.0
    pool.mark_key_error(0, 429, cooldown=tls.rate_limit_retry_after)
    key = pool.keys[0]
    assert not key.is_available() and pool.keys[1].is_available()
    key.last_error_time -= 6
    assert key.is_available()   # the server's 5 s, not the key's 60 s

    monkeypatch.setenv("RESPECT_RETRY_AFTER", "0")
    tls.rate_limit_retry_after = None
    assert client._rest_key_for_server_wait(error) is None
    assert tls.rate_limit_retry_after is None   # off: the key's own cooldown, as before


def test_a_routes_own_wait_is_kept():
    client = bare_client()
    tls = SimpleNamespace(rate_limit_retry_after=float("inf"))   # authds/ mute
    client._get_thread_local_client = lambda: tls
    client._rest_key_for_server_wait(_wrapped(FakeSDKRateLimit({"retry-after": "5"})))
    assert tls.rate_limit_retry_after == float("inf")


# --------------------------------------------------------------------------- settings rules / migration
def test_turning_one_on_turns_the_other_off():
    import settings_rules

    config = {"ignore_retry_after": True}
    changed, env = settings_rules.apply_change(config, "respect_retry_after", True)
    assert config == {"ignore_retry_after": False, "respect_retry_after": True}
    assert env == {"RESPECT_RETRY_AFTER": "1", "IGNORE_RETRY_AFTER": "0"}
    changed, env = settings_rules.apply_change(config, "ignore_retry_after", True)
    assert config == {"ignore_retry_after": True, "respect_retry_after": False}
    assert env == {"RESPECT_RETRY_AFTER": "0", "IGNORE_RETRY_AFTER": "1"}
    settings_rules.apply_change(config, "ignore_retry_after", False)
    assert config == {"ignore_retry_after": False, "respect_retry_after": False}   # off stays off


def test_policy_of_saved_configs():
    import settings_rules

    assert settings_rules.retry_after_policy({}) == (True, False)
    assert settings_rules.retry_after_policy({"respect_retry_after": False}) == (False, False)
    # both saved on: the explicit ignore wins (a config from before the toggle reads the same)
    assert settings_rules.retry_after_policy({"respect_retry_after": True, "ignore_retry_after": True}) == (False, True)
    assert settings_rules.retry_after_policy({"ignore_retry_after": True}) == (False, True)


def test_set_choice_updates_config_vars_and_every_listener():
    import settings_rules

    owner = SimpleNamespace(config={}, respect_retry_after_var=True, ignore_retry_after_var=False)
    seen = []
    settings_rules.add_retry_after_listener(owner, lambda: seen.append((owner.respect_retry_after_var,
                                                                         owner.ignore_retry_after_var)))

    def broken():
        raise RuntimeError("Internal C++ object already deleted")

    settings_rules.add_retry_after_listener(owner, broken)
    settings_rules.set_retry_after_choice(owner, "ignore_retry_after", True)
    assert owner.config == {"ignore_retry_after": True, "respect_retry_after": False}
    assert seen == [(False, True)]
    assert broken not in owner._retry_after_listeners   # a dead dialog is dropped
    settings_rules.set_retry_after_choice(owner, "respect_retry_after", True)
    assert seen[-1] == (True, False)


@pytest.mark.parametrize("config,expected", [
    ({}, (True, False)),
    ({"ignore_retry_after": True}, (False, True)),
    ({"ignore_retry_after": True, "respect_retry_after": True}, (False, True)),
    ({"respect_retry_after": False}, (False, False)),
])
def test_startup_migrates_a_contradictory_config(tmp_path, monkeypatch, config, expected):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _headless_env import headless_owner, run_envs

    with headless_owner(tmp_path / "owner", monkeypatch, config) as owner:
        assert (bool(owner.respect_retry_after_var), bool(owner.ignore_retry_after_var)) == expected
    envs = run_envs(tmp_path / "run", monkeypatch, config)
    assert envs["translation"]["RESPECT_RETRY_AFTER"] == ("1" if expected[0] else "0")


def test_schema_and_mobile_list_the_new_setting():
    import settings_schema

    spec = settings_schema.spec("respect_retry_after")
    assert spec.default is True and spec.type == "bool"
    assert spec.label == "Respect provider wait time (Retry-After)"
    assert "RESPECT_RETRY_AFTER" in settings_schema.env_names("respect_retry_after")
    model = (SRC / "mobile" / "app" / "glossarion_mobile" / "ui" / "settings" / "model.py").read_text(encoding="utf-8")
    assert '"indefinite_rate_limit_retry", "respect_retry_after", "ignore_retry_after"' in model


# --------------------------------------------------------------------------- the two dialogs (offscreen Qt)
_DIALOGS = textwrap.dedent(r"""
    import json, sys
    sys.path.insert(0, sys.argv[1])
    from PySide6.QtWidgets import QApplication, QGridLayout, QWidget
    app = QApplication.instance() or QApplication([])
    import other_settings
    from multi_api_key_manager import MultiAPIKeyDialog

    class Owner(QWidget):
        def __init__(self):
            super().__init__()
            self.config = {}
            self.respect_retry_after_var = True
            self.ignore_retry_after_var = False

        def save_config(self, *args, **kwargs):
            pass

        def append_log(self, *args, **kwargs):
            pass

    owner = Owner()
    other_settings.setup_other_settings_methods(owner)
    parent = QWidget()
    QGridLayout(parent)
    other_settings._create_response_handling_section(owner, parent)
    dialog = MultiAPIKeyDialog(None, owner)
    os_respect, os_ignore = owner.respect_retry_after_checkbox, owner.ignore_retry_after_checkbox
    mk_respect = dialog.respect_retry_after_checkbox

    def state(step):
        return {"step": step, "os_respect": os_respect.isChecked(), "os_ignore": os_ignore.isChecked(),
                "mk_respect": mk_respect.isChecked(), "cooldown_enabled": dialog.cooldown_spinbox.isEnabled(),
                "label_enabled": dialog.cooldown_label.isEnabled(), "tooltip": dialog.cooldown_spinbox.toolTip(),
                "config": dict(owner.config), "vars": [owner.respect_retry_after_var, owner.ignore_retry_after_var]}

    states = [state("open")]
    mk_respect.setChecked(False); states.append(state("manager off"))
    os_respect.setChecked(True); states.append(state("other settings on"))
    os_ignore.setChecked(True); states.append(state("ignore on"))
    mk_respect.setChecked(True); states.append(state("manager on"))
    print("STATES=" + json.dumps(states))
""")


def test_both_dialogs_follow_one_setting_live_and_explain_the_cooldown(tmp_path):
    pytest.importorskip("PySide6")
    env = dict(os.environ)
    for name in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "GLOSSARION_LIBRARY_DIR", "OUTPUT_DIRECTORY",
                 "GLOSSARION_DATA_DIR", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME"):
        folder = tmp_path / name.lower()
        folder.mkdir()
        env[name] = str(folder)
    env.update(QT_QPA_PLATFORM="offscreen", GLOSSARION_HTTP_LOG="0", PYTHONIOENCODING="utf-8")
    script = tmp_path / "retry_after_dialogs.py"
    script.write_text(_DIALOGS, encoding="utf-8")
    result = subprocess.run([sys.executable, str(script), str(SRC)], cwd=str(tmp_path), env=env,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    line = next((ln for ln in result.stdout.splitlines() if ln.startswith("STATES=")), None)
    assert line is not None, (result.returncode, result.stdout[-2000:], result.stderr[-3000:])
    states = {item["step"]: item for item in json.loads(line[len("STATES="):])}

    import settings_rules
    opened = states["open"]
    assert (opened["os_respect"], opened["os_ignore"], opened["mk_respect"]) == (True, False, True)
    # the cooldown stays editable (the fallback for a 429 without Retry-After); its tooltip explains it
    assert opened["cooldown_enabled"] is True and opened["label_enabled"] is True
    assert opened["tooltip"] == settings_rules.RETRY_AFTER_COOLDOWN_HINT
    assert "only when a 429 has no Retry-After" in opened["tooltip"]

    off = states["manager off"]   # the manager's copy -> Other Settings follows at once
    assert (off["os_respect"], off["mk_respect"]) == (False, False)
    assert off["cooldown_enabled"] is True and off["tooltip"] == ""
    assert off["config"]["respect_retry_after"] is False and off["vars"] == [False, False]

    on = states["other settings on"]   # and the other way round
    assert (on["os_respect"], on["mk_respect"], on["cooldown_enabled"]) == (True, True, True)
    assert on["tooltip"] == settings_rules.RETRY_AFTER_COOLDOWN_HINT

    ignore = states["ignore on"]   # ignore turns respect off in both dialogs
    assert (ignore["os_respect"], ignore["os_ignore"], ignore["mk_respect"]) == (False, True, False)
    assert ignore["cooldown_enabled"] is True
    assert (ignore["config"]["respect_retry_after"], ignore["config"]["ignore_retry_after"]) == (False, True)

    back = states["manager on"]   # respect from the manager turns ignore off in Other Settings
    assert (back["os_respect"], back["os_ignore"], back["mk_respect"]) == (True, False, True)
    assert (back["config"]["respect_retry_after"], back["config"]["ignore_retry_after"]) == (True, False)
    assert back["vars"] == [True, False]
