import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

import autharena as arena


MODEL_ID = "01a07d42-938f-7267-9398-4529e857491c"


@pytest.fixture(autouse=True)
def clean_cancel(monkeypatch, tmp_path):
    monkeypatch.delenv("TRANSLATION_CANCELLED", raising=False)
    arena._cancel_event.clear()
    monkeypatch.setattr(arena, "_profiles_root", lambda: tmp_path)
    monkeypatch.setattr(arena, "_pool_cursor", 0)
    monkeypatch.setattr(arena, "_ensure_browser_bridge", lambda: {'url': 'http://127.0.0.1:18874', 'control_token': 'test-only'})
    yield
    arena.cancel_stream()
    arena._cancel_event.clear()


def model(**overrides):
    return {"id": MODEL_ID, "name": "internal-name", "publicName": "public-name",
            "displayName": "Public Model", "provider": "provider", "organization": "organization",
            "userSelectable": True,
            "capabilities": {"inputCapabilities": {"text": True}, "outputCapabilities": {"text": True}},
            **overrides}


def flight_page(models):
    data = '8:["$",{},{"initialModels":' + json.dumps(models) + "}]"
    # The real array can span multiple flight scripts. Decode first, then join.
    point = len(data) // 2
    return "".join("<script>self.__next_f.push(" + json.dumps([1, part]) + ")</script>"
                   for part in (data[:point], data[point:]))


def test_catalog_flight_chunks_selectability_aliases_and_stable_duplicates():
    entries = [
        model(name=None, displayName='Model "Quoted" \\ path'),
        model(displayName='Model "Quoted" \\ path', id=str(uuid.uuid4())),
        model(name="hidden", userSelectable=False),
        model(name="image", capabilities={"inputCapabilities": {"text": True},
                                         "outputCapabilities": {"image": True}}),
        model(name="retired", provider=None),
        model(name="missing-org", organization=None),
        model(displayName="bad-id", id="not-a-uuid"),
        model(displayName=None, publicName="public-fallback", id=str(uuid.uuid4())),
        model(displayName="Max", publicName="Max", name="boss-bandit", id=str(uuid.uuid4())),
    ]
    actual = arena._parse_model_catalog(flight_page(entries))
    assert set(actual) == {'Model "Quoted" \\ path', "public-fallback", "max"}
    assert actual['Model "Quoted" \\ path'] == MODEL_ID


@pytest.mark.parametrize("page", ["<html>Access denied</html>", '{"initialModels":[]}', '{"initialModels":null}'])
def test_catalog_missing_or_empty_is_an_error(page):
    with pytest.raises(RuntimeError, match="catalog"):
        arena._parse_model_catalog(page)


def test_model_polling_is_http_only_and_closes_response(monkeypatch):
    class Response:
        status_code = 200
        text = flight_page([model()])
        closed = False

        def close(self):
            self.closed = True

    response = Response()
    seen = []
    monkeypatch.setattr(arena.requests, "get", lambda url, **kw: seen.append((url, kw)) or response)
    monkeypatch.setattr(arena, "_run_browser_helper", lambda *a, **kw: pytest.fail("poll opened browser"))
    assert arena.fetch_available_models(timeout=7, account_id=2) == ["Public Model"]
    assert seen[0][0] == arena.CATALOG_URL
    assert seen[0][1]["timeout"] == 7
    assert "Cookie" not in seen[0][1]["headers"]
    assert response.closed
    response.status_code = 403
    with pytest.raises(RuntimeError, match="HTTP 403"):
        arena.fetch_available_models()


def test_payload_fresh_uuid7_identifiers_and_chat_modality():
    first = arena._build_payload(MODEL_ID, "hello")
    second = arena._build_payload(MODEL_ID, "hello")
    ids = [payload[key] for payload in (first, second)
           for key in ("id", "userMessageId", "modelAMessageId")]
    assert len(set(ids)) == 6
    assert all(uuid.UUID(value).version == 7 for value in ids)
    assert first["modality"] == "chat"
    assert first["mode"] == "direct"
    assert first["userMessage"]["content"] == "hello"
    assert "recaptchaV3Token" not in first


def test_messages_preserve_roles_and_text_without_silent_tools_or_images():
    assert arena._render_messages([{"role": "user", "content": "hello"}]) == "hello"
    messages = [{"role": "system", "content": "Translate faithfully."},
                {"role": "assistant", "name": "previous", "content": 'Answer "A"'},
                {"role": "user", "content": [{"type": "text", "text": "Line1"},
                                           {"type": "input_text", "text": "Line2"}]}]
    rendered = arena._render_messages(messages)
    conversation = json.loads(rendered.split("BEGIN CONVERSATION JSON\n", 1)[1].split("\nEND CONVERSATION JSON")[0])
    assert conversation == [messages[0], messages[1], {"role": "user", "content": "Line1\nLine2"}]
    for message in ({"role": "tool", "content": "secret"},
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "x"}]}):
        with pytest.raises(ValueError, match="support"):
            arena._render_messages([message])
    with pytest.raises(ValueError, match="nonblank"):
        arena._render_messages([{"role": "user", "content": " \n"}])


def test_explicit_unknown_terminal_is_preserved():
    parser = arena._ArenaStreamParser()
    parser.feed('a0:"answer"\nad:{"finishReason":"unknown"}\n')
    assert parser.finish()["finish_reason"] == "unknown"


def test_stream_split_utf8_reasoning_metadata_usage_and_length_finish():
    events = []
    parser = arena._ArenaStreamParser(lambda *event: events.append(event))
    data = ('af:{"messageId":"m"}\nag:"Think…"\na0:"你好"\n'
            'a2:[{"type":"routed_model","organization":"openai"}]\n'
            'ad:{"finishReason":"length","usage":{"promptTokens":12,"completionTokens":3}}').encode()
    for byte in data:
        parser.feed(bytes([byte]))
    assert parser.finish() == {"content": "你好", "reasoning_content": "Think…",
                               "finish_reason": "length", "finish_reason_explicit": True,
                               "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}}
    assert events == [("reasoning", "Think…"), ("content", "你好")]


@pytest.mark.parametrize("data, match", [
    (b'a0:"partial"\n', "without a terminal"),
    (b'a0:"cut off', "malformed or truncated"),
    (b'a0:"\xe4', "truncated UTF-8"),
    (b'a3:"model overloaded"\n', "model overloaded"),
    (b'ad:{"finishReason":"error"}\n', "generation failed"),
    (b'ad:{"finishReason":null}\n', "finishReason"),
    (b'b0:"other model"\n', "second model"),
    (b'a2:[{"type":"image"}]\n', "nontext"),
    (b'<html>Blocked</html>', "unrecognized"),
    (b'ad:{"finishReason":"stop"}\na0:"late"\n', "after the terminal"),
])
def test_stream_errors_never_return_partial_success(data, match):
    with pytest.raises(RuntimeError, match=match):
        parser = arena._ArenaStreamParser()
        parser.feed(data)
        parser.finish()


def test_cancel_env_and_graceful_callback_semantics(monkeypatch):
    monkeypatch.setenv("TRANSLATION_CANCELLED", "1")
    assert arena._cancelled()
    assert not arena._cancelled(lambda: False)
    assert arena._cancelled(lambda: True)
    arena._cancel_event.set()
    arena.reset_cancel()
    assert arena._cancel_event.is_set()


def test_numbered_prefix_validation():
    assert arena._normalize_model("AuthArena2/gpt-example") == "gpt-example"
    assert arena._model_account("AuthArena2/gpt-example", 0) == 2
    with pytest.raises(ValueError, match="conflicts"):
        arena._model_account("autharena2/model", 3)
    for invalid in ("../test", -1, True, 10000):
        with pytest.raises(ValueError):
            arena._account_number(invalid)


def test_account_ids_ignore_files_and_noncanonical_directories(tmp_path):
    for name in ("0", "1", "9999", "01", "-1", "10000", "unrelated"):
        (tmp_path / name).mkdir()
    (tmp_path / "2").write_text("not a profile")
    assert arena.get_account_ids() == [0, 1, 9999]
    assert not arena.get_account_status(0)["logged_in"]
    # Chromium cookie files do not establish signed-in status.
    (tmp_path / "0" / "Cookies").write_text("cookie-like data")
    assert not arena.get_account_status(0)["logged_in"]


@pytest.mark.parametrize("marker", [
    [], {"logged_in": True},
    {"version": 1, "account_id": 2, "logged_in": True, "verified_at": 10},
    {"version": 1, "account_id": 1, "logged_in": "true", "verified_at": 10},
    {"version": 1, "account_id": 1, "logged_in": True, "verified_at": "yesterday"},
    {"version": 1, "account_id": 1, "logged_in": True, "verified_at": float("nan")},
    {"version": 1, "account_id": 1, "logged_in": True, "verified_at": 10**15},
])
def test_account_status_rejects_invalid_verification_markers(tmp_path, marker):
    (tmp_path / "1").mkdir()
    (tmp_path / "1" / arena._LOGIN_STATUS_FILE).write_text(json.dumps(marker))
    assert not arena.get_account_status(1)["logged_in"]


def test_verified_markers_store_no_credentials_and_can_be_invalidated(tmp_path):
    arena._write_account_status(2, logged_in=True)
    status = arena.get_account_status(2)
    assert status["logged_in"] and status["status_cached"]
    marker = json.loads((tmp_path / "2" / arena._LOGIN_STATUS_FILE).read_text())
    assert set(marker) == {"version", "account_id", "logged_in", "verified_at"}
    assert arena.get_rotating_account_pool() == [2]
    arena.mark_account_logged_out(2)
    assert not arena.get_account_status(2)["logged_in"]
    assert arena.get_account_status(2)["verification_state"] == "logged_out"
    assert arena.get_rotating_account_pool() == []


def test_pool_fair_concurrent_rotation_includes_verified_default_only(tmp_path):
    for account in (0, 2, 5):
        arena._write_account_status(account, logged_in=True)
    (tmp_path / "8").mkdir()
    with ThreadPoolExecutor(max_workers=8) as executor:
        pools = list(executor.map(lambda _: arena.get_rotating_account_pool(), range(30)))
    assert all(set(pool) == {0, 2, 5} for pool in pools)
    assert [pool[0] for pool in pools].count(0) == 10
    assert [pool[0] for pool in pools].count(2) == 10
    assert [pool[0] for pool in pools].count(5) == 10


def test_explicit_pool_empty_errors_without_starting_browser(monkeypatch):
    monkeypatch.setattr(arena, "_run_browser_helper", lambda *a, **k: pytest.fail("opened browser"))
    with pytest.raises(arena.AuthArenaError, match="no verified signed-in accounts"):
        arena.send_chat_completion(messages=[{"role": "user", "content": "hello"}], model="autharena0/model")


@pytest.mark.parametrize("captcha_status", [401, 403])
def test_pool_tries_next_safe_rejection_and_preserves_captcha_account(monkeypatch, captcha_status):
    for account in (0, 2):
        arena._write_account_status(account, logged_in=True)
    attempts = []
    rejection = []

    def helper(config, deadline, **kwargs):
        attempts.append(config)
        if config["account_id"] == 0:
            raise arena.AuthArenaError("captcha required", captcha_status, safe_to_rotate=True)
        return {"content": "answer"}

    monkeypatch.setattr(arena, "_run_browser_helper", helper)
    result = arena.send_chat_completion(messages=[{"role": "user", "content": "hello"}], model="AuthArena0/model",
                                        after_rejection_callback=lambda: rejection.append("reset"))
    assert result == {"content": "answer", "account_id": 2}
    assert [item["account_id"] for item in attempts] == [0, 2]
    assert all(item["allow_interactive"] is False for item in attempts)
    assert attempts[0]["payload"]["id"] != attempts[1]["payload"]["id"]
    assert arena.get_account_status(0)["logged_in"]
    assert rejection == ["reset"]


def test_pool_login_rejection_invalidates_account_and_rate_exhaustion_is_bounded(monkeypatch):
    for account in (1, 3):
        arena._write_account_status(account, logged_in=True)
    calls = []

    def helper(config, deadline, **kwargs):
        calls.append(config["account_id"])
        status = 401 if config["account_id"] == 1 else 429
        raise arena.AuthArenaError("rejected", status, "60", safe_to_rotate=True, auth_invalid=status == 401)

    monkeypatch.setattr(arena, "_run_browser_helper", helper)
    with pytest.raises(arena.AuthArenaError, match="exhausted") as caught:
        arena.send_chat_completion(messages=[{"role": "user", "content": "hello"}], model="autharena0/model")
    assert calls == [1, 3]
    assert not arena.get_account_status(1)["logged_in"]
    assert arena.get_account_status(3)["logged_in"]
    assert caught.value.status_code == 429
    assert caught.value.retry_after == "60"


def test_pool_never_retries_ambiguous_dispatched_failure(monkeypatch):
    for account in (0, 1):
        arena._write_account_status(account, logged_in=True)
    calls = []

    def helper(config, deadline, **kwargs):
        calls.append(config["account_id"])
        raise arena.AuthArenaError("transport failed", request_dispatched=True)

    monkeypatch.setattr(arena, "_run_browser_helper", helper)
    with pytest.raises(arena.AuthArenaError, match="transport failed"):
        arena.send_chat_completion(messages=[{"role": "user", "content": "hello"}], model="autharena0/model")
    assert calls == [0]


def test_plain_default_and_numbered_routes_do_not_rotate(monkeypatch):
    calls = []
    monkeypatch.setattr(arena, "_run_browser_helper",
                        lambda config, deadline, **kw: calls.append(config) or {"content": "ok"})
    monkeypatch.setattr(arena, "get_rotating_account_pool", lambda: pytest.fail("unexpected rotation"))
    for name in ("autharena/model", "autharena3/model"):
        arena.send_chat_completion(messages=[{"role": "user", "content": "hello"}], model=name)
    assert [config["account_id"] for config in calls] == [0, 3]
    assert all(config["allow_interactive"] for config in calls)


@pytest.mark.parametrize("prefix", ["autharena", "AUTHARENA"])
def test_bare_route_stays_on_first_account_while_explicit_zero_rotates(monkeypatch, tmp_path, prefix):
    for account_id in (0, 2, 5):
        arena._write_account_status(account_id, logged_in=True)
    (tmp_path / "9").mkdir()  # An unverified slot never enters the pool.
    calls = []

    def helper(config, deadline, **kwargs):
        calls.append(config)
        return {"content": "ok", "finish_reason": "stop", "finish_reason_explicit": True}

    monkeypatch.setattr(arena, "_run_browser_helper", helper)
    messages = [{"role": "user", "content": "hello"}]
    for pooled_account in (0, 2, 5, 0):
        fixed = arena.send_chat_completion(messages=messages, model=f"{prefix}/model")
        pooled = arena.send_chat_completion(messages=messages, model=f"{prefix}0/model")
        numbered = arena.send_chat_completion(messages=messages, model=f"{prefix}2/model")
        assert (fixed["account_id"], pooled["account_id"], numbered["account_id"]) == (0, pooled_account, 2)

    assert [config["account_id"] for config in calls[::3]] == [0, 0, 0, 0]
    assert [config["account_id"] for config in calls[1::3]] == [0, 2, 5, 0]
    assert all(config["allow_interactive"] for config in calls[::3])
    assert all(not config["allow_interactive"] for config in calls[1::3])
    assert all(config["model"] == "model" for config in calls)
    assert len({config["payload"]["id"] for config in calls}) == len(calls)


def test_literal_bare_route_cannot_be_redirected_by_account_argument(monkeypatch):
    calls = []
    monkeypatch.setattr(arena, "_run_browser_helper",
                        lambda config, deadline, **kwargs: calls.append(config) or {"content": "ok"})
    messages = [{"role": "user", "content": "hello"}]
    with pytest.raises(ValueError, match="prefix conflicts with account_id"):
        arena.send_chat_completion(messages=messages, model="autharena/model", account_id=3)
    assert not calls
    # Internal callers and --account-id use a canonical model without a prefix.
    result = arena.send_chat_completion(messages=messages, model="model", account_id=3)
    assert result["account_id"] == 3
    assert calls[0]["account_id"] == 3


def test_bare_route_does_not_fall_back_to_other_verified_accounts(monkeypatch):
    for account_id in (0, 2):
        arena._write_account_status(account_id, logged_in=True)
    calls = []

    def helper(config, deadline, **kwargs):
        calls.append(config["account_id"])
        raise arena.AuthArenaError("first account rejected", 429, safe_to_rotate=True)

    monkeypatch.setattr(arena, "_run_browser_helper", helper)
    with pytest.raises(arena.AuthArenaError, match="first account rejected"):
        arena.send_chat_completion(messages=[{"role": "user", "content": "hello"}], model="autharena/model")
    assert calls == [0]


def test_waiting_for_profile_is_cancellable(tmp_path):
    ready = threading.Event()
    release = threading.Event()

    def owner():
        with arena._profile_gate(0, time.monotonic() + 5):
            ready.set()
            release.wait(3)

    thread = threading.Thread(target=owner)
    thread.start()
    assert ready.wait(2)
    try:
        with pytest.raises(RuntimeError, match="cancelled"):
            with arena._profile_gate(0, time.monotonic() + 3, lambda: True):
                pytest.fail("concurrent profile use")
    finally:
        release.set()
        thread.join(2)


def child_script(monkeypatch, tmp_path, source):
    script = tmp_path / "helper.py"
    script.write_text(
        "import json,sys,time\n"
        "config=json.loads(sys.stdin.readline())\n"
        "def emit(event, **kw):\n"
        " print(json.dumps(dict(autharena=1,event=event,**kw)),flush=True)\n" + source,
        encoding="utf-8")
    monkeypatch.setattr(arena, "_helper_command", lambda: [sys.executable, str(script)])


def test_helper_drains_stderr_and_calls_ready_callback_before_dispatch(monkeypatch, tmp_path):
    marker = tmp_path / "ready"
    child_script(monkeypatch, tmp_path,
                 "sys.stderr.write('diagnostic\\n'*10000);sys.stderr.flush()\n"
                 "emit('ready')\n"
                 "command=json.loads(sys.stdin.readline())\n"
                 "from pathlib import Path\n"
                 "assert Path(config['marker']).exists()\n"
                 "assert command['command']=='dispatch'\n"
                 "emit('chunk',data='a0:\"ok\"\\nad:{\"finishReason\":\"stop\"}\\n')\n"
                 "emit('done')\n")
    result = arena._run_browser_helper({"marker": str(marker)}, time.monotonic() + 8,
                                       before_send_callback=lambda: marker.write_text("ready"))
    assert result["content"] == "ok"
    assert not arena._active_helpers


def test_helper_ready_callback_error_never_dispatches(monkeypatch, tmp_path):
    sent = tmp_path / "sent"
    child_script(monkeypatch, tmp_path,
                 "emit('ready')\n"
                 "if json.loads(sys.stdin.readline()).get('command') == 'dispatch':\n"
                 " from pathlib import Path\n"
                 " Path(config['sent']).touch()\n")

    def failed_callback():
        raise RuntimeError("rate limiter cancelled")

    with pytest.raises(RuntimeError, match="rate limiter cancelled"):
        arena._run_browser_helper({"sent": str(sent)}, time.monotonic() + 5, before_send_callback=failed_callback)
    assert not sent.exists()
    assert not arena._active_helpers


def test_helper_error_preserves_http_metadata(monkeypatch, tmp_path):
    child_script(monkeypatch, tmp_path, "emit('error',message='rate limited',status_code=429,retry_after='60')\n")
    with pytest.raises(arena.AuthArenaError) as caught:
        arena._run_browser_helper({}, time.monotonic() + 5)
    assert caught.value.status_code == 429
    assert caught.value.retry_after == "60"


def test_login_done_without_verified_event_cannot_report_success(monkeypatch, tmp_path):
    child_script(monkeypatch, tmp_path, "emit('done')\n")
    with pytest.raises(arena.AuthArenaError, match="not verified"):
        arena._run_browser_helper({"login": True, "account_id": 1}, time.monotonic() + 5)
    assert not arena.get_account_status(1)["logged_in"]


def test_login_verified_event_records_minimal_marker_and_returns_account(monkeypatch, tmp_path):
    child_script(monkeypatch, tmp_path,
                 "emit('verified',logged_in=True,tou_accepted=True)\nemit('done')\n")
    result = arena._run_browser_helper({"login": True, "account_id": 3}, time.monotonic() + 5)
    assert result == {"profile_saved": True, "logged_in": True, "account_id": 3}
    assert arena.get_account_status(3)["logged_in"]


@pytest.mark.parametrize("status, auth_invalid, remains_signed_in",
                         [(401, True, False), (401, False, True), (403, False, True), (429, False, True)])
def test_helper_auth_invalidation_does_not_discard_captcha_sessions(monkeypatch, tmp_path, status, auth_invalid, remains_signed_in):
    arena._write_account_status(2, logged_in=True)
    child_script(monkeypatch, tmp_path, f"emit('error',message='rejected',status_code={status},safe_to_rotate=True,auth_invalid={auth_invalid})\n")
    with pytest.raises(arena.AuthArenaError) as caught:
        arena._run_browser_helper({"account_id": 2}, time.monotonic() + 5)
    assert caught.value.safe_to_rotate
    assert arena.get_account_status(2)["logged_in"] is remains_signed_in


@pytest.mark.parametrize("kind,exception", [("timeout", TimeoutError), ("configuration", ImportError)])
def test_helper_error_preserves_error_type(monkeypatch, tmp_path, kind, exception):
    child_script(monkeypatch, tmp_path, f"emit('error',message='classified failure',error_type={kind!r})\n")
    with pytest.raises(exception, match="classified failure"):
        arena._run_browser_helper({}, time.monotonic() + 5)


def test_helper_timeout_and_abrupt_exit_are_not_retried(monkeypatch, tmp_path):
    child_script(monkeypatch, tmp_path, "time.sleep(30)\n")
    with pytest.raises(TimeoutError):
        arena._run_browser_helper({}, time.monotonic() + 0.2)
    assert not arena._active_helpers
    child_script(monkeypatch, tmp_path, "emit('ready')\nsys.stdin.readline()\nsys.exit(2)\n")
    with pytest.raises(RuntimeError, match="after dispatch; the request was not retried") as caught:
        arena._run_browser_helper({}, time.monotonic() + 5)
    assert caught.value.request_dispatched is True
    assert caught.value.safe_to_rotate is False


@pytest.mark.parametrize("response", [
    "emit('chunk',data='a0:\"partial\"\\n')\nemit('done')\n",
    "emit('chunk',data='not-stream-data\\n')\n",
    "emit('error',message='transport timeout',error_type='timeout')\n",
])
def test_dispatched_parser_and_helper_timeout_errors_carry_no_retry_metadata(monkeypatch, tmp_path, response):
    child_script(monkeypatch, tmp_path, "emit('ready')\nsys.stdin.readline()\n" + response)
    with pytest.raises((RuntimeError, TimeoutError)) as caught:
        arena._run_browser_helper({}, time.monotonic() + 5)
    assert caught.value.request_dispatched is True
    assert caught.value.safe_to_rotate is False


def test_parent_timeout_after_dispatch_carries_no_retry_metadata(monkeypatch, tmp_path):
    child_script(monkeypatch, tmp_path, "emit('ready')\nsys.stdin.readline()\ntime.sleep(30)\n")
    with pytest.raises(TimeoutError) as caught:
        arena._run_browser_helper({}, time.monotonic() + 1)
    assert caught.value.request_dispatched is True
    assert caught.value.safe_to_rotate is False


def test_helper_rejection_rearms_graceful_cancel(monkeypatch, tmp_path):
    child_script(monkeypatch, tmp_path,
                 "emit('ready')\nsys.stdin.readline()\nemit('rejected')\ntime.sleep(30)\n")
    stopped = []
    with pytest.raises(RuntimeError, match="cancelled"):
        arena._run_browser_helper({}, time.monotonic() + 5, cancel_check=lambda: bool(stopped),
                                  after_rejection_callback=lambda: stopped.append(True))
    assert stopped == [True]


def test_send_logs_options_once_and_uses_new_payloads(monkeypatch):
    configs = []
    logs = []
    callbacks = []
    arena._warned_options.clear()

    def fake_helper(config, deadline, **kwargs):
        configs.append(config)
        kwargs["before_send_callback"]()
        kwargs["on_chunk"]("content", "answer")
        return {"content": "answer", "reasoning_content": "", "finish_reason": "stop",
                "finish_reason_explicit": True, "usage": {}}

    monkeypatch.setattr(arena, "_run_browser_helper", fake_helper)
    for _ in range(2):
        arena.send_chat_completion(messages=[{"role": "user", "content": "hello"}], model="AuthArena2/test",
                                   max_tokens=100, log_fn=logs.append, progress_label="API call in progress",
                                   before_send_callback=lambda: callbacks.append("send"))
    assert configs[0]["payload"]["id"] != configs[1]["payload"]["id"]
    assert configs[0]["account_id"] == 2
    assert configs[0]["model"] == "test"
    assert sum("unsupported options" in log for log in logs) == 1
    assert logs.count("API call in progress") == 2
    assert logs.count("answer") == 2
    assert callbacks == ["send", "send"]


@pytest.mark.parametrize("batch,shared,batch_allow,provider,expected", [
    ("0", "1", "0", "1", True),
    ("0", "0", "1", "1", False),
    ("0", "1", "1", "0", False),
    ("1", "1", "0", "1", False),
    ("1", "0", "1", "1", True),
    ("1", "1", "true", "0", True),
])
def test_stream_visibility_uses_forced_provider_settings(monkeypatch, batch, shared, batch_allow, provider, expected):
    monkeypatch.setenv("BATCH_TRANSLATION", batch)
    monkeypatch.setenv("LOG_STREAM_CHUNKS", shared)
    monkeypatch.setenv("ALLOW_AUTHGPT_BATCH_STREAM_LOGS", batch_allow)
    monkeypatch.setenv("AUTHARENA_LOG_STREAM_CHUNKS", provider)
    monkeypatch.setenv("ENABLE_STREAMING", "0")
    monkeypatch.setenv("AUTHARENA_STREAM", "0")
    assert arena._stream_logs_enabled() is expected
    assert arena._stream_logs_enabled(False) is False
    assert arena._stream_logs_enabled(True) is True


def test_small_content_and_reasoning_deltas_are_visible_before_helper_returns(monkeypatch):
    logs = []
    fragments = []
    monkeypatch.setenv("BATCH_TRANSLATION", "0")
    monkeypatch.setenv("LOG_STREAM_CHUNKS", "1")
    monkeypatch.setenv("AUTHARENA_LOG_STREAM_CHUNKS", "1")
    monkeypatch.setenv("AUTHARENA_STREAM_THINKING_LOGS", "1")
    monkeypatch.setenv("ENABLE_STREAMING", "0")

    def helper(config, deadline, **kwargs):
        for channel, text in [("reasoning", "hmm"), ("content", "A"), ("content", " "),
                              ("content", "你\n"), ("reasoning", "check"), ("content", "\tB")]:
            count = len(fragments)
            kwargs["on_chunk"](channel, text)
            assert fragments[count:] == [(channel, text)]  # Already emitted, before next delta.
        assert not any("Stream finished" in line for line in logs)
        return {"content": "A 你\n\tB", "reasoning_content": "hmmcheck",
                "finish_reason": "length", "finish_reason_explicit": True, "usage": {}}

    monkeypatch.setattr(arena, "_run_browser_helper", helper)
    result = arena.send_chat_completion(messages=[{"role": "user", "content": "go"}], model="test",
                                       stream=False, log_fn=logs.append,
                                       log_chunk_fn=lambda channel, text: fragments.append((channel, text)))
    assert result["finish_reason"] == "length"
    assert result["reasoning_content"] == "hmmcheck"
    assert logs.count("🧠 [autharena] Thinking...") == 2
    assert logs.count("📡 AuthArena: Text streaming...") == 2
    assert logs[-1].startswith("📡 AuthArena: Stream finished in")
    assert "A" not in logs  # Structured chunks are not duplicated through log_fn.


def test_plain_logger_gets_tiny_deltas_immediately_and_hidden_reasoning_is_retained(monkeypatch):
    logs = []
    monkeypatch.setenv("AUTHARENA_STREAM_THINKING_LOGS", "0")

    def helper(config, deadline, **kwargs):
        kwargs["on_chunk"]("reasoning", "hidden")
        assert not any("Thinking" in line or "hidden" in line for line in logs)
        kwargs["on_chunk"]("content", "x")
        assert logs[-1] == "x"
        kwargs["on_chunk"]("content", " ")
        assert logs[-1] == " "
        return {"content": "x ", "reasoning_content": "hidden", "finish_reason": "stop",
                "finish_reason_explicit": True, "usage": {}}

    monkeypatch.setattr(arena, "_run_browser_helper", helper)
    result = arena.send_chat_completion(messages=[{"role": "user", "content": "go"}], model="test",
                                       log_stream=True, log_fn=logs.append)
    assert result["reasoning_content"] == "hidden"


def test_stream_error_closes_visible_phase_without_changing_failure_metadata(monkeypatch):
    logs = []

    def helper(config, deadline, **kwargs):
        kwargs["on_chunk"]("content", "partial")
        error = RuntimeError("stream was truncated")
        error.request_dispatched = True
        raise error

    monkeypatch.setattr(arena, "_run_browser_helper", helper)
    with pytest.raises(RuntimeError) as caught:
        arena.send_chat_completion(messages=[{"role": "user", "content": "go"}], model="test",
                                   log_stream=True, log_fn=logs.append)
    assert caught.value.request_dispatched is True
    assert logs[-1] == "📡 AuthArena: Stream finished with an error"


def test_disabled_stream_visibility_keeps_final_content_without_fragment_callbacks(monkeypatch):
    logs = []

    def helper(config, deadline, **kwargs):
        kwargs["on_chunk"]("content", "answer")
        kwargs["on_chunk"]("reasoning", "reason")
        return {"content": "answer", "reasoning_content": "reason", "finish_reason": "stop",
                "finish_reason_explicit": True, "usage": {}}

    monkeypatch.setattr(arena, "_run_browser_helper", helper)
    result = arena.send_chat_completion(messages=[{"role": "user", "content": "go"}], model="test",
                                       log_stream=False, log_fn=logs.append,
                                       log_chunk_fn=lambda *args: pytest.fail("hidden stream was logged"))
    assert result["content"] == "answer"
    assert result["reasoning_content"] == "reason"
    assert not any("streaming" in line or "Thinking" in line for line in logs)


def test_cli_messages_stdin_json_and_no_browser_for_list(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["autharena.py", "--messages", "-", "--system", "Translate", "--json", "--quiet"])
    monkeypatch.setattr(sys, "stdin", io.StringIO('[{"role":"user","content":"你好"}]'))
    seen = []
    monkeypatch.setattr(arena, "send_chat_completion", lambda **kwargs: seen.append(kwargs) or {"content": "Hello"})
    assert arena._main() == 0
    assert json.loads(capsys.readouterr().out)["content"] == "Hello"
    assert seen[0]["messages"] == [{"role": "system", "content": "Translate"}, {"role": "user", "content": "你好"}]
    monkeypatch.setattr(sys, "argv", ["autharena.py", "--list-models", "--json"])
    monkeypatch.setattr(arena, "fetch_available_models", lambda **kwargs: ["model-a", "model-b"])
    assert arena._main() == 0
    assert json.loads(capsys.readouterr().out) == ["model-a", "model-b"]


def test_cli_invalid_inputs_and_frozen_command(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["autharena.py", "--prompt", "text", "--prompt-file", "file"])
    assert arena._main() == 1
    assert "not both" in capsys.readouterr().err
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert arena._helper_command() == [sys.executable, "--autharena-helper"]


def test_cli_stream_fragments_are_readable_contiguous_stderr(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["autharena.py", "--prompt", "go"])

    def send(**kwargs):
        kwargs["log_fn"]("Text streaming")
        kwargs["log_chunk_fn"]("content", "hel")
        kwargs["log_chunk_fn"]("content", "lo ")
        kwargs["log_chunk_fn"]("content", "world")
        assert capsys.readouterr().err == "Text streaming\nhello world"
        kwargs["log_fn"]("Stream finished")
        assert capsys.readouterr().err == "\nStream finished\n"
        return {"content": "hello world"}

    monkeypatch.setattr(arena, "send_chat_completion", send)
    assert arena._main() == 0
    assert capsys.readouterr().out == "hello world\n"


@pytest.mark.skipif(not shutil.which("node"), reason="Node optional for browser JavaScript simulation")
@pytest.mark.parametrize("scenario", ["success", "anonymous", "terms", "transport", "rate", "challenge", "alias", "hydrated",
                                     "login_success", "login_anonymous", "login_terms",
                                     "noninteractive_anonymous", "noninteractive_challenge", "noninteractive_challenge401"])
def test_browser_script_runs_normal_protocol_without_exporting_tokens(tmp_path, scenario):
    entries = [model(displayName='Name "quoted" \\ backslash', publicName="public-model", name="internal-model")]
    wanted = "internal-model" if scenario == "alias" else entries[0]["displayName"]
    prompt = "__MODEL__ __PAYLOAD__ __TIMEOUT_MS__ \u2028 hello"
    user = None if scenario in ("anonymous", "login_anonymous", "noninteractive_anonymous") else {
        "email": "user@example.test", "touConsentTimestamp": None if scenario in ("terms", "login_terms") else "2026-01-01"}
    flight = json.dumps({"initialModels": entries, "user": user})
    source = arena._prepare_script(arena._build_payload(None, prompt), wanted, 3,
                                   login_only=scenario.startswith("login_"),
                                   allow_interactive=not scenario.startswith("noninteractive"))
    script = r'''
const vm = require('vm');
const fs = require('fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
global.window = global;
global.location = {origin:'https://arena.ai'};
window.__next_f = [[1,input.flight]];
const requests = [];
const actions = [];
let challenge;
global.document = {
  scripts:[{src:'https://www.google.com/recaptcha/enterprise.js?render=public-v3-key'}],
  getElementById:()=>null,
  createElement:()=>({style:{}, append(){}, remove(){}}),
  body:{append(){}}
};
if(input.scenario==='hydrated') {
  window.__next_f=[];
  const middle=Math.floor(input.flight.length/2);
  for(const part of [input.flight.slice(0,middle),input.flight.slice(middle)]) {
    document.scripts.push({textContent:'self.__next_f.push('+JSON.stringify([1,part])+')'});
  }
}
window.grecaptcha = {enterprise:{
  ready:cb=>cb(),
  execute:async(key,args)=>{actions.push(args.action);return 'PRIVATE-V3-TOKEN';},
  render:(widget,opts)=>{challenge=opts.callback;}
}};
global.fetch = async(url, options)=>{
  requests.push({url, body:options.body?JSON.parse(options.body):null,credentials:options.credentials});
  if(url==='/text/direct')return {ok:true,text:async()=>input.flight};
  if(input.scenario==='transport') throw Error('disconnected');
  if(input.scenario==='rate') return {ok:false,status:429,headers:{get:()=> '30'},json:async()=>({error:'Rate limit exceeded'})};
  if(['challenge','noninteractive_challenge','noninteractive_challenge401'].includes(input.scenario) && requests.length===2) return {ok:false,status:input.scenario==='noninteractive_challenge401'?401:403,headers:{get:()=>null},json:async()=>({error:'reCAPTCHA validation failed'})};
  let read=false;
  return {ok:true,status:200,headers:{get:()=> 'text/plain'},body:{getReader:()=>({
    read:async()=>{
      if(read)return {done:true};
      read=true;
      return {done:false,value:new TextEncoder().encode('ag:"thinking"\na0:"answer"\nad:{"finishReason":"stop"}\n')};
    }
  })}};
};
global.DOMParser = class {parseFromString(text){
  return {scripts:[{textContent:'self.__next_f.push('+JSON.stringify([1,text])+')'}]};
}};
if(input.scenario==='login_success'){
  // Verify the fresh snapshot, not the stale pre-login page bootstrap.
  window.__next_f=[[1,JSON.stringify({initialModels:[],user:null})]];
}
(async()=>{
  vm.runInThisContext(input.source);
  await new Promise(resolve=>setTimeout(resolve,50));
  let state=window.__glossarionArena;
  if(state.phase==='ready') await state.dispatch();
  if(challenge){challenge('PRIVATE-V2-TOKEN');await state.dispatch();}
  console.log(JSON.stringify({requests,actions,events:state.events,phase:state.phase}));
})().catch(error=>{console.error(error);process.exit(1)});
'''
    result = subprocess.run([shutil.which("node"), "-e", script], input=json.dumps({
        "flight": flight, "source": source, "scenario": scenario}), text=True,
        capture_output=True, timeout=10, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    assert "PRIVATE" not in json.dumps(actual["events"])
    assert "user@example.test" not in json.dumps(actual["events"])
    generation = [request for request in actual["requests"] if request["url"] == arena.CREATE_EVALUATION_PATH]
    if scenario.startswith("login_"):
        assert generation == []
        assert actual["actions"] == []
        assert len(actual["requests"]) == 1
        if scenario == "login_success":
            assert actual["events"][-1]["event"] == "done"
            assert any(event["event"] == "verified" for event in actual["events"])
        else:
            assert actual["events"][-1]["event"] == "action"
            assert not any(event["event"] == "verified" for event in actual["events"])
        return
    if scenario in ("anonymous", "terms"):
        assert generation == []
        assert actual["events"][-1]["event"] == "action"
        return
    if scenario == "noninteractive_anonymous":
        assert generation == []
        assert actual["events"][-1]["event"] == "error"
        assert actual["events"][-1]["status_code"] == 401
        assert actual["events"][-1]["safe_to_rotate"] is True
        return
    assert actual["actions"] == ["chat_submit"]
    assert all(item["url"] == arena.CREATE_EVALUATION_PATH and item["credentials"] == "same-origin"
               and item["body"]["userMessage"]["content"] == prompt for item in generation)
    if scenario == "transport":
        assert len(generation) == 1
        assert actual["events"][-1]["event"] == "error"
        assert "not retried" in actual["events"][-1]["message"]
    elif scenario == "rate":
        assert actual["events"][-1]["status_code"] == 429
        assert actual["events"][-1]["retry_after"] == "30"
    elif scenario in ("noninteractive_challenge", "noninteractive_challenge401"):
        assert len(generation) == 1
        assert actual["events"][-1]["status_code"] == (401 if scenario.endswith("401") else 403)
        assert actual["events"][-1]["safe_to_rotate"] is True
        assert actual["events"][-1]["auth_invalid"] is False
        assert not any(event["event"] == "action" for event in actual["events"])
        assert not any(event["event"] == "logged_out" for event in actual["events"])
    else:
        assert actual["phase"] == "done"
        parser = arena._ArenaStreamParser()
        for event in actual["events"]:
            if event["event"] == "chunk":
                parser.feed(event["data"])
        assert parser.finish()["content"] == "answer"
        if scenario == "challenge":
            assert len(generation) == 2
            assert generation[1]["body"]["recaptchaV2Token"] == "PRIVATE-V2-TOKEN"
            assert "recaptchaV3Token" not in generation[1]["body"]
            assert any(event["event"] == "rejected" for event in actual["events"])
