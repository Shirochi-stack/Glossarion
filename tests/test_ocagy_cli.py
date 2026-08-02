import json
import os
import queue
import stat
import sys
import threading
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import ocagy_cli


def _fake_opencode(tmp_path: Path) -> Path:
    script = tmp_path / "fake_opencode.py"
    script.write_text(
        "import json, sys\n"
        "args = sys.argv[1:]\n"
        "if '--version' in args:\n"
        "    print('1.4.11')\n"
        "    raise SystemExit(0)\n"
        "if args[:2] == ['models', 'google']:\n"
        "    print('google/antigravity-gemini-3.1-pro\\ngoogle/antigravity-gemini-3-flash')\n"
        "    raise SystemExit(0)\n"
        "if args[:2] == ['auth', 'login']:\n"
        "    print('login')\n"
        "    raise SystemExit(0)\n"
        "prompt = sys.stdin.read()\n"
        "model = args[args.index('--model') + 1]\n"
        "variant = args[args.index('--variant') + 1] if '--variant' in args else ''\n"
        "print(json.dumps({'type':'step_finish','part':{'usage':{'input_tokens':12,'output_tokens':2,'total_tokens':14}}}))\n"
        "print(json.dumps({'type':'text','part':{'text':'OK:' + model + ':' + variant + ':' + str(len(prompt))}}))\n",
        encoding="utf-8",
    )

    if os.name == "nt":
        exe = tmp_path / "opencode.cmd"
        exe.write_text(
            f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n',
            encoding="utf-8",
        )
    else:
        exe = tmp_path / "opencode"
        exe.write_text(
            f"#!{sys.executable}\n" + script.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return exe


def _write_enabled_account(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "antigravity-accounts.json").write_text(
        json.dumps({
            "accounts": [
                {"email": "one@example.com", "refreshToken": "secret", "enabled": True},
            ]
        }),
        encoding="utf-8",
    )


def _write_accounts(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "antigravity-accounts.json").write_text(
        json.dumps({
            "version": 1,
            "accounts": [
                {"email": "one@example.com", "refreshToken": "token-one", "enabled": True},
                {"email": "two@example.com", "refreshToken": "token-two", "enabled": True},
                {"email": "three@example.com", "refreshToken": "token-three", "enabled": False},
            ],
            "activeIndex": 1,
            "activeIndexByFamily": {"gemini": 1, "claude": 0},
        }),
        encoding="utf-8",
    )


def test_numbered_prefix_account_mapping():
    expected_model = "gemini-3.1-pro-high"
    assert ocagy_cli.parse_account_route(f"ocagy0/{expected_model}") == (0, expected_model)
    assert ocagy_cli.parse_account_route(f"ocagy/{expected_model}") == (1, expected_model)
    assert ocagy_cli.parse_account_route(f"ocagy1/{expected_model}") == (2, expected_model)
    assert ocagy_cli.parse_account_route(f"ocagy2/{expected_model}") == (3, expected_model)
    assert ocagy_cli.parse_account_route(expected_model) == (0, expected_model)
    assert ocagy_cli.resolve_model(f"ocagy1/{expected_model}") == (
        "google/antigravity-gemini-3.1-pro",
        "high",
    )


def test_numbered_prefix_isolates_selected_account(tmp_path, monkeypatch):
    exe = _fake_opencode(tmp_path)
    config_dir = tmp_path / "config"
    _write_accounts(config_dir)
    (config_dir / "antigravity.json").write_text(
        json.dumps({
            "pid_offset_enabled": True,
            "account_selection_strategy": "round-robin",
            "quota_fallback": True,
        }),
        encoding="utf-8",
    )
    captured = {}

    def fake_server_send(**kwargs):
        isolated_dir = Path(kwargs["subprocess_env"]["OPENCODE_CONFIG_DIR"])
        account_store = json.loads(
            (isolated_dir / "antigravity-accounts.json").read_text(encoding="utf-8")
        )
        plugin_settings = json.loads(
            (isolated_dir / "antigravity.json").read_text(encoding="utf-8")
        )
        captured.update({
            "isolated_dir": isolated_dir,
            "account_store": account_store,
            "plugin_settings": plugin_settings,
        })
        return {
            "content": "PINNED",
            "finish_reason": "stop",
            "usage": None,
            "raw_response": [],
        }

    monkeypatch.setenv("OCAGY_CLI_PATH", str(exe))
    monkeypatch.setenv("OCAGY_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(config_dir))
    monkeypatch.setattr(ocagy_cli, "_send_via_server", fake_server_send)
    ocagy_cli.reset_cancel()

    result = ocagy_cli.send_chat_completion(
        messages=[{"role": "user", "content": "test"}],
        model="ocagy1/gemini-3.1-pro-high",
        timeout=30,
    )

    assert result["content"] == "PINNED"
    assert captured["isolated_dir"] != config_dir
    assert not captured["isolated_dir"].exists()
    assert [item["email"] for item in captured["account_store"]["accounts"]] == ["two@example.com"]
    assert captured["account_store"]["activeIndex"] == 0
    assert captured["plugin_settings"]["pid_offset_enabled"] is False
    assert captured["plugin_settings"]["account_selection_strategy"] == "sticky"
    assert captured["plugin_settings"]["quota_fallback"] is True


def test_numbered_prefix_rejects_missing_and_disabled_slots(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    _write_accounts(config_dir)
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(config_dir))

    with pytest.raises(ocagy_cli.OcAgyError, match=r"account #4 is not linked"):
        ocagy_cli._require_oauth_account(4)
    with pytest.raises(ocagy_cli.OcAgyError, match=r"account #3 is disabled"):
        ocagy_cli._require_oauth_account(3)


def test_send_uses_live_server_and_variant(tmp_path, monkeypatch):
    exe = _fake_opencode(tmp_path)
    config_dir = tmp_path / "config"
    _write_enabled_account(config_dir)
    captured = {}

    def fake_server_send(**kwargs):
        captured.update(kwargs)
        return {
            "content": (
                "OK:"
                + kwargs["model_id"]
                + ":"
                + str(kwargs["variant"] or "")
                + ":"
                + str(len(kwargs["prompt"]))
            ),
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
            "raw_response": [],
        }

    monkeypatch.setenv("OCAGY_CLI_PATH", str(exe))
    monkeypatch.setenv("OCAGY_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(config_dir))
    monkeypatch.setattr(ocagy_cli, "_send_via_server", fake_server_send)
    monkeypatch.delenv("TRANSLATION_CANCELLED", raising=False)
    ocagy_cli.reset_cancel()

    result = ocagy_cli.send_chat_completion(
        messages=[
            {"role": "system", "content": "Translate faithfully."},
            {"role": "user", "content": "한글 test " * 10000},
        ],
        model="ocagy0/gemini-3.1-pro-high",
        timeout=30,
    )

    assert result["content"].startswith("OK:google/antigravity-gemini-3.1-pro:high:")
    assert result["variant"] == "high"
    assert result["usage"]["total_tokens"] == 14
    assert captured["log_stream"] is True
    assert captured["model_id"] == "google/antigravity-gemini-3.1-pro"
    assert Path(captured["subprocess_env"]["OPENCODE_CONFIG_DIR"]) == config_dir
    config = json.loads((tmp_path / "workspace" / "opencode.json").read_text(encoding="utf-8"))
    assert config["plugin"] == ["opencode-antigravity-auth@latest"]
    assert config["permission"]["*"] == "deny"


def test_status_discovers_plugin_models_and_accounts(tmp_path, monkeypatch):
    exe = _fake_opencode(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "antigravity-accounts.json").write_text(
        json.dumps({
            "accounts": [
                {"email": "one@example.com", "refreshToken": "secret", "enabled": True},
                {"email": "off@example.com", "refreshToken": "secret", "enabled": False},
            ]
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("OCAGY_CLI_PATH", str(exe))
    monkeypatch.setenv("OCAGY_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(config_dir))

    status = ocagy_cli.get_status()
    assert status["installed"] is True
    assert status["plugin_ready"] is True
    assert status["authenticated"] is True
    assert status["account_count"] == 1
    assert status["emails"] == ["one@example.com"]
    assert "google/antigravity-gemini-3.1-pro" in status["models"]

    account_summary = ocagy_cli.get_account_summary()
    assert account_summary["account_count"] == 1
    assert account_summary["emails"] == ["one@example.com"]
    assert "refreshToken" not in account_summary


def test_quota_status_fetches_live_plugin_pools_without_exposing_tokens(tmp_path, monkeypatch):
    exe = _fake_opencode(tmp_path)
    config_dir = tmp_path / "config"
    _write_enabled_account(config_dir)
    monkeypatch.setenv("OCAGY_CLI_PATH", str(exe))
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(config_dir))
    calls = []

    def fake_form(url, values, timeout):
        calls.append((url, values["refresh_token"], timeout))
        return {"access_token": "temporary-access-token"}

    def fake_post(url, access_token, payload, timeout, *, user_agent=ocagy_cli._ANTIGRAVITY_QUOTA_USER_AGENT):
        assert access_token == "temporary-access-token"
        calls.append((url, dict(payload), user_agent))
        if url.endswith(":loadCodeAssist"):
            return {"cloudaicompanionProject": {"id": "managed-project"}}
        assert payload == {"project": "managed-project"}
        if url.endswith(":fetchAvailableModels"):
            return {
                "models": {
                    "claude-sonnet-4-6": {
                        "quotaInfo": {
                            "remainingFraction": 0.75,
                            "resetTime": "2030-01-02T00:00:00Z",
                        }
                    },
                    "gemini-3.1-pro-high": {
                        "quotaInfo": {
                            "remainingFraction": 0.6,
                            "resetTime": "2030-01-01T23:00:00Z",
                        }
                    },
                    "gemini-3.1-pro-low": {
                        "quotaInfo": {
                            "remainingFraction": 0.8,
                            "resetTime": "2030-01-02T00:00:00Z",
                        }
                    },
                    "gemini-3-flash": {
                        "quotaInfo": {
                            "remainingFraction": 0.15,
                            "resetTime": "2030-01-01T22:00:00Z",
                        }
                    },
                }
            }
        if url.endswith(":retrieveUserQuota"):
            return {
                "buckets": [{
                    "modelId": "gemini-2.5-pro",
                    "remainingFraction": 0.9,
                    "resetTime": "2030-01-02T00:00:00Z",
                }]
            }
        raise AssertionError(f"Unexpected quota URL: {url}")

    monkeypatch.setattr(ocagy_cli, "_quota_form_json", fake_form)
    monkeypatch.setattr(ocagy_cli, "_quota_post_json", fake_post)
    monkeypatch.setattr(
        ocagy_cli,
        "_plugin_oauth_client_credentials",
        lambda: ("test-client-id", "test-client-secret"),
    )

    status = ocagy_cli.get_quota_status(timeout=7)

    assert status["account_count"] == 1
    account = status["quota_accounts"][0]
    assert account["status"] == "ok"
    assert account["antigravity"]["claude"]["remaining_fraction"] == 0.75
    assert account["antigravity"]["gemini-pro"]["remaining_fraction"] == 0.6
    assert account["antigravity"]["gemini-pro"]["model_count"] == 2
    assert account["antigravity"]["gemini-flash"]["remaining_fraction"] == 0.15
    assert account["gemini_cli"][0]["remaining_fraction"] == 0.9
    serialized = json.dumps(status)
    assert "secret" not in serialized
    assert "temporary-access-token" not in serialized
    assert any(call[0].endswith(":fetchAvailableModels") for call in calls)
    assert any(call[0].endswith(":retrieveUserQuota") for call in calls)


def test_oauth_application_metadata_is_loaded_from_installed_plugin(tmp_path, monkeypatch):
    plugin_root = tmp_path / "opencode-antigravity-auth"
    constants = plugin_root / "dist" / "src" / "constants.js"
    constants.parent.mkdir(parents=True)
    constants.write_text(
        "export const ANTIGRAVITY_CLIENT_ID = 'test-client-id';\n"
        "export const ANTIGRAVITY_CLIENT_SECRET = 'test-client-secret';\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OCAGY_PLUGIN_ROOT", str(plugin_root))
    monkeypatch.delenv("OCAGY_GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.delenv("OCAGY_GOOGLE_CLIENT_SECRET", raising=False)

    assert ocagy_cli._plugin_oauth_client_credentials() == (
        "test-client-id",
        "test-client-secret",
    )


def test_poll_models_expands_opencode_models_into_ocagy_variants(tmp_path, monkeypatch):
    exe = _fake_opencode(tmp_path)
    config_dir = tmp_path / "config"
    _write_enabled_account(config_dir)
    monkeypatch.setenv("OCAGY_CLI_PATH", str(exe))
    monkeypatch.setenv("OCAGY_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(config_dir))

    models = ocagy_cli.poll_models(timeout=5)

    assert models == [
        "ocagy/gemini-3.1-pro-high",
        "ocagy/gemini-3.1-pro-low",
        "ocagy/gemini-3-flash-minimal",
        "ocagy/gemini-3-flash-low",
        "ocagy/gemini-3-flash-medium",
        "ocagy/gemini-3-flash-high",
    ]


def test_polled_unknown_antigravity_model_remains_routable():
    models = ocagy_cli.normalize_polled_models([
        "google/gemini-2.5-pro",
        "google/antigravity-future-model",
    ])

    assert models == ["ocagy/antigravity-future-model"]
    assert ocagy_cli.resolve_model(models[0]) == (
        "google/antigravity-future-model",
        None,
    )


def test_send_without_oauth_account_stops_before_launch(tmp_path, monkeypatch):
    exe = _fake_opencode(tmp_path)
    config_dir = tmp_path / "empty-config"
    monkeypatch.setenv("OCAGY_CLI_PATH", str(exe))
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(config_dir))
    monkeypatch.setattr(
        ocagy_cli.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("OpenCode should not launch")),
    )

    try:
        ocagy_cli.send_chat_completion(
            messages=[{"role": "user", "content": "test"}],
            model="gemini-3.1-pro-high",
        )
    except ocagy_cli.OcAgyError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected an OAuth setup error")

    assert "does not use or require a Google API key" in message
    assert "OAuth with Google (Antigravity)" in message
    assert "Do not select Manually enter API Key" in message


def test_google_api_key_fallback_error_is_rewritten_as_oauth_guidance():
    error = ocagy_cli._classify_error(
        "Google Generative AI API key is missing. Pass GOOGLE_GENERATIVE_AI_API_KEY.",
        1,
    )

    message = str(error)
    assert "fell back to its built-in Google provider" in message
    assert "does not use or require a Google API key" in message
    assert "OAuth with Google (Antigravity)" in message


def test_live_event_state_collects_deltas_reasoning_usage_and_completion():
    state = ocagy_cli._OpenCodeStreamState("session-1")
    assert state.feed({
        "type": "message.updated",
        "properties": {"info": {"id": "assistant-1", "role": "assistant"}},
    }) == []

    reasoning = state.feed({
        "type": "message.part.updated",
        "properties": {
            "part": {
                "id": "reason-1",
                "sessionID": "session-1",
                "messageID": "assistant-1",
                "type": "reasoning",
                "text": "Think",
            },
            "delta": "Think",
        },
    })
    first = state.feed({
        "type": "message.part.updated",
        "properties": {
            "part": {
                "id": "text-1",
                "sessionID": "session-1",
                "messageID": "assistant-1",
                "type": "text",
                "text": "Hel",
            },
            "delta": "Hel",
        },
    })
    second = state.feed({
        "type": "message.part.updated",
        "properties": {
            "part": {
                "id": "text-1",
                "sessionID": "session-1",
                "messageID": "assistant-1",
                "type": "text",
                "text": "Hello",
            },
            "delta": "lo",
        },
    })
    state.feed({
        "type": "message.part.updated",
        "properties": {
            "part": {
                "id": "step-1",
                "sessionID": "session-1",
                "messageID": "assistant-1",
                "type": "step-finish",
                "reason": "stop",
                "tokens": {"input": 8, "output": 2, "total": 10},
            }
        },
    })
    state.feed({
        "type": "session.status",
        "properties": {"sessionID": "session-1", "status": {"type": "idle"}},
    })

    assert reasoning == [("reasoning", "Think")]
    assert first == [("text", "Hel")]
    assert second == [("text", "lo")]
    assert state.content() == "Hello"
    assert state.complete is True
    assert state.finish_reason == "stop"
    assert ocagy_cli._usage_from_value(state.step_events)["total_tokens"] == 10


def test_live_event_state_handles_opencode_part_delta_events():
    state = ocagy_cli._OpenCodeStreamState("session-1")
    state.feed({
        "type": "message.updated",
        "properties": {"info": {"id": "assistant-1", "role": "assistant"}},
    })
    state.feed({
        "type": "message.part.updated",
        "properties": {
            "part": {
                "id": "text-1",
                "sessionID": "session-1",
                "messageID": "assistant-1",
                "type": "text",
            }
        },
    })

    first = state.feed({
        "type": "message.part.delta",
        "properties": {
            "sessionID": "session-1",
            "messageID": "assistant-1",
            "partID": "text-1",
            "field": "text",
            "delta": "Hel",
        },
    })
    second = state.feed({
        "type": "message.part.delta",
        "properties": {
            "sessionID": "session-1",
            "messageID": "assistant-1",
            "partID": "text-1",
            "field": "text",
            "delta": "lo",
        },
    })

    assert first == [("text", "Hel")]
    assert second == [("text", "lo")]
    assert state.content() == "Hello"


def test_httpx_sse_reader_emits_event_before_stream_finishes():
    blocked_after_first_event = threading.Event()
    release_stream = threading.Event()
    output = queue.Queue()

    class FakeHttpxResponse:
        def iter_lines(self):
            yield 'data: {"type":"server.connected"}'
            yield ""
            blocked_after_first_event.set()
            release_stream.wait(timeout=2)
            yield 'data: {"type":"session.idle","properties":{"sessionID":"s1"}}'
            yield ""

    reader = threading.Thread(
        target=ocagy_cli._read_sse_events,
        args=(FakeHttpxResponse(), output),
        daemon=True,
    )
    reader.start()
    try:
        assert blocked_after_first_event.wait(timeout=1)
        kind, event = output.get(timeout=1)
        assert kind == "event"
        assert event == {"type": "server.connected"}
        assert reader.is_alive(), "The first event must arrive while the HTTP stream is still open"
    finally:
        release_stream.set()
        reader.join(timeout=2)


def test_ocagy_server_setup_uses_request_deadline_not_short_hardcoded_timeouts(tmp_path, monkeypatch):
    import io
    from types import SimpleNamespace

    http_calls = []
    stream_timeouts = {}

    class FakeProcess:
        pid = 12345
        stdout = io.StringIO("")
        stderr = io.StringIO("")

        def poll(self):
            return None

    class FakeEventResponse:
        status_code = 200

        def close(self):
            pass

    class FakeStreamContext:
        def __enter__(self):
            return FakeEventResponse()

        def __exit__(self, *args):
            pass

    class FakeTimeout:
        def __init__(self, default, *, connect):
            stream_timeouts.update(default=default, connect=connect)

    def fake_http_json(base_url, path, **kwargs):
        http_calls.append((path, kwargs["timeout"]))
        if path == "/global/health":
            return {"healthy": True}
        if path == "/session":
            return {"id": "session-1"}
        return {}

    def fake_read_events(_response, output):
        output.put(("event", {
            "type": "server.connected",
        }))
        output.put(("event", {
            "type": "message.updated",
            "properties": {"info": {"id": "assistant-1", "role": "assistant"}},
        }))
        output.put(("event", {
            "type": "message.part.updated",
            "properties": {
                "part": {
                    "id": "text-1",
                    "sessionID": "session-1",
                    "messageID": "assistant-1",
                    "type": "text",
                    "text": "OK",
                },
                "delta": "OK",
            },
        }))
        output.put(("event", {
            "type": "session.idle",
            "properties": {"sessionID": "session-1"},
        }))

    fake_httpx = SimpleNamespace(
        Timeout=FakeTimeout,
        stream=lambda *args, **kwargs: FakeStreamContext(),
        TimeoutException=type("FakeTimeoutException", (Exception,), {}),
        RequestError=type("FakeRequestError", (Exception,), {}),
    )

    monkeypatch.setattr(ocagy_cli.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(ocagy_cli, "_http_json", fake_http_json)
    monkeypatch.setattr(ocagy_cli, "_read_sse_events", fake_read_events)
    monkeypatch.setattr(ocagy_cli, "_terminate_process_tree", lambda proc: None)
    monkeypatch.setattr(ocagy_cli, "httpx", fake_httpx)

    result = ocagy_cli._send_via_server(
        exe="opencode",
        request_dir=tmp_path,
        prompt="test",
        model_id="google/antigravity-gemini-3.1-pro",
        variant="high",
        timeout_seconds=120,
        logger=lambda message: None,
        log_stream=False,
        subprocess_env={},
    )

    timeout_by_path = {path: timeout for path, timeout in http_calls}
    assert result["content"] == "OK"
    assert timeout_by_path["/session"] > 100
    assert timeout_by_path["/session/session-1/prompt_async"] > 100
    assert stream_timeouts["default"] > 100
    assert stream_timeouts["connect"] > 100


def test_ocagy_forced_stream_includes_thinking_when_thinking_toggle_is_off(monkeypatch):
    monkeypatch.setenv("STREAM_THINKING_LOGS", "0")

    assert ocagy_cli._forced_stream_thinking_logging_enabled(True) is True
    assert ocagy_cli._forced_stream_thinking_logging_enabled(False) is False


def test_prompt_preserves_roles():
    prompt = ocagy_cli.build_prompt([
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
        {"role": "assistant", "content": "MEMORY"},
    ])
    assert "<SYSTEM_INSTRUCTIONS>\nSYS" in prompt
    assert "<USER>\nUSER\n</USER>" in prompt
    assert "<ASSISTANT>\nMEMORY\n</ASSISTANT>" in prompt


def test_missing_cli_error_has_copyable_install_commands(tmp_path, monkeypatch):
    monkeypatch.setenv("OCAGY_CLI_PATH", str(tmp_path / "missing-opencode.exe"))
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(ocagy_cli, "_candidate_paths", lambda: [])

    try:
        ocagy_cli.find_executable()
    except ocagy_cli.OcAgyError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected a missing-CLI error")

    assert ocagy_cli.OPENCODE_NPM_INSTALL_COMMAND in message
    assert "Searched:" not in message
    if os.name == "nt":
        assert ocagy_cli.NODEJS_WINGET_INSTALL_COMMAND in message


def test_model_catalog_exposes_ocagy_high_variant():
    from model_options import get_model_options

    options = set(get_model_options())
    assert "ocagy/gemini-3.1-pro-high" in options
    assert "ocagy/gemini-3.1-pro-low" in options


def test_unified_client_routes_ocagy_without_api_key(monkeypatch):
    import unified_api_client as unified

    captured = {}

    def fake_ocagy_send(**kwargs):
        captured.update(kwargs)
        return {
            "content": "ROUTED",
            "finish_reason": "stop",
            "usage": {"total_tokens": 3},
        }

    monkeypatch.setattr(
        unified,
        "_ocagy_send",
        fake_ocagy_send,
    )
    monkeypatch.setattr(unified, "_ocagy_is_cancelled", lambda: False)
    monkeypatch.setattr(unified, "_ocagy_reset_cancel", lambda: None)
    monkeypatch.setenv("BATCH_TRANSLATION", "1")
    monkeypatch.setenv("ALLOW_AUTHGPT_BATCH_STREAM_LOGS", "1")

    client = unified.UnifiedClient(
        "",
        "ocagy/gemini-3.1-pro-high",
        _skip_cancel_reset=True,
    )
    assert client.client_type == "ocagy"
    assert client._model_needs_api_key("ocagy/gemini-3.1-pro-high") is False
    response = client._send_ocagy(
        [{"role": "user", "content": "test"}],
        0.2,
        1024,
        "test",
    )
    assert response.content == "ROUTED"
    assert response.usage["total_tokens"] == 3
    assert captured["log_stream"] is True
    assert captured["model"] == "ocagy/gemini-3.1-pro-high"


def test_ocagy_uses_effectively_unlimited_timeout_when_overrides_are_off(monkeypatch):
    import unified_api_client as unified

    captured = {}

    def fake_ocagy_send(**kwargs):
        captured.update(kwargs)
        return {"content": "OK", "finish_reason": "stop", "usage": None}

    monkeypatch.setattr(unified, "_ocagy_send", fake_ocagy_send)
    monkeypatch.setattr(unified, "_ocagy_is_cancelled", lambda: False)
    monkeypatch.setattr(unified, "_ocagy_reset_cancel", lambda: None)
    monkeypatch.setenv("RETRY_TIMEOUT", "0")
    monkeypatch.setenv("ENABLE_HTTP_TUNING", "0")

    client = unified.UnifiedClient(
        "",
        "ocagy/gemini-3.1-pro-high",
        _skip_cancel_reset=True,
    )
    client._send_ocagy(
        [{"role": "user", "content": "test"}],
        0.2,
        1024,
        "test",
    )

    assert captured["timeout"] == 36000


def test_ocagy_uses_read_timeout_only_when_http_overrides_are_on(monkeypatch):
    import unified_api_client as unified

    captured = {}

    def fake_ocagy_send(**kwargs):
        captured.update(kwargs)
        return {"content": "OK", "finish_reason": "stop", "usage": None}

    monkeypatch.setattr(unified, "_ocagy_send", fake_ocagy_send)
    monkeypatch.setattr(unified, "_ocagy_is_cancelled", lambda: False)
    monkeypatch.setattr(unified, "_ocagy_reset_cancel", lambda: None)
    monkeypatch.setenv("RETRY_TIMEOUT", "0")
    monkeypatch.setenv("ENABLE_HTTP_TUNING", "1")
    monkeypatch.setenv("READ_TIMEOUT", "12.5")

    client = unified.UnifiedClient(
        "",
        "ocagy/gemini-3.1-pro-high",
        _skip_cancel_reset=True,
    )
    client._send_ocagy(
        [{"role": "user", "content": "test"}],
        0.2,
        1024,
        "test",
    )

    assert captured["timeout"] == 12.5


def test_ocagy_content_filter_raises_prohibited_content(monkeypatch):
    import unified_api_client as unified

    def fake_ocagy_send(**kwargs):
        raise ocagy_cli.OcAgyError(
            "OpenCode Antigravity failed (exit 1): "
            "The response was blocked by the provider's content filter"
        )

    monkeypatch.setattr(unified, "_ocagy_send", fake_ocagy_send)
    monkeypatch.setattr(unified, "_ocagy_is_cancelled", lambda: False)
    monkeypatch.setattr(unified, "_ocagy_reset_cancel", lambda: None)

    client = unified.UnifiedClient(
        "",
        "ocagy/gemini-3.1-pro-high",
        _skip_cancel_reset=True,
    )

    with pytest.raises(unified.UnifiedClientError) as exc_info:
        client._send_ocagy(
            [{"role": "user", "content": "test"}],
            0.2,
            1024,
            "test",
        )

    assert exc_info.value.error_type == "prohibited_content"
    assert "blocked by the provider's content filter" in str(exc_info.value)


def test_ocagy_content_filter_skips_generic_api_retries(monkeypatch):
    import unified_api_client as unified

    attempts = []

    def fake_ocagy_send(**kwargs):
        attempts.append(1)
        raise ocagy_cli.OcAgyError(
            "OpenCode Antigravity failed (exit 1): "
            "The response was blocked by the provider's content filter"
        )

    monkeypatch.setattr(unified, "_ocagy_send", fake_ocagy_send)
    monkeypatch.setattr(unified, "_ocagy_is_cancelled", lambda: False)
    monkeypatch.setattr(unified, "_ocagy_reset_cancel", lambda: None)
    monkeypatch.setenv("DISABLE_REFUSAL_CHECKS", "1")
    monkeypatch.setenv("MAX_RETRIES", "3")
    monkeypatch.setenv("USE_FALLBACK_KEYS", "0")

    client = unified.UnifiedClient(
        "",
        "ocagy/gemini-3.1-pro-high",
        _skip_cancel_reset=True,
    )
    monkeypatch.setattr(client, "_save_payload", lambda *args, **kwargs: None)
    monkeypatch.setattr(client, "_save_failed_request", lambda *args, **kwargs: None)
    monkeypatch.setattr(client, "_track_stats", lambda *args, **kwargs: None)

    _content, finish_reason = client._send_internal(
        [{"role": "user", "content": "test"}],
        temperature=0.2,
        max_tokens=1024,
        context="translation",
        request_id="ocagy-filter-test",
    )

    assert finish_reason == "prohibited_content"
    assert len(attempts) == 1


def test_disabled_timeout_retry_does_not_retry_provider_timeout(monkeypatch):
    import unified_api_client as unified

    attempts = []

    def fake_response(*args, **kwargs):
        attempts.append(1)
        raise unified.UnifiedClientError("timed out", error_type="provider_error")

    monkeypatch.setenv("RETRY_TIMEOUT", "0")
    monkeypatch.setenv("MAX_RETRIES", "3")

    client = unified.UnifiedClient(
        "",
        "ocagy/gemini-3.1-pro-high",
        _skip_cancel_reset=True,
    )
    monkeypatch.setattr(client, "_get_response", fake_response)
    monkeypatch.setattr(client, "_save_payload", lambda *args, **kwargs: None)
    monkeypatch.setattr(client, "_save_failed_request", lambda *args, **kwargs: None)

    with pytest.raises(unified.UnifiedClientError, match="timed out"):
        client._send_internal(
            [{"role": "user", "content": "test"}],
            temperature=0.2,
            max_tokens=1024,
            context="translation",
            request_id="timeout-test",
        )

    assert len(attempts) == 1


def test_unified_client_preserves_numbered_ocagy_route(monkeypatch):
    import unified_api_client as unified

    captured = {}

    def fake_ocagy_send(**kwargs):
        captured.update(kwargs)
        return {"content": "PINNED", "finish_reason": "stop", "usage": None}

    monkeypatch.setattr(unified, "_ocagy_send", fake_ocagy_send)
    monkeypatch.setattr(unified, "_ocagy_is_cancelled", lambda: False)
    monkeypatch.setattr(unified, "_ocagy_reset_cancel", lambda: None)
    monkeypatch.setenv("BATCH_TRANSLATION", "0")

    client = unified.UnifiedClient(
        "",
        "ocagy2/gemini-3.1-pro-high",
        _skip_cancel_reset=True,
    )
    response = client._send_ocagy(
        [{"role": "user", "content": "test"}],
        0.2,
        1024,
        "test",
    )

    assert client.client_type == "ocagy"
    assert response.content == "PINNED"
    assert captured["model"] == "ocagy2/gemini-3.1-pro-high"


def test_ocagy_forced_stream_logs_ignore_general_streaming_toggle(monkeypatch):
    import unified_api_client as unified

    captured = {}

    def fake_ocagy_send(**kwargs):
        captured.update(kwargs)
        return {"content": "LIVE", "finish_reason": "stop", "usage": None}

    monkeypatch.setattr(unified, "_ocagy_send", fake_ocagy_send)
    monkeypatch.setattr(unified, "_ocagy_is_cancelled", lambda: False)
    monkeypatch.setattr(unified, "_ocagy_reset_cancel", lambda: None)
    monkeypatch.setenv("BATCH_TRANSLATION", "0")
    monkeypatch.setenv("ENABLE_STREAMING", "0")
    monkeypatch.setenv("LOG_STREAM_CHUNKS", "1")

    client = unified.UnifiedClient(
        "",
        "ocagy/gemini-3.1-pro-high",
        _skip_cancel_reset=True,
    )
    response = client._send_ocagy(
        [{"role": "user", "content": "test"}],
        0.2,
        1024,
        "test",
    )

    assert response.content == "LIVE"
    assert captured["log_stream"] is True
