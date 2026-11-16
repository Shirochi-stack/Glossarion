import ast
import json
import os
import queue
import re
import stat
import sys
import threading
import time
from types import SimpleNamespace
from pathlib import Path

import pytest
from _src_corpus import DESKTOP_GUI_MODULES, SRC_DIR, desktop_gui_source, module_tree

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import ocagy_cli


@pytest.fixture(autouse=True)
def _no_plugin_update_network(monkeypatch, tmp_path):
    """Tests never reach npm or start the background plugin update.

    The workspace base (state and lock files) and the OpenCode cache default to
    tmp folders, so no test reads or writes the real ones of this machine.
    """

    def offline(*_args, **_kwargs):
        raise OSError("npm registry disabled in tests")

    monkeypatch.setattr(ocagy_cli, "_fetch_latest_plugin_version", offline)
    monkeypatch.setattr(ocagy_cli, "_start_background_plugin_update", lambda *_a, **_k: None)
    monkeypatch.setenv("OCAGY_WORKSPACE", str(tmp_path / "default-workspace"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "default-cache"))
    ocagy_cli.reset_cancel()


def test_zen_poll_accepts_only_free_cli_catalog_rows(monkeypatch):
    calls = []
    def listing(args, timeout):
        calls.append((args, timeout))
        return SimpleNamespace(returncode=0, stdout=(
            'opencode/mimo-v2.6-flash-free\n'
            'opencode/paid-model\n'
            'warning: opencode/deepseek-v4-flash-free\n'
            'google/other-free\n'
            'opencode/mimo-v2.6-flash-free\n'
        ), stderr='')
    monkeypatch.setattr(ocagy_cli, '_run_short', listing)
    assert ocagy_cli.poll_zen_models(timeout=9) == ['ocz/mimo-v2.6-flash-free']
    assert calls == [(['models', 'opencode'], 9)]


def test_zen_poll_failure_does_not_confirm_models_in_error_logs(monkeypatch):
    monkeypatch.setattr(ocagy_cli, '_run_short', lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stdout='', stderr='opencode/invalid-free'))
    with pytest.raises(ocagy_cli.OcAgyError, match='polling failed'):
        ocagy_cli.poll_zen_models()


def test_zen_error_detail_unwraps_effect_causes_without_duplicate_logs():
    detail = (
        'Model not found: opencode/deepseek-v4-flash-free. '
        'Did you mean: mimo-v2.6-flash-free?'
    )
    logs = '\n'.join(
        'timestamp=2026-10-02T16:29:19Z level=ERROR message="prompt failed" cause='
        + json.dumps(f'Cause([{kind}(ProviderModelNotFoundError: {detail})])')
        for kind in ('Fail', 'Die')
    )
    assert ocagy_cli._zen_error_detail(logs) == detail


def test_zen_error_detail_preserves_unstructured_error():
    assert ocagy_cli._zen_error_detail('Connection refused') == 'Connection refused'


@pytest.mark.parametrize('error_type,expect_traceback', [
    ('config_error', False), ('cancelled', False), ('api_error', True),
])
def test_text_translation_error_handler_limits_expected_tracebacks(monkeypatch, error_type, expect_traceback):
    # Execute the actual handler without loading the GUI or starting a translation.
    # U3: the handler lives in text_jobs.TextJobsMixin._process_text_file
    trees = [module_tree(m) for m in DESKTOP_GUI_MODULES if (SRC_DIR / f"{m}.py").exists()]
    handler = next(
        node for tree in trees for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler)
        and any(isinstance(child, ast.Name) and child.id == 'is_config_error'
                for child in ast.walk(node))
    )
    function = ast.FunctionDef(
        name='handle',
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='self'), ast.arg(arg='e')],
                           kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=handler.body, decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))

    class ClientError(Exception):
        def __init__(self, message):
            super().__init__(message)
            self.error_type = error_type

    monkeypatch.setitem(sys.modules, 'unified_api_client', SimpleNamespace(UnifiedClientError=ClientError))
    namespace = {}
    exec(compile(module, '<text-error-handler>', 'exec'), namespace)
    messages = []
    assert namespace['handle'](SimpleNamespace(append_log=messages.append), ClientError('Model not found')) is False
    assert any('Full error:' in message for message in messages) == expect_traceback
    if error_type == 'config_error':
        assert messages == ['❌ Translation error: Model not found']


@pytest.fixture(autouse=True)
def isolate_credential_runtime_patching(monkeypatch):
    # CLI fakes must never patch the user's real installed OAuth plugin.
    # Real copied-runtime adapter coverage lives in test_proxy_token_storage.
    monkeypatch.setattr(ocagy_cli, "_secure_plugin_account_storage", lambda **kwargs: None)


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


def test_generation_controls_are_encoded_for_opencode():
    config = ocagy_cli._workspace_config(temperature=0.3)

    assert config["agent"]["glossarion"]["temperature"] == 0.3
    assert all(
        model["temperature"] is True
        for model in config["provider"]["google"]["models"].values()
    )
    assert ocagy_cli._effective_output_limit(
        "google/antigravity-gemini-3.1-pro",
        128000,
    ) == 65535
    assert ocagy_cli._effective_output_limit(
        "google/antigravity-claude-sonnet-4-6",
        4096,
    ) == 4096


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
        from proxy_token_storage import load_accounts
        from token_encryption import is_encrypted
        assert is_encrypted(str(isolated_dir / "antigravity-accounts.json"))
        account_store = load_accounts(isolated_dir / "antigravity-accounts.json")
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
        captured["request_config"] = json.loads(
            (Path(kwargs["request_dir"]) / "opencode.json").read_text(encoding="utf-8")
        )
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
    logs = []

    result = ocagy_cli.send_chat_completion(
        messages=[
            {"role": "system", "content": "Translate faithfully."},
            {"role": "user", "content": "한글 test " * 10000},
        ],
        model="ocagy0/gemini-3.1-pro-high",
        temperature=0.25,
        max_tokens=128000,
        timeout=30,
        log_fn=logs.append,
    )

    assert result["content"].startswith("OK:google/antigravity-gemini-3.1-pro:high:")
    assert result["variant"] == "high"
    assert result["usage"]["total_tokens"] == 14
    assert captured["log_stream"] is True
    assert captured["model_id"] == "google/antigravity-gemini-3.1-pro"
    assert Path(captured["subprocess_env"]["OPENCODE_CONFIG_DIR"]) == config_dir
    assert captured["subprocess_env"]["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] == "128000"
    assert captured["request_config"]["agent"]["glossarion"]["temperature"] == 0.25
    assert any(
        "Glossarion controls OpenCode generation" in message
        and "max_tokens=65,535" in message
        and "requested 128,000" in message
        for message in logs
    )
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
    assert state.finish_reason is None
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
    assert state.raw_finish_reason == "stop"
    assert state.finish_reason_source == "sse_step_finish"
    assert ocagy_cli._usage_from_value(state.step_events)["total_tokens"] == 10


def test_live_event_state_captures_out_of_order_content_filter_finish():
    state = ocagy_cli._OpenCodeStreamState("session-1")

    # OpenCode can publish a part before its message.updated notification is
    # observed by this client. Finish metadata must not depend on that ordering.
    state.feed({
        "type": "message.part.updated",
        "properties": {
            "part": {
                "id": "step-1",
                "sessionID": "session-1",
                "messageID": "assistant-1",
                "type": "step-finish",
                "reason": "content-filter",
            }
        },
    })

    assert state.finish_reason == "prohibited_content"
    assert state.raw_finish_reason == "content-filter"
    assert state.finish_reason_source == "sse_step_finish"


def test_persisted_message_finish_prefers_step_finish_reason():
    reason, source, evidence = ocagy_cli._finish_reason_from_message_records(
        [{
            "info": {
                "id": "assistant-1",
                "role": "assistant",
                "finish": "stop",
            },
            "parts": [{
                "id": "step-1",
                "type": "step-finish",
                "reason": "max_tokens",
            }],
        }],
        ["assistant-1"],
    )

    assert reason == "max_tokens"
    assert source == "server_message_step_finish"
    assert evidence == {
        "message_id": "assistant-1",
        "part_id": "step-1",
    }
    assert ocagy_cli._normalize_opencode_finish_reason(reason) == "length"


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
        if path == "/session/session-1/message/assistant-1":
            return {
                "info": {
                    "id": "assistant-1",
                    "role": "assistant",
                    "finish": "stop",
                },
                "parts": [{
                    "id": "step-1",
                    "type": "step-finish",
                    "reason": "max_tokens",
                }],
            }
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
    assert result["finish_reason"] == "length"
    assert result["raw_finish_reason"] == "max_tokens"
    assert result["finish_reason_source"] == "server_message_step_finish"
    assert result["finish_reason_fallback"] is False
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


@pytest.mark.skipif(os.name != "nt", reason="Windows npm shim behavior")
def test_find_executable_prefers_native_binary_over_npm_shim(tmp_path, monkeypatch):
    npm_dir = tmp_path / "npm"
    native = npm_dir / "node_modules" / "opencode-ai" / "node_modules" / "opencode-windows-x64" / "bin" / "opencode.exe"
    native.parent.mkdir(parents=True)
    native.touch()
    (npm_dir / "opencode.cmd").touch()
    monkeypatch.setenv("APPDATA", str(tmp_path))
    # APPDATA/npm is the package location; a stale PATH can still find the shim.
    monkeypatch.setattr(ocagy_cli.shutil, "which", lambda name: str(npm_dir / "opencode.cmd") if name == "opencode" else None)

    assert ocagy_cli.find_executable() == str(native.resolve())


@pytest.mark.skipif(os.name != "nt", reason="Windows Node.js PATH behavior")
def test_subprocess_env_adds_newly_installed_node_to_stale_path(tmp_path, monkeypatch):
    node_dir = tmp_path / "Local" / "Programs" / "nodejs"
    node_dir.mkdir(parents=True)
    (node_dir / "node.exe").touch()
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    monkeypatch.setenv("PATH", str(tmp_path / "old-path"))

    env = ocagy_cli._subprocess_env()

    assert str(node_dir) in env["PATH"].split(os.pathsep)


def test_ensure_opencode_installed_runs_installer_and_redetects_cli(monkeypatch):
    state = {"installed": False}
    logs = []
    setup_calls = []

    def fake_find(_explicit_path=None):
        if state["installed"]:
            return "/home/test/.opencode/bin/opencode"
        raise ocagy_cli.OcAgyError(ocagy_cli.get_install_instructions())

    def fake_setup(command, **kwargs):
        setup_calls.append((command, kwargs))
        state["installed"] = True
        return {"ok": True, "output": "installed"}

    monkeypatch.setattr(ocagy_cli, "find_executable", fake_find)
    monkeypatch.setattr(
        ocagy_cli,
        "_opencode_install_command",
        lambda: ["npm", "install", "-g", "opencode-ai"],
    )
    monkeypatch.setattr(ocagy_cli, "_run_setup_command", fake_setup)

    executable = ocagy_cli.ensure_opencode_installed(log_fn=logs.append)

    assert executable == "/home/test/.opencode/bin/opencode"
    assert setup_calls[0][0] == ["npm", "install", "-g", "opencode-ai"]
    assert any("installing it automatically" in message for message in logs)
    assert any("OpenCode installed successfully" in message for message in logs)


def test_ensure_opencode_installed_keeps_manual_commands_as_failure_fallback(monkeypatch):
    def missing(_explicit_path=None):
        raise ocagy_cli.OcAgyError(ocagy_cli.get_install_instructions())

    monkeypatch.setattr(ocagy_cli, "find_executable", missing)
    monkeypatch.setattr(
        ocagy_cli,
        "_opencode_install_command",
        lambda: ["npm", "install", "-g", "opencode-ai"],
    )
    monkeypatch.setattr(
        ocagy_cli,
        "_run_setup_command",
        lambda *_args, **_kwargs: {"ok": False, "error": "network blocked"},
    )

    with pytest.raises(ocagy_cli.OcAgyError) as error:
        ocagy_cli.ensure_opencode_installed()

    message = str(error.value)
    assert "could not install OpenCode automatically" in message
    assert "network blocked" in message
    assert ocagy_cli.OPENCODE_NPM_INSTALL_COMMAND in message


def test_ensure_auth_plugin_installed_resolves_plugin_through_opencode(tmp_path, monkeypatch):
    executable = str(tmp_path / "opencode")
    logs = []
    calls = []
    key = str(Path(executable).resolve(strict=False)).casefold()
    ocagy_cli._PLUGIN_BOOTSTRAPPED_EXECUTABLES.discard(key)

    def fake_setup(command, **kwargs):
        calls.append((command, kwargs))
        return {
            "ok": True,
            "output": "google/antigravity-gemini-3.1-pro",
        }

    monkeypatch.setattr(ocagy_cli, "_workspace_dir", lambda: tmp_path)
    monkeypatch.setattr(ocagy_cli, "_run_setup_command", fake_setup)

    ocagy_cli.ensure_auth_plugin_installed(executable, log_fn=logs.append)

    assert calls[0][0] == [executable, "models", "google"]
    assert calls[0][1]["cwd"] == tmp_path
    assert key in ocagy_cli._PLUGIN_BOOTSTRAPPED_EXECUTABLES
    assert any("auth plugin is installed and ready" in message for message in logs)


def test_ensure_auth_plugin_installed_reports_manual_fallback_after_failure(tmp_path, monkeypatch):
    executable = str(tmp_path / "opencode-failing")
    key = str(Path(executable).resolve(strict=False)).casefold()
    ocagy_cli._PLUGIN_BOOTSTRAPPED_EXECUTABLES.discard(key)

    monkeypatch.setattr(ocagy_cli, "_workspace_dir", lambda: tmp_path)
    monkeypatch.setattr(
        ocagy_cli,
        "_run_setup_command",
        lambda *_args, **_kwargs: {"ok": False, "error": "plugin download blocked"},
    )

    with pytest.raises(ocagy_cli.OcAgyError) as error:
        ocagy_cli.ensure_auth_plugin_installed(executable)

    message = str(error.value)
    assert "could not install or verify opencode-antigravity-auth automatically" in message
    assert "plugin download blocked" in message
    assert "Fallback:" in message
    assert "opencode models google" in message


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


MISSING_FINISH_PROVIDERS = (
    (
        "ocagy",
        "ocagy/gemini-3.1-pro-high",
        "_ocagy_send",
        {
            "content": "translated",
            "finish_reason": "stop",
            "finish_reason_fallback": True,
            "finish_reason_source": "fallback_stop",
            "usage": None,
        },
    ),
    (
        "authnd",
        "authnd/z-ai/glm-5.1",
        "_authnd_send",
        {
            "content": "translated",
            "finish_reason": "stop",
            "finish_reason_explicit": False,
            "finish_reason_inference": "done_without_finish_reason",
            "usage": None,
        },
    ),
)


def _make_missing_finish_client(monkeypatch, model):
    import unified_api_client as unified

    monkeypatch.setattr(unified, "_ocagy_is_cancelled", lambda: False)
    monkeypatch.setattr(unified, "_ocagy_reset_cancel", lambda: None)
    monkeypatch.setattr(unified, "_authnd_reset_cancel", lambda: None)
    client = unified.UnifiedClient("", model, _skip_cancel_reset=True)
    monkeypatch.setattr(client, "_save_payload", lambda *args, **kwargs: None)
    monkeypatch.setattr(client, "_save_failed_request", lambda *args, **kwargs: None)
    monkeypatch.setattr(client, "_track_stats", lambda *args, **kwargs: None)
    monkeypatch.setattr(client, "_compute_backoff", lambda *args, **kwargs: 0)
    monkeypatch.setattr(unified.time, "sleep", lambda _seconds: None)
    return unified, client


def test_authnd_graceful_stop_cannot_cross_cleared_provider_boundary(monkeypatch):
    import unified_api_client as unified

    monkeypatch.setenv("GRACEFUL_STOP", "0")
    monkeypatch.setenv("GRACEFUL_STOP_COMPLETED", "0")
    monkeypatch.setenv("MAX_RETRIES", "1")
    monkeypatch.setattr(unified, "_api_watchdog_external_write", lambda _state: None)
    entered_preflight = threading.Event()
    release_preflight = threading.Event()
    provider_posts = []
    progress_updates = []
    preflight_cancel_checks = []

    def fake_authnd_send(**kwargs):
        entered_preflight.set()
        assert release_preflight.wait(2.0)
        preflight_cancel_checks.append(kwargs["cancel_check"]())
        # Exercise the final atomic boundary as well, even though the real
        # AuthND helper exits as soon as the cancellation check turns true.
        kwargs["before_send_callback"]()
        provider_posts.append(True)
        return {
            "content": "translated",
            "finish_reason": "stop",
            "finish_reason_explicit": True,
            "finish_reason_inference": "provider",
            "usage": None,
        }

    monkeypatch.setattr(unified, "_authnd_send", fake_authnd_send)
    unified, client = _make_missing_finish_client(
        monkeypatch, "authnd/moonshotai/kimi-k3",
    )
    request_id = "authnd-graceful-pending"
    unified._api_watchdog_reset()
    unified._api_watchdog_started(
        "glossary", request_id=request_id, chapter=12, queued=True,
    )
    outcome = {}

    def run_request():
        try:
            tls = client._get_thread_local_client()
            tls.current_request_id = request_id
            tls.pre_api_call_callback = lambda: progress_updates.append(True)
            outcome["result"] = client._send_authnd([], 0.2, 1024, "test")
        except Exception as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=run_request)
    worker.start()
    assert entered_preflight.wait(2.0)
    monkeypatch.setenv("GRACEFUL_STOP", "1")
    assert unified._api_watchdog_clear_pending_requests() == 1
    release_preflight.set()
    worker.join(2.0)

    assert not worker.is_alive()
    assert isinstance(outcome.get("error"), unified.UnifiedClientError)
    assert outcome["error"].error_type == "cancelled"
    assert preflight_cancel_checks == [True]
    assert progress_updates == []
    assert provider_posts == []
    assert unified.get_api_watchdog_state()["in_flight"] == 0
    unified._api_watchdog_reset()


def test_authnd_graceful_stop_preserves_claimed_provider_call(monkeypatch):
    import unified_api_client as unified

    monkeypatch.setenv("GRACEFUL_STOP", "0")
    monkeypatch.setenv("GRACEFUL_STOP_COMPLETED", "0")
    monkeypatch.setenv("MAX_RETRIES", "1")
    monkeypatch.setattr(unified, "_api_watchdog_external_write", lambda _state: None)
    provider_started = threading.Event()
    release_provider = threading.Event()
    progress_updates = []

    def fake_authnd_send(**kwargs):
        kwargs["before_send_callback"]()
        provider_started.set()
        assert release_provider.wait(2.0)
        # AuthND checks this repeatedly while consuming SSE chunks.  A graceful
        # stop must not turn it true after the provider boundary was claimed.
        assert kwargs["cancel_check"]() is False
        return {
            "content": "translated",
            "finish_reason": "stop",
            "finish_reason_explicit": True,
            "finish_reason_inference": "provider",
            "usage": None,
        }

    monkeypatch.setattr(unified, "_authnd_send", fake_authnd_send)
    unified, client = _make_missing_finish_client(
        monkeypatch, "authnd/moonshotai/kimi-k3",
    )
    request_id = "authnd-graceful-active"
    unified._api_watchdog_reset()
    unified._api_watchdog_started(
        "glossary", request_id=request_id, chapter=12, queued=True,
    )
    outcome = {}

    def run_request():
        try:
            tls = client._get_thread_local_client()
            tls.current_request_id = request_id
            tls.pre_api_call_callback = lambda: progress_updates.append(True)
            outcome["result"] = client._send_authnd([], 0.2, 1024, "test")
        except Exception as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=run_request)
    worker.start()
    assert provider_started.wait(2.0)
    monkeypatch.setenv("GRACEFUL_STOP", "1")
    assert unified._api_watchdog_clear_pending_requests() == 0
    assert unified.get_api_watchdog_state()["in_flight"] == 1
    release_provider.set()
    worker.join(2.0)

    assert not worker.is_alive()
    assert "error" not in outcome
    assert outcome["result"].content == "translated"
    assert progress_updates == [True]
    unified._api_watchdog_reset()


def test_graceful_stop_blocks_next_empty_truncation_retry(monkeypatch):
    """A claimed AuthND call may finish, but it must not launch retry N+1."""
    import unified_api_client as unified

    monkeypatch.setenv("GRACEFUL_STOP", "0")
    monkeypatch.setenv("GRACEFUL_STOP_COMPLETED", "0")
    monkeypatch.setenv("TRANSLATION_CANCELLED", "0")
    monkeypatch.setenv("RETRY_TRUNCATED", "1")
    monkeypatch.setenv("TRUNCATION_RETRY_ATTEMPTS", "3")
    unified, client = _make_missing_finish_client(
        monkeypatch, "authnd/moonshotai/kimi-k3",
    )
    provider_requests = []

    def fake_send_internal(**_kwargs):
        provider_requests.append(True)
        # This represents Stop being pressed while the already-claimed AuthND
        # stream is running. Its truncated result is still allowed to return.
        monkeypatch.setenv("GRACEFUL_STOP", "1")
        return "", "length"

    monkeypatch.setattr(client, "_send_internal", fake_send_internal)

    with pytest.raises(unified.UnifiedClientError) as exc_info:
        client._retry_empty_truncated_response(
            messages=[{"role": "user", "content": "translate"}],
            temperature=0.2,
            max_tokens=1024,
            max_completion_tokens=None,
            context="glossary",
            retry_reason=None,
            request_id="authnd-truncation-graceful-stop",
            image_data=None,
            finish_reason="length",
        )

    assert exc_info.value.error_type == "cancelled"
    assert provider_requests == [True]


@pytest.mark.parametrize(
    "provider,model,send_attr,result",
    MISSING_FINISH_PROVIDERS[:1],
)
def test_missing_finish_reason_toggle_on_routes_to_prohibited_content(
    monkeypatch,
    provider,
    model,
    send_attr,
    result,
):
    import unified_api_client as unified

    attempts = []

    def fake_send(**_kwargs):
        attempts.append(1)
        return dict(result)

    monkeypatch.setattr(unified, send_attr, fake_send)
    monkeypatch.setenv("MISSING_FINISH_AS_PROHIBITED", "1")
    monkeypatch.setenv("MAX_RETRIES", "3")
    monkeypatch.setenv("USE_FALLBACK_KEYS", "0")
    monkeypatch.setenv("DISABLE_REFUSAL_CHECKS", "1")
    unified, client = _make_missing_finish_client(monkeypatch, model)
    logs = []
    monkeypatch.setattr(
        unified,
        "print",
        lambda *values, **_kwargs: logs.append(" ".join(str(value) for value in values)),
        raising=False,
    )

    _content, finish_reason = client._send_internal(
        [{"role": "user", "content": "test"}],
        temperature=0.2,
        max_tokens=1024,
        context="translation",
        request_id=f"{provider}-missing-finish-toggle-on",
    )

    output = "\n".join(logs)
    assert finish_reason == "prohibited_content"
    assert len(attempts) == 1
    assert f"{provider}: missing finish_reason fallback triggered" in output.lower()
    assert "toggle is on" in output.lower()


@pytest.mark.parametrize(
    "provider,model,send_attr,result",
    MISSING_FINISH_PROVIDERS[:1],
)
def test_missing_finish_reason_toggle_off_is_api_error_and_globally_retried(
    monkeypatch,
    provider,
    model,
    send_attr,
    result,
):
    import unified_api_client as unified

    attempts = []

    def fake_send(**_kwargs):
        attempts.append(1)
        return dict(result)

    monkeypatch.setattr(unified, send_attr, fake_send)
    monkeypatch.setenv("MISSING_FINISH_AS_PROHIBITED", "0")
    monkeypatch.setenv("MAX_RETRIES", "3")
    monkeypatch.setenv("USE_FALLBACK_KEYS", "0")
    monkeypatch.setenv("DISABLE_REFUSAL_CHECKS", "1")
    unified, client = _make_missing_finish_client(monkeypatch, model)
    logs = []
    monkeypatch.setattr(
        unified,
        "print",
        lambda *values, **_kwargs: logs.append(" ".join(str(value) for value in values)),
        raising=False,
    )

    with pytest.raises(unified.UnifiedClientError) as exc_info:
        if provider == "ocagy":
            client._send_ocagy([], 0.2, 1024, "test")
        else:
            client._send_authnd([], 0.2, 1024, "test")
    assert exc_info.value.error_type == "api_error"
    attempts.clear()
    logs.clear()

    _content, finish_reason = client._send_internal(
        [{"role": "user", "content": "test"}],
        temperature=0.2,
        max_tokens=1024,
        context="translation",
        request_id=f"{provider}-missing-finish-toggle-off",
    )

    output = "\n".join(logs)
    assert finish_reason == "error"
    assert len(attempts) == 3
    assert f"{provider}: missing finish_reason fallback triggered" in output.lower()
    assert "toggle is off" in output.lower()
    assert "api_error for global retry" in output.lower()
    assert "server error (api error)" in output.lower()


def test_authnd_completed_done_stream_is_accepted_without_duplicate_retry(
    monkeypatch,
):
    import unified_api_client as unified

    attempts = []

    def fake_authnd_send(**_kwargs):
        attempts.append(1)
        return dict(MISSING_FINISH_PROVIDERS[1][3])

    monkeypatch.setattr(unified, "_authnd_send", fake_authnd_send)
    monkeypatch.setenv("MISSING_FINISH_AS_PROHIBITED", "0")
    monkeypatch.setenv("MAX_RETRIES", "3")
    monkeypatch.setenv("USE_FALLBACK_KEYS", "0")
    monkeypatch.setenv("DISABLE_REFUSAL_CHECKS", "1")
    unified, client = _make_missing_finish_client(
        monkeypatch, "authnd/z-ai/glm-5.1",
    )

    content, finish_reason = client._send_internal(
        [{"role": "user", "content": "Hi"}],
        temperature=0.2,
        max_tokens=1024,
        context="translation",
        request_id="authnd-completed-done",
    )

    assert content == "translated"
    assert finish_reason == "stop"
    assert attempts == [1]


def test_authnd_interruptible_transport_retains_caller_thread_identity(monkeypatch):
    import authnd_auth

    monkeypatch.delenv("TRANSLATION_CANCELLED", raising=False)
    monkeypatch.delenv("GRACEFUL_STOP", raising=False)
    authnd_auth.reset_cancel()
    observed_names = []

    @authnd_auth._interruptible_transport
    def fake_transport():
        observed_names.append(threading.current_thread().name)
        return "ok"

    caller_name = threading.current_thread().name
    assert fake_transport() == "ok"
    assert observed_names == [f"AuthNDTransport[{caller_name}]"]


def test_direct_text_maps_authnd_transport_logs_to_api_worker():
    import translator_gui

    resolve = translator_gui._InputOutputDialog._thread_key_from_log
    source = "AuthNDTransport[Thread-2 (api_call)]"

    assert resolve("🧠 [authnd] Thinking...", source) == "Thread-2 (api_call)"
    assert resolve("translated text", source) == "Thread-2 (api_call)"
    assert (
        resolve("translated text", "AuthNDTransport[TranslationWorker_7]")
        == "TranslationWorker_7"
    )


def test_direct_text_displays_authnd_queue_state_on_request_card():
    import translator_gui

    dialog_type = translator_gui._InputOutputDialog
    segment = {"phase": "processing", "complete": False}

    class Harness:
        _DIRECT_RESPONSE_PAYLOAD_PREFIX = dialog_type._DIRECT_RESPONSE_PAYLOAD_PREFIX
        _DIRECT_GLOSSARY_STREAM_START_PREFIX = (
            dialog_type._DIRECT_GLOSSARY_STREAM_START_PREFIX
        )
        _HIDDEN_STREAM_START_LOG_PHRASES = (
            dialog_type._HIDDEN_STREAM_START_LOG_PHRASES
        )
        _STREAM_END_LOG_PHRASES = dialog_type._STREAM_END_LOG_PHRASES
        _PIPELINE_LOG_PHRASES = dialog_type._PIPELINE_LOG_PHRASES
        _stream_phase_by_thread = {}

        @staticmethod
        def _request_segment_for_thread(_source_thread=None, create=True):
            assert create is True
            return segment

    kind = dialog_type._classify_line(
        Harness(),
        "⏳ AuthND: NVIDIA queue / prefill — response headers received "
        "in 12.4s; waiting for first token",
        "Thread-2 (api_call)",
        channel_hint="processing",
    )

    assert kind == "log"
    assert segment["phase"] == "queue"
    assert segment["status_label"] == (
        "NVIDIA queue / prefill · Headers in 12.4s · Waiting for first token"
    )


@pytest.mark.parametrize(
    "provider,model,send_attr,result",
    MISSING_FINISH_PROVIDERS,
)
def test_explicit_unknown_finish_reason_is_not_treated_as_missing(
    monkeypatch,
    provider,
    model,
    send_attr,
    result,
):
    import unified_api_client as unified

    explicit_result = dict(result)
    explicit_result["finish_reason"] = "unknown"
    if provider == "ocagy":
        explicit_result["finish_reason_fallback"] = False
    else:
        explicit_result["finish_reason_explicit"] = True
        explicit_result["finish_reason_inference"] = "provider"

    monkeypatch.setattr(unified, send_attr, lambda **_kwargs: explicit_result)
    monkeypatch.setenv("MISSING_FINISH_AS_PROHIBITED", "1")
    _unified, client = _make_missing_finish_client(monkeypatch, model)

    if provider == "ocagy":
        response = client._send_ocagy([], 0.2, 1024, "test")
    else:
        response = client._send_authnd([], 0.2, 1024, "test")

    assert response.finish_reason == "unknown"


def test_translation_environment_exports_disable_empty_safety_toggle():
    gui_source = desktop_gui_source()
    environment_builder = gui_source.split(
        "    def _get_environment_variables(self, epub_path, api_key):",
        1,
    )[1].split("\n    def ", 1)[0]

    assert "'DISABLE_EMPTY_SAFETY_HEURISTIC':" in environment_builder

    # U2: the glossary run builds env_updates in the split-out _build_glossary_extraction_env
    glossary_environment_builder = gui_source.split(
        "    def _build_glossary_extraction_env(",
        1,
    )[1].split("\n    def ", 1)[0]

    assert "'DISABLE_EMPTY_SAFETY_HEURISTIC':" in glossary_environment_builder


@pytest.mark.parametrize("explicit_finish_reason", ("stop", "unknown"))
def test_authnd_explicit_empty_finish_is_not_prohibited_when_empty_safety_disabled(
    monkeypatch,
    explicit_finish_reason,
):
    import unified_api_client as unified

    attempts = []

    def fake_authnd_send(**_kwargs):
        attempts.append(1)
        return {
            "content": "",
            "finish_reason": explicit_finish_reason,
            "finish_reason_explicit": True,
            "finish_reason_inference": "provider",
            "usage": None,
            "reasoning_content": "Enkidu blocked Sieg's hand-blade.",
            "raw_response": {
                "choices": [{
                    "delta": {
                        "reasoning_content": "Enkidu blocked Sieg's hand-blade.",
                        "content": "",
                    },
                    "finish_reason": explicit_finish_reason,
                }],
            },
        }

    monkeypatch.setattr(unified, "_authnd_send", fake_authnd_send)
    monkeypatch.setenv("MISSING_FINISH_AS_PROHIBITED", "0")
    monkeypatch.setenv("DISABLE_EMPTY_SAFETY_HEURISTIC", "1")
    monkeypatch.setenv("MAX_RETRIES", "3")
    monkeypatch.setenv("USE_FALLBACK_KEYS", "0")
    monkeypatch.setenv("DISABLE_REFUSAL_CHECKS", "1")
    unified, client = _make_missing_finish_client(
        monkeypatch,
        "authnd/deepseek-ai/deepseek-v4-flash",
    )
    monkeypatch.setattr(client, "_dump_raw_response", lambda *args, **kwargs: None)
    monkeypatch.setattr(client, "_log_truncation_failure", lambda *args, **kwargs: None)
    monkeypatch.setattr(client, "_attach_usage_to_last_payload", lambda *args, **kwargs: None)
    monkeypatch.setattr(client, "_handle_empty_result", lambda *args, **kwargs: "[EMPTY]")
    logs = []
    monkeypatch.setattr(
        unified,
        "print",
        lambda *values, **_kwargs: logs.append(" ".join(str(value) for value in values)),
        raising=False,
    )

    content, finish_reason = client._send_internal(
        [{"role": "user", "content": "test"}],
        temperature=0.2,
        max_tokens=1024,
        context="translation",
        request_id=f"authnd-empty-explicit-{explicit_finish_reason}",
    )

    assert content == "[EMPTY]"
    assert finish_reason == "error"
    assert len(attempts) == 3
    output = "\n".join(logs).lower()
    assert "routing api_error through global retry logic" in output
    assert "global api_error retries exhausted after 3 attempt(s)" in output
    assert "qa issue: api_error" in output
    assert "prohibited content detected" not in output


def test_finish_reason_error_uses_global_retries_then_returns_api_error_for_qa(
    monkeypatch,
):
    import unified_api_client as unified

    attempts = []

    def fake_response(*_args, **_kwargs):
        attempts.append(1)
        return unified.UnifiedResponse(
            content="partial provider response",
            finish_reason="error",
            raw_response={"finish_reason": "error"},
        )

    monkeypatch.setenv("MAX_RETRIES", "3")
    monkeypatch.setenv("DISABLE_REFUSAL_CHECKS", "1")
    unified, client = _make_missing_finish_client(
        monkeypatch,
        "ocagy/gemini-3.1-pro-high",
    )
    monkeypatch.setattr(client, "_get_response", fake_response)
    monkeypatch.setattr(client, "_save_response", lambda *args, **kwargs: None)
    monkeypatch.setattr(client, "_attach_usage_to_last_payload", lambda *args, **kwargs: None)
    logs = []
    monkeypatch.setattr(
        unified,
        "print",
        lambda *values, **_kwargs: logs.append(" ".join(str(value) for value in values)),
        raising=False,
    )

    content, finish_reason = client._send_internal(
        [{"role": "user", "content": "test"}],
        temperature=0.2,
        max_tokens=1024,
        context="translation",
        request_id="explicit-finish-reason-error",
    )

    output = "\n".join(logs).lower()
    assert content == "partial provider response"
    assert finish_reason == "error"
    assert len(attempts) == 3
    assert "server error (api error)" in output
    assert "global api_error retries exhausted after 3 attempt(s)" in output
    assert "qa issue: api_error" in output


@pytest.mark.parametrize("explicit_finish_reason", ("stop", "unknown"))
def test_authnd_explicit_empty_finish_is_prohibited_when_empty_safety_enabled(
    monkeypatch,
    explicit_finish_reason,
):
    import unified_api_client as unified

    monkeypatch.setattr(
        unified,
        "_authnd_send",
        lambda **_kwargs: {
            "content": "",
            "finish_reason": explicit_finish_reason,
            "finish_reason_explicit": True,
            "finish_reason_inference": "provider",
            "usage": None,
            "raw_response": {"finish_reason": explicit_finish_reason},
        },
    )
    monkeypatch.setenv("MISSING_FINISH_AS_PROHIBITED", "0")
    monkeypatch.setenv("DISABLE_EMPTY_SAFETY_HEURISTIC", "0")
    monkeypatch.setenv("MAX_RETRIES", "1")
    monkeypatch.setenv("USE_FALLBACK_KEYS", "0")
    monkeypatch.setenv("DISABLE_REFUSAL_CHECKS", "1")
    unified, client = _make_missing_finish_client(
        monkeypatch,
        "authnd/deepseek-ai/deepseek-v4-flash",
    )
    monkeypatch.setattr(client, "_dump_raw_response", lambda *args, **kwargs: None)
    monkeypatch.setattr(client, "_log_truncation_failure", lambda *args, **kwargs: None)
    monkeypatch.setattr(client, "_attach_usage_to_last_payload", lambda *args, **kwargs: None)
    monkeypatch.setattr(client, "_handle_empty_result", lambda *args, **kwargs: "[EMPTY]")

    content, finish_reason = client._send_internal(
        [{"role": "user", "content": "test"}],
        temperature=0.2,
        max_tokens=1024,
        context="translation",
        request_id=f"authnd-empty-safety-enabled-{explicit_finish_reason}",
    )

    assert content == "[EMPTY]"
    assert finish_reason == "content_filter"


def test_authnd_empty_inferred_stop_toggle_off_is_api_error_not_prohibited(
    monkeypatch,
):
    import unified_api_client as unified

    attempts = []

    def fake_authnd_send(**_kwargs):
        attempts.append(1)
        return {
            "content": "",
            "finish_reason": "stop",
            "finish_reason_explicit": False,
            "finish_reason_inference": "done_without_finish_reason",
            "usage": None,
            "raw_response": {"finish_reason": None},
        }

    monkeypatch.setattr(unified, "_authnd_send", fake_authnd_send)
    monkeypatch.setenv("MISSING_FINISH_AS_PROHIBITED", "0")
    monkeypatch.setenv("DISABLE_EMPTY_SAFETY_HEURISTIC", "1")
    monkeypatch.setenv("MAX_RETRIES", "3")
    monkeypatch.setenv("USE_FALLBACK_KEYS", "0")
    monkeypatch.setenv("DISABLE_REFUSAL_CHECKS", "1")
    unified, client = _make_missing_finish_client(
        monkeypatch,
        "authnd/deepseek-ai/deepseek-v4-flash",
    )
    logs = []
    monkeypatch.setattr(
        unified,
        "print",
        lambda *values, **_kwargs: logs.append(" ".join(str(value) for value in values)),
        raising=False,
    )

    _content, finish_reason = client._send_internal(
        [{"role": "user", "content": "test"}],
        temperature=0.2,
        max_tokens=1024,
        context="translation",
        request_id="authnd-empty-inferred-stop",
    )

    output = "\n".join(logs).lower()
    assert finish_reason == "error"
    assert len(attempts) == 3
    assert "api_error for global retry" in output
    assert "prohibited content detected" not in output


def test_empty_safety_toggle_applies_to_providers_outside_old_allowlist(
    monkeypatch,
):
    import unified_api_client as unified

    client = object.__new__(unified.UnifiedClient)
    response = unified.UnifiedResponse(
        content="",
        finish_reason="stop",
        raw_response={"finish_reason": "stop"},
    )
    monkeypatch.setenv("MISSING_FINISH_AS_PROHIBITED", "0")

    monkeypatch.setenv("DISABLE_EMPTY_SAFETY_HEURISTIC", "0")
    assert (
        client._detect_safety_filter(
            [],
            "",
            "stop",
            response,
            "future-provider",
        )
        is True
    )

    monkeypatch.setenv("DISABLE_EMPTY_SAFETY_HEURISTIC", "1")
    assert (
        client._detect_safety_filter(
            [],
            "",
            "stop",
            response,
            "future-provider",
        )
        is False
    )


def test_empty_safety_toggle_defaults_to_enabled(monkeypatch):
    import unified_api_client as unified

    client = object.__new__(unified.UnifiedClient)
    response = unified.UnifiedResponse(
        content="",
        finish_reason="stop",
        raw_response={"finish_reason": "stop"},
    )
    monkeypatch.delenv("DISABLE_EMPTY_SAFETY_HEURISTIC", raising=False)
    monkeypatch.setenv("MISSING_FINISH_AS_PROHIBITED", "0")

    assert (
        client._detect_safety_filter(
            [],
            "",
            "stop",
            response,
            "future-provider",
        )
        is False
    )

# --- Plugin auto-update, plugin-declared models and quota grouping ---------
# ocagy/ keeps opencode-antigravity-auth (and, when the plugin needs it, the
# OpenCode CLI) current without manual steps; models and quota follow the plugin.

NAME = ocagy_cli._PLUGIN_NAME
# The real functions, captured before any test replaces them.
_REAL_INSTALL_CANDIDATE = ocagy_cli._install_plugin_candidate
_REAL_VERIFY = ocagy_cli._verify_auth_plugin
_REAL_OPENCODE_VERSION = ocagy_cli._opencode_version
_REAL_START_BACKGROUND = ocagy_cli._start_background_plugin_update
_REAL_UPDATE_AUTH_PLUGIN = ocagy_cli._update_auth_plugin


def _fake_plugin(cache: Path, spec: str, version: str, *, readme: str = "", models_js: str = "", models_v2_js: str = "") -> Path:
    root = cache / "opencode" / "packages" / spec / "node_modules" / NAME
    for relative in ("dist/src/plugin/storage.js", "dist/src/plugin/auth.js", "dist/src/antigravity/oauth.js"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// fake\n", encoding="utf-8")
    (root / "package.json").write_text(json.dumps({"name": NAME, "version": version}), encoding="utf-8")
    if readme:
        (root / "README.md").write_text(readme, encoding="utf-8")
    config = root / "dist" / "src" / "plugin" / "config"
    if models_js:
        config.mkdir(parents=True, exist_ok=True)
        (config / "models.js").write_text(models_js, encoding="utf-8")
    if models_v2_js:
        config.mkdir(parents=True, exist_ok=True)
        (config / "models-v2.js").write_text(models_v2_js, encoding="utf-8")
    return root


README_2X = "OpenCode 1.18.29+ remains supported through the legacy entrypoint"


@pytest.fixture
def env(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    monkeypatch.setenv("OCAGY_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(ocagy_cli, "_PLUGIN_PACKAGE_OVERRIDE", "")
    monkeypatch.setattr(ocagy_cli, "_start_background_plugin_update", lambda *_a, **_k: None)
    patched = []
    import proxy_token_storage

    monkeypatch.setattr(proxy_token_storage, "patch_ocagy", lambda root: patched.append(Path(root)))
    monkeypatch.setattr(ocagy_cli, "_opencode_version", lambda _exe: (1, 19, 5))
    monkeypatch.setattr(ocagy_cli, "_verify_auth_plugin", lambda *_a, **_k: (True, ""))
    installs = []
    readme = {"text": README_2X}

    def fake_install(executable, spec, logger, **_kwargs):
        installs.append(spec)
        _fake_plugin(cache, spec, spec.rsplit("@", 1)[1], readme=readme["text"])
        return ""

    monkeypatch.setattr(ocagy_cli, "_install_plugin_candidate", fake_install)
    # `npm prefix -g` of the stand-in npm: the layout of _npm_global_opencode().
    monkeypatch.setattr(ocagy_cli, "_npm_global_prefix", lambda _npm: tmp_path / "npm-prefix")
    # An existing user: OpenCode resolved @latest once, to 1.6.0, and kept it.
    _fake_plugin(cache, f"{NAME}@latest", "1.6.0")
    return {"cache": cache, "patched": patched, "installs": installs, "tmp": tmp_path, "readme": readme}


def _latest(version):
    return lambda *_a, **_k: version


def _npm_global_opencode(tmp_path: Path) -> str:
    """An executable laid out like npm's global opencode-ai install."""
    exe = tmp_path / "npm-prefix" / "node_modules" / "opencode-ai" / "bin" / "opencode"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text("", encoding="utf-8")
    return str(exe)


def test_cached_latest_is_detected_as_installed_version(env):
    assert ocagy_cli._plugin_spec() == f"{NAME}@latest"
    assert ocagy_cli._installed_plugin_version(ocagy_cli._plugin_spec()) == "1.6.0"


def test_registry_unreachable_keeps_installed_plugin_and_logs_one_line(env, monkeypatch):
    def offline(*_a, **_k):
        raise OSError("network is unreachable")

    monkeypatch.setattr(ocagy_cli, "_fetch_latest_plugin_version", offline)
    logs = []

    assert ocagy_cli._update_auth_plugin("opencode", logs.append) is False

    assert len(logs) == 1 and "could not check npm" in logs[0] and "1.6.0" in logs[0]
    assert env["installs"] == []
    assert ocagy_cli._plugin_spec() == f"{NAME}@latest"
    state = ocagy_cli._read_plugin_state()
    assert state["check_failed_at"] > 0
    # No second network attempt within the retry window.
    assert ocagy_cli._plugin_update_due(state, state["check_failed_at"] + 60) is False


def test_same_version_does_nothing(env, monkeypatch):
    monkeypatch.setattr(ocagy_cli, "_fetch_latest_plugin_version", _latest("1.6.0"))
    logs = []

    assert ocagy_cli._update_auth_plugin("opencode", logs.append) is False

    assert logs == [] and env["installs"] == []
    state = ocagy_cli._read_plugin_state()
    assert state["version"] == "1.6.0"
    assert ocagy_cli._plugin_update_due(state, state["checked_at"] + 3600) is False
    assert ocagy_cli._plugin_update_due(state, state["checked_at"] + ocagy_cli.OCAGY_PLUGIN_UPDATE_INTERVAL_SECONDS) is True


def test_newer_version_is_pinned_patched_and_logged(env, monkeypatch):
    monkeypatch.setattr(ocagy_cli, "_fetch_latest_plugin_version", _latest("2.1.2"))
    logs = []

    assert ocagy_cli._update_auth_plugin("opencode", logs.append) is True

    assert env["installs"] == [f"{NAME}@2.1.2"]
    assert env["patched"] == [ocagy_cli._plugin_root(f"{NAME}@2.1.2")]
    assert ocagy_cli._plugin_spec() == f"{NAME}@2.1.2"
    config = json.loads((env["tmp"] / "workspace" / "opencode.json").read_text(encoding="utf-8"))
    assert config["plugin"] == [f"{NAME}@2.1.2"]
    assert any(f"updated {NAME} 1.6.0 → 2.1.2" in line for line in logs)
    # The previous version stays cached for a rollback without a download.
    assert ocagy_cli._installed_plugin_version(f"{NAME}@latest") == "1.6.0"
    assert ocagy_cli._read_plugin_state()["previous_spec"] == f"{NAME}@latest"


def test_hook_incompatible_version_is_not_used(env, monkeypatch):
    import proxy_token_storage

    def incompatible(_root):
        raise RuntimeError("OcAgy credential adapter is incompatible; refusing plaintext storage")

    monkeypatch.setattr(proxy_token_storage, "patch_ocagy", incompatible)
    monkeypatch.setattr(ocagy_cli, "_fetch_latest_plugin_version", _latest("3.0.0"))
    logs = []

    assert ocagy_cli._update_auth_plugin("opencode", logs.append) is False

    assert ocagy_cli._plugin_spec() == f"{NAME}@latest"
    assert not ocagy_cli._plugin_package_dir(f"{NAME}@3.0.0").exists()
    assert any("storage hook" in line and "Keeping 1.6.0" in line for line in logs)
    state = ocagy_cli._read_plugin_state()
    assert ocagy_cli._plugin_version_rejected(state, "3.0.0", state["checked_at"] + 30 * 86400)


def test_failed_check_after_switch_restores_previous_setup(env, monkeypatch):
    monkeypatch.setattr(ocagy_cli, "_fetch_latest_plugin_version", _latest("2.1.2"))
    monkeypatch.setattr(ocagy_cli, "_verify_auth_plugin", lambda *_a, **_k: (False, "plugin crashed on load"))
    logs = []

    assert ocagy_cli._update_auth_plugin("opencode", logs.append) is False

    assert ocagy_cli._plugin_spec() == f"{NAME}@latest"
    config = json.loads((env["tmp"] / "workspace" / "opencode.json").read_text(encoding="utf-8"))
    assert config["plugin"] == [f"{NAME}@latest"]
    assert any("restored 1.6.0" in line for line in logs)


def test_failed_download_keeps_old_setup_and_retries_later(env, monkeypatch):
    monkeypatch.setattr(ocagy_cli, "_fetch_latest_plugin_version", _latest("2.1.2"))
    monkeypatch.setattr(ocagy_cli, "_install_plugin_candidate", lambda *_a, **_k: "timed out after 900 s")
    logs = []

    assert ocagy_cli._update_auth_plugin("opencode", logs.append) is False

    assert ocagy_cli._plugin_spec() == f"{NAME}@latest"
    assert any("could not download" in line and "Keeping 1.6.0" in line for line in logs)
    state = ocagy_cli._read_plugin_state()
    assert not ocagy_cli._plugin_version_rejected(state, "2.1.2", state["check_failed_at"])
    assert ocagy_cli._plugin_update_due(state, state["check_failed_at"] + ocagy_cli.OCAGY_PLUGIN_UPDATE_RETRY_SECONDS)


def test_old_cli_is_updated_by_the_background_check(env, monkeypatch):
    versions = {"cli": (1, 18, 16)}
    monkeypatch.setattr(ocagy_cli, "_opencode_version", lambda _exe: versions["cli"])
    monkeypatch.setattr(ocagy_cli, "_fetch_latest_plugin_version", _latest("2.1.2"))
    monkeypatch.setattr(ocagy_cli, "_npm_for_global_opencode", lambda _exe: "npm")
    majors = []

    def download(npm, major):
        # The slow download runs while OpenCode can still start.
        assert not ocagy_cli._LAUNCH_LOCAL_LOCK.held_exclusive_by_me()
        majors.append(major)
        return "1.19.5", ""

    monkeypatch.setattr(ocagy_cli, "_download_opencode_cli", download)
    monkeypatch.setattr(ocagy_cli, "find_executable", lambda *_a: "opencode")
    commands = []

    def fake_setup(command, **kwargs):
        # No OpenCode process starts while npm replaces the CLI.
        assert ocagy_cli._LAUNCH_LOCAL_LOCK.held_exclusive_by_me()
        assert kwargs["timeout"] == ocagy_cli.OCAGY_CLI_UPDATE_TIMEOUT_SECONDS
        commands.append(command)
        versions["cli"] = (1, 19, 5)
        return {"ok": True, "output": ""}

    monkeypatch.setattr(ocagy_cli, "_run_setup_command", fake_setup)
    logs = []

    assert ocagy_cli._update_auth_plugin("opencode", logs.append) is True

    assert majors == [1]  # stays within the installed major
    # The global CLI is replaced offline, from what the download put in npm's cache.
    assert commands == [[
        "npm", "install", "-g", "--offline", "--no-audit", "--no-fund", "--allow-scripts=opencode-ai", "opencode-ai@1.19.5"
    ]]
    assert ocagy_cli._plugin_spec() == f"{NAME}@2.1.2"
    assert not ocagy_cli._LAUNCH_LOCAL_LOCK.held_exclusive_by_me()
    assert any("needs OpenCode 1.18.29 or newer (installed: 1.18.16)" in line for line in logs)
    assert any("OpenCode CLI updated 1.18.16 → 1.19.5" in line for line in logs)


def _fake_npm_runs(monkeypatch, *, npm_returncode=0, download_returncode=0, versions=None):
    """Replace run_logged_subprocess with npm / OpenCode stand-ins that print like the real tools."""
    calls = []
    versions = versions if versions is not None else {"cli": "1.18.16"}

    def fake(command, *, log_fn, timeout, cwd=None, env=None, **_kwargs):
        calls.append(list(command))
        logger = log_fn or (lambda _m: None)
        if "install" in command and "--prefix" in command:  # the download into a throw-away folder
            for line in ["npm warn deprecated something", "added 3 packages in 40s"]:
                logger(f"   ↳ {line}")
            if download_returncode:
                logger("   ↳ npm error code ETIMEDOUT")
                return {"returncode": download_returncode, "output": "npm error code ETIMEDOUT", "timed_out": False}
            package = Path(command[command.index("--prefix") + 1]) / "node_modules" / "opencode-ai" / "package.json"
            package.parent.mkdir(parents=True, exist_ok=True)
            package.write_text(json.dumps({"version": "1.19.5"}), encoding="utf-8")
            return {"returncode": 0, "output": "added 3 packages in 40s", "timed_out": False}
        if "install" in command:
            dump = [
                "npm error code EACCES",
                "npm error path C:\\Users\\someone\\AppData\\Roaming\\npm\\node_modules\\opencode-ai",
                "npm error A complete log of this run can be found in: C:\\Users\\someone\\AppData\\Local\\npm-cache\\_logs\\debug.log",
            ] if npm_returncode else ["changed 3 packages in 6s"]
            for line in dump:
                logger(f"   ↳ {line}")
            if not npm_returncode:
                versions["cli"] = "1.19.5"
            return {"returncode": npm_returncode, "output": "\n".join(dump), "timed_out": False}
        if command[1:] == ["--version"]:
            logger(f"   ↳ {versions['cli']}")
            return {"returncode": 0, "output": versions["cli"], "timed_out": False}
        # opencode models google: the whole catalog
        catalog = ["google/antigravity-gemini-3.8-flash-high", "google/gemini-2.5-flash", "google/gemma-4-31b-it"]
        for line in catalog:
            logger(f"   ↳ {line}")
        if env and env.get("OPENCODE_CONFIG"):  # the throw-away download run
            spec = json.loads(Path(env["OPENCODE_CONFIG"]).read_text(encoding="utf-8"))["plugin"][0]
            _fake_plugin(Path(env["XDG_CACHE_HOME"]), spec, spec.rsplit("@", 1)[1], readme=README_2X)
            return {"returncode": 1, "output": "Error: Provider not found: google", "timed_out": False}
        return {"returncode": 0, "output": "\n".join(catalog), "timed_out": False}

    monkeypatch.setattr(ocagy_cli, "run_logged_subprocess", fake)
    return calls


_LOG_LEAKS = ("↳", "npm error", "C:\\Users", "debug.log", "gemini-2.5-flash", "gemma", "Running:", "Provider not found")


def test_background_update_logs_one_short_line_per_event(env, monkeypatch, tmp_path):
    """The real download, version, npm and check runs, with stand-in tools: no output reaches the log."""
    exe = _npm_global_opencode(tmp_path)
    monkeypatch.setattr(ocagy_cli, "_install_plugin_candidate", _REAL_INSTALL_CANDIDATE)
    monkeypatch.setattr(ocagy_cli, "_verify_auth_plugin", _REAL_VERIFY)
    monkeypatch.setattr(ocagy_cli, "_opencode_version", _REAL_OPENCODE_VERSION)
    monkeypatch.setattr(ocagy_cli, "_candidate_command", lambda name: "npm" if name == "npm" else None)
    monkeypatch.setattr(ocagy_cli, "find_executable", lambda *_a: exe)
    monkeypatch.setattr(ocagy_cli, "_fetch_latest_plugin_version", _latest("2.1.2"))
    calls = _fake_npm_runs(monkeypatch)
    logs = []

    assert ocagy_cli._update_auth_plugin(exe, logs.append) is True

    download = next(call for call in calls if "--prefix" in call)
    assert download[-1] == "opencode-ai@1" and "--ignore-scripts" in download
    assert ["npm", "install", "-g", "--offline", "--no-audit", "--no-fund", "--allow-scripts=opencode-ai", "opencode-ai@1.19.5"] in calls
    assert calls.index(download) < calls.index(next(call for call in calls if "-g" in call))
    assert not Path(download[download.index("--prefix") + 1]).exists()  # throw-away folder removed
    assert len(logs) == 4, logs  # downloading, updating OpenCode, CLI updated, plugin updated
    assert logs[0].startswith("⬆️ OcAgy: downloading") and logs[-1].startswith("✅ OcAgy: updated")
    assert not [line for line in logs if any(leak in line for leak in _LOG_LEAKS)]


def test_failed_cli_update_logs_one_line_without_npm_output(env, monkeypatch, tmp_path):
    exe = _npm_global_opencode(tmp_path)
    monkeypatch.setattr(ocagy_cli, "_opencode_version", _REAL_OPENCODE_VERSION)
    monkeypatch.setattr(ocagy_cli, "_candidate_command", lambda name: "npm" if name == "npm" else None)
    monkeypatch.setattr(ocagy_cli, "find_executable", lambda *_a: exe)
    monkeypatch.setattr(ocagy_cli, "_fetch_latest_plugin_version", _latest("2.1.2"))
    calls = _fake_npm_runs(monkeypatch, npm_returncode=243)
    logs = []

    assert ocagy_cli._update_auth_plugin(exe, logs.append) is False

    assert sum("-g" in call for call in calls) == 1
    assert ocagy_cli._plugin_spec() == f"{NAME}@latest"
    failure = [line for line in logs if "did not reach 1.18.29" in line]
    assert len(failure) == 1 and "exited with code 243 (npm EACCES)" in failure[0] and "Keeping 1.6.0" in failure[0]
    assert "npm install -g opencode-ai@1" in failure[0]
    assert not [line for line in logs if any(leak in line for leak in _LOG_LEAKS)]
    state = ocagy_cli._read_plugin_state()
    # Not retried on the next start or within the retry hour, only with the next regular check.
    assert ocagy_cli._plugin_update_due(state, state["checked_at"] + ocagy_cli.OCAGY_PLUGIN_UPDATE_RETRY_SECONDS) is False
    assert ocagy_cli._plugin_update_due(state, state["checked_at"] + ocagy_cli.OCAGY_PLUGIN_UPDATE_INTERVAL_SECONDS) is True
    monkeypatch.setattr(ocagy_cli, "_verify_auth_plugin", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr(ocagy_cli, "_PLUGIN_BOOTSTRAPPED_EXECUTABLES", set())
    monkeypatch.setattr(ocagy_cli, "_secure_plugin_account_storage", lambda **_k: None)
    ocagy_cli.ensure_auth_plugin_installed(exe, log_fn=logs.append)
    assert sum("-g" in call for call in calls) == 1


def test_unexpected_errors_reach_the_log_without_local_paths(env, monkeypatch, tmp_path):
    home = str(Path.home())
    disk_full = OSError(28, "No space left on device")
    disk_full.filename = os.path.join(home, ".glossarion", "opencode_antigravity", "plugin-update-x")
    assert ocagy_cli._error_reason(disk_full) == "OSError: No space left on device"
    busy = PermissionError(13, "Permission denied", os.path.join(home, "cache", "storage.js"))
    assert ocagy_cli._error_reason(busy) == "PermissionError: Permission denied"
    assert "~" in ocagy_cli._short_reason(f"cannot read {home}{os.sep}x.json") and home not in ocagy_cli._short_reason(home)

    # The background thread's catch-all.
    monkeypatch.setattr(ocagy_cli, "_start_background_plugin_update", _REAL_START_BACKGROUND)
    monkeypatch.setattr(ocagy_cli, "_PLUGIN_UPDATE_THREAD", None)

    def explode(*_a, **_k):
        raise disk_full

    monkeypatch.setattr(ocagy_cli, "_update_auth_plugin", explode)
    logs = []
    ocagy_cli._start_background_plugin_update("opencode", logs.append)
    ocagy_cli._PLUGIN_UPDATE_THREAD.join(5)
    assert logs == ["⚠️ OcAgy: plugin update check failed: OSError: No space left on device"]

    # A storage-hook failure on a locked file.
    import proxy_token_storage

    def locked(_root):
        raise busy

    monkeypatch.setattr(ocagy_cli, "_update_auth_plugin", _REAL_UPDATE_AUTH_PLUGIN)
    monkeypatch.setattr(proxy_token_storage, "patch_ocagy", locked)
    monkeypatch.setattr(ocagy_cli, "_fetch_latest_plugin_version", _latest("2.1.2"))
    logs = []
    assert ocagy_cli._update_auth_plugin("opencode", logs.append) is False
    assert any("PermissionError: Permission denied" in line for line in logs)
    assert not [line for line in logs if home in line or "storage.js" in line]


def test_failed_cli_download_never_touches_the_installed_cli(env, monkeypatch, tmp_path):
    exe = _npm_global_opencode(tmp_path)
    monkeypatch.setattr(ocagy_cli, "_opencode_version", _REAL_OPENCODE_VERSION)
    monkeypatch.setattr(ocagy_cli, "_candidate_command", lambda name: "npm" if name == "npm" else None)
    monkeypatch.setattr(ocagy_cli, "find_executable", lambda *_a: exe)
    monkeypatch.setattr(ocagy_cli, "_fetch_latest_plugin_version", _latest("2.1.2"))
    calls = _fake_npm_runs(monkeypatch, download_returncode=1)
    logs = []

    assert ocagy_cli._update_auth_plugin(exe, logs.append) is False

    assert not [call for call in calls if "-g" in call]  # the global install never ran
    failure = [line for line in logs if "could not download the new OpenCode" in line]
    assert len(failure) == 1 and "exited with code 1" in failure[0] and "Keeping 1.6.0" in failure[0]
    assert not [line for line in logs if any(leak in line for leak in _LOG_LEAKS)]
    assert not list((env["tmp"] / "workspace").glob("cli-download-*"))


def test_cli_killed_half_way_logs_the_repair_command(env, monkeypatch, tmp_path):
    exe = _npm_global_opencode(tmp_path)
    versions = {"cli": (1, 18, 16)}
    monkeypatch.setattr(ocagy_cli, "_opencode_version", lambda _exe: versions["cli"])
    monkeypatch.setattr(ocagy_cli, "_candidate_command", lambda name: "npm" if name == "npm" else None)
    monkeypatch.setattr(ocagy_cli, "_download_opencode_cli", lambda _npm, _major: ("1.19.5", ""))
    monkeypatch.setattr(ocagy_cli, "find_executable", lambda *_a: exe)
    monkeypatch.setattr(ocagy_cli, "_fetch_latest_plugin_version", _latest("2.1.2"))

    def killed(command, **_kwargs):
        versions["cli"] = None  # npm moved the old CLI aside and was killed
        return {"ok": False, "reason": "timed out after 600 s", "output": "npm http fetch ..."}

    monkeypatch.setattr(ocagy_cli, "_run_setup_command", killed)
    logs = []

    assert ocagy_cli._update_auth_plugin(exe, logs.append) is False

    repair = [line for line in logs if "To repair it, run: npm install -g opencode-ai@1.19.5" in line]
    assert len(repair) == 1 and "timed out after 600 s" in repair[0]
    assert ocagy_cli._plugin_spec() == f"{NAME}@latest"


# --- A: the global OpenCode CLI is updated only when that is safe ----------


@pytest.mark.parametrize(
    "case, cli, readme, npm, npm_global, manual",
    [
        ("unreadable or pre-release version", None, README_2X, "npm", True, "npm install -g opencode-ai@1"),
        ("higher required major", (1, 19, 5), "OpenCode 2.0.0+ remains supported", "npm", True, "npm install -g opencode-ai@2"),
        ("not an npm-global install", (1, 18, 16), README_2X, "npm", False, "opencode upgrade"),
        # Without npm, Glossarion cannot tell how OpenCode was installed.
        ("npm is not available", (1, 18, 16), README_2X, None, True, "opencode upgrade"),
        ("installed by bun, npm on PATH", (1, 18, 16), README_2X, "npm", "bun", "opencode upgrade"),
    ],
)
def test_cli_is_left_alone_unless_the_update_is_safe(env, monkeypatch, tmp_path, case, cli, readme, npm, npm_global, manual):
    env["readme"]["text"] = readme
    if npm_global == "bun":
        # ~/.bun/bin/opencode -> ~/.bun/install/global/node_modules/opencode-ai/bin/opencode
        exe = tmp_path / "home" / ".bun" / "install" / "global" / "node_modules" / "opencode-ai" / "bin" / "opencode"
        exe.parent.mkdir(parents=True)
        exe.write_text("", encoding="utf-8")
        exe = str(exe)
    else:
        exe = _npm_global_opencode(tmp_path) if npm_global else str(tmp_path / "elsewhere" / "opencode.exe")
    monkeypatch.setattr(ocagy_cli, "_opencode_version", lambda _exe: cli)
    monkeypatch.setattr(ocagy_cli, "_candidate_command", lambda name: npm if name == "npm" else None)
    # The install override is for a missing CLI only; it never runs as an update.
    monkeypatch.setenv("OCAGY_OPENCODE_INSTALL_CMD", "npm install -g opencode-ai@latest")
    monkeypatch.setattr(ocagy_cli, "_run_setup_command", lambda *_a, **_k: pytest.fail(f"command run: {case}"))
    monkeypatch.setattr(ocagy_cli, "_fetch_latest_plugin_version", _latest("2.1.2"))
    logs = []

    assert ocagy_cli._update_auth_plugin(exe, logs.append) is False

    assert ocagy_cli._plugin_spec() == f"{NAME}@latest"
    hint = [line for line in logs if "needs OpenCode" in line]
    assert len(hint) == 1 and f"run: {manual}" in hint[0] and "Keeping 1.6.0" in hint[0], logs
    # One line per check, not on every request: the next call within 12 h is silent.
    logs.clear()
    assert ocagy_cli._update_auth_plugin(exe, logs.append) is False
    assert logs == []


def test_cli_counts_as_npm_global_only_under_the_prefix_of_the_npm_that_would_update_it(env, monkeypatch, tmp_path):
    prefix = tmp_path / "npm-prefix"  # what `npm prefix -g` prints (env fixture)
    monkeypatch.setattr(ocagy_cli, "_candidate_command", lambda name: "npm" if name == "npm" else None)
    monkeypatch.setenv("OCAGY_OPENCODE_INSTALL_CMD", "my-installer --latest")

    def touch(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return str(path)

    # Windows layout: <prefix>/node_modules/opencode-ai, shim <prefix>/opencode.cmd
    assert ocagy_cli._npm_for_global_opencode(_npm_global_opencode(tmp_path)) == "npm"
    assert ocagy_cli._npm_for_global_opencode(touch(prefix / "opencode.cmd")) == "npm"
    native = prefix / "node_modules" / "opencode-ai" / "node_modules" / "opencode-windows-x64" / "bin" / "opencode.exe"
    assert ocagy_cli._npm_for_global_opencode(touch(native)) == "npm"
    # POSIX layout: <prefix>/lib/node_modules/opencode-ai (bin/opencode links into it)
    posix = tmp_path / "posix-prefix"
    monkeypatch.setattr(ocagy_cli, "_npm_global_prefix", lambda _npm: posix)
    assert ocagy_cli._npm_for_global_opencode(touch(posix / "lib" / "node_modules" / "opencode-ai" / "bin" / "opencode")) == "npm"
    assert ocagy_cli._npm_for_global_opencode(touch(posix / "bin" / "opencode")) == "npm"
    # Elsewhere: bun / yarn globals, another Node version's npm (nvm), the install script.
    for other in (
        tmp_path / "home" / ".bun" / "install" / "global" / "node_modules" / "opencode-ai" / "bin" / "opencode",
        tmp_path / "nvm" / "v18.20.0" / "node_modules" / "opencode-ai" / "bin" / "opencode",
        tmp_path / "home" / ".opencode" / "bin" / "opencode",
        tmp_path / "elsewhere" / "opencode.exe",
    ):
        assert ocagy_cli._npm_for_global_opencode(touch(other)) is None, other
    # npm prints a UUID-like folder name as "***": matched against the CLI's own folders.
    masked = tmp_path / "0f6b2a51-1a51-418f-a84d-a6f4a388cde5" / "prefix"
    monkeypatch.setattr(ocagy_cli, "_npm_global_prefix", lambda _npm: tmp_path / "***" / "prefix")
    assert ocagy_cli._npm_for_global_opencode(touch(masked / "node_modules" / "opencode-ai" / "bin" / "opencode")) == "npm"
    assert ocagy_cli._npm_for_global_opencode(touch(masked / "opencode.cmd")) == "npm"
    assert ocagy_cli._npm_for_global_opencode(touch(tmp_path / "other" / "x" / "node_modules" / "opencode-ai" / "bin" / "opencode")) is None
    # `npm prefix -g` failing, or no npm at all: never treated as npm-global.
    monkeypatch.setattr(ocagy_cli, "_npm_global_prefix", lambda _npm: None)
    assert ocagy_cli._npm_for_global_opencode(_npm_global_opencode(tmp_path)) is None
    monkeypatch.setattr(ocagy_cli, "_candidate_command", lambda name: None)
    assert ocagy_cli._npm_for_global_opencode(_npm_global_opencode(tmp_path)) is None


def test_npm_global_prefix_is_read_from_npm_without_logging(monkeypatch, tmp_path):
    prefix = tmp_path / "npm prefix"
    seen = {}

    def fake_setup(command, **kwargs):
        seen.update(command=command, **kwargs)
        return {"ok": True, "output": f"npm warn config some-option is deprecated\n{prefix}\n"}

    monkeypatch.setattr(ocagy_cli, "_run_setup_command", fake_setup)

    assert ocagy_cli._npm_global_prefix("npm") == prefix
    assert seen["command"] == ["npm", "prefix", "-g"] and seen["log_fn"] is ocagy_cli._quiet
    monkeypatch.setattr(ocagy_cli, "_run_setup_command", lambda *_a, **_k: {"ok": False, "reason": "timed out"})
    assert ocagy_cli._npm_global_prefix("npm") is None


@pytest.mark.parametrize(
    "output, expected",
    [
        ("1.18.16", (1, 18, 16)),
        ("opencode 1.19.5\n", (1, 19, 5)),
        ("0.0.0-dev-202611092046", None),
        ("0.0.0-beta-202608110357", None),
        ("2.0.0-beta.1", None),
        ("", None),
    ],
)
def test_opencode_version_accepts_only_stable_releases(monkeypatch, output, expected):
    monkeypatch.setattr(ocagy_cli, "_run_setup_command", lambda *_a, **_k: {"ok": True, "output": output})
    assert ocagy_cli._opencode_version("opencode") == expected


def test_opencode_version_timeout_is_unreadable(monkeypatch):
    monkeypatch.setattr(
        ocagy_cli,
        "_run_setup_command",
        lambda *_a, **_k: {"ok": False, "reason": "timed out after 30 s", "output": "1.18.16"},
    )
    assert ocagy_cli._opencode_version("opencode") is None


def test_first_install_never_runs_npm_on_the_request(env, monkeypatch, tmp_path):
    import shutil

    shutil.rmtree(env["cache"] / "opencode" / "packages")
    exe = _npm_global_opencode(tmp_path)
    monkeypatch.setattr(ocagy_cli, "_opencode_version", lambda _exe: (1, 18, 16))
    monkeypatch.setattr(ocagy_cli, "_candidate_command", lambda name: "npm" if name == "npm" else None)
    monkeypatch.setattr(ocagy_cli, "_run_setup_command", lambda *_a, **_k: pytest.fail("npm run on a request"))
    monkeypatch.setattr(ocagy_cli, "_fetch_latest_plugin_version", _latest("2.1.2"))
    logs = []

    assert ocagy_cli._update_auth_plugin(exe, logs.append, force=True) is False

    assert any("updates OpenCode in the background" in line for line in logs)
    state = ocagy_cli._read_plugin_state()
    # The request uses the patched download, not an unpatched @latest fetch.
    assert ocagy_cli._plugin_spec() == f"{NAME}@2.1.2" and env["patched"]
    assert state["cli_needed"] == {"version": "2.1.2", "required": "1.18.29"}
    # The background check that follows the first start is due at once.
    assert ocagy_cli._plugin_update_due(state, time.time())


def test_background_check_finishes_the_cli_update_a_first_install_deferred(env, monkeypatch, tmp_path):
    import shutil

    shutil.rmtree(env["cache"] / "opencode" / "packages")
    exe = _npm_global_opencode(tmp_path)
    monkeypatch.setattr(ocagy_cli, "_opencode_version", lambda _exe: (1, 18, 16))
    monkeypatch.setattr(ocagy_cli, "_candidate_command", lambda name: "npm" if name == "npm" else None)
    monkeypatch.setattr(ocagy_cli, "_fetch_latest_plugin_version", _latest("2.1.2"))
    assert ocagy_cli._update_auth_plugin(exe, lambda _m: None, force=True) is False

    calls = []
    outcome = {"result": None}

    def update_cli(executable, cli_version, required, target, current_text, logger, *, allow_update):
        calls.append((cli_version, required, target, allow_update))
        return outcome["result"]

    monkeypatch.setattr(ocagy_cli, "_update_opencode_cli", update_cli)
    # @latest is already the downloaded release, yet the CLI step runs.
    assert ocagy_cli._update_auth_plugin(exe, lambda _m: None) is False
    assert calls == [((1, 18, 16), (1, 18, 29), "2.1.2", True)]
    assert "cli_needed" in ocagy_cli._read_plugin_state()  # failed: tried again with the next check
    assert env["installs"] == [f"{NAME}@2.1.2"]  # no second download

    state = ocagy_cli._read_plugin_state()
    state.pop("checked_at")
    ocagy_cli._write_plugin_state(state)
    outcome["result"] = exe
    assert ocagy_cli._update_auth_plugin(exe, lambda _m: None) is False
    assert len(calls) == 2 and "cli_needed" not in ocagy_cli._read_plugin_state()
    assert ocagy_cli._plugin_spec() == f"{NAME}@2.1.2"


def test_first_install_with_a_cli_glossarion_cannot_update_logs_once(env, monkeypatch, tmp_path):
    import shutil

    shutil.rmtree(env["cache"] / "opencode" / "packages")
    monkeypatch.setattr(ocagy_cli, "_opencode_version", lambda _exe: None)  # e.g. a dev build
    monkeypatch.setattr(ocagy_cli, "_fetch_latest_plugin_version", _latest("2.1.2"))
    logs = []

    assert ocagy_cli._update_auth_plugin(str(tmp_path / "opencode.exe"), logs.append, force=True) is False

    assert len([line for line in logs if "needs OpenCode" in line]) == 1
    # The background check that follows does not repeat the same line.
    state = ocagy_cli._read_plugin_state()
    assert not ocagy_cli._plugin_update_due(state, time.time())
    assert "cli_needed" not in state and ocagy_cli._plugin_spec() == f"{NAME}@2.1.2"


def test_hook_rejection_is_void_after_the_hook_changes(env, monkeypatch):
    state = {}
    ocagy_cli._reject_plugin_version(state, "3.0.0", "incompatible", permanent=True)
    far_future = time.time() + 365 * 86400
    assert ocagy_cli._plugin_version_rejected(state, "3.0.0", far_future)

    # A Glossarion release with new anchors: the rejected version is tried again.
    monkeypatch.setattr(ocagy_cli, "_storage_hook_fingerprint", lambda: "new-anchors")
    assert not ocagy_cli._plugin_version_rejected(state, "3.0.0", far_future)


def test_storage_hook_fingerprint_follows_the_real_hook():
    import proxy_token_storage

    fingerprint = ocagy_cli._storage_hook_fingerprint()
    assert re.fullmatch(r"[0-9a-f]{16}", fingerprint)
    assert fingerprint == ocagy_cli._storage_hook_fingerprint()
    assert proxy_token_storage.patch_ocagy.__code__ is not None


def test_cli_new_enough_is_never_touched(env, monkeypatch):
    monkeypatch.setattr(ocagy_cli, "_opencode_version", lambda _exe: (1, 18, 32))
    monkeypatch.setattr(ocagy_cli, "_fetch_latest_plugin_version", _latest("2.1.2"))
    monkeypatch.setattr(ocagy_cli, "_update_opencode_cli", lambda *_a, **_k: pytest.fail("CLI update attempted"))

    assert ocagy_cli._update_auth_plugin("opencode", lambda _m: None) is True


def test_explicit_plugin_package_disables_auto_update(env, monkeypatch):
    monkeypatch.setattr(ocagy_cli, "_PLUGIN_PACKAGE_OVERRIDE", f"{NAME}@1.6.0")
    monkeypatch.setattr(ocagy_cli, "_fetch_latest_plugin_version", lambda *_a: pytest.fail("registry used"))

    assert ocagy_cli._update_auth_plugin("opencode", lambda _m: None, force=True) is False
    assert ocagy_cli._plugin_spec() == f"{NAME}@1.6.0"


def test_candidate_download_never_sees_the_real_account_store(tmp_path, monkeypatch):
    monkeypatch.setenv("OCAGY_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(tmp_path / "real-config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "real-data"))
    captured = {}

    def fake_setup(command, **kwargs):
        captured.update(kwargs)
        captured["config"] = json.loads((Path(kwargs["cwd"]) / "opencode.json").read_text(encoding="utf-8"))
        _fake_plugin(tmp_path / "cache", f"{NAME}@2.1.2", "2.1.2")
        return {"ok": True, "output": ""}

    monkeypatch.setattr(ocagy_cli, "_run_setup_command", fake_setup)

    assert ocagy_cli._install_plugin_candidate("opencode", f"{NAME}@2.1.2", lambda _m: None) == ""

    env = captured["env"]
    assert captured["config"]["plugin"] == [f"{NAME}@2.1.2"]
    assert "provider" not in captured["config"]
    assert str(tmp_path / "real-config") != env["OPENCODE_CONFIG_DIR"]
    assert str(tmp_path / "real-data") != env["XDG_DATA_HOME"]
    assert env["XDG_CACHE_HOME"] == str(tmp_path / "cache")  # installs into the real cache
    # Only the throw-away config is loaded, never the workspace opencode.json above the stage.
    assert env["OPENCODE_CONFIG"] == str(Path(captured["cwd"]) / "opencode.json")
    assert env["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"
    assert not Path(captured["cwd"]).exists()  # throw-away stage removed


def test_first_install_pins_newest_release_before_opencode_loads_it(env, monkeypatch):
    import shutil

    shutil.rmtree(env["cache"] / "opencode" / "packages")
    monkeypatch.setattr(ocagy_cli, "_fetch_latest_plugin_version", _latest("2.1.2"))
    monkeypatch.setattr(ocagy_cli, "_secure_plugin_account_storage", lambda **_k: None)
    executable = str(env["tmp"] / "opencode-first")
    ocagy_cli._PLUGIN_BOOTSTRAPPED_EXECUTABLES.discard(str(Path(executable).resolve(strict=False)).casefold())
    logs = []

    ocagy_cli.ensure_auth_plugin_installed(executable, log_fn=logs.append)

    assert env["installs"] == [f"{NAME}@2.1.2"]
    assert ocagy_cli._plugin_spec() == f"{NAME}@2.1.2"
    assert any("ready (opencode-antigravity-auth 2.1.2)" in line for line in logs)


# --- E: a wrong clock never blocks checks or keeps a rejection -------------


def test_timestamps_ahead_of_the_clock_are_invalid():
    now = time.time()
    next_year = now + 365 * 86400
    assert ocagy_cli._plugin_update_due({"checked_at": next_year}, now) is True
    assert ocagy_cli._plugin_update_due({"check_failed_at": next_year}, now) is True
    assert ocagy_cli._plugin_update_due({"checked_at": "garbage"}, now) is True
    state = {"rejected": {"2.1.2": {"at": next_year, "permanent": False}}}
    assert ocagy_cli._plugin_version_rejected(state, "2.1.2", now) is False
    # Within the small margin a timestamp still counts (clocks of two processes).
    assert ocagy_cli._plugin_update_due({"checked_at": now + 60}, now) is False
    assert ocagy_cli._plugin_version_rejected({"rejected": {"2.1.2": {"at": now + 60}}}, "2.1.2", now) is True


# --- B: every external process and every wait is bounded --------------------

_HEARTBEAT = (
    "import os, sys, time\n"
    "path = sys.argv[1]\n"
    "for n in range(100000):\n"
    "    with open(path, 'w') as stream:\n"
    "        stream.write(f'{os.getpid()} {n}')\n"
    "    time.sleep(0.1)\n"
)


def _shim_with_long_running_grandchild(tmp_path):
    """A command whose real worker is a grandchild that keeps the output pipe open.

    That is the shape of npm.cmd / opencode.cmd on Windows (cmd.exe -> node).
    """
    script = tmp_path / "heartbeat.py"
    script.write_text(_HEARTBEAT, encoding="utf-8")
    beat = tmp_path / "beat.txt"
    if os.name == "nt":
        shim = tmp_path / "slow-tool.cmd"
        shim.write_text(f'@echo started\r\n@"{sys.executable}" "{script}" "{beat}"\r\n', encoding="utf-8")
        return [str(shim)], beat
    shim = tmp_path / "slow-tool.sh"
    shim.write_text(f'#!/bin/sh\necho started\n"{sys.executable}" "{script}" "{beat}"\necho done\n', encoding="utf-8")
    shim.chmod(0o755)
    return [str(shim)], beat


def _assert_grandchild_stops(beat: Path) -> None:
    deadline = time.monotonic() + 5
    while not beat.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    time.sleep(0.5)
    first = beat.read_text(encoding="utf-8") if beat.exists() else ""
    time.sleep(1.0)
    second = beat.read_text(encoding="utf-8") if beat.exists() else ""
    assert first == second, "the grandchild is still running"


def test_setup_command_timeout_kills_the_whole_tree_within_the_grace_period(tmp_path):
    from installer_utils import KILL_GRACE_SECONDS

    command, beat = _shim_with_long_running_grandchild(tmp_path)
    started = time.monotonic()

    result = ocagy_cli._run_setup_command(command, action="Slow tool", timeout=2)

    elapsed = time.monotonic() - started
    assert result["ok"] is False and result["reason"] == "timed out after 2 s"
    assert elapsed < 2 + KILL_GRACE_SECONDS + 3, elapsed
    _assert_grandchild_stops(beat)


def test_setup_command_is_stopped_by_stop(tmp_path):
    command, beat = _shim_with_long_running_grandchild(tmp_path)
    stop_at = time.monotonic() + 1.5
    started = time.monotonic()

    result = ocagy_cli._run_setup_command(
        command, action="Slow tool", timeout=60, stop_check=lambda: time.monotonic() >= stop_at
    )

    assert result.get("cancelled") is True and result["reason"] == "stopped"
    assert time.monotonic() - started < 10
    _assert_grandchild_stops(beat)


def test_shared_runner_returns_after_the_grace_period_even_if_a_grandchild_survives(tmp_path):
    """Other routes' callers (no tree kill) are no longer held by a surviving grandchild."""
    from installer_utils import KILL_GRACE_SECONDS, run_logged_subprocess

    import signal

    command, beat = _shim_with_long_running_grandchild(tmp_path)
    started = time.monotonic()
    try:
        result = run_logged_subprocess(
            command,
            log_fn=None,
            timeout=1,
            popen_kwargs={"creationflags": ocagy_cli._creation_flags(), "start_new_session": os.name != "nt"},
        )
        assert result["timed_out"] is True
        assert time.monotonic() - started < 1 + KILL_GRACE_SECONDS + 3 + 3  # terminate waits up to 3 s
    finally:
        # The default stop only reaches the direct child: end the surviving grandchild.
        try:
            os.kill(int(beat.read_text(encoding="utf-8").split()[0]), signal.SIGTERM)
        except (OSError, ValueError, IndexError):
            pass


def test_install_candidate_on_a_request_is_stopped_by_stop(tmp_path):
    command, beat = _shim_with_long_running_grandchild(tmp_path)
    stop_at = time.monotonic() + 1.0
    started = time.monotonic()

    with pytest.raises(ocagy_cli.OcAgyError, match="cancelled"):
        _REAL_INSTALL_CANDIDATE(
            command[0], f"{NAME}@2.1.2", lambda _m: None, timeout=60,
            stop_check=lambda: time.monotonic() >= stop_at,
        )
    assert time.monotonic() - started < 10
    _assert_grandchild_stops(beat)


def _hold_lock_in_thread(filename, local, *, exclusive=True):
    held = threading.Event()
    release = threading.Event()

    def holder():
        with ocagy_cli._coordination_lock(filename, local, exclusive=exclusive, timeout=5):
            held.set()
            release.wait(30)

    thread = threading.Thread(target=holder, daemon=True)
    thread.start()
    assert held.wait(5)
    return release, thread


def _run_in_thread(fn):
    outcome = {}

    def target():
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - reported to the test
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return outcome, thread


def test_request_waiting_for_a_cli_update_is_released_by_stop(monkeypatch):
    monkeypatch.setattr(ocagy_cli, "find_executable", lambda *_a: "opencode")
    release, holder = _hold_lock_in_thread(ocagy_cli._LAUNCH_LOCK_FILENAME, ocagy_cli._LAUNCH_LOCAL_LOCK)
    try:
        outcome, worker = _run_in_thread(ocagy_cli.ensure_opencode_installed)
        worker.join(0.6)
        assert worker.is_alive()  # waits while the update runs
        stopped = time.monotonic()
        ocagy_cli.cancel_stream()
        worker.join(3)
        assert not worker.is_alive() and time.monotonic() - stopped < 2
        assert isinstance(outcome.get("error"), ocagy_cli.OcAgyError) and "cancelled by user" in str(outcome["error"])
    finally:
        ocagy_cli._CANCEL_EVENT.clear()
        release.set()
        holder.join(5)


def test_request_waiting_for_a_cli_update_gives_up_at_the_cap(monkeypatch):
    monkeypatch.setattr(ocagy_cli, "find_executable", lambda *_a: "opencode")
    monkeypatch.setattr(ocagy_cli, "OCAGY_LAUNCH_WAIT_SECONDS", 1.0)
    release, holder = _hold_lock_in_thread(ocagy_cli._LAUNCH_LOCK_FILENAME, ocagy_cli._LAUNCH_LOCAL_LOCK)
    try:
        started = time.monotonic()
        with pytest.raises(ocagy_cli.OcAgyError, match="still running"):
            ocagy_cli.ensure_opencode_installed()
        assert time.monotonic() - started < 3
    finally:
        release.set()
        holder.join(5)
    assert ocagy_cli.ensure_opencode_installed() == "opencode"  # free again


def test_request_waits_for_a_running_cli_update_then_starts(monkeypatch):
    monkeypatch.setattr(ocagy_cli, "find_executable", lambda *_a: "opencode")
    release, holder = _hold_lock_in_thread(ocagy_cli._LAUNCH_LOCK_FILENAME, ocagy_cli._LAUNCH_LOCAL_LOCK)
    outcome, worker = _run_in_thread(ocagy_cli.ensure_opencode_installed)
    worker.join(0.5)
    assert worker.is_alive()
    release.set()
    worker.join(5)
    holder.join(5)
    assert outcome.get("value") == "opencode"


def test_many_waiting_requests_log_one_waiting_line(monkeypatch):
    monkeypatch.setattr(ocagy_cli, "find_executable", lambda *_a: "opencode")
    monkeypatch.setattr(ocagy_cli, "_LAST_LAUNCH_WAIT_NOTICE", float("-inf"))
    logs = []
    release, holder = _hold_lock_in_thread(ocagy_cli._LAUNCH_LOCK_FILENAME, ocagy_cli._LAUNCH_LOCAL_LOCK)
    workers = [_run_in_thread(lambda: ocagy_cli.ensure_opencode_installed(log_fn=logs.append)) for _ in range(4)]
    time.sleep(0.6)
    release.set()
    for outcome, worker in workers:
        worker.join(5)
        assert outcome.get("value") == "opencode"
    holder.join(5)
    assert logs == ["⏳ OcAgy: waiting for the OpenCode update to finish..."]


def test_first_install_waiting_for_another_update_is_released_by_stop(env, monkeypatch):
    import shutil

    shutil.rmtree(env["cache"] / "opencode" / "packages")
    release, holder = _hold_lock_in_thread(ocagy_cli._UPDATE_LOCK_FILENAME, ocagy_cli._UPDATE_LOCAL_LOCK)
    try:
        outcome, worker = _run_in_thread(lambda: ocagy_cli._update_auth_plugin("opencode", lambda _m: None, force=True))
        worker.join(0.6)
        assert worker.is_alive()
        ocagy_cli.cancel_stream()
        worker.join(3)
        assert not worker.is_alive()
        assert "cancelled by user" in str(outcome.get("error"))
    finally:
        ocagy_cli._CANCEL_EVENT.clear()
        release.set()
        holder.join(5)
    # The background check never waits: it skips while another one runs.
    release, holder = _hold_lock_in_thread(ocagy_cli._UPDATE_LOCK_FILENAME, ocagy_cli._UPDATE_LOCAL_LOCK)
    try:
        started = time.monotonic()
        assert ocagy_cli._update_auth_plugin("opencode", lambda _m: None) is False
        assert time.monotonic() - started < 2
    finally:
        release.set()
        holder.join(5)


def test_plugin_setup_wait_is_released_by_stop_and_capped(monkeypatch):
    monkeypatch.setattr(ocagy_cli, "_PLUGIN_BOOTSTRAPPED_EXECUTABLES", set())
    ocagy_cli._PLUGIN_BOOTSTRAP_LOCK.acquire()
    try:
        outcome, worker = _run_in_thread(lambda: ocagy_cli.ensure_auth_plugin_installed("opencode"))
        worker.join(0.6)
        assert worker.is_alive()
        ocagy_cli.cancel_stream()
        worker.join(3)
        assert "cancelled by user" in str(outcome.get("error"))
        ocagy_cli._CANCEL_EVENT.clear()
        monkeypatch.setattr(ocagy_cli, "OCAGY_PLUGIN_SETUP_WAIT_SECONDS", 0.5)
        with pytest.raises(ocagy_cli.OcAgyError, match="did not finish within"):
            ocagy_cli.ensure_auth_plugin_installed("opencode")
    finally:
        ocagy_cli._CANCEL_EVENT.clear()
        ocagy_cli._PLUGIN_BOOTSTRAP_LOCK.release()


def test_unusable_lock_file_falls_back_to_this_process_and_logs_once(monkeypatch):
    def broken(_fd, _exclusive):
        raise OSError(38, "Function not implemented")

    monkeypatch.setattr(ocagy_cli, "_os_try_lock", broken)
    monkeypatch.setattr(ocagy_cli, "_LOCK_FALLBACK_LOGGED", set())
    monkeypatch.setattr(ocagy_cli, "find_executable", lambda *_a: "opencode")
    logs = []

    assert ocagy_cli.ensure_opencode_installed(log_fn=logs.append) == "opencode"
    assert ocagy_cli.ensure_opencode_installed(log_fn=logs.append) == "opencode"

    assert len(logs) == 1 and "cannot lock glossarion-launch.lock" in logs[0]


def test_missing_cli_install_is_coordinated_stop_aware_and_rechecked(monkeypatch):
    installed = {"value": False}
    runs = []

    def find(*_a):
        if installed["value"]:
            return "opencode"
        raise ocagy_cli.OcAgyError("not found")

    def fake_setup(command, **kwargs):
        # One installer, and no OpenCode start anywhere while it writes the CLI.
        assert ocagy_cli._UPDATE_LOCAL_LOCK.held_exclusive_by_me()
        assert ocagy_cli._LAUNCH_LOCAL_LOCK.held_exclusive_by_me()
        assert kwargs["stop_check"] is ocagy_cli.is_cancelled
        runs.append(command)
        installed["value"] = True
        return {"ok": True, "output": ""}

    monkeypatch.setattr(ocagy_cli, "find_executable", find)
    monkeypatch.setattr(ocagy_cli, "_opencode_install_command", lambda: ["npm", "install", "-g", "opencode-ai"])
    monkeypatch.setattr(ocagy_cli, "_run_setup_command", fake_setup)

    # A request waiting for another installer (or update) is released by Stop.
    release, holder = _hold_lock_in_thread(ocagy_cli._UPDATE_LOCK_FILENAME, ocagy_cli._UPDATE_LOCAL_LOCK)
    try:
        outcome, worker = _run_in_thread(ocagy_cli.ensure_opencode_installed)
        worker.join(0.6)
        assert worker.is_alive()
        stopped = time.monotonic()
        ocagy_cli.cancel_stream()
        worker.join(3)
        assert not worker.is_alive() and time.monotonic() - stopped < 2
        assert "cancelled by user" in str(outcome.get("error"))
        ocagy_cli._CANCEL_EVENT.clear()
        # ... and gives up at the cap with a readable error.
        monkeypatch.setattr(ocagy_cli, "OCAGY_CLI_INSTALL_WAIT_SECONDS", 0.5)
        with pytest.raises(ocagy_cli.OcAgyError, match="kept the OpenCode setup busy"):
            ocagy_cli.ensure_opencode_installed()
        # Installed by the other holder meanwhile: found under the lock, nothing runs.
        monkeypatch.setattr(ocagy_cli, "OCAGY_CLI_INSTALL_WAIT_SECONDS", 30)
        outcome, worker = _run_in_thread(ocagy_cli.ensure_opencode_installed)
        worker.join(0.4)
        assert worker.is_alive()  # missed the CLI, waits for the lock
        installed["value"] = True
        release.set()
        worker.join(5)
        assert outcome.get("value") == "opencode" and runs == []
    finally:
        ocagy_cli._CANCEL_EVENT.clear()
        release.set()
        holder.join(5)
    # Still missing once the lock is free: this request installs it.
    installed["value"] = False
    assert ocagy_cli.ensure_opencode_installed() == "opencode"
    assert runs == [["npm", "install", "-g", "opencode-ai"]]


def test_short_checks_on_a_request_wait_for_the_launch_lock_stop_aware(monkeypatch):
    """`opencode --version` / `npm prefix -g` on a first install: Stop ends their lock wait."""
    explicit = lambda: False  # noqa: E731
    assert ocagy_cli._launch_wait_stop(explicit) is explicit
    assert ocagy_cli._launch_wait_stop(None) is ocagy_cli.is_cancelled  # a request thread
    seen = {}
    background = threading.Thread(target=lambda: seen.update(stop=ocagy_cli._launch_wait_stop(None)))
    monkeypatch.setattr(ocagy_cli, "_PLUGIN_UPDATE_THREAD", background)
    background.start()
    background.join(5)
    assert seen == {"stop": None}  # a translation's Stop never interrupts the background update

    release, holder = _hold_lock_in_thread(ocagy_cli._LAUNCH_LOCK_FILENAME, ocagy_cli._LAUNCH_LOCAL_LOCK)
    try:
        outcome, worker = _run_in_thread(lambda: ocagy_cli._opencode_version("opencode"))
        worker.join(0.6)
        assert worker.is_alive()
        ocagy_cli.cancel_stream()
        worker.join(3)
        assert not worker.is_alive() and "cancelled by user" in str(outcome.get("error"))
    finally:
        ocagy_cli._CANCEL_EVENT.clear()
        release.set()
        holder.join(5)


def test_missing_cli_install_is_stopped_by_stop(monkeypatch):
    monkeypatch.setattr(ocagy_cli, "find_executable", lambda *_a: (_ for _ in ()).throw(ocagy_cli.OcAgyError("not found")))
    monkeypatch.setattr(ocagy_cli, "_opencode_install_command", lambda: ["npm", "install", "-g", "opencode-ai"])
    monkeypatch.setattr(
        ocagy_cli, "_run_setup_command", lambda *_a, **_k: {"ok": False, "cancelled": True, "reason": "stopped"}
    )

    with pytest.raises(ocagy_cli.OcAgyError, match="cancelled by user"):
        ocagy_cli.ensure_opencode_installed()


# --- C: several Glossarion processes ----------------------------------------

_CHILD_PRELUDE = """
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, {src!r})
import ocagy_cli
import proxy_token_storage
"""


def _run_children(tmp_path, body, count, *, extra_env=None, timeout=60):
    script = tmp_path / "child.py"
    script.write_text(_CHILD_PRELUDE.format(src=str(SRC)) + body, encoding="utf-8")
    env = dict(os.environ)
    env.update({
        "OCAGY_WORKSPACE": str(tmp_path / "workspace"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "OPENCODE_CONFIG_DIR": str(tmp_path / "config"),
        "PYTHONIOENCODING": "utf-8",
    })
    env.pop("OCAGY_PLUGIN_PACKAGE", None)
    env.update(extra_env or {})
    import subprocess

    children = [
        subprocess.Popen([sys.executable, str(script), str(index)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        for index in range(count)
    ]
    (tmp_path / "go").write_text("go", encoding="utf-8")
    outputs = []
    for child in children:
        stdout, stderr = child.communicate(timeout=timeout)
        assert child.returncode == 0, stderr
        outputs.append(stdout.strip())
    return outputs


_UPDATE_CHILD = """
cache = Path(os.environ["XDG_CACHE_HOME"])
installs = Path(os.environ["OCAGY_WORKSPACE"]).parent / "installs.log"

def fake_install(executable, spec, logger, **kwargs):
    with open(installs, "a", encoding="utf-8") as stream:
        stream.write(f"{os.getpid()} {spec}\\n")
    time.sleep(2)  # a slow download
    root = cache / "opencode" / "packages" / spec / "node_modules" / ocagy_cli._PLUGIN_NAME
    for relative in ("dist/src/plugin/storage.js", "dist/src/plugin/auth.js", "dist/src/antigravity/oauth.js"):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text("// fake", encoding="utf-8")
    (root / "package.json").write_text(json.dumps({"version": spec.rsplit("@", 1)[1]}), encoding="utf-8")
    return ""

ocagy_cli._install_plugin_candidate = fake_install
ocagy_cli._fetch_latest_plugin_version = lambda *a, **k: "2.1.2"
ocagy_cli._opencode_version = lambda exe: (1, 19, 5)
ocagy_cli._verify_auth_plugin = lambda *a, **k: (True, "")
proxy_token_storage.patch_ocagy = lambda root: None
go = Path(os.environ["OCAGY_WORKSPACE"]).parent / "go"
while not go.exists():
    time.sleep(0.01)
force = os.environ.get("FORCE_INDEX") == sys.argv[1]
if force:
    while not installs.exists():  # starts while the other process downloads
        time.sleep(0.01)
print(json.dumps({"force": force, "result": ocagy_cli._update_auth_plugin("opencode", lambda m: None, force=force)}))
"""


def test_two_processes_never_both_update(tmp_path):
    outputs = _run_children(tmp_path, _UPDATE_CHILD, 2)

    installs = (tmp_path / "installs.log").read_text(encoding="utf-8").splitlines()
    assert len(installs) == 1, installs
    results = sorted(json.loads(line)["result"] for line in outputs)
    assert results == [False, True]  # the other background check skipped
    state = json.loads((tmp_path / "workspace" / ocagy_cli._PLUGIN_STATE_FILENAME).read_text(encoding="utf-8"))
    assert state["spec"] == f"{NAME}@2.1.2"


def test_first_install_in_another_process_waits_for_the_running_update(tmp_path):
    outputs = _run_children(tmp_path, _UPDATE_CHILD, 2, extra_env={"FORCE_INDEX": "1"})

    installs = (tmp_path / "installs.log").read_text(encoding="utf-8").splitlines()
    assert len(installs) == 1, installs
    by_mode = {json.loads(line)["force"]: json.loads(line)["result"] for line in outputs}
    # The first-install process waited for the download, then found it installed.
    assert by_mode == {False: True, True: False}


_STATE_WRITER_CHILD = """
go = Path(os.environ["OCAGY_WORKSPACE"]).parent / "go"
while not go.exists():
    time.sleep(0.01)
written = 0
for n in range(150):
    written += bool(ocagy_cli._write_plugin_state({"writer": sys.argv[1], "n": n, "pad": "x" * 2000}))
    ocagy_cli._read_plugin_state()
print(written)
"""


def test_state_file_survives_concurrent_writers(tmp_path):
    outputs = _run_children(tmp_path, _STATE_WRITER_CHILD, 4)

    assert all(int(count) > 0 for count in outputs)
    state = json.loads((tmp_path / "workspace" / ocagy_cli._PLUGIN_STATE_FILENAME).read_text(encoding="utf-8"))
    assert state["writer"] in {"0", "1", "2", "3"} and isinstance(state["n"], int)
    assert len(state["pad"]) == 2000
    assert not list((tmp_path / "workspace").glob("*.tmp"))


_LAUNCH_HOLDER_CHILD = """
go = Path(os.environ["OCAGY_WORKSPACE"]).parent / "go"
while not go.exists():
    time.sleep(0.01)
with ocagy_cli._coordination_lock(ocagy_cli._LAUNCH_LOCK_FILENAME, ocagy_cli._LAUNCH_LOCAL_LOCK, exclusive=True, timeout=10):
    (Path(os.environ["OCAGY_WORKSPACE"]).parent / "held").write_text("held", encoding="utf-8")
    time.sleep(2.0)  # npm replacing the CLI
    released = time.time()
print(released)
"""


def test_no_process_starts_opencode_while_another_runs_npm(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setenv("OCAGY_WORKSPACE", str(tmp_path / "workspace"))
    script = tmp_path / "child.py"
    script.write_text(_CHILD_PRELUDE.format(src=str(SRC)) + _LAUNCH_HOLDER_CHILD, encoding="utf-8")
    env = dict(os.environ, OCAGY_WORKSPACE=str(tmp_path / "workspace"))
    child = subprocess.Popen([sys.executable, str(script)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    (tmp_path / "go").write_text("go", encoding="utf-8")
    deadline = time.monotonic() + 30
    while not (tmp_path / "held").exists():
        assert time.monotonic() < deadline and child.poll() is None, child.communicate()
        time.sleep(0.02)

    with ocagy_cli._opencode_launch_slot():
        started = time.time()
    stdout, stderr = child.communicate(timeout=30)
    assert child.returncode == 0, stderr
    assert started >= float(stdout.strip()) - 0.05

    # Shared holders do not block each other.
    def second_start():
        with ocagy_cli._opencode_launch_slot():
            return True

    with ocagy_cli._opencode_launch_slot():
        outcome, worker = _run_in_thread(second_start)
        worker.join(3)
        assert outcome.get("value") is True and not worker.is_alive()


# --- F: the real plugin 2.x layout parses on every machine ----------------
#
# Trimmed copy of opencode-antigravity-auth 2.1.2 (npm, MIT licence,
# github.com/NoeFabris/opencode-antigravity-auth): dist/src/plugin/config/models.js
# (two of its antigravity entries and one Gemini CLI entry, which is not an
# Antigravity model) and the Gemini 3.8 Flash tier loop of models-v2.js.

PLUGIN_212_MODELS_JS = """const DEFAULT_MODALITIES = {
    input: ["text", "image", "pdf"],
    output: ["text"],
};
export const OPENCODE_MODEL_DEFINITIONS = {
    "antigravity-gemini-3-flash": {
        name: "Gemini 3 Flash (Antigravity)",
        limit: { context: 1048576, output: 65536 },
        modalities: DEFAULT_MODALITIES,
        variants: {
            minimal: { thinkingLevel: "minimal" },
            low: { thinkingLevel: "low" },
            medium: { thinkingLevel: "medium" },
            high: { thinkingLevel: "high" },
        },
    },
    "antigravity-claude-opus-4-6-thinking": {
        name: "Claude Opus 4.6 Thinking (Antigravity)",
        limit: { context: 200000, output: 64000 },
        modalities: DEFAULT_MODALITIES,
        variants: {
            low: { thinkingConfig: { thinkingBudget: 8192 } },
            max: { thinkingConfig: { thinkingBudget: 32768 } },
        },
    },
    "gemini-2.5-flash": {
        name: "Gemini 2.5 Flash (Gemini CLI)",
        limit: { context: 1048576, output: 65536 },
        modalities: DEFAULT_MODALITIES,
    },
};
//# sourceMappingURL=models.js.map
"""

PLUGIN_212_MODELS_V2_JS = """import { OPENCODE_MODEL_DEFINITIONS } from "./models.js";
// Used before login or when discovery is unavailable. Once connected, Google's
// catalog replaces this list, including model IDs, names and token limits.
const fallbackModels = Object.fromEntries(["antigravity-gemini-3-flash", "antigravity-claude-sonnet-4-6", "antigravity-claude-opus-4-6-thinking"]
    .flatMap(id => OPENCODE_MODEL_DEFINITIONS[id] ? [[id, OPENCODE_MODEL_DEFINITIONS[id]]] : []));
for (const tier of ["low", "medium", "high"]) {
    fallbackModels[`antigravity-gemini-3.8-flash-${tier}`] = {
        name: `Gemini 3.8 Flash (${tier}) (Antigravity)`,
        limit: { context: 1048576, output: 65536 },
        modalities: { input: ["text", "image", "pdf"], output: ["text"] },
    };
}
//# sourceMappingURL=models-v2.js.map
"""


def test_real_plugin_212_declarations_parse(env):
    root = _fake_plugin(
        env["cache"], f"{NAME}@2.1.2", "2.1.2",
        models_js=PLUGIN_212_MODELS_JS, models_v2_js=PLUGIN_212_MODELS_V2_JS,
    )
    declared = ocagy_cli._plugin_declared_models(root)

    assert set(declared) == {
        "antigravity-gemini-3-flash",
        "antigravity-claude-opus-4-6-thinking",
        "antigravity-gemini-3.8-flash-low",
        "antigravity-gemini-3.8-flash-medium",
        "antigravity-gemini-3.8-flash-high",
    }
    high = declared["antigravity-gemini-3.8-flash-high"]
    assert high["name"] == "Gemini 3.8 Flash (high) (Antigravity OAuth)"
    assert high["limit"] == {"context": 1048576, "output": 65536} and "variants" not in high
    assert declared["antigravity-claude-opus-4-6-thinking"]["variants"]["max"] == {"thinkingConfig": {"thinkingBudget": 32768}}
    assert declared["antigravity-gemini-3-flash"]["variants"]["minimal"] == {"thinkingLevel": "minimal"}


def test_gemini_38_flash_tiers_resolve_to_model_ids_without_variant(env):
    for tier in ("low", "medium", "high"):
        for prefix in ("ocagy", "ocagy0", "ocagy1"):
            assert ocagy_cli.resolve_model(f"{prefix}/gemini-3.8-flash-{tier}") == (
                f"google/antigravity-gemini-3.8-flash-{tier}",
                None,
            )
    # Existing saved ids keep resolving.
    assert ocagy_cli.resolve_model("ocagy/gemini-3-flash-high") == ("google/antigravity-gemini-3-flash", "high")
    assert ocagy_cli.resolve_model("ocagy/antigravity-gemini-3.8-flash-high") == (
        "google/antigravity-gemini-3.8-flash-high",
        None,
    )


def test_workspace_declares_gemini_38_flash_and_polls_it(env):
    models = ocagy_cli._workspace_config()["provider"]["google"]["models"]
    for tier in ("low", "medium", "high"):
        assert models[f"antigravity-gemini-3.8-flash-{tier}"]["limit"]["output"] == 65536
        assert "variants" not in models[f"antigravity-gemini-3.8-flash-{tier}"]
    polled = ocagy_cli.normalize_polled_models([
        "google/antigravity-gemini-3.1-pro",
        "google/antigravity-gemini-3.8-flash-high",
        "google/antigravity-gemini-3.8-flash-low",
        "google/antigravity-gemini-3.8-flash-medium",
    ])
    assert polled == [
        "ocagy/gemini-3.1-pro-high",
        "ocagy/gemini-3.1-pro-low",
        "ocagy/gemini-3.8-flash-low",
        "ocagy/gemini-3.8-flash-medium",
        "ocagy/gemini-3.8-flash-high",
    ]


def test_model_declared_only_by_the_plugin_is_offered_and_routed(env):
    root = ocagy_cli._plugin_root(f"{NAME}@latest")
    config = root / "dist" / "src" / "plugin" / "config"
    config.mkdir(parents=True, exist_ok=True)
    (config / "models.js").write_text(
        "export const OPENCODE_MODEL_DEFINITIONS = {\n"
        '    "antigravity-gemini-9-pro": {\n'
        '        name: "Gemini 9 Pro (Antigravity)",\n'
        "        limit: { context: 2000000, output: 131072 },\n"
        "        modalities: DEFAULT_MODALITIES,\n"
        "        variants: {\n"
        '            low: { thinkingLevel: "low" },\n'
        '            high: { thinkingLevel: "high" },\n'
        "        },\n"
        "    },\n"
        "};\n",
        encoding="utf-8",
    )
    models = ocagy_cli._workspace_config()["provider"]["google"]["models"]
    assert models["antigravity-gemini-9-pro"]["limit"] == {"context": 2000000, "output": 131072}
    assert models["antigravity-gemini-9-pro"]["variants"]["high"] == {"thinkingLevel": "high"}
    assert models["antigravity-gemini-9-pro"]["temperature"] is True
    assert ocagy_cli.normalize_polled_models(["google/antigravity-gemini-9-pro"]) == [
        "ocagy/gemini-9-pro-low",
        "ocagy/gemini-9-pro-high",
    ]
    assert ocagy_cli.resolve_model("ocagy/gemini-9-pro-high") == ("google/antigravity-gemini-9-pro", "high")
    assert ocagy_cli._model_output_limit("google/antigravity-gemini-9-pro") == 131072


def test_quota_groups_include_gemini_38_flash_per_model():
    groups = ocagy_cli._aggregate_antigravity_quota({
        "gemini-3-flash": {"quotaInfo": {"remainingFraction": 0.9, "resetTime": "2030-01-01T00:00:00Z"}},
        "gemini-3.8-flash-high": {
            "displayName": "Gemini 3.8 Flash (High)",
            "quotaInfo": {"remainingFraction": 0.4, "resetTime": "2030-01-01T05:00:00Z"},
        },
        "gemini-3.8-flash-low": {"quotaInfo": {"remainingFraction": 0.5}},
        "gemini-3.1-pro-high": {"quotaInfo": {"remainingFraction": 0.7}},
        "gemini-4-argon-high": {"quotaInfo": {"remainingFraction": 0.2}},
        "claude-opus-4-6-thinking": {"quotaInfo": {"remainingFraction": 1.0}},
        "gemini-3-pro-image": {"quotaInfo": {"remainingFraction": 0.0}},
        "gemini-2.5-flash-disabled": {"quotaInfo": {"remainingFraction": 0.0}},
        "gpt-oss-120b-medium": {"quotaInfo": {"remainingFraction": 0.0}},
        "gemini-3-old": {"disabled": True, "quotaInfo": {"remainingFraction": 0.0}},
    })

    flash = groups["gemini-flash"]
    assert flash["model_count"] == 3
    assert flash["remaining_fraction"] == 0.4
    labels = [model["label"] for model in flash["models"]]
    assert "Gemini 3.8 Flash (High)" in labels and "gemini-3.8-flash-low" in labels
    high = next(model for model in flash["models"] if model["model_id"] == "gemini-3.8-flash-high")
    assert high["remaining_fraction"] == 0.4 and high["reset_in"]
    assert groups["gemini-pro"]["model_count"] == 2  # 3.1 Pro and Gemini 4
    assert groups["gemini-pro"]["remaining_fraction"] == 0.2
    assert groups["claude"]["model_count"] == 1


@pytest.mark.skipif(sys.platform != "win32", reason="the PowerShell login window is Windows-only")
def test_login_window_points_to_glossarion_quota_button(tmp_path, monkeypatch):
    commands = []

    class FakeProcess:
        pid = 1

    monkeypatch.setattr(ocagy_cli, "ensure_opencode_installed", lambda log_fn=None: "opencode")
    monkeypatch.setattr(ocagy_cli, "ensure_auth_plugin_installed", lambda *_a, **_k: None)
    monkeypatch.setattr(ocagy_cli, "_workspace_dir", lambda: tmp_path)
    monkeypatch.setattr(ocagy_cli.subprocess, "Popen", lambda args, **_k: commands.append(args) or FakeProcess())

    ocagy_cli.launch_login()

    script = commands[0][-1]
    assert "quota" in script.lower() and "not the plugin menu" in script
