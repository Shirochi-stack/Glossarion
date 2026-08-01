import json
import os
import stat
import sys
from pathlib import Path

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
        model="gemini-3.1-pro-high",
        timeout=30,
    )

    assert result["content"].startswith("OK:google/antigravity-gemini-3.1-pro:high:")
    assert result["variant"] == "high"
    assert result["usage"]["total_tokens"] == 14
    assert captured["log_stream"] is True
    assert captured["model_id"] == "google/antigravity-gemini-3.1-pro"
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
