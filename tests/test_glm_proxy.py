import json
import os
from pathlib import Path

import pytest

import glm_proxy
import unified_api_client
from model_options import get_model_options
from unified_api_client import UnifiedClient


@pytest.fixture(autouse=True)
def _isolated_glm_proxy(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM_PROXY_DATA_DIR", str(tmp_path / "glm-proxy"))
    for name in list(os.environ):
        if name.startswith("GLM_PROXY_PORT") or name.startswith("GLM_PROXY_URL"):
            monkeypatch.delenv(name, raising=False)
        if name.startswith("GLM_PROXY_API_KEY"):
            monkeypatch.delenv(name, raising=False)
    glm_proxy._proxy_processes.clear()
    glm_proxy.reset_cancel()
    glm_proxy.set_proxy_started_callback(None)
    yield
    glm_proxy._proxy_processes.clear()
    glm_proxy.reset_cancel()
    glm_proxy.set_proxy_started_callback(None)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text
        self.closed = False

    def json(self):
        return self._payload

    def close(self):
        self.closed = True


class FakeStreamResponse(FakeResponse):
    def __init__(self, lines):
        super().__init__(200, {})
        self.lines = lines

    def iter_lines(self, decode_unicode=True, chunk_size=1):
        yield from self.lines


def test_account_paths_urls_and_ports_are_isolated(monkeypatch):
    assert glm_proxy.get_proxy_url(0) == "http://127.0.0.1:18870"
    assert glm_proxy.get_proxy_url(3) == "http://127.0.0.1:18873"
    assert glm_proxy._credentials_path(0) != glm_proxy._credentials_path(3)

    monkeypatch.setenv("GLM_PROXY_PORT_BASE", "21000")
    monkeypatch.setenv("GLM_PROXY_PORT_3", "22003")
    assert glm_proxy._get_proxy_port(2) == 21002
    assert glm_proxy._get_proxy_port(3) == 22003


def test_authza_models_are_available_in_the_main_dropdown():
    models = get_model_options()
    assert "authza/glm-5.3" in models
    assert "authza/glm-4.5-air" in models


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("authza/", 0),
        (" AUTHZA/glm-5.3 ", 0),
        ("authza2/glm-5.3", 2),
        ("authza9999/glm-5", 9999),
        ("authza10000/glm-5", None),
        ("openai/gpt-5", None),
    ],
)
def test_authza_model_account_parser(model, expected):
    assert glm_proxy.account_id_from_model(model) == expected


def test_account_config_uses_oauth_and_local_auth():
    path = Path(glm_proxy._ensure_account_config(2))
    contents = path.read_text(encoding="utf-8")

    assert 'host: "127.0.0.1"' in contents
    assert "port: 18872" in contents
    assert 'mode: "oauth"' in contents
    assert 'provider: "zai"' in contents
    assert 'plan: "coding-plan"' in contents
    assert "glm-5.3" in contents
    assert json.dumps(glm_proxy._credentials_path(2)) in contents

    saved = json.loads(Path(glm_proxy._secrets_path(2)).read_text(encoding="utf-8"))
    assert saved["proxy_api_key"].startswith("sk-glossarion-")
    assert len(saved["credential_secret"]) > 30
    assert saved["device_mid"] in contents

    # Rewriting the managed config must preserve the account's device identity.
    glm_proxy._ensure_account_config(2)
    saved_again = json.loads(Path(glm_proxy._secrets_path(2)).read_text(encoding="utf-8"))
    assert saved_again["device_mid"] == saved["device_mid"]


def test_runtime_patch_redirects_current_upstream_store(tmp_path):
    store = tmp_path / "src" / "auth" / "store.ts"
    store.parent.mkdir(parents=True)
    store.write_text(
        'const STORE_DIR = join(homedir(), ".zcode-proxy");\n'
        'const STORE_FILE = join(STORE_DIR, "credentials.json");\n',
        encoding="utf-8",
    )

    glm_proxy._patch_credentials_store(str(tmp_path))
    patched = store.read_text(encoding="utf-8")

    assert "ZCODE_PROXY_CREDENTIALS_PATH" in patched
    assert 'join(STORE_DIR, "credentials.json")' in patched


def test_dependencies_are_installed_with_frozen_lockfile(tmp_path, monkeypatch):
    (tmp_path / "bun.lock").write_text("", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        (tmp_path / "node_modules").mkdir()
        return {"returncode": 0, "output": "ok", "timed_out": False}

    monkeypatch.setattr(glm_proxy, "run_logged_subprocess", fake_run)
    glm_proxy._ensure_dependencies(str(tmp_path), ["bun"], log_fn=lambda _message: None)

    assert calls[0][0] == ["bun", "install", "--frozen-lockfile"]
    assert calls[0][1]["cwd"] == str(tmp_path)
    assert (tmp_path / ".glossarion-dependencies.json").is_file()


def test_bun_launch_uses_node_npx_as_automatic_fallback(monkeypatch):
    monkeypatch.delenv("GLM_PROXY_BUN_CMD", raising=False)
    monkeypatch.setattr(
        glm_proxy,
        "_candidate_executable",
        lambda name: "C:/node/npx.cmd" if name == "npx" else None,
    )

    assert glm_proxy._bun_command() == [
        "C:/node/npx.cmd",
        "--yes",
        "--package",
        glm_proxy.BUN_NPM_PACKAGE,
        "bun",
    ]


def test_automatic_runtime_installer_is_available(monkeypatch):
    monkeypatch.delenv("GLM_PROXY_BUN_INSTALL_CMD", raising=False)
    if glm_proxy.sys.platform == "win32":
        monkeypatch.setattr(glm_proxy, "_candidate_executable", lambda _name: "powershell.exe")
        command = glm_proxy._automatic_bun_install_command()
        assert command[0] == "powershell.exe"
        assert "bun.sh/install.ps1" in command[-1]
    else:
        monkeypatch.setattr(
            glm_proxy,
            "_candidate_executable",
            lambda name: f"/usr/bin/{name}" if name in {"bash", "curl"} else None,
        )
        command = glm_proxy._automatic_bun_install_command()
        assert command[:2] == ["/usr/bin/bash", "-c"]
        assert "bun.sh/install" in command[-1]


def test_login_uses_isolated_credentials_and_upstream_cli(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    entry = runtime / "src" / "index.ts"
    entry.parent.mkdir(parents=True)
    entry.write_text("", encoding="utf-8")
    observed = {}

    monkeypatch.setattr(
        glm_proxy,
        "_ensure_runtime_and_dependencies",
        lambda log_fn=None: (str(runtime), ["bun"]),
    )
    monkeypatch.setattr(glm_proxy, "has_credentials", lambda account_id: True)

    def fake_run(command, **kwargs):
        observed.update(command=command, **kwargs)
        return {"returncode": 0, "output": "logged in", "timed_out": False}

    monkeypatch.setattr(glm_proxy, "run_logged_subprocess", fake_run)
    glm_proxy._login(4)

    assert observed["command"] == ["bun", "run", str(entry), "auth", "login", "zai"]
    assert observed["env"]["ZCODE_PROXY_CREDENTIALS_PATH"] == glm_proxy._credentials_path(4)
    assert observed["env"]["ZCODE_PROXY_CONFIG"] == glm_proxy._config_path(4)


def test_ensure_proxy_running_launches_account_runtime(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    entry = runtime / "src" / "index.ts"
    entry.parent.mkdir(parents=True)
    entry.write_text("", encoding="utf-8")
    health = iter(({"healthy": False}, {"healthy": False}, {"healthy": True}))
    observed = {}
    started_accounts = []

    class FakeProcess:
        pid = 4242
        returncode = None

        @staticmethod
        def poll():
            return None

    monkeypatch.setattr(glm_proxy, "check_proxy_health", lambda account_id=None: next(health))
    monkeypatch.setattr(glm_proxy, "has_credentials", lambda account_id=None: True)
    monkeypatch.setattr(
        glm_proxy,
        "_ensure_runtime_and_dependencies",
        lambda log_fn=None: (str(runtime), ["bun"]),
    )
    monkeypatch.setattr(glm_proxy.time, "sleep", lambda _seconds: None)

    def fake_popen(command, **kwargs):
        observed.update(command=command, **kwargs)
        return FakeProcess()

    monkeypatch.setattr(glm_proxy.subprocess, "Popen", fake_popen)
    glm_proxy.set_proxy_started_callback(started_accounts.append)
    status = glm_proxy.ensure_proxy_running(account_id=2)

    assert status["running"] is True
    assert observed["command"] == [
        "bun",
        "run",
        str(entry),
        "serve",
        glm_proxy._config_path(2),
    ]
    assert observed["env"]["ZCODE_PROXY_CREDENTIALS_PATH"] == glm_proxy._credentials_path(2)
    assert started_accounts == [2]


def test_health_check_authenticates_probe(monkeypatch):
    observed = {}

    def fake_get(url, **kwargs):
        observed.update(url=url, **kwargs)
        return FakeResponse(200, {"status": "ok"})

    monkeypatch.setattr(glm_proxy.requests, "get", fake_get)
    result = glm_proxy.check_proxy_health(1)

    assert result["healthy"] is True
    assert observed["url"].endswith(":18871/health")
    assert observed["headers"]["Authorization"].startswith("Bearer sk-glossarion-")
    assert observed["headers"]["x-api-key"].startswith("sk-glossarion-")


def test_send_message_parses_openai_response(monkeypatch):
    monkeypatch.setattr(glm_proxy, "_ensure_for_request", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        glm_proxy.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(
            200,
            {
                "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 3},
            },
        ),
    )

    result = glm_proxy.send_message([{"role": "user", "content": "hi"}], account_id=1)

    assert result["content"] == "hello"
    assert result["finish_reason"] == "stop"
    assert result["usage"] == {"total_tokens": 3}


def test_send_message_stream_collects_content_and_usage(monkeypatch):
    monkeypatch.setattr(glm_proxy, "_ensure_for_request", lambda *args, **kwargs: None)
    response = FakeStreamResponse(
        [
            'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}',
            'data: {"choices":[{"delta":{"content":"hel"}}]}',
            'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}],"usage":{"total_tokens":4}}',
            "data: [DONE]",
        ]
    )
    monkeypatch.setattr(glm_proxy.requests, "post", lambda *args, **kwargs: response)
    logs = []

    result = glm_proxy.send_message_stream(
        [{"role": "user", "content": "hi"}],
        account_id=3,
        log_fn=logs.append,
    )

    assert result["content"] == "hello"
    assert result["finish_reason"] == "stop"
    assert result["usage"] == {"total_tokens": 4}
    assert response.closed is True
    assert "think" in logs


def test_fetch_available_models_deduplicates(monkeypatch):
    monkeypatch.setattr(glm_proxy, "_ensure_for_request", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        glm_proxy.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(
            200,
            {"data": [{"id": "glm-5.3"}, {"id": "glm-5.2"}, {"id": "glm-5.3"}]},
        ),
    )

    assert glm_proxy.fetch_available_models(2) == ["glm-5.3", "glm-5.2"]


def test_cancel_stream_closes_active_response():
    response = FakeStreamResponse([])
    glm_proxy._register_response(response)

    glm_proxy.cancel_stream()

    assert glm_proxy.is_cancelled() is True
    assert response.closed is True
    glm_proxy._unregister_response(response)


def test_unified_authza_route_forwards_numbered_account(tmp_path, monkeypatch):
    observed = {}

    def fake_send(**kwargs):
        observed.update(kwargs)
        return {"content": "translated", "finish_reason": "stop", "usage": None}

    monkeypatch.setattr(unified_api_client, "_authza_send", fake_send)
    monkeypatch.setattr(unified_api_client, "AUTHZA_AVAILABLE", True)
    client = UnifiedClient(
        api_key="",
        model="authza2/glm-5.3",
        output_dir=str(tmp_path),
    )
    monkeypatch.setattr(client, "_get_max_retries", lambda: 1)

    result = client._send_authza(
        [{"role": "user", "content": "translate"}],
        temperature=0.2,
        max_tokens=256,
        response_name="translation",
    )

    assert result.content == "translated"
    assert observed["model"] == "glm-5.3"
    assert observed["account_id"] == 2
    assert observed["auto_login"] is True


def test_global_stop_propagates_to_glm_proxy(monkeypatch):
    calls = []
    monkeypatch.setattr(unified_api_client, "_authza_cancel_stream", lambda: calls.append("cancel"))
    monkeypatch.setattr(unified_api_client, "_authza_reset_cancel", lambda: calls.append("reset"))

    unified_api_client.set_stop_flag(True)
    unified_api_client.set_stop_flag(False)

    assert calls == ["cancel", "reset"]
