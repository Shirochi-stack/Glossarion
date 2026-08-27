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
    monkeypatch.delenv("AUTHZA_USE_GENERAL_API", raising=False)
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
    assert glm_proxy._credentials_path(0, general_api=False).endswith("credentials.json")
    assert glm_proxy._credentials_path(0, general_api=True).endswith(
        "credentials-general-api.json"
    )

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


def test_account_config_uses_zcode_login_plan_and_local_auth():
    path = Path(glm_proxy._ensure_account_config(2))
    contents = path.read_text(encoding="utf-8")

    assert 'host: "127.0.0.1"' in contents
    assert "port: 18872" in contents
    assert 'mode: "oauth"' in contents
    assert 'provider: "zai"' in contents
    assert 'plan: "start-plan"' in contents
    assert f'appVersion: "{glm_proxy.ZCODE_APP_VERSION}"' in contents
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


def test_account_config_uses_general_api_when_enabled(monkeypatch):
    monkeypatch.setenv("AUTHZA_USE_GENERAL_API", "1")

    path = Path(glm_proxy._ensure_account_config(2))
    contents = path.read_text(encoding="utf-8")

    assert 'plan: "coding-plan"' in contents
    assert f'openaiBase: "{glm_proxy.GENERAL_API_BASE}"' in contents
    assert "credentials-general-api.json" in contents
    assert "endpointRouting:\n  enabled: false" in contents
    assert "clientSigning:\n  enabled: false" in contents
    env = glm_proxy._runtime_env(2)
    assert env["GLOSSARION_ZCODE_LOGIN_PLAN_ONLY"] == "0"
    assert env["ZCODE_PROXY_CREDENTIALS_PATH"].endswith(
        "credentials-general-api.json"
    )


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


def test_runtime_patch_forces_numbered_login_through_zai_account_switch(tmp_path):
    index = tmp_path / "src" / "index.ts"
    index.parent.mkdir(parents=True)
    index.write_text(
        "function openBrowser(url: string): void {\n"
        "  console.log(url);\n"
        "}\n",
        encoding="utf-8",
    )

    glm_proxy._patch_numbered_account_switch(str(tmp_path))
    glm_proxy._patch_numbered_account_switch(str(tmp_path))
    patched = index.read_text(encoding="utf-8")

    assert patched.count("GLOSSARION_AUTHZA_ACCOUNT_SWITCH") == 1
    assert 'ZCODE_OAUTH_FORCE_ACCOUNT_SELECTION === "1"' in patched
    assert "https://chat.z.ai/auth?redirect=" in patched
    assert "&switch_account=true" in patched
    assert 'authorizeTarget.hostname === "chat.z.ai"' in patched


def test_runtime_patch_uses_login_jwt_without_provisioning_api_key(tmp_path):
    index = tmp_path / "src" / "index.ts"
    index.parent.mkdir(parents=True)
    index.write_text(
        '    const { accessToken, userId, jwt } = await runOAuth(provider);\n'
        '    console.log("\\nResolving API key...");\n'
        '    const resolver = new KeyResolver();\n'
        '    cred = await resolver.resolveCodingPlanCredential(accessToken, provider, userId);\n'
        '    if (jwt) cred.jwt = jwt;\n'
        '  console.log(`  API Key: ${cred.apiKey.substring(0, 12)}...`);\n'
        '  console.log(`  API Key: ${cred.apiKey.substring(0, 12)}...`);\n',
        encoding="utf-8",
    )

    glm_proxy._patch_login_plan_only_auth(str(tmp_path))
    glm_proxy._patch_login_plan_only_auth(str(tmp_path))
    patched = index.read_text(encoding="utf-8")

    assert patched.count("GLOSSARION_ZCODE_LOGIN_PLAN_JWT_ONLY") == 1
    assert 'cred = { apiKey: "zcode-login", provider: "zai", userId, jwt }' in patched
    assert "no API key provisioned" in patched
    assert "resolveCodingPlanCredential(accessToken" in patched
    assert patched.count("Login credential: ZCode JWT") == 2
    assert patched.count("API Key: stored securely") == 2
    assert "cred.apiKey.substring" not in patched


def test_runtime_patch_uses_current_zcode_login_plan_protocol(tmp_path):
    proxy = tmp_path / "src" / "proxy"
    proxy.mkdir(parents=True)
    upstream = proxy / "upstream.ts"
    upstream.write_text(
        'const STARTPLAN_OPENAI_BASE = "https://zcode.z.ai/api/v1/zcode-plan";\n'
        'return `${STARTPLAN_OPENAI_BASE}/chat/completions`;\n',
        encoding="utf-8",
    )
    handler = proxy / "handler.ts"
    handler.write_text(
        '  const startPlan = config.plan === "start-plan";\n'
        '  const translateAnthropicToOpenAI = format === "anthropic" && startPlan;\n'
        '  const translateOpenAIToAnthropic = format === "openai" && !startPlan;\n'
        '  const upstreamFormat: Format = startPlan ? "openai" : "anthropic";\n',
        encoding="utf-8",
    )

    glm_proxy._patch_zcode_login_plan_endpoint(str(tmp_path))
    glm_proxy._patch_zcode_login_plan_endpoint(str(tmp_path))

    patched_upstream = upstream.read_text(encoding="utf-8")
    patched_handler = handler.read_text(encoding="utf-8")
    assert patched_upstream.count("GLOSSARION_ZCODE_LOGIN_PLAN_ANTHROPIC") == 1
    assert "https://zcode.z.ai/api/v1/zcode-plan/anthropic" in patched_upstream
    assert "`${STARTPLAN_ANTHROPIC_BASE}/v1/messages`" in patched_upstream
    assert patched_handler.count("GLOSSARION_ZCODE_DUAL_ACCESS_ROUTING") == 1
    assert 'openaiBase.includes("/api/paas/v4")' in patched_handler
    assert 'translateOpenAIToAnthropic = !generalApi && format === "openai"' in patched_handler
    assert 'upstreamFormat: Format = generalApi ? "openai" : "anthropic"' in patched_handler


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
    assert observed["env"]["ZCODE_OAUTH_FORCE_ACCOUNT_SELECTION"] == "1"


def test_default_authza_login_does_not_force_account_switch(tmp_path, monkeypatch):
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
    glm_proxy._login(0)

    assert observed["env"]["ZCODE_OAUTH_FORCE_ACCOUNT_SELECTION"] == "0"


def test_general_api_login_enables_key_provisioning_and_separate_store(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    entry = runtime / "src" / "index.ts"
    entry.parent.mkdir(parents=True)
    entry.write_text("", encoding="utf-8")
    observed = {}
    monkeypatch.setenv("AUTHZA_USE_GENERAL_API", "1")
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
    glm_proxy._login(3)

    assert observed["env"]["GLOSSARION_ZCODE_LOGIN_PLAN_ONLY"] == "0"
    assert observed["env"]["ZCODE_PROXY_CREDENTIALS_PATH"].endswith(
        "credentials-general-api.json"
    )


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
    monkeypatch.setattr(glm_proxy, "has_credentials", lambda account_id: account_id == 2)
    monkeypatch.setattr(
        glm_proxy,
        "_ensure_runtime_and_dependencies",
        lambda: ("C:/runtime", ["bun"]),
    )
    observed = {}

    def fake_run(command, **kwargs):
        observed.update(command=command, **kwargs)
        return {
            "returncode": 0,
            "output": 'warning line\nGLOSSARION_MODELS=["glm-5.3","glm-5.2","GLM-5.3"]',
            "timed_out": False,
        }

    monkeypatch.setattr(glm_proxy, "run_logged_subprocess", fake_run)

    assert glm_proxy.fetch_available_models(2) == ["glm-5.3", "glm-5.2"]
    assert observed["command"][:2] == ["bun", "-e"]
    assert observed["cwd"] == "C:/runtime"
    assert observed["env"]["ZCODE_PROXY_CREDENTIALS_PATH"] == glm_proxy._credentials_path(2)
    assert observed["env"]["ZCODE_APP_VERSION"] == glm_proxy.ZCODE_APP_VERSION


def test_fetch_available_models_uses_general_api_catalog(monkeypatch):
    monkeypatch.setenv("AUTHZA_USE_GENERAL_API", "1")
    monkeypatch.setattr(glm_proxy, "has_credentials", lambda account_id: True)
    monkeypatch.setattr(
        glm_proxy,
        "_ensure_runtime_and_dependencies",
        lambda: ("C:/runtime", ["bun"]),
    )
    observed = {}

    def fake_run(command, **kwargs):
        observed.update(command=command, **kwargs)
        return {
            "returncode": 0,
            "output": 'GLOSSARION_MODELS=["glm-5.3","glm-4.7"]',
            "timed_out": False,
        }

    monkeypatch.setattr(glm_proxy, "run_logged_subprocess", fake_run)

    assert glm_proxy.fetch_available_models(1) == ["glm-5.3", "glm-4.7"]
    assert "credentialString" in observed["command"][2]
    assert (
        observed["env"]["ZCODE_GENERAL_API_MODELS_ENDPOINT"]
        == glm_proxy.GENERAL_API_MODELS_ENDPOINT
    )
    assert "ZCODE_LOGIN_PLAN_MODELS_ENDPOINT" not in observed["env"]
    assert observed["env"]["ZCODE_PROXY_CREDENTIALS_PATH"].endswith(
        "credentials-general-api.json"
    )


def test_switching_access_mode_stops_managed_proxies(monkeypatch):
    stopped = []
    glm_proxy._proxy_processes.update({0: object(), 3: object()})
    monkeypatch.setattr(glm_proxy, "shutdown_proxy", stopped.append)

    assert glm_proxy.set_general_api_mode(True) is True

    assert os.environ["AUTHZA_USE_GENERAL_API"] == "1"
    assert stopped == [0, 3]


def test_other_settings_exposes_billable_authza_general_api_toggle():
    root = Path(__file__).resolve().parents[1]
    settings_source = (root / "src" / "other_settings.py").read_text(encoding="utf-8")
    gui_source = (root / "src" / "translator_gui.py").read_text(encoding="utf-8")

    assert "Use auto-provisioned API key with Z.AI General API" in settings_source
    assert "https://api.z.ai/api/paas/v4" in settings_source
    assert "may incur charges" in settings_source
    assert "authza_use_general_api" in settings_source
    assert "('authza_use_general_api', ['authza_use_general_api_var']" in gui_source
    assert "'AUTHZA_USE_GENERAL_API'" in gui_source


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


def test_authza_entitlement_error_is_not_retried_as_rate_limit(tmp_path, monkeypatch):
    calls = []

    def rejected_send(**kwargs):
        calls.append(kwargs)
        raise RuntimeError(
            'GLM proxy: HTTP 502 - upstream returned 429: '
            '{"error":{"code":"1113","message":"Insufficient balance or no resource package"}}'
        )

    monkeypatch.setattr(unified_api_client, "_authza_send", rejected_send)
    monkeypatch.setattr(unified_api_client, "AUTHZA_AVAILABLE", True)
    client = UnifiedClient(
        api_key="",
        model="authza/glm-5.3",
        output_dir=str(tmp_path),
    )
    monkeypatch.setattr(client, "_get_max_retries", lambda: 7)

    with pytest.raises(unified_api_client.UnifiedClientError) as exc_info:
        client._send_authza(
            [{"role": "user", "content": "translate"}],
            temperature=0.2,
            max_tokens=256,
            response_name="translation",
        )

    assert len(calls) == 1
    assert exc_info.value.error_type == "entitlement_error"
    assert "Coding Plan resource package" in str(exc_info.value)
    assert client._is_rate_limit_error(exc_info.value) is False


def test_authza_model_not_allowed_stops_without_retry(tmp_path, monkeypatch):
    calls = []

    def rejected_send(**kwargs):
        calls.append(kwargs)
        raise RuntimeError(
            'GLM proxy: HTTP 502 - upstream returned 400: '
            '{"code":3006,"msg":"model not allowed"}'
        )

    monkeypatch.setattr(unified_api_client, "_authza_send", rejected_send)
    monkeypatch.setattr(unified_api_client, "AUTHZA_AVAILABLE", True)
    client = UnifiedClient(
        api_key="",
        model="authza/glm-4.7",
        output_dir=str(tmp_path),
    )
    monkeypatch.setattr(client, "_get_max_retries", lambda: 7)

    with pytest.raises(unified_api_client.UnifiedClientError) as exc_info:
        client._send_authza(
            [{"role": "user", "content": "translate"}],
            temperature=0.2,
            max_tokens=256,
            response_name="translation",
        )

    assert len(calls) == 1
    assert exc_info.value.error_type == "validation"
    assert "Poll Providers" in str(exc_info.value)


def test_global_stop_propagates_to_glm_proxy(monkeypatch):
    calls = []
    monkeypatch.setattr(unified_api_client, "_authza_cancel_stream", lambda: calls.append("cancel"))
    monkeypatch.setattr(unified_api_client, "_authza_reset_cancel", lambda: calls.append("reset"))

    unified_api_client.set_stop_flag(True)
    unified_api_client.set_stop_flag(False)

    assert calls == ["cancel", "reset"]
