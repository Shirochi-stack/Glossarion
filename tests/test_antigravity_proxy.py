import json

import pytest

import antigravity_proxy
from html_output_utils import ensure_utf8_html_document
from model_options import get_model_options


class FakeStreamResponse:
    status_code = 200

    def __init__(self, lines):
        self._lines = lines
        self.closed = False

    def iter_lines(self, decode_unicode=True, chunk_size=1):
        yield from self._lines

    def close(self):
        self.closed = True


class FakeHttpxStreamResponse:
    status_code = 200

    def __init__(self, lines):
        self._lines = lines
        self.closed = False

    def iter_lines(self):
        yield from self._lines

    def close(self):
        self.closed = True


class FakeHTTPResponse:
    def __init__(self, json_data=None, text="", content=b""):
        self._json_data = json_data
        self.text = text
        self.content = content

    def json(self):
        return self._json_data

    def raise_for_status(self):
        return None


class EncodingAwareStreamResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload
        self.encoding = "iso-8859-1"
        self.closed = False

    def iter_lines(self, decode_unicode=True, chunk_size=1):
        raw = ("data: " + json.dumps(self.payload, ensure_ascii=False)).encode("utf-8")
        yield raw.decode(self.encoding) if decode_unicode else raw
        yield "data: [DONE]"

    def close(self):
        self.closed = True


def _sse_event(payload):
    return "data: " + json.dumps(payload)


def test_normalize_model_name_prefixes_gemini_ids_for_upstream_proxy():
    assert (
        antigravity_proxy._normalize_model_name("gemini-2.5-flash")
        == "antigravity-gemini-2.5-flash"
    )
    assert (
        antigravity_proxy._normalize_model_name("antigravity/gemini-2.5-pro")
        == "antigravity-gemini-2.5-pro"
    )


def test_normalize_model_name_prefixes_sandbox_ids_for_upstream_proxy():
    assert (
        antigravity_proxy._normalize_model_name("claude-sonnet-4-6")
        == "antigravity-claude-sonnet-4-6"
    )
    assert (
        antigravity_proxy._normalize_model_name("antigravity/gemini-3.1-pro-low")
        == "antigravity-gemini-3.1-pro-low"
    )
    assert (
        antigravity_proxy._normalize_model_name("antigravity-claude-opus-4-6-thinking-high")
        == "antigravity-claude-opus-4-6-thinking-high"
    )


def test_parse_openai_chat_response():
    data = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "translated text"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }

    parsed = antigravity_proxy._parse_openai_chat_response(data)

    assert parsed["content"] == "translated text"
    assert parsed["finish_reason"] == "stop"
    assert parsed["usage"]["total_tokens"] == 5
    assert parsed["raw_response"] is data


def test_consume_openai_stream_collects_content_and_usage():
    response = FakeStreamResponse(
        [
            _sse_event({"choices": [{"delta": {"reasoning_content": "think"}, "finish_reason": None}]}),
            _sse_event(
                {
                    "choices": [{"delta": {"content": "Hel"}, "finish_reason": None}],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
                }
            ),
            _sse_event({"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]}),
            "data: [DONE]",
        ]
    )

    result = antigravity_proxy._consume_openai_stream(response, log_fn=lambda _: None, log_stream=False)

    assert result["content"] == "Hello"
    assert result["finish_reason"] == "stop"
    assert result["usage"]["total_tokens"] == 5
    assert response.closed is True


def test_consume_openai_stream_supports_httpx_iter_lines():
    response = FakeHttpxStreamResponse(
        [
            _sse_event({"choices": [{"delta": {"content": "Hel"}, "finish_reason": None}]}),
            _sse_event({"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]}),
            "data: [DONE]",
        ]
    )

    result = antigravity_proxy._consume_openai_stream(response, log_fn=lambda _: None, log_stream=False)

    assert result["content"] == "Hello"
    assert result["finish_reason"] == "stop"
    assert response.closed is True


def test_consume_openai_stream_raises_on_error_event():
    response = FakeStreamResponse(
        [
            _sse_event(
                {
                    "error": {
                        "message": "Quota Exhausted: All accounts failed or are exhausted for this model.",
                        "code": "insufficient_quota",
                    }
                }
            ),
            "data: [DONE]",
        ]
    )

    with pytest.raises(RuntimeError, match="Quota Exhausted"):
        antigravity_proxy._consume_openai_stream(response, log_fn=lambda _: None, log_stream=False)

    assert response.closed is True


def test_consume_openai_stream_forces_utf8_for_unicode_content():
    response = EncodingAwareStreamResponse(
        {"choices": [{"delta": {"content": "I’m telling you."}, "finish_reason": "stop"}]}
    )

    result = antigravity_proxy._consume_openai_stream(response, log_fn=lambda _: None, log_stream=False)

    assert result["content"] == "I’m telling you."
    assert response.encoding == "utf-8"


def test_antigravity_payload_clamps_model_token_limits():
    claude_payload = antigravity_proxy._payload_for_openai_chat(
        [], "claude-sonnet-4-6", 0.2, 200000, False
    )
    gemini_payload = antigravity_proxy._payload_for_openai_chat(
        [], "gemini-3.5-flash-low", 0.2, 200000, False
    )

    assert claude_payload["max_tokens"] == 64000
    assert gemini_payload["max_tokens"] == 64000


def test_antigravity_token_limit_log_reports_clamp():
    payload = antigravity_proxy._payload_for_openai_chat(
        [], "gemini-2.5-flash", 0.2, 65536, False
    )
    messages = []

    antigravity_proxy._log_payload_token_limit(messages.append, 65536, payload)

    assert payload["max_tokens"] == 64000
    assert messages == [
        "🎚️ Antigravity: max_tokens clamped 65,536 -> 64,000 (model=antigravity-gemini-2.5-flash)"
    ]


def test_min_accounts_for_auth_retry_requires_new_account_on_quota(monkeypatch):
    monkeypatch.setattr(antigravity_proxy, "_proxy_account_count", lambda: 1)

    assert antigravity_proxy._min_accounts_for_auth_retry("Quota Exhausted: All accounts failed") == 2
    assert antigravity_proxy._min_accounts_for_auth_retry("No accounts configured") == 1


def test_stream_chat_with_httpx_disables_compression(monkeypatch):
    class FakeTimeout:
        def __init__(self, timeout, connect=None):
            self.timeout = timeout
            self.connect = connect

    class FakeHttpx:
        Timeout = FakeTimeout

        def __init__(self):
            self.call = None

        def stream(self, method, url, **kwargs):
            self.call = {"method": method, "url": url, **kwargs}
            return object()

    fake_httpx = FakeHttpx()
    monkeypatch.setattr(antigravity_proxy, "httpx", fake_httpx)

    result = antigravity_proxy._stream_chat_with_httpx(
        "http://localhost:3000/v1/chat/completions",
        {"stream": True},
        {"Content-Type": "application/json"},
        300,
    )

    assert result is not None
    assert fake_httpx.call["method"] == "POST"
    assert fake_httpx.call["headers"]["Accept"] == "text/event-stream"
    assert fake_httpx.call["headers"]["Accept-Encoding"] == "identity"
    assert fake_httpx.call["timeout"].timeout == 300
    assert fake_httpx.call["timeout"].connect == 30.0


def test_wait_for_auth_keeps_httpx_stream_open_until_consumer_closes(monkeypatch):
    response = FakeHttpxStreamResponse(["data: [DONE]"])

    class FakeStreamContext:
        def __init__(self):
            self.exited = False

        def __enter__(self):
            return response

        def __exit__(self, exc_type, exc, tb):
            self.exited = True

    context = FakeStreamContext()

    monkeypatch.setattr(antigravity_proxy, "httpx", object())
    monkeypatch.setattr(antigravity_proxy.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(antigravity_proxy, "_open_auth_browser_once", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(antigravity_proxy, "_proxy_has_accounts", lambda: True)
    monkeypatch.setattr(antigravity_proxy, "_proxy_account_count", lambda: 1)
    monkeypatch.setattr(
        antigravity_proxy,
        "_stream_chat_with_httpx",
        lambda *_args, **_kwargs: context,
    )

    retry_resp = antigravity_proxy._wait_for_auth(
        "http://localhost:3000/v1/chat/completions",
        {"stream": True},
        {"Content-Type": "application/json"},
        "http://localhost:3000",
        log_fn=lambda _message: None,
        max_wait=5,
        poll_interval=5,
        stream=True,
        request_timeout=300,
        prefer_httpx_stream=True,
    )

    assert retry_resp is not None
    assert retry_resp.status_code == 200
    assert response.closed is False
    assert context.exited is False

    retry_resp.close()

    assert response.closed is True
    assert context.exited is True


def test_utf8_html_output_helper_adds_charset_to_fragments_and_documents():
    fragment = ensure_utf8_html_document("<h1>Title</h1><p>I’m here.</p>")
    document = ensure_utf8_html_document("<html><body><p>I’m here.</p></body></html>")

    assert '<meta charset="utf-8">' in fragment
    assert "<body>" in fragment
    assert '<meta charset="utf-8">' in document
    assert document.index("<head>") < document.index("<body>")


def test_model_options_include_current_antigravity_catalog_entries():
    options = set(get_model_options())

    expected = {
        "antigravity/claude-sonnet-4-6",
        "antigravity/claude-sonnet-4-6-thinking-low",
        "antigravity/claude-sonnet-4-6-thinking-medium",
        "antigravity/claude-sonnet-4-6-thinking-high",
        "antigravity/claude-sonnet-4-5",
        "antigravity/claude-sonnet-4-5-thinking-low",
        "antigravity/claude-sonnet-4-5-thinking-medium",
        "antigravity/claude-sonnet-4-5-thinking-high",
        "antigravity/claude-opus-4-6-thinking-low",
        "antigravity/claude-opus-4-6-thinking-medium",
        "antigravity/claude-opus-4-6-thinking-high",
        "antigravity/gemini-3.5-flash",
        "antigravity/gemini-3.5-flash-medium",
        "antigravity/gemini-3.5-flash-high",
        "antigravity/gemini-3.5-flash-low",
        "antigravity/gemini-3.1-pro-low",
        "antigravity/gemini-3.1-pro-high",
        "antigravity/gemini-3-pro-low",
        "antigravity/gemini-3-pro-high",
        "antigravity/gemini-3-flash",
        "antigravity/gemini-2.5-flash",
        "antigravity/gemini-2.5-pro",
    }

    assert expected.issubset(options)


def test_latest_proxy_version_prefers_newer_github_tag_without_git(monkeypatch):
    def fake_get(url, headers=None, timeout=15):
        assert url == antigravity_proxy.PROXY_GITHUB_API_TAGS
        return FakeHTTPResponse(
            json_data=[
                {"name": "v0.7.0", "zipball_url": "https://example.test/v0.7.0.zip"},
                {"name": "v1.7.1", "zipball_url": "https://example.test/v1.7.1.zip"},
            ]
        )

    monkeypatch.delenv("ANTIGRAVITY_PROXY_TAG", raising=False)
    monkeypatch.delenv("ANTIGRAVITY_PROXY_VERSION", raising=False)
    monkeypatch.setattr(antigravity_proxy.requests, "get", fake_get)

    release = antigravity_proxy._latest_proxy_release()

    assert release["version"] == "1.7.1"
    assert release["archive_url"].endswith("/archive/refs/tags/v1.7.1.zip")


def test_latest_antigravity_client_version_uses_google_public_bundle(monkeypatch):
    def fake_get(url, timeout=15):
        if url == antigravity_proxy.ANTIGRAVITY_SITE_URL:
            return FakeHTTPResponse(text='<script src="main.js" type="module"></script>')
        if url == "https://antigravity.google/main.js":
            return FakeHTTPResponse(
                text=(
                    'href:"https://storage.googleapis.com/antigravity-public/'
                    'antigravity-hub/2.2.1-5287492581195776/windows-x64/Antigravity-x64.exe",'
                    'version:"2.1.4<br>June 11, 2026"'
                )
            )
        raise AssertionError(url)

    monkeypatch.delenv("ANTIGRAVITY_CLIENT_VERSION", raising=False)
    monkeypatch.setattr(antigravity_proxy.requests, "get", fake_get)

    assert antigravity_proxy._latest_antigravity_client_version() == "2.2.1"


def test_patch_runtime_antigravity_client_version(tmp_path):
    headers_dir = tmp_path / "src" / "utils"
    headers_dir.mkdir(parents=True)
    headers_file = headers_dir / "headers.ts"
    headers_file.write_text(
        'const ANTIGRAVITY_VERSION = "2.0.1";\n'
        'export const ua = `antigravity/${ANTIGRAVITY_VERSION}`;\n',
        encoding="utf-8",
    )

    assert antigravity_proxy._patch_runtime_antigravity_client_version(str(tmp_path), "2.2.1")
    assert 'const ANTIGRAVITY_VERSION = "2.2.1";' in headers_file.read_text(encoding="utf-8")


def test_patch_runtime_gemini35_flash_support(tmp_path):
    utils_dir = tmp_path / "src" / "utils"
    utils_dir.mkdir(parents=True)
    transform_file = utils_dir / "transform.ts"
    transform_file.write_text(
        '    const nativelySupported = [\n'
        '      "gemini-3.1-pro-preview",\n'
        '      "gemini-3.5-flash-low",\n'
        '      "gemini-3-flash",\n'
        '  ];\n'
        '        if (isNative) {\n'
        '            if (baseModel.includes("gemini-3.5-flash")) {\n'
        '                googleModel = "gemini-3.5-flash-low";\n'
        '            } else if (baseModel.includes("gemini-3.1-pro")) {\n'
        '                googleModel = `gemini-3.1-pro-${extractedTier || "high"}`;\n'
        '            } else if (baseModel.includes("gemini-3-pro")) {\n'
        '                googleModel = `gemini-3-pro-${extractedTier || "high"}`;\n'
        '            }\n'
        '        }\n',
        encoding="utf-8",
    )

    assert antigravity_proxy._patch_runtime_gemini35_flash_support(str(tmp_path))

    content = transform_file.read_text(encoding="utf-8")
    assert '"gemini-3.5-flash-high"' in content
    assert '"gemini-3.5-flash-medium"' in content
    assert '"gemini-3.5-flash-low"' in content
    assert '"gemini-3.5-flash",' in content
    assert 'googleModel = "gemini-3.5-flash-low";' not in content
    assert '`gemini-3.5-flash-${extractedTier || "medium"}`' in content


def test_patch_runtime_account_reset_support_clears_capabilities(tmp_path):
    server_file = tmp_path / "src" / "server.ts"
    manager_file = tmp_path / "src" / "auth" / "manager.ts"
    server_file.parent.mkdir(parents=True)
    manager_file.parent.mkdir(parents=True)
    server_file.write_text(
        "for (const acc of accounts) {\n"
        "            acc.modelScores = {};\n"
        "            acc.history = [];\n"
        "}\n",
        encoding="utf-8",
    )
    manager_file.write_text(
        "resetAccount(account) {\n"
        "        account.modelScores = {};\n"
        "        account.history = [];\n"
        "}\n",
        encoding="utf-8",
    )

    assert antigravity_proxy._patch_runtime_account_reset_support(str(tmp_path))
    assert "acc.capabilities = {};" in server_file.read_text(encoding="utf-8")
    assert "account.capabilities = {};" in manager_file.read_text(encoding="utf-8")


def test_account_summary_strips_tokens_and_reports_unsupported_models():
    summary = antigravity_proxy._safe_account_summary(
        {
            "email": "user@example.test",
            "accessToken": "secret-access-token",
            "refreshToken": "secret-refresh-token",
            "projectId": "project-id",
            "healthScore": 42,
            "quota": [{"groupName": "Gemini", "quotaLeft": "100%", "resetIn": "1h"}],
            "modelScores": {"antigravity-gemini-3.5-flash-medium|sandbox": 90},
            "capabilities": {"antigravity-gemini-3.5-flash-high": False},
        }
    )

    assert "accessToken" not in summary
    assert "refreshToken" not in summary
    assert summary["email"] == "user@example.test"
    assert summary["quota"][0]["name"] == "Gemini"
    assert summary["unsupported_models"] == ["antigravity-gemini-3.5-flash-high"]


def test_find_proxy_launch_command_uses_npx_bun_for_downloaded_runtime(tmp_path, monkeypatch):
    def fake_candidate(name):
        return "npx" if name == "npx" else None

    runtime_dir = str(tmp_path)
    monkeypatch.delenv("ANTIGRAVITY_PROXY_LAUNCH_CMD", raising=False)
    monkeypatch.setattr(antigravity_proxy, "_candidate_executable", fake_candidate)

    cmd = antigravity_proxy._find_proxy_launch_command(runtime_dir)

    assert cmd[:5] == ["npx", "--yes", "--package", antigravity_proxy.BUN_NPM_PACKAGE, "bun"]
    assert cmd[-2] == "run"
    assert cmd[-1].endswith("src\\server.ts") or cmd[-1].endswith("src/server.ts")


def test_write_proxy_runtime_package_json_updates_stale_version(tmp_path):
    package_json = tmp_path / "package.json"
    package_json.write_text(
        json.dumps({"name": "glossarion-antigravity-proxy-data", "version": "0.0.0"}),
        encoding="utf-8",
    )

    antigravity_proxy._write_proxy_runtime_package_json(str(tmp_path), version="1.7.1")

    data = json.loads(package_json.read_text(encoding="utf-8"))
    assert data["name"] == "antigravity-proxy"
    assert data["version"] == "1.7.1"
    assert data["private"] is True
