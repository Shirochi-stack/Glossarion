import base64
import json
import time
from urllib.parse import parse_qs, urlparse

import pytest

import authgrok_auth as authgrok


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def test_build_auth_url_uses_xai_pkce_state_and_nonce():
    url = authgrok.build_auth_url(
        "challenge-value",
        "state-value",
        "http://127.0.0.1:56121/callback",
        "nonce-value",
    )
    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == authgrok.XAI_OAUTH_AUTHORIZATION_URL
    assert query["client_id"] == [authgrok.XAI_OAUTH_CLIENT_ID]
    assert query["response_type"] == ["code"]
    assert query["code_challenge_method"] == ["S256"]
    assert query["code_challenge"] == ["challenge-value"]
    assert query["state"] == ["state-value"]
    assert query["nonce"] == ["nonce-value"]
    assert "conversations:write" in query["scope"][0]


def test_build_auth_url_can_force_a_fresh_numbered_account_login():
    url = authgrok.build_auth_url(
        "challenge-value",
        "state-value",
        "http://127.0.0.1:56121/callback",
        "nonce-value",
        force_account_selection=True,
    )

    query = parse_qs(urlparse(url).query)
    assert query["prompt"] == ["login"]
    assert query["max_age"] == ["0"]


def test_oidc_discovery_cache_honors_server_max_age(monkeypatch):
    calls = []

    class FakeResponse:
        is_redirect = False
        is_permanent_redirect = False
        headers = {"Cache-Control": "public, max-age=60"}

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "issuer": authgrok.XAI_OAUTH_ISSUER,
                "authorization_endpoint": authgrok.XAI_OAUTH_AUTHORIZATION_URL,
                "token_endpoint": authgrok.XAI_OAUTH_TOKEN_URL,
                "jwks_uri": authgrok.XAI_OAUTH_JWKS_URL,
                "id_token_signing_alg_values_supported": ["ES256"],
            }

    monkeypatch.setattr(authgrok, "_oidc_discovery_cache", None)
    monkeypatch.setattr(
        authgrok.requests,
        "get",
        lambda *args, **kwargs: calls.append((args, kwargs)) or FakeResponse(),
    )

    first = authgrok._load_oidc_discovery()
    second = authgrok._load_oidc_discovery()

    assert first == second
    assert len(calls) == 1


def test_numbered_login_signs_out_xai_then_returns_to_device_authorization():
    url = authgrok.build_signed_out_device_url(
        "https://accounts.x.ai/oauth2/device?user_code=ABCD-EFGH"
    )
    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == (
        authgrok.XAI_ACCOUNTS_SIGN_OUT_URL
    )
    assert query["redirect"] == ["oauth2-provider"]
    assert query["return_to"] == ["/oauth2/device?user_code=ABCD-EFGH"]


def test_numbered_login_rejects_untrusted_device_verification_url():
    with pytest.raises(RuntimeError, match="unsafe verification URL"):
        authgrok.build_signed_out_device_url(
            "https://example.test/oauth2/device?user_code=ABCD-EFGH"
        )


def test_numbered_oauth_flow_uses_auto_polled_device_authorization(monkeypatch):
    opened = []
    polled = []
    device_code = {
        "device_code": "device-secret",
        "user_code": "ABCD-EFGH",
        "verification_uri": "https://accounts.x.ai/oauth2/device",
        "verification_uri_complete": (
            "https://accounts.x.ai/oauth2/device?user_code=ABCD-EFGH"
        ),
        "expires_in": 600,
        "interval": 5,
    }
    monkeypatch.setattr(
        authgrok,
        "_load_oidc_discovery",
        lambda: {"jwks_uri": authgrok.XAI_OAUTH_JWKS_URL},
    )
    monkeypatch.setattr(authgrok, "request_device_code", lambda timeout=30: device_code)
    monkeypatch.setattr(authgrok, "_warm_jwks_cache", lambda _discovery: None)
    monkeypatch.setattr(
        authgrok,
        "_open_oauth_browser",
        lambda url: opened.append(url),
    )
    monkeypatch.setattr(
        authgrok,
        "poll_device_code_tokens",
        lambda value, timeout=300: (
            polled.append((value, timeout))
            or {
                "id_token": "signed-token",
                "access_token": "access-token",
                "refresh_token": "refresh-token",
            }
        ),
    )
    monkeypatch.setattr(
        authgrok,
        "_validate_id_token",
        lambda *_args: {"email": "second@example.test", "sub": "second"},
    )

    tokens = authgrok.run_oauth_flow(force_account_selection=True)

    assert len(opened) == 1
    opened_query = parse_qs(urlparse(opened[0]).query)
    assert opened[0].startswith(authgrok.XAI_ACCOUNTS_SIGN_OUT_URL)
    assert opened_query["return_to"] == ["/oauth2/device?user_code=ABCD-EFGH"]
    assert polled == [(device_code, 300)]
    assert tokens["account"]["email"] == "second@example.test"


def test_request_device_code_uses_xai_device_endpoint(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200
        is_redirect = False
        is_permanent_redirect = False

        def json(self):
            return {
                "device_code": "device-secret",
                "user_code": "ABCD-EFGH",
                "verification_uri": "https://accounts.x.ai/oauth2/device",
                "verification_uri_complete": (
                    "https://accounts.x.ai/oauth2/device?user_code=ABCD-EFGH"
                ),
                "expires_in": 900,
                "interval": 4,
            }

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(authgrok.requests, "post", fake_post)

    result = authgrok.request_device_code()

    assert captured["url"] == authgrok.XAI_OAUTH_DEVICE_CODE_URL
    assert captured["data"]["referrer"] == "grok-build"
    assert result["user_code"] == "ABCD-EFGH"
    assert result["interval"] == 4


def test_device_token_polling_waits_until_authorized(monkeypatch):
    responses = [
        (400, {"error": "authorization_pending"}),
        (
            200,
            {
                "access_token": "access-token",
                "refresh_token": "refresh-token",
                "id_token": "signed-token",
                "expires_in": 3600,
            },
        ),
    ]

    class FakeResponse:
        is_redirect = False
        is_permanent_redirect = False

        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        def json(self):
            return self._payload

    monkeypatch.setattr(authgrok.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        authgrok.requests,
        "post",
        lambda *_args, **_kwargs: FakeResponse(*responses.pop(0)),
    )

    result = authgrok.poll_device_code_tokens(
        {
            "device_code": "device-secret",
            "expires_in": 600,
            "interval": 1,
        }
    )

    assert result["access_token"] == "access-token"
    assert result["refresh_token"] == "refresh-token"
    assert responses == []


def test_numbered_store_auto_login_forces_account_selection(tmp_path, monkeypatch):
    captured = []

    def fake_login(*, force_account_selection=False, timeout=300):
        captured.append(force_account_selection)
        return {
            "access_token": "numbered-token",
            "refresh_token": "numbered-refresh",
            "expires_at": time.time() + 3600,
        }

    monkeypatch.setattr(authgrok, "run_oauth_flow", fake_login)
    store = authgrok.AuthGrokTokenStore(
        str(tmp_path / "authgrok_tokens_2.json"),
        account_id=2,
    )

    assert store.get_valid_access_token(auto_login=True) == "numbered-token"
    assert captured == [True]


def test_authgrok_pool_deduplicates_emails_and_rotates_start(monkeypatch):
    class FakeStore:
        def __init__(self, account_id, email):
            self.account_id = account_id
            self._tokens = {
                "access_token": f"token-{account_id}",
                "account": {"email": email, "subject": f"subject-{account_id}"},
            }

        def load_tokens(self):
            return self._tokens

    stores = {
        0: FakeStore(0, "first@example.test"),
        1: FakeStore(1, "second@example.test"),
        2: FakeStore(2, "FIRST@example.test"),
    }
    monkeypatch.setattr(authgrok, "_numbered_account_ids", lambda: [1, 2])
    monkeypatch.setattr(authgrok, "get_store", lambda account_id=0: stores[int(account_id or 0)])
    monkeypatch.setattr(authgrok, "_pool_rotation_cursor", 0)

    assert [slot for slot, _store in authgrok.get_account_pool()] == [0, 1]
    assert [slot for slot, _store in authgrok.get_rotating_account_pool()] == [0, 1]
    assert [slot for slot, _store in authgrok.get_rotating_account_pool()] == [1, 0]


def test_get_saved_account_ids_returns_each_credential_bearing_slot(monkeypatch):
    class FakeStore:
        def __init__(self, account_id, tokens):
            self.account_id = account_id
            self._tokens = tokens

        def load_tokens(self):
            return self._tokens

    stores = {
        0: FakeStore(0, {"access_token": "default-token"}),
        1: FakeStore(1, {"refresh_token": "numbered-refresh"}),
        2: FakeStore(2, {}),
    }
    monkeypatch.setattr(authgrok, "_numbered_account_ids", lambda: [1, 2])
    monkeypatch.setattr(authgrok, "get_store", lambda account_id=0: stores[int(account_id or 0)])

    assert authgrok.get_saved_account_ids() == [0, 1]


def test_get_next_account_id_uses_first_unreserved_positive_gap(monkeypatch):
    monkeypatch.setattr(authgrok, "get_saved_account_ids", lambda: [0, 1, 3])

    assert authgrok.get_next_account_id() == 2
    assert authgrok.get_next_account_id([2]) == 4


def test_numbered_slot_rejects_an_account_already_saved_elsewhere(monkeypatch):
    class FakeStore:
        def load_tokens(self):
            return {
                "account": {
                    "email": "first@example.test",
                    "subject": "first-subject",
                }
            }

    monkeypatch.setattr(authgrok, "get_saved_account_ids", lambda: [0])
    monkeypatch.setattr(authgrok, "get_store", lambda _account_id=0: FakeStore())

    with pytest.raises(RuntimeError, match=r"already saved in Grok account slot #0"):
        authgrok.validate_account_slot_tokens(
            2,
            {"account": {"email": "FIRST@example.test", "subject": "other"}},
        )


def test_numbered_slot_accepts_a_distinct_account(monkeypatch):
    class FakeStore:
        def load_tokens(self):
            return {"account": {"email": "first@example.test"}}

    monkeypatch.setattr(authgrok, "get_saved_account_ids", lambda: [0])
    monkeypatch.setattr(authgrok, "get_store", lambda _account_id=0: FakeStore())

    authgrok.validate_account_slot_tokens(
        2,
        {"account": {"email": "second@example.test"}},
    )


def test_build_responses_body_converts_messages_images_and_reasoning():
    body = authgrok._build_responses_body(
        [
            {"role": "system", "content": "Translate faithfully."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Translate this image."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            },
            {"role": "assistant", "content": "Prior answer"},
        ],
        "grok-4.5",
        temperature=0.2,
        max_tokens=1234,
        reasoning={"effort": "xhigh", "summary": "detailed"},
    )

    assert body["model"] == "grok-4.5"
    assert body["instructions"] == "Translate faithfully."
    assert body["store"] is False
    assert body["stream"] is True
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["max_output_tokens"] == 1234
    assert body["temperature"] == 0.2
    assert body["reasoning"] == {"effort": "high"}
    assert body["input"][0]["content"][1]["type"] == "input_image"
    assert body["input"][1]["content"] == [{"type": "output_text", "text": "Prior answer"}]


def test_build_responses_body_maps_explicit_none_reasoning_to_low():
    body = authgrok._build_responses_body(
        [{"role": "user", "content": "Hello"}],
        "grok-4.5",
        reasoning={"effort": "none"},
    )

    assert body["reasoning"] == {"effort": "low"}


def test_stream_log_breaks_after_html_block_tags_across_deltas():
    text_buffer = []
    logged = []

    authgrok._append_stream_log_delta(text_buffer, "<h2>Title</h", logged.append)
    authgrok._append_stream_log_delta(
        text_buffer,
        "2><p>First paragraph.</p><p>Second paragraph",
        logged.append,
    )
    authgrok._append_stream_log_delta(text_buffer, ".</p>", logged.append)
    authgrok._flush_stream_log_buffer(text_buffer, logged.append)

    assert logged == [
        "<h2>Title</h2>",
        "<p>First paragraph.</p>",
        "<p>Second paragraph.</p>",
    ]


def test_batch_stream_log_respects_forced_stream_toggle(monkeypatch):
    monkeypatch.setenv("LOG_STREAM_CHUNKS", "1")
    monkeypatch.setenv("BATCH_TRANSLATION", "1")
    monkeypatch.setenv("ALLOW_AUTHGPT_BATCH_STREAM_LOGS", "0")
    assert authgrok._stream_logging_enabled() is False

    monkeypatch.setenv("ALLOW_AUTHGPT_BATCH_STREAM_LOGS", "1")
    assert authgrok._stream_logging_enabled() is True

    monkeypatch.setenv("BATCH_TRANSLATION", "0")
    monkeypatch.setenv("LOG_STREAM_CHUNKS", "0")
    assert authgrok._stream_logging_enabled() is False


def test_reasoning_summary_stream_is_separate_from_response_text(monkeypatch):
    monkeypatch.setenv("STREAM_THINKING_LOGS", "1")
    state = authgrok._new_stream_display_state()
    logged = []
    events = [
        {"type": "response.reasoning_summary_text.delta", "delta": "Check the "},
        {"type": "response.reasoning_summary_text.delta", "delta": "translation.\nPreserve HTML."},
        {"type": "response.output_text.delta", "delta": "<p>Translated text.</p>"},
    ]
    for event in events:
        authgrok._process_stream_event(event, state, logged.append, log_stream=True)

    lines = [f"data: {json.dumps(event)}" for event in events]
    lines.append(
        'data: {"type":"response.completed","response":{"id":"resp_2","status":"completed","output":[]}}'
    )
    result = authgrok._finalize_stream_result(lines, state, logged.append, log_stream=True)

    assert result["content"] == "<p>Translated text.</p>"
    assert result["thinking_text"] == "Check the translation.\nPreserve HTML."
    assert result["thinking_chunks"] == 2
    assert logged[0] == "🧠 [authgrok] Thinking..."
    assert "    Check the translation." in logged
    assert "    Preserve HTML." in logged
    assert "📡 AuthGrok: Text streaming..." in logged
    assert "<p>Translated text.</p>" in logged
    assert not any("Check the translation" in line and "Translated text" in line for line in logged)


def test_reasoning_summary_is_captured_but_hidden_when_toggle_is_off(monkeypatch):
    monkeypatch.setenv("BATCH_TRANSLATION", "1")
    monkeypatch.setenv("STREAM_THINKING_LOGS", "0")
    state = authgrok._new_stream_display_state()
    logged = []
    authgrok._process_stream_event(
        {"type": "response.reasoning_text.delta", "delta": "Hidden summary"},
        state,
        logged.append,
        log_stream=True,
    )
    authgrok._process_stream_event(
        {"type": "response.output_text.delta", "delta": "Visible answer"},
        state,
        logged.append,
        log_stream=True,
    )
    result = authgrok._finalize_stream_result([], state, logged.append, log_stream=True)

    assert result["thinking_text"] == "Hidden summary"
    assert logged == ["Visible answer"]


def test_reasoning_summary_always_streams_outside_batch_mode(monkeypatch):
    monkeypatch.setenv("BATCH_TRANSLATION", "0")
    monkeypatch.setenv("STREAM_THINKING_LOGS", "0")
    state = authgrok._new_stream_display_state()
    logged = []

    authgrok._process_stream_event(
        {"type": "response.reasoning_summary_text.delta", "delta": "Always visible"},
        state,
        logged.append,
        log_stream=False,
    )
    authgrok._process_stream_event(
        {"type": "response.output_text.delta", "delta": "Final answer"},
        state,
        logged.append,
        log_stream=False,
    )

    assert logged[0] == "🧠 [authgrok] Thinking..."
    assert "    Always visible" in logged
    assert "📡 AuthGrok: Text streaming..." not in logged
    assert "Final answer" not in logged


def test_parse_responses_sse_uses_deltas_and_completed_usage():
    stream = "\n".join([
        'event: response.output_text.delta',
        'data: {"type":"response.output_text.delta","delta":"Hello "}',
        'data: {"type":"response.output_text.delta","delta":"world"}',
        'data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[],"usage":{"input_tokens":7,"output_tokens":2,"total_tokens":9}}}',
        'data: [DONE]',
    ])

    result = authgrok._parse_sse_responses(stream)

    assert result["content"] == "Hello world"
    assert result["finish_reason"] == "stop"
    assert result["conversation_id"] == "resp_1"
    assert result["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 2,
        "total_tokens": 9,
    }


def test_terminal_sse_detection_covers_failure_and_incomplete_events():
    assert authgrok._is_terminal_sse_line(
        'data: {"type":"response.incomplete","response":{"status":"incomplete"}}'
    )
    assert authgrok._is_terminal_sse_line(
        'data: {"type":"response.failed","response":{"status":"failed"}}'
    )
    assert authgrok._is_terminal_sse_line('data: {"type":"error","message":"bad request"}')
    assert authgrok._is_terminal_sse_line("data: [DONE]")
    assert not authgrok._is_terminal_sse_line(
        'data: {"type":"response.output_text.delta","delta":"still streaming"}'
    )


def test_proxy_headers_bind_model_and_request_identity():
    headers = authgrok._proxy_headers(
        "secret-token",
        "authgrok/grok-4.5",
        session_id="session-1",
        request_id="request-1",
    )

    assert headers["Authorization"] == "Bearer secret-token"
    assert headers["Accept"] == "text/event-stream"
    assert headers["X-XAI-Token-Auth"] == "xai-grok-cli"
    assert headers["x-grok-client-identifier"] == "glossarion"
    assert headers["x-grok-model-override"] == "grok-4.5"
    assert headers["x-grok-conv-id"] == "session-1"
    assert headers["x-grok-session-id"] == "session-1"
    assert headers["x-grok-req-id"] == "request-1"


def test_refresh_access_token_uses_pinned_public_client(monkeypatch):
    captured = {}

    def fake_post(payload, timeout=30):
        captured.update(payload)
        return {"access_token": "new", "refresh_token": "rotated", "expires_at": time.time() + 3600}

    monkeypatch.setattr(authgrok, "_post_oauth_token", fake_post)

    result = authgrok.refresh_access_token("old-refresh")

    assert captured == {
        "grant_type": "refresh_token",
        "refresh_token": "old-refresh",
        "client_id": authgrok.XAI_OAUTH_CLIENT_ID,
    }
    assert result["access_token"] == "new"
    assert result["refresh_token"] == "rotated"


def test_validate_id_token_verifies_es256_signature_and_nonce(monkeypatch):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

    private_key = ec.generate_private_key(ec.SECP256R1())
    numbers = private_key.public_key().public_numbers()
    header = {"alg": "ES256", "kid": "test-key", "typ": "JWT"}
    claims = {
        "iss": authgrok.XAI_OAUTH_ISSUER,
        "aud": authgrok.XAI_OAUTH_CLIENT_ID,
        "sub": "account-123",
        "iat": int(time.time()) - 1,
        "exp": int(time.time()) + 300,
        "nonce": "expected-nonce",
        "email": "reader@example.com",
    }
    encoded_header = _b64url(json.dumps(header, separators=(",", ":")).encode())
    encoded_claims = _b64url(json.dumps(claims, separators=(",", ":")).encode())
    signed = f"{encoded_header}.{encoded_claims}".encode("ascii")
    der_signature = private_key.sign(signed, ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(der_signature)
    token = f"{encoded_header}.{encoded_claims}.{_b64url(r.to_bytes(32, 'big') + s.to_bytes(32, 'big'))}"

    class FakeResponse:
        is_redirect = False
        is_permanent_redirect = False

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "keys": [{
                    "kid": "test-key",
                    "kty": "EC",
                    "crv": "P-256",
                    "alg": "ES256",
                    "x": _b64url(numbers.x.to_bytes(32, "big")),
                    "y": _b64url(numbers.y.to_bytes(32, "big")),
                }]
            }

    get_calls = []
    monkeypatch.setattr(
        authgrok,
        "_jwks_cache",
        ({"keys": [{"kid": "previous-key"}]}, time.monotonic() + 60),
    )
    monkeypatch.setattr(
        authgrok.requests,
        "get",
        lambda *args, **kwargs: get_calls.append((args, kwargs)) or FakeResponse(),
    )
    discovery = {"jwks_uri": authgrok.XAI_OAUTH_JWKS_URL}

    assert authgrok._validate_id_token(token, "expected-nonce", discovery)["email"] == "reader@example.com"
    with pytest.raises(RuntimeError, match="nonce mismatch"):
        authgrok._validate_id_token(token, "wrong-nonce", discovery)
    assert len(get_calls) == 1


def test_unified_client_routes_authgrok_without_api_key():
    from unified_api_client import UnifiedClient

    assert UnifiedClient._provider_from_model_name("authgrok/grok-4.5") == "authgrok"
    assert UnifiedClient._provider_from_model_name("authgrok12/grok-build") == "authgrok"
    assert any("authgrok" in prefix for prefix in UnifiedClient._NO_API_KEY_PREFIXES)


def test_authgrok_http_error_reads_only_provider_error_metadata():
    error = authgrok.AuthGrokHTTPError(
        403,
        json.dumps({
            "code": "permission-denied",
            "error": (
                "Content violates usage guidelines. "
                "Failed check: SAFETY_CHECK_TYPE_CSAM"
            ),
        }),
    )

    assert error.is_safety_error is True
    assert error.safety_check == "SAFETY_CHECK_TYPE_CSAM"
    assert error.provider_code == "permission-denied"


def test_authgrok_http_error_ignores_safety_words_in_echoed_request_content():
    error = authgrok.AuthGrokHTTPError(
        403,
        json.dumps({
            "code": "permission-denied",
            "error": "This account is not authorized",
            "request": {
                "input": (
                    "A story literally says Content violates usage guidelines and "
                    "SAFETY_CHECK_TYPE_CSAM in dialogue."
                ),
            },
        }),
    )

    assert error.is_safety_error is False
    assert error.safety_check is None


def test_authgrok_safety_http_error_maps_to_prohibited_without_retry(monkeypatch):
    import unified_api_client as unified

    send_calls = []

    class FakeStore:
        def get_valid_access_token(self, auto_login=True, force_refresh=False):
            return "access-token"

    monkeypatch.setattr(unified, "_authgrok_get_store", lambda: FakeStore())

    def fake_send(**kwargs):
        send_calls.append(kwargs)
        raise authgrok.AuthGrokHTTPError(
            403,
            json.dumps({
                "code": "permission-denied",
                "error": {
                    "message": (
                        "Content violates usage guidelines. "
                        "Failed check: SAFETY_CHECK_TYPE_CSAM"
                    ),
                },
            }),
        )

    monkeypatch.setattr(unified, "_authgrok_send", fake_send)
    client = unified.UnifiedClient.__new__(unified.UnifiedClient)
    client.request_timeout = 30
    client._get_active_request_model = lambda: "authgrok/grok-4.6"
    client._get_max_retries = lambda: 7
    client._is_stop_requested = lambda: False
    client._should_abort_retry = lambda: False
    client._get_authgrok_reasoning_param = lambda: {"effort": "low"}
    client._log_once = lambda _message: None

    with pytest.raises(unified.UnifiedClientError) as raised:
        client._send_authgrok(
            [{"role": "user", "content": "Translate this chapter."}],
            temperature=None,
            max_tokens=100,
            response_name="test",
        )

    assert raised.value.error_type == "prohibited_content"
    assert raised.value.http_status == 403
    assert raised.value.details == {
        "provider": "authgrok",
        "source": "authgrok_http_error",
        "safety_check": "SAFETY_CHECK_TYPE_CSAM",
        "provider_code": "permission-denied",
    }
    assert len(send_calls) == 1


def test_authgrok_safety_http_error_skips_global_api_retries(monkeypatch):
    import unified_api_client as unified

    send_calls = []

    class FakeStore:
        def get_valid_access_token(self, auto_login=True, force_refresh=False):
            return "access-token"

    monkeypatch.setattr(unified, "_authgrok_get_store", lambda: FakeStore())

    def fake_send(**kwargs):
        send_calls.append(kwargs)
        raise authgrok.AuthGrokHTTPError(
            403,
            json.dumps({
                "code": "permission-denied",
                "error": (
                    "Content violates usage guidelines. "
                    "Failed check: SAFETY_CHECK_TYPE_CSAM"
                ),
            }),
        )

    monkeypatch.setattr(unified, "_authgrok_send", fake_send)
    monkeypatch.setattr(unified, "_authgrok_reset_cancel", lambda: None)
    monkeypatch.setenv("DISABLE_REFUSAL_CHECKS", "1")
    monkeypatch.setenv("MAX_RETRIES", "3")
    monkeypatch.setenv("USE_FALLBACK_KEYS", "0")
    monkeypatch.setenv("USE_GLOSSARY_KEYS", "0")
    monkeypatch.setenv("USE_GLOSSARY_REFINEMENT_KEYS", "0")

    client = unified.UnifiedClient(
        "",
        "authgrok/grok-4.6",
        _skip_cancel_reset=True,
    )
    monkeypatch.setattr(client, "_save_payload", lambda *args, **kwargs: None)
    monkeypatch.setattr(client, "_save_failed_request", lambda *args, **kwargs: None)
    monkeypatch.setattr(client, "_track_stats", lambda *args, **kwargs: None)

    _content, finish_reason = client._send_internal(
        [{"role": "user", "content": "Translate this chapter."}],
        temperature=0.2,
        max_tokens=1024,
        context="translation",
        request_id="authgrok-filter-test",
    )

    assert finish_reason == "prohibited_content"
    assert len(send_calls) == 1


def test_authgrok_ordinary_403_remains_an_auth_error(monkeypatch):
    import unified_api_client as unified

    class FakeStore:
        def get_valid_access_token(self, auto_login=True, force_refresh=False):
            return "access-token"

    monkeypatch.setattr(unified, "_authgrok_get_store", lambda: FakeStore())
    monkeypatch.setattr(
        unified,
        "_authgrok_send",
        lambda **_kwargs: (_ for _ in ()).throw(
            authgrok.AuthGrokHTTPError(
                403,
                json.dumps({
                    "code": "permission-denied",
                    "error": "This account is not authorized for the requested model",
                }),
            )
        ),
    )
    client = unified.UnifiedClient.__new__(unified.UnifiedClient)
    client.request_timeout = 30
    client._get_active_request_model = lambda: "authgrok/grok-4.6"
    client._get_max_retries = lambda: 7
    client._is_stop_requested = lambda: False
    client._should_abort_retry = lambda: False
    client._get_authgrok_reasoning_param = lambda: {"effort": "low"}
    client._log_once = lambda _message: None

    with pytest.raises(unified.UnifiedClientError) as raised:
        client._send_authgrok(
            [{"role": "user", "content": "Translate this chapter."}],
            temperature=None,
            max_tokens=100,
            response_name="test",
        )

    assert raised.value.error_type == "auth_error"


def test_authgrok_zero_pool_fails_over_to_next_saved_account(monkeypatch):
    import unified_api_client as unified

    token_calls = []
    sent_tokens = []
    logged = []

    class FakeStore:
        def __init__(self, token, email):
            self.token = token
            self.account_info = {"email": email}

        def get_valid_access_token(self, auto_login=True, force_refresh=False):
            token_calls.append((self.token, auto_login, force_refresh))
            return self.token

    first = FakeStore("first-token", "first@example.test")
    second = FakeStore("second-token", "second@example.test")
    monkeypatch.setattr(
        unified,
        "_authgrok_get_rotating_account_pool",
        lambda: [(1, first), (2, second)],
    )

    def fake_send(**kwargs):
        sent_tokens.append(kwargs["access_token"])
        if kwargs["access_token"] == "first-token":
            raise RuntimeError("AuthGrok HTTP 429: quota exhausted")
        return {"content": "rotated", "finish_reason": "stop", "usage": None}

    monkeypatch.setattr(unified, "_authgrok_send", fake_send)
    monkeypatch.setattr(
        unified,
        "print",
        lambda message, *args, **kwargs: logged.append(str(message)),
    )
    client = unified.UnifiedClient.__new__(unified.UnifiedClient)
    client.request_timeout = 30
    client._get_active_request_model = lambda: "authgrok0/grok-4.5"
    client._get_max_retries = lambda: 1
    client._is_stop_requested = lambda: False
    client._should_abort_retry = lambda: False
    client._get_authgrok_reasoning_param = lambda: {"effort": "low"}
    client._log_once = lambda _message: None

    result = client._send_authgrok(
        [{"role": "user", "content": "hello"}],
        temperature=None,
        max_tokens=100,
        response_name="test",
    )

    assert result.content == "rotated"
    assert sent_tokens == ["first-token", "second-token"]
    assert token_calls == [
        ("first-token", False, False),
        ("second-token", False, False),
    ]
    assert "🔄 AuthGrok pool: Using account slot #1 (first@example.test)" in logged
    assert "🔄 AuthGrok pool: Using account slot #2 (second@example.test)" in logged


def test_unified_client_maps_disabled_or_none_authgrok_reasoning_to_low(monkeypatch):
    from unified_api_client import UnifiedClient

    client = UnifiedClient.__new__(UnifiedClient)
    monkeypatch.setenv("ENABLE_GPT_THINKING", "0")
    monkeypatch.setenv("GPT_EFFORT", "none")
    assert client._get_authgrok_reasoning_param() == {"effort": "low"}

    monkeypatch.setenv("ENABLE_GPT_THINKING", "1")
    assert client._get_authgrok_reasoning_param() == {"effort": "low"}
