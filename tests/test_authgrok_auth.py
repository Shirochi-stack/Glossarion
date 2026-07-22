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

    monkeypatch.setattr(authgrok.requests, "get", lambda *args, **kwargs: FakeResponse())
    discovery = {"jwks_uri": authgrok.XAI_OAUTH_JWKS_URL}

    assert authgrok._validate_id_token(token, "expected-nonce", discovery)["email"] == "reader@example.com"
    with pytest.raises(RuntimeError, match="nonce mismatch"):
        authgrok._validate_id_token(token, "wrong-nonce", discovery)


def test_unified_client_routes_authgrok_without_api_key():
    from unified_api_client import UnifiedClient

    assert UnifiedClient._provider_from_model_name("authgrok/grok-4.5") == "authgrok"
    assert UnifiedClient._provider_from_model_name("authgrok12/grok-build") == "authgrok"
    assert any("authgrok" in prefix for prefix in UnifiedClient._NO_API_KEY_PREFIXES)


def test_unified_client_maps_disabled_or_none_authgrok_reasoning_to_low(monkeypatch):
    from unified_api_client import UnifiedClient

    client = UnifiedClient.__new__(UnifiedClient)
    monkeypatch.setenv("ENABLE_GPT_THINKING", "0")
    monkeypatch.setenv("GPT_EFFORT", "none")
    assert client._get_authgrok_reasoning_param() == {"effort": "low"}

    monkeypatch.setenv("ENABLE_GPT_THINKING", "1")
    assert client._get_authgrok_reasoning_param() == {"effort": "low"}
