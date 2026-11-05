"""AuthGem image output: image models ask for an image reply and get it back as a
[GENERATED_IMAGE:path] marker, the way the native Gemini route returns one."""
import base64
import json
import os
import threading

import pytest

import authgem_auth as auth
import unified_api_client as api

PNG = b"\x89PNG\r\n\x1a\nfake-image-bytes"
MESSAGES = [
    {"role": "system", "content": "Remove the text."},
    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]},
]


@pytest.fixture(autouse=True)
def isolation(monkeypatch):
    # Earlier tests in the same run can leave a hard stop or thinking settings behind.
    for name in ("ENABLE_IMAGE_OUTPUT_MODE", "ENABLE_GEMINI_THINKING", "THINKING_BUDGET", "GEMINI_THINKING_LEVEL",
                 "STREAM_THINKING_LOGS", "TRANSLATION_CANCELLED", "GRACEFUL_STOP", "ENABLE_STREAMING"):
        monkeypatch.delenv(name, raising=False)
    auth._cancel_event.clear()
    monkeypatch.setattr(api, "_save_outgoing_request", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(api, "_save_incoming_response", lambda *a, **kw: None, raising=False)
    yield
    auth._cancel_event.clear()


def image_part(data=PNG, mime="image/png", thought=False):
    part = {"inlineData": {"mimeType": mime, "data": base64.b64encode(data).decode("ascii")}}
    if thought:
        part["thought"] = True
    return part


# ---------------------------------------------------------------- request body

def test_image_request_asks_for_an_image_reply_without_thinking():
    body = auth._build_gemini_request_body(MESSAGES, 0.3, 4096, model="gemini-nano-banana-2.1",
                                           image_output={"size": "1K", "aspect_ratio": "auto"})
    config = body["generationConfig"]
    assert config["responseModalities"] == ["TEXT", "IMAGE"]
    assert config["imageConfig"] == {"imageSize": "1K"}
    assert "thinkingConfig" not in config


def test_image_request_passes_a_fixed_aspect_ratio():
    body = auth._build_gemini_request_body(MESSAGES, None, None, model="gemini-nano-banana-2.1",
                                           image_output={"size": "2K", "aspect_ratio": "3:4"})
    assert body["generationConfig"]["imageConfig"] == {"imageSize": "2K", "aspectRatio": "3:4"}


def test_text_request_is_unchanged():
    body = auth._build_gemini_request_body(MESSAGES, 0.3, 4096, model="gemini-2.5-flash")
    config = body["generationConfig"]
    assert "responseModalities" not in config and "imageConfig" not in config
    assert config["temperature"] == 0.3 and config["maxOutputTokens"] == 4096


def test_every_sender_forwards_image_output(monkeypatch):
    seen = []
    monkeypatch.setattr(auth, "_stream_gemini_common", lambda url, body, *a, **kw: seen.append(body) or {})
    monkeypatch.setattr(auth, "_code_assist_setup", lambda *a, **kw: "project")
    monkeypatch.setattr(auth, "detect_gcp_project", lambda *a, **kw: "project")
    opts = {"size": "1K", "aspect_ratio": "auto"}
    auth.send_chat_completion_aistudio("token", MESSAGES, model="m", image_output=opts)
    auth.send_chat_completion_aistudio_key("key", MESSAGES, model="m", image_output=opts)
    auth.send_chat_completion_vertex("token", MESSAGES, model="m", image_output=opts)
    configs = [b.get("request", b)["generationConfig"] for b in seen]
    assert len(configs) == 3
    assert all(c["responseModalities"] == ["TEXT", "IMAGE"] for c in configs)


# ---------------------------------------------------------------- reply parsing

def stream(*chunks):
    state = auth._new_gemini_stream_state()
    for chunk in chunks:
        auth._process_gemini_sse_line("data: " + json.dumps(chunk), state, lambda *a: None, False, 0.0)
    return auth._finalize_gemini_stream(state, lambda *a: None, False, 0.0)


def test_streamed_image_reply_is_collected_and_not_a_silent_block():
    result = stream(
        {"candidates": [{"content": {"parts": [{"text": "plan", "thought": True}, image_part(thought=True)]}}]},
        {"candidates": [{"content": {"parts": [image_part()]}, "finishReason": "STOP"}],
         "usageMetadata": {"candidatesTokenCount": 0}},
    )
    assert result["images"] == [{"mime": "image/png", "data": PNG}]
    assert result["_silent_block"] is False
    assert result["finish_reason"] == "stop"


def test_streamed_text_reply_has_no_images():
    result = stream({"candidates": [{"content": {"parts": [{"text": "hello"}]}, "finishReason": "STOP"}]})
    assert result["content"] == "hello"
    assert "images" not in result


def test_empty_reply_is_still_a_silent_block():
    result = stream({"candidates": [{"content": {"parts": []}, "finishReason": "STOP"}],
                     "usageMetadata": {"candidatesTokenCount": 0}})
    assert result["_silent_block"] is True


def test_non_streaming_image_reply_is_collected(monkeypatch):
    class Reply:
        status_code = 200

        def json(self):
            return {"candidates": [{"content": {"parts": [image_part(mime="image/jpeg")]}, "finishReason": "STOP"}],
                    "usageMetadata": {"candidatesTokenCount": 0}}

    monkeypatch.setenv("ENABLE_STREAMING", "0")
    import httpx
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: Reply())
    result = auth._stream_gemini_common("https://example.invalid/m:streamGenerateContent?alt=sse",
                                        {"generationConfig": {}}, {}, 30, lambda *a: None)
    assert result["images"] == [{"mime": "image/jpeg", "data": PNG}]
    assert result["_silent_block"] is False


# ---------------------------------------------------------------- client side

def make_client(monkeypatch, tmp_path, model):
    client = api.UnifiedClient("", model, str(tmp_path))
    monkeypatch.setattr(client, "_get_max_retries", lambda: 2)
    monkeypatch.setattr(client, "_is_stop_requested", lambda: False)
    return client


def test_image_model_turns_image_output_on(monkeypatch, tmp_path):
    client = make_client(monkeypatch, tmp_path, "authgem-vertex/gemini-nano-banana-2.1")
    assert client._authgem_image_output("gemini-nano-banana-2.1") == {"size": "1K", "aspect_ratio": "auto"}
    assert client._authgem_image_output("gemini-2.5-flash") is None
    client._force_image_output_mode = True
    client._forced_image_output_resolution = "2K"
    assert client._authgem_image_output("gemini-2.5-flash")["size"] == "2K"


def test_vision_only_scan_never_asks_for_an_image(monkeypatch, tmp_path):
    client = make_client(monkeypatch, tmp_path, "authgem-vertex/gemini-nano-banana-2.1")
    monkeypatch.setattr(client, "_is_vision_input_only_context", lambda *a, **kw: True)
    assert client._authgem_image_output("gemini-nano-banana-2.1") is None


def test_retry_loop_returns_a_generated_image_marker(monkeypatch, tmp_path):
    client = make_client(monkeypatch, tmp_path, "authgem-vertex/gemini-nano-banana-2.1")
    reply = {"content": "", "finish_reason": "stop", "usage": None, "_silent_block": False,
             "images": [{"mime": "image/png", "data": PNG}]}
    response = client._authgem_retry_loop(lambda attempt: dict(reply), "AuthGem-Vertex", "gemini-nano-banana-2.1",
                                          MESSAGES, 0.3, 4096, response_name="Inpainter_response_1.png")
    path = response.content[len("[GENERATED_IMAGE:"):-1]
    assert response.content.startswith("[GENERATED_IMAGE:") and os.path.isfile(path)
    assert open(path, "rb").read() == PNG
    assert "images" not in response.raw_response
    tls = client._get_thread_local_client()
    assert tls._last_generated_image_bytes == PNG and tls._last_generated_image_path == path


def test_vertex_route_sends_image_output_for_an_image_model(monkeypatch, tmp_path):
    client = make_client(monkeypatch, tmp_path, "authgem-vertex/gemini-nano-banana-2.1")

    class Store:
        def get_valid_access_token(self, auto_login=True):
            return "token"

    sent = {}

    def fake_send(**kwargs):
        sent.update(kwargs)
        return {"content": "", "finish_reason": "stop", "usage": None, "_silent_block": False,
                "images": [{"mime": "image/png", "data": PNG}]}

    monkeypatch.setattr(api, "_authgem_get_store", lambda: Store())
    monkeypatch.setattr(api, "_authgem_send_vertex", fake_send)
    monkeypatch.setattr(client, "_get_active_request_model", lambda: "authgem-vertex/gemini-nano-banana-2.1")
    response = client._send_authgem_vertex(MESSAGES, 0.3, 4096, "Inpainter_response_1.png")
    assert sent["model"] == "gemini-nano-banana-2.1"
    assert sent["image_output"] == {"size": "1K", "aspect_ratio": "auto"}
    assert response.content.startswith("[GENERATED_IMAGE:")
