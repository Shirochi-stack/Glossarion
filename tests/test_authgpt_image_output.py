"""authgpt/ image output: the ChatGPT Codex responses endpoint runs the Responses
image_generation tool (checked live on 2026-11-06 with gpt-6.1-sol: generate and
edit both return an image_generation_call item with the PNG as base64). Without
the tool in the request the model can only answer that it cannot make images.

A local HTTP server plays the Codex backend: it streams the event sequence the
real endpoint sent, and answers like the real model when no tool is offered."""
import base64
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

import authgpt_auth as auth  # noqa: E402
import unified_api_client as api  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\nfake-authgpt-image"
REFUSAL = "I can't directly edit or generate an image in this chat."
MESSAGES = [
    {"role": "system", "content": "Remove the text from this page and redraw the background."},
    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]},
]


class FakeCodex:
    """The /codex/responses endpoint, recording every request body."""

    def __init__(self, image_status="completed"):
        self.bodies = []
        self.image_status = image_status

    def events(self, body):
        yield {"type": "response.created", "response": {"id": "resp_1", "status": "in_progress"}}
        if body.get("tools") and body.get("reasoning"):
            # what the live backend streams for any reasoning effort next to the image tool
            error = {"type": "image_generation_user_error", "code": "invalid_value", "param": "complexity",
                     "message": f"Complexity '{body['reasoning']['effort']}' is not supported for this image model."}
            yield {"type": "error", "error": error}
            yield {"type": "response.failed", "response": {"id": "resp_1", "status": "failed", "error": error}}
            return
        if body.get("tools") and body["tools"][0].get("type") == "image_generation":
            item = {"id": "ig_1", "type": "image_generation_call", "status": "in_progress"}
            yield {"type": "response.output_item.added", "output_index": 0, "item": item}
            yield {"type": "response.image_generation_call.in_progress", "item_id": "ig_1"}
            yield {"type": "response.image_generation_call.generating", "item_id": "ig_1"}
            done = dict(item, status=self.image_status, action="edit", background="opaque",
                        output_format=body["tools"][0].get("output_format", "png"), quality="low",
                        size="1254x1254", revised_prompt="Remove the text.")
            if self.image_status == "completed":
                done["result"] = base64.b64encode(PNG).decode("ascii")
            yield {"type": "response.image_generation_call.completed", "item_id": "ig_1"}
            yield {"type": "response.output_item.done", "output_index": 0, "item": done}
            text = ""
        else:
            text = REFUSAL
            yield {"type": "response.output_text.delta", "delta": text}
        message = {"id": "msg_1", "type": "message", "role": "assistant",
                   "content": [{"type": "output_text", "text": text}]}
        yield {"type": "response.output_item.done", "output_index": 1, "item": message}
        yield {"type": "response.completed",
               "response": {"id": "resp_1", "status": "completed", "output": [],
                            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}}

    def __enter__(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.bodies.append(body)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for event in fake.events(body):
                    self.wfile.write(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture(autouse=True)
def isolation(monkeypatch):
    for name in ("ENABLE_IMAGE_OUTPUT_MODE", "OPENAI_IMAGE_SIZE", "OPENAI_IMAGE_QUALITY", "IMAGE_OUTPUT_ASPECT_RATIO",
                 "TRANSLATION_CANCELLED", "GRACEFUL_STOP", "BATCH_TRANSLATION", "AUTHGPT_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    auth.reset_cancel()
    monkeypatch.setattr(api, "_save_outgoing_request", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(api, "_save_incoming_response", lambda *a, **kw: None, raising=False)
    yield
    auth.reset_cancel()


IMAGE_OPTS = {"size": "auto", "quality": "auto", "output_format": "png"}


# ---------------------------------------------------------------- request body

def test_image_request_carries_the_required_image_tool():
    body = auth._build_responses_body(MESSAGES, "gpt-6.1-sol", max_tokens=4096, image_output=IMAGE_OPTS,
                                      reasoning={"effort": "low"})
    assert body["tools"] == [{"type": "image_generation", "size": "auto", "quality": "auto", "output_format": "png"}]
    assert body["tool_choice"] == {"type": "image_generation"}
    assert body["input"][0]["content"] == [{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}]
    # the backend hands the effort to the image model, which rejects every value
    assert "reasoning" not in body


def test_text_request_has_no_tools():
    body = auth._build_responses_body(MESSAGES, "gpt-6.1-sol", max_tokens=4096, reasoning={"effort": "low"})
    assert "tools" not in body and "tool_choice" not in body
    assert body["reasoning"] == {"effort": "low"}


# ---------------------------------------------------------------- streamed reply

@pytest.mark.parametrize("transport", ["httpx", "requests"])
def test_image_reply_is_streamed_back(monkeypatch, transport):
    if transport == "requests":
        monkeypatch.setitem(sys.modules, "httpx", None)
    logs = []
    with FakeCodex() as server:
        result = auth.send_chat_completion("token", MESSAGES, model="gpt-6.1-sol", max_tokens=4096,
                                           base_url=server.base_url, log_fn=logs.append, image_output=IMAGE_OPTS,
                                           reasoning={"effort": "low"})
    assert server.bodies[0]["tool_choice"] == {"type": "image_generation"}
    assert "reasoning" not in server.bodies[0]
    assert len(result["images"]) == 1
    image = result["images"][0]
    assert image["data"] == PNG and image["mime"] == "image/png" and image["action"] == "edit"
    assert result["content"] == "" and result["finish_reason"] == "stop"
    text = "\n".join(logs)
    assert "sending the image_generation tool" in text
    assert "reasoning_effort=low is not sent with the image tool" in text
    assert "Generating image" in text and "Image returned (edit, 1254x1254" in text
    assert "Empty content" not in text and "has no image" not in text


def test_text_request_gets_the_model_text(monkeypatch):
    with FakeCodex() as server:
        result = auth.send_chat_completion("token", MESSAGES, model="gpt-6.1-sol",
                                           base_url=server.base_url, log_fn=lambda *a: None)
    assert "tools" not in server.bodies[0]
    assert result["content"] == REFUSAL and "images" not in result


def test_failed_image_call_is_reported(monkeypatch):
    logs = []
    with FakeCodex(image_status="failed") as server:
        result = auth.send_chat_completion("token", MESSAGES, model="gpt-6.1-sol", base_url=server.base_url,
                                           log_fn=logs.append, image_output=IMAGE_OPTS)
    assert "images" not in result
    text = "\n".join(logs)
    assert "Image generation call ended without an image (status=failed)" in text
    assert "Image output was requested but the reply has no image" in text


def test_failed_response_error_is_logged(monkeypatch):
    """What the owner would have seen with reasoning sent next to the tool."""
    logs = []
    with FakeCodex() as server:
        body = auth._build_responses_body(MESSAGES, "gpt-6.1-sol", image_output=IMAGE_OPTS)
        body["reasoning"] = {"effort": "low"}
        import httpx
        result = auth._stream_with_httpx(httpx, server.base_url + auth.RESPONSES_ENDPOINT, body, {}, 30, 0.0,
                                         logs.append, False)
    assert result["finish_reason"] == "error" and "images" not in result
    assert ("❌ AuthGPT: Response failed (image_generation_user_error): "
            "Complexity 'low' is not supported for this image model.") in logs


def test_non_streaming_json_result_collects_the_image():
    item = {"id": "ig_9", "type": "image_generation_call", "status": "completed", "output_format": "jpeg",
            "result": base64.b64encode(PNG).decode("ascii")}
    result = auth._parse_responses_result({"id": "r", "status": "completed", "output": [item, item]})
    assert [(i["mime"], i["data"]) for i in result["images"]] == [("image/jpeg", PNG)]


# ---------------------------------------------------------------- client side

class Store:
    def get_valid_access_token(self, auto_login=True):
        return "token"


def make_client(monkeypatch, tmp_path, model="authgpt/gpt-6.1-sol"):
    client = api.UnifiedClient("", model, str(tmp_path))
    monkeypatch.setattr(client, "_get_max_retries", lambda: 1)
    monkeypatch.setattr(client, "_is_stop_requested", lambda: False)
    monkeypatch.setattr(client, "_get_active_request_model", lambda: model)
    monkeypatch.setattr(api, "_authgpt_get_store", lambda: Store())
    return client


def test_image_translation_returns_a_generated_image_marker(monkeypatch, tmp_path):
    """Run A in the owner's log: an image edit on authgpt/ came back as a refusal."""
    monkeypatch.setenv("ENABLE_IMAGE_OUTPUT_MODE", "1")
    client = make_client(monkeypatch, tmp_path)
    with FakeCodex() as server:
        monkeypatch.setenv("AUTHGPT_BASE_URL", server.base_url)
        response = client._send_authgpt(MESSAGES, 0.3, 4096, "Chikuwa_NoChibi_normal.png")
    assert server.bodies[0]["model"] == "gpt-6.1-sol"
    assert server.bodies[0]["tools"][0]["type"] == "image_generation"
    assert response.content.startswith("[GENERATED_IMAGE:"), response.content
    path = response.content[len("[GENERATED_IMAGE:"):-1]
    assert os.path.isfile(path) and open(path, "rb").read() == PNG
    assert os.path.basename(path) == "Chikuwa_NoChibi_normal.png"
    assert "images" not in response.raw_response
    tls = client._get_thread_local_client()
    assert tls._last_generated_image_bytes == PNG and tls._last_generated_image_path == path


def test_text_translation_is_unchanged(monkeypatch, tmp_path):
    client = make_client(monkeypatch, tmp_path)
    with FakeCodex() as server:
        monkeypatch.setenv("AUTHGPT_BASE_URL", server.base_url)
        response = client._send_authgpt(MESSAGES, 0.3, 4096, "chapter_1.html")
    assert "tools" not in server.bodies[0]
    assert response.content == REFUSAL


def test_image_tool_options(monkeypatch, tmp_path):
    client = make_client(monkeypatch, tmp_path)
    assert client._authgpt_image_output("gpt-6.1-sol") is None
    monkeypatch.setenv("ENABLE_IMAGE_OUTPUT_MODE", "1")
    assert client._authgpt_image_output("gpt-6.1-sol") == IMAGE_OPTS
    monkeypatch.setenv("IMAGE_OUTPUT_ASPECT_RATIO", "3:4")
    assert client._authgpt_image_output("gpt-6.1-sol")["size"] == "1024x1536"
    monkeypatch.setenv("IMAGE_OUTPUT_ASPECT_RATIO", "16:9")
    assert client._authgpt_image_output("gpt-6.1-sol")["size"] == "1536x1024"
    monkeypatch.setenv("IMAGE_OUTPUT_ASPECT_RATIO", "1:1")
    assert client._authgpt_image_output("gpt-6.1-sol")["size"] == "1024x1024"
    monkeypatch.setenv("OPENAI_IMAGE_SIZE", "1024x1536")
    monkeypatch.setenv("OPENAI_IMAGE_QUALITY", "High")
    assert client._authgpt_image_output("gpt-6.1-sol") == {"size": "1024x1536", "quality": "high",
                                                         "output_format": "png"}


def test_inpainter_forced_image_mode_asks_for_an_image(monkeypatch, tmp_path):
    client = make_client(monkeypatch, tmp_path)
    client._force_image_output_mode = True  # local_inpainter's custom image edit on the current provider
    assert client._authgpt_image_output("gpt-6.1-sol") == IMAGE_OPTS


@pytest.mark.parametrize("context", ["glossary", "summary", "metadata"])
def test_text_steps_of_an_image_run_stay_text(monkeypatch, tmp_path, context):
    """translation_pipeline's image glossary calls send_image(context='glossary') with image
    output mode on; the forced image tool would turn its JSON answer into a picture."""
    monkeypatch.setenv("ENABLE_IMAGE_OUTPUT_MODE", "1")
    # these contexts may route to a dedicated key pool; another test's pool must not leak in (CI order)
    for pool in ("USE_GLOSSARY_KEYS", "USE_ROLLING_SUMMARY_KEYS", "USE_METADATA_KEYS", "USE_VISION_KEYS",
                 "USE_FALLBACK_KEYS", "USE_MAIN_KEY_FALLBACK", "USE_MULTI_API_KEYS", "USE_MULTI_KEYS"):
        monkeypatch.setenv(pool, "0")
    client = make_client(monkeypatch, tmp_path)
    with FakeCodex() as server:
        monkeypatch.setenv("AUTHGPT_BASE_URL", server.base_url)
        content, _finish = client.send_image(MESSAGES, "AAAA", temperature=0.3, max_tokens=4096, context=context)
    assert "tools" not in server.bodies[0]
    assert content == REFUSAL
    with FakeCodex() as server:
        monkeypatch.setenv("AUTHGPT_BASE_URL", server.base_url)
        content, _finish = client.send_image(MESSAGES, "AAAA", temperature=0.3, max_tokens=4096,
                                             context="image_translation")
    assert server.bodies[0]["tools"][0]["type"] == "image_generation"
    assert content.startswith("[GENERATED_IMAGE:")


def test_vision_only_scan_never_asks_for_an_image(monkeypatch, tmp_path):
    """OCR / image-scan requests read the image as text even while image output mode is on."""
    monkeypatch.setenv("ENABLE_IMAGE_OUTPUT_MODE", "1")
    client = make_client(monkeypatch, tmp_path)
    monkeypatch.setattr(client, "_is_vision_input_only_context", lambda *a, **kw: True)
    assert client._authgpt_image_output("gpt-6.1-sol") is None
