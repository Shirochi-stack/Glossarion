"""Glossarion bugs the headless webtoon run (2026-11-06, 31 Korean strips) hit, fixed in the app:

* A1 authgpt/ (and the ChatGPT web chat) returned "[GENERATED_IMAGE:...]" instead of the JSON that
  had streamed in full when the reply also carried an image: a request that asked for text keeps
  its text (UnifiedClient._chat_route_reply).
* A2 the manga translator followed the main tab's Output mode ("image"): its clients are text-only
  (UnifiedClient._text_output_only, set by MangaTranslator.client and the editor's Translate
  clients), so the authgpt image tool is never offered to a manga translation.
* A3 authgpt/gpt-6.1-sol (accepted by the live endpoint) is in the static authgpt list.
* A4 Import OCR could only replace a page's detection (an SFX-only OCR file dropped all dialogue):
  its merge option keeps the detected / existing regions and adds the imported ones, an imported
  region replacing every region it overlaps by half the smaller box (manga_ocr_io.merge_regions).

Sockets: only loopback connections (the local fake Codex server) are allowed.
"""
import ast
import os
import socket
import sys
import types
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
for path in (str(SRC_DIR), str(TESTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

import authgpt_auth as auth  # noqa: E402
import manga_ocr_io  # noqa: E402
import unified_api_client as api  # noqa: E402
from test_authgpt_image_output import FakeCodex, MESSAGES, PNG, Store  # noqa: E402

FAKE_KEY = "sk-local-test-webtoon-run"
JSON_REPLY = '{"S01-1": "Hello", "S01-2": "Goodbye"}'


@pytest.fixture(autouse=True)
def isolation(monkeypatch, tmp_path):
    real_connect = socket.socket.connect

    def loopback_only(sock, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise OSError(f"network blocked in tests: {address!r}")
        return real_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", loopback_only)
    for name in ("ENABLE_IMAGE_OUTPUT_MODE", "OPENAI_IMAGE_SIZE", "OPENAI_IMAGE_QUALITY", "IMAGE_OUTPUT_ASPECT_RATIO",
                 "TRANSLATION_CANCELLED", "GRACEFUL_STOP", "BATCH_TRANSLATION", "AUTHGPT_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    for pool in ("USE_GLOSSARY_KEYS", "USE_ROLLING_SUMMARY_KEYS", "USE_METADATA_KEYS", "USE_VISION_KEYS",
                 "USE_FALLBACK_KEYS", "USE_MAIN_KEY_FALLBACK", "USE_MULTI_API_KEYS", "USE_MULTI_KEYS",
                 "USE_INPAINTER_KEYS"):
        monkeypatch.setenv(pool, "0")
    auth.reset_cancel()
    monkeypatch.setattr(api, "_save_outgoing_request", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(api, "_save_incoming_response", lambda *a, **kw: None, raising=False)
    yield
    auth.reset_cancel()


def make_client(monkeypatch, tmp_path, model="authgpt/gpt-6.1-sol"):
    client = api.UnifiedClient(FAKE_KEY, model, str(tmp_path))
    monkeypatch.setattr(client, "_get_max_retries", lambda: 1)
    monkeypatch.setattr(client, "_is_stop_requested", lambda: False)
    monkeypatch.setattr(client, "_get_active_request_model", lambda: model)
    monkeypatch.setattr(api, "_authgpt_get_store", lambda: Store())
    return client


class ImageAndTextCodex(FakeCodex):
    """The webtoon run's S01 reply: the image tool ran AND the region JSON streamed in full, whether
    the request offered the tool or not (ChatGPT may run its image tool on its own)."""

    def events(self, body):
        yield {"type": "response.created", "response": {"id": "resp_1", "status": "in_progress"}}
        import base64

        item = {"id": "ig_1", "type": "image_generation_call", "status": "completed", "action": "generate",
                "output_format": "png", "size": "1024x1024", "result": base64.b64encode(PNG).decode("ascii")}
        yield {"type": "response.output_item.added", "output_index": 0,
               "item": {"id": "ig_1", "type": "image_generation_call", "status": "in_progress"}}
        yield {"type": "response.output_item.done", "output_index": 0, "item": item}
        yield {"type": "response.output_text.delta", "delta": JSON_REPLY}
        message = {"id": "msg_1", "type": "message", "role": "assistant",
                   "content": [{"type": "output_text", "text": JSON_REPLY}]}
        yield {"type": "response.output_item.done", "output_index": 1, "item": message}
        yield {"type": "response.completed",
               "response": {"id": "resp_1", "status": "completed", "output": [],
                            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}}


# ---------------------------------------------------------------------------- A1


def test_text_request_keeps_its_text_when_an_image_also_comes_back(monkeypatch, tmp_path):
    client = make_client(monkeypatch, tmp_path)
    with ImageAndTextCodex() as server:
        monkeypatch.setenv("AUTHGPT_BASE_URL", server.base_url)
        response = client._send_authgpt(MESSAGES, 0.3, 4096, "S01.png")
    assert "tools" not in server.bodies[0]
    assert response.content == JSON_REPLY
    assert not list(tmp_path.glob("S01*.png"))  # the unrequested image is not saved as the reply


def test_image_request_still_gets_the_image_marker(monkeypatch, tmp_path):
    monkeypatch.setenv("ENABLE_IMAGE_OUTPUT_MODE", "1")
    client = make_client(monkeypatch, tmp_path)
    with ImageAndTextCodex() as server:
        monkeypatch.setenv("AUTHGPT_BASE_URL", server.base_url)
        response = client._send_authgpt(MESSAGES, 0.3, 4096, "page.png")
    assert server.bodies[0]["tools"][0]["type"] == "image_generation"
    assert response.content.startswith("[GENERATED_IMAGE:")
    assert open(response.content[len("[GENERATED_IMAGE:"):-1], "rb").read() == PNG


@pytest.mark.parametrize("image_output", [False, True])
def test_web_chat_route_follows_the_same_rule(monkeypatch, tmp_path, image_output):
    client = make_client(monkeypatch, tmp_path, model="authchatgpt/auto")

    def send_chat_completion(**_kwargs):
        return {"content": JSON_REPLY, "finish_reason": "stop", "images": [{"data": PNG, "mime": "image/png"}]}

    response = client._send_web_chat(MESSAGES, "authchatgpt", "AuthChatGPT", send_chat_completion, RuntimeError,
                                     extra={"image_output": image_output}, response_name="S01.png")
    if image_output:
        assert response.content.startswith("[GENERATED_IMAGE:")
    else:
        assert response.content == JSON_REPLY


# ---------------------------------------------------------------------------- A2


def test_manga_translator_client_ignores_the_image_output_mode(monkeypatch, tmp_path):
    """The webtoon run's first S01 test: Output mode "image" made the manga request run the image tool."""
    import manga_translator

    monkeypatch.setenv("ENABLE_IMAGE_OUTPUT_MODE", "1")  # the owner's main tab Output mode
    client = make_client(monkeypatch, tmp_path)
    translator = manga_translator.MangaTranslator.__new__(manga_translator.MangaTranslator)
    translator.client = client  # MangaTranslator.__init__ and the runner's client swap both assign it
    assert translator.client is client and client._text_output_only is True
    assert client._image_output_mode_enabled() is False
    with ImageAndTextCodex() as server:
        monkeypatch.setenv("AUTHGPT_BASE_URL", server.base_url)
        response = client._send_authgpt(MESSAGES, 0.3, 4096, "S01.png")
    assert "tools" not in server.bodies[0] and "tool_choice" not in server.bodies[0]
    assert response.content == JSON_REPLY


def test_other_image_output_users_keep_working(monkeypatch, tmp_path):
    monkeypatch.setenv("ENABLE_IMAGE_OUTPUT_MODE", "1")
    plain = make_client(monkeypatch, tmp_path)
    assert plain._image_output_mode_enabled() is True  # the main tab's own image run
    inpainter = make_client(monkeypatch, tmp_path)
    inpainter._text_output_only = True
    inpainter._force_image_output_mode = True  # local_inpainter's custom image edit client
    assert inpainter._image_output_mode_enabled() is True


def test_temporary_retry_clients_inherit_text_only(monkeypatch, tmp_path):
    client = make_client(monkeypatch, tmp_path)
    client._text_output_only = True
    temp = types.SimpleNamespace()
    client._copy_retry_request_context_to_temp_client(temp)
    assert temp._text_output_only is True


@pytest.mark.parametrize("function, variable", [("_translate_individually", "client"),
                                                ("_translate_this_text_background", "unified_client")])
def test_editor_translate_clients_are_text_only(function, variable):
    """The manga editor's Translate (individual regions) and Translate This Text build their own clients."""
    source = (SRC_DIR / "manga_editor_core.py").read_text(encoding="utf-8")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == function)
    lines = ast.get_source_segment(source, node).replace("\r\n", "\n").split("\n")
    created = next(i for i, line in enumerate(lines)
                   if line.strip() == f"{variable} = UnifiedClient(model=model, api_key=api_key)")
    assert lines[created + 1].strip().startswith(f"{variable}._text_output_only = True")


# ---------------------------------------------------------------------------- A3


def test_gpt_6_1_sol_is_in_the_static_authgpt_list():
    import model_options

    static = model_options._get_static_model_options()
    for model in ("authgpt/gpt-6.1-sol", "authgpt/gpt-5.6-sol"):
        assert static.count(model) == 1, model
    authgpt = [m for m in static if m.startswith("authgpt/")]
    assert authgpt.index("authgpt/gpt-6.1-sol") < authgpt.index("authgpt/gpt-6-luna")  # newest first
    assert "authgpt/gpt-6.1-sol" in model_options.get_model_options()


# ---------------------------------------------------------------------------- A4


def _rec(text, box, translated=None):
    return {"text": text, "bounding_box": list(box), "translated_text": translated}


DIALOGUE = [_rec("안녕", (40, 60, 150, 100)), _rec("AF OF OF OF", (335, 590, 127, 207)),
            _rec("잘 가", (200, 900, 160, 120))]
SFX = [_rec("슈우우우욱", (330, 600, 140, 180)), _rec("띠링", (500, 20, 60, 40))]


def test_merge_regions_replaces_only_overlapping_regions():
    merged, replaced = manga_ocr_io.merge_regions(DIALOGUE, SFX)
    assert [r["text"] for r in replaced] == ["AF OF OF OF"]
    # detection order kept: the SFX takes the place of the box it replaced, the new one goes above
    assert [r["text"] for r in merged] == ["띠링", "안녕", "슈우우우욱", "잘 가"]
    assert manga_ocr_io.overlap_of_smaller((0, 0, 10, 10), (5, 0, 10, 10)) == 0.5
    assert manga_ocr_io.merge_regions([], SFX)[0] == SFX


def test_editor_state_merge_keeps_the_page_boxes():
    existing = manga_ocr_io.editor_state_from_page({"regions": DIALOGUE})
    page = {"regions": SFX, "editor_state": {"viewer_rectangles": []}}
    replaced = manga_ocr_io.editor_state_for_import(page, existing)
    merged = manga_ocr_io.editor_state_for_import(page, existing, merge=True)
    assert [t["text"] for t in replaced["recognized_texts"]] == ["슈우우우욱", "띠링"]
    assert [t["text"] for t in merged["recognized_texts"]] == ["띠링", "안녕", "슈우우우욱", "잘 가"]
    assert len(merged["viewer_rectangles"]) == 4
    assert [t["region_index"] for t in merged["recognized_texts"]] == [0, 1, 2, 3]


class _ImportState:
    """The bits of a MangaTranslationTab / HeadlessMangaState the import uses."""

    def __init__(self, files):
        self.selected_files = list(files)
        self._imported_ocr_document = None
        self._imported_ocr_page_map = {}

    def _current_manga_processing_files(self):
        return list(self.selected_files)


def _bind_ocr_session(state):
    import manga_env

    for name in ("_refresh_imported_ocr_page_map", "_resolve_imported_ocr_regions"):
        setattr(state, name, types.MethodType(getattr(manga_env.MangaOcrSessionMixin, name), state))
    return state


def _document(page_path, regions):
    page = manga_ocr_io.make_page(str(page_path), regions, index=1)
    return manga_ocr_io.create_document([page], workflow="manual-editor")


def _write_page(path, size=(600, 1100)):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (255, 255, 255)).save(path)
    return path


def test_import_ocr_session_carries_the_merge_option(tmp_path):
    import manga_env

    page = _write_page(tmp_path / "S01.png")
    for merge in (False, True):
        state = _bind_ocr_session(_ImportState([str(page)]))
        matches = manga_env.import_ocr_session(state, document=_document(page, SFX), files=[str(page)], merge=merge)
        assert len(matches) == 1 and state._imported_ocr_merge is merge
        regions = state._resolve_imported_ocr_regions(str(page))
        assert [r.text for r in regions] == ["슈우우우욱", "띠링"]
        assert all(r._imported_ocr_session and r._imported_ocr_merge is merge for r in regions)
    # merge with an empty imported page: nothing to merge, the page is detected as usual
    state = _bind_ocr_session(_ImportState([str(page)]))
    manga_env.import_ocr_session(state, document=_document(page, []), files=[str(page)], merge=True)
    assert state._resolve_imported_ocr_regions(str(page)) is None


def _ocr_pass(tmp_path, monkeypatch, merge, *, webtoon=False):
    """``MangaTranslator.process_image`` OCR-only pass (the glossary workflow's) on a page with an import."""
    import manga_env
    import manga_translator
    from _headless_env import headless_owner

    page = _write_page(tmp_path / "pages" / "S01.png", size=(600, 2400) if webtoon else (600, 1100))
    state = _bind_ocr_session(_ImportState([str(page)]))
    manga_env.import_ocr_session(state, document=_document(page, SFX), files=[str(page)], merge=merge)
    calls = []

    def detected():
        return [manga_ocr_io.region_record_to_text_region(r, manga_translator.TextRegion) for r in DIALOGUE]

    with headless_owner(tmp_path / "owner", monkeypatch, {"model": "gpt-4o-mini", "api_key": FAKE_KEY}) as owner:
        translator = manga_translator.MangaTranslator({"provider": "custom-api"}, None, owner,
                                                      log_callback=lambda *_a, **_k: None,
                                                      skip_inpainter_init=True, skip_ocr_init=True)
        try:
            translator.ocr_import_resolver = state._resolve_imported_ocr_regions
            translator.detect_text_regions = lambda path: calls.append(("detect", path)) or detected()
            translator._detect_webtoon_chunk_regions = lambda path: calls.append(("chunks", path)) or (None, detected(), 3)
            translator._process_webtoon_chunks = lambda *a, **k: pytest.fail("the webtoon translate path skips the merge")
            if webtoon:
                translator.manga_settings.setdefault("advanced", {}).update(format_detection=True, webtoon_mode="auto")
            result = translator.process_image(str(page), str(tmp_path / "out" / "S01.png"), ocr_only=True)
        finally:
            translator.restore_print()
    return result, calls


def test_ocr_pass_merges_imported_sfx_with_the_detected_dialogue(tmp_path, monkeypatch):
    result, calls = _ocr_pass(tmp_path, monkeypatch, merge=True)
    assert calls and calls[0][0] == "detect"
    texts = [r.text for r in result["_region_objects"]]
    assert texts == ["띠링", "안녕", "슈우우우욱", "잘 가"]  # dialogue kept, the misread SFX box replaced
    imported = [r.text for r in result["_region_objects"] if getattr(r, "_imported_ocr_session", False)]
    assert imported == ["띠링", "슈우우우욱"]
    assert [r.rect_index for r in result["_region_objects"]] == [0, 1, 2, 3]


def test_ocr_pass_replace_stays_the_default(tmp_path, monkeypatch):
    result, calls = _ocr_pass(tmp_path, monkeypatch, merge=False)
    assert calls == []  # the import replaces detection
    assert [r.text for r in result["_region_objects"]] == ["슈우우우욱", "띠링"]


def test_webtoon_merge_detects_in_chunks(tmp_path, monkeypatch):
    result, calls = _ocr_pass(tmp_path, monkeypatch, merge=True, webtoon=True)
    assert [c[0] for c in calls] == ["chunks"]
    assert [r.text for r in result["_region_objects"]] == ["띠링", "안녕", "슈우우우욱", "잘 가"]


def test_desktop_import_merges_into_the_page_state(monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import manga_integration

    boxes = []
    monkeypatch.setattr(manga_integration, "QMessageBox", types.SimpleNamespace(
        warning=lambda *a, **k: boxes.append(a[1]), information=lambda *a, **k: boxes.append(a[1])))
    page_path = os.path.abspath("S01.png")

    class StateManager:
        def __init__(self):
            self.states = {page_path: manga_ocr_io.editor_state_from_page({"regions": DIALOGUE})}

        def get_state(self, path):
            return self.states.get(path)

        def set_state(self, path, state, save=True):
            self.states[path] = state

        def flush(self):
            pass

    logs = []
    for merge, expected in ((True, ["띠링", "안녕", "슈우우우욱", "잘 가"]), (False, ["슈우우우욱", "띠링"])):
        tab = types.SimpleNamespace(image_state_manager=StateManager(), image_preview_widget=None,
                                    _imported_ocr_merge=merge, _log=lambda *a, **k: logs.append(a), dialog=None)
        page = manga_ocr_io.make_page(page_path, SFX, index=1)
        document = manga_ocr_io.create_document([page], workflow="manual-editor")
        ok = manga_integration.MangaTranslationTab._apply_imported_batch_ocr_document(
            tab, "sfx.json", document, [page_path], matches_override={page_path: page})
        assert ok
        state = tab.image_state_manager.states[page_path]
        assert [t["text"] for t in state["recognized_texts"]] == expected, merge
