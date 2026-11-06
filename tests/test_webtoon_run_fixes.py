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
* B1 the loaded glossary's 마왕 = Mawang never reached the prompt of the pages where OCR had glued the
  word to its neighbours (부록의마왕, 먼저마왕잡으러): the matcher's Hangul word-edge rule assumes spaced
  prose. Manga compression now matches OCR text with GLOSSARY_MATCH_HANGUL_SPACED=0.
* B2 the built-in Manga_KR profile said only "Retain honorifics ... in Romaji": it now keeps Korean
  honorifics and family address terms like the Korean text profiles (saved old defaults migrated).
* B3 the manga glossary prompt said nothing about the attached page images: it now tells the model to
  fix OCR from the image, fill gender from the art, keep titles used as names and the story's own names
  (saved old defaults migrated). And dedupe let an "Unknown" gender outweigh a later "female" copy of
  the same name (extract_glossary_from_epub._dedup_field_count), which emptied most genders.

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
    # A MangaTranslator's print hijack (and its restore, also from __del__) rewrites builtins.print
    # and unified_api_client's own print: both are put back after the test, as test_manga_env does.
    import builtins
    monkeypatch.setattr(builtins, "print", builtins.print)
    monkeypatch.setitem(vars(api), "print", vars(api).get("print"))
    yield
    import gc
    gc.collect()  # a translator's __del__ runs before the monkeypatch teardown restores the prints
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


@pytest.mark.parametrize("text_only, expected", [(True, "chat"), (False, "video")])
def test_manga_client_ignores_the_video_output_mode_too(monkeypatch, tmp_path, text_only, expected):
    """Output mode "video" on the main tab sent a manga request to the video endpoint the same way."""
    monkeypatch.setenv("ENABLE_VIDEO_OUTPUT_MODE", "1")
    monkeypatch.setenv("ENABLE_IMAGE_OUTPUT_MODE", "0")
    client = make_client(monkeypatch, tmp_path, model="nan/some-chat-model")
    client._text_output_only = text_only
    monkeypatch.setattr(client, "_get_active_request_api_key", lambda: FAKE_KEY)
    routes = []
    monkeypatch.setattr(client, "_send_nanogpt_video", lambda *a, **k: routes.append("video"))
    monkeypatch.setattr(client, "_send_nanogpt_image", lambda *a, **k: routes.append("image"))
    monkeypatch.setattr(client, "_send_openai_compatible", lambda *a, **k: routes.append("chat"))
    client._send_nanogpt(MESSAGES, 0.3, 4096, "S01.png")
    assert routes == [expected]
    source = (SRC_DIR / "unified_api_client.py").read_text(encoding="utf-8")
    # every Output mode "video" read goes through the client's own check
    assert source.count('"ENABLE_VIDEO_OUTPUT_MODE", "0") == "1"') == 1
    assert "os.environ.get('ENABLE_VIDEO_OUTPUT_MODE')" not in source


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


def test_editor_state_merge_drops_the_old_per_box_indices():
    """A merge numbers the boxes again: the page's rendered positions, overlays and font sizes keyed
    by the old indices must not land on other boxes (the imported 띠링 is now box 0)."""
    existing = manga_ocr_io.editor_state_from_page({"regions": [_rec("안녕", (40, 60, 150, 100), "Hello")]})
    existing["last_render_positions"] = {"0": [40, 60, 150, 100]}
    existing["overlay_rects"] = [{"x": 40}]
    existing["region_font_sizes"] = {"0": 18}
    page = {"regions": [_rec("띠링", (500, 20, 60, 40))]}
    merged = manga_ocr_io.editor_state_for_import(page, existing, merge=True)
    assert [t["text"] for t in merged["recognized_texts"]] == ["띠링", "안녕"]
    assert [(t["original"]["region_index"], t["translation"]) for t in merged["translated_texts"]] == [(1, "Hello")]
    assert merged["last_render_positions"] == {} and merged["overlay_rects"] == []
    assert merged["region_font_sizes"] == {}
    state = dict(existing)
    state.update(merged)  # how the desktop and the editor session apply it
    assert state["last_render_positions"] == {}
    # nothing renumbered (no existing boxes): the imported page's own state as it is
    assert "last_render_positions" not in manga_ocr_io.editor_state_for_import(page, {}, merge=True)


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


# ---------------------------------------------------------------------------- B1

#: Entries of the run's corrected glossary (token-efficient format), trimmed.
GLOSSARY = (
    "Glossary Columns: raw_name, translated_name, gender, description\n\n"
    "=== CHARACTERS ===\n"
    "* 마왕 = Mawang [Female]: The Mawang (never \"Demon King\").\n"
    "* 할가코스 = Halgakos [Female]: The protagonist translator.\n\n"
    "=== TERMS ===\n"
    "* 짐 = We: The Mawang's royal first person.\n\n"
    "=== TITLES ===\n"
    "* 아가씨 = Agassi: Penchok-jipsa's address for Illya.\n"
)

#: Hayai OCR of the run's lines that came out as "Demon King" (S05, S08, S16, S21 x2, S24).
OCR_LINES = (
    "폐하께서는 '뿔달린번역가는마왕에다름아니므로, 체포되어져야한다'라고말씀되어지셨습니다.",
    "집에가는길은마지막페이지뿐이에용~ 가는길에부록이있는데, 거기마왕님이살아용~",
    "교관은옛날에마왕한테혼자덤볐다가졌습니다. 그러나희는셋이서싸웁니다. 알겠습니다",
    "공식의뢰마왕토별 (S급) 1달려와는 '대리와의의오기(기됐詐)임.",
    "합격입니다! 훌쩍... 직진파티녀석들이먼저마왕잡으러갔습니다! 찾아서데려오십시오!",
    "부록의마왕.",
)


def _compress(text, **kwargs):
    from glossary_compressor import compress_glossary

    return compress_glossary(GLOSSARY, text, glossary_format="auto", glossary_path="webtoon_glossary.csv",
                             chapter_ref={"chapter_num": None, "chapter_file": "S24.png"}, **kwargs)


@pytest.fixture
def match_env(monkeypatch):
    for name in ("GLOSSARY_MATCH_ENGINE", "GLOSSARY_MATCH_HANGUL_SPACED", "GLOSSARY_MATCH_CJK_BOUNDARY",
                 "GLOSSARY_MATCH_SHORT_CJK_LEN", "COMPRESS_GLOSSARY_STRICT_MATCHING_MODE",
                 "COMPRESS_GLOSSARY_PROMPT", "AUTO_GLOSSARY_MODE"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("line", OCR_LINES)
def test_ocr_text_keeps_a_name_glued_to_its_neighbours(match_env, line):
    from glossary_compressor import OCR_SOURCE_MATCH_SETTINGS

    assert "* 마왕 = Mawang" in _compress(line, settings=OCR_SOURCE_MATCH_SETTINGS)


def test_spaced_text_and_short_forms_keep_the_word_edge_rule(match_env):
    from glossary_compressor import OCR_SOURCE_MATCH_SETTINGS

    # a book's spaced prose: unchanged (a glued 마왕 is part of another word there)
    assert "* 마왕 =" not in _compress("부록의마왕.")
    assert "* 마왕 =" in _compress("부록의 마왕.")
    # OCR text: a one-syllable term and a shortened form (아가씨 minus 씨) still need a word edge
    ocr = _compress("이삿짐작업이끝났다. 돌아가는길이다.", settings=OCR_SOURCE_MATCH_SETTINGS)
    assert "* 짐 =" not in ocr and "* 아가씨 =" not in ocr
    assert "* 짐 =" in _compress("혼자라면, 짐이건네주마.", settings=OCR_SOURCE_MATCH_SETTINGS)


def test_manga_translator_matches_the_page_text_as_ocr_text(match_env):
    import manga_translator

    translator = types.SimpleNamespace(
        _manga_glossary_compression_enabled=lambda: True, _get_manga_glossary_path=lambda: "webtoon_glossary.csv",
        _log=lambda *a, **k: None)
    out = manga_translator.MangaTranslator._compress_manga_glossary_for_source(translator, GLOSSARY, "부록의마왕.",
                                                                                "S24.png")
    assert "* 마왕 = Mawang" in out


def test_editor_workflow_matches_the_page_text_as_ocr_text(match_env):
    import manga_editor_core

    tab = types.SimpleNamespace(main_gui=types.SimpleNamespace(config={"compress_glossary_prompt": True}),
                                manga_generated_glossary_path="webtoon_glossary.csv", _log=lambda *a, **k: None)
    out = manga_editor_core._compress_loaded_manga_glossary_for_workflow(tab, GLOSSARY, "거기마왕님이살아용~", "S08.png")
    assert "* 마왕 = Mawang" in out


# ---------------------------------------------------------------------------- B2

def _default_prompts():
    import owner_state

    view = types.SimpleNamespace(default_refinement_system_prompt="")
    owner_state.ConfigStateMixin._init_default_prompt_profiles(view)
    return view.default_prompts


def test_manga_kr_keeps_korean_honorifics_like_the_korean_text_profiles():
    prompts = _default_prompts()
    korean_rules = [line for line in prompts["Korean_html2text"].split("\n")
                    if line.startswith(("- Retain Korean honorifics", "- Retain Korean familial address terms"))]
    assert len(korean_rules) == 2
    manga_kr = prompts["Manga_KR"]
    for line in korean_rules:
        assert line in manga_kr.split("\n")
    assert "Retain honorifics and onomatopoeia in Romaji" not in manga_kr
    assert "\\n" not in manga_kr  # the quotation-mark line ended in a literal backslash-n pair


def test_a_saved_old_manga_kr_default_gets_the_new_lines_and_nothing_else_changes():
    import prompt_defaults

    new = _default_prompts()["Manga_KR"]
    old = new
    for old_lines, new_lines in prompt_defaults.LEGACY_DEFAULT_PROFILE_LINES["Manga_KR"]:
        old = old.replace(new_lines, old_lines)
    assert "- Retain honorifics and onomatopoeia in Romaji.\n" in old and old != new
    edited = old.replace("- Retain honorifics and onomatopoeia in Romaji.\n", "- Retain honorifics.\n")
    config = {"sanitization_korean_quotes_fixed": True,
              "prompt_profiles": {"Manga_KR": old, "My_Manga_KR": old, "TEST": "generate an edit of this image"}}
    assert prompt_defaults.sanitize_prompt_profiles(config) is True
    assert config["prompt_profiles"] == {"Manga_KR": new, "My_Manga_KR": old, "TEST": "generate an edit of this image"}
    assert prompt_defaults.sanitize_prompt_profiles(config) is None  # nothing left to migrate
    dict_profile = {"prompt_profiles": {"Manga_KR": {"prompt": old}}}
    assert prompt_defaults.sanitize_prompt_profiles(dict_profile) is True
    assert dict_profile["prompt_profiles"]["Manga_KR"]["prompt"] == new
    own = {"sanitization_korean_quotes_fixed": True, "prompt_profiles": {"Manga_KR": edited}}
    assert prompt_defaults.sanitize_prompt_profiles(own) is None and own["prompt_profiles"]["Manga_KR"] == edited


# ---------------------------------------------------------------------------- B3

OLD_GLOSSARY_LAST_LINE = "- Keep raw_name exactly in the source script when possible."


def test_manga_glossary_prompt_uses_the_page_images_and_the_story_names():
    import manga_env

    prompt = manga_env.default_manga_glossary_prompt()
    for rule in ("PAGE IMAGE RULES (when the page images are attached)",
                 "write raw_name as it is printed there, not as the OCR misread it",
                 "Fill gender from the art",
                 "A title or epithet that the story uses instead of a name for one specific character",
                 "keep the story's own wording",
                 "Keep honorifics and family address terms romanized",
                 "follow these manga rules"):
        assert rule in prompt, rule
    assert "Halgakos" not in prompt and "마왕" not in prompt  # general, not about the webtoon


def test_a_saved_old_glossary_default_gets_the_new_rules(tmp_path, monkeypatch):
    import manga_env
    from _headless_env import headless_owner

    new = manga_env.default_manga_glossary_prompt()
    old = new[:new.index(OLD_GLOSSARY_LAST_LINE) + len(OLD_GLOSSARY_LAST_LINE)]
    migrate = manga_env.MangaEnvMixin._migrate_legacy_manga_glossary_prompt
    assert migrate(old, new) == new and migrate(new, new) == new
    own = old + "\n- My own rule."
    assert migrate(own, new) == own  # the user's own additions: left alone
    with headless_owner(tmp_path, monkeypatch, {"model": "gpt-4o-mini", "manga_glossary_prompt": old}) as owner:
        state = manga_env.HeadlessMangaState(owner)
        assert state.manga_glossary_prompt == new
        assert owner.config["manga_glossary_prompt"] == new


@pytest.mark.parametrize("custom_fields", ['["description"]', "[]"])
def test_dedupe_keeps_the_gender_a_later_copy_read(monkeypatch, tmp_path, custom_fields):
    """The run's glossary: 일리야 / 인주 / 루네 came back "Unknown" from one page chunk and "female" from the
    next; dedupe kept the first copy, so the glossary had no gender for 17 of its 24 characters."""
    import extract_glossary_from_epub as eg

    monkeypatch.setenv("GLOSSARY_CUSTOM_FIELDS", custom_fields)
    monkeypatch.delenv("GLOSSARY_CUSTOM_ENTRY_TYPES", raising=False)
    entries = [
        {"type": "character", "raw_name": "인주", "translated_name": "Inju", "gender": "Unknown",
         "description": "A guild employee identified as the second receptionist."},
        {"type": "character", "raw_name": "인주", "translated_name": "Injoo", "gender": "female",
         "description": "The bespectacled black-haired guild receptionist."},
        {"type": "character", "raw_name": "인주", "translated_name": "Inju", "gender": "Unknown",
         "description": "A character mentioned later."},
    ]
    out = eg.skip_duplicate_entries([dict(e) for e in entries], output_dir=str(tmp_path),
                                    glossary_path=str(tmp_path / "manga_glossary.json"))
    assert len(out) == 1
    assert eg._normalize_gender_value(out[0]["gender"]) == "female"
    assert out[0]["translated_name"] == "Inju"  # the first kept name stays
