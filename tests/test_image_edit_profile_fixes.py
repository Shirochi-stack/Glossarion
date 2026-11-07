"""Fixes after the owner's first run with the image edit prompt profiles (PR 170), 2026-11-07.

* L: under a translate prompt (Translate SFX + background text) the model translated an adventurer card
  that sat in the text mask; the leftover-text check counted its ink (91 % of the Korean strokes' places
  were inked by the English) and put the source art back. The check now asks whether the source's own
  letters are still drawn there (``image_edit_check.lettering_similarity``).
* F: the local inpainter (anime_onnx LaMa) then wiped the model's redraw. The image edit path no longer
  has a local inpainter fallback, under any prompt (owner: "lama should have not been triggered at all
  when the custom-image-edit model was selected").
* W: the webtoon run (``_process_webtoon_chunks``) saved no cleaned page and reported none, so the
  editor's re-render of an edited box ("Rendering edited translation...") used the older editor Clean's
  page and the run's image edit redraw was gone from the output.
* E: that re-render built its regions without the boxes' detector class, so the renderer guessed it from
  the box shape (8 "free text" of 16) and lettered regions the run leaves to the model's redraw.
* P: the profile combo is wide enough for "Translate SFX + background text".

Synthetic pages, isolated configs, sockets blocked, fake keys only.
"""
import base64
import os
import socket
import sys
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

import manga_env  # noqa: E402

SFX_BG = "Translate SFX + background text"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise OSError("network access is blocked in this test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    for name in ("OUTPUT_LANGUAGE", "GLOSSARY_TARGET_LANGUAGE", "TARGET_LANGUAGE", "TARGET_LANG",
                 "QWEN_IMAGE_EDIT_INPAINT_PROMPT", "USE_INPAINTER_KEYS", "GRACEFUL_STOP", "TRANSLATION_CANCELLED"):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# L / F: the leftover-text check under a translate prompt, no local inpainter
# ---------------------------------------------------------------------------

CARD_MASK = (slice(50, 150), slice(45, 215))


def _card(lines, thickness=2):
    """A card held in the art (a light rectangle on a darker background) with lines of lettering."""
    img = np.full((200, 260, 3), (200, 170, 140), np.uint8)
    cv2.rectangle(img, (40, 40), (220, 160), (225, 235, 240), -1)
    for i, text in enumerate(lines):
        cv2.putText(img, text, (52, 70 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), thickness)
    return img


SOURCE_CARD = ["MOHEOMGA KADEU", "IREUM HALGA", "JIGEOP BEONYEOK", "DEUNGGEUP F"]
TRANSLATED_CARD = ["ADVENTURER CARD", "NAME HULGAROS", "CLASS TRANSLATR", "RANK F"]


def _card_mask():
    mask = np.zeros((200, 260), np.uint8)
    mask[CARD_MASK] = 255
    return mask


def test_translated_card_is_not_text_left_but_untouched_lettering_still_is():
    from image_edit_check import unremoved_text_mask

    source, translated, mask = _card(SOURCE_CARD), _card(TRANSLATED_CARD), _card_mask()
    _failed, stats = unremoved_text_mask(source, translated, mask)
    assert stats[0]["kept"] >= 0.5 and stats[0]["failed"], "ink alone: the translation looks left in place"

    _failed, stats = unremoved_text_mask(source, translated, mask, same_lettering=True)
    assert not stats[0]["failed"] and stats[0]["similarity"] < 0.75
    for left in (source.copy(), cv2.GaussianBlur(source, (0, 0), 1.2)):  # left as drawn (also softened)
        failed, stats = unremoved_text_mask(source, left, mask, same_lettering=True)
        assert stats[0]["failed"] and stats[0]["similarity"] > 0.9 and failed[100, 100]
    _failed, stats = unremoved_text_mask(source, _card([]), mask, same_lettering=True)
    assert not stats[0]["failed"]  # emptied


def _inpainter(**config):
    from local_inpainter import LocalInpainter

    inpainter = object.__new__(LocalInpainter)
    inpainter.config = dict(config)
    inpainter.current_method = "custom-image-edit"
    inpainter.model_loaded = True
    inpainter._mp_enabled = False
    inpainter._check_stop = lambda: False
    inpainter.logs = []
    inpainter._log = lambda message, level="info": inpainter.logs.append((level, message))
    inpainter._sync_inpainter_key_pool_from_config = lambda: None
    inpainter._custom_image_edit_use_current_provider = False
    inpainter._custom_image_edit_endpoint = "http://edit.test/v1"
    return inpainter


def _fake_endpoint(monkeypatch, output):
    import requests

    ok, png = cv2.imencode(".png", output)
    assert ok

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"data": [{"b64_json": base64.b64encode(png.tobytes()).decode("ascii")}]}

    monkeypatch.setattr(requests, "post", lambda url, **kwargs: Response())
    monkeypatch.setenv("CUSTOM_IMAGE_EDIT_MODEL", "fake-edit-model")
    monkeypatch.setenv("CUSTOM_IMAGE_EDIT_API_KEY", "sk-fake")
    monkeypatch.setenv("CUSTOM_IMAGE_EDIT_FULL_PAGE_OUTPUT", "100")


def _profile_config(name):
    owner = types.SimpleNamespace(config={})
    manga_env.ImageEditPromptProfiles(owner).select(name)
    return owner.config


def test_translate_prompt_keeps_the_card_the_model_translated(monkeypatch):
    """The owner's y=5451 card: the model replaced the Korean with English; the edit stays and no
    region is reported as left (nor handed to a local inpainter)."""
    source, translated, mask = _card(SOURCE_CARD), _card(TRANSLATED_CARD), _card_mask()
    _fake_endpoint(monkeypatch, translated)
    inpainter = _inpainter(**_profile_config(SFX_BG))
    inpainter._image_edit_local_fallback = lambda *a: pytest.fail("no local inpainter with the image edit")
    inpainter._register_custom_image_edit_output = lambda crop, edited, crop_mask: edited

    result = inpainter._custom_image_edit_inpaint_chunked(source, mask)

    assert result is not None
    assert inpainter.last_image_edit_unremoved_mask is None
    assert np.abs(result[CARD_MASK].astype(int) - translated[CARD_MASK].astype(int)).mean() < 1
    assert not any("left the text in place" in message for _level, message in inpainter.logs)


@pytest.mark.parametrize("profile", ["Default", SFX_BG])
def test_text_the_model_left_keeps_its_art_without_a_local_inpainter(monkeypatch, profile):
    """A card the model returned untouched is reported under every profile; its art stays as drawn (not
    lettered) and no local inpainter runs, even when one is handed to the inpainter."""
    source, mask = _card(SOURCE_CARD), _card_mask()
    _fake_endpoint(monkeypatch, cv2.GaussianBlur(source, (0, 0), 0.8))
    inpainter = _inpainter(**_profile_config(profile))
    calls = []
    inpainter._image_edit_local_fallback = lambda image, redo: calls.append(1) or image
    inpainter._register_custom_image_edit_output = lambda crop, edited, crop_mask: edited

    result = inpainter._custom_image_edit_inpaint_chunked(source, mask)

    assert result is not None and not calls
    left = inpainter.last_image_edit_unremoved_mask
    assert left is not None and left[100, 100]
    assert np.array_equal(result[90:110, 90:110], source[90:110, 90:110])
    assert any("no local inpainter fallback" in message for _level, message in inpainter.logs)


# ---------------------------------------------------------------------------
# W: the webtoon run saves and reports its cleaned page
# ---------------------------------------------------------------------------

@pytest.fixture
def translator():
    import builtins

    import manga_translator
    import unified_api_client
    from manga_settings_defaults import MANGA_TOP_LEVEL_DEFAULTS, default_manga_settings

    original_print = builtins.print
    original_uc_print = vars(unified_api_client).get("print")
    config = dict(MANGA_TOP_LEVEL_DEFAULTS, manga_settings=default_manga_settings())
    instance = manga_translator.MangaTranslator(
        {"provider": "custom-api"}, None, types.SimpleNamespace(config=config),
        log_callback=None, skip_inpainter_init=True, skip_ocr_init=True)
    try:
        yield instance
    finally:
        instance.restore_print = lambda: None
        builtins.print = original_print
        if original_uc_print is not None:
            vars(unified_api_client)["print"] = original_uc_print


def test_webtoon_run_saves_its_cleaned_page_for_the_editor(translator, tmp_path, monkeypatch):
    from manga_translator import TextRegion

    page = np.full((300, 120, 3), 200, np.uint8)
    cv2.putText(page, "AB", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cleaned = page.copy()
    cleaned[120:170, 10:110] = 255  # the image edit's page
    region = TextRegion(text="AB", vertices=[(10, 120), (110, 120), (110, 170), (10, 170)],
                        bounding_box=(10, 120, 100, 50), confidence=1.0, region_type="text_block")
    region.translated_text = "Hi"
    output = tmp_path / "page_translated" / "page.png"
    output.parent.mkdir()
    stale = tmp_path / "page_translated" / "page_cleaned.png"
    stale.write_bytes(b"an older editor Clean")
    monkeypatch.setattr(translator, "_detect_webtoon_chunk_regions", lambda path: (page.copy(), [region], 2))
    monkeypatch.setattr(translator, "translate_regions", lambda regions, path: regions)
    monkeypatch.setattr(translator, "full_page_context_enabled", False, raising=False)
    monkeypatch.setattr(translator, "skip_inpainting", False, raising=False)
    monkeypatch.setattr(translator, "inpaint_regions", lambda image, mask: cleaned.copy())
    monkeypatch.setattr(translator, "render_translated_text", lambda image, regions: image)

    result = translator._process_webtoon_chunks(
        str(tmp_path / "page.png"), str(output), {"errors": [], "format_info": {}})

    assert result["success"] and result["output_path"] == str(output)
    assert result.get("cleaned_image_path") == str(stale), "the run reports its cleaned page"
    assert np.array_equal(cv2.imread(str(stale)), cleaned), "this run's cleaned page replaced the older one"


# ---------------------------------------------------------------------------
# E: the editor re-render classifies the boxes like the run
# ---------------------------------------------------------------------------

class _States:
    def __init__(self, states):
        self._states = states

    def get_state(self, path):
        return self._states.get(path)

    def set_state(self, path, state, save=True):
        self._states[path] = state

    def update_state(self, path, values, save=True):
        self._states.setdefault(path, {}).update(values)


class _RecordingTranslator:
    def __init__(self):
        self.seen = []

    def render_translated_text(self, image, regions):
        self.seen = [(r.translated_text, getattr(r, "bubble_type", None)) for r in regions]
        return image


def test_editor_rerender_regions_carry_the_boxes_detector_class(tmp_path):
    import manga_editor_core
    from manga_translator import TextRegion

    page = tmp_path / "page.png"
    cv2.imwrite(str(page), np.full((200, 200, 3), 255, np.uint8))
    boxes = [(10, 10, 80, 40), (10, 100, 120, 40)]
    state = {
        "translated_texts": [
            {"original": {"text": "말풍선", "region_index": 0}, "translation": "Hello", "bbox": list(boxes[0])},
            {"original": {"text": "쾅", "region_index": 1}, "translation": "BOOM", "bbox": list(boxes[1])},
        ],
        "viewer_rectangles": [
            {"x": 10, "y": 10, "width": 80, "height": 40, "bubble_type": "text_bubble"},
            {"x": 10, "y": 100, "width": 120, "height": 40, "bubble_type": "free_text"},
        ],
    }
    recorder = _RecordingTranslator()
    host = types.SimpleNamespace(
        image_state_manager=_States({str(page): state}), translator=recorder, logs=[],
        image_preview_widget=types.SimpleNamespace(current_image_path=None, viewer=types.SimpleNamespace(rectangles=[])),
        main_gui=types.SimpleNamespace(config={}))
    host._log = lambda message, level="info": host.logs.append(message)
    regions = []
    for (x, y, w, h), text in zip(boxes, ("Hello", "BOOM")):
        region = TextRegion(text="src", vertices=[(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                            bounding_box=(x, y, w, h), confidence=1.0, region_type="text_block")
        region.translated_text = text
        regions.append(region)

    out = manga_editor_core._render_with_manga_translator(
        host, str(page), regions, output_path=str(tmp_path / "page_translated" / "page.png"),
        original_image_path=str(page), switch_tab=False, refresh_preview=False, use_viewer_exclusions=False)

    assert out, host.logs
    assert recorder.seen == [("Hello", "text_bubble"), ("BOOM", "free_text")]


def test_translate_prompt_rerender_leaves_the_free_text_unlettered(translator, monkeypatch):
    """With the box class known, the renderer's own rule (MangaTranslator._mark_regions_left_to_image_edit)
    leaves the free text to the model's redraw under a translate prompt, as the batch run does."""
    from manga_translator import TextRegion

    monkeypatch.setattr(translator, "_image_edit_translates_text", lambda: True)
    bubble = TextRegion(text="a", vertices=[], bounding_box=(0, 0, 10, 10), confidence=1.0, region_type="text_block")
    sfx = TextRegion(text="b", vertices=[], bounding_box=(0, 20, 10, 10), confidence=1.0, region_type="text_block")
    bubble.bubble_type, sfx.bubble_type = "text_bubble", "free_text"
    translator._mark_regions_left_to_image_edit([bubble, sfx])
    assert translator._region_left_unlettered(sfx) and not getattr(bubble, "image_edit_redraws", False)


# ---------------------------------------------------------------------------
# P: the profile combo shows the longest name
# ---------------------------------------------------------------------------

def test_profile_combo_is_wide_enough_for_the_longest_name():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    import review_dialog

    app = QApplication.instance() or QApplication([])
    owner = types.SimpleNamespace(config={})
    profiles = manga_env.ImageEditPromptProfiles(owner)
    profiles.select(SFX_BG)
    row, combo, buttons = review_dialog.build_prompt_profile_row(
        "Profile:", "image_edit", "tip", {"new": lambda: None, "save": lambda: None, "delete": lambda: None},
        lambda name: None)
    try:
        review_dialog.refresh_prompt_profile_row(combo, buttons, profiles, "Reset '{name}'", "Delete")
        text = combo.fontMetrics().horizontalAdvance(SFX_BG)
        # the text, the icon and the 18 px drop-down arrow fit in the minimum width
        # (capped at the row's maximum; Qt offscreen without fonts measures a wide fallback font)
        needed = min(review_dialog.PROMPT_PROFILE_COMBO_MAX_WIDTH, text + combo.iconSize().width() + 18 + 20)
        assert combo.minimumWidth() >= needed > review_dialog.PROMPT_PROFILE_COMBO_MIN_WIDTH
        # short names keep the old minimum (the review dialog's rows look as before)
        short = types.SimpleNamespace(names=["Default"], selected_name=lambda: "Default", is_builtin=lambda n: True)
        review_dialog.refresh_prompt_profile_row(combo, buttons, short, "Reset '{name}'", "Delete")
        assert combo.minimumWidth() == review_dialog.PROMPT_PROFILE_COMBO_MIN_WIDTH
    finally:
        row.deleteLater()
        app.processEvents()


# ---------------------------------------------------------------------------
# M: moving a text box re-renders on the newest cleaned page
# ---------------------------------------------------------------------------

def test_moving_a_box_rerenders_on_the_newest_cleaned_page(tmp_path):
    """Owner, 2026-11-07: after a run, moving a text box re-rendered the page on an older cleaned page (the
    earlier Clean's, still named by the page state), and the run's cleaning / image edit redraw was gone.
    The editor's base is the newest cleaned page of the page; the state is repaired to it."""
    import time as _time

    import manga_editor_core

    page = tmp_path / "page.png"
    cv2.imwrite(str(page), np.full((60, 60, 3), 200, np.uint8))
    older = tmp_path / "old_clean" / "page_cleaned.png"  # an earlier Clean saved elsewhere
    older.parent.mkdir()
    cv2.imwrite(str(older), np.full((60, 60, 3), 10, np.uint8))
    newer = tmp_path / "page_translated" / "page_cleaned.png"  # the last run's cleaned page
    newer.parent.mkdir()
    cv2.imwrite(str(newer), np.full((60, 60, 3), 250, np.uint8))
    os.utime(older, (_time.time() - 600, _time.time() - 600))
    states = _States({str(page): {"cleaned_image_path": str(older)}})
    host = types.SimpleNamespace(image_state_manager=states, _cleaned_image_path=str(older),
                                 main_gui=types.SimpleNamespace(config={}))

    assert manga_editor_core._resolve_cleaned_image_for_render(host, str(page)) == str(newer)
    assert states.get_state(str(page))["cleaned_image_path"] == str(newer)
    assert host._cleaned_image_path == str(newer)

    # a later Clean that names its page in the state wins again (the newest, not "the run's folder")
    os.utime(newer, (_time.time() - 300, _time.time() - 300))
    os.utime(older, None)
    states.update_state(str(page), {"cleaned_image_path": str(older)})
    assert manga_editor_core._resolve_cleaned_image_for_render(host, str(page)) == str(older)
