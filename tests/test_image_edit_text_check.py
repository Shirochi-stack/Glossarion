"""Custom image edit output checks (webtoon QA run, 2026-11-06, report items 2, 4 and 10).

* C1: the image-edit model sometimes returns a crop with the text still in it, and English was
  lettered over the Korean. After the edit, each text-mask component is checked against the
  source (share of the source ink strokes still drawn); a failed one gets the source art back
  and is cleaned once by the configured local inpainter, or is left un-lettered when there is
  no local inpainter.
* C2: the model returns the crop at its own size (656x1640 for 800x2000) with the content
  moved or rescaled; the resized output was pasted without registration and broke bubble
  outlines. The output is now aligned to the source (ECC outside the text mask) first.
* C3: number-only signs (11:59, 12:00) whose translation is the same digits are neither
  cleaned nor lettered.

Synthetic pages only, sockets blocked, fake keys only.
"""
import base64
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise OSError("network access is blocked in this test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


# ---------------------------------------------------------------------------
# synthetic pages
# ---------------------------------------------------------------------------

BUBBLES = ((200, 200), (200, 700))  # centres; ellipse axes 120x52, text box +-100 x +-32


def _page(h=1000, w=400, text=True, seed=1):
    """Hatched art, white oval bubbles with a black outline, dark lettering inside."""
    img = np.full((h, w, 3), 235, np.uint8)
    rng = np.random.default_rng(seed)
    for _ in range(60):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(img, p1, p2, (90, 60, 40), 2)
    mask = np.zeros((h, w), np.uint8)
    for cx, cy in BUBBLES:
        cv2.ellipse(img, (cx, cy), (120, 52), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (cx, cy), (120, 52), 0, 0, 360, (0, 0, 0), 3)
        if text:
            cv2.putText(img, "HELLO", (cx - 80, cy + 14), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 4)
        cv2.rectangle(mask, (cx - 100, cy - 32), (cx + 100, cy + 32), 255, -1)
    return img, mask


def _bubble_mask(shape, index):
    cx, cy = BUBBLES[index]
    mask = np.zeros(shape[:2], np.uint8)
    cv2.rectangle(mask, (cx - 100, cy - 32), (cx + 100, cy + 32), 255, -1)
    return mask


def _dark_text_pixels(image, index):
    cx, cy = BUBBLES[index]
    inner = image[cy - 20:cy + 20, cx - 75:cx + 75]
    return int(np.count_nonzero(inner.max(axis=2) < 100))


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
    return inpainter


# ---------------------------------------------------------------------------
# C1: text left in place by the image edit
# ---------------------------------------------------------------------------

def _edit_that_misses_bubble_two(crop, crop_mask, iterations=None, _pool_prepared=False):
    """Cleans bubble 1 and returns bubble 2 untouched (what Nano Banana did 18 times)."""
    edited = crop.copy()
    edited[_bubble_mask(crop.shape, 0) > 0] = 255
    return edited


def test_text_left_by_the_edit_is_cleaned_once_by_the_local_inpainter():
    source, mask = _page()
    inpainter = _inpainter()
    inpainter._custom_image_edit_inpaint = _edit_that_misses_bubble_two
    calls = []

    def local_inpaint(image, redo_mask):
        calls.append(redo_mask.copy())
        cleaned = image.copy()
        cleaned[redo_mask > 0] = 255
        return cleaned

    inpainter._image_edit_local_fallback = local_inpaint
    result = inpainter.inpaint(source, mask)

    assert _dark_text_pixels(source, 1) > 200
    assert _dark_text_pixels(result, 0) == 0
    assert _dark_text_pixels(result, 1) == 0, "the Korean the edit left must not stay under the lettering"
    assert len(calls) == 1, "one fallback request for the page, not a retry loop"
    redo = calls[0] > 0
    assert redo[_bubble_mask(source.shape, 1) > 0].all()
    assert not redo[_bubble_mask(source.shape, 0) > 0].any()
    assert inpainter.last_image_edit_unremoved_mask is None
    assert any("left the text in place" in message for _level, message in inpainter.logs)


def test_text_left_without_a_local_inpainter_keeps_the_art_and_is_reported():
    source, mask = _page()
    inpainter = _inpainter()
    inpainter._custom_image_edit_inpaint = _edit_that_misses_bubble_two

    result = inpainter.inpaint(source, mask)

    bubble_two = _bubble_mask(source.shape, 1) > 0
    assert np.array_equal(result[bubble_two], source[bubble_two])
    unremoved = inpainter.last_image_edit_unremoved_mask
    assert unremoved is not None
    assert (unremoved[bubble_two] > 0).all()
    assert not (unremoved[_bubble_mask(source.shape, 0) > 0] > 0).any()
    assert _dark_text_pixels(result, 0) == 0  # the bubble the edit did clean stays clean


def test_lossy_whole_crop_output_undoes_only_the_missed_bubble():
    # A model that returns the whole crop re-encoded (every pixel off by a level or two, the
    # Full-page output / edit-area 100% case) changes one connected area: the missed bubble's
    # art must come back without undoing the bubble the edit did clean.
    source, mask = _page()

    def lossy_edit(crop, crop_mask, iterations=None, _pool_prepared=False):
        edited = _edit_that_misses_bubble_two(crop, crop_mask)
        return np.clip(edited.astype(np.int16) - 3, 0, 255).astype(np.uint8)

    inpainter = _inpainter()
    inpainter._custom_image_edit_inpaint = lossy_edit
    result = inpainter.inpaint(source, mask)

    assert _dark_text_pixels(result, 0) == 0, "the cleaned bubble stays clean"
    unremoved = inpainter.last_image_edit_unremoved_mask
    assert unremoved is not None
    assert (unremoved[_bubble_mask(source.shape, 1) > 0] > 0).all()
    assert not (unremoved[_bubble_mask(source.shape, 0) > 0] > 0).any(), \
        "the bubble the edit cleaned is not reported as left (it is lettered)"


def test_clean_edit_passes_the_check_untouched():
    source, mask = _page()
    clean, _ = _page(text=False)
    inpainter = _inpainter()
    inpainter._custom_image_edit_inpaint = lambda crop, crop_mask, iterations=None, _pool_prepared=False: clean.copy()
    inpainter._image_edit_local_fallback = lambda *_a: pytest.fail("no fallback for a clean edit")

    result = inpainter.inpaint(source, mask)

    assert np.array_equal(result, clean)
    assert inpainter.last_image_edit_unremoved_mask is None


def test_unremoved_text_measure_separates_kept_and_removed_text():
    from image_edit_check import unremoved_text_mask

    source, mask = _page()
    clean, _ = _page(text=False)
    failed, stats = unremoved_text_mask(source, clean, mask)
    assert not failed.any() and stats and all(entry["kept"] < 0.2 for entry in stats)
    failed, stats = unremoved_text_mask(source, source.copy(), mask)
    assert (failed[mask > 0] > 0).all() and all(entry["kept"] > 0.9 for entry in stats)


# ---------------------------------------------------------------------------
# C2: returned crop registered to the source before pasting
# ---------------------------------------------------------------------------

def _model_output(clean, shape):
    """The textless page, shifted and rescaled, at the model's own 0.82x size (656x1640 style)."""
    h, w = shape[:2]
    warp = np.float32([[0.98, 0, 3], [0, 0.985, 12]])
    moved = cv2.warpAffine(clean, warp, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return cv2.resize(moved, (int(w * 0.82), int(h * 0.82)), interpolation=cv2.INTER_AREA)


def test_returned_crop_is_aligned_to_the_source_before_pasting(monkeypatch):
    import requests

    source, mask = _page()
    clean, _ = _page(text=False)
    ok, png = cv2.imencode(".png", _model_output(clean, source.shape))
    assert ok
    posted = []

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"data": [{"b64_json": base64.b64encode(png.tobytes()).decode("ascii")}]}

    def fake_post(url, **_kwargs):
        posted.append(url)
        return Response()

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setenv("CUSTOM_IMAGE_EDIT_MODEL", "fake-edit-model")
    monkeypatch.setenv("CUSTOM_IMAGE_EDIT_API_KEY", "sk-fake")
    monkeypatch.setenv("CUSTOM_IMAGE_EDIT_FULL_PAGE_OUTPUT", "10")
    monkeypatch.delenv("USE_INPAINTER_KEYS", raising=False)
    inpainter = _inpainter()
    inpainter._custom_image_edit_use_current_provider = False
    inpainter._custom_image_edit_endpoint = "http://edit.test/v1"

    result = inpainter._custom_image_edit_inpaint(source, mask, _pool_prepared=True)

    assert posted == ["http://edit.test/v1/images/edits"]
    assert result is not None and result.shape == source.shape
    pasted = cv2.dilate(mask, np.ones((61, 61), np.uint8)) > 0
    error = np.abs(result.astype(int) - clean.astype(int)).max(axis=2)
    # The bubble outlines inside the pasted area must sit where the source draws them.
    outline = np.zeros(source.shape[:2], np.uint8)
    for cx, cy in BUBBLES:
        cv2.ellipse(outline, (cx, cy), (120, 52), 0, 0, 360, 255, 3)
    on_outline = (outline > 0) & pasted
    assert float(np.mean(result[on_outline].max(axis=1) < 120)) > 0.8, "bubble outline broken by the paste"
    assert float(np.mean(error[pasted])) < 8.0
    assert any("aligned to the source" in message for _level, message in inpainter.logs)


def test_registration_recovers_shift_and_scale():
    from image_edit_check import register_to_source

    source, mask = _page()
    clean, _ = _page(text=False)
    returned = cv2.resize(_model_output(clean, source.shape), (source.shape[1], source.shape[0]),
                          interpolation=cv2.INTER_LANCZOS4)
    aligned, info = register_to_source(source, returned, mask)
    assert info["applied"]
    assert info["dx"] == pytest.approx(3, abs=1) and info["dy"] == pytest.approx(12, abs=1)
    assert info["scale_y"] == pytest.approx(0.985, abs=0.01)
    assert info["error_after"] < info["error_before"] / 2
    # An output that already matches is left alone.
    same, info = register_to_source(source, clean.copy(), mask)
    assert not info["applied"] and np.array_equal(same, clean)


# ---------------------------------------------------------------------------
# translator side: C1 un-lettered regions, C3 number-only regions
# ---------------------------------------------------------------------------

@pytest.fixture
def mt():
    import builtins

    import manga_translator
    import unified_api_client
    from manga_settings_defaults import MANGA_TOP_LEVEL_DEFAULTS, default_manga_settings

    original_print = builtins.print
    original_uc_print = vars(unified_api_client).get("print")
    config = dict(MANGA_TOP_LEVEL_DEFAULTS, manga_settings=default_manga_settings())
    translator = manga_translator.MangaTranslator(
        {"provider": "custom-api"}, None, SimpleNamespace(config=config),
        log_callback=None, skip_inpainter_init=True, skip_ocr_init=True)
    try:
        if not (translator.selected_font_style or translator.font_path):
            pytest.skip("no TrueType font on this machine")
        yield translator
    finally:
        translator.restore_print = lambda: None
        builtins.print = original_print
        if original_uc_print is not None:
            vars(unified_api_client)["print"] = original_uc_print


def _region(box, text, translated):
    from manga_translator import TextRegion

    x, y, w, h = box
    region = TextRegion(text=text, vertices=[(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                        bounding_box=box, confidence=1.0, region_type="text_bubble")
    region.translated_text = translated
    region.bubble_bounds = box
    region.bubble_type = "text_bubble"
    region.should_inpaint = True
    return region


CLOCK_BOX = (40, 40, 160, 60)
SPEECH_BOX = (40, 200, 260, 90)


def _sign_page():
    page = np.full((340, 360, 3), 255, np.uint8)
    cv2.putText(page, "11:59", (50, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)
    return page


def _box(image, box):
    x, y, w, h = box
    return image[y:y + h, x:x + w]


def test_number_only_sign_is_neither_cleaned_nor_lettered(mt):
    page = _sign_page()
    clock = _region(CLOCK_BOX, "11:59", "11:59")
    speech = _region(SPEECH_BOX, "안녕하세요", "Hello there")

    mask = mt.create_text_mask(page, [clock, speech])
    assert not _box(mask, CLOCK_BOX).any(), "the clock must not be inpainted"
    assert _box(mask, SPEECH_BOX).any()

    rendered = mt.render_translated_text(page.copy(), [clock, speech])
    assert np.array_equal(_box(rendered, CLOCK_BOX), _box(page, CLOCK_BOX)), "no second copy of the digits"
    assert not np.array_equal(_box(rendered, SPEECH_BOX), _box(page, SPEECH_BOX))


def test_number_sign_with_a_real_translation_is_still_cleaned(mt):
    page = _sign_page()
    noon = _region(CLOCK_BOX, "12:00", "Noon")
    assert _box(mt.create_text_mask(page, [noon]), CLOCK_BOX).any()


@pytest.mark.parametrize("source, late_translation", [
    ("11:59", "11:59 AM"),
    ("?!", "What?!"),
])
def test_translation_arriving_after_the_mask_is_not_lettered_over_kept_art(mt, source, late_translation):
    # Inpainting runs alongside translation: the mask is made before the translation is in,
    # keeps the sign's art, and a translation that differs must not be drawn over it.
    page = _sign_page()
    sign = _region(CLOCK_BOX, source, None)
    speech = _region(SPEECH_BOX, "안녕하세요", None)
    mask = mt.create_text_mask(page, [sign, speech])
    assert not _box(mask, CLOCK_BOX).any(), "the sign keeps its art"
    sign.translated_text = late_translation
    speech.translated_text = "Hello there"
    rendered = mt.render_translated_text(page.copy(), [sign, speech])
    assert np.array_equal(_box(rendered, CLOCK_BOX), _box(page, CLOCK_BOX)), \
        "no translation is lettered over the uncleaned sign"
    assert not np.array_equal(_box(rendered, SPEECH_BOX), _box(page, SPEECH_BOX))


def test_regions_whose_text_the_edit_left_are_not_lettered(mt, monkeypatch):
    page = np.full((340, 360, 3), 255, np.uint8)
    left = _region((40, 40, 260, 90), "그날 밤", "That night")
    speech = _region(SPEECH_BOX, "안녕하세요", "Hello there")
    text_mask = np.zeros(page.shape[:2], np.uint8)
    text_mask[40:130, 40:300] = 255
    text_mask[200:290, 40:300] = 255
    seen = {}

    class FakeImageEdit:
        model_loaded = True
        config = {}
        log_callback = None
        last_image_edit_unremoved_mask = None

        def set_log_callback(self, callback):
            self.log_callback = callback

        def inpaint(self, image, mask, **_kwargs):
            seen["fallback"] = getattr(self, "_image_edit_local_fallback", None)
            unremoved = np.zeros(mask.shape, np.uint8)
            unremoved[40:130, 40:300] = 255
            self.last_image_edit_unremoved_mask = unremoved
            return image.copy()

    monkeypatch.setattr(mt, "_get_live_local_inpaint_method", lambda: "custom-image-edit")
    fake = FakeImageEdit()
    edited = mt.inpaint_regions(page, text_mask, inpainter=fake)

    assert callable(seen["fallback"]), "the image edit gets the local inpainter fallback"
    assert getattr(fake, "_image_edit_local_fallback", None) is None
    rendered = mt.render_translated_text(edited, [left, speech])
    assert np.array_equal(_box(rendered, left.bounding_box), _box(edited, left.bounding_box))
    assert not np.array_equal(_box(rendered, SPEECH_BOX), _box(edited, SPEECH_BOX))


def test_fallback_uses_the_configured_local_model(mt, monkeypatch, tmp_path):
    import local_inpainter
    import manga_translator

    model = tmp_path / "lama-manga.onnx"
    model.write_bytes(b"onnx")
    mt.main_gui.config = dict(mt.main_gui.config)
    for method in manga_translator.MangaTranslator._IMAGE_EDIT_FALLBACK_METHODS:
        mt.main_gui.config.pop(f"manga_{method}_model_path", None)
    mt.main_gui.config["manga_settings"] = {"inpainting": {}}
    assert mt._image_edit_fallback_model() == (None, None)
    assert mt._image_edit_local_fallback(np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8), np.uint8)) is None

    mt.main_gui.config["manga_aot_onnx_model_path"] = str(model)
    assert mt._image_edit_fallback_model() == ("aot_onnx", str(model))
    loads = []

    class FakeLocal:
        def __init__(self, enable_worker_process=None):
            self.model_loaded = False

        def load_model(self, method, path, force_reload=False):
            loads.append((method, path))
            self.model_loaded = True
            return True

        def inpaint(self, image, mask, **_kwargs):
            out = image.copy()
            out[mask > 0] = 255
            return out

    monkeypatch.setattr(local_inpainter, "LocalInpainter", FakeLocal)
    monkeypatch.setattr(manga_translator.MangaTranslator, "_image_edit_fallback_inpainters", {})
    image = np.zeros((8, 8, 3), np.uint8)
    mask = np.zeros((8, 8), np.uint8)
    mask[2:4, 2:4] = 255
    out = mt._image_edit_local_fallback(image, mask)
    assert (out[2:4, 2:4] == 255).all() and not out[5:, 5:].any()
    mt._image_edit_local_fallback(image, mask)
    assert loads == [("aot_onnx", str(model))], "the fallback model is loaded once"


@pytest.mark.parametrize("source, translated, expected", [
    ("11:59", "11:59", True),
    ("１２：００", "12:00", True),
    ("12:00~13:00", "12:00-13:00", False),
    ("12:00", "Noon", False),
    ("12시", "12 o'clock", False),
    ("...!", "...!", True),  # symbol-only text keeps its art too (webtoon QA D4)
    ("...!", "Huh?!", False),
    ("3/4", "", True),
])
def test_number_only_rule(source, translated, expected):
    from image_edit_check import region_needs_no_lettering

    assert region_needs_no_lettering(source, translated) is expected


# ---------------------------------------------------------------------------
# PR 170: a translate image edit prompt (Translate SFX profiles) letters only bubbles and captions
# ---------------------------------------------------------------------------

FREE_BOX = (40, 40, 260, 90)  # a sound effect / sign outside the bubbles (detector class: free text)


def _free_text(box, text, translated):
    region = _region(box, text, translated)
    region.region_type = region.bubble_type = "free_text"
    return region


def _image_edit_prompt(mt, monkeypatch, prompt):
    monkeypatch.setattr(mt, "_get_live_local_inpaint_method", lambda: "custom-image-edit")
    mt.main_gui.config["custom_image_edit_system_prompt"] = prompt
    mt.main_gui.config["custom_image_edit_user_prompt"] = ""


def _builtin_prompt(name):
    import manga_env

    if name == "Default":
        return manga_env.default_custom_image_edit_system_prompt()
    return manga_env.BUILTIN_IMAGE_EDIT_PROMPT_PROFILES[name]["system"]


@pytest.mark.parametrize("profile", ["Translate SFX", "Translate SFX + background text"])
def test_translate_prompt_leaves_free_text_to_the_model(mt, monkeypatch, profile):
    """Live test: with a translate prompt the model redraws a detected sound effect / sign translated and
    Glossarion lettered its own translation over it ("SHUUUUK" over SWOOOOSH, "WANTED" over WANTED; under
    Translate SFX a title it was told to leave got English lettered over the Korean). Free text is now
    neither masked nor lettered; bubbles and captions are cleaned and lettered as before."""
    _image_edit_prompt(mt, monkeypatch, _builtin_prompt(profile))
    page = np.full((340, 360, 3), 255, np.uint8)
    cv2.putText(page, "KWANG", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 160), 4)
    sfx = _free_text(FREE_BOX, "쾅!", "BOOM!")
    speech = _region(SPEECH_BOX, "안녕하세요", "Hello there")

    mask = mt.create_text_mask(page, [sfx, speech])
    assert not _box(mask, FREE_BOX).any(), "the free text is left to the model's redraw"
    assert _box(mask, SPEECH_BOX).any(), "the bubble is still cleaned"

    rendered = mt.render_translated_text(page.copy(), [sfx, speech])
    assert np.array_equal(_box(rendered, FREE_BOX), _box(page, FREE_BOX)), "no second lettering"
    assert not np.array_equal(_box(rendered, SPEECH_BOX), _box(page, SPEECH_BOX))


def test_default_prompt_still_cleans_and_letters_free_text(mt, monkeypatch):
    _image_edit_prompt(mt, monkeypatch, _builtin_prompt("Default"))
    page = np.full((340, 360, 3), 255, np.uint8)
    sfx = _free_text(FREE_BOX, "쾅!", "BOOM!")
    speech = _region(SPEECH_BOX, "안녕하세요", "Hello there")

    mask = mt.create_text_mask(page, [sfx, speech])
    assert _box(mask, FREE_BOX).any() and _box(mask, SPEECH_BOX).any()
    rendered = mt.render_translated_text(page.copy(), [sfx, speech])
    assert not np.array_equal(_box(rendered, FREE_BOX), _box(page, FREE_BOX))
    assert not getattr(sfx, "image_edit_redraws", False)


@pytest.mark.parametrize("profile, edited", [("Default", False), ("Translate SFX", True)])
def test_page_with_only_free_text_still_reaches_a_translate_image_edit(mt, monkeypatch, profile, edited):
    """The free text is out of the mask, so a page whose only text is a sound effect has an empty mask:
    a translate prompt still sends it (the model redraws the sound effect), the Default prompt skips it."""
    _image_edit_prompt(mt, monkeypatch, _builtin_prompt(profile))
    page = np.full((200, 200, 3), 255, np.uint8)
    calls = []

    class FakeImageEdit:
        model_loaded = True
        config = {}
        log_callback = None
        last_image_edit_unremoved_mask = None

        def set_log_callback(self, callback):
            self.log_callback = callback

        def inpaint(self, image, mask, **_kwargs):
            calls.append(mask)
            return image.copy()

    mt.inpaint_regions(page, np.zeros(page.shape[:2], np.uint8), inpainter=FakeImageEdit())
    assert bool(calls) is edited
