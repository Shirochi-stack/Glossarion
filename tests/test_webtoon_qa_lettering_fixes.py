"""Webtoon QA run fixes (2026-11-06 QA report items 1, 3, 6, 8, 9 and the Webtoon Mode setting).

* D1: webtoon chunks overlap; a text crossing a chunk seam is whole in one chunk and cut off in
  the other. The cut copy (half-glyph OCR noise) was kept, translated and lettered. Now the copy a
  chunk edge cut is dropped when the other chunk has the text whole, and two halves of a text
  taller than the overlap are joined into one region.
* D2: an unreadable (upside-down) bubble came back as "[Illegible upside-down text]" and that
  placeholder was lettered into the comic. A placeholder translation is not lettered and the
  region keeps its art. Hayai OCR now scores its reads and reads a low-confidence crop once more
  turned 180 degrees, keeping that read only when it is clearly surer.
* D3: every leading/trailing quote mark was stripped from translations ('Bread" means "sharing"!').
  Only quotes that wrap the whole text (or a lone stray double quote) are removed now.
* D4: detector hits on the art (神, キ, an icon, a dashed border) were translated and lettered.
  A 1-2 character read the OCR model scored very low in a free-text region is dropped, and
  symbol-only text keeps its art unlettered.
* D5: the bubble-aware fit took a flat patch of the art beside a sound effect (or the clear half
  of a sign) for the text's container and shrank the text into it (CLICK at 9 px in a 96x71 box).
  An open patch that leaves much less room than the text area is no container.
* D6: Webtoon Mode "enabled" (what the settings dialog saves) did nothing; only "auto" and "force"
  were acted on.

Synthetic images only, sockets blocked, no API calls.

Run:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_webtoon_qa_lettering_fixes.py
"""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
for _p in (str(TESTS_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
from PIL import Image  # noqa: E402

from test_glossary_chunk_tall_images import webtoon_translator  # noqa: E402,F401  (fixture)
from test_manga_bubble_fit import _busy, _ink, _region, mt  # noqa: E402,F401  (mt is a fixture)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise OSError("network access is blocked in this test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def _text_region(box, text, chunk_index=None, chunk_range=None):
    from manga_translator import TextRegion

    x, y, w, h = box
    region = TextRegion(text=text, vertices=[(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                        bounding_box=box, confidence=0.9, region_type="text_block")
    if chunk_index is not None:
        region.chunk_index, region.chunk_y_range = chunk_index, chunk_range
    return region


# ===========================================================================
# D1: chunk seam copies
# ===========================================================================

UPPER, LOWER = (0, 1000), (900, 1900)  # chunk_height 1000, overlap 100


def test_copy_cut_by_the_lower_chunks_top_edge_is_dropped(webtoon_translator):
    whole = _text_region((100, 860, 200, 80), "큰적은잘라서칩니다! 분할번역, 실시!", 0, UPPER)
    cut = _text_region((101, 900, 198, 40), "분할번역, 실시!", 1, LOWER)  # 900-940: top cut off
    kept = webtoon_translator._deduplicate_chunk_regions([cut, whole], 100)
    assert kept == [whole]


def test_copy_cut_by_the_upper_chunks_bottom_edge_is_dropped(webtoon_translator):
    cut = _text_region((300, 975, 250, 25), "빵이돌아왔어요!", 0, UPPER)  # ends at the chunk end
    whole = _text_region((298, 978, 255, 90), "빵이돌아왔어요! 저도따라갈래요!", 1, LOWER)
    kept = webtoon_translator._deduplicate_chunk_regions([whole, cut], 100)
    assert kept == [whole]


def test_a_text_taller_than_the_overlap_is_joined_from_both_halves(webtoon_translator):
    top_half = _text_region((60, 700, 120, 300), "읽어 ...주게", 0, UPPER)
    bottom_half = _text_region((70, 900, 100, 200), "주게 ...", 1, LOWER)
    line = _text_region((70, 950, 100, 20), "주게 ...")
    bottom_half.original_regions = [line]
    kept = webtoon_translator._deduplicate_chunk_regions([top_half, bottom_half], 100)
    assert kept == [top_half]
    assert top_half.bounding_box == (60, 700, 120, 400)
    assert top_half.text == "읽어 ...주게 ..."
    # the mask covers both halves: the upper half's own box and the lower half's lines
    assert [r.bounding_box for r in top_half.original_regions] == [(60, 700, 120, 300), (70, 950, 100, 20)]


def test_texts_side_by_side_in_the_overlap_are_both_kept(webtoon_translator):
    left = _text_region((20, 880, 150, 60), "왼쪽", 0, UPPER)
    right = _text_region((400, 900, 150, 40), "오른쪽", 1, LOWER)
    kept = webtoon_translator._deduplicate_chunk_regions([left, right], 100)
    assert kept == [left, right]


def test_chunked_detection_keeps_each_seam_text_once_and_whole(tmp_path, webtoon_translator):
    """End to end through _detect_webtoon_chunk_regions with an OCR that also reports the
    cut-off part of a text at a chunk edge (as the detector did in the QA run)."""
    page = tmp_path / "strip.png"
    image = Image.new("RGB", (400, 2700), (255, 255, 255))
    texts = {10: (40, 860, 120, 80), 30: (200, 1880, 100, 70), 50: (60, 1400, 140, 70)}
    pixels = np.array(image)
    for gray, (x, y, w, h) in texts.items():
        pixels[y:y + h, x:x + w] = gray
    Image.fromarray(pixels).save(page)

    def detect(image_path):
        gray = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
        found = []
        for value in texts:
            ys, xs = np.nonzero(gray == value)
            if not len(ys):
                continue
            box = (int(xs.min()), int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1)
            whole = box[2:] == texts[value][2:]
            found.append(_text_region(box, f"text-{value}" if whole else "#cut#"))
        return found

    webtoon_translator.detect_text_regions = detect
    _image, regions, chunks = webtoon_translator._detect_webtoon_chunk_regions(str(page))
    assert chunks == 3
    assert sorted((r.text, r.bounding_box) for r in regions) == sorted(
        (f"text-{value}", box) for value, box in texts.items())


# ===========================================================================
# D2: placeholder translations and upside-down text
# ===========================================================================

@pytest.mark.parametrize("text, expected", [
    ("[Illegible upside-down text]", True),
    ("[Unintelligible text]", True),
    ("[Illegible text]", True),
    ("(text unreadable)", True),
    ("[AI RESPONSE UNAVAILABLE]", True),
    ("[TRANSLATION FAILED]", True),
    ("[REDACTED] betrays us next chapter!", False),
    ("[REDACTED]", False),
    ("[QA Scan] Bareum: Charmed", False),
    ("(She also packed the Brick Bread as a keepsake.)", False),
    ("[Illegible] he wrote, and left.", False),
    ("Hello", False),
    ("[No text]", True),
    # thoughts and asides come back in parentheses; system windows in square brackets
    ("(I can't read him at all...)", False),
    ("(This is gibberish!)", False),
    ("(No texts from him all day...)", False),
    ("[You have no text messages]", False),
    ("[The letter is old, faded and nearly illegible, but the seal is still bright red]", False),
])
def test_unreadable_translation_markers(text, expected):
    from manga_translator import _is_unreadable_translation

    assert _is_unreadable_translation(text) is expected


def test_bracketed_source_keeps_its_bracketed_translation():
    from manga_translator import _is_unreadable_translation

    assert _is_unreadable_translation("[Unreadable]", "읽을 수 없는 글씨") is True
    # a system window: the source is bracketed too, so the brackets are the translation's own
    assert _is_unreadable_translation("[Status: Unreadable]", "[상태: 판독 불가]") is False
    assert _is_unreadable_translation("(Illegible...)", "(알아볼 수가 없어...)") is False


def _two_bubble_page():
    source = np.full((300, 300, 3), 255, np.uint8)
    for y in (40, 180):
        for k in range(4):
            cv2.line(source, (50 + 22 * k, y + 8), (60 + 22 * k, y + 32), (20, 20, 20), 3)
    return source


def test_placeholder_region_keeps_its_art_and_is_not_lettered(mt):
    source = _two_bubble_page()
    cleaned = np.full_like(source, 255)  # inpainting ran before the translations were known
    unreadable = _region((40, 40, 110, 40), "[Illegible upside-down text]")
    readable = _region((40, 180, 110, 40), "Hello there")
    restored = mt._restore_art_of_unreadable_regions(source, cleaned, [unreadable, readable])
    assert restored == 1
    assert (cleaned[40:80, 40:150] == source[40:80, 40:150]).all(), "the source text is back"
    assert (cleaned[180:220] == 255).all(), "the lettered region stays clean"
    before = cleaned.copy()
    out = mt.render_translated_text(cleaned, [unreadable, readable])
    ink = _ink(before, out)
    assert not ink[30:90].any(), "no placeholder is lettered"
    assert ink[170:230].any(), "the real translation is lettered"


def _scored_engine(tmp_path, peaks):
    """A HayaiOCR on fake sessions whose decoder emits token 4 then EOS with logit peaks."""
    import hayai_ocr as H
    from test_hayai_ocr import FakeVision, _tokenizer_file

    class Decoder:
        def __init__(self):
            self.step = 0

        def run(self, _outputs, feeds):
            token = 4 if self.step == 0 else H.EOS_ID
            logits = np.zeros((1, 16004), np.float32)
            logits[0, token] = peaks[min(self.step, len(peaks) - 1)]
            self.step += 1
            new = feeds["vision"].shape[1] + feeds["tokens"].shape[1]
            present = np.concatenate([feeds["past"], np.zeros((24, 1, 2, new, 64), np.float32)], axis=3)
            return [logits, present]

    return H.HayaiOCR("x/vision.onnx", "x/decoder.onnx", str(_tokenizer_file(tmp_path)),
                      session_factory=lambda p: FakeVision() if "vision" in p else Decoder())


def test_hayai_scores_its_reads(tmp_path):
    sure_text, sure = _scored_engine(tmp_path, [30.0]).recognize_scored(np.zeros((20, 20, 3), np.uint8))
    _text, unsure = _scored_engine(tmp_path, [1.0]).recognize_scored(np.zeros((20, 20, 3), np.uint8))
    assert sure_text == "Hello" and sure > 0.99
    assert unsure < 0.01
    assert _scored_engine(tmp_path, [30.0]).generate(np.zeros((20, 20, 3), np.uint8)) == [4]


class _OrientationEngine:
    """Reads 'upright' text only when the crop's dark marker is in its top-left corner."""

    def __init__(self, turned_confidence=0.95):
        self.turned_confidence = turned_confidence
        self.calls = 0

    def recognize_scored(self, rgb, stop=None):
        self.calls += 1
        if rgb[:5, :5].mean() < 50:
            return "뿔달린번역가가왔다", self.turned_confidence
        return "바르주 닸온8", 0.12


def _provider(engine):
    import ocr_manager

    provider = ocr_manager.HayaiOCRProvider()
    provider.model, provider.is_loaded = engine, True
    return provider


def _upside_down_crop():
    crop = np.full((40, 60, 3), 255, np.uint8)
    crop[-5:, -5:] = 0  # the marker sits bottom-right: the text is upside down
    return crop


def test_low_confidence_crop_is_read_once_turned_180_degrees():
    engine = _OrientationEngine()
    (result,) = _provider(engine).detect_text(_upside_down_crop(), confidence=0.0)
    assert result.text == "뿔달린번역가가왔다" and result.confidence == pytest.approx(0.95)
    assert result.confidence_scored is True
    assert engine.calls == 2


def test_turned_read_is_kept_only_when_clearly_surer():
    engine = _OrientationEngine(turned_confidence=0.3)
    (result,) = _provider(engine).detect_text(_upside_down_crop(), confidence=0.0)
    assert result.text == "바르주 닸온8" and result.confidence == pytest.approx(0.12)
    assert engine.calls == 2


def test_sure_read_is_not_read_again():
    engine = _OrientationEngine()
    crop = np.full((40, 60, 3), 255, np.uint8)
    crop[:5, :5] = 0  # upright
    (result,) = _provider(engine).detect_text(crop)
    assert result.text == "뿔달린번역가가왔다" and engine.calls == 1


# ===========================================================================
# D3: quote marks
# ===========================================================================

@pytest.mark.parametrize("text, expected", [
    ('"Bread" means "sharing"!', '"Bread" means "sharing"!'),
    ('"A friend" is "someone who goes with you."', '"A friend" is "someone who goes with you."'),
    ("Glossary updated: 'Praise' → 'Sincerity'", "Glossary updated: 'Praise' → 'Sincerity'"),
    ('We said, "It\'s easier being alone, you see."', 'We said, "It\'s easier being alone, you see."'),
    ("Yesterday she hit this plaque 'normally.'", "Yesterday she hit this plaque 'normally.'"),
    ('"Hello there!"', "Hello there!"),
    ("“Hello there!”", "Hello there!"),
    ("'I can't go.'", "I can't go."),
    ('"He said "no"."', '"He said "no"."'),  # straight quotes cannot nest: ambiguous, kept
    ('Hello there!"', "Hello there!"),
    ('"Hello there!', "Hello there!"),
    ("goin'", "goin'"),
    ("「こんにちは」", "「こんにちは」"),
])
def test_only_wrapping_or_stray_quotes_are_removed(webtoon_translator, text, expected):
    assert webtoon_translator._clean_translation_text(text) == expected


# ===========================================================================
# D4: detections on the art
# ===========================================================================

@pytest.mark.parametrize("text, confidence, kind, scored, expected", [
    ("神", 0.146, "free_text", True, True),
    ("キ", 0.106, "free_text", True, True),
    ("✉", 0.139, "free_text", True, True),
    ("千斤", 0.981, "free_text", True, False),     # a real two-character sign
    ("BANDAI", 0.162, "free_text", True, False),   # a misread real sign: more than 2 characters
    ("P.", 0.06, "text_bubble", True, False),      # inside a bubble: never dropped
    ("神", 0.0, "free_text", False, False),        # the caller's threshold, not a score
])
def test_art_misread_filter(text, confidence, kind, scored, expected):
    from manga_translator import _ocr_read_is_art_noise

    result = SimpleNamespace(confidence=confidence, confidence_scored=scored)
    region = SimpleNamespace(text=text, bubble_type=kind, region_type=kind)
    assert _ocr_read_is_art_noise(result, region) is expected


def test_symbol_only_text_keeps_its_art_and_is_not_lettered(mt):
    page = np.full((200, 400, 3), 230, np.uint8)
    for x in range(60, 340, 24):  # a dashed bubble border, read as "ー"
        cv2.line(page, (x, 100), (x + 12, 100), (30, 30, 30), 3)
    dash = _region((50, 90, 300, 20), "—")
    dash.text = "ー"
    assert not mt.create_text_mask(page, [dash]).any(), "the border is not inpainted"
    out = mt.render_translated_text(page.copy(), [dash])
    assert not _ink(page, out).any(), "nothing is lettered over it"


# ===========================================================================
# D5: a flat patch of the art is not a text container
# ===========================================================================

def _sfx_page():
    """Flat floor with a busy strip (a boot) over the left part of a sound effect's box."""
    page = np.full((300, 420, 3), (170, 200, 225), np.uint8)
    page[:, 60:135] = _busy(300, 75, seed=5)  # covers the box's left 35 px, not its middle
    return page, (100, 100, 160, 90)


def test_open_flat_patch_beside_free_text_is_no_container(mt):
    page, box = _sfx_page()
    assert mt._find_text_container(page, box, None) is None
    region = _region(box, "CLICK", bubble_bounds=None)
    area = mt._container_render_area(page, region, box)
    l, t, r, b = mt._text_ink_pads(int(mt.max_font_size_limit))
    assert area == (box[0] + l, box[1] + t, box[2] - l - r, box[3] - t - b)
    out = mt.render_translated_text(page.copy(), [_region(box, "CLICK", bubble_bounds=None)])
    xs = np.nonzero(_ink(page, out).any(axis=0))[0]
    assert xs.min() < 135, "the text uses the whole box, not only the flat part"


def test_closed_box_inside_a_larger_detector_box_is_still_the_container(mt):
    page = np.full((300, 700, 3), (90, 40, 40), np.uint8)
    cv2.rectangle(page, (60, 100), (640, 180), (235, 248, 252), -1)
    cv2.rectangle(page, (60, 100), (640, 180), (30, 30, 30), 3)
    container = mt._find_text_container(page, (48, 70, 605, 140), None)
    assert container is not None
    x, y, w, h = container["rect"]
    assert x > 60 and y > 100 and x + w < 640 and y + h < 180


# ===========================================================================
# D6: Webtoon Mode "enabled"
# ===========================================================================

@pytest.mark.parametrize("value, expected", [
    ("auto", "auto"), ("enabled", "enabled"), ("disabled", "disabled"), ("force", "enabled"),
    ("force_webtoon", "enabled"), ("force_manga", "disabled"), (True, "enabled"), (False, "disabled"),
    ("", "auto"), (None, "auto"), ("Enabled", "enabled"),
])
def test_webtoon_mode_values(value, expected):
    from manga_translator import _normalize_webtoon_mode

    assert _normalize_webtoon_mode(value) == expected


def _ocr_calls(translator, tmp_path, size, mode, format_detection=True):
    translator.main_gui.config["manga_settings"]["advanced"]["webtoon_mode"] = mode
    translator.main_gui.config["manga_settings"]["advanced"]["format_detection"] = format_detection
    page = tmp_path / f"page_{mode}_{format_detection}.png"
    Image.new("RGB", size, (255, 255, 255)).save(page)
    calls = []

    def detect(image_path):
        calls.append(tuple(cv2.imread(image_path).shape[:2]))
        return []

    translator.detect_text_regions = detect
    result = translator.process_image(str(page), str(tmp_path / "out" / page.name), ocr_only=True)
    assert result["success"]
    return calls


def test_enabled_webtoon_mode_chunks_a_tall_page(tmp_path, webtoon_translator):
    # 600 x 1500: aspect 2.5, so "auto" does not take it for a webtoon; chunk_height is 1000
    assert _ocr_calls(webtoon_translator, tmp_path, (600, 1500), "auto") == [(1500, 600)]
    assert _ocr_calls(webtoon_translator, tmp_path, (600, 1500), "enabled") == [(1000, 600), (600, 600)]
    assert _ocr_calls(webtoon_translator, tmp_path, (600, 1500), "force") == [(1000, 600), (600, 600)]
    assert _ocr_calls(webtoon_translator, tmp_path, (600, 1500), "disabled") == [(1500, 600)]


def test_enabled_webtoon_mode_works_without_format_detection(tmp_path, webtoon_translator):
    assert _ocr_calls(webtoon_translator, tmp_path, (400, 2700), "enabled", format_detection=False) == [
        (1000, 400), (1000, 400), (900, 400)]
    assert _ocr_calls(webtoon_translator, tmp_path, (400, 2700), "auto", format_detection=False) == [(2700, 400)]


def test_enabled_webtoon_mode_leaves_a_page_shorter_than_a_chunk_alone(tmp_path, webtoon_translator):
    assert _ocr_calls(webtoon_translator, tmp_path, (600, 900), "enabled") == [(900, 600)]
