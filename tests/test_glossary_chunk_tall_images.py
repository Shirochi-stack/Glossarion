"""Glossary generation chunks tall images the way translation does (owner request, 2026-11-06).

1. Manga Translator glossary workflow: the OCR-only pass (``process_image(..., ocr_only=True)``)
   OCRs a webtoon strip chunk by chunk with the webtoon path's chunking and overlap de-duplication
   (``MangaTranslator._detect_webtoon_chunk_regions``), so its regions equal the ones the webtoon
   translation path detects, and the translation pass reuses them without OCRing again.
2. Main-window image glossary extraction (``GlossaryPipelineMixin._process_image_folder_for_glossary``)
   cuts a tall image with the image translator's chunk ranges (``image_translator.ImageChunkPlanner``):
   one request per chunk (merged + de-duplicated) or, with SINGLE_API_IMAGE_CHUNKS=1, every chunk in
   one request; a short image stays one request with the whole file.

Run:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_glossary_chunk_tall_images.py
"""

from __future__ import annotations

import base64
import gc
import io
import json
import os
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
for _p in (str(TESTS_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
from PIL import Image, ImageDraw  # noqa: E402

from _headless_env import headless_owner  # noqa: E402

# ===========================================================================
# 1. Manga glossary OCR pass on a webtoon strip
# ===========================================================================

#: (x, y, w, h, gray) text boxes on a 400 x 2700 strip. With chunk_height 1000 / overlap 100 the
#: chunks are y 0-1000, 900-1900, 1800-2700: the boxes at y 920 and 1820 lie inside an overlap, so
#: two chunks each see them and both passes run the same overlap de-duplication on them.
STRIP_BOXES = ((40, 120, 120, 60, 10), (200, 920, 100, 50, 30), (60, 1400, 140, 70, 50),
               (220, 1820, 90, 60, 70), (100, 2500, 150, 80, 90))
WEBTOON_SETTINGS = {"advanced": {"format_detection": True, "webtoon_mode": "auto"},
                    "preprocessing": {"chunk_height": 1000, "chunk_overlap": 100}}
#: (height, width) of every image the OCR sees on the strip: the three chunks
STRIP_CHUNKS = [(1000, 400), (1000, 400), (900, 400)]


def _write_strip(path):
    image = Image.new("RGB", (400, 2700), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for x, y, w, h, gray in STRIP_BOXES:
        draw.rectangle((x, y, x + w - 1, y + h - 1), fill=(gray, gray, gray))
    image.save(path)


def _fake_detect(calls):
    """A stand-in OCR: every fully visible dark box of the (chunk) image is one region."""
    from manga_translator import TextRegion

    def detect_text_regions(image_path):
        image = cv2.imread(image_path)
        calls.append(tuple(image.shape[:2]))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        regions = []
        for x, y, w, h, value in STRIP_BOXES:
            found = np.argwhere(gray == value)
            if not len(found):
                continue
            (y0, x0), (y1, x1) = found.min(axis=0), found.max(axis=0)
            if (y1 - y0 + 1, x1 - x0 + 1) != (h, w):
                continue  # cut by the chunk edge
            regions.append(TextRegion(text=f"text-{value}", vertices=[(int(x0), int(y0)), (int(x1), int(y1))],
                                      bounding_box=(int(x0), int(y0), w, h), confidence=0.99,
                                      region_type="text_block"))
        return regions

    return detect_text_regions


@pytest.fixture
def webtoon_translator(tmp_path, monkeypatch):
    import builtins

    import manga_translator
    import unified_api_client

    monkeypatch.delenv("GLOSSARION_MOBILE", raising=False)
    original_print = builtins.print
    uc_print = vars(unified_api_client).get("print")
    with headless_owner(tmp_path / "owner", monkeypatch, {"model": "gpt-4o-mini"}) as owner:
        owner.save_config = lambda show_message=True: None
        owner.config["manga_settings"] = json.loads(json.dumps(WEBTOON_SETTINGS))
        translator = manga_translator.MangaTranslator({"provider": "custom-api"}, None, owner,
                                                      log_callback=lambda *_a: None,
                                                      skip_inpainter_init=True, skip_ocr_init=True)
        translator.skip_inpainting = True
        translator.full_page_context_enabled = False
        try:
            yield translator
        finally:
            translator.restore_print()
            # pytest keeps the fixture value until after this teardown, so __del__ runs during a
            # later test; its restore would put the plain print back into unified_api_client there
            translator.restore_print = lambda: None
            del translator
            gc.collect()
            builtins.print = original_print
            vars(unified_api_client)["print"] = uc_print


def _region_rows(regions):
    rows = []
    for region in regions:
        data = region.to_dict() if hasattr(region, "to_dict") else dict(region)
        rows.append((data["text"], tuple(data["bounding_box"]), [tuple(v) for v in data["vertices"]]))
    return rows


def test_glossary_ocr_pass_chunks_a_webtoon_like_the_translation_path(tmp_path, webtoon_translator):
    translator = webtoon_translator
    page = tmp_path / "strip.png"
    _write_strip(page)

    # glossary OCR pass
    ocr_calls = []
    translator.detect_text_regions = _fake_detect(ocr_calls)
    glossary = translator.process_image(str(page), str(tmp_path / "out" / "strip.png"), ocr_only=True)
    assert glossary["success"] and glossary["ocr_only"], glossary
    assert ocr_calls == STRIP_CHUNKS  # three chunks, never the whole strip
    assert glossary["format_info"]["chunks_processed"] == 3

    # the webtoon translation path on the same page (translation and rendering stubbed)
    translated_with = []

    def translate_regions(regions, image_path):
        translated_with.append(_region_rows(regions))
        for region in regions:
            region.translated_text = region.text.upper()
        return regions

    translator.translate_regions = translate_regions
    translator.render_translated_text = lambda image, regions: image
    webtoon_calls = []
    translator.detect_text_regions = _fake_detect(webtoon_calls)
    webtoon = translator.process_image(str(page), str(tmp_path / "out2" / "strip.png"))
    assert webtoon["success"] and webtoon["format_info"]["chunks_processed"] == 3, webtoon
    assert webtoon_calls == ocr_calls

    # the glossary pass hands over exactly the regions the webtoon translation path translates
    rows = _region_rows(glossary["_region_objects"])
    assert translated_with == [rows]
    brief = [(text, box) for text, box, _v in rows]
    assert [(r["text"], tuple(r["bounding_box"])) for r in webtoon["regions"]] == brief
    assert [(r["text"], tuple(r["bounding_box"])) for r in glossary["regions"]] == brief
    # ... in full-image coordinates, every box of the strip found once (the copies the two chunks of
    # an overlap both see are de-duplicated), in reading order
    expected = [(f"text-{g}", (x, y, w, h), [(x, y), (x + w - 1, y + h - 1)]) for x, y, w, h, g in STRIP_BOXES]
    assert rows == expected


def test_chunk_regions_shift_bubble_bounds_and_merged_lines(tmp_path, webtoon_translator):
    """The RT-DETR bubble (mask/render area) and a merged region's lines move to full-image coordinates."""
    from manga_translator import TextRegion

    translator = webtoon_translator
    page = tmp_path / "strip.png"
    _write_strip(page)
    detect = _fake_detect([])

    def detect_with_bubbles(image_path):
        regions = detect(image_path)
        for region in regions:
            x, y, w, h = region.bounding_box
            region.bubble_bounds = (x - 5, y - 5, w + 10, h + 10)
            region.original_regions = [TextRegion(text=region.text, vertices=[(x, y), (x + w, y)],
                                                  bounding_box=(x, y, w, 10), confidence=0.9,
                                                  region_type="text_block")]
        return regions

    translator.detect_text_regions = detect_with_bubbles
    result = translator.process_image(str(page), str(tmp_path / "out" / "strip.png"), ocr_only=True)
    regions = result["_region_objects"]
    assert [r.bounding_box for r in regions] == [(x, y, w, h) for x, y, w, h, _g in STRIP_BOXES]
    for region in regions:
        x, y, w, h = region.bounding_box
        assert region.bubble_bounds == (x - 5, y - 5, w + 10, h + 10)
        assert [line.bounding_box for line in region.original_regions] == [(x, y, w, 10)]
        assert region.original_regions[0].vertices == [(x, y), (x + w, y)]
    assert [tuple(r["bubble_bounds"]) for r in result["regions"]] == [r.bubble_bounds for r in regions]


def test_overlap_dedupe_finds_the_previous_chunk_copy_sorted_after_it(webtoon_translator):
    """OCR jitter can put the lower chunk's copy of an overlap text a pixel above the upper one's."""
    from manga_translator import TextRegion

    def region(y, chunk_index, chunk_range):
        r = TextRegion(text="same", vertices=[], bounding_box=(10, y, 50, 20), confidence=0.9,
                       region_type="text_block")
        r.chunk_index, r.chunk_y_range = chunk_index, chunk_range
        return r

    upper, lower = region(921, 0, (0, 1000)), region(920, 1, (900, 1900))
    kept = webtoon_translator._deduplicate_chunk_regions([upper, lower], 100)
    assert kept == [upper]


def test_a_stopped_chunk_ocr_is_interrupted_not_a_partial_page(tmp_path, webtoon_translator):
    translator = webtoon_translator
    page = tmp_path / "strip.png"
    _write_strip(page)
    detect = _fake_detect([])
    exported = []

    def detect_then_stop(image_path):
        regions = detect(image_path)
        translator.cancel_requested = True  # the user stops during the first chunk
        return regions

    translator.detect_text_regions = detect_then_stop
    translator.ocr_export_callback = lambda path, regions: exported.append(regions)
    monkeypatch_env = os.environ.pop("GRACEFUL_STOP", None)
    try:
        result = translator.process_image(str(page), str(tmp_path / "out" / "strip.png"), ocr_only=True)
    finally:
        translator.cancel_requested = False
        if monkeypatch_env is not None:
            os.environ["GRACEFUL_STOP"] = monkeypatch_env
    assert result.get("interrupted") and not result.get("success")
    assert "_region_objects" not in result and exported == []


def test_translation_pass_reuses_the_chunked_glossary_regions(tmp_path, webtoon_translator):
    translator = webtoon_translator
    page = tmp_path / "strip.png"
    _write_strip(page)
    translator.detect_text_regions = _fake_detect([])
    glossary = translator.process_image(str(page), str(tmp_path / "out" / "strip.png"), ocr_only=True)
    regions = glossary["_region_objects"]

    detect_calls = []
    translator.detect_text_regions = _fake_detect(detect_calls)
    webtoon_path = []
    translator._process_webtoon_chunks = lambda *a, **k: webtoon_path.append(a) or {}
    translated_with = []

    def translate_regions(regs, image_path):
        translated_with.append(_region_rows(regs))
        raise RuntimeError("stop after the reuse check")

    translator.translate_regions = translate_regions
    translator.process_image(str(page), str(tmp_path / "out3" / "strip.png"), precomputed_regions=regions)
    assert detect_calls == [] and webtoon_path == []  # no second OCR, no webtoon shortcut
    assert translated_with == [_region_rows(regions)]


@pytest.mark.parametrize("mode, chunked", (("auto", True), ("force", True), ("disabled", False)))
def test_webtoon_mode_and_short_pages_keep_the_single_ocr(tmp_path, webtoon_translator, mode, chunked):
    translator = webtoon_translator
    translator.main_gui.config["manga_settings"]["advanced"]["webtoon_mode"] = mode
    page = tmp_path / "strip.png"
    _write_strip(page)
    calls = []
    translator.detect_text_regions = _fake_detect(calls)
    result = translator.process_image(str(page), str(tmp_path / "out" / "strip.png"), ocr_only=True)
    assert result["success"]
    assert calls == (STRIP_CHUNKS if chunked else [(2700, 400)])
    assert ("chunks_processed" in result["format_info"]) == chunked

    # a normal page (aspect <= 3) is one OCR call in every mode
    short = tmp_path / "page.png"
    Image.new("RGB", (400, 600), (255, 255, 255)).save(short)
    calls.clear()
    result = translator.process_image(str(short), str(tmp_path / "out" / "page.png"), ocr_only=True)
    assert result["success"] and calls == [(600, 400)]
    assert "chunks_processed" not in result["format_info"]


# ===========================================================================
# 2. Main-window image glossary extraction
# ===========================================================================

HEADER = "type\x1Fraw_name\x1Ftranslated_name\x1Fgender"
#: per chunk request: the names the model "sees" (the overlap repeats Kim in chunks 1 and 2; the
#: last chunk has no names and answers with the CSV header only)
CHUNK_ANSWERS = (
    HEADER + "\ncharacter\x1F김철수\x1FKim Cheolsu\x1FMale",
    HEADER + "\ncharacter\x1F김철수\x1FKim Cheolsu\x1FMale\nterm\x1F서울역\x1FSeoul Station\x1F",
    HEADER,
)
CHUNK_ENV = {"IMAGE_CHUNK_HEIGHT": "1000", "IMAGE_SMART_CHUNKING": "0", "IMAGE_CHUNK_OVERLAP_PERCENT": "3",
             "IMAGE_CHUNK_MIN_OVERLAP_PIXELS": "80", "SINGLE_API_IMAGE_CHUNKS": "0"}


def _write_png(path, height, width=400):
    image = Image.new("RGB", (width, height), (255, 255, 255))
    ImageDraw.Draw(image).rectangle((20, 20, 60, 60), fill=(0, 0, 0))
    image.save(path)
    return str(path)


def _decoded_height(image_base64):
    with Image.open(io.BytesIO(base64.b64decode(image_base64))) as image:
        return image.size[1]


def _expected_chunk_heights(path):
    from image_translator import ImageChunkPlanner, ImageTranslator

    # the glossary uses the translator's own chunk planner
    assert ImageTranslator._image_chunk_ranges is ImageChunkPlanner._image_chunk_ranges
    with Image.open(path) as image:
        return [end - start for start, end in ImageChunkPlanner()._image_chunk_ranges(image.size[1], image)]


def test_send_image_glossary_request_chunks_merges_and_dedupes(tmp_path, monkeypatch):
    from image_translator import send_image_glossary_request

    for key, value in CHUNK_ENV.items():
        monkeypatch.setenv(key, value)
    tall = _write_png(tmp_path / "tall.png", 2500)
    sent = []

    def send_fn(messages, image_base64):
        sent.append((messages, image_base64))
        return CHUNK_ANSWERS[len(sent) - 1], "stop"

    messages = [{"role": "user", "content": "Extract the glossary"}]
    whole = base64.b64encode(Path(tall).read_bytes()).decode()
    response = send_image_glossary_request(send_fn, messages, tall, whole, log=lambda _m: None)
    assert [_decoded_height(b64) for _m, b64 in sent] == _expected_chunk_heights(tall) == [1000, 1000, 660]
    assert all(m is messages for m, _b in sent)
    entries = json.loads(response)
    assert [(e["type"], e["raw_name"], e["translated_name"]) for e in entries] == [
        ("character", "김철수", "Kim Cheolsu"), ("term", "서울역", "Seoul Station")]


def test_send_image_glossary_request_no_entities_is_not_a_failure(tmp_path, monkeypatch):
    from image_translator import send_image_glossary_request

    for key, value in CHUNK_ENV.items():
        monkeypatch.setenv(key, value)
    tall = _write_png(tmp_path / "tall.png", 2500)
    answers = iter((HEADER, "[]", ""))
    response = send_image_glossary_request(lambda _m, _b: next(answers), [{"role": "user", "content": "p"}],
                                           tall, "unused", log=lambda _m: None)
    assert json.loads(response) == []


def test_short_image_stays_one_request_with_the_whole_file(tmp_path, monkeypatch):
    from image_translator import send_image_glossary_request

    for key, value in CHUNK_ENV.items():
        monkeypatch.setenv(key, value)
    short = _write_png(tmp_path / "short.png", 1000)  # == IMAGE_CHUNK_HEIGHT: not chunked
    whole = base64.b64encode(Path(short).read_bytes()).decode()
    sent = []
    raw = ("raw answer", "stop")
    response = send_image_glossary_request(lambda m, b: sent.append((m, b)) or raw,
                                           [{"role": "user", "content": "p"}], short, whole, log=lambda _m: None)
    assert response is raw and len(sent) == 1 and sent[0][1] == whole


def test_a_failed_request_is_not_sent_again(tmp_path, monkeypatch):
    """An API error (or a stop) on the whole-image request propagates; the image is not re-sent."""
    from image_translator import send_image_glossary_request
    from unified_api_client import UnifiedClientError

    for key, value in CHUNK_ENV.items():
        monkeypatch.setenv(key, value)
    short = _write_png(tmp_path / "short.png", 600)
    sent = []
    logs = []

    def send_fn(messages, image_base64):
        sent.append(image_base64)
        raise UnifiedClientError("Glossary extraction stopped by user")

    with pytest.raises(UnifiedClientError):
        send_image_glossary_request(send_fn, [{"role": "user", "content": "p"}], short, "whole", log=logs.append)
    assert sent == ["whole"] and logs == []

    # an image that cannot be split is still sent whole, once
    sent.clear()
    with pytest.raises(UnifiedClientError):
        send_image_glossary_request(send_fn, [{"role": "user", "content": "p"}], str(tmp_path / "missing.png"),
                                    "whole", log=logs.append)
    assert sent == ["whole"] and len(logs) == 1


def test_single_api_mode_sends_every_chunk_in_one_request(tmp_path, monkeypatch):
    from image_translator import send_image_glossary_request

    for key, value in CHUNK_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("SINGLE_API_IMAGE_CHUNKS", "1")
    tall = _write_png(tmp_path / "tall.png", 2500)
    sent = []
    raw = (CHUNK_ANSWERS[1], "stop")
    response = send_image_glossary_request(lambda m, b: sent.append((m, b)) or raw,
                                           [{"role": "user", "content": "Extract"}], tall, "whole",
                                           log=lambda _m: None)
    assert response is raw and len(sent) == 1
    messages, image_base64 = sent[0]
    assert image_base64 is None
    content = messages[-1]["content"]
    assert content[0] == {"type": "text", "text": "Extract"}
    urls = [part["image_url"]["url"] for part in content[1:]]
    assert [_decoded_height(u.split(",", 1)[1]) for u in urls] == [1000, 1000, 660]


class _FakeClient:
    """unified_api_client.UnifiedClient stand-in: answers each send_image with the next chunk answer."""

    sent = []

    def __init__(self, model=None, api_key=None, **_kwargs):
        self.model = model

    @staticmethod
    def _model_needs_api_key(_model):
        return True

    def send_image(self, messages, image_data, temperature=None, max_tokens=None, context=None, **_kw):
        type(self).sent.append((context, image_data))
        if len(type(self).sent) == 1:
            return "[]", "stop"  # the short image: one request, no names
        return CHUNK_ANSWERS[len(type(self).sent) - 2], "stop"  # the tall image's chunks


def test_image_folder_glossary_sends_one_request_per_chunk_and_merges(tmp_path, monkeypatch):
    import unified_api_client

    _FakeClient.sent = []
    monkeypatch.setattr(unified_api_client, "UnifiedClient", _FakeClient)
    folder = tmp_path / "images"
    folder.mkdir()
    short = _write_png(folder / "a_short.png", 600)
    tall = _write_png(folder / "b_tall.png", 2500)
    logs = []
    with headless_owner(tmp_path / "owner", monkeypatch, {"model": "gpt-4o-mini", "api_key": "sk-test"}) as owner:
        owner.save_config = lambda show_message=True: None
        os.environ.update(CHUNK_ENV)
        owner.append_log = logs.append
        owner.manual_glossary_prompt = "Extract {entries}\n{fields1}"
        if hasattr(owner, "api_key_entry") and hasattr(owner.api_key_entry, "setText"):
            owner.api_key_entry.setText("sk-test")
        owner.model_var = "gpt-4o-mini"
        owner.delay_entry = type("E", (), {"text": lambda self: "0"})()
        out_dir = str(tmp_path / "Glossary")
        assert owner._process_image_folder_for_glossary("images", [short, tall], out_dir) is True, "\n".join(logs)
        prog = owner.glossary_progress_manager.prog
        statuses = {info["name"]: info["status"] for info in prog["images"].values()}
        tall_hash = next(h for h, info in prog["images"].items() if info["name"] == "b_tall.png")
        tall_cached = prog["extracted_data"][tall_hash]

    # one request for the short image (whole file), one per chunk for the tall one
    assert [ctx for ctx, _b in _FakeClient.sent] == ["glossary"] * 4
    assert _decoded_height(_FakeClient.sent[0][1]) == 600
    assert _FakeClient.sent[0][1] == base64.b64encode(Path(short).read_bytes()).decode()
    assert [_decoded_height(b) for _c, b in _FakeClient.sent[1:]] == [1000, 1000, 660]
    glossary = json.loads((tmp_path / "Glossary" / "images_glossary.json").read_text(encoding="utf-8"))
    assert sorted(e["raw_name"] for e in glossary) == ["김철수", "서울역"]
    assert any("Merged 3 chunk entries into 2 unique entries" in line for line in logs), logs
    # progress stays one entry per image; the short image without names is completed, not an error
    assert statuses == {"a_short.png": "completed", "b_tall.png": "completed"}
    assert sorted(e["raw_name"] for e in tall_cached) == ["김철수", "서울역"]
