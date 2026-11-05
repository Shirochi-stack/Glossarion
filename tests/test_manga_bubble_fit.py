"""Auto font size keeps translated text inside the bubble / box drawn on the page.

MangaTranslator.render_translated_text used to fit text to the region's bounding RECTANGLE
(bounding_box with the safe area off, the default), so line ends crossed an oval bubble's outline
and narration boxes smaller than their detector box overflowed. _container_render_area now reads
the bubble interior from the page (_find_text_container) and _fit_lines_to_container fits every
line to the interior's width at its own height, outline stroke and shadow included.

Synthetic pages: oval speech bubble, rectangular and rounded narration boxes (detector box larger
than the drawn box), free text over a busy background, speed lines. Rendering runs with the app
defaults (manga_settings_defaults), headless.
"""
import builtins
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
from PIL import Image, ImageDraw  # noqa: E402

LONG = "Don't worry! Glossarion will translate this whole book for you tonight!"
NARRATION = "That night, there was no one in the library."


@pytest.fixture
def mt():
    """A MangaTranslator with the app's default rendering settings.

    Rendering never hijacks print, and this translator registers no log callback: its
    restore_print (also run by __del__ whenever it is collected) would only reset
    unified_api_client's own print on desktop, so it is switched off and both prints are put back
    as found.
    """
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


def _region(box, text, bubble_bounds="same"):
    from manga_translator import TextRegion

    x, y, w, h = box
    r = TextRegion(text="x", vertices=[(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                   bounding_box=box, confidence=1.0, region_type="text_block")
    r.translated_text = text
    r.bubble_bounds = box if bubble_bounds == "same" else bubble_bounds
    return r


def _ink(before, after):
    return np.abs(after.astype(int) - before.astype(int)).max(axis=2) > 40


def _busy(h, w, seed=3):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0)


def _render(mt, page, regions):
    return mt.render_translated_text(page.copy(), regions)


def _oval_page():
    page = _busy(500, 640)
    cv2.ellipse(page, (320, 250), (230, 150), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(page, (320, 250), (230, 150), 0, 0, 360, (20, 20, 20), 4)
    inside = np.zeros(page.shape[:2], np.uint8)
    cv2.ellipse(inside, (320, 250), (230 - 2 - 3, 150 - 2 - 3), 0, 0, 360, 1, -1)
    return page, inside.astype(bool), (84, 94, 472, 312)


def _box_page(rounded):
    page = np.zeros((300, 700, 3), np.uint8)
    page[:] = (90, 40, 40)  # night sky (BGR)
    pil = Image.fromarray(page)
    d = ImageDraw.Draw(pil)
    shape = (60, 100, 640, 180)  # drawn box: 580 x 80, detector box 12 px larger on each side
    if rounded:
        d.rounded_rectangle(shape, radius=28, fill=(235, 248, 252), outline=(30, 30, 30), width=3)
    else:
        d.rectangle(shape, fill=(235, 248, 252), outline=(30, 30, 30), width=3)
    page = np.array(pil)
    inside = np.zeros(page.shape[:2], np.uint8)
    pil_in = Image.fromarray(inside)
    di = ImageDraw.Draw(pil_in)
    inner = (shape[0] + 3 + 3, shape[1] + 3 + 3, shape[2] - 3 - 3, shape[3] - 3 - 3)
    if rounded:
        di.rounded_rectangle(inner, radius=22, fill=1)
    else:
        di.rectangle(inner, fill=1)
    return page, np.array(pil_in).astype(bool), (48, 88, 605, 105)


def test_oval_bubble_text_stays_inside_outline(mt):
    page, inside, box = _oval_page()
    region = _region(box, LONG)
    out = _render(mt, page, [region])
    ink = _ink(page, out)
    assert ink.sum() > 2000, "text was not rendered"
    assert not (ink & ~inside).any(), "text crosses the oval's outline / margin"
    assert region._fit_container is not None
    # line ends follow the oval: the first and last lines are narrower than the middle one
    rows = np.nonzero(ink.any(axis=1))[0]
    widths = [np.ptp(ink[r].nonzero()[0]) for r in rows]
    assert widths[0] < max(widths) and widths[-1] < max(widths)


@pytest.mark.parametrize("rounded", [False, True], ids=["rectangle", "rounded"])
def test_narration_box_text_stays_inside_box(mt, rounded):
    page, inside, box = _box_page(rounded)
    region = _region(box, NARRATION)
    out = _render(mt, page, [region])
    ink = _ink(page, out)
    assert ink.sum() > 1000, "text was not rendered"
    assert not (ink & ~inside).any(), "text spills over the narration box border"


def test_detector_box_wider_than_bubble_uses_the_drawn_box(mt):
    """The old fit used the 105 px detector box for an 80 px box: it overflowed at the old size."""
    page, _inside, box = _box_page(False)
    region = _region(box, NARRATION)
    x, y, w, h = mt._container_render_area(page, region, box)
    assert 60 < x and 100 < y and x + w < 640 and y + h < 180
    assert region._fit_container["margin"] >= 3


def test_text_box_inside_bubble_grows_to_the_bubble(mt):
    """OCR text box inside the bubble, bubble_bounds = the bubble: text may use the interior."""
    page, inside, bubble = _oval_page()
    region = _region((250, 210, 140, 80), LONG, bubble_bounds=bubble)
    x, y, w, h = mt._container_render_area(page, region, region.bounding_box)
    assert w > 140 and h > 80
    out = _render(mt, page, [_region((250, 210, 140, 80), LONG, bubble_bounds=bubble)])
    assert not (_ink(page, out) & ~inside).any()


def test_bubble_bounds_still_count_on_pages_with_several_regions(mt):
    """With several regions render_translated_text lays out _adjust_overlapping_regions' copies,
    which carry no bubble_bounds: the bubble box is taken from the region each copy was made from."""
    page, inside, bubble = _oval_page()
    page = np.vstack([page, page])
    inside = np.vstack([inside, inside])

    def regions():
        return [_region((250, 210 + 500 * i, 140, 80), LONG,
                        bubble_bounds=(bubble[0], bubble[1] + 500 * i, bubble[2], bubble[3]))
                for i in range(2)]

    out = _render(mt, page, regions())
    ink = _ink(page, out)
    for top in (0, 500):
        ys, xs = np.nonzero(ink[top:top + 500])
        assert xs.min() < 250 and xs.max() > 390, "text stayed in the OCR box instead of the bubble"
    assert not (ink & ~inside).any()
    single = _ink(page[:500], _render(mt, page[:500], regions()[:1]))
    assert (single == ink[:500]).all()


def test_free_text_on_busy_background_only_gains_the_stroke_margin(mt):
    page = _busy(300, 500, seed=7)
    box = (100, 80, 300, 120)
    region = _region(box, "Hmph! Whatever you say.")
    assert mt._find_text_container(page, box, box) is None
    area = mt._container_render_area(page, region, box)
    assert region._fit_container is None
    l, t, r, b = mt._text_ink_pads(int(mt.max_font_size_limit))
    assert area == (100 + l, 80 + t, 300 - l - r, 120 - t - b)
    # the fit itself is today's rectangle fit on that area
    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    size, lines = mt._fit_text_to_region(region.translated_text.upper(), area[2], area[3], draw, region,
                                         use_as_is=True)
    assert (size, lines) == mt._fit_text_to_region(region.translated_text.upper(), area[2], area[3],
                                                   draw, None, use_as_is=True)
    out = _render(mt, page, [_region(box, "Hmph! Whatever you say.")])
    ink = _ink(page, out)
    ys, xs = np.nonzero(ink)
    assert ink.sum() > 0
    assert xs.min() >= 100 and xs.max() < 400 and ys.min() >= 80 and ys.max() < 200


def test_speed_lines_are_not_a_bubble(mt):
    page = np.zeros((200, 300, 3), np.uint8)
    page[:] = (40, 90, 230)
    for k in range(-300, 300, 22):
        cv2.line(page, (150 + k, 0), (150 + k // 3, 200), (200, 240, 255), 4)
    assert mt._find_text_container(page, (60, 60, 180, 80), (60, 60, 180, 80)) is None


def test_overflow_allowed_keeps_the_area(mt):
    page, _inside, box = _oval_page()
    mt.constrain_to_bubble = False
    region = _region(box, LONG)
    assert mt._container_render_area(page, region, box) == box
    assert region._fit_container is None


def test_multiplier_mode_stays_inside(mt):
    page, inside, box = _oval_page()
    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    region = _region(box, LONG)
    area = mt._container_render_area(page, region, box)
    text = LONG.upper()
    base, _ = mt._fit_text_to_region(text, area[2], area[3], draw, region, use_as_is=True)
    mt.font_size_mode = "multiplier"
    mt.font_size_multiplier = 0.5
    half, _ = mt._fit_text_to_region(text, area[2], area[3], draw, region, use_as_is=True)
    assert half == int(base * 0.5)
    mt.font_size_multiplier = 3.0  # would exceed the bubble: kept at the fitted size
    big, _ = mt._fit_text_to_region(text, area[2], area[3], draw, region, use_as_is=True)
    assert big == base
    out = _render(mt, page, [_region(box, LONG)])
    assert not (_ink(page, out) & ~inside).any()


def test_fixed_font_size_mode_still_renders_at_that_size(mt):
    page, _inside, box = _oval_page()
    mt.custom_font_size = 18
    out = _render(mt, page, [_region(box, LONG)])
    ink = _ink(page, out)
    rows = np.nonzero(ink.any(axis=1))[0]
    assert ink.sum() > 0 and np.ptp(rows) < 300


def test_fit_is_fast(mt):
    page = _busy(1400, 900, seed=11)
    regions = []
    for i in range(4):
        cy = 170 + i * 340
        cv2.ellipse(page, (450, cy), (260, 140), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(page, (450, cy), (260, 140), 0, 0, 360, (0, 0, 0), 4)
        regions.append(_region((186, cy - 144, 528, 288), LONG + " " + NARRATION))
    _render(mt, page, [_region((186, 26, 528, 288), "warm up")])
    t0 = time.perf_counter()
    _render(mt, page, regions)
    assert time.perf_counter() - t0 < 3.0
    t0 = time.perf_counter()
    for _ in range(10):
        mt._find_text_container(page, (186, 26, 528, 288), (186, 26, 528, 288))
    assert (time.perf_counter() - t0) / 10 < 0.1
