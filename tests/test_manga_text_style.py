"""Per-box text style of the manga editor (the Format toolbar): renderer, page state and OCR export.

* R: ``MangaTranslator.render_translated_text`` with ``region.style``: fixed size (no auto-fit),
  alignment per line, underline pixels, bold / italic from a real face of the family or
  synthesized (bold: same-colour stroke; italic: shear), text colour; without a style the output
  is pixel-identical to the renderer before the feature (``git show BASE_SHA``), also on the
  mixed-font (emote) path.
* F: ``find_font_face`` picks the family's bold / italic face beside the font file.
* S: ``manga_ocr_io`` style normalization, ``region_styles`` in the page state, the OCR export /
  import round trip (per-region ``style`` and the saved editor state) and ``TextRegion`` records.
* A: ``manga_editor_core._attach_region_styles`` (styles onto the regions a render path built) and
  ``_update_region_styles`` / ``_clear_region_styles``.

Fixed font: tests/fixtures/manga_editor/DejaVuSans-Latin.ttf (DejaVu Sans, Latin subset).
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
from PIL import ImageFont  # noqa: E402

import manga_editor_core as core  # noqa: E402
import manga_ocr_io  # noqa: E402
import manga_translator  # noqa: E402

#: main just before the Format toolbar: the renderer a page without styles must still match.
BASE_SHA = "8b3cba3466f081e5233cf07b305a464933f645d0"
FONT_PATH = TESTS_DIR / "fixtures" / "manga_editor" / "DejaVuSans-Latin.ttf"
#: Two boxes on a 400x300 white page: a bubble and a wide caption.
BOXES = ((30, 30, 150, 110), (40, 180, 320, 90))


# ===========================================================================
# Helpers
# ===========================================================================

def _translator(module=manga_translator):
    translator = module.MangaTranslator(ocr_config={'provider': 'custom-api'},
                                        unified_client=types.SimpleNamespace(model='gpt-test'),
                                        main_gui=types.SimpleNamespace(config={}), log_callback=None,
                                        skip_inpainter_init=True)
    translator.selected_font_style = str(FONT_PATH)
    return translator


def _regions(items, module=manga_translator):
    """``items``: (translation, style or None) per box of BOXES."""
    regions = []
    for (x, y, w, h), (text, style) in zip(BOXES, items):
        region = module.TextRegion(text='src', vertices=[(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                                   bounding_box=(x, y, w, h), confidence=1.0, region_type='text_block',
                                   translated_text=text)
        if style is not None:
            region.style = style
        regions.append(region)
    return regions


def _page():
    return np.full((300, 400, 3), 255, np.uint8)


def _render(translator, items, module=manga_translator):
    return translator.render_translated_text(_page(), _regions(items, module))


def _ink(image, box=None):
    """Mask of the non-white pixels (inside ``box`` = (x, y, w, h) when given)."""
    mask = np.any(image < 200, axis=2)
    if box is not None:
        x, y, w, h = box
        out = np.zeros_like(mask)
        out[max(0, y - 20):y + h + 20, max(0, x - 20):x + w + 20] = mask[max(0, y - 20):y + h + 20,
                                                                          max(0, x - 20):x + w + 20]
        mask = out
    return mask


def _ink_box(mask):
    ys, xs = np.nonzero(mask)
    assert len(xs), "nothing was rendered"
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


@pytest.fixture(scope="module")
def frozen_module():
    """manga_translator.py as it was at BASE_SHA, loaded under its own module name."""
    try:
        source = subprocess.run(["git", "show", f"{BASE_SHA}:src/manga_translator.py"], cwd=str(REPO_ROOT),
                                capture_output=True, check=True, timeout=120).stdout.decode("utf-8")
    except Exception as error:  # a shallow checkout without the base commit
        pytest.skip(f"git show {BASE_SHA} failed: {error}")
    name = "manga_translator_before_format_toolbar"
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(SRC_DIR / "manga_translator.py")
    sys.modules[name] = module
    try:
        exec(compile(source, module.__file__, "exec"), module.__dict__)
    except Exception:
        sys.modules.pop(name, None)
        raise
    yield module
    sys.modules.pop(name, None)


@pytest.fixture(scope="module", autouse=True)
def _keep_print():
    """A MangaTranslator's ``__del__`` (restore_print) resets ``builtins.print`` and
    ``unified_api_client.print``: put both back for the tests that run after this module."""
    import builtins
    import gc

    import unified_api_client

    original, uc_print = builtins.print, vars(unified_api_client).get("print")
    yield
    gc.collect()
    builtins.print = original
    if uc_print is not None:
        vars(unified_api_client)["print"] = uc_print


@pytest.fixture(autouse=True)
def _font_cache():
    manga_translator._FONT_FACE_CACHE.clear()
    yield
    manga_translator._FONT_FACE_CACHE.clear()


def test_fixed_font_is_present():
    assert FONT_PATH.exists(), "whitelist tests/fixtures/manga_editor/ in .gitignore so CI checks it out"


# ===========================================================================
# R: renderer
# ===========================================================================

#: Rendering setting variants the no-style parity runs on (attribute -> value).
SETTINGS = {
    'auto': {},
    'fixed_size': {'custom_font_size': 22},
    'box_background': {'text_bg_opacity': 200, 'text_bg_style': 'box'},
    'wrap_background_shadow': {'text_bg_opacity': 180, 'text_bg_style': 'wrap', 'shadow_enabled': True},
}
NO_STYLES = (None, {}, {'align': 'center', 'font_size': 0, 'bold': False, 'color': [300, 0, 0], 'font': ''})


@pytest.mark.parametrize("setting", sorted(SETTINGS))
@pytest.mark.parametrize("emote", (False, True), ids=("primary_font", "mixed_font"))
def test_no_style_renders_exactly_as_before(frozen_module, setting, emote):
    outputs = []
    for module in (frozen_module, manga_translator):
        translator = _translator(module)
        for name, value in SETTINGS[setting].items():
            setattr(translator, name, value)
        if emote:  # the Meiryo mixing path, with the fixed font standing in for Meiryo
            translator._is_truly_custom_font = lambda: False
            translator._get_emote_fallback_font = lambda size: ImageFont.truetype(str(FONT_PATH), size)
        if module is frozen_module:
            outputs.append(_render(translator, [("Hello there friend", None), ("World ♥", None)], module))
        else:
            for style in NO_STYLES:
                outputs.append(_render(translator, [("Hello there friend", style), ("World ♥", style)]))
    assert int(np.count_nonzero(_ink(outputs[0]))) > 200
    for output in outputs[1:]:
        assert np.array_equal(outputs[0], output)


def test_fixed_size_skips_the_auto_fit():
    translator = _translator()
    sizes = {}
    for size in (20, 40):
        out = _render(translator, [("HI", {'font_size': size}), ("", None)])
        x0, y0, x1, y1 = _ink_box(_ink(out, BOXES[0]))
        sizes[size] = y1 - y0
    auto = _render(translator, [("HI", None), ("", None)])
    _ax0, ay0, _ax1, ay1 = _ink_box(_ink(auto, BOXES[0]))
    expected = ImageFont.truetype(str(FONT_PATH), 40).getbbox("HI")
    outline = max(1, 40 // translator.outline_width_factor)
    assert abs(sizes[40] - (expected[3] - expected[1])) <= 2 * outline + 2
    assert 1.7 < sizes[40] / sizes[20] < 2.3
    assert sizes[40] != ay1 - ay0


def test_alignment_left_center_right():
    translator = _translator()
    centers = {}
    for align in ('left', 'center', 'right'):
        out = _render(translator, [("", None), ("Hi", {'align': align, 'font_size': 24})])
        x0, _y0, x1, _y1 = _ink_box(_ink(out, BOXES[1]))
        centers[align] = (x0 + x1) / 2.0
    x, _y, w, _h = BOXES[1]
    assert centers['left'] < x + w / 4 < centers['center'] < x + 3 * w / 4 < centers['right']
    assert abs(centers['center'] - (x + w / 2)) < 15
    # per line: a two-line text keeps both lines flush left
    out = _render(translator, [("", None), ("Hello wide world and a long second line", {'align': 'left', 'font_size': 22})])
    mask = _ink(out, BOXES[1])
    rows = np.nonzero(mask.any(axis=1))[0]
    middle = (rows.min() + rows.max()) // 2
    first, second = _ink_box(mask[:middle]), None
    lower = mask.copy()
    lower[:middle] = False
    second = _ink_box(lower)
    assert abs(first[0] - second[0]) <= 3


def test_underline_draws_a_line_under_each_line():
    translator = _translator()
    plain = _render(translator, [("Hi you", {'font_size': 30}), ("", None)])
    underlined = _render(translator, [("Hi you", {'font_size': 30, 'underline': True}), ("", None)])
    x0, _y0, x1, y1 = _ink_box(_ink(plain, BOXES[0]))
    added = _ink(underlined, BOXES[0]) & ~_ink(plain, BOXES[0])
    rows = np.nonzero(added.any(axis=1))[0]
    assert len(rows), "no underline"
    widest = max(int(added[row].sum()) for row in rows)
    assert widest >= 0.8 * (x1 - x0)
    assert rows.min() >= y1 - 6  # under the glyphs (the descender-free text ends at the baseline)


def test_bold_without_a_bold_face_is_synthesized():
    translator = _translator()
    style = translator._region_text_style(types.SimpleNamespace(style={'bold': True}))
    assert translator._resolve_region_font(style) == (None, 0, True, False)
    plain = _render(translator, [("Hello", {'font_size': 30}), ("", None)])
    bold = _render(translator, [("Hello", {'font_size': 30, 'bold': True}), ("", None)])
    assert np.count_nonzero(_ink(bold, BOXES[0])) > 1.15 * np.count_nonzero(_ink(plain, BOXES[0]))


def test_italic_without_an_italic_face_is_sheared():
    translator = _translator()

    def lean(image):
        mask = _ink(image, BOXES[0])
        x0, y0, x1, y1 = _ink_box(mask)
        middle = (y0 + y1) // 2
        top = np.nonzero(mask[y0:middle])[1].mean()
        bottom = np.nonzero(mask[middle:y1 + 1])[1].mean()
        return top - bottom

    plain = _render(translator, [("HHH", {'font_size': 40}), ("", None)])
    italic = _render(translator, [("HHH", {'font_size': 40, 'italic': True}), ("", None)])
    assert lean(italic) - lean(plain) > 2.0  # the top leans right
    assert 0.8 < np.count_nonzero(_ink(italic)) / np.count_nonzero(_ink(plain)) < 1.25


def test_bold_italic_on_the_mixed_font_path():
    translator = _translator()
    translator._is_truly_custom_font = lambda: False
    translator._get_emote_fallback_font = lambda size: ImageFont.truetype(str(FONT_PATH), size)
    plain = _render(translator, [("Hi ♥", {'font_size': 30}), ("", None)])
    styled = _render(translator, [("Hi ♥", {'font_size': 30, 'bold': True, 'italic': True}), ("", None)])
    assert np.count_nonzero(_ink(styled, BOXES[0])) > 1.1 * np.count_nonzero(_ink(plain, BOXES[0]))


def test_text_colour():
    translator = _translator()
    out = _render(translator, [("Hello", {'font_size': 30, 'color': [0, 160, 0]}), ("", None)])
    green = np.all(out == (0, 160, 0), axis=2)  # BGR
    assert np.count_nonzero(green) > 50
    plain = _render(translator, [("Hello", {'font_size': 30}), ("", None)])
    assert np.count_nonzero(np.all(plain == (0, 160, 0), axis=2)) == 0


def test_own_font_file(tmp_path):
    """A box's font file is used (a missing file falls back to the Rendering Settings font)."""
    translator = _translator()
    translator.selected_font_style = None
    copy = tmp_path / "BoxFont.ttf"
    shutil.copyfile(FONT_PATH, copy)
    style = {'font': str(copy)}
    assert translator._resolve_region_font(style) == (str(copy), 0, False, False)
    assert translator._resolve_region_font({'font': str(tmp_path / "missing.ttf")}) == (None, 0, False, False)
    assert translator._get_font(20, font_path=str(copy)).path == str(copy)


# ===========================================================================
# F: the family's real bold / italic faces
# ===========================================================================

def _fake_family(tmp_path, monkeypatch, faces):
    """Font files in tmp_path (copies of the fixed font) with fake (family, style) names."""
    names = {}
    for file_name, family, style in faces:
        path = tmp_path / file_name
        shutil.copyfile(FONT_PATH, path)
        names[os.path.normcase(str(path))] = (family, style)
    real = manga_translator._font_face_name

    def fake(path, index=0):
        if index:
            return None
        return names.get(os.path.normcase(os.path.abspath(path))) or real(path, index)

    monkeypatch.setattr(manga_translator, "_font_face_name", fake)
    return {file_name: str(tmp_path / file_name) for file_name, _f, _s in faces}


def test_find_font_face_uses_the_family_files(tmp_path, monkeypatch):
    paths = _fake_family(tmp_path, monkeypatch, [
        ("Fam-Regular.ttf", "Fam", "Regular"), ("Fam-Bold.ttf", "Fam", "Bold"),
        ("Fam-BoldCondensed.ttf", "Fam", "Bold Condensed"), ("Fam-Italic.ttf", "Fam", "Italic"),
        ("Famous-Bold.ttf", "Famous", "Bold"),
    ])
    regular = paths["Fam-Regular.ttf"]
    assert manga_translator.find_font_face(regular, True, False) == (paths["Fam-Bold.ttf"], 0)
    assert manga_translator.find_font_face(regular, False, True) == (paths["Fam-Italic.ttf"], 0)
    assert manga_translator.find_font_face(regular, True, True) is None
    assert manga_translator.find_font_face(paths["Fam-Bold.ttf"], False, False) == (regular, 0)

    translator = _translator()
    # a real bold face: no synthesis
    assert translator._resolve_region_font({'font': regular, 'bold': True}) == (paths["Fam-Bold.ttf"], 0, False, False)
    # bold italic: the bold face, italic synthesized
    assert translator._resolve_region_font({'font': regular, 'bold': True, 'italic': True}) == (
        paths["Fam-Bold.ttf"], 0, False, True)
    # the Rendering Settings font is already bold: Bold keeps it, Italic looks for Bold Italic
    translator.selected_font_style = paths["Fam-Bold.ttf"]
    assert translator._resolve_region_font({'bold': True}) == (None, 0, False, False)
    assert translator._resolve_region_font({'italic': True}) == (None, 0, False, True)  # no Bold Italic face


@pytest.mark.parametrize("size", (None, 26), ids=("auto_fit", "fixed_size"))
def test_a_collection_face_is_measured_with_that_face(tmp_path, monkeypatch, size):
    """A bold / italic face inside a font collection (macOS Helvetica.ttc) wraps and fits with that
    face, not the collection's first (regular) face it would otherwise be measured with."""
    translator = _translator()
    collection = tmp_path / "Family.ttf"  # stands in for the collection's file
    shutil.copyfile(FONT_PATH, collection)
    monkeypatch.setattr(translator, "_resolve_region_font", lambda style: (str(collection), 1, False, False))
    real = ImageFont.truetype
    indexes = []

    def truetype(font=None, size=10, index=0, *args, **kwargs):
        if os.path.normcase(str(font)) == os.path.normcase(str(collection)):
            indexes.append(index)
            index = 0  # the fixture font has one face
        return real(font, size, index, *args, **kwargs)

    monkeypatch.setattr(ImageFont, "truetype", truetype)
    style = {'bold': True, **({'font_size': size} if size else {})}
    _render(translator, [("Hello there my friend", style), ("", None)])
    assert indexes and set(indexes) == {1}


def test_find_font_face_on_a_system_family():
    """A real installed family when there is one (Windows Arial, Linux DejaVu Sans)."""
    for regular, bold in ((r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
                          ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                           "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")):
        if os.path.isfile(regular) and os.path.isfile(bold):
            face = manga_translator.find_font_face(regular, True, False)
            assert face is not None and os.path.normcase(face[0]) == os.path.normcase(bold)
            return
    pytest.skip("no Arial / DejaVu Sans family installed")


# ===========================================================================
# S: style records, page state and the OCR export
# ===========================================================================

def test_normalize_region_style():
    normalize = manga_ocr_io.normalize_region_style
    assert normalize(None) == {} and normalize("bold") == {} and normalize({}) == {}
    assert normalize({'font': ' a.ttf ', 'font_size': '24', 'bold': True, 'italic': False, 'underline': True,
                      'align': 'RIGHT', 'color': (1, 2, 3), 'other': 1}) == {
        'font': 'a.ttf', 'font_size': 24, 'bold': True, 'underline': True, 'align': 'right', 'color': [1, 2, 3]}
    assert normalize({'align': 'center', 'font_size': 0, 'color': [1, 2], 'bold': 1}) == {}
    assert normalize({'font_size': 10 ** 6})['font_size'] == manga_ocr_io.REGION_STYLE_MAX_FONT_SIZE
    assert normalize({'color': [0, 0, 256]}) == {} and normalize({'font_size': True}) == {}


def test_region_styles_in_the_page_state():
    state = {'region_styles': {'0': {'bold': True}, 'x': {'bold': True}, '2': {}, '3': 'bad'}}
    assert manga_ocr_io.region_styles_from_state(state) == {0: {'bold': True}}
    assert manga_ocr_io.set_region_style(state, 4, {'align': 'left'}) == {'align': 'left'}
    assert state['region_styles'] == {'0': {'bold': True}, '4': {'align': 'left'}}
    manga_ocr_io.set_region_style(state, 0, {})
    manga_ocr_io.set_region_style(state, 4, None)
    assert 'region_styles' not in state


def _editor_state():
    return {
        'viewer_rectangles': [{'x': x, 'y': y, 'width': w, 'height': h, 'shape': 'rect'} for x, y, w, h in BOXES],
        'recognized_texts': [{'text': 'a', 'bbox': list(BOXES[0]), 'region_index': 0},
                             {'text': 'b', 'bbox': list(BOXES[1]), 'region_index': 1}],
        'translated_texts': [{'original': {'text': 'a', 'region_index': 0}, 'translation': 'A', 'bbox': list(BOXES[0])},
                             {'original': {'text': 'b', 'region_index': 1}, 'translation': 'B', 'bbox': list(BOXES[1])}],
        'region_styles': {'1': {'font_size': 30, 'underline': True, 'color': [10, 20, 30]}},
    }


def test_ocr_export_import_round_trip(tmp_path):
    state = _editor_state()
    regions = manga_ocr_io.canonical_regions_from_editor_state(state)
    assert 'style' not in regions[0]
    assert regions[1]['style'] == {'font_size': 30, 'underline': True, 'color': [10, 20, 30]}

    image = tmp_path / "001.png"
    image.write_bytes(b"png")
    for editor_state in (state, None):  # with the saved editor state, and regions only
        page = manga_ocr_io.make_page(str(image), regions, index=1, editor_state=editor_state)
        path = tmp_path / "export.json"
        manga_ocr_io.write_document(str(path), manga_ocr_io.create_document([page], workflow='manga'))
        loaded = manga_ocr_io.load_document(str(path))['pages'][0]
        restored = manga_ocr_io.editor_state_from_page(loaded)
        assert restored['region_styles'] == state['region_styles']
        assert manga_ocr_io.region_styles_from_state(restored) == {1: state['region_styles']['1']}
        # the pipeline's TextRegion records carry it too (batch import renders it)
        text_region = manga_ocr_io.region_record_to_text_region(loaded['regions'][1], manga_translator.TextRegion)
        assert text_region.style == state['region_styles']['1']
        assert not hasattr(manga_ocr_io.region_record_to_text_region(loaded['regions'][0], manga_translator.TextRegion),
                           'style')
        assert manga_ocr_io.serialize_region(text_region)['style'] == state['region_styles']['1']


def test_text_region_record_keeps_no_style_key_without_a_style():
    region = _regions([("A", None)])[0]
    assert 'style' not in region.to_dict()
    region.style = {'bold': True}
    assert region.to_dict()['style'] == {'bold': True}


# ===========================================================================
# A: the editor core helpers
# ===========================================================================

class _States:
    def __init__(self, states=None):
        self.states = dict(states or {})
        self.saves = 0

    def get_state(self, path):
        return self.states.get(path, {})

    def set_state(self, path, state, save=True):
        self.states[path] = state
        self.saves += int(bool(save))


def _host(page, state, translation_data=None, current=True):
    return types.SimpleNamespace(
        image_state_manager=_States({page: state}),
        image_preview_widget=types.SimpleNamespace(current_image_path=page if current else None),
        _translation_data=translation_data or {})


def test_attach_region_styles_maps_regions_to_their_boxes(tmp_path):
    page = str(tmp_path / "001.png")
    state = {
        'translated_texts': [
            {'original': {'text': 'a', 'region_index': 0}, 'translation': 'Same'},
            {'original': {'text': 'b', 'region_index': 1}, 'translation': 'Other'},
            {'original': {'text': 'c', 'region_index': 2}, 'translation': 'Same'},
        ],
        'region_styles': {'2': {'bold': True}, '1': {'align': 'left'}},
    }
    # box 1's translation was just edited (live data only); two boxes carry the same text
    host = _host(page, state, {1: {'original': 'b', 'translation': 'Edited'}})
    regions = _regions([("Same", None), ("Edited", None)]) + _regions([("Same", None)])
    core._attach_region_styles(host, page, regions)
    assert [getattr(r, 'style', None) for r in regions] == [None, {'align': 'left'}, {'bold': True}]
    # a region that already carries a style (an imported record) keeps it
    regions = _regions([("Same", {'italic': True}), ("Unknown", None)])
    core._attach_region_styles(host, page, regions)
    assert regions[0].style == {'italic': True} and not hasattr(regions[1], 'style')
    # no styles: the regions are left as they were
    plain = _regions([("Same", None), ("Other", None)])
    core._attach_region_styles(_host(page, {'translated_texts': state['translated_texts']}), page, plain)
    assert not any(hasattr(r, 'style') for r in plain)
    # another page's live data is not used
    other = _regions([("Edited", None)])
    core._attach_region_styles(_host(page, state, {1: {'translation': 'Edited'}}, current=False), page, other)
    assert not hasattr(other[0], 'style')


def test_update_and_clear_region_styles(tmp_path):
    page = str(tmp_path / "001.png")
    host = _host(page, {'translated_texts': []})
    assert core._update_region_styles(host, page, [0, 2], {'bold': True, 'font_size': 30}) == {
        0: {'bold': True, 'font_size': 30}, 2: {'bold': True, 'font_size': 30}}
    assert core._update_region_styles(host, page, [0], {'bold': False, 'align': 'center', 'underline': True}) == {
        0: {'font_size': 30, 'underline': True}}
    assert core._update_region_styles(host, page, [2], {'font_size': None, 'align': 'right'}) == {
        2: {'bold': True, 'align': 'right'}}
    assert core._region_text_styles(host, page) == {0: {'font_size': 30, 'underline': True},
                                                   2: {'bold': True, 'align': 'right'}}
    assert core._update_region_styles(host, page, [2], reset=True) == {2: {}}
    assert host.image_state_manager.get_state(page)['region_styles'] == {'0': {'font_size': 30, 'underline': True}}
    core._clear_region_styles(host, page)
    assert 'region_styles' not in host.image_state_manager.get_state(page)
    assert core._update_region_styles(host, None, [0], {'bold': True}) == {}


def test_style_change_survives_a_render_that_holds_the_page_state(tmp_path):
    """A render on the executor reads the page's state dict, then set_state()s that same dict back
    with its last_render_positions: a style changed meanwhile must still be there afterwards."""
    page = str(tmp_path / "001.png")
    host = _host(page, {'translated_texts': [], 'region_styles': {'0': {'bold': True}}})
    held = host.image_state_manager.get_state(page)  # the render's read
    core._update_region_styles(host, page, [0], {'italic': True})
    held['last_render_positions'] = {'0': [0, 0, 10, 10]}
    host.image_state_manager.set_state(page, held, save=True)  # the render's write
    assert core._region_text_styles(host, page) == {0: {'bold': True, 'italic': True}}
    held = host.image_state_manager.get_state(page)
    core._clear_region_styles(host, page)
    host.image_state_manager.set_state(page, held, save=True)
    assert core._region_text_styles(host, page) == {}


def test_attach_region_styles_reaches_the_desktop_namespace():
    assert '_attach_region_styles' in core.SHARED_NAMES
