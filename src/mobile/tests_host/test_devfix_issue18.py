"""Owner item 18 (devfix4): "Chat header subtitle ellipsizes on narrow phones; Android UI test harness fixed".

What the owner saw: on phones narrower than about 470 dp the chat header's "Model · Profile · → Target ▾"
subtitle did not ellipsize. Its Row (subtitle Text + the "custom" chip, ``tight=True``) gave the one-line
Text its natural width, so the row overflowed the app bar's title slot: the text was cut off at the
edge, the "custom" chip was pushed out of sight, and in debug builds Flutter reported "A RenderFlex
overflowed by 148 pixels on the right" (Build Mobile run 37800059580, 320 dp emulator, title slot
w<=88). That FlutterError also failed both optional Android UI tests at teardown, next to the harness
bugs (swipes dropped by the integration-test binding, the Welcome race, timedelta pumps).

This acceptance test checks it end to end on the real objects, headlessly:

* The real app (``test_ui_foundations._start``: ``GlossarionApp`` on the fake Flet session as Android)
  runs at 320, 360 and 411 dp. The chat header's ``AppBar`` is encoded exactly as Flet's transport sends
  it to the Flutter client (the session's msgpack encoder), and a model of Flutter's layout runs on that
  wire tree: NavigationToolbar (56 dp leading, 48 dp actions, 16 dp title spacing: the CI log's 88 dp
  slot at 320 dp) and RenderFlex for the subtitle row, with Flet 1.0.3's ``_expandable`` mapping
  (``expand`` on a child of a ``host_expanded`` Row -> ``Expanded``, ``+ expand_loose`` -> loose
  ``Flexible``). Text widths use Roboto label-small calibrated on the CI device (the 44-character
  subtitle measured 236 px). No RenderFlex overflow, the subtitle ellipsized but still shown, the
  "custom" chip whole - for a fresh chat, a chat with "This chat only" overrides (the owner's case),
  and a scratch chat with overrides (two more header actions). The pre-fix row (no flex) is run
  through the same model and must reproduce the CI overflow, so the model would catch a regression. A resize
  within the phone class (411 -> 360 / 320 dp, 360 -> 411 dp) re-applies the narrow-phone header (DF2 verify).
* The transcript's cards at 320 dp (DF2 verify, the next CI errors): the user file card fits the 296 dp
  transcript column (the fixed 320 dp card overflowed by 24 px, the CI log's number, through the same model),
  and every ExpansionTile header of the JobCard / Plan card / batch Plan card sits on a Material ("ListTile
  background color or ink splashes may be invisible" under a coloured Container).
* The device test driver changes on the host twin: the device smoke flow (first run: Welcome -> Skip,
  drawer, Library, Settings, the lower Settings groups, self-test) at 320 dp, with the header layout
  checked on every pump; ``UiDriver`` pumps int milliseconds; the device tests treat every run as a
  fresh install.
* The ``flet test`` driver patch on ``integration_test/app_test.dart`` rendered from Flet 1.0.3's own
  cookiecutter template for this project (jinja2, as ``flet build`` renders it): applied by the device
  conftest's ``pytest_configure`` once, byte-identical on every later run (cached test host), the two
  Flutter flags set after ``runFletDeviceTest`` created the binding and before any test body, LF and
  CRLF, and a loud error when the template is not the expected one.

Run from src/mobile (mobile venv):
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_devfix_issue18.py
"""

from __future__ import annotations

import ast
import asyncio
import copy
import importlib.util
import math
import re
import sys
import time
from dataclasses import is_dataclass
from pathlib import Path
from typing import Optional

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
TESTS_DIR = MOBILE_DIR / "tests"
for _path in (APP_DIR, TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed")

_TB_SPEC = importlib.util.spec_from_file_location("_glossarion_tb_helpers_issue18",
                                                  Path(__file__).with_name("test_bootstrap.py"))
_TB = importlib.util.module_from_spec(_TB_SPEC)
_TB_SPEC.loader.exec_module(_TB)
storage = _TB.storage
app_env = _TB.app_env


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _foundations():
    return _load("_glossarion_tf_helpers_issue18", Path(__file__).with_name("test_ui_foundations.py"))


#: phone widths: the CI emulator (320x640 mdpi), a common small phone, a Pixel-class phone
PHONE_DP = (320, 360, 411)
#: a "This chat only" model long enough that the subtitle needs an ellipsis at every phone width
LONG_MODEL = "gemini-3-pro-preview"
DEFAULT_SUBTITLE = "authgpt/gpt-6-luna · Universal · → English ▾"


# ---- what the Flutter client receives ------------------------------------------------------------

_PREV = ("__prev_lists", "__prev_dicts", "__prev_classes")
_MISSING = object()


def wire(control) -> dict:
    """``control`` as Flet's transport encodes it for the Flutter client (msgpack with the session's
    encoder). The encoder records diff snapshots on every control it visits; they are put back, so
    the live app's next patch is unaffected."""
    import msgpack
    from flet.controls.base_control import BaseControl
    from flet.messaging.protocol import configure_encode_object_for_msgpack

    base = configure_encode_object_for_msgpack(BaseControl)
    saved: list = []

    def encode(obj):
        if is_dataclass(obj) and not isinstance(obj, type) and not hasattr(obj, "_frozen"):
            saved.append((obj, [getattr(obj, name, _MISSING) for name in _PREV]))
        return base(obj)

    try:
        return msgpack.unpackb(msgpack.packb(control, default=encode), strict_map_key=False)
    finally:
        for obj, values in reversed(saved):
            for name, value in zip(_PREV, values):
                if value is _MISSING:
                    try:
                        delattr(obj, name)
                    except AttributeError:
                        pass
                else:
                    setattr(obj, name, value)


def _visible(nodes) -> list:
    """Flet builds only visible children (``visible: False`` children take no slot, no spacing)."""
    return [n for n in (nodes or []) if isinstance(n, dict) and n.get("visible", True) is not False]


def _find(node, control_id: int) -> Optional[dict]:
    if isinstance(node, dict):
        if node.get("_i") == control_id:
            return node
        children = node.values()
    elif isinstance(node, list):
        children = node
    else:
        return None
    for child in children:
        found = _find(child, control_id)
        if found is not None:
            return found
    return None


# ---- a model of Flutter's layout on the wire tree ------------------------------------------------

#: Roboto label-small (11 sp, w500, +0.5 letter spacing) per character, calibrated on the CI emulator:
#: the 44-character subtitle "authgpt/gpt-6-luna · Universal · → English ▾" measured 236 px
#: ("overflowed by 148 pixels" in the 88 px title slot at 320 dp)
LABEL_SMALL_PX_PER_CHAR = 236 / 44
#: whole label-small labels with wider-than-average glyphs, so the chips are not under-sized by the
#: per-character average: Roboto Medium advance widths (the font's hmtx table, read with fontTools)
#: at 11 px plus 0.5 px letter spacing per character. The same measurement gives 232 px for the
#: subtitle above (CI: 236 px), so these agree with the calibration within ~2%.
LABEL_SMALL_PX = {"custom": 40.0, "Scratch": 41.2}
#: label-large (TextButton) per character, and Material 3 TextButton padding / minimum width
LABEL_LARGE_PX_PER_CHAR = 7.8
TEXT_BUTTON_PADDING, TEXT_BUTTON_MIN_WIDTH = 24, 64
#: Material: AppBar leading slot (kToolbarHeight), icon-button hit target, NavigationToolbar.kMiddleSpacing
LEADING_DP, ICON_BUTTON_DP, MIDDLE_SPACING_DP = 56, 48, 16


def _text_of(node: dict) -> str:
    value = node.get("value")
    if isinstance(value, str) and value:
        return value
    return "".join(span.get("text") or "" for span in node.get("spans") or [] if isinstance(span, dict))


def natural_width(node: dict) -> float:
    """The width a control takes with an unbounded main axis (a non-flex Row child)."""
    kind = node.get("_c")
    if isinstance(node.get("width"), (int, float)):
        return float(node["width"])
    if kind == "Text":
        text = _text_of(node)
        assert node.get("theme_style") == "labelSmall", f"no width model for a {node.get('theme_style')} Text"
        return LABEL_SMALL_PX.get(text, len(text) * LABEL_SMALL_PX_PER_CHAR)
    if kind == "Container":
        padding = node.get("padding") or {}
        return natural_width(node["content"]) + float(padding.get("left", 0)) + float(padding.get("right", 0))
    if kind == "TextButton":
        content = node.get("content")
        label = content if isinstance(content, str) else _text_of(content or {})
        return max(TEXT_BUTTON_MIN_WIDTH, len(label) * LABEL_LARGE_PX_PER_CHAR + TEXT_BUTTON_PADDING)
    if kind in ("IconButton", "PopupMenuButton"):
        constraints = node.get("size_constraints") or {}
        return max(ICON_BUTTON_DP, float(constraints.get("min_width") or 0))
    raise AssertionError(f"no width model for {kind}")


def _flex(child: dict, parent: dict) -> int:
    """Flet 1.0.3 ``_expandable``: ``expand`` wraps the child in Expanded / Flexible only when its parent
    is a flex host (``_internals.host_expanded``: Row, Column, View)."""
    if not (parent.get("_internals") or {}).get("host_expanded"):
        return 0
    expand = child.get("expand")
    if expand is True:
        return 1
    return int(expand) if isinstance(expand, int) and not isinstance(expand, bool) else 0


def layout_row(row: dict, max_width: float) -> dict:
    """Flutter's RenderFlex (horizontal, no wrap, no scroll) under a bounded max width: non-flex children
    at their natural width, the free space shared by flex children (loose Flexible: up to their natural
    width; Expanded: all of it). ``overflow`` > 0 is the debug "A RenderFlex overflowed" error."""
    assert row.get("_c") == "Row" and not row.get("wrap") and row.get("scroll") is None, row.get("_c")
    children = _visible(row.get("controls"))
    flexes = [_flex(child, row) for child in children]
    if any(flexes):
        assert math.isfinite(max_width), "RenderFlex children have non-zero flex but the width is unbounded"
    spacing = float(row.get("spacing", 10)) * max(0, len(children) - 1)
    sizes: dict = {}
    allocated = spacing
    for child, flex in zip(children, flexes):
        if not flex:
            sizes[child["_i"]] = (natural_width(child), natural_width(child))
            allocated += sizes[child["_i"]][1]
    free = max(0.0, max_width - allocated)
    total_flex = sum(flexes)
    for child, flex in zip(children, flexes):
        if flex:
            share = free * flex / total_flex
            natural = natural_width(child)
            sizes[child["_i"]] = (natural, min(natural, share) if child.get("expand_loose") else share)
    used = spacing + sum(width for _natural, width in sizes.values())
    return {"overflow": max(0.0, used - max_width), "sizes": sizes, "used": used}


def title_slot(bar: dict, screen_width: float) -> float:
    """NavigationToolbar's middle slot: the bar minus the leading slot, the actions (laid out first, at
    their own width) and the title spacing on both sides."""
    assert bar.get("_c") == "AppBar"
    # the model uses Flutter's defaults for these; a change to them needs the model updated
    assert bar.get("title_spacing") is None and bar.get("leading_width") is None and not bar.get("center_title")
    leading = bar.get("leading")
    leading_px = LEADING_DP if isinstance(leading, dict) and leading.get("visible", True) is not False else 0
    trailing = sum(natural_width(action) for action in _visible(bar.get("actions")))
    return max(0.0, screen_width - leading_px - trailing - 2 * MIDDLE_SPACING_DP)


def subtitle_row(bar: dict) -> dict:
    """The title is ``Container`` (no width: passes the slot's loose constraints) -> ``Column`` (start
    alignment: children get loose widths up to the slot) -> the subtitle ``Row``."""
    title = bar["title"]
    assert title["_c"] == "Container" and "width" not in title and "alignment" not in title
    column = title["content"]
    assert column["_c"] == "Column" and column.get("horizontal_alignment") in (None, "start")
    rows = [c for c in _visible(column.get("controls")) if c.get("_c") == "Row"]
    assert len(rows) == 1, rows
    return rows[0]


def header_layout(bar: dict, screen_width: float, header) -> dict:
    slot = title_slot(bar, screen_width)
    row = subtitle_row(bar)
    result = layout_row(row, slot)
    result.update(slot=slot, row=row)
    sub = _find(row, header.subtitle._i)
    badge = _find(row, header.custom_badge._i)
    result["subtitle"] = result["sizes"].get(header.subtitle._i)
    result["badge"] = result["sizes"].get(header.custom_badge._i) if badge is not None else None
    result["subtitle_node"], result["badge_node"] = sub, badge
    return result


def pre_fix_overflow(bar: dict, screen_width: float, header) -> float:
    """The same row as it was before the fix (the subtitle not flexible)."""
    row = copy.deepcopy(subtitle_row(bar))
    node = _find(row, header.subtitle._i)
    node.pop("expand", None)
    node.pop("expand_loose", None)
    return layout_row(row, title_slot(bar, screen_width))["overflow"]


def assert_header_fits(layout: dict, *, width: int, state: str, custom: bool) -> None:
    where = f"{state} at {width} dp (title slot {layout['slot']:.1f} px)"
    assert layout["overflow"] == 0, (f"{where}: the header subtitle row overflows by {layout['overflow']:.1f} px "
                                     f"(Flutter: 'A RenderFlex overflowed'): {layout['sizes']}")
    sub = layout["subtitle_node"]
    assert sub is not None and sub.get("max_lines") == 1 and sub.get("overflow") == "ellipsis", where
    natural, laid_out = layout["subtitle"]
    assert laid_out < natural, f"{where}: the subtitle is not ellipsized ({laid_out:.1f} of {natural:.1f} px)"
    # still shown: room for at least a character and the ellipsis
    assert laid_out >= 2 * LABEL_SMALL_PX_PER_CHAR, f"{where}: the subtitle got {laid_out:.1f} px"
    if custom:
        assert layout["badge"] is not None, f"{where}: the 'custom' chip is not shown"
        badge_natural, badge = layout["badge"]
        assert badge == badge_natural > 0, f"{where}: the 'custom' chip is cut ({badge:.1f} of {badge_natural:.1f} px)"
        assert layout["used"] <= layout["slot"] + 1e-6, where
    else:
        assert layout["badge"] is None, f"{where}: a 'custom' chip without per-chat overrides"


async def _until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return bool(predicate())


def _phone_bar(page, header):
    import flet as ft

    bar = page.views[0].appbar
    assert isinstance(bar, ft.AppBar) and bar is header.wrapper, "the phone layout puts the header in View.appbar"
    return bar


# ---- the header on narrow phones -----------------------------------------------------------------


@needs_flet
@pytest.mark.parametrize("width", PHONE_DP)
def test_chat_header_subtitle_ellipsizes_in_the_app_bar(app_env, width):
    """The owner's header at 320 / 360 / 411 dp: a fresh chat, then the same chat with a "This chat only"
    model (the "custom" chip). The client gets a loose Flexible subtitle; nothing overflows."""
    tf = _foundations()

    async def scenario():
        _m, conn, session, page, app = await tf._start("android", width=width)
        try:
            view = app.chat_view
            header = view.header
            bar = _phone_bar(page, header)

            # a fresh chat: three icon actions (New scratch chat, New chat, More)
            assert header.subtitle_text == DEFAULT_SUBTITLE and not header.custom_badge.visible
            encoded = wire(bar)
            layout = header_layout(encoded, width, header)
            assert layout["slot"] == width - LEADING_DP - 3 * ICON_BUTTON_DP - 2 * MIDDLE_SPACING_DP
            # what Flet's Row sends: a flex host whose subtitle child is a loose Flexible
            assert (layout["row"].get("_internals") or {}).get("host_expanded") is True and layout["row"].get("tight")
            assert layout["subtitle_node"].get("expand") is True and layout["subtitle_node"].get("expand_loose") is True
            assert not (layout["badge_node"] or {}).get("expand"), "the chip keeps its own width"
            assert_header_fits(layout, width=width, state="fresh chat", custom=False)
            before = pre_fix_overflow(encoded, width, header)
            assert before > 0, f"the pre-fix row should overflow at {width} dp"
            if width == 320:  # the CI emulator: Build Mobile run 37800059580's numbers
                assert layout["slot"] == 88 and before == pytest.approx(148)

            # the owner's case: per-chat overrides show the "custom" chip next to the subtitle
            view._on_model_sheet_select("model", LONG_MODEL, True)
            assert await _until(lambda: header.subtitle_text.startswith(LONG_MODEL)), "the chat model did not apply"
            assert header.subtitle_text.startswith(LONG_MODEL + " · ")
            encoded = wire(_phone_bar(page, header))
            layout = header_layout(encoded, width, header)
            assert_header_fits(layout, width=width, state="chat with 'This chat only' overrides", custom=False)  # no "custom" chip any more
            assert pre_fix_overflow(encoded, width, header) > 0
        finally:
            app.jobs.close()
            await tf._stop(app)

    asyncio.run(scenario())


@needs_flet
@pytest.mark.parametrize("width", PHONE_DP)
def test_scratch_chat_header_with_custom_chip_fits(app_env, width):
    """A scratch chat (header actions: the "Scratch" chip, Save, New chat, More) with a "This chat only"
    model: the same subtitle row, with a narrower title slot, must not overflow either. The four
    actions take ~217 dp, so at 320 dp the title slot is ~15 dp: narrower than the "custom" chip
    (52 dp + 6 dp spacing) itself, which a flexible subtitle alone cannot absorb."""
    tf = _foundations()

    async def scenario():
        _m, conn, session, page, app = await tf._start("android", width=width)
        try:
            view = app.chat_view
            header = view.header
            cid = view._on_new_scratch()
            assert cid and await _until(lambda: header.scratch_chip.visible and header.save_scratch_button.visible)
            layout = header_layout(wire(_phone_bar(page, header)), width, header)
            assert_header_fits(layout, width=width, state="scratch chat", custom=False)

            view._on_model_sheet_select("model", LONG_MODEL, True)
            assert await _until(lambda: header.subtitle_text.startswith(LONG_MODEL)), "the chat model did not apply"
            assert header.scratch_chip.visible and header.save_scratch_button.visible
            layout = header_layout(wire(_phone_bar(page, header)), width, header)
            assert_header_fits(layout, width=width, state="scratch chat with 'This chat only' overrides", custom=False)  # no "custom" chip any more
        finally:
            app.jobs.close()
            await tf._stop(app)

    asyncio.run(scenario())


@needs_flet
@pytest.mark.parametrize("start,end", [(411, 360), (411, 320), (360, 411)])
def test_scratch_chat_header_follows_a_resize_within_the_phone_class(app_env, start, end):
    """The window changes width inside the phone size class (split screen, a pop-up window, a display-size
    change while the app runs: ``page.on_resize`` -> ``app.on_resize``). The narrow-phone header follows it:
    below 400 dp the scratch chat's actions are compact and the subtitle row fits; above, they are not.
    Before the fix the shell re-applied the chat layout only when the composer's state changed, so the
    header kept the start width's actions (411 -> 320 dp: a 43 px overflow)."""
    import types

    from glossarion_mobile.ui.chat.header import NARROW_HEADER_DP

    tf = _foundations()

    async def scenario():
        _m, conn, session, page, app = await tf._start("android", width=start)
        try:
            view = app.chat_view
            header = view.header
            cid = view._on_new_scratch()
            assert cid and await _until(lambda: header.scratch_chip.visible)
            view._on_model_sheet_select("model", LONG_MODEL, True)
            assert await _until(lambda: header.subtitle_text.startswith(LONG_MODEL)), "the chat model did not apply"
            assert header.narrow == (start < NARROW_HEADER_DP), (start, header.narrow)
            page.width = end
            app.on_resize(types.SimpleNamespace(width=end, height=800))
            assert app.shell.size_class.value == "phone"
            narrow = end < NARROW_HEADER_DP
            assert header.narrow == narrow and header.compact_actions == narrow, \
                (start, end, header.narrow, header.compact_actions)
            layout = header_layout(wire(_phone_bar(page, header)), end, header)
            assert_header_fits(layout, width=end, state=f"scratch chat resized from {start} dp", custom=False)
        finally:
            app.jobs.close()
            await tf._stop(app)

    asyncio.run(scenario())


# ---- the transcript's cards at the CI emulator's 320 dp (Build Mobile run 37940790686) ----------------


def _ink_problems(root) -> list:
    """``(tile, coloured Container)`` pairs: an ExpansionTile (its header is a ListTile) or a ListTile whose
    nearest surface is a Container with a background colour (Flutter: a DecoratedBox) instead of a Material
    (a Card). Flutter reports "ListTile background color or ink splashes may be invisible", a failure under
    ``flet test``. Invisible controls are checked too (a tile shown later must be fine as well); an
    ExpansionTile's own children are not (they are the content's business, e.g. the Run options' settings
    tiles, the shared settings component)."""
    import flet as ft
    from host_tester import _children

    problems: list = []

    def walk(control, surface) -> None:
        if isinstance(control, (ft.ExpansionTile, ft.ListTile)):
            if surface is not None:
                problems.append((control, surface))
            return
        if isinstance(control, ft.Card):
            surface = None
        elif isinstance(control, ft.Container) and getattr(control, "bgcolor", None) is not None:
            surface = control
        for child in _children(control):
            walk(child, surface)

    walk(root, None)
    return problems


def _contains(root, target) -> bool:
    from host_tester import _children

    return root is target or any(_contains(child, target) for child in _children(root))


@needs_flet
def test_job_cards_paint_their_tiles_on_a_material(app_env):
    """The running / Result JobCard (Requests, OCR, Extraction report tiles), the Plan card's Run options
    and the batch Plan card: every ExpansionTile header sits on the card's Material surface. A tile in a
    coloured Container is caught by the same check (the cards before the fix)."""
    import types

    import flet as ft

    from glossarion_mobile.ui.chat.batch_plan import BatchPlanCard
    from glossarion_mobile.ui.chat.cards import JobCard
    from glossarion_mobile.ui.chat.job_binding import CardPhase

    record = {"name": "book.epub", "extension": ".epub", "size": 1024}
    card = JobCard(attachment=record, phase=CardPhase("running"), requests_expanded=True)
    card.set_requests([{"label": "Chapter 1 · Request 1", "content": "Hello", "phase": "text"}])
    card.report_tile.visible = True
    assert all(_contains(card, tile) for tile in (card.requests_tile, card.ocr_tile, card.report_tile))
    assert _ink_problems(card) == []
    plan = JobCard(attachment=record, phase=CardPhase("plan"))
    plan.set_plan([ft.ExpansionTile(title="Run options", dense=True)])
    assert _ink_problems(plan) == []
    run_options = types.SimpleNamespace(control=ft.ExpansionTile(title="Run options", dense=True))
    batch = BatchPlanCard(files=["a.epub", "b.epub"], run_options=run_options)
    assert _contains(batch, run_options.control) and _ink_problems(batch) == []
    before = ft.Container(content=ft.Column([ft.ExpansionTile(title="Requests (1)", dense=True)]),
                          bgcolor=ft.Colors.SURFACE_CONTAINER, border_radius=16)
    assert len(_ink_problems(before)) == 1


@needs_flet
def test_user_file_card_and_plan_card_fit_a_320_dp_phone(app_env, tmp_path):
    """The CI flow on the real app at 320 dp: ＋ › Files with the self-test EPUB, Send. The chat's file card
    (an end-aligned Row) gets the transcript column, 320 - 2 × 12 dp: its card is a loose Flexible capped at
    320 dp, so it takes 296 dp and nothing overflows (the fixed 320 dp card overflowed by 24 px, the CI
    log's number, which the same layout model reproduces). The Plan card's tiles sit on a Material."""
    import flet as ft
    import flows
    from glossarion_mobile.diagnostics import fixtures
    from glossarion_mobile.diagnostics.fake_llm_server import FAKE_MODEL
    from glossarion_mobile.ui import tokens
    from glossarion_mobile.ui.chat.cards import JobCard
    from glossarion_mobile.ui.chat.messages import UserFileCard

    epub = fixtures.build_tiny_epub(tmp_path / flows.EPUB_NAME, chapters=3)
    twin = _load("_glossarion_ui_flows_issue18_cards", Path(__file__).with_name("test_ui_flows.py"))
    tf = _foundations()
    width = 320

    async def scenario():
        app, tester, driver = await twin._host_driver(tf, {epub.name: epub}, width=width)
        try:
            await flows.wait_home(driver)
            store = app.config_store
            store.set_many(flows.ui_config("http://127.0.0.1:9/v1", FAKE_MODEL))
            store.flush()
            await flows.go_home(driver)
            await driver.tap(tooltip="New chat")
            await driver.tap(tooltip=flows.ATTACH_TOOLTIP)
            await driver.pick_file(epub.name, lambda: driver.tap(key="attach-files"))
            await driver.wait(contains=epub.stem, timeout=60)
            await driver.tap(key="send-idle_ready", timeout=60)
            await driver.wait(text="Ready to translate", timeout=60)
            view = app.chat_view
            cards = [getattr(slot, "card", slot) for slot in view.transcript.messages]
            files = [c for c in cards if isinstance(c, UserFileCard)]
            assert len(files) == 1, [type(c).__name__ for c in cards]
            gutter = tokens.SPACING["md"]
            assert view.transcript.padding == ft.Padding.all(gutter)
            column = width - 2 * gutter
            row = wire(files[0])
            layout = layout_row(row, column)
            card_node = _find(row, files[0].card._i)
            assert layout["overflow"] == 0, layout
            assert layout["sizes"][files[0].card._i] == (320.0, float(column)), layout["sizes"]
            assert card_node.get("expand") is True and card_node.get("expand_loose") is True
            fixed = copy.deepcopy(row)
            node = _find(fixed, files[0].card._i)
            node.pop("expand", None)
            node.pop("expand_loose", None)
            assert layout_row(fixed, column)["overflow"] == pytest.approx(24), "the model no longer reproduces CI"
            plans = [c for c in cards if isinstance(c, JobCard)]
            assert plans and all(_ink_problems(c) == [] for c in plans)
            assert _ink_problems(view.transcript) == []
        finally:
            app.jobs.close()
            await tf._stop(app)

    asyncio.run(scenario())


# ---- the device flows on the host twin -----------------------------------------------------------


@needs_flet
def test_device_smoke_flow_on_the_host_twin_keeps_the_header_in_its_slot(app_env):
    """``tests/test_ui_smoke.py``'s flow as the device runs it (a fresh install: the Welcome is skipped,
    ``first_run=True``) at the CI emulator's 320 dp, through ``UiDriver`` on the host tester. The chat
    header is laid out on every pump the flow makes."""
    import flows
    from host_tester import PyTester

    twin = _load("_glossarion_ui_flows_issue18", Path(__file__).with_name("test_ui_flows.py"))
    tf = _foundations()
    checked: list = []
    problems: dict = {}

    class HeaderCheckingTester(PyTester):
        async def pump(self, duration=None) -> None:
            app = getattr(self.page, "data", None)
            header = getattr(getattr(app, "chat_view", None), "header", None)
            views = list(self.page.views or [])
            if header is not None and views and views[0].appbar is header.wrapper and header.wrapper is not None:
                layout = header_layout(wire(header.wrapper), 320, header)
                checked.append(layout["overflow"])
                if layout["overflow"]:
                    problems[header.subtitle_text] = layout["overflow"]
            await super().pump(duration)

    async def scenario():
        app, tester, driver = await twin._host_driver(tf, {}, first_run=True, width=320, tester_cls=HeaderCheckingTester)
        try:
            pumps: list = []
            real_pump = tester.pump

            async def recording_pump(duration=None):
                pumps.append(duration)
                await real_pump(duration)

            tester.pump = recording_pump
            assert await flows.dismiss_welcome(driver, timeout=30, first_run=True) is True
            await flows.smoke_navigation(driver)
            assert await flows.run_selftest(driver, timeout=30) == "PASS"
            # UiDriver sends Flet's DurationValue as int milliseconds (never a timedelta)
            assert pumps and all(isinstance(p, int) and not isinstance(p, bool) for p in pumps), pumps[:5]
        finally:
            app.jobs.close()
            await tf._stop(app)

    asyncio.run(scenario())
    assert checked, "the header was never laid out during the flow"
    assert problems == {}, f"the chat header overflowed during the device flow: {problems}"


def _dismiss_welcome_calls(path: Path) -> list:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == "dismiss_welcome"]


def test_device_tests_treat_every_run_as_a_fresh_install():
    """Every ``flet test`` device test is a fresh install (``flutter test`` uninstalls after each, the
    conftest uninstalls a leftover first): the device tests wait for the Welcome and skip it."""
    import flows

    files = sorted(TESTS_DIR.glob("test_ui_*.py"))
    assert {f.name for f in files} >= {"test_ui_smoke.py", "test_ui_chat_epub.py"}
    for path in files:
        calls = _dismiss_welcome_calls(path)
        assert calls, f"{path.name} does not dismiss the Welcome"
        for call in calls:
            first_run = {kw.arg: kw.value for kw in call.keywords}.get("first_run")
            assert isinstance(first_run, ast.Constant) and first_run.value is True, (
                f"{path.name}:{call.lineno}: dismiss_welcome without first_run=True settles for the chat home "
                "that the Welcome covers a frame later (the 'dest-library' timeout)")
    # the lower Settings groups are a dozen swipes down a 320x640 emulator
    assert flows.SCROLL_TIMEOUT >= 60


# ---- the `flet test` driver patch on Flet's own template -------------------------------------------

#: flet 1.0.3 build template, ``{{cookiecutter.out_dir}}/integration_test/app_test.dart`` (verbatim,
#: including its missing final newline after ``{% endif %}``)
FLET_103_TEMPLATE = (
    "{% if cookiecutter.test_mode %}import 'package:flet_integration_test/flet_integration_test.dart';\n"
    "import 'package:{{ cookiecutter.project_name }}/main.dart' as app;\n"
    "\n"
    "// Device-mode integration test entry point. The app under test runs on-device\n"
    "// with embedded Python over dart_bridge; a RemoteWidgetTester drives it over a\n"
    "// raw socket connected to the pytest RemoteTester server (FLET_TEST_SERVER_URL).\n"
    "void main() => runFletDeviceTest(appMain: app.main);\n"
    "{% endif %}"
)


def _project_name() -> str:
    """``flet build``'s ``project_name``: the slugified [project] name with '-' -> '_'."""
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python < 3.11
        import tomli as tomllib
    name = tomllib.loads((MOBILE_DIR / "pyproject.toml").read_text(encoding="utf-8"))["project"]["name"]
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-").replace("-", "_")


def _render_driver(test_mode: bool) -> str:
    context = {"test_mode": test_mode, "project_name": _project_name()}
    plain = FLET_103_TEMPLATE.replace("{{ cookiecutter.project_name }}", context["project_name"])
    plain = re.sub(r"\{% if cookiecutter\.test_mode %\}(.*)\{% endif %\}", lambda m: m.group(1) if test_mode else "",
                   plain, flags=re.S)
    if _has("jinja2"):  # what cookiecutter renders with (keep_trailing_newline)
        import jinja2

        rendered = jinja2.Environment(keep_trailing_newline=True).from_string(FLET_103_TEMPLATE).render(cookiecutter=context)
        assert rendered == plain
    return plain


def _device_conftest():
    return _load("_glossarion_device_conftest_issue18", TESTS_DIR / "conftest.py")


def _dart_main_body(text: str) -> list:
    lines = text.replace("\r\n", "\n").split("\n")
    start = lines.index("void main() {")
    end = lines.index("}", start)
    return [line.strip() for line in lines[start + 1:end]]


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_flet_test_driver_patch_on_the_rendered_template(tmp_path, monkeypatch, capsys, newline):
    import android_device
    import driver_patch

    rendered = _render_driver(True)
    assert "import 'package:glossarion/main.dart' as app;" in rendered
    assert rendered.count(driver_patch.DRIVER_LINE) == 1
    driver = driver_patch.driver_path(tmp_path)
    driver.parent.mkdir()
    driver.write_bytes(rendered.replace("\n", newline).encode("utf-8"))

    uninstalls: list = []

    class FakeAdb:  # never reach a real device from a host test
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self, *args, **kwargs):
            uninstalls.append(args)
            return ""

    monkeypatch.setattr(android_device, "Adb", FakeAdb)
    monkeypatch.setattr(android_device, "adb_available", lambda: "adb")
    monkeypatch.setenv("FLET_TEST_PLATFORM", "android")
    monkeypatch.setenv("FLET_TEST_FLUTTER_APP_DIR", str(tmp_path))
    monkeypatch.delenv("GLOSSARION_PACKAGE", raising=False)
    conftest = _device_conftest()

    conftest.pytest_configure(None)  # `flet test android` starts pytest after rendering the driver
    assert "flet test driver: patched" in capsys.readouterr().out
    data = driver.read_bytes()
    assert data.count(b"\r\n") in (0, data.count(b"\n")) and (b"\r\n" in data) == (newline == "\r\n")
    text = data.decode("utf-8").replace("\r\n", "\n")
    lines = text.split("\n")
    # directives first (Dart: imports precede declarations), the template's own lines kept
    assert lines[0] == ("import 'package:flutter_test/flutter_test.dart' "
                        "show LiveTestWidgetsFlutterBinding, WidgetController;")
    original = rendered.split("\n")
    assert lines[1:3] == original[0:2] and lines[4:7] == original[3:6]
    first_code = next(i for i, line in enumerate(lines) if line and not line.startswith(("import ", "//")))
    assert all(not line.startswith("import ") for line in lines[first_code:])
    # one entry point; the flags are set after runFletDeviceTest created the binding, before any test body
    assert text.count("void main()") == 1 and driver_patch.DRIVER_LINE not in text
    assert _dart_main_body(text) == [
        "runFletDeviceTest(appMain: app.main);",
        "LiveTestWidgetsFlutterBinding.instance.shouldPropagateDevicePointerEvents = true;",
        "WidgetController.hitTestWarningShouldBeFatal = true;",
    ]
    assert text.count("{") == text.count("}") and text.count("(") == text.count(")")
    assert driver_patch.MARK in text
    assert uninstalls == [("uninstall", "com.glossarion.app")]

    # a cached test host (`--flutter-test-host`, or the next CI run): already patched, byte-identical
    for _ in range(2):
        conftest.pytest_configure(None)
        assert "flet test driver: already patched" in capsys.readouterr().out
        assert driver.read_bytes() == data
    assert driver_patch.patch_device_driver(tmp_path) is False and driver.read_bytes() == data

    # a Flet template that no longer renders the expected driver fails the run loudly
    driver.write_bytes(_render_driver(False).encode("utf-8"))
    with pytest.raises(pytest.UsageError, match="Flet template changed"):
        conftest.pytest_configure(None)
    driver.unlink()
    with pytest.raises(pytest.UsageError, match="was not generated"):
        conftest.pytest_configure(None)

    # plain pytest (tests_host, or no device): the hook does nothing
    driver.write_bytes(rendered.encode("utf-8"))
    monkeypatch.delenv("FLET_TEST_PLATFORM")
    calls_before = list(uninstalls)
    conftest.pytest_configure(None)
    assert driver.read_bytes() == rendered.encode("utf-8") and uninstalls == calls_before
