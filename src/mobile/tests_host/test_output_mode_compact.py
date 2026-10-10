"""Owner report (2026-10-07): "the output mode needs to be more compact, since it is reducing the
type to text area by creating an empty line beneath it".

The composer's output mode no longer has a line of its own (UI_SPEC §2.3 item 3): it sits in the
action row after ＋. Chat columns under 600 dp (or >= 160% text) get one chip with the active
mode's emoji and a ▾ that opens a menu of the six modes plus "Options for <Mode>…"; from 600 dp
the six toggles sit inline; labelled toggles once they fit. The ＋ sheet keeps its six toggles.

Run from src/mobile (Flet venv or the 3.13 review venv):
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_output_mode_compact.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from glossarion_mobile.ui import responsive  # noqa: E402
from glossarion_mobile.ui.chat import output_modes  # noqa: E402

_TF_SPEC = importlib.util.spec_from_file_location(
    "_glossarion_tf_helpers_outmode", Path(__file__).with_name("test_ui_foundations.py")
)
TF = importlib.util.module_from_spec(_TF_SPEC)
_TF_SPEC.loader.exec_module(TF)
storage = TF.storage
app_env = TF.app_env
needs_flet = TF.needs_flet

PHONE_WIDTHS = (320, 360, 393, 412)
SCALES = (1.0, 1.3, 2.0)


# ==========================================================================
# the width rule and the action-row estimate (pure)
# ==========================================================================


def test_output_row_style_is_sized_from_the_chat_column():
    for width in (0, 320, 412, 599):
        assert responsive.output_row_style(width) == "chip"
    assert responsive.output_row_style(600) == "icons" and responsive.output_row_style(760) == "icons"
    assert responsive.output_row_style(860) == "full"
    assert responsive.output_row_style(860, text_scale=1.3) == "icons"  # labels need ~880 dp at 130%
    assert responsive.output_row_style(860, text_scale=1.6) == "chip"
    # window -> chat column: the tablet sidebar (300 / 320 dp) and the 760 / 860 dp caps count
    cases = {320: (320, "chip"), 412: (412, "chip"), 599: (599, "chip"), 600: (600, "icons"),
             899: (760, "icons"), 900: (600, "icons"), 1000: (700, "icons"), 1150: (850, "full"),
             1300: (860, "full")}
    for window, (column, style) in cases.items():
        layout = responsive.layout_for(window)
        assert (layout.chat_width, layout.output_row) == (column, style), window
    assert responsive.layout_for(1300, text_scale=1.6).output_row == "chip"


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("width", PHONE_WIDTHS)
def test_phone_action_row_fits_on_one_line(width, scale):
    layout = responsive.layout_for(width, scale)
    assert layout.output_row == "chip"
    room = layout.chat_width - responsive.COMPOSER_INSET
    used = responsive.action_row_width(layout.output_row, scale)
    assert used <= room, (width, scale, used, room)
    # the chip costs one 48 dp target (+ its ▾), never the six toggles' 298 dp
    assert responsive.mode_control_width("chip", scale) < 80
    assert responsive.mode_control_width("icons", scale) >= 6 * 48
    if scale >= 1.6:  # §7.5: the token hint hides, so ＋ · chip · Send leave room for the pills
        assert used == responsive.action_row_width("chip", scale, token_hint="")


@pytest.mark.parametrize("scale", (0.85, 1.0, 1.15, 1.3, 1.6, 2.0))
def test_every_window_width_gets_a_style_that_fits(scale):
    for window in range(300, 2001, 10):
        layout = responsive.layout_for(window, scale)
        room = layout.chat_width - responsive.COMPOSER_INSET
        assert responsive.action_row_width(layout.output_row, scale) <= room, (window, scale, layout.output_row)


def test_output_mode_strings():
    assert output_modes.mode_tooltip("vision") == "Output mode: Vision"
    assert output_modes.mode_tooltip("vision", True) == "Output mode: Vision · auto"
    assert output_modes.chip_semantics_label("refine") == "Output mode: Refine"
    assert output_modes.chip_semantics_label("vision", True) == "Output mode: Vision, automatic"


# ==========================================================================
# the composer controls
# ==========================================================================


@needs_flet
def test_phone_composer_puts_the_mode_chip_in_the_action_row():
    import flet as ft

    from glossarion_mobile.ui.chat.composer import Composer
    from glossarion_mobile.ui.chat.output_mode_row import OutputModeRow

    composer = Composer(row_style=responsive.layout_for(393).output_row)
    row = composer.output_row
    # chips row · [text field + expand] · action row: no standalone output row any more
    assert composer.content.controls[0] is composer.chips_row and composer.content.controls[2] is composer.action_row
    assert len(composer.content.controls) == 3
    assert not any(isinstance(c, OutputModeRow) for c in composer.content.controls)
    text_row = composer.content.controls[1]
    assert text_row.controls == [composer.text_field, composer.expand_button]
    plus, mode, pills, hint, send = composer.action_row.controls
    assert (plus, mode, hint, send) == (composer.plus_button, row, composer.token_hint, composer.send_button)
    assert pills.content is composer.pills_row and pills.expand
    assert composer.action_row.height == 40 and not composer.action_row.wrap
    # the chip: active emoji + ▾ in a 48 dp target, tooltip + button semantics
    assert row.inline and row.effective_style == "chip" and row.tight and row.scroll is None
    assert row.controls == [row.chip_semantics] and row.toggles == {}
    assert isinstance(row.chip, ft.PopupMenuButton) and row.chip.content.height == 48
    assert row.chip_emoji.value == "📝" and row.chip.tooltip == "Output mode: Text"
    assert row.chip_semantics.label == "Output mode: Text" and row.chip_semantics.button
    # the menu: six modes (emoji + label, a check on the active one), a divider, "Options for <Mode>…"
    items = row.chip.items
    assert [i.content for i in items[:6]] == ["📝  Text", "👁️  Vision", "🖼️  Image", "🎬  Video", "🔊  Audio", "✨  Refine"]
    assert [i.checked for i in items[:6]] == [True, False, False, False, False, False]
    assert items[6].content is None and items[7] is row.options_item
    assert row.options_item.content == "Options for Text…"


@needs_flet
def test_chip_menu_selects_a_mode_and_opens_the_options_sheet():
    from glossarion_mobile.ui.chat.composer import Composer

    opened = []
    composer = Composer(on_open_mode_options=opened.append)
    row = composer.output_row
    row.menu_items["vision"].on_click(None)
    assert composer.mode_signal.value.mode == "vision" and not composer.mode_signal.value.automatic
    assert row.chip_emoji.value == "👁️" and row.chip.tooltip == "Output mode: Vision"
    assert [i.checked for i in row.menu_items.values()] == [False, True, False, False, False, False]
    assert row.options_item.content == "Options for Vision…" and opened == []
    assert row.select("vision") == "unchanged" and opened == []  # the checked mode: nothing to do
    row.options_item.on_click(None)
    assert opened == ["vision"]
    # the automatic Vision switch: "· auto" in the tooltip, ", automatic" spoken, a dot on the chip
    row.mode_signal.set(row.state.select("text").attachment_changed("page.png"))
    row._sync()
    assert row.chip.tooltip == "Output mode: Vision · auto"
    assert row.chip_semantics.label == "Output mode: Vision, automatic" and row.chip_visual.badge is not None
    row.menu_items["audio"].on_click(None)  # a manual pick clears it
    assert row.chip_visual.badge is None and row.chip_emoji.value == "🔊"


@needs_flet
def test_inline_toggles_from_600_dp_and_labels_when_they_fit():
    import flet as ft

    from glossarion_mobile.ui.chat.composer import Composer

    opened = []
    composer = Composer(row_style=responsive.layout_for(700).output_row, on_open_mode_options=opened.append)
    row = composer.output_row
    assert row.effective_style == "icons" and row.chip is None
    assert row.label_text not in [getattr(c, "content", None) for c in row.controls]  # no "Output:" label inline
    assert len(row.controls) == 6 and all(isinstance(t, ft.IconButton) for t in row.toggles.values())
    assert all(t.size_constraints.min_width == 48 and t.size_constraints.min_height == 48 for t in row.toggles.values())
    assert row.tap("image") == "selected" and row.toggles["image"].selected
    assert row.tap("image") == "options" and opened == ["image"]  # the active toggle opens its options
    row.mode_signal.set(row.state.attachment_changed("page.webp"))
    row._sync()
    assert row.toggles["vision"].badge is not None and row.toggles["vision"].tooltip == "Output mode: Vision · auto"
    assert composer.set_row_style(responsive.layout_for(1300).output_row)
    assert row.effective_style == "full" and len(row.controls) == 6  # still no "Output:" label
    assert all(isinstance(t, ft.Container) and t.height == 48 for t in row.toggles.values())
    assert row.toggles["text"].content.content.value == "📝 Text"
    assert row.toggles["vision"].content.bgcolor is not None and row.toggles["vision"].content.badge is not None
    assert composer.set_row_style("chip") and row.chip is not None and row.chip_emoji.value == "👁️"
    assert not composer.set_row_style("chip")


@needs_flet
def test_plus_sheet_keeps_its_six_toggles():
    from glossarion_mobile.ui.chat.composer import Composer
    from glossarion_mobile.ui.sheets.plus_sheet import PlusSheet

    composer = Composer(row_style="chip")
    plus = PlusSheet(mode_signal=composer.mode_signal, row_style="chip")
    sheet_row = plus.output_row
    assert not sheet_row.inline and sheet_row.effective_style == "icons" and sheet_row.chip is None
    assert len(sheet_row.toggles) == 6
    for style, labelled in (("label", True), ("full", True), ("icons", False)):
        sheet_row.set_style(style)
        assert (sheet_row.label_text in [getattr(c, "content", None) for c in sheet_row.controls]) is labelled, style


@needs_flet
def test_token_hint_hides_at_compact_text_scale():
    from glossarion_mobile.ui.chat.composer import Composer

    composer = Composer()
    composer.set_token_hint("≈1.2k tok")
    assert composer.token_hint.visible and composer.token_hint.no_wrap
    composer.set_compact_text(True)
    assert not composer.token_hint.visible and composer.text_field.max_lines == 4
    composer.set_compact_text(False)
    assert composer.token_hint.visible and composer.token_hint.value == "≈1.2k tok"
    composer.clear()
    composer.set_compact_text(False)
    assert not composer.token_hint.visible and composer.token_hint.value == ""


# ==========================================================================
# on the app shell: phone layout, the ＋ sheet row follows the chip, Options for <Mode>…
# ==========================================================================


@needs_flet
def test_phone_chip_and_plus_sheet_stay_in_sync_on_the_app(app_env):
    async def scenario():
        _m, _conn, _session, page, app = await TF._start("android", width=393)
        try:
            view = app.chat_view
            composer = view.composer
            row = composer.output_row
            assert row.effective_style == "chip" and row in composer.action_row.controls
            assert row not in composer.content.controls
            plus = view.open_plus_sheet()
            sheet_row = plus.output_row
            assert sheet_row.effective_style == "icons" and len(sheet_row.toggles) == 6
            row.menu_items["audio"].on_click(None)
            assert app.state.output_mode.value.mode == "audio"
            assert sheet_row.toggles["audio"].selected and not sheet_row.toggles["text"].selected
            sheet_row.tap("image")  # and the other way round
            assert row.chip_emoji.value == "🖼️" and row.menu_items["image"].checked
            assert row.options_item.content == "Options for Image…"
            plus.close()
            row.options_item.on_click(None)
            assert view.mode_sheet is not None and view.mode_sheet.mode == "image"
            view.mode_sheet.close()
            row.menu_items["text"].on_click(None)
        finally:
            await TF._stop(app)

    asyncio.run(scenario())
