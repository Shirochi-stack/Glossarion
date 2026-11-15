"""Sheets are only as tall as their content (owner 2026-11-15: "update all these info menu to not use up the
entire screen space. Same for the add relations menu").

Flet 1.0.3 lays a scrolling Column out at least as tall as its bounded parent, so a sheet whose body scrolled
opened at full height with empty space under its text. ``components.sheet``: ``scroll_sheet`` (every
``glossary.common.sheet``: the relations character sheet, Add relation, the edit forms) builds a measured body
(``fitted_box``); ``InfoSheet`` and ``ActionSheet`` keep a compact body when it surely fits and use the
measured one otherwise (``compact_or_fitted``).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import flet as ft  # noqa: E402

from glossarion_mobile.ui.components import sheet as sheet_module  # noqa: E402

#: the owner's phone (a wide foldable screen, about 600 dp) and a common narrow one
WIDE = (600.0, 960.0)
NARROW = (360.0, 640.0)
HELP = ("Balanced/Full extraction asks the AI who is what to whom (Father, Rival, Best Friend...) and records it per "
        "chapter in a *_relations_tracker.json file beside the glossary. The column is never saved in the glossary "
        "itself.\nWhen translating, each character's glossary line gets the relations known up to that chapter.\n"
        "Needs the {relations_instruction} line in the glossary prompt.\n\nConfig key: glossary_enable_relations_tracker"
        "\n\nEnvironment: GLOSSARY_ENABLE_RELATIONS_TRACKER\n\nDefault: On (Off while an on-device model is selected)")


def _page(size):
    shown: list = []
    return types.SimpleNamespace(width=size[0], height=size[1], shown=shown, show_dialog=shown.append)


def _body(dialog):
    """The control inside the sheet's frame (SafeArea > padding Container > body)."""
    return sheet_module.sheet_body(dialog).content.content


def _measure(box, height, screen=WIDE, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setattr(sheet_module, "_page_size", lambda page: screen)
        monkeypatch.setattr(ft.Container, "page", types.SimpleNamespace(), raising=False)
    inner = box.content.controls[0]
    inner.on_size_change(types.SimpleNamespace(width=screen[0], height=height))


def test_scroll_sheet_is_as_tall_as_its_rows(monkeypatch):
    rows = [ft.TextField(label=f"Field {i}") for i in range(5)]
    add = ft.FilledButton(content="Add")
    dialog = sheet_module.scroll_sheet("Add relation", rows, actions=[ft.TextButton(content="Cancel"), add])
    box = _body(dialog)
    assert isinstance(box, ft.Container) and box.height is None
    scroller = box.content
    assert scroller.scroll == ft.ScrollMode.AUTO  # past the screen (or above the keyboard) it scrolls
    inner = scroller.controls[0]
    assert inner.scroll is None and inner.controls[-1].controls[-1] is add  # the buttons scroll with the fields
    _measure(box, 420.0, monkeypatch=monkeypatch)
    assert box.height == 420.0  # not the whole screen
    _measure(box, 470.0)  # a suggestion row appeared: it grows
    assert box.height == 470.0
    _measure(box, 2000.0)  # longer than the screen: the full sheet, scrolling
    assert box.height is None
    # a body that surely fits stays a plain column (never measured)
    plain = _body(sheet_module.scroll_sheet("Short", [ft.Text("a")], scroll=False))
    assert isinstance(plain, ft.Column) and plain.scroll is None


def test_glossary_sheet_helper_builds_the_measured_body():
    from glossarion_mobile.ui.glossary.common import sheet

    dialog = sheet("Drex", [ft.Text("No relations recorded for this character up to chapter 3.")],
                   actions=[ft.FilledButton(content="Add relation")])
    assert isinstance(_body(dialog), ft.Container) and _body(dialog).content.scroll is not None


def test_info_sheet_long_help_is_measured_short_note_compact():
    from glossarion_mobile.ui.components.info_sheet import InfoSheet

    # the setting help of the owner's screenshot: too long to surely fit at 200 % text, so it was full screen
    page = _page(WIDE)
    info = InfoSheet(title="Enable Relations Tracker", body=HELP, actions=[ft.TextButton(content="Reset to default")])
    info.show(page)
    assert page.shown == [info.dialog]
    box = _body(info.dialog)
    assert isinstance(box, ft.Container) and box.content.controls[0] is info.column
    assert info.column.scroll is None and box.content.scroll == ft.ScrollMode.AUTO
    # until the first measure: the 100 % text estimate, well under the screen
    assert 120 <= box.height < WIDE[1] - sheet_module.SHEET_CHROME
    # a short note keeps the compact body (no measuring, no scroll)
    short = InfoSheet(title="Note", body="Saved.")
    short.show(_page(WIDE))
    assert _body(short.dialog) is short.column and short.column.scroll is None
    # shown again on a small phone: measured there too
    info.show(_page(NARROW))
    assert isinstance(_body(info.dialog), ft.Container)


def test_action_sheet_long_menu_is_measured():
    from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet

    long_menu = ActionSheet([ActionItem(f"Action number {i}") for i in range(14)], title="Chapter 12")
    long_menu.show(_page(NARROW))
    box = _body(long_menu.dialog)
    assert isinstance(box, ft.Container) and box.content.controls[0] is long_menu.body
    short = ActionSheet([ActionItem("Open"), ActionItem("Delete", destructive=True)])
    short.show(_page(WIDE))
    assert _body(short.dialog) is short.body and short.body.scroll is None
    tablet = ActionSheet([ActionItem(f"Action {i}") for i in range(30)], tablet=True)
    tablet.show(_page(NARROW))  # the tablet dialog keeps its own scrolling column
    assert tablet.body.scroll == ft.ScrollMode.AUTO


def test_compact_or_fitted_reuses_one_column():
    column = ft.Column([ft.Text("x" * 4000)], tight=True)
    fitted = sheet_module.compact_or_fitted(column, _page(NARROW), 3000.0)
    assert isinstance(fitted, ft.Container) and column.on_size_change is not None
    compact = sheet_module.compact_or_fitted(column, _page(WIDE), 100.0)
    assert compact is column and column.on_size_change is None and column.scroll is None
