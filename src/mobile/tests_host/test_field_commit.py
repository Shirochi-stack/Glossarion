"""Host tests: value fields commit on Done *and* on tapping away (``ui.components.field_commit``).

Owner: "why do all input fields require me to click done. Shouldn't clicking away count as well".
On Android / iOS a touch outside a focused field does not unfocus it, so the field's
``on_tap_outside`` commits too; Done followed by its blur commits once; an unchanged value writes
nothing; an invalid value shows the same error as on Done.

Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore tests_host/test_field_commit.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")


class _Store:
    """A stored number with the field's own validation (like the Keys rotation field)."""

    def __init__(self, field, value=5):
        self.field = field
        self.value = value
        self.writes: list = []

    def commit(self, text):
        try:
            number = int(text.strip())
            if number < 1:
                raise ValueError
        except ValueError:
            self.field.error = "Enter a whole number ≥ 1"
            return None
        self.field.error = None
        self.value = number
        self.writes.append(number)
        return number


def _wired(value="5"):
    import flet as ft

    from glossarion_mobile.ui.components.field_commit import commit_on_blur

    field = ft.TextField(value=value)
    store = _Store(field)
    commit_on_blur(field, store.commit, current=lambda: store.value)
    return field, store


@needs_flet
def test_tapping_away_commits_like_done():
    field, store = _wired()
    field.on_focus(None)
    field.value = "7"
    field.on_tap_outside(None)  # a touch outside the field (no blur on Android / iOS)
    assert store.value == 7 and store.writes == [7]
    field.on_focus(None)
    field.value = "8"
    field.on_blur(None)  # the focus moved to another field
    assert store.writes == [7, 8]


@needs_flet
def test_done_then_blur_commits_once():
    field, store = _wired()
    field.on_focus(None)
    field.value = "9"
    field.on_submit(None)
    field.on_blur(None)  # Done closes the keyboard: its blur follows
    field.on_tap_outside(None)
    assert store.writes == [9]
    # a new edit of the same field commits again
    field.on_focus(None)
    field.value = "10"
    field.on_tap_outside(None)
    assert store.writes == [9, 10]


@needs_flet
def test_unchanged_value_writes_nothing():
    field, store = _wired()
    field.on_focus(None)
    field.on_tap_outside(None)
    field.on_blur(None)
    field.value = " 5 "
    field.on_submit(None)
    assert store.writes == []


@needs_flet
def test_invalid_value_shows_the_error_and_keeps_the_stored_value():
    field, store = _wired()
    field.on_focus(None)
    field.value = "0"
    field.on_tap_outside(None)
    assert field.error == "Enter a whole number ≥ 1" and store.value == 5 and store.writes == []
    # Done then blur: one validation, the error stays
    field.value = "abc"
    field.on_submit(None)
    field.error = "shown once"
    field.on_blur(None)
    assert field.error == "shown once"
    # typing the stored value back clears the error (an unchanged value with an error still commits)
    field.error = "Enter a whole number ≥ 1"
    field.value = "5"
    field.on_tap_outside(None)
    assert field.error is None and store.value == 5


@needs_flet
def test_the_fields_own_focus_handlers_keep_running():
    import flet as ft

    from glossarion_mobile.ui.components.field_commit import commit_on_blur

    seen: list = []
    field = ft.TextField(value="a", on_focus=lambda e: seen.append("focus"), on_blur=lambda e: seen.append("blur"))
    commit_on_blur(field, lambda text: seen.append(("commit", text)))
    field.on_focus(None)
    field.value = "b"
    field.on_blur(None)
    assert seen == ["focus", "blur", ("commit", "b")]


@needs_flet
def test_an_async_commit_runs_as_a_task():
    import flet as ft

    from glossarion_mobile.ui.components.field_commit import commit_on_blur

    done: list = []

    async def commit(text):
        done.append(text)

    async def scenario():
        field = commit_on_blur(ft.TextField(value=""), commit)
        field.value = "x"
        await field.on_tap_outside(None)

    asyncio.run(scenario())
    assert done == ["x"]


@needs_flet
def test_box_move_fields_place_the_box_on_tap_away():
    from glossarion_mobile.ui.tools.manga.box_sheet import BoxPanel

    placed: list = []
    box = {"x": 10, "y": 20, "width": 30, "height": 40, "ocr_text": "hi"}
    panel = BoxPanel(box, 0, on_action=lambda *a: None, on_geometry=lambda i, g: placed.append((i, g)), tab="move")
    x = panel.geometry_fields["x"]
    x.on_focus(None)
    x.on_tap_outside(None)  # nothing typed: the box stays
    assert placed == []
    x.value = "15"
    x.on_tap_outside(None)
    x.on_submit(None)
    x.on_blur(None)
    assert placed == [(0, {"x": 15.0, "y": 20.0, "width": 30.0, "height": 40.0})]
    panel.set_geometry({**box, "x": 15})
    x.on_focus(None)
    x.on_blur(None)  # the box is there now: no second move
    assert len(placed) == 1
    # an unfinished number places nothing (as Apply)
    x.value = "1a"
    x.on_tap_outside(None)
    assert len(placed) == 1


class _FakeOAuth:
    def __init__(self):
        self.pushed: list = []

    def set_gemini_project(self, project_id, account_id):
        self.pushed.append((project_id, account_id))


@needs_flet
def test_typed_gcp_project_is_set_on_tap_away():
    from glossarion_mobile.ui.screens.accounts import GcpProjectPicker

    config: dict = {"authgem_project": "p-old"}
    oauth = _FakeOAuth()

    async def io(fn, *args):
        return fn(*args)

    picker = GcpProjectPicker(oauth, config_get=config.get, config_set=config.update, io=io, slot=lambda: 1)
    picker.build()
    picker.field.on_focus(None)
    picker.field.on_tap_outside(None)  # blank: nothing set
    assert config["authgem_project"] == "p-old" and oauth.pushed == []
    picker.field.value = "  p-new "
    picker.field.on_tap_outside(None)
    picker.field.on_submit(None)
    picker.field.on_blur(None)
    assert config["authgem_project"] == "p-new" and oauth.pushed == [("p-new", 1)]
    assert picker.dropdown.value == "p-new"
