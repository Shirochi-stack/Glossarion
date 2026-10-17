"""Owner report: Book page (Progress manager) › Output: a large light-grey rectangle fills the screen under
"Cloud & sharing".

Root cause: with no cloud destination (every install until one is set up, and every session without the
cloud service) the section's first line was ``Row([Text(expand=True), TextButton("Set up")], wrap=True)``.
Flet 1.0.3 marks every Row as a flex host (``Row.init``: ``_internals["host_expanded"] = True``), so the
Flutter client wraps the ``expand`` child in ``Expanded`` even when ``wrap=True`` turns the Row into a
``Wrap``. ``Expanded`` needs a ``Flex`` parent: mounting it under a ``Wrap`` throws ("Incorrect use of
ParentDataWidget"; a failed ``FlexParentData`` cast in release builds) and Flutter replaces the subtree
with its ErrorWidget - a grey ``RenderErrorBox`` that, inside the tab's ListView (unbounded height), grows
to fill the rest of the screen.

The test models the constraint rules of Flet 1.0.3 + Flutter that end in that ErrorWidget (or the debug
"non-zero flex but unbounded" / "Vertical viewport was given unbounded height" errors) and walks the real
control trees:

* the Output tab in every Cloud & sharing state (no cloud service, no destination, a destination with
  per-file lines and saved links, a lost destination), and the pre-fix row, which the model must reject;
* every Progress manager tab (Overview, Chapters, Glossary, Output) of a real fixture book on the fake Flet
  session, so the same pattern elsewhere fails here too.

Real data is never touched (the fixtures point every data directory into ``tmp_path``).

Run from src/mobile (mobile venv):
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_progress_layout_constraints.py
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any

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


pytestmark = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")


def _load(name: str, file: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # its dataclasses look their module up
    spec.loader.exec_module(module)
    return module


if _has("flet") and _has("msgpack"):
    import flet as ft

    U10 = _load("_glossarion_u10_helpers_layout", "test_u10_ui.py")
    BPW = _load("_glossarion_bpw_helpers_layout", "test_book_page_workspace.py")
    env = BPW.env  # the fixture Library (organized book with a real workspace)


# ---- the constraint model -----------------------------------------------------------------------------


def _children(control: Any) -> list:
    """Every child control of a Flet 1.0.3 control (its dataclass fields holding controls)."""
    out: list = []
    if not dataclasses.is_dataclass(control):
        return out
    for f in dataclasses.fields(control):
        if f.name in ("parent", "page"):
            continue
        value = getattr(control, f.name, None)
        items = value if isinstance(value, (list, tuple)) else [value]
        out.extend(item for item in items if isinstance(item, ft.BaseControl))
    return out


def _visible(control: Any) -> bool:
    return getattr(control, "visible", True) is not False


def _flex(control: Any) -> bool:
    expand = getattr(control, "expand", None)
    return bool(expand) and expand is not False


def _name(control: Any) -> str:
    key = getattr(control, "key", None)
    return f"{type(control).__name__}" + (f"[{key}]" if key is not None else "")


def layout_errors(control: Any, *, bounded_w: bool = True, bounded_h: bool = True, path: str = "") -> list:
    """Layout errors Flutter raises for this Flet tree under the given incoming constraints.

    * Flet 1.0.3 ``_expandable`` wraps an ``expand`` child of a Row / Column / View (``host_expanded``) in
      ``Expanded``; under a ``Wrap`` (``Row(wrap=True)``) that throws - the grey ErrorWidget.
    * A flex child along an unbounded main axis (``scroll`` set, or the Row/Column itself unbounded there):
      "RenderFlex children have non-zero flex but incoming constraints are unbounded".
    * A vertical ListView / GridView without a height under unbounded height (and horizontal ones under
      unbounded width): "viewport was given unbounded height/width".
    Children of a Row get unbounded width unless they expand; a Wrap's children get unbounded height;
    a ListView's children are unbounded along its axis; ``width`` / ``height`` bound that axis again.
    """
    errors: list = []
    if control is None or not _visible(control):
        return errors
    here = f"{path}/{_name(control)}"
    if isinstance(getattr(control, "width", None), (int, float)):
        bounded_w = True
    if isinstance(getattr(control, "height", None), (int, float)):
        bounded_h = True
    kids = [c for c in _children(control) if _visible(c)]
    if isinstance(control, (ft.Row, ft.Column)):
        horizontal = isinstance(control, ft.Row)
        wrap = bool(getattr(control, "wrap", False)) if horizontal else False
        scroll = getattr(control, "scroll", None) is not None
        main_bounded = (bounded_w if horizontal else bounded_h) and not scroll
        cross_bounded = bounded_h if horizontal else bounded_w
        for child in kids:
            flex = _flex(child)
            if wrap and flex:
                errors.append(f"{here}: expand child {_name(child)} inside Row(wrap=True) (Expanded under Wrap)")
                flex = False
            elif flex and not main_bounded:
                errors.append(f"{here}: expand child {_name(child)} along an unbounded main axis")
            child_main = bool(flex) or (wrap and horizontal and bounded_w)
            if horizontal:
                errors += layout_errors(child, bounded_w=child_main, bounded_h=False if wrap else cross_bounded,
                                        path=here)
            else:
                errors += layout_errors(child, bounded_w=cross_bounded, bounded_h=child_main, path=here)
        return errors
    if isinstance(control, (ft.ListView, ft.GridView)):
        horizontal = bool(getattr(control, "horizontal", False))
        if not (bounded_w if horizontal else bounded_h):
            errors.append(f"{here}: {'horizontal' if horizontal else 'vertical'} scroll view with unbounded "
                          f"{'width' if horizontal else 'height'}")
        for child in kids:
            errors += layout_errors(child, bounded_w=bounded_w and not horizontal,
                                    bounded_h=bounded_h and horizontal, path=here)
        return errors
    for child in kids:
        errors += layout_errors(child, bounded_w=bounded_w, bounded_h=bounded_h, path=here)
    return errors


def _count(control: Any) -> int:
    return 1 + sum(_count(child) for child in _children(control) if _visible(child))


def test_the_model_rejects_the_pre_fix_cloud_row_and_accepts_the_fix():
    pre_fix = ft.Row([ft.Text("No cloud destination yet", expand=True), ft.TextButton(content="Set up")], wrap=True)
    fixed = ft.Row([ft.Text("No cloud destination yet", expand=True), ft.TextButton(content="Set up")], spacing=8)
    in_list = lambda row: ft.ListView([ft.Column([row], spacing=4)], expand=True)  # noqa: E731
    assert any("Row(wrap=True)" in e for e in layout_errors(in_list(pre_fix)))
    assert layout_errors(in_list(fixed)) == []
    # the other errors of the model
    assert layout_errors(ft.ListView([ft.Column([ft.Text("x", expand=True)])]))  # flex under unbounded height
    assert layout_errors(ft.ListView([ft.ListView([ft.Text("x")])]))  # nested viewport without a height
    assert layout_errors(ft.ListView([ft.Container(content=ft.ListView([ft.Text("x")]), height=80)])) == []
    assert layout_errors(ft.Row([ft.Row([ft.Text("x", expand=True)])]))  # flex row as a non-flex Row child


# ---- the Output tab in every Cloud & sharing state ------------------------------------------------------


def _states(tmp_path: Path) -> list:
    """(name, tab) for each Cloud & sharing state, loaded through ``reload`` like the Book page does."""

    async def build() -> list:
        out = []
        tab, _e = U10._output_tab(tmp_path / "noservice", extras={})
        await tab.reload()
        out.append(("no cloud service", tab))
        tab, _e = U10._output_tab(tmp_path / "nodest", cloud=U10.FakeCloud(), shares=U10.FakeShares())
        await tab.reload()
        out.append(("no destination", tab))
        cloud, shares = U10.FakeCloud(), U10.FakeShares()
        shares.enabled.update(gofile=True)
        shares.consented.update(gofile=True)
        cloud.dest = U10.Dest("folder-abc", "folder", label="Drive › Glossarion", provider_label="Drive")
        tab, e = U10._output_tab(tmp_path / "dest", cloud=cloud, shares=shares, page=U10.FakePage())
        identity = os.path.abspath(e.workspace)
        cloud.files[identity] = {
            "epub": {"status": "ok", "name": "Novel.epub", "synced_at": time.time(), "source": e.epub_new},
            "pdf": {"status": "needs_pick", "name": ""},
        }
        shares.links.append(shares._link("gofile", e.epub_new, identity, "https://gofile.example/d/1#k"))
        await tab.reload()
        out.append(("destination, file lines and a saved link", tab))
        cloud.dest = U10.Dest("folder-abc", "folder", label="Drive › Glossarion", needs_relink="revoked")
        await tab.refresh_cloud()
        out.append(("lost destination", tab))
        return out

    return asyncio.run(build())


def test_cloud_and_sharing_lays_out_in_every_state(tmp_path):
    for name, tab in _states(tmp_path):
        # the tab body sits in the Book page's TabBarView: bounded both ways
        assert layout_errors(tab.list) == [], name
        assert tab.cloud_column.controls, name


def test_no_destination_row_keeps_its_text_and_set_up_button(tmp_path):
    async def scenario():
        tab, _e = U10._output_tab(tmp_path, cloud=U10.FakeCloud(), shares=U10.FakeShares())
        await tab.reload()
        row = U10.find_key(tab.cloud_column, "out-cloud-dest")
        assert isinstance(row, ft.Row) and not row.wrap
        assert "No cloud destination yet" in U10.texts(row)
        assert U10.find_key(row, "out-cloud-setup") is not None

    asyncio.run(scenario())


# ---- every Progress manager tab of a real book -------------------------------------------------------------


def test_every_progress_manager_tab_lays_out(env):
    service = env["service"]

    async def scenario():
        await service.refresh()
        row = next(b for b in service.snapshot.in_progress + service.snapshot.completed
                   if b.get("output_folder") and BPW._norm(b["output_folder"]) == BPW._norm(env["ws"]))
        screen, _ctx, _page = await BPW._open(env, service.bid_for(row), "output", files=BPW.FakeFiles())
        await screen.output.reload()
        if screen.glossary is None:
            await screen.reload_glossary()
        for name, body in screen.tab_bodies.items():
            assert layout_errors(body) == [], name
            assert _count(body) > 5, name  # the walk reached the tab's content
        # the whole page as mounted (strip + tabs) as well
        assert layout_errors(screen.get_body()) == []
        assert screen.output.cloud_column.controls

    asyncio.run(scenario())
