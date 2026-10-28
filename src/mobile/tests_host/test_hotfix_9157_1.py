"""v9.15.7 hotfix 1, two owner reports from a phone (2026-10-28):

1. "In the manga translator the include page image request toggle is clipping out of the gui". Flet's
   ``Switch(label=...)`` lays its label out in a Row as wide as its children (Flet 1.0.3 ``switch.dart``),
   so "Include page image in translation requests (visual context)" ran past the screen edge; on the
   emulator (360 dp) "Compress Glossary Prompt (same setting as Glo" and "Safe area (keep text inside the
   bubble's inner are" were cut off too. Every labelled switch of the Manga translator's Settings and Files
   tabs now goes through ``components.switch_row`` (the label is its own expanded, wrapping Text; a tap on
   it toggles the switch), the helper Chat settings rows already used inline (extracted, not copied). The
   switches that share a wrapping Row with a dropdown or a button keep their short built-in labels.
2. "The log Auto scroll triggers even when no log is being output, and the auto scroll itself visibly
   scrolls down beyond the bottom, so it looks like someone spamming scroll down". Reproduced on the
   emulator with the v9.15.7 APK and a fake model (a 30 fps screen recording of a running job's page):
   scroll the Job page up past the Log card and back down and the log box is empty for ~150 ms, shows a
   few lines at its top over blank space, then scrolls on for about a second, with no new log line. The
   page's ListView is lazy, so the Log card was dropped off-screen and its log list built again at its top
   when it came back; Flet's ``auto_scroll`` then jumped to ``maxScrollExtent``, which a lazy list only
   estimates for blocks it has not measured: past the real end, a spring back, more jumps as blocks got
   measured. The log list and the pages that host a LogConsole (Job page, Manga Files tab, Diagnostics)
   now lay out every child (``LAYOUT_ALL_EXTENT`` as ``cache_extent``): the console is never dropped and
   rebuilt by its page, and the end a pin jumps to is measured.
3. Found while checking 2 on the emulator: changing ``auto_scroll`` on a mounted list builds a new Flutter
   Scrollable at offset 0 (Flet 1.0.3 wraps it in its auto-scroll listener only while it is on). The log's
   auto-scroll button threw the log to its first line, and a 300 dp scroll up from the Job page's end threw
   the page to its top (the page turned its pin off). The console keeps ``auto_scroll`` on and holds new
   lines back while paused; the Job page turns its pin on once, when the reader first reaches the end, and
   never off (``test_hotfix2_job_strip_scroll`` now models the new Scrollable and catches the old page).

What only a device shows: the scrolling itself (the emulator check before / after is in the hotfix
notes); these tests pin the layout that removes it.

Run from src/mobile with the mobile venv::

    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_hotfix_9157_1.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
for entry in (str(MOBILE_DIR / "app"), str(MOBILE_DIR.parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

ft = pytest.importorskip("flet")


def _load(name: str, filename: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


mu = _load("_glossarion_hf9157_manga_ui", "test_manga_ui.py")
iso = mu.iso  # every manga path in tmp_path (autouse there, and here)

from glossarion_mobile.ui.components import log_console as lc  # noqa: E402
from glossarion_mobile.ui.components.log_console import LAYOUT_ALL_EXTENT, LogConsole  # noqa: E402
from glossarion_mobile.ui.components.switch_row import switch_row  # noqa: E402

#: switches that sit in a wrapping Row next to a dropdown / button keep a short built-in label
SHORT_LABEL = 20
LONG_LABELS = ("Include page image in translation requests (visual context)",
               "Compress Glossary Prompt (same setting as Glossary Settings)",
               "Safe area (keep text inside the bubble's inner area)",
               "Use loaded/generated glossary for translation",
               "Auto-load the generated glossary for this folder")


def walk(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        for name in ("controls", "content", "trailing", "leading", "title", "subtitle", "actions"):
            child = getattr(node, name, None)
            if isinstance(child, (list, tuple)):
                stack.extend(reversed([c for c in child if isinstance(c, ft.Control)]))
            elif isinstance(child, ft.Control):
                stack.append(child)


def parents(root) -> dict:
    """``{id(child): parent}`` for the tree ``walk`` visits."""
    found = {}
    for node in walk(root):
        for name in ("controls", "content", "trailing", "leading", "title", "subtitle", "actions"):
            child = getattr(node, name, None)
            for c in child if isinstance(child, (list, tuple)) else [child]:
                if isinstance(c, ft.Control):
                    found[id(c)] = node
    return found


def wrapped_label(root, switch) -> str:
    """The text a ``switch_row`` sets beside ``switch`` (asserting the row's shape)."""
    row = parents(root)[id(switch)]
    assert isinstance(row, ft.Row) and row.controls[0] is switch, row
    holder = row.controls[1]
    assert isinstance(holder, ft.Container) and holder.expand and callable(holder.on_click)
    assert isinstance(holder.content, ft.Text) and not holder.content.max_lines and not holder.content.no_wrap
    return holder.content.value


# ==========================================================================
# 1. switch_row
# ==========================================================================


def test_switch_row_wraps_the_label_and_a_tap_on_it_toggles_the_switch():
    events = []
    switch = ft.Switch(value=False, key="k", on_change=lambda e: events.append((e.name, e.control, e.control.value)))
    badge = ft.Text("Inherited from: All chats")
    row = switch_row(switch, LONG_LABELS[0], trailing=[badge])
    assert row.controls[0] is switch and row.controls[-1] is badge and not switch.label
    assert wrapped_label(row, switch) == LONG_LABELS[0]

    row.controls[1].on_click(None)
    assert switch.value is True and events == [("change", switch, True)]
    row.controls[1].on_click(None)
    assert switch.value is False and events[-1] == ("change", switch, False)

    switch.disabled = True
    row.controls[1].on_click(None)
    assert switch.value is False and len(events) == 2  # a disabled switch ignores its label too


def test_switch_row_runs_an_async_handler():
    seen = []

    async def handler(e):
        seen.append(e.control.value)

    async def scenario():
        switch = ft.Switch(value=True, on_change=handler)
        switch_row(switch, "Batch translation").controls[1].on_click(None)
        await asyncio.sleep(0)
        assert seen == [False]

    asyncio.run(scenario())


# ==========================================================================
# 1. the Manga translator: no clipped switch label
# ==========================================================================


def _labelled_switches(root) -> list:
    return [c for c in walk(root) if isinstance(c, ft.Switch) and c.label]


@mu.needs_flet
@pytest.mark.skipif(not mu._has("settings_schema"), reason="settings_schema not importable")
def test_manga_settings_switch_labels_wrap(iso, tmp_path, monkeypatch):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.settings.context import SettingsContext
    from glossarion_mobile.ui.tools.manga.screen import MangaScreen

    monkeypatch.setenv("GLOSSARION_MOBILE", "1")
    monkeypatch.setitem(sys.modules, "manga_models", mu.FakeModels())
    store, schema = mu._settings_store(tmp_path)

    async def scenario():
        _conn, session = mu._tb()._fake_session("android")
        page = session.page
        ctx = mu._ctx(page, store, settings=SettingsContext(page=page, store=store, schema=schema),
                      output_root=str(iso["Output"]))
        screen = MangaScreen(parse_route("/tools/manga?tab=settings"), ctx, session=mu._session(tmp_path, {}))
        mu._mount(page, screen.get_body())
        tab = screen.settings_tab
        tab.did_show()
        root = tab.root
        keyed = mu._by_key(root)
        # the owner's toggle: label beside the switch, wrapping, and its tap writes the setting
        visual = keyed["ms-visual"]
        assert wrapped_label(root, visual) == LONG_LABELS[0]
        before = bool(visual.value)
        parents(root)[id(visual)].controls[1].on_click(None)
        assert visual.value is (not before) and store.get("manga_visual_context_enabled") is (not before)
        keyed = mu._by_key(root)  # a write rebuilds the cards (new generation keys)
        for key, label in (("ms-glossary-compress", LONG_LABELS[1]), ("ms-glossary-on", LONG_LABELS[3]),
                           ("ms-glossary-auto", LONG_LABELS[4]), ("ms-glossary-debug", None),
                           ("ms-ocr-no-thinking", None), ("ms-bubble-detection", None)):
            if key in keyed:  # the OCR / detection rows depend on the chosen provider
                text = wrapped_label(root, keyed[key])
                assert label is None or text == label
        safe_area = [t for t in walk(root) if isinstance(t, ft.Text) and t.value == LONG_LABELS[2]]
        assert safe_area, "the rendering card's safe area switch is not built"
        # nothing long is left as a built-in Switch label anywhere on the tab
        assert [s.label for s in _labelled_switches(root) if len(s.label) > SHORT_LABEL] == []

    asyncio.run(scenario())


@mu.needs_flet
@mu.needs_cores
def test_manga_files_switch_labels_wrap_and_its_sections_stay_laid_out(iso, tmp_path):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.manga.screen import MangaScreen

    async def scenario():
        _conn, session = mu._tb()._fake_session("android")
        page = session.page
        store: dict = {}
        ctx = mu._ctx(page, store, output_root=str(iso["Output"]))
        screen = MangaScreen(parse_route("/tools/manga"), ctx, session=mu._session(tmp_path, store))
        mu._mount(page, screen.get_body())
        tab = screen.files_tab
        root = tab.root
        assert wrapped_label(root, tab.split_switch) == "Split first-level subfolders into groups"
        assert wrapped_label(root, tab.create_cbz_switch) == "Create CBZ at end"
        assert wrapped_label(root, tab.consolidate_switch) == "Auto consolidate images at translation end"
        assert _labelled_switches(root) == []
        # the Run section (the log console's holder) is never dropped off-screen and rebuilt
        assert isinstance(root, ft.ListView) and root.cache_extent == LAYOUT_ALL_EXTENT

    asyncio.run(scenario())


# ==========================================================================
# 2. the log console and its pages lay every child out
# ==========================================================================


@pytest.fixture
def _following():
    saved = dict(lc.SESSION)
    lc.SESSION["following"] = True
    yield
    lc.SESSION.clear()
    lc.SESSION.update(saved)


def test_log_console_lays_out_every_block(_following):
    console = LogConsole()
    view = console.list_view
    assert view.cache_extent == LAYOUT_ALL_EXTENT
    # more than the console can ever hold: every block wrapped to 4 lines of 20 dp
    assert LAYOUT_ALL_EXTENT > console.max_blocks * console.block_lines * 4 * 20
    # following is still Flet's instant client pin, now onto a measured end
    assert view.auto_scroll and view.auto_scroll_animation.duration == 0


def test_log_console_never_changes_auto_scroll_and_holds_lines_while_paused(_following):
    from glossarion_mobile.services.dispatcher import LogLine

    console = LogConsole()
    view = console.list_view
    seen = []

    def check():
        seen.append(view.auto_scroll)

    def lines(*texts, start=[0]):
        out = []
        for text in texts:
            start[0] += 1
            out.append(LogLine(start[0], 0.0, text, "info"))
        return out

    def scroll(pixels, maximum=1000.0):
        console._on_scroll(type("E", (), {"pixels": pixels, "max_scroll_extent": maximum})())

    console.on_lines(lines("one", "two"), 0)
    check()
    console._toggle_follow()  # the auto-scroll button: paused, the view stays put
    console.on_lines(lines("three"), 2)
    check()
    assert [line.text for line in console.pending] == ["three"] and console.pending_gap == 2
    assert console.jump_holder.visible and "three" not in console.list_view.controls[-1].value
    assert console.visible_text().endswith("three")  # Copy / Share still get every line
    console.set_filter("errors")
    console.set_filter("all")  # a filter change keeps the held lines held
    assert [line.text for line in console.pending] == ["three"]
    console.jump_to_end()  # ↓: the held lines (and the dropped-lines note) go in, following again
    check()
    shown = console.list_view.controls[-1].value
    assert not console.pending and shown.endswith("… 2 earlier lines were dropped\nthree")
    assert console.following and lc.SESSION["following"] is False  # ↓ does not change the session choice
    scroll(1000.0)
    scroll(400.0)  # a scroll up pauses this console only; lines wait
    console.on_lines(lines("four"), 0)
    check()
    assert not console.following and [line.text for line in console.pending] == ["four"]
    scroll(995.0)  # back at the end: they go in below and it follows again
    check()
    assert console.following and not console.pending and console.list_view.controls[-1].value.endswith("four")
    assert seen and all(seen), "auto_scroll changed: Flet would rebuild the scroll view at the log's top"


#: the longest built-in Switch label that still fits beside its switch on a 360 dp phone at 100 % text
SWITCH_LABEL_FITS = 32


def test_no_long_built_in_switch_label_anywhere_in_the_app():
    """Owner (2026-10-28, after hotfix 1): every long switch label in the app wraps, not only the manga
    translator's (Keys "Fallback Key Shuffle (skip keys on API delay cooldown)", QA custom "Check all file
    pairs ...", glossary backups, Library review, cloud copy, on-device CJK block, notifications, async
    batch). A label longer than SWITCH_LABEL_FITS goes through switch_row."""
    import ast

    app = MOBILE_DIR / "app" / "glossarion_mobile"
    long_labels = []
    for path in sorted(app.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "Switch"):
                for kw in node.keywords:
                    if (kw.arg == "label" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str)
                            and len(kw.value.value) > SWITCH_LABEL_FITS):
                        long_labels.append((path.relative_to(app).as_posix(), kw.value.value))
    assert long_labels == [], long_labels


def test_no_screen_switches_auto_scroll_on_a_mounted_list():
    """A change of ``auto_scroll`` rebuilds Flutter's Scrollable at offset 0 (the log console, the Job page,
    the Reader's live panel and the chat card's request list all threw their reader to the top). Only two
    assignments are left: the transcript sets it once in its constructor, and the Job page turns its pin on
    once, when the reader first reaches the end."""
    import re

    app = MOBILE_DIR / "app" / "glossarion_mobile"
    found = []
    for path in sorted(app.rglob("*.py")):
        for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\.auto_scroll\s*=(?!=)", text):
                found.append((path.relative_to(app).as_posix(), text.strip()))
    assert found == [("ui/chat/transcript.py", "self.auto_scroll = False"),
                     ("ui/screens/job_detail.py", "page_list.auto_scroll = True")], found


def test_job_page_keeps_its_cards_and_log_laid_out(tmp_path, _following):
    jobs = _load("_glossarion_hf9157_jobs", "test_jobs.py")
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.job_detail import JobDetailScreen

    service, _backend = jobs.make_service(tmp_path)
    try:
        jid = service.submit(jobs.JobSpec("translate", "Book", (jobs.epub(tmp_path),)))
        detail = JobDetailScreen(parse_route(f"/jobs/{jid}"), service=service, copy_handler=lambda text: None)
        body = detail.build_body()
        page_list = detail.page_list
        assert page_list in list(walk(body)) and isinstance(page_list, ft.ListView)
        assert page_list.cache_extent == LAYOUT_ALL_EXTENT
        assert detail.console.list_view.cache_extent == LAYOUT_ALL_EXTENT
        # the page still pins to its end only while the reader is there (hotfix 2), instantly
        assert page_list.auto_scroll is False and page_list.auto_scroll_animation.duration == 0
    finally:
        service.close()


def test_diagnostics_page_keeps_its_log_laid_out(tmp_path):
    lt = _load("_glossarion_hf9157_log_text", "test_log_text.py")
    body = lt._diagnostics(tmp_path).build_body()
    lists = [c for c in walk(body) if isinstance(c, ft.ListView) and getattr(c, "expand", None)]
    assert lists and lists[0].cache_extent == LAYOUT_ALL_EXTENT
