"""Mobile batch m5 (owner, 2026-11-16): five Book page / chat sheet requests.

1. Chapters: no "📁 <folder>" chip and no mode badge (the band above the tabs has both); Run translation and
   Run QA scan in one row sharing the width; Files stays in the Chapters ⋯ menu and Overview's icon row.
2. Both lists always show special files, marked "Special file (Skipped)" (the Skipped status / chip group); no
   "Show special files" filter item on mobile; skipped rows are not counted in Total.
3. The Glossary list follows the extractor's own special-file decision built from the user's config (never the
   UI process's environment), protect-interior rule included; the files it skips are listed at their spine
   position with the mark of request 2, and have no progress actions.
4. Overview: no METADATA section; publisher (only when set) and ✏️ Edit sit beside the cover with the title,
   author and chips.
5. The chat card's Glossary sheet opens as a short preview on its title and Glossary mode tile (the ＋ sheet's
   ``drag_sheet.preview_sheet``), the rest under a swipe up; without the extension, the ordinary sheet.

Nothing here runs on a device: the phone app's DraggableSheet is the extension's Python control on a fake
Android client. Real data stays untouched (the Library fixture of ``test_book_page_workspace``: HOME,
USERPROFILE, APPDATA, LOCALAPPDATA, GLOSSARION_LIBRARY_DIR and OUTPUT_DIRECTORY under ``tmp_path``).

Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_m5_book_page_batch.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
EXT_DIR = MOBILE_DIR / "extensions" / "flet_glossarion_native"
for entry in (str(APP_DIR), str(EXT_DIR / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

_TBW_SPEC = importlib.util.spec_from_file_location("_glossarion_tbw_helpers_m5",
                                                   Path(__file__).with_name("test_book_page_workspace.py"))
TBW = importlib.util.module_from_spec(_TBW_SPEC)
_TBW_SPEC.loader.exec_module(TBW)
TLU = TBW.TLU
env = TBW.env  # the Library fixture (real shared cores, everything under tmp_path)
needs_flet = TBW.needs_flet

from glossarion_mobile.ui.library import progress_model as pm  # noqa: E402

SPEC_SPINE = ["ch001.xhtml", "ch002.xhtml", "copyright.xhtml"]


def _norm(path) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(str(path))))


def _special_book(env, *, glossary: dict | None = None) -> Path:
    """Library/Raw/Spec.epub (two chapters and a trailing copyright page) and its workspace Output/Spec with both
    chapters translated; ``glossary``: Output/Glossary/Spec/Spec_glossary_progress.json."""
    raw = TLU.make_epub(env["library"] / "Raw" / "Spec.epub", SPEC_SPINE, title="Spec")
    ws = env["output"] / "Spec"
    ws.mkdir(parents=True)
    (ws / "source_epub.txt").write_text(raw, encoding="utf-8")
    chapters = {}
    for number, name in enumerate(("ch001", "ch002"), start=1):
        (ws / f"response_{name}.html").write_text(f"<html><body>{name}</body></html>", encoding="utf-8")
        chapters[str(number)] = {"actual_num": number, "status": "completed", "output_file": f"response_{name}.html",
                                 "original_basename": f"{name}.xhtml", "content_hash": f"h{number}"}
    (ws / "translation_progress.json").write_text(json.dumps({"version": "2.1", "chapters": chapters}),
                                                  encoding="utf-8")
    if glossary is not None:
        gp = env["output"] / "Glossary" / "Spec" / "Spec_glossary_progress.json"
        gp.parent.mkdir(parents=True)
        gp.write_text(json.dumps(glossary), encoding="utf-8")
    return ws


async def _spec_row(service, ws) -> dict:
    await service.refresh()
    rows = [*service.snapshot.in_progress, *service.snapshot.completed]
    return next(b for b in rows if _norm(service.workspace_for(b) or "") == _norm(ws))


# ==========================================================================
# 1. Chapters header
# ==========================================================================


@needs_flet
def test_chapters_header_is_one_row_of_run_translation_and_run_qa_scan(env):
    service = env["service"]

    async def scenario():
        row = await _spec_row(service, env["ws"])
        screen, ctx, session = await TBW._open(env, service.bid_for(row), "chapters", files=TBW.FakeFiles())
        chapters = screen.chapters
        assert not hasattr(chapters, "folder_chip") and not hasattr(chapters, "mode")
        texts = TLU._texts(chapters.root)
        assert not any(t.startswith("\U0001f4c1") for t in texts) and not any(t.startswith("Mode: ") for t in texts)
        header = next(c for c in chapters.root.controls[0].content.controls if getattr(c, "key", None) == "ch-actions")
        assert header.controls == [chapters.run_button, chapters.qa_button]
        assert not header.wrap and chapters.run_button.expand and chapters.qa_button.expand  # one row, shared width
        # the band above the tabs keeps the book's name and mode; Files is in ⋯ and on Overview
        assert screen.strip_mode.content.value.startswith("Mode: ")
        assert "Files" in [str(item.content) for item in chapters.more_menu.items]
        assert any(c.key == "ov-files" for c in screen.overview.icon_row.controls)
        screen.dispose()

    asyncio.run(scenario())


#: Roboto Medium advance widths in em (the M3 label font), for the characters of the two labels.
_ROBOTO_MEDIUM_EM = {"R": 0.63, "Q": 0.69, "A": 0.66, "u": 0.55, "n": 0.55, "t": 0.33, "r": 0.34, "a": 0.54,
                     "s": 0.51, "l": 0.25, "i": 0.25, "o": 0.56, "c": 0.52, " ": 0.25}


@needs_flet
def test_the_header_buttons_fit_a_360_dp_phone_side_by_side(env):
    """Each button gets half of a 360 dp phone's row (12 dp gutters, 8 dp gap = 164 dp). M3's icon button would
    need 16 + 18 (icon) + 8 (gap) + 24 dp around the label: ~162 dp for "Run translation" (Roboto Medium 14 sp,
    0.1 letter spacing, ~96 dp), a 2 dp margin. The header buttons use 12 dp side padding (~17 dp of room) and a
    one-line label that ends in "…" instead of wrapping when a wider font or text size still does not fit.
    Arithmetic and control properties only: nothing is rendered on a device."""
    from glossarion_mobile.ui.library import chapters_tab as ct

    half = (360 - 2 * 12 - 8) / 2
    chrome = 2 * ct.HEADER_BUTTON_PADDING + 18 + 8
    for label in (ct.RUN_TRANSLATION_LABEL, ct.RUN_QA_LABEL):
        width = sum(_ROBOTO_MEDIUM_EM[ch] for ch in label) * 14 + 0.1 * len(label)
        assert chrome + width + 12 <= half, (label, chrome + width, half)  # at least 12 dp to spare
    service = env["service"]

    async def scenario():
        row = await _spec_row(service, env["ws"])
        screen, ctx, session = await TBW._open(env, service.bid_for(row), "chapters", files=TBW.FakeFiles())
        for button in (screen.chapters.run_button, screen.chapters.qa_button):
            assert button.style.padding.left == button.style.padding.right == ct.HEADER_BUTTON_PADDING
            label = button.content
            assert label.max_lines == 1 and label.no_wrap and label.overflow == ct.ft.TextOverflow.ELLIPSIS
        assert [b.content.value for b in (screen.chapters.run_button, screen.chapters.qa_button)] == [
            "Run translation", "Run QA scan"]
        screen.dispose()

    asyncio.run(scenario())


# ==========================================================================
# 2. Special files always listed in the Chapters list
# ==========================================================================


def test_skipped_special_rows_read_special_file_skipped():
    def pres(status, *, special=False, hidden=False):
        return types.SimpleNamespace(row_id=f"r-{status}-{special}", kind="chapter", status=status, icon="⏭️",
                                     label="Skipped", title="t", is_special=special, hidden=hidden, info={})

    assert pm._row_vm(pres("skipped", special=True)).label == pm.SPECIAL_SKIPPED_LABEL == "Special file (Skipped)"
    assert pm._row_vm(pres("skipped", hidden=True)).skipped_special
    assert pm._row_vm(pres("skipped")).label == "Skipped"  # a skipped row that is not a special file
    assert not pm._row_vm(pres("completed", special=True)).skipped_special  # a special file the run translates


@needs_flet
def test_chapters_always_list_special_files_marked_skipped_and_not_counted(env):
    service = env["service"]
    ws = _special_book(env)

    async def scenario():
        row = await _spec_row(service, ws)
        screen, ctx, session = await TBW._open(env, service.bid_for(row), "chapters", files=TBW.FakeFiles())
        chapters = screen.chapters
        assert chapters.show_special is True
        assert "f-special" not in [getattr(item, "key", None) for item in chapters._filter_items()]
        assert "Show special files" not in [str(item.content) for item in chapters._filter_items()]
        rows = [r for r in chapters.visible if r.kind == "chapter"]
        assert [r.filename for r in rows] == SPEC_SPINE  # listed without any toggle
        special = rows[-1]
        assert special.status == "skipped" and special.skipped_special and special.label == pm.SPECIAL_SKIPPED_LABEL
        assert special.status in screen.progress.groups["skipped"]
        listed = [r for r in chapters.visible if r.kind != "chunk"]  # + the Metadata / TOC / headers rows
        # the skipped special file is listed but not counted (Total, the strip's bar)
        assert screen.progress.total_text == f"Total: {len(listed) - 1}"
        assert screen.progress.total == len(listed) - 1 and screen.progress.completed == 2
        assert next(c for c in screen.progress.chips if c.group == "skipped").count == 1
        # its row sheet keeps the desktop "Do not skip" for a skipped special row
        sheet = await chapters.show_row_sheet(special)
        assert "⏭️ Do not skip (remove keyword 'copyright')" in TLU._texts(sheet.dialog if hasattr(sheet, "dialog")
                                                                          else sheet)
        screen.dispose()

    asyncio.run(scenario())


# ==========================================================================
# 3. The Glossary list: the extractor's decision from the user's config
# ==========================================================================


def _glossary_rows(service, book):
    view = pm.load_glossary_view(service, book)
    return view, [r for r in view.rows if r.kind in ("chapter", "special")]


@pytest.mark.parametrize("progress", [{"in_progress": [0]},
                                      {"completed": [0, 1], "indexing": "chapter_index_zero_based",
                                       "chapter_filenames": {"0": "ch001.xhtml", "1": "ch002.xhtml"}}],
                         ids=["running", "finished"])
def test_glossary_list_shows_the_skipped_special_file_at_its_spine_position(env, monkeypatch, progress):
    service = env["service"]
    ws = _special_book(env, glossary=progress)
    # the UI process's environment would say "translate special files"; the list follows the config instead
    monkeypatch.setenv("TRANSLATE_SPECIAL_FILES", "1")
    book = asyncio.run(_spec_row(service, ws))
    view, rows = _glossary_rows(service, book)
    assert [(r.kind, r.subtitle.split(" → ")[0]) for r in rows] == [
        ("chapter", "ch001.xhtml"), ("chapter", "ch002.xhtml"), ("special", "copyright.xhtml")]
    special = rows[-1]
    assert special.status == "skipped_special" and special.label == pm.SPECIAL_SKIPPED_LABEL
    assert special.title == "[003] Ch.000" and special.index is None and not special.pinned
    assert special.status in view.groups["skipped"]
    import glossary_progress_core as gpc

    stats = gpc.glossary_stats(view.state)
    chips = {c.group: c.count for c in view.chips}
    assert chips["skipped"] == int(stats.get("skipped", 0) or 0) + 1  # the Skipped chip counts the special file
    assert view.total_text == f"Total: {stats['total']}"  # ... the total does not
    assert gpc.glossary_chapter_map(view.state)[1] == 2
    if "completed" in progress:
        assert [r.status for r in rows[:2]] == ["completed", "completed"]


def test_glossary_list_settings_come_from_the_users_config(env, monkeypatch):
    service = env["service"]
    ws = _special_book(env, glossary={"in_progress": [0]})
    book = asyncio.run(_spec_row(service, ws))
    for key in ("TRANSLATE_SPECIAL_FILES", "SPECIAL_FILE_KEYWORDS", "SPECIAL_FILE_EXACT",
                "GLOSSARY_NEVER_CONSIDER_IN_BETWEEN_FILES_AS_SPECIAL"):
        monkeypatch.delenv(key, raising=False)
    settings = pm.glossary_special_files(service)
    assert settings["translate_special_files"] is False and settings["protect_interior_special_files"] is True
    service.config["translate_special_files"] = True  # Settings › Translate special files
    assert pm.glossary_special_files(service)["translate_special_files"] is True
    _view, rows = _glossary_rows(service, book)
    assert [r.kind for r in rows] == ["chapter"] * 3  # the run sends it, so it is a chapter row
    service.config["translate_special_files"] = False
    service.config["special_file_exact"] = "ch002"  # a custom exact name: the numbered tail stays protected
    _view, rows = _glossary_rows(service, book)
    assert [r.kind for r in rows] == ["chapter", "chapter", "special"]
    service.config["never_consider_in_between_files_as_special"] = False  # ... until protection is off
    _view, rows = _glossary_rows(service, book)
    assert [r.kind for r in rows] == ["chapter", "special", "special"]


def test_glossary_list_shows_special_files_before_any_extraction(env, monkeypatch):
    """No glossary progress file yet: the empty state stays, and the file the first extraction will skip is
    listed under it at its spine position with the Skipped special mark (built from the config, not the env)."""
    service = env["service"]
    ws = _special_book(env)  # no Output/Glossary/Spec/Spec_glossary_progress.json
    monkeypatch.setenv("TRANSLATE_SPECIAL_FILES", "1")  # the UI process's environment is ignored
    book = asyncio.run(_spec_row(service, ws))
    view = pm.load_glossary_view(service, book)
    assert view.empty and view.path is None and view.state is None and view.signature is None
    assert [(r.kind, r.status, r.label, r.title) for r in view.rows] == [
        ("special", "skipped_special", pm.SPECIAL_SKIPPED_LABEL, "[003] Ch.000")]
    assert view.rows[0].subtitle == "copyright.xhtml" and view.rows[0].status in view.groups["skipped"]
    assert not list((env["output"]).rglob("*_glossary_progress.json"))  # listing them writes nothing
    service.config["translate_special_files"] = True  # the run would send it: nothing is listed
    assert pm.load_glossary_view(service, book).rows == ()


def test_glossary_list_follows_a_settings_change_on_reload(env):
    """The open list re-reads the special-file settings on every reload (the model keeps the ones it was built
    with), so it never disagrees with the next extraction."""
    service = env["service"]
    ws = _special_book(env, glossary={"in_progress": [0]})
    book = asyncio.run(_spec_row(service, ws))
    view = pm.load_glossary_view(service, book)
    assert [r.kind for r in view.rows if r.kind in ("chapter", "special")] == ["chapter", "chapter", "special"]
    service.config["translate_special_files"] = True
    view = pm.load_glossary_view(service, book, previous=view)
    assert [r.kind for r in view.rows if r.kind in ("chapter", "special")] == ["chapter"] * 3
    service.config["translate_special_files"] = False
    service.config["special_file_exact"] = "ch002"
    service.config["never_consider_in_between_files_as_special"] = False
    view = pm.load_glossary_view(service, book, previous=view)
    assert [r.kind for r in view.rows if r.kind in ("chapter", "special")] == ["chapter", "special", "special"]
    import glossary_progress_core as gpc

    assert gpc.glossary_chapter_map(view.state)[1] == 1


@needs_flet
def test_book_page_reloads_the_glossary_list_when_the_settings_change(env):
    service = env["service"]
    ws = _special_book(env, glossary={"in_progress": [0]})

    async def scenario():
        row = await _spec_row(service, ws)
        screen, ctx, session = await TBW._open(env, service.bid_for(row), "glossary", files=TBW.FakeFiles())
        screen.did_show()
        assert [r.kind for r in screen.glossary.rows if r.kind in ("chapter", "special")][-1] == "special"
        assert set(pm.GLOSSARY_SPECIAL_FILE_KEYS) == {
            "translate_special_files", "special_file_keywords", "special_file_exact",
            "never_consider_in_between_files_as_special"}
        service.config["translate_special_files"] = True  # Settings › Translate special files
        for _ in range(100):
            await asyncio.sleep(0.05)
            kinds = [r.kind for r in screen.glossary.rows if r.kind in ("chapter", "special")]
            if "special" not in kinds:
                break
        assert kinds == ["chapter"] * 3
        screen.dispose()

    asyncio.run(scenario())


def test_special_glossary_rows_have_no_progress_actions():
    from glossarion_mobile.ui.library.glossary_tab import SPECIAL_ROW_REASON, GlossaryTab

    tab = GlossaryTab.__new__(GlossaryTab)
    tab.page = types.SimpleNamespace(service=types.SimpleNamespace(
        core=types.SimpleNamespace(available=lambda *a: True)))
    tab.view = types.SimpleNamespace(state=object(), path="p.json", deleted=False)
    special = pm.GlossaryRowVM(key="special:3", kind="special", status="skipped_special", icon="⏭️",
                               label=pm.SPECIAL_SKIPPED_LABEL, title="[003] Ch.000")
    chapter = pm.GlossaryRowVM(key="chapter:0", kind="chapter", status="not_completed", icon="⬜", label="x",
                               title="[001] Ch.001")
    assert tab._reason("mark_completed", [special]) == SPECIAL_ROW_REASON
    assert tab._reason("remove", [special]) == SPECIAL_ROW_REASON
    assert tab._reason("mark_completed", [special, chapter]) is None


# ==========================================================================
# 4. Overview: the metadata beside the cover
# ==========================================================================


def _overview(metadata, *, workspace="W"):
    from glossarion_mobile.ui.library.common import LibraryContext
    from glossarion_mobile.ui.library.overview_tab import OverviewTab

    service = types.SimpleNamespace(covers={}, format_of=lambda book: "epub",
                                    workspace_for=lambda book: book.get("output_folder") or "")
    page = types.SimpleNamespace(
        ctx=LibraryContext(service=service, platform="android"), service=service,
        book={"name": "Novel", "output_folder": workspace}, bid="b1",
        details={"chapters_info": [], "metadata_json": metadata}, details_model=lambda: None,
        metadata_reason=lambda: None, progress=None, glossary=None, job_view=None, open_reader=lambda **kw: None)
    overview = OverviewTab(page)
    return overview, overview.build()


@needs_flet
def test_overview_has_no_metadata_section_and_edits_beside_the_cover():
    overview, body = _overview({"title": "The Book", "creator": "Writer", "publisher": "House", "language": "ko",
                                "date": "2024-05-01"})
    hero_column = overview.hero.controls[1]
    assert overview.hero.controls[0] is overview.cover
    assert hero_column.controls == [overview.title_text, overview.author_text, overview.publisher_text,
                                    overview.hero_chips, overview.tags_row, overview.edit_button]
    texts = TLU._texts(body)
    assert "METADATA" not in texts and "—" not in texts and not hasattr(overview, "metadata_rows")
    assert overview.title_text.value == "The Book" and overview.author_text.value == "Writer"
    assert overview.publisher_text.visible and overview.publisher_text.value.endswith("House")
    chips = [c.label.value for c in overview.hero_chips.controls]
    assert "\U0001f310 ko" in chips and "\U0001f4c5 2024" in chips
    # nothing twice: each value appears once on the page
    assert texts.count("The Book") == 1 and texts.count("Writer") == 1
    assert not overview.edit_button.disabled
    assert all(item is not overview.edit_button for item in [*overview.left_items, *overview.right_items])
    # empty values are not shown; no workspace: Edit disabled with the reason (the same rule as before)
    bare, bare_body = _overview({"title": "Bare"}, workspace="")
    assert not bare.publisher_text.visible and not bare.author_text.visible and bare.hero_chips.controls == []
    assert bare.edit_button.disabled and bare.edit_button.tooltip == "No output workspace yet"
    assert "—" not in TLU._texts(bare_body)
    # the landscape arrangement keeps the hero (with Edit) in the left column
    bare.columns = True
    left = bare._arranged()[0].controls[0]
    assert bare.hero in left.controls and "METADATA" not in TLU._texts(bare._arranged()[0])


# ==========================================================================
# 5. The chat card's Glossary sheet opens as a preview
# ==========================================================================


def _plan_sheet(page, calls):
    from glossarion_mobile.services.glossary import GlossaryService
    from glossarion_mobile.ui.glossary.common import GlossaryContext
    from glossarion_mobile.ui.glossary.sheets import PlanGlossarySheet

    ctx = GlossaryContext(service=GlossaryService(config={"auto_glossary_mode": "balanced",
                                                          "manual_glossary_prompt3": "BAL"}),
                          page=page, platform="android", notify=lambda *a, **k: None, navigate=lambda *a, **k: None)
    noop = lambda: None  # noqa: E731
    return PlanGlossarySheet(ctx, on_load_file=noop, on_clear=noop, on_review=noop,
                             on_mode=lambda: calls.append("mode"), effective="Glossary: Balanced (auto)",
                             on_settings=noop, prompt_bucket="balanced_full")


def _inside(root, target) -> bool:
    if root is target:
        return True
    for name in ("content", "controls"):
        value = getattr(root, name, None)
        for child in (value if isinstance(value, list) else [value]):
            if child is not None and not isinstance(child, str) and _inside(child, target):
                return True
    return False


def _phone(width=412, height=915):
    conn, session = TLU._TB._fake_session("android")
    session.apply_page_patch({"width": width, "height": height})
    return conn, session, session.page


@needs_flet
def test_chat_card_glossary_sheet_opens_short_on_the_mode_tile(monkeypatch):
    from flet_glossarion_native import DraggableSheet

    monkeypatch.delenv("GLOSSARION_FORCE_NATIVE_STUB", raising=False)

    async def scenario():
        from glossarion_mobile.ui.components.sheet import sheet_body

        calls: list = []
        _conn, _session, page = _phone()
        sheet = _plan_sheet(page, calls).show(page, preview=True)
        body = sheet_body(sheet.dialog)
        assert sheet.native and isinstance(body, DraggableSheet)
        header = body.header.content
        assert sheet.mode_tile in header.controls and header.controls[0].value == sheet.title
        # a short preview: the title and the whole mode tile, then a swipe up for the rest (full height)
        assert 0.12 <= body.initial_size <= 0.35 and body.min_size < body.initial_size and body.max_size == 1.0
        rows = body.controls[0].content.controls
        assert sheet.profile_row in rows and sheet.prompt_card in rows and sheet.mode_tile not in rows
        assert sheet.done_button in rows[-1].controls
        assert sheet.dialog.open
        # the controls keep working inside it: the mode tile opens the ModeSheet (this sheet closes first)
        sheet.mode_tile.on_click(None)
        assert calls == ["mode"]

    asyncio.run(scenario())


@needs_flet
def test_glossary_sheet_keeps_todays_layout_elsewhere(monkeypatch):
    async def scenario():
        calls: list = []
        _conn, _session, page = _phone()
        # the Library TranslateSheet's chip (no preview): the ordinary measured sheet
        library = _plan_sheet(page, calls).show(page)
        assert not library.native and _inside(library.body, library.mode_tile)
        assert type(library.dialog.content).__name__ != "DraggableSheet"
        # a client without the extension: the chat card's sheet falls back to the same sheet
        monkeypatch.setenv("GLOSSARION_FORCE_NATIVE_STUB", "1")
        fallback = _plan_sheet(page, calls).show(page, preview=True)
        assert not fallback.native and _inside(fallback.body, fallback.mode_tile)

    asyncio.run(scenario())


def test_chat_cards_ask_for_the_preview_and_other_openers_do_not():
    import inspect

    from glossarion_mobile.ui.chat.chat_view import ChatView
    from glossarion_mobile.ui.glossary.feature import GlossaryFeature

    seen: dict = {}
    view = types.SimpleNamespace(plan_glossary_opener=lambda **kw: seen.update(kw) or "sheet",
                                 _plan_glossary_label=lambda: "Glossary: Full (auto)", apply_settings_changed=None,
                                 settings=lambda: types.SimpleNamespace(glossary_override_mode="none"), bound=False)
    assert ChatView.open_plan_glossary(view, ["/b.epub"]) == "sheet"
    assert seen["preview"] is True and seen["inputs"] == ["/b.epub"]
    # the feature's default (the Library TranslateSheet chip) stays the ordinary sheet
    assert inspect.signature(GlossaryFeature.open_plan_glossary_sheet).parameters["preview"].default is False


def test_plus_sheet_and_glossary_sheet_share_one_preview_helper():
    import inspect

    from glossarion_mobile.ui.components import drag_sheet
    from glossarion_mobile.ui.glossary import sheets
    from glossarion_mobile.ui.sheets import plus_sheet

    assert "preview_sheet(" in inspect.getsource(plus_sheet.PlusSheet._use_draggable)
    assert "preview_sheet(" in inspect.getsource(sheets.PlanGlossarySheet._use_preview)
    assert "draggable_sheet_class" not in inspect.getsource(plus_sheet.PlusSheet._use_draggable)
    assert drag_sheet.preview_sheet(types.SimpleNamespace(platform="windows", web=False, height=900), header=None,
                                    controls=[], header_height=100, key="k") is None
