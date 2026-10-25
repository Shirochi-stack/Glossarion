"""Hotfix 1 issue 9 acceptance (owner, on the phone): "Lack of feedback when interacting with progress manager, like
setting an entry to in progress just happens with no refresh".

End to end on the real app objects in the headless Flet session (``test_bootstrap._fake_session``): the real
``BookPageScreen`` (Chapters = the Progress Manager, Glossary = Glossary Progress) over the real
``LibraryService`` and shared cores pinned to the ``test_library_ui`` fixture Library, the real ``UiDispatcher``,
the app's own snackbar (``GlossarionApp.notify`` -> ``show_snackbar``) and, for Retranslate, the real
``JobService`` running the real ``retranslate`` job kind. Every tap goes through Flet's event dispatch
(``Session.dispatch_event`` on the control the phone shows: row ⋯, the sheet row, the bulk bar button, the
confirmation's Yes); what the phone saw is read back from the messages the session sent to the client.

What the owner must see after every Progress Manager action:

* while the action writes, the list's progress bar runs (the write is held inside the shared per-file progress
  lock, so the screen is looked at mid-write, never with a sleep);
* the rows, the stats chips and the summary strip refresh at once - already on screen (and sent to the client)
  when the confirmation snackbar shows, with no poll tick (the Book page poller is not running);
* a snackbar confirms each action with the desktop result text (a long, multi-line desktop result box opens as
  the result sheet instead: Retranslate's "Success");
* a refused or failing action says why in a snackbar, the bar stops and the rows keep the truth.

Covered: Restore In Progress Status from the row menu (an In progress entry -> Failed / Not Translated), Remove
Pending Mark / Remove QA Failed Mark (-> Completed), the bulk Retranslate (a queued job that says it is queued,
then the reset row and the result box), row ⋯ › Retranslate this chapter (the entry goes back to Pending / In
progress while its single-chapter job runs: the row must not keep showing Completed), the bulk selection bar
(Remove QA mark, More › Restore In Progress Status), the Glossary Progress tab (Mark as completed, Remove from
progress, bulk), and the refusal / corrupt-file / failed-save error paths.

Run from src/mobile with the mobile venv (Python 3.12):
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \
        tests_host/test_hotfix1_issue9.py
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.util
import json
import sys
import threading
import time
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# The fixture Library (Raw/Book.epub + Output/Book with completed / qa_failed / chunked pending / not translated
# rows and a glossary progress file), the real shared cores pinned to it, and the Flet fake session.
_TLU_SPEC = importlib.util.spec_from_file_location("_glossarion_tlu_hotfix1_issue9",
                                                   Path(__file__).with_name("test_library_ui.py"))
_TLU = importlib.util.module_from_spec(_TLU_SPEC)
_TLU_SPEC.loader.exec_module(_TLU)

real_env = _TLU.real_env  # noqa: F811 - the fixture, re-exported for this module
needs_flet = _TLU.needs_flet
TIMEOUT = 20.0


# ==========================================================================
# Isolation: no real home / app data / config is ever touched
# ==========================================================================


def _md5(path: Path):
    try:
        return hashlib.md5(path.read_bytes()).hexdigest()
    except OSError:
        return None


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    for sub in ("AppData/Roaming", "AppData/Local", "data"):
        (home / sub).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("APPDATA", str(home / "AppData" / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData" / "Local"))
    monkeypatch.setenv("GLOSSARION_DATA_DIR", str(home / "data"))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    config = SRC_DIR / "config.json"
    before = _md5(config)
    yield
    assert _md5(config) == before, "src/config.json was modified"


def _needs_cores():
    if _TLU._core("progress_core", "build_book_progress", "_retranslation_progress_lock") is None or _TLU._core(
            "progress_actions", "restore_in_progress", "remove_qa_marks", "remove_pending_marks") is None:
        pytest.skip("progress cores not importable")


# ==========================================================================
# Fixture edits (on disk, before the Book page opens)
# ==========================================================================


def _progress_path(env) -> Path:
    return env["ws"] / "translation_progress.json"


def _read_progress(env) -> dict:
    return json.loads(_progress_path(env).read_text(encoding="utf-8"))


def _set_status(env, key, status):
    prog = _read_progress(env)
    prog["chapters"][key]["status"] = status
    _progress_path(env).write_text(json.dumps(prog), encoding="utf-8")


# ==========================================================================
# Reading what the client was sent
# ==========================================================================


def _children(control):
    import dataclasses

    import flet as ft

    for field in dataclasses.fields(control):
        if field.name.startswith("_") or field.name in ("parent", "page"):
            continue
        try:
            value = getattr(control, field.name)
        except Exception:
            continue
        if isinstance(value, ft.BaseControl):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, ft.BaseControl):
                    yield item


def _walk(control, seen=None):
    seen = set() if seen is None else seen
    if control is None or id(control) in seen:
        return
    seen.add(id(control))
    yield control
    for child in _children(control):
        yield from _walk(child, seen)


def _key_of(control):
    key = getattr(control, "key", None)
    return getattr(key, "value", key)


def _find_key(root, key, kind=None):
    for control in _walk(root):
        if _key_of(control) == key and (kind is None or isinstance(control, kind)):
            return control
    return None


def _text_values(root):
    import flet as ft

    return [str(c.value) for c in _walk(root) if isinstance(c, ft.Text) and c.value]


def _patch_controls(conn, start, end=None):
    """Every control object carried by the PATCH messages sent to the client in ``[start, end)``."""
    from flet.controls.base_control import BaseControl

    out = []

    def visit(value):
        if isinstance(value, BaseControl):
            out.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    for message in conn.messages[start:end]:
        visit(getattr(message.body, "patch", None))
    return out


def _patched_values(session, message):
    """``(value, ancestors)`` for every Replace op of a PATCH: the op's node is resolved through the patch's path
    tree (``[node, {prop or list index: subtree}]``) on the server-side controls, from the patched root down."""
    patch = getattr(message.body, "patch", None)
    if not patch:
        return []
    root = session.index.get(getattr(message.body, "id", None))
    nodes = {}

    def walk(tree, obj, ancestors):
        if not isinstance(tree, (list, tuple)) or not tree:
            return
        nodes[tree[0]] = ancestors + [obj]
        for step, sub in (tree[1].items() if len(tree) > 1 and isinstance(tree[1], dict) else ()):
            try:
                child = obj[step] if isinstance(step, int) else getattr(obj, step)
            except (IndexError, KeyError, AttributeError, TypeError):
                continue
            walk(sub, child, ancestors + [obj])

    walk(patch[0], root, [])
    out = []
    for op in patch[1:]:
        if isinstance(op, (list, tuple)) and len(op) == 4 and getattr(op[0], "value", op[0]) == 0:
            out.append((op[3], nodes.get(op[1], [])))
    return out


def _client_got_row(phone, start, end, key, status_text):
    """The client was sent the row ``key`` showing ``status_text``: a new row control, or the kept row's status
    text changed in place (Flet's diff of a same-key control sends only the changed property)."""
    for control in _patch_controls(phone.conn, start, end):
        for node in _walk(control):
            if _key_of(node) == key and status_text in _text_values(node):
                return True
    for message in phone.conn.messages[start:end]:
        for value, ancestors in _patched_values(phone.session, message):
            if value == status_text and any(_key_of(a) == key for a in ancestors):
                return True
    return False


def _patch_strings(conn, start, end=None) -> set:
    """Every string a PATCH op set in ``[start, end)`` (an in-place diff of a kept control sends only the changed
    property: ``[Replace, node, 'value', '✅ Completed']``)."""
    out = set()

    def visit(value):
        if isinstance(value, str):
            out.add(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    for message in conn.messages[start:end]:
        for op in (getattr(message.body, "patch", None) or [])[1:]:
            visit(op)
    return out


def _client_got_text(conn, start, end, text):
    """The client was sent ``text``: in a new control, or as a kept control's changed property."""
    return (text in _patch_strings(conn, start, end)
            or any(text in _text_values(control) for control in _patch_controls(conn, start, end)))


def _client_got_prop(conn, start, end, control, prop, value):
    """A PATCH for ``control`` itself replaced ``prop`` with ``value`` (``control.update()``)."""
    for message in conn.messages[start:end]:
        if getattr(message.body, "id", None) != control._i:
            continue
        for op in (getattr(message.body, "patch", None) or [])[1:]:
            if isinstance(op, (list, tuple)) and len(op) == 4 and op[1] == 0 and op[2] == prop and op[3] == value:
                return True
    return False


def _count(chip_texts, prefix) -> int:
    """The count of the chip ``prefix`` ("✅ Completed") among on-screen chip texts (0 when it is hidden)."""
    for text in chip_texts:
        if text.startswith(prefix + " ") and text[len(prefix) + 1:].isdigit():
            return int(text[len(prefix) + 1:])
    return 0


async def _until(predicate, timeout=TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


# ==========================================================================
# The phone
# ==========================================================================


class ProgressLockHold:
    """Holds the shared per-file Progress Manager lock (``progress_core._retranslation_progress_lock``, taken by
    every ``mutate_progress`` write) on another thread: a tapped action's write then waits inside the real shared
    code while the test looks at the screen; ``release()`` lets it write."""

    def __init__(self, path):
        import progress_core

        self.lock = progress_core._retranslation_progress_lock(str(path))
        self.held, self.go = threading.Event(), threading.Event()
        self.thread = threading.Thread(target=self._run, name="issue9-lock-hold", daemon=True)

    def _run(self):
        with self.lock:
            self.held.set()
            self.go.wait(TIMEOUT * 3)

    def __enter__(self):
        self.thread.start()
        assert self.held.wait(TIMEOUT), "could not take the progress lock"
        return self

    def release(self):
        self.go.set()
        self.thread.join(TIMEOUT)

    def __exit__(self, *exc):
        self.release()


class Phone:
    """Book page › Chapters on a headless Android phone, wired like ``LibraryFeature.context`` (real dispatcher,
    the app's snackbar)."""

    def __init__(self, env, jobs_factory=None):
        self.env = env
        self.jobs_factory = jobs_factory  # (dispatcher) -> JobService, built on the phone's dispatcher like the app
        self.jobs = None
        self.notes = []

    async def open(self, tab="chapters"):
        from glossarion_mobile.app import GlossarionApp
        from glossarion_mobile.services.dispatcher import UiDispatcher
        from glossarion_mobile.state.store import LoopGuard
        from glossarion_mobile.ui.library.book_page import BookPageScreen
        from glossarion_mobile.ui.library.common import LibraryContext
        from glossarion_mobile.ui.router import parse_route

        service = self.env["service"]
        self.conn, self.session = _TLU._TB._fake_session("android")
        self.page = self.session.page
        await service.refresh()
        self.book = service.snapshot.in_progress[0]
        # GlossarionApp binds its dispatcher and starts the pump (latest-wins channels, batched updates); a private
        # LoopGuard keeps the process-wide UI_LOOP unbound for the tests after this one
        self.dispatcher = UiDispatcher(self.page, guard=LoopGuard()).bind()
        self.dispatcher.start()
        if self.jobs_factory is not None:
            self.jobs = self.jobs_factory(self.dispatcher)
            service.jobs = self.jobs
        app = types.SimpleNamespace(page=self.page)
        self._app_notify = lambda message, action=None, on_action=None: GlossarionApp.notify(
            app, message, action, on_action)
        self.navigated = []
        phone = self

        class PhoneContext(LibraryContext):
            def show(self, dialog):
                """``LibraryContext.show``; a desktop result box long enough for a sheet (``show_message``) is
                recorded like a snackbar, with what the screen showed when it opened."""
                from glossarion_mobile.ui.components.info_sheet import InfoSheet

                if not isinstance(dialog, InfoSheet):
                    return super().show(dialog)
                note = phone._note(dialog.body, kind="sheet", title=dialog.title)
                result = super().show(dialog)
                note.bar, note.sent_to = dialog.dialog, len(phone.conn.messages)
                return result

        self.ctx = PhoneContext(service=service, page=self.page, dispatcher=self.dispatcher, jobs=self.jobs,
                                prefs=service.prefs, platform="android", notify=self._notify,
                                navigate=lambda name, params=None, query=None: self.navigated.append(name))
        self.screen = BookPageScreen(parse_route(f"/library/book/{service.bid_for(self.book)}?tab={tab}"), self.ctx)
        self.screen.actions()
        _TLU._mount(self.page, self.screen.get_body())
        await self.screen.load()
        await self.settle()
        self.chapters = self.screen.chapters
        assert self.chapters.view is not None and self.chapters.view.state is not None, self.chapters.view
        assert not self.chapters.loading.visible
        return self

    async def close(self):
        screen = getattr(self, "screen", None)
        if screen is not None:
            screen.dispose()
        if self.jobs is not None:
            self.jobs.close()
        dispatcher = getattr(self, "dispatcher", None)
        if dispatcher is not None:
            await dispatcher.stop()

    # ---- what the user sees ----------------------------------------------------------------------

    def _note(self, message, *, kind, action=None, on_action=None, title=None):
        note = types.SimpleNamespace(message=message, kind=kind, title=title, action=action, on_action=on_action,
                                     at=len(self.conn.messages), screen=self.snapshot(), bar=None, sent_to=None)
        self.notes.append(note)
        return note

    def _notify(self, message, action=None, on_action=None):
        """``GlossarionApp.notify`` (the real snackbar), recording what the screen showed at that moment."""
        note = self._note(message, kind="snackbar", action=action, on_action=on_action)
        note.bar = self._app_notify(message, action, on_action)
        note.sent_to = len(self.conn.messages)
        return note.bar

    def snapshot(self) -> dict:
        ch = self.screen.chapters
        gp = self.screen.glossary_tab
        rows = {}
        for key, control in dict(ch.controls).items():
            status = _find_key(control, "status")
            rows[key] = status.value if status is not None else None
        return {
            "rows": rows,
            "chips": self.chip_texts(ch.stats_row),
            "busy": bool(ch.loading.visible),
            "strip": self.screen.strip_bar.value if self.screen.strip_bar.visible else None,
            "selecting": ch.selecting,
            "gp_rows": {r.key: r.status for r in (gp.view.rows if gp.view is not None else ())},
            # the Glossary Progress rows as mounted: row key -> its texts (the status line "✅ Completed")
            "gp_screen": {_key_of(c): _text_values(c) for c in _walk(getattr(gp, "root", None))
                          if isinstance(_key_of(c), str) and str(_key_of(c)).startswith(("chapter:", "minimal:",
                                                                                         "refinement:"))},
            "gp_chips": self.chip_texts(gp.stats_row) if getattr(gp, "stats_row", None) is not None else [],
            "gp_busy": bool(getattr(getattr(gp, "loading", None), "visible", False)),
        }

    def chip(self, group, delta=0, tab=None) -> str:
        """The stats chip text of ``group`` with its current count + ``delta`` ("✅ Completed 3")."""
        view = (tab or self.chapters).view
        chip = next(c for c in view.chips if c.group == group)
        return f"{chip.emoji} {chip.label} {chip.count + delta}"

    @staticmethod
    def chip_texts(row) -> list:
        return [text for chip in row.controls if chip.visible for text in _text_values(chip)]

    def row_status(self, key) -> str:
        status = _find_key(self.chapters.controls[key], "status")
        return status.value

    def row(self, status=None, *, key=None, filename=None):
        for row in self.chapters.visible:
            if row.kind != "chapter":
                continue
            if (key is None or row.key == key) and (status is None or row.status == status) and (
                    filename is None or row.filename == filename):
                return row
        raise AssertionError(f"no chapter row {status or key}: {[(r.key, r.status) for r in self.chapters.visible]}")

    def open_dialogs(self, kind=None):
        dialogs = getattr(getattr(self.page, "_dialogs", None), "controls", None) or []
        return [d for d in dialogs if getattr(d, "open", False) and (kind is None or isinstance(d, kind))]

    # ---- taps ------------------------------------------------------------------------------------------

    async def tap(self, control, event="click"):
        assert control is not None
        assert self.session.index.get(control._i) is control, f"{type(control).__name__} is not on the phone"
        await self.session.dispatch_event(control._i, event, None)

    async def long_press(self, control):
        await self.tap(control, "long_press")

    async def open_row_menu(self, key, tab=None):
        """Tap the row's ⋯ and wait for its action sheet."""
        import flet as ft

        root = tab.root if tab is not None else self.chapters.controls[key]
        row_control = _find_key(root, key) if tab is not None else root
        more = next(c for c in _walk(row_control) if isinstance(c, ft.IconButton) and c.icon == ft.Icons.MORE_VERT)
        before = set(map(id, self.open_dialogs(ft.BottomSheet)))
        await self.tap(more)
        assert await _until(lambda: any(id(d) not in before for d in self.open_dialogs(ft.BottomSheet))), \
            "the row menu did not open"
        return next(d for d in self.open_dialogs(ft.BottomSheet) if id(d) not in before)

    @staticmethod
    def sheet_row(sheet, key):
        import flet as ft

        tile = _find_key(sheet, key, ft.ListTile)
        assert tile is not None, f"no {key!r} in {[_key_of(t) for t in _walk(sheet) if isinstance(t, ft.ListTile)]}"
        return tile

    async def confirm(self, title, button="Yes"):
        """Wait for the confirmation dialog ``title`` and tap ``button``; returns its body text."""
        import flet as ft

        def dialog():
            return next((d for d in self.open_dialogs(ft.AlertDialog)
                         if isinstance(d.title, ft.Text) and d.title.value == title), None)

        assert await _until(lambda: dialog() is not None), \
            f"no {title!r} dialog: {[getattr(d.title, 'value', d) for d in self.open_dialogs(ft.AlertDialog)]}"
        found = dialog()
        target = next(b for b in found.actions if getattr(b, "content", None) == button)
        body = _text_values(found.content)
        await self.tap(target)
        return body

    async def bulk(self, tab, action_id):
        bar = tab.bulk_bar
        assert bar.control.visible, "the bulk bar is not showing"
        button = bar.buttons[action_id]
        await self.tap(getattr(button, "content", button))

    async def note(self, count):
        """Wait for the ``count``-th snackbar; returns it."""
        assert await _until(lambda: len(self.notes) >= count), \
            f"no snackbar (have {[n.message for n in self.notes]})"
        return self.notes[count - 1]

    async def settle(self):
        """Every UI task the taps spawned has finished."""
        assert await _until(lambda: not self.dispatcher._tasks), f"UI tasks still running: {self.dispatcher._tasks}"


def _assert_snackbar_shown(phone, note, kind="snackbar"):
    """The app's snackbar (``kind="sheet"``: the result sheet a long desktop text opens in) is open on the page
    and its text reached the client."""
    import flet as ft

    assert note.kind == kind, (note.kind, note.message)
    expected = ft.SnackBar if kind == "snackbar" else ft.BottomSheet
    assert isinstance(note.bar, expected) and note.bar in phone.open_dialogs(expected), f"the {kind} is not showing"
    assert note.message in _text_values(note.bar)
    assert _client_got_text(phone.conn, note.at, note.sent_to, note.message), f"the {kind} never reached the client"


def _assert_refreshed_before(phone, note, start, key, status_text, chip_texts=()):
    """At the moment the snackbar showed, the row and the chips on screen were already the new ones, and the
    client had been sent them (no later poll did it)."""
    assert note.screen["rows"].get(key) == status_text, (key, note.screen["rows"])
    for chip in chip_texts:
        assert chip in note.screen["chips"], (chip, note.screen["chips"])
    assert not note.screen["busy"], "the progress bar was still running when the result showed"
    assert _client_got_row(phone, start, note.at, key, status_text), \
        f"the refreshed row {key} ({status_text}) was not sent before the snackbar"
    for chip in chip_texts:
        assert _client_got_text(phone.conn, start, note.at, chip), f"chip {chip!r} was not sent before the snackbar"


# ==========================================================================
# 1. The owner's case: an In progress row's action from the row menu
# ==========================================================================


@needs_flet
def test_restore_in_progress_from_the_row_menu_shows_work_then_refreshed_row_counts_and_a_snackbar(real_env):
    _needs_cores()
    _set_status(real_env, "1", "in_progress")

    async def scenario():
        phone = await Phone(real_env).open()
        try:
            ch = phone.chapters
            row = phone.row("in_progress")
            assert phone.row_status(row.key) == "\U0001f504 In Progress"
            chips = phone.chip_texts(ch.stats_row)
            assert phone.chip("in_progress") in chips and phone.chip("failed") in chips, chips
            failed_after = phone.chip("failed", +1)
            sheet = await phone.open_row_menu(row.key)
            restore = phone.sheet_row(sheet, "row-restore")
            assert restore.subtitle is None, "Restore In Progress Status is disabled on an In progress row"
            start = len(phone.conn.messages)
            with ProgressLockHold(_progress_path(real_env)) as hold:
                await phone.tap(restore)
                assert not sheet.open, "the sheet stayed open"
                # mid-write: the list's progress bar runs (sent to the client), nothing pretends to be done yet
                assert await _until(lambda: ch.loading.visible), "no progress bar while the action writes"
                assert _client_got_prop(phone.conn, start, None, ch.loading, "visible", True)
                assert phone.notes == [] and phone.row_status(row.key) == "\U0001f504 In Progress"
                assert _read_progress(real_env)["chapters"]["1"]["status"] == "in_progress"
                hold.release()
                note = await phone.note(1)
            await phone.settle()
            assert note.message == "Successfully marked 1 as failed."
            _assert_snackbar_shown(phone, note)
            _assert_refreshed_before(phone, note, start, row.key, "❌ Failed", chip_texts=(failed_after,))
            assert not any(t.startswith("\U0001f504 In Progress") for t in note.screen["chips"]), note.screen["chips"]
            assert not ch.loading.visible
            assert _client_got_prop(phone.conn, start, None, ch.loading, "visible", False)
            assert _read_progress(real_env)["chapters"]["1"]["status"] == "failed"
            assert len(phone.notes) == 1, [n.message for n in phone.notes]  # one confirmation per action
        finally:
            await phone.close()

    asyncio.run(scenario())


# ==========================================================================
# 2. Marking chapters Completed (Remove Pending Mark, Remove QA Failed Mark with its confirmation)
# ==========================================================================


@needs_flet
def test_remove_pending_and_remove_qa_mark_complete_rows_with_refreshed_counts_strip_and_snackbars(real_env):
    _needs_cores()
    # chapter 1 is Pending while its saved HTML exists, so Remove Pending Mark restores it (HTML is authoritative)
    _set_status(real_env, "1", "pending")

    async def scenario():
        phone = await Phone(real_env).open()
        try:
            ch = phone.chapters
            strip_before = phone.screen.strip_bar.value
            completed_before = _count(phone.chip_texts(ch.stats_row), "✅ Completed")
            assert phone.chip("completed") in phone.chip_texts(ch.stats_row)
            # Remove Pending Mark: Pending -> Completed
            pending = phone.row("pending", filename="ch001.xhtml")
            sheet = await phone.open_row_menu(pending.key)
            start = len(phone.conn.messages)
            await phone.tap(phone.sheet_row(sheet, "row-remove-pending"))
            note = await phone.note(1)
            await phone.settle()
            assert note.message.startswith("Restored 1 pending entries."), note.message
            _assert_snackbar_shown(phone, note)
            _assert_refreshed_before(phone, note, start, pending.key, "✅ Completed")
            assert _count(note.screen["chips"], "✅ Completed") > completed_before, note.screen["chips"]
            assert _client_got_text(phone.conn, start, note.at, phone.chip("completed"))
            assert note.screen["strip"] > strip_before, (strip_before, note.screen["strip"])
            # Remove QA Failed Mark: the desktop confirmation, then QA Failed -> Completed
            qa = phone.row("qa_failed")
            expected = (phone.chip("completed", +1), phone.chip("failed", -1))
            sheet = await phone.open_row_menu(qa.key)
            start = len(phone.conn.messages)
            await phone.tap(phone.sheet_row(sheet, "row-remove-qa"))
            body = await phone.confirm("Confirm Remove Failed Mark")
            assert body == ["Remove failed mark from 1 chapters?"]
            note = await phone.note(2)
            await phone.settle()
            assert note.message == "Removed failed mark from 1 chapters."
            _assert_snackbar_shown(phone, note)
            _assert_refreshed_before(phone, note, start, qa.key, "✅ Completed", chip_texts=expected)
            assert note.screen["strip"] > phone.notes[0].screen["strip"]
            prog = _read_progress(real_env)["chapters"]
            assert prog["1"]["status"] == "completed" and prog["2"]["status"] == "completed", prog
            assert not ch.loading.visible
        finally:
            await phone.close()

    asyncio.run(scenario())


# ==========================================================================
# 3. Retranslate (a job, queued behind the running one): the chapter's output is deleted, the entry Pending
# ==========================================================================


class _Backend:
    """``services.jobs.JobBackend`` stand-in (no shared translation pipeline is needed by these job kinds)."""

    def __init__(self):
        self.job_lock = threading.RLock()

    def scoped_process_state(self, *, capture_stdout=None, lock, **_kw):
        @contextlib.contextmanager
        def scope():
            with lock:
                yield

        return scope()

    def make_owner(self, config, *, host, **_kw):
        return types.SimpleNamespace(stop_requested=False, graceful_stop_var=True, wait_for_chunks_var=True,
                                     config=config, host=host)

    def reset_for_new_run(self, kind="translation", **_kw):
        return 1

    def request_stop(self, *, graceful, wait_for_chunks, force, set_stop_requested, log=None, **_kw):
        set_stop_requested()

    def progress_watcher(self, resolver, host, interval=2.0, **_kw):
        return None

    def restore_in_progress(self, **_kw):
        return None


@needs_flet
def test_retranslate_queued_behind_a_job_says_so_then_resets_the_row_with_the_result_box(real_env, tmp_path):
    _needs_cores()
    from glossarion_mobile import job_kinds
    from glossarion_mobile.services.jobs import JobService, JobSpec

    release, entered = threading.Event(), threading.Event()

    def blocker(ctx):  # this book's compile runs while the user taps Retranslate
        entered.set()
        release.wait(TIMEOUT * 3)
        return {"ok": True, "outputs": []}

    def kinds(kind):
        if kind == "compile_epub":
            return job_kinds.KindInfo(kind=kind, verb="Compiling EPUB", icon="MENU_BOOK", stop_kind="compile",
                                      run=blocker, resumable=False)
        return job_kinds.get_kind(kind)  # the real retranslate kind

    def make_jobs(dispatcher):  # the app builds its JobService on the UI dispatcher (JobsFeature)
        return JobService(jobs_dir=str(tmp_path / "jobs"), backend=_Backend(), kinds=kinds,
                          config_loader=lambda: {}, dispatcher=dispatcher)

    async def scenario():
        phone = Phone(real_env, make_jobs)
        try:
            await phone.open()
            screen, jobs = phone.screen, phone.jobs
            screen.did_show()  # the Book page's own jobs subscription
            screen.poller.stop()  # no 2 s poll: every refresh below is the action's own
            await phone.settle()
            ch = phone.chapters
            service = real_env["service"]
            jobs.submit(JobSpec(kind="compile_epub", title="Book", inputs=(str(real_env["ws"]),),
                                origin=service.origin_for(phone.book)))
            assert entered.wait(TIMEOUT)
            assert await _until(lambda: screen.job is not None and screen.strip_job.visible)
            done = phone.row("completed")
            assert phone.row_status(done.key) == "✅ Completed"
            expected = (phone.chip("pending", +1), phone.chip("completed", -1))
            await phone.long_press(ch.controls[done.key])
            assert ch.selecting and ch.bulk_bar.control.visible
            await phone.bulk(ch, "retranslate")
            await phone.confirm("Confirm Retranslation")
            # queued behind the compile: said at once (the strip still shows the compile)
            queued = await phone.note(1)
            assert queued.message == "Queued: resetting 1 row(s) for retranslation · runs after the current job"
            assert queued.action == "Jobs" and callable(queued.on_action)
            _assert_snackbar_shown(phone, queued)
            assert not ch.selecting and not ch.bulk_bar.control.visible
            assert queued.screen["rows"][done.key] == "✅ Completed"  # nothing reset yet: the job waits
            assert phone.row_status(done.key) == "✅ Completed"
            start = len(phone.conn.messages)
            release.set()
            # the reset ran: the desktop result box (long: the result sheet) after the rows refreshed
            note = await phone.note(2)
            assert note.title == "Success", (note.kind, note.title, note.message)
            assert note.message.startswith("Successfully deleted 1 files") or note.message.startswith(
                "Successfully Deleted 1 files"), note.message
            _assert_snackbar_shown(phone, note, kind="sheet")
            assert "reset 1 chapter statuses to pending" in note.message, note.message
            _assert_refreshed_before(phone, note, start, done.key, "❓ Pending", chip_texts=expected)
            assert _read_progress(real_env)["chapters"]["1"]["status"] == "pending"
            assert not (real_env["ws"] / "response_ch001.html").exists()
            assert await _until(lambda: screen.job is None and not screen.strip_job.visible)
            await phone.settle()
        finally:
            release.set()
            await phone.close()

    asyncio.run(scenario())


# ==========================================================================
# 4. Bulk selection actions
# ==========================================================================


@needs_flet
def test_bulk_bar_remove_qa_mark_and_more_restore_in_progress_refresh_every_selected_row(real_env):
    _needs_cores()
    _set_status(real_env, "1", "in_progress")

    async def scenario():
        phone = await Phone(real_env).open()
        try:
            ch = phone.chapters
            first, qa = phone.row("in_progress"), phone.row("qa_failed")
            # select two rows (long-press, then tap), bulk "Remove QA mark" -> confirmation -> one row cleared
            await phone.long_press(ch.controls[first.key])
            await phone.tap(ch.controls[qa.key])
            assert ch.selected == {first.key, qa.key} and ch.bulk_bar.control.visible
            expected = (phone.chip("completed", +1), phone.chip("failed", -1))
            start = len(phone.conn.messages)
            await phone.bulk(ch, "remove_qa")
            assert await phone.confirm("Confirm Remove Failed Mark") == ["Remove failed mark from 1 chapters?"]
            note = await phone.note(1)
            await phone.settle()
            assert note.message == "Removed failed mark from 1 chapters."
            _assert_snackbar_shown(phone, note)
            _assert_refreshed_before(phone, note, start, qa.key, "✅ Completed", chip_texts=expected)
            assert note.screen["rows"][first.key] == "\U0001f504 In Progress"
            assert not note.screen["selecting"] and not ch.bulk_bar.control.visible  # the selection ended
            # more rows In progress (written by a translation meanwhile; chapter 3 never saved its HTML); bulk
            # More › Restore In Progress Status
            _set_status(real_env, "2", "in_progress")
            _set_status(real_env, "3", "in_progress")
            await phone.screen.reload_progress(force=True)
            second, third = phone.row(key=qa.key), phone.row(filename="ch003.xhtml")
            assert second.status == "in_progress" and third.status == "in_progress"
            await phone.long_press(ch.controls[first.key])
            await phone.tap(ch.controls[second.key])
            await phone.tap(ch.controls[third.key])
            assert ch.selected == {first.key, second.key, third.key}
            not_translated, failed_after = phone.chip("missing", +1), phone.chip("failed", +2)
            start = len(phone.conn.messages)
            await phone.bulk(ch, "more")
            sheet = ch.bulk_bar.more_sheet
            assert sheet is not None and sheet.dialog.open
            await phone.tap(phone.sheet_row(sheet.dialog, "bulk-restore_in_progress"))
            note = await phone.note(2)
            await phone.settle()
            assert note.message == "Successfully removed 1 not-translated placeholder(s), marked 2 as failed."
            _assert_snackbar_shown(phone, note)
            for key in (first.key, second.key):
                _assert_refreshed_before(phone, note, start, key, "❌ Failed", chip_texts=(failed_after,))
            _assert_refreshed_before(phone, note, start, third.key, "⬜ Not Translated", chip_texts=(not_translated,))
            assert not any(t.startswith("\U0001f504 In Progress") for t in note.screen["chips"])
            assert not ch.selecting
        finally:
            await phone.close()

    asyncio.run(scenario())


# ==========================================================================
# 5. Refused and failing actions say why
# ==========================================================================


@needs_flet
def test_a_refused_action_and_a_failing_write_say_why_and_keep_the_rows_true(real_env):
    _needs_cores()
    _set_status(real_env, "1", "in_progress")

    async def scenario():
        phone = await Phone(real_env).open()
        try:
            ch = phone.chapters
            # refused: More › Restore In Progress Status on a row that is not In progress
            qa = phone.row("qa_failed")
            await phone.long_press(ch.controls[qa.key])
            await phone.bulk(ch, "more")
            await phone.tap(phone.sheet_row(ch.bulk_bar.more_sheet.dialog, "bulk-restore_in_progress"))
            note = await phone.note(1)
            await phone.settle()
            assert note.message == "None of the selected chapters have 'in_progress' status."
            _assert_snackbar_shown(phone, note)
            assert phone.row_status(qa.key) == "❌ QA Failed" and not ch.loading.visible
            ch.exit_selection()
            # failing: the progress file went bad on disk after the list loaded; the write fails and says why
            row = phone.row("in_progress")
            _progress_path(real_env).write_text("{ not json", encoding="utf-8")
            sheet = await phone.open_row_menu(row.key)
            start = len(phone.conn.messages)
            await phone.tap(phone.sheet_row(sheet, "row-restore"))
            note = await phone.note(2)
            await phone.settle()
            assert note.message.startswith("Could not update progress: "), note.message
            assert len(note.message) > len("Could not update progress: "), "the error does not say why"
            _assert_snackbar_shown(phone, note)
            assert _client_got_prop(phone.conn, start, note.at, ch.loading, "visible", True)  # it tried
            assert not ch.loading.visible, "the progress bar kept running after the failure"
            assert _client_got_prop(phone.conn, start, None, ch.loading, "visible", False)
            assert phone.row_status(row.key) == "\U0001f504 In Progress", "the row claims a change that failed"
            assert len(phone.notes) == 2, [n.message for n in phone.notes]
        finally:
            await phone.close()

    asyncio.run(scenario())


# ==========================================================================
# 5b. Row menu › Retranslate this chapter (the entry goes back to Pending, then In progress while it runs)
# ==========================================================================


@needs_flet
def test_retranslate_this_chapter_from_the_row_menu_refreshes_the_row_and_shows_the_job_at_once(real_env, tmp_path):
    _needs_cores()
    from glossarion_mobile import job_kinds
    from glossarion_mobile.services.jobs import JobService

    release, entered = threading.Event(), threading.Event()
    progress_file = _progress_path(real_env)

    def single_chapter(ctx):  # the pipeline marks the chapter In progress, then translates it
        import progress_core

        def mark(prog):
            prog["chapters"]["1"]["status"] = "in_progress"

        progress_core.mutate_progress(str(progress_file), mark)
        entered.set()
        release.wait(TIMEOUT * 3)
        return {"ok": True, "outputs": []}

    def kinds(kind):
        if kind == "single_chapter":
            return job_kinds.KindInfo(kind=kind, verb="Translating chapter", icon="TRANSLATE",
                                      stop_kind="translation", run=single_chapter, resumable=False)
        return job_kinds.get_kind(kind)

    def make_jobs(dispatcher):
        return JobService(jobs_dir=str(tmp_path / "jobs"), backend=_Backend(), kinds=kinds,
                          config_loader=lambda: {}, dispatcher=dispatcher)

    async def scenario():
        phone = Phone(real_env, make_jobs)
        try:
            await phone.open()
            screen = phone.screen
            screen.did_show()
            screen.poller.stop()  # the action's own refresh, not a poll tick
            await phone.settle()
            done = phone.row("completed", filename="ch001.xhtml")
            sheet = await phone.open_row_menu(done.key)
            start = len(phone.conn.messages)
            await phone.tap(phone.sheet_row(sheet, "row-translate"))
            await phone.confirm("Retranslate chapter")
            await phone.settle()
            assert entered.wait(TIMEOUT)
            # the job shows on the Book page strip at once (the strip is this action's feedback)
            assert await _until(lambda: screen.job is not None and screen.strip_job.visible), "no sign of the job"
            # the row no longer claims the translation it just discarded
            assert _read_progress(real_env)["chapters"]["1"]["status"] in ("pending", "in_progress")
            assert phone.row_status(done.key) != "✅ Completed", \
                f"the row still shows {phone.row_status(done.key)!r} after its translation was reset"
            assert _client_got_row(phone, start, None, done.key, phone.row_status(done.key))
        finally:
            release.set()
            await phone.close()

    asyncio.run(scenario())


# ==========================================================================
# 6. Glossary Progress (the book's glossary progress manager)
# ==========================================================================


def _gp_row(tab, key):
    return next(r for r in tab.view.rows if r.key == key)


@needs_flet
def test_glossary_progress_mark_completed_remove_and_bulk_refresh_rows_counts_with_a_snackbar(real_env, monkeypatch):
    gpc = _TLU._core("glossary_progress_core", "open_glossary_progress", "mark_glossary_completed",
                     "remove_glossary_progress")
    if gpc is None:
        pytest.skip("glossary_progress_core not importable")
    _needs_cores()

    async def scenario():
        phone = await Phone(real_env).open()
        screen = phone.screen
        try:
            screen.set_tab("glossary")
            tab = screen.glossary_tab
            assert await _until(lambda: tab.view is not None and tab.view.state is not None and not tab.loading.visible)
            await phone.settle()
            chips = phone.chip_texts(tab.stats_row)
            assert phone.chip("completed", tab=tab) in chips and phone.chip("remaining", tab=tab) in chips, chips
            expected = (phone.chip("completed", +1, tab=tab), phone.chip("remaining", -1, tab=tab))
            # row ⋯ › Mark as completed
            assert _gp_row(tab, "chapter:1").status == "not_completed"
            sheet = await phone.open_row_menu("chapter:1", tab=tab)
            start = len(phone.conn.messages)
            await phone.tap(phone.sheet_row(sheet, "✅ Mark as completed"))
            note = await phone.note(1)
            await phone.settle()
            assert note.message == "Marked 1 chapter as completed"
            _assert_snackbar_shown(phone, note)
            assert note.screen["gp_rows"]["chapter:1"] == "completed" and not note.screen["gp_busy"]
            assert "✅ Completed" in note.screen["gp_screen"]["chapter:1"], note.screen["gp_screen"]["chapter:1"]
            assert all(chip in note.screen["gp_chips"] for chip in expected), (expected, note.screen["gp_chips"])
            assert _client_got_row(phone, start, note.at, "chapter:1", "✅ Completed")
            assert all(_client_got_text(phone.conn, start, note.at, chip) for chip in expected)
            # row ⋯ › Remove from progress (confirmation) on a completed chapter
            completed_now = phone.chip("completed", -1, tab=tab)
            sheet = await phone.open_row_menu("chapter:0", tab=tab)
            label = next(_key_of(t) for t in _walk(sheet) if str(_key_of(t) or "").startswith("\U0001f5d1️ Remove"))
            start = len(phone.conn.messages)
            await phone.tap(phone.sheet_row(sheet, label))
            await phone.confirm("Remove from progress", "Remove")
            note = await phone.note(2)
            await phone.settle()
            assert note.message.startswith("Removed ") and note.message.endswith(" from progress"), note.message
            _assert_snackbar_shown(phone, note)
            removed = note.screen["gp_rows"]["chapter:0"]
            assert removed != "completed", removed
            assert "✅ Completed" not in note.screen["gp_screen"]["chapter:0"], note.screen["gp_screen"]["chapter:0"]
            assert completed_now in note.screen["gp_chips"], note.screen["gp_chips"]
            assert _client_got_text(phone.conn, start, note.at, completed_now)
            # bulk: select the two chapters not completed, Mark as Completed
            todo = [r.key for r in tab.view.rows if r.kind == "chapter" and r.status != "completed"
                    and r.status != "failed"]
            assert len(todo) == 2, [(r.key, r.status) for r in tab.view.rows]
            await phone.long_press(_find_key(tab.root, todo[0]))
            await phone.tap(_find_key(tab.root, todo[1]))
            assert tab.selected == set(todo) and tab.bulk_bar.control.visible
            completed_now = phone.chip("completed", +2, tab=tab)
            start = len(phone.conn.messages)
            await phone.bulk(tab, "mark_completed")
            note = await phone.note(3)
            await phone.settle()
            assert note.message == "Marked 2 chapters as completed"
            _assert_snackbar_shown(phone, note)
            assert all(note.screen["gp_rows"][k] == "completed" for k in todo)
            assert all("✅ Completed" in note.screen["gp_screen"][k] for k in todo)
            assert not tab.selecting and not tab.bulk_bar.control.visible
            assert _client_got_text(phone.conn, start, note.at, completed_now)
            # a failing write says why (the shared core cannot save the glossary progress) and nothing changes
            target = next(r for r in tab.view.rows if r.kind == "chapter" and r.status == "failed")
            before = real_env["gp"].read_bytes()

            def cannot_save(*_a, **_kw):
                raise OSError(28, "No space left on device")

            monkeypatch.setattr(gpc, "mark_glossary_completed", cannot_save)
            sheet = await phone.open_row_menu(target.key, tab=tab)
            start = len(phone.conn.messages)
            await phone.tap(phone.sheet_row(sheet, "✅ Mark as completed"))
            note = await phone.note(4)
            await phone.settle()
            assert note.message == "Could not update the glossary progress: [Errno 28] No space left on device"
            _assert_snackbar_shown(phone, note)
            assert _client_got_prop(phone.conn, start, note.at, tab.loading, "visible", True)  # it tried
            assert not tab.loading.visible, "the progress bar kept running after the failure"
            assert _client_got_prop(phone.conn, start, None, tab.loading, "visible", False)
            assert note.screen["gp_rows"][target.key] == "failed"
            assert "✅ Completed" not in note.screen["gp_screen"][target.key]
            assert real_env["gp"].read_bytes() == before
        finally:
            await phone.close()

    asyncio.run(scenario())
