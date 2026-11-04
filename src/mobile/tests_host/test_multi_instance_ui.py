"""Host tests for the mobile multi-instance UI (design §6 "UI" bullet).

* Multi-Key Manager: every key card starts with its instance ("🔢2"), the KeyEditor edits it (1..99, the field is
  stored only when it is not 1, like ``APIKeyEntry.to_dict``), the selection ⋯ "🔢 Set instance…" sets it on the
  selected keys, and the ⏰ schedule sheet's title names the keys' instance.
* Library "Load N for translation" -> TranslateSheet: with the Translation keys on 2+ instances each book row gets an
  instance dropdown prefilled from ``instance_file_map`` (else round-robin); Start saves the choice and queues one
  ``translate`` job per instance with only its books (``instance=N``); a single-instance config keeps the one job of
  before (no picker, no ``instance``).
* Jobs page: one Running card per running job, each with its own Stop; the strip shows one job plus "+N running",
  and its Stop / Open act on the job it shows; the Book page and the job page see a job of another instance.

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_multi_instance_ui.py
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
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")
os.environ.setdefault("GLOSSARION_HEADLESS_KEY_MANAGER", "1")

from glossarion_mobile.services.jobs import INSTANCE_PARAM, JobSnapshot, JobSpec, JobState, JobsView  # noqa: E402
from glossarion_mobile.services.library import LibraryService, SharedCore  # noqa: E402
from glossarion_mobile.state.config_store import MobileConfigStore  # noqa: E402
from glossarion_mobile.ui.router import parse_route  # noqa: E402


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")

_TB_SPEC = importlib.util.spec_from_file_location("_glossarion_tb_helpers_multi_instance_ui",
                                                  Path(__file__).with_name("test_bootstrap.py"))
_TB = importlib.util.module_from_spec(_TB_SPEC)
_TB_SPEC.loader.exec_module(_TB)


def _store(tmp_path, data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    store = MobileConfigStore(path, debounce=30,
                              reader=lambda p, decrypt=True: json.loads(Path(p).read_text(encoding="utf-8")),
                              writer=lambda disk, p, backup=False: Path(p).write_text(json.dumps(disk), encoding="utf-8"))
    store.load()
    return store


def _settings_ctx(store, page):
    from glossarion_mobile.ui.settings.context import SettingsContext
    from glossarion_mobile.ui.settings.schema_access import SchemaAccess

    return SettingsContext(page=page, store=store, schema=SchemaAccess())


def _texts(control, out=None):
    """Every ``Text.value`` / string label under a control (dataclass walk)."""
    import flet as ft

    out = out if out is not None else []
    if control is None:
        return out
    if isinstance(control, ft.Text) and control.value:
        out.append(str(control.value))
    for name in ("content", "controls", "label", "leading", "title", "subtitle", "trailing"):
        value = getattr(control, name, None)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, ft.BaseControl):
                    _texts(item, out)
        elif isinstance(value, ft.BaseControl):
            _texts(value, out)
        elif isinstance(value, str) and name in ("content", "label"):
            out.append(value)
    return out


TWO_INSTANCES = [
    {"api_key": "sk-one-aaaaaaaa1111", "model": "gpt-6", "enabled": True},
    {"api_key": "sk-two-bbbbbbbb2222", "model": "gpt-6", "enabled": True, "instance": 2},
]


# ==========================================================================
# Multi-Key Manager
# ==========================================================================


def test_key_instance_helpers_follow_instance_plan():
    from glossarion_mobile.ui.screens.keys import instance_text, key_instance_number, set_instance_field

    assert key_instance_number({}) == 1 and key_instance_number({"instance": "3"}) == 3
    assert key_instance_number({"instance": 0}) == 1 and key_instance_number({"instance": "x"}) == 1
    assert instance_text([{"instance": 2}]) == "🔢2" and instance_text([{}, {"instance": 2}, {}]) == "🔢1, 2"
    assert set_instance_field({"instance": 4}, 1) == {}  # 1 is the missing field (APIKeyEntry.to_dict)
    assert set_instance_field({}, "2") == {"instance": 2}


def test_bulk_instance_writes_only_non_default_numbers(tmp_path):
    from glossarion_mobile.ui.screens.keys import KeyBackend, KeysController, parse_bulk_value

    store = _store(tmp_path, {"use_multi_api_keys": True, "multi_api_keys": [dict(k) for k in TWO_INSTANCES],
                              "glossary_keys": [{"api_key": "g", "model": "m"}]})
    keys = KeysController(store, KeyBackend())
    assert parse_bulk_value("instance", "0")[1] and parse_bulk_value("instance", "100")[1]
    assert parse_bulk_value("instance", "x")[1] == "Enter a whole number"
    values, error = parse_bulk_value("instance", "3")
    assert error is None and values == {"instance": 3}
    assert keys.set_key_fields("main", [0, 1], values) == (2, [])
    assert [k.get("instance") for k in keys.keys("main")] == [3, 3]
    assert keys.set_key_fields("main", [1], {"instance": 1}) == (1, [])
    assert "instance" not in keys.keys("main")[1]
    # every pool's keys carry an instance (design §6b)
    assert keys.set_key_fields("glossary", [0], {"instance": 2}) == (1, [])
    assert keys.keys("glossary")[0]["instance"] == 2
    store._saver.close()


@needs_flet
def test_keys_screen_shows_and_edits_each_keys_instance(tmp_path):
    from glossarion_mobile.ui.screens.keys import (
        INSTANCE_HELP,
        BulkFieldSheet,
        KeyBackend,
        KeysController,
        KeysScreen,
    )

    async def scenario():
        conn, session = _TB._fake_session("android")
        page = session.page
        store = _store(tmp_path, {"use_multi_api_keys": True, "multi_api_keys": [dict(k) for k in TWO_INSTANCES]})
        keys = KeysController(store, KeyBackend(), clock=lambda: 1234.5)
        screen = KeysScreen(parse_route("/settings/keys/translation"), controller=keys,
                            ctx=_settings_ctx(store, page), page=page, run_io=lambda fn, *a: asyncio.to_thread(fn, *a),
                            spawn=lambda c: asyncio.ensure_future(c))
        page.views[0].controls.append(screen.get_body())
        page.update()
        cards = [_texts(card) for card in screen.cards]
        assert any(text.startswith("🔢1 · gpt-6") for text in cards[0])
        assert any(text.startswith("🔢2 · gpt-6") for text in cards[1])

        # the KeyEditor's Instance field (the desktop Instance column's editor)
        editor = screen.open_editor(0)
        assert editor.instance_field.value == "1" and INSTANCE_HELP in _texts(editor.body)
        editor.instance_field.value = "0"
        assert not editor.save() and "between 1 and 99" in editor.error_text.value
        editor.instance_field.value = "100"
        assert not editor.save()
        editor.instance_field.value = "3"
        assert editor.save()
        assert keys.keys("main")[0]["instance"] == 3
        editor = screen.open_editor(0)
        assert editor.instance_field.value == "3"
        editor.instance_field.value = "1"
        assert editor.save()
        assert "instance" not in keys.keys("main")[0]  # back to the default: the field is gone again

        # selection ⋯ "🔢 Set instance…" (desktop shared key menu "🔢 Set Instance")
        screen.toggle_select(0)
        screen.toggle_select(1)
        more = screen.open_bulk_more()
        item = next(i for i in more.items if i.label == "🔢 Set instance…")
        item.on_select()
        sheet = screen.last_sheet
        assert isinstance(sheet, BulkFieldSheet) and sheet.input.value == "1"  # the first key's instance
        assert "🔢 Instance · 2 keys" in _texts(sheet.sheet)
        sheet.input.value = "2"
        assert sheet.apply() is None
        assert [k.get("instance") for k in keys.keys("main")] == [2, 2]
        assert all(any(t.startswith("🔢2 · ") for t in _texts(card)) for card in screen.cards)

        # the ⏰ schedule sheet names the keys' instance
        schedule = screen.open_schedule_sheet()
        assert schedule.title.endswith("2 keys · 🔢2")
        screen.clear_selection()
        screen.toggle_select(1)
        assert screen.open_schedule_sheet().title.endswith("gpt-6 · 🔢2")
        # every key back on instance 1 (a single-instance user's pool): no 🔢 on the cards or the schedule sheet
        screen.toggle_select(0)
        assert screen.apply_bulk_fields([0, 1], {"instance": 1}) is None
        assert not any("🔢" in t for card in screen.cards for t in _texts(card))
        assert screen.open_schedule_sheet().title.endswith("· 2 keys")
        screen.dispose()
        store._saver.close()

    asyncio.run(scenario())


@pytest.mark.skipif(not _has("multi_api_key_manager"), reason="multi_api_key_manager not importable")
def test_real_key_service_keeps_the_instance_on_save():
    """``key_pool_service.validate_entry`` (the KeyEditor's save) keeps a Translation key's instance through
    ``APIKeyEntry`` and the other pools' instance as stored."""
    from glossarion_mobile.ui.screens.keys import KeyBackend

    backend = KeyBackend()
    clean, error = backend.validate({"api_key": "k", "model": "m", "instance": 2}, "main")
    assert error is None and clean["instance"] == 2
    clean, error = backend.validate({"api_key": "k", "model": "m"}, "main")
    assert error is None and "instance" not in clean
    clean, error = backend.validate({"api_key": "k", "model": "m", "instance": 3}, "fallback")
    assert error is None and clean["instance"] == 3


# ==========================================================================
# TranslateSheet
# ==========================================================================


class FakePrefs:
    def __init__(self):
        self.refs = {}

    def get(self, key, default=None):
        return default

    def set(self, key, value):
        pass

    def library_id(self, identity):
        return self.refs.setdefault(identity, f"{len(self.refs):012d}")


class Jobs:
    """``JobsFeature`` as the Library sees it: ``submit`` / ``submit_instances``."""

    def __init__(self, *, many=True):
        self.submitted = []  # (spec, instance or None)
        if not many:
            self.submit_instances = None

    def has_kind(self, kind):
        return kind == "translate"

    async def submit(self, spec, **kwargs):
        self.submitted.append((spec, kwargs.get("instance")))
        return f"job{len(self.submitted)}"

    async def submit_instances(self, pairs):
        out = []
        for number, spec in pairs:
            out.append(await self.submit(spec, instance=number))
        return out


def _books(tmp_path, count=3):
    raws = []
    for index in range(count):
        raw = tmp_path / "Raw" / f"Book{index + 1}.epub"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(b"PK")
        raws.append(str(raw))
    books = [{"name": f"Book {i + 1}", "raw_source_path": raw, "path": raw, "state": "not_started"}
             for i, raw in enumerate(raws)]
    return books, raws


def _library_ctx(page, service):
    from glossarion_mobile.ui.library.common import LibraryContext

    notes = []
    ctx = LibraryContext(service=service, page=page, navigate=lambda *a, **k: None,
                         notify=lambda message, action=None, on_action=None: notes.append(message),
                         prefs=service.prefs, push_overlay=lambda view: None, pop_overlay=lambda: None,
                         platform="android")
    ctx.notes = notes
    return ctx


@needs_flet
def test_translate_sheet_assigns_books_to_instances_and_starts_one_job_each(tmp_path):
    from glossarion_mobile.ui.library.translate_sheet import TranslateSheet

    books, raws = _books(tmp_path)
    jobs = Jobs()
    config = {"model": "authgpt/gpt-6-luna", "output_language": "English", "use_multi_api_keys": True,
              "multi_api_keys": [dict(k) for k in TWO_INSTANCES],
              # Book 3 ran in instance 2 last time; a saved instance that no key has any more is not kept
              "instance_file_map": {os.path.abspath(raws[2]): 2, os.path.abspath(raws[0]): 7,
                                    os.path.abspath(str(tmp_path / "other.epub")): 2}}
    service = LibraryService(core=SharedCore({"library_core": types.ModuleType("library_core")}),
                             prefs=FakePrefs(), jobs=jobs, config=config)

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _library_ctx(session.page, service)
        sheet = TranslateSheet(ctx, books, raws)
        sheet.show(session.page)
        assert sheet.instances == [1, 2]
        # round-robin over the books without a (valid) saved instance, the saved one kept
        assert sheet.assignment == {raws[0]: 1, raws[1]: 2, raws[2]: 2}
        assert [sheet.instance_pickers[r].value for r in raws] == ["1", "2", "2"]
        assert [o.text for o in sheet.instance_pickers[raws[0]].options] == ["Instance 1", "Instance 2"]
        texts = _texts(sheet.sheet.content)
        assert "2 parallel jobs · Instance 1: 1 book · Instance 2: 2 books" in texts
        assert sheet.start_button.content == "Start"
        # the dropdown moves Book 2 to instance 1
        picker = sheet.instance_pickers[raws[1]]
        picker.value = "1"
        picker.on_select(types.SimpleNamespace(control=picker))
        assert sheet.assignment[raws[1]] == 1
        assert sheet.instance_text.value.endswith("Instance 1: 2 books · Instance 2: 1 book")
        assert sheet.set_instance(raws[1], "5") == 1  # an instance without keys is refused
        job = await sheet.start()
        assert job == "job1" and sheet.started_jobs == ["job1", "job2"]
        (first, one), (second, two) = jobs.submitted
        assert (one, two) == (1, 2)
        assert first.kind == second.kind == "translate"
        assert first.inputs == (raws[0], raws[1]) and second.inputs == (raws[2],)
        assert first.title == "Book 1 +1" and second.title == "Book 3"
        assert second.origin["type"] == "library" and second.origin.get("bid")  # one book: its Book page
        saved = config["instance_file_map"]
        assert saved[os.path.abspath(raws[0])] == 1 and saved[os.path.abspath(raws[1])] == 1
        assert saved[os.path.abspath(raws[2])] == 2
        assert saved[os.path.abspath(str(tmp_path / "other.epub"))] == 2  # merged, like the desktop's Run

        # reopened: the saved choice is the prefill
        again = TranslateSheet(ctx, books, raws)
        assert again.assignment == {raws[0]: 1, raws[1]: 1, raws[2]: 2}
        # every book on one instance: one job on that instance's keys
        for raw in raws:
            again.set_instance(raw, 2)
        assert again.instance_text.value == "1 job on Instance 2's keys · Instance 2: 3 books"
        jobs.submitted.clear()
        await again.start()
        assert [(spec.inputs, number) for spec, number in jobs.submitted] == [(tuple(raws), 2)]

    asyncio.run(scenario())


@needs_flet
def test_translate_sheet_single_instance_keeps_one_plain_job(tmp_path):
    from glossarion_mobile.ui.library.translate_sheet import TranslateSheet

    books, raws = _books(tmp_path, 2)

    async def run(config):
        jobs = Jobs(many=False)
        service = LibraryService(core=SharedCore({"library_core": types.ModuleType("library_core")}),
                                 prefs=FakePrefs(), jobs=jobs, config=config)
        conn, session = _TB._fake_session("android")
        sheet = TranslateSheet(_library_ctx(session.page, service), books, raws)
        sheet.show(session.page)
        assert not sheet.assignment and not sheet.instance_pickers and not sheet.instance_text.visible
        assert sheet.start_button.content == "Start"
        assert await sheet.start() == "job1"
        assert [(spec.inputs, number) for spec, number in jobs.submitted] == [(tuple(raws), None)]
        assert "instance_file_map" not in config

    base = {"model": "authgpt/gpt-6-luna", "output_language": "English"}
    # keys on two instances but the Translation pool switched off; one enabled instance; no multi keys at all
    asyncio.run(run(dict(base, use_multi_api_keys=False, multi_api_keys=[dict(k) for k in TWO_INSTANCES])))
    asyncio.run(run(dict(base, use_multi_api_keys=True,
                         multi_api_keys=[TWO_INSTANCES[0], dict(TWO_INSTANCES[1], enabled=False)])))
    asyncio.run(run(dict(base)))


def test_library_service_splits_a_run_per_instance_without_the_jobs_feature(tmp_path):
    """``translate_instance_specs`` + ``submit_instances`` with a jobs service that has only ``submit``."""
    books, raws = _books(tmp_path, 3)

    class PlainJobs:
        def __init__(self):
            self.calls = []

        def submit(self, spec, instance=None):
            self.calls.append((spec.inputs, instance))
            return f"id{len(self.calls)}"

    jobs = PlainJobs()
    service = LibraryService(core=SharedCore({"library_core": types.ModuleType("library_core")}),
                             prefs=FakePrefs(), jobs=jobs, config={})
    pairs = service.translate_instance_specs(books, sources=raws, assignment={raws[0]: 2, raws[1]: 1, raws[2]: 2},
                                             review_glossary=True, config_overrides={"model": "m"})
    assert [(n, spec.inputs) for n, spec in pairs] == [(1, (raws[1],)), (2, (raws[0], raws[2]))]
    assert all(spec.params == {"review_glossary": True, "config_overrides": {"model": "m"}} for _n, spec in pairs)
    assert asyncio.run(service.submit_instances(pairs)) == ["id1", "id2"]
    assert jobs.calls == [((raws[1],), 1), ((raws[0], raws[2]), 2)]
    assert service.instance_assignment(raws) is None  # no multi keys: single instance


# ==========================================================================
# Jobs page, strip, Book page, job page
# ==========================================================================


def _snap(job_id, title, *, instance=None, state=JobState.RUNNING, origin=None, inputs=None):
    params = {INSTANCE_PARAM: instance} if instance is not None else {}
    spec = JobSpec("translate", title, tuple(inputs or (f"/raw/{job_id}.epub",)), params=params,
                   origin=origin or {"type": "library", "label": f"Library · {title}"})
    return JobSnapshot(id=job_id, spec=spec, state=state, created=1.0, started=2.0)


class FakeJobService:
    def __init__(self, view):
        self._view = view
        self.stopped = []
        self.paused = False

    def view(self):
        return self._view

    def subscribe(self, callback):
        return lambda: None

    def snapshot(self, job_id=None):
        if job_id is None:
            return self._view.instance_job(1)
        return self._view.find(job_id)

    def request_stop(self, job_id=None, *, force=False, **kwargs):
        self.stopped.append(job_id)
        return "graceful"

    def device_row(self, job_id=None, *, threads=False):
        return f"📱 row of {job_id}"

    def log_buffer(self, job_id):
        return None

    def read_log_tail(self, job_id):
        return []

    def request_segments(self, job_id):
        return []

    def has_request_stream(self, job_id):
        return True


def _two_running():
    one = _snap("job000000001", "Book A", instance=1)
    two = _snap("job000000002", "Book B", instance=2)
    queued = _snap("job000000003", "Book C", instance=2, state=JobState.QUEUED)
    return JobsView(active=one, running=(one, two), queue=(queued,)), one, two


@needs_flet
def test_jobs_page_lists_a_running_card_with_its_own_stop_per_instance():
    from glossarion_mobile.ui.screens.jobs import JobsScreen

    view, one, two = _two_running()
    service = FakeJobService(view)
    screen = JobsScreen(parse_route("/jobs"), service=service)
    screen.get_body()
    keys = [getattr(c, "key", None) for c in screen.list_view.controls]
    assert f"job-row-{one.id}" in keys and f"job-row-{two.id}" in keys
    assert keys.index(f"job-row-{one.id}") < keys.index(f"job-row-{two.id}") < keys.index("jobs-queue")
    texts = _texts(screen.list_view)
    assert "Running (2)" in texts and "Library · Book B · Instance 2" in texts
    assert "Library · Book C · Instance 2" in _texts(screen.queue_list)
    rows = {c.key: c for c in screen.list_view.controls if getattr(c, "key", None)}

    def stop_button(row, job_id):
        found = []

        def walk(control):
            if getattr(control, "key", None) == f"job-stop-{job_id}":
                found.append(control)
            for name in ("content", "controls"):
                value = getattr(control, name, None)
                for child in (value if isinstance(value, list) else [value]):
                    if child is not None and hasattr(child, "key"):
                        walk(child)

        walk(row)
        return found[0]

    stop_button(rows[f"job-row-{two.id}"], two.id).on_click(None)
    stop_button(rows[f"job-row-{one.id}"], one.id).on_click(None)
    assert service.stopped == [two.id, one.id]

    # one plain job (no multi-instance run): as before, no count, no instance in its row
    plain = _snap("job000000009", "Solo")
    screen._render(JobsView(active=plain, running=(plain,)))
    texts = _texts(screen.list_view)
    assert "Running" in texts and not any("Instance" in t for t in texts)
    # only another instance running (instance 1 idle): it still has its card
    screen._render(JobsView(active=two, running=(two,)))
    assert f"job-row-{two.id}" in [getattr(c, "key", None) for c in screen.list_view.controls]


@needs_flet
def test_strip_shows_one_job_plus_n_running_and_acts_on_the_job_it_shows(tmp_path):
    from glossarion_mobile.state.app_state import AppState, JobStripModel
    from glossarion_mobile.ui.screens.jobs import JobsFeature
    from glossarion_mobile.ui.shell.job_strip import JobStrip

    view, one, two = _two_running()
    service = FakeJobService(view)
    paths = types.SimpleNamespace(data=str(tmp_path), logs=str(tmp_path / "logs"), cache=None, temp=None,
                                  output=str(tmp_path / "out"), library=str(tmp_path / "lib"), docs=None)
    navigated = []
    app = types.SimpleNamespace(page=None, dispatcher=None, state=AppState(), prefs=None, paths=paths,
                                navigate_to=lambda name, params=None, query=None: navigated.append((name, params)))
    feature = JobsFeature(app, service=service)
    model = feature._strip_model(view)
    assert model.job_id == one.id and model.running_more == 1 and model.queued == 1
    app.state.job_strip.set(model)
    strip = JobStrip()
    strip.set_model(model)
    assert strip.running_badge.visible and strip.running_text.value == "+1 running"
    assert strip.stop_button.badge is not None  # the queued "+1" stays on Stop
    assert "1 more job running" in strip.semantics.label
    feature.stop_from_strip()
    assert service.stopped == [one.id]
    feature.open_strip_job()
    assert navigated[-1] == ("jobs", None)  # several running: the Jobs page with every card

    # instance 1 idle, instance 2 running: the strip shows (and stops) instance 2's job
    only_two = JobsView(active=two, running=(two,))
    service._view = only_two
    model = feature._strip_model(only_two)
    assert model.job_id == two.id and model.running_more == 0
    app.state.job_strip.set(model)
    strip.set_model(model)
    assert not strip.running_badge.visible
    feature.stop_from_strip()
    assert service.stopped[-1] == two.id
    feature.open_strip_job()
    assert navigated[-1] == ("jobs.detail", {"jid": two.id})
    strip.set_model(JobStripModel("Done · Book", "2/2 chapters", 1.0, state="done", running_more=2))
    assert not strip.running_badge.visible  # an ended job's strip has no running badge


def test_chat_strip_stop_stops_the_job_the_strip_shows():
    from glossarion_mobile.ui.chat.chat_view import ChatView

    view, one, two = _two_running()
    service = FakeJobService(JobsView(active=two, running=(two,)))
    notes = []
    stub = types.SimpleNamespace(env=types.SimpleNamespace(jobs=service),
                                 job_strip=types.SimpleNamespace(model=types.SimpleNamespace(job_id=two.id)),
                                 notify=notes.append)
    ChatView._on_strip_stop(stub)
    assert service.stopped == [two.id] and not notes
    stub.job_strip.model = None  # no strip job: instance 1's, as before
    service._view = JobsView()
    ChatView._on_strip_stop(stub)
    assert notes == ["No job is running"]


def test_book_page_sees_a_book_running_in_another_instance():
    from glossarion_mobile.ui.library.book_page import RUN_WAIT_TRANSLATION, job_for_book, run_wait_reason

    book_b = {"name": "Book B", "raw_source_path": "/raw/job000000002.epub"}
    view, one, two = _two_running()
    assert job_for_book(view, "nobid", book_b) is two  # instance 2's running job
    # instance 1 idle while instance 2 runs another book: this book's Run translation does not wait
    other = {"name": "Other", "raw_source_path": "/raw/other.epub"}
    assert run_wait_reason(JobsView(active=two, running=(two,)), "x", other) is None
    assert run_wait_reason(view, "x", other) == RUN_WAIT_TRANSLATION  # instance 1 busy: it queues behind it
    # a view of a service without instances: its active job, as before
    legacy = types.SimpleNamespace(active=one, queue=())
    assert run_wait_reason(legacy, "x", other) == RUN_WAIT_TRANSLATION


@needs_flet
def test_job_page_names_the_instance_and_reads_its_own_device_row():
    from glossarion_mobile.ui.screens.job_detail import JobDetailScreen

    view, one, two = _two_running()
    service = FakeJobService(view)
    detail = JobDetailScreen(parse_route(f"/jobs/{two.id}"), service=service)
    detail.get_body()
    assert detail.instance_chip.visible and _texts(detail.instance_chip.label) == ["Instance 2"]
    assert detail.device_row() == f"📱 row of {two.id}"
    plain = _snap("job000000009", "Solo")
    service._view = JobsView(active=plain, running=(plain,))
    solo = JobDetailScreen(parse_route(f"/jobs/{plain.id}"), service=service)
    solo.get_body()
    assert not solo.instance_chip.visible
