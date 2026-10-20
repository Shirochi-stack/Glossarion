"""Owner reports (phone), the chat Result card's Run options:

1. "Updating the run options still used the old setting when I clicked resume." Resume / Retry failed runs
   with exactly what the stopped card's Run options show: "Only for this run" off, the settings the panel
   just wrote (also a number typed into a field and not yet submitted when Resume is tapped); on, the
   panel's own values. Checked on the config the resumed job really starts with
   (``JobService._config_snapshot``: the config snapshot + the job's ``config_overrides``).
2. "The run options summary is not updating when I changed the settings like batch size and temperature."
   The collapsed summary follows the config while "Only for this run" is off (the panel observes the
   store for ``RUN_OPTION_KEYS``) and stops observing once released.
3. "Batch size should have a limit of 1." The batch size tile never goes below 1 (stepper, typed value).

Run from src/mobile (mobile venv):
    python -m pytest -p no:cacheprovider -W ignore tests_host/test_run_options_resume.py -q
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

MOBILE_DIR = Path(__file__).resolve().parents[1]
for _path in (MOBILE_DIR / "app", MOBILE_DIR / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TC = _load("_glossarion_chat_helpers_run_options_resume", Path(__file__).with_name("test_chat.py"))
desktop_store_cls = TC.desktop_store_cls
isolated_env = TC.isolated_env
needs_flet = TC.needs_flet


@dataclasses.dataclass
class _SettingsCtx:
    """A SettingsContext stand-in the Run options can re-bind to their overlay (``dataclasses.replace``)."""
    store: Any
    schema: Any
    page: Any = None

    def push(self, *controls: Any) -> None:
        pass

    def say(self, *args: Any, **kwargs: Any) -> None:
        pass


def _job_config(store, spec) -> dict:
    """The config a job of ``spec`` starts with (the real ``JobService._config_snapshot``)."""
    from glossarion_mobile.services.jobs import JobService

    owner = types.SimpleNamespace(config_store=store, _config_loader=None)
    try:
        return JobService._config_snapshot(owner, types.SimpleNamespace(spec=spec))
    finally:
        store.set_job_running(False)


def _resume_panel(view):
    panels = [p for k, p in view.run_panels.items() if k[1] == "resume"]
    assert len(panels) == 1, view.run_panels
    return panels[0]


async def _settle(calls):
    """``TC._settle`` without the live run's stream loop (it runs until the run ends)."""
    def waiting():
        return [t for t in calls.tasks if not t.done() and "_stream_loop" not in repr(t.get_coro())]

    while waiting():
        _done, pending = await asyncio.wait(waiting(), timeout=10)
        assert not pending, pending
    for task in calls.tasks:
        if task.done() and not task.cancelled() and task.exception() is not None:
            raise task.exception()


async def _stop(chat):
    """Stop the chat's live run (no output yet) and render its stopped Result card."""
    jobs, runs, view = chat.jobs, chat.runs, chat.view
    jobs.publish("RUNNING")
    runs.request_stop(view.cid)
    jobs.publish("CANCELLED")
    TC._finish_all(runs)
    run = runs.run_for(view.cid)
    assert run is not None and not run.live and run.state == "stopped", run and run.state
    view.render_transcript()
    return _resume_panel(view)


async def _resume(chat) -> dict:
    """Tap Resume on the card that shows the Run options; the config the resumed job starts with."""
    from glossarion_mobile.ui.chat.cards import JobCard

    view, jobs = chat.view, chat.jobs
    panel = _resume_panel(view)
    card = next(c for c in TC._cards(view, JobCard) if panel.control in c.options_box.controls)
    before = len(jobs.submitted)
    card.action_buttons["resume"].on_click(None)
    await _settle(chat.calls)
    assert len(jobs.submitted) == before + 1, chat.calls.notes
    return _job_config(chat.store, jobs.submitted[-1])


@needs_flet
def test_resume_runs_with_what_the_result_card_shows(desktop_store_cls, isolated_env):
    tmp_path = isolated_env

    async def scenario():
        chat = TC._chat_view(desktop_store_cls, tmp_path)
        view, store = chat.view, chat.store
        try:
            await _settle(chat.calls)
            store.set_many({"batch_translation": True, "batch_size": 3, "translation_temperature": 0.7})
            book = tmp_path / "novel.epub"
            book.write_bytes(b"PK\x03\x04")
            assert view.attach_file(str(book))
            view.begin_send()
            plan = chat.adapter.meta(view.cid)["pending_plan"]
            panel = view._run_panel(plan)
            # owner #2: switched off, the Plan card's summary follows the settings its tiles write
            assert panel.only is False and "Batch 3" in panel.summary_text.value
            assert panel.tiles["batch_size"].apply(2)
            assert store.get("batch_size") == 2 and "Batch 2" in panel.summary_text.value
            # Send with Run options "Only for this run": batch 10, temperature 0.3
            panel.switch.value = True
            panel._on_switch()
            assert panel.tiles["batch_size"].apply(10) and panel.tiles["translation_temperature"].apply(0.3)
            view.start_plan()
            await _settle(chat.calls)
            first = _job_config(store, chat.jobs.submitted[-1])
            assert (first["batch_size"], first["translation_temperature"]) == (10, 0.3)

            # stopped; the card's switch is off: edits change the settings and Resume runs with them
            panel = await _stop(chat)
            assert panel.only is False
            assert panel.tiles["batch_size"].apply(4) and panel.tiles["translation_temperature"].apply(0.9)
            assert store.get("batch_size") == 4 and store.get("translation_temperature") == 0.9
            assert "Batch 4" in panel.summary_text.value and "0.9" in panel.summary_text.value
            config = await _resume(chat)
            assert (config["batch_size"], config["translation_temperature"]) == (4, 0.9)

            # stopped again, a value typed into the field and Resume tapped before the field lost focus
            panel = await _stop(chat)
            panel.tiles["batch_size"].field.value = "6"
            config = await _resume(chat)
            assert config["batch_size"] == 6 and store.get("batch_size") == 6

            # stopped again; switched on: this run's values, Settings keep theirs
            panel = await _stop(chat)
            panel.switch.value = True
            panel._on_switch()
            assert panel.tiles["batch_size"].apply(8) and panel.tiles["translation_temperature"].apply(0.5)
            assert "Batch 8" in panel.summary_text.value
            config = await _resume(chat)
            assert (config["batch_size"], config["translation_temperature"]) == (8, 0.5)
            assert store.get("batch_size") == 6 and store.get("translation_temperature") == 0.9
            # every released panel stopped observing the config (only the view's own observers are left)
            assert not view.run_panels
            assert store._observers.get("batch_size", []) == []
        finally:
            TC._close(chat)

    asyncio.run(scenario())


@needs_flet
def test_resume_of_a_book_in_the_library_takes_the_cards_run_options(desktop_store_cls, isolated_env):
    """A workspace that moved into the Library resumes in the Library translate sheet: the card's "Only for
    this run" values go with it (the sheet starts switched on with them); switched off, nothing to carry."""
    tmp_path = isolated_env
    handed: list = []

    async def library_translate(folder, source="", **extra):
        handed.append((folder, extra))
        return "sheet"

    async def scenario():
        chat = TC._chat_view(desktop_store_cls, tmp_path, library_translate=library_translate)
        view = chat.view
        try:
            await _settle(chat.calls)
            item = TC._job_item(chat.adapter)
            moved = str(tmp_path / "Output" / "book")

            async def state(fn, *args):
                return (moved, False) if fn == view._job_workspace_state else fn(*args)

            view.env.run_io = state
            params = {"run": {"temp_root": str(tmp_path / "root")}, "config_overrides": {"batch_size": 7}}
            panel = view._resume_panel(item, params)
            await view._resume_turn(item)
            assert handed[-1] == (moved, {})  # switched off: the sheet reads the settings itself
            panel = view._resume_panel(item, params)
            panel.switch.value = True
            panel._on_switch()
            assert panel.tiles["batch_size"].apply(9)
            await view._resume_turn(item)
            assert handed[-1] == (moved, {"run_config": {"batch_size": 9}})
        finally:
            TC._close(chat)

    asyncio.run(scenario())


@needs_flet
def test_library_translate_sheet_starts_with_the_handed_run_options(desktop_store_cls, isolated_env, monkeypatch):
    from glossarion_mobile.ui.library import translate_sheet

    store, schema = TC._config_store(isolated_env)
    settings_ctx = _SettingsCtx(store, schema)
    monkeypatch.setattr(translate_sheet, "_settings_ctx", lambda: settings_ctx)
    raw = isolated_env / "Book.epub"
    raw.write_bytes(b"PK")
    service = types.SimpleNamespace(cfg=lambda key, default=None: store.effective(key) or default,
                                    jobs=object(), has_job_kind=lambda kind: True)
    ctx = types.SimpleNamespace(service=service, push=lambda *a: None, intents=None)
    try:
        store.set_many({"batch_translation": True, "batch_size": 3})
        sheet = translate_sheet.TranslateSheet(ctx, [{"name": "Book"}], [str(raw)], run_config={"batch_size": 9})
        panel = sheet.run_panel
        assert panel.only is True and panel.config_overrides() == {"batch_size": 9}
        assert "Batch 9" in panel.summary_text.value
        plain = translate_sheet.TranslateSheet(ctx, [{"name": "Book"}], [str(raw)])
        assert plain.run_panel.only is False and "Batch 3" in plain.run_panel.summary_text.value
        # owner #2 in the Library sheet: switched off, the summary follows the settings
        assert plain.run_panel.tiles["batch_size"].apply(4)
        assert "Batch 4" in plain.run_panel.summary_text.value
        sheet.close()
        plain.close()
        assert store._observers.get("batch_size", []) == []
    finally:
        store._saver.close()


@needs_flet
def test_run_options_summary_follows_the_config_until_released(desktop_store_cls, isolated_env):
    from glossarion_mobile.ui.chat.plan_card import RunOptionsPanel

    store, schema = TC._config_store(isolated_env)
    ctx = _SettingsCtx(store, schema)
    try:
        store.set_many({"batch_translation": True, "batch_size": 5, "translation_temperature": 0.3})
        panel = RunOptionsPanel(ctx)
        on = RunOptionsPanel(ctx, values={"batch_size": 2}, only_this_run=True)
        assert panel.summary_text.value.startswith("Batch 5 · Temp 0.3")
        # a change made elsewhere (Settings, another card) or by the panel's own tiles
        store.set("translation_temperature", 1.1)
        assert "Temp 1.1" in panel.summary_text.value and "Temp 1.1" in on.summary_text.value
        assert panel.tiles["batch_size"].apply(6) and panel.summary_text.value.startswith("Batch 6")
        assert on.summary_text.value.startswith("Batch 2")  # this run's own value wins
        store.set("contextual", True)
        store.set("use_rolling_summary", True)
        assert "Rolling summary" in panel.summary_text.value
        panel.release()
        on.release()
        store.set("batch_size", 8)
        assert panel.summary_text.value.startswith("Batch 6")
        assert all(not callbacks for callbacks in store._observers.values())
    finally:
        store._saver.close()


@needs_flet
def test_batch_size_never_goes_below_one(desktop_store_cls, isolated_env):
    """Owner #3: the batch size tile (Settings and Run options) stops at 1: the stepper, a typed 0 or -3."""
    from glossarion_mobile.ui.chat.plan_card import RunOptionsPanel
    from glossarion_mobile.ui.settings.tiles import NumberTile, make_tile

    store, schema = TC._config_store(isolated_env)
    ctx = _SettingsCtx(store, schema)
    try:
        store.set_many({"batch_translation": True, "batch_size": 1})
        settings_tile = make_tile(schema.spec("batch_size"), ctx)
        panel = RunOptionsPanel(ctx, values={"batch_size": 1}, only_this_run=True)
        for tile, read in ((settings_tile, lambda: store.get("batch_size")),
                           (panel.tiles["batch_size"], lambda: panel.values["batch_size"])):
            assert isinstance(tile, NumberTile) and tile.bounds()[0] == 1
            tile.step(-1)
            assert read() == 1
            for typed in ("0", "-3"):
                tile.field.value = typed
                tile._on_submit()
                assert read() == 1 and tile.error == "Must be at least 1"
            assert not tile.apply(0) and read() == 1
            assert tile.apply(2) and read() == 2 and tile.error is None
        panel.release()
    finally:
        store._saver.close()


def test_jobs_page_resume_of_a_chat_job_runs_with_the_settings(desktop_store_cls, tmp_path):
    """A chat job resumed outside the chat (Jobs page, launch banner: ``JobService.resume_params``) runs with
    what its stopped card shows by default: the settings' Run options, the chat's keys kept."""
    from glossarion_mobile.services.jobs import JobSpec
    from glossarion_mobile.ui.chat.run_request import DIRECT_TEXT_JOB_KIND

    jobs = TC.FakeJobService()
    jobs.resume_params = None
    adapter, runs = TC._runs(desktop_store_cls, tmp_path, jobs)
    try:
        runs.model_name = lambda: "gpt-x"
        sent = {"run": {"temp_root": "r"}, "config_overrides": {"model": "gpt-x", "batch_size": 10,
                                                                "translation_temperature": 0.3}}
        spec = JobSpec(DIRECT_TEXT_JOB_KIND, "Chat", (), sent, {"type": "chat", "cid": "5"})
        adapter.set_override("5", "model", "gpt-x")
        params = jobs.resume_params(spec)
        assert params["config_overrides"] == {"model": "gpt-x"}
        assert sent["config_overrides"]["batch_size"] == 10  # the remembered spec is not changed
    finally:
        runs.detach()
        adapter.close()


#: Global run settings the owner changes in Settings between a stop and Resume: (key, Send-time value, new value).
#: Every number / text Run option shares the bug (their tiles hold a field the panel's commit writes back).
_SETTINGS_CHANGES = (
    ("translation_temperature", 0.7, 0.5),
    ("batch_size", 3, 5),
    ("max_output_tokens", 8192, 4096),
    ("translation_history_limit", 2, 4),
    ("chapter_range", "1-3", "4-6"),
)


def _settings_write(store, schema, key, value) -> None:
    """The Settings page's own write: its schema tile for ``key`` (``settings.tiles.make_tile``)."""
    from glossarion_mobile.ui.settings.tiles import make_tile

    tile = make_tile(schema.spec(key), _SettingsCtx(store, schema))
    assert tile.apply(value), (key, tile.error)
    assert store.get(key) == value


def _shown(panel, key) -> str:
    tile = panel.tiles[key]
    field = getattr(tile, "field", None)
    return str(field.value) if field is not None else str(tile.value())


@needs_flet
def test_resume_runs_with_settings_changed_after_the_card_was_shown(desktop_store_cls, isolated_env):
    """Owner 2026-10-20: "I changed translation default temperature to 0.5, but when I jumped back to chat to
    click resume, it reset to 0.7 ... I had to create a new chat for the change to apply." The stopped card's
    Run options were built while the temperature was 0.7; Resume committed the panel and its stale field
    ("0.7", never typed into) was written back to Settings over the user's 0.5. Resume (and the panel) now
    follow the settings; the Settings value survives Resume."""
    tmp_path = isolated_env

    async def scenario():
        chat = TC._chat_view(desktop_store_cls, tmp_path)
        view, store = chat.view, chat.store
        schema = view.env.schema
        try:
            await _settle(chat.calls)
            store.set_many({"batch_translation": True, **{k: old for k, old, _new in _SETTINGS_CHANGES}})
            book = tmp_path / "novel.epub"
            book.write_bytes(b"PK\x03\x04")
            assert view.attach_file(str(book))
            view.begin_send()
            view.start_plan()
            await _settle(chat.calls)
            first = _job_config(store, chat.jobs.submitted[-1])
            assert all(first[k] == old for k, old, _new in _SETTINGS_CHANGES)

            panel = await _stop(chat)  # the card (and its Run options) shown with the Send-time settings
            assert panel.only is False and _shown(panel, "translation_temperature") == "0.7"
            for key, _old, new in _SETTINGS_CHANGES:  # the user goes to Settings and changes them there
                _settings_write(store, schema, key, new)
            # back in the chat: the card's Run options show the settings now
            for key, _old, new in _SETTINGS_CHANGES:
                assert _shown(panel, key) == str(new), key
            assert "Temp 0.5" in panel.summary_text.value and "Batch 5" in panel.summary_text.value
            config = await _resume(chat)
            for key, _old, new in _SETTINGS_CHANGES:
                assert config[key] == new, (key, config[key])
                assert store.get(key) == new, (key, store.get(key))  # Resume did not write the old value back

            # an explicit per-run value still wins: "Only for this run" on the next stopped card
            panel = await _stop(chat)
            panel.switch.value = True
            panel._on_switch()
            assert panel.tiles["translation_temperature"].apply(1.1)
            _settings_write(store, schema, "translation_temperature", 0.2)  # Settings change meanwhile
            config = await _resume(chat)
            assert config["translation_temperature"] == 1.1 and store.get("translation_temperature") == 0.2

            # a value typed into the card's field (not submitted yet) still goes with Resume
            panel = await _stop(chat)
            _settings_write(store, schema, "batch_size", 7)
            panel.tiles["batch_size"].field.value = "9"
            config = await _resume(chat)
            assert config["batch_size"] == 9 and store.get("batch_size") == 9
            assert config["translation_temperature"] == 0.2  # the settings now (last card was off)
        finally:
            TC._close(chat)

    asyncio.run(scenario())


@needs_flet
def test_plan_start_runs_with_settings_changed_after_the_plan_was_shown(desktop_store_cls, isolated_env):
    """The same stale-field write-back on the Plan card: a Settings change made while the plan waits is what
    Start runs with, and Start does not put the old value back into Settings."""
    tmp_path = isolated_env

    async def scenario():
        chat = TC._chat_view(desktop_store_cls, tmp_path)
        view, store = chat.view, chat.store
        schema = view.env.schema
        try:
            await _settle(chat.calls)
            store.set_many({"batch_translation": True, "batch_size": 3, "translation_temperature": 0.7})
            book = tmp_path / "novel.epub"
            book.write_bytes(b"PK\x03\x04")
            assert view.attach_file(str(book))
            view.begin_send()
            panel = view._run_panel(chat.adapter.meta(view.cid)["pending_plan"])
            assert _shown(panel, "translation_temperature") == "0.7"
            _settings_write(store, schema, "translation_temperature", 0.5)
            _settings_write(store, schema, "batch_size", 6)
            assert _shown(panel, "translation_temperature") == "0.5" and _shown(panel, "batch_size") == "6"
            view.start_plan()
            await _settle(chat.calls)
            config = _job_config(store, chat.jobs.submitted[-1])
            assert (config["translation_temperature"], config["batch_size"]) == (0.5, 6)
            assert (store.get("translation_temperature"), store.get("batch_size")) == (0.5, 6)
        finally:
            TC._close(chat)

    asyncio.run(scenario())


@needs_flet
def test_run_options_commit_saves_only_typed_fields(desktop_store_cls, isolated_env):
    """``RunOptionsPanel.commit`` (Start / Resume) saves a number typed and not submitted yet, never a field
    the user did not touch (its text is what the panel last showed), whatever changed in Settings since."""
    from glossarion_mobile.ui.chat.plan_card import RunOptionsPanel

    store, schema = TC._config_store(isolated_env)
    ctx = _SettingsCtx(store, schema)
    try:
        store.set_many({"batch_translation": True, "batch_size": 3, "translation_temperature": 0.7})
        panel = RunOptionsPanel(ctx)
        panel.release()  # no observer: the fields cannot follow the store, commit still must not write them
        store.set_many({"batch_size": 5, "translation_temperature": 0.5})
        panel.tiles["batch_size"].field.value = "8"
        panel.commit()
        assert store.get("translation_temperature") == 0.5 and store.get("batch_size") == 8
    finally:
        store._saver.close()
