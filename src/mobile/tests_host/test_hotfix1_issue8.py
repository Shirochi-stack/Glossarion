"""Acceptance test for the owner's hotfix 1 report, issue 8 (current APK, 2026-10-25):
"Response card run options are missing the api call delay variable."

What the owner saw: the chat's Plan card ("Ready to translate") and the stopped turn's Result card (Resume) list
"Run options" (batch, temperature, chunk size, ...) but not the desktop main window's "API call delay (s, or range
5-7)", so a phone user could not slow one run down (a rate-limited key) without changing the setting for every run.

This file proves the fix end to end on the REAL app objects, as the other tests_host acceptance tests do: the app
started on a fake Flet session (``test_ui_flows._host_driver`` / ``test_ui_foundations._start``), the real
JobService, ``HeadlessOwner`` + the shared translation pipeline (``run_env`` exports the desktop
SEND_INTERVAL_SECONDS / SEND_INTERVAL_MAX_SECONDS, ``unified_api_client`` paces the calls), the real ChatView /
ChatRuns / RunOptionsPanel and settings tiles. Only the model is the offline fake OpenAI server on 127.0.0.1
(``diagnostics.fake_llm_server``); the E2E network guard refuses every other connection. The owner's taps go
through the UI tree: New chat, ＋ › Files, Send, the Plan card's Run options (expand, "Only for this run", the
delay field), Start; the running card's Stop, the Result card's Run options and Resume.

The global API call delay is the range 0-0.1 s (config ``delay`` 0.0 + ``delay_max`` 0.1). A recorder on the fake
model's ``on_request`` hook reads the process environment at the moment each request arrives (the job's env, set
by the shared pipeline while the job holds JOB_LOCK), so every check is about what the run really used.

1. ``test_plan_card_api_call_delay_is_this_runs_only``: the Plan card's Run options show the delay tile with the
   desktop label (``settings_schema_data`` / translator_gui QLabel) and range hint, right after the chunk size,
   showing the setting ("0-0.1") and the collapsed line "Delay 0-0.1 s". With "Only for this run" on, "0.4-0.6"
   (Done on the keyboard) shows in the summary while the setting keeps 0.0 / 0.1 (store and config.json). Start:
   the job's ``config_overrides`` carry ``delay`` 0.4 / ``delay_max`` 0.6, every request of the run arrives with
   SEND_INTERVAL_SECONDS=0.4 / SEND_INTERVAL_MAX_SECONDS=0.6, and the calls really are paced by it (the median
   arrival spacing reaches 0.4 s; ``_paced``). After the run the setting is unchanged, the
   process env is back to what it was, and the next attachment's Plan card starts from the setting again (switch
   off, "0-0.1"); its run sends with 0.0 / 0.1. (Its spacing is printed next to the paced run's, not asserted: on
   the setting the pipeline's own speed sets it, ~0.25 s per chapter on the dev box, slower on a slow runner.)
2. ``test_result_card_api_call_delay_reaches_the_resumed_run_only``: a run started on the setting (0.0 / 0.1 in
   its requests) is stopped mid-book (the fake model's own reply hook holds it, ``HoldGate``). The Result card's
   Run options show the same tile ("0-0.1"); "Only for this run" on, "0.5" typed and Resume tapped straight away
   (the field never lost focus): the resumed job's overrides are 0.5 / '' (a fixed delay clears the setting's
   range max), its requests arrive with 0.5 / '' and are paced by 0.5 s; the setting stays 0.0 / 0.1.

Real data stays untouched: the app runs in temp FLET_APP_STORAGE_* dirs (HOME, OUTPUT_DIRECTORY,
GLOSSARION_LIBRARY_DIR, GLOSSARION_DATA_DIR and CONFIG_FILE point there; asserted), USERPROFILE / APPDATA /
LOCALAPPDATA point into tmp_path and HTTP logging is off (``test_devfix_issue12.iso``). A failing scenario
force-stops its jobs before the app closes.

Run from src/mobile with the mobile env (Windows dev box: ``.venv``, Python 3.12, keep the shell PYTHONPATH)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_hotfix1_issue8.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
TESTS_DIR = MOBILE_DIR / "tests"
for entry in (str(APP_DIR), str(SRC_DIR), str(TESTS_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


_NEEDED = ("flet", "msgpack", "ebooklib", "openai", "httpx", "tiktoken", "bs4", "lxml")
pytestmark = pytest.mark.skipif(not all(_has(m) for m in _NEEDED),
                                reason=f"needs {', '.join(_NEEDED)} (the project venv)")


def _load(alias: str, file_name: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(file_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# the issue-12 acceptance test's real-app harness: iso / offline fixtures, Probe, HoldGate, _settled, _end_job
D12 = _load("_glossarion_devfix12_hotfix1_issue8", "test_devfix_issue12.py")
_UF = D12._UF
storage = D12.storage
app_env = D12.app_env
iso = D12.iso
offline = D12.offline

#: The global API call delay of these tests: the range 0-0.1 s (fast, and a range, so a fixed per-run value
#: must clear its max).
GLOBAL = {"delay": 0.0, "delay_max": 0.1}
GLOBAL_TEXT = "0-0.1"
GLOBAL_ENV = ("0.0", "0.1")
RUN_TIMEOUT = 180.0
#: The client spaces the run's calls by its delay (each call takes the next slot, ``_apply_api_call_stagger``, whose
#: "Queued ... Sending API call in 0.30s" lines step by the delay). The fake model stamps a call when its handler has
#: read the request, after what the client does past its slot: some 30 ms of jitter (the loopback hop, the GIL), and
#: a fresh API thread first builds its HTTP client (~0.3-0.6 s on this box), so the first two calls of a run can
#: arrive together (seen: chapters 5 and 6 of a resumed run 3 ms apart, their slots 0.3 s apart). The median spacing
#: of a run's chapter calls (11 of them here) is the delay; on the global 0-0.1 s it is a few hundredths.
SPACING_SLACK = 0.05
CHAPTER_HEADING = re.compile(r"제\s*(\d+)\s*장")


# ==========================================================================
# Helpers
# ==========================================================================


class EnvTap:
    """The fake model's ``on_request`` hook: the API call delay env each translation request arrived with."""

    def __init__(self, server) -> None:
        self._lock = threading.Lock()
        self.rows: dict = {}
        server.on_request.append(self)

    def __call__(self, record) -> None:
        if record.kind != "translation":
            return
        with self._lock:
            self.rows[record.id] = (os.environ.get("SEND_INTERVAL_SECONDS"),
                                    os.environ.get("SEND_INTERVAL_MAX_SECONDS"))

    def envs(self, records) -> list:
        with self._lock:
            return [self.rows.get(r.id) for r in records]


def _global(store) -> tuple:
    return store.get("delay"), store.get("delay_max")


def _config_file_delay() -> tuple:
    """``delay`` / ``delay_max`` in the app's config.json on disk (CONFIG_FILE, inside the test's tmp dir)."""
    path = os.environ.get("CONFIG_FILE") or ""
    data = json.loads(Path(path).read_text(encoding="utf-8")) if path and os.path.isfile(path) else {}
    return data.get("delay"), data.get("delay_max")


def _desktop_label() -> str:
    """The desktop main window's label of the delay field (translator_gui QLabel, frozen in the schema data)."""
    import settings_schema_data

    return str(settings_schema_data.SETTINGS["delay"]["label"]).rstrip(":").strip()


def _env_now() -> tuple:
    return os.environ.get("SEND_INTERVAL_SECONDS"), os.environ.get("SEND_INTERVAL_MAX_SECONDS")


def _chapter_calls(records) -> list:
    """The run's chapter requests (one "제N장" chapter document each): not the EPUB metadata translation (sent
    before the chapters) nor the end-of-book chapter-titles batch (sent after all of them)."""
    return [r for r in records if str(r.preview or "").startswith("<html")
            and len(set(CHAPTER_HEADING.findall(r.preview or ""))) == 1]


def _spacing(records) -> list:
    """Seconds between consecutive arrivals of the run's chapter requests (one stagger scope, "default")."""
    starts = sorted(r.started for r in _chapter_calls(records))
    return [round(b - a, 3) for a, b in zip(starts, starts[1:])]


def _timeline(records) -> list:
    """(id, arrival, answer, prompt start) of each call, in seconds from the first arrival (failure messages)."""
    ordered = sorted(records, key=lambda r: r.started)
    t0 = ordered[0].started if ordered else 0.0
    return [(r.id, round(r.started - t0, 3), round((r.finished or r.started) - t0, 3), (r.preview or "")[:24])
            for r in ordered]


def _paced(spacing: list, delay: float) -> bool:
    """The calls were spaced by ``delay``: the median spacing reaches it (less the stamp jitter)."""
    ordered = sorted(spacing)
    median = ordered[len(ordered) // 2]
    return median >= delay - SPACING_SLACK


def _panels(view, kind: str) -> list:
    return [p for k, p in view.run_panels.items() if k[0] == view.cid and k[1] == kind]


async def _attach_and_send(driver, book: Path) -> None:
    """＋ › Files with ``book``, then Send: the Plan card ("Ready to translate")."""
    import flows

    await driver.tap(tooltip=flows.ATTACH_TOOLTIP)
    await driver.pick_file(book.name, lambda: driver.tap(key="attach-files"))
    await driver.wait(contains=book.stem, timeout=60)  # the composer pill
    await driver.tap(key="send-idle_ready", timeout=60)
    await driver.wait(text="Ready to translate", timeout=60)


async def _new_run(app, driver, cid: str, previous=None, timeout: float = 60.0):
    """The chat's run a Start / Resume tap submitted (the first long job on Android asks "Keep translations
    running": "Not now", as the device flow answers it)."""
    runs = app.chat_feature.runs
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await driver.count(text="Keep translations running"):
            await driver.tap(text="Not now")
        run = runs.run_for(cid)
        if run is not None and run is not previous and run.job_id is not None:
            return run
        await asyncio.sleep(0.05)
    raise AssertionError("the tap did not submit the chat's job")


async def _type(probe, field, text: str, *, done: bool) -> None:
    """The owner types ``text`` into ``field`` (focus, the text); ``done``: then the keyboard's Done."""
    assert probe.on_screen(field), "the API call delay field is not on screen"
    await probe.tester._dispatch(field, "focus")
    field.value = text
    if done:
        await probe.tester._dispatch(field, "submit")
    await asyncio.sleep(0.1)


def _check_tile(probe, panel, expected_text: str) -> object:
    """The Run options panel lists the API call delay tile as the desktop main window does: its label, the range
    hint, right after the chunk size; its field and the collapsed line show ``expected_text``."""
    from glossarion_mobile.ui.settings.tiles import DelayRangeTile

    names = list(panel.tiles)
    assert "delay" in names, f"the Run options have no API call delay tile: {names}"
    if "manual_chunk_size" in names:
        assert names.index("delay") == names.index("manual_chunk_size") + 1, names
    tile = panel.tiles["delay"]
    assert isinstance(tile, DelayRangeTile), type(tile)
    assert tile.title_text.value == _desktop_label() == "API call delay (s, or range 5-7)", tile.title_text.value
    assert probe.on_screen(tile.title_text) and probe.on_screen(tile.field), "the tile is not on screen"
    texts = probe.visible_texts()
    assert _desktop_label() in texts and "or a range like 5-7" in texts, sorted(t for t in texts if "delay" in t.lower())
    assert tile.field.value == expected_text, (tile.field.value, expected_text)
    assert f"Delay {expected_text} s" in str(panel.summary_text.value or ""), panel.summary_text.value
    assert probe.on_screen(panel.summary_text)
    return tile


async def _set_up(iso, server, books: list):
    """The app on temp storage, the owner's settings (the fake endpoint, glossary off) with the global delay
    0-0.1, and a new chat."""
    import flows

    tf = _UF._foundations()
    app, tester, driver = await _UF._host_driver(tf, {b.name: b for b in books})
    app.job_service.watcher_interval = 0.2
    await flows.wait_home(driver)
    D12._assert_isolated(iso.tmp)
    D12._configure(app, server, "off")
    store = app.config_store
    store.set_many(dict(GLOBAL))
    store.flush()
    assert store.save_error is None, store.save_error
    assert _global(store) == (0.0, 0.1) and _config_file_delay() == (0.0, 0.1)
    await flows.go_home(driver)
    await driver.tap(tooltip="New chat")
    return tf, app, tester, driver


async def _tear_down(tf, app, server, runs: list) -> None:
    try:
        for run in runs:
            await D12._end_job(app, server, run)
    finally:
        try:
            app.jobs.close()
        finally:
            await tf._stop(app)


# ==========================================================================
# 1. The Plan card's delay is the run's, and only the run's
# ==========================================================================


def test_plan_card_api_call_delay_is_this_runs_only(iso, offline):
    chapters = 12  # enough calls for a median spacing past the first calls of fresh API threads
    first = D12._book(iso.picks, chapters)  # big_novel.epub
    second_dir = iso.picks / "second"
    second_dir.mkdir()
    from glossarion_mobile.diagnostics import fixtures

    second = fixtures.build_tiny_epub(second_dir / "second_novel.epub", chapters=chapters)
    seen: dict = {}

    async def scenario(server, tap):
        tf, app, tester, driver = await _set_up(iso, server, [first, second])
        started: list = []
        try:
            store, view = app.config_store, app.chat_view
            await _attach_and_send(driver, first)
            cid = view.cid
            probe = D12.Probe(app, tester, cid)
            panels = _panels(view, "plan")
            assert len(panels) == 1, view.run_panels
            panel = panels[0]
            assert probe.on_screen(panel.control) and panel.control.title == "Run options"
            # the owner opens Run options: the delay is there, showing the setting
            await driver.tap(key="plan-run-options")
            assert panel.control.expanded
            assert not panel.only and panel.switch.value is False
            _check_tile(probe, panel, GLOBAL_TEXT)

            # "Only for this run", then 0.4-0.6 and Done on the keyboard
            await driver.tap(key="plan-run-options-only")
            assert panel.only and panel.switch.value is True
            tile = _check_tile(probe, panel, GLOBAL_TEXT)  # the run's tiles start from the setting
            await _type(probe, tile.field, "0.4-0.6", done=True)
            tile = panel.tiles["delay"]
            assert tile.error is None, tile.error
            assert tile.field.value == "0.4-0.6"
            assert "Delay 0.4-0.6 s" in str(panel.summary_text.value), panel.summary_text.value
            assert panel.config_overrides() == {"delay": 0.4, "delay_max": 0.6}, panel.config_overrides()
            assert _global(store) == (0.0, 0.1), "the run's value reached the setting"
            store.flush()
            assert _config_file_delay() == (0.0, 0.1), "the run's value reached config.json"

            # ---- Start: the run sends with 0.4-0.6 -----------------------------------------------------------
            env_before = _env_now()
            mark = server.mark()
            await driver.tap(text="Start")
            run = await _new_run(app, driver, cid)
            started.append(run)
            snap = app.job_service.snapshot(run.job_id)
            overrides = dict(snap.spec.params.get("config_overrides") or {})
            assert (overrides.get("delay"), overrides.get("delay_max")) == (0.4, 0.6), overrides
            await D12._finished(app, run, RUN_TIMEOUT)
            assert run.state == "done" and run.error is None, (run.state, run.error)
            records = server.records("translation", since=mark)
            assert len(records) >= chapters, len(records)
            envs = tap.envs(records)
            assert set(envs) == {("0.4", "0.6")}, envs
            spacing = _spacing(records)
            assert _paced(spacing, 0.4), f"the calls were not paced by the run's delay: {spacing} {_timeline(records)}"

            # ---- after the run: the setting, config.json and the process env are as before --------------------
            await asyncio.to_thread(app.job_service.wait_idle, 60)
            assert _global(store) == (0.0, 0.1), _global(store)
            store.flush()
            assert _config_file_delay() == (0.0, 0.1), _config_file_delay()
            assert _env_now() == env_before, (_env_now(), env_before)

            # ---- the next attachment's Plan card starts from the setting; its run sends with it ---------------
            await _attach_and_send(driver, second)
            panels = _panels(view, "plan")
            assert len(panels) == 1 and panels[0] is not panel, view.run_panels
            nxt = panels[0]
            await driver.tap(key="plan-run-options")
            assert not nxt.only and nxt.switch.value is False and nxt.config_overrides() == {}
            _check_tile(probe, nxt, GLOBAL_TEXT)
            mark = server.mark()
            await driver.tap(text="Start")
            run2 = await _new_run(app, driver, cid, previous=run)
            started.append(run2)
            overrides = dict(app.job_service.snapshot(run2.job_id).spec.params.get("config_overrides") or {})
            assert "delay" not in overrides and "delay_max" not in overrides, overrides
            await D12._finished(app, run2, RUN_TIMEOUT)
            assert run2.state == "done" and run2.error is None, (run2.state, run2.error)
            records = server.records("translation", since=mark)
            assert records and set(tap.envs(records)) == {GLOBAL_ENV}, tap.envs(records)
            on_setting = _spacing(records)  # printed for the report (the runner's speed sets it)
            assert _global(store) == (0.0, 0.1)
            seen["spacing"] = (spacing, on_setting)
        finally:
            await _tear_down(tf, app, server, started)

    with D12._server(0.0) as server:
        tap = EnvTap(server)
        asyncio.run(scenario(server, tap))
    assert seen.get("spacing"), "the scenario did not reach its end"
    print(f"\nspacing with the run's 0.4-0.6 s: {seen['spacing'][0]}\nspacing on the setting 0-0.1 s: {seen['spacing'][1]}")
    D12._assert_offline(offline)


# ==========================================================================
# 2. The Result card's delay reaches the resumed run, and only it
# ==========================================================================


def test_result_card_api_call_delay_reaches_the_resumed_run_only(iso, offline):
    chapters = 16
    book = D12._book(iso.picks, chapters)
    seen: dict = {}

    async def scenario(server, tap, gate):
        tf, app, tester, driver = await _set_up(iso, server, [book])
        started: list = []
        try:
            store, view = app.config_store, app.chat_view
            await _attach_and_send(driver, book)
            cid = view.cid
            probe = D12.Probe(app, tester, cid)
            mark = server.mark()
            await driver.tap(text="Start")  # Run options untouched: the setting
            run = await _new_run(app, driver, cid)
            started.append(run)

            # ---- ■ Stop mid-book while the model holds the requests already sent ------------------------------
            assert await D12._until(lambda: D12._settled(server), RUN_TIMEOUT), D12._held_state(probe, server)
            await D12._tap(probe, view.live_job_card.action_buttons["stop"])
            assert await D12._until(lambda: run.stop_requested, 5)
            assert await D12._until(lambda: probe.job()["state"] == "STOPPING", 10), probe.job()
            server.release()
            await D12._finished(app, run, RUN_TIMEOUT)
            assert run.state == "stopped", run.state
            records = server.records("translation", since=mark)
            assert records and set(tap.envs(records)) == {GLOBAL_ENV}, tap.envs(records)
            on_setting = _spacing(records)  # printed for the report (the runner's speed sets it)
            assert probe.job()["completed"] < chapters, probe.job()

            # ---- the Result card: Resume and its Run options with the delay -----------------------------------
            cards = probe.job_cards()
            assert len(cards) == 1, cards
            ended = cards[0]
            assert "resume" in ended.action_buttons and probe.on_screen(ended.action_buttons["resume"])
            panels = _panels(view, "resume")
            assert len(panels) == 1, view.run_panels
            panel = panels[0]
            from glossarion_mobile.ui.components.tap_away import closes_on_tap_away

            assert closes_on_tap_away(panel.control) in ended.options_box.controls and ended.options_box.visible
            assert probe.on_screen(panel.control) and panel.control.title == "Run options"
            await driver.tap(key=panel.control.key)  # expand
            assert panel.control.expanded
            _check_tile(probe, panel, GLOBAL_TEXT)
            await driver.tap(key=panel.switch.key)  # Only for this run
            assert panel.only and panel.switch.value is True
            tile = _check_tile(probe, panel, GLOBAL_TEXT)
            # 0.5 typed and Resume tapped straight away (the field still has the focus)
            await _type(probe, tile.field, "0.5", done=False)
            assert _global(store) == (0.0, 0.1)

            env_before = _env_now()
            mark = server.mark()
            await D12._tap(probe, ended.action_buttons["resume"])
            resumed = await _new_run(app, driver, cid, previous=run)
            started.append(resumed)
            overrides = dict(app.job_service.snapshot(resumed.job_id).spec.params.get("config_overrides") or {})
            assert (overrides.get("delay"), overrides.get("delay_max")) == (0.5, ""), overrides
            await D12._finished(app, resumed, RUN_TIMEOUT)
            assert resumed.state == "done" and resumed.error is None, (resumed.state, resumed.error)
            records = server.records("translation", since=mark)
            assert len(records) >= 2, len(records)
            envs = tap.envs(records)
            assert set(envs) == {("0.5", "")}, envs
            spacing = _spacing(records)
            assert _paced(spacing, 0.5), f"the calls were not paced by the run's delay: {spacing} {_timeline(records)}"

            # ---- the setting is unchanged -----------------------------------------------------------------------
            await asyncio.to_thread(app.job_service.wait_idle, 60)
            assert _global(store) == (0.0, 0.1), _global(store)
            store.flush()
            assert _config_file_delay() == (0.0, 0.1), _config_file_delay()
            assert _env_now() == env_before, (_env_now(), env_before)
            seen["spacing"] = (spacing, on_setting)
        finally:
            server.release()
            await _tear_down(tf, app, server, started)

    with D12._server(0.0) as server:
        tap = EnvTap(server)
        gate = D12.HoldGate(server, 4)  # the EPUB metadata request + 3 chapters, then ■ Stop
        asyncio.run(scenario(server, tap, gate))
    assert seen.get("spacing"), "the scenario did not reach its end"
    print(f"\nspacing on the setting 0-0.1 s (stopped run): {seen['spacing'][1]}\n"
          f"spacing of the resumed run with 0.5 s: {seen['spacing'][0]}")
    D12._assert_offline(offline)
