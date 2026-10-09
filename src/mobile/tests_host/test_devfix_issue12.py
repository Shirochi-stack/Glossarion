"""Acceptance test for the owner's device report #12 on the U8 APK (2026-10-08):
"a chat attachment job shows Done right after the glossary step, and the translation's request
cards do not appear in the chat until the Jobs page is opened".

The cause (diagnosis wf_eea03b3c-0ff): at the glossary approval gate ``ChatStore.commit_request_phase``
freezes the glossary requests into the chat; the desktop message window (20 messages / 120,000
characters) then started after the turn's ``user_file``, the turn's JobCard lost its turn (index -1)
and the RUNNING job was drawn as a static "Done" / "Attachment" card while every live repaint went to
a card that was no longer on screen. A book big enough for three merged Balanced glossary requests
with long descriptions (or twenty requests) triggers it; the 12-chapter self-test book never did,
which is why the device E2E stayed green.

This file proves the fix end to end on the REAL app objects: the app started on a fake Flet session
(``test_ui_foundations._start`` through ``test_ui_flows._host_driver``, like the other tests_host UI
tests), the real JobService on the app's real ``UiDispatcher``, ``HeadlessOwner`` and the shared
translation pipeline (Balanced glossary pre-pass + the Direct Text approval question,
``TransateKRtoEN``), the real ``ChatFeature`` / ``ChatRuns`` / ``ChatView`` and the desktop
``direct_text_store.ChatStore``. Only the model is the offline fake OpenAI server on 127.0.0.1
(``diagnostics.fake_llm_server``, the E2E fixture); the E2E's network guard refuses every
non-loopback connection for the whole test. The owner's taps go through the UI: New chat, ＋ › Files
with a 210-chapter EPUB, Send, Start, then the approval card's ✓ Yes (and the running card's Stop).
A sampler records, every 40 ms on the UI loop, what the chat shows next to what JobService (the Jobs
page's source) reports.

1. ``test_book_card_stays_live_through_the_glossary_gate``: at the gate the chat shows ONE JobCard for
   the turn; it is the live card, mounted on the client (in the Flet session's index and the visible
   tree), Running, "Waiting for your glossary decision", titled with the book's file name and listing
   the committed glossary cards. The old desktop message window is checked to have cut the turn's
   file card off, so the owner's trigger really is present. After ✓ Yes the same card object lists
   the committed glossary cards + the live translation requests (the job's own requests, a repaint
   behind at most) and "Chapter n/210" follows the job's progress. No sample of the whole run shows
   "Done" / "Attachment" or a second card while the job runs. The Result reads
   "Done · 210/210 chapters" with the book's title and every request of the turn once.
2. ``test_stop_shows_stopped_with_resume``: the running card's Stop (graceful) ends the run as
   "Stopped · n/210 chapters" with Resume, the book's title and the turn's requests.
3. ``test_hidden_app_never_shows_done_and_catches_up``: the app goes to the background before the
   gate (Android lifecycle events through the session) and comes back at the gate (still the live
   card, "Waiting for your glossary decision"); ✓ Yes; it goes to the background again
   mid-translation and comes back: the card was never "Done", and the catch-up repaint shows the job's
   current rows and progress; the Result reads "Done · 210/210 chapters".
4. ``test_a_render_while_the_run_commits_lists_each_request_once`` (found by this test): a render
   while ``ChatStore.finish_run`` has committed the run's cards and the run is still finishing lists
   each request once: the committed rows, as the Result will (the commit is recognised by the request
   number a committed label ends in, " · Request N"; the header commit renames its card and the
   shared commit drops lifecycle-only rows). Transient (it lasts until the run ends) and not seen in
   the natural runs above.
5. ``test_resume_lists_the_resumed_runs_live_requests`` (found by the DF2 verify): Stop, then the ended
   card's Resume. While the resumed job translates, its card lists the stopped run's committed cards
   followed by the resumed run's live requests, which continue the chat's request numbers. Before the
   fix the resumed run reused the stopped run's request numbers, the card took every live row for an
   already committed one, and it stayed frozen at the stopped run's rows for the whole resumed run.
   The live rows keep the shared stream's spine order, as on the desktop: the run's first request (the
   EPUB metadata translation) is a lifecycle-only "Request N" row listed after the chapter rows, so a
   chapter dispatched later is inserted above it. The card is compared with the stream once the held
   fake model keeps both from changing (an append-only comparison failed whenever a chapter was
   dispatched between the card's last repaint and the check); no step depends on the runner's speed.
   The Stop lands while the model holds the run's sent requests: a graceful Stop waits for them (the
   run stays Stopping, with no Resume) and commits their answers with the stopped run, in order.
6. ``test_a_force_stopped_runs_late_answers_stay_out_of_the_resumed_run`` (the late-answer question of
   test 5's flake, which was the spine order above): Force stop while the model holds the run's sent
   requests, Resume, and only then let them through. The force-stopped run ended without them (their
   status-only rows are dropped), and nothing of it reaches the chat, the resumed run's stream or its
   card after Resume; the Result lists every chapter once.

Real data stays untouched: the app runs in temp FLET_APP_STORAGE_* dirs (the bootstrap points HOME,
OUTPUT_DIRECTORY, GLOSSARION_LIBRARY_DIR, GLOSSARION_DATA_DIR and CONFIG_FILE there; asserted below),
USERPROFILE / APPDATA / LOCALAPPDATA point into tmp_path and HTTP logging is off. A failing scenario
force-stops its job before the app closes, so no job thread outlives its test.

Run from src/mobile with the mobile env (CI: the uv env; on the Windows dev box: ``.venv``, Python 3.12,
which borrows the 3.12 user site, so keep that ``PYTHONPATH``)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_devfix_issue12.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import re
import sys
import threading
import time
import types
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


_UF = _load("_glossarion_uiflows_devfix12", "test_ui_flows.py")  # _host_driver / _foundations
storage = _UF.storage
app_env = _UF.app_env

CHAPTERS = 210
EPUB_NAME = "big_novel.epub"
#: Description characters of the padded glossary term: three merged Balanced glossary requests
#: (99 chapters each) answer with more than 120,000 characters together, so the desktop message
#: window (``tail_window``: 3 messages minimum, 120,000-character budget) starts after the turn's
#: file card - the owner's trigger. Asserted at the gate.
PAD_CHARS = 45_000
RUN_TIMEOUT = 300.0
WAITING = "Waiting for your glossary decision"
#: The live card repaints on the stream ticks (280-900 ms) and on every job snapshot, while the
#: job-side drain classifies requests as they stream: a visible card may be this many requests behind.
REPAINT_LAG = 25
#: Android "Home" / back to the app (``app_lifecycle_state_change``; test_devfix_issue6's sequences)
LEAVE = ("inactive", "hide", "pause")
COME_BACK = ("show", "resume")
CHAPTER_RE = re.compile(r"Chapter (\d+)/(\d+)")
ENDED_PHASES = ("done", "stopped", "failed", "interrupted")


# ==========================================================================
# Fixtures and helpers
# ==========================================================================


@pytest.fixture
def iso(tmp_path, monkeypatch, request):
    """The app on temp storage, with the Windows profile variables redirected as well."""
    user = tmp_path / "user"
    for key, sub in (("USERPROFILE", "."), ("APPDATA", "AppData/Roaming"), ("LOCALAPPDATA", "AppData/Local")):
        folder = (user / sub).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(key, str(folder))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    request.getfixturevalue("app_env")  # FLET_APP_STORAGE_* + bootstrap (HOME, OUTPUT_DIRECTORY, Library, config)
    picks = tmp_path / "picks"
    picks.mkdir()
    return types.SimpleNamespace(tmp=tmp_path, picks=picks, user=user)


@pytest.fixture
def offline():
    """The E2E's network guard (``diagnostics.e2e.ProcessGuards``): every non-loopback connect / DNS
    lookup fails at once and is recorded."""
    import traceback

    from glossarion_mobile.diagnostics.e2e import ProcessGuards

    guards = ProcessGuards()
    record = guards._record

    def record_with_origin(bucket, **event):
        # the guard keeps the last 12 frames (often all stdlib); add the Glossarion frames that asked
        event["origin"] = [f"{os.path.relpath(f.filename, SRC_DIR)}:{f.lineno} {f.name}"
                           for f in traceback.extract_stack()[:-2] if _inside(f.filename, SRC_DIR)][-12:]
        record(bucket, **event)

    guards._record = record_with_origin
    guards._install_network_guard()
    try:
        yield guards
    finally:
        guards.uninstall()


def _catalog_poll(event: dict) -> bool:
    """The selected model's 24 h provider-catalog auto-poll (``model_options._fetch_provider_catalog``
    on a ``model-catalog`` thread; desktop parity, once per process): not part of the chat run. The
    guard blocks it like every other non-loopback attempt (the catalog falls back to its static list)."""
    return any("_fetch_provider_catalog" in frame for frame in event.get("origin") or [])


def _assert_offline(guards) -> None:
    """Nothing but the catalog auto-poll tried to leave the machine (and that was blocked)."""
    others = [e for e in guards.network if not _catalog_poll(e)]
    lines = []
    for event in others[:3]:
        lines.append(f"{event.get('api')} {event.get('target')} [{event.get('thread')}]")
        lines.extend(f"    {frame}" for frame in (event.get("origin") or []) + (event.get("stack") or []))
    assert not others, "\n".join(lines)


def _inside(path, root) -> bool:
    try:
        a = os.path.normcase(os.path.abspath(str(path)))
        b = os.path.normcase(os.path.abspath(str(root)))
        return os.path.commonpath([a, b]) == b
    except ValueError:
        return False


def _assert_isolated(tmp) -> None:
    for key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "OUTPUT_DIRECTORY", "GLOSSARION_LIBRARY_DIR",
                "GLOSSARION_DATA_DIR", "CONFIG_FILE"):
        value = os.environ.get(key)
        assert value and _inside(value, tmp), f"{key}={value!r} is not inside the test's tmp dir"
    assert os.environ.get("GLOSSARION_HTTP_LOG") == "0"


async def _until(predicate, timeout: float = 30.0, step: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(step)
    return bool(predicate())


def _book(picks: Path, chapters: int = CHAPTERS) -> Path:
    from glossarion_mobile.diagnostics import fixtures

    return fixtures.build_tiny_epub(picks / EPUB_NAME, chapters=chapters)


def _server(translation_delay: float):
    """The fake model with one recurring term (in every chapter) whose description is long."""
    from glossarion_mobile.diagnostics.fake_llm_server import FakeLLMServer, GlossaryEntry

    server = FakeLLMServer()
    server.glossary = server.glossary + (
        GlossaryEntry("term", "마왕성", "Demon King's Castle", "", ("lorem " * (PAD_CHARS // 6)).strip()),)
    server.set_delay("translation", translation_delay)  # timing only: samples during every chapter
    return server


def _configure(app, server, glossary: str = "balanced") -> None:
    """The settings the owner's phone had: the fake OpenAI endpoint + a dummy key (``flows.ui_config``)
    and the Welcome's Balanced glossary (approval asked; "Always accept" stays off)."""
    import flows
    from glossarion_mobile.diagnostics.fake_llm_server import FAKE_MODEL
    from glossarion_mobile.ui.chat.direct_text_rules import AUTO_ACCEPT_GLOSSARY_PREF
    from glossarion_mobile.ui.screens.welcome_flow import welcome_glossary_updates

    values = flows.ui_config(server.url, FAKE_MODEL)
    values.update(welcome_glossary_updates(glossary))
    store = app.config_store
    store.set_many(values)
    store.flush()
    assert store.save_error is None, store.save_error
    assert store.get("auto_glossary_mode") == glossary
    assert store.get("enable_auto_glossary") is (glossary == "balanced")
    prefs = getattr(app, "prefs", None)
    assert prefs is None or not prefs.get(AUTO_ACCEPT_GLOSSARY_PREF, False)


def _lifecycle_only(segment: dict) -> bool:
    """A request segment the shared ``DirectTextStream`` commit drops: an API client's lifecycle-only request
    (status-only, no content, thinking or tokens; the EPUB metadata bookkeeping channel)."""
    return bool(segment.get("status_only")) and not str(segment.get("content") or "").strip() \
        and not str(segment.get("thinking") or "").strip() and not int(segment.get("thinking_tokens") or 0) \
        and not int(segment.get("text_tokens") or 0)


def _labels(card) -> list:
    """The labels of every request a JobCard lists (``request_segments``; a JobCard from before the
    fix had no such list: its rows then, so this test also runs against the old code)."""
    segments = getattr(card, "request_segments", None)
    if segments is not None:
        return [str(s.get("label") or "") for s in segments]
    labels = []
    for row in card.requests_column.controls:
        try:
            labels.append(str(row.content.controls[0].controls[0].value or ""))
        except (AttributeError, IndexError):
            labels.append("")
    return labels


def _chapters(labels) -> list:
    """The chapter number of each chapter request label ("Chapter 12 (chunk 1/1) · chapter0012.xhtml …"), in order."""
    return [int(m.group(1)) for m in (re.match(r"Chapter (\d+) \(chunk 1/1\)", str(label)) for label in labels) if m]


def _sent_chapters(records) -> list:
    """The chapters the fake model's requests ``records`` translate: the test book's "제N장" headings in their prompt
    previews (``RequestRecord.chapters`` reads the self-test book's "제N화" headings)."""
    return sorted({int(n) for r in records for n in re.findall(r"제\s*(\d+)\s*장", r.preview or "")})


def _request_rows(segments) -> list:
    """(request number, label) of each request segment; a label turns from "Request N" into its chapter once the
    request's content is classified, the number stays."""
    from glossarion_mobile.ui.chat.stream_bridge import segment_request_number

    return [(segment_request_number(s), str(s.get("label") or "")) for s in segments or []]


def _threads(segments) -> set:
    """The API client threads of request segments ("Thread-24 (api_call)"; unique in the process)."""
    return {str(s.get("thread") or "") for s in segments or []} - {""}


def _in_stream_order(shown: list, live: list) -> bool:
    """``shown`` lists rows of ``live`` in ``live``'s order, a repaint behind at most (the newest rows may be missing,
    anywhere). The shared ``DirectTextStream`` keeps its rows in spine order, not in dispatch order
    (``_sort_active_request_segments``, as the desktop Direct Text dialog lists them): the EPUB metadata request's
    lifecycle-only "Request N" row and the end-of-book header batch sort after the chapter rows, so a row dispatched
    later can be inserted above an older one, and a repaint behind is not a prefix of the stream."""
    rest = iter(live)
    return all(row in rest for row in shown)


def _card_shows_stream(view, run, offset: int, snap: dict, minimum: int = 1) -> bool:
    """The live card's rows after the first ``offset`` (the turn's committed cards) are exactly ``run``'s live
    request rows, and there are at least ``minimum``. One snapshot of both goes to ``snap``: assert on it with no
    await in between, so the card (repainted on the UI loop) cannot change in the meantime."""
    card = view.live_job_card
    live = _request_rows(run.stream.segments())
    shown = None if card is None else _request_rows(card.request_segments[offset:])
    snap.update(shown=shown, live=live)
    return shown == live and len(live) >= minimum


def _settled(server) -> bool:
    """The fake model holds and every request in flight is parked: the job cannot end, and its stream stops changing
    once the workers still handling earlier replies have dispatched (and parked) their next chapter."""
    return server.holding and server.parked >= 1 and server.in_flight() == server.parked


def _held_state(probe, server) -> tuple:
    return probe.describe(), server.holding, server.parked, server.in_flight()


def _next_run(probe, ended):
    """The chat's live run after ``ended`` (Resume starts a new run for the same turn)."""
    current = probe.run()
    return current if current is not None and current is not ended and current.live else None


class HoldGate:
    """The fake model's own reply hook: ``hold()`` once ``count`` translation replies went out to requests sent after
    ``hold_after`` was called (a run's first request is the EPUB metadata translation, then come its chapters).
    A step then waits on the model's state, never on the runner's speed. Released requests answer; nothing is
    aborted (an abort makes the client retry)."""

    def __init__(self, server, count: "int | None" = None) -> None:
        self.server = server
        self._lock = threading.Lock()
        self.since = 0
        self.replies = 0
        self.hold_at = count
        server.on_response.append(self)

    def hold_after(self, count: int) -> None:
        with self._lock:
            self.since = self.server.mark()
            self.replies = 0
            self.hold_at = count

    def __call__(self, record) -> None:
        if record.kind != "translation" or record.id <= self.since:
            return
        with self._lock:
            self.replies += 1
            if self.hold_at is None or self.replies < self.hold_at:
                return
            self.hold_at = None
        self.server.hold()


class Probe:
    """What the chat shows (the Python control tree the fake Flet session mirrors to the client) next
    to what the job is doing (JobService: the Jobs page's source)."""

    def __init__(self, app, tester, cid: str) -> None:
        self.app = app
        self.tester = tester
        self.page = tester.page
        self.session = tester.session
        self.cid = cid
        self.view = app.chat_view
        self.runs = app.chat_feature.runs
        self.chats = app.chat_feature.chats
        self.service = app.job_service
        self.samples: list = []
        self.problems: list = []
        self.last_live_labels: list = []
        self.gate_t = 0.0
        self.visible_since = time.monotonic()

    def run(self):
        return self.runs.run_for(self.cid)

    def job_cards(self) -> list:
        from glossarion_mobile.ui.chat.cards import JobCard

        shown = list(self.view.transcript.cards) + [getattr(c, "card", c) for c in self.view.transcript.tail]
        return [c for c in shown if isinstance(c, JobCard)]

    def mounted(self, control) -> bool:
        """In the Flet session's index (sent to the client and not removed since)."""
        return self.session.index.get(control._i) is control

    def on_screen(self, control) -> bool:
        """``control`` is in the visible tree of the top view (what the tester can tap)."""
        return any(c is control for c in self.tester._walk())

    def visible_texts(self, rows: bool = False) -> set:
        """Every text of the transcript as the user sees it (visible controls only; host_tester's rules).
        Without ``rows`` the request rows are left out: each finished request's row has its own "Done"
        chip (that request, not the job); the owner's "Done" was the card's own state."""
        from host_tester import _children, _texts

        skip = set() if rows else {id(c.requests_column) for c in self.job_cards()}
        out: set = set()
        stack = [self.view.transcript]
        seen: set = set()
        while stack:
            control = stack.pop()
            if id(control) in seen or id(control) in skip or getattr(control, "visible", True) is False:
                continue
            seen.add(id(control))
            out.update(_texts(control))
            stack.extend(_children(control))
        return out

    def job(self) -> dict:
        from glossarion_mobile.ui.chat.job_binding import progress_counts

        run = self.run()
        job_id = getattr(run, "job_id", None)
        snap = self.service.snapshot(job_id) if job_id else None
        counts = progress_counts(snap) if snap is not None else {}
        return {
            "id": job_id,
            "state": getattr(getattr(snap, "state", None), "value", None),
            "completed": counts.get("completed", 0),
            "total": counts.get("total", 0),
            "segments": len(self.service.request_segments(job_id)) if job_id else 0,
        }

    def card_state(self, card) -> dict:
        match = CHAPTER_RE.search(str(card.line_text.value or ""))
        return {
            "id": id(card),
            "live": card is self.view.live_job_card,
            "mounted": self.mounted(card),
            "phase": card.phase.name,
            "state": str(card.state_text.value or ""),
            "title": str(card.title_text.value or ""),
            "requests": len(_labels(card)),
            "rows": len(card.requests_column.controls),
            "tile": str(card.requests_tile.title or ""),
            "line": str(card.line_text.value or ""),
            "chapter": (int(match.group(1)), int(match.group(2))) if match else None,
        }

    def sample(self, note: str = "") -> dict:
        run = self.run()
        now = time.monotonic()
        visible = getattr(self.page, "app_visible", True) is not False
        entry = {
            "t": now,
            "note": note,
            "visible": visible,
            # visible long enough for the catch-up repaint (the stream loop parks while hidden)
            "steady": visible and now - self.visible_since > 1.5,
            "run_state": getattr(run, "state", None),
            "run_live": bool(run is not None and run.live),
            "awaiting": bool(run is not None and run.awaiting_glossary),
            "live_segments": len(run.stream.segments() or []) if run is not None else 0,
            "job": self.job(),
            "messages": len(self.chats.messages(self.cid)),
            "cards": [self.card_state(c) for c in self.job_cards()],
        }
        live = self.view.live_job_card
        if live is not None:
            self.last_live_labels = _labels(live)
        self.samples.append(entry)
        return entry

    async def sampler(self, stop: asyncio.Event, interval: float = 0.04) -> None:
        while not stop.is_set():
            try:
                self.sample()
            except Exception as exc:  # a sampling bug must not hide the scenario's own failure
                self.problems.append(f"sampling failed: {type(exc).__name__}: {exc}")
            await asyncio.sleep(interval)

    async def lifecycle(self, *states: str) -> None:
        """The app leaves / comes back (Flet's ``app_lifecycle_state_change`` through the session)."""
        for state in states:
            await self.session.dispatch_event(self.page._i, "app_lifecycle_state_change", {"state": state})
        if getattr(self.page, "app_visible", True) is not False:
            self.visible_since = time.monotonic()
        await asyncio.sleep(0.05)

    def turn_item(self):
        run = self.run()
        return next((i for i in self.view.items if i.kind == "job" and i.index == run.user_index), None)

    def committed_labels(self) -> list:
        """Labels of the turn's committed request messages (the glossary gate's, at the gate)."""
        messages = self.chats.messages(self.cid)
        item = self.turn_item()
        return [str(messages[i][5]) for i in (item.requests if item is not None else [])]

    def describe(self) -> str:
        return "\n".join(repr(s) for s in self.samples[-6:])


async def _tap(probe: Probe, control) -> None:
    """The tap Flutter would send to ``control`` (it must be in the visible tree)."""
    assert probe.on_screen(control), f"{type(control).__name__} is not on screen"
    await probe.tester._dispatch(control, "click")


async def _finished(app, run, timeout: float = RUN_TIMEOUT) -> None:
    runs = app.chat_feature.runs
    assert await _until(lambda: not run.live, timeout), f"the chat run did not end (state {run.state})"
    for thread in list(runs.finish_threads):
        await asyncio.to_thread(thread.join, 60)
    await asyncio.sleep(0.5)  # the posted re-render and the card extras


async def _end_job(app, server, run) -> None:
    """Teardown: a scenario that failed mid-run force-stops its job (and joins the chat's finish
    threads) before the app closes, so no job thread outlives the test or reaches the next one."""
    service = getattr(app, "job_service", None)
    job_id = getattr(run, "job_id", None)
    if service is None or job_id is None:
        return

    def ended() -> bool:
        snap = service.snapshot(job_id)
        return snap is None or snap.is_terminal

    if not ended():
        server.release(abort=True)
        for _attempt in range(3):
            service.request_stop(job_id, force=True, reason="test teardown")
            if await _until(ended, 30):
                break
    await asyncio.to_thread(service.wait_idle, 60)
    for thread in list(app.chat_feature.runs.finish_threads):
        await asyncio.to_thread(thread.join, 60)
    assert ended(), f"job {job_id} did not stop"


async def _owner_flow(iso, server, body, *, chapters: int = CHAPTERS, glossary: str = "balanced") -> None:
    """Start the app, apply the owner's settings, and make the owner's taps: New chat, ＋ › Files (the
    book), Send (the Plan card), Start (the first long job on Android asks "Keep translations running":
    "Not now", as the device flow answers it). ``body(ctx)`` then drives and checks the run; the
    sampler runs from Start to the end of ``body``."""
    import flows

    epub = _book(iso.picks, chapters)
    tf = _UF._foundations()
    app, tester, driver = await _UF._host_driver(tf, {EPUB_NAME: epub})
    # The card's "Chapter n/210" moves on ProgressWatcher's polls of translation_progress.json (the app's
    # 2 s default): a fast runner translates the fake book in a few polls (CI saw 42, 84, 170, 210). Poll
    # often, so the card's own repaint cadence (280-900 ms) bounds what the sampler sees, on any runner.
    app.job_service.watcher_interval = 0.2
    stop = asyncio.Event()
    sampler = None
    run = None
    try:
        await flows.wait_home(driver)
        _assert_isolated(iso.tmp)
        _configure(app, server, glossary)
        await flows.go_home(driver)
        await driver.tap(tooltip="New chat")
        await driver.tap(tooltip=flows.ATTACH_TOOLTIP)
        await driver.pick_file(epub.name, lambda: driver.tap(key="attach-files"))
        await driver.wait(contains=epub.stem, timeout=60)  # the composer pill
        await driver.tap(key="send-idle_ready", timeout=60)
        await driver.wait(text="Ready to translate", timeout=60)  # the Plan card
        await driver.tap(text="Start")
        cid = app.chat_view.cid
        runs = app.chat_feature.runs
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if await driver.count(text="Keep translations running"):
                await driver.tap(text="Not now")
            candidate = runs.run_for(cid)
            if candidate is not None and candidate.job_id is not None:
                run = candidate
                break
            await asyncio.sleep(0.05)
        assert run is not None, "Start did not submit the chat's job"
        probe = Probe(app, tester, cid)
        sampler = asyncio.ensure_future(probe.sampler(stop))
        ctx = types.SimpleNamespace(app=app, tester=tester, driver=driver, server=server, run=run, probe=probe,
                                    stop_sampler=stop, sampler=sampler)
        await body(ctx)
    finally:
        stop.set()
        if sampler is not None:
            try:
                await asyncio.wait_for(sampler, 5)
            except Exception:
                sampler.cancel()
        try:
            if getattr(tester.page, "app_visible", True) is False:
                for state in COME_BACK:
                    await tester.session.dispatch_event(tester.page._i, "app_lifecycle_state_change", {"state": state})
            await _end_job(app, server, run)
        finally:
            try:
                app.jobs.close()
            finally:
                await tf._stop(app)


async def _stop_sampling(ctx) -> None:
    ctx.stop_sampler.set()
    await ctx.sampler
    assert not ctx.probe.problems, ctx.probe.problems


def _check_gate(probe: Probe, *, glossary_requests: int) -> dict:
    """At the glossary gate: the turn's one JobCard is the mounted live card for the book."""
    from glossarion_mobile.ui.chat.direct_text_rules import DEFAULT_RENDERED_CARD_LIMIT
    from glossarion_mobile.ui.chat.transcript_model import tail_window

    view, run = probe.view, probe.run()
    messages = probe.chats.messages(probe.cid)
    assert run.awaiting_glossary and run.live
    job = probe.job()
    assert job["state"] == "RUNNING", job
    # the gate froze the glossary requests into the chat (ChatStore.commit_request_phase) ...
    assert len(messages) == 1 + glossary_requests, [m[0] for m in messages]
    assert messages[run.user_index][0] == "user_file" and messages[run.user_index][1] == EPUB_NAME
    # ... and the old desktop message window starts after the turn's file card: the owner's trigger
    limit = view.settings().rendered_card_limit
    assert limit == DEFAULT_RENDERED_CARD_LIMIT
    assert tail_window(messages, limit)[0] > run.user_index, \
        "the scenario no longer reproduces the owner's trigger (raise PAD_CHARS)"

    cards = probe.job_cards()
    assert len(cards) == 1, f"{[probe.card_state(c) for c in cards]}"
    card = cards[0]
    state = probe.card_state(card)
    assert card is view.live_job_card, f"the card at the gate is not the live card: {state}"
    assert state["mounted"], "the live card is not on the client"
    assert state["phase"] == "running" and state["state"] == WAITING, f"{state}"
    assert state["title"] == EPUB_NAME, f"{state}"
    committed = probe.committed_labels()
    assert len(committed) == glossary_requests
    labels = _labels(card)
    assert labels == committed, (labels, committed)
    assert state["tile"] == f"Requests ({glossary_requests})" and state["rows"] == glossary_requests, state
    assert card.requests_tile.expanded
    texts = probe.visible_texts()
    assert WAITING in texts and EPUB_NAME in texts, sorted(t for t in texts if len(t) < 60)[:80]
    assert "Done" not in texts and "Attachment" not in texts, sorted(t for t in texts if len(t) < 60)[:80]
    assert set(committed) <= probe.visible_texts(rows=True), "the committed glossary rows are not on screen"
    assert probe.on_screen(card.state_text) and probe.on_screen(card.title_text)
    approval = view.approval_card
    assert approval is not None and approval in view.transcript.tail and probe.mounted(approval.yes_button)
    assert probe.on_screen(approval.yes_button)
    probe.gate_t = probe.sample("at the gate")["t"]
    return {"card": card, "committed": committed}


def _check_live_card(sample: dict, card, committed: list) -> dict:
    """One sample while the job translates: the same live card, mounted, Running, the book's title,
    the committed glossary cards + the live requests (a repaint behind at most)."""
    assert len(sample["cards"]) == 1 and sample["cards"][0]["id"] == id(card), sample
    c = sample["cards"][0]
    assert c["live"] and c["mounted"] and c["phase"] == "running" and c["title"] == EPUB_NAME, c
    assert c["state"] in ("Translating", "Translating headers…"), c
    assert c["chapter"] is not None and c["chapter"][1] == CHAPTERS, c
    assert c["chapter"][0] <= sample["job"]["completed"], (c, sample["job"])
    expected = len(committed) + sample["live_segments"]
    assert expected - REPAINT_LAG <= c["requests"] <= expected, (c, sample["live_segments"])
    assert c["tile"] == f"Requests ({c['requests']})" and c["rows"] == min(c["requests"], 20), c
    return c


def _check_running_samples(probe: Probe, card, committed: list) -> list:
    """Every sample while the run was live: one JobCard for the turn, titled with the book, never an
    ended ("Done") card. From the gate to the end of the run: the same live card object, mounted,
    Running; the rows are the committed glossary cards + the live requests (never one twice, a repaint
    behind at most while the app is visible); "Chapter n/210" never runs ahead of the job."""
    early = [s for s in probe.samples if s["run_live"] and s["t"] < probe.gate_t]
    bad = [s for s in early if len(s["cards"]) != 1 or s["cards"][0]["title"] != EPUB_NAME
           or s["cards"][0]["phase"] in ENDED_PHASES or s["cards"][0]["state"].startswith("Done")]
    assert not bad, "\n".join(repr(s) for s in bad[:5])
    running = [s for s in probe.samples if s["run_live"] and s["t"] >= probe.gate_t]
    assert running, "no sample while the run was live"
    bad = []
    for s in running:
        cards = s["cards"]
        if len(cards) != 1:
            bad.append(("cards", s))
            continue
        c = cards[0]
        expected = len(committed) + s["live_segments"]
        if c["id"] != id(card) or not c["live"] or not c["mounted"]:
            bad.append(("not the live card", s))
        elif c["phase"] in ENDED_PHASES or c["state"].startswith("Done"):
            bad.append(("ended look", s))
        elif c["title"] != EPUB_NAME:
            bad.append(("title", s))
        elif c["requests"] < len(committed):
            bad.append(("lost the committed cards", s))
        elif c["requests"] > expected:
            bad.append(("a request is listed twice", s))
        elif s["steady"] and s["run_state"] == "running" and c["requests"] < expected - REPAINT_LAG:
            bad.append(("the rows do not follow the job", s))
        elif c["chapter"] is not None and (c["chapter"][1] != CHAPTERS or c["chapter"][0] > s["job"]["completed"]):
            bad.append(("progress", s))
    assert not bad, "\n".join(f"{why}: {s}" for why, s in bad[:5])
    # the committed glossary cards stay listed first, the live translation requests follow
    labels = _labels(card)
    assert labels[:len(committed)] == committed, labels[:len(committed) + 2]
    assert len(set(labels)) == len(labels), "a request is listed twice"
    return running


def _check_result(probe: Probe, committed: list, phase: str, status: str) -> dict:
    """The turn's Result card: mounted, not live, the ended state, the book's title, every request of
    the turn once (the committed glossary cards first)."""
    result = probe.sample("result")
    assert len(result["cards"]) == 1, result
    c = result["cards"][0]
    ended = probe.job_cards()[0]
    assert c["mounted"] and not c["live"] and c["phase"] == phase, c
    assert c["state"] == status, (c, status)
    assert c["title"] == EPUB_NAME, c
    item = probe.turn_item()
    assert item is not None and not item.detached and item.index == probe.run().user_index
    assert c["requests"] == len(item.requests) and c["tile"] == f"Requests ({len(item.requests)})", (c, item)
    labels = _labels(ended)
    assert labels[:len(committed)] == committed and len(set(labels)) == len(labels), labels[:6]
    texts = probe.visible_texts()
    assert status in texts and EPUB_NAME in texts and "Attachment" not in texts, sorted(t for t in texts if len(t) < 60)
    assert probe.on_screen(ended.state_text)
    source = probe.view._turn_source(item)  # Reader / Library / QA know the turn's file
    assert source and Path(source).name == EPUB_NAME, source
    return {"card": ended, "state": c, "item": item, "labels": labels}


# ==========================================================================
# 1. The owner's flow: the gate, ✓ Yes, the live requests, Done · 210/210
# ==========================================================================


def test_book_card_stays_live_through_the_glossary_gate(iso, offline):
    async def body(ctx):
        app, server, run, probe = ctx.app, ctx.server, ctx.run, ctx.probe

        # ---- the gate ----------------------------------------------------------------------------
        assert await _until(lambda: run.awaiting_glossary, RUN_TIMEOUT), probe.describe()
        await asyncio.sleep(0.6)  # the gate's commit, the re-render and a few stream ticks
        glossary_requests = server.count("glossary")
        assert glossary_requests >= 3, glossary_requests
        gate = _check_gate(probe, glossary_requests=glossary_requests)
        card, committed = gate["card"], gate["committed"]
        assert server.count("translation") == 0, "chapters were sent before the approval"

        # ---- ✓ Yes on the chat's approval card ---------------------------------------------------
        await _tap(probe, app.chat_view.approval_card.yes_button)
        assert await _until(lambda: probe.job()["completed"] >= 30, RUN_TIMEOUT), probe.describe()
        # Hold the fake model while the mid-translation card is checked, so a fast runner cannot
        # finish the remaining chapters inside the wait (the job must still be RUNNING).
        server.hold()
        try:
            await asyncio.sleep(0.6)  # a stream tick (280-900 ms) and the next snapshot
            mid = probe.sample("mid-translation")
            assert mid["job"]["state"] == "RUNNING" and mid["job"]["total"] == CHAPTERS, mid
            c = _check_live_card(mid, card, committed)
            assert c["requests"] >= len(committed) + 25, c
            texts = probe.visible_texts()
            assert c["line"] in texts and c["state"] in texts and EPUB_NAME in texts
            assert "Done" not in texts and "Attachment" not in texts
            # the newest 20 rows are shown; "↑ Show … earlier requests" pages back to the glossary cards
            assert card.earlier_requests_button.visible and probe.on_screen(card.earlier_requests_button)
        finally:
            server.release()

        # ---- the end -------------------------------------------------------------------------------
        await _finished(app, run)
        await _stop_sampling(ctx)
        assert run.state == "done" and run.error is None, (run.state, run.error)
        assert probe.job()["state"] == "DONE"
        seen = _check_running_samples(probe, card, committed)
        # the card followed the job: its chapter count and its rows grew during the run
        chapters = [s["cards"][0]["chapter"][0] for s in seen if s["cards"][0]["chapter"]]
        requests = [s["cards"][0]["requests"] for s in seen]
        # How many distinct counts a sampler sees depends on the runner's speed: the card repaints on
        # the stream timer (280-900 ms) and the fake server finishes 210 chapters in a few seconds on
        # CI (Build Mobile 37937223020 saw 42, 84, 170, 210 with the 2 s progress poll; _owner_flow
        # polls every 0.2 s). What matters: intermediate progress was shown, it only rose, and it
        # reached the end of the book.
        distinct = sorted(set(chapters))
        assert len(distinct) >= 3 and any(0 < n < CHAPTERS for n in distinct), distinct[:20]
        assert max(chapters) >= 150, distinct[:20]
        assert chapters == sorted(chapters), "the progress went backwards"
        assert requests[0] == len(committed) and requests == sorted(requests), "the rows did not only grow"
        # the live card listed the job's own requests (Jobs › job: JobService.request_segments);
        # the last request or two may end between the card's last repaint and the run's commit
        job_segments = app.job_service.request_segments(run.job_id)
        job_labels = [str(s.get("label") or "") for s in job_segments]
        assert sum(1 for label in job_labels if label.startswith("Chapter ")) == CHAPTERS, len(job_labels)
        # an API client's lifecycle-only request (the EPUB metadata bookkeeping: "Sending API call" and no model
        # output) is a live row like on the desktop, and the shared commit drops it (DirectTextStream)
        lifecycle = [s for s in job_segments if _lifecycle_only(s)]
        shown_live = probe.last_live_labels[len(committed):]
        assert _in_stream_order(shown_live, job_labels) and len(job_labels) - len(shown_live) <= 2 + len(lifecycle), \
            (shown_live[:3], shown_live[-2:], job_labels[:3], job_labels[-2:], len(shown_live), len(job_labels))

        # ---- the Result: Done · 210/210 chapters, the book's title, every request once ----------------
        result = _check_result(probe, committed, "done", f"Done · {CHAPTERS}/{CHAPTERS} chapters")
        assert result["item"].report is not None
        assert sorted(_chapters(result["labels"])) == list(range(1, CHAPTERS + 1)), \
            "the Result does not list every chapter once"
        assert result["state"]["requests"] == len(committed) + len(job_labels) - len(lifecycle), \
            (result["state"], len(job_labels), [s.get("label") for s in lifecycle])

    with _server(0.01) as server:
        asyncio.run(_owner_flow(iso, server, body))
    _assert_offline(offline)


# ==========================================================================
# 2. Stop: Stopped + Resume on the turn's card
# ==========================================================================


def test_stop_shows_stopped_with_resume(iso, offline):
    async def body(ctx):
        app, server, run, probe = ctx.app, ctx.server, ctx.run, ctx.probe
        assert await _until(lambda: run.awaiting_glossary, RUN_TIMEOUT), probe.describe()
        await asyncio.sleep(0.6)
        gate = _check_gate(probe, glossary_requests=server.count("glossary"))
        card, committed = gate["card"], gate["committed"]
        await _tap(probe, app.chat_view.approval_card.yes_button)
        assert await _until(lambda: probe.job()["completed"] >= 20, RUN_TIMEOUT), probe.describe()
        await asyncio.sleep(0.6)
        _check_live_card(probe.sample("before Stop"), card, committed)

        # ---- ■ Stop on the running card ------------------------------------------------------------
        assert "stop" in card.action_buttons and probe.view.live_job_card is card
        await _tap(probe, card.action_buttons["stop"])
        assert await _until(lambda: run.stop_requested, 5)
        await _finished(app, run)
        await _stop_sampling(ctx)
        job = probe.job()
        assert job["state"] == "CANCELLED", job
        assert 20 <= job["completed"] < CHAPTERS, job
        assert run.state == "stopped", run.state
        _check_running_samples(probe, card, committed)

        result = _check_result(probe, committed, "stopped", f"Stopped · {job['completed']}/{CHAPTERS} chapters")
        ended = result["card"]
        assert "resume" in ended.action_buttons and probe.mounted(ended.action_buttons["resume"])
        assert probe.on_screen(ended.action_buttons["resume"])
        assert len(result["item"].requests) >= len(committed) + 20, result["item"]
        texts = probe.visible_texts()
        assert "Resume" in texts and not any(t == "Done" or t.startswith("Done ·") for t in texts)

    with _server(0.05) as server:  # a run long enough to stop in the middle
        asyncio.run(_owner_flow(iso, server, body))
    _assert_offline(offline)


# ==========================================================================
# 3. The app in the background: never "Done", one catch-up when it comes back
# ==========================================================================


def test_hidden_app_never_shows_done_and_catches_up(iso, offline):
    async def body(ctx):
        app, server, run, probe = ctx.app, ctx.server, ctx.run, ctx.probe
        page = probe.page

        # ---- Home pressed while the glossary is generated; the gate is reached in the background
        await probe.lifecycle(*LEAVE)
        assert page.app_visible is False
        assert await _until(lambda: run.awaiting_glossary, RUN_TIMEOUT), probe.describe()
        await asyncio.sleep(1.0)  # the gate's commit runs on the loop while hidden
        hidden_gate = probe.sample("hidden at the gate")
        assert not hidden_gate["visible"] and len(hidden_gate["cards"]) == 1, hidden_gate
        c = hidden_gate["cards"][0]
        assert c["phase"] not in ENDED_PHASES and not c["state"].startswith("Done") and c["title"] == EPUB_NAME, \
            f"the card at the gate (app hidden): {c}"

        # ---- back in the app at the gate ----------------------------------------------------------
        await probe.lifecycle(*COME_BACK)
        assert page.app_visible is not False
        await asyncio.sleep(1.0)  # the catch-up repaint
        gate = _check_gate(probe, glossary_requests=server.count("glossary"))
        card, committed = gate["card"], gate["committed"]
        await _tap(probe, app.chat_view.approval_card.yes_button)

        # ---- Home again mid-translation, back much later --------------------------------------------
        assert await _until(lambda: probe.job()["completed"] >= 15, RUN_TIMEOUT), probe.describe()
        await probe.lifecycle(*LEAVE)
        left = probe.sample("left mid-translation")
        assert await _until(lambda: probe.job()["completed"] >= left["job"]["completed"] + 60, RUN_TIMEOUT), \
            probe.describe()
        frozen = probe.sample("hidden mid-translation")
        assert not frozen["visible"] and len(frozen["cards"]) == 1 and frozen["cards"][0]["id"] == id(card), frozen
        c = frozen["cards"][0]
        assert c["live"] and c["phase"] == "running" and not c["state"].startswith("Done"), c
        # Hold the fake model across the return so the job is still RUNNING when the catch-up is
        # checked, whatever the runner speed.
        server.hold()
        try:
            await probe.lifecycle(*COME_BACK)
            await asyncio.sleep(1.2)  # the stream loop's catch-up tick and the next job snapshot
            back = probe.sample("returned to the app")
            assert back["job"]["state"] == "RUNNING", back  # still translating: the catch-up is a live one
            c = _check_live_card(back, card, committed)
            assert c["requests"] > frozen["cards"][0]["requests"] + 30, (c, frozen["cards"][0])
            assert c["chapter"][0] >= frozen["job"]["completed"], (c, frozen["job"])
            texts = probe.visible_texts()
            assert c["line"] in texts and "Done" not in texts and "Attachment" not in texts
        finally:
            server.release()

        await _finished(app, run)
        await _stop_sampling(ctx)
        assert run.state == "done" and probe.job()["state"] == "DONE", (run.state, probe.job())
        _check_running_samples(probe, card, committed)
        _check_result(probe, committed, "done", f"Done · {CHAPTERS}/{CHAPTERS} chapters")

    with _server(0.03) as server:
        asyncio.run(_owner_flow(iso, server, body))
    _assert_offline(offline)


# ==========================================================================
# 4. Found by this acceptance test: a render while the run commits listed every request twice
# ==========================================================================


class DoubleListing(AssertionError):
    """The running card listed a request twice."""


def test_a_render_while_the_run_commits_lists_each_request_once(iso, offline, monkeypatch):
    """A 12-chapter book, glossary off (only the finish commits cards): one render of the chat while
    ``ChatStore.finish_run`` has committed the run's cards and the run is still finishing (live) must
    list each of the turn's requests once ("Requests (N)" = the turn's committed requests)."""
    import direct_text_store

    seen: dict = {}

    async def body(ctx):
        app, run, probe = ctx.app, ctx.run, ctx.probe
        loop = asyncio.get_running_loop()
        view = app.chat_view
        original = direct_text_store.ChatStore.finish_run

        def finish_run(self, session, state, segments=None, **kwargs):
            result = original(self, session, state, segments, **kwargs)

            async def render():  # what any render does meanwhile (a job snapshot, an "earlier" tap)
                view.render_transcript()
                card = view.live_job_card
                return None if card is None else (_labels(card), str(card.requests_tile.title), run.state)

            seen["render"] = asyncio.run_coroutine_threadsafe(render(), loop).result(10)
            return result

        monkeypatch.setattr(direct_text_store.ChatStore, "finish_run", finish_run)
        await _finished(app, run)
        await _stop_sampling(ctx)
        assert run.state == "done" and probe.job()["state"] == "DONE", (run.state, probe.job())
        item = probe.turn_item()
        assert item is not None and len(item.requests) >= 12, item
        rendered = seen.get("render")
        assert rendered is not None, "the render during the commit found no live card"
        labels, tile, state = rendered
        assert state == "finishing", state
        if len(labels) != len(item.requests) or tile != f"Requests ({len(item.requests)})":
            raise DoubleListing(f"{tile} for {len(item.requests)} requests: {labels[:2]} … {labels[-2:]}")

    with _server(0.0) as server:
        asyncio.run(_owner_flow(iso, server, body, chapters=12, glossary="off"))
    _assert_offline(offline)


# ==========================================================================
# 5. Found by the DF2 verify: the resumed run's live requests are on its card
# ==========================================================================


def test_resume_lists_the_resumed_runs_live_requests(iso, offline):
    """A 40-chapter book, glossary off: ■ Stop after 8 chapters, then the ended card's Resume. While the
    resumed job translates, the running card lists the stopped run's committed cards followed by the
    resumed run's live requests (UI_SPEC §2.12.3: "the cards the run already committed (… an earlier run
    of the same turn) followed by the live ones"), "Requests (N)" grows with them, and their rows are on
    screen. The resumed run's requests continue the chat's request numbers, as a new Send would.

    The Stop is graceful and lands while the model holds the chapter requests the run already sent: the
    run stays Stopping, with no Resume, until they have answered, and their answers are committed with the
    stopped run, in order (desktop parity; test 6 covers Force stop, where they never answer).

    The live rows keep the shared stream's spine order, as the desktop Direct Text dialog lists them
    (``DirectTextStream._sort_active_request_segments``): the run's first request, the EPUB metadata
    translation, is a lifecycle-only "Request N" row that sorts after every chapter row, so a chapter
    dispatched later is inserted above it, not appended. The card (a repaint behind at most) is therefore
    compared with the stream once neither can change: the fake model parks every request after a fixed
    number of replies, counted by its own reply hook, and the check waits for the card's next repaint.
    No step depends on the runner's speed."""
    chapters = 40
    seen: dict = {}

    async def body(ctx):
        app, run, probe, server = ctx.app, ctx.run, ctx.probe, ctx.server
        view = app.chat_view

        # ---- ■ Stop mid-book, while the model holds the requests the run already sent -------------------------
        # Wait on the model only, not on the job's chapter count: translation_progress.json can lag the replies
        # while a request dispatched with them is parked (seen: completed 0 after 8 chapter replies, until the
        # fake model's 120 s hold timeout let the parked requests through).
        assert await _until(lambda: _settled(server), RUN_TIMEOUT), _held_state(probe, server)
        await _tap(probe, view.live_job_card.action_buttons["stop"])
        assert await _until(lambda: run.stop_requested, 5)
        # a graceful Stop waits for the sent requests: Stopping, and no Resume, while the model holds them
        assert await _until(lambda: _settled(server) and probe.job()["state"] == "STOPPING", 10), \
            (probe.job(), _held_state(probe, server))
        held = server.records("translation", status="pending")
        await asyncio.sleep(0.5)  # a stream tick: nothing ends the run while its sent requests are unanswered
        assert run.live and run.state == "stopping" and probe.job()["state"] == "STOPPING", (run.state, probe.job())
        assert "resume" not in view.live_job_card.action_buttons
        server.release()  # the held requests answer now, after the Stop
        await _finished(app, run)
        assert run.state == "stopped", run.state
        committed = probe.committed_labels()
        assert len(committed) >= 8, committed
        assert probe.job()["completed"] <= chapters - 20, (probe.job(), "too little left for the resumed run")
        # the late answers were committed with the stopped run, in the book's order
        late = _sent_chapters(held)
        assert held and all(r.status == "ok" for r in held), [(r.id, r.status) for r in held]
        stopped_chapters = _chapters(committed)
        assert stopped_chapters == sorted(set(stopped_chapters)), stopped_chapters
        assert late and set(late) <= set(stopped_chapters), (late, stopped_chapters)
        next_number = probe.chats.request_count(probe.cid) + 1
        assert next_number > int(run.params.get("request_number") or 0), (next_number, run.params.get("request_number"))

        # ---- Resume on the ended card (the model holds again after 6 replies of the resumed run) -------------
        ended = probe.job_cards()[0]
        assert "resume" in ended.action_buttons
        gate.hold_after(6)
        await _tap(probe, ended.action_buttons["resume"])
        assert await _until(lambda: _next_run(probe, run) is not None, 30)
        resumed = _next_run(probe, run)
        resume_t = time.monotonic()
        assert resumed.params.get("request_number") == next_number, resumed.params.get("request_number")
        try:
            assert await _until(lambda: _settled(server), RUN_TIMEOUT), _held_state(probe, server)
            # Once the held stream stops changing, the card's next repaint (a stream tick, 280-900 ms) lists it.
            snap: dict = {}
            assert await _until(lambda: _card_shows_stream(view, resumed, len(committed), snap, 5), 30), snap

            # ---- the running card: the stopped run's rows, then the resumed run's live ones -----------------
            # (no await from here to the end of the block: the card and the stream are still those of ``snap``)
            sample = probe.sample("resumed run")
            assert sample["job"]["state"] == "RUNNING" and sample["run_state"] == "running", sample
            card = view.live_job_card
            assert card is not None and len(sample["cards"]) == 1 and sample["cards"][0]["id"] == id(card), sample
            c = sample["cards"][0]
            assert c["live"] and c["mounted"] and c["phase"] == "running" and c["title"] == EPUB_NAME, c
            labels = _labels(card)
            assert labels[:len(committed)] == committed, labels[:len(committed) + 2]
            # exactly the resumed run's live requests, in the stream's (spine) order: the chapter rows, then the
            # lifecycle-only metadata row, as the desktop lists ``_active_request_segments``
            assert labels[len(committed):] == [label for _number, label in snap["live"]], \
                (labels[len(committed):][:3], snap["live"][:3])
            assert c["tile"] == f"Requests ({len(labels)})" and len(labels) > len(committed), c
            numbers = [number for number, _label in snap["live"] if number]
            assert numbers and min(numbers) >= next_number, (numbers[:5], next_number)
            assert set(labels[len(committed):]) & probe.visible_texts(rows=True), "the live request rows are not on screen"
            seen["committed"] = committed
        finally:
            server.release()

        # ---- the end: Done, the stopped run's cards first ----------------------------------------------
        await _finished(app, resumed)
        await _stop_sampling(ctx)
        assert resumed.state == "done" and resumed.error is None, (resumed.state, resumed.error)
        # every sample of the resumed run: the committed rows, never a row twice, and the live rows follow
        bad = []
        for s in probe.samples:
            if s["t"] < resume_t or not s["run_live"] or len(s["cards"]) != 1:
                continue
            count = s["cards"][0]["requests"]
            if count < len(committed):
                bad.append(("lost the committed cards", s))
            elif count > len(committed) + s["live_segments"]:
                bad.append(("a request is listed twice", s))
            elif s["steady"] and s["run_state"] == "running" and \
                    count < len(committed) + s["live_segments"] - REPAINT_LAG:
                bad.append(("the rows do not follow the resumed job", s))
        assert not bad, "\n".join(f"{why}: {s}" for why, s in bad[:5])
        result = _check_result(probe, committed, "done", f"Done · {chapters}/{chapters} chapters")
        assert len(result["labels"]) > len(committed), result["state"]

    with _server(0.08) as server:  # timing only: the holds decide when each step happens
        gate = HoldGate(server, 9)  # the EPUB metadata request + 8 chapters, then ■ Stop
        asyncio.run(_owner_flow(iso, server, body, chapters=chapters, glossary="off"))
    assert seen.get("committed"), "the scenario did not reach the resumed run"
    _assert_offline(offline)


# ==========================================================================
# 6. A force-stopped run's late answers stay out of the resumed run
# ==========================================================================


def test_a_force_stopped_runs_late_answers_stay_out_of_the_resumed_run(iso, offline):
    """A 40-chapter book, glossary off. The model holds the chapter requests the run already sent; ■ Stop,
    then Force stop: unlike the graceful Stop of test 5, which waits for them and commits their answers with
    the run, the run ends while they are unanswered, and the shared commit drops their status-only rows.
    Resume starts the next run while the model still holds them, and only then are they let through, so the
    stopped run's answers come after Resume (the force stop has hung up on them by then). Nothing of the
    stopped run reaches the chat, the resumed run's stream or its card: the chat keeps the stopped run's
    committed cards as they were, the resumed run's card lists them followed by its own live requests only,
    and the Result lists every chapter once, the stopped run's first (desktop parity: the Direct Text
    dialog's Force stop, then Resume). Every step waits on the fake model's state."""
    chapters = 40
    seen: dict = {}

    async def body(ctx):
        app, run, probe, server = ctx.app, ctx.run, ctx.probe, ctx.server
        view = app.chat_view
        service = app.job_service

        # ---- the model holds the run's sent requests; ■ Stop, then Force stop -----------------------------
        assert await _until(lambda: _settled(server), RUN_TIMEOUT), _held_state(probe, server)
        sent = {r.id for r in server.records("translation", status="pending")}
        await _tap(probe, view.live_job_card.action_buttons["stop"])

        def force_stop():
            card = view.live_job_card
            return None if card is None else card.action_buttons.get("force_stop")

        assert await _until(lambda: force_stop() is not None and probe.on_screen(force_stop()), 10), probe.describe()
        await _tap(probe, force_stop())
        await _finished(app, run)
        assert run.state == "stopped", run.state
        # the run ended before its sent requests answered: the model still holds every one of them
        held = server.records("translation", status="pending")
        assert sent and sent <= {r.id for r in held} and server.parked == len(held), \
            (sent, [(r.id, r.status) for r in held], server.parked)
        committed = probe.committed_labels()
        stopped_threads = _threads(service.request_segments(run.job_id))
        late = _sent_chapters(held)
        assert committed and stopped_threads and late, (committed[-2:], stopped_threads, late)
        # their status-only rows were dropped by the commit
        assert not set(late) & set(_chapters(committed)), (late, committed[-3:])
        next_number = probe.chats.request_count(probe.cid) + 1

        # ---- Resume while the model still holds the stopped run's requests ---------------------------------
        ended = probe.job_cards()[0]
        assert "resume" in ended.action_buttons
        gate.hold_after(6)  # the resumed run's own replies (the stopped run's late ones do not count)
        await _tap(probe, ended.action_buttons["resume"])
        assert await _until(lambda: _next_run(probe, run) is not None, 30)
        resumed = _next_run(probe, run)
        assert resumed.params.get("request_number") == next_number, resumed.params.get("request_number")
        # the resumed run's first request parks behind the stopped run's held ones
        assert await _until(lambda: server.parked > len(held), RUN_TIMEOUT), _held_state(probe, server)
        messages = len(probe.chats.messages(probe.cid))
        try:
            # ---- the stopped run's answers come now, after Resume -----------------------------------------
            server.release()
            assert await _until(lambda: all(r.status != "pending" for r in held), 30), [(r.id, r.status) for r in held]

            def stopped_alive() -> set:
                return stopped_threads & {thread.name for thread in threading.enumerate()}

            # its request threads have ended: nothing of the stopped run can arrive later
            assert await _until(lambda: not stopped_alive(), 60), sorted(stopped_alive())
            # the resumed run holds again after 6 replies: its card is checked while it runs
            assert await _until(lambda: _settled(server), RUN_TIMEOUT), _held_state(probe, server)
            snap: dict = {}
            assert await _until(lambda: _card_shows_stream(view, resumed, len(committed), snap, 5), 30), snap
            # (no await from here to the end of the block: the card and the stream are still those of ``snap``)
            assert resumed.live and probe.job()["state"] == "RUNNING", (resumed.state, probe.job())
            assert probe.committed_labels() == committed and len(probe.chats.messages(probe.cid)) == messages, \
                "the stopped run's late answers reached the chat"
            assert not _threads(resumed.stream.segments()) & stopped_threads, \
                "a request of the stopped run is among the resumed run's rows"
            assert _labels(view.live_job_card)[:len(committed)] == committed
            numbers = [number for number, _label in snap["live"] if number]
            assert numbers and min(numbers) >= next_number, (numbers[:5], next_number)
            seen["late"] = late
        finally:
            server.release()

        # ---- the end: Done, every chapter once, the stopped run's cards first ----------------------------
        await _finished(app, resumed)
        await _stop_sampling(ctx)
        assert resumed.state == "done" and resumed.error is None, (resumed.state, resumed.error)
        assert not _threads(service.request_segments(resumed.job_id)) & stopped_threads
        result = _check_result(probe, committed, "done", f"Done · {chapters}/{chapters} chapters")
        assert sorted(_chapters(result["labels"])) == list(range(1, chapters + 1)), \
            "the Result does not list every chapter once"

    with _server(0.08) as server:  # timing only: the holds decide when each step happens
        gate = HoldGate(server, 9)  # the EPUB metadata request + 8 chapters, then ■ Stop
        asyncio.run(_owner_flow(iso, server, body, chapters=chapters, glossary="off"))
    assert seen.get("late"), "the scenario did not reach the stopped run's late answers"
    _assert_offline(offline)
