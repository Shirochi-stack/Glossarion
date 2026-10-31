"""Hotfix 2: "our jobs pop up keeps nudging our page scroll up randomly, making it briefly hover over logs
while I was scrolled down to the bottom" (owner, Android phone, on the Job page of a running translation:
the Log card at the bottom of the page, the JobStrip "Translating · <book> · Ch 19/107 · 1 in flight ·
19:08" pinned below it).

What happened. The JobStrip's own ticks never move the page: ``JobStrip.set_model`` patches only the strip
(a patch rooted at the strip), its height is fixed by its row (single-line title and subtitle, the 48 dp
Stop button) and it sits in the View's footer, outside the page's ListView. But the same JobService snapshot
that ticks the strip also refreshes the Job page, and the page's content above the Log card grows while the
job runs: the Requests tile (expanded while the job is active) gets a card per request and every card's
preview grows to three lines, header lines come and go. Flutter keeps a list's scroll offset when content
above the viewport grows, so the Log card slid down under a reader parked at the page's end - the page
"scrolled up". The old remedy (``JobDetailScreen._keep_page_at_end``, 2026-10-21) chased the end with
``page_list.scroll_to(offset=-1)`` after each console batch: that invoke left in the same tick right behind
the patch with the new content, so the client ran it before laying the patch out, against the previous
``maxScrollExtent`` (a no-op), and the page stayed off its end until the next batch of log lines - the
"brief hover over the logs", in step with the strip's ticks.

The fix pins the page on the client, after layout: while the reader is at the page's end the page's
ListView has Flet's ``auto_scroll`` on (instant ``auto_scroll_animation``; Flet suspends it as soon as the
reader scrolls up and the screen turns it off), and every page refresh made while the reader is there also
changes the ListView itself, so its ScrollableControl rebuilds in the frame that lays out the new content
and re-pins in that frame's post-frame callback.

How it is measured. The REAL app runs on the fake Flet session (``test_bootstrap._fake_session``) with the
real JobsFeature / JobService (``test_jobs.FakeBackend``: a worker the test steps), the real shell, JobStrip,
JobDetailScreen and LogConsole, the dispatcher flushed tick by tick. ``PageListClient`` plays Flutter's part
for the page's ListView from the messages the session sends, following Flutter's scroll rules and Flet
1.0.3's ``scrollable_control.dart`` / ``scroll_notification_control.dart``: invokes run when their message
arrives (before the frame lays the patch out), the scroll offset stays when content changes (clamped to the
new end), ``auto_scroll`` pins after layout (a rebuild schedules a post-frame jump in the same frame, a
``ScrollMetricsNotification`` one in the next frame), its pin is suspended when the offset decreases and
resumed within 40 px of the end, and every offset change reaches the screen as a ``scroll`` event. Heights
come from ``estimate_height``, a coarse layout of the control tree (text lines by length and ``max_lines``,
fixed sizes for chips, buttons and bars): only its changes matter.

1. ``test_strip_ticks_patch_only_the_strip_and_never_move_the_page``: on the chat home and on the Job page,
   strip ticks send patches rooted at the strips only (nothing for the page's list, its View or the Page, no
   scroll invoke), the footer keeps its height and the reader's offset does not change.
2. ``test_job_page_stays_at_its_end_while_the_job_ticks``: the owner's complaint. A reader at the end of the
   Job page stays exactly there (the Log card does not move on screen) through job ticks that tick the strip
   and grow the Requests cards above the Log; scrolled up, nothing moves under them any more; no page
   ``scroll_to`` is ever sent.

What only a device can show: the frame that lays out new content above the Log is still painted with the
old offset before the post-frame pin (one 16 ms frame); Flutter has no anchored-at-end list short of a
reversed one.

Run from src/mobile with the mobile venv (Python 3.12 on the dev box; keep its PYTHONPATH)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_hotfix2_job_strip_scroll.py
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.util
import math
import queue
import re
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="needs flet and msgpack")


def _load(alias: str, file_name: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(file_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The fake Flet session and the app fixtures (test_bootstrap), the JobService fakes (test_jobs): one copy each.
_TB = _load("_glossarion_tb_hotfix2_strip", "test_bootstrap.py")
_J = _load("_glossarion_jobs_hotfix2_strip", "test_jobs.py")
storage = _TB.storage
app_env = _TB.app_env

TIMEOUT = 10.0
PAGE_HEIGHT = 860.0
PIN_SLACK = 40.0  # scrollable_control.dart: re-pins within 40 px of maxScrollExtent


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    for name in ("USERPROFILE", "APPDATA", "LOCALAPPDATA"):
        target = tmp_path / "env" / name
        target.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(name, str(target))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")


# ==========================================================================
# Coarse layout of a Flet control tree
# ==========================================================================


def _type_scale() -> dict:
    from glossarion_mobile.ui import tokens

    return tokens.TYPE_SCALE


def _sides(value: Any) -> tuple[float, float, float, float]:
    """(left, top, right, bottom) of an ``ft.Padding`` / ``ft.Margin`` / number / None."""
    if value is None:
        return 0.0, 0.0, 0.0, 0.0
    if isinstance(value, (int, float)):
        v = float(value)
        return v, v, v, v
    return tuple(float(getattr(value, side, 0) or 0) for side in ("left", "top", "right", "bottom"))  # type: ignore[return-value]


def _text_metrics(text: Any) -> tuple[float, float]:
    """(font size, line height) of an ``ft.Text``."""
    style = getattr(text, "theme_style", None)
    if style is not None:
        key = re.sub(r"(?<!^)(?=[A-Z])", "_", str(getattr(style, "value", style))).lower()
        spec = _type_scale().get(key)
        if spec is not None:
            return float(spec.size), float(spec.line)
    size = float(getattr(text, "size", None) or 14.0)
    ratio = getattr(getattr(text, "style", None), "height", None)
    return size, size * float(ratio or 1.43)


def _children(control: Any, name: str = "controls") -> list:
    return [c for c in (getattr(control, name, None) or []) if c is not None]


def _visible(control: Any) -> bool:
    return control is not None and getattr(control, "visible", True) is not False


def estimate_height(control: Any, width: float = 412.0) -> float:
    """The height Flutter would give ``control`` at ``width`` (coarse: only its changes matter)."""
    import flet as ft

    if not _visible(control):
        return 0.0
    _ml, mt, _mr, mb = _sides(getattr(control, "margin", None))
    margin = mt + mb
    if isinstance(control, ft.Text):
        size, line = _text_metrics(control)
        value = "" if control.value is None else str(control.value)
        per_line = max(8, int(width / (size * 0.55)))
        lines = sum(max(1, math.ceil(len(part) / per_line)) for part in value.split("\n"))
        if control.max_lines:
            lines = min(lines, int(control.max_lines))
        return lines * line + margin
    if isinstance(control, ft.IconButton):
        return 48.0 + margin
    if isinstance(control, (ft.Button, ft.OutlinedButton, ft.TextButton)):  # Filled / FilledTonal are Buttons
        return 40.0 + margin
    if isinstance(control, ft.Chip):
        return 32.0 + margin
    if isinstance(control, ft.ProgressBar):
        return 4.0 + margin
    if isinstance(control, ft.ProgressRing):
        return float(control.height or 36.0) + margin
    if isinstance(control, ft.Icon):
        return float(control.size or 24.0) + margin
    if isinstance(control, ft.ExpansionTile):
        body = 0.0
        if control.expanded:
            _l, top, _r, bottom = _sides(control.controls_padding)
            body = top + bottom + sum(estimate_height(c, width) for c in _children(control))
        return 56.0 + body + margin
    if isinstance(control, (ft.ListView, ft.Column)):
        left, top, right, bottom = _sides(getattr(control, "padding", None))
        if control.height is not None:
            return float(control.height) + margin
        items = [c for c in _children(control) if _visible(c)]
        spacing = float(control.spacing or 0)
        inner = width - left - right
        return top + bottom + sum(estimate_height(c, inner) for c in items) + spacing * max(0, len(items) - 1) + margin
    if isinstance(control, (ft.Row, ft.Stack)):
        heights = [estimate_height(c, width) for c in _children(control) if _visible(c)]
        own = float(getattr(control, "height", None) or 0)
        return max([own] + heights) + margin
    content = getattr(control, "content", None)
    if isinstance(control, ft.Container):
        left, top, right, bottom = _sides(control.padding)
        if control.height is not None:
            return float(control.height) + margin
        if not isinstance(content, ft.Control):
            return margin
        return top + bottom + estimate_height(content, width - left - right) + margin
    if isinstance(content, ft.Control):  # SafeArea, Semantics, GestureDetector, ...
        return estimate_height(content, width) + margin
    return 24.0 + margin


def item_offset(list_view: Any, target: Any, width: float = 412.0) -> float:
    """Content offset of ``target`` (one of ``list_view.controls``) from the list's top."""
    _l, top, _r, _b = _sides(list_view.padding)
    left, _t, right, _b2 = _sides(list_view.padding)
    offset = top
    spacing = float(list_view.spacing or 0)
    for item in _children(list_view):
        if item is target:
            return offset
        if _visible(item):
            offset += estimate_height(item, width - left - right) + spacing
    raise AssertionError("the control is not an item of the list")


# ==========================================================================
# The client side of the page's ListView
# ==========================================================================


def _ancestor_ids(control: Any) -> set:
    ids = set()
    parent = getattr(control, "parent", None)
    while parent is not None:
        ids.add(parent._i)
        parent = getattr(parent, "parent", None)
    return ids


def _inside(control: Any, root: Any) -> bool:
    node = control
    while node is not None:
        if node is root:
            return True
        node = getattr(node, "parent", None)
    return False


def _patch_targets(session: Any, body: Any) -> list[tuple[Any, Any, Any]]:
    """(operation, owning control, key) for every operation of a PATCH_CONTROL body."""
    from flet.controls.base_control import BaseControl

    root = session.index.get(body.id)
    tree, *ops = body.patch
    owners: dict[int, Any] = {}

    def walk(node: list, obj: Any, owner: Any) -> None:
        number = node[0]
        owner = obj if isinstance(obj, BaseControl) else owner
        owners[number] = owner
        for key, sub in (node[1] if len(node) > 1 else {}).items():
            try:
                child = obj[key] if isinstance(obj, (list, tuple, dict)) else getattr(obj, key)
            except Exception:
                child = None
            walk(sub, child, owner)

    walk(tree, root, root)
    return [(op[0], owners.get(op[1]), op[2] if len(op) > 2 else None) for op in ops]


class PageListClient:
    """Flutter's ScrollPosition plus Flet 1.0.3's ScrollableControl for one vertical ``ft.ListView``.

    * an INVOKE_METHOD ``scroll_to`` runs when its message is handled, before the frame lays out the patches
      that came with it: ``offset < 0`` targets the previous layout's ``maxScrollExtent + offset + 1``;
    * layout keeps the offset when content changes; an offset past the new end is clamped to it;
    * the controller listener (always attached) suspends the pin when the offset decreases by more than
      0.5 px and resumes it within 40 px of the end;
    * with ``auto_scroll`` on, a rebuild of the ListView (one of its own properties patched) schedules a
      post-frame jump to the end in that frame; a ``ScrollMetricsNotification`` (dispatched in a microtask
      after the frame) whose ``maxScrollExtent`` grew schedules one for the next frame;
    * a change of ``auto_scroll`` itself builds a new Flutter Scrollable (ScrollableControl wraps it in its
      listener only while ``auto_scroll`` is on): the offset starts again at 0 with the pin on (v9.15.7
      hotfix 1: on the emulator a scroll up that turned it off threw the Job page to its top);
    * every offset change reaches the server as a ``scroll`` event (scroll_notification_control.dart).
    """

    def __init__(self, session: Any, conn: Any, page_list: Any, viewport: Callable[[], float]) -> None:
        self.session = session
        self.conn = conn
        self.list = page_list
        self.viewport_fn = viewport
        self.cursor = len(conn.messages)
        self.pixels = 0.0
        self.max_extent: Optional[float] = None
        self.viewport = viewport()
        self.pinned = True  # _pinnedToEnd starts true
        self.auto = bool(page_list.auto_scroll)
        self.resets = 0  # new Scrollables built by an auto_scroll change
        self.last_pixels = 0.0
        self.last_auto_max = 0.0
        self.next_frame: list[Callable[[], None]] = []
        self.pending_events: list[dict] = []
        self.list_invokes: list[Any] = []
        self.frames = 0

    # ---- Flutter -------------------------------------------------------------------------------------

    def content_height(self) -> float:
        return estimate_height(self.list)

    def _set_pixels(self, value: float) -> None:
        value = max(0.0, min(float(self.max_extent or 0.0), float(value)))
        if abs(value - self.pixels) < 1e-6:
            return
        self.pixels = value
        # ScrollableControl._onScroll (the controller listener)
        if value < self.last_pixels - 0.5:
            self.pinned = False
        elif value >= float(self.max_extent or 0.0) - PIN_SLACK:
            self.pinned = True
        self.last_pixels = value
        self.pending_events.append({"event_type": "end", "pixels": value, "min_scroll_extent": 0.0,
                                    "max_scroll_extent": float(self.max_extent or 0.0),
                                    "viewport_dimension": self.viewport})

    def _scroll_to_end(self) -> None:  # _scheduleScrollToEnd's post-frame callback
        self.last_auto_max = float(self.max_extent or 0.0)
        if self.pinned:
            self._set_pixels(float(self.max_extent or 0.0))
            self.pinned = True

    def _rebuilds_list(self, body: Any) -> bool:
        return any(getattr(op, "value", op) == 0 and owner is self.list and isinstance(key, str)  # Replace
                   for op, owner, key in _patch_targets(self.session, body))

    async def frame(self) -> None:
        """Handle the messages sent since the last frame, lay out, run the post-frame callbacks."""
        from flet.messaging.protocol import MessageAction

        messages = self.conn.messages[self.cursor:]
        self.cursor = len(self.conn.messages)
        rebuilt = False
        for message in messages:
            if message.action == MessageAction.PATCH_CONTROL:
                rebuilt = rebuilt or self._rebuilds_list(message.body)
            elif message.action == MessageAction.INVOKE_METHOD and message.body.control_id == self.list._i:
                self.list_invokes.append(message.body)
                args = message.body.args or {}
                if message.body.name == "scroll_to" and args.get("offset") is not None:
                    offset = float(args["offset"])
                    if offset < 0:
                        offset = float(self.max_extent or 0.0) + offset + 1
                    self._set_pixels(offset)
        if bool(self.list.auto_scroll) != self.auto:  # a new Scrollable: offset 0, pin on, no scroll event
            self.auto = bool(self.list.auto_scroll)
            self.pixels = self.last_pixels = self.last_auto_max = 0.0
            self.pinned = True
            self.resets += 1
        # this frame's post-frame callbacks: the ones registered after the previous frame, then a rebuild's
        callbacks, self.next_frame = self.next_frame, []
        if rebuilt and self.list.auto_scroll:
            callbacks.append(self._scroll_to_end)
        # layout
        self.viewport = self.viewport_fn()
        self.max_extent = max(0.0, self.content_height() - self.viewport)
        if self.pixels > self.max_extent:
            self._set_pixels(self.max_extent)
        for callback in callbacks:
            callback()
        # the ScrollMetricsNotification microtask (listener wrapped while auto_scroll is on)
        if self.list.auto_scroll:
            if self.max_extent > self.last_auto_max + 0.5 and self.pinned:
                self.next_frame.append(self._scroll_to_end)
            self.last_auto_max = self.max_extent
        self.frames += 1
        await self.send_events()

    async def send_events(self) -> None:
        events, self.pending_events = self.pending_events, []
        for data in events:
            await self.session.dispatch_event(self.list._i, "scroll", data)

    # ---- the reader ----------------------------------------------------------------------------------

    async def user_scroll_to(self, pixels: float) -> None:
        """A drag that ends at ``pixels`` (then the frame that handles what the screen sent back)."""
        self._set_pixels(pixels)
        await self.send_events()
        await self.frame()

    @property
    def distance_from_end(self) -> float:
        return float(self.max_extent or 0.0) - self.pixels


# ==========================================================================
# App + job harness
# ==========================================================================


class Steps:
    """The job's worker: runs the test's steps on the job thread until told to finish."""

    def __init__(self) -> None:
        self.queue: "queue.Queue[Any]" = queue.Queue()
        self.entered = threading.Event()

    def __call__(self, owner: Any, request: Any) -> None:
        self.entered.set()
        while not owner.host.is_stop_requested():
            try:
                item = self.queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is None:
                return None
            fn, done = item
            try:
                fn(owner.host)
            finally:
                done.set()
        return None

    async def run(self, fn: Callable[[Any], None]) -> None:
        done = threading.Event()
        self.queue.put((fn, done))
        assert await asyncio.to_thread(done.wait, TIMEOUT), "the job thread did not run the step"

    def finish(self) -> None:
        self.queue.put(None)


async def _turns(count: int = 8) -> None:
    """Let the loop run what is queued (posted transitions, the scroll_to tasks): loop turns, not time."""
    for _ in range(count):
        await asyncio.sleep(0)


class Harness:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.steps = Steps()
        self.segments: list[dict] = []

    async def start(self) -> None:
        from glossarion_mobile.services.jobs import JobService, JobSpec
        from glossarion_mobile.ui.screens.jobs import JobsFeature

        spec = importlib.util.spec_from_file_location("glossarion_mobile_app_main_hotfix2_strip", APP_DIR / "main.py")
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        self.conn, self.session = _TB._fake_session("android")
        self.session.apply_page_patch({"width": 412, "height": PAGE_HEIGHT})
        self.page = self.session.page
        await main_module.main(self.page)
        await self.session.after_event(self.page)
        self.app = self.page.data
        # the pump is stopped: the test flushes the dispatcher one tick at a time
        await self.app.dispatcher.stop()
        backend = _J.FakeBackend(str(self.tmp_path / "out"))
        backend.behavior = self.steps
        self.service = JobService(jobs_dir=self.tmp_path / "jobs", logs_dir=self.tmp_path / "logs",
                                  data_dir=self.app.paths.data, config_store=_J.FakeStore(),
                                  dispatcher=self.app.dispatcher, backend=backend)
        # the request cards' segments (the shared request classifier's output) are the test's
        self.service.request_segments = lambda job_id: [dict(s) for s in self.segments]
        self.service.has_request_stream = lambda job_id: True
        self.feature = JobsFeature(self.app, service=self.service)
        self.feature.attach(self.app)
        self.feature.background.permissions = _J.FakePermissions()

        async def confirm(title: str, body: str) -> bool:
            return True

        self.feature.background.confirm = confirm
        book = Path(self.app.paths.data) / "Inbox" / "Book.epub"
        book.parent.mkdir(parents=True, exist_ok=True)
        book.write_bytes(b"PK fake epub")
        self.job_id = await self.feature.submit(JobSpec("translate", "나 혼자 메카닉 천재 소년.epub", (str(book),)))
        assert await asyncio.to_thread(self.steps.entered.wait, TIMEOUT)
        await self.steps.run(lambda host: (host.emit("progress", total=107, completed=18, in_progress=1, failed=0),
                                           host.emit("api_state", in_flight=1, queued=0)))
        await self.tick()
        # Hotfix 4: the job's start flips the root View's back handling once (AppShell._on_jobs_changed,
        # a whole-View update). That is the job starting, not a strip tick: let it land before the
        # ticks are measured (a fast runner delivered the Jobs badge inside the first tick).
        shell = self.app.shell
        keeps = getattr(shell, "_keeps_running", None)
        for _ in range(50):
            if keeps is None or (keeps() and shell._kept):
                break
            await self.tick()
        self.settled_back = keeps is None or (keeps() and shell._kept)

    async def tick(self) -> list:
        """One dispatcher tick; returns the messages it sent."""
        start = len(self.conn.messages)
        await _turns()
        self.app.dispatcher.flush()
        await _turns()
        return self.conn.messages[start:]

    async def job_tick(self, n: int) -> list:
        """What a running translation does between two strip ticks: a chapter done, its log lines, the
        Requests cards growing (a new request every other tick, the last preview a line longer each tick)."""
        if n % 2 == 0 or not self.segments:
            self.segments.append({"id": f"req-{len(self.segments)}", "label": f"Chapter {19 + len(self.segments)}",
                                  "phase": "streaming", "text": "", "text_tokens": 0})
        last = self.segments[-1]
        last["text"] = (last["text"] + " " + "<p>The aether that had been consumed by rage settled down.</p>").strip()
        last["text_tokens"] = int(last["text_tokens"]) + 40

        def work(host: Any) -> None:
            host.emit("progress", total=107, completed=19 + n, in_progress=1, failed=0)
            for k in range(3):
                host.log(f"<p>Swoooosh... line {n}.{k}<br/></p>")

        await self.steps.run(work)
        return await self.tick()

    async def stop(self) -> None:
        try:
            self.steps.finish()
            service = getattr(self, "service", None)
            if service is not None:
                await asyncio.to_thread(_J.wait_for, lambda: (service.snapshot(self.job_id) is None
                                                              or service.snapshot(self.job_id).is_terminal))
            feature = getattr(self, "feature", None)
            if feature is not None:
                feature.close()
            if service is not None:
                service.close()
        finally:
            app = getattr(self, "app", None)
            if app is not None:
                if getattr(app, "settings", None) is not None:
                    app.settings.close()
                await app.dispatcher.stop()
            await asyncio.sleep(0)

    # ---- what is on screen ------------------------------------------------------------------------------

    def job_screen(self) -> Any:
        from glossarion_mobile.ui.screens.job_detail import JobDetailScreen

        screen = self.app.shell.top_screen
        assert isinstance(screen, JobDetailScreen), screen
        return screen

    def footer_height(self) -> float:
        entry = self.app.shell.stack[-1]
        return estimate_height(entry.footer) if entry.footer is not None else 0.0

    def viewport(self) -> float:
        from glossarion_mobile.ui import tokens

        return PAGE_HEIGHT - tokens.SIZES["app_bar"] - self.footer_height()

    async def strip_tick(self, second: int) -> list:
        """The strip's elapsed timer moves on (a snapshot whose only visible change is the strip's subtitle);
        returns the messages it sent (the shell patches the strip as soon as the signal changes)."""
        model = self.app.state.job_strip.value
        assert model is not None and model.state == "running", model
        head = model.subtitle.rsplit(" · ", 1)[0]
        start = len(self.conn.messages)
        self.app.state.job_strip.set(dataclasses.replace(model, subtitle=f"{head} · 19:{second:02d}"))
        await self.tick()
        return self.conn.messages[start:]


def _strip_roots(h: Harness) -> set:
    roots = {h.app.shell.global_strip._i}
    chat_strip = getattr(getattr(h.app, "chat_view", None), "job_strip", None)
    if chat_strip is not None:
        roots.add(chat_strip._i)
    return roots


def _assert_strip_only(h: Harness, messages: list, page_list: Any = None) -> None:
    """Every patch of a strip tick is rooted at a strip (or outside the page's list and its ancestors), and
    nothing asks a list to scroll."""
    from flet.messaging.protocol import MessageAction

    strips = _strip_roots(h)
    patched = [m.body.id for m in messages if m.action == MessageAction.PATCH_CONTROL]
    assert any(i in strips for i in patched), "the strip tick did not reach the strip"
    guarded = set()
    if page_list is not None:
        guarded = _ancestor_ids(page_list) | {page_list._i}
    for message in messages:
        if message.action == MessageAction.PATCH_CONTROL:
            root = h.session.index.get(message.body.id)
            assert message.body.id in strips or (
                message.body.id not in guarded and (page_list is None or not _inside(root, page_list))), (
                f"a strip tick re-diffed {type(root).__name__} ({message.body.id}), not just the strip")
        elif message.action == MessageAction.INVOKE_METHOD:
            assert message.body.name != "scroll_to", f"a strip tick asked control {message.body.control_id} to scroll"


# ==========================================================================
# Tests
# ==========================================================================


def test_strip_ticks_patch_only_the_strip_and_never_move_the_page(app_env, tmp_path):
    async def scenario() -> None:
        h = Harness(tmp_path)
        try:
            await h.start()
            assert h.settled_back, "the running job never reached the root View's back handling"
            # chat home: the chat column's strip shows the job (not this chat's), above the composer
            assert h.app.chat_view.job_strip.visible
            for second in range(10, 13):
                _assert_strip_only(h, await h.strip_tick(second), page_list=h.app.chat_view.transcript)
            assert h.app.chat_view.job_strip.subtitle_text.value.endswith("19:12")

            # the Job page, read at its end
            await _TB._route(h.session, f"/jobs/{h.job_id}")
            await h.tick()
            screen = h.job_screen()
            assert h.app.shell.stack[-1].footer.content is h.app.shell.global_strip and h.app.shell.global_strip.visible
            client = PageListClient(h.session, h.conn, screen.page_list, h.viewport)
            await client.frame()
            await client.user_scroll_to(client.max_extent)
            await client.frame()
            assert client.distance_from_end == 0.0
            footer = h.footer_height()
            pixels = client.pixels
            log_top = item_offset(screen.page_list, _log_card(screen)) - client.pixels
            for second in range(13, 19):
                _assert_strip_only(h, await h.strip_tick(second), page_list=screen.page_list)
                await client.frame()
                assert h.footer_height() == footer, "the strip changed its height on a tick"
                assert client.pixels == pixels and client.distance_from_end == 0.0
                assert item_offset(screen.page_list, _log_card(screen)) - client.pixels == log_top
            assert h.app.shell.global_strip.subtitle_text.value.endswith("19:18")
            assert not client.list_invokes
        finally:
            await h.stop()

    asyncio.run(scenario())


def _log_card(screen: Any) -> Any:
    for item in screen.page_list.controls:
        if getattr(item, "key", None) == "job-log":
            return item
    raise AssertionError("the Job page has no Log card")


def test_job_page_stays_at_its_end_while_the_job_ticks(app_env, tmp_path, monkeypatch):
    from glossarion_mobile.ui.screens import job_detail

    monkeypatch.setattr(job_detail, "REQUEST_REFRESH_S", 0.0)  # every tick refreshes the cards (no clock)

    async def scenario() -> None:
        h = Harness(tmp_path)
        try:
            await h.start()
            await _TB._route(h.session, f"/jobs/{h.job_id}")
            await h.tick()
            screen = h.job_screen()
            assert screen.requests_tile.expanded  # the Requests cards are open while the job runs
            client = PageListClient(h.session, h.conn, screen.page_list, h.viewport)
            for n in range(3):  # a few requests and log lines before the reader goes down
                await h.job_tick(n)
                await client.frame()
            first = 3
            while screen.requests_box.height is None:  # the page's height is still once the cards fill their box
                await h.job_tick(first)
                await client.frame()
                first += 1
                assert first < 15, "the cards never filled their box"
            await client.user_scroll_to(client.max_extent)
            await client.frame()
            assert client.distance_from_end == 0.0
            log_card = _log_card(screen)
            log_top = item_offset(screen.page_list, log_card) - client.pixels
            footer = h.footer_height()
            heights = [client.content_height()]
            strip_texts = [h.app.shell.global_strip.subtitle_text.value]
            for n in range(first, 15):
                await h.job_tick(n)
                await client.frame()
                heights.append(client.content_height())
                strip_texts.append(h.app.shell.global_strip.subtitle_text.value)
                assert h.footer_height() == footer
                moved = item_offset(screen.page_list, log_card) - client.pixels - log_top
                assert client.distance_from_end == 0.0 and moved == 0.0, (
                    f"tick {n}: the page left its end by {client.distance_from_end:.0f} dp - the Log card moved "
                    f"{moved:+.0f} dp under the reader (content {heights[-2]:.0f} -> {heights[-1]:.0f} dp)")
            # the scenario did what the phone does: the strip ticked and the requests above the Log grew - in
            # their box of fixed height since the v9.15.7 batch (test_mobile_batch_9157_log_scroll), so the
            # page itself kept its height
            assert len(set(strip_texts)) > 5
            assert len(screen.requests_column.controls) >= 8 and screen.requests_box.height is not None
            assert heights[-1] == heights[0]
            assert not [b for b in client.list_invokes if b.name == "scroll_to"], "the page chased its end"

            # a reader who scrolled up keeps their place: nothing moves under them any more. auto_scroll stays
            # on (the client suspended its pin): turning it off threw the page to its top (v9.15.7 hotfix 1)
            resets = client.resets
            target = client.pixels - 300
            await client.user_scroll_to(target)
            await client.frame()
            assert client.pixels == target, f"a scroll up threw the page to {client.pixels:.0f} dp"
            assert screen.page_list.auto_scroll and client.resets == resets
            pixels = client.pixels
            for n in range(15, 18):
                await h.job_tick(n)
                await client.frame()
                assert client.pixels == pixels
            assert client.distance_from_end >= 300  # the page no longer grows below them (requests box)

            # back at the end, the page follows it again
            await client.user_scroll_to(client.max_extent)
            await client.frame()
            assert screen.page_list.auto_scroll and client.resets == resets  # re-pinned, no new Scrollable
            await h.job_tick(18)
            await client.frame()
            assert client.distance_from_end == 0.0
            assert client.resets == 1  # turned on once, when the reader first reached the end
        finally:
            await h.stop()

    asyncio.run(scenario())
