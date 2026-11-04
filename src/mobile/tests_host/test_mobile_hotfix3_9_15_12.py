"""Mobile v9.15.12 hotfix 3: the on-device progress line says what happens before the first output token.

Owner: Gemma 4 reads a long prompt for minutes ("SDK stream opened in 147.5s"), and hotfix 2's line showed
"0 tokens, 0.0 tok/s" all that time. Until the first output token the line is the prompt's progress
(``processing, 1,024 of ~1,536 tokens``), then the tokens written.

Run from src/mobile (mobile venv):
    python -m pytest -p no:cacheprovider -W ignore tests_host/test_mobile_hotfix3_9_15_12.py -q
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import types
from pathlib import Path

MOBILE_DIR = Path(__file__).resolve().parents[1]
if str(MOBILE_DIR / "app") not in sys.path:
    sys.path.insert(0, str(MOBILE_DIR / "app"))

BATCH = 512


class _Ctx:
    ctx = "ctx-pointer"

    def decode(self, batch):
        return 0


class _Llama:
    """Reads a prompt in three 512-token batches (slowly), then writes four tokens."""

    def __init__(self) -> None:
        self._ctx = _Ctx()

    def tokenize(self, text, add_bos=False, special=True):
        return list(range(3 * BATCH))  # count_prompt: ~1,536

    def create_chat_completion(self, **kwargs):
        for _ in range(3):
            time.sleep(0.12)
            self._ctx.decode(types.SimpleNamespace(batch=types.SimpleNamespace(n_tokens=BATCH)))
        for _ in range(4):
            time.sleep(0.03)
            yield {"choices": [{"delta": {"content": "x"}, "finish_reason": None}]}
        time.sleep(0.12)
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}


def test_the_progress_line_shows_the_prompt_being_read_then_the_tokens_written(monkeypatch):
    from glossarion_mobile.services.local_llm import engine as eng

    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(
        llama_set_n_threads=lambda *a: None, llama_n_threads=lambda ctx: 4, LogitsProcessorList=list))
    monkeypatch.setattr(eng.os, "sched_getaffinity", lambda pid: set(range(8)), raising=False)
    monkeypatch.setattr(eng, "HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(eng, "_cpuset", lambda: "/moderate")

    lines: list = []

    class Grab(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    parent = eng.progress_log
    handler, level = Grab(), parent.level
    parent.addHandler(handler)
    parent.setLevel(logging.INFO)
    try:
        engine = eng.LlamaEngine()
        engine.llm, engine.path, engine.n_ctx, engine.n_threads = _Llama(), "m.gguf", 8192, 4
        assert "".join(engine.chat([{"role": "user", "content": "hi"}], {"max_tokens": 8}, lambda: False)) == "xxxx"
    finally:
        parent.removeHandler(handler)
        parent.setLevel(level)
    beats = [line for line in lines if line.startswith("📱 on-device: ") and "threads, CPUs" in line]
    reading = [line for line in beats if "processing" in line]
    written = [line for line in beats if "tokens written" in line]
    assert reading and written and beats.index(reading[-1]) < beats.index(written[0])
    assert not any("0 tokens" in line or "0.0 tok/s" in line for line in reading)
    assert any("of ~1,536 tokens" in line for line in reading)  # the prompt's progress, not "0 tokens"
    assert all(line.endswith("4 threads, CPUs 0-7, cpuset /moderate") for line in beats)
    assert "4 tokens written" in written[-1]


def test_line_texts():
    from glossarion_mobile.services.local_llm import engine as eng

    beat = eng._Heartbeat(types.SimpleNamespace(llm=None, n_threads=4))
    beat.prompt_total = 5896
    assert beat.line(15, 0).startswith("📱 on-device: processing (~5,896 tokens), 4 threads")
    beat.read(512)
    beat.read(512)
    beat.started = time.monotonic() - 32  # the average since reading began, not the last interval
    assert "processing, 1,024 of ~5,896 tokens (32.0 tok/s)" in beat.line(15, 0)
    beat.reading, beat.tokens = False, 274
    assert "274 tokens written, 4.4 tok/s (last 15 s)" in beat.line(15, 66)
    unknown = eng._Heartbeat(types.SimpleNamespace(llm=None, n_threads=2))
    assert "processing (? tokens)" in unknown.line(15, 0)
    assert eng._batch_tokens(types.SimpleNamespace(batch=types.SimpleNamespace(n_tokens=64))) == 64
    assert eng._batch_tokens(None) == 0 and threading.active_count() >= 1


# ---- the SDLXLIFF reviewer (owner: the workspace did not load; the Notepad switch beside Threshold) ----------


class _Prefs:
    def __init__(self) -> None:
        self.refs: dict = {}
        self.data: dict = {}

    def file_ref(self, path, *, kind=None):
        from glossarion_mobile.state.prefs import file_ref_id

        fid = file_ref_id(str(path))
        self.refs[fid] = str(path)
        return fid

    def resolve_file_ref(self, fid, *, touch=True):
        return self.refs.get(fid)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value


def test_the_output_tabs_sdlxliff_file_opens_its_workspace_and_piece(tmp_path):
    """The book page's Output tab sent the book id as ``?out=``, which the reviewer reads as a file reference:
    it opened empty. It now opens the sidecar's workspace, focused on its piece, with the book's raw file."""
    from glossarion_mobile.ui.library.output_tab import OutputTab
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools import sdlxliff as sx

    workspace = tmp_path / "Output" / "Book"
    (workspace / "SDLXLIFF").mkdir(parents=True)
    sidecar = workspace / "SDLXLIFF" / "response_ch002.html.sdlxliff"
    sidecar.write_text("<xliff/>", encoding="utf-8")
    raw = tmp_path / "Book.epub"
    raw.write_bytes(b"PK")
    went: list = []
    prefs = _Prefs()
    ctx = types.SimpleNamespace(prefs=prefs, go=lambda *a: went.append(a) or "ok")
    page = types.SimpleNamespace(bid="b1", book={"name": "Book"}, service=types.SimpleNamespace(
        raw_source=lambda book: str(raw), bid_for=lambda book: "b1"))
    tab = types.SimpleNamespace(ctx=ctx, page=page)
    fid = OutputTab._open_reviewer(tab, str(sidecar))
    assert went[-1] == ("tools.sdlxliff", None, {"out": fid})
    screen = sx.SdlxliffScreen(parse_route(f"/tools/sdlxliff?out={fid}"), ctx)
    assert screen.folder == str(workspace) and screen.error is None
    assert screen.request.focus == "response_ch002.html" and screen.request.source == str(raw)


def test_tools_hub_scans_the_library_before_looking_for_the_last_workspace(monkeypatch):
    import asyncio

    from glossarion_mobile.ui.tools import sdlxliff as sx

    order: list = []

    class Service:
        snapshot = types.SimpleNamespace(scanned_at=None)

        async def refresh(self, quiet=False, reason=""):
            order.append("refresh")
            self.snapshot.scanned_at = 1.0

    async def io(fn, *args):
        return fn(*args)

    monkeypatch.setattr(sx, "last_workspace", lambda prefs, service, root: order.append("look") or None)
    screen = sx.SdlxliffScreen.__new__(sx.SdlxliffScreen)
    screen.ctx = types.SimpleNamespace(service=Service(), prefs=None, chats_root="", io=io)
    screen.folder, screen._visible = "", True
    assert asyncio.run(screen.open_last()) is None
    assert order == ["refresh", "look"]
    assert asyncio.run(screen.open_last()) is None
    assert order == ["refresh", "look", "look"]  # scanned once


def test_the_notepad_switch_shares_the_actions_row():
    import flet as ft

    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools import sdlxliff as sx

    ctx = types.SimpleNamespace(prefs=_Prefs(), tablet=False)
    screen = sx.SdlxliffScreen(parse_route("/tools/sdlxliff"), ctx)
    screen.build_body()
    rows = [c for c in screen.header_column.controls if isinstance(c, ft.Row)]
    actions = next(r for r in rows if r.key == "sdl-actions")
    # hotfix 4: Provider and Threshold went behind the options button; the switch is still in this row
    assert actions.controls[:3] == [screen.mt_button, screen.flag_button, screen.options_button]
    assert actions.controls[3].content.controls[-1] is screen.layout_switch


def test_notepad_layout_fills_the_screen_and_the_header_steps_aside_for_the_keyboard():
    """Owner: in the Notepad layout the keyboard covered the text and a tap closed the keyboard. The editor was
    a fixed-height WebView inside a scrolling list; it now fills the page (which no longer scrolls) and the
    header hides while the keyboard is up, so the WebView shrinks with it and keeps the caret in view."""
    import flet as ft

    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools import sdlxliff as sx

    media = lambda bottom: types.SimpleNamespace(view_insets=types.SimpleNamespace(bottom=bottom))  # noqa: E731
    page = types.SimpleNamespace(on_media_change=None, media=media(0))
    shell_calls: list = []
    page.on_media_change = lambda e=None: shell_calls.append(e)
    shell = page.on_media_change
    ctx = types.SimpleNamespace(prefs=_Prefs(), tablet=False, page=page, spawn=lambda c: c.close())
    screen = sx.SdlxliffScreen(parse_route("/tools/sdlxliff"), ctx)
    screen.build_body()
    screen._push = lambda *c: None
    assert isinstance(screen.list_view, ft.Column) and screen.list_view.scroll == ft.ScrollMode.AUTO
    screen._hook_keyboard()
    assert page.on_media_change is not shell

    page.on_media_change(media(300))  # rows layout: the keyboard changes nothing, the shell still hears it
    assert screen.header_column.visible is not False and shell_calls
    page.on_media_change(media(0))
    screen.notepad = True
    screen.controls_open = True  # hotfix 4: the header shows in the Notepad layout only behind its Controls button
    screen._apply_notepad_layout()
    assert screen.list_view.scroll is None and screen.body_holder.expand is True
    page.on_media_change(media(320))  # keyboard up: the editor gets the whole visible height
    assert screen.header_column.visible is False
    page.on_media_change(media(0))
    assert screen.header_column.visible is True
    screen.notepad = False
    screen._apply_notepad_layout()
    assert screen.list_view.scroll == ft.ScrollMode.AUTO and not screen.body_holder.expand
    screen._unhook_keyboard()
    assert page.on_media_change is shell


# ---- the job notification: an on-device row and the streamed text, updated every second ------------------------


def _jobs_helpers():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_glossarion_jobs_helpers_hf3", Path(__file__).with_name("test_jobs.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stream_row_is_the_end_of_the_streamed_text_without_tags():
    from glossarion_mobile.services.jobs import NOTIFICATION_STREAM_CHARS, stream_row

    assert stream_row("") == "" and stream_row("<br/><br/>") == ""
    assert stream_row("<p>Hello &amp; welcome</p>\n<br/><p>back.</p>") == "Hello & welcome back."
    long = "<p>" + " ".join(f"word{i}" for i in range(80)) + "</p>"
    row = stream_row(long)
    assert row.startswith("…word") and row.endswith("word79") and len(row) <= NOTIFICATION_STREAM_CHARS + 1


def test_job_keeps_the_tail_of_its_streamed_text_and_drops_it_for_the_next_request(tmp_path):
    from glossarion_mobile.services import jobs as jb

    job = types.SimpleNamespace(stream=types.SimpleNamespace(feed=lambda raw: "suppress-main-log" if raw.startswith("<") else None),
                                stream_tail="", stream_open=False, buffer=None, sign_in_flagged=True, last_line="",
                                spec=None, id="j1")
    service = jb.JobService.__new__(jb.JobService)
    service._deliver_message = lambda job, raw: None
    service._write_log_file = lambda job, line: None
    service._note_issue = lambda job, line: None
    service._job_log(job, "<p>first part</p>", {})
    service._job_log(job, "<p>second</p>", {})
    assert job.stream_tail == "<p>first part</p><p>second</p>"
    service._job_log(job, "📤 [Thread-2 (api_call)] Chapter 7 — Sending API call now", {})
    assert job.stream_tail == ""
    service._job_log(job, "<p>" + "x" * 3000 + "</p>", {})
    assert len(job.stream_tail) == jb.STREAM_TAIL_KEEP


def test_notification_shows_the_on_device_row_and_the_stream_and_ticks_while_visible(monkeypatch):
    import asyncio

    from glossarion_mobile.services import jobs as jb
    from glossarion_mobile.services.jobs import JobState, Progress

    helpers = _jobs_helpers()
    background, native, clock, fake_jobs = helpers._background("android", update_interval=0.05)
    tails = {"text": ""}
    fake_jobs.stream_tail = lambda job_id=None: tails["text"]
    rows = {"ondevice": ""}
    monkeypatch.setattr(jb, "on_device_row", lambda: rows["ondevice"])

    def texts():
        return [c[2].get("text") for c in native.calls if c[0] == "update_job_service"]

    async def scenario():
        await background.job_started(helpers._snap(state=JobState.STARTING))
        snap = helpers._snap(progress=Progress(total=80, completed=1))
        fake_jobs.snap = snap
        clock.now += 1
        assert await background.job_progress(snap)
        assert texts()[-1] == "Book.epub" and not background.ticker_running  # nothing live: no rows, no ticker
        rows["ondevice"] = "📱 Processing · 512 / ~5,896 tokens · 32.0 tok/s"
        clock.now += 1
        assert await background.job_progress(snap)
        assert texts()[-1] == "Book.epub\n📱 Processing · 512 / ~5,896 tokens · 32.0 tok/s"
        assert background.ticker_running  # live rows: the ticker runs though the app is visible
        rows["ondevice"] = "📱 12 tokens written · 6.5 tok/s"
        tails["text"] = "<p>The secret proposal of the noble faction</p>"
        clock.now += 1
        assert await helpers._until(lambda: texts()[-1] == "Book.epub\n📱 12 tokens written · 6.5 tok/s\n"
                                                           "✍️ The secret proposal of the noble faction")
        rows["ondevice"], tails["text"] = "", ""  # the request ended: rows go, the ticker stops
        clock.now += 1
        assert await helpers._until(lambda: texts()[-1] == "Book.epub")
        assert await helpers._until(lambda: not background.ticker_running)
        # a question: no rows
        rows["ondevice"] = "📱 1 tokens written · 1.0 tok/s"
        assert background._live_rows(helpers._snap(question={"kind": "glossary"})) == []
        background.close()

    asyncio.run(scenario())
