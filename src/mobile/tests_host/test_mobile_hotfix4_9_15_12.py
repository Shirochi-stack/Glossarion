"""Mobile v9.15.12 hotfix 4.

Owner: "move that on device log to a status bar next to the Log label here and poll it like the notification
area does" and "I don't see any real time streaming in the notification area".

1. The engine's progress / thread lines leave the job's Log (diagnostics log only); the job screen shows the
   notification's on-device row with the thread count beside the Log title, read every second.
2. The OpenAI-compatible routes (on-device ``local/`` among them) log no "Text streaming" banner, so the request
   stream never marked their text as streamed and the notification's stream rows stayed empty. A job now takes
   the messages between a stream's start and its end that are not status lines.

Run from src/mobile (mobile venv):
    python -m pytest -p no:cacheprovider -W ignore tests_host/test_mobile_hotfix4_9_15_12.py -q
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")
HERE = Path(__file__).resolve().parent
for path in (HERE.parent / "app", HERE.parent.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

#: what a streaming ``local/`` request logs (captured from ``UnifiedClient`` against the on-device server), with
#: the waits a long Gemma prompt adds before its first token
LOCAL_STREAM = (
    "📤 [Thread-2 (api_call)] Chapter 8 — Sending API call now",
    "⏳ [Thread-2 (api_call)] Chapter 8 API call in progress",
    "📱 [local] On-device request: gemma-4-e2b",
    "🛰️ [local] SDK stream start (model=gemma-4-e2b, base_url=http://127.0.0.1:60223/…/v1)",
    "⏳ [Thread-2 (api_call)] Chapter 8 API call in progress",
    "🛰️ [local] SDK stream opened in 147.5s",
    "<p>After all, between friends of opposite sexes,",
    " it often develops into a romantic relationship.</p>",
    "",
    "<p>But the problem is that there was no such atmosphere between Erica and me.</p>",
    "⏳ [Thread-2 (api_call)] Chapter 8 API call in progress",
    "<p>In the end, she is just a toy worth mocking to me now.</p>",
    "   ✅ Received response (412 chars, finish_reason: stop)",
    "💾 Saved chapter 8",
)


def _job(stream):
    return types.SimpleNamespace(stream=stream, stream_tail="", stream_open=False, buffer=None, sign_in_flagged=True,
                                 last_line="", spec=None, id="j1")


def _service():
    from glossarion_mobile.services import jobs as jb

    service = jb.JobService.__new__(jb.JobService)
    service._deliver_message = lambda job, raw: None
    service._write_log_file = lambda job, line: None
    service._note_issue = lambda job, line: None
    return service


def test_on_device_streamed_text_reaches_the_notification_rows():
    import direct_text_stream

    from glossarion_mobile.services.jobs import stream_row

    service = _service()
    job = _job(direct_text_stream.make_stream(source_is_attachment=True, model="local/gemma-4-e2b"))
    tails = []
    for message in LOCAL_STREAM:
        service._job_log(job, message, {})
        tails.append(job.stream_tail)
    assert tails[5] == ""  # nothing before the first token
    assert job.stream_tail == ("<p>After all, between friends of opposite sexes, it often develops into a romantic "
                               "relationship.</p><p>But the problem is that there was no such atmosphere between "
                               "Erica and me.</p><p>In the end, she is just a toy worth mocking to me now.</p>")
    assert not job.stream_open  # "Received response" closed it: the save line is not text
    assert stream_row(job.stream_tail).endswith("she is just a toy worth mocking to me now.")
    service._job_log(job, LOCAL_STREAM[0], {})  # the next request
    assert job.stream_tail == "" and not job.stream_open


def test_status_lines_the_stream_marks_as_text_stay_out_of_the_notification():
    """Owner's phone (AuthND, 5 in flight): "…acquired; sending NVIDIA request⏳ AuthND: NVIDIA queue / prefill —
    request submitted" in the notification. The request stream's own hint marked those lines as streamed."""
    from glossarion_mobile.services.jobs import live_rows, stream_status_line

    service = _service()
    job = _job(types.SimpleNamespace(feed=lambda raw: "suppress-main-log"))  # the hint says "text" for all
    for message in ("📨 AuthND: captcha token acquired; sending NVIDIA request",
                    "⏳ AuthND: NVIDIA queue / prefill — request submitted; waiting for response headers",
                    "<p>Erica smiled.</p>", "[DEBUG] chunk", "<p>\"......?\"</p>"):
        service._job_log(job, message, {})
    assert job.stream_tail == "<p>Erica smiled.</p><p>\"......?\"</p>"
    assert not stream_status_line("<p>text</p>") and not stream_status_line("") and stream_status_line("✓ done")
    snap = types.SimpleNamespace(question=None, is_active=True)
    assert live_rows(snap, job.stream_tail)[-1] == "✍️ Erica smiled. \"......?\""


def test_streaming_off_and_status_lines_add_nothing():
    import direct_text_stream

    service = _service()
    job = _job(direct_text_stream.make_stream(source_is_attachment=True, model="local/gemma-4-e2b"))
    for message in ("📤 [Thread-2 (api_call)] Chapter 8 — Sending API call now",
                    "📱 [local] On-device request: gemma-4-e2b",
                    "🛰️ [local] SDK call finished in 31.0s, got choices=1",
                    "   ✅ Received response (412 chars, finish_reason: stop)",
                    "<p>a whole answer some route logs after it ends</p>"):
        service._job_log(job, message, {})
    assert job.stream_tail == "" and not job.stream_open
    for message in ("🛰️ [openai] SDK stream start (model=gpt, base_url=x)", "[DEBUG] chunk 3", "⚠️ slow chunk",
                    "Stream finished in 2.0s"):
        service._job_log(job, message, {})
    assert job.stream_tail == "" and not job.stream_open


def test_engine_progress_stays_out_of_the_job_log_and_the_row_names_the_threads():
    from glossarion_mobile.services.local_llm import engine as eng

    assert not eng.progress_log.name.startswith("unified_api_client")
    beat = eng._Heartbeat(types.SimpleNamespace(llm=None, n_threads=4))
    beat.reading, beat.tokens = False, 412
    assert beat.notification_row().startswith("📱 412 tokens written · ") and "threads" not in beat.notification_row()
    assert beat.notification_row(threads=True).endswith(" tok/s · 4 threads")


def test_the_log_title_shows_the_on_device_status_every_second(monkeypatch):
    pytest.importorskip("flet")
    import flet as ft

    from glossarion_mobile.services.jobs import JobState
    from glossarion_mobile.ui.components.section_card import SectionCard
    from glossarion_mobile.ui.screens import job_detail as jd

    # the status sits in the Log card's title row, taking its free width
    extra = ft.Text("status")
    card = SectionCard(title="Log", icon="TERMINAL", title_extra=extra)
    card.init()
    header = card.content.controls[0]
    title_row = header.controls[1].controls[0]
    assert title_row.controls[-1] is extra and not title_row.tight
    plain = SectionCard(title="Log", icon="TERMINAL")
    plain.init()
    assert plain.content.controls[0].controls[1].controls[0].tight

    rows = {"row": ""}
    seen: list = []
    monkeypatch.setattr(jd, "on_device_row",
                        lambda threads=False: rows["row"] + (" · 4 threads" if threads and rows["row"] else ""))
    monkeypatch.setattr(jd, "DEVICE_STATUS_SECONDS", 0.01)
    screen = jd.JobDetailScreen.__new__(jd.JobDetailScreen)
    screen.snap = types.SimpleNamespace(question=None, is_active=True, is_terminal=False, state=JobState.RUNNING)
    screen.device_text = ft.Text("")
    screen.device_status = ft.Container(content=screen.device_text, visible=False)
    screen._unsubs = [lambda: None]
    screen._update_device_status = lambda: seen.append((screen.device_text.value, screen.device_status.visible))

    async def scenario():
        task = asyncio.ensure_future(screen._follow_device_status())
        await asyncio.sleep(0.05)
        assert seen == []  # no request: nothing shown, nothing sent
        rows["row"] = "📱 Processing · 512 / ~5,896 tokens · 32.0 tok/s"
        await asyncio.sleep(0.05)
        assert seen == [("📱 Processing · 512 / ~5,896 tokens · 32.0 tok/s · 4 threads", True)]
        rows["row"] = "📱 12 tokens written · 6.5 tok/s"
        await asyncio.sleep(0.05)
        assert seen[-1] == ("📱 12 tokens written · 6.5 tok/s · 4 threads", True)
        screen.snap = types.SimpleNamespace(question=None, is_active=False, is_terminal=True, state=JobState.DONE)
        await asyncio.wait_for(task, 1)  # the job ended: the status goes, the loop ends
        assert seen[-1] == ("", False)

    asyncio.run(scenario())
    # waiting for an answer: no status
    screen.snap = types.SimpleNamespace(question={"kind": "glossary"}, is_active=True, is_terminal=False)
    rows["row"] = "📱 1 tokens written · 1.0 tok/s"
    assert screen.device_row() == ""


# ---- the SDLXLIFF reviewer: a compact header, a full-screen Notepad, the source tooltip off the text -------------


class _Prefs:
    def __init__(self) -> None:
        self.data: dict = {}

    def resolve_file_ref(self, fid, *, touch=True):
        return None

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value


def _reviewer(page=None):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools import sdlxliff as sx

    ctx = types.SimpleNamespace(prefs=_Prefs(), tablet=False, page=page, spawn=lambda c: c.close())
    screen = sx.SdlxliffScreen(parse_route("/tools/sdlxliff"), ctx)
    screen.build_body()
    screen._push = lambda *c: None
    return screen


def test_the_reviewer_actions_fit_one_row():
    pytest.importorskip("flet")
    import flet as ft

    screen = _reviewer()
    rows = [c for c in screen.header_column.controls if isinstance(c, ft.Row)]
    actions = next(r for r in rows if r.key == "sdl-actions")
    assert not actions.wrap and len(actions.controls) == 4
    assert screen.mt_button.expand and screen.flag_button.expand and screen.options_button in actions.controls
    assert screen.summary.max_lines == 1 and screen.status_text.visible is False
    sheet = screen.open_options_sheet()  # Provider and Threshold behind one button
    keys = [getattr(item, "key", None) for item in sheet.items]
    assert keys == ["sdl-provider", "sdl-threshold"]


def test_the_notepad_layout_is_full_screen_with_the_controls_behind_a_button():
    pytest.importorskip("flet")

    screen = _reviewer()
    immersive: list = []
    screen.immersive_handler = immersive.append
    assert screen.notepad_bar.visible is False and screen.header_column.visible is not False
    screen.notepad = True
    screen._apply_notepad_layout()
    assert screen.notepad_bar.visible and screen.header_column.visible is False and immersive == [True]
    assert screen.list_view.controls[0] is screen.notepad_bar
    screen.toggle_controls()  # the bar's Controls button
    assert screen.header_column.visible is True
    screen._keyboard_up = True  # typing: the controls step aside again
    screen._apply_notepad_layout()
    assert screen.header_column.visible is False and screen.notepad_bar.visible
    screen._keyboard_up = False
    screen.toggle_controls()
    assert screen.header_column.visible is False
    screen.notepad = False
    screen._apply_notepad_layout()
    assert screen.notepad_bar.visible is False and screen.header_column.visible is True and immersive[-1] is False


def test_the_shell_hides_the_app_bar_and_job_strip_while_immersive():
    pytest.importorskip("flet")
    import flet as ft

    from glossarion_mobile.ui.screens.base import Screen
    from glossarion_mobile.ui.shell.app_shell import AppShell, StackEntry

    screen = Screen(None)
    view = ft.View(route="/x", appbar=ft.AppBar(title=ft.Text("x")))
    entry = StackEntry(match=None, screen=screen, view=view, footer=ft.Container())
    shell = AppShell.__new__(AppShell)
    shell._immersive(entry, True)
    assert view.appbar is None and entry.footer.visible is False
    shell._immersive(entry, True)  # again: nothing to do
    shell._immersive(entry, False)
    assert isinstance(view.appbar, ft.AppBar) and entry.footer.visible is True
    Screen(None).set_immersive(True)  # no shell: nothing happens


# ---- the model menu: Done on a route prefix, the typed auto-poll, "Add key" on keyless routes ---------------------


def _model_sheet(models, *, catalog=None, spawn=None, selected=None):
    from glossarion_mobile.ui.sheets.model_sheet import ModelSheet, SheetEnv

    env = SheetEnv(catalog=catalog, spawn=spawn)
    sheet = ModelSheet(current_model="gpt-6", models=list(models), env=env,
                       on_select=lambda *a: (selected if selected is not None else []).append(a))
    sheet._push_all = lambda: None
    return sheet


def test_done_on_a_route_prefix_keeps_the_menu_open():
    pytest.importorskip("flet")

    selected: list = []
    sheet = _model_sheet(["gpt-6", "authnd/google/gemma-4-31b-it", "authnd/z-ai/glm-5.3-flash"], selected=selected)
    for prefix in ("authnd/", "authnd"):  # two models start with it: Done only closes the keyboard
        sheet.query = prefix
        sheet.submit_query()
    assert selected == []
    sheet.query = "authnd/z-ai/glm"
    sheet.submit_query()
    assert [s[:2] for s in selected] == [("model", "authnd/z-ai/glm-5.3-flash")]  # the one match left
    sheet.query = "chutes/some/model"  # a whole id that is not listed is still used as typed
    sheet.submit_query()
    assert selected[-1][:2] == ("model", "chutes/some/model")


def test_a_typed_route_is_auto_polled_once_typing_settles(monkeypatch):
    pytest.importorskip("flet")
    from glossarion_mobile.ui.sheets import model_sheet as ms

    monkeypatch.setattr(ms, "TYPED_POLL_DELAY", 0.02)
    polled: list = []

    class Catalog:
        snapshot = None

        async def maybe_auto_poll(self, model):
            polled.append(model)

    async def scenario():
        tasks: list = []
        sheet = _model_sheet(["gpt-6"], catalog=Catalog(),
                             spawn=lambda coro: tasks.append(asyncio.ensure_future(coro)) or tasks[-1])
        sheet.tab = "model"
        for text in ("a", "au", "authnd/"):  # typing: only the settled text is polled
            sheet._schedule_typed_poll(text)
        await asyncio.gather(*tasks)
        assert polled == ["authnd/"]
        sheet._schedule_typed_poll("authgpt/")
        sheet.closed = True  # the menu closed meanwhile: no poll
        await asyncio.gather(*tasks)
        assert polled == ["authnd/"]
        assert sheet._schedule_typed_poll("   ") is None

    asyncio.run(scenario())


def test_models_a_poll_adds_are_classified_for_their_key():
    pytest.importorskip("flet")
    import unified_api_client  # noqa: F401  (loaded in the app before the menu lists models)

    sheet = _model_sheet(["gpt-6"])
    sheet._keyless = {"some/keyless-model"}  # classified when the menu opened
    assert sheet._needs_key("authnd/deepseek-ai/deepseek-v4.1-flash") is False  # added by a poll later
    assert sheet._needs_key("gpt-6") is True
    assert sheet._needs_key("some/keyless-model") is False


def test_the_source_tooltip_docks_away_from_the_text_on_a_touch_screen():
    from glossarion_mobile.ui.tools import sdlxliff_notepad as sn

    html = sn.page_html("<html><body><p>x</p></body></html>", "/*editor*/", "n0")
    editor, touch, capture = html.index("/*editor*/"), html.index("pointer: coarse"), html.index("__glSdlCapture")
    assert editor < touch < capture  # after the editing script's listeners
    assert "left: 8px !important; right: 8px !important" in sn.TOUCH_TOOLTIP_SCRIPT
    assert "tip.style.top = lower ? '8px' : 'auto'" in sn.TOUCH_TOOLTIP_SCRIPT

