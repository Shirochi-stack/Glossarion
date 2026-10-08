"""Owner device report #6 (D): "a toggle to always auto-accept the generated glossary", and contract C1
(one glossary-question predicate, the ``question_resolved`` job event, the per-kind job notification text).

Run from src/mobile with the mobile venv:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_glossary_auto_accept.py

* The desktop has no such setting: every Direct Text glossary gate blocks on the Edit / Yes / No question
  (``translation_pipeline._await_direct_text_glossary_approval`` -> ``_ui_request`` -> ``host.ask``). On mobile
  ``host.ask`` is ``JobService._job_ask``; a chat send whose "Always accept generated glossaries" is on carries
  ``params["auto_accept_glossary"]`` and the job answers the gate Yes itself, with the desktop's accepted log
  line, delivering nothing (no card, no notification, works with the screen off).
* The setting is mobile-only: the All-chats value lives in Prefs (``chat_auto_accept_glossary``), the
  per-chat override in the chat's sidecar meta; config.json gets no new key (UI_SPEC Appendix B).

The job tests reuse ``test_jobs``' FakeBackend / ``make_service`` (no shared backend); the settings and the
real-gate tests need the shared backend (the project venv), the sheet test Flet too. Every test runs with
HOME / USERPROFILE / APPDATA / GLOSSARION_LIBRARY_DIR / OUTPUT_DIRECTORY pointed at tmp_path.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import threading
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
APP_PACKAGE = APP_DIR / "glossarion_mobile"
for entry in (str(APP_DIR), str(SRC_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from glossarion_mobile.services import jobs as jobs_module  # noqa: E402
from glossarion_mobile.services.jobs import (  # noqa: E402
    AUTO_ACCEPT_GLOSSARY_PARAM,
    GLOSSARY_ACCEPTED_LINE,
    GLOSSARY_AUTO_ACCEPTED_LINE,
    GLOSSARY_QUESTION_KINDS,
    JobSnapshot,
    JobSpec,
    JobState,
    Progress,
    is_glossary_question,
    notification_text,
    progress_line,
)

_TJ_SPEC = importlib.util.spec_from_file_location("_glossarion_tj_helpers_autoaccept", Path(__file__).with_name("test_jobs.py"))
_TJ = importlib.util.module_from_spec(_TJ_SPEC)
_TJ_SPEC.loader.exec_module(_TJ)
make_service = _TJ.make_service
epub = _TJ.epub
wait_for = _TJ.wait_for
TIMEOUT = _TJ.TIMEOUT


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _backend_error() -> str:
    try:
        import direct_text_stream  # noqa: F401  (imports direct_text_store and the pipeline)
        import translation_pipeline  # noqa: F401
    except ImportError as exc:
        return f"the shared backend is not importable here ({exc}); use the project venv (uv sync)"
    return ""


_BACKEND_ERROR = _backend_error()
needs_backend = pytest.mark.skipif(bool(_BACKEND_ERROR), reason=_BACKEND_ERROR or "backend importable")
needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed")

GATE = "direct_text_glossary_approval"


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """No probe here may reach the owner's real Library, output root, home or HTTP log."""
    home = tmp_path / "home"
    home.mkdir()
    for name in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA"):
        monkeypatch.setenv(name, str(home))
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(tmp_path / "Library"))
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(tmp_path / "Output"))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")


def _log_lines(service, job_id) -> list:
    buffer = service.log_buffer(job_id)
    lines = [line.text for line in buffer.snapshot()] if buffer is not None else []
    return lines or service.read_log_tail(job_id)


def _pipeline_gate(host):
    """The shared Direct Text gate itself: ``translation_pipeline``'s ``_await_direct_text_glossary_approval``
    and ``_ui_request`` (HeadlessOwner's GUI-free question path) on a minimal owner around a job's host."""
    import translation_pipeline as tp

    class GateOwner(tp.PipelineHooksMixin):
        _await_direct_text_glossary_approval = tp.TranslationPipelineMixin._await_direct_text_glossary_approval

        def __init__(self, job_host):
            self.host = job_host
            self.stop_requested = False

        def append_log(self, message):
            self.host.log(message)

    return GateOwner(host)


# ==========================================================================
# JobService._job_ask: the single funnel of every glossary gate
# ==========================================================================


def test_auto_accept_glossary_param_answers_without_asking(tmp_path):
    service, backend = make_service(tmp_path)
    asked = []

    def listener(snap, question):
        asked.append(question["kind"])
        threading.Timer(0.02, service.answer, (question["id"], "asked")).start()

    service.on_question(listener)
    answers = {}

    def asking(owner, request):
        answers[owner.host.job_id] = [
            owner.host.ask(GATE, path="g.csv"),
            owner.host.ask("glossary_approval", path="g.csv", default=False),
            owner.host.ask("async_batch_question", level="question", title="Start Async Processing", text="?",
                           default="no"),
        ]
        return None

    backend.behavior = asking
    auto = service.submit(JobSpec("translate", "Auto", (epub(tmp_path),), params={AUTO_ACCEPT_GLOSSARY_PARAM: True}))
    assert service.wait_idle(TIMEOUT)
    # both glossary gates answered Yes on the job thread; only the async-batch question reached the UI
    assert answers[auto] == [True, True, "asked"]
    assert asked == ["async_batch_question"]
    assert _log_lines(service, auto).count(GLOSSARY_AUTO_ACCEPTED_LINE) == 2
    assert GLOSSARY_AUTO_ACCEPTED_LINE == "✅ Direct Text: generated glossary accepted (Always accept is on)"

    asked.clear()
    plain = service.submit(JobSpec("translate", "Plain", (epub(tmp_path),), params={AUTO_ACCEPT_GLOSSARY_PARAM: False}))
    legacy = service.submit(JobSpec("translate", "Legacy", (epub(tmp_path),)))  # a job checkpointed before the flag
    assert service.wait_idle(TIMEOUT)
    assert answers[plain] == ["asked", "asked", "asked"] and answers[legacy] == ["asked", "asked", "asked"]
    assert asked == [GATE, "glossary_approval", "async_batch_question"] * 2
    assert not any(GLOSSARY_ACCEPTED_LINE in line for line in _log_lines(service, plain))
    service.close()


def test_auto_accept_gives_up_like_the_card_once_the_job_is_stopping(tmp_path):
    """A Stop that lands before the gate: the job never logs "accepted" and gets the default (declined),
    as when the approval card is waiting and the user stops the run."""
    service, backend = make_service(tmp_path)
    seen = []
    service.on_question(lambda snap, question: seen.append(question["kind"]))
    answers = []

    def stopping(owner, request):
        owner.host._job.stop_event.set()  # the job's latch, as JobService.request_stop sets it first
        answers.append(owner.host.ask(GATE, path="g.csv", default=False))
        return None

    backend.behavior = stopping
    job_id = service.submit(JobSpec("translate", "Book", (epub(tmp_path),), params={AUTO_ACCEPT_GLOSSARY_PARAM: True}))
    assert service.wait_idle(TIMEOUT)
    assert answers == [False] and seen == []
    assert not any(GLOSSARY_ACCEPTED_LINE in line for line in _log_lines(service, job_id))
    service.close()


@needs_backend
def test_the_shared_direct_text_gate_is_answered_by_the_flag(tmp_path):
    """The real pipeline gate (``_await_direct_text_glossary_approval`` -> ``_ui_request`` -> ``host.ask``) of a
    job: accepted at once with the flag, the card's answer without it."""
    service, backend = make_service(tmp_path)
    delivered = []

    def listener(snap, question):
        delivered.append((question["kind"], dict(question["data"])))
        threading.Timer(0.02, service.answer, (question["id"], False)).start()

    service.on_question(listener)
    results = {}

    def gate(owner, request):
        results[owner.host.job_id] = _pipeline_gate(owner.host)._await_direct_text_glossary_approval(
            str(tmp_path / "glossary.csv"))
        return None

    backend.behavior = gate
    auto = service.submit(JobSpec("translate", "Chat", (epub(tmp_path),), params={AUTO_ACCEPT_GLOSSARY_PARAM: True}))
    assert wait_for(lambda: auto in results) and service.wait_idle(TIMEOUT)
    asked = service.submit(JobSpec("translate", "Card", (epub(tmp_path),)))
    assert wait_for(lambda: asked in results) and service.wait_idle(TIMEOUT)
    assert results[auto] is True and results[asked] is False
    assert delivered == [(GATE, {"path": os.path.abspath(str(tmp_path / "glossary.csv"))})]
    assert GLOSSARY_AUTO_ACCEPTED_LINE in _log_lines(service, auto)
    service.close()


# ==========================================================================
# question_resolved (contract C1): answered or given up, also while the UI pump is parked
# ==========================================================================


def test_answered_and_stopped_questions_emit_question_resolved(tmp_path):
    service, backend = make_service(tmp_path)
    events, questions = [], []
    service.on_event(lambda job_id, kind, data: events.append((job_id, kind, dict(data))))

    def listener(snap, question):
        questions.append(question)
        if question["kind"] == GATE:
            threading.Timer(0.02, service.answer, (question["id"], True)).start()

    service.on_question(listener)

    def asking(owner, request):
        owner.host.ask(GATE, path="g.csv")
        owner.host.ask("async_batch_question", title="Start", text="?", default="no")  # never answered: Stop
        return None

    backend.behavior = asking
    job_id = service.submit(JobSpec("translate", "Book", (epub(tmp_path),)))
    assert wait_for(lambda: len(questions) == 2)
    service.request_stop(job_id, force=True)
    assert service.wait_idle(TIMEOUT)
    assert [e for e in events if e[1] == "question_resolved"] == [
        (job_id, "question_resolved", {"id": questions[0]["id"], "kind": GATE}),
        (job_id, "question_resolved", {"id": questions[1]["id"], "kind": "async_batch_question"}),
    ]

    # nothing was delivered, so nothing is resolved: an auto-accepted gate, an ask without listeners
    events.clear()
    backend.behavior = lambda owner, request: owner.host.ask(GATE, path="g.csv") and None
    service.submit(JobSpec("translate", "Auto", (epub(tmp_path),), params={AUTO_ACCEPT_GLOSSARY_PARAM: True}))
    assert service.wait_idle(TIMEOUT)
    quiet, _backend = make_service(tmp_path / "quiet")
    quiet.on_event(lambda job_id, kind, data: events.append((job_id, kind, dict(data))))
    _backend.behavior = lambda owner, request: owner.host.ask(GATE, path="g.csv") and None
    quiet.submit(JobSpec("translate", "Nobody", (epub(tmp_path),)))
    assert quiet.wait_idle(TIMEOUT)
    assert [e for e in events if e[1] == "question_resolved"] == []
    service.close()
    quiet.close()


class _HiddenPage:
    """A Flet page in the background (``app_visible`` False): the UiDispatcher pump parks on
    ``wait_until_visible()`` (test_ui_foundations' _FakePage pattern)."""

    def __init__(self):
        self.app_visible = False
        self.visible = asyncio.Event()

    def update(self, *controls):
        pass

    async def wait_until_visible(self):
        await self.visible.wait()


def test_question_resolved_arrives_while_the_dispatcher_is_parked(tmp_path):
    """The job notification is cancelled from this event (notify-delivery), so it must not wait for the
    app to come back to the foreground."""
    from glossarion_mobile.services.dispatcher import UiDispatcher
    from glossarion_mobile.state.store import LoopGuard

    async def scenario():
        page = _HiddenPage()
        dispatcher = UiDispatcher(page, interval=0.01, guard=LoopGuard()).bind()
        dispatcher.start()
        service, backend = make_service(tmp_path, dispatcher=dispatcher)
        loop_thread = threading.get_ident()
        questions, events = [], []

        def listener(snap, question):
            questions.append(question)
            service.answer(question["id"], True)  # e.g. the notification's Accept

        service.on_question(listener)
        service.on_event(lambda job_id, kind, data: events.append((threading.get_ident(), kind, dict(data))))
        backend.behavior = lambda owner, request: owner.host.ask(GATE, path="g.csv") and None
        service.submit(JobSpec("translate", "Book", (epub(tmp_path),)))
        for _ in range(500):
            if events:
                break
            await asyncio.sleep(0.01)
        assert events == [(loop_thread, "question_resolved", {"id": questions[0]["id"], "kind": GATE})]
        assert page.app_visible is False and dispatcher.ticks == 0  # the pump never ran a tick: parked
        page.app_visible = True
        page.visible.set()
        service.close()
        await dispatcher.stop()

    asyncio.run(scenario())


# ==========================================================================
# Per-kind texts and the one predicate
# ==========================================================================


def _snap(question=None, **kwargs):
    base = dict(id="abc123abc123", spec=JobSpec("translate", "Book.epub", ("a",), origin={"type": "chat", "cid": "7"}),
                state=JobState.RUNNING, created=0.0, started=100.0, progress=Progress(total=80, completed=12),
                question=question)
    base.update(kwargs)
    return JobSnapshot(**base)


def test_notification_text_and_strip_subtitle_follow_the_question_kind():
    glossary = _snap({"id": "q1", "kind": GATE, "data": {"path": "g.csv"}})
    library = _snap({"id": "q2", "kind": "glossary_approval", "data": {"path": "g.csv"}})
    batch = _snap({"id": "q3", "kind": "async_batch_question", "data": {"title": "Start Async Processing"}})
    other = _snap({"id": "q4", "kind": "something_else", "data": {}})
    for snap in (glossary, library):
        assert notification_text(snap).endswith(": waiting for your glossary decision")
        assert progress_line(snap) == "Waiting for your glossary decision"
    for snap in (batch, other):
        assert notification_text(snap).endswith(": waiting for your answer")
        assert "glossary" not in notification_text(snap)
    assert progress_line(batch) == "Waiting for your answer: Start Async Processing"
    assert progress_line(other) == "Waiting for your answer"
    assert notification_text(_snap()) == "Translating Book.epub: 12/80 chapters"


def _old_run_controller_predicate(kind):
    """run_controller.is_glossary_question as it was at cddd73a4 (ui/chat/run_controller.py:98-106)."""
    value = str(kind or "").lower()
    return value in ("glossary_approval", "direct_text_glossary_approval") or ("glossary" in value and "approv" in value)


def test_is_glossary_question_moved_verbatim_into_services_jobs():
    kinds = [GATE, "glossary_approval", "Direct_Text_Glossary_Approval", "series_glossary_approval_v2",
             "glossary_approve", "async_batch_question", "glossary", "approval", "", None, 3]
    assert GLOSSARY_QUESTION_KINDS == ("glossary_approval", "direct_text_glossary_approval")
    assert "is_glossary_question" in jobs_module.__all__ and "GLOSSARY_QUESTION_KINDS" in jobs_module.__all__
    for kind in kinds:
        assert is_glossary_question(kind) == _old_run_controller_predicate(kind), kind
    from glossarion_mobile.ui.chat import run_controller

    for kind in kinds:  # the chat still answers exactly the same questions
        assert run_controller.is_glossary_question(kind) == is_glossary_question(kind), kind
    # the Library review gate asks one of the kinds
    from glossarion_mobile.job_kinds.translate import GLOSSARY_REVIEW_QUESTION

    assert GLOSSARY_REVIEW_QUESTION in GLOSSARY_QUESTION_KINDS


def test_desktop_still_logs_the_accepted_line():
    """The auto-accept logs the desktop's own line (``translator_gui._resolve_glossary_approval``); the
    desktop file is never edited, so this pins that the literal still exists there."""
    gui = SRC_DIR / "translator_gui.py"
    if not gui.is_file():
        pytest.skip("src/translator_gui.py not present")
    source = gui.read_text(encoding="utf-8-sig")
    assert f'"{GLOSSARY_ACCEPTED_LINE}"' in source
    assert "auto_accept" not in source  # the desktop has no such setting: it always asks


def test_the_library_review_gate_never_sets_the_flag():
    """Library › Translate… "Review glossary before translating" is an explicit per-run review: only the
    chat send (run_request.job_params) carries the flag."""
    offenders = []
    for path in [*sorted((APP_PACKAGE / "ui" / "library").rglob("*.py")), APP_PACKAGE / "job_kinds" / "translate.py",
                 APP_PACKAGE / "services" / "library.py"]:
        if path.is_file() and AUTO_ACCEPT_GLOSSARY_PARAM in path.read_text(encoding="utf-8"):
            offenders.append(path.name)
    assert offenders == []


# ==========================================================================
# Settings: Prefs (All chats) and the chat's sidecar meta (This chat); run params captured at Send
# ==========================================================================


def _getter(data):
    return lambda key, default=None: data.get(key, default)


@needs_backend
def test_direct_text_settings_auto_accept_from_prefs_and_sidecar(tmp_path):
    from glossarion_mobile.ui.chat import direct_text_rules as rules
    from glossarion_mobile.ui.chat.direct_text_rules import (
        AUTO_ACCEPT_GLOSSARY_PREF,
        SIDECAR_META_FIELDS,
        DirectTextSettings,
        merge_meta_overrides,
    )
    from glossarion_mobile.ui.chat.run_request import DirectTextRun, job_params
    from glossarion_mobile.ui.sheets import chat_settings

    assert AUTO_ACCEPT_GLOSSARY_PREF == "chat_auto_accept_glossary"
    assert SIDECAR_META_FIELDS == ("skip_plan", "auto_accept_glossary")
    assert DirectTextSettings().auto_accept_glossary is False
    # the global value comes from Prefs only (default off); a config.json key of that name means nothing
    config = _getter({AUTO_ACCEPT_GLOSSARY_PREF: True, "auto_accept_glossary": True})
    assert DirectTextSettings.from_config(config).auto_accept_glossary is False
    assert DirectTextSettings.from_config(config, prefs_get=_getter({})).auto_accept_glossary is False
    on = DirectTextSettings.from_config(_getter({}), prefs_get=_getter({AUTO_ACCEPT_GLOSSARY_PREF: True}))
    assert on.auto_accept_glossary is True

    def broken(key, default=None):
        raise RuntimeError("prefs not loaded")

    assert DirectTextSettings.from_config(_getter({}), prefs_get=broken).auto_accept_glossary is False

    # the chat's sidecar meta layers over its overrides (unset = inherit All chats)
    merged = merge_meta_overrides({"model": "gpt-5"}, {"skip_plan": 1, "auto_accept_glossary": False, "text_scale": 1.2,
                                                        "series_id": "s1"})
    assert merged == {"model": "gpt-5", "skip_plan": True, "auto_accept_glossary": False}
    assert merge_meta_overrides({"model": "gpt-5"}, {"auto_accept_glossary": None}) == {"model": "gpt-5"}
    assert merge_meta_overrides(None, None) == {}
    assert on.with_overrides(merge_meta_overrides({}, {"auto_accept_glossary": False})).auto_accept_glossary is False
    off = DirectTextSettings()
    assert off.with_overrides(merge_meta_overrides({}, {"auto_accept_glossary": True})).auto_accept_glossary is True
    assert on.with_overrides(merge_meta_overrides({"model": "x"}, {})).auto_accept_glossary is True
    assert off.with_overrides({"skip_plan": True}).skip_plan is True  # skip_plan unchanged

    # never a config.json key (UI_SPEC Appendix B)
    assert not any("accept" in key for key in on.config_updates())
    assert "auto_accept_glossary" not in chat_settings.CHAT_SETTING_KEYS
    assert chat_settings.global_updates_for("auto_accept_glossary", True) == {}
    assert chat_settings.global_updates_for("skip_plan", True) == {}
    assert {"AUTO_ACCEPT_GLOSSARY_PREF", "SIDECAR_META_FIELDS", "merge_meta_overrides"} <= set(rules.__all__)

    # captured at Send into the job's params (JSON, checkpointable)
    run = DirectTextRun(temp_root=str(tmp_path / "run"), source_path=str(tmp_path / "run" / "in.txt"),
                        source_extension=".txt", is_attachment=False, expected_output="")
    assert job_params(chat_id=3, user_index=1, run=run, settings=on)[AUTO_ACCEPT_GLOSSARY_PARAM] is True
    assert job_params(chat_id=3, user_index=1, run=run, settings=off)[AUTO_ACCEPT_GLOSSARY_PARAM] is False


class _Prefs:
    def __init__(self, data=None):
        self.data = dict(data or {})
        self.writes = []

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.writes.append((key, value))
        self.data[key] = value


class _Chats:
    def __init__(self):
        self.values, self.metas = {}, {}

    def overrides(self, cid):
        return dict(self.values)

    def meta(self, cid):
        return dict(self.metas)

    def set_override(self, cid, name, value):
        if value is None:
            self.values.pop(name, None)
        else:
            self.values[name] = value

    def set_meta(self, cid, name, value):
        if value is None:
            self.metas.pop(name, None)
        else:
            self.metas[name] = value

    def reset_overrides(self, cid):
        self.values.clear()


class _Config(dict):
    def set_many(self, updates):
        self.update(updates)


@needs_flet
@needs_backend
def test_owner_toggle_reaches_the_gate_without_a_card(tmp_path):
    """The owner's complaint end to end: Chat settings › All chats › "Always accept generated glossaries" on
    -> the next send's params -> the job answers the real Direct Text gate itself; nothing is asked."""
    from glossarion_mobile.ui.chat.direct_text_rules import AUTO_ACCEPT_GLOSSARY_PREF
    from glossarion_mobile.ui.chat.run_request import DirectTextRun, job_params
    from glossarion_mobile.ui.sheets.chat_settings import ChatSettingsSheet

    prefs, config, chats = _Prefs(), _Config(), _Chats()
    sheet = ChatSettingsSheet(cid="7", config=config, chats=chats, prefs=prefs, scope="global")
    sheet.set_value("auto_accept_glossary", True)
    assert prefs.data == {AUTO_ACCEPT_GLOSSARY_PREF: True} and dict(config) == {} and chats.metas == {}
    settings = ChatSettingsSheet(cid="7", config=config, chats=chats, prefs=prefs).effective()
    assert settings.auto_accept_glossary is True

    run = DirectTextRun(temp_root=str(tmp_path / "run"), source_path=epub(tmp_path), source_extension=".epub",
                        is_attachment=True, expected_output="")
    params = job_params(chat_id=7, user_index=1, run=run, settings=settings)
    service, backend = make_service(tmp_path)
    asked = []
    service.on_question(lambda snap, question: asked.append(question))
    results = []
    backend.behavior = lambda owner, request: results.append(
        _pipeline_gate(owner.host)._await_direct_text_glossary_approval(str(tmp_path / "glossary.csv")))
    job_id = service.submit(JobSpec("translate", "Book.epub", (run.source_path,), params=params))
    assert service.wait_idle(TIMEOUT)
    assert results == [True] and asked == []
    assert GLOSSARY_AUTO_ACCEPTED_LINE in _log_lines(service, job_id)
    service.close()
