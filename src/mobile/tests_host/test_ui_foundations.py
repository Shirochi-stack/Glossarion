"""Host tests for the U1 UI foundations (state, dispatcher, theme, router, shell, chat).

Run from src/mobile with the mobile venv (Flet installed):
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore tests_host/test_ui_foundations.py

The pure parts (router whitelist, Send/Stop machine, dispatcher, tokens,
responsive classes, output modes) run on any Python 3.10+ without Flet. The
Flet parts build controls into an in-memory fake Flet session (messages are
msgpack-encoded like the real transport, see ``test_bootstrap._fake_session``)
and are skipped when Flet is missing. Nothing opens a window.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import importlib.util
import sys
import threading
import time
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
EXTENSION_SRC = MOBILE_DIR / "extensions" / "flet_glossarion_native" / "src"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from glossarion_mobile import runtime_bootstrap as rb  # noqa: E402
from glossarion_mobile.services import secure_keys  # noqa: E402
from glossarion_mobile.services.dispatcher import (  # noqa: E402
    LogBuffer,
    LogBufferHandler,
    UiDispatcher,
    classify_log_line,
)
from glossarion_mobile.state.app_state import AppState, ChatContext, JobStripModel  # noqa: E402
from glossarion_mobile.state.chat_index import ChatSummary, InMemoryChatIndex, group_recents  # noqa: E402
from glossarion_mobile.state.store import Computed, LoopGuard, Signal, WrongThreadError  # noqa: E402
from glossarion_mobile.ui import responsive, tokens  # noqa: E402
from glossarion_mobile.ui.chat import output_modes  # noqa: E402
from glossarion_mobile.ui.chat.send_state import (  # noqa: E402
    BLOCK_ENGINE_NOT_READY,
    BLOCK_MANUAL_GLOSSARY,
    DEFAULT_MODEL,
    SendAction,
    SendInputs,
    SendState,
    SendStopMachine,
    chatgpt_sign_in_reason,
    derive_state,
    requires_chatgpt_sign_in,
    status_caption,
    visual_for,
)
from glossarion_mobile.ui.router import (  # noqa: E402
    ROUTES,
    ROUTES_BY_NAME,
    RouteError,
    Router,
    build_route,
    launch_links,
    parse_route,
)

HEX = "ab12cd34ef56"

# The fake Flet session and fixtures are shared with test_bootstrap.py (one copy).
_TB_SPEC = importlib.util.spec_from_file_location("_glossarion_tb_helpers", Path(__file__).with_name("test_bootstrap.py"))
_TB = importlib.util.module_from_spec(_TB_SPEC)
_TB_SPEC.loader.exec_module(_TB)
storage = _TB.storage
app_env = _TB.app_env
_fake_session = _TB._fake_session
_route = _TB._route


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed")


# ==========================================================================
# Router whitelist (UI_SPEC §1.4, §1.5)
# ==========================================================================


@pytest.mark.parametrize(
    "raw, name, params, query",
    [
        ("/", "home", {}, {}),
        ("/chat/12", "chat", {"cid": "12"}, {}),
        ("/chat/s0123456789abcdef0123456789abcdef", "chat", {"cid": "s0123456789abcdef0123456789abcdef"}, {}),
        ("/chat/s12345678-1234-1234-1234-123456789abc/settings", "chat.settings", {"cid": "s12345678-1234-1234-1234-123456789abc"}, {}),
        ("/chat/3/attachments", "chat.attachments", {"cid": "3"}, {}),
        ("/chat/3/compose", "chat.compose", {"cid": "3"}, {}),
        (f"/chat/3/m/{HEX}", "chat.message", {"cid": "3", "mid": HEX}, {}),
        (f"/chat/3/m/{HEX}/edit", "chat.message.edit", {"cid": "3", "mid": HEX}, {}),
        ("/series/my-series_1", "series", {"sid": "my-series_1"}, {}),
        ("/library?shelf=completed", "library", {}, {"shelf": "completed"}),
        ("/library/scan-raw", "library.scan_raw", {}, {}),
        (f"/library/book/{HEX}?tab=chapters&filter=failed", "library.book", {"bid": HEX}, {"tab": "chapters", "filter": "failed"}),
        (f"/library/book/{HEX}/metadata", "library.book.metadata", {"bid": HEX}, {}),
        (f"/reader/{HEX}?ch=12&mode=bilingual", "reader", {"bid": HEX}, {"ch": "12", "mode": "bilingual"}),
        ("/jobs", "jobs", {}, {}),
        ("/jobs/3f2a9c", "jobs.detail", {"jid": "3f2a9c"}, {}),
        ("/glossary", "glossary", {}, {}),
        (f"/glossary/{HEX}?tab=minimal", "glossary.detail", {"gid": HEX}, {"tab": "minimal"}),
        (f"/glossary/{HEX}/entry/7", "glossary.entry", {"gid": HEX, "n": "7"}, {}),
        ("/glossary/unified", "glossary.unified", {}, {}),
        ("/glossary/parallel-pair", "glossary.parallel_pair", {}, {}),
        ("/tools", "tools", {}, {}),
        (f"/tools/progress?out={HEX}", "tools.progress", {}, {"out": HEX}),
        (f"/tools/progress/glossary?out={HEX}", "tools.progress.glossary", {}, {"out": HEX}),
        ("/tools/qa", "tools.qa", {}, {}),
        ("/tools/convert", "tools.convert", {}, {}),
        ("/tools/headers", "tools.headers", {}, {}),
        ("/tools/async", "tools.async", {}, {}),
        ("/tools/review", "tools.review", {}, {}),
        ("/tools/sdlxliff", "tools.sdlxliff", {}, {}),
        ("/tools/manga?tab=editor", "tools.manga", {}, {"tab": "editor"}),
        ("/tools/rpgmaker", "tools.rpgmaker", {}, {}),
        (f"/tools/qa/report/{HEX}", "tools.qa.report", {"rid": HEX}, {}),
        ("/tools/files/output", "tools.files", {"root": "output"}, {}),
        (f"/tools/files/{HEX}", "tools.files", {"root": HEX}, {}),
        (f"/tools/files/inbox/{HEX}", "tools.files.folder", {"root": "inbox", "fid": HEX}, {}),
        (f"/tools/text/{HEX}?hit=4", "tools.text", {"fid": HEX}, {"hit": "4"}),
        ("/settings", "settings", {}, {}),
        ("/settings/s/response_handling", "settings.section", {"section": "response_handling"}, {}),
        ("/settings/models", "settings.models", {}, {}),
        ("/settings/keys", "settings.keys", {}, {}),
        ("/settings/keys/glossary_refinement", "settings.keys.pool", {"pool": "glossary_refinement"}, {}),
        ("/settings/accounts", "settings.accounts", {}, {}),
        ("/settings/profiles/korean_bs", "settings.profiles.detail", {"pid": "korean_bs"}, {}),
        ("/settings/logs", "settings.logs", {}, {}),
        ("/settings/danger", "settings.danger", {}, {}),
        ("/welcome", "welcome", {}, {}),
        ("/oauth/return?p=authgpt", "oauth.return", {}, {"p": "authgpt"}),
        ("/__selftest__?suite=smoke", "selftest", {}, {"suite": "smoke"}),
    ],
)
def test_router_accepts_spec_routes(raw, name, params, query):
    match = parse_route(raw)
    assert match is not None, raw
    assert (match.name, match.params, match.query) == (name, params, query)
    # the same route as a full deep link
    deep = parse_route("glossarion://app" + raw)
    assert deep is not None and deep.deep_link
    assert (deep.name, deep.params, deep.query) == (name, params, query)


@pytest.mark.parametrize(
    "raw, path, query",
    [  # the U0 CI/OAuth contract keeps working
        (None, "/", {}),
        ("", "/", {}),
        ("/__selftest__/", "/__selftest__", {}),
        ("glossarion://app/__selftest__?suite=smoke", "/__selftest__", {"suite": "smoke"}),
        ("GLOSSARION://APP/oauth/return?p=spike&nonce=ab12", "/oauth/return", {"p": "spike", "nonce": "ab12"}),
    ],
)
def test_router_keeps_u0_contract(raw, path, query):
    match = parse_route(raw)
    assert match is not None and match.path == path and match.query == query


@pytest.mark.parametrize(
    "raw",
    [
        "content://com.android.providers.downloads.documents/document/raw%3A%2Fstorage%2Fx.epub",
        "file:///storage/emulated/0/Download/book.epub",
        "FILE:///private/var/mobile/x.pdf",
        "intent://scan/#Intent;scheme=zxing;end",
        "javascript:alert(1)",
        "data:text/html,hi",
        "https://example.com/library",
        "http://127.0.0.1:1455/auth/callback?code=x",
        "otherapp://app/library",
        "glossarion://evil/library",
        "glossarion://app:99/library",
        "glossarion://user@app/library",
        "/document/raw%3A%2Fstorage%2Femulated%2F0%2Fx.epub",
        "/storage/emulated/0/Download/book.epub",
        "/tools/text/book.epub",  # file names never appear in routes
        "/tools/text/ab12cd34ef5",  # 11 hex chars
        "/tools/text/ab12cd34ef567",  # 13 hex chars
        "/reader/My Book",
        "/reader/zz12cd34ef56",
        "/library/book/../../etc",
        f"/library/book/{HEX}/../metadata",
        "/chat/abc",  # not an int and not s<hex>
        "/chat/3/m/not-a-hash",
        "/settings/keys/nope",
        "/settings/secret",
        "/settings/s/Bad-Section",
        "/tools/unknown",
        "/jobs/a b",
        "/oauth%2Freturn",
        "//evil.example/oauth/return",
        "relative/path",
        "/a\\b",
        "/library\x00",
        "/" + "x" * 3000,
    ],
)
def test_router_rejects_everything_else(raw):
    assert parse_route(raw) is None


def test_router_drops_text_from_query_and_fragment():
    match = parse_route("/library?shelf=../../etc&q=my%20book&shelf=completed")
    assert match is not None and match.query == {"shelf": "completed"}  # invalid value and unknown key dropped
    assert parse_route("/library?shelf=../../etc").query == {}
    match = parse_route(f"/reader/{HEX}?ch=twelve&mode=bilingual&path=/sdcard/x.epub")
    assert match.query == {"mode": "bilingual"}
    match = parse_route("/settings/s/response_handling#retry_timeout")
    assert match.fragment == "retry_timeout"
    assert parse_route("/settings/logs#retry_timeout").fragment is None  # fragments only on sections
    assert parse_route("/settings/s/response_handling#bad key").fragment is None


def test_router_alias_and_history():
    match = parse_route("/job/abc123")
    assert match.name == "jobs.detail" and match.path == "/jobs/abc123"
    router = Router()
    t0 = time.time()
    assert router.handle("/library?q=x") is not None
    assert router.handle("content://x/y") is None
    reasons = [r.reason for r in router.history]
    assert reasons[0].startswith("ok (dropped q")
    assert router.rejected_since(t0)[0].reason.startswith("blocked scheme")


def test_build_route_validates_everything():
    assert build_route("home") == "/"
    assert build_route("library.book", {"bid": HEX}, {"tab": "chapters"}) == f"/library/book/{HEX}?tab=chapters"
    assert build_route("settings.section", {"section": "response_handling"}, fragment="retry_timeout") == (
        "/settings/s/response_handling#retry_timeout"
    )
    with pytest.raises(RouteError):
        build_route("reader", {"bid": "/sdcard/book.epub"})
    with pytest.raises(RouteError):
        build_route("library", query={"q": "my book"})
    with pytest.raises(RouteError):
        build_route("library", query={"shelf": "everything"})
    with pytest.raises(RouteError):
        build_route("no.such.route")
    with pytest.raises(RouteError):
        build_route("settings.logs", fragment="x")


def test_route_table_is_consistent():
    names = [spec.name for spec in ROUTES]
    assert len(names) == len(set(names))
    for spec in ROUTES:
        assert spec.presentation in ("root", "view", "sheet", "fullscreen", "handled"), spec
        assert spec.milestone in {f"U{i}" for i in range(11)}, spec
        assert spec.parent is None or spec.parent in ROUTES_BY_NAME, spec
        if spec.is_static and spec.alias_of is None:
            assert parse_route(spec.pattern).name == spec.name
    # every UI_SPEC §1.4 family is whitelisted
    for name in ("home", "chat", "chat.settings", "library", "library.book", "reader", "jobs", "jobs.detail",
                 "glossary", "tools", "tools.files", "tools.text", "settings", "settings.section",
                 "settings.keys.pool", "welcome", "oauth.return", "selftest"):
        assert name in ROUTES_BY_NAME
    assert ROUTES_BY_NAME["settings.logs"].milestone == "U1"
    assert ROUTES_BY_NAME["selftest"].presentation == "handled"


def test_launch_links_from_shared_items():
    items = [
        {"id": "1", "kind": "url", "source": "launch", "text": "glossarion://app/__selftest__?suite=smoke"},
        {"id": "2", "kind": "url", "source": "share", "text": "glossarion://app/library"},
        {"id": "3", "kind": "file", "path": "/x/book.epub"},
        {"id": "4", "kind": "url", "source": "launch", "text": "https://example.com"},
    ]
    assert launch_links(items) == [("1", "glossarion://app/__selftest__?suite=smoke")]


# ==========================================================================
# Send/Stop state machine (UI_SPEC §2.4)
# ==========================================================================

SIGN_IN = chatgpt_sign_in_reason(DEFAULT_MODEL)


@pytest.mark.parametrize(
    "inputs, expected",
    [
        (SendInputs(), SendState.IDLE_EMPTY),
        (SendInputs(has_content=True), SendState.IDLE_READY),
        (SendInputs(has_content=True, other_job_running=True), SendState.QUEUE),
        (SendInputs(has_content=False, other_job_running=True), SendState.IDLE_EMPTY),
        (SendInputs(has_content=True, block=SIGN_IN), SendState.BLOCKED),
        (SendInputs(has_content=False, block=SIGN_IN), SendState.BLOCKED),  # reason visible up front
        (SendInputs(has_content=True, block=BLOCK_ENGINE_NOT_READY, other_job_running=True), SendState.BLOCKED),
        (SendInputs(own_job_state="RUNNING", block=SIGN_IN), SendState.RUNNING),
        (SendInputs(own_job_state="STARTING"), SendState.RUNNING),
        (SendInputs(own_job_state="STOPPING"), SendState.FINISHING),
        (SendInputs(own_job_state="FORCE_STOPPING"), SendState.STOPPING),
        (SendInputs(own_job_state="DONE", has_content=True), SendState.IDLE_READY),
        (SendInputs(own_job_state="FAILED"), SendState.IDLE_EMPTY),
        (SendInputs(own_job_state="QUEUED", has_content=True), SendState.IDLE_READY),
    ],
)
def test_derive_state_precedence(inputs, expected):
    assert derive_state(inputs) is expected


def test_send_machine_taps_and_long_press():
    machine = SendStopMachine()
    assert machine.state is SendState.IDLE_EMPTY and machine.tap() is SendAction.NONE
    machine.apply(SendInputs(has_content=True))
    assert machine.tap() is SendAction.SEND
    assert [a for a, _ in machine.long_press_items()] == [
        SendAction.SEND_ONCE_WITH_MODEL,
        SendAction.ADD_WITHOUT_TRANSLATING,
        SendAction.SEND_AS_SCRATCH,
    ]
    machine.apply(SendInputs(has_content=True, other_job_running=True, other_job_title="Book.epub"))
    assert machine.state is SendState.QUEUE and machine.tap() is SendAction.QUEUE
    assert machine.caption == "Send queues this message · runs after Book.epub"
    assert [label for _, label in machine.long_press_items()] == ["Queue", "Stop current & send"]
    machine.apply(SendInputs(has_content=True, block=SIGN_IN))
    assert machine.tap() is SendAction.EXPLAIN_BLOCK
    assert machine.caption == "Sign in with ChatGPT to use GPT-6 Luna"
    assert machine.long_press_items() == ()


def test_send_machine_graceful_then_force_stop():
    now = [100.0]
    machine = SendStopMachine(clock=lambda: now[0])
    machine.apply(SendInputs(own_job_state="RUNNING"))
    assert machine.state is SendState.RUNNING and machine.caption == "Translating…"
    assert machine.tap() is SendAction.STOP
    assert machine.state is SendState.FINISHING  # optimistic, JobService still says RUNNING
    assert machine.caption == "Finishing current request… Tap again to force stop"
    assert machine.within_force_window()
    now[0] += 5.0
    assert not machine.within_force_window()
    machine.apply(SendInputs(own_job_state="RUNNING"))  # same job state: keep the local step
    assert machine.state is SendState.FINISHING
    assert machine.tap() is SendAction.FORCE_STOP  # after the 2 s window a tap still forces (desktop)
    assert machine.state is SendState.STOPPING and machine.tap() is SendAction.NONE
    assert machine.caption == "Force stopping…"
    machine.apply(SendInputs(own_job_state="FORCE_STOPPING"))
    assert machine.state is SendState.STOPPING
    machine.apply(SendInputs(own_job_state="DONE"))
    assert machine.state is SendState.IDLE_EMPTY and machine.caption is None


def test_send_machine_job_service_transitions_and_menu():
    machine = SendStopMachine()
    machine.apply(SendInputs(own_job_state="RUNNING", graceful_stop=False))
    assert machine.tap() is SendAction.FORCE_STOP and machine.state is SendState.STOPPING
    machine.apply(SendInputs(own_job_state="RUNNING"))  # unchanged job state -> keep local
    assert machine.state is SendState.STOPPING
    machine = SendStopMachine()
    machine.apply(SendInputs(own_job_state="RUNNING"))
    assert machine.long_press_items() == ((SendAction.FORCE_STOP, "Force stop now"),)
    assert machine.select(SendAction.FORCE_STOP) is SendAction.FORCE_STOP
    assert machine.state is SendState.STOPPING
    machine.apply(SendInputs(own_job_state="STOPPING"))  # JobService moved on: local step cleared
    assert machine.state is SendState.FINISHING
    assert machine.finishing_since is not None


def test_send_visuals_and_reasons():
    blocked = visual_for(SendState.BLOCKED, SIGN_IN)
    assert blocked.enabled and blocked.style == "muted" and blocked.tooltip == SIGN_IN.message
    assert not visual_for(SendState.STOPPING).enabled
    assert visual_for(SendState.RUNNING).icon == "STOP"
    assert visual_for(SendState.QUEUE).icon == "SCHEDULE_SEND"
    assert SIGN_IN.fix_label == "Sign in with ChatGPT" and SIGN_IN.secondary_label == "Choose another model"
    assert chatgpt_sign_in_reason("authgpt2/gpt-5").message == "Sign in with ChatGPT to use authgpt2/gpt-5"
    for model, needed in [
        ("authgpt/gpt-6-luna", True),
        ("authgpt2/gpt-5", True),
        ("AUTHGPT/x", True),
        ("authgptx/y", False),
        ("gemini-3.5-flash", False),
        ("oc/gpt-6-luna", False),
        ("", False),
    ]:
        assert requires_chatgpt_sign_in(model) is needed, model
    assert status_caption(SendState.IDLE_READY, SendInputs(has_content=True)) is None
    assert status_caption(SendState.BLOCKED, SendInputs(block=BLOCK_MANUAL_GLOSSARY)) == "Manual glossary required"


# ==========================================================================
# Output modes (desktop Direct Text parity)
# ==========================================================================


def _desktop_constants() -> dict:
    """The Direct Text dialog constants, wherever the desktop code keeps them now: the dialog
    (translator_gui) and, since U3, the shared modules it inherits / imports them from
    (``direct_text_store.ChatStoreMixin``, ``translation_pipeline.IMAGE_ATTACHMENT_EXTENSIONS``,
    which the dialog's ``_IMAGE_ATTACHMENT_EXTENSIONS`` aliases)."""
    wanted = {
        "_OUTPUT_MODE_CHOICES": "_OUTPUT_MODE_CHOICES",
        "_IMAGE_ATTACHMENT_EXTENSIONS": "_IMAGE_ATTACHMENT_EXTENSIONS",
        "IMAGE_ATTACHMENT_EXTENSIONS": "_IMAGE_ATTACHMENT_EXTENSIONS",
        "_VISION_ARCHIVE_ATTACHMENT_EXTENSIONS": "_VISION_ARCHIVE_ATTACHMENT_EXTENSIONS",
    }
    found: dict = {}
    for module in ("translator_gui", "direct_text_store", "translation_pipeline"):
        path = SRC_DIR / f"{module}.py"
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                key = wanted.get(node.targets[0].id)
                if key is None or key in found:
                    continue
                try:
                    found[key] = ast.literal_eval(node.value)
                except ValueError:
                    continue  # an alias (``_IMAGE_ATTACHMENT_EXTENSIONS = IMAGE_ATTACHMENT_EXTENSIONS``)
    return found


def test_output_modes_match_desktop_direct_text():
    desktop = _desktop_constants()
    if "_OUTPUT_MODE_CHOICES" not in desktop:
        pytest.skip("translator_gui.py _OUTPUT_MODE_CHOICES not found")
    assert output_modes.OUTPUT_MODE_CHOICES == desktop["_OUTPUT_MODE_CHOICES"]
    assert [m.emoji for m in output_modes.OUTPUT_MODES] == ["📝", "👁️", "🖼️", "🎬", "🔊", "✨"]
    assert output_modes.IMAGE_ATTACHMENT_EXTENSIONS == frozenset(desktop["_IMAGE_ATTACHMENT_EXTENSIONS"])
    assert output_modes.VISION_ARCHIVE_ATTACHMENT_EXTENSIONS == frozenset(desktop["_VISION_ARCHIVE_ATTACHMENT_EXTENSIONS"])


def test_output_mode_state_rules():
    assert output_modes.normalize_mode("refine") == "refinement"
    assert output_modes.normalize_mode("nonsense") == "text"
    assert output_modes.mode_label("text") == "Output: Text"
    assert output_modes.mode_label("refine") == "Output: Refine"
    assert output_modes.semantics_label("vision", True) == "Vision output mode, selected"
    state = output_modes.OutputModeState().select("audio")
    assert state.label == "Output: Audio"
    auto = state.attachment_changed("/inbox/page01.PNG")
    assert (auto.mode, auto.automatic, auto.previous, auto.label) == ("vision", True, "audio", "Output: Vision · auto")
    assert auto.attachment_changed("/inbox/vol1.cbz") is auto
    restored = auto.attachment_changed(None)
    assert (restored.mode, restored.automatic) == ("audio", False)
    assert state.attachment_changed("/inbox/book.epub") == state
    assert auto.select("text") == output_modes.OutputModeState("text", False, None)


# ==========================================================================
# Signal store and loop guard
# ==========================================================================


def test_signal_notifies_on_change_only():
    guard = LoopGuard()
    sig = Signal(1, name="n", guard=guard)
    seen = []
    unsubscribe = sig.subscribe(seen.append)
    assert sig.set(1) is False and seen == []
    assert sig.set(2) is True and seen == [2] and sig.version == 1
    assert sig.update(lambda v: v + 1) and seen == [2, 3]
    assert sig.set(3, force=True) and seen == [2, 3, 3]
    unsubscribe()
    unsubscribe()  # idempotent
    sig.set(9)
    assert seen == [2, 3, 3]
    a, b = Signal(2, guard=guard), Signal(3, guard=guard)
    total = Computed(lambda: a.value * b.value, [a, b], guard=guard)
    assert total.value == 6
    a.set(5)
    assert total.value == 15
    with pytest.raises(TypeError):
        total.set(1)


def test_signal_set_from_worker_thread_raises_once_bound():
    guard = LoopGuard()
    sig = Signal(0, guard=guard)

    def worker(out):
        try:
            sig.set(1)
            out.append("ok")
        except WrongThreadError as exc:
            out.append(exc)

    unbound = []
    t = threading.Thread(target=worker, args=(unbound,))
    t.start()
    t.join()
    assert unbound == ["ok"]  # nothing bound yet: not enforced

    async def scenario():
        guard.bind()
        out = []
        t2 = threading.Thread(target=worker, args=(out,), name="gl-test-worker")
        t2.start()
        t2.join()
        assert isinstance(out[0], WrongThreadError) and "gl-test-worker" in str(out[0])
        assert sig.set(5)  # loop thread is fine

    asyncio.run(scenario())


# ==========================================================================
# UiDispatcher, Channel, LogBuffer
# ==========================================================================


def test_log_buffer_sequence_gap_and_threads():
    buf = LogBuffer(maxlen=5)
    for i in range(3):
        assert buf.append(f"line {i}") == i + 1
    lines, gap = buf.since(0)
    assert [ln.seq for ln in lines] == [1, 2, 3] and gap == 0
    for i in range(3, 10):
        buf.append(f"line {i}")
    lines, gap = buf.since(2)
    assert gap == 3 and [ln.seq for ln in lines] == [6, 7, 8, 9, 10]
    lines, gap = buf.since(7, limit=2)
    assert [ln.seq for ln in lines] == [8, 9] and gap == 0
    assert buf.since(10) == ([], 0)
    assert buf.first_seq == 6 and len(buf) == 5

    big = LogBuffer(maxlen=100_000)
    threads = [threading.Thread(target=lambda n=n: [big.append(f"t{n} {i}") for i in range(2000)]) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines, gap = big.since(0)
    assert big.last_seq == 16000 and gap == 0
    assert [ln.seq for ln in lines] == list(range(1, 16001))


def test_classify_log_lines():
    assert classify_log_line("❌ API error: 429") == "error"
    assert classify_log_line("12:00:01 ERROR glossarion.app: boom") == "error"
    assert classify_log_line("🧠 Thinking: planning the chapter") == "thinking"
    assert classify_log_line("📤 Sending API request to gemini") == "api"
    assert classify_log_line("Chapter 3/12 done · 0 failed") == "info"


def test_log_buffer_handler_routes_records():
    import logging

    buf = LogBuffer()
    handler = LogBufferHandler(buf)
    logger = logging.getLogger("glossarion.test.handler")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        logger.info("hello")
        logger.error("broken")
    finally:
        logger.removeHandler(handler)
    lines, _ = buf.since(0)
    assert [ln.kind for ln in lines] == ["info", "error"] and "hello" in lines[0].text


class _FakeControl:
    def __init__(self, parent=None, mounted=True):
        self.parent = parent
        self._mounted = mounted

    @property
    def page(self):
        if not self._mounted:
            raise RuntimeError("Control must be added to the page first")
        return object()


class _FakePage:
    def __init__(self):
        self.calls = []
        self.app_visible = True
        self._visible = None

    def update(self, *controls):
        self.calls.append(controls)

    async def wait_until_visible(self):
        await self._visible.wait()


def test_dispatcher_posts_channels_and_batches_across_threads():
    async def scenario():
        page = _FakePage()
        dispatcher = UiDispatcher(page, guard=LoopGuard()).bind()
        loop_thread = threading.get_ident()

        # post(): runs on the loop thread, in order, from many workers
        seen = []
        workers = [
            threading.Thread(target=lambda n=n: dispatcher.post(lambda: seen.append((n, threading.get_ident()))))
            for n in range(5)
        ]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        await asyncio.sleep(0.05)
        assert sorted(n for n, _ in seen) == list(range(5))
        assert {tid for _, tid in seen} == {loop_thread}

        # Channel: latest value wins
        channel = dispatcher.channel("progress")
        delivered = []
        channel.subscribe(delivered.append)
        writer = threading.Thread(target=lambda: [channel.set(i) for i in range(1, 1001)])
        writer.start()
        writer.join()
        assert channel.writes == 1000 and channel.get() == 1000
        dispatcher.flush()
        assert delivered == [1000]
        dispatcher.flush()
        assert delivered == [1000]  # nothing new

        # mark_dirty: one update; covered descendants and unmounted controls skipped
        parent = _FakeControl()
        child = _FakeControl(parent=parent)
        other = _FakeControl()
        ghost = _FakeControl(mounted=False)
        dispatcher.mark_dirty(child, parent, other, ghost, parent)
        assert dispatcher.dirty_count == 4
        assert dispatcher.flush() == 2
        assert page.calls == [(parent, other)]
        assert dispatcher.flush() == 0

        # mark_dirty from a worker thread is refused
        errors = []
        t = threading.Thread(target=lambda: _capture(errors, dispatcher.mark_dirty, other))
        t.start()
        t.join()
        assert isinstance(errors[0], WrongThreadError)

        # run_in_thread: value and exception
        assert await dispatcher.run_in_thread(lambda: threading.current_thread().name, name="gl-x") == "gl-x"
        with pytest.raises(ZeroDivisionError):
            await dispatcher.run_in_thread(lambda: 1 / 0)
        await dispatcher.stop()

    asyncio.run(scenario())


def _capture(out, fn, *args):
    try:
        fn(*args)
    except Exception as exc:
        out.append(exc)


def test_dispatcher_pump_delivers_logs_and_parks_while_hidden():
    async def scenario():
        page = _FakePage()
        page._visible = asyncio.Event()
        dispatcher = UiDispatcher(page, interval=0.02, log_batch=3, guard=LoopGuard()).bind()
        buffer = dispatcher.log_buffer("job")
        got = []
        dispatcher.subscribe_log(buffer, lambda lines, gap: got.append(([ln.text for ln in lines], gap)))
        dispatcher.start()
        threading.Thread(target=lambda: [buffer.append(f"l{i}") for i in range(7)]).start()
        for _ in range(100):
            if sum(len(batch) for batch, _ in got) >= 7:
                break
            await asyncio.sleep(0.02)
        assert [t for batch, _ in got for t in batch] == [f"l{i}" for i in range(7)]
        assert all(len(batch) <= 3 for batch, _ in got)

        # hidden app: the pump parks on wait_until_visible()
        page.app_visible = False
        channel = dispatcher.channel("x")
        values = []
        channel.subscribe(values.append)
        ticks = dispatcher.ticks
        threading.Thread(target=lambda: channel.set("while hidden")).start()
        await asyncio.sleep(0.2)
        assert values == [] and dispatcher.ticks == ticks
        page.app_visible = True
        page._visible.set()
        for _ in range(50):
            if values:
                break
            await asyncio.sleep(0.02)
        assert values == ["while hidden"]
        await dispatcher.stop()
        assert not dispatcher.running

    asyncio.run(scenario())


def test_dispatcher_pump_wakes_for_log_lines_after_parking():
    """A parked pump (idle for many intervals) still delivers a new subscription's
    backlog and lines appended later from worker threads."""

    async def scenario():
        page = _FakePage()
        dispatcher = UiDispatcher(page, interval=0.02, guard=LoopGuard()).bind()
        buffer = dispatcher.log_buffer("main")
        buffer.extend(["early-1", "early-2"])
        dispatcher.start()
        await asyncio.sleep(0.3)  # nothing subscribed: the pump parks on its wake event
        got = []
        dispatcher.subscribe_log(buffer, lambda lines, gap: got.extend(ln.text for ln in lines), backlog=500)
        assert await _wait(lambda: got == ["early-1", "early-2"], 2.0), got

        await asyncio.sleep(0.3)  # parked again
        ticks = dispatcher.ticks
        worker = threading.Thread(target=lambda: [buffer.append(f"w{i}") for i in range(50)])
        worker.start()
        worker.join()
        assert await _wait(lambda: len(got) == 52, 2.0), got
        assert got[2:] == [f"w{i}" for i in range(50)]
        assert dispatcher.ticks - ticks <= 10  # off-loop wake-ups are coalesced

        # lines for a buffer nobody reads do not wake the pump
        await asyncio.sleep(0.3)
        ticks = dispatcher.ticks
        idle = dispatcher.log_buffer("job")
        threading.Thread(target=lambda: [idle.append("unread") for _ in range(20)]).start()
        await asyncio.sleep(0.3)
        assert dispatcher.ticks == ticks

        # a buffer the dispatcher did not create gets the same hook when subscribed
        external = LogBuffer(name="external")
        seen = []
        dispatcher.subscribe_log(external, lambda lines, gap: seen.extend(ln.text for ln in lines))
        await asyncio.sleep(0.3)
        threading.Thread(target=external.append, args=("ext",)).start()
        assert await _wait(lambda: seen == ["ext"], 2.0), seen
        await dispatcher.stop()

    asyncio.run(scenario())


def test_log_subscription_reports_gaps():
    async def scenario():
        dispatcher = UiDispatcher(None, guard=LoopGuard()).bind()
        buffer = LogBuffer(maxlen=4)
        got = []
        dispatcher.subscribe_log(buffer, lambda lines, gap: got.append(([ln.seq for ln in lines], gap)))
        for i in range(10):
            buffer.append(str(i))
        dispatcher.flush()
        assert got == [([7, 8, 9, 10], 6)]

    asyncio.run(scenario())


# ==========================================================================
# Encryption keys in SecureStorage (services.secure_keys)
# ==========================================================================


class _FakeSecureStorage:
    def __init__(self, values=None, *, fail_get=False, fail_set=False):
        self.values = dict(values or {})
        self.fail_get = fail_get
        self.fail_set = fail_set
        self.sets = []

    async def get(self, key):
        if self.fail_get:
            raise RuntimeError("Keystore unavailable")
        return self.values.get(key)

    async def set(self, key, value):
        if self.fail_set:
            raise RuntimeError("Keystore read-only")
        self.values[key] = value
        self.sets.append(key)


def test_secure_keys_create_reuse_fallback_and_migrate(tmp_path):
    api_name, token_name = secure_keys.API_KEY_NAME, secure_keys.TOKEN_KEY_NAME
    fallback = tmp_path / "data" / secure_keys.FALLBACK_FILE_NAME

    async def scenario():
        # first run: both keys are created and stored
        storage = _FakeSecureStorage()
        first = await secure_keys.load_or_create(storage, fallback)
        assert (first.source, first.degraded) == ("created", False)
        assert storage.sets == [api_name, token_name]
        assert storage.values[api_name] == first.api_key.decode("ascii")
        assert len(base64.urlsafe_b64decode(first.api_key)) == 32
        assert base64.b64decode(storage.values[token_name]) == first.token_key and len(first.token_key) == 32
        # next run (e.g. after an app update): the same keys, nothing rewritten
        again = await secure_keys.load_or_create(storage, fallback)
        assert again.source == "secure_storage" and (again.api_key, again.token_key) == (first.api_key, first.token_key)
        assert storage.sets == [api_name, token_name] and not fallback.exists()

        # SecureStorage unusable: keys from a fallback file in the data dir, reused next time
        broken = _FakeSecureStorage(fail_get=True)
        degraded = await secure_keys.load_or_create(broken, fallback)
        assert degraded.degraded and degraded.source == "fallback_file" and fallback.is_file()
        assert (await secure_keys.load_or_create(broken, fallback)).api_key == degraded.api_key
        assert (await secure_keys.load_or_create(None, fallback)).token_key == degraded.token_key
        # SecureStorage works again: the fallback keys move into it and the file goes away
        fresh = _FakeSecureStorage()
        migrated = await secure_keys.load_or_create(fresh, fallback)
        assert (migrated.source, migrated.degraded) == ("migrated", False)
        assert (migrated.api_key, migrated.token_key) == (degraded.api_key, degraded.token_key)
        assert fresh.values[api_name] == degraded.api_key.decode("ascii") and not fallback.exists()

        # a failed write keeps the new keys in the fallback file for the next run
        read_only = _FakeSecureStorage(fail_set=True)
        kept = await secure_keys.load_or_create(read_only, fallback)
        assert kept.degraded and fallback.is_file()
        recovered = await secure_keys.load_or_create(_FakeSecureStorage(), fallback)
        assert recovered.api_key == kept.api_key and recovered.source == "migrated"

        # a stored key that does not parse is never overwritten
        odd = _FakeSecureStorage({api_name: "not-a-fernet-key", token_name: storage.values[token_name]})
        result = await secure_keys.load_or_create(odd, tmp_path / "odd" / secure_keys.FALLBACK_FILE_NAME)
        assert result.degraded and odd.sets == [] and odd.values[api_name] == "not-a-fernet-key"

    asyncio.run(scenario())


def test_secure_keys_install_reaches_the_backend(monkeypatch):
    pytest.importorskip("cryptography")
    monkeypatch.syspath_prepend(str(SRC_DIR))
    import api_key_encryption
    import token_encryption

    secure_keys.reset()
    try:
        api_key, token_key = secure_keys.new_api_key(), secure_keys.new_token_key()
        status = secure_keys.install_backend_keys(api_key, token_key, source="test")
        assert status.installed and status.errors == [] and secure_keys.current_status() is status
        handler = api_key_encryption.get_handler()
        assert handler.key_file is None  # no key file next to the backend
        token = handler.encrypt_value("sk-test")
        assert secure_keys.decrypt_with_installed_api_key(base64.b64decode(token[4:])) == b"sk-test"
        assert token_encryption._get_symmetric_key() == token_key
        assert status.as_dict()["api_key_fingerprint"] == secure_keys.fingerprint(api_key)
    finally:
        secure_keys.reset()
    assert secure_keys.current_status() is None and api_key_encryption._key_material is None


# ==========================================================================
# Design tokens, responsive classes, app state
# ==========================================================================


def test_design_tokens():
    assert tokens.SEED_COLOR == "#E18F98" and tokens.TERTIARY_LIGHT == "#5B3D57"
    assert tokens.ACCENTS["desktop_blue"] == "#5A9FD4" and tokens.ACCENTS["library_violet"] == "#6C63FF"
    sizes = [tokens.TYPE_SCALE[n].size for n in ("headline_small", "title_large", "title_medium", "title_small",
                                                  "body_large", "body_medium", "body_small")]
    assert sizes == [22, 18, 16, 14, 15, 14, 12]
    assert tokens.MONO_STYLE.size == 13
    assert [tokens.SPACING[k] for k in ("none", "xxs", "xs", "sm", "md", "lg", "xl", "xxl", "xxxl")] == [
        0, 2, 4, 8, 12, 16, 20, 24, 32
    ]
    assert tokens.RADII["badge"] == 6 and tokens.RADII["composer"] == 24
    assert tokens.SIZES["hit_target"] == 48 and tokens.SIZES["job_strip"] == 44 and tokens.SIZES["app_bar"] == 56
    assert tokens.semantic_color("success", False) == "#2E7D4F" and tokens.semantic_color("locked", True) == "#B388FF"
    assert tokens.status_color("completed", True) == "#27AE60"
    assert tokens.status_color("failed", False) == "role:error"
    assert tokens.status_color("cooling", True) == "#FFB74D"
    assert tokens.status_color("whatever", False) == "role:outline"
    assert set(tokens.STATUS_PALETTE) <= set(tokens.STATUS_STYLES)
    assert tokens.mono_family("android") == "monospace" and tokens.mono_family("ios") == "Menlo"
    assert tokens.gutter("phone") == 12 and tokens.gutter("large_phone") == 16 and tokens.gutter("tablet") == 24


@pytest.mark.parametrize(
    "width, size_class",
    [(None, "phone"), (0, "phone"), (412, "phone"), (599, "phone"), (600, "large_phone"), (899, "large_phone"),
     (900, "tablet"), (1199, "tablet"), (1200, "wide"), (2000, "wide")],
)
def test_size_classes(width, size_class):
    assert responsive.size_class_for(width).value == size_class


def test_layout_metrics():
    assert responsive.drawer_width(412) == pytest.approx(329.6)
    assert responsive.drawer_width(800) == 360
    # composer output mode (§2.3 item 3), from the chat-column width: chip < 600, toggles, labels once they fit
    assert responsive.output_row_style(380) == "chip" and responsive.output_row_style(412) == "chip"
    assert responsive.output_row_style(700) == "icons"
    assert responsive.output_row_style(700, text_scale=1.6) == "chip"
    assert responsive.output_row_style(1000) == "full"
    phone, large, tablet, wide = (responsive.layout_for(w) for w in (412, 700, 1000, 1300))
    assert (phone.chat_max_width, large.chat_max_width, tablet.chat_max_width) == (None, 760, 860)
    assert not phone.persistent_sidebar and tablet.persistent_sidebar and wide.persistent_sidebar
    assert (tablet.sidebar_width, wide.sidebar_width, tablet.side_panel_width) == (300, 320, 380)
    assert (phone.gutter, large.gutter, tablet.gutter) == (12, 16, 24)


def test_recents_grouping_and_placeholder_index():
    now = time.mktime((2026, 10, 15, 12, 0, 0, 0, 0, -1))
    day = 86400
    chats = [
        ChatSummary("1", "today", now - 60),
        ChatSummary("2", "yesterday", now - day),
        ChatSummary("3", "last week", now - 5 * day),
        ChatSummary("4", "august", time.mktime((2026, 8, 3, 12, 0, 0, 0, 0, -1))),
        ChatSummary("5", "pinned", now - 2 * day, pinned=True),
    ]
    groups = group_recents(chats, now)
    assert [(label, [c.cid for c in rows]) for label, rows in groups] == [
        ("Today", ["1"]),
        ("Yesterday", ["2"]),
        ("Previous 7 days", ["3"]),
        ("August 2026", ["4"]),
    ]
    index = InMemoryChatIndex(chats, clock=lambda: now)
    changes = []
    index.subscribe(lambda: changes.append(1))
    assert [c.cid for c in index.pinned()] == ["5"]
    assert index.set_pinned("1", True) and not index.set_pinned("1", True)
    assert [c.cid for c in index.pinned()] == ["1", "5"] or [c.cid for c in index.pinned()] == ["5", "1"]
    assert [c.cid for c in index.search("AUG")] == ["4"]
    assert changes == [1]
    default = InMemoryChatIndex()
    assert [c.title for c in default.all()] == ["New chat"]


def test_app_state_send_block_progression():
    state = AppState(guard=LoopGuard())
    assert state.chat_context.value == ChatContext("authgpt/gpt-6-luna", "Universal", "English", False)
    assert state.send_block() is BLOCK_ENGINE_NOT_READY
    state.backend.set({"ok": False, "failed": {"x": "boom"}})
    assert state.send_block().code == "engine_failed"
    state.backend.set({"ok": True})
    assert state.send_block().message == "Sign in with ChatGPT to use GPT-6 Luna"
    state.signed_in.set(frozenset({"authgpt"}))
    assert state.send_block() is None
    state.signed_in.set(frozenset())
    state.chat_context.set(ChatContext(model="gemini-3.5-flash"))
    assert state.send_block() is None


def test_app_state_send_block_uses_the_models_chatgpt_slot():
    """``signed_in`` holds slot keys (U4): ``authgptN/`` needs slot #N, the ``authgpt0/`` pool any slot."""
    state = AppState(guard=LoopGuard())
    state.backend.set({"ok": True})

    def block(model, signed):
        state.chat_context.set(ChatContext(model=model))
        state.signed_in.set(frozenset(signed))
        return state.send_block()

    assert block("authgpt2/gpt-6-luna", {"authgpt2"}) is None
    assert block("authgpt2/gpt-6-luna", {"authgpt"}).fix_action == "sign_in_chatgpt"
    assert block("authgpt0/gpt-6-luna", set()).code == "chatgpt_sign_in"
    assert block("authgpt0/gpt-6-luna", {"authgpt3"}) is None
    assert block("authgpt/gpt-6-luna", {"authgpt2"}).code == "chatgpt_sign_in"
    assert block("authgem2/gemini-3.5-pro", set()) is None  # only ChatGPT routes gate Send


@needs_flet
def test_drawer_status_uses_the_models_chatgpt_slot():
    from glossarion_mobile.ui.shell.drawer import drawer_status

    state = AppState(guard=LoopGuard())
    state.backend.set({"ok": True})
    state.chat_context.set(ChatContext(model="authgpt2/gpt-6-luna"))
    state.signed_in.set(frozenset({"authgpt2"}))
    assert drawer_status(state) == ("authgpt2/gpt-6-luna · Ready", False)
    state.signed_in.set(frozenset({"authgpt"}))
    assert drawer_status(state) == ("authgpt2/gpt-6-luna · Sign in with ChatGPT", True)


# ==========================================================================
# Flet-backed parts (theme, components, chat, shell, app)
# ==========================================================================


@needs_flet
def test_theme_build_and_apply():
    import flet as ft

    from glossarion_mobile.ui import theme

    light = theme.build_theme(dark=False)
    dark = theme.build_theme(dark=True)
    amoled = theme.build_theme(dark=True, amoled=True)
    for t in (light, dark, amoled):
        assert t.color_scheme_seed == "#E18F98"
        assert t.use_material3 and t.visual_density == ft.VisualDensity.COMPACT
        assert t.text_theme.title_medium.size == 16 and t.text_theme.body_large.size == 15
    assert light.color_scheme.tertiary == "#5B3D57"
    assert dark.color_scheme is None
    assert amoled.color_scheme.surface == "#000000" and amoled.color_scheme.surface_container_low == "#0A0A0A"
    assert amoled.scaffold_bgcolor == "#000000"
    assert theme.build_theme(text_scale=2.0).text_theme.body_medium.size == 28
    assert theme.HIT_TARGET.min_width == 48 and theme.HIT_TARGET.min_height == 48
    assert theme.status_color("failed") == ft.Colors.ERROR
    assert theme.status_color("pending") == ft.Colors.OUTLINE
    for name, style in tokens.STATUS_STYLES.items():
        assert hasattr(ft.Icons, style.icon), (name, style.icon)

    class P:
        theme = dark_theme = theme_mode = None
        platform_brightness = None

    page = P()
    theme.apply_theme(page, theme.Appearance.AMOLED)
    assert page.theme_mode == ft.ThemeMode.DARK and page.dark_theme.color_scheme.surface == "#000000"
    assert theme.is_dark(page)
    theme.apply_theme(page, theme.Appearance.SYSTEM)
    assert page.theme_mode == ft.ThemeMode.SYSTEM and not theme.is_dark(page)


def _mount(controls):
    """Put controls on a fake page and send one update (full msgpack encoding)."""

    async def scenario():
        conn, session = _fake_session("android")
        page = session.page
        page.views[0].controls.extend(controls)
        page.update()
        return conn, session

    return asyncio.run(scenario())


@needs_flet
def test_components_construct_and_serialize():
    import flet as ft

    from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
    from glossarion_mobile.ui.components.dialogs import ConfirmDialog
    from glossarion_mobile.ui.components.empty_state import EmptyState
    from glossarion_mobile.ui.components.reason_chip import ReasonChip
    from glossarion_mobile.ui.components.section_card import SectionCard
    from glossarion_mobile.ui.components.status import StatusChip
    from glossarion_mobile.ui.shell.job_strip import JobStrip

    picked = []
    sheet = ActionSheet(
        [
            ActionItem("Rename", lambda: picked.append("rename"), icon="DRIVE_FILE_RENAME_OUTLINE"),
            ActionItem("Delete", lambda: picked.append("delete"), destructive=True),
            ActionItem("Move to Series…", disabled_reason="Arrives in U9"),
        ],
        title="My chat",
    )
    assert isinstance(sheet.dialog, ft.BottomSheet) and sheet.dialog.show_drag_handle and sheet.dialog.scrollable
    assert len(sheet.tiles) == 3 and sheet.tiles[1].title.color == ft.Colors.ERROR
    # U9: an unavailable item keeps an enabled row (its ReasonChip must stay tappable); a tap explains it
    assert not sheet.tiles[2].disabled and isinstance(sheet.tiles[2].trailing, ReasonChip)
    assert sheet.tiles[2].title.color == ft.Colors.ON_SURFACE_VARIANT
    sheet._on_select(None, sheet.item("Rename"))
    sheet._on_select(None, sheet.item("Move to Series…"))
    assert picked == ["rename"] and sheet.explained is sheet.item("Move to Series…")
    assert isinstance(ActionSheet([ActionItem("A")], tablet=True).dialog, ft.AlertDialog)

    confirm = ConfirmDialog(title="Delete chat?", body="This permanently removes…", destructive=True, items=["Folder A"])
    assert confirm.dialog.modal and confirm.confirm_button.style.bgcolor == ft.Colors.ERROR

    empty = EmptyState(title="No jobs yet", body="Translations… appear here.", icon="WORK_HISTORY",
                       primary=("Open Library", lambda e: None), suggestions=[("Paste text", lambda e: None)])
    card = SectionCard(title="Runtime", children=[ft.Text("x")], icon="INFO_OUTLINE", modified_count=2)
    folded = SectionCard(title="Advanced", children=[ft.Text("y")], collapsible=True, expanded=False)
    assert isinstance(folded.content, ft.ExpansionTile) and folded.content.expanded is False
    assert card.modified_dot.visible
    chip = StatusChip(status="qa_failed", count=3)
    assert chip.label_text.value == "QA Failed · 3" and chip.icon_control.icon == ft.Icons.REPORT
    assert chip.icon_control.color == ft.Colors.ERROR
    assert not StatusChip(status="pending", count=0).visible
    strip = JobStrip()
    assert not strip.visible
    strip.set_model(JobStripModel("Translating · Book.epub", "Ch 12/80 · 3 in flight", 0.15, queued=2))
    assert strip.visible and strip.ring.value == pytest.approx(0.15)
    assert strip.stop_button.badge.label == "+2" and strip.semantics.live_region
    conn, _session = _mount([empty, card, folded, chip, strip, ReasonChip(reason="Not available on mobile")])
    assert conn.bytes_sent > 0


@needs_flet
def test_log_console_blocks_filters_and_cap():
    from glossarion_mobile.services.dispatcher import LogLine
    from glossarion_mobile.ui.components.log_console import LogConsole

    console = LogConsole(block_lines=3, max_blocks=2, list_height=200)
    lines = [LogLine(i + 1, 0.0, f"line {i}", "error" if i % 2 else "info") for i in range(8)]
    console.on_lines(lines[:4], 0)
    assert [b.value for b in console.list_view.controls] == ["line 0\nline 1\nline 2", "line 3"]
    console.on_lines(lines[4:], 0)
    assert len(console.list_view.controls) == 2  # at most max_blocks mounted
    assert console.list_view.controls[-1].value == "line 6\nline 7"
    console.set_filter("errors")
    assert "\n".join(b.value for b in console.list_view.controls) == "line 3\nline 5\nline 7"
    assert console.visible_text() == "line 3\nline 5\nline 7"  # lines 0-1 fell out of the 6-line store
    console.set_filter("thinking")
    assert console.list_view.controls == [console.empty_text]
    console.on_lines([], 5)
    assert console.gap_total == 5


@needs_flet
def test_output_mode_row_and_send_button():
    import flet as ft

    from glossarion_mobile.ui.chat.output_mode_row import OutputModeRow
    from glossarion_mobile.ui.chat.send_button import SendStopButton

    opened = []
    row = OutputModeRow(on_open_options=opened.append)
    assert row.label_text.value == "Output: Text"
    assert [t.icon.value for t in row.toggles.values()] == ["📝", "👁️", "🖼️", "🎬", "🔊", "✨"]
    assert all(t.size_constraints.min_width == 48 for t in row.toggles.values())
    assert row.toggles["text"].selected and row.semantics["text"].label == "Text output mode, selected"
    assert row.toggles["vision"].tooltip == "Output mode: Vision"
    assert row.tap("vision") == "selected" and row.mode == "vision"
    assert row.label_text.value == "Output: Vision" and row.toggles["vision"].selected
    assert row.tap("vision") == "options" and opened == ["vision"]
    row.mode_signal.set(row.state.attachment_changed("x.png").select("image").attachment_changed("x.jpg"))
    row._sync()
    assert row.label_text.value == "Output: Vision · auto"
    assert row.set_style("icons") and row.label_text not in [getattr(c, "content", None) for c in row.controls]
    assert row.set_style("full") and isinstance(row.toggles["audio"], ft.Container)
    assert not row.set_style("full")

    actions = []
    button = SendStopButton(on_action=actions.append)
    for inputs, state, kind in [
        (SendInputs(), SendState.IDLE_EMPTY, ft.IconButton),
        (SendInputs(has_content=True), SendState.IDLE_READY, ft.FilledIconButton),
        (SendInputs(has_content=True, other_job_running=True), SendState.QUEUE, ft.FilledTonalIconButton),
        (SendInputs(block=SIGN_IN), SendState.BLOCKED, ft.IconButton),
        (SendInputs(own_job_state="RUNNING"), SendState.RUNNING, ft.FilledIconButton),
        (SendInputs(own_job_state="STOPPING"), SendState.FINISHING, ft.Container),
        (SendInputs(own_job_state="FORCE_STOPPING"), SendState.STOPPING, ft.Container),
    ]:
        button.apply(inputs)
        assert button.state is state
        visual = button.switcher.content
        assert type(visual) is kind, (state, type(visual))
        assert visual.key == f"send-{state.value}"
        assert not getattr(visual, "disabled", False)  # blocked stays tappable
    button.apply(SendInputs(block=SIGN_IN))
    assert button.switcher.content.tooltip == "Sign in with ChatGPT to use GPT-6 Luna"
    assert button.menu.primary_trigger == ft.ContextMenuTrigger.LONG_PRESS and button.menu.secondary_trigger is None
    assert button.tap() is SendAction.EXPLAIN_BLOCK and actions == [SendAction.EXPLAIN_BLOCK]
    button.apply(SendInputs(own_job_state="RUNNING"))
    assert [i.content for i in button.menu.primary_items] == ["Force stop now"]
    assert button.tap() is SendAction.STOP and button.state is SendState.FINISHING
    button._on_menu(SendAction.FORCE_STOP)
    assert button.state is SendState.STOPPING and actions[-1] is SendAction.FORCE_STOP


@needs_flet
def test_composer_header_transcript_and_plus_sheet():
    import flet as ft

    from glossarion_mobile.ui.chat.composer import HINT_ATTACHMENT, HINT_EMPTY, PASTE_CHIP_THRESHOLD, Composer
    from glossarion_mobile.ui.chat.header import ChatHeader
    from glossarion_mobile.ui.chat.plus_sheet import TOOLS, PlusSheet
    from glossarion_mobile.ui.chat.transcript import EMPTY_BODY, EMPTY_TITLE, Transcript
    from glossarion_mobile.ui.components.reason_chip import ReasonChip

    changes = []
    composer = Composer(on_content_changed=changes.append)
    field = composer.text_field
    assert (field.multiline, field.min_lines, field.max_lines, field.shift_enter) == (True, 1, 6, True)
    assert field.hint_text == HINT_EMPTY == "Message to translate…"
    assert not composer.has_content and not composer.chips_row.visible
    composer.handle_text("hello")
    assert composer.has_content and changes[-1] is True and not composer.expand_button.visible
    composer.handle_text("hello\nworld\nthird line")
    assert composer.expand_button.visible
    composer.handle_text("typed ")
    pasted = "x" * (PASTE_CHIP_THRESHOLD + 1)
    composer.handle_text("typed " + pasted)
    assert field.value == "typed " and len(composer.pasted_chips) == 1 and composer.chips_row.visible
    assert composer.pasted_chips[0].label.value == f"Pasted text · {len(pasted):,} chars"
    composer.handle_text("")
    assert composer.has_content  # the chip still counts
    composer.pasted_chips[0]._remove()
    assert not composer.has_content and not composer.chips_row.visible
    composer.set_attachment_hint(True)
    assert field.hint_text == HINT_ATTACHMENT and composer.has_content
    composer.set_plus_open(True)
    assert composer.plus_button.rotate.angle == pytest.approx(3.14159265 / 4)
    composer.set_compact_text(True)
    assert field.max_lines == 4
    assert composer.border_radius == 24 and composer.output_row.label_text.value == "Output: Text"

    header = ChatHeader()
    assert header.subtitle_text == "authgpt/gpt-6-luna · Universal · → English ▾"
    bar = header.build(tablet=False)
    assert isinstance(bar, ft.AppBar) and bar.bgcolor == ft.Colors.SURFACE and bar.elevation_on_scroll == 0
    assert bar.leading is header.menu_button and header.menu_button.size_constraints.min_height == 48

    class E:
        pixels = 30

    header.on_transcript_scroll(E())
    assert bar.bgcolor == ft.Colors.SURFACE_CONTAINER and header.scrolled
    E.pixels = 0
    header.on_transcript_scroll(E())
    assert bar.bgcolor == ft.Colors.SURFACE
    header.set_compact_text(True)
    assert header.subtitle_text == "authgpt/gpt-6-luna ▾"
    assert isinstance(header.build(tablet=True), ft.Container)
    opened = []
    header.on_open_model_sheet = opened.append
    header.profile_span.on_click(None)
    assert opened == ["profile"]

    transcript = Transcript()
    assert transcript.is_empty and transcript.controls == [transcript.empty_state]
    assert transcript.empty_state.title_text.value == EMPTY_TITLE == "What would you like to translate?"
    assert EMPTY_BODY.startswith("Paste text into the composer below, or attach a supported file.")
    assert [c.label.value for c in transcript.empty_state.suggestion_chips] == [
        "Paste text", "Attach a book", "From Library", "Translate a manga page"
    ]
    assert transcript.build_controls_on_demand is False and transcript.auto_scroll is False

    plus = PlusSheet()
    assert plus.dialog.show_drag_handle and plus.dialog.draggable and plus.dialog.scrollable
    assert list(plus.tiles) == ["files", "library", "photos", "camera", "clipboard"]
    camera = plus.tiles["camera"]
    assert camera.on_click is None and any(isinstance(c, ReasonChip) for c in camera.content.controls)
    assert len(plus.tool_tiles) == len(TOOLS) == 10
    conn, _ = _mount([composer, header.build(tablet=True), transcript])
    assert conn.bytes_sent > 0


# ---- app + shell on the fake session -------------------------------------------------------


def _load_main_module():
    spec = importlib.util.spec_from_file_location("glossarion_mobile_app_main_u1", APP_DIR / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def _start(platform="android", width=412, *, first_run=False):
    """Start the real app on a fake session. ``app_env`` starts it as a returning user (the
    first-run Welcome flow, U3, is marked done in ``mobile_state.json``); ``first_run=True``
    removes that mark, so the Welcome flow opens over the chat home."""
    if first_run:
        paths = rb.get_paths()
        if paths is not None:
            (Path(paths.data) / "mobile_state.json").unlink(missing_ok=True)
    main_module = _load_main_module()
    conn, session = _fake_session(platform)
    session.apply_page_patch({"width": width, "height": 860})
    page = session.page
    await main_module.main(page)
    await session.after_event(page)
    return main_module, conn, session, page, page.data


async def _stop(app):
    if app.spike is not None:
        app.spike._fgs_stop.set()
        app.spike._cp_stop.set()
        app.spike._close_oauth()
    await app.dispatcher.stop()
    await asyncio.sleep(0)


async def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return predicate()


def _routes(page):
    return [v.route for v in page.views]


def _echo_push_routes(conn, session):
    """Make the fake client answer ``push_route`` like Flutter does: patch page.route and
    fire ``route_change`` for the pushed route."""
    from flet.messaging.protocol import MessageAction

    send = conn.send_message
    echoed = []

    def send_message(message):
        send(message)
        if message.action == MessageAction.INVOKE_METHOD and message.body.name == "push_route":
            route = (message.body.args or {}).get("route")
            echoed.append(route)
            conn.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(_route(session, route)))

    conn.send_message = send_message
    return echoed


@needs_flet
def test_app_builds_chat_shell_offline(app_env, capsys):
    import flet as ft

    async def scenario():
        main_module, conn, session, page, app = await _start("windows")
        try:
            assert main_module.PATHS is rb.get_paths()
            assert type(app).__name__ == "GlossarionApp" and app.native.is_stub
            assert page.theme.color_scheme_seed == "#E18F98" and page.theme.visual_density == ft.VisualDensity.COMPACT
            assert _routes(page) == ["/"]
            root = page.views[0]
            assert isinstance(root.appbar, ft.AppBar) and isinstance(root.drawer, ft.NavigationDrawer)
            assert root.drawer.controls[0].content is app.drawer.content
            assert app.chat_view.transcript.is_empty
            assert app_env, "warm import was not started"
            await _wait(lambda: app.state.engine_ready)
            assert app.chat_view.composer.send_state is SendState.BLOCKED
            assert app.chat_view.caption.visible and app.chat_view.caption.text.value == "Sign in with ChatGPT to use GPT-6 Luna"
            assert app.chat_view.caption.fix_button.content == "Sign in with ChatGPT"
            assert app.drawer.status_text.value == "authgpt/gpt-6-luna · Sign in with ChatGPT"
            assert app.drawer.section_titles == ["Today"] and "1" in app.drawer.chat_rows
            assert conn.bytes_sent > 0
        finally:
            await _stop(app)

    asyncio.run(scenario())
    assert "GLOSSARION_READY" in capsys.readouterr().err.splitlines()


@needs_flet
def test_app_routes_selftest_and_view_stack(app_env):
    from glossarion_mobile.ui.screens.base import HubScreen, PlaceholderScreen
    from glossarion_mobile.ui.screens.diagnostics import DiagnosticsScreen

    async def scenario():
        _m, conn, session, page, app = await _start("android")
        try:
            await _route(session, "/settings/logs")
            assert _routes(page) == ["/", "/settings", "/settings/logs"]
            assert isinstance(app.shell.top_screen, DiagnosticsScreen)
            assert isinstance(app.shell.stack[0].screen, HubScreen)

            # ignored: open-with paths, foreign links; the stack does not change
            for raw in ("/document/raw%3A%2Fx.epub", "content://media/external/file/12", "https://evil.example/jobs"):
                await _route(session, raw)
            assert _routes(page) == ["/", "/settings", "/settings/logs"]
            rejected = [r.raw for r in app.router.history if not r.accepted]
            assert "/document/raw%3A%2Fx.epub" in rejected

            # the self-test deep link runs on a worker thread and leaves the stack alone
            await _route(session, "glossarion://app/__selftest__?suite=smoke")
            assert await _wait(lambda: app.state.selftest_result.value is not None)
            assert app.state.selftest_result.value["ok"] and app.state.selftest_result.value["source"] == "route/event"
            assert _routes(page) == ["/", "/settings", "/settings/logs"]
            assert "push_route" in conn.invoked()  # route restored so the link can fire again
            screen = app.shell.top_screen
            assert screen.result_text.value.startswith("PASS · 1 passed")

            # the Diagnostics button uses the same runner
            result = await screen._on_run()
            assert result["source"] == "diagnostics" and app.selftest.runs == 2

            # OAuth return is handled without a View
            await _route(session, "/oauth/return?p=authgpt")
            assert _routes(page) == ["/", "/settings", "/settings/logs"]

            # Android back pops one level
            await session.dispatch_event(page._i, "view_pop", {"route": "/settings/logs"})
            assert _routes(page) == ["/", "/settings"]

            # the Library (U5), the Glossaries (U6), Tools › Manga (U8) and Series (U9); in-app
            # navigation syncs the client route
            from glossarion_mobile.ui.glossary.home import GlossariesScreen
            from glossarion_mobile.ui.library.home import LibraryScreen

            await app.navigate("/library")
            assert _routes(page) == ["/", "/library"] and isinstance(app.shell.top_screen, LibraryScreen)
            await app.navigate("/glossary")
            assert _routes(page) == ["/", "/glossary"] and isinstance(app.shell.top_screen, GlossariesScreen)
            await app.navigate("/tools/manga")
            assert _routes(page) == ["/", "/glossary", "/tools/manga"]
            assert not isinstance(app.shell.top_screen, PlaceholderScreen)
            assert getattr(app, "manga", None) is not None and "tools.manga" in app.manga.screens_built
            await app.navigate("/series/s1")
            assert _routes(page)[-1] == "/series/s1"
            from glossarion_mobile.ui.chat.series_page import SeriesScreen
            assert isinstance(app.shell.top_screen, SeriesScreen)
            app.navigate_to("jobs")
            assert await _wait(lambda: _routes(page) == ["/", "/jobs"])
            await _route(session, "/")
            assert _routes(page) == ["/"] and app.shell.stack == []
        finally:
            await _stop(app)

    asyncio.run(scenario())


@needs_flet
def test_shell_switches_phone_and_tablet_on_resize(app_env):
    import flet as ft

    async def scenario():
        _m, conn, session, page, app = await _start("android", width=412)
        try:
            await _route(session, "/settings/logs")
            builds = app.shell.builds
            await session.dispatch_event(page._i, "resize", {"width": 1000, "height": 800})
            assert app.shell.tablet and app.state.size_class.value is responsive.SizeClass.TABLET
            assert _routes(page) == ["/"]  # tablet: one root View
            row = page.views[0].controls[0]
            assert isinstance(row, ft.Row) and row.controls[0] is app.shell.sidebar
            assert app.shell.sidebar.width == 300 and page.views[0].drawer is None
            assert app.drawer.content in _walk(app.shell.sidebar)
            assert app.shell.top_screen is not None and app.shell.top_screen.body in _walk(app.shell.main_area)
            assert app.chat_view.composer.output_row.style_name == "icons"  # 700 dp chat column
            # within the same class nothing is rebuilt; the composer still follows the chat column
            builds_tablet = app.shell.builds
            await session.dispatch_event(page._i, "resize", {"width": 1150, "height": 800})
            assert app.shell.builds == builds_tablet
            assert app.chat_view.composer.output_row.style_name == "full"  # 850 dp: labelled toggles fit
            await session.dispatch_event(page._i, "resize", {"width": 1300, "height": 800})
            assert app.shell.size_class is responsive.SizeClass.WIDE and app.shell.sidebar.width == 320
            assert app.shell.builds == builds_tablet  # tablet -> wide keeps the panes
            # tablet back button pops the main-area stack
            app.back()
            assert app.shell.current_route == "/settings"
            # back to phone: modal drawer again, stack becomes Views
            await session.dispatch_event(page._i, "resize", {"width": 412, "height": 860})
            assert not app.shell.tablet and app.shell.builds > builds
            assert _routes(page) == ["/", "/settings"]
            assert isinstance(page.views[0].drawer, ft.NavigationDrawer)
            assert app.chat_view.composer.output_row.style_name == "chip"
            await session.dispatch_event(page._i, "resize", {"width": 380, "height": 860})
            assert app.chat_view.composer.output_row.style_name == "chip"  # same class, same chip
            assert conn.bytes_sent > 0
        finally:
            await _stop(app)

    asyncio.run(scenario())


def _walk(control, depth=0):
    """All controls under ``control`` (content / controls / leading / trailing)."""
    out = [control]
    if depth > 40 or control is None:
        return out
    for attr in ("content", "controls", "leading", "trailing", "title"):
        child = getattr(control, attr, None)
        if isinstance(child, list):
            for c in child:
                out.extend(_walk(c, depth + 1))
        elif child is not None and hasattr(child, "_i"):
            out.extend(_walk(child, depth + 1))
    return out


@needs_flet
def test_drawer_chat_view_actions_and_worker_updates(app_env):
    async def scenario():
        _m, conn, session, page, app = await _start("android")
        try:
            await _wait(lambda: app.state.engine_ready)
            drawer = app.drawer
            # destination chip -> pushed view
            await session.dispatch_event(drawer.destination_chips["tools"]._i, "click", None)
            assert await _wait(lambda: _routes(page) == ["/", "/tools"])
            await _route(session, "/")
            # pin through the long-press action sheet -> Pinned section
            sheet = app._chat_actions(app.state.chats.get("1"))
            sheet._on_select(None, sheet.item("Pin"))
            assert drawer.section_titles[0] == "Pinned"
            # search
            drawer.set_query("new")
            assert "Chats" in drawer.section_titles
            drawer.set_query("zzz")
            assert drawer.chat_rows == {}
            drawer.set_query("")
            # send tap while blocked explains the reason (snackbar with the fix action)
            assert app.chat_view.composer.send_button.tap() is SendAction.EXPLAIN_BLOCK
            # ＋ sheet and the active-mode options sheet
            plus = app.chat_view.open_plus_sheet()
            assert plus.dialog.open and app.chat_view.composer.plus_button.rotate.angle > 0
            plus._select(plus.on_tool, "qa")
            assert await _wait(lambda: _routes(page) == ["/", "/tools", "/tools/qa"])
            assert app.chat_view.composer.plus_button.rotate.angle == 0
            await _route(session, "/")
            sheet = app.chat_view.open_mode_options("text")
            assert sheet.dialog.open
            # a worker thread publishes a job for another chat through the dispatcher
            model = JobStripModel("Translating · Book.epub", "Ch 3/80", 0.04, owner_chat="99")
            t = threading.Thread(target=lambda: app.dispatcher.post(app.state.job_strip.set, model))
            t.start()
            t.join()
            assert await _wait(lambda: app.chat_view.job_strip.visible)
            app.chat_view.composer.handle_text("translate me")
            assert app.chat_view.composer.send_state is SendState.BLOCKED  # still not signed in
            app.state.signed_in.set(frozenset({"authgpt"}))
            assert app.chat_view.composer.send_state is SendState.QUEUE
            assert app.chat_view.caption.text.value == "Send queues this message · runs after Translating · Book.epub"
            app.state.job_strip.set(JobStripModel("Mine", owner_chat="1"))
            assert not app.chat_view.job_strip.visible
            assert app.chat_view.composer.send_state is SendState.IDLE_READY
            # Signals refuse worker-thread writes once the app is running
            errors = []
            t = threading.Thread(target=lambda: _capture(errors, app.state.job_strip.set, None))
            t.start()
            t.join()
            assert isinstance(errors[0], WrongThreadError)
        finally:
            await _stop(app)

    asyncio.run(scenario())


@needs_flet
def test_device_checks_open_from_diagnostics_and_oauth_return(app_env):
    import urllib.request

    async def scenario():
        _m, conn, session, page, app = await _start("windows")
        try:
            await _route(session, "/settings/logs")
            app.shell.top_screen._on_device_checks()
            assert _routes(page)[-1] == "/settings/logs/device-checks"
            spike = app.spike
            assert type(spike).__name__ == "SpikeApp" and spike.embedded and spike.native is app.native
            assert set(spike.cards) >= {"selftest", "secure", "fgs", "notify", "share", "oauth", "stack", "background"}
            opened = []
            rb.set_url_opener(opened.append)  # stand-in for UrlLauncher
            await spike.oauth_test()
            assert opened and opened[0].startswith("http://127.0.0.1:")
            body = await asyncio.to_thread(lambda: urllib.request.urlopen(opened[0], timeout=10).read().decode())
            assert "glossarion://app/oauth/return?p=spike&amp;nonce=" in body
            await asyncio.sleep(0.2)
            nonce = spike._oauth["nonce"]
            echoed = _echo_push_routes(conn, session)  # from here on the client echoes push_route
            await _route(session, f"/oauth/return?p=spike&nonce={nonce}")
            assert spike.cards["oauth"].status == "pass", spike.cards["oauth"].result.value
            # the app restores the client route; its route_change echo must not close the overlay
            assert await _wait(lambda: echoed == ["/settings/logs"] and page.route == "/settings/logs")
            await asyncio.sleep(0.2)
            assert _routes(page)[-1] == "/settings/logs/device-checks"  # the screen stays open
            # same for a self-test deep link fired while Device checks is open
            await _route(session, "glossarion://app/__selftest__?suite=smoke")
            assert await _wait(lambda: app.state.selftest_result.value is not None)
            assert await _wait(lambda: len(echoed) == 2 and page.route == "/settings/logs")
            await asyncio.sleep(0.2)
            assert _routes(page)[-1] == "/settings/logs/device-checks"
            assert app._route_echoes == []  # every echo was consumed
            await spike.stack_test()
            assert spike.cards["stack"].status == "pass"
            await session.dispatch_event(page._i, "view_pop", {"route": "/settings/logs/device-checks"})
            assert _routes(page) == ["/", "/settings", "/settings/logs"]
        finally:
            await _stop(app)

    asyncio.run(scenario())


@needs_flet
def test_app_installs_secure_storage_keys_before_warm_import(app_env, monkeypatch):
    import flet_secure_storage
    from glossarion_mobile.diagnostics import selftest

    pytest.importorskip("cryptography")
    import api_key_encryption  # backend dir (repo src/) is on sys.path after bootstrap
    import token_encryption

    events = []
    stored = {}

    async def fake_get(self, key):
        events.append(("get", key))
        return stored.get(key)

    async def fake_set(self, key, value):
        events.append(("set", key))
        stored[key] = value

    monkeypatch.setattr(flet_secure_storage.SecureStorage, "get", fake_get)
    monkeypatch.setattr(flet_secure_storage.SecureStorage, "set", fake_set)
    real_api_setter = api_key_encryption.set_key_material
    real_token_setter = token_encryption.set_symmetric_key

    def api_setter(key):
        events.append(("set_key_material", key))
        real_api_setter(key)

    def token_setter(key):
        events.append(("set_symmetric_key", key))
        real_token_setter(key)

    monkeypatch.setattr(api_key_encryption, "set_key_material", api_setter)
    monkeypatch.setattr(token_encryption, "set_symmetric_key", token_setter)
    warm = rb.start_warm_import  # app_env's fake

    def ordered_warm(*args, **kwargs):
        events.append(("warm_import",))
        return warm(*args, **kwargs)

    monkeypatch.setattr(rb, "start_warm_import", ordered_warm)

    def kinds():
        return [e[0] for e in events]

    async def scenario():
        _m, conn, session, page, app = await _start("android")
        try:
            # first launch: keys created in SecureStorage, handed to the backend, then the warm import
            assert kinds() == ["get", "get", "set", "set", "set_key_material", "set_symmetric_key", "warm_import"]
            assert app.keys_ready.is_set()
            assert (app.key_status.source, app.key_status.degraded, app.key_status.installed) == ("created", False, True)
            api_key = events[4][1]
            assert api_key.decode("ascii") == stored[secure_keys.API_KEY_NAME]
            assert base64.b64decode(stored[secure_keys.TOKEN_KEY_NAME]) == events[5][1]
            assert api_key_encryption.get_handler().key_file is None
            detail = selftest.check_encryption_keys(selftest.Context(strict=False))
            assert detail["source"] == "created" and detail["api_key"] == secure_keys.fingerprint(api_key)
            paths = rb.get_paths()
            for directory in (paths.data, paths.home):
                assert not (directory / ".glossarion_key").exists()
                assert not (directory / secure_keys.FALLBACK_FILE_NAME).exists()
        finally:
            await _stop(app)

        # next launch (e.g. after an update): the stored keys are reused, nothing is written
        events.clear()
        _m, conn, session, page, app = await _start("android")
        try:
            assert kinds() == ["get", "get", "set_key_material", "set_symmetric_key", "warm_import"]
            assert app.key_status.source == "secure_storage"
            assert events[2][1].decode("ascii") == stored[secure_keys.API_KEY_NAME]
        finally:
            await _stop(app)

    asyncio.run(scenario())


@needs_flet
def test_app_with_native_extension_on_android(app_env, monkeypatch):
    if not (EXTENSION_SRC / "flet_glossarion_native" / "__init__.py").is_file():
        pytest.skip("flet_glossarion_native extension sources not present")
    monkeypatch.syspath_prepend(str(EXTENSION_SRC))

    async def scenario():
        _m, conn, session, page, app = await _start("android")
        try:
            native = app.native.native
            assert type(native).__name__ == "GlossarionNative" and not app.native.is_stub
            await asyncio.sleep(0.2)
            assert "get_initial_shared" in conn.invoked()
            # an iOS-style cold-start launch link delivered as a shared item is routed once
            await session.dispatch_event(
                native._i,
                "share",
                {"items": [{"id": "L1", "kind": "url", "source": "launch", "text": "glossarion://app/__selftest__?suite=smoke"}]},
            )
            assert await _wait(lambda: app.state.selftest_result.value is not None)
            app.open_device_checks()
            spike = app.spike
            await spike.fgs_start()
            await asyncio.sleep(1.5)
            await session.dispatch_event(native._i, "foreground", {"type": "button", "button_id": "stop"})
            assert await _wait(lambda: spike.cards["fgs"].status != "running")
            assert spike.cards["fgs"].status == "pass", spike.cards["fgs"].result.value
            assert "stop_job_service" in conn.invoked()
            route_before = page.route
            await session.dispatch_event(
                native._i, "share", {"items": [{"id": "1", "kind": "file", "path": "/x/book.epub", "name": "book.epub"}]}
            )
            await asyncio.sleep(2.8)
            assert spike.cards["share"].status == "pass", spike.cards["share"].result.value
            assert page.route == route_before
        finally:
            await _stop(app)

    asyncio.run(scenario())


@needs_flet
def test_launch_env_selftest_runs_once(app_env, monkeypatch, capsys):
    """iOS simulator smoke: ``simctl openurl`` stops at an "Open in ...?" alert, so CI launches with
    SIMCTL_CHILD_GLOSSARION_CI_SELFTEST=smoke; the app runs that suite after the warm import, once,
    through the same route as the deep link, and announces the start (fail-fast marker)."""
    import os

    monkeypatch.setenv(rb.CI_SELFTEST_ENV, "smoke")

    async def scenario():
        _m, conn, session, page, app = await _start("ios")
        try:
            assert await _wait(lambda: app.state.selftest_result.value is not None, timeout=20)
            result = app.state.selftest_result.value
            assert result["ok"] and result["suite"] == "smoke" and result["source"] == "route/launch-env"
            assert rb.CI_SELFTEST_ENV not in os.environ  # consumed: never re-run, never leaks into job envs
            await asyncio.sleep(0.3)
            assert app.selftest.runs == 1
            assert _routes(page) == ["/"]
        finally:
            await _stop(app)

    asyncio.run(scenario())
    lines = capsys.readouterr().err.splitlines()
    assert any(line.startswith(rb.MARKER_SELFTEST_START + " ") and "launch-env" in line for line in lines)


# ==========================================================================
# Import hygiene
# ==========================================================================


def test_pure_modules_do_not_import_flet():
    import subprocess

    script = (
        "import sys, json; sys.path.insert(0, %r)\n"
        "import glossarion_mobile.state.store, glossarion_mobile.state.app_state, glossarion_mobile.state.chat_index\n"
        "import glossarion_mobile.services.dispatcher, glossarion_mobile.services.diagnostics\n"
        "import glossarion_mobile.services.secure_keys\n"
        "import glossarion_mobile.ui.router, glossarion_mobile.ui.tokens, glossarion_mobile.ui.responsive\n"
        "import glossarion_mobile.ui.chat.send_state, glossarion_mobile.ui.chat.output_modes\n"
        "print(json.dumps(sorted(m for m in ('flet', 'PySide6', 'translator_gui', 'TransateKRtoEN') if m in sys.modules)))\n"
    ) % str(APP_DIR)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "[]"


def test_app_package_parses_as_python_310():
    package = APP_DIR / "glossarion_mobile"
    files = sorted(package.rglob("*.py")) + [APP_DIR / "main.py"]
    assert len(files) > 20
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10))
