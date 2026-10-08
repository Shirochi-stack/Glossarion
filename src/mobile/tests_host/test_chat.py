"""Host tests for the U3 chat (Direct Text) on mobile.

Run from src/mobile with the 3.13 venv (Flet installed):
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_chat.py

Persistence is exercised against the desktop's own code: since U3 the Direct Text
dialog's chat persistence lives in ``direct_text_store.ChatStoreMixin`` (the dialog
inherits it; tests/test_direct_text_core.py proves the move verbatim and the GUI-free
``ChatStore`` equal to the real dialog), so ``DesktopDialogStore`` is ``ChatStore``
with the dialog's constructor arguments and the adapter is proven to read and write
the exact desktop ``direct_text_chats.json`` v2 format.

The chat calls the shared rules (glossary policy, attachment size, rendered-card
window, auto-title, timestamp, tokens, request cards, finishing; since U7 also the
rename rule, the glossary override writes / read and the manual-glossary sniffing, which
moved out of the dialog's Qt handlers into ``direct_text_store``); the tests check it
delegates. The few rules still inline in the dialog (the repaint cadence, the other
settings reads of its ``__init__``, the history window, the welcome cards) are compared
with the dialog source, so they cannot drift.
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import types
from datetime import datetime, timedelta
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
TRANSLATOR_GUI = SRC_DIR / "translator_gui.py"

for entry in (str(APP_DIR), str(SRC_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from glossarion_mobile.state.chat_store_adapter import (  # noqa: E402
    SIDECAR_NAME,
    ChatStoreAdapter,
    ChatStoreBinding,
    message_fingerprints,
    message_id,
)
from glossarion_mobile.ui.chat import direct_text_rules as rules  # noqa: E402
from glossarion_mobile.ui.chat.direct_text_rules import DirectTextSettings, ManualGlossarySource  # noqa: E402
from glossarion_mobile.ui.chat.job_binding import CardPhase, EtaEstimator, JobsAdapter, progress_line  # noqa: E402
from glossarion_mobile.ui.chat.run_controller import STATUS, ChatRuns  # noqa: E402
from glossarion_mobile.ui.chat.run_request import (  # noqa: E402
    DirectTextRun,
    job_params,
    prepare_direct_text_run,
    run_options_dict,
)
from glossarion_mobile.ui.chat.stream_bridge import RunStream  # noqa: E402
from glossarion_mobile.ui.chat.transcript_model import build_items, slide_window, tail_window  # noqa: E402

_TB_SPEC = importlib.util.spec_from_file_location("_glossarion_tb_helpers_chat", Path(__file__).with_name("test_bootstrap.py"))
_TB = importlib.util.module_from_spec(_TB_SPEC)
_TB_SPEC.loader.exec_module(_TB)
storage = _TB.storage
app_env = _TB.app_env


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed")
needs_desktop_source = pytest.mark.skipif(not TRANSLATOR_GUI.is_file(), reason="src/translator_gui.py not present")


def _backend_error() -> str:
    """The chat runs on the shared Direct Text code; a venv without the backend dependencies
    (the flet-only src/mobile/.venv lacks bs4 & co.) cannot import it."""
    try:
        import direct_text_stream  # noqa: F401  (imports direct_text_store and the pipeline)
    except ImportError as exc:
        return f"the shared backend is not importable here ({exc}); use the project venv (uv sync)"
    return ""


_BACKEND_ERROR = _backend_error()
pytestmark = pytest.mark.skipif(bool(_BACKEND_ERROR), reason=_BACKEND_ERROR or "backend importable")


# ==========================================================================
# The desktop Direct Text code: the dialog + the shared mixins it inherits
# ==========================================================================

#: (file, class) whose members make up the desktop Direct Text dialog since U3.
DESKTOP_PARTS = (
    ("translator_gui.py", "_InputOutputDialog"),
    ("direct_text_store.py", "ChatStoreMixin"),
    ("direct_text_stream.py", "DirectTextStreamMixin"),
)

_SOURCE_CACHE: dict = {}


def _desktop_source():
    """``methods``: name -> (source lines, node), dialog first; ``text``: all three files."""
    if "methods" not in _SOURCE_CACHE:
        methods: dict = {}
        consts: list = []
        texts = []
        tree_gui = None
        for file_name, class_name in DESKTOP_PARTS:
            text = (SRC_DIR / file_name).read_text(encoding="utf-8-sig")
            texts.append(text)
            tree = ast.parse(text)
            if file_name == "translator_gui.py":
                tree_gui = tree
            lines = text.splitlines()
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
            for node in cls.body:
                if isinstance(node, ast.FunctionDef):
                    methods.setdefault(node.name, (lines, node))
                elif isinstance(node, ast.Assign):
                    consts.append((lines, node))
        _SOURCE_CACHE.update(methods=methods, consts=consts, text="\n".join(texts), tree=tree_gui)
    return _SOURCE_CACHE


def _node_lines(lines, node) -> str:
    start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno])
    return "\n".join(lines[start - 1: node.end_lineno])


def _method_source(name: str) -> str:
    lines, node = _desktop_source()["methods"][name]
    return _node_lines(lines, node)


def _gui_function_source(name: str) -> str:
    text = (SRC_DIR / "translator_gui.py").read_text(encoding="utf-8-sig")
    node = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == name)
    return _node_lines(text.splitlines(), node)


def make_desktop_store_class():
    """``DesktopDialogStore``: the dialog's chat persistence (``direct_text_store.ChatStore``) built
    with the dialog's own state: ``_chat_history_path`` and ``_saved_env['OUTPUT_DIRECTORY']``."""
    import direct_text_store

    class DesktopDialogStore(direct_text_store.ChatStore):
        def __init__(self, history_path, output_root):
            super().__init__(history_path, output_root=output_root)

    return DesktopDialogStore


@pytest.fixture
def desktop_store_cls():
    if not (SRC_DIR / "direct_text_store.py").is_file():
        pytest.skip("src/direct_text_store.py not present")
    return make_desktop_store_class()


def _desktop_history(root: Path) -> Path:
    """A desktop-format history with externalised bodies, an attachment turn and its cards."""
    history = root / "direct_text_chats.json"
    folder = root / "Output" / "Direct Text" / "My novel - 20261001_101010_abcdef12"
    messages_dir = folder / "Chat Messages"
    messages_dir.mkdir(parents=True)
    (messages_dir / "000002-response.md").write_text("안녕 → **Hello**", encoding="utf-8")
    (messages_dir / "000002-thinking.md").write_text("thinking about it", encoding="utf-8")
    (folder / "Attachments" / "book").mkdir(parents=True)
    book = root / "book.epub"
    book.write_bytes(b"PK\x03\x04fake epub")
    rel = lambda p: os.path.relpath(p, root).replace("\\", "/")  # noqa: E731
    payload = {
        "version": 2,
        "current_chat_id": 2,
        "sessions": [
            {
                "id": 2,
                "title": "My novel",
                "messages": [
                    ["user", "안녕"],
                    ["assistant", "", "", "Token summary  ·  Thinking 3  ·  Text 2", str(folder), "Request 1",
                     {"content_path": rel(messages_dir / "000002-response.md"),
                      "thinking_path": rel(messages_dir / "000002-thinking.md"),
                      "content_chars": 15, "thinking_chars": 17, "created_at": "2026-10-01T10:10:10+09:00"}],
                    ["user_file", "book.epub", str(book), 15, "keep honorifics", "system"],
                    ["assistant", "Chapter one text", "", "Processing", str(folder / "Attachments" / "book"),
                     "Chapter 1 (chunk 1/1) · ch001.xhtml · Request 2", {"created_at": "2026-10-01T10:11:00+09:00"}],
                    ["assistant", "## Extraction report\n- Chapter payloads: 1/1 ready", "", "Processing",
                     str(folder / "Attachments" / "book"), "Extraction report", {"created_at": "2026-10-01T10:12:00+09:00"}],
                ],
                "draft": "half-typed",
                "attachment": None,
                "output_folder": str(folder),
                "output_folder_name": folder.name,
                "next_output_index": 2,
                "expanded": [1],
            },
            {"id": 5, "title": "New chat", "messages": [], "draft": "", "attachment": None, "output_folder": "",
             "output_folder_name": "", "next_output_index": 1, "expanded": []},
        ],
    }
    history.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return history


def _adapter(desktop_store_cls, root: Path, **kwargs) -> ChatStoreAdapter:
    history = root / "direct_text_chats.json"
    store = desktop_store_cls(str(history), str(root / "Output"))
    binding = ChatStoreBinding(store, history_path=str(history))
    adapter = ChatStoreAdapter(binding, history_path=str(history), **kwargs)
    assert adapter.load(), adapter.load_error
    return adapter


# ==========================================================================
# Persistence: desktop-format round trip
# ==========================================================================


def test_store_round_trip_with_desktop_history(desktop_store_cls, tmp_path):
    history = _desktop_history(tmp_path)
    original = json.loads(history.read_text(encoding="utf-8"))
    clock = [datetime(2026, 10, 1, 12, 0).timestamp()]
    adapter = _adapter(desktop_store_cls, tmp_path, clock=lambda: clock[0], save_delay=0.01)

    # drawer index from the real history (InMemoryChatIndex interface)
    assert [c.cid for c in adapter.all()] == ["2", "5"]
    assert adapter.current_cid() == "2" and adapter.get("2").title == "My novel"
    assert adapter.get("2").attachments == 1  # Attachments/book workspace
    labels = [label for label, _rows in adapter.recents(now=clock[0])]
    assert labels and labels[0] == "Today"
    assert [c.cid for c in adapter.search("honorifics")] == ["2"]

    # lazy bodies come from Chat Messages/ (relative references resolve against the history file)
    assert adapter.message_text("2", 1, "content") == "안녕 → **Hello**"
    assert adapter.message_text("2", 1, "thinking") == "thinking about it"
    assert adapter.draft("2") == "half-typed" and adapter.expanded("2") == {1}

    # mobile-only data goes to the sidecar, never into the v2 file
    assert adapter.set_pinned("2", True)
    adapter.set_override("2", "glossary_override_mode", "manual")
    adapter.flush()
    saved = json.loads(history.read_text(encoding="utf-8"))
    assert saved["version"] == 2 and saved["current_chat_id"] == 2
    allowed = {"id", "title", "messages", "draft", "attachment", "output_folder", "output_folder_name",
               "next_output_index", "expanded"}
    assert all(set(s) <= allowed for s in saved["sessions"])
    # byte-for-byte what the desktop dialog writes for the same history (inline bodies externalised)
    reference_path = tmp_path / "desktop_reference.json"
    reference_path.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")
    reference = desktop_store_cls(str(reference_path), str(tmp_path / "Output"))
    sessions, current = reference._load_chat_history()
    reference._chat_sessions = sessions
    reference._current_chat_index = next(i for i, s in enumerate(sessions) if s["id"] == current)
    reference._save_chat_history()
    desktop_written = json.loads((tmp_path / "desktop_reference.json").read_text(encoding="utf-8"))
    assert saved["sessions"] == desktop_written["sessions"]
    chapter = saved["sessions"][0]["messages"][3]
    assert chapter[1] == "" and chapter[6]["content_path"].endswith("Chat Messages/000004-response.md")
    sidecar = json.loads((tmp_path / SIDECAR_NAME).read_text(encoding="utf-8"))
    assert sidecar["chats"]["2"]["pinned"] is True
    assert sidecar["chats"]["2"]["overrides"] == {"glossary_override_mode": "manual"}

    # a send: user turn + auto-title on a fresh chat, inline assistant body externalised by the desktop code
    assert adapter.new_chat() == "6"  # the current chat has content -> a new session (desktop _new_chat)
    assert adapter.select("5") and adapter.current_cid() == "5"
    cid = adapter.new_chat()
    assert cid == "5" and len(adapter.all()) == 3  # an empty current chat is reused (desktop _new_chat)
    index = adapter.record_user_turn(cid, ("user", "Translate this please, it is long enough to cut"), "Translate this please, it is long enough to cut")
    assert index == 0 and adapter.get("5").title == "Translate this please, it is long enough " + "…"  # [:41] + "…"
    adapter.append_messages(cid, [("assistant", "Translated!", "hmm", "Processing", "", "Request 1",
                                   {"created_at": "2026-10-01T12:00:00+09:00"})])
    adapter.flush()
    reloaded = desktop_store_cls(str(history), str(tmp_path / "Output"))
    sessions, current = reloaded._load_chat_history()
    assert current == 5 and [s["id"] for s in sessions] == [2, 5, 6]
    fresh = next(s for s in sessions if s["id"] == 5)
    stored = fresh["messages"][1]
    assert stored[1] == "" and stored[6]["content_path"].endswith("Chat Messages/000002-response.md")
    assert stored[6]["content_chars"] == len("Translated!")
    body = tmp_path / "Output" / "Direct Text"
    assert any(p.name == "000002-response.md" for p in body.rglob("*.md"))
    assert adapter.message_text("5", 1, "content") == "Translated!"

    # a second adapter (app restart) sees the same chats, pins and overrides
    again = _adapter(desktop_store_cls, tmp_path)
    assert again.get("2").pinned and again.overrides("2") == {"glossary_override_mode": "manual"}
    assert again.get("5").title.startswith("Translate this please")

    # fingerprints / opaque ids (Appendix B)
    fps = again.fingerprints("2")
    assert fps[0].startswith("u:") and fps[1] == "a:2026-10-01T10:10:10+09:00:000002-response.md"
    assert fps[2].startswith("f:")
    mid = message_id(fps[3])
    assert len(mid) == 12 and again.index_for_mid("2", mid) == 3 and again.mid_for_index("2", 3) == mid
    adapter.close()
    again.close()


def test_store_delete_rename_and_validated_folder(desktop_store_cls, tmp_path):
    _desktop_history(tmp_path)
    adapter = _adapter(desktop_store_cls, tmp_path, save_delay=0.01)
    folder = Path(adapter.output_folder("2"))
    assert folder.is_dir()
    title, body = adapter.delete_notice("2")
    assert title == "Delete chat?" and "This permanently removes the conversation" in body and str(folder) in body
    assert adapter.rename("2", "  Renamed   chat  " + "x" * 200)
    assert adapter.get("2").title.startswith("Renamed chat") and len(adapter.get("2").title) == 120
    ok, error = adapter.delete("2")
    assert ok, error
    assert not folder.exists()  # validated rmtree of the direct child of Direct Text/
    assert [c.cid for c in adapter.all()] == ["5"] and adapter.current_cid() == "5"
    # a hand-edited folder outside Direct Text/ is refused (desktop _validated_chat_output_folder)
    victim = tmp_path / "precious"
    victim.mkdir()
    cid = adapter.new_chat()
    adapter.record_user_turn(cid, ("user", "x"), "x")
    adapter.session(cid)["output_folder"] = str(victim)
    ok, error = adapter.delete(cid)
    assert not ok and "could not be safely removed" in error and victim.exists()
    adapter.close()


def test_real_desktop_history_loads_if_present(desktop_store_cls, tmp_path):
    real = SRC_DIR / "direct_text_chats.json"
    if not real.is_file():
        pytest.skip("no desktop direct_text_chats.json in src/")
    copy = tmp_path / "direct_text_chats.json"
    copy.write_bytes(real.read_bytes())
    adapter = _adapter(desktop_store_cls, tmp_path)
    store = desktop_store_cls(str(copy), str(tmp_path / "Output"))
    sessions, current = store._load_chat_history()
    assert [c.cid for c in adapter.all()] == [str(s["id"]) for s in sessions]
    assert all(isinstance(m, tuple) for s in adapter.sessions for m in s["messages"])
    adapter.close()


def test_store_round_trip_with_shared_direct_text_store(tmp_path, monkeypatch):
    """The production binding (no store passed in) writes what the dialog writes for the same history."""
    history = _desktop_history(tmp_path)
    original = history.read_text(encoding="utf-8")
    monkeypatch.setenv("GLOSSARION_DIRECT_TEXT_HISTORY", str(history))
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(tmp_path / "Output"))
    adapter = ChatStoreAdapter(ChatStoreBinding(history_path=str(history), output_root=str(tmp_path / "Output")))
    assert adapter.load(), adapter.load_error
    assert [c.cid for c in adapter.all()] == ["2", "5"]
    assert adapter.message_text("2", 1, "content") == "안녕 → **Hello**"
    adapter.flush()
    saved = json.loads(history.read_text(encoding="utf-8"))
    # the dialog's save of the same history (it externalises inline bodies into Chat Messages/)
    reference_path = tmp_path / "desktop_reference.json"
    reference_path.write_text(original, encoding="utf-8")
    reference = make_desktop_store_class()(str(reference_path), str(tmp_path / "Output"))
    sessions, current = reference._load_chat_history()
    reference._chat_sessions = sessions
    reference._current_chat_index = next(i for i, s in enumerate(sessions) if s["id"] == current)
    reference._save_chat_history()
    assert saved["sessions"] == json.loads(reference_path.read_text(encoding="utf-8"))["sessions"]
    adapter.close()


# ==========================================================================
# Rules: the chat calls the shared desktop code; dialog-inline mirrors cannot drift
# ==========================================================================


def test_rules_call_the_shared_desktop_code():
    import direct_text_store as dts
    import direct_text_stream as dtm

    for mode in ("none", "attachments_only", "no_glossary", "manual", "", None, "Bogus"):
        for has_attachment in (True, False, None, 1):
            assert rules.force_no_glossary_for_mode(mode, has_attachment) == dts.ChatStoreMixin._force_no_glossary_for_mode(
                mode, has_attachment)
    for value in (0, 1, 1023, 1024, 1536, 1048575, 1048576, 5_000_000, -3, "x", None, "2048"):
        assert rules.format_attachment_size(value) == dts.ChatStoreMixin._format_attachment_size(value)
    for value in (0, 3, 4, 20, 200, 201, "50", None, "x", 7.9):
        assert rules.normalize_rendered_card_limit(value) == dts.ChatStoreMixin._normalize_rendered_card_limit(value)
    assert (rules.DEFAULT_RENDERED_CARD_LIMIT, rules.MIN_RENDERED_CARD_LIMIT, rules.MAX_RENDERED_CARD_LIMIT) == (
        dts.ChatStoreMixin._DEFAULT_RENDERED_CARD_LIMIT, dts.ChatStoreMixin._MIN_RENDERED_CARD_LIMIT,
        dts.ChatStoreMixin._MAX_RENDERED_CARD_LIMIT)
    for total in (0, 1, 5, 20, 21, 100):
        for limit_value in (1, 4, 20, 50):
            for focus in (None, 0, 3, 10, 99, 500):
                assert rules.history_window_bounds(total, limit_value, focus) == tuple(
                    dts.ChatStoreMixin._history_window_bounds(total, limit_value, focus))
    for ext in (".txt", ".epub", ".pdf", ".md", ".html", ".xml", ".srt", ".vtt", ".log", ".cbz", ".png", ".heic", ".jxl",
                ".exe", ".zip", ".sdlxliff", ".mp4", ".docx"):
        expected = dts.ChatStoreMixin._is_supported_dropped_text_file(f"x{ext}") or ext in rules.MOBILE_EXTRA_ATTACHMENT_EXTENSIONS
        assert rules.is_supported_attachment(f"x{ext}") is bool(expected), ext
    now = datetime.now().astimezone()
    for created in (now.isoformat(timespec="seconds"), (now - timedelta(days=40)).isoformat(timespec="seconds"),
                    "2019-02-03T04:05:06+00:00", "garbage", "", "2026-10-01T10:10:10Z"):
        assert rules.timestamp_label(created) == dts.timestamp_label(created)
    if _has("tiktoken"):
        for text in ("hello world", "안녕하세요 " * 20, "\u200b", ""):
            assert rules.count_tokens(text, "gpt-4o") == dtm.count_tokens(text, "gpt-4o")
    # auto-title = the dialog's _title_current_chat_from_text on a store session
    store = make_desktop_store_class()(str(Path(tempfile.mkdtemp()) / "h.json"), tempfile.mkdtemp())
    for current, text in (("New chat", "hello"), ("New chat", "  a   b\nc "), ("Other", "x"), ("New chat", ""),
                          ("New chat", "y" * 41), ("New chat", "y" * 42), ("New chat", "z" * 43)):
        session = store.new_chat_session(9)
        session["title"] = current
        expected = store.title_chat_from_text(session, text)
        assert (rules.auto_title(current, text) or current) == expected


def test_dialog_rules_moved_to_the_shared_store_are_the_ones_the_chat_calls():
    """U7: ``_rename_chat``, ``_on_glossary_override_toggled``, the ``__init__`` override read and the
    "Provide Manual Glossary" ``_accept`` call ``direct_text_store`` functions; the chat calls the same."""
    if not (SRC_DIR / "translator_gui.py").is_file():
        pytest.skip("src/translator_gui.py not present")
    import direct_text_store as dts

    assert "chat_rename_title(new_title)" in _method_source("_rename_chat")
    assert "glossary_override_config_updates(mode)" in _method_source("_on_glossary_override_toggled")
    assert "configured_glossary_override_mode(" in _method_source("__init__")
    manual = _method_source("_request_direct_text_manual_glossary")
    assert "manual_glossary_source_record(" in manual and "MANUAL_GLOSSARY_EXTENSIONS" in manual
    for text in ("  Renamed   chat  " + "x" * 200, "", "a\tb", None):
        assert rules.rename_title(text) == dts.chat_rename_title(text)
    for mode in ("none", "attachments_only", "no_glossary", "manual", "Manual", "bogus", ""):
        assert rules.glossary_override_updates(mode or "none") == dts.glossary_override_config_updates(mode or "none")
        assert rules.normalize_glossary_override_mode(mode) == dts.configured_glossary_override_mode(mode)
    assert rules.GLOSSARY_OVERRIDE_MODES == dts.GLOSSARY_OVERRIDE_MODES
    assert rules.MANUAL_GLOSSARY_EXTENSIONS == dts.MANUAL_GLOSSARY_EXTENSIONS
    for content in ('{"a": "b"}', "[not json", "raw,translated\n김,Kim", "plain", "a,b"):
        record = dts.manual_glossary_source_record(content)
        assert rules.manual_glossary_source(content).as_dict() == record
    # the mirrors are gone: nothing in the chat re-implements the sniffing
    source = (APP_DIR / "glossarion_mobile" / "ui" / "chat" / "direct_text_rules.py").read_text(encoding="utf-8")
    assert "nonempty_lines" not in source and "_sniff_glossary_extension" not in source


def test_dialog_inline_rules_the_chat_mirrors_match_the_dialog():
    """Rules still inline in the dialog's Qt handlers (no shared function to call yet)."""
    if not (SRC_DIR / "translator_gui.py").is_file():
        pytest.skip("src/translator_gui.py not present")
    render = _method_source("_schedule_stream_render")  # the dialog's Qt override (the mixin's is a hook)
    assert "interval = min(900, 280 + active_characters // 350)" in render and "interval = max(interval, 450)" in render
    for chars in (0, 1, 349, 350, 10_000, 1_000_000):
        assert rules.stream_render_interval_ms(chars) == min(900, 280 + chars // 350)
        assert rules.stream_render_interval_ms(chars, True) == max(min(900, 280 + chars // 350), 450)
    init = _method_source("__init__")
    for snippet in ("translator.config.get('direct_text_force_simple_mode', True)",
                    "'direct_text_force_multipass_off', legacy_simple_mode",
                    "'direct_text_skip_system_prompt_profile', False",
                    "translator.config.get('direct_text_attachment_prompt_role', 'user')"):
        assert snippet in init, snippet
    text = _desktop_source()["text"]
    for message in ("*No translated output was produced for this message.*",
                    "*No translated output was emitted for this request.*",
                    "**Translation could not be started.**"):
        assert message in text

    # rendered-card window (the dialog's _reset_history_window over the shared size helpers)
    wns: dict = {}
    exec("class D:\n    _DEFAULT_RENDERED_CARD_LIMIT = 20\n    _MIN_RENDERED_CARD_LIMIT = 4\n"
         "    _MAX_RENDERED_CARD_LIMIT = 200\n" + _method_source("_normalize_rendered_card_limit") + "\n"
         + _method_source("_assistant_message_char_count") + "\n" + _method_source("_assistant_storage_for") + "\n"
         + _method_source("_reset_history_window"), wns)
    import random

    rng = random.Random(7)
    for _ in range(60):
        messages = []
        for _i in range(rng.randint(0, 40)):
            if rng.random() < 0.4:
                messages.append(("user", "u" * rng.randint(0, 9000)))
            else:
                messages.append(("assistant", "", "t" * rng.randint(0, 5000), "", "", "",
                                 {"content_chars": rng.randint(0, 60000), "thinking_chars": rng.randint(0, 9000)}))
        expanded = {rng.randrange(len(messages))} if messages and rng.random() < 0.5 else set()
        limit = rng.choice([4, 6, 20, 40])
        host = wns["D"]()
        host._chat_messages = messages
        host._history_card_limit = limit
        host._history_character_budget = 120000
        host._expanded_processing_messages = expanded
        wns["D"]._reset_history_window(host)
        assert tail_window(messages, limit, expanded) == (host._history_visible_start, host._history_visible_end)

    # Welcome cards = the desktop first-run glossary mode page
    welcome = _gui_function_source("_show_glossary_mode_welcome")
    tree = ast.parse("def f():\n" + "\n".join("    " + line for line in welcome.splitlines()))
    mode_data = next(ast.literal_eval(n.value) for n in ast.walk(tree)
                     if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "mode_data")
    from glossarion_mobile.ui.screens.welcome_flow import GLOSSARY_MODE_CARDS

    assert [(m["value"], m["emoji"], m["title"], m["subtitle"], tuple(m["features"]), m["rec"]) for m in mode_data] == list(
        GLOSSARY_MODE_CARDS
    )
    assert "selected_mode = ['balanced']" in welcome
    assert "self.config['enable_auto_glossary'] = (mode_val not in ('off', 'off_fuzzy_automap', 'off_no_automap', 'no_glossary'))" in welcome
    assert "if mode_val not in ('off', 'off_no_automap', 'no_glossary'):" in welcome
    assert "self.config['glossary_mode_dialog_shown'] = True" in welcome


def _fed_stream(lines, *, attachment=False, request_number=None, model=None, thread="Thread-2 (api_call)"):
    import direct_text_stream

    stream = direct_text_stream.make_stream(source_is_attachment=attachment, request_number=request_number, model=model)
    for line in lines:
        stream.feed(line, thread)
    return stream


#: Log lines that make the shared classifier open one streaming request card with text.
CARD_LINES = ("🚀 [Thread-2 (api_call)] Sending API call now", "📡 [Thread-2 (api_call)] Text streaming...",
              "Hello there")


def test_run_stream_reads_the_job_stream_and_the_shared_card_messages():
    from glossarion_mobile.ui.chat.stream_bridge import RunStream, segment_message, segment_processing_label
    import direct_text_stream

    holder = {"stream": None}
    run_stream = RunStream(provider=lambda: holder["stream"])
    assert run_stream.segments() == [] and not run_stream.drain()  # queued: no job stream yet
    holder["stream"] = _fed_stream(CARD_LINES, request_number=3)
    assert run_stream.drain() and run_stream.version == 1
    assert not run_stream.drain()  # nothing new
    segments = run_stream.segments()
    assert segments and segments[0]["label"].startswith("Request") and "Hello there" in segments[0]["content"]
    assert run_stream.model() is holder["stream"]
    for segment in segments:
        assert segment_message(segment, "/out") == tuple(direct_text_stream.request_segment_message(segment, "/out"))
        assert segment_processing_label(segment) == direct_text_stream.request_segment_message(segment)[3]
    assert 280 <= run_stream.render_interval_ms() <= 900
    assert RunStream(auto_scroll_disabled=True).render_interval_ms() == 450


def test_manual_glossary_source_and_settings_migrations(tmp_path):
    assert rules.manual_glossary_source("   ") is None
    assert rules.manual_glossary_source('{"a": "b"}').extension == ".json"
    assert rules.manual_glossary_source("[not json").extension == ".txt"
    assert rules.manual_glossary_source("raw,translated\n김,Kim").extension == ".csv"
    path = tmp_path / "g.csv"
    path.write_text("a,b\nc,d", encoding="utf-8")
    source = rules.manual_glossary_source("a,b\nc,d", source_path=str(path), source_text="a,b\nc,d", source_extension=".csv")
    assert source.kind == "path" and source.path == str(path)
    edited = rules.manual_glossary_source("a,b\nc,e", source_path=str(path), source_text="a,b\nc,d")
    assert edited.kind == "content" and edited.extension == ".csv"

    def getter(values):
        return lambda key, default=None: values.get(key, default)

    fresh = DirectTextSettings.from_config(getter({}))
    assert (fresh.force_multipass_off, fresh.glossary_override_mode, fresh.attachment_prompt_role,
            fresh.skip_prompt_profile, fresh.rendered_card_limit, fresh.output_mode) == (True, "attachments_only", "user", False, 20, "text")
    legacy = DirectTextSettings.from_config(getter({
        "direct_text_force_simple_mode": False, "direct_text_skip_user_prompt_profile": True,
        "direct_text_glossary_override_mode": "bogus", "direct_text_attachment_prompt_role": "ROBOT",
        "direct_text_rendered_card_limit": 999, "output_mode": "refine",
    }))
    assert (legacy.force_multipass_off, legacy.skip_prompt_profile, legacy.glossary_override_mode,
            legacy.attachment_prompt_role, legacy.rendered_card_limit, legacy.output_mode) == (
        False, True, "attachments_only", "user", 200, "refinement")
    chat = fresh.with_overrides({"glossary_override_mode": "manual", "disable_thinking": True, "skip_plan": True,
                                 "attachment_prompt_role": "system", "model": "ignored-here"})
    assert (chat.glossary_override_mode, chat.disable_thinking, chat.skip_plan, chat.attachment_prompt_role) == (
        "manual", True, True, "system")
    assert rules.glossary_override_updates("no_glossary") == {
        "direct_text_glossary_override_mode": "no_glossary", "direct_text_force_no_glossary": True,
        "direct_text_manual_glossary": False}


# ==========================================================================
# DirectTextRunOptions mapping per chat setting
# ==========================================================================


@pytest.mark.parametrize("policy, attachment, manual, expected_force_no", [
    ("attachments_only", False, False, True),
    ("attachments_only", True, False, False),
    ("none", False, False, False),
    ("none", True, False, False),
    ("no_glossary", True, False, True),
    ("no_glossary", False, False, True),
    ("manual", False, True, False),
    ("manual", True, True, False),
])
def test_run_options_mapping_per_chat_setting(tmp_path, policy, attachment, manual, expected_force_no):
    settings = DirectTextSettings().with_overrides({
        "glossary_override_mode": policy, "force_multipass_off": False, "disable_thinking": True,
        "skip_prompt_profile": True, "attachment_prompt_role": "assistant",
    })
    record = None
    if attachment:
        book = tmp_path / "book.epub"
        book.write_bytes(b"epub")
        record = {"path": str(book), "name": "book.epub", "extension": ".epub", "size": 4}
    source = ManualGlossarySource("content", content="raw,translated\n김,Kim", extension=".csv") if manual else None
    run = prepare_direct_text_run(text="  keep honorifics  ", attachment=record, output_mode="vision",
                                  attachment_prompt_role=settings.attachment_prompt_role, manual_glossary=source,
                                  temp_dir=str(tmp_path))
    options = run_options_dict(run, settings)
    assert options["force_no_glossary"] is expected_force_no
    assert options["force_multipass_off"] is False and options["skip_thinking"] is True
    assert options["skip_prompt_profile"] is True and options["force_stream_all"] is True
    assert options["output_mode"] == "vision" and options["attachment_prompt_role"] == "assistant"
    assert options["selected_files"] == [run.source_path]
    assert options["archive_conversion_dir"] == os.path.join(run.temp_root, "_archive_input")
    assert Path(run.temp_root).name.startswith("glossarion_input_output_")
    assert Path(run.temp_root).parent == tmp_path  # the app's resumable run folder, not the OS temp dir
    if attachment:
        assert run.source_path == str(tmp_path / "book.epub") and run.is_attachment  # passed straight through
        assert options["attachment_prompt"] == "keep honorifics"
        assert run.expected_output.endswith(os.path.join("book", "book_translated.txt"))
    else:
        assert Path(run.source_path).read_text(encoding="utf-8") == "keep honorifics"
        assert options["attachment_prompt"] == ""
    if manual:
        assert options["manual_glossary_path"].endswith("direct_text_manual_glossary.csv")
        assert Path(options["manual_glossary_path"]).read_text(encoding="utf-8").startswith("raw,translated")
    params = job_params(chat_id=3, user_index=4, run=run, settings=settings,
                        overrides={"model": "gpt-5", "target_language": "Japanese", "profile": " "})
    assert json.loads(json.dumps(params)) == params  # checkpointable
    assert "env" not in params  # the job applies direct_text_store.apply_direct_text_run_environment
    assert params["config_overrides"] == {"model": "gpt-5", "output_language": "Japanese"}
    assert DirectTextRun.from_dict(params["run"]) == run

    from headless_owner import DirectTextRunOptions

    owner = types.SimpleNamespace()
    DirectTextRunOptions(**params["options"]).apply_to(owner)
    assert owner._input_output_run_active is True and owner._direct_text_force_no_glossary is expected_force_no
    assert owner.enable_image_translation_var is True and owner._direct_text_output_mode == "vision"
    assert owner._direct_text_use_manual_glossary is bool(manual)


def test_markup_and_subtitle_attachments_are_adapted(tmp_path):
    page = tmp_path / "page.html"
    page.write_text("\ufeff<p>안녕</p>", encoding="utf-8")
    run = prepare_direct_text_run(text="", attachment={"path": str(page), "name": "page.html"}, temp_dir=str(tmp_path))
    assert run.source_path == os.path.join(run.temp_root, "page.txt")
    assert Path(run.source_path).read_text(encoding="utf-8") == "<p>안녕</p>"  # utf-8-sig read, like desktop
    subs = tmp_path / "ep1.srt"
    subs.write_text("1\n00:00:01,000 --> 00:00:02,000\n안녕\n", encoding="utf-8")
    run = prepare_direct_text_run(text="", attachment={"path": str(subs), "name": "ep1.srt"}, temp_dir=str(tmp_path))
    assert run.source_path == str(subs) and run.expected_output.endswith("ep1_translated.srt")
    with pytest.raises(FileNotFoundError):
        prepare_direct_text_run(text="", attachment={"path": str(tmp_path / "gone.epub")}, temp_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        prepare_direct_text_run(text="x", attachment=None, temp_dir=str(tmp_path),
                                manual_glossary=ManualGlossarySource("path", path=str(tmp_path / "nope.csv")))
    zipped = tmp_path / "pages.zip"  # a mobile extra (ZIP -> EPUB in the pipeline): handed over as-is
    zipped.write_bytes(b"PK\x03\x04")
    run = prepare_direct_text_run(text="", attachment={"path": str(zipped), "name": "pages.zip"}, temp_dir=str(tmp_path))
    assert run.source_path == str(zipped) and run.is_attachment
    assert rules.needs_plan(".epub", 0) and rules.needs_plan(".txt", 20_001) and not rules.needs_plan(".txt", 20_000)
    assert not rules.needs_plan(".epub", 0, skip_plan=True) and not rules.needs_plan(".png", 0)


# ==========================================================================
# Runs: Send/Stop transitions, approval answers, finishing (fake JobService)
# ==========================================================================


class FakeSignal:
    def __init__(self, value=None):
        self.value = value
        self.subs = []

    def subscribe(self, callback):
        self.subs.append(callback)
        return lambda: self.subs.remove(callback) if callback in self.subs else None

    def set(self, value):
        self.value = value
        for callback in list(self.subs):
            callback(value)


class FakeJobService:
    """JobService stand-in: signals, submit/stop/answer records and one shared
    ``direct_text_stream.DirectTextStream`` per started job (``request_stream``), fed by ``line()``."""

    def __init__(self):
        self.snapshot = FakeSignal(None)
        self.queue = FakeSignal([])
        self.submitted = []
        self.stops = []
        self.answers = []
        self.cancelled = []
        self.listeners = []
        self.streams = {}
        self.current = None

    def request_stream(self, job_id):
        return self.streams.get(job_id)

    async def submit(self, spec):
        self.submitted.append(spec)
        return f"job{len(self.submitted)}"

    def request_stop(self, force=False):
        self.stops.append(force)

    def answer(self, question_id, value):
        self.answers.append((question_id, value))

    def cancel_queued(self, job_id):
        self.cancelled.append(job_id)

    def add_log_listener(self, callback):
        self.listeners.append(callback)
        return lambda: self.listeners.remove(callback)

    def line(self, message, thread="Thread-2 (api_call)"):
        stream = self.streams.get(self.current)
        if stream is not None:
            stream.feed(message, thread)

    def publish(self, state, *, index=-1, question=None, progress=None, in_flight=0, last_line=""):
        spec = self.submitted[index]
        job_id = f"job{len(self.submitted) + index + 1 if index < 0 else index + 1}"
        if state not in ("QUEUED",) and job_id not in self.streams:
            import direct_text_stream

            params = spec.params
            self.streams[job_id] = direct_text_stream.make_stream(
                source_is_attachment=bool(params.get("is_attachment")), request_number=params.get("request_number"),
                model=params.get("model"))
        self.current = job_id
        snap = types.SimpleNamespace(
            id=job_id, spec=spec, state=state, question=question, progress=progress, in_flight=in_flight,
            last_line=last_line, started=None, result={},
        )
        self.snapshot.set(snap)
        return snap


def _runs(desktop_store_cls, tmp_path, jobs):
    _desktop_history(tmp_path)
    adapter = _adapter(desktop_store_cls, tmp_path, save_delay=0.01)
    runs = ChatRuns(adapter, JobsAdapter(jobs), temp_dir=str(tmp_path))
    runs.attach()
    return adapter, runs


def _finish_all(runs, timeout=10):
    for thread in list(runs.finish_threads):
        thread.join(timeout)


def test_send_stop_transitions_and_finish(desktop_store_cls, tmp_path):
    jobs = FakeJobService()
    adapter, runs = _runs(desktop_store_cls, tmp_path, jobs)
    settings = DirectTextSettings()
    run = asyncio.run(runs.send("5", text="안녕하세요", attachment=None, settings=settings, output_mode="text"))
    spec = jobs.submitted[0]
    assert spec.kind == "direct_text" and spec.params["chat_id"] == 5 and spec.params["user_index"] == 0
    assert spec.params["options"]["force_no_glossary"] is True  # attachments_only + typed text
    assert spec.inputs == (run.run.source_path,) and spec.params["input_path"] == run.run.source_path
    assert spec.params["output_root"] == run.run.temp_root and spec.origin == {"type": "chat", "cid": "5", "label": "Chat · 안녕하세요"}
    assert adapter.messages("5")[0] == ("user", "안녕하세요") and adapter.get("5").title == "안녕하세요"
    assert runs.own_job_state("5") == "RUNNING" and adapter.get("5").running
    on_disk = json.loads((tmp_path / "direct_text_chats.json").read_text(encoding="utf-8"))
    assert any(s["id"] == 5 and s["messages"] == [["user", "안녕하세요"]] for s in on_disk["sessions"])  # saved at send

    jobs.publish("RUNNING", last_line="📤 sending api call now")
    assert run.state == "running" and runs.caption("5") == STATUS["translating"]
    assert spec.params["request_number"] == 1  # the chat's next request number seeds the job's stream
    for line in CARD_LINES:
        jobs.line(line)
    assert run.stream.drain() and run.stream.segments()[0]["label"] == "Request 1"
    runs.request_stop("5")
    assert jobs.stops == [False] and run.state == "stopping" and runs.caption("5") == STATUS["finishing"]
    jobs.publish("STOPPING")
    assert runs.own_job_state("5") == "STOPPING"
    runs.request_stop("5", force=True)
    assert jobs.stops == [False, True] and runs.own_job_state("5") == "FORCE_STOPPING"

    # the run wrote its translated file; DONE -> ChatStore.finish_run (the dialog's _finish_translation)
    Path(run.run.expected_output).parent.mkdir(parents=True, exist_ok=True)
    Path(run.run.expected_output).write_text("Hello there, translated", encoding="utf-8")
    jobs.publish("DONE")
    _finish_all(runs)
    assert not run.live and run.state == "stopped" and runs.own_job_state("5") is None
    messages = adapter.messages("5")
    last = len(messages) - 1
    assert messages[-1][0] == "assistant" and messages[-1][5] == "Request 1"
    assert adapter.message_text("5", last, "content") == "Hello there, translated"
    assert not adapter.get("5").running and runs.caption("5") is None  # Ready -> caption hidden
    assert os.path.isdir(run.run.temp_root)  # a stopped run keeps its run root (Resume)
    folder = Path(adapter.output_folder("5"))
    assert (folder / "Direct Text 1.txt").read_text(encoding="utf-8") == "Hello there, translated"
    adapter.close()


def test_queue_failure_and_could_not_start(desktop_store_cls, tmp_path):
    jobs = FakeJobService()
    adapter, runs = _runs(desktop_store_cls, tmp_path, jobs)
    other = types.SimpleNamespace(id="other", spec=types.SimpleNamespace(kind="translate", title="Book", params={}),
                                  state="RUNNING", question=None, progress=None, in_flight=0, last_line="")
    jobs.snapshot.value = other
    run = asyncio.run(runs.send("5", text="queued text", attachment=None, settings=DirectTextSettings(), output_mode="text"))
    assert run.state == "queued" and runs.own_job_state("5") == "RUNNING"
    runs.request_stop("5")
    assert jobs.cancelled == ["job1"] and jobs.stops == []
    _finish_all(runs)
    assert not run.live

    class Broken(FakeJobService):
        async def submit(self, spec):
            raise RuntimeError("engine offline")

    broken_runs = ChatRuns(adapter, JobsAdapter(Broken()), temp_dir=str(tmp_path))
    with pytest.raises(RuntimeError):
        asyncio.run(broken_runs.send("2", text="x", attachment=None, settings=DirectTextSettings(), output_mode="text"))
    # the body may already be externalised by the debounced save: read it like the transcript does
    last = adapter.message_text("2", len(adapter.messages("2")) - 1, "content")
    assert last.startswith("**Translation could not be started.**") and "`RuntimeError: engine offline`" in last
    assert broken_runs.caption("2") == STATUS["could_not_start"]
    adapter.close()


def test_glossary_approval_answers(desktop_store_cls, tmp_path):
    jobs = FakeJobService()
    adapter, runs = _runs(desktop_store_cls, tmp_path, jobs)
    book = tmp_path / "book.epub"
    record = {"path": str(book), "name": "book.epub", "extension": ".epub", "size": 15}
    run = asyncio.run(runs.send("5", text="", attachment=record, settings=DirectTextSettings(), output_mode="text"))
    assert run.run.is_attachment and jobs.submitted[0].params["options"]["force_no_glossary"] is False
    jobs.publish("RUNNING")
    for line in ("🚀 [Thread-2 (api_call)] Sending API call now", "📡 [Thread-2 (api_call)] Text streaming...",
                 "type,raw_name,translated_name"):
        jobs.line(line)
    before = len(adapter.messages("5"))
    question = {"id": "q1", "kind": "glossary_approval", "data": {"path": str(tmp_path / "glossary.csv")}}
    jobs.publish("RUNNING", question=question)
    assert run.awaiting_glossary and runs.caption("5") == STATUS["glossary"]
    # the gate froze the glossary request card into the chat (_commit_active_request_phase)
    gate_cards = adapter.messages("5")[before:]
    assert len(gate_cards) == 1 and gate_cards[0][0] == "assistant" and gate_cards[0][3].startswith("Token summary")
    assert run.stream.segments() == []  # the translation phase starts with fresh cards
    assert runs.answer_glossary("5", True)
    assert jobs.answers == [("q1", True)] and not run.awaiting_glossary and runs.caption("5") == STATUS["starting"]
    jobs.publish("RUNNING", question=dict(question, id="q2"))
    assert runs.answer_glossary("5", False)
    assert jobs.answers[-1] == ("q2", False) and jobs.stops == [False]  # No = reject + stop_translation
    assert not runs.answer_glossary("5", True)  # nothing pending any more
    adapter.close()


class _ChatBackend:
    """Minimal ``services.jobs.JobBackend`` stand-in (no shared pipeline needed)."""

    def __init__(self):
        import contextlib

        self.job_lock = threading.RLock()
        self.stops = []
        self._contextlib = contextlib

    def scoped_process_state(self, *, capture_stdout=None, lock, **kwargs):
        @self._contextlib.contextmanager
        def scope():
            with lock:
                yield

        return scope()

    def make_owner(self, config, *, host, **kwargs):
        return types.SimpleNamespace(stop_requested=False, graceful_stop_var=True, wait_for_chunks_var=True,
                                     config=config, host=host)

    def reset_for_new_run(self, kind="translation", **kwargs):
        return 1

    def request_stop(self, *, graceful, wait_for_chunks, force, set_stop_requested, log=None, **kwargs):
        self.stops.append("force" if force else ("graceful" if graceful else "immediate"))
        set_stop_requested()

    def progress_watcher(self, resolver, host, interval=2.0, **kwargs):
        return None

    def restore_in_progress(self, **kwargs):
        pass


def test_chat_runs_with_the_real_job_service(desktop_store_cls, tmp_path):
    """ChatRuns <-> JobService: blocking glossary question, answer, result commit, graceful stop."""
    from glossarion_mobile import job_kinds
    from glossarion_mobile.services.jobs import JobService, JobState

    seen = {}

    def fake_direct_text(ctx):
        params = ctx.params
        seen["params"] = dict(params)
        seen["config"] = dict(ctx.config)
        ctx.log("📤 sending api call now")
        out_dir = Path(params["output_root"]) / "out"
        out_dir.mkdir(exist_ok=True)
        ctx.set_output_dir(str(out_dir))
        glossary = Path(params["output_root"]) / "glossary.csv"
        glossary.write_text("type,raw_name,translated_name\ncharacter,김,Kim\n", encoding="utf-8")
        if params.get("loop"):
            while not ctx.stop_requested():
                time.sleep(0.02)
            return None
        accepted = ctx.ask("direct_text_glossary_approval", path=str(glossary), default=False)
        seen["accepted"] = accepted
        if not accepted:
            return None
        expected = Path(params["run"]["expected_output"])
        expected.parent.mkdir(parents=True, exist_ok=True)
        expected.write_text("Hello from the pipeline", encoding="utf-8")
        return None

    info = job_kinds.KindInfo(kind="direct_text", verb="Translating", icon="CHAT_BUBBLE_OUTLINE",
                              stop_kind="translation", run=fake_direct_text)

    def fake_compile(ctx):
        folder = Path(ctx.params["folder"])
        (folder / "out.epub").write_bytes(b"epub")
        return {"ok": True, "path": str(folder / "out.epub")}

    compile_info = job_kinds.KindInfo(kind="compile_epub", verb="Compiling", icon="MENU_BOOK",
                                      stop_kind="epub", run=fake_compile)
    _desktop_history(tmp_path)
    adapter = _adapter(desktop_store_cls, tmp_path, save_delay=0.01)
    backend = _ChatBackend()
    service = JobService(jobs_dir=str(tmp_path / "jobs"), backend=backend,
                         kinds=lambda kind: compile_info if str(kind) == "compile_epub" else info,
                         config_loader=lambda: {"model": "authgpt/gpt-6-luna"})
    runs = ChatRuns(adapter, JobsAdapter(service), temp_dir=str(tmp_path))
    runs.attach()
    try:
        adapter.set_override("5", "model", "gpt-5")
        settings = DirectTextSettings()
        run = asyncio.run(runs.send("5", text="안녕", attachment=None, settings=settings, output_mode="text",
                                    overrides=adapter.overrides("5")))
        deadline = time.monotonic() + TIMEOUT_S
        while not run.awaiting_glossary and time.monotonic() < deadline:
            time.sleep(0.02)
        assert run.awaiting_glossary, "the job did not block on the glossary question"
        assert runs.caption("5") == STATUS["glossary"] and runs.own_job_state("5") == "RUNNING"
        assert seen["config"]["model"] == "gpt-5"  # per-chat override applied to the job's config snapshot
        assert seen["params"]["chat_id"] == 5 and seen["params"]["options"]["force_no_glossary"] is True
        assert runs.answer_glossary("5", True)
        assert service.wait_idle(TIMEOUT_S)
        _finish_all(runs)
        assert seen["accepted"] is True and not run.live and run.state == "done"
        assert run.output_dir.endswith("out")  # JobSnapshot.output_dir followed
        # a finished run's root is cleaned like the dialog's temp root; compile then uses the
        # folder finish_run persisted the run into
        assert not os.path.isdir(run.run.temp_root) and os.path.isdir(run.output_folder)
        compile_id = asyncio.run(runs.compile("5"))
        assert compile_id and service.wait_idle(TIMEOUT_S)
        assert service.snapshot(compile_id).state is JobState.DONE and (Path(run.output_folder) / "out.epub").is_file()
        assert service.snapshot(compile_id).spec.params == {"folder": run.output_folder}
        last = adapter.messages("5")[-1]
        assert last[0] == "assistant" and adapter.message_text("5", len(adapter.messages("5")) - 1) == "Hello from the pipeline"

        # ■ No: the job is told False and a stop follows (desktop _resolve_glossary_approval + stop_translation)
        run2 = asyncio.run(runs.send("5", text="again", attachment=None, settings=settings, output_mode="text"))
        deadline = time.monotonic() + TIMEOUT_S
        while not run2.awaiting_glossary and time.monotonic() < deadline:
            time.sleep(0.02)
        assert runs.answer_glossary("5", False)
        assert service.wait_idle(TIMEOUT_S)
        _finish_all(runs)
        assert seen["accepted"] is False and backend.stops == ["graceful"]
        assert service.snapshot(run2.job_id).state is JobState.CANCELLED and run2.state == "stopped"

        # graceful stop, then force on the second tap (a job that keeps running until forced)
        def until_forced(ctx):
            ctx.log("📤 sending api call now")
            deadline = time.monotonic() + TIMEOUT_S
            while "force" not in backend.stops and time.monotonic() < deadline:
                time.sleep(0.02)
            return None

        looping = job_kinds.KindInfo(kind="direct_text", verb="Translating", icon="CHAT_BUBBLE_OUTLINE",
                                     stop_kind="translation", run=until_forced)
        service._kinds = lambda kind: looping
        run3 = asyncio.run(runs.send("5", text="third", attachment=None, settings=settings, output_mode="text"))
        deadline = time.monotonic() + TIMEOUT_S
        while run3.state != "running" and time.monotonic() < deadline:
            time.sleep(0.02)
        neighbour = service.submit(JobsAdapter(service).build_spec("compile_epub", "Other book", (),
                                                                   {"folder": str(tmp_path)}))
        assert service.snapshot(neighbour).state is JobState.QUEUED
        time.sleep(0.1)
        assert run3.live and run3.state == "running"  # a queued neighbour never ends the chat's run
        runs.request_stop("5")
        assert backend.stops[-1] == "graceful" and runs.own_job_state("5") == "STOPPING"
        runs.request_stop("5", force=True)
        assert backend.stops[-1] == "force"
        assert service.wait_idle(TIMEOUT_S)
        _finish_all(runs)
        assert service.snapshot(run3.job_id).state is JobState.CANCELLED and not run3.live

        # Resume from the chat: the stopped run's params again (same run root)
        service._kinds = lambda kind: job_kinds.KindInfo(kind="direct_text", verb="Translating", icon="CHAT",
                                                         stop_kind="translation", run=lambda ctx: None)
        again = asyncio.run(runs.resubmit("5"))
        assert again is not None and again.run.temp_root == run3.run.temp_root
        assert service.wait_idle(TIMEOUT_S)
        _finish_all(runs)
        assert not again.live

        # Interrupted -> Resume resubmits the same JobSpec (same run root): the chat re-attaches from params["run"]
        def finish_resumed(ctx):
            expected = Path(ctx.params["run"]["expected_output"])
            expected.parent.mkdir(parents=True, exist_ok=True)
            expected.write_text("Resumed translation", encoding="utf-8")
            return None

        service._kinds = lambda kind: job_kinds.KindInfo(kind="direct_text", verb="Translating", icon="CHAT",
                                                         stop_kind="translation", run=finish_resumed)
        resumed_spec = service.snapshot(run3.job_id).spec
        before = len(adapter.messages("5"))
        resumed_id = service.submit(resumed_spec)
        assert service.wait_idle(TIMEOUT_S)
        deadline = time.monotonic() + TIMEOUT_S
        while (runs.run_for("5").job_id != resumed_id or runs.run_for("5").live) and time.monotonic() < deadline:
            time.sleep(0.02)
        _finish_all(runs)
        resumed = runs.run_for("5")
        assert resumed.job_id == resumed_id and resumed.run.temp_root == run3.run.temp_root and resumed.state == "done"
        assert len(adapter.messages("5")) == before + 1
        assert adapter.message_text("5", before) == "Resumed translation"
    finally:
        runs.detach()
        service.close()
        adapter.close()


TIMEOUT_S = 10.0


@needs_flet
def test_approval_card_and_bom_preserving_editor(tmp_path):
    from glossarion_mobile.ui.chat.cards import (
        NO_GLOSSARY_FILE_TEXT,
        GlossaryApprovalCard,
        GlossaryEditorView,
        glossary_preview,
    )

    glossary = tmp_path / "glossary.csv"
    glossary.write_bytes("\ufefftype,raw_name,translated_name\ncharacter,김철수,Kim Cheolsu\nterm,마나,Mana\n".encode("utf-8"))
    info = glossary_preview(str(glossary))
    assert info["exists"] and info["bom"] and info["entries"] == 2
    assert info["preview"] == [("김철수", "Kim Cheolsu"), ("마나", "Mana")]
    answers = []
    card = GlossaryApprovalCard(path=str(glossary), info=info, on_answer=answers.append, on_edit=lambda p: None)
    assert not card.edit_button.disabled
    assert card.answer(True) and answers == [True] and card.yes_button.disabled
    assert not card.answer(False)  # one decision only
    missing = GlossaryApprovalCard(path=str(tmp_path / "none.csv"), on_answer=answers.append)
    assert missing.edit_button.disabled and NO_GLOSSARY_FILE_TEXT in [getattr(c, "value", None) for c in missing.content.controls]
    missing.no_button.on_click(None)
    assert answers == [True, False]
    editor = GlossaryEditorView(str(glossary), info["text"] + "term,검,Sword\n", has_bom=True)
    assert editor.save() and glossary.read_bytes().startswith(b"\xef\xbb\xbf")
    assert glossary.read_text(encoding="utf-8-sig").endswith("term,검,Sword\n")
    json_glossary = tmp_path / "g.json"
    json_glossary.write_text(json.dumps([{"raw_name": "A", "translated_name": "B"}] * 7), encoding="utf-8")
    assert glossary_preview(str(json_glossary))["entries"] == 7


def test_reader_workspace_finds_the_turns_translation(tmp_path):
    from glossarion_mobile.ui.chat.integration import _reader_workspace

    chat = tmp_path / "Direct Text" / "chat_1"
    first, second = chat / "Book", chat / "Other"
    for folder in (first, second):
        folder.mkdir(parents=True)
        (folder / "translation_progress.json").write_text("{}", encoding="utf-8")
    os.utime(first / "translation_progress.json", (1_000, 1_000))
    os.utime(second / "translation_progress.json", (2_000, 2_000))
    assert _reader_workspace(str(first)) == str(first)  # the run's pipeline folder
    assert _reader_workspace(str(chat), str(tmp_path / "Inbox" / "Book.epub")) == str(first)  # named after the file
    assert _reader_workspace(str(chat), "") == str(second)  # else the newest workspace
    assert _reader_workspace(str(tmp_path / "missing")) == "" and _reader_workspace("") == ""


def test_open_reader_hands_the_reader_a_library_row(tmp_path):
    """Job card Read / Open reader (U5): the run's workspace over its attachment, else the EPUB alone."""
    from glossarion_mobile.ui.chat.integration import ChatFeature

    opened, notes = [], []
    reader = types.SimpleNamespace(open_book=lambda book=None, **kw: opened.append((book, kw)) or "abcdef012345")
    library = types.SimpleNamespace(raw_source=lambda book: str(tmp_path / "resolved.epub"))
    feature = ChatFeature.__new__(ChatFeature)
    feature.dispatcher = None
    feature.app = types.SimpleNamespace(reader=reader, library=library, notify=notes.append)
    workspace = tmp_path / "Output" / "Book"
    workspace.mkdir(parents=True)
    (workspace / "translation_progress.json").write_text("{}", encoding="utf-8")
    source = tmp_path / "Inbox" / "Book.epub"
    source.parent.mkdir()
    source.write_bytes(b"PK")
    assert asyncio.run(feature.open_reader(str(workspace), str(source))) == "abcdef012345"
    book, _kw = opened[-1]
    assert book["path"] == book["output_folder"] == str(workspace) and book["is_in_progress"]
    assert book["raw_source_path"] == str(source) and book["progress_file"].endswith("translation_progress.json")
    asyncio.run(feature.open_reader(str(workspace), ""))  # no attachment: the shared resolvers
    assert opened[-1][0]["raw_source_path"] == str(tmp_path / "resolved.epub")
    asyncio.run(feature.open_reader(str(tmp_path / "not-yet"), str(source)))  # no workspace yet: the EPUB
    assert opened[-1] == (None, {"path": str(source)})
    assert asyncio.run(feature.open_reader("", "")) is None and notes[-1].startswith("Nothing to read yet")
    feature.app = types.SimpleNamespace(notify=notes.append)
    assert asyncio.run(feature.open_reader(str(workspace), str(source))) is None
    assert notes[-1] == "The Reader is not available in this session"


@needs_flet
def test_job_card_read_actions_are_enabled():
    from glossarion_mobile.ui.chat.cards import JobCard

    actions = []
    card = JobCard(attachment={"name": "Book.epub", "path": "/x/Book.epub", "extension": ".epub", "size": 1},
                   phase=CardPhase("running"), on_action=actions.append)
    card.set_phase(CardPhase("running"))
    assert not card.action_buttons["open_reader"].disabled
    card.set_phase(CardPhase("done"), status="Done")
    assert not card.action_buttons["read"].disabled


def test_job_binding_progress_line_and_card_phase(tmp_path):
    snap = types.SimpleNamespace(progress={"total": 48, "completed": 12, "failed": 1}, in_flight=3, started=None)
    clock = [0.0]
    eta = EtaEstimator(clock=lambda: clock[0])
    progress_line(snap, eta=eta)
    clock[0] = 60.0
    snap.progress = {"total": 48, "completed": 13, "failed": 1}
    line = progress_line(snap, eta=eta)
    assert line.startswith("Chapter 13/48 · 3 in flight · 1 failed · ETA 35 min")
    assert CardPhase.for_turn("DONE", stop_requested=True).name == "stopped"
    assert CardPhase.for_turn("RUNNING", plan_pending=True).name == "plan"
    assert CardPhase.for_turn("QUEUED").live


def test_token_hint_and_job_title(tmp_path):
    assert rules.token_hint(0) == "" and rules.token_hint(850) == "≈850 tok" and rules.token_hint(1234) == "≈1.2k tok"
    if _has("tiktoken"):
        assert rules.count_tokens("hello world", "authgpt/gpt-6-luna") > 0
    assert rules.count_tokens("   ", "gpt-4o") == 0
    from glossarion_mobile.ui.chat.run_request import job_title

    run = DirectTextRun(temp_root="", source_path="", source_extension=".epub", is_attachment=True, expected_output="",
                        display_name="Book.epub")
    assert job_title(run, "My chat") == "Book.epub"  # the strip shows "Translating · Book.epub"
    run.is_attachment = False
    assert job_title(run, "My chat") == "My chat" and job_title(run) == "Direct Text"


def test_transcript_window_and_grouping():
    messages = [("user", "a"), ("assistant", "b", "", "", "", "Request 1", {}),
                ("user_file", "book.epub", "/x/book.epub", 1, "", "user"),
                ("assistant", "c", "", "", "", "Chapter 1 · Request 2", {}),
                ("assistant", "d", "", "", "", "Extraction report", {}),
                ("assistant", "e", "", "", "", "Attachment actions", {}),
                ("user", "next")]
    items = build_items(messages)
    assert [i.kind for i in items] == ["user", "assistant", "user_file", "job", "user"]
    job = items[3]
    assert (job.index, job.requests, job.report, job.actions) == (2, [3], 4, 5)
    partial = build_items(messages, 4, 7)  # window starting inside the job group still groups its cards
    assert [i.kind for i in partial] == ["job", "user"] and partial[0].report == 4 and partial[0].index == -1
    many = [("user", str(i)) for i in range(100)]
    assert tail_window(many, 20) == (80, 100)
    assert slide_window((80, 100), 100, 20, -1) == (74, 94)
    assert slide_window((74, 94), 100, 20, 1) == (80, 100)


# ==========================================================================
# Welcome flow state
# ==========================================================================


def test_welcome_flow_state():
    from glossarion_mobile.ui.screens.welcome_flow import STEPS, WelcomeFlow, welcome_glossary_updates

    flow = WelcomeFlow()
    assert flow.step_id == "sign_in" and flow.glossary_mode == "balanced"
    assert flow.next() == "providers" and flow.skipped_sign_in  # Skip for now
    assert flow.back() == "sign_in"
    assert flow.mark_signed_in() == "language_glossary" and flow.signed_in and not flow.skipped_sign_in
    flow.select_mode("full")
    flow.select_mode("bogus")
    assert flow.glossary_mode == "full"
    flow.target_language = "Japanese"
    assert flow.next() == "permissions" and flow.next() == "done" and flow.is_last
    updates = flow.finish_updates()
    assert updates == {"auto_glossary_mode": "full", "enable_auto_glossary": True, "append_glossary": True,
                       "append_glossary_auto_load": True, "output_language": "Japanese",
                       "glossary_mode_dialog_shown": True}
    assert flow.finished and len(STEPS) == 5
    assert WelcomeFlow().skip_updates() == {"glossary_mode_dialog_shown": True}
    assert welcome_glossary_updates("off_no_automap") == {
        "auto_glossary_mode": "off_no_automap", "enable_auto_glossary": False, "append_glossary_auto_load": False}
    assert welcome_glossary_updates("off") == {"auto_glossary_mode": "off", "enable_auto_glossary": False}
    assert welcome_glossary_updates("off_fuzzy_automap")["append_glossary"] is True


# ==========================================================================
# ChatView + ChatFeature on the real app shell (fake JobService, desktop store)
# ==========================================================================


def _load_foundations():
    spec = importlib.util.spec_from_file_location("_glossarion_tf_helpers_chat", Path(__file__).with_name("test_ui_foundations.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeOAuth:
    def __init__(self):
        from glossarion_mobile.services.oauth import SignInState

        self.state = SignInState()
        self.signed_in = set()
        self.returns = []

    def subscribe(self, callback):
        return lambda: None

    async def refresh_status(self, account_id=0):
        return {"signed_in": bool(self.signed_in)}

    def on_return_link(self, provider=None):
        self.returns.append(provider)


@needs_flet
def test_chat_feature_on_the_app_shell(app_env, desktop_store_cls, tmp_path):
    from glossarion_mobile.ui.chat.integration import ChatFeature
    from glossarion_mobile.ui.chat.send_state import SendAction, SendState
    from glossarion_mobile.ui.screens.accounts import AccountsScreen
    from glossarion_mobile.ui.screens.output_editor import OutputEditorScreen
    from glossarion_mobile.ui.screens.welcome import WelcomeScreen

    tf = _load_foundations()
    _desktop_history(tmp_path)

    async def scenario():
        _m, conn, session, page, app = await tf._start("android")
        try:
            await tf._wait(lambda: app.state.engine_ready)
            adapter = _adapter(desktop_store_cls, tmp_path, save_delay=0.01)
            jobs = FakeJobService()
            oauth = _FakeOAuth()
            feature = await ChatFeature.install(app, chats=adapter, jobs=jobs, oauth=oauth)
            view = app.chat_view
            # the drawer and the transcript come from the desktop history
            assert app.state.chats is adapter and set(app.drawer.chat_rows) == {"2", "5"}
            assert app.state.current_chat.value == "2" and view.header.title_text.value == "My novel"
            assert view.composer.text == "half-typed"  # draft restored
            kinds = [type(c).__name__ for c in view.transcript.cards]
            assert kinds == ["UserBubble", "AssistantMessage", "UserFileCard", "JobCard"]
            assert view.transcript.cards[1].content_md.value == "안녕 → **Hello**"
            assert view.transcript.cards[1].thinking_body.visible  # expanded index 1 persisted
            # blocked until ChatGPT sign-in (default model authgpt/gpt-6-luna)
            assert view.composer.send_state is SendState.BLOCKED
            assert view.caption.fix_button.content == "Sign in with ChatGPT"
            oauth.signed_in.add("authgpt")  # the token store is authoritative (refresh_sign_in re-reads it)
            await feature.refresh_sign_in()
            assert app.state.signed_in.value == frozenset({"authgpt"})
            assert view.composer.send_state is SendState.IDLE_READY
            # send -> direct_text job with the chat's settings; composer cleared; draft saved empty
            view.composer.set_text("다음 문장")
            assert view.composer.send_button.tap() is SendAction.SEND
            await tf._wait(lambda: jobs.submitted)
            spec = jobs.submitted[0]
            assert spec.params["chat_id"] == 2 and spec.params["user_index"] == 5
            assert view.composer.text == "" and adapter.draft("2") == ""
            await tf._wait(lambda: view.composer.send_state is SendState.RUNNING)
            # stream: RUNNING snapshot + the job's log lines -> live card at the tail
            jobs.publish("RUNNING", last_line="📤 sending api call now")
            for line in CARD_LINES:
                jobs.line(line)
            await tf._wait(lambda: any(type(c).__name__ == "AssistantMessage" for c in view.transcript.tail), timeout=5)
            # graceful stop then force (desktop double-click)
            view.composer.send_button.tap()
            assert jobs.stops == [False] and view.composer.send_state is SendState.FINISHING
            view.composer.send_button.tap()
            assert jobs.stops == [False, True]
            # finish: output file -> committed card
            run = adapter and feature.runs.run_for("2")
            Path(run.run.expected_output).parent.mkdir(parents=True, exist_ok=True)
            Path(run.run.expected_output).write_text("Next sentence", encoding="utf-8")
            jobs.publish("DONE")
            await tf._wait(lambda: not run.live, timeout=10)
            await tf._wait(lambda: view.composer.send_state is not SendState.STOPPING, timeout=5)
            assert adapter.message_text("2", len(adapter.messages("2")) - 1, "content") == "Next sentence"
            assert view.transcript.tail == []
            # an EPUB attachment: Plan card first (UI_SPEC §2.12.1); Cancel restores the composer
            book = tmp_path / "book2.epub"
            book.write_bytes(b"PK\x03\x04")
            image = tmp_path / "page.png"
            image.write_bytes(b"\x89PNG")
            assert view.attach_file(str(image)) and app.state.output_mode.value.label == "Output: Vision · auto"
            assert view.attach_file(str(book)) and app.state.output_mode.value.label == "Output: Text"
            assert view.composer.attachment["name"] == "book2.epub" and adapter.attachment("2")["name"] == "book2.epub"
            assert view.send_inputs().block is None, view.send_inputs()
            assert view.composer.send_button.tap() is SendAction.SEND
            plan = adapter.meta("2").get("pending_plan")
            assert plan and adapter.messages("2")[plan["user_index"]][0] == "user_file"
            assert type(view.transcript.cards[-1]).__name__ == "JobCard" and view.transcript.cards[-1].phase.name == "plan"
            count = len(adapter.messages("2"))
            view.cancel_plan()
            assert len(adapter.messages("2")) == count - 1 and view.composer.attachment["name"] == "book2.epub"
            view.composer.send_button.tap()
            submitted = len(jobs.submitted)
            view.start_plan()
            await tf._wait(lambda: len(jobs.submitted) > submitted)
            spec = jobs.submitted[-1]
            assert spec.params["is_attachment"] and spec.inputs == (str(book),)
            assert spec.params["user_index"] == count - 1 and spec.params["options"]["force_no_glossary"] is False
            jobs.publish("RUNNING")
            await tf._wait(lambda: view.live_job_card is not None)
            jobs.publish("CANCELLED")
            await tf._wait(lambda: not feature.runs.run_for("2").live, timeout=10)
            # a toggle tap persists direct_text_output_mode (desktop _set_direct_output_mode(persist=True))
            assert view.composer.output_row.tap("refinement") == "selected"
            assert app.config_store.get("direct_text_output_mode") == "refinement"
            view.composer.output_row.tap("text")
            # token hint for longer text (tiktoken in a worker, debounced)
            view.composer.set_text("안녕하세요 " * 60)
            await tf._wait(lambda: view.composer.token_hint.visible, timeout=5)
            assert view.composer.token_hint.value.startswith("≈")
            view.composer.set_text("")
            # long-press Send -> "Translate once with another model…" (UI_SPEC §2.2 one-shot): the choice
            # overrides only this send, never the chat override or the global model
            assert view.open_model_once() is None  # nothing to send yet
            view.composer.set_text("한 번만")
            sheet = view.open_model_once()
            assert sheet is not None and sheet.one_shot and sheet.heading == "Use once"
            submitted = len(jobs.submitted)
            sheet.select("model", "gpt-6-mini")
            await tf._wait(lambda: len(jobs.submitted) > submitted)
            assert jobs.submitted[-1].params["config_overrides"]["model"] == "gpt-6-mini"
            assert "model" not in adapter.overrides("2") and app.config_store.get("model") != "gpt-6-mini"
            jobs.publish("CANCELLED")
            await tf._wait(lambda: not feature.runs.run_for("2").live, timeout=10)
            # Vision / Refine mode options: the shared settings tiles of their schema keys (global keys)
            vision = view.open_mode_options("vision")
            assert {"vision_ocr_skip_translation", "process_webnovel_images", "vision_ocr_keep_images"} <= set(vision.tiles)
            assert vision.tiles["vision_ocr_skip_translation"].apply(True)
            assert app.config_store.get("vision_ocr_skip_translation") is True
            refine = view.open_mode_options("refinement")
            assert "multipass_refinement_mode" in refine.tiles
            # U7: Image / Video / Audio options (UI_SPEC §2.6), no milestone ReasonChip any more
            image = view.open_mode_options("image")
            assert {"image_output_resolution", "vision_ocr_batch_translation", "vision_ocr_batch_size"} <= set(image.tiles)
            assert image.generate_button is not None and image.generate_button.disabled  # empty composer
            video = view.open_mode_options("video")
            assert video.tiles["nanogpt_video_duration"].label == "Video Duration"
            assert "tts_voice" in view.open_mode_options("audio").tiles
            assert view.open_mode_options("text").generate_button is None
            # sheets and screens
            settings_sheet = view.open_chat_settings()
            settings_sheet.set_value("glossary_override_mode", "manual")
            assert adapter.overrides("2")["glossary_override_mode"] == "manual"
            assert ("glossary", "Glossary: Manual") in view.option_pills()
            assert isinstance(feature.make_screen(tf.parse_route("/settings/accounts")), AccountsScreen)
            assert isinstance(feature.make_screen(tf.parse_route("/welcome")), WelcomeScreen)
            mid = adapter.mid_for_index("2", 1)
            editor = feature.make_screen(tf.parse_route(f"/chat/2/m/{mid}/edit"))
            assert isinstance(editor, OutputEditorScreen) and editor.reason is None
            editor.get_body()
            editor.field.value = "Edited **Hello**"
            assert await editor.save()
            assert adapter.message_text("2", 1, "content") == "Edited **Hello**"
            assert not feature.needs_welcome()  # a returning user (tf._start marks the welcome done)
            app.prefs.set("welcome_completed", False)
            assert feature.needs_welcome()  # and config.json has no desktop glossary-mode choice yet
            # OAuth return deep link reaches the bridge
            await tf._route(session, "/oauth/return?p=authgpt")
            assert oauth.returns == ["authgpt"]
            # drawer: open the other chat
            app._open_chat("5")
            await tf._wait(lambda: view.cid == "5")
            assert view.transcript.is_empty and view.header.title_text.value == "New chat"
            feature.close()
        finally:
            await tf._stop(app)

    asyncio.run(scenario())


@needs_flet
def test_app_start_installs_jobs_chat_and_the_first_run_welcome(app_env, tmp_path):
    """GlossarionApp.start wires the U3 features: JobsFeature (JobService, files, intents),
    ChatFeature (the shared Direct Text store), the Translate-in-new-chat intent, shared text into
    the composer, and the Welcome flow on a first run that opened on the chat home."""
    from glossarion_mobile.services.intents import ACTION_TRANSLATE_NEW_CHAT, IntentImport
    from glossarion_mobile.services.jobs import JobService
    from glossarion_mobile.ui.chat.integration import ChatFeature
    from glossarion_mobile.ui.screens.jobs import JobsFeature

    tf = _load_foundations()

    async def scenario():
        _m, conn, session, page, app = await tf._start("android", first_run=True)
        try:
            assert isinstance(app.jobs, JobsFeature) and isinstance(app.job_service, JobService)
            assert app.files is app.jobs.files and app.intents is app.jobs.intents
            assert isinstance(app.chat_feature, ChatFeature) and app.chat_feature.jobs.jobs is app.jobs
            assert app.state.chats is app.chat_feature.chats and app.chat_view.bound
            assert app.intents.handlers[ACTION_TRANSLATE_NEW_CHAT] == app._translate_in_new_chat
            # first run (no desktop glossary-mode choice yet) on the chat home -> Welcome
            assert await tf._wait(lambda: tf._routes(page)[-1] == "/welcome")
            await tf._route(session, "/")
            # Open-with "Translate in new chat": a fresh chat with the file attached
            book = tmp_path / "Shared Book.epub"
            book.write_bytes(b"PK\x03\x04")
            imported = types.SimpleNamespace(path=str(book), name=book.name)
            cid = app._translate_in_new_chat(IntentImport(item={"kind": "file", "path": str(book)}, imported=imported))
            assert cid and app.chat_view.cid == cid
            assert app.chat_view.composer.attachment["name"] == "Shared Book.epub"
            # shared text -> the composer (and the chat's draft)
            app.prefill_composer("번역해 주세요")
            assert app.chat_view.composer.text == "번역해 주세요" and app.state.chats.draft(cid) == "번역해 주세요"
        finally:
            app.jobs.close()
            await tf._stop(app)

    asyncio.run(scenario())
    from glossarion_mobile import runtime_bootstrap as rb

    # the user finished the Welcome flow (WelcomeFlow -> prefs welcome_completed)
    (Path(rb.get_paths().data) / "mobile_state.json").write_text(json.dumps({"welcome_completed": True}),
                                                                 encoding="utf-8")

    async def returning():
        _m, conn, session, page, app = await tf._start("android")
        try:
            await asyncio.sleep(0.3)
            assert tf._routes(page) == ["/"] and not app.chat_feature.welcome_shown
        finally:
            app.jobs.close()
            await tf._stop(app)

    asyncio.run(returning())


# ==========================================================================
# Hygiene and theme
# ==========================================================================


def test_pure_chat_modules_do_not_import_flet():
    import subprocess

    script = (
        "import sys, json; sys.path.insert(0, %r)\n"
        "import glossarion_mobile.state.chat_store_adapter, glossarion_mobile.services.oauth\n"
        "import glossarion_mobile.ui.chat.direct_text_rules, glossarion_mobile.ui.chat.run_request\n"
        "import glossarion_mobile.ui.chat.stream_bridge, glossarion_mobile.ui.chat.job_binding\n"
        "import glossarion_mobile.ui.chat.run_controller, glossarion_mobile.ui.chat.transcript_model\n"
        "import glossarion_mobile.ui.screens.welcome_flow\n"
        "import glossarion_mobile.ui.chat.media_model, glossarion_mobile.ui.chat.chat_ops\n"
        "print(json.dumps(sorted(m for m in ('flet', 'PySide6', 'translator_gui', 'TransateKRtoEN') if m in sys.modules)))\n"
    ) % str(APP_DIR)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "[]"


@needs_flet
def test_text_theme_carries_the_on_surface_colour():
    import flet as ft

    from glossarion_mobile.ui import theme

    for dark in (False, True):
        styles = theme.build_theme(dark=dark).text_theme
        assert styles.title_medium.color == ft.Colors.ON_SURFACE and styles.label_small.color == ft.Colors.ON_SURFACE
    assert theme.build_theme(dark=True, amoled=True).text_theme.body_medium.color == ft.Colors.ON_SURFACE


# ---------------------------------------------------------------------------
# U3 review fixes: relaunch card state + Resume resolving the Interrupted entry, the finish
# never reading the live environment, non-draining repaints, the Plan card's glossary chip
# ---------------------------------------------------------------------------


def test_relaunch_card_state_and_resume_resolve_the_interrupted_job(desktop_store_cls, tmp_path):
    """After a kill ChatRuns knows no run of the chat: the turn's JobCard reads the job's real end
    (Interrupted, with the chapter counts) from JobService, and the chat's Resume goes through
    ``JobService.resume`` so the Jobs page / launch banner cannot run the same work again."""
    from glossarion_mobile import job_kinds
    from glossarion_mobile.services.jobs import JobService, JobSnapshot, JobState, Progress
    from glossarion_mobile.ui.chat.job_binding import ended_card, ended_kind

    _desktop_history(tmp_path)
    adapter = _adapter(desktop_store_cls, tmp_path, save_delay=0.01)
    before_kill = FakeJobService()
    sent = ChatRuns(adapter, JobsAdapter(before_kill), temp_dir=str(tmp_path))
    run = asyncio.run(sent.send("5", text="안녕", attachment=None, settings=DirectTextSettings(), output_mode="text"))
    spec = before_kill.submitted[0]
    # cards were committed before the kill (e.g. at the glossary gate): the card must still say Interrupted
    killed = JobSnapshot(id="killedjob001", spec=spec, state=JobState.RUNNING, created=1.0, started=2.0,
                         progress=Progress(total=12, completed=3))
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    (jobs_dir / "active.state").write_text(json.dumps({"version": 1, "saved_at": 3.0, "active": killed.to_dict(),
                                                       "queue": []}), encoding="utf-8")
    ran = []
    info = job_kinds.KindInfo(kind="direct_text", verb="Translating", icon="CHAT", stop_kind="translation",
                              run=lambda ctx: ran.append(ctx.params["user_index"]))
    service = JobService(jobs_dir=str(jobs_dir), backend=_ChatBackend(), kinds=lambda kind: info,
                         config_loader=lambda: {})
    relaunched = ChatRuns(adapter, JobsAdapter(service), temp_dir=str(tmp_path))
    relaunched.attach()
    try:
        assert relaunched.run_for("5") is None
        snap = relaunched.persisted_job("5", run.user_index)
        assert snap is not None and snap.id == "killedjob001"
        assert ended_card(ended_kind(snap), snap) == ("interrupted", "Interrupted · 3/12 chapters")
        assert relaunched.persisted_job("5", run.user_index + 7) is None
        resumed = asyncio.run(relaunched.resubmit("5"))
        assert resumed is not None and service.wait_idle(TIMEOUT_S)
        _finish_all(relaunched)
        assert service.interrupted == ()  # resolved "resumed" by the chat's Resume
        assert service.resume("killedjob001") is None  # the banner / Jobs page cannot run it again
        assert ran == [run.user_index]
        newest = relaunched.persisted_job("5", run.user_index)
        assert newest.id == resumed.job_id and ended_kind(newest) == "done"
    finally:
        relaunched.detach()
        service.close()
        adapter.close()


def test_ended_card_labels():
    from glossarion_mobile.services.jobs import JobSnapshot, JobSpec, JobState, Progress
    from glossarion_mobile.ui.chat.job_binding import ended_card, ended_kind

    def snap(state, **kwargs):
        return JobSnapshot(id="j", spec=JobSpec("direct_text", "Book"), state=state, created=0.0, **kwargs)

    counts = Progress(total=48, completed=12)
    assert ended_card(ended_kind(snap(JobState.CANCELLED, progress=counts)), snap(JobState.CANCELLED, progress=counts)) \
        == ("stopped", "Stopped · 12/48 chapters")
    assert ended_card(ended_kind(snap(JobState.FAILED)), snap(JobState.FAILED)) == ("failed", "Failed")
    assert ended_kind(snap(JobState.INTERRUPTED, resolution="discarded")) == "discarded"
    assert ended_card("done", snap(JobState.DONE, progress=Progress(total=4, completed=4, failed=1))) == (
        "done", "Finished with issues · 1 failed")
    assert ended_card("done", snap(JobState.DONE, progress=Progress(total=4, completed=4))) == ("done", "Done · 4/4 chapters")


def test_finish_reads_the_job_recorded_environment_not_the_live_one(tmp_path, monkeypatch):
    """The chat finishes on gl-chat-finish, maybe while the next queued job exported its own
    MANUAL_GLOSSARY: the attachment workspace must get this run's glossary."""
    import direct_text_store

    out_root = tmp_path / "Output"
    out_root.mkdir()
    store = direct_text_store.ChatStore(str(tmp_path / "direct_text_chats.json"), output_root=str(out_root))
    session = store.new_chat_session(1)
    session["messages"] = [("user_file", "book.epub", str(tmp_path / "book.epub"), 10, "", "user")]
    run_root = tmp_path / "run"
    (run_root / "book").mkdir(parents=True)
    (run_root / "book.epub").write_text("x", encoding="utf-8")
    (run_root / "book" / "book.epub").write_text("compiled A", encoding="utf-8")
    glossary_a = run_root / "book" / "glossary.csv"
    glossary_a.write_text("type,raw_name,translated_name\ncharacter,A,Chat-A name\n", encoding="utf-8")
    glossary_b = tmp_path / "other_job" / "glossary.csv"
    glossary_b.parent.mkdir()
    glossary_b.write_text("type,raw_name,translated_name\ncharacter,B,Chat-B name\n", encoding="utf-8")
    run = {"temp_root": str(run_root), "source_path": str(run_root / "book.epub"), "source_extension": ".epub",
           "is_attachment": True, "expected_output": str(run_root / "book" / "book.epub"), "manual_glossary_path": "",
           "started_at": 0.0, "output_mode": "text", "force_no_glossary": False, "glossary_path": str(glossary_a),
           "run_env": {"MANUAL_GLOSSARY": str(glossary_a)}}
    monkeypatch.setenv("MANUAL_GLOSSARY", str(glossary_b))  # the next queued job's glossary
    result = store.finish_run(session, run, [], cleanup=False)
    persisted = Path(result["output_folder"]) / "glossary.csv"
    assert "Chat-A name" in persisted.read_text(encoding="utf-8")
    # the dialog itself (no recorded run environment) keeps reading the live os.environ
    assert direct_text_store.ChatStoreMixin._RUN_ENVIRONMENT is None


def test_finish_state_carries_the_recorded_run_environment(desktop_store_cls, tmp_path):
    jobs = FakeJobService()
    adapter, runs = _runs(desktop_store_cls, tmp_path, jobs)
    run = asyncio.run(runs.send("5", text="hi", attachment=None, settings=DirectTextSettings(), output_mode="text"))
    snapshot = types.SimpleNamespace(spec=jobs.submitted[0], result={
        "glossary_path": "", "run_env": {"MANUAL_GLOSSARY": "C:/g.csv"}, "model": "m"})
    state = runs.finish_run_state(run, snapshot)
    assert state["run_env"] == {"MANUAL_GLOSSARY": "C:/g.csv"} and state["model"] == "m"
    assert runs.finish_run_state(run, types.SimpleNamespace(spec=jobs.submitted[0], result={}))["run_env"] == {}
    adapter.close()


def test_repaints_read_the_cards_without_draining_the_backlog():
    import direct_text_stream

    stream = direct_text_stream.make_stream(source_is_attachment=False, request_number=1, model="gpt-4o")
    for line in CARD_LINES:
        stream.feed(line, "Thread-2 (api_call)")
    bridge = RunStream(provider=lambda: stream)
    assert bridge.segments() == [] and stream.segments(drain=False) == []  # nothing classified yet
    assert bridge.drain()  # the budgeted UI tick (or the job-side drain) classifies them
    assert bridge.segments() and bridge.segments()[0]["label"] == "Request 1"
    assert stream.segments() == stream.segments(drain=False)  # the default still drains everything


@pytest.mark.parametrize("override,config,expected", [
    ("attachments_only", {"auto_glossary_mode": "balanced"}, "Glossary: Balanced (auto)"),
    ("none", {"auto_glossary_mode": "Off (Fuzzy Mapping)"}, "Glossary: Off (Fuzzy Mapping) (auto)"),
    ("none", {"enable_auto_glossary": True}, "Glossary: Minimal (auto)"),
    ("none", {}, "Glossary: Off (auto)"),
    ("no_glossary", {"auto_glossary_mode": "full"}, "Glossary: Off"),
    ("manual", {"auto_glossary_mode": "full"}, "Glossary: Manual"),
])
def test_plan_card_shows_the_effective_glossary_mode(override, config, expected):
    assert rules.effective_glossary_label(override, lambda key, default=None: config.get(key, default)) == expected


# ==========================================================================
# Device fixes (owner report on the U8 APK, 2026-10-08): QA scan from the chat, "Open in Library"
# for workspaces that moved into the Library by themselves, "Always accept" on the glossary card,
# a Library book attached in the chat continues in its own workspace ("Save to: Library").
# A real ChatView on the desktop-format history, a fake JobService; every path in pytest's tmp dir.
# ==========================================================================


class _Prefs:
    """Prefs stand-in (``get`` / ``set`` / ``file_ref``)."""

    def __init__(self):
        self.values: dict = {}
        self.refs: dict = {}

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value

    def file_ref(self, path, kind=None):
        rid = f"r{len(self.refs) + 1}"
        self.refs[rid] = (str(path), kind)
        return rid


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """No test here reads or writes the user's Library, output folders, home or app data."""
    for name, sub in (("HOME", "home"), ("USERPROFILE", "home"), ("APPDATA", "appdata"),
                      ("GLOSSARION_LIBRARY_DIR", "Library"), ("GLOSSARION_DATA_DIR", "data")):
        folder = tmp_path / sub
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(name, str(folder))
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(tmp_path / "Output"))  # where a workspace moves (migrate)
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    return tmp_path


def _config_store(tmp_path):
    from glossarion_mobile.state.config_store import MobileConfigStore
    from glossarion_mobile.ui.settings.schema_access import SchemaAccess

    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")
    schema = SchemaAccess()
    store = MobileConfigStore(path, debounce=10, defaults=schema.effective_default,
                              reader=lambda p, decrypt=True: json.loads(Path(p).read_text(encoding="utf-8")),
                              writer=lambda disk, p, backup=False: Path(p).write_text(json.dumps(disk),
                                                                                       encoding="utf-8"))
    store.load()
    return store, schema


def _chat_view(desktop_store_cls, tmp_path, *, cid="2", prefs=None, jobs=None, **env_fields):
    """A real ChatView (call inside a running loop) on ``_desktop_history``; ``calls`` records
    navigation, snackbars, sheets and the spawned tasks (``_settle`` waits for them)."""
    from glossarion_mobile.state.app_state import AppState
    from glossarion_mobile.ui.chat.chat_view import ChatView
    from glossarion_mobile.ui.chat.context import ChatEnv

    jobs = jobs or FakeJobService()
    adapter, runs = _runs(desktop_store_cls, tmp_path, jobs)
    store, schema = _config_store(tmp_path)
    calls = types.SimpleNamespace(nav=[], notes=[], dialogs=[], tasks=[])
    page = types.SimpleNamespace(width=412, height=900, show_dialog=calls.dialogs.append, update=lambda *a: None)
    env = ChatEnv(page=page, store=store, schema=schema, prefs=prefs, chats=adapter, runs=runs,
                  jobs=JobsAdapter(jobs), **env_fields)

    def spawn(coro):
        task = asyncio.ensure_future(coro)
        calls.tasks.append(task)
        return task

    env.spawn = spawn
    state = AppState()
    state.current_chat.set(cid)
    view = ChatView(page, state=state, navigate=lambda name, params=None: calls.nav.append((name, params)),
                    notify=lambda message, action_label=None, on_action=None: calls.notes.append(
                        (message, action_label, on_action)), env=env)
    return types.SimpleNamespace(view=view, adapter=adapter, runs=runs, jobs=jobs, calls=calls, store=store)


async def _settle(calls):
    """Wait for every task the view spawned (also the ones they spawn); re-raise a failure."""
    while any(not task.done() for task in calls.tasks):
        await asyncio.gather(*[t for t in calls.tasks if not t.done()], return_exceptions=True)
    for task in calls.tasks:
        if not task.cancelled() and task.exception() is not None:
            raise task.exception()


def _close(chat):
    chat.adapter.close()
    chat.store._saver.close()


def _job_item(adapter, cid="2"):
    return next(i for i in build_items(adapter.messages(cid)) if i.kind == "job" and i.index >= 0)


def _cards(view, cls):
    return [card for card in view.transcript.cards if isinstance(card, cls)]


def test_transcript_gives_a_qa_scan_message_its_own_item():
    from glossarion_mobile.ui.chat.transcript_model import LIBRARY_LABEL, QA_LABEL

    messages = [
        ("user_file", "book.epub", "/x/book.epub", 1, "", "user"),
        ("assistant", "ch1", "", "", "/w/Attachments/book", "Request 1", {}),
        ("assistant", "🔎 QA", "", "", "/w/Attachments/book", QA_LABEL, {"qa_job": "j1"}),
        ("assistant", "ch2", "", "", "/w/Attachments/book", "Request 2", {}),  # a Resume after the scan
        ("user", "hi"),
        ("assistant", "🔎 QA", "", "", "/w/Book", QA_LABEL, {"qa_job": "j2"}),
        ("user_file", "saga.epub", "/x/saga.epub", 1, "", "user"),
        ("assistant", "📚", "", "", "", LIBRARY_LABEL, {"library_job": "j3"}),
    ]
    items = build_items(messages)
    assert [(i.kind, i.index) for i in items] == [("user_file", 0), ("job", 0), ("qa", 2), ("user", 4), ("qa", 5),
                                                   ("user_file", 6), ("job", 6)]
    assert items[1].requests == [1, 3] and items[-1].library == 7


@needs_flet
def test_result_card_qa_scans_the_turn_workspace(desktop_store_cls, isolated_env):
    """Result card › QA scan (owner #7): the turn's own workspace (never a finished run's deleted temp
    root), Quick Scan through the shared scanner (``qa_model.chat_qa_job``); its "QA scan" card offers
    Job · Report · Chapters, and the Result card's "N QA failed" chip re-reads the workspace after."""
    import flet as ft

    from glossarion_mobile.ui.chat import chat_view as cv
    from glossarion_mobile.ui.chat.cards import ATTACHMENT_ACTION_REASONS, JobCard
    from glossarion_mobile.ui.chat.transcript_model import QA_LABEL
    from glossarion_mobile.ui.tools import qa_model

    tmp_path = isolated_env
    prefs = _Prefs()
    opened: list = []
    went: list = []
    tools_ctx = types.SimpleNamespace(prefs=prefs, go=lambda name, params=None: went.append((name, params)),
                                      say=lambda *a: went.append(("say",) + a))

    async def open_progress(folder, source=""):
        opened.append((folder, source))
        return "progress"

    async def scenario():
        chat = _chat_view(desktop_store_cls, tmp_path, prefs=prefs, tools_context=lambda: tools_ctx,
                          open_progress=open_progress)
        view, adapter, jobs, calls = chat.view, chat.adapter, chat.jobs, chat.calls
        try:
            await _settle(calls)
            messages = adapter.messages("2")
            item = _job_item(adapter)
            workspace = Path(messages[3][4])
            source = str(messages[2][2])
            assert "qa" not in ATTACHMENT_ACTION_REASONS
            result = next(c for c in _cards(view, JobCard) if c.title_text.value == "book.epub")
            assert isinstance(result.action_buttons["qa"], ft.FilledTonalButton)
            assert not result.action_buttons["qa"].disabled
            # nothing translated in the workspace yet
            view._on_job_action("qa", item)
            await _settle(calls)
            assert calls.notes[-1][0] == cv.NOTHING_TO_SCAN and not jobs.submitted
            (workspace / "response_001_ch001.html").write_text("<p>Chapter one</p>", encoding="utf-8")
            # a desktop 1000 saved by the U6-U9 QA screen: the one-time mobile move to 0 (owner decision)
            chat.store.set(("qa_scanner_settings", "quick_scan_sample_size"), 1000)
            # a finished run's temp root is deleted: the turn's workspace comes first (U9 Progress bug)
            gone = tmp_path / "cache" / "glossarion_input_output_x" / "book"
            view.env.runs.run_for = lambda cid: types.SimpleNamespace(output_dir=str(gone), output_folder=str(gone),
                                                                      user_index=2, live=False)
            view._run_turn = lambda run: 2
            try:
                assert view._reader_target(item) == (str(workspace), source)
            finally:
                del view.env.runs.run_for
                del view._run_turn
            view._on_job_action("qa", item)
            await _settle(calls)
            spec = jobs.submitted[-1]
            assert str(getattr(spec.kind, "value", spec.kind)) == "qa_scan"
            assert spec.inputs == (str(workspace),) and spec.params["mode"] == qa_model.CHAT_QA_MODE
            target = spec.params["targets"][0]
            assert target["folder"] == str(workspace) and target["source"] == source and target["direct_text"] is True
            assert spec.origin["type"] == "chat" and spec.origin["cid"] == "2" and "chat_id" not in spec.params
            last = adapter.messages("2")[-1]
            assert last[5] == QA_LABEL and last[4] == str(workspace)
            # the history's save keeps only the desktop storage keys: the card's data lives in the sidecar
            stored = view._tool_storage(last)
            assert stored["qa_job"] == "job1" and stored["source"] == source
            assert adapter.meta("2")[cv.TOOL_JOBS_META][last[6]["created_at"]]["qa_job"] == "job1"
            assert "duplicate check off (sample size 0)" in stored["summary"]  # the mobile default
            assert chat.store.get(("qa_scanner_settings", "quick_scan_sample_size")) == 0
            assert prefs.values[qa_model.QUICK_SAMPLE_MIGRATION_PREF] is True
            assert calls.notes[-1][0] == "QA scan · book"
            # its own card: Job · Report · Chapters
            card = next(c for c in _cards(view, JobCard) if c.title_text.value == "QA scan · book")
            buttons = {b.key: b for b in card.buttons.controls}
            assert set(buttons) == {"qajob-job", "qajob-report", "qajob-chapters"}
            buttons["qajob-report"].on_click(None)
            await _settle(calls)
            assert calls.notes[-1][0] == "The scan has not finished yet" and went == []
            report = Path(qa_model.report_path_for(str(workspace)))
            report.parent.mkdir(parents=True)
            report.write_text("<html></html>", encoding="utf-8")
            buttons["qajob-report"].on_click(None)
            await _settle(calls)
            assert went == [("tools.qa.report", {"rid": "r1"})] and prefs.refs["r1"] == (str(report), "qa_report")
            buttons["qajob-chapters"].on_click(None)
            await _settle(calls)
            assert opened == [(str(workspace), source)]
            buttons["qajob-job"].on_click(None)
            assert calls.nav[-1] == ("jobs.detail", {"jid": "job1"})
            # the scan marked two chapters: once the job ends the Result card's chip reads the workspace
            assert not result.failed_chip.visible
            (workspace / "translation_progress.json").write_text(json.dumps({"chapters": {
                "1": {"status": "qa_failed"}, "2": {"status": "failed"}, "3": {"status": "completed"}}}),
                encoding="utf-8")
            snap = types.SimpleNamespace(id="job1", spec=spec, state="DONE", question=None, progress=None,
                                         in_flight=0, last_line="", result={})
            view._on_job_snapshot(snap)
            await _settle(calls)
            result = next(c for c in _cards(view, JobCard) if c.title_text.value == "book.epub")
            assert result.failed_chip.visible and result.failed_chip.label.value == "2 QA failed"
            assert not view._track_tool_job(snap)  # the same state again changes nothing
        finally:
            _close(chat)

    asyncio.run(scenario())


@needs_flet
def test_plus_sheet_and_slash_qa_scan_the_chats_latest_workspace(desktop_store_cls, isolated_env):
    """＋ › QA scan and ``/qa`` scan the chat's latest book workspace (also one that moved into the
    Library); a chat without one goes to Tools › QA Scanner."""
    tmp_path = isolated_env

    async def scenario():
        chat = _chat_view(desktop_store_cls, tmp_path)
        view, adapter, jobs, calls = chat.view, chat.adapter, chat.jobs, chat.calls
        try:
            await _settle(calls)
            workspace = Path(adapter.messages("2")[3][4])
            (workspace / "response_001_ch001.html").write_text("<p>x</p>", encoding="utf-8")
            (workspace / "translation_progress.json").write_text(json.dumps({"chapters": {}}), encoding="utf-8")
            view._on_tool("qa")
            await _settle(calls)
            assert jobs.submitted[-1].inputs == (str(workspace),)
            assert view.run_slash("/qa") == "qa"
            await _settle(calls)
            assert len(jobs.submitted) == 2 and jobs.submitted[-1].inputs == (str(workspace),)
            # the workspace moves into the Library (auto-migrate): the chat still scans it
            moved = adapter.migrate_attachment("2", str(workspace), lambda target: True)
            assert moved["ok"]
            target = tmp_path / "Output" / "book"
            assert target.is_dir() and not workspace.exists()
            view.on_workspace_migrated("2", str(target), str(tmp_path / "book.epub"))
            await _settle(calls)
            assert await view.env.run_io(view._latest_workspace) == (str(target), str(tmp_path / "book.epub"))
            view._on_tool("qa")
            await _settle(calls)
            assert jobs.submitted[-1].inputs == (str(target),)
            assert not jobs.submitted[-1].params["targets"][0].get("direct_text")  # a Library book now
            # a chat without a book workspace: Tools › QA Scanner
            view.load_chat("5")
            view._on_tool("qa")
            await _settle(calls)
            assert calls.nav[-1] == ("tools.qa", None) and len(jobs.submitted) == 3
        finally:
            _close(chat)

    asyncio.run(scenario())


@needs_flet
def test_open_in_library_action_and_moved_workspace_retry(desktop_store_cls, isolated_env):
    """No Migrate on mobile (owner #4): the Result card's "Open in Library" waits (ReasonChip) while
    the workspace is in Attachments/, then opens the Library book it became; Retry failed of a moved
    workspace goes to the Library's translate sheet (``ChatEnv.library_translate``)."""
    import flet as ft

    from glossarion_mobile.ui.chat.cards import ATTACHMENT_ACTIONS, LIBRARY_WAIT_REASON, JobCard
    from glossarion_mobile.ui.components.reason_chip import ReasonChip

    tmp_path = isolated_env
    looked: list = []
    handed: list = []

    async def library_book(folder):
        looked.append(folder)
        return "bid7" if folder == str(tmp_path / "Output" / "book") else None

    async def library_translate(folder, source=""):
        handed.append((folder, source))
        return "sheet"

    assert [a[0] for a in ATTACHMENT_ACTIONS if a[0] in ("migrate", "library")] == ["library"]
    assert dict((a[0], a[1:3]) for a in ATTACHMENT_ACTIONS)["library"] == ("Open in Library", "LOCAL_LIBRARY")

    async def scenario():
        chat = _chat_view(desktop_store_cls, tmp_path, library_book=library_book,
                          library_translate=library_translate)
        view, adapter, calls = chat.view, chat.adapter, chat.calls
        try:
            await _settle(calls)
            item = _job_item(adapter)
            workspace = Path(adapter.messages("2")[3][4])
            (workspace / "translation_progress.json").write_text(json.dumps({"chapters": {}}), encoding="utf-8")
            result = next(c for c in _cards(view, JobCard) if c.title_text.value == "book.epub")
            waiting = result.action_buttons["library"]
            assert isinstance(waiting, ft.Row) and isinstance(waiting.controls[1], ReasonChip)
            assert waiting.controls[1].reason == LIBRARY_WAIT_REASON
            assert await view.env.run_io(view._job_workspace_state, item) == (str(workspace), True)
            # Resume / Retry of a workspace still in the chat: the chat's own run (nothing to resume here)
            view._on_job_action("retry", item)
            await _settle(calls)
            assert handed == [] and calls.notes[-1][0] == "Nothing to resume in this chat"
            # the workspace moves into the Library: the card links to the book
            assert adapter.migrate_attachment("2", str(workspace), lambda target: True)["ok"]
            target = tmp_path / "Output" / "book"
            view.on_workspace_migrated("2", str(target), "")
            await _settle(calls)
            result = next(c for c in _cards(view, JobCard) if c.title_text.value == "book.epub")
            button = result.action_buttons["library"]
            assert isinstance(button, ft.FilledTonalButton) and not button.disabled
            assert await view.env.run_io(view._job_workspace_state, item) == (str(target), False)
            button.on_click(None)
            await _settle(calls)
            assert looked[-1] == str(target) and calls.nav[-1] == ("library.book", {"bid": "bid7"})
            view._on_job_action("retry", item)
            await _settle(calls)
            assert handed == [(str(target), str(tmp_path / "book.epub"))]
            # no book for a folder: the Library home
            assert await view.open_in_library(types.SimpleNamespace(index=99, requests=[], report=None,
                                                                    actions=None)) is None
            assert calls.nav[-1] == ("library", None)
        finally:
            _close(chat)

    asyncio.run(scenario())


@needs_flet
def test_approval_card_always_accept_sets_pref_and_answers_yes(desktop_store_cls, isolated_env):
    """Owner #6: "Always accept" on the chat's glossary card stores the All-chats switch (Prefs
    ``chat_auto_accept_glossary``, never config.json) and answers this question Yes through the run
    controller; the chat settings read the pref and the chat's own sidecar value."""
    from glossarion_mobile.ui.chat.cards import GlossaryApprovalCard
    from glossarion_mobile.ui.chat.direct_text_rules import AUTO_ACCEPT_GLOSSARY_PREF

    tmp_path = isolated_env
    answered: list = []
    always: list = []
    card = GlossaryApprovalCard(path="", on_answer=answered.append, on_always=lambda: always.append(True))
    assert card.always_button.visible
    assert card.always() and always == [True] and answered == []  # one answer: on_always answers
    assert all(b.disabled for b in (card.edit_button, card.yes_button, card.no_button, card.always_button))
    assert not card.always() and not card.answer(True) and always == [True]
    plain = GlossaryApprovalCard(path="", on_answer=answered.append)  # the Library review gate's sheet
    assert not plain.always_button.visible and not plain.always()

    prefs = _Prefs()

    async def scenario():
        chat = _chat_view(desktop_store_cls, tmp_path, cid="5", prefs=prefs)
        view, adapter, runs, jobs, calls = chat.view, chat.adapter, chat.runs, chat.jobs, chat.calls
        try:
            await _settle(calls)
            assert view.settings().auto_accept_glossary is False  # default off (desktop always asks)
            book = tmp_path / "book.epub"
            record = {"path": str(book), "name": "book.epub", "extension": ".epub", "size": 15}
            run = await runs.send("5", text="", attachment=record, settings=DirectTextSettings(), output_mode="text")
            jobs.publish("RUNNING", question={"id": "q1", "kind": "direct_text_glossary_approval",
                                              "data": {"path": str(tmp_path / "glossary.csv")}})
            assert run.awaiting_glossary
            approval = view._approval_control(run)
            assert approval.on_always is not None and approval.always_button.visible
            approval.always()
            assert jobs.answers == [("q1", True)] and not run.awaiting_glossary
            assert prefs.values == {AUTO_ACCEPT_GLOSSARY_PREF: True}
            assert calls.notes[-1][0] == "Generated glossaries are accepted automatically from now on"
            assert "chat_auto_accept_glossary" not in json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
            assert view.settings().auto_accept_glossary is True
            adapter.set_meta("5", "auto_accept_glossary", False)  # this chat asks (Chat settings › This chat)
            assert view.settings().auto_accept_glossary is False
            view._always_accept_glossary()  # tapped in this chat: its own "ask" gives way
            assert "auto_accept_glossary" not in adapter.meta("5") and view.settings().auto_accept_glossary is True
            sheet = view.open_chat_settings()
            assert sheet.prefs is prefs
            jobs.publish("CANCELLED")
            _finish_all(runs)
        finally:
            _close(chat)

    asyncio.run(scenario())


def _library_book(tmp_path, name="Novel"):
    raw = tmp_path / "Library" / "Raw" / f"{name}.epub"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(b"PK\x03\x04 raw")
    return raw


@needs_flet
def test_library_attachment_plan_defaults_to_library_destination(desktop_store_cls, isolated_env, monkeypatch):
    """Owner #8: a Library book attached in the chat sends with "Save to: Library" (its own workspace)
    and the Plan card shows the book; a file picked from Files stays "This chat"; with the plan skipped
    the book's Library job starts at once; Start asks the output-root question first and Cancel keeps
    the plan; the Library-job card opens the book."""
    from glossarion_mobile.ui.chat.chat_view import LIBRARY_ATTACHMENT_META
    from glossarion_mobile.ui.chat.transcript_model import LIBRARY_LABEL
    from glossarion_mobile.ui.library import translate_sheet
    from glossarion_mobile.ui.tools.targets import ToolTarget

    tmp_path = isolated_env
    raw = _library_book(tmp_path)
    other = tmp_path / "Inbox" / "Other.epub"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"PK")
    answers: list = []
    asked: list = []

    async def confirm(ctx, books):
        asked.append([dict(b) for b in books])
        return answers.pop(0) if answers else True

    monkeypatch.setattr(translate_sheet, "confirm_output_root", confirm)
    book_row = {"name": "Novel", "type": "in_progress", "raw_source_path": str(raw)}
    service = types.SimpleNamespace(book_for_bid=lambda bid: dict(book_row) if bid == "b1" else None)
    tools_ctx = types.SimpleNamespace(service=service)

    async def scenario():
        chat = _chat_view(desktop_store_cls, tmp_path, cid="5", tools_context=lambda: tools_ctx)
        view, adapter, jobs, calls = chat.view, chat.adapter, chat.jobs, chat.calls
        try:
            await _settle(calls)
            target = ToolTarget(title="Novel", source=str(raw), origin="library", kind="epub", bid="b1")
            assert view.attach_library_book(target)
            assert adapter.meta("5")[LIBRARY_ATTACHMENT_META] == {"bid": "b1", "path": str(raw)}
            assert calls.notes[-1][0] == "Attached Novel.epub from the Library"
            view.begin_send()
            plan = adapter.meta("5")["pending_plan"]
            assert plan["destination"] == "library" and plan["library_bid"] == "b1"
            assert plan["attachment"]["path"] == str(raw)
            await _settle(calls)
            keys = {getattr(c, "key", None) for c in _plan_box(view)}
            assert {"plan-library", "plan-facts"} <= keys
            open_book = next(b for b in _plan_buttons(view) if b.key == "plan-open-book")
            open_book.on_click(None)
            assert calls.nav[-1] == ("library.book", {"bid": "b1"})
            # Start: the output-root question first; Cancel keeps the plan
            answers.append(False)
            view.start_plan()
            await _settle(calls)
            assert asked and asked[-1][0]["name"] == "Novel" and not jobs.submitted
            assert adapter.meta("5")["pending_plan"]["created"] == plan["created"]
            view.start_plan()
            await _settle(calls)
            spec = jobs.submitted[-1]
            assert str(getattr(spec.kind, "value", spec.kind)) == "translate" and spec.inputs == (str(raw),)
            assert spec.origin["cid"] == "5" and spec.origin["bid"] == "b1"
            assert "pending_plan" not in adapter.meta("5")
            last = adapter.messages("5")[-1]
            stored = view._tool_storage(last)
            assert last[5] == LIBRARY_LABEL and stored["bid"] == "b1" and stored["source"] == str(raw)
            assert stored["library_job"] == "job1"
            library_button = _cards_with_key(view, "libjob-library")[0]
            library_button.on_click(None)
            assert calls.nav[-1] == ("library.book", {"bid": "b1"})
            # a file picked from Files (another path) forgets the Library book: "This chat"
            assert view.attach_file(str(other))
            assert LIBRARY_ATTACHMENT_META not in adapter.meta("5")
            view.begin_send()
            plan = adapter.meta("5")["pending_plan"]
            assert "destination" not in plan and "library_bid" not in plan
            view.cancel_plan()
            view.composer._remove_attachment()
            # the plan skipped (Chat settings): the Library book's job starts at once
            adapter.set_meta("5", "skip_plan", True)
            view.attach_library_book(target)
            submitted = len(jobs.submitted)
            view.begin_send()
            await _settle(calls)
            assert len(jobs.submitted) == submitted + 1 and jobs.submitted[-1].inputs == (str(raw),)
            assert jobs.submitted[-1].origin["bid"] == "b1" and "pending_plan" not in adapter.meta("5")
        finally:
            _close(chat)

    asyncio.run(scenario())


def _plan_card(view):
    from glossarion_mobile.ui.chat.cards import JobCard

    return next(c for c in _cards(view, JobCard) if c.phase.name == "plan")


def _plan_box(view):
    return list(_plan_card(view).plan_box.controls)


def _plan_buttons(view):
    import flet as ft

    rows = [c for c in _plan_box(view) if isinstance(c, ft.Row)]
    return [b for row in rows for b in row.controls if getattr(b, "key", "").startswith("plan-")]


def _cards_with_key(view, key):
    from glossarion_mobile.ui.chat.cards import JobCard

    return [b for card in _cards(view, JobCard) for b in card.buttons.controls if getattr(b, "key", None) == key]


@needs_flet
def test_library_book_from_chat_runs_in_its_workspace(desktop_store_cls, isolated_env):
    """The owner's flow end to end on the real chat store: pick a Library/Raw book in the chat, Send,
    Start -> one ``translate`` job over the raw file (the book's own workspace, as Library ›
    Translate… does), tied to the chat and the book."""
    from glossarion_mobile.ui.tools.targets import ToolTarget

    tmp_path = isolated_env
    raw = _library_book(tmp_path, "Saga")

    async def scenario():
        chat = _chat_view(desktop_store_cls, tmp_path, cid="5")
        view, adapter, jobs, calls = chat.view, chat.adapter, chat.jobs, chat.calls
        try:
            await _settle(calls)
            assert view.attach_library_book(ToolTarget(title="Saga", source=str(raw), origin="library", bid="b9"))
            view.composer.set_text("keep the honorifics")
            view.begin_send()
            assert adapter.messages("5")[0][0] == "user_file" and adapter.messages("5")[0][2] == str(raw)
            view._on_job_action("start", _job_item(adapter, "5"))
            await _settle(calls)
            assert len(jobs.submitted) == 1
            spec = jobs.submitted[0]
            assert str(getattr(spec.kind, "value", spec.kind)) == "translate" and spec.inputs == (str(raw),)
            assert spec.origin["type"] == "chat" and spec.origin["cid"] == "5" and spec.origin["bid"] == "b9"
            assert "chat_id" not in spec.params  # the chat's JobStrip shows it (no Job card run)
            assert view.sent[-1][2]["path"] == str(raw)
        finally:
            _close(chat)

    asyncio.run(scenario())
