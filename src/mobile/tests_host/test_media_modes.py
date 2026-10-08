"""Host tests for the U7 chat: output modes with generated media, and the chat surfaces U7 ships.

Run from src/mobile with the 3.13 venv (Flet installed):
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_media_modes.py

* media: the shared desktop rule decides which file a response shows
  (``direct_text_store`` ``_assistant_generated_media`` / ``_generated_media_references_from_text``
  through ``ChatStoreAdapter.message_media``); the cards (gallery, VideoCard, AudioCard), the
  full-screen MediaViewer, the Refine "Compare with original";
* "Generate from prompt (no input)": ``ChatRuns.generate`` -> a ``generate_media`` job whose
  adapter puts the composer text in ``image_job.GENERATIVE_PROMPT_ATTR`` and runs the desktop
  generative-only branch; ``ChatStore.finish_run`` promotes the generated file into the chat folder;
* the chat placeholders of U3/U4 that U7 ships: scratch chats, Delete message, versions
  (Edit & resend / Retranslate), the Attachments manager + Migrate (the desktop
  ``ChatStore.migrate_attachment``), Jump to, Search in chat, Export chat;
* ``job_kinds.image`` (standalone image / video translation).

Persistence runs against the desktop's own code (``direct_text_store.ChatStore``), like test_chat.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import os
import sys
import types
import zipfile
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent

for entry in (str(APP_DIR), str(SRC_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

_TC_SPEC = importlib.util.spec_from_file_location("_glossarion_tc_helpers_media", Path(__file__).with_name("test_chat.py"))
_TC = importlib.util.module_from_spec(_TC_SPEC)
_TC_SPEC.loader.exec_module(_TC)

storage = _TC.storage
app_env = _TC.app_env
needs_flet = _TC.needs_flet
FakeJobService = _TC.FakeJobService
pytestmark = _TC.pytestmark

from glossarion_mobile.state.chat_store_adapter import (  # noqa: E402
    ChatStoreAdapter,
    message_fingerprints,
)
from glossarion_mobile.ui.chat import chat_ops, media_model  # noqa: E402
from glossarion_mobile.ui.chat.direct_text_rules import DirectTextSettings  # noqa: E402
from glossarion_mobile.ui.chat.job_binding import JobsAdapter  # noqa: E402
from glossarion_mobile.ui.chat.run_controller import ChatRuns  # noqa: E402

PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
       b"\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xa7\x35\x81\x84\x00\x00\x00\x00IEND\xaeB`\x82")


@pytest.fixture
def desktop_store_cls():
    if not (SRC_DIR / "direct_text_store.py").is_file():
        pytest.skip("src/direct_text_store.py not present")
    return _TC.make_desktop_store_class()


@contextlib.contextmanager
def restored_env(*keys: str):
    """Put these environment variables back exactly as they were (set or absent) afterwards.

    (``monkeypatch.delenv`` of an absent variable records nothing, so a value the code under
    test exports would leak into later tests.)"""
    saved = {key: os.environ.get(key) for key in keys}
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _adapter(store_cls, root: Path, **kwargs) -> ChatStoreAdapter:
    kwargs.setdefault("scratch_dir", str(root / "cache" / "Direct Text Scratch"))
    return _TC._adapter(store_cls, root, **kwargs)


def _chat_folder(adapter, cid="2") -> Path:
    return Path(adapter.output_folder(cid))


# ==========================================================================
# Pure helpers
# ==========================================================================


def test_media_items_display_content_and_clock(tmp_path):
    image = tmp_path / "gen.png"
    image.write_bytes(PNG)
    gone = tmp_path / "missing.mp4"
    content = f"Here you go\n\n[GENERATED_IMAGE:{image}]\n\n[GENERATED_VIDEO:{gone}]"
    items = media_model.media_items([("image", str(image), True), ("video", str(gone), False), ("text", "x", True)])
    assert [(i.kind, i.exists, i.name) for i in items] == [("image", True, "gen.png"), ("video", False, "missing.mp4")]
    shown = media_model.display_content(content, items)
    assert "GENERATED_IMAGE" not in shown and "Here you go" in shown
    assert "**Generated video unavailable.**" in shown and "`missing.mp4`" in shown
    assert media_model.display_content("plain text", items) == "plain text"
    assert [media_model.format_clock(ms) for ms in (0, 999, 61_000, 3_661_000, None, "bad")] == [
        "0:00", "0:00", "1:01", "1:01:01", "0:00", "0:00"]
    assert media_model.AUDIO_DEFAULT_VOLUME == 0.75 and media_model.VIDEO_ASPECT_RATIO == 16 / 9
    # Vision: the run's cached OCR (OCR/single, then OCR/chunks; empty files skipped)
    (tmp_path / "OCR" / "single").mkdir(parents=True)
    (tmp_path / "OCR" / "chunks").mkdir(parents=True)
    (tmp_path / "OCR" / "single" / "p2.txt").write_text("둘", encoding="utf-8")
    (tmp_path / "OCR" / "single" / "p1.txt").write_text("하나", encoding="utf-8")
    (tmp_path / "OCR" / "single" / "blank.txt").write_text("  ", encoding="utf-8")
    (tmp_path / "OCR" / "chunks" / "p3_chunk_1.txt").write_text("셋", encoding="utf-8")
    assert media_model.ocr_entries(str(tmp_path)) == [("p1.txt", "하나"), ("p2.txt", "둘"), ("p3_chunk_1.txt", "셋")]
    assert media_model.ocr_entries(str(tmp_path), limit=1) == [("p1.txt", "하나")]
    assert media_model.ocr_entries(str(tmp_path / "none")) == []


def test_compare_blocks_and_the_unrefined_backup(tmp_path):
    blocks = media_model.compare_blocks("A\n\nB\n\nC", "A\n\nB2\n\nC\n\nD")
    assert blocks == [("equal", "A", "A"), ("replace", "B", "B2"), ("equal", "C", "C"), ("insert", "", "D")]
    assert media_model.compare_blocks("A\n\nB", "A") == [("equal", "A", "A"), ("delete", "B", "")]
    workspace = tmp_path / "Attachments" / "book"
    (workspace / "unrefined_backup").mkdir(parents=True)
    (workspace / "unrefined_backup" / "response_001_ch1.html").write_text("<p>old one</p>", encoding="utf-8")
    (workspace / "unrefined_backup" / "response_002_ch2.html").write_text("<p>old two</p>", encoding="utf-8")
    progress = {"chapters": {
        "1": {"output_file": "response_001_ch1.html", "unrefined_backup_file": "unrefined_backup/response_001_ch1.html"},
        "2": {"output_file": "response_002_ch2.html", "unrefined_backup_file": "unrefined_backup/response_002_ch2.html"},
        "3": {"output_file": "response_003_ch3.html"},
    }}
    (workspace / "translation_progress.json").write_text(json.dumps(progress), encoding="utf-8")
    found = media_model.find_unrefined_backup(str(workspace), "Chapter 2 (chunk 1/1) · response_002_ch2.html · Request 4")
    assert found and found.endswith("response_002_ch2.html")
    assert media_model.find_unrefined_backup(str(workspace), "Request 9") is None  # two candidates, no name
    attachments = str(workspace.parent)  # a chat folder's Attachments/: the workspaces one level down
    assert media_model.find_unrefined_backup(attachments, "response_001_ch1.html").endswith("response_001_ch1.html")
    assert media_model.find_unrefined_backup(str(tmp_path / "nope"), "") is None


def test_versions_jump_search_and_markdown():
    messages = [
        ("user", "first"),
        ("assistant", "one", "", "", "", "Request 1", {"created_at": "a"}),
        ("user", "second"),
        ("assistant", "two", "", "", "", "Request 2", {"created_at": "b"}),
        ("user", "first edited"),
        ("assistant", "one again", "", "", "", "Request 3", {"created_at": "c"}),
        ("user_file", "book.epub", "/x/book.epub", 10, "keep names", "user"),
        ("assistant", "chapter", "", "", "", "Chapter 1 · ch1.xhtml · Request 4", {"created_at": "d"}),
    ]
    fps = message_fingerprints(messages)
    groups = {fps[0]: {"members": [fps[0], fps[4]], "selected": 1}}
    view = chat_ops.version_view(messages, fps, groups)
    assert view.hidden == {0, 1} and view.switchers == {4: (fps[0], 1, 2)}
    view = chat_ops.version_view(messages, fps, {fps[0]: {"members": [fps[0], fps[4]], "selected": 0}})
    assert view.hidden == {4, 5} and view.switchers == {0: (fps[0], 0, 2)}
    assert chat_ops.version_view(messages, fps, {fps[0]: {"members": [fps[0], "gone"], "selected": 1}}).hidden == set()
    assert chat_ops.turn_span(messages, 6) == [6, 7] and chat_ops.turn_span(messages, 1) == []
    inputs, outputs = chat_ops.jump_entries(messages, hidden={0, 1})
    assert [e.index for e in inputs] == [2, 4, 6] and inputs[2].label == "📎 book.epub — keep names"
    assert inputs[0].label == "1. second" and outputs[0].label == "1. Request 2"
    bodies = {1: "one", 3: "two", 5: "ONE again", 7: "chapter"}
    assert chat_ops.search_matches(messages, "one", bodies.get) == [1, 5]
    assert chat_ops.search_matches(messages, "keep", bodies.get) == [6]
    assert chat_ops.search_matches(messages, "one", bodies.get, hidden={1}) == [5]
    assert chat_ops.search_matches(messages, "  ", bodies.get) == []
    from glossarion_mobile.ui.chat.jump_to import step_target

    assert step_target(inputs, 4, -1) == 2 and step_target(inputs, 4, 1) == 6 and step_target(inputs, 6, 1) is None
    assert step_target(inputs, None, 1) == 2 and step_target([], 3, 1) is None
    text = chat_ops.transcript_markdown("Chat", messages[:4], lambda i: bodies.get(i, ""))
    assert text.startswith("# Chat\n") and "**You:**\n\nfirst" in text and "### GLOSSARION · Request 2\n\ntwo" in text
    assert chat_ops.safe_file_stem('a/b:c*') == "a_b_c_" and chat_ops.safe_file_stem("") == "Chat"


def test_mode_option_rules():
    from glossarion_mobile.ui.chat import mode_options_sheet as mos

    assert mos.generate_block_reason(False, False) == "Type a prompt in the composer first"
    assert mos.generate_block_reason(True, True) == "Remove the attachment to generate from a prompt"
    assert mos.generate_block_reason(True, False) is None
    values = {"use_custom_image_edit_endpoint": True, "custom_image_edit_endpoint": "https://edit.example/v1/images"}
    assert mos.image_edit_endpoint_status(lambda k, d=None: values.get(k, d)) == "Custom image-edit endpoint: On · edit.example"
    assert mos.image_edit_endpoint_status(lambda k, d=None: d) == "Custom image-edit endpoint: Off"
    assert set(mos.MODE_OPTION_KEYS) == {"vision", "image", "video", "audio", "refinement"}
    assert not hasattr(mos, "MODE_MILESTONES")  # no "arrives in U7" chip any more
    import settings_schema

    for mode, keys in mos.MODE_OPTION_KEYS.items():
        for key in keys:
            assert settings_schema.has_spec(key), (mode, key)
    import key_pool_service

    from glossarion_mobile.ui.screens.keys import SLUG_TO_POOL

    for slug, _label in mos.MODE_KEY_POOLS.values():
        assert SLUG_TO_POOL[slug] in key_pool_service.POOL_IDS


# ==========================================================================
# Store: media, delete message, versions, scratch chats, migrate
# ==========================================================================


def _add_media_response(adapter, cid, folder: Path, *names: str, kind: str = "IMAGE") -> int:
    files = []
    for name in names:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(PNG)
        files.append(path)
    content = "\n\n".join(f"[GENERATED_{kind}:{p}]" for p in files)
    [index] = adapter.append_messages(cid, [("assistant", content, "", "Processing", str(folder), "Request 9",
                                             {"created_at": "2026-10-06T10:00:00+09:00"})])
    return index


def test_message_media_uses_the_shared_rule(desktop_store_cls, tmp_path):
    _TC._desktop_history(tmp_path)
    adapter = _adapter(desktop_store_cls, tmp_path, save_delay=0.01)
    folder = _chat_folder(adapter)
    index = _add_media_response(adapter, "2", folder, "Direct Text 2.png", "Direct Text 3.png")
    media = adapter.message_media("2", index)
    assert [(k, os.path.basename(p), e) for k, p, e in media] == [
        ("image", "Direct Text 2.png", True), ("image", "Direct Text 3.png", True)]
    (folder / "Direct Text 3.png").unlink()
    assert adapter.message_media("2", index)[1][2] is False  # the card says "unavailable"
    assert adapter.message_media("2", 0) == []  # a user turn
    audio = folder / "voice.mp3"
    audio.write_bytes(b"ID3")
    [audio_index] = adapter.append_messages("2", [("assistant", f"[GENERATED_AUDIO:{audio}]", "", "", str(folder), "Request 10",
                                                   {"created_at": "x"})])
    assert adapter.message_media("2", audio_index) == [("audio", str(audio), True)]
    adapter.close()


def test_delete_messages_reindexes_and_remaps_the_sidecar(desktop_store_cls, tmp_path):
    history = _TC._desktop_history(tmp_path)
    adapter = _adapter(desktop_store_cls, tmp_path, save_delay=0.01)
    adapter.append_messages("2", [("user", "안녕"), ("assistant", "Hello again", "", "", "", "Request 3", {"created_at": "z"})])
    fps = adapter.fingerprints("2")
    assert adapter.add_version("2", 0, 5) == fps[0]
    adapter.set_meta("2", "add_only", [fps[2]])
    adapter.set_meta("2", "pending_plan", {"user_index": 2, "text": ""})
    assert adapter.expanded("2") == {1}
    assert adapter.delete_messages("2", [0, 1])  # the first turn and its response
    messages = adapter.messages("2")
    assert [m[0] for m in messages] == ["user_file", "assistant", "assistant", "user", "assistant"]
    assert adapter.expanded("2") == set()
    meta = adapter.meta("2")
    assert "versions" not in meta  # the group lost a member: one left, no versions
    new_fps = adapter.fingerprints("2")
    assert meta["add_only"] == [new_fps[0]] and meta["pending_plan"]["user_index"] == 0
    adapter.flush()
    on_disk = json.loads(history.read_text(encoding="utf-8"))
    session = next(s for s in on_disk["sessions"] if s["id"] == 2)
    assert [m[0] for m in session["messages"]] == ["user_file", "assistant", "assistant", "user", "assistant"]
    reloaded = _adapter(desktop_store_cls, tmp_path)  # the desktop store reads it back
    assert reloaded.message_text("2", 1, "content") == "Chapter one text"
    assert not adapter.delete_messages("2", [99])
    adapter.close()
    reloaded.close()


def test_delete_message_renames_the_surviving_response_files(desktop_store_cls, tmp_path):
    """The shared store names a response's managed files after its message index; Delete message
    moves the survivors' files to their new index, so the next response or an Edit translation at
    a reused index never writes over a surviving response (the desktop reads the same files)."""
    history = _TC._desktop_history(tmp_path)
    adapter = _adapter(desktop_store_cls, tmp_path, save_delay=0.01)
    cid = "5"

    def response(text):
        return ("assistant", text, f"thinking {text}", "Processing", "", "Request", {"created_at": f"t {text}"})

    adapter.record_user_turn(cid, ("user", "A"), "A")
    adapter.append_messages(cid, [response("RESPONSE A")])
    adapter.record_user_turn(cid, ("user", "B"), "B")
    adapter.append_messages(cid, [response("RESPONSE B")])
    adapter.flush()
    chat_messages = Path(adapter.output_folder(cid)) / "Chat Messages"
    names = sorted(p.name for p in chat_messages.iterdir())
    assert {n[:7] for n in names} == {"000002-", "000004-"} and len(names) == 10
    assert adapter.delete_messages(cid, [0, 1])
    # A's files are gone, B's moved from 000004 to 000002; the history saved with them
    names = sorted(p.name for p in chat_messages.iterdir())
    assert len(names) == 5 and all(n.startswith("000002-") for n in names)
    assert (chat_messages / "000002-response.md").read_text(encoding="utf-8") == "RESPONSE B"
    session = next(s for s in json.loads(history.read_text(encoding="utf-8"))["sessions"] if s["id"] == 5)
    storage = session["messages"][1][6]
    assert storage["content_path"].endswith("Chat Messages/000002-response.md")
    assert storage["thinking_path"].endswith("Chat Messages/000002-thinking.md")
    assert storage["content_html_path"].endswith("Chat Messages/000002-response.html")
    # a new turn takes index 3 (000004): it no longer lands on B's file
    adapter.record_user_turn(cid, ("user", "C"), "C")
    adapter.append_messages(cid, [response("RESPONSE C")])
    adapter.flush()
    adapter.forget_bodies()
    assert [adapter.message_text(cid, i) for i in range(4)] == ["B", "RESPONSE B", "C", "RESPONSE C"]
    assert adapter.message_text(cid, 1, "thinking") == "thinking RESPONSE B"
    # Edit translation of the moved response writes its own (index-named) files
    adapter.save_response_edit(cid, 1, "RESPONSE B edited")
    adapter.flush()
    adapter.forget_bodies()
    assert [adapter.message_text(cid, i) for i in (1, 3)] == ["RESPONSE B edited", "RESPONSE C"]
    # a single response deleted from its ⋯ menu: the later ones move down too
    assert adapter.delete_messages(cid, [1])
    adapter.forget_bodies()
    assert [adapter.message_text(cid, i) for i in range(3)] == ["B", "C", "RESPONSE C"]
    assert sorted(p.name[:7] for p in chat_messages.iterdir()) == ["000003-"] * 5
    adapter.close()
    reloaded = _adapter(desktop_store_cls, tmp_path)  # the desktop store reads the same thing back
    assert [reloaded.message_text(cid, i) for i in range(3)] == ["B", "C", "RESPONSE C"]
    reloaded.close()


def test_delete_message_keeps_job_cards_on_their_turns(desktop_store_cls, tmp_path):
    """Job cards find their job by turn index: Delete message records the turns of the chat's jobs
    (sidecar ``job_turns``), so a later turn keeps its own job and Resume targets the moved turn."""
    from glossarion_mobile.ui.chat.run_controller import ChatRun, turn_key
    from glossarion_mobile.ui.chat.run_request import DirectTextRun
    from glossarion_mobile.ui.chat.stream_bridge import RunStream

    _TC._desktop_history(tmp_path)
    adapter = _adapter(desktop_store_cls, tmp_path, save_delay=0.01)
    cid = "5"
    for name in ("a.txt", "b.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")

    def file_turn(name):
        return ("user_file", name, str(tmp_path / name), 1, "", "user")

    def card(label):
        return ("assistant", f"{label} out", "", "Processing", "", f"{label} · Request 1", {"created_at": label})

    adapter.append_messages(cid, [file_turn("a.txt"), card("A"), file_turn("b.txt"), card("B")])

    def run_dict(job_id):
        return DirectTextRun(temp_root=str(tmp_path / f"run_{job_id}"), source_path=str(tmp_path / "x.txt"),
                             source_extension=".txt", is_attachment=True, expected_output="").as_dict()

    def snap(job_id, index, state, when):
        params = {"chat_id": 5, "user_index": index, "run": run_dict(job_id)}
        spec = types.SimpleNamespace(kind="direct_text", params=params, title=job_id)
        return types.SimpleNamespace(id=job_id, spec=spec, state=state, created=when, started=when, finished=when,
                                     resolution=None, error=None)

    done_a, failed_b = snap("jobA", 0, "DONE", 1.0), snap("jobB", 2, "FAILED", 2.0)
    submitted = []

    async def submit(kind, title, inputs, params, origin):
        submitted.append(dict(params))
        return "jobB2"

    service = types.SimpleNamespace(view=lambda: types.SimpleNamespace(interrupted=(), history=(failed_b, done_a)))
    jobs = types.SimpleNamespace(jobs=service, resume=lambda job_id: None, submit=submit)
    runs = ChatRuns(adapter, jobs, temp_dir=str(tmp_path))
    # this session's last run: B's (ended)
    last = ChatRun(cid=cid, run=DirectTextRun.from_dict(run_dict("jobB")), stream=RunStream(), user_index=2,
                   job_id="jobB", params=dict(failed_b.spec.params), finished=True, state="failed")
    runs.runs[cid] = last
    assert runs.persisted_job(cid, 0) is done_a and runs.persisted_job(cid, 2) is failed_b
    assert runs.turn_index(cid, last) == 2
    assert adapter.delete_messages(cid, [0, 1], jobs=runs.chat_job_turns(cid))
    # B's turn is index 0 now: its card shows B's Failed job, never A's Done one
    assert runs.persisted_job(cid, 0) is failed_b and runs.persisted_job(cid, 2) is None
    assert runs.turn_index(cid, last) == 0
    turns = adapter.meta(cid)["job_turns"]
    assert turns[turn_key(done_a.spec.params)] is None and turns[turn_key(failed_b.spec.params)]
    # Resume / Retry: the new run belongs to the moved turn
    resumed = asyncio.run(runs.resubmit(cid))
    assert resumed.user_index == 0 and submitted[-1]["user_index"] == 0
    # a relaunch reads the turns back from the sidecar
    adapter.close()
    reloaded = _adapter(desktop_store_cls, tmp_path)
    again = ChatRuns(reloaded, jobs, temp_dir=str(tmp_path))
    assert again.persisted_job(cid, 0) is failed_b and again.persisted_job(cid, 2) is None
    # the turn of a deleted job's run is gone: nothing to resume into
    reloaded.delete_messages(cid, [0, 1], jobs=again.chat_job_turns(cid))
    service.view = lambda: types.SimpleNamespace(interrupted=(), history=(failed_b,))
    assert asyncio.run(again.resubmit(cid)) is None
    reloaded.close()


@needs_flet
def test_card_extras_load_off_the_loop_once_and_follow_the_newest_card():
    """Chat cards' OCR / Refine lookups run on the io pool, are cached per key and land on the card
    the newest render built (``ChatView._io_extra``)."""
    from glossarion_mobile.ui.chat.chat_view import ChatView

    calls, applied, threads = [], [], []
    import threading

    class Card:
        def __init__(self, name):
            self.name = name
            self.updates = 0

        def update(self):
            self.updates += 1

    async def run_io(fn, *args):
        return await asyncio.to_thread(fn, *args)

    async def scenario():
        tasks = []
        view = types.SimpleNamespace(cid="2", _extras={}, _extras_pending=set(), _extra_targets={}, _render_gen=1,
                                     env=types.SimpleNamespace(run_io=run_io),
                                     _spawn=lambda coro: tasks.append(asyncio.ensure_future(coro)))

        def compute():
            threads.append(threading.current_thread() is threading.main_thread())
            calls.append(1)
            return [("p1.txt", "하나")]

        first, second, twin = Card("first"), Card("second"), Card("twin")
        key = ("ocr", "2", "mid", (3,), None)
        ChatView._io_extra(view, key, compute, lambda value, c=first: applied.append((c.name, value)), first)
        view._render_gen = 2  # a re-render while the lookup runs: its cards get the value
        ChatView._io_extra(view, key, compute, lambda value, c=second: applied.append((c.name, value)), second)
        ChatView._io_extra(view, key, compute, lambda value, c=twin: applied.append((c.name, value)), twin)
        assert applied == [] and len(tasks) == 1  # one lookup in flight; nothing read on the loop
        await asyncio.gather(*tasks)
        assert calls == [1] and threads == [False]
        value = [("p1.txt", "하나")]
        assert applied == [("second", value), ("twin", value)] and second.updates == twin.updates == 1
        assert first.updates == 0  # the card the older render built is gone
        third = Card("third")
        ChatView._io_extra(view, key, compute, lambda value, c=third: applied.append((c.name, value)), third)
        assert applied[-1] == ("third", [("p1.txt", "하나")]) and calls == [1]  # cached: applied at once
        ChatView._drop_extras(view, "2")
        assert view._extras == {}

    asyncio.run(scenario())


@needs_flet
def test_single_generated_image_is_at_most_760_tall(tmp_path):
    from PIL import Image

    from glossarion_mobile.ui.chat import media_cards as mc
    from glossarion_mobile.ui.chat.media_model import MediaItem, capped_image_height, image_size

    tall, wide = tmp_path / "webtoon.png", tmp_path / "banner.png"
    Image.new("RGB", (100, 1000)).save(tall)
    Image.new("RGB", (1000, 100)).save(wide)
    assert image_size(str(tall)) == (100, 1000) and image_size(str(tmp_path / "gone.png")) is None
    assert capped_image_height(304.0, (100, 1000)) == 760.0 and capped_image_height(304.0, None) is None
    gallery = mc.ImageGallery([MediaItem("image", str(tall), True)], available_width=400)
    assert gallery.images[0].height == 760.0 and gallery.controls[0].height == 760.0
    gallery = mc.ImageGallery([MediaItem("image", str(wide), True)], available_width=400)
    assert abs(gallery.images[0].height - 30.4) < 1e-6  # shorter than the cap: its own aspect


def test_versions_add_select_and_hide(desktop_store_cls, tmp_path):
    _TC._desktop_history(tmp_path)
    adapter = _adapter(desktop_store_cls, tmp_path, save_delay=0.01)
    adapter.append_messages("2", [("user", "안녕 (edited)"), ("assistant", "Hi!", "", "", "", "Request 3", {"created_at": "q"})])
    anchor = adapter.add_version("2", 0, 5)
    groups = adapter.version_groups("2")
    assert groups == {anchor: {"members": [adapter.fingerprints("2")[0], adapter.fingerprints("2")[5]], "selected": 1}}
    view = chat_ops.version_view(adapter.messages("2"), adapter.fingerprints("2"), groups)
    assert view.hidden == {0, 1} and 5 in view.switchers
    assert adapter.select_version("2", anchor, 0) and not adapter.select_version("2", anchor, 0)
    view = chat_ops.version_view(adapter.messages("2"), adapter.fingerprints("2"), adapter.version_groups("2"))
    assert view.hidden == {5, 6}
    sidecar = json.loads(Path(adapter.sidecar_path).read_text(encoding="utf-8"))
    assert sidecar["chats"]["2"]["versions"][anchor]["selected"] == 0
    # a third version joins the same group (anchored on the original)
    adapter.append_messages("2", [("user", "third")])
    assert adapter.add_version("2", 5, 7) == anchor and len(adapter.version_groups("2")[anchor]["members"]) == 3
    adapter.close()


def test_a_cancelled_plan_turn_leaves_its_version_group(desktop_store_cls, tmp_path):
    """U7 review: Run again / Retranslate links the new user_file turn as a version when the Plan is
    created; Plan › Cancel truncates that turn, and the sidecar must forget it too. Otherwise a later
    ordinary send of the same file (same fingerprint) lands in the group and hides the original."""
    _TC._desktop_history(tmp_path)
    adapter = _adapter(desktop_store_cls, tmp_path, save_delay=0.01)
    cid = "2"
    messages = adapter.messages(cid)
    anchor = next(i for i, m in enumerate(messages) if m[0] == "user_file")
    file_turn = tuple(messages[anchor])
    adapter.set_meta(cid, "add_only", [adapter.fingerprints(cid)[0]])
    new_index = adapter.record_user_turn(cid, file_turn, file_turn[1])  # Run again -> Plan
    assert adapter.add_version(cid, anchor, new_index)
    adapter.truncate_messages(cid, new_index)  # Plan › Cancel
    assert adapter.version_groups(cid) == {}
    assert adapter.meta(cid)["add_only"] == [adapter.fingerprints(cid)[0]]  # the kept prefix stays
    sidecar = json.loads(Path(adapter.sidecar_path).read_text(encoding="utf-8"))
    assert "versions" not in sidecar["chats"][cid]
    again = adapter.record_user_turn(cid, file_turn, file_turn[1])  # later: the same book, sent normally
    messages = adapter.messages(cid)
    view = chat_ops.version_view(messages, message_fingerprints(messages), adapter.version_groups(cid))
    assert again == new_index and not view.hidden and not view.switchers
    adapter.close()


def test_scratch_chat_save_moves_the_folder_and_rewrites_paths(desktop_store_cls, tmp_path):
    history = _TC._desktop_history(tmp_path)
    adapter = _adapter(desktop_store_cls, tmp_path, save_delay=0.01)
    cid = adapter.new_scratch()
    assert cid.startswith("s") and adapter.is_scratch(cid) and adapter.session(cid)["messages"] == []
    summary = adapter.get(cid)
    assert summary.scratch and summary.title == "Scratch chat" and cid in [c.cid for c in adapter.all()]
    root = Path(adapter.scratch_dir) / cid[1:]
    assert root.is_dir()
    adapter.record_user_turn(cid, ("user", "scratch hello"), "scratch hello")
    folder = Path(adapter.output_folder(cid, create=True))
    assert folder.parent.name == "Direct Text" and root in folder.parents  # inside the scratch root
    (folder / "Direct Text 1.txt").write_text("scratch reply", encoding="utf-8")
    scratch_store = adapter.binding_for(cid).store
    reference = scratch_store.history_file_reference(str(folder / "Direct Text 1.txt"))
    adapter.append_messages(cid, [("assistant", "", "", "", str(folder), "Request 1",
                                   {"content_path": reference, "created_at": "2026-10-06T11:00:00+09:00"})])
    assert adapter.message_text(cid, 1, "content") == "scratch reply"
    adapter.set_override(cid, "model", "gpt-6-mini")
    adapter.flush()
    on_disk = json.loads(history.read_text(encoding="utf-8"))
    assert all(str(s["id"]) != cid for s in on_disk["sessions"]) and len(on_disk["sessions"]) == 2
    sidecar = json.loads(Path(adapter.sidecar_path).read_text(encoding="utf-8"))
    assert [i["cid"] for i in sidecar["scratch"]] == [cid] and cid not in sidecar["chats"]

    new_cid = adapter.save_scratch(cid)
    assert new_cid == "6" and not adapter.is_scratch(cid) and not root.exists()
    session = adapter.session(new_cid)
    target = Path(session["output_folder"])
    assert target.parent == tmp_path / "Output" / "Direct Text" and (target / "Direct Text 1.txt").is_file()
    assert adapter.message_text(new_cid, 1, "content") == "scratch reply"
    stored = session["messages"][1][6]["content_path"]
    assert not os.path.isabs(stored) and stored.replace("\\", "/").startswith("Output/Direct Text/")
    assert session["messages"][1][4] == str(target)
    assert adapter.overrides(new_cid) == {"model": "gpt-6-mini"} and adapter.current_cid() == new_cid
    on_disk = json.loads(history.read_text(encoding="utf-8"))
    assert any(s["id"] == 6 for s in on_disk["sessions"])
    assert json.loads(Path(adapter.sidecar_path).read_text(encoding="utf-8"))["scratch"] == []
    # the desktop store reads the saved chat
    reloaded = _adapter(desktop_store_cls, tmp_path)
    assert reloaded.message_text("6", 1, "content") == "scratch reply"
    reloaded.close()
    adapter.close()


def test_scratch_discard_duplicate_and_stale_cleanup(desktop_store_cls, tmp_path):
    _TC._desktop_history(tmp_path)
    adapter = _adapter(desktop_store_cls, tmp_path, save_delay=0.01)
    dup = adapter.duplicate_as_scratch("2")
    assert adapter.is_scratch(dup) and len(adapter.messages(dup)) == 5
    assert adapter.message_text(dup, 1, "content") == "안녕 → **Hello**"  # its own copy of the body
    assert adapter.delete_notice(dup)[0] == "Discard scratch chat?"
    root = Path(adapter.scratch_dir) / dup[1:]
    ok, _error = adapter.delete(dup)
    assert ok and not root.exists() and adapter.session(dup) is None
    stale = adapter.new_scratch()
    stale_root = Path(adapter.scratch_dir) / stale[1:]
    (stale_root / "leftover.txt").write_text("x", encoding="utf-8")
    adapter.flush()
    adapter.close()
    again = _adapter(desktop_store_cls, tmp_path)  # a relaunch: unsaved scratch chats are gone
    assert not stale_root.exists() and again.scratch_ids() == []
    assert json.loads(Path(again.sidecar_path).read_text(encoding="utf-8"))["scratch"] == []
    again.close()


def test_duplicate_as_scratch_owns_its_response_bodies(desktop_store_cls, tmp_path):
    """U7 review: the copy's response bodies are copies of the original's ``Chat Messages`` files.
    Delete message in the original renames / removes those files (``_rehome_bodies``), which must
    not change what the scratch copy (or the chat it is later saved as) shows."""
    _TC._desktop_history(tmp_path)
    adapter = _adapter(desktop_store_cls, tmp_path, save_delay=0.01)
    cid = "5"

    def response(text):
        return ("assistant", text, f"thinking {text}", "Processing", "", "Request", {"created_at": f"t {text}"})

    adapter.record_user_turn(cid, ("user", "A"), "A")
    adapter.append_messages(cid, [response("RESPONSE A")])
    adapter.record_user_turn(cid, ("user", "B"), "B")
    adapter.append_messages(cid, [response("RESPONSE B")])
    adapter.flush()
    adapter.forget_bodies()
    original_dir = Path(adapter.output_folder(cid)) / "Chat Messages"
    assert (original_dir / "000004-response.md").is_file()  # the store externalised the bodies
    scratch = adapter.duplicate_as_scratch(cid)
    texts = lambda c: [adapter.message_text(c, i) for i in range(4)]  # noqa: E731
    before = texts(scratch)
    assert before == ["A", "RESPONSE A", "B", "RESPONSE B"]
    scratch_root = Path(adapter.scratch_dir) / scratch[1:]
    storage = adapter.messages(scratch)[3][6]
    copy = Path(adapter.binding_for(scratch).resolve_reference(storage["content_path"]))
    assert copy.is_file() and scratch_root in copy.parents and copy.name == "000004-response.md"
    # Delete message in the original: its files are removed / renamed; the copy is unchanged
    assert adapter.delete_messages(cid, [0, 1])
    adapter.forget_bodies()
    assert not (original_dir / "000004-response.md").exists()
    assert texts(scratch) == before
    assert adapter.message_text(scratch, 3, "thinking") == "thinking RESPONSE B"
    # a scratch copy saved as a chat keeps its own bodies too
    saved = adapter.save_scratch(scratch)
    adapter.forget_bodies()
    assert saved and texts(saved) == before
    saved_ref = Path(adapter.binding_for(saved).resolve_reference(adapter.messages(saved)[1][6]["content_path"]))
    assert saved_ref.is_file() and not str(saved_ref).startswith(str(original_dir))
    adapter.close()


def test_migrate_attachment_and_delete_workspace(desktop_store_cls, tmp_path, monkeypatch):
    _TC._desktop_history(tmp_path)
    out = tmp_path / "Out2"
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(out))
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    adapter = _adapter(desktop_store_cls, tmp_path, save_delay=0.01)
    workspace = Path(adapter.attachment_folders("2")[0])
    (workspace / "book.epub").write_bytes(b"PK")
    (workspace / "translation_progress.json").write_text(json.dumps({"chapters": {
        "1": {"status": "completed"}, "2": {"status": "pending"}}}), encoding="utf-8")
    assert chat_ops.workspace_summary(str(workspace)) == "1/2 · EPUB ready"
    assert adapter.migration_target("2", str(workspace)) == str(out / "book")
    result = adapter.migrate_attachment("2", str(workspace))
    assert result["ok"] and (out / "book" / "book.epub").is_file() and not workspace.exists()
    assert adapter.messages("2")[3][4] == str(out / "book")  # the shared relocation
    assert adapter.attachment_folders("2") == []
    # a name collision: cancelled unless "Merge and replace"
    second = Path(adapter.output_folder("2")) / "Attachments" / "book"
    second.mkdir(parents=True)
    (second / "note.txt").write_text("x", encoding="utf-8")
    assert not adapter.migrate_attachment("2", str(second))["ok"] and second.exists()
    assert adapter.migrate_attachment("2", str(second), lambda target: True)["ok"]
    assert (out / "book" / "note.txt").is_file() and (out / "book" / "book.epub").is_file()
    third = Path(adapter.output_folder("2")) / "Attachments" / "other"
    third.mkdir(parents=True)
    ok, _ = adapter.delete_attachment_workspace("2", str(third))
    assert ok and not third.exists()
    ok, error = adapter.delete_attachment_workspace("2", str(out / "book"))
    assert not ok and "no longer a managed attachment" in error and (out / "book").exists()
    adapter.close()


# ==========================================================================
# Generate from prompt: ChatRuns + the generate_media job kind
# ==========================================================================


def _walk_controls(control):
    """Every control under ``control`` (content / controls / title / leading ...)."""
    yield control
    for name in ("content", "controls", "title", "subtitle", "leading", "trailing", "actions"):
        value = getattr(control, name, None)
        for child in value if isinstance(value, (list, tuple)) else ([value] if value is not None else []):
            if hasattr(child, "__dict__") and not isinstance(child, str):
                yield from _walk_controls(child)


@needs_flet
def test_attachments_guard_sees_every_job_writing_the_workspace(tmp_path):
    """U7 review (UI_SPEC §2.17 "blocked while a job is writing that workspace"): not only the chat's
    own run but any active / queued job whose folder is the workspace or inside it (the job card's
    Compile, ＋ › Retranslate chapters) blocks the merge and Delete workspace, per workspace.
    Device report #4: the cards have no Migrate button (finished books move by themselves)."""
    from glossarion_mobile.ui.chat.attachments import (
        BUSY_REASON,
        INTRO,
        JOBS_RUNNING_TEXT,
        MERGE_ACTION,
        AttachmentsScreen,
        job_writes_into,
        workspace_busy,
    )
    from glossarion_mobile.ui.chat.chat_view import ChatView
    from glossarion_mobile.ui.components.reason_chip import ReasonChip

    attachments = tmp_path / "Output" / "Direct Text" / "My novel" / "Attachments"
    book, other = attachments / "book", attachments / "other"
    book.mkdir(parents=True)
    other.mkdir()

    def snap(kind, inputs=(), output_dir=None, **params):
        return types.SimpleNamespace(spec=types.SimpleNamespace(kind=kind, inputs=tuple(inputs), params=params),
                                     output_dir=output_dir, output_dirs={})

    compile_job = snap("compile_epub", folder=str(book))
    retranslate_job = snap("retranslate", plan="token", count=2, output_dir=str(book / "sub"))
    unrelated = snap("translate", [str(tmp_path / "Inbox" / "book.epub")], output_dir=str(tmp_path / "Output" / "book"),
                     title="book")
    assert job_writes_into(compile_job, str(book)) and not job_writes_into(compile_job, str(other))
    assert job_writes_into(retranslate_job, str(book)) and not job_writes_into(unrelated, str(book))

    class Service:  # JobService.view(): the active job and the queue
        def view(self):
            return types.SimpleNamespace(active=unrelated, queue=(compile_job,))

    adapter = JobsAdapter(Service())
    assert adapter.pending() == [unrelated, compile_job]
    view = types.SimpleNamespace(env=types.SimpleNamespace(runs=types.SimpleNamespace(live_run=lambda cid: None),
                                                           jobs=adapter))
    assert ChatView._workspace_busy(view, "2", str(book)) and not ChatView._workspace_busy(view, "2", str(other))
    view.env.runs = types.SimpleNamespace(live_run=lambda cid: object())  # the chat's own run: every workspace
    assert ChatView._workspace_busy(view, "2", str(other))
    # the module rule the Attachments screen and the Library auto-migrate share; ended jobs never count
    idle_runs = types.SimpleNamespace(live_run=lambda cid: None)
    assert workspace_busy(idle_runs, adapter, "2", str(book)) and not workspace_busy(idle_runs, adapter, "2", str(other))
    assert workspace_busy(view.env.runs, adapter, "2", str(other))
    ended = types.SimpleNamespace(spec=compile_job.spec, output_dir=None, output_dirs={}, state="DONE")
    assert not workspace_busy(idle_runs, types.SimpleNamespace(pending=lambda: [ended]), "2", str(book))

    migrated, notes, merged_into = [], [], []
    targets = {str(other): ""}

    class Chats:
        def session(self, cid):
            return {"title": "My novel"}

        def attachment_folders(self, cid):
            return [str(book), str(other)]

        def messages(self, cid):
            return []

        def migration_target(self, cid, folder):
            return targets.get(folder, "")

        def migrate_attachment(self, cid, folder, confirm=None):
            migrated.append(folder)
            if confirm is not None:
                merged_into.append(folder)
            return {"ok": True, "notices": [{"title": "Attachment migrated",
                                             "text": f"The attachment workspace was moved to:\n{tmp_path / 'Out' / 'x'}"}]}

    async def io(fn, *args):
        return fn(*args)

    moved = []

    async def scenario():
        screen = AttachmentsScreen(None, chats=Chats(), cid="2", run_io=io, notify=notes.append,
                                   busy=lambda folder: job_writes_into(compile_job, folder),
                                   on_migrated=lambda target, source: moved.append(target))  # still accepted
        body = screen.get_body()
        assert body.controls[0].value == INTRO and "Library" in INTRO
        screen.reload_sync()
        busy_card = screen.cards[str(book)].content.controls
        free_card = screen.cards[str(other)].content.controls
        assert isinstance(busy_card[2].controls[0], ReasonChip) and busy_card[2].controls[0].reason == BUSY_REASON
        assert len(free_card) == 2  # name row + summary: no button row
        for folder in (book, other):  # no Migrate button (nor its destination long-press)
            for control in _walk_controls(screen.cards[str(folder)]):
                assert "migrate" not in str(getattr(control, "key", "") or "").lower()
                texts = [getattr(control, name, None) for name in ("content", "value", "text")]
                assert not any(isinstance(t, str) and "Migrate" in t for t in texts)
                assert getattr(control, "on_long_press_start", None) is None
        labels = [item.label for item in screen.more_items(str(other))]
        assert labels == ["Open in Reader", "Progress", "Share output", "Delete workspace"]
        # a different Library book already has the name: ⋯ offers the desktop merge dialog
        targets[str(other)] = str(tmp_path)  # exists
        items = {item.label: item for item in screen.more_items(str(other))}
        assert MERGE_ACTION in items and items[MERGE_ACTION].disabled_reason is None
        assert {i.label: i for i in screen.more_items(str(book))}["Delete workspace"].disabled_reason == BUSY_REASON
        dialog = screen.migrate(str(other))
        assert dialog is not None and dialog.title == "Attachment folder already exists" and migrated == []
        assert screen.migrate(str(book)) is None and notes[-1] == BUSY_REASON and migrated == []
        assert (await screen.run_migrate(str(book)))["ok"] is False and migrated == []
        assert (await screen.run_migrate(str(other), merge=True))["ok"] is True and migrated == [str(other)]
        assert merged_into == [str(other)] and moved == [str(tmp_path / "Out" / "x")]
        # the move never runs while a job owns the process environment (job_runner.JOB_LOCK)
        import threading

        import job_runner

        held, release = threading.Event(), threading.Event()

        def hold():
            with job_runner.JOB_LOCK:
                held.set()
                release.wait(10)

        holder = threading.Thread(target=hold)
        holder.start()
        assert held.wait(5)
        try:
            assert (await screen.run_migrate(str(other)))["ok"] is False and notes[-1] == JOBS_RUNNING_TEXT
            assert migrated == [str(other)]
        finally:
            release.set()
            holder.join(5)

    asyncio.run(scenario())


def test_generate_submits_a_generate_media_job_and_finish_promotes_the_media(desktop_store_cls, tmp_path):
    jobs = FakeJobService()
    _TC._desktop_history(tmp_path)
    adapter = _adapter(desktop_store_cls, tmp_path, save_delay=0.01)
    runs = ChatRuns(adapter, JobsAdapter(jobs), temp_dir=str(tmp_path / "runs"))
    runs.attach()
    run = asyncio.run(runs.generate("5", prompt="a red fox in snow", output_mode="image", settings=DirectTextSettings()))
    spec = jobs.submitted[0]
    assert spec.kind == "generate_media" and spec.inputs == ()
    assert spec.params["prompt"] == "a red fox in snow" and spec.params["options"]["selected_files"] == ["__generative_mode__"]
    assert spec.params["options"]["output_mode"] == "image" and spec.params["output_root"] == run.run.temp_root
    assert adapter.messages("5") == [("user", "a red fox in snow")] and run.kind == "generate_media"
    jobs.publish("RUNNING")
    assert run.state == "running"
    generated = Path(run.run.temp_root) / "generated_media_ab12cd34.png"
    generated.write_bytes(PNG)
    jobs.publish("DONE")
    _TC._finish_all(runs)
    assert not run.live
    messages = adapter.messages("5")
    assert messages[-1][0] == "assistant"
    content = adapter.message_text("5", len(messages) - 1, "content")
    persisted = Path(adapter.output_folder("5")) / "Direct Text 1.png"
    assert persisted.is_file() and f"[GENERATED_IMAGE:{persisted}]" in content
    assert adapter.message_media("5", len(messages) - 1) == [("image", str(persisted), True)]
    # Resume / Retry resubmits the same kind
    asyncio.run(runs.resubmit("5"))
    assert jobs.submitted[-1].kind == "generate_media" and jobs.submitted[-1].inputs == ()
    with pytest.raises(ValueError):
        asyncio.run(runs.generate("5", prompt="  ", output_mode="image", settings=DirectTextSettings()))
    adapter.close()


class _Shim:
    def __init__(self, value=""):
        self.value = value

    def text(self):
        return self.value

    def toPlainText(self):
        return self.value


def test_generate_media_job_sets_the_prompt_and_runs_the_generative_branch(tmp_path):
    with restored_env("OUTPUT_DIRECTORY", "OUTPUT_DIR", "DIRECT_TEXT_ACTIVE", "DIRECT_TEXT_PRESERVE_MARKUP",
                      "DIRECT_TEXT_ORDERED_BATCH", "ORDER_BATCH_REQUESTS_BY_SPINE"):
        _check_generate_media_job(tmp_path)


def _check_generate_media_job(tmp_path):
    import image_job

    from glossarion_mobile.job_kinds import generate_media

    calls = []

    class Owner:
        def _prepare_translation_run(self, files=None):
            calls.append(("prepare", list(files), getattr(self, image_job.GENERATIVE_PROMPT_ATTR, None),
                          self.output_mode_var, list(self.selected_files)))
            return {"request": 1}

        def _translation_worker(self, request):
            calls.append(("worker", request))
            return True

        def _run_generative_prompt_mode(self):
            return True

        def _apply_forced_streaming_environment(self):
            calls.append(("forced",))

        def _apply_direct_text_runtime_environment(self):
            calls.append(("runtime",))

    results = {}
    ctx = types.SimpleNamespace(
        owner=Owner(), inputs=(), log=lambda text: None, phase=lambda label: None,
        stop_requested=lambda: False, set_result=lambda **data: results.update(data),
        params={"prompt": "  a red fox  ", "output_root": str(tmp_path), "is_attachment": False,
                "options": {"output_mode": "video", "selected_files": ["/whatever.txt"]}},
    )
    assert generate_media.run(ctx) == {"ok": True, "outputs": [], "error": None}
    assert calls[0] == ("forced",) and calls[1] == ("runtime",)
    assert calls[2] == ("prepare", ["__generative_mode__"], "a red fox", "video", ["__generative_mode__"])
    assert calls[3] == ("worker", {"request": 1}) and os.environ["OUTPUT_DIRECTORY"] == str(tmp_path)
    assert "run_env" in results
    ctx.params = {**ctx.params, "prompt": " "}
    from glossarion_mobile.services.jobs import JobError

    with pytest.raises(JobError):
        generate_media.run(ctx)
    assert generate_media.KINDS["generate_media"]["verb"] == "Generating"


def test_generate_media_and_translate_image_run_on_the_job_service(tmp_path, monkeypatch):
    """The two kinds through the real JobService (registration in KIND_MODULES / JobKind is the
    integration step; the test registers them itself)."""
    spec = importlib.util.spec_from_file_location("_glossarion_tj_helpers_media", Path(__file__).with_name("test_jobs.py"))
    tj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tj)
    from glossarion_mobile import job_kinds
    from glossarion_mobile.services.jobs import JobSpec, JobState

    monkeypatch.setitem(job_kinds.KIND_MODULES, "generate_media", "generate_media")
    monkeypatch.setitem(job_kinds.KIND_MODULES, "translate_image", "image")
    for name, value in (("_run_generative_prompt_mode", lambda self: True),
                        ("_apply_forced_streaming_environment", lambda self: None),
                        ("_apply_direct_text_runtime_environment", lambda self: None)):
        monkeypatch.setattr(tj.FakeOwner, name, value, raising=False)
    service, backend = tj.make_service(tmp_path)
    seen = []

    def worker(owner, request):
        seen.append((request["files"], getattr(owner, "_generative_prompt_override", None),
                     getattr(owner, "output_mode_var", None)))
        return True

    backend.behavior = worker
    picture = tmp_path / "page.png"
    picture.write_bytes(PNG)
    try:
        with restored_env("OUTPUT_DIRECTORY", "OUTPUT_DIR", "DIRECT_TEXT_ACTIVE", "DIRECT_TEXT_PRESERVE_MARKUP",
                          "DIRECT_TEXT_ORDERED_BATCH", "ORDER_BATCH_REQUESTS_BY_SPINE", "OUTPUT_MODE",
                          "ENABLE_IMAGE_OUTPUT_MODE", "ENABLE_VIDEO_OUTPUT_MODE", "ENABLE_AUDIO_OUTPUT_MODE",
                          "ENABLE_REFINEMENT_OUTPUT_MODE", "ENABLE_IMAGE_TRANSLATION"):
            generate = service.submit(JobSpec("generate_media", "Chat", (), params={
                "prompt": "a red fox", "output_root": str(tmp_path / "run"), "is_attachment": False,
                "options": {"output_mode": "audio"}}))
            assert service.wait_idle(tj.TIMEOUT)
            assert state_of(service, generate) is JobState.DONE, service.snapshot(generate).error
            translate = service.submit(JobSpec("translate_image", "page.png", (str(picture),),
                                               params={"output_mode": "vision"}))
            assert service.wait_idle(tj.TIMEOUT)
            assert state_of(service, translate) is JobState.DONE, service.snapshot(translate).error
            assert os.environ.get("DIRECT_TEXT_ACTIVE") is None  # the job scope restored the environment
    finally:
        service.close()
    assert seen == [(["__generative_mode__"], "a red fox", "audio"), ([str(picture)], None, "vision")]


def state_of(service, job_id):
    snap = service.snapshot(job_id)
    return snap.state if snap is not None else None


def test_the_shared_generative_run_sends_the_composer_text(monkeypatch, tmp_path):
    """image_job's hook: with GENERATIVE_PROMPT_ATTR set the request's prompt is the composer text;
    without it the desktop source (the prompt editor) is used."""
    monkeypatch.setenv("GLOSSARION_DATA_DIR", str(tmp_path))  # Generated_Media lands in a scratch dir
    with restored_env("ENABLE_IMAGE_OUTPUT_MODE", "ENABLE_VIDEO_OUTPUT_MODE", "NANOGPT_VIDEO_DURATION",
                      "NANOGPT_VIDEO_RESOLUTION", "OPENAI_API_KEY"):
        _check_generative_run(monkeypatch, tmp_path)


def _check_generative_run(monkeypatch, tmp_path):
    import image_job
    import unified_api_client

    from glossarion_mobile.job_kinds.generate_media import apply_prompt

    sent = []
    image = tmp_path / "out.png"
    image.write_bytes(PNG)

    class FakeClient:
        def __init__(self, api_key=None, model=None, **kwargs):
            self.model = model

        def send(self, messages, **kwargs):
            sent.append(messages)
            return f"[GENERATED_IMAGE:{image}]", "stop"

    monkeypatch.setattr(unified_api_client, "UnifiedClient", FakeClient)

    class Owner(image_job.ImageJobMixin):
        model_var = "gpt-image-2"
        config: dict = {}
        prompt_text = _Shim("desktop system prompt")
        api_key_entry = _Shim("")
        trans_temp = _Shim("0.7")

        def append_log(self, message):
            pass

        def _get_allowed_image_output_mode(self):
            return "1"

        def _get_allowed_video_output_mode(self):
            return "0"

    owner = Owner()
    assert apply_prompt(owner, "  a red fox  ") == image_job.GENERATIVE_PROMPT_ATTR
    assert owner._run_generative_prompt_mode() is True
    assert sent[-1] == [{"role": "user", "content": "a red fox"}]
    delattr(owner, image_job.GENERATIVE_PROMPT_ATTR)
    assert owner._run_generative_prompt_mode() is True
    assert sent[-1] == [{"role": "user", "content": "desktop system prompt"}]


def test_translate_image_kind_selects_the_output_mode(tmp_path):
    with restored_env("ENABLE_IMAGE_OUTPUT_MODE", "ENABLE_VIDEO_OUTPUT_MODE", "ENABLE_AUDIO_OUTPUT_MODE",
                      "ENABLE_REFINEMENT_OUTPUT_MODE", "ENABLE_IMAGE_TRANSLATION", "OUTPUT_MODE"):
        _check_translate_image(tmp_path)


def _check_translate_image(tmp_path):
    from glossarion_mobile.job_kinds import image
    from glossarion_mobile.services.jobs import JobError

    picture = tmp_path / "page.png"
    picture.write_bytes(PNG)
    generated = tmp_path / "page_generated.png"
    generated.write_bytes(PNG)
    seen = {}

    class Owner:
        config: dict = {}

        def _prepare_translation_run(self, files=None):
            seen["files"] = list(files)
            seen["mode"] = (self.output_mode_var, self.enable_image_output_mode_var, self.enable_image_translation_var)
            return {"r": 1}

        def _translation_worker(self, request):
            self.generated_images = [str(generated), str(tmp_path / "gone.png")]
            return True

    owner = Owner()
    ctx = types.SimpleNamespace(owner=owner, inputs=(str(picture),), params={"output_mode": "image"}, output_dir=None,
                                log=lambda t: None, phase=lambda l: None, stop_requested=lambda: False,
                                set_output_dirs=lambda m: None)
    result = image.run(ctx)
    assert seen == {"files": [str(picture)], "mode": ("image", True, True)} and result["ok"] is True
    assert result["outputs"] == [str(generated)] and os.environ["OUTPUT_MODE"] == "image"
    assert owner.config["output_mode"] == "image"
    text = tmp_path / "notes.txt"
    text.write_text("x", encoding="utf-8")
    ctx.inputs = (str(text),)
    with pytest.raises(JobError):
        image.run(ctx)
    assert ".mp4" in image.media_input_extensions() and ".png" in image.media_input_extensions()


# ==========================================================================
# Flet controls
# ==========================================================================


def _media_files(tmp_path):
    names = {"a.png": PNG, "b.png": PNG, "clip.mp4": b"\x00\x00\x00 ftyp", "voice.mp3": b"ID3"}
    paths = {}
    for name, data in names.items():
        path = tmp_path / name
        path.write_bytes(data)
        paths[name] = str(path)
    return paths


@needs_flet
def test_media_cards_and_the_audio_hub(tmp_path, monkeypatch):
    from glossarion_mobile.ui.chat import media_cards as mc
    from glossarion_mobile.ui.chat.media_model import MediaItem

    paths = _media_files(tmp_path)
    opened, saved, shared, external = [], [], [], []
    actions = mc.MediaActions(open_viewer=lambda items, i: opened.append((len(items), i)), save=saved.append,
                              share=shared.append, open_external=external.append)
    items = [MediaItem("image", paths["a.png"], True), MediaItem("image", paths["b.png"], True),
             MediaItem("image", str(tmp_path / "gone.png"), False), MediaItem("video", paths["clip.mp4"], True),
             MediaItem("audio", paths["voice.mp3"], True)]
    section = mc.media_section(items, actions=actions, available_width=400)
    kinds = [type(c).__name__ for c in section.controls]
    assert kinds == ["ImageGallery", "VideoCard", "AudioCard"]
    gallery = section.controls[0]
    assert len(gallery.images) == 2 and any(str(getattr(c, "key", "")).startswith("media-missing") for c in gallery.controls)
    gallery.open(1)
    assert opened == [(2, 1)]
    assert mc.image_width(400) == 304.0 and mc.image_width(100) == 280.0 and mc.image_width(5000) == 980.0
    menu = actions.menu_items(items[3])
    assert [i.label for i in menu] == ["Save as…", "Share", "Open externally"]
    menu[1].on_select()
    assert shared == [paths["clip.mp4"]]
    assert actions.menu_items(items[2])[0].disabled_reason == "The generated file is missing"
    video = section.controls[1]
    assert video.player is not None and video.player.aspect_ratio == 16 / 9 and not video.player.autoplay
    monkeypatch.setitem(sys.modules, "flet_video", None)  # a build without flet-video
    fallback = mc.VideoCard(items[3], actions=actions)
    assert fallback.player is None and not mc.video_available()

    class FakeAudio:
        def __init__(self):
            self.calls = []
            self.src = ""
            self.volume = 0.75

        def update(self):
            pass

        async def play(self, position=0):
            self.calls.append(("play", self.src, position))

        async def pause(self):
            self.calls.append(("pause",))

        async def resume(self):
            self.calls.append(("resume",))

        async def seek(self, position):
            self.calls.append(("seek", position))

    hub = mc.AudioHub()
    fake = FakeAudio()
    monkeypatch.setattr(hub, "_ensure", lambda: fake)
    card = mc.AudioCard(items[4], hub=hub, actions=actions)
    second = tmp_path / "second.mp3"
    second.write_bytes(b"ID3")
    other = mc.AudioCard(MediaItem("audio", str(second), True), hub=hub, actions=actions)
    assert card.volume == 0.75 and card.volume_slider.value == 0.75 and not card.volume_slider.visible

    async def scenario():
        assert await hub.toggle(card) == "playing" and fake.calls[-1] == ("play", paths["voice.mp3"], 0)
        hub._on_duration(types.SimpleNamespace(duration=types.SimpleNamespace(in_milliseconds=61_000)))
        hub._on_position(types.SimpleNamespace(position=5_000))
        assert card.time_text.value == "0:05 / 1:01" and card.seek.max == 61_000
        assert await hub.toggle(card) == "paused" and card.play_button.tooltip == "Play"
        assert await hub.toggle(card) == "playing"
        await hub.seek(card, 30_000)
        assert fake.calls[-1] == ("seek", 30_000)
        # a transcript re-render (or the MediaViewer) builds a new card for the playing file: it
        # adopts the playback, follows the events, and its Play/Pause acts on the same playback
        rebuilt = mc.AudioCard(MediaItem("audio", paths["voice.mp3"], True), hub=hub, actions=actions)
        assert rebuilt.state == "playing" and rebuilt.play_button.tooltip == "Pause" and hub.active is rebuilt
        assert rebuilt.time_text.value == "0:30 / 1:01" and rebuilt.seek.value == 30_000
        hub._on_position(types.SimpleNamespace(position=31_000))
        assert rebuilt.time_text.value == "0:31 / 1:01" and card.time_text.value == "0:31 / 1:01"
        plays = len([c for c in fake.calls if c[0] == "play"])
        assert await hub.toggle(rebuilt) == "paused" and fake.calls[-1] == ("pause",)
        assert rebuilt.state == card.state == "paused"
        assert await hub.toggle(rebuilt) == "playing" and fake.calls[-1] == ("resume",)
        assert len([c for c in fake.calls if c[0] == "play"]) == plays  # never restarted from 0:00
        # another file: the playing one stops, the new one plays from its own position
        assert await hub.toggle(other) == "playing" and card.state == rebuilt.state == "stopped"
        assert hub.active is other and fake.calls[-1] == ("play", str(second), 0)
        hub.set_volume(other, 0.3)
        assert fake.volume == 0.3

    asyncio.run(scenario())
    card._toggle_volume()
    assert card.volume_slider.visible
    broken = mc.AudioHub()
    monkeypatch.setattr(broken, "_ensure", lambda: (_ for _ in ()).throw(ImportError("no flet_audio")))
    lonely = mc.AudioCard(items[4], hub=broken, actions=actions)
    assert asyncio.run(broken.toggle(lonely)) == "unavailable" and lonely.unavailable.visible
    missing = mc.AudioCard(MediaItem("audio", str(tmp_path / "gone.mp3"), False), actions=actions)
    assert missing.play_button.disabled
    sheet = mc.CompareSheet("<p>Old</p>\n\nSame", "New\n\nSame")
    assert [tag for tag, _o, _n in sheet.blocks] == ["replace", "equal"]
    # Vision: the job card's OCR section and the image attachment's source thumbnail
    from glossarion_mobile.ui.chat.cards import JobCard
    from glossarion_mobile.ui.chat.messages import UserFileCard

    job = JobCard(attachment={"name": "a.png", "extension": ".png", "size": 3})
    assert not job.ocr_tile.visible
    job.set_ocr([("p1.txt", "하나"), ("p2.txt", "둘")])
    assert job.ocr_tile.visible and job.ocr_tile.title == "OCR (2)"
    card = UserFileCard("a.png", paths["a.png"], 3, thumbnail=True)
    assert any(getattr(c, "key", None) == "source-thumbnail" for c in card.card.content.controls)
    plain = UserFileCard("a.png", str(tmp_path / "gone.png"), 3, thumbnail=True, missing=True)
    assert not any(getattr(c, "key", None) == "source-thumbnail" for c in plain.card.content.controls)


@needs_flet
def test_media_viewer(tmp_path):
    from glossarion_mobile.ui.chat.media_model import MediaItem
    from glossarion_mobile.ui.components.media_viewer import MAX_ZOOM, MediaViewer

    paths = _media_files(tmp_path)
    calls = []
    images = [MediaItem("image", paths["a.png"], True), MediaItem("image", paths["b.png"], True)]
    viewer = MediaViewer(images, 1, on_close=lambda: calls.append("close"), on_save=lambda p: calls.append(("save", p)),
                         on_share=lambda p: calls.append(("share", p)), on_open_external=lambda p: calls.append(("ext", p)))
    assert viewer.pages is not None and len(viewer.pages.controls) == 2 and viewer.pages.selected_index == 1
    assert viewer.title.value == "2 / 2 · b.png" and viewer.viewers[0].max_scale == MAX_ZOOM
    viewer.show_index(0)
    assert viewer.title.value == "1 / 2 · a.png"
    viewer._act(viewer.on_share)
    viewer._act(viewer.on_save)
    viewer.close()
    assert calls == [("share", paths["a.png"]), ("save", paths["a.png"]), "close"]
    single = MediaViewer([images[0]])
    assert single.pages is None and single.title.value == "a.png"
    video = MediaViewer([MediaItem("video", paths["clip.mp4"], True)])
    assert type(video.body.content).__name__ == "VideoCard"
    audio = MediaViewer([MediaItem("audio", paths["voice.mp3"], True)])
    assert type(audio.body.content).__name__ == "AudioCard"


@needs_flet
def test_mode_options_content(monkeypatch):
    from glossarion_mobile.ui.chat import mode_options_sheet as mos

    generated, scoped, keys = [], [], []
    content = mos.ModeOptionsContent("image", has_text=True, on_generate=generated.append, on_this_chat=scoped.append,
                                     on_open_keys=keys.append, config_get=lambda k, d=None: d)
    assert content.generate_button is not None and not content.generate_button.disabled
    content._generate()
    assert generated == ["image"]
    content.this_chat_switch.value = True
    content._this_chat_changed()
    assert scoped == [True]
    content._open_keys("inpainter")
    assert keys == ["inpainter"]
    blocked = mos.ModeOptionsContent("audio", has_text=True, has_attachment=True, on_generate=generated.append)
    assert blocked.generate_button.disabled and blocked._generate() is None and generated == ["image"]
    monkeypatch.setattr(mos, "_google_tts_available", lambda: False)
    audio = mos.ModeOptionsContent("audio")
    labels = [getattr(c, "reason", None) for c in audio.column.controls]
    assert mos.TTS_SDK_REASON in labels
    assert mos.ModeOptionsContent("refinement").generate_button is None
    sheet = mos.ModeOptionsSheet("video", has_text=True, on_generate=generated.append)
    assert sheet.generate() is None or generated[-1] == "video"


# ==========================================================================
# The chat on the app shell
# ==========================================================================


@needs_flet
def test_chat_u7_on_the_app_shell(app_env, desktop_store_cls, tmp_path):
    from glossarion_mobile.ui.chat.attachments import AttachmentsScreen
    from glossarion_mobile.ui.chat.integration import ChatFeature
    from glossarion_mobile.ui.chat.media_cards import ImageGallery
    from glossarion_mobile.ui.chat.messages import VersionSwitcher
    from glossarion_mobile.ui.chat.send_state import SendAction

    tf = _TC._load_foundations()
    _TC._desktop_history(tmp_path)

    async def scenario():
        _m, conn, session, page, app = await tf._start("android")
        try:
            await tf._wait(lambda: app.state.engine_ready)
            adapter = _adapter(desktop_store_cls, tmp_path, save_delay=0.01)
            jobs = FakeJobService()
            oauth = _TC._FakeOAuth()
            oauth.signed_in.add("authgpt")
            feature = await ChatFeature.install(app, chats=adapter, jobs=jobs, oauth=oauth)
            await feature.refresh_sign_in()
            view = app.chat_view
            assert view.header.menu_items["attachments"].content == "Attachments (1)"
            view.composer.set_text("")  # the desktop history restored a draft

            # ---- Image mode: options, "This chat only", Generate from prompt -------------------
            assert view.composer.output_row.tap("image") == "selected"
            sheet = view.open_mode_options("image")
            assert sheet.generate_button.disabled  # the composer is empty
            sheet.this_chat_switch.value = True
            sheet.content._this_chat_changed()
            assert adapter.overrides("2")["output_mode"] == "image"
            view.composer.set_text("a red fox in snow")
            sheet = view.open_mode_options("image")
            assert not sheet.generate_button.disabled and sheet.this_chat_switch.value is True
            sheet.generate()
            await tf._wait(lambda: jobs.submitted)
            spec = jobs.submitted[-1]
            assert spec.kind == "generate_media" and spec.params["prompt"] == "a red fox in snow"
            assert view.composer.text == "" and adapter.messages("2")[-1] == ("user", "a red fox in snow")
            jobs.publish("RUNNING")
            run = feature.runs.run_for("2")
            (Path(run.run.temp_root) / "generated_media_1.png").write_bytes(PNG)
            jobs.publish("DONE")
            await tf._wait(lambda: not run.live, timeout=10)
            await tf._wait(lambda: any(getattr(c, "media", None) for c in view.transcript.cards), timeout=5)
            card = next(c for c in view.transcript.cards if getattr(c, "media", None))
            assert card.media_box.visible and isinstance(card.media_box.content.controls[0], ImageGallery)
            assert "GENERATED_IMAGE" not in card.content_md.value
            gallery = card.media_box.content.controls[0]
            viewer = gallery.open(0)
            assert viewer is view.media_viewer and viewer.title.value.endswith(".png")
            viewer.close()
            sheet = view._message_more(card)
            assert "Save media as…" in [i.label for i in sheet.items]
            view.set_mode_scope(False)
            assert "output_mode" not in adapter.overrides("2")
            assert app.config_store.get("direct_text_output_mode") == "image"
            view.composer.output_row.tap("text")

            # ---- Edit & resend -> a version; the switcher shows the selected one -------------
            first_user = 0
            view.edit_and_resend(first_user, "안녕 다시")
            assert view.composer.text == "안녕 다시" and view.version_anchor == first_user
            submitted = len(jobs.submitted)
            view.composer.send_button.tap()
            await tf._wait(lambda: len(jobs.submitted) > submitted)
            await tf._wait(lambda: adapter.version_groups("2"))
            jobs.publish("CANCELLED")
            await tf._wait(lambda: not feature.runs.run_for("2").live, timeout=10)
            view.render_transcript()
            switchers = [c for c in view.transcript.messages if isinstance(c, VersionSwitcher)]
            assert len(switchers) == 1 and switchers[0].label.value == "2/2"
            assert 0 in view.hidden_indices
            switchers[0].step(-1)
            switchers = [c for c in view.transcript.messages if isinstance(c, VersionSwitcher)]
            assert switchers[0].label.value == "1/2" and 0 not in view.hidden_indices

            # ---- Delete message ------------------------------------------------------------
            count = len(adapter.messages("2"))
            dialog = view.confirm_delete_messages([count - 1])
            await dialog._on_confirm()
            assert len(adapter.messages("2")) == count - 1

            # ---- Jump to, Search, Export -----------------------------------------------------
            jump = view.open_jump_to()
            assert jump.inputs and jump.outputs and jump.input_label.value.startswith("Input ")
            bar = view.open_search()
            assert view.header.searching and view.header.title_slot.content is bar
            view._on_search_query("Hello")
            await tf._wait(lambda: view.search_hits, timeout=5)
            assert bar.count.value.endswith(f"/{len(view.search_hits)}")
            view.close_search()
            assert not view.header.searching
            markdown = await view.env.run_io(view.build_export, "2", "markdown")
            assert Path(markdown).read_text(encoding="utf-8").startswith("# My novel")
            archive = await view.env.run_io(view.build_export, "2", "zip")
            with zipfile.ZipFile(archive) as zf:
                assert "direct_text_chats.json" in zf.namelist()

            # ---- Attachments manager -----------------------------------------------------------
            screen = feature.make_screen(tf.parse_route("/chat/2/attachments"))
            assert isinstance(screen, AttachmentsScreen) and screen.title == "Attachments — My novel"
            screen.get_body()
            assert screen.cards == {}  # the first paint lists nothing on the loop
            screen.did_show()
            await tf._wait(lambda: len(screen.cards) == 1, timeout=5)

            # ---- Scratch chats (the drawer's "New scratch chat": app.py wiring) -------------------
            scratch_button = app.drawer.header.content.controls[3]
            assert scratch_button.tooltip == "New scratch chat"
            cid = scratch_button.on_click(None)
            await tf._wait(lambda: view.cid == cid)
            assert view.header.is_scratch and view.header.save_scratch_button.visible
            assert view.transcript.messages and view.transcript.messages[0].key == "scratch-banner"
            view.composer.set_text("scratch text")
            submitted = len(jobs.submitted)
            view.composer.send_button.tap()
            await tf._wait(lambda: len(jobs.submitted) > submitted)
            assert jobs.submitted[-1].params["chat_id"] == cid
            jobs.publish("CANCELLED")
            await tf._wait(lambda: not feature.runs.run_for(cid).live, timeout=10)
            view.save_scratch()
            await tf._wait(lambda: not view.header.is_scratch, timeout=5)
            assert not adapter.is_scratch(cid) and view.cid.isdigit()
            assert adapter.messages(view.cid)[0] == ("user", "scratch text")

            # Send as scratch from a normal chat
            view.composer.set_text("one-off")
            submitted = len(jobs.submitted)
            view.on_send_action(SendAction.SEND_AS_SCRATCH)
            await tf._wait(lambda: len(jobs.submitted) > submitted)
            assert adapter.is_scratch(view.cid) and str(jobs.submitted[-1].params["chat_id"]).startswith("s")
            jobs.publish("CANCELLED")
            await tf._wait(lambda: not feature.runs.run_for(view.cid).live, timeout=10)

            # ＋ sheet: the active mode's options inline; Retranslate chapters opens the Progress manager
            plus = view.open_plus_sheet()
            assert plus.mode_options.content is not None
            plus.close()
            app._open_chat("2")
            await tf._wait(lambda: view.cid == "2")
            workspace = Path(adapter.attachment_folders("2")[0])
            (workspace / "translation_progress.json").write_text(json.dumps({"chapters": {}}), encoding="utf-8")
            bid = await view.open_retranslate()
            assert bid
            await tf._wait(lambda: tf._routes(page)[-1].startswith("/tools/progress?out="), timeout=5)
            # drawer row actions: Attachments / Export chat / Duplicate as scratch (Save / Discard for scratch)
            row = adapter.get("2")
            labels = {i.label: i for i in feature.chat_actions(row).items}
            assert all(labels[k].disabled_reason is None for k in ("Attachments (1)", "Export chat", "Duplicate as scratch"))
            scratch_row = adapter.get(adapter.new_scratch())
            assert {"Save", "Discard"} <= {i.label for i in feature.chat_actions(scratch_row).items}
            feature.close()
        finally:
            await tf._stop(app)

    asyncio.run(scenario())


class _FakeAudio:
    """``flet_audio.Audio`` stand-in: records the calls the AudioHub makes."""

    def __init__(self):
        self.calls = []
        self.src = ""
        self.volume = 0.75

    def update(self):
        pass

    async def play(self, position=0):
        self.calls.append(("play", self.src, position))

    async def pause(self):
        self.calls.append(("pause",))

    async def resume(self):
        self.calls.append(("resume",))

    async def seek(self, position):
        self.calls.append(("seek", position))


def test_transcript_cards_stay_live_across_rerenders(app_env, desktop_store_cls, tmp_path, monkeypatch):
    """Flet 1.0.3 freezes a new control matched to an old one by key, so a card rebuilt under its
    ScrollKey could never be updated again. The transcript keeps every card in a stable CardSlot and
    reuses unchanged cards: audio playback, Copy ✓, a Vision run's OCR section after a run ends and
    the jump highlight all keep working after any number of re-renders."""
    import glossarion_mobile.ui.chat.chat_view as chat_view_module
    from glossarion_mobile.ui.chat.cards import JobCard
    from glossarion_mobile.ui.chat.integration import ChatFeature
    from glossarion_mobile.ui.chat.media_cards import AudioCard
    from glossarion_mobile.ui.chat.messages import AssistantMessage
    from glossarion_mobile.ui.chat.transcript import CardSlot

    monkeypatch.setattr(chat_view_module, "HIGHLIGHT_SECONDS", 0.2)
    tf = _TC._load_foundations()
    _TC._desktop_history(tmp_path)
    workspace = tmp_path / "Output" / "Direct Text" / "My novel - 20261001_101010_abcdef12" / "Attachments" / "book"
    (workspace / "OCR" / "single").mkdir(parents=True)
    (workspace / "OCR" / "single" / "page1.txt").write_text("페이지 하나 OCR", encoding="utf-8")

    def job_card(view):
        return next(c for c in view.transcript.cards if isinstance(c, JobCard))

    def audio_card(view):
        for card in view.transcript.cards:
            box = getattr(card, "media_box", None)
            if box is not None and box.content is not None:
                return next(c for c in box.content.controls if isinstance(c, AudioCard))
        return None

    async def scenario():
        _m, conn, session, page, app = await tf._start("android")
        try:
            await tf._wait(lambda: app.state.engine_ready)
            adapter = _adapter(desktop_store_cls, tmp_path, save_delay=0.01)
            folder = _chat_folder(adapter)
            voice = folder / "voice.mp3"
            voice.write_bytes(b"ID3")
            jobs = FakeJobService()
            oauth = _TC._FakeOAuth()
            oauth.signed_in.add("authgpt")
            feature = await ChatFeature.install(app, chats=adapter, jobs=jobs, oauth=oauth)
            view = app.chat_view
            await tf._wait(lambda: job_card(view).ocr_tile.visible, timeout=5)
            adapter.append_messages("2", [("user", "say it")])
            adapter.append_messages("2", [("assistant", f"[GENERATED_AUDIO:{voice}]", "", "", str(folder),
                                           "Request 10", {"created_at": "2026-10-06T10:00:00+09:00"})])
            view.render_transcript(follow=True)
            await asyncio.sleep(0.3)  # the store externalises the new body
            fake = _FakeAudio()
            view.audio_hub._ensure = lambda: fake

            # ---- two re-renders: the same slots and (unchanged) the same card objects ---------------
            view.render_transcript()
            slots = list(view.transcript.messages)
            cards = list(view.transcript.cards)
            view.render_transcript()
            assert all(isinstance(s, CardSlot) for s in view.transcript.messages)
            assert view.transcript.messages == slots and all(a is b for a, b in zip(view.transcript.cards, cards))
            assert not any(hasattr(c, "_frozen") for c in view.transcript.messages + view.transcript.cards)
            assert all(slot.key.value == slot.slot_key and slot.card.key is None for slot in view.transcript.messages)

            # ---- AudioCard: play, pause and position events after the re-renders --------------------
            card = audio_card(view)
            assert card is not None and not hasattr(card, "_frozen")
            assert await view.audio_hub.toggle(card) == "playing"
            assert card.state == "playing" and card.play_button.tooltip == "Pause"
            view.render_transcript()
            assert audio_card(view) is card  # nothing changed: the playing card stays on the page
            view.audio_hub._on_position(types.SimpleNamespace(position=5000))
            assert card.time_text.value.startswith("0:05")
            assert await view.audio_hub.toggle(card) == "paused"
            assert fake.calls[0][:2] == ("play", str(voice)) and fake.calls[-1] == ("pause",)
            assert card.play_button.tooltip == "Play"

            # ---- Copy ✓ on a re-rendered card -----------------------------------------------------
            message = next(c for c in view.transcript.cards if isinstance(c, AssistantMessage))
            message.show_copied()
            assert message.copy_button.tooltip == "Copied"

            # ---- a run of the chat ends: the OCR section comes back on the re-rendered card ---------
            view._drop_extras(view.cid)
            view.render_transcript(follow=True)
            await tf._wait(lambda: job_card(view).ocr_tile.visible and any(k[0] == "ocr" for k in view._extras),
                           timeout=5)
            assert job_card(view).ocr_tile.title == "OCR (1)"

            # ---- Jump to: the highlight pulses on the slot after a re-render ------------------------
            view.render_transcript()
            task = asyncio.ensure_future(view.jump_to(0))
            await asyncio.sleep(0.05)
            target = view.transcript.slot_for(view._scroll_key_for(0))
            assert target is not None and target.opacity == 0.5
            await task
            assert target.opacity == 1.0
            feature.close()
        finally:
            await tf._stop(app)

    asyncio.run(scenario())


# ==========================================================================
# Device report #4: chat books move into the Library by themselves (no Migrate step)
# ==========================================================================


@pytest.fixture
def iso_env(tmp_path, monkeypatch):
    """Real data stays untouched: the Library, the output root and HOME live under tmp_path."""
    out = tmp_path / "Out"
    library = tmp_path / "Library"
    home = tmp_path / "home"
    for folder in (out, library, home):
        folder.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(out))
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(library))
    for key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA"):
        monkeypatch.setenv(key, str(home))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    return types.SimpleNamespace(out=out, library=library, home=home)


class _IdleJobs(FakeJobService):
    """FakeJobService plus ``JobService.view`` (active / queue / history) and ``on_transition``."""

    def __init__(self):
        super().__init__()
        self.active = None
        self.queued = []
        self.history = []
        self.transitions = []

    def view(self):
        return types.SimpleNamespace(active=self.active, queue=tuple(self.queued), interrupted=(),
                                     history=tuple(self.history))

    def on_transition(self, callback):
        self.transitions.append(callback)
        return lambda: self.transitions.remove(callback) if callback in self.transitions else None


def _chat_feature(adapter, jobs, tmp_path, **app_fields):
    """ChatFeature over the desktop store on a bare app (no page / dispatcher: work runs inline)."""
    from glossarion_mobile.ui.chat.integration import ChatFeature

    notes, routes = [], []
    app = types.SimpleNamespace(
        page=None, dispatcher=None, state=None, chat_view=None, shell=None, settings=None, prefs=None, library=None,
        paths=types.SimpleNamespace(data=str(tmp_path / "data"), temp=str(tmp_path / "tmp"), cache=str(tmp_path / "cache")),
        notify=lambda *args: notes.append(args), navigate_to=lambda *args, **kw: routes.append(args),
    )
    for key, value in app_fields.items():
        setattr(app, key, value)
    feature = ChatFeature(app, chats=adapter, jobs=jobs, oauth=types.SimpleNamespace())
    feature.attach()
    return feature, app, notes, routes


def _write_workspace(folder: Path, raw: Path, *, done=2, total=2, compiled="book.epub") -> Path:
    """A finished chat workspace: progress file, a response, a compiled EPUB, ``source_epub.txt``."""
    folder.mkdir(parents=True, exist_ok=True)
    chapters = {str(i): {"status": "completed" if i <= done else "pending", "output_file": f"response_{i:03d}.html"}
                for i in range(1, total + 1)}
    (folder / "translation_progress.json").write_text(json.dumps({"chapters": chapters}), encoding="utf-8")
    (folder / "response_001.html").write_text("<p>one</p>", encoding="utf-8")
    if compiled:
        (folder / compiled).write_bytes(b"PK\x03\x04compiled")
    (folder / "source_epub.txt").write_text(str(raw), encoding="utf-8")
    return folder


def _book_workspace(adapter, tmp_path) -> tuple:
    """Chat 2's ``Attachments/book`` (from its ``book.epub`` turn) as a finished translation."""
    raw = tmp_path / "book.epub"
    workspace = Path(adapter.attachment_folders("2")[0])
    _write_workspace(workspace, raw)
    return workspace, raw


def test_glossary_predicate_has_one_home():
    """Contract C1: run_controller re-exports the services.jobs predicate (no second copy)."""
    from glossarion_mobile.services import jobs as jobs_service
    from glossarion_mobile.ui.chat import run_controller

    assert run_controller.is_glossary_question is jobs_service.is_glossary_question
    assert run_controller._GLOSSARY_QUESTION_KINDS == jobs_service.GLOSSARY_QUESTION_KINDS
    assert run_controller.is_glossary_question("direct_text_glossary_approval")


def test_finished_attachment_run_moves_into_the_library(desktop_store_cls, tmp_path, iso_env):
    """The owner's complaint: a chat EPUB translation reaches the Library with no Migrate tap. A DONE
    run commits into ``Attachments/<stem>``, ``subscribe_finished`` fires and the desktop Migrate moves
    the tree into the output folder: the stored paths follow, the raw stays in the Inbox (no
    "X (2)" copy), the Library scan lists the book and the snackbar offers "Open book"."""
    _TC._desktop_history(tmp_path)
    adapter = _adapter(desktop_store_cls, tmp_path, save_delay=0.01)
    jobs = _IdleJobs()
    feature, app, notes, _routes = _chat_feature(adapter, jobs, tmp_path)
    runs = feature.runs
    assert runs.temp_dir == str(tmp_path / "data" / "direct_text_runs")
    inbox = tmp_path / "Inbox"
    inbox.mkdir()
    raw = inbox / "Novel.epub"
    raw.write_bytes(b"PK\x03\x04novel")
    seen = []
    runs.subscribe_finished(lambda cid, run, state: seen.append((cid, state, adapter.get(cid).running)))
    run = asyncio.run(runs.send("5", text="", attachment={"name": "Novel.epub", "path": str(raw), "size": 12},
                                settings=DirectTextSettings(), output_mode="text"))
    jobs.publish("RUNNING")
    generated = Path(run.run.temp_root) / "Novel"
    _write_workspace(generated, raw, compiled="Novel.epub")
    Path(run.run.expected_output).write_text("Translated novel", encoding="utf-8")
    jobs.publish("DONE", progress={"total": 2, "completed": 2, "failed": 0})
    _TC._finish_all(runs)
    assert seen == [("5", "DONE", False)]  # after set_running(False)
    target = iso_env.out / "Novel"
    workspace = Path(adapter.output_folder("5")) / "Attachments" / "Novel"
    assert not workspace.exists() and (target / "Novel.epub").is_file()
    assert (target / "translation_progress.json").is_file()
    assert adapter.attachment_folders("5") == []
    folders = {m[4] for m in adapter.messages("5") if m[0] == "assistant" and len(m) > 4 and m[4]}
    assert str(target) in folders and not any("Attachments" in Path(f).parts for f in folders)
    assert run.output_folder == str(target)  # Compile / Reader from the card follow the move
    assert raw.is_file() and not any(p.name.startswith("Novel (") for p in iso_env.out.iterdir())
    assert not (iso_env.library / "Raw").exists() or not any((iso_env.library / "Raw").iterdir())
    assert notes[-1] == ("Added to the Library",)  # no Library installed in this bare app: no "Open book"

    from glossarion_mobile.services.library import LibraryService

    import library_core

    service = LibraryService(paths=types.SimpleNamespace(library=iso_env.library, output=iso_env.out,
                                                         cache=tmp_path / "cache"), config={}, prefs=None)
    service.ensure_env()
    try:
        snap = service.scan_blocking()
        rows = [b for b in snap.all_books() if Path(str(b.get("output_folder") or b.get("path") or "")).parts[-2:]
                in (("Out", "Novel"), ("Novel", "Novel.epub"))]
        assert rows, [b.get("name") for b in snap.all_books()]
        assert service.raw_source(rows[0]) == str(raw)  # the registry knows the Inbox raw
    finally:
        library_core.uninstall_library_env()
    adapter.close()


def test_auto_migrate_waits_while_a_job_holds_the_lock(desktop_store_cls, tmp_path, iso_env, monkeypatch):
    """Regression for the race the skeptic probe showed: the shared Migrate reads the live
    OUTPUT_DIRECTORY, which a running job points at its temporary run root. While JOB_LOCK is held,
    a job is active, or OUTPUT_DIRECTORY names a run root, the move waits; the idle transition then
    moves it into Output/<stem>."""
    import threading

    import job_runner

    _TC._desktop_history(tmp_path)
    adapter = _adapter(desktop_store_cls, tmp_path, save_delay=0.01)
    jobs = _IdleJobs()
    feature, app, notes, _routes = _chat_feature(adapter, jobs, tmp_path)
    assert feature._on_job_transition in jobs.transitions  # subscribed at attach
    workspace, _raw = _book_workspace(adapter, tmp_path)
    held, release = threading.Event(), threading.Event()

    def hold():
        with job_runner.JOB_LOCK:
            held.set()
            release.wait(10)

    holder = threading.Thread(target=hold)
    holder.start()
    assert held.wait(5)
    try:
        outcome = feature.auto_migrate_blocking("2", str(workspace))
    finally:
        release.set()
        holder.join(5)
    assert outcome["status"] == "deferred" and outcome["jobs_running"] and workspace.is_dir()
    # a job is active (its env is set) although the lock is free for a moment
    jobs.active = types.SimpleNamespace(id="j9", state="RUNNING", spec=types.SimpleNamespace(kind="translate", inputs=(),
                                                                                            params={}))
    assert feature.auto_migrate_blocking("2", str(workspace))["status"] == "deferred"
    assert feature._on_job_transition(types.SimpleNamespace(state="DONE")) is None  # not idle yet
    jobs.active = None
    # OUTPUT_DIRECTORY still names a run root
    run_root = Path(feature.runs.temp_dir) / "glossarion_input_output_next"
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(run_root))
    assert feature.auto_migrate_blocking("2", str(workspace))["status"] == "deferred"
    assert not (run_root / "book").exists() and workspace.is_dir()
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(iso_env.out))

    async def idle():
        assert feature._on_job_transition(types.SimpleNamespace(state="RUNNING")) is None
        task = feature._on_job_transition(types.SimpleNamespace(state="DONE"))
        return await task

    outcomes = asyncio.run(idle())
    assert [o["status"] for o in outcomes] == ["moved"]
    assert not workspace.exists() and (iso_env.out / "book" / "book.epub").is_file()
    assert notes[-1] == ("Added to the Library",)
    adapter.close()


def test_auto_migrate_merges_the_same_book_and_asks_for_another(desktop_store_cls, tmp_path, iso_env):
    """A same-named Library folder of the SAME book (its source_epub.txt points at the same raw) is
    merged into silently (the newest compiled document kept); a different book's folder is left
    alone: the workspace stays in Attachments and the snackbar's action opens the desktop
    "Attachment folder already exists" dialog, whose Merge and replace moves it."""
    _TC._desktop_history(tmp_path)
    adapter = _adapter(desktop_store_cls, tmp_path, save_delay=0.01)
    jobs = _IdleJobs()
    feature, app, notes, _routes = _chat_feature(adapter, jobs, tmp_path)
    workspace, raw = _book_workspace(adapter, tmp_path)
    existing = _write_workspace(iso_env.out / "book", raw, compiled="Old.epub")
    os.utime(existing / "Old.epub", (1, 1))
    (existing / "notes.txt").write_text("kept", encoding="utf-8")
    outcome = feature.auto_migrate_blocking("2", str(workspace))
    assert outcome["status"] == "moved" and outcome["reason"] == "merged" and not workspace.exists()
    epubs = sorted(p.name for p in existing.glob("*.epub"))
    assert epubs == ["book.epub"] and (existing / "notes.txt").is_file()

    # a different book already owns Out/other
    other_raw = tmp_path / "other.epub"
    other_raw.write_bytes(b"PK\x03\x04other book")
    stranger = tmp_path / "stranger.epub"
    stranger.write_bytes(b"PK\x03\x04a different book")
    second = _write_workspace(Path(adapter.output_folder("2")) / "Attachments" / "other", other_raw)
    foreign = _write_workspace(iso_env.out / "other", stranger)
    before = (foreign / "translation_progress.json").read_bytes()
    outcome = feature.auto_migrate_blocking("2", str(second))
    assert outcome["status"] == "collision" and second.is_dir()
    assert (foreign / "translation_progress.json").read_bytes() == before
    feature._announce(outcome)
    feature._announce(outcome)  # once per workspace and session
    collisions = [n for n in notes if n and str(n[0]).startswith("A Library book named")]
    assert collisions == [notes[-1]] and notes[-1][:2] == ("A Library book named other already exists", "Merge…")
    dialog = notes[-1][2]()
    assert dialog.title == "Attachment folder already exists"
    asyncio.run(dialog._on_confirm())
    assert dialog.result is True and not second.exists() and (foreign / "source_epub.txt").read_text(
        encoding="utf-8") == str(other_raw)
    assert notes[-1] == ("Added to the Library",)
    adapter.close()


def test_auto_migrate_leaves_resumable_scratch_and_non_book_workspaces(desktop_store_cls, tmp_path, iso_env):
    """Workspaces whose job card still offers Resume / Retry failed (stopped, done with failed
    chapters, a failed job remembered after a relaunch), scratch chats and non-book attachments
    (an image) stay in Attachments."""
    from glossarion_mobile.ui.chat.run_controller import ChatRun
    from glossarion_mobile.ui.chat.run_request import DirectTextRun
    from glossarion_mobile.ui.chat.stream_bridge import RunStream

    _TC._desktop_history(tmp_path)
    adapter = _adapter(desktop_store_cls, tmp_path, save_delay=0.01)
    jobs = _IdleJobs()
    feature, app, notes, _routes = _chat_feature(adapter, jobs, tmp_path)
    workspace, raw = _book_workspace(adapter, tmp_path)
    runs = feature.runs

    def session_run(state, failed=0):
        prepared = DirectTextRun(temp_root=str(tmp_path / "runs" / "r1"), source_path=str(raw), source_extension=".epub",
                                 is_attachment=True, expected_output="")
        snap = types.SimpleNamespace(progress={"total": 2, "completed": 2 - failed, "failed": failed})
        runs.runs["2"] = ChatRun(cid="2", run=prepared, stream=RunStream(), user_index=2, params={"run": {}},
                                 state=state, finished=True, output_folder=str(workspace), last_snapshot=snap)

    session_run("done", failed=1)  # "Finished with issues · 1 failed": Retry failed
    assert runs.last_job_ending("2")[0] == "issues"
    outcome = feature.auto_migrate_blocking("2", str(workspace))
    assert (outcome["status"], outcome["reason"]) == ("skipped", "its run can still be resumed")
    session_run("stopped")
    assert feature.auto_migrate_blocking("2", str(workspace))["status"] == "skipped"
    # after a relaunch: JobService remembers the chat's failed job on this turn (Resume)
    runs.runs.clear()
    failed_job = types.SimpleNamespace(
        id="old", state="FAILED", resolution=None, finished=5.0, started=1.0, created=1.0, progress=None,
        spec=types.SimpleNamespace(kind="direct_text", title="book", inputs=(str(raw),),
                                   origin={"type": "chat", "cid": "2", "label": "Chat · My novel"},
                                   params={"chat_id": 2, "user_index": 2, "run": {"source_path": str(raw)}}))
    assert feature.job_workspace(failed_job) == str(workspace)  # the turn's workspace, not a name guess
    jobs.history = [failed_job]
    assert runs.last_job_ending("2")[0] == "failed"
    assert feature.auto_migrate_blocking("2", str(workspace))["reason"] == "its run can still be resumed"
    assert workspace.is_dir()
    # the same job finished cleanly: the workspace moves on the next sweep
    jobs.history = [types.SimpleNamespace(**{**vars(failed_job), "state": "DONE"})]
    assert runs.last_job_ending("2")[0] == "done"

    # an image attachment's workspace is not a book
    png = tmp_path / "photo.png"
    png.write_bytes(PNG)
    adapter.append_messages("2", [("user_file", "photo.png", str(png), len(PNG), "", "user")])
    photo = Path(adapter.output_folder("2")) / "Attachments" / "photo"
    _write_workspace(photo, png, compiled="")
    # a scratch chat's workspace never leaves the scratch folder
    scratch = adapter.new_scratch()
    scratch_ws = Path(adapter.output_folder(scratch, create=True)) / "Attachments" / "book"
    _write_workspace(scratch_ws, raw)
    assert feature.auto_migrate_blocking(scratch, str(scratch_ws))["status"] == "skipped"
    outcomes = {Path(o["folder"]).name: o for o in feature.sweep_blocking()}
    assert outcomes["photo"]["reason"] == "not a book" and photo.is_dir()
    assert outcomes["book"]["status"] == "moved" and (iso_env.out / "book").is_dir()
    assert scratch_ws.is_dir() and all(o["cid"] != scratch for o in outcomes.values())
    adapter.close()


def test_startup_sweep_moves_earlier_chat_books_once(desktop_store_cls, tmp_path, iso_env):
    """Chat books finished by an earlier version (or before the app was killed) join the Library
    on the next launch; a second sweep finds nothing to do."""
    _TC._desktop_history(tmp_path)
    adapter = _adapter(desktop_store_cls, tmp_path, save_delay=0.01)
    feature, app, notes, _routes = _chat_feature(adapter, _IdleJobs(), tmp_path)
    workspace, _raw = _book_workspace(adapter, tmp_path)
    first = asyncio.run(feature.startup_sweep(wait=0))
    assert [(Path(o["folder"]).name, o["status"]) for o in first] == [("book", "moved")]
    assert not workspace.exists() and (iso_env.out / "book" / "book.epub").is_file()
    assert asyncio.run(feature.sweep("again")) == []
    assert sorted(p.name for p in iso_env.out.iterdir()) == ["book"]
    assert notes == [("Added to the Library",)]
    # several books in one sweep: one summary snackbar instead of one per book
    for name in ("alpha", "beta"):
        raw = tmp_path / f"{name}.epub"
        raw.write_bytes(f"PK\x03\x04{name}".encode())
        _write_workspace(Path(adapter.output_folder("2")) / "Attachments" / name, raw)
    moved = asyncio.run(feature.sweep("idle"))
    assert sorted(Path(o["target"]).name for o in moved if o["status"] == "moved") == ["alpha", "beta"]
    assert notes[1:] and len(notes) == 2 and notes[-1][:2] == ("Added 2 books to the Library", "Library")
    notes[-1][2]()
    assert _routes[-1] == ("library",)
    adapter.close()


def test_finished_listeners_run_after_the_commit_and_the_library_hooks(desktop_store_cls, tmp_path, iso_env,
                                                                       monkeypatch):
    """``subscribe_finished`` runs once the chat is saved and idle; the chat env's Library hooks (C2):
    ``library_book`` finds the moved book's id, ``library_translate`` opens the Library translate sheet
    for it (Resume / Retry failed of a moved workspace), and "Open book" opens it."""
    from glossarion_mobile.ui.chat.run_controller import ChatRun, ChatRuns
    from glossarion_mobile.ui.chat.run_request import DirectTextRun
    from glossarion_mobile.ui.chat.stream_bridge import RunStream

    calls = []

    class Store:
        def session(self, cid):
            return None

        def set_running(self, cid, running):
            calls.append(("set_running", running))

        def refresh_attachments(self, cid):
            calls.append(("refresh_attachments", cid))

        def flush(self):
            calls.append(("flush",))

    runs = ChatRuns(Store(), JobsAdapter(None))
    unsubscribe = runs.subscribe_finished(lambda cid, run, state: calls.append(("finished", cid, state)))
    prepared = DirectTextRun(temp_root="", source_path="", source_extension=".txt", is_attachment=False, expected_output="")
    run = ChatRun(cid="7", run=prepared, stream=RunStream(), user_index=0)
    runs.finish(run, types.SimpleNamespace(state="DONE", started=1.0))
    assert calls == [("set_running", False), ("refresh_attachments", "7"), ("flush",), ("finished", "7", "DONE")]
    unsubscribe()
    runs.finish(ChatRun(cid="8", run=prepared, stream=RunStream(), user_index=0), types.SimpleNamespace(state="DONE"))
    assert not any(c[0] == "finished" and c[1] == "8" for c in calls)

    from glossarion_mobile.services.library import LibraryService

    import library_core

    _TC._desktop_history(tmp_path)
    adapter = _adapter(desktop_store_cls, tmp_path, save_delay=0.01)
    service = LibraryService(paths=types.SimpleNamespace(library=iso_env.library, output=iso_env.out,
                                                         cache=tmp_path / "cache"), config={}, prefs=None)
    service.ensure_env()
    sheets = []

    async def fake_sheet(ctx, books):
        sheets.append((ctx, [dict(b) for b in books]))
        return "sheet"

    import glossarion_mobile.ui.library.translate_sheet as translate_sheet

    monkeypatch.setattr(translate_sheet, "open_translate_sheet", fake_sheet)
    try:
        feature, app, notes, routes = _chat_feature(
            adapter, _IdleJobs(), tmp_path, library=service,
            library_feature=types.SimpleNamespace(context=lambda: "library-ctx"))
        env = feature.env
        assert env.library_service() is service and env.prefs is None
        workspace, raw = _book_workspace(adapter, tmp_path)
        outcome = feature.auto_migrate_blocking("2", str(workspace))
        feature._announce(outcome)
        target = str(iso_env.out / "book")
        assert service.dirty and notes[-1][:2] == ("Added to the Library", "Open book")

        async def hooks():
            bid = await env.library_book(target)
            sheet = await env.library_translate(target, str(raw))
            missing = await env.library_book(str(tmp_path / "nowhere"))
            return bid, sheet, missing

        bid, sheet, missing = asyncio.run(hooks())
        assert bid and bid == service.bid_for(service.book_for_bid(bid)) and missing is None
        assert Path(service.book_for_bid(bid)["output_folder"]) == Path(target)
        assert sheet == "sheet" and sheets[0][0] == "library-ctx"
        assert Path(sheets[0][1][0]["output_folder"]) == Path(target)
        notes[-1][2]()  # Open book
        assert routes[-1] == ("library.book", {"bid": bid})
    finally:
        library_core.uninstall_library_env()
        adapter.close()
