"""U5: progress_actions (Progress Manager row actions) parity and API tests.

* V: moved helpers / closure bodies equal the frozen Retranslation_GUI modulo the
  documented re-parameterisation (``data['prog']`` -> ``prog`` ...);
* F: every non-retranslate action run through the real offscreen dialog (context menu /
  buttons, dialogs answered Yes) on fixture copies: legacy dialog vs working-tree
  dialog -> progress JSON + output tree, messages and the refreshed rows are equal;
* C: an action keeps a translator save that landed after the dialog read the file
  (the frozen whole-file writes lost it: recorded in DISCREPANCIES.md, U5);
* M: the mobile functions produce the desktop result without Qt.
"""

from __future__ import annotations

import ast
import copy
import json
import os
import re
import sys
import textwrap
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
for _p in (str(TESTS_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from parity import progress_legacy as pl  # noqa: E402

import progress_actions as pa  # noqa: E402
import progress_core as pc  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(tmp_path / "_library"))
    # Workspaces resolve under $OUTPUT_DIRECTORY / $OUTPUT_DIR before the config's output
    # directory: never let a caller's value send them outside tmp_path.
    monkeypatch.delenv("OUTPUT_DIRECTORY", raising=False)
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _source(name):
    return (SRC_DIR / f"{name}.py").read_text(encoding="utf-8").replace("\r\n", "\n")


def _legacy_lines(start, end, strip, add=0):
    out = []
    for line in pl.legacy_source_lines()[start - 1:end]:
        if line.strip():
            assert line.startswith(" " * strip), (start, line)
            out.append(" " * add + line[strip:])
        else:
            out.append("")
    return "\n".join(out)


def _apply(text, edits):
    for old, new, count in edits:
        assert text.count(old) == count, (old, text.count(old))
        text = text.replace(old, new)
    return text


def _top_level(source):
    tree = ast.parse(source)
    lines = source.split("\n")
    return {
        node.name: "\n".join(lines[node.lineno - 1:node.end_lineno])
        for node in tree.body if isinstance(node, ast.FunctionDef)
    }


# ===========================================================================
# Tier V
# ===========================================================================

MOVED_HELPERS = {
    # name: (start, end) in the frozen RG (module level)
    "clear_progress_entry_qa_mark": (230, 264),
    "clear_chunk_row_qa_mark": (267, 289),
    "_recover_pending_marks": (328, 411),
    "_remove_pending_marks": (414, 423),
    "_without_llm_token_qa": (542, 566),
    "_clear_llm_token_qa_markers": (569, 618),
    "_qa_scalar_is_missing_image_issue": (650, 654),
    "_qa_mapping_is_missing_image_issue": (657, 666),
    "_without_missing_image_qa": (669, 708),
    "_clear_missing_image_qa_markers": (711, 759),
    "_repair_empty_attribute_qa_file": (762, 842),
    "_bulk_retranslation_sidecar_updates": (1675, 1740),
}


@pytest.mark.parametrize("name", sorted(MOVED_HELPERS))
def test_moved_helpers_are_verbatim(name):
    start, end = MOVED_HELPERS[name]
    assert _top_level(_source("progress_actions"))[name] == _legacy_lines(start, end, 0)


#: Closure bodies (frozen RG ranges) -> the module function holding them, with edits.
CLOSURE_BODIES = [
    ("_normalize_filename", 27900, 27911, 8, 0, []),
    ("_find_progress_entry", 27913, 27921, 8, 0, []),
    ("_restore_regular_in_progress_entry", 27935, 27973, 8, 0, [
        ("os.path.exists(os.path.join(data['output_dir'], output_file))",
         "os.path.exists(os.path.join(output_dir, output_file))", 1)]),
    ("_apply_restore_in_progress", 27992, 28021, 8, 0, [
        ("data['prog']", "prog", 5),
        ("_restore_regular_in_progress_entry(entry)", "_restore_regular_in_progress_entry(entry, output_dir)", 1)]),
    ("plan_remove_qa_marks", 28072, 28093, 8, 0, []),
    ("_apply_remove_qa_failed_mark", 28108, 28150, 8, 0, [
        ("data['prog']", "prog", 7), ("data['output_dir']", "output_dir", 2)]),
    ("refinement_status_keys", 28180, 28215, 8, 0, [
        ("chapters = data.get('prog', {}).get('chapters', {})", "chapters = prog.get('chapters', {})", 1)]),
    ("_apply_remove_refinement_status", 28239, 28242, 8, 0, []),
    ("_apply_reset_tts", 28413, 28478, 12, 0, [
        ("data['prog']", "prog", 3), ("data['output_dir']", "output_dir", 1)]),
    ("_find_audio_file_for_item", 30509, 30544, 8, 0, []),
    ("_reset_tts_progress_for_output", 30547, 30561, 8, 0, [
        ("data.get('prog', {}).get('chapters', {})", "prog.get('chapters', {})", 1)]),
    ("_llm_token_qa_targets", 30679, 30705, 8, 0, [
        ("chapters = data.get('prog', {}).get('chapters', {})", "chapters = prog.get('chapters', {})", 1)]),
    ("_clear_llm_token_targets", 30707, 30712, 8, 0, []),
    ("_missing_image_qa_targets", 31048, 31084, 28, 0, [
        ("chapters = data['prog'].get('chapters', {})", "chapters = prog.get('chapters', {})", 1)]),
    ("_clear_missing_image_targets", 31086, 31095, 28, 0, []),
    ("_partial_b_target", 18953, 18983, 4, 0, []),
]


@pytest.mark.parametrize("name,start,end,strip,add,edits", CLOSURE_BODIES, ids=[c[0] for c in CLOSURE_BODIES])
def test_closure_bodies_are_verbatim(name, start, end, strip, add, edits):
    function = _top_level(_source("progress_actions"))[name]
    block = _apply(_legacy_lines(start, end, strip, add), edits)
    assert block in function, name


def test_retranslation_gui_closures_call_the_shared_actions():
    source = _source("Retranslation_GUI")
    for call in ("restore_in_progress(", "plan_remove_qa_marks(", "remove_qa_marks(",
                 "refinement_status_keys(", "_progress_remove_refinement_status(", "reset_tts(",
                 "find_row_audio(", "resolve_llm_token_qa(", "insert_missing_images(",
                 "prepare_single_qa_resolution("):
        assert call in source, call
    # U7: the single-entry Partial.b preflight is shared (progress_actions)
    preflight = _top_level(_source("progress_actions"))["prepare_single_qa_resolution"]
    assert "_partial_b_target(" in preflight and "_partial_b_request(" in preflight
    # the plain whole-file progress dumps of these actions are gone
    assert "json.dump(data['prog'], f, ensure_ascii=False, indent=2)" not in source.split(
        "def retranslate_selected():")[0].split("def _add_retranslation_buttons_opf(")[1]


# ===========================================================================
# Tier F: the real dialogs
# ===========================================================================


def _normalize_chunks(tree, out_root=None):
    """The state the next refresh reconciles to (DISCREPANCIES U5, "display-time state").

    The frozen actions wrote the dialog's whole in-memory snapshot, which also carried
    what the view had reconciled for display (chunk-ledger schema normalisation, TTS
    status synced to the audio files on disk).  The shared actions write only their own
    change, so both sides are compared after that same reconciliation.
    """
    from chapter_chunk_progress import ensure_chunk_entry_schema

    out = {}
    for rel, (kind, value) in tree.items():
        if kind == "json" and isinstance(value, dict) and isinstance(value.get("chapters"), dict):
            value = copy.deepcopy(value)
            for entry in (value.get("chapter_chunks") or {}).values():
                if isinstance(entry, dict):
                    ensure_chunk_entry_schema(entry)
            if out_root is not None:
                owner = pc.ProgressOwner({})
                owner._reconcile_tts_audio_files({
                    "prog": value,
                    "output_dir": str(Path(out_root) / Path(rel).parent),
                })
                for entry in value["chapters"].values():
                    if not isinstance(entry, dict):
                        continue
                    # 'no_tts' without a file == no TTS state; the reconcile stamps
                    # last_updated on the entries it touches.
                    if entry.get("tts_status") == "no_tts" and not entry.get("tts_file"):
                        entry.pop("tts_status", None)
                    entry.pop("last_updated", None)
            value = pl.normalize_progress(value)
        out[rel] = (kind, value)
    return out


def _normalize_messages(messages, workspace):
    text = json.dumps(messages, ensure_ascii=False)
    for raw in (str(workspace), str(workspace).replace("\\", "/")):
        text = text.replace(json.dumps(raw)[1:-1], "<ROOT>")
    return json.loads(text)


@pytest.fixture(scope="module")
def epub_fixture(tmp_path_factory):
    base = tmp_path_factory.mktemp("actions_epub")
    source, _out, config = pl.epub_workspace(base)
    return base, Path(source).name, config


def _diff(x, y, path=""):
    """Differing paths of two nested values (assertion messages)."""
    if isinstance(x, tuple) and isinstance(y, tuple) and len(x) == len(y) == 2 and x[0] == y[0]:
        return _diff(x[1], y[1], path)
    if isinstance(x, dict) and isinstance(y, dict):
        out = []
        for key in sorted(set(x) | set(y), key=str):
            if key not in x or key not in y:
                out.append(f"{path}/{key}: {'only legacy' if key in x else 'only current'}")
            else:
                out.extend(_diff(x[key], y[key], f"{path}/{key}"))
        return out
    return [] if x == y else [f"{path}: {x!r:.120} != {y!r:.120}"]


def _run_action(module, fixture, work, *, label=None, predicate=None, button=None,
                audio=False, show_special=False, before_action=None):
    saved_env = dict(os.environ)
    try:
        return _run_action_inner(module, fixture, work, label=label, predicate=predicate, button=button,
                                 audio=audio, show_special=show_special, before_action=before_action)
    finally:
        os.environ.clear()
        os.environ.update(saved_env)


def _run_action_inner(module, fixture, work, *, label=None, predicate=None, button=None,
                      audio=False, show_special=False, before_action=None):
    base, source_name, config = fixture
    work = pl.copy_workspace(base, work)
    cfg = dict(config, output_directory=str(work / "out"))
    host = pl.make_host(module, cfg, **({"output_mode_var": "audio"} if audio else {}))
    data = pl.open_progress_manager(host, work / source_name)
    if show_special:
        data['show_special_files_cb'].setChecked(True)
        pl.pump(20, timeout=0.2)
    if before_action is not None:
        before_action(work, data)
    pl.MESSAGES.clear()
    chosen = None
    if label is not None:
        chosen = pl.context_action(module, data, label, predicate)
    else:
        pl.select_rows(data, predicate)
        pl.click_button(data, button)
    pl.pump(20, timeout=0.5)
    result = {
        "chosen": chosen,
        "rows": pl.view_snapshot(data),
        "tree": _normalize_chunks(pl.tree_snapshot(work / "out", workspace=work), work / "out"),
        "messages": _normalize_messages(list(pl.MESSAGES), work),
        "config": {k: v for k, v in cfg.items() if k != "output_directory"},
        "work": work,
    }
    data['dialog'].hide()
    return result


def _is(name, chunk=False):
    def predicate(info):
        return info.get('original_filename') == name and bool(info.get('is_chunk_progress')) == chunk
    return predicate


ACTIONS = {
    "restore_in_progress": dict(label="Restore In Progress Status",
                                predicate=lambda i: i.get('status') == 'in_progress' and not i.get('is_chunk_progress')),
    "remove_qa_mark": dict(label="🧹 Remove QA Failed Mark", predicate=_is("chapter0002.xhtml")),
    "remove_qa_mark_chunk": dict(label="🧹 Remove QA Failed Mark",
                                 predicate=lambda i: i.get('is_chunk_progress') and i.get('chunk_index') == 2),
    "remove_qa_mark_button": dict(button="Remove QA Failed Mark",
                                  predicate=lambda i: i.get('original_filename') in ('chapter0002.xhtml', 'chapter0003.xhtml')
                                  and not i.get('is_chunk_progress')),
    "remove_pending_mark": dict(label="🧽 Remove Pending Mark", predicate=_is("chapter0004.xhtml")),
    "remove_refinement": dict(label="⭐ Remove refinement status",
                              predicate=lambda i: i.get('original_filename') in ('chapter0001.xhtml', 'chapter0005.xhtml')
                              and not i.get('is_chunk_progress')),
    "resolve_llm_token_qa": dict(label="⚠️ Resolve QA issue", predicate=_is("chapter0002.xhtml")),
    "delete_audio": dict(label="🗑️ Delete Audio File", predicate=_is("chapter0001.xhtml")),
    "do_not_skip": dict(label="⏭️ Do not skip", predicate=_is("title.xhtml"), show_special=True),
    "insert_missing_image": dict(label="🖼️ Insert Missing Image", predicate=_is("chapter0002.xhtml")),
    "reset_tts": dict(button="Reset TTS Selected", audio=True,
                      predicate=lambda i: i.get('original_filename') in ('chapter0001.xhtml', 'chapter0003.xhtml')
                      and not i.get('is_chunk_progress')),
}


@pytest.mark.parametrize("action", sorted(ACTIONS))
def test_action_matches_frozen_desktop(action, epub_fixture, tmp_path):
    spec = ACTIONS[action]
    legacy = _run_action(pl.legacy_rg(), epub_fixture, tmp_path / "legacy", **spec)
    current = _run_action(pl.current_rg(), epub_fixture, tmp_path / "current", **spec)
    if "label" in spec:
        assert legacy["chosen"] and legacy["chosen"] == current["chosen"]
    assert current["messages"] == legacy["messages"]
    assert current["tree"] == legacy["tree"], _diff(legacy["tree"], current["tree"])
    assert current["rows"] == legacy["rows"]
    assert current["config"] == legacy["config"]
    assert legacy["messages"], "the action showed no result"


def test_actions_change_what_they_should(epub_fixture, tmp_path):
    """Spot checks of the action results (both sides are equal by the parity test)."""
    def raw(result):
        return json.loads((result["work"] / "out" / "Book" / "translation_progress.json").read_text(encoding="utf-8"))["chapters"]

    result = _run_action(pl.current_rg(), epub_fixture, tmp_path / "w1", **ACTIONS["restore_in_progress"])
    chapters = raw(result)
    assert chapters["5"]["status"] == "completed" and chapters["5"]["model_name"] == "gpt-old"
    assert "6" not in chapters
    result = _run_action(pl.current_rg(), epub_fixture, tmp_path / "w2", **ACTIONS["reset_tts"])
    chapters = raw(result)
    assert chapters["1"]["tts_status"] == "no_tts" and "tts_file" not in chapters["1"]
    assert "Book/text_to_speech/response_chapter0001.mp3" not in result["tree"]


# ===========================================================================
# Tier C: a translator save between the dialog's read and the action survives
# ===========================================================================


def _translator_saves(work, _data):
    progress_file = work / "out" / "Book" / "translation_progress.json"
    data = json.loads(progress_file.read_text(encoding="utf-8"))
    data["chapters"]["7"] = dict(data["chapters"].get("7", {}), model_name="translator-save", status="completed")
    data["chapters"]["10"] = {"actual_num": 10, "status": "in_progress", "output_file": "response_chapter0010.html"}
    pc.write_progress_atomic(progress_file, data)


CONCURRENT = ("restore_in_progress", "remove_qa_mark", "remove_refinement", "resolve_llm_token_qa",
              "delete_audio", "insert_missing_image", "reset_tts", "remove_pending_mark")


@pytest.mark.parametrize("action", CONCURRENT)
def test_action_keeps_a_concurrent_translator_save(action, epub_fixture, tmp_path):
    spec = dict(ACTIONS[action], before_action=_translator_saves)
    current = _run_action(pl.current_rg(), epub_fixture, tmp_path / "current", **spec)
    chapters = current["tree"]["Book/translation_progress.json"][1]["chapters"]
    assert chapters["7"]["model_name"] == "translator-save"
    assert chapters["10"]["status"] == "in_progress"
    legacy = _run_action(pl.legacy_rg(), epub_fixture, tmp_path / "legacy", **spec)
    legacy_chapters = legacy["tree"]["Book/translation_progress.json"][1]["chapters"]
    if action != "remove_pending_mark":   # already merge-written before U5
        # Recorded desktop bug (DISCREPANCIES U5): the whole-file write dropped the save.
        assert legacy_chapters.get("10", {}).get("status") != "in_progress"


# ===========================================================================
# Tier M: the mobile functions
# ===========================================================================


def _workspace(epub_fixture, tmp_path, name="m"):
    base, source_name, config = epub_fixture
    work = pl.copy_workspace(base, tmp_path / name)
    config = dict(config, output_directory=str(work / "out"))
    book = pc.build_book_progress(str(work / source_name), config, show_special_files=True)
    return work, book


def _rows(book, predicate):
    return [row.info for row in book.rows if predicate(row.info)]


def _desktop_tree(action, epub_fixture, tmp_path):
    return _run_action(pl.legacy_rg(), epub_fixture, tmp_path / f"desk_{action}", **ACTIONS[action])["tree"]


def _assert_matches_desktop(work, action, epub_fixture, tmp_path):
    mobile = _normalize_chunks(pl.tree_snapshot(work / "out", workspace=work), work / "out")
    desktop = _desktop_tree(action, epub_fixture, tmp_path)
    assert mobile == desktop, _diff(desktop, mobile)


def test_mobile_restore_in_progress_matches_desktop(epub_fixture, tmp_path):
    work, book = _workspace(epub_fixture, tmp_path)
    rows = _rows(book, ACTIONS["restore_in_progress"]["predicate"])
    result = pa.restore_in_progress(book.progress_file, book.output_dir, rows)
    assert pa.restore_in_progress_message(result) == "Successfully restored 1, removed 1 not-translated placeholder(s)."
    _assert_matches_desktop(work, "restore_in_progress", epub_fixture, tmp_path)


def test_mobile_remove_qa_marks_matches_desktop(epub_fixture, tmp_path):
    work, book = _workspace(epub_fixture, tmp_path)
    selected = _rows(book, ACTIONS["remove_qa_mark"]["predicate"])
    failed = pa.plan_remove_qa_marks(book.data["prog"], selected)
    assert len(failed) == 1
    assert pa.remove_qa_marks(book.progress_file, book.output_dir, failed)["cleared"] == 1
    _assert_matches_desktop(work, "remove_qa_mark", epub_fixture, tmp_path)


def test_mobile_remove_refinement_and_reset_tts(epub_fixture, tmp_path):
    work, book = _workspace(epub_fixture, tmp_path)
    selected = _rows(book, ACTIONS["remove_refinement"]["predicate"])
    keys = pa.refinement_status_keys(book.data["prog"], selected)
    assert sorted(keys) == ["1", "5"]
    assert pa.remove_refinement_status(book.progress_file, keys) == 2
    _assert_matches_desktop(work, "remove_refinement", epub_fixture, tmp_path)

    work2, book2 = _workspace(epub_fixture, tmp_path, "m2")
    result = pa.reset_tts(book2.owner, book2.progress_file, book2.output_dir,
                          _rows(book2, ACTIONS["reset_tts"]["predicate"]))
    assert pa.reset_tts_message(result) == (
        "Successfully deleted 1 TTS file(s), marked 2 chapter(s) as No TTS, 1 chapter(s) had no audio file on disk.")
    _assert_matches_desktop(work2, "reset_tts", epub_fixture, tmp_path)


def test_reset_tts_reports_a_failed_progress_write(epub_fixture, tmp_path, monkeypatch):
    """The former TTS-reset write sat in a try that printed the failure and went on to
    refresh and report; the shared reset keeps that through ``result['error']``."""
    def _write_fails(path, fn, **_kwargs):
        fn(copy.deepcopy(pc._read_progress_file(path)))   # audio deleted, then the write fails
        raise PermissionError("progress file locked")

    monkeypatch.setattr(pa, "mutate_progress", _write_fails)
    work, book = _workspace(epub_fixture, tmp_path)
    progress_path = work / "out" / "Book" / "translation_progress.json"
    before = progress_path.read_bytes()
    result = pa.reset_tts(book.owner, book.progress_file, book.output_dir,
                          _rows(book, ACTIONS["reset_tts"]["predicate"]))
    assert result["error"] == "progress file locked"
    assert (result["deleted"], result["status_reset"], result["missing_audio"]) == (1, 2, 1)
    assert progress_path.read_bytes() == before

    desktop = _run_action(pl.current_rg(), epub_fixture, tmp_path / "desk", **ACTIONS["reset_tts"])
    assert any("Successfully deleted 1 TTS file(s)" in json.dumps(m, ensure_ascii=False)
               for m in desktop["messages"]), desktop["messages"]


def test_mobile_audio_llm_token_and_image_actions(epub_fixture, tmp_path):
    work, book = _workspace(epub_fixture, tmp_path)
    ch1 = _rows(book, _is("chapter0001.xhtml"))[0]
    audio = pa.find_row_audio(book.owner, book.data, ch1)
    assert audio and audio.endswith("response_chapter0001.mp3")
    assert pa.delete_row_audio(book.owner, book.data, ch1, audio) >= 1
    assert not os.path.exists(audio)
    _assert_matches_desktop(work, "delete_audio", epub_fixture, tmp_path)

    work2, book2 = _workspace(epub_fixture, tmp_path, "m2")
    ch2 = _rows(book2, _is("chapter0002.xhtml"))[0]
    output = os.path.join(book2.output_dir, ch2["output_file"])
    outcome = pa.resolve_llm_token_qa(book2.progress_file, ch2, output)
    assert outcome["repair"]["resolved"] and outcome["progress_changed"]
    assert pa.llm_token_repair_summary(outcome, output).endswith("Other QA issues remain on this entry.")
    _assert_matches_desktop(work2, "resolve_llm_token_qa", epub_fixture, tmp_path)

    work3, book3 = _workspace(epub_fixture, tmp_path, "m3")
    ch2 = _rows(book3, _is("chapter0002.xhtml"))[0]
    kind, title, message, refreshed = pa.insert_missing_images(book3.data, ch2)
    assert (kind, title, refreshed) == ("info", "Success", True)
    _assert_matches_desktop(work3, "insert_missing_image", epub_fixture, tmp_path)


def test_mobile_pending_mark_and_special_keyword(epub_fixture, tmp_path):
    work, book = _workspace(epub_fixture, tmp_path)
    ch4 = _rows(book, _is("chapter0004.xhtml"))[0]
    result = pa.remove_pending_marks(book.progress_file, book.output_dir, [ch4])
    assert pa.remove_pending_message(result).startswith("Restored 1 pending entries.")
    title = _rows(book, _is("title.xhtml"))[0]
    assert pa.special_keyword_for(book.owner, title) == "title"
    assert pa.row_actions(book.owner, book.data, title) == ["do_not_skip"]
    assert pa.remove_special_keyword(book.owner, "title")
    assert book.owner.config["special_file_keywords"] == "notice"


def test_row_actions_mirror_the_context_menu(epub_fixture, tmp_path):
    """Menu entries of the real dialog vs ``row_actions`` for every visible row."""
    from PySide6.QtWidgets import QMenu

    base, source_name, config = epub_fixture
    work = pl.copy_workspace(base, tmp_path / "menu")
    host = pl.make_host(pl.current_rg(), dict(config, output_directory=str(work / "out")))
    data = pl.open_progress_manager(host, work / source_name)
    labels = {
        "📂 Open File": "open_file", "🔊 Open Audio File": "open_audio", "🗑️ Delete Audio File": "delete_audio",
        "📋 Copy QA issue": "copy_qa", "📖 Open in EPUB reader": "open_reader",
        "🔁 Retranslate Selected": "retranslate", "⚠️ Resolve QA issue": "resolve_qa",
        "🖼️ Insert Missing Image": "insert_missing_image", "🧹 Remove QA Failed Mark": "remove_qa",
        "🧽 Remove Pending Mark": "remove_pending", "⭐ Remove refinement status": "remove_refinement",
        "Restore In Progress Status": "restore_in_progress",
    }
    module = pl.current_rg()
    for index, info in enumerate(data["chapter_display_info"]):
        item = data["listbox"].item(index)
        if item.isHidden():
            continue
        seen = []

        class Recorder(QMenu):
            def exec(self, *args, **kwargs):
                seen.extend(action.text() for action in self.actions())
                return None

        original = module.QMenu
        module.QMenu = Recorder
        try:
            data["listbox"].clearSelection()
            item.setSelected(True)
            data["listbox"].customContextMenuRequested.emit(data["listbox"].visualItemRect(item).center())
            pl.pump(10, timeout=0.05)
        finally:
            module.QMenu = original
        expected = [labels[text] for text in seen if text in labels]
        assert pa.row_actions(host, data, info) == expected, (info.get("original_filename"), seen)
    data["dialog"].hide()


def test_partial_b_request_matches_the_desktop_request(epub_fixture, tmp_path):
    base, source_name, config = epub_fixture
    work = pl.copy_workspace(base, tmp_path / "pb")
    host = pl.make_host(pl.current_rg(), dict(config, output_directory=str(work / "out")))
    data = pl.open_progress_manager(host, work / source_name)
    ch2 = [i for i in data["chapter_display_info"] if i.get("original_filename") == "chapter0002.xhtml"
           and not i.get("is_chunk_progress")][0]
    host.run_translation_thread = lambda: None
    host._start_single_progress_qa_resolution(data, ch2)
    desktop_request = host._single_qa_resolution_request
    assert desktop_request is None   # the thread did not start, so the desktop clears it
    captured = {}
    host.run_translation_thread = lambda: captured.update(host._single_qa_resolution_request)
    host._start_single_progress_qa_resolution(data, ch2)
    assert pa.build_partial_b_request(data, ch2) == captured
    assert captured["progress_key"] == "2" and captured["output_file"] == "response_chapter0002.html"
    data["dialog"].hide()
