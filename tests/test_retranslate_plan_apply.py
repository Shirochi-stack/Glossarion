"""U7: Retranslate Selected plan/apply (progress_actions) parity and API tests.

The ``retranslate_selected`` generator of ``_add_retranslation_buttons_opf`` was split
into ``progress_actions.plan_retranslation`` (selection normalisation, guards and the
confirmation copy), ``apply_retranslation`` (deletes, resets, sidecar / Machine
Translation cleanup and the merge-write) and ``retranslation_result_message``; the
desktop generator keeps its dialogs and worker thread around them.  The oracle is
Retranslation_GUI frozen at ``U7_BASE_SHA`` (the U6 commit, the parent of the move).

* V: the moved blocks equal the frozen generator (modulo the documented edits);
* F: FILE-SYSTEM goldens -- every scenario runs through the real offscreen Progress
  Manager twice (frozen generator / working tree) with scripted confirmation answers;
  the progress JSON + output tree, every dialog text, the refreshed rows and the
  config are equal;
* M: the GUI-free plan/apply on ``progress_core.build_book_progress`` gives the
  desktop's confirmation copy and the desktop's file-system result;
* Q: the Resolve QA (Partial.b) preflight is shared and behaves like the desktop.
"""

from __future__ import annotations

import ast
import copy
import json
import os
import sys
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

#: Parent of the U7 move (the U6 commit): the frozen Retranslation_GUI oracle.
U7_BASE_SHA = "41814faa95e273e956870bd3aac5a2c6fb7d66b1"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(tmp_path / "_library"))
    monkeypatch.delenv("OUTPUT_DIRECTORY", raising=False)
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    monkeypatch.delenv("EXTRACTION_WORKERS", raising=False)
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "1")
    saved = dict(os.environ)
    yield
    monkeypatch.undo()
    os.environ.clear()
    os.environ.update(saved)


def legacy_rg():
    return pl.legacy_rg(U7_BASE_SHA)


def _legacy_lines(start, end, strip, add=0):
    out = []
    for line in pl.legacy_source_lines(U7_BASE_SHA)[start - 1:end]:
        if line.strip():
            assert line.startswith(" " * strip), (start, line)
            out.append(" " * add + line[strip:])
        else:
            out.append("")
    return "\n".join(out)


def _source(name):
    return (SRC_DIR / f"{name}.py").read_text(encoding="utf-8").replace("\r\n", "\n")


def _top_level(source):
    tree = ast.parse(source)
    lines = source.split("\n")
    return {
        node.name: "\n".join(lines[node.lineno - 1:node.end_lineno])
        for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


def _apply_edits(text, edits):
    for old, new, count in edits:
        assert text.count(old) == count, (old, text.count(old))
        text = text.replace(old, new)
    return text


# ===========================================================================
# Tier V: the moved blocks are the frozen generator's
# ===========================================================================


def _frozen_anchor(text, start_hint):
    """1-based line of ``text`` in the frozen RG (asserted near ``start_hint``)."""
    lines = pl.legacy_source_lines(U7_BASE_SHA)
    for offset in range(0, 400):
        for line_no in (start_hint + offset, start_hint - offset):
            if 0 < line_no <= len(lines) and lines[line_no - 1] == text:
                return line_no
    raise AssertionError(f"{text!r} not near frozen line {start_hint}")


def test_frozen_generator_is_where_the_split_expects_it():
    lines = pl.legacy_source_lines(U7_BASE_SHA)
    assert lines[21913 - 1] == "        def retranslate_selected():"
    assert lines[22268 - 1] == '            yield "run_background"'
    assert lines[22854 - 1] == '            yield "apply_ui"'
    assert lines[22961 - 1] == "        def _launch_retranslate_selected():"
    assert lines[16647 - 1] == "    def _start_single_progress_qa_resolution(self, data, display_info):"


PLAN_BLOCKS = [
    # (frozen start, end, edits) -- dedented 12 -> 4 inside plan_retranslation
    (21925, 22006, []),
    (22095, 22207, [("manual_editing_retranslation = _manual_editing_enabled()",
                     "manual_editing_retranslation = _manual_editing_enabled()", 1)]),
    (22210, 22235, []),
    (22245, 22253, []),
]


@pytest.mark.parametrize("start,end,edits", PLAN_BLOCKS, ids=[f"{b[0]}-{b[1]}" for b in PLAN_BLOCKS])
def test_plan_blocks_are_verbatim(start, end, edits):
    plan_source = _top_level(_source("progress_actions"))["plan_retranslation"]
    block = _apply_edits(_legacy_lines(start, end, 12, 4), edits)
    assert block in plan_source


def test_plan_guards_and_confirmations_keep_the_frozen_texts():
    """Every dialog text of the frozen generator's Qt half is in the plan (as data)."""
    plan_source = _top_level(_source("progress_actions"))["plan_retranslation"]
    frozen = _legacy_lines(21913, 22262, 0)
    for text in (
        '"Metadata Translation Disabled"',
        "\"Enable 'Translate Book Title / Metadata' before requesting metadata regeneration.\"",
        '"Translation Feature Disabled"',
        '"Enable the matching TOC/header translation toggle before "',
        '"requesting regeneration for: "',
        '"Mixed Selection"',
        '"Select metadata.json, TOC.txt, and translated_headers.txt "',
        '"separately from chapter rows when resetting Audio output."',
        '"Confirm TTS Reset"',
        'f"This will delete only generated TTS audio for {count} selected chapter(s), mark them as No TTS, '
        'and leave translated HTML files untouched.\\n\\nContinue?"',
        '"Confirm Retranslation"',
        '"No Selection", "Please select at least one chapter."',
    ):
        assert text in frozen, text
        assert text in plan_source, text


def test_apply_body_is_verbatim():
    apply_source = _top_level(_source("progress_actions"))["apply_retranslation"]
    prologue = _legacy_lines(22273, 22299, 12, 4)
    assert prologue in apply_source
    body = _legacy_lines(22301, 22850, 12, 4)
    anchor = "        sidecar_workers = raw_sidecar_workers if parallel_enabled else 1\n"
    body = _apply_edits(body, [(
        anchor,
        anchor + "        if sidecar_workers_override is not None:\n"
                 "            sidecar_workers = sidecar_workers_override\n",
        1,
    )])
    assert body in apply_source
    # the two path closures the body uses are the frozen ones
    assert _legacy_lines(21748, 21757, 8, 4) in apply_source


def test_result_message_is_verbatim():
    message_source = _top_level(_source("progress_actions"))["retranslation_result_message"]
    block = _apply_edits(_legacy_lines(22869, 22959, 12, 4), [
        ("        self._styled_msgbox(\n"
         "            QMessageBox.Warning,\n"
         "            data.get('dialog', self),\n"
         "            \"Chunk HTML Not Updated\",\n"
         "            warning_message,\n"
         "        )",
         "        return ('warning', \"Chunk HTML Not Updated\", warning_message)", 1),
        ("total_to_translate = len(selected_chapters) + merged_cleared_count",
         "total_to_translate = selected_count + merged_cleared_count", 1),
        ("        self._styled_msgbox(QMessageBox.Information, data.get('dialog', self), \"Success\", success_msg)",
         "        return ('info', \"Success\", success_msg)", 1),
        ("        self._styled_msgbox(QMessageBox.Information, data.get('dialog', self), \"Info\", \"No changes made.\")",
         "        return ('info', \"Info\", \"No changes made.\")", 1),
    ])
    assert block in message_source


def test_qa_preflight_is_verbatim():
    source = _top_level(_source("progress_actions"))["prepare_single_qa_resolution"]
    for start, end in ((16651, 16653), (16668, 16669), (16679, 16685), (16694, 16701), (16708, 16713)):
        assert _legacy_lines(start, end, 8, 4) in source, (start, end)


def test_desktop_generator_calls_the_shared_plan_and_apply():
    source = _source("Retranslation_GUI")
    start = source.index("        def retranslate_selected():")
    end = source.index("        def _launch_retranslate_selected():", start)
    generator = source[start:end]
    assert "plan_retranslation(" in generator
    assert "apply_retranslation(data, plan, linked_choice, owner=self)" in generator
    assert "retranslation_result_message(result)" in generator
    assert generator.index('yield "run_background"') < generator.index("apply_retranslation(")
    assert generator.index("apply_retranslation(") < generator.index('yield "apply_ui"')
    # no file or progress work left in the Qt half
    for gone in ("os.remove(", "_merge_and_write_retranslation_progress", "working_progress",
                 "remove_chunk_segments_from_file", "invalidate_compiled_pdf"):
        assert gone not in generator, gone
    qa = source[source.index("    def _start_single_progress_qa_resolution"):]
    qa = qa[:qa.index("\n    def ", 10)]
    assert "prepare_single_qa_resolution(self, data, display_info)" in qa


# ===========================================================================
# Fixtures
# ===========================================================================


def _html(body):
    return pl._html(body)


def _write(path, text):
    pl._write(path, text)


def _sidecar(out, output_name, source_html, target_html):
    from sdlxliff_sidecar_writer import _write_html_sdlxliff_sidecar

    path = _write_html_sdlxliff_sidecar(
        str(out), output_name, {"original_basename": output_name.replace("response_", "").replace(".html", ".xhtml")},
        source_html, target_html, raise_errors=True, record_freshness=False,
    )
    assert path and os.path.isfile(path)
    return path


def _mt_preview(out, output_name):
    from sdlxliff_review_core import _sdlxliff_machine_translation_path

    path = _sdlxliff_machine_translation_path(str(out), output_name)
    _write(path, json.dumps({"version": 1, "entries": {"x": {"text": "mt"}}}))
    return path


def u7_epub_workspace(base):
    """The U5 EPUB fixture plus RECYCLED TOC/header artifacts, two metadata phases,
    SDLXLIFF sidecars + Machine Translation previews and a chunked chapter whose HTML
    lost its chunk markers."""
    base = Path(base)
    source, out, config = pl.epub_workspace(base)
    _write(out / "translated_headers.txt", "Translated: Headers\n")
    _write(out / "response_chapter0008.html", _html("<p>Chapter eight without markers</p>"))
    progress_file = out / "translation_progress.json"
    prog = json.loads(progress_file.read_text(encoding="utf-8"))
    chapters = prog["chapters"]
    chapters["__translation_artifact__:toc"] = {
        "actual_num": -2, "output_file": "TOC.txt", "original_basename": "TOC.txt", "status": "completed",
        "content_hash": "toc-hash", "model_name": "gpt-x", "last_updated": 1010.0, "is_special": True,
        "special_type": "toc", "translation_artifact_progress_key": "__translation_artifact__:toc",
        "translation_artifact_label": "Table of Contents", "artifact_translation_enabled": True,
    }
    chapters["__translation_artifact__:headers"] = {
        "actual_num": -3, "output_file": "translated_headers.txt", "original_basename": "translated_headers.txt",
        "status": "completed", "content_hash": "hdr-hash", "model_name": "RECYCLED", "last_updated": 1011.0,
        "is_special": True, "special_type": "headers",
        "translation_artifact_progress_key": "__translation_artifact__:headers",
        "translation_artifact_label": "Chapter Headers", "artifact_translation_enabled": True,
    }
    chapters["8"]["content_hash"] = "hash-ch8"
    chapters["8"]["output_file"] = "response_chapter0008.html"
    prog["chapter_chunks"]["hash-ch8"] = {
        "schema_version": 2, "total": 2, "completed": [1, 2],
        "chunks": {"1": "<p>cached eight one</p>", "2": "<p>cached eight two</p>"},
        "chunk_metadata": {},
        "entries": {"1": {"index": 1, "status": "completed"}, "2": {"index": 2, "status": "completed"}},
        "chapter_status": "completed",
    }
    for output_name, source_html, target_html in (
        ("response_chapter0001.html", "<p>원문 1</p>", "<p>Chapter one</p>"),
        ("response_chapter0002.html", "<p>원문 2</p>", "<p>Chapter two</p>"),
    ):
        _sidecar(out, output_name, source_html, target_html)
        _mt_preview(out, output_name)
    pl._write(progress_file, json.dumps(prog, ensure_ascii=False, indent=2))
    pl._set_mtime(base)
    config = dict(
        config,
        batch_translate_headers=True,
        metadata_translation_mode="metadata_separate",
        translate_metadata_fields={"title": True, "creator": True},
    )
    return source, out, config


def many_chapters_workspace(base, total=24, translated=12):
    base = Path(base)
    source = base / "Long.epub"
    names = [f"chapter{index:04d}.xhtml" for index in range(1, total + 1)]
    pl.make_epub(source, [(name, _html(f"<p>원문 {name}</p>")) for name in names])
    out = base / "out" / "Long"
    out.mkdir(parents=True)
    chapters = {}
    for index in range(1, translated + 1):
        output = f"response_chapter{index:04d}.html"
        _write(out / output, _html(f"<p>Chapter {index}</p>"))
        chapters[str(index)] = {
            "actual_num": index, "content_hash": f"h{index}", "output_file": output, "status": "completed",
            "last_updated": 1000.0 + index, "original_basename": names[index - 1], "model_name": "gpt-x",
        }
    _write(out / "translation_progress.json",
           json.dumps({"version": "2.1", "chapters": chapters, "chapter_chunks": {}}, indent=2))
    pl._set_mtime(base)
    config = {"output_directory": str(base / "out"), "translate_book_title": False, "use_toc_ncx": False,
              "batch_translate_headers": False, "special_file_keywords": "", "special_file_exact": "",
              "translate_special_files": False, "translate_all_numbered_html": True}
    return source, out, config


def pdf_compiled_workspace(base):
    """PDF bookmark sections with a compiled HTML/PDF copy and an API-chunked section."""
    from chapter_chunk_progress import wrap_chunk_html
    from pdf_workspace_compiler import wrap_compiled_pdf_source_section

    base = Path(base)
    source = base / "Manual.pdf"
    source.write_bytes(b"%PDF-1.4 fixture")
    out = base / "out" / "Manual"
    out.mkdir(parents=True)
    plan = [
        {"num": 1, "title": "Intro", "start_page": 1, "end_page": 3, "level": 0},
        {"num": 2, "title": "Setup", "start_page": 4, "end_page": 6, "level": 0},
        {"num": 3, "title": "Usage", "start_page": 7, "end_page": 9, "level": 0},
    ]
    chunk_key = "pdf-chunk-hash"
    section_html = {
        1: "<p>s1</p>",
        2: "\n".join(wrap_chunk_html(chunk_key, index, 3, f"<p>s2 part {index}</p>") for index in (1, 2, 3)),
        3: "<p>s3</p>",
    }
    chapters = {}
    compiled_parts = []
    for num, body in section_html.items():
        output = f"response_pdf_section_{num:03d}.html"
        _write(out / output, _html(body))
        chapters[str(num)] = {
            "actual_num": num, "content_hash": chunk_key if num == 2 else f"p{num}", "output_file": output,
            "status": "completed", "last_updated": 1000.0 + num, "original_basename": f"pdf_section_{num}.html",
            "pdf_toc_section": True, "pdf_toc_title": plan[num - 1]["title"], "title": plan[num - 1]["title"],
            "pdf_toc_level": 0, "pdf_start_page": plan[num - 1]["start_page"],
            "pdf_end_page": plan[num - 1]["end_page"], "model_name": "gpt-x",
        }
        compiled_parts.append(wrap_compiled_pdf_source_section(output, body))
    _write(out / "Manual_translated.html", _html("\n".join(compiled_parts)))
    (out / "Manual_translated.pdf").write_bytes(b"%PDF-1.4 compiled")
    prog = {"version": "2.1", "chapters": chapters, "chapter_chunks": {chunk_key: {
        "schema_version": 2, "total": 3, "completed": [1, 2, 3],
        "chunks": {str(i): f"<p>s2 part {i}</p>" for i in (1, 2, 3)}, "chunk_metadata": {},
        "entries": {str(i): {"index": i, "status": "completed"} for i in (1, 2, 3)},
        "chapter_status": "completed",
    }}}
    _write(out / "translation_progress.json", json.dumps(prog, indent=2))
    pl._set_mtime(base)
    config = {"output_directory": str(base / "out"), "translate_book_title": False,
              "pdf_use_toc_sections": True, "use_toc_ncx": False, "batch_translate_headers": False}
    return source, out, config, plan


FIXTURES = {
    "epub": u7_epub_workspace,
    "long": many_chapters_workspace,
    "pdf": pdf_compiled_workspace,
    "subtitle": pl.subtitle_zip_workspace,
    "text": pl.text_workspace,
}


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory):
    roots = {}
    for kind, builder in FIXTURES.items():
        base = tmp_path_factory.mktemp(f"u7_{kind}")
        built = builder(base)
        source, _out, config = built[:3]
        plan = built[3] if len(built) > 3 else None
        roots[kind] = (base, Path(source).name, config, plan)
    return roots


# ===========================================================================
# Running Retranslate Selected through the real dialog
# ===========================================================================


def _is(name, chunk=False):
    def predicate(info):
        return info.get('original_filename') == name and bool(info.get('is_chunk_progress')) == chunk
    return predicate


def _chunk(name, *indices):
    def predicate(info):
        return (info.get('original_filename') == name and info.get('is_chunk_progress')
                and int(info.get('chunk_index') or 0) in indices)
    return predicate


def _names(*names):
    def predicate(info):
        return info.get('original_filename') in names and not info.get('is_chunk_progress')
    return predicate


def _output(*names):
    def predicate(info):
        return os.path.basename(str(info.get('output_file') or '')) in names and not info.get('is_chunk_progress')
    return predicate


def _kinds(*kinds):
    def predicate(info):
        special = info.get('special_type') or (info.get('info') or {}).get('special_type')
        return special in kinds
    return predicate


def _any(*predicates):
    def predicate(info):
        return any(p(info) for p in predicates)
    return predicate


def _set_config(**values):
    def hook(host, _data, _work):
        host.config.update(values)
    return hook


SCENARIOS = {
    # chapters: two existing (one with a merged child, refinement and an SDLXLIFF sidecar + MT preview)
    "chapters_existing": dict(fixture="epub", predicate=_names("chapter0001.xhtml", "chapter0002.xhtml")),
    "chapters_manual_editing": dict(fixture="epub", predicate=_names("chapter0001.xhtml", "chapter0002.xhtml"),
                                    config={"retranslation_manual_editing": True}),
    "chapter_missing_and_existing": dict(fixture="epub", predicate=_names("chapter0004.xhtml", "notice.xhtml",
                                                                          "title.xhtml")),
    "chapter_in_progress": dict(fixture="epub", predicate=_names("chapter0005.xhtml", "chapter0006.xhtml")),
    "chapter_untracked_output": dict(fixture="epub", predicate=_names("chapter0007.xhtml")),
    # chunks
    "chunk_one_segment": dict(fixture="epub", predicate=_chunk("chapter0003.xhtml", 2)),
    "chunk_every_segment": dict(fixture="epub", predicate=_chunk("chapter0003.xhtml", 1, 2, 3)),
    "chunk_parent_absorbs_children": dict(fixture="epub", predicate=_any(_names("chapter0003.xhtml"),
                                                                         _chunk("chapter0003.xhtml", 1, 3))),
    "chunk_segment_not_found": dict(fixture="epub", predicate=_chunk("chapter0008.xhtml", 1)),
    "chunk_mixed_files": dict(fixture="epub", predicate=_any(_chunk("chapter0003.xhtml", 1, 2, 3),
                                                             _chunk("chapter0008.xhtml", 2))),
    # metadata phases
    "metadata_all": dict(fixture="epub", predicate=_kinds("metadata"), show_special=True),
    "metadata_one_phase": dict(fixture="epub", show_special=True,
                               predicate=lambda i: i.get('progress_key') == "__metadata__:title"),
    "metadata_disabled": dict(fixture="epub", predicate=_kinds("metadata"), show_special=True,
                              after_open=_set_config(translate_book_title=False)),
    # TOC / header artifacts (RECYCLED pair)
    "artifact_recycled_delete_both": dict(fixture="epub", predicate=_kinds("toc"), linked="both", show_special=True),
    "artifact_recycled_keep_counterpart": dict(fixture="epub", predicate=_kinds("toc"), linked="selected_only",
                                               show_special=True),
    "artifact_recycled_cancel": dict(fixture="epub", predicate=_kinds("headers"), linked="cancel",
                                     show_special=True),
    "artifact_pair_both_selected": dict(fixture="epub", predicate=_kinds("toc", "headers"), show_special=True),
    "artifact_toggle_off": dict(fixture="epub", predicate=_kinds("toc"), show_special=True,
                                after_open=_set_config(use_toc_ncx=False)),
    "artifact_and_chapter": dict(fixture="epub", predicate=_any(_kinds("headers"), _names("chapter0002.xhtml")),
                                 linked="both", show_special=True),
    # confirmation answered No
    "declined": dict(fixture="epub", predicate=_names("chapter0001.xhtml"), confirm=False),
    # audio output
    "audio_reset_tts": dict(fixture="epub", predicate=_names("chapter0001.xhtml", "chapter0003.xhtml"), audio=True),
    "audio_reset_tts_declined": dict(fixture="epub", predicate=_names("chapter0001.xhtml"), audio=True,
                                     confirm=False),
    "audio_mixed_selection": dict(fixture="epub", predicate=_any(_kinds("metadata"), _names("chapter0001.xhtml")),
                                  audio=True, show_special=True),
    "audio_metadata_only": dict(fixture="epub", predicate=_kinds("metadata"), audio=True, show_special=True),
    # more than ten rows
    "many_mixed": dict(fixture="long", predicate=lambda i: not i.get('is_chunk_progress')
                       and str(i.get('original_filename') or '').startswith('chapter')),
    "many_existing": dict(fixture="long", predicate=lambda i: i.get('status') != 'not_translated'
                          and str(i.get('original_filename') or '').startswith('chapter')),
    "many_existing_manual": dict(fixture="long", predicate=lambda i: i.get('status') != 'not_translated'
                                 and str(i.get('original_filename') or '').startswith('chapter'),
                                 config={"retranslation_manual_editing": True}),
    "many_missing": dict(fixture="long", predicate=lambda i: i.get('status') == 'not_translated'),
    # PDF bookmark sections (compiled HTML/PDF invalidation)
    "pdf_section": dict(fixture="pdf", predicate=_output("response_pdf_section_001.html")),
    "pdf_api_chunk": dict(fixture="pdf", predicate=lambda i: i.get('is_chunk_progress')
                          and int(i.get('chunk_index') or 0) == 2),
    "pdf_api_every_chunk": dict(fixture="pdf", predicate=lambda i: bool(i.get('is_chunk_progress'))),
    "pdf_chunked_parent": dict(fixture="pdf", predicate=_output("response_pdf_section_002.html")),
    # subtitles / plain text
    "subtitle_existing": dict(fixture="subtitle", predicate=lambda i: i.get('original_filename') == 'ep01.srt'),
    "subtitle_missing": dict(fixture="subtitle", predicate=lambda i: i.get('original_filename') == 'ep02.srt'),
    "text_sections": dict(fixture="text", predicate=lambda i: str(i.get('output_file') or '').endswith(
        ('response_section_1.txt', 'response_section_2.txt'))),
}


def _answers(record, *, confirm=True, linked=None):
    from PySide6.QtWidgets import QMessageBox

    def styled(icon, parent, title, message, buttons=None):
        record.append(("msgbox", str(icon), title, message))
        if buttons is not None and title in ("Confirm Retranslation", "Confirm TTS Reset"):
            return QMessageBox.Yes if confirm else QMessageBox.No
        return QMessageBox.Yes

    def linked_choice(parent, message, counterpart_filename):
        record.append(("linked", counterpart_filename, message))
        return linked or "cancel"

    return styled, linked_choice


def _normalize_messages(messages, workspace):
    text = json.dumps(messages, ensure_ascii=False)
    for raw in (str(workspace), str(workspace).replace("\\", "/")):
        text = text.replace(json.dumps(raw)[1:-1], "<ROOT>")
    return json.loads(text)


def _open(module, fixture, work, *, config=None, audio=False, show_special=False):
    base, source_name, base_config, plan = fixture
    work = pl.copy_workspace(base, work)
    cfg = dict(base_config, output_directory=str(work / "out"), **(config or {}))
    host = pl.make_host(module, cfg, **({"output_mode_var": "audio"} if audio else {}))
    if plan is not None:
        host._pdf_outline_progress_plan = lambda file_path, _plan=plan: [dict(s) for s in _plan]
    data = pl.open_progress_manager(host, work / source_name)
    assert data is not None
    if show_special:
        data['show_special_files_cb'].setChecked(True)
        pl.pump(20, timeout=0.2)
    return work, cfg, host, data


def _wait_for_retranslate(data, timeout=15.0):
    pl.pump(20, until=lambda: not data.get('_retranslate_selected_active')
            and '_retranslate_selected_bridge' not in data, timeout=timeout)
    # the debounced refresh the generator starts, then its list population
    debounce = data.get('_progress_watch_debounce')
    pl.pump(20, until=lambda: debounce is None or not debounce.isActive(), timeout=3.0)
    pl.pump(20, timeout=0.2)
    pl.pump(20, until=lambda: not data.get('_listbox_populate_active') and not data.get('_prefetch_running'),
            timeout=5.0)
    pl.pump(20, timeout=0.1)


def _run(module, fixture, work, *, predicate, confirm=True, linked=None, config=None, audio=False,
         show_special=False, after_open=None, before_click=None):
    saved_env = dict(os.environ)
    try:
        work, cfg, host, data = _open(module, fixture, work, config=config, audio=audio,
                                      show_special=show_special)
        if after_open is not None:
            after_open(host, data, work)
        record = []
        host._styled_msgbox, host._recycled_artifact_retranslation_choice = _answers(
            record, confirm=confirm, linked=linked)
        pl.MESSAGES.clear()
        selected = pl.select_rows(data, predicate)
        assert selected is not None, "no row matched"
        if before_click is not None:
            before_click(work, data)
        button = "Reset TTS Selected" if audio else "Retranslate Selected"
        pl.click_button(data, button)
        _wait_for_retranslate(data)
        assert not data.get('_retranslate_selected_active')
        result = {
            "tree": pl.tree_snapshot(work / "out", workspace=work),
            "messages": _normalize_messages(record + list(pl.MESSAGES), work),
            "rows": pl.view_snapshot(data),
            "config": {k: v for k, v in host.config.items() if k != "output_directory"},
            "work": work,
            "data": data,
        }
        data['dialog'].hide()
        return result
    finally:
        os.environ.clear()
        os.environ.update(saved_env)


def _diff(x, y, path=""):
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
    return [] if x == y else [f"{path}: {x!r:.160} != {y!r:.160}"]


_LEGACY_CACHE = {}


def _legacy_result(name, fixture_root, tmp_path_factory):
    if name not in _LEGACY_CACHE:
        spec = dict(SCENARIOS[name])
        fixture = fixture_root[spec.pop("fixture")]
        work = tmp_path_factory.mktemp(f"legacy_{name}") / "w"
        _LEGACY_CACHE[name] = _run(legacy_rg(), fixture, work, **spec)
    return _LEGACY_CACHE[name]


# ===========================================================================
# Tier F: file-system goldens against the frozen generator
# ===========================================================================


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_retranslate_matches_frozen_desktop(name, fixture_root, tmp_path, tmp_path_factory):
    spec = dict(SCENARIOS[name])
    fixture = fixture_root[spec.pop("fixture")]
    legacy = _legacy_result(name, fixture_root, tmp_path_factory)
    current = _run(pl.current_rg(), fixture, tmp_path / "current", **spec)
    assert current["messages"] == legacy["messages"]
    assert current["tree"] == legacy["tree"], _diff(legacy["tree"], current["tree"])
    assert current["rows"] == legacy["rows"]
    assert current["config"] == legacy["config"]
    assert legacy["messages"], "Retranslate Selected showed nothing"


def _progress(result, rel):
    return result["tree"][rel][1]["chapters"]


def test_scenarios_do_what_they_should(fixture_root, tmp_path_factory):
    """Spot checks of the goldens (both sides are equal by the parity test)."""
    r = _legacy_result("chapters_existing", fixture_root, tmp_path_factory)
    chapters = _progress(r, "Book/translation_progress.json")
    assert chapters["1"]["status"] == "pending" and "refinement_status" not in chapters["1"]
    assert "9" not in chapters                       # merged child of chapter 1 dropped
    assert "Book/response_chapter0001.html" not in r["tree"]
    assert "Book/SDLXLIFF/response_chapter0001.html.sdlxliff" not in r["tree"]
    assert not any(k.startswith("Book/SDLXLIFF/Machine_Translation/") for k in r["tree"])

    r = _legacy_result("chapters_manual_editing", fixture_root, tmp_path_factory)
    chapters = _progress(r, "Book/translation_progress.json")
    assert chapters["1"]["manual_editing_pending"] is True
    assert "Book/SDLXLIFF/response_chapter0001.html.sdlxliff" in r["tree"]

    r = _legacy_result("chunk_one_segment", fixture_root, tmp_path_factory)
    ledger = r["tree"]["Book/translation_progress.json"][1]["chapter_chunks"]["hash-ch3"]
    assert "2" not in ledger["chunks"] and ledger["entries"]["2"]["status"] == "pending"
    html = r["tree"]["Book/response_chapter0003.html"][1]
    assert "<p>a</p>" in html and "<p>b</p>" not in html

    r = _legacy_result("chunk_every_segment", fixture_root, tmp_path_factory)
    assert "Book/response_chapter0003.html" not in r["tree"]

    r = _legacy_result("chunk_segment_not_found", fixture_root, tmp_path_factory)
    assert any(m[2] == "Chunk HTML Not Updated" for m in r["messages"] if m[0] == "msgbox")

    r = _legacy_result("metadata_all", fixture_root, tmp_path_factory)
    assert "Book/metadata.json" not in r["tree"]
    r = _legacy_result("metadata_one_phase", fixture_root, tmp_path_factory)
    assert "Book/metadata.json" in r["tree"]

    r = _legacy_result("artifact_recycled_delete_both", fixture_root, tmp_path_factory)
    assert "Book/TOC.txt" not in r["tree"] and "Book/translated_headers.txt" not in r["tree"]
    r = _legacy_result("artifact_recycled_keep_counterpart", fixture_root, tmp_path_factory)
    assert "Book/TOC.txt" not in r["tree"] and "Book/translated_headers.txt" in r["tree"]
    r = _legacy_result("artifact_recycled_cancel", fixture_root, tmp_path_factory)
    assert "Book/TOC.txt" in r["tree"] and "Book/translated_headers.txt" in r["tree"]

    r = _legacy_result("pdf_section", fixture_root, tmp_path_factory)
    assert "Manual/Manual_translated.pdf" not in r["tree"]
    assert "GLOSSARION_PDF_SOURCE" in r["tree"]["Manual/Manual_translated.html"][1]
    r = _legacy_result("pdf_api_chunk", fixture_root, tmp_path_factory)
    assert "s2 part 2" not in r["tree"]["Manual/Manual_translated.html"][1]
    assert "s2 part 1" in r["tree"]["Manual/Manual_translated.html"][1]

    r = _legacy_result("declined", fixture_root, tmp_path_factory)
    assert "Book/response_chapter0001.html" in r["tree"]


# ===========================================================================
# Tier M: the GUI-free plan / apply on build_book_progress
# ===========================================================================


MOBILE_SCENARIOS = (
    "chapters_existing", "chapters_manual_editing", "chapter_missing_and_existing", "chunk_one_segment",
    "chunk_every_segment", "chunk_parent_absorbs_children", "chunk_segment_not_found", "metadata_all",
    "metadata_one_phase", "artifact_recycled_delete_both", "artifact_recycled_keep_counterpart",
    "artifact_pair_both_selected", "many_mixed", "many_existing_manual", "pdf_section", "pdf_api_chunk",
    "subtitle_existing", "text_sections",
)


def _mobile_book(fixture, work, config=None, audio=False):
    base, source_name, base_config, plan = fixture
    work = pl.copy_workspace(base, work)
    cfg = dict(base_config, output_directory=str(work / "out"), **(config or {}))
    owner = pc.ProgressOwner(cfg)
    if audio:
        owner.output_mode_var = "audio"
    if plan is not None:
        owner._pdf_outline_progress_plan = lambda file_path, _plan=plan: [dict(s) for s in _plan]
    book = pc.build_book_progress(str(work / source_name), cfg, owner=owner, show_special_files=True)
    assert book is not None
    return work, book


@pytest.mark.parametrize("name", MOBILE_SCENARIOS)
def test_mobile_plan_apply_matches_desktop(name, fixture_root, tmp_path, tmp_path_factory):
    spec = dict(SCENARIOS[name])
    fixture = fixture_root[spec.pop("fixture")]
    legacy = _legacy_result(name, fixture_root, tmp_path_factory)
    work, book = _mobile_book(fixture, tmp_path / "m", spec.get("config"))
    predicate = spec["predicate"]
    rows = [index for index, info in enumerate(book.data["chapter_display_info"]) if predicate(info)]
    manual = book.owner._get_retranslation_manual_editing_state()
    plan = pa.plan_retranslation(book, rows, {"manual_editing": manual})
    assert plan.mode == "retranslate"
    desktop_confirm = [m for m in legacy["messages"]
                       if (m[0] == "msgbox" and m[2] == "Confirm Retranslation") or m[0] == "linked"]
    assert len(desktop_confirm) == 1
    expected_text = desktop_confirm[0][3] if desktop_confirm[0][0] == "msgbox" else desktop_confirm[0][2]
    assert _normalize_messages([plan.confirm_message], work) == [expected_text]
    if plan.needs_linked_choice:
        assert desktop_confirm[0][0] == "linked" and desktop_confirm[0][1] == plan.counterpart_filename
    result = pa.apply_retranslation(book, plan, spec.get("linked"), sidecar_workers=None)
    kind, title, message = pa.retranslation_result_message(result)
    desktop_result = [m for m in legacy["messages"] if m[0] == "msgbox" and m[2] in ("Success", "Info",
                                                                                    "Chunk HTML Not Updated")]
    assert _normalize_messages([[title, message]], work) == [[desktop_result[-1][2], desktop_result[-1][3]]]
    mobile_tree = pl.tree_snapshot(work / "out", workspace=work)
    assert mobile_tree == legacy["tree"], _diff(legacy["tree"], mobile_tree)


def test_mobile_plan_refusals_and_tts_mode(fixture_root, tmp_path):
    work, book = _mobile_book(fixture_root["epub"], tmp_path / "r")
    infos = book.data["chapter_display_info"]
    plan = pa.plan_retranslation(book, [], {})
    assert plan.mode == "refused" and plan.refusal == ("warning", "No Selection", "Please select at least one chapter.")
    metadata_rows = [i for i, info in enumerate(infos) if _kinds("metadata")(info)]
    book.owner.config["translate_book_title"] = False
    plan = pa.plan_retranslation(book, metadata_rows, {})
    assert plan.refusal[1] == "Metadata Translation Disabled"
    book.owner.config["translate_book_title"] = True
    book.owner.output_mode_var = "audio"
    chapter_rows = [i for i, info in enumerate(infos) if _names("chapter0001.xhtml")(info)]
    plan = pa.plan_retranslation(book, metadata_rows + chapter_rows, {})
    assert plan.refusal[1] == "Mixed Selection"
    plan = pa.plan_retranslation(book, chapter_rows, {})
    assert plan.mode == "reset_tts" and plan.confirm_title == "Confirm TTS Reset"
    with pytest.raises(ValueError):
        pa.apply_retranslation(book, plan)
    # rows given as RowPresentation objects resolve to the same selection
    plan_from_rows = pa.plan_retranslation(book, [book.rows[i] for i in chapter_rows], {})
    assert plan_from_rows.selected_chapters == plan.selected_chapters


def test_retranslate_rows_skips_a_needed_linked_choice(fixture_root, tmp_path):
    work, book = _mobile_book(fixture_root["epub"], tmp_path / "l")
    rows = [i for i, info in enumerate(book.data["chapter_display_info"]) if _kinds("toc")(info)]
    plan, result = pa.retranslate_rows(book, rows, {})
    assert plan.needs_linked_choice and result is None
    assert [c for c, _label in plan.linked_choice_labels] == ["both", "selected_only", "cancel"]
    assert (work / "out" / "Book" / "TOC.txt").exists()
    plan, result = pa.retranslate_rows(book, rows, {}, linked_choice="both")
    assert result is not None and result.deleted_count == 2


def test_apply_keeps_a_concurrent_translator_save(fixture_root, tmp_path):
    """The merge-write (unchanged) keeps a translator save made after the view's read."""
    work, book = _mobile_book(fixture_root["epub"], tmp_path / "c")
    progress_file = Path(book.progress_file)
    saved = json.loads(progress_file.read_text(encoding="utf-8"))
    saved["chapters"]["10"] = {"actual_num": 10, "status": "in_progress", "output_file": "response_chapter0010.html"}
    pc.write_progress_atomic(progress_file, saved)
    rows = [i for i, info in enumerate(book.data["chapter_display_info"]) if _names("chapter0002.xhtml")(info)]
    plan, result = pa.retranslate_rows(book, rows, {})
    assert result.status_reset_count == 1
    final = json.loads(progress_file.read_text(encoding="utf-8"))
    assert final["chapters"]["10"]["status"] == "in_progress"
    assert final["chapters"]["2"]["status"] == "pending"


# ===========================================================================
# Tier Q: Resolve QA (raw foreign text) preflight
# ===========================================================================


class _Thread:
    def __init__(self, alive):
        self._alive = alive

    def is_alive(self):
        return self._alive


def test_qa_preflight_refusals_and_run_state(fixture_root, tmp_path):
    work, book = _mobile_book(fixture_root["epub"], tmp_path / "q")
    infos = book.data["chapter_display_info"]
    ch1 = [i for i in infos if _names("chapter0001.xhtml")(i)][0]
    ch2 = [i for i in infos if _names("chapter0002.xhtml")(i)][0]
    owner = book.owner
    outcome = pa.prepare_single_qa_resolution(owner, book.data, ch1)
    assert outcome["refusal"][1] == "QA Issue Already Resolved" and outcome["refresh"]
    owner.translation_thread = _Thread(True)
    outcome = pa.prepare_single_qa_resolution(owner, book.data, ch2)
    assert outcome["refusal"] == ("info", "Process Running",
                                  "Wait for the current translation or glossary process to finish first.")
    assert getattr(owner, "_single_qa_resolution_request", None) is None
    owner.translation_thread = _Thread(False)
    owner._metadata_only_run = True
    owner._single_chapter_filter = "x"
    owner._force_stream_all = True
    outcome = pa.prepare_single_qa_resolution(owner, book.data, ch2)
    assert outcome["ok"] and outcome["refusal"] is None
    assert owner._single_qa_resolution_request == pa.build_partial_b_request(book.data, ch2) == outcome["request"]
    assert owner.selected_files == [outcome["source_path"]] and owner.current_file_index == 0
    assert (owner._metadata_only_run, owner._single_chapter_filter, owner._force_stream_all) == (False, None, False)
    assert outcome["log"] == "⚠️ Queued Partial.b QA resolution for response_chapter0002.html only"
    missing = dict(book.data, file_path=str(work / "gone.epub"))
    outcome = pa.prepare_single_qa_resolution(owner, missing, ch2)
    assert outcome["refusal"][:2] == ("error", "Source File Missing")


def _qa_desktop(module, fixture, work, prepare):
    work, cfg, host, data = _open(module, fixture, work)
    ch2 = [i for i in data["chapter_display_info"] if _names("chapter0002.xhtml")(i)][0]
    pl.MESSAGES.clear()
    captured = {}
    host.run_translation_thread = lambda: captured.update(
        request=copy.deepcopy(host._single_qa_resolution_request),
        files=list(host.selected_files),
        flags=(host.current_file_index, host._metadata_only_run, host._single_chapter_filter,
               host._force_stream_all))
    prepare(host, data, ch2)
    started = host._start_single_progress_qa_resolution(data, ch2)
    out = {
        "started": started,
        "captured": _normalize_messages(captured, work),
        "messages": _normalize_messages(list(pl.MESSAGES), work),
        "logs": _normalize_messages(list(host.logs), work),
        "request_after": getattr(host, "_single_qa_resolution_request", "<unset>"),
    }
    data["dialog"].hide()
    return out


@pytest.mark.parametrize("case", ["ready", "busy", "resolved", "source_missing"])
def test_qa_resolution_desktop_matches_frozen(case, fixture_root, tmp_path):
    def prepare(host, data, ch2):
        host.translation_thread = _Thread(case == "busy")
        host._metadata_only_run = True
        host._single_chapter_filter = "only"
        host._force_stream_all = True
        if case == "resolved":
            data["prog"]["chapters"]["2"]["qa_issues_found"] = ["llm_token_issue_empty_attr"]
        if case == "source_missing":
            data["file_path"] = str(Path(data["file_path"]).with_name("gone.epub"))

    legacy = _qa_desktop(legacy_rg(), fixture_root["epub"], tmp_path / "legacy", prepare)
    current = _qa_desktop(pl.current_rg(), fixture_root["epub"], tmp_path / "current", prepare)
    assert current == legacy
    if case == "ready":
        assert legacy["captured"]["request"]["output_file"] == "response_chapter0002.html"
    else:
        assert legacy["messages"] and not legacy["captured"]


# ===========================================================================
# Image-folder Progress Manager (shared scan / Mark as Skipped / Delete Selected)
# ===========================================================================


def image_folder_fixture(base, layout):
    """An image folder with its output; ``layout``: the progress file's shape
    (``flat`` / ``nested`` pre-2.1, ``v21`` chapters, ``none``)."""
    base = Path(base)
    folder = base / "Pics"
    folder.mkdir(parents=True)
    for name in ("a.png", "b.png", "c.png"):
        (folder / name).write_bytes(b"\x89PNG fixture " + name.encode())
    out = base / "Pics_translated"
    (out / "images").mkdir(parents=True)
    for index, name in ((1, "a"), (2, "b"), (3, "c")):
        _write(out / f"response_{index:03d}_{name}.html", _html(f"<p>{name}</p>"))
    (out / "images" / "cover.png").write_bytes(b"\x89PNG cover")
    entries = {
        f"hash{name.upper()}": {"output_file": f"response_{index:03d}_{name}.html", "status": "completed"}
        for index, name in ((1, "a"), (2, "b"))
    }
    if layout == "flat":
        prog = dict(entries, version="1.0")
    elif layout == "nested":
        prog = {"images": entries, "version": "2.0"}
    elif layout == "v21":
        prog = {"version": "2.1", "chapters": {
            str(i): dict(entry, actual_num=i) for i, entry in enumerate(entries.values(), start=1)}}
    else:
        prog = None
    if prog is not None:
        _write(out / "translation_progress.json", json.dumps(prog, indent=2))
    pl._set_mtime(base)
    return folder, out


IMAGE_LAYOUTS = ("flat", "nested", "v21", "none")


@pytest.fixture(scope="module")
def image_roots(tmp_path_factory):
    roots = {}
    for layout in IMAGE_LAYOUTS:
        base = tmp_path_factory.mktemp(f"u7_images_{layout}")
        image_folder_fixture(base, layout)
        roots[layout] = base
    return roots


def _image_dialog(module, base, work):
    from PySide6.QtWidgets import QListWidget

    work = pl.copy_workspace(base, work)
    host = pl.make_host(module, {"output_directory": str(work)})
    record = []
    host._styled_msgbox, _unused = _answers(record)
    folder = work / "Pics"
    host._force_retranslation_images_folder(str(folder))
    dialog = host._image_retranslation_dialog_cache[os.path.abspath(str(folder))]
    listbox = dialog.findChild(QListWidget)
    # the dialog clicks Refresh once when it opens (rescan after 50 ms, the button
    # comes back after the 0.8 s minimum animation): act on the refreshed list
    refresh = dialog._refresh_button
    _pump_for(0.2)
    pl.pump(20, until=refresh.isEnabled, timeout=5.0)
    _pump_for(0.1)
    return work, host, dialog, listbox, record


def _pump_for(seconds):
    import time as _time

    deadline = _time.monotonic() + seconds
    pl.pump(20, until=lambda: _time.monotonic() >= deadline, timeout=seconds + 1.0)


def _image_click(dialog, listbox, text, predicate):
    from PySide6.QtWidgets import QPushButton

    listbox.clearSelection()
    for index in range(listbox.count()):
        if predicate(listbox.item(index).text()):
            listbox.item(index).setSelected(True)
    for button in dialog.findChildren(QPushButton):
        if button.text().strip() == text:
            button.click()
            _pump_for(0.3)
            return
    raise AssertionError(text)


def _image_run(module, base, work, actions):
    saved_env = dict(os.environ)
    try:
        work, host, dialog, listbox, record = _image_dialog(module, base, work)
        snapshots = [[listbox.item(i).text() for i in range(listbox.count())]]
        for text, predicate in actions:
            _image_click(dialog, listbox, text, predicate)
            snapshots.append([listbox.item(i).text() for i in range(listbox.count())])
        out = {
            "tree": pl.tree_snapshot(work, workspace=work),
            "messages": _normalize_messages(record, work),
            "lists": snapshots,
        }
        dialog.hide()
        return out
    finally:
        os.environ.clear()
        os.environ.update(saved_env)


IMAGE_ACTIONS = {
    "mark_skipped": [("Mark as Skipped", lambda t: " a " in t or "| a |" in t)],
    "delete_selected": [("Delete Selected", lambda t: "| b |" in t or "Cover" in t)],
    "skip_then_delete": [("Mark as Skipped", lambda t: "| a |" in t),
                         ("Delete Selected", lambda t: "| c |" in t)],
}


@pytest.mark.parametrize("layout", IMAGE_LAYOUTS)
@pytest.mark.parametrize("action", sorted(IMAGE_ACTIONS))
def test_image_folder_actions_match_frozen_desktop(layout, action, image_roots, tmp_path):
    base = image_roots[layout]
    legacy = _image_run(legacy_rg(), base, tmp_path / "legacy", IMAGE_ACTIONS[action])
    current = _image_run(pl.current_rg(), base, tmp_path / "current", IMAGE_ACTIONS[action])
    assert current["lists"] == legacy["lists"]
    assert current["messages"] == legacy["messages"]
    assert current["tree"] == legacy["tree"], _diff(legacy["tree"], current["tree"])
    assert legacy["messages"]


def test_image_folder_progress_hash_removal_follows_the_layout(image_roots, tmp_path):
    """Pre-2.1 layouts lose the deleted image's entry; a v2.1 file is never matched
    (DISCREPANCIES U5 desktop bug 3, kept)."""
    for layout in ("flat", "nested", "v21"):
        result = _image_run(pl.current_rg(), image_roots[layout], tmp_path / layout,
                            IMAGE_ACTIONS["delete_selected"])
        prog = result["tree"]["Pics_translated/translation_progress.json"][1]
        if layout == "flat":
            assert "hashB" not in prog and "hashA" in prog
        elif layout == "nested":
            assert "hashB" not in prog["images"] and "hashA" in prog["images"]
        else:
            assert len(prog["chapters"]) == 2


def test_mobile_image_folder_progress_matches_the_dialog(image_roots, tmp_path):
    for layout in IMAGE_LAYOUTS:
        desktop = _image_run(pl.current_rg(), image_roots[layout], tmp_path / f"d_{layout}", [])
        work = pl.copy_workspace(image_roots[layout], tmp_path / f"m_{layout}")
        data, problem = pc.build_image_folder_progress(
            str(work / "Pics"), {"output_directory": str(work)}, script_dir=str(work / "nowhere"))
        assert problem is None
        assert data["rows"] == desktop["lists"][0]
        # Delete Selected through the shared functions == the dialog's result
        indices = [i for i, text in enumerate(data["rows"]) if "| b |" in text or "Cover" in text]
        assert pc.image_folder_delete_confirmation(data["file_info"], indices) == (
            "This will delete 1 translated image(s) and 1 cover image(s).\n\nContinue?")
        deleted = pc.delete_image_folder_items(data["progress_file"], data["progress_data"],
                                               data["file_info"], indices)
        assert deleted == 2
        expected = _image_run(legacy_rg(), image_roots[layout], tmp_path / f"l_{layout}",
                              IMAGE_ACTIONS["delete_selected"])["tree"]
        assert pl.tree_snapshot(work, workspace=work) == expected
    empty = tmp_path / "empty_case"
    (empty / "Lone").mkdir(parents=True)
    data, problem = pc.build_image_folder_progress(str(empty / "Lone"), {"output_directory": str(empty)},
                                                   script_dir=str(empty / "app"))
    assert data is None and problem[1] == "Info"
    assert problem[2].startswith("No translation output found for 'Lone'.")


# ===========================================================================
# Manual glossary refinement (glossary_progress_core plan / run)
# ===========================================================================


import glossary_progress_core as gpc  # noqa: E402

_GLOSSARY_CSV = (
    "type,raw_name,translated_name,gender,description\n"
    "character,김철수,Kim Cheolsu,male,\n"
    "character,이영희,Lee Younghee,female,\n"
    "terms,마나,Mana,,\n"
)


def test_manual_refinement_code_is_verbatim():
    funcs = _top_level(_source("glossary_progress_core"))
    run_body = _legacy_lines(15100, 15196, 4, 0)
    assert run_body in funcs["run_manual_glossary_refinement"]
    prepare = _apply_edits(_legacy_lines(18239, 18324, 8, 0), [
        ("        self._show_message(\n"
         "            'warning',\n"
         "            'Glossary Not Found',\n"
         "            'No saved glossary file was found for this book.',\n"
         "            parent=parent,\n"
         "        )\n"
         "        return False",
         "        preview.refusal = (\n"
         "            'warning',\n"
         "            'Glossary Not Found',\n"
         "            'No saved glossary file was found for this book.',\n"
         "        )\n"
         "        return preview", 1),
        ("        self._show_message('error', 'Glossary Read Failed', str(exc), parent=parent)\n        return False",
         "        preview.refusal = ('error', 'Glossary Read Failed', str(exc))\n        return preview", 1),
        ("        self._show_message(\n"
         "            'info',\n"
         "            'Nothing to Refine',\n"
         "            'The selected entry type(s) contain no glossary entries.',\n"
         "            parent=parent,\n"
         "        )\n"
         "        return False",
         "        preview.refusal = (\n"
         "            'info',\n"
         "            'Nothing to Refine',\n"
         "            'The selected entry type(s) contain no glossary entries.',\n"
         "        )\n"
         "        return preview", 1),
        ("        self._show_message('error', 'Refinement Preview Failed', str(exc), parent=parent)\n"
         "        return False",
         "        preview.refusal = ('error', 'Refinement Preview Failed', str(exc))\n"
         "        return preview", 1),
    ])
    assert prepare in funcs["prepare_manual_glossary_refinement"]
    rg = _source("Retranslation_GUI")
    runner = rg[rg.index("    def _run_manual_glossary_refinement("):]
    runner = runner[:runner.index("\n    @staticmethod")]
    assert "run_manual_glossary_refinement(\n            self," in runner and "extractor" not in runner
    closure = rg[rg.index("        def _confirm_manual_glossary_refinement("):]
    closure = closure[:closure.index("        def _build_gp_panel(")]
    assert "prepare_manual_glossary_refinement(" in closure
    assert "finish_manual_glossary_refinement(" in closure
    assert "parse_glossary_file(" not in closure


class _RefineOwner:
    def __init__(self, config, model="gpt-x"):
        self.config = config
        self.model_var = model
        self.messages = []
        self.logs = []

    def _show_message(self, kind, title, message, parent=None):
        self.messages.append((kind, title, message))

    def append_log(self, message):
        self.logs.append(message)


def _refine_workspace(base, csv_text=_GLOSSARY_CSV):
    gdir = Path(base) / "out" / "Glossary" / "Book"
    gdir.mkdir(parents=True)
    glossary = gdir / "Book_glossary.csv"
    glossary.write_text(csv_text, encoding="utf-8")
    progress = gdir / "Book_glossary_progress.json"
    progress.write_text(json.dumps({"chapters": {}}), encoding="utf-8")
    return glossary, progress


def _frozen_plan_step(owner, source_path, progress_path, selected_types, model, find, active):
    """The frozen confirm closure's plan step (RG 18239-18324) as a function."""
    function = pl.block_function(
        legacy_rg(), 18239, 18324,
        ["self", "parent", "source_path", "progress_path", "selected_types", "model",
         "_find_glossary_for_refinement", "_active_glossary_refinement_types", "_refinement_type_key"],
        dedent=12, sha=U7_BASE_SHA,
    )
    return function(owner, None, source_path, progress_path, selected_types, model, find, active,
                    gpc._glossary_refinement_type_key)


def _plan_view(plan):
    if plan is None:
        return None
    return (plan.total_chunks, plan.total_payload_tokens, dict(plan.per_type_counts))


REFINE_CASES = {
    "characters": dict(types=["character"]),
    "all_types": dict(types=["character", "terms", "locations"]),
    "case_insensitive": dict(types=["Character", "TERMS"]),
    "empty_type": dict(types=["locations"]),
    "separate_mode": dict(types=["character", "terms"], config={"glossary_refinement_chunking_mode": "separate"}),
    "custom_types": dict(types=["character", "spells"],
                         config={"custom_entry_types": {"character": {"enabled": True}, "spells": {"enabled": True},
                                                        "terms": {"enabled": False}}}),
    "custom_fields_json": dict(types=["terms"], config={"custom_glossary_fields": '["notes"]'}),
    "no_glossary": dict(types=["character"], glossary=False),
}


@pytest.mark.parametrize("case", sorted(REFINE_CASES))
def test_manual_refinement_plan_matches_the_frozen_closure(case, tmp_path, monkeypatch):
    spec = REFINE_CASES[case]
    glossary, progress = _refine_workspace(tmp_path)
    if spec.get("glossary") is False:
        glossary.unlink()
    config = dict({"output_directory": str(tmp_path / "out")}, **spec.get("config", {}))
    source = str(tmp_path / "Book.epub")
    locator = gpc.glossary_progress_locator(_RefineOwner(config), source, None)
    monkeypatch.delenv("GLOSSARY_CUSTOM_FIELDS", raising=False)

    legacy_owner = _RefineOwner(config)
    legacy = _frozen_plan_step(legacy_owner, source, str(progress), list(spec["types"]), "gpt-x",
                               locator._find_glossary_for_refinement, locator._active_glossary_refinement_types)
    legacy_env = os.environ.get("GLOSSARY_CUSTOM_FIELDS")
    monkeypatch.delenv("GLOSSARY_CUSTOM_FIELDS", raising=False)

    owner = _RefineOwner(config)
    preview = gpc.prepare_manual_glossary_refinement(
        owner, source, str(progress), list(spec["types"]), "gpt-x",
        find_glossary=locator._find_glossary_for_refinement,
        active_types_fn=locator._active_glossary_refinement_types)
    assert os.environ.get("GLOSSARY_CUSTOM_FIELDS") == legacy_env
    if legacy is False:
        assert preview.refusal == legacy_owner.messages[-1]
        return
    assert preview.refusal is None and not legacy_owner.messages
    for name in ("glossary_path", "entries", "active_types", "selected_types", "request_mode", "system_prompt",
                 "user_prompt", "refinement_type_config", "non_empty_types"):
        assert getattr(preview, name) == legacy[name], name
    assert dict(preview.entry_counts) == dict(legacy["entry_counts"])
    assert _plan_view(preview.automatic_plan) == _plan_view(legacy["automatic_plan"])
    options, plan = gpc.finish_manual_glossary_refinement(preview, preview.selected_types, None)
    assert options.selected_types == preview.selected_types and options.force and options.run_when_disabled
    assert plan is preview.automatic_plan
    _options, forced = gpc.finish_manual_glossary_refinement(preview, preview.selected_types, 2)
    assert _options.target_chunk_count == 2
    assert _plan_view(forced) == _plan_view(preview.plan_for(preview.selected_types, 2))


def test_mobile_plan_manual_refinement(tmp_path):
    glossary, progress = _refine_workspace(tmp_path)
    owner = _RefineOwner({"output_directory": str(tmp_path / "out")})
    logs = []
    planned = gpc.plan_manual_glossary_refinement(
        owner, glossary_path=str(glossary), progress_path=str(progress), source_path=None,
        selected_types=["character"], log=logs.append)
    options, plan = planned
    assert options.selected_types == ["character"] and plan.total_chunks >= 1
    assert gpc.plan_manual_glossary_refinement(
        owner, glossary_path=str(glossary), selected_types=["locations"]) is None
    owner.model_var = ""
    with pytest.raises(RuntimeError, match="Select or enter a model"):
        gpc.plan_manual_glossary_refinement(owner, glossary_path=str(glossary), selected_types=["character"])
    owner.model_var = "gpt-x"
    with pytest.raises(RuntimeError, match="No saved glossary file"):
        gpc.plan_manual_glossary_refinement(owner, glossary_path=str(tmp_path / "missing.csv"),
                                            selected_types=["character"])


def _run_refinement(runner, owner, glossary, progress, monkeypatch, plan_fn):
    import extract_glossary_from_epub as extractor
    import glossary_refinement

    calls = {}

    def fake_refine(entries, **kwargs):
        calls["kwargs"] = {k: v for k, v in kwargs.items()
                           if k in ("temp", "mtoks", "available_tokens", "chunk_timeout", "progress_file",
                                    "output_path")}
        refined = [dict(entry) for entry in entries]
        refined[0]["translated_name"] = "Kim Cheol-su"
        return refined

    monkeypatch.setattr(glossary_refinement, "refine_glossary_entries", fake_refine)
    monkeypatch.setattr(extractor, "create_client_with_multi_key_support", lambda *a, **k: object())
    options, plan = plan_fn()
    runner(owner, str(glossary), str(progress), options, plan)
    env = {k: os.environ.get(k) for k in (
        "GLOSSARY_CUSTOM_ENTRY_TYPES", "GLOSSARY_CUSTOM_FIELDS", "GLOSSARY_REFINEMENT_SYSTEM_PROMPT",
        "GLOSSARY_REFINEMENT_USER_PROMPT", "GLOSSARY_REFINEMENT_CHUNKING_MODE", "GLOSSARY_REFINEMENT_SKIP_DEDUPE",
        "GLOSSARY_REFINEMENT_REOPEN_ON_SOURCE_CHANGE", "GLOSSARY_OUTPUT_LEGACY_JSON")}
    calls["kwargs"]["progress_file"] = os.path.basename(calls["kwargs"]["progress_file"])
    calls["kwargs"]["output_path"] = os.path.basename(calls["kwargs"]["output_path"])
    return {
        "env": env,
        "calls": calls["kwargs"],
        "csv": glossary.read_text(encoding="utf-8"),
        "files": sorted(p.name for p in glossary.parent.iterdir()),
        "logs": [str(m).replace(str(glossary.parent), "<G>") for m in owner.logs],
    }


@pytest.mark.parametrize("legacy_json", [False, True])
def test_manual_refinement_run_matches_the_frozen_runner(legacy_json, tmp_path, monkeypatch):
    config = {"glossary_refinement_user_prompt": "Be terse", "glossary_output_legacy_json": legacy_json,
              "glossary_refinement_skip_dedupe": True}
    results = []
    for side, runner in (("legacy", legacy_rg().RetranslationMixin._run_manual_glossary_refinement),
                         ("current", gpc.run_manual_glossary_refinement),
                         ("desktop", pl.current_rg().RetranslationMixin._run_manual_glossary_refinement)):
        work = tmp_path / side
        glossary, progress = _refine_workspace(work)
        owner = _RefineOwner(dict(config, output_directory=str(work / "out")))

        def planned(owner=owner, glossary=glossary):
            return gpc.plan_manual_glossary_refinement(owner, glossary_path=str(glossary),
                                                       selected_types=["character"])

        results.append(_run_refinement(runner, owner, glossary, progress, monkeypatch, planned))
    assert results[0] == results[1] == results[2]
    assert "Kim Cheol-su" in results[1]["csv"]
    assert ("Book_glossary.json" in results[1]["files"]) == legacy_json
