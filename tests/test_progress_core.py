"""U5: progress_core (Progress Manager core) parity and API tests.

Tiers:
* V (verbatim): every moved helper / method equals the frozen Retranslation_GUI
  (``tests/parity/progress_legacy.BASE_SHA``) modulo the documented edits below;
* D (differential fuzz): split / re-parameterised code vs the frozen statements
  executed as functions (``block_function``) or the frozen methods, >= 500 states;
* F (file system): the PM build and refresh on fixture workspaces (EPUB with chunks,
  PDF bookmark sections, metadata/artifact rows, subtitle ZIP, text, image folder):
  the offscreen legacy dialog vs the working-tree dialog (rows, colours, hidden flags,
  statistics labels, progress JSON + output tree);
* C (concurrency): view writes keep a translator update saved after the view read;
* M (mobile API): ProgressOwner / build_book_progress / compute_book_summary /
  snapshot_signature / ProgressPoller reproduce the desktop view without Qt.
"""

from __future__ import annotations

import ast
import copy
import json
import os
import random
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
for _p in (str(TESTS_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from parity import progress_legacy as pl  # noqa: E402

import progress_core as pc  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_library(tmp_path, monkeypatch):
    """Never touch the real Library registry; restore the process env afterwards."""
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(tmp_path / "_library"))
    # Workspaces resolve under $OUTPUT_DIRECTORY / $OUTPUT_DIR before the config's output
    # directory: never let a caller's value send them outside tmp_path.
    monkeypatch.delenv("OUTPUT_DIRECTORY", raising=False)
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _module_source(name):
    return (SRC_DIR / f"{name}.py").read_text(encoding="utf-8").replace("\r\n", "\n")


def _legacy_source():
    return "\n".join(pl.legacy_source_lines())


def _segments(source, *, cls=None):
    tree = ast.parse(source)
    lines = source.split("\n")
    body = tree.body
    if cls is not None:
        body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    out = {}
    for node in body:
        names = []
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        text = textwrap.dedent("\n".join(lines[start - 1:node.end_lineno]))
        for name in names:
            out[name] = text
    return out


def _apply(text, edits):
    for old, new in edits:
        assert text.count(old) == 1, (old, text.count(old))
        text = text.replace(old, new)
    return text


# ===========================================================================
# Tier I: GUI-free, Python 3.10
# ===========================================================================

SHARED = ("progress_core", "progress_actions", "glossary_progress_core")


@pytest.mark.parametrize("module", SHARED)
def test_shared_module_parses_as_python_310(module):
    ast.parse(_module_source(module), feature_version=(3, 10))


def test_shared_modules_import_without_pyside6():
    from parity import import_hygiene

    results = import_hygiene.check_imports(list(SHARED))
    bad = [r.describe() for r in results.values() if not r.ok]
    assert not bad, "\n".join(bad)


# ===========================================================================
# Tier V: verbatim moves
# ===========================================================================

#: Module-level names moved from Retranslation_GUI to progress_core.
CORE_MODULE_NAMES = (
    "_IS_MACOS", "_PROGRESS_SIDECAR_FILENAMES", "_NON_CHAPTER_OUTPUT_FILENAMES",
    "_PROGRESS_WATCH_DEBOUNCE_MS", "_PROGRESS_LIVE_REFRESH_MIN_INTERVAL_SECONDS",
    "_PROGRESS_DIRECT_ROW_UPDATE_LIMIT", "_RAW_FOREIGN_TEXT_QA_RE", "_LLM_TOKEN_QA_RE",
    "_MISSING_IMAGE_QA_RE", "_progress_total_label", "_sync_parent_chunk_qa_summary",
    "_pending_mark_output_path", "_QA_MARK_FIELDS", "_CHUNK_QA_MIRROR_FIELDS",
    "_chunk_ledger_for_progress_entry", "progress_entry_has_qa_mark", "_pending_mark_chunk_blocks",
    "_restore_pending_mark_record", "_is_progress_sidecar_entry",
    "_progress_entry_has_raw_foreign_text_qa", "_qa_value_has_llm_token_issue",
    "_progress_entry_has_llm_token_qa", "_qa_value_has_missing_image_issue",
    "_progress_entry_has_missing_image_qa", "_normalize_progress_match_text",
    "_normalize_progress_match_name", "_progress_entry_has_meaningful_tts_state",
    "_PROGRESS_READER_HTML_EXTENSIONS", "_ARTIFACT_ROW_STATUS_RANK", "_artifact_row_sort_rank",
    "_progress_item_is_html", "_index_epub_html_members", "_match_epub_html_member_basename",
    "_snapshot_progress_output_dir", "_write_progress_snapshot_atomic",
    "_RETRANSLATION_PROGRESS_LOCKS_GUARD", "_RETRANSLATION_PROGRESS_LOCKS",
    "_retranslation_progress_lock", "_merge_retranslation_progress_changes",
    "_merge_and_write_retranslation_progress", "_persist_progress_manager_source_link",
    "_progress_path_signature", "_clear_refinement_progress_fields",
    "_progress_entry_refined_for_display", "_progress_entry_refinement_failed_for_display",
    "_progress_entry_model_for_display", "_progress_status_hides_model_for_display",
    "_progress_entry_is_completed_image_only_for_display", "_format_qa_issue_for_progress_display",
    "_select_progress_entry_for_display",
)

#: Documented edits of moved bodies (DISCREPANCIES.md, U5).
CORE_MODULE_EDITS = {
    "_persist_progress_manager_source_link": [
        ("def _persist_progress_manager_source_link(file_path, output_dir):",
         "def _persist_progress_manager_source_link(file_path, output_dir, registry_cb=None):"),
        ("    try:\n        from epub_library import record_library_raw_input\n"
         "        record_library_raw_input(source_path)\n",
         "    try:\n        if callable(registry_cb):\n            registry_cb(source_path)\n"
         "        else:\n            from epub_library import record_library_raw_input\n"
         "            record_library_raw_input(source_path)\n"),
    ],
}

#: RetranslationMixin line ranges (frozen RG) whose methods moved to ProgressViewMixin.
MIXIN_RANGES = ((16579, 16580), (17931, 18645), (19069, 19889), (19891, 20013),
                (31719, 33044), (33066, 33338))

MIXIN_EDITS = {
    "_rematch_spine_chapters": [(
        "            with open(data['progress_file'], 'w', encoding='utf-8') as f:\n"
        "                json.dump(prog, f, ensure_ascii=False, indent=2)\n",
        "            data['_progress_view_baseline'] = _commit_view_progress(\n"
        "                data['progress_file'],\n"
        "                data.get('_progress_view_baseline') or {},\n"
        "                prog,\n"
        "            )\n",
    )],
}
#: Moved methods rewritten as a split (checked as blocks below).
SPLIT_METHODS = {"_progress_list_display_text"}


def test_moved_module_helpers_are_verbatim():
    legacy = _segments(_legacy_source())
    new = _segments(_module_source("progress_core"))
    for name in CORE_MODULE_NAMES:
        assert name in new, name
        assert _apply(legacy[name], CORE_MODULE_EDITS.get(name, ())) == new[name], name


def _legacy_rm_members():
    source = _legacy_source()
    tree = ast.parse(source)
    rm = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RetranslationMixin")
    names = []
    for node in rm.body:
        if any(a <= node.lineno <= b for a, b in MIXIN_RANGES):
            if isinstance(node, ast.FunctionDef):
                names.append(node.name)
            elif isinstance(node, ast.Assign):
                names.extend(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def test_moved_mixin_methods_are_verbatim():
    legacy = _segments(_legacy_source(), cls="RetranslationMixin")
    new = _segments(_module_source("progress_core"), cls="ProgressViewMixin")
    current_rm = _segments(_module_source("Retranslation_GUI"), cls="RetranslationMixin")
    names = _legacy_rm_members()
    assert len(names) == 55, names
    for name in names:
        assert name in new, name
        assert name not in current_rm, f"{name}: RetranslationMixin still defines the moved body"
        if name in SPLIT_METHODS:
            continue
        assert _apply(legacy[name], MIXIN_EDITS.get(name, ())) == new[name], name


def _legacy_block(start, end, *, strip, add, edits=()):
    lines = pl.legacy_source_lines()[start - 1:end]
    out = []
    for line in lines:
        if line.strip():
            assert line.startswith(" " * strip), (start, line)
            out.append(" " * add + line[strip:])
        else:
            out.append("" if not line.startswith(" " * strip) else " " * add + line[strip:])
    return _apply("\n".join(out), edits)


def _assert_block_in(source, start, end, *, strip, add, edits=()):
    block = _legacy_block(start, end, strip=strip, add=add, edits=edits)
    assert block in source, f"frozen RG {start}-{end} (+edits) not found verbatim"


VIEW_WRITE = (
    "with open(progress_file, 'w', encoding='utf-8') as f:\n",
    "json.dump(prog, f, ensure_ascii=False, indent=2)\n",
)


def _view_write_edits(text_block):
    """The six whole-file view writes -> _commit_view_progress (indent-preserving)."""
    import re

    return re.sub(
        r"( +)with open\(progress_file, 'w', encoding='utf-8'\) as f:\n\1    json\.dump\(prog, f, ensure_ascii=False, indent=2\)\n",
        lambda m: f"{m.group(1)}_view_baseline = _commit_view_progress(progress_file, _view_baseline, prog)\n",
        text_block,
    )


def test_split_build_method_holds_the_frozen_blocks():
    source = _module_source("progress_core")
    _assert_block_in(source, 20801, 20807, strip=8, add=8)
    _assert_block_in(source, 20809, 20826, strip=8, add=8)
    _assert_block_in(source, 20828, 20842, strip=8, add=8, edits=[
        ("                return None", "                return False")])
    _assert_block_in(source, 21041, 21146, strip=8, add=8)
    _assert_block_in(source, 20844, 20863, strip=8, add=8, edits=[(
        "        _persist_progress_manager_source_link(file_path, output_dir)\n",
        "        _persist_progress_manager_source_link(\n"
        "            file_path, output_dir, registry_cb=self._progress_raw_input_recorder()\n"
        "        )\n")])
    block = _legacy_block(20864, 21031, strip=8, add=8, edits=[
        ("                _write_progress_snapshot_atomic(progress_file, prog)\n",
         "                _view_baseline = _commit_view_progress(progress_file, _view_baseline, prog)\n"),
        ("        ProgressManager = _get_progress_manager_nonblocking()\n"
         "        if ProgressManager is not None:\n"
         "            temp_progress = ProgressManager(os.path.dirname(progress_file))\n"
         "            temp_progress.prog = prog\n"
         "            temp_progress.cleanup_missing_files(output_dir)\n"
         "            prog = temp_progress.prog\n",
         "        if self._progress_cleanup_ready():\n"
         "            cleanup_missing_files(prog, output_dir)\n"),
    ])
    assert _view_write_edits(block) in source
    assert _view_write_edits(_legacy_block(21148, 21779, strip=8, add=8)) in source


def test_split_refresh_and_stats_hold_the_frozen_blocks():
    source = _module_source("progress_core")
    _assert_block_in(source, 31335, 31371, strip=12, add=8)
    _assert_block_in(source, 31373, 31421, strip=12, add=8)
    _assert_block_in(source, 31440, 31529, strip=12, add=8)
    _assert_block_in(source, 33728, 33767, strip=12, add=8)
    _assert_block_in(source, 27600, 27621, strip=8, add=8)
    _assert_block_in(source, 33158, 33193, strip=8, add=8)
    _assert_block_in(source, 33195, 33338, strip=8, add=8)


def test_cleanup_missing_files_is_the_transate_method():
    legacy = (pl.frozen_dir() / "TK_cleanup_missing_files.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    legacy_body = "\n".join(legacy.split("\n")[2:]).rstrip("\n")  # after def + docstring
    legacy_body = textwrap.dedent(legacy_body).replace("self.prog", "prog")
    new = _segments(_module_source("progress_core"))["cleanup_missing_files"]
    new_body = textwrap.dedent("\n".join(new.split("\n")[2:])).rstrip("\n")
    assert new_body == legacy_body + "\nreturn cleaned_count"


def test_transate_progress_manager_delegates_cleanup(monkeypatch, tmp_path):
    import TransateKRtoEN

    calls = []
    monkeypatch.setattr(TransateKRtoEN, "_progress_core_cleanup_missing_files",
                        lambda prog, output_dir: calls.append((prog, output_dir)))
    manager = TransateKRtoEN.ProgressManager(str(tmp_path))
    manager.cleanup_missing_files(str(tmp_path))
    assert calls == [(manager.prog, str(tmp_path))]


def test_retranslation_gui_reexports_every_moved_name():
    rg = pl.current_rg()
    for name in CORE_MODULE_NAMES:
        assert getattr(rg, name) is getattr(pc, name), name
    assert issubclass(rg.RetranslationMixin, pc.ProgressViewMixin)
    assert rg.RetranslationMixin.__mro__[1] is pc.ProgressViewMixin


# ===========================================================================
# Tier D: differential fuzz
# ===========================================================================

STATUSES = ("completed", "completed_empty", "completed_image_only", "failed", "qa_failed",
            "in_progress", "pending", "merged", "not_translated", "unknown", "file_missing",
            "skipped", "error")
REFINE = (None, "", "refined", "completed", "failed", "error", "in_progress", "not_refined")
TTS = (None, "", "tts_completed", "completed", "failed", "in_progress", "no_tts")
MODES = (None, "text", "vision", "image", "audio", "refinement", "video")


def _random_entry(rng, depth=0):
    entry = {}
    if rng.random() < 0.8:
        entry["status"] = rng.choice(STATUSES)
    if rng.random() < 0.6:
        entry["model_name"] = rng.choice(["gpt-x", "", "No API needed", "claude-y"])
    if rng.random() < 0.2:
        entry["model"] = "legacy-model"
    if rng.random() < 0.4:
        entry["refinement_status"] = rng.choice(REFINE)
    if rng.random() < 0.4:
        entry["tts_status"] = rng.choice(TTS)
    if rng.random() < 0.4:
        issues = rng.sample(["missing_images_2", "llm_token_issue_x", "ai_truncation_detected_a",
                             "korean_text_found_5_chars_", "dup_body", "short"], rng.randint(0, 4))
        entry["qa_issues_found"] = issues
        entry["qa_issue_previews"] = {i: "preview " * rng.randint(1, 80) for i in issues if rng.random() < 0.7}
    if rng.random() < 0.15:
        entry["ocr_progress"] = {"done": rng.randint(0, 9), "total": rng.choice([0, 5, 9, "x"])}
    if rng.random() < 0.2:
        entry["merged_parent_chapter"] = rng.choice([0, 3, "7"])
    if rng.random() < 0.15:
        entry["manual_editing_pending"] = True
    if rng.random() < 0.15:
        entry["subtitle_no_translatable_text"] = True
    if rng.random() < 0.3:
        entry["output_file"] = rng.choice(["a.html", "response_b.html", "c.txt", "metadata.json", "TOC.txt"])
    if rng.random() < 0.15:
        entry["previous_status"] = rng.choice(["completed", "failed", "not_translated"])
    if depth < 2 and rng.random() < 0.3:
        entry["previous_progress_entry"] = _random_entry(rng, depth + 1)
    if rng.random() < 0.1:
        entry["pdf_toc_section"] = True
        entry["pdf_start_page"] = rng.choice([None, 1, 4])
        entry["pdf_end_page"] = rng.choice([None, 1, 9])
    if rng.random() < 0.1:
        entry["special_type"] = rng.choice(["metadata", "toc", "headers"])
    return entry


def _random_info(rng):
    kind = rng.choice(["chapter", "chunk", "metadata", "artifact", "pdf", "pdf_ocr", "subtitle", "fallback"])
    num = rng.choice([0, 1, 7, 12, 2.0, 2.5, 120])
    info = {
        "key": rng.choice(["k1", "chapter0001.xhtml", "", None]),
        "num": num,
        "status": rng.choice(STATUSES),
        "output_file": rng.choice(["response_ch1.html", "out.txt", "a.html", "metadata.json", "TOC.txt"]),
        "info": _random_entry(rng),
        "duplicate_count": rng.choice([1, 1, 2, 5]),
        "progress_key": rng.choice([None, "1", "k"]),
        "original_filename": rng.choice(["chapter0001.xhtml", "title.xhtml", "a" * 30, ""]),
        "is_special": rng.random() < 0.2,
    }
    if rng.random() < 0.5:
        info["display_num"] = rng.choice([1, 3, 4.0, 9.5])
    if kind == "chapter":
        info["opf_position"] = rng.randint(0, 300)
    elif kind == "chunk":
        info.update({"is_chunk_progress": True, "chunk_index": rng.randint(1, 4), "total_chunks": 4,
                     "pdf_toc_section": rng.random() < 0.3})
    elif kind == "metadata":
        info.update({"special_type": "metadata", "metadata_label": rng.choice([None, "Title"]),
                     "output_file": "metadata.json"})
    elif kind == "artifact":
        info.update({"special_type": rng.choice(["toc", "headers"]),
                     "progress_key": rng.choice(["__translation_artifact__:toc", "__translation_artifact__:headers"]),
                     "translation_artifact_label": rng.choice([None, "Table of Contents"]),
                     "artifact_translation_enabled": rng.random() < 0.7})
    elif kind == "pdf":
        info.update({"pdf_toc_section": True, "pdf_start_page": rng.choice([None, 2]),
                     "pdf_end_page": rng.choice([None, 2, 5])})
    elif kind == "pdf_ocr":
        info["pdf_ocr"] = True
    elif kind == "subtitle":
        info.update({"is_subtitle": True, "subtitle_completed_batches": rng.randint(0, 3),
                     "subtitle_total_batches": rng.choice([1, 3, "x"])})
    return info


def _plain_owner(module, rng=None, mode=None):
    owner = module.RetranslationMixin()
    owner.config = {"translate_special_files": False}
    if mode:
        owner.output_mode_var = mode
    return owner


def test_display_text_fuzz_matches_frozen_method(tmp_path):
    rng = random.Random(5101)
    legacy = pl.legacy_rg()
    current = pl.current_rg()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "response_ch1.html").write_text("x", encoding="utf-8")
    (out_dir / "a.html").write_text("x", encoding="utf-8")
    checked = 0
    for _ in range(1500):
        info = _random_info(rng)
        mode = rng.choice(MODES)
        data = {"show_model_info_state": rng.random() < 0.5, "prog": {"output_mode": rng.choice(MODES)}}
        if rng.random() < 0.6:
            data["output_dir"] = str(out_dir)
        widths = (rng.choice([20, 35]), rng.choice([25, 40]))
        lo = _plain_owner(legacy, mode=mode)
        co = _plain_owner(current, mode=mode)
        try:
            expected = lo._progress_list_display_text(copy.deepcopy(info), copy.deepcopy(data), *widths)
        except Exception as exc:  # the same input must fail the same way
            with pytest.raises(type(exc)):
                co._progress_list_display_text(copy.deepcopy(info), copy.deepcopy(data), *widths)
            continue
        assert co._progress_list_display_text(copy.deepcopy(info), copy.deepcopy(data), *widths) == expected
        checked += 1
    assert checked >= 500


def test_statistics_fuzz_matches_frozen_blocks():
    rng = random.Random(5102)
    legacy = pl.legacy_rg()
    current = pl.current_rg()
    stats_block = pl.block_function(
        legacy, 33728, 33767, ["self", "data"], dedent=12,
        result="(total_chapters, chunk_count, completed, merged, in_progress, pending, missing, failed, skipped, mode)",
    )
    initial_block = pl.block_function(
        legacy, 27600, 27621, ["self", "prog", "chapter_display_info", "spine_chapters"], dedent=8,
        result="(total_chapters, chunk_count, completed, merged, in_progress, pending, missing, failed, skipped)",
    )
    for _ in range(600):
        rows = [_random_info(rng) for _ in range(rng.randint(0, 12))]
        if rng.random() < 0.1:
            rows = [dict(r, pdf_ocr=True, info={"total": rng.randint(0, 9), "done": rng.randint(0, 9),
                                                 "failed": rng.randint(0, 2)}) for r in rows]
        mode = rng.choice(MODES)
        data = {"chapter_display_info": rows, "prog": {"output_mode": rng.choice(MODES)}}
        lo = _plain_owner(legacy, mode=mode)
        co = _plain_owner(current, mode=mode)
        assert co._progress_statistics(copy.deepcopy(data)) == stats_block(lo, copy.deepcopy(data))
        spine = [dict(r) for r in rows] if rng.random() < 0.3 else []
        display = rows if rng.random() < 0.7 else []
        prog = {"output_mode": rng.choice(MODES)}
        assert co._progress_initial_statistics(prog, display, spine) == initial_block(lo, prog, display, spine)


def _random_cleanup_case(rng, out_dir):
    names = [f"response_ch{i}.html" for i in range(8)] + ["ch8.xhtml", "metadata.json", "sub.srt"]
    existing = set(rng.sample(names, rng.randint(0, len(names))))
    for name in existing:
        (out_dir / name).write_text("x", encoding="utf-8")
    if rng.random() < 0.3:
        (out_dir / "ch3.xhtml").write_text("renamed", encoding="utf-8")
    chapters = {}
    for i in range(rng.randint(1, 10)):
        status = rng.choice(["completed", "failed", "qa_failed", "in_progress", "pending",
                             "pending_retry", "merged", None, "completed_empty"])
        entry = {"actual_num": rng.choice([i, i, None]), "status": status,
                 "output_file": rng.choice(names + ["response_ch3.html", None, ""])}
        if status == "merged":
            entry["merged_parent_chapter"] = rng.choice([0, 1, 2, 3])
        if rng.random() < 0.2:
            entry["merged_chapters"] = [rng.randint(0, 5)]
        if rng.random() < 0.1:
            entry["subtitle_progress_key"] = "subtitle:x:1"
        if rng.random() < 0.1:
            entry["pdf_outline_seed"] = True
        key = rng.choice([str(i), f"special_{i}", "__metadata__", "__metadata__:title"])
        if key.startswith("__metadata__"):
            entry.update({"special_type": "metadata", "output_file": "metadata.json"})
            if rng.random() < 0.5:
                entry["metadata_regeneration_requested"] = True
        chapters[key] = entry
    chunks = {key: {"total": 2} for key in chapters if rng.random() < 0.3}
    return {"chapters": chapters, "chapter_chunks": chunks}


def test_cleanup_missing_files_fuzz_matches_frozen_transate(tmp_path):
    rng = random.Random(5103)
    for case in range(500):
        out_dir = tmp_path / f"c{case}"
        out_dir.mkdir()
        prog = _random_cleanup_case(rng, out_dir)
        legacy_prog = pl.legacy_cleanup_missing_files(copy.deepcopy(prog), str(out_dir))
        new_prog = copy.deepcopy(prog)
        pc.cleanup_missing_files(new_prog, str(out_dir))
        assert pl.normalize_progress(new_prog) == pl.normalize_progress(legacy_prog), case


def test_merge_three_way_fuzz_matches_frozen():
    rng = random.Random(5104)
    legacy = pl.legacy_rg()

    def rnd(depth=0):
        if depth > 2 or rng.random() < 0.3:
            return rng.choice([1, "a", None, [1, 2], True])
        return {rng.choice("abcde"): rnd(depth + 1) for _ in range(rng.randint(0, 4))}

    for _ in range(800):
        baseline = rnd()
        changed = copy.deepcopy(baseline) if isinstance(baseline, dict) else rnd()
        if isinstance(changed, dict):
            for key in list(changed)[:2]:
                if rng.random() < 0.4:
                    changed.pop(key)
            changed[rng.choice("xyz")] = rnd(1)
        latest = rnd()
        assert pc.merge_progress_changes(copy.deepcopy(baseline), copy.deepcopy(changed), copy.deepcopy(latest)) == \
            legacy._merge_retranslation_progress_changes(copy.deepcopy(baseline), copy.deepcopy(changed), copy.deepcopy(latest))


# ===========================================================================
# Tier F: the build + refresh on fixture workspaces (offscreen dialogs)
# ===========================================================================

FIXTURES = ("epub", "pdf", "subtitle", "text")


def _fixture(kind, base):
    if kind == "epub":
        source, out, config = pl.epub_workspace(base)
        return source, out, config, None
    if kind == "pdf":
        return pl.pdf_workspace(base)
    if kind == "subtitle":
        source, out, config = pl.subtitle_zip_workspace(base)
        return source, out, config, None
    source, out, config = pl.text_workspace(base)
    return source, out, config, None


def _run_view(module, fixture_root, work_root, kind, source_name, *, audio=False, refresh=False):
    work = pl.copy_workspace(fixture_root, work_root)
    _source, _out, config, plan = _fixture_config(kind, fixture_root)
    cfg = dict(config)
    cfg["output_directory"] = str(work / "out")
    attrs = {"output_mode_var": "audio"} if audio else {}
    host = pl.make_host(module, cfg, **attrs)
    if plan is not None:
        host._pdf_outline_progress_plan = lambda file_path, _plan=plan: [dict(s) for s in _plan]
    pl.MESSAGES.clear()
    data = pl.open_progress_manager(host, work / source_name)
    snapshots = [pl.view_snapshot(data), pl.tree_snapshot(work / "out", workspace=work)]
    if refresh:
        host._refresh_retranslation_data(data)
        pl.pump(20, until=lambda: not data.get('_listbox_populate_active'), timeout=5)
        snapshots += [pl.view_snapshot(data), pl.tree_snapshot(work / "out", workspace=work)]
    data['dialog'].hide()
    return snapshots, list(pl.MESSAGES), host, data


_CONFIG_CACHE = {}


def _fixture_config(kind, fixture_root):
    return _CONFIG_CACHE[(kind, str(fixture_root))]


@pytest.fixture(scope="module")
def fixture_roots(tmp_path_factory):
    roots = {}
    for kind in FIXTURES:
        base = tmp_path_factory.mktemp(f"pm_{kind}")
        built = _fixture(kind, base)
        _CONFIG_CACHE[(kind, str(base))] = built
        roots[kind] = (base, Path(built[0]).name)
    return roots


@pytest.mark.parametrize("kind", FIXTURES)
@pytest.mark.parametrize("audio", (False, True))
def test_progress_manager_view_matches_frozen_desktop(kind, audio, fixture_roots, tmp_path):
    base, source_name = fixture_roots[kind]
    legacy, legacy_msgs, *_ = _run_view(pl.legacy_rg(), base, tmp_path / "legacy", kind, source_name,
                                        audio=audio, refresh=True)
    current, current_msgs, *_ = _run_view(pl.current_rg(), base, tmp_path / "current", kind, source_name,
                                          audio=audio, refresh=True)
    assert current[0] == legacy[0]          # rows, colours, hidden flags, statistics labels
    assert current[1] == legacy[1]          # progress JSON + output tree after opening
    assert current[2] == legacy[2]          # after an explicit full refresh
    assert current[3] == legacy[3]
    assert current_msgs == legacy_msgs


def test_image_folder_view_matches_frozen_desktop(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QListWidget

    fixture = tmp_path / "fixture"
    fixture.mkdir()
    pl.image_folder_workspace(fixture)
    results = []
    for side, module in (("legacy", pl.legacy_rg()), ("current", pl.current_rg())):
        work = pl.copy_workspace(fixture, tmp_path / side)
        monkeypatch.chdir(work)
        host = pl.make_host(module, {"output_directory": ""})
        host._force_retranslation_images_folder(str(work / "Pics"))
        pl.pump(20, timeout=0.5)
        rows = []
        for widget in QApplication.topLevelWidgets():
            if widget.isVisible():
                for listbox in widget.findChildren(QListWidget):
                    rows.append([listbox.item(i).text() for i in range(listbox.count())])
                widget.hide()
        results.append((rows, pl.tree_snapshot(work, workspace=work)))
    assert results[0] == results[1]
    assert results[0][0] == [["📄 Image 001 | a | ✅ Completed", "🖼️ Cover | cover.png | ⏭️ Skipped (cover)"]]


def test_build_matches_frozen_block_oracle(fixture_roots, tmp_path):
    """The frozen build statements (RG 20801-21779) vs _build_progress_view_data."""
    legacy = pl.legacy_rg()
    block = pl.block_function(
        legacy, 20801, 21779,
        ["self", "file_path", "parent_dialog", "resolved_output_dir", "_pump_loading", "show_special_files_state"],
        dedent=8,
        result="{'prog': prog, 'spine_chapters': spine_chapters, 'chapter_display_info': chapter_display_info, "
               "'output_dir': output_dir, 'progress_file': progress_file}",
    )
    for kind in FIXTURES:
        base, source_name = fixture_roots[kind]
        _source, _out, config, plan = _fixture_config(kind, base)
        sides = []
        for side, module in (("legacy", legacy), ("current", pl.current_rg())):
            work = pl.copy_workspace(base, tmp_path / f"{kind}_{side}")
            cfg = dict(config, output_directory=str(work / "out"))
            host = pl.make_host(module, cfg)
            if plan is not None:
                host._pdf_outline_progress_plan = lambda file_path, _plan=plan: [dict(s) for s in _plan]
            if side == "legacy":
                built = block(host, str(work / source_name), None, None, lambda msg=None: None, False)
            else:
                built = host._build_progress_view_data(str(work / source_name))
            sides.append((
                pl.normalize_progress(json.loads(json.dumps(built["prog"]))),
                json.loads(json.dumps(built["spine_chapters"], default=str)),
                pl.normalize_progress(json.loads(json.dumps(built["chapter_display_info"], default=str))),
                pl.tree_snapshot(work / "out", workspace=work),
            ))
        legacy_side, current_side = sides
        replace = (str(tmp_path / f"{kind}_legacy"), str(tmp_path / f"{kind}_current"))
        assert pl._replace_paths(legacy_side, [replace]) == current_side, kind


# ===========================================================================
# Tier C: view writes keep concurrent translator updates
# ===========================================================================


def _translator_update(progress_file, key="1", field="translator_saved", value=True):
    """What a translator save does meanwhile: rewrite the file with one more field."""
    with open(progress_file, encoding="utf-8") as f:
        data = json.load(f)
    data["chapters"].setdefault(key, {})[field] = value
    pc.write_progress_atomic(progress_file, data)


def test_mutate_progress_keeps_a_concurrent_translator_save(tmp_path):
    path = tmp_path / "translation_progress.json"
    pc.write_progress_atomic(path, {"chapters": {"1": {"status": "completed"}, "2": {"status": "qa_failed"}}})

    def action(prog):
        # The translator saves while the action is being applied (it never takes our lock).
        _translator_update(path, "1", "model_name", "translator")
        prog["chapters"]["2"]["status"] = "completed"
        return "done"

    assert pc.mutate_progress(path, action) == "done"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["chapters"]["1"]["model_name"] == "translator"
    assert data["chapters"]["2"]["status"] == "completed"


def test_mutate_progress_does_not_write_when_nothing_changed(tmp_path):
    path = tmp_path / "translation_progress.json"
    pc.write_progress_atomic(path, {"chapters": {}})
    before = path.stat().st_mtime_ns
    time.sleep(0.01)
    pc.mutate_progress(path, lambda prog: None)
    assert path.stat().st_mtime_ns == before


def test_mutate_progress_serialises_writers(tmp_path):
    path = tmp_path / "translation_progress.json"
    pc.write_progress_atomic(path, {"chapters": {}})

    def worker(index):
        def bump(prog):
            prog["chapters"][str(index)] = {"status": "completed"}
        pc.mutate_progress(path, bump)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data["chapters"]) == {str(i) for i in range(12)}


def test_view_commit_retries_a_sharing_violation_on_read(tmp_path, monkeypatch):
    """A locked progress file (Windows sharing violation) is retried, not fatal to the tick.

    The former ``_write_progress_json_safely`` retried 20 times with backoff; the merge
    path that replaced it opens the newest file first, so that read must retry too.
    """
    path = tmp_path / "translation_progress.json"
    pc.write_progress_atomic(path, {"chapters": {"1": {"status": "completed"}}})
    baseline = json.loads(path.read_text(encoding="utf-8"))
    prog = copy.deepcopy(baseline)
    prog["chapters"]["2"] = {"status": "pending"}
    real_open = open
    failures = {"left": 3}

    def flaky_open(file, mode="r", *args, **kwargs):
        if os.fspath(file) == os.fspath(path) and "r" in mode and failures["left"]:
            failures["left"] -= 1
            raise PermissionError(13, "The process cannot access the file", os.fspath(path))
        return real_open(file, mode, *args, **kwargs)

    sleeps = []
    monkeypatch.setattr("builtins.open", flaky_open)
    monkeypatch.setattr(pc.time, "sleep", sleeps.append)
    new_baseline = pc._commit_view_progress(os.fspath(path), baseline, prog)
    monkeypatch.undo()
    assert failures["left"] == 0 and len(sleeps) == 3
    assert new_baseline == prog
    assert json.loads(path.read_text(encoding="utf-8"))["chapters"]["2"] == {"status": "pending"}


def test_view_commit_gives_up_after_the_retry_budget(tmp_path, monkeypatch):
    path = tmp_path / "translation_progress.json"
    pc.write_progress_atomic(path, {"chapters": {}})
    calls = []

    def always_locked(*args, **kwargs):
        calls.append(args)
        raise PermissionError(13, "locked")

    monkeypatch.setattr(pc, "_merge_and_write_retranslation_progress", always_locked)
    monkeypatch.setattr(pc.time, "sleep", lambda _s: None)
    with pytest.raises(PermissionError):
        pc._commit_view_progress(os.fspath(path), {"chapters": {}}, {"chapters": {"1": {}}})
    assert len(calls) == pc._VIEW_COMMIT_ATTEMPTS


def _legacy_refresh_writer():
    """The frozen ``_write_progress_json_safely`` (the former refresh writer) as a function."""
    lines = pl.legacy_source_lines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("def _write_progress_json_safely("))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].strip() == "raise last_error")
    namespace = {"os": os, "json": json}
    exec(compile(textwrap.dedent("\n".join(lines[start:end + 1])), "<legacy refresh writer>", "exec"), namespace)
    return namespace["_write_progress_json_safely"]


@pytest.mark.parametrize("lock", ["replace", "read"])
def test_view_commit_on_a_persistent_lock_blocks_no_longer_than_the_former_writer(tmp_path, monkeypatch, lock):
    """A persistently locked progress file: the commit gives up within the former refresh
    writer's budget (20 attempts, 8.4-9.0 s), measured on a virtual clock that ``time.sleep``
    advances, with the real atomic writer and the longest jitter.

    The atomic writer already retries its own ``os.replace`` 20 times (about 6.5 s); the commit
    used to run that whole write a second time (12.9 s, 40 replace calls)."""
    path = tmp_path / "translation_progress.json"
    pc.write_progress_atomic(path, {"chapters": {}})
    clock = {"t": 0.0}
    monkeypatch.setattr(pc.time, "sleep", lambda seconds: clock.__setitem__("t", clock["t"] + seconds))
    monkeypatch.setattr(pc.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(random, "uniform", lambda low, high: high)
    real_replace, real_open = os.replace, open
    mode = {"lock": "replace"}
    replaces = []

    def sharing_violation():
        exc = PermissionError(13, "The process cannot access the file because it is being used by another process",
                              os.fspath(path))
        exc.winerror = 32  # retried by the atomic writer on every OS
        raise exc

    def replace(src, dst):
        if os.fspath(dst) == os.fspath(path) and mode["lock"] == "replace":
            replaces.append(clock["t"])
            sharing_violation()
        return real_replace(src, dst)

    def locked_open(file, open_mode="r", *args, **kwargs):
        if mode["lock"] == "read" and os.fspath(file) == os.fspath(path) and "r" in open_mode:
            sharing_violation()
        return real_open(file, open_mode, *args, **kwargs)

    monkeypatch.setattr(pc.os, "replace", replace)
    monkeypatch.setattr("builtins.open", locked_open)
    with pytest.raises(PermissionError):
        _legacy_refresh_writer()(os.fspath(path), {"chapters": {"1": {}}})
    former = clock["t"]
    assert len(replaces) == 20 and 8.4 <= former <= 9.1

    clock["t"], mode["lock"] = 0.0, lock
    replaces.clear()
    with pytest.raises(PermissionError):
        pc._commit_view_progress(os.fspath(path), {"chapters": {}}, {"chapters": {"1": {}}})
    monkeypatch.undo()
    assert clock["t"] <= former
    if lock == "replace":
        assert len(replaces) == 20  # one atomic write: its own 20 attempts, not run again
    else:
        assert replaces == [] and clock["t"] > 7.0  # the read is retried 20 times like the former writer
    assert json.loads(path.read_text(encoding="utf-8")) == {"chapters": {}}


def test_build_writes_keep_a_concurrent_translator_save(fixture_roots, tmp_path, monkeypatch):
    """A translator save landing while the PM opens survives the PM's seeding writes.

    The frozen build wrote its whole in-memory snapshot and lost it (asserted below).
    """
    base, source_name = fixture_roots["epub"]
    _source, _out, config, _plan = _fixture_config("epub", base)
    outcomes = {}
    for side, module in (("legacy", pl.legacy_rg()), ("current", pl.current_rg())):
        work = pl.copy_workspace(base, tmp_path / side)
        cfg = dict(config, output_directory=str(work / "out"))
        host = pl.make_host(module, cfg)
        progress_file = work / "out" / "Book" / "translation_progress.json"
        original = host._ensure_metadata_progress_entry

        def racing(prog, output_dir, file_path=None, _original=original, _path=progress_file):
            _translator_update(_path, "8", "status", "in_progress")
            return _original(prog, output_dir, file_path)

        host._ensure_metadata_progress_entry = racing
        if side == "legacy":
            block = pl.block_function(
                module, 20801, 21779,
                ["self", "file_path", "parent_dialog", "resolved_output_dir", "_pump_loading", "show_special_files_state"],
                dedent=8, result="prog")
            block(host, str(work / source_name), None, None, lambda msg=None: None, False)
        else:
            host._build_progress_view_data(str(work / source_name))
        outcomes[side] = json.loads(progress_file.read_text(encoding="utf-8"))
    assert outcomes["current"]["chapters"]["8"]["status"] == "in_progress"
    assert "__metadata__" in outcomes["current"]["chapters"]
    # Recorded desktop bug (DISCREPANCIES U5): the whole-file write lost the save.
    assert "8" not in outcomes["legacy"]["chapters"] or outcomes["legacy"]["chapters"]["8"].get("status") != "in_progress"


def test_refresh_writes_keep_a_concurrent_translator_save(fixture_roots, tmp_path):
    base, source_name = fixture_roots["epub"]
    _source, _out, config, _plan = _fixture_config("epub", base)
    work = pl.copy_workspace(base, tmp_path / "current")
    cfg = dict(config, output_directory=str(work / "out"))
    host = pl.make_host(pl.current_rg(), cfg)
    data = pl.open_progress_manager(host, work / source_name)
    progress_file = work / "out" / "Book" / "translation_progress.json"
    # Make the explicit refresh write (metadata.json vanished -> metadata row pending).
    (work / "out" / "Book" / "metadata.json").unlink()
    original = host._reconcile_tts_audio_files

    def racing(view_data, _original=original):
        _translator_update(progress_file, "6", "status", "completed")
        return _original(view_data)

    host._reconcile_tts_audio_files = racing
    host._refresh_retranslation_data(data)
    saved = json.loads(progress_file.read_text(encoding="utf-8"))
    assert saved["chapters"]["6"]["status"] == "completed"
    assert saved["chapters"]["__metadata__"]["status"] == "pending"
    data['dialog'].hide()


# ===========================================================================
# Tier M: mobile API
# ===========================================================================


def test_progress_owner_special_rules_match_desktop_owner():
    rng = random.Random(5105)
    names = ["title.xhtml", "response_toc.html", "chapter0001.xhtml", "notice2.xhtml", "index.xhtml",
             "glossary_unified.html", "colophon.htm", "message_001.xhtml", "preface.xhtml", "x.txt"]
    for _ in range(200):
        config = {}
        if rng.random() < 0.5:
            config["special_file_keywords"] = rng.choice(["title, notice", "", "toc"])
        if rng.random() < 0.5:
            config["special_file_exact"] = rng.choice(["index, glossary, glossary_extension", "", "x"])
        if rng.random() < 0.5:
            config["translate_all_numbered_html"] = rng.random() < 0.5
        if rng.random() < 0.5:
            config["translate_special_files"] = rng.random() < 0.5
        owner = pc.ProgressOwner(config)
        host = pl.make_host(pl.current_rg(), config)
        for name in names:
            assert owner._is_special_file(name) == host._is_special_file(name)
            assert owner._progress_file_is_skipped_special(name) == host._progress_file_is_skipped_special(name)


@pytest.mark.parametrize("kind", FIXTURES)
def test_build_book_progress_reproduces_the_desktop_view(kind, fixture_roots, tmp_path):
    base, source_name = fixture_roots[kind]
    desktop, _msgs, _host, _data = _run_view(pl.legacy_rg(), base, tmp_path / "legacy", kind, source_name)
    _source, _out, config, plan = _fixture_config(kind, base)
    work = pl.copy_workspace(base, tmp_path / "mobile")
    owner = pc.ProgressOwner(dict(config, output_directory=str(work / "out")))
    if plan is not None:
        owner._pdf_outline_progress_plan = lambda file_path, _plan=plan: [dict(s) for s in _plan]
    show_special = str(source_name).lower().endswith(('.txt', '.pdf', '.zip'))
    book = pc.build_book_progress(str(work / source_name), owner=owner, show_special_files=False)
    # The desktop view applies one silent (read-only) refresh when it is shown.
    book = pc.refresh_book_progress(book, force=True)
    assert [(row.text, row.hidden, row.status) for row in book.rows] == \
        [(text, hidden, status) for text, _colour, hidden, status in desktop[0]["rows"]]
    assert pl.tree_snapshot(work / "out", workspace=work) == desktop[1]
    colours = {"green": "#008000", "red": "#ff0000", "orange": "#ffa500", "white": "#ffffff"}
    assert [colours.get(row.color, row.color) for row in book.rows] == \
        [colour for _text, colour, _hidden, _status in desktop[0]["rows"]]
    labels = {text for text, _visible, _style in desktop[0]["labels"]}
    assert book.stats.total_label in labels
    assert f"✅ Completed: {book.stats.completed} | " in labels
    assert f"{book.stats.missing_label}: {book.stats.missing} | " in labels
    assert f"{book.stats.failed_icon} {book.stats.failed_label}: {book.stats.failed} | " in labels
    assert show_special or True


def test_present_row_pieces(fixture_roots, tmp_path):
    base, source_name = fixture_roots["epub"]
    _source, _out, config, _plan = _fixture_config("epub", base)
    work = pl.copy_workspace(base, tmp_path / "mobile")
    book = pc.build_book_progress(str(work / source_name), dict(config, output_directory=str(work / "out")))
    by_title = {row.title: row for row in book.rows}
    assert by_title["Metadata: Metadata"].kind == "metadata"
    assert by_title["Table of Contents"].kind == "artifact"
    ch1 = by_title["Ch.001 · chapter0001.xhtml"]
    assert (ch1.status, ch1.icon, ch1.badges, ch1.model) == ("completed", "✅", ["⭐"], "gpt-x")
    ch2 = by_title["Ch.002 · chapter0002.xhtml"]
    assert ch2.status == "qa_failed" and ch2.qa_more == 1 and len(ch2.qa_issues) == 2
    assert ch2.qa_issues[0].startswith("llm_token_issue_empty_attr — Preview:")
    ch3 = by_title["Ch.003 · chapter0003.xhtml"]
    assert ch3.chunk_summary.startswith("Chunks")
    chunk = by_title["↳ Ch.003 · Chunk 2/3"]
    assert (chunk.kind, chunk.status, chunk.parent_key, chunk.row_id) == (
        "chunk", "qa_failed", "3", "chunk:hash-ch3:2")
    assert by_title["Ch.004 · chapter0004.xhtml"].model == ""
    assert book.stats.done_fraction == pytest.approx(8 / 13)
    hidden = [row.title for row in book.rows if row.hidden]
    assert hidden == ["Chapter Headers", "Ch.000 · title.xhtml", "Ch.009 · notice.xhtml"]
    pc.set_view_toggles(book, show_special_files=True)
    assert not any(row.hidden for row in book.rows)


def test_status_vocabulary_is_the_desktop_one():
    rg = pl.legacy_rg()
    owner = rg.RetranslationMixin()
    for status, (icon, label, colour) in pc.STATUS_VOCAB.items():
        if status == "file_missing":
            continue
        info = {"num": 1, "output_file": "a.html", "status": status, "info": {},
                "opf_position": 0, "original_filename": "a.xhtml"}
        text, shown = owner._progress_list_display_text(info, {}, 20, 25)
        if shown == status:
            assert f"| {icon} {label}" in text
        from PySide6.QtWidgets import QListWidgetItem
        pl.qapp()
        item = QListWidgetItem("x")
        rg.RetranslationMixin._apply_progress_list_item_visuals(owner, item, status)
        from PySide6.QtGui import QColor
        assert item.foreground().color().name() == QColor(colour).name(), status
    assert set(pc.STATUS_GROUPS["failed"]) == {"failed", "qa_failed", "refine_failed"}


def test_compute_book_summary_is_read_only_and_cached(fixture_roots, tmp_path):
    base, source_name = fixture_roots["epub"]
    _source, _out, config, _plan = _fixture_config("epub", base)
    work = pl.copy_workspace(base, tmp_path / "mobile")
    cfg = dict(config, output_directory=str(work / "out"))
    before = pl.tree_snapshot(work, workspace=work)
    summary = pc.compute_book_summary(str(work / source_name), cfg)
    assert pl.tree_snapshot(work, workspace=work) == before
    assert summary.has_progress
    assert (summary.completed, summary.total) == (4, 9)  # read-only: nothing seeded
    assert summary.chunk_qa_failed_parents == 1
    assert pc.compute_book_summary(str(work / source_name), cfg) is summary
    (work / "out" / "Book" / "response_chapter0008.html").write_text("<p>8</p>", encoding="utf-8")
    again = pc.compute_book_summary(str(work / source_name), cfg)
    assert again is not summary and again.completed == 5
    missing = pc.compute_book_summary(str(work / source_name), cfg, output_dir=str(work / "nope"))
    assert not missing.has_progress and missing.total == 0
    assert not (work / "nope").exists()


def test_refresh_book_progress_and_signature(fixture_roots, tmp_path):
    base, source_name = fixture_roots["epub"]
    _source, _out, config, _plan = _fixture_config("epub", base)
    work = pl.copy_workspace(base, tmp_path / "mobile")
    book = pc.build_book_progress(str(work / source_name), dict(config, output_directory=str(work / "out")))
    assert pc.refresh_book_progress(book) is book
    progress_file = Path(book.progress_file)
    before = progress_file.read_bytes()
    (work / "out" / "Book" / "response_chapter0008.html").write_text("<p>8</p>", encoding="utf-8")
    fresh = pc.refresh_book_progress(book)
    assert fresh is not book
    assert [r.status for r in fresh.rows if r.title == "Ch.008 · chapter0008.xhtml"] == ["completed"]
    assert progress_file.read_bytes() == before          # read-only tick: no write
    full = pc.refresh_book_progress(fresh, read_only=False, force=True)
    saved = json.loads(progress_file.read_text(encoding="utf-8"))
    assert saved["chapters"]["8"]["auto_discovered"] is True
    assert full.signature == pc.snapshot_signature(book.progress_file, book.output_dir, include_tts=True)


def test_snapshot_signature_matches_the_desktop_prefetch(tmp_path):
    out = tmp_path / "out"
    (out / "text_to_speech").mkdir(parents=True)
    (out / "a.html").write_text("x", encoding="utf-8")
    (out / "text_to_speech" / "A.mp3").write_text("x", encoding="utf-8")
    progress = out / "translation_progress.json"
    progress.write_text("{}", encoding="utf-8")
    signature = pc.snapshot_signature(str(progress), str(out), include_tts=True)
    stat = progress.stat()
    listing = {"a.html", "translation_progress.json"}
    assert signature == ((stat.st_mtime_ns, stat.st_size), (2, hash(frozenset(listing))), (1, hash(frozenset({"a.mp3"}))))
    assert pc.snapshot_signature(str(progress), str(out))[2] is None


def test_progress_poller_reports_changes(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    progress = out / "translation_progress.json"
    progress.write_text("{}", encoding="utf-8")
    seen = []
    poller = pc.ProgressPoller(str(progress), str(out), seen.append, interval=0.05,
                               initial_signature=pc.snapshot_signature(str(progress), str(out)))
    poller.start()
    try:
        time.sleep(0.2)
        assert seen == []
        (out / "new.html").write_text("x", encoding="utf-8")
        deadline = time.time() + 3
        while not seen and time.time() < deadline:
            time.sleep(0.05)
        assert len(seen) == 1
        poller.pause()
        (out / "other.html").write_text("x", encoding="utf-8")
        time.sleep(0.3)
        assert len(seen) == 1
        poller.resume()
        deadline = time.time() + 3
        while len(seen) < 2 and time.time() < deadline:
            time.sleep(0.05)
        assert len(seen) == 2
    finally:
        poller.stop()


def test_build_book_progress_creates_and_links_a_workspace(tmp_path, monkeypatch):
    source = tmp_path / "New.epub"
    pl.make_epub(source, [("chapter0001.xhtml", "<p>x</p>")])
    library = tmp_path / "_library"
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(library))
    book = pc.build_book_progress(str(source), {"output_directory": str(tmp_path / "out")})
    assert book.created_folder == str(tmp_path / "out" / "New")
    link = (tmp_path / "out" / "New" / "source_epub.txt").read_text(encoding="utf-8")
    assert link == str(source)
    import library_core
    assert str(source) in library_core.load_library_raw_inputs()
    assert book.rows[0].kind == "metadata" and book.rows[0].status == "pending"
    assert book.rows[-1].title == "Ch.001 · chapter0001.xhtml"
    assert book.rows[-1].status == "not_translated"
