"""U7: sdlxliff_review_core (the GUI-free SDLXLIFF reviewer) parity and API tests.

The SDLXLIFF reviewer's widget-free methods (SDLXLIFFReviewDialog) and RetranslationMixin's
sidecar auto-generation moved verbatim into ``sdlxliff_review_core``
(``SdlxliffReviewCoreMixin`` / ``SdlxliffAutogenMixin``); the dialog inherits the mixin and
overrides its GUI hooks with the original widget code.  The oracle is Retranslation_GUI
frozen at ``U7_BASE_SHA`` (the U6 commit, the parent of the move).

* V: every moved method / module helper / constant equals the frozen one (modulo the listed
  class-name edits and three hook replacements); the dialog keeps no duplicate;
* F: offscreen reviewer goldens -- the frozen dialog and the working-tree dialog open the
  same fixture workspace (sidecar generation on open, alignment / row status analysis)
  and run the same actions (row edit, Notepad document edit, Manual-editing edit, Mark as
  Completed / Undo, Machine Translation preview with a scripted translator, Flag
  inaccurate, inject, threshold / provider settings); the output tree (sidecars, output
  HTML, ``translation_progress.json``, ``review_status_overrides.json``, the Machine
  Translation JSON, the freshness manifest), the piece rows and the status texts are equal;
* M: ``SdlxliffReviewSession`` (mobile) gives the same rows and the same file results;
* I: the module imports without PySide6 and parses as Python 3.10.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import textwrap
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

import sdlxliff_review_core as core  # noqa: E402

#: Parent of the U7 move (the U6 commit): the frozen Retranslation_GUI oracle.
U7_BASE_SHA = "41814faa95e273e956870bd3aac5a2c6fb7d66b1"


def legacy_rg():
    return pl.legacy_rg(U7_BASE_SHA)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(tmp_path / "_library"))
    monkeypatch.delenv("OUTPUT_DIRECTORY", raising=False)
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    monkeypatch.delenv("RETAIN_SOURCE_EXTENSION", raising=False)
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "1")
    # Reviewer settings without a parent window fall back to <app dir>/config.json:
    # never let a test write the real src/config.json.
    app_dir = tmp_path / "_app"
    app_dir.mkdir()
    current = pl.current_rg()   # imported before the patch: its re-export must stay the real function
    fake_app_dir = lambda: str(app_dir)  # noqa: E731
    for module in (core, current, legacy_rg()):
        monkeypatch.setattr(module, "_get_app_dir", fake_app_dir)
    saved = dict(os.environ)
    yield
    monkeypatch.undo()
    os.environ.clear()
    os.environ.update(saved)


def _frozen_lines():
    return pl.legacy_source_lines(U7_BASE_SHA)


def _frozen_text(start, end):
    return "\n".join(_frozen_lines()[start - 1:end])


def _source(name):
    return (SRC_DIR / f"{name}.py").read_text(encoding="utf-8").replace("\r\n", "\n")


def _class_methods(source, class_name):
    """{name: source incl. decorators and the comment lines right above} of a class."""
    tree = ast.parse(source)
    lines = source.split("\n")
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name][0]
    out = {}
    for node in cls.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
        while start - 2 >= 0 and lines[start - 2].strip().startswith("#"):
            start -= 1
        out[node.name] = "\n".join(lines[start - 1:node.end_lineno])
    return out


# ===========================================================================
# Tier V
# ===========================================================================

MOVED_DIALOG = (
    '_align_review_units', '_align_review_units_by_dom_position', '_annotate_review_tag_labels',
    '_append_machine_translation_note', '_apply_manual_green_override_to_piece', '_apply_notepad_document_edit',
    '_apply_persisted_manual_green_override', '_apply_target_edit', '_apply_tooltip_translation_status',
    '_build_notepad_review_rows', '_build_piece', '_build_review_piece_render_model_from_rows',
    '_build_review_refresh_scan_result', '_candidate_epub_paths_from_context', '_canonical_basename',
    '_canonical_review_path', '_changed_review_autogen_outputs', '_changed_review_signature_paths',
    '_chapter_number_from_name', '_clean_notepad_browser_html', '_clear_machine_accuracy_promotions',
    '_clear_piece_manual_green_override', '_clear_top_skew_promotions', '_compact_machine_translation_error',
    '_compact_machine_translation_text', '_compact_review_row_visible', '_compact_translator_note_display_label',
    '_comparison_tokens', '_current_machine_translation_signature', '_current_review_autogen_signature',
    '_current_review_signature', '_decrypt_machine_translation_api_key', '_dedupe_heading_paragraph_units',
    '_deduplicate_review_sidecar_paths', '_detect_notepad_mode_support', '_discover_review_books',
    '_emit_review_log_message', '_encrypt_machine_translation_api_key', '_ensure_review_image_assets',
    '_extract_text_units', '_extract_text_units_bs4', '_extract_text_units_lxml',
    '_extract_tooltip_batch_translations', '_filter_review_pieces', '_find_opf_path',
    '_flag_current_piece_inaccurate_translations', '_flush_target_edits', '_format_chapter_number',
    '_has_linguistic_letters', '_heading_paragraph_dedupe_key', '_heading_paragraph_tag_changed',
    '_heading_tag_level_changed', '_html_with_output_image_renames', '_infer_user_added_empty_target_indexes',
    '_initial_review_book_index', '_inject_current_machine_translation_to_target',
    '_inject_machine_translation_to_target', '_inner_xml_or_text', '_invalid_review_sidecar_outputs',
    '_invalid_review_sidecar_regen_key', '_invalidate_piece_render_model', '_latin_token_overlap',
    '_load_machine_translation_file_for_piece', '_load_pieces', '_local_name',
    '_machine_translation_accuracy_score', '_machine_translation_api_options',
    '_machine_translation_config_value', '_machine_translation_content_tokens',
    '_machine_translation_inaccuracy_threshold', '_machine_translation_path_for_piece',
    '_machine_translation_pending_text', '_machine_translation_provider', '_machine_translation_provider_label',
    '_machine_translation_result_note', '_machine_translation_row_key', '_machine_translation_source_hash',
    '_machine_translation_text_too_short_for_accuracy', '_machine_translation_translator',
    '_manual_green_empty_override_data', '_manual_green_override_entry_for_piece', '_manual_green_override_key',
    '_manual_green_overrides_path_for_output_dir', '_manual_green_overrides_path_for_piece',
    '_manual_green_piece_content_hash', '_manual_output_name_for_piece', '_mark_piece_progress_completed',
    '_mark_piece_progress_pending', '_mark_review_sidecars_completed', '_mark_review_sidecars_green',
    '_mark_tooltip_translation_pending', '_matching_review_progress_entries', '_maybe_regenerate_review_sidecars',
    '_merge_sdlxliff_generation_stats', '_missing_review_sidecar_outputs', '_non_empty_text_unit_count',
    '_non_empty_text_units', '_normalize_machine_translation_provider', '_normalize_review_text',
    '_normalized_machine_translation_text', '_notepad_document_prefix', '_notepad_initial_document_html',
    '_notepad_user_added_break_positions', '_notepad_user_added_target_indexes', '_original_name_from_output',
    '_output_dir_has_sdlxliff_sidecars', '_output_name_for_piece', '_output_path_for_piece',
    '_paragraph_list_item_tag_changed', '_persist_piece_manual_green_override', '_persist_review_config_value',
    '_piece_needs_manual_green_override', '_piece_render_snapshot', '_piece_tooltip_work',
    '_progress_path_for_review_piece', '_promote_inaccurate_machine_translation_rows',
    '_promote_top_skewed_row_for_count_mismatch', '_queue_generated_sidecar_stream_piece',
    '_read_machine_translation_file', '_read_manual_green_override_data', '_read_progress_metadata',
    '_read_sdlxliff_html_pair', '_read_spine_positions', '_recompute_piece_row_statuses',
    '_refresh_piece_summary', '_refresh_review_progress_completed_list',
    '_regenerate_manual_review_sidecars_for_refresh_scan', '_regenerate_review_sidecars_for_refresh_scan',
    '_reload_machine_translation_previews', '_remove_piece_manual_green_override',
    '_reset_machine_translation_threshold', '_restore_piece_progress_before_manual_completion',
    '_retain_source_extension_enabled', '_review_autogen_has_output_html', '_review_autogen_output_names',
    '_review_chars_per_line_for_width', '_review_context_menu_is_open', '_review_file_signature',
    '_review_generation_summary', '_review_label_from_metadata', '_review_layout_trailing_stretch_index',
    '_review_lxml_available', '_review_normalized_unit_text', '_review_notepad_mode_is_available',
    '_review_output_dirs', '_review_output_dirs_for_epub', '_review_piece_is_empty_sidecar',
    '_review_piece_non_empty_count', '_review_piece_worker_count', '_review_preload_order',
    '_review_progress_chapters', '_review_remove_duplicate_h1_p_enabled', '_review_row_height_for_width',
    '_review_row_index_property', '_review_row_line_counts_for_width', '_review_row_rendered_or_rendering',
    '_review_row_snapshot', '_review_row_text_heights', '_review_rows_for_current_layout',
    '_review_scroll_recently_active', '_review_selection_recently_changed', '_review_signature_path_map',
    '_review_signature_path_set_changed', '_review_signature_settings', '_review_source_epub_for_image_assets',
    '_review_status_colors', '_review_target_language', '_review_target_language_code',
    '_review_two_column_layout_enabled', '_review_unit_is_heading', '_review_unit_is_paragraph',
    '_review_units_are_compatible', '_review_wrapped_lines', '_row_expected_comparison_text',
    '_row_for_piece_path', '_row_length_ratio', '_row_machine_translation_preview_from_snapshot',
    '_row_machine_translation_preview_state', '_row_skew_metrics', '_row_status', '_row_tooltip_translation',
    '_schedule_generation_stream_flush', '_schedule_notepad_document_edit', '_schedule_target_edit',
    '_sdlxliff_sidecar_needs_source_regeneration', '_sdlxliff_sidecar_paths_for_output_dir',
    '_set_machine_translation_inaccuracy_threshold', '_set_machine_translation_provider',
    '_set_row_tooltip_translation', '_sidebar_label_for_piece', '_sidecar_metadata', '_sidecar_output_name',
    '_stale_review_sidecar_outputs', '_sync_cached_review_progress_data', '_tag_label_ordinal_font_point_size',
    '_tag_label_rich_text', '_tag_label_text', '_tag_mismatch_status', '_target_html_with_edit',
    '_tooltip_batch_html', '_tooltip_batch_tag_name', '_tooltip_translation_key', '_trace_review_perf',
    '_undo_all_target_edits', '_undo_review_sidecars_completed', '_undo_review_sidecars_green',
    '_unescape_html_document', '_validate_tooltip_batch_translations', '_wrapped_tooltip',
    '_write_machine_translation_entries', '_write_machine_translation_entry', '_write_manual_green_override_data',
    '_write_piece_target_html', '_write_review_progress_data',
)

MOVED_MIXIN = (
    '_generate_sdlxliff_sidecars_from_completed_entries',
    '_generate_sdlxliff_sidecars_from_untranslated_entries', '_output_dir_has_sdlxliff_generatable_html',
    '_output_dir_has_sdlxliff_sidecars', '_sdlxliff_add_spine_position', '_sdlxliff_autogen_bulk_read_sources',
    '_sdlxliff_autogen_decode', '_sdlxliff_autogen_epub_candidates', '_sdlxliff_autogen_output_path',
    '_sdlxliff_autogen_read_source_from_directory', '_sdlxliff_autogen_read_source_html',
    '_sdlxliff_autogen_source_candidates', '_sdlxliff_current_input_file_candidates',
    '_sdlxliff_exact_input_epub_candidates', '_sdlxliff_is_extracted_epub_dir', '_sdlxliff_preferred_input_epub',
    '_sdlxliff_sidecar_current_for_output', '_sdlxliff_source_spine_positions',
    '_sdlxliff_spine_position_for_entry', '_sdlxliff_update_source_epub_ref',
    '_sdlxliff_valid_current_input_epub_candidates', '_sdlxliff_valid_epub_path',
)

#: Class-qualified references inside moved code now name the mixin the code lives in.
CLASS_REF_EDITS = {
    "_non_empty_text_unit_count": [("SDLXLIFFReviewDialog.", "SdlxliffReviewCoreMixin.", 1)],
    "_non_empty_text_units": [("SDLXLIFFReviewDialog.", "SdlxliffReviewCoreMixin.", 1)],
    "_refresh_review_progress_completed_list": [("SDLXLIFFReviewDialog.", "SdlxliffReviewCoreMixin.", 1)],
    "_generate_sdlxliff_sidecars_from_completed_entries": [("SDLXLIFFReviewDialog.", "SdlxliffReviewCoreMixin.", 1)],
    "_candidate_epub_paths_from_context": [("RetranslationMixin.", "SdlxliffAutogenMixin.", 1)],
    "_sdlxliff_autogen_read_source_from_directory": [("RetranslationMixin.", "SdlxliffAutogenMixin.", 1)],
    "_sdlxliff_valid_epub_path": [("RetranslationMixin.", "SdlxliffAutogenMixin.", 1)],
    "_stale_review_sidecar_outputs": [("RetranslationMixin.", "SdlxliffAutogenMixin.", 1)],
}

_EDITOR_BLOCK = (
    "        try:\n"
    "            if isinstance(editor, QPlainTextEdit) and not editor.isReadOnly():\n"
    "                self._replace_editor_text_preserving_undo(editor, {var})\n"
    "                editor.setFocus(Qt.OtherFocusReason)\n"
    "                return\n"
    "        except Exception:\n"
    "            pass\n"
)

#: GUI hooks: GUI-free defaults in the mixin; the dialog keeps (or gets) the widget code.
HOOKS = (
    "_emit_review_generation_progress", "_displayed_piece_row", "_queue_review_data_preload",
    "_refresh_piece_list_item", "_refresh_piece_header", "_refresh_visible_review_row_status",
    "_invalidate_piece_page_for_refresh", "_refresh_open_notepad_machine_translation_context",
    "_queue_refresh_current_visible_dirty_source_previews", "_update_review_row_source_previews",
    "_update_machine_translation_button_tooltip", "_start_flag_accuracy_button_animation",
    "_queue_stop_flag_accuracy_button_animation", "_prepare_streaming_piece_list", "_stream_piece_list_item",
    "_finish_streaming_piece_list", "_pump_review_loading_events", "_set_loading_progress",
    "_refresh_notepad_page_after_save", "_insert_into_review_editor", "_prompt_machine_translation_credentials",
)


@pytest.fixture(scope="module")
def frozen_methods():
    text = "\n".join(_frozen_lines())
    return {
        "SDLXLIFFReviewDialog": _class_methods(text, "SDLXLIFFReviewDialog"),
        "RetranslationMixin": _class_methods(text, "RetranslationMixin"),
    }


@pytest.fixture(scope="module")
def core_methods():
    text = _source("sdlxliff_review_core")
    return {
        "SdlxliffReviewCoreMixin": _class_methods(text, "SdlxliffReviewCoreMixin"),
        "SdlxliffAutogenMixin": _class_methods(text, "SdlxliffAutogenMixin"),
    }


def _expected_moved(frozen, name):
    src = frozen + "\n"
    for old, new, count in CLASS_REF_EDITS.get(name, []):
        assert src.count(old) == count, (name, old)
        src = src.replace(old, new)
    if name == "_apply_notepad_document_edit":
        start = src.index("        page = self._piece_pages.get(piece_index)\n")
        end = src.index("        try:\n            self._last_review_signature = self._current_review_signature()\n")
        src = src[:start] + (
            "        self._refresh_notepad_page_after_save(piece_index, rebuilt, saved_html, html_text)\n"
        ) + src[end:]
    for method, var in (("_inject_machine_translation_to_target", "translated"),
                        ("_undo_all_target_edits", "original")):
        if name == method:
            block = _EDITOR_BLOCK.format(var=var)
            assert src.count(block) == 1
            src = src.replace(block, f"        if self._insert_into_review_editor(editor, {var}):\n            return\n")
    return src.rstrip("\n")


@pytest.mark.parametrize("name", MOVED_DIALOG)
def test_moved_dialog_methods_are_verbatim(name, frozen_methods, core_methods):
    assert core_methods["SdlxliffReviewCoreMixin"][name] == _expected_moved(
        frozen_methods["SDLXLIFFReviewDialog"][name], name)


@pytest.mark.parametrize("name", MOVED_MIXIN)
def test_moved_autogen_methods_are_verbatim(name, frozen_methods, core_methods):
    assert core_methods["SdlxliffAutogenMixin"][name] == _expected_moved(
        frozen_methods["RetranslationMixin"][name], name)


def test_module_helpers_and_constants_are_verbatim():
    source = _source("sdlxliff_review_core")
    for start, end in ((342, 668), (714, 737), (739, 745), (846, 925), (9378, 9397), (14979, 14990)):
        assert _frozen_text(start, end) in source, (start, end)


def test_split_helpers_hold_the_frozen_blocks(core_methods):
    methods = core_methods["SdlxliffReviewCoreMixin"]
    assert _frozen_text(939, 958) in methods["_init_review_state"]
    lines = _frozen_lines()
    assert "\n".join(line[8:] if line.strip() else "" for line in lines[10145 - 1:10157]) in \
        methods["_translate_tooltip_work"]
    assert "\n".join(line[4:] if line.strip() else "" for line in lines[10316 - 1:10341]) in \
        methods["_store_tooltip_translations"]
    header = methods["_piece_header_text"]
    assert "\n".join(line[4:] for line in lines[7385 - 1:7390]) in header
    assert "\n".join(line[4:] for line in lines[7392 - 1:7394]) in header
    message = methods["_tooltip_translation_result_message"]
    for text in ('f"Machine translation preview failed: {self._compact_machine_translation_error(error)}"',
                 'message = f"Generated {len(translations)} {provider_label} machine translation preview(s)"',
                 'message = f"{message}. {self._compact_machine_translation_error(error)}"'):
        assert text in _frozen_text(10361, 10377) and text in message


def test_dialog_inherits_the_core_and_keeps_only_gui_code():
    rg = pl.current_rg()
    dialog = rg.SDLXLIFFReviewDialog
    assert dialog.__mro__[1] is core.SdlxliffReviewCoreMixin
    assert issubclass(rg.RetranslationMixin, core.SdlxliffAutogenMixin)
    assert rg.RetranslationMixin.__mro__[1].__name__ == "ProgressViewMixin"
    for name in MOVED_DIALOG:
        assert name not in vars(dialog), name
        assert name in vars(core.SdlxliffReviewCoreMixin), name
    for name in MOVED_MIXIN:
        assert name not in vars(rg.RetranslationMixin), name
        assert name in vars(core.SdlxliffAutogenMixin), name
    for name in HOOKS:
        assert name in vars(core.SdlxliffReviewCoreMixin), name
        assert name in vars(dialog), name
    # the dialog's hook overrides are the frozen widget code
    frozen = _class_methods("\n".join(_frozen_lines()), "SDLXLIFFReviewDialog")
    current = _class_methods(_source("Retranslation_GUI"), "SDLXLIFFReviewDialog")
    for name in HOOKS:
        if name in frozen and name != "_refresh_piece_header":
            assert current[name] == frozen[name], name
    notepad = frozen["_apply_notepad_document_edit"]
    block = notepad[notepad.index("        page = self._piece_pages.get(piece_index)"):
                    notepad.index("        try:\n            self._last_review_signature")]
    assert block.rstrip("\n") in current["_refresh_notepad_page_after_save"]
    assert "self._piece_header_text(piece_index)" in current["_refresh_piece_header"]
    assert "self._translate_tooltip_work(translator, work)" in current["_start_tooltip_translation"]
    assert "self._translate_tooltip_work(translator, work)" in current["_start_piece_list_tooltip_translation"]
    assert "self._store_tooltip_translations(row, translations, error)" in current["_apply_tooltip_translations"]
    assert "self._tooltip_translation_result_message(translations, error)" in current["_apply_tooltip_translations"]
    assert "self._init_review_state(" in current["__init__"]


def test_retranslation_gui_reexports_the_moved_helpers():
    rg = pl.current_rg()
    names = [n for n in vars(core) if n.startswith(("_sdlxliff", "_SDLXLIFF", "_read_sdlxliff", "_update_sdlxliff",
                                                     "_existing_sdlxliff", "_manual_editing_output"))]
    names += ["_MACHINE_TRANSLATION_DIR", "_get_app_dir"]
    assert len(names) > 15
    for name in names:
        assert getattr(rg, name) is getattr(core, name), name


def test_core_imports_without_qt_and_parses_as_310():
    ast.parse(_source("sdlxliff_review_core"), feature_version=(3, 10))
    code = (
        "import sys; sys.modules['PySide6'] = None; sys.path.insert(0, %r);"
        "import sdlxliff_review_core, progress_actions;"
        "assert 'Retranslation_GUI' not in sys.modules and 'translator_gui' not in sys.modules;"
        "print('ok')" % str(SRC_DIR)
    )
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0 and out.stdout.strip().endswith("ok"), out.stderr[-2000:]


# ===========================================================================
# Fixture workspace
# ===========================================================================


def _html(body):
    return f"<html><head><title>t</title></head><body>{body}</body></html>"


SOURCE_CHAPTERS = [
    ("chapter0001.xhtml", "<h1>제1장</h1><p>첫 문장입니다.</p><p>두 번째 문장입니다.</p>"),
    ("chapter0002.xhtml", "<h1>제2장</h1><p>하나입니다</p><p>둘입니다</p><p>셋입니다</p>"),
    ("chapter0003.xhtml", "<h1>제3장</h1><p>본문입니다</p>"),
    ("chapter0004.xhtml", "<p>번역되지 않은 문장입니다.</p>"),
    ("chapter0005.xhtml", "<p>짧은 원문입니다</p>"),
    ("chapter0006.xhtml", "<p>수동 번역 원문입니다.</p><p>두 번째 수동 문장입니다.</p>"),
]
TARGETS = {
    1: "<h1>Chapter 1</h1><p>This is the first sentence.</p><p>This is the second sentence.</p>",  # aligned
    2: "<h1>Chapter 2</h1><p>It is one</p><p>It is three</p>",          # a dropped paragraph
    3: "<p>Chapter 3</p><p>It is the body</p>",                          # heading became a paragraph
    4: "<p>번역되지 않은 문장입니다.</p>",                                   # left untranslated
    5: "<p>A short source</p><p>An extra added paragraph.</p>",          # an added paragraph
}
MT_TEXT = {
    "제2장": "Chapter 2", "하나입니다": "It is one", "둘입니다": "It is two", "셋입니다": "It is three",
    "제1장": "Chapter 1", "첫 문장입니다.": "This is the first sentence.",
    "두 번째 문장입니다.": "This is the second sentence.",
}
CONFIG = {"output_language": "English"}


def review_workspace(base):
    """An EPUB, its output folder (5 translated chapters, one manual-editing sidecar for an
    untranslated chapter) and the SDLXLIFF sidecars of the translated chapters."""
    from sdlxliff_sidecar_writer import _write_html_sdlxliff_sidecar

    base = Path(base)
    source = base / "Novel.epub"
    pl.make_epub(source, [(name, _html(body)) for name, body in SOURCE_CHAPTERS])
    out = base / "out" / "Novel"
    out.mkdir(parents=True)
    prog = {"version": "2.1", "chapters": {}, "chapter_chunks": {}}
    for num, target in TARGETS.items():
        name = f"response_chapter{num:04d}.html"
        (out / name).write_text(_html(target), encoding="utf-8")
        prog["chapters"][str(num)] = {
            "actual_num": num, "content_hash": f"h{num}", "output_file": name, "status": "completed",
            "original_basename": f"chapter{num:04d}.xhtml", "last_updated": 1000.0 + num,
        }
        _write_html_sdlxliff_sidecar(str(out), name, {"original_basename": f"chapter{num:04d}.xhtml"},
                                     _html(SOURCE_CHAPTERS[num - 1][1]), _html(target), raise_errors=True,
                                     record_freshness=False)
    prog["chapters"]["6"] = {"actual_num": 6, "content_hash": "h6", "output_file": "response_chapter0006.html",
                             "status": "not_translated", "original_basename": "chapter0006.xhtml",
                             "last_updated": 1006.0}
    manual_source = _html(SOURCE_CHAPTERS[5][1])
    _write_html_sdlxliff_sidecar(str(out), "response_chapter0006.html",
                                 {"original_basename": "chapter0006.xhtml"}, manual_source, manual_source,
                                 raise_errors=True, manual_untranslated=True, record_freshness=False)
    (out / "translation_progress.json").write_text(json.dumps(prog, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    pl._set_mtime(base)
    return source, out


@pytest.fixture(scope="module")
def workspace_root(tmp_path_factory):
    base = tmp_path_factory.mktemp("u7_sdlxliff")
    review_workspace(base)
    return base


_TIME_KEYS = ("mtime", "updated", "_at", "timestamp", "time")


def _normalize(value):
    """Run-generated timestamps (json numbers past 2017 under time-like keys) -> '<t>'."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if (isinstance(item, (int, float)) and not isinstance(item, bool) and item >= 1_500_000_000
                    and any(part in str(key).lower() for part in _TIME_KEYS)):
                out[key] = "<t>"
            else:
                out[key] = _normalize(item)
        return out
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def _tree(work):
    tree = pl.tree_snapshot(work / "out", workspace=work)
    return {rel: (kind, _normalize(value) if kind == "json" else value) for rel, (kind, value) in tree.items()}


def _rows(pieces):
    keep = ("status", "reason", "source", "target", "source_tag", "target_tag", "source_index", "target_index",
            "tooltip_translation")
    return [
        {
            "name": piece.get("name"), "output_name": piece.get("output_name"),
            "label": piece.get("review_label"), "mismatch": piece.get("mismatch"),
            "counts": (piece.get("source_count"), piece.get("target_count"), piece.get("red_count"),
                       piece.get("yellow_count"), piece.get("purple_count")),
            "manual": (piece.get("manual_editing"), piece.get("manual_untranslated"),
                       bool(piece.get("manual_green_override"))),
            "rows": [{k: row.get(k) for k in keep} for row in piece.get("rows") or []],
        }
        for piece in pieces
    ]


class _Parent:
    """The translator window as the reviewer sees it: config + save_config."""

    def __init__(self, config):
        self.config = config
        self.saves = 0

    def save_config(self, show_message=False):
        self.saves += 1


class _Translator:
    """google_free_translate stand-in: fills every ``data-sdl-tip`` node."""

    def translate(self, batch_html):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(batch_html, "html.parser")
        for node in soup.find_all(attrs={"data-sdl-tip": True}):
            text = node.get_text(" ", strip=True)
            node.string = MT_TEXT.get(text, f"MT {text}")
        return {"translatedText": str(soup)}


def _translator_factory(_target_code, status_callback=None):
    return _Translator()


# --- the two front ends behind one interface --------------------------------


class _Desktop:
    def __init__(self, module, work, config):
        self.module = module
        out = work / "out" / "Novel"
        self.parent = _Parent(config)
        host = pl.make_host(module, config)
        self.dialog = module.SDLXLIFFReviewDialog(
            str(out), None, None, config=config, autogen_owner=host,
            autogen_file_path=str(work / "Novel.epub"),
        )
        self.dialog._context_parent = self.parent
        self.dialog._machine_translation_translator = _translator_factory
        self.dialog.show()
        pl.pump(30, until=lambda: bool(self.dialog._review_data_loaded) and not self.dialog._review_refresh_scan_running,
                timeout=20.0)
        timer = getattr(self.dialog, "_auto_refresh_timer", None)
        if timer is not None:
            timer.stop()
        pl.pump(20, until=lambda: not getattr(self.dialog, "_review_piece_reload_running", False), timeout=5.0)

    @property
    def pieces(self):
        return self.dialog.pieces

    def status(self):
        return self.dialog.save_status_label.text()

    def save_row(self, piece, row, text):
        self.dialog._schedule_target_edit(piece, row, text)
        self.dialog._edit_save_timer.stop()
        self.dialog._flush_target_edits()

    def save_document(self, piece, html_text):
        self.dialog._schedule_notepad_document_edit(piece, html_text)
        self.dialog._edit_save_timer.stop()
        self.dialog._flush_target_edits()

    def mark(self, rows):
        self.dialog._mark_review_sidecars_completed(rows)

    def undo(self, rows):
        self.dialog._undo_review_sidecars_completed(rows)

    def preview(self, piece):
        assert self.dialog._start_tooltip_translation(piece, self.dialog._piece_tooltip_work(piece))
        pl.pump(20, until=lambda: not self.dialog._tooltip_translation_running, timeout=10.0)

    def flag(self, piece):
        self.dialog._displayed_piece_row = lambda: piece
        self.dialog._flag_current_piece_inaccurate_translations()

    def inject(self, piece, row):
        self.dialog._inject_current_machine_translation_to_target(piece, row, None)

    def settings(self):
        self.dialog._set_machine_translation_provider("google")
        statuses = [self.status()]
        self._prompt_threshold(80)
        statuses.append(self.status())
        self.dialog._reset_machine_translation_threshold()
        statuses.append(self.status())
        return statuses

    def _prompt_threshold(self, value):
        # "Set Score Threshold..." answers its QInputDialog prompt
        from PySide6.QtWidgets import QInputDialog

        original = QInputDialog.getDouble
        QInputDialog.getDouble = staticmethod(lambda *args, **kwargs: (value, True))
        try:
            self.dialog._prompt_machine_translation_threshold()
        finally:
            QInputDialog.getDouble = original

    def close(self):
        try:
            self.dialog._edit_save_timer.stop()
            self.dialog.hide()
        except Exception:
            pass


class _Mobile:
    def __init__(self, work, config):
        out = work / "out" / "Novel"
        self.parent = _Parent(config)
        self.session = core.open_sdlxliff_review(
            str(out), config, context_parent=self.parent, source_path=str(work / "Novel.epub"),
        )
        self.session._machine_translation_translator = _translator_factory

    @property
    def pieces(self):
        return self.session.pieces

    def status(self):
        return self.session.status

    def save_row(self, piece, row, text):
        self.session.save_row(piece, row, text)

    def save_document(self, piece, html_text):
        self.session.edit_document(piece, html_text)
        self.session.flush_edits()

    def mark(self, rows):
        self.session.mark_completed(rows)

    def undo(self, rows):
        self.session.undo_completed(rows)

    def preview(self, piece):
        self.session.machine_translation_preview(piece)

    def flag(self, piece):
        self.session.flag_inaccurate(piece)

    def inject(self, piece, row):
        self.session.inject_machine_translation(piece, row)

    def settings(self):
        statuses = [self.session.set_provider("google")]
        self.session.set_inaccuracy_threshold(80)
        statuses.append(self.status())
        self.session.reset_inaccuracy_threshold()
        statuses.append(self.status())
        return statuses

    def close(self):
        pass


def _piece_index(front, chapter):
    for index, piece in enumerate(front.pieces):
        if piece.get("chapter_num") == chapter:
            return index
    raise AssertionError(chapter)


def _scenario_load(front):
    return []


def _scenario_edit_row(front):
    p = _piece_index(front, 2)
    dropped = [i for i, row in enumerate(front.pieces[p]["rows"]) if not row.get("target")][0]
    front.save_row(p, dropped, "It is three")
    return [front.status()]


def _scenario_edit_document(front):
    p = _piece_index(front, 3)
    front.save_document(p, _html("<h1>Chapter 3</h1><p>It is the body</p>"))
    return [front.status()]


def _scenario_edit_manual(front):
    p = _piece_index(front, 6)
    front.save_row(p, 0, "This is the manual sentence.")
    return [front.status()]


def _scenario_mark_completed(front):
    front.mark([_piece_index(front, 2), _piece_index(front, 4), _piece_index(front, 1)])
    return [front.status()]


def _scenario_mark_then_undo(front):
    p = _piece_index(front, 2)
    front.mark([p])
    statuses = [front.status()]
    front.undo([p])
    statuses.append(front.status())
    return statuses


def _scenario_preview_flag_inject(front):
    p = _piece_index(front, 2)
    front.preview(p)
    statuses = [front.status()]
    front.flag(p)
    statuses.append(front.status())
    yellow = [i for i, row in enumerate(front.pieces[p]["rows"]) if row.get("source") == "둘입니다"][0]
    front.inject(p, yellow)
    statuses.append(front.status())
    return statuses


def _scenario_settings(front):
    return front.settings() + [json.dumps({k: v for k, v in sorted(front.parent.config.items())
                                           if k.startswith("sdlxliff_")})]


SCENARIOS = {
    "load": (_scenario_load, {}, ()),
    "regenerate_missing_sidecars": (_scenario_load, {}, ("response_chapter0002.html.sdlxliff",
                                                         "response_chapter0005.html.sdlxliff")),
    "edit_row": (_scenario_edit_row, {}, ()),
    "edit_notepad_document": (_scenario_edit_document, {}, ()),
    "edit_manual_editing_sidecar": (_scenario_edit_manual, {"retranslation_manual_editing": True}, ()),
    "mark_as_completed": (_scenario_mark_completed, {}, ()),
    "mark_then_undo": (_scenario_mark_then_undo, {}, ()),
    "mt_preview_flag_inject": (_scenario_preview_flag_inject, {}, ()),
    "threshold_and_provider_settings": (_scenario_settings, {}, ()),
}


def _run(kind, workspace_root, work, scenario):
    action, extra_config, delete_sidecars = SCENARIOS[scenario]
    saved_env = dict(os.environ)
    work = pl.copy_workspace(workspace_root, work)
    for name in delete_sidecars:
        (work / "out" / "Novel" / "SDLXLIFF" / name).unlink()
    config = dict(CONFIG, output_directory=str(work / "out"), **extra_config)
    front = None
    try:
        if kind == "mobile":
            front = _Mobile(work, config)
        else:
            front = _Desktop(legacy_rg() if kind == "legacy" else pl.current_rg(), work, config)
        loaded_rows = _rows(front.pieces)
        statuses = action(front)
        result = {
            "loaded": loaded_rows,
            "rows": _rows(front.pieces),
            "statuses": [str(s).replace(str(work), "<ROOT>") for s in statuses],
            "tree": _tree(work),
            "parent_saves": front.parent.saves,
            "work": work,
        }
        return result
    finally:
        if front is not None:
            front.close()
        os.environ.clear()
        os.environ.update(saved_env)


_LEGACY = {}


def _legacy(scenario, workspace_root, tmp_path_factory):
    if scenario not in _LEGACY:
        _LEGACY[scenario] = _run("legacy", workspace_root, tmp_path_factory.mktemp(f"legacy_{scenario}") / "w",
                                 scenario)
    return _LEGACY[scenario]


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
    return [] if x == y else [f"{path}: {x!r:.200} != {y!r:.200}"]


# ===========================================================================
# Tier F: the reviewer against the frozen dialog
# ===========================================================================


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_reviewer_matches_frozen_desktop(scenario, workspace_root, tmp_path, tmp_path_factory):
    legacy = _legacy(scenario, workspace_root, tmp_path_factory)
    current = _run("current", workspace_root, tmp_path / "current", scenario)
    assert current["loaded"] == legacy["loaded"]
    assert current["rows"] == legacy["rows"]
    assert current["statuses"] == legacy["statuses"]
    assert current["tree"] == legacy["tree"], _diff(legacy["tree"], current["tree"])
    assert current["parent_saves"] == legacy["parent_saves"]


# ===========================================================================
# Tier M: the mobile session
# ===========================================================================


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_mobile_session_matches_desktop(scenario, workspace_root, tmp_path, tmp_path_factory):
    legacy = _legacy(scenario, workspace_root, tmp_path_factory)
    mobile = _run("mobile", workspace_root, tmp_path / "mobile", scenario)
    assert mobile["loaded"] == legacy["loaded"]
    assert mobile["rows"] == legacy["rows"]
    assert mobile["statuses"] == legacy["statuses"]
    assert mobile["tree"] == legacy["tree"], _diff(legacy["tree"], mobile["tree"])


def test_scenarios_do_what_they_should(workspace_root, tmp_path_factory):
    """Spot checks of the goldens (every front end is equal by the parity tests)."""
    load = _legacy("load", workspace_root, tmp_path_factory)
    by_name = {piece["output_name"]: piece for piece in load["loaded"]}
    assert len(by_name) == 6
    assert {row["status"] for row in by_name["response_chapter0001.html"]["rows"]} == {"green"}
    assert "red" in {row["status"] for row in by_name["response_chapter0002.html"]["rows"]}
    assert by_name["response_chapter0003.html"]["rows"][0]["status"] == "yellow"
    assert by_name["response_chapter0004.html"]["rows"][0]["reason"] == "untranslated"
    regenerated = _legacy("regenerate_missing_sidecars", workspace_root, tmp_path_factory)
    assert "Novel/SDLXLIFF/response_chapter0002.html.sdlxliff" in regenerated["tree"]
    assert "Novel/SDLXLIFF/sdlxliff_manifest.json" in regenerated["tree"]

    edit = _legacy("edit_row", workspace_root, tmp_path_factory)
    assert "It is three</p>" in edit["tree"]["Novel/response_chapter0002.html"][1]
    manual = _legacy("edit_manual_editing_sidecar", workspace_root, tmp_path_factory)
    assert "This is the manual sentence." in manual["tree"]["Novel/response_chapter0006.html"][1]
    chapters = manual["tree"]["Novel/translation_progress.json"][1]["chapters"]
    assert chapters["6"]["status"] == "not_translated"   # the existing entry stays authoritative

    marked = _legacy("mark_as_completed", workspace_root, tmp_path_factory)
    overrides = marked["tree"]["Novel/SDLXLIFF/review_status_overrides.json"][1]
    assert len(overrides["entries"]) == 2
    assert marked["tree"]["Novel/translation_progress.json"][1]["chapters"]["2"]["manually_marked_completed"]
    undone = _legacy("mark_then_undo", workspace_root, tmp_path_factory)
    assert "manually_marked_completed" not in undone["tree"]["Novel/translation_progress.json"][1]["chapters"]["2"]

    mt = _legacy("mt_preview_flag_inject", workspace_root, tmp_path_factory)
    assert any(rel.startswith("Novel/SDLXLIFF/Machine_Translation/") for rel in mt["tree"])
    assert "It is two" in mt["tree"]["Novel/response_chapter0002.html"][1]
    assert mt["statuses"][0].startswith("Generated ")


def test_mobile_session_api(workspace_root, tmp_path, monkeypatch):
    import api_key_encryption

    from cryptography.fernet import Fernet

    monkeypatch.setenv("GLOSSARION_API_KEY_FERNET", Fernet.generate_key().decode("ascii"))
    monkeypatch.setattr(api_key_encryption, "_handler", None)
    work = pl.copy_workspace(workspace_root, tmp_path / "api")
    config = dict(CONFIG, output_directory=str(work / "out"))
    parent = _Parent(config)
    progress = []
    session = core.open_sdlxliff_review(str(work / "out" / "Novel"), config, context_parent=parent,
                                        source_path=str(work / "Novel.epub"), progress_callback=progress.append)
    assert len(session.pieces) == 6 and session.books and session.books[0]["output_dir"]
    # The pieces follow the SDLXLIFF folder's listing order when the progress has no spine
    # positions (desktop rule): ext4 lists in hash order, so find chapter 1 by name.
    idx = next(i for i, p in enumerate(session.pieces) if p["output_name"] == "response_chapter0001.html")
    summary = session.piece_summary(idx)
    assert "| response_chapter0001.html  - " in summary["header"]
    assert not session.changed_on_disk()
    assert session.refresh()["reloaded"] is False
    # Notepad document of a piece
    assert "This is the first sentence." in session.notepad_document(idx)
    # credentials: missing -> refused with the desktop text; stored encrypted
    assert session.set_provider("deepl") == "DeepL requires an API key"
    assert session.provider == "auto"
    assert session.set_machine_translation_credentials("deepl", api_key="secret-key")
    assert config["sdlxliff_machine_translation_deepl_api_key"] != "secret-key"
    assert session._machine_translation_api_options() == {"deepl": {"api_key": "secret-key"}}
    assert session.set_provider("deepl").startswith("Machine translation provider: DeepL")
    assert parent.saves >= 2
    assert not session.set_machine_translation_credentials("yandex", api_key="k")
    assert session.set_machine_translation_credentials("yandex", folder_id="folder")
    # an output edit on disk is seen by the 2 s poll
    out_file = work / "out" / "Novel" / "SDLXLIFF" / "response_chapter0001.html.sdlxliff"
    os.utime(out_file, (time.time() + 5, time.time() + 5))
    assert session.changed_on_disk()
    # undo a row edit restores the original output text
    session.save_row(idx, 1, "Changed sentence.")
    assert "Changed sentence." in (work / "out" / "Novel" / "response_chapter0001.html").read_text(encoding="utf-8")
    session.undo_row(idx, 1)
    assert "This is the first sentence." in (work / "out" / "Novel" / "response_chapter0001.html").read_text(
        encoding="utf-8")
