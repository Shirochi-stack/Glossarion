import ast
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import pytest
from PIL import Image

import Chapter_Extractor as chapter_extractor
import epub_converter
import translate_headers_standalone
from QA_Scanner_GUI import _normalize_qa_dialog_path
from enhanced_text_extractor import EnhancedTextExtractor
from epub_converter import EPUBCompiler, FileUtils, HTMLEntityDecoder, XMLValidator
from html_tag_entities import unescape_valid_html_tag_entities
from metadata_batch_translator import BatchHeaderTranslator
from qa_scan_runtime import (
    active_qa_output_folder_for_source,
    automatic_qa_output_candidates,
    default_qa_scan_settings,
    is_direct_text_qa_path,
    run_qa_scan_path,
)
from scan_html_folder import (
    _count_quotation_marks,
    _missing_ending_quotation_paragraphs,
    cross_reference_word_counts,
    detect_quotation_mismatch,
    extract_epub_punctuation_info,
    extract_epub_quotation_info,
    extract_html_word_counts,
    generate_reports,
    process_html_file_batch,
    scan_html_folder,
    update_new_format_progress,
)


def test_header_fallback_parser_respects_matched_quote_delimiters():
    translator = BatchHeaderTranslator(None, {})
    malformed_response = (
        "Translation results:\n"
        '"1": "It\'s \\"ready\\".",\n'
        "'2': 'She said \"go\"; it\\'s time.',\n"
        '3: "Punctuation: commas, braces {}, and [brackets]"\n'
    )

    assert translator._parse_json_response(
        malformed_response,
        {1: "source one", 2: "source two", 3: "source three"},
    ) == {
        1: 'It\'s "ready".',
        2: 'She said "go"; it\'s time.',
        3: "Punctuation: commas, braces {}, and [brackets]",
    }


def test_cancelled_native_folder_dialog_values_are_rejected(tmp_path):
    assert _normalize_qa_dialog_path(False) == ""
    assert _normalize_qa_dialog_path(None) == ""
    assert _normalize_qa_dialog_path("") == ""
    assert _normalize_qa_dialog_path("false") == ""
    assert _normalize_qa_dialog_path(("false", "ignored")) == ""

    # Only a bare sentinel is rejected; a legitimate absolute path whose final
    # component happens to use that name remains a valid path value.
    real_path = tmp_path / "false"
    assert _normalize_qa_dialog_path(real_path) == str(real_path)


def test_qa_executor_worker_does_not_construct_qt_dialogs():
    source_path = Path(__file__).resolve().parents[1] / "src" / "QA_Scanner_GUI.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    run_scan_worker = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "QAScannerMixin":
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == "run_qa_scan":
                    run_scan_worker = next(
                        child
                        for child in method.body
                        if isinstance(child, ast.FunctionDef) and child.name == "run_scan"
                    )
                    break

    assert run_scan_worker is not None
    referenced_names = {
        node.id for node in ast.walk(run_scan_worker) if isinstance(node, ast.Name)
    }
    assert "QMessageBox" not in referenced_names
    assert "QFileDialog" not in referenced_names


def test_standalone_html_source_matches_response_html_by_filename(tmp_path, monkeypatch):
    monkeypatch.setenv("QA_EXACT_CHAR_COUNT", "1")
    monkeypatch.delenv("QA_USE_WORD_COUNT", raising=False)
    source_path = tmp_path / "chapter0001.html"
    source_path.write_text(
        "<html><body><h1>Chapter One</h1><p>Source text for comparison.</p></body></html>",
        encoding="utf-8",
    )

    source_counts = extract_html_word_counts(
        source_path,
        log=lambda _message: None,
    )
    result = cross_reference_word_counts(
        source_counts,
        "response_chapter0001.xhtml",
        "Chapter One\nSource text for comparison.",
        log=lambda _message: None,
        qa_settings={
            "min_duplicate_word_count": 0,
            "source_language": "english",
            "target_language": "english",
            "word_count_multipliers": {"english": 1.0, "other": 1.0},
        },
    )

    assert source_counts[1]["filename"] == source_path.name
    assert source_counts[1]["has_headers"] is True
    assert result["found_match"] is True
    assert result["original_file"] == source_path.name
    assert result["ratio"] == 1.0
    assert result["is_reasonable"] is True


def test_standalone_html_source_supports_word_count_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("QA_USE_WORD_COUNT", "1")
    monkeypatch.delenv("QA_EXACT_CHAR_COUNT", raising=False)
    source_path = tmp_path / "single.htm"
    source_path.write_text(
        "<html><body><p>one two three four</p></body></html>",
        encoding="utf-8",
    )

    source_counts = extract_html_word_counts(source_path, log=lambda _message: None)

    assert source_counts[1]["word_count"] == 4
    assert source_counts[1]["small_file_word_count"] == 4


def test_qa_scan_cross_references_standalone_html_source_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("QA_EXACT_CHAR_COUNT", "1")
    monkeypatch.delenv("QA_USE_WORD_COUNT", raising=False)
    source_path = tmp_path / "chapter0002.html"
    source_path.write_text(
        "<html><body><h1>Chapter Two</h1><p>A complete standalone source chapter.</p></body></html>",
        encoding="utf-8",
    )
    output_dir = tmp_path / "translated"
    output_dir.mkdir()
    (output_dir / "response_chapter0002.xhtml").write_text(
        "<html><body><h1>Chapter Two</h1><p>A complete standalone source chapter.</p></body></html>",
        encoding="utf-8",
    )
    settings = default_qa_scan_settings()
    settings.update(
        {
            "check_word_count_ratio": True,
            "min_duplicate_word_count": 0,
            "check_missing_images": False,
            "check_punctuation_mismatch": False,
            "check_quotation_mismatch": False,
            "check_silent_truncation": False,
            "check_ai_truncation_detection": False,
            "check_multiple_headers": False,
            "check_repetition": False,
            "check_translation_artifacts": False,
            "check_glossary_leakage": False,
            "use_thread_executor": True,
            "source_language": "english",
            "target_language": "english",
        }
    )

    scan_html_folder(
        str(output_dir),
        log=lambda _message: None,
        mode="quick-scan",
        qa_settings=settings,
        epub_path=str(source_path),
    )

    report_path = (
        output_dir
        / f"{output_dir.name}_Scan Report"
        / "validation_results.json"
    )
    results = json.loads(report_path.read_text(encoding="utf-8"))
    translated_result = next(
        row for row in results if row["filename"] == "response_chapter0002.xhtml"
    )
    word_count_check = translated_result["word_count_check"]
    assert word_count_check["found_match"] is True
    assert word_count_check["original_file"] == source_path.name
    assert word_count_check["ratio"] == 1.0


def test_html_entity_decoder_basic_entities():
    text = "&lt;Hello&gt; &amp; &quot;World&quot; &apos;!&apos;"
    decoded = HTMLEntityDecoder.decode(text)
    # Expect: <Hello> & "World" '!'
    assert decoded == "<Hello> & \"World\" '!'"


def test_quotation_check_defaults_off():
    settings = default_qa_scan_settings()
    assert settings["check_quotation_mismatch"] is False
    assert settings["ignore_excess_quotation_marks"] is False
    assert settings["only_check_incomplete_quotations"] is False
    assert settings["ignore_consecutive_missing_quotations"] is False
    assert settings["skip_stylistic_single_quotes"] is False
    assert settings["include_square_brackets_as_quotations"] is False


def test_quotation_counter_handles_styles_entities_and_apostrophes():
    text = (
        '&quot;double&quot; '
        '&#39;single&#39; '
        '&#x27;hex&#x27; '
        '&#x2018;curly&#x2019; '
        '&apos;named&apos; '
        '&#x39;source typo&#x39; '
        '「corner」 『white corner』 《book title》 '
        "don't"
    )

    assert _count_quotation_marks(text) == 18


def test_quotation_counter_can_skip_balanced_stylistic_single_quotes():
    text = "Use 'Naught' here, but preserve “dialogue” and an unmatched ' mark."

    assert _count_quotation_marks(text) == 5
    assert _count_quotation_marks(text, skip_stylistic_single_quotes=True) == 3


def test_stylistic_toggle_skips_possessives_and_balanced_double_quotes():
    text = "A Girls' Love story highlights the protagonists' love through a \"honey\" job."

    assert _count_quotation_marks(text) == 4
    assert _count_quotation_marks(text, skip_stylistic_single_quotes=True) == 0


def test_stylistic_toggle_skips_curly_possessive_but_keeps_curly_quote_pair():
    text = "The ladies’ popularity increased. ‘Quoted thought’"

    assert _count_quotation_marks(text) == 3
    assert _count_quotation_marks(text, skip_stylistic_single_quotes=True) == 2


def test_square_brackets_are_optional_quotation_marks():
    text = "[quoted text]"

    assert _count_quotation_marks(text) == 0
    assert _count_quotation_marks(text, include_square_brackets=True) == 2


def test_square_bracket_option_detects_incomplete_opening_bracket():
    html = '<p>[complete]</p><p>[missing ending</p>'

    assert _missing_ending_quotation_paragraphs(html) == []
    missing = _missing_ending_quotation_paragraphs(
        html,
        include_square_brackets=True,
    )

    assert [item["paragraph_index"] for item in missing] == [2]
    assert missing[0]["missing_marks"] == ["]"]


def test_missing_ending_quotation_uses_odd_straight_quote_paragraph_check():
    html = (
        '<p>&quot;complete pair&quot;</p>'
        '<p>&quot;missing ending</p>'
        '<p>“curly missing ending.</p>'
        '<p>「CJK complete pair」</p>'
        '<p>『CJK missing ending.</p>'
        '<p><span title="attribute">no text quote</span></p>'
    )

    missing = _missing_ending_quotation_paragraphs(html)

    assert [item["paragraph_index"] for item in missing] == [2, 3, 5]
    assert missing[1]["missing_marks"] == ["”"]
    assert missing[2]["missing_marks"] == ["』"]


def test_multi_dialogue_option_suppresses_only_consecutive_missing_endings():
    html = (
        '<p>"first paragraph without an ending</p>'
        '<p>"second paragraph without an ending</p>'
        '<p>plain paragraph</p>'
        '<p>"isolated paragraph without an ending</p>'
    )

    assert [
        item["paragraph_index"]
        for item in _missing_ending_quotation_paragraphs(html)
    ] == [1, 2, 4]
    assert [
        item["paragraph_index"]
        for item in _missing_ending_quotation_paragraphs(
            html,
            ignore_consecutive=True,
        )
    ] == [4]


def test_quotation_mismatch_allows_style_changes_and_reports_count_changes():
    source_info = {1: {"quotation_marks": 4}}

    has_mismatch, issues = detect_quotation_mismatch(
        '“double” and 「corner」',
        1,
        source_info,
    )
    assert has_mismatch is False
    assert issues == []

    has_missing, missing_issues = detect_quotation_mismatch('“only one pair”', 1, source_info)
    assert has_missing is True
    assert missing_issues[0]["type"] == "missing_quotation_marks"
    assert missing_issues[0]["difference"] == 2

    has_excess, excess_issues = detect_quotation_mismatch(
        '“one” “two” “three”',
        1,
        source_info,
    )
    assert has_excess is True
    assert excess_issues[0]["type"] == "excess_quotation_marks"
    assert excess_issues[0]["difference"] == 2

    ignored_excess, ignored_issues = detect_quotation_mismatch(
        '“one” “two” “three”',
        1,
        source_info,
        ignore_excess=True,
    )
    assert ignored_excess is False
    assert ignored_issues == []

    missing_still_flagged, _ = detect_quotation_mismatch(
        '“only one pair”',
        1,
        source_info,
        ignore_excess=True,
    )
    assert missing_still_flagged is True


def test_epub_quotation_extraction_decodes_html_character_references(tmp_path):
    epub_path = tmp_path / "quotes.epub"
    content_opf = """<?xml version="1.0" encoding="utf-8"?>
    <package xmlns="http://www.idpf.org/2007/opf" version="3.0">
      <manifest>
        <item id="chapter" href="text/chapter.xhtml" media-type="application/xhtml+xml" />
      </manifest>
      <spine><itemref idref="chapter" /></spine>
    </package>
    """
    chapter = """<html><head><title>&quot;ignored&quot;</title></head><body>
    <h1>&quot;ignored heading&quot;</h1>
    <div>&quot;ignored div&quot;</div>
    &quot;ignored loose text&quot;
    <p>&quot;double&quot; &#39;single&#39; 「corner」</p>
    </body></html>"""

    with zipfile.ZipFile(epub_path, "w") as epub:
        epub.writestr("OEBPS/content.opf", content_opf)
        epub.writestr("OEBPS/text/chapter.xhtml", chapter)

    source_info = extract_epub_quotation_info(epub_path, log=lambda _message: None)

    assert source_info[1]["quotation_marks"] == 12
    assert source_info[1]["filename"] == "chapter.xhtml"


def test_active_qa_output_folder_uses_translated_folder_not_raw_folder(tmp_path, monkeypatch):
    raw_folder = tmp_path / "raw"
    output_root = tmp_path / "translated"
    source_path = raw_folder / "book.epub"
    translated_folder = output_root / "book"
    raw_folder.mkdir()
    translated_folder.mkdir(parents=True)
    source_path.write_bytes(b"source")
    monkeypatch.setenv("EPUB_OUTPUT_DIR", str(translated_folder))

    assert active_qa_output_folder_for_source(source_path) == str(translated_folder.resolve())


def test_report_directory_is_created_inside_scanned_output_folder(tmp_path):
    raw_folder = tmp_path / "raw" / "book"
    translated_folder = tmp_path / "translated" / "book"
    raw_folder.mkdir(parents=True)
    translated_folder.mkdir(parents=True)
    logs = []

    generate_reports(
        [],
        str(translated_folder),
        {},
        log=logs.append,
        qa_settings={"report_format": "summary", "auto_save_report": True},
    )

    expected_report_folder = translated_folder / "book_Scan Report"
    assert (expected_report_folder / "scan_summary.txt").is_file()
    assert not (raw_folder / "book_Scan Report").exists()
    assert any(str(expected_report_folder) in message for message in logs)


@pytest.mark.parametrize(
    "relative_path",
    (
        Path("Direct Text") / "Chat 001",
        Path("direct_text_20260720_000000_abcd1234"),
        Path("glossarion_input_output_abcd") / "book",
        Path("glossarion_direct_text_chat_abcd"),
    ),
)
def test_direct_text_paths_are_excluded_from_qa(tmp_path, relative_path):
    assert is_direct_text_qa_path(tmp_path / relative_path)


def test_regular_output_path_is_not_excluded_from_qa(tmp_path):
    assert not is_direct_text_qa_path(tmp_path / "Translated Books" / "book")


def test_windows_automatic_output_discovery_never_uses_raw_epub_sibling(tmp_path, monkeypatch):
    downloads = tmp_path / "Downloads"
    app_dir = tmp_path / "app" / "src"
    source_path = downloads / "book.epub"
    raw_extraction = downloads / "book"
    translated_output = app_dir / "book"
    raw_extraction.mkdir(parents=True)
    translated_output.mkdir(parents=True)
    source_path.write_bytes(b"source")
    monkeypatch.delenv("EPUB_OUTPUT_DIR", raising=False)

    candidates = automatic_qa_output_candidates(
        source_path,
        current_dir=app_dir,
        script_dir=app_dir,
        platform_name="win32",
    )

    assert candidates[0] == str(translated_output)
    assert str(raw_extraction) not in candidates


def test_macos_automatic_output_discovery_can_use_epub_sibling(tmp_path, monkeypatch):
    source_path = tmp_path / "Downloads" / "book.epub"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"source")
    monkeypatch.delenv("EPUB_OUTPUT_DIR", raising=False)

    candidates = automatic_qa_output_candidates(
        source_path,
        current_dir=tmp_path / "app",
        script_dir=tmp_path / "app" / "src",
        platform_name="darwin",
    )

    assert str(source_path.parent / "book") in candidates


def test_automatic_qa_scan_refuses_direct_text_folder(tmp_path):
    direct_text_folder = tmp_path / "Direct Text" / "Chat 001"
    direct_text_folder.mkdir(parents=True)
    logs = []

    result = run_qa_scan_path(direct_text_folder, log=logs.append)

    assert result is None
    assert any("Direct Text folders" in message for message in logs)
    assert not (direct_text_folder / "Chat 001_Scan Report").exists()


def test_quotation_scan_counts_all_visible_html_text(tmp_path):
    chapter_path = tmp_path / "chapter.html"
    chapter_path.write_text(
        '<h1>&quot;ignored heading&quot;</h1>'
        '&quot;ignored loose text&quot;'
        '<div>&quot;ignored div&quot;</div>'
        '<p class="dialog">&quot;one pair&quot;</p>',
        encoding="utf-8",
    )
    settings = default_qa_scan_settings()
    settings.update({
        "check_quotation_mismatch": True,
        "check_missing_html_tag": False,
        "check_missing_images": False,
        "check_repetition": False,
        "check_translation_artifacts": False,
        "check_ai_artifacts": False,
        "check_glossary_leakage": False,
        "check_word_count_ratio": False,
    })
    source_info = {
        "chapter.html": {
            "question_marks": 0,
            "exclamation_marks": 0,
            "quotation_marks": 8,
            "filename": "chapter.html",
        }
    }

    results = process_html_file_batch((
        [(0, "chapter.html")],
        str(tmp_path),
        settings,
        "quick-scan",
        {},
        {},
        True,
        {},
        {},
        source_info,
    ))

    assert not any("quotation_marks" in issue for issue in results[0]["issues"])


def test_non_epub_plain_text_quotation_scan_needs_no_html_tags(tmp_path):
    chapter_path = tmp_path / "chapter.txt"
    chapter_path.write_text('&quot;one translated pair&quot;', encoding="utf-8")
    settings = default_qa_scan_settings()
    settings.update({
        "check_quotation_mismatch": True,
        "check_missing_html_tag": False,
        "check_missing_images": False,
        "check_repetition": False,
        "check_translation_artifacts": False,
        "check_ai_artifacts": False,
        "check_glossary_leakage": False,
        "check_word_count_ratio": False,
    })
    source_info = {
        "chapter.txt": {
            "quotation_marks": 4,
            "filename": "chapter.txt",
        }
    }

    results = process_html_file_batch((
        [(0, "chapter.txt")],
        str(tmp_path),
        settings,
        "quick-scan",
        {},
        {},
        True,
        {},
        {},
        source_info,
    ))

    assert "quotation_marks_2_missing_(2/4)" in results[0]["issues"]


def test_quotation_scan_flags_missing_curly_quote_ending_per_paragraph(tmp_path):
    chapter_path = tmp_path / "chapter.html"
    chapter_path.write_text('<p>“This is insane.</p>', encoding="utf-8")
    settings = default_qa_scan_settings()
    settings.update({
        "check_quotation_mismatch": True,
        "check_missing_html_tag": False,
        "check_missing_images": False,
        "check_repetition": False,
        "check_translation_artifacts": False,
        "check_ai_artifacts": False,
        "check_glossary_leakage": False,
        "check_word_count_ratio": False,
    })
    source_info = {
        "chapter.html": {
            "quotation_marks": 2,
            "filename": "chapter.html",
        }
    }

    results = process_html_file_batch((
        [(0, "chapter.html")], str(tmp_path), settings, "quick-scan",
        {}, {}, True, {}, {}, source_info,
    ))

    assert "missing_ending_quotation_p1" in results[0]["issues"]
    assert results[0]["qa_issue_previews"][
        "missing_ending_quotation_p1"
    ].endswith("This is insane.")
    assert "quotation_marks_1_missing_(1/2)" in results[0]["issues"]


def test_quotation_scan_can_only_check_incomplete_quotes_without_source_counts(tmp_path):
    chapter_path = tmp_path / "chapter.html"
    chapter_path.write_text(
        '<p>"broken</p><p>""</p><p>[broken bracket</p>',
        encoding="utf-8",
    )
    settings = default_qa_scan_settings()
    settings.update({
        "check_quotation_mismatch": True,
        "only_check_incomplete_quotations": True,
        "include_square_brackets_as_quotations": True,
        "check_missing_html_tag": False,
        "check_missing_images": False,
        "check_repetition": False,
        "check_translation_artifacts": False,
        "check_ai_artifacts": False,
        "check_glossary_leakage": False,
        "check_word_count_ratio": False,
    })

    results = process_html_file_batch((
        [(0, "chapter.html")], str(tmp_path), settings, "quick-scan",
        {}, {}, True, {}, {}, {},
    ))

    assert "missing_ending_quotation_p1" in results[0]["issues"]
    assert "missing_ending_quotation_p3" in results[0]["issues"]
    assert results[0]["qa_issue_previews"] == {
        "missing_ending_quotation_p1": '"broken',
        "missing_ending_quotation_p3": "[broken bracket",
    }
    assert not any("quotation_marks" in issue for issue in results[0]["issues"])


def test_quotation_scan_applies_multi_dialogue_setting_to_reported_issues(tmp_path):
    chapter_path = tmp_path / "chapter.html"
    chapter_path.write_text(
        '<p>"first adjacent opening</p>'
        '<p>"second adjacent opening</p>'
        '<p>plain paragraph</p>'
        '<p>"isolated opening</p>',
        encoding="utf-8",
    )
    settings = default_qa_scan_settings()
    settings.update({
        "check_quotation_mismatch": True,
        "only_check_incomplete_quotations": True,
        "ignore_consecutive_missing_quotations": True,
        "check_missing_html_tag": False,
        "check_missing_images": False,
        "check_repetition": False,
        "check_translation_artifacts": False,
        "check_ai_artifacts": False,
        "check_glossary_leakage": False,
        "check_word_count_ratio": False,
    })

    results = process_html_file_batch((
        [(0, "chapter.html")], str(tmp_path), settings, "quick-scan",
        {}, {}, True, {}, {}, {},
    ))

    assert "missing_ending_quotation_p1" not in results[0]["issues"]
    assert "missing_ending_quotation_p2" not in results[0]["issues"]
    assert "missing_ending_quotation_p4" in results[0]["issues"]
    assert results[0]["qa_issue_previews"] == {
        "missing_ending_quotation_p4": '"isolated opening',
    }


def test_missing_quotation_preview_is_saved_with_progress_entry(tmp_path):
    progress = {
        "chapters": {
            "9": {
                "actual_num": 9,
                "output_file": "chapter0009.xhtml",
                "status": "completed",
            }
        }
    }
    issue_code = "missing_ending_quotation_p125"
    faulty_chapters = [{
        "filename": "chapter0009.xhtml",
        "chapter_num": 9,
        "issues": [issue_code],
        "qa_issue_previews": {
            issue_code: '"This sentence is missing its closing quotation.',
        },
    }]

    update_new_format_progress(
        progress,
        faulty_chapters,
        [],
        lambda _message: None,
        str(tmp_path),
    )

    chapter = progress["chapters"]["9"]
    assert chapter["qa_issues_found"] == [issue_code]
    assert chapter["qa_issue_previews"] == {
        issue_code: '"This sentence is missing its closing quotation.',
    }


def test_quotation_scan_can_skip_stylistic_single_quote_pairs(tmp_path):
    chapter_path = tmp_path / "chapter.html"
    chapter_path.write_text("<p>Use 'Naught' here.</p>", encoding="utf-8")
    settings = default_qa_scan_settings()
    settings.update({
        "check_quotation_mismatch": True,
        "skip_stylistic_single_quotes": True,
        "check_missing_html_tag": False,
        "check_missing_images": False,
        "check_repetition": False,
        "check_translation_artifacts": False,
        "check_ai_artifacts": False,
        "check_glossary_leakage": False,
        "check_word_count_ratio": False,
    })
    source_info = {
        "chapter.html": {
            "quotation_marks": 0,
            "filename": "chapter.html",
        }
    }

    results = process_html_file_batch((
        [(0, "chapter.html")], str(tmp_path), settings, "quick-scan",
        {}, {}, True, {}, {}, source_info,
    ))

    assert not any("quotation" in issue for issue in results[0]["issues"])


def test_quotation_option_does_not_change_punctuation_matching(tmp_path):
    chapter_path = tmp_path / "chapter.html"
    chapter_path.write_text("<html><body><p>plain output</p></body></html>", encoding="utf-8")
    settings = default_qa_scan_settings()
    settings.update({
        "check_punctuation_mismatch": True,
        "punctuation_loss_threshold": 49,
        "check_quotation_mismatch": True,
        "check_missing_html_tag": False,
        "check_missing_images": False,
        "check_repetition": False,
        "check_translation_artifacts": False,
        "check_ai_artifacts": False,
        "check_glossary_leakage": False,
        "check_word_count_ratio": False,
    })
    source_punctuation = {
        "chapter.html": {
            "question_marks": 2,
            "exclamation_marks": 0,
            "filename": "chapter.html",
        }
    }
    source_quotations = {
        "chapter.html": {
            "quotation_marks": 0,
            "filename": "chapter.html",
        }
    }

    results = process_html_file_batch((
        [(0, "chapter.html")],
        str(tmp_path),
        settings,
        "quick-scan",
        {},
        {},
        True,
        {},
        source_punctuation,
        source_quotations,
    ))

    assert "?_punctuation_100%_lost_(0/2)" in results[0]["issues"]
    assert not any("quotation_marks" in issue for issue in results[0]["issues"])


def test_valid_html_tag_entities_preserve_angle_bracket_prose():
    prose = "&lt;A talent possessing both a clean character and noble integrity. Who exactly is Riyan?&gt;"

    assert unescape_valid_html_tag_entities(prose) == prose


def test_html2text_multipass_preserves_entity_angle_prose_starting_with_a():
    extractor = EnhancedTextExtractor()
    text, _, _ = extractor.extract_chapter_content(
        "<html><body><p>&lt;A spatial quake is a simple means.&gt;</p></body></html>",
        "full",
    )

    assert text == "<A spatial quake is a simple means.>"


def test_valid_html_tag_entities_rehydrate_real_markup():
    html = (
        "&lt;p&gt;text&lt;/p&gt;"
        '&lt;a href="chapter.xhtml"&gt;link&lt;/a&gt;'
        '&lt;img src="cover.jpg" /&gt;'
    )

    assert unescape_valid_html_tag_entities(html) == (
        "<p>text</p>"
        '<a href="chapter.xhtml">link</a>'
        '<img src="cover.jpg" />'
    )


def test_xhtml_converter_keeps_escaped_angle_bracket_prose():
    sample = (
        "<p>&lt;A talent possessing both a clean character and noble integrity. "
        "Who exactly is Riyan, the new professor of the Imperial Academy?&gt;</p>"
    )

    converted = epub_converter.XHTMLConverter.ensure_compliance(sample, "Chapter 15")

    assert "&lt;A talent possessing both a clean character and noble integrity." in converted
    assert "Imperial Academy?&gt;" in converted
    assert "<a talent=" not in converted.lower()


def test_xhtml_converter_escapes_raw_angle_bracket_prose():
    sample = (
        "<p><A talent possessing both a clean character and noble integrity. "
        "Who exactly is Riyan, the new professor of the Imperial Academy?></p>"
    )

    converted = epub_converter.XHTMLConverter.ensure_compliance(sample, "Chapter 15")

    assert "&lt;A talent possessing both a clean character and noble integrity." in converted
    assert "Imperial Academy?&gt;" in converted
    assert "<a talent=" not in converted.lower()


def test_xhtml_converter_empty_attr_fix_respects_epub_toggle_off(monkeypatch):
    monkeypatch.setenv("FIX_EMPTY_ATTR_TAGS_EPUB", "0")
    sample = '<p><a talent="" possessing="" both="" /></p>'

    converted = epub_converter.XHTMLConverter.ensure_compliance(sample, "Empty Attr Off")

    assert 'talent=""' in converted
    assert 'possessing=""' in converted
    assert "&lt;a talent possessing both" not in converted


def test_xhtml_converter_empty_attr_fix_respects_epub_toggle_on(monkeypatch):
    monkeypatch.setenv("FIX_EMPTY_ATTR_TAGS_EPUB", "1")
    sample = '<p><a talent="" possessing="" both="" /></p>'

    converted = epub_converter.XHTMLConverter.ensure_compliance(sample, "Empty Attr On")

    assert "&lt;a talent possessing both/&gt;" in converted
    assert 'talent=""' not in converted


def test_html_entity_decoder_encoding_fixes_no_crash():
    mojibake = "Ã¢â‚¬â„¢ and â€¦ and Â©"
    decoded = HTMLEntityDecoder.decode(mojibake)
    # Should replace with reasonable characters and not raise
    assert isinstance(decoded, str) and len(decoded) >= 3


@pytest.mark.parametrize(
    ("source", "expected_src", "expected_alt"),
    [
        (
            '<img alt="”16”" src="”../Images/chapter0007_img_1.webp" width="”100%”"/>',
            "../Images/chapter0007_img_1.webp",
            "16",
        ),
        (
            '<img alt="”17”" src="”../Images/chapter0007_img_2.webp" width="”100%”"/>',
            "../Images/chapter0007_img_2.webp",
            "17",
        ),
    ],
)
def test_epub_image_repair_normalizes_nested_smart_quote_attributes(
    tmp_path, source, expected_src, expected_alt
):
    compiler = EPUBCompiler(str(tmp_path), log_callback=lambda _msg: None)

    repaired = compiler._fix_encoding_issues(source)

    assert f'src="{expected_src}"' in repaired
    assert f'alt="{expected_alt}"' in repaired
    assert 'width="100%"' in repaired
    assert 'src=""' not in repaired


def test_epub_image_repair_preserves_and_resolves_real_image_path(tmp_path):
    compiler = EPUBCompiler(str(tmp_path), log_callback=lambda _msg: None)
    source = '<img alt="”16”" src="”../Images/chapter0007_img_1.webp" width="”100%”"/>'
    repaired = compiler._fix_encoding_issues(source)
    xhtml = epub_converter.XHTMLConverter.ensure_compliance(repaired, "Chapter 7")

    processed, missing = compiler._process_chapter_images(
        xhtml,
        {"chapter0007_img_1.webp": "chapter0007_img_1.webp"},
    )
    validated = epub_converter.XHTMLConverter.validate(processed)

    assert missing == []
    assert 'src="images/chapter0007_img_1.webp"' in validated
    assert '..=""' not in validated


def test_epub_html_discovery_never_uses_unrefined_backup(tmp_path):
    working_name = "response_chapter_notice0002.html"
    working_html = "<html><body><p>MANUAL REFINED COPY</p></body></html>"
    backup_html = "<html><body><p>STALE UNREFINED BACKUP</p></body></html>"
    (tmp_path / working_name).write_text(working_html, encoding="utf-8")
    backup_dir = tmp_path / "unrefined_backup"
    backup_dir.mkdir()
    backup_path = backup_dir / working_name
    backup_path.write_text(backup_html, encoding="utf-8")

    compiler = EPUBCompiler(str(tmp_path), log_callback=lambda _msg: None)

    assert compiler._find_html_files() == [working_name]
    assert epub_converter._is_forbidden_epub_source_path(
        str(backup_path), str(tmp_path)
    )
    assert not epub_converter._is_forbidden_epub_source_path(
        str(tmp_path / working_name), str(tmp_path)
    )


def test_epub_chapter_processing_does_not_rewrite_manual_source_html(tmp_path):
    filename = "response_chapter_notice0002.html"
    source_path = tmp_path / filename
    original = (
        "<!DOCTYPE html><html><head><title>Manual notice</title></head>"
        "<body><h1>Manual notice</h1><p>MANUAL REFINED COPY</p>"
        '<p><img src="images/does-not-exist.webp"/></p></body></html>'
    )
    source_path.write_text(original, encoding="utf-8")

    compiler = EPUBCompiler(str(tmp_path), log_callback=lambda _msg: None)
    compiler.max_workers = 1
    book = epub_converter.epub.EpubBook()
    spine = []
    toc = []

    added = compiler._process_chapters(
        book,
        [filename],
        {0: ("Manual notice", 1.0, "manual")},
        [],
        {},
        spine,
        toc,
        {"language": "en"},
    )

    assert added == 1
    assert source_path.read_text(encoding="utf-8") == original
    packaged = "\n".join(
        item.get_content().decode("utf-8", errors="replace")
        for item in book.get_items_of_type(epub_converter.ITEM_DOCUMENT)
    )
    assert "MANUAL REFINED COPY" in packaged
    assert "does-not-exist.webp" not in packaged


def test_cached_header_translation_remains_authoritative_for_existing_heading(tmp_path, monkeypatch):
    monkeypatch.delenv("BATCH_HEADER_PREPEND_NUMBER_PATTERN", raising=False)
    filename = "response_chapter_notice0002.html"
    html_path = tmp_path / filename
    manual_html = (
        "<html><head><title>My manual notice</title></head>"
        "<body><h1>My manual notice</h1><p>MANUAL BODY EDIT</p></body></html>"
    )
    html_path.write_text(manual_html, encoding="utf-8")
    translator = BatchHeaderTranslator(None, {})

    translator._update_html_headers_exact(
        str(tmp_path),
        {3: "Cached translated notice"},
        {
            3: {
                "title": "My manual notice",
                "source_title": "Original source notice",
                "filename": filename,
            }
        },
    )

    updated = html_path.read_text(encoding="utf-8")
    assert "<h1>Cached translated notice</h1>" in updated
    assert "<p>MANUAL BODY EDIT</p>" in updated


def test_cached_header_translation_can_update_untouched_source_heading(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("BATCH_HEADER_PREPEND_NUMBER_PATTERN", raising=False)
    filename = "response_chapter_notice0002.html"
    html_path = tmp_path / filename
    html_path.write_text(
        "<html><body><h1>Original source notice</h1><p>BODY</p></body></html>",
        encoding="utf-8",
    )
    translator = BatchHeaderTranslator(None, {})

    translator._update_html_headers_exact(
        str(tmp_path),
        {3: "Cached translated notice"},
        {
            3: {
                "title": "Original source notice",
                "source_title": "Original source notice",
                "filename": filename,
            }
        },
    )

    updated = html_path.read_text(encoding="utf-8")
    assert "<h1>Cached translated notice</h1>" in updated
    assert "<p>BODY</p>" in updated


def test_apply_existing_translations_does_not_promote_working_html_header(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("BATCH_HEADER_PREPEND_NUMBER_PATTERN", raising=False)
    filename = "response_chapter0500.html"
    html_path = tmp_path / filename
    html_path.write_text(
        "<html><body><h1>Working HTML heading</h1><p>REFINED BODY</p></body></html>",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        translate_headers_standalone,
        "load_translations_from_file",
        lambda _path, _log=None: (
            {1: "Original source heading"},
            {1: "Authoritative cached heading"},
            {1: "chapter0500"},
        ),
    )
    monkeypatch.setattr(
        translate_headers_standalone,
        "extract_source_chapters_with_opf_mapping",
        lambda _path, _log=None: (
            {"chapter0500": "Original source heading"},
            ["Text/chapter0500.xhtml"],
        ),
    )

    result = translate_headers_standalone.apply_existing_translations(
        "source.epub",
        str(tmp_path),
        str(tmp_path / "translated_headers.txt"),
        update_html=True,
        log_callback=lambda _message: None,
    )

    updated = html_path.read_text(encoding="utf-8")
    assert "<h1>Authoritative cached heading</h1>" in updated
    assert "<p>REFINED BODY</p>" in updated
    assert result[filename] == "Authoritative cached heading"


def test_xml_validator_valid_codepoints():
    # Basic BMP and some punctuation
    assert XMLValidator.is_valid_char_code(ord('A')) is True
    # Some punctuation may be filtered based on implementation; ensure it doesn't raise and returns a bool
    res = XMLValidator.is_valid_char_code(0x2019)
    assert isinstance(res, bool)


def test_sanitize_filename_handles_windows_only_invalid_titles():
    assert FileUtils.sanitize_filename("My Book.", allow_unicode=True) == "My Book"
    assert FileUtils.sanitize_filename("My Book   ", allow_unicode=True) == "My Book"
    assert FileUtils.sanitize_filename("AUX", allow_unicode=True) == "AUX_"
    assert FileUtils.sanitize_filename("CON.txt", allow_unicode=True) == "CON_.txt"


def test_sanitize_filename_for_windows_path_shortens_long_titles(tmp_path):
    max_path = len(os.path.abspath(tmp_path)) + 1 + 20 + len(".epub")

    safe_title = FileUtils.sanitize_filename_for_windows_path(
        "A" * 80,
        str(tmp_path),
        extension=".epub",
        allow_unicode=True,
        max_path=max_path,
    )

    assert safe_title == "A" * 20
    assert len(os.path.join(str(tmp_path), f"{safe_title}.epub")) <= max_path


def test_epub_writer_renames_windows_invalid_and_too_long_title(tmp_path, monkeypatch):
    compiler = EPUBCompiler(str(tmp_path), log_callback=lambda _msg: None)
    max_path = len(os.path.abspath(tmp_path)) + 1 + 18 + len(".epub")
    captured = {}

    def fake_write_epub(out_path, _book, _opts):
        captured["out_path"] = out_path
        Path(out_path).write_bytes(b"epub")

    monkeypatch.setattr(FileUtils, "WINDOWS_MAX_PATH", max_path)
    monkeypatch.setattr(epub_converter.epub, "write_epub", fake_write_epub)
    monkeypatch.setattr(epub_converter, "_replace_organized_library_epub", lambda *_args: None)

    class Book:
        title = f"{'A' * 80}."

    compiler._write_epub(Book(), {})

    output_path = captured["out_path"]
    output_name = os.path.basename(output_path)
    output_stem = os.path.splitext(output_name)[0]
    assert output_name.endswith(".epub")
    assert not output_stem.endswith(".")
    assert len(output_path) <= max_path


def _remote_test_png_bytes(color=(30, 80, 140, 255)):
    output = io.BytesIO()
    Image.new('RGBA', (3, 2), color).save(output, format='PNG')
    return output.getvalue()


def test_remote_raster_bytes_are_converted_to_real_png():
    jpeg = io.BytesIO()
    Image.new('RGB', (4, 3), (140, 80, 30)).save(jpeg, format='JPEG')

    converted = chapter_extractor._convert_remote_image_to_png(jpeg.getvalue())

    assert converted.startswith(b'\x89PNG\r\n\x1a\n')
    with Image.open(io.BytesIO(converted)) as image:
        assert image.format == 'PNG'
        assert image.size == (4, 3)


def test_remote_image_start_throttle_spaces_request_starts():
    clock = [100.0]
    sleeps = []

    def fake_monotonic():
        return clock[0]

    def fake_sleep(delay):
        sleeps.append(delay)
        clock[0] += delay

    throttle = chapter_extractor._RemoteImageStartThrottle(
        0.5,
        monotonic=fake_monotonic,
        sleeper=fake_sleep,
    )

    assert throttle.wait() == 0.0
    assert throttle.wait() == pytest.approx(0.5)
    assert throttle.wait() == pytest.approx(0.5)
    assert sleeps == pytest.approx([0.5, 0.5])


def test_remote_images_are_localized_once_before_chapter_rename(monkeypatch, tmp_path):
    remote_url = (
        'https://images.novelpia.com/imagebox/b1/'
        'b1b11a46e497175bfdc6278959170d99_1958056_1779373634_ori.file'
    )
    markup = f'''<html><body>
        <img class="remote-image" src="{remote_url}"/>
        <svg><image href="{remote_url}"/></svg>
        <object data="{remote_url}"></object>
        <video poster="{remote_url}"></video>
        <div style="background-image: url('{remote_url}')"></div>
    </body></html>'''
    chapters = [{
        'num': 1,
        'title': 'Chapter 1',
        'filename': 'chapter0001.xhtml',
        'original_basename': 'chapter0001',
        'body': markup,
        'original_html': markup,
    }]
    saved_html = tmp_path / 'chapter0001.xhtml'
    saved_html.write_text(markup, encoding='utf-8')

    calls = []

    def fake_download(url):
        calls.append(url)
        return _remote_test_png_bytes()

    monkeypatch.setattr(
        chapter_extractor,
        '_download_remote_image_as_png',
        fake_download,
    )

    localized = chapter_extractor._localize_remote_images(chapters, str(tmp_path))

    digest = hashlib.sha256(remote_url.encode('utf-8')).hexdigest()[:20]
    temporary_name = f'remote_{digest}.png'
    temporary_ref = f'images/{temporary_name}'
    temporary_path = tmp_path / 'images' / temporary_name
    assert calls == [remote_url]
    assert temporary_path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert remote_url not in localized[0]['body']
    assert localized[0]['body'].count(temporary_ref) == 5
    assert remote_url not in localized[0]['original_html']
    assert remote_url not in saved_html.read_text(encoding='utf-8')

    renamed = chapter_extractor._rename_images_to_chapter_format(
        localized,
        str(tmp_path),
    )

    final_name = 'chapter0001_img_1.png'
    assert not temporary_path.exists()
    assert (tmp_path / 'images' / final_name).is_file()
    assert renamed[0]['body'].count(f'images/{final_name}') == 5
    assert remote_url not in renamed[0]['body']
    rename_map = json.loads(
        (tmp_path / 'image_rename_map.json').read_text(encoding='utf-8')
    )
    assert rename_map == {temporary_name: final_name}
    progress_manifest = json.loads((
        tmp_path
        / 'images'
        / '.cache'
        / 'remote_image_download_progress.json'
    ).read_text(encoding='utf-8'))
    assert progress_manifest['items'][0]['filename'] == final_name
    assert progress_manifest['items'][0]['local_reference'] == (
        f'images/{final_name}'
    )


def test_failed_remote_image_download_keeps_original_url(monkeypatch, tmp_path):
    remote_url = 'https://images.example.invalid/blocked.file'
    chapters = [{
        'num': 2,
        'body': f'<img src="{remote_url}">',
    }]

    def fail_download(_url):
        raise OSError('download failed')

    monkeypatch.setattr(
        chapter_extractor,
        '_download_remote_image_as_png',
        fail_download,
    )

    localized = chapter_extractor._localize_remote_images(chapters, str(tmp_path))

    assert remote_url in localized[0]['body']
    assert not list((tmp_path / 'images').glob('*.png'))


def test_remote_image_download_reports_counted_progress(monkeypatch, tmp_path):
    successful_url = 'https://images.example.test/success.file'
    failed_url = 'https://images.example.test/failure.file'
    chapters = [{
        'num': 3,
        'body': (
            f'<img src="{successful_url}">'
            f'<img src="{failed_url}">'
        ),
    }]
    progress_messages = []

    def fake_download(url):
        if url == failed_url:
            raise OSError('simulated failure')
        return _remote_test_png_bytes()

    monkeypatch.setattr(
        chapter_extractor,
        '_download_remote_image_as_png',
        fake_download,
    )

    chapter_extractor._localize_remote_images(
        chapters,
        str(tmp_path),
        progress_callback=progress_messages.append,
    )

    assert progress_messages[0].startswith(
        'Downloading remote images: 0/2 (0%) | 0 saved, 0 failed'
    )
    assert any(
        'Downloading remote images: 1/2 (50%)' in message
        for message in progress_messages
    )
    assert any(
        'Downloading remote images: 2/2 (100%) | 1 saved, 1 failed'
        in message
        for message in progress_messages
    )
    assert any('images/s' in message and 'ETA ' in message
               for message in progress_messages)
    assert any(
        'Warning: remote image download failed; keeping original URL'
        in message
        for message in progress_messages
    )
    assert progress_messages[-1].startswith(
        'Remote image localization complete: 1/2 saved, 1 failed'
    )

    progress_path = (
        tmp_path
        / 'images'
        / '.cache'
        / 'remote_image_download_progress.json'
    )
    manifest = json.loads(progress_path.read_text(encoding='utf-8'))
    assert manifest['status'] == 'completed_with_errors'
    assert manifest['output_format'] == 'png'
    assert manifest['total'] == 2
    assert manifest['completed'] == 2
    assert manifest['successful'] == 1
    assert manifest['failed'] == 1
    assert manifest['progress_percent'] == 100
    items = {item['url']: item for item in manifest['items']}
    assert items[successful_url]['status'] == 'completed'
    assert items[successful_url]['filename'].endswith('.png')
    assert items[failed_url]['status'] == 'failed'
    assert items[failed_url]['error'] == 'simulated failure'


def test_remote_image_progress_cache_resumes_completed_png(monkeypatch, tmp_path):
    remote_url = 'https://images.example.test/resumable.file'

    monkeypatch.setattr(
        chapter_extractor,
        '_download_remote_image_as_png',
        lambda _url: _remote_test_png_bytes(),
    )
    chapter_extractor._localize_remote_images(
        [{'num': 4, 'body': f'<img src="{remote_url}">'}],
        str(tmp_path),
    )

    def unexpected_download(_url):
        raise AssertionError('completed cached PNG should be reused')

    monkeypatch.setattr(
        chapter_extractor,
        '_download_remote_image_as_png',
        unexpected_download,
    )
    fresh_chapters = [{'num': 4, 'body': f'<img src="{remote_url}">'}]
    localized = chapter_extractor._localize_remote_images(
        fresh_chapters,
        str(tmp_path),
    )

    manifest = json.loads((
        tmp_path
        / 'images'
        / '.cache'
        / 'remote_image_download_progress.json'
    ).read_text(encoding='utf-8'))
    assert manifest['status'] == 'completed'
    assert manifest['completed'] == 1
    assert manifest['successful'] == 1
    assert manifest['resumed'] == 1
    assert remote_url not in localized[0]['body']


def test_remote_image_progress_cache_is_excluded_from_epub_sources(tmp_path):
    cache_file = (
        tmp_path
        / 'images'
        / '.cache'
        / 'remote_image_download_progress.json'
    )
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text('{}', encoding='utf-8')

    assert epub_converter._is_forbidden_epub_source_path(
        str(cache_file),
        str(tmp_path),
    )


def test_async_remote_image_progress_keeps_label_and_one_percent_cadence():
    manager_source = (
        Path(__file__).resolve().parents[1]
        / 'src'
        / 'chapter_extraction_manager.py'
    ).read_text(encoding='utf-8')

    assert 'prefix = "🌐 Remote image URL progress"' in manager_source
    assert 'if prog_type == "remote_images":' in manager_source
    assert 'should_show = percent > last_percent' in manager_source
    assert 'formatted_message += f" {detail}"' in manager_source
