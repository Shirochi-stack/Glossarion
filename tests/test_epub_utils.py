import ast
import hashlib
import io
import json
import os
import re
import zipfile
from pathlib import Path

import pytest
from PIL import Image

import Chapter_Extractor as chapter_extractor
import epub_converter
import translate_headers_standalone
from QA_Scanner_GUI import _normalize_qa_dialog_path, _wrapped_tooltip_html
from enhanced_text_extractor import EnhancedTextExtractor
from epub_converter import EPUBCompiler, FileUtils, HTMLEntityDecoder, XMLValidator
from epub_package import find_epub_opf_member, find_opf_path
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
    _format_ai_truncation_last_p_preview,
    _count_quotation_marks,
    _html_preservation_profile,
    _missing_ending_quotation_paragraphs,
    _preservation_count_issues,
    _record_ai_truncation_issue,
    build_pdf_qa_source_aliases,
    cross_reference_word_counts,
    detect_missing_images,
    detect_quotation_mismatch,
    extract_epub_punctuation_info,
    extract_epub_quotation_info,
    extract_html_word_counts,
    generate_reports,
    has_repeating_sentences,
    process_html_file_batch,
    run_ai_truncation_check,
    scan_html_folder,
    update_new_format_progress,
)


class _AITruncationYesClient:
    def send(self, messages, temperature=0.0, max_tokens=None, context=None):
        return "YES"


def test_ai_truncation_issue_previews_source_and_output_last_nonempty_html_p():
    ai_result = run_ai_truncation_check(
        "<html><body><p>Earlier source.</p><p>Source <em>final</em> paragraph.</p><p></p></body></html>",
        "<html><body><p>Earlier output.</p><p>Output <strong>cut off</strong></p><p>&nbsp;</p></body></html>",
        client=_AITruncationYesClient(),
        log=lambda _message: None,
    )

    assert ai_result["flagged"] is True
    assert ai_result["details"] == "ai_verdict=YES"
    assert ai_result["source_last_p"] == "Source final paragraph."
    assert ai_result["output_last_p"] == "Output cut off"

    row = {"issues": [], "qa_issue_previews": {}, "score": 0}
    issue_code = _record_ai_truncation_issue(row, ai_result)

    assert row["issues"] == [issue_code]
    assert row["qa_issue_previews"][issue_code] == (
        "Source last <p>: Source final paragraph. | "
        "Output last <p>: Output cut off"
    )
    assert row["score"] == 1


def test_ai_truncation_last_p_preview_distinguishes_empty_and_missing_tags():
    assert _format_ai_truncation_last_p_preview("", None) == (
        "Source last <p>: [empty] | Output last <p>: [no <p> tag]"
    )


def test_progress_display_keeps_both_ai_truncation_paragraph_previews():
    from Retranslation_GUI import _format_qa_issue_for_progress_display

    issue_code = "ai_truncation_detected (ai_verdict=YES)"
    preview = _format_ai_truncation_last_p_preview(
        "source ending " * 20,
        "output ending " * 20,
    )

    display = _format_qa_issue_for_progress_display(
        issue_code,
        {issue_code: preview},
    )

    assert "Source last <p>:" in display
    assert "Output last <p>:" in display
    assert len(display) > 160


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


def test_quotation_option_tooltips_use_bounded_rich_text():
    tooltip = _wrapped_tooltip_html("First line\nUse <marks> & quotes.", width=320)
    assert tooltip.startswith("<qt>")
    assert "white-space: normal" in tooltip
    assert "width: 320px" in tooltip
    assert "First line<br>Use &lt;marks&gt; &amp; quotes." in tooltip

    source_path = Path(__file__).resolve().parents[1] / "src" / "QA_Scanner_GUI.py"
    source = source_path.read_text(encoding="utf-8")
    for checkbox_name in (
        "check_quotation_checkbox",
        "ignore_excess_quotation_checkbox",
        "only_check_incomplete_quotations_checkbox",
        "ignore_consecutive_quotations_checkbox",
        "skip_stylistic_single_quotes_checkbox",
        "include_square_brackets_checkbox",
    ):
        assert f"{checkbox_name}.setToolTip(_wrapped_tooltip_html(" in source


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


def test_pdf_qa_source_aliases_use_progress_and_normalize_section_names(tmp_path):
    workspace = tmp_path / "translated"
    word_count = workspace / "word_count"
    word_count.mkdir(parents=True)
    (word_count / "pdf_section_1.html").write_text("one", encoding="utf-8")
    (word_count / "pdf_section_2.html").write_text("two", encoding="utf-8")
    (workspace / "response_pdf_section_stable-bookmark-id.html").write_text(
        "one", encoding="utf-8"
    )
    (workspace / "response_pdf_section_002.html").write_text(
        "two", encoding="utf-8"
    )
    (workspace / "translation_progress.json").write_text(
        json.dumps({
            "chapters": {
                "pdf:stable-bookmark-id": {
                    "actual_num": 1,
                    "output_file": "response_pdf_section_stable-bookmark-id.html",
                    "pdf_toc_section": True,
                    "pdf_section_id": "stable-bookmark-id",
                },
                "pdf:second-section": {
                    "actual_num": 2,
                    "output_file": "response_pdf_section_002.html",
                    "pdf_toc_section": True,
                    "pdf_section_id": "second-section",
                },
            }
        }),
        encoding="utf-8",
    )

    aliases = build_pdf_qa_source_aliases(str(workspace))

    assert aliases["pdf_section_stable-bookmark-id.html"] == "pdf_section_1.html"
    assert aliases["pdf_section_002.html"] == "pdf_section_2.html"


def test_missing_image_check_detects_replacement_and_honors_rename_map():
    source_info = {
        "pdf_section_1.html": {
            "image_count": 1,
            "image_srcs": ["images/pdfimg_hash.png"],
        }
    }

    changed, changed_issues = detect_missing_images(
        '<p><img src="images/unrelated.png"></p>',
        "pdf_section_1.html",
        source_info,
    )
    assert changed is True
    assert changed_issues[0]["type"] == "changed_image_references"
    assert changed_issues[0]["missing_srcs"] == ["pdfimg_hash.png"]
    assert changed_issues[0]["unexpected_srcs"] == ["unrelated.png"]

    renamed, renamed_issues = detect_missing_images(
        '<p><img src="images/pdf_section_1_img_1.png"></p>',
        "pdf_section_1.html",
        source_info,
        image_rename_map={"pdfimg_hash.png": "pdf_section_1_img_1.png"},
    )
    assert renamed is False
    assert renamed_issues == []


def test_svg_graphic_reference_check_honors_image_rename_map():
    rename_map = {"kuchie-002.jpg": "p-fmatter-002_img_1.jpg"}
    source = (
        '<html><body><svg><image xlink:href="../image/kuchie-002.jpg"/>'
        '</svg></body></html>'
    )
    output = (
        '<html><body><svg><image '
        'xlink:href="../image/p-fmatter-002_img_1.jpg"/></svg></body></html>'
    )

    source_profile = _html_preservation_profile(source, rename_map)
    output_profile = _html_preservation_profile(output, rename_map)

    assert source_profile["graphic_refs"] == output_profile["graphic_refs"]
    assert not any(
        issue.startswith("changed_graphic_references_")
        for issue, _preview in _preservation_count_issues(
            source_profile,
            output_profile,
        )
    )


def test_pdf_bookmark_qa_preserves_structure_without_exact_sdlxliff_name(tmp_path):
    source_pdf = tmp_path / "book.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    workspace = tmp_path / "book"
    word_count = workspace / "word_count"
    sidecars = workspace / "SDLXLIFF"
    word_count.mkdir(parents=True)
    sidecars.mkdir()

    (word_count / "pdf_section_1.html").write_text(
        """<html><body>
        <h1>Chapter One</h1><h2>Legitimate subsection</h2>
        <p>First source paragraph.</p><p>Second source paragraph.</p>
        <a href="https://example.test/original">source link</a>
        <table><tbody><tr><td>source cell</td></tr></tbody></table>
        <svg><path d="M0 0"></path></svg>
        </body></html>""",
        encoding="utf-8",
    )
    output_name = "response_pdf_section_001.html"
    (workspace / output_name).write_text(
        """<html><body>
        <h1>Chapter One</h1><h2>Legitimate subsection</h2>
        <p>Only one translated paragraph remains.</p>
        <a href="https://example.test/changed">translated link</a>
        </body></html>""",
        encoding="utf-8",
    )
    # A stale pre-rename sidecar must not make this check silently disappear.
    (sidecars / "response_pdf_section_old-hash.html.sdlxliff").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
        <xliff version="1.2"><file original="pdf_section_1.html"><body>
        <trans-unit id="1"><source>&lt;p&gt;old&lt;/p&gt;</source>
        <target>&lt;p&gt;old&lt;/p&gt;</target></trans-unit>
        </body></file></xliff>""",
        encoding="utf-8",
    )
    (workspace / "translation_progress.json").write_text(
        json.dumps({
            "chapters": {
                "pdf:bookmark-one": {
                    "actual_num": 1,
                    "output_file": output_name,
                    "original_basename": "pdf_section_1.html",
                    "pdf_toc_section": True,
                    "pdf_section_id": "bookmark-one",
                    "status": "completed",
                }
            }
        }),
        encoding="utf-8",
    )

    settings = default_qa_scan_settings()
    settings.update({
        "check_word_count_ratio": False,
        "check_missing_images": True,
        "check_missing_beautifulsoup_tags": True,
        "sdlxliff_tag_retention_threshold": 1.0,
        "sdlxliff_tag_surplus_tolerance": 0.0,
        "sdlxliff_min_source_paragraph_tags": 0,
        "check_multiple_headers": True,
        "check_punctuation_mismatch": False,
        "check_quotation_mismatch": False,
        "check_silent_truncation": False,
        "check_ai_truncation_detection": False,
        "check_repetition": False,
        "check_translation_artifacts": False,
        "check_glossary_leakage": False,
        "use_thread_executor": True,
        "source_language": "english",
        "target_language": "english",
    })

    scan_html_folder(
        str(workspace),
        log=lambda _message: None,
        mode="quick-scan",
        qa_settings=settings,
        epub_path=str(source_pdf),
    )

    report_path = workspace / f"{workspace.name}_Scan Report" / "validation_results.json"
    results = json.loads(report_path.read_text(encoding="utf-8"))
    translated = next(row for row in results if row["filename"] == output_name)
    issues = translated["issues"]

    assert "multiple_headers_2_found" not in issues
    assert "missing_tags: 4/3 (-1)" in issues
    assert "changed_link_targets_1_found" in issues
    assert "missing_table_elements_table_1_lost_(0/1)" in issues
    assert "missing_table_elements_tr_1_lost_(0/1)" in issues
    assert "missing_table_elements_td_1_lost_(0/1)" in issues
    assert "missing_graphics_svg_1_lost_(0/1)" in issues
    assert "missing_graphics_path_1_lost_(0/1)" in issues


@pytest.mark.parametrize(
    ("use_word_count", "exact_char_count"),
    [("1", "0"), ("0", "1"), ("0", "0")],
    ids=("word", "exact-character", "sampled-character"),
)
def test_pdf_qa_scan_pairs_padded_output_with_raw_bookmark_chunk(
    tmp_path, monkeypatch, use_word_count, exact_char_count
):
    monkeypatch.setenv("QA_USE_WORD_COUNT", use_word_count)
    monkeypatch.setenv("QA_EXACT_CHAR_COUNT", exact_char_count)
    source_pdf = tmp_path / "book.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    workspace = tmp_path / "book"
    word_count = workspace / "word_count"
    word_count.mkdir(parents=True)
    (word_count / "pdf_section_1.html").write_text(
        "<html><body><h1>Chapter One</h1><p>one two</p></body></html>",
        encoding="utf-8",
    )
    (workspace / "response_pdf_section_001.html").write_text(
        "<html><body><p>Chapter One one two</p></body></html>",
        encoding="utf-8",
    )
    (workspace / "translation_progress.json").write_text(
        json.dumps({
            "chapters": {
                "pdf:bookmark-one": {
                    "actual_num": 1,
                    "output_file": "response_pdf_section_001.html",
                    "pdf_toc_section": True,
                    "pdf_section_id": "bookmark-one",
                    "status": "completed",
                }
            }
        }),
        encoding="utf-8",
    )
    settings = default_qa_scan_settings()
    settings.update({
        "check_word_count_ratio": True,
        "min_duplicate_word_count": 0,
        "check_missing_header_tags": True,
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
    })

    scan_html_folder(
        str(workspace),
        log=lambda _message: None,
        mode="quick-scan",
        qa_settings=settings,
        epub_path=str(source_pdf),
    )

    report_path = workspace / f"{workspace.name}_Scan Report" / "validation_results.json"
    results = json.loads(report_path.read_text(encoding="utf-8"))
    translated_result = next(
        row for row in results if row["filename"] == "response_pdf_section_001.html"
    )
    assert translated_result["word_count_check"]["found_match"] is True
    assert translated_result["word_count_check"]["ratio"] == 1.0
    assert "missing_header_tags" in translated_result["issues"]


def test_repetition_check_allows_same_repetition_count_in_cjk_source():
    source_sentence = "これは作者が意図して何度も繰り返している十分に長い原文の一文です"
    translated_sentence = (
        "This is an intentionally repeated translated sentence with enough "
        "content to qualify for repetition checking"
    )

    source_text = "。".join([source_sentence] * 10) + "。"
    translated_text = ". ".join([translated_sentence] * 10) + "."

    assert has_repeating_sentences(translated_text, source_text=source_text) is False


def test_repetition_check_flags_repetition_beyond_source_count():
    source_sentence = "これは作者が意図して何度も繰り返している十分に長い原文の一文です"
    translated_sentence = (
        "This is an intentionally repeated translated sentence with enough "
        "content to qualify for repetition checking"
    )

    source_text = "。".join([source_sentence] * 10) + "。"
    translated_text = ". ".join([translated_sentence] * 11) + "."

    assert has_repeating_sentences(translated_text, source_text=source_text) is True


def test_qa_scan_uses_standalone_source_html_for_repetition_allowance(tmp_path):
    source_sentence = "これは作者が意図して何度も繰り返している十分に長い原文の一文です。"
    translated_sentence = (
        "This is an intentionally repeated translated sentence with enough "
        "content to qualify for repetition checking."
    )
    source_path = tmp_path / "chapter0003.html"
    source_path.write_text(
        "<html><body>" + "".join(f"<p>{source_sentence}</p>" for _ in range(10)) + "</body></html>",
        encoding="utf-8",
    )
    output_dir = tmp_path / "translated"
    output_dir.mkdir()
    (output_dir / "response_chapter0003.xhtml").write_text(
        "<html><body>" + "".join(f"<p>{translated_sentence}</p>" for _ in range(10)) + "</body></html>",
        encoding="utf-8",
    )
    settings = default_qa_scan_settings()
    settings.update({
        "check_repetition": True,
        "check_word_count_ratio": False,
        "check_missing_images": False,
        "check_punctuation_mismatch": False,
        "check_quotation_mismatch": False,
        "check_silent_truncation": False,
        "check_ai_truncation_detection": False,
        "check_multiple_headers": False,
        "check_translation_artifacts": False,
        "check_glossary_leakage": False,
        "use_thread_executor": True,
    })

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
        row for row in results if row["filename"] == "response_chapter0003.xhtml"
    )
    assert "excessive_repetition" not in translated_result["issues"]


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


def test_qa_failure_removes_existing_refinement_status(tmp_path):
    progress = {
        "chapters": {
            "9": {
                "actual_num": 9,
                "output_file": "chapter0009.xhtml",
                "status": "completed",
                "refinement_status": "refined",
                "refined_at": 123.0,
                "refinement_error": "stale error",
                "unrefined_backup_file": (
                    "_unrefined/chapter0009.xhtml"
                ),
                "previous_progress_entry": {
                    "status": "completed",
                    "refinement_status": "refined",
                    "refined_at": 122.0,
                    "unrefined_backup_file": (
                        "_unrefined/older_chapter0009.xhtml"
                    ),
                },
            }
        }
    }

    update_new_format_progress(
        progress,
        [{
            "filename": "chapter0009.xhtml",
            "chapter_num": 9,
            "issues": ["Chinese_text_found_2_chars_[失败]"],
        }],
        [],
        lambda _message: None,
        str(tmp_path),
    )

    chapter = progress["chapters"]["9"]
    assert chapter["status"] == "qa_failed"
    for field in (
        "refinement_status",
        "refined_at",
        "refinement_error",
        "unrefined_backup_file",
    ):
        assert field not in chapter
        assert field not in chapter["previous_progress_entry"]


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


def test_valid_html_tag_entities_rehydrate_ruby_base_markup():
    html = (
        "&lt;ruby&gt;&lt;rb&gt;Tomoki&lt;/rb&gt;"
        "&lt;rt&gt;tomoki&lt;/rt&gt;&lt;rtc&gt;"
        "&lt;rt&gt;reading&lt;/rt&gt;&lt;/rtc&gt;&lt;/ruby&gt;"
    )

    assert unescape_valid_html_tag_entities(html) == (
        "<ruby><rb>Tomoki</rb><rt>tomoki</rt>"
        "<rtc><rt>reading</rt></rtc></ruby>"
    )


def test_valid_html_tag_entities_rehydrate_svg_image_link():
    html = (
        '&lt;svg xmlns:xlink="http://www.w3.org/1999/xlink"&gt;'
        '&lt;image height="2560" width="1804" '
        'xlink:href="../Images/cover_img_1.jpg" /&gt;'
        '&lt;/svg&gt;'
    )

    assert unescape_valid_html_tag_entities(html) == (
        '<svg xmlns:xlink="http://www.w3.org/1999/xlink">'
        '<image height="2560" width="1804" '
        'xlink:href="../Images/cover_img_1.jpg" />'
        '</svg>'
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


def test_partial_opf_numerically_inserts_unmapped_chapters(tmp_path):
    (tmp_path / "content.opf").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <manifest>
    <item id="cover" href="Text/cover.xhtml" media-type="application/xhtml+xml"/>
    <item id="c3000" href="Text/chapter3000.xhtml" media-type="application/xhtml+xml"/>
    <item id="c3107" href="Text/chapter3107.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="cover"/>
    <itemref idref="c3000"/>
    <itemref idref="c3107"/>
  </spine>
</package>
""",
        encoding="utf-8",
    )
    for name in (
        "response_cover.html",
        "chapter3000.xhtml",
        "chapter3107.xhtml",
        "response_chapter0001.html",
        "response_chapter2999.html",
        "bonus.xhtml",
    ):
        (tmp_path / name).write_text(
            f"<html><body><p>{name}</p></body></html>",
            encoding="utf-8",
        )

    compiler = EPUBCompiler(str(tmp_path), log_callback=lambda _msg: None)

    assert compiler._find_html_files() == [
        "response_cover.html",
        "response_chapter0001.html",
        "response_chapter2999.html",
        "chapter3000.xhtml",
        "chapter3107.xhtml",
        "bonus.xhtml",
    ]


def test_non_spine_special_html_is_included_by_default_and_optionally_skipped(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "content.opf").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <manifest>
    <item id="c1" href="Text/chapter0001.xhtml"
          media-type="application/xhtml+xml"/>
    <item id="notice" href="Text/chapter_notice0003.xhtml"
          media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="c1"/></spine>
</package>
""",
        encoding="utf-8",
    )
    for name in (
        "response_chapter0001.html",
        "response_chapter_notice0003.html",
    ):
        (tmp_path / name).write_text(
            f"<html><body><p>{name}</p></body></html>",
            encoding="utf-8",
        )

    logs = []
    compiler = EPUBCompiler(str(tmp_path), log_callback=logs.append)
    monkeypatch.setenv("SPECIAL_FILE_KEYWORDS", "notice")
    monkeypatch.delenv("SKIP_NON_SPINE_SPECIAL_FILES", raising=False)

    assert compiler._find_html_files() == [
        "response_chapter0001.html",
        "response_chapter_notice0003.html",
    ]
    assert not any("Skipping non-spine special file" in log for log in logs)

    logs.clear()
    monkeypatch.setenv("SKIP_NON_SPINE_SPECIAL_FILES", "1")
    assert compiler._find_html_files() == ["response_chapter0001.html"]
    assert any(
        "Skipping non-spine special file: response_chapter_notice0003.html"
        in log
        for log in logs
    )


def test_unreferenced_epub_image_filter_is_opt_in_and_preserves_cover(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "response_chapter0001.html").write_text(
        '<html><body><img src="images/used.jpg"/></body></html>',
        encoding="utf-8",
    )
    (tmp_path / "css").mkdir()
    (tmp_path / "css" / "book.css").write_text(
        ".ornament { background-image: url('../images/css-only.png'); }",
        encoding="utf-8",
    )
    processed = {
        "used.jpg": "used.jpg",
        "css-only.png": "css-only.png",
        "unused.jpg": "unused.jpg",
        "cover.jpg": "cover.jpg",
    }
    logs = []
    compiler = EPUBCompiler(str(tmp_path), log_callback=logs.append)
    monkeypatch.delenv("SKIP_UNREFERENCED_EPUB_IMAGES", raising=False)

    assert compiler._filter_embedded_images_for_ocr(
        processed,
        None,
        protected_images=("cover.jpg",),
    ) == list(processed.items())

    monkeypatch.setenv("SKIP_UNREFERENCED_EPUB_IMAGES", "1")
    assert compiler._filter_embedded_images_for_ocr(
        processed,
        None,
        protected_images=("cover.jpg",),
    ) == [
        ("used.jpg", "used.jpg"),
        ("css-only.png", "css-only.png"),
        ("cover.jpg", "cover.jpg"),
    ]
    assert compiler._filter_gallery_images_for_ocr(
        processed,
        "cover.jpg",
    ) == ["used.jpg", "css-only.png"]
    assert any(
        "skipped 1 unreferenced image(s)" in log
        for log in logs
    )


def test_epub_optional_filter_settings_default_off_and_propagate():
    root = Path(__file__).resolve().parents[1]
    settings_source = (root / "src" / "other_settings.py").read_text(
        encoding="utf-8",
    )
    gui_source = (root / "src" / "translator_gui.py").read_text(
        encoding="utf-8",
    )
    async_source = (root / "src" / "async_api_processor.py").read_text(
        encoding="utf-8",
    )

    assert '"Skip Non-Spine Special Files in EPUB"' in settings_source
    assert (
        "self.config.get('skip_non_spine_special_files', False)"
        in settings_source
    )
    assert '"Skip Unreferenced Images in EPUB"' in settings_source
    assert (
        "self.config.get('skip_unreferenced_epub_images', False)"
        in settings_source
    )
    assert (
        "('skip_non_spine_special_files_var', "
        "'skip_non_spine_special_files', False)"
        in gui_source
    )
    assert (
        "('skip_unreferenced_epub_images_var', "
        "'skip_unreferenced_epub_images', False)"
        in gui_source
    )
    assert "'SKIP_NON_SPINE_SPECIAL_FILES'" in async_source
    assert "'SKIP_UNREFERENCED_EPUB_IMAGES'" in async_source
    assert (
        settings_source.index('"Disable Image Gallery in EPUB"')
        < settings_source.index('"Disable Automatic Cover Creation"')
        < settings_source.index('"Skip Non-Spine Special Files in EPUB"')
        < settings_source.index('"Skip Unreferenced Images in EPUB"')
    )


def test_other_settings_epub_and_resume_tooltips_are_bounded_rich_text():
    source_path = Path(__file__).resolve().parents[1] / "src" / "other_settings.py"
    source = source_path.read_text(encoding="utf-8")

    assert "def _wrapped_tooltip_html(text, width=430):" in source
    assert "white-space: normal" in source
    for checkbox_name in (
        "chunk_progress_cb",
        "numbered_html_cb",
        "gallery_cb",
        "cover_cb",
        "skip_non_spine_cb",
        "skip_unreferenced_images_cb",
    ):
        match = re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(checkbox_name)}\.setToolTip\(",
            source,
        )
        assert match is not None
        call_start = match.start()
        assert "_wrapped_tooltip_html(" in source[call_start:call_start + 120]

    for technical_marker in (
        "API-cached input-token budget fingerprints",
        "SPECIAL_FILE_EXACT",
        "synthetic gallery XHTML item",
        "heuristic cover selection",
        "source OPF spine",
        "stylesheets, and image rename mappings",
    ):
        assert technical_marker in source


def test_lightweight_thinking_slider_ignores_mouse_wheel():
    source_path = Path(__file__).resolve().parents[1] / "src" / "other_settings.py"
    source = source_path.read_text(encoding="utf-8")

    slider_start = source.index('think_slider = QSlider(Qt.Horizontal)')
    slider_end = source.index("gemini_levels =", slider_start)
    slider_setup = source[slider_start:slider_end]

    assert "think_slider.wheelEvent = lambda event: event.ignore()" in slider_setup


@pytest.mark.parametrize(
    ("first_html_has_image", "expected_html"),
    [
        (True, "response_front.html"),
        (False, "response_chapter.html"),
    ],
)
def test_disabled_cover_fallback_selects_first_image_bearing_opf_html(
    tmp_path,
    monkeypatch,
    first_html_has_image,
    expected_html,
):
    monkeypatch.setenv("DISABLE_AUTOMATIC_COVER_CREATION", "1")
    monkeypatch.setenv("EXTRACTION_WORKERS", "1")
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "cover.jpg").write_bytes(b"cover bytes")

    # Put the later document first in the manifest to prove that OPF spine
    # order, rather than manifest order or directory order, is authoritative.
    (tmp_path / "content.opf").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <manifest>
    <item id="chapter" href="Text/chapter.xhtml" media-type="application/xhtml+xml"/>
    <item id="front" href="Text/front.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="front"/>
    <itemref idref="chapter"/>
  </spine>
</package>
""",
        encoding="utf-8",
    )
    front_body = (
        '&lt;img src="images/cover.jpg" alt="Cover" /&gt;'
        if first_html_has_image
        else "<p>Front matter without an image.</p>"
    )
    chapter_body = (
        "<p>Chapter text.</p>"
        if first_html_has_image
        else '<img src="images/cover.jpg" alt="Later image" />'
    )
    (tmp_path / "response_front.html").write_text(
        f"<html><body>{front_body}</body></html>",
        encoding="utf-8",
    )
    (tmp_path / "response_chapter.html").write_text(
        f"<html><body>{chapter_body}</body></html>",
        encoding="utf-8",
    )

    logs = []
    compiler = EPUBCompiler(str(tmp_path), log_callback=logs.append)
    html_files = ["response_front.html", "response_chapter.html"]
    fallback_html = compiler._first_opf_spine_html_with_image(
        html_files
    )
    processed_images, cover_file = compiler._process_images(
        skip_automatic_cover=bool(fallback_html)
    )

    assert fallback_html == expected_html
    assert processed_images == {"cover.jpg": "cover.jpg"}
    assert cover_file is None
    assert any(
        "first image-bearing HTML accepted in OPF reading order" in log
        for log in logs
    )


def test_cover_logs_unresolved_designation_before_automatic_fallback(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("DISABLE_AUTOMATIC_COVER_CREATION", "0")
    monkeypatch.setenv("EXTRACTION_WORKERS", "1")
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "chapter001.jpg").write_bytes(b"image bytes")

    logs = []
    compiler = EPUBCompiler(str(tmp_path), log_callback=logs.append)
    _processed, cover_file = compiler._process_images(
        preferred_cover_names=["missing-cover.jpg"],
    )

    assert cover_file == "chapter001.jpg"
    assert any(
        "designated cover image reference(s) could not be resolved" in log
        for log in logs
    )
    assert any("attempting automatic image selection" in log for log in logs)
    assert any("Using first image" in log for log in logs)


def test_disabled_cover_fallback_scans_workspace_html_without_opf(tmp_path):
    (tmp_path / "response_001.html").write_text(
        "<html><body><p>Text only.</p></body></html>",
        encoding="utf-8",
    )
    (tmp_path / "response_002.html").write_text(
        '<html><body><img src="images/illustration.jpg"/></body></html>',
        encoding="utf-8",
    )

    logs = []
    compiler = EPUBCompiler(str(tmp_path), log_callback=logs.append)

    assert compiler._first_opf_spine_html_with_image(
        ["response_001.html", "response_002.html"]
    ) == "response_002.html"
    assert any(
        "first image-bearing HTML accepted: response_002.html" in log
        for log in logs
    )


@pytest.mark.parametrize("disable_automatic_cover", ["0", "1"])
def test_opf_cover_image_and_svg_cover_page_override_automatic_setting(
    tmp_path,
    monkeypatch,
    disable_automatic_cover,
):
    monkeypatch.setenv(
        "DISABLE_AUTOMATIC_COVER_CREATION",
        disable_automatic_cover,
    )
    monkeypatch.setenv("EXTRACTION_WORKERS", "1")
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "p-cover_img_1.jpg").write_bytes(b"cover bytes")
    (tmp_path / "images" / "p-000_img_1.jpg").write_bytes(b"chapter bytes")
    (tmp_path / "container.xml").write_text(
        """<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="item/standard.opf"
              media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
""",
        encoding="utf-8",
    )
    (tmp_path / "standard.opf").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <manifest>
    <item id="cover" href="image/cover.jpg" media-type="image/jpeg"
          properties="cover-image"/>
    <item id="p-cover" href="xhtml/p-cover.xhtml"
          media-type="application/xhtml+xml" properties="svg"/>
    <item id="p-000" href="xhtml/p-000.xhtml"
          media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="p-cover"/>
    <itemref idref="p-000"/>
  </spine>
</package>
""",
        encoding="utf-8",
    )
    (tmp_path / "image_rename_map.json").write_text(
        json.dumps({"cover.jpg": "p-cover_img_1.jpg"}),
        encoding="utf-8",
    )
    (tmp_path / "response_p-cover.html").write_text(
        """<html xmlns:epub="http://www.idpf.org/2007/ops">
<body epub:type="cover"><svg><image
xlink:href="../image/p-cover_img_1.jpg"/></svg></body></html>""",
        encoding="utf-8",
    )
    (tmp_path / "response_p-000.html").write_text(
        '<html><body><img src="images/p-000_img_1.jpg"/></body></html>',
        encoding="utf-8",
    )

    compiler = EPUBCompiler(str(tmp_path), log_callback=lambda _msg: None)
    html_files = compiler._find_html_files()
    designation = compiler._get_opf_cover_designation()
    existing_page = compiler._find_existing_cover_html(html_files, designation)
    processed, cover_file = compiler._process_images(
        preferred_cover_names=[designation["image_href"]],
    )

    assert compiler._find_opf_path() == str(tmp_path / "standard.opf")
    assert designation["image_href"] == "image/cover.jpg"
    assert designation["method"] == "EPUB 3 cover-image property"
    assert existing_page == "response_p-cover.html"
    assert processed["p-cover_img_1.jpg"] == "p-cover_img_1.jpg"
    assert cover_file == "p-cover_img_1.jpg"

    filename_map = compiler._build_opf_filename_map()
    assert compiler._restore_opf_filename(
        "response_p-cover.html",
        filename_map,
    ) == "p-cover.xhtml"
    assert compiler._restore_opf_filename(
        "response_p-000.html",
        filename_map,
    ) == "p-000.xhtml"

    # Verify the restoration is applied to the actual EpubHtml item, not only
    # exposed as a cosmetic workspace-name helper.
    compiler._opf_filename_map = filename_map
    book = epub_converter.epub.EpubBook()
    spine = []
    toc = []
    assert compiler._process_chapters(
        book,
        ["response_p-000.html"],
        {0: ("Chapter", 1.0, "test")},
        [],
        processed,
        spine,
        toc,
        {"language": "en"},
    ) == 1
    assert [item.file_name for item in spine] == ["p-000.xhtml"]


def test_cover_html_fallback_accepts_cover_substring_only_with_image(tmp_path):
    (tmp_path / "response_cover-notes.html").write_text(
        "<html><body><p>Not a cover image.</p></body></html>",
        encoding="utf-8",
    )
    (tmp_path / "response_p-cover.xhtml").write_text(
        '<html><body><img src="images/cover.jpg"/></body></html>',
        encoding="utf-8",
    )
    compiler = EPUBCompiler(str(tmp_path), log_callback=lambda _msg: None)

    assert compiler._find_existing_cover_html(
        ["response_cover-notes.html", "response_p-cover.xhtml"],
        {},
    ) == "response_p-cover.xhtml"


def test_epub2_cover_metadata_resolves_manifest_image(tmp_path):
    (tmp_path / "content.opf").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
  <metadata><meta name="cover" content="cover-art"/></metadata>
  <manifest>
    <item id="cover-art" href="Images/book-art.jpg" media-type="image/jpeg"/>
  </manifest>
</package>
""",
        encoding="utf-8",
    )
    compiler = EPUBCompiler(str(tmp_path), log_callback=lambda _msg: None)

    designation = compiler._get_opf_cover_designation()

    assert designation["image_id"] == "cover-art"
    assert designation["image_href"] == "Images/book-art.jpg"
    assert designation["method"] == "EPUB 2 cover metadata"


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


def test_load_translations_excludes_entries_marked_translation_failed(tmp_path):
    translations_path = tmp_path / "translated_headers.txt"
    translations_path.write_text(
        """Chapter 1:
  Original:   第14章 苍山庶家
  Translated: 第14章 苍山庶家
  Output File: chapter0389
  Status:     ⚠️ Using original (translation failed)
----------------------------------------
Chapter 2:
  Original:   第15章 烧，烧，烧
  Translated: Chapter 15: Burn, Burn, Burn
  Output File: chapter0390
----------------------------------------
""",
        encoding="utf-8",
    )

    source_headers, translated_headers, output_files = (
        translate_headers_standalone.load_translations_from_file(
            str(translations_path), log_callback=lambda _message: None
        )
    )

    assert source_headers == {
        1: "第14章 苍山庶家",
        2: "第15章 烧，烧，烧",
    }
    assert translated_headers == {2: "Chapter 15: Burn, Burn, Burn"}
    assert output_files == {1: "chapter0389", 2: "chapter0390"}


def test_failed_cached_header_cannot_overwrite_corrected_html(tmp_path, monkeypatch):
    monkeypatch.delenv("BATCH_HEADER_PREPEND_NUMBER_PATTERN", raising=False)
    filename = "response_chapter0389.html"
    html_path = tmp_path / filename
    corrected_html = (
        "<html><body><h1>Chapter 14: Cangshan Gu Tomb</h1>"
        "<p>REFINED BODY</p></body></html>"
    )
    html_path.write_text(corrected_html, encoding="utf-8")

    translations_path = tmp_path / "translated_headers.txt"
    translations_path.write_text(
        """Chapter 1:
  Original:   第14章 苍山庶家
  Translated: 第14章 苍山庶家
  Output File: chapter0389
  Status:     ⚠️ Using original (translation failed)
----------------------------------------
""",
        encoding="utf-8",
    )

    def unexpected_epub_read(*_args, **_kwargs):
        raise AssertionError("An all-failed cache should not reach EPUB matching")

    monkeypatch.setattr(
        translate_headers_standalone,
        "extract_source_chapters_with_opf_mapping",
        unexpected_epub_read,
    )

    result = translate_headers_standalone.apply_existing_translations(
        "source.epub",
        str(tmp_path),
        str(translations_path),
        update_html=True,
        log_callback=lambda _message: None,
    )

    assert result == {}
    assert html_path.read_text(encoding="utf-8") == corrected_html


def test_failed_header_cache_entries_are_retranslated_and_replaced(tmp_path):
    translations_path = tmp_path / "translated_headers.txt"
    translations_path.write_text(
        """Chapter Header Translations
==================================================

Chapter 1:
  Original:   第1章 雪夜
  Translated: Chapter 1: Snowy Night
  Output File: chapter0001
----------------------------------------
Chapter 2:
  Original:   第2章 龙战于野
  Translated: 第2章 龙战于野
  Output File: chapter0002
  Status:     ⚠️ Using original (translation failed)
----------------------------------------

Summary:
Total chapters: 2
Successfully translated: 1
Failed chapters: 2
""",
        encoding="utf-8",
    )

    class RecordingTranslator:
        def __init__(self):
            self.calls = []

        def translate_headers_batch(
            self, headers_dict, batch_size=None, translation_type="header"
        ):
            self.calls.append((dict(headers_dict), batch_size, translation_type))
            return {2: "Chapter 2: Dragons Fight in the Wild"}

    translator = RecordingTranslator()
    source, translated, outputs = (
        translate_headers_standalone.retry_failed_header_translations(
            str(translations_path),
            translator=translator,
            batch_size=25,
            log_callback=lambda _message: None,
        )
    )

    assert translator.calls == [
        ({2: "第2章 龙战于野"}, 25, "header")
    ]
    assert source == {1: "第1章 雪夜", 2: "第2章 龙战于野"}
    assert translated == {
        1: "Chapter 1: Snowy Night",
        2: "Chapter 2: Dragons Fight in the Wild",
    }
    assert outputs == {1: "chapter0001", 2: "chapter0002"}

    rewritten = translations_path.read_text(encoding="utf-8")
    assert "translation failed" not in rewritten
    assert "Failed chapters:" not in rewritten
    assert rewritten.count("Chapter 2:\n") == 1
    assert "Successfully translated: 2" in rewritten


def test_failed_header_retry_uses_three_attempts_and_only_resends_pending(tmp_path):
    translations_path = tmp_path / "translated_headers.txt"
    translations_path.write_text(
        """Chapter 1:
  Original:   One
  Translated: One
  Status:     ⚠️ Using original (translation failed)
----------------------------------------
Chapter 2:
  Original:   Two
  Translated: Two
  Status:     ⚠️ Using original (translation failed)
----------------------------------------
Chapter 3:
  Original:   Three
  Translated: Three
  Status:     ⚠️ Using original (translation failed)
----------------------------------------
""",
        encoding="utf-8",
    )

    class OneAtATimeTranslator:
        def __init__(self):
            self.calls = []

        def translate_headers_batch(
            self, entries, batch_size=None, translation_type="header"
        ):
            self.calls.append(dict(entries))
            number = min(entries)
            return {number: f"Translated {entries[number]}"}

    translator = OneAtATimeTranslator()
    _, translated, _ = (
        translate_headers_standalone.retry_failed_header_translations(
            str(translations_path),
            translator=translator,
            max_retry_attempts=3,
            log_callback=lambda _message: None,
        )
    )

    assert translator.calls == [
        {1: "One", 2: "Two", 3: "Three"},
        {2: "Two", 3: "Three"},
        {3: "Three"},
    ]
    assert translated == {
        1: "Translated One",
        2: "Translated Two",
        3: "Translated Three",
    }
    assert "translation failed" not in translations_path.read_text(
        encoding="utf-8"
    )


def test_failed_translation_retry_attempts_default_and_bounds(monkeypatch):
    monkeypatch.delenv("FAILED_TRANSLATION_RETRY_ATTEMPTS", raising=False)
    assert translate_headers_standalone.get_failed_translation_retry_attempts() == 3

    monkeypatch.setenv("FAILED_TRANSLATION_RETRY_ATTEMPTS", "7")
    assert translate_headers_standalone.get_failed_translation_retry_attempts() == 7
    assert translate_headers_standalone.get_failed_translation_retry_attempts(-2) == 0
    assert translate_headers_standalone.get_failed_translation_retry_attempts(999) == 20
    assert translate_headers_standalone.get_failed_translation_retry_attempts("bad") == 3


def test_new_header_cache_retries_partial_failures_on_same_run(tmp_path, monkeypatch):
    (tmp_path / "response_chapter0001.html").write_text(
        "<html><body><h1>原一</h1></body></html>", encoding="utf-8"
    )
    (tmp_path / "response_chapter0002.html").write_text(
        "<html><body><h1>原二</h1></body></html>", encoding="utf-8"
    )
    monkeypatch.setattr(
        translate_headers_standalone,
        "extract_source_chapters_with_opf_mapping",
        lambda _path, _log=None: (
            {"chapter0001": "原一", "chapter0002": "原二"},
            ["Text/chapter0001.xhtml", "Text/chapter0002.xhtml"],
        ),
    )
    monkeypatch.setattr(
        translate_headers_standalone,
        "match_output_to_source_chapters",
        lambda *_args, **_kwargs: {
            "response_chapter0001.html": (
                "原一",
                "原一",
                "response_chapter0001.html",
            ),
            "response_chapter0002.html": (
                "原二",
                "原二",
                "response_chapter0002.html",
            ),
        },
    )

    class PartialThenRecoveredTranslator:
        instances = []

        def __init__(self, _client, _config):
            self.retry_calls = []
            self.update_calls = []
            self.stop_flag = False
            self.__class__.instances.append(self)

        def translate_and_save_headers(self, **_kwargs):
            return {1: "Header One"}

        def _save_translations_to_file(
            self, original, translated, output_path, current_titles
        ):
            outputs = {
                num: Path(info["filename"]).stem.removeprefix("response_")
                for num, info in current_titles.items()
            }
            translate_headers_standalone._write_header_translation_cache(
                output_path, original, translated, outputs
            )

        def translate_headers_batch(
            self, headers_dict, batch_size=None, translation_type="header"
        ):
            self.retry_calls.append(
                (dict(headers_dict), batch_size, translation_type)
            )
            return {2: "Header Two"}

        def _update_html_headers_exact(
            self, _output_dir, translated, _current_titles
        ):
            self.update_calls.append(dict(translated))

        def set_stop_flag(self, value):
            self.stop_flag = value

    monkeypatch.setattr(
        "metadata_batch_translator.BatchHeaderTranslator",
        PartialThenRecoveredTranslator,
    )

    result = translate_headers_standalone.translate_headers_standalone(
        epub_path="source.epub",
        output_dir=str(tmp_path),
        api_client=object(),
        config={"headers_per_batch": 20},
        update_html=True,
        save_to_file=True,
        log_callback=lambda _message: None,
    )

    translator = PartialThenRecoveredTranslator.instances[-1]
    assert translator.retry_calls == [({2: "原二"}, 20, "header")]
    assert translator.update_calls == [{2: "Header Two"}]
    assert result == {
        "response_chapter0001.html": "Header One",
        "response_chapter0002.html": "Header Two",
    }
    _, cached, _ = translate_headers_standalone.load_translations_from_file(
        str(tmp_path / "translated_headers.txt"),
        log_callback=lambda _message: None,
    )
    assert cached == {1: "Header One", 2: "Header Two"}


def test_standalone_header_progress_uses_current_output_directory(
    tmp_path, monkeypatch
):
    from unified_api_client import set_current_thread_actual_request_model

    current_output = tmp_path / "current-book"
    stale_output = tmp_path / "previous-book"
    current_output.mkdir()
    stale_output.mkdir()
    (current_output / "response_chapter0001.html").write_text(
        "<html><body><h1>Original One</h1></body></html>",
        encoding="utf-8",
    )

    current_progress_path = current_output / "translation_progress.json"
    current_progress_path.write_text(
        json.dumps({
            "version": "2.1",
            "chapters": {
                "__translation_artifact__:headers": {
                    "status": "pending",
                    "model_name": "old-main-model",
                }
            },
        }),
        encoding="utf-8",
    )
    stale_progress_path = stale_output / "translation_progress.json"
    stale_progress = {
        "version": "2.1",
        "chapters": {"sentinel": {"status": "completed"}},
    }
    stale_progress_path.write_text(json.dumps(stale_progress), encoding="utf-8")

    monkeypatch.setattr(
        translate_headers_standalone,
        "extract_source_chapters_with_opf_mapping",
        lambda _path, _log=None: (
            {"chapter0001": "Original One"},
            ["Text/chapter0001.xhtml"],
        ),
    )
    monkeypatch.setattr(
        translate_headers_standalone,
        "match_output_to_source_chapters",
        lambda *_args, **_kwargs: {
            "response_chapter0001.html": (
                "Original One",
                "Original One",
                "response_chapter0001.html",
            )
        },
    )

    class Client:
        output_dir = str(stale_output)
        model = "main-key-model"

    def send_with_metadata_key(_translator, **kwargs):
        queued = json.loads(current_progress_path.read_text(encoding="utf-8"))
        queued_entry = queued["chapters"]["__translation_artifact__:headers"]
        assert queued_entry["status"] == "in_progress"
        assert "model_name" not in queued_entry

        set_current_thread_actual_request_model(
            "metadata-key-model", "MetadataKey#1 (metadata-key-model)"
        )
        kwargs["before_send_callback"]()
        live = json.loads(current_progress_path.read_text(encoding="utf-8"))
        live_entry = live["chapters"]["__translation_artifact__:headers"]
        assert live_entry["status"] == "in_progress"
        assert live_entry["model_name"] == "metadata-key-model"
        return '{"1": "Translated One"}'

    monkeypatch.setattr(
        BatchHeaderTranslator, "_send_with_retry", send_with_metadata_key
    )

    result = translate_headers_standalone.translate_headers_standalone(
        epub_path="source.epub",
        output_dir=str(current_output),
        api_client=Client(),
        config={"headers_per_batch": 1},
        update_html=True,
        save_to_file=True,
        log_callback=lambda _message: None,
    )

    assert result == {"response_chapter0001.html": "Translated One"}
    current_progress = json.loads(
        current_progress_path.read_text(encoding="utf-8")
    )
    current_entry = current_progress["chapters"][
        "__translation_artifact__:headers"
    ]
    assert current_entry["status"] == "completed"
    assert current_entry["model_name"] == "metadata-key-model"
    assert json.loads(stale_progress_path.read_text(encoding="utf-8")) == stale_progress


def test_standalone_headers_without_source_header_tags_complete_as_noop(
    tmp_path,
):
    source_epub = tmp_path / "no-headers.epub"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    container_xml = """<?xml version="1.0" encoding="UTF-8"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OPS/content.opf"
              media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""
    content_opf = """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <manifest>
    <item id="chapter-1" href="chapter0001.xhtml"
          media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="chapter-1"/></spine>
</package>
"""
    with zipfile.ZipFile(source_epub, "w") as archive:
        archive.writestr("META-INF/container.xml", container_xml)
        archive.writestr("OPS/content.opf", content_opf)
        archive.writestr(
            "OPS/chapter0001.xhtml",
            "<html><body><p>This chapter has no heading tags.</p></body></html>",
        )

    progress_path = output_dir / "translation_progress.json"
    progress_path.write_text(
        json.dumps({
            "version": "2.1",
            "chapters": {
                "__translation_artifact__:headers": {
                    "status": "pending",
                    "model_name": "old-model",
                }
            },
        }),
        encoding="utf-8",
    )
    logs = []

    result = translate_headers_standalone.translate_headers_standalone(
        epub_path=str(source_epub),
        output_dir=str(output_dir),
        api_client=object(),
        config={},
        log_callback=logs.append,
    )

    assert result == {}
    assert bool(result) is True
    assert result.successful_noop is True
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    entry = progress["chapters"]["__translation_artifact__:headers"]
    assert entry["status"] == "completed"
    assert entry["model_name"] == "No Header Tags Founds"
    assert "error_message" not in entry
    assert any("No source header tags found" in message for message in logs)


def test_epub_compile_accepts_live_client_from_standalone_rebuild(
    tmp_path, monkeypatch
):
    client = object()
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)

    compiler = EPUBCompiler(
        str(tmp_path),
        log_callback=lambda _message: None,
        api_client=client,
    )

    assert compiler.api_client is client

    captured = {}

    class RecordingCompiler:
        def __init__(self, base_dir, log_callback=None, api_client=None):
            captured.update({
                "base_dir": base_dir,
                "log_callback": log_callback,
                "api_client": api_client,
            })

        def compile(self):
            return "rebuilt.epub"

    log_callback = lambda _message: None
    monkeypatch.setattr(epub_converter, "EPUBCompiler", RecordingCompiler)

    assert epub_converter.compile_epub(
        str(tmp_path),
        log_callback=log_callback,
        api_client=client,
    ) == "rebuilt.epub"
    assert captured == {
        "base_dir": str(tmp_path),
        "log_callback": log_callback,
        "api_client": client,
    }


def test_failed_toc_cache_entries_are_retranslated_and_replaced(tmp_path):
    toc_path = tmp_path / "TOC.txt"
    toc_path.write_text(
        """TOC Translations
==================================================

Chapter 1:
  Original:   夜雪篇
  Translated: Night Snow Arc
  Target URI: Section0001.xhtml
----------------------------------------
Chapter 2:
  Original:   第1章 出门半步即江湖
  Translated: 第1章 出门半步即江湖
  Target URI: Chapter0001.xhtml
  Status:     ⚠️ Using original (translation failed)
----------------------------------------
""",
        encoding="utf-8",
    )

    class RecordingTocTranslator:
        def __init__(self):
            self.calls = []

        def translate_headers_batch(
            self, entries, batch_size=None, translation_type="toc"
        ):
            self.calls.append((dict(entries), batch_size, translation_type))
            return {2: "Chapter 1: Half a Step Into the Jianghu"}

    translator = RecordingTocTranslator()
    compiler = EPUBCompiler(str(tmp_path), log_callback=lambda _message: None)

    source, translated, outputs = compiler._retry_failed_toc_translations(
        str(toc_path), translator=translator
    )

    assert translator.calls == [
        ({2: "第1章 出门半步即江湖"}, None, "toc")
    ]
    assert source == {1: "夜雪篇", 2: "第1章 出门半步即江湖"}
    assert translated == {
        1: "Night Snow Arc",
        2: "Chapter 1: Half a Step Into the Jianghu",
    }
    assert outputs == {
        1: "Section0001.xhtml",
        2: "Chapter0001.xhtml",
    }
    rewritten = toc_path.read_text(encoding="utf-8")
    assert "translation failed" not in rewritten
    assert rewritten.count("Chapter 2:\n") == 1
    assert "Successfully translated: 2" in rewritten


def test_failed_toc_retry_respects_configured_attempt_limit(tmp_path):
    toc_path = tmp_path / "TOC.txt"
    toc_path.write_text(
        """Chapter 1:
  Original:   Still Failed
  Translated: Still Failed
  Status:     ⚠️ Using original (translation failed)
----------------------------------------
""",
        encoding="utf-8",
    )

    class AlwaysFailingTranslator:
        def __init__(self):
            self.calls = []

        def translate_headers_batch(
            self, entries, batch_size=None, translation_type="toc"
        ):
            self.calls.append(dict(entries))
            return {}

    translator = AlwaysFailingTranslator()
    compiler = EPUBCompiler(str(tmp_path), log_callback=lambda _message: None)
    _, translated, _ = compiler._retry_failed_toc_translations(
        str(toc_path),
        translator=translator,
        max_retry_attempts=2,
    )

    assert translator.calls == [{1: "Still Failed"}, {1: "Still Failed"}]
    assert translated == {}
    assert "translation failed" in toc_path.read_text(encoding="utf-8")


def test_new_toc_cache_retries_partial_failures_on_same_run(tmp_path, monkeypatch):
    source_epub = tmp_path / "source.epub"
    source_epub.write_bytes(b"placeholder")
    (tmp_path / "Chapter0001.xhtml").write_text(
        "<html><body><h1>One</h1></body></html>", encoding="utf-8"
    )
    (tmp_path / "Chapter0002.xhtml").write_text(
        "<html><body><h1>Two</h1></body></html>", encoding="utf-8"
    )

    class PartialThenRecoveredTocTranslator:
        instances = []

        def __init__(self, _client, _config):
            self.calls = []
            self.config = dict(_config)
            self.__class__.instances.append(self)

        def translate_headers_batch(
            self, entries, batch_size=None, translation_type="toc"
        ):
            self.calls.append((dict(entries), batch_size, translation_type))
            if len(self.calls) == 1:
                return {1: "TOC One"}
            return {2: "TOC Two"}

    monkeypatch.setattr(
        "metadata_batch_translator.BatchHeaderTranslator",
        PartialThenRecoveredTocTranslator,
    )
    monkeypatch.setenv("EPUB_PATH", str(source_epub))
    monkeypatch.setenv("TOC_NCX_PER_BATCH", "10")

    compiler = EPUBCompiler(str(tmp_path), log_callback=lambda _message: None)
    compiler.translate_toc_ncx = True
    compiler.api_client = object()
    monkeypatch.setattr(
        compiler,
        "_extract_source_toc_ncx_entries",
        lambda _path: [
            {"label": "原一", "src": "Chapter0001.xhtml"},
            {"label": "原二", "src": "Chapter0002.xhtml"},
        ],
    )
    monkeypatch.setattr(compiler, "_get_chapter_order_from_opf", lambda: {})
    spine = [
        epub_converter.epub.EpubHtml(
            title="One", file_name="Chapter0001.xhtml"
        ),
        epub_converter.epub.EpubHtml(
            title="Two", file_name="Chapter0002.xhtml"
        ),
    ]

    toc = compiler._build_toc_from_source_toc_ncx(spine, [], {})

    translator = PartialThenRecoveredTocTranslator.instances[-1]
    assert translator.config["output_dir"] == str(tmp_path)
    assert translator.calls == [
        ({1: "原一", 2: "原二"}, 10, "toc"),
        ({2: "原二"}, 10, "toc"),
    ]
    assert [(item.href, item.title) for item in toc] == [
        ("Chapter0001.xhtml", "TOC One"),
        ("Chapter0002.xhtml", "TOC Two"),
    ]
    _, cached, _ = compiler._load_toc_translations_file(
        str(tmp_path / "TOC.txt")
    )
    assert cached == {1: "TOC One", 2: "TOC Two"}


def test_failed_header_cache_is_not_reused_for_toc_translation(tmp_path):
    headers_path = tmp_path / "translated_headers.txt"
    headers_path.write_text(
        """Chapter 1:
  Original:   第14章 苍山庶家
  Translated: 第14章 苍山庶家
  Output File: chapter0389
  Status:     ⚠️ Using original (translation failed)
----------------------------------------
""",
        encoding="utf-8",
    )
    compiler = EPUBCompiler(str(tmp_path), log_callback=lambda _message: None)

    reused, remaining = compiler._cross_reference_from_other_file(
        {8: "第14章 苍山庶家"}, str(headers_path), "toc"
    )

    assert reused == {}
    assert remaining == {8: "第14章 苍山庶家"}


def test_fully_recycled_toc_is_completed_with_recycled_model(
    tmp_path, monkeypatch
):
    source_epub = tmp_path / "source.epub"
    source_epub.write_bytes(b"placeholder")
    monkeypatch.setenv("EPUB_PATH", str(source_epub))
    for filename in ("Chapter0001.xhtml", "Chapter0002.xhtml"):
        (tmp_path / filename).write_text(
            "<html><body><p>chapter</p></body></html>", encoding="utf-8"
        )

    (tmp_path / "translated_headers.txt").write_text(
        """Chapter Header Translations
==================================================

Chapter 1:
  Original:   Original One
  Translated: Translated One
  Output File: Chapter0001
----------------------------------------
Chapter 2:
  Original:   Original Two
  Translated: Translated Two
  Output File: Chapter0002
----------------------------------------
""",
        encoding="utf-8",
    )
    progress_path = tmp_path / "translation_progress.json"
    progress_path.write_text(
        json.dumps({
            "version": "2.1",
            "chapters": {
                "__translation_artifact__:toc": {
                    "status": "in_progress",
                    "model_name": "main-key-model",
                    "output_file": "TOC.txt",
                }
            },
        }),
        encoding="utf-8",
    )

    compiler = EPUBCompiler(str(tmp_path), log_callback=lambda _message: None)
    compiler.translate_toc_ncx = True
    compiler.api_client = object()
    monkeypatch.setattr(
        compiler,
        "_extract_source_toc_ncx_entries",
        lambda _path: [
            {"label": "Original One", "src": "Chapter0001.xhtml"},
            {"label": "Original Two", "src": "Chapter0002.xhtml"},
        ],
    )
    monkeypatch.setattr(compiler, "_get_chapter_order_from_opf", lambda: {})
    monkeypatch.setattr(
        BatchHeaderTranslator,
        "translate_headers_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fully recycled TOC must not make an API call")
        ),
    )
    spine = [
        epub_converter.epub.EpubHtml(
            title="One", file_name="Chapter0001.xhtml"
        ),
        epub_converter.epub.EpubHtml(
            title="Two", file_name="Chapter0002.xhtml"
        ),
    ]

    toc = compiler._build_toc_from_source_toc_ncx(spine, [], {})

    assert [item.title for item in toc] == ["Translated One", "Translated Two"]
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    entry = progress["chapters"]["__translation_artifact__:toc"]
    assert entry["status"] == "completed"
    assert entry["model_name"] == "RECYCLED"


def test_stopped_partial_toc_reuse_preserves_recycled_model(
    tmp_path, monkeypatch
):
    source_epub = tmp_path / "source.epub"
    source_epub.write_bytes(b"placeholder")
    monkeypatch.setenv("EPUB_PATH", str(source_epub))
    monkeypatch.setenv("FAILED_TRANSLATION_RETRY_ATTEMPTS", "0")
    monkeypatch.setenv("TRANSLATION_CANCELLED", "1")
    for filename in ("Chapter0001.xhtml", "Chapter0002.xhtml"):
        (tmp_path / filename).write_text(
            "<html><body><p>chapter</p></body></html>", encoding="utf-8"
        )

    (tmp_path / "translated_headers.txt").write_text(
        """Chapter Header Translations
==================================================

Chapter 1:
  Original:   Original One
  Translated: Translated One
  Output File: Chapter0001
----------------------------------------
Chapter 2:
  Original:   Original Two
  Translated: Original Two
  Output File: Chapter0002
  Status:     ⚠️ Using original (translation failed)
----------------------------------------
""",
        encoding="utf-8",
    )
    progress_path = tmp_path / "translation_progress.json"
    progress_path.write_text(
        json.dumps({
            "version": "2.1",
            "chapters": {
                "__translation_artifact__:toc": {
                    "status": "pending",
                    "output_file": "TOC.txt",
                }
            },
        }),
        encoding="utf-8",
    )

    compiler = EPUBCompiler(str(tmp_path), log_callback=lambda _message: None)
    compiler.translate_toc_ncx = True
    compiler.api_client = object()
    monkeypatch.setattr(compiler, "is_stopped", lambda: False)
    monkeypatch.setattr(
        compiler,
        "_extract_source_toc_ncx_entries",
        lambda _path: [
            {"label": "Original One", "src": "Chapter0001.xhtml"},
            {"label": "Original Two", "src": "Chapter0002.xhtml"},
        ],
    )
    monkeypatch.setattr(compiler, "_get_chapter_order_from_opf", lambda: {})
    monkeypatch.setattr(
        BatchHeaderTranslator,
        "translate_headers_batch",
        lambda *_args, **_kwargs: {},
    )
    spine = [
        epub_converter.epub.EpubHtml(
            title="One", file_name="Chapter0001.xhtml"
        ),
        epub_converter.epub.EpubHtml(
            title="Two", file_name="Chapter0002.xhtml"
        ),
    ]

    toc = compiler._build_toc_from_source_toc_ncx(spine, [], {})

    assert [item.title for item in toc] == ["Translated One", "Original Two"]
    _, cached, _ = compiler._load_toc_translations_file(
        str(tmp_path / "TOC.txt")
    )
    assert cached == {1: "Translated One"}
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    entry = progress["chapters"]["__translation_artifact__:toc"]
    assert entry["status"] == "pending"
    assert entry["model_name"] == "RECYCLED"


def test_stopped_toc_retry_reuse_replaces_failed_api_model(
    tmp_path, monkeypatch
):
    source_epub = tmp_path / "source.epub"
    source_epub.write_bytes(b"placeholder")
    monkeypatch.setenv("EPUB_PATH", str(source_epub))
    monkeypatch.setenv("FAILED_TRANSLATION_RETRY_ATTEMPTS", "0")
    monkeypatch.setenv("TRANSLATION_CANCELLED", "1")
    for filename in ("Chapter0001.xhtml", "Chapter0002.xhtml"):
        (tmp_path / filename).write_text(
            "<html><body><p>chapter</p></body></html>", encoding="utf-8"
        )

    (tmp_path / "translated_headers.txt").write_text(
        """Chapter 1:
  Original:   Original One
  Translated: Translated One
  Output File: Chapter0001
----------------------------------------
Chapter 2:
  Original:   Original Two
  Translated: Original Two
  Output File: Chapter0002
  Status:     ⚠️ Using original (translation failed)
----------------------------------------
""",
        encoding="utf-8",
    )
    (tmp_path / "TOC.txt").write_text(
        """Chapter 1:
  Original:   Original One
  Translated: Original One
  Output File: Chapter0001.xhtml
  Status:     ⚠️ Using original (translation failed)
----------------------------------------
Chapter 2:
  Original:   Original Two
  Translated: Original Two
  Output File: Chapter0002.xhtml
  Status:     ⚠️ Using original (translation failed)
----------------------------------------
""",
        encoding="utf-8",
    )
    progress_path = tmp_path / "translation_progress.json"
    progress_path.write_text(
        json.dumps({
            "version": "2.1",
            "chapters": {
                "__translation_artifact__:toc": {
                    "status": "failed",
                    "model_name": "or/stepfun/step-3.7-flash",
                    "output_file": "TOC.txt",
                }
            },
        }),
        encoding="utf-8",
    )

    compiler = EPUBCompiler(str(tmp_path), log_callback=lambda _message: None)
    compiler.translate_toc_ncx = True
    compiler.api_client = object()
    monkeypatch.setattr(compiler, "is_stopped", lambda: False)
    monkeypatch.setattr(
        compiler,
        "_extract_source_toc_ncx_entries",
        lambda _path: [
            {"label": "Original One", "src": "Chapter0001.xhtml"},
            {"label": "Original Two", "src": "Chapter0002.xhtml"},
        ],
    )
    monkeypatch.setattr(compiler, "_get_chapter_order_from_opf", lambda: {})
    spine = [
        epub_converter.epub.EpubHtml(
            title="One", file_name="Chapter0001.xhtml"
        ),
        epub_converter.epub.EpubHtml(
            title="Two", file_name="Chapter0002.xhtml"
        ),
    ]

    toc = compiler._build_toc_from_source_toc_ncx(spine, [], {})

    assert [item.title for item in toc] == ["Translated One", "Original Two"]
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    entry = progress["chapters"]["__translation_artifact__:toc"]
    assert entry["status"] == "failed"
    assert entry["model_name"] == "RECYCLED"


def test_successful_header_translation_repairs_matching_failed_toc_entry(tmp_path):
    headers_path = tmp_path / "translated_headers.txt"
    headers_path.write_text(
        """Chapter 14:
  Original:   第14章 苍山庶家
  Translated: Chapter 14: Cangshan Gu Tomb
  Output File: chapter0389
----------------------------------------
""",
        encoding="utf-8",
    )
    toc_path = tmp_path / "TOC.txt"
    toc_path.write_text(
        """TOC Translations
==================================================

Chapter 1:
  Original:   第14章 苍山庶家
  Translated: 第14章 苍山庶家
  Target URI: response_chapter0389.html
  Status:     ⚠️ Using original (translation failed)
----------------------------------------
""",
        encoding="utf-8",
    )
    compiler = EPUBCompiler(str(tmp_path), log_callback=lambda _message: None)
    compiler.epub_path = str(tmp_path / "source.epub")

    compiler._reconcile_toc_and_headers()

    toc_source, toc_translated, toc_outputs = (
        translate_headers_standalone.load_translations_from_file(
            str(toc_path), log_callback=lambda _message: None
        )
    )
    assert toc_source == {1: "第14章 苍山庶家"}
    assert toc_translated == {1: "Chapter 14: Cangshan Gu Tomb"}
    assert toc_outputs == {1: "response_chapter0389.html"}
    assert "translation failed" not in toc_path.read_text(encoding="utf-8")


def test_source_toc_falls_back_to_epub3_nav_and_uses_toc_txt(tmp_path, monkeypatch):
    source_epub = tmp_path / "nav-only.epub"
    container_xml = """<?xml version="1.0" encoding="UTF-8"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""
    content_opf = """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <manifest>
    <item id="navigation" href="Text/navigation.xhtml"
          media-type="application/xhtml+xml" properties="nav"/>
    <item id="part" href="Text/Section0001.xhtml" media-type="application/xhtml+xml"/>
    <item id="chapter" href="Text/Chapter0001.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="part"/>
    <itemref idref="chapter"/>
  </spine>
</package>
"""
    nav_xhtml = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"
      xmlns:epub="http://www.idpf.org/2007/ops">
  <body>
    <nav epub:type="toc" id="toc" role="doc-toc">
      <ol>
        <li><a href="Section0001.xhtml">Night Snow Chapter</a>
          <ol><li><a href="Chapter0001.xhtml">Chapter One</a></li></ol>
        </li>
      </ol>
    </nav>
    <nav epub:type="landmarks">
      <ol><li><a href="cover.xhtml">Cover</a></li></ol>
    </nav>
  </body>
</html>
"""
    with zipfile.ZipFile(source_epub, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", container_xml)
        archive.writestr("OEBPS/content.opf", content_opf)
        archive.writestr("OEBPS/Text/navigation.xhtml", nav_xhtml)

    toc_txt = tmp_path / "TOC.txt"
    toc_txt.write_text(
        """TOC Translations
==================================================

Chapter 1:
  Original:   Night Snow Chapter
  Translated: Night Snow Arc
  Target URI: Section0001.xhtml
----------------------------------------
Chapter 2:
  Original:   Chapter One
  Translated: Chapter 1: Into the Jianghu
  Target URI: Chapter0001.xhtml
----------------------------------------
""",
        encoding="utf-8",
    )
    (tmp_path / "Section0001.xhtml").write_text(
        "<html><body><h1>Section</h1></body></html>", encoding="utf-8"
    )
    (tmp_path / "Chapter0001.xhtml").write_text(
        "<html><body><h1>Chapter</h1></body></html>", encoding="utf-8"
    )

    logs = []
    compiler = EPUBCompiler(str(tmp_path), log_callback=logs.append)
    compiler.translate_toc_ncx = True
    compiler.api_client = None
    monkeypatch.setenv("EPUB_PATH", str(source_epub))
    spine = [
        epub_converter.epub.EpubHtml(
            title="Section", file_name="Section0001.xhtml"
        ),
        epub_converter.epub.EpubHtml(
            title="Chapter", file_name="Chapter0001.xhtml"
        ),
    ]

    entries = compiler._extract_source_toc_ncx_entries(str(source_epub))
    toc = compiler._build_toc_from_source_toc_ncx(spine, [], {})

    assert entries == [
        {"label": "Night Snow Chapter", "src": "Section0001.xhtml"},
        {"label": "Chapter One", "src": "Chapter0001.xhtml"},
    ]
    assert [(item.href, item.title) for item in toc] == [
        ("Section0001.xhtml", "Night Snow Arc"),
        ("Chapter0001.xhtml", "Chapter 1: Into the Jianghu"),
    ]
    assert any("using EPUB 3 navigation document" in message for message in logs)
    assert all("Cover" not in item.title for item in toc)


def test_source_toc_prefers_ncx_when_ncx_and_epub3_nav_both_exist(tmp_path):
    source_epub = tmp_path / "dual-navigation.epub"
    container_xml = """<?xml version="1.0" encoding="UTF-8"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles><rootfile full-path="OPS/package.opf"/></rootfiles>
</container>
"""
    content_opf = """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <manifest>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
  </manifest>
  <spine toc="ncx"/>
</package>
"""
    ncx = """<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/">
  <navMap><navPoint id="one"><navLabel><text>NCX title</text></navLabel>
    <content src="chapter.xhtml"/></navPoint></navMap>
</ncx>
"""
    nav = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"
      xmlns:epub="http://www.idpf.org/2007/ops">
  <body><nav epub:type="toc"><ol>
    <li><a href="chapter.xhtml">NAV title</a></li>
  </ol></nav></body>
</html>
"""
    with zipfile.ZipFile(source_epub, "w") as archive:
        archive.writestr("META-INF/container.xml", container_xml)
        archive.writestr("OPS/package.opf", content_opf)
        archive.writestr("OPS/toc.ncx", ncx)
        archive.writestr("OPS/nav.xhtml", nav)

    compiler = EPUBCompiler(str(tmp_path), log_callback=lambda _message: None)

    assert compiler._extract_source_toc_ncx_entries(str(source_epub)) == [
        {"label": "NCX title", "src": "chapter.xhtml"}
    ]


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


def test_full_image_rename_replays_complete_map_on_later_runs(tmp_path):
    images_dir = tmp_path / 'images'
    images_dir.mkdir()
    source_names = ('1.png', '10.png', '100.png')
    for source_name in source_names:
        (images_dir / source_name).write_bytes(_remote_test_png_bytes())

    source_markup = ''.join(
        f'<img src="images/{source_name}">' for source_name in source_names
    )

    def fresh_chapters():
        return [{
            'num': 1,
            'title': 'Chapter 1',
            'filename': 'chapter0001.xhtml',
            'original_basename': 'chapter0001',
            'body': source_markup,
            'original_html': source_markup,
        }]

    first = chapter_extractor._rename_images_to_chapter_format(
        fresh_chapters(), str(tmp_path)
    )
    first_map = json.loads(
        (tmp_path / 'image_rename_map.json').read_text(encoding='utf-8')
    )
    first_files = {
        path.name for path in images_dir.iterdir() if path.is_file()
    }

    second = chapter_extractor._rename_images_to_chapter_format(
        fresh_chapters(), str(tmp_path)
    )
    second_map = json.loads(
        (tmp_path / 'image_rename_map.json').read_text(encoding='utf-8')
    )
    second_files = {
        path.name for path in images_dir.iterdir() if path.is_file()
    }

    assert first_map == {
        '1.png': 'chapter0001_img_1.png',
        '10.png': 'chapter0001_img_2.png',
        '100.png': 'chapter0001_img_3.png',
    }
    assert second_map == first_map
    assert second_files == first_files == set(first_map.values())
    assert all(target in second[0]['body'] for target in first_map.values())
    assert all(target in first[0]['body'] for target in first_map.values())


def test_image_rename_map_loader_resolves_terminal_chain_target(tmp_path):
    (tmp_path / 'image_rename_map.json').write_text(
        json.dumps({
            '1.png': 'chapter001_img_1.png',
            'chapter001_img_1.png': 'chapter002_img_1.png',
        }),
        encoding='utf-8',
    )

    exact, folded = chapter_extractor._load_image_rename_targets(str(tmp_path))

    assert exact['1.png'] == 'chapter002_img_1.png'
    assert folded['1.png'] == 'chapter002_img_1.png'


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
        source_epub_image_count=7,
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
    assert manifest['source_epub_image_count'] == 7
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


def test_remote_image_progress_cache_resolves_image_rename_map(
    monkeypatch, tmp_path
):
    remote_url = 'https://images.example.test/renamed-resumable.file'
    digest = hashlib.sha256(remote_url.encode('utf-8')).hexdigest()[:20]
    download_name = f'remote_{digest}.png'
    final_name = 'chapter0004_img_1.png'
    images_dir = tmp_path / 'images'
    cache_dir = images_dir / '.cache'
    cache_dir.mkdir(parents=True)
    (images_dir / final_name).write_bytes(_remote_test_png_bytes())
    (tmp_path / 'image_rename_map.json').write_text(
        json.dumps({download_name: final_name}),
        encoding='utf-8',
    )
    # Simulate an interrupted subsequent run that already reset the item to
    # pending even though the prior renamed PNG remains valid on disk.
    (cache_dir / 'remote_image_download_progress.json').write_text(
        json.dumps({
            'items': [{
                'url': remote_url,
                'status': 'pending',
                'filename': download_name,
                'download_filename': download_name,
            }],
        }),
        encoding='utf-8',
    )

    def unexpected_download(_url):
        raise AssertionError('rename-mapped cached PNG should be reused')

    monkeypatch.setattr(
        chapter_extractor,
        '_download_remote_image_as_png',
        unexpected_download,
    )
    localized = chapter_extractor._localize_remote_images(
        [{'num': 4, 'body': f'<img src="{remote_url}">'}],
        str(tmp_path),
    )

    manifest = json.loads((
        cache_dir / 'remote_image_download_progress.json'
    ).read_text(encoding='utf-8'))
    assert manifest['status'] == 'completed'
    assert manifest['resumed'] == 1
    assert manifest['items'][0]['filename'] == final_name
    assert manifest['items'][0]['local_reference'] == f'images/{final_name}'
    assert f'images/{final_name}' in localized[0]['body']
    assert remote_url not in localized[0]['body']


def test_remote_image_cache_preservation_requires_matching_source_count(
    monkeypatch, tmp_path
):
    from TransateKRtoEN import _should_preserve_remote_image_cache

    source_epub = tmp_path / 'source.epub'
    with zipfile.ZipFile(source_epub, 'w') as archive:
        archive.writestr('OEBPS/Images/one.png', _remote_test_png_bytes())
        archive.writestr('OEBPS/Images/two.jpg', b'jpeg placeholder')
        archive.writestr('OEBPS/Text/chapter.xhtml', '<p>Chapter</p>')

    output_dir = tmp_path / 'output'
    cache_dir = output_dir / 'images' / '.cache'
    cache_dir.mkdir(parents=True)
    manifest_path = cache_dir / 'remote_image_download_progress.json'
    manifest_path.write_text(
        json.dumps({
            'version': 2,
            'source_epub_image_count': 2,
            'items': [],
        }),
        encoding='utf-8',
    )
    monkeypatch.setenv('DOWNLOAD_REMOTE_IMAGE_URLS', '1')

    assert _should_preserve_remote_image_cache(
        str(source_epub), str(output_dir)
    ) is True

    with zipfile.ZipFile(source_epub, 'a') as archive:
        archive.writestr('OEBPS/Images/three.webp', b'webp placeholder')

    assert _should_preserve_remote_image_cache(
        str(source_epub), str(output_dir)
    ) is False


def test_resource_cleanup_preserves_remote_image_cache(monkeypatch, tmp_path):
    images_dir = tmp_path / 'images'
    cache_dir = images_dir / '.cache'
    cache_dir.mkdir(parents=True)
    cached_png = images_dir / 'chapter0001_img_1.png'
    cached_png.write_bytes(_remote_test_png_bytes())
    cache_manifest = cache_dir / 'remote_image_download_progress.json'
    cache_manifest.write_text('{}', encoding='utf-8')
    (tmp_path / 'css').mkdir()
    (tmp_path / 'css' / 'old.css').write_text('old', encoding='utf-8')
    monkeypatch.delenv('PRESERVE_REMOTE_IMAGE_CACHE', raising=False)

    chapter_extractor._cleanup_old_resources(
        str(tmp_path),
        preserve_images=True,
    )

    assert cached_png.is_file()
    assert cache_manifest.is_file()
    assert not (tmp_path / 'css').exists()


def test_single_chapter_retry_restores_persisted_image_names_and_refs(tmp_path):
    images_dir = tmp_path / 'images'
    images_dir.mkdir()
    original_name = 'source-illustration.png'
    prior_canonical_name = 'chapter0041_img_1.png'
    canonical_name = 'chapter0042_img_1.png'
    (images_dir / original_name).write_bytes(_remote_test_png_bytes())
    (tmp_path / 'image_rename_map.json').write_text(
        json.dumps({
            original_name: prior_canonical_name,
            prior_canonical_name: canonical_name,
        }),
        encoding='utf-8',
    )
    markup = (
        '<html><body>'
        f'<img src="../images/{original_name}">'
        f'<div style="background-image: url(\'../images/{original_name}\')"></div>'
        '</body></html>'
    )
    chapters = [{
        'num': 42,
        'body': markup,
        'original_html': markup,
    }]

    restored = chapter_extractor._prepare_single_chapter_image_renames(
        chapters,
        str(tmp_path),
    )

    assert not (images_dir / original_name).exists()
    assert not (images_dir / prior_canonical_name).exists()
    assert (images_dir / canonical_name).is_file()
    assert original_name not in restored[0]['body']
    assert restored[0]['body'].count(canonical_name) == 2
    assert original_name not in restored[0]['original_html']
    assert restored[0]['original_html'].count(canonical_name) == 2


def test_first_reader_translation_creates_a_targeted_image_rename_map(tmp_path):
    images_dir = tmp_path / 'images'
    images_dir.mkdir()
    original_name = 'first-reader-image.png'
    canonical_name = 'chapter0007_img_1.png'
    (images_dir / original_name).write_bytes(_remote_test_png_bytes())
    chapters = [{
        'num': 7,
        'filename': 'Text/chapter0007.xhtml',
        'original_basename': 'chapter0007',
        'body': f'<html><body><img src="../images/{original_name}"></body></html>',
    }]

    prepared = chapter_extractor._prepare_single_chapter_image_renames(
        chapters,
        str(tmp_path),
    )

    assert not (images_dir / original_name).exists()
    assert (images_dir / canonical_name).is_file()
    assert canonical_name in prepared[0]['body']
    assert json.loads(
        (tmp_path / 'image_rename_map.json').read_text(encoding='utf-8')
    ) == {original_name: canonical_name}


def test_first_reader_extraction_bootstraps_missing_epub_images(
    monkeypatch, tmp_path
):
    epub_path = tmp_path / 'reader-source.epub'
    chapter_markup = (
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        '<h1>Chapter 7</h1><p>Reader translation target.</p>'
        '<img src="../Images/illustration.png"/>'
        '</body></html>'
    )
    container_xml = '''<?xml version="1.0"?>
    <container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
      <rootfiles><rootfile full-path="OEBPS/content.opf"
        media-type="application/oebps-package+xml"/></rootfiles>
    </container>'''
    content_opf = '''<?xml version="1.0" encoding="UTF-8"?>
    <package xmlns="http://www.idpf.org/2007/opf" version="2.0"
      unique-identifier="book-id">
      <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
        <dc:identifier id="book-id">reader-test</dc:identifier>
        <dc:title>Reader Test</dc:title><dc:language>en</dc:language>
      </metadata>
      <manifest>
        <item id="chapter7" href="Text/chapter0007.xhtml"
          media-type="application/xhtml+xml"/>
        <item id="image1" href="Images/illustration.png"
          media-type="image/png"/>
      </manifest>
      <spine><itemref idref="chapter7"/></spine>
    </package>'''
    with zipfile.ZipFile(epub_path, 'w') as archive:
        archive.writestr('mimetype', 'application/epub+zip')
        archive.writestr('META-INF/container.xml', container_xml)
        archive.writestr('OEBPS/content.opf', content_opf)
        archive.writestr('OEBPS/Text/chapter0007.xhtml', chapter_markup)
        archive.writestr(
            'OEBPS/Images/illustration.png',
            _remote_test_png_bytes(),
        )

    output_dir = tmp_path / 'reader-output'
    output_dir.mkdir()
    monkeypatch.setenv('SINGLE_CHAPTER_FILTER', 'chapter0007.xhtml')
    monkeypatch.setenv('EXTRACTION_MODE', 'comprehensive')
    monkeypatch.setenv('EXTRACTION_WORKERS', '1')
    monkeypatch.setenv('DOWNLOAD_REMOTE_IMAGE_URLS', '0')

    with zipfile.ZipFile(epub_path, 'r') as archive:
        chapters = chapter_extractor.extract_chapters(
            archive,
            str(output_dir),
            parser='html.parser',
        )

    canonical_name = 'chapter0007_img_1.png'
    assert len(chapters) == 1
    assert not (output_dir / 'images' / 'illustration.png').exists()
    assert (output_dir / 'images' / canonical_name).is_file()
    assert canonical_name in chapters[0]['body']
    assert json.loads(
        (output_dir / 'image_rename_map.json').read_text(encoding='utf-8')
    ) == {'illustration.png': canonical_name}


def test_single_chapter_cleanup_preserves_the_full_epub_workspace(tmp_path):
    from TransateKRtoEN import cleanup_previous_extraction

    images_dir = tmp_path / 'images'
    images_dir.mkdir()
    image_path = images_dir / 'chapter0001_img_1.png'
    image_path.write_bytes(_remote_test_png_bytes())
    marker_path = tmp_path / '.resources_extracted'
    marker_path.write_text('ready', encoding='utf-8')
    opf_path = tmp_path / 'content.opf'
    opf_path.write_text('<package/>', encoding='utf-8')
    map_path = tmp_path / 'image_rename_map.json'
    map_path.write_text('{}', encoding='utf-8')

    cleaned = cleanup_previous_extraction(
        str(tmp_path),
        preserve_workspace=True,
    )

    assert cleaned == 0
    assert image_path.is_file()
    assert marker_path.is_file()
    assert opf_path.is_file()
    assert map_path.is_file()


def test_epub_fingerprint_managed_cleanup_preserves_resources(tmp_path):
    from TransateKRtoEN import cleanup_previous_extraction

    images_dir = tmp_path / 'images'
    images_dir.mkdir()
    image_path = images_dir / 'chapter0001_img_1.png'
    image_path.write_bytes(_remote_test_png_bytes())
    marker_path = tmp_path / '.resources_extracted'
    marker_path.write_text('fingerprint', encoding='utf-8')
    opf_path = tmp_path / 'content.opf'
    opf_path.write_text('<package/>', encoding='utf-8')
    ncx_path = tmp_path / 'toc.ncx'
    ncx_path.write_text('<ncx/>', encoding='utf-8')

    cleaned = cleanup_previous_extraction(
        str(tmp_path),
        fingerprint_managed=True,
    )

    assert cleaned == 0
    assert marker_path.is_file()
    assert image_path.is_file()
    assert opf_path.is_file()
    assert ncx_path.is_file()


def test_chapter_extractor_preserves_cache_only_for_matching_source_count(
    monkeypatch, tmp_path
):
    images_dir = tmp_path / 'images'
    cache_dir = images_dir / '.cache'
    cache_dir.mkdir(parents=True)
    (cache_dir / 'remote_image_download_progress.json').write_text(
        json.dumps({
            'version': 2,
            'source_epub_image_count': 12,
            'items': [],
        }),
        encoding='utf-8',
    )
    monkeypatch.setenv('DOWNLOAD_REMOTE_IMAGE_URLS', '1')

    assert chapter_extractor._remote_image_cache_matches_source(
        str(tmp_path), 12
    ) is True
    assert chapter_extractor._remote_image_cache_matches_source(
        str(tmp_path), 11
    ) is False

    monkeypatch.setenv('DOWNLOAD_REMOTE_IMAGE_URLS', '0')
    assert chapter_extractor._remote_image_cache_matches_source(
        str(tmp_path), 12
    ) is False


def test_changed_partial_epub_preserves_images_used_by_unmapped_html(
    monkeypatch,
    tmp_path,
):
    source_epub = tmp_path / 'partial.epub'
    opf = '''<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Partial update</dc:title>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="c3000" href="Text/chapter3000.xhtml"
          media-type="application/xhtml+xml"/>
    <item id="current-image" href="Images/current.png" media-type="image/png"/>
  </manifest>
  <spine><itemref idref="c3000"/></spine>
</package>'''
    chapter = (
        '<html><body><h1>Chapter 3000</h1>'
        '<p>Current partial source chapter.</p>'
        '<img src="../Images/current.png"/>'
        '</body></html>'
    )
    with zipfile.ZipFile(source_epub, 'w') as archive:
        archive.writestr('OEBPS/content.opf', opf)
        archive.writestr('OEBPS/Text/chapter3000.xhtml', chapter)
        archive.writestr('OEBPS/Images/current.png', b'current-image-bytes')

    output_dir = tmp_path / 'output'
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True)
    retained_image = images_dir / 'chapter0001_img_1.png'
    retained_image.write_bytes(b'retained-image-bytes')
    stale_image = images_dir / 'unused-stale.png'
    stale_image.write_bytes(b'unused-stale-bytes')
    retained_html = output_dir / 'response_chapter0001.html'
    retained_html.write_text(
        '<html><body><img src="images/chapter0001_img_1.png"/></body></html>',
        encoding='utf-8',
    )

    monkeypatch.delenv('SINGLE_CHAPTER_FILTER', raising=False)
    monkeypatch.setenv('EXTRACTION_MODE', 'comprehensive')
    monkeypatch.setenv('EXTRACTION_WORKERS', '1')
    monkeypatch.setenv('DOWNLOAD_REMOTE_IMAGE_URLS', '0')
    monkeypatch.setenv('DISABLE_CHAPTER_MERGING', '1')

    with zipfile.ZipFile(source_epub, 'r') as archive:
        chapters = chapter_extractor.extract_chapters(
            archive,
            str(output_dir),
            parser='html.parser',
        )

    assert len(chapters) == 1
    assert retained_image.read_bytes() == b'retained-image-bytes'
    assert not stale_image.exists()
    assert (images_dir / 'chapter3000_img_1.png').read_bytes() == b'current-image-bytes'
    assert 'chapter0001_img_1.png' in retained_html.read_text(encoding='utf-8')
    rename_map = json.loads(
        (output_dir / 'image_rename_map.json').read_text(encoding='utf-8')
    )
    assert rename_map == {'current.png': 'chapter3000_img_1.png'}


def test_resource_marker_does_not_suppress_mismatched_remote_cache_refresh(
    monkeypatch, tmp_path
):
    output_dir = tmp_path / 'output'
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True)
    stale_image = images_dir / 'stale.png'
    stale_image.write_bytes(_remote_test_png_bytes())
    (output_dir / '.resources_extracted').write_text(
        'previous extraction', encoding='utf-8'
    )
    source_epub = tmp_path / 'source.epub'
    with zipfile.ZipFile(source_epub, 'w') as archive:
        archive.writestr('OEBPS/Images/current.png', _remote_test_png_bytes())

    monkeypatch.setenv('DOWNLOAD_REMOTE_IMAGE_URLS', '1')
    with zipfile.ZipFile(source_epub, 'r') as archive:
        chapter_extractor._extract_all_resources(
            archive,
            str(output_dir),
            preserve_images=False,
        )

    assert not stale_image.exists()
    assert (images_dir / 'current.png').is_file()


def _write_resource_fingerprint_epub(epub_path, *, comment=b'A'):
    with zipfile.ZipFile(epub_path, 'w') as archive:
        archive.writestr('OEBPS/Images/picture.png', b'packaged image bytes')
        archive.writestr('OEBPS/Styles/book.css', b'body { color: black; }')
        archive.writestr('OEBPS/Fonts/book.woff', b'packaged font bytes')
        archive.writestr('OEBPS/content.opf', b'<package/>')
        archive.writestr('META-INF/container.xml', b'<container/>')
        archive.writestr('OEBPS/Scripts/book.js', b'window.book = true;')
        archive.writestr(
            'OEBPS/Text/chapter.xhtml',
            '<html><body><p>Fingerprint source text.</p></body></html>',
        )
        archive.comment = comment


def _write_chapter_cache_epub(epub_path):
    container_xml = '''<?xml version="1.0"?>
    <container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
      <rootfiles><rootfile full-path="OEBPS/content.opf"
        media-type="application/oebps-package+xml"/></rootfiles>
    </container>'''
    content_opf = '''<?xml version="1.0" encoding="UTF-8"?>
    <package xmlns="http://www.idpf.org/2007/opf" version="3.0"
      unique-identifier="book-id">
      <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
        <dc:identifier id="book-id">chapter-cache-test</dc:identifier>
        <dc:title>Chapter Cache Test</dc:title>
        <dc:language>en</dc:language>
      </metadata>
      <manifest>
        <item id="chapter1" href="Text/chapter0001.xhtml"
          media-type="application/xhtml+xml"/>
      </manifest>
      <spine><itemref idref="chapter1"/></spine>
    </package>'''
    chapter_markup = (
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        '<h1>Chapter One</h1><p>Stable chapter cache payload.</p>'
        '</body></html>'
    )
    with zipfile.ZipFile(epub_path, 'w') as archive:
        archive.writestr('mimetype', 'application/epub+zip')
        archive.writestr('META-INF/container.xml', container_xml)
        archive.writestr('OEBPS/content.opf', content_opf)
        archive.writestr('OEBPS/Text/chapter0001.xhtml', chapter_markup)


def _extract_chapter_cache_epub(epub_path, output_dir, monkeypatch):
    monkeypatch.delenv('SINGLE_CHAPTER_FILTER', raising=False)
    monkeypatch.setenv('EXTRACTION_WORKERS', '1')
    monkeypatch.setenv('DOWNLOAD_REMOTE_IMAGE_URLS', '0')
    monkeypatch.setenv('DISABLE_CHAPTER_MERGING', '1')
    with zipfile.ZipFile(epub_path, 'r') as archive:
        return chapter_extractor.extract_chapters(
            archive,
            str(output_dir),
            parser='html.parser',
            progress_callback=lambda _message: None,
        )


def test_chapter_cache_skips_scan_and_processing_for_same_engine(
    monkeypatch,
    tmp_path,
):
    epub_path = tmp_path / 'source.epub'
    output_dir = tmp_path / 'output'
    output_dir.mkdir()
    _write_chapter_cache_epub(epub_path)
    monkeypatch.setenv('EXTRACTION_MODE', 'comprehensive')

    first = _extract_chapter_cache_epub(epub_path, output_dir, monkeypatch)
    assert len(first) == 1
    assert (output_dir / '.chapters_extracted').is_file()
    assert (output_dir / 'chapters_full.json').is_file()

    def fail_if_epub_is_scanned(*_args, **_kwargs):
        raise AssertionError('valid chapter cache unexpectedly scanned EPUB')

    monkeypatch.setattr(
        chapter_extractor,
        '_extract_chapters_universal',
        fail_if_epub_is_scanned,
    )
    second = _extract_chapter_cache_epub(epub_path, output_dir, monkeypatch)

    assert second == first


def test_chapter_cache_never_uses_beautifulsoup_for_html2text_selection(
    monkeypatch,
    tmp_path,
):
    epub_path = tmp_path / 'source.epub'
    output_dir = tmp_path / 'output'
    output_dir.mkdir()
    _write_chapter_cache_epub(epub_path)
    monkeypatch.setenv('EXTRACTION_MODE', 'comprehensive')
    beautifulsoup_chapters = _extract_chapter_cache_epub(
        epub_path,
        output_dir,
        monkeypatch,
    )
    assert beautifulsoup_chapters[0]['body']

    calls = []

    def fake_html2text_extraction(*_args, **_kwargs):
        calls.append(True)
        return ([{
            'num': 1,
            'title': 'Chapter One',
            'body': 'HTML2TEXT ONLY PAYLOAD',
            'filename': 'OEBPS/Text/chapter0001.xhtml',
            'original_basename': 'chapter0001',
            'file_size': 22,
            'has_images': False,
            'image_count': 0,
            'detection_method': 'enhanced_sequential_no_merge',
            'content_hash': 'html2text-cache-test',
            'extraction_mode': 'enhanced',
            'enhanced_extraction': True,
            'html2text_blocks': ['HTML2TEXT ONLY PAYLOAD'],
            'html2text_blocks_source_hash': 'html2text-cache-test',
        }], 'english')

    monkeypatch.setenv('EXTRACTION_MODE', 'enhanced')
    monkeypatch.setattr(
        chapter_extractor,
        '_extract_chapters_universal',
        fake_html2text_extraction,
    )
    html2text_chapters = _extract_chapter_cache_epub(
        epub_path,
        output_dir,
        monkeypatch,
    )

    assert calls == [True]
    assert html2text_chapters[0]['body'] == 'HTML2TEXT ONLY PAYLOAD'
    marker = json.loads(
        (output_dir / '.chapters_extracted').read_text(encoding='utf-8')
    )
    assert marker['signature']['engine'] == 'html2text'

    def fail_if_html2text_cache_is_not_used(*_args, **_kwargs):
        raise AssertionError('valid html2text cache unexpectedly rebuilt')

    monkeypatch.setattr(
        chapter_extractor,
        '_extract_chapters_universal',
        fail_if_html2text_cache_is_not_used,
    )
    cached_html2text = _extract_chapter_cache_epub(
        epub_path,
        output_dir,
        monkeypatch,
    )

    assert cached_html2text[0]['body'] == 'HTML2TEXT ONLY PAYLOAD'


def _extract_resource_fingerprint_epub(epub_path, output_dir, monkeypatch):
    monkeypatch.setenv('EXTRACTION_WORKERS', '1')
    monkeypatch.setenv('DOWNLOAD_REMOTE_IMAGE_URLS', '0')
    with zipfile.ZipFile(epub_path, 'r') as archive:
        return chapter_extractor._extract_all_resources(
            archive,
            str(output_dir),
            progress_callback=lambda _message: None,
        )


def test_resource_fingerprint_accepts_image_rename_map_targets(
    monkeypatch, tmp_path
):
    epub_path = tmp_path / 'source.epub'
    output_dir = tmp_path / 'output'
    output_dir.mkdir()
    _write_resource_fingerprint_epub(epub_path)
    _extract_resource_fingerprint_epub(epub_path, output_dir, monkeypatch)

    images_dir = output_dir / 'images'
    original = images_dir / 'picture.png'
    renamed = images_dir / 'chapter001_img_1.png'
    original.rename(renamed)
    (output_dir / 'image_rename_map.json').write_text(
        json.dumps({'picture.png': renamed.name}),
        encoding='utf-8',
    )

    def fail_if_cleanup_runs(*_args, **_kwargs):
        raise AssertionError('valid resource fingerprint unexpectedly re-extracted')

    monkeypatch.setattr(
        chapter_extractor,
        '_cleanup_old_resources',
        fail_if_cleanup_runs,
    )
    resources = _extract_resource_fingerprint_epub(
        epub_path, output_dir, monkeypatch
    )

    marker = json.loads(
        (output_dir / '.resources_extracted').read_text(encoding='utf-8')
    )
    assert marker['version'] == 3
    assert marker['source_epub']['algorithm'] == 'sha256'
    assert marker['images']['source_filenames'] == ['picture.png']
    assert marker['resources'] == {
        'css': ['book.css'],
        'fonts': ['book.woff'],
        'epub_structure': ['container.xml', 'content.opf'],
        'other': ['book.js'],
    }
    assert resources['images'] == [renamed.name]
    assert renamed.is_file()


def test_resource_fingerprint_missing_renamed_image_forces_reextract(
    monkeypatch, tmp_path
):
    epub_path = tmp_path / 'source.epub'
    output_dir = tmp_path / 'output'
    output_dir.mkdir()
    _write_resource_fingerprint_epub(epub_path)
    _extract_resource_fingerprint_epub(epub_path, output_dir, monkeypatch)

    images_dir = output_dir / 'images'
    original = images_dir / 'picture.png'
    renamed = images_dir / 'chapter001_img_1.png'
    original.rename(renamed)
    (output_dir / 'image_rename_map.json').write_text(
        json.dumps({'picture.png': renamed.name}),
        encoding='utf-8',
    )
    renamed.unlink()

    with zipfile.ZipFile(epub_path, 'r') as archive:
        source_fingerprint = (
            chapter_extractor._source_epub_content_fingerprint(archive)
        )
    marker_valid, marker_reason = (
        chapter_extractor._validate_resource_extraction_marker(
            str(output_dir / '.resources_extracted'),
            str(output_dir),
            source_fingerprint,
        )
    )
    assert marker_valid is False
    assert marker_reason == (
        'missing extracted image file(s): chapter001_img_1.png'
    )

    _extract_resource_fingerprint_epub(epub_path, output_dir, monkeypatch)

    assert (images_dir / 'picture.png').read_bytes() == b'packaged image bytes'
    assert not (output_dir / 'image_rename_map.json').exists()


@pytest.mark.parametrize(
    ('relative_path', 'expected_bytes'),
    [
        ('css/book.css', b'body { color: black; }'),
        ('fonts/book.woff', b'packaged font bytes'),
        ('content.opf', b'<package/>'),
        ('container.xml', b'<container/>'),
        ('book.js', b'window.book = true;'),
    ],
)
def test_resource_fingerprint_missing_non_image_resource_forces_reextract(
    monkeypatch,
    tmp_path,
    relative_path,
    expected_bytes,
):
    epub_path = tmp_path / 'source.epub'
    output_dir = tmp_path / 'output'
    output_dir.mkdir()
    _write_resource_fingerprint_epub(epub_path)
    _extract_resource_fingerprint_epub(epub_path, output_dir, monkeypatch)

    missing_resource = output_dir / relative_path
    assert missing_resource.is_file()
    missing_resource.unlink()

    _extract_resource_fingerprint_epub(epub_path, output_dir, monkeypatch)

    assert missing_resource.read_bytes() == expected_bytes


def test_resource_fingerprint_one_byte_epub_change_forces_reextract(
    monkeypatch, tmp_path
):
    epub_path = tmp_path / 'source.epub'
    output_dir = tmp_path / 'output'
    output_dir.mkdir()
    _write_resource_fingerprint_epub(epub_path, comment=b'A')
    _extract_resource_fingerprint_epub(epub_path, output_dir, monkeypatch)
    first_marker = json.loads(
        (output_dir / '.resources_extracted').read_text(encoding='utf-8')
    )

    images_dir = output_dir / 'images'
    original = images_dir / 'picture.png'
    renamed = images_dir / 'chapter001_img_1.png'
    original.rename(renamed)
    (output_dir / 'image_rename_map.json').write_text(
        json.dumps({'picture.png': renamed.name}),
        encoding='utf-8',
    )

    # The ZIP comment is the final byte of this archive. Change that one byte
    # without touching a member, timestamp, filename, or archive length.
    with open(epub_path, 'r+b') as source:
        source.seek(-1, os.SEEK_END)
        assert source.read(1) == b'A'
        source.seek(-1, os.SEEK_END)
        source.write(b'B')

    _extract_resource_fingerprint_epub(epub_path, output_dir, monkeypatch)
    second_marker = json.loads(
        (output_dir / '.resources_extracted').read_text(encoding='utf-8')
    )

    assert first_marker['source_epub']['sha256'] != second_marker['source_epub']['sha256']
    assert (images_dir / 'picture.png').read_bytes() == b'packaged image bytes'
    assert not renamed.exists()
    assert not (output_dir / 'image_rename_map.json').exists()


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


@pytest.mark.parametrize('use_markdown2', [False, True])
def test_convert_br_to_paragraphs_toggle_is_on_by_default(
    monkeypatch, use_markdown2
):
    from TransateKRtoEN import convert_enhanced_text_to_html

    monkeypatch.setenv('SKIP_MARKDOWN_TO_HTML', '0')
    monkeypatch.setenv(
        'USE_MARKDOWN2_CONVERTER', '1' if use_markdown2 else '0'
    )
    monkeypatch.delenv('CONVERT_BR_TO_PARAGRAPHS', raising=False)
    source = 'First translated line\nSecond translated line'

    default_html = convert_enhanced_text_to_html(
        source, {'preserve_structure': True}
    )

    assert '<br' not in default_html.lower()
    assert '<p>First translated line</p>' in default_html
    assert '<p>Second translated line</p>' in default_html

    monkeypatch.setenv('CONVERT_BR_TO_PARAGRAPHS', '0')
    retained_html = convert_enhanced_text_to_html(
        source, {'preserve_structure': True}
    )

    assert retained_html.lower().count('<br') == 1
    assert 'First translated line' in retained_html
    assert 'Second translated line' in retained_html


@pytest.mark.parametrize('use_markdown2', [False, True])
def test_convert_br_to_paragraphs_does_not_change_list_markup(
    monkeypatch, use_markdown2
):
    from TransateKRtoEN import convert_enhanced_text_to_html

    monkeypatch.setenv('SKIP_MARKDOWN_TO_HTML', '0')
    monkeypatch.setenv(
        'USE_MARKDOWN2_CONVERTER', '1' if use_markdown2 else '0'
    )
    source = (
        'Opening first line\nOpening second line\n\n'
        '- First bullet\n- Second bullet'
    )

    monkeypatch.setenv('CONVERT_BR_TO_PARAGRAPHS', '0')
    retained_html = convert_enhanced_text_to_html(
        source, {'preserve_structure': True}
    )
    monkeypatch.setenv('CONVERT_BR_TO_PARAGRAPHS', '1')
    converted_html = convert_enhanced_text_to_html(
        source, {'preserve_structure': True}
    )

    retained_list = re.search(r'<ul\b.*?</ul>', retained_html, re.DOTALL)
    converted_list = re.search(r'<ul\b.*?</ul>', converted_html, re.DOTALL)
    assert retained_list is not None
    assert converted_list is not None
    assert converted_list.group(0) == retained_list.group(0)
    assert converted_list.group(0).count('<li>') == 2


def test_convert_br_to_paragraphs_preserves_inline_and_surrounding_markup():
    from html_output_utils import convert_br_to_paragraphs

    prefix = "<UL class='original'><LI>Keep list bytes.</LI></UL>"
    paragraph = '<p class="body"><em>First<br/>Second</em></p>'
    suffix = '<OL><LI>Keep this too.</LI></OL>'

    converted = convert_br_to_paragraphs(prefix + paragraph + suffix)

    assert converted == (
        prefix
        + '<p class="body"><em>First</em></p>'
        + '<p class="body"><em>Second</em></p>'
        + suffix
    )


def test_manual_br_conversion_only_changes_root_html_outputs(tmp_path):
    from html_output_utils import convert_br_in_output_folder

    root_html = tmp_path / 'response_001.html'
    original_list = '<ul class="keep"><li>Bullet</li></ul>'
    root_html.write_text(
        '<p>First<br/>Second</p>' + original_list,
        encoding='utf-8',
    )
    unchanged_html = tmp_path / 'response_002.xhtml'
    unchanged_html.write_text('<p>Already separate.</p>', encoding='utf-8')
    extracted_html = tmp_path / 'EPUB' / 'Text' / 'chapter.xhtml'
    extracted_html.parent.mkdir(parents=True)
    extracted_source = '<p>Do not<br/>touch extracted EPUB content.</p>'
    extracted_html.write_text(extracted_source, encoding='utf-8')

    audit = convert_br_in_output_folder(str(tmp_path))

    assert audit['scanned'] == 2
    assert audit['changed'] == 1
    assert audit['unchanged'] == 1
    assert audit['failed'] == 0
    assert root_html.read_text(encoding='utf-8') == (
        '<p>First</p><p>Second</p>' + original_list
    )
    assert extracted_html.read_text(encoding='utf-8') == extracted_source


def test_manual_br_conversion_preserves_utf8_bom(tmp_path):
    from html_output_utils import convert_br_in_output_folder

    html_path = tmp_path / 'response_bom.html'
    html_path.write_bytes(
        b'\xef\xbb\xbf' + '<p>First<br>Second</p>'.encode('utf-8')
    )

    audit = convert_br_in_output_folder(str(tmp_path))

    converted_bytes = html_path.read_bytes()
    assert audit['changed'] == 1
    assert converted_bytes.startswith(b'\xef\xbb\xbf')
    assert converted_bytes[3:].decode('utf-8') == (
        '<p>First</p><p>Second</p>'
    )


@pytest.mark.parametrize('use_markdown2', [False, True])
def test_preserve_asterisk_separator_lines_toggle_defaults_on(
    monkeypatch, use_markdown2
):
    from TransateKRtoEN import convert_enhanced_text_to_html

    monkeypatch.setenv('SKIP_MARKDOWN_TO_HTML', '0')
    monkeypatch.setenv(
        'USE_MARKDOWN2_CONVERTER', '1' if use_markdown2 else '0'
    )
    monkeypatch.delenv('PRESERVE_ASTERISK_SEPARATOR_LINES', raising=False)
    source = 'Before\n\n*****\n\nAfter'

    preserved_html = convert_enhanced_text_to_html(
        source, {'preserve_structure': True}
    )

    assert '<hr' not in preserved_html.lower()
    assert '<p>*****</p>' in preserved_html

    monkeypatch.setenv('PRESERVE_ASTERISK_SEPARATOR_LINES', '0')
    converted_html = convert_enhanced_text_to_html(
        source, {'preserve_structure': True}
    )

    assert '<hr' in converted_html.lower()
    assert '<p>*****</p>' not in converted_html


_STANDARD_OPF_CONTAINER = """<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"
           version="1.0">
  <rootfiles>
    <rootfile full-path="item/standard.opf"
              media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""


def _write_opf_discovery_epub(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def test_opf_discovery_container_rootfile_wins_over_content_opf(tmp_path):
    epub_path = tmp_path / "book.epub"
    _write_opf_discovery_epub(
        epub_path,
        {
            "META-INF/container.xml": _STANDARD_OPF_CONTAINER,
            "content.opf": "<package id='decoy'/>",
            "item/standard.opf": "<package id='real'/>",
        },
    )

    with zipfile.ZipFile(epub_path) as archive:
        assert find_epub_opf_member(archive) == "item/standard.opf"


def test_opf_discovery_prefers_content_only_as_archive_fallback(tmp_path):
    epub_path = tmp_path / "book.epub"
    _write_opf_discovery_epub(
        epub_path,
        {
            "OPS/standard.opf": "<package/>",
            "OEBPS/content.opf": "<package/>",
        },
    )

    with zipfile.ZipFile(epub_path) as archive:
        assert find_epub_opf_member(archive) == "OEBPS/content.opf"


def test_opf_discovery_accepts_any_opf_as_last_archive_fallback(tmp_path):
    epub_path = tmp_path / "book.epub"
    _write_opf_discovery_epub(epub_path, {"item/standard.opf": "<package/>"})

    with zipfile.ZipFile(epub_path) as archive:
        assert find_epub_opf_member(archive) == "item/standard.opf"


def test_opf_discovery_resolves_flat_container_declared_workspace(tmp_path):
    (tmp_path / "container.xml").write_text(
        _STANDARD_OPF_CONTAINER,
        encoding="utf-8",
    )
    expected = tmp_path / "standard.opf"
    expected.write_text("<package/>", encoding="utf-8")
    (tmp_path / "content.opf").write_text(
        "<package id='decoy'/>",
        encoding="utf-8",
    )

    assert find_opf_path(str(tmp_path)) == str(expected)


def test_opf_discovery_accepts_any_workspace_opf(tmp_path):
    expected = tmp_path / "standard.opf"
    expected.write_text("<package/>", encoding="utf-8")

    assert find_opf_path(str(tmp_path)) == str(expected)
