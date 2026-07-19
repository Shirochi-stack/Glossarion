import pytest
import os
import zipfile
from pathlib import Path

import epub_converter
from epub_converter import EPUBCompiler, FileUtils, HTMLEntityDecoder, XMLValidator
from html_tag_entities import unescape_valid_html_tag_entities
from qa_scan_runtime import default_qa_scan_settings
from scan_html_folder import (
    _count_quotation_marks,
    detect_quotation_mismatch,
    extract_epub_punctuation_info,
    process_html_file_batch,
)


def test_html_entity_decoder_basic_entities():
    text = "&lt;Hello&gt; &amp; &quot;World&quot; &apos;!&apos;"
    decoded = HTMLEntityDecoder.decode(text)
    # Expect: <Hello> & "World" '!'
    assert decoded == "<Hello> & \"World\" '!'"


def test_quotation_check_defaults_off():
    assert default_qa_scan_settings()["check_quotation_mismatch"] is False


def test_quotation_counter_handles_styles_entities_and_apostrophes():
    text = (
        '&quot;double&quot; '
        '&#39;single&#39; '
        '&#x27;hex&#x27; '
        '&#x2018;curly&#x2019; '
        '&apos;named&apos; '
        '「corner」 『white corner』 '
        "don't"
    )

    assert _count_quotation_marks(text) == 14


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
    <p>&quot;double&quot; &#39;single&#39; 「corner」</p>
    </body></html>"""

    with zipfile.ZipFile(epub_path, "w") as epub:
        epub.writestr("OEBPS/content.opf", content_opf)
        epub.writestr("OEBPS/text/chapter.xhtml", chapter)

    source_info = extract_epub_punctuation_info(
        epub_path,
        log=lambda _message: None,
        include_quotation_marks=True,
    )

    assert source_info[1]["quotation_marks"] == 6
    assert source_info[1]["filename"] == "chapter.xhtml"


def test_quotation_scan_counts_only_visible_html_text(tmp_path):
    chapter_path = tmp_path / "chapter.html"
    chapter_path.write_text('<p class="dialog">&quot;one pair&quot;</p>', encoding="utf-8")
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
            "quotation_marks": 4,
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
        source_info,
    ))

    assert "quotation_marks_2_missing_(2/4)" in results[0]["issues"]


def test_valid_html_tag_entities_preserve_angle_bracket_prose():
    prose = "&lt;A talent possessing both a clean character and noble integrity. Who exactly is Riyan?&gt;"

    assert unescape_valid_html_tag_entities(prose) == prose


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
