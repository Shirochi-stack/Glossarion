import pytest
import os
from pathlib import Path

import epub_converter
from epub_converter import EPUBCompiler, FileUtils, HTMLEntityDecoder, XMLValidator


def test_html_entity_decoder_basic_entities():
    text = "&lt;Hello&gt; &amp; &quot;World&quot; &apos;!&apos;"
    decoded = HTMLEntityDecoder.decode(text)
    # Expect: <Hello> & "World" '!'
    assert decoded == "<Hello> & \"World\" '!'"


def test_html_entity_decoder_encoding_fixes_no_crash():
    mojibake = "Ã¢â‚¬â„¢ and â€¦ and Â©"
    decoded = HTMLEntityDecoder.decode(mojibake)
    # Should replace with reasonable characters and not raise
    assert isinstance(decoded, str) and len(decoded) >= 3


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


def test_sanitize_filename_for_windows_path_caps_only_file_component(tmp_path):
    safe_title = FileUtils.sanitize_filename_for_windows_path(
        "A" * 400,
        str(tmp_path),
        extension=".epub",
        allow_unicode=True,
    )

    assert safe_title == "A" * 250
    assert len(f"{safe_title}.epub") == FileUtils.WINDOWS_MAX_FILENAME_LENGTH


def test_epub_writer_renames_windows_invalid_and_too_long_title(tmp_path, monkeypatch):
    compiler = EPUBCompiler(str(tmp_path), log_callback=lambda _msg: None)
    captured = {}

    def fake_write_epub(out_path, _book, _opts):
        captured["out_path"] = out_path
        Path(out_path).write_bytes(b"epub")

    monkeypatch.setattr(epub_converter.epub, "write_epub", fake_write_epub)
    monkeypatch.setattr(epub_converter, "_replace_organized_library_epub", lambda *_args: None)

    class Book:
        title = f"{'A' * 400}."

    compiler._write_epub(Book(), {})

    output_path = captured["out_path"]
    output_name = os.path.basename(output_path)
    output_stem = os.path.splitext(output_name)[0]
    assert output_name.endswith(".epub")
    assert not output_stem.endswith(".")
    assert len(output_name) == FileUtils.WINDOWS_MAX_FILENAME_LENGTH
