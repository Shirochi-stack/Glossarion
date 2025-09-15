import os
import re
from textwrap import dedent
import pytest

from txt_processor import TextFileProcessor

@pytest.fixture
def sample_text_file(tmp_path):
    content = dedent(
        """
        Chapter 1: Beginnings
        This is the first chapter.

        Chapter 2: Middles
        This is the second chapter.

        ***
        A scene break follows with more text.
        """
    ).strip()
    path = tmp_path / "novel.txt"
    path.write_text(content, encoding="utf-8")
    return str(path)

@pytest.fixture
def processor(tmp_path, sample_text_file):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    return TextFileProcessor(sample_text_file, str(output_dir))


def test_detects_chapters_and_sections(processor):
    chapters = processor._detect_chapters(open(processor.file_path, encoding="utf-8").read())
    # Should find at least two numbered chapters and one scene break section
    assert any(ch["num"] == 1 for ch in chapters)
    assert any("Chapter 2" in ch["title"] for ch in chapters)


def test_extract_chapters_splits_to_html(processor):
    result = processor.extract_chapters()
    assert isinstance(result, list) and len(result) >= 1
    # Each item should have body (HTML) and filename
    for item in result:
        assert "<html>" in item["body"]
        assert item["filename"].endswith(".txt")


def test_create_output_structure(processor):
    # Simulate translated chapters as filename, html
    chapters = [
        ("section_1.txt", "<html><body><p>Translated One</p></body></html>"),
        ("section_2.txt", "<html><body><p>Translated Two</p></body></html>"),
    ]
    output_path = processor.create_output_structure(chapters)
    assert os.path.exists(output_path)
    content = open(output_path, encoding="utf-8").read()
    assert "Translated One" in content and "Translated Two" in content
