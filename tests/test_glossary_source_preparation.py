"""Prepare glossary source text once, retaining chapter identities and skips."""

import os
import sys
import zipfile
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from ebooklib import epub

import extract_glossary_from_epub as extractor
import extract_glossary_from_txt as text_extractor


@pytest.fixture
def prepare_source(monkeypatch, tmp_path):
    # main exports config defaults to the environment and updates module state.
    # Keep those changes local to each preparation run.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["extract_glossary_from_epub.py"])
    for name in (
        "PROGRESS_FILE", "_GLOSSARY_OUTPUT_FILE", "_GLOSSARY_CHAPTER_POSITIONS",
        "_GLOSSARY_CHAPTER_NUMBERS", "_GLOSSARY_CHAPTER_FILENAMES",
        "_GLOSSARY_TOTAL_CHAPTERS", "BOOK_TITLE_RAW", "BOOK_TITLE_TRANSLATED",
        "BOOK_TITLE_PRESENT", "BOOK_TITLE_VALUE", "GLOSSARY_SOURCE_LANGUAGE",
        "GLOSSARY_SOURCE_LANGUAGE_PATH", "_GLOSSARY_SOURCE_LANGUAGE_LOADED",
        "_GLOSSARY_SOURCE_LANGUAGE_LOGGED", "GLOSSARY_SOURCE_SCRIPT",
        "GLOSSARY_SOURCE_SCRIPT_IS_CJK", "_GLOSSARY_SOURCE_SCRIPT_READY",
        "_GLOSSARY_SOURCE_SCRIPT_LOGGED",
    ):
        monkeypatch.setattr(extractor, name, getattr(extractor, name))
    for name, value in {
        "OUTPUT_PATH": str(tmp_path / "glossary.json"),
        "DIRECT_TEXT_ACTIVE": "0",
        "TRANSLATION_CANCELLED": "0",
        "GLOSSARY_INCLUDE_BOOK_TITLE": "0",
        "USE_SPINE_ORDER": "0",
        "TRANSLATE_SPECIAL_FILES": "0",
        "SPECIAL_FILE_KEYWORDS": "notice",
        "SPECIAL_FILE_EXACT": "index",
        "GLOSSARY_SKIP_TITLE_HEADER_ONLY": "1",
        "CHAPTER_RANGE": "",
        "BATCH_TRANSLATION": "0",
        "BATCH_SIZE": "1",
        "BATCH_GROUP_SIZE": "1",
        "SEND_INTERVAL_SECONDS": "0",
        "GLOSSARY_COMPRESSION_FACTOR": "1.5",
        "GLOSSARY_REFINEMENT_COMPRESSION_FACTOR": "1.5",
        "GLOSSARY_TEMPERATURE": "0.1",
        "GLOSSARY_CONTEXT_LIMIT": "3",
        "GLOSSARY_CUSTOM_FIELDS": "[]",
        "GLOSSARY_ENABLE_ANTI_DUPLICATE": "0",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(extractor, "_glossary_is_graceful_stop_active", lambda: False)
    monkeypatch.setattr(extractor, "is_stop_requested", lambda: False)
    monkeypatch.setattr(extractor, "load_config", lambda _path: {"model": "test-model"})
    monkeypatch.setattr(extractor, "_log_assistant_prompt_once", lambda: None)
    monkeypatch.setattr(extractor, "_set_glossary_source_language_from_metadata", lambda *a, **kw: None)
    monkeypatch.setattr(extractor, "_extract_raw_title_from_epub", lambda _path: None)
    monkeypatch.setattr(extractor, "_extract_translated_title_from_metadata", lambda *a: None)
    monkeypatch.setattr(extractor, "create_client_with_multi_key_support", lambda *a: SimpleNamespace())
    monkeypatch.setattr(extractor, "_effective_glossary_output_limit", lambda *a: 128000)
    monkeypatch.setattr(extractor, "ChapterSplitter", lambda **kw: SimpleNamespace())
    contexts = []
    make_context = extractor.make_glossary_progress_context

    def capture_context(**kwargs):
        context = make_context(**kwargs)
        contexts.append(context)
        return context

    monkeypatch.setattr(extractor, "make_glossary_progress_context", capture_context)
    load_progress = Mock(side_effect=AssertionError("Preparation should stop before processing"))
    monkeypatch.setattr(extractor, "load_progress", load_progress)

    def run(source):
        monkeypatch.setenv("EPUB_PATH", str(source))
        extractor.main(stop_callback=lambda: bool(contexts and contexts[-1].total_chapters))
        assert len(contexts) == 1
        load_progress.assert_not_called()
        return contexts[0]

    return run


def _write_epub(path):
    book = epub.EpubBook()
    book.set_identifier("single-source-preparation")
    book.set_title("Preparation test")
    book.set_language("en")
    pages = [
        ("chapter_notice0001.xhtml", "<p>Announcement.</p>"),
        ("chapter0042.xhtml", "<p>Alice entered the tower.</p>"),
        ("chapter0043.xhtml", '<img src="illustration.jpg"/>'),
        ("chapter0044.xhtml", "<h1>Chapter Forty Four</h1>"),
        ("chapter0045.xhtml", ""),
    ]
    for index, (filename, content) in enumerate(pages):
        chapter = epub.EpubHtml(uid=f"chapter-{index}", file_name=filename)
        chapter.content = f"<html><head></head><body>{content}</body></html>"
        book.add_item(chapter)
        book.spine.append(chapter)
    epub.write_epub(str(path), book)
    return path


@pytest.mark.parametrize("spine_order", [False, True], ids=["filename-order", "spine-order"])
@pytest.mark.parametrize("skip_headers", [True, False], ids=["skip-headers", "keep-headers"])
def test_main_reads_epub_once_and_preserves_chapter_metadata(
    prepare_source, monkeypatch, tmp_path, capsys, spine_order, skip_headers,
):
    source = _write_epub(tmp_path / "book.epub")
    monkeypatch.setenv("USE_SPINE_ORDER", "1" if spine_order else "0")
    monkeypatch.setenv("GLOSSARY_SKIP_TITLE_HEADER_ONLY", "1" if skip_headers else "0")
    monkeypatch.setenv("CHAPTER_RANGE", "1-4" if spine_order else "42-45")
    read_archive = Mock(wraps=extractor.epub.read_epub)
    monkeypatch.setattr(extractor.epub, "read_epub", read_archive)

    context = prepare_source(source)

    read_archive.assert_called_once_with(str(source))
    assert context.total_chapters == 4
    assert context.chapter_filenames == {
        0: "chapter0042.xhtml", 1: "chapter0043.xhtml",
        2: "chapter0044.xhtml", 3: "chapter0045.xhtml",
    }
    assert context.chapter_positions == (
        {0: 1, 1: 2, 2: 3, 3: 4} if spine_order else {0: 42, 1: 43, 2: 44, 3: 45}
    )
    assert context.chapter_numbers == {0: 42, 1: 43, 2: 44, 3: 45}
    expected_skips = {1: "skipped_image_only", 3: "skipped_empty"}
    if skip_headers:
        expected_skips[2] = "skipped_title_header_only"
    assert context.chapter_status_overrides == expected_skips
    output = capsys.readouterr().out
    assert output.count("Reading EPUB archive and chapter list") == 1
    assert output.count("EPUB text extraction complete") == 1
    assert "Skipped 1 special file(s)" in output


@pytest.mark.parametrize("extension", [".txt", ".pdf", ".sdlxliff", ".srt", ".zip"])
def test_main_loads_each_other_source_format_once(
    prepare_source, monkeypatch, tmp_path, extension,
):
    source = tmp_path / f"book{extension}"
    if extension == ".zip":
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("episode.srt", "1\n00:00:00,000 --> 00:00:01,000\nAlice.\n")
    else:
        source.write_text("source fixture", encoding="utf-8")
    subtitles = extension in (".srt", ".zip")
    chapters = (
        [("Alice.", "episode0099.srt"), ("Bob.", "episode0100.srt")]
        if subtitles else ["Alice.", "Bob."]
    )
    load_source = Mock(return_value=chapters)
    if subtitles:
        monkeypatch.setattr(extractor, "extract_chapters_from_subtitle", load_source)
    elif extension == ".txt":
        monkeypatch.setattr(text_extractor, "extract_chapters_from_txt", load_source)
    elif extension == ".pdf":
        monkeypatch.setattr(extractor, "_extract_pdf_chapters_for_glossary", load_source)
    else:
        monkeypatch.setattr(extractor, "_extract_sdlxliff_chapters_for_glossary", load_source)

    context = prepare_source(source)

    assert load_source.call_count == 1
    assert load_source.call_args.args[0] == str(source)
    if subtitles:
        assert load_source.call_args.kwargs["return_metadata"] is True
    assert context.total_chapters == 2
    assert context.chapter_positions == {0: 1, 1: 2}
    assert context.chapter_filenames == (
        {0: "episode0099.srt", 1: "episode0100.srt"} if subtitles else {}
    )
    assert context.chapter_status_overrides == {}
