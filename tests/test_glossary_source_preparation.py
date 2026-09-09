"""Reuse validated glossary EPUB text while retaining chapter identities and skips."""

import json
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
        context_count = len(contexts)
        monkeypatch.setenv("EPUB_PATH", str(source))
        extractor.main(stop_callback=lambda: bool(
            len(contexts) > context_count and contexts[-1].total_chapters
        ))
        assert len(contexts) == context_count + 1
        load_progress.assert_not_called()
        return contexts[-1]

    return run


def _write_epub(path):
    book = epub.EpubBook()
    book.set_identifier("single-source-preparation")
    book.set_title("Preparation test")
    book.set_language("en")
    book.add_item(epub.EpubNcx())
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
    assert os.path.isfile(os.path.join(os.path.dirname(context.progress_file), ".cache", "epub_text.json"))
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

    cached_context = prepare_source(source)

    read_archive.assert_called_once_with(str(source))
    assert cached_context.total_chapters == context.total_chapters
    assert cached_context.chapter_filenames == context.chapter_filenames
    assert cached_context.chapter_positions == context.chapter_positions
    assert cached_context.chapter_numbers == context.chapter_numbers
    assert cached_context.chapter_status_overrides == context.chapter_status_overrides


def test_main_reapplies_structural_skips_and_numbering_to_cached_documents(
    prepare_source, monkeypatch, tmp_path,
):
    source = _write_epub(tmp_path / "book.epub")
    read_archive = Mock(wraps=extractor.epub.read_epub)
    monkeypatch.setattr(extractor.epub, "read_epub", read_archive)

    initial = prepare_source(source)
    monkeypatch.setenv("GLOSSARY_SKIP_TITLE_HEADER_ONLY", "0")
    monkeypatch.setenv("USE_SPINE_ORDER", "1")
    monkeypatch.setenv("CHAPTER_RANGE", "2-3")
    cached = prepare_source(source)

    read_archive.assert_called_once_with(str(source))
    assert initial.chapter_positions == {0: 42, 1: 43, 2: 44, 3: 45}
    assert initial.chapter_status_overrides[2] == "skipped_title_header_only"
    assert cached.chapter_positions == {0: 1, 1: 2, 2: 3, 3: 4}
    assert cached.chapter_numbers == initial.chapter_numbers
    assert cached.chapter_filenames == initial.chapter_filenames
    assert cached.chapter_status_overrides == {1: "skipped_image_only", 3: "skipped_empty"}


def test_main_spine_positions_include_protected_interior_special_document(
    prepare_source, monkeypatch, tmp_path,
):
    source = _write_epub(tmp_path / "book.epub")
    monkeypatch.setenv("GLOSSARY_NEVER_CONSIDER_IN_BETWEEN_FILES_AS_SPECIAL", "1")
    monkeypatch.setenv("SPECIAL_FILE_EXACT", "index,chapter0043")
    monkeypatch.setenv("USE_SPINE_ORDER", "1")
    read_archive = Mock(wraps=extractor.epub.read_epub)
    monkeypatch.setattr(extractor.epub, "read_epub", read_archive)

    initial = prepare_source(source)
    cached = prepare_source(source)

    read_archive.assert_called_once_with(str(source))
    for context in (initial, cached):
        assert context.total_chapters == 4
        assert context.chapter_filenames == {
            0: "chapter0042.xhtml", 1: "chapter0043.xhtml",
            2: "chapter0044.xhtml", 3: "chapter0045.xhtml",
        }
        assert context.chapter_positions == {0: 1, 1: 2, 2: 3, 3: 4}
        assert context.chapter_numbers == {0: 42, 1: 43, 2: 44, 3: 45}


@pytest.fixture
def cached_epub(monkeypatch, tmp_path):
    monkeypatch.setattr(extractor, "is_stop_requested", lambda: False)
    monkeypatch.setenv("TRANSLATE_SPECIAL_FILES", "0")
    monkeypatch.setenv("SPECIAL_FILE_KEYWORDS", "notice")
    monkeypatch.setenv("SPECIAL_FILE_EXACT", "index")
    source = _write_epub(tmp_path / "book.epub")
    cache = tmp_path / ".glossary_epub_text_cache.json"
    read_archive = Mock(wraps=extractor.epub.read_epub)
    monkeypatch.setattr(extractor.epub, "read_epub", read_archive)

    def extract(**kwargs):
        return extractor.extract_chapters_from_epub(str(source), cache_path=str(cache), **kwargs)

    return SimpleNamespace(source=source, cache=cache, read_archive=read_archive, extract=extract)


@pytest.mark.parametrize("first_mode", [{}, {"return_metadata": True}, {"return_document_metadata": True}])
def test_epub_text_cache_preserves_all_documents_across_reader_modes(cached_epub, first_mode):
    cached_epub.extract(**first_mode)

    documents = cached_epub.extract(return_document_metadata=True)
    with_filenames = cached_epub.extract(return_metadata=True)
    texts = cached_epub.extract()

    cached_epub.read_archive.assert_called_once_with(str(cached_epub.source))
    assert [doc["filename"] for doc in documents] == [
        "chapter0042.xhtml", "chapter0043.xhtml", "chapter0044.xhtml", "chapter0045.xhtml",
    ]
    assert [doc["structural_kind"] for doc in documents] == [
        "", "image_only", "title_header_only", "empty",
    ]
    assert [doc["text"] for doc in documents] == [
        "Alice entered the tower.", "", "Chapter Forty Four", "",
    ]
    assert with_filenames == [
        ("Alice entered the tower.", "chapter0042.xhtml"),
        ("Chapter Forty Four", "chapter0044.xhtml"),
    ]
    assert texts == ["Alice entered the tower.", "Chapter Forty Four"]


def _set_zip_comment_preserving_stat(source, comment):
    previous = source.stat()
    with zipfile.ZipFile(source, "a") as archive:
        archive.comment = comment
    os.utime(source, ns=(previous.st_atime_ns, previous.st_mtime_ns))


def test_epub_text_cache_invalidates_same_size_same_mtime_source_changes(cached_epub):
    _set_zip_comment_preserving_stat(cached_epub.source, b"A")
    original = cached_epub.extract(return_document_metadata=True)
    previous = cached_epub.source.stat()

    _set_zip_comment_preserving_stat(cached_epub.source, b"B")
    current = cached_epub.source.stat()
    assert current.st_size == previous.st_size
    assert current.st_mtime_ns == previous.st_mtime_ns
    rebuilt = cached_epub.extract(return_document_metadata=True)
    assert cached_epub.read_archive.call_count == 2
    assert rebuilt == original

    assert cached_epub.extract(return_document_metadata=True) == rebuilt
    assert cached_epub.read_archive.call_count == 2


@pytest.mark.parametrize(("setting", "value", "expected_names"), [
    ("TRANSLATE_SPECIAL_FILES", "1", [
        "chapter_notice0001.xhtml", "chapter0042.xhtml", "chapter0043.xhtml",
        "chapter0044.xhtml", "chapter0045.xhtml",
    ]),
    ("SPECIAL_FILE_KEYWORDS", "copyright", [
        "chapter_notice0001.xhtml", "chapter0042.xhtml", "chapter0043.xhtml",
        "chapter0044.xhtml", "chapter0045.xhtml",
    ]),
    ("SPECIAL_FILE_EXACT", "chapter0042", [
        "chapter0043.xhtml", "chapter0044.xhtml", "chapter0045.xhtml",
    ]),
])
def test_epub_text_cache_invalidates_when_special_file_scope_changes(
    cached_epub, monkeypatch, setting, value, expected_names,
):
    cached_epub.extract(return_document_metadata=True)
    monkeypatch.setenv(setting, value)

    rebuilt = cached_epub.extract(return_document_metadata=True)

    assert cached_epub.read_archive.call_count == 2
    assert [doc["filename"] for doc in rebuilt] == expected_names
    assert cached_epub.extract(return_document_metadata=True) == rebuilt
    assert cached_epub.read_archive.call_count == 2


@pytest.mark.parametrize("broken_cache", ["{unfinished", "null", "{}", "[]"])
def test_epub_text_cache_rebuilds_after_cache_corruption(cached_epub, broken_cache):
    original = cached_epub.extract(return_document_metadata=True)
    cached_epub.cache.write_text(broken_cache, encoding="utf-8")

    assert cached_epub.extract(return_document_metadata=True) == original
    assert cached_epub.read_archive.call_count == 2
    assert cached_epub.extract(return_document_metadata=True) == original
    assert cached_epub.read_archive.call_count == 2


def test_epub_text_cache_rejects_valid_json_with_same_length_text_corruption(cached_epub):
    original = cached_epub.extract(return_document_metadata=True)
    saved = cached_epub.cache.read_bytes()
    corrupted = saved.replace(b"Alice", b"Clara")
    assert corrupted != saved
    assert len(corrupted) == len(saved)
    json.loads(corrupted)
    cached_epub.cache.write_bytes(corrupted)

    assert cached_epub.extract(return_document_metadata=True) == original
    assert cached_epub.read_archive.call_count == 2
    assert cached_epub.extract(return_document_metadata=True) == original
    assert cached_epub.read_archive.call_count == 2


def test_epub_source_change_during_preparation_prevents_cache_publication(cached_epub, monkeypatch):
    _set_zip_comment_preserving_stat(cached_epub.source, b"A")
    classify = extractor._classify_glossary_html_document
    changed = False

    def change_source_after_first_document(raw):
        nonlocal changed
        result = classify(raw)
        if not changed:
            _set_zip_comment_preserving_stat(cached_epub.source, b"B")
            changed = True
        return result

    monkeypatch.setattr(extractor, "_classify_glossary_html_document", change_source_after_first_document)
    prepared = cached_epub.extract(return_document_metadata=True)

    assert changed
    assert len(prepared) == 4
    assert not cached_epub.cache.exists()

    monkeypatch.setattr(extractor, "_classify_glossary_html_document", classify)
    assert cached_epub.extract(return_document_metadata=True) == prepared
    assert cached_epub.read_archive.call_count == 2
    assert cached_epub.extract(return_document_metadata=True) == prepared
    assert cached_epub.read_archive.call_count == 2


def test_epub_cache_commit_failure_preserves_previous_cache_and_removes_temporary_file(
    cached_epub, monkeypatch,
):
    _set_zip_comment_preserving_stat(cached_epub.source, b"A")
    original = cached_epub.extract(return_document_metadata=True)
    previous_cache = cached_epub.cache.read_bytes()
    _set_zip_comment_preserving_stat(cached_epub.source, b"B")
    files_before = set(cached_epub.cache.parent.iterdir())
    replace = extractor.os.replace
    attempted_temporary_paths = []

    def fail_cache_commit(source, destination):
        if os.path.abspath(destination) == os.path.abspath(cached_epub.cache):
            attempted_temporary_paths.append(source)
            raise PermissionError("cache commit fixture")
        return replace(source, destination)

    monkeypatch.setattr(extractor.os, "replace", fail_cache_commit)
    prepared = cached_epub.extract(return_document_metadata=True)

    assert prepared == original
    assert cached_epub.read_archive.call_count == 2
    assert len(attempted_temporary_paths) == 1
    assert all(not os.path.exists(path) for path in attempted_temporary_paths)
    assert set(cached_epub.cache.parent.iterdir()) == files_before
    assert cached_epub.cache.read_bytes() == previous_cache

    monkeypatch.setattr(extractor.os, "replace", replace)
    assert cached_epub.extract(return_document_metadata=True) == original
    assert cached_epub.read_archive.call_count == 3
    assert cached_epub.extract(return_document_metadata=True) == original
    assert cached_epub.read_archive.call_count == 3


def test_epub_preparation_honors_explicit_stop_callback_with_no_global_stop(cached_epub, monkeypatch):
    stopped = False
    classify = extractor._classify_glossary_html_document

    def stop_after_first_document(raw):
        nonlocal stopped
        result = classify(raw)
        stopped = True
        return result

    monkeypatch.setattr(extractor, "_classify_glossary_html_document", stop_after_first_document)
    partial = cached_epub.extract(return_document_metadata=True, stop_check=lambda: stopped)

    assert extractor.is_stop_requested() is False
    assert len(partial) == 1
    assert not cached_epub.cache.exists()

    monkeypatch.setattr(extractor, "_classify_glossary_html_document", classify)
    full = cached_epub.extract(return_document_metadata=True, stop_check=lambda: False)
    assert len(full) == 4
    assert cached_epub.read_archive.call_count == 2


@pytest.mark.parametrize("existing_cache", [False, True], ids=["first-extraction", "stale-cache"])
def test_cancelled_epub_preparation_never_caches_partial_documents(
    cached_epub, monkeypatch, existing_cache,
):
    previous_cache = None
    if existing_cache:
        _set_zip_comment_preserving_stat(cached_epub.source, b"A")
        cached_epub.extract(return_document_metadata=True)
        previous_cache = cached_epub.cache.read_bytes()
        _set_zip_comment_preserving_stat(cached_epub.source, b"B")

    stopped = False
    classify = extractor._classify_glossary_html_document

    def stop_after_first_document(raw):
        nonlocal stopped
        result = classify(raw)
        stopped = True
        return result

    monkeypatch.setattr(extractor, "is_stop_requested", lambda: stopped)
    monkeypatch.setattr(extractor, "_classify_glossary_html_document", stop_after_first_document)
    partial = cached_epub.extract(return_document_metadata=True)

    assert len(partial) < 4
    if previous_cache is None:
        assert not cached_epub.cache.exists()
    else:
        assert cached_epub.cache.read_bytes() == previous_cache

    monkeypatch.setattr(extractor, "is_stop_requested", lambda: False)
    monkeypatch.setattr(extractor, "_classify_glossary_html_document", classify)
    assert len(cached_epub.extract(return_document_metadata=True)) == 4
    assert cached_epub.read_archive.call_count == (3 if existing_cache else 2)


def test_corrupted_chapter_does_not_make_partial_epub_result_reusable(cached_epub, monkeypatch):
    classify = extractor._classify_glossary_html_document
    calls = 0

    def fail_first_document(raw):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("damaged chapter fixture")
        return classify(raw)

    monkeypatch.setattr(extractor, "_classify_glossary_html_document", fail_first_document)
    partial = cached_epub.extract(return_document_metadata=True)
    assert len(partial) == 3
    assert not cached_epub.cache.exists()

    monkeypatch.setattr(extractor, "_classify_glossary_html_document", classify)
    full = cached_epub.extract(return_document_metadata=True)
    assert len(full) == 4
    assert full[0]["text"] == "Alice entered the tower."
    assert cached_epub.read_archive.call_count == 2
    assert cached_epub.extract(return_document_metadata=True) == full
    assert cached_epub.read_archive.call_count == 2


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
