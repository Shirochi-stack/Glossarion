"""Keep EPUB file progress distinct from filename-derived chapter labels."""

import json
import os
import sys
from types import SimpleNamespace

import pytest

import extract_glossary_from_epub as extractor


@pytest.fixture(autouse=True)
def isolated_numbering(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.chdir(tmp_path)
    for name in (
        "_GLOSSARY_CHAPTER_POSITIONS", "_GLOSSARY_CHAPTER_NUMBERS",
        "_GLOSSARY_CHAPTER_FILENAMES", "_GLOSSARY_QA_ISSUES_FOUND",
    ):
        monkeypatch.setattr(extractor, name, {})
    monkeypatch.setattr(extractor, "_GLOSSARY_TOTAL_CHAPTERS", 0)
    for name, value in {
        "EPUB_PATH": str(tmp_path / "book.epub"),
        "SPECIAL_FILE_KEYWORDS": "cover,toc,notice",
        "SPECIAL_FILE_EXACT": "index",
        "DIRECT_TEXT_ACTIVE": "0",
        "TRANSLATION_CANCELLED": "0",
        "GRACEFUL_STOP": "0",
        "GRACEFUL_STOP_COMPLETED": "0",
        "THREAD_SUBMISSION_DELAY_SECONDS": "0",
        "ASSISTANT_PROMPT": "",
    }.items():
        monkeypatch.setenv(name, value)


def _book_context():
    names = ["cover.xhtml", "info.xhtml", "toc.xhtml", "notice0007.xhtml"] + [
        f"chapter{number:04d}.xhtml" for number in range(1, 174)
    ]
    filenames = dict(enumerate(names))
    positions = {index: index + 1 for index in filenames}
    return extractor.make_glossary_progress_context(
        chapter_filenames=filenames,
        chapter_positions=positions,
        chapter_numbers=extractor._glossary_chapter_display_number_map(filenames, positions),
        total_chapters=len(names),
    )


@pytest.mark.parametrize(("index", "chapter"), [(0, 0), (3, 0), (29, 26), (176, 173)])
def test_epub_log_labels_separate_spine_position_from_chapter(index, chapter):
    context = _book_context()

    assert extractor._glossary_chapter_actual_num(index, context=context) == chapter
    assert extractor._glossary_chapter_log_label(index, context=context) == (
        f"[Spine {index + 1}/177 — Chapter {chapter}]"
    )


def test_log_label_uses_file_count_and_ordinal_not_range_positions():
    context = extractor.make_glossary_progress_context(
        chapter_filenames={0: "chapter0042.xhtml", 1: "chapter0043.xhtml"},
        chapter_positions={0: 42, 1: 43}, chapter_numbers={0: 42, 1: 43}, total_chapters=2,
    )

    assert extractor._glossary_chapter_log_label(0, context=context) == "[Spine 1/2 — Chapter 42]"
    assert extractor._glossary_chapter_log_label(1, context=context) == "[Spine 2/2 — Chapter 43]"


def test_explicit_book_contexts_do_not_share_display_numbers(monkeypatch):
    first = _book_context()
    second = extractor.make_glossary_progress_context(
        chapter_filenames={29: "chapter0098.xhtml"}, chapter_positions={29: 500},
        chapter_numbers={29: 98}, total_chapters=40,
    )
    monkeypatch.setattr(extractor, "_GLOSSARY_CHAPTER_NUMBERS", {29: 999})
    monkeypatch.setattr(extractor, "_GLOSSARY_TOTAL_CHAPTERS", 999)

    assert extractor._glossary_chapter_log_label(29, context=first) == "[Spine 30/177 — Chapter 26]"
    assert extractor._glossary_chapter_log_label(29, context=second) == "[Spine 30/40 — Chapter 98]"
    assert extractor._glossary_chapter_log_label(29, context=first) == "[Spine 30/177 — Chapter 26]"


def test_standalone_text_log_has_no_spine_counter(monkeypatch):
    monkeypatch.setenv("EPUB_PATH", "book.txt")
    context = extractor.make_glossary_progress_context(
        chapter_filenames={0: "section0042.txt"}, chapter_numbers={0: 42}, total_chapters=1,
    )

    assert extractor._glossary_chapter_log_label(0, context=context) == "[Chapter 42]"


def test_numbered_specials_and_split_restarts_match_translation_display():
    names = dict(enumerate([
        "notice0099.xhtml", "info.xhtml", "chapter0026.xhtml",
        "part0003_split_000.xhtml", "part0003_split_001.xhtml",
    ]))

    assert extractor._glossary_chapter_display_number_map(names) == {
        0: 0, 1: 0, 2: 26, 3: 27, 4: 28,
    }


def test_api_worker_defaults_to_display_chapter_number(monkeypatch, capsys):
    monkeypatch.setattr(extractor, "_GLOSSARY_CHAPTER_NUMBERS", {29: 26})
    sent = []

    def send(**kwargs):
        sent.append(kwargs)
        return "[]", "stop", None

    monkeypatch.setattr(extractor, "send_with_interrupt", send)
    result = extractor.process_single_chapter_api_call(
        29, "Example text", [{"role": "user", "content": "Example text"}],
        SimpleNamespace(), 0.1, 1000, lambda: False,
    )

    assert result["idx"] == 29
    assert sent[0]["chapter_idx"] == 29
    assert sent[0]["chapter_num"] == 26
    output = capsys.readouterr().out
    assert "Chapter 26" in output
    assert "Chapter 30" not in output


def test_send_retry_uses_actual_chapter_and_source_filename(monkeypatch, capsys):
    monkeypatch.setattr(extractor, "_GLOSSARY_CHAPTER_NUMBERS", {29: 26})
    monkeypatch.setattr(extractor, "_GLOSSARY_CHAPTER_FILENAMES", {29: "chapter0026.xhtml"})
    monkeypatch.setenv("TIMEOUT_RETRY_ATTEMPTS", "1")
    monkeypatch.setenv("SEND_INTERVAL_SECONDS", "0")
    contexts = []
    calls = []

    def send(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise extractor.UnifiedClientError("Synthetic transport timed out")
        return "[]", "stop"

    client = SimpleNamespace(send=send, set_chapter_context=lambda **kw: contexts.append(kw))
    result = extractor.send_with_interrupt(
        [{"role": "user", "content": "Extract terms"}], client, 0.1, 1000,
        lambda: False, chapter_idx=29,
    )

    assert result[:2] == ("[]", "stop")
    assert [context["chapter"] for context in contexts] == [26, 26]
    output = capsys.readouterr().out
    assert "Chapter 26 · chapter0026.xhtml" in output
    assert "retrying (1/1)" in output
    assert "Chapter 30" not in output


@pytest.mark.parametrize("split", [False, True])
def test_split_worker_preserves_display_number_for_every_chunk(monkeypatch, capsys, split):
    monkeypatch.setattr(extractor, "_GLOSSARY_CHAPTER_NUMBERS", {29: 26})
    sent = []

    def process(index, chapter, *args, **kwargs):
        sent.append((index, kwargs["chapter_num"]))
        return {"idx": index, "data": [], "resp": "[]", "chap": chapter, "error": None}

    monkeypatch.setattr(extractor, "process_single_chapter_api_call", process)
    splitter = SimpleNamespace(
        count_tokens=lambda _text: 100,
        split_chapter=lambda *_: [("<p>first</p>", 1, 2), ("<p>second</p>", 2, 2)],
    )
    extractor.process_single_chapter_with_split(
        29, "Example text", lambda text: ("Extract terms", text), splitter, 50,
        split, False, [], 0, False, SimpleNamespace(), 0.1, 1000, lambda: False,
    )

    assert sent == [(29, 26)] * (2 if split else 1)
    output = capsys.readouterr().out
    assert "Chapter 30" not in output
    if split:
        assert "chunk 1/2 of Chapter 26" in output
        assert "chunk 2/2 of Chapter 26" in output


@pytest.mark.parametrize("batch", [False, True], ids=["sequential", "batch"])
def test_main_reports_html_file_totals_and_actual_chapter_labels(monkeypatch, tmp_path, capsys, batch):
    monkeypatch.setattr(sys, "argv", ["extract_glossary_from_epub.py"])
    for name in (
        "PROGRESS_FILE", "_GLOSSARY_OUTPUT_FILE", "BOOK_TITLE_RAW", "BOOK_TITLE_TRANSLATED",
        "BOOK_TITLE_PRESENT", "BOOK_TITLE_VALUE", "GLOSSARY_SOURCE_LANGUAGE",
        "GLOSSARY_SOURCE_LANGUAGE_PATH", "_GLOSSARY_SOURCE_LANGUAGE_LOADED",
        "_GLOSSARY_SOURCE_LANGUAGE_LOGGED", "GLOSSARY_SOURCE_SCRIPT",
        "GLOSSARY_SOURCE_SCRIPT_IS_CJK", "_GLOSSARY_SOURCE_SCRIPT_READY", "_GLOSSARY_SOURCE_SCRIPT_LOGGED",
    ):
        monkeypatch.setattr(extractor, name, getattr(extractor, name))
    for name, value in {
        "OUTPUT_PATH": str(tmp_path / "glossary.json"),
        "GLOSSARY_INCLUDE_BOOK_TITLE": "0", "USE_SPINE_ORDER": "0",
        "TRANSLATE_SPECIAL_FILES": "1", "CHAPTER_RANGE": "",
        "BATCH_TRANSLATION": "1" if batch else "0", "BATCH_SIZE": "1", "BATCH_GROUP_SIZE": "1",
        "SEND_INTERVAL_SECONDS": "0", "GLOSSARY_COMPRESSION_FACTOR": "1.5",
        "GLOSSARY_REFINEMENT_COMPRESSION_FACTOR": "1.5", "GLOSSARY_TEMPERATURE": "0.1",
        "GLOSSARY_CONTEXT_LIMIT": "0", "GLOSSARY_CUSTOM_FIELDS": "[]",
        "GLOSSARY_ENABLE_ANTI_DUPLICATE": "0", "CONTEXTUAL": "0",
        "GLOSSARY_REQUEST_MERGING_ENABLED": "0", "REQUEST_MERGING_ENABLED": "0",
        "GLOSSARY_ENABLE_CHAPTER_SPLIT": "0", "GLOSSARY_REFINEMENT_ENABLED": "0",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(extractor, "_glossary_is_graceful_stop_active", lambda: False)
    monkeypatch.setattr(extractor, "is_stop_requested", lambda: False)
    monkeypatch.setattr(extractor, "load_config", lambda _: {"model": "test-model"})
    monkeypatch.setattr(extractor, "_log_assistant_prompt_once", lambda: None)
    monkeypatch.setattr(extractor, "_set_glossary_source_language_from_metadata", lambda *a, **kw: None)
    monkeypatch.setattr(extractor, "_extract_raw_title_from_epub", lambda _: None)
    monkeypatch.setattr(extractor, "_extract_translated_title_from_metadata", lambda *a: None)
    monkeypatch.setattr(extractor, "create_client_with_multi_key_support", lambda *a: SimpleNamespace())
    monkeypatch.setattr(extractor, "_effective_glossary_output_limit", lambda *a: 128000)
    monkeypatch.setattr(extractor, "ChapterSplitter", lambda **kw: SimpleNamespace(count_tokens=lambda _: 50))
    names = ["cover.xhtml", "info.xhtml", "chapter0026.xhtml", "chapter0027.xhtml", "chapter0028.xhtml"]
    documents = [{"filename": name, "text": "Alice entered the tower.", "structural_kind": ""} for name in names]
    monkeypatch.setattr(extractor, "extract_chapters_from_epub", lambda *a, **kw: documents)
    monkeypatch.setattr(extractor, "load_progress", lambda **kw: {
        "completed": [0, 1], "failed": [], "in_progress": [], "chapters": {},
    })
    monkeypatch.setattr(extractor, "validate_extracted_entry", lambda _: True)
    requests = []

    def send(*args, **kwargs):
        requests.append(kwargs)
        callback = kwargs.get("before_send_callback")
        if callback:
            callback()
        return json.dumps([{"type": "term", "raw_name": "Alice", "translated_name": "Alice"}]), "stop", None

    monkeypatch.setattr(extractor, "send_with_interrupt", send)
    extractor.main()

    output = capsys.readouterr().out
    assert "Processing 3 out of 5 HTML files" in output
    assert requests
    assert [request["chapter_num"] for request in requests] == [26, 27, 28]
    if batch:
        assert "[Spine 3/5 — Chapter 26]" in output
    else:
        assert "Processing Spine 3/5 — Chapter 26 (chapter0026.xhtml)" in output
    assert "Chapter 26/28" not in output
