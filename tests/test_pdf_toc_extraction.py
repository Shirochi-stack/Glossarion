import json
import sys
import types
from pathlib import Path

from pdf_extractor import (
    build_pdf_toc_section_plan,
    extract_pdf_toc_section_plan,
    group_pdf_pages_by_toc,
    group_pdf_page_texts_by_toc,
)


def test_toc_plan_uses_deepest_bookmark_and_keeps_front_matter():
    toc = [
        [1, "Book title", 3],
        [2, "Chapter One", 3],
        [2, "Chapter Two", 6],
        [2, "External link", -1],
        [2, "Past the end", 99],
    ]

    plan = build_pdf_toc_section_plan(toc, total_pages=8)

    assert [section["title"] for section in plan] == [
        "Front Matter",
        "Chapter One",
        "Chapter Two",
    ]
    assert [
        (section["start_page"], section["end_page"])
        for section in plan
    ] == [(1, 2), (3, 5), (6, 8)]


def test_real_pymupdf_outline_is_read_as_section_ranges(tmp_path):
    import pytest

    fitz = pytest.importorskip("fitz")
    pdf_path = tmp_path / "outlined.pdf"
    doc = fitz.open()
    for page_num in range(1, 7):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {page_num}")
    doc.set_toc([[1, "Opening", 1], [1, "Middle", 3], [1, "Ending", 5]])
    doc.save(pdf_path)
    doc.close()

    plan = extract_pdf_toc_section_plan(str(pdf_path))

    assert [section["title"] for section in plan] == [
        "Opening",
        "Middle",
        "Ending",
    ]
    assert [(section["start_page"], section["end_page"]) for section in plan] == [
        (1, 2),
        (3, 4),
        (5, 6),
    ]


def test_toc_sections_preserve_page_boundaries_and_styles():
    pages = [
        (
            page_num,
            (
                "<html><head><style>.page-text{color:black}</style></head>"
                f"<body><p class='page-text'>Page {page_num}</p></body></html>"
            ),
        )
        for page_num in range(1, 7)
    ]
    toc = [[1, "Opening", 1], [1, "Middle", 3], [1, "Ending", 5]]

    grouped = group_pdf_pages_by_toc(
        "unused.pdf",
        pages,
        toc_entries=toc,
        total_pages=6,
    )

    assert [section["title"] for section in grouped] == [
        "Opening",
        "Middle",
        "Ending",
    ]
    assert "Page 1" in grouped[0]["html"]
    assert "Page 2" in grouped[0]["html"]
    assert "Page 3" not in grouped[0]["html"]
    assert grouped[0]["html"].count("pdf-toc-page-break") >= 1
    assert grouped[0]["html"].count(".page-text{color:black}") == 1


def test_missing_outline_requests_legacy_page_fallback():
    assert group_pdf_pages_by_toc(
        "unused.pdf",
        [(1, "<p>One</p>"), (2, "<p>Two</p>")],
        toc_entries=[],
        total_pages=2,
    ) == []


def test_glossary_text_uses_the_same_toc_ranges():
    grouped = group_pdf_page_texts_by_toc(
        "unused.pdf",
        [(1, "one"), (2, "two"), (4, "four")],
        toc_entries=[[1, "First", 1], [1, "Second", 4]],
        total_pages=4,
    )

    assert [section["text"] for section in grouped] == ["one\n\ntwo", "four"]
    assert [(section["start_page"], section["end_page"]) for section in grouped] == [
        (1, 3),
        (4, 4),
    ]


def test_worker_serializes_toc_sections(tmp_path, monkeypatch):
    import _pdf_extraction_worker as worker

    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"synthetic")
    output_dir = tmp_path / "out"
    result_path = tmp_path / "result.json"
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({
            "pdf_path": str(pdf_path),
            "output_dir": str(output_dir),
            "render_mode": "xhtml",
            "use_toc_sections": True,
            "extract_images": False,
            "generate_css": False,
            "result_path": str(result_path),
        }),
        encoding="utf-8",
    )

    fake_extractor = types.ModuleType("pdf_extractor")
    fake_extractor.extract_pdf_with_formatting = lambda *_args, **_kwargs: (
        [(1, "<p>one</p>"), (2, "<p>two</p>"), (3, "<p>three</p>")],
        {},
    )
    fake_extractor.generate_css_from_pdf = lambda _path: ""
    fake_extractor.group_pdf_pages_by_toc = lambda _path, _pages: [
        {
            "num": 1,
            "title": "First",
            "level": 1,
            "start_page": 1,
            "end_page": 2,
            "page_count": 2,
            "html": "<p>one two</p>",
        },
        {
            "num": 2,
            "title": "Second",
            "level": 1,
            "start_page": 3,
            "end_page": 3,
            "page_count": 1,
            "html": "<p>three</p>",
        },
    ]
    monkeypatch.setitem(sys.modules, "pdf_extractor", fake_extractor)

    result = worker._run_pdf_extraction_inner(str(config_path))

    assert result["success"] is True
    assert result["page_count"] == 3
    assert result["entry_count"] == 2
    assert result["separation_mode"] == "toc"
    assert [section["title"] for section in result["section_info"]] == [
        "First",
        "Second",
    ]

    legacy_result_path = tmp_path / "legacy-result.json"
    config_path.write_text(
        json.dumps({
            "pdf_path": str(pdf_path),
            "output_dir": str(output_dir),
            "render_mode": "xhtml",
            "use_toc_sections": False,
            "extract_images": False,
            "generate_css": False,
            "result_path": str(legacy_result_path),
        }),
        encoding="utf-8",
    )
    fake_extractor.group_pdf_pages_by_toc = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("legacy extraction must not call TOC grouping")
    )

    legacy_result = worker._run_pdf_extraction_inner(str(config_path))

    assert legacy_result["separation_mode"] == "pages"
    assert legacy_result["page_count"] == 3
    assert legacy_result["entry_count"] == 3
    assert legacy_result["section_info"] == []


def test_toc_section_can_split_when_it_exceeds_token_budget(tmp_path, monkeypatch):
    from txt_processor import TextFileProcessor

    class Splitter:
        @staticmethod
        def count_tokens(_content):
            return 2000

        @staticmethod
        def split_chapter(_content, _available, filename=None):
            assert filename.endswith(".pdf")
            return [("<p>part one</p>", 1, 2), ("<p>part two</p>", 2, 2)]

    processor = object.__new__(TextFileProcessor)
    processor.file_path = str(tmp_path / "book.pdf")
    processor.output_dir = str(tmp_path / "out")
    processor.cache_suffix = ""
    processor.chapter_splitter = Splitter()
    monkeypatch.setenv("MAX_OUTPUT_TOKENS", "1500")
    monkeypatch.setenv("COMPRESSION_FACTOR", "1")

    chapters = processor._process_chapters_for_splitting([{
        "num": 1,
        "title": "Long chapter",
        "content": "<html><body><p>long</p></body></html>",
        "is_html": True,
        "pdf_toc_section": True,
        "allow_token_splitting": True,
        "pdf_start_page": 1,
        "pdf_end_page": 20,
    }])

    assert [chapter["filename"] for chapter in chapters] == [
        "pdf_section_1_0.html",
        "pdf_section_1_1.html",
    ]
    assert all(chapter["pdf_toc_section"] for chapter in chapters)


def test_pdf_progress_reconciliation_removes_old_page_rows(monkeypatch):
    from TransateKRtoEN import FileUtilities, ProgressManager

    monkeypatch.setenv("RETAIN_SOURCE_EXTENSION", "0")

    manager = object.__new__(ProgressManager)
    manager.prog = {
        "chapters": {
            "1": {
                "actual_num": 1,
                "output_file": "response_001.html",
                "content_hash": "old-page-one",
                "status": "completed",
            },
            "2": {
                "actual_num": 2,
                "output_file": "response_pdf_section_2.html",
                "content_hash": "section-two",
                "status": "completed",
            },
            "61": {
                "actual_num": 61,
                "output_file": "response_061.html",
                "content_hash": "old-page-sixty-one",
                "status": "completed",
            },
            "artifact": {
                "actual_num": 999,
                "output_file": "translated_headers.json",
                "status": "completed",
                "special_type": "translated_headers",
            },
        },
        "chapter_chunks": {"1": {}, "2": {}, "61": {}},
    }
    current_sections = [
        {
            "num": 1,
            "title": "First",
            "body": "one",
            "content_hash": "section-one",
            "filename": "pdf_section_1.html",
            "pdf_toc_section": True,
            "is_chunk": False,
        },
        {
            "num": 2,
            "title": "Second",
            "body": "two",
            "content_hash": "section-two",
            "filename": "pdf_section_2.html",
            "pdf_toc_section": True,
            "is_chunk": False,
        },
    ]

    removed = manager.reconcile_pdf_chapter_entries(current_sections)

    assert removed == 2
    assert set(manager.prog["chapters"]) == {"2", "artifact"}
    assert set(manager.prog["chapter_chunks"]) == {"2"}
    assert FileUtilities.create_chapter_filename(current_sections[0], 1) == (
        "response_pdf_section_1.html"
    )
    assert FileUtilities.create_chapter_filename(
        {**current_sections[0], "num": 1.1, "is_chunk": True},
        1.1,
    ) == "response_pdf_section_1_1.html"


def test_pdf_toc_setting_is_defaulted_exported_and_persisted():
    root = Path(__file__).resolve().parents[1]
    gui_source = (root / "src" / "translator_gui.py").read_text(encoding="utf-8")
    settings_source = (root / "src" / "other_settings.py").read_text(encoding="utf-8")

    assert "self.pdf_use_toc_sections_var = self.config.get('pdf_use_toc_sections', True)" in gui_source
    assert "'PDF_USE_TOC_SECTIONS': '1' if getattr(self, 'pdf_use_toc_sections_var', True) else '0'" in gui_source
    assert "('pdf_use_toc_sections', ['pdf_use_toc_sections_var'], True, bool)" in gui_source
    assert "Use PDF table of contents for sections" in settings_source
    assert "legacy page-by-page extraction" in settings_source
