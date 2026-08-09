import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from pdf_bookmarks import (
    remove_pdf_source_page_break_markers,
    replace_with_chapter_bookmarks,
)
from pdf_extractor import (
    build_pdf_toc_section_plan,
    extract_pdf_toc_section_plan,
    group_pdf_pages_by_toc,
    group_pdf_page_texts_by_toc,
)


def test_progress_manager_routes_pdf_html_entries_to_workspace_reader():
    source = Path(__file__).resolve().parents[1] / "src" / "Retranslation_GUI.py"
    text = source.read_text(encoding="utf-8")
    pdf_branch = text.index('workspace_source.lower().endswith(".pdf")')
    epub_gate = text.index("source_epubs = _source_epub_candidates()", pdf_branch)
    constructor = text.index("workspace_dir=data['output_dir']", pdf_branch)

    assert pdf_branch < constructor < epub_gate
    assert "initial_show_raw=False" in text[pdf_branch:epub_gate]


class _FakeBookmarkPage:
    def __init__(self, anchors=None, bookmarks=None):
        self.anchors = anchors or {}
        self.bookmarks = list(bookmarks or [])


def test_legacy_toc_source_page_break_marker_is_removed_before_pdf_render():
    html = (
        "<p>Page one text</p>"
        '<div data-next-pdf-page="2" '
        'class="page-break pdf-toc-page-break"></div>'
        "<p>Page two text</p>"
    )

    cleaned = remove_pdf_source_page_break_markers(html)

    assert "pdf-toc-page-break" not in cleaned
    assert "Page one text" in cleaned
    assert "Page two text" in cleaned


def test_pdf_bookmarks_replace_sentences_with_one_entry_per_html_file():
    pages = [
        _FakeBookmarkPage(
            anchors={"chapter-1": (10, 20)},
            bookmarks=[
                (1, "Chapter One", (10, 20), "open"),
                (2, "First sentence", (10, 40), "open"),
                (3, "Second sentence", (10, 60), "open"),
            ],
        ),
        _FakeBookmarkPage(
            anchors={"chapter-2": (10, 20)},
            bookmarks=[
                (1, "Another heading", (10, 20), "open"),
                (4, "Paragraph promoted by EPUB CSS", (10, 40), "open"),
            ],
        ),
    ]

    added = replace_with_chapter_bookmarks(
        pages,
        [
            ("chapter-1.html", 1, "Chapter One"),
            ("chapter-2.html", 2, "Chapter Two"),
        ],
    )

    assert added == 2
    assert pages[0].bookmarks == [(1, "Chapter One", (10, 20), "open")]
    assert pages[1].bookmarks == [(1, "Chapter Two", (10, 20), "open")]


def test_pdf_worker_writes_only_html_file_bookmarks(tmp_path):
    fitz = pytest.importorskip("fitz")
    output_dir = tmp_path / "output"
    images_dir = output_dir / "images"
    css_dir = output_dir / "css"
    images_dir.mkdir(parents=True)
    css_dir.mkdir()
    (css_dir / "hostile-outline.css").write_text(
        "p, div { bookmark-level: 4 !important; }"
        ".page-break { page-break-before: always; break-before: page; }",
        encoding="utf-8",
    )
    (output_dir / "chapter-1.html").write_text(
        """<html><body>
        <h1 style="bookmark-level: 2 !important">First sentence</h1>
        <p style="bookmark-level: 3 !important">Second sentence</p>
        <div class="page-break pdf-toc-page-break"></div>
        <p>Third sentence</p>
        </body></html>""",
        encoding="utf-8",
    )
    (output_dir / "chapter-2.html").write_text(
        """<html><body>
        <h2 style="bookmark-level: 2 !important">Fourth sentence</h2>
        <p style="bookmark-level: 5 !important">Fifth sentence</p>
        </body></html>""",
        encoding="utf-8",
    )
    config_path = tmp_path / "pdf-config.json"
    config_path.write_text(
        json.dumps({
            "output_dir": str(output_dir),
            "images_dir": str(images_dir),
            "css_dir": str(css_dir),
            "html_files": ["chapter-1.html", "chapter-2.html"],
            "chapter_titles_info": {
                "1": ["Chapter One", 1.0, "chapter-1.html"],
                "2": ["Chapter Two", 1.0, "chapter-2.html"],
            },
            "processed_images": {},
            "cover_file": None,
            "metadata": {"title": "Outline Fixture"},
            "env_vars": {
                "PDF_PAGE_NUMBERS": "0",
                "PDF_GENERATE_TOC": "1",
                "PDF_TOC_PAGE_NUMBERS": "0",
                "PDF_RENDER_BATCH_SIZE": "50",
                "PDF_FAST_RENDERING": "0",
                "ENABLE_IMAGE_COMPRESSION": "0",
                "DEDUPLICATE_TOC": "0",
            },
        }),
        encoding="utf-8",
    )
    worker_env = os.environ.copy()
    for key in ("FONTCONFIG_FILE", "FONTCONFIG_PATH", "FC_CONFIG_FILE"):
        worker_env.pop(key, None)
    worker_path = Path(__file__).parents[1] / "src" / "_pdf_worker.py"

    result = subprocess.run(
        [sys.executable, str(worker_path), str(config_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=worker_env,
        timeout=60,
        check=False,
    )

    # Some Windows WeasyPrint builds exit with a native-library teardown code
    # after emitting a successful result and closing the complete PDF. The
    # worker protocol's RESULT record and the reopened file are authoritative.
    assert '"success": true' in result.stdout.lower(), result.stdout + result.stderr
    pdf_files = list(output_dir.glob("*.pdf"))
    assert len(pdf_files) == 1
    with fitz.open(pdf_files[0]) as document:
        outline = document.get_toc(simple=True)
        assert document.page_count == 3
        preview = document[0].get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
        assert preview.width > 0 and preview.height > 0
    assert [(level, title) for level, title, _page in outline] == [
        (1, "Chapter One"),
        (1, "Chapter Two"),
    ]


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


def test_toc_sections_preserve_source_markers_without_hard_page_breaks():
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
    assert grouped[0]["html"].count("data-pdf-page=") == 2
    assert "pdf-toc-page-break" not in grouped[0]["html"]
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


def test_progress_manager_seeds_bookmark_rows_and_hides_source_sidecar(tmp_path):
    from Retranslation_GUI import RetranslationMixin, _is_progress_sidecar_entry

    pdf_path = tmp_path / "outlined.pdf"
    pdf_path.write_bytes(b"%PDF synthetic outline fixture")

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "source_epub.txt").write_text(
        str(pdf_path), encoding="utf-8"
    )
    prog = {
        "chapters": {
            "special_source_epub": {
                "actual_num": 0,
                "output_file": "source_epub.txt",
                "status": "completed",
                "original_basename": "source_epub.txt",
            }
        },
        "chapter_chunks": {},
    }

    mixin = object.__new__(RetranslationMixin)
    mixin.config = {"pdf_use_toc_sections": True}
    mixin.pdf_use_toc_sections_var = True
    mixin._is_special_file = lambda _name: False
    mixin._pdf_outline_progress_plan = lambda _path: [
        {
            "num": 1, "title": "Opening", "level": 1,
            "start_page": 1, "end_page": 2,
        },
        {
            "num": 2, "title": "Middle", "level": 1,
            "start_page": 3, "end_page": 4,
        },
        {
            "num": 3, "title": "Ending", "level": 1,
            "start_page": 5, "end_page": 6,
        },
    ]

    assert mixin._seed_pdf_outline_progress_entries(
        str(pdf_path), str(output_dir), prog
    )
    bookmark_rows = [
        entry
        for entry in prog["chapters"].values()
        if entry.get("pdf_toc_section")
    ]
    assert [entry["pdf_toc_title"] for entry in bookmark_rows] == [
        "Opening", "Middle", "Ending"
    ]
    assert [
        (entry["pdf_start_page"], entry["pdf_end_page"])
        for entry in bookmark_rows
    ] == [(1, 2), (3, 4), (5, 6)]
    assert all(entry["status"] == "not_translated" for entry in bookmark_rows)
    assert _is_progress_sidecar_entry(
        prog["chapters"]["special_source_epub"]
    )

    data = {
        "prog": prog,
        "output_dir": str(output_dir),
        "file_path": str(pdf_path),
        "show_special_files_state": True,
    }
    mixin._rebuild_chapter_display_info(data)
    assert len(data["chapter_display_info"]) == 3
    assert all(
        row["output_file"] != "source_epub.txt"
        for row in data["chapter_display_info"]
    )

    display, status = mixin._progress_list_display_text(
        data["chapter_display_info"][0],
        {"show_model_info_state": False},
        20,
        25,
    )
    assert status == "not_translated"
    assert "Opening" in display
    assert "Pages 1-2" in display
    assert "Chapter" not in display
