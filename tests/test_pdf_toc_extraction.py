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


def test_pdf_worker_rapid_compiler_uses_pdf_workers_and_process_jobs(tmp_path):
    fitz = pytest.importorskip("fitz")
    output_dir = tmp_path / "rapid-output"
    images_dir = output_dir / "images"
    css_dir = output_dir / "css"
    images_dir.mkdir(parents=True)
    css_dir.mkdir()
    html_files = []
    title_info = {}
    for chapter_number in range(1, 5):
        filename = f"chapter-{chapter_number}.html"
        html_files.append(filename)
        title_info[str(chapter_number)] = [
            f"Chapter {chapter_number}",
            1.0,
            filename,
        ]
        (output_dir / filename).write_text(
            f"<html><body><h1>Heading {chapter_number}</h1>"
            f"<p>Body {chapter_number}</p></body></html>",
            encoding="utf-8",
        )

    config_path = tmp_path / "rapid-pdf-config.json"
    config_path.write_text(
        json.dumps({
            "output_dir": str(output_dir),
            "images_dir": str(images_dir),
            "css_dir": str(css_dir),
            "html_files": html_files,
            "chapter_titles_info": title_info,
            "processed_images": {},
            "cover_file": None,
            "metadata": {"title": "Rapid Fixture"},
            "env_vars": {
                "PDF_PAGE_NUMBERS": "1",
                "PDF_PAGE_NUMBER_ALIGNMENT": "right",
                "PDF_GENERATE_TOC": "0",
                "PDF_RENDER_BATCH_SIZE": "50",
                "PDF_FAST_RENDERING": "0",
                "PDF_USE_RAPID_WORKSPACE_COMPILER": "1",
                "PDF_EXTRACTION_WORKERS": "2",
                "ENABLE_IMAGE_COMPRESSION": "0",
                "DEDUPLICATE_TOC": "0",
            },
        }),
        encoding="utf-8",
    )
    worker_env = os.environ.copy()
    worker_env["PYTHONIOENCODING"] = "utf-8"
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
        timeout=90,
        check=False,
    )

    assert '"success": true' in result.stdout.lower(), result.stdout + result.stderr
    assert "PDF_EXTRACTION_WORKERS=2 → 2" in result.stdout
    assert "2 bookmark-aware job(s) on 2 process worker(s)" in result.stdout
    pdf_files = list(output_dir.glob("*.pdf"))
    assert len(pdf_files) == 1
    with fitz.open(pdf_files[0]) as document:
        assert document.page_count == 4
        assert [row[1] for row in document.get_toc(simple=True)] == [
            "Chapter 1",
            "Chapter 2",
            "Chapter 3",
            "Chapter 4",
        ]
        assert "1" in document[0].get_text()
        assert "4" in document[3].get_text()


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


def test_pdf_range_preview_uses_bookmark_order_and_page_ranges(tmp_path):
    fitz = pytest.importorskip("fitz")
    from translator_gui import TranslatorGUI

    pdf_path = tmp_path / "outlined-preview.pdf"
    document = fitz.open()
    for page_num in range(1, 7):
        page = document.new_page()
        page.insert_text((72, 72), f"Page {page_num}")
    document.set_toc([
        [1, "Opening", 1],
        [1, "Middle", 3],
        [1, "Ending", 5],
    ])
    document.save(pdf_path)
    document.close()

    class Dummy:
        pdf_use_toc_sections_var = True
        pdf_render_mode_var = "fast_semantic"

    rows, scope, total = TranslatorGUI._get_pdf_range_entries_for_preview(
        Dummy(), str(pdf_path), 2, 3
    )

    assert scope == "bookmark"
    assert total == 3
    assert rows == [
        ("[002]", "Middle  •  Pages 3–4", False),
        ("[003]", "Ending  •  Pages 5–6", False),
    ]

    Dummy.pdf_use_toc_sections_var = False
    rows, scope, total = TranslatorGUI._get_pdf_range_entries_for_preview(
        Dummy(), str(pdf_path), 2, 3
    )
    assert scope == "page"
    assert total == 6
    assert rows == [
        ("[002]", "Page 2", False),
        ("[003]", "Page 3", False),
    ]


def test_pdf_bookmark_range_ignores_epub_spine_toggle(monkeypatch):
    from TransateKRtoEN import (
        _chapter_allowed_by_multipass_range,
        _pdf_bookmark_section_number,
        _vision_chapter_allowed_by_current_range,
    )

    selected = {
        "num": 99,
        "actual_chapter_num": 99,
        "pdf_toc_section": True,
        "pdf_section_num": 4,
    }
    excluded = {**selected, "pdf_section_num": 8}
    selected_page = {
        "num": 4,
        "actual_chapter_num": 4,
        "source_file": "book.pdf",
    }
    excluded_page = {**selected_page, "num": 8, "actual_chapter_num": 8}
    monkeypatch.setenv("CHAPTER_RANGE", "3-5")
    monkeypatch.setenv("USE_SPINE_ORDER", "1")

    assert _pdf_bookmark_section_number(selected) == 4
    assert _chapter_allowed_by_multipass_range(selected) is True
    assert _chapter_allowed_by_multipass_range(excluded) is False
    assert _chapter_allowed_by_multipass_range(selected_page) is True
    assert _chapter_allowed_by_multipass_range(excluded_page) is False
    assert _vision_chapter_allowed_by_current_range(selected, 0) is True
    assert _vision_chapter_allowed_by_current_range(excluded, 1) is False
    assert _vision_chapter_allowed_by_current_range(selected_page, 2) is True
    assert _vision_chapter_allowed_by_current_range(excluded_page, 3) is False


def test_pdf_failed_multipass_targets_follow_bookmark_preview_with_spine_toggle(
    tmp_path,
):
    from translator_gui import TranslatorGUI

    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"progress filtering does not open the source")

    class RangeEntry:
        @staticmethod
        def text():
            return "2-3"

    class Checked:
        @staticmethod
        def isChecked():
            return True

    class Dummy:
        config = {"chapter_range": "", "use_spine_order": False}
        chapter_range_entry = RangeEntry()
        use_spine_order_checkbox = Checked()
        translate_special_files_var = False

    failures = [
        {
            "source_path": str(pdf_path),
            "chapter": 99,
            "pdf_section_num": 2,
            "progress_key": "pdf:selected",
        },
        {
            "source_path": str(pdf_path),
            "chapter": 3,
            "pdf_section_num": 5,
            "progress_key": "pdf:outside",
        },
    ]

    scoped = TranslatorGUI._filter_translation_qa_failures_to_current_range(
        Dummy(), failures
    )

    assert [failure["progress_key"] for failure in scoped] == ["pdf:selected"]


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


def test_toc_section_stays_whole_for_shared_chunk_progress(tmp_path, monkeypatch):
    from txt_processor import TextFileProcessor

    class Splitter:
        @staticmethod
        def count_tokens(_content):
            return 2000

        @staticmethod
        def split_chapter(_content, _available, filename=None):
            raise AssertionError(
                "PDF bookmark sections must reach translation-time splitting whole"
            )

    processor = object.__new__(TextFileProcessor)
    processor.file_path = str(tmp_path / "book.pdf")
    processor.output_dir = str(tmp_path / "out")
    processor.cache_suffix = ""
    processor.chapter_splitter = Splitter()
    processor.pdf_render_mode = "legacy_layout"
    monkeypatch.setenv("MAX_OUTPUT_TOKENS", "1500")
    monkeypatch.setenv("COMPRESSION_FACTOR", "1")

    chapters = processor._process_chapters_for_splitting([{
        "num": 1,
        "title": "Long chapter",
        "content": "<html><body><p>long</p></body></html>",
        "is_html": True,
        "pdf_toc_section": True,
        "pdf_section_num": 1,
        "allow_token_splitting": True,
        "pdf_start_page": 1,
        "pdf_end_page": 20,
    }])

    assert len(chapters) == 1
    assert chapters[0]["filename"] == "pdf_section_1.html"
    assert chapters[0]["pdf_toc_section"] is True
    assert chapters[0]["pdf_section_num"] == 1
    assert chapters[0]["is_chunk"] is False
    assert chapters[0]["body"] == "<html><body><p>long</p></body></html>"


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
        "chapter_chunks": {
            "1": {},
            "2": {},
            "61": {},
            "old-page-one": {},
            "old-page-sixty-one": {},
            "section-two": {},
        },
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
    assert set(manager.prog["chapter_chunks"]) == {"2", "section-two"}
    assert FileUtilities.create_chapter_filename(current_sections[0], 1) == (
        "response_pdf_section_001.html"
    )
    very_long_title = "A" * 1000
    assert FileUtilities.create_chapter_filename(
        {**current_sections[0], "title": very_long_title},
        1,
    ) == "response_pdf_section_001.html"
    from pdf_output_naming import safe_pdf_book_filename_stem

    bounded_book_stem = safe_pdf_book_filename_stem("\U0001F680" * 300)
    assert len(bounded_book_stem.encode("utf-16-le")) // 2 <= 180


def test_pdf_bookmark_uses_one_progress_entry_and_chunk_qa_mapping(
        tmp_path, monkeypatch):
    from chapter_chunk_progress import wrap_chunk_html
    from scan_html_folder import _attach_chunk_results_to_scan
    from TransateKRtoEN import FileUtilities, ProgressManager

    monkeypatch.setenv("RETAIN_SOURCE_EXTENSION", "0")
    manager = ProgressManager(str(tmp_path))
    section_id = "stable-bookmark-id"
    budget = {
        "initial_output_token_limit": 12000,
        "cached_output_token_limit": 9000,
        "compression_factor": 2.0,
        "safety_margin": 500,
        "minimum_chunk_size": 1000,
        "initial_chunk_size": 5750,
        "cached_chunk_size": 4250,
    }
    content_hash = "whole-pdf-section-hash"
    chapter = {
        "num": 1,
        "actual_chapter_num": 1,
        "title": "Bookmark",
        "body": "whole source section",
        "filename": "pdf_section_1.html",
        "content_hash": content_hash,
        "pdf_toc_section": True,
        "pdf_section_id": section_id,
        "pdf_section_title": "Bookmark",
        "pdf_start_page": 1,
        "pdf_end_page": 20,
        "is_chunk": False,
    }
    output_name = FileUtilities.create_chapter_filename(chapter, 1)
    progress_key = f"pdf:{section_id}"
    manager.prog["chapters"][progress_key] = {
        "actual_num": 1,
        "content_hash": content_hash,
        "output_file": output_name,
        "status": "completed",
        "pdf_toc_section": True,
        "pdf_section_id": section_id,
        "pdf_progress_key": progress_key,
        "pdf_content_hash_version": 2,
    }
    manager.prepare_chapter_chunk_progress(
        content_hash, 2, budget, enabled=True
    )
    manager.record_chapter_chunk(
        content_hash,
        1,
        2,
        "<p>Good first chunk</p>",
        budget,
        source_text="source first chunk",
        model_name="provider/model-a",
    )
    failing_html = "<p>BADTWO second chunk</p>"
    manager.record_chapter_chunk(
        content_hash,
        2,
        2,
        failing_html,
        budget,
        source_text="source second chunk",
        model_name="provider/model-b",
    )
    manager.mark_chapter_chunk_progress_status(content_hash, "completed")
    output_path = tmp_path / output_name
    output_path.write_text(
        "\n".join((
            wrap_chunk_html(
                content_hash, 1, 2, "<p>Good first chunk</p>"
            ),
            wrap_chunk_html(content_hash, 2, 2, failing_html),
        )),
        encoding="utf-8",
    )
    scan_rows = [{
        "filename": output_name,
        "filepath": str(output_path),
        "file_index": 0,
        "chapter_num": 1,
        "issues": ["llm_token_issue: 'BADTWO'"],
        "qa_issue_previews": {},
        "duplicate_confidence": 0,
    }]

    assert manager.reconcile_pdf_chapter_entries([chapter]) == 0
    manager.migrate_to_content_hash([chapter])
    manager.save()

    assert set(manager.prog["chapters"]) == {progress_key}
    assert manager.prog["chapters"][progress_key]["output_file"] == (
        "response_pdf_section_001.html"
    )
    assert set(manager.prog["chapter_chunks"]) == {content_hash}

    logs = []
    assert _attach_chunk_results_to_scan(
        str(tmp_path),
        scan_rows,
        {},
        logs.append,
        progress_path=manager.PROGRESS_FILE,
    ) == 1
    assert scan_rows[0]["chunk_results"][0]["issues"] == []
    assert scan_rows[0]["chunk_results"][1]["issues"] == scan_rows[0]["issues"]


def test_pdf_toc_setting_is_defaulted_exported_and_persisted():
    root = Path(__file__).resolve().parents[1]
    gui_source = (root / "src" / "translator_gui.py").read_text(encoding="utf-8")
    settings_source = (root / "src" / "other_settings.py").read_text(encoding="utf-8")

    assert "self.pdf_use_toc_sections_var = self.config.get('pdf_use_toc_sections', True)" in gui_source
    assert "'PDF_USE_TOC_SECTIONS': '1' if getattr(self, 'pdf_use_toc_sections_var', True) else '0'" in gui_source
    assert "('pdf_use_toc_sections', ['pdf_use_toc_sections_var'], True, bool)" in gui_source
    assert "Use PDF table of contents for sections" in settings_source
    assert "legacy page-by-page extraction" in settings_source


def test_progress_manager_labels_pdf_api_chunks_like_epub(tmp_path):
    from Retranslation_GUI import RetranslationMixin
    from TransateKRtoEN import ProgressManager

    section_id = "stable-section"
    content_hash = "whole-section-hash"
    progress_key = f"pdf:{section_id}"
    output_file = "response_pdf_section_002.html"
    (tmp_path / output_file).write_text("translated", encoding="utf-8")
    manager = ProgressManager(str(tmp_path))
    manager.prog["chapters"][progress_key] = {
        "actual_num": 2,
        "content_hash": content_hash,
        "output_file": output_file,
        "status": "completed",
        "model_name": "provider/model-parent",
        "pdf_toc_section": True,
        "pdf_section_id": section_id,
        "pdf_progress_key": progress_key,
        "pdf_toc_title": "Middle",
        "pdf_start_page": 3,
        "pdf_end_page": 16,
    }
    budget = {
        "initial_output_token_limit": 12000,
        "cached_output_token_limit": 9000,
        "compression_factor": 2.0,
        "safety_margin": 500,
        "minimum_chunk_size": 1000,
        "initial_chunk_size": 5750,
        "cached_chunk_size": 4250,
    }
    for index, model in enumerate((
        "provider/model-a", "provider/model-b", "provider/model-c"
    ), 1):
        manager.record_chapter_chunk(
            content_hash,
            index,
            3,
            f"<p>chunk {index}</p>",
            budget,
            source_text=f"source {index}",
            model_name=model,
        )
    manager.mark_chapter_chunk_progress_status(content_hash, "completed")

    mixin = object.__new__(RetranslationMixin)
    mixin.config = {"pdf_use_toc_sections": True}
    mixin.pdf_use_toc_sections_var = True
    mixin._is_special_file = lambda _name: False
    data = {
        "prog": manager.prog,
        "output_dir": str(tmp_path),
        "file_path": str(tmp_path / "book.pdf"),
        "show_special_files_state": False,
        "show_model_info_state": True,
    }

    mixin._rebuild_chapter_display_info(data)

    rows = data["chapter_display_info"]
    assert len(rows) == 4
    assert rows[0]["progress_key"] == progress_key
    assert rows[0]["status"] == "completed"
    assert [row["chunk_index"] for row in rows[1:]] == [1, 2, 3]
    assert all(row["is_chunk_progress"] for row in rows[1:])
    assert all(row["pdf_toc_section"] for row in rows[1:])
    displays = [
        mixin._progress_list_display_text(row, data, 20, 25)[0]
        for row in rows
    ]
    assert displays[0].startswith("Section 002 |")
    assert displays[1].startswith("   ↳ Section 002 Chunk 1/3")
    assert displays[2].startswith("   ↳ Section 002 Chunk 2/3")
    assert displays[3].startswith("   ↳ Section 002 Chunk 3/3")
    assert "Pages 3-16" in displays[0]
    assert [mixin._progress_entry_model_name(row, data) for row in rows[1:]] == [
        "provider/model-a",
        "provider/model-b",
        "provider/model-c",
    ]


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
    assert "Section 001" in display
    assert "Opening" not in display
    assert "Pages 1-2" in display
    assert "Chapter" not in display


def test_progress_manager_merges_stable_pdf_rows_with_outline_seeds(tmp_path):
    from Retranslation_GUI import RetranslationMixin

    pdf_path = tmp_path / "outlined.pdf"
    pdf_path.write_bytes(b"%PDF synthetic outline fixture")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    section_id = "stable-bookmark-id"
    output_name = f"response_pdf_section_{section_id}.html"
    (output_dir / output_name).write_text("translated", encoding="utf-8")
    prog = {
        "chapters": {
            f"pdf:{section_id}": {
                "actual_num": 1,
                "content_hash": "old-hash",
                "output_file": output_name,
                "status": "completed",
            },
            "pdf:outline:1": {
                "actual_num": 1,
                "content_hash": "",
                "output_file": "response_pdf_section_1.html",
                "status": "not_translated",
                "pdf_outline_seed": True,
                "pdf_toc_section": True,
            },
        },
        "chapter_chunks": {},
    }

    mixin = object.__new__(RetranslationMixin)
    mixin.config = {"pdf_use_toc_sections": True}
    mixin.pdf_use_toc_sections_var = True
    mixin._pdf_outline_progress_plan = lambda _path: [{
        "num": 1,
        "title": "Opening",
        "level": 1,
        "start_page": 1,
        "end_page": 3,
        "section_id": section_id,
    }]

    assert mixin._seed_pdf_outline_progress_entries(
        str(pdf_path), str(output_dir), prog
    )
    assert set(prog["chapters"]) == {f"pdf:{section_id}"}
    entry = prog["chapters"][f"pdf:{section_id}"]
    assert entry["status"] == "completed"
    readable_output = "response_pdf_section_001.html"
    assert entry["output_file"] == readable_output
    assert not (output_dir / output_name).exists()
    assert (output_dir / readable_output).read_text(encoding="utf-8") == "translated"
    assert entry["pdf_section_id"] == section_id
    assert entry["pdf_progress_key"] == f"pdf:{section_id}"
    assert entry["pdf_hash_migration_pending"] is True
