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
    assert FileUtilities.create_chapter_filename(
        {**current_sections[0], "num": 1.1, "is_chunk": True},
        1.1,
    ) == "response_pdf_section_001_100.html"
    very_long_title = "A" * 1000
    assert FileUtilities.create_chapter_filename(
        {**current_sections[0], "title": very_long_title},
        1,
    ) == "response_pdf_section_001.html"
    from pdf_output_naming import safe_pdf_book_filename_stem

    bounded_book_stem = safe_pdf_book_filename_stem("\U0001F680" * 300)
    assert len(bounded_book_stem.encode("utf-16-le")) // 2 <= 180


def test_split_pdf_bookmark_keeps_distinct_chunk_qa_mapping(
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
    chapters = []
    scan_rows = []

    for actual_num, part_index, content_hash, bad_token in (
        (1.0, 1, "pdf-part-one-hash", "BADONE"),
        (1.1, 2, "pdf-part-two-hash", "BADTWO"),
    ):
        chapter = {
            "num": actual_num,
            "actual_chapter_num": actual_num,
            "title": f"Bookmark (Part {part_index}/2)",
            "body": f"source part {part_index}",
            "filename": f"pdf_section_1_{part_index - 1}.html",
            "content_hash": content_hash,
            "pdf_toc_section": True,
            "pdf_section_id": section_id,
            "pdf_section_title": "Bookmark",
            "pdf_start_page": 1,
            "pdf_end_page": 20,
            "is_chunk": True,
            "chunk_info": {
                "chunk_idx": part_index,
                "total_chunks": 2,
                "original_chapter": 1,
            },
        }
        output_name = FileUtilities.create_chapter_filename(
            chapter, actual_num
        )
        progress_key = f"pdf:{section_id}:{actual_num}"
        manager.prog["chapters"][progress_key] = {
            "actual_num": actual_num,
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
            f"<p>Good part {part_index}</p>",
            budget,
            source_text=f"source good {part_index}",
        )
        failing_html = f"<p>{bad_token} part {part_index}</p>"
        manager.record_chapter_chunk(
            content_hash,
            2,
            2,
            failing_html,
            budget,
            source_text=f"source bad {part_index}",
        )
        manager.mark_chapter_chunk_progress_status(
            content_hash, "completed"
        )
        output_path = tmp_path / output_name
        output_path.write_text(
            "\n".join((
                wrap_chunk_html(
                    content_hash,
                    1,
                    2,
                    f"<p>Good part {part_index}</p>",
                ),
                wrap_chunk_html(
                    content_hash, 2, 2, failing_html
                ),
            )),
            encoding="utf-8",
        )
        scan_rows.append({
            "filename": output_name,
            "filepath": str(output_path),
            "file_index": part_index - 1,
            "chapter_num": actual_num,
            "issues": [f"llm_token_issue: '{bad_token}'"],
            "qa_issue_previews": {},
            "duplicate_confidence": 0,
        })
        chapters.append(chapter)

    assert manager.reconcile_pdf_chapter_entries(chapters) == 0
    manager.migrate_to_content_hash(chapters)
    manager.save()

    expected_progress_keys = {
        f"pdf:{section_id}:1.0",
        f"pdf:{section_id}:1.1",
    }
    assert set(manager.prog["chapters"]) == expected_progress_keys
    assert {
        entry["content_hash"]
        for entry in manager.prog["chapters"].values()
    } == {"pdf-part-one-hash", "pdf-part-two-hash"}
    assert {
        entry["pdf_split_chunk_index"]
        for entry in manager.prog["chapters"].values()
    } == {1, 2}
    assert {
        entry["pdf_split_total_chunks"]
        for entry in manager.prog["chapters"].values()
    } == {2}
    assert {
        entry["pdf_split_parent_num"]
        for entry in manager.prog["chapters"].values()
    } == {1}
    assert set(manager.prog["chapter_chunks"]) == {
        "pdf-part-one-hash", "pdf-part-two-hash"
    }

    logs = []
    assert _attach_chunk_results_to_scan(
        str(tmp_path),
        scan_rows,
        {},
        logs.append,
        progress_path=manager.PROGRESS_FILE,
    ) == 2
    for row in scan_rows:
        assert row["chunk_results"][0]["issues"] == []
        assert row["chunk_results"][1]["issues"] == row["issues"]


def test_pdf_toc_setting_is_defaulted_exported_and_persisted():
    root = Path(__file__).resolve().parents[1]
    gui_source = (root / "src" / "translator_gui.py").read_text(encoding="utf-8")
    settings_source = (root / "src" / "other_settings.py").read_text(encoding="utf-8")

    assert "self.pdf_use_toc_sections_var = self.config.get('pdf_use_toc_sections', True)" in gui_source
    assert "'PDF_USE_TOC_SECTIONS': '1' if getattr(self, 'pdf_use_toc_sections_var', True) else '0'" in gui_source
    assert "('pdf_use_toc_sections', ['pdf_use_toc_sections_var'], True, bool)" in gui_source
    assert "Use PDF table of contents for sections" in settings_source
    assert "legacy page-by-page extraction" in settings_source


def test_progress_manager_labels_split_pdf_bookmark_parts_as_chunks(
        tmp_path):
    from Retranslation_GUI import RetranslationMixin

    section_id = "stable-section"
    chapters = {}
    for actual_num, chunk_index in ((2.0, 1), (2.1, 2), (2.2, 3)):
        progress_key = f"pdf:{section_id}:{actual_num}"
        chapters[progress_key] = {
            "actual_num": actual_num,
            "content_hash": f"hash-{chunk_index}",
            "output_file": f"response_pdf_part_{chunk_index}.html",
            "status": "in_progress",
            "model_name": "provider/model-a",
            "pdf_toc_section": True,
            "pdf_section_id": section_id,
            "pdf_progress_key": progress_key,
            "pdf_toc_title": f"Middle (Part {chunk_index}/3)",
            "pdf_start_page": 3,
            "pdf_end_page": 16,
        }

    mixin = object.__new__(RetranslationMixin)
    mixin.config = {"pdf_use_toc_sections": True}
    mixin.pdf_use_toc_sections_var = True
    mixin._is_special_file = lambda _name: False
    data = {
        "prog": {"chapters": chapters, "chapter_chunks": {}},
        "output_dir": str(tmp_path),
        "file_path": str(tmp_path / "book.pdf"),
        "show_special_files_state": False,
        "show_model_info_state": True,
    }

    mixin._rebuild_chapter_display_info(data)

    rows = data["chapter_display_info"]
    assert len(rows) == 4
    assert rows[0]["is_pdf_split_parent"] is True
    assert rows[0]["status"] == "in_progress"
    assert len(rows[0]["pdf_split_children"]) == 3
    assert [row["pdf_split_chunk_index"] for row in rows[1:]] == [1, 2, 3]
    assert {row["pdf_split_total_chunks"] for row in rows[1:]} == {3}
    displays = [
        mixin._progress_list_display_text(row, data, 20, 25)[0]
        for row in rows
    ]
    assert displays[0].startswith("Section 002 |")
    assert displays[1].startswith("   ↳ Section 002 Chunk 1/3")
    assert displays[2].startswith("   ↳ Section 002 Chunk 2/3")
    assert displays[3].startswith("   ↳ Section 002 Chunk 3/3")
    assert all("002.1" not in display and "002.2" not in display
               for display in displays)
    assert all("Pages 3-16" in display for display in displays)
    assert all("provider/model-a" in display for display in displays)

    assert mixin._expand_pdf_split_parent_rows([rows[0], rows[2]]) == rows[1:]
    assert mixin._pdf_compiled_section_ordinal(data["prog"], rows[2]) == 2


def test_pdf_split_parent_aggregates_child_status_and_model():
    from Retranslation_GUI import RetranslationMixin

    rows = []
    for actual_num, chunk_index, status, model in (
        (4.0, 1, "completed", "provider/model-a"),
        (4.1, 2, "pending", "provider/model-b"),
    ):
        progress_key = f"pdf:section-four:{actual_num}"
        entry = {
            "actual_num": actual_num,
            "output_file": f"part-{chunk_index}.html",
            "status": status,
            "model_name": model,
            "pdf_toc_section": True,
            "pdf_section_id": "section-four",
            "pdf_progress_key": progress_key,
            "pdf_split_chunk_index": chunk_index,
            "pdf_split_total_chunks": 2,
            "pdf_split_parent_num": 4,
        }
        rows.append({
            "key": progress_key,
            "progress_key": progress_key,
            "num": actual_num,
            "info": entry,
            "output_file": entry["output_file"],
            "status": status,
        })

    mixin = object.__new__(RetranslationMixin)
    mixin._annotate_pdf_split_chunk_display_info(rows)

    assert len(rows) == 3
    assert rows[0]["is_pdf_split_parent"] is True
    assert rows[0]["status"] == "pending"
    assert rows[0]["info"]["model_name"] == "(multiple models)"
    assert [row["info"]["model_name"] for row in rows[1:]] == [
        "provider/model-a",
        "provider/model-b",
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


def test_old_pdf_split_cache_rehydrates_stable_bookmark_identity(
        tmp_path, monkeypatch):
    from txt_processor import TextFileProcessor

    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"pdf")
    output_dir = tmp_path / "out"
    processor = TextFileProcessor(str(pdf_path), str(output_dir))
    processor.pdf_render_mode = "legacy_layout"
    monkeypatch.setattr(
        processor,
        "_load_split_cache",
        lambda *_args: [{
            "num": 1,
            "title": "Opening",
            "filename": "pdf_section_1.html",
            "body": "<p>opening</p>",
            "content_hash": "cached-hash",
            "is_chunk": False,
            "pdf_toc_section": True,
        }],
    )
    section_id = "stable-bookmark-id"
    chapters = processor._process_chapters_for_splitting([{
        "num": 1,
        "title": "Opening",
        "content": "<p>opening</p>",
        "is_html": True,
        "pdf_toc_section": True,
        "pdf_section_id": section_id,
        "pdf_section_title": "Opening",
        "pdf_start_page": 1,
        "pdf_end_page": 3,
    }])

    assert chapters[0]["pdf_section_id"] == section_id
    assert chapters[0]["pdf_section_title"] == "Opening"
