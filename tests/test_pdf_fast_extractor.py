import json
from pathlib import Path

import pytest

from pdf_extractor import build_pdf_toc_section_plan, group_pdf_pages_by_toc
from pdf_fast_extractor import extract_pdf_fast
from _pdf_extraction_worker import run_pdf_extraction
from TransateKRtoEN import FileUtilities, ProgressManager


def _make_image_pdf(path: Path, page_count=3):
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 24, 24), False)
    pixmap.clear_with(0x4477AA)
    image_bytes = pixmap.tobytes("png")
    shared_xref = 0
    for page_index in range(page_count):
        page = document.new_page(width=320, height=480)
        page.insert_text((36, 48), f"Chapter {page_index + 1}", fontsize=16)
        page.insert_text(
            (36, 82),
            f"This is source text from page {page_index + 1}.",
            fontsize=11,
        )
        if shared_xref:
            page.insert_image(
                fitz.Rect(36, 110, 108, 182),
                xref=shared_xref,
            )
        else:
            shared_xref = page.insert_image(
                fitz.Rect(36, 110, 108, 182),
                stream=image_bytes,
            )
    document.set_toc([[1, "Chapter 1", 1], [1, f"Chapter {page_count}", page_count]])
    document.save(path)
    document.close()


def _manifest(output_dir: Path, mode="fast_semantic"):
    return json.loads(
        (output_dir / ".pdf_extraction_cache" / f"manifest_{mode}.json").read_text(
            encoding="utf-8"
        )
    )


def test_fast_semantic_deduplicates_repeated_images(tmp_path, monkeypatch):
    pdf_path = tmp_path / "book.pdf"
    output_dir = tmp_path / "output"
    _make_image_pdf(pdf_path)
    monkeypatch.setenv("EXTRACTION_WORKERS", "1")

    pages, images_by_page = extract_pdf_fast(
        str(pdf_path),
        str(output_dir),
        mode="fast_semantic",
        extract_images=True,
        page_by_page=True,
    )

    assert [page_number for page_number, _html in pages] == [1, 2, 3]
    assert all("This is source text" in page_html for _, page_html in pages)
    assert all("data:image" not in page_html for _, page_html in pages)
    assert sorted(images_by_page) == [0, 1, 2]
    filenames = {
        image["filename"]
        for page_images in images_by_page.values()
        for image in page_images
    }
    assert len(filenames) == 1
    assert len(list((output_dir / "images").glob("pdfimg_*"))) == 1


def test_fast_layout_externalizes_images_without_base64(tmp_path, monkeypatch):
    pdf_path = tmp_path / "layout.pdf"
    output_dir = tmp_path / "output"
    _make_image_pdf(pdf_path, page_count=1)
    monkeypatch.setenv("EXTRACTION_WORKERS", "1")

    pages, images_by_page = extract_pdf_fast(
        str(pdf_path),
        str(output_dir),
        mode="fast_layout",
        extract_images=True,
        page_by_page=True,
    )

    assert len(pages) == 1
    assert "pdf-fast-layout-page" in pages[0][1]
    assert "pdf-fast-layout-image" in pages[0][1]
    assert "data:image" not in pages[0][1]
    assert images_by_page[0][0]["filename"].startswith("pdfimg_")


def test_outline_only_update_reuses_cached_pages_and_regroups(tmp_path, monkeypatch):
    fitz = pytest.importorskip("fitz")
    pdf_path = tmp_path / "updated-outline.pdf"
    output_dir = tmp_path / "output"
    _make_image_pdf(pdf_path)
    monkeypatch.setenv("EXTRACTION_WORKERS", "1")

    first_pages, _ = extract_pdf_fast(
        str(pdf_path),
        str(output_dir),
        mode="fast_semantic",
        page_by_page=True,
    )
    first_manifest = _manifest(output_dir)
    cache_mtimes = {
        page: (output_dir / ".pdf_extraction_cache" / entry["cache_file"]).stat().st_mtime_ns
        for page, entry in first_manifest["pages"].items()
    }

    with fitz.open(pdf_path) as document:
        document.set_toc(
            [[1, "Chapter 1", 1], [1, "Chapter 2", 2], [1, "Chapter 3", 3]]
        )
        document.saveIncr()

    second_pages, _ = extract_pdf_fast(
        str(pdf_path),
        str(output_dir),
        mode="fast_semantic",
        page_by_page=True,
    )
    second_manifest = _manifest(output_dir)

    assert second_manifest["outline_digest"] != first_manifest["outline_digest"]
    assert second_manifest["stats"]["reused_pages"] == 3
    assert second_manifest["stats"]["extracted_pages"] == 0
    assert cache_mtimes == {
        page: (output_dir / ".pdf_extraction_cache" / entry["cache_file"]).stat().st_mtime_ns
        for page, entry in second_manifest["pages"].items()
    }
    grouped = group_pdf_pages_by_toc(str(pdf_path), second_pages)
    assert [section["title"] for section in grouped] == [
        "Chapter 1",
        "Chapter 2",
        "Chapter 3",
    ]
    assert first_pages == second_pages


def test_page_content_update_only_reextracts_changed_page(tmp_path, monkeypatch):
    fitz = pytest.importorskip("fitz")
    pdf_path = tmp_path / "updated-page.pdf"
    output_dir = tmp_path / "output"
    _make_image_pdf(pdf_path)
    monkeypatch.setenv("EXTRACTION_WORKERS", "1")

    extract_pdf_fast(
        str(pdf_path),
        str(output_dir),
        mode="fast_semantic",
        page_by_page=True,
    )
    with fitz.open(pdf_path) as document:
        document[1].insert_text((36, 220), "Newly added source text.", fontsize=11)
        document.saveIncr()

    pages, _ = extract_pdf_fast(
        str(pdf_path),
        str(output_dir),
        mode="fast_semantic",
        page_by_page=True,
    )
    manifest = _manifest(output_dir)

    assert manifest["stats"]["reused_pages"] == 2
    assert manifest["stats"]["extracted_pages"] == 1
    assert "Newly added source text" in pages[1][1]


def test_exact_source_cache_hit_skips_all_page_extraction(tmp_path, monkeypatch):
    pdf_path = tmp_path / "cached.pdf"
    output_dir = tmp_path / "output"
    _make_image_pdf(pdf_path, page_count=2)
    monkeypatch.setenv("EXTRACTION_WORKERS", "1")

    extract_pdf_fast(
        str(pdf_path),
        str(output_dir),
        mode="fast_semantic",
        page_by_page=True,
    )
    pages, _ = extract_pdf_fast(
        str(pdf_path),
        str(output_dir),
        mode="fast_semantic",
        page_by_page=True,
    )

    manifest = _manifest(output_dir)
    assert len(pages) == 2
    assert manifest["stats"]["reused_pages"] == 2
    assert manifest["stats"]["extracted_pages"] == 0


def test_parallel_page_range_pool_returns_pages_in_source_order(tmp_path, monkeypatch):
    pdf_path = tmp_path / "parallel.pdf"
    output_dir = tmp_path / "output"
    _make_image_pdf(pdf_path, page_count=9)
    monkeypatch.setenv("EXTRACTION_WORKERS", "2")
    monkeypatch.setenv("PDF_FAST_CHUNK_PAGES", "2")

    pages, images_by_page = extract_pdf_fast(
        str(pdf_path),
        str(output_dir),
        mode="fast_semantic",
        page_by_page=True,
    )

    assert [page_number for page_number, _ in pages] == list(range(1, 10))
    assert len(images_by_page) == 9
    assert len(list((output_dir / "images").glob("pdfimg_*"))) == 1


def test_pdf_worker_groups_fast_pages_into_bookmark_entries(tmp_path, monkeypatch):
    pdf_path = tmp_path / "worker.pdf"
    output_dir = tmp_path / "output"
    result_path = tmp_path / "result.json"
    config_path = tmp_path / "config.json"
    _make_image_pdf(pdf_path, page_count=3)
    monkeypatch.setenv("EXTRACTION_WORKERS", "1")
    config_path.write_text(
        json.dumps(
            {
                "pdf_path": str(pdf_path),
                "output_dir": str(output_dir),
                "render_mode": "fast_semantic",
                "use_toc_sections": True,
                "extract_images": True,
                "generate_css": False,
                "html2text": False,
                "result_path": str(result_path),
            }
        ),
        encoding="utf-8",
    )

    result = run_pdf_extraction(str(config_path))

    assert result["success"] is True
    assert result["page_count"] == 3
    assert result["entry_count"] == 2
    assert result["separation_mode"] == "toc"
    assert [section["title"] for section in result["section_info"]] == [
        "Chapter 1",
        "Chapter 3",
    ]


def test_other_settings_exposes_new_modes_and_legacy_fallback():
    root = Path(__file__).resolve().parents[1]
    settings_source = (root / "src" / "other_settings.py").read_text(encoding="utf-8")
    gui_source = (root / "src" / "translator_gui.py").read_text(encoding="utf-8")

    assert '("Fast Semantic", "fast_semantic")' in settings_source
    assert '("Fast Layout", "fast_layout")' in settings_source
    assert '("Legacy Layout", "legacy_layout")' in settings_source
    assert "self.config.get('pdf_render_mode', 'fast_semantic')" in gui_source
    assert "self.config['pdf_fast_engine_migrated'] = True" in gui_source


def test_unchanged_bookmark_sections_keep_stable_identity_when_one_is_added():
    original = build_pdf_toc_section_plan(
        [[1, "Opening", 1], [1, "Middle", 3], [1, "Ending", 5]],
        6,
    )
    updated = build_pdf_toc_section_plan(
        [
            [1, "Opening", 1],
            [1, "Inserted", 2],
            [1, "Middle", 3],
            [1, "Ending", 5],
        ],
        6,
    )
    original_ids = {section["title"]: section["section_id"] for section in original}
    updated_ids = {section["title"]: section["section_id"] for section in updated}

    assert original_ids["Middle"] == updated_ids["Middle"]
    assert original_ids["Ending"] == updated_ids["Ending"]
    assert original_ids["Opening"] != updated_ids["Opening"]


def test_pdf_progress_uses_stable_section_id_across_display_number_changes(tmp_path):
    payloads = tmp_path / "Payloads"
    payloads.mkdir()
    manager = ProgressManager(str(payloads))
    section_id = "abc123stable"
    output_file = f"response_pdf_section_{section_id}.html"
    (tmp_path / output_file).write_text("translated", encoding="utf-8")
    manager.prog["chapters"] = {
        f"pdf:{section_id}": {
            "actual_num": 2,
            "chapter_num": 2,
            "content_hash": "same-source-hash",
            "output_file": output_file,
            "status": "completed",
            "pdf_toc_section": True,
            "pdf_section_id": section_id,
            "pdf_progress_key": f"pdf:{section_id}",
        }
    }
    current_chapter = {
        "num": 3,
        "title": "Middle",
        "filename": "pdf_section_3.html",
        "content_hash": "same-source-hash",
        "pdf_toc_section": True,
        "pdf_section_id": section_id,
        "pdf_section_title": "Middle",
        "pdf_start_page": 3,
        "pdf_end_page": 4,
    }

    assert FileUtilities.create_chapter_filename(current_chapter, 3) == output_file
    assert manager.reconcile_pdf_chapter_entries([current_chapter]) == 0
    assert manager.prog["chapters"][f"pdf:{section_id}"]["actual_num"] == 3
    manager.migrate_to_content_hash([current_chapter])
    assert f"pdf:{section_id}" in manager.prog["chapters"]
