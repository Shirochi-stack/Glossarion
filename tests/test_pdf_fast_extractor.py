import hashlib
import json
import time
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from pdf_extractor import build_pdf_toc_section_plan, group_pdf_pages_by_toc
from pdf_fast_extractor import (
    PDFExtractionCancelled,
    _fast_pdf_worker_count,
    _layout_paragraph_alignment,
    _semantic_page_html,
    _text_alignment,
    apply_pdf_image_rename_logic,
    extract_pdf_fast,
    extract_pdf_page_range_for_reader,
    normalize_pdf_paragraph_alignment,
    normalize_pdf_paragraph_justification,
    pdf_rtl_paragraph_layout_enabled,
    resolve_pdf_paragraph_alignment,
    resolve_pdf_extraction_workers,
)
from workspace_reader import (
    build_workspace_reader_manifest,
    ensure_pdf_raw_section,
)
from _pdf_extraction_worker import run_pdf_extraction
from TransateKRtoEN import (
    FileUtilities,
    ProgressManager,
    retroactive_update_image_references,
)
from txt_processor import TextFileProcessor


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


def test_fast_semantic_preserves_links_tables_and_vector_graphics(
    tmp_path,
    monkeypatch,
):
    fitz = pytest.importorskip("fitz")
    pdf_path = tmp_path / "structured.pdf"
    output_dir = tmp_path / "output"
    document = fitz.open()
    page = document.new_page(width=420, height=520)
    page.insert_text((36, 42), "Reference: finish the attack for details.", fontsize=11)
    page.insert_text((36, 68), "Plain URL: https://example.org/guide", fontsize=11)
    link_rect = page.search_for("finish the attack")[0]
    page.insert_link(
        {
            "kind": fitz.LINK_URI,
            "from": link_rect,
            "uri": "https://academy.example/program",
        }
    )

    for x in (36, 156, 276):
        page.draw_line((x, 100), (x, 180), color=(0, 0, 0))
    for y in (100, 140, 180):
        page.draw_line((36, y), (276, y), color=(0, 0, 0))
    page.insert_text((46, 126), "Name", fontsize=10)
    page.insert_text((166, 126), "URL", fontsize=10)
    page.insert_text((46, 166), "Example", fontsize=10)
    page.insert_text((166, 166), "https://table.example", fontsize=10)

    page.draw_circle(
        (340, 130),
        32,
        color=(0.8, 0.1, 0.1),
        fill=(1.0, 0.8, 0.2),
        width=3,
    )
    document.set_toc([[1, "Structured Page", 1]])
    document.save(pdf_path)
    document.close()
    monkeypatch.setenv("EXTRACTION_WORKERS", "1")

    pages, images_by_page = extract_pdf_fast(
        str(pdf_path),
        str(output_dir),
        mode="fast_semantic",
        extract_images=True,
        page_by_page=True,
    )

    assert len(pages) == 1
    soup = BeautifulSoup(pages[0][1], "html.parser")
    annotation = soup.find("a", href="https://academy.example/program")
    assert annotation is not None
    assert annotation.get_text(" ", strip=True) == "finish the attack"
    assert soup.find("a", href="https://example.org/guide") is not None
    table = soup.find("table", class_="pdf-table")
    assert table is not None
    assert [cell.get_text(" ", strip=True) for cell in table.find_all(["th", "td"])] == [
        "Name",
        "URL",
        "Example",
        "https://table.example",
    ]
    assert table.find("a", href="https://table.example") is not None
    assert not any("Example" in paragraph.get_text() for paragraph in soup.find_all("p"))

    vector = soup.find("figure", class_="pdf-vector-graphic")
    assert vector is not None
    vector_src = vector.img["src"]
    assert vector_src.lower().endswith(".svg")
    vector_path = output_dir / vector_src
    assert vector_path.is_file()
    assert "<path" in vector_path.read_text(encoding="utf-8")
    assert images_by_page[0][0]["kind"] == "vector"

    chapter = {
        "num": 1,
        "filename": "pdf_section_structured.html",
        "body": pages[0][1],
    }
    apply_pdf_image_rename_logic([chapter], str(output_dir))
    renamed = BeautifulSoup(chapter["body"], "html.parser").find(
        "figure", class_="pdf-vector-graphic"
    )
    assert renamed.img["src"] == "images/pdf_section_structured_img_1.svg"
    assert (output_dir / renamed.img["src"]).is_file()


def test_fast_pdf_images_receive_chapter_names_without_breaking_cache(
        tmp_path, monkeypatch):
    pdf_path = tmp_path / "book.pdf"
    output_dir = tmp_path / "output"
    _make_image_pdf(pdf_path)
    monkeypatch.setenv("EXTRACTION_WORKERS", "1")

    pages, _images_by_page = extract_pdf_fast(
        str(pdf_path),
        str(output_dir),
        mode="fast_semantic",
        extract_images=True,
        page_by_page=True,
    )
    chapters = [
        {
            "num": page_number,
            "filename": f"pdf_section_{page_number}.html",
            "body": page_html,
        }
        for page_number, page_html in pages
    ]

    apply_pdf_image_rename_logic(chapters, str(output_dir))

    image_names = sorted(path.name for path in (output_dir / "images").iterdir())
    assert image_names == ["pdf_section_1_img_1.png"]
    assert all("pdfimg_" not in chapter["body"] for chapter in chapters)
    assert all("pdf_section_1_img_1.png" in chapter["body"] for chapter in chapters)
    assert all(
        chapter["content_hash"]
        == hashlib.sha256(chapter["body"].encode("utf-8")).hexdigest()
        for chapter in chapters
    )
    rename_map = json.loads(
        (output_dir / "image_rename_map.json").read_text(encoding="utf-8")
    )
    assert set(rename_map.values()) == {"pdf_section_1_img_1.png"}

    cached_pages, cached_images = extract_pdf_fast(
        str(pdf_path),
        str(output_dir),
        mode="fast_semantic",
        extract_images=True,
        page_by_page=True,
    )
    assert _manifest(output_dir)["stats"]["reused_pages"] == 3
    assert all("pdf_section_1_img_1.png" in html for _page, html in cached_pages)
    assert {
        image["filename"]
        for page_images in cached_images.values()
        for image in page_images
    } == {"pdf_section_1_img_1.png"}

    legacy_chapter = {
        "num": 1,
        "filename": "pdf_section_1.html",
        "body": cached_pages[0][1].replace(
            "pdf_section_1_img_1.png", "page_1_img_1.png"
        ),
    }
    apply_pdf_image_rename_logic([legacy_chapter], str(output_dir))
    assert "page_1_img_1.png" not in legacy_chapter["body"]
    assert "pdf_section_1_img_1.png" in legacy_chapter["body"]


def test_fast_semantic_alignment_does_not_center_full_width_paragraphs():
    assert _text_alignment([81.0, 70.0, 510.0, 101.0], 595.0) == "left"
    assert _text_alignment([214.0, 112.0, 382.0, 145.0], 595.0) == "center"

    class _Rect:
        width = 595.0

    class _Page:
        rect = _Rect()

        @staticmethod
        def get_text(_kind, **_kwargs):
            return [
                (214.0, 112.0, 382.0, 145.0, "Chapter", 0, 0),
                (81.0, 186.0, 510.0, 218.0, "Body paragraph", 1, 0),
                (214.0, 230.0, 382.0, 262.0, "Short body sentence", 2, 0),
            ]

    rendered = _semantic_page_html(_Page(), 1, [], "Chapter")
    assert '<h1 style="text-align:center">Chapter</h1>' in rendered
    assert (
        '<p class="pdf-align-left" data-pdf-source-alignment="left" '
        'style="text-align:left">'
        'Body paragraph</p>'
    ) in rendered
    assert (
        '<p class="pdf-align-left" data-pdf-source-alignment="left" '
        'style="text-align:left">'
        'Short body sentence</p>'
    ) in rendered
    assert '<p class="pdf-align-center"' not in rendered


def test_multiline_alignment_requires_both_edges_to_move_for_centering():
    left_aligned = {
        "bbox": [80.0, 100.0, 500.0, 140.0],
        "lines": [
            {
                "bbox": [80.0, 100.0, 500.0, 118.0],
                "spans": [{"text": "First wrapped line"}],
            },
            {
                "bbox": [80.0, 120.0, 470.0, 138.0],
                "spans": [{"text": "Second wrapped line"}],
            },
        ],
    }
    centered = {
        "bbox": [140.0, 100.0, 460.0, 140.0],
        "lines": [
            {
                "bbox": [140.0, 100.0, 460.0, 118.0],
                "spans": [{"text": "Long centered line"}],
            },
            {
                "bbox": [190.0, 120.0, 410.0, 138.0],
                "spans": [{"text": "Short centered line"}],
            },
        ],
    }

    assert _layout_paragraph_alignment(
        left_aligned,
        600.0,
        (80.0, 500.0),
        "First wrapped line Second wrapped line",
    ) == "left"
    assert _layout_paragraph_alignment(
        centered,
        600.0,
        (80.0, 500.0),
        "Long centered line Short centered line",
    ) == "center"


def test_fast_semantic_detects_source_justification_and_honors_overrides(monkeypatch):
    class _Rect:
        width = 595.0

    class _Page:
        rect = _Rect()

        @staticmethod
        def get_text(kind, **_kwargs):
            if kind == "dict":
                return {
                    "blocks": [
                        {
                            "number": 0,
                            "type": 0,
                            "bbox": [70.8, 90.0, 250.0, 110.0],
                            "lines": [{
                                "bbox": [70.8, 90.0, 250.0, 110.0],
                                "spans": [{"text": "What is rest defence?"}],
                            }],
                        },
                        {
                            "number": 1,
                            "type": 0,
                            "bbox": [70.8, 130.0, 528.2, 220.0],
                            "lines": [
                                {"bbox": [70.8, 130.0, 528.2, 145.0], "spans": [{"text": "First full line"}]},
                                {"bbox": [70.8, 150.0, 528.1, 165.0], "spans": [{"text": "Second full line"}]},
                                {"bbox": [70.8, 170.0, 528.2, 185.0], "spans": [{"text": "Third full line"}]},
                                {"bbox": [70.8, 190.0, 260.0, 205.0], "spans": [{"text": "Short last line"}]},
                            ],
                        },
                    ]
                }
            return [
                (70.8, 90.0, 250.0, 110.0, "What is rest defence?", 0, 0),
                (
                    70.8,
                    130.0,
                    528.2,
                    220.0,
                    "First full line Second full line Third full line Short last line",
                    1,
                    0,
                ),
            ]

    monkeypatch.setenv("PDF_PARAGRAPH_ALIGNMENT", "source")
    monkeypatch.setenv("PDF_PARAGRAPH_JUSTIFICATION", "source")
    rendered = _semantic_page_html(
        _Page(), 1, [], "What is rest defence?", tables=[], links=[]
    )
    paragraph = BeautifulSoup(rendered, "html.parser").find("p")
    assert paragraph["class"] == ["pdf-align-justify"]
    assert paragraph["data-pdf-source-alignment"] == "justify"
    assert paragraph["style"] == "text-align:justify"

    monkeypatch.setenv("PDF_PARAGRAPH_JUSTIFICATION", "none")
    rendered = _semantic_page_html(
        _Page(), 1, [], "What is rest defence?", tables=[], links=[]
    )
    paragraph = BeautifulSoup(rendered, "html.parser").find("p")
    assert paragraph["class"] == ["pdf-align-left"]
    assert paragraph["data-pdf-source-alignment"] == "justify"
    assert paragraph["style"] == "text-align:left"

    monkeypatch.setenv("PDF_PARAGRAPH_ALIGNMENT", "right")
    monkeypatch.setenv("PDF_PARAGRAPH_JUSTIFICATION", "source")
    rendered = _semantic_page_html(
        _Page(), 1, [], "What is rest defence?", tables=[], links=[]
    )
    paragraph = BeautifulSoup(rendered, "html.parser").find("p")
    assert paragraph["class"] == ["pdf-align-right"]
    assert paragraph["data-pdf-source-alignment"] == "justify"
    assert paragraph["style"] == "text-align:right"


def test_pdf_paragraph_override_normalization_and_precedence():
    assert normalize_pdf_paragraph_alignment("centre") == "center"
    assert normalize_pdf_paragraph_alignment("invalid") == "source"
    assert normalize_pdf_paragraph_justification("justified") == "justify"
    assert normalize_pdf_paragraph_justification("off") == "none"
    assert resolve_pdf_paragraph_alignment(
        "left",
        "Body",
        alignment_override="right",
        justification_override="justify",
    ) == "justify"
    assert resolve_pdf_paragraph_alignment(
        "left",
        "Arabic body",
        alignment_override="source",
        justification_override="source",
        rtl_layout=True,
    ) == "right"
    assert resolve_pdf_paragraph_alignment(
        "left",
        "Arabic body",
        alignment_override="left",
        justification_override="source",
        rtl_layout=True,
    ) == "left"
    assert resolve_pdf_paragraph_alignment(
        "justify",
        "Arabic body",
        alignment_override="source",
        justification_override="source",
        rtl_layout=True,
    ) == "justify"


def test_pdf_rtl_paragraph_layout_marks_semantic_document(monkeypatch):
    class _Rect:
        width = 595.0

    class _Page:
        rect = _Rect()

        def get_text(self, kind, **_kwargs):
            if kind == "dict":
                return {
                    "blocks": [{
                        "type": 0,
                        "lines": [{
                            "bbox": [70.0, 100.0, 520.0, 130.0],
                            "spans": [{"text": "Ù†Øµ Ø¹Ø±Ø¨ÙŠ"}],
                        }],
                    }]
                }
            return [(70.0, 100.0, 520.0, 130.0, "Ù†Øµ Ø¹Ø±Ø¨ÙŠ", 0, 0)]

    monkeypatch.setenv("PDF_RTL_PARAGRAPH_LAYOUT", "1")
    assert pdf_rtl_paragraph_layout_enabled() is True
    rendered = _semantic_page_html(_Page(), 1, [], "", tables=[], links=[])
    article = BeautifulSoup(rendered, "html.parser").article
    assert article["dir"] == "rtl"
    assert article["data-pdf-rtl-layout"] == "true"
    assert "pdf-rtl-layout" in article["class"]
    assert "unicode-bidi:plaintext" in rendered
    assert "text-align-last:right" in rendered
    paragraph = article.find("p")
    assert paragraph["class"] == ["pdf-align-right"]
    assert paragraph["style"] == "text-align:right"

    monkeypatch.setenv("PDF_RTL_PARAGRAPH_LAYOUT", "0")
    assert pdf_rtl_paragraph_layout_enabled() is False
    rendered = _semantic_page_html(_Page(), 1, [], "", tables=[], links=[])
    assert BeautifulSoup(rendered, "html.parser").article.get("dir") is None


def test_fast_semantic_matches_bookmark_title_across_pdf_line_wrap_spaces():
    class _Rect:
        width = 595.0

    class _Page:
        rect = _Rect()

        @staticmethod
        def get_text(_kind, **_kwargs):
            return [
                (
                    150.0,
                    112.0,
                    445.0,
                    145.0,
                    "결혼 전에 관계를 가지는 건 옳지 않으 니까.",
                    0,
                    0,
                ),
                (81.0, 186.0, 510.0, 218.0, "본문 문장입니다.", 1, 0),
            ]

    rendered = _semantic_page_html(
        _Page(),
        310,
        [],
        "결혼 전에 관계를 가지는 건 옳지 않으니까.",
    )

    assert (
        '<h1 style="text-align:center">'
        '결혼 전에 관계를 가지는 건 옳지 않으 니까.</h1>'
    ) in rendered
    assert '<p class="pdf-align-left"' in rendered


def test_text_processor_automatically_applies_pdf_image_rename_logic(
        tmp_path, monkeypatch):
    pdf_path = tmp_path / "book.pdf"
    output_dir = tmp_path / "output"
    _make_image_pdf(pdf_path)
    monkeypatch.setenv("EXTRACTION_WORKERS", "1")
    monkeypatch.setenv("PDF_RENDER_MODE", "fast_semantic")
    monkeypatch.setenv("PDF_EXTRACT_IMAGES", "1")
    monkeypatch.setenv("PDF_GENERATE_CSS", "0")
    monkeypatch.setenv("PDF_USE_TOC_SECTIONS", "1")

    chapters = TextFileProcessor(str(pdf_path), str(output_dir)).extract_chapters()

    assert [chapter["filename"] for chapter in chapters] == [
        "pdf_section_1.html",
        "pdf_section_2.html",
    ]
    assert sorted(path.name for path in (output_dir / "images").iterdir()) == [
        "pdf_section_1_img_1.png"
    ]
    assert all("pdfimg_" not in chapter["body"] for chapter in chapters)
    assert sorted(
        path.name for path in (output_dir / "word_count" / "images").iterdir()
    ) == ["pdf_section_1_img_1.png"]

    canonical = output_dir / "images" / "pdf_section_1_img_1.png"
    response_named = output_dir / "images" / "pdf_section_stableid_img_1.png"
    canonical.rename(response_named)
    response_html = output_dir / "response_pdf_section_stableid.html"
    response_html.write_text(
        '<img src="images/pdf_section_stableid_img_1.png">',
        encoding="utf-8",
    )
    rename_map_path = output_dir / "image_rename_map.json"
    rename_map = json.loads(rename_map_path.read_text(encoding="utf-8"))
    rename_map["pdf_section_1_img_1.png"] = "pdf_section_stableid_img_1.png"
    rename_map_path.write_text(json.dumps(rename_map), encoding="utf-8")

    apply_pdf_image_rename_logic(
        chapters,
        str(output_dir),
        word_count_dir=str(output_dir / "word_count"),
    )

    assert canonical.is_file()
    assert not response_named.exists()
    assert "pdf_section_1_img_1.png" in response_html.read_text(encoding="utf-8")


def test_epub_image_repair_does_not_rename_pdf_section_images(tmp_path):
    workspace = tmp_path / "Book"
    images = workspace / "images"
    images.mkdir(parents=True)
    source_pdf = tmp_path / "Book.pdf"
    source_pdf.write_bytes(b"pdf")
    (workspace / "source_epub.txt").write_text(
        str(source_pdf), encoding="utf-8"
    )
    canonical_name = "pdf_section_1_img_1.png"
    (images / canonical_name).write_bytes(b"image")
    response = workspace / "response_pdf_section_stableid.html"
    response.write_text(
        f'<html><body><img src="images/{canonical_name}"></body></html>',
        encoding="utf-8",
    )

    retroactive_update_image_references(str(workspace))

    assert (images / canonical_name).is_file()
    assert not (images / "pdf_section_stableid_img_1.png").exists()
    assert canonical_name in response.read_text(encoding="utf-8")


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


def test_pdf_paragraph_override_change_invalidates_page_cache(tmp_path, monkeypatch):
    pdf_path = tmp_path / "formatting.pdf"
    output_dir = tmp_path / "output"
    _make_image_pdf(pdf_path, page_count=2)
    monkeypatch.setenv("EXTRACTION_WORKERS", "1")
    monkeypatch.setenv("PDF_PARAGRAPH_ALIGNMENT", "source")
    monkeypatch.setenv("PDF_PARAGRAPH_JUSTIFICATION", "source")

    extract_pdf_fast(
        str(pdf_path),
        str(output_dir),
        mode="fast_semantic",
        page_by_page=True,
    )
    monkeypatch.setenv("PDF_PARAGRAPH_JUSTIFICATION", "justify")
    pages, _ = extract_pdf_fast(
        str(pdf_path),
        str(output_dir),
        mode="fast_semantic",
        page_by_page=True,
    )

    manifest = _manifest(output_dir)
    assert manifest["settings"]["paragraph_justification"] == "justify"
    assert manifest["stats"]["reused_pages"] == 0
    assert manifest["stats"]["extracted_pages"] == 2
    assert all('class="pdf-align-justify"' in html for _page, html in pages)


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


def test_fast_pdf_worker_count_uses_configured_parallel_capacity(monkeypatch):
    import pdf_fast_extractor as fast_extractor

    monkeypatch.setattr(fast_extractor.os, "cpu_count", lambda: 16)
    monkeypatch.delenv("PDF_EXTRACTION_WORKERS", raising=False)
    monkeypatch.delenv("EXTRACTION_WORKERS", raising=False)
    monkeypatch.delenv("PDF_FAST_MAX_WORKERS", raising=False)
    assert _fast_pdf_worker_count(816, 68) == 8

    monkeypatch.setenv("EXTRACTION_WORKERS", "6")
    assert _fast_pdf_worker_count(816, 68) == 6

    monkeypatch.setenv("PDF_EXTRACTION_WORKERS", "auto")
    assert _fast_pdf_worker_count(816, 68) == 8

    monkeypatch.setenv("PDF_EXTRACTION_WORKERS", "12")
    monkeypatch.setenv("PDF_FAST_MAX_WORKERS", "10")
    assert _fast_pdf_worker_count(816, 68) == 10

    assert resolve_pdf_extraction_workers("auto", cpu_count=28) == 14
    assert resolve_pdf_extraction_workers("40", cpu_count=28) == 28
    assert _fast_pdf_worker_count(7, 7) == 1


def test_fast_extractor_reports_job_progress(tmp_path, monkeypatch, capsys):
    pdf_path = tmp_path / "progress.pdf"
    output_dir = tmp_path / "output"
    _make_image_pdf(pdf_path, page_count=4)
    monkeypatch.setenv("EXTRACTION_WORKERS", "1")
    monkeypatch.setenv("PDF_FAST_CHUNK_PAGES", "2")
    monkeypatch.setenv("PDF_PROGRESS_HEARTBEAT_SECONDS", "0.05")
    import pdf_fast_extractor as fast_extractor

    real_extract_range = fast_extractor._extract_page_range

    def slow_extract_range(args):
        time.sleep(0.08)
        return real_extract_range(args)

    monkeypatch.setattr(fast_extractor, "_extract_page_range", slow_extract_range)

    extract_pdf_fast(
        str(pdf_path),
        str(output_dir),
        mode="fast_semantic",
        page_by_page=True,
    )

    output = capsys.readouterr().out
    assert "Fast PDF phase: fingerprinting source and reading bookmarks" in output
    assert "Fast PDF extraction heartbeat:" in output
    assert "Fast PDF progress: 2/4 pages (50%)" in output
    assert "Fast PDF progress: 4/4 pages (100%)" in output


def test_fast_extractor_stop_callback_cancels_and_cleans_stop_file(
    tmp_path,
    monkeypatch,
):
    pdf_path = tmp_path / "cancel.pdf"
    output_dir = tmp_path / "output"
    _make_image_pdf(pdf_path, page_count=4)
    monkeypatch.setenv("EXTRACTION_WORKERS", "1")
    monkeypatch.delenv("PDF_EXTRACTION_STOP_FILE", raising=False)

    with pytest.raises(PDFExtractionCancelled):
        extract_pdf_fast(
            str(pdf_path),
            str(output_dir),
            mode="fast_semantic",
            page_by_page=True,
            stop_callback=lambda: True,
        )

    assert "PDF_EXTRACTION_STOP_FILE" not in __import__("os").environ
    assert not (
        output_dir / ".pdf_extraction_cache" / "active_extraction.stop"
    ).exists()


def test_text_processor_does_not_swallow_pdf_cancellation(tmp_path, monkeypatch):
    pdf_path = tmp_path / "cancel.pdf"
    pdf_path.write_bytes(b"placeholder")

    def cancelled_extraction(*_args, **_kwargs):
        raise PDFExtractionCancelled("stopped")

    monkeypatch.setattr(
        "txt_processor.extract_pdf_with_formatting",
        cancelled_extraction,
    )
    processor = TextFileProcessor(
        str(pdf_path),
        str(tmp_path / "output"),
        stop_callback=lambda: True,
    )

    with pytest.raises(PDFExtractionCancelled):
        processor.extract_chapters()


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
    translation_source = (root / "src" / "TransateKRtoEN.py").read_text(
        encoding="utf-8"
    )

    assert '("Fast Semantic", "fast_semantic")' in settings_source
    assert '("Fast Layout", "fast_layout")' in settings_source
    assert '("Legacy Layout", "legacy_layout")' in settings_source
    assert "self.config.get('pdf_render_mode', 'fast_semantic')" in gui_source
    assert "self.config['pdf_fast_engine_migrated'] = True" in gui_source
    assert "preserve_fast_pdf_images" in translation_source
    assert "or preserve_fast_pdf_images" in translation_source
    assert "PDF_EXTRACTION_STOP_FILE" in gui_source
    assert 'QLabel("PDF Input Settings")' in settings_source
    assert "pdf_extraction_workers_var" in settings_source
    assert 'QPushButton("Auto")' in settings_source
    assert "CPU cores:" in settings_source
    assert "PDF_EXTRACTION_WORKERS" in gui_source
    assert 'QLabel("Paragraph alignment:")' in settings_source
    assert 'QLabel("Paragraph justification:")' in settings_source
    assert 'pdf_paragraph_alignment' in gui_source
    assert 'pdf_paragraph_justification' in gui_source
    assert "PDF_PARAGRAPH_ALIGNMENT" in gui_source
    assert "PDF_PARAGRAPH_JUSTIFICATION" in gui_source
    assert '"Right-to-left paragraph layout (RTL)"' in settings_source
    assert "pdf_rtl_paragraph_layout" in gui_source
    assert "PDF_RTL_PARAGRAPH_LAYOUT" in gui_source


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
    (payloads / output_file).write_text("translated", encoding="utf-8")
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

    readable_output = "response_pdf_section_003.html"
    assert FileUtilities.create_chapter_filename(current_chapter, 3) == readable_output
    assert manager.reconcile_pdf_chapter_entries([current_chapter]) == 0
    assert manager.prog["chapters"][f"pdf:{section_id}"]["actual_num"] == 3
    assert manager.prog["chapters"][f"pdf:{section_id}"]["output_file"] == readable_output
    assert not (payloads / output_file).exists()
    assert (payloads / readable_output).read_text(encoding="utf-8") == "translated"
    assert FileUtilities.create_chapter_filename(current_chapter, 3) == readable_output
    manager.migrate_to_content_hash([current_chapter])
    assert f"pdf:{section_id}" in manager.prog["chapters"]


def test_legacy_pdf_progress_hash_migrates_once_then_detects_source_changes(
        tmp_path):
    payloads = tmp_path / "Payloads"
    payloads.mkdir()
    manager = ProgressManager(str(payloads))
    section_id = "legacy-stable-id"
    manager.prog["chapters"] = {
        f"pdf:{section_id}": {
            "actual_num": 1,
            "content_hash": "pre-image-rename-hash",
            "output_file": f"response_pdf_section_{section_id}.html",
            "status": "completed",
        },
        "pdf:outline:1": {
            "actual_num": 1,
            "content_hash": "",
            "output_file": "response_pdf_section_1.html",
            "status": "not_translated",
            "pdf_toc_section": True,
            "pdf_outline_seed": True,
        },
    }
    current = {
        "num": 1,
        "title": "Opening",
        "filename": "pdf_section_1.html",
        "content_hash": "canonical-payload-hash",
        "pdf_toc_section": True,
        "pdf_section_id": section_id,
    }

    assert manager.reconcile_pdf_chapter_entries([current]) == 1
    migrated = manager.prog["chapters"][f"pdf:{section_id}"]
    assert migrated["content_hash"] == "canonical-payload-hash"
    assert migrated["pdf_section_id"] == section_id
    assert migrated["pdf_content_hash_version"] == 2

    changed = dict(current, content_hash="genuinely-updated-source")
    assert manager.reconcile_pdf_chapter_entries([changed]) == 1
    assert manager.prog["chapters"] == {}


def test_workspace_reader_manifest_orders_pdf_entries_and_hides_sidecars(tmp_path):
    workspace = tmp_path / "Book"
    workspace.mkdir()
    source = tmp_path / "Book.pdf"
    source.write_bytes(b"pdf placeholder")
    (workspace / "source_epub.txt").write_text(str(source), encoding="utf-8")
    (workspace / "response_two.html").write_text("<p>two</p>", encoding="utf-8")
    (workspace / "response_ten.html").write_text("<p>ten</p>", encoding="utf-8")
    (workspace / "translation_progress.json").write_text(
        json.dumps(
            {
                "chapters": {
                    "pdf:outline:10": {
                        "actual_num": 10,
                        "output_file": "response_ten.html",
                        "original_basename": "pdf_section_10.html",
                        "pdf_toc_section": True,
                        "pdf_toc_title": "Chapter Ten",
                        "pdf_start_page": 20,
                        "pdf_end_page": 24,
                    },
                    "special_source_epub": {
                        "actual_num": 0,
                        "output_file": "source_epub.txt",
                        "original_basename": "source_epub.txt",
                    },
                    "pdf:outline:2": {
                        "actual_num": 2,
                        "output_file": "response_two.html",
                        "original_basename": "pdf_section_2.html",
                        "pdf_toc_section": True,
                        "pdf_toc_title": "Chapter Two",
                        "pdf_start_page": 3,
                        "pdf_end_page": 7,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    manifest = build_workspace_reader_manifest(str(workspace))

    assert manifest["source_format"] == "pdf"
    assert [entry["title"] for entry in manifest["entries"]] == [
        "Chapter Two",
        "Chapter Ten",
    ]
    assert manifest["entries"][0]["filename"] == "response_two.html"
    assert manifest["entries"][0]["pdf_start_page"] == 3
    assert manifest["entries"][0]["pdf_end_page"] == 7


def test_pdf_reader_raw_cache_extracts_only_requested_range_and_invalidates(
        tmp_path, monkeypatch):
    workspace = tmp_path / "Book"
    workspace.mkdir()
    source = tmp_path / "Book.pdf"
    source.write_bytes(b"first version")
    manifest = {
        "workspace": str(workspace),
        "source_path": str(source),
        "source_format": "pdf",
    }
    entry = {
        "key": "pdf:outline:8",
        "title": "Chapter Eight",
        "pdf_start_page": 41,
        "pdf_end_page": 52,
    }
    calls = []

    def fake_extract(_source, _workspace, **kwargs):
        calls.append(kwargs)
        return [
            (page, f"<html><body><p>raw page {page}</p></body></html>")
            for page in range(kwargs["start_page"], kwargs["end_page"] + 1)
        ]

    monkeypatch.setattr(
        "pdf_fast_extractor.extract_pdf_page_range_for_reader",
        fake_extract,
    )

    first = ensure_pdf_raw_section(manifest, entry, extract_images=False)
    second = ensure_pdf_raw_section(manifest, entry, extract_images=False)

    assert first == second
    assert len(calls) == 1
    assert calls[0]["start_page"] == 41
    assert calls[0]["end_page"] == 52
    assert "raw page 41" in Path(first).read_text(encoding="utf-8")
    assert "raw page 52" in Path(first).read_text(encoding="utf-8")

    metadata_path = next(
        (workspace / ".pdf_reader_cache" / "fast_semantic").glob("section_*.json")
    )
    stale_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    stale_metadata["version"] = 1
    metadata_path.write_text(json.dumps(stale_metadata), encoding="utf-8")
    ensure_pdf_raw_section(manifest, entry, extract_images=False)
    assert len(calls) == 2

    source.write_bytes(b"updated PDF version with an added bookmark")
    ensure_pdf_raw_section(manifest, entry, extract_images=False)
    assert len(calls) == 3

    monkeypatch.setenv("PDF_PARAGRAPH_JUSTIFICATION", "justify")
    ensure_pdf_raw_section(manifest, entry, extract_images=False)
    assert len(calls) == 4

    monkeypatch.setenv("PDF_RTL_PARAGRAPH_LAYOUT", "1")
    ensure_pdf_raw_section(manifest, entry, extract_images=False)
    assert len(calls) == 5


def test_reader_page_range_extractor_never_walks_unrequested_pdf_pages(
        tmp_path):
    pdf_path = tmp_path / "range.pdf"
    output_dir = tmp_path / "output"
    _make_image_pdf(pdf_path, page_count=6)

    pages = extract_pdf_page_range_for_reader(
        str(pdf_path),
        str(output_dir),
        start_page=2,
        end_page=3,
        mode="fast_semantic",
        extract_images=False,
        section_title="Middle",
    )

    assert [page_number for page_number, _html in pages] == [2, 3]
    page_cache = output_dir / ".pdf_extraction_cache" / "pages" / "fast_semantic"
    assert not (page_cache / "page_000001.json").exists()
    assert (page_cache / "page_000002.json").is_file()
    assert (page_cache / "page_000003.json").is_file()
    assert not (page_cache / "page_000004.json").exists()
