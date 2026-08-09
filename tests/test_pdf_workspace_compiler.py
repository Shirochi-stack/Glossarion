import json
import time
from pathlib import Path

import fitz
from bs4 import BeautifulSoup

from output_workspace import write_workspace_source_reference
from pdf_workspace_compiler import (
    _workspace_response_entries,
    compile_pdf_workspace,
)


def _make_pdf_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "Novel_PDF"
    write_workspace_source_reference(workspace, tmp_path / "Novel.pdf")
    (workspace / "response_pdf_section_10.html").write_text(
        "<html><body><h3>Body sentence heading</h3><p>Second body.</p></body></html>",
        encoding="utf-8",
    )
    (workspace / "response_pdf_section_2.html").write_text(
        "<html><body><h3>Another sentence heading</h3><p>First body.</p></body></html>",
        encoding="utf-8",
    )
    (workspace / "translation_progress.json").write_text(
        json.dumps({
            "chapters": {
                "pdf:second": {
                    "actual_num": 10,
                    "status": "completed",
                    "output_file": "response_pdf_section_10.html",
                    "pdf_section_title": "Chapter Ten",
                },
                "pdf:first": {
                    "actual_num": 2,
                    "status": "completed",
                    "output_file": "response_pdf_section_2.html",
                    "pdf_section_title": "Chapter Two",
                },
            }
        }),
        encoding="utf-8",
    )
    return workspace


def test_workspace_responses_follow_progress_order_not_lexical_order(tmp_path):
    workspace = _make_pdf_workspace(tmp_path)

    entries = _workspace_response_entries(str(workspace))

    assert [Path(path).name for path, _title in entries] == [
        "response_pdf_section_2.html",
        "response_pdf_section_10.html",
    ]
    assert [title for _path, title in entries] == ["Chapter Two", "Chapter Ten"]


def test_compile_pdf_workspace_has_one_bookmark_per_response(tmp_path, monkeypatch):
    workspace = _make_pdf_workspace(tmp_path)

    def fake_create_pdf_from_html(
        html_content, output_path, css_path=None, images_dir=None
    ):
        del css_path, images_dir
        soup = BeautifulSoup(html_content, "html.parser")
        titles = [
            node.get_text(" ", strip=True)
            for node in soup.select(".pdf-bookmark-anchor")
        ]
        body_text = soup.get_text(" ", strip=True)
        document = fitz.open()
        page = document.new_page()
        text_rect = fitz.Rect(
            50, 50, page.rect.width - 50, page.rect.height - 50)
        page.insert_textbox(text_rect, body_text)
        document.set_toc([[1, title, 1] for title in titles])
        document.save(output_path)
        document.close()
        return True

    import pdf_extractor

    monkeypatch.setattr(
        pdf_extractor, "create_pdf_from_html", fake_create_pdf_from_html)

    output = compile_pdf_workspace(str(workspace))

    assert Path(output).name == "Novel_translated.pdf"
    assert Path(output).is_file()
    with fitz.open(output) as document:
        assert document.page_count >= 1
        toc_titles = [row[1] for row in document.get_toc(simple=True)]
        assert toc_titles == [
            "Chapter Two",
            "Chapter Ten",
        ]
        page_text = "\n".join(page.get_text() for page in document)
    assert "First body." in page_text
    assert "Second body." in page_text
    assert "Body sentence heading" not in toc_titles


def test_compile_pdf_repairs_legacy_page_images_from_only_requested_pages(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "Illustrated.pdf"
    document = fitz.open()
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 20, 20), False)
    pixmap.clear_with(0x336699)
    image_bytes = pixmap.tobytes("png")
    for page_number in range(1, 5):
        page = document.new_page(width=240, height=320)
        page.insert_text((20, 30), f"Page {page_number}")
        if page_number == 4:
            page.insert_image(fitz.Rect(20, 50, 120, 150), stream=image_bytes)
    document.save(source)
    document.close()

    workspace = tmp_path / "Illustrated_PDF"
    write_workspace_source_reference(workspace, source)
    response = workspace / "response_pdf_section_1.html"
    response.write_text(
        '<html><body><p>Translated</p><img src="images/page_4_img_1.png"></body></html>',
        encoding="utf-8",
    )
    (workspace / "translation_progress.json").write_text(
        json.dumps({
            "chapters": {
                "pdf:first": {
                    "actual_num": 1,
                    "status": "completed",
                    "output_file": response.name,
                    "pdf_section_title": "Illustrated",
                }
            }
        }),
        encoding="utf-8",
    )
    captured = {}

    def fake_create_pdf_from_html(
        html_content, output_path, css_path=None, images_dir=None
    ):
        del css_path
        soup = BeautifulSoup(html_content, "html.parser")
        src = soup.find("img")["src"]
        captured["src"] = src
        captured["image_exists"] = Path(images_dir, Path(src).name).is_file()
        time.sleep(0.08)
        output = fitz.open()
        output.new_page()
        output.save(output_path)
        output.close()
        return True

    import pdf_extractor

    monkeypatch.setattr(
        pdf_extractor, "create_pdf_from_html", fake_create_pdf_from_html
    )
    monkeypatch.setenv("PDF_COMPILE_HEARTBEAT_SECONDS", "0.05")
    logs = []
    output = compile_pdf_workspace(str(workspace), log_callback=logs.append)

    assert Path(output).is_file()
    assert captured["src"].startswith("images/pdfimg_")
    assert captured["image_exists"] is True
    assert any("1 specifically referenced PDF page" in message for message in logs)
    assert any("1 reference(s) repaired, 0 unresolved" in message for message in logs)
    assert any("PDF renderer heartbeat:" in message for message in logs)
    targeted = workspace / ".pdf_extraction_cache" / "targeted_images"
    assert len(list(targeted.glob("*_page_000004.json"))) == 1
    assert not list(targeted.glob("*_page_000001.json"))
    assert not list(targeted.glob("*_page_000002.json"))
    assert not list(targeted.glob("*_page_000003.json"))


def test_library_compile_action_is_pdf_aware():
    root = Path(__file__).resolve().parents[1]
    library_source = (root / "src" / "epub_library.py").read_text(encoding="utf-8")
    gui_source = (root / "src" / "translator_gui.py").read_text(encoding="utf-8")

    assert 'menu.addAction("\\U0001f4c4  Compile PDF")' in library_source
    assert '_workspace_compile_kind(book, compile_folder)' in library_source
    assert 'converter_name = (' in library_source
    assert '"pdf_converter" if compile_kind == "pdf"' in library_source
    assert "def pdf_converter(self, folder=None):" in gui_source
