import json
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


def test_library_compile_action_is_pdf_aware():
    root = Path(__file__).resolve().parents[1]
    library_source = (root / "src" / "epub_library.py").read_text(encoding="utf-8")
    gui_source = (root / "src" / "translator_gui.py").read_text(encoding="utf-8")

    assert 'menu.addAction("\\U0001f4c4  Compile PDF")' in library_source
    assert '_workspace_compile_kind(book, compile_folder)' in library_source
    assert 'converter_name = (' in library_source
    assert '"pdf_converter" if compile_kind == "pdf"' in library_source
    assert "def pdf_converter(self, folder=None):" in gui_source
