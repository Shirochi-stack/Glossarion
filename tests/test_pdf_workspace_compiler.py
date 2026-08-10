import json
import time
from pathlib import Path

import fitz
from bs4 import BeautifulSoup

from output_workspace import write_workspace_source_reference
from pdf_workspace_compiler import (
    _workspace_response_entries,
    compile_pdf_workspace,
    normalize_fast_semantic_paragraph_alignment,
    translate_pdf_workspace_artifacts,
)


def _make_pdf_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "Novel_PDF"
    write_workspace_source_reference(workspace, tmp_path / "Novel.pdf")
    (workspace / "response_pdf_section_10.html").write_text(
        "<html><body><h3>Body sentence heading</h3><p>Second body.</p></body></html>",
        encoding="utf-8",
    )
    (workspace / "response_pdf_section_2.html").write_text(
        '<html><body><article class="pdf-fast-semantic-page">'
        '<h3>Another sentence heading</h3>'
        '<p class="pdf-align-center" style="text-align:center">First body.</p>'
        '</article></body></html>',
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


def test_workspace_responses_prefer_translated_pdf_bookmark_titles(tmp_path):
    workspace = _make_pdf_workspace(tmp_path)
    progress_path = workspace / "translation_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["chapters"]["pdf:first"][
        "pdf_toc_title_translated"
    ] = "Translated Chapter Two"
    progress_path.write_text(json.dumps(progress), encoding="utf-8")

    entries = _workspace_response_entries(str(workspace))

    assert entries[0][1] == "Translated Chapter Two"


def test_pdf_bookmarks_and_html_headers_use_shared_batch_translation(
    tmp_path,
    monkeypatch,
):
    workspace = tmp_path / "Book_PDF"
    workspace.mkdir()
    response_one = workspace / "response_pdf_section_1.html"
    response_two = workspace / "response_pdf_section_2.html"
    response_one.write_text(
        "<html><body><h1>첫 장</h1><p>Body one.</p></body></html>",
        encoding="utf-8",
    )
    response_two.write_text(
        "<html><body><h1>둘째 장</h1><h2>소제목</h2><p>Body two.</p></body></html>",
        encoding="utf-8",
    )
    progress_path = workspace / "translation_progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "version": "2.1",
                "chapters": {
                    "pdf:one": {
                        "actual_num": 1,
                        "status": "completed",
                        "output_file": response_one.name,
                        "pdf_toc_section": True,
                        "pdf_section_id": "one",
                        "pdf_toc_title": "첫 장",
                    },
                    "pdf:two": {
                        "actual_num": 2,
                        "status": "completed",
                        "output_file": response_two.name,
                        "pdf_toc_section": True,
                        "pdf_section_id": "two",
                        "pdf_toc_title": "둘째 장",
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    chapters = [
        {
            "num": 1,
            "title": "첫 장",
            "body": "<html><body><h1>첫 장</h1></body></html>",
            "pdf_toc_section": True,
            "pdf_section_id": "one",
            "pdf_toc_title": "첫 장",
        },
        {
            "num": 2,
            "title": "둘째 장",
            "body": (
                "<html><body><h1>둘째 장</h1>"
                "<h2>소제목</h2></body></html>"
            ),
            "pdf_toc_section": True,
            "pdf_section_id": "two",
            "pdf_toc_title": "둘째 장",
        },
    ]
    calls = []

    def fake_translate(self, headers, batch_size=None, translation_type="header"):
        calls.append((translation_type, batch_size, dict(headers)))
        return {
            number: f"EN:{source}"
            for number, source in headers.items()
        }

    from metadata_batch_translator import BatchHeaderTranslator

    monkeypatch.setattr(
        BatchHeaderTranslator, "translate_headers_batch", fake_translate
    )
    monkeypatch.setenv("USE_TOC_NCX", "1")
    monkeypatch.setenv("BATCH_TRANSLATE_HEADERS", "1")
    monkeypatch.setenv("TOC_NCX_PER_BATCH", "7")
    monkeypatch.setenv("HEADERS_PER_BATCH", "5")

    class FakeClient:
        model = "test-model"
        output_dir = str(workspace)

    result = translate_pdf_workspace_artifacts(
        chapters,
        str(workspace),
        FakeClient(),
    )

    assert result == {"toc": 2, "headers": 3}
    assert [(kind, size) for kind, size, _headers in calls] == [
        ("toc", 7),
        ("headers", 5),
    ]
    assert (workspace / "TOC.txt").is_file()
    assert (workspace / "translated_headers.txt").is_file()
    first_soup = BeautifulSoup(
        response_one.read_text(encoding="utf-8"), "html.parser"
    )
    second_soup = BeautifulSoup(
        response_two.read_text(encoding="utf-8"), "html.parser"
    )
    assert first_soup.h1.get_text(strip=True) == "EN:첫 장"
    assert [
        node.get_text(strip=True)
        for node in second_soup.find_all(["h1", "h2"])
    ] == ["EN:둘째 장", "EN:소제목"]
    saved_progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert saved_progress["chapters"]["pdf:one"][
        "pdf_toc_title_translated"
    ] == "EN:첫 장"
    assert "__translation_artifact__:toc" in saved_progress["chapters"]
    assert "__translation_artifact__:headers" in saved_progress["chapters"]

    # Simulate an updated source PDF that inserted one bookmark. Existing
    # labels must be matched by source text rather than their shifted numeric
    # positions, so only the new bookmark reaches the API. Its identical h1 is
    # then reused from the freshly updated TOC cache.
    response_new = workspace / "response_pdf_section_15.html"
    response_new.write_text(
        "<html><body><h1>새 장</h1><p>New body.</p></body></html>",
        encoding="utf-8",
    )
    saved_progress["chapters"]["pdf:new"] = {
        "actual_num": 1.5,
        "status": "completed",
        "output_file": response_new.name,
        "pdf_toc_section": True,
        "pdf_section_id": "new",
        "pdf_toc_title": "새 장",
    }
    progress_path.write_text(
        json.dumps(saved_progress, ensure_ascii=False), encoding="utf-8"
    )
    updated_chapters = [
        chapters[0],
        {
            "num": 1.5,
            "title": "새 장",
            "body": "<html><body><h1>새 장</h1></body></html>",
            "pdf_toc_section": True,
            "pdf_section_id": "new",
            "pdf_toc_title": "새 장",
        },
        chapters[1],
    ]
    calls.clear()

    updated_result = translate_pdf_workspace_artifacts(
        updated_chapters,
        str(workspace),
        FakeClient(),
    )

    assert updated_result == {"toc": 3, "headers": 4}
    assert len(calls) == 1
    assert calls[0][0:2] == ("toc", 7)
    assert list(calls[0][2].values()) == ["새 장"]
    assert BeautifulSoup(
        response_new.read_text(encoding="utf-8"), "html.parser"
    ).h1.get_text(strip=True) == "EN:새 장"


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
    compiled_html = (workspace / "Novel_translated.html").read_text(
        encoding="utf-8"
    )
    assert ".pdf-fast-semantic-page p.pdf-align-justify {" in compiled_html
    assert ".compiled-pdf-section + .compiled-pdf-section {" in compiled_html
    assert "break-before: page;" in compiled_html
    assert "page-break-before: always;" in compiled_html
    compiled_soup = BeautifulSoup(compiled_html, "html.parser")
    first_body = compiled_soup.find("p", string="First body.")
    assert first_body["class"] == ["pdf-align-center"]
    assert first_body["style"] == "text-align:center"


def test_pdf_compiler_paragraph_formatting_overrides(monkeypatch):
    content = (
        '<article class="pdf-fast-semantic-page">'
        '<p class="pdf-align-center" data-pdf-source-alignment="justify" '
        'style="color:red;text-align:center">Body.</p>'
        '</article>'
    )

    monkeypatch.setenv("PDF_PARAGRAPH_ALIGNMENT", "source")
    monkeypatch.setenv("PDF_PARAGRAPH_JUSTIFICATION", "source")
    preserved = BeautifulSoup(
        normalize_fast_semantic_paragraph_alignment(content), "html.parser"
    ).p
    assert preserved["class"] == ["pdf-align-justify"]
    assert preserved["style"] == "color:red;text-align:justify"

    monkeypatch.setenv("PDF_PARAGRAPH_ALIGNMENT", "right")
    monkeypatch.setenv("PDF_PARAGRAPH_JUSTIFICATION", "none")
    right_aligned = BeautifulSoup(
        normalize_fast_semantic_paragraph_alignment(content), "html.parser"
    ).p
    assert right_aligned["class"] == ["pdf-align-right"]
    assert right_aligned["style"] == "color:red;text-align:right"

    monkeypatch.setenv("PDF_PARAGRAPH_ALIGNMENT", "left")
    monkeypatch.setenv("PDF_PARAGRAPH_JUSTIFICATION", "justify")
    justified = BeautifulSoup(
        normalize_fast_semantic_paragraph_alignment(content), "html.parser"
    ).p
    assert justified["class"] == ["pdf-align-justify"]
    assert justified["style"] == "color:red;text-align:justify"

    monkeypatch.setenv("PDF_RTL_PARAGRAPH_LAYOUT", "1")
    rtl = BeautifulSoup(
        normalize_fast_semantic_paragraph_alignment(content), "html.parser"
    ).article
    assert rtl["dir"] == "rtl"
    assert rtl["data-pdf-rtl-layout"] == "true"
    assert "pdf-rtl-layout" in rtl["class"]

    monkeypatch.setenv("PDF_RTL_PARAGRAPH_LAYOUT", "0")
    without_rtl = BeautifulSoup(
        normalize_fast_semantic_paragraph_alignment(str(rtl)), "html.parser"
    ).article
    assert without_rtl.get("dir") is None
    assert without_rtl.get("data-pdf-rtl-layout") is None
    assert "pdf-rtl-layout" not in without_rtl["class"]


def test_compile_pdf_workspace_keeps_links_tables_and_graphics(tmp_path, monkeypatch):
    workspace = tmp_path / "Structured_PDF"
    workspace.mkdir()
    write_workspace_source_reference(workspace, tmp_path / "Structured.pdf")
    images_dir = workspace / "images"
    images_dir.mkdir()
    (images_dir / "diagram.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="160" height="80" '
        'viewBox="0 0 160 80"><rect x="5" y="5" width="150" height="70" '
        'rx="8" fill="#ffcc33" stroke="#b02020" stroke-width="4"/></svg>',
        encoding="utf-8",
    )
    response = workspace / "response_pdf_section_structured.html"
    response.write_text(
        '<html><body><article class="pdf-fast-semantic-page">'
        '<p>Read <a href="https://academy.example/program">the linked lesson</a>.</p>'
        '<table class="pdf-table"><thead><tr><th>Item</th><th>Value</th></tr></thead>'
        '<tbody><tr><td>Speed</td><td>Fast</td></tr></tbody></table>'
        '<figure class="pdf-vector-graphic"><img src="images/diagram.svg" '
        'alt="Diagram"></figure></article></body></html>',
        encoding="utf-8",
    )
    (workspace / "translation_progress.json").write_text(
        json.dumps(
            {
                "chapters": {
                    "pdf:structured": {
                        "actual_num": 1,
                        "status": "completed",
                        "output_file": response.name,
                        "pdf_section_title": "Structured",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    def fake_create_pdf_from_html(
        html_content,
        output_path,
        css_path=None,
        images_dir=None,
    ):
        del css_path, images_dir
        soup = BeautifulSoup(html_content, "html.parser")
        document = fitz.open()
        page = document.new_page(width=595, height=842)
        page.insert_textbox(
            fitz.Rect(50, 50, 545, 500),
            soup.get_text(" ", strip=True),
            fontsize=11,
        )
        link = soup.find("a", href=True)
        page.insert_link(
            {
                "kind": fitz.LINK_URI,
                "from": fitz.Rect(50, 50, 220, 72),
                "uri": link["href"],
            }
        )
        page.draw_rect(
            fitz.Rect(50, 520, 210, 600),
            color=(0.7, 0.1, 0.1),
            fill=(1.0, 0.8, 0.2),
        )
        document.save(output_path)
        document.close()
        return True

    import pdf_extractor

    monkeypatch.setattr(
        pdf_extractor,
        "create_pdf_from_html",
        fake_create_pdf_from_html,
    )

    output = compile_pdf_workspace(str(workspace))

    compiled_html = (workspace / "Structured_translated.html").read_text(
        encoding="utf-8"
    )
    compiled_soup = BeautifulSoup(compiled_html, "html.parser")
    assert compiled_soup.find("a", href="https://academy.example/program") is not None
    assert compiled_soup.find("table", class_="pdf-table") is not None
    assert compiled_soup.find("img", src="images/diagram.svg") is not None
    with fitz.open(output) as document:
        links = [link for page in document for link in (page.get_links() or [])]
        text = "\n".join(page.get_text() for page in document)
        drawings = [drawing for page in document for drawing in page.get_drawings()]
    assert any(link.get("uri") == "https://academy.example/program" for link in links)
    assert "Item" in text and "Speed" in text and "Fast" in text
    assert drawings


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
    assert "api_client=pdf_api_client" in gui_source
    assert "'USE_TOC_NCX'" in gui_source
    assert "'BATCH_TRANSLATE_HEADERS'" in gui_source


def test_new_runtime_modules_are_packaged_in_all_desktop_specs():
    root = Path(__file__).resolve().parents[1]
    required_modules = (
        "installer_utils",
        "pdf_bookmarks",
        "output_workspace",
        "pdf_fast_extractor",
        "pdf_workspace_compiler",
        "workspace_reader",
        "pdf_output_naming",
    )
    spec_names = (
        "translator.spec",
        "translator_Heavy.spec",
        "translator_NoCuda.spec",
        "translator_TurboLite.spec",
        "translator_lite.spec",
        "translator_lite_linux.spec",
        "translator_lite_mac.spec",
        "translator_lite_mac_NoCuda.spec",
        "translator_lite_mac_intel.spec",
        "translator_lite_mac_intel_NoCuda.spec",
        "translatoronefileoff.spec",
    )

    for spec_name in spec_names:
        source = (root / "src" / spec_name).read_text(encoding="utf-8")
        for module in required_modules:
            assert source.count(f"('{module}.py', '.')") == 1, (
                spec_name,
                module,
                "app_files",
            )
            assert source.count(f"'{module}',") == 1, (
                spec_name,
                module,
                "app_modules",
            )
