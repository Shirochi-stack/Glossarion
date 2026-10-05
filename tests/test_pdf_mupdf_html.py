"""pdf_mupdf_html: the WeasyPrint-subset shim over PyMuPDF ``fitz.Story``.

Glossarion Mobile has no WeasyPrint, so the PDF call sites import ``HTML``,
``CSS`` and ``FontConfiguration`` from the shim when it is selected (mobile,
or ``GLOSSARION_PDF_ENGINE=mupdf``). These tests render through the shim
directly and through the real call sites (_pdf_worker in-process mode via
PdfGenerationManager, pdf_extractor.create_pdf_from_html).
"""
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

fitz = pytest.importorskip("fitz")

import mobile_runtime  # noqa: E402
import pdf_mupdf_html  # noqa: E402
from pdf_bookmarks import replace_with_chapter_bookmarks  # noqa: E402

_MOBILE_ENV = (
    "GLOSSARION_MOBILE",
    "GLOSSARION_NO_PROCESSES",
    "GLOSSARION_DATA_DIR",
    "GLOSSARION_PDF_ENGINE",
    "FLET_PLATFORM",
)


@pytest.fixture
def clean_env(monkeypatch):
    for name in _MOBILE_ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def import_pdf_worker():
    """Import _pdf_worker without the worker-process stdout re-wrap (see its header)."""
    if "_pdf_worker" not in sys.modules:
        previous = os.environ.get("GLOSSARION_NO_PROCESSES")
        os.environ["GLOSSARION_NO_PROCESSES"] = "1"
        try:
            import _pdf_worker  # noqa: F401
        finally:
            if previous is None:
                os.environ.pop("GLOSSARION_NO_PROCESSES", None)
            else:
                os.environ["GLOSSARION_NO_PROCESSES"] = previous
    return sys.modules["_pdf_worker"]


def _png(path, color=(30, 120, 200), size=(80, 50)):
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, size[0], size[1]), False)
    pixmap.set_rect(pixmap.irect, color)
    pixmap.save(str(path))


def _three_chapter_html():
    chapters = []
    samples = ("한국어 본문 문장입니다.", "中文正文句子。", "日本語の本文です。")
    for number, sample in enumerate(samples, 1):
        style = ' style="page-break-before: always;"' if number > 1 else ""
        chapters.append(
            f'<div id="chapter-{number}"{style}>'
            f"<h1>Chapter {number} 第{number}章</h1>"
            f"<p>{sample} Plain English line {number}.</p>"
            f'<img src="images/picture.png" alt="picture"/>'
            "</div>"
        )
    return (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"/><title>Shim Fixture</title>"
        "<style>body { font-family: serif; } img { max-width: 100%; }</style>"
        "</head><body>" + "".join(chapters) + "</body></html>"
    )


def test_engine_selection_gate(clean_env):
    assert pdf_mupdf_html.is_available()
    assert not pdf_mupdf_html.is_selected()
    clean_env.setenv("GLOSSARION_NO_PROCESSES", "1")
    assert not pdf_mupdf_html.is_selected()  # the process gate does not pick the engine
    clean_env.setenv("GLOSSARION_PDF_ENGINE", "mupdf")
    assert pdf_mupdf_html.is_selected()
    clean_env.delenv("GLOSSARION_PDF_ENGINE")
    clean_env.setenv("GLOSSARION_MOBILE", "1")
    assert mobile_runtime.is_mobile()
    assert pdf_mupdf_html.is_selected()


def test_three_chapter_cjk_document_with_image_renders_with_outline(tmp_path):
    base = tmp_path / "출력_книга"
    (base / "images").mkdir(parents=True)
    _png(base / "images" / "picture.png")
    target = base / "book.pdf"

    document = pdf_mupdf_html.HTML(string=_three_chapter_html(), base_url=str(base)).render()
    assert len(document.pages) == 3
    assert [sorted(page.anchors) for page in document.pages] == [
        ["chapter-1"], ["chapter-2"], ["chapter-3"]]
    document.write_pdf(str(target))

    with fitz.open(str(target)) as pdf:
        assert pdf.page_count > 0
        assert [(level, title, page) for level, title, page in pdf.get_toc()] == [
            (1, "Chapter 1 第1章", 1),
            (1, "Chapter 2 第2章", 2),
            (1, "Chapter 3 第3章", 3),
        ]
        text = "".join(page.get_text() for page in pdf)
        for sample in ("한국어 본문", "中文正文", "日本語の本文"):
            assert sample in text
        assert all(page.get_images() for page in pdf)
        assert pdf.metadata.get("title") == "Shim Fixture"
        assert round(pdf[0].rect.width) == 595 and round(pdf[0].rect.height) == 842


def test_page_size_margins_and_page_number_boxes(tmp_path):
    html = (
        "<html><head><style>"
        "@page { size: A5 landscape; margin: 20mm; }"
        "@page { @bottom-right { content: counter(page); color: rgba(0,0,0,0.4); font-size: 10pt; } }"
        "</style></head><body><p>one</p><p style='page-break-before: always'>two</p></body></html>"
    )
    data = pdf_mupdf_html.HTML(string=html).write_pdf()
    assert data.startswith(b"%PDF")
    with fitz.open("pdf", data) as pdf:
        assert pdf.page_count == 2
        width, height = pdf[0].rect.width, pdf[0].rect.height
        assert round(width) == 595 and round(height) == 420
        for index, page in enumerate(pdf, 1):
            footer = fitz.Rect(width / 2, height - 20 * 72 / 25.4, width, height)
            assert page.get_textbox(footer).strip() == str(index)
            words = page.get_text("words")
            body = [w for w in words if w[4] in ("one", "two")]
            assert body and body[0][0] >= 20 * 72 / 25.4 - 1

    # A later "content: none" (the cover/TOC pattern) turns the number off.
    suppressed = html.replace(
        "</style>", "@page { @bottom-right { content: none; } }</style>")
    with fitz.open("pdf", pdf_mupdf_html.HTML(string=suppressed).write_pdf()) as pdf:
        assert pdf[0].get_text().split() == ["one"]


def test_cross_document_copy_keeps_links_and_compiler_bookmarks(tmp_path):
    toc_html = (
        "<html><head><style>* { bookmark-level: none !important; }</style></head><body>"
        "<h1>Table of Contents</h1><ul><li><a href=\"#chapter-2\">Second</a></li></ul>"
        "</body></html>"
    )
    toc = pdf_mupdf_html.HTML(string=toc_html).render()
    assert toc.pages[0].bookmarks == []  # bookmark-level: none, like WeasyPrint

    chapters = pdf_mupdf_html.HTML(string=_three_chapter_html()).render()
    added = replace_with_chapter_bookmarks(
        chapters.pages,
        [("a.xhtml", 1, "One"), ("b.xhtml", 2, "Two"), ("c.xhtml", 3, "Three")],
    )
    assert added == 3

    merged = toc.copy([page for doc in (toc, chapters) for page in doc.pages])
    sink = io.BytesIO()
    assert merged.write_pdf(sink) is None
    with fitz.open("pdf", sink.getvalue()) as pdf:
        assert pdf.page_count == 4
        assert [(title, page) for _level, title, page in pdf.get_toc()] == [
            ("One", 2), ("Two", 3), ("Three", 4)]
        links = pdf[0].get_links()
        assert links and {link["page"] for link in links} == {2}


def test_page_breaks_on_first_element_and_css3_breaks_do_not_stall():
    started = time.time()
    document = pdf_mupdf_html.HTML(
        string="<body><div style='page-break-before: always'><h2>First</h2><p>x</p></div></body>"
    ).render()
    assert len(document.pages) == 1
    assert document.pages[0].bookmarks[0][1] == "First"
    css3 = pdf_mupdf_html.HTML(
        string="<html><head><style>.b { break-before: page; }</style></head>"
               "<body><p class='b'>a</p><p class='b'>b</p><p>c</p></body></html>"
    ).render()
    assert len(css3.pages) == 2
    assert time.time() - started < 30


def test_css_html_and_font_configuration_inputs(tmp_path):
    css_file = tmp_path / "style.css"
    css_file.write_text("@page { size: letter; margin: 1in; } p { color: #333; }", encoding="utf-8")
    html_file = tmp_path / "page.html"
    html_file.write_text("<html><body><h1>From file</h1><p>body</p></body></html>", encoding="utf-8")
    font_config = pdf_mupdf_html.FontConfiguration()
    sheets = [
        pdf_mupdf_html.CSS(filename=str(css_file), font_config=font_config),
        pdf_mupdf_html.CSS(string="h1 { font-size: 20pt; }", font_config=font_config),
    ]
    target = tmp_path / "from_file.pdf"
    result = pdf_mupdf_html.HTML(filename=str(html_file)).write_pdf(
        str(target), stylesheets=sheets, font_config=font_config,
        jpeg_quality=80, optimize_images=True)
    assert result is None
    with fitz.open(str(target)) as pdf:
        assert round(pdf[0].rect.width) == 612 and round(pdf[0].rect.height) == 792
        assert pdf.get_toc() == [[1, "From file", 1]]
    uncompressed = pdf_mupdf_html.HTML(string="<p>x</p>").render().write_pdf(uncompressed_pdf=True)
    assert uncompressed.startswith(b"%PDF")


def test_concurrent_renders_are_serialised():
    results = []
    errors = []

    def work(number):
        try:
            data = pdf_mupdf_html.HTML(string=f"<h1>T{number}</h1><p>{'x ' * 400}</p>").write_pdf()
            with fitz.open("pdf", data) as pdf:
                results.append(pdf.get_toc()[0][1])
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert errors == []
    assert sorted(results) == ["T0", "T1", "T2", "T3"]


def test_create_pdf_from_html_uses_shim_when_selected(clean_env, tmp_path, capsys):
    import pdf_extractor

    clean_env.setenv("GLOSSARION_PDF_ENGINE", "mupdf")
    clean_env.setenv("PDF_REMOVE_BLANK_PAGES", "1")
    (tmp_path / "images").mkdir()
    _png(tmp_path / "images" / "p.png")
    css_path = tmp_path / "styles.css"
    css_path.write_text("body { margin: 1em; }", encoding="utf-8")
    output = tmp_path / "translated.pdf"

    ok = pdf_extractor.create_pdf_from_html(
        "<h1>Converted</h1><p>본문</p><img src=\"images/p.png\"/>",
        str(output), css_path=str(css_path), images_dir=str(tmp_path / "images"))

    assert ok is True
    assert "mupdf-story" in capsys.readouterr().out
    with fitz.open(str(output)) as pdf:
        assert pdf.page_count == 1
        assert "본문" in pdf[0].get_text()
        assert pdf[0].get_images()
        assert pdf.get_toc() == [[1, "Converted", 1]]


def _worker_fixture(tmp_path, chapters=2, batch_size="50"):
    """The hostile-outline fixture of test_pdf_toc_extraction, sized by ``chapters``."""
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
    names = ["One", "Two", "Three", "Four"]
    html_files, titles = [], {}
    for number in range(1, chapters + 1):
        name = f"chapter-{number}.html"
        html_files.append(name)
        titles[str(number)] = [f"Chapter {names[number - 1]}", 1.0, name]
        extra = ('<div class="page-break pdf-toc-page-break"></div><p>Third sentence</p>'
                 if number == 1 else "")
        (output_dir / name).write_text(
            f"<html><body><h{min(number, 2)} style=\"bookmark-level: 2 !important\">Sentence {number}</h{min(number, 2)}>"
            f"<p style=\"bookmark-level: 3 !important\">Body {number}</p>{extra}</body></html>",
            encoding="utf-8",
        )
    config_path = tmp_path / "pdf-config.json"
    config_path.write_text(json.dumps({
        "output_dir": str(output_dir),
        "images_dir": str(images_dir),
        "css_dir": str(css_dir),
        "html_files": html_files,
        "chapter_titles_info": titles,
        "processed_images": {},
        "cover_file": None,
        "metadata": {"title": "Outline Fixture"},
        "env_vars": {
            "PDF_PAGE_NUMBERS": "0",
            "PDF_GENERATE_TOC": "1",
            "PDF_TOC_PAGE_NUMBERS": "0",
            "PDF_RENDER_BATCH_SIZE": batch_size,
            "PDF_FAST_RENDERING": "0",
            "ENABLE_IMAGE_COMPRESSION": "0",
            "DEDUPLICATE_TOC": "0",
        },
    }), encoding="utf-8")
    return config_path, output_dir


def test_mobile_pdf_generation_runs_in_process_through_the_shim(clean_env, tmp_path, monkeypatch):
    import subprocess

    import pdf_generation_manager

    import_pdf_worker()
    clean_env.setenv("GLOSSARION_MOBILE", "1")  # selects the shim and disables subprocesses
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("subprocess spawned"))
    config_path, output_dir = _worker_fixture(tmp_path)
    logs = []
    finished = threading.Event()
    outcome = {}

    def completion(success, result):
        outcome.update(success=success, result=result)
        finished.set()

    manager = pdf_generation_manager.PdfGenerationManager(log_callback=logs.append)
    assert manager.generate_pdf_async(str(config_path), completion_callback=completion)
    assert finished.wait(120)

    assert outcome["success"] is True, logs
    assert any("mupdf-story" in message for message in logs)
    assert any("standard sequential compiler" in message for message in logs)
    pdf_files = list(output_dir.glob("*.pdf"))
    assert [Path(outcome["result"]["pdf_path"])] == pdf_files
    # Same structure the WeasyPrint subprocess test expects for this fixture.
    with fitz.open(str(pdf_files[0])) as document:
        assert document.page_count == 3
        assert [(level, title) for level, title, _page in document.get_toc()] == [
            (1, "Chapter One"), (1, "Chapter Two")]
        assert "Table of Contents" in document[0].get_text()


def test_inprocess_worker_stops_between_render_batches(clean_env, tmp_path):
    _pdf_worker = import_pdf_worker()
    clean_env.setenv("GLOSSARION_MOBILE", "1")
    config_path, output_dir = _worker_fixture(tmp_path, chapters=3, batch_size="1")
    lines = []

    def should_stop():
        return any("Standard render batch 1/3:" in line for line in lines)

    with pytest.raises(_pdf_worker.PdfGenerationStopped):
        _pdf_worker.run_pdf_generation(str(config_path), emit=lines.append, should_stop=should_stop)

    assert not any("render batch 2/3" in line for line in lines)
    assert not any(line.startswith("[RESULT]") for line in lines)
    assert list(output_dir.glob("*.pdf")) == []
    # The module-wide emitter is released after the in-process run.
    assert _pdf_worker._EMIT is None and _pdf_worker._SHOULD_STOP is None
