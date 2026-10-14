import posixpath
from urllib.parse import unquote
import xml.etree.ElementTree as ET
import zipfile

import pytest

from epub_package import find_epub_opf_member
from html_archive_epub import (
    convert_html_archive_to_epub, convert_html_file_to_epub,
    generated_html_epub_chapter_count, is_html_archive,
)
from image_archive_epub import ImageArchiveConversionCancelled, is_epub_zip


def _xhtml(title):
    return (f'<?xml version="1.0" encoding="utf-8"?>\n'
            f'<html xmlns="http://www.w3.org/1999/xhtml" lang="ko">'
            f'<head><title>{title}</title></head>'
            f'<body><h1>{title}</h1><p>Original 한국어 text.</p></body></html>').encode("utf-8")


def _archive(path, entries):
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return path


def _spine(archive):
    opf_name = find_epub_opf_member(archive)
    root = ET.fromstring(archive.read(opf_name))
    ns = {"opf": "http://www.idpf.org/2007/opf"}
    items = {item.get("id"): posixpath.normpath(posixpath.join(
        posixpath.dirname(opf_name), unquote(item.get("href")),
    )) for item in root.findall("opf:manifest/opf:item", ns)}
    return [items[item.get("idref")] for item in root.findall("opf:spine/opf:itemref", ns)]


def test_wrapped_xhtml_zip_preserves_chapters_and_uses_natural_order(tmp_path):
    chapters = {f"Book/chapter{number}.xhtml": _xhtml(f"Chapter {number}") for number in [10, 2, 1]}
    source = _archive(tmp_path / "Book.zip", {"Book/": b"", **chapters})
    output = tmp_path / "Book.epub"

    assert is_html_archive(source)
    assert not is_epub_zip(source)
    result = convert_html_archive_to_epub(source, output)

    assert result.epub_path == str(output)
    assert result.chapter_count == generated_html_epub_chapter_count(output) == 3
    assert is_epub_zip(output)
    with zipfile.ZipFile(output) as archive:
        assert archive.infolist()[0].filename == "mimetype"
        assert archive.infolist()[0].compress_type == zipfile.ZIP_STORED
        assert _spine(archive) == [f"Book/chapter{number}.xhtml" for number in [1, 2, 10]]
        for name, original in chapters.items():
            assert archive.read(name) == original
        for name in ["META-INF/container.xml", "_glossarion/content.opf", "_glossarion/nav.xhtml", "_glossarion/toc.ncx"]:
            ET.fromstring(archive.read(name))
    assert source.is_file()


def test_loose_html_normalized_with_relative_assets_and_links_intact(tmp_path):
    source = _archive(tmp_path / "loose.zip", {
        "Book/chapter 1.html": b'<html><head><title>First &amp; last</title></head><body><p>A&nbsp;B<br>Next<img src="images/one.png"><a href="chapter%202.htm">next</a>',
        "Book/chapter 2.htm": "<p>Chapter two 한국어".encode("utf-8"),
        "Book/images/one.png": b"image bytes", "Book/style.css": b"p { color: red; }",
    })
    output = tmp_path / "loose.epub"
    convert_html_archive_to_epub(source, output)
    with zipfile.ZipFile(output) as archive:
        chapter = ET.fromstring(archive.read("Book/chapter 1.html"))
        assert chapter.tag == "{http://www.w3.org/1999/xhtml}html"
        assert "A\u00a0B" in "".join(chapter.itertext())
        assert chapter.find(".//{http://www.w3.org/1999/xhtml}img").get("src") == "images/one.png"
        assert archive.read("Book/images/one.png") == b"image bytes"
        assert archive.read("Book/style.css") == b"p { color: red; }"
        assert _spine(archive) == ["Book/chapter 1.html", "Book/chapter 2.htm"]
        normalized = archive.read("Book/chapter 2.htm")
        assert b"<body>" in normalized
        assert "한국어" in normalized.decode("utf-8")
        assert ET.fromstring(normalized).find(".//{http://www.w3.org/1999/xhtml}title") is not None
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(normalized, "html.parser")
        assert soup.body is not None
        assert "한국어" in soup.body.find("p").get_text()


def test_existing_opf_spine_and_metadata_preserved(tmp_path):
    opf = b'''<package xmlns="http://www.idpf.org/2007/opf" version="3.0"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Original book title</dc:title><dc:creator>Original author</dc:creator></metadata><manifest><item id="one" href="chapter1.xhtml" media-type="application/xhtml+xml"/><item id="two" href="chapter2.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="two"/><itemref idref="one"/></spine></package>'''
    source = _archive(tmp_path / "repacked.zip", {
        "Novel/standard.opf": opf, "Novel/chapter1.xhtml": _xhtml("One"),
        "Novel/chapter2.xhtml": _xhtml("Two"), "Novel/nav.xhtml": _xhtml("Contents"),
    })
    output = tmp_path / "repacked.epub"
    result = convert_html_archive_to_epub(source, output)
    assert result.chapter_count == 2
    with zipfile.ZipFile(output) as archive:
        assert find_epub_opf_member(archive) == "Novel/standard.opf"
        assert archive.read("Novel/standard.opf") == opf
        assert _spine(archive) == ["Novel/chapter2.xhtml", "Novel/chapter1.xhtml"]
    assert generated_html_epub_chapter_count(output) == 2


@pytest.mark.parametrize("extra", ["other.zip", "notes.txt", "captions.srt", "../escape.html"])
def test_mixed_or_unsafe_archive_rejected_without_output(tmp_path, extra):
    source = _archive(tmp_path / "mixed.zip", {"chapter.html": _xhtml("One"), extra: b"extra"})
    output = tmp_path / "mixed.epub"
    assert not is_html_archive(source)
    with pytest.raises(ValueError):
        convert_html_archive_to_epub(source, output)
    assert not output.exists()


def test_cancellation_preserves_previous_output_and_removes_partial_file(tmp_path):
    source = _archive(tmp_path / "book.zip", {"chapter.xhtml": _xhtml("One")})
    output = tmp_path / "book.epub"
    output.write_bytes(b"existing output")

    def stop_after_output_created():
        return bool(list(tmp_path.glob(".html_archive_*.epub.tmp")))

    with pytest.raises(ImageArchiveConversionCancelled):
        convert_html_archive_to_epub(source, output, should_stop=stop_after_output_created)
    assert output.read_bytes() == b"existing output"
    assert not list(tmp_path.glob(".html_archive_*.epub.tmp"))


def test_generated_count_does_not_accept_unrelated_epub_or_invalid_zip(tmp_path):
    unrelated = _archive(tmp_path / "other.epub", {"mimetype": "application/epub+zip"})
    assert generated_html_epub_chapter_count(unrelated) == 0
    assert generated_html_epub_chapter_count(tmp_path / "missing.epub") == 0
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    assert not is_html_archive(bad)
    assert generated_html_epub_chapter_count(bad) == 0


@pytest.mark.parametrize("extension", [".html", ".htm", ".xhtml"])
def test_standalone_html_keeps_one_source_chapter_and_exact_xhtml(tmp_path, extension):
    source = tmp_path / ("Selected" + extension)
    original = _xhtml("Selected chapter")
    source.write_bytes(original)
    (tmp_path / "sibling.xhtml").write_bytes(_xhtml("Other chapter"))
    (tmp_path / "content.opf").write_text("Broken sibling OPF", encoding="utf-8")
    output = tmp_path / "Selected.epub"

    result = convert_html_file_to_epub(source, output)

    assert result.chapter_count == generated_html_epub_chapter_count(output) == 1
    assert result.epub_path == str(output)
    assert source.read_bytes() == original
    with zipfile.ZipFile(output) as archive:
        assert archive.read(source.name) == original
        assert _spine(archive) == [source.name]
        assert "sibling.xhtml" not in archive.namelist()
        assert "content.opf" not in archive.namelist()
        opf = ET.fromstring(archive.read(find_epub_opf_member(archive)))
        assert opf.find(".//{http://purl.org/dc/elements/1.1/}title").text == "Selected"
    assert not list(tmp_path.glob(".html_source_*.zip"))


def test_standalone_html_normalizes_loose_unicode_without_changing_source(tmp_path):
    from bs4 import BeautifulSoup

    source = tmp_path / "chapter.html"
    original = "<html><head><title>한국어 chapter</title></head><body><p>원본 이야기<br>다음 줄".encode("utf-8")
    source.write_bytes(original)
    output = tmp_path / "chapter.epub"

    convert_html_file_to_epub(source, output)

    with zipfile.ZipFile(output) as archive:
        chapter = archive.read(source.name)
        assert ET.fromstring(chapter).tag == "{http://www.w3.org/1999/xhtml}html"
        soup = BeautifulSoup(chapter, "html.parser")
        assert soup.body is not None
        assert soup.p.get_text(" ") == "원본 이야기 다음 줄"
    assert source.read_bytes() == original


def test_standalone_collects_only_referenced_local_assets_and_css_dependencies(tmp_path):
    chapters = tmp_path / "chapters"
    styles = tmp_path / "assets" / "css"
    images = tmp_path / "assets" / "images"
    fonts = tmp_path / "assets" / "fonts"
    for directory in (chapters, styles, images, fonts):
        directory.mkdir(parents=True)
    source = chapters / "chapter.xhtml"
    original = b'''<html xmlns="http://www.w3.org/1999/xhtml"><head><title>One</title><link rel="stylesheet" href="../assets/css/main.css"/></head><body><p>Only this chapter</p><img src="../assets/images/one%20image.png"/><a href="sibling.xhtml">next chapter</a><img src="https://example.invalid/remote.png"/><img src="data:image/png;base64,AAAA"/></body></html>'''
    source.write_bytes(original)
    (chapters / "sibling.xhtml").write_bytes(_xhtml("Sibling"))
    dependencies = {
        styles / "main.css": b'@import "base.css"; body { background: url(../images/bg.png) }',
        styles / "base.css": b'@import url("main.css"); @font-face { src: url(../fonts/book.woff2) }',
        images / "one image.png": b"selected image",
        images / "bg.png": b"background image",
        fonts / "book.woff2": b"font bytes",
    }
    for path, data in dependencies.items():
        path.write_bytes(data)
    (images / "unused.png").write_bytes(b"not referenced")
    output = tmp_path / "chapter.epub"

    convert_html_file_to_epub(source, output)

    with zipfile.ZipFile(output) as archive:
        assert _spine(archive) == ["chapters/chapter.xhtml"]
        assert archive.read("chapters/chapter.xhtml") == original
        for path, data in dependencies.items():
            assert archive.read(path.relative_to(tmp_path).as_posix()) == data
        assert "chapters/sibling.xhtml" not in archive.namelist()
        assert "assets/images/unused.png" not in archive.namelist()
        assert not any("remote" in name for name in archive.namelist())
    assert source.read_bytes() == original


def test_standalone_cancellation_removes_both_temporary_archives(tmp_path):
    source = tmp_path / "chapter.xhtml"
    original = _xhtml("One")
    source.write_bytes(original)
    output = tmp_path / "chapter.epub"
    output.write_bytes(b"previous output")

    def stop_after_output_created():
        return bool(list(tmp_path.glob(".html_archive_*.epub.tmp")))

    with pytest.raises(ImageArchiveConversionCancelled):
        convert_html_file_to_epub(source, output, should_stop=stop_after_output_created)

    assert output.read_bytes() == b"previous output"
    assert source.read_bytes() == original
    assert not list(tmp_path.glob(".html_source_*.zip"))
    assert not list(tmp_path.glob(".html_archive_*.epub.tmp"))


def test_standalone_rejects_overwriting_its_source(tmp_path):
    source = tmp_path / "chapter.html"
    original = _xhtml("One")
    source.write_bytes(original)
    with pytest.raises(ValueError, match="must differ"):
        convert_html_file_to_epub(source, source)
    assert source.read_bytes() == original


@pytest.mark.parametrize("filename", ["index.html", "cover.xhtml"])
def test_standalone_special_filenames_extract_as_regular_chapters(tmp_path, monkeypatch, filename):
    import Chapter_Extractor

    for name, value in {
        "EXTRACTION_MODE": "smart", "DOWNLOAD_REMOTE_IMAGE_URLS": "0",
        "EXTRACTION_WORKERS": "1", "DISABLE_CHAPTER_MERGING": "0",
        "ENABLE_GUI_YIELD": "0", "BATCH_TRANSLATE_HEADERS": "0",
        "REMOVE_DUPLICATE_H1_P": "0", "TRANSLATE_SPECIAL_FILES": "0",
    }.items():
        monkeypatch.setenv(name, value)
    for name in ("SINGLE_CHAPTER_FILTER", "OUTPUT_DIRECTORY", "SPECIAL_FILE_KEYWORDS", "SPECIAL_FILE_EXACT"):
        monkeypatch.delenv(name, raising=False)
    source = tmp_path / filename
    original = _xhtml("Selected chapter")
    source.write_bytes(original)
    output = tmp_path / (filename + ".epub")
    extracted = tmp_path / "extracted"
    extracted.mkdir()

    convert_html_file_to_epub(source, output)

    alias = "chapter0001" + source.suffix
    with zipfile.ZipFile(output) as archive:
        assert _spine(archive) == [alias]
        assert archive.read(alias) == original
        chapters = Chapter_Extractor.extract_chapters(
            archive, str(extracted), progress_callback=lambda _message: None,
        )
    selected = [chapter for chapter in chapters if chapter["filename"].endswith(alias)]
    assert len(selected) == 1
    assert selected[0]["num"] == 1
    assert selected[0]["detection_method"] != "configured_special_file"
    assert "Original 한국어 text." in selected[0]["body"]
    assert source.read_bytes() == original


def test_standalone_special_filename_updates_only_self_links(tmp_path):
    source = tmp_path / "index.xhtml"
    original = b'''<html xmlns="http://www.w3.org/1999/xhtml"><head><title>One</title></head><body><p id="here">Source</p><a href="./index.xhtml#here">self</a><a href="sibling.xhtml#here">other</a><a href="#here">fragment</a></body></html>'''
    source.write_bytes(original)
    output = tmp_path / "index.epub"

    convert_html_file_to_epub(source, output)

    with zipfile.ZipFile(output) as archive:
        chapter = archive.read("chapter0001.xhtml")
        assert chapter == original.replace(b"./index.xhtml#here", b"./chapter0001.xhtml#here")
    assert source.read_bytes() == original
