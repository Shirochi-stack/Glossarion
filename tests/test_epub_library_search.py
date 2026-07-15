import zipfile

import pytest

pytest.importorskip("PySide6")

from epub_library import (
    EpubLibraryDialog,
    FORMAT_ALL,
    _EPUB_SEARCH_METADATA_CACHE,
    _book_matches_library_query,
    _extract_epub_subjects,
)


def _book():
    return {
        "name": "The Shut-in Tower Master",
        "subjects": ["Modern", "Apocalypse", "No Romance"],
        "raw_subjects": ["현대", "아포칼립스", "노맨스"],
        "metadata_json": {
            "subject": ["Mage", "System"],
            "original_subject": ["마법사", "시스템"],
            "tags": "#Community #Gallery",
            "genres": ["Fantasy"],
        },
    }


@pytest.mark.parametrize(
    "query",
    [
        "tower master",
        "apocalypse",
        "romance",
        "아포칼립스",
        "마법사",
        "community",
        "fantasy",
    ],
)
def test_library_query_matches_titles_and_tag_sources(query):
    assert _book_matches_library_query(_book(), query)


def test_library_query_rejects_unrelated_text():
    assert not _book_matches_library_query(_book(), "historical cooking")


def test_dialog_filter_uses_tag_matching():
    class Search:
        @staticmethod
        def text():
            return "Gallery"

    class FakeDialog:
        _search = Search()
        _format_filter = FORMAT_ALL

        @staticmethod
        def _sorted_books(books):
            return books

        @staticmethod
        def _format_of_book(book):
            return book.get("type", "epub")

    matching = _book()
    unrelated = {"name": "Another Book", "subjects": ["Drama"]}

    result = EpubLibraryDialog._filtered(FakeDialog(), [unrelated, matching])

    assert result == [matching]


def test_extract_epub_subjects_preserves_all_opf_tags(tmp_path):
    epub_path = tmp_path / "tagged.epub"
    opf = """<?xml version="1.0" encoding="UTF-8"?>
    <package xmlns="http://www.idpf.org/2007/opf"
             xmlns:dc="http://purl.org/dc/elements/1.1/" version="2.0">
      <metadata>
        <dc:title>Tagged Book</dc:title>
        <dc:subject>Modern</dc:subject>
        <dc:subject>Apocalypse</dc:subject>
        <dc:subject>TS</dc:subject>
        <dc:subject>Apocalypse</dc:subject>
      </metadata>
    </package>"""
    with zipfile.ZipFile(epub_path, "w") as epub:
        epub.writestr("OEBPS/content.opf", opf.encode("utf-8"))

    _EPUB_SEARCH_METADATA_CACHE.clear()

    assert _extract_epub_subjects(str(epub_path)) == (
        "Modern",
        "Apocalypse",
        "TS",
    )
