"""Only edge runs of configured special files may be skipped when enabled."""

from unittest.mock import Mock

import pytest
from ebooklib import epub

import extract_glossary_from_epub as extractor
from parallel_epub_glossary import auto_map_epub_chapters


SETTING = "GLOSSARY_NEVER_CONSIDER_IN_BETWEEN_FILES_AS_SPECIAL"
SPINE_NAMES = [
    "title0900.xhtml",
    "notice0901.xhtml",
    "zulu_chapter0001.xhtml",
    "0056_Chapter_56_Disregarded_Honor_Message.xhtml",
    "index.xhtml",
    "author_scene.xhtml",
    "alpha_chapter0200.xhtml",
    "notice0001.xhtml",
    "title0002.xhtml",
]
ORDINARY_NAMES = [SPINE_NAMES[2], SPINE_NAMES[6]]
INTERIOR_NAMES = SPINE_NAMES[2:7]


def _write_epub(path, spine_names):
    book = epub.EpubBook()
    book.set_identifier(path.stem)
    book.set_title("Interior special filename regression")
    book.set_language("en")
    documents = {}
    # Deliberately disagree with both filename order and reading order.
    for index, name in enumerate(reversed(spine_names)):
        document = epub.EpubHtml(uid=f"page-{index}", file_name=f"Text/{name}")
        document.content = (
            "<html><body><p>Readable story prose belonging to "
            f"{name}, with enough content for glossary extraction.</p></body></html>"
        )
        book.add_item(document)
        documents[name] = document
    book.spine = [documents[name] for name in spine_names]
    book.add_item(epub.EpubNcx())
    epub.write_epub(str(path), book)
    return path


@pytest.fixture(autouse=True)
def special_file_settings(monkeypatch):
    monkeypatch.setattr(extractor, "is_stop_requested", lambda: False)
    monkeypatch.setenv("TRANSLATE_SPECIAL_FILES", "0")
    monkeypatch.setenv("SPECIAL_FILE_KEYWORDS", "Title, Notice, Message, Author")
    monkeypatch.setenv("SPECIAL_FILE_EXACT", "INDEX")
    monkeypatch.delenv(SETTING, raising=False)


def _extract_names(path, **kwargs):
    return [
        name
        for _text, name in extractor.extract_chapters_from_epub(
            str(path), return_metadata=True, **kwargs,
        )
    ]


@pytest.mark.parametrize("setting", [None, "0", "1"], ids=["default", "disabled", "enabled"])
@pytest.mark.parametrize("use_spine_numbering", ["0", "1"])
def test_special_boundaries_follow_opf_spine_regardless_of_filename_numbering(
    tmp_path, monkeypatch, setting, use_spine_numbering,
):
    source = _write_epub(tmp_path / "book.epub", SPINE_NAMES)
    monkeypatch.setenv("USE_SPINE_ORDER", use_spine_numbering)
    if setting is not None:
        monkeypatch.setenv(SETTING, setting)

    assert _extract_names(source) == (INTERIOR_NAMES if setting == "1" else ORDINARY_NAMES)


@pytest.mark.parametrize("override", ["environment", "argument"])
def test_include_special_files_still_keeps_leading_and_trailing_files(
    tmp_path, monkeypatch, override,
):
    source = _write_epub(tmp_path / "book.epub", SPINE_NAMES)
    monkeypatch.setenv(SETTING, "1")
    kwargs = {}
    if override == "environment":
        monkeypatch.setenv("TRANSLATE_SPECIAL_FILES", "1")
    else:
        kwargs["include_special_files"] = True

    assert _extract_names(source, **kwargs) == SPINE_NAMES


@pytest.mark.parametrize(("spine_names", "expected_names"), [
    (["notice0001.xhtml", "index.xhtml", "title.xhtml"], []),
    (["notice0001.xhtml", "chapter0056.xhtml", "title.xhtml"], ["chapter0056.xhtml"]),
])
def test_zero_or_one_ordinary_anchor_does_not_protect_special_files(
    tmp_path, monkeypatch, spine_names, expected_names,
):
    source = _write_epub(tmp_path / "book.epub", spine_names)
    monkeypatch.setenv(SETTING, "1")

    assert _extract_names(source) == expected_names


def test_custom_keyword_and_exact_rules_define_both_boundaries(tmp_path, monkeypatch):
    names = [
        "bespoke_front0009.xhtml",
        "my_extra.xhtml",
        "start.xhtml",
        "bespoke_scene.xhtml",
        "MY_EXTRA.xhtml",
        "end.xhtml",
        "bespoke_back0001.xhtml",
    ]
    source = _write_epub(tmp_path / "book.epub", names)
    monkeypatch.setenv("SPECIAL_FILE_KEYWORDS", "BESPOKE")
    monkeypatch.setenv("SPECIAL_FILE_EXACT", "my_extra")
    monkeypatch.setenv(SETTING, "1")

    assert _extract_names(source) == names[2:6]


def test_changing_interior_protection_invalidates_cache_and_same_setting_reuses_it(
    tmp_path, monkeypatch,
):
    source = _write_epub(tmp_path / "book.epub", SPINE_NAMES)
    cache_path = str(tmp_path / "source_cache.json")
    read_archive = Mock(wraps=extractor.epub.read_epub)
    monkeypatch.setattr(extractor.epub, "read_epub", read_archive)

    assert _extract_names(source, cache_path=cache_path) == ORDINARY_NAMES
    assert _extract_names(source, cache_path=cache_path) == ORDINARY_NAMES
    assert read_archive.call_count == 1

    monkeypatch.setenv(SETTING, "1")
    assert _extract_names(source, cache_path=cache_path) == INTERIOR_NAMES
    assert _extract_names(source, cache_path=cache_path) == INTERIOR_NAMES
    assert read_archive.call_count == 2

    monkeypatch.setenv(SETTING, "0")
    assert _extract_names(source, cache_path=cache_path) == ORDINARY_NAMES
    assert _extract_names(source, cache_path=cache_path) == ORDINARY_NAMES
    assert read_archive.call_count == 3


@pytest.mark.parametrize("enable_auto_offset", [False, True])
@pytest.mark.parametrize("special_side", ["raw", "translated"])
def test_parallel_mapping_protects_interior_keyword_chapter_on_either_side(
    enable_auto_offset, special_side,
):
    sides = {
        side: [{"filename": f"{side}{number:04d}.xhtml", "text": "Readable story."}
               for number in range(1, 6)]
        for side in ("raw", "translated")
    }
    sides[special_side][0]["filename"] = f"{special_side}_notice0001.xhtml"
    sides[special_side][2]["filename"] = f"{special_side}_message0003.xhtml"
    sides[special_side][4]["filename"] = f"{special_side}_notice0005.xhtml"
    kwargs = {
        "enable_auto_offset": enable_auto_offset,
        "special_file_predicate": lambda name: "notice" in name or "message" in name,
    }

    default = auto_map_epub_chapters(sides["raw"], sides["translated"], **kwargs)
    protected = auto_map_epub_chapters(
        sides["raw"], sides["translated"], protect_interior_special_files=True, **kwargs,
    )

    assert [entry["translated_index"] for entry in default] == [None, 1, None, 3, None]
    assert [entry["translated_index"] for entry in protected] == [None, 1, 2, 3, None]
    assert protected[0]["strategy"] == "Special file — Unmapped"
    assert protected[4]["strategy"] == "Special file — Unmapped"
