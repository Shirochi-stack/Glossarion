"""Special-file rules must affect automatic pairing, never chapter visibility."""

import os
import time
from types import SimpleNamespace

import pytest
from ebooklib import epub
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QStyleOptionViewItem

import extract_glossary_from_epub as extractor
from parallel_epub_glossary import (
    ParallelEpubPairDialog,
    auto_map_epub_chapters,
    chapter_filename,
    chapter_text,
    compact_parallel_epub_selection,
    restore_parallel_epub_pairs,
)


SPECIAL_KEYWORDS = ("message", "title", "author", "notice")
TRANSLATED_NAMES = [
    "0055_Chapter_55_Everything_Under_Control.xhtml",
    "0056_Chapter_56_Disregarded_Honor_Message_from_Ancient_Sky_Domain.xhtml",
    "0057_Chapter_57_Yan_Jis_Perturbation.xhtml",
    "0802_Chapter_802_Title.xhtml",
    "0815_Chapter_815_Authority.xhtml",
    "1050_Chapter_1050_Noticed.xhtml",
    "1051_Chapter_1051_Arrival.xhtml",
]
RAW_NAMES = [f"{number:04d}_{number:03d}_.xhtml" for number in (55, 56, 57, 802, 815, 1050, 1051)]
ENV_KEYS = ("TRANSLATE_SPECIAL_FILES", "SPECIAL_FILE_KEYWORDS", "SPECIAL_FILE_EXACT")


def _is_special(filename):
    return any(keyword in filename.casefold() for keyword in SPECIAL_KEYWORDS)


def _chapters(names):
    return [{"filename": name, "text": f"Readable chapter prose for {name}."} for name in names]


def _write_epub(path, names, *, textless_names=()):
    book = epub.EpubBook()
    book.set_identifier(path.stem)
    book.set_title("Parallel visibility regression")
    book.set_language("en")
    documents = []
    for index, name in enumerate(names):
        document = epub.EpubHtml(title=f"Chapter {index}", file_name=f"Text/{name}")
        document.content = (
            f"<html><body><h1>Chapter {index}</h1>"
            f"<p>Readable narrative text for document {index}, with enough prose to extract.</p>"
            "</body></html>"
        )
        if name in textless_names:
            document.content = '<html><body><img src="illustration.jpg"/></body></html>'
        book.add_item(document)
        documents.append(document)
    book.spine = documents
    book.toc = tuple(documents)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(path), book)
    return path


@pytest.fixture
def filtered_settings(monkeypatch):
    monkeypatch.setenv("TRANSLATE_SPECIAL_FILES", "0")
    monkeypatch.setenv("SPECIAL_FILE_KEYWORDS", ",".join(SPECIAL_KEYWORDS))
    monkeypatch.setenv("SPECIAL_FILE_EXACT", "index")
    monkeypatch.setattr(extractor, "is_stop_requested", lambda: False)


@pytest.mark.parametrize("loader", ["default_dialog", "main_window"])
def test_pair_loaders_keep_spine_documents_without_changing_filter_settings(
    tmp_path, monkeypatch, filtered_settings, loader,
):
    path = _write_epub(tmp_path / "translated.epub", TRANSLATED_NAMES)
    expected_environment = {key: os.environ.get(key) for key in ENV_KEYS}
    original_extract = extractor.extract_chapters_from_epub
    observed_environments = []

    def observe_extraction(*args, **kwargs):
        observed_environments.append({key: os.environ.get(key) for key in ENV_KEYS})
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(extractor, "extract_chapters_from_epub", observe_extraction)
    if loader == "main_window":
        from translator_gui import TranslatorGUI

        gui = SimpleNamespace(special_file_keywords_var="different", special_file_exact_var="also_different")
        loaded = TranslatorGUI._load_parallel_epub_chapters(gui, str(path))
    else:
        loaded = ParallelEpubPairDialog._default_chapter_loader(str(path))

    assert [chapter_filename(chapter) for chapter in loaded] == TRANSLATED_NAMES
    assert all(chapter_text(chapter).strip() for chapter in loaded)
    assert observed_environments == [expected_environment]
    assert {key: os.environ.get(key) for key in ENV_KEYS} == expected_environment
    ordinary = original_extract(str(path), return_metadata=True)
    assert [name for _text, name in ordinary] == [name for name in TRANSLATED_NAMES if not _is_special(name)]


def test_filtered_cache_cannot_hide_documents_requested_for_pairing(tmp_path, filtered_settings):
    path = _write_epub(tmp_path / "translated.epub", TRANSLATED_NAMES)
    cache_path = str(tmp_path / "source_cache.json")
    filtered = extractor.extract_chapters_from_epub(str(path), return_metadata=True, cache_path=cache_path)
    complete = extractor.extract_chapters_from_epub(
        str(path), return_metadata=True, cache_path=cache_path, include_special_files=True,
    )
    filtered_again = extractor.extract_chapters_from_epub(str(path), return_metadata=True, cache_path=cache_path)

    assert [name for _text, name in complete] == TRANSLATED_NAMES
    assert filtered_again == filtered
    assert len(filtered) == 3


@pytest.mark.parametrize("enable_auto_offset", [False, True])
def test_special_chapter_slots_stay_unmapped_without_shifting_later_chapters(enable_auto_offset):
    mappings = auto_map_epub_chapters(
        _chapters(RAW_NAMES), _chapters(TRANSLATED_NAMES),
        enable_auto_offset=enable_auto_offset, special_file_predicate=_is_special,
    )

    assert [entry["raw_index"] for entry in mappings] == list(range(7))
    assert [entry["translated_index"] for entry in mappings] == [0, None, 2, None, None, None, 6]
    assert all(mappings[index]["strategy"] == "Special file — Unmapped" for index in (1, 3, 4, 5))


@pytest.mark.parametrize("enable_auto_offset", [False, True])
@pytest.mark.parametrize("special_side", ["raw", "translated"])
def test_exact_special_rule_unmaps_only_its_pair_with_asymmetric_frontmatter(enable_auto_offset, special_side):
    raw = _chapters(["0000_Information.xhtml", "chapter55.xhtml", "chapter56.xhtml", "chapter57.xhtml"])
    translated = _chapters(["chapter55.xhtml", "chapter56.xhtml", "chapter57.xhtml"])
    special_name = f"{special_side}_56.xhtml"
    (raw[2] if special_side == "raw" else translated[1])["filename"] = special_name

    mappings = auto_map_epub_chapters(
        raw, translated, enable_auto_offset=enable_auto_offset,
        special_file_predicate=lambda name: name == special_name,
    )

    assert [entry["translated_index"] for entry in mappings] == [None, 0, None, 2]
    assert mappings[2]["strategy"] == "Special file — Unmapped"


@pytest.fixture
def qt_app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


def _wait_for_mapping(app, dialog):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        app.processEvents()
        if dialog.raw_chapters and dialog.translated_chapters and not (
            dialog._active_load or dialog._pending_loads or dialog._mapping_building
        ):
            return
        time.sleep(0.005)
    pytest.fail("Timed out while loading and mapping the small fixture EPUBs")


def test_dialog_keeps_special_document_available_for_manual_mapping_and_saved_restore(
    tmp_path, monkeypatch, qt_app, filtered_settings,
):
    raw_path = _write_epub(tmp_path / "raw.epub", RAW_NAMES[:3])
    translated_path = _write_epub(tmp_path / "translated.epub", TRANSLATED_NAMES[:3])
    dialog = ParallelEpubPairDialog(special_file_predicate=_is_special)
    restored_dialog = None
    editor = None
    try:
        dialog._load_epub("raw", str(raw_path))
        dialog._load_epub("translated", str(translated_path))
        _wait_for_mapping(qt_app, dialog)

        assert len(dialog.raw_chapters) == len(dialog.translated_chapters) == 3
        assert dialog.mapping_table.rowCount() == 3
        assert dialog.mapping_table.item(1, 1).data(Qt.UserRole) == -1
        assert dialog.mapping_table.item(1, 2).text() == "Special file — Unmapped"
        assert dialog.mapping_table.item(2, 1).text() == TRANSLATED_NAMES[2]

        dialog._apply_mapping_offset(1)
        assert dialog.mapping_table.item(2, 1).data(Qt.UserRole) == -1
        assert dialog.mapping_table.item(2, 2).text() == "Special file — Unmapped"
        dialog._apply_mapping_offset(-1)
        assert dialog.mapping_table.item(2, 1).data(Qt.UserRole) == 2

        model = dialog.mapping_table.model()
        index = model.index(1, 1)
        delegate = dialog.mapping_table.itemDelegateForColumn(1)
        monkeypatch.setattr(delegate, "_show_popup", lambda _editor: None)
        editor = delegate.createEditor(dialog.mapping_table, QStyleOptionViewItem(), index)
        delegate.setEditorData(editor, index)
        special_choice = editor.findText(TRANSLATED_NAMES[1])
        assert special_choice >= 0
        editor.setCurrentIndex(special_choice)
        delegate.setModelData(editor, model, index)
        assert dialog.mapping_table.item(1, 1).data(Qt.UserRole) == 1
        assert dialog.mapping_table.item(1, 2).text() == "Manual"

        selected_pairs, skipped = restore_parallel_epub_pairs(
            dialog.raw_chapters, dialog.translated_chapters,
            [
                {
                    **entry,
                    "raw_filename": dialog.raw_chapters[entry["raw_index"]]["filename"],
                    "translated_filename": dialog.translated_chapters[entry["translated_index"]]["filename"],
                }
                for entry in dialog._selected_mapping()
            ],
        )
        assert skipped == 0
        saved = compact_parallel_epub_selection({
            "raw_path": str(raw_path), "translated_path": str(translated_path), "pairs": selected_pairs,
        })
        restored_dialog = ParallelEpubPairDialog(special_file_predicate=_is_special)
        assert restored_dialog.restore_persisted_selection(saved)
        _wait_for_mapping(qt_app, restored_dialog)
        assert restored_dialog._selected_mapping() == [
            {"raw_index": index, "translated_index": index} for index in range(3)
        ]
        assert restored_dialog.mapping_table.item(1, 1).text() == TRANSLATED_NAMES[1]
        assert restored_dialog.mapping_table.item(1, 2).text() == "Saved Mapping"
    finally:
        if editor is not None:
            editor.close()
        if restored_dialog is not None:
            restored_dialog.close()
        dialog.close()
        qt_app.processEvents()


@pytest.mark.parametrize("textless_index", [0, 2], ids=["first-anchor", "last-anchor"])
def test_interior_pairing_uses_textless_boundary_documents(
    tmp_path, qt_app, filtered_settings, textless_index,
):
    raw_names = RAW_NAMES[:3]
    translated_names = TRANSLATED_NAMES[:3]
    raw_path = _write_epub(
        tmp_path / "raw.epub", raw_names,
        textless_names=[raw_names[textless_index]],
    )
    translated_path = _write_epub(
        tmp_path / "translated.epub", translated_names,
        textless_names=[translated_names[textless_index]],
    )
    config = {"never_consider_in_between_files_as_special": True}
    dialog = ParallelEpubPairDialog(config=config, special_file_predicate=_is_special)
    try:
        dialog._load_epub("raw", str(raw_path))
        dialog._load_epub("translated", str(translated_path))
        _wait_for_mapping(qt_app, dialog)

        assert len(dialog.raw_chapters) == len(dialog.translated_chapters) == 2
        assert dialog.translated_reading_order == translated_names
        special_row = next(
            row for row, chapter in enumerate(dialog.raw_chapters)
            if chapter["filename"] == raw_names[1]
        )
        assert dialog.mapping_table.item(special_row, 1).text() == translated_names[1]

        config["never_consider_in_between_files_as_special"] = False
        dialog._rebuild_mapping()
        _wait_for_mapping(qt_app, dialog)
        assert dialog.mapping_table.item(special_row, 1).data(Qt.UserRole) == -1

        config["never_consider_in_between_files_as_special"] = True
        dialog._rebuild_mapping()
        _wait_for_mapping(qt_app, dialog)
        assert dialog.mapping_table.item(special_row, 1).text() == translated_names[1]
    finally:
        dialog.close()
        qt_app.processEvents()
