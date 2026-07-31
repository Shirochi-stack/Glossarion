import os

from PySide6.QtCore import QMimeData, QUrl

from manga_integration import MangaTranslationTab


class _FakeListWidget:
    def __init__(self):
        self.current_row = -1

    def currentRow(self):
        return self.current_row

    def setCurrentRow(self, row):
        self.current_row = row


class _DropHarness:
    def __init__(self):
        self.selected_files = []
        self.manga_selected_folder_roots = []
        self.file_listbox = _FakeListWidget()
        self.list_items = []
        self.logs = []

    def _can_add_manga_paths_from_single_source(self, _paths):
        return True

    def _add_manga_file_item(self, path):
        self.list_items.append(path)

    def _add_cbz_archive_images(self, _path, _image_extensions):
        return 0

    def _update_manga_image_range_display(self):
        pass

    def _persist_selected_files(self):
        pass

    def _log(self, message, level):
        self.logs.append((message, level))


def test_drop_payload_keeps_supported_local_paths_only(tmp_path):
    image = tmp_path / "page.png"
    image.write_bytes(b"image")
    ignored = tmp_path / "notes.txt"
    ignored.write_text("notes", encoding="utf-8")
    folder = tmp_path / "chapter"
    folder.mkdir()

    mime = QMimeData()
    mime.setUrls([
        QUrl.fromLocalFile(str(image)),
        QUrl.fromLocalFile(str(image)),
        QUrl.fromLocalFile(str(ignored)),
        QUrl.fromLocalFile(str(folder)),
        QUrl("https://example.com/page.png"),
    ])

    paths = MangaTranslationTab._manga_drop_local_paths(object(), mime)
    assert paths == [os.path.abspath(image), os.path.abspath(folder)]


def test_folder_drop_recurses_and_skips_generated_output_folders(tmp_path):
    chapter = tmp_path / "chapter"
    nested = chapter / "nested"
    translated = chapter / "001_translated"
    ocr_dir = chapter / "OCR Text"
    nested.mkdir(parents=True)
    translated.mkdir()
    ocr_dir.mkdir()
    page_10 = chapter / "10.png"
    page_2 = chapter / "2.png"
    nested_page = nested / "3.webp"
    generated_page = translated / "1.png"
    ocr_artifact = ocr_dir / "preview.jpg"
    for path in (page_10, page_2, nested_page, generated_page, ocr_artifact):
        path.write_bytes(b"image")

    harness = _DropHarness()
    MangaTranslationTab._add_dropped_manga_paths(harness, [str(chapter)])

    assert harness.selected_files == [
        os.path.abspath(page_2),
        os.path.abspath(page_10),
        os.path.abspath(nested_page),
    ]
    assert harness.list_items == harness.selected_files
    assert harness.file_listbox.current_row == 0
    assert harness.manga_selected_folder_roots == [os.path.abspath(chapter)]
