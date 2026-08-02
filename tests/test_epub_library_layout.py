import os
import time
import zipfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

import epub_library
from PySide6.QtCore import QEventLoop, QRect
from PySide6.QtWidgets import QApplication, QWidget

from epub_library import (
    BookDetailsDialog,
    EpubLibraryDialog,
    EpubReaderDialog,
    SIZE_NORMAL,
    _FlowLayout,
    _configure_epub_reader_web_settings,
    _merge_manual_metadata_edits,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _pump_events(app: QApplication, timeout: float = 0.8) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents(QEventLoop.AllEvents, 50)
        time.sleep(0.005)


def _books(prefix: str, count: int = 8) -> list[dict]:
    return [
        {
            "path": f"C:/layout-test/{prefix}-{idx}.txt",
            "name": f"Book {idx}",
            "size": 1024,
            "mtime": float(idx),
            "type": "txt",
        }
        for idx in range(count)
    ]


def _make_dialog(
    qapp: QApplication, width: int, height: int = 700
) -> EpubLibraryDialog:
    dialog = EpubLibraryDialog(
        config={
            "epub_library_card_size": SIZE_NORMAL,
            "epub_library_tab": 0,
        }
    )
    # Keep the test focused on layout: showEvent normally starts a real scan.
    dialog._initial_scan_started = True
    dialog._auto_refresh = lambda: None
    dialog.resize(width, height)
    dialog.show()
    dialog._auto_refresh_timer.stop()
    dialog._hide_loading()
    qapp.processEvents()
    return dialog


def _layout_rows(layout) -> list[int]:
    return [layout.getItemPosition(i)[0] for i in range(layout.count())]


def test_resize_before_first_streamed_card_reflows_without_zoom(qapp):
    """A resize in the stream's zero-card window must not freeze old columns."""
    dialog = _make_dialog(qapp, 700)
    try:
        books = _books("active")
        dialog._in_progress_books = books
        dialog._cover_path_cache.update(
            {book["path"]: "_none_" for book in books}
        )

        dialog._refresh_view()
        assert dialog._is_card_stream_active("ip")
        assert dialog._ip_cards == []

        # Reproduce the startup race: the stream captured the narrow width,
        # then the window widened before its first 16 ms card batch ran.
        dialog.resize(1800, 1000)
        _pump_events(qapp)

        assert len(dialog._ip_cards) == 8
        assert set(_layout_rows(dialog._ip_grid_layout)) == {0}
    finally:
        dialog._auto_refresh_timer.stop()
        dialog.close()
        qapp.processEvents()


def test_hidden_tab_uses_settled_visible_width(qapp):
    """Inactive-tab preloading must not trust Qt's default ~640 px width."""
    dialog = _make_dialog(qapp, 1200)
    try:
        books = _books("hidden")
        dialog._completed_books = books
        dialog._cover_path_cache.update(
            {book["path"]: "_none_" for book in books}
        )

        assert not dialog._comp_scroll.isVisible()
        dialog._populate_tab("comp")
        _pump_events(qapp)

        assert len(dialog._comp_cards) == 8
        assert set(_layout_rows(dialog._comp_grid_layout)) == {0}
    finally:
        dialog._auto_refresh_timer.stop()
        dialog.close()
        qapp.processEvents()


def test_scrollbar_and_window_resize_do_not_repeat_full_reflow(qapp):
    """Scrollbar toggles and one resize should produce at most one new stream."""
    dialog = _make_dialog(qapp, 1200, 450)
    try:
        books = _books("many", count=20)
        dialog._in_progress_books = books
        dialog._cover_path_cache.update(
            {book["path"]: "_none_" for book in books}
        )

        dialog._refresh_view()
        _pump_events(qapp)

        assert dialog._ip_scroll.verticalScrollBar().maximum() > 0
        assert dialog._card_stream_generation["ip"] == 1
        assert not dialog._is_card_stream_active("ip")
        assert "ip" not in dialog._pending_card_reflow

        dialog.resize(1400, 450)
        _pump_events(qapp)

        assert dialog._card_stream_generation["ip"] == 2
        assert not dialog._is_card_stream_active("ip")
        assert "ip" not in dialog._pending_card_reflow
    finally:
        dialog._auto_refresh_timer.stop()
        dialog.close()
        qapp.processEvents()


def test_details_tags_keep_full_text_and_wrap_to_multiple_rows(qapp):
    container = QWidget()
    layout = _FlowLayout(container, horizontal_spacing=8, vertical_spacing=8)

    class DetailsStub:
        _tags_row = container

    tags = [
        "Modern",
        "Apocalypse",
        "TS",
        "Genius",
        "Mage",
        "System",
        "Growth",
        "Gallery",
        "Community",
        "No Romance",
        "Regression",
        "Tower Master",
        "Survival",
        "Magic",
        "Post-apocalyptic",
    ]
    BookDetailsDialog._fill_chip_row(DetailsStub(), layout, tags)

    width = 380
    height = layout.heightForWidth(width)
    container.resize(width, height)
    layout.setGeometry(QRect(0, 0, width, height))
    qapp.processEvents()

    chips = [layout.itemAt(i).widget() for i in range(layout.count())]
    assert len(chips) == len(tags)
    assert [chip.text() for chip in chips] == tags
    assert len({chip.y() for chip in chips}) > 1
    assert height > max(chip.height() for chip in chips)
    assert all(chip.width() >= chip.sizeHint().width() for chip in chips)


def test_reader_enables_remote_images_for_local_chapter_pages():
    if not epub_library._HAS_WEBENGINE:
        pytest.skip("Qt WebEngine is unavailable")

    class FakeSettings:
        def __init__(self):
            self.attributes = {}

        def setAttribute(self, attribute, enabled):
            self.attributes[attribute] = enabled

    class FakeView:
        def __init__(self):
            self.web_settings = FakeSettings()

        def settings(self):
            return self.web_settings

    view = FakeView()

    assert _configure_epub_reader_web_settings(view) is True
    assert view.web_settings.attributes[
        epub_library.QWebEngineSettings.AutoLoadImages
    ] is True
    assert view.web_settings.attributes[
        epub_library.QWebEngineSettings.LocalContentCanAccessRemoteUrls
    ] is True


def test_remote_image_url_survives_local_image_processing(tmp_path):
    remote_url = (
        "https://images.novelpia.com/imagebox/b1/"
        "b1b11a46e497175bfdc6278959170d99_1958056_1779373634_ori.file"
    )
    reader = EpubReaderDialog.__new__(EpubReaderDialog)
    reader._epub_path = str(tmp_path / "book.epub")
    reader._images = {}
    reader._extra_image_dirs = []

    processed = reader._process_html(
        f'<p><img class="remote-image" alt="Image 1" src="{remote_url}"/></p>'
    )

    assert remote_url in processed
    assert 'class="remote-image"' in processed


def test_remote_cover_page_is_downloaded_and_cached(tmp_path, monkeypatch):
    remote_url = (
        "https://images.novelpia.com/imagebox/cover/"
        "7699c81bc9ee1228b9fb46cf4c3af980_358677_ori.file"
    )
    remote_bytes = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c6360f8cff00000040101005fe5c34b0000000049454e44"
        "ae426082"
    )
    epub_path = tmp_path / "remote-cover.epub"
    with zipfile.ZipFile(epub_path, "w") as epub:
        epub.writestr(
            "OEBPS/Text/chapter0001.xhtml",
            '<html><body><img src="https://example.com/chapter.file"/></body></html>',
        )
        epub.writestr(
            "OEBPS/Text/cover.html",
            f'<html><body><img class="remote-image" src="{remote_url}"/></body></html>',
        )

    seen_urls = []
    monkeypatch.setattr(epub_library, "_cover_cache_dir", lambda: str(tmp_path / "cache"))
    monkeypatch.setattr(
        epub_library,
        "_download_remote_cover_image",
        lambda url: seen_urls.append(url) or remote_bytes,
    )
    (tmp_path / "cache").mkdir()

    cover_path = epub_library._extract_cover(str(epub_path))

    assert seen_urls == [remote_url]
    assert cover_path
    assert open(cover_path, "rb").read() == remote_bytes


def test_other_settings_exposes_remote_image_download_toggle():
    source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    settings_source = open(
        os.path.join(source_root, "src", "other_settings.py"),
        encoding="utf-8",
    ).read()
    gui_source = open(
        os.path.join(source_root, "src", "translator_gui.py"),
        encoding="utf-8",
    ).read()

    assert '"Download remote image URLs"' in settings_source
    assert "self.config['download_remote_image_urls'] = enabled" in settings_source
    assert "os.environ['DOWNLOAD_REMOTE_IMAGE_URLS']" in settings_source
    assert 'QLabel("Threads:")' in settings_source
    assert 'QLabel("Interval:")' in settings_source
    assert "remote_image_workers_spin = QDoubleSpinBox()" in settings_source
    assert "remote_image_workers_spin.setDecimals(0)" in settings_source
    assert "remote_image_spin_style" not in settings_source
    assert "remote_image_workers_spin.wheelEvent" in settings_source
    assert "remote_image_interval_spin.wheelEvent" in settings_source
    assert "REMOTE_IMAGE_DOWNLOAD_WORKERS" in settings_source
    assert "REMOTE_IMAGE_DOWNLOAD_INTERVAL" in settings_source
    assert "self.download_remote_image_urls_var = self.config.get(" in gui_source
    assert "('download_remote_image_urls', ['download_remote_image_urls_var']" in gui_source
    assert "('remote_image_download_workers', ['remote_image_download_workers_var']" in gui_source
    assert "('remote_image_download_interval', ['remote_image_download_interval_var']" in gui_source


def test_details_tags_prefer_translated_metadata_subjects():
    class DetailsStub:
        _details = {"subjects": ["판타지", "빙의"]}
        _metadata_json = {
            "subject": ["Fantasy", "Possession"],
        }

        def _collect_tag_values(self, *sources):
            return BookDetailsDialog._collect_tag_values(self, *sources)

    assert BookDetailsDialog._display_tag_values(DetailsStub()) == [
        "Fantasy",
        "Possession",
    ]


def test_details_tags_fall_back_to_source_epub_subjects():
    class DetailsStub:
        _details = {"subjects": ["판타지", "빙의"]}
        _metadata_json = {}

        def _collect_tag_values(self, *sources):
            return BookDetailsDialog._collect_tag_values(self, *sources)

    assert BookDetailsDialog._display_tag_values(DetailsStub()) == [
        "판타지",
        "빙의",
    ]


def test_details_tags_allow_manually_cleared_metadata_subjects():
    class DetailsStub:
        _details = {"subjects": ["판타지", "빙의"]}
        _metadata_json = {"subject": []}

        def _collect_tag_values(self, *sources):
            return BookDetailsDialog._collect_tag_values(self, *sources)

    assert BookDetailsDialog._display_tag_values(DetailsStub()) == []


def test_details_author_prefers_metadata_creator():
    class DetailsStub:
        _details = {"authors": ["원작자"]}
        _metadata_json = {"creator": "Edited Author"}

    assert BookDetailsDialog._metadata_author_values(DetailsStub()) == [
        "Edited Author",
    ]


def test_manual_metadata_edits_preserve_originals_and_unknown_fields():
    existing = {
        "title": "Translated Title",
        "original_title": "원제",
        "identifier": "book-123",
        "subject": ["Fantasy", "Possession"],
    }
    updated, changed = _merge_manual_metadata_edits(
        existing,
        {
            "title": "My Preferred Title",
            "subject": "Fantasy\nDrama; Fantasy",
        },
        {
            "title": "원제",
            "subject": ["판타지", "빙의"],
        },
    )

    assert changed == {"title", "subject"}
    assert updated["title"] == "My Preferred Title"
    assert updated["original_title"] == "원제"
    assert updated["title_translated"] is True
    assert updated["subject"] == ["Fantasy", "Drama"]
    assert updated["original_subject"] == ["판타지", "빙의"]
    assert updated["subject_translated"] is True
    assert updated["identifier"] == "book-123"
