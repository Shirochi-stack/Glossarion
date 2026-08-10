import json
import os
import time
import zipfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

import epub_library
from PySide6.QtCore import QEventLoop, QPoint, QRect, Qt
from PySide6.QtWidgets import QApplication, QPushButton, QWidget

from epub_library import (
    BookDetailsDialog,
    EpubLibraryDialog,
    EpubReaderDialog,
    SIZE_NORMAL,
    _FlowLayout,
    _BookDetailsLoader,
    _OverlayMergeThread,
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


def test_library_angle_wheel_scroll_uses_accumulated_animation(qapp):
    dialog = _make_dialog(qapp, 700, 500)
    try:
        area = dialog._ip_scroll
        area.widget().setMinimumHeight(2400)
        qapp.processEvents()
        bar = area.verticalScrollBar()
        assert bar.maximum() > 300

        assert dialog._animate_library_scroll("ip", area, -120)
        first_target = dialog._library_scroll_targets["ip"]
        assert first_target == dialog._WHEEL_SCROLL_PIXELS
        assert (dialog._library_scroll_animations["ip"].state()
                == epub_library.QAbstractAnimation.Running)
        _pump_events(qapp, 0.05)
        assert 0 < bar.value() < first_target

        # A second tick while the first animation is running adds to its end
        # target instead of discarding the remaining movement.
        assert dialog._animate_library_scroll("ip", area, -120)
        final_target = dialog._library_scroll_targets["ip"]
        assert final_target == first_target + dialog._WHEEL_SCROLL_PIXELS
        _pump_events(qapp, 0.25)
        assert bar.value() == final_target
    finally:
        dialog.close()
        qapp.processEvents()


def test_reader_accepts_translated_pdf_html_workspace(qapp, tmp_path, monkeypatch):
    workspace = tmp_path / "Translated PDF"
    workspace.mkdir()
    source = tmp_path / "Raw.pdf"
    source.write_bytes(b"placeholder")
    (workspace / "source_epub.txt").write_text(str(source), encoding="utf-8")
    (workspace / "response_pdf_section_1.html").write_text(
        "<html><body><h1>Abertura traduzida</h1>"
        "<p>translated</p></body></html>",
        encoding="utf-8",
    )
    (workspace / "translation_progress.json").write_text(
        json.dumps(
            {
                "chapters": {
                    "pdf:outline:1": {
                        "actual_num": 1,
                        "output_file": "response_pdf_section_1.html",
                        "original_basename": "pdf_section_1.html",
                        "pdf_toc_section": True,
                        "pdf_toc_title": "Opening",
                        "pdf_start_page": 1,
                        "pdf_end_page": 2,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        epub_library,
        "_epub_reader_webengine_is_warmed",
        lambda: False,
    )

    dialog = EpubReaderDialog(
        str(source),
        config={},
        workspace_dir=str(workspace),
        initial_show_raw=False,
    )

    assert dialog._workspace_mode is True
    assert dialog._workspace_has_raw is True
    assert dialog._workspace_manifest["entries"][0]["title"] == "Opening"
    assert dialog._workspace_manifest["entries"][0]["filename"] == \
        "response_pdf_section_1.html"
    try:
        dialog.show()
        _pump_events(qapp, timeout=1.2)
        assert len(dialog._chapters_overlaid) == 1
        assert "translated" in dialog._chapters_overlaid[0][1]
        assert dialog._chapters_overlaid[0][0] == "Abertura traduzida"
        assert dialog._chapters_raw[0][0] == "Opening"
        assert dialog._chapter_filenames == ["response_pdf_section_1.html"]

        scheduled = []

        class _Signal:
            def connect(self, callback):
                self.callback = callback

        class _RawThreadStub:
            def __init__(self, *args, **kwargs):
                self.done = _Signal()
                self.error = _Signal()
                self.started = False

            def isRunning(self):
                return self.started

            def start(self):
                self.started = True

        monkeypatch.setattr(
            epub_library, "_PdfRawSectionLoaderThread", _RawThreadStub)
        monkeypatch.setattr(
            epub_library.QTimer,
            "singleShot",
            lambda delay, callback: scheduled.append((delay, callback)),
        )

        assert dialog._ensure_workspace_raw_chapter(0) is False
        raw_thread = dialog._workspace_raw_thread
        assert raw_thread.started is False
        assert "Loading raw PDF pages" in dialog._chapters_raw[0][1]
        assert scheduled[0][0] == dialog._RAW_PDF_WORKER_PAINT_DELAY_MS
        scheduled[0][1]()
        assert raw_thread.started is True
    finally:
        dialog.close()
        qapp.processEvents()


def test_pdf_book_details_uses_bookmark_entries_not_workspace_artifacts(
        qapp, tmp_path):
    workspace = tmp_path / "PDF details"
    workspace.mkdir()
    source = tmp_path / "Raw.pdf"
    source.write_bytes(b"placeholder")
    response = workspace / "response_pdf_section_1.html"
    response.write_text("<html><body>translated</body></html>", encoding="utf-8")
    progress_path = workspace / "translation_progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "chapters": {
                    "special_image_rename_map": {
                        "actual_num": 0,
                        "output_file": "image_rename_map.json",
                        "original_basename": "image_rename_map.json",
                        "status": "completed",
                        "auto_discovered": True,
                    },
                    "pdf:outline:1": {
                        "actual_num": 1,
                        "output_file": response.name,
                        "original_basename": "pdf_section_1.html",
                        "status": "completed",
                        "pdf_toc_section": True,
                        "pdf_toc_title": "Opening",
                        "pdf_start_page": 1,
                        "pdf_end_page": 4,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    payload = {}
    loader = _BookDetailsLoader(
        {
            "name": "PDF details",
            "path": str(source),
            "type": "pdf",
            "output_folder": str(workspace),
            "progress_file": str(progress_path),
            "raw_source_path": str(source),
            "compiled_output_kind": "pdf",
        },
        {},
    )
    loader.done.connect(lambda result: payload.update(result))

    loader.run()

    chapters = payload["chapters_info"]
    assert len(chapters) == 1
    assert chapters[0]["raw_title"] == "Opening"
    assert chapters[0]["output_file"] == response.name
    assert chapters[0]["pdf_start_page"] == 1
    assert chapters[0]["pdf_end_page"] == 4


def test_library_pagination_only_builds_current_card_page(qapp):
    dialog = _make_dialog(qapp, 1100, 700)
    try:
        books = _books("paged", 2006)
        dialog._in_progress_books = books
        dialog._cover_path_cache.update(
            {book["path"]: "_none_" for book in books}
        )

        dialog._populate_tab("ip")
        _pump_events(qapp)
        pager = dialog._library_pagers["ip"]
        assert len(dialog._ip_cards) == 20
        assert len(dialog._ip_card_cache) == 20
        assert pager["label"].text() == "Page 1 / 101 · 1-20 of 2006"
        assert dialog._ip_count_label.text() == "2006 novels"
        first_page_paths = {card.book["path"] for card in dialog._ip_cards}

        dialog._on_library_page_action("ip", "next")
        _pump_events(qapp)
        second_page_paths = {card.book["path"] for card in dialog._ip_cards}
        assert len(second_page_paths) == 20
        assert first_page_paths.isdisjoint(second_page_paths)
        # Off-page widgets are deleted instead of accumulating in the cache.
        assert set(dialog._ip_card_cache) == second_page_paths
        assert pager["label"].text() == "Page 2 / 101 · 21-40 of 2006"

        dialog._on_library_page_action("ip", "last")
        _pump_events(qapp)
        assert len(dialog._ip_cards) == 6
        assert len(dialog._ip_card_cache) == 6
        assert pager["label"].text() == (
            "Page 101 / 101 · 2001-2006 of 2006")
        assert not pager["next"].isEnabled()
        assert not pager["last"].isEnabled()
    finally:
        dialog.close()
        qapp.processEvents()


def test_auto_scan_replaces_only_changed_mounted_card(qapp):
    dialog = _make_dialog(qapp, 1100, 450)
    try:
        books = _books("targeted-auto-refresh", 20)
        for book in books:
            book.update({
                "type": "in_progress",
                "workspace_kind": "txt",
                "completed_chapters": 1,
                "total_chapters": 20,
                "translation_state": "in_progress",
            })
        dialog._in_progress_books = books
        dialog._cover_path_cache.update(
            {book["path"]: "_none_" for book in books}
        )
        dialog._populate_tab("ip")
        _pump_events(qapp)

        cards_before = {card.book["path"]: card for card in dialog._ip_cards}
        changed_path = books[5]["path"]
        changed_before = cards_before[changed_path]
        dialog._selected_paths_ip.add(changed_path)
        changed_before.set_selected(True)
        dialog._set_hovered_card(changed_before)
        generation_before = dialog._card_stream_generation["ip"]

        bar = dialog._ip_scroll.verticalScrollBar()
        assert bar.maximum() > 0
        bar.setValue(min(90, bar.maximum()))
        scroll_before = bar.value()

        fresh = [dict(book) for book in books]
        fresh[5]["completed_chapters"] = 2
        fresh[5]["size"] = 4096
        fresh[5]["mtime"] = 10_000.0
        dialog._on_auto_scan_done(fresh, [])

        cards_after = {card.book["path"]: card for card in dialog._ip_cards}
        assert cards_after[changed_path] is not changed_before
        assert cards_after[changed_path].book["completed_chapters"] == 2
        assert cards_after[changed_path]._selected is True
        assert dialog._hovered_card is cards_after[changed_path]
        assert cards_after[changed_path]._hovered is True
        qapp.processEvents()
        for path, old_card in cards_before.items():
            if path != changed_path:
                assert cards_after[path] is old_card
        assert dialog._card_stream_generation["ip"] == generation_before
        assert not dialog._is_card_stream_active("ip")
        assert bar.value() == scroll_before
    finally:
        dialog.close()
        qapp.processEvents()


def test_auto_scan_new_workspace_keeps_full_refresh_fallback(qapp):
    dialog = _make_dialog(qapp, 900, 600)
    try:
        books = _books("auto-refresh-structure", 3)
        dialog._in_progress_books = books
        refresh_calls = []
        dialog._refresh_view = lambda: refresh_calls.append(True)

        fresh = [*books, _books("new-workspace", 1)[0]]
        dialog._on_auto_scan_done(fresh, [])

        assert refresh_calls == [True]
        assert dialog._in_progress_books == fresh
    finally:
        dialog.close()
        qapp.processEvents()


def test_library_page_size_is_shared_and_filter_resets_page(qapp):
    dialog = _make_dialog(qapp, 1100, 700)
    try:
        books = _books("shared-page-size", 66)
        dialog._in_progress_books = books
        dialog._completed_books = list(books)
        dialog._cover_path_cache.update(
            {book["path"]: "_none_" for book in books}
        )
        dialog._populate_tab("ip")
        _pump_events(qapp)
        dialog._on_library_page_action("ip", "last")
        _pump_events(qapp)
        assert dialog._library_pages["ip"] == 3

        ip_combo = dialog._library_pagers["ip"]["page_size"]
        comp_combo = dialog._library_pagers["comp"]["page_size"]
        ip_combo.setCurrentIndex(ip_combo.findData(50))
        _pump_events(qapp)
        assert comp_combo.currentData() == 50
        assert dialog._config["epub_library_page_size"] == 50
        assert dialog._library_pages == {"ip": 0, "comp": 0}
        assert len(dialog._ip_cards) == 50

        dialog._on_library_page_action("ip", "next")
        _pump_events(qapp)
        assert dialog._library_pages["ip"] == 1
        dialog._search.setText("Book 1")
        _pump_events(qapp)
        assert dialog._library_pages["ip"] == 0
        assert dialog._library_pagers["ip"]["label"].text().startswith(
            "Page 1 / 1 · ")
    finally:
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


def test_reader_html_reserves_extra_space_below_page_content():
    class ReaderStub:
        _font_family = "Georgia"
        _font_size = 14
        _line_spacing = 1.8
        _translated_overlay = {}
        _raw_epub_alt_path = ""
        _translated_css_dirs = []
        _workspace_mode = False
        _show_raw = False

        @staticmethod
        def _get_theme():
            return {
                "bg": "#111111",
                "fg": "#eeeeee",
                "heading": "#ffffff",
                "link": "#88aaff",
                "code_bg": "#222222",
                "border": "#333333",
            }

        @staticmethod
        def _get_embedded_css():
            return ""

    paginated = EpubReaderDialog._wrap_html(
        ReaderStub(), "<p>Page text</p>", paginated=True
    )
    scrolling = EpubReaderDialog._wrap_html(
        ReaderStub(), "<p>Page text</p>", paginated=False
    )

    assert "padding: 10px 0 26px 0" in paginated
    assert "window.innerHeight - 36" in paginated
    assert "function _pageCountFor(c)" in paginated
    assert "Math.ceil((c.scrollWidth + gap) / span)" in paginated
    assert "padding: 10px 20px 28px 20px" in scrolling


def test_translated_pdf_workspace_normalizes_legacy_h3_body_weight():
    class ReaderStub:
        _font_family = "Embedded CSS"
        _font_size = 14
        _line_spacing = 1.8
        _translated_overlay = {}
        _raw_epub_alt_path = ""
        _translated_css_dirs = []
        _workspace_mode = True
        _show_raw = False

        @staticmethod
        def _get_theme():
            return {
                "bg": "#111111",
                "fg": "#eeeeee",
                "heading": "#ffffff",
                "link": "#88aaff",
                "code_bg": "#222222",
                "border": "#333333",
            }

        @staticmethod
        def _get_embedded_css():
            return "h3 { font-weight: bold; }"

    expected = (
        "body h3 { font-size: 1em; font-weight: normal !important; margin: 0.6em 0; "
        "padding: 0; }"
    )
    alignment_fix = (
        ".pdf-fast-semantic-page p.pdf-align-justify { "
        "text-align: justify !important; text-justify: auto; }"
    )
    paginated = EpubReaderDialog._wrap_html(
        ReaderStub(), "<h3>Legacy PDF body</h3>", paginated=True
    )
    scrolling = EpubReaderDialog._wrap_html(
        ReaderStub(), "<h3>Legacy PDF body</h3>", paginated=False
    )

    assert expected in paginated
    assert expected in scrolling
    assert alignment_fix in paginated
    assert alignment_fix in scrolling
    assert paginated.index("font-weight: bold") < paginated.index(expected)

    ReaderStub._show_raw = True
    raw = EpubReaderDialog._wrap_html(
        ReaderStub(), "<h3>Real source heading</h3>", paginated=True
    )
    assert expected not in raw
    assert alignment_fix not in raw


def test_reader_recounts_pages_after_hidden_prime_is_revealed(monkeypatch):
    scheduled = []

    class StackStub:
        shown = False

        def show(self):
            self.shown = True

    class ReaderStub:
        _priming_initial_render = True
        _prime_toc_sizes = [240, 960]
        _reader_stack = StackStub()

        def _end_toc_width_lock(self, sizes):
            self.restored_sizes = sizes

        def _resync_page_count(self):
            self.recounted = True

    reader = ReaderStub()
    monkeypatch.setattr(
        epub_library.QTimer,
        "singleShot",
        lambda delay, callback: scheduled.append((delay, callback)),
    )

    EpubReaderDialog._reveal_reader_stack_after_prime(reader)

    assert reader._reader_stack.shown is True
    assert reader._priming_initial_render is False
    assert reader.restored_sizes == [240, 960]
    assert reader._prime_toc_sizes is None
    assert len(scheduled) == 1
    assert scheduled[0][0] == 0

    scheduled[0][1]()
    assert reader.recounted is True


def test_reader_resync_ignores_transitional_page_count(monkeypatch):
    counts = iter([198, 8, 8])

    class ReaderStub:
        _closing = False
        _layout_mode = epub_library.LAYOUT_SINGLE
        _chapters = [("Chapter", "<p>Text</p>")]
        _current_row = 0
        _current_page = 0
        _chapter_page_cache = {0: 7}
        _reader = object()
        _pagination_resync_token = 0

        def _js_page_count(self, browser, callback):
            callback(next(counts))

        def _clamp_page_for_layout(self, page, count):
            return max(0, min(page, count - 1))

        def _js_scroll_to(self, browser, page, animate=True):
            raise AssertionError("Page zero should not need clamping")

        def _update_nav_buttons(self):
            self.nav_updated = True

        def _schedule_search_realign(self):
            self.search_realigned = True

    reader = ReaderStub()
    delays = []

    def run_timer(delay, callback):
        delays.append(delay)
        callback()

    monkeypatch.setattr(epub_library.QTimer, "singleShot", run_timer)

    EpubReaderDialog._resync_page_count(reader)

    assert reader._chapter_page_cache == {0: 8}
    assert delays == [60, 80, 80]
    assert reader.nav_updated is True
    assert reader.search_realigned is True


def test_reader_next_rechecks_live_count_before_leaving_chapter(monkeypatch):
    class TocStub:
        def blockSignals(self, blocked):
            pass

        def setCurrentRow(self, row):
            raise AssertionError("The live final page must be shown first")

    class ReaderStub:
        _layout_mode = epub_library.LAYOUT_SINGLE
        _current_row = 0
        _current_page = 6
        _chapters = [("Current", ""), ("Next", "")]
        _chapter_page_cache = {0: 7}
        _reader = object()
        _toc_list = TocStub()

        def _js_page_count(self, browser, callback):
            callback(8)

        def _advance_paginated_next(self, count):
            EpubReaderDialog._advance_paginated_next(self, count)

        def _scroll_to_page_single(self):
            self.scrolled_to = self._current_page

        def _scroll_to_page_double(self):
            raise AssertionError("Single-page mode should use the single scroller")

        def _render_current(self):
            raise AssertionError("The reader must not skip to the next chapter")

    reader = ReaderStub()
    monkeypatch.setattr(epub_library, "_HAS_WEBENGINE", True)

    EpubReaderDialog._next_chapter(reader)

    assert reader._chapter_page_cache[0] == 8
    assert reader._current_row == 0
    assert reader._current_page == 7
    assert reader.scrolled_to == 7


def test_overlay_worker_extracts_translated_title_from_path_only_entry(tmp_path):
    translated = tmp_path / "response_chapter0039.html"
    translated.write_text(
        "<html><head><meta charset='utf-8'></head>"
        "<body><h1>Episode 38. God Killer (3)</h1>"
        "<p>Translated chapter text.</p></body></html>",
        encoding="utf-8",
    )
    result = {}
    worker = _OverlayMergeThread(
        raw_chapters=[("38화. 신살자 (3)", "<p>Source chapter text.</p>")],
        images={},
        filenames=["chapter0039.xhtml"],
        overlay_map={"chapter0039.xhtml": {"path": os.fspath(translated)}},
        extra_image_dirs=[],
        config={},
    )
    worker.done.connect(
        lambda chapters, images, applied: result.update(
            chapters=chapters,
            images=images,
            applied=applied,
        )
    )

    # Call run directly so the test remains deterministic; production starts
    # the same method on the worker thread.
    worker.run()

    assert result["applied"] is True
    assert result["chapters"] == [
        (
            "Episode 38. God Killer (3)",
            translated.read_text(encoding="utf-8"),
        )
    ]


def test_reader_paints_shell_before_creating_browser_views(
    qapp, monkeypatch,
):
    load_starts = []
    monkeypatch.setattr(epub_library, "_HAS_WEBENGINE", False)
    monkeypatch.setattr(
        EpubReaderDialog,
        "_start_loading",
        lambda self, preserve_shell=False: load_starts.append(preserve_shell),
    )

    dialog = EpubReaderDialog("C:/layout-test/deferred-reader.epub", config={})
    try:
        # Construction remains lightweight; showing the native shell does not
        # synchronously create the expensive reader panes.
        assert dialog._reader is None
        assert not dialog._reader_views_ready
        assert load_starts == []

        dialog.show()
        assert dialog.isVisible()
        assert dialog._reader is None
        assert load_starts == []
        assert dialog._loading_widget.isVisible()
        assert not dialog._toolbar_widget.isVisible()
        assert not dialog._content_widget.isVisible()

        _pump_events(qapp, timeout=0.2)
        assert dialog._reader_views_ready
        assert dialog._reader is not None
        # Double-page mode is a two-column layout in the primary browser; the
        # old auxiliary panes stay unconstructed.
        assert dialog._reader_left is None
        assert dialog._reader_right is None
        assert load_starts == [True]
        assert dialog._loading_widget.isVisible()
        assert not dialog._toolbar_widget.isVisible()
        assert not dialog._content_widget.isVisible()
    finally:
        dialog.close()
        qapp.processEvents()


def test_reader_installs_browser_before_show_when_engine_is_prewarmed(
    qapp, monkeypatch,
):
    load_starts = []
    monkeypatch.setattr(epub_library, "_HAS_WEBENGINE", False)
    monkeypatch.setattr(
        epub_library,
        "_epub_reader_webengine_is_warmed",
        lambda: True,
    )
    monkeypatch.setattr(
        EpubReaderDialog,
        "_start_loading",
        lambda self, preserve_shell=False: load_starts.append(preserve_shell),
    )

    dialog = EpubReaderDialog("C:/layout-test/prewarmed-reader.epub", config={})
    try:
        assert dialog._reader_views_ready
        assert dialog._reader is not None
        assert load_starts == [True]

        dialog.show()
        qapp.processEvents()
        assert dialog.isVisible()
        assert not dialog._reader_init_queued
        assert load_starts == [True]
    finally:
        dialog.close()
        qapp.processEvents()


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


def test_leading_workspace_pdf_image_does_not_start_on_blank_column(tmp_path):
    reader = EpubReaderDialog.__new__(EpubReaderDialog)
    reader._epub_path = str(tmp_path / "source.pdf")
    reader._images = {}
    reader._extra_image_dirs = [str(tmp_path / "images")]
    reader._img_temp_dir = str(tmp_path / "reader-images")
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "cover.png").write_bytes(b"image" * 2000)

    processed = reader._process_html(
        '<article class="pdf-fast-semantic-page">'
        '<a id="page-1"></a>'
        '<figure class="pdf-image"><img src="images/cover.png"></figure>'
        '</article>'
    )

    assert 'class="full-page-img full-page-img-first"' in processed

    class ReaderStub:
        _font_family = "Georgia"
        _font_size = 14
        _line_spacing = 1.8
        _translated_overlay = {}
        _raw_epub_alt_path = ""
        _translated_css_dirs = []
        _workspace_mode = True
        _show_raw = False

        @staticmethod
        def _get_theme():
            return {
                "bg": "#111111",
                "fg": "#eeeeee",
                "heading": "#ffffff",
                "link": "#88aaff",
                "code_bg": "#222222",
                "border": "#333333",
            }

        @staticmethod
        def _get_embedded_css():
            return ""

    wrapped = EpubReaderDialog._wrap_html(
        ReaderStub(), processed, paginated=True
    )
    assert (
        ".full-page-img-first { break-before: avoid !important; }"
        in wrapped
    )


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


def test_pdf_cover_extracts_first_embedded_image_and_reuses_cache(tmp_path, monkeypatch):
    fitz = pytest.importorskip("fitz")
    Image = pytest.importorskip("PIL.Image")
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(epub_library, "_cover_cache_dir", lambda: str(cache_dir))
    pdf_path = tmp_path / "source.pdf"
    first_image = tmp_path / "first.png"
    second_image = tmp_path / "second.png"
    Image.new("RGB", (200, 300), "blue").save(first_image)
    Image.new("RGB", (200, 300), "red").save(second_image)

    with fitz.open() as document:
        first = document.new_page(width=300, height=500)
        first.insert_image(fitz.Rect(50, 80, 250, 380), filename=str(first_image))
        second = document.new_page(width=300, height=500)
        second.insert_image(fitz.Rect(50, 80, 250, 380), filename=str(second_image))
        document.save(str(pdf_path))

    first_result = epub_library._extract_pdf_cover(str(pdf_path))
    second_result = epub_library._extract_pdf_cover(str(pdf_path))

    assert first_result == second_result
    assert first_result and os.path.isfile(first_result)
    image = epub_library.QImage(first_result)
    assert not image.isNull()
    assert image.size() == epub_library.QSize(200, 300)
    # The native first image is blue; a page screenshot would be 300x500 with
    # white margins, while extracting the second image would make this red.
    center = image.pixelColor(image.width() // 2, image.height() // 2)
    assert center.blue() > center.red()


def test_pdf_without_embedded_images_does_not_fall_back_to_page_screenshot(
    tmp_path, monkeypatch
):
    fitz = pytest.importorskip("fitz")
    monkeypatch.setattr(
        epub_library, "_cover_cache_dir", lambda: str(tmp_path / "cache"))
    pdf_path = tmp_path / "vector-only.pdf"
    with fitz.open() as document:
        page = document.new_page(width=300, height=500)
        page.draw_rect(page.rect, fill=(0.1, 0.3, 0.8))
        page.insert_text((40, 80), "NO IMAGE OBJECT")
        document.save(str(pdf_path))

    assert epub_library._extract_pdf_cover(str(pdf_path)) is None


def test_pdf_cover_image_search_stops_after_first_five_pages(
    tmp_path, monkeypatch
):
    fitz = pytest.importorskip("fitz")
    Image = pytest.importorskip("PIL.Image")
    monkeypatch.setattr(
        epub_library, "_cover_cache_dir", lambda: str(tmp_path / "cache"))
    late_image = tmp_path / "late-cover.png"
    Image.new("RGB", (120, 180), "green").save(late_image)
    pdf_path = tmp_path / "late-image.pdf"
    with fitz.open() as document:
        for page_number in range(6):
            page = document.new_page(width=300, height=500)
            if page_number == 5:
                page.insert_image(
                    fitz.Rect(50, 50, 170, 230), filename=str(late_image))
        document.save(str(pdf_path))

    assert epub_library._PDF_COVER_SCAN_PAGE_LIMIT == 5
    assert epub_library._extract_pdf_cover(str(pdf_path)) is None


def test_in_progress_pdf_uses_raw_source_first_image_as_cover(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    raw_pdf = tmp_path / "raw.pdf"
    raw_pdf.write_bytes(b"%PDF-test")
    rendered = str(tmp_path / "first-image.png")
    seen = []
    monkeypatch.setattr(
        epub_library,
        "_extract_pdf_cover",
        lambda path: seen.append(path) or rendered,
    )

    loader = epub_library._CoverLoader(
        str(workspace),
        file_type="in_progress",
        raw_source_path=str(raw_pdf),
    )
    results = []
    loader.result_ready.connect(lambda path, cover: results.append((path, cover)))
    loader.run()

    assert seen == [str(raw_pdf)]
    assert results == [(str(workspace), rendered)]


def test_book_details_never_lists_source_epub_sidecar(qapp, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    raw_pdf = tmp_path / "raw.pdf"
    raw_pdf.write_bytes(b"%PDF-test")
    (workspace / "source_epub.txt").write_text(
        str(raw_pdf), encoding="utf-8")
    (workspace / "image_rename_map.json").write_text(
        "{}", encoding="utf-8")
    (workspace / "response_chapter_001.html").write_text(
        "<h1>Chapter One</h1>", encoding="utf-8")
    progress_file = workspace / "translation_progress.json"
    progress_file.write_text(json.dumps({
        "chapters": {
            "special_source_epub": {
                "actual_num": 0,
                "original_basename": "source_epub.txt",
                "output_file": "source_epub.txt",
                "status": "completed",
            },
            "special_image_rename_map": {
                "actual_num": 0,
                "original_basename": "image_rename_map.json",
                "output_file": "image_rename_map.json",
                "status": "completed",
            },
            "chapter_001": {
                "actual_num": 1,
                "original_basename": "chapter_001.xhtml",
                "output_file": "response_chapter_001.html",
                "status": "completed",
            },
        }
    }), encoding="utf-8")

    summary = epub_library._read_progress_summary(str(progress_file))
    assert summary["total"] == 1
    assert summary["completed"] == 1

    loader = epub_library._BookDetailsLoader({
        "path": str(workspace),
        "type": "in_progress",
        "output_folder": str(workspace),
        "progress_file": str(progress_file),
        "raw_source_path": str(raw_pdf),
    })
    payloads = []
    loader.done.connect(payloads.append)
    loader.run()

    assert len(payloads) == 1
    assert [row["filename"] for row in payloads[0]["chapters_info"]] == [
        "chapter_001.xhtml"
    ]


def test_gif_cover_detection_uses_content_and_movie_advances(qapp, tmp_path):
    Image = pytest.importorskip("PIL.Image")
    disguised_gif = tmp_path / "legacy-cache-name.jpg"
    frames = [
        Image.new("RGB", (40, 60), "red"),
        Image.new("RGB", (40, 60), "blue"),
    ]
    # The cache historically gives every EPUB cover a .jpg suffix. Explicitly
    # encode GIF data under that name to verify content-based detection.
    frames[0].save(
        disguised_gif,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=40,
        loop=0,
    )
    label = epub_library.QLabel()
    movie = epub_library._build_cover_movie(
        str(disguised_gif), label, 120, 180)

    assert movie is not None
    assert movie.isValid()
    # The decoder remains at native size; our frame callback performs smooth
    # scaling instead of QMovie's low-quality setScaledSize path.
    assert not movie.scaledSize().isValid()
    seen_frames = []
    movie.frameChanged.connect(seen_frames.append)
    movie.start()
    _pump_events(qapp, 0.15)
    assert movie.frameCount() == 2
    assert 0 in seen_frames and 1 in seen_frames
    assert movie.state() == epub_library.QMovie.Running
    assert label.pixmap() is not None
    assert label.pixmap().size() == epub_library.QSize(120, 180)
    epub_library._dispose_cover_movie(movie)


def test_other_settings_exposes_remote_image_download_toggle():
    source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    settings_source = open(
        os.path.join(source_root, "src", "other_settings.py"),
        encoding="utf-8",
    ).read()
    extractor_source = open(
        os.path.join(source_root, "src", "Chapter_Extractor.py"),
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
    assert "self.config.get('remote_image_download_interval', 0.5)" in settings_source
    assert "'REMOTE_IMAGE_DOWNLOAD_INTERVAL', '0.5'" in extractor_source
    assert "self.download_remote_image_urls_var = self.config.get(" in gui_source
    assert "('download_remote_image_urls', ['download_remote_image_urls_var']" in gui_source
    assert "('remote_image_download_workers', ['remote_image_download_workers_var']" in gui_source
    assert "('remote_image_download_interval', ['remote_image_download_interval_var']" in gui_source
    assert "self.config.get('remote_image_download_interval', 0.5)" in gui_source


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


def test_output_card_uses_source_epub_when_only_artifact_progress_exists(
    tmp_path, monkeypatch,
):
    """Generated TOC/header rows must not invalidate source_epub.txt."""
    output_root = tmp_path / "Output"
    workspace = output_root / "[1058] Test Book"
    raw_root = tmp_path / "Raws"
    library_raw = tmp_path / "Library" / "Raw"
    workspace.mkdir(parents=True)
    raw_root.mkdir()
    library_raw.mkdir(parents=True)

    source_epub = raw_root / "[1058] Test Book.epub"
    container_xml = """<?xml version="1.0"?>
    <container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
      <rootfiles>
        <rootfile full-path="OEBPS/content.opf"
                  media-type="application/oebps-package+xml"/>
      </rootfiles>
    </container>"""
    opf = """<?xml version="1.0" encoding="UTF-8"?>
    <package xmlns="http://www.idpf.org/2007/opf" version="2.0">
      <manifest>
        <item id="c1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
        <item id="c2" href="chapter2.xhtml" media-type="application/xhtml+xml"/>
      </manifest>
      <spine>
        <itemref idref="c1"/>
        <itemref idref="c2"/>
      </spine>
    </package>"""
    with zipfile.ZipFile(source_epub, "w") as epub:
        epub.writestr("META-INF/container.xml", container_xml)
        epub.writestr("OEBPS/content.opf", opf)
        epub.writestr("OEBPS/chapter1.xhtml", "<p>One</p>")
        epub.writestr("OEBPS/chapter2.xhtml", "<p>Two</p>")

    (workspace / "source_epub.txt").write_text(
        str(source_epub), encoding="utf-8",
    )
    progress = {
        "version": "2.1",
        "chapters": {
            "__metadata__": {
                "original_basename": "metadata.json",
                "output_file": "metadata.json",
                "status": "pending",
                "special_type": "metadata",
            },
            "__translation_artifact__:toc": {
                "original_basename": "TOC.txt",
                "output_file": "TOC.txt",
                "status": "pending",
                "special_type": "toc",
                "translation_artifact_progress_key": (
                    "__translation_artifact__:toc"
                ),
            },
            "__translation_artifact__:headers": {
                "original_basename": "translated_headers.txt",
                "output_file": "translated_headers.txt",
                "status": "pending",
                "special_type": "headers",
                "translation_artifact_progress_key": (
                    "__translation_artifact__:headers"
                ),
            },
        },
    }
    (workspace / "translation_progress.json").write_text(
        json.dumps(progress), encoding="utf-8",
    )

    monkeypatch.setattr(
        epub_library, "_resolve_output_roots",
        lambda _config=None: [str(output_root)],
    )
    monkeypatch.setattr(
        epub_library, "_origins_raw_sources_for_stem", lambda _stem: [],
    )
    monkeypatch.setattr(epub_library, "load_library_raw_inputs", lambda: [])
    monkeypatch.setattr(
        epub_library, "get_library_raw_dir", lambda: str(library_raw),
    )

    rows = epub_library.scan_output_folders({})

    assert len(rows) == 1
    assert os.path.normpath(rows[0]["raw_source_path"]) == os.path.normpath(
        str(source_epub)
    )
    assert rows[0]["missing_raw_file"] is False
    assert rows[0]["total_chapters"] == 2


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


def test_details_chapter_filter_shows_only_qa_failures():
    class DetailsStub:
        _show_special_files = False
        _show_qa_failures_only = True
        _toc_search = None
        _chapters_info = [
            {"index": 0, "status": "completed"},
            {"index": 1, "status": "qa_failed"},
            {"index": 2, "status": "failed"},
            {"index": 3, "status": "qa_failed", "is_special": True},
        ]
        _chapter_base_infos = BookDetailsDialog._chapter_base_infos

    filtered = BookDetailsDialog._filtered_chapter_infos(DetailsStub())

    assert [chapter["index"] for chapter in filtered] == [1]


def test_details_chapters_button_switches_to_failures_view(qapp, monkeypatch):
    monkeypatch.setattr(BookDetailsDialog, "_start_loading", lambda self: None)
    dialog = BookDetailsDialog(
        {"path": "C:/layout-test/book.epub", "name": "Book"},
        config={},
    )
    dialog._auto_refresh_timer.stop()
    dialog._chapters_info = [
        {"index": 0, "status": "completed"},
        {"index": 1, "status": "qa_failed"},
    ]
    dialog._populate_chapters = (
        lambda silent=False: dialog._update_toc_toggle_label()
    )

    assert isinstance(dialog._toc_title, QPushButton)
    assert dialog._toc_title.contextMenuPolicy() == Qt.DefaultContextMenu

    dialog._toc_title.click()
    assert dialog._toc_title.isChecked() is True
    assert dialog._toc_title.text() == "Failures  (1)"

    dialog._toc_title.click()
    assert dialog._toc_title.isChecked() is False
    assert dialog._toc_title.text() == "Chapters  (1/2)"
    dialog.close()


def test_empty_failure_view_keeps_chapter_toolbar_vertical_anchor(
    qapp, monkeypatch
):
    monkeypatch.setattr(BookDetailsDialog, "_start_loading", lambda self: None)
    dialog = BookDetailsDialog(
        {"path": "C:/layout-test/book.epub", "name": "Book"},
        config={},
    )
    dialog._auto_refresh_timer.stop()
    dialog._chapters_info = [
        {
            "index": index,
            "filename": f"chapter-{index}.xhtml",
            "title": f"Chapter {index}",
            "status": "completed",
        }
        for index in range(358)
    ]
    first_page = dialog._chapters_info[:20]
    dialog._chap_list.append_specs([
        {
            "info": info,
            "primary_text": info["title"],
            "filename": info["filename"],
        }
        for info in first_page
    ])

    try:
        dialog.resize(1565, 800)
        dialog._chap_list.show()
        dialog._chap_container.hide()
        dialog._toc_bottom_pager.show()
        dialog._chap_loading_lbl.hide()
        dialog._chapters_loaded = True
        dialog._update_toc_toggle_label()
        dialog.show()
        qapp.processEvents()

        viewport = dialog._scroll.viewport()
        body = dialog._scroll.widget()
        toolbar_body_y = dialog._toc_title.mapTo(body, QPoint(0, 0)).y()
        dialog._scroll.verticalScrollBar().setValue(toolbar_body_y - 70)
        qapp.processEvents()
        before_y = dialog._toc_title.mapTo(viewport, QPoint(0, 0)).y()

        dialog._toc_title.click()
        qapp.processEvents()
        after_y = dialog._toc_title.mapTo(viewport, QPoint(0, 0)).y()

        assert dialog._toc_title.text() == "Failures  (0)"
        assert dialog._chap_list.count() == 0
        assert dialog._chap_list.reserved_height() > 0
        assert after_y == before_y

        # Restoring from an empty failure view used to expose a hybrid frame:
        # the pager jumped to the chapter count while the old Failures label
        # and blank list remained visible until background row prep finished.
        failure_page_text = dialog._toc_page_label.text()
        assert failure_page_text == "Page 1 / 1 · 0 of 0"
        assert not dialog._toc_bottom_pager.isVisible()

        dialog._toc_title.click()

        # Before queued row-prep results are applied, the complete previous
        # state remains intact rather than showing half of the new state.
        assert dialog._toc_title.text() == "Failures  (0)"
        assert dialog._toc_page_label.text() == failure_page_text
        assert not dialog._toc_bottom_pager.isVisible()

        _pump_events(qapp, timeout=0.3)

        restored_y = dialog._toc_title.mapTo(viewport, QPoint(0, 0)).y()
        assert dialog._toc_title.text() == "Chapters  (358/358)"
        assert dialog._toc_page_label.text() == "Page 1 / 18 · 1-20 of 358"
        assert dialog._toc_bottom_pager.isVisible()
        assert dialog._chap_list.count() == 20
        assert restored_y == before_y
    finally:
        dialog.close()
        qapp.processEvents()
