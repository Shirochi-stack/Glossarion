import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop
from PySide6.QtWidgets import QApplication

from epub_library import EpubLibraryDialog, SIZE_NORMAL


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
