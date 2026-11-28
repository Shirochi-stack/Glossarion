"""The Progress Manager window: one window for one or many files, a compact header with one
file dropdown shared by every tab, and the Translation / Glossary / Interactive Images tabs
(built on first open). The Glossary tab is the old Glossary Progress dialog's page."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
for _p in (str(TESTS_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from parity import progress_legacy as pl  # noqa: E402

TAB_NAMES = ("translation", "glossary", "images")


@pytest.fixture(autouse=True)
def delete_widgets_after_test():
    """Close and delete every window a test built (Linux CI crashes at exit otherwise)."""
    from PySide6.QtCore import QCoreApplication, QEvent

    app = pl.qapp()
    before = list(app.topLevelWidgets())
    yield
    for widget in app.topLevelWidgets():
        if any(widget is known for known in before):
            continue
        widget.hide()
        widget.deleteLater()
    for _ in range(3):
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def _books(base, count):
    """``count`` EPUBs (Book1.epub ...) with their own output folders and glossary progress."""
    base = Path(base)
    (base / "out").mkdir(parents=True, exist_ok=True)
    sources, config = [], None
    for index in range(1, count + 1):
        work = base / f"w{index}"
        work.mkdir(parents=True)
        source, out, config = pl.epub_workspace(work)
        name = f"Book{index}"
        target = base / f"{name}.epub"
        shutil.move(str(source), str(target))
        shutil.move(str(out), str(base / "out" / name))
        pl.glossary_progress_fixture(base, target)
        sources.append(str(target))
    return sources, dict(config, output_directory=str(base / "out"))


def _open(tmp_path, count):
    sources, config = _books(tmp_path, count)
    host = pl.make_host(pl.current_rg(), config)
    host.selected_files = list(sources)
    if count == 1:
        window = host._show_retranslation_shell_then_build(sources[0])
        pl.pump(30, until=lambda: window._pm_page_data, timeout=10)
    else:
        host._force_retranslation_multiple_files(list(sources))
        window = host._multi_file_retranslation_dialog
        pl.pump(30, until=lambda: not getattr(window, "_multi_file_tabs_building", True), timeout=20)
    pl.pump(30, timeout=0.3)
    return host, window, sources


def _tab_name(window):
    return window._pm_current_tab_name()


@pytest.mark.parametrize("count", (1, 2, 4, 8))
def test_one_window_one_dropdown_three_tabs(count, tmp_path):
    from PySide6.QtWidgets import QComboBox, QTabWidget

    _host, window, sources = _open(tmp_path, count)
    tabs = window._pm_tabs
    assert [tabs.widget(i)._pm_tab_name for i in range(tabs.count())] == list(TAB_NAMES)
    assert [tabs.tabText(i).split(" (")[0] for i in range(tabs.count())] == [
        "📖 Translation", "📚 Glossary", "🖼 Interactive Images"]
    # no per-file tabs any more: the window's only QTabWidget is the 3-tab one
    assert window.findChildren(QTabWidget) == [tabs]
    combo = window._pm_file_combo
    assert isinstance(combo, QComboBox) and combo.count() == count
    assert [os.path.normcase(f) for f in window._pm_files] == [
        os.path.normcase(os.path.abspath(s)) for s in sources]
    # the file row (dropdown, arrows, "1 / n" counter) only shows with something to switch to
    assert window._pm_nav.isVisibleTo(window) is (count > 1)
    assert window._pm_nav.prev_button.isVisibleTo(window) is (count > 1)
    assert window._pm_translation_stack.count() == count


def test_file_dropdown_is_the_house_combo(tmp_path):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent

    _host, window, _sources = _open(tmp_path, 2)
    combo = window._pm_file_combo
    sheet = combo.styleSheet()
    assert "Halgakos.ico" in sheet and "width: 22px" in sheet and "#4a5568" in sheet
    assert "drop-down { border:none; }" not in sheet
    combo.setCurrentIndex(0)
    event = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, -120),
                        Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    combo.wheelEvent(event)
    assert combo.currentIndex() == 0


def test_glossary_tab_and_its_file_pages_build_lazily(tmp_path, monkeypatch):
    rg = pl.current_rg()
    built = []
    original = rg.make_glossary_progress_model

    def recording(owner, fp, gp_path, **kwargs):
        built.append((os.path.basename(fp), kwargs.get("output_dir")))
        return original(owner, fp, gp_path, **kwargs)

    monkeypatch.setattr(rg, "make_glossary_progress_model", recording)
    _host, window, sources = _open(tmp_path, 3)
    assert getattr(window, "_pm_glossary_page", None) is None and built == []

    window._pm_show_tab("glossary")
    pl.pump(30, timeout=0.5)
    assert getattr(window, "_pm_glossary_page", None) is not None
    assert [name for name, _out in built] == ["Book1.epub"]

    window._pm_file_combo.setCurrentIndex(2)
    pl.pump(30, timeout=0.5)
    assert [name for name, _out in built] == ["Book1.epub", "Book3.epub"]
    # the Interactive Images tab is a placeholder in this version
    from PySide6.QtWidgets import QLabel

    window._pm_show_tab("images")
    texts = [label.text() for label in window._pm_tab_pages["images"].findChildren(QLabel)]
    assert "No interactive images yet" in texts


def test_each_book_uses_its_own_output_folder(tmp_path, monkeypatch):
    """Regression: every book's glossary panel used the last-built book's output folder
    and progress data (footnotes, summaries)."""
    rg = pl.current_rg()
    seen = {}
    original = rg.make_glossary_progress_model

    def recording(owner, fp, gp_path, **kwargs):
        seen[os.path.basename(fp)] = (kwargs.get("output_dir"), kwargs.get("prog"))
        return original(owner, fp, gp_path, **kwargs)

    monkeypatch.setattr(rg, "make_glossary_progress_model", recording)
    _host, window, sources = _open(tmp_path, 2)
    window._pm_show_tab("glossary", sources[0])
    pl.pump(30, timeout=0.5)
    window._pm_select_file(sources[1])
    pl.pump(30, timeout=0.5)
    assert os.path.basename(os.path.normpath(seen["Book1.epub"][0])) == "Book1"
    assert os.path.basename(os.path.normpath(seen["Book2.epub"][0])) == "Book2"
    data = window._pm_page_data
    for source in sources:
        assert seen[os.path.basename(source)][1] is data[os.path.abspath(source)]["prog"]


def test_glossary_timer_runs_only_while_visible_and_current(tmp_path):
    _host, window, sources = _open(tmp_path, 2)
    window._pm_show_tab("glossary")
    pl.pump(30, timeout=0.3)
    page = window._glossary_progress_dialog
    timer = page._gp_auto_refresh_timer
    assert timer.interval() == 2000 and timer.isActive()
    window._pm_show_tab("translation")
    assert not timer.isActive()
    window._pm_show_tab("glossary")
    assert timer.isActive()
    window.hide()
    assert not timer.isActive()
    window.show()
    assert timer.isActive()


def test_glossary_button_switches_to_the_glossary_tab_on_the_same_file(tmp_path):
    from PySide6.QtWidgets import QPushButton

    _host, window, sources = _open(tmp_path, 3)
    window._pm_select_file(sources[1])
    pl.pump(30, timeout=0.2)
    page = window._pm_page_data[os.path.abspath(sources[1])]["container"]
    button = next(b for b in page.findChildren(QPushButton) if "Glossary Progress" in b.text())
    button.click()
    pl.pump(30, timeout=0.3)
    assert _tab_name(window) == "glossary"
    assert os.path.normcase(window._pm_current_file()) == os.path.normcase(os.path.abspath(sources[1]))
    assert window._glossary_progress_dialog is window._pm_glossary_page
    assert window._glossary_progress_dialog.window() is window


def test_input_change_on_the_glossary_tab_reopens_the_glossary_tab(tmp_path):
    sources, config = _books(tmp_path, 1)
    other, _config = _books(tmp_path / "second", 1)
    host = pl.make_host(pl.current_rg(), config)
    opened = []

    def force_retranslation():
        source = host.selected_files[0]
        window = host._show_retranslation_shell_then_build(source)
        host._retranslation_dialog_cache = {os.path.abspath(source): {"dialog": window}}
        opened.append(window)

    host.force_retranslation = force_retranslation
    host.selected_files = [sources[0]]
    force_retranslation()
    first = opened[0]
    pl.pump(30, until=lambda: first._pm_page_data, timeout=10)
    first._show_glossary_progress()
    pl.pump(30, timeout=0.3)
    assert _tab_name(first) == "glossary"
    host._refresh_open_progress_managers_for_input_change()  # stamps the signature
    host.selected_files = [other[0]]
    host._refresh_open_progress_managers_for_input_change()
    pl.pump(30, until=lambda: len(opened) == 2 and opened[1]._pm_page_data, timeout=10)
    pl.pump(30, until=lambda: _tab_name(opened[1]) == "glossary", timeout=5)
    assert _tab_name(opened[1]) == "glossary" and opened[1].isVisible()
    assert getattr(opened[1], "_pm_glossary_page", None) is not None


def test_glossary_manager_refresh_reaches_the_window(tmp_path):
    """GlossaryManager's Minimal-pass toggle calls ``_gp_refresh_visible`` on every visible
    top-level window: the Progress Manager forwards it to its Glossary tab when current."""
    from PySide6.QtWidgets import QApplication

    _host, window, _sources = _open(tmp_path, 1)
    window._pm_show_tab("glossary")
    pl.pump(30, timeout=0.3)
    calls = []
    window._pm_glossary_page._gp_refresh_visible = lambda *a, **k: calls.append(1)
    hooks = [w for w in QApplication.topLevelWidgets()
             if callable(getattr(w, "_gp_refresh_visible", None)) and w.isVisible()]
    assert window in hooks
    window._gp_refresh_visible()
    assert calls == [1]
    window._pm_show_tab("translation")
    window._gp_refresh_visible()
    assert calls == [1]


def test_header_fonts_follow_the_gui_font_scale(tmp_path):
    import dpi_setup
    from PySide6.QtWidgets import QWidget

    sources, config = _books(tmp_path, 2)
    sizes = {}
    was_installed = dpi_setup._stylesheet_patch_installed
    previous_factor = dpi_setup._stylesheet_scale_factor
    try:
        for factor in (1.0, 1.5):
            dpi_setup._install_stylesheet_font_scale_patch(factor)
            host = pl.make_host(pl.current_rg(), config)
            window = host._build_progress_manager_window("Progress Manager", summary_text="Selected: 2")
            window._pm_add_file(sources[0], "Book1")
            label = window._pm_nav.label
            label.ensurePolished()
            sizes[factor] = label.font().pointSizeF()
            window.deleteLater()
    finally:
        dpi_setup._stylesheet_scale_factor = previous_factor
        if not was_installed and getattr(QWidget, "_glossarion_original_setStyleSheet", None) is not None:
            QWidget.setStyleSheet = QWidget._glossarion_original_setStyleSheet
            dpi_setup._stylesheet_patch_installed = False
    assert sizes[1.0] == pytest.approx(9.0)
    assert sizes[1.5] == pytest.approx(13.5)


def test_standalone_manager_opens_the_glossary_page_in_a_one_tab_window(tmp_path):
    sources, config = _books(tmp_path, 1)
    host = pl.make_host(pl.current_rg(), config)
    data = pl.open_progress_manager(host, sources[0])
    data["dialog"]._show_glossary_progress()
    pl.pump(30, timeout=0.5)
    page = data["dialog"]._glossary_progress_dialog
    window = page.window()
    assert window is not data["dialog"] and window._pm_tabs.count() == 1
    assert window._pm_tabs.tabBar().isHidden()
    assert page._gp_auto_refresh_timer.isActive()
    # the button again shows the same page
    data["dialog"]._show_glossary_progress()
    assert data["dialog"]._glossary_progress_dialog is page
    window.hide()
    assert not page._gp_auto_refresh_timer.isActive()
