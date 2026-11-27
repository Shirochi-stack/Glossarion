"""Interactive Images desktop UI: the house widgets (Halgakos combos without wheel, native spin box),
the reference-image warning, the styles preview (WebEngine and the QTextBrowser fallback), the
Other Settings rows and the Progress Manager tab page."""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SRC = TESTS_DIR.parent / "src"
for _p in (str(TESTS_DIR), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")
pytest.importorskip("PySide6")


def _app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def delete_widgets_after_test():
    """Close and delete every window a test built (Linux CI crashes at exit otherwise)."""
    from PySide6.QtCore import QCoreApplication, QEvent

    app = _app()
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


def _owner(config=None):
    logs = []
    owner = SimpleNamespace(config=dict(config or {}), logs=logs)
    owner.save_config = lambda show_message=False: True
    owner.append_log = logs.append
    return owner


def _wheel(widget):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    event = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, -120),
                        Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    widget.wheelEvent(event)


def test_settings_dialog_uses_the_house_widgets(tmp_path):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QStyle, QStyleOptionSpinBox

    import interactive_images_gui as ixg

    _app()
    owner = _owner({'interactive_images_image_model': 'dall-e-3'})
    dialog = ixg.InteractiveImagesDialog(owner, None, output_dir=str(tmp_path))
    dialog.show()
    for combo in (dialog.per_chapter, dialog.resolution, dialog.aspect, dialog.image_model, dialog.text_model):
        assert "Halgakos.ico" in combo.styleSheet() and "width: 22px" in combo.styleSheet()
    dialog.per_chapter.setCurrentIndex(1)
    _wheel(dialog.per_chapter)
    assert dialog.per_chapter.currentIndex() == 1
    assert dialog.per_chapter.currentText() == '1'          # default 1 image per chapter
    # the spin box: native (no sheet reaches it), default 5, its up button works, the wheel does nothing
    spin = dialog.max_refs
    assert spin.value() == 5 and spin.styleSheet() == '' and 'QSpinBox' not in dialog.styleSheet()
    _wheel(spin)
    assert spin.value() == 5
    option = QStyleOptionSpinBox()
    spin.initStyleOption(option)
    up = spin.style().subControlRect(QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxUp, spin)
    QTest.mouseClick(spin, Qt.LeftButton, Qt.NoModifier, up.center())
    assert spin.value() == 6
    # a model without reference images gets the warning
    assert dialog.ref_warning.isVisibleTo(dialog) and 'no reference images' in dialog.ref_warning.text()
    dialog.image_model.setEditText('gemini-3-pro-image-preview')
    assert '14 reference images' in dialog.ref_warning.text()
    assert dialog.keys_btn is not None and 'Image Keys' in dialog.keys_btn.text()
    # Save writes the settings and the profile's runtime keys
    dialog.per_chapter.setCurrentText('3')
    dialog._save_all()
    assert owner.config['interactive_images_max_references'] == 6
    assert owner.config['interactive_images_per_chapter'] == '3'
    assert owner.config['interactive_images_image_model'] == 'gemini-3-pro-image-preview'
    assert dialog.save_btn.text().startswith('💾')


def test_styles_dialog_preview_fallback_and_presets():
    import interactive_css_presets as presets
    import interactive_images_gui as ixg

    _app()
    owner = _owner()
    dialog = ixg.InteractiveStylesDialog(owner, None, force_fallback=True)
    assert not dialog.preview.uses_webengine
    assert dialog.preview.note.text() == ixg.FALLBACK_NOTE
    assert dialog.combo.count() == 12 and "Halgakos.ico" in dialog.combo.styleSheet()
    dialog.combo.setCurrentText('Retro Terminal')
    assert dialog.preview._css == presets.BUILTIN_PRESETS['Retro Terminal']
    assert owner.config['interactive_css_preset'] == 'Retro Terminal'
    assert dialog.editor.isReadOnly() and not dialog.save_btn.isEnabled()
    dialog._duplicate()
    assert dialog.current_name() == 'Retro Terminal copy'
    assert not dialog.editor.isReadOnly() and 'Retro Terminal copy' in owner.config['interactive_css_user_presets']
    dialog.editor.setPlainText('.glx-book { color: red; }')
    dialog._save()
    assert owner.config['interactive_css_user_presets']['Retro Terminal copy'].startswith('.glx-book {')


def test_styles_dialog_webengine_preview():
    import interactive_images_gui as ixg

    if not ixg._HAS_WEBENGINE:
        pytest.skip("Qt WebEngine is not installed")
    _app()
    dialog = ixg.InteractiveStylesDialog(_owner(), None)
    try:
        assert dialog.preview.uses_webengine
        assert dialog.preview._css
    finally:
        view = dialog.preview.view
        view.setParent(None)
        view.deleteLater()


def test_other_settings_rows_keep_the_attach_css_override(tmp_path):
    from PySide6.QtWidgets import QVBoxLayout, QWidget

    import interactive_images_gui as ixg

    _app()
    owner = _owner({'epub_css_override_path': str(tmp_path / 'mine.css'), 'interactive_css_enabled': True})
    host = QWidget()
    layout = QVBoxLayout(host)
    ixg.add_interactive_styles_rows(owner, layout, host)
    checkbox = owner.interactive_css_enabled_checkbox
    assert checkbox.isChecked()
    assert "Halgakos.ico" in owner.interactive_css_preset_combo.styleSheet()
    owner.interactive_css_preset_combo.setCurrentText('Wuxia Ink')
    assert owner.config['interactive_css_preset'] == 'Wuxia Ink'
    clear = [b for b in host.findChildren(__import__('PySide6.QtWidgets', fromlist=['QPushButton']).QPushButton)
             if b.text() == 'Clear'][0]
    clear.click()
    assert owner.config['interactive_css_enabled'] is False
    assert owner.config['epub_css_override_path'] == str(tmp_path / 'mine.css')


def _book(tmp_path):
    out = tmp_path / 'book'
    out.mkdir()
    for n in (1, 2, 3):
        (out / f'response_00{n}_c.html').write_text(f'<body><p>Chapter {n} text.</p></body>', encoding='utf-8')
    (out / 'translation_progress.json').write_text(json.dumps({'chapters': {
        str(n): {'actual_num': n, 'output_file': f'response_00{n}_c.html', 'status': 'completed'} for n in (1, 2, 3)}}),
        encoding='utf-8')
    import interactive_images_core as core
    import interactive_images_progress_core as ipc
    path = core.progress_path(str(out))
    html1 = (out / 'response_001_c.html').read_text(encoding='utf-8')
    ipc.set_entry(path, 'response_001_c.html', {'chapter': 1, 'status': 'completed',
                                               'source_hash': core.visible_text_hash(html1),
                                               'images': [{'file': 'Glossarion_ch001_1.png', 'status': 'completed'}]})
    ipc.set_entry(path, 'response_002_c.html', {'chapter': 2, 'status': 'completed', 'source_hash': 'old',
                                               'images': []})
    return out


def test_images_tab_page_rows_stats_and_timer_gating(tmp_path):
    from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

    import interactive_images_gui as ixg

    _app()
    out = _book(tmp_path)
    fp = str(tmp_path / 'Book.epub')
    window = QWidget()
    window.setLayout(QVBoxLayout())
    tabs = QTabWidget(window)
    other = QWidget()
    host = QWidget()
    host.setLayout(QVBoxLayout())
    tabs.addTab(other, "Translation")
    tabs.addTab(host, "Interactive Images")
    counts = {}
    window._pm_tabs = tabs
    window._pm_images_output_dirs = {os.path.normcase(os.path.abspath(fp)): str(out)}
    window._pm_current_file = lambda: os.path.abspath(fp)
    window._pm_current_tab_name = lambda: 'images' if tabs.currentWidget() is host else 'translation'
    window._pm_set_tab_count = lambda name, text=None: counts.__setitem__(name, text)
    window._pm_tab_callbacks = {}
    window._pm_file_changed_callbacks = []
    ixg.build_images_page(_owner(), window, host)
    panel = host._ix_stack.currentWidget()
    assert isinstance(panel, ixg.ImagesFilePanel)
    assert [row.status for row in panel._rows] == ['completed', 'stale', 'not_built']
    assert panel.list.count() == 3 and 'Ch.   1' in panel.list.item(0).text()
    assert panel.stats['built'] == 1 and panel.stats['stale'] == 1 and panel.stats['not_built'] == 1
    assert counts['images'] == '1/3, 1 stale'
    shown, hidden = window._pm_tab_callbacks['images']
    window.show()
    tabs.setCurrentWidget(host)
    shown()
    assert host._ix_timer.isActive()
    hidden()
    assert not host._ix_timer.isActive()
    # Mark for rebuild through the shared action, then the panel shows it
    import interactive_images_progress_core as ipc
    import interactive_images_core as core
    ipc.mark_for_rerun(core.progress_path(str(out)), [panel._rows[0]])
    panel.refresh(force=True)
    assert panel._rows[0].status == 'pending'
