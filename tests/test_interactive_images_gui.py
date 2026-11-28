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
    assert dialog.combo.count() == 22 and "Halgakos.ico" in dialog.combo.styleSheet()   # 12 + 10 fantasy/leveling
    assert dialog.combo.findText('Dungeon System') >= 0 and dialog.combo.findText("Dragon's Hoard") >= 0
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
    html1 = (out / 'response_001_c.html').read_text(encoding='utf-8').replace(
        '</p>', '</p>' + core.figure_html(core.image_src('Glossarion_ch001_1.png')), 1)
    (out / 'response_001_c.html').write_text(html1, encoding='utf-8')  # built: the file holds its figure
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


# ---- round 2: gender / subjects / Auto ratio / placeholder texts / gallery ---------------------------------

def _png_bytes(color=(10, 20, 30), size=(60, 90)):
    from io import BytesIO
    from PIL import Image
    buffer = BytesIO()
    Image.new('RGB', size, color).save(buffer, format='PNG')
    return buffer.getvalue()


def test_settings_dialog_gender_subjects_and_the_auto_ratio_checklist(tmp_path):
    from PySide6.QtCore import QCoreApplication, QPoint, Qt
    from PySide6.QtTest import QTest

    import interactive_images_core as core
    import interactive_images_gui as ixg

    _app()
    owner = _owner()
    dialog = ixg.InteractiveImagesDialog(owner, None, output_dir=str(tmp_path))
    dialog.show()
    assert dialog.gender_cb.isChecked()                                  # default On
    assert [dialog.subjects.itemText(i) for i in range(dialog.subjects.count())] == list(core.SUBJECTS)
    assert dialog.subjects.currentText() == 'characters' and "Halgakos.ico" in dialog.subjects.styleSheet()
    assert 'auto' in [dialog.aspect.itemText(i) for i in range(dialog.aspect.count())]
    assert dialog.aspect.findText('21:9') >= 0
    assert not dialog.ratios_btn.isVisibleTo(dialog)                     # a fixed ratio: no checklist button
    dialog.aspect.setCurrentText('auto')
    assert dialog.ratios_btn.isVisibleTo(dialog) and dialog.ratios_btn.text() == 'Ratios (4)…'
    # the button toggles: open, second click closes
    popup = dialog.toggle_ratios()
    assert popup.isVisible() and popup.selected() == ['1:1', '3:4', '4:3', '16:9']
    assert dialog.toggle_ratios() is popup and not popup.isVisible()
    popup.boxes['21:9'].setChecked(True)
    popup.boxes['1:1'].setChecked(False)
    assert dialog.aspect_choices == ['3:4', '4:3', '16:9', '21:9'] and dialog.ratios_btn.text() == 'Ratios (4)…'
    # a click right after the list closed (the click that closed it) must not reopen it
    popup.show()
    popup.hide()
    assert dialog.toggle_ratios() is popup and not popup.isVisible()
    dialog.gender_cb.setChecked(False)
    dialog.subjects.setCurrentText('both')
    dialog._save_all()
    assert owner.config['interactive_images_include_gender'] is False
    assert owner.config['interactive_images_subjects'] == 'both'
    assert owner.config['interactive_images_aspect_ratio'] == 'auto'
    assert owner.config['interactive_images_aspect_choices'] == ['3:4', '4:3', '16:9', '21:9']
    settings = core.settings_from_config(owner.config)
    assert settings.include_gender is False and settings.subjects == 'both'
    assert settings.aspect_choices == ('3:4', '4:3', '16:9', '21:9')
    # the prompt editors carry the placeholder menu (Available placeholders + Edit placeholder text...)
    assert all(editor.contextMenuPolicy() == Qt.CustomContextMenu for editor in dialog.editors.values())
    assert dialog.placeholders_btn.text().startswith('✏') and dialog.gallery_btn.text().startswith('🖼')


def test_placeholder_text_dialog_edits_resets_and_saves(tmp_path):
    import interactive_images_core as core
    import interactive_images_gui as ixg

    _app()
    owner = _owner()
    dialog = ixg.PlaceholderTextDialog(owner, None, select='character_gender')
    assert dialog.combo.count() == len(core.placeholder_text_defs()) and "Halgakos.ico" in dialog.combo.styleSheet()
    assert dialog.editor.toPlainText() == '{name} ({gender})' and dialog.status.text() == 'Default text'
    dialog.editor.setPlainText('{name} [{gender}]')
    assert 'unsaved' in dialog.status.text()
    dialog.save()
    assert owner.config[core.PLACEHOLDER_TEXTS_KEY] == {'character_gender': '{name} [{gender}]'}
    assert core.format_character('Mira', 'female', core.settings_from_config(owner.config)) == 'Mira [female]'
    assert dialog.save_btn.text().startswith('💾')
    # another placeholder, then back: the unsaved draft is kept while the window is open
    dialog.combo.setCurrentIndex(dialog.combo.findData('gender_male'))
    dialog.editor.setPlainText('he')
    dialog.combo.setCurrentIndex(dialog.combo.findData('character_gender'))
    dialog.combo.setCurrentIndex(dialog.combo.findData('gender_male'))
    assert dialog.editor.toPlainText() == 'he'
    dialog.reset()
    assert dialog.editor.toPlainText() == 'male'
    dialog.save()
    assert owner.config[core.PLACEHOLDER_TEXTS_KEY] == {'character_gender': '{name} [{gender}]'}   # default: no override
    dialog.combo.setCurrentIndex(dialog.combo.findData('reference_line'))
    dialog.editor.setPlainText('Ref {n}: {name}')
    dialog.save()
    assert owner.config['interactive_images_reference_line'] == 'Ref {n}: {name}'


def _gallery_book(tmp_path):
    import interactive_images_core as core
    import interactive_images_progress_core as ipc

    out = tmp_path / 'gal'
    out.mkdir()
    folder = out / core.INTERACTIVE_FOLDER
    folder.mkdir()
    chapters, entries = {}, {}
    for n, characters in ((1, ['Mira']), (2, ['Mira', 'Tomas']), (3, [])):
        name = f'response_00{n}_c.html'
        image = f'Glossarion_ch00{n}_1.png'
        html = (f'<html><body><p>Chapter {n} text.</p>' + core.figure_html(core.image_src(image)) + '</body></html>')
        (out / name).write_text(html, encoding='utf-8')
        (folder / image).write_bytes(_png_bytes((40 * n, 80, 120)))
        chapters[str(n)] = {'actual_num': n, 'output_file': name, 'status': 'completed'}
        entries[name] = {'chapter': n, 'status': 'completed', 'source_hash': core.visible_text_hash(html),
                         'images': [{'file': image, 'status': 'completed', 'characters': characters,
                                     'scene': characters, 'prompt': f'scene {n}'}], 'classes': []}
    (out / 'translation_progress.json').write_text(json.dumps({'chapters': chapters}), encoding='utf-8')
    (out / 'glossary.csv').write_text('type,raw_name,translated_name\ncharacter,미라,Mira\ncharacter,토마스,Tomas\n'
                                      'character,하나,Hana\n', encoding='utf-8')
    for key, entry in entries.items():
        ipc.set_entry(core.progress_path(str(out)), key, entry)
    return out


def test_gallery_dialog_filters_selects_and_deletes_several(tmp_path):
    from PySide6.QtCore import Qt

    import interactive_images_core as core
    import interactive_images_gui as ixg

    _app()
    out = _gallery_book(tmp_path)
    owner = _owner()
    dialog = ixg.InteractiveImagesGalleryDialog(owner, str(out), None, confirm=False)
    dialog.show()
    assert dialog.list.count() == 3
    assert [dialog.list.item(i).text() for i in range(3)] == ['Ch. 1 · Mira', 'Ch. 2 · Mira, Tomas', 'Ch. 3']
    assert not dialog.list.item(0).icon().isNull()
    # the character filter lists the glossary's names (also those without an image)
    names = [dialog.character_combo.itemText(i) for i in range(dialog.character_combo.count())]
    assert names == [ixg.ALL_CHARACTERS, 'Hana', 'Mira', 'Tomas']
    dialog.character_combo.setCurrentIndex(dialog.character_combo.findText('Tomas'))
    assert dialog.list.count() == 1 and dialog.list.item(0).text().startswith('Ch. 2')
    dialog.character_combo.setCurrentIndex(0)
    dialog.chapter_combo.setCurrentIndex(dialog.chapter_combo.findData('response_003_c.html'))
    assert dialog.list.count() == 1
    dialog.chapter_combo.setCurrentIndex(0)
    # multi selection: a ticked box and a highlighted card (ctrl / shift click) both count
    assert not dialog.delete_btn.isEnabled()
    dialog.list.item(0).setCheckState(Qt.Checked)
    dialog.list.item(2).setSelected(True)
    assert len(dialog.selected_images()) == 2 and dialog.delete_btn.isEnabled() and '2 selected' in dialog.count_label.text()
    dialog.select_none()
    assert dialog.selected_images() == []
    dialog.select_all()
    assert len(dialog.selected_images()) == 3
    dialog.list.item(1).setCheckState(Qt.Unchecked)
    assert dialog.delete_selected() == 2
    assert dialog.list.count() == 1 and dialog.list.item(0).text().startswith('Ch. 2')
    folder = out / core.INTERACTIVE_FOLDER
    assert not (folder / 'Glossarion_ch001_1.png').exists() and (folder / 'Glossarion_ch002_1.png').exists()
    assert 'data-glx' not in (out / 'response_001_c.html').read_text(encoding='utf-8')
    assert 'data-glx' in (out / 'response_002_c.html').read_text(encoding='utf-8')
    assert any('deleted 2 image' in line for line in owner.logs)


def test_gallery_asks_before_deleting_and_the_viewer_deletes_one(tmp_path, monkeypatch):
    import interactive_images_core as core
    import interactive_images_gui as ixg

    _app()
    out = _gallery_book(tmp_path)
    dialog = ixg.InteractiveImagesGalleryDialog(_owner(), str(out), None)
    asked = []
    import review_dialog
    monkeypatch.setattr(review_dialog, 'confirm_prompt_profile_action',
                        lambda parent, title, text: asked.append((title, text)) or False)
    assert dialog.delete_images(dialog.images[:1]) == 0                  # refused: nothing deleted
    assert asked and asked[0][0] == 'Delete Images' and '1 image' in asked[0][1]
    assert (out / core.INTERACTIVE_FOLDER / 'Glossarion_ch001_1.png').exists()
    monkeypatch.setattr(review_dialog, 'confirm_prompt_profile_action', lambda *a: True)
    viewer = dialog.open_viewer(dialog.images[0])
    assert 'Mira' in viewer.windowTitle() and viewer.picture.pixmap() is not None
    viewer.delete_btn.click()
    assert dialog.list.count() == 2 and not (out / core.INTERACTIVE_FOLDER / 'Glossarion_ch001_1.png').exists()
    opened = []
    monkeypatch.setattr(ixg, 'open_path', lambda path, make=False: opened.append(path) or True)
    assert dialog.go_to_chapter(dialog.images[0]) and opened[-1].endswith('response_002_c.html')
    viewer.close()
    viewer.deleteLater()


def test_gallery_is_reachable_from_the_progress_tab_and_the_settings(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    import interactive_images_gui as ixg

    _app()
    out = _gallery_book(tmp_path)
    owner = _owner()
    panel = ixg.ImagesFilePanel(owner, None, str(out))
    assert panel.gallery_btn.text() == '🖼 Gallery'
    panel.gallery_btn.click()
    dialog = owner._interactive_gallery_dialog
    assert dialog.isVisible() and dialog.list.count() == 3
    panel.gallery_btn.click()                                           # the same button closes it
    assert not dialog.isVisible()
    settings = ixg.InteractiveImagesDialog(owner, None, output_dir=str(out))
    settings.gallery_btn.click()
    assert owner._interactive_gallery_dialog.isVisible()
    owner._interactive_gallery_dialog.close()
    panel.deleteLater()
    settings.deleteLater()


def test_gallery_checkbox_click_ticks_without_selecting_and_a_card_click_selects(tmp_path):
    from PySide6.QtCore import QRect, Qt
    from PySide6.QtTest import QTest

    import interactive_images_gui as ixg

    _app()
    out = _gallery_book(tmp_path)
    dialog = ixg.InteractiveImagesGalleryDialog(_owner(), str(out), None, confirm=False)
    dialog.resize(1000, 700)
    dialog.show()
    delegate = dialog.list.itemDelegate()
    assert isinstance(delegate, ixg._CardDelegate)
    item = dialog.list.item(1)
    rect = dialog.list.visualItemRect(item)
    box = delegate.box_rect(type('Opt', (), {'rect': rect})())
    QTest.mouseClick(dialog.list.viewport(), Qt.LeftButton, Qt.NoModifier, box.center())
    assert item.checkState() == Qt.Checked and not item.isSelected()    # ticked, not highlighted
    assert len(dialog.selected_images()) == 1
    QTest.mouseClick(dialog.list.viewport(), Qt.LeftButton, Qt.NoModifier, box.center())
    assert item.checkState() == Qt.Unchecked and dialog.selected_images() == []
    QTest.mouseClick(dialog.list.viewport(), Qt.LeftButton, Qt.NoModifier, rect.center())
    assert item.isSelected() and len(dialog.selected_images()) == 1      # a click on the card highlights it
    QTest.mouseClick(dialog.list.viewport(), Qt.LeftButton, Qt.ControlModifier, dialog.list.visualItemRect(
        dialog.list.item(2)).center())
    assert len(dialog.selected_images()) == 2                            # ctrl-click adds