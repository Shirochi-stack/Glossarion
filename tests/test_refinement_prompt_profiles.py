"""Refinement prompt pairs through the real tab and runtime combo helpers."""

import copy

import pytest
from PySide6.QtCore import QEvent, QObject, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QComboBox, QDialog, QMessageBox, QPushButton

from test_prompt_profile_renaming import gui, qapp, saved


PROFILES = 'glossary_refinement_prompt_profiles'
DEFAULT = 'glossary_refinement_prompt_profile_default'
ACTIVE = 'active_glossary_refinement_prompt_profile'
SYSTEM = 'glossary_refinement_system_prompt'
USER = 'glossary_refinement_user_prompt'


def open_tab(gui):
    tab = QDialog(gui)
    gui._setup_glossary_refinement_tab(tab)
    combo = tab.findChild(QComboBox, 'refinement_prompt_profile_combo')
    assert combo is not None
    buttons = {b.text(): b for b in combo.parentWidget().findChildren(QPushButton)}
    return tab, combo, buttons


@pytest.fixture
def refinement(gui):
    gui.config.update({
        'glossary_refinement_enabled': True,
        SYSTEM: 'Legacy system {fields}',
        USER: 'Legacy user {chunk_index}',
    })
    return open_tab(gui)


def edit_pair(gui, system, user):
    gui.glossary_refinement_system_prompt_text.setPlainText(gui._sep_for_display(system))
    gui.glossary_refinement_user_prompt_text.setPlainText(gui._sep_for_display(user))


def current_pair(gui):
    return {
        'system': gui._glossary_prompt_text(gui.glossary_refinement_system_prompt_text),
        'user': gui._glossary_prompt_text(gui.glossary_refinement_user_prompt_text),
    }


def create_named(gui, combo, buttons, name, system, user):
    buttons['+ New Profile'].click()
    edit_pair(gui, system, user)
    combo.setEditText(name)
    buttons['💾 Save Profile'].click()


def test_legacy_pair_becomes_default_and_round_trips(gui, refinement):
    _, combo, buttons = refinement
    assert combo.currentText() == 'Default'
    assert current_pair(gui) == {'system': 'Legacy system {fields}', 'user': 'Legacy user {chunk_index}'}
    buttons['💾 Save Profile'].click()
    config = saved(gui)
    assert config[DEFAULT] == current_pair(gui)
    assert config[PROFILES] == {}
    assert config[ACTIVE] == ''
    assert config['unrelated_setting'] == 'preserved'
    gui.config = config
    _, reopened_combo, _ = open_tab(gui)
    assert reopened_combo.currentText() == 'Default'
    assert current_pair(gui) == config[DEFAULT]


def test_new_save_rename_keeps_pair_and_order_without_duplicates(gui, refinement):
    _, combo, buttons = refinement
    create_named(gui, combo, buttons, 'Alpha', 'System A {fields1}\x1f{entries}', 'User A {columns}')
    create_named(gui, combo, buttons, 'Beta', 'System B', '')
    combo.setCurrentIndex(combo.findText('Alpha'))
    combo.setEditText('Renamed')
    edit_pair(gui, 'Updated {fields1}\x1f{entries}', 'Updated {chunk_index}/{total_chunks}')
    buttons['💾 Save Profile'].click()
    config = saved(gui)
    assert list(config[PROFILES]) == ['Renamed', 'Beta']
    assert config[ACTIVE] == 'Renamed'
    assert config[PROFILES]['Renamed'] == current_pair(gui)
    assert config[SYSTEM] == current_pair(gui)['system']
    assert config[USER] == current_pair(gui)['user']
    assert config[DEFAULT]['system'] == 'Legacy system {fields}'
    gui.config = config
    _, reopened_combo, _ = open_tab(gui)
    assert reopened_combo.currentText() == 'Renamed'
    assert current_pair(gui) == config[PROFILES]['Renamed']
    reopened_combo.setCurrentIndex(reopened_combo.findText('Beta'))
    assert current_pair(gui) == {'system': 'System B', 'user': ''}


def test_switching_retains_unsaved_edits_to_both_prompts(gui, refinement):
    _, combo, buttons = refinement
    create_named(gui, combo, buttons, 'Alpha', 'System A', 'User A')
    edit_pair(gui, 'Draft system', 'Draft user')
    combo.setCurrentIndex(0)
    assert current_pair(gui) == gui.config[DEFAULT]
    combo.setCurrentIndex(combo.findText('Alpha'))
    assert current_pair(gui) == {'system': 'Draft system', 'user': 'Draft user'}


@pytest.mark.parametrize('name', ['Default', 'default', 'Beta', ' '])
def test_rename_rejects_reserved_empty_and_existing_names(gui, refinement, name):
    _, combo, buttons = refinement
    create_named(gui, combo, buttons, 'Alpha', 'System A', 'User A')
    create_named(gui, combo, buttons, 'Beta', 'System B', 'User B')
    combo.setCurrentIndex(combo.findText('Alpha'))
    before = copy.deepcopy(gui.config[PROFILES])
    combo.setEditText(name)
    buttons['💾 Save Profile'].click()
    assert gui.config[PROFILES] == before
    assert gui.config[ACTIVE] == 'Alpha'


def test_default_can_be_copied_to_named_profile(gui, refinement):
    _, combo, buttons = refinement
    default = copy.deepcopy(gui.config[DEFAULT])
    combo.setEditText('Custom')
    edit_pair(gui, 'Custom system', 'Custom user')
    buttons['💾 Save Profile'].click()
    assert saved(gui)[PROFILES]['Custom'] == current_pair(gui)
    assert saved(gui)[DEFAULT] == default


def test_delete_protects_default_and_preserves_other_pair(gui, refinement, monkeypatch):
    _, combo, buttons = refinement
    monkeypatch.setattr(QMessageBox, 'exec', lambda *_: QMessageBox.Yes)
    create_named(gui, combo, buttons, 'Alpha', 'System A', 'User A')
    create_named(gui, combo, buttons, 'Beta', 'System B', 'User B')
    buttons['🗑 Delete Profile'].click()
    assert combo.currentText() == 'Alpha'
    assert current_pair(gui) == {'system': 'System A', 'user': 'User A'}
    assert list(saved(gui)[PROFILES]) == ['Alpha']
    buttons['🗑 Delete Profile'].click()
    assert combo.currentText() == 'Default'
    assert saved(gui)[PROFILES] == {}
    assert current_pair(gui) == saved(gui)[DEFAULT]
    before = copy.deepcopy(gui.config)
    buttons['🗑 Delete Profile'].click()
    assert gui.config == before


@pytest.mark.parametrize('action', ['+ New Profile', '💾 Save Profile', '🗑 Delete Profile'])
def test_failed_persistence_restores_profile_and_editors(gui, refinement, monkeypatch, action):
    _, combo, buttons = refinement
    create_named(gui, combo, buttons, 'Alpha', 'System A', 'User A')
    if action == '💾 Save Profile':
        combo.setEditText('Renamed')
    before = copy.deepcopy(gui.config)
    pair = current_pair(gui)
    name = combo.currentText()
    monkeypatch.setattr(gui, 'save_config', lambda **_: False)
    monkeypatch.setattr(QMessageBox, 'exec', lambda *_: QMessageBox.Yes)
    buttons[action].click()
    assert gui.config == before
    assert current_pair(gui) == pair
    assert combo.currentText() == name
    assert gui.glossary_refinement_system_prompt == pair['system']
    assert gui.glossary_refinement_user_prompt == pair['user']


def test_reset_updates_selected_profile_only(gui, refinement, monkeypatch):
    tab, combo, buttons = refinement
    default = copy.deepcopy(gui.config[DEFAULT])
    create_named(gui, combo, buttons, 'Alpha', 'System A', 'User A')
    monkeypatch.setattr(QMessageBox, 'question', lambda *_: QMessageBox.Yes)
    next(b for b in tab.findChildren(QPushButton) if b.text() == 'Reset to Default').click()
    assert gui.config[PROFILES]['Alpha'] == {
        'system': gui._default_glossary_refinement_system_prompt(),
        'user': gui._default_glossary_refinement_user_prompt(),
    }
    assert gui.config[DEFAULT] == default


def test_popup_selects_default_and_custom_with_one_click(gui, refinement, qapp):
    class SuppressRelease(QObject):
        def eventFilter(self, watched, event):
            return event.type() == QEvent.MouseButtonRelease

    tab, combo, buttons = refinement
    create_named(gui, combo, buttons, 'Alpha', 'System A', 'User A')
    edit_pair(gui, 'Draft system', 'Draft user')
    tab.show()
    qapp.processEvents()
    release_filter = SuppressRelease(combo)
    combo.view().viewport().installEventFilter(release_filter)
    for name in ('Default', 'Alpha', 'Default', 'Alpha'):
        QTest.mouseClick(combo, Qt.LeftButton, pos=QPoint(combo.width() - 12, combo.height() // 2))
        qapp.processEvents()
        view = combo.view()
        assert view.isVisible()
        index = combo.model().index(combo.findText(name), 0)
        QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=view.visualRect(index).center())
        qapp.processEvents()
        assert combo.currentText() == name
        expected = gui.config[DEFAULT] if name == 'Default' else {'system': 'Draft system', 'user': 'Draft user'}
        assert current_pair(gui) == expected
