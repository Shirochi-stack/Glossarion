# -*- coding: utf-8 -*-
"""Relations Tracker desktop UI: General-tab rows, Custom Fields lock, prompt check and the diagram."""
import os
import pathlib
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

import pytest

pytest.importorskip("PySide6")

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtWidgets import QApplication, QLineEdit, QMainWindow, QWidget  # noqa: E402

import relations_tracking as rt  # noqa: E402

OLD_PROMPT = "Extract names.\n- {subject_tracking_instruction}\nNo relations line here."


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    for key, sub in (("GLOSSARION_APP_DIR", "app"), ("GLOSSARION_LIBRARY_DIR", "lib"),
                     ("OUTPUT_DIRECTORY", "out")):
        path = tmp_path / sub
        path.mkdir()
        monkeypatch.setenv(key, str(path))
    for key in (rt.ENV_ENABLE, rt.ENV_HISTORY_MODE, rt.ENV_FIELD):
        monkeypatch.delenv(key, raising=False)
    import review_dialog
    calls = []
    answer = {"value": False}

    def _confirm(parent, title, text):
        calls.append(text)
        return answer["value"]

    monkeypatch.setattr(review_dialog, "confirm_prompt_profile_action", _confirm)
    return calls, answer


def _make_gui(config):
    from GlossaryManager_GUI import GlossaryManagerMixin

    class Harness(GlossaryManagerMixin, QMainWindow):
        def __init__(self):
            super().__init__()
            self.config = config
            self.saves = 0
            self.logs = []
            self._screen = QApplication.primaryScreen().geometry()
            self.dialog = self

        def save_config(self, show_message=False):
            self.saves += 1
            return True

        def append_log(self, text):
            self.logs.append(text)

    gui = Harness()
    gui.pages = {}
    for name in ("_setup_manual_glossary_tab", "_setup_auto_glossary_tab", "_setup_glossary_general_tab"):
        page = QWidget()
        getattr(gui, name)(page)
        gui.pages[name] = page
    return gui


def _old_prompt_config(**extra):
    config = {
        "glossary_prompt_profiles": {"balanced_full": {"Old": OLD_PROMPT}, "minimal": {}},
        "active_glossary_prompt_profiles": {"balanced_full": "Old"},
        "manual_glossary_prompt3": OLD_PROMPT,
        "custom_glossary_fields": ["description"],
    }
    config.update(extra)
    return config


@pytest.fixture
def gui(qapp, isolated):
    window = _make_gui(_old_prompt_config())
    yield window
    window.close()
    window.deleteLater()
    qapp.processEvents()


def _fields(gui):
    box = gui.custom_fields_listbox
    return [box.item(i).text() for i in range(box.count())]


def _relations_item(gui):
    box = gui.custom_fields_listbox
    for i in range(box.count()):
        if box.item(i).data(gui._RELATIONS_FIELD_ROLE):
            return box.item(i)
    return None


def test_toggle_sits_directly_above_skip_gender_tracking(gui):
    layout = gui.pages["_setup_glossary_general_tab"].layout()
    rows = [layout.itemAt(i).widget() for i in range(layout.count())]
    toggle_row = gui.enable_relations_tracker_checkbox.parentWidget()
    gender_row = gui.skip_gender_tracking_checkbox.parentWidget()
    assert gui.enable_relations_tracker_checkbox.text() == "Enable Relations Tracker"
    i = rows.index(toggle_row)
    # toggle, then its indented options and the warning row, then Skip Gender Tracking
    assert rows[i + 1] is gui.relations_history_mode_combo.parentWidget()
    assert rows[i + 2] is gui.relations_warning_widget
    assert rows[i + 3] is gender_row
    assert rows[i - 1] is gui.save_glossary_in_output_checkbox.parentWidget()
    assert [gui.relations_history_mode_combo.itemText(k) for k in range(3)] == [
        "Previous one only", "Full history", "Current only"]
    assert not gui.enable_relations_tracker_checkbox.isChecked()
    assert not gui.relations_history_mode_combo.isEnabled() and not gui.relations_diagram_btn.isEnabled()
    assert gui.relations_warning_widget.isHidden()
    assert "Halgakos.ico" in gui.relations_history_mode_combo.styleSheet()


def test_toggle_adds_and_removes_the_amber_locked_field_live(gui, isolated):
    gui.enable_relations_tracker_checkbox.setChecked(True)
    assert _fields(gui) == ["description", "relations"]
    assert gui.custom_glossary_fields == ["description", "relations"]
    assert gui.config["custom_glossary_fields"] == ["description", "relations"]
    assert gui.config["glossary_enable_relations_tracker"] is True
    assert os.environ[rt.ENV_ENABLE] == "1" and os.environ[rt.ENV_FIELD] == "relations"
    assert os.environ[rt.ENV_HISTORY_MODE] == "previous"
    assert gui.relations_history_mode_combo.isEnabled() and gui.relations_diagram_btn.isEnabled()
    item = _relations_item(gui)
    assert item is not None and item.text() == "relations"
    assert item.background().color().name() in ("#7a5200", "#ffe08a")
    assert item.foreground().color().name() in ("#ffe7a3", "#5c3d00")
    plain = gui.custom_fields_listbox.item(0)
    assert not (plain.flags() & item.flags().ItemIsEditable)
    assert gui.saves >= 1

    gui.relations_history_mode_combo.setCurrentIndex(1)
    assert gui.config["glossary_relations_history_mode"] == "full" and os.environ[rt.ENV_HISTORY_MODE] == "full"

    gui.enable_relations_tracker_checkbox.setChecked(False)
    assert _fields(gui) == ["description"]
    assert gui.config["custom_glossary_fields"] == ["description"]
    assert os.environ[rt.ENV_ENABLE] == "0"
    assert gui.relations_warning_widget.isHidden()


def test_remove_is_refused_and_add_does_not_duplicate(gui):
    gui.enable_relations_tracker_checkbox.setChecked(True)
    box = gui.custom_fields_listbox
    box.setCurrentRow(1)
    gui.custom_fields_remove_btn.click()
    assert _fields(gui) == ["description", "relations"]
    assert "stays while Enable Relations Tracker is on" in gui.custom_fields_status_label.text()
    gui.custom_field_entry.setText("Relations")
    gui.custom_fields_add_btn.click()
    assert _fields(gui) == ["description", "relations"]
    # other fields still remove normally
    box.setCurrentRow(0)
    gui.custom_fields_remove_btn.click()
    assert _fields(gui) == ["relations"]


def test_rename_updates_config_env_and_rejects_bad_names(gui):
    gui.enable_relations_tracker_checkbox.setChecked(True)
    item = _relations_item(gui)
    item.setText("bonds")
    assert gui.config["glossary_relations_field_name"] == "bonds"
    assert os.environ[rt.ENV_FIELD] == "bonds"
    assert _fields(gui) == ["description", "bonds"]
    assert gui.config["custom_glossary_fields"] == ["description", "bonds"]
    item = _relations_item(gui)
    item.setText("  ")
    assert _fields(gui) == ["description", "bonds"]
    _relations_item(gui).setText("Description")
    assert _fields(gui) == ["description", "bonds"]
    assert "built-in glossary column" in gui.custom_fields_status_label.text()
    gui.custom_glossary_fields = ["description", "notes", "bonds"]
    gui._reload_custom_fields_listbox()
    _relations_item(gui).setText("Notes")
    assert _fields(gui) == ["description", "notes", "bonds"]
    assert "already a custom field" in gui.custom_fields_status_label.text()
    gui.custom_glossary_fields = ["description", "bonds"]
    gui._reload_custom_fields_listbox()
    assert gui.config["glossary_relations_field_name"] == "bonds"
    gui.enable_relations_tracker_checkbox.setChecked(False)
    assert _fields(gui) == ["description"]


@pytest.mark.parametrize("name", ["gender", "raw_name", "translated_name", "type", "description"])
def test_rename_to_a_glossary_column_is_refused(gui, name):
    gui.enable_relations_tracker_checkbox.setChecked(True)
    gui.custom_glossary_fields = ["relations"]  # description removed from Custom Fields
    gui._reload_custom_fields_listbox()
    _relations_item(gui).setText(name)
    assert _fields(gui) == ["relations"]
    assert "built-in glossary column" in gui.custom_fields_status_label.text()
    assert gui.config.get("glossary_relations_field_name") in (None, "relations")
    assert os.environ[rt.ENV_FIELD] == "relations"


def test_enabling_keeps_a_users_own_relations_column(qapp, isolated):
    window = _make_gui(_old_prompt_config(custom_glossary_fields=["description", "relations"]))
    try:
        window.enable_relations_tracker_checkbox.setChecked(True)
        assert _fields(window) == ["description", "relations", "relations_tracker"]
        assert window.config["glossary_relations_field_name"] == "relations_tracker"
        assert os.environ[rt.ENV_FIELD] == "relations_tracker"
        assert _relations_item(window).text() == "relations_tracker"
        window.enable_relations_tracker_checkbox.setChecked(False)
        assert _fields(window) == ["description", "relations"]
    finally:
        window.close()
        window.deleteLater()


def test_missing_placeholder_no_shows_copy_row(gui, isolated, qapp):
    calls, answer = isolated
    answer["value"] = False
    gui.enable_relations_tracker_checkbox.setChecked(True)
    assert len(calls) == 1 and "Balanced/Full" in calls[0]
    assert not gui.relations_warning_widget.isHidden()
    assert "Balanced/Full" in gui.relations_warning_label.text()
    line = gui.relations_placeholder_line
    assert isinstance(line, QLineEdit) and line.isReadOnly()
    assert line.text() == "- {relations_instruction}"
    gui.relations_copy_btn.click()
    assert QApplication.clipboard().text() == "- {relations_instruction}"
    # Adding the line to the prompt clears the warning; removing it brings it back.
    gui.manual_prompt_text.setPlainText(OLD_PROMPT + "\n- {relations_instruction}")
    assert gui.relations_warning_widget.isHidden()
    gui.manual_prompt_text.setPlainText(OLD_PROMPT)
    assert not gui.relations_warning_widget.isHidden()
    # Switching to the Default profile (signals blocked) re-checks too.
    combo = gui._glossary_prompt_profile_widgets["balanced_full"]["combo"]
    combo.setCurrentIndex(combo.findText("Default"))
    expected = rt.RELATIONS_PLACEHOLDER in gui._default_glossary_prompt_profile_text("balanced_full")
    assert gui.relations_warning_widget.isHidden() is expected
    gui.enable_relations_tracker_checkbox.setChecked(False)
    assert gui.relations_warning_widget.isHidden()


def test_missing_placeholder_yes_resets_the_prompt(gui, isolated):
    calls, answer = isolated
    answer["value"] = True
    gui.enable_relations_tracker_checkbox.setChecked(True)
    assert len(calls) == 1
    text = gui._sep_from_display(gui.manual_prompt_text.toPlainText())
    assert text == gui.default_manual_glossary_prompt
    # The reset text is stored into the active profile like a manual Reset to Default.
    assert gui.config["glossary_prompt_profiles"]["balanced_full"]["Old"] == gui.default_manual_glossary_prompt
    missing = rt.RELATIONS_PLACEHOLDER not in gui.default_manual_glossary_prompt
    assert gui.relations_warning_widget.isHidden() is not missing


def test_on_device_profile_is_not_offered_a_reset(qapp, isolated):
    calls, answer = isolated
    answer["value"] = True
    import glossary_document
    on_device = glossary_document.ON_DEVICE_GLOSSARY_PROMPTS
    # The current prompt keys hold the active profile's text, as the profile saver keeps them.
    window = _make_gui(_old_prompt_config(
        glossary_prompt_profiles={"balanced_full": {}, "minimal": {}},
        active_glossary_prompt_profiles={"balanced_full": "On-device", "minimal": "On-device"},
        manual_glossary_prompt3=on_device["balanced_full"],
        unified_auto_glosary_prompt3=on_device["minimal"]))
    try:
        window.enable_relations_tracker_checkbox.setChecked(True)
        assert calls == []
        assert not window.relations_warning_widget.isHidden()
        assert "Balanced/Full and Minimal" in window.relations_warning_label.text()
        assert window.relations_warning_label.text().endswith("Add this line to them:")
    finally:
        window.close()
        window.deleteLater()


def test_placeholder_rows_and_tooltips_on_both_tabs(gui):
    for name in ("_setup_manual_glossary_tab", "_setup_auto_glossary_tab"):
        lines = [w for w in gui.pages[name].findChildren(QLineEdit)
                 if w.text().startswith("Available placeholders")]
        assert len(lines) == 1, name
        assert "{relations_instruction}" in lines[0].text()
        tip = lines[0].toolTip()
        assert "{relations_instruction}" in tip
        assert "Active when Enable Relations Tracker is ON; stripped otherwise" in tip


def _synthetic_tracker():
    tracker = rt.empty_tracker()
    rows = [
        (1, "서윤", "Seo-yun", "도현", "Older Brother"), (1, "도현", "Do-hyeon", "서윤", "Younger Sister"),
        (1, "서윤", "Seo-yun", "하린", "Best Friend"), (1, "하린", "Ha-rin", "서윤", "Best Friend"),
        (3, "도현", "Do-hyeon", "강우", "Rival"), (6, "무진", "Mu-jin", "서윤", "Daughter"),
        (40, "서윤", "Seo-yun", "하린", "Enemy"),
    ]
    # Each tuple is the stored meaning (label = what target is to raw); the model writes
    # it on the target's row as "raw: label".
    for chapter, raw, translated, target, label in rows:
        rt.record_relations(tracker, [{"raw_name": target, "relations": f"{raw}: {label}"},
                                      {"raw_name": raw, "translated_name": translated}],
                            field="relations", chapter_index=chapter - 1, chapter_num=chapter,
                            chapter_file=f"ch{chapter:03d}.xhtml")
    return tracker


def test_diagram_builds_and_slider_changes_edges(qapp):
    from relations_diagram_dialog import RelationsDiagramDialog, edge_label_text
    entries = [{"raw_name": "서윤", "translated_name": "Seo-yun", "gender": "female"},
               {"raw_name": "도현", "translated_name": "Do-hyeon", "gender": "male"},
               {"raw_name": "하린", "translated_name": "Ha-rin", "gender": "female"},
               {"raw_name": "무진", "translated_name": "Mu-jin", "gender": "male"},
               {"raw_name": "강우", "translated_name": "Kang-woo", "gender": "male"}]
    dialog = RelationsDiagramDialog(None, tracker=_synthetic_tracker(), glossary_entries=entries)
    try:
        assert dialog.chapter_slider.minimum() == 1 and dialog.chapter_slider.maximum() == 40
        assert dialog.chapter_label.text() == "Up to chapter 40"
        full = dialog.edge_count()
        assert full == 4 and dialog.node_count() == 5
        dialog.chapter_slider.setValue(1)
        assert dialog.edge_count() == 2
        assert dialog.chapter_label.text() == "Up to chapter 1"
        dialog.chapter_slider.setValue(40)
        assert dialog.edge_count() == full
        # focus places a character in the centre (radial layout) without changing the edges
        dialog.focus_combo.setCurrentIndex(dialog.focus_combo.findData(rt.tracker_key("서윤")))
        assert dialog.edge_count() == full
        history = dialog.show_history(rt.tracker_key("서윤"))
        assert any(h["current"] == "Enemy" for h in history)
        assert "Enemy" in dialog.history_panel.toPlainText()
        assert edge_label_text({"label_ab": "Enemy", "former_ab": ["Best Friend"], "label_ba": "Father"},
                               show_former=True) == "Enemy [Best Friend] / Father"
    finally:
        dialog.close()
        dialog.deleteLater()


def _team_tracker():
    """A leader, a team whose members are all related to each other, and five subordinates."""
    tracker = rt.empty_tracker()
    team = ["민", "성규", "춘후", "준호", "민주"]
    rows = []
    for i, name in enumerate(team):
        cell = "; ".join(f"{other}: {'Rival' if (i + j) % 2 else 'Teammate'}"
                         for j, other in enumerate(team) if other != name)
        rows.append({"raw_name": name, "translated_name": f"T{i}", "relations": cell})
    for i in range(5):
        rows.append({"raw_name": f"부하{i}", "translated_name": f"S{i}", "relations": "민: Subordinate"})
    rows[0]["relations"] += "; " + "; ".join(f"부하{i}: Team Leader" for i in range(5))
    rt.record_relations(tracker, rows, field="relations", chapter_index=0, chapter_num=1, chapter_file="c1.xhtml")
    return tracker


def test_diagram_is_tidy_and_fitted(qapp):
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QLabel
    from relations_diagram_dialog import DIMMED_OPACITY, HISTORY_HINT, RelationsDiagramDialog
    dialog = RelationsDiagramDialog(None, tracker=_team_tracker())
    try:
        dialog.resize(1300, 850)
        dialog.show()
        for _ in range(4):
            qapp.processEvents()
        view = dialog._view
        leader = rt.tracker_key("민")
        assert dialog.node_count() == 10 and dialog.edge_count() == 15
        # The same relation from both sides is one word, and every label sits on its own
        # line clear of the names and of the other labels.
        texts = [i.text() for i in dialog.scene.items() if hasattr(i, "text") and i.parentItem() is None]
        assert "Teammate" in texts and "Teammate / Teammate" not in texts
        boxes = dialog._label_boxes
        assert len(boxes) == 15
        names = [item.sceneBoundingRect() for item in dialog._node_items.values()]
        # A label stays ON its line and slides along it past whatever already sits there.
        start, end = QPointF(0, 5000), QPointF(4000, 5000)
        edge = {"a": leader, "b": rt.tracker_key("성규"), "label_ab": "Rival", "label_ba": "Rival"}
        taken = [QRectF(1700, 4900, 600, 200)]  # covers the middle of the line
        dialog._add_edge_label(edge, start, end, QColor("#8b95a5"), False, taken)
        box = taken[-1]
        assert box is dialog._label_boxes[-1] and not box.intersects(taken[0])
        assert abs(box.center().y() - 5000) < 0.01 and 0 < box.center().x() < 4000
        dialog.rebuild()
        # Opened fitted: every name is inside the window, with no scroll bars around it.
        visible = view.mapToScene(view.viewport().rect()).boundingRect()
        assert all(visible.contains(name) for name in names)
        assert view.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
        # The two switches are the app's checkboxes with the drawn tick.
        for box in (dialog.show_former_checkbox, dialog.show_unlisted_checkbox):
            assert [w.text() for w in box.findChildren(QLabel)] == ["✓"]
            assert "QCheckBox::indicator:checked" in box.styleSheet()

        # A click on a name keeps its lines and labels and fades the rest back.
        dialog._on_node_clicked(leader)
        lines = [i for i in dialog.scene.items() if hasattr(i, "line")]
        assert sorted({round(i.opacity(), 2) for i in lines}) == [DIMMED_OPACITY, 1.0]
        assert sum(1 for i in lines if i.opacity() == 1.0) == 9 and len(dialog._label_boxes) == 9
        assert "T0" in dialog.history_title.text()
        # Empty space (or the same name again) brings the whole picture back.
        dialog.clear_selection()
        assert dialog._selected is None and len(dialog._label_boxes) == 15
        assert all(i.opacity() == 1.0 for i in dialog.scene.items())
        assert dialog.history_title.text() == HISTORY_HINT
        dialog._on_node_clicked(leader)
        dialog._on_node_clicked(leader)
        assert dialog._selected is None
    finally:
        dialog.close()
        dialog.deleteLater()


def test_diagram_panel_shows_the_description_written_at_the_sliders_chapter(qapp):
    from relations_diagram_dialog import RelationsDiagramDialog
    tracker = _synthetic_tracker()  # chapters 1 to 40
    entries = [{"raw_name": "서윤", "translated_name": "Seo-yun", "gender": "female",
                "description": "The glossary's present text"},
               {"raw_name": "도현", "translated_name": "Do-hyeon", "gender": "male", "description": "Her brother"},
               {"raw_name": "하린", "translated_name": "Ha-rin", "gender": "female"},
               {"raw_name": "무진", "translated_name": "Mu-jin", "gender": "male"},
               {"raw_name": "강우", "translated_name": "Kang-woo", "gender": "male"}]
    for chapter, text in ((1, "The heir of the Baek house"), (40, "The heir, betrayed by <Ha-rin>")):
        rt.record_relations(tracker, [{"type": "character", "raw_name": "서윤", "description": text}],
                            field="relations", chapter_index=chapter - 1, chapter_num=chapter,
                            chapter_file=f"ch{chapter:03d}.xhtml")
    dialog = RelationsDiagramDialog(None, tracker=tracker, glossary_entries=entries)
    try:
        seo, brother, harin = (rt.tracker_key(n) for n in ("서윤", "도현", "하린"))
        dialog._on_node_clicked(seo)
        text = dialog.history_panel.toPlainText()
        assert "Description written at chapter 40" in text and "The heir, betrayed by <Ha-rin>" in text
        assert "Earlier descriptions" in text and "Ch. 1: The heir of the Baek house" in text
        assert "The glossary's present text" not in text
        assert "Enemy" in text  # the relation history still follows
        # Moving the slider shows what was written by then, with the panel left open.
        dialog.chapter_slider.setValue(20)
        text = dialog.history_panel.toPlainText()
        assert "Description written at chapter 1" in text and "The heir of the Baek house" in text
        assert "betrayed" not in text and "Earlier descriptions" not in text
        assert "The heir of the Baek house" in dialog._node_items[seo].toolTip()
        # The relation history stops at the slider's chapter too: the chapter-40 change is not there yet.
        assert "Best Friend" in text and "Enemy" not in text and "Ch. 40" not in text
        dialog.chapter_slider.setValue(40)
        assert "Enemy" in dialog.history_panel.toPlainText()
        dialog.chapter_slider.setValue(20)
        # No description recorded for him: the glossary's own, said to be undated.
        dialog._on_node_clicked(brother)
        text = dialog.history_panel.toPlainText()
        assert "Her brother" in text and "not recorded per chapter" in text
        # None anywhere: the panel is the relation history alone.
        dialog._on_node_clicked(harin)
        assert "Description" not in dialog.history_panel.toPlainText()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_diagram_uses_the_configured_manual_glossary(gui, tmp_path):
    glossary = tmp_path / "Book_glossary.csv"
    glossary.write_text("type,raw_name,translated_name\n", encoding="utf-8")
    gui.manual_glossary_path = None
    gui.config["manual_glossary_path"] = str(glossary)
    assert gui._relations_glossary_path() == str(glossary)


def test_diagram_click_on_a_large_graph_rebuilds_after_the_press(qapp):
    """Over MAX_LABELLED_EDGES a click rebuilds the scene; it must not run inside the press handler."""
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from relations_diagram_dialog import MAX_LABELLED_EDGES, RelationsDiagramDialog
    tracker = rt.empty_tracker()
    names = [f"인물{i:02d}" for i in range(18)]  # 153 pairs
    rows = [{"raw_name": n, "translated_name": f"P{i}",
             "relations": "; ".join(f"{o}: Friend" for o in names if o != n)} for i, n in enumerate(names)]
    rt.record_relations(tracker, rows, field="relations", chapter_index=0, chapter_num=1, chapter_file="c1.xhtml")
    dialog = RelationsDiagramDialog(None, tracker=tracker)
    try:
        dialog.resize(1200, 780)
        dialog.show()
        qapp.processEvents()
        assert dialog.edge_count() > MAX_LABELLED_EDGES
        key = rt.tracker_key(names[0])
        item = dialog._node_items[key]
        view = item.scene().views()[0]
        view.centerOn(item)
        qapp.processEvents()
        point = view.mapFromScene(item.sceneBoundingRect().center())
        QTest.mousePress(view.viewport(), Qt.LeftButton, Qt.NoModifier, point)
        assert dialog._node_items[key] is item  # nothing rebuilt inside the press handler
        qapp.processEvents()
        assert dialog._selected == key and dialog._node_items[key] is not item
        QTest.mouseMove(view.viewport(), point + QPoint(5, 5))
        QTest.mouseRelease(view.viewport(), Qt.LeftButton, Qt.NoModifier, point)
        qapp.processEvents()
        assert names[0] in dialog.history_title.text() or "P0" in dialog.history_title.text()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_diagram_empty_state(qapp, tmp_path):
    from relations_diagram_dialog import EMPTY_TEXT, RelationsDiagramDialog
    glossary = tmp_path / "Book_glossary.csv"
    glossary.write_text("type,raw_name,translated_name\ncharacter,서윤,Seo-yun\n", encoding="utf-8")
    dialog = RelationsDiagramDialog(None, glossary_path=str(glossary))
    try:
        assert dialog._stack.currentWidget() is dialog.empty_label
        assert dialog.empty_label.text() == EMPTY_TEXT
        assert dialog.edge_count() == 0
    finally:
        dialog.close()
        dialog.deleteLater()


def test_save_settings_and_reopen_wiring_cover_the_toggle():
    source = (SRC / "GlossaryManager_GUI.py").read_text(encoding="utf-8-sig")
    assert "('enable_relations_tracker_checkbox', 'glossary_enable_relations_tracker_var')," in source
    assert "('enable_relations_tracker_checkbox', 'glossary_enable_relations_tracker', False)," in source
    assert "('🔗 Relations', lambda: self._open_relations_diagram(" in source.replace('"', "'")
    diagram = (SRC / "relations_diagram_dialog.py").read_text(encoding="utf-8")
    assert "manga_image_preview" not in diagram
