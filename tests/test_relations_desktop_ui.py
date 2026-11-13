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
    for key in (rt.ENV_ENABLE, rt.ENV_HISTORY_MODE, rt.ENV_FIELD, rt.ENV_MINIMAL):
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
        unified_auto_glosary_prompt3=on_device["minimal"],
        glossary_minimal_include_relations=True))
    try:
        window.enable_relations_tracker_checkbox.setChecked(True)
        assert calls == []
        assert not window.relations_warning_widget.isHidden()
        assert "Balanced/Full and Minimal" in window.relations_warning_label.text()
        assert window.relations_warning_label.text().endswith("Add this line to them:")
    finally:
        window.close()
        window.deleteLater()


def test_fresh_config_has_both_toggles_on_and_asks_no_prompt_question(qapp, isolated):
    """A fresh desktop config (translator_gui marks it; relations_tracking.apply_fresh_defaults) shows both
    toggles on with the amber relations column, and its built-in prompts need no reset."""
    calls, answer = isolated
    answer["value"] = True
    config = {rt.FRESH_DEFAULTS_KEY: list(rt.FRESH_DEFAULT_KEYS)}
    rt.apply_fresh_defaults(config)
    window = _make_gui(config)
    try:
        assert window.enable_relations_tracker_checkbox.isChecked()
        assert window.minimal_include_relations_checkbox.isChecked()
        assert window.config["custom_glossary_fields"] == ["description", "relations"]
        assert window._check_relations_prompts() == [] and calls == []
        assert window.relations_warning_widget.isHidden()
    finally:
        window.close()
        window.deleteLater()


def test_minimal_include_relations_checkbox(qapp, isolated):
    """Minimal tab: "Include Relations Column" right under "Include Description Column", off by default;
    it writes config + env at once and, turned on with the tracker on, checks the Minimal prompt."""
    calls, answer = isolated
    answer["value"] = False
    window = _make_gui(_old_prompt_config())
    try:
        box = window.minimal_include_relations_checkbox
        assert box.text() == "Include Relations Column" and not box.isChecked()
        assert "Enable Relations Tracker" in box.toolTip()
        description_row = window.include_description_checkbox.parentWidget()
        relations_row = box.parentWidget()
        layout = description_row.parentWidget().layout()
        assert relations_row.parentWidget() is description_row.parentWidget()
        assert layout.indexOf(relations_row) == layout.indexOf(description_row) + 1

        window.auto_prompt_text.setPlainText("Minimal prompt without the relations line")
        saves = window.saves
        box.setChecked(True)  # tracker off: stored, no prompt question
        assert window.config[rt.CONFIG_MINIMAL] is True and os.environ[rt.ENV_MINIMAL] == "1"
        assert window.saves == saves + 1 and calls == []
        box.setChecked(False)
        assert window.config[rt.CONFIG_MINIMAL] is False and os.environ[rt.ENV_MINIMAL] == "0"

        window.enable_relations_tracker_checkbox.setChecked(True)  # Minimal off: only Balanced/Full is asked about
        assert len(calls) == 1 and "Balanced/Full prompt" in calls[0] and "Minimal" not in calls[0]
        box.setChecked(True)  # on with the tracker on: the Minimal prompt is checked too
        assert len(calls) == 2 and "Minimal" in calls[1]
        assert "Minimal" in window.relations_warning_label.text()
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


def _team_rows():
    """The model's chapter-1 rows: a leader, a team all related to each other, five subordinates."""
    team = ["민", "성규", "춘후", "준호", "민주"]
    rows = []
    for i, name in enumerate(team):
        cell = "; ".join(f"{other}: {'Rival' if (i + j) % 2 else 'Teammate'}"
                         for j, other in enumerate(team) if other != name)
        rows.append({"raw_name": name, "translated_name": f"T{i}", "relations": cell})
    for i in range(5):
        rows.append({"raw_name": f"부하{i}", "translated_name": f"S{i}", "relations": "민: Subordinate"})
    rows[0]["relations"] += "; " + "; ".join(f"부하{i}: Team Leader" for i in range(5))
    return rows


def _record_chapter_1(tracker):
    """What a re-extraction of chapter 1 records (the AI's original wording)."""
    return rt.record_relations(tracker, _team_rows(), field="relations", chapter_index=0, chapter_num=1,
                               chapter_file="c1.xhtml")


def _team_tracker():
    """A leader, a team whose members are all related to each other, and five subordinates."""
    tracker = rt.empty_tracker()
    _record_chapter_1(tracker)
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


def test_a_name_in_the_side_panel_jumps_to_it_in_the_diagram(qapp):
    from PySide6.QtCore import QUrl
    from relations_diagram_dialog import CHANGED_COLOR, RelationsDiagramDialog
    dialog = RelationsDiagramDialog(None, tracker=_team_tracker())
    try:
        dialog.resize(1300, 850)
        dialog.show()
        for _ in range(4):
            qapp.processEvents()
        view = dialog._view
        leader = rt.tracker_key("민")
        dialog._on_node_clicked(leader)
        # Every name listed is drawn, so every one is a link, in the order shown.
        html_text = dialog.history_panel.toHtml()
        assert len(dialog._history_links) == 9 and html_text.count('href="node:') == 9
        target = dialog._history_links[3]

        def in_sight(node):
            return view.mapToScene(view.viewport().rect()).boundingRect().contains(node.sceneBoundingRect())

        # The owner's "clicking the name is not updating the right side column": the name is
        # selected (panel, relations, outline); the whole diagram is in sight, so nothing moves.
        before = view.transform().m11(), view.horizontalScrollBar().value(), view.verticalScrollBar().value()
        dialog._on_history_link(QUrl("node:3"))
        item = dialog._node_items[target]
        assert dialog._selected == target
        assert dialog._name(target) in dialog.history_title.text() and "T0" not in dialog.history_title.text()
        assert item.pen().color() == CHANGED_COLOR and item.pen().widthF() > 4
        assert (view.transform().m11(), view.horizontalScrollBar().value(), view.verticalScrollBar().value()) == before
        # Its own relations are the ones drawn bright now; the panel lists its relations.
        assert dialog.history_title.text() and "T0" in dialog.history_panel.toPlainText()
        dialog._end_flash(target, item)
        assert item.pen().color().name() == "#ffffff" and item.pen().widthF() < 4  # back to the selected look
        dialog._on_node_clicked(leader)
        item = dialog._node_items[target]

        # Zoomed in on another name: the jump brings the clicked one into view.
        far = max(dialog._history_links, key=lambda key: (dialog._node_items[key].pos() - item.pos()).manhattanLength())
        view._auto_fit = False
        view.scale(4.0, 4.0)
        view.centerOn(dialog._node_items[far])
        qapp.processEvents()
        assert not in_sight(item)
        assert dialog.jump_to(target) is True
        qapp.processEvents()
        assert in_sight(item) and view.transform().m11() > 3  # moved to it, zoom kept
        # The selected name's own outline comes back after the flash.
        dialog.jump_to(leader)
        dialog._end_flash(leader, dialog._node_items[leader])
        assert dialog._node_items[leader].pen().color().name() == "#ffffff"
        # A link that no longer points anywhere does nothing.
        dialog._on_history_link(QUrl("node:99"))
        dialog._on_history_link(QUrl("https://example.invalid/"))
        assert dialog.jump_to("nobody") is False
        # A flash that outlives its item (the scene was rebuilt) is dropped.
        dialog.jump_to(target)
        dialog.rebuild()
        dialog._end_flash(target, item)
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


def _write_book(folder, tracker, extra_names=()):
    """A glossary file with a relations tracker beside it."""
    import json
    glossary = folder / "Book_glossary.csv"
    names = sorted({str(item.get("raw_name") or key) for key, item in tracker["entries"].items()} | set(extra_names))
    glossary.write_text("type,raw_name,translated_name\n" + "".join(f"character,{n},{n}\n" for n in names),
                        encoding="utf-8")
    (folder / "Book_relations_tracker.json").write_text(json.dumps(tracker, ensure_ascii=False), encoding="utf-8")
    return glossary


def _wait_loaded(qapp, dialog, seconds=20.0):
    import time
    deadline = time.monotonic() + seconds
    while dialog.is_loading() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    qapp.processEvents()
    assert not dialog.is_loading()


def test_diagram_opens_at_once_with_a_loading_screen(qapp, tmp_path, monkeypatch):
    """The owner's "View Diagram takes too long to appear": the window comes first, the data after."""
    import threading
    import relations_diagram_dialog as mod
    glossary = _write_book(tmp_path, _team_tracker())
    gate = threading.Event()
    reads = []
    real_read = mod.RelationsDiagramDialog._read_data.__func__

    def slow_read(cls, path, tracker=None, glossary_entries=None):
        reads.append(threading.current_thread().name)
        gate.wait(20)  # hold the loading thread: the constructor must not wait for it
        return real_read(cls, path, tracker, glossary_entries)

    monkeypatch.setattr(mod.RelationsDiagramDialog, "_read_data", classmethod(slow_read))
    dialog = mod.RelationsDiagramDialog(None, glossary_path=str(glossary), load_async=True)
    try:
        # Back at once: the files have not been read, the loading screen is up and spinning.
        assert dialog.is_loading() and dialog._stack.currentWidget() is dialog.loading_widget
        assert dialog.loading_widget.text_label.text() == mod.LOADING_TEXT
        assert dialog._page is None and dialog.node_count() == 0 and dialog.edge_count() == 0
        dialog.show()
        qapp.processEvents()
        assert dialog.loading_widget.is_spinning() and len(dialog.loading_widget._frames) == 60
        dialog.rebuild()  # a no-op while loading
        assert dialog._stack.currentWidget() is dialog.loading_widget
        gate.set()
        _wait_loaded(qapp, dialog)
        assert reads == ["RelationsDiagramLoad"]  # read on the thread, not on the UI thread
        assert dialog._stack.currentWidget() is dialog._page
        assert dialog.node_count() == 10 and dialog.edge_count() == 15
        assert not dialog.loading_widget.is_spinning()  # hidden: its timer is stopped
        # The same layout as the synchronous dialog gives.
        sync = mod.RelationsDiagramDialog(None, tracker=dialog.tracker, glossary_entries=dialog.glossary_entries)
        try:
            assert sync._positions == dialog._positions and not sync.is_loading()
        finally:
            sync.close()
            sync.deleteLater()
    finally:
        gate.set()
        dialog.close()
        dialog.deleteLater()


def test_diagram_async_empty_state_and_early_close(qapp, tmp_path):
    from relations_diagram_dialog import EMPTY_TEXT, RelationsDiagramDialog
    glossary = tmp_path / "Book_glossary.csv"
    glossary.write_text("type,raw_name,translated_name\ncharacter,서윤,Seo-yun\n", encoding="utf-8")
    dialog = RelationsDiagramDialog(None, glossary_path=str(glossary), load_async=True)
    try:
        _wait_loaded(qapp, dialog)
        assert dialog._stack.currentWidget() is dialog.empty_label and dialog.empty_label.text() == EMPTY_TEXT
    finally:
        dialog.close()
        dialog.deleteLater()
    # Closed (and deleted) before the data arrives: the late result is dropped quietly.
    early = RelationsDiagramDialog(None, glossary_path=str(_write_book(tmp_path, _team_tracker())), load_async=True)
    early.close()
    early.deleteLater()
    for _ in range(40):
        qapp.processEvents()
    # A tracker handed in needs no loading, whatever was asked.
    ready = RelationsDiagramDialog(None, tracker=_team_tracker(), load_async=True)
    try:
        assert not ready.is_loading() and ready.node_count() == 10
    finally:
        ready.close()
        ready.deleteLater()


def test_view_diagram_button_opens_the_window_before_the_data(gui, tmp_path):
    from PySide6.QtWidgets import QApplication
    glossary = _write_book(tmp_path, _team_tracker())
    dialog = gui._open_relations_diagram(None, glossary_path=str(glossary))
    try:
        assert dialog is gui._relations_diagram_dialog and dialog.isVisible()
        assert dialog.is_loading() or dialog.node_count() == 10  # shown first; the data follows
        _wait_loaded(QApplication.instance(), dialog)
        assert dialog.node_count() == 10 and dialog.edge_count() == 15
    finally:
        dialog.close()
        dialog.deleteLater()


def test_halgakos_loading_widget_spins_only_while_shown(qapp):
    import time
    from qt_widget_helpers import HalgakosLoadingWidget
    widget = HalgakosLoadingWidget("Loading relations…")
    try:
        assert widget.text_label.text() == "Loading relations…" and not widget.is_spinning()
        assert len(widget._frames) == 60  # one turn in 6-degree steps, rendered once
        widget.show()
        qapp.processEvents()
        assert widget.is_spinning()
        widget._started = time.perf_counter() - widget.PERIOD_S / 2  # half a turn ago
        widget.tick()
        assert widget._frame == 30
        widget.set_text("Almost there")
        assert widget.text_label.text() == "Almost there"
        widget.hide()
        assert not widget.is_spinning()
        # No icon file: text only, nothing to spin, no error.
        plain = HalgakosLoadingWidget("x", icon_path="missing.ico")
        plain.tick()
        assert plain._frames == []
        plain.deleteLater()
    finally:
        widget.deleteLater()


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


# ── Relations & Gender Tracker: selection, auto-update, edits, gender, filters ──────────────

def _wait_idle(qapp, dialog, seconds=20.0):
    """Until the loading thread has nothing left to do (a reload or a new layout)."""
    import time
    deadline = time.monotonic() + seconds
    while dialog.is_busy() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    qapp.processEvents()
    assert not dialog.is_busy()


def _open_book(qapp, glossary):
    from relations_diagram_dialog import RelationsDiagramDialog
    dialog = RelationsDiagramDialog(None, glossary_path=str(glossary), load_async=True)
    dialog.resize(1300, 850)
    dialog.show()
    _wait_loaded(qapp, dialog)
    _wait_idle(qapp, dialog)
    return dialog


def _close(qapp, dialog):
    dialog.close()
    dialog.deleteLater()
    qapp.processEvents()


def _answer_dialogs(monkeypatch, filler):
    """Edit dialogs do not run: ``filler(dialog)`` fills the fields and returns True (Save) or False."""
    from relations_diagram_dialog import RelationsDiagramDialog
    seen = []

    def _fake_exec(self, dialog):
        seen.append(dialog)
        return filler(dialog)

    monkeypatch.setattr(RelationsDiagramDialog, "_exec", _fake_exec)
    return seen


def _labels(path, a, b):
    return rt.relation_labels(rt.load_tracker(str(path)), a, b)


def _scene_texts(dialog):
    return [i.text() for i in dialog.scene.items() if hasattr(i, "text") and i.parentItem() is None]


def _chapters_tracker():
    """The team at chapter 1 and one more record at chapter 5 (slider 1 to 5)."""
    tracker = _team_tracker()
    rt.record_relations(tracker, [{"raw_name": "성규", "relations": "민: Teammate"}, {"raw_name": "민"}],
                        field="relations", chapter_index=4, chapter_num=5, chapter_file="c5.xhtml")
    return tracker


def test_tracker_window_title_filter_row_and_menus(qapp):
    from PySide6.QtWidgets import QLabel
    from relations_diagram_dialog import WINDOW_TITLE, RelationsDiagramDialog, _item_key
    dialog = RelationsDiagramDialog(None, tracker=_team_tracker())
    try:
        assert dialog.windowTitle() == WINDOW_TITLE == "Relations & Gender Tracker"
        for box in (dialog.show_relations_checkbox, dialog.show_gender_checkbox):
            assert box.isChecked() and [w.text() for w in box.findChildren(QLabel)] == ["✓"]
        assert dialog.show_relations_checkbox.text() == "Relations" and dialog.show_gender_checkbox.text() == "Gender"
        assert dialog.hide_button.text() == "Hide…" and not dialog.undo_button.isEnabled()
        leader = rt.tracker_key("민")
        menu = dialog.node_menu(leader)
        texts = [a.text() for a in menu.actions() if a.text()]
        assert texts == ["Add relation…", "Edit description…", "Set gender…", "Hide from view", "Remove from tracker…"]
        # A tracker handed in without a file: nothing to write to, but hiding works for this window.
        assert [a.text() for a in menu.actions() if a.text() and a.isEnabled()] == ["Hide from view"]
        edge = dialog.edge_menu(leader, rt.tracker_key("성규"))
        assert [a.text() for a in edge.actions()] == ["Edit relation…", "Delete relation"]
        # Right-click / double-click find the name or relation under the mouse.
        item = dialog._node_items[leader]
        assert _item_key(item.text_item) == ("node", leader)
        labels = [i for i in dialog.scene.items() if hasattr(i, "text") and i.parentItem() is None]
        assert labels and all(_item_key(i)[0] == "edge" for i in labels)
        # Hiding without a file still works (for this window only).
        next(a for a in menu.actions() if a.text() == "Hide from view").trigger()
        assert leader not in dialog._node_items and dialog.node_count() == 9
    finally:
        _close(qapp, dialog)


def test_no_question_mark_for_a_one_sided_relation(qapp):
    """The owner's 'why are there question marks': only the known side is written."""
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QColor
    from relations_diagram_dialog import RelationsDiagramDialog
    tracker = rt.empty_tracker()
    rt.record_relations(tracker, [{"raw_name": "지수", "relations": "하나: Club Member"}, {"raw_name": "하나"}],
                        field="relations", chapter_index=0, chapter_num=1, chapter_file="c1.xhtml")
    dialog = RelationsDiagramDialog(None, tracker=tracker)
    try:
        texts = _scene_texts(dialog)
        assert "Club Member" in texts and not any("?" in t for t in texts)
        tips = [i.toolTip() for i in dialog.scene.items() if i.toolTip()]
        assert any(" is Club Member to " in t for t in tips) and not any("?" in t for t in tips)
        # Neither side known: no pill at all.
        boxes = len(dialog._label_boxes)
        edge = {"a": rt.tracker_key("지수"), "b": rt.tracker_key("하나"), "label_ab": "", "label_ba": ""}
        assert dialog._add_edge_label(edge, QPointF(0, 0), QPointF(100, 0), QColor("#8b95a5"), False, []) is None
        assert len(dialog._label_boxes) == boxes
    finally:
        _close(qapp, dialog)


def test_auto_update_reloads_an_outside_write_and_keeps_the_view(qapp, tmp_path):
    from relations_diagram_dialog import POLL_MS
    glossary = _write_book(tmp_path, _chapters_tracker())
    path = rt.tracker_path_for_glossary(str(glossary))
    dialog = _open_book(qapp, glossary)
    try:
        assert dialog._poll_timer.isActive() and dialog._poll_timer.interval() == POLL_MS == 1000
        assert dialog._poll() is False  # nothing changed: no reload
        leader, member = rt.tracker_key("민"), rt.tracker_key("성규")
        dialog._on_node_clicked(leader)
        dialog.chapter_slider.setValue(1)
        view = dialog._view
        view._auto_fit = False
        view.scale(2.0, 2.0)
        view.centerOn(dialog._node_items[member])
        qapp.processEvents()
        zoom, centre = view.transform().m11(), view.mapToScene(view.viewport().rect().center())
        positions = dict(dialog._positions)
        # Another writer (an extraction, the phone) renames a relation and adds chapter 9.
        rt.edit_tracker(path, lambda t: rt.set_relation_label(t, "민", "성규", "Sworn Brother", chapter=1))
        rt.edit_tracker(path, lambda t: rt.record_relations(
            t, [{"raw_name": "민", "relations": "준호: Rival"}, {"raw_name": "준호"}], field="relations",
            chapter_index=8, chapter_num=9, chapter_file="c9.xhtml"))
        assert dialog._poll() is True
        assert dialog._node_items  # the picture stays while the new data is read
        _wait_idle(qapp, dialog)
        assert dialog.chapter_slider.maximum() == 9 and dialog.chapter_slider.value() == 1  # kept its chapter
        assert dialog._selected == leader and "Sworn Brother" in dialog.history_panel.toPlainText()
        assert any("Sworn Brother" in t for t in _scene_texts(dialog))
        assert dialog._positions == positions  # the same names: nobody moved
        assert view.transform().m11() == zoom
        assert (view.mapToScene(view.viewport().rect().center()) - centre).manhattanLength() < 2
        assert dialog._poll() is False  # loaded: quiet again
        # At the last chapter the slider follows the new last chapter.
        dialog.chapter_slider.setValue(9)
        rt.edit_tracker(path, lambda t: rt.record_relations(
            t, [{"raw_name": "민", "relations": "민주: Rival"}, {"raw_name": "민주"}], field="relations",
            chapter_index=11, chapter_num=12, chapter_file="c12.xhtml"))
        assert dialog._poll() is True
        _wait_idle(qapp, dialog)
        assert dialog.chapter_slider.maximum() == 12 and dialog.chapter_slider.value() == 12
        # Not while an edit dialog of this window is open.
        rt.edit_tracker(path, lambda t: rt.set_relation_label(t, "민", "성규", "Brother", old_label="Sworn Brother"))
        dialog._dialogs_open += 1
        assert dialog._poll() is False
        dialog._dialogs_open -= 1
        assert dialog._poll() is True
        _wait_idle(qapp, dialog)
        assert "Brother" in dialog.history_panel.toPlainText()
    finally:
        _close(qapp, dialog)


def test_auto_update_skips_a_half_written_file(qapp, tmp_path):
    import json
    glossary = _write_book(tmp_path, _team_tracker())
    path = rt.tracker_path_for_glossary(str(glossary))
    dialog = _open_book(qapp, glossary)
    try:
        shown = dialog.tracker
        good = pathlib.Path(path).read_text(encoding="utf-8")
        pathlib.Path(path).write_text(good[: len(good) // 2], encoding="utf-8")  # mid-write
        assert dialog._poll() is True
        _wait_idle(qapp, dialog)
        assert dialog.tracker is shown and dialog.node_count() == 10 and dialog.edge_count() == 15  # never empty
        assert dialog._poll() is True  # tried again at the next check
        _wait_idle(qapp, dialog)
        tracker = json.loads(good)
        rt.set_relation_label(tracker, "민", "성규", "Sworn Brother")
        rt.write_tracker(path, tracker)
        assert dialog._poll() is True
        _wait_idle(qapp, dialog)
        assert dialog.tracker is not shown and any("Sworn Brother" in t for t in _scene_texts(dialog))
        assert dialog._poll() is False
    finally:
        _close(qapp, dialog)


def test_edits_write_the_file_stick_after_a_re_extraction_and_undo(qapp, tmp_path, isolated, monkeypatch):
    from PySide6.QtCore import QUrl
    calls, answer = isolated
    answer["value"] = True
    glossary = _write_book(tmp_path, _team_tracker())
    path = rt.tracker_path_for_glossary(str(glossary))
    dialog = _open_book(qapp, glossary)
    leader, member, third = rt.tracker_key("민"), rt.tracker_key("성규"), rt.tracker_key("춘후")
    try:
        old_ab, old_ba = _labels(path, "민", "성규")
        assert old_ab and old_ba

        def edit_ab(d):
            d.fields[0].setText("Sworn Brother")  # what 성규 is to 민; the other side untouched
            return True

        seen = _answer_dialogs(monkeypatch, edit_ab)
        dialog._on_item_double(("edge", leader, member))  # double-click a relation label
        assert seen and seen[-1].windowTitle() == "Edit relation"
        assert _labels(path, "민", "성규") == ("Sworn Brother", old_ba)
        assert dialog._poll() is False  # our own save: no second reload
        assert any("Sworn Brother" in t for t in _scene_texts(dialog)) and dialog.undo_button.isEnabled()
        rt.edit_tracker(path, _record_chapter_1)  # the AI says the old label again
        assert _labels(path, "민", "성규") == ("Sworn Brother", old_ba)

        # Undo takes the last edit back (the re-extraction did not touch the corrected records).
        assert dialog.undo() is True
        assert _labels(path, "민", "성규") == (old_ab, old_ba) and not dialog.undo_button.isEnabled()
        assert dialog._poll() is False

        # Delete a relation from the side panel (asks first); the AI cannot bring it back.
        dialog._on_node_clicked(leader)
        assert len(rt.character_history(rt.load_tracker(path), "민")) == 9
        entry = dialog.history_action("delete relation ")
        assert entry is not None
        dialog._on_history_link(QUrl(f"act:{dialog._history_actions.index(entry)}"))
        qapp.processEvents()
        assert calls and "Delete the relation between" in calls[-1]
        assert len(rt.character_history(rt.load_tracker(path), "민")) == 8
        rt.edit_tracker(path, _record_chapter_1)
        assert len(rt.character_history(rt.load_tracker(path), "민")) == 8

        # A refused confirmation writes nothing.
        answer["value"] = False
        before = pathlib.Path(path).read_bytes()
        assert dialog.delete_relation_between(leader, third) is None
        assert pathlib.Path(path).read_bytes() == before
        answer["value"] = True

        # Remove a junk name: its entry and every line to it go, for good.
        junk = rt.tracker_key("부하1")
        dialog.remove_character_for(junk)
        assert junk not in rt.load_tracker(path)["entries"]
        rt.edit_tracker(path, _record_chapter_1)
        tracker = rt.load_tracker(path)
        assert junk not in tracker["entries"]
        assert all(junk not in (item.get("relations") or {}) for item in tracker["entries"].values())
        _wait_idle(qapp, dialog)
        assert junk not in dialog._node_items

        # Add a relation to a new name (typed in) at the slider's chapter.
        def add_new(d):
            d.fields[0].setEditText("새인물")
            d.fields[1].setText("Older Sister")
            return True

        _answer_dialogs(monkeypatch, add_new)
        dialog.add_relation_for(rt.tracker_key("부하2"))
        assert _labels(path, "부하2", "새인물") == ("Older Sister", "")
        _wait_idle(qapp, dialog)
        # Not in the glossary: shown with "Show unlisted names" (the window says so).
        assert rt.tracker_key("새인물") not in dialog._node_items and "Show unlisted names" in dialog.notice_label.text()
        dialog.show_unlisted_checkbox.setChecked(True)
        assert rt.tracker_key("새인물") in dialog._node_items
        dialog.show_unlisted_checkbox.setChecked(False)

        # Add a relation to a known name picked from the list.
        def add_known(d):
            combo = d.fields[0]
            d.fields[0].setCurrentIndex(combo.findData("부하3"))
            d.fields[2].setText("Mentor")
            return True

        _answer_dialogs(monkeypatch, add_known)
        dialog.add_relation_for(rt.tracker_key("부하2"))
        assert _labels(path, "부하2", "부하3") == ("", "Mentor")

        # Rename one history label, then delete a history row; both stick.
        history = rt.character_history(rt.load_tracker(path), "민")
        row = next(rel for rel in history if rel["target_key"] == third)
        label = row["timeline"][0]["label"]

        def rename(d):
            d.fields[0].setText("Old Rival")
            return True

        _answer_dialogs(monkeypatch, rename)
        dialog.edit_history_label(leader, third, label)
        assert _labels(path, "민", "춘후")[0] == "Old Rival"
        rt.edit_tracker(path, _record_chapter_1)
        assert _labels(path, "민", "춘후")[0] == "Old Rival"
        dialog.delete_history_row(leader, third, "Old Rival", 1, 1)
        assert _labels(path, "민", "춘후")[0] == ""
        rt.edit_tracker(path, _record_chapter_1)
        assert _labels(path, "민", "춘후")[0] == ""

        # Descriptions: write one, then delete it (asked first).
        def describe(d):
            d.fields[1].setPlainText("The team leader")
            return True

        _answer_dialogs(monkeypatch, describe)
        dialog.edit_description_for(leader)
        assert rt.description_at(rt.load_tracker(path), "민")["text"] == "The team leader"
        if dialog._selected != leader:
            dialog._on_node_clicked(leader)
        assert "The team leader" in dialog.history_panel.toPlainText()
        assert dialog.history_action("edit description ch 1") is not None
        dialog.delete_description_for(leader, 1)
        assert rt.description_at(rt.load_tracker(path), "민") is None
        assert dialog._poll() is False
    finally:
        _close(qapp, dialog)


def test_undo_is_refused_when_someone_else_changed_the_same_character(qapp, tmp_path, monkeypatch):
    glossary = _write_book(tmp_path, _team_tracker())
    path = rt.tracker_path_for_glossary(str(glossary))
    dialog = _open_book(qapp, glossary)
    try:
        leader, member = rt.tracker_key("민"), rt.tracker_key("성규")

        def edit_ab(d):
            d.fields[0].setText("Sworn Brother")
            return True

        _answer_dialogs(monkeypatch, edit_ab)
        dialog.edit_relation_between(leader, member)
        rt.edit_tracker(path, lambda t: rt.set_relation_label(t, "민", "성규", "Blood Brother"))
        assert dialog.undo() is False
        assert "Cannot undo" in dialog.notice_label.text()
        assert _labels(path, "민", "성규")[0] == "Blood Brother"
        # Cancel writes nothing and offers no undo.
        _answer_dialogs(monkeypatch, lambda d: False)
        before = pathlib.Path(path).read_bytes()
        assert dialog.edit_relation_between(leader, member) is None
        assert pathlib.Path(path).read_bytes() == before and not dialog.undo_button.isEnabled()
    finally:
        _close(qapp, dialog)


def test_gender_tints_history_gender_only_names_and_set_gender(qapp, tmp_path, monkeypatch):
    from relations_diagram_dialog import GENDER_TINTS, NEUTRAL_TINT
    glossary = _write_book(tmp_path, _chapters_tracker(), extra_names=("외톨이",))
    rt.set_gender(str(glossary), "민", "male", 1)
    rt.set_gender(str(glossary), "민", "female", 5)
    rt.set_gender(str(glossary), "외톨이", "female", 2)
    gender_path = rt.gender_tracker_path_for_glossary(str(glossary))
    dialog = _open_book(qapp, glossary)
    leader, loner = rt.tracker_key("민"), rt.tracker_key("외톨이")
    try:
        assert dialog.chapter_slider.value() == 5
        assert dialog._node_items[leader].tint == GENDER_TINTS["female"]
        # Tracked for gender, no relations: a name without lines.
        assert loner in dialog._node_items and dialog.node_count() == 11 and dialog.edge_count() == 15
        dialog._on_node_clicked(leader)
        assert "Gender: Male (Ch. 1), Female (Ch. 5)" in dialog.history_panel.toPlainText()
        dialog.chapter_slider.setValue(1)  # the tint and the history follow the slider
        assert dialog._node_items[leader].tint == GENDER_TINTS["male"]
        assert "Gender: Male (Ch. 1)" in dialog.history_panel.toPlainText()
        assert "Female" not in dialog.history_panel.toPlainText()
        assert loner not in dialog._node_items  # first tracked at chapter 2
        dialog.chapter_slider.setValue(5)

        # Set a gender for all chapters (the gender tracker's own writer); no reload of our own write.
        def all_male(d):
            d.fields[0].setCurrentIndex(d.fields[0].findData("male"))
            d.fields[1].setCurrentIndex(d.fields[1].findData("all"))
            return True

        _answer_dialogs(monkeypatch, all_male)
        dialog.set_gender_for(loner)
        stored = rt.read_gender_tracker(gender_path)
        assert rt.gender_history(stored, "외톨이")["decision"] == "male"
        assert dialog._node_items[loner].tint == GENDER_TINTS["male"] and dialog._poll() is False
        assert dialog.undo() is True
        assert rt.gender_history(rt.read_gender_tracker(gender_path), "외톨이")["decision"] == "auto"

        # One chapter only (the slider's).
        def chapter_male(d):
            d.fields[0].setCurrentIndex(d.fields[0].findData("male"))
            d.fields[1].setCurrentIndex(d.fields[1].findData("chapter"))
            return True

        _answer_dialogs(monkeypatch, chapter_male)
        dialog.set_gender_for(leader)
        assert rt.gender_history_text(rt.gender_history(rt.read_gender_tracker(gender_path), "민")) == "Male (Ch. 1-5)"
        # An outside gender write is picked up too.
        rt.set_gender(str(glossary), "외톨이", "male", None)
        assert dialog._poll() is True
        _wait_idle(qapp, dialog)
        assert dialog._node_items[loner].tint == GENDER_TINTS["male"]

        # Gender off: neutral names, no gender-only names, no gender history.
        dialog.show_gender_checkbox.setChecked(False)
        assert loner not in dialog._node_items
        assert all(item.tint == NEUTRAL_TINT for item in dialog._node_items.values())
        dialog._on_node_clicked(rt.tracker_key("성규"))
        assert "Gender:" not in dialog.history_panel.toPlainText()
    finally:
        _close(qapp, dialog)


def test_filters_hide_names_and_persist_in_the_trackers_view(qapp, tmp_path):
    import json
    glossary = _write_book(tmp_path, _team_tracker())
    path = rt.tracker_path_for_glossary(str(glossary))
    dialog = _open_book(qapp, glossary)
    member, leader = rt.tracker_key("성규"), rt.tracker_key("민")

    def stored():
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))

    try:
        entries = stored()["entries"]
        dialog._on_node_clicked(leader)
        dialog.show_relations_checkbox.setChecked(False)
        assert dialog.edge_count() == 0 and dialog._label_boxes == [] and dialog.node_count() == 10
        assert "Relations are switched off" in dialog.history_panel.toPlainText()
        assert stored()["view"] == {"hidden": [], "show_relations": False, "show_gender": True}
        assert stored()["entries"] == entries  # the relations data is not touched
        assert dialog._poll() is False  # the view write does not reload
        dialog.show_relations_checkbox.setChecked(True)
        assert "view" not in stored()  # back to the defaults

        # The Hide checklist: search, check a name, Show all.
        positions = {k: (i.pos().x(), i.pos().y()) for k, i in dialog._node_items.items()}
        popup = dialog.hide_popup()
        assert popup.tree.topLevelItemCount() == 10
        popup.search.setText("부하")
        assert sum(1 for i in range(10) if not popup.tree.topLevelItem(i).isHidden()) == 5
        popup.search.clear()
        assert popup.set_checked(member, True)
        assert member not in dialog._node_items and dialog.node_count() == 9 and dialog.edge_count() < 15
        assert all((i.pos().x(), i.pos().y()) == positions[k] for k, i in dialog._node_items.items())  # others stay
        assert dialog.hide_button.text() == "Hide (1)…"
        assert stored()["view"]["hidden"] == [member]
        # "Hide from view" on a name's menu.
        next(a for a in dialog.node_menu(leader).actions() if a.text() == "Hide from view").trigger()
        assert leader not in dialog._node_items and dialog._selected is None
        assert dialog.view_settings()["hidden"] == [member, leader]
        # Remembered for the book: a new window opens with the same choices.
        dialog.show_gender_checkbox.setChecked(False)
        again = _open_book(qapp, glossary)
        try:
            assert again.node_count() == 8 and not again.show_gender_checkbox.isChecked()
            assert again.hide_button.text() == "Hide (2)…"
        finally:
            _close(qapp, again)
        dialog.show_gender_checkbox.setChecked(True)
        dialog.hide_popup().show_all()
        assert dialog.node_count() == 10 and "view" not in stored()
        # Undo works for view choices too.
        assert dialog.undo() is True and dialog.view_settings()["hidden"] == [member, leader]
    finally:
        _close(qapp, dialog)


def _two_sided_later_tracker():
    """갑 says 을 is a Friend at chapter 1; 을's row says what 을 is to 갑 (Rival) only at chapter 10."""
    tracker = rt.empty_tracker()
    rt.record_relations(tracker, [{"raw_name": "갑", "relations": "을: Friend"}, {"raw_name": "을"}],
                        field="relations", chapter_index=0, chapter_num=1, chapter_file="c1.xhtml")
    rt.record_relations(tracker, [{"raw_name": "을", "relations": "갑: Rival"}, {"raw_name": "갑"}],
                        field="relations", chapter_index=9, chapter_num=10, chapter_file="c10.xhtml")
    return tracker


def test_filling_a_side_unknown_at_the_sliders_chapter_keeps_its_later_label(qapp, tmp_path, monkeypatch):
    """Review: an empty side at the slider's chapter is added there, never a rename of a later label."""
    glossary = _write_book(tmp_path, _two_sided_later_tracker())
    path = rt.tracker_path_for_glossary(str(glossary))
    dialog = _open_book(qapp, glossary)
    a, b = rt.tracker_key("갑"), rt.tracker_key("을")
    try:
        dialog.chapter_slider.setValue(3)
        assert rt.relation_labels(rt.load_tracker(path), "갑", "을", 3) == ("", "Friend")

        def fill(d):
            assert d.fields[0].text() == ""
            d.fields[0].setText("Classmate")  # the side unknown at chapter 3
            d.fields[2].setValue(3)
            return True

        _answer_dialogs(monkeypatch, fill)
        dialog.edit_relation_between(a, b)
        saved = rt.load_tracker(path)
        assert rt.relation_labels(saved, "갑", "을", 3) == ("Classmate", "Friend")
        assert rt.relation_labels(saved, "갑", "을", 10) == ("Rival", "Friend")  # not renamed
        assert not (saved.get("edits") or {}).get("corrections")  # nothing remembered as a correction
    finally:
        _close(qapp, dialog)


def test_panel_of_a_name_that_is_not_drawn_follows_edits(qapp, tmp_path, isolated):
    """Review: a name selected from the panel but not drawn (unlisted) still gets its panel redrawn."""
    calls, answer = isolated
    answer["value"] = True
    tracker = rt.empty_tracker()
    rt.record_relations(tracker, [{"raw_name": "갑", "relations": "을: Friend"},
                                  {"raw_name": "을", "relations": "병: Rival"}, {"raw_name": "병"}],
                        field="relations", chapter_index=0, chapter_num=1, chapter_file="c1.xhtml")
    glossary = _write_book(tmp_path, tracker)
    glossary.write_text("type,raw_name,translated_name\ncharacter,갑,Gap\n", encoding="utf-8")  # 을, 병 unlisted
    dialog = _open_book(qapp, glossary)
    a, b = rt.tracker_key("갑"), rt.tracker_key("을")
    try:
        assert b not in dialog._node_items and b in dialog._node_info
        dialog.select(b)
        assert dialog._selected == b and "Friend" in dialog.history_panel.toPlainText()
        dialog.delete_relation_between(b, a)
        assert a not in (rt.load_tracker(rt.tracker_path_for_glossary(str(glossary)))["entries"][b].get("relations") or {})
        assert dialog._selected == b and "Friend" not in dialog.history_panel.toPlainText()
    finally:
        _close(qapp, dialog)


def test_clearing_one_chapters_gender_asks_first_and_gender_only_names_can_be_removed(qapp, tmp_path, monkeypatch,
                                                                                       isolated):
    calls, answer = isolated
    glossary = _write_book(tmp_path, _chapters_tracker(), extra_names=("외톨이",))
    rt.set_gender(str(glossary), "민", "female", 5)
    rt.set_gender(str(glossary), "외톨이", "female", 2)
    gender_path = rt.gender_tracker_path_for_glossary(str(glossary))
    dialog = _open_book(qapp, glossary)
    leader, loner = rt.tracker_key("민"), rt.tracker_key("외톨이")
    try:
        def clear_chapter(d):
            d.fields[0].setCurrentIndex(d.fields[0].findData("auto"))
            d.fields[1].setCurrentIndex(d.fields[1].findData("chapter"))
            return True

        _answer_dialogs(monkeypatch, clear_chapter)
        answer["value"] = False
        assert dialog.set_gender_for(leader) is None
        assert calls and "Delete the gender recorded for" in calls[-1]
        assert rt.gender_history(rt.read_gender_tracker(gender_path), "민")["timeline"]
        answer["value"] = True
        dialog.set_gender_for(leader)
        assert not rt.gender_history(rt.read_gender_tracker(gender_path), "민")["timeline"]
        # A name only the gender tracker knows: Remove from tracker takes it off the diagram for good.
        assert loner in dialog._node_items
        dialog.remove_character_for(loner)
        assert loner not in dialog._node_items
        assert loner in rt.load_tracker(rt.tracker_path_for_glossary(str(glossary)))["edits"]["removed"]
    finally:
        _close(qapp, dialog)


def test_auto_update_skips_a_glossary_mid_save(qapp, tmp_path):
    """Review: the Glossary Editor rewrites the CSV in place; an empty file is not shown as no names."""
    glossary = _write_book(tmp_path, _team_tracker())
    dialog = _open_book(qapp, glossary)
    try:
        entries = dialog.glossary_entries
        good = glossary.read_bytes()
        glossary.write_bytes(b"")  # truncated, mid-save
        assert dialog._poll() is True
        _wait_idle(qapp, dialog)
        assert dialog.glossary_entries is entries and dialog.node_count() == 10
        assert dialog._poll() is True  # tried again at the next check
        _wait_idle(qapp, dialog)
        glossary.write_bytes(good)
        assert dialog._poll() is True
        _wait_idle(qapp, dialog)
        assert dialog._poll() is False and dialog.node_count() == 10
    finally:
        _close(qapp, dialog)


def test_tracker_window_still_appears_at_once(qapp, tmp_path):
    """Nothing new runs before the window is shown: constructor + show stay around 0.1 s."""
    import time
    from relations_diagram_dialog import RelationsDiagramDialog
    tracker = rt.empty_tracker()
    names = [f"인물{i:02d}" for i in range(15)]
    rows = [{"raw_name": n, "translated_name": f"P{i}", "relations": f"{names[(i + 1) % 15]}: Friend"}
            for i, n in enumerate(names)]
    rt.record_relations(tracker, rows, field="relations", chapter_index=0, chapter_num=1, chapter_file="c1.xhtml")
    glossary = _write_book(tmp_path, tracker)
    RelationsDiagramDialog(None, tracker=rt.empty_tracker()).deleteLater()  # warm the imports
    started = time.perf_counter()
    dialog = RelationsDiagramDialog(None, glossary_path=str(glossary), load_async=True)
    dialog.show()
    elapsed = time.perf_counter() - started
    try:
        assert dialog.is_loading() and elapsed < 0.5, elapsed
        _wait_loaded(qapp, dialog)
        assert dialog.node_count() == 15
    finally:
        _close(qapp, dialog)
