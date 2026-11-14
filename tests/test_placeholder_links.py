# -*- coding: utf-8 -*-
"""AI prompt placeholders shown like links (owner 2026-11-14): "highlight AI prompt placeholders in blue
everywhere that has them like a URL. i want users to see exactly what they say at the time on a tooltip
when they hover over it. so any changes to types will reflect in real time when they hover over."

qt_widget_helpers.link_placeholders / link_placeholders_in_label mark the placeholders a prompt really
replaces (blue, underlined) and show, on hover, what each expands to at that moment. Display only.

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_placeholder_links.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

pytest.importorskip("PySide6")

TYPES = {
    "character": {"enabled": True, "has_gender": True},
    "terms": {"enabled": True, "has_gender": False},
    "titles": {"enabled": True, "has_gender": True},
    "nicknames": {"enabled": True, "has_gender": True},
}


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("GLOSSARY_"):
            monkeypatch.delenv(key, raising=False)
    for key, sub in (("GLOSSARION_APP_DIR", "app"), ("GLOSSARION_LIBRARY_DIR", "lib"), ("OUTPUT_DIRECTORY", "out")):
        (tmp_path / sub).mkdir()
        monkeypatch.setenv(key, str(tmp_path / sub))


def _underlined_ranges(editor):
    out = []
    block = editor.document().firstBlock()
    while block.isValid():
        text = block.text()
        for r in block.layout().formats():
            if r.format.fontUnderline():
                out.append(text[r.start:r.start + r.length])
        block = block.next()
    return out


def _point_in(editor, text, offset=1):
    """Viewport point over the character ``offset`` into the first occurrence of ``text``."""
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QTextCursor
    position = editor.toPlainText().index(text) + offset
    cursor = QTextCursor(editor.document())
    cursor.setPosition(position)
    left = editor.cursorRect(cursor)
    cursor.setPosition(position + 1)
    right = editor.cursorRect(cursor)
    return QPoint((left.left() + right.left()) // 2, left.center().y())


# =============================================================================================
# the shared helper
# =============================================================================================

def test_highlighter_marks_exactly_the_known_tokens_and_keeps_text_undo_and_modified(qapp):
    from PySide6.QtWidgets import QTextEdit
    from qt_widget_helpers import link_placeholders
    editor = QTextEdit()
    editor.setAcceptRichText(False)
    text = "Use {language} and {fields}; not {unknown}, {lang uage} or {{x}}.\nAgain {language}"
    editor.setPlainText(text)
    editor.document().setModified(False)
    names = {"language", "fields"}
    links = link_placeholders(editor, lambda: names, lambda name: name)
    links.highlighter.rehighlight()
    assert _underlined_ranges(editor) == ["{language}", "{fields}", "{language}"]
    assert editor.toPlainText() == text
    assert not editor.document().isModified()
    assert not editor.document().isUndoAvailable()  # formatting is not an edit
    # the valid names can change live
    names.discard("fields")
    links.refresh()
    assert _underlined_ranges(editor) == ["{language}", "{language}"]
    # typing, undo and redo work and the format follows the text
    cursor = editor.textCursor()
    cursor.movePosition(cursor.MoveOperation.End)
    cursor.insertText(" {fields}{language}")
    assert editor.document().isUndoAvailable()
    assert _underlined_ranges(editor)[-1] == "{language}"
    editor.undo()
    assert editor.toPlainText() == text
    assert _underlined_ranges(editor) == ["{language}", "{language}"]
    editor.redo()
    assert editor.toPlainText() == text + " {fields}{language}"
    assert _underlined_ranges(editor) == ["{language}", "{language}", "{language}"]


def test_token_lookup_by_position(qapp):
    from PySide6.QtWidgets import QPlainTextEdit
    from qt_widget_helpers import link_placeholders, placeholder_at, placeholder_spans
    assert placeholder_spans("a {x} {y} {z}", {"x", "z"}) == [(2, 5, "x"), (10, 13, "z")]
    assert placeholder_at("a {x} b", 2, {"x"}) == (2, 5, "x")
    assert placeholder_at("a {x} b", 5, {"x"}) == (2, 5, "x")
    assert placeholder_at("a {x} b", 1, {"x"}) is None
    assert placeholder_at("a {x} b", 6, {"x"}) is None
    assert placeholder_at("a {x} b", 3, {"y"}) is None

    editor = QPlainTextEdit()
    editor.setPlainText("Translate to {target_lang} now, keep {other}.")
    editor.resize(600, 120)
    editor.show()
    qapp.processEvents()
    links = link_placeholders(editor, {"target_lang"}, lambda name: "German")
    assert links._hit(_point_in(editor, "{target_lang}", 3))[0] == "target_lang"
    assert links._hit(_point_in(editor, "Translate", 2)) is None
    assert links._hit(_point_in(editor, "{other}", 2)) is None  # not replaced there: not a placeholder
    editor.hide()


def test_tooltip_is_resolved_at_hover_time_and_shown_verbatim(qapp):
    from PySide6.QtCore import QEvent, QPoint
    from PySide6.QtGui import QHelpEvent
    from PySide6.QtWidgets import QApplication, QTextEdit, QToolTip
    from qt_widget_helpers import link_placeholders, placeholder_tooltip
    editor = QTextEdit()
    editor.setPlainText("Language: {target_lang}")
    editor.resize(500, 100)
    editor.show()
    qapp.processEvents()
    current = {"value": "German"}
    link_placeholders(editor, {"target_lang"}, lambda name: current["value"])
    point = _point_in(editor, "{target_lang}", 4)
    for value in ("German", "Korean <b>&"):
        current["value"] = value
        event = QHelpEvent(QEvent.ToolTip, point, editor.viewport().mapToGlobal(point))
        QApplication.sendEvent(editor.viewport(), event)
        assert QToolTip.text() == placeholder_tooltip("target_lang", value)
    assert "Korean &lt;b&gt;&amp;" in QToolTip.text()
    QToolTip.hideText()
    editor.hide()


def test_tooltip_texts():
    from qt_widget_helpers import placeholder_tooltip
    tip = placeholder_tooltip("relations_instruction", "Line one\n  `a: b`\n" + "x" * 90)
    assert "<b>{relations_instruction}</b>" in tip
    assert "Line one<br>  `a: b`<br>" in tip and "width='520'" in tip and "pre-wrap" in tip
    assert "width=" not in placeholder_tooltip("language", "English")
    inactive = placeholder_tooltip("relations_instruction", {
        "active": False, "text": "", "note": "Active when the Relations Tracker is on",
        "inactive": "Inactive right now: its line is removed from the prompt."})
    assert "Inactive right now: its line is removed from the prompt." in inactive
    assert "Active when the Relations Tracker is on" in inactive
    per_request = placeholder_tooltip("marker", {"per_request": True, "text": "context window marker count"})
    assert "Filled in for each request: context window marker count" in per_request
    both = placeholder_tooltip("column_rule", [("Balanced/Full", "A"), ("Minimal", "B")])
    assert "<i>Balanced/Full</i>" in both and "<i>Minimal</i>" in both
    assert "(empty)" in placeholder_tooltip("x", "")


def test_label_keeps_its_visible_text_selection_and_wrap(qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QTextDocument
    from PySide6.QtWidgets import QLabel
    from qt_widget_helpers import link_placeholders_in_label
    text = "Use {target_lang}  for <the> language & {html}.\nIn {unknown} too."
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    links = link_placeholders_in_label(label, {"target_lang", "html"}, lambda name: name)
    document = QTextDocument()
    document.setHtml(label.text())
    assert document.toPlainText() == text
    assert label.text().count("<a href=") == 2 and "placeholder:target_lang" in label.text()
    assert label.wordWrap()
    assert label.textInteractionFlags() & Qt.TextSelectableByMouse
    assert label.textInteractionFlags() & Qt.LinksAccessibleByMouse
    assert not label.openExternalLinks()
    # a new text goes through the helper and is linked as well
    link_placeholders_in_label(label, {"target_lang"}, lambda name: name, text="Placeholder {target_lang}")
    assert links.text == "Placeholder {target_lang}" and label.text().count("<a href=") == 1


def test_line_edit_keeps_text_and_history(qapp):
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QLineEdit
    from qt_widget_helpers import link_placeholders
    line = QLineEdit("Available placeholders: {fields}, {language},  Separator: \\x1F")
    line.setReadOnly(True)
    line.resize(700, 30)
    line.show()
    qapp.processEvents()
    links = link_placeholders(line, {"fields", "language"}, lambda name: name)
    assert line.text() == "Available placeholders: {fields}, {language},  Separator: \\x1F"
    assert not line.isModified() and not line.isUndoAvailable()
    hits = {links._hit(QPoint(x, line.height() // 2)) for x in range(0, 700, 2)}
    assert hits == {None, ("fields", None), ("language", None)}
    line.hide()


# =============================================================================================
# Glossary Manager: the live values (unsaved widget state)
# =============================================================================================

@pytest.fixture
def gui(qapp, monkeypatch):
    import copy
    from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
    from GlossaryManager_GUI import GlossaryManagerMixin
    import review_dialog
    monkeypatch.setattr(review_dialog, "confirm_prompt_profile_action", lambda *args: True)

    class Harness(GlossaryManagerMixin, QMainWindow):
        def __init__(self):
            super().__init__()
            self.config = {"custom_entry_types": copy.deepcopy(TYPES), "custom_glossary_fields": ["description"],
                           "glossary_enable_relations_tracker": True}
            self._screen = QApplication.primaryScreen().geometry()
            self.dialog = self

        def save_config(self, show_message=False):
            return True

        def append_log(self, text):
            pass

    window = Harness()
    window.pages = []
    for name in ("_setup_manual_glossary_tab", "_setup_auto_glossary_tab"):
        page = QWidget()
        getattr(window, name)(page)
        window.pages.append(page)
    yield window
    dialog = getattr(window, "_placeholder_text_dialog", None)
    if dialog is not None:
        try:
            dialog.hide()
            dialog.deleteLater()
        except RuntimeError:
            pass
    window.deleteLater()


def _resolver(widget):
    return widget._placeholder_links._resolver


def test_prompt_editors_and_lines_are_linked(gui):
    from PySide6.QtWidgets import QLineEdit
    from extract_glossary_from_epub import known_placeholder_names
    for editor, mode in ((gui.manual_prompt_text, "balanced"), (gui.auto_prompt_text, "minimal")):
        links = editor._placeholder_links
        assert links._current_names() == frozenset(known_placeholder_names(mode, tokens=False))
        links.highlighter.rehighlight()
        underlined = _underlined_ranges(editor)
        assert underlined and all(u.startswith("{") and u.endswith("}") for u in underlined)
    lines = [w for page in gui.pages for w in page.findChildren(QLineEdit)
             if w.text().startswith("Available placeholders:")]
    assert len(lines) == 2 and all(hasattr(w, "_placeholder_links") for w in lines)


def test_tooltips_follow_unsaved_entry_type_changes(gui):
    """Untick has gender / an entry type: the next hover already says so (nothing saved or reopened)."""
    from qt_widget_helpers import placeholder_tooltip
    balanced = _resolver(gui.manual_prompt_text)
    minimal = _resolver(gui.auto_prompt_text)
    gender_before = balanced("gender_instruction")["text"]
    relations_before = balanced("relations_instruction")["text"]
    assert "character, titles, nicknames" in gender_before
    assert "for character, titles, nicknames entries only" in relations_before
    before_tips = [placeholder_tooltip(n, balanced(n)) for n in ("gender_instruction", "relations_instruction")]

    gui.custom_entry_types["titles"]["has_gender"] = False  # live dict, not saved
    gui.type_enabled_checkboxes["nicknames"].setChecked(False)  # unsaved checkbox
    gender_after = balanced("gender_instruction")["text"]
    assert "character" in gender_after and "titles" not in gender_after and "nicknames" not in gender_after
    assert "for character entries only" in balanced("relations_instruction")["text"]
    assert minimal("gender_instruction")["text"] == gender_after
    after_tips = [placeholder_tooltip(n, balanced(n)) for n in ("gender_instruction", "relations_instruction")]
    assert all(a != b for a, b in zip(after_tips, before_tips))

    gui.custom_entry_types["character"]["has_gender"] = False
    off = balanced("gender_instruction")
    assert off["active"] is False and off["text"] == ""
    assert "Inactive right now: replaced with nothing." in placeholder_tooltip("gender_instruction", off)


def test_inactive_placeholder_says_inactive(gui):
    from qt_widget_helpers import placeholder_tooltip
    gui.config["glossary_enable_relations_tracker"] = False
    value = _resolver(gui.manual_prompt_text)("relations_instruction")
    assert value["active"] is False
    tip = placeholder_tooltip("relations_instruction", value)
    assert "Inactive right now: its line is removed from the prompt." in tip
    assert "Active when the Relations Tracker is on" in tip


def test_placeholder_text_editor_tokens_and_selected_text_use_the_box(gui):
    dialog = gui._open_placeholder_text_dialog()
    combo, editor = dialog.placeholder_combo, dialog.placeholder_editor
    combo.setCurrentIndex(combo.findData("relations_instruction"))
    links = editor._placeholder_links
    assert links._current_names() == frozenset(
        {"field", "language", "gender_types", "column_rule", "example_row_1", "example_row_2"})
    assert dialog.placeholder_note.text().count("placeholder:gender_types") == 1
    value = dialog.placeholder_token_value("gender_types")
    assert (value if isinstance(value, dict) else value[0][1])["text"] == "character, titles, nicknames"
    gui.custom_entry_types["titles"]["has_gender"] = False
    value = dialog.placeholder_token_value("gender_types")
    assert (value if isinstance(value, dict) else value[0][1])["text"] == "character, nicknames"
    # the selected placeholder says what the box says now, unsaved
    editor.setPlainText("Relations for {gender_types} only.")
    selected = dict(dialog.placeholder_selected_value("relations_instruction"))
    assert selected["Balanced/Full"]["text"] == "Relations for character, nicknames only."
    assert selected["Minimal"]["active"] is False  # Minimal's Include Relations Column is off
    gui.minimal_include_relations_checkbox.setChecked(True)  # unsaved toggle
    selected = dialog.placeholder_selected_value("relations_instruction")
    minimal_text = selected["text"] if isinstance(selected, dict) else dict(selected)["Minimal"]["text"]
    assert "Relations for character, nicknames only." in minimal_text
    assert "glossary_placeholder_texts" not in gui.config  # nothing saved
    # a placeholder text without tokens: nothing in its box is a placeholder
    combo.setCurrentIndex(combo.findData("description_example"))
    assert links._current_names() == frozenset()


def test_chapter_text_says_it_is_replaced_with_nothing(gui):
    from qt_widget_helpers import placeholder_tooltip
    for name in ("chapter_text", "text"):
        value = _resolver(gui.manual_prompt_text)(name)
        assert value["active"] is False and not value["per_request"] and value["text"] == ""
        tip = placeholder_tooltip(name, value)
        assert "Replaced with nothing in the system prompt" in tip and "Filled in" not in tip


def test_glossary_refinement_placeholders_say_the_live_schema_and_types(gui):
    gui.custom_glossary_fields = ["description", "notes"]  # live list, not saved
    value = gui._glossary_refinement_placeholder_value
    assert value("fields")["text"] == "Columns:\ntype, raw_name, translated_name, description, notes"
    assert value("columns")["text"] == value("fields")["text"]
    assert value("fields1")["text"] == (
        "Columns (separated by Unit Separator character \\x1F):\n"
        "type\\x1Fraw_name\\x1Ftranslated_name\\x1Fdescription\\x1Fnotes")
    gui.custom_glossary_fields = ["notes"]
    assert value("fields")["text"] == "Columns:\ntype, raw_name, translated_name, notes"
    # "Send all entry types" (the unsaved default): the active types, or the selected ones
    assert value("entries") == "character, terms, titles, nicknames"
    gui.type_enabled_checkboxes["terms"].setChecked(False)  # unsaved
    assert value("entries") == "character, titles, nicknames"
    gui.config.update(glossary_refinement_type_mode="selected", glossary_refinement_selected_types=["Titles"])
    assert value("entries") == "titles"
    gui.config["glossary_refinement_chunking_mode"] = "separate"
    assert value("entries")["per_request"]
    for name in ("entry_type", "chunk_index", "total_chunks"):
        assert value(name)["per_request"] and value(name)["text"]


def test_single_pass_header_placeholders_say_the_live_prompts(gui):
    from PySide6.QtWidgets import QTextEdit
    from extract_glossary_from_epub import balanced_system_prompt_preview
    glossary = gui._single_pass_header_placeholder_value("glossary_prompt")
    assert glossary == balanced_system_prompt_preview(gui._glossary_prompt_editor_text("balanced"),
                                                      gui._live_glossary_prompt_settings())
    assert "{fields}" not in glossary and "type, raw_name, translated_name" in glossary
    gui.manual_prompt_text.setPlainText("Only {entries} please.")  # unsaved edit of the Balanced/Full prompt
    assert gui._single_pass_header_placeholder_value("glossary_prompt").startswith(
        "Only Character, Terms, Titles, & Nicknames entries please.")
    gui.prompt_text = QTextEdit()
    gui.prompt_text.setPlainText("Translate to {target_lang}.")
    gui._prompt_placeholder_value = lambda name: "German"
    translation = gui._single_pass_header_placeholder_value("translation_prompt")
    assert translation["text"] == "Translate to German." and translation["extra"]


def test_described_and_target_language_helpers():
    from qt_widget_helpers import described_placeholder_resolver, target_language_now
    names, resolver = described_placeholder_resolver({"{raw_text}": "the raw text."})
    assert names == frozenset({"raw_text"})
    assert resolver("raw_text") == {"per_request": True, "text": "the raw text."}

    class Plain:
        config = {"output_language": "French"}

    class Main(Plain):
        def _prompt_placeholder_value(self, name):
            return "Korean"

    assert target_language_now(Plain()) == "French" and target_language_now(Main()) == "Korean"


def test_metadata_prompts_say_the_dialog_output_language():
    from types import SimpleNamespace
    from metadata_batch_translator import MetadataBatchTranslatorUI
    owner = SimpleNamespace(gui=SimpleNamespace(config={"output_language": "Spanish"}), output_lang_combo=None)
    value = MetadataBatchTranslatorUI._prompt_placeholder_value
    assert value(owner, "target_lang") == "Spanish"
    owner.output_lang_combo = SimpleNamespace(currentText=lambda: "Japanese")
    assert value(owner, "target_lang") == "Japanese"
    assert value(owner, "field_value")["per_request"]


def test_assistant_prompt_placeholders_say_refinement_only():
    import ast
    from _src_corpus import find_method  # the method from translator_gui's source, without importing it
    namespace = {}
    exec(compile(ast.Module(body=[find_method("_assistant_prompt_placeholder_value")], type_ignores=[]),
                 "translator_gui.py", "exec"), namespace)
    owner = type("Owner", (), {"_prompt_placeholder_value": lambda self, name: "German"})()
    value = namespace["_assistant_prompt_placeholder_value"]
    target = value(owner, "target_lang")
    assert target["text"] == "German" and "refinement requests only" in target["extra"]
    assert value(owner, "QA_Issues")["active"] is False


def test_the_other_prompt_editors_are_wired():
    """Every desktop prompt editor whose placeholders the code replaces goes through the shared helpers."""
    def source(name):
        return (SRC / name).read_text(encoding="utf-8")
    assert source("metadata_batch_translator.py").count("self._link_prompt_placeholders(") == 4
    assert source("review_dialog.py").count("self._link_review_placeholders(") == 2
    other = source("other_settings.py")
    assert "link_placeholders(self.summary_system_text, {'target_lang'}" in other
    assert "link_described_placeholders(self.summary_user_text, summary_inputs)" in other
    assert other.count("link_described_placeholders_in_label(placeholder_lbl,") == 2
    assert "link_described_placeholders(user_text, ocr_inputs)" in other
    manga = source("manga_integration.py")
    assert "link_placeholders(prompt_widget, {'target_lang'}, target_lang_now)" in manga
    assert "link_glossary(prompt_editor, 'balanced', prompt=prompt_editor.toPlainText)" in manga
    parallel = source("parallel_epub_glossary.py")
    assert "self._link_wrapper_placeholders(wrapper_help)" in parallel
    assert 'link_glossary(self.system_prompt_edit, "balanced", prompt=self.system_prompt_edit.toPlainText)' in parallel
    assert "self._assistant_prompt_placeholder_value)" in source("translator_gui.py")
