from pathlib import Path
import json
from types import SimpleNamespace


from output_workspace import (
    read_workspace_source_path,
    rename_input_for_workspace_collision,
    source_format_label,
    workspace_source_format,
    write_workspace_source_reference,
)


def test_source_format_labels_are_limited_to_collision_sensitive_inputs():
    assert source_format_label("Novel.EPUB") == "EPUB"
    assert source_format_label("Novel.PdF") == "PDF"
    assert source_format_label("Novel.txt") == "TXT"
    assert source_format_label("Novel.md") == ""
    assert rename_input_for_workspace_collision("Novel.pdf", "") == "Novel.pdf"


def test_same_named_pdf_is_renamed_for_existing_epub_workspace(tmp_path):
    workspace = tmp_path / "Same Name"
    write_workspace_source_reference(workspace, tmp_path / "raw" / "Same Name.epub")
    source = tmp_path / "Same Name.pdf"
    source.write_bytes(b"updated pdf")

    renamed = rename_input_for_workspace_collision(str(source), str(workspace))

    assert Path(renamed) == tmp_path / "Same Name_PDF.pdf"
    assert not source.exists()
    assert Path(renamed).read_bytes() == b"updated pdf"


def test_same_named_epub_is_renamed_for_existing_pdf_workspace(tmp_path):
    workspace = tmp_path / "Same Name"
    write_workspace_source_reference(workspace, tmp_path / "raw" / "Same Name.pdf")
    source = tmp_path / "Same Name.epub"
    source.write_bytes(b"updated epub")

    renamed = rename_input_for_workspace_collision(str(source), str(workspace))

    assert Path(renamed) == tmp_path / "Same Name_EPUB.epub"
    assert not source.exists()
    assert Path(renamed).read_bytes() == b"updated epub"


def test_matching_format_keeps_original_input_name(tmp_path):
    workspace = tmp_path / "Same Name"
    write_workspace_source_reference(workspace, tmp_path / "old" / "Same Name.txt")
    source = tmp_path / "Same Name.txt"
    source.write_text("updated", encoding="utf-8")

    unchanged = rename_input_for_workspace_collision(str(source), str(workspace))

    assert Path(unchanged) == source
    assert source.exists()


def test_existing_renamed_source_is_never_overwritten(tmp_path):
    workspace = tmp_path / "Novel"
    write_workspace_source_reference(workspace, tmp_path / "Novel.epub")
    source = tmp_path / "Novel.pdf"
    source.write_bytes(b"new")
    existing = tmp_path / "Novel_PDF.pdf"
    existing.write_bytes(b"old")

    renamed = rename_input_for_workspace_collision(str(source), str(workspace))

    assert Path(renamed) == tmp_path / "Novel_PDF_2.pdf"
    assert existing.read_bytes() == b"old"
    assert Path(renamed).read_bytes() == b"new"


def test_source_reference_is_absolute_and_reports_workspace_format(tmp_path):
    workspace = tmp_path / "Novel"
    raw = tmp_path / "raw" / "Novel.pdf"

    pointer = write_workspace_source_reference(workspace, raw)

    assert Path(pointer) == workspace / "source_epub.txt"
    assert Path(read_workspace_source_path(workspace)).is_absolute()
    assert Path(read_workspace_source_path(workspace)) == raw.resolve()
    assert workspace_source_format(workspace) == "PDF"


def test_gui_renames_at_selection_and_engine_uses_plain_stem_output():
    root = Path(__file__).resolve().parents[1]
    gui_source = (root / "src" / "translator_gui.py").read_text(encoding="utf-8")
    engine_source = (root / "src" / "TransateKRtoEN.py").read_text(encoding="utf-8")

    assert "self._rename_input_for_existing_workspace_collision(path)" in gui_source
    assert "resolve_source_aware_workspace" not in gui_source
    assert "resolve_source_aware_workspace" not in engine_source
    assert "write_workspace_source_reference(out, input_path)" in engine_source


def test_parallel_pair_open_output_creates_and_opens_raw_glossary_folder(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.delenv("OUTPUT_DIRECTORY", raising=False)
    from translator_gui import TranslatorGUI

    output_root = tmp_path / "configured output"
    raw_path = tmp_path / "books" / "Raw Novel.epub"
    gui = TranslatorGUI.__new__(TranslatorGUI)
    gui.config = {"output_directory": str(output_root)}
    gui._parallel_epub_pair_source = {"raw_path": str(raw_path)}
    gui._parallel_epub_pair_is_selected = lambda: True
    opened = []
    logs = []
    gui._open_folder_in_file_manager = opened.append
    gui.append_log = logs.append

    TranslatorGUI.open_output_folder(gui)

    expected = output_root / "Glossary" / "Raw Novel"
    assert expected.is_dir()
    assert opened == [str(expected.resolve())]
    assert "Parallel EPUB Pair glossary folder" in logs[-1]


def test_parallel_pair_mapping_sidecar_is_saved_in_raw_glossary_folder(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.delenv("OUTPUT_DIRECTORY", raising=False)
    from translator_gui import TranslatorGUI

    output_root = tmp_path / "configured output"
    raw_path = tmp_path / "books" / "Raw Novel.epub"
    translated_path = tmp_path / "books" / "Translated Novel.epub"
    gui = TranslatorGUI.__new__(TranslatorGUI)
    gui.config = {"output_directory": str(output_root)}
    selection = {
        "version": 1,
        "raw_path": str(raw_path.resolve()),
        "translated_path": str(translated_path.resolve()),
        "mapping": [
            {
                "raw_index": 1,
                "translated_index": 4,
                "raw_filename": "Text/chapter0002.xhtml",
                "translated_filename": "Text/0002_Chapter.xhtml",
            }
        ],
        "wrapper_prompt": "{raw_text}\n{translated_text}",
        "system_prompt": "Use established terms.",
        "profile_name": "Parallel EPUB Glossary",
    }

    saved_path = TranslatorGUI._write_parallel_epub_mapping_sidecar(
        gui, selection
    )

    expected = (
        output_root
        / "Glossary"
        / "Raw Novel"
        / "Raw Novel_parallel_epub_mapping.json"
    )
    assert Path(saved_path) == expected.resolve()
    assert json.loads(expected.read_text(encoding="utf-8")) == selection
    assert TranslatorGUI._read_parallel_epub_mapping_sidecar(
        gui, str(raw_path)
    ) == selection


def test_parallel_pair_glossary_only_notice_centers_and_widens_ok(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QDialogButtonBox, QMessageBox
    from translator_gui import TranslatorGUI

    app = QApplication.instance() or QApplication([])
    message_box = TranslatorGUI._create_parallel_epub_glossary_only_notice(None)
    button_box = message_box.findChild(QDialogButtonBox)
    ok_button = message_box.button(QMessageBox.Ok)

    assert app is not None
    assert button_box is not None and button_box.centerButtons()
    assert ok_button is not None
    assert ok_button.minimumWidth() == 140
    assert ok_button.minimumHeight() == 44
    message_box.deleteLater()


def test_parallel_epub_mapper_is_cached_and_not_deleted_on_close(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    import parallel_epub_glossary
    from PySide6.QtCore import Qt
    from translator_gui import TranslatorGUI

    created = []

    class FakeParallelDialog:
        def __init__(self, parent, **kwargs):
            self.parent = parent
            self.kwargs = kwargs
            self.attributes = {}
            created.append(self)

        def setAttribute(self, attribute, enabled=True):
            self.attributes[attribute] = enabled

        @staticmethod
        def windowTitle():
            return "Parallel EPUB Pair"

    monkeypatch.setattr(
        parallel_epub_glossary, "ParallelEpubPairDialog", FakeParallelDialog
    )
    gui = SimpleNamespace(
        config={},
        _load_parallel_epub_chapters=lambda path: path,
        _is_special_file=lambda filename: False,
    )

    first = TranslatorGUI._get_or_create_parallel_epub_pair_dialog(gui)
    second = TranslatorGUI._get_or_create_parallel_epub_pair_dialog(gui)

    assert first is second
    assert created == [first]
    assert first.attributes[Qt.WA_DeleteOnClose] is False


def test_clear_file_selection_also_clears_cached_parallel_mapper(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from translator_gui import TranslatorGUI

    class Entry:
        def clear(self):
            pass

        def setText(self, _text):
            pass

        def setToolTip(self, _text):
            pass

    class PairDialog:
        cleared = False

        def clear_selection(self):
            self.cleared = True

    dialog = PairDialog()
    gui = TranslatorGUI.__new__(TranslatorGUI)
    gui.config = {
        "parallel_epub_pair_selection": {"mapping": [{}]},
        "parallel_epub_glossary_last_raw_epub": "C:/Books/raw.epub",
        "parallel_epub_glossary_last_translated_epub": "C:/Books/translated.epub",
    }
    gui._parallel_epub_pair_dialog = dialog
    gui._parallel_epub_pair_source = None
    gui.entry_epub = Entry()
    gui.selected_files = ["paired.epub"]
    gui.file_path = "paired.epub"
    gui.current_file_index = 0
    gui._subtitle_zip_output_groups = {}
    gui.save_config = lambda **_kwargs: None
    gui.append_log = lambda _message: None

    TranslatorGUI.clear_file_selection(gui)

    assert dialog.cleared
    assert "parallel_epub_pair_selection" not in gui.config
    assert "parallel_epub_glossary_last_raw_epub" not in gui.config
    assert "parallel_epub_glossary_last_translated_epub" not in gui.config
    assert gui.selected_files == []


def test_selecting_replacement_input_clears_cached_parallel_mapper(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from translator_gui import TranslatorGUI

    class PairDialog:
        cleared = False

        def clear_selection(self):
            self.cleared = True

    dialog = PairDialog()
    gui = TranslatorGUI.__new__(TranslatorGUI)
    gui.config = {
        "parallel_epub_pair_selection": {"mapping": [{}]},
        "parallel_epub_glossary_last_raw_epub": "C:/Books/raw.epub",
        "parallel_epub_glossary_last_translated_epub": "C:/Books/translated.epub",
    }
    gui._parallel_epub_pair_dialog = dialog
    gui._parallel_epub_pair_source = None

    def stop_after_pair_reset(_paths):
        raise RuntimeError("selection continued after pair reset")

    gui._normalize_windows_input_filenames = stop_after_pair_reset
    try:
        TranslatorGUI._handle_file_selection(gui, ["C:/Books/other.epub"])
    except RuntimeError as exc:
        assert str(exc) == "selection continued after pair reset"
    else:
        raise AssertionError("replacement selection did not continue")

    assert dialog.cleared
    assert "parallel_epub_pair_selection" not in gui.config
    assert "parallel_epub_glossary_last_raw_epub" not in gui.config
    assert "parallel_epub_glossary_last_translated_epub" not in gui.config
