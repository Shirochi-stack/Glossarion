from pathlib import Path


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
