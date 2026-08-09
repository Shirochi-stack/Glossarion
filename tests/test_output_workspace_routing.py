from pathlib import Path


from output_workspace import (
    read_workspace_source_path,
    resolve_source_aware_workspace,
    source_format_label,
    workspace_source_format,
    write_workspace_source_reference,
)


def test_source_format_labels_are_limited_to_collision_sensitive_inputs():
    assert source_format_label("Novel.EPUB") == "EPUB"
    assert source_format_label("Novel.PdF") == "PDF"
    assert source_format_label("Novel.txt") == "TXT"
    assert source_format_label("Novel.md") == ""
    assert resolve_source_aware_workspace("Novel.pdf", "") == ""


def test_same_named_pdf_is_routed_away_from_existing_epub_workspace(tmp_path):
    workspace = tmp_path / "Same Name"
    write_workspace_source_reference(workspace, tmp_path / "raw" / "Same Name.epub")

    resolved = resolve_source_aware_workspace(
        str(tmp_path / "updated" / "Same Name.pdf"), str(workspace)
    )

    assert Path(resolved) == tmp_path / "Same Name_PDF"


def test_same_named_epub_is_routed_away_from_existing_pdf_workspace(tmp_path):
    workspace = tmp_path / "Same Name"
    write_workspace_source_reference(workspace, tmp_path / "raw" / "Same Name.pdf")

    resolved = resolve_source_aware_workspace(
        str(tmp_path / "updated" / "Same Name.epub"), str(workspace)
    )

    assert Path(resolved) == tmp_path / "Same Name_EPUB"


def test_txt_gets_its_own_workspace_and_matching_updates_reuse_it(tmp_path):
    workspace = tmp_path / "Same Name"
    write_workspace_source_reference(workspace, tmp_path / "Same Name.epub")
    txt_workspace = Path(resolve_source_aware_workspace(
        str(tmp_path / "Same Name.txt"), str(workspace)
    ))
    write_workspace_source_reference(txt_workspace, tmp_path / "Same Name.txt")

    updated = resolve_source_aware_workspace(
        str(tmp_path / "new location" / "Same Name.txt"), str(workspace)
    )

    assert txt_workspace == tmp_path / "Same Name_TXT"
    assert Path(updated) == txt_workspace


def test_same_format_reuses_unsuffixed_workspace(tmp_path):
    workspace = tmp_path / "Novel"
    write_workspace_source_reference(workspace, tmp_path / "old" / "Novel.pdf")

    resolved = resolve_source_aware_workspace(
        str(tmp_path / "new" / "Novel.pdf"), str(workspace)
    )

    assert Path(resolved) == workspace


def test_matching_format_specific_workspace_is_reused(tmp_path):
    workspace = tmp_path / "Novel"
    pdf_workspace = tmp_path / "Novel_PDF"
    write_workspace_source_reference(workspace, tmp_path / "Novel.epub")
    write_workspace_source_reference(pdf_workspace, tmp_path / "Novel.pdf")

    resolved = resolve_source_aware_workspace(
        str(tmp_path / "updated" / "Novel.pdf"), str(workspace)
    )

    assert Path(resolved) == pdf_workspace


def test_conflicting_format_specific_workspace_uses_numbered_fallback(tmp_path):
    workspace = tmp_path / "Novel"
    pdf_workspace = tmp_path / "Novel_PDF"
    write_workspace_source_reference(workspace, tmp_path / "Novel.epub")
    write_workspace_source_reference(pdf_workspace, tmp_path / "Novel.txt")

    resolved = resolve_source_aware_workspace(
        str(tmp_path / "Novel.pdf"), str(workspace)
    )

    assert Path(resolved) == tmp_path / "Novel_PDF_2"


def test_source_reference_is_absolute_and_reports_workspace_format(tmp_path):
    workspace = tmp_path / "Novel"
    raw = tmp_path / "raw" / "Novel.pdf"

    pointer = write_workspace_source_reference(workspace, raw)

    assert Path(pointer) == workspace / "source_epub.txt"
    assert Path(read_workspace_source_path(workspace)).is_absolute()
    assert Path(read_workspace_source_path(workspace)) == raw.resolve()
    assert workspace_source_format(workspace) == "PDF"


def test_gui_and_engine_both_use_shared_workspace_resolver():
    root = Path(__file__).resolve().parents[1]
    gui_source = (root / "src" / "translator_gui.py").read_text(encoding="utf-8")
    engine_source = (root / "src" / "TransateKRtoEN.py").read_text(encoding="utf-8")

    assert "resolve_source_aware_workspace(input_file, default_output)" in gui_source
    assert "out = resolve_source_aware_workspace(input_path, unresolved_out)" in engine_source
    assert "write_workspace_source_reference(out, input_path)" in engine_source
