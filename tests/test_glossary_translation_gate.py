"""The optional gate covers every stage represented in glossary progress."""

import json

from glossary_translation_gate import glossary_complete, progress_path_for_source


def test_balanced_glossary_requires_chapters_minimal_pass_and_refinement(tmp_path):
    book_dir = tmp_path / "Glossary" / "Book"
    book_dir.mkdir(parents=True)
    source = tmp_path / "Book.epub"
    progress_path = book_dir / "Book_glossary_progress.json"
    glossary_path = book_dir / "Book_glossary.csv"
    glossary_path.write_text(
        "type,raw_name,translated_name\ncharacter,A,B\n", encoding="utf-8"
    )
    progress = {
        "chapter_count": 2,
        "completed": [0, 1],
        "minimal_pass": {"status": "completed"},
        "refinement": {"type::character": {"status": "completed"}},
    }
    config = {"glossary_refinement_enabled": True}

    def check():
        progress_path.write_text(json.dumps(progress), encoding="utf-8")
        return glossary_complete(
            str(progress_path), str(glossary_path), config,
            require_minimal_pass=True, is_epub=True,
        )

    assert check()[0]
    assert progress_path_for_source(str(source), str(tmp_path)) == str(progress_path)

    progress["completed"] = [0]
    assert not check()[0]
    progress["completed"] = [0, 1]

    progress["minimal_pass"]["status"] = "failed"
    assert "Minimal pass" in check()[1]
    progress["minimal_pass"]["status"] = "completed"

    progress["refinement"]["type::character"]["status"] = "failed"
    assert "refinement" in check()[1]
    progress["refinement"]["type::character"]["status"] = "completed"
    assert check()[0]


def test_gate_is_optional_for_refinement(tmp_path):
    progress_path = tmp_path / "glossary_progress.json"
    glossary_path = tmp_path / "Book_glossary.csv"
    progress_path.write_text(json.dumps({"chapter_count": 1, "completed": [0]}), encoding="utf-8")
    glossary_path.write_text("type,raw_name,translated_name\ncharacter,A,B\n", encoding="utf-8")
    assert glossary_complete(str(progress_path), str(glossary_path), {})[0]
