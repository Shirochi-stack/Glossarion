"""The minimal-pass refinement records its statuses in the book's <book>_glossary_progress.json."""

import json

import GlossaryManager


def test_minimal_refinement_uses_the_books_progress_and_carries_the_generic_one(tmp_path, monkeypatch):
    source = tmp_path / "Book.epub"
    source.write_bytes(b"x")
    out = tmp_path / "Book"
    out.mkdir()
    monkeypatch.setenv("EPUB_PATH", str(source))
    legacy = out / "glossary_progress.json"
    legacy.write_text(json.dumps({"refinement": {
        "type::character": {"status": "completed"},
        "type::term": {"status": "failed"},
    }}), encoding="utf-8")

    path = GlossaryManager._minimal_refinement_progress_file(str(out))

    assert path.endswith("Book_glossary_progress.json") and path != str(legacy)
    with open(path, encoding="utf-8") as handle:
        refinement = json.load(handle)["refinement"]
    assert refinement["type::character"]["status"] == "completed"
    assert refinement["type::term"]["status"] == "failed"

    # the book's own statuses win; carrying is once per missing key
    from glossary_refinement import update_refinement_progress
    update_refinement_progress(path, "type::term", {"status": "completed"})
    assert GlossaryManager._minimal_refinement_progress_file(str(out)) == path
    with open(path, encoding="utf-8") as handle:
        assert json.load(handle)["refinement"]["type::term"]["status"] == "completed"
