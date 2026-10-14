import json
import pytest

from reader_overlay import make_epub_overlay_provider


def _progress(tmp_path, chapters, **kwargs):
    path = tmp_path / "translation_progress.json"
    path.write_text(json.dumps({"chapters": chapters, **kwargs}), encoding="utf-8")
    return path


def test_provider_discovers_new_translations_without_ui_refresh(tmp_path):
    _progress(tmp_path, {})
    provider = make_epub_overlay_provider(
        tmp_path, ["OEBPS/chapter0436.xhtml", "OEBPS/chapter0437.xhtml"],
    )
    assert provider() == ({}, [])

    translated = tmp_path / "response_chapter0436.html"
    translated.write_text("<h1>Translated heading</h1>", encoding="utf-8")
    _progress(tmp_path, {"hash": {
        "original_basename": "chapter0436", "output_file": translated.name,
        "status": "completed",
    }})
    (tmp_path / "translated_images").mkdir()

    assert provider() == ({
        "chapter0436.xhtml": {"path": str(translated), "status": "completed"},
    }, [str(tmp_path / "translated_images")])


@pytest.mark.parametrize("source_field", [
    "original_basename", "original_filename", "chapter_file", "source_filename",
    "filename", "key", "output_file",
])
def test_provider_matches_fresh_progress_filenames_and_preserves_qa_status(
    tmp_path, source_field,
):
    translated = tmp_path / "response_chapter0436.xhtml.html"
    translated.write_text("<p>Translated body</p>", encoding="utf-8")
    entry = {"status": "qa_failed", "output_file": translated.name}
    key = "OEBPS\\chapter0436.xhtml" if source_field == "key" else "hash"
    if source_field not in ("key", "output_file"):
        entry[source_field] = "OEBPS\\chapter0436.xhtml"
    _progress(tmp_path, {key: entry})
    provider = make_epub_overlay_provider(tmp_path, ["OEBPS/chapter0436.xhtml"])

    assert provider()[0] == {
        "chapter0436.xhtml": {"path": str(translated), "status": "qa_failed"},
    }


def test_provider_uses_recorded_output_path_and_reads_updated_status(tmp_path):
    translated = tmp_path / "custom.xhtml"
    translated.write_text("<p>Translated body</p>", encoding="utf-8")
    entry = {"original_basename": "chapter0436.xhtml",
             "output_file": str(translated), "status": "qa_failed"}
    _progress(tmp_path, {"hash": entry})
    provider = make_epub_overlay_provider(tmp_path, ["chapter0436.xhtml"])
    assert provider()[0]["chapter0436.xhtml"]["status"] == "qa_failed"

    entry["status"] = "completed"
    _progress(tmp_path, {"hash": entry})
    assert provider()[0]["chapter0436.xhtml"] == {
        "path": str(translated), "status": "completed",
    }


@pytest.mark.parametrize("broken_snapshot", ["{", "[]", '{"chapters": []}', None])
def test_provider_ignores_broken_progress_and_recovers(tmp_path, broken_snapshot):
    progress_file = _progress(tmp_path, {})
    translated = tmp_path / "response_chapter0436.html"
    translated.write_text("<p>Translated body</p>", encoding="utf-8")
    provider = make_epub_overlay_provider(tmp_path, ["chapter0436.xhtml"])
    expected = provider()
    if broken_snapshot is None:
        progress_file.unlink()
    else:
        progress_file.write_text(broken_snapshot, encoding="utf-8")
    assert provider() is None

    _progress(tmp_path, {})
    assert provider() == expected


def test_provider_discovers_files_before_progress_exists_and_tracks_deletion(tmp_path):
    provider = make_epub_overlay_provider(tmp_path, ["chapter0436.xhtml"])
    assert provider() == ({}, [])
    translated = tmp_path / "response_chapter0436.html"
    translated.write_text("<p>Translated body</p>", encoding="utf-8")
    assert "chapter0436.xhtml" in provider()[0]
    translated.unlink()
    assert provider() == ({}, [])


def test_provider_preserves_initial_custom_path_without_stale_title(tmp_path):
    translated = tmp_path / "custom.xhtml"
    translated.write_text("<h1>Fresh heading</h1>", encoding="utf-8")
    provider = make_epub_overlay_provider(
        tmp_path, ["chapter0436.xhtml"], initial_overlay={
            "chapter0436.xhtml": {"path": str(translated),
                                  "title": "Old heading", "status": "qa_failed"},
        },
    )
    assert provider()[0] == {
        "chapter0436.xhtml": {"path": str(translated), "status": "qa_failed"},
    }


def test_provider_does_not_mark_incomplete_chunks_completed(tmp_path):
    translated = tmp_path / "response_chapter0436.html"
    translated.write_text("<p>Translated body</p>", encoding="utf-8")
    _progress(tmp_path, {"hash": {
        "original_basename": "chapter0436.xhtml", "status": "completed",
    }}, chapter_chunks={"hash": {
        "schema_version": 2, "total": 2, "chunks": {"1": "Translated part"}, "entries": {
            "1": {"index": 1, "status": "completed"},
            "2": {"index": 2, "status": "pending"},
        },
    }})
    provider = make_epub_overlay_provider(tmp_path, ["chapter0436.xhtml"])
    assert provider()[0]["chapter0436.xhtml"]["status"] == "pending"
