import json
import re
import zipfile
from pathlib import Path

import pytest

from subtitle_processor import (
    DEFAULT_SUBTITLE_TRANSLATION_PROMPT,
    SubtitleArchiveError,
    convert_subtitle,
    convert_subtitle_bundle,
    convert_subtitle_bundle_source,
    extract_subtitle_archive,
    extract_subtitle_bundle_to_chapters,
    extract_subtitle_to_chapters,
    grouped_subtitle_output_layout,
    plan_subtitle_archive_outputs,
)


def _translated_batch(chapter, translations):
    source_records = json.loads(chapter["body"])
    return json.dumps(
        [
            {"id": record["id"], "target": translations[record["id"]](record["source"])}
            for record in source_records
        ],
        ensure_ascii=False,
        indent=2,
    )


def test_srt_round_trip_preserves_timing_numbers_tags_and_crlf(tmp_path):
    source = tmp_path / "sample.srt"
    source.write_bytes(
        (
            "1\r\n"
            "00:00:01,250 --> 00:00:03,500\r\n"
            "<i>Hello</i>\r\n"
            "world!\r\n"
            "\r\n"
            "2\r\n"
            "00:00:04,000 --> 00:00:05,000 position:50%\r\n"
            "Goodbye\r\n"
        ).encode("utf-8")
    )
    output_dir = tmp_path / "out"

    result = extract_subtitle_to_chapters(str(source), str(output_dir))
    chapters = json.loads(Path(result["chapters_path"]).read_text(encoding="utf-8"))
    assert result["segments"] == 2
    assert len(chapters) == 1
    assert chapters[0]["subtitle_batch"] is True

    translated = _translated_batch(
        chapters[0],
        {
            "1": lambda text: text.replace("Hello", "Bonjour").replace("world!", "monde !"),
            "2": lambda text: text.replace("Goodbye", "Au revoir"),
        },
    )
    (output_dir / "response_section_1.txt").write_text(translated, encoding="utf-8")

    converted = convert_subtitle(str(output_dir))
    output_bytes = Path(converted["output_path"]).read_bytes()
    output = output_bytes.decode("utf-8")
    assert converted["updated"] == 2
    assert b"\r\n" in output_bytes
    assert "1\r\n00:00:01,250 --> 00:00:03,500\r\n" in output
    assert "<i>Bonjour</i>\r\nmonde !" in output
    assert "00:00:04,000 --> 00:00:05,000 position:50%" in output
    assert output.endswith("Au revoir\r\n")


def test_ass_round_trip_changes_only_dialogue_text(tmp_path):
    source_text = (
        "\ufeff[Script Info]\n"
        "Title: Example\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize\n"
        "Style: Default,Arial,24\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        r"Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,{\an8}Hello\Nworld"
        "\n"
        "Comment: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Do not translate\n"
        r"Dialogue: 0,0:00:04.00,0:00:05.00,Default,,0,0,0,,{\p1}m 0 0 l 10 10"
        "\n"
    )
    source = tmp_path / "sample.ass"
    source.write_bytes(source_text.encode("utf-8"))
    output_dir = tmp_path / "out"

    result = extract_subtitle_to_chapters(str(source), str(output_dir))
    chapters = json.loads(Path(result["chapters_path"]).read_text(encoding="utf-8"))
    assert result["segments"] == 1

    translated = _translated_batch(
        chapters[0],
        {
            "1": lambda text: text.replace("Hello", "Bonjour").replace("world", "monde"),
        },
    )
    (output_dir / "section_1.txt").write_text(translated, encoding="utf-8")

    converted = convert_subtitle(str(output_dir))
    output = Path(converted["output_path"]).read_text(encoding="utf-8-sig")
    assert converted["updated"] == 1
    assert r"{\an8}Bonjour\Nmonde" in output
    assert "Comment: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Do not translate" in output
    assert r"{\p1}m 0 0 l 10 10" in output
    assert "Style: Default,Arial,24" in output


def test_missing_subtitle_placeholder_does_not_create_final_file(tmp_path):
    source = tmp_path / "tagged.srt"
    source.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n<i>Hello</i>\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    result = extract_subtitle_to_chapters(str(source), str(output_dir))
    chapters = json.loads(Path(result["chapters_path"]).read_text(encoding="utf-8"))
    record = json.loads(chapters[0]["body"])[0]
    invalid_output = json.dumps(
        [{"id": record["id"], "target": "Bonjour"}],
        ensure_ascii=False,
    )
    (output_dir / "response_section_1.txt").write_text(
        invalid_output, encoding="utf-8"
    )

    converted = convert_subtitle(str(output_dir))
    assert converted["updated"] == 0
    assert converted["skipped"] == 1
    assert converted["ready"] is False
    assert Path(converted["output_path"]).exists() is False


def test_subtitle_zip_extracts_all_srt_ass_and_ignores_other_members(tmp_path):
    archive_path = tmp_path / "season.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "episode 01/one.srt",
            "1\n00:00:00,000 --> 00:00:01,000\nHello\n",
        )
        archive.writestr(
            "episode 02/two.ass",
            "[Events]\nFormat: Start, End, Text\n"
            "Dialogue: 0:00:00.00,0:00:01.00,Hello\n",
        )
        archive.writestr("notes/readme.txt", "not a subtitle")

    extraction_root = tmp_path / "extracted"
    result = extract_subtitle_archive(
        str(archive_path),
        str(extraction_root),
    )

    assert result["subtitle_count"] == 2
    assert result["ignored_count"] == 1
    assert {Path(path).suffix for path in result["files"]} == {".srt", ".ass"}
    for extracted_path in result["files"]:
        resolved = Path(extracted_path).resolve()
        assert resolved.is_relative_to(extraction_root.resolve())
        assert resolved.is_file()


def test_subtitle_zip_rejects_path_traversal_without_writing_outside_root(tmp_path):
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "../escape.srt",
            "1\n00:00:00,000 --> 00:00:01,000\nHello\n",
        )

    extraction_root = tmp_path / "extracted"
    with pytest.raises(SubtitleArchiveError, match="Unsafe subtitle member path"):
        extract_subtitle_archive(
            str(archive_path),
            str(extraction_root),
        )

    assert not (tmp_path / "escape.srt").exists()


def test_subtitle_zip_outputs_share_archive_folder_with_isolated_work_dirs(tmp_path):
    archive_path = tmp_path / "My Show.zip"
    extracted_paths = [
        tmp_path / "extract" / "episode_1" / "dialogue.srt",
        tmp_path / "extract" / "episode_2" / "dialogue.srt",
        tmp_path / "extract" / "episode_3" / "final.ass",
    ]

    plan = plan_subtitle_archive_outputs(
        str(archive_path),
        [str(path) for path in extracted_paths],
        str(tmp_path / "outputs"),
        work_base_dir=str(tmp_path / "temporary_work"),
    )

    assert len(plan) == 3
    output_dirs = {info["output_dir"] for info in plan.values()}
    assert output_dirs == {str((tmp_path / "outputs" / "My Show").resolve())}
    output_paths = [Path(info["output_path"]) for info in plan.values()]
    assert len({path.name.casefold() for path in output_paths}) == 3
    assert [path.name for path in output_paths] == [
        "dialogue.srt",
        "dialogue_2.srt",
        "final.ass",
    ]
    assert all(path.parent == tmp_path / "outputs" / "My Show" for path in output_paths)

    work_dirs = []
    for source_path, info in zip(extracted_paths, plan.values()):
        layout = grouped_subtitle_output_layout(
            str(source_path),
            info["output_dir"],
            info["output_path"],
            work_dir=info["work_dir"],
        )
        work_dirs.append(Path(layout["work_dir"]))
        assert Path(layout["output_path"]).parent == tmp_path / "outputs" / "My Show"
        assert work_dirs[-1].parent == tmp_path / "temporary_work"
    assert len(set(work_dirs)) == 3


def test_grouped_subtitle_output_rejects_file_outside_archive_folder(tmp_path):
    with pytest.raises(ValueError, match="must stay inside"):
        grouped_subtitle_output_layout(
            str(tmp_path / "source.srt"),
            str(tmp_path / "group"),
            str(tmp_path / "outside.srt"),
        )


def test_grouped_subtitle_round_trip_writes_final_file_to_archive_folder(tmp_path):
    source = tmp_path / "extracted" / "episode.srt"
    source.parent.mkdir()
    source.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nHello\n",
        encoding="utf-8",
    )
    plan = plan_subtitle_archive_outputs(
        str(tmp_path / "Season 1.zip"),
        [str(source)],
        str(tmp_path / "outputs"),
        work_base_dir=str(tmp_path / "temporary_work"),
    )
    info = next(iter(plan.values()))
    layout = grouped_subtitle_output_layout(
        str(source),
        info["output_dir"],
        info["output_path"],
        work_dir=info["work_dir"],
    )

    extraction = extract_subtitle_to_chapters(
        str(source),
        layout["work_dir"],
    )
    chapter = json.loads(
        Path(extraction["chapters_path"]).read_text(encoding="utf-8")
    )[0]
    translated = _translated_batch(
        chapter,
        {"1": lambda text: text.replace("Hello", "Bonjour")},
    )
    (Path(layout["work_dir"]) / "response_section_1.txt").write_text(
        translated,
        encoding="utf-8",
    )

    converted = convert_subtitle(
        layout["work_dir"],
        output_path=layout["output_path"],
    )
    final_path = Path(converted["output_path"])
    assert final_path.parent == tmp_path / "outputs" / "Season 1"
    assert final_path.name == "episode.srt"
    assert "Bonjour" in final_path.read_text(encoding="utf-8")
    assert not (Path(layout["work_dir"]) / "episode.srt").exists()


def test_subtitle_bundle_exposes_files_as_parallel_chapters_and_rebuilds_each(tmp_path):
    first = tmp_path / "extracted" / "episode_1.srt"
    second = tmp_path / "extracted" / "episode_2.srt"
    first.parent.mkdir()
    first.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nHello\n",
        encoding="utf-8",
    )
    second.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nGoodbye\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "bundle_work"
    first_output = tmp_path / "outputs" / "Show" / "episode_1.srt"
    second_output = tmp_path / "outputs" / "Show" / "episode_2.srt"

    extraction = extract_subtitle_bundle_to_chapters(
        [str(first), str(second)],
        str(output_dir),
        output_paths={
            str(first): str(first_output),
            str(second): str(second_output),
        },
    )
    chapters = json.loads(
        Path(extraction["chapters_path"]).read_text(encoding="utf-8")
    )

    assert extraction["source_count"] == 2
    assert len(chapters) == 2
    assert [chapter["num"] for chapter in chapters] == [1, 2]
    assert [chapter["filename"] for chapter in chapters] == [
        "section_1.txt",
        "section_2.txt",
    ]
    assert all(chapter["subtitle_bundle"] is True for chapter in chapters)
    assert chapters[0]["original_basename"] == "episode_1.srt"
    assert chapters[0]["subtitle_source_batch_num"] == 1
    assert chapters[0]["subtitle_source_batch_count"] == 1
    assert chapters[0]["subtitle_progress_id"] == str(first_output)
    assert chapters[1]["subtitle_progress_id"] == str(second_output)

    translated_first = _translated_batch(
        chapters[0],
        {"1": lambda text: text.replace("Hello", "Bonjour")},
    )
    translated_second = _translated_batch(
        chapters[1],
        {"1": lambda text: text.replace("Goodbye", "Au revoir")},
    )
    (output_dir / "response_section_1.txt").write_text(
        translated_first,
        encoding="utf-8",
    )
    (output_dir / "response_section_2.txt").write_text(
        translated_second,
        encoding="utf-8",
    )

    converted = convert_subtitle_bundle(str(output_dir))
    assert converted["files"] == 2
    assert converted["updated"] == 2
    assert "Bonjour" in first_output.read_text(encoding="utf-8")
    assert "Au revoir" in second_output.read_text(encoding="utf-8")


def test_bundle_writes_each_file_when_its_batches_finish(tmp_path, monkeypatch):
    source = tmp_path / "extracted" / "episode.srt"
    source.parent.mkdir()
    source.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nHello\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nGoodbye\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SUBTITLE_AVAILABLE_TOKENS", "1000")
    monkeypatch.setattr(
        "subtitle_processor._count_tokens",
        lambda text: 2000 if text.count('"id"') > 1 else 10,
    )
    output_dir = tmp_path / "bundle_work"
    final_output = tmp_path / "outputs" / "Show" / "episode.srt"
    extraction = extract_subtitle_bundle_to_chapters(
        [str(source)],
        str(output_dir),
        output_paths={str(source): str(final_output)},
    )
    chapters = json.loads(
        Path(extraction["chapters_path"]).read_text(encoding="utf-8")
    )
    assert len(chapters) == 2

    first_response = _translated_batch(
        chapters[0],
        {"1": lambda text: text.replace("Hello", "Bonjour")},
    )
    (output_dir / "response_section_1.txt").write_text(
        first_response,
        encoding="utf-8",
    )
    (output_dir / "translation_progress.json").write_text(
        json.dumps(
            {
                "chapters": {
                    "first": {
                        "status": "completed",
                        "output_file": "response_section_1.txt",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    waiting = convert_subtitle_bundle_source(str(output_dir), 1)
    assert waiting["ready"] is False
    assert final_output.exists() is False

    second_response = _translated_batch(
        chapters[1],
        {"2": lambda text: text.replace("Goodbye", "Au revoir")},
    )
    (output_dir / "response_section_2.txt").write_text(
        second_response,
        encoding="utf-8",
    )
    (output_dir / "translation_progress.json").write_text(
        json.dumps(
            {
                "chapters": {
                    "first": {
                        "status": "completed",
                        "output_file": "response_section_1.txt",
                    },
                    "second": {
                        "status": "completed",
                        "output_file": "response_section_2.txt",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    completed = convert_subtitle_bundle_source(str(output_dir), 1)
    assert completed["ready"] is True
    assert completed["created"] is True
    assert "Bonjour" in final_output.read_text(encoding="utf-8")
    assert "Au revoir" in final_output.read_text(encoding="utf-8")

    final_pass = convert_subtitle_bundle(str(output_dir))
    assert final_pass["files"] == 1
    assert final_pass["results"][0]["already_exists"] is True


def test_subtitle_progress_uses_source_file_identity(tmp_path, monkeypatch):
    from TransateKRtoEN import ProgressManager, _direct_text_html_source_name

    source = tmp_path / "episode_07.srt"
    output = tmp_path / "Show" / "episode_07.srt"
    mirror_path = output.parent / "translation_progress.json"
    monkeypatch.setenv(
        "SUBTITLE_PROGRESS_MIRROR_FILE",
        str(mirror_path),
    )
    chapter = {
        "num": 12,
        "filename": "section_12.txt",
        "source_file": str(source),
        "original_basename": source.name,
        "subtitle_batch": True,
        "subtitle_progress_id": str(output),
        "subtitle_source_batch_num": 2,
        "subtitle_source_batch_count": 3,
        "subtitle_bundle_source_index": 7,
        "subtitle_output_file": str(output),
    }
    manager = ProgressManager(str(tmp_path / "work"))

    key = manager._get_chapter_key(
        12,
        "response_section_12.txt",
        chapter,
        "hash",
    )
    assert key == "subtitle:episode_07.srt:2"
    assert _direct_text_html_source_name(chapter) == "episode_07.srt"

    manager.update(
        11,
        12,
        "hash",
        "response_section_12.txt",
        status="in_progress",
        chapter_obj=chapter,
    )
    tracked = manager.prog["chapters"][key]
    assert tracked["original_basename"] == "episode_07.srt"
    assert tracked["subtitle_source_file"] == str(source)
    assert tracked["subtitle_source_batch_num"] == 2
    assert tracked["subtitle_source_batch_count"] == 3
    assert tracked["subtitle_output_file"] == str(output)
    assert tracked["subtitle_progress_key"] == key

    manager.migrate_to_content_hash([chapter])
    assert key in manager.prog["chapters"]
    file_summary = manager.prog["subtitle_files"][output.name]
    assert file_summary["source_file"] == str(source)
    assert file_summary["output_file"] == str(output)
    assert file_summary["total_batches"] == 3
    assert file_summary["in_progress_batches"] == 1

    mirrored = json.loads(mirror_path.read_text(encoding="utf-8"))
    assert key in mirrored["chapters"]
    assert mirrored["chapters"][key]["output_file"] == str(output)
    assert (
        mirrored["chapters"][key]["batch_output_file"]
        == "response_section_12.txt"
    )
    assert output.name in mirrored["subtitle_files"]


def test_subtitle_watchdog_keeps_source_filename():
    import unified_api_client

    unified_api_client._api_watchdog_reset()
    try:
        unified_api_client._api_watchdog_started(
            "translation",
            request_id="subtitle-source-test",
            chapter=4,
            chunk=1,
            total_chunks=1,
            queued=True,
            source_file="episode_04.srt",
        )
        entries = unified_api_client.get_api_watchdog_state()[
            "in_flight_entries"
        ]
        entry = next(
            item
            for item in entries
            if item.get("request_id") == "subtitle-source-test"
        )
        assert entry["source_file"] == "episode_04.srt"
        assert "episode_04.srt" in entry["label"]
    finally:
        unified_api_client._api_watchdog_reset()


def test_zip_selection_has_explicit_automatic_glossary_cleanup():
    gui_source = (
        Path(__file__).resolve().parents[1] / "src" / "translator_gui.py"
    ).read_text(encoding="utf-8")
    helper_start = gui_source.index(
        "def _clear_automatic_glossary_for_non_epub_selection"
    )
    helper_end = gui_source.index(
        "def _resolve_translation_output_dir", helper_start
    )
    helper_body = gui_source[helper_start:helper_end]

    assert 'path.lower().endswith(".epub")' in helper_body
    assert '"manual_glossary_manually_loaded", False' in helper_body
    assert "self.manual_glossary_path = None" in helper_body
    assert 'os.environ.pop("MANUAL_GLOSSARY", None)' in helper_body
    assert (
        "self._clear_automatic_glossary_for_non_epub_selection(processed_paths)"
        in gui_source
    )
    assert (
        "self._clear_automatic_glossary_for_non_epub_selection(\n"
        "            self.selected_files\n"
        "        )"
        in gui_source
    )


def test_subtitle_prompt_profile_is_built_in_and_mirrored():
    source_root = Path(__file__).resolve().parents[1] / "src"
    gui_source = (source_root / "translator_gui.py").read_text(encoding="utf-8")
    app_source = (source_root / "app.py").read_text(encoding="utf-8")
    discord_source = (source_root / "discord_bot.py").read_text(encoding="utf-8")

    assert "concise, natural spoken dialogue" in DEFAULT_SUBTITLE_TRANSLATION_PROMPT
    assert "[[SUB_TAG_000001_0000]]" in DEFAULT_SUBTITLE_TRANSLATION_PROMPT
    assert '"Subtitle Translation": DEFAULT_SUBTITLE_TRANSLATION_PROMPT' in gui_source
    assert re.search(
        r"protected = \{[\s\S]*?Subtitle Translation[\s\S]*?\}",
        gui_source,
    )
    assert re.search(
        r"always_include_profiles = \[[\s\S]*?Subtitle Translation[\s\S]*?\]",
        gui_source,
    )
    assert '"Subtitle Translation": DEFAULT_SUBTITLE_TRANSLATION_PROMPT' in app_source
    assert '"Subtitle Translation"' in discord_source


def test_subtitle_zip_grouping_is_exported_to_translation_backend():
    source_root = Path(__file__).resolve().parents[1] / "src"
    gui_source = (source_root / "translator_gui.py").read_text(encoding="utf-8")
    backend_source = (source_root / "TransateKRtoEN.py").read_text(encoding="utf-8")

    assert "'SUBTITLE_OUTPUT_GROUP_DIR'" in gui_source
    assert "'SUBTITLE_OUTPUT_FILE'" in gui_source
    assert "'SUBTITLE_WORK_DIR'" in gui_source
    assert "'SUBTITLE_BUNDLE_FILES_JSON'" in gui_source
    assert 'large_env.get_env(name, "")' in backend_source
    assert "extract_subtitle_bundle_to_chapters" in backend_source
    assert "convert_subtitle_bundle" in backend_source
    assert "if len(unique_output_paths) == 1:" in gui_source
    assert 'os.getenv("SUBTITLE_OUTPUT_GROUP_DIR"' in backend_source
    assert "output_path=grouped_subtitle_output_path" in backend_source


def test_translation_pipeline_injects_and_validates_subtitle_json_contract():
    from TransateKRtoEN import (
        _validate_sdlxliff_batch_output,
        _with_structured_batch_prompt,
    )

    chapter = {
        "subtitle_batch": True,
        "structured_translation_batch": True,
        "body": json.dumps(
            [{"id": "1", "source": "[[SUB_TAG_000001_0000]]Hello"}]
        ),
    }
    messages = _with_structured_batch_prompt(
        [
            {"role": "system", "content": "Translate naturally."},
            {"role": "user", "content": chapter["body"]},
        ],
        chapter,
    )
    assert "FORMAT OVERRIDE" in messages[0]["content"]
    assert "exactly id and target fields" in messages[0]["content"]
    assert "[[SUB_TAG_...]]" in messages[0]["content"]

    valid = json.dumps(
        [{"id": "1", "target": "[[SUB_TAG_000001_0000]]Bonjour"}]
    )
    invalid = json.dumps([{"id": "1", "target": "Bonjour"}])
    assert _validate_sdlxliff_batch_output(chapter, valid) == (True, None)
    assert _validate_sdlxliff_batch_output(chapter, invalid) == (
        False,
        "PLACEHOLDER_MISMATCH",
    )
