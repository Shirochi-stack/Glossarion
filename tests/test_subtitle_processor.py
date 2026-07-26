import json
import os
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


def test_subtitle_batch_packing_avoids_per_cue_full_rescans(monkeypatch):
    import subtitle_processor

    segments = [
        {
            "id": str(index),
            "source_text": f"Dialogue {index}",
        }
        for index in range(1, 1001)
    ]
    token_calls = []

    def count_records(payload):
        token_calls.append(len(payload))
        return payload.count('"id"')

    monkeypatch.setattr(subtitle_processor, "_count_tokens", count_records)
    batches = subtitle_processor._pack_batches(segments, 40)

    assert [len(batch) for batch in batches] == [40] * 25
    assert [
        segment["id"]
        for batch in batches
        for segment in batch
    ] == [str(index) for index in range(1, 1001)]
    assert len(token_calls) < 400


def test_subtitle_extraction_worker_count_is_bounded(monkeypatch):
    import subtitle_processor

    monkeypatch.setenv("SUBTITLE_EXTRACTION_WORKERS", "3")

    assert subtitle_processor._subtitle_extraction_worker_count(1) == 1
    assert subtitle_processor._subtitle_extraction_worker_count(2) == 2
    assert subtitle_processor._subtitle_extraction_worker_count(20) == 3


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
    (output_dir / "response_episode_1.txt").write_text(
        translated_first,
        encoding="utf-8",
    )
    (output_dir / "response_episode_2.txt").write_text(
        translated_second,
        encoding="utf-8",
    )

    converted = convert_subtitle_bundle(str(output_dir))
    assert converted["files"] == 2
    assert converted["updated"] == 2
    assert "Bonjour" in first_output.read_text(encoding="utf-8")
    assert "Au revoir" in second_output.read_text(encoding="utf-8")

    # A fresh temporary ZIP work directory may no longer have the old JSON
    # checkpoints. Matching completed progress can safely reuse final files.
    (output_dir / "response_episode_1.txt").unlink()
    (output_dir / "response_episode_2.txt").unlink()
    reused = convert_subtitle_bundle(
        str(output_dir),
        reuse_existing_source_indices={1, 2},
    )
    assert reused["success"] is True
    assert reused["files"] == 2
    assert reused["incomplete_files"] == 0
    assert all(result["already_exists"] for result in reused["results"])


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


def test_subtitle_checkpoint_names_are_unique_for_multiple_batches(monkeypatch):
    from TransateKRtoEN import FileUtilities

    monkeypatch.setenv("RETAIN_SOURCE_EXTENSION", "0")
    first_batch = {
        "filename": "section_1.txt",
        "original_basename": "第六期.srt",
        "subtitle_batch": True,
        "subtitle_source_batch_num": 1,
        "subtitle_source_batch_count": 2,
    }
    second_batch = dict(
        first_batch,
        subtitle_source_batch_num=2,
    )

    assert FileUtilities.create_chapter_filename(first_batch) == (
        "response_第六期_batch_1.txt"
    )
    assert FileUtilities.create_chapter_filename(second_batch) == (
        "response_第六期_batch_2.txt"
    )


def test_subtitle_progress_keys_preserve_unicode_identity(tmp_path):
    from TransateKRtoEN import ProgressManager

    manager = ProgressManager(str(tmp_path / "work"))
    first = {
        "subtitle_batch": True,
        "subtitle_progress_id": str(tmp_path / "第六期.srt"),
        "subtitle_source_batch_num": 1,
    }
    second = {
        "subtitle_batch": True,
        "subtitle_progress_id": str(tmp_path / "第七期.srt"),
        "subtitle_source_batch_num": 1,
    }

    first_key = manager._get_chapter_key(0, chapter_obj=first)
    second_key = manager._get_chapter_key(0, chapter_obj=second)

    assert first_key == "subtitle:第六期.srt:1"
    assert second_key == "subtitle:第七期.srt:1"
    assert first_key != second_key


def test_subtitle_batch_uses_assigned_index_not_digits_in_filename():
    from TransateKRtoEN import FileUtilities

    chapter = {
        "subtitle_batch": True,
        "num": 7,
        "filename": "section_7.txt",
        "original_basename": "Audience Affinity 100.srt",
    }

    assert FileUtilities.extract_actual_chapter_number(chapter) == 7


def test_retranslation_progress_builds_one_indexed_subtitle_row(tmp_path):
    from Retranslation_GUI import RetranslationMixin

    source = tmp_path / "Audience Affinity 100.ass"
    output = tmp_path / "Show" / source.name
    entries = [
        (
            "subtitle:Audience Affinity 100.ass:1",
            {
                "actual_num": 1,
                "status": "completed",
                "output_file": str(output),
                "subtitle_output_file": str(output),
                "subtitle_source_file": str(source),
                "subtitle_bundle_source_index": 7,
                "subtitle_source_batch_num": 1,
                "subtitle_source_batch_count": 2,
            },
        ),
        (
            "subtitle:Audience Affinity 100.ass:2",
            {
                "actual_num": 2,
                "status": "in_progress",
                "output_file": str(output),
                "subtitle_output_file": str(output),
                "subtitle_source_file": str(source),
                "subtitle_bundle_source_index": 7,
                "subtitle_source_batch_num": 2,
                "subtitle_source_batch_count": 2,
            },
        ),
    ]
    prog = {
        "subtitle_files": {
            source.name: {
                "source_file": str(source),
                "output_file": str(output),
                "status": "in_progress",
                "total_batches": 2,
                "completed_batches": 1,
            }
        }
    }
    mixin = RetranslationMixin.__new__(RetranslationMixin)

    row = mixin._build_subtitle_progress_row(prog, entries, str(output))

    assert row["is_subtitle"] is True
    assert row["is_special"] is False
    assert row["num"] == 7
    assert row["original_filename"] == source.name
    assert row["output_file"] == str(output)
    assert row["status"] == "in_progress"
    assert row["subtitle_completed_batches"] == 1
    assert row["subtitle_total_batches"] == 2
    assert row["progress_keys"] == [key for key, _ in entries]
    assert mixin._progress_entry_needs_special_visibility(row) is False
    display, display_status = mixin._progress_list_display_text(
        row,
        {"show_model_info_state": False},
        20,
        25,
    )
    assert display_status == "in_progress"
    assert "Subtitle 007" in display
    assert source.name in display
    assert "Batches 1/2" in display


def test_cleanup_preserves_completed_subtitle_batch_before_final_file_exists(
    tmp_path,
):
    from TransateKRtoEN import ProgressManager

    output = tmp_path / "Show" / "episode.srt"
    manager = ProgressManager(str(tmp_path / "work"))
    manager.prog["chapters"] = {
        "subtitle:episode.srt:1": {
            "actual_num": 1,
            "status": "completed",
            "output_file": str(output),
            "subtitle_output_file": str(output),
            "subtitle_source_file": str(tmp_path / "episode.srt"),
            "subtitle_progress_key": "subtitle:episode.srt:1",
            "subtitle_source_batch_num": 1,
            "subtitle_source_batch_count": 2,
        }
    }

    manager.cleanup_missing_files(str(output.parent))

    assert "subtitle:episode.srt:1" in manager.prog["chapters"]


def test_fresh_subtitle_work_manager_restores_and_preserves_output_mirror(
    tmp_path,
    monkeypatch,
):
    from TransateKRtoEN import ProgressManager

    output_dir = tmp_path / "Show"
    output_dir.mkdir()
    mirror_path = output_dir / "translation_progress.json"
    output = output_dir / "episode.srt"
    chapters = {
        f"subtitle:episode.srt:{batch_num}": {
            "actual_num": batch_num,
            "content_hash": f"hash-{batch_num}",
            "status": "completed",
            "output_file": str(output),
            "batch_output_file": f"response_episode_batch_{batch_num}.txt",
            "subtitle_output_file": str(output),
            "subtitle_source_file": str(tmp_path / "episode.srt"),
            "subtitle_progress_key": f"subtitle:episode.srt:{batch_num}",
            "subtitle_source_batch_num": batch_num,
            "subtitle_source_batch_count": 2,
            "subtitle_bundle_source_index": 1,
        }
        for batch_num in (1, 2)
    }
    mirror_path.write_text(
        json.dumps(
            {
                "chapters": chapters,
                "chapter_chunks": {},
                "subtitle_files": {
                    output.name: {
                        "source_file": str(tmp_path / "episode.srt"),
                        "output_file": str(output),
                        "status": "completed",
                        "total_batches": 2,
                        "completed_batches": 2,
                    }
                },
                "version": "2.1",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SUBTITLE_PROGRESS_MIRROR_FILE", str(mirror_path))

    manager = ProgressManager(str(tmp_path / "fresh-work"))

    assert manager._loaded_subtitle_progress_from_mirror is True
    assert set(manager.prog["chapters"]) == set(chapters)

    # Even a later partial/empty temporary snapshot must not erase the stable
    # per-subtitle rows already visible in the output folder.
    manager.prog["chapters"] = {}
    manager.prog.pop("subtitle_files", None)
    manager.save()

    saved_mirror = json.loads(mirror_path.read_text(encoding="utf-8"))
    assert set(saved_mirror["chapters"]) == set(chapters)
    assert output.name in saved_mirror["subtitle_files"]


def test_completed_mirrored_subtitle_batch_reuses_matching_final_output(
    tmp_path,
):
    from TransateKRtoEN import ProgressManager

    output = tmp_path / "Show" / "episode.srt"
    output.parent.mkdir()
    output.write_text("translated subtitle", encoding="utf-8")
    work_dir = tmp_path / "work"
    manager = ProgressManager(str(work_dir))
    chapter = {
        "num": 1,
        "subtitle_batch": True,
        "subtitle_progress_id": str(output),
        "subtitle_source_batch_num": 1,
        "subtitle_source_batch_count": 1,
    }
    key = manager._get_chapter_key(1, chapter_obj=chapter)
    manager.prog["chapters"][key] = {
        "actual_num": 1,
        "content_hash": "matching-hash",
        "status": "completed",
        "output_file": str(output),
        "subtitle_output_file": str(output),
        "subtitle_progress_key": key,
    }

    needs_translation, _, existing_batch = manager.check_chapter_status(
        0,
        1,
        "matching-hash",
        str(work_dir),
        chapter_obj=chapter,
    )

    assert needs_translation is False
    assert existing_batch is None
    assert chapter["_subtitle_final_output_reused"] is True


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


def test_non_epub_selection_clears_environment_only_stale_glossary(
    tmp_path,
    monkeypatch,
):
    from translator_gui import TranslatorGUI

    stale_glossary = tmp_path / "old_glossary.csv"
    stale_glossary.write_text("term,translation\n", encoding="utf-8")
    gui = TranslatorGUI.__new__(TranslatorGUI)
    gui.config = {"manual_glossary_path": str(stale_glossary)}
    gui.manual_glossary_path = None
    gui.manual_glossary_manually_loaded = False
    gui.auto_loaded_glossary_path = None
    gui.auto_loaded_glossary_for_file = None
    gui.logs = []
    gui.append_log = gui.logs.append
    gui._update_manual_glossary_status = lambda: None
    monkeypatch.setenv("MANUAL_GLOSSARY", str(stale_glossary))

    cleared = gui._clear_automatic_glossary_for_non_epub_selection(
        [str(tmp_path / "subtitles.zip")]
    )

    assert cleared is True
    assert "MANUAL_GLOSSARY" not in os.environ
    assert gui.config["manual_glossary_path"] == ""
    assert gui.manual_glossary_path is None
    assert gui.logs


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


def test_progress_manager_collapses_extracted_zip_members_to_one_bundle(
    tmp_path,
):
    from Retranslation_GUI import RetranslationMixin

    archive = tmp_path / "season.zip"
    output_dir = tmp_path / "season"
    extracted = [
        tmp_path / "extract" / "episode_01.srt",
        tmp_path / "extract" / "episode_02.srt",
    ]
    bundle_files = [str(path) for path in extracted]
    mappings = {
        os.path.normcase(os.path.abspath(path)): {
            "archive_path": str(archive),
            "bundle_id": os.path.normcase(os.path.abspath(archive)),
            "bundle_files": bundle_files,
            "group_name": "season",
            "output_dir": str(output_dir),
        }
        for path in extracted
    }
    gui = RetranslationMixin()
    gui.selected_files = bundle_files
    gui._subtitle_zip_output_info = lambda path: mappings.get(
        os.path.normcase(os.path.abspath(path))
    )

    target = gui._selected_subtitle_bundle_progress_target()

    assert target["archive_path"] == os.path.abspath(archive)
    assert target["output_dir"] == os.path.abspath(output_dir)
    assert target["bundle_files"] == [
        os.path.abspath(path) for path in extracted
    ]


def test_progress_manager_does_not_collapse_partial_subtitle_bundle(tmp_path):
    from Retranslation_GUI import RetranslationMixin

    archive = tmp_path / "season.zip"
    output_dir = tmp_path / "season"
    extracted = [
        tmp_path / "extract" / "episode_01.srt",
        tmp_path / "extract" / "episode_02.srt",
    ]
    bundle_files = [str(path) for path in extracted]
    info = {
        "archive_path": str(archive),
        "bundle_id": os.path.normcase(os.path.abspath(archive)),
        "bundle_files": bundle_files,
        "group_name": "season",
        "output_dir": str(output_dir),
    }
    gui = RetranslationMixin()
    gui.selected_files = [bundle_files[0]]
    gui._subtitle_zip_output_info = lambda _path: info

    assert gui._selected_subtitle_bundle_progress_target() is None


def test_force_retranslation_routes_complete_subtitle_bundle_before_multifile(
    tmp_path,
):
    from Retranslation_GUI import RetranslationMixin

    archive = tmp_path / "season.zip"
    output_dir = tmp_path / "season"
    extracted = [
        tmp_path / "extract" / "episode_01.srt",
        tmp_path / "extract" / "episode_02.srt",
    ]
    bundle_files = [str(path) for path in extracted]
    info = {
        "archive_path": str(archive),
        "bundle_id": os.path.normcase(os.path.abspath(archive)),
        "bundle_files": bundle_files,
        "group_name": "season",
        "output_dir": str(output_dir),
    }
    gui = RetranslationMixin()
    gui.selected_files = bundle_files
    gui._subtitle_zip_output_info = lambda _path: info
    opened = []
    gui._open_subtitle_bundle_progress_manager = opened.append
    gui._force_retranslation_multiple_files = lambda: pytest.fail(
        "subtitle ZIP members must not use the multi-file dialog"
    )

    gui.force_retranslation()

    assert len(opened) == 1
    assert opened[0]["archive_path"] == os.path.abspath(archive)


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
