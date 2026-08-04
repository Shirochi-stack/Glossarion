import json

from Retranslation_GUI import RetranslationMixin
from TransateKRtoEN import (
    ProgressManager,
    _apply_partial_refinement_response,
    _append_partial_b_translation_artifact_chapters,
    _partial_refinement_target_fragment,
)
from qa_scan_runtime import default_qa_scan_settings
from scan_html_folder import scan_html_folder, update_new_format_progress
from translate_headers_standalone import load_translations_from_file
from translation_artifacts import (
    apply_translation_artifact_response,
    collect_translation_artifact_partial_targets,
    render_translation_artifact_document,
    translation_artifact_qa_text,
    translation_artifact_target_fragment,
)


def _contains_cjk(text):
    return any("\u3400" <= char <= "\u9fff" for char in str(text))


def test_artifact_qa_text_ignores_intentional_source_values():
    metadata = {
        "title": "Translated title",
        "original_title": "\u539f\u59cb\u4e66\u540d",
        "title_translated": True,
        "description": ["English summary", "\u6b8b\u7559\u6587\u5b57"],
        "description_translated": True,
        "chapter_titles": ["\u539f\u59cb\u7ae0\u8282"],
    }
    metadata_text = translation_artifact_qa_text(
        "metadata.json", json.dumps(metadata, ensure_ascii=False)
    )

    assert "Translated title" in metadata_text
    assert "English summary" in metadata_text
    assert "\u6b8b\u7559\u6587\u5b57" in metadata_text
    assert "\u539f\u59cb\u4e66\u540d" not in metadata_text
    assert "\u539f\u59cb\u7ae0\u8282" not in metadata_text

    cache = (
        "Chapter 1:\n"
        "  Original: \u539f\u59cb\u7ae0\u8282\n"
        "  Translated: Chapter One\n"
        "Chapter 2:\n"
        "  Original: \u539f\u59cb\u4e8c\n"
        "  Translated: \u5931\u8d25\u7ffb\u8bd1\n"
    )
    cache_text = translation_artifact_qa_text("TOC.txt", cache)

    assert cache_text == "Chapter One\n\u5931\u8d25\u7ffb\u8bd1"
    assert "\u539f\u59cb\u7ae0\u8282" not in cache_text


def test_cache_partial_refinement_only_replaces_failed_translated_line(tmp_path):
    cache = (
        "Chapter 1:\r\n"
        "  Original: \u539f\u59cb\u7ae0\u8282\r\n"
        "  Translated: Chapter One\r\n"
        "----------------------------------------\r\n"
        "Chapter 2:\r\n"
        "  Original: \u539f\u59cb\u4e8c\r\n"
        "  Translated: \u5931\u8d25\u7ffb\u8bd1\r\n"
        "  Status: Using original (translation failed)\r\n"
        "----------------------------------------\r\n"
    )
    document, targets = collect_translation_artifact_partial_targets(
        "translated_headers.txt", cache, _contains_cjk
    )

    assert len(targets) == 1
    assert translation_artifact_target_fragment(
        document, targets[0]
    ) == "\u5931\u8d25\u7ffb\u8bd1"

    apply_translation_artifact_response(
        document, targets[0], "Translated Chapter Two"
    )
    rendered = render_translation_artifact_document(document)

    assert "Original: \u539f\u59cb\u7ae0\u8282" in rendered
    assert "Original: \u539f\u59cb\u4e8c" in rendered
    assert "Translated: Chapter One" in rendered
    assert "Translated: Translated Chapter Two" in rendered
    assert "translation failed" not in rendered
    assert rendered.count("\r\n") == cache.count("\r\n")

    repaired_cache = tmp_path / "translated_headers.txt"
    repaired_cache.write_text(rendered, encoding="utf-8", newline="")
    _source, translated, _outputs = load_translations_from_file(
        str(repaired_cache), log_callback=lambda _message: None
    )
    assert translated[2] == "Translated Chapter Two"


def test_metadata_partial_refinement_preserves_source_and_json_structure():
    metadata = {
        "title": "\u5931\u8d25\u4e66\u540d",
        "original_title": "\u539f\u59cb\u4e66\u540d",
        "title_translated": True,
        "description": "Line one\n\u5931\u8d25\u63cf\u8ff0",
        "description_translated": True,
        "chapter_titles": ["\u539f\u59cb\u7ae0\u8282"],
    }
    content = json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    document, targets = collect_translation_artifact_partial_targets(
        "metadata.json", content, _contains_cjk
    )

    assert {tuple(target["path"]) for target in targets} == {
        ("title",),
        ("description",),
    }
    description_target = next(
        target for target in targets if tuple(target["path"]) == ("description",)
    )
    apply_translation_artifact_response(
        document,
        description_target,
        "First translated line\nSecond translated line",
    )
    rendered = render_translation_artifact_document(document)
    parsed = json.loads(rendered)

    assert parsed["original_title"] == "\u539f\u59cb\u4e66\u540d"
    assert parsed["chapter_titles"] == ["\u539f\u59cb\u7ae0\u8282"]
    assert parsed["description"] == (
        "First translated line\nSecond translated line"
    )
    assert rendered.endswith("\n")


def test_artifact_partial_placeholder_treats_cache_value_as_plain_text():
    content = "Original: source\nTranslated: \u5931\u8d25 <Arc> & More\n"
    document, targets = collect_translation_artifact_partial_targets(
        "TOC.txt", content, _contains_cjk
    )

    fragment = _partial_refinement_target_fragment(targets[0], document)
    assert fragment == "\u5931\u8d25 &lt;Arc&gt; &amp; More"

    _apply_partial_refinement_response(
        document, targets[0], "Fixed &lt;Arc&gt; &amp; More"
    )
    assert "Translated: Fixed <Arc> & More" in (
        render_translation_artifact_document(document)
    )


def test_progress_rows_follow_toc_and_header_translation_toggles(tmp_path):
    output_dir = tmp_path / "book"
    output_dir.mkdir()
    (output_dir / "TOC.txt").write_text(
        "Original: \u539f\u59cb\nTranslated: Contents\n", encoding="utf-8"
    )
    gui = RetranslationMixin()
    gui.config = {
        "use_toc_ncx": True,
        "batch_translate_headers": False,
    }
    prog = {"chapters": {}, "version": "2.1"}

    assert gui._ensure_translation_artifact_progress_entries(
        prog, str(output_dir), str(tmp_path / "book.epub")
    ) is True
    assert set(prog["chapters"]) == {"__translation_artifact__:toc"}
    assert prog["chapters"]["__translation_artifact__:toc"]["status"] == (
        "completed"
    )

    rows = []
    gui._append_translation_artifact_display_info(
        {
            "file_path": str(tmp_path / "book.epub"),
            "output_dir": str(output_dir),
            "prog": prog,
        },
        rows,
    )

    assert [row["output_file"] for row in rows] == [
        "TOC.txt",
        "translated_headers.txt",
    ]
    assert [row["status"] for row in rows] == ["completed", "skipped"]
    assert gui._progress_entry_needs_special_visibility(rows[0]) is False
    assert gui._progress_entry_needs_special_visibility(rows[1]) is True


def test_scanner_progress_updates_all_three_artifacts_and_metadata_siblings(
    tmp_path,
):
    for filename in ("metadata.json", "TOC.txt", "translated_headers.txt"):
        (tmp_path / filename).write_text("payload", encoding="utf-8")
    prog = {
        "version": "2.1",
        "chapters": {
            "__metadata__:title": {
                "actual_num": -1,
                "output_file": "metadata.json",
                "status": "completed",
                "special_type": "metadata",
            },
            "__metadata__:fields": {
                "actual_num": -1,
                "output_file": "metadata.json",
                "status": "completed",
                "special_type": "metadata",
            },
            "__translation_artifact__:toc": {
                "actual_num": -2,
                "output_file": "TOC.txt",
                "status": "completed",
                "special_type": "toc",
            },
            "__translation_artifact__:headers": {
                "actual_num": -3,
                "output_file": "translated_headers.txt",
                "status": "completed",
                "special_type": "headers",
            },
        },
    }
    issue = "Chinese_text_found_2_chars_[\u5931\u8d25]"
    faulty = [
        {"filename": filename, "issues": [issue], "file_index": index}
        for index, filename in enumerate(
            ("metadata.json", "TOC.txt", "translated_headers.txt")
        )
    ]

    update_new_format_progress(
        prog, faulty, [], lambda _message: None, str(tmp_path)
    )

    for entry in prog["chapters"].values():
        assert entry["status"] == "qa_failed"
        assert entry["qa_issues_found"] == [issue]


def test_scan_folder_checks_only_translated_payloads_in_all_three_artifacts(
    tmp_path,
):
    (tmp_path / "chapter.xhtml").write_text(
        "<html><body><p>English chapter text.</p></body></html>",
        encoding="utf-8",
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "title": "English book title",
                "original_title": "\u539f\u59cb\u4e66\u540d",
                "title_translated": True,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "TOC.txt").write_text(
        "Original: \u539f\u59cb\u76ee\u5f55\nTranslated: \u5931\u8d25\u76ee\u5f55\n",
        encoding="utf-8",
    )
    (tmp_path / "translated_headers.txt").write_text(
        "Original: \u539f\u59cb\u6807\u9898\nTranslated: Translated Header\n",
        encoding="utf-8",
    )
    progress_path = tmp_path / "translation_progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "version": "2.1",
                "chapters": {
                    "1": {
                        "actual_num": 1,
                        "output_file": "chapter.xhtml",
                        "status": "completed",
                    },
                    "__metadata__": {
                        "actual_num": -1,
                        "output_file": "metadata.json",
                        "status": "completed",
                        "special_type": "metadata",
                    },
                    "__translation_artifact__:toc": {
                        "actual_num": -2,
                        "output_file": "TOC.txt",
                        "status": "completed",
                        "special_type": "toc",
                    },
                    "__translation_artifact__:headers": {
                        "actual_num": -3,
                        "output_file": "translated_headers.txt",
                        "status": "completed",
                        "special_type": "headers",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    settings = default_qa_scan_settings()
    settings.update(
        {
            "target_language": "english",
            "foreign_char_threshold": 0,
            "check_word_count_ratio": False,
            "check_missing_images": False,
            "check_punctuation_mismatch": False,
            "check_quotation_mismatch": False,
            "check_silent_truncation": False,
            "check_ai_truncation_detection": False,
            "check_multiple_headers": False,
            "check_repetition": False,
            "check_translation_artifacts": False,
            "check_glossary_leakage": False,
            "check_encoding_issues": False,
            "use_thread_executor": True,
        }
    )

    scan_html_folder(
        str(tmp_path),
        log=lambda _message: None,
        mode="quick-scan",
        qa_settings=settings,
        progress_path=str(progress_path),
    )

    report = json.loads(
        (
            tmp_path
            / f"{tmp_path.name}_Scan Report"
            / "validation_results.json"
        ).read_text(encoding="utf-8")
    )
    by_name = {row["filename"]: row for row in report}
    assert {
        "metadata.json",
        "TOC.txt",
        "translated_headers.txt",
    }.issubset(by_name)
    assert by_name["metadata.json"]["issues"] == []
    assert by_name["translated_headers.txt"]["issues"] == []
    assert any(
        "Chinese_text_found" in issue
        for issue in by_name["TOC.txt"]["issues"]
    )

    updated_progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert updated_progress["chapters"][
        "__translation_artifact__:toc"
    ]["status"] == "qa_failed"
    assert updated_progress["chapters"]["__metadata__"]["status"] == (
        "completed"
    )
    assert updated_progress["chapters"][
        "__translation_artifact__:headers"
    ]["status"] == "completed"


def test_partial_b_adds_only_enabled_foreign_qa_artifacts(tmp_path, monkeypatch):
    for filename in ("metadata.json", "TOC.txt", "translated_headers.txt"):
        (tmp_path / filename).write_text("payload", encoding="utf-8")
    progress = ProgressManager(str(tmp_path))
    issue = "Chinese_text_found_2_chars_[\u5931\u8d25]"
    progress.prog["chapters"] = {
        "__metadata__": {
            "actual_num": -1,
            "output_file": "metadata.json",
            "status": "qa_failed",
            "qa_issues_found": [issue],
        },
        "__translation_artifact__:toc": {
            "actual_num": -2,
            "output_file": "TOC.txt",
            "status": "qa_failed",
            "qa_issues_found": [issue],
        },
        "__translation_artifact__:headers": {
            "actual_num": -3,
            "output_file": "translated_headers.txt",
            "status": "qa_failed",
            "qa_issues_found": [issue],
        },
    }

    class Config:
        MULTIPASS_REFINEMENT_MODE = "partial.b2"

    monkeypatch.setenv("TRANSLATE_BOOK_TITLE", "1")
    monkeypatch.setenv("USE_TOC_NCX", "1")
    monkeypatch.setenv("BATCH_TRANSLATE_HEADERS", "0")

    chapters = _append_partial_b_translation_artifact_chapters(
        [], str(tmp_path), progress, Config()
    )

    assert [chapter["translation_artifact_file"] for chapter in chapters] == [
        "metadata.json",
        "TOC.txt",
    ]
    assert all(chapter["is_special"] for chapter in chapters)


def test_progress_update_keeps_artifact_identity_and_clears_qa(tmp_path):
    progress = ProgressManager(str(tmp_path))
    key = "__translation_artifact__:toc"
    progress.prog["chapters"][key] = {
        "actual_num": -2,
        "output_file": "TOC.txt",
        "status": "qa_failed",
        "qa_issues": True,
        "qa_issues_found": ["Chinese_text_found_2_chars_[\u5931\u8d25]"],
        "special_type": "toc",
        "translation_artifact_progress_key": key,
        "translation_artifact_label": "Table of Contents",
    }
    chapter = {
        "actual_chapter_num": -2,
        "original_basename": "TOC.txt",
        "translation_artifact_file": "TOC.txt",
        "translation_artifact_progress_key": key,
        "translation_artifact_label": "Table of Contents",
        "special_type": "toc",
    }

    progress.update(
        0,
        -2,
        "new-hash",
        "TOC.txt",
        status="completed",
        chapter_obj=chapter,
    )

    entry = progress.prog["chapters"][key]
    assert entry["status"] == "completed"
    assert entry["translation_artifact_progress_key"] == key
    assert entry["translation_artifact_label"] == "Table of Contents"
    assert entry["translation_artifact_file"] == "TOC.txt"
    assert "qa_issues" not in entry
    assert "qa_issues_found" not in entry
