import json
import os

from Retranslation_GUI import (
    RetranslationMixin,
    _progress_entry_has_raw_foreign_text_qa,
)
from TransateKRtoEN import (
    ProgressManager,
    _apply_partial_refinement_response,
    _append_partial_b_translation_artifact_chapters,
    _partial_b_target_request_matches,
    _partial_refinement_target_fragment,
)
from qa_scan_runtime import default_qa_scan_settings
from scan_html_folder import scan_html_folder, update_new_format_progress
from translate_headers_standalone import load_translations_from_file
from translator_gui import TranslatorGUI
from translation_artifacts import (
    apply_translation_artifact_response,
    collect_translation_artifact_partial_targets,
    render_translation_artifact_document,
    translation_artifact_qa_text,
    translation_artifact_target_fragment,
    update_translation_artifact_progress,
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


def test_artifact_in_progress_is_preserved_before_cache_file_exists(tmp_path):
    output_dir = tmp_path / "book"
    output_dir.mkdir()
    gui = RetranslationMixin()
    gui.config = {"use_toc_ncx": True, "batch_translate_headers": True}
    prog = {
        "chapters": {
            "__translation_artifact__:headers": {
                "status": "in_progress",
                "output_file": "translated_headers.txt",
                "special_type": "headers",
            }
        }
    }

    gui._ensure_translation_artifact_progress_entries(
        prog, str(output_dir), str(tmp_path / "book.epub")
    )

    assert prog["chapters"][
        "__translation_artifact__:headers"
    ]["status"] == "in_progress"


def test_batch_header_translation_publishes_live_progress(tmp_path, monkeypatch):
    from metadata_batch_translator import BatchHeaderTranslator

    progress_path = tmp_path / "translation_progress.json"
    progress_path.write_text(
        json.dumps({"version": "2.1", "chapters": {}}), encoding="utf-8"
    )

    class Client:
        output_dir = str(tmp_path)
        model = "test-model"

    translator = BatchHeaderTranslator(Client(), {"headers_per_batch": 1})

    def send_while_checking_progress(**_kwargs):
        live = json.loads(progress_path.read_text(encoding="utf-8"))
        assert live["chapters"][
            "__translation_artifact__:headers"
        ]["status"] == "in_progress"
        return '{"1": "Chapter One"}'

    monkeypatch.setattr(translator, "_send_with_retry", send_while_checking_progress)
    translated = translator.translate_headers_batch(
        {1: "Original"}, batch_size=1, translation_type="header"
    )

    assert translated == {1: "Chapter One"}
    final = json.loads(progress_path.read_text(encoding="utf-8"))
    entry = final["chapters"]["__translation_artifact__:headers"]
    assert entry["status"] == "completed"
    assert entry["model_name"] == "test-model"


def test_toc_progress_writer_preserves_chapter_rows(tmp_path):
    progress_path = tmp_path / "translation_progress.json"
    progress_path.write_text(
        json.dumps({
            "version": "2.1",
            "chapters": {"1": {"status": "completed", "output_file": "ch1.xhtml"}},
        }),
        encoding="utf-8",
    )

    assert update_translation_artifact_progress(
        str(tmp_path), "toc", "in_progress", model_name="test-model"
    )

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["chapters"]["1"]["status"] == "completed"
    assert progress["chapters"][
        "__translation_artifact__:toc"
    ]["status"] == "in_progress"


def test_managed_artifact_rows_do_not_use_special_file_keywords():
    gui = RetranslationMixin()
    gui.config = {
        "translate_special_files": False,
        "special_file_keywords": "title, toc, header",
    }

    managed_rows = (
        {"output_file": "metadata.json", "special_type": "metadata"},
        {
            "output_file": "TOC.txt",
            "special_type": "toc",
            "translation_artifact": True,
        },
        {
            "output_file": "translated_headers.txt",
            "special_type": "headers",
            "translation_artifact": True,
        },
    )
    assert all(
        gui._special_skip_keyword_for_progress_info(row) is None
        for row in managed_rows
    )
    assert gui._special_skip_keyword_for_progress_info({
        "original_filename": "toc.xhtml",
        "is_special": True,
    }) == "toc"


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


def test_resolve_qa_action_visibility_requires_raw_foreign_text_issue():
    assert _progress_entry_has_raw_foreign_text_qa(
        {
            "status": "qa_failed",
            "qa_issues_found": [
                "Chinese_text_found_4_chars_[\u5931\u8d25\u6587\u5b57]"
            ],
        }
    ) is True
    assert _progress_entry_has_raw_foreign_text_qa(
        {
            "status": "in_progress",
            "previous_progress_entry": {
                "status": "qa_failed",
                "qa_issues_found": [
                    "Japanese_text_found_2_chars_[\u5931\u6557]"
                ],
            },
        }
    ) is True
    assert _progress_entry_has_raw_foreign_text_qa(
        {
            "status": "qa_failed",
            "qa_issues_found": ["missing_images: cover.jpg"],
        }
    ) is False


def test_targeted_partial_b_matches_only_requested_progress_entry():
    target = {
        "target_progress_key": "chapter:12",
        "target_output_file": "response_chapter0012.xhtml",
        "target_actual_num": 12,
    }

    assert _partial_b_target_request_matches(
        **target,
        candidate_progress_key="chapter:12",
        candidate_output_file="different.xhtml",
        candidate_actual_num=99,
    ) is True
    assert _partial_b_target_request_matches(
        **target,
        candidate_progress_key="different-key",
        candidate_output_file="response_Chapter0012.xhtml",
        candidate_actual_num=99,
    ) is True
    assert _partial_b_target_request_matches(
        **target,
        candidate_progress_key="chapter:13",
        candidate_output_file="response_chapter0013.xhtml",
        candidate_actual_num=12,
    ) is False


def test_prepare_single_qa_resolution_filters_other_foreign_failures(
    tmp_path, monkeypatch
):
    progress_path = tmp_path / "translation_progress.json"
    source_path = tmp_path / "book.epub"
    failures = [
        {
            "source": "book.epub",
            "source_path": str(source_path),
            "progress_path": str(progress_path),
            "progress_key": "11",
            "chapter": 11,
            "output_file": "chapter0011.xhtml",
            "issues": ["Chinese_text_found_2_chars_[\u5931\u8d25]"],
        },
        {
            "source": "book.epub",
            "source_path": str(source_path),
            "progress_path": str(progress_path),
            "progress_key": "12",
            "chapter": 12,
            "output_file": "chapter0012.xhtml",
            "issues": ["Chinese_text_found_2_chars_[\u5931\u8d25]"],
        },
    ]

    class Dummy:
        _translation_qa_failure_key = (
            TranslatorGUI._translation_qa_failure_key
        )
        _qa_failure_matches_resolution_request = staticmethod(
            TranslatorGUI._qa_failure_matches_resolution_request
        )

        def _get_output_mode(self):
            return "translation"

        def _collect_translation_qa_failures(
            self, files=None, *, foreign_character_only=False
        ):
            return list(failures)

    dummy = Dummy()
    request = {
        "source_path": str(source_path),
        "progress_path": str(progress_path),
        "progress_key": "12",
        "output_file": "chapter0012.xhtml",
        "actual_num": 12,
    }
    monkeypatch.setenv("PARTIAL_B_TARGET_PROGRESS_KEY", "")
    monkeypatch.setenv("PARTIAL_B_TARGET_OUTPUT_FILE", "")
    monkeypatch.setenv("PARTIAL_B_TARGET_ACTUAL_NUM", "")

    targeted = TranslatorGUI._prepare_multipass_qa_refinement_run(
        dummy,
        True,
        "partial.b",
        requested_target=request,
    )

    assert [failure["progress_key"] for failure in targeted] == ["12"]
    assert dummy._translation_run_followup_translation_after_refinement is False
    assert dummy._translation_run_forced_multipass_mode == "partial.b"
    assert os.environ["PARTIAL_B_TARGET_PROGRESS_KEY"] == "12"
    assert os.environ["PARTIAL_B_TARGET_OUTPUT_FILE"] == "chapter0012.xhtml"


def test_progress_context_queues_one_clicked_qa_entry(tmp_path):
    source_path = tmp_path / "book.epub"
    source_path.write_bytes(b"epub")
    progress_path = tmp_path / "translation_progress.json"
    progress_path.write_text("{}", encoding="utf-8")
    issue = "Chinese_text_found_2_chars_[\u5931\u8d25]"
    progress_entry = {
        "actual_num": 12,
        "output_file": "chapter0012.xhtml",
        "status": "qa_failed",
        "qa_issues_found": [issue],
    }

    class AliveThread:
        @staticmethod
        def is_alive():
            return True

    class Dummy:
        translation_thread = None
        glossary_thread = None
        entry_epub = None

        def append_log(self, message):
            self.log_message = message

        def run_translation_thread(self):
            self.translation_thread = AliveThread()

        def _show_message(self, *_args, **_kwargs):
            raise AssertionError("unexpected message box")

    dummy = Dummy()
    data = {
        "file_path": str(source_path),
        "progress_file": str(progress_path),
        "prog": {"chapters": {"12": progress_entry}},
    }
    display_info = {
        "progress_key": "12",
        "output_file": "chapter0012.xhtml",
        "info": progress_entry,
    }

    started = RetranslationMixin._start_single_progress_qa_resolution(
        dummy, data, display_info
    )

    assert started is True
    assert dummy.selected_files == [str(source_path.resolve())]
    assert dummy._single_qa_resolution_request["progress_key"] == "12"
    assert dummy._single_qa_resolution_request["output_file"] == (
        "chapter0012.xhtml"
    )
