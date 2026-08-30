import inspect
import json
import os
import threading
import time
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

import other_settings
import Retranslation_GUI as retranslation_gui_module
import TransateKRtoEN as translation_module
import title_tag_translation as title_translation_module
from chapter_chunk_progress import extract_marked_chunks, wrap_chunk_html
from emoticon_patterns import DEFAULT_EMOTICON_PATTERNS, mask_whitelisted_emoticons

from Retranslation_GUI import (
    RetranslationMixin,
    _progress_entry_has_raw_foreign_text_qa,
)
from TransateKRtoEN import (
    BatchTranslationProcessor,
    ProgressManager,
    TranslationConfig,
    _apply_partial_refinement_response,
    _append_partial_b_translation_artifact_chapters,
    _escape_invalid_html_tags,
    _failure_output_for_save,
    _filter_multipass_chapters_to_current_range,
    _partial_b_target_request_matches,
    _partial_refinement_target_fragment,
    _should_save_failure_response,
    _should_save_truncated_response,
)
from qa_scan_runtime import (
    apply_qa_scan_env_from_settings,
    default_qa_scan_settings,
    prepare_qa_scan_settings,
    restore_env,
)
from scan_html_folder import (
    _attach_chunk_results_to_scan,
    _chunk_issue_matches,
    _foreign_character_qa_text,
    detect_ai_artifacts,
    detect_non_english_content,
    process_html_file_batch,
    scan_html_folder,
    update_new_format_progress,
)
from translate_headers_standalone import load_translations_from_file
from translator_gui import TranslatorGUI
from translation_artifacts import (
    apply_translation_artifact_response,
    collect_translation_artifact_partial_targets,
    reset_translation_artifact_progress_entries,
    render_translation_artifact_document,
    translation_artifact_path,
    translation_artifacts_are_recycled_linked,
    translation_artifact_qa_text,
    translation_artifact_target_fragment,
    update_translation_artifact_progress,
)
from title_tag_translation import (
    DEFAULT_IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT,
    image_only_title_tag_messages,
    restore_translated_title_tags,
    should_translate_title_tags,
    title_tag_translation_payload,
)


def test_title_tag_translation_is_enabled_by_default(monkeypatch):
    monkeypatch.delenv("SKIP_TITLE_TAG_TRANSLATION", raising=False)
    monkeypatch.delenv("USE_TITLE", raising=False)

    assert should_translate_title_tags() is True

    monkeypatch.setenv("SKIP_TITLE_TAG_TRANSLATION", "1")
    assert should_translate_title_tags() is False

    monkeypatch.setenv("SKIP_TITLE_TAG_TRANSLATION", "0")
    assert should_translate_title_tags() is True


def test_parallel_prequeue_stop_mode_distinguishes_graceful_and_force(monkeypatch):
    translation_module.set_stop_flag(False)
    monkeypatch.setenv("GRACEFUL_STOP", "0")
    monkeypatch.setenv("GRACEFUL_STOP_COMPLETED", "0")
    monkeypatch.setenv("TRANSLATION_CANCELLED", "0")

    assert translation_module._translation_prequeue_stop_mode(lambda: False) is None

    monkeypatch.setenv("GRACEFUL_STOP", "1")
    assert (
        translation_module._translation_prequeue_stop_mode(lambda: True)
        == "graceful"
    )

    monkeypatch.setenv("GRACEFUL_STOP", "0")
    monkeypatch.setenv("TRANSLATION_CANCELLED", "1")
    assert translation_module._translation_prequeue_stop_mode(lambda: False) == "force"

    monkeypatch.setenv("TRANSLATION_CANCELLED", "0")
    assert translation_module._translation_prequeue_stop_mode(lambda: True) == "force"


def test_parallel_chapter_scan_polls_stop_before_queueing_titles():
    source = Path(translation_module.__file__).read_text(encoding="utf-8")
    scan_start = source.index("progress_manager.begin_deferred_save()")
    scan_end = source.index("# Thread scheduling must not redefine book order", scan_start)
    scan_source = source[scan_start:scan_end]

    assert scan_source.count(
        "_translation_prequeue_stop_mode(stop_callback)"
    ) >= 3
    assert scan_source.index("progress_manager.end_deferred_save()") < scan_source.index(
        "stopped parallel queue preparation immediately"
    )
    assert "_prepare_image_only_title_plans_parallel(" in scan_source
    assert '"📝 Queued image-only <title> translation for "' in scan_source
    assert 'f"📝 Image-only chapter {log_num}: queued its "' not in scan_source


def test_parallel_preflight_defers_duplicate_chunk_splitting():
    source = Path(translation_module.__file__).read_text(encoding="utf-8")
    preflight_start = source.index('print("📊 Calculating total chunks needed...")')
    preflight_end = source.index("# Print range skip summary", preflight_start)
    preflight_source = source[preflight_start:preflight_end]

    batch_defer = preflight_source.index("if config.BATCH_TRANSLATION:")
    authoritative_split = preflight_source.index(
        "chunks = _split_chapter_for_translation("
    )
    assert batch_defer < authoritative_split
    assert "exact chunk splitting is deferred to translation workers" in source
    assert 'print("📊 Verifying chapter numbers...")' in source
    assert 'print("📊 Setting chapter numbers...")' not in source


def test_title_only_batch_requests_disable_chunk_progress_and_rate_limit_saves():
    process_source = inspect.getsource(
        translation_module.BatchTranslationProcessor.process_single_chapter
    )

    assert (
        "self.chunk_progress_enabled and not title_tag_only_chapter"
        in process_source
    )
    assert "self.progress_manager.save_rate_limited()" in process_source


def test_progress_manager_rate_limits_bursty_live_saves(tmp_path, monkeypatch):
    manager = translation_module.ProgressManager(str(tmp_path))
    saves = []
    monkeypatch.setattr(manager, "save", lambda: saves.append(time.monotonic()))
    manager._last_progress_save_monotonic = time.monotonic()

    for _ in range(31):
        assert manager.save_rate_limited(update_batch=32, max_interval=60) is False
    assert manager.save_rate_limited(update_batch=32, max_interval=60) is True
    assert len(saves) == 1


def test_progress_update_reuses_indexed_legacy_output_key(tmp_path):
    manager = translation_module.ProgressManager(str(tmp_path))
    manager.prog["chapters"] = {
        "legacy-row": {
            "actual_num": 5,
            "output_file": "response_part0005.html",
            "status": "pending",
            "content_hash": "old",
        }
    }
    manager._progress_output_index_signature = None
    manager.update(
        4,
        5,
        "new-hash",
        "part0005.xhtml",
        status="in_progress",
    )

    assert set(manager.prog["chapters"]) == {"legacy-row"}
    assert manager.prog["chapters"]["legacy-row"]["status"] == "in_progress"


def test_parallel_title_queue_uses_cached_xhtml_without_source_recovery(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("SKIP_TITLE_TAG_TRANSLATION", "0")
    original = (
        '<html><head><title>Cached title</title></head>'
        '<body><img src="cover.jpg" /></body></html>'
    )

    def fail_source_recovery(*_args, **_kwargs):
        raise AssertionError("cached complete XHTML must not reopen the source EPUB")

    monkeypatch.setattr(
        translation_module,
        "_read_source_epub_html",
        fail_source_recovery,
    )
    chapter = {"original_html": original, "body": original}

    assert translation_module._image_only_title_translation_plan(
        chapter,
        str(tmp_path),
    ) == (original, "<title>Cached title</title>")


def test_image_only_title_plans_are_prepared_concurrently(tmp_path, monkeypatch):
    worker_count = 4
    barrier = threading.Barrier(worker_count)
    worker_names = set()
    worker_names_lock = threading.Lock()

    def fake_plan(chapter, _output_dir):
        with worker_names_lock:
            worker_names.add(threading.current_thread().name)
        barrier.wait(timeout=3)
        value = chapter["num"]
        return f"<html>{value}</html>", f"<title>{value}</title>"

    monkeypatch.setattr(
        translation_module,
        "_image_only_title_translation_plan",
        fake_plan,
    )
    candidates = [
        (index, {"num": index}, index, index, f"hash-{index}")
        for index in range(12)
    ]

    plans, stop_mode, errors = (
        translation_module._prepare_image_only_title_plans_parallel(
            candidates,
            str(tmp_path),
            max_workers=worker_count,
            stop_callback=lambda: False,
        )
    )

    assert stop_mode is None
    assert errors == 0
    assert set(plans) == set(range(12))
    assert len(worker_names) == worker_count


def test_image_only_title_queue_summary_is_compact_and_sorted():
    values = [102, 74, 98, 76, 100, 78, 82, 84, 86, 88, 90, 92, 96]
    assert translation_module._format_chapter_log_values(values) == (
        "74, 76, 78, 82, 84, 86, 88, 90, 92, 96, 98, 100, 102"
    )


def test_image_only_titles_do_not_inflate_the_legacy_body_executor():
    source = Path(translation_module.__file__).read_text(encoding="utf-8")
    scan_start = source.index("chapters_to_translate = []")
    scan_end = source.index("# Print skip summary for batch mode", scan_start)
    scan_source = source[scan_start:scan_end]

    assert "image_only_titles_to_translate = []" in scan_source
    assert "image_only_titles_to_translate.append((idx, c))" in scan_source
    assert "chapters_to_translate.append((idx, c))" in scan_source

    title_plan_start = scan_source.index(
        "if _apply_image_only_title_translation_plan(c, plan):"
    )
    title_plan_end = scan_source.index("continue", title_plan_start)
    assert "chapters_to_translate.append" not in scan_source[
        title_plan_start:title_plan_end
    ]
    assert (
        "ThreadPoolExecutor(max_workers=config.BATCH_SIZE)" in source
    )


def test_image_only_title_phase_is_parallel_but_independently_bounded(
    monkeypatch,
):
    monkeypatch.setenv("GRACEFUL_STOP", "0")
    monkeypatch.setenv("GRACEFUL_STOP_COMPLETED", "0")
    monkeypatch.setenv("TRANSLATION_CANCELLED", "0")
    monkeypatch.setenv("IMAGE_ONLY_TITLE_MAX_WORKERS", "4")
    translation_module.set_stop_flag(False)

    class Config:
        BATCH_SIZE = 2000

    class Progress:
        def __init__(self):
            self.saves = 0

        def save(self):
            self.saves += 1

    active = 0
    max_active = 0
    active_lock = threading.Lock()
    first_wave = threading.Barrier(4)

    class Processor:
        def process_single_chapter(self, chapter_data):
            nonlocal active, max_active
            with active_lock:
                active += 1
                max_active = max(max_active, active)
            if chapter_data[0] < 4:
                first_wave.wait(timeout=3)
            time.sleep(0.01)
            with active_lock:
                active -= 1
            return True, chapter_data[1]["num"], None, None, None

    progress = Progress()
    stats = translation_module._process_image_only_titles_parallel(
        [(index, {"num": index}) for index in range(12)],
        Processor(),
        progress,
        lambda: False,
        Config(),
    )

    assert stats == {
        "processed": 12,
        "successful": 12,
        "failed": 0,
        "stopped": False,
        "workers": 4,
    }
    assert max_active == 4
    assert progress.saves == 1


def test_image_only_title_phase_honors_stop_before_submitting(monkeypatch):
    monkeypatch.setenv("GRACEFUL_STOP", "1")
    monkeypatch.setenv("TRANSLATION_CANCELLED", "0")
    translation_module.set_stop_flag(False)
    calls = []

    class Config:
        BATCH_SIZE = 2000

    class Processor:
        def process_single_chapter(self, chapter_data):
            calls.append(chapter_data)
            return True, 1, None, None, None

    class Progress:
        def save(self):
            pass

    stats = translation_module._process_image_only_titles_parallel(
        [(0, {"num": 1})],
        Processor(),
        Progress(),
        lambda: True,
        Config(),
    )

    assert stats["stopped"] is True
    assert stats["processed"] == 0
    assert calls == []


def test_legacy_use_title_is_only_a_fallback(monkeypatch):
    monkeypatch.delenv("SKIP_TITLE_TAG_TRANSLATION", raising=False)
    monkeypatch.setenv("USE_TITLE", "0")
    assert should_translate_title_tags() is False

    monkeypatch.setenv("USE_TITLE", "1")
    assert should_translate_title_tags() is True

    monkeypatch.setenv("SKIP_TITLE_TAG_TRANSLATION", "0")
    monkeypatch.setenv("USE_TITLE", "0")
    assert should_translate_title_tags() is True


def test_title_only_reconstruction_preserves_image_xhtml_exactly():
    original = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml">\n'
        '<head><title class="book-title">義妹生活</title></head>\n'
        '<body><svg><image xlink:href="../image/kuchie-004.jpg" /></svg></body>\n'
        '</html>'
    )

    assert title_tag_translation_payload(original) == (
        '<title class="book-title">義妹生活</title>'
    )

    restored = restore_translated_title_tags(
        original,
        "```html\n<title>Days with My Stepsister & More</title>\n```",
    )
    assert restored == original.replace(
        "義妹生活",
        "Days with My Stepsister &amp; More",
    )


def test_title_payload_does_not_parse_the_image_document(monkeypatch):
    def fail_parser(*_args, **_kwargs):
        raise AssertionError("title extraction must not parse the whole document")

    monkeypatch.setattr(title_translation_module, "BeautifulSoup", fail_parser)
    markup = (
        "<html><head><title>Only this text</title></head><body>"
        + '<img src="page.jpg" />' * 10_000
        + "</body></html>"
    )

    assert title_translation_module.title_tag_translation_payload(markup) == (
        "<title>Only this text</title>"
    )


def test_dedicated_image_only_title_messages_are_strictly_gated(monkeypatch):
    class Config:
        IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT = (
            "TITLE-ONLY REQUEST FOR {target_lang}"
        )

    monkeypatch.setenv("OUTPUT_LANGUAGE", "Spanish")
    payload = '<title class="book-title">義妹生活</title>'

    assert image_only_title_tag_messages(
        Config(),
        {"body": payload},
        payload,
    ) is None
    assert image_only_title_tag_messages(
        Config(),
        {
            "body": payload,
            "has_images": True,
            "_title_tag_only_translation": True,
        },
        payload,
    ) == [
        {"role": "system", "content": "TITLE-ONLY REQUEST FOR Spanish"},
        {"role": "user", "content": payload},
    ]


def test_translation_config_loads_dedicated_title_prompt(monkeypatch):
    custom_prompt = "CUSTOM IMAGE TITLE PROMPT FOR {target_lang}"
    monkeypatch.setenv("IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT", custom_prompt)

    assert TranslationConfig().IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT == custom_prompt

    monkeypatch.setenv("IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT", "")
    assert (
        TranslationConfig().IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT
        == DEFAULT_IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT
    )

    assert DEFAULT_IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT.startswith(
        "Translate only the text inside the provided <title> tag"
    )


def test_sequential_retry_transport_does_not_augment_title_messages(
    tmp_path,
    monkeypatch,
):
    class Config:
        MODEL = "gpt-test"
        CONTEXTUAL = True
        TEMP = 0
        MAX_OUTPUT_TOKENS = 1024
        MAX_RETRY_TOKENS = 1024
        SPLIT_FAILED_RETRY_ATTEMPTS = 1
        DISABLE_MERGE_FALLBACK = False
        RETRY_SPLIT_FAILED = False
        RETRY_TRUNCATED = False
        RETRY_TIMEOUT = False
        CHUNK_TIMEOUT = 30
        EMERGENCY_IMAGE_RESTORE = False
        IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT = "ONLY {target_lang}"

    sent = []

    class Client:
        def send(self, messages, **_kwargs):
            sent.append(messages)
            return "<title>Translated</title>", "stop"

    monkeypatch.setenv("OUTPUT_LANGUAGE", "English")
    monkeypatch.setenv("RETRY_DUPLICATE_BODIES", "0")
    monkeypatch.setenv("RETRY_TIMEOUT", "0")
    monkeypatch.setenv("RETRY_TRUNCATED", "0")
    monkeypatch.setattr(
        translation_module,
        "_single_pass_glossary_mode",
        lambda: True,
    )
    monkeypatch.setattr(
        translation_module,
        "_build_single_pass_glossary_messages",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("single-pass glossary must not touch title requests")
        ),
    )

    chapter = {
        "num": 1,
        "body": "<title>原題</title>",
        "filename": "chapter.xhtml",
        "_title_tag_only_translation": True,
    }
    messages = image_only_title_tag_messages(
        Config(),
        chapter,
        chapter["body"],
    )
    processor = translation_module.TranslationProcessor(
        Config(),
        Client(),
        str(tmp_path),
        stop_callback=lambda: False,
    )

    result, finish_reason, _raw = processor.translate_with_retry(
        messages,
        chapter["body"],
        chapter,
        1,
        1,
    )

    assert result == "<title>Translated</title>"
    assert finish_reason == "stop"
    assert sent == [messages]


def test_image_only_page_becomes_title_only_translation_unit(tmp_path, monkeypatch):
    monkeypatch.setenv("SKIP_TITLE_TAG_TRANSLATION", "0")
    original = (
        '<html><head><title>義妹生活</title></head>'
        '<body><img src="../image/kuchie-004.jpg" /></body></html>'
    )
    chapter = {
        "body": original,
        "original_html": original,
        "has_images": True,
        "image_count": 1,
        "file_size": len(original),
    }

    assert translation_module._prepare_image_only_title_translation(
        chapter,
        str(tmp_path),
    ) is True
    assert chapter["body"] == "<title>義妹生活</title>"
    assert chapter["_title_tag_original_markup"] == original
    assert chapter["has_images"] is False
    assert "img" not in chapter["body"]


def test_skip_setting_keeps_image_only_page_out_of_title_queue(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("SKIP_TITLE_TAG_TRANSLATION", "1")
    original = (
        '<html><head><title>義妹生活</title></head>'
        '<body><img src="../image/kuchie-004.jpg" /></body></html>'
    )
    chapter = {"body": original, "original_html": original}

    assert translation_module._prepare_image_only_title_translation(
        chapter,
        str(tmp_path),
    ) is False
    assert chapter["body"] == original
    assert "_title_tag_only_translation" not in chapter


def test_parallel_worker_rebuilds_title_only_xhtml(tmp_path, monkeypatch):
    class Config:
        MODEL = "gpt-test"
        BATCH_SIZE = 2
        CONTEXTUAL = False
        HIST_LIMIT = 0
        ENABLE_IMAGE_TRANSLATION = False
        USE_ROLLING_SUMMARY = False
        ASSISTANT_PROMPT = ""
        INCLUDE_PREVIOUS_CHUNK = False
        TEMP = 0
        MAX_OUTPUT_TOKENS = 1024
        MAX_RETRY_TOKENS = 1024
        EMERGENCY_IMAGE_RESTORE = False
        REMOVE_AI_ARTIFACTS = "off"
        IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT = (
            "DEDICATED TITLE PROMPT FOR {target_lang}"
        )

        @staticmethod
        def get_system_prompt(actual_merge_count=1):
            raise AssertionError("normal profile prompt must not be read")

        @staticmethod
        def get_effective_output_limit():
            return 4096

        @staticmethod
        def get_effective_compression_factor():
            return 1

    sent = []

    class Client:
        def send(self, messages, **_kwargs):
            sent.append(messages)
            return "<title>Days with My Stepsister</title>", "stop"

    monkeypatch.setenv("THREAD_SUBMISSION_DELAY_SECONDS", "0")
    monkeypatch.setenv("SEND_INTERVAL_SECONDS", "0")
    monkeypatch.setenv("ORDER_BATCH_REQUESTS_BY_SPINE", "0")
    monkeypatch.setenv("CHAR_RATIO_TRUNCATION_ENABLED", "0")
    monkeypatch.setenv("DIRECT_TEXT_ACTIVE", "0")
    monkeypatch.setenv("OUTPUT_LANGUAGE", "English")
    monkeypatch.setattr(
        translation_module.ContentProcessor,
        "image_processing_html",
        staticmethod(lambda chapter: chapter["body"]),
    )
    monkeypatch.setattr(
        translation_module.ContentProcessor,
        "is_mostly_image_html",
        staticmethod(lambda _body: False),
    )
    monkeypatch.setattr(
        translation_module,
        "_split_chapter_for_translation",
        lambda _splitter, chapter, _tokens, filename=None: [
            (chapter["body"], 1, 1)
        ],
    )
    monkeypatch.setattr(translation_module, "find_glossary_file", lambda _out: None)
    monkeypatch.setattr(
        translation_module,
        "build_system_prompt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("normal/glossary prompt builder must not run")
        ),
    )
    monkeypatch.setattr(
        translation_module,
        "apply_emergency_glossary_compliance",
        lambda text, _out: text,
    )
    monkeypatch.setattr(
        translation_module,
        "_build_translation_chunk_prompt_parts",
        lambda system_prompt, chunk, *_args, **_kwargs: (
            system_prompt,
            [],
            chunk,
        ),
    )

    original = (
        '<html><head><title>義妹生活</title></head>'
        '<body><img src="../image/kuchie-004.jpg" /></body></html>'
    )
    progress_updates = []
    processor = BatchTranslationProcessor(
        Config(),
        Client(),
        [],
        str(tmp_path),
        threading.RLock(),
        lambda: None,
        lambda *args, **kwargs: progress_updates.append((args, kwargs)),
        lambda: False,
    )
    result = processor.process_single_chapter(
        (
            0,
            {
                "num": 1,
                "actual_chapter_num": 1,
                "body": "<title>義妹生活</title>",
                "_title_tag_only_translation": True,
                "_title_tag_original_markup": original,
                "filename": "chapter001.xhtml",
                "original_basename": "chapter001.xhtml",
            },
        )
    )

    assert result[0] is True
    assert sent == [[
        {
            "role": "system",
            "content": "DEDICATED TITLE PROMPT FOR English",
        },
        {"role": "user", "content": "<title>義妹生活</title>"},
    ]]
    assert progress_updates[-1][1]["status"] == "completed"
    output_files = [path for path in tmp_path.iterdir() if path.is_file()]
    assert len(output_files) == 1
    output_soup = BeautifulSoup(
        output_files[0].read_text(encoding="utf-8"),
        "html.parser",
    )
    assert output_soup.title.get_text() == "Days with My Stepsister"
    assert output_soup.img["src"] == "../image/kuchie-004.jpg"


def test_other_settings_places_new_toggle_after_image_title_setting():
    source = Path(translation_module.__file__).with_name(
        "other_settings.py"
    ).read_text(encoding="utf-8")
    image_toggle = source.index(
        '"Skip image title translation (use original filename)"'
    )
    title_tag_toggle = source.index('"Skip title tag translation"')
    configure_button = source.index(
        '"Configure Title Prompt"',
        title_tag_toggle,
    )

    assert title_tag_toggle > image_toggle
    assert configure_button > title_tag_toggle
    assert "def configure_image_only_title_tag_prompt" in source
    assert "image tags but no meaningful body text" in source
    title_row_start = source.index("title_tag_row = QWidget()")
    title_row_end = source.index(
        "section_v.addWidget(title_tag_row)",
        title_row_start,
    )
    title_row_source = source[title_row_start:title_row_end]
    button_add = title_row_source.index(
        "title_tag_row_h.addWidget(configure_title_prompt_btn)"
    )
    trailing_stretch = title_row_source.index("title_tag_row_h.addStretch()")
    assert trailing_stretch > button_add
    assert "configure_title_prompt_btn.setFixedWidth(" in title_row_source
    assert "configure_title_prompt_btn.sizeHint().width() + 12" in title_row_source
    assert "Use title (Legacy)" not in source


def test_sequential_title_only_path_uses_isolated_message_builder():
    source = Path(translation_module.__file__).read_text(encoding="utf-8")
    sequential_start = source.index(
        "for chunk_idx_enumerate, (chunk_html, chunk_idx, total_chunks)"
    )
    sequential_end = source.index(
        "result, finish_reason, raw_obj = translation_processor.translate_with_retry",
        sequential_start,
    )
    sequential_source = source[sequential_start:sequential_end]

    assert "title_tag_messages = image_only_title_tag_messages(" in sequential_source
    assert "msgs = title_tag_messages" in sequential_source
    assert "if not title_tag_only_request and config.CONTEXTUAL" in sequential_source


def test_scanner_foreign_character_filter_respects_title_skip_setting():
    html = (
        "<html><head><title>義妹生活</title></head>"
        "<body><p>English body text.</p></body></html>"
    )
    raw_text = BeautifulSoup(html, "html.parser").get_text("\n", strip=True)

    checked_text = _foreign_character_qa_text(
        raw_text,
        html,
        {
            "skip_title_tag_translation": False,
            "target_language": "english",
        },
    )
    skipped_text = _foreign_character_qa_text(
        raw_text,
        html,
        {
            "skip_title_tag_translation": True,
            "target_language": "english",
        },
    )

    assert detect_non_english_content(
        checked_text,
        {"target_language": "english", "foreign_char_threshold": 0},
    )[0] is True
    assert detect_non_english_content(
        skipped_text,
        {"target_language": "english", "foreign_char_threshold": 0},
    )[0] is False
    assert "義妹生活" not in skipped_text
    assert "English body text." in skipped_text


def test_scan_worker_excludes_skipped_title_from_foreign_qa(tmp_path):
    html = (
        "<html><head><title>義妹生活</title></head>"
        "<body><p>English body text.</p></body></html>"
    )
    (tmp_path / "chapter.html").write_text(html, encoding="utf-8")
    settings = default_qa_scan_settings()
    settings["skip_title_tag_translation"] = True

    results = process_html_file_batch(
        (
            [(0, "chapter.html")],
            str(tmp_path),
            settings,
            "quick-scan",
            {},
            {},
            False,
            {},
            {},
        )
    )

    assert len(results) == 1
    assert "義妹生活" in results[0]["raw_text"]
    assert "義妹生活" not in results[0]["foreign_qa_text"]
    assert detect_non_english_content(
        results[0]["foreign_qa_text"],
        settings,
    )[0] is False


def test_chunk_foreign_character_localization_ignores_skipped_title():
    chunk_html = "<head><title>義妹生活</title></head>"
    chunk_text = "義妹生活"
    issue = detect_non_english_content(
        chunk_text,
        {"target_language": "english", "foreign_char_threshold": 0},
    )[1][0]

    assert _chunk_issue_matches(
        issue,
        chunk_html,
        chunk_text,
        "",
        {
            "skip_title_tag_translation": False,
            "target_language": "english",
            "foreign_char_threshold": 0,
        },
    ) is True
    assert _chunk_issue_matches(
        issue,
        chunk_html,
        chunk_text,
        "",
        {
            "skip_title_tag_translation": True,
            "target_language": "english",
            "foreign_char_threshold": 0,
        },
    ) is False


def test_qa_runtime_copies_live_title_skip_toggle_into_scan_settings():
    class Owner:
        config = {
            "skip_title_tag_translation": False,
            "output_language": "English",
        }
        skip_title_tag_translation_var = True

    settings = prepare_qa_scan_settings({}, owner=Owner())

    assert settings["skip_title_tag_translation"] is True


def _contains_cjk(text):
    return any("\u3400" <= char <= "\u9fff" for char in str(text))


def test_invalid_html_tag_escaping_preserves_ruby_annotation_tags():
    ruby_html = (
        "<p><ruby><rb>Tomoki</rb><rt>tomoki</rt>"
        "<rtc><rt>reading</rt></rtc></ruby></p>"
    )

    assert _escape_invalid_html_tags(ruby_html) == ruby_html


def test_preserve_original_toggle_enables_failed_output_save(monkeypatch):
    monkeypatch.setenv("PRESERVE_ORIGINAL_TEXT_ON_FAILURE", "1")
    monkeypatch.setenv("SAVE_PROHIBITED_RESULTS", "0")
    monkeypatch.setenv("SAVE_PARTIAL_RESULTS", "0")

    assert _should_save_failure_response(
        "raw source",
        config=None,
        qa_issue=["API_ERROR"],
    ) is True


def test_failure_specific_save_toggles_take_priority_over_preserved_source(monkeypatch):
    monkeypatch.setenv("PRESERVE_ORIGINAL_TEXT_ON_FAILURE", "1")
    monkeypatch.setenv("SAVE_PROHIBITED_RESULTS", "1")
    monkeypatch.setenv("SAVE_PARTIAL_RESULTS", "1")

    assert _failure_output_for_save(
        "streamed or truncated translation",
        "raw source",
        qa_issue=["PROHIBITED_CONTENT"],
    ) == "streamed or truncated translation"
    assert _failure_output_for_save(
        "truncated translation",
        "raw source",
        qa_issue=["TRUNCATED"],
    ) == "truncated translation"
    assert _should_save_truncated_response() is True


def test_unrelated_failure_toggle_does_not_override_preserved_source(monkeypatch):
    monkeypatch.setenv("PRESERVE_ORIGINAL_TEXT_ON_FAILURE", "1")
    monkeypatch.setenv("SAVE_PROHIBITED_RESULTS", "1")
    monkeypatch.setenv("SAVE_PARTIAL_RESULTS", "0")
    assert _failure_output_for_save(
        "truncated translation",
        "raw source",
        qa_issue=["TRUNCATED"],
    ) == "raw source"

    monkeypatch.setenv("SAVE_PROHIBITED_RESULTS", "0")
    monkeypatch.setenv("SAVE_PARTIAL_RESULTS", "1")
    assert _failure_output_for_save(
        "blocked translation",
        "raw source",
        qa_issue=["PROHIBITED_CONTENT"],
    ) == "raw source"


def test_truncated_save_gate_still_respects_legacy_toggle_without_preservation(
    monkeypatch,
):
    monkeypatch.setenv("PRESERVE_ORIGINAL_TEXT_ON_FAILURE", "0")
    monkeypatch.setenv("SAVE_PARTIAL_RESULTS", "0")
    assert _should_save_truncated_response() is False

    monkeypatch.setenv("SAVE_PARTIAL_RESULTS", "1")
    assert _should_save_truncated_response() is True
    assert _failure_output_for_save("partial translation", "raw source") == (
        "partial translation"
    )


@pytest.mark.parametrize(
    "error_info",
    [
        {"error": "parse failure"},
        {"error": "rate limit"},
        {"error": "prohibited content"},
        {"error": "unexpected provider failure"},
    ],
)
def test_preserve_original_returns_verbatim_text_for_all_failure_types(
    monkeypatch, error_info
):
    from unified_api_client import UnifiedClient

    monkeypatch.setenv("PRESERVE_ORIGINAL_TEXT_ON_FAILURE", "1")
    monkeypatch.setenv("SAVE_PROHIBITED_RESULTS", "0")
    monkeypatch.setenv("SAVE_PARTIAL_RESULTS", "0")
    client = object.__new__(UnifiedClient)
    source = "<p>원문 그대로</p>"

    assert client._handle_empty_result(
        [{"role": "user", "content": source}],
        "translation",
        error_info,
    ) == source


@pytest.mark.parametrize(
    ("provider_finish_reason", "expected_finish_reason"),
    [
        ("error", "error"),
        ("content_filter", "content_filter"),
        ("prohibited_content", "prohibited_content"),
        ("length", "length"),
        ("timeout", "error"),
        ("other_error", "error"),
    ],
)
def test_send_replaces_any_failed_result_with_original_text(
    monkeypatch, provider_finish_reason, expected_finish_reason
):
    from unified_api_client import UnifiedClient

    monkeypatch.setenv("PRESERVE_ORIGINAL_TEXT_ON_FAILURE", "1")
    monkeypatch.setenv("SAVE_PROHIBITED_RESULTS", "0")
    monkeypatch.setenv("SAVE_PARTIAL_RESULTS", "0")
    client = object.__new__(UnifiedClient)
    monkeypatch.setattr(
        client,
        "_send_core",
        lambda *args, **kwargs: (
            "partially streamed translation",
            provider_finish_reason,
        ),
    )

    assert client.send(
        [{"role": "user", "content": "raw source"}],
        context="translation",
    ) == ("raw source", expected_finish_reason)


def test_send_replaces_raised_provider_failure_but_not_cancellation(monkeypatch):
    from unified_api_client import UnifiedClient, UnifiedClientError

    monkeypatch.setenv("PRESERVE_ORIGINAL_TEXT_ON_FAILURE", "1")
    client = object.__new__(UnifiedClient)

    def fail(*args, **kwargs):
        raise UnifiedClientError("provider timed out", error_type="timeout")

    monkeypatch.setattr(client, "_send_core", fail)
    messages = [{"role": "user", "content": "raw source"}]
    assert client.send(messages, context="translation") == ("raw source", "error")

    def cancel(*args, **kwargs):
        raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")

    monkeypatch.setattr(client, "_send_core", cancel)
    with pytest.raises(UnifiedClientError, match="cancelled"):
        client.send(messages, context="translation")


def test_blocked_stream_text_is_recovered_from_provider_error(monkeypatch):
    from unified_api_client import UnifiedClient, UnifiedClientError

    monkeypatch.delenv("PRESERVE_ORIGINAL_TEXT_ON_FAILURE", raising=False)
    client = object.__new__(UnifiedClient)
    error = UnifiedClientError(
        "blocked at end of stream",
        error_type="prohibited_content",
        details={"partial_content": "streamed response"},
    )

    assert client._failure_response_content(
        [{"role": "user", "content": "raw source"}],
        "translation",
        error=error,
        fallback="[CONTENT BLOCKED]",
    ) == "streamed response"

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(client, "_send_core", fail)
    assert client.send(
        [{"role": "user", "content": "raw source"}],
        context="translation",
    ) == ("streamed response", "prohibited_content")


def test_save_prohibited_yields_preserved_source_to_streamed_blocked_text(monkeypatch):
    from unified_api_client import UnifiedClient, UnifiedClientError

    monkeypatch.setenv("PRESERVE_ORIGINAL_TEXT_ON_FAILURE", "1")
    monkeypatch.setenv("SAVE_PROHIBITED_RESULTS", "1")
    client = object.__new__(UnifiedClient)
    error = UnifiedClientError(
        "blocked at end of stream",
        error_type="prohibited_content",
        details={"partial_content": "streamed response"},
    )

    assert client._failure_response_content(
        [{"role": "user", "content": "raw source"}],
        "translation",
        error=error,
        fallback="[CONTENT BLOCKED]",
    ) == "streamed response"


@pytest.mark.parametrize(
    ("finish_reason", "save_prohibited", "save_partial"),
    [
        ("prohibited_content", "1", "0"),
        ("error", "1", "0"),
        ("length", "0", "1"),
    ],
)
def test_send_failure_toggle_yields_preserved_source_to_provider_text(
    monkeypatch,
    finish_reason,
    save_prohibited,
    save_partial,
):
    from unified_api_client import UnifiedClient

    monkeypatch.setenv("PRESERVE_ORIGINAL_TEXT_ON_FAILURE", "1")
    monkeypatch.setenv("SAVE_PROHIBITED_RESULTS", save_prohibited)
    monkeypatch.setenv("SAVE_PARTIAL_RESULTS", save_partial)
    client = object.__new__(UnifiedClient)
    monkeypatch.setattr(
        client,
        "_send_core",
        lambda *args, **kwargs: ("streamed provider text", finish_reason),
    )

    content, returned_reason = client.send(
        [{"role": "user", "content": "raw source"}],
        context="translation",
    )

    assert content == "streamed provider text"
    assert returned_reason == finish_reason


@pytest.mark.parametrize(
    ("error_type", "save_prohibited", "save_partial", "expected_reason"),
    [
        ("api_error", "1", "0", "error"),
        ("truncated", "0", "1", "length"),
    ],
)
def test_send_raised_failure_toggle_recovers_stream_before_preserved_source(
    monkeypatch,
    error_type,
    save_prohibited,
    save_partial,
    expected_reason,
):
    from unified_api_client import UnifiedClient, UnifiedClientError

    monkeypatch.setenv("PRESERVE_ORIGINAL_TEXT_ON_FAILURE", "1")
    monkeypatch.setenv("SAVE_PROHIBITED_RESULTS", save_prohibited)
    monkeypatch.setenv("SAVE_PARTIAL_RESULTS", save_partial)
    client = object.__new__(UnifiedClient)
    error = UnifiedClientError(
        "stream failed",
        error_type=error_type,
        details={"partial_content": "streamed before failure"},
    )
    monkeypatch.setattr(
        client,
        "_send_core",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    assert client.send(
        [{"role": "user", "content": "raw source"}],
        context="translation",
    ) == ("streamed before failure", expected_reason)


def _run_batch_chapter_failure(
    tmp_path,
    monkeypatch,
    *,
    provider_text="streamed response",
    finish_reason="prohibited_content",
    raised_error=None,
):
    class Config:
        MODEL = "gpt-test"
        BATCH_SIZE = 1
        CONTEXTUAL = False
        HIST_LIMIT = 0
        ENABLE_IMAGE_TRANSLATION = False
        USE_ROLLING_SUMMARY = False
        ASSISTANT_PROMPT = ""
        INCLUDE_PREVIOUS_CHUNK = False
        TEMP = 0
        MAX_OUTPUT_TOKENS = 1024
        MAX_RETRY_TOKENS = 1024

        @staticmethod
        def get_system_prompt(actual_merge_count=1):
            return "Translate."

        @staticmethod
        def get_effective_output_limit():
            return 4096

        @staticmethod
        def get_effective_compression_factor():
            return 1

    class Client:
        def send(self, messages, temperature=None, max_tokens=None, context=None):
            if raised_error is not None:
                raise raised_error
            return provider_text, finish_reason

    monkeypatch.setenv("THREAD_SUBMISSION_DELAY_SECONDS", "0")
    monkeypatch.setenv("SEND_INTERVAL_SECONDS", "0")
    monkeypatch.setenv("ORDER_BATCH_REQUESTS_BY_SPINE", "0")
    monkeypatch.setenv("CHAR_RATIO_TRUNCATION_ENABLED", "0")
    monkeypatch.setenv("DIRECT_TEXT_ACTIVE", "0")
    monkeypatch.setattr(
        translation_module.ContentProcessor,
        "image_processing_html",
        staticmethod(lambda chapter: chapter["body"]),
    )
    monkeypatch.setattr(
        translation_module.ContentProcessor,
        "is_mostly_image_html",
        staticmethod(lambda _body: False),
    )
    monkeypatch.setattr(
        translation_module,
        "_split_chapter_for_translation",
        lambda _splitter, chapter, _tokens, filename=None: [
            (chapter["body"], 1, 1)
        ],
    )
    monkeypatch.setattr(translation_module, "find_glossary_file", lambda _out: None)
    monkeypatch.setattr(
        translation_module,
        "build_system_prompt",
        lambda base, *_args, **_kwargs: base,
    )
    monkeypatch.setattr(
        translation_module,
        "apply_emergency_glossary_compliance",
        lambda text, _out: text,
    )
    monkeypatch.setattr(
        translation_module,
        "_build_translation_chunk_prompt_parts",
        lambda system_prompt, chunk, *_args, **_kwargs: (
            system_prompt,
            [],
            chunk,
        ),
    )

    progress_updates = []
    processor = BatchTranslationProcessor(
        Config(),
        Client(),
        [],
        str(tmp_path),
        threading.RLock(),
        lambda: None,
        lambda *args, **kwargs: progress_updates.append((args, kwargs)),
        lambda: False,
    )
    source = "<p>원문 그대로</p>"
    result = processor.process_single_chapter(
        (
            0,
            {
                "num": 1,
                "actual_chapter_num": 1,
                "body": source,
                "filename": "chapter001.xhtml",
                "original_basename": "chapter001.xhtml",
            },
        )
    )
    output_files = [path for path in tmp_path.iterdir() if path.is_file()]
    assert result[0] is False
    assert progress_updates[-1][1]["status"] in {"qa_failed", "failed"}
    assert len(output_files) == 1
    return source, output_files[0].read_text(encoding="utf-8")


@pytest.mark.parametrize("source_extension", [".epub", ".pdf"])
def test_batch_document_chunk_progress_submits_only_missing_chunks(
    tmp_path,
    monkeypatch,
    source_extension,
):
    chunk_budget = {
        "initial_output_token_limit": 12000,
        "cached_output_token_limit": 9000,
        "compression_factor": 2.0,
        "safety_margin": 500,
        "minimum_chunk_size": 1000,
        "initial_chunk_size": 5750,
        "cached_chunk_size": 4250,
    }

    class Config:
        MODEL = "model-a"
        input_path = str(tmp_path / f"book{source_extension}")
        ENABLE_CHUNK_PROGRESS = True
        BATCH_SIZE = 3
        CONTEXTUAL = False
        HIST_LIMIT = 0
        ENABLE_IMAGE_TRANSLATION = False
        USE_ROLLING_SUMMARY = False
        ASSISTANT_PROMPT = ""
        TEMP = 0
        MAX_OUTPUT_TOKENS = 12000
        MAX_RETRY_TOKENS = 12000
        EMERGENCY_IMAGE_RESTORE = False
        REMOVE_AI_ARTIFACTS = "off"

        @staticmethod
        def get_system_prompt(actual_merge_count=1):
            return "Translate."

        @staticmethod
        def get_chunk_budget_snapshot(safety_margin=500, minimum=1000):
            return chunk_budget

    chunks = [
        ("chunk-1", 1, 3),
        ("chunk-2", 2, 3),
        ("chunk-3", 3, 3),
    ]
    sent = []
    in_flight_seen = {}

    def fake_send(messages, *_args, **kwargs):
        from unified_api_client import set_current_thread_actual_request_model

        source = messages[-1]["content"]
        model_name, key_identifier = {
            "chunk-2": ("provider/model-two", "key-two"),
            "chunk-3": ("provider/model-three", "key-three"),
        }[source]
        set_current_thread_actual_request_model(model_name, key_identifier)
        callback = kwargs.get("before_send_callback")
        try:
            if callback:
                callback()
        finally:
            set_current_thread_actual_request_model(None, None)
        chunk_index = int(source.rsplit("-", 1)[-1])
        in_flight_seen[source] = progress.prog["chapter_chunks"][
            "hash-1"
        ]["entries"][str(chunk_index)]["status"]
        sent.append(source)
        return f"translated:{source}", "stop", None

    monkeypatch.setenv("THREAD_SUBMISSION_DELAY_SECONDS", "0")
    monkeypatch.setenv("CHAR_RATIO_TRUNCATION_ENABLED", "0")
    monkeypatch.setenv("DIRECT_TEXT_ACTIVE", "0")
    monkeypatch.setattr(
        translation_module.ContentProcessor,
        "image_processing_html",
        staticmethod(lambda chapter: chapter["body"]),
    )
    monkeypatch.setattr(
        translation_module.ContentProcessor,
        "is_mostly_image_html",
        staticmethod(lambda _body: False),
    )
    monkeypatch.setattr(
        translation_module,
        "_split_chapter_for_translation",
        lambda *_args, **_kwargs: chunks,
    )
    monkeypatch.setattr(
        translation_module,
        "find_glossary_file",
        lambda _out: None,
    )
    monkeypatch.setattr(
        translation_module,
        "build_system_prompt",
        lambda base, *_args, **_kwargs: base,
    )
    monkeypatch.setattr(
        translation_module,
        "apply_emergency_glossary_compliance",
        lambda text, _out: text,
    )
    monkeypatch.setattr(
        translation_module,
        "_build_translation_chunk_prompt_parts",
        lambda system_prompt, chunk, *_args, **_kwargs: (
            system_prompt,
            [],
            chunk,
        ),
    )
    monkeypatch.setattr(translation_module, "send_with_interrupt", fake_send)
    monkeypatch.setattr(
        translation_module,
        "is_qa_failed_response",
        lambda _text: False,
    )

    progress = ProgressManager(str(tmp_path))
    progress.prepare_chapter_chunk_progress(
        "hash-1", 3, chunk_budget, enabled=True
    )
    progress.record_chapter_chunk(
        "hash-1", 1, 3, "cached-one", chunk_budget
    )
    progress.save()

    class Client:
        pass

    processor = BatchTranslationProcessor(
        Config(),
        Client(),
        [],
        str(tmp_path),
        threading.RLock(),
        progress.save,
        lambda idx, actual_num, content_hash, output_file=None,
        status="completed", **kwargs: progress.update(
            idx,
            actual_num,
            content_hash,
            output_file,
            status,
            **kwargs,
        ),
        lambda: False,
        progress_manager=progress,
    )
    result = processor.process_single_chapter(
        (
            0,
            {
                "num": 1,
                "actual_chapter_num": 1,
                "content_hash": "hash-1",
                "body": "original",
                "filename": "chapter001.xhtml",
                "original_basename": "chapter001.xhtml",
            },
        )
    )

    assert result[0] is True
    assert sorted(sent) == ["chunk-2", "chunk-3"]
    assert in_flight_seen == {
        "chunk-2": "in_progress",
        "chunk-3": "in_progress",
    }
    marked = extract_marked_chunks(result[3], "hash-1")
    assert [marked[index]["content"] for index in (1, 2, 3)] == [
        "cached-one",
        "translated:chunk-2",
        "translated:chunk-3",
    ]
    entry = progress.prog["chapter_chunks"]["hash-1"]
    assert entry["schema_version"] == 2
    assert entry["completed"] == [1, 2, 3]
    assert entry["chapter_status"] == "completed"
    assert entry["entries"]["2"]["source"] == "chunk-2"
    assert entry["entries"]["3"]["source"] == "chunk-3"
    assert entry["entries"]["2"]["model_name"] == "provider/model-two"
    assert entry["entries"]["2"]["key_identifier"] == "key-two"
    assert entry["entries"]["3"]["model_name"] == "provider/model-three"
    assert entry["entries"]["3"]["key_identifier"] == "key-three"


def test_html_scanner_marks_and_clears_only_the_failing_chunk(tmp_path):
    progress = ProgressManager(str(tmp_path))
    budget = {
        "initial_output_token_limit": 12000,
        "cached_output_token_limit": 9000,
        "compression_factor": 2.0,
        "safety_margin": 500,
        "minimum_chunk_size": 1000,
        "initial_chunk_size": 5750,
        "cached_chunk_size": 4250,
    }
    progress.prog["chapters"]["1"] = {
        "actual_num": 1,
        "content_hash": "chapter-hash",
        "output_file": "chapter.html",
        "status": "completed",
    }
    progress.prepare_chapter_chunk_progress(
        "chapter-hash", 2, budget, enabled=True
    )
    progress.record_chapter_chunk(
        "chapter-hash",
        1,
        2,
        "<p>Good output</p>",
        budget,
        source_text="<p>Source one</p>",
    )
    failing_chunk = "<p>BADTOKEN output</p>" + ("unwrapped prose " * 8)
    progress.record_chapter_chunk(
        "chapter-hash",
        2,
        2,
        failing_chunk,
        budget,
        source_text="<p>Source two</p>",
    )
    progress.mark_chapter_chunk_progress_status("chapter-hash", "completed")
    output_path = tmp_path / "chapter.html"
    output_path.write_text(
        "\n".join(
            [
                wrap_chunk_html(
                    "chapter-hash", 1, 2, "<p>Good output</p>"
                ),
                wrap_chunk_html(
                    "chapter-hash", 2, 2, failing_chunk
                ),
            ]
        ),
        encoding="utf-8",
    )
    progress.save()

    logs = []
    faulty = {
        "filename": "chapter.html",
        "filepath": str(output_path),
        "file_index": 0,
        "chapter_num": 1,
        "issues": [
            "llm_token_issue: 'BADTOKEN'",
            "unwrapped_text_content",
        ],
        "qa_issue_previews": {},
        "duplicate_confidence": 0,
    }
    assert _attach_chunk_results_to_scan(
        str(tmp_path),
        [faulty],
        {},
        logs.append,
        progress_path=progress.PROGRESS_FILE,
    ) == 1
    chunk_issues = {
        result["chunk_index"]: result["issues"]
        for result in faulty["chunk_results"]
    }
    assert chunk_issues[1] == []
    assert set(chunk_issues[2]) == {
        "llm_token_issue: 'BADTOKEN'",
        "unwrapped_text_content",
    }
    update_new_format_progress(
        progress.prog, [faulty], [], logs.append, str(tmp_path)
    )

    chapter = progress.prog["chapters"]["1"]
    entry = progress.prog["chapter_chunks"]["chapter-hash"]
    assert chapter["status"] == "completed"
    assert chapter["has_chunk_qa_failures"] is True
    assert entry["entries"]["1"]["status"] == "completed"
    assert entry["entries"]["2"]["status"] == "qa_failed"
    assert progress.get_reusable_chapter_chunks("chapter-hash") == {
        "1": "<p>Good output</p>"
    }

    resolved = {
        "filename": "chapter.html",
        "filepath": str(output_path),
        "file_index": 0,
        "chapter_num": 1,
        "issues": [],
        "qa_issue_previews": {},
        "duplicate_confidence": 0,
    }
    assert _attach_chunk_results_to_scan(
        str(tmp_path),
        [resolved],
        {},
        logs.append,
        progress_path=progress.PROGRESS_FILE,
    ) == 1
    update_new_format_progress(
        progress.prog, [], [resolved], logs.append, str(tmp_path)
    )

    assert chapter["status"] == "completed"
    assert chapter["has_chunk_qa_failures"] is False
    assert entry["entries"]["1"]["status"] == "completed"
    assert entry["entries"]["2"]["status"] == "completed"

    output_path.write_text(
        wrap_chunk_html(
            "chapter-hash", 1, 2, "<p>Good output</p>"
        ),
        encoding="utf-8",
    )
    progress.save()
    missing = {
        "filename": "chapter.html",
        "filepath": str(output_path),
        "file_index": 0,
        "chapter_num": 1,
        "issues": [],
        "qa_issue_previews": {},
        "duplicate_confidence": 0,
    }
    assert _attach_chunk_results_to_scan(
        str(tmp_path),
        [missing],
        {},
        logs.append,
        progress_path=progress.PROGRESS_FILE,
    ) == 1
    assert missing["issues"] == ["missing_chunk_segment"]
    update_new_format_progress(
        progress.prog, [missing], [], logs.append, str(tmp_path)
    )
    assert chapter["status"] == "completed"
    assert entry["entries"]["1"]["status"] == "completed"
    assert entry["entries"]["2"]["status"] == "qa_failed"

    progress.reset_chapter_chunks("chapter-hash", [2])
    chapter["status"] = "pending"
    progress.save()
    incomplete = {
        "filename": "chapter.html",
        "filepath": str(output_path),
        "file_index": 0,
        "chapter_num": 1,
        "issues": [],
        "qa_issue_previews": {},
        "duplicate_confidence": 0,
    }
    assert _attach_chunk_results_to_scan(
        str(tmp_path),
        [incomplete],
        {},
        logs.append,
        progress_path=progress.PROGRESS_FILE,
    ) == 1
    update_new_format_progress(
        progress.prog, [], [incomplete], logs.append, str(tmp_path)
    )

    assert chapter["status"] == "pending"
    assert entry["chapter_status"] == "incomplete"
    assert entry["entries"]["1"]["status"] == "completed"
    assert entry["entries"]["2"]["status"] == "pending"


@pytest.mark.parametrize(
    ("finish_reason", "save_partial", "save_prohibited"),
    [
        ("prohibited_content", "0", "1"),
        ("length", "1", "0"),
    ],
)
def test_batch_failure_toggles_save_provider_output(
    tmp_path,
    monkeypatch,
    finish_reason,
    save_partial,
    save_prohibited,
):
    monkeypatch.setenv("PRESERVE_ORIGINAL_TEXT_ON_FAILURE", "0")
    monkeypatch.setenv("SAVE_PARTIAL_RESULTS", save_partial)
    monkeypatch.setenv("SAVE_PROHIBITED_RESULTS", save_prohibited)
    monkeypatch.setenv("RETRY_TRUNCATED", "1")

    _source, saved = _run_batch_chapter_failure(
        tmp_path,
        monkeypatch,
        finish_reason=finish_reason,
    )

    assert saved == "streamed response"


@pytest.mark.parametrize(
    "finish_reason",
    ["prohibited_content", "length", "error"],
)
def test_batch_preserve_original_wins_for_every_returned_failure(
    tmp_path,
    monkeypatch,
    finish_reason,
):
    monkeypatch.setenv("PRESERVE_ORIGINAL_TEXT_ON_FAILURE", "1")
    monkeypatch.setenv("SAVE_PARTIAL_RESULTS", "0")
    monkeypatch.setenv("SAVE_PROHIBITED_RESULTS", "0")
    monkeypatch.setenv("RETRY_TRUNCATED", "1")

    source, saved = _run_batch_chapter_failure(
        tmp_path,
        monkeypatch,
        finish_reason=finish_reason,
    )

    assert saved == source


@pytest.mark.parametrize(
    ("finish_reason", "save_partial", "save_prohibited"),
    [
        ("prohibited_content", "0", "1"),
        ("error", "0", "1"),
        ("length", "1", "0"),
    ],
)
def test_batch_failure_toggle_yields_preserved_source_to_provider_output(
    tmp_path,
    monkeypatch,
    finish_reason,
    save_partial,
    save_prohibited,
):
    monkeypatch.setenv("PRESERVE_ORIGINAL_TEXT_ON_FAILURE", "1")
    monkeypatch.setenv("SAVE_PARTIAL_RESULTS", save_partial)
    monkeypatch.setenv("SAVE_PROHIBITED_RESULTS", save_prohibited)
    monkeypatch.setenv("RETRY_TRUNCATED", "1")

    _source, saved = _run_batch_chapter_failure(
        tmp_path,
        monkeypatch,
        finish_reason=finish_reason,
    )

    assert saved == "streamed response"


def test_batch_raised_api_error_saves_streamed_text(tmp_path, monkeypatch):
    from unified_api_client import UnifiedClientError

    monkeypatch.setenv("PRESERVE_ORIGINAL_TEXT_ON_FAILURE", "0")
    monkeypatch.setenv("SAVE_PARTIAL_RESULTS", "0")
    monkeypatch.setenv("SAVE_PROHIBITED_RESULTS", "1")
    error = UnifiedClientError(
        "provider API error",
        error_type="api_error",
        details={"partial_content": "streamed before API error"},
    )

    _source, saved = _run_batch_chapter_failure(
        tmp_path,
        monkeypatch,
        raised_error=error,
    )

    assert saved == "streamed before API error"


@pytest.mark.parametrize(
    ("error_type", "save_partial", "save_prohibited"),
    [
        ("api_error", "0", "1"),
        ("truncated", "1", "0"),
    ],
)
def test_batch_raised_failure_toggle_wins_over_preserved_source(
    tmp_path,
    monkeypatch,
    error_type,
    save_partial,
    save_prohibited,
):
    from unified_api_client import UnifiedClientError

    monkeypatch.setenv("PRESERVE_ORIGINAL_TEXT_ON_FAILURE", "1")
    monkeypatch.setenv("SAVE_PARTIAL_RESULTS", save_partial)
    monkeypatch.setenv("SAVE_PROHIBITED_RESULTS", save_prohibited)
    error = UnifiedClientError(
        "provider stream failed",
        error_type=error_type,
        details={"partial_content": "streamed before raised failure"},
    )

    _source, saved = _run_batch_chapter_failure(
        tmp_path,
        monkeypatch,
        raised_error=error,
    )

    assert saved == "streamed before raised failure"


def test_batch_preserve_original_handles_unclassified_exception(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("PRESERVE_ORIGINAL_TEXT_ON_FAILURE", "1")
    monkeypatch.setenv("SAVE_PARTIAL_RESULTS", "0")
    monkeypatch.setenv("SAVE_PROHIBITED_RESULTS", "0")

    source, saved = _run_batch_chapter_failure(
        tmp_path,
        monkeypatch,
        raised_error=RuntimeError("unexpected provider crash"),
    )

    assert saved == source


def test_google_free_failure_is_preserved_and_stays_a_failure(monkeypatch):
    from google_free_translate import GoogleFreeTranslateNew
    from unified_api_client import UnifiedClient

    monkeypatch.setenv("PRESERVE_ORIGINAL_TEXT_ON_FAILURE", "1")
    monkeypatch.setattr(
        GoogleFreeTranslateNew,
        "translate",
        lambda self, text: {
            "translatedText": text,
            "detectedSourceLanguage": "ko",
            "provider": "google",
            "error": "all endpoints failed",
        },
    )
    client = object.__new__(UnifiedClient)
    source = "raw source"

    response = client._send_google_translate_free(
        [{"role": "user", "content": source}],
    )

    assert response.content == source
    assert response.finish_reason == "error"


def test_terminal_blocked_response_returns_accumulated_stream_text(monkeypatch):
    from unified_api_client import UnifiedClient, UnifiedResponse

    monkeypatch.delenv("PRESERVE_ORIGINAL_TEXT_ON_FAILURE", raising=False)
    monkeypatch.setenv("MAX_RETRIES", "1")
    monkeypatch.setenv("USE_MULTI_API_KEYS", "0")
    monkeypatch.setenv("USE_FALLBACK_KEYS", "0")
    monkeypatch.setenv("USE_GLOSSARY_KEYS", "0")
    monkeypatch.setenv("USE_GLOSSARY_REFINEMENT_KEYS", "0")

    client = UnifiedClient(api_key="test-key", model="gpt-test")
    streamed = "translated text received before the block"
    response = UnifiedResponse(
        content=streamed,
        finish_reason="prohibited_content",
    )
    monkeypatch.setattr(client, "_get_response", lambda *args, **kwargs: response)
    monkeypatch.setattr(
        client,
        "_get_file_names",
        lambda messages, context=None: ("payload.json", "response.txt"),
    )
    for method_name in (
        "_save_payload",
        "_save_response",
        "_save_failed_request",
        "_track_stats",
        "_attach_usage_to_last_payload",
    ):
        monkeypatch.setattr(client, method_name, lambda *args, **kwargs: None)

    assert client._send_internal(
        messages=[{"role": "user", "content": "raw source"}],
        temperature=0.2,
        max_tokens=100,
        context="translation",
        request_id="streamed-block-test",
    ) == (streamed, "prohibited_content")


def test_ai_artifact_check_defaults_off():
    settings = default_qa_scan_settings()

    assert settings["check_ai_artifacts"] is False


def test_pdf_title_skip_setting_is_persisted_and_defaults_off():
    root = Path(__file__).resolve().parents[1]
    settings_source = (root / "src" / "other_settings.py").read_text(
        encoding="utf-8"
    )
    gui_source = (root / "src" / "translator_gui.py").read_text(
        encoding="utf-8"
    )
    translator_source = (root / "src" / "TransateKRtoEN.py").read_text(
        encoding="utf-8"
    )

    assert '"Skip .pdf book title translation"' in settings_source
    assert "self.config.get('skip_pdf_title_translation', False)" in settings_source
    assert "'SKIP_PDF_TITLE_TRANSLATION'" in gui_source
    assert "os.getenv('SKIP_PDF_TITLE_TRANSLATION', '0')" in translator_source


def test_pdf_source_structure_uses_filename_stem_as_book_title(tmp_path):
    from txt_processor import TextFileProcessor

    pdf_path = tmp_path / "[433975] ê°“ê²œì˜ ë””ë ‰í„°ê°€ ë˜ì—ˆë‹¤.pdf"
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    processor = TextFileProcessor(str(pdf_path), str(output_dir))
    processor.save_original_structure()

    metadata = json.loads(
        (output_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["type"] == "pdf"
    assert metadata["source_file"] == pdf_path.name
    assert metadata["title"] == pdf_path.stem


def test_pdf_source_structure_preserves_translation_only_for_same_filename(tmp_path):
    from txt_processor import build_source_structure_metadata

    translated = {
        "title": "Translated title",
        "original_title": "Original title",
        "title_translated": True,
    }

    same_source = build_source_structure_metadata(
        str(tmp_path / "Original title.pdf"), translated
    )
    assert same_source["title"] == "Translated title"
    assert same_source["original_title"] == "Original title"
    assert same_source["title_translated"] is True

    renamed_source = build_source_structure_metadata(
        str(tmp_path / "Updated title.pdf"), translated
    )
    assert renamed_source["title"] == "Updated title"
    assert "original_title" not in renamed_source
    assert "title_translated" not in renamed_source

    false_no_request_completion = build_source_structure_metadata(
        str(tmp_path / "Original title.pdf"),
        {"title": "Original title", "title_translated": True},
    )
    assert false_no_request_completion["title"] == "Original title"
    assert "title_translated" not in false_no_request_completion


def test_valid_html_doctype_is_not_an_ai_artifact():
    artifacts = detect_ai_artifacts(
        "<!DOCTYPE html>\n<html><body><p>Normal output.</p></body></html>"
    )

    assert not any(item["type"] == "ai_artifact_leading_line" for item in artifacts)


def test_ai_artifact_leading_phrases_are_customizable():
    disabled_default = detect_ai_artifacts(
        "Sure, here is the translation.",
        ai_artifact_patterns=[],
    )
    custom_phrase = detect_ai_artifacts(
        "Generated response: translated text",
        ai_artifact_patterns=["Generated response:"],
    )

    assert not any(
        item["type"] == "ai_artifact_leading_line"
        for item in disabled_default
    )
    assert any(
        item["type"] == "ai_artifact_leading_line"
        for item in custom_phrase
    )


def test_ai_artifact_phrases_are_forwarded_to_worker_environment():
    settings = default_qa_scan_settings()
    settings["ai_artifact_patterns"] = ["Generated response:"]
    settings["ai_artifact_patterns_are_regex"] = True

    previous = apply_qa_scan_env_from_settings(settings)
    try:
        assert json.loads(os.environ["QA_AI_ARTIFACT_PATTERNS_JSON"]) == [
            "Generated response:"
        ]
        assert os.environ["QA_AI_ARTIFACT_PATTERNS_ARE_REGEX"] == "1"
    finally:
        restore_env(previous)


@pytest.mark.parametrize(
    "text,expected_script",
    (
        ("A shrug ¯\\_(ツ)_/¯ in prose.", "Japanese_text_found"),
        ("A crying face (ㅠ_ㅠ) in prose.", "Korean_text_found"),
        ("An angry face щ(ﾟДﾟщ) in prose.", "Cyrillic_text_found"),
    ),
)
def test_multiscript_emoticons_are_only_ignored_when_toggle_is_enabled(
    text, expected_script
):
    settings = default_qa_scan_settings()

    has_issue, issues = detect_non_english_content(text, settings)
    assert has_issue is True
    assert any(expected_script in issue for issue in issues)

    settings["whitelist_emoticon_patterns"] = True
    assert detect_non_english_content(text, settings) == (False, [])


def test_emoticon_whitelist_only_masks_the_complete_pattern():
    settings = default_qa_scan_settings()
    settings["whitelist_emoticon_patterns"] = True

    has_issue, issues = detect_non_english_content(
        "Allowed face щ(ﾟДﾟщ), but this Д remains foreign.", settings
    )

    assert has_issue is True
    assert any("Cyrillic_text_found_1_chars_[Д]" in issue for issue in issues)
    assert not any("Japanese_text_found" in issue for issue in issues)


def test_custom_literal_and_regex_emoticon_patterns_are_supported():
    literal_settings = default_qa_scan_settings()
    literal_settings.update(
        {
            "whitelist_emoticon_patterns": True,
            "emoticon_patterns": ["(Я_Я)"],
            "emoticon_patterns_are_regex": False,
        }
    )
    assert detect_non_english_content("Custom face (Я_Я).", literal_settings) == (
        False,
        [],
    )
    assert detect_non_english_content("Different face (Ж_Ж).", literal_settings)[0] is True

    regex_settings = default_qa_scan_settings()
    regex_settings.update(
        {
            "whitelist_emoticon_patterns": True,
            "emoticon_patterns": [r"[ㅠㅜ]{2,}"],
            "emoticon_patterns_are_regex": True,
        }
    )
    assert detect_non_english_content("Variable crying ㅠㅜㅠㅜㅠ.", regex_settings) == (
        False,
        [],
    )


def test_invalid_emoticon_regex_is_skipped_without_disabling_valid_entries():
    text = mask_whitelisted_emoticons(
        "Keep (ㅠ_ㅠ), still flag Ж.",
        ["(", r"\(ㅠ_ㅠ\)"],
        patterns_are_regex=True,
    )

    assert "ㅠ" not in text
    assert "Ж" in text


def test_emoticon_whitelist_defaults_and_worker_environment_are_persisted():
    settings = default_qa_scan_settings()
    assert settings["whitelist_emoticon_patterns"] is False
    assert settings["emoticon_patterns"] == list(DEFAULT_EMOTICON_PATTERNS)
    assert settings["emoticon_patterns_are_regex"] is False

    settings.update(
        {
            "whitelist_emoticon_patterns": True,
            "emoticon_patterns": ["(Я_Я)", "(ㅠ_ㅠ)"],
            "emoticon_patterns_are_regex": True,
        }
    )
    previous = apply_qa_scan_env_from_settings(settings)
    try:
        assert os.environ["QA_WHITELIST_EMOTICON_PATTERNS"] == "1"
        assert json.loads(os.environ["QA_EMOTICON_PATTERNS_JSON"]) == [
            "(Я_Я)",
            "(ㅠ_ㅠ)",
        ]
        assert os.environ["QA_EMOTICON_PATTERNS_ARE_REGEX"] == "1"
    finally:
        restore_env(previous)


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


def test_pdf_progress_rows_include_toc_and_header_artifacts(tmp_path):
    output_dir = tmp_path / "book_PDF"
    output_dir.mkdir()
    (output_dir / "TOC.txt").write_text(
        "Original: 원본\nTranslated: Contents\n", encoding="utf-8"
    )
    (output_dir / "translated_headers.txt").write_text(
        "Original: 제목\nTranslated: Chapter\n", encoding="utf-8"
    )
    gui = RetranslationMixin()
    gui.config = {
        "use_toc_ncx": True,
        "batch_translate_headers": True,
    }
    prog = {"chapters": {}, "version": "2.1"}

    assert gui._ensure_translation_artifact_progress_entries(
        prog, str(output_dir), str(tmp_path / "book.pdf")
    ) is True

    rows = []
    gui._append_translation_artifact_display_info(
        {
            "file_path": str(tmp_path / "book.pdf"),
            "output_dir": str(output_dir),
            "prog": prog,
        },
        rows,
    )
    assert [row["output_file"] for row in rows] == [
        "TOC.txt",
        "translated_headers.txt",
    ]
    assert [row["status"] for row in rows] == ["completed", "completed"]


def test_pdf_progress_rows_replace_generic_artifact_rows_without_duplicates(tmp_path):
    output_dir = tmp_path / "book_PDF"
    output_dir.mkdir()
    (output_dir / "TOC.txt").write_text("Contents", encoding="utf-8")
    (output_dir / "translated_headers.txt").write_text(
        "Chapter", encoding="utf-8"
    )
    gui = RetranslationMixin()
    gui.config = {
        "use_toc_ncx": True,
        "batch_translate_headers": True,
    }
    prog = {"chapters": {}, "version": "2.1"}
    gui._ensure_translation_artifact_progress_entries(
        prog, str(output_dir), str(tmp_path / "book.pdf")
    )

    chapters = prog["chapters"]
    rows = [
        {
            "key": "__translation_artifact__:headers",
            "info": chapters["__translation_artifact__:headers"],
            "output_file": "translated_headers.txt",
            "status": "completed",
        },
        {
            "key": "__translation_artifact__:toc",
            "info": chapters["__translation_artifact__:toc"],
            "output_file": "TOC.txt",
            "status": "completed",
        },
        {
            "key": "chapter_000",
            "info": {"status": "completed"},
            "output_file": "response_pdf_section_000.html",
            "status": "completed",
        },
    ]

    gui._append_translation_artifact_display_info(
        {
            "file_path": str(tmp_path / "book.pdf"),
            "output_dir": str(output_dir),
            "prog": prog,
        },
        rows,
    )

    assert [row["output_file"] for row in rows] == [
        "TOC.txt",
        "translated_headers.txt",
        "response_pdf_section_000.html",
    ]
    assert sum(row["output_file"] == "TOC.txt" for row in rows) == 1
    assert sum(
        row["output_file"] == "translated_headers.txt" for row in rows
    ) == 1


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


@pytest.mark.parametrize(
    ("translation_type", "artifact_kind"),
    (("header", "headers"), ("toc", "toc")),
)
def test_partial_batch_translation_marks_artifact_failed(
    tmp_path, monkeypatch, translation_type, artifact_kind
):
    from metadata_batch_translator import BatchHeaderTranslator

    progress_path = tmp_path / "translation_progress.json"
    progress_path.write_text(
        json.dumps({"version": "2.1", "chapters": {}}), encoding="utf-8"
    )

    class Client:
        output_dir = str(tmp_path)
        model = "test-model"

    translator = BatchHeaderTranslator(Client(), {})
    monkeypatch.setattr(
        translator,
        "_send_with_retry",
        lambda **_kwargs: '{"1": "Chapter One"}',
    )

    translated = translator.translate_headers_batch(
        {1: "Original One", 2: "Original Two"},
        batch_size=2,
        translation_type=translation_type,
    )

    assert translated == {1: "Chapter One"}
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    entry = progress["chapters"][f"__translation_artifact__:{artifact_kind}"]
    assert entry["status"] == "failed"
    assert "1 of 2 entries translated" in entry["error_message"]
    assert "1 unresolved" in entry["error_message"]


@pytest.mark.parametrize(
    "translation_type", ("header", "toc")
)
def test_header_toc_batches_are_sequential_when_batch_toggle_is_off(
    tmp_path, monkeypatch, translation_type
):
    import threading
    import time

    from metadata_batch_translator import BatchHeaderTranslator

    class Client:
        output_dir = str(tmp_path)
        model = "test-model"

    monkeypatch.setenv("BATCH_TRANSLATION", "0")
    monkeypatch.setenv("EXTRACTION_WORKERS", "2")
    translator = BatchHeaderTranslator(Client(), {})
    active_calls = 0
    maximum_active_calls = 0
    call_count = 0
    call_lock = threading.Lock()

    def send_batch(**kwargs):
        nonlocal active_calls, maximum_active_calls, call_count
        payload = kwargs["messages"][-1]["content"].rsplit(
            "Titles to translate:\n", 1
        )[1]
        batch = json.loads(payload)
        with call_lock:
            active_calls += 1
            call_count += 1
            maximum_active_calls = max(maximum_active_calls, active_calls)
        time.sleep(0.05)
        with call_lock:
            active_calls -= 1
        return json.dumps({key: f"Translated {key}" for key in batch})

    monkeypatch.setattr(translator, "_send_with_retry", send_batch)
    translated = translator.translate_headers_batch(
        {1: "One", 2: "Two"},
        batch_size=1,
        translation_type=translation_type,
    )

    assert translated == {1: "Translated 1", 2: "Translated 2"}
    assert call_count == 2
    assert maximum_active_calls == 1


@pytest.mark.parametrize(
    "context", ("batch_header_translation", "batch_toc_translation")
)
def test_sequential_metadata_chunks_are_isolated_without_bypassing_lock(
    monkeypatch, context
):
    """Two sequential chunks get fresh clients while retaining serialization."""
    import unified_api_client as unified_module
    from TransateKRtoEN import send_with_interrupt
    from unified_api_client import UnifiedClient

    monkeypatch.setattr(
        UnifiedClient,
        "_setup_client",
        lambda self: setattr(self, "client_type", "test"),
    )
    client = UnifiedClient(api_key="main-key", model="gpt-test")

    class RecordingLock:
        def __init__(self):
            self.events = []

        def acquire(self):
            self.events.append("acquire")
            return True

        def release(self):
            self.events.append("release")

    send_lock = RecordingLock()
    internal_calls = []
    client._sequential_send_lock = send_lock
    monkeypatch.setenv("BATCH_TRANSLATION", "0")
    monkeypatch.setenv("THREAD_SUBMISSION_DELAY_SECONDS", "0")
    monkeypatch.setenv("SEND_INTERVAL_SECONDS", "0")
    monkeypatch.setenv("USE_METADATA_KEYS", "1")
    monkeypatch.setenv(
        "METADATA_API_KEYS",
        json.dumps([{"api_key": "metadata-key", "model": "metadata-model"}]),
    )
    monkeypatch.setattr(UnifiedClient, "_metadata_key_pool", None)
    monkeypatch.setattr(client, "reset_cleanup_state", lambda: None)
    monkeypatch.setattr(client, "_log_pre_stagger", lambda *_args: None)
    monkeypatch.setattr(client, "_apply_thread_submission_delay", lambda: None)
    monkeypatch.setattr(client, "_should_abort_retry", lambda: False)
    monkeypatch.setattr(client, "_extract_chapter_label", lambda _messages: None)
    monkeypatch.setattr(
        client,
        "_apply_dedicated_key_pool_override",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("metadata chunk mutated the shared client")
        ),
    )

    def fake_send_internal(temp_client, *_args, **_kwargs):
        internal_calls.append(temp_client)
        temp_client.last_actual_request_model = temp_client.model
        return "ok", "stop"

    monkeypatch.setattr(UnifiedClient, "_send_internal", fake_send_internal)
    monkeypatch.setattr(unified_module, "_api_watchdog_started", lambda *_a, **_k: None)
    monkeypatch.setattr(unified_module, "_api_watchdog_finished", lambda *_a, **_k: None)

    for _ in range(2):
        response = send_with_interrupt(
            messages=[{"role": "user", "content": "test"}],
            client=client,
            temperature=0.0,
            max_tokens=100,
            stop_check_fn=lambda: False,
            context=context,
        )
        assert response[0] == "ok"

    assert len(internal_calls) == 2
    assert all(temp_client is not client for temp_client in internal_calls)
    assert send_lock.events == ["acquire", "release", "acquire", "release"]


def test_batch_header_progress_uses_actual_metadata_key_model(tmp_path, monkeypatch):
    from metadata_batch_translator import BatchHeaderTranslator
    from unified_api_client import set_current_thread_actual_request_model

    progress_path = tmp_path / "translation_progress.json"
    progress_path.write_text(
        json.dumps({
            "version": "2.1",
            "chapters": {
                "__translation_artifact__:headers": {
                    "status": "completed",
                    "model_name": "main-key-model",
                }
            },
        }),
        encoding="utf-8",
    )

    class Client:
        output_dir = str(tmp_path)
        model = "main-key-model"

    translator = BatchHeaderTranslator(Client(), {"headers_per_batch": 1})

    def send_with_metadata_key(**_kwargs):
        queued = json.loads(progress_path.read_text(encoding="utf-8"))
        queued_entry = queued["chapters"]["__translation_artifact__:headers"]
        assert queued_entry["status"] == "in_progress"
        assert "model_name" not in queued_entry

        set_current_thread_actual_request_model(
            "metadata-key-model", "MetadataKey#1 (metadata-key-model)"
        )
        _kwargs["before_send_callback"]()
        live = json.loads(progress_path.read_text(encoding="utf-8"))
        live_entry = live["chapters"]["__translation_artifact__:headers"]
        assert live_entry["status"] == "in_progress"
        assert live_entry["model_name"] == "metadata-key-model"
        return '{"1": "Chapter One"}'

    monkeypatch.setattr(translator, "_send_with_retry", send_with_metadata_key)

    assert translator.translate_headers_batch(
        {1: "Original"}, batch_size=1, translation_type="header"
    ) == {1: "Chapter One"}

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    entry = progress["chapters"]["__translation_artifact__:headers"]
    assert entry["status"] == "completed"
    assert entry["model_name"] == "metadata-key-model"


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


def test_recycled_model_links_and_resets_both_artifact_rows(tmp_path):
    progress = {
        "chapters": {
            "__translation_artifact__:toc": {
                "output_file": "TOC.txt",
                "status": "completed",
                "model": " recycled ",
            },
            "__translation_artifact__:headers": {
                "output_file": "translated_headers.txt",
                "status": "completed",
                "model_name": "metadata-key-model",
            },
        }
    }

    assert translation_artifacts_are_recycled_linked(progress)
    assert reset_translation_artifact_progress_entries(
        progress, ("toc", "headers")
    ) == 2

    for entry in progress["chapters"].values():
        assert entry["status"] == "pending"
        assert entry["content_hash"] == ""
        assert "model" not in entry
        assert "model_name" not in entry
    assert not translation_artifacts_are_recycled_linked(progress)

    (tmp_path / "toc.txt").write_text("TOC", encoding="utf-8")
    assert os.path.normcase(translation_artifact_path(
        str(tmp_path), "toc", existing_only=True
    )) == os.path.normcase(str(tmp_path / "toc.txt"))


class _ArtifactDeleteMessageBox:
    Critical = 1
    Information = 2
    Question = 3
    Yes = 4
    No = 8
    Cancel = 16

    class ButtonRole:
        DestructiveRole = 1
        AcceptRole = 2

    next_click = "Delete Both Linked Files"
    shown_texts = []

    def __init__(self, *_args, **_kwargs):
        self._text = ""
        self._buttons = {}
        self._clicked = None

    def setIcon(self, _icon):
        pass

    def setWindowTitle(self, _title):
        pass

    def setText(self, text):
        self._text = text

    def setStandardButtons(self, _buttons):
        pass

    def setDefaultButton(self, _button):
        pass

    def setWindowIcon(self, _icon):
        pass

    def setStyleSheet(self, _style):
        pass

    def addButton(self, label, _role=None):
        button = object()
        button_label = "Cancel" if label == self.Cancel else str(label)
        self._buttons[button_label] = button
        return button

    def exec(self):
        type(self).shown_texts.append(self._text)
        self._clicked = self._buttons.get(type(self).next_click)
        return 0

    def clickedButton(self):
        return self._clicked


class _ArtifactDeleteGui:
    def __init__(self, epub_path):
        self.selected_files = [str(epub_path)]
        self.config = {}
        self.logs = []

    def append_log(self, message):
        self.logs.append(message)

    def get_current_epub_path(self):
        return self.selected_files[0]


def _write_recycled_artifact_workspace(root, book_name):
    epub_path = root / f"{book_name}.epub"
    epub_path.write_bytes(b"")
    output_dir = root / book_name
    output_dir.mkdir()
    (output_dir / "chapter.xhtml").write_text("<p>chapter</p>", encoding="utf-8")
    (output_dir / "TOC.txt").write_text("toc", encoding="utf-8")
    (output_dir / "translated_headers.txt").write_text(
        "headers", encoding="utf-8"
    )
    (output_dir / "translation_progress.json").write_text(
        json.dumps({
            "chapters": {
                "__translation_artifact__:toc": {
                    "output_file": "TOC.txt",
                    "status": "completed",
                    "model_name": "RECYCLED",
                },
                "__translation_artifact__:headers": {
                    "output_file": "translated_headers.txt",
                    "status": "completed",
                    "model_name": "metadata-key-model",
                },
            }
        }),
        encoding="utf-8",
    )
    return epub_path, output_dir


def test_manual_delete_buttons_offer_and_delete_both_recycled_files(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(other_settings, "QMessageBox", _ArtifactDeleteMessageBox)
    monkeypatch.setattr(other_settings, "QIcon", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        other_settings, "_center_messagebox_buttons", lambda _message_box: None
    )
    _ArtifactDeleteMessageBox.next_click = "Delete Both Linked Files"

    for delete_function, book_name in (
        (other_settings.delete_translated_headers_file, "HeaderDelete"),
        (other_settings.delete_toc_txt_file, "TocDelete"),
    ):
        root = tmp_path / book_name
        root.mkdir()
        epub_path, output_dir = _write_recycled_artifact_workspace(
            root, book_name
        )
        monkeypatch.chdir(root)
        _ArtifactDeleteMessageBox.shown_texts = []

        delete_function(_ArtifactDeleteGui(epub_path))

        assert not (output_dir / "TOC.txt").exists()
        assert not (output_dir / "translated_headers.txt").exists()
        progress = json.loads(
            (output_dir / "translation_progress.json").read_text(
                encoding="utf-8"
            )
        )
        for entry in progress["chapters"].values():
            assert entry["status"] == "pending"
            assert "model_name" not in entry
        assert any(
            "without a new API translation" in text
            and "Delete both linked files" in text
            for text in _ArtifactDeleteMessageBox.shown_texts
        )


def test_manual_header_delete_can_keep_recycled_toc_file(tmp_path, monkeypatch):
    monkeypatch.setattr(other_settings, "QMessageBox", _ArtifactDeleteMessageBox)
    monkeypatch.setattr(other_settings, "QIcon", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        other_settings, "_center_messagebox_buttons", lambda _message_box: None
    )
    _ArtifactDeleteMessageBox.next_click = "Delete Only Header Files"
    epub_path, output_dir = _write_recycled_artifact_workspace(
        tmp_path, "KeepToc"
    )
    monkeypatch.chdir(tmp_path)

    other_settings.delete_translated_headers_file(
        _ArtifactDeleteGui(epub_path)
    )

    assert (output_dir / "TOC.txt").exists()
    assert not (output_dir / "translated_headers.txt").exists()
    progress = json.loads(
        (output_dir / "translation_progress.json").read_text(encoding="utf-8")
    )
    assert progress["chapters"]["__translation_artifact__:toc"][
        "model_name"
    ] == "RECYCLED"
    assert "model_name" not in progress["chapters"][
        "__translation_artifact__:headers"
    ]


def test_retranslate_recycled_artifact_dialog_offers_both_keep_and_cancel(
    monkeypatch,
):
    monkeypatch.setattr(
        retranslation_gui_module,
        "QMessageBox",
        _ArtifactDeleteMessageBox,
    )

    _ArtifactDeleteMessageBox.next_click = "Delete Both Linked Files"
    assert RetranslationMixin._recycled_artifact_retranslation_choice(
        None, "linked warning", "TOC.txt"
    ) == "both"

    _ArtifactDeleteMessageBox.next_click = "Keep TOC.txt"
    assert RetranslationMixin._recycled_artifact_retranslation_choice(
        None, "linked warning", "TOC.txt"
    ) == "selected_only"

    _ArtifactDeleteMessageBox.next_click = "Cancel"
    assert RetranslationMixin._recycled_artifact_retranslation_choice(
        None, "linked warning", "TOC.txt"
    ) == "cancel"


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


def test_do_not_skip_updates_live_other_settings_keyword_editors(monkeypatch):
    class FakeKeywordEditor:
        def __init__(self, text):
            self.text = text
            self.signals_blocked = False

        def toPlainText(self):
            return self.text

        def setPlainText(self, text):
            self.text = text

        def blockSignals(self, blocked):
            previous = self.signals_blocked
            self.signals_blocked = bool(blocked)
            return previous

    substring_text = "title, toc, colophon, appendix"
    exact_text = "index, glossary, glossary_extension"
    gui = RetranslationMixin()
    gui.config = {
        "special_file_keywords": substring_text,
        "special_file_exact": exact_text,
    }
    gui.special_file_keywords_var = substring_text
    gui.special_file_exact_var = exact_text
    gui._special_file_keywords_edit = FakeKeywordEditor(substring_text)
    gui._special_file_exact_edit = FakeKeywordEditor(exact_text)
    saved = []
    gui.save_config = lambda show_message=False: saved.append(show_message)
    monkeypatch.setenv("SPECIAL_FILE_KEYWORDS", substring_text)
    monkeypatch.setenv("SPECIAL_FILE_EXACT", exact_text)

    assert gui._remove_special_skip_keyword("colophon") is True
    assert gui.special_file_keywords_var == "title, toc, appendix"
    assert gui.config["special_file_keywords"] == "title, toc, appendix"
    assert os.environ["SPECIAL_FILE_KEYWORDS"] == "title, toc, appendix"
    assert gui._special_file_keywords_edit.toPlainText() == "title, toc, appendix"
    assert gui._special_file_exact_edit.toPlainText() == exact_text

    assert gui._remove_special_skip_keyword("index") is True
    assert gui.special_file_exact_var == "glossary, glossary_extension"
    assert gui.config["special_file_exact"] == "glossary, glossary_extension"
    assert os.environ["SPECIAL_FILE_EXACT"] == "glossary, glossary_extension"
    assert gui._special_file_exact_edit.toPlainText() == "glossary, glossary_extension"
    assert saved == [False, False]


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


def test_pdf_scan_skips_combined_html_and_checks_only_translated_sidecars(
    tmp_path,
):
    source_pdf = tmp_path / "The Reincarnated Girl's Resume.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    workspace = tmp_path / "pdf_workspace"
    workspace.mkdir()
    (workspace / "response_pdf_section_001.html").write_text(
        "<html><body><h1>Translated Chapter</h1>"
        "<p>English bookmark text.</p></body></html>",
        encoding="utf-8",
    )
    combined_filename = "The Reincarnated Girl's Resume_translated.html"
    (workspace / combined_filename).write_text(
        "<html><body><p>\u5931\u8d25\u7ffb\u8bd1 \u672c\u6587</p></body></html>",
        encoding="utf-8",
    )
    (workspace / "source_epub.txt").write_text(
        "\u539f\u59cb PDF control path", encoding="utf-8"
    )
    (workspace / "TOC.txt").write_text(
        "Original: \u539f\u59cb\u76ee\u5f55\nTranslated: Table of Contents\n",
        encoding="utf-8",
    )
    (workspace / "translated_headers.txt").write_text(
        "Original: \u539f\u59cb\u6807\u9898\nTranslated: Translated Chapter\n",
        encoding="utf-8",
    )
    progress_path = workspace / "translation_progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "version": "2.1",
                "chapters": {
                    "1": {
                        "actual_num": 1,
                        "output_file": "response_pdf_section_001.html",
                        "status": "completed",
                        "pdf_toc_section": True,
                    },
                    "__translation_artifact__:toc": {
                        "actual_num": -2,
                        "output_file": "TOC.txt",
                        "status": "qa_failed",
                        "special_type": "toc",
                        "qa_issues_found": ["old_issue"],
                    },
                    "__translation_artifact__:headers": {
                        "actual_num": -3,
                        "output_file": "translated_headers.txt",
                        "status": "qa_failed",
                        "special_type": "headers",
                        "qa_issues_found": ["old_issue"],
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
    logs = []

    scan_html_folder(
        str(workspace),
        log=logs.append,
        mode="quick-scan",
        qa_settings=settings,
        epub_path=str(source_pdf),
        progress_path=str(progress_path),
    )

    report = json.loads(
        (
            workspace
            / f"{workspace.name}_Scan Report"
            / "validation_results.json"
        ).read_text(encoding="utf-8")
    )
    by_name = {row["filename"]: row for row in report}
    assert "response_pdf_section_001.html" in by_name
    assert "TOC.txt" in by_name
    assert "translated_headers.txt" in by_name
    assert combined_filename not in by_name
    assert "source_epub.txt" not in by_name
    assert by_name["TOC.txt"]["issues"] == []
    assert by_name["translated_headers.txt"]["issues"] == []
    assert not any(combined_filename in message for message in logs)

    updated_progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert not any(
        entry.get("output_file") == combined_filename
        for entry in updated_progress["chapters"].values()
    )
    for key in (
        "__translation_artifact__:toc",
        "__translation_artifact__:headers",
    ):
        assert updated_progress["chapters"][key]["status"] == "completed"
        assert updated_progress["chapters"][key].get("qa_issues_found", []) == []


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


@pytest.mark.parametrize(
    "multipass_mode",
    ["full", "full_with_raw", "failed", "partial", "partial.b", "partial.b2"],
)
def test_every_multipass_mode_uses_authoritative_chapter_range_marker(
    multipass_mode, monkeypatch
):
    monkeypatch.setenv("CHAPTER_RANGE", "2-3")
    monkeypatch.setenv("USE_SPINE_ORDER", "0")
    chapters = [
        {
            "actual_chapter_num": 1,
            "_range_allowed_for_translation": False,
            "original_basename": "chapter0001.xhtml",
        },
        {
            "actual_chapter_num": 2,
            "_range_allowed_for_translation": True,
            "original_basename": "chapter0002.xhtml",
        },
        {
            "actual_chapter_num": 3,
            "_range_allowed_for_translation": True,
            "original_basename": "chapter0003.xhtml",
        },
        {
            "actual_chapter_num": 4,
            "_range_allowed_for_translation": False,
            "original_basename": "chapter0004.xhtml",
        },
    ]

    # The candidate helper is shared by every mode in the multipass block.
    selected = _filter_multipass_chapters_to_current_range(chapters)

    assert multipass_mode in translation_module.MULTIPASS_REFINEMENT_MODES
    assert [chapter["actual_chapter_num"] for chapter in selected] == [2, 3]


def test_multipass_range_filter_honors_spine_marker_over_chapter_number(monkeypatch):
    monkeypatch.setenv("CHAPTER_RANGE", "5-6")
    monkeypatch.setenv("USE_SPINE_ORDER", "1")
    chapters = [
        {
            "actual_chapter_num": 99,
            "_range_spine_position": 5,
            "_range_allowed_for_translation": True,
        },
        {
            "actual_chapter_num": 5,
            "_range_spine_position": 9,
            "_range_allowed_for_translation": False,
        },
    ]

    selected = _filter_multipass_chapters_to_current_range(chapters)

    assert selected == [chapters[0]]


def test_partial_b_does_not_append_whole_book_artifacts_to_range(tmp_path, monkeypatch):
    (tmp_path / "metadata.json").write_text("payload", encoding="utf-8")
    progress = ProgressManager(str(tmp_path))
    progress.prog["chapters"] = {
        "__metadata__": {
            "actual_num": -1,
            "output_file": "metadata.json",
            "status": "qa_failed",
            "qa_issues_found": ["Chinese_text_found_2_chars_[失败]"],
        }
    }

    class Config:
        MULTIPASS_REFINEMENT_MODE = "partial.b"

    monkeypatch.setenv("CHAPTER_RANGE", "2-3")
    monkeypatch.setenv("TRANSLATE_BOOK_TITLE", "1")
    in_range_chapter = {"actual_chapter_num": 2, "filename": "chapter0002.xhtml"}

    chapters = _append_partial_b_translation_artifact_chapters(
        [in_range_chapter], str(tmp_path), progress, Config()
    )

    assert chapters == [in_range_chapter]


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


def test_prepare_multipass_filters_existing_failures_by_spine_preview(
    tmp_path, monkeypatch
):
    source_path = tmp_path / "book.epub"
    source_path.write_bytes(b"preview is stubbed")
    failures = [
        {
            "source": "book.epub",
            "source_path": str(source_path),
            "progress_path": str(tmp_path / "translation_progress.json"),
            "progress_key": "inside",
            "chapter": 99,
            "original_basename": "chapter0099.xhtml",
            "output_file": "response_chapter0099.xhtml",
            "issues": ["Chinese_text_found_2_chars_[失败]"],
        },
        {
            "source": "book.epub",
            "source_path": str(source_path),
            "progress_path": str(tmp_path / "translation_progress.json"),
            "progress_key": "outside",
            "chapter": 2,
            "original_basename": "chapter0002.xhtml",
            "output_file": "response_chapter0002.xhtml",
            "issues": ["Chinese_text_found_2_chars_[失败]"],
        },
    ]

    class TextEntry:
        @staticmethod
        def text():
            return "5-5"

    class Checked:
        @staticmethod
        def isChecked():
            return True

    class Dummy:
        config = {"chapter_range": "", "use_spine_order": False}
        chapter_range_entry = TextEntry()
        use_spine_order_checkbox = Checked()
        translate_special_files_var = False
        _translation_qa_failure_key = TranslatorGUI._translation_qa_failure_key
        _qa_failure_matches_resolution_request = staticmethod(
            TranslatorGUI._qa_failure_matches_resolution_request
        )

        def _get_output_mode(self):
            return "translation"

        def _collect_translation_qa_failures(
            self, files=None, *, foreign_character_only=False
        ):
            return list(failures)

        def _get_spine_filenames_for_preview(self, *_args, **_kwargs):
            return [("[005]", "chapter0099.xhtml", False)]

    dummy = Dummy()
    monkeypatch.delenv("PARTIAL_B_TARGET_PROGRESS_KEY", raising=False)
    monkeypatch.delenv("PARTIAL_B_TARGET_OUTPUT_FILE", raising=False)
    monkeypatch.delenv("PARTIAL_B_TARGET_ACTUAL_NUM", raising=False)

    targeted = TranslatorGUI._prepare_multipass_qa_refinement_run(
        dummy, True, "failed"
    )

    assert [failure["progress_key"] for failure in targeted] == ["inside"]
    assert dummy._translation_run_followup_translation_after_refinement is False


def test_blank_live_chapter_range_clears_stale_runtime_scope(monkeypatch):
    class BlankEntry:
        @staticmethod
        def text():
            return ""

    class Unchecked:
        @staticmethod
        def isChecked():
            return False

    class Dummy:
        config = {"chapter_range": "9-12", "use_spine_order": True}
        chapter_range_entry = BlankEntry()
        use_spine_order_checkbox = Unchecked()

    dummy = Dummy()
    monkeypatch.setenv("CHAPTER_RANGE", "9-12")
    monkeypatch.setenv("USE_SPINE_ORDER", "1")

    exported = TranslatorGUI._export_chapter_range_runtime_env(dummy)

    assert exported == ("", None, False)
    assert "CHAPTER_RANGE" not in os.environ
    assert "USE_SPINE_ORDER" not in os.environ
    assert dummy.config["chapter_range"] == ""


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
