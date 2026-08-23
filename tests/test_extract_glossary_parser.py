import json
import os
import threading
import time
import zipfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRect, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialogButtonBox,
    QMessageBox,
    QComboBox,
    QListWidget,
    QListWidgetItem,
)

from extract_glossary_from_epub import (
    DEFAULT_GLOSSARY_PROMPT,
    extract_chapters_from_epub,
    extract_chapters_from_subtitle,
    is_subtitle_glossary_source,
    parse_api_response,
    skip_duplicate_entries,
)
from parallel_epub_glossary import (
    DEFAULT_PARALLEL_EPUB_WRAPPER_PROMPT,
    PARALLEL_EPUB_SYSTEM_INSTRUCTIONS,
    apply_parallel_epub_wrapper,
    auto_map_epub_chapters,
    compact_parallel_epub_selection,
    default_parallel_epub_system_prompt,
    parallel_epub_working_filename,
    ParallelEpubPairDialog,
    restore_parallel_epub_pairs,
    write_parallel_epub,
)
from glossary_refinement import (
    load_refinement_progress,
    refine_glossary_entries,
    refinement_chunking_mode,
)
from GlossaryManager_GUI import (
    GlossaryManagerMixin,
    _GlossaryFilterItemDelegate,
    _collect_glossary_filter_values,
    _recent_glossary_filter_dismissal,
)


class _RefinementTestSplitter:
    @staticmethod
    def count_tokens(text):
        return len(str(text or ""))

    @staticmethod
    def split_chapter(text, available_tokens, filename=None):
        return [(text, 1, 1)]


def _enable_refinement(monkeypatch):
    monkeypatch.setenv("GLOSSARY_REFINEMENT_ENABLED", "1")
    monkeypatch.setenv("GLOSSARY_REFINEMENT_TYPE_MODE", "all")
    monkeypatch.setenv("GLOSSARY_REFINEMENT_CHUNKING_MODE", "separate")
    monkeypatch.setenv("GLOSSARY_REFINEMENT_SKIP_DEDUPE", "1")
    monkeypatch.setenv("GLOSSARY_CUSTOM_FIELDS", json.dumps(["description"]))


def test_glossary_refinement_request_mode_defaults_to_all_types(monkeypatch):
    monkeypatch.delenv("GLOSSARY_REFINEMENT_CHUNKING_MODE", raising=False)
    assert refinement_chunking_mode() == "all"
    assert "GLOSSARY_REFINEMENT_CHUNKING_MODE" not in os.environ

    monkeypatch.setenv("GLOSSARY_REFINEMENT_CHUNKING_MODE", "separate")
    assert refinement_chunking_mode() == "separate"


def test_glossary_refinement_gui_only_applies_explicit_request_mode():
    resolve = GlossaryManagerMixin._configured_glossary_refinement_chunking_mode
    combo_index = GlossaryManagerMixin._glossary_refinement_chunking_combo_index
    config = {}

    assert resolve(config) is None
    assert config == {}
    assert resolve({"glossary_refinement_chunking_mode": "unknown"}) is None
    assert resolve({"glossary_refinement_chunking_mode": "separate"}) == "separate"
    assert resolve({"glossary_refinement_chunking_mode": "all"}) == "all"
    assert combo_index(config) == 1
    assert combo_index({"glossary_refinement_chunking_mode": "unknown"}) == 1
    assert combo_index({"glossary_refinement_chunking_mode": "separate"}) == 0
    assert combo_index({"glossary_refinement_chunking_mode": "all"}) == 1


def test_glossary_filter_apply_uses_all_checked_values_without_a_search():
    states = [
        ("visible", True, True),
        ("hidden", False, True),
        ("unchecked", True, False),
    ]

    assert _collect_glossary_filter_values(states) == {"visible", "hidden"}


def test_glossary_filter_apply_limits_search_to_visible_checked_values():
    states = [
        ("BIG SISTER IS WATCHING YOU", True, True),
        ("hidden but checked", False, True),
        ("visible but unchecked", True, False),
    ]

    assert _collect_glossary_filter_values(
        states,
        restrict_to_visible=True,
    ) == {"BIG SISTER IS WATCHING YOU"}


def test_glossary_filter_same_header_click_suppresses_immediate_reopen():
    dismissed = (3, 100.0)

    assert _recent_glossary_filter_dismissal(dismissed, 3, now=100.1)
    assert not _recent_glossary_filter_dismissal(dismissed, 2, now=100.1)
    assert not _recent_glossary_filter_dismissal(dismissed, 3, now=100.3)


def test_glossary_filter_search_interactions_are_wired():
    source = (
        Path(__file__).resolve().parents[1] / "src" / "GlossaryManager_GUI.py"
    ).read_text(encoding="utf-8")
    filter_start = source.index("def _open_glossary_column_filter")
    filter_end = source.index(
        "self.glossary_tree.header().sectionClicked",
        filter_start,
    )
    filter_body = source[filter_start:filter_end]

    assert "search_entry.returnPressed.connect(_apply_selected_values)" in source
    assert "value_list.setUniformItemSizes(True)" in filter_body
    assert "value_list.setItemDelegate(_GlossaryFilterItemDelegate(value_list))" in filter_body
    assert "Qt.ItemIsUserCheckable" in filter_body
    assert "value_list.setItemWidget" not in filter_body
    assert "_set_filter_item_checked(list_item, matches)" in filter_body
    assert "value_list.itemClicked.connect(_toggle_filter_item)" in filter_body
    assert "selection_before_search" in source
    assert "filter_timer.setInterval(60)" in source


def test_glossary_filter_delegate_renders_without_checkbox_widgets():
    app = QApplication.instance() or QApplication([])
    value_list = QListWidget()
    value_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
    value_list.setItemDelegate(_GlossaryFilterItemDelegate(value_list))
    item = QListWidgetItem("A Rank")
    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
    item.setCheckState(Qt.Checked)
    value_list.addItem(item)
    value_list.resize(320, 80)
    value_list.show()
    app.processEvents()

    assert not value_list.grab().isNull()
    assert value_list.findChildren(QCheckBox) == []
    value_list.deleteLater()
    app.processEvents()


def _run_test_refinement(entries, progress_file, parsed_entries, send_fn):
    return refine_glossary_entries(
        entries,
        client=None,
        temp=0.1,
        mtoks=4096,
        check_stop=lambda: False,
        chapter_splitter=_RefinementTestSplitter(),
        available_tokens=100000,
        chunk_timeout=None,
        parse_response_fn=lambda _response: [dict(entry) for entry in parsed_entries],
        dedupe_fn=lambda value: value,
        custom_entry_types_fn=lambda: {"character": {"enabled": True}},
        send_fn=send_fn,
        progress_file=str(progress_file),
        output_path=str(progress_file.parent / "glossary.csv"),
        log=lambda _message: None,
    )


def _set_glossary_env(monkeypatch):
    monkeypatch.setenv("GLOSSARY_CUSTOM_FIELDS", json.dumps(["description"]))
    monkeypatch.setenv(
        "GLOSSARY_CUSTOM_ENTRY_TYPES",
        json.dumps(
            {
                "item": {"enabled": True, "has_gender": False},
                "system_term": {"enabled": True, "has_gender": False},
                "title": {"enabled": True, "has_gender": True},
                "character": {"enabled": True, "has_gender": True},
            }
        ),
    )
    monkeypatch.setenv("GLOSSARY_ENTRY_TYPE_FILTER_MODE", "strict")


def test_no_header_non_gender_description_with_and_without_blank_gender(monkeypatch):
    _set_glossary_env(monkeypatch)
    response = "\n".join(
        [
            'item,\ub9c8\uc815\uc11d,Mana Stone,"Crystals containing magical energy."',
            'item,\ub9c8\uc774\ud06c,Microphone,,"A device created by Damian."',
            'system_term,\ud06c\ub85c\uc2dc\uc548\ub825,Crossian Calendar,,"The chronological system used within the Crossian Principality."',
        ]
    )

    entries = parse_api_response(response)

    assert [entry["translated_name"] for entry in entries] == [
        "Mana Stone",
        "Microphone",
        "Crossian Calendar",
    ]
    assert [entry["description"] for entry in entries] == [
        "Crystals containing magical energy.",
        "A device created by Damian.",
        "The chronological system used within the Crossian Principality.",
    ]


def test_no_header_gender_type_description_without_gender_column(monkeypatch):
    _set_glossary_env(monkeypatch)
    response = 'title,\uc18c\ub4dc\ub9c8\uc2a4\ud130,Sword Master,"The pinnacle of swordsmanship."'

    entries = parse_api_response(response)

    assert len(entries) == 1
    assert entries[0]["gender"] == "Unknown"
    assert entries[0]["description"] == "The pinnacle of swordsmanship."


def test_no_header_gender_type_keeps_real_gender_before_description(monkeypatch):
    _set_glossary_env(monkeypatch)
    response = 'character,\ub2e4\ubbf8\uc548,Damian,male,"A creator of mana devices."'

    entries = parse_api_response(response)

    assert len(entries) == 1
    assert entries[0]["gender"] == "male"
    assert entries[0]["description"] == "A creator of mana devices."


def test_dedup_keeps_later_description_when_description_is_active(monkeypatch):
    _set_glossary_env(monkeypatch)
    monkeypatch.setenv("GLOSSARY_USE_ADVANCED_DETECTION", "0")
    monkeypatch.setenv("GLOSSARY_DEDUPE_TRANSLATIONS", "0")

    entries = [
        {"type": "item", "raw_name": "\ub9c8\uc774\ud06c", "translated_name": "Microphone"},
        {
            "type": "item",
            "raw_name": "\ub9c8\uc774\ud06c",
            "translated_name": "Microphone",
            "description": "A device created by Damian.",
        },
    ]

    result = skip_duplicate_entries(entries)

    assert len(result) == 1
    assert result[0]["description"] == "A device created by Damian."


def test_completed_refinement_uses_stable_identities_and_only_reopens_for_new_entries(
    tmp_path,
    monkeypatch,
):
    _enable_refinement(monkeypatch)
    progress_file = tmp_path / "glossary_progress.json"
    calls = []
    parsed_entries = [
        {
            "type": "character",
            "raw_name": "Alice",
            "translated_name": "Alicia",
            "description": "Refined description",
        },
        {
            "type": "character",
            "raw_name": "Bob",
            "translated_name": "Robert",
        },
    ]

    def send_fn(*_args, **_kwargs):
        calls.append("sent")
        return "ignored", "stop", None

    initial = [
        {"type": "character", "raw_name": "Alice", "translated_name": "Alice"},
        {"type": "character", "raw_name": "Bob", "translated_name": "Bob"},
    ]
    _run_test_refinement(initial, progress_file, parsed_entries, send_fn)
    assert calls == ["sent"]

    # Simulate the normal save/reload path: reordered rows, changed translated
    # fields, and persistence-only internal metadata.  This used to invalidate
    # the completed full-dictionary hash and resend the entire category.
    persisted_shape = [
        {
            "translated_name": "Robert",
            "raw_name": "Bob",
            "type": "character",
            "description": "Manually edited",
        },
        {
            "type": "character",
            "raw_name": "Alice",
            "translated_name": "Alicia",
            "_gender_tracker_source": "save-only metadata",
        },
    ]
    skipped = _run_test_refinement(persisted_shape, progress_file, parsed_entries, send_fn)

    assert calls == ["sent"]
    assert skipped == persisted_shape

    # A genuinely new raw/source entry changes the identity set and reopens
    # only this category for refinement.
    parsed_entries.append(
        {"type": "character", "raw_name": "Carol", "translated_name": "Carol"}
    )
    _run_test_refinement(
        persisted_shape
        + [{"type": "character", "raw_name": "Carol", "translated_name": "Carol"}],
        progress_file,
        parsed_entries,
        send_fn,
    )

    assert calls == ["sent", "sent"]
    completed = load_refinement_progress(str(progress_file))["type::character"]
    assert completed["status"] == "completed"
    assert completed["identity_hash_version"] == "raw-name-v1"
    assert completed["input_identity_hash"]
    assert completed["output_identity_hash"]


def test_legacy_completed_refinement_is_migrated_without_resending(tmp_path, monkeypatch):
    _enable_refinement(monkeypatch)
    progress_file = tmp_path / "glossary_progress.json"
    progress_file.write_text(
        json.dumps(
            {
                "refinement": {
                    "type::character": {
                        "entry_type": "character",
                        "status": "completed",
                        "input_hash": "old-full-dictionary-input-hash",
                        "output_hash": "old-full-dictionary-output-hash",
                        "entry_count_after": 2,
                        "output_file": "glossary.csv",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    entries = [
        {"type": "character", "raw_name": "Alice", "translated_name": "Alicia"},
        {"type": "character", "raw_name": "Bob", "translated_name": "Robert"},
    ]

    def unexpected_send(*_args, **_kwargs):
        raise AssertionError("completed legacy refinement was resent")

    result = _run_test_refinement(entries, progress_file, entries, unexpected_send)

    assert result == entries
    migrated = load_refinement_progress(str(progress_file))["type::character"]
    assert migrated["status"] == "completed"
    assert migrated["identity_hash_version"] == "raw-name-v1"
    assert migrated["input_identity_hash"] == migrated["output_identity_hash"]


def test_glossary_extracts_only_dialogue_from_nested_subtitle_zip(tmp_path):
    archive = tmp_path / "season.zip"
    with zipfile.ZipFile(archive, "w") as subtitle_zip:
        subtitle_zip.writestr(
            "Season 01/episode.srt",
            "1\n"
            "00:00:01,000 --> 00:00:03,000\n"
            "<i>Alice</i> visits <b>Wonderland</b>\n",
        )
        subtitle_zip.writestr(
            "Season 01/episode.ass",
            "[Script Info]\n"
            "Title: Do not include\n"
            "[Events]\n"
            "Format: Start, End, Text\n"
            r"Dialogue: 0:00:01.00,0:00:03.00,{\an8}Bob\NArcadia"
            "\n",
        )
        subtitle_zip.writestr(
            "Season 01/theme.lrc",
            "[ar:Do not include]\n"
            "[00:01.00]<00:01.10>Carol in Dreamland\n",
        )
        subtitle_zip.writestr("Season 01/readme.txt", "Ignore this file")

    assert is_subtitle_glossary_source(str(archive)) is True
    chapters = extract_chapters_from_subtitle(
        str(archive),
        return_metadata=True,
    )

    assert [filename for _text, filename in chapters] == [
        "Season 01/episode.srt",
        "Season 01/episode.ass",
        "Season 01/theme.lrc",
    ]
    assert chapters[0][0] == "Alice visits Wonderland"
    assert chapters[1][0] == "Bob\nArcadia"
    assert chapters[2][0] == "Carol in Dreamland"
    combined = "\n".join(text for text, _filename in chapters)
    assert "-->" not in combined
    assert "Dialogue:" not in combined
    assert "[ar:" not in combined
    assert "<i>" not in combined
    assert r"{\an8}" not in combined
    assert "<00:01.10>" not in combined


def test_glossary_accepts_direct_srt_ass_and_lrc_sources(tmp_path):
    cases = {
        "episode.srt": (
            "1\n00:00:01,000 --> 00:00:03,000\n<i>Alice</i>\n",
            "Alice",
        ),
        "episode.ass": (
            "[Events]\n"
            "Format: Start, End, Text\n"
            r"Dialogue: 0:00:01.00,0:00:03.00,{\an8}Bob"
            "\n",
            "Bob",
        ),
        "theme.lrc": (
            "[ar:Ignore]\n[00:01.00]<00:01.10>Carol\n",
            "Carol",
        ),
    }

    for filename, (source_text, expected_text) in cases.items():
        source = tmp_path / filename
        source.write_text(source_text, encoding="utf-8")

        assert is_subtitle_glossary_source(str(source)) is True
        assert extract_chapters_from_subtitle(
            str(source),
            return_metadata=True,
        ) == [(expected_text, filename)]


def test_epub_shaped_zip_is_not_a_subtitle_glossary_source(tmp_path):
    archive = tmp_path / "book.zip"
    with zipfile.ZipFile(archive, "w") as epub_zip:
        epub_zip.writestr("mimetype", "application/epub+zip")
        epub_zip.writestr("META-INF/container.xml", "<container/>")
        epub_zip.writestr("OEBPS/content.opf", "<package/>")
        epub_zip.writestr("OEBPS/media/captions.srt", "incidental")

    assert is_subtitle_glossary_source(str(archive)) is False


def test_parallel_epub_prompt_prepends_pair_rules_to_canonical_prompt_verbatim():
    assert default_parallel_epub_system_prompt() == (
        f"{PARALLEL_EPUB_SYSTEM_INSTRUCTIONS}\n\n{DEFAULT_GLOSSARY_PROMPT}"
    )


def test_parallel_epub_prompt_requires_verified_translated_renderings():
    prompt = default_parallel_epub_system_prompt()

    assert "only translate or transliterate it yourself" not in prompt
    assert "skip the entry entirely" in prompt
    assert "never invent one yourself" in prompt
    assert "never translate, transliterate, normalize, improve" not in prompt


def test_parallel_epub_auto_map_prefers_name_then_number_then_reading_order():
    raw = [
        ("raw prologue", "prologue.xhtml"),
        ("raw two", "raw_chapter_02.xhtml"),
        ("raw ending", "shared-ending-03.xhtml"),
    ]
    translated = [
        ("translated opening", "opening.xhtml"),
        ("translated ending", "shared-ending-03.xhtml"),
        ("translated two", "translated_02.xhtml"),
    ]

    mapping = auto_map_epub_chapters(raw, translated)

    assert [item["translated_index"] for item in mapping] == [None, 2, 1]
    assert [item["strategy"] for item in mapping] == [
        "Unmatched",
        "Chapter number",
        "Exact filename",
    ]


def test_parallel_epub_auto_offset_leaves_matching_nonpositive_files_unmapped():
    raw = [
        ("raw opening", "opening.xhtml"),
        ("raw information", "raw_0000.xhtml"),
        ("raw chapter", "raw_0001.xhtml"),
    ]
    translated = [
        ("translated opening", "opening.xhtml"),
        ("translated information", "raw_0000.xhtml"),
        ("translated chapter", "translated_0001.xhtml"),
    ]

    mapping = auto_map_epub_chapters(raw, translated)
    assert [item["translated_index"] for item in mapping] == [None, None, 2]
    assert [item["strategy"] for item in mapping] == [
        "Unmatched",
        "Unmatched",
        "Chapter number",
    ]

    mapping_without_offset = auto_map_epub_chapters(
        raw,
        translated,
        enable_auto_offset=False,
    )
    assert [item["translated_index"] for item in mapping_without_offset] == [0, 1, 2]


def test_parallel_epub_auto_map_offsets_zero_only_front_matter_from_chapters():
    raw = [
        ("raw one", "raw_0002.xhtml"),
        ("raw two", "raw_0003.xhtml"),
        ("raw three", "raw_0004.xhtml"),
    ]
    translated = [
        ("information", "0000_Information.xhtml"),
        ("translated one", "translated_0001.xhtml"),
        ("translated two", "translated_0002.xhtml"),
        ("translated three", "translated_0003.xhtml"),
    ]

    mapping = auto_map_epub_chapters(raw, translated)

    assert [item["translated_index"] for item in mapping] == [1, 2, 3]
    assert [item["auto_offset"] for item in mapping] == [-1, -1, -1]
    assert [item["strategy"] for item in mapping] == [
        "Auto offset -1",
        "Auto offset -1",
        "Auto offset -1",
    ]

    mapping_without_offset = auto_map_epub_chapters(
        raw,
        translated,
        enable_auto_offset=False,
    )
    assert [item["translated_index"] for item in mapping_without_offset] == [
        2,
        3,
        0,
    ]
    assert all(item["auto_offset"] == 0 for item in mapping_without_offset)


def test_parallel_epub_auto_map_does_not_mix_unnumbered_with_numbered_files():
    raw = [
        ("raw front matter", "opening.xhtml"),
        ("raw one", "raw_0001.xhtml"),
        ("raw two", "raw_0002.xhtml"),
    ]
    translated = [
        ("translated one", "0001_Chapter_1.xhtml"),
        ("translated two", "0002_Chapter_2.xhtml"),
    ]

    mapping = auto_map_epub_chapters(raw, translated)

    assert [item["translated_index"] for item in mapping] == [None, 0, 1]
    assert [item["auto_offset"] for item in mapping] == [0, 1, 1]
    assert mapping[1]["strategy"] == "Auto offset +1"


def test_parallel_epub_auto_map_does_not_offset_positive_numbered_extras():
    raw = [
        ("raw one", "raw_0001.xhtml"),
        ("raw two", "raw_0002.xhtml"),
    ]
    translated = [
        ("numbered bonus", "translated_0009.xhtml"),
        ("translated one", "translated_0001.xhtml"),
        ("translated two", "translated_0002.xhtml"),
    ]

    mapping = auto_map_epub_chapters(raw, translated)

    assert [item["translated_index"] for item in mapping] == [1, 2]
    assert [item["auto_offset"] for item in mapping] == [0, 0]
    assert [item["strategy"] for item in mapping] == [
        "Chapter number",
        "Chapter number",
    ]


def test_parallel_epub_wrapper_preserves_unrelated_braces():
    result = apply_parallel_epub_wrapper(
        "{raw_filename}\n{raw_text}\n{translated_filename}\n"
        "{translated_text}\nKeep {custom_field}",
        raw_text="原文",
        translated_text="Translation",
        raw_filename="raw.xhtml",
        translated_filename="translated.xhtml",
    )

    assert result == (
        "raw.xhtml\n原文\ntranslated.xhtml\nTranslation\nKeep {custom_field}"
    )


def test_parallel_epub_working_filename_preserves_raw_epub_name():
    assert parallel_epub_working_filename(
        "C:/Books/source-novel.epub"
    ) == "source-novel.epub"
    assert parallel_epub_working_filename("raw novel") == "raw novel.epub"


def test_parallel_epub_persistent_selection_excludes_extracted_text(tmp_path):
    result = {
        "raw_path": str(tmp_path / "raw.epub"),
        "translated_path": str(tmp_path / "translated.epub"),
        "wrapper_prompt": "{raw_text}\n{translated_text}",
        "system_prompt": "Cross-check both editions.",
        "profile_name": "Pair profile",
        "pairs": [
            {
                "raw_index": 4,
                "translated_index": 7,
                "raw_filename": "Text/chapter0005.xhtml",
                "raw_text": "large raw chapter that must not enter config",
                "translated_filename": "Text/0005_Chapter.xhtml",
                "translated_text": "large translated chapter that must not enter config",
            }
        ],
    }

    saved = compact_parallel_epub_selection(result)

    assert saved["raw_path"] == os.path.abspath(result["raw_path"])
    assert saved["translated_path"] == os.path.abspath(result["translated_path"])
    assert saved["mapping"] == [
        {
            "raw_index": 4,
            "translated_index": 7,
            "raw_filename": "Text/chapter0005.xhtml",
            "translated_filename": "Text/0005_Chapter.xhtml",
        }
    ]
    serialized = json.dumps(saved)
    assert "large raw chapter" not in serialized
    assert "large translated chapter" not in serialized


def test_parallel_epub_saved_mapping_restores_by_filename_after_reordering():
    raw_chapters = [
        ("raw chapter two", "Text/chapter0002.xhtml"),
        ("raw chapter one", "Text/chapter0001.xhtml"),
    ]
    translated_chapters = [
        ("translated chapter one", "Text/0001_Chapter.xhtml"),
        ("translated chapter two", "Text/0002_Chapter.xhtml"),
    ]
    stored_mapping = [
        {
            "raw_index": 0,
            "translated_index": 0,
            "raw_filename": "Text/chapter0001.xhtml",
            "translated_filename": "Text/0001_Chapter.xhtml",
        },
        {
            "raw_index": 1,
            "translated_index": 1,
            "raw_filename": "Text/chapter0002.xhtml",
            "translated_filename": "Text/0002_Chapter.xhtml",
        },
        {
            "raw_index": 2,
            "translated_index": 2,
            "raw_filename": "Text/deleted.xhtml",
            "translated_filename": "Text/deleted.xhtml",
        },
    ]

    restored, skipped = restore_parallel_epub_pairs(
        raw_chapters, translated_chapters, stored_mapping
    )

    assert skipped == 1
    assert [pair["raw_filename"] for pair in restored] == [
        "Text/chapter0001.xhtml",
        "Text/chapter0002.xhtml",
    ]
    assert [pair["raw_text"] for pair in restored] == [
        "raw chapter one",
        "raw chapter two",
    ]
    assert [pair["translated_text"] for pair in restored] == [
        "translated chapter one",
        "translated chapter two",
    ]


def test_parallel_epub_round_trips_through_shared_extractor(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSLATE_SPECIAL_FILES", "1")
    output = tmp_path / "paired.epub"
    write_parallel_epub(
        str(output),
        [
            {
                "raw_filename": "chapter-01.xhtml",
                "raw_text": "原文 Alice",
                "translated_filename": "chapter-01-en.xhtml",
                "translated_text": "Translated Alice",
            },
            {
                "raw_filename": "chapter-02.xhtml",
                "raw_text": "原文 Bob",
                "translated_filename": "chapter-02-en.xhtml",
                "translated_text": "Translated Bob",
            },
        ],
        DEFAULT_PARALLEL_EPUB_WRAPPER_PROMPT,
    )

    chapters = extract_chapters_from_epub(str(output), return_metadata=True)

    assert [filename for _text, filename in chapters] == [
        "pair_0001.xhtml",
        "pair_0002.xhtml",
    ]
    assert "[RAW EPUB START — chapter-01.xhtml]" in chapters[0][0]
    assert "原文 Alice" in chapters[0][0]
    assert "[TRANSLATED EPUB START — chapter-01-en.xhtml]" in chapters[0][0]
    assert "Translated Alice" in chapters[0][0]


def test_parallel_epub_requires_both_text_placeholders(tmp_path):
    pair = {
        "raw_filename": "raw.xhtml",
        "raw_text": "raw",
        "translated_filename": "translated.xhtml",
        "translated_text": "translated",
    }

    try:
        write_parallel_epub(
            str(tmp_path / "invalid.epub"),
            [pair],
            "Only {raw_text}",
        )
    except ValueError as exc:
        assert "{raw_text} and {translated_text}" in str(exc)
    else:
        raise AssertionError("missing translated placeholder should be rejected")


def test_parallel_epub_dialog_loads_in_background_and_shows_drop_feedback(tmp_path):
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "raw.epub"
    source.write_bytes(b"")
    release_loader = threading.Event()

    def delayed_loader(_path):
        release_loader.wait(timeout=2)
        return [
            ("raw text", "chapter-01.xhtml"),
            ("title text", "title.xhtml"),
        ]

    dialog = ParallelEpubPairDialog(
        config={},
        chapter_loader=delayed_loader,
        special_file_predicate=lambda filename: "title" in filename.casefold(),
    )
    assert dialog.content_splitter.orientation() == Qt.Horizontal
    assert dialog.content_splitter.count() == 2
    assert dialog.mapping_table.parentWidget() is dialog.mapping_panel
    idle_style = dialog.raw_drop.styleSheet()
    dialog.raw_drop.set_hovered(True)
    assert "Release to load this EPUB" == dialog.raw_drop.hint_label.text()
    assert "solid" in dialog.raw_drop.styleSheet()
    assert dialog.raw_drop.styleSheet() != idle_style
    dialog.raw_drop.set_hovered(False)

    started = time.perf_counter()
    dialog._load_epub("raw", str(source))
    elapsed = time.perf_counter() - started

    assert elapsed < 0.25
    assert dialog._active_load is not None
    assert "background" in dialog.raw_drop.count_label.text()

    release_loader.set()
    deadline = time.monotonic() + 3
    while dialog._active_load is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)

    assert dialog._active_load is None
    assert dialog.raw_path == str(source)
    assert dialog.raw_chapters == [
        {"text": "raw text", "filename": "chapter-01.xhtml"}
    ]
    assert "eligible HTML" in dialog.raw_drop.count_label.text()
    dialog.deleteLater()
    app.processEvents()


def test_parallel_epub_dialog_restores_saved_mapping_after_async_load(
    tmp_path, monkeypatch
):
    app = QApplication.instance() or QApplication([])
    load_errors = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, text, *_args, **_kwargs: load_errors.append(text),
    )
    raw_path = tmp_path / "raw.epub"
    translated_path = tmp_path / "translated.epub"
    raw_path.write_bytes(b"raw")
    translated_path.write_bytes(b"translated")

    raw_chapters = [
        (f"raw {index}", f"Text/chapter-{index:02d}.xhtml")
        for index in range(1, 4)
    ]
    translated_chapters = [
        (f"translated {index}", f"Text/translated-{index:02d}.xhtml")
        for index in range(1, 4)
    ]

    def loader(path):
        return raw_chapters if Path(path) == raw_path else translated_chapters

    dialog = ParallelEpubPairDialog(config={}, chapter_loader=loader)
    restored = dialog.restore_persisted_selection(
        {
            "version": 1,
            "raw_path": str(raw_path),
            "translated_path": str(translated_path),
            "mapping": [
                {
                    "raw_index": 0,
                    "translated_index": 2,
                    "raw_filename": "Text/chapter-01.xhtml",
                    "translated_filename": "Text/translated-03.xhtml",
                },
                {
                    "raw_index": 2,
                    "translated_index": 0,
                    "raw_filename": "Text/chapter-03.xhtml",
                    "translated_filename": "Text/translated-01.xhtml",
                },
            ],
            "wrapper_prompt": "Before\n{raw_text}\n{translated_text}",
            "system_prompt": "Saved exact prompt",
            "profile_name": "Parallel EPUB Glossary",
        }
    )

    deadline = time.monotonic() + 3
    while (
        dialog._active_load is not None
        or dialog._pending_loads
        or dialog._mapping_building
    ) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)

    assert restored
    assert not load_errors
    assert dialog._selected_mapping() == [
        {"raw_index": 0, "translated_index": 2},
        {"raw_index": 2, "translated_index": 0},
    ]
    assert dialog.mapping_table.item(1, 1).data(Qt.UserRole) == -1
    assert dialog.mapping_table.item(1, 2).text() == "Saved — Unmapped"
    assert dialog.wrapper_edit.toPlainText() == (
        "Before\n{raw_text}\n{translated_text}"
    )
    assert dialog.system_prompt_edit.toPlainText() == "Saved exact prompt"
    dialog.deleteLater()
    app.processEvents()


def test_parallel_epub_dialog_clear_selection_empties_loaded_pair():
    app = QApplication.instance() or QApplication([])
    dialog = ParallelEpubPairDialog(config={})
    raw_chapters = [
        {"text": "raw one", "filename": "chapter-01.xhtml"},
        {"text": "raw two", "filename": "chapter-02.xhtml"},
    ]
    translated_chapters = [
        {"text": "translated one", "filename": "translated-01.xhtml"},
        {"text": "translated two", "filename": "translated-02.xhtml"},
    ]
    dialog._apply_loaded_epub("raw", "C:/Books/raw.epub", raw_chapters)
    dialog._apply_loaded_epub(
        "translated", "C:/Books/translated.epub", translated_chapters
    )
    deadline = time.monotonic() + 3
    while dialog._mapping_building and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert dialog.mapping_table.rowCount() == 2

    dialog.clear_selection()

    assert dialog.raw_path == ""
    assert dialog.translated_path == ""
    assert dialog.raw_chapters == []
    assert dialog.translated_chapters == []
    assert dialog.mapping_table.rowCount() == 0
    assert dialog.raw_drop.path_label.text() == "Drop an .epub here"
    assert dialog.translated_drop.path_label.text() == "Drop an .epub here"
    assert dialog.mapping_status.text() == "Load both EPUBs to create a map."
    assert not dialog.use_pair_button.isEnabled()
    dialog.deleteLater()
    app.processEvents()


def test_parallel_epub_mapping_combos_lock_wheel_use_icon_and_support_offsets():
    app = QApplication.instance() or QApplication([])
    dialog = ParallelEpubPairDialog(config={})
    assert dialog.width() == 1520
    assert dialog.minimumWidth() == 1120
    assert "font-size: 8pt" in dialog.mapping_status.styleSheet()
    assert dialog.auto_offset_checkbox.isChecked()
    assert hasattr(dialog.auto_offset_checkbox, "_checkmark_label")
    assert "QCheckBox::indicator:checked" in dialog.auto_offset_checkbox.styleSheet()
    assert dialog.profile_combo.objectName() == "parallelPromptProfileCombo"
    assert "QComboBox#parallelPromptProfileCombo::down-arrow" in dialog.styleSheet()
    raw_chapters = [
        {"text": f"raw {index}", "filename": f"chapter-{index:02d}.xhtml"}
        for index in range(1, 4)
    ]
    translated_chapters = [
        {"text": f"translated {index}", "filename": f"chapter-{index:02d}.xhtml"}
        for index in range(1, 4)
    ]
    dialog._apply_loaded_epub("raw", "raw.epub", raw_chapters)
    dialog._apply_loaded_epub(
        "translated", "translated.epub", translated_chapters
    )
    assert dialog._mapping_building
    assert dialog.use_pair_button.text() == "Mapping..."
    deadline = time.monotonic() + 3
    while dialog._mapping_building and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)

    assert not dialog._mapping_building
    assert dialog.use_pair_button.text() == "Use Mapped Pair"
    assert dialog.offset_down_button.text() == "\N{MINUS SIGN} Offset"
    assert dialog.offset_up_button.text() == "+ Offset"
    assert dialog._selected_mapping() == [
        {"raw_index": 0, "translated_index": 0},
        {"raw_index": 1, "translated_index": 1},
        {"raw_index": 2, "translated_index": 2},
    ]

    dialog.show()
    app.processEvents()
    translated_item = dialog.mapping_table.item(0, 1)
    translated_rect = dialog.mapping_table.visualItemRect(translated_item)
    QTest.mouseClick(
        dialog.mapping_table.viewport(),
        Qt.LeftButton,
        pos=translated_rect.center(),
    )
    app.processEvents()
    combo = next(
        editor
        for editor in dialog.mapping_table.findChildren(QComboBox)
        if editor.objectName() == "parallelMappingCombo"
    )
    assert combo.currentIndex() == 1
    assert combo.view().isVisible()

    class WheelEvent:
        ignored = False

        def ignore(self):
            self.ignored = True

    wheel_event = WheelEvent()
    combo.wheelEvent(wheel_event)
    assert wheel_event.ignored
    assert combo.objectName() == "parallelMappingCombo"
    assert "Halgakos.ico" in dialog.styleSheet()
    icon_rect = dialog._mapping_delegate._arrow_rect(QRect(10, 40, 200, 38), 184)
    assert icon_rect.size().width() == 16
    assert icon_rect.center().y() == QRect(10, 40, 200, 38).center().y()
    combo.hidePopup()
    dialog._mapping_delegate._commit_and_close(combo)
    app.processEvents()

    dialog._apply_mapping_offset(1)
    assert dialog._mapping_offset == 1
    assert dialog._selected_mapping() == [
        {"raw_index": 1, "translated_index": 0},
        {"raw_index": 2, "translated_index": 1},
    ]
    assert "offset +1" in dialog.mapping_status.text()

    dialog._apply_mapping_offset(-1)
    assert dialog._mapping_offset == 0
    assert dialog._selected_mapping() == [
        {"raw_index": 0, "translated_index": 0},
        {"raw_index": 1, "translated_index": 1},
        {"raw_index": 2, "translated_index": 2},
    ]
    assert "offset" not in dialog.mapping_status.text()

    dialog._set_rows_unmapped([0, 2])
    assert dialog._selected_mapping() == [
        {"raw_index": 1, "translated_index": 1},
    ]
    assert dialog.mapping_table.item(0, 2).text() == "Manual — Unmapped"
    assert dialog.mapping_table.item(2, 2).text() == "Manual — Unmapped"
    assert "2 raw unmatched" in dialog.mapping_status.text()
    unmatched_raw, unused_translated = dialog._unpaired_file_counts(
        dialog._selected_mapping()
    )
    assert (unmatched_raw, unused_translated) == (2, 2)
    warning = dialog._unpaired_warning_text(dialog._selected_mapping())
    assert "4 HTML file(s) are not part of a mapped pair" in warning
    assert "Unmatched raw HTML files: 2" in warning
    assert "Unused translated HTML files: 2" in warning
    assert "overflow beyond the available raw rows" in warning
    warning_box = dialog._create_unpaired_warning_box(dialog._selected_mapping())
    button_box = warning_box.findChild(QDialogButtonBox)
    assert button_box is not None
    assert button_box.centerButtons()
    assert button_box.layout().spacing() == 20
    assert warning_box.defaultButton() is warning_box.button(QMessageBox.No)
    for standard_button in (QMessageBox.Yes, QMessageBox.No):
        button = warning_box.button(standard_button)
        assert button.minimumWidth() == 120
        assert button.minimumHeight() == 46
    warning_box.deleteLater()
    dialog.deleteLater()
    app.processEvents()


def test_parallel_epub_default_profile_reset_requires_confirmation(monkeypatch):
    app = QApplication.instance() or QApplication([])
    dialog = ParallelEpubPairDialog(config={})
    dialog.system_prompt_edit.setPlainText("custom unsaved prompt")

    class Confirmation:
        def __init__(self, result):
            self.result = result

        def exec(self):
            return self.result

    monkeypatch.setattr(
        dialog,
        "_create_centered_question_box",
        lambda *_args, **_kwargs: Confirmation(QMessageBox.No),
    )
    dialog._delete_or_reset_profile()
    assert dialog.system_prompt_edit.toPlainText() == "custom unsaved prompt"

    monkeypatch.setattr(
        dialog,
        "_create_centered_question_box",
        lambda *_args, **_kwargs: Confirmation(QMessageBox.Yes),
    )
    dialog._delete_or_reset_profile()
    assert dialog.system_prompt_edit.toPlainText() == default_parallel_epub_system_prompt()
    dialog.deleteLater()
    app.processEvents()
