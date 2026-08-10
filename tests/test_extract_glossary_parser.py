import json
import zipfile

from extract_glossary_from_epub import (
    extract_chapters_from_subtitle,
    is_subtitle_glossary_source,
    parse_api_response,
    skip_duplicate_entries,
)
from glossary_refinement import (
    load_refinement_progress,
    refine_glossary_entries,
    refinement_chunking_mode,
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

    monkeypatch.setenv("GLOSSARY_REFINEMENT_CHUNKING_MODE", "separate")
    assert refinement_chunking_mode() == "separate"


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
