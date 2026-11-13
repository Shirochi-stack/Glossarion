"""Relations Tracker at translation time: build_system_prompt, merged cut-off, log order, sidecars."""
import copy
import inspect
import json
import os
import sys

import pytest

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import relations_tracking as rt  # noqa: E402
from glossary_compressor import merge_glossary_texts  # noqa: E402

HEADER = "Glossary Columns: raw_name, translated_name, gender, description"
TOKEN_GLOSSARY = (
    HEADER + "\n"
    "\n"
    "=== CHARACTERS ===\n"
    "* 서윤 = Seo-yun [Female]: Heir of the Baek trading house\n"
    "* 도현 = Do-hyeon [Male]: Royal guard captain\n"
    "* 하린 = Ha-rin [Female]: Apprentice healer\n"
    "\n"
    "=== TERMS ===\n"
    "* 청월검 = Azure Moon Sword: Heirloom blade\n"
)
LEGACY_GLOSSARY = (
    "type,raw_name,translated_name,gender,description\n"
    "character,서윤,Seo-yun,Female,Heir\n"
    "character,하린,Ha-rin,Female,Healer\n"
)
APPEND_PROMPT = "Use this glossary:"
RELATION_ENVS = (rt.ENV_ENABLE, rt.ENV_HISTORY_MODE, rt.ENV_FIELD)


@pytest.fixture(scope="module")
def tk(tmp_path_factory):
    """TransateKRtoEN imported with app/library/output folders under a temp dir."""
    root = tmp_path_factory.mktemp("relations_translation_app")
    patch = pytest.MonkeyPatch()
    for name in ("GLOSSARION_APP_DIR", "GLOSSARION_LIBRARY_DIR", "OUTPUT_DIRECTORY"):
        path = root / name.lower()
        path.mkdir()
        patch.setenv(name, str(path))
    patch.setenv("GLOSSARION_HTTP_LOG", "0")
    try:
        import TransateKRtoEN
    finally:
        patch.undo()
    return TransateKRtoEN


@pytest.fixture
def env(monkeypatch, tmp_path):
    for name in RELATION_ENVS + ("BATCH_TRANSLATION", "TRANSLATION_CANCELLED", "MANUAL_GLOSSARY",
                                 "GLOSSARY_COMPRESSION_SOURCE_EPUB", "VISION_OCR_SOURCE_EPUB"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    monkeypatch.setenv("APPEND_GLOSSARY", "1")
    monkeypatch.setenv("APPEND_GLOSSARY_PROMPT", APPEND_PROMPT)
    monkeypatch.setenv("COMPRESS_GLOSSARY_PROMPT", "0")
    monkeypatch.setenv("ADD_ADDITIONAL_GLOSSARY", "0")
    monkeypatch.setenv("ENABLE_UNIFIED_GLOSSARY", "0")
    monkeypatch.setenv("OUTPUT_LANGUAGE", "English")
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(tmp_path / "out_root"))
    # defer_batch_log prints through the module's logger-backed print; send it to
    # stdout so capsys sees deferred lines in the same order as the other prints.
    import builtins
    import unified_api_client
    monkeypatch.setattr(unified_api_client, "print", builtins.print)
    return monkeypatch


def _entry(raw, translated, cell):
    return {"type": "character", "raw_name": raw, "translated_name": translated,
            "gender": "female", "description": "x", "relations": cell}


def _write_book(tmp_path, glossary_text=TOKEN_GLOSSARY, name="Book_glossary.csv"):
    """Glossary + relations tracker: 서윤 -> 하린 is Best Friend (1), Rival (20), Enemy (40)."""
    glossary = tmp_path / name
    glossary.write_text(glossary_text, encoding="utf-8")
    tracker = rt.empty_tracker()
    # Written the way the model writes it: on X's row, "Y: what X is to Y".
    story = [
        (1, [_entry("도현", "Do-hyeon", "서윤: Older Brother"),
             _entry("하린", "Ha-rin", "서윤: Best Friend"),
             _entry("서윤", "Seo-yun", "하린: Best Friend")]),
        (20, [_entry("하린", "Ha-rin", "서윤: Rival"), _entry("서윤", "Seo-yun", "")]),
        (40, [_entry("하린", "Ha-rin", "서윤: Enemy"), _entry("서윤", "Seo-yun", "")]),
    ]
    for chapter, rows in story:
        rt.record_relations(tracker, copy.deepcopy(rows), field="relations", chapter_index=chapter - 1,
                            chapter_num=chapter, chapter_file=f"ch{chapter:03d}.xhtml")
    tracker_path = rt.tracker_path_for_glossary(str(glossary))
    with open(tracker_path, "w", encoding="utf-8") as handle:
        json.dump(tracker, handle, ensure_ascii=False)
    return str(glossary)


def _build(tk, glossary, ref, source_text=None, settings=None):
    return tk.build_system_prompt("SYSTEM", glossary, source_text=source_text,
                                  chapter_ref=ref, settings=settings)


def _glossary_block(prompt):
    """The glossary block after the append prompt (and after the relations legend, if any)."""
    block = prompt.split(APPEND_PROMPT + "\n", 1)[1]
    legend = rt.legend_line() + "\n"
    return block[len(legend):] if block.startswith(legend) else block


# --------------------------------------------------------------------------- toggle off

@pytest.mark.parametrize("compress", ["0", "1"])
def test_toggle_off_never_touches_relations(tk, env, tmp_path, capsys, compress):
    env.setenv("COMPRESS_GLOSSARY_PROMPT", compress)
    glossary = _write_book(tmp_path)
    source = "서윤 하린 청월검"
    ref = {"chapter_num": 45, "chapter_file": "ch045.xhtml"}
    _build(tk, glossary, ref, source)  # "Loading glossary from" is printed once per path
    capsys.readouterr()
    prompt_off = _build(tk, glossary, ref, source)
    log_off = capsys.readouterr().out
    # Without the module at all the result is the same: nothing is consulted when off.
    env.setitem(sys.modules, "relations_tracking", None)
    prompt_without_module = _build(tk, glossary, ref, source)
    log_without_module = capsys.readouterr().out
    assert prompt_off == prompt_without_module
    assert log_off == log_without_module
    assert "relations" not in prompt_off and "🔗" not in log_off
    assert "Relations" not in log_off


def test_snapshot_setting_wins_over_env(tk, env, tmp_path, capsys):
    glossary = _write_book(tmp_path)
    env.setenv(rt.ENV_ENABLE, "1")
    ref = {"chapter_num": 45, "chapter_file": "ch045.xhtml"}
    off = _build(tk, glossary, ref, settings={rt.ENV_ENABLE: "0"})
    assert "relations" not in off
    env.setenv(rt.ENV_ENABLE, "0")
    on = _build(tk, glossary, ref, settings={rt.ENV_ENABLE: "1", rt.ENV_HISTORY_MODE: "current"})
    assert "(relations: Do-hyeon: Older Brother; Ha-rin: Enemy)" in on
    capsys.readouterr()


# --------------------------------------------------------------------------- toggle on

def test_uncompressed_token_glossary_gets_column(tk, env, tmp_path, capsys):
    env.setenv(rt.ENV_ENABLE, "1")
    glossary = _write_book(tmp_path)
    prompt = _build(tk, glossary, {"chapter_num": 25, "chapter_file": "ch025.xhtml"})
    block = _glossary_block(prompt)
    lines = block.split("\n")
    assert lines[0] == HEADER + ", relations"
    assert ("* 서윤 = Seo-yun [Female]: Heir of the Baek trading house "
            "(relations: Do-hyeon: Older Brother; Ha-rin: Rival [formerly Best Friend])") in lines
    assert "* 하린 = Ha-rin [Female]: Apprentice healer (relations: Seo-yun: Best Friend)" in lines
    assert "* 청월검 = Azure Moon Sword: Heirloom blade" in lines
    # The legend sits on its own line between the append prompt and the glossary block.
    assert (APPEND_PROMPT + "\n" + rt.legend_line() + "\n" + HEADER + ", relations\n") in prompt
    assert prompt.count(rt.legend_line()) == 1
    out = capsys.readouterr().out
    assert "🔗 Relations: 3 shown for 2 characters (up to Ch. 25) | changed here:" not in out
    assert "🔗 Relations: 3 shown for 2 characters (up to Ch. 25)" in out


def test_compressed_glossary_only_shows_targets_in_the_request(tk, env, tmp_path, capsys):
    env.setenv(rt.ENV_ENABLE, "1")
    env.setenv("COMPRESS_GLOSSARY_PROMPT", "1")
    glossary = _write_book(tmp_path)
    # 도현 is not in the chapter, so compression drops him and his relation is not shown.
    prompt = _build(tk, glossary, {"chapter_num": 45, "chapter_file": "ch045.xhtml"},
                    source_text="서윤은 하린을 보았다.")
    block = _glossary_block(prompt)
    assert "Do-hyeon" not in block
    assert "* 서윤 = Seo-yun [Female]: Heir of the Baek trading house (relations: Ha-rin: Enemy [formerly Rival])" in block
    assert block.split("\n")[0] == HEADER + ", relations"
    out = capsys.readouterr().out
    assert "🔗 Relations: 2 shown for 2 characters (up to Ch. 45)" in out


def test_legacy_csv_glossary_gets_a_column(tk, env, tmp_path, capsys):
    env.setenv(rt.ENV_ENABLE, "1")
    glossary = _write_book(tmp_path, LEGACY_GLOSSARY, "Legacy_glossary.csv")
    block = _glossary_block(_build(tk, glossary, {"chapter_num": 45}))
    lines = block.split("\n")
    assert lines[0] == "type,raw_name,translated_name,gender,description,relations"
    assert lines[1] == "character,서윤,Seo-yun,Female,Heir,Ha-rin: Enemy [formerly Rival]"
    assert lines[2] == "character,하린,Ha-rin,Female,Healer,Seo-yun: Best Friend"
    capsys.readouterr()


def test_merged_request_uses_last_chapter_of_the_group(tk, env, tmp_path, capsys):
    env.setenv(rt.ENV_ENABLE, "1")
    glossary = _write_book(tmp_path)
    parent_only = _build(tk, glossary, {"chapter_num": 38, "chapter_file": "ch038.xhtml"})
    assert "Ha-rin: Rival [formerly Best Friend]" in parent_only
    capsys.readouterr()
    merged = _build(tk, glossary, {"chapter_num": 38, "chapter_num_end": 41, "chapter_file": "ch038.xhtml"})
    assert "Ha-rin: Enemy [formerly Rival]" in merged
    out = capsys.readouterr().out
    assert ("🔗 Relations: 3 shown for 2 characters (up to Ch. 41) | changed here: "
            "Seo-yun → Ha-rin: Enemy [formerly Rival]") in out


def test_metadata_requests_and_missing_chapter_get_nothing(tk, env, tmp_path, capsys):
    env.setenv(rt.ENV_ENABLE, "1")
    glossary = _write_book(tmp_path)
    for ref in ({"use_storage_gender": True}, None):
        prompt = _build(tk, glossary, ref)
        assert "relations" not in prompt
    assert "🔗" not in capsys.readouterr().out


def test_legend_only_when_something_was_injected_and_uses_the_field_name(tk, env, tmp_path, capsys):
    env.setenv(rt.ENV_ENABLE, "1")
    glossary = _write_book(tmp_path)
    before_any = _build(tk, glossary, {"chapter_num": 0.5, "chapter_file": "prologue.xhtml"})
    assert '"Name: role"' not in before_any and "relations" not in before_any
    bonds = _build(tk, glossary, {"chapter_num": 45}, settings={rt.ENV_ENABLE: "1", rt.ENV_FIELD: "bonds"})
    assert (APPEND_PROMPT + "\n" + rt.legend_line("bonds") + "\n" + HEADER + ", bonds\n") in bonds
    # JSON glossary: the legend is outside the JSON, so the block still parses.
    as_json = json.dumps([{"type": "character", "raw_name": "서윤", "translated_name": "Seo-yun"},
                          {"type": "character", "raw_name": "하린", "translated_name": "Ha-rin"}],
                         ensure_ascii=False)
    json_glossary = _write_book(tmp_path, as_json, "Json_glossary.json")
    prompt = _build(tk, json_glossary, {"chapter_num": 45})
    block = _glossary_block(prompt)
    assert prompt.count(rt.legend_line()) == 1
    assert json.loads(block)[0]["relations"] == "Ha-rin: Enemy [formerly Rival]"
    capsys.readouterr()


def test_header_batch_request_leaves_relations_out(tk, env, tmp_path, monkeypatch):
    import metadata_batch_translator as mbt
    seen = []

    def _fake(system_prompt, payload, **kwargs):
        seen.append(kwargs.get("settings"))
        return system_prompt, payload, None

    monkeypatch.setattr(tk, "prepare_glossary_aware_request", _fake)
    mbt._prepare_artifact_glossary_request("S", {"1": "x"}, {"glossary_enable_relations_tracker": True}, None,
                                           chapter_ref={"chapter_num": 1}, relations=False)
    mbt._prepare_artifact_glossary_request("S", {"1": "x"}, {}, None, chapter_ref={"chapter_num": 1}, relations=False)
    mbt._prepare_artifact_glossary_request("S", {"1": "x"}, {"glossary_enable_relations_tracker": True}, None)
    assert seen[0][rt.ENV_ENABLE] == "0" and seen[1] == {rt.ENV_ENABLE: "0"}
    assert seen[2] == {rt.ENV_ENABLE: "1"}
    source = inspect.getsource(mbt)
    assert "chapter_ref=_first_header_chapter_ref(batch_headers),\n                        relations=False," in (
        source.replace("\r\n", "\n"))


def test_extension_shares_the_header_and_gets_no_relations(tk, env, tmp_path, capsys):
    env.setenv(rt.ENV_ENABLE, "1")
    env.setenv("ADD_ADDITIONAL_GLOSSARY", "1")
    glossary = _write_book(tmp_path)
    (tmp_path / "glossary_extension.csv").write_text(
        HEADER + "\n\n=== CHARACTERS ===\n* 무진 = Mu-jin [Male]: Head of the house\n", encoding="utf-8")
    block = _glossary_block(_build(tk, glossary, {"chapter_num": 45}))
    assert block.count("Glossary Columns:") == 1
    assert block.split("\n")[0] == HEADER + ", relations"
    assert block.count("=== CHARACTERS ===") == 1
    assert "* 무진 = Mu-jin [Male]: Head of the house" in block.split("\n")
    capsys.readouterr()


# --------------------------------------------------------------------------- log order

@pytest.mark.parametrize("batch", [None, "1"])
def test_relations_line_comes_right_after_the_glossary_line(tk, env, tmp_path, capsys, batch):
    import unified_api_client
    if batch:
        env.setenv("BATCH_TRANSLATION", batch)
    env.setenv(rt.ENV_ENABLE, "1")
    glossary = _write_book(tmp_path)
    unified_api_client.pop_deferred_batch_logs()
    capsys.readouterr()
    _build(tk, glossary, {"chapter_num": 45, "chapter_file": "ch045.xhtml"})
    if batch:
        lines = unified_api_client.pop_deferred_batch_logs()
        assert not any(l.startswith(("✅ Glossary appended", "🔗")) for l in capsys.readouterr().out.splitlines())
    else:
        lines = capsys.readouterr().out.splitlines()
    glossary_at = next(i for i, l in enumerate(lines) if l.startswith("✅ Glossary appended"))
    assert lines[glossary_at + 1].startswith("🔗 Relations: ")
    assert lines[glossary_at + 2].startswith("🎯 Final system prompt")


# --------------------------------------------------------------------------- helpers and sidecars

def test_merged_chapter_num_end_helper(tk):
    assert tk._merged_chapter_num_end([38, 39, 41, 40]) == 41
    assert tk._merged_chapter_num_end([1.5, "2", None, "x"]) == "2"
    assert tk._merged_chapter_num_end([]) is None
    source = inspect.getsource(tk)
    assert source.count('["chapter_num_end"] = merged_num_end') == 2


def test_merge_glossary_texts_extra_columns():
    book = HEADER + ", relations\n\n=== CHARACTERS ===\n* a = A (relations: B: Friend)\n"
    extension = HEADER + "\n\n=== CHARACTERS ===\n* b = B\n"
    merged = merge_glossary_texts([extension, book], extra_header_columns=("relations",))
    assert merged == (HEADER + ", relations\n\n=== CHARACTERS ===\n* b = B\n* a = A (relations: B: Friend)")
    # Without the argument the old rule holds: different headers are kept apart.
    assert merge_glossary_texts([extension, book]).count("Glossary Columns:") == 2
    other = "Glossary Columns: raw_name, translated_name\n\n=== TERMS ===\n* c = C\n"
    assert merge_glossary_texts([book, other], extra_header_columns=("relations",)).count("Glossary Columns:") == 2


def test_pre_extracted_copy_brings_the_tracker(tk, tmp_path, capsys):
    source = _write_book(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    target = out / "glossary.csv"
    target.write_text(TOKEN_GLOSSARY, encoding="utf-8")
    copied = tk._copy_relations_tracker_with_glossary(source, str(target))
    assert copied == str(out / "relations_tracker.json")
    with open(copied, encoding="utf-8") as handle:
        assert handle.read() == open(rt.tracker_path_for_glossary(source), encoding="utf-8").read()
    # An existing tracker beside the copy is never overwritten; no source tracker = nothing.
    assert tk._copy_relations_tracker_with_glossary(source, str(target)) is None
    assert tk._copy_relations_tracker_with_glossary(str(tmp_path / "None_glossary.csv"),
                                                    str(out / "other.csv")) is None
    assert "Copied relations tracker to output: relations_tracker.json" in capsys.readouterr().out


def test_glossary_discovery_skips_relations_trackers(tk, env, tmp_path):
    out = tmp_path / "book_out"
    out.mkdir()
    tracker = out / "Book_relations_tracker.json"
    tracker.write_text(json.dumps(rt.empty_tracker()), encoding="utf-8")
    (out / "relations_tracker.json").write_text("{}", encoding="utf-8")
    assert tk._is_relations_tracker_file(str(tracker))
    assert not tk._is_relations_tracker_file("Book_glossary.csv")
    settings = {"OUTPUT_DIRECTORY": str(tmp_path / "out_root"), "AUTO_GLOSSARY_MODE": "minimal"}
    assert tk.find_glossary_file(str(out), manual_glossary=str(tracker), settings=settings) is None
    (out / "glossary.csv").write_text(TOKEN_GLOSSARY, encoding="utf-8")
    found = tk.find_glossary_file(str(out), manual_glossary=str(tracker), settings=settings)
    assert os.path.basename(found) == "glossary.csv"


def test_env_passthrough_lists_the_relations_settings(tk):
    source = inspect.getsource(tk)
    assert ("'GLOSSARY_ENABLE_RELATIONS_TRACKER', 'GLOSSARY_RELATIONS_HISTORY_MODE', "
            "'GLOSSARY_RELATIONS_FIELD',") in source
