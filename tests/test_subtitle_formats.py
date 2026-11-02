"""Owner request (2026-11-02): .vtt, .ssa and .sbv translate like .srt, .ass and .lrc, directly and through a ZIP.

Before this a .vtt was a plain-text attachment (its timestamps went to the model and it came back as .txt), .ssa and .sbv
were rejected, and a ZIP of .vtt files extracted nothing while mobile's Plan card said it would translate them.
"""

import ast
import importlib.util
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MOBILE = SRC / "mobile" / "app" / "glossarion_mobile"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

import direct_text_store as dts  # noqa: E402
import subtitle_processor as sp  # noqa: E402
from extract_glossary_from_epub import is_subtitle_glossary_source  # noqa: E402
from input_preparation import InputPreparationMixin  # noqa: E402
from progress_core import ProgressViewMixin  # noqa: E402

# (the source, the lines that must come back byte for byte)
SAMPLES = {
    ".srt": ("1\n00:00:01,000 --> 00:00:02,000\n안녕 친구\n\n2\n00:00:03,000 --> 00:00:04,000\n잘 가\n",
             ["1", "00:00:01,000 --> 00:00:02,000", "00:00:03,000 --> 00:00:04,000"]),
    ".ass": ("[Script Info]\nTitle: 제목\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
             "Effect, Text\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,안녕, 친구\n"
             "Dialogue: 0,0:00:03.00,0:00:04.00,Default,,0,0,0,,잘 가\n",
             ["Title: 제목"]),
    ".ssa": ("[Script Info]\nTitle: 제목\nScriptType: v4.00\n\n[V4 Styles]\nFormat: Name, Fontname\nStyle: Default,Arial\n\n"
             "[Events]\nFormat: Marked, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
             "Comment: Marked=0,0:00:00.00,0:00:01.00,Default,,0,0,0,,메모\n"
             "Dialogue: Marked=0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\i1}안녕, 친구{\\i0}\\n잘 가\n",
             ["Title: 제목", "ScriptType: v4.00", "Style: Default,Arial",
              "Comment: Marked=0,0:00:00.00,0:00:01.00,Default,,0,0,0,,메모"]),
    ".vtt": ("WEBVTT - 제목\nKind: captions\nLanguage: ko\n\nNOTE 메모 입니다\n\nSTYLE\n::cue(b) { color: red }\n\n"
             "REGION\nid:아래\n\ncue-1\n00:01.000 --> 00:02.000 align:start line:0\n<v Bob>안녕 &amp; 친구</v>\n"
             "<i>잘 가</i>\n\n01:00:03.000 --> 01:00:04.000\n잘 가\n",
             ["WEBVTT - 제목", "Kind: captions", "Language: ko", "NOTE 메모 입니다", "::cue(b) { color: red }", "id:아래",
              "cue-1", "00:01.000 --> 00:02.000 align:start line:0", "01:00:03.000 --> 01:00:04.000"]),
    ".sbv": ("0:00:01.000,0:00:02.000\n안녕 친구[br]잘 가\n\n0:00:03.000,0:00:04.000\n잘 가\n",
             ["0:00:01.000,0:00:02.000", "0:00:03.000,0:00:04.000"]),
    ".lrc": ("[ti:제목]\n[00:01.00]안녕 친구\n[00:03.00]잘 가\n", ["[ti:제목]"]),
}
FORMATS = sorted(SAMPLES)
TRANSLATE = {"안녕": "Hello", "친구": "friend", "잘 가": "Bye"}


def _translated(chapter):
    def target(text):
        for source, english in TRANSLATE.items():
            text = text.replace(source, english)
        return text

    return json.dumps([{"id": record["id"], "target": target(record["source"])} for record in json.loads(chapter["body"])],
                      ensure_ascii=False)


def _assert_translated(extension, output):
    lines = output.replace("\r\n", "\n").split("\n")
    for line in SAMPLES[extension][1]:
        assert line in lines, line
    assert "Hello" in output and "Bye" in output


def _round_trip(path, work):
    result = sp.extract_subtitle_to_chapters(str(path), str(work))
    for chapter in json.loads(Path(result["chapters_path"]).read_text(encoding="utf-8")):
        (work / f"response_section_{chapter['num']}.txt").write_text(_translated(chapter), encoding="utf-8")
    return result, sp.convert_subtitle(str(work))


def _zip(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in members.items():
            archive.writestr(name, text)
    return path


def test_the_six_formats_are_the_list():
    assert set(sp.SUBTITLE_EXTENSIONS) == set(SAMPLES)
    assert all(f"*{extension}" in sp.SUBTITLE_FILE_PATTERNS.split() for extension in FORMATS)


@pytest.mark.parametrize("extension", FORMATS)
def test_a_file_round_trips_with_its_structure(tmp_path, extension):
    source = tmp_path / f"episode{extension}"
    source.write_text(SAMPLES[extension][0], encoding="utf-8", newline="")
    _result, converted = _round_trip(source, tmp_path / "out")
    assert converted["output_path"].endswith(extension)
    _assert_translated(extension, Path(converted["output_path"]).read_text(encoding="utf-8"))


@pytest.mark.parametrize("extension", [".vtt", ".sbv", ".ssa"])
def test_crlf_survives(tmp_path, extension):
    source = tmp_path / f"episode{extension}"
    source.write_bytes(SAMPLES[extension][0].replace("\n", "\r\n").encode("utf-8"))
    _result, converted = _round_trip(source, tmp_path / "out")
    data = Path(converted["output_path"]).read_bytes()
    assert b"\r\n" in data and b"\n" not in data.replace(b"\r\n", b"")
    _assert_translated(extension, data.decode("utf-8"))


def test_vtt_sends_only_the_dialogue_and_masks_its_markup(tmp_path):
    source = tmp_path / "e.vtt"
    source.write_text(SAMPLES[".vtt"][0], encoding="utf-8", newline="")
    assert sp.extract_subtitle_text_segments(str(source)) == ["안녕 & 친구\n잘 가", "잘 가"]
    result = sp.extract_subtitle_to_chapters(str(source), str(tmp_path / "out"))
    body = json.loads(Path(result["chapters_path"]).read_text(encoding="utf-8"))[0]["body"]
    for kept_out in ("-->", "WEBVTT", "메모", "cue-1", "Bob", "&amp;", "<i>", "color"):
        assert kept_out not in body


def test_vtt_output_restores_voice_tags_and_entities(tmp_path):
    source = tmp_path / "e.vtt"
    source.write_text(SAMPLES[".vtt"][0], encoding="utf-8")
    _result, converted = _round_trip(source, tmp_path / "out")
    assert "<v Bob>Hello &amp; friend</v>\n<i>Bye</i>" in Path(converted["output_path"]).read_text(encoding="utf-8")


def test_sbv_keeps_its_line_break_and_analysis_reads_it_as_one(tmp_path):
    source = tmp_path / "e.sbv"
    source.write_text(SAMPLES[".sbv"][0], encoding="utf-8")
    assert sp.extract_subtitle_text_segments(str(source))[0] == "안녕 친구\n잘 가"
    _result, converted = _round_trip(source, tmp_path / "out")
    assert "Hello friend[br]Bye" in Path(converted["output_path"]).read_text(encoding="utf-8")


def test_a_cue_without_a_blank_separator_leaves_the_next_number_alone():
    text = "WEBVTT\n\n00:01.000 --> 00:02.000\n안녕\n2\n00:03.000 --> 00:04.000\n잘 가\n"
    assert [span[2] for span in sp._extract_subtitle_spans(text, ".vtt")] == ["안녕", "잘 가"]


# ---- ZIPs ----------------------------------------------------------------------------------------------------


def _bundle(tmp_path, members):
    """Everything between a ZIP and its translated files, as the pipeline calls it."""
    archive = _zip(tmp_path / "Show.zip", members)
    extracted = sp.extract_subtitle_archive(str(archive), str(tmp_path / "extracted"))
    assert extracted["subtitle_count"] == len([n for n in members if n.endswith(sp.SUBTITLE_EXTENSIONS)])
    plan = sp.plan_subtitle_archive_outputs(str(archive), extracted["files"], str(tmp_path / "outputs"))
    outputs = {path: plan[os.path.normcase(os.path.abspath(path))]["output_path"] for path in extracted["files"]}
    work = tmp_path / "work"
    result = sp.extract_subtitle_bundle_to_chapters(extracted["files"], str(work), output_paths=outputs)
    for chapter in json.loads(Path(result["chapters_path"]).read_text(encoding="utf-8")):
        stem = os.path.splitext(chapter["filename"])[0]
        name = os.path.splitext(chapter["original_basename"])[0]
        batch = chapter.get("subtitle_source_batch_num", 1)
        assert batch == 1, stem
        (work / f"response_{name}.txt").write_text(_translated(chapter), encoding="utf-8")
    converted = sp.convert_subtitle_bundle(str(work))
    assert converted["success"] is True and converted["incomplete_files"] == 0
    return outputs, converted


@pytest.mark.parametrize("extension", FORMATS)
def test_a_zip_of_one_format_translates_whole(tmp_path, extension):
    outputs, converted = _bundle(tmp_path, {f"Season 1/deep/one{extension}": SAMPLES[extension][0],
                                            f"two{extension}": SAMPLES[extension][0], "notes.txt": "x"})
    assert converted["files"] == 2 and len(outputs) == 2
    for output in outputs.values():
        assert output.endswith(extension)
        _assert_translated(extension, Path(output).read_text(encoding="utf-8"))


def test_a_zip_of_all_formats_translates_each_in_its_own_format(tmp_path):
    members = {f"Season {i}/ep{i}{extension}": SAMPLES[extension][0] for i, extension in enumerate(FORMATS)}
    outputs, converted = _bundle(tmp_path, members)
    assert converted["files"] == len(FORMATS)
    assert sorted(os.path.splitext(output)[1] for output in outputs.values()) == FORMATS
    for output in outputs.values():
        _assert_translated(os.path.splitext(output)[1], Path(output).read_text(encoding="utf-8"))


def test_a_zips_batches_queue_across_its_files():
    items = [("a", 1), ("a", 2), ("a", 3), ("b", 1), ("c", 1), ("c", 2)]
    assert sp.interleave_bundle_batches(items, lambda item: item[0]) == [
        ("a", 1), ("b", 1), ("c", 1), ("a", 2), ("c", 2), ("a", 3)]
    assert sp.interleave_bundle_batches([], lambda item: item) == []


class _Owner(InputPreparationMixin):
    def __init__(self, root):
        self.config = {"output_directory": str(root / "outputs")}
        self.subtitle_zip_temp_root = str(root / "temp")
        os.makedirs(self.subtitle_zip_temp_root)
        self.logs = []

    def append_log(self, line):
        self.logs.append(line)


@pytest.mark.parametrize("extension", FORMATS)
def test_every_router_calls_a_zip_of_the_format_subtitles(tmp_path, monkeypatch, extension):
    monkeypatch.delenv("OUTPUT_DIRECTORY", raising=False)
    archive = _zip(tmp_path / "Show.zip", {f"a/b/one{extension}": SAMPLES[extension][0]})

    owner = _Owner(tmp_path)
    resolved = owner._extract_subtitle_zip_input_if_needed(str(archive))
    assert [os.path.splitext(path)[1] for path in resolved] == [extension]
    assert next(iter(owner._subtitle_zip_output_groups.values()))["output_path"].endswith(extension)

    assert ProgressViewMixin._zip_is_subtitle_archive(object.__new__(ProgressViewMixin), str(archive)) is True
    assert is_subtitle_glossary_source(str(archive)) is True
    assert _plan_model().classify_archive(str(archive))["kind"] == "subtitles"


# ---- direct attachments and the lists -------------------------------------------------------------------------


@pytest.mark.parametrize("extension", FORMATS)
def test_a_direct_text_attachment_goes_in_and_comes_out_as_itself(tmp_path, extension):
    source = tmp_path / f"ep{extension}"
    source.write_text(SAMPLES[extension][0], encoding="utf-8")
    assert dts.ChatStoreMixin._is_supported_dropped_text_file(str(source))
    prepared = dts.prepare_direct_text_input("", {"path": str(source)}, temp_parent=str(tmp_path))
    assert prepared["temp_input"] == str(source)
    assert prepared["expected_output"].endswith(f"ep_translated{extension}")


def _plan_model():
    spec = importlib.util.spec_from_file_location("_plan_model_for_subtitles", MOBILE / "ui" / "chat" / "plan_model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _literal(path, name):
    """A mobile list, read from its source: importing its module needs Flet."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0] if isinstance(node, ast.Assign) else None
        if getattr(target, "id", None) == name:
            value = node.value
            if isinstance(value, ast.Call):  # frozenset({...})
                value = value.args[0]
            return [ast.literal_eval(item) for item in value.elts if not isinstance(item, ast.Starred)]
    raise AssertionError(name)


@pytest.mark.parametrize("path, name, shape", [
    (MOBILE / "ui" / "chat" / "direct_text_rules.py", "PLAN_EXTENSIONS", "{}"),
    (MOBILE / "ui" / "chat" / "plan_model.py", "BATCH_EXTENSIONS", "{}"),
    (MOBILE / "ui" / "chat" / "plan_model.py", "_SUBTITLE_EXTENSIONS", "{}"),
    (MOBILE / "job_kinds" / "__init__.py", "TRANSLATED_SUFFIXES", "_translated{}"),
    (MOBILE / "services" / "files.py", "IMPORT_EXTENSIONS", "bare"),
    (MOBILE / "ui" / "chat" / "chat_view.py", "_ATTACH_EXTENSIONS", "bare"),
])
def test_the_mobile_lists_cover_every_format(path, name, shape):
    listed = set(_literal(path, name))
    for extension in sp.SUBTITLE_EXTENSIONS:
        wanted = extension.lstrip(".") if shape == "bare" else shape.format(extension)
        assert wanted in listed, f"{name} misses {wanted}"
