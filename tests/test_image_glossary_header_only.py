"""An image without names is a finished glossary extraction, not a failure.

Owner log (v9.15.14): Auto Glossary Full on an image folder whose one image has no text. The
model answered with only the CSV header (``type,raw_name,translated_name,gender``); the image
glossary logged "Failed to parse response as JSON or CSV", "No glossary entries extracted from
any images", and the "complete glossary before translation" gate then blocked the translation
("glossary is below 100% (glossary progress is unavailable)"): the gate looked for a per-image
``<image>_glossary_progress.json`` although the image glossary keeps one
``<folder>_glossary_progress.json``.

Checked here: the header-only / ``[]`` answer is recorded as a completed image with 0 entries,
the extraction succeeds, the gate finds the folder progress and lets the translation run, while
a refusal, a cut-off reply and an errored image still fail / block.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

from glossary_translation_gate import (  # noqa: E402
    glossary_complete,
    image_glossary_complete,
    is_empty_glossary_answer,
    progress_path_for_source,
)

HEADER = "type,raw_name,translated_name,gender"


@pytest.mark.parametrize("reply", [
    HEADER,
    HEADER + "\n",
    "```csv\n" + HEADER + "\n```",
    "type\x1fraw_name\x1ftranslated_name\x1fgender",
    "type\\x1Fraw_name\\x1Ftranslated_name\\x1Fgender",
    "raw_name,translated_name,gender",
    "| type | raw_name | translated_name |\n|---|---|---|",
    "<think>no text in the image</think>\n" + HEADER,
    "[]",
    "```json\n[]\n```",
    '{"entries": []}',
])
def test_header_only_and_empty_list_mean_zero_entries(reply):
    assert is_empty_glossary_answer(reply)
    assert is_empty_glossary_answer(reply, "stop")


@pytest.mark.parametrize("reply", [
    "",
    "   ",
    "{}",
    "I can't read any text in this image.",
    HEADER + "\ncharacter,김철수,Kim Cheolsu,Male",
    HEADER + "\nNo names found.",
    "type,raw_na",
    "type,raw_name,번역,gender",
    '[{"type": "character", "raw_name": "A"}]',
])
def test_garbage_and_rows_are_not_empty_answers(reply):
    assert not is_empty_glossary_answer(reply)


def test_cut_off_header_is_not_an_answer():
    assert not is_empty_glossary_answer(HEADER, "length")
    assert not is_empty_glossary_answer("[]", "max_tokens")


# ---------------------------------------------------------------------------
# the image-folder glossary end to end (stubbed API)
# ---------------------------------------------------------------------------


class _FakeClient:
    def __init__(self, model=None, api_key=None):
        self.model = model

    @staticmethod
    def _model_needs_api_key(model):
        return False


class _Entry:
    def __init__(self, text=""):
        self._text = text

    def text(self):
        return self._text


@pytest.fixture
def image_owner(tmp_path, monkeypatch):
    import unified_api_client
    from translation_pipeline import GlossaryPipelineMixin

    saved_env = dict(os.environ)
    monkeypatch.setattr(unified_api_client, "UnifiedClient", _FakeClient)
    monkeypatch.chdir(tmp_path)

    class Owner(GlossaryPipelineMixin):
        def __init__(self):
            self.logs = []
            self.replies = []
            self.api_key_entry = _Entry("")
            self.model_var = "authgpt/gpt-6.1-sol"
            self.manual_glossary_prompt = "Extract names. {fields1}"
            self.config = {"manual_glossary_temperature": 0.1, "glossary_max_output_tokens": 4096}
            self.max_output_tokens = 4096
            self.stop_requested = False
            self.custom_entry_types = {"character": {"enabled": True, "has_gender": True},
                                       "term": {"enabled": True, "has_gender": False}}

        def append_log(self, message):
            self.logs.append(str(message))

        def _custom_prefix_routes_env_json(self):
            return "[]"

        def _ollama_settings_env_json(self):
            return "{}"

        def _call_api_with_interrupt(self, client, messages, image_base64, temperature, max_tokens,
                                     context="image_translation"):
            return self.replies.pop(0)

    folder = tmp_path / "Chikuwa"
    folder.mkdir()
    image = folder / "Chikuwa_NoChibi_normal.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n not really a png, no text either")
    owner = Owner()
    owner.image = str(image)
    owner.glossary_dir = str(tmp_path / "Glossary")
    try:
        yield owner
    finally:
        os.environ.clear()
        os.environ.update(saved_env)


def _image_rows(owner):
    progress_path = Path(owner.glossary_dir) / "Chikuwa_glossary_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    return [row["status"] for row in progress["images"].values()]


def _gate(owner, tmp_path):
    progress_path = progress_path_for_source(owner.image, str(tmp_path))
    return glossary_complete(progress_path, None, {}, source_path=owner.image)


@pytest.mark.parametrize("reply", [HEADER, "[]"])
def test_image_without_text_completes_with_zero_entries(image_owner, tmp_path, reply):
    image_owner.replies = [(reply, "stop")]
    assert image_owner._process_image_folder_for_glossary("Chikuwa", [image_owner.image], image_owner.glossary_dir)
    log = "\n".join(image_owner.logs)
    assert "Failed to parse response" not in log
    assert "No glossary entries extracted from any images" not in log
    assert _image_rows(image_owner) == ["completed"]

    # the translation gate finds the folder progress and does not block
    assert _gate(image_owner, tmp_path) == (True, "complete")

    # a second run resumes: the image is already done, nothing is sent again
    image_owner.logs.clear()
    assert image_owner._process_image_folder_for_glossary("Chikuwa", [image_owner.image], image_owner.glossary_dir)
    assert any("Already processed" in line for line in image_owner.logs)


@pytest.mark.parametrize("reply", [
    ("I can’t directly edit or generate an image in this chat.", "stop"),
    (HEADER, "length"),
])
def test_refusal_or_cut_off_reply_stays_a_failure(image_owner, tmp_path, reply):
    image_owner.replies = [reply]
    assert not image_owner._process_image_folder_for_glossary(
        "Chikuwa", [image_owner.image], image_owner.glossary_dir)
    log = "\n".join(image_owner.logs)
    assert "Failed to parse response as JSON or CSV" in log
    assert "No glossary entries extracted from any images" in log
    assert _image_rows(image_owner) == ["error"]

    ready, reason = _gate(image_owner, tmp_path)
    assert not ready
    assert "Chikuwa_NoChibi_normal.png: error" in reason


def test_gate_without_any_image_progress_still_blocks(tmp_path):
    image = tmp_path / "Chikuwa" / "page.png"
    image.parent.mkdir()
    image.write_bytes(b"png")
    assert progress_path_for_source(str(image), str(tmp_path)) is None
    assert glossary_complete(None, None, {}, source_path=str(image)) == (
        False, "glossary progress is unavailable")


def test_gate_reads_the_folder_progress_under_an_output_override(tmp_path):
    image = tmp_path / "in" / "Chikuwa" / "page.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png")
    out = tmp_path / "out"
    (out / "Glossary").mkdir(parents=True)
    import hashlib
    digest = hashlib.sha256(b"png").hexdigest()
    progress_path = out / "Glossary" / "Chikuwa_glossary_progress.json"
    progress_path.write_text(json.dumps({"images": {digest: {"status": "skipped_cover"}}}), encoding="utf-8")
    assert progress_path_for_source(str(image), str(out)) == str(progress_path)
    assert image_glossary_complete(str(progress_path), [str(image)]) == (True, "complete")


def test_gate_finds_a_second_folders_progress_beside_the_first_selected_file(tmp_path):
    """Without an output override every image folder's progress goes to Glossary beside the
    first selected file; the worker passes that folder as ``image_glossary_dir``."""
    first = tmp_path / "Chikuwa" / "a.png"
    second = tmp_path / "other" / "Gari" / "b.png"
    for image in (first, second):
        image.parent.mkdir(parents=True)
        image.write_bytes(image.name.encode())
    glossary_dir = first.parent / "Glossary"
    glossary_dir.mkdir()
    import hashlib
    progress_path = glossary_dir / "Gari_glossary_progress.json"
    progress_path.write_text(json.dumps({"images": {
        hashlib.sha256(b"b.png").hexdigest(): {"status": "completed"}}}), encoding="utf-8")
    cwd_root = str(tmp_path / "cwd")
    assert progress_path_for_source(str(second), cwd_root) is None
    found = progress_path_for_source(str(second), cwd_root, image_glossary_dir=str(glossary_dir))
    assert found == str(progress_path)
    assert glossary_complete(found, None, {}, source_path=str(second)) == (True, "complete")


def test_worker_gate_passes_the_image_glossary_folder():
    """The pre-translation gate looks where run_glossary_extraction_direct wrote the image progress."""
    text = (SRC / "translation_pipeline.py").read_text(encoding="utf-8")
    gate = text[text.index("from glossary_translation_gate import glossary_complete, progress_path_for_source"):]
    gate = gate[:gate.index("Translation blocked for")]
    assert "image_glossary_dir=os.path.join(" in gate
    assert "os.path.dirname(os.path.abspath(self.selected_files[0]))" in gate
    assert "source_path=source" in gate


ENTRY = {"type": "character", "raw_name": "김철수", "translated_name": "Kim Cheolsu", "gender": "Male"}


@pytest.mark.parametrize("reply", [
    json.dumps({"entries": [ENTRY]}, ensure_ascii=False),
    json.dumps({"glossary": [ENTRY]}, ensure_ascii=False),
    json.dumps(ENTRY, ensure_ascii=False),
])
def test_json_object_reply_keeps_its_entries(image_owner, reply):
    """A wrapped list or a single entry object is the glossary, not an image with 0 entries."""
    image_owner.replies = [(reply, "stop")]
    assert image_owner._process_image_folder_for_glossary("Chikuwa", [image_owner.image], image_owner.glossary_dir)
    log = "\n".join(image_owner.logs)
    assert "Extracted 1 total entries" in log
    assert "No glossary entries in" not in log
    assert _image_rows(image_owner) == ["completed"]
    glossary = json.loads((Path(image_owner.glossary_dir) / "Chikuwa_glossary.json").read_text(encoding="utf-8"))
    assert "김철수" in json.dumps(glossary, ensure_ascii=False)


def test_empty_wrapped_json_list_is_zero_entries(image_owner, tmp_path):
    image_owner.replies = [('{"entries": []}', "stop")]
    assert image_owner._process_image_folder_for_glossary("Chikuwa", [image_owner.image], image_owner.glossary_dir)
    assert _image_rows(image_owner) == ["completed"]
    assert _gate(image_owner, tmp_path) == (True, "complete")


@pytest.mark.parametrize("reply", ["{}", '{"error": "cannot read the image"}', '"I cannot read any text"', "42"])
def test_json_that_is_not_a_glossary_stays_a_failure(image_owner, tmp_path, reply):
    """Before the fix these were "completed with 0 entries", so the folder now counted as done."""
    image_owner.replies = [(reply, "stop")]
    assert not image_owner._process_image_folder_for_glossary(
        "Chikuwa", [image_owner.image], image_owner.glossary_dir)
    assert "No glossary entries extracted from any images" in "\n".join(image_owner.logs)
    assert _image_rows(image_owner) == ["error"]
    assert not _gate(image_owner, tmp_path)[0]
