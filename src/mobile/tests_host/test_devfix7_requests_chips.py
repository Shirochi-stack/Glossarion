"""Mobile device fixes 7 (owner, 2026-10-31).

* "The request card area has empty space when there is less than 3 requests": Jobs › job kept a running job's
  cards in a ListView in a box of fixed height, so one card sat over an empty area. The rows are now a plain
  Column at their natural height until they would pass the box's height; then the box scrolls as before. The
  chat's job card follows the same rule, for rows with nothing streamed yet too.
* "Remove the direct stream translated.txt button from the response card": the raw copy of a streamed reply
  (``direct_text_store._persist_output_folder``) is no output. Real files keep their chips, and saved chats
  show the same, since the chips are read from the folder at every render.

A host test sees which control is shown and its height, not the painted pixels: the card heights are
estimates (``request_card_height``) that only a device can check.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="needs flet and msgpack")


def _load(alias: str, file_name: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(file_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_S = _load("_glossarion_devfix7_requests_strip", "test_hotfix2_job_strip_scroll.py")
storage = _S.storage
app_env = _S.app_env
_isolated_env = _S._isolated_env


def _segment(n: int, text: str = "") -> dict:
    return {"id": f"req-{n}", "label": f"Chapter {n}", "phase": "streaming", "text": text, "text_tokens": 40}


# ---- Jobs › job ------------------------------------------------------------------------------------------------


def test_running_job_shows_few_cards_at_their_natural_height_then_scrolls(app_env, tmp_path, monkeypatch):
    from glossarion_mobile.ui.screens import job_detail

    monkeypatch.setattr(job_detail, "REQUEST_REFRESH_S", 0.0)  # every tick refreshes the cards (no clock)
    line = "A line of the reply that fills about one preview line."

    async def scenario() -> None:
        h = _S.Harness(tmp_path)
        try:
            await h.start()
            await _S._TB._route(h.session, f"/jobs/{h.job_id}")
            await h.tick()
            screen = h.job_screen()
            assert not screen.snap.is_terminal and screen.requests_tile.expanded

            # the owner's screenshot: one request of a running job, no empty area under it
            h.segments.append(_segment(1))
            await h.steps.run(lambda host: host.log("request one"))
            await h.tick()
            assert screen.requests_card.visible
            assert screen.requests_box.height is None and screen.requests_box.content is screen.requests_short
            assert len(screen.requests_short.controls) == 1 and not screen.requests_list.controls

            # still a plain Column while the rows fit the box
            h.segments.append(_segment(2))
            await h.steps.run(lambda host: host.log("request two"))
            await h.tick()
            assert screen.requests_box.height is None and len(screen.requests_short.controls) == 2

            # past the maximum: the box of the old height that scrolls itself, with every row, none lost
            for n in range(3, 7):
                h.segments.append(_segment(n, line * 3))
            await h.steps.run(lambda host: host.log("more requests"))
            await h.tick()
            assert screen.requests_box.height == job_detail.JOB_REQUESTS_BOX_HEIGHT
            assert screen.requests_box.content is screen.requests_list and screen.requests_list.auto_scroll
            assert len(screen.requests_list.controls) == 6 and not screen.requests_short.controls

            # a running box stays a box while previews shrink, so the page keeps its height (v9.15.7 batch)
            for segment in h.segments:
                segment["text"] = ""
            await h.steps.run(lambda host: host.log("previews restart"))
            await h.tick()
            assert screen.requests_box.height == job_detail.JOB_REQUESTS_BOX_HEIGHT

            # a list that got shorter (a Resume's new stream) does not keep a box around it
            del h.segments[1:]
            await h.steps.run(lambda host: host.log("a new stream"))
            await h.tick()
            assert screen.requests_box.height is None and screen.requests_box.content is screen.requests_short
            assert len(screen.requests_short.controls) == 1
        finally:
            await h.stop()

    asyncio.run(scenario())


def test_card_height_estimate_follows_what_the_card_shows():
    from glossarion_mobile.ui.screens.job_detail import (JOB_REQUESTS_BOX_HEIGHT, RequestCardData,
                                                         request_card_height)

    bare = RequestCardData("Chapter 1", "", "", "", "0")
    chip = RequestCardData("Chapter 1", "streaming", "", "", "0")
    full = RequestCardData("Chapter 1", "streaming", "40 text", "x" * 280, "0")
    one_line = RequestCardData("Chapter 1", "streaming", "40 text", "x" * 10, "0")
    assert request_card_height(bare) < request_card_height(chip) < request_card_height(one_line)
    assert request_card_height(one_line) < request_card_height(full)
    # a preview is at most 3 lines (``max_lines=3``), however long
    assert request_card_height(RequestCardData("Chapter 1", "streaming", "40 text", "x" * 2800, "0")) == request_card_height(full)
    # the owner's "about 3": two cards never fill the box, four streaming ones always do
    assert request_card_height(full) * 2 + 6 <= JOB_REQUESTS_BOX_HEIGHT
    assert request_card_height(one_line) * 4 + 18 > JOB_REQUESTS_BOX_HEIGHT


# ---- the chat's job card -------------------------------------------------------------------------------------


def _rows(count: int, content: str) -> list:
    return [{"label": f"Chapter {n}", "content": content, "phase": "text", "index": n} for n in range(count)]


def test_chat_card_box_only_once_its_rows_fill_it():
    from glossarion_mobile.ui.chat.cards import REQUESTS_BOX_HEIGHT, JobCard

    card = JobCard(icon="DESCRIPTION", meta="", requests_expanded=True)
    for count in (1, 2, 3, 5):
        card.set_requests(_rows(count, "streamed text"))
        assert card.requests_box.height is None and card.requests_box.content is card.requests_short
        assert len(card.requests_short.controls) == count
    # label-only rows are lower: six of them used to leave an empty area in the box
    card.set_requests(_rows(6, ""))
    assert card.requests_box.height is None and len(card.requests_short.controls) == 6
    card.set_requests(_rows(9, ""))
    assert card.requests_box.height == REQUESTS_BOX_HEIGHT and card.requests_box.content is card.requests_list
    # past the maximum it scrolls as before, with its rows moved over
    card.set_requests(_rows(6, "streamed text"))
    assert card.requests_box.height == REQUESTS_BOX_HEIGHT and card.requests_box.content is card.requests_list
    assert len(card.requests_list.controls) == 6 and not card.requests_short.controls
    assert card.requests_list.auto_scroll


# ---- the stream file's chip ----------------------------------------------------------------------------------


def test_stream_file_is_no_output_and_real_files_keep_their_chips(tmp_path):
    from glossarion_mobile.job_kinds import STREAM_OUTPUT_NAME, compiled_outputs
    from glossarion_mobile.ui.chat.cards import JobCard
    from glossarion_mobile.ui.chat.chat_ops import export_outputs, workspace_outputs

    folder = tmp_path / "Attachments" / "story"
    folder.mkdir(parents=True)
    # what a saved chat's workspace holds on disk: the old copy next to real outputs
    for name in (STREAM_OUTPUT_NAME, "story_translated.txt", "story.epub", "story_translated.pdf", "glossary.csv"):
        (folder / name).write_text("x", encoding="utf-8")

    outputs = workspace_outputs(str(folder))
    names = [Path(path).name for path, _kind in outputs]
    assert STREAM_OUTPUT_NAME not in names
    assert {"story_translated.txt", "story.epub", "story_translated.pdf", "glossary.csv"} <= set(names)
    assert STREAM_OUTPUT_NAME not in [Path(path).name for path, _kind in export_outputs(str(folder), "story.txt")]
    # the Job screen's Outputs card and its other readers
    found = [Path(path).name for path in compiled_outputs([str(folder)])]
    assert STREAM_OUTPUT_NAME not in found and "story_translated.txt" in found

    # the card a saved chat renders: a chip for each real file, none for the stream copy
    card = JobCard(icon="DESCRIPTION", meta="")
    card.set_outputs(outputs)
    labels = [chip.label.value for chip in card.outputs_row.controls]
    assert labels and not [label for label in labels if STREAM_OUTPUT_NAME in label]
    assert any("story_translated.txt" in label for label in labels)

    # only the stream copy: the Result has no chip row at all (and the file stays on disk)
    alone = tmp_path / "Attachments" / "alone"
    alone.mkdir()
    (alone / STREAM_OUTPUT_NAME).write_text("x", encoding="utf-8")
    assert workspace_outputs(str(alone)) == []
    card.set_outputs(workspace_outputs(str(alone)))
    assert not card.outputs_row.controls and not card.outputs_row.visible
    assert (alone / STREAM_OUTPUT_NAME).is_file()


def test_stream_output_name_is_the_one_the_shared_code_writes():
    from glossarion_mobile.job_kinds import STREAM_OUTPUT_NAME

    source = (MOBILE_DIR.parent / "direct_text_store.py").read_text(encoding="utf-8")
    assert f'"{STREAM_OUTPUT_NAME}"' in source


def test_a_chapter_that_is_not_split_names_no_chunk():
    from glossarion_mobile.services.jobs import request_position
    from glossarion_mobile.ui.chat.stream_bridge import request_label_text
    from glossarion_mobile.ui.screens.job_detail import request_card_data

    assert request_label_text("Chapter 2 (chunk 1/1) · ch2.xhtml") == "Chapter 2 · ch2.xhtml"
    assert request_label_text("Chapter 2 (chunk 1/14) · ch2.xhtml") == "Chapter 2 (chunk 1/14) · ch2.xhtml"
    assert request_card_data({"label": "Chapter 2 (chunk 1/1)"}, 0).label == "Chapter 2"
    assert request_position([{"label": "Chapter 2 (chunk 1/1)"}]) == "Ch. 2"
    assert request_position([{"label": "Chapter 2 (chunk 1/14)"}]) == "Ch. 2 (1/14)"
