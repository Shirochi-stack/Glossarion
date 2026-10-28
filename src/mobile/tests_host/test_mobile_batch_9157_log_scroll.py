"""v9.15.7 batch, issue 2 (owner, 2026-10-28, phone on the hotfix-1 APK): "The logs still do this force scroll
down nonsense. It scrolls past the bottom making it look like I'm passing scroll down" - on the Job page of a
running translation, read at its end (the Log card at the bottom, the JobStrip below it).

What the emulator showed (hotfix-1 x86_64 APK, the app's fake OpenAI server streaming, ``screenrecord`` and a
per-frame position of the Log box): every time a request started, the Job page painted ONE frame with the Log
card 100-300 dp lower, then snapped back - 74 one-frame jumps in 40 s. The Requests tile above the Log is
expanded while the job runs and got a card (~100 dp) per request, so the page grew above the reader. The page's
ListView keeps its offset when content above grows (Flutter), and Flet 1.0.3's ``auto_scroll`` pin is a
post-frame callback (``ScrollableControl._scheduleScrollToEnd``): the frame that lays the new card out is
painted with the old offset - the Log card visibly drops - and the jump back to the end lands one frame later.
The v9.15.7 hotfix-1 tests modelled the pin as if that frame were never seen (``test_hotfix2_job_strip_scroll``
measured after the post-frame callbacks). A shrink above the reader is worse on the device: Flutter does not
clamp the offset at layout, the frame is painted past the new end (blank below the Log) and a spring
simulation starts back - "scrolls past the bottom".

The fix keeps the page's height still while the job runs: the Job page's request cards sit in a box of fixed
height (``JOB_REQUESTS_BOX_HEIGHT``) that scrolls itself to the newest card, like the chat card's Requests list
(U12 item 3), so a new or growing card moves only the box's own content. A finished job's short list is a plain
Column at its natural height again.

``PaintedPageList`` extends ``test_hotfix2_job_strip_scroll.PageListClient`` with the frame Flutter paints:
the Log card's position on screen at layout, before the post-frame pin, with the offset NOT clamped (an offset
past a shrunk end is painted as is). The test reads the Job page at its end through request starts, streamed
previews and a request card that shrinks, and requires the painted Log card never to move.

What only a device can show: the box's own pin has the same one-frame lag inside the box (its newest card is
painted one frame below its end), and rare header changes (an issue chip, the ETA line) still grow the page by
a line for one frame.

Run from src/mobile with the mobile venv::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic -q tests_host/test_mobile_batch_9157_log_scroll.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any

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


# The page-list client simulator, the coarse layout and the running-job harness: one copy, in the hotfix-2 test.
_S = _load("_glossarion_batch9157_log_scroll_strip", "test_hotfix2_job_strip_scroll.py")
_TB = _S._TB
storage = _S.storage
app_env = _S.app_env
_isolated_env = _S._isolated_env


class PaintedPageList(_S.PageListClient):
    """``PageListClient`` plus what each frame paints: the Log card's top on screen at layout (before the
    post-frame pin), with the offset unclamped - Flutter paints an offset past a shrunk end as is and only
    then springs back (``ClampingScrollPhysics.createBallisticSimulation`` out of range)."""

    def __init__(self, *args: Any, log_card: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.log_card = log_card
        self.painted: list[float] = []

    def content_height(self) -> float:  # called once per frame, at layout, before the pin
        height = super().content_height()
        self.painted.append(_S.item_offset(self.list, self.log_card) - self.pixels)
        return height


def test_job_page_never_paints_the_log_card_off_its_place_while_requests_start(app_env, tmp_path, monkeypatch):
    from glossarion_mobile.ui.components.log_console import LAYOUT_ALL_EXTENT
    from glossarion_mobile.ui.screens import job_detail

    monkeypatch.setattr(job_detail, "REQUEST_REFRESH_S", 0.0)  # every tick refreshes the cards (no clock)

    async def scenario() -> None:
        h = _S.Harness(tmp_path)
        try:
            await h.start()
            await _TB._route(h.session, f"/jobs/{h.job_id}")
            await h.tick()
            screen = h.job_screen()
            assert screen.requests_tile.expanded  # the Requests cards are open while the job runs
            log_card = _S._log_card(screen)
            client = PaintedPageList(h.session, h.conn, screen.page_list, h.viewport, log_card=log_card)
            for n in range(2):  # a request or two before the reader goes down
                await h.job_tick(n)
                await client.frame()
            await client.user_scroll_to(client.max_extent)
            await client.frame()
            assert client.distance_from_end == 0.0
            log_top = _S.item_offset(screen.page_list, log_card) - client.pixels
            client.painted.clear()
            box = getattr(getattr(screen, "requests_box", None), "height", None)
            for n in range(2, 14):  # a new request every other tick, the newest preview a line longer each tick
                await h.job_tick(n)
                await client.frame()
            # a request card shrinks (its preview restarts short, the phase chip goes): content above the
            # reader shrinks - on the device that frame is painted past the end, blank under the Log
            h.segments[-1].update(text="<p>Short.</p>", phase="")
            await h.steps.run(lambda host: host.log("<p>Swoooosh... a shorter preview</p>"))
            await h.tick()
            await client.frame()
            await client.frame()
            moved = [round(top - log_top) for top in client.painted if abs(top - log_top) > 0.5]
            assert not moved, (f"the Job page painted the Log card off its place in {len(moved)} of "
                               f"{len(client.painted)} frames (dp, + = down): {moved}")
            assert client.distance_from_end == 0.0
            assert len(h.segments) >= 7 and len(screen.requests_list.controls) == len(h.segments)
            # the cards live in a box of fixed height that follows its newest card itself
            assert screen.requests_box.height == box == job_detail.JOB_REQUESTS_BOX_HEIGHT
            assert screen.requests_box.content is screen.requests_list
            assert screen.requests_list.auto_scroll and screen.requests_list.cache_extent == LAYOUT_ALL_EXTENT
            assert not [b for b in client.list_invokes if b.name == "scroll_to"], "the page chased its end"

            # a finished job's short list is a plain Column at its natural height
            del h.segments[1:]
            h.steps.finish()
            await asyncio.to_thread(_S._J.wait_for, lambda: h.service.snapshot(h.job_id).is_terminal)
            for _ in range(4):
                await h.tick()
            assert screen.snap.is_terminal
            assert screen.requests_box.height is None and screen.requests_box.content is screen.requests_short
            assert len(screen.requests_short.controls) == 1 and not screen.requests_list.controls
        finally:
            await h.stop()

    asyncio.run(scenario())
