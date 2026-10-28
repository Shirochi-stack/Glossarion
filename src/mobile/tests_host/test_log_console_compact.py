"""Owner 2026-10-18 (the job detail "Log" card on a phone):

1. "we need a button to disable auto scroll in the logs here. Add it next to the share button." The
   LogConsole's follow toggle is an auto-scroll button (key ``log-autoscroll``, pause / resume icons, a
   tooltip naming the state, the 48 dp hit target of Share); the job detail shows it in the Log card
   header next to Share. Off: new lines are appended without scrolling; on again: jumps to the end. The
   choice holds for the rest of the app session (every new LogConsole opens with it).
2. "reduce the width on the buttons next to the Download and copy button so the API tab wouldn't need me
   to horizontally scroll on it." All / Errors / Thinking / API and Copy fit one row of a 360 dp phone at
   100 % text (compact chips: LABEL_MEDIUM, small padding; Copy 40 dp wide, still 48 dp tall); the chip
   row keeps its sideways scroll for larger text.

The width math is the estimate test_welcome_drawer_compact uses (Roboto advances, capitals ~0.66 em,
lowercase ~0.52 em) plus Flutter's chip layout: padding + label padding on both sides and a 1 dp border.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
for entry in (str(MOBILE_DIR / "app"), str(MOBILE_DIR.parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

ft = pytest.importorskip("flet")

from glossarion_mobile.services.dispatcher import LogLine  # noqa: E402
from glossarion_mobile.ui import tokens  # noqa: E402
from glossarion_mobile.ui.components import log_console as lc  # noqa: E402
from glossarion_mobile.ui.components.log_console import FILTERS, LogConsole  # noqa: E402

PHONES = (360, 412)  # dp widths
HIT = tokens.SIZES["hit_target"]
#: the job detail ListView's padding (md) and the SectionCard's card_padding, on both sides
INSET = 2 * tokens.SPACING["md"] + 2 * tokens.SPACING["card_padding"]


def text_width(text: str, size: float) -> float:
    return sum((0.66 if ch.isupper() else 0.30 if ch == " " else 0.52) * size for ch in str(text))


def walk(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        for name in ("controls", "content", "trailing"):
            child = getattr(node, name, None)
            if isinstance(child, list):
                stack.extend(reversed([c for c in child if isinstance(c, ft.Control)]))
            elif isinstance(child, ft.Control):
                stack.append(child)


@pytest.fixture(autouse=True)
def _session_reset():
    saved = dict(lc.SESSION)
    lc.SESSION["following"] = True
    yield
    lc.SESSION.clear()
    lc.SESSION.update(saved)


def chip_width(chip, scale: float = 1.0) -> float:
    size = chip.label_text_style.size * scale
    pad = (chip.padding.left + chip.padding.right) + (chip.label_padding.left + chip.label_padding.right)
    return text_width(chip.label.value, size) + pad + 2  # 1 dp border a side


def button_width(button) -> float:
    return button.size_constraints.min_width


def toolbar_width(console: LogConsole, scale: float = 1.0) -> float:
    toolbar = console.toolbar
    chips = list(console.filter_chips.values())
    width = sum(chip_width(c, scale) for c in chips) + console.filter_row.spacing * (len(chips) - 1)
    buttons = [c for c in toolbar.controls if c is not console.filter_row and c.visible is not False]
    width += sum(button_width(b) for b in buttons) + toolbar.spacing * len(buttons)
    return width


# ==========================================================================
# 2. the filter row fits a phone
# ==========================================================================


@pytest.mark.parametrize("phone", PHONES)
@pytest.mark.parametrize("header_follow", (True, False))
def test_filter_chips_and_copy_fit_a_phone_without_scrolling(phone, header_follow):
    console = LogConsole(copy_handler=lambda text: None, header_follow=header_follow)
    assert [c.label.value for c in console.filter_chips.values()] == [label for _f, label, _k in FILTERS]
    assert console.copy_button in console.toolbar.controls
    assert (console.follow_button in console.toolbar.controls) is not header_follow
    width = toolbar_width(console)
    assert width <= phone - INSET, (phone, header_follow, width)


def test_compact_controls_keep_their_hit_targets_and_the_scroll_fallback():
    console = LogConsole(copy_handler=lambda text: None)
    for chip in console.filter_chips.values():
        assert chip.label_text_style.size <= tokens.TYPE_SCALE["label_medium"].size
        assert chip.visual_density == ft.VisualDensity.COMPACT
    # Copy and the in-row follow toggle are narrower, never shorter than the 48 dp tap target
    for button in (console.copy_button, console.follow_button):
        assert 40 <= button.size_constraints.min_width < HIT
        assert button.size_constraints.min_height >= HIT
    # larger text still scrolls sideways instead of clipping the API chip
    assert console.filter_row.scroll == ft.ScrollMode.AUTO and console.filter_row.expand
    assert toolbar_width(console, scale=2.0) > 360 - INSET


# ==========================================================================
# 1. auto-scroll toggle
# ==========================================================================


def line(text: str, kind: str = "info") -> LogLine:
    return LogLine(0, 0.0, text, kind)


class _Dispatcher:
    def __init__(self):
        self.dirty = []

    def on_loop_thread(self):
        return True

    def mark_dirty(self, control):
        self.dirty.append(control)


def _live(console: LogConsole, monkeypatch) -> list:
    """Pretend the console is mounted; return the list of scroll_to calls it makes."""
    scrolls = []

    async def done():
        return None

    def scroll_to(**kw):
        scrolls.append(kw)
        return done()

    monkeypatch.setattr(console, "_page_or_none", lambda: object())
    monkeypatch.setattr(lc.asyncio, "ensure_future", lambda coro: coro.close())
    object.__setattr__(console.list_view, "scroll_to", scroll_to)
    return scrolls


def test_the_host_hears_every_painted_change_to_keep_its_page_at_the_end(monkeypatch):
    """Owner 2026-10-21: the job page crept up off its bottom as lines arrived; the host re-pins it after each change."""
    heard = []
    console = LogConsole(dispatcher=_Dispatcher(), on_after_change=lambda: heard.append(console.following))
    scrolls = _live(console, monkeypatch)
    console.on_lines([line("one")], 0)
    # hotfix 5: following is the list's auto_scroll (the client pins after layout), never a scroll_to per batch
    assert heard == [True] and scrolls == [] and console.list_view.auto_scroll
    console.set_following(False)
    console.on_lines([line("two")], 0)
    # the host hears; paused, the line waits (auto_scroll itself never changes, v9.15.7 hotfix 1)
    assert heard == [True, False, False] and scrolls == [] and console.list_view.auto_scroll
    assert [held.text for held in console.pending] == ["two"]


def test_autoscroll_toggle_pauses_and_resumes_following(monkeypatch):
    console = LogConsole(dispatcher=_Dispatcher(), header_follow=True)
    button = console.follow_button
    assert button.key == "log-autoscroll" and console.following and button.selected
    assert button.size_constraints.min_width >= HIT and button.size_constraints.min_height >= HIT
    on_icon, on_tip = button.icon, button.tooltip
    assert "on" in on_tip.lower()
    scrolls = _live(console, monkeypatch)

    console.on_lines([line("first")], 0)
    assert scrolls == [] and console.list_view.auto_scroll  # following: the client pins the end after layout

    button.on_click(None)  # off: new lines wait (the view stays put), Copy / Share still get them
    assert not console.following and not button.selected and console.list_view.auto_scroll
    assert button.icon != on_icon and button.tooltip != on_tip and "paused" in button.tooltip.lower()
    console.on_lines([line("second"), line("third")], 0)
    assert scrolls == [] and [held.text for held in console.pending] == ["second", "third"]
    assert console.visible_text() == "first\nsecond\nthird"

    button.on_click(None)  # on again: the held lines go in and it jumps to the end now
    assert console.following and button.selected and button.icon == on_icon and button.tooltip == on_tip
    assert console.list_view.auto_scroll and len(scrolls) == 1 and scrolls[-1]["offset"] == -1
    assert not console.pending and "third" in console.list_view.controls[-1].value


def test_autoscroll_choice_lasts_for_the_session():
    first = LogConsole()
    first._toggle_follow()
    assert lc.SESSION["following"] is False
    second = LogConsole()
    assert not second.following and not second.follow_button.selected
    assert "paused" in second.follow_button.tooltip.lower()
    second._toggle_follow()
    assert LogConsole().following


# ==========================================================================
# the job detail Log card: auto-scroll next to Share
# ==========================================================================


def _job_helpers():
    spec = importlib.util.spec_from_file_location("_glossarion_jobs_helpers_logcompact",
                                                  Path(__file__).with_name("test_jobs.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_job_detail_log_header_has_autoscroll_next_to_share(tmp_path):
    jobs = _job_helpers()
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.job_detail import JobDetailScreen

    service, _backend = jobs.make_service(tmp_path)
    try:
        jid = service.submit(jobs.JobSpec("translate", "Book", (jobs.epub(tmp_path),)))
        detail = JobDetailScreen(parse_route(f"/jobs/{jid}"), service=service, copy_handler=lambda text: None)
        body = detail.build_body()
        card = next(c for c in walk(body) if getattr(c, "key", None) == "job-log")
        header = card.trailing
        keys = [getattr(c, "key", None) for c in header.controls]
        assert keys == ["log-autoscroll", "log-share"]
        assert header.controls[0] is detail.console.follow_button
        assert detail.console.header_follow and detail.console.follow_button not in detail.console.toolbar.controls
        assert toolbar_width(detail.console) <= 360 - INSET
    finally:
        service.close()
