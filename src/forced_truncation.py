"""Forced Truncation: stop an answer that repeats the same text for longer than allowed (shared, GUI-free).

Owner 2026-10-30: small models can loop on the same lines until the token limit. When the end of the answer has been
the same text over and over for ``seconds``, and the source does not repeat like that, the stream is stopped and the
answer ends with finish reason ``length``, so the Progress Manager marks the chapter truncated and the run moves on.

Users: ``ollamapull`` and the OpenAI-compatible stream of ``unified_api_client`` (config ``forced_truncation_enabled``
/ ``forced_truncation_seconds``, env ``FORCED_TRUNCATION`` / ``FORCED_TRUNCATION_SECONDS``, off by default) and the
on-device engine of Glossarion Mobile (its own switch, on by default). "Mark glossary forced truncation as complete"
(config ``mark_glossary_forced_truncation_complete``, off; on device its own switch, on by default) lets a stopped
glossary request end with ``stop`` instead.

The source check follows the QA scanner's (``scan_html_folder.has_repeating_sentences``): repetition the source has
as often is the author's. Its count of repeated sentences is language-independent; a run of a short unit ("ㅋㅋㅋ")
is not a sentence, so it is looked for directly.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Iterable, Tuple

log = logging.getLogger(__name__)

DEFAULT_SECONDS = 3
#: the same text this many times in a row is a repeat; shorter runs are ordinary text
MIN_REPEATS = 3
_SIGNATURE = 32  # the characters of the answer's end looked for earlier in it
_MAX_CANDIDATES = 64  # earlier copies of them tried as the unit's start: lines of a loop can share their ending
#: the shortest sentence that counts as the source repeating itself: the QA scanner's lowest floor for a source
_SOURCE_SENTENCE_MIN = 10
_TAIL = 12000  # the characters of the answer's end a repeat is looked for in
_CHECK_EVERY = 0.25  # seconds between checks: the answer's end is compared, not every token
_TAG = re.compile(r"<[^>]+>")


def repeat_run(text: str) -> Tuple[int, int]:
    """(length of the repeated unit, times in a row) at the end of ``text``; (0, 0) when its end is not a repeat."""
    size = len(text)
    if size < _SIGNATURE * 2:
        return 0, 0
    signature = text[-_SIGNATURE:]
    search_end = size - 1
    for _ in range(_MAX_CANDIDATES):
        found = text.rfind(signature, 0, search_end)
        if found < 0:
            break
        period = size - _SIGNATURE - found
        unit = text[-period:]
        count, end = 1, size - period
        while end >= period and text[end - period:end] == unit:
            count += 1
            end -= period
        if count >= MIN_REPEATS:
            return period, count
        search_end = found + _SIGNATURE - 1  # the next copy further back
    return 0, 0


def trim_repeats(text: str) -> str:
    """``text`` with the repeat at its end cut to its first time, ending on a whole line where it has lines."""
    if not isinstance(text, str):
        return text
    period, count = repeat_run(text)
    if count < MIN_REPEATS:
        return text
    start = len(text) - period * count
    kept = text[:start + period]
    line_end = kept.rfind("\n")  # the answer stopped mid-line: the half line goes too
    return kept[:line_end + 1] if line_end >= start else kept


def source_repeats(source: Any, count: int) -> bool:
    """The source has text repeated at least ``count`` times: a sentence or line (the QA scanner's count, with its
    lowest length floor), or a run of a short unit in a row on one line ("ㅋㅋㅋ"). Markup, punctuation ("......") and
    short lines ("네.") do not count: every chapter has those, and they say nothing about the model's loop."""
    body = _TAG.sub(" ", source if isinstance(source, str) else "\n".join(_texts(source)))
    if not body.strip():
        return False
    try:
        from scan_html_folder import _repeating_sentence_counts

        if _repeating_sentence_counts(body, min_repeats=count, min_sentence_length=_SOURCE_SENTENCE_MIN):
            return True
    except Exception:
        log.debug("the QA scanner's repetition count is unavailable", exc_info=True)
    run = re.compile(r"([^\W\d_][^\n]{0,59}?)(?:[ \t]*\1){%d,}" % max(1, count - 1))
    return run.search(body) is not None


def _texts(source: Any) -> Iterable[str]:
    for item in source or ():
        yield str(item)


class RepetitionWatch:
    """Fed the answer as it is generated; ``feed`` says when to stop it. ``seconds``: how long the answer's end may
    be a repeat; ``source``: the request's text (``source_repeats``)."""

    def __init__(self, source: Any, seconds: Any = DEFAULT_SECONDS, *, clock: Callable[[], float] = time.monotonic) -> None:
        try:
            self.seconds = max(1.0, float(seconds))
        except (TypeError, ValueError):
            self.seconds = float(DEFAULT_SECONDS)
        self.source = source
        self.clock = clock
        self.fired = False
        self._tail = ""
        self._since = None
        self._checked = clock()
        self._consumed = 0
        self._fed = 0  # characters of the answer so far
        self._since_fed = 0  # ... when the repeat began
        self._since_count = 0  # the repeat's count then

    def feed(self, text: Any) -> bool:
        """Add the answer's next text; True once the answer is to be stopped."""
        if text:
            text = str(text)
            self._fed += len(text)
            self._tail = (self._tail + text)[-_TAIL:]
        now = self.clock()
        if self.fired or now - self._checked < _CHECK_EVERY:
            return self.fired
        self._checked = now
        period, count = repeat_run(self._tail)
        if count < MIN_REPEATS:
            self._since = None
            return False
        if self._since is None:
            self._since, self._since_fed, self._since_count = now, self._fed, count
        # the tail holds only so many copies: past that the count goes on from what was generated
        count = max(count, self._since_count + (self._fed - self._since_fed) // period)
        if now - self._since < self.seconds or source_repeats(self.source, count):
            return False
        self.fired = True
        log.info("Forced Truncation: the same %d characters repeated %d times for %.0f s", period, count, self.seconds)
        return True

    def feed_parts(self, parts: list) -> bool:
        """``feed`` for a growing list of fragments: only those not seen yet."""
        fresh = parts[self._consumed:]
        self._consumed = len(parts)
        return self.feed("".join(str(part) for part in fresh))
