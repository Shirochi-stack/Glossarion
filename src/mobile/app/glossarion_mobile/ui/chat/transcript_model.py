"""Transcript window and turn grouping (pure Python, no Flet, Python 3.10).

Window (UI_SPEC §2.8, desktop ``_reset_history_window``): walking back from the
newest message, render up to ``direct_text_rendered_card_limit`` messages (default
20) within a 120,000-character budget, always at least 3. Assistant sizes come from
the inline text or the persisted ``content_chars`` (expanded thinking counts too),
so saved bodies are never read just to size the window. Scrolling near the top
slides the window by ``limit // 3`` (desktop ``_history_page_size``).

Grouping (UI_SPEC §2.12): a ``user_file`` turn is followed by its JobCard, which
groups every assistant message up to the next user turn - the request cards, the
"Extraction report" and the "Attachment actions" cards. Assistant messages after a
plain text turn render as ordinary cards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from glossarion_mobile.ui.chat.direct_text_rules import (
    DEFAULT_RENDERED_CARD_LIMIT,
    HISTORY_CHARACTER_BUDGET,
    history_window_bounds,
    normalize_rendered_card_limit,
)

__all__ = [
    "ACTIONS_LABEL",
    "REPORT_LABEL",
    "TranscriptItem",
    "build_items",
    "message_size",
    "page_size",
    "slide_window",
    "tail_window",
]

REPORT_LABEL = "Extraction report"
ACTIONS_LABEL = "Attachment actions"
LIBRARY_LABEL = "Library job"


def _role(message: object) -> str:
    return str(message[0]) if isinstance(message, (list, tuple)) and message else ""


def _assistant_chars(message: Sequence, kind: str) -> int:
    inline_index = 2 if kind == "thinking" else 1
    inline = str(message[inline_index] if len(message) > inline_index else "")
    if inline:
        return len(inline)
    storage = message[6] if len(message) > 6 and isinstance(message[6], dict) else {}
    try:
        return max(0, int(storage.get(f"{kind}_chars", 0) or 0))
    except (TypeError, ValueError):
        return 0


def message_size(message: Sequence, index: int, expanded: Iterable[int] = ()) -> int:
    """Characters one message adds to the rendered window (desktop rule)."""
    if _role(message) == "assistant":
        size = _assistant_chars(message, "content")
        if index in set(expanded):
            size += _assistant_chars(message, "thinking")
        return size
    size = len(str(message[1] if len(message) > 1 else ""))
    if index in set(expanded):
        size += len(str(message[2] if len(message) > 2 else ""))
    return size


def tail_window(
    messages: Sequence,
    limit: int = DEFAULT_RENDERED_CARD_LIMIT,
    expanded: Iterable[int] = (),
    budget: int = HISTORY_CHARACTER_BUDGET,
) -> tuple:
    """``(start, end)`` of the newest-messages window (``_reset_history_window``)."""
    expanded = set(expanded)
    card_limit = normalize_rendered_card_limit(limit)
    start = len(messages)
    visible_count = 0
    visible_characters = 0
    while start > 0:
        size = message_size(messages[start - 1], start - 1, expanded)
        if visible_count >= card_limit:
            break
        if visible_count >= 3 and visible_characters + size > budget:
            break
        start -= 1
        visible_count += 1
        visible_characters += size
    return (start, len(messages))


def page_size(limit: int) -> int:
    """Desktop ``_history_page_size``: ``max(2, limit // 3)``."""
    return max(2, normalize_rendered_card_limit(limit) // 3)


def slide_window(bounds: tuple, total: int, limit: int, direction: int) -> tuple:
    """Slide ``bounds`` by one page towards older (-1) or newer (+1) messages."""
    start, end = bounds
    limit = normalize_rendered_card_limit(limit)
    step = page_size(limit)
    if direction < 0:
        start = max(0, start - step)
        end = min(total, max(end - step, start + min(limit, total)))
        if end - start > limit:
            end = start + limit
    else:
        end = min(total, end + step)
        start = max(0, min(start + step, end - limit))
        if end - start > limit:
            start = end - limit
    return (max(0, start), max(0, end))


def window_around(total: int, limit: int, index: int) -> tuple:
    """Centre the window on ``index`` (jump procedure, desktop ``_set_history_window_around``)."""
    limit = normalize_rendered_card_limit(limit)
    if total <= 0:
        return (0, 0)
    if index >= total:
        return history_window_bounds(total, limit)
    return history_window_bounds(total, limit, index)


@dataclass
class TranscriptItem:
    kind: str  # user | user_file | assistant | job
    index: int  # message index (job: the owning user_file index, or -1 when it is outside the window)
    requests: list = field(default_factory=list)  # job: assistant indices of request cards
    report: Optional[int] = None  # job: index of the "Extraction report" card
    actions: Optional[int] = None  # job: index of the "Attachment actions" card
    library: Optional[int] = None  # job: index of the "Library job" card (Plan "Save to: Library", U9)

    @property
    def key(self) -> str:
        if self.kind == "job":
            anchor = self.index if self.index >= 0 else (self.requests[0] if self.requests else self.report)
            return f"job-{anchor}"
        return f"m-{self.index}"


def build_items(messages: Sequence, start: int = 0, end: Optional[int] = None) -> list:
    """Render items for ``messages[start:end]`` with attachment turns grouped into JobCards."""
    end = len(messages) if end is None else min(end, len(messages))
    # Which user_file owns each assistant message (scan from the beginning so a window
    # starting mid-group still groups its cards).
    owner: dict = {}
    current_file: Optional[int] = None
    for i, message in enumerate(messages[:end]):
        role = _role(message)
        if role == "user_file":
            current_file = i
        elif role == "user":
            current_file = None
        elif role == "assistant" and current_file is not None:
            owner[i] = current_file
    items: list = []
    jobs: dict = {}
    for i in range(max(0, start), end):
        message = messages[i]
        role = _role(message)
        if role == "user":
            items.append(TranscriptItem("user", i))
        elif role == "user_file":
            items.append(TranscriptItem("user_file", i))
            job = TranscriptItem("job", i)
            jobs[i] = job
            items.append(job)
        elif role == "assistant":
            file_index = owner.get(i)
            if file_index is None:
                items.append(TranscriptItem("assistant", i))
                continue
            job = jobs.get(file_index)
            if job is None:
                job = TranscriptItem("job", -1)
                jobs[file_index] = job
                items.append(job)
            label = str(message[5] if len(message) > 5 else "")
            if label == REPORT_LABEL:
                job.report = i
            elif label == ACTIONS_LABEL:
                job.actions = i
            elif label == LIBRARY_LABEL:
                job.library = i
            else:
                job.requests.append(i)
    return items
