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
plain text turn render as ordinary cards. A "QA scan" message (a chat QA job started from a
Result card, the ＋ sheet or ``/qa``) is its own item wherever it sits.

Window unit (devfix4, owner issues 12 + 13): the desktop renders every saved message as a card,
mobile renders an attachment turn's request messages as rows of one JobCard. So the chat windows
over its rendered cards, the ``build_items`` of the whole chat: ``item_sizes`` (the desktop
``message_size``; a JobCard its report and the previews of the newest request rows it shows) feed
the desktop loop (``tail_start``), and the window moves with the desktop's arithmetic:
``shift_window`` (``_shift_history_window``: one page, ``end = min(total, start + limit)``) and
``window_after_append`` (``_update_history_window_after_append``: follow new cards only when the
window ended at the old tail). A text chat's cards are its messages, so it windows exactly like the
dialog; a book turn is never cut in two, and its JobCard always knows its turn.
``tests_host/test_chat_transcript_window.py`` compares the arithmetic with the dialog source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

from glossarion_mobile.ui.chat.direct_text_rules import (
    DEFAULT_RENDERED_CARD_LIMIT,
    HISTORY_CHARACTER_BUDGET,
    history_window_bounds,
    normalize_rendered_card_limit,
)

__all__ = [
    "ACTIONS_LABEL",
    "LIBRARY_LABEL",
    "QA_LABEL",
    "REPORT_LABEL",
    "ROW_PREVIEW_CHARS",
    "TranscriptItem",
    "build_items",
    "item_position",
    "item_size",
    "item_sizes",
    "message_size",
    "page_size",
    "shift_window",
    "slide_window",
    "tail_start",
    "tail_window",
    "window_after_append",
    "window_around",
]

REPORT_LABEL = "Extraction report"
ACTIONS_LABEL = "Attachment actions"
#: characters of a request row's preview in a JobCard (``cards._request_row``)
ROW_PREVIEW_CHARS = 400
LIBRARY_LABEL = "Library job"
#: A chat QA scan's card (storage ``{"qa_job", "folder", "source"}``; the scanned workspace in message[4])
QA_LABEL = "QA scan"


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


def _tail_start(count: int, size_of: Callable[[int], int], limit: int, budget: int) -> int:
    """The ``_reset_history_window`` loop: walk back from the newest of ``count`` cards."""
    card_limit = normalize_rendered_card_limit(limit)
    start = count
    visible_count = 0
    visible_characters = 0
    while start > 0:
        size = size_of(start - 1)
        if visible_count >= card_limit:
            break
        if visible_count >= 3 and visible_characters + size > budget:
            break
        start -= 1
        visible_count += 1
        visible_characters += size
    return start


def tail_window(
    messages: Sequence,
    limit: int = DEFAULT_RENDERED_CARD_LIMIT,
    expanded: Iterable[int] = (),
    budget: int = HISTORY_CHARACTER_BUDGET,
) -> tuple:
    """``(start, end)`` of the newest-messages window (``_reset_history_window``)."""
    expanded = set(expanded)
    start = _tail_start(len(messages), lambda i: message_size(messages[i], i, expanded), limit, budget)
    return (start, len(messages))


def tail_start(
    sizes: Sequence[int],
    limit: int = DEFAULT_RENDERED_CARD_LIMIT,
    budget: int = HISTORY_CHARACTER_BUDGET,
) -> int:
    """Start of the newest window over per-card ``sizes`` (``item_sizes``; the same desktop loop)."""
    return _tail_start(len(sizes), lambda i: max(0, int(sizes[i] or 0)), limit, budget)


def page_size(limit: int) -> int:
    """Desktop ``_history_page_size``: ``max(2, limit // 3)``."""
    return max(2, normalize_rendered_card_limit(limit) // 3)


def shift_window(bounds: tuple, total: int, limit: int, direction: int) -> Optional[tuple]:
    """Desktop ``_shift_history_window``: one page (``limit // 3``) towards older (-1) or newer (+1)
    cards, keeping an overlap (``end = min(total, start + limit)``); None when nothing moves."""
    total = max(0, int(total or 0))
    if total <= 0:
        return None
    limit = normalize_rendered_card_limit(limit)
    start = max(0, min(int(bounds[0] or 0), total))
    end = max(start, min(int(bounds[1] or total), total))
    step = max(1, min(limit - 1, page_size(limit)))
    if int(direction) < 0:
        if start <= 0:
            return None
        new_start = max(0, start - step)
        new_end = min(total, new_start + limit)
    else:
        if end >= total:
            return None
        new_end = min(total, end + step)
        new_start = max(0, new_end - limit)
    if (new_start, new_end) == (start, end):
        return None
    return (new_start, new_end)


def window_after_append(bounds: Optional[tuple], previous_count: int, total: int) -> Optional[tuple]:
    """Desktop ``_update_history_window_after_append``: None when the window ended at the old tail
    (it follows the new cards: re-tail), else the window kept and clamped to ``total``."""
    previous_count = max(0, int(previous_count or 0))
    visible_end = int((bounds[1] if bounds is not None else previous_count) or 0)
    if visible_end >= previous_count:
        return None
    total = max(0, int(total or 0))
    start = max(0, min(int(bounds[0] or 0), total))
    return (start, max(start, min(visible_end, total)))


def slide_window(bounds: tuple, total: int, limit: int, direction: int) -> tuple:
    """Slide ``bounds`` by one page towards older (-1) or newer (+1) cards (``shift_window``);
    ``bounds`` itself when it cannot move."""
    shifted = shift_window(bounds, total, limit, direction)
    return shifted if shifted is not None else (max(0, int(bounds[0] or 0)), max(0, int(bounds[1] or 0)))


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
    kind: str  # user | user_file | assistant | job | qa
    # message index; a job's is the index of the user_file turn that owns it, also when that message
    # lies outside ``build_items``' range (``detached``): the card keeps its turn (title, live run,
    # Result state / Resume, workspace) wherever the range starts
    index: int
    requests: list = field(default_factory=list)  # job: assistant indices of request cards
    report: Optional[int] = None  # job: index of the "Extraction report" card
    actions: Optional[int] = None  # job: index of the "Attachment actions" card
    library: Optional[int] = None  # job: index of the "Library job" card (Plan "Save to: Library", U9)
    detached: bool = False  # job: its user_file is before ``build_items``' ``start``

    @property
    def key(self) -> str:
        if self.kind == "job":
            return f"job-{self.index}"
        return f"m-{self.index}"

    def shows(self, index: int) -> bool:
        """True when this item's card shows message ``index`` (a JobCard: its turn's request, report,
        actions and Library cards; the user_file itself is its own file card)."""
        if self.kind != "job":
            return self.index == index
        return index in self.requests or index in (self.report, self.actions, self.library)


def build_items(messages: Sequence, start: int = 0, end: Optional[int] = None) -> list:
    """Render items for ``messages[start:end]`` with attachment turns grouped into JobCards.

    A JobCard is always its turn's: when the range starts inside a turn, its job item still carries
    the owning user_file index (``detached``). The transcript builds the whole chat and windows the
    items (``tail_start`` / ``shift_window``), so a turn is never cut in two."""
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
            label = str(message[5] if len(message) > 5 else "")
            if label == QA_LABEL:
                # its own card, never one of a turn's request cards (it is appended whenever the scan starts)
                items.append(TranscriptItem("qa", i))
                continue
            file_index = owner.get(i)
            if file_index is None:
                items.append(TranscriptItem("assistant", i))
                continue
            job = jobs.get(file_index)
            if job is None:  # the turn's file card is before the range: its JobCard keeps the turn
                job = TranscriptItem("job", file_index, detached=True)
                jobs[file_index] = job
                items.append(job)
            if label == REPORT_LABEL:
                job.report = i
            elif label == ACTIONS_LABEL:
                job.actions = i
            elif label == LIBRARY_LABEL:
                job.library = i
            else:
                job.requests.append(i)
    return items


def item_size(item: TranscriptItem, messages: Sequence, expanded: Iterable[int] = (),
              row_page: int = DEFAULT_RENDERED_CARD_LIMIT) -> int:
    """Characters one rendered card adds to the window: the desktop ``message_size`` of its message; a
    JobCard its "Extraction report" / "Library job" text plus the previews of the newest ``row_page``
    request rows it shows (``cards.JobCard``: ``ROW_PREVIEW_CHARS`` each)."""
    count = len(messages)
    if item.kind != "job":
        return message_size(messages[item.index], item.index, expanded) if 0 <= item.index < count else 0
    size = 0
    for index in (item.report, item.library):
        if index is not None and 0 <= index < count:
            size += message_size(messages[index], index, expanded)
    page = normalize_rendered_card_limit(row_page)
    for index in item.requests[-page:]:
        if 0 <= index < count:
            size += min(ROW_PREVIEW_CHARS, _assistant_chars(messages[index], "content"))
    return size


def item_sizes(items: Sequence[TranscriptItem], messages: Sequence, expanded: Iterable[int] = (),
               row_page: int = DEFAULT_RENDERED_CARD_LIMIT) -> list:
    """``item_size`` of every item (the ``tail_start`` input)."""
    expanded = frozenset(expanded)
    return [item_size(item, messages, expanded, row_page) for item in items]


def item_position(items: Sequence[TranscriptItem], index: int) -> Optional[int]:
    """Where in ``items`` the card showing message ``index`` is (``TranscriptItem.shows``), or None."""
    for position, item in enumerate(items):
        if item.shows(index):
            return position
    return None
