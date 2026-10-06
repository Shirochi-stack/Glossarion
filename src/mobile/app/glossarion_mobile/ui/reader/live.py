"""Live "Translate this chapter" feed (UI_SPEC §3.11 LivePanel) over the shared ``live_stream``.

``live_stream.LiveLineClassifier`` is the desktop live view's routing without Qt
(``_classify_live_line`` / ``_drain_live_queue`` byte-for-byte): each job log line
goes to the streamed chapter content, the thinking text or the pipeline log, and
thinking + log share the collapsible pane whose line count is the "🧠 Thinking (n)"
label. ``LiveFeed`` feeds it the job's log lines and keeps render versions for the
native panel (the content is shown as Markdown via ``blocks.html_to_markdown``,
never inside the WebView).

The status strings are the shared ones (``live_stream.LIVE_*``);
``finish_outcome`` is the desktop ``_finish_live_translation`` check (the shared
``chapter_completed_in_progress`` decides; otherwise the shared
``cleanup_incomplete_chapter_output`` clears the partial response) for the output
folder the job recorded.

Imports the backend lazily (the UI module stays importable before the engine is
warm); Python 3.10 compatible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from glossarion_mobile.ui.reader.blocks import html_to_markdown

__all__ = [
    "LiveFeed",
    "LiveOutcome",
    "finish_outcome",
    "outcome_text",
    "retranslate_question",
    "return_delay",
    "status_stopping",
    "status_waiting",
    "thinking_label",
]

RETRANSLATE_TITLE = "Retranslate chapter"


def _live_stream() -> Any:
    import live_stream  # shared GUI-free module (U5)

    return live_stream


def status_waiting(chapter_file: str) -> str:
    """``🛰️ Translating “f” — waiting for stream…`` (``live_stream.LIVE_WAITING_TEXT``)."""
    return _live_stream().LIVE_WAITING_TEXT.format(name=os.path.basename(str(chapter_file or "")))


def status_stopping() -> str:
    return _live_stream().LIVE_STOPPING_TEXT


def outcome_text(completed: bool, stopped: bool) -> str:
    return _live_stream().live_outcome_text(bool(completed), bool(stopped))


def return_delay(completed: bool) -> float:
    """Seconds before the Reader returns to the page (desktop 2.2 s after success, 1.2 s otherwise)."""
    return _live_stream().LIVE_RETURN_DELAY_MS[bool(completed)] / 1000.0


def retranslate_question(title: str) -> str:
    """The desktop confirmation before a completed chapter is translated again live."""
    return f"“{title}” is already translated.\n\nDelete its current translation and retranslate it live?"


def thinking_label(count: int) -> str:
    return f"🧠 Thinking ({count})" if count else "🧠 Thinking"


class LiveFeed:
    """One live run's stream (call ``feed`` on the UI loop)."""

    def __init__(self, chapter_file: str = "", classifier: Any = None) -> None:
        self.classifier = classifier if classifier is not None else _live_stream().LiveLineClassifier(
            chapter_file=chapter_file)
        self.content_version = 0
        self.side_version = 0
        self._markdown_cache: tuple[int, str] = (-1, "")

    def feed(self, messages: Iterable[Any]) -> bool:
        """Route ``messages`` (log lines or whole messages); True when the content grew."""
        classifier = self.classifier
        for raw in messages:
            text = getattr(raw, "text", raw)
            classifier.feed("" if text is None else str(text))
        content_added = False
        side_added = False
        while True:
            result = classifier.drain()
            if result.get("content_added"):
                content_added = True
            if result.get("thinking") or result.get("log"):
                side_added = True
            if not result.get("drained"):
                break
        if content_added:
            self.content_version += 1
        if side_added:
            self.side_version += 1
        return content_added

    @property
    def content(self) -> str:
        return self.classifier.content

    @property
    def side_text(self) -> str:
        return self.classifier.thinking_text

    @property
    def side_count(self) -> int:
        return int(self.classifier.thinking_lines)

    @property
    def streaming(self) -> bool:
        return bool(self.classifier.content)

    def markdown(self) -> str:
        version, cached = self._markdown_cache
        if version != self.content_version:
            cached = html_to_markdown(self.content)
            self._markdown_cache = (self.content_version, cached)
        return cached


@dataclass(frozen=True)
class LiveOutcome:
    completed: bool
    stopped: bool
    cleaned: bool
    output_folder: str
    text: str


def finish_outcome(
    output_folder: str,
    chapter_file: str,
    *,
    stopped: bool,
    chapter_completed: Optional[Callable[[str, str], Any]],
    cleanup_incomplete: Optional[Callable[[str, str], Any]],
) -> LiveOutcome:
    """``_finish_live_translation``'s outcome check for ``output_folder`` (blocking)."""
    completed = False
    cleaned = False
    if output_folder and chapter_file and chapter_completed is not None:
        try:
            completed = bool(chapter_completed(output_folder, chapter_file))
        except Exception:
            completed = False
    if not completed and output_folder and chapter_file and cleanup_incomplete is not None:
        try:
            cleaned = bool(cleanup_incomplete(output_folder, chapter_file))
        except Exception:
            cleaned = False
    return LiveOutcome(completed=completed, stopped=bool(stopped), cleaned=cleaned,
                       output_folder=output_folder or "", text=outcome_text(completed, bool(stopped)))
