"""live_stream: the reader's live "Translate this chapter" stream logic (moved from epub_library).

Shared GUI-free core (Glossarion mobile rewrite, milestone U5). ``EpubReaderDialog``
streams a single-chapter translation into a live panel: every main-log line is routed
to the page (``content``), the thinking pane (``thinking``) or the pipeline log
(``log``). ``LiveStreamMixin`` holds those methods byte-for-byte
(``_LIVE_STATUS_CHARS``, ``_classify_live_line``, ``_wrap_live_html``,
``_resolve_live_output_folder``) plus two hooks split out of the dialog without
changing them: ``_drain_live_lines`` (the queue-draining loop of ``_drain_live_queue``)
and ``_live_outcome`` (the completion check + incomplete-output cleanup of
``_finish_live_translation``). ``EpubReaderDialog`` inherits the mixin; Glossarion
Mobile uses :class:`LiveLineClassifier` and renders the stream natively.

Rules: Python 3.10 compatible; never import PySide6, translator_gui or dpi_setup.
"""

import logging
import os
import re
from collections import deque

from library_core import (
    _chapter_completed_in_progress,
    _cleanup_incomplete_chapter_output,
    _resolve_output_roots,
)

# Same logger as before the move (records keep the "epub_library" name).
logger = logging.getLogger("epub_library")

#: Status line shown while the worker spins up (``.format(name=...)``).
LIVE_WAITING_TEXT = "\U0001f6f0️ Translating “{name}” — waiting for stream…"
LIVE_STOPPING_TEXT = "⏹ Stopping…"
LIVE_FINISHED_TEXT = "✅ Translation finished — loading the translated chapter…"
LIVE_STOPPED_TEXT = "⏹ Translation stopped — incomplete output cleared."
LIVE_INCOMPLETE_TEXT = "⚠️ Translation did not complete — incomplete output cleared."
#: Desktop returns to the reader page 2.2 s after a success, 1.2 s otherwise.
LIVE_RETURN_DELAY_MS = {True: 2200, False: 1200}
#: Desktop drain / poll cadence (ms) and the liveness poll's start delay.
LIVE_DRAIN_INTERVAL_MS = 90
LIVE_POLL_INTERVAL_MS = 700
LIVE_POLL_START_DELAY_MS = 2500


def live_outcome_text(completed: bool, was_stopped: bool) -> str:
    """The live panel's final status line (``_finish_live_translation``)."""
    if completed:
        return LIVE_FINISHED_TEXT
    if was_stopped:
        return LIVE_STOPPED_TEXT
    return LIVE_INCOMPLETE_TEXT


class LiveStreamMixin:
    """Live-stream routing shared by ``EpubReaderDialog`` and :class:`LiveLineClassifier`.

    State attributes: ``_live_in_thinking``, ``_live_streaming_text``,
    ``_live_log_queue`` (deque), ``_live_content_buf``, ``_live_think_pending``,
    ``_live_log_pending``, ``_live_chapter_file``, ``_live_epub_path``, ``_epub_path``,
    ``_translated_overlay``, ``_config``; ``_wrap_live_html`` also reads ``_get_theme()``,
    ``_font_family``, ``_font_size`` and ``_line_spacing``.
    """

    # First characters that mark a log line as pipeline status output
    # rather than streamed translation content. Streamed chapter text is
    # printed raw (usually HTML fragments), while status lines from the
    # GUI / pipeline / API client virtually always lead with an emoji,
    # bracket tag or separator run.
    _LIVE_STATUS_CHARS = set(
        "🚀📄📃📜📋✅⚠❌📚📦🔧📊🔍💾🖼🔄📌📸🧠🛰📡⏱⏳🟢🟡🟠🔴🎯📑📖🌐⚡🧪✨🎨💡"
        "🔠🗑🧹📂📁🔁🔂📝🔑🗝🔒🔓🚫⛔💬🌍🌏🌎🐛📈📉🤖🆗═─=[#"
    )

    def _classify_live_line(self, line: str) -> str:
        """Route one log line to ``content`` / ``thinking`` / ``log``."""
        s = line.rstrip("\n")
        stripped = s.strip()
        low = stripped.lower()

        # Thinking block state markers (emitted by unified_api_client when
        # STREAM_THINKING_LOGS is on — which the live view forces).
        if "thinking complete" in low:
            self._live_in_thinking = False
            return "log"
        if stripped.startswith("\U0001f9e0") or " thinking..." in low:
            self._live_in_thinking = "thinking..." in low
            return "log"
        # Thinking content is printed with a 4-space indent.
        if self._live_in_thinking and (s.startswith("    ")
                                       or stripped == "​"):
            return "thinking"

        # Text-stream lifecycle markers.
        if ("text streaming" in low or "first text token" in low):
            self._live_streaming_text = True
            return "log"
        if "stream complete" in low or "translation completed" in low:
            self._live_streaming_text = False
            return "log"

        if not stripped:
            return "content" if self._live_streaming_text else "log"
        first = stripped[0]
        if first in self._LIVE_STATUS_CHARS:
            return "log"
        if stripped.startswith(("Traceback", "File \"", "[DEBUG]", "[INFO]",
                                "[WARN", "[ERROR")):
            return "log"
        # Raw HTML fragments are always chapter content, even if a
        # provider path never printed an explicit stream-start marker.
        if first == "<":
            self._live_streaming_text = True
            return "content"
        return "content" if self._live_streaming_text else "log"

    def _drain_live_lines(self):
        """Route queued log lines into the content / thinking / log buffers."""
        drained = 0
        content_added = False
        while self._live_log_queue and drained < 400:
            try:
                raw = self._live_log_queue.popleft()
            except IndexError:
                break
            drained += 1
            for line in str(raw).split("\n"):
                kind = self._classify_live_line(line)
                if kind == "content":
                    self._live_content_buf += line + "\n"
                    content_added = True
                elif kind == "thinking":
                    self._live_think_pending += line[4:] if line.startswith("    ") else line
                    self._live_think_pending += "\n"
                else:
                    self._live_log_pending += line + "\n"
        return drained, content_added

    def _wrap_live_html(self, body: str) -> str:
        """Style the streamed fragment buffer with the reader's theme/CSS."""
        t = self._get_theme()
        # Real-time line-break handling: HTML fragments rely on their own
        # block tags; plain-text streams need explicit <br> conversion.
        if not re.search(r"<(p|h[1-6]|div|br|li|ul|ol|table|blockquote)\b",
                         body, re.I):
            body = body.replace("\n", "<br>")
        family = self._font_family
        if not family or family == "Embedded CSS":
            family = "Georgia, 'Noto Serif', serif"
        else:
            family = f"'{family}'"
        return (
            "<html><head><style>"
            f"body {{ background: {t['bg']}; color: {t['fg']};"
            f" font-family: {family}; font-size: {self._font_size}pt;"
            f" line-height: {self._line_spacing}; padding: 24px 36px; }}"
            f" h1, h2, h3, h4 {{ color: {t.get('heading', t['fg'])}; }}"
            f" p {{ margin: 0 0 0.9em 0; }}"
            "</style></head><body>"
            f"{body}"
            "</body></html>"
        )

    def _resolve_live_output_folder(self) -> str:
        """Locate the output workspace the live run wrote into.

        Prefers the directory of an existing overlay response file for the
        target chapter; otherwise resolves ``<output_root>/<epub_base>``
        the same way the pipeline computes it.
        """
        chapter_file = (self._live_chapter_file or "").lower()
        try:
            entry = (self._translated_overlay or {}).get(chapter_file)
            if entry and entry.get("path"):
                folder = os.path.dirname(str(entry["path"]))
                if os.path.isdir(folder):
                    return folder
        except Exception:
            pass
        epub_path = self._live_epub_path or self._epub_path
        if not epub_path:
            return ""
        file_base = os.path.splitext(os.path.basename(epub_path))[0]
        with_progress = ""
        plain = ""
        try:
            for root in _resolve_output_roots(self._config):
                cand = os.path.join(root, file_base)
                if not os.path.isdir(cand):
                    continue
                if os.path.isfile(os.path.join(cand,
                                               "translation_progress.json")):
                    with_progress = with_progress or cand
                plain = plain or cand
        except Exception:
            pass
        return with_progress or plain

    def _live_outcome(self):
        """Completion check + incomplete-output cleanup: ``(chapter_file, out_dir, completed, cleaned)``."""
        chapter_file = self._live_chapter_file or ""
        out_dir = self._resolve_live_output_folder() if chapter_file else ""
        completed = bool(
            out_dir and _chapter_completed_in_progress(out_dir, chapter_file))
        cleaned = False
        if not completed and out_dir:
            cleaned = _cleanup_incomplete_chapter_output(out_dir, chapter_file)
        return chapter_file, out_dir, completed, cleaned


class LiveLineClassifier(LiveStreamMixin):
    """Plain live-stream state for Glossarion Mobile's native LivePanel.

    ``feed(message)`` may be called from the job thread (deque append is thread-safe);
    ``drain()`` runs on the UI side every :data:`LIVE_DRAIN_INTERVAL_MS` like desktop.
    """

    def __init__(self, chapter_file: str = "", epub_path: str = "", translated_overlay=None,
                 config=None, theme=None, font_family: str = "Embedded CSS", font_size=14,
                 line_spacing=1.8):
        self._live_log_queue = deque()
        self._live_chapter_file = chapter_file or ""
        self._live_epub_path = epub_path or ""
        self._epub_path = epub_path or ""
        self._translated_overlay = translated_overlay or {}
        self._config = config if config is not None else {}
        self._theme = theme
        self._font_family = font_family
        self._font_size = font_size
        self._line_spacing = line_spacing
        self.thinking_text = ""
        self.log_text = ""
        self.thinking_lines = 0
        self.reset()

    def reset(self) -> None:
        """Clear every buffer and the stream state (desktop ``_reset_live_state``)."""
        self._live_log_queue.clear()
        self._live_content_buf = ""
        self._live_think_pending = ""
        self._live_log_pending = ""
        self._live_in_thinking = False
        self._live_streaming_text = False
        self.thinking_text = ""
        self.log_text = ""
        self.thinking_lines = 0

    def _get_theme(self):
        from reader_doc import reader_theme
        return reader_theme(self._theme if self._theme is not None else 0)

    def classify(self, line: str) -> str:
        """``content`` / ``thinking`` / ``log`` for one line (updates the stream state)."""
        return self._classify_live_line(line)

    def feed(self, message) -> None:
        """Queue one main-log message (call from any thread)."""
        try:
            self._live_log_queue.append(str(message))
        except Exception:
            pass

    def drain(self) -> dict:
        """Drain queued lines; returns ``{drained, content_added, thinking, log}``.

        ``thinking`` / ``log`` are the newly routed texts (the desktop appends them to
        its thinking pane); they also accumulate in ``thinking_text`` / ``log_text``.
        """
        drained, content_added = self._drain_live_lines()
        thinking, log = self._live_think_pending, self._live_log_pending
        self._live_think_pending = ""
        self._live_log_pending = ""
        if thinking or log:
            self.thinking_text += thinking + log
            self.thinking_lines = self.thinking_text.count("\n")
        return {"drained": drained, "content_added": content_added, "thinking": thinking, "log": log}

    @property
    def content(self) -> str:
        """The streamed chapter text so far (HTML fragments or plain text)."""
        return self._live_content_buf

    @property
    def streaming(self) -> bool:
        return bool(self._live_streaming_text)

    def html(self) -> str:
        """The content wrapped like the desktop live page (``_wrap_live_html``)."""
        return self._wrap_live_html(self._live_content_buf)

    def waiting_text(self) -> str:
        return LIVE_WAITING_TEXT.format(name=os.path.basename(self._live_chapter_file or ""))

    def output_folder(self) -> str:
        return self._resolve_live_output_folder()

    def outcome(self, was_stopped: bool = False) -> dict:
        """Finish a run: ``{chapter_file, out_dir, completed, cleaned, status_text, delay_ms}``."""
        chapter_file, out_dir, completed, cleaned = self._live_outcome()
        return {
            "chapter_file": chapter_file,
            "out_dir": out_dir,
            "completed": completed,
            "cleaned": cleaned,
            "status_text": live_outcome_text(completed, was_stopped),
            "delay_ms": LIVE_RETURN_DELAY_MS[bool(completed)],
        }


LIVE_STATUS_CHARS = LiveStreamMixin._LIVE_STATUS_CHARS


def classify_live_lines(lines) -> list:
    """Classify a sequence of lines with fresh state: ``[(kind, line), ...]``."""
    classifier = LiveLineClassifier()
    return [(classifier.classify(line), line) for line in lines]


def wrap_live_html(body: str, theme=None, font_family: str = "Embedded CSS", font_size=14,
                   line_spacing=1.8) -> str:
    """Style a streamed fragment buffer like the reader's live page."""
    return LiveLineClassifier(theme=theme, font_family=font_family, font_size=font_size,
                              line_spacing=line_spacing)._wrap_live_html(body)


def resolve_live_output_folder(chapter_file: str, translated_overlay=None, epub_path: str = "",
                               config=None) -> str:
    """The workspace a live run writes into (overlay response dir, else ``<root>/<epub stem>``)."""
    return LiveLineClassifier(chapter_file=chapter_file, epub_path=epub_path,
                              translated_overlay=translated_overlay,
                              config=config)._resolve_live_output_folder()


__all__ = [
    "LIVE_DRAIN_INTERVAL_MS",
    "LIVE_FINISHED_TEXT",
    "LIVE_INCOMPLETE_TEXT",
    "LIVE_POLL_INTERVAL_MS",
    "LIVE_POLL_START_DELAY_MS",
    "LIVE_RETURN_DELAY_MS",
    "LIVE_STATUS_CHARS",
    "LIVE_STOPPED_TEXT",
    "LIVE_STOPPING_TEXT",
    "LIVE_WAITING_TEXT",
    "LiveLineClassifier",
    "LiveStreamMixin",
    "classify_live_lines",
    "live_outcome_text",
    "resolve_live_output_folder",
    "wrap_live_html",
]
