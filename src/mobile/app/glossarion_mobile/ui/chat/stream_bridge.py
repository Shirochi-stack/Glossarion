"""RunStream: live request cards for one chat run (pure Python, no Flet, Python 3.10).

The desktop chat builds itself from the main log stream (UI_SPEC §2.8, inventory
§11): the log listener queues every record, ``_drain_log_queue`` classifies each line
and grows one *request segment* per outbound API call, and
``_request_segment_message`` turns a segment into the persisted v2 assistant tuple.
That model is the shared ``direct_text_stream.DirectTextStream`` (U3), the same code
the desktop dialog inherits.

On mobile JobService owns the model of a job: it creates it when the job starts
(``direct_text_stream.make_stream`` with the run's attachment flag, first request
number and model) and feeds it every raw log line of the job, including the token
payloads the job log suppresses. ``RunStream`` reads it through ``provider`` (the
chat binds ``JobService.request_stream(job_id)``; None while the job is queued):

* ``drain(final=False)`` - classify the queued lines (UI loop tick, the desktop's
  12 ms / 1200-record budget); bumps ``version`` when the cards changed;
* ``segments()`` - copies of the live segment dicts (label, phase, content,
  thinking, thinking_tokens, text_tokens, complete, status_only, order_key, ...) as
  the last drain left them (never drains: JobService also drains on the job side, so
  no backlog builds up while the chat is not on screen);
* ``render_interval_ms()`` - the desktop repaint cadence (280-900 ms, >= 450 ms
  with auto-scroll off) used by the transcript to coalesce streaming updates;
* ``model()`` - the job's ``DirectTextStream`` itself, which the chat hands to
  ``direct_text_store.ChatStore.commit_request_phase`` (the glossary gate) and
  ``ChatStore.finish_run`` (``_finish_translation``) so they continue from its exact
  state.

``segment_processing_label`` is the card's disclosure label from the shared
``request_segment_message`` (Job detail and the chat render live segments with it).
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

from glossarion_mobile.ui.chat.direct_text_rules import stream_render_interval_ms

__all__ = ["RunStream", "as_segment", "segment_message", "segment_processing_label"]

log = logging.getLogger("glossarion.chat.stream")


def as_segment(value: Any) -> dict:
    """A request segment as a dict (copies; the shared model hands out dicts)."""
    if isinstance(value, dict):
        return dict(value)
    try:
        return dict(vars(value))
    except TypeError:
        return {}


def segment_message(segment: dict, output_folder: str = "") -> tuple:
    """The v2 assistant tuple for a request segment (the dialog's ``_request_segment_message``)."""
    from direct_text_stream import request_segment_message  # shared (U3)

    return tuple(request_segment_message(dict(segment), output_folder))


def _model_segments(model: Any) -> list:
    """The model's segments without its unbudgeted drain (``DirectTextStream.segments(drain=False)``)."""
    try:
        return list(model.segments(drain=False))
    except TypeError:  # a model without the keyword (test doubles)
        return list(model.segments())


def segment_processing_label(segment: dict) -> str:
    """The card's disclosure label: "Thinking (1,234 tokens)", "Generating Text (56 tokens)",
    "Token summary  ·  Thinking …  ·  Text …", the NVIDIA queue status or "Processing"."""
    return str(segment_message(segment)[3])


class RunStream:
    """One run's live request cards; see the module docstring."""

    def __init__(
        self,
        *,
        provider: Optional[Callable[[], Any]] = None,
        auto_scroll_disabled: bool = False,
    ) -> None:
        self.provider = provider
        self.auto_scroll_disabled = auto_scroll_disabled
        self._lock = threading.RLock()
        self._signature: Any = None
        self.version = 0

    def model(self) -> Any:
        """The job's ``direct_text_stream.DirectTextStream`` (None before the job starts)."""
        if self.provider is None:
            return None
        try:
            return self.provider()
        except Exception:
            log.debug("request stream provider failed", exc_info=True)
            return None

    @staticmethod
    def _cards_signature(segments: list) -> tuple:
        return tuple(
            (str(s.get("label") or ""), str(s.get("phase") or ""), len(str(s.get("content") or "")),
             len(str(s.get("thinking") or "")), bool(s.get("complete")), int(s.get("text_tokens") or 0),
             int(s.get("thinking_tokens") or 0), str(s.get("status_label") or ""))
            for s in segments
        )

    def drain(self, final: bool = False) -> bool:
        """Classify the queued log lines; True when the cards changed since the last drain.

        ``final=False`` (UI tick) is the desktop's budget (12 ms / 1200 records per call); the
        unbudgeted ``final=True`` belongs on a worker thread (finishing the run)."""
        model = self.model()
        if model is None:
            return False
        with self._lock:
            try:
                model.drain(final=final)
                signature = self._cards_signature(_model_segments(model))
            except Exception:
                log.exception("stream drain failed")
                return False
            changed = signature != self._signature
            self._signature = signature
            if changed:
                self.version += 1
            return changed

    def segments(self) -> list:
        """Copies of the live request segments (spine order) as the last drain left them.

        Never drains: a repaint must not classify a long backlog on the UI loop (the budgeted
        ``drain()`` ticks and JobService's job-side drain keep the cards current)."""
        model = self.model()
        if model is None:
            return []
        try:
            return [as_segment(s) for s in _model_segments(model)]
        except Exception:
            log.debug("reading the request segments failed", exc_info=True)
            return []

    def active_characters(self) -> int:
        return sum(len(str(s.get("content", "") or "")) + len(str(s.get("thinking", "") or "")) for s in self.segments())

    def render_interval_ms(self) -> int:
        return stream_render_interval_ms(self.active_characters(), self.auto_scroll_disabled)
