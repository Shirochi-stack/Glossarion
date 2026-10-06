"""RETRANSLATE: apply a confirmed Retranslate Selected plan (Book page › Chapters, UI_SPEC §3.7).

The Chapters tab plans on the io pool (``progress_actions.plan_retranslation`` on a
detached copy of the book's ``BookProgress``), shows the desktop confirmation copy and,
for a RECYCLED TOC/header pair, the three-button dialog. The confirmed plan is kept in
memory (``stash``) and this job applies it - the worker half of the desktop
``retranslate_selected`` generator:

* ``progress_actions.apply_retranslation(book, plan, linked_choice)`` deletes the
  selected outputs (whole files or only the selected chunk segments), invalidates
  compiled PDF output, resets metadata / subtitle / artifact / chapter rows (refinement
  state, cached chunks, merged children), resets or deletes the SDLXLIFF sidecars and
  the Machine Translation previews, and merge-writes the progress
  (``_merge_and_write_retranslation_progress``). The sidecar thread count is the core's
  default (the owner's extraction workers when parallel extraction is on, else 1);
* ``progress_actions.retranslation_result_message(result)`` is the desktop result dialog
  ``(kind, title, message)``; it is logged and recorded on the job result
  (``retranslate_kind`` / ``retranslate_title`` / ``retranslate_message``) for the
  Chapters tab's snackbar or sheet.

The job is not resumable: the plan is the confirmation the user saw (with the progress
baseline its three-way merge starts from), so a plan lost with the process is never
re-guessed - the user selects the rows again.

params: ``plan`` (the ``stash`` token), ``linked_choice`` (``both`` / ``selected_only`` /
None), ``count`` (rows, for the title), ``output_dir`` (display only).
"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Optional

__all__ = ["KINDS", "PLAN_GONE", "discard", "pending_tokens", "run", "stash", "take"]

#: The plan of a queued job is gone (the app restarted, or the job was resumed).
PLAN_GONE = ("This retranslation plan is no longer available (the app was restarted). "
             "Select the chapters again and tap Retranslate.")
_MAX_PLANS = 32

_PLANS: dict = {}
_LOCK = threading.Lock()


def stash(book: Any, plan: Any) -> str:
    """Keep a confirmed plan (and the detached book it was made on) for its job; returns the token."""
    token = uuid.uuid4().hex[:16]
    with _LOCK:
        _PLANS[token] = (book, plan)
        while len(_PLANS) > _MAX_PLANS:  # plans of jobs cancelled while queued
            _PLANS.pop(next(iter(_PLANS)))
    return token


def take(token: Optional[str]) -> Optional[tuple]:
    """Remove and return ``(book, plan)`` for ``token`` (None when unknown)."""
    if not token:
        return None
    with _LOCK:
        return _PLANS.pop(str(token), None)


def discard(token: Optional[str]) -> None:
    take(token)


def pending_tokens() -> tuple:
    with _LOCK:
        return tuple(_PLANS)


def _progress_actions() -> Any:
    from glossarion_mobile.services.jobs import JobError

    try:
        import progress_actions
    except Exception as exc:  # pragma: no cover - bundle without the progress core
        raise JobError(f"The shared progress actions (progress_actions) are not in this build ({exc}).") from exc
    if not callable(getattr(progress_actions, "apply_retranslation", None)):
        raise JobError("This build's progress_actions has no apply_retranslation().")
    return progress_actions


def run(ctx: Any) -> dict:
    from glossarion_mobile.services.jobs import JobError

    params = ctx.params or {}
    entry = take(params.get("plan"))
    if entry is None:
        raise JobError(PLAN_GONE)
    book, plan = entry
    pa = _progress_actions()
    count = int(getattr(plan, "count", 0) or params.get("count") or 0)
    ctx.phase("Resetting")
    ctx.log(f"🔁 Retranslate Selected: {count} row(s)")
    result = pa.apply_retranslation(book, plan, params.get("linked_choice"))
    message_fn = getattr(pa, "retranslation_result_message", None)
    kind, title, message = (message_fn(result) if callable(message_fn)
                            else ("info", "Success", "Retranslation reset finished."))
    for line in str(message).splitlines():
        if line.strip():
            ctx.log(line)
    counts = result.as_dict() if callable(getattr(result, "as_dict", None)) else {}
    ctx.set_result(retranslate_kind=str(kind), retranslate_title=str(title), retranslate_message=str(message),
                   retranslate_counts={k: v for k, v in counts.items() if isinstance(v, (int, bool))})
    return {"ok": True, "outputs": []}


KINDS = {
    "retranslate": {"verb": "Resetting for retranslation", "icon": "REPLAY", "stop_kind": "translation",
                    "run": run, "resumable": False},
}
