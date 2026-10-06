"""RESOLVE_QA: resolve one raw foreign-text QA issue with a targeted Partial.b run.

Desktop ``_start_single_progress_qa_resolution`` (Book page › Chapters row ⋯ ›
"⚠️ Resolve QA issue" when the entry has no LLM-token issue, UI_SPEC §3.7):

1. ``progress_actions.prepare_single_qa_resolution(owner, data, display_info)`` - the
   shared preflight - re-checks the entry against the newest ``translation_progress.json``
   ("QA Issue Already Resolved" / "Source File Missing" refusals) and sets the owner's run
   state exactly like the desktop (``_single_qa_resolution_request``, ``selected_files``,
   ``current_file_index`` and the cleared ``_metadata_only_run`` /
   ``_single_chapter_filter`` / ``_force_stream_all``);
2. its log line ("⚠️ Queued Partial.b QA resolution for <label> only");
3. the translation run (``translate.run_translation``: ``_prepare_translation_run`` turns
   the request into a Partial.b multipass run on that entry only, then the worker).

The Book page's busy check happens before the job is queued (the desktop "Process Running"
refusal); the job itself queues behind a running one like every other job.

params: ``request`` (``progress_actions.build_partial_b_request``: ``source_path``,
``progress_path``, ``progress_key``, ``output_file``, ``actual_num``), ``display_info``
(the row fields ``_partial_b_target`` reads) and ``label``. Input: the raw source.
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

from glossarion_mobile.job_kinds.translate import check_inputs, run_translation

__all__ = ["KINDS", "load_progress_data", "run"]


def load_progress_data(request: Mapping[str, Any]) -> dict:
    """The Progress Manager ``data`` the preflight reads, from the newest progress file."""
    progress_file = str(request.get("progress_path") or "")
    source = str(request.get("source_path") or "")
    prog: Any = {}
    if progress_file and os.path.isfile(progress_file):
        with open(progress_file, "r", encoding="utf-8") as handle:
            prog = json.load(handle)
    return {
        "prog": prog if isinstance(prog, dict) else {},
        "progress_file": progress_file,
        "output_dir": os.path.dirname(progress_file) if progress_file else "",
        "file_path": source,
    }


def run(ctx: Any) -> dict:
    from glossarion_mobile.services.jobs import JobError

    params = ctx.params or {}
    request = dict(params.get("request") or {})
    source = str(request.get("source_path") or (ctx.inputs[0] if ctx.inputs else "") or "")
    request.setdefault("source_path", source)
    display_info = dict(params.get("display_info") or {})
    for key in ("progress_key", "output_file"):
        if key not in display_info and request.get(key):
            display_info[key] = request.get(key)
    try:
        import progress_actions
    except Exception as exc:  # pragma: no cover - bundle without the progress core
        raise JobError(f"The shared progress actions (progress_actions) are not in this build ({exc}).") from exc
    prepare = getattr(progress_actions, "prepare_single_qa_resolution", None)
    if not callable(prepare):
        raise JobError("This build's progress_actions has no prepare_single_qa_resolution().")
    try:
        data = load_progress_data(request)
    except (OSError, ValueError) as exc:
        raise JobError(f"The progress file could not be read: {exc}") from exc
    owner = ctx.owner
    preflight = prepare(owner, data, display_info)
    refusal = preflight.get("refusal") if isinstance(preflight, Mapping) else None
    if refusal:
        kind, title, message = (tuple(refusal) + ("", "", ""))[:3]
        ctx.log(f"{'❌' if kind == 'error' else 'ℹ️'} {title}: {message}")
        ctx.set_result(resolve_qa_refusal={"kind": str(kind), "title": str(title), "message": str(message)})
        if kind == "error":
            return {"ok": False, "outputs": [], "error": f"{title}: {message}"}
        return {"ok": True, "outputs": []}
    files = check_inputs([preflight.get("source_path") or source])
    line = preflight.get("log") if isinstance(preflight, Mapping) else ""
    if line:
        ctx.log(line)
    try:
        return run_translation(ctx, files)
    finally:
        # The pipeline clears the request when its run ends; a run that never started must not
        # leave it on the owner (desktop: cleared when run_translation_thread did not start).
        owner._single_qa_resolution_request = None


KINDS = {
    "resolve_qa": {"verb": "Resolving QA issue", "icon": "BUILD", "stop_kind": "translation", "run": run},
}
