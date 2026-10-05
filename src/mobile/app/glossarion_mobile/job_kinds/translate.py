"""TRANSLATE: translate input files with the owner's settings (desktop "Run Translation").

Calls ``owner._prepare_translation_run(files)`` and
``owner._translation_worker(request)``, the desktop ``run_translation_thread``
body moved into ``translation_pipeline.TranslationPipelineMixin``. ZIP/HTML to
EPUB preparation, the automatic glossary pass (the effective glossary mode of
the config snapshot), chunking, header translation, EPUB compilation and the
post-translation QA scan therefore follow the owner exactly as on desktop, and
a resumed job continues from ``translation_progress.json``.

The adapter only validates the inputs, records each input's output folder
(``owner._resolve_translation_output_dir``: checkpointed for recovery and read
by ``ProgressWatcher``), maps the worker's outcome to the job state (False:
Failed, or Stopped after a Stop) and lists the compiled files afterwards.
"""

from __future__ import annotations

import os
from typing import Any

from glossarion_mobile.job_kinds import compiled_outputs, owner_method, result_fields

__all__ = ["KINDS", "NOT_COMPLETED", "check_inputs", "record_output_dirs", "run", "run_translation"]


def check_inputs(paths: Any) -> list:
    from glossarion_mobile.services.jobs import JobError

    files = [os.path.abspath(os.fspath(p)) for p in (paths or ()) if p]
    if not files:
        raise JobError("Nothing to translate: no input file.")
    missing = [os.path.basename(p) for p in files if not os.path.exists(p)]
    if missing:
        raise JobError(f"File not found: {', '.join(missing)}")
    return files


def record_output_dirs(ctx: Any, files: list) -> dict:
    """``{input: output folder}`` from the owner's resolver (empty when it has none)."""
    resolve = getattr(ctx.owner, "_resolve_translation_output_dir", None)
    mapping: dict = {}
    if callable(resolve):
        for path in files:
            try:
                folder = resolve(path)
            except Exception:
                folder = None
            if folder:
                mapping[path] = os.path.abspath(os.fspath(folder))
    if mapping:
        ctx.set_output_dirs(mapping)
    return mapping


#: The worker reported an unfinished run (``_translation_worker`` returned False) without a Stop.
NOT_COMPLETED = "The translation did not complete (see the log)."


def run_translation(ctx: Any, files: list) -> dict:
    """The shared prepare + worker pair; returns ``{"ok", "outputs", "error"}``.

    ``_translation_worker`` returns the run's outcome (``run_translation_direct``'s result,
    False when it ended early); False after a Stop is a stopped run (``ok`` None: Stopped),
    otherwise a failure. A Stop tapped before the set-up's "Reset stop flags" block (the job
    was starting) survives only in the job's own latch (``ctx.stop_requested()``, which the
    reset does not touch), so the worker is not started then: on desktop the set-up runs
    inside the Run click and no Stop can come before the reset.
    """
    owner = ctx.owner
    prepare = owner_method(owner, "_prepare_translation_run")
    worker = owner_method(owner, "_translation_worker")
    mapping = record_output_dirs(ctx, files)
    ctx.phase("Translating")
    request = prepare(files)
    if request is None or request is False:
        if ctx.stop_requested():
            return {"ok": None, "outputs": []}
        return {"ok": False, "outputs": [], "error": "The translation did not start (see the log)."}
    request_ok, _outs, request_error = result_fields(request) if isinstance(request, dict) else (None, [], None)
    if request_ok is False:
        return {"ok": False, "outputs": [], "error": request_error or "The translation did not start (see the log)."}
    if ctx.stop_requested():
        ctx.log("⏹️ Translation stopped before it started")
        return {"ok": None, "outputs": []}
    outcome = worker(request)
    ok, outputs, error = result_fields(outcome)
    if ok is False and ctx.stop_requested():
        ok, error = None, None  # stopped: the run ends as Stopped (Resume continues it)
    elif ok is False and not error:
        error = NOT_COMPLETED
    folders = list(mapping.values()) or [ctx.output_dir]
    for path in compiled_outputs(folders):
        if path not in outputs:
            outputs.append(path)
    return {"ok": ok, "outputs": outputs, "error": error}


def run(ctx: Any) -> dict:
    files = check_inputs(ctx.inputs)
    return run_translation(ctx, files)


KINDS = {
    "translate": {"verb": "Translating", "icon": "TRANSLATE", "stop_kind": "translation", "run": run},
}
