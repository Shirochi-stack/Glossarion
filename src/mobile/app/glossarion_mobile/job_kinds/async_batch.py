"""ASYNC_BATCH: the desktop Async Processing dialog's actions (Tools › Async batch, UI_SPEC §4.7).

Every action runs ``async_batch_core.HeadlessAsyncBatch`` - the ``AsyncProcessingDialog``
workflow moved into ``AsyncBatchJobMixin`` - on the job's ``HeadlessOwner`` (``self.gui``:
model, API key, the run environment of ``_prepare_environment_variables``) with the job host
attached:

* questions ("Possibly Unsupported", "Gemini Batch API", "Start Async Processing", cancel
  confirmations, ...) go to ``host.ask('async_batch_question', level=, title=, text=,
  buttons=, default=)`` - the Async batch screen answers them with a dialog (unanswered: the
  core's safe default, "no");
* notices ("Batch Submitted", "Job Status", errors) are ``host.emit('async_batch_message')``
  events, the cost text ``async_batch_cost`` and the job list ``async_batch_jobs``; all of
  them are also recorded on the job result for a screen opened later.

Actions (``params["action"]``): ``submit`` (Start Async Processing for the input file;
"Wait for completion" polls on the job until the batch ends or Stop), ``estimate`` (Estimate
Cost Only), ``refresh`` (check every pending / processing batch), ``check`` (Check Status of
one batch), ``retrieve`` (Retrieve Results: download completed batches and write their
chapters and progress into the output folders), ``cancel``, ``delete`` (local list only) and
``clear_completed``. ``job_ids`` selects the batches; ``jobs_file`` is the job list
(default ``async_batch_core.default_jobs_file()``: ``async_jobs.json`` in the app data
folder). None of these is resumable: a submission must never be repeated by a Resume.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

from glossarion_mobile.job_kinds import compiled_outputs

__all__ = ["ACTIONS", "KINDS", "QUESTION_KIND", "run"]

ACTIONS = ("submit", "estimate", "refresh", "check", "retrieve", "cancel", "delete", "clear_completed")
#: ``JobService.on_question`` kind of the dialog questions.
QUESTION_KIND = "async_batch_question"
_PHASES = {
    "submit": "Submitting batch", "estimate": "Estimating cost", "refresh": "Checking batches",
    "check": "Checking status", "retrieve": "Retrieving results", "cancel": "Cancelling",
    "delete": "Deleting", "clear_completed": "Clearing completed",
}


def _core() -> Any:
    from glossarion_mobile.services.jobs import JobError

    try:
        import async_batch_core
    except Exception as exc:  # pragma: no cover - bundle without the async core
        raise JobError(f"The shared async batch core (async_batch_core) is not in this build ({exc}).") from exc
    if not hasattr(async_batch_core, "HeadlessAsyncBatch"):
        raise JobError("This build's async_batch_core has no HeadlessAsyncBatch.")
    return async_batch_core


def _jsonable_messages(messages: Any) -> list:
    out = []
    for message in messages or ():
        if isinstance(message, Mapping):
            out.append({k: str(v) for k, v in message.items() if k in ("level", "title", "text", "answer")})
    return out


def run(ctx: Any) -> dict:
    from glossarion_mobile.services.jobs import JobError

    params = ctx.params or {}
    action = str(params.get("action") or "")
    if action not in ACTIONS:
        raise JobError(f"Unknown async batch action: {action or '(none)'}")
    core = _core()
    owner = ctx.owner
    source = None
    if action in ("submit", "estimate"):
        from glossarion_mobile.job_kinds.translate import check_inputs

        source = check_inputs(ctx.inputs)[0]
        owner.file_path = source  # the dialog reads the main window's selected file
    jobs_file = str(params.get("jobs_file") or "") or core.default_jobs_file()
    os.makedirs(os.path.dirname(os.path.abspath(jobs_file)), exist_ok=True)
    batch = core.HeadlessAsyncBatch(owner, host=ctx.host, jobs_file=jobs_file, answers=params.get("answers"))
    job_ids = [str(j) for j in (params.get("job_ids") or ()) if j]
    ctx.phase(_PHASES[action])
    result: dict = {"async_action": action}
    outputs: list = []
    if action == "submit":
        submitted = batch.submit()
        if submitted is not None:
            result["async_job_id"] = str(getattr(submitted, "job_id", "") or "")
    elif action == "estimate":
        result["async_cost"] = str(batch.estimate() or "")
    elif action == "refresh":
        batch.refresh_statuses()
    elif action == "check":
        if not job_ids:
            raise JobError("Please select a job to check status")
        batch.check_status(job_ids[0])
    elif action == "retrieve":
        folders = [str(f) for f in batch.retrieve(job_ids) or ()]
        result["async_saved_dirs"] = folders
        if folders:
            ctx.set_output_dirs({folder: folder for folder in folders})
            outputs = compiled_outputs(folders)
    elif action == "cancel":
        batch.cancel(job_ids)
    elif action == "delete":
        batch.delete(job_ids)
    elif action == "clear_completed":
        batch.clear_completed()
    messages = _jsonable_messages(getattr(batch, "messages", ()))
    result["async_messages"] = messages
    result["async_cost_info"] = str(getattr(batch, "cost_info", "") or "")
    ctx.set_result(**result)
    if outputs:
        ctx.add_outputs(outputs)
    errors = [m for m in messages if m.get("level") == "critical"]
    if ctx.stop_requested():
        return {"ok": None, "outputs": outputs}
    if errors and action in ("submit", "estimate", "retrieve"):
        return {"ok": False, "outputs": outputs, "error": errors[-1].get("text") or "Async batch action failed"}
    return {"ok": True, "outputs": outputs}


KINDS = {
    "async_batch": {"verb": "Async batch", "icon": "SCHEDULE_SEND", "stop_kind": "translation", "run": run,
                    "resumable": False},
}
