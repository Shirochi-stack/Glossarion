"""DIRECT_TEXT: one Direct Text send (the chat) through the translation pipeline.

Desktop ``_InputOutputDialog._start_translation`` is "only a front end": it
points the owner at one input, applies ``DirectTextRunOptions``, exports the
Direct Text run environment and calls ``run_translation_thread``. This adapter
does the same on the job's ``HeadlessOwner``, with the same shared code:

1. ``headless_owner.DirectTextRunOptions(**params["options"]).apply_to(owner)``
   (``selected_files`` defaults to the run input);
2. ``direct_text_store.apply_direct_text_run_environment(owner, output_root,
   is_attachment)``: ``OUTPUT_DIRECTORY``/``OUTPUT_DIR`` = the run's output root,
   ``DIRECT_TEXT_*``, ``ORDER_BATCH_REQUESTS_BY_SPINE``, then the owner's
   ``_apply_forced_streaming_environment`` / ``_apply_direct_text_runtime_environment``
   (the dialog's block, in its order);
3. ``owner._prepare_translation_run([input])`` + ``owner._translation_worker``;
4. ``ctx.set_result(...)``: the glossary the run used (the run's ``MANUAL_GLOSSARY``,
   else the owner's ``manual_glossary_path``), the run's ``MANUAL_GLOSSARY`` itself
   (``run_env``), the Direct Text no-glossary flag and the model - what the dialog's
   ``_finish_translation`` reads from the live owner and environment, recorded before the
   job scope restores them (the chat finishes on another thread, maybe while the next
   queued job runs with its own environment).

The job runs inside ``job_runner.job_process_state``, so every variable set here is
restored when the job ends (the dialog's ``_restore_run_context``). Persisting the
result into the chat (``direct_text_store.ChatStore.finish_run``) is the chat's job,
on the terminal transition.

params: ``input_path`` (or ``spec.inputs[0]``), ``options`` (DirectTextRunOptions
fields), ``output_root``, ``is_attachment`` (+ ``request_number`` / ``model`` for the
job's request stream).
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Mapping

from glossarion_mobile.job_kinds import owner_method
from glossarion_mobile.job_kinds.translate import check_inputs, run_translation

__all__ = ["FINISH_ENV_KEYS", "KINDS", "apply_run_environment", "build_options", "run", "run_result"]


def build_options(params: Mapping[str, Any], input_path: str) -> Any:
    """``headless_owner.DirectTextRunOptions`` from ``params["options"]`` (unknown keys ignored)."""
    from headless_owner import DirectTextRunOptions  # shared (U2)

    raw = dict(params.get("options") or {})
    names = {f.name for f in dataclasses.fields(DirectTextRunOptions)}
    values = {k: v for k, v in raw.items() if k in names}
    if not values.get("selected_files"):
        values["selected_files"] = [input_path]
    return DirectTextRunOptions(**values)


def apply_run_environment(owner: Any, params: Mapping[str, Any]) -> None:
    """The dialog's run environment (``direct_text_store.apply_direct_text_run_environment``)."""
    from glossarion_mobile.services.jobs import JobError

    output_root = params.get("output_root")
    if not output_root:
        raise JobError("The Direct Text run has no output folder.")
    try:
        from direct_text_store import apply_direct_text_run_environment  # shared (U3)
    except ImportError as exc:
        raise JobError(f"The shared module 'direct_text_store' is not in this build ({exc}).") from exc
    apply_direct_text_run_environment(owner, os.fspath(output_root), bool(params.get("is_attachment")))


#: Run environment values the dialog's ``_finish_translation`` reads live (``_effective_run_glossary_path``);
#: recorded on the job thread so the chat finishes the run with the job's values, never another job's.
FINISH_ENV_KEYS = ("MANUAL_GLOSSARY",)


def run_result(owner: Any) -> dict:
    """What ``ChatStore.finish_run`` needs from the live run (see the module docstring)."""
    candidates = (os.environ.get("MANUAL_GLOSSARY", ""), getattr(owner, "manual_glossary_path", "") or "")
    glossary_path = next((os.path.abspath(str(c)) for c in candidates if str(c or "").strip()
                          and os.path.isfile(str(c))), "")
    result: dict = {"glossary_path": glossary_path,
                    "run_env": {key: os.environ.get(key, "") for key in FINISH_ENV_KEYS}}
    force_none = getattr(owner, "_direct_text_force_no_glossary", None)
    if force_none is not None:
        result["force_no_glossary"] = bool(force_none)
    model = os.environ.get("MODEL") or str(getattr(owner, "model_var", "") or "")
    if model:
        result["model"] = model
    return result


def run(ctx: Any) -> dict:
    params = ctx.params
    input_path = params.get("input_path") or (ctx.inputs[0] if ctx.inputs else "")
    files = check_inputs([input_path])
    owner = ctx.owner
    owner_method(owner, "_prepare_translation_run")  # fail early when the pipeline is missing
    options = build_options(params, files[0])
    options.apply_to(owner)
    apply_run_environment(owner, params)
    try:
        return run_translation(ctx, files)
    finally:
        try:
            ctx.set_result(**run_result(owner))
        except Exception:
            pass


KINDS = {
    "direct_text": {"verb": "Translating", "icon": "CHAT_BUBBLE_OUTLINE", "stop_kind": "translation", "run": run},
}
