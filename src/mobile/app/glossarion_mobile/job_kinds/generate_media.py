"""GENERATE_MEDIA: "Generate from prompt (no input)" in the chat (UI_SPEC §2.6).

The desktop generative-only run: with an image / video / audio output mode (or an image /
video generation model) and no input file, ``run_translation_direct`` skips the file loop and
calls ``_run_generative_prompt_mode()`` (``image_job.ImageJobMixin``, moved verbatim), which
sends one request built from the prompt editor. On mobile the prompt is the chat composer text
(decided, Appendix C): this adapter puts it in the owner attribute the shared runner reads
first (``image_job.GENERATIVE_PROMPT_ATTR``) and then runs the same desktop composition a chat
send uses (``job_kinds.direct_text``):

1. ``headless_owner.DirectTextRunOptions`` with ``selected_files = [GENERATIVE_MODE_SENTINEL]``
   and the chat's output mode (Image / Video / Audio);
2. ``direct_text_store.apply_direct_text_run_environment``: ``OUTPUT_DIRECTORY`` = the run root,
   so the client's ``generated_media_*`` / Veo / Lyria / TTS files land there;
3. ``owner._prepare_translation_run([sentinel])`` + ``owner._translation_worker(request)``.

The chat then finishes the run with ``ChatStore.finish_run`` (``ChatRuns``): the shared
``_discover_generated_output`` finds the media in the run root, ``_persist_output_folder``
copies it into the chat folder as ``Direct Text N.<ext>`` and
``_promote_generated_media_reference`` points the response's ``[GENERATED_*]`` marker at it,
exactly like a desktop Direct Text media response.

params (``ChatRuns.generate``): ``prompt``, ``options`` (DirectTextRunOptions fields),
``output_root``, ``run`` (+ ``request_number`` / ``model`` for the job's request stream).
"""

from __future__ import annotations

from typing import Any

from glossarion_mobile.job_kinds import owner_method, result_fields
from glossarion_mobile.job_kinds.direct_text import apply_run_environment, build_options, run_result

__all__ = ["KINDS", "apply_prompt", "generative_names", "run"]


def generative_names() -> tuple:
    """``(sentinel, prompt attribute)`` from the shared ``image_job`` module."""
    try:
        import image_job  # shared (U7)
    except ImportError as exc:
        from glossarion_mobile.services.jobs import JobError

        raise JobError(f"The shared module 'image_job' is not in this build ({exc}).") from exc
    return image_job.GENERATIVE_MODE_SENTINEL, image_job.GENERATIVE_PROMPT_ATTR


def apply_prompt(owner: Any, prompt: str) -> str:
    """Set the generation prompt where the shared runner reads it first; returns the attribute name."""
    _sentinel, attribute = generative_names()
    setattr(owner, attribute, str(prompt or "").strip())
    return attribute


def run(ctx: Any) -> dict:
    from glossarion_mobile.services.jobs import JobError

    params = ctx.params
    prompt = str(params.get("prompt") or "").strip()
    if not prompt:
        raise JobError("Type a prompt in the composer first")
    owner = ctx.owner
    prepare = owner_method(owner, "_prepare_translation_run")
    worker = owner_method(owner, "_translation_worker")
    owner_method(owner, "_run_generative_prompt_mode")
    sentinel, _attribute = generative_names()
    options = build_options({**params, "options": {**dict(params.get("options") or {}), "selected_files": [sentinel]}},
                            sentinel)
    options.apply_to(owner)
    apply_run_environment(owner, params)
    apply_prompt(owner, prompt)
    try:
        ctx.phase("Generating")
        request = prepare([sentinel])
        if request is None or request is False:
            if ctx.stop_requested():
                return {"ok": None, "outputs": []}
            return {"ok": False, "outputs": [], "error": "The generation did not start (see the log)."}
        if ctx.stop_requested():
            ctx.log("⏹️ Generation stopped before it started")
            return {"ok": None, "outputs": []}
        outcome = worker(request)
        ok, outputs, error = result_fields(outcome)
        if ok is False and ctx.stop_requested():
            ok, error = None, None
        elif ok is False and not error:
            error = "The generation did not complete (see the log)."
        return {"ok": ok, "outputs": outputs, "error": error}
    finally:
        try:
            ctx.set_result(**run_result(owner))
        except Exception:
            pass


KINDS = {
    "generate_media": {"verb": "Generating", "icon": "AUTO_AWESOME", "stop_kind": "translation", "run": run},
}
