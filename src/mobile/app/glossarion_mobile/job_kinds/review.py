"""REVIEW: the desktop Review Generator run (Tools › Review generator, UI_SPEC §4.8).

The run is ``review_generator``'s orchestration moved out of ``review_dialog``
(``_on_start_review`` / ``_on_generate_all``) on the job's ``HeadlessOwner`` (the dialog's
``translator_gui``: API key, model, endpoint, temperature, the multi-key config, the token
limit, the output language, batch settings, the streaming toggle):

* ``single`` / ``volume`` - "🚀 Start Review": ``run_review_session(owner, prompt=…,
  spoiler_mode=…, chunk_mode=…, wrap_chunks=…, final_review_prompt=…, file_path=…,
  volume_paths=…, volume_mode=…, stop_check_fn=…)`` (its log goes through the owner's
  ``append_log`` with the "[Review]" prefix, i.e. the job log); Volume mode reviews the
  inputs, in order, as one book and saves the combined review in every volume;
* ``all`` - "📚 Review all Files": the dialog's sequence ``review_run_params`` ->
  ``review_all_batch_size`` -> ``reset_review_stop_flags`` -> ``apply_review_streaming_env``
  -> ``run_all_reviews`` (one review per input, in parallel when batch translation is on;
  its queue messages become job log lines).

The prompts are the dialog's: ``review_system_prompt`` / ``review_final_prompt`` of the
config snapshot, else ``DEFAULT_REVIEW_PROMPT`` / ``DEFAULT_FINAL_REVIEW_PROMPT``
(``_load_saved_prompt``). params: ``mode``, ``spoiler_mode`` (50/50 split), ``chunk_mode``
(Full Review Mode), ``wrap_chunks``. Outputs: the review files that exist afterwards
(``review_paths_for``).
"""

from __future__ import annotations

import os
from typing import Any

from glossarion_mobile.job_kinds.translate import check_inputs

__all__ = ["KINDS", "MODES", "review_prompts", "run"]

MODES = ("single", "volume", "all")


def review_prompts(review_generator: Any, config: Any) -> tuple:
    """The dialog's prompts (``_load_saved_prompt``): saved text, else the shared defaults."""
    config = config if isinstance(config, dict) else {}
    prompt = config.get("review_system_prompt", "") or getattr(review_generator, "DEFAULT_REVIEW_PROMPT", "")
    final = config.get("review_final_prompt", "") or getattr(review_generator, "DEFAULT_FINAL_REVIEW_PROMPT", "")
    return str(prompt), str(final)


def _existing(paths: Any) -> list:
    return [str(p) for p in paths or () if p and os.path.isfile(str(p))]


def run(ctx: Any) -> dict:
    from glossarion_mobile.services.jobs import JobError

    params = ctx.params or {}
    mode = str(params.get("mode") or "single")
    if mode not in MODES:
        raise JobError(f"Unknown review mode: {mode}")
    files = check_inputs(ctx.inputs)
    try:
        import review_generator as rg
    except Exception as exc:  # pragma: no cover - bundle without the review generator
        raise JobError(f"The review generator (review_generator) is not in this build ({exc}).") from exc
    owner = ctx.owner
    config = ctx.config or getattr(owner, "config", {}) or {}
    prompt, final_prompt = review_prompts(rg, config)
    spoiler_mode = bool(params.get("spoiler_mode", False))
    chunk_mode = bool(params.get("chunk_mode", False))
    wrap_chunks = bool(params.get("wrap_chunks", True))
    stop = ctx.stop_requested
    paths_for = getattr(rg, "review_paths_for", None)
    if mode == "all" and len(files) > 1:
        needed = ("review_run_params", "review_all_batch_size", "reset_review_stop_flags",
                  "apply_review_streaming_env", "run_all_reviews")
        if not all(callable(getattr(rg, name, None)) for name in needed):
            raise JobError("This build's review_generator has no shared 'Review all Files' runner.")
        ctx.phase(f"Reviewing {len(files)} files")
        run_params = rg.review_run_params(owner, prompt, spoiler_mode, final_prompt)
        batch_size = rg.review_all_batch_size(owner)
        rg.reset_review_stop_flags()
        rg.apply_review_streaming_env(owner)

        def put(item: Any) -> None:
            kind, data = (tuple(item) + (None, None))[:2]
            if kind == "log":
                ctx.log(data)

        completed, errors = rg.run_all_reviews(run_params, list(files), chunk_mode=chunk_mode,
                                               wrap_chunks=wrap_chunks, batch_size=batch_size, put=put,
                                               stop_check=stop)
        outputs: list = []
        if callable(paths_for):
            for path in files:
                outputs.extend(_existing(paths_for(path, [], False, config)))
        ctx.set_result(review_mode=mode, review_paths=outputs, review_completed=int(completed or 0),
                       review_errors=int(errors or 0))
        if outputs:
            ctx.add_outputs(outputs)
        if stop():
            return {"ok": None, "outputs": outputs}
        if errors and not outputs:
            return {"ok": False, "outputs": [], "error": f"{errors} review(s) failed (see the log)."}
        return {"ok": True, "outputs": outputs}
    session = getattr(rg, "run_review_session", None)
    if not callable(session):
        raise JobError("This build's review_generator has no shared review runner (run_review_session).")
    volume = mode == "volume"
    ctx.phase("Reviewing volumes" if volume else "Reviewing")
    text = session(owner, prompt=prompt, spoiler_mode=spoiler_mode, chunk_mode=chunk_mode, wrap_chunks=wrap_chunks,
                   final_review_prompt=final_prompt, file_path=files[0], volume_paths=list(files) if volume else None,
                   volume_mode=volume, stop_check_fn=stop)
    outputs = _existing(paths_for(files[0], list(files) if volume else [], volume, config)) if callable(paths_for) else []
    if outputs:
        ctx.add_outputs(outputs)
    ctx.set_result(review_mode=mode, review_paths=outputs)
    if stop():
        return {"ok": None, "outputs": outputs}
    if not text:
        return {"ok": False, "outputs": outputs, "error": "No review was generated (see the log)."}
    return {"ok": True, "outputs": outputs}


KINDS = {
    "review": {"verb": "Reviewing", "icon": "RATE_REVIEW", "stop_kind": "translation", "run": run},
}
