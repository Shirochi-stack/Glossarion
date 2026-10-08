"""SINGLE_CHAPTER: translate one chapter of an EPUB (desktop ``start_single_chapter_translation``).

The Reader's "Translate this chapter" (and the Book page's chapter rows) run
this kind. Like the desktop method (translator_gui ``start_single_chapter_translation``)
it sets the owner's ``_single_chapter_filter`` to the chapter's file name and
``_force_stream_all`` (every streaming toggle on, so the live view gets the
streamed text), then runs the same shared prepare + worker pair as a full
translation (``translate.run_translation``): the extraction pulls only that
chapter, the auto-glossary pass is skipped, and the pipeline resets both
flags at the end of the run (``translation_pipeline``).

Params: ``chapter_file`` (the chapter's source file name; required) and
``force_stream_all`` (default False). Input: the raw EPUB.

The desktop always forces streaming for the live view; on mobile ``force_stream_all`` is honoured
only while the Streaming switch is on (owner 2026-10-08: off stops streaming everywhere, the
Reader's live translation included; ``direct_text.streaming_allowed`` of the job's config). The
chapter then arrives when it is done.
"""

from __future__ import annotations

import os
from typing import Any

from glossarion_mobile.job_kinds.direct_text import streaming_allowed
from glossarion_mobile.job_kinds.translate import check_inputs, run_translation

__all__ = ["KINDS", "run"]


def run(ctx: Any) -> dict:
    from glossarion_mobile.services.jobs import JobError

    files = check_inputs(ctx.inputs)
    if len(files) != 1 or not files[0].lower().endswith(".epub"):
        raise JobError("Single-chapter translation needs one EPUB file.")
    chapter = os.path.basename(str((ctx.params or {}).get("chapter_file") or ""))
    if not chapter:
        raise JobError("Single-chapter translate: no chapter filename given")
    owner = ctx.owner
    owner._single_chapter_filter = chapter
    force = bool((ctx.params or {}).get("force_stream_all", False))
    if force and not streaming_allowed(owner):
        force = False
        ctx.log("🛰️ Streaming is off (Settings › Response handling & retries): the chapter appears when it is done")
    owner._force_stream_all = force
    ctx.log(f"🎯 Queued single-chapter translation: {chapter} ({os.path.basename(files[0])})")
    try:
        return run_translation(ctx, files)
    finally:
        # The pipeline clears both at the end of a run; a run that never started must not leak them.
        owner._single_chapter_filter = None
        owner._force_stream_all = False


KINDS = {
    "single_chapter": {"verb": "Translating", "icon": "TRANSLATE", "stop_kind": "translation", "run": run},
}
