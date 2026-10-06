"""METADATA: translate the metadata (book title + selected fields) of one or more EPUBs.

The desktop "Translate Metadata" (Library selection bar, Book Details, Headers & metadata)
calls ``TranslatorGUI.start_metadata_translation(paths, output_roots=)``, which validates
the EPUBs, sets the owner's ``_metadata_output_roots`` / ``selected_files`` /
``_metadata_only_run`` (single-chapter and forced-stream flags off) and starts the regular
translation thread. The shared pipeline (``translation_pipeline``) then runs the metadata
phase only: one EPUB through ``_process_text_file`` with ``METADATA_ONLY=1``
(``run_env._metadata_only_environment_for_file``: OUTPUT_DIRECTORY from
``_metadata_output_roots``), several EPUBs through ``_run_parallel_metadata_files`` when
batch translation is on (thread pool, one ``metadata_translation_worker`` job per book).
The translation mode (together / metadata_separate / parallel), the prompts and the
fields (``translate_metadata_fields``, per-EPUB ``_per_epub`` selections) come from the
job's config snapshot, as the desktop reads them from its config.

This adapter does what ``start_metadata_translation`` does before ``run_translation_thread``
(same validation, normalisation and log lines) and then runs the shared prepare + worker pair
(``translate.run_translation``); the pipeline resets ``_metadata_only_run`` /
``_metadata_output_roots`` when the run ends.

Inputs: the raw EPUBs. params: ``output_roots`` - ``{source: output root}`` (the folder that
holds the book's output folder, desktop shape), or a list of book output *folders* aligned
with the inputs (``LibraryService.metadata_spec``; their parent folders become the roots).
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from glossarion_mobile.job_kinds.translate import run_translation

__all__ = ["KINDS", "output_roots_for", "run", "valid_epubs"]


def valid_epubs(paths: Sequence[Any]) -> list:
    """Existing ``.epub`` files, absolute, de-duplicated by normcase (desktop order kept)."""
    valid: list = []
    seen: set = set()
    for path in paths or ():
        if not path:
            continue
        text = os.fspath(path)
        if not os.path.isfile(text) or not text.lower().endswith(".epub"):
            continue
        absolute = os.path.abspath(text)
        key = os.path.normcase(absolute)
        if key in seen:
            continue
        seen.add(key)
        valid.append(absolute)
    return valid


def output_roots_for(raw: Any, inputs: Sequence[str]) -> dict:
    """``{normcase(abspath(source)): abspath(root)}`` (the owner's ``_metadata_output_roots``)."""
    pairs: list = []
    if isinstance(raw, Mapping):
        pairs = [(source, root) for source, root in raw.items()]
    elif isinstance(raw, (list, tuple)):
        if len(raw) == len(inputs):  # LibraryService.metadata_spec: book output folders
            pairs = [(source, os.path.dirname(os.path.abspath(os.fspath(folder))) if folder else "")
                     for source, folder in zip(inputs, raw)]
    normalized: dict = {}
    for path, root in pairs:
        if not path or not root:
            continue
        normalized[os.path.normcase(os.path.abspath(os.fspath(path)))] = os.path.abspath(os.fspath(root))
    return normalized


def run(ctx: Any) -> dict:
    from glossarion_mobile.services.jobs import JobError

    valid = valid_epubs(ctx.inputs)
    if not valid:
        ctx.log("❌ Metadata translation: no valid EPUB sources were found")
        raise JobError("Metadata translation: no valid EPUB sources were found")
    owner = ctx.owner
    owner._metadata_output_roots = output_roots_for((ctx.params or {}).get("output_roots"), ctx.inputs)
    owner.selected_files = list(valid)
    owner.current_file_index = 0
    owner._metadata_only_run = True
    owner._single_chapter_filter = None
    owner._force_stream_all = False
    if len(valid) == 1:
        ctx.log(f"🌐 Queued metadata translation: {os.path.basename(valid[0])}")
    else:
        batch = bool(getattr(owner, "batch_translation_var", (getattr(owner, "config", {}) or {}).get(
            "batch_translation", True)))
        run_mode = "thread-pool batch" if batch else "sequential"
        ctx.log(f"🌐 Queued metadata translation for {len(valid)} EPUBs ({run_mode})")
    ctx.phase("Translating metadata")
    try:
        return run_translation(ctx, list(valid))
    finally:
        # The pipeline resets both at the end of a run; a run that never started must not leak them.
        owner._metadata_only_run = False
        owner._metadata_output_roots = {}
        os.environ.pop("METADATA_ONLY", None)


KINDS = {
    "metadata": {"verb": "Translating metadata", "icon": "LABEL", "stop_kind": "translation", "run": run},
}
