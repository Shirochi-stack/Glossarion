"""EXTRACT_GLOSSARY: the desktop "Extract Glossary" run over the selected files.

``owner.run_glossary_extraction_direct()`` (``translation_pipeline.GlossaryPipelineMixin``)
iterates ``owner.selected_files`` and dispatches per type, exactly as on
desktop (EPUB/TXT/PDF through ``_extract_glossary_from_text_file``, image
folders through the image glossary path). The adapter only sets the selection
(the desktop file list) and records the output folders.

params: ``force_balanced_request_merging`` (bool, desktop default False).
"""

from __future__ import annotations

from typing import Any

from glossarion_mobile.job_kinds import owner_method
from glossarion_mobile.job_kinds.translate import check_inputs, record_output_dirs

__all__ = ["KINDS", "run"]


def run(ctx: Any) -> dict:
    files = check_inputs(ctx.inputs)
    owner = ctx.owner
    extract = owner_method(owner, "run_glossary_extraction_direct")
    owner.selected_files = list(files)
    record_output_dirs(ctx, files)
    ctx.phase("Generating glossary…")
    result = extract(force_balanced_request_merging=bool(ctx.params.get("force_balanced_request_merging", False)))
    if result is False and not ctx.stop_requested():
        return {"ok": False, "outputs": [], "error": "Glossary extraction stopped (see the log)."}
    return {"ok": None, "outputs": []}


KINDS = {
    "extract_glossary": {"verb": "Extracting glossary", "icon": "SPELLCHECK", "stop_kind": "glossary", "run": run},
}
