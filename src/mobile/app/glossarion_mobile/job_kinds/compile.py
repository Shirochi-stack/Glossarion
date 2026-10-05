"""COMPILE_EPUB / COMPILE_PDF: compile a translation output folder.

``owner._run_epub_compile(folder)`` / ``owner._run_pdf_compile(folder)``
(``text_jobs.TextJobsMixin``, the desktop compile runners) return a
``CompileResult``; its path becomes the job output (Share / Save / Reader).

The folder is ``params["folder"]``, else the first input: a folder is used as
is, a source file (EPUB/TXT/PDF) is mapped to its output folder with the
owner's ``_resolve_translation_output_dir``, as the desktop does for the file
in the input field.
"""

from __future__ import annotations

import os
from typing import Any

from glossarion_mobile.job_kinds import compiled_outputs, owner_method, result_fields

__all__ = ["KINDS", "resolve_folder", "run_epub", "run_pdf"]


def resolve_folder(ctx: Any) -> str:
    from glossarion_mobile.services.jobs import JobError

    target = ctx.params.get("folder") or (ctx.inputs[0] if ctx.inputs else "")
    if not target:
        raise JobError("Nothing to compile: no output folder.")
    target = os.path.abspath(os.fspath(target))
    if os.path.isfile(target):
        resolve = getattr(ctx.owner, "_resolve_translation_output_dir", None)
        if callable(resolve):
            target = os.path.abspath(os.fspath(resolve(target)))
    if not os.path.isdir(target):
        raise JobError(f"Folder not found: {os.path.basename(target) or target}")
    return target


def _compile(ctx: Any, method: str, phase: str, extension: str) -> dict:
    folder = resolve_folder(ctx)
    runner = owner_method(ctx.owner, method)
    ctx.set_output_dir(folder)
    ctx.phase(phase)
    ok, outputs, error = result_fields(runner(folder))
    if not outputs and ok is not False:
        outputs = [p for p in compiled_outputs([folder]) if p.lower().endswith(extension)]
    return {"ok": ok, "outputs": outputs, "error": error}


def run_epub(ctx: Any) -> dict:
    return _compile(ctx, "_run_epub_compile", "Compiling EPUB…", ".epub")


def run_pdf(ctx: Any) -> dict:
    return _compile(ctx, "_run_pdf_compile", "Compiling PDF…", ".pdf")


KINDS = {
    "compile_epub": {"verb": "Compiling EPUB", "icon": "MENU_BOOK", "stop_kind": "translation", "run": run_epub},
    "compile_pdf": {"verb": "Compiling PDF", "icon": "PICTURE_AS_PDF", "stop_kind": "translation", "run": run_pdf},
}
