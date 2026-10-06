"""COMPILE_EPUB / COMPILE_PDF: compile a translation output folder.

``owner._run_epub_compile(folder)`` / ``owner._run_pdf_compile(folder)``
(``text_jobs.TextJobsMixin``, the desktop compile runners) return a
``CompileResult``; its path becomes the job output (Share / Save / Reader).

The folder is ``params["folder"]``, else the first input: a folder is used as
is, a source file (EPUB/TXT/PDF) is mapped to its output folder with the
owner's ``_resolve_translation_output_dir``, as the desktop does for the file
in the input field.

``params["pdf_after_epub"]`` (Compile PDF of an EPUB workspace: the EPUB compile with the
``enable_pdf_output`` override) adds the PDFs the run wrote or rewrote in the folder to the
outputs after the EPUB, so the Result card offers them.

U6 adds two Converter actions as jobs (UI_SPEC §4.5), so they never run while a
translation writes the same folder and their backend prints land in the job log:

* ``validate_epub`` - Validate EPUB structure: ``TransateKRtoEN.validate_epub_structure``
  + ``check_epub_readiness`` per folder, classified with the desktop's result lines
  (``other_settings.validate_epub_structure_gui``); ``result["validation"]`` holds the
  lines and ``result["all_passed"]`` the verdict (failures do not fail the job).
* ``rename_outputs`` - Rename Files: ``output_naming._rename_output_files_for_retain(owner,
  retain, output_dir=folder)`` (the shared helper the desktop button calls), ``retain`` =
  ``params["retain"]`` else the config's ``retain_source_extension``; the outcome text is
  the desktop button's ("✅ N files renamed" / "📁 No OPF package found" / "📄 No files to
  rename" / "⚠️ Rename failed") in ``result["rename"]``.
"""

from __future__ import annotations

import os
from typing import Any

from glossarion_mobile.job_kinds import compiled_outputs, owner_method, result_fields

__all__ = ["KINDS", "rename_message", "resolve_folder", "resolve_folders", "run_epub", "run_pdf", "run_rename",
           "run_validate", "validation_line"]


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


def _pdf_mtimes(folder: str) -> dict:
    """{path: mtime} of the compiled PDFs at the top of ``folder``."""
    found: dict = {}
    for path in compiled_outputs([folder]):
        if path.lower().endswith(".pdf"):
            try:
                found[path] = os.path.getmtime(path)
            except OSError:
                pass
    return found


def _compile(ctx: Any, method: str, phase: str, extension: str) -> dict:
    folder = resolve_folder(ctx)
    runner = owner_method(ctx.owner, method)
    ctx.set_output_dir(folder)
    ctx.phase(phase)
    # "Compile PDF" of an EPUB workspace: the EPUB compile with "Create PDF after EPUB"; its
    # CompileResult names only the EPUB, so the PDFs the run wrote are added as outputs too.
    pdf_after = extension == ".epub" and bool(ctx.params.get("pdf_after_epub"))
    pdfs_before = _pdf_mtimes(folder) if pdf_after else {}
    ok, outputs, error = result_fields(runner(folder))
    if not outputs and ok is not False:
        outputs = [p for p in compiled_outputs([folder]) if p.lower().endswith(extension)]
    if pdf_after and ok is not False:
        for path, mtime in _pdf_mtimes(folder).items():
            if pdfs_before.get(path) != mtime and path not in outputs:
                outputs.append(path)
    return {"ok": ok, "outputs": outputs, "error": error}


def run_epub(ctx: Any) -> dict:
    return _compile(ctx, "_run_epub_compile", "Compiling EPUB…", ".epub")


def run_pdf(ctx: Any) -> dict:
    return _compile(ctx, "_run_pdf_compile", "Compiling PDF…", ".pdf")


def resolve_folders(ctx: Any) -> list:
    """Every target folder: ``params["folders"]``, else ``params["folder"]`` / the inputs."""
    raw = ctx.params.get("folders")
    if not raw:
        return [resolve_folder(ctx)]
    from glossarion_mobile.services.jobs import JobError

    folders: list = []
    for item in raw:
        if not item:
            continue
        folder = os.path.abspath(os.fspath(item))
        if os.path.isdir(folder) and folder not in folders:
            folders.append(folder)
    if not folders:
        raise JobError("Folder not found.")
    return folders


def validation_line(base: str, structure_ok: bool, readiness_ok: bool) -> tuple:
    """(log line, result line, passed) of one validated folder (desktop wording)."""
    if structure_ok and readiness_ok:
        return f"  ✅ {base}: PASSED", f"✅ {base}: All structure files present", True
    if structure_ok:
        return f"  ⚠️ {base}: Structure OK, some issues", f"⚠️ {base}: Structure OK, some issues found", False
    return f"  ❌ {base}: Missing critical files", f"❌ {base}: Missing critical EPUB files", False


def run_validate(ctx: Any) -> dict:
    folders = resolve_folders(ctx)
    ctx.phase("Validating EPUB structure")
    lines: list = []
    all_passed = True
    for folder in folders:
        base = os.path.basename(folder.rstrip("/\\"))
        ctx.log(f"🔍 Validating EPUB structure for: {base}")
        try:
            from TransateKRtoEN import check_epub_readiness, validate_epub_structure

            structure_ok = validate_epub_structure(folder)
            readiness_ok = check_epub_readiness(folder)
            log_line, result_line, passed = validation_line(base, bool(structure_ok), bool(readiness_ok))
            ctx.log(log_line)
        except Exception as exc:
            ctx.log(f"  ❌ Validation error for {base}: {exc}")
            result_line, passed = f"❌ {base}: {exc}", False
        lines.append(result_line)
        all_passed = all_passed and passed
    ctx.set_result(validation=lines, all_passed=bool(all_passed and lines))
    return {"ok": True, "outputs": []}


def rename_message(result: Any) -> str:
    """The desktop Rename Files button text for ``_rename_output_files_for_retain``'s result."""
    if result and result[0] == "renamed":
        return f"✅ {result[1]} files renamed"
    if result and result[0] == "no_opf":
        return "📁 No OPF package found"
    return "📄 No files to rename"


def run_rename(ctx: Any) -> dict:
    folder = resolve_folder(ctx)
    owner = ctx.owner
    retain = ctx.params.get("retain")
    if retain is None:
        retain = bool((getattr(owner, "config", {}) or {}).get("retain_source_extension", False))
    ctx.set_output_dir(folder)
    ctx.phase("Renaming files")
    try:
        from output_naming import _rename_output_files_for_retain

        message = rename_message(_rename_output_files_for_retain(owner, bool(retain), output_dir=folder))
    except Exception as exc:
        ctx.log(f"❌ Rename failed: {exc}")
        message = "⚠️ Rename failed"
    ctx.log(message)
    ctx.set_result(rename=message, retain=bool(retain))
    return {"ok": message != "⚠️ Rename failed", "outputs": [], "error": None if message != "⚠️ Rename failed"
            else "Rename failed (see the log)."}


KINDS = {
    "compile_epub": {"verb": "Compiling EPUB", "icon": "MENU_BOOK", "stop_kind": "translation", "run": run_epub},
    "compile_pdf": {"verb": "Compiling PDF", "icon": "PICTURE_AS_PDF", "stop_kind": "translation", "run": run_pdf},
    "validate_epub": {"verb": "Validating EPUB", "icon": "RULE", "stop_kind": "translation", "run": run_validate,
                      "resumable": False},
    "rename_outputs": {"verb": "Renaming files", "icon": "DRIVE_FILE_RENAME_OUTLINE", "stop_kind": "translation",
                       "run": run_rename, "resumable": False},
}
