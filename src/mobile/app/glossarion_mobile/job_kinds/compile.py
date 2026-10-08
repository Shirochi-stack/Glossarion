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
  (``other_settings.validate_epub_structure_gui``; the loop is the shared
  ``output_tools_core.validate_epub_outputs``); ``result["validation"]`` holds the
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
           "run_validate"]


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


def run_validate(ctx: Any) -> dict:
    """Validate EPUB Structure for each output folder (``output_tools_core.validate_epub_outputs``).

    The desktop names each book after its EPUB and looks its folder up; here the folder is
    known, so each one is passed as ``<folder>/<folder name>.epub`` with the folder as its
    output folder (the log and result lines name the folder).
    """
    folders = resolve_folders(ctx)
    ctx.phase("Validating EPUB structure")
    from output_tools_core import validate_epub_outputs

    named = {os.path.join(folder, os.path.basename(folder.rstrip("/\\")) + ".epub"): folder for folder in folders}
    all_passed, lines = validate_epub_outputs(
        list(named),
        config=getattr(ctx.owner, "config", None),
        log=ctx.log,
        output_dir_for=named.get,
    )
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


def md_txt_message(kind: str, ok: int, fail: int) -> str:
    """The desktop Generate MD / Generate TXT button result (``_start_retro_md_txt_gen._finish``)."""
    return f"✅ {ok} done" if not fail else f"✅ {ok} / ⚠️ {fail}"


def run_md_txt(ctx: Any) -> dict:
    """Generate MD / Generate TXT (desktop Other Settings retroactive buttons): every root HTML file of the
    output folder through ``md_txt_sidecar_writer.generate_md_txt_for_output_dir`` (thread pool,
    existing sidecars overwritten), with the desktop log lines."""
    folders = resolve_folders(ctx)
    kind = str(ctx.params.get("format") or "md").lower()
    if kind not in ("md", "txt"):
        kind = "md"
    ctx.phase(f"Generating {kind.upper()}")
    from md_txt_sidecar_writer import generate_md_txt_for_output_dir

    ok = fail = total = 0
    for folder in folders:
        ctx.log(f"📝 [{kind.upper()}] {folder}")
        r_ok, r_fail, r_total = generate_md_txt_for_output_dir(
            folder, do_md=kind == "md", do_txt=kind == "txt", max_workers=4, log=ctx.log,
            should_stop=ctx.stop_requested)
        ok += r_ok
        fail += r_fail
        total += r_total
    ctx.log(f"✅ [{kind.upper()}] Generation complete: {ok} file(s)" + (f", {fail} failed" if fail else ""))
    ctx.set_result(md_txt={"format": kind, "ok": ok, "failed": fail, "total": total,
                           "message": md_txt_message(kind, ok, fail)})
    if ctx.stop_requested():
        return {"ok": None, "outputs": []}
    return {"ok": True, "outputs": []}


#: The desktop "Apply to Existing Outputs" confirmation (other_settings manual BR→P conversion).
BR_CONFIRM_TITLE = "Confirm Manual BR→P Conversion"
BR_CONFIRM_TEXT = ("This immediately replaces <br> boundaries inside paragraphs with separate <p> paragraphs in "
                   "each selected input's output folder.\n\nExample:\n<p>First<br/>Second</p>  →  "
                   "<p>First</p><p>Second</p>\n\nIt runs even when the checkbox is off and does not create an "
                   "automatic backup.")


def run_br_to_paragraphs(ctx: Any) -> dict:
    """Apply <br> → <p> to existing outputs (``html_output_utils.convert_br_in_output_folder`` on each
    folder's root HTML files; atomic replace, BOM kept); the audit goes to the result and the log
    in the desktop "[BR→P]" form."""
    folders = resolve_folders(ctx)
    ctx.phase("Converting <br> to <p>")
    from html_output_utils import convert_br_in_output_folder

    audits: list = []
    failed_inputs = 0
    for folder in folders:
        name = os.path.basename(folder.rstrip("/\\"))
        try:
            audit = convert_br_in_output_folder(folder)
        except Exception as exc:
            failed_inputs += 1
            ctx.log(f"⚠️ [BR→P] {name}: {exc}")
            continue
        failed = int(audit.get("failed") or 0)
        if failed:
            failed_inputs += 1
        ctx.log(f"{'⚠️' if failed else '✅'} [BR→P] {name}: {audit.get('changed', 0)} updated, "
                f"{audit.get('unchanged', 0)} unchanged, {failed} failed ({audit.get('scanned', 0)} scanned)")
        audits.append({key: audit.get(key) for key in ("output_dir", "scanned", "changed", "unchanged", "failed")})
    ctx.set_result(br_audit=audits)
    ok_count = len(audits) - sum(1 for a in audits if a.get("failed"))
    ctx.set_result(br_message=f"✅ {ok_count} / ⚠️ {failed_inputs}" if failed_inputs else f"✅ {ok_count} done")
    return {"ok": True if audits else False, "outputs": [],
            "error": None if audits else "No output folder could be converted (see the log)."}


KINDS = {
    "md_txt_sidecars": {"verb": "Generating MD / TXT", "icon": "DESCRIPTION", "stop_kind": "translation",
                        "run": run_md_txt, "resumable": False},
    "br_to_paragraphs": {"verb": "Converting <br> to <p>", "icon": "FORMAT_PARAGRAPH", "stop_kind": "translation",
                         "run": run_br_to_paragraphs, "resumable": False},
    "compile_epub": {"verb": "Compiling EPUB", "icon": "MENU_BOOK", "stop_kind": "translation", "run": run_epub},
    "compile_pdf": {"verb": "Compiling PDF", "icon": "PICTURE_AS_PDF", "stop_kind": "translation", "run": run_pdf},
    "validate_epub": {"verb": "Validating EPUB", "icon": "RULE", "stop_kind": "translation", "run": run_validate,
                      "resumable": False},
    "rename_outputs": {"verb": "Renaming files", "icon": "DRIVE_FILE_RENAME_OUTLINE", "stop_kind": "translation",
                       "run": run_rename, "resumable": False},
}
