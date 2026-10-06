# output_tools_core.py
"""GUI-free cores of Other Settings' output-folder tools (U7).

Moved out of ``other_settings.py`` (which keeps the buttons, file pickers and message boxes and
calls these) so Glossarion Mobile's Converter / Headers & metadata screens run the same code:

* ``import_custom_fonts`` - the "Load Font…" copy rules (font files and the fonts inside ZIPs);
* ``validate_epub_outputs`` - the "Validate EPUB Structure" loop: output folder lookup,
  ``TransateKRtoEN.validate_epub_structure`` / ``check_epub_readiness`` and the log / result
  wording;
* ``plan_artifact_cache_delete`` / ``delete_planned_artifact_caches`` and the dialog texts of
  "Delete Header Files" (``translated_headers.txt``) and "Delete TOC.txt": folder lookup, the
  RECYCLED-link counterparts, the deletion with the progress reset to ``pending`` and the
  summary / question / result wording.

The desktop looks each EPUB's output folder up by name (``find_epub_output_dir``: the
``OUTPUT_DIRECTORY`` / ``output_directory`` override, the working directory, the program
folder); a front end that already knows the folder (the mobile Library) passes
``output_dir_for(epub_path)``.

Python 3.10 compatible; never imports PySide6, translator_gui or dpi_setup.
"""

import os

from translation_artifacts import (
    load_translation_artifact_progress,
    translation_artifact_path,
    translation_artifacts_are_recycled_linked,
    update_translation_artifact_progress,
)

__all__ = [
    "ARTIFACT_DELETE_TEXTS",
    "ArtifactDeletePlan",
    "FONT_EXTS",
    "artifact_delete_result",
    "delete_planned_artifact_caches",
    "find_artifact_cache_file",
    "import_custom_fonts",
    "plan_artifact_cache_delete",
    "validate_epub_outputs",
]

#: Font files the "Load Font…" button copies (other_settings ``_FONT_EXTS``).
FONT_EXTS = ('.ttf', '.otf', '.woff', '.woff2')


# ---------------------------------------------------------------------------
# Load Font…
# ---------------------------------------------------------------------------

def import_custom_fonts(files, gdir, font_exts=FONT_EXTS):
    """Copy the chosen font files and the fonts inside chosen ZIPs into ``gdir``; returns the count."""
    import shutil
    import zipfile
    copied = 0
    for src in files:
        ext = os.path.splitext(src)[1].lower()
        if ext == '.zip':
            # Extract all font files from the zip (including subfolders)
            try:
                with zipfile.ZipFile(src, 'r') as zf:
                    for entry in zf.namelist():
                        entry_ext = os.path.splitext(entry)[1].lower()
                        if entry_ext in font_exts:
                            font_name = os.path.basename(entry)
                            if not font_name:
                                continue
                            dst = os.path.join(gdir, font_name)
                            with zf.open(entry) as zin, open(dst, 'wb') as zout:
                                zout.write(zin.read())
                            copied += 1
            except Exception:
                pass
        elif ext in font_exts:
            dst = os.path.join(gdir, os.path.basename(src))
            try:
                shutil.copy2(src, dst)
                copied += 1
            except Exception:
                pass
    return copied


# ---------------------------------------------------------------------------
# Validate EPUB Structure
# ---------------------------------------------------------------------------

def validate_epub_outputs(epub_files_to_process, *, config=None, log=print, output_dir_for=None):
    """Validate each EPUB's output folder; returns ``(all_passed, all_results)`` (desktop wording)."""
    current_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    override_dir = os.environ.get('OUTPUT_DIRECTORY') or (config.get('output_directory') if config is not None else None)

    all_passed = True
    all_results = []

    for epub_path in epub_files_to_process:
        epub_base = os.path.splitext(os.path.basename(epub_path))[0]
        log(f"🔍 Validating EPUB structure for: {epub_base}")

        if output_dir_for is not None:
            output_dir = output_dir_for(epub_path) or None
        else:
            candidates = [
                os.path.join(current_dir, epub_base),
                os.path.join(script_dir, epub_base),
                os.path.join(current_dir, 'src', epub_base),
            ]
            if override_dir:
                candidates.insert(0, os.path.join(override_dir, epub_base))

            output_dir = None
            for candidate in candidates:
                if os.path.isdir(candidate):
                    try:
                        files = os.listdir(candidate)
                        html_files = [f for f in files if f.lower().endswith(('.html', '.xhtml', '.htm'))]
                        if html_files:
                            output_dir = candidate
                            break
                    except Exception:
                        continue

        if not output_dir:
            log(f"  ⚠️ No output directory found for {epub_base}")
            all_results.append(f"⚠️ {epub_base}: No output directory found")
            all_passed = False
            continue

        try:
            from TransateKRtoEN import validate_epub_structure, check_epub_readiness
            structure_ok = validate_epub_structure(output_dir)
            readiness_ok = check_epub_readiness(output_dir)

            if structure_ok and readiness_ok:
                log(f"  ✅ {epub_base}: PASSED")
                all_results.append(f"✅ {epub_base}: All structure files present")
            elif structure_ok:
                log(f"  ⚠️ {epub_base}: Structure OK, some issues")
                all_results.append(f"⚠️ {epub_base}: Structure OK, some issues found")
                all_passed = False
            else:
                log(f"  ❌ {epub_base}: Missing critical files")
                all_results.append(f"❌ {epub_base}: Missing critical EPUB files")
                all_passed = False
        except Exception as e:
            log(f"  ❌ Validation error for {epub_base}: {e}")
            all_results.append(f"❌ {epub_base}: {e}")
            all_passed = False

    return all_passed, all_results


# ---------------------------------------------------------------------------
# Delete Header Files / Delete TOC.txt
# ---------------------------------------------------------------------------

#: Per cache kind: the texts of other_settings ``delete_translated_headers_file`` ("headers")
#: and ``delete_toc_txt_file`` ("toc").
ARTIFACT_DELETE_TEXTS = {
    "headers": {
        "file": "translated_headers.txt",
        "counterpart": "toc",
        "counterpart_file": "TOC.txt",
        "not_found": "translated_headers.txt not found",
        "retranslate": "This will allow headers to be re-translated on the next run.",
        "linked": (
            "\n\nRECYCLED link detected: one of TOC.txt and "
            "translated_headers.txt was created by reusing the other. "
            "Deleting only translated_headers.txt leaves TOC.txt available, "
            "so the header cache may be rebuilt from it without a new API "
            "translation.\n\nDelete both linked files to force fresh TOC and "
            "header translation, or delete only the header file?"
        ),
        "delete_both": "Delete Both Linked Files",
        "delete_only": "Delete Only Header Files",
        "failed_delete": "❌ Failed to delete translated_headers.txt from {epub_base}: {e}",
        "failed_linked": "Failed to delete linked TOC.txt from {epub_base}: {e}",
    },
    "toc": {
        "file": "TOC.txt",
        "counterpart": "headers",
        "counterpart_file": "translated_headers.txt",
        "not_found": "TOC.txt not found",
        "retranslate": "This will allow TOC entries to be re-translated on the next run.",
        "linked": (
            "\n\nRECYCLED link detected: one of TOC.txt and "
            "translated_headers.txt was created by reusing the other. "
            "Deleting only TOC.txt leaves translated_headers.txt available, "
            "so the TOC cache may be rebuilt from it without a new API "
            "translation.\n\nDelete both linked files to force fresh TOC and "
            "header translation, or delete only the TOC file?"
        ),
        "delete_both": "Delete Both Linked Files",
        "delete_only": "Delete Only TOC Files",
        "failed_delete": "❌ Failed to delete TOC.txt from {epub_base}: {e}",
        "failed_linked": "Failed to delete linked translated_headers.txt from {epub_base}: {e}",
    },
}


def find_artifact_cache_file(output_dir, kind):
    """The cache file the button deletes: ``translated_headers.txt``, or ``TOC.txt`` / ``toc.txt``."""
    if kind == "toc":
        # Look for TOC.txt (case-insensitive)
        toc_upper = os.path.join(output_dir, "TOC.txt")
        toc_lower = os.path.join(output_dir, "toc.txt")
        return toc_upper if os.path.exists(toc_upper) else (toc_lower if os.path.exists(toc_lower) else None)
    # Look for translated_headers.txt in the output directory
    headers_file = os.path.join(output_dir, "translated_headers.txt")
    return headers_file if os.path.exists(headers_file) else None


class ArtifactDeletePlan:
    """The first pass of a cache deletion: what was found, missing or failed, and the RECYCLED links."""

    def __init__(self, kind, epub_files):
        self.kind = kind
        self.epub_files = list(epub_files)
        self.files_found = []  # [(epub_base, path)]
        self.files_not_found = []  # [(epub_base, reason)]
        self.errors = []  # [(epub_base, error)]
        self.linked_counterparts = {}  # normcase(path) -> counterpart path

    @property
    def nothing_processed(self):
        return not self.files_found and not self.files_not_found and not self.errors

    # Read-only aliases for GUI-free front ends (Glossarion Mobile's Headers & metadata screen)
    @property
    def found(self):
        return self.files_found

    @property
    def not_found(self):
        return self.files_not_found

    @property
    def linked(self):
        return self.linked_counterparts

    @property
    def has_linked(self):
        return bool(self.linked_counterparts)

    @property
    def total(self):
        return len(self.epub_files)

    def summary_text(self):
        """The dialog body (summary, then the re-translate line when files were found)."""
        summary_text = f"Summary for {len(self.epub_files)} EPUB file(s):\n\n"

        if self.files_found:
            summary_text += f"✅ Files to delete ({len(self.files_found)}):\n"
            for epub_base, file_path in self.files_found:
                summary_text += f"  • {epub_base}\n"
            summary_text += "\n"

        if self.files_not_found:
            summary_text += f"⚠️ Files not found ({len(self.files_not_found)}):\n"
            for epub_base, reason in self.files_not_found:
                summary_text += f"  • {epub_base}: {reason}\n"
            summary_text += "\n"

        if self.errors:
            summary_text += f"❌ Errors ({len(self.errors)}):\n"
            for epub_base, error in self.errors:
                summary_text += f"  • {epub_base}: {error}\n"
            summary_text += "\n"

        if self.files_found:
            summary_text += ARTIFACT_DELETE_TEXTS[self.kind]["retranslate"]
        return summary_text

    def question_text(self):
        """The confirmation text (with the RECYCLED question when a linked pair was found)."""
        summary_text = self.summary_text()
        if self.linked_counterparts:
            return summary_text + ARTIFACT_DELETE_TEXTS[self.kind]["linked"]
        return summary_text


def plan_artifact_cache_delete(kind, epub_files_to_process, *, config=None, log=print, output_dir_for=None):
    """First pass of "Delete Header Files" / "Delete TOC.txt": find each EPUB's cache file."""
    texts = ARTIFACT_DELETE_TEXTS[kind]
    plan = ArtifactDeletePlan(kind, epub_files_to_process)
    config = config if config is not None else {}

    current_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # First pass: scan for files
    for epub_path in epub_files_to_process:
        try:
            epub_base = os.path.splitext(os.path.basename(epub_path))[0]
            log(f"🔍 Processing EPUB: {epub_base}")

            if output_dir_for is not None:
                output_dir = output_dir_for(epub_path) or None
            else:
                # Check the most common locations in order of priority (same as QA scanner)
                candidates = [
                    os.path.join(current_dir, epub_base),        # current working directory
                    os.path.join(script_dir, epub_base),         # src directory (where output typically goes)
                    os.path.join(current_dir, 'src', epub_base), # src subdirectory from current dir
                ]

                # Add output directory override if configured (matches QA scanner behavior)
                override_dir = os.environ.get('OUTPUT_DIRECTORY') or config.get('output_directory')
                if override_dir:
                    candidates.insert(0, os.path.join(override_dir, epub_base))
                    log(f"  🔍 Checking override directory: {override_dir}")

                output_dir = None
                for candidate in candidates:
                    if os.path.isdir(candidate):
                        # Verify the folder actually contains HTML/XHTML files
                        try:
                            files = os.listdir(candidate)
                            html_files = [f for f in files if f.lower().endswith(('.html', '.xhtml', '.htm'))]
                            if html_files:
                                output_dir = candidate
                                break
                        except Exception:
                            continue

            if not output_dir:
                log(f"  ⚠️ No output directory found for {epub_base}")
                plan.files_not_found.append((epub_base, "No output directory found"))
                continue

            cache_file = find_artifact_cache_file(output_dir, kind)

            if cache_file:
                plan.files_found.append((epub_base, cache_file))
                progress = load_translation_artifact_progress(output_dir)
                if translation_artifacts_are_recycled_linked(progress):
                    plan.linked_counterparts[os.path.normcase(cache_file)] = (
                        translation_artifact_path(output_dir, texts["counterpart"])
                    )
                log(f"  ✓ Found {os.path.basename(cache_file)} in {os.path.basename(output_dir)}")
            else:
                plan.files_not_found.append((epub_base, texts["not_found"]))
                log(f"  ⚠️ No {texts['file']} in {os.path.basename(output_dir)}")

        except Exception as e:
            epub_base = os.path.splitext(os.path.basename(epub_path))[0]
            plan.errors.append((epub_base, str(e)))
            log(f"  ❌ Error processing {epub_base}: {e}")
    return plan


def delete_planned_artifact_caches(plan, *, delete_linked=False, log=print):
    """Delete the planned files (and, with ``delete_linked``, their RECYCLED counterparts).

    Returns ``(files_deleted, deleted_file_count)``; failures are appended to ``plan.errors``.
    """
    texts = ARTIFACT_DELETE_TEXTS[plan.kind]
    files_deleted = []
    deleted_file_count = 0
    errors = plan.errors
    for epub_base, cache_file in plan.files_found:
        try:
            os.remove(cache_file)
            files_deleted.append(epub_base)
            deleted_file_count += 1
            update_translation_artifact_progress(
                os.path.dirname(cache_file),
                plan.kind,
                "pending",
                clear_model_name=True,
            )
            log(f"✅ Deleted {os.path.basename(cache_file)} from {epub_base}")
        except Exception as e:
            errors.append((epub_base, f"Delete failed: {e}"))
            log(texts["failed_delete"].format(epub_base=epub_base, e=e))

        counterpart = plan.linked_counterparts.get(
            os.path.normcase(cache_file)
        )
        if delete_linked and counterpart:
            try:
                if os.path.exists(counterpart):
                    os.remove(counterpart)
                    deleted_file_count += 1
                    log(
                        f"Deleted linked {os.path.basename(counterpart)} "
                        f"from {epub_base}"
                    )
                update_translation_artifact_progress(
                    os.path.dirname(cache_file),
                    texts["counterpart"],
                    "pending",
                    clear_model_name=True,
                )
            except Exception as e:
                errors.append((epub_base, f"Linked delete failed: {e}"))
                log(texts["failed_linked"].format(epub_base=epub_base, e=e))
    return files_deleted, deleted_file_count


def artifact_delete_result(files_deleted, deleted_file_count, errors):
    """``(ok, title, text)`` of the dialog's final message."""
    if files_deleted:
        success_msg = f"Successfully deleted {deleted_file_count} file(s):\n"
        success_msg += "\n".join([f"• {epub_base}" for epub_base in files_deleted])
        if errors:
            success_msg += f"\n\nErrors: {len(errors)} file(s) failed to delete."
        return True, "Success", success_msg
    return False, "Error", "No files were successfully deleted."
