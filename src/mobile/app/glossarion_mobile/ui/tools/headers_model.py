"""Headers & metadata screen model (UI_SPEC §4.5; FEATURE_MAP qa-epub-pdf 54-58).

Pure Python (no Flet).

* **Header / TOC caches** (``translated_headers.txt`` / ``TOC.txt``): ``plan_artifact_delete``
  + ``execute_artifact_delete`` follow the desktop "Delete Header Files" / "Delete TOC.txt"
  buttons (``other_settings.delete_translated_headers_file`` / ``delete_toc_txt_file``) over
  the shared ``translation_artifacts`` primitives: the same file lookup, the RECYCLED-link
  question (``translation_artifacts_are_recycled_linked``), the progress reset to
  ``pending`` with the model name cleared (``update_translation_artifact_progress``) and the
  desktop summary / question / result texts. The desktop functions resolve each EPUB's
  output folder by name; here the folder comes from the Library row.
* **Metadata fields**: ``detect_fields`` is the desktop dialog's own detector
  (``MetadataBatchTranslatorUI._detect_all_metadata_fields_for_epub``, called unbound: it
  only logs through ``self.gui``); ``STANDARD_FIELDS`` / ``DEFAULT_ENABLED_FIELDS`` are the
  "Configure Metadata Translation" dialog's tables and ``saved_selection`` /
  ``fields_config`` its load and save rules (flat dict for one EPUB, ``_per_epub`` + the
  merged flat dict for several). ``tests_host/test_tools_ui.py`` keeps the tables equal to
  the desktop literals.
* **Translation modes** and **prompts**: the dialog's radio labels and the "Configure All
  Prompts" tabs (schema keys the screen renders as prompt tiles).
* Job specs: ``headers_spec`` (``translate_headers``) and ``metadata_spec`` (``metadata``,
  desktop ``output_roots`` = each book's output folder's parent).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

__all__ = [
    "ARTIFACT_TEXTS",
    "ArtifactPlan",
    "DEFAULT_ENABLED_FIELDS",
    "METADATA_MODES",
    "PROMPT_GROUPS",
    "STANDARD_FIELDS",
    "artifact_status",
    "detect_fields",
    "execute_artifact_delete",
    "existing_metadata_warning",
    "fields_config",
    "headers_spec",
    "metadata_spec",
    "plan_artifact_delete",
    "saved_selection",
]

# ---- header / TOC caches ---------------------------------------------------------------------

#: Per kind: file label, the "allow ... re-translated" line, the linked-delete question and
#: the two button labels of the desktop dialog.
ARTIFACT_TEXTS = {
    "headers": {
        "file": "translated_headers.txt",
        "counterpart": "toc",
        "counterpart_file": "TOC.txt",
        "retranslate": "This will allow headers to be re-translated on the next run.",
        "linked": ("\n\nRECYCLED link detected: one of TOC.txt and "
                   "translated_headers.txt was created by reusing the other. "
                   "Deleting only translated_headers.txt leaves TOC.txt available, "
                   "so the header cache may be rebuilt from it without a new API "
                   "translation.\n\nDelete both linked files to force fresh TOC and "
                   "header translation, or delete only the header file?"),
        "both": "Delete Both Linked Files",
        "only": "Delete Only Header Files",
        "not_found": "translated_headers.txt not found",
    },
    "toc": {
        "file": "TOC.txt",
        "counterpart": "headers",
        "counterpart_file": "translated_headers.txt",
        "retranslate": "This will allow TOC entries to be re-translated on the next run.",
        "linked": ("\n\nRECYCLED link detected: one of TOC.txt and "
                   "translated_headers.txt was created by reusing the other. "
                   "Deleting only TOC.txt leaves translated_headers.txt available, "
                   "so the TOC cache may be rebuilt from it without a new API "
                   "translation.\n\nDelete both linked files to force fresh TOC and "
                   "header translation, or delete only the TOC file?"),
        "both": "Delete Both Linked Files",
        "only": "Delete Only TOC Files",
        "not_found": "TOC.txt not found",
    },
}


def _artifacts() -> Any:
    import translation_artifacts

    return translation_artifacts


def _artifact_file(folder: str, kind: str) -> Optional[str]:
    """The cache file the desktop button deletes (headers: exact name; TOC: TOC.txt or toc.txt)."""
    if kind == "toc":
        upper = os.path.join(folder, "TOC.txt")
        lower = os.path.join(folder, "toc.txt")
        return upper if os.path.exists(upper) else (lower if os.path.exists(lower) else None)
    path = os.path.join(folder, "translated_headers.txt")
    return path if os.path.exists(path) else None


def artifact_status(folder: str) -> dict:
    """``{"headers": path|None, "toc": path|None}`` for a workspace (status chips)."""
    if not folder or not os.path.isdir(folder):
        return {"headers": None, "toc": None}
    return {"headers": _artifact_file(folder, "headers"), "toc": _artifact_file(folder, "toc")}


@dataclass
class ArtifactPlan:
    kind: str  # headers | toc
    total: int  # books looked at
    found: list = field(default_factory=list)  # [(base, path)]
    not_found: list = field(default_factory=list)  # [(base, reason)]
    errors: list = field(default_factory=list)  # [(base, error)]
    linked: dict = field(default_factory=dict)  # normcase(path) -> counterpart path

    @property
    def has_linked(self) -> bool:
        return bool(self.linked)

    def summary_text(self) -> str:
        """The desktop confirmation body (without the RECYCLED question)."""
        texts = ARTIFACT_TEXTS[self.kind]
        text = f"Summary for {self.total} EPUB file(s):\n\n"
        if self.found:
            text += f"✅ Files to delete ({len(self.found)}):\n"
            for base, _path in self.found:
                text += f"  • {base}\n"
            text += "\n"
        if self.not_found:
            text += f"⚠️ Files not found ({len(self.not_found)}):\n"
            for base, reason in self.not_found:
                text += f"  • {base}: {reason}\n"
            text += "\n"
        if self.errors:
            text += f"❌ Errors ({len(self.errors)}):\n"
            for base, error in self.errors:
                text += f"  • {base}: {error}\n"
            text += "\n"
        if self.found:
            text += texts["retranslate"]
        return text

    def question_text(self) -> str:
        return self.summary_text() + (ARTIFACT_TEXTS[self.kind]["linked"] if self.linked else "")


def plan_artifact_delete(targets: Sequence[Any], kind: str, log: Callable[[str], Any] = lambda _m: None
                         ) -> ArtifactPlan:
    """Find each target's cache file (desktop first pass). ``targets``: ToolTargets / (title, folder)."""
    if kind not in ARTIFACT_TEXTS:
        raise ValueError(kind)
    ta = _artifacts()
    plan = ArtifactPlan(kind=kind, total=len(targets))
    counterpart = ARTIFACT_TEXTS[kind]["counterpart"]
    for target in targets:
        base = str(getattr(target, "title", "") or (target[0] if isinstance(target, tuple) else ""))
        folder = str(getattr(target, "folder", "") or (target[1] if isinstance(target, tuple) else ""))
        try:
            log(f"🔍 Processing EPUB: {base}")
            if not folder or not os.path.isdir(folder):
                log(f"  ⚠️ No output directory found for {base}")
                plan.not_found.append((base, "No output directory found"))
                continue
            path = _artifact_file(folder, kind)
            if path:
                plan.found.append((base, path))
                progress = ta.load_translation_artifact_progress(folder)
                if ta.translation_artifacts_are_recycled_linked(progress):
                    plan.linked[os.path.normcase(path)] = ta.translation_artifact_path(folder, counterpart)
                log(f"  ✓ Found {os.path.basename(path)} in {os.path.basename(folder)}")
            else:
                plan.not_found.append((base, ARTIFACT_TEXTS[kind]["not_found"]))
                log(f"  ⚠️ No {ARTIFACT_TEXTS[kind]['file']} in {os.path.basename(folder)}")
        except Exception as exc:
            plan.errors.append((base, str(exc)))
            log(f"  ❌ Error processing {base}: {exc}")
    return plan


def execute_artifact_delete(plan: ArtifactPlan, *, delete_linked: bool = False,
                            log: Callable[[str], Any] = lambda _m: None) -> tuple:
    """Delete the planned files (+ the linked counterparts); returns ``(message, ok)`` (desktop texts)."""
    ta = _artifacts()
    kind = plan.kind
    counterpart_kind = ARTIFACT_TEXTS[kind]["counterpart"]
    deleted: list = []
    count = 0
    errors = list(plan.errors)
    for base, path in plan.found:
        try:
            os.remove(path)
            deleted.append(base)
            count += 1
            ta.update_translation_artifact_progress(os.path.dirname(path), kind, "pending", clear_model_name=True)
            log(f"✅ Deleted {os.path.basename(path)} from {base}")
        except Exception as exc:
            errors.append((base, f"Delete failed: {exc}"))
            log(f"❌ Failed to delete {ARTIFACT_TEXTS[kind]['file']} from {base}: {exc}")
        counterpart = plan.linked.get(os.path.normcase(path))
        if delete_linked and counterpart:
            try:
                if os.path.exists(counterpart):
                    os.remove(counterpart)
                    count += 1
                    log(f"Deleted linked {os.path.basename(counterpart)} from {base}")
                ta.update_translation_artifact_progress(os.path.dirname(path), counterpart_kind, "pending",
                                                        clear_model_name=True)
            except Exception as exc:
                errors.append((base, f"Linked delete failed: {exc}"))
                log(f"Failed to delete linked {ARTIFACT_TEXTS[kind]['counterpart_file']} from {base}: {exc}")
    if deleted:
        message = f"Successfully deleted {count} file(s):\n" + "\n".join(f"• {b}" for b in deleted)
        if errors:
            message += f"\n\nErrors: {len(errors)} file(s) failed to delete."
        return message, True
    return "No files were successfully deleted.", False


# ---- metadata ----------------------------------------------------------------------------------

#: "Configure Metadata Translation" standard fields: field -> (label, description).
STANDARD_FIELDS = {
    'title': ('Title', 'The book title'),
    'creator': ('Author/Creator', 'The author or creator'),
    'publisher': ('Publisher', 'The publishing company'),
    'subject': ('Subject/Genre', 'Subject categories or genres'),
    'description': ('Description', 'Book synopsis'),
    'series': ('Series Name', 'Name of the book series'),
    'language': ('Language', 'Original language'),
    'date': ('Publication Date', 'When published'),
    'rights': ('Rights', 'Copyright information'),
}
#: Fields enabled when nothing was saved (and after Reset).
DEFAULT_ENABLED_FIELDS = frozenset({'title', 'description', 'subject'})

#: The dialog's "Translation Mode" radios: value, label, tooltip.
METADATA_MODES = (
    ("together", "Translate together (single API call)", ""),
    ("metadata_separate", "Translate Metadata separately (2 API calls)",
     "Translate the book title first, then translate all selected metadata fields together in a second request."),
    ("parallel", "Translate separately (parallel API calls)", ""),
)

#: "Configure All Prompts" tabs -> the schema keys they edit.
PROMPT_GROUPS = (
    ("Book Title", ("book_title_system_prompt", "book_title_prompt")),
    ("Chapter Headers", ("batch_header_system_prompt", "batch_header_prompt", "batch_header_prepend_number_pattern")),
    ("Metadata Fields", ("metadata_system_prompt", "metadata_batch_prompt", "metadata_field_prompts")),
)


def detect_fields(epub_path: str, log: Callable[[str], Any] = lambda _m: None) -> dict:
    """Every metadata field of an EPUB (the desktop dialog's detector, unbound)."""
    from metadata_batch_translator import MetadataBatchTranslatorUI

    fake_ui = SimpleNamespace(gui=SimpleNamespace(append_log=log))
    return dict(MetadataBatchTranslatorUI._detect_all_metadata_fields_for_epub(fake_ui, epub_path) or {})


def saved_selection(fields_config: Mapping[str, Any], epub_path: str) -> dict:
    """``_get_saved_fields_for_epub``: the EPUB's ``_per_epub`` entry, else the flat dict."""
    config = dict(fields_config or {})
    basename = os.path.basename(epub_path) if epub_path else ''
    per_epub = config.get('_per_epub', {}) or {}
    if basename in per_epub:
        return dict(per_epub[basename])
    return config


def initial_checks(detected: Mapping[str, Any], saved: Mapping[str, Any],
                   sync: Optional[dict] = None) -> dict:
    """Checkbox states of one EPUB (desktop ``_rebuild_fields``).

    A field already shown for another EPUB keeps that state (the dialog's shared
    ``field_sync_state``, updated in place here); otherwise the saved value, defaulting to
    ``DEFAULT_ENABLED_FIELDS`` for standard fields and off for custom ones.
    """
    sync = sync if sync is not None else {}
    checks: dict = {}
    names = [n for n in STANDARD_FIELDS if n in detected] + [n for n in detected if n not in STANDARD_FIELDS]
    for name in names:
        if name in sync:
            checks[name] = bool(sync[name])
        else:
            default = name in DEFAULT_ENABLED_FIELDS if name in STANDARD_FIELDS else False
            checks[name] = bool(saved.get(name, default))
            sync[name] = checks[name]
    return checks


def fields_config(previous: Mapping[str, Any], selections: Mapping[str, Mapping[str, bool]],
                  epub_paths: Sequence[str]) -> dict:
    """``translate_metadata_fields`` after Save (desktop ``save_metadata_config``).

    One EPUB: the flat dict of its checkboxes. Several: every EPUB's checkboxes under
    ``_per_epub`` (by basename, keeping earlier EPUBs' saved entries) plus their merge.
    """
    config = {k: v for k, v in dict(previous or {}).items()}
    if len(epub_paths) > 1:
        per_epub = dict(config.get('_per_epub', {}) or {})
        for path, checks in selections.items():
            per_epub[os.path.basename(path)] = {k: bool(v) for k, v in checks.items()}
        combined: dict = {}
        for _basename, fields in per_epub.items():
            combined.update(fields)
        combined['_per_epub'] = per_epub
        return combined
    checks = next(iter(selections.values()), {}) if selections else {}
    return {k: bool(v) for k, v in checks.items() if k != '_per_epub'}


def existing_metadata_warning(folders: Sequence[str]) -> Optional[str]:
    """The desktop Library's "Metadata Already Exists" text, or None when no metadata.json exists."""
    existing = [os.path.join(f, "metadata.json") for f in folders if f and
                os.path.isfile(os.path.join(f, "metadata.json"))]
    if not existing:
        return None
    total = len(folders)
    if total == 1:
        text = ("metadata.json already exists for this EPUB.\n\n"
                "Continuing will regenerate the selected translated "
                "metadata fields and replace their current translated "
                "values.")
    else:
        text = (f"metadata.json already exists for {len(existing)} of the {total} "
                "selected EPUBs.\n\n"
                "Continuing will regenerate the selected translated "
                "metadata fields and replace their current translated "
                "values.")
    preview = existing[:8]
    detail = "\n".join(preview)
    if len(existing) > len(preview):
        detail += f"\n… and {len(existing) - len(preview)} more"
    return text + "\n\n" + detail


# ---- job specs ----------------------------------------------------------------------------------


def _origin(targets: Sequence[Any], tool: str, label: str) -> dict:
    if len(targets) == 1 and getattr(targets[0], "bid", ""):
        return {"type": "library", "bid": targets[0].bid, "label": f"Library · {targets[0].title}"}
    return {"type": "tools", "tool": tool, "label": label}


def headers_spec(targets: Sequence[Any], *, rebuild_epub: bool = True) -> Any:
    from glossarion_mobile.services.jobs import JobSpec

    rows = [t for t in targets if getattr(t, "source", "") and str(t.source).lower().endswith(".epub")]
    if not rows:
        raise ValueError("No EPUB or PDF file selected, or the file does not exist.")
    title = str(rows[0].title or os.path.basename(rows[0].source))
    if len(rows) > 1:
        title = f"{title} +{len(rows) - 1}"
    params = {"targets": [{"source": t.source, "folder": t.folder or None} for t in rows],
              "rebuild_epub": bool(rebuild_epub)}
    return JobSpec(kind="translate_headers", title=title, inputs=tuple(t.source for t in rows), params=params,
                   origin=_origin(rows, "headers", "Tools · Headers & metadata"))


def metadata_spec(targets: Sequence[Any]) -> Any:
    """``metadata`` job: the raw EPUBs + ``{source: output root}`` (desktop ``output_roots``)."""
    from glossarion_mobile.services.jobs import JobSpec

    rows = [t for t in targets if getattr(t, "source", "") and str(t.source).lower().endswith(".epub")]
    if not rows:
        raise ValueError("No raw EPUB resolves for the selection")
    roots = {t.source: os.path.dirname(os.path.abspath(t.folder)) for t in rows if t.folder}
    title = str(rows[0].title) if len(rows) == 1 else f"{len(rows)} books"
    return JobSpec(kind="metadata", title=title, inputs=tuple(t.source for t in rows),
                   params={"output_roots": roots}, origin=_origin(rows, "headers", "Tools · Headers & metadata"))


def epub_targets(targets: Iterable[Any]) -> list:
    return [t for t in targets if str(getattr(t, "source", "")).lower().endswith(".epub")]
