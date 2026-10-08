"""Headers & metadata screen model (UI_SPEC §4.5; FEATURE_MAP qa-epub-pdf 54-58).

Pure Python (no Flet).

* **Header / TOC caches** (``translated_headers.txt`` / ``TOC.txt``): ``plan_artifact_delete``
  + ``execute_artifact_delete`` run the desktop "Delete Header Files" / "Delete TOC.txt"
  cores (``output_tools_core.plan_artifact_cache_delete`` / ``delete_planned_artifact_caches``
  / ``artifact_delete_result``, which ``other_settings.delete_translated_headers_file`` /
  ``delete_toc_txt_file`` call): the file lookup, the RECYCLED-link question, the progress
  reset to ``pending`` with the model name cleared and the summary / question / result texts.
  The desktop looks each EPUB's output folder up by name; here the folder comes from the
  Library row (``output_dir_for``).
* **Metadata fields**: ``detect_fields`` is the desktop dialog's own detector
  (``MetadataBatchTranslatorUI._detect_all_metadata_fields_for_epub``, called unbound: it
  only logs through ``self.gui``); ``STANDARD_FIELDS`` / ``DEFAULT_ENABLED_FIELDS`` are the
  "Configure Metadata Translation" tables and ``saved_selection`` / ``initial_checks`` /
  ``fields_config`` its load, checkbox and save rules - all from metadata_batch_translator
  (``METADATA_STANDARD_FIELDS``, ``saved_metadata_fields_for_epub``, ``metadata_field_checked``,
  ``store_metadata_field_selection``, ``final_metadata_fields_config``), which the desktop
  dialog calls.
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
    "reset_prompt_changes",
    "saved_selection",
]

# ---- header / TOC caches ---------------------------------------------------------------------

def _tools_core() -> Any:
    import output_tools_core

    return output_tools_core


def _metadata_rules() -> Any:
    import metadata_batch_translator

    return metadata_batch_translator


def __getattr__(name: str) -> Any:
    """Shared tables, resolved on first use (the backend modules load lazily)."""
    if name == "ARTIFACT_TEXTS":
        # the desktop texts per cache kind (+ "both" / "only" button aliases for the screen)
        return {kind: dict(texts, both=texts["delete_both"], only=texts["delete_only"])
                for kind, texts in _tools_core().ARTIFACT_DELETE_TEXTS.items()}
    if name == "ArtifactPlan":
        return _tools_core().ArtifactDeletePlan
    if name == "STANDARD_FIELDS":
        return _metadata_rules().METADATA_STANDARD_FIELDS
    if name == "DEFAULT_ENABLED_FIELDS":
        return _metadata_rules().METADATA_DEFAULT_ENABLED_FIELDS
    raise AttributeError(name)


def _pseudo_source(target: Any) -> tuple:
    """``(path, folder)``: the target's source (else ``<folder>/<title>.epub``) and its output folder."""
    title = str(getattr(target, "title", "") or (target[0] if isinstance(target, tuple) else ""))
    folder = str(getattr(target, "folder", "") or (target[1] if isinstance(target, tuple) else ""))
    source = str(getattr(target, "source", "") or "")
    if source and os.path.splitext(os.path.basename(source))[0] == title:
        return source, folder
    return os.path.join(folder or os.getcwd(), f"{title}.epub"), folder


def artifact_status(folder: str) -> dict:
    """``{"headers": path|None, "toc": path|None}`` for a workspace (status chips)."""
    if not folder or not os.path.isdir(folder):
        return {"headers": None, "toc": None}
    find = _tools_core().find_artifact_cache_file
    return {"headers": find(folder, "headers"), "toc": find(folder, "toc")}


def plan_artifact_delete(targets: Sequence[Any], kind: str, log: Callable[[str], Any] = lambda _m: None) -> Any:
    """First pass of the desktop button (``output_tools_core.plan_artifact_cache_delete``).

    ``targets``: ToolTargets / (title, folder); each book is named after its title and its
    folder comes from the row.
    """
    core = _tools_core()
    if kind not in core.ARTIFACT_DELETE_TEXTS:
        raise ValueError(kind)
    pairs = [_pseudo_source(target) for target in targets]
    folders = {path: folder for path, folder in pairs}
    return core.plan_artifact_cache_delete(kind, [path for path, _folder in pairs], log=log,
                                           output_dir_for=lambda path: folders.get(path) or None)


def execute_artifact_delete(plan: Any, *, delete_linked: bool = False,
                            log: Callable[[str], Any] = lambda _m: None) -> tuple:
    """Delete the planned files (+ the linked counterparts); returns ``(message, ok)`` (desktop texts)."""
    core = _tools_core()
    deleted, count = core.delete_planned_artifact_caches(plan, delete_linked=delete_linked, log=log)
    ok, _title, message = core.artifact_delete_result(deleted, count, plan.errors)
    return message, ok


# ---- metadata ----------------------------------------------------------------------------------


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
    # the dialog's ⚙️ Advanced tab: Language Detection radios + "Language to use", Output Language
    ("Advanced", ("lang_prompt_behavior", "forced_source_lang", "output_language")),
)


def reset_prompt_changes(config: Mapping[str, Any]) -> tuple:
    """``(keys to remove, {key: value} to write)`` for "Reset all prompts to defaults" on ``config``
    (``metadata_defaults.reset_metadata_prompts``: the desktop ``_reset_all_prompts_to_defaults`` key
    list, ``book_title_prompt`` blanked, the default prompts re-seeded)."""
    from metadata_defaults import reset_metadata_prompts

    before = dict(config)
    after = dict(config)
    reset_metadata_prompts(after)
    removed = tuple(key for key in before if key not in after)
    written = {key: value for key, value in after.items() if key not in before or before[key] != value}
    return removed, written


def detect_fields(epub_path: str, log: Callable[[str], Any] = lambda _m: None) -> dict:
    """Every metadata field of an EPUB (the desktop dialog's detector, unbound)."""
    from metadata_batch_translator import MetadataBatchTranslatorUI

    fake_ui = SimpleNamespace(gui=SimpleNamespace(append_log=log))
    return dict(MetadataBatchTranslatorUI._detect_all_metadata_fields_for_epub(fake_ui, epub_path) or {})


def saved_selection(fields_config: Mapping[str, Any], epub_path: str) -> dict:
    """The dialog's ``_get_saved_fields_for_epub``: the EPUB's ``_per_epub`` entry, else the flat dict."""
    return dict(_metadata_rules().saved_metadata_fields_for_epub(dict(fields_config or {}), epub_path) or {})


def initial_checks(detected: Mapping[str, Any], saved: Mapping[str, Any],
                   sync: Optional[dict] = None) -> dict:
    """Checkbox states of one EPUB (desktop ``_rebuild_fields``: standard fields, then custom ones).

    A field already shown for another EPUB keeps that state (the dialog's shared
    ``field_sync_state``, updated in place here); otherwise the saved value, defaulting to
    ``DEFAULT_ENABLED_FIELDS`` for standard fields and off for custom ones
    (``metadata_batch_translator.metadata_field_checked``).
    """
    rules = _metadata_rules()
    standard = rules.METADATA_STANDARD_FIELDS
    defaults = rules.METADATA_DEFAULT_ENABLED_FIELDS
    sync = sync if sync is not None else {}
    checks: dict = {}
    for name in standard:
        if name in detected:
            checks[name] = bool(rules.metadata_field_checked(name, saved, sync, name in defaults))
    for name in detected:
        if name not in standard:
            checks[name] = bool(rules.metadata_field_checked(name, saved, sync, False))
    return checks


def fields_config(previous: Mapping[str, Any], selections: Mapping[str, Mapping[str, bool]],
                  epub_paths: Sequence[str]) -> dict:
    """``translate_metadata_fields`` after Save (desktop ``save_metadata_config``).

    Every EPUB's checkboxes go through ``store_metadata_field_selection`` (``_per_epub`` by
    basename with several EPUBs, the flat dict for one) and the result through
    ``final_metadata_fields_config``. The previous config is copied, never changed.
    """
    rules = _metadata_rules()
    config = dict(previous or {})
    if isinstance(config.get("_per_epub"), Mapping):
        config["_per_epub"] = dict(config["_per_epub"])
    paths = list(epub_paths)
    for path, checks in selections.items():
        rules.store_metadata_field_selection(config, paths, path, {k: bool(v) for k, v in checks.items()})
    return rules.final_metadata_fields_config(config, paths)


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
