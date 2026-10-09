"""Job kind adapters (plan §4 "job_kinds/* adapters only"; mobile-app design §5.1).

An adapter turns a ``JobSpec`` into calls on the job's ``HeadlessOwner``: it
checks its parameters, records the output folders (checkpointed for Resume and
read by ``ProgressWatcher``) and calls the shared owner methods the desktop
runs for the same action:

* ``translate``: ``owner._prepare_translation_run(files)`` +
  ``owner._translation_worker(request)`` (the desktop ``run_translation_thread``
  body, ``translation_pipeline.TranslationPipelineMixin``);
* ``direct_text``: ``headless_owner.DirectTextRunOptions.apply_to(owner)``, the
  Direct Text run environment, then the same two calls;
* ``extract_glossary``: ``owner.run_glossary_extraction_direct()`` over the
  selected files (it dispatches to ``_extract_glossary_from_text_file``);
* ``compile_epub`` / ``compile_pdf``: ``owner._run_epub_compile(folder)`` /
  ``owner._run_pdf_compile(folder)`` (``text_jobs.TextJobsMixin``);
* ``single_chapter``: the translate pair with the owner's ``_single_chapter_filter`` /
  ``_force_stream_all`` set (desktop ``start_single_chapter_translation``; the Reader's
  "Translate this chapter" and the Book page's chapter rows);
* ``qa_scan``: ``qa_scan_runtime.run_qa_scan_path`` per output folder (U6, QA Scanner);
* ``validate_epub`` / ``rename_outputs``: Converter actions (``TransateKRtoEN`` validation,
  ``output_naming._rename_output_files_for_retain``);
* ``translate_headers``: ``translate_headers_standalone.run_translation`` per EPUB + rebuild;
* ``metadata``: the owner's metadata-only run (desktop ``start_metadata_translation``);
* ``manga`` / ``manga_step``: Tools › Manga's batch run (``manga_runner.HeadlessMangaRunner``, the
  desktop ``MangaTranslationTab`` Start without Qt, the owner as ``main_gui``) and the editor steps
  (``manga_editor_core.MangaEditorSession``), U8.

No backend logic lives here. ``get_kind(kind)`` returns a ``KindInfo`` (verb and
icon for the strip and notifications, the ``stop_control.reset_for_new_run``
kind, and ``run(ctx)``); modules are imported lazily.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional

__all__ = ["KindInfo", "KIND_MODULES", "available_kinds", "compiled_outputs", "get_kind", "owner_method", "result_fields"]

#: kind -> adapter module (inside this package)
KIND_MODULES = {
    "translate": "translate",
    "direct_text": "direct_text",
    "extract_glossary": "glossary",
    "glossary_refine": "glossary",
    "unified_glossary": "glossary",
    "parallel_pair": "glossary",
    "compile_epub": "compile",
    "compile_pdf": "compile",
    "single_chapter": "single_chapter",
    # U6 tools (Tools › QA Scanner / Converter / Headers & metadata, Library metadata)
    "qa_scan": "qa",
    "validate_epub": "compile",
    "rename_outputs": "compile",
    "translate_headers": "headers",
    "metadata": "metadata",
    # U7 (Book page Retranslate / Resolve QA, Tools › Async batch / Review / RPG Maker, chat media modes)
    "retranslate": "retranslate",
    "resolve_qa": "resolve_qa",
    "async_batch": "async_batch",
    "review": "review",
    "rpgmaker": "rpgmaker",
    "generate_media": "generate_media",
    "translate_image": "image",
    # U8 (Tools › Manga: the batch Start / Generate glossary and the editor steps)
    "manga": "manga",
    "manga_step": "manga",
    # U9 (Tools › Converter: the desktop Other Settings retroactive actions)
    "md_txt_sidecars": "compile",
    "br_to_paragraphs": "compile",
}


@dataclass(frozen=True)
class KindInfo:
    kind: str
    verb: str  # "Translating" (strip title "Translating · Book.epub")
    icon: str  # Material icon name for the strip ring / Jobs rows
    stop_kind: str  # stop_control.reset_for_new_run(kind=...)
    run: Callable[[Any], Any]
    resumable: bool = True
    #: The adapter routes the API client's log records itself (``services.jobs.client_log_handlers``):
    #: JobService then attaches no handler of its own to the job's host (``JobBackend.client_logs``).
    own_client_logs: bool = False


_CACHE: dict[str, KindInfo] = {}


def get_kind(kind: Any) -> KindInfo:
    """``KindInfo`` for ``kind`` (``KeyError`` for unknown kinds)."""
    key = getattr(kind, "value", kind)
    key = str(key)
    info = _CACHE.get(key)
    if info is not None:
        return info
    module_name = KIND_MODULES.get(key)
    if module_name is None:
        raise KeyError(f"unknown job kind {key!r}")
    module = importlib.import_module(f"{__name__}.{module_name}")
    spec = module.KINDS[key]
    info = KindInfo(kind=key, **spec)
    _CACHE[key] = info
    return info


def available_kinds() -> tuple:
    return tuple(KIND_MODULES)


def owner_method(owner: Any, name: str) -> Callable[..., Any]:
    """``getattr(owner, name)``; a missing shared method fails the job with a clear message."""
    method = getattr(owner, name, None)
    if not callable(method):
        from glossarion_mobile.services.jobs import JobError

        raise JobError(f"This build's engine has no {name}() (the shared pipeline module is missing).")
    return method


def result_fields(result: Any) -> tuple:
    """(ok, outputs, error) of a shared result (None, bool, mapping or object such as CompileResult)."""
    if result is None:
        return None, [], None
    if isinstance(result, bool):
        return result, [], None
    if isinstance(result, Mapping):
        get = result.get
    else:
        def get(key: str, default: Any = None) -> Any:
            return getattr(result, key, default)

    ok = get("ok")
    if ok is None:
        ok = get("success")
    outputs: list = []
    for key in ("outputs", "paths"):
        for path in get(key) or ():
            if path:
                outputs.append(os.fspath(path))
    for key in ("path", "output_path", "epub_path", "pdf_path"):
        value = get(key)
        if value and os.fspath(value) not in outputs:
            outputs.append(os.fspath(value))
    error = get("error") or get("message") if ok is False else get("error")
    return (None if ok is None else bool(ok)), outputs, (str(error) if error else None)


#: Top-level files of an output folder listed as job outputs (the Result card / export).
OUTPUT_EXTENSIONS = (".epub", ".pdf")
TRANSLATED_SUFFIXES = ("_translated.txt", "_translated.srt", "_translated.ass", "_translated.lrc")


def compiled_outputs(folders: Iterable[Optional[str]]) -> list:
    """Compiled EPUB/PDF files and ``*_translated.*`` files at the top of each output folder."""
    found: list = []
    for folder in folders:
        if not folder or not os.path.isdir(folder):
            continue
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            continue
        for name in names:
            lower = name.lower()
            if lower.endswith(OUTPUT_EXTENSIONS) or lower.endswith(TRANSLATED_SUFFIXES):
                path = os.path.join(folder, name)
                if os.path.isfile(path) and path not in found:
                    found.append(path)
    return found
