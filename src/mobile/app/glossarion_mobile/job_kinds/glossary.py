"""Glossary jobs: Extract Glossary, the manual refinement pass, Unified Rebuild Now, the Parallel EPUB pair.

* ``extract_glossary`` - the desktop "Extract Glossary" run over the selected files:
  ``owner.run_glossary_extraction_direct()`` (``translation_pipeline.GlossaryPipelineMixin``)
  iterates ``owner.selected_files`` and dispatches per type, exactly as on desktop
  (EPUB/TXT/PDF through ``_extract_glossary_from_text_file``; image files are grouped
  by folder into one combined glossary - the mobile "image folder" source passes the
  folder's images). params: ``force_balanced_request_merging`` (bool, desktop default
  False).
* ``glossary_refine`` - the Glossary Progress "✨ Refine this" / refinement pass
  (Retranslation_GUI ``_confirm_manual_glossary_refinement`` → ``_start_manual_glossary_refinement``
  → ``_run_manual_glossary_refinement``): the plan and the run are the shared
  ``glossary_progress_core.plan_manual_glossary_refinement`` /
  ``run_manual_glossary_refinement`` (or the owner's ``_run_manual_glossary_refinement``
  when its mixin provides it). params: ``glossary_path``, ``progress_path``,
  ``source_path``, ``selected_types``, ``target_chunk_count`` (optional).
* ``unified_glossary`` - Unified Glossary "🔄 Rebuild Now":
  ``unified_glossary.rebuild_now(shared_dir, settings, log)`` with the desktop settings
  snapshot (``_rebuild_unified_glossary_now``). params: ``shared_dir``.
* ``parallel_pair`` - Parallel EPUB Pair Accept + Extract Glossary, the desktop's saved-pair
  restore + activation (``_start_parallel_epub_pair_restore`` / ``_activate_parallel_epub_pair_source``)
  through ``parallel_epub_core``: ``rebuild_parallel_epub_pair_result`` reads both EPUBs again
  and reattaches the saved compact mapping, ``build_parallel_epub_pair_artifact`` writes the
  paired working EPUB to a temporary folder (named so the glossary lands in the raw book's
  folder) and ``parallel_epub_pair_source_state`` is the owner's ``_parallel_epub_pair_source``
  (``run_env._parallel_epub_system_prompt_for_file`` reads the pair's system prompt from it);
  then its glossary is extracted. params: ``selection`` (compact, chapter-text free),
  ``wrapper_prompt``, ``system_prompt``, ``profile_name``.

No glossary logic lives here: adapters only check parameters and call the shared functions.

The smart filter of an automatic glossary (``GlossaryManager._filter_text_for_glossary``) runs in a child interpreter on
every path that reaches it (``filter_in_child``), and on a thread where a child cannot start. The dedupe of a unified
glossary stays on its thread: it lives on RapidFuzz, and a child gets RapidFuzz's pure Python (its C++ module is refused):
2,000 entries 0.7 s on a thread and 16 s in a child, the same list.
"""

from __future__ import annotations

import contextlib
import importlib
import os
from typing import Any, Callable, Iterator, Optional

from glossarion_mobile.job_kinds import INDEXING, owner_method
from glossarion_mobile.job_kinds.translate import check_inputs, record_output_dirs, resolve_workspace_collisions

__all__ = ["FILTER_CHILD_MIN_CHARS", "FILTER_IN_CHILD", "KINDS", "filter_in_child", "glossary_outputs",
           "progress_for_glossary", "run", "run_pair", "run_refine", "run_unified"]


#: The smart filter runs in a child interpreter. False: on the job's thread, like a device where a child cannot start.
FILTER_IN_CHILD = True
#: From this much text. The filter is pure CPU on the whole book, next to the Flet loop. Measured on the emulator (6 vCPU,
#: a real Korean book, threads against a child, the same text and terms): 20,000 characters 0.40 and 0.60 s; 50,000 1.97 and
#: 1.21 s; 169,000 7.3 and 3.7 s, the loop's lateness p95 37 and 2 ms, +32 MB; 746,000 43.5 and 24.9 s, 103 and 1.9 ms,
#: +51 MB; 746,000 with the gender context 114 and 24 s, +69 MB. A child starts in 0.28 s with what it needs.
FILTER_CHILD_MIN_CHARS = 30_000
#: The module a child runs; only a child imports it (``services.glossary_child``).
_FILTER_CHILD = "glossarion_mobile.services.glossary_child"


@contextlib.contextmanager
def filter_in_child(log: Callable[[Any], Any] = print) -> Iterator[None]:
    """A job that may reach ``_filter_text_for_glossary`` (a translation's automatic glossary, Extract Glossary's minimal
    pass): the whole function runs in a child with its own GIL, so none of its sentences compete with the loop, and it
    finishes sooner because each of its checks for a Stop no longer asks the file system. A child is started for each
    filter and released right after: a translation runs for hours after it. Its log lines come as they are printed.

    The result is the shared function's in both places; where a child cannot start, or compares names differently
    (``name_comparison.comparison_modules``), it runs here on the job's thread and says so once. A Stop reaches the child
    (``_check_filter_stop`` asked here at ``STOP_POLL``) and ends it like on a thread."""
    backend = None
    if FILTER_IN_CHILD:
        try:
            import GlossaryManager as backend
        except Exception:  # pragma: no cover - the job itself reports a build without the backend
            backend = None
    real = getattr(backend, "_filter_text_for_glossary", None)
    if not callable(real):
        yield
        return
    from glossarion_mobile.services import interpreters
    from glossarion_mobile.services.name_comparison import comparison_modules

    job = interpreters.JobChildren(log, "Filtering the text", preload=(_FILTER_CHILD,))

    def stopped() -> bool:
        try:
            backend._check_filter_stop()
        except backend._GlossaryFilteringStopped:
            return True
        return False

    def in_child(text: Any, min_frequency: Any = 2, max_sentences: Any = None) -> Any:
        if not isinstance(text, str) or len(text) < FILTER_CHILD_MIN_CHARS:
            return real(text, min_frequency, max_sentences)
        pool = job.pool(1)
        if pool is None:
            return real(text, min_frequency, max_sentences)
        try:
            return pool.call(_FILTER_CHILD, "filter_text", (text, min_frequency, max_sentences, comparison_modules()),
                             should_stop=stopped, live_output=True)
        except (interpreters.InterpreterUnavailable, ImportError) as exc:
            job.fall_back(str(exc))
            return real(text, min_frequency, max_sentences)
        finally:
            job.release()

    backend._filter_text_for_glossary = in_child
    try:
        yield
    finally:
        backend._filter_text_for_glossary = real
        job.close()


def _job_error(message: str) -> Exception:
    from glossarion_mobile.services.jobs import JobError

    return JobError(message)


def _module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise _job_error(f"The shared module '{name}' is not in this build ({exc}).") from exc


def glossary_outputs(owner: Any, files: list) -> list:
    """The book glossary files an extraction wrote (``Glossary/<book>/<book>_glossary.csv|json``),
    for the job's Result card; best effort (nothing is listed when the layout is unknown)."""
    out: list = []
    try:
        from glossary_paths import get_book_glossary_dir
    except Exception:
        return out
    roots = []
    override = os.environ.get("OUTPUT_DIRECTORY") or (getattr(owner, "config", {}) or {}).get("output_directory")
    if override:
        roots.append(os.path.join(os.path.abspath(str(override)), "Glossary"))
    try:
        from app_paths import _get_app_dir

        roots.append(os.path.join(_get_app_dir(), "Glossary"))
    except Exception:
        pass
    bases = []
    for path in files:
        if os.path.splitext(path)[1].lower() in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"):
            base = os.path.basename(os.path.dirname(path)) or "images"
        else:
            base = os.path.splitext(os.path.basename(path))[0]
        if base not in bases:
            bases.append(base)
    for base in bases:
        for root in roots:
            try:
                folder = get_book_glossary_dir(root, base, create=False)
            except Exception:
                continue
            for ext in (".csv", ".json"):
                candidate = os.path.join(folder, f"{base}_glossary{ext}")
                if os.path.isfile(candidate) and candidate not in out:
                    out.append(candidate)
                    break
            if any(os.path.dirname(p) == folder for p in out):
                break
    return out


def run(ctx: Any) -> dict:
    files = resolve_workspace_collisions(ctx, check_inputs(ctx.inputs))  # the desktop selection's rename
    owner = ctx.owner
    extract = owner_method(owner, "run_glossary_extraction_direct")
    owner.selected_files = list(files)
    record_output_dirs(ctx, files)
    ctx.phase("Generating glossary…")
    with filter_in_child():
        result = extract(force_balanced_request_merging=bool(ctx.params.get("force_balanced_request_merging", False)))
    if result is False and not ctx.stop_requested():
        return {"ok": False, "outputs": [], "error": "Glossary extraction stopped (see the log)."}
    return {"ok": None, "outputs": glossary_outputs(owner, files)}


# ---------------------------------------------------------------------------
# glossary_refine
# ---------------------------------------------------------------------------


def progress_for_glossary(glossary_path: str) -> Optional[str]:
    """The ``*_glossary_progress.json`` that belongs to a glossary file, else None (blocking).

    Issue 14: a refinement started without the progress file (the approval card's ✨ Refine, Settings ›
    "Refine a glossary now…", the Glossary Manager) never recorded its entry types as completed, so the
    Glossary Progress refinement rows stayed open. The extractor writes ``<book>_glossary.csv`` and
    ``<book>_glossary_progress.json`` side by side (``Glossary/<book>/``); a workspace's canonical
    ``glossary.csv`` belongs to the book's Glossary folder beside or inside the workspace."""
    if not glossary_path:
        return None
    folder = os.path.dirname(os.path.abspath(str(glossary_path)))
    stem = os.path.splitext(os.path.basename(str(glossary_path)))[0]
    base = stem[: -len("_glossary")] if stem.lower().endswith("_glossary") else ""
    candidates = []
    if base:
        candidates.append(os.path.join(folder, f"{base}_glossary_progress.json"))
    else:
        book = os.path.basename(folder)
        try:
            from glossary_paths import sanitize_glossary_folder_name

            book_dir = sanitize_glossary_folder_name(book)
        except Exception:
            book_dir = book
        candidates += [os.path.join(os.path.dirname(folder), "Glossary", book_dir, f"{book}_glossary_progress.json"),
                       os.path.join(folder, "Glossary", f"{book}_glossary_progress.json")]
    candidates.append(os.path.join(folder, "glossary_progress.json"))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    try:
        matches = [os.path.join(folder, n) for n in os.listdir(folder) if n.lower().endswith("_glossary_progress.json")]
    except OSError:
        matches = []
    return matches[0] if len(matches) == 1 else None


def run_refine(ctx: Any) -> dict:
    params = ctx.params
    glossary_path = os.path.abspath(str(params.get("glossary_path") or (ctx.inputs[0] if ctx.inputs else "")))
    if not glossary_path or not os.path.isfile(glossary_path):
        raise _job_error("No saved glossary file was found for this progress entry.")
    selected = [str(t) for t in (params.get("selected_types") or []) if str(t or "").strip()]
    if not selected:
        raise _job_error("Choose at least one entry type to refine.")
    progress_path = str(params.get("progress_path") or "") or progress_for_glossary(glossary_path)
    source_path = str(params.get("source_path") or "") or None
    target = params.get("target_chunk_count")
    owner = ctx.owner
    gpc = _module("glossary_progress_core")
    planner = getattr(gpc, "plan_manual_glossary_refinement", None)
    runner = getattr(gpc, "run_manual_glossary_refinement", None)
    if not callable(planner) or not (callable(runner) or callable(getattr(owner, "_run_manual_glossary_refinement",
                                                                           None))):
        raise _job_error("This build's engine has no manual glossary refinement runner "
                         "(glossary_progress_core.run_manual_glossary_refinement).")
    from glossarion_mobile.services.library import bind_call

    ctx.phase("Planning the refinement…")
    planned = bind_call(planner, owner, glossary_path=glossary_path, progress_path=progress_path,
                        source_path=source_path, selected_types=list(selected),
                        target_chunk_count=int(target) if target else None, log=ctx.log)
    if not planned:
        raise _job_error("The selected entry type(s) contain no glossary entries.")
    options, plan = planned[0], planned[1]
    if ctx.stop_requested():
        return {"ok": None, "outputs": []}
    ctx.set_output_dir(os.path.dirname(glossary_path))
    ctx.log(f"\n✨ Starting manual glossary refinement: {', '.join(getattr(options, 'selected_types', None) or selected)}")
    ctx.phase("Refining glossary…")
    if callable(runner):
        runner(owner, glossary_path, progress_path, options, plan)
    else:
        owner._run_manual_glossary_refinement(glossary_path, progress_path, options, plan)
    outputs = [p for p in (os.path.splitext(glossary_path)[0] + ".csv", glossary_path) if os.path.isfile(p)]
    return {"ok": None, "outputs": outputs[:1]}


# ---------------------------------------------------------------------------
# unified_glossary
# ---------------------------------------------------------------------------


def _unified_shared_dir(owner: Any, params: Any) -> str:
    """``_unified_glossary_shared_dir``: OUTPUT_DIRECTORY/Glossary, else <app dir>/Glossary."""
    shared = str(params.get("shared_dir") or "")
    if shared:
        return os.path.abspath(shared)
    override = os.environ.get("OUTPUT_DIRECTORY") or (getattr(owner, "config", {}) or {}).get("output_directory")
    if override and str(override).strip():
        return os.path.join(os.path.abspath(str(override)), "Glossary")
    try:
        from app_paths import _get_app_dir

        return os.path.join(_get_app_dir(), "Glossary")
    except Exception:
        return os.path.join(os.getcwd(), "Glossary")


def run_unified(ctx: Any) -> dict:
    ug = _module("unified_glossary")
    rebuild_now = getattr(ug, "rebuild_now", None)
    if not callable(rebuild_now):
        raise _job_error("This build's unified_glossary has no rebuild_now().")
    owner = ctx.owner
    config = getattr(owner, "config", None) or ctx.config or {}
    shared_dir = _unified_shared_dir(owner, ctx.params)
    # The snapshot the desktop's Rebuild Now passes (GlossaryManager_GUI._rebuild_unified_glossary_now).
    settings = _module("glossary_document").unified_rebuild_settings(config, shared_dir)
    ctx.set_output_dir(shared_dir)
    ctx.log(f"📚 Unified glossary: Rebuild Now started ({shared_dir})")
    ctx.phase("Rebuilding the unified glossary…")
    ran = rebuild_now(shared_dir=shared_dir, settings=settings, log=ctx.log)
    if ran is False:
        return {"ok": False, "outputs": [], "error": "A unified glossary rebuild is already running."}
    outputs = []
    try:
        root = ug.unified_root(shared_dir)
        for key in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            csv_path = ug.unified_paths(shared_dir, key)[2]
            if os.path.isfile(csv_path):
                outputs.append(csv_path)
    except Exception:
        outputs = []
    return {"ok": None, "outputs": outputs}


# ---------------------------------------------------------------------------
# parallel_pair
# ---------------------------------------------------------------------------


def _pair_core() -> Any:
    for name in ("parallel_epub_core",):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise _job_error("The shared module 'parallel_epub_core' is not in this build.")


def run_pair(ctx: Any) -> dict:
    params = ctx.params
    selection = dict(params.get("selection") or {})
    for key in ("raw_path", "translated_path", "wrapper_prompt", "system_prompt", "profile_name"):
        if params.get(key):
            selection[key] = str(params[key])
    selection["mapping"] = [dict(item) for item in selection.get("mapping") or [] if isinstance(item, dict)]
    raw_path = str(selection.get("raw_path") or "")
    translated_path = str(selection.get("translated_path") or "")
    missing = [p for p in (raw_path, translated_path) if not os.path.isfile(p)]
    if missing:
        raise _job_error("Saved Parallel EPUB Pair could not be restored; missing: " + ", ".join(missing))
    pec = _pair_core()
    owner = ctx.owner
    extract = owner_method(owner, "run_glossary_extraction_direct")
    config = getattr(owner, "config", None) or ctx.config or {}
    ctx.phase("Reading both EPUBs…")
    try:
        result, skipped = pec.rebuild_parallel_epub_pair_result(selection, config)
    except ValueError as exc:  # "None of the saved HTML filename mappings still exist."
        raise _job_error(str(exc)) from exc
    pair_temp_dir, generated = pec.build_parallel_epub_pair_artifact(result)
    try:
        # The GUI-free half of TranslatorGUI._activate_parallel_epub_pair_source: the state that
        # run_env._parallel_epub_system_prompt_for_file reads, and the paired file as the selection.
        owner._parallel_epub_pair_source = pec.parallel_epub_pair_source_state(
            result, persistent_selection=pec.compact_parallel_epub_selection(result), mapping_sidecar_path="",
            generated_path=generated, pair_temp_dir=pair_temp_dir)
        state = owner._parallel_epub_pair_source
        owner.selected_files = [generated]
        owner.file_path = generated
        ctx.log(f"📚 Parallel EPUB Pair ready: {os.path.basename(state['raw_path'])} ↔ "
                f"{os.path.basename(state['translated_path'])}")
        ctx.log(f"🔗 Mapped {state['pair_count']} HTML file pair(s).")
        if skipped:
            ctx.log(f"⚠️ {skipped} saved mapping(s) could not be restored because the referenced HTML filename no "
                    "longer exists.")
        ctx.phase("Generating glossary…")
        with filter_in_child():
            result = extract(force_balanced_request_merging=bool(params.get("force_balanced_request_merging", False)))
        if result is False and not ctx.stop_requested():
            return {"ok": False, "outputs": [], "error": "Glossary extraction stopped (see the log)."}
        return {"ok": None, "outputs": glossary_outputs(owner, [raw_path])}
    finally:
        try:
            owner._parallel_epub_pair_source = None
        except Exception:
            pass
        pair_temp_dir.cleanup()


KINDS = {
    "extract_glossary": {"verb": INDEXING, "icon": "SPELLCHECK", "stop_kind": "glossary", "run": run},
    "glossary_refine": {"verb": "Refining glossary", "icon": "AUTO_FIX_HIGH", "stop_kind": "glossary",
                        "run": run_refine},
    "unified_glossary": {"verb": "Rebuilding unified glossary", "icon": "MERGE_TYPE", "stop_kind": "glossary",
                         "run": run_unified, "resumable": False},
    "parallel_pair": {"verb": "Extracting pair glossary", "icon": "COMPARE_ARROWS", "stop_kind": "glossary",
                      "run": run_pair},
}


def kind_ready(kind: str) -> Optional[str]:
    """None when ``kind``'s shared modules import, else why not (for disabled reasons)."""
    needs = {"glossary_refine": ("glossary_progress_core",), "unified_glossary": ("unified_glossary",),
             "parallel_pair": ("parallel_epub_core", "extract_glossary_from_epub")}.get(kind, ())
    for name in needs:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            return f"{name} is not available ({exc.__class__.__name__})"
    return None
