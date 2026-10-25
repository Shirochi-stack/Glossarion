"""TRANSLATE: translate input files with the owner's settings (desktop "Run Translation").

Calls ``owner._prepare_translation_run(files)`` and
``owner._translation_worker(request)``, the desktop ``run_translation_thread``
body moved into ``translation_pipeline.TranslationPipelineMixin``. ZIP/HTML to
EPUB preparation, the automatic glossary pass (the effective glossary mode of
the config snapshot), chunking, header translation, EPUB compilation and the
post-translation QA scan therefore follow the owner exactly as on desktop, and
a resumed job continues from ``translation_progress.json``.

Before that, an app-owned input copy (Inbox, Library/Raw) whose workspace already belongs to
another format is renamed exactly like the desktop file selection does
(``owner._rename_input_for_existing_workspace_collision``, moved into ``run_env.RunEnvMixin``:
``Novel.epub`` next to a ``Novel`` workspace of ``Novel.pdf`` becomes ``Novel_EPUB.epub``), so the
two books never share a workspace; the Library registry follows the rename and the job result
lists it (``renamed_inputs``). User originals are never renamed.

The adapter only validates the inputs, records each input's output folder
(``owner._resolve_translation_output_dir``: checkpointed for recovery and read
by ``ProgressWatcher``), maps the worker's outcome to the job state (False:
Failed, or Stopped after a Stop) and lists the compiled files afterwards.

The post-translation QA scan (hotfix 1 issue 6): the worker decides it exactly as on desktop
(scanning phase on, the run completed and was not stopped, no CSV/JSON/image/SDLXLIFF/subtitle
input, not metadata-only, not the refinement / image output mode, not a Direct Text run) and asks
the front end with ``self._ui_request('trigger_qa_scan')``, which the GUI-free default only logs
as unavailable. ``post_translation_qa_trigger`` catches that request on this job's owner (as
``glossary_review_gate`` wraps the glossary pre-pass) and, once the worker has returned (the
desktop scan starts on the GUI thread after the worker emitted it), ``post_translation_qa`` runs
the desktop handler's scan in the same job (``qa.run_scan_phase``); the job ends after it.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, Callable, Iterator, Optional

from glossarion_mobile.job_kinds import compiled_outputs, owner_method, result_fields

__all__ = ["GLOSSARY_REVIEW_QUESTION", "KINDS", "NOT_COMPLETED", "QA_SCAN_TRIGGER", "SUPPORTS_GLOSSARY_REVIEW",
           "app_owned_roots", "check_inputs", "follow_input_rename", "glossary_review_gate", "is_app_owned",
           "post_translation_qa", "post_translation_qa_trigger", "record_output_dirs",
           "resolve_workspace_collisions", "run", "run_translation"]


def _renamed_sibling(path: str) -> str:
    """A resumed job's input that an earlier run renamed for a workspace collision
    (``<stem>_<FORMAT><ext>``, ``output_workspace``'s rule), else ``path``."""
    try:
        from output_workspace import source_format_label

        label = source_format_label(path)
    except Exception:
        return path
    if not label:
        return path
    stem, ext = os.path.splitext(path)
    candidate = f"{stem}_{label}{ext}"
    return candidate if os.path.isfile(candidate) else path


def check_inputs(paths: Any) -> list:
    from glossarion_mobile.services.jobs import JobError

    files = [os.path.abspath(os.fspath(p)) for p in (paths or ()) if p]
    if not files:
        raise JobError("Nothing to translate: no input file.")
    files = [p if os.path.exists(p) else _renamed_sibling(p) for p in files]
    missing = [os.path.basename(p) for p in files if not os.path.exists(p)]
    if missing:
        raise JobError(f"File not found: {', '.join(missing)}")
    return files


def app_owned_roots() -> list:
    """Folders whose files are the app's own copies (renaming them never touches a user file):
    ``<data>/Inbox`` and ``Library/Raw``."""
    roots = []
    data = os.environ.get("GLOSSARION_DATA_DIR", "").strip()
    if data:
        roots.append(os.path.join(data, "Inbox"))
    try:
        from library_core import library_root_path

        roots.append(os.path.join(library_root_path(), "Raw"))
    except Exception:
        pass
    return [os.path.normcase(os.path.abspath(r)) for r in roots]


def is_app_owned(path: str, roots: Any = None) -> bool:
    key = os.path.normcase(os.path.abspath(path))
    for root in (app_owned_roots() if roots is None else roots):
        if key.startswith(root.rstrip("\\/") + os.sep):
            return True
    return False


def follow_input_rename(old: str, new: str) -> None:
    """The Library follows a renamed Library/Raw copy: the raw-inputs registry entry and the
    ``library_origins.txt`` raw / pairs names (``library_core``'s registry files)."""
    try:
        import library_core as lc
    except Exception:
        return
    try:
        registered = {os.path.normcase(os.path.abspath(p)) for p in lc.load_library_raw_inputs()}
        if os.path.normcase(os.path.abspath(old)) in registered:
            lc.remove_library_raw_input(old)
            lc.record_library_raw_input(new)
    except Exception:
        pass
    try:
        origins = lc._load_origins()
        old_name, new_name = os.path.basename(old), os.path.basename(new)
        raw = origins.get("raw") if isinstance(origins, dict) else None
        changed = False
        if isinstance(raw, dict) and old_name in raw and new_name not in raw:
            raw[new_name] = raw.pop(old_name)
            changed = True
        pairs = origins.get("pairs") if isinstance(origins, dict) else None
        if isinstance(pairs, dict):
            for key, value in list(pairs.items()):
                if value == old_name:
                    pairs[key] = new_name
                    changed = True
        if changed:
            lc._save_origins(origins)
    except Exception:
        pass


def resolve_workspace_collisions(ctx: Any, files: list) -> list:
    """The desktop selection's workspace-collision rename (``owner._rename_input_for_existing_
    workspace_collision``) for the app's own copies; returns the (possibly renamed) inputs."""
    rename = getattr(ctx.owner, "_rename_input_for_existing_workspace_collision", None)
    if not callable(rename):
        return list(files)
    roots = app_owned_roots()
    out: list = []
    renamed: dict = {}
    for path in files:
        new = path
        if is_app_owned(path, roots):
            try:
                new = os.path.abspath(rename(path) or path)
            except Exception as exc:
                ctx.log(f"⚠️ Workspace collision check failed for {os.path.basename(path)}: {exc}")
                new = path
        if new != path:
            renamed[path] = new
            follow_input_rename(path, new)
        out.append(new)
    if renamed:
        try:
            ctx.set_result(renamed_inputs=renamed)
        except Exception:
            pass
    return out


def record_output_dirs(ctx: Any, files: list) -> dict:
    """``{input: output folder}`` from the owner's resolver (empty when it has none)."""
    resolve = getattr(ctx.owner, "_resolve_translation_output_dir", None)
    mapping: dict = {}
    if callable(resolve):
        for path in files:
            try:
                folder = resolve(path)
            except Exception:
                folder = None
            if folder:
                mapping[path] = os.path.abspath(os.fspath(folder))
    if mapping:
        ctx.set_output_dirs(mapping)
    return mapping


#: The worker reported an unfinished run (``_translation_worker`` returned False) without a Stop.
NOT_COMPLETED = "The translation did not complete (see the log)."

#: The worker's post-translation scan request (``_ui_request`` kind; desktop ``trigger_qa_scan_signal``).
QA_SCAN_TRIGGER = "trigger_qa_scan"


@contextlib.contextmanager
def post_translation_qa_trigger(ctx: Any) -> Iterator[list]:
    """While the worker runs, this job's owner answers ``_ui_request('trigger_qa_scan')`` itself.

    The request is recorded (the yielded list) instead of reaching the GUI-free default, which would log
    that no front end runs the scan; every other ``_ui_request`` kind goes to the owner's own method.
    The owner instance gets the wrapper (the shared method is unchanged) and afterwards what it had before."""
    owner = ctx.owner
    original = getattr(owner, "_ui_request", None)
    own = getattr(owner, "__dict__", {})
    had_own, previous = "_ui_request" in own, own.get("_ui_request")
    requests: list = []
    if not callable(original):
        yield requests
        return

    def ui_request(kind: Any, *args: Any, **data: Any) -> Any:
        if kind == QA_SCAN_TRIGGER:
            requests.append(kind)
            return None
        return original(kind, *args, **data)

    wrapped = False
    try:
        owner._ui_request = ui_request
        wrapped = True
    except Exception:
        ctx.log("⚠️ The post-translation QA scan could not attach to the translation run")
    try:
        yield requests
    finally:
        if wrapped:
            try:
                if had_own:
                    owner._ui_request = previous
                else:
                    del owner._ui_request
            except Exception:
                pass


def post_translation_qa(ctx: Any, files: list, mapping: Optional[dict] = None, outputs: Any = ()) -> list:
    """The scan the worker asked for, on the translated output (``qa.run_scan_phase``); returns its reports.

    The sources are the owner's selection (the desktop scan reads ``selected_files``), else the job's files;
    each one's folder is the output folder this job recorded (``record_output_dirs``), else the owner's
    resolver. ``outputs`` (the compiled files) go into the job's outputs first, so they stay ahead of the
    reports the scan adds."""
    from glossarion_mobile.job_kinds import qa

    owner = ctx.owner
    selected = [p for p in (getattr(owner, "selected_files", None) or ()) if p] or list(files)
    sources = [os.path.abspath(os.fspath(p)) for p in selected]
    known = {os.path.normcase(os.path.abspath(os.fspath(k))): v for k, v in (mapping or {}).items() if v}
    resolve = getattr(owner, "_resolve_translation_output_dir", None)

    def folder_for(source: str) -> Optional[str]:
        folder = known.get(os.path.normcase(source))
        if not folder and callable(resolve):
            try:
                folder = resolve(source)
            except Exception:
                folder = None
        return os.path.abspath(os.fspath(folder)) if folder else None

    compiled = [p for p in (outputs or ()) if p]
    if compiled:
        ctx.add_outputs(compiled)
    return qa.run_scan_phase(ctx, sources, folder_for)


def run_translation(ctx: Any, files: list) -> dict:
    """The shared prepare + worker pair; returns ``{"ok", "outputs", "error"}``.

    ``_translation_worker`` returns the run's outcome (``run_translation_direct``'s result,
    False when it ended early); False after a Stop is a stopped run (``ok`` None: Stopped),
    otherwise a failure. A Stop tapped before the set-up's "Reset stop flags" block (the job
    was starting) survives only in the job's own latch (``ctx.stop_requested()``, which the
    reset does not touch), so the worker is not started then: on desktop the set-up runs
    inside the Run click and no Stop can come before the reset.
    """
    owner = ctx.owner
    prepare = owner_method(owner, "_prepare_translation_run")
    worker = owner_method(owner, "_translation_worker")
    mapping = record_output_dirs(ctx, files)
    ctx.phase("Translating")
    request = prepare(files)
    if request is None or request is False:
        if ctx.stop_requested():
            return {"ok": None, "outputs": []}
        return {"ok": False, "outputs": [], "error": "The translation did not start (see the log)."}
    request_ok, _outs, request_error = result_fields(request) if isinstance(request, dict) else (None, [], None)
    if request_ok is False:
        return {"ok": False, "outputs": [], "error": request_error or "The translation did not start (see the log)."}
    if ctx.stop_requested():
        ctx.log("⏹️ Translation stopped before it started")
        return {"ok": None, "outputs": []}
    with post_translation_qa_trigger(ctx) as qa_requests:
        outcome = worker(request)
    ok, outputs, error = result_fields(outcome)
    if ok is False and ctx.stop_requested():
        ok, error = None, None  # stopped: the run ends as Stopped (Resume continues it)
    elif ok is False and not error:
        error = NOT_COMPLETED
    folders = list(mapping.values()) or [ctx.output_dir]
    for path in compiled_outputs(folders):
        if path not in outputs:
            outputs.append(path)
    if qa_requests and ok is not False:
        # the desktop scanning phase: the scan the worker asked for, then the job ends
        for path in post_translation_qa(ctx, files, mapping, outputs):
            if path not in outputs:
                outputs.append(path)
    return {"ok": ok, "outputs": outputs, "error": error}


#: Library › Translate… "Review glossary before translating" (UI_SPEC §3.10, Appendix C "Book-origin
#: glossary gate"): ``params["review_glossary"]`` pauses the job after its automatic glossary phase.
SUPPORTS_GLOSSARY_REVIEW = True
#: The blocking question the gate asks (``JobContext.ask(kind, path=...)``): the Book page's approval
#: sheet, Jobs › job and the "Glossary ready: review needed" notification answer it.
GLOSSARY_REVIEW_QUESTION = "glossary_approval"
GLOSSARY_REVIEW_DECLINED = "Translation cancelled at the glossary review step"


@contextlib.contextmanager
def glossary_review_gate(ctx: Any) -> Iterator[Optional[Callable[[str], bool]]]:
    """``params["review_glossary"]``: ask before translating with the generated glossary.

    Reuses the shared approval points instead of copying them; the desktop never sets the param:

    * the backend's in-run glossary phase (TransateKRtoEN's Minimal / Vision pass) asks through
      ``TransateKRtoEN.set_direct_text_glossary_approval_callback``, installed for this job only;
    * the pipeline's Balanced / Full pre-pass (``run_translation_direct``) loads the file it generated
      with ``_auto_load_glossary_after_extraction``; this job's owner instance gets a wrapper that
      asks once that returns (the shared method itself is unchanged).

    ■ No stops the job (Stopped: the generated glossary and the workspace stay, Resume asks again);
    ✏️ Edit happens in the app while the job waits. Yields the approval callable (None: no gate)."""
    if not (ctx.params or {}).get("review_glossary"):
        yield None
        return
    asked: list = []

    def approve(path: Any) -> bool:
        path = os.path.abspath(os.fspath(path)) if path else ""
        asked.append(path)
        ctx.log("⏸️ Glossary generation is complete; waiting for your review before translation")
        try:
            accepted = bool(ctx.ask(GLOSSARY_REVIEW_QUESTION, path=path, default=False))
        except Exception as exc:
            ctx.log(f"⚠️ Could not ask for the glossary review: {exc}")
            accepted = False
        if not accepted:
            ctx.log(f"⏹️ {GLOSSARY_REVIEW_DECLINED}")
            if not ctx.stop_requested():
                ctx.request_stop(GLOSSARY_REVIEW_DECLINED)
        return accepted

    owner = ctx.owner
    original = getattr(owner, "_auto_load_glossary_after_extraction", None)
    wrapped = False
    if callable(original):
        def auto_load_then_review(*args: Any, **kwargs: Any) -> Any:
            generated = original(*args, **kwargs)
            approve(generated or getattr(owner, "manual_glossary_path", "") or "")
            return generated

        try:
            owner._auto_load_glossary_after_extraction = auto_load_then_review
            wrapped = True
        except Exception:
            ctx.log("⚠️ The glossary review gate could not attach to the glossary pre-pass")
    backend = None
    try:
        import TransateKRtoEN as backend  # the backend's own glossary phase

        backend.set_direct_text_glossary_approval_callback(approve)
    except Exception:
        backend = None
    try:
        yield approve
    finally:
        if wrapped:
            try:
                del owner._auto_load_glossary_after_extraction
            except Exception:
                pass
        if backend is not None and getattr(backend, "_direct_text_glossary_approval_callback", None) is approve:
            backend.set_direct_text_glossary_approval_callback(None)


def run(ctx: Any) -> dict:
    files = resolve_workspace_collisions(ctx, check_inputs(ctx.inputs))
    with glossary_review_gate(ctx):
        return run_translation(ctx, files)


KINDS = {
    "translate": {"verb": "Translating", "icon": "TRANSLATE", "stop_kind": "translation", "run": run},
}
