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
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, Callable, Iterator, Optional

from glossarion_mobile.job_kinds import compiled_outputs, owner_method, result_fields

__all__ = ["GLOSSARY_REVIEW_QUESTION", "KINDS", "NOT_COMPLETED", "SUPPORTS_GLOSSARY_REVIEW", "app_owned_roots",
           "check_inputs", "follow_input_rename", "glossary_review_gate", "is_app_owned", "record_output_dirs",
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
