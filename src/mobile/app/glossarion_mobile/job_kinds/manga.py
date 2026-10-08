"""MANGA / MANGA_STEP: the manga translator's batch run and editor steps (Tools › Manga, UI_SPEC §4.6).

``manga`` — Start (and Settings › Generate glossary). ``manga_runner.HeadlessMangaRunner`` is the
desktop ``MangaTranslationTab`` without Qt: its run replays the Start click (range check, the
automatic OCR export, imported-OCR matching) and ``_start_translation_heavy`` →
``_translation_worker`` with the ``process_image`` calls, the glossary workflow and the CBZ
jobs, on the job thread, with the job's ``HeadlessOwner`` as the duck-typed ``main_gui`` (plan
§1: built only there, under JOB_LOCK, from the config snapshot)::

    runner = HeadlessMangaRunner(owner, host=ctx.host, files=..., image_range=..., ...)
    summary = runner.run()   # {ok, completed, failed, total, outputs, cbz_paths, stopped, error, ...}

The adapter only bridges the JobService: page progress becomes ``progress`` events, a Stop
reaches ``runner.request_stop`` (graceful / immediate as the JobService decided, a second Stop
forces), the translated pages and CBZ files become the job's outputs. An OCR JSON imported
earlier (``params["imported_ocr"]``) is handed to the runner first with
``manga_env.import_ocr_session``, as the desktop's Import OCR does for the next Start.

``manga_step`` — the editor's Detect / Clean / Recognize / Translate / Translate all / Save &
Update Overlay buttons, the per-box OCR / Translate / Clean actions and Import OCR (which renders
imported translations): ``manga_editor_core.MangaEditorSession`` methods called on the job
thread with the job's owner (``session.translate(image, owner=owner)``). The session lives in
the app (the editor screen owns it; ``params["session"]`` is its registry token), so the screen
sees the result in the session's page snapshot when the job ends.

Before either kind runs the manga code (mobile, ``services.manga``): the job's config snapshot
gets the phone defaults Settings shows and the Azure credential mitigation
(``prepare_run``), ``OUTPUT_DIRECTORY`` is hidden for the job so pages are written next to their
source like a desktop without an output folder (``hide_output_override``; the job's process
state puts it back), and the registered models the run loads that are not on the device yet
are downloaded with progress, verified and resumable, and stopped by the job's Stop
(``ensure_run_models``, ``params["model_kinds"]``).

Neither kind is resumable: a stopped batch is started again (the runner's own rules decide what
is redone); an editor step is a click.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Callable, Mapping, Optional

__all__ = ["KINDS", "run_batch", "run_step"]

_STOP_POLL = 0.25
_STEP_PHASES = {"detect": "Detecting text", "clean": "Cleaning", "recognize": "Recognizing text",
                "translate": "Translating", "translate_all": "Translating pages",
                "render": "Rendering overlay", "ocr_box": "Recognizing box", "translate_box": "Translating box",
                "clean_box": "Cleaning box", "import_ocr": "Importing OCR"}


def _job_error(message: str) -> Exception:
    from glossarion_mobile.services.jobs import JobError

    return JobError(message)


def _stop_mode(ctx: Any) -> Optional[str]:
    """The JobService stop mode of this job: ``graceful`` / ``immediate`` / ``force`` (None: running)."""
    if not ctx.stop_requested():
        return None
    job = getattr(ctx, "_job", None)
    mode = getattr(job, "stop_mode", None) if job is not None else None
    if mode:
        return str(mode)
    host = getattr(ctx, "host", None)
    checker = getattr(host, "is_graceful_stop", None)
    try:
        return "graceful" if callable(checker) and checker() else "immediate"
    except Exception:
        return "immediate"


class _StopBridge:
    """Watches the job's stop latch and forwards each new stop mode to ``on_stop(mode)``
    (graceful → immediate → force escalate; a mode is forwarded once)."""

    _ORDER = {"graceful": 0, "immediate": 1, "force": 2}

    def __init__(self, ctx: Any, on_stop: Callable[[str], Any]) -> None:
        self.ctx = ctx
        self.on_stop = on_stop
        self.sent: Optional[str] = None
        self._done = threading.Event()
        self.thread = threading.Thread(target=self._watch, name="gl-manga-stop", daemon=True)

    def __enter__(self) -> "_StopBridge":
        self.thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._done.set()
        self.thread.join(timeout=2.0)

    def poll(self) -> None:
        mode = _stop_mode(self.ctx)
        if mode is None:
            return
        if self.sent is not None and self._ORDER.get(mode, 1) <= self._ORDER.get(self.sent, 1):
            return
        self.sent = mode
        try:
            self.on_stop(mode)
        except Exception:
            pass

    def _watch(self) -> None:
        while not self._done.wait(_STOP_POLL):
            self.poll()


def _progress_emitter(ctx: Any, total: int) -> Callable[..., None]:
    state = {"completed": 0, "failed": 0, "total": total}

    def progress(current: Any = None, total_now: Any = None, label: str = "", failed: Any = None, **_: Any) -> None:
        """``HeadlessMangaRunner`` progress: ``(current, total, label=status, failed=n)``."""
        for key, value in (("completed", current), ("total", total_now), ("failed", failed)):
            if value is not None:
                try:
                    state[key] = int(value)
                except (TypeError, ValueError):
                    pass
        text = str(label or "")
        ctx.host.emit("progress", total=state["total"], completed=state["completed"], failed=state["failed"],
                      label=text or f"Page {min(state['completed'] + 1, max(1, state['total']))}/{state['total']}")

    return progress


def _reuse_imported_ocr(ctx: Any, runner: Any, path: str, files: list) -> int:
    """The desktop's batch OCR import (``manga_env.import_ocr_session``) on the runner: Start then
    reuses the imported OCR for the matching pages ("Imported OCR will be reused for N/M pages").
    A missing or unreadable file is logged and the run goes on without it."""
    from glossarion_mobile.services import manga as svc

    importer = svc.core_attr("manga_env", "import_ocr_session")
    name = os.path.basename(path)
    if not callable(importer) or not os.path.isfile(path):
        ctx.log(f"⚠️ Imported OCR not reused: {name} is not available")
        return 0
    try:
        matches = importer(runner, path, files=list(files))
    except Exception as exc:
        ctx.log(f"⚠️ Imported OCR not reused ({name}): {exc}")
        return 0
    if not matches:
        ctx.log(f"⚠️ Imported OCR not reused: {name} matches none of these pages")
    return len(matches or {})


def _prepare(ctx: Any) -> bool:
    """Mobile (``services.manga``): the job's config snapshot gets the phone defaults Settings shows
    and the Azure credential mitigation (``prepare_run``), and ``OUTPUT_DIRECTORY`` is hidden for
    the job (``hide_output_override``). True when the output override is hidden."""
    from glossarion_mobile.services import manga as svc

    config = getattr(ctx.owner, "config", None)
    if isinstance(config, dict):
        for note in svc.prepare_run(config):
            ctx.log(note)
    return svc.hide_output_override()


def _model_kinds(params: Mapping[str, Any], default: Optional[tuple]) -> Optional[tuple]:
    kinds = params.get("model_kinds")
    if kinds is None:
        return default
    return tuple(str(kind) for kind in kinds)


def _download_models(ctx: Any, mobile: bool, kinds: Optional[tuple], report: Callable[[str, int], Any]) -> bool:
    """Mobile: the registered models the run loads that are not on the device yet, downloaded
    before it starts (``services.manga.ensure_run_models``: verified, resumable, the job's Stop
    cancels and keeps the partial file). False when the job was stopped meanwhile. A failed
    download fails the job (a run without its model would stall on every page: desktop bug 3 in
    tests/parity/DISCREPANCIES.md "U8 Manga run env"), and so does a download cancelled by
    anything but the job's own Stop (the Cancel of a model row: ``manga_models.cancel`` stops
    every download of that model), so the job never ends "done" with nothing done."""
    from glossarion_mobile.services import manga as svc

    if not mobile or kinds == ():
        return True
    config = getattr(ctx.owner, "config", None)
    models = svc.core("manga_models")
    if models is None or not isinstance(config, dict):
        return True
    cancelled = getattr(models, "DownloadCancelled", None)
    try:
        svc.ensure_run_models(config, kinds=kinds, cancel=ctx.stop_requested, on_progress=report, log_fn=ctx.log)
    except Exception as exc:
        if cancelled is not None and isinstance(exc, cancelled):
            if ctx.stop_requested():
                ctx.log(f"⏹️ Model download stopped; it resumes on the next run ({exc})")
                return False
            raise _job_error("The model download was cancelled; start again to resume it") from exc
        raise _job_error(f"A model this run needs could not be downloaded: {exc}") from exc
    return True


def _output_folder(outputs: list, fallback: str) -> Optional[str]:
    """The job's output folder: the pages' folder, or the folder they all share."""
    folders = sorted({os.path.dirname(p) for p in outputs})
    if len(folders) == 1:
        return folders[0]
    if folders:
        try:
            return os.path.commonpath(folders)
        except ValueError:
            return fallback or folders[0]
    return fallback or None


def run_batch(ctx: Any) -> dict:
    from glossarion_mobile.services import manga as svc

    params = dict(ctx.params or {})
    files = [os.fspath(p) for p in (params.get("files") or ctx.inputs or ()) if p]
    if not files:
        raise _job_error("No images to process.")
    missing = [os.path.basename(p) for p in files if not os.path.isfile(p)]
    if missing:
        raise _job_error(f"Missing image(s): {', '.join(missing[:5])}")
    runner_module = svc.core("manga_runner")
    runner_cls = getattr(runner_module, "HeadlessMangaRunner", None) if runner_module is not None else None
    if runner_cls is None:
        raise _job_error("This build has no shared manga runner (manga_runner).")
    run_files = list(params.get("run_files") or files)
    progress = _progress_emitter(ctx, len(run_files))
    session = svc.editor_session(params.get("editor_session"))
    state_manager = getattr(session, "image_state_manager", None)
    output_root = str(params.get("output_root") or "")
    ctx.phase("Preparing manga translator")
    progress(0, len(run_files))
    hidden = _prepare(ctx)
    glossary_only = bool(params.get("glossary_only"))
    if not _download_models(ctx, hidden, _model_kinds(params, ("detector",) if glossary_only else None),
                            lambda title, pct: progress(0, len(run_files), label=f"Downloading {title} · {pct}%")):
        return {"ok": None, "outputs": []}
    if ctx.stop_requested():
        return {"ok": None, "outputs": []}
    try:
        runner = runner_cls(
            ctx.owner, host=ctx.host, files=files, image_range=params.get("image_range") or "",
            folder_roots=params.get("folder_roots") or [], split_first_level=params.get("split_first_level"),
            skipped=params.get("skipped") or [], cbz_jobs=params.get("cbz_jobs") or {},
            cbz_image_to_job=params.get("cbz_image_to_job") or {}, glossary_only=glossary_only,
            progress=progress, output_root=None if hidden else (output_root or None),
            image_state_manager=state_manager)
    except Exception as exc:
        raise _job_error(f"The manga translator could not start: {exc}") from exc
    imported = str(params.get("imported_ocr") or "")
    if imported:
        _reuse_imported_ocr(ctx, runner, imported, run_files)

    def on_stop(mode: str) -> None:
        if mode == "force":
            runner.request_stop(force=True)
        else:
            runner.request_stop(graceful=mode == "graceful")

    error_cls = getattr(runner_module, "MangaRunError", None)
    with _StopBridge(ctx, on_stop):
        ctx.phase("Generating glossary" if params.get("glossary_only") else "Translating pages")
        try:
            summary = runner.run()
        except Exception as exc:
            if error_cls is not None and isinstance(exc, error_cls):
                raise _job_error(str(exc)) from exc
            raise
    summary = dict(summary or {})
    outputs = [os.fspath(p) for p in (summary.get("outputs") or ()) if p]
    cbz = [os.fspath(p) for p in (summary.get("cbz_paths") or ()) if p]
    glossary = str(summary.get("glossary_path") or "")
    ctx.set_result(manga_outputs=outputs, manga_cbz=cbz, manga_completed=int(summary.get("completed") or 0),
                   manga_failed=int(summary.get("failed") or 0), manga_total=int(summary.get("total") or 0),
                   manga_glossary_path=glossary, manga_glossary_only=bool(summary.get("glossary_only")),
                   manga_stopped=bool(summary.get("stopped")))
    all_outputs = outputs + [p for p in cbz if p not in outputs]
    if glossary and os.path.isfile(glossary):
        all_outputs.append(glossary)
    if all_outputs:
        ctx.add_outputs(all_outputs)
        ctx.set_output_dir(_output_folder(outputs, "" if hidden else output_root))
    progress(int(summary.get("completed") or 0), int(summary.get("total") or len(run_files)),
             failed=summary.get("failed"))
    if summary.get("stopped") or ctx.stop_requested():
        return {"ok": None, "outputs": all_outputs}
    if summary.get("error") or summary.get("ok") is False:
        return {"ok": False, "outputs": all_outputs,
                "error": str(summary.get("error") or f"{int(summary.get('failed') or 0)} page(s) failed")}
    return {"ok": True, "outputs": all_outputs}


# ---------------------------------------------------------------------------
# Editor steps
# ---------------------------------------------------------------------------


def _call_step(session: Any, step: str, params: Mapping[str, Any], owner: Any) -> Any:
    image = params.get("image")
    index = params.get("index")
    if step == "detect":
        return session.detect(image, owner=owner)
    if step == "clean":
        return session.clean(image, owner=owner)
    if step == "recognize":
        return session.recognize(image, owner=owner)
    if step == "translate":
        return session.translate(image, owner=owner)
    if step == "translate_all":
        return session.translate_all(params.get("images") or None, owner=owner)
    if step == "render":
        return session.save_and_update_overlay(image, owner=owner)
    if step == "import_ocr":
        return session.import_ocr(params.get("path"), params.get("images") or None, owner=owner, render=True)
    current = session.current_page
    if image and (not current or os.path.normcase(os.path.abspath(current)) != os.path.normcase(os.path.abspath(image))):
        session.open_page(image)  # a per-box action runs on the step's page
    if step == "ocr_box":
        return session.ocr_box(int(index), owner=owner)
    if step == "translate_box":
        return session.translate_box(int(index), owner=owner)
    if step == "clean_box":
        return session.clean_box(int(index), owner=owner)
    raise _job_error(f"Unknown manga editor step: {step}")


def run_step(ctx: Any) -> dict:
    from glossarion_mobile.services import manga as svc

    params = dict(ctx.params or {})
    step = str(params.get("step") or "")
    if step not in svc.STEPS:
        raise _job_error(f"Unknown manga editor step: {step or '(none)'}")
    session = svc.editor_session(params.get("session"))
    if session is None:
        raise _job_error("The editor session is gone (the app was restarted): open the page in the editor again.")
    images = [os.fspath(p) for p in (params.get("images") or [params.get("image")]) if p]
    missing = [os.path.basename(p) for p in images if not os.path.isfile(p)]
    if missing:
        raise _job_error(f"Missing image(s): {', '.join(missing[:5])}")
    if step == "import_ocr" and not os.path.isfile(str(params.get("path") or "")):
        raise _job_error("The OCR file is missing.")
    mobile = _prepare(ctx)
    if not _download_models(ctx, mobile, _model_kinds(params, ()),
                            lambda title, pct: ctx.phase(f"Downloading {title} · {pct}%")):
        return {"ok": None, "outputs": []}
    if ctx.stop_requested():
        return {"ok": None, "outputs": []}
    ctx.phase(_STEP_PHASES.get(step, step))
    previous_log = getattr(session, "_log_callback", None)

    def log_line(text: Any, level: str = "info") -> None:
        ctx.log(text)
        if callable(previous_log):
            try:
                previous_log(text, level)
            except Exception:
                pass

    session._log_callback = log_line
    try:
        with _StopBridge(ctx, lambda mode: session.stop(force=mode != "graceful")):
            result = _call_step(session, step, params, ctx.owner)
    except (ValueError, IndexError) as exc:
        raise _job_error(str(exc)) from exc
    finally:
        session._log_callback = previous_log
    snapshot = {}
    try:
        snapshot = session.page_snapshot(params.get("image"))
    except Exception:
        snapshot = {}
    rendered = str(snapshot.get("rendered_path") or snapshot.get("translated_path") or "")
    outputs = [p for p in (rendered, str(snapshot.get("cleaned_path") or "")) if p and os.path.isfile(p)]
    if step == "translate_all" and isinstance(result, Mapping):
        outputs = [str(p) for p in result.values() if p and os.path.isfile(str(p))]
    summary: dict = {"manga_step": step, "manga_image": params.get("image"), "manga_step_outputs": outputs,
                     "manga_revision": int(getattr(session, "output_revision", 0) or 0)}
    if step == "import_ocr" and isinstance(result, Mapping):
        summary["manga_import"] = {"matched": int(result.get("matched") or 0), "files": int(result.get("files") or 0),
                                   "translated_regions": int(result.get("translated_regions") or 0)}
    if step in ("ocr_box", "translate_box") and isinstance(result, str):
        summary["manga_box_text"] = result
    ctx.set_result(**summary)
    ctx.host.emit("manga_step", step=step, image=params.get("image"))
    if outputs:
        ctx.add_outputs(outputs)
    if ctx.stop_requested():
        return {"ok": None, "outputs": outputs}
    return {"ok": True, "outputs": outputs}


KINDS = {
    "manga": {"verb": "Translating manga", "icon": "AUTO_STORIES", "stop_kind": "translation", "run": run_batch,
              "resumable": False},
    "manga_step": {"verb": "Manga editor", "icon": "BRUSH", "stop_kind": "translation", "run": run_step,
                   "resumable": False},
}
