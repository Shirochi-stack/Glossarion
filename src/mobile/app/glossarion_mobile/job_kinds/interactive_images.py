"""INTERACTIVE_IMAGES: the Interactive Images step on a translated book (desktop Progress Manager ›
Interactive Images › ▶ Build pending / the row menu's Build / Retry; mobile Book page › Images).

Runs ``interactive_images_runner.run_interactive_images`` on the job's owner config, exactly as the
desktop worker thread does: per chapter plan (text model) -> place with placeholder sources -> text
check -> images (image model, with reference images) -> text check -> atomic write -> progress entry.
A failed or rejected chapter keeps its file; the translation output is never touched otherwise.
Recompiling afterwards (``interactive_images_recompile_after``) uses the owner's own compile methods
(``interactive_images_runner.owner_recompiler``), like the desktop.

inputs: the book's output folder. params: ``keys`` (output file names; absent = every chapter that needs
a build). With ``css_request`` instead (Styles › New from prompt): the profile's CSS prompt is sent to the
text model (``interactive_images_runner.generate_stylesheet``) and the scoped stylesheet is the job's result
(``interactive_css``); no book is needed.
"""

from __future__ import annotations

import os
import threading
from typing import Any

__all__ = ["KINDS", "run"]

ICON = "IMAGE"


class _CtxStop:
    """A stop event that follows the job's Stop (checked between requests by the shared runner)."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self._event = threading.Event()

    def is_set(self) -> bool:
        return self._event.is_set() or bool(self.ctx.stop_requested())

    def set(self) -> None:
        self._event.set()


def run(ctx: Any) -> dict:
    from glossarion_mobile.services.jobs import JobError

    try:
        import interactive_images_core as core
        import interactive_images_progress_core as ipc
        import interactive_images_runner as runner
    except ImportError as exc:
        raise JobError(f"Interactive Images is not in this build ({exc}).") from exc
    owner = ctx.owner
    config = getattr(owner, "config", None) or ctx.config
    request = str(ctx.params.get("css_request") or "").strip()
    if request:
        # Styles › ✨ New from prompt: the profile's CSS prompt on the text model (desktop
        # InteractiveStylesDialog._new_from_prompt); the screen stores the reply as a user preset
        ctx.phase("Writing the style")
        try:
            css, warnings = runner.generate_stylesheet(config, request, output_dir=ctx.output_dir)
        except Exception as exc:
            raise JobError(f"The text model did not return a stylesheet: {exc}") from exc
        if not str(css or "").strip():
            raise JobError("The text model returned no CSS.")
        ctx.set_result(interactive_css={"css": css, "warnings": list(warnings), "request": request})
        return {"ok": True, "outputs": []}
    folders = [str(p) for p in ctx.inputs if p]
    if not folders or not os.path.isdir(folders[0]):
        raise JobError("The book has no output folder (translate it first)")
    folder = folders[0]
    ctx.set_output_dir(folder)
    keys = ctx.params.get("keys")
    total = len(keys) if keys is not None else len(ipc.pending_keys(ipc.open_progress(folder)))
    done = [0]
    ctx.phase("Illustrating")
    ctx.host.emit("progress", label="Illustrating", completed=0, total=total)

    def on_progress(_key: str, _status: str) -> None:
        done[0] += 1
        ctx.host.emit("progress", label="Illustrating", completed=min(done[0], total), total=total)

    result = runner.run_interactive_images(
        config, folder, keys, log=ctx.log, stop_event=_CtxStop(ctx), on_progress=on_progress,
        recompile=runner.owner_recompiler(owner), batch=runner.batch_settings_from_owner(owner),
        glossary_path=core.find_book_glossary(folder, owner) or "")
    summary = {"processed": result.processed, "completed": result.completed, "failed": result.failed,
               "rejected": result.rejected, "skipped": result.skipped, "images": result.images,
               "stopped": result.stopped}
    ctx.set_result(interactive_images=summary)
    if result.no_image_model:
        return {"ok": False, "outputs": [], "error": result.no_image_model}
    if result.stopped or ctx.stop_requested():
        return {"ok": None, "outputs": []}
    if result.errors and not result.processed:
        return {"ok": False, "outputs": [], "error": "; ".join(result.errors[:3])}
    ctx.phase("Interactive Images done")
    return {"ok": True, "outputs": []}


KINDS = {
    "interactive_images": {"verb": "Illustrating", "icon": ICON, "stop_kind": "translation", "run": run},
}
