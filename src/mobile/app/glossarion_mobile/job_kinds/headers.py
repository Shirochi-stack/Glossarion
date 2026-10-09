"""TRANSLATE_HEADERS: "Translate Headers Now" for one or more translated EPUB/PDF workspaces.

Desktop: Other Settings › "Translate Headers Now" (``other_settings.run_standalone_translate_headers``)
checks the API key / model and runs ``translate_headers_standalone.run_translate_headers_now``
on a worker thread: the API client set-up (the multi-key environment), the header
translation for every selected EPUB/PDF (``translate_headers_now``: output folder per book,
an existing ``translated_headers.txt`` reconciled / repaired / re-applied, otherwise
``translate_headers_standalone`` -> ``translated_headers.txt`` + the chapter ``<h1>`` /
``<title>`` updates; PDF workspaces through ``pdf_workspace_compiler``) and the rebuild of
the current EPUB.

This adapter runs that same shared function (U7) on the job's ``HeadlessOwner`` through a
thin view that answers the desktop attributes: ``selected_files`` (the targets' sources),
``get_current_epub_path`` (the first one) and ``_headers_stop_requested`` (the job's Stop).
Each book's output folder comes from the Library row (``output_dir_for``), errors the
desktop shows in a message box are logged. Before it the compile environment of the first
workspace is exported (``owner._build_epub_compile_env``: header prompts, batch size,
temperature, max tokens - the desktop's live GUI environment).

The EPUB rebuild runs through the mobile compile path (``owner._run_epub_compile(folder)`` for
the first EPUB, unless ``rebuild_epub`` is False or the job was stopped), which also lists
the compiled EPUB as a job output; the desktop calls ``fallback_compile_epub`` on the folder it
finds by name (recorded divergence, DISCREPANCIES "U7 tools").

params: ``targets`` (``[{"source", "folder"}]``; else the inputs are the source EPUBs and the
owner resolves their output folders), ``rebuild_epub`` (default True).
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from glossarion_mobile.job_kinds import owner_method, result_fields

__all__ = ["KINDS", "normalize_targets", "run"]


def normalize_targets(params: Mapping[str, Any], inputs: Sequence[str], resolve: Any = None) -> list:
    """``[(source, folder)]`` - EPUB sources with their output folders (folder may be '')."""
    raw = params.get("targets")
    items: list = []
    if isinstance(raw, (list, tuple)) and raw:
        for item in raw:
            if isinstance(item, Mapping):
                items.append((item.get("source"), item.get("folder")))
            elif isinstance(item, (list, tuple)) and item:
                items.append((item[0], item[1] if len(item) > 1 else None))
    else:
        items = [(path, None) for path in inputs or ()]
    out: list = []
    seen: set = set()
    for source, folder in items:
        if not source:
            continue
        source = os.path.abspath(os.fspath(source))
        key = os.path.normcase(source)
        if key in seen:
            continue
        seen.add(key)
        if not folder and callable(resolve):
            try:
                folder = resolve(source)
            except Exception:
                folder = None
        out.append((source, os.path.abspath(os.fspath(folder)) if folder else ""))
    return out


class _HeadersView:
    """The owner as the desktop header translation reads it (attributes fall through to it)."""

    def __init__(self, owner: Any, ctx: Any, sources: Sequence[str]) -> None:
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_ctx", ctx)
        object.__setattr__(self, "selected_files", list(sources))

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_owner"), name)

    @property
    def _headers_stop_requested(self) -> bool:
        return bool(object.__getattribute__(self, "_ctx").stop_requested())

    def get_current_epub_path(self) -> str:
        files = object.__getattribute__(self, "selected_files")
        return files[0] if files else ""

    def append_log(self, message: Any) -> None:
        object.__getattribute__(self, "_ctx").log(message)


def _model_and_key(owner: Any) -> tuple:
    model = getattr(owner, "model_var", "") or ""
    model = model.get() if hasattr(model, "get") else str(model)
    entry = getattr(owner, "api_key_entry", None)
    key = entry.text() if entry is not None and hasattr(entry, "text") else ""
    return model.strip(), (key or "").strip()


def run(ctx: Any) -> dict:
    from glossarion_mobile.services.jobs import JobError, client_log_handlers

    owner = ctx.owner
    params = ctx.params or {}
    targets = normalize_targets(params, ctx.inputs, getattr(owner, "_resolve_translation_output_dir", None))
    if not targets:
        raise JobError("No EPUB or PDF file selected, or the file does not exist.")
    try:
        import translate_headers_standalone as headers_core
    except Exception as exc:  # pragma: no cover - bundle without the module
        raise JobError(f"The shared header translator is not in this build ({exc}).") from exc
    if not hasattr(headers_core, "run_translate_headers_now"):  # pragma: no cover - older bundle
        raise JobError("This build's translate_headers_standalone has no run_translate_headers_now (U7).")
    folders = {os.path.normcase(source): folder for source, folder in targets if folder}
    first_folder = next((folder for _source, folder in targets if folder and os.path.isdir(folder)), "")
    if first_folder:
        owner_method(owner, "_build_epub_compile_env")(first_folder)
    os.environ["EPUB_PATH"] = targets[0][0]
    view = _HeadersView(owner, ctx, [source for source, _folder in targets])
    ctx.phase("Translating headers")
    counts: dict = {}

    def runner(gui: Any) -> Any:
        result = headers_core.translate_headers_now(
            gui,
            show_error=lambda _title, text: ctx.log(f"❌ {text}"),
            output_dir_for=lambda source: folders.get(os.path.normcase(os.path.abspath(source))) or None,
        )
        counts["result"] = result
        return result

    model, api_key = _model_and_key(owner)
    # The API client's log records reach the job through ONE handler, the view's: translate_headers_now
    # attaches it at its start (the desktop re-attach, same outer id), so this kind runs without JobService's
    # host handler (``own_client_logs``) and attaches the view's from here on (the API client set-up logs
    # too). It is detached when the headers are done, so it never outlives the job.
    with client_log_handlers(view):
        headers_core.run_translate_headers_now(view, model, api_key, headers_runner=runner, rebuild_epub=False)
    result = counts.get("result")
    successful, failed = result if isinstance(result, tuple) else (0, len(targets))
    ctx.set_result(headers_successful=successful, headers_failed=failed)
    outputs: list = []
    if ctx.stop_requested():
        return {"ok": None, "outputs": outputs}
    epub_targets = [(s, f) for s, f in targets if s.lower().endswith(".epub") and f and os.path.isdir(f)]
    if params.get("rebuild_epub", True) and successful and epub_targets:
        source, folder = epub_targets[0]
        ctx.log("\n📦 Rebuilding EPUB with translated headers...")
        ctx.log(f"📂 Output directory: {folder}")
        ctx.set_output_dir(folder)
        ctx.phase("Rebuilding EPUB")
        os.environ["EPUB_PATH"] = source
        compile_epub = owner_method(owner, "_run_epub_compile")
        ok, compiled, error = result_fields(compile_epub(folder))
        if ok is False:
            ctx.log(f"⚠️ Failed to rebuild EPUB: {error}")
        else:
            outputs.extend(compiled)
            if not ctx.stop_requested():
                ctx.log("✅ EPUB rebuilt successfully with translated headers!")
    if successful == 0:
        return {"ok": False, "outputs": outputs, "error": "No chapter headers were translated (see the log)."}
    return {"ok": True, "outputs": outputs}


KINDS = {
    "translate_headers": {"verb": "Translating headers", "icon": "TITLE", "stop_kind": "translation", "run": run,
                          "own_client_logs": True},
}
