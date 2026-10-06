"""Host tests for the U7 tools: Retranslate / Resolve QA on the Book page, the Async batch, Review
generator, RPG Maker and SDLXLIFF reviewer screens, the text editor and file tools, the per-job
whole-message log listener (Reader live panel) and the shared Gemini project chooser.

* Job adapters (``retranslate``, ``resolve_qa``, ``async_batch``, ``rpgmaker``, ``review``) run
  against a fake ``ctx`` and fake shared modules (the cores are written concurrently: the tests
  pin the contracts this side codes against), and the registered kinds run through the real
  ``JobService`` with the U3 ``FakeBackend``.
* With the real ``progress_actions`` / ``progress_core`` in the tree, Retranslate plans and
  applies on a fixture workspace (``test_library_ui.make_workspace``) and the progress file and
  output tree are compared before / after.
* Screens are built in the in-memory Flet session from ``test_bootstrap``; dialogs are scripted
  through ``ctx.extras["answers"]``.

Real data is never touched: every Library / output / config path is a pytest tmp dir.

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_tools_u7.py
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import os
import sys
import threading
import types
import zipfile
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from glossarion_mobile import job_kinds  # noqa: E402
from glossarion_mobile.job_kinds import async_batch as async_kind  # noqa: E402
from glossarion_mobile.job_kinds import resolve_qa as resolve_kind  # noqa: E402
from glossarion_mobile.job_kinds import retranslate as retranslate_kind  # noqa: E402
from glossarion_mobile.job_kinds import review as review_kind  # noqa: E402
from glossarion_mobile.job_kinds import rpgmaker as rpg_kind  # noqa: E402
from glossarion_mobile.services.jobs import JobError, JobSpec, JobState  # noqa: E402

U7_KINDS = {"retranslate": "retranslate", "resolve_qa": "resolve_qa", "async_batch": "async_batch",
            "rpgmaker": "rpgmaker", "review": "review"}


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")


def _load(name: str, filename: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


class _ToolHelpers:
    """The tool-screen fakes of test_tools_ui (copied: that module is tested on its own)."""

    class FakeJobs:
        """JobsFeature surface the tool screens use (submit, has_kind, snapshot, on_transition, request_stop)."""

        def __init__(self) -> None:
            self.specs: list = []
            self.listeners: list = []
            self.stops: list = []
            self.snaps: dict = {}

        def has_kind(self, kind):
            return kind in job_kinds.KIND_MODULES or kind in U7_KINDS

        async def submit(self, spec):
            from glossarion_mobile.services.jobs import JobSnapshot

            self.specs.append(spec)
            job_id = f"job{len(self.specs)}"
            self.snaps[job_id] = JobSnapshot(id=job_id, spec=spec, state=JobState.QUEUED, created=1.0)
            return job_id

        def snapshot(self, job_id=None):
            return self.snaps.get(job_id)

        def view(self):
            live = [snap for snap in self.snaps.values() if not snap.is_terminal]
            active = next((snap for snap in live if snap.state != JobState.QUEUED), None)
            return types.SimpleNamespace(active=active, queue=tuple(snap for snap in live if snap is not active))

        def on_transition(self, callback):
            self.listeners.append(callback)
            return lambda: self.listeners.remove(callback) if callback in self.listeners else None

        def request_stop(self, job_id=None, **kwargs):
            self.stops.append(job_id)
            return "graceful"

        def finish(self, job_id, *, state=JobState.DONE, outputs=(), result=None, error=None):
            import dataclasses

            snap = dataclasses.replace(self.snaps[job_id], state=state, started=2.0, finished=3.0,
                                       outputs=tuple(outputs), result=dict(result or {}), error=error)
            self.snaps[job_id] = snap
            for callback in list(self.listeners):
                callback(snap, JobState.RUNNING)
            return snap

    class FakePrefs:
        def __init__(self) -> None:
            self.data: dict = {}
            self.refs: dict = {}

        def get(self, key, default=None):
            return self.data.get(key, default)

        def set(self, key, value):
            self.data[key] = value

        def file_ref(self, path, *, kind=None):
            from glossarion_mobile.state.prefs import file_ref_id

            fid = file_ref_id(os.fspath(path))
            self.refs[fid] = os.path.abspath(os.fspath(path))
            return fid

        def resolve_file_ref(self, fid, *, touch=True):
            return self.refs.get(fid)

    class FakeFiles:
        def __init__(self, picks=None) -> None:
            self.picks = list(picks or [])
            self.shared: list = []

        async def pick_files(self, **kwargs):
            paths = self.picks.pop(0) if self.picks else []
            return [types.SimpleNamespace(path=p, name=os.path.basename(p)) for p in paths]

        async def share(self, paths, **kwargs):
            self.shared.append(list(paths))
            return True

        def export_options(self, path):
            from glossarion_mobile.services.files import ExportOption

            return [ExportOption("share", "Share…", "IOS_SHARE"), ExportOption("save", "Save to…", "SAVE_ALT")]

    @staticmethod
    def make_epub(path: Path, chapters=("ch001.xhtml", "ch002.xhtml"), title="Raw Book") -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("mimetype", "application/epub+zip")
            zf.writestr("META-INF/container.xml",
                        '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:'
                        'container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/'
                        'oebps-package+xml"/></rootfiles></container>')
            manifest = "".join(f'<item id="c{i}" href="{n}" media-type="application/xhtml+xml"/>'
                               for i, n in enumerate(chapters))
            spine = "".join(f'<itemref idref="c{i}"/>' for i in range(len(chapters)))
            zf.writestr("OEBPS/content.opf",
                        '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0"><metadata '
                        f'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>{title}</dc:title></metadata>'
                        f'<manifest>{manifest}</manifest><spine>{spine}</spine></package>')
            for n in chapters:
                zf.writestr(f"OEBPS/{n}", f"<html><body><h1>{n}</h1><p>text</p></body></html>")
        return str(path)

    @classmethod
    def tool_target(cls, tmp_path: Path, name="Book", *, source=True, folder=True, **kwargs):
        from glossarion_mobile.ui.tools import targets as tg

        out = tmp_path / "Output" / name
        if folder:
            out.mkdir(parents=True, exist_ok=True)
            (out / "response_ch001.html").write_text("<html><body><p>one</p></body></html>", encoding="utf-8")
        raw = cls.make_epub(tmp_path / "Library" / "Raw" / f"{name}.epub") if source else ""
        return tg.ToolTarget(title=name, folder=str(out) if folder else "", source=raw, kind="epub", **kwargs)

    @classmethod
    def _ctx(cls, page, *, service=None, store=None, jobs=None, files=None, prefs=None, **kwargs):
        from glossarion_mobile.ui.tools.common import ToolsContext

        navigated: list = []
        notes: list = []
        ctx = ToolsContext(service=service, page=page,
                           navigate=lambda name, params=None, query=None: navigated.append((name, params, query)),
                           notify=lambda message, action=None, on_action=None: notes.append(message),
                           jobs=jobs if jobs is not None else cls.FakeJobs(), files=files,
                           prefs=prefs or cls.FakePrefs(), platform="android",
                           store=store if store is not None else {}, **kwargs)
        ctx.navigated, ctx.notes = navigated, notes
        ctx.extras["answers"] = []
        return ctx

    @staticmethod
    def _mount(page, body):
        page.views[0].controls.append(body)
        page.update()

    @staticmethod
    async def _settle(times=5):
        for _ in range(times):
            await asyncio.sleep(0.01)


def _tools():
    return _ToolHelpers


def _library():
    return _load("_glossarion_u7_library_helpers", "test_library_ui.py")


def _tb():
    return _load("_glossarion_u7_tb_helpers", "test_bootstrap.py")


def _jobs_helpers():
    return _load("_glossarion_u7_jobs_helpers", "test_jobs.py")


@pytest.fixture
def u7_kinds(monkeypatch):
    """Register the U7 kinds (Integrate adds them to ``KIND_MODULES`` / ``JobKind``)."""
    for kind, module in U7_KINDS.items():
        monkeypatch.setitem(job_kinds.KIND_MODULES, kind, module)
    for kind in U7_KINDS:
        job_kinds._CACHE.pop(kind, None)
    yield
    for kind in U7_KINDS:
        job_kinds._CACHE.pop(kind, None)


class FakeCtx:
    """What a job adapter gets (JobContext surface)."""

    def __init__(self, owner=None, *, params=None, inputs=(), config=None, host=None) -> None:
        self.owner = owner if owner is not None else types.SimpleNamespace()
        self.params = dict(params or {})
        self.inputs = tuple(inputs)
        self.config = dict(config or {})
        self.host = host
        self.logs: list = []
        self.phases: list = []
        self.outputs: list = []
        self.results: dict = {}
        self.output_dir = None
        self.output_dirs: dict = {}
        self.stop = False

    def log(self, text):
        self.logs.append(str(text))

    def stop_requested(self):
        return self.stop

    def phase(self, label):
        self.phases.append(label)

    def set_output_dir(self, path):
        self.output_dir = path

    def set_output_dirs(self, mapping):
        self.output_dirs.update(mapping)

    def add_outputs(self, paths):
        self.outputs.extend(paths)

    def set_result(self, **data):
        self.results.update(data)


class TranslateOwner:
    """The translate pair of a HeadlessOwner, recording the run state it was started with."""

    def __init__(self, out_root: str) -> None:
        self.out_root = out_root
        self.seen: dict = {}
        self.logs: list = []
        self._single_qa_resolution_request = None
        self.selected_files = []

    def append_log(self, message):
        self.logs.append(str(message))

    def _resolve_translation_output_dir(self, path):
        return os.path.join(self.out_root, os.path.splitext(os.path.basename(path.rstrip("/\\")))[0])

    def _prepare_translation_run(self, files):
        self.seen = {"files": list(files), "request": self._single_qa_resolution_request,
                     "selected": list(self.selected_files)}
        return {"files": list(files)}

    def _translation_worker(self, request):
        self.seen["worker"] = request
        return None


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


# ==========================================================================
# RETRANSLATE adapter
# ==========================================================================


class FakeResult:
    def __init__(self, **counts) -> None:
        self.counts = counts

    def as_dict(self):
        return dict(self.counts, merged_progress=None)


def test_retranslate_job_applies_the_stashed_plan_and_records_the_desktop_message(monkeypatch):
    applied = []
    book, plan = object(), types.SimpleNamespace(count=3, mode="retranslate")

    def apply_retranslation(b, p, linked_choice=None, sidecar_workers=None):
        applied.append((b, p, linked_choice, sidecar_workers))
        return FakeResult(deleted_count=2, status_reset_count=3, manual_editing=False)

    fake = _module("progress_actions", apply_retranslation=apply_retranslation,
                   retranslation_result_message=lambda r: ("info", "Success",
                                                           "Successfully Deleted 2 files.\n\nTotal 3 chapters ready for translation."))
    monkeypatch.setitem(sys.modules, "progress_actions", fake)
    token = retranslate_kind.stash(book, plan)
    assert token in retranslate_kind.pending_tokens()
    ctx = FakeCtx(params={"plan": token, "linked_choice": "both", "count": 3})
    result = retranslate_kind.run(ctx)
    assert result == {"ok": True, "outputs": []}
    # the core's own sidecar default (the owner's extraction workers) is used
    assert applied == [(book, plan, "both", None)]
    assert ctx.logs[0] == "🔁 Retranslate Selected: 3 row(s)"
    assert ctx.logs[1:] == ["Successfully Deleted 2 files.", "Total 3 chapters ready for translation."]
    assert ctx.results["retranslate_title"] == "Success" and ctx.results["retranslate_kind"] == "info"
    assert ctx.results["retranslate_counts"] == {"deleted_count": 2, "status_reset_count": 3, "manual_editing": False}
    assert token not in retranslate_kind.pending_tokens()
    # the plan is gone (restart / resume): the job fails with the instruction to select again
    with pytest.raises(JobError, match="no longer available"):
        retranslate_kind.run(FakeCtx(params={"plan": token}))
    assert retranslate_kind.KINDS["retranslate"]["resumable"] is False


def test_retranslate_plan_store_is_bounded():
    tokens = [retranslate_kind.stash(object(), object()) for _ in range(retranslate_kind._MAX_PLANS + 5)]
    pending = retranslate_kind.pending_tokens()
    assert len(pending) == retranslate_kind._MAX_PLANS and tokens[-1] in pending and tokens[0] not in pending
    for token in tokens:
        retranslate_kind.discard(token)
    assert not set(tokens) & set(retranslate_kind.pending_tokens())


# ==========================================================================
# RESOLVE_QA adapter
# ==========================================================================


def _partial_b_progress(tmp_path, *, qa=True):
    ws = tmp_path / "Output" / "Book"
    ws.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "Raw" / "Book.epub"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"PK")
    entry = {"actual_num": 2, "status": "qa_failed", "output_file": "response_ch002.html"}
    if qa:
        entry["qa_issues_found"] = ["korean_text_detected"]
    progress = ws / "translation_progress.json"
    progress.write_text(json.dumps({"chapters": {"2": entry}}), encoding="utf-8")
    request = {"source_path": str(source), "progress_path": str(progress), "progress_key": "2",
               "output_file": "response_ch002.html", "actual_num": 2}
    return request, ws


def test_resolve_qa_job_runs_the_shared_preflight_then_the_translation_pair(tmp_path, monkeypatch):
    calls = []

    def prepare_single_qa_resolution(owner, data, display_info):
        calls.append((data, dict(display_info)))
        entry = data["prog"]["chapters"]["2"]
        if not entry.get("qa_issues_found"):
            return {"ok": False, "refusal": ("info", "QA Issue Already Resolved",
                                             "This entry no longer has a raw foreign-text QA issue."),
                    "refresh": True, "request": None, "source_path": "", "label": "", "log": ""}
        owner._single_qa_resolution_request = {"progress_key": "2"}
        owner.selected_files = [data["file_path"]]
        return {"ok": True, "refusal": None, "refresh": False, "request": owner._single_qa_resolution_request,
                "source_path": data["file_path"], "label": "response_ch002.html",
                "log": "⚠️ Queued Partial.b QA resolution for response_ch002.html only"}

    monkeypatch.setitem(sys.modules, "progress_actions",
                        _module("progress_actions", prepare_single_qa_resolution=prepare_single_qa_resolution))
    request, ws = _partial_b_progress(tmp_path)
    owner = TranslateOwner(str(tmp_path / "Output"))
    ctx = FakeCtx(owner, params={"request": request, "display_info": {"progress_key": "2"}, "label": "x"},
                  inputs=(request["source_path"],))
    result = resolve_kind.run(ctx)
    assert result["ok"] is None or result["ok"] is True  # the worker returned None (no outcome)
    data, display = calls[0]
    assert data["progress_file"] == request["progress_path"] and data["file_path"] == request["source_path"]
    assert data["output_dir"] == str(ws) and display == {"progress_key": "2", "output_file": "response_ch002.html"}
    # the run started with the preflight's request; the adapter clears it afterwards
    assert owner.seen["request"] == {"progress_key": "2"} and owner.seen["files"] == [request["source_path"]]
    assert owner._single_qa_resolution_request is None
    assert "⚠️ Queued Partial.b QA resolution for response_ch002.html only" in ctx.logs
    # the issue went away meanwhile: an informational refusal, no run
    request2, _ws = _partial_b_progress(tmp_path / "b", qa=False)
    owner2 = TranslateOwner(str(tmp_path / "Output"))
    ctx2 = FakeCtx(owner2, params={"request": request2}, inputs=(request2["source_path"],))
    assert resolve_kind.run(ctx2) == {"ok": True, "outputs": []}
    assert owner2.seen == {} and ctx2.results["resolve_qa_refusal"]["title"] == "QA Issue Already Resolved"
    assert ctx2.logs[-1].startswith("ℹ️ QA Issue Already Resolved")


def test_resolve_qa_job_error_refusal_and_missing_core(tmp_path, monkeypatch):
    def prepare(owner, data, display_info):
        return {"ok": False, "refusal": ("error", "Source File Missing", "gone"), "refresh": False}

    monkeypatch.setitem(sys.modules, "progress_actions", _module("progress_actions",
                                                                 prepare_single_qa_resolution=prepare))
    request, _ws = _partial_b_progress(tmp_path)
    ctx = FakeCtx(TranslateOwner(str(tmp_path)), params={"request": request}, inputs=(request["source_path"],))
    assert resolve_kind.run(ctx) == {"ok": False, "outputs": [], "error": "Source File Missing: gone"}
    monkeypatch.setitem(sys.modules, "progress_actions", _module("progress_actions"))
    with pytest.raises(JobError, match="prepare_single_qa_resolution"):
        resolve_kind.run(FakeCtx(TranslateOwner(str(tmp_path)), params={"request": request},
                                 inputs=(request["source_path"],)))


# ==========================================================================
# ASYNC_BATCH adapter
# ==========================================================================


class FakeHeadlessBatch:
    instances: list = []

    def __init__(self, owner, *, host=None, jobs_file=None, answers=None) -> None:
        self.owner, self.host, self.jobs_file, self.answers = owner, host, jobs_file, answers
        self.messages = []
        self.cost_info = "Select chapters to see cost estimate"
        self.calls = []
        FakeHeadlessBatch.instances.append(self)

    def submit(self):
        self.calls.append(("submit", self.owner.file_path))
        self.messages.append({"level": "information", "title": "Batch Submitted", "text": "ok", "extra": 1})
        return types.SimpleNamespace(job_id="batch_1")

    def estimate(self):
        self.calls.append(("estimate", self.owner.file_path))
        self.cost_info = "Estimated cost: $1.23"
        return self.cost_info

    def refresh_statuses(self):
        self.calls.append(("refresh",))
        return []

    def check_status(self, job_id):
        self.calls.append(("check", job_id))
        return []

    def retrieve(self, job_ids):
        self.calls.append(("retrieve", list(job_ids)))
        return [self.saved] if getattr(self, "saved", None) else []

    def cancel(self, job_ids):
        self.calls.append(("cancel", list(job_ids)))

    def delete(self, job_ids):
        self.calls.append(("delete", list(job_ids)))

    def clear_completed(self):
        self.calls.append(("clear",))


def test_async_batch_job_runs_the_dialog_handlers_on_a_headless_batch(tmp_path, monkeypatch):
    FakeHeadlessBatch.instances = []
    monkeypatch.setitem(sys.modules, "async_batch_core",
                        _module("async_batch_core", HeadlessAsyncBatch=FakeHeadlessBatch,
                                default_jobs_file=lambda: str(tmp_path / "data" / "async_jobs.json")))
    book = tmp_path / "Book.epub"
    book.write_bytes(b"PK")
    owner = types.SimpleNamespace()
    host = object()
    ctx = FakeCtx(owner, params={"action": "submit"}, inputs=(str(book),), host=host)
    assert async_kind.run(ctx) == {"ok": True, "outputs": []}
    batch = FakeHeadlessBatch.instances[-1]
    assert owner.file_path == str(book) and batch.host is host and batch.calls == [("submit", str(book))]
    assert batch.jobs_file == str(tmp_path / "data" / "async_jobs.json") and (tmp_path / "data").is_dir()
    assert ctx.results["async_job_id"] == "batch_1"
    assert ctx.results["async_messages"] == [{"level": "information", "title": "Batch Submitted", "text": "ok"}]
    # estimate keeps the cost text
    ctx = FakeCtx(owner, params={"action": "estimate", "jobs_file": str(tmp_path / "j.json")}, inputs=(str(book),))
    async_kind.run(ctx)
    assert ctx.results["async_cost"] == "Estimated cost: $1.23"
    assert FakeHeadlessBatch.instances[-1].jobs_file == str(tmp_path / "j.json")
    # selection-based actions
    for action, expected in (("check", ("check", "a")), ("cancel", ("cancel", ["a", "b"])),
                             ("delete", ("delete", ["a", "b"])), ("refresh", ("refresh",)),
                             ("clear_completed", ("clear",))):
        ctx = FakeCtx(owner, params={"action": action, "job_ids": ["a", "b"]})
        assert async_kind.run(ctx)["ok"] is True
        assert FakeHeadlessBatch.instances[-1].calls == [expected]
    # retrieve writes the output folders: they become the job's outputs
    out = tmp_path / "Output" / "Book"
    out.mkdir(parents=True)
    (out / "Book.epub").write_bytes(b"epub")
    FakeHeadlessBatch.saved = str(out)
    ctx = FakeCtx(owner, params={"action": "retrieve", "job_ids": ["a"]})
    result = async_kind.run(ctx)
    assert result["outputs"] == [str(out / "Book.epub")] and ctx.output_dirs == {str(out): str(out)}
    del FakeHeadlessBatch.saved
    with pytest.raises(JobError, match="Unknown async batch action"):
        async_kind.run(FakeCtx(owner, params={"action": "nope"}))
    with pytest.raises(JobError, match="check status"):
        async_kind.run(FakeCtx(owner, params={"action": "check"}))


def test_async_batch_job_maps_critical_messages_to_a_failure(tmp_path, monkeypatch):
    class Failing(FakeHeadlessBatch):
        def submit(self):
            self.messages.append({"level": "critical", "title": "Error", "text": "API key is required"})
            return None

    monkeypatch.setitem(sys.modules, "async_batch_core",
                        _module("async_batch_core", HeadlessAsyncBatch=Failing,
                                default_jobs_file=lambda: str(tmp_path / "async_jobs.json")))
    book = tmp_path / "Book.epub"
    book.write_bytes(b"PK")
    ctx = FakeCtx(types.SimpleNamespace(), params={"action": "submit"}, inputs=(str(book),))
    assert async_kind.run(ctx) == {"ok": False, "outputs": [], "error": "API key is required"}
    assert "async_job_id" not in ctx.results


# ==========================================================================
# RPGMAKER adapter
# ==========================================================================


class RpgOwner(TranslateOwner):
    def __init__(self, out_root: str, game_dir: str) -> None:
        super().__init__(out_root)
        self.game_dir = game_dir
        self.registered = []

    def _register_rpgmaker_game_input(self, source, work_dir=None):
        self.registered.append((source, work_dir))
        if source.endswith("bad.zip"):
            raise ValueError("No RPG Maker game found in bad.zip")
        os.makedirs(os.path.join(self.game_dir, "GTool_Translation"), exist_ok=True)
        return self.game_dir


def test_rpgmaker_job_registers_the_game_folder_then_runs_the_translation_pair(tmp_path):
    game_zip = tmp_path / "Game.zip"
    game_zip.write_bytes(b"PK")
    game_dir = str(tmp_path / "work" / "Game")
    owner = RpgOwner(str(tmp_path / "Output"), game_dir)
    ctx = FakeCtx(owner, params={"work_dir": str(tmp_path / "work")}, inputs=(str(game_zip),))
    rpg_kind.run(ctx)
    assert owner.registered == [(str(game_zip), str(tmp_path / "work"))]
    assert owner.seen["files"] == [game_dir] and ctx.results["rpgmaker_game_dir"] == game_dir
    assert ctx.output_dir == os.path.join(game_dir, "GTool_Translation")
    assert "🎮 Game folder: " + game_dir in ctx.logs
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"PK")
    with pytest.raises(JobError, match="No RPG Maker game found"):
        rpg_kind.run(FakeCtx(RpgOwner(str(tmp_path), game_dir), inputs=(str(bad),)))
    txt = tmp_path / "notes.txt"
    txt.write_text("x", encoding="utf-8")
    with pytest.raises(JobError, match="game's folder or a ZIP"):
        rpg_kind.run(FakeCtx(RpgOwner(str(tmp_path), game_dir), inputs=(str(txt),)))
    with pytest.raises(JobError, match="no _register_rpgmaker_game_input"):
        rpg_kind.run(FakeCtx(TranslateOwner(str(tmp_path)), inputs=(str(game_zip),)))


# ==========================================================================
# REVIEW adapter
# ==========================================================================


def test_review_job_runs_the_shared_review_orchestration(tmp_path, monkeypatch):
    calls = []
    out = tmp_path / "Output"

    def review_paths_for(file_path, volume_paths, volume_mode, config):
        stem = lambda p: os.path.splitext(os.path.basename(p))[0]  # noqa: E731
        if volume_mode:
            return [str(out / stem(p) / "review" / "combined_review" / "review.md") for p in volume_paths]
        return [str(out / stem(file_path) / "review" / "review.md")]

    def run_review_session(gui, *, prompt, spoiler_mode=False, chunk_mode=False, wrap_chunks=True,
                           final_review_prompt="", file_path=None, volume_paths=None, volume_mode=False,
                           stop_check_fn=None, log_fn=None):
        calls.append(("session", prompt, final_review_prompt, spoiler_mode, chunk_mode, wrap_chunks, file_path,
                      volume_paths, volume_mode, stop_check_fn()))
        for path in review_paths_for(file_path, volume_paths or [], volume_mode, {}):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            Path(path).write_text("# Review", encoding="utf-8")
        return "# Review"

    def run_all_reviews(params, all_paths, *, chunk_mode, wrap_chunks, batch_size, put, stop_check):
        calls.append(("all", params, list(all_paths), chunk_mode, wrap_chunks, batch_size))
        put(("log", "📖 [1/2] A.epub"))
        put(("nav", 0))
        put(("all_done", None))
        return 2, 0

    order = []
    fake = _module("review_generator", DEFAULT_REVIEW_PROMPT="DEFAULT", DEFAULT_FINAL_REVIEW_PROMPT="FINAL",
                   review_paths_for=review_paths_for, run_review_session=run_review_session,
                   run_all_reviews=run_all_reviews,
                   review_run_params=lambda gui, prompt, spoiler, final: order.append("params") or
                   {"prompt": prompt, "spoiler": spoiler, "final": final},
                   review_all_batch_size=lambda gui: order.append("batch") or 3,
                   reset_review_stop_flags=lambda: order.append("reset"),
                   apply_review_streaming_env=lambda gui: order.append("stream"))
    monkeypatch.setitem(sys.modules, "review_generator", fake)
    a, b = tmp_path / "A.epub", tmp_path / "B.epub"
    a.write_bytes(b"PK")
    b.write_bytes(b"PK")
    owner = object()
    ctx = FakeCtx(owner, params={"mode": "single", "spoiler_mode": True, "chunk_mode": True, "wrap_chunks": False},
                  inputs=(str(a),), config={"review_system_prompt": "Mine {target_lang}"})
    result = review_kind.run(ctx)
    review = str(out / "A" / "review" / "review.md")
    assert result == {"ok": True, "outputs": [review]} and ctx.results["review_paths"] == [review]
    # the dialog's prompts: the saved one, else the shared defaults
    assert calls[-1] == ("session", "Mine {target_lang}", "FINAL", True, True, False, str(a), None, False, False)
    ctx = FakeCtx(owner, params={"mode": "volume"}, inputs=(str(b), str(a)))
    review_kind.run(ctx)
    assert calls[-1][1] == "DEFAULT" and calls[-1][6:9] == (str(b), [str(b), str(a)], True)
    assert ctx.outputs == [str(out / "B" / "review" / "combined_review" / "review.md"),
                           str(out / "A" / "review" / "combined_review" / "review.md")]
    ctx = FakeCtx(owner, params={"mode": "all", "chunk_mode": True}, inputs=(str(a), str(b)))
    result = review_kind.run(ctx)
    assert order == ["params", "batch", "reset", "stream"]  # the dialog's Generate All sequence
    assert calls[-1][0] == "all" and calls[-1][2] == [str(a), str(b)] and calls[-1][5] == 3
    assert ctx.logs == ["📖 [1/2] A.epub"] and ctx.results["review_completed"] == 2
    assert result["ok"] is True and str(out / "A" / "review" / "review.md") in result["outputs"]
    monkeypatch.setitem(sys.modules, "review_generator", _module("review_generator"))
    with pytest.raises(JobError, match="run_review_session"):
        review_kind.run(FakeCtx(owner, params={"mode": "single"}, inputs=(str(a),)))
    with pytest.raises(JobError, match="Review all Files"):
        review_kind.run(FakeCtx(owner, params={"mode": "all"}, inputs=(str(a), str(b))))
    with pytest.raises(JobError, match="Unknown review mode"):
        review_kind.run(FakeCtx(owner, params={"mode": "weird"}, inputs=(str(a),)))


# ==========================================================================
# The U7 kinds through the real JobService
# ==========================================================================


def test_u7_kinds_through_the_job_service(tmp_path, monkeypatch, u7_kinds):
    tj = _jobs_helpers()
    applied = []
    fake = _module("progress_actions",
                   apply_retranslation=lambda b, p, linked_choice=None, sidecar_workers=None: applied.append(p) or
                   FakeResult(deleted_count=1),
                   retranslation_result_message=lambda r: ("info", "Success", "Successfully Deleted 1 files."))
    monkeypatch.setitem(sys.modules, "progress_actions", fake)
    service, backend = tj.make_service(tmp_path)
    plan = types.SimpleNamespace(count=1)
    job_id = service.submit(JobSpec("retranslate", "Book · 1 selected",
                                    params={"plan": retranslate_kind.stash(object(), plan), "count": 1},
                                    resumable=False))
    assert service.wait_idle(tj.TIMEOUT)
    snap = service.snapshot(job_id)
    assert snap.state is JobState.DONE and applied == [plan]
    assert snap.result["retranslate_message"] == "Successfully Deleted 1 files."
    assert ("reset", "translation") in backend.events
    # a question from the async batch core reaches on_question and the answer flows back
    answers = {}

    class Asking(FakeHeadlessBatch):
        def submit(self):
            answers["start"] = self.host.ask("async_batch_question", level="question", title="Start Async Processing",
                                             text="Start async batch processing?", buttons=["yes", "no"], default="no")
            return None

    monkeypatch.setitem(sys.modules, "async_batch_core",
                        _module("async_batch_core", HeadlessAsyncBatch=Asking,
                                default_jobs_file=lambda: str(tmp_path / "async_jobs.json")))
    seen = []

    def listener(snap, question):
        from glossarion_mobile.services.jobs import progress_line

        seen.append((question["kind"], question["data"]["title"], question["data"]["buttons"]))
        seen.append(progress_line(snap))
        threading.Timer(0.02, service.answer, (question["id"], "yes")).start()

    service.on_question(listener)
    book = tmp_path / "Book.epub"
    book.write_bytes(b"PK")
    job_id = service.submit(JobSpec("async_batch", "Submit", inputs=(str(book),), params={"action": "submit"},
                                    resumable=False))
    assert service.wait_idle(tj.TIMEOUT)
    assert seen == [("async_batch_question", "Start Async Processing", ["yes", "no"]),
                    "Waiting for your answer: Start Async Processing"] and answers["start"] == "yes"
    assert service.snapshot(job_id).state is JobState.DONE
    service.close()


# ==========================================================================
# Book page › Chapters: Retranslate / Resolve QA (real progress cores on a fixture workspace)
# ==========================================================================


def _real_core(name: str, *attrs: str):
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return module if all(hasattr(module, a) for a in attrs) else None


@pytest.fixture
def real_env(tmp_path, monkeypatch):
    """The real shared cores pinned to a fixture Library (the developer's Library is never read)."""
    tl = _library()
    lc = _real_core("library_core", "install_library_env", "scan_library", "LibraryShelf")
    if lc is None:
        pytest.skip("library_core not importable")
    fixture = tl.make_workspace(tmp_path)
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(fixture["library"]))
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(fixture["output"]))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    from glossarion_mobile.services.library import LibraryService

    paths = types.SimpleNamespace(library=fixture["library"], output=fixture["output"], cache=fixture["cache"])
    service = LibraryService(paths=paths, config={}, prefs=tl.FakePrefs())
    service.ensure_env()
    fixture["service"] = service
    try:
        yield fixture
    finally:
        try:
            lc.uninstall_library_env()
        except Exception:
            pass


def _tree(folder: Path) -> dict:
    """Relative path -> size of every file under ``folder``."""
    out = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            out[path.relative_to(folder).as_posix()] = path.stat().st_size
    return out


def test_real_retranslate_plan_then_job_resets_the_workspace(real_env):
    if _real_core("progress_actions", "plan_retranslation", "apply_retranslation",
                  "retranslation_result_message") is None:
        pytest.skip("progress_actions.plan_retranslation not in this tree")
    from glossarion_mobile.ui.library import progress_model as pm

    service = real_env["service"]
    ws = real_env["ws"]
    asyncio.run(service.refresh())
    book = service.snapshot.in_progress[0]
    view = pm.load_progress_view(service, book)
    done = next(r for r in view.rows if r.status == "completed")
    before_tree = _tree(ws)
    vm = pm.plan_retranslation(service, view, [done])
    assert vm.mode == "retranslate" and vm.refusal is None and not vm.choices and vm.count == 1
    assert vm.title == "Confirm Retranslation"
    assert vm.message == ("This will process:\n\nCh.1\n\n• 1 existing chapters and SDLXLIFF sidecars will be "
                          "deleted and retranslated\n\nContinue?")
    # planned on a detached copy: the Book page's live data is not the job's
    assert vm.book is not view.state and vm.book.data is not view.state.data
    assert vm.book.data["prog"] == view.state.data["prog"]
    assert _tree(ws) == before_tree  # planning never touches disk
    spec = pm.retranslate_spec(service, book, vm)
    assert spec.kind == "retranslate" and spec.resumable is False and spec.params["count"] == 1
    assert spec.origin["type"] == "library" and spec.title.endswith("1 selected")
    ctx = FakeCtx(params=spec.params)
    retranslate_kind.run(ctx)
    prog = json.loads((ws / "translation_progress.json").read_text(encoding="utf-8"))
    assert prog["chapters"]["1"]["status"] == "pending"
    assert prog["chapters"]["2"]["status"] == "qa_failed"  # untouched rows keep their state
    assert not (ws / "response_ch001.html").exists() and (ws / "response_ch002.html").exists()
    assert set(_tree(ws)) == set(before_tree) - {"response_ch001.html"}
    assert ctx.results["retranslate_title"] == "Success"
    assert ctx.results["retranslate_message"].startswith("Successfully Deleted 1 files")
    assert ctx.results["retranslate_message"].endswith("Total 1 chapters ready for translation.")
    # Manual editing on: the confirmation keeps the SDLXLIFF sidecars
    view = pm.load_progress_view(service, book, previous=view, full=True)
    qa_row = next(r for r in view.rows if r.status == "qa_failed")
    pm.set_manual_editing(service, view, True)
    assert service.cfg("retranslation_manual_editing") is True and view.state.data["manual_editing_state"] is True
    vm = pm.plan_retranslation(service, view, [qa_row])
    assert "retained with translated targets cleared for manual editing" in vm.message
    retranslate_kind.discard(pm.retranslate_spec(service, book, vm).params["plan"])


class InlineJobs:
    """Runs a submitted U7 job inline when asked, then delivers its terminal transition."""

    def __init__(self):
        self.busy = False
        self.specs = []
        self.listeners = []

    def has_kind(self, kind):
        return kind in U7_KINDS or kind == "single_chapter"

    def on_transition(self, callback):
        self.listeners.append(callback)
        return lambda: self.listeners.remove(callback)

    def submit(self, spec):
        self.specs.append(spec)
        return f"job{len(self.specs)}"

    def finish(self, job_id, *, owner=None):
        spec = self.specs[int(job_id[3:]) - 1]
        ctx = FakeCtx(owner, params=spec.params, inputs=spec.inputs)
        job_kinds.get_kind(spec.kind).run(ctx)
        snap = types.SimpleNamespace(id=job_id, is_terminal=True, state=JobState.DONE, error=None,
                                     result=dict(ctx.results))
        for callback in list(self.listeners):
            callback(snap, JobState.RUNNING)
        return ctx


@needs_flet
def test_chapters_tab_retranslate_confirms_then_queues_the_job_and_shows_the_result(real_env, u7_kinds):
    if _real_core("progress_actions", "plan_retranslation", "apply_retranslation") is None:
        pytest.skip("progress_actions.plan_retranslation not in this tree")
    from glossarion_mobile.ui.library.book_page import BookPageScreen
    from glossarion_mobile.ui.router import parse_route

    tl = _library()
    service = real_env["service"]
    ws = real_env["ws"]

    async def scenario():
        _conn, session = _tb()._fake_session("android")
        page = session.page
        await service.refresh()
        book = service.snapshot.in_progress[0]
        ctx = tl._ctx(page, service)
        screen = BookPageScreen(parse_route(f"/library/book/{service.bid_for(book)}?tab=chapters"), ctx)
        screen.actions()
        tl._mount(page, screen.get_body())
        await screen.load()
        chapters = screen.chapters
        jobs = InlineJobs()
        service.jobs = jobs
        completed = next(r for r in chapters.visible if r.status == "completed")
        asked = []

        async def confirm(title, body, **_kw):
            asked.append((title, body))
            return confirm.value

        chapters._confirm = confirm
        confirm.value = False
        assert await chapters.retranslate([completed]) is None and jobs.specs == []
        assert asked[-1][0] == "Confirm Retranslation" and asked[-1][1].endswith("Continue?")
        confirm.value = True
        job_id = await chapters.retranslate([completed])
        assert job_id == "job1" and jobs.specs[0].kind == "retranslate"
        assert (ws / "response_ch001.html").exists()  # nothing deleted before the job runs
        jobs.finish(job_id)
        for _ in range(20):
            await asyncio.sleep(0.02)
        assert not (ws / "response_ch001.html").exists()
        assert chapters.view.state.data.get("skip_cleanup") is True
        title, message = chapters.last_message
        assert title == "Success" and message.startswith("Successfully Deleted 1 files")
        statuses = {r.filename: r.status for r in chapters.rows if r.kind == "chapter"}
        assert statuses[completed.filename] in ("pending", "not_translated")
        screen.dispose()

    asyncio.run(scenario())


class FakeService:
    """The LibraryService surface the Chapters tab / progress_model use."""

    def __init__(self, modules=None, *, config=None, jobs=None):
        from glossarion_mobile.services.library import SharedCore

        self.core = SharedCore(dict(modules or {}))
        self.config = dict(config or {})
        self.jobs = jobs
        self.dirty = 0
        self.submitted = []
        self.prefs = _tools().FakePrefs()

    def cfg(self, key, default=None):
        return self.config.get(key, default)

    def set_cfg(self, key, value):
        self.config[key] = value

    def mark_dirty(self):
        self.dirty += 1

    def has_job_kind(self, kind):
        return kind in U7_KINDS or kind == "single_chapter"

    def origin_for(self, book):
        return {"type": "library", "bid": "b1", "label": f"Library · {book.get('name')}"}

    def raw_source(self, book):
        return book.get("raw_source_path") or ""

    def config_snapshot(self):
        return dict(self.config)

    def save_owner_config(self, config):
        self.config.update(config)

    async def submit(self, spec):
        self.submitted.append(spec)
        return f"job{len(self.submitted)}"

    async def io(self, fn, *args):
        return fn(*args)


def _view(**state):
    from glossarion_mobile.ui.library import progress_model as pm

    data = {"prog": {"chapters": {}}, "output_dir": state.pop("output_dir", ""), "progress_file": "p.json",
            "file_path": state.pop("file_path", "")}
    data.update(state.pop("data", {}))
    book_state = types.SimpleNamespace(owner=types.SimpleNamespace(config={}), data=data)
    return pm.ProgressView(state=book_state, output_dir=data["output_dir"], mode=state.pop("mode", "text"))


def _row(**info):
    from glossarion_mobile.ui.library import progress_model as pm

    raw = types.SimpleNamespace(info=info)
    return pm.RowVM(key=info.get("key", "r1"), kind="chapter", status=info.get("status", "qa_failed"), icon="❌",
                    label="QA Failed", title="Ch.002", output_file=info.get("output_file", ""),
                    filename=info.get("original_filename", ""), raw=raw)


def test_resolve_qa_prefers_the_llm_token_repair_like_the_desktop_menu(tmp_path):
    from glossarion_mobile.ui.library import progress_model as pm

    out = tmp_path / "out"
    out.mkdir()
    (out / "response_ch002.html").write_text("<p>x</p>", encoding="utf-8")
    requested = []
    pa = _module("progress_actions",
                 build_partial_b_request=lambda data, info: requested.append(info) or {
                     "source_path": str(tmp_path / "Book.epub"), "progress_path": str(out / "p.json"),
                     "progress_key": "2", "output_file": "response_ch002.html", "actual_num": 2})
    llm = {"value": True}
    pc = _module("progress_core", _progress_entry_has_llm_token_qa=lambda entry: llm["value"])
    service = FakeService({"progress_actions": pa, "progress_core": pc})
    view = _view(output_dir=str(out))
    row = _row(output_file="response_ch002.html", info={"qa_issues_found": ["llm_token"]}, progress_key="2")
    plan = pm.plan_action(service, view, "resolve_qa", [row])
    assert plan.extra == {"output_path": str(out / "response_ch002.html")} and requested == []
    # an LLM-token entry whose output file is gone still goes to the repair (desktop: the exact
    # output path is None and the shared repair reports it), never to Partial.b or a refusal
    repairs = []
    pa.resolve_llm_token_qa = lambda progress_file, info, path: repairs.append(path) or {
        "repair": {"resolved": False, "error": f"Output file not found: {path}"}, "error": ""}
    gone = _row(output_file="response_ch009.html", info={"qa_issues_found": ["llm_token"]}, progress_key="9")
    plan = pm.plan_action(service, view, "resolve_qa", [gone])
    assert plan.refusal is None and plan.extra == {"output_path": None} and requested == []
    assert pm.apply_action(service, view, plan) == "Output file not found: None" and repairs == [None]
    pa.resolve_llm_token_qa = lambda progress_file, info, path: {"repair": {"resolved": False}, "error": ""}
    assert pm.apply_action(service, view, plan) == "The empty-attribute repair did not remove the LLM token issue."
    llm["value"] = False  # raw foreign text only: the Partial.b job
    plan = pm.plan_action(service, view, "resolve_qa", [row])
    assert plan.extra["partial_b"]["progress_key"] == "2"
    with pytest.raises(ValueError, match="resolve_qa job"):
        pm.apply_action(service, view, plan)
    spec = pm.resolve_qa_spec(service, {"name": "Book"}, plan)
    assert spec.kind == "resolve_qa" and spec.inputs == (str(tmp_path / "Book.epub"),)
    assert spec.params["display_info"] == {"progress_key": "2", "output_file": "response_ch002.html"}
    assert spec.params["label"] == "response_ch002.html" and spec.title == "Book · response_ch002.html"
    assert spec.params["request"]["actual_num"] == 2
    # neither an LLM-token issue nor a Partial.b request (source gone, issue cleared): refused
    pa.build_partial_b_request = lambda data, info: None
    assert pm.plan_action(service, view, "resolve_qa", [row]).refusal == "This chapter has no resolvable QA issue."


def test_reset_tts_marks_skip_cleanup_like_the_desktop(tmp_path):
    from glossarion_mobile.ui.library import progress_model as pm

    pa = _module("progress_actions", reset_tts=lambda owner, pf, od, rows: {"deleted": 1, "status_reset": 1},
                 reset_tts_message=lambda result: "Successfully deleted 1 TTS file(s).")
    service = FakeService({"progress_actions": pa})
    view = _view(output_dir=str(tmp_path), mode="audio")
    message = pm.apply_action(service, view, pm.ActionPlan("reset_tts", [{"key": "a"}], 1))
    assert message == "Successfully deleted 1 TTS file(s)." and view.state.data["skip_cleanup"] is True


def test_audio_path_and_the_missing_image_core(tmp_path):
    from glossarion_mobile.ui.library import progress_model as pm

    audio = tmp_path / "text_to_speech" / "ch1.mp3"
    audio.parent.mkdir()
    audio.write_bytes(b"ID3")
    pa = _module("progress_actions", find_row_audio=lambda owner, data, info: str(audio))
    service = FakeService({"progress_actions": pa})
    assert pm.audio_path_for(service, _view(), _row()) == str(audio)
    assert pm.audio_path_for(FakeService({"progress_actions": _module("progress_actions")}), _view(), _row()) is None
    missing = pm.load_image_folder_view(FakeService({"progress_core": _module("progress_core")}), str(tmp_path))
    assert missing.error and missing.missing == ("progress_core.build_image_folder_progress",)


def _image_workspace(tmp_path):
    """Images/{a,b}.png + Output/Images with two translated pages, a cover and a pre-2.1 progress file."""
    folder = tmp_path / "Images"
    folder.mkdir()
    for name in ("a", "b"):
        (folder / f"{name}.png").write_bytes(b"\x89PNG" + name.encode())
    out = tmp_path / "Output" / "Images"
    (out / "images").mkdir(parents=True)
    (out / "response_001_a.html").write_text("<p>a</p>", encoding="utf-8")
    (out / "response_002_b.html").write_text("<p>b</p>", encoding="utf-8")
    (out / "images" / "cover.png").write_bytes(b"\x89PNG")
    progress = {"ha": {"output_file": "response_001_a.html", "status": "completed"},
                "hb": {"output_file": "response_002_b.html", "status": "completed"}}
    (out / "translation_progress.json").write_text(json.dumps(progress), encoding="utf-8")
    return folder, out


def _image_service(tmp_path):
    from glossarion_mobile.services.library import LibraryService, SharedCore

    if _real_core("progress_core", "build_image_folder_progress", "mark_image_folder_items_skipped",
                  "delete_image_folder_items", "image_folder_delete_confirmation") is None:
        pytest.skip("progress_core image-folder functions not in this tree")
    service = LibraryService(core=SharedCore(), config={})
    return service


def test_real_image_folder_view_mark_skipped_and_delete(tmp_path, monkeypatch):
    folder, out = _image_workspace(tmp_path)
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(tmp_path / "Output"))
    monkeypatch.chdir(tmp_path)
    service = _image_service(tmp_path)
    from glossarion_mobile.ui.library import progress_model as pm

    view = pm.load_image_folder_view(service, str(folder))
    assert view.error is None and view.output_dir == str(out)
    assert [(i.kind, i.title, i.label) for i in view.items] == [
        ("translated", "📄 Image 001 | a", "✅ Completed"), ("translated", "📄 Image 002 | b", "✅ Completed"),
        ("cover", "🖼️ Cover | cover.png", "⏭️ Skipped (cover)")]
    assert pm.image_delete_confirmation(service, view, view.items[1:]) == (
        "This will delete 1 translated image(s) and 1 cover image(s).\n\nContinue?")
    title, message = pm.image_folder_action(service, view, "mark_skipped", view.items[:1])
    assert (title, message) == ("Success", "Moved 1 image(s) to the images folder.\n"
                                           "They will be skipped in future translations.")
    assert not (out / "response_001_a.html").exists() and (out / "images" / "a.png").read_bytes() == b"\x89PNGa"
    assert json.loads((out / "translation_progress.json").read_text(encoding="utf-8")) == {
        "hb": {"output_file": "response_002_b.html", "status": "completed"}}
    view = pm.load_image_folder_view(service, str(folder))
    b_page = next(i for i in view.items if i.title.endswith("| b"))
    title, message = pm.image_folder_action(service, view, "delete", [b_page])
    assert message == "Deleted 1 file(s).\n\nThey will be retranslated on the next run."
    assert not (out / "response_002_b.html").exists()
    assert json.loads((out / "translation_progress.json").read_text(encoding="utf-8")) == {}
    empty = tmp_path / "Nothing"
    empty.mkdir()
    gone = pm.load_image_folder_view(service, str(empty))
    assert gone.error_title == "Info" and gone.error.startswith("No translation output found for 'Nothing'.")


def test_image_folder_confirmations_match_the_desktop_view():
    from glossarion_mobile.ui.library import chapters_tab as ct

    items = [types.SimpleNamespace(kind="translated"), types.SimpleNamespace(kind="translated"),
             types.SimpleNamespace(kind="cover")]
    title, body = ct.image_confirm_copy("mark_skipped", items)
    assert title == "Confirm Mark as Skipped" and body.startswith("Move 2 translated image(s) to the images folder?")
    assert ct.image_confirm_copy("mark_skipped", items[2:]) is None
    assert ct.image_confirm_copy("delete", items) == ("Confirm Deletion", "")  # body: the shared text
    assert ct.image_confirm_copy("delete", []) is None
    source = (SRC_DIR / "Retranslation_GUI.py").read_text(encoding="utf-8-sig")
    for text in ("Confirm Mark as Skipped", "• Delete the translated HTML files",
                 "• Copy source images to the images folder", "• Skip these images in future translations",
                 "Confirm Deletion", "Selected items are already in the images folder (skipped).",
                 "Please select at least one image to mark as skipped."):
        assert text in source


# ==========================================================================
# Chapters tab flows over fake cores (refusal, RECYCLED choice, TTS reset, Resolve QA job,
# Manual editing, Edit Translation / Edit file / audio, image-folder grid)
# ==========================================================================


class FakePlan:
    def __init__(self, **kw):
        self.mode = kw.get("mode", "retranslate")
        self.refusal = kw.get("refusal")
        self.confirm_title = kw.get("confirm_title", "Confirm Retranslation")
        self.confirm_message = kw.get("confirm_message", "This will process:\n\nCh.2\n\nContinue?")
        self.needs_linked_choice = kw.get("needs_linked_choice", False)
        self.counterpart_filename = kw.get("counterpart_filename", "")
        self.selected_chapters = kw.get("selected_chapters", [{"key": "r1"}])

    @property
    def count(self):
        return len(self.selected_chapters)

    @property
    def linked_choice_labels(self):
        if not self.needs_linked_choice:
            return ()
        return (("both", "Delete Both Linked Files"), ("selected_only", f"Keep {self.counterpart_filename}"),
                ("cancel", "Cancel"))


class FakeBookPage:
    def __init__(self, ctx, service, book, view=None):
        self.ctx = ctx
        self.service = service
        self.book = book
        self.progress = view
        self.reloads = []
        self.tabs = []

    async def reload_progress(self, *, full=False, force=False):
        self.reloads.append((full, force))
        return self.progress

    async def full_refresh(self):
        self.reloads.append(("full",))

    def open_files(self):
        self.ctx.go("tools.files", {"root": "output"})

    def open_reader(self, **kwargs):
        self.ctx.go("reader", {"bid": "b1"}, kwargs)

    def set_tab(self, name, **_kw):
        self.tabs.append(name)

    async def open_translate(self):
        return None


def _library_ctx(page, service, **kwargs):
    from glossarion_mobile.ui.library.common import LibraryContext

    navigated, notes = [], []
    ctx = LibraryContext(service=service, page=page, prefs=service.prefs,
                         navigate=lambda name, params=None, query=None: navigated.append((name, params, query)),
                         notify=lambda message, action=None, on_action=None: notes.append(message),
                         platform="android", **kwargs)
    ctx.navigated, ctx.notes = navigated, notes
    ctx.extras["answers"] = []
    return ctx


def _tab(service, *, book=None, view=None, page=None):
    from glossarion_mobile.ui.library.chapters_tab import ChaptersTab

    ctx = _library_ctx(page, service)
    host = FakeBookPage(ctx, service, book or {"name": "Book", "output_folder": "", "workspace_kind": "epub"}, view)
    tab = ChaptersTab(host)
    return tab, ctx, host


@needs_flet
def test_chapters_tab_retranslate_refusal_recycled_choice_and_tts_reset(tmp_path, u7_kinds):
    plans = []
    pa = _module("progress_actions", plan_retranslation=lambda book, rows, settings=None: plans.pop(0),
                 reset_tts=lambda owner, pf, od, rows: {"rows": list(rows)},
                 reset_tts_message=lambda result: f"reset {len(result['rows'])}")
    service = FakeService({"progress_actions": pa}, jobs=InlineJobs())
    view = _view(output_dir=str(tmp_path))
    row = _row(output_file="response_ch002.html")

    async def scenario():
        tab, ctx, host = _tab(service, view=view)
        tab.view = view
        confirms = []

        async def confirm(title, body, **_kw):
            confirms.append((title, body))
            return True

        tab._confirm = confirm
        # refused: the desktop message box, nothing queued
        plans.append(FakePlan(mode="refused", refusal=("info", "Metadata Translation Disabled",
                                                       "Enable 'Translate Book Title / Metadata' before requesting "
                                                       "metadata regeneration.")))
        vm = await tab.retranslate([row])
        assert vm.mode == "refused" and tab.last_message[0] == "Metadata Translation Disabled"
        assert service.submitted == []
        # RECYCLED pair: three buttons; Cancel stops, "both" queues with the choice
        plans.append(FakePlan(needs_linked_choice=True, counterpart_filename="translated_headers.txt",
                              confirm_title="Confirm Linked Retranslation", confirm_message="linked?"))
        ctx.extras["answers"] = ["cancel"]
        assert await tab.retranslate([row]) is None and service.submitted == []
        assert ctx.extras["asked"][-1] == ("Confirm Linked Retranslation", "linked?")
        plans.append(FakePlan(needs_linked_choice=True, counterpart_filename="translated_headers.txt"))
        ctx.extras["answers"] = ["both"]
        job_id = await tab.retranslate([row])
        spec = service.submitted[-1]
        assert job_id == "job1" and spec.kind == "retranslate" and spec.params["linked_choice"] == "both"
        assert spec.params["plan"] in retranslate_kind.pending_tokens()
        retranslate_kind.discard(spec.params["plan"])
        # an ordinary plan: Yes / No with the verbatim copy
        plans.append(FakePlan())
        await tab.retranslate([row])
        assert confirms[-1] == ("Confirm Retranslation", "This will process:\n\nCh.2\n\nContinue?")
        retranslate_kind.discard(service.submitted[-1].params["plan"])
        # audio output: the plan is the TTS reset; it runs in place (no job)
        before = len(service.submitted)
        plans.append(FakePlan(mode="reset_tts", confirm_title="Confirm TTS Reset", confirm_message="tts?",
                              selected_chapters=[{"key": "a"}, {"key": "b"}]))
        message = await tab.retranslate([row])
        assert message == "reset 2" and confirms[-1] == ("Confirm TTS Reset", "tts?")
        assert len(service.submitted) == before and view.state.data["skip_cleanup"] is True
        assert host.reloads[-1] == (False, True)

    asyncio.run(scenario())


@needs_flet
def test_chapters_tab_retranslate_plans_under_the_progress_lock_and_reports_a_failed_reset(tmp_path, u7_kinds):
    """The plan deep-copies the loaded progress the Book page's reloads refresh in place: it runs
    under the page's progress lock. A FAILED retranslate job (``JobState.FAILED``) shows the
    desktop "Retranslation Reset Failed" message."""
    from glossarion_mobile.ui.library import chapters_tab as ct

    held, tabs = [], []

    def plan(book, rows, settings=None):
        held.append(tabs[0].page._progress_lock.locked())
        return FakePlan(mode="refused", refusal=("info", "Nothing", "Nothing to do."))

    pa = _module("progress_actions", plan_retranslation=plan)
    service = FakeService({"progress_actions": pa}, jobs=InlineJobs())
    view = _view(output_dir=str(tmp_path))

    async def scenario():
        tab_, _ctx, _host = _tab(service, view=view)
        tab_.view = view
        tabs.append(tab_)
        await tab_.retranslate([_row(output_file="response_ch002.html")])
        assert held == [True] and not tab_.page._progress_lock.locked()
        failed = types.SimpleNamespace(id="job9", is_terminal=True, state=JobState.FAILED, error="progress locked",
                                       result={})
        assert await tab_._on_retranslate_end(failed) == "progress locked"
        assert tab_.last_message == (ct.RETRANSLATE_FAILED, "progress locked")
        done = types.SimpleNamespace(id="job10", is_terminal=True, state=JobState.DONE, error=None,
                                     result={"retranslate_title": "Retranslate", "retranslate_message": "Reset 1"})
        assert await tab_._on_retranslate_end(done) == "Reset 1"

    asyncio.run(scenario())


@needs_flet
def test_chapters_tab_resolve_qa_runs_a_partial_b_job_unless_busy(tmp_path, u7_kinds):
    request = {"source_path": str(tmp_path / "Book.epub"), "progress_path": str(tmp_path / "p.json"),
               "progress_key": "2", "output_file": "response_ch002.html", "actual_num": 2}
    pa = _module("progress_actions", build_partial_b_request=lambda data, info: dict(request))
    pc = _module("progress_core", _progress_entry_has_llm_token_qa=lambda entry: False)
    jobs = InlineJobs()
    service = FakeService({"progress_actions": pa, "progress_core": pc}, jobs=jobs)
    view = _view(output_dir=str(tmp_path))
    row = _row(output_file="response_ch002.html", progress_key="2")

    async def scenario():
        tab, ctx, _host = _tab(service, view=view)
        tab.view = view
        jobs.busy = True
        assert await tab.run_action("resolve_qa", [row]) is None
        assert tab.last_message == ("Process Running",
                                    "Wait for the current translation or glossary process to finish first.")
        jobs.busy = False
        job_id = await tab.run_action("resolve_qa", [row])
        spec = service.submitted[-1]
        assert job_id == "job1" and spec.kind == "resolve_qa" and spec.params["request"] == request
        assert ctx.notes[-1] == "⚠️ Queued Partial.b QA resolution for response_ch002.html only"
        assert job_id in tab.job_watch

    asyncio.run(scenario())


@needs_flet
def test_chapters_tab_manual_editing_reviewer_text_editor_and_audio(tmp_path, monkeypatch):
    out = tmp_path / "Output" / "Book"
    out.mkdir(parents=True)
    (out / "response_ch002.html").write_text("<p>'원문' text</p>", encoding="utf-8")
    audio = out / "text_to_speech" / "ch2.mp3"
    audio.parent.mkdir()
    audio.write_bytes(b"ID3")
    pa = _module("progress_actions", find_row_audio=lambda owner, data, info: str(audio),
                 qa_issue_search_target=lambda path, issues: ("원문", 1))
    service = FakeService({"progress_actions": pa})
    view = _view(output_dir=str(out), file_path=str(tmp_path / "Book.epub"))
    row = _row(output_file="response_ch002.html", info={"qa_issues_found": ["korean_text '원문'"]})
    shared = []

    class Files:
        async def share(self, paths):
            shared.append(list(paths))
            return True

    async def scenario():
        tab, ctx, _host = _tab(service, view=view, book={"name": "Book", "output_folder": str(out)})
        ctx.files = Files()
        tab.view = view
        tab.more_menu = types.SimpleNamespace(items=[], update=lambda: None)
        assert tab.manual_editing is False
        assert tab.edit_translation_reason(_row(output_file="")) is not None
        assert await tab.toggle_manual_editing() is True
        assert service.config["retranslation_manual_editing"] is True and view.state.data["manual_editing_state"]
        assert [i.content for i in tab.more_menu.items][:2] == ["Manual editing", "\U0001f50d Edit Translation"]
        assert tab.more_menu.items[0].checked is True
        assert tab.edit_translation_reason(_row(output_file="", original_filename="ch.xhtml")) is None
        # 🔍 Edit Translation: the reviewer route on the output folder, focused in-process on the row
        from glossarion_mobile.ui.tools import sdlxliff

        await tab.open_reviewer(row)
        name, params, query = ctx.navigated[-1]
        assert name == "tools.sdlxliff" and params is None and service.prefs.resolve_file_ref(query["out"]) == str(out)
        request = sdlxliff.take_request(str(out))
        assert request.focus == "response_ch002.html" and request.source == str(tmp_path / "Book.epub")
        assert request.manual_editing is True
        # ✏️ Edit file (find QA issue): the text editor at the shared search term (in-process)
        from glossarion_mobile.ui.tools import text_editor

        fid = tab.edit_file(row)
        assert ctx.navigated[-1] == ("tools.text", {"fid": fid}, {"hit": 1})
        assert text_editor.take_request(fid).find == "원문"
        # 🔊 Open Audio File: handed to a player app
        assert await tab.open_audio(row) is True and shared[-1] == [str(audio)]

    asyncio.run(scenario())


@needs_flet
def test_chapters_tab_image_folder_grid_selection_and_actions(tmp_path, monkeypatch):
    folder, out = _image_workspace(tmp_path)
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(tmp_path / "Output"))
    monkeypatch.chdir(tmp_path)
    _image_service(tmp_path)  # skips without the shared image-folder functions
    service = FakeService({})  # the real shared modules, imported lazily
    book = {"name": "Images", "output_folder": str(out), "workspace_kind": "image", "raw_source_path": str(folder)}

    async def scenario():
        _conn, session = _tb()._fake_session("android")
        page = session.page
        tab, ctx, _host = _tab(service, book=book, view=_view(output_dir=str(out)), page=page)
        assert tab.image_folder is True
        body = tab.build()
        page.views[0].controls.append(body)
        page.update()
        await tab.reload_images()
        assert len(tab.image_grid.controls) == 3
        tab.select_translated_images()
        assert len(tab.image_selected) == 2 and tab.image_count.value == "Selected: 2"
        confirms = []

        async def confirm(title, body, **_kw):
            confirms.append((title, body))
            return True

        tab._confirm = confirm
        message = await tab.run_image_action("delete")
        assert confirms[-1] == ("Confirm Deletion", "This will delete 2 translated image(s).\n\nContinue?")
        assert message == "Deleted 2 file(s).\n\nThey will be retranslated on the next run."
        assert not (out / "response_001_a.html").exists() and not (out / "response_002_b.html").exists()
        assert tab.image_selected == set() and tab.last_message[0] == "Success"
        assert len(tab.image_grid.controls) == 1  # only the cover is left
        tab.clear_images()
        assert await tab.run_image_action("mark_skipped") is None
        assert ctx.notes[-1] == "Please select at least one image to mark as skipped."
        tab.select_all_images()
        assert await tab.run_image_action("mark_skipped") is None  # only the cover: already skipped
        assert ctx.notes[-1] == "Selected items are already in the images folder (skipped)."

    asyncio.run(scenario())


def test_no_u7_placeholders_left_in_the_owned_surfaces():
    app = APP_DIR / "glossarion_mobile"
    for rel in ("ui/library/chapters_tab.py", "ui/library/progress_model.py", "ui/screens/files.py",
                "ui/tools/async_batch.py", "ui/tools/review.py", "ui/tools/rpgmaker.py", "ui/tools/sdlxliff.py",
                "ui/tools/text_editor.py"):
        text = (app / rel).read_text(encoding="utf-8")
        assert "arrive in u7" not in text.lower() and "arrives in u7" not in text.lower(), rel
        assert "(U7)" not in text and "wired in U7" not in text, rel


# ==========================================================================
# Per-job whole-message log listener (desktop add_log_listener) and the Reader live feed
# ==========================================================================


def test_add_log_listener_replays_the_backlog_and_keeps_blank_lines(tmp_path):
    tj = _jobs_helpers()
    service, backend = tj.make_service(tmp_path)
    gate = threading.Event()
    received = []

    def worker(owner, request):
        owner.host.log("Chapter 1")
        owner.host.log("")  # a paragraph break of a plain-text stream
        gate.wait(2)
        owner.host.log("Para 1\n\nPara 2")
        return None

    backend.behavior = worker
    job_id = service.submit(JobSpec("translate", "A", (tj.epub(tmp_path),)))
    assert tj.wait_for(lambda: any("Chapter 1" == m for m in list(service._find_live(job_id).messages)))
    remove = service.add_log_listener(job_id, received.append)
    assert remove is not None
    assert "Chapter 1" in received and "" in received  # backlog, blank message included
    gate.set()
    assert service.wait_idle(tj.TIMEOUT)
    assert "Para 1\n\nPara 2" in received  # whole message, blank line inside kept
    buffer_lines = [line.text for line in service.log_buffer(job_id).snapshot()]
    assert "" not in buffer_lines  # the LogBuffer still drops blank lines (job detail view)
    count = len(received)
    remove()
    service._job_log(service._find_live(job_id), "after", {})
    assert len(received) == count
    assert service.add_log_listener("nope", received.append) is None
    service.close()


def test_reader_live_panel_feeds_whole_messages_from_the_job_listener():
    from glossarion_mobile.ui.reader.reader_view import LiveRun, ReaderScreen, _LiveMessages

    fed = []
    panel = types.SimpleNamespace(add_lines=lambda batch: fed.append(list(batch)))
    feed = _LiveMessages(panel, dispatcher=None)  # host test: no bound dispatcher -> synchronous flush
    feed.on_message("Para 1")
    feed.on_message("")
    assert fed == [["Para 1"], [""]]

    posted = []

    class Dispatcher:
        bound = True

        def post(self, fn, *args):
            posted.append(fn)
            return True

    fed.clear()
    feed = _LiveMessages(panel, Dispatcher())
    feed.on_message("a")
    feed.on_message("")
    feed.on_message("b")
    assert len(posted) == 1 and fed == []  # one flush scheduled at a time
    posted[0]()
    assert fed == [["a", "", "b"]]

    listeners = {}

    class Jobs:
        def add_log_listener(self, job_id, callback):
            listeners[job_id] = callback
            callback("backlog")
            return lambda: listeners.pop(job_id, None)

        def log_buffer(self, job_id):
            raise AssertionError("the LogBuffer is not used when the listener exists")

    screen = ReaderScreen.__new__(ReaderScreen)
    screen.deps = types.SimpleNamespace(jobs=Jobs(), dispatcher=None)
    fed.clear()
    live = LiveRun(job_id="j1", chapter_file="c.xhtml", row=0, epub_path="b.epub", panel=panel, feed=None)
    screen._attach_live_log(live)
    assert fed == [["backlog"]] and live.flush_log is not None and "j1" in listeners
    listeners["j1"]("next")
    assert fed[-1] == ["next"]
    live.unsub_log()
    assert "j1" not in listeners


# ==========================================================================
# Gemini GCP project chooser (shared authgem_auth rule) in the Accounts picker
# ==========================================================================


def _authgem_chooser():
    module = _real_core("authgem_auth", "authgem_project_items", "choose_authgem_project_index")
    if module is None:
        pytest.skip("authgem_auth has no shared project chooser in this tree")
    return module


def test_gemini_project_choice_applies_the_desktop_selection_rule(tmp_path):
    authgem = _authgem_chooser()
    from glossarion_mobile.services.oauth import OAuthBridge

    fake = types.SimpleNamespace(authgem_project_items=authgem.authgem_project_items,
                                 choose_authgem_project_index=authgem.choose_authgem_project_index)
    bridge = OAuthBridge(auth_modules={"authgem": fake}, opener=lambda url: None, timeout=1,
                         safe_root=str(tmp_path))
    projects = [("p-billed", "billed"), ("p-unknown", "unknown"), ("p-unbilled", "unbilled")]
    assert bridge.gemini_project_items(projects) == [("✅ p-billed", "p-billed"), ("❔ p-unknown", "p-unknown"),
                                                     ("⚠️ p-unbilled (no billing)", "p-unbilled")]
    assert bridge.gemini_project_choice(projects, "") == "p-billed"
    assert bridge.gemini_project_choice(projects, "p-unknown") == "p-unknown"  # the saved one is kept
    assert bridge.gemini_project_choice(projects, "p-unbilled") == "p-billed"  # unless known unbilled
    assert bridge.gemini_project_choice([("u1", "unknown"), ("x", "unbilled")], "") == "u1"
    assert bridge.gemini_project_choice([("x", "unbilled")], "") is None
    plain = OAuthBridge(auth_modules={"authgem": types.SimpleNamespace()}, opener=lambda url: None, timeout=1,
                        safe_root=str(tmp_path))
    assert plain.gemini_project_items(projects) is None and plain.gemini_project_choice(projects, "") is None


@needs_flet
def test_accounts_picker_selects_the_shared_choice_after_listing(tmp_path):
    authgem = _authgem_chooser()
    ap = _load("_glossarion_u7_accounts_helpers", "test_accounts_profiles.py")
    from glossarion_mobile.services.oauth import OAuthBridge
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.accounts import AccountsScreen

    token_dir = tmp_path / "home" / ".glossarion"
    token_dir.mkdir(parents=True)
    gem = ap.FakeAuthModule("authgem", token_dir)
    gem.authgem_project_items = authgem.authgem_project_items
    gem.choose_authgem_project_index = authgem.choose_authgem_project_index
    bridge = OAuthBridge(auth_modules={"authgem": gem}, opener=ap.Opener(), timeout=5, safe_root=str(tmp_path / "home"))
    config = {"model": "authgem/gemini-3", "authgem_project": "p2"}
    writes = []
    screen = AccountsScreen(parse_route("/settings/accounts"), oauth=bridge,
                            config_get=lambda k, d=None: config.get(k, d),
                            config_set=lambda values: (writes.append(values), config.update(values)))
    screen.get_body()
    gem.get_store(0).save_tokens({"access_token": "tok"})
    gem.projects = [("p1", "billed"), ("p2", "unbilled")]
    asyncio.run(screen.load_projects())
    # the saved p2 is known unbilled: the first billed project wins and is applied (desktop rule)
    assert writes[-1] == {"authgem_project": "p1"} and screen.project_dropdown.value == "p1"
    assert [o.text for o in screen.project_dropdown.options] == ["✅ p1", "⚠️ p2 (no billing)"]
    assert screen.project_note.value == "Found 1 GCP project(s) with billing enabled"


# ==========================================================================
# File browser tools and the text editor
# ==========================================================================


def test_rename_and_delete_stay_inside_the_roots(tmp_path):
    from glossarion_mobile.ui.screens.files import delete_entry, rename_entry, rename_problem

    root = tmp_path / "Output"
    (root / "Book").mkdir(parents=True)
    target = root / "Book" / "a.txt"
    target.write_text("x", encoding="utf-8")
    (root / "Book" / "taken.txt").write_text("y", encoding="utf-8")
    roots = {"output": str(root)}
    assert rename_problem(str(target), "") == "Enter a file name"
    assert rename_problem(str(target), "a/b.txt").startswith("A file name cannot contain")
    assert rename_problem(str(target), "taken.txt") == "A file with that name already exists"
    assert rename_problem(str(target), "a.txt") == "That is already the file's name"
    new = rename_entry(str(target), "b.txt", roots)
    assert new == str(root / "Book" / "b.txt") and not target.exists() and Path(new).exists()
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("z", encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        rename_entry(str(outside), "w.txt", roots)
    with pytest.raises(ValueError, match="outside"):
        delete_entry(str(outside), roots)
    with pytest.raises(ValueError, match="Folders"):
        delete_entry(str(root / "Book"), roots)
    delete_entry(new, roots)
    assert not Path(new).exists() and outside.exists()


def test_text_editor_load_save_keeps_bom_and_line_endings(tmp_path):
    from glossarion_mobile.ui.tools import text_editor as te

    path = tmp_path / "a.html"
    path.write_bytes(b"\xef\xbb\xbf<p>one</p>\r\n<p>two</p>\r\n")
    loaded = te.load_text(str(path))
    assert loaded.bom and loaded.crlf and loaded.text == "<p>one</p>\n<p>two</p>\n" and loaded.read_only_reason is None
    te.save_text(str(path), loaded.text.replace("two", "2"), bom=loaded.bom, crlf=loaded.crlf)
    assert path.read_bytes() == b"\xef\xbb\xbf<p>one</p>\r\n<p>2</p>\r\n"
    latin = tmp_path / "b.txt"
    latin.write_bytes("caf\xe9".encode("latin-1"))
    assert te.load_text(str(latin)).read_only_reason == te.NOT_UTF8
    assert te.find_matches("Aa aA aa", "aa") == [0, 3, 6] and te.find_matches("x", "") == []
    assert te.editor_language("x.xhtml") == "XML" and te.editor_language("x.csv") == "PLAINTEXT"
    assert te.editor_language("x.json") == "JSON" and te.editor_language("x.md") == "MARKDOWN"
    assert te.editor_language("x.sdlxliff") == "XML" and te.editor_language("x.css") == "CSS"


@needs_flet
def test_text_editor_screen_finds_edits_saves_and_guards_the_roots(tmp_path):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools import text_editor as te

    root = tmp_path / "Output"
    root.mkdir()
    path = root / "ch.html"
    path.write_text("<p>원문 one</p>\n<p>원문 two</p>\n", encoding="utf-8")
    prefs = _tools().FakePrefs()
    fid = prefs.file_ref(str(path))
    notes = []
    tb = _tb()

    async def io(fn, *args):
        return fn(*args)

    async def scenario():
        _conn, session = tb._fake_session("android")
        page = session.page
        te.request_open(fid, find="원문")
        screen = te.TextEditorScreen(parse_route(f"/tools/text/{fid}?hit=2"), roots={"output": str(root)},
                                     resolve_ref=prefs.resolve_file_ref, page=page, notify=notes.append, run_io=io)
        assert screen.find_term == "원문" and screen.hit == 2 and screen.error is None
        page.views[0].controls.append(screen.get_body())
        page.update()
        await screen.load()
        assert screen.matches == [3, 17] and screen.match_index == 1 and screen.find_count.value == "2/2"
        assert screen.editor.selection.base_offset == 17
        screen.editor.value = screen.editor.value.replace("two", "2")
        screen._on_text_change()
        assert screen.dirty and not screen.save_action.disabled
        assert await screen.save() is True
        assert path.read_text(encoding="utf-8") == "<p>원문 one</p>\n<p>원문 2</p>\n" and not screen.dirty
        assert notes[-1] == "Saved ch.html"
        screen.read_only_switch.value = True
        screen._on_read_only()
        assert screen.editor.read_only and await screen.save() is False
        # outside the roots / unknown links never open
        outside = tmp_path / "secret.txt"
        outside.write_text("s", encoding="utf-8")
        other = te.TextEditorScreen(parse_route(f"/tools/text/{prefs.file_ref(str(outside))}"),
                                    roots={"output": str(root)}, resolve_ref=prefs.resolve_file_ref)
        assert other.error == te.OUTSIDE
        gone = te.TextEditorScreen(parse_route("/tools/text/0123456789ab"), roots={"output": str(root)},
                                   resolve_ref=prefs.resolve_file_ref)
        assert gone.error == "This file link has expired"

    asyncio.run(scenario())


@needs_flet
def test_text_editor_back_with_unsaved_edits_asks_first(tmp_path, monkeypatch):
    """U7 review: Back (Android back, the iOS swipe, the app-bar arrow, the tablet main-area back) must
    not drop unsaved edits. The editor's View cannot pop while it is dirty; Back asks Save / Discard /
    Cancel and the screen leaves (``on_close``) only on Save or Discard. The shell's leave guard and
    Back share the one question."""
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.base import build_screen_view, intercepts_back
    from glossarion_mobile.ui.tools import common as tools_common
    from glossarion_mobile.ui.tools import text_editor as te

    root = tmp_path / "Output"
    root.mkdir()
    path = root / "note.txt"
    path.write_text("original\n", encoding="utf-8")
    prefs = _tools().FakePrefs()
    fid = prefs.file_ref(str(path))
    shown, closed = [], []
    show = tools_common.ChoiceDialog.show
    monkeypatch.setattr(tools_common.ChoiceDialog, "show", lambda self, page: (shown.append(self), show(self, page))[1])

    async def io(fn, *args):
        return fn(*args)

    async def answer(value, count):
        for _ in range(200):
            if len(shown) >= count and shown[count - 1]._future is not None:
                break
            await asyncio.sleep(0.01)
        shown[count - 1].choose(value)
        await _tools()._settle()

    async def scenario():
        _conn, session = _tb()._fake_session("android")
        page = session.page
        screen = te.TextEditorScreen(parse_route(f"/tools/text/{fid}"), roots={"output": str(root)},
                                     resolve_ref=prefs.resolve_file_ref, page=page, run_io=io,
                                     on_close=lambda: closed.append(1))
        view = build_screen_view(screen, f"/tools/text/{fid}")
        page.views.append(view)
        page.update()
        await screen.load()
        assert intercepts_back(screen) and view.can_pop is False and callable(view.on_confirm_pop)
        assert screen.handle_back() is False  # nothing edited: Back pops as usual
        screen.editor.value = "edited\n"
        assert screen.dirty
        await view.on_confirm_pop(None)  # Back -> "Unsaved changes"
        await answer("cancel", 1)
        assert closed == [] and screen.dirty and path.read_text(encoding="utf-8") == "original\n"
        # the shell's leave guard while Back's question is open: one dialog, one answer for both
        assert screen.handle_back() is True
        guard = asyncio.ensure_future(screen.confirm_leave())
        await _tools()._settle()
        assert len(shown) == 2
        await answer("discard", 2)
        assert await guard is True and closed == [1] and path.read_text(encoding="utf-8") == "original\n"
        # Save writes the file, then leaves
        screen.editor.value = "saved edit\n"
        assert screen.handle_back() is True
        await answer("save", 3)
        assert closed == [1, 1] and path.read_text(encoding="utf-8") == "saved edit\n" and not screen.dirty

    asyncio.run(scenario())


@needs_flet
def test_file_browser_open_with_rename_and_delete(tmp_path):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.files import FileBrowserScreen, FileEntry

    root = tmp_path / "Output"
    (root / "Book").mkdir(parents=True)
    note = root / "Book" / "notes.txt"
    note.write_text("x", encoding="utf-8")
    picture = root / "Book" / "Direct Text 1.png"
    picture.write_bytes(b"\x89PNG\r\n\x1a\n")
    prefs = _tools().FakePrefs()
    navigated, notes, opened, overlays = [], [], [], []
    tb = _tb()

    async def io(fn, *args):
        return fn(*args)

    async def scenario():
        _conn, session = tb._fake_session("android")
        page = session.page
        screen = FileBrowserScreen(parse_route("/tools/files/output"), roots={"output": str(root)},
                                   files=_tools().FakeFiles(), page=page,
                                   navigate=lambda name, params=None: navigated.append((name, params)),
                                   notify=notes.append, file_ref=prefs.file_ref, resolve_ref=prefs.resolve_file_ref,
                                   run_io=io, open_reader=lambda **kw: opened.append(kw),
                                   push_overlay=overlays.append, pop_overlay=overlays.remove)
        page.views[0].controls.append(screen.get_body())
        page.update()
        entry = FileEntry("notes.txt", str(note), False, 1, 0.0)
        sheet = screen._on_entry(entry)
        labels = [item.label for item in sheet.items]
        assert labels[-3:] == ["Open with…", "Rename", "Delete"]
        assert all(item.disabled_reason is None for item in sheet.items[-3:])
        open_sheet = screen.open_with(entry)
        assert [i.label for i in open_sheet.items] == ["Text editor", "Reader", "Media viewer", "Another app…"]
        assert open_sheet.items[0].disabled_reason is None
        assert open_sheet.items[2].disabled_reason == "Not an image, video or audio file"
        # UI_SPEC §4.10 Open with › MediaViewer: images, video and audio open in the full-screen viewer
        image = FileEntry(picture.name, str(picture), False, 8, 0.0)
        media_sheet = screen.open_with(image)
        assert media_sheet.items[2].disabled_reason is None and media_sheet.items[0].disabled_reason == "Not a text file"
        media_sheet.items[2].on_select()
        viewer = screen.media_viewer
        assert overlays == [viewer.view] and viewer.current.kind == "image" and viewer.current.path == str(picture)
        viewer._act(viewer.on_share)
        await _tools()._settle()
        assert screen.files.shared[-1] == [str(picture)]
        viewer.close()
        assert overlays == []
        from glossarion_mobile.ui.screens import files as files_screen

        assert files_screen.media_kind("a.MP4") == "video" and files_screen.media_kind("b.ogg") == "audio"
        assert files_screen.media_kind("c.webp") == "image" and files_screen.media_kind("d.epub") is None
        fid = screen.open_text(entry)
        assert navigated[-1] == ("tools.text", {"fid": fid}) and prefs.resolve_file_ref(fid) == str(note)
        open_sheet.items[1].on_select()  # Reader: a .txt book opens in the Reader
        assert opened == [{"path": str(note)}]
        new = await screen.rename(entry, "renamed.txt")
        assert new == str(root / "Book" / "renamed.txt") and notes[-1] == "Renamed to renamed.txt"
        assert await screen.rename(FileEntry("renamed.txt", new, False, 1, 0.0), "") is None
        assert notes[-1] == "Enter a file name"
        assert await screen.delete(FileEntry("renamed.txt", new, False, 1, 0.0)) is True
        assert not Path(new).exists() and notes[-1] == "Deleted renamed.txt"

    asyncio.run(scenario())


# ==========================================================================
# Tool screens: Async batch, RPG Maker, Review generator, SDLXLIFF reviewer
# ==========================================================================


def _tools_ctx(page, **kwargs):
    t = _tools()
    ctx = t._ctx(page, **kwargs)
    return ctx


class U7Jobs:
    """The test_tools_ui FakeJobs plus the question / event channels and the U7 kinds."""

    def __new__(cls):
        base = _tools().FakeJobs

        class Jobs(base):
            def __init__(self):
                super().__init__()
                self.question_listeners = []
                self.event_listeners = []
                self.answers = []

            def has_kind(self, kind):
                return kind in U7_KINDS or super().has_kind(kind)

            def on_question(self, callback):
                self.question_listeners.append(callback)
                return lambda: self.question_listeners.remove(callback)

            def on_event(self, callback):
                self.event_listeners.append(callback)
                return lambda: self.event_listeners.remove(callback)

            def answer(self, question_id, value):
                self.answers.append((question_id, value))
                return True

            def pending_question(self, job_id=None):
                return None

        return Jobs()


def _fake_async_core(tmp_path):
    from datetime import datetime

    class Status:
        def __init__(self, value):
            self.value = value

    jobs = {"batch_a": types.SimpleNamespace(status=Status("completed")),
            "batch_b": types.SimpleNamespace(status=Status("processing"))}

    class Processor:
        def __init__(self, gui, jobs_file=None):
            self.gui, self.jobs_file, self.jobs = gui, jobs_file, dict(jobs)

    def job_display_row(job_id, job):
        return {"job_id": job_id, "display_id": job_id, "provider": "OPENAI", "model": "gpt-x",
                "status": job.status.value.capitalize(), "progress": "100% (Complete)", "progress_pct": 100,
                "created": datetime(2026, 1, 1).strftime("%Y-%m-%d %H:%M"), "source_file": "Book.epub",
                "cost": "N/A", "state": job.status.value}

    return _module("async_batch_core", AsyncAPIProcessor=Processor, job_display_row=job_display_row,
                   async_support_status=lambda processor, model: (model.startswith("gpt"),
                                                                  "✓ Supported (OPENAI)" if model.startswith("gpt")
                                                                  else "✗ Not supported for async"),
                   default_jobs_file=lambda: str(tmp_path / "data" / "async_jobs.json"))


@needs_flet
def test_async_batch_screen_lists_jobs_submits_and_answers_the_dialog_questions(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "async_batch_core", _fake_async_core(tmp_path))
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools import async_batch as ab

    target = _tools().tool_target(tmp_path, "Book")

    async def scenario():
        _conn, session = _tb()._fake_session("android")
        page = session.page
        jobs = U7Jobs()
        store = {"model": "gpt-5", "async_poll_interval": 60}
        ctx = _tools_ctx(page, store=store, jobs=jobs)
        screen = ab.AsyncBatchScreen(parse_route("/tools/async"), ctx)
        _tools()._mount(page, screen.get_body())
        screen.did_show()
        await screen.reload()
        assert screen.support_text.value == "✓ Supported (OPENAI)" and screen.model_text.value == "Current Model: gpt-5"
        assert [c.key for c in screen.jobs_column.controls] == ["async-job-batch_a", "async-job-batch_b"]
        assert screen.start_button.disabled  # no source yet
        # poll interval: the dialog's 10-600 s range
        screen.poll_field.value = "5"
        assert screen._on_poll() == 10 and store["async_poll_interval"] == 10
        screen.wait_switch.value = True
        screen._on_wait()
        assert store["async_wait_for_completion"] is True
        screen.set_target([target])
        assert not screen.start_button.disabled
        job_id = await screen.start("submit")
        spec = jobs.specs[-1]
        assert spec.kind == "async_batch" and spec.params["action"] == "submit" and spec.inputs == (target.source,)
        assert spec.params["jobs_file"] == str(tmp_path / "data" / "async_jobs.json") and spec.resumable is False
        # the job asks the dialog's question: answered from the screen
        ctx.extras["answers"] = ["yes"]
        question = {"id": "q1", "job_id": job_id, "kind": "async_batch_question",
                    "data": {"title": "Start Async Processing", "text": "Start async batch processing?",
                             "buttons": ["yes", "no"]}, "default": "no"}
        dialog = screen._on_question(jobs.snaps[job_id], question)
        assert dialog.title == "Start Async Processing" and jobs.answers == [("q1", "yes")]
        assert [o[0] for o in ab.question_options(["yes", "no", "cancel"])] == ["cancel", "no", "yes"]
        assert screen._on_question(jobs.snaps[job_id], dict(question, job_id="other")) is None
        screen._on_event(job_id, "async_batch_cost", {"text": "Estimated cost: $1.00"})
        assert screen.cost_text.value == "Estimated cost: $1.00"
        screen._on_event(job_id, "async_batch_message", {"level": "information", "title": "Batch Submitted",
                                                         "text": "ok"})
        assert ctx.notes[-1] == "Batch Submitted: ok"
        jobs.finish(job_id, result={"async_action": "submit", "async_job_id": "batch_c"})
        await _tools()._settle()
        assert screen.run_status.value == "Done" and not screen.progress.visible
        # selection actions
        assert await screen.start("retrieve") is None and ctx.notes[-1] == "Please select a job first"
        screen.toggle_job("batch_a")
        job_id = await screen.start("retrieve")
        assert jobs.specs[-1].params["job_ids"] == ["batch_a"] and jobs.specs[-1].params["action"] == "retrieve"
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_rpgmaker_screen_scans_translates_and_exports(tmp_path, monkeypatch):
    game = tmp_path / "Inbox" / "MyGame"
    (game / "www" / "data").mkdir(parents=True)
    (game / "www" / "data" / "System.json").write_text("{}", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "rpgmaker_job", _module(
        "rpgmaker_job", prepare_rpgmaker_game=lambda source, work_dir, log=print, fresh=False: source))
    monkeypatch.setitem(sys.modules, "rpgmaker_handler", _module(
        "rpgmaker_handler", detect_version=lambda d: ("mv", os.path.join(d, "www", "data")),
        extract_all=lambda d, log: ("mv", os.path.join(d, "www", "data"), ["a", "b", "c"])))
    import flet as ft

    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools import rpgmaker as rm

    scan = rm.scan_game(str(game), str(tmp_path / "work"))
    assert scan.summary == "RPG Maker MV · 3 strings" and scan.game_dir == str(game)

    async def scenario():
        _conn, session = _tb()._fake_session("android")
        page = session.page
        jobs = U7Jobs()
        files = _tools().FakeFiles()
        ctx = _tools_ctx(page, jobs=jobs, files=files, output_root=str(tmp_path / "Output"),
                         data_dir=str(tmp_path / "data"))
        screen = rm.RpgMakerScreen(parse_route("/tools/rpgmaker"), ctx)
        _tools()._mount(page, screen.get_body())
        screen.did_show()
        assert screen.translate_button.disabled and screen.scan_button.disabled
        screen.set_source(str(game))
        await screen.run_scan()
        assert screen.scan_text.value == "RPG Maker MV · 3 strings" and not screen.translate_button.disabled
        # a scan alone prepares a copy, it translates nothing: reopening the tool keeps Share disabled
        again = rm.RpgMakerScreen(parse_route("/tools/rpgmaker"), ctx)
        again.get_body()
        assert again.scan == screen.scan and again.last_game_dir == ""
        assert isinstance(again.export_button, ft.Row) and again.export_button.controls[0].disabled
        # while a scan extracts into the work folder, Scan and Translate wait for it
        gate = threading.Event()

        def slow_prepare(source, work_dir, log=print, fresh=False):
            gate.wait(10)
            return source

        monkeypatch.setattr(sys.modules["rpgmaker_job"], "prepare_rpgmaker_game", slow_prepare)
        scanning = asyncio.ensure_future(screen.run_scan())
        for _ in range(100):
            if screen.scanning:
                break
            await asyncio.sleep(0.01)
        assert screen.scanning and screen.translate_button.disabled and screen.scan_button.disabled
        assert await screen.start() is None and jobs.specs == [] and ctx.notes[-1] == "Wait for the scan to finish"
        assert await screen.run_scan() is None  # one scan at a time
        gate.set()
        await scanning
        assert not screen.scanning and not screen.translate_button.disabled
        job_id = await screen.start()
        spec = jobs.specs[-1]
        assert spec.kind == "rpgmaker" and spec.inputs == (str(game),)
        assert spec.params["work_dir"] == os.path.join(str(tmp_path / "Output"), rm.WORK_FOLDER)
        jobs.finish(job_id, result={"rpgmaker_game_dir": str(game)})
        await _tools()._settle()
        assert screen.run_status.value == "Done · applied to MyGame"
        archive = await screen.export()
        assert archive.endswith("MyGame_translated.zip") and files.shared[-1] == [archive]
        with zipfile.ZipFile(archive) as zf:
            assert "www/data/System.json" in zf.namelist()
        assert rm.RpgMakerScreen(parse_route("/tools/rpgmaker"), ctx).last_game_dir == str(game)  # a run does
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_review_screen_modes_prompts_display_and_runs(tmp_path, monkeypatch):
    rg_calls = []
    output_root = tmp_path / "Output"

    def review_paths_for(file_path, volume_paths, volume_mode, config):
        stem = lambda p: os.path.splitext(os.path.basename(p))[0]  # noqa: E731
        if volume_mode:
            return [str(output_root / stem(p) / "review" / "combined_review" / "review.md") for p in volume_paths]
        return [str(output_root / stem(file_path) / "review" / "review.md")]

    monkeypatch.setitem(sys.modules, "review_generator", _module(
        "review_generator", DEFAULT_REVIEW_PROMPT="DEFAULT {target_lang}", DEFAULT_FINAL_REVIEW_PROMPT="FINAL",
        review_paths_for=review_paths_for, count_review_tokens=lambda review_input, log_fn=None: 1234,
        _save_review_text=lambda text, output_dir, log_fn, review_output_paths=None: rg_calls.append(
            (text, output_dir, list(review_output_paths or [])))))
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools import review as rv

    a = _tools().tool_target(tmp_path, "VolA")
    b = _tools().tool_target(tmp_path, "VolB")
    review_md = Path(a.folder) / "review" / "review.md"
    review_md.parent.mkdir(parents=True)
    review_md.write_text("# Great book\n\n- point", encoding="utf-8")

    async def scenario():
        _conn, session = _tb()._fake_session("android")
        page = session.page
        jobs = U7Jobs()
        store = {}
        ctx = _tools_ctx(page, store=store, jobs=jobs)
        screen = rv.ReviewScreen(parse_route("/tools/review"), ctx)
        _tools()._mount(page, screen.get_body())
        screen.did_show()
        screen.set_targets([a, b])
        await screen.load_current()
        assert screen.review_text.startswith("# Great book") and screen.output.visible
        assert screen.token_text.value == "~1,234 tokens"
        assert not screen.all_button.disabled and not screen.switches["wrap_chunks"].visible
        # the dialog's mode toggles persist under its config keys
        screen.set_mode("chunk_mode", True)
        assert store["review_chunk_mode"] is True and screen.switches["wrap_chunks"].visible is True
        screen.switches["spoiler_mode"].value = True
        screen.set_mode("spoiler_mode", True)
        # a review exists: the overwrite question first
        ctx.extras["answers"] = ["no"]
        assert await screen.start("single") is None and ctx.extras["asked"][-1][0] == rv.OVERWRITE_TITLE
        ctx.extras["answers"] = ["yes"]
        job_id = await screen.start("single")
        spec = jobs.specs[-1]
        assert spec.kind == "review" and spec.params["mode"] == "single" and spec.inputs == (a.source,)
        assert spec.params["spoiler_mode"] is True and spec.params["chunk_mode"] is True
        jobs.finish(job_id)
        await _tools()._settle()
        # Review all Files: confirmation with the desktop text, every book on its own
        ctx.extras["answers"] = ["yes"]
        await screen.start("all")
        assert ctx.extras["asked"][-1] == (rv.GENERATE_ALL_TITLE, "Generate reviews for all 2 EPUBs?\n\n"
                                           "This will process each EPUB and save the review automatically.")
        assert jobs.specs[-1].params["mode"] == "all" and jobs.specs[-1].inputs == (a.source, b.source)
        jobs.finish(f"job{len(jobs.specs)}")
        await _tools()._settle()
        # Volume mode: the ordered files, the combined review path in every volume
        screen.switches["volume_mode"].value = True
        screen.set_mode("volume_mode", True)
        # "↕ File Order…": a drag reorders the sheet's rows and the run order alike
        sheet = screen.open_order_sheet()
        listing = screen.order_listing
        assert sheet is not None and [c.key for c in listing.controls] == ["order-0", "order-1"]
        screen._on_reorder(types.SimpleNamespace(old_index=1, new_index=0, control=listing))
        assert [t.title for t in screen.targets] == ["VolB", "VolA"]
        assert [c.key for c in listing.controls] == ["order-1", "order-0"]  # rows follow targets
        screen._on_reorder(types.SimpleNamespace(old_index=0, new_index=1, control=listing))
        assert [t.title for t in screen.targets] == ["VolA", "VolB"]
        assert [c.key for c in listing.controls] == ["order-0", "order-1"]
        screen.move_target(1, 0)
        assert [t.title for t in screen.targets] == ["VolB", "VolA"]
        assert screen.review_paths() == [os.path.join(b.folder, "review", "combined_review", "review.md"),
                                         os.path.join(a.folder, "review", "combined_review", "review.md")]
        ctx.extras["answers"] = []
        await screen.start("single")
        assert jobs.specs[-1].params["mode"] == "volume" and jobs.specs[-1].inputs == (b.source, a.source)
        # display settings: the desktop defaults and Reset
        screen.set_display("review_font_size", "12")
        assert store["review_font_size"] == 12
        screen.reset_display()
        assert {k: store[k] for k in rv.DISPLAY_DEFAULTS} == rv.DISPLAY_DEFAULTS
        # prompts reset to review_generator's defaults
        ctx.extras["answers"] = ["yes"]
        await screen.reset_prompts()
        assert store["review_system_prompt"] == "DEFAULT {target_lang}" and store["review_final_prompt"] == "FINAL"
        # Delete / Restore need the shared backup helpers (disabled with the reason otherwise)
        reasons = [getattr(c, "controls", [None, None])[1] for c in screen.file_actions.controls[1:]]
        assert all(r is not None for r in reasons)
        screen.dispose()

    asyncio.run(scenario())


def test_review_display_defaults_match_the_desktop_dialog():
    from glossarion_mobile.ui.tools import review as rv

    source = (SRC_DIR / "review_dialog.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    found = {}
    for node in ast.walk(tree):
        # translator_gui.config.get('review_font_size', 9) -> {'review_font_size': 9}
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                and len(node.args) == 2 and isinstance(node.args[0], ast.Constant)
                and str(node.args[0].value).startswith("review_") and isinstance(node.args[1], ast.Constant)):
            found.setdefault(node.args[0].value, node.args[1].value)
    for key, value in rv.DISPLAY_DEFAULTS.items():
        assert found.get(key) == value, key
    for key, (config_key, default, label) in rv.MODE_KEYS.items():
        assert found.get(config_key) == default, config_key
        assert f'"{label}"' in source, label
    assert rv.OVERWRITE_TEXT.replace("\n", "\\n") in source.replace("\n\"", "").replace('"\n', "") or \
        "Starting a new review will replace it." in source


def _client_set(control, name, value):
    """A property change coming from the client (Flet applies those to frozen, re-keyed controls too)."""
    frozen = getattr(control, "_frozen", None)
    if frozen is not None:
        del control._frozen
    try:
        setattr(control, name, value)
    finally:
        if frozen is not None:
            control._frozen = frozen


class FakeReviewSession:
    """The ``sdlxliff_review_core.SdlxliffReviewSession`` surface the screen calls."""

    MACHINE_TRANSLATION_INACCURACY_THRESHOLD = 150.0
    opened: list = []

    def __init__(self, output_dir, config, *, current_path=None, context_parent=None, source_path=None,
                 progress_data=None, load=True, **_kw):
        self.output_dir, self.config, self.current_path = output_dir, config, current_path
        self.context_parent, self.source_path, self.progress_data = context_parent, source_path, progress_data
        self.pieces = []
        self.calls = []
        self.status = ""
        self._provider = "auto"
        self._threshold = 150.0
        self._displayed_review_row = -1
        self.changed = False
        FakeReviewSession.opened.append(self)

    @property
    def books(self):
        return [{"output_dir": self.output_dir, "label": "Book"}]

    def refresh(self, force=False, validate=False):
        self.calls.append(("refresh", force))
        self.pieces = [
            {"name": "response_ch001.html.sdlxliff", "output_name": "response_ch001.html", "source_count": 2,
             "target_count": 2, "rows": [
                 {"source": "원문", "target": "Text", "status": "green", "reason": "ok"},
                 {"source": "둘", "target": "", "status": "red", "reason": "empty", "tooltip_translation": "Two"}]},
            {"name": "response_ch002.html.sdlxliff", "output_name": "response_ch002.html", "source_count": 1,
             "target_count": 1, "rows": [{"source": "셋", "target": "Three", "status": "green"}]}]
        if self.current_path:
            for index, piece in enumerate(self.pieces):
                if self.current_path.endswith(piece["name"]):
                    self._displayed_review_row = index
        return {"reloaded": True}

    def changed_on_disk(self):
        return self.changed

    def piece_summary(self, index):
        return {"label": f"Ch.{index + 1}"}

    def select_piece(self, index):
        self._displayed_review_row = index

    def switch_book(self, index):
        return False

    def save_row(self, piece_index, row_index, text):
        self.calls.append(("edit", piece_index, row_index, text))
        self.pieces[piece_index]["rows"][row_index].update(target=text, status="green")
        self.status = "Saved"
        return self.status

    def notepad_document(self, piece_index):
        return "<p>Three</p>" if piece_index == 1 else "<p>Text</p>"

    def edit_document(self, piece_index, html_text, **kw):
        self.calls.append(("document", piece_index, html_text))

    def flush_edits(self):
        self.status = "Saved"
        return self.status

    def output_path(self, piece_index):
        return os.path.join(self.output_dir, self.pieces[piece_index]["output_name"])

    def mark_completed(self, indices):
        for index in indices:
            self.pieces[index]["manual_green_override"] = True
        self.status = f"Marked {len(indices)} SDLXLIFF sidecar completed"
        return self.status

    def undo_completed(self, indices):
        for index in indices:
            self.pieces[index].pop("manual_green_override", None)
        self.status = f"Undid completed mark for {len(indices)} SDLXLIFF sidecar"
        return self.status

    @property
    def provider(self):
        return self._provider

    def set_machine_translation_credentials(self, provider, api_key=None, region=None, folder_id=None):
        self.calls.append(("credentials", provider, api_key, region, folder_id))
        return bool(api_key)

    def set_provider(self, provider):
        if provider == "deepl" and not any(c[0] == "credentials" and c[1] == "deepl" and c[2] for c in self.calls):
            self.status = "DeepL requires an API key"
            return self.status
        self._provider = provider
        self.status = f"Machine translation provider: {provider}"
        return self.status

    def machine_translation_preview(self, piece_index, status_callback=None):
        self.calls.append(("mt", piece_index))
        for row in self.pieces[piece_index]["rows"]:
            row["tooltip_translation"] = "MT " + row["source"]
        self.status = "Machine translation preview ready"
        return {"translated": 2, "error": "", "message": self.status}

    def inject_machine_translation(self, piece_index, row_index):
        row = self.pieces[piece_index]["rows"][row_index]
        self.calls.append(("inject", piece_index, row_index, row.get("tooltip_translation")))
        row["target"] = row.get("tooltip_translation")
        self.status = "Injected"
        return self.status

    def flag_inaccurate(self, piece_index):
        self.calls.append(("flag", piece_index, self._threshold))
        self.pieces[piece_index]["rows"][0]["status"] = "purple"
        self.status = "Flagged 1 MT inaccurate row"
        return self.status

    @property
    def inaccuracy_threshold(self):
        return self._threshold

    def set_inaccuracy_threshold(self, value):
        self._threshold = max(1.0, min(1000.0, float(value)))
        self.status = f"MT inaccuracy threshold set to {self._threshold:g}"
        return self._threshold

    def reset_inaccuracy_threshold(self):
        self._threshold = self.MACHINE_TRANSLATION_INACCURACY_THRESHOLD
        self.status = f"MT inaccuracy threshold reset to {self._threshold:g}"
        return self._threshold


def _fake_sdl_core():
    FakeReviewSession.opened = []
    return _module("sdlxliff_review_core", open_sdlxliff_review=FakeReviewSession)


@needs_flet
def test_sdlxliff_screen_compact_review_edit_complete_mt_flag_and_notepad_rule(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "sdlxliff_review_core", _fake_sdl_core())
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools import sdlxliff as sx
    from glossarion_mobile.ui.tools import text_editor as te

    out = tmp_path / "Output" / "Book"
    out.mkdir(parents=True)
    (out / "response_ch002.html").write_text("<p>Three</p>", encoding="utf-8")

    async def scenario():
        _conn, session = _tb()._fake_session("android")
        page = session.page
        store = {"sdlxliff_machine_translation_provider": "auto"}
        ctx = _tools_ctx(page, store=store)
        fid = sx.open_reviewer(ctx, str(out), source=str(tmp_path / "Book.epub"), focus="response_ch002.html",
                               progress_data={"chapters": {}})
        assert ctx.navigated[-1] == ("tools.sdlxliff", None, {"out": fid})
        screen = sx.SdlxliffScreen(parse_route(f"/tools/sdlxliff?out={fid}"), ctx)
        assert screen.folder == str(out) and screen.request.focus == "response_ch002.html"
        _tools()._mount(page, screen.get_body())
        binding = await screen.open()
        fake = binding.session
        assert fake.source_path == str(tmp_path / "Book.epub") and fake.progress_data == {"chapters": {}}
        assert fake.current_path == os.path.join(str(out), "SDLXLIFF", "response_ch002.html.sdlxliff")
        assert fake.context_parent is binding.parent and ("refresh", False) in fake.calls
        assert screen.piece_index == 1 and screen.piece_dropdown.value == "1"  # the focused piece
        assert [o.text for o in screen.piece_dropdown.options] == ["Ch.1", "Ch.2"]
        assert [c.key for c in screen.legend.controls] == [f"sdl-legend-{s}" for s, _t in sx.LEGEND]
        # phones: the Notepad layout is a tablet-only toggle
        assert screen.layout_switch.disabled is True
        screen.layout_switch.value = True
        screen._on_layout()
        assert screen.notepad is False and ctx.notes[-1] == sx.NOTEPAD_REASON
        # Piece ⋯ Edit Output: the text editor on the piece's output
        out_fid = screen.edit_output()
        assert ctx.navigated[-1] == ("tools.text", {"fid": out_fid}, None)
        te.take_request(out_fid)
        # compact edit: a changed field saves through the shared session
        screen.step(-1)
        assert screen.piece_index == 0 and len(screen.row_fields) == 2
        screen.set_filter("red")
        assert list(screen.row_fields) == [1]
        _client_set(screen.row_fields[1], "value", "Two!")
        assert await screen.save_row(1) is True and ("edit", 0, 1, "Two!") in fake.calls
        assert await screen.save_row(1) is False  # unchanged
        screen.set_filter(None)
        # Mark as Completed / Undo
        assert await screen.toggle_completed() == "Marked 1 SDLXLIFF sidecar completed"
        assert screen.piece_menu.items[0].content == "Undo Completed"
        assert (await screen.toggle_completed()).startswith("Undid completed mark")
        # MT provider: credentials first (the dialog's prompts), then the provider
        ctx.extras["credentials"] = [None]
        assert await screen.choose_provider("deepl") == "auto" and ctx.notes[-1] == "DeepL requires an API key"
        ctx.extras["credentials"] = [{"api_key": "deepl-key"}]
        assert await screen.choose_provider("deepl") == "deepl"
        assert ("credentials", "deepl", "deepl-key", None, None) in fake.calls
        assert await screen.choose_provider("argos") == "deepl" and ctx.notes[-1] == sx.ARGOS_REASON
        ctx.extras["credentials"] = [{"api_key": "bing-key", "region": "westeurope"}]
        assert await screen.configure("bing") is True
        assert ("credentials", "bing", "bing-key", "westeurope", None) in fake.calls
        assert await screen.translate_piece() == "Machine translation preview ready" and ("mt", 0) in fake.calls
        assert await screen.inject(0) == "Injected" and ("inject", 0, 0, "MT 원문") in fake.calls
        # Flag inaccurate with the persisted threshold; set / reset
        assert await screen.flag() == "Flagged 1 MT inaccurate row" and ("flag", 0, 150.0) in fake.calls
        assert await screen.set_threshold("42") == 42.0 and ctx.notes[-1] == "MT inaccuracy threshold set to 42"
        assert await screen.reset_threshold() == 150.0 and ctx.notes[-1] == "MT inaccuracy threshold reset to 150"
        # the 2 s poll reloads when the sidecars changed on disk
        fake.changed = True
        screen.dispose()
        # no core in this build: the screen says so instead of failing
        monkeypatch.setitem(sys.modules, "sdlxliff_review_core", _module("sdlxliff_review_core"))
        fid2 = sx.open_reviewer(ctx, str(out))
        empty = sx.SdlxliffScreen(parse_route(f"/tools/sdlxliff?out={fid2}"), ctx)
        _tools()._mount(page, empty.get_body())
        assert await empty.open() is None and empty.error == sx.NO_CORE

    asyncio.run(scenario())


@needs_flet
def test_sdlxliff_row_save_keeps_the_other_fields_and_their_typed_text(tmp_path, monkeypatch):
    """A blur save replaces only its own row card: the next row's field (focus, typed text) is
    untouched; the poll does not reload over an edit; a navigation saves the typed text first."""
    monkeypatch.setitem(sys.modules, "sdlxliff_review_core", _fake_sdl_core())
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools import sdlxliff as sx

    out = tmp_path / "Output" / "Book"
    out.mkdir(parents=True)

    async def scenario():
        _conn, session = _tb()._fake_session("android")
        page = session.page
        ctx = _tools_ctx(page)
        fid = sx.open_reviewer(ctx, str(out), focus="response_ch001.html")
        screen = sx.SdlxliffScreen(parse_route(f"/tools/sdlxliff?out={fid}"), ctx)
        _tools()._mount(page, screen.get_body())
        binding = await screen.open()
        fake = binding.session
        assert screen.piece_index == 0 and set(screen.row_fields) == {0, 1}
        first, second = screen.row_fields[0], screen.row_fields[1]
        second_card = screen.row_cards[1]
        # row 0 edited and left; the user is already typing in row 1
        _client_set(first, "value", "Text, edited")
        screen._set_focused(1, True)
        _client_set(second, "value", "typing…")
        assert screen._editing()  # the 2 s poll leaves the rows alone meanwhile
        assert await screen.save_row(0, first, 0) is True and ("edit", 0, 0, "Text, edited") in fake.calls
        assert screen.row_fields[1] is second and screen.row_cards[1] is second_card and second.value == "typing…"
        assert screen.row_fields[0] is not first and screen.row_fields[0].value == "Text, edited"
        assert any(c is second_card for c in screen.rows_column.controls)
        assert "edit" not in [c[0] for c in fake.calls if c[2:3] == (1,)]
        # a navigation while row 1 holds unsaved text: saved into its own piece, then re-rendered
        screen._set_focused(1, False)
        screen.step(1)
        for _ in range(200):
            if screen.rows_piece_index == 1:
                break
            await asyncio.sleep(0.01)
        assert ("edit", 0, 1, "typing…") in fake.calls and screen.piece_index == 1
        assert list(screen.row_fields) == [0] and not screen._editing()
        # a late blur save of a field from the piece left behind still targets that piece
        stale = first
        _client_set(stale, "value", "late text")
        assert await screen.save_row(0, stale, 0) is True and ("edit", 0, 0, "late text") in fake.calls
        assert screen.row_fields[0].value == "Three"  # piece 1's row is not touched
        # a polled refresh that lands while a field has focus waits for the edit to end
        field = screen.row_fields[0]
        screen._set_focused(0, True)
        await screen.reload(polled=True)
        assert screen._render_pending and screen.row_fields[0] is field
        screen._set_focused(0, False)
        await screen.reload()
        assert not screen._render_pending and screen.row_fields[0] is not field
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_sdlxliff_notepad_layout_on_tablets(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "sdlxliff_review_core", _fake_sdl_core())
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools import sdlxliff as sx

    out = tmp_path / "Output" / "Book"
    out.mkdir(parents=True)

    async def scenario():
        _conn, session = _tb()._fake_session("android")
        page = session.page
        ctx = _tools_ctx(page, tablet=True)
        fid = sx.open_reviewer(ctx, str(out), focus="response_ch002.html")
        screen = sx.SdlxliffScreen(parse_route(f"/tools/sdlxliff?out={fid}"), ctx)
        _tools()._mount(page, screen.get_body())
        binding = await screen.open()
        assert screen.layout_switch.disabled is False
        screen.layout_switch.value = True
        screen._on_layout()
        assert screen.notepad and screen.notepad_editor is None  # the document is built on the io pool
        for _ in range(200):
            if screen.notepad_editor is not None:
                break
            await asyncio.sleep(0.01)
        assert screen.notepad_editor.value == "<p>Three</p>" and screen.notepad_base == "<p>Three</p>"
        assert not screen._editing()
        _client_set(screen.notepad_editor, "value", "<p>Drei</p>")
        assert screen._editing()  # an edited document: the 2 s poll does not reload over it
        assert await screen.save_document() is True
        assert ("document", 1, "<p>Drei</p>") in binding.session.calls
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_sdlxliff_edits_are_saved_on_leave_and_the_poll_runs_only_when_shown(tmp_path, monkeypatch):
    """U7 review: like the dialog's closeEvent (Notepad HTML captured, queued edits flushed), leaving
    the reviewer saves an edited Notepad document and typed row text, and a piece switch saves the
    document into its own piece first. The 2 s changed_on_disk poll runs only while the reviewer is
    on top (not under its own Edit Output editor) and the app is in the foreground."""
    monkeypatch.setitem(sys.modules, "sdlxliff_review_core", _fake_sdl_core())
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools import sdlxliff as sx

    monkeypatch.setattr(sx, "POLL_SECONDS", 0.02)
    out = tmp_path / "Output" / "Book"
    out.mkdir(parents=True)

    async def wait_for(predicate):
        for _ in range(300):
            if predicate():
                return True
            await asyncio.sleep(0.01)
        return predicate()

    async def scenario():
        _conn, session = _tb()._fake_session("android")
        page = session.page
        # ---- Notepad layout (tablets): a piece switch, then leaving the reviewer -------------------
        ctx = _tools_ctx(page, tablet=True)
        fid = sx.open_reviewer(ctx, str(out), focus="response_ch002.html")
        screen = sx.SdlxliffScreen(parse_route(f"/tools/sdlxliff?out={fid}"), ctx)
        _tools()._mount(page, screen.get_body())
        binding = await screen.open()
        calls = binding.session.calls
        screen.layout_switch.value = True
        screen._on_layout()
        assert await wait_for(lambda: screen.notepad_editor is not None)
        _client_set(screen.notepad_editor, "value", "<p>Drei</p>")
        screen.step(-1)  # piece 1 -> piece 0: the edited document is saved into piece 1 first
        assert await wait_for(lambda: screen.notepad_editor is not None and screen.notepad_piece == 0)
        assert ("document", 1, "<p>Drei</p>") in calls and screen.notepad_editor.value == "<p>Text</p>"
        _client_set(screen.notepad_editor, "value", "<p>Eins</p>")
        screen.dispose()  # Back / a drawer destination
        assert await wait_for(lambda: ("document", 0, "<p>Eins</p>") in calls)
        # ---- compact rows: text typed in a field that never lost focus ---------------------------
        _conn, session = _tb()._fake_session("android")  # a fresh page for each reviewer
        page = session.page
        fid = sx.open_reviewer(ctx, str(out), focus="response_ch001.html")
        rows = sx.SdlxliffScreen(parse_route(f"/tools/sdlxliff?out={fid}"), _tools_ctx(page))
        _tools()._mount(page, rows.get_body())
        row_binding = await rows.open()
        rows._set_focused(1, True)
        _client_set(rows.row_fields[1], "value", "still typing")
        rows.dispose()
        assert await wait_for(lambda: ("edit", 0, 1, "still typing") in row_binding.session.calls)
        # ---- the poll: only on top and in the foreground -----------------------------------------
        _conn, session = _tb()._fake_session("android")
        page = session.page
        shown = {"top": False, "foreground": True}
        poll_ctx = _tools_ctx(page)
        poll_ctx.is_top = lambda s: shown["top"]
        poll_ctx.foreground = lambda: shown["foreground"]
        fid = sx.open_reviewer(poll_ctx, str(out))
        polled = sx.SdlxliffScreen(parse_route(f"/tools/sdlxliff?out={fid}"), poll_ctx)
        _tools()._mount(page, polled.get_body())
        polled.did_show()
        assert await wait_for(lambda: polled.binding is not None and polled._poll_task is not None)
        checks = []
        session_ = polled.binding.session
        original = session_.changed_on_disk
        session_.changed_on_disk = lambda: (checks.append(1), original())[1]
        await asyncio.sleep(0.2)  # covered (e.g. its own Edit Output editor on top)
        assert checks == []
        shown.update(top=True, foreground=False)  # the app in the background
        await asyncio.sleep(0.2)
        assert checks == []
        shown["foreground"] = True
        assert await wait_for(lambda: len(checks) >= 2)
        polled.dispose()

    asyncio.run(scenario())


def test_real_sdlxliff_session_through_the_binding(tmp_path, monkeypatch):
    core = _real_core("sdlxliff_review_core", "open_sdlxliff_review", "SdlxliffReviewSession")
    writer = _real_core("sdlxliff_sidecar_writer", "_write_html_sdlxliff_sidecar")
    if core is None or writer is None:
        pytest.skip("sdlxliff_review_core.open_sdlxliff_review not in this tree")
    from glossarion_mobile.ui.tools import sdlxliff as sx

    monkeypatch.setenv("OUTPUT_SDLXLIFF", "1")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    out = tmp_path / "Output" / "Book"
    out.mkdir(parents=True)
    source_html = "<html><body><h1>제목</h1><p>첫 문장입니다.</p><p>두 번째 문장입니다.</p></body></html>"
    target_html = "<html><body><h1>Title</h1><p>The first sentence.</p><p></p></body></html>"
    (out / "response_ch001.html").write_text(target_html, encoding="utf-8")
    (out / "translation_progress.json").write_text(json.dumps({"chapters": {"1": {
        "actual_num": 1, "status": "completed", "output_file": "response_ch001.html"}}}), encoding="utf-8")
    path = writer._write_html_sdlxliff_sidecar(str(out), "response_ch001.html", {}, source_html, target_html,
                                               raise_errors=True)
    assert path and os.path.isfile(path)
    saved = {}
    binding = sx.ReviewerBinding(core, str(out), focus="response_ch001.html", config={},
                                 save=lambda key, value: saved.__setitem__(key, value))
    pieces = binding.load()
    assert len(pieces) == 1 and binding.focus_index() == 0
    rows = pieces[0]["rows"]
    statuses = [row.get("status") for row in rows]
    assert "red" in statuses  # the emptied paragraph
    empty = statuses.index("red")
    binding.save_row(0, empty, "The second sentence.")
    reloaded = binding.load(force=True)[0]["rows"]
    assert reloaded[empty]["target"] == "The second sentence."
    assert "The second sentence." in (out / "response_ch001.html").read_text(encoding="utf-8")
    threshold = binding.set_threshold(42)
    assert threshold == 42.0 and saved.get("sdlxliff_machine_translation_inaccuracy_threshold") == 42.0
    assert binding.reset_threshold() == binding.default_threshold()


# ==========================================================================
# The shared cores this side calls (the names must stay in step; a module that does not
# import in this environment skips its check)
# ==========================================================================


def _importable(name):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


@pytest.mark.parametrize("module, names", [
    ("progress_actions", ("plan_retranslation", "apply_retranslation", "retranslation_result_message",
                          "prepare_single_qa_resolution", "build_partial_b_request", "find_row_audio", "reset_tts",
                          "reset_tts_message")),
    ("progress_core", ("build_image_folder_progress", "mark_image_folder_items_skipped",
                       "image_folder_mark_skipped_message", "image_folder_delete_confirmation",
                       "delete_image_folder_items", "_progress_entry_has_llm_token_qa")),
    ("async_batch_core", ("HeadlessAsyncBatch", "AsyncAPIProcessor", "job_display_row", "async_support_status",
                          "default_jobs_file")),
    ("rpgmaker_job", ("prepare_rpgmaker_game", "RpgMakerJobMixin")),
    ("review_generator", ("run_review_session", "review_run_params", "review_all_batch_size",
                          "reset_review_stop_flags", "apply_review_streaming_env", "run_all_reviews",
                          "review_paths_for", "count_review_tokens", "_save_review_text", "DEFAULT_REVIEW_PROMPT",
                          "DEFAULT_FINAL_REVIEW_PROMPT")),
    ("sdlxliff_review_core", ("open_sdlxliff_review", "SdlxliffReviewSession")),
    ("authgem_auth", ("authgem_project_items", "choose_authgem_project_index", "list_gcp_projects")),
])
def test_the_shared_cores_offer_what_the_u7_screens_call(module, names):
    mod = _importable(module)
    if mod is None:
        pytest.skip(f"{module} does not import here")
    missing = [name for name in names if not hasattr(mod, name)]
    assert not missing, f"{module} lacks {missing}"
    if module == "rpgmaker_job":
        assert hasattr(mod.RpgMakerJobMixin, "_register_rpgmaker_game_input")
    if module == "sdlxliff_review_core":
        session = mod.SdlxliffReviewSession
        for name in ("refresh", "changed_on_disk", "piece_summary", "select_piece", "switch_book", "save_row",
                     "notepad_document", "edit_document", "flush_edits", "output_path", "mark_completed",
                     "undo_completed", "machine_translation_preview", "inject_machine_translation",
                     "flag_inaccurate", "set_inaccuracy_threshold", "reset_inaccuracy_threshold", "set_provider",
                     "set_machine_translation_credentials"):
            assert hasattr(session, name), name
    if module == "async_batch_core":
        batch = mod.HeadlessAsyncBatch
        for name in ("submit", "estimate", "refresh_statuses", "check_status", "retrieve", "cancel", "delete",
                     "clear_completed"):
            assert hasattr(batch, name), name


def test_real_async_snapshot_and_rpgmaker_scan(tmp_path, monkeypatch):
    core = _importable("async_batch_core")
    from glossarion_mobile.ui.tools import async_batch as ab
    from glossarion_mobile.ui.tools import rpgmaker as rm

    if core is not None and hasattr(core, "AsyncAPIProcessor"):
        jobs_file = tmp_path / "data" / "async_jobs.json"
        snap = ab.load_async_snapshot("gpt-5", str(jobs_file))
        assert snap.error is None and snap.rows == () and snap.supported is True
        assert snap.status_text == "✓ Supported (OPENAI)"
        assert ab.load_async_snapshot("nonsense-model", str(jobs_file)).supported is False
    handler = _importable("rpgmaker_handler")
    entry = _importable("rpgmaker_job")
    if handler is None or entry is None or not hasattr(entry, "prepare_rpgmaker_game"):
        pytest.skip("rpgmaker_job / rpgmaker_handler not importable")
    game = tmp_path / "Inbox" / "MyGame"
    (game / "www" / "data").mkdir(parents=True)
    (game / "www" / "data" / "System.json").write_text(json.dumps({"gameTitle": "テスト"}), encoding="utf-8")
    archive = tmp_path / "MyGame.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(game / "www" / "data" / "System.json", "MyGame/www/data/System.json")
    scan = rm.scan_game(str(game), str(tmp_path / "work"))
    assert scan.error is None and scan.version == "mv" and scan.game_dir == str(game)
    assert scan.summary.startswith("RPG Maker MV")
    zipped = rm.scan_game(str(archive), str(tmp_path / "work"))
    assert zipped.error is None and zipped.game_dir.startswith(str(tmp_path / "work"))
    assert os.path.isfile(os.path.join(zipped.game_dir, "www", "data", "System.json"))
    assert rm.scan_game(str(tmp_path / "Inbox"), "").version in ("mv", "")  # the shallowest game root


def test_real_resolve_qa_preflight_and_job_on_a_fixture_workspace(tmp_path):
    pa = _real_core("progress_actions", "prepare_single_qa_resolution", "build_partial_b_request")
    if pa is None:
        pytest.skip("progress_actions.prepare_single_qa_resolution not in this tree")
    from glossarion_mobile.ui.library import progress_model as pm

    ws = tmp_path / "Output" / "Book"
    ws.mkdir(parents=True)
    source = tmp_path / "Raw" / "Book.epub"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"PK")
    (ws / "response_ch002.html").write_text("<p>원문 text</p>", encoding="utf-8")
    entry = {"actual_num": 2, "status": "qa_failed", "output_file": "response_ch002.html",
             "qa_issues_found": ["korean_text_found_12_chars_원문"]}
    progress = ws / "translation_progress.json"
    progress.write_text(json.dumps({"chapters": {"2": entry}}), encoding="utf-8")
    data = {"prog": {"chapters": {"2": dict(entry)}}, "progress_file": str(progress), "output_dir": str(ws),
            "file_path": str(source)}
    info = {"progress_key": "2", "output_file": "response_ch002.html", "status": "qa_failed", "info": dict(entry)}
    service = FakeService({})
    view = _view(output_dir=str(ws), file_path=str(source), data=data)
    row = _row(**info)
    plan = pm.plan_action(service, view, "resolve_qa", [row])
    assert plan.extra["partial_b"]["progress_key"] == "2"
    spec = pm.resolve_qa_spec(service, {"name": "Book"}, plan)
    owner = TranslateOwner(str(tmp_path / "Output"))
    ctx = FakeCtx(owner, params=spec.params, inputs=spec.inputs)
    resolve_kind.run(ctx)
    # the shared preflight set the desktop run state; the pipeline saw the request
    assert owner.seen["request"]["progress_key"] == "2" and owner.seen["files"] == [str(source)]
    assert owner.seen["selected"] == [str(source)] and owner._single_qa_resolution_request is None
    assert owner._metadata_only_run is False and owner._single_chapter_filter is None
    assert "⚠️ Queued Partial.b QA resolution for response_ch002.html only" in ctx.logs
    # the issue is gone on disk: the preflight refuses with the desktop text, no run
    progress.write_text(json.dumps({"chapters": {"2": dict(entry, qa_issues_found=[], status="completed")}}),
                        encoding="utf-8")
    owner2 = TranslateOwner(str(tmp_path / "Output"))
    ctx2 = FakeCtx(owner2, params=spec.params, inputs=spec.inputs)
    assert resolve_kind.run(ctx2) == {"ok": True, "outputs": []} and owner2.seen == {}
    assert ctx2.results["resolve_qa_refusal"] == {"kind": "info", "title": "QA Issue Already Resolved",
                                                  "message": "This entry no longer has a raw foreign-text QA issue."}



def test_u7_integration_wiring():
    """Integrate: the U7 kinds are registered (KIND_MODULES + JobKind), U7 is a shipped milestone, the
    four Tools routes build their screens and the text editor route belongs to the Jobs feature."""
    from glossarion_mobile.services.jobs import JobKind
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.base import SHIPPED_MILESTONES
    from glossarion_mobile.ui.screens.jobs import SCREEN_ROUTES as JOB_ROUTES
    from glossarion_mobile.ui.tools.feature import IMPLEMENTED_ROUTES, ToolsFeature
    from glossarion_mobile.ui.tools.hub import HUB_GROUPS, tile_available

    assert "U7" in SHIPPED_MILESTONES
    kinds = dict(U7_KINDS, generate_media="generate_media", translate_image="image")
    for kind, module in kinds.items():
        assert job_kinds.KIND_MODULES[kind] == module and JobKind(kind).value == kind
        assert callable(job_kinds.get_kind(kind).run), kind
    assert job_kinds.get_kind("retranslate").resumable is False
    app = types.SimpleNamespace(page=None, dispatcher=None, shell=None, state=None, paths=None, settings=None,
                                library=None, files=None, jobs=None, prefs=None, haptics=None, intents=None,
                                config_store=None, library_feature=None)
    feature = ToolsFeature(app)
    built = {route: type(feature.make_screen(parse_route(route))).__name__
             for route in ("/tools/async", "/tools/review", "/tools/sdlxliff", "/tools/rpgmaker")}
    assert built == {"/tools/async": "AsyncBatchScreen", "/tools/review": "ReviewScreen",
                     "/tools/sdlxliff": "SdlxliffScreen", "/tools/rpgmaker": "RpgMakerScreen"}
    for _group, tiles in HUB_GROUPS:
        for tile in tiles:
            if tile.route in ("tools.async", "tools.review", "tools.sdlxliff", "tools.rpgmaker"):
                assert tile.route in IMPLEMENTED_ROUTES and tile_available(tile, IMPLEMENTED_ROUTES) is None
    assert "tools.text" in JOB_ROUTES


def test_review_screen_delete_and_restore_use_the_shared_helpers(tmp_path):
    """🗑️ Delete / ↩️ Restore run review_generator's helpers (moved out of review_dialog at the U7
    integration): backups next to the review, the desktop overwrite question with the backup name."""
    import review_generator as real

    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools import review as rv

    review_md = tmp_path / "Book" / "review" / "review.md"
    review_md.parent.mkdir(parents=True)
    review_md.write_text("# Review\n", encoding="utf-8")

    async def _no_reload():
        return None

    async def scenario():
        _conn, session = _tb()._fake_session("android")
        page = session.page
        ctx = _tools_ctx(page, store={}, jobs=U7Jobs())
        screen = rv.ReviewScreen(parse_route("/tools/review"), ctx)
        _tools()._mount(page, screen.get_body())
        screen.review_paths = lambda: [str(review_md)]
        screen.load_current = _no_reload
        assert rv._helper(rv.DELETE_HELPERS) is real.move_review_to_backups
        assert rv._helper(rv.RESTORE_HELPERS) is real.restore_review_backup
        assert await screen.restore() is False  # no backup yet
        assert await screen.delete() is True and not review_md.exists()
        backups = sorted((review_md.parent / "backups").glob("review_*.md"))
        assert len(backups) == 1 and backups[0].read_text(encoding="utf-8") == "# Review\n"
        review_md.write_text("# Newer\n", encoding="utf-8")
        ctx.extras["answers"] = ["no"]
        assert await screen.restore() is False and review_md.read_text(encoding="utf-8") == "# Newer\n"
        assert ctx.extras["asked"][-1] == ("Restore Backup", real.review_restore_question(backups[0].name))
        ctx.extras["answers"] = ["yes"]
        assert await screen.restore() is True and review_md.read_text(encoding="utf-8") == "# Review\n"
        screen.dispose()

    asyncio.run(scenario())
